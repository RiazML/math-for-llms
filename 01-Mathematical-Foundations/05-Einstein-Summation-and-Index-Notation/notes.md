[← Back to Mathematical Foundations](../README.md) | [Previous: Summation and Product Notation →](../04-Summation-and-Product-Notation/notes.md) | [Next: Proof Techniques →](../06-Proof-Techniques/notes.md)

# Einstein Summation and Index Notation

> _"I have made a great discovery in mathematics; I have suppressed the summation sign every time that the summation must be made over an index which occurs twice."_ — **Albert Einstein**, 1916

## Overview

Einstein summation convention is the notational engine of modern tensor computation. Wherever the same index appears exactly twice in a single term, summation over all values of that index is implied — no $\Sigma$ needed. This convention, born in general relativity, now powers every matrix multiply, every attention score, every gradient computation in deep learning. Once you can read index notation, you can read any transformer paper, derive any backpropagation formula, and debug any shape error — mechanically.

## Prerequisites

- Summation and product notation ($\Sigma$, $\Pi$) from Module 04
- Matrix operations: multiplication, transpose, trace
- Basic calculus: partial derivatives, chain rule
- Familiarity with NumPy arrays / PyTorch tensors

## Learning Objectives

After completing this section, you will be able to:

1. Read and write tensor expressions in Einstein index notation
2. Identify free indices (result dimensions) and dummy indices (contracted/summed) in any expression
3. Derive gradient formulas mechanically using index notation and the Kronecker delta
4. Translate any tensor operation into a `np.einsum` / `torch.einsum` string
5. Express transformer attention, backpropagation, and tensor decompositions in index form
6. Analyse symmetry, equivariance, and contraction order using index structure

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive code: einsum operations, gradient verification, contraction order benchmarks, attention in index notation |
| [exercises.ipynb](exercises.ipynb) | Practice problems: index identification, Kronecker delta manipulation, gradient derivation, einsum implementation |

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is Einstein Summation?](#11-what-is-einstein-summation)
  - [1.2 Why This Notation Transforms Thinking](#12-why-this-notation-transforms-thinking)
  - [1.3 The Core Idea in One Sentence](#13-the-core-idea-in-one-sentence)
  - [1.4 Where This Appears in AI](#14-where-this-appears-in-ai)
  - [1.5 Historical and Modern Context](#15-historical-and-modern-context)
  - [1.6 The Hierarchy of Tensor Operations](#16-the-hierarchy-of-tensor-operations)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Tensors as Indexed Arrays](#21-tensors-as-indexed-arrays)
  - [2.2 The Einstein Summation Convention — Precise Statement](#22-the-einstein-summation-convention--precise-statement)
  - [2.3 Free vs Dummy Indices — Precise Definitions](#23-free-vs-dummy-indices--precise-definitions)
  - [2.4 Index Ranges](#24-index-ranges)
  - [2.5 Covariant and Contravariant Indices](#25-covariant-and-contravariant-indices)
- [3. Basic Operations in Index Notation](#3-basic-operations-in-index-notation)
- [4. Contraction Operations](#4-contraction-operations)
- [5. The Kronecker Delta and Levi-Civita Symbol](#5-the-kronecker-delta-and-levi-civita-symbol)
- [6. Index Manipulation Rules](#6-index-manipulation-rules)
- [7. Einstein Notation for Gradients and Backpropagation](#7-einstein-notation-for-gradients-and-backpropagation)
- [8. Batched Operations and Batch Indices](#8-batched-operations-and-batch-indices)
- [9. Einsum in Practice](#9-einsum-in-practice)
- [10. Tensor Products and Decompositions](#10-tensor-products-and-decompositions)
- [11. Index Notation for Common AI Architectures](#11-index-notation-for-common-ai-architectures)
- [12. Symmetries, Invariances, and Equivariances](#12-symmetries-invariances-and-equivariances)
- [13. Advanced Index Manipulations](#13-advanced-index-manipulations)
- [14. Common Mistakes](#14-common-mistakes)
- [15. Exercises](#15-exercises)
- [16. Why This Matters for AI](#16-why-this-matters-for-ai)
- [17. Conceptual Bridge](#17-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is Einstein Summation?

Einstein summation convention is a notational shorthand: whenever the same index appears **exactly twice** in a single term, summation over all values of that index is implied automatically. No $\Sigma$ symbol needed — the repetition of an index *is* its own summation instruction.

Consider the matrix-vector product. In standard notation:

$$w_i = \sum_{j=1}^{n} A_{ij} v_j$$

In Einstein notation, the same expression is simply:

$$w_i = A_{ij} v_j$$

The index $j$ appears twice (once on $A$, once on $v$), so summation over $j$ is implied. The index $i$ appears once — it is a **free index**, meaning the equation holds for each value of $i$.

The result: tensor equations that would fill pages compress to single lines. The structure of the operation is immediately visible in the index pattern.

**For AI:** every matmul, every dot product, every attention score, every gradient computation is a sum over repeated indices. Einstein notation makes this structure explicit and manipulable.

### 1.2 Why This Notation Transforms Thinking

Without index notation, $AB$ means "matrix multiply $A$ and $B$" — the mechanics are hidden. With index notation:

$$(AB)_{ij} = A_{ik}B_{kj}$$

The repeated $k$ tells you exactly **what is summed**, **over what range**, and **how inputs connect to outputs**.

**Pattern recognition from indices alone:**

| Expression | Repeated Index | Free Indices | Operation |
|---|---|---|---|
| $A_{ik}B_{kj}$ | $k$ | $i, j$ | Matrix multiply |
| $A_{ij}B_{ij}$ | $i, j$ | none | Frobenius inner product (scalar) |
| $A_{ik}B_{jk}$ | $k$ | $i, j$ | $AB^\top$ |
| $u_i v_j$ | none | $i, j$ | Outer product |
| $u_i v_i$ | $i$ | none | Dot product (scalar) |

**Debugging power:** If an equation has a free index on one side but not the other, it is **wrong**. Index balance checking catches errors mechanically — before running any code.

**Generalisation:** Every operation on any-order tensor is expressed uniformly. There is no special notation for matrices vs vectors vs 4D tensors — just indices.

### 1.3 The Core Idea in One Sentence

Two rules govern everything:

- **Free index**: appears once → result has that index → no summation
- **Dummy index**: appears twice → summed over → disappears from result

That's it. Everything else follows from these two rules.

```
THE TWO RULES
═══════════════════════════════════════════════════════════════════

Expression: C_ij = A_ik B_kj

  Index i: appears ONCE → FREE    → C has index i (rows)
  Index j: appears ONCE → FREE    → C has index j (cols)
  Index k: appears TWICE → DUMMY  → summed over → gone from result

  Result shape: C is a matrix (has i,j) ✓
  Computation: C_ij = Σ_k A_ik B_kj ✓

Expression: s = u_i v_i

  Index i: appears TWICE → DUMMY  → summed over → gone from result

  Result shape: s is a scalar (no free indices) ✓
  Computation: s = Σ_i u_i v_i = dot product ✓
```

### 1.4 Where This Appears in AI

Every core operation in deep learning is an Einstein summation:

| Operation | Index Notation | Dummy | Free | Description |
|---|---|---|---|---|
| Matrix multiply | $C_{ij} = A_{ik}B_{kj}$ | $k$ | $i,j$ | Forward pass linear layer |
| Dot product | $s = a_i b_i$ | $i$ | — | Scalar score |
| Attention score | $S_{ij} = Q_{ik}K_{jk}/\sqrt{d_k}$ | $k$ | $i,j$ | Query-key similarity |
| Attention output | $O_{ia} = \alpha_{ij}V_{ja}$ | $j$ | $i,a$ | Weighted value sum |
| Gradient of matmul | $\frac{\partial L}{\partial A_{ik}} = \frac{\partial L}{\partial C_{ij}} B_{kj}$ | $j$ | $i,k$ | Backprop through $C = AB$ |
| Layer norm gradient | $\frac{\partial L}{\partial x_i}$ involves $\sum_j$ | $j$ | $i$ | Non-local gradient |
| Multi-head attention | $O_{ia} = \sum_h W^h_{a,\text{out}} \alpha^h_{ij} V^h_{j,d}$ | $h,j$ | $i,a$ | Across heads |

### 1.5 Historical and Modern Context

| Year | Development | Significance |
|---|---|---|
| 1900 | Ricci & Levi-Civita: absolute differential calculus | Index notation precursor |
| 1916 | Einstein: general relativity paper | Introduced implicit summation convention |
| 1920s–50s | Standard in physics | GR, electromagnetism, continuum mechanics |
| 1960s–90s | Penrose: abstract index notation | Diagrammatic tensor calculus |
| 2006 | NumPy `numpy.einsum` | Einstein convention for array computing |
| 2016 | PyTorch `torch.einsum` | GPU-accelerated einsum |
| 2017 | Vaswani et al.: "Attention Is All You Need" | Attention expressed as index contractions |
| 2018 | `opt_einsum` library | Optimal contraction order for multi-tensor einsum |
| 2022 | Dao et al.: FlashAttention | Attention kernels designed from index structure |
| 2024 | DeepSeek: MLA (Multi-head Latent Attention) | Low-rank contraction $K_{ja} = c_{jr} W^K_{ra}$ |
| 2025–26 | JAX einsum + JIT; einsum as ML lingua franca | Seamless compilation of arbitrary contractions |

### 1.6 The Hierarchy of Tensor Operations

```
TENSOR RANK HIERARCHY
═══════════════════════════════════════════════════════════════════

  Scalar (rank 0):   s            — no indices
  Vector (rank 1):   v_i          — one index
  Matrix (rank 2):   M_ij         — two indices
  3-Tensor (rank 3): T_ijk        — three indices
  n-Tensor (rank n): T_{i₁i₂…iₙ} — n indices

RANK-CHANGING OPERATIONS:
  ┌─────────────────────────────────────────────────────────────┐
  │  Contract one pair of indices  → reduce rank by 2           │
  │  Outer product                 → rank = sum of input ranks  │
  │  Transpose                    → relabel indices (same rank) │
  │  Trace                        → contract diagonal (rank −2) │
  └─────────────────────────────────────────────────────────────┘

EXAMPLES:
  dot product:     rank 1 + rank 1 − 2 = rank 0 (scalar)
  matrix-vector:   rank 2 + rank 1 − 2 = rank 1 (vector)
  matrix-matrix:   rank 2 + rank 2 − 2 = rank 2 (matrix)
  trace:           rank 2 − 2           = rank 0 (scalar)
  outer product:   rank 1 + rank 1      = rank 2 (matrix)
```

---

## 2. Formal Definitions

### 2.1 Tensors as Indexed Arrays

A tensor $T$ of rank $r$ over index sets $I_1, I_2, \ldots, I_r$ is a function:

$$T: I_1 \times I_2 \times \cdots \times I_r \to \mathbb{R}$$

It assigns a real number $T_{i_1 i_2 \ldots i_r}$ to each tuple of indices.

| Rank | Name | Notation | Example | Shape |
|---|---|---|---|---|
| 0 | Scalar | $s$ | Loss value, learning rate | () |
| 1 | Vector | $v_i$ | Embedding, gradient | $(n,)$ |
| 2 | Matrix | $M_{ij}$ | Weight matrix, attention scores | $(m, n)$ |
| 3 | 3-tensor | $T_{ijk}$ | Batched matrices, $Q/K/V$ projections | $(B, n, d)$ |
| 4 | 4-tensor | $T_{bhij}$ | Multi-head attention scores | $(B, H, n, n)$ |

The **shape** is the tuple of index set sizes $(n_1, n_2, \ldots, n_r)$. It determines memory layout and operation validity.

### 2.2 The Einstein Summation Convention — Precise Statement

**Convention:** In any expression, if an index label $\alpha$ appears **exactly twice** in a single term, it is a **dummy index** and is implicitly summed over all values in its range.

Formally, for dummy index $\alpha$ over range $\{1, \ldots, n\}$:

$$A_{\ldots\alpha\ldots} B_{\ldots\alpha\ldots} \equiv \sum_{\alpha=1}^{n} A_{\ldots\alpha\ldots} B_{\ldots\alpha\ldots}$$

An index appearing **once** is a **free index**: the equation holds for each value of that index independently. The range of each index is determined by the tensor dimensions at that position.

**Example walkthrough:**

$$C_{ij} = A_{ik} B_{kj}$$

- $k$ appears twice → dummy → implicit sum: $C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}$
- $i$ appears once → free → equation holds for all $i \in \{1,\ldots,m\}$
- $j$ appears once → free → equation holds for all $j \in \{1,\ldots,n\}$
- Result: $C \in \mathbb{R}^{m \times n}$, same as standard matrix multiply $C = AB$

### 2.3 Free vs Dummy Indices — Precise Definitions

**Free index** — appears exactly once in a term:
- Result expression carries that index
- The equation holds for every value of the free index independently
- Example: $A_{ij} = B_{ik}C_{kj}$ — indices $i$ and $j$ are free; equation holds for all $(i,j)$

**Dummy index (bound/contracted index)** — appears exactly twice in a single term:
- Implicitly summed over its entire range
- Does not appear in the result
- Example: $B_{ik}C_{kj}$ — index $k$ is dummy; summed from 1 to $p$; gone from result

**Invalid** — index appearing three or more times:
- $A_{iii}$ is ambiguous: which two occurrences to contract?
- Einstein convention forbids this; write explicit $\Sigma$ if needed
- Example: $A_{iij}B_i$ has index $i$ appearing three times — not valid

**Consistency rule** — free indices must be the same on both sides of an equation:

$$v_i = A_{ij} u_j \quad \checkmark \quad \text{(both sides have free index } i \text{)}$$

$$v_i = A_{jk} u_k \quad \times \quad \text{(left has free } i \text{; right has free } j \text{; inconsistent)}$$

This mechanical check catches dimension errors before code runs.

### 2.4 Index Ranges

Each index position has a defined range from the tensor's shape:

- In $A_{ij}$ where $A \in \mathbb{R}^{m \times n}$: $i \in \{1,\ldots,m\}$, $j \in \{1,\ldots,n\}$
- Contracted indices **must have matching ranges**: $A_{ik}B_{kj}$ valid only if $A$'s second dimension equals $B$'s first dimension
- This is exactly the matrix multiply shape compatibility condition: $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times n}$; $k$ ranges over $\{1,\ldots,p\}$

**Shape inference** — from index structure alone, the result shape is determined:
- $A_{ik}B_{kj}$: $i$ ranges over $m$, $j$ ranges over $n$ → result shape $(m, n)$
- $v_i w_i$: no free indices → result shape () — scalar
- $T_{ijk}U_{jl}V_{kl}$: $j$ and $k$ and $l$ are all dummy; $i$ is free → result shape $(n_i,)$ — vector

### 2.5 Covariant and Contravariant Indices

In differential geometry and general relativity, upper (contravariant) and lower (covariant) indices are distinguished:

- **Upper index** $v^i$: contravariant vector; transforms like coordinate differences
- **Lower index** $w_i$: covariant vector; transforms like gradient components
- Einstein convention in physics: sum over **one upper and one lower** index only: $A_{ij}B^j{}_k$
- **Metric tensor** $g_{ij}$: raises and lowers indices; $v_i = g_{ij}v^j$

**For machine learning:** we almost always ignore the up/down distinction. All indices are effectively lower. No metric tensor is needed because we work in flat Euclidean space where $g_{ij} = \delta_{ij}$. When reading physics literature, understand the convention; in ML, treat all indices as lower.

```
COVARIANT vs CONTRAVARIANT — PHYSICS vs ML
═══════════════════════════════════════════════════════════════════

  PHYSICS (General Relativity):
    Upper index:  v^i         (contravariant — transforms one way)
    Lower index:  w_i         (covariant — transforms the other)
    Contraction:  v^i w_i     (one up, one down → valid)
    Metric:       v_i = g_ij v^j   (raises/lowers indices)

  MACHINE LEARNING:
    All indices:  v_i, w_j    (no up/down distinction)
    Contraction:  v_i w_i     (two of same name → valid)
    Metric:       g_ij = δ_ij (identity — flat space)

  → In ML, upper/lower distinction collapses. All that matters:
    - Same index twice = sum over it
    - Same index once  = free dimension
```

---

## 3. Basic Operations in Index Notation

### 3.1 Scalar Multiplication

Scalar times vector: $(sv)_i = s \cdot v_i$

Scalar times matrix: $(sA)_{ij} = s \cdot A_{ij}$

The scalar $s$ has no indices; it multiplies each component. No summation occurs — there is no repeated index.

### 3.2 Vector Addition

$$(u + v)_i = u_i + v_i$$

Same free index $i$ on both terms. No repeated index, no summation. Purely elementwise.

### 3.3 Inner Product (Dot Product)

$$s = u_i v_i = \sum_i u_i v_i$$

Both $u$ and $v$ carry index $i$. Since $i$ appears twice, it is a dummy index — summed over. The result $s$ is a scalar (no free indices).

Equivalently: $s = u^\top v = \langle u, v \rangle$

**Sign check:** result has no free indices → result must be scalar ✓

### 3.4 Outer Product

$$C_{ij} = u_i v_j$$

No repeated index — $i$ appears once, $j$ appears once. Both are free. No summation. The result $C \in \mathbb{R}^{m \times n}$ where $m = \dim(u)$, $n = \dim(v)$.

Equivalently: $C = uv^\top$

Each entry $C_{ij} = u_i \times v_j$. This always produces a **rank-1 matrix**.

**Key distinction from dot product:** $u_i v_i$ (same index → sum → scalar) vs $u_i v_j$ (different indices → no sum → matrix).

### 3.5 Matrix-Vector Multiply

$$w_i = A_{ij} v_j = \sum_j A_{ij} v_j$$

- $j$ is dummy (summed): connects $A$'s columns to $v$'s entries
- $i$ is free (result index): selects row of $A$ and row of result $w$
- $w \in \mathbb{R}^m$, $A \in \mathbb{R}^{m \times n}$, $v \in \mathbb{R}^n$; $j$ ranges over $n$ ✓

This is the standard matrix-vector product $w = Av$.

### 3.6 Matrix-Matrix Multiply

$$C_{ij} = A_{ik} B_{kj} = \sum_k A_{ik} B_{kj}$$

- $k$ is dummy: summed over; connects $A$'s columns to $B$'s rows
- $i, j$ are free: $i$ selects row of $A$, $j$ selects column of $B$
- $C \in \mathbb{R}^{m \times n}$, $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times n}$; $k$ ranges over $p$ ✓

Equivalently: $C = AB$

### 3.7 Transpose

$$(A^\top)_{ij} = A_{ji}$$

Simply swap index labels. No summation; pure relabelling. Note that $A_{ji}$ and $A_{ij}$ are **different** expressions — different index order means transposed access.

### 3.8 Trace

$$\text{tr}(A) = A_{ii} = \sum_i A_{ii}$$

The **same index** appears twice on the **same tensor** — sum over diagonal elements. Result is scalar; no free index. Valid only for square matrices (both index ranges must be equal).

**Trace of product:**

$$\text{tr}(AB) = A_{ij} B_{ji} = A_{ik} B_{ki}$$

Both pairs $(i,j)$ or $(i,k)$ are dummy → scalar. This equals $\text{tr}(BA)$ — rename dummy index to verify: $A_{ij}B_{ji} = B_{ji}A_{ij} = B_{ij}A_{ji}$ (relabel $i \leftrightarrow j$) $= \text{tr}(BA)$ ✓

### 3.9 Frobenius Inner Product

$$\langle A, B \rangle_F = A_{ij} B_{ij} = \sum_i \sum_j A_{ij} B_{ij}$$

Both $i$ and $j$ are dummy — double summation. Result is scalar. Equivalently:

$$\langle A, B \rangle_F = \text{tr}(A^\top B) = (A^\top)_{ji} B_{ji} = A_{ij} B_{ij} \quad \checkmark$$

**Frobenius norm:** $\|A\|_F^2 = A_{ij} A_{ij}$

### 3.10 Element-wise (Hadamard) Product

$$(A \odot B)_{ij} = A_{ij} B_{ij}$$

Both $A$ and $B$ share indices $i, j$. But here $i, j$ are **free** — both appear in the result. No summation; elementwise multiplication.

**Critical distinction:** The expression $A_{ij}B_{ij}$ looks the same as the Frobenius inner product. The difference is whether $i, j$ are free or dummy — which depends on whether the result carries those indices:

| Einsum string | $i, j$ status | Operation | Result |
|---|---|---|---|
| `"ij,ij->ij"` | free | Elementwise multiply | Matrix |
| `"ij,ij->"` | dummy | Frobenius inner product | Scalar |

In standard mathematical notation, context determines interpretation. In `einsum`, the output string makes it explicit.

```
COMPLETE BASIC OPERATIONS REFERENCE
═══════════════════════════════════════════════════════════════════

  Operation          Index Form       Einsum     Free  Dummy  Result
  ─────────────────  ──────────────   ─────────  ────  ─────  ──────
  Scalar multiply    s·v_i            scalar     i     —      vector
  Vector add         u_i + v_i        —          i     —      vector
  Dot product        u_i v_i          "i,i->"    —     i      scalar
  Outer product      u_i v_j          "i,j->ij"  i,j   —      matrix
  Matrix-vector      A_ij v_j         "ij,j->i"  i     j      vector
  Matrix multiply    A_ik B_kj        "ik,kj->ij" i,j  k      matrix
  Transpose          A_ji             "ij->ji"   i,j   —      matrix
  Trace              A_ii             "ii->"     —     i      scalar
  Frobenius inner    A_ij B_ij        "ij,ij->"  —     i,j    scalar
  Hadamard product   A_ij B_ij        "ij,ij->ij" i,j  —      matrix
```

---

## 4. Contraction Operations

### 4.1 What Is Contraction?

**Contraction** is the operation of summing over one or more pairs of repeated indices, reducing tensor rank by 2 for each contracted pair.

Given input tensors of ranks $r_1$ and $r_2$, contracting over $k$ pairs yields a result of rank:

$$\text{result rank} = r_1 + r_2 - 2k$$

| Inputs | Ranks | Contracted pairs | Result rank | Operation |
|---|---|---|---|---|
| scalar × scalar | 0 + 0 | 0 | 0 | Scalar multiply |
| vector · vector | 1 + 1 | 1 | 0 | Dot product |
| matrix × vector | 2 + 1 | 1 | 1 | Linear map |
| matrix × matrix | 2 + 2 | 1 | 2 | Matrix multiply |
| Frobenius inner product | 2 + 2 | 2 | 0 | Scalar |
| trace | 2 | 1 (self) | 0 | Diagonal sum |

### 4.2 Single Tensor Contraction (Trace-Like)

Contracting two indices of the **same** tensor generalises the matrix trace:

$$T_{ijk} \xrightarrow{\text{contract } i,j} S_k = T_{iik} = \sum_i T_{iik}$$

Result: rank-1 tensor (vector). $k$ is free; $i$ is dummy.

This is a **partial trace** — tracing over a subset of indices. It generalises the matrix trace to higher-order tensors.

**AI example:** In multi-head attention, computing per-head statistics involves contracting over query/key positions while keeping the head dimension free.

### 4.3 Two-Tensor Contraction

General form: contract tensors $A$ (rank $p$) and $B$ (rank $q$) over $k$ shared indices → result rank $p + q - 2k$.

| Contraction | Index form | $p + q - 2k$ | Operation |
|---|---|---|---|
| 1 pair | $A_{ik}B_{kj} \to C_{ij}$ | $2+2-2=2$ | Matrix multiply |
| 2 pairs | $A_{ij}B_{ij} \to s$ | $2+2-4=0$ | Frobenius inner product |
| 2 pairs (mixed rank) | $A_{ijk}B_{jk} \to v_i$ | $3+2-4=1$ | Vector result |

### 4.4 Contraction Order and Computational Cost

For expression $A_{ij}B_{jk}C_{kl}$ (three matrices), there are two possible contraction orders:

**(AB)C — left to right:**
1. Compute $D_{ik} = A_{ij}B_{jk}$: cost $O(m \cdot p \cdot n)$ multiplications
2. Compute $D_{ik}C_{kl}$: cost $O(m \cdot n \cdot q)$ multiplications

**A(BC) — right to left:**
1. Compute $E_{jl} = B_{jk}C_{kl}$: cost $O(p \cdot n \cdot q)$ multiplications
2. Compute $A_{ij}E_{jl}$: cost $O(m \cdot p \cdot q)$ multiplications

For rectangular tensors, different orders can differ by **orders of magnitude**.

**Example:** $A \in \mathbb{R}^{1000 \times 10}$, $B \in \mathbb{R}^{10 \times 10}$, $C \in \mathbb{R}^{10 \times 1000}$

- $(AB)C$: $1000 \times 10 \times 10 + 1000 \times 10 \times 1000 = 10^5 + 10^7 = 10.1 \times 10^6$
- $A(BC)$: $10 \times 10 \times 1000 + 1000 \times 10 \times 1000 = 10^5 + 10^7 = 10.1 \times 10^6$

**But:** $A \in \mathbb{R}^{1000 \times 1000}$, $B \in \mathbb{R}^{1000 \times 2}$, $C \in \mathbb{R}^{2 \times 1000}$

- $(AB)C$: $1000^2 \times 2 + 1000 \times 2 \times 1000 = 2 \times 10^6 + 2 \times 10^6 = 4 \times 10^6$
- $A(BC)$: $1000 \times 2 \times 1000 + 1000^2 \times 1000 = 2 \times 10^6 + 10^9$ ← **250× slower!**

Finding the optimal contraction order is NP-hard in general, but greedy approaches work well in practice. The `opt_einsum` library finds near-optimal contraction orders for multi-tensor einsum expressions.

### 4.5 The Generalised Contraction Formula

For tensors $A_{i_1\ldots i_p}$ and $B_{j_1\ldots j_q}$, contracting over shared index set $S$:

$$C_{\text{free indices}} = \sum_{\alpha \in S} A_{\ldots\alpha\ldots} B_{\ldots\alpha\ldots}$$

- **Free indices:** union of all non-contracted indices from both tensors
- **Result shape:** determined by free index ranges
- This is the most general single-step operation expressible in `einsum`

```
CONTRACTION VISUALISED
═══════════════════════════════════════════════════════════════════

Matrix Multiply: C_ij = A_ik B_kj

  A (rank 2)      B (rank 2)        C (rank 2)
  ┌───┐           ┌───┐             ┌───┐
  │   │──k──────k─│   │    =        │   │
  │   │           │   │             │   │
  └───┘           └───┘             └───┘
   ↕ i              ↕ j              ↕ i  ↕ j
   (free)           (free)           (free indices)

  k is contracted (summed) → disappears from result
  i, j are free → appear in result

Frobenius Inner Product: s = A_ij B_ij

  A (rank 2)      B (rank 2)        s (rank 0)
  ┌───┐           ┌───┐
  │   │──i──────i─│   │    =        ● (scalar)
  │   │──j──────j─│   │
  └───┘           └───┘

  Both i and j contracted → no free indices → scalar result
```

---

## 5. The Kronecker Delta and Levi-Civita Symbol

### 5.1 Kronecker Delta

The Kronecker delta is the identity matrix expressed in index notation:

$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

As a matrix: $\delta_{ij} = I_{ij}$ — the identity matrix.

**The key property — index substitution (index killing):**

$$A_{i\ldots} \delta_{ij} = A_{j\ldots}$$

**Proof:** $\sum_i A_i \delta_{ij} = A_j \times 1 = A_j$ (only the term with $i = j$ survives the sum).

This is the most important property in all of index notation. It says: contracting any tensor with $\delta_{ij}$ **replaces** index $i$ with index $j$ (or vice versa). The delta "kills" the summed index and substitutes the other.

**Other properties:**
- Trace of identity: $\delta_{ii} = \sum_i 1 = n$ (dimension of the index space)
- Acts as identity: $\delta_{ij} u_j = u_i$ (same as $Iu = u$ in matrix notation)
- Idempotent: $\delta_{ij}\delta_{jk} = \delta_{ik}$ (identity times identity = identity)

### 5.2 Kronecker Delta Identities

| Identity | Proof | Matrix Equivalent |
|---|---|---|
| $\delta_{ij}\delta_{jk} = \delta_{ik}$ | $\sum_j \delta_{ij}\delta_{jk} = \delta_{ik}$ (only $j=i=k$ survives) | $I \cdot I = I$ |
| $\delta_{ii} = n$ | $\sum_{i=1}^n 1 = n$ | $\text{tr}(I) = n$ |
| $\delta_{ij}A_{jk} = A_{ik}$ | Index substitution: $j \to i$ | $IA = A$ |
| $A_{ij}\delta_{jk} = A_{ik}$ | Index substitution: $j \to k$ | $AI = A$ |
| $\delta_{ij}A_{ij} = A_{ii} = \text{tr}(A)$ | Substitution reduces to trace | $\text{tr}(IA) = \text{tr}(A)$ |

**AI use:** The Kronecker delta appears in every gradient derivation. When you differentiate $C_{ij} = A_{ik}B_{kj}$ with respect to $A_{lm}$, the derivative $\partial A_{ik}/\partial A_{lm} = \delta_{il}\delta_{km}$ — the delta selects which element you're differentiating. This is the engine of backpropagation derivations.

### 5.3 Levi-Civita Symbol (Permutation Symbol)

The Levi-Civita symbol is a completely antisymmetric tensor. In 3D:

$$\varepsilon_{ijk} = \begin{cases} +1 & \text{if } (i,j,k) \text{ is an even permutation of } (1,2,3) \\ -1 & \text{if } (i,j,k) \text{ is an odd permutation of } (1,2,3) \\ 0 & \text{if any two indices are equal} \end{cases}$$

**Even permutations** (cyclic): $(1,2,3)$, $(2,3,1)$, $(3,1,2)$ → $\varepsilon = +1$

**Odd permutations** (one swap): $(1,3,2)$, $(3,2,1)$, $(2,1,3)$ → $\varepsilon = -1$

The symbol is non-zero only when all three indices are distinct. In $n$ dimensions, $\varepsilon_{i_1 i_2 \ldots i_n}$ generalises to $n!$ non-zero values out of $n^n$ possible.

### 5.4 Levi-Civita Applications

**Cross product:**

$$(u \times v)_i = \varepsilon_{ijk} u_j v_k = \sum_j \sum_k \varepsilon_{ijk} u_j v_k$$

The index structure encodes the cross product formula: $i$ is free (result component), $j$ and $k$ are dummy (summed over).

**Determinant (3×3):**

$$\det(A) = \varepsilon_{ijk} A_{1i} A_{2j} A_{3k}$$

All of $i, j, k$ are dummy → result is scalar. The Levi-Civita selects only the six permutations with correct signs.

**Key identity — $\varepsilon$-$\delta$ contraction:**

$$\varepsilon_{ijk} \varepsilon_{ilm} = \delta_{jl}\delta_{km} - \delta_{jm}\delta_{kl}$$

This allows simplifying double Levi-Civita contractions into Kronecker deltas — essential for proofs involving cross products and curl operations.

**AI context:** The Levi-Civita symbol is less directly used than the Kronecker delta in ML. It appears in:
- Determinant computations for normalising flows ($\log |\det J|$)
- Theoretical analyses of rotation equivariance
- Differential geometry of neural manifolds

### 5.5 Generalised Kronecker Delta

The generalised Kronecker delta extends the concept to multi-index antisymmetry:

$$\delta^{i_1 i_2 \ldots i_k}_{j_1 j_2 \ldots j_k} = \begin{cases} +1 & \text{if } (i_1,\ldots,i_k) \text{ is an even permutation of } (j_1,\ldots,j_k) \\ -1 & \text{if odd permutation} \\ 0 & \text{otherwise} \end{cases}$$

This expresses antisymmetrisation and symmetrisation operators, and connects to the Levi-Civita symbol:

$$\varepsilon_{ijk} \varepsilon_{lmn} = \delta^{ijk}_{lmn} = \delta_{il}\delta_{jm}\delta_{kn} + \delta_{im}\delta_{jn}\delta_{kl} + \delta_{in}\delta_{jl}\delta_{km} - \delta_{il}\delta_{jn}\delta_{km} - \delta_{im}\delta_{jl}\delta_{kn} - \delta_{in}\delta_{jm}\delta_{kl}$$

---

## 6. Index Manipulation Rules

### 6.1 Renaming Dummy Indices

Dummy indices are **bound variables** — any name works. This is alpha-equivalence, just like in lambda calculus.

$$A_{ij}B_{jk} = A_{il}B_{lk} \quad \text{(rename } j \to l \text{; identical expression)}$$

**Rules for renaming:**
1. Rename **consistently** throughout the entire term: both occurrences change together
2. Don't rename to an index already in use (avoid clashes)
3. Free indices cannot be renamed within a term without renaming on both sides of the equation

**Use renaming to:**
- Avoid index clashes when combining expressions
- Make contraction structure explicit
- Prepare expressions for interchange or simplification

### 6.2 Symmetry and Antisymmetry

**Symmetric tensor:** $A_{ij} = A_{ji}$ (unchanged under index swap)
- Examples: metric tensor $g_{ij}$; Hessian $\partial^2 f / \partial x_i \partial x_j$; covariance matrix $\Sigma_{ij}$; Gram matrix $G_{ij} = x_i^\top x_j$

**Antisymmetric tensor:** $A_{ij} = -A_{ji}$ (sign flip under index swap)
- Examples: Levi-Civita $\varepsilon_{ijk}$; cross product matrix $[\omega]_\times$
- Consequence: $A_{ii} = -A_{ii} \implies A_{ii} = 0$ — diagonal elements are always zero

**Critical identity — antisymmetric contracted with symmetric vanishes:**

If $A_{ij}$ is antisymmetric and $S_{ij}$ is symmetric:

$$A_{ij} S_{ij} = 0$$

**Proof:**
$$A_{ij}S_{ij} = A_{ji}S_{ji} \quad \text{(rename } i \leftrightarrow j \text{)} = (-A_{ij})S_{ij} \quad \text{(use antisymmetry of } A \text{, symmetry of } S \text{)}$$
$$\implies 2A_{ij}S_{ij} = 0 \implies A_{ij}S_{ij} = 0 \quad \checkmark$$

**AI relevance:** Attention score matrices $S_{ij} = Q_i \cdot K_j$ are generally **not** symmetric (queries ≠ keys). Weight matrices are not symmetric. Understanding symmetry properties helps simplify gradient expressions.

### 6.3 Symmetrisation and Antisymmetrisation

Any rank-2 tensor decomposes uniquely into symmetric and antisymmetric parts:

$$T_{ij} = T_{(ij)} + T_{[ij]}$$

where:

**Symmetric part:**

$$T_{(ij)} = \frac{1}{2}(T_{ij} + T_{ji})$$

**Antisymmetric part:**

$$T_{[ij]} = \frac{1}{2}(T_{ij} - T_{ji})$$

For higher-rank tensors, symmetrisation/antisymmetrisation extends to any subset of indices:

$$T_{(ij)k} = \frac{1}{2}(T_{ijk} + T_{jik})$$

**AI application:** Decomposing the attention weight gradient into symmetric and antisymmetric components can reveal which parts drive updates to query vs key projections independently.

### 6.4 Index Raising and Lowering (Physics)

In differential geometry, the **metric tensor** $g_{ij}$ converts between covariant and contravariant indices:

$$v_i = g_{ij} v^j \quad \text{(lower index)}$$
$$v^i = g^{ij} v_j \quad \text{(raise index, using inverse metric } g^{ij} \text{)}$$

In flat Euclidean space: $g_{ij} = \delta_{ij}$ → raising and lowering are trivial (no change).

**ML context:** All ML computations happen in Euclidean space (or spaces treated as Euclidean). The metric is identity; upper/lower distinction collapses. When reading physics papers, mentally replace $g_{ij} \to \delta_{ij}$.

**Exception — natural gradient:** Fisher information matrix $F_{ij}$ acts as a metric on parameter space. In natural gradient descent, $F_{ij}$ plays the role of $g_{ij}$, and the "covariant gradient" $\tilde{\nabla}_i = F^{ij} \nabla_j$ differs from the ordinary gradient. This is one place where the physics convention is genuinely useful in ML.

### 6.5 Partial Derivatives in Index Notation

Partial derivatives get their own index notation, making gradient calculations compact:

| Notation | Standard form | Einstein form | Result |
|---|---|---|---|
| Gradient | $(\nabla f)_i = \frac{\partial f}{\partial x_i}$ | $\partial_i f$ | Vector |
| Jacobian | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ | $\partial_j f_i$ | Matrix |
| Hessian | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | $\partial_i \partial_j f$ | Matrix (symmetric) |
| Divergence | $\nabla \cdot v = \sum_i \frac{\partial v_i}{\partial x_i}$ | $\partial_i v_i$ | Scalar ($i$ is dummy) |
| Laplacian | $\nabla^2 f = \sum_i \frac{\partial^2 f}{\partial x_i^2}$ | $\partial_i \partial_i f$ | Scalar ($i$ is dummy) |

The divergence $\partial_i v_i$ has $i$ appearing twice → summed → scalar. The Laplacian $\partial_i \partial_i f$ has the same structure — a "trace" of second derivatives.

**AI connection:** Gradient computation in index notation is the foundation of Section 7. The Jacobian-vector product $J_{ij}v_j = \partial_j f_i \cdot v_j$ is exactly what automatic differentiation computes in forward mode. The vector-Jacobian product $u_i J_{ij} = u_i \partial_j f_i$ is reverse mode (backpropagation).

---

## 7. Einstein Notation for Gradients and Backpropagation

Index notation turns gradient derivations from pattern matching into mechanical computation. The core tool is the **Kronecker delta** — the derivative of a tensor component with respect to itself.

### 7.1 Gradient of Linear Form

$f(x) = a_i x_i$ (dot product $a \cdot x$)

$$\frac{\partial f}{\partial x_j} = \frac{\partial}{\partial x_j}(a_i x_i) = a_i \frac{\partial x_i}{\partial x_j} = a_i \delta_{ij} = a_j$$

The $\delta_{ij}$ kills the sum over $i$, substituting $j$ for $i$. Result: $\nabla_x(a^\top x) = a$ ✓

### 7.2 Gradient of Quadratic Form

$f(x) = x_i A_{ij} x_j$ (quadratic form $x^\top A x$)

$$\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k}(x_i A_{ij} x_j)$$

Apply product rule (both $x_i$ and $x_j$ depend on $x_k$):

$$= \frac{\partial x_i}{\partial x_k} A_{ij} x_j + x_i A_{ij} \frac{\partial x_j}{\partial x_k} = \delta_{ik} A_{ij} x_j + x_i A_{ij} \delta_{jk}$$

$$= A_{kj} x_j + x_i A_{ik} = (Ax)_k + (A^\top x)_k = (A + A^\top)_{kj} x_j$$

**When $A$ is symmetric** ($A_{ij} = A_{ji}$): $\frac{\partial f}{\partial x_k} = 2A_{kj}x_j = 2(Ax)_k$

$$\boxed{\nabla_x(x^\top A x) = (A + A^\top)x \quad \text{; for symmetric } A\text{: } 2Ax}$$

### 7.3 Gradient of Matrix Multiply (A side)

Loss $L$; output $C = AB$ where $A \in \mathbb{R}^{m \times p}$, $B \in \mathbb{R}^{p \times n}$; $C_{ij} = A_{ik}B_{kj}$.

Given upstream gradient $\frac{\partial L}{\partial C_{ij}}$, find $\frac{\partial L}{\partial A_{lm}}$:

$$\frac{\partial L}{\partial A_{lm}} = \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial A_{lm}}$$

Compute the inner derivative:

$$\frac{\partial C_{ij}}{\partial A_{lm}} = \frac{\partial}{\partial A_{lm}}(A_{ik}B_{kj}) = \delta_{il}\delta_{km} B_{kj} = \delta_{il} B_{mj}$$

Substitute back:

$$\frac{\partial L}{\partial A_{lm}} = \frac{\partial L}{\partial C_{ij}} \delta_{il} B_{mj} = \frac{\partial L}{\partial C_{lj}} B_{mj}$$

In index notation: $\left(\frac{\partial L}{\partial A}\right)_{lm} = \left(\frac{\partial L}{\partial C}\right)_{lj} B_{mj}$

$$\boxed{\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^\top}$$

This is the standard backpropagation formula — **derived mechanically** using only $\delta_{ij}$ substitution.

### 7.4 Gradient of Matrix Multiply (B side)

$$\frac{\partial L}{\partial B_{mn}} = \frac{\partial L}{\partial C_{ij}} \frac{\partial C_{ij}}{\partial B_{mn}}$$

$$\frac{\partial C_{ij}}{\partial B_{mn}} = \frac{\partial}{\partial B_{mn}}(A_{ik}B_{kj}) = A_{im}\delta_{jn}$$

$$\frac{\partial L}{\partial B_{mn}} = \frac{\partial L}{\partial C_{ij}} A_{im}\delta_{jn} = A_{im} \frac{\partial L}{\partial C_{in}}$$

$$\boxed{\frac{\partial L}{\partial B} = A^\top \cdot \frac{\partial L}{\partial C}}$$

### 7.5 Gradient of Dot Product

$s = a_i b_i$

$$\frac{\partial s}{\partial a_j} = b_i \delta_{ij} = b_j \implies \nabla_a(a^\top b) = b$$

$$\frac{\partial s}{\partial b_j} = a_i \delta_{ij} = a_j \implies \nabla_b(a^\top b) = a$$

### 7.6 Gradient of Trace

$f = \text{tr}(AB) = A_{ij}B_{ji}$

$$\frac{\partial f}{\partial A_{lm}} = \frac{\partial}{\partial A_{lm}}(A_{ij}B_{ji}) = \delta_{il}\delta_{jm} B_{ji} = B_{ml} = (B^\top)_{lm}$$

$$\boxed{\nabla_A \text{tr}(AB) = B^\top}$$

Similarly:
- $\nabla_A \text{tr}(A^\top B) = B$
- $\nabla_A \text{tr}(A^\top A) = 2A$ (since A appears twice, both contribute)

### 7.7 Gradient of Softmax in Index Notation

Softmax: $p_v = \frac{\exp(z_v)}{\sum_w \exp(z_w)} = \frac{\exp(z_v)}{Z}$

The Jacobian $\frac{\partial p_v}{\partial z_w}$ is derived via quotient rule:

$$\frac{\partial p_v}{\partial z_w} = \frac{\frac{\partial \exp(z_v)}{\partial z_w} \cdot Z - \exp(z_v) \cdot \frac{\partial Z}{\partial z_w}}{Z^2}$$

where $\frac{\partial \exp(z_v)}{\partial z_w} = \delta_{vw} \exp(z_v)$ and $\frac{\partial Z}{\partial z_w} = \exp(z_w)$:

$$= \frac{\delta_{vw}\exp(z_v) Z - \exp(z_v)\exp(z_w)}{Z^2} = p_v\delta_{vw} - p_v p_w$$

$$\boxed{\frac{\partial p_v}{\partial z_w} = p_v(\delta_{vw} - p_w)}$$

**Gradient of loss through softmax:**

$$\frac{\partial L}{\partial z_w} = \sum_v \frac{\partial L}{\partial p_v} \frac{\partial p_v}{\partial z_w} = \sum_v \frac{\partial L}{\partial p_v} p_v(\delta_{vw} - p_w)$$

$$= p_w \frac{\partial L}{\partial p_w} - p_w \sum_v p_v \frac{\partial L}{\partial p_v}$$

For cross-entropy loss with true class $y$: $\frac{\partial L}{\partial p_v} = -1/p_v$ for $v = y$, zero otherwise. This simplifies to:

$$\frac{\partial L}{\partial z_w} = p_w - \mathbb{1}[w = y]$$

The elegant result: the gradient of cross-entropy loss through softmax is just predicted probability minus the one-hot target. This simplicity comes directly from the index-notation derivation.

```
GRADIENT DERIVATION CHEAT SHEET
═══════════════════════════════════════════════════════════════════

  Function                  Gradient via Index Notation
  ────────────────────────  ──────────────────────────────────────
  f = a_i x_i               ∂f/∂x_j = a_j          (linear)
  f = x_i A_ij x_j          ∂f/∂x_k = (A+Aᵀ)_kj x_j (quadratic)
  C_ij = A_ik B_kj          ∂L/∂A_lm = (∂L/∂C)_lj B_mj   (A side)
                             ∂L/∂B_mn = A_im (∂L/∂C)_in    (B side)
  f = tr(AB)                 ∂f/∂A_lm = B_ml = (Bᵀ)_lm
  p_v = softmax(z)_v         ∂p_v/∂z_w = p_v(δ_vw - p_w)

  TOOL: ∂T_ab/∂T_cd = δ_ac δ_bd  (Kronecker delta — the engine)
```

---

## 8. Batched Operations and Batch Indices

### 8.1 Introducing the Batch Index

In ML, the same operation is applied to $B$ independent examples simultaneously. The batch dimension is represented by adding a leading index $b \in \{1, \ldots, B\}$.

**Batched matrix-vector (per-sample weights):**

$$w_{bi} = A_{bij} v_{bj}$$

For each batch element $b$, compute $A_b v_b$ independently.

**Shared weight matrix (common in neural nets):**

$$w_{bi} = A_{ij} v_{bj}$$

$A$ has no $b$ index — the same weight matrix is applied to every sample. $v_{bj}$ varies per batch element.

In einsum: `"ij,bj->bi"` — $b$ is free (appears in output), $j$ is dummy (contracted).

### 8.2 Batched Matrix Multiply

$$C_{bij} = A_{bik} B_{bkj}$$

Each batch element $b$ has its own $A_b$ and $B_b$; multiply independently. $k$ is dummy; $b, i, j$ are free.

Einsum: `"bik,bkj->bij"`

PyTorch: `torch.bmm(A, B)` for rank-3 tensors.

### 8.3 Multi-Head Attention in Index Notation

Full multi-head attention with explicit indices:

- $Q, K, V \in \mathbb{R}^{B \times H \times n \times d}$ where $B$ = batch, $H$ = heads, $n$ = sequence length, $d$ = head dimension

**Attention scores:**

$$S_{bhij} = \frac{Q_{bhia} K_{bhja}}{\sqrt{d}}$$

- $a$ is dummy (summed over head dimension → dot product)
- $b, h, i, j$ are free (per batch, per head, per query-key pair)
- Einsum: `"bhid,bhjd->bhij"` (this is $QK^\top$ per head)

**After softmax** (applied over $j$ for each $b, h, i$):

$$\alpha_{bhij} = \text{softmax}_j(S_{bhij})$$

**Attention output:**

$$O_{bhia} = \alpha_{bhij} V_{bhja}$$

- $j$ is dummy (weighted sum over values)
- $b, h, i, a$ are free (per batch, per head, per position, per dimension)
- Einsum: `"bhij,bhjd->bhid"`

### 8.4 Broadcasting as Implicit Index Repetition

NumPy/PyTorch broadcasting corresponds to an index **not appearing** on one tensor — it is implicitly repeated:

| Code operation | Index notation | Broadcasting rule |
|---|---|---|
| $A_{bi} + c_i$ | $b$ absent on $c$ | Same bias for all batch elements |
| $(Av)_{bi} = A_{bij}v_j$ | $b$ absent on $v$ | Same vector for all batch elements |
| $x_{bti} + \text{PE}_{ti}$ | $b$ absent on PE | Same positional encoding for all batches |

Understanding broadcasting as implicit indexing prevents shape errors. If you write the index notation first and check which tensors carry which indices, the broadcasting rules become obvious.

### 8.5 Reduction Operations

Summing over an index eliminates it from the result:

| Operation | Index notation | Einsum | Description |
|---|---|---|---|
| Sum over batch | $s_i = x_{bi}$ ($b$ dummy) | `"bi->i"` | Average features across batch |
| Mean over batch | $\mu_i = \frac{1}{B} x_{bi}$ | `"bi->i"` ÷ B | Batch mean |
| Sum over sequence | $s_{bi} = x_{bti}$ ($t$ dummy) | `"bti->bi"` | Sum over positions |
| Global sum | $s = x_{bti}$ ($b,t,i$ all dummy) | `"bti->"` | Total sum → scalar |

Pooling operations are all forms of reduction over specific index dimensions. Average pooling = sum + normalise. Max pooling breaks the einsum pattern (non-linear) but is still a reduction over an index.

---

## 9. Einsum in Practice

### 9.1 Einsum String Syntax

The `einsum` function (available in NumPy, PyTorch, JAX, TensorFlow) implements Einstein summation directly:

**Format:** `"input1_indices, input2_indices, ... -> output_indices"`

**Rules:**
- Input operands separated by commas; output after arrow (`->`)
- Each character is an index label (single letter)
- Index appearing on multiple inputs → contraction (sum) if not in output
- Index in output → free (kept in result)
- Index omitted from output → dummy (summed over)

**Example:** `"ik,kj->ij"` means $C_{ij} = A_{ik}B_{kj}$ — matrix multiply. Index $k$ appears on both inputs but not in output → contracted.

### 9.2 Standard Operations as Einsum Strings

| Operation | Mathematical | Einsum String | Notes |
|---|---|---|---|
| Dot product | $a_i b_i$ | `"i,i->"` | Both indices contracted |
| Outer product | $a_i b_j$ | `"i,j->ij"` | No contraction |
| Matrix-vector | $A_{ij}v_j$ | `"ij,j->i"` | $j$ contracted |
| Matrix multiply | $A_{ik}B_{kj}$ | `"ik,kj->ij"` | $k$ contracted |
| Batched matmul | $A_{bik}B_{bkj}$ | `"bik,bkj->bij"` | $k$ contracted; $b$ free |
| Transpose | $A_{ji}$ | `"ij->ji"` | Relabel only |
| Trace | $A_{ii}$ | `"ii->"` | Self-contraction |
| Row sum | $\sum_j A_{ij}$ | `"ij->i"` | $j$ contracted |
| Column sum | $\sum_i A_{ij}$ | `"ij->j"` | $i$ contracted |
| Element-wise multiply | $A_{ij}B_{ij}$ | `"ij,ij->ij"` | No contraction |
| Frobenius inner product | $A_{ij}B_{ij}$ | `"ij,ij->"` | Both contracted |
| Attention scores ($QK^\top$) | $Q_{ia}K_{ja}$ | `"ia,ja->ij"` | $a$ contracted |
| Attention output | $\alpha_{ij}V_{ja}$ | `"ij,ja->ia"` | $j$ contracted |
| Quadratic form | $x_i A_{ij} x_j$ | `"i,ij,j->"` | $i,j$ contracted |
| Bilinear form | $x_i M_{ij} y_j$ | `"i,ij,j->"` | $i,j$ contracted |
| Batch outer product | $a_{bi} b_{bj}$ | `"bi,bj->bij"` | $b$ free |
| Diagonal extraction | $A_{ii}$ | `"ii->i"` | Diagonal as vector |

### 9.3 Multi-Operand Einsum

Einsum supports more than two input tensors:

```python
torch.einsum("ijk,jl,km->ilm", A, B, C)
```

This computes $D_{ilm} = A_{ijk} B_{jl} C_{km}$ — contracting $j$ (between $A$ and $B$) and $k$ (between $A$ and $C$) simultaneously.

**Contraction order matters.** For three-tensor contraction $A_{ij}B_{jk}C_{kl} \to D_{il}$:

- **Option 1:** $(AB)C$ — compute $E_{ik} = A_{ij}B_{jk}$ first, then $E_{ik}C_{kl}$
- **Option 2:** $A(BC)$ — compute $F_{jl} = B_{jk}C_{kl}$ first, then $A_{ij}F_{jl}$

For rectangular tensors, one order can be **orders of magnitude** faster. The `opt_einsum` library finds near-optimal contraction paths automatically:

```python
import opt_einsum
# Finds optimal contraction order for large multi-tensor expressions
path, info = opt_einsum.contract_path("ijk,jl,km->ilm", A, B, C)
result = opt_einsum.contract("ijk,jl,km->ilm", A, B, C, optimize=path)
```

### 9.4 Implicit vs Explicit Mode

**Implicit mode** (no arrow): output indices = all indices appearing exactly once, in alphabetical order.

- `"ij,jk"` → implicit output `"ik"` ($j$ appears twice → contracted)
- Convenient shorthand; may be ambiguous for complex expressions

**Explicit mode** (with arrow): output indices specified exactly.

- `"ij,jk->ki"` → transposed result (column-major output)
- Always use explicit mode for clarity

**Rule of thumb:** Always use explicit mode in production code. Implicit mode is fine for quick interactive exploration but introduces ambiguity in complex expressions.

### 9.5 Performance Considerations

- **Einsum vs manual matmul:** Modern frameworks compile einsum to optimised CUDA kernels. For standard patterns (matmul, bmm), performance is equivalent.
- **Fused operations:** Einsum can fuse multiple contractions, avoiding materialisation of intermediate tensors. Example: `"bhid,bhjd->bhij"` computes attention scores without explicitly computing the transpose.
- **Limitations:** Not all einsum patterns are equally optimised on all GPUs. For critical paths (attention in production), manually optimised kernels (FlashAttention) outperform generic einsum.
- **Memory:** Einsum can be memory-efficient by not materialising intermediates, but can also consume unexpected memory for large multi-operand expressions.

### 9.6 Common Pitfalls in Code

| Pitfall | Problem | Fix |
|---|---|---|
| Shape mismatch | $A_{ik}B_{kj}$ requires `A.shape[1] == B.shape[0]` | Check shapes before einsum |
| Wrong index string | `"ij,jk->ij"` is NOT matmul | Should be `"ij,jk->ik"` |
| Missing batch dim | `"ij,jk->ik"` on batched input | Use `"bij,bjk->bik"` |
| Implicit contraction surprise | Accidentally sharing index names | Use explicit output `->` |
| Integer tensors | Einsum with int tensors may give wrong results | Cast to float first |
| Repeated index same tensor | `"ij,ij->ij"` vs `"ii->"` | Understand the difference |

---

## 10. Tensor Products and Decompositions

### 10.1 Tensor Product (Outer Product Generalisation)

The tensor product of $A$ (rank $p$) and $B$ (rank $q$) produces a rank-$(p+q)$ tensor:

$$(A \otimes B)_{i_1\ldots i_p j_1\ldots j_q} = A_{i_1\ldots i_p} B_{j_1\ldots j_q}$$

No contraction — all indices are free. The result carries all indices of both operands.

**Special cases:**
- Outer product of vectors: $(u \otimes v)_{ij} = u_i v_j$ → rank-2 (matrix)
- Outer product of matrices: $(A \otimes B)_{ijkl} = A_{ij} B_{kl}$ → rank-4
- Kronecker product: specific arrangement of matrix tensor product

**AI connection — LoRA:** The low-rank update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ is a sum of $r$ rank-1 outer products: $\Delta W_{ij} = B_{ik}A_{kj} = \sum_{k=1}^r B_{ik}A_{kj}$. The index $k$ ranges over the LoRA rank $r$, making the low-rank structure explicit.

### 10.2 Tensor Unfolding (Matricisation)

Reshaping a rank-$r$ tensor into a matrix by selecting which indices form rows vs columns:

**Mode-$k$ unfolding** $T_{(k)}$: index $k$ forms rows; all other indices (combined) form columns.

$$\text{Shape: } n_k \times (n_1 \times \cdots \times n_{k-1} \times n_{k+1} \times \cdots \times n_r)$$

**Example:** $T \in \mathbb{R}^{3 \times 4 \times 5}$

- Mode-1 unfolding: $3 \times 20$ matrix
- Mode-2 unfolding: $4 \times 15$ matrix
- Mode-3 unfolding: $5 \times 12$ matrix

Used in Tucker decomposition, tensor-train networks, and model compression. The unfolding is the bridge between higher-order tensor operations and standard matrix algorithms (SVD, eigendecomposition).

### 10.3 CP Decomposition (CANDECOMP/PARAFAC)

Decompose a tensor as a sum of rank-1 tensors:

$$T_{ijk} = \sum_{r=1}^{R} \lambda_r \, a^{(1)}_{ir} \, a^{(2)}_{jr} \, a^{(3)}_{kr}$$

In Einstein notation: $T_{ijk} = \lambda_r \, a^{(1)}_{ir} \, a^{(2)}_{jr} \, a^{(3)}_{kr}$ — the index $r$ is dummy (summed over rank components).

Each term $\lambda_r \, a^{(1)}_{ir} \, a^{(2)}_{jr} \, a^{(3)}_{kr}$ is a rank-1 tensor (outer product of three vectors). $R$ is the **tensor rank** — the minimum number of rank-1 terms needed for exact representation.

**AI use:** Low-rank CP decomposition for compressing embedding tables, convolutional filters, and MoE routing tensors.

### 10.4 Tucker Decomposition

$$T_{ijk} = G_{abc} \, U^{(1)}_{ia} \, U^{(2)}_{jb} \, U^{(3)}_{kc}$$

where $a, b, c$ are dummy indices (contracted):
- $G$ = core tensor (small, captures interactions between modes)
- $U^{(n)}$ = factor matrices (orthogonal, capture principal components per mode)

This generalises SVD to higher-order tensors. Truncating the core tensor dimensions yields compression.

**AI use:** Compressing convolutional filters (4D tensors: output channels × input channels × height × width). Compressing transformer weight tensors. The Tucker structure explicitly shows which modes interact through the core tensor.

### 10.5 Tensor Train (TT) Decomposition / Matrix Product States

Decompose a rank-$r$ tensor as a chain of 3-index cores:

$$T_{i_1 i_2 \ldots i_r} = G^{(1)}_{i_1 \alpha_1} G^{(2)}_{\alpha_1 i_2 \alpha_2} G^{(3)}_{\alpha_2 i_3 \alpha_3} \cdots G^{(r)}_{\alpha_{r-1} i_r}$$

- $\alpha_k$: **bond indices** (contracted between adjacent cores)
- $i_k$: **physical indices** (free; correspond to original tensor dimensions)
- Bond dimension $\chi = \max(\text{range of } \alpha_k)$: controls approximation quality

**Compression ratio:** An order-$r$ tensor of size $n$ stores $n^r$ values. TT format stores $O(r \, n \, \chi^2)$ parameters — exponential compression when $\chi$ is small.

**AI connections:**
- Tensor train for embedding tables: compress $|V| \times d$ embedding into chain of small matrices
- Matrix product states for language modelling (theoretical connection to autoregressive models)
- Quantum-inspired ML: DMRG-like training of tensor network classifiers

```
TENSOR DECOMPOSITION OVERVIEW
═══════════════════════════════════════════════════════════════════

  CP Decomposition:
    T_ijk = Σ_r λ_r · a_ir · b_jr · c_kr
    → Sum of rank-1 outer products
    → Parameters: R(n₁ + n₂ + n₃)

  Tucker Decomposition:
    T_ijk = G_abc · U_ia · V_jb · W_kc
    → Core tensor + factor matrices
    → Parameters: r₁r₂r₃ + n₁r₁ + n₂r₂ + n₃r₃

  Tensor Train:
    T_{i₁i₂...iₙ} = G¹_{i₁α₁} · G²_{α₁i₂α₂} · ... · Gⁿ_{αₙ₋₁iₙ}
    → Chain of 3-index cores
    → Parameters: O(n·d·χ²) for order-d, mode-n, bond-χ

  SVD (matrix, rank-2):
    A_ij = U_ik Σ_k V_jk
    → Special case of all three decompositions

  LoRA (practical):
    ΔW_ij = B_ir A_rj    (r = LoRA rank, typically 4-64)
    → Rank-r matrix update; r is the dummy index
```

---

## 11. Index Notation for Common AI Architectures

### 11.1 Fully Connected Layer

**Forward pass:**

$$h_i = \sigma(z_i) \quad \text{where} \quad z_i = W_{ij} x_j + b_i$$

- $j$ is dummy: summed over input dimension
- $i$ is free: indexes output neurons
- $\sigma$ is activation function (ReLU, GELU, etc.) applied elementwise

**Gradient w.r.t. weights:**

$$\frac{\partial L}{\partial W_{ij}} = \frac{\partial L}{\partial h_i} \sigma'(z_i) x_j = \delta_i x_j$$

where $\delta_i = \frac{\partial L}{\partial h_i} \sigma'(z_i)$ is the error signal at neuron $i$.

In matrix form: $\frac{\partial L}{\partial W} = \delta x^\top$ — an outer product of error and input.

**Gradient w.r.t. input:**

$$\frac{\partial L}{\partial x_j} = \delta_i W_{ij} = (W^\top \delta)_j$$

The index $i$ is dummy (summed over output neurons). This propagates the error backward through the layer.

### 11.2 Embedding Lookup

$E \in \mathbb{R}^{|V| \times d}$: embedding matrix; token $t \in \{1, \ldots, |V|\}$: discrete index.

**Forward:**

$$e_i = E_{ti}$$

Select row $t$ from $E$. Index $i$ is free (embedding dimension); $t$ is a fixed value, not a summation index.

**Gradient:**

$$\frac{\partial L}{\partial E_{vi}} = \frac{\partial L}{\partial e_i} \cdot \mathbb{1}[v = t] = \delta_{vt} \frac{\partial L}{\partial e_i}$$

Only row $t$ has non-zero gradient — the Kronecker delta $\delta_{vt}$ enforces sparsity. This is why embedding updates use sparse gradient operations rather than dense matrix updates.

### 11.3 Layer Normalisation

Input: $x_i$ (index $i$ over feature dimension $d$).

**Statistics (computed over features):**

$$\mu = \frac{1}{d} \sum_i x_i = \frac{1}{d} x_i \quad \text{(i is dummy)} \qquad \sigma^2 = \frac{1}{d} \sum_i (x_i - \mu)^2$$

**Normalisation:** $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}$

**Output:** $y_i = \gamma_i \hat{x}_i + \beta_i$ (elementwise; $\gamma, \beta$ are learnable scale and shift)

**Gradient complexity:** $\frac{\partial L}{\partial x_i}$ involves sums over $j$ of gradient terms because $\mu$ and $\sigma^2$ depend on **all** $x_j$. The gradient is non-local — changing any single $x_j$ affects the normalisation of every $x_i$. In index notation:

$$\frac{\partial L}{\partial x_i} = \frac{\gamma_i}{\sqrt{\sigma^2 + \varepsilon}} \left[ \frac{\partial L}{\partial y_i} - \frac{1}{d}\sum_j \frac{\partial L}{\partial y_j}\gamma_j \hat{x}_j \cdot \hat{x}_i - \frac{1}{d}\sum_j \frac{\partial L}{\partial y_j}\gamma_j \right]$$

The sums over $j$ (dummy index) make the non-locality explicit.

### 11.4 Causal Self-Attention — Complete Index Derivation

This is the central computation of the transformer. Every step in index notation:

**Input:** $X \in \mathbb{R}^{n \times d}$ (sequence of $n$ tokens, each $d$-dimensional)

**Step 1: Projections**
$$Q_{ia} = X_{ib} W^Q_{ba} \qquad K_{ia} = X_{ib} W^K_{ba} \qquad V_{ia} = X_{ib} W^V_{ba}$$

Index $b$ is dummy (summed over model dimension $d$). Indices $i$ (position) and $a$ (head dimension $d_k$) are free.

**Step 2: Raw scores**
$$S_{ij} = \frac{Q_{ia} K_{ja}}{\sqrt{d_k}}$$

Index $a$ is dummy (dot product over head dimension). Indices $i$ (query position) and $j$ (key position) are free. This is $QK^\top / \sqrt{d_k}$.

**Step 3: Causal mask**
$$\tilde{S}_{ij} = S_{ij} - \infty \cdot \mathbb{1}[j > i]$$

Mask out future positions. Both $i, j$ remain free.

**Step 4: Attention weights**
$$\alpha_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_l \exp(\tilde{S}_{il})}$$

Index $l$ is dummy in the denominator (sum over key positions). Softmax is applied per query position $i$.

**Step 5: Attention output**
$$O_{ia} = \alpha_{ij} V_{ja}$$

Index $j$ is dummy (weighted sum over value positions). Indices $i$ (position) and $a$ (head dimension) are free.

**Step 6: Output projection**
$$Y_{ia} = O_{ib} W^O_{ba}$$

Index $b$ is dummy. This projects the concatenated head outputs back to model dimension.

### 11.5 Feed-Forward Network (SwiGLU)

The modern transformer FFN uses gated linear units:

**Input:** $x_i$ (index $i$ over model dimension $d$)

**Gate projection:** $g_j = W^g_{ji} x_i$ ($j$ over FFN dimension $d_{ff}$; $i$ is dummy)

**Up projection:** $u_j = W^u_{ji} x_i$ ($j$ over $d_{ff}$; $i$ is dummy)

**SwiGLU activation:** $h_j = \text{Swish}(g_j) \cdot u_j$ (elementwise; $j$ is free; no summation)

**Down projection:** $y_i = W^d_{ij} h_j$ ($j$ is dummy; $i$ is free)

**Index flow:** $d$-dim input → $d_{ff}$-dim hidden (via two parallel projections) → $d$-dim output. The index dimensions trace the information flow explicitly.

### 11.6 Gradient of Attention Output

Given upstream gradient $\frac{\partial L}{\partial O_{ia}}$, derive gradients w.r.t. $V$, $\alpha$, and $S$:

**Gradient w.r.t. $V$:**

$$\frac{\partial L}{\partial V_{ja}} = \sum_i \frac{\partial L}{\partial O_{ia}} \alpha_{ij} = \alpha_{ij} \frac{\partial L}{\partial O_{ia}}$$

In matrix form: $\frac{\partial L}{\partial V} = \alpha^\top \frac{\partial L}{\partial O}$ ✓ ($i$ is dummy; $j, a$ are free)

**Gradient w.r.t. $\alpha$:**

$$\frac{\partial L}{\partial \alpha_{ij}} = \sum_a \frac{\partial L}{\partial O_{ia}} V_{ja} = \left(\frac{\partial L}{\partial O} \cdot V^\top\right)_{ij}$$

**Gradient w.r.t. $S$ (through softmax):**

$$\frac{\partial L}{\partial S_{ij}} = \alpha_{ij}\left[\frac{\partial L}{\partial \alpha_{ij}} - \sum_l \alpha_{il} \frac{\partial L}{\partial \alpha_{il}}\right]$$

This is the softmax Jacobian from Section 7.7 applied to each row $i$. These are exactly the formulas implemented in FlashAttention's backward pass.

---

## 12. Symmetries, Invariances, and Equivariances

### 12.1 Permutation Invariance in Index Notation

A function $f$ is **permutation-invariant** if $f(\{x_i\}) = f(\{x_{\pi(i)}\})$ for any permutation $\pi$. In index form: the output depends only on the **set** $\{x_j\}$, not on their order.

**Example — sum pooling:** $s_a = \sum_i X_{ia} = X_{ia}$ ($i$ is dummy)

The sum doesn't change if we permute the rows of $X$. Any reduction over the position index $i$ that doesn't depend on position yields a permutation-invariant result.

### 12.2 Equivariance via Index Structure

A function $f: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ is **permutation-equivariant** if permuting inputs permutes outputs the same way:

$$f(PX)_{ia} = (Pf(X))_{ia}$$

where $P$ is a permutation matrix ($P_{ij} = \delta_{i\pi(j)}$).

**Attention without positional encoding is permutation-equivariant.** Proof via index notation:

Permute input: $X'_{ia} = P_{ib} X_{ba}$ (apply permutation to positions).

Compute $Q'_{ia} = X'_{ib} W^Q_{ba} = P_{ic} X_{cb} W^Q_{ba} = P_{ic} Q_{ca}$

Similarly: $K'_{ia} = P_{ic} K_{ca}$, $V'_{ia} = P_{ic} V_{ca}$

Scores: $S'_{ij} = Q'_{ia} K'_{ja} = P_{ic}Q_{ca} P_{jd}K_{da} = P_{ic}P_{jd}S_{cd}$

After softmax (preserves permutation structure): $\alpha'_{ij} = P_{ic}P_{jd}\alpha_{cd}$

Output: $O'_{ia} = \alpha'_{ij}V'_{ja} = P_{ic}P_{jd}\alpha_{cd} P_{je}V_{ea}$

Since $P_{jd}P_{je} = \delta_{de}$ (orthogonality of permutation): $O'_{ia} = P_{ic}\alpha_{cd}V_{da} = P_{ic}O_{ca}$

Therefore: $O' = PO$ — the output is permuted the same way as the input ✓

**With positional encoding:** The encoding $\text{PE}_{ia}$ breaks equivariance because it depends on position $i$ directly. Different permutations map positions to different encodings.

### 12.3 Invariant and Equivariant Aggregation

| Aggregation | Index Form | Property | Use in AI |
|---|---|---|---|
| Sum pooling | $s_a = X_{ia}$ ($i$ dummy) | Permutation-invariant | Set functions, DeepSets |
| Mean pooling | $s_a = \frac{1}{n}X_{ia}$ | Permutation-invariant | Sentence embeddings |
| Max pooling | $s_a = \max_i X_{ia}$ | Permutation-invariant | PointNet |
| Attention | $O_{ia} = \alpha_{ij}V_{ja}$ | Permutation-equivariant | Set Transformer |

The presence or absence of the position index $i$ in the output determines whether the operation is invariant (no $i$) or equivariant (keeps $i$).

### 12.4 Group Equivariance in Index Notation

For a group $G$ acting on inputs via representation $\rho$:

$$(ρ(g)X)_{ia} = X_{o(g,i),a}$$

where $o(g,i)$ applies group element $g$ to position $i$.

A function $f$ is $G$-equivariant if:

$$f(\rho(g)X)_{ia} = \rho(g)f(X)_{ia} = f(X)_{o(g,i),a}$$

**Constraint on weights:** A linear map $W_{ij}$ is $G$-equivariant iff:

$$W_{ij} = W_{o(g,i),\,o(g,j)} \quad \forall g \in G$$

This constrains $W$ to a specific **shared-weight structure**:

| Group $G$ | Constraint | Result |
|---|---|---|
| Cyclic (translations) | $W_{ij} = W_{i+1,j+1}$ | Circular convolution |
| Full symmetric ($S_n$) | $W_{ij}$ same for all $i \neq j$; $W_{ii}$ same for all $i$ | Sum/mean pooling |
| Rotation ($SO(3)$) | $W$ commutes with rotation matrices | Spherical harmonics convolution |

Index notation makes these symmetry constraints explicit and verifiable — you can check equivariance by substituting the group action into the index expression.

---

## 13. Advanced Index Manipulations

### 13.1 Diagrammatic Notation (Penrose / Tensor Networks)

Tensor network diagrams represent tensors and their contractions visually:

- Each tensor is a **node** (box or circle) with one **wire** (line) per index
- **Contraction:** connect wires between tensors (shared index = connected wire)
- **Free indices:** dangling wires (not connected to another tensor)
- **Trace:** loop a wire back to the same tensor

```
TENSOR NETWORK DIAGRAMS
═══════════════════════════════════════════════════════════════════

  Vector v_i:          Matrix M_ij:         Rank-3 T_ijk:
      │ i                  │ i  │ j              │ i  │ j
      ●                    ┌─┐                   ┌─┐
                           │M│                   │T│──── k
                           └─┘                   └─┘

  Matrix multiply C_ij = A_ik B_kj:
      │ i           │ j
      ┌─┐     k     ┌─┐           k is contracted (internal wire)
      │A│───────────│B│           i, j are free (dangling wires)
      └─┘           └─┘

  Trace: tr(A) = A_ii:
      ┌──────┐
      │  ┌─┐ │                    Wire loops back → self-contraction
      └──│A│─┘                    No dangling wires → scalar
         └─┘

  Frobenius: A_ij B_ij:
      ┌─┐  i,j   ┌─┐             Both wires connected
      │A│════════│B│             → all contracted → scalar
      └─┘        └─┘
```

**AI applications of tensor networks:**
- Matrix Product States (MPS) for language modelling: autoregressive probability as tensor train
- MERA (Multi-scale Entanglement Renormalisation Ansatz): hierarchical tensor decomposition
- Tensor network contraction for efficient inference in graphical models

### 13.2 Differential Forms and the Wedge Product

Differential $k$-forms are completely antisymmetric rank-$k$ tensors $\omega_{i_1 \ldots i_k}$.

**Wedge product:** $(α \wedge β)_{i_1 \ldots i_k j_1 \ldots j_l}$ = antisymmetrisation of $α_{i_1 \ldots i_k} β_{j_1 \ldots j_l}$

**Exterior derivative:** $d\omega$ of a $k$-form is a $(k+1)$-form:

$$(d\omega)_{i_0 i_1 \ldots i_k} = \sum_{a=0}^{k} (-1)^a \partial_{i_a} \omega_{i_0 \ldots \hat{i}_a \ldots i_k}$$

where $\hat{i}_a$ means "omit index $i_a$".

**Connection to ML:**
- Normalising flows require computing $\log|\det J|$ where $J$ is the Jacobian. The determinant is expressible via the top exterior power: $\det(J) = J_{1i_1} J_{2i_2} \cdots J_{ni_n} \varepsilon_{i_1 i_2 \ldots i_n}$
- Riemannian geometry of neural network loss surfaces uses differential forms for curvature tensors
- The Jacobian log-determinant in variational inference is a wedge product computation

### 13.3 Tensor Symmetrisation Operators

**Symmetriser** $S$ and **antisymmetriser** $A$ as projection operators in index space:

$$S^{kl}_{ij} = \frac{1}{2}(\delta^k_i \delta^l_j + \delta^l_i \delta^k_j)$$

$$A^{kl}_{ij} = \frac{1}{2}(\delta^k_i \delta^l_j - \delta^l_i \delta^k_j)$$

Applied to a rank-2 tensor $T$:

- $(ST)_{ij} = S^{kl}_{ij} T_{kl} = \frac{1}{2}(T_{ij} + T_{ji})$ — symmetric part
- $(AT)_{ij} = A^{kl}_{ij} T_{kl} = \frac{1}{2}(T_{ij} - T_{ji})$ — antisymmetric part

**Projection properties:**
- $S^2 = S$ and $A^2 = A$ (idempotent — projecting twice = projecting once)
- $S + A = I$ (every tensor = symmetric part + antisymmetric part)
- $SA = AS = 0$ (orthogonal projections)

### 13.4 Spectral Methods in Index Notation

**Eigendecomposition** (symmetric $A$):

$$A_{ij} = \lambda_k v_{ik} v_{jk}$$

Here $k$ is dummy — summed over eigenvalues. $v_{ik}$ is the $i$-th component of the $k$-th eigenvector. In matrix form: $A = V \Lambda V^\top$.

**SVD:**

$$A_{ij} = U_{ik} \Sigma_k V_{jk}$$

$k$ ranges over rank. $\Sigma_k$ is implicitly $\Sigma_{kl} = \sigma_k \delta_{kl}$ (diagonal), which simplifies to just $\Sigma_k$ when contracted.

**Low-rank approximation:** truncate the sum to $k \leq r$ → rank-$r$ approximation.

**LoRA in index notation:**

$$\Delta W_{ij} = B_{ik} A_{kj} \quad (k \text{ ranges over rank } r)$$

The index $k$ is the bottleneck. Parameters: $(d_1 + d_2) \times r$ instead of $d_1 \times d_2$. For typical values ($d_1 = d_2 = 4096$, $r = 16$): compression ratio = $4096^2 / (2 \times 4096 \times 16) = 128\times$.

### 13.5 The Score Function and Stein's Lemma

**Score function:** $s_i(x) = \frac{\partial}{\partial x_i} \log p(x) = \partial_i \log p(x)$

**Stein's identity** (integration by parts):

$$\mathbb{E}_p[f(x) s_i(x)] = -\mathbb{E}_p\left[\frac{\partial f}{\partial x_i}\right]$$

In index notation with tensor-valued $f$:

$$\mathbb{E}_p[f_a(x) s_i(x)] = -\mathbb{E}_p[\partial_i f_a(x)]$$

**AI applications:**
- **Score-based generative models** (diffusion models): learn $s_i(x, t) \approx \nabla_{x_i} \log p_t(x)$ directly; sample via Langevin dynamics $x_{t+1,i} = x_{t,i} + \eta s_i(x_t) + \sqrt{2\eta}\,\xi_i$
- **REINFORCE (policy gradient):** $\nabla_\theta J = \mathbb{E}_\pi[R \cdot \nabla_\theta \log \pi(a|s)]$ is a score function estimator; in index form: $\partial_\theta J = \mathbb{E}[R \cdot s_\theta]$
- **Stein Variational Gradient Descent (SVGD):** uses the score function with a kernel to approximate posterior inference

---

## 14. Common Mistakes

### 14.1 Mistake Table

| # | Mistake | Why Wrong | Correct |
|---|---|---|---|
| 1 | $A_{ij}B_{ij} = (\sum_i A_{ij})(\sum_j B_{ij})$ | Sums over BOTH $i$ and $j$ simultaneously; each pair $(i,j)$ contributes $A_{ij}B_{ij}$ | $A_{ij}B_{ij} = \sum_i\sum_j A_{ij}B_{ij}$ = Frobenius inner product |
| 2 | $A_{ij}B_{ji} = A_{ij}B_{ij}$ | $B_{ji}$ is the **transpose** of $B_{ij}$; different unless $B$ is symmetric | $A_{ij}B_{ji} = \text{tr}(AB)$ while $A_{ij}B_{ij} = \text{tr}(A^\top B)$ |
| 3 | Free index mismatch: $v_i = A_{jk}u_k$ | Left has free $i$; right has free $j$; **inconsistent** | Free indices must match both sides |
| 4 | Triple index: $A_{iii}B_i$ | Index $i$ appears **three** times — ambiguous | Write explicit $\Sigma$; Einstein requires exactly two |
| 5 | $\delta_{ij}$ means "set $i = j$ everywhere" | $\delta_{ij}$ acts as index substitutor **when contracted**; it doesn't globally equate $i$ and $j$ | $A_{ki}\delta_{ij} = A_{kj}$ — $i$ is replaced by $j$ in $A$; $\delta$ disappears |
| 6 | `"ij,jk->ij"` is matrix multiply | This keeps both $i$ and $j$ in output; $j$ is NOT contracted | Matrix multiply is `"ij,jk->ik"` |
| 7 | Outer product has repeated index | $a_i b_j$ has DIFFERENT index names; no repetition = no contraction | Different indices = free = outer product |
| 8 | Tensor rank = matrix rank | **Tensor rank** (number of indices) ≠ **matrix rank** (dimension of column space) | Completely different concepts; "rank" is overloaded |
| 9 | Renaming free index changes expression | Renaming consistently on BOTH sides is valid (alpha-equivalence) | $v_i = A_{ij}u_j$ and $v_k = A_{kj}u_j$ are identical |
| 10 | Einstein notation only works for two tensors | Multi-tensor contractions follow same rules | $A_{ij}B_{jk}C_{kl} \to D_{il}$ contracts two pairs |

### 14.2 The Einsum Debugging Checklist

When an einsum expression gives unexpected results, check:

1. **Index count:** Does each index appear the correct number of times? (Exactly once = free; exactly twice = contracted)
2. **Shape compatibility:** Do contracted indices have matching dimensions?
3. **Output indices:** Are the output indices exactly what you want? (Missing one = extra contraction; extra one = missing contraction)
4. **Index collision:** Did you accidentally reuse an index name? (`"ij,ik->jk"` contracts over $i$; is that intended?)
5. **Batch dimension:** Does the batch index appear in all the right places?
6. **Contraction vs elementwise:** `"ij,ij->"` vs `"ij,ij->ij"` — contracted vs elementwise

### 14.3 Index Balance as Type Checking

Think of index balance like type checking in a programming language:

```
TYPE CHECKING ANALOGY
═══════════════════════════════════════════════════════════════════

  Expression              Free indices    "Type"
  ────────────────────    ─────────────   ──────────────
  A_ij B_kj               i, k           Matrix (2 free)
  v_i                     i              Vector (1 free)
  A_ii                    (none)         Scalar (0 free)

  Type-correct equation:
    v_i = A_ij u_j        ← both sides: 1 free index (i)  ✓

  Type error:
    v_i = A_jk u_k        ← left: free i; right: free j   ✗
    s = A_ij B_jk          ← left: 0 free; right: i,k     ✗

  → If free indices don't match, the equation is WRONG.
  → This is a mechanical check — no mathematical understanding needed.
```

---

## 15. Exercises

### Exercise 1: Index Identification

For each expression below, identify all **free indices**, all **dummy indices**, the **resulting shape**, and write the equivalent **`np.einsum`** string.

| # | Expression | Free | Dummy | Shape | Einsum |
|---|-----------|------|-------|-------|--------|
| 1 | $c_i = A_{ij} b_j$ | | | | |
| 2 | $C_{ik} = A_{ij} B_{jk}$ | | | | |
| 3 | $s = v_i v_i$ | | | | |
| 4 | $T_{ij} = u_i v_j$ | | | | |
| 5 | $D_{il} = A_{ij} B_{jk} C_{kl}$ | | | | |

<details>
<summary><strong>Solution</strong></summary>

| # | Free | Dummy | Shape | Einsum |
|---|------|-------|-------|--------|
| 1 | $i$ | $j$ | $(n,)$ vector | `ij,j->i` |
| 2 | $i, k$ | $j$ | $(n, p)$ matrix | `ij,jk->ik` |
| 3 | (none) | $i$ | scalar | `i,i->` |
| 4 | $i, j$ | (none) | $(n, m)$ matrix | `i,j->ij` |
| 5 | $i, l$ | $j, k$ | $(n, q)$ matrix | `ij,jk,kl->il` |

</details>

---

### Exercise 2: Kronecker Delta Manipulation

Simplify each expression using the **substitution property** $\delta_{ij} x_j = x_i$:

1. $\delta_{ij} \delta_{jk} = \;?$
2. $\delta_{ij} A_{jk} \delta_{kl} = \;?$
3. $\delta_{ii}$ where $i$ ranges over $1, \dots, n$ $= \;?$
4. $\varepsilon_{ijk} \delta_{jk} = \;?$

<details>
<summary><strong>Solution</strong></summary>

1. $\delta_{ij} \delta_{jk} = \delta_{ik}$

   Contract over $j$: the first delta forces $j = i$, substituting gives $\delta_{ik}$.

2. $\delta_{ij} A_{jk} \delta_{kl} = A_{il}$

   First delta: $j \to i$, giving $A_{ik} \delta_{kl}$. Second delta: $k \to l$, giving $A_{il}$.

3. $\delta_{ii} = n$

   This is the trace of the identity matrix: $\sum_{i=1}^{n} \delta_{ii} = \sum_{i=1}^{n} 1 = n$.

4. $\varepsilon_{ijk} \delta_{jk} = 0$

   $\delta_{jk}$ is symmetric in $(j,k)$; $\varepsilon_{ijk}$ is antisymmetric in $(j,k)$. Contraction of symmetric with antisymmetric always gives zero.

</details>

---

### Exercise 3: Gradient Derivation

Derive the gradient of each scalar loss with respect to the indicated variable using **index notation**. Show all steps.

**3a.** $L = x_i A_{ij} x_j$ (quadratic form). Find $\frac{\partial L}{\partial x_k}$.

<details>
<summary><strong>Solution</strong></summary>

$$
\frac{\partial L}{\partial x_k} = \frac{\partial}{\partial x_k} \left( x_i A_{ij} x_j \right)
$$

Apply the product rule ($A_{ij}$ is constant):

$$
= \delta_{ik} A_{ij} x_j + x_i A_{ij} \delta_{jk}
= A_{kj} x_j + x_i A_{ik}
$$

In the second term, rename $i \to j$: $x_j A_{jk}$. So:

$$
\frac{\partial L}{\partial x_k} = A_{kj} x_j + A_{jk} x_j = (A_{kj} + A_{jk}) x_j
$$

$$\boxed{\nabla_x L = (A + A^T) x}$$

If $A$ is symmetric ($A = A^T$), this simplifies to $2Ax$.

</details>

**3b.** $L = \| Y - X W \|_F^2$ where $Y_{ij}$, $X_{ik}$, $W_{kj}$ are matrices. Find $\frac{\partial L}{\partial W_{pq}}$.

<details>
<summary><strong>Solution</strong></summary>

Write $L$ in index notation. Let $R_{ij} = Y_{ij} - X_{ik} W_{kj}$ be the residual.

$$
L = R_{ij} R_{ij} = (Y_{ij} - X_{ik} W_{kj})(Y_{ij} - X_{ik} W_{kj})
$$

Differentiate with respect to $W_{pq}$:

$$
\frac{\partial L}{\partial W_{pq}} = 2 R_{ij} \cdot \frac{\partial R_{ij}}{\partial W_{pq}}
$$

Now $\frac{\partial R_{ij}}{\partial W_{pq}} = -X_{ik} \delta_{kp} \delta_{jq} = -X_{ip} \delta_{jq}$.

$$
\frac{\partial L}{\partial W_{pq}} = 2 R_{ij} \cdot (-X_{ip} \delta_{jq}) = -2 X_{ip} R_{iq}
$$

$$\boxed{\frac{\partial L}{\partial W} = -2 X^T (Y - XW)}$$

Setting to zero gives the normal equation $X^T X W = X^T Y$.

</details>

**3c.** $L = -y_j \log(\text{softmax}(z)_j)$ where $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$ and $y$ is one-hot. Find $\frac{\partial L}{\partial z_m}$.

<details>
<summary><strong>Solution</strong></summary>

Let $p_i = \text{softmax}(z)_i$. Then $L = -y_j \log p_j$. By chain rule:

$$
\frac{\partial L}{\partial z_m} = -y_j \frac{1}{p_j} \frac{\partial p_j}{\partial z_m}
$$

The softmax Jacobian is $\frac{\partial p_j}{\partial z_m} = p_j (\delta_{jm} - p_m)$. Substituting:

$$
\frac{\partial L}{\partial z_m} = -y_j \frac{1}{p_j} \cdot p_j (\delta_{jm} - p_m)
= -y_j (\delta_{jm} - p_m)
$$

$$
= -y_m + p_m \sum_j y_j = -y_m + p_m \cdot 1 = p_m - y_m
$$

$$\boxed{\frac{\partial L}{\partial z} = \text{softmax}(z) - y}$$

This elegant result is why cross-entropy + softmax is universally used: the gradient is simply the difference between predictions and targets.

</details>

---

### Exercise 4: Einsum Implementation

Write NumPy `einsum` calls (and verify with standard NumPy operations) for:

**4a.** Batched trace: given $A$ with shape $(B, n, n)$, compute the trace of each matrix in the batch.

```python
# Your solution:
traces = np.einsum('bii->b', A)

# Verification:
assert np.allclose(traces, np.trace(A, axis1=1, axis2=2))
```

**4b.** Bilinear form: given $x$ shape $(n,)$, $M$ shape $(n, n)$, $y$ shape $(n,)$, compute $s = x_i M_{ij} y_j$.

```python
# Your solution:
s = np.einsum('i,ij,j->', x, M, y)

# Verification:
assert np.isclose(s, x @ M @ y)
```

**4c.** Multi-head attention scores: given $Q$ shape $(B, H, T, d_k)$, $K$ shape $(B, H, T, d_k)$, compute $S_{bhij} = Q_{bhid} K_{bhjd}$.

```python
# Your solution:
S = np.einsum('bhid,bhjd->bhij', Q, K)

# Verification:
assert np.allclose(S, Q @ K.transpose(0, 1, 3, 2))
```

---

### Exercise 5: Contraction Order Optimisation

Consider the expression $C_{il} = A_{ij} B_{jk} D_{kl}$ where:
- $A$ is $(10000, 512)$
- $B$ is $(512, 4)$
- $D$ is $(4, 512)$

**5a.** Compute the total FLOPs for left-to-right evaluation: $(A_{ij} B_{jk}) D_{kl}$.

**5b.** Compute the total FLOPs for right-to-left evaluation: $A_{ij} (B_{jk} D_{kl})$.

**5c.** Which order is faster and by how much?

<details>
<summary><strong>Solution</strong></summary>

**5a.** Left-to-right: first compute $T_{ik} = A_{ij} B_{jk}$, shape $(10000, 4)$:
- FLOPs: $10000 \times 4 \times 512 = 20{,}480{,}000$

Then $C_{il} = T_{ik} D_{kl}$, shape $(10000, 512)$:
- FLOPs: $10000 \times 512 \times 4 = 20{,}480{,}000$

Total: $\mathbf{40{,}960{,}000}$ FLOPs.

**5b.** Right-to-left: first compute $E_{jl} = B_{jk} D_{kl}$, shape $(512, 512)$:
- FLOPs: $512 \times 512 \times 4 = 1{,}048{,}576$

Then $C_{il} = A_{ij} E_{jl}$, shape $(10000, 512)$:
- FLOPs: $10000 \times 512 \times 512 = 2{,}621{,}440{,}000$

Total: $\mathbf{2{,}622{,}488{,}576}$ FLOPs.

**5c.** Left-to-right is $\approx 64\times$ faster! This mirrors the LoRA pattern: always multiply the low-rank factors first. The intermediate tensor $T_{ik}$ has shape $(10000, 4)$ — far smaller than $E_{jl}$ at $(512, 512)$.

</details>

---

### Exercise 6: Softmax Gradient Verification

**6a.** Derive the Jacobian $\frac{\partial p_i}{\partial z_j}$ where $p_i = \text{softmax}(z)_i$ using quotient rule in index notation.

<details>
<summary><strong>Solution</strong></summary>

$$
p_i = \frac{e^{z_i}}{S}, \quad S = \sum_k e^{z_k}
$$

$$
\frac{\partial p_i}{\partial z_j} = \frac{\delta_{ij} e^{z_i} \cdot S - e^{z_i} \cdot e^{z_j}}{S^2}
= \frac{e^{z_i}}{S} \left( \delta_{ij} - \frac{e^{z_j}}{S} \right)
= p_i (\delta_{ij} - p_j)
$$

$$\boxed{J_{ij} = p_i (\delta_{ij} - p_j) = \text{diag}(p) - p p^T}$$

</details>

**6b.** Verify numerically with finite differences:

```python
import numpy as np

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

z = np.random.randn(5)
p = softmax(z)

# Analytical Jacobian
J_analytical = np.diag(p) - np.outer(p, p)

# Numerical Jacobian (finite differences)
eps = 1e-7
J_numerical = np.zeros((5, 5))
for j in range(5):
    z_plus = z.copy(); z_plus[j] += eps
    z_minus = z.copy(); z_minus[j] -= eps
    J_numerical[:, j] = (softmax(z_plus) - softmax(z_minus)) / (2 * eps)

print("Max error:", np.max(np.abs(J_analytical - J_numerical)))
# Should be < 1e-6
```

---

### Exercise 7: Symmetry Analysis

For each tensor expression, determine if it is **symmetric**, **antisymmetric**, or **neither** in the indicated indices.

| # | Expression | Indices | Answer |
|---|-----------|---------|--------|
| 1 | $S_{ij} = A_{ik} A_{jk}$ (i.e. $A A^T$) | $(i, j)$ | |
| 2 | $T_{ij} = A_{ik} B_{kj} - A_{jk} B_{ki}$ | $(i, j)$ | |
| 3 | $R_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ (Hessian) | $(i, j)$ | |

<details>
<summary><strong>Solution</strong></summary>

1. **Symmetric.** Swap $i \leftrightarrow j$: $S_{ji} = A_{jk} A_{ik} = A_{ik} A_{jk} = S_{ij}$.
   This is $AA^T$, which is always symmetric.

2. **Antisymmetric.** Swap $i \leftrightarrow j$:
   $T_{ji} = A_{jk} B_{ki} - A_{ik} B_{kj} = -(A_{ik} B_{kj} - A_{jk} B_{ki}) = -T_{ij}$.

3. **Symmetric** (for $C^2$ functions, by Schwarz's theorem).
   $\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$ when $f$ has continuous second partial derivatives.

</details>

---

### Exercise 8: Full Attention Pass

Implement the **complete forward and backward pass** for single-head self-attention using only index notation and `np.einsum`.

Given: input $X$ with shape $(T, d)$, weight matrices $W^Q, W^K, W^V$ each $(d, d_k)$.

**Forward pass** — write each step in index notation and einsum:

```python
import numpy as np

T, d, d_k = 8, 16, 4
X = np.random.randn(T, d)
WQ = np.random.randn(d, d_k)
WK = np.random.randn(d, d_k)
WV = np.random.randn(d, d_k)

# Step 1: Queries, Keys, Values
# Q_ia = X_id WQ_da   →   einsum: 'id,da->ia'
Q = np.einsum('id,da->ia', X, WQ)
K = np.einsum('id,da->ia', X, WK)
V = np.einsum('id,da->ia', X, WV)

# Step 2: Attention scores
# S_ij = Q_ia K_ja / sqrt(d_k)   →   einsum: 'ia,ja->ij'
S = np.einsum('ia,ja->ij', Q, K) / np.sqrt(d_k)

# Step 3: Attention weights (softmax over j for each i)
S_max = S.max(axis=1, keepdims=True)
exp_S = np.exp(S - S_max)
alpha = exp_S / exp_S.sum(axis=1, keepdims=True)  # shape (T, T)

# Step 4: Output
# O_ia = alpha_ij V_ja   →   einsum: 'ij,ja->ia'
O = np.einsum('ij,ja->ia', alpha, V)
```

**Backward pass** — given upstream gradient $\frac{\partial L}{\partial O_{ia}}$:

```python
dO = np.random.randn(T, d_k)  # upstream gradient

# Step 4 backward: dL/d(alpha) and dL/dV
# dL/dV_ja = alpha_ij * dO_ia   →   einsum: 'ij,ia->ja'
dV = np.einsum('ij,ia->ja', alpha, dO)
# dL/d(alpha_ij) = dO_ia * V_ja   →   einsum: 'ia,ja->ij'
dalpha = np.einsum('ia,ja->ij', dO, V)

# Step 3 backward: softmax gradient
# dL/dS_ij = alpha_ij * (dalpha_ij - sum_k alpha_ik dalpha_ik)
sum_term = np.einsum('ij,ij->i', alpha, dalpha)  # shape (T,)
dS = alpha * (dalpha - sum_term[:, None]) / np.sqrt(d_k)

# Step 2 backward: dL/dQ and dL/dK
# dL/dQ_ia = dS_ij K_ja   →   einsum: 'ij,ja->ia'
dQ = np.einsum('ij,ja->ia', dS, K)
# dL/dK_ja = dS_ij Q_ia   →   einsum: 'ij,ia->ja'
dK = np.einsum('ij,ia->ja', dS, Q)

# Step 1 backward: dL/dW^Q, dL/dW^K, dL/dW^V
# dL/dWQ_da = X_id * dQ_ia   →   einsum: 'id,ia->da'
dWQ = np.einsum('id,ia->da', X, dQ)
dWK = np.einsum('id,ia->da', X, dK)
dWV = np.einsum('id,ia->da', X, dV)
```

Verify your implementation with numerical gradient checking (finite differences on each weight matrix).

---

## 16. Why This Matters for AI

### 16.1 Impact Across the AI Stack

Einstein summation notation is not merely academic shorthand - it is the computational lingua franca of modern AI systems. Every layer of the stack, from theoretical research papers to production inference engines, benefits from thinking in indices.

| Domain | Without Index Notation | With Index Notation |
|--------|------------------------|---------------------|
| Research papers | Ambiguous matrix expressions, dimensions unclear | Precise, self-documenting equations |
| Attention mechanisms | Nested loops or opaque `bmm` calls | Single `einsum` string: `bhid,bhjd->bhij` |
| Backpropagation | Memorize gradient formulas | Derive gradients mechanically |
| Tensor compilers | Manual kernel fusion | Automatic optimization from contraction specs |
| LoRA / adapters | Ad-hoc low-rank tricks | Clear decomposition: $W_{ij} \approx A_{ik} B_{kj}$ |
| Multi-head attention | Reshapes and transposes | Natural batch index: $Q_{bhid}$ |
| Distributed training | Confusing sharding specs | Shard along a named index |
| Quantization | Unclear where precision is lost | Per-index scaling: $\hat{W}_{ij} = s_i W_{ij}^{(q)}$ |
| Sparse attention | Masking heuristics | Constraint on index range: $|i-j| < w$ |
| Graph neural networks | Aggregation as a special case | Generalized: $h_i^{(l+1)} = \sigma(A_{ij} h_j^{(l)} W_{kk'})$ |
| Diffusion models | Score-function algebra | Index-level Stein identity derivation |

### 16.2 Key Takeaways

1. Einstein summation is a language, not just notation. Learning it changes how you read papers, write code, and debug models.
2. Free indices determine the output shape; dummy indices determine the computation.
3. `einsum` is the universal API for turning index notation into executable tensor code.
4. Once a forward pass is written in index notation, many backward-pass steps become mechanical through product-rule reasoning and identities such as $\frac{\partial x_i}{\partial x_j} = \delta_{ij}$.
5. Index notation reveals optimization opportunities: contraction order, low-rank structure, sparsity, and parallel axes all become easier to see.
6. Modern architectures are index expressions. Transformers, GNNs, diffusion models, and mixture-of-experts all benefit from this viewpoint.
7. Symmetries often appear as constraints on how indices transform, which makes the notation useful for principled architecture design.

## 17. Conceptual Bridge

### 17.1 From Mathematical Notation to Working Code

```text
paper equation
  -> attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
  -> index notation
  -> O_ia = softmax_j(Q_id K_jd / sqrt(d_k)) V_ja
  -> einsum strings
  -> scores = einsum('bhid,bhjd->bhij', Q, K)
  -> output = einsum('bhij,bhja->bhia', attn, V)
  -> optimized kernels and hardware-specific implementations
```

At every level, index notation tells you what data flows where, which dimensions contract, and what the output shape must be. That is the bridge from paper math to production tensor programs.

### 17.2 Where This Leads Next

The next chapters turn indices into the working language of vectors, matrices, tensors, gradients, and efficient kernels. Once you can read repeated-index structure fluently, large model equations become far less mysterious: they collapse into a small set of contraction patterns you can analyze, implement, and optimize.

## References and Further Reading

| Resource | Topic | Type |
|----------|-------|------|
| Einstein, A. (1916). "The Foundation of the General Theory of Relativity" | Original convention introduction | Paper |
| Kolda & Bader (2009). "Tensor Decompositions and Applications" | CP, Tucker, tensor networks | Survey |
| Vaswani et al. (2017). "Attention Is All You Need" | Transformer architecture | Paper |
| Laue, Mitterreiter, and Giesen (2020). "A Simple and Efficient Tensor Calculus" | Automatic tensor differentiation | Paper |
| `opt_einsum` documentation | Contraction-order optimization | Software |
| Penrose (1971). "Applications of Negative Dimensional Tensors" | Diagrammatic tensor notation | Paper |
| Dao et al. (2022). "FlashAttention" | Memory-efficient attention kernels | Paper |
| Hu et al. (2021). "LoRA: Low-Rank Adaptation" | Low-rank weight decomposition | Paper |
| NumPy documentation - `numpy.einsum` | Reference implementation | Documentation |

---

[← Previous: 04-Summation-and-Product-Notation](../04-Summation-and-Product-Notation/notes.md) | [Home](../../README.md) | [Next: 06-Proof-Techniques →](../06-Proof-Techniques/notes.md)
