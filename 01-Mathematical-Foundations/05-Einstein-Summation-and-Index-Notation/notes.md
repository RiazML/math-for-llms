[← Summation and Product Notation](../04-Summation-and-Product-Notation/notes.md) | [Home](../../README.md) | [Proof Techniques →](../06-Proof-Techniques/notes.md)

---

# Einstein Summation and Index Notation

## Introduction

Einstein summation convention is a **compact notation for tensor operations** where repeated indices imply summation. Instead of writing explicit Σ symbols everywhere, you simply repeat an index and the sum is understood.

```
Explicit Summation                    Einstein Convention
─────────────────────────────────    ─────────────────────────────
  n
 ___
 \
  \   Aᵢₖ Bₖⱼ = Cᵢⱼ                  Aᵢₖ Bₖⱼ = Cᵢⱼ
  /
 /___                                (repeated index k is summed)
 k=1
```

This notation was introduced by Albert Einstein in his 1916 general relativity paper. He later joked it was "his greatest contribution to mathematics." Today it's essential for ML because **every tensor operation in PyTorch and NumPy can be expressed as einsum**.

### Why This Matters for ML

| ML Operation     | Explicit                   | einsum              | Framework Call     |
| ---------------- | -------------------------- | ------------------- | ------------------ |
| Dot product      | $\sum_i a_i b_i$           | `'i,i->'`           | `torch.dot(a,b)`   |
| Matrix multiply  | $\sum_k A_{ik}B_{kj}$      | `'ik,kj->ij'`       | `A @ B`            |
| Batch matmul     | $\sum_k A_{bik}B_{bkj}$    | `'bik,bkj->bij'`    | `torch.bmm(A,B)`   |
| Attention scores | $\sum_d Q_{bhid}K_{bhjd}$  | `'bhid,bhjd->bhij'` | Scaled dot-product |
| Trace            | $\sum_i A_{ii}$            | `'ii->'`            | `torch.trace(A)`   |
| Outer product    | $a_i b_j$                  | `'i,j->ij'`         | `torch.outer(a,b)` |
| Bilinear form    | $\sum_{ij} x_i M_{ij} y_j$ | `'i,ij,j->'`        | `x @ M @ y`        |

---

## Prerequisites

- Summation and product notation (Section 04)
- Basic linear algebra (vectors, matrices)
- Python/NumPy basics

## Learning Objectives

By the end of this section, you will be able to:

1. ✅ Read and write Einstein summation convention
2. ✅ Distinguish free indices from dummy (summed) indices
3. ✅ Translate between Σ-notation, index notation, and `np.einsum`/`torch.einsum`
4. ✅ Express all common tensor operations in einsum form
5. ✅ Apply einsum to real ML operations (attention, convolution, contraction)

---

## Notation Reference

| Symbol            | Meaning                   | Example                                                  |
| ----------------- | ------------------------- | -------------------------------------------------------- |
| Subscript index   | Component of tensor       | $A_{ij}$ = element at row i, col j                       |
| Repeated index    | Summed over (dummy index) | $A_{i\mathbf{k}} B_{\mathbf{k}j}$ → sum over k           |
| Free index        | Appears in output         | $C_{\mathbf{i}\mathbf{j}} = A_{ik}B_{kj}$ → i,j are free |
| Superscript index | Contravariant component   | $v^i$ (used in physics, less common in ML)               |
| $\delta_{ij}$     | Kronecker delta           | 1 if i=j, 0 otherwise                                    |
| $\epsilon_{ijk}$  | Levi-Civita symbol        | Encodes orientation/cross products                       |

---

## Table of Contents

1. [From Σ to Einstein Convention](#1-from-σ-to-einstein-convention)
2. [Index Types: Free vs Dummy](#2-index-types-free-vs-dummy)
3. [Common Tensor Operations](#3-common-tensor-operations)
4. [The einsum Function](#4-the-einsum-function)
5. [Multi-Dimensional Tensors](#5-multi-dimensional-tensors)
6. [AI/ML Applications](#6-aiml-applications)
7. [Kronecker Delta and Levi-Civita](#7-kronecker-delta-and-levi-civita)
8. [Einsum Optimization](#8-einsum-optimization)
9. [Common Pitfalls & Interview Questions](#9-common-pitfalls--interview-questions)
10. [Summary](#10-summary)
11. [Further Reading](#11-further-reading)

---

## 1. From Σ to Einstein Convention

### The Rule

**Einstein summation convention**: When an index appears **exactly twice** in a single term, it is implicitly summed over.

$$\text{Explicit: } c = \sum_{i=1}^{n} a_i b_i \qquad \xrightarrow{\text{Einstein}} \qquad c = a_i b_i$$

The repeated index `i` tells you: "sum over all values of i."

### Step-by-Step Translation

| #   | Explicit Σ Form                   | Einstein Form            | What's Summed                     |
| --- | --------------------------------- | ------------------------ | --------------------------------- |
| 1   | $c = \sum_i a_i b_i$              | $c = a_i b_i$            | i (dot product)                   |
| 2   | $C_{ij} = \sum_k A_{ik} B_{kj}$   | $C_{ij} = A_{ik} B_{kj}$ | k (matrix multiply)               |
| 3   | $\text{tr}(A) = \sum_i A_{ii}$    | $A_{ii}$                 | i (trace)                         |
| 4   | $c_i = \sum_j A_{ij} b_j$         | $c_i = A_{ij} b_j$       | j (matrix-vector)                 |
| 5   | $s = \sum_i \sum_j A_{ij} B_{ij}$ | $s = A_{ij} B_{ij}$      | i and j (Frobenius inner product) |

### Worked Example: Matrix Multiplication

$$C_{ij} = A_{ik} B_{kj}$$

Expand for a 2×2 case:

```
C₁₁ = A₁₁B₁₁ + A₁₂B₂₁     (k goes 1, 2)
C₁₂ = A₁₁B₁₂ + A₁₂B₂₂     (k goes 1, 2)
C₂₁ = A₂₁B₁₁ + A₂₂B₂₁     (k goes 1, 2)
C₂₂ = A₂₁B₁₂ + A₂₂B₂₂     (k goes 1, 2)

Index breakdown:
  i, j = FREE indices (appear once → appear in output C)
  k    = DUMMY index  (appears twice → summed over)
```

---

## 2. Index Types: Free vs Dummy

### Free Indices

**Free indices** appear exactly **once** on each side of an equation. They label the components of the output.

$$C_{\underbrace{i}_{free}\underbrace{j}_{free}} = A_{ik} B_{kj}$$

- i ranges over rows of C
- j ranges over columns of C
- The output has shape determined by free indices

### Dummy (Summation) Indices

**Dummy indices** appear exactly **twice** in a term and are summed over. The name doesn't matter — you can rename them freely.

$$A_{i\mathbf{k}} B_{\mathbf{k}j} = A_{i\mathbf{m}} B_{\mathbf{m}j}$$

Both expressions mean the same thing: $\sum_k A_{ik}B_{kj}$.

### Rules

```
EINSTEIN CONVENTION RULES
═══════════════════════════════════════════════════════════════════════

Rule 1: An index appearing TWICE in a term → summed over (dummy)
Rule 2: An index appearing ONCE → labels output component (free)
Rule 3: An index appearing THREE+ times → INVALID (ambiguous)
Rule 4: Free indices must match on both sides of an equation
Rule 5: Dummy indices can be renamed without changing meaning

VALID:
  cᵢ = Aᵢⱼ bⱼ           ✓  j is dummy (summed), i is free
  C = Aᵢⱼ Bᵢⱼ           ✓  both i,j are dummy → scalar output
  Cᵢⱼ = Aᵢₖ Bₖⱼ         ✓  k is dummy, i,j are free

INVALID:
  cᵢ = Aᵢⱼ bⱼ cⱼ        ✗  j appears THREE times
  Cᵢ = Aᵢⱼ Bₖⱼ          ✗  k is free on right but not on left
```

---

## 3. Common Tensor Operations

### Vectors (Rank-1 Tensors)

| Operation       | Math                | Einstein                   | einsum      |
| --------------- | ------------------- | -------------------------- | ----------- |
| Dot product     | $\sum_i a_i b_i$    | $a_i b_i$                  | `'i,i->'`   |
| Outer product   | $M_{ij} = a_i b_j$  | $a_i b_j$                  | `'i,j->ij'` |
| Element-wise    | $c_i = a_i b_i$     | — (need explicit notation) | `'i,i->i'`  |
| Scalar multiply | $c_i = \lambda a_i$ | $\lambda a_i$              | `',i->i'`   |
| Sum all         | $\sum_i a_i$        | $a_i$ (with scalar output) | `'i->'`     |

### Matrices (Rank-2 Tensors)

| Operation                | Math                      | Einstein       | einsum          |
| ------------------------ | ------------------------- | -------------- | --------------- |
| Matrix multiply          | $\sum_k A_{ik}B_{kj}$     | $A_{ik}B_{kj}$ | `'ik,kj->ij'`   |
| Matrix-vector            | $\sum_j A_{ij}x_j$        | $A_{ij}x_j$    | `'ij,j->i'`     |
| Trace                    | $\sum_i A_{ii}$           | $A_{ii}$       | `'ii->'`        |
| Transpose                | $B_{ji} = A_{ij}$         | — (relabeling) | `'ij->ji'`      |
| Frobenius inner product  | $\sum_{ij} A_{ij}B_{ij}$  | $A_{ij}B_{ij}$ | `'ij,ij->'`     |
| Frobenius norm²          | $\sum_{ij} A_{ij}^2$      | $A_{ij}A_{ij}$ | `'ij,ij->'`     |
| Diagonal                 | $d_i = A_{ii}$            | $A_{ii}$       | `'ii->i'`       |
| Outer product (matrices) | $C_{ijkl} = A_{ij}B_{kl}$ | $A_{ij}B_{kl}$ | `'ij,kl->ijkl'` |

### Diagram: How einsum Determines the Computation

```
EINSUM DECODING
═══════════════════════════════════════════════════════════════════════

  'ik,kj->ij'
   ──  ──  ──
    │   │   │
    │   │   └── Output indices (free): shape of result
    │   └────── Second input indices
    └────────── First input indices

  Repeated index (k): appears in inputs but NOT in output → SUMMED
  Free indices (i, j): appear in output → kept

EXAMPLES:

  'i,i->':     a(i) × b(i), sum over i → scalar (dot product)
  'i,i->i':    a(i) × b(i), keep i    → vector (element-wise)
  'ij,jk->ik': A(i,j) × B(j,k), sum j → matrix (matmul)
  'ij->ji':    just rearrange          → matrix (transpose)
  'bij,bjk->bik': batch matmul, sum j, keep b,i,k
```

---

## 4. The einsum Function

### NumPy Syntax

```python
import numpy as np

# np.einsum('subscript_string', *operands)
# The subscript string uses:
#   - lowercase letters for indices
#   - commas to separate operands
#   - -> to indicate output indices

# Dot product: c = Σᵢ aᵢbᵢ
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.einsum('i,i->', a, b)  # 32

# Matrix multiply: Cᵢⱼ = Σₖ AᵢₖBₖⱼ
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
C = np.einsum('ik,kj->ij', A, B)  # same as A @ B
```

### PyTorch Syntax

```python
import torch

# Same syntax as NumPy
Q = torch.randn(2, 8, 10, 64)  # (batch, heads, seq, d_k)
K = torch.randn(2, 8, 10, 64)

# Attention scores: score_bhij = Σ_d Q_bhid K_bhjd
scores = torch.einsum('bhid,bhjd->bhij', Q, K)
# Shape: (2, 8, 10, 10) — attention matrix per head per batch
```

### Implicit Mode (No `->`)

When you omit `->`, NumPy/torch follow these rules:

1. Output indices = sorted alphabetical free indices
2. Repeated indices are summed

```python
# Explicit:  np.einsum('ij,jk->ik', A, B)
# Implicit:  np.einsum('ij,jk', A, B)  — same result, ik inferred
```

**Recommendation**: Always use explicit `->` for clarity.

---

## 5. Multi-Dimensional Tensors

### Rank/Order of Tensors

| Rank | Name     | Shape Example   | ML Example                           |
| ---- | -------- | --------------- | ------------------------------------ |
| 0    | Scalar   | ()              | Loss value                           |
| 1    | Vector   | (n,)            | Bias, embedding                      |
| 2    | Matrix   | (m, n)          | Weight matrix, attention scores      |
| 3    | 3-tensor | (b, m, n)       | Batch of matrices                    |
| 4    | 4-tensor | (b, h, s, d)    | Multi-head attention Q/K/V           |
| 5    | 5-tensor | (b, c, d, h, w) | Video (batch, channels, depth, h, w) |

### Batch Operations

The real power of einsum appears with batched operations:

```
BATCH OPERATIONS WITH EINSUM
═══════════════════════════════════════════════════════════════════════

Single matmul:   'ik,kj->ij'        A(m,n) × B(n,p) → C(m,p)
Batch matmul:    'bik,bkj->bij'     A(b,m,n) × B(b,n,p) → C(b,m,p)
                  ↑
                  batch index (free, not summed)

Multi-head attention:
  Q: (batch, heads, seq_q, d_k)     'bhid'
  K: (batch, heads, seq_k, d_k)     'bhjd'
  scores = Q Kᵀ / √d_k
  einsum: 'bhid,bhjd->bhij'

  b = batch     (free)
  h = head      (free)
  i = query pos (free)
  j = key pos   (free)
  d = d_k       (SUMMED — this is the dot product dimension)
```

### Contraction

**Contraction** = summing over a pair of indices. Each contraction reduces the total rank by 2.

$$T_{ij} = \sum_k A_{ik} B_{kj} \quad \text{(rank 2+2 → rank 2, contracted over k)}$$

$$s = \sum_{ij} A_{ij} B_{ij} \quad \text{(rank 2+2 → rank 0, contracted over i,j)}$$

---

## 6. AI/ML Applications

### 1. Self-Attention (The Core of Transformers)

The scaled dot-product attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

In index notation with batch and head dimensions:

$$\text{score}_{bhij} = \frac{1}{\sqrt{d_k}} \sum_d Q_{bhid} K_{bhjd}$$

$$\text{output}_{bhid} = \sum_j \alpha_{bhij} V_{bhjd}$$

```python
# Multi-head attention in einsum
scores = torch.einsum('bhid,bhjd->bhij', Q, K) / math.sqrt(d_k)
attn_weights = torch.softmax(scores, dim=-1)
output = torch.einsum('bhij,bhjd->bhid', attn_weights, V)
```

### 2. Convolution as Matrix Operation

A 1D convolution can be expressed:

$$y_i = \sum_k w_k \cdot x_{i+k}$$

In einsum with a proper Toeplitz-like expansion:

```python
# 1D conv via einsum (educational, not efficient)
# For batch of signals: y_bi = Σ_k w_k × x_b(i+k)
# Usually done with torch.conv1d, but einsum shows the math
```

### 3. Bilinear Layers

A bilinear operation: $y_k = \sum_{ij} x_i W_{ijk} z_j$

```python
# Bilinear form: y_k = Σ_ij x_i W_ijk z_j
y = torch.einsum('bi,ijk,bj->bk', x, W, z)
```

### 4. Tensor Decomposition (CP Decomposition)

Approximate a tensor as sum of rank-1 components:

$$T_{ijk} \approx \sum_r a_{ir} b_{jr} c_{kr}$$

```python
# CP decomposition reconstruction
T_approx = torch.einsum('ir,jr,kr->ijk', a, b, c)
```

### 5. Graph Neural Networks

Message passing:

$$h_i^{(l+1)} = \sum_j A_{ij} h_j^{(l)} W^{(l)}$$

```python
# GNN message passing
# A: adjacency (n, n), H: node features (n, d), W: weights (d, d')
H_new = torch.einsum('ij,jd,dk->ik', A, H, W)
```

### 6. Loss Functions in Index Notation

**Cross-entropy** (batch of N samples, C classes):

$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{nc} \log(\hat{p}_{nc})$$

Einstein: $\mathcal{L} = -\frac{1}{N} y_{nc} \log(\hat{p}_{nc})$ (both n,c summed)

**MSE Loss**:

$$\mathcal{L} = \frac{1}{N} \sum_n (y_n - \hat{y}_n)^2$$

---

## 7. Kronecker Delta and Levi-Civita

### Kronecker Delta $\delta_{ij}$

$$\delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

This is the identity matrix in index notation: $I_{ij} = \delta_{ij}$

**Key property**: $\delta_{ij} a_j = a_i$ (selects the i-th component)

```
KRONECKER DELTA — THE INDEX "SELECTOR"
═══════════════════════════════════════════════════════════════════════

δᵢⱼ as a matrix:
┌─────────────┐
│ 1  0  0  0  │
│ 0  1  0  0  │    δᵢⱼ = I (identity matrix)
│ 0  0  1  0  │
│ 0  0  0  1  │
└─────────────┘

Key contractions:
  δᵢⱼ aⱼ = aᵢ           (selects component)
  δᵢⱼ Aⱼₖ = Aᵢₖ         (substitutes index)
  δᵢᵢ = n               (trace of I = dimension)
  δᵢⱼ δⱼₖ = δᵢₖ         (transitivity)

ML usage:
  • One-hot encoding: y_class = δ(class, c) for each c
  • Skip connections: output = x + f(x) = δᵢⱼxⱼ + f(x)ᵢ
  • Identity initialization: W = δᵢⱼ (start as identity)
```

### Levi-Civita Symbol $\epsilon_{ijk}$

$$\epsilon_{ijk} = \begin{cases} +1 & \text{if } (ijk) \text{ is an even permutation of } (123) \\ -1 & \text{if } (ijk) \text{ is an odd permutation of } (123) \\ 0 & \text{if any index repeats} \end{cases}$$

Used for cross products and determinants:

$$(a \times b)_i = \epsilon_{ijk} a_j b_k$$

$$\det(A) = \epsilon_{ijk} A_{1i} A_{2j} A_{3k}$$

Less common in ML, but appears in:

- 3D geometry and rotation (computer vision)
- Physics-informed neural networks
- Equivariant neural networks

---

## 8. Einsum Optimization

### Performance: einsum vs Explicit Ops

```
WHEN TO USE EINSUM
═══════════════════════════════════════════════════════════════════════

USE EINSUM:
  ✓ Complex multi-tensor contractions
  ✓ Batch operations not covered by standard library
  ✓ Prototyping — easy to read and modify
  ✓ When you need to express the math directly
  ✓ Multi-head attention, bilinear forms

DON'T USE EINSUM:
  ✗ Simple matmul (A @ B is faster, better optimized)
  ✗ Element-wise ops (*, +, torch.relu)
  ✗ When a dedicated function exists (torch.bmm, F.linear)
  ✗ Inner loops of hot paths (dedicated ops use cuBLAS)

OPTIMIZATION TIPS:
  • torch.einsum with optimize=True tries contraction ordering
  • opt_einsum package finds optimal contraction paths
  • For repeated patterns, benchmark einsum vs manual
```

### Contraction Order Matters

For three matrices A(m×n), B(n×p), C(p×q):

$$D_{iq} = A_{ij} B_{jk} C_{kq}$$

Order 1: $(AB)C$ → cost $O(mnp + mpq)$
Order 2: $A(BC)$ → cost $O(npq + mnq)$

For `m=1000, n=10, p=1000, q=10`:

- Order 1: 10M + 10M = 20M ops
- Order 2: 100K + 100K = 200K ops → **100× faster!**

```python
# opt_einsum finds the best order automatically
import opt_einsum
path = opt_einsum.contract_path('ij,jk,kq->iq', A, B, C)
result = opt_einsum.contract('ij,jk,kq->iq', A, B, C)
```

---

## 9. Common Pitfalls & Interview Questions

### Common Pitfalls

1. **Index Appears 3+ Times**
   - _Issue_: Writing $A_{ij}B_{jk}C_{jl}$ — j appears 3 times
   - _Fix_: Break into steps: $D_{ik} = A_{ij}B_{jk}$, then $E_{ikl} = D_{ik}C_{kl}$... or reconsider the expression

2. **Free Index Mismatch**
   - _Issue_: $C_i = A_{ij}B_{jk}$ — k is free on right but not on left
   - _Fix_: Either sum over k (→ $C_i = A_{ij}B_{ji}$) or include k in output ($C_{ik}$)

3. **Forgetting Batch Dimensions**
   - _Issue_: Using `'ik,kj->ij'` on batched tensors of shape (B, m, n)
   - _Fix_: Add batch index: `'bik,bkj->bij'`

4. **einsum vs Element-wise Confusion**
   - _Issue_: `'i,i->'` (dot product) vs `'i,i->i'` (element-wise multiply)
   - _Fix_: The `->` output determines what happens: indices in output = kept, indices not in output = summed

5. **Transposing in einsum**
   - _Issue_: Writing `'ij,kj->ik'` and not realizing this is $AB^T$
   - _Fix_: Read it aloud: "sum over j, where j is second index of both → second input is transposed"

### Interview Questions

1. **What does `torch.einsum('bhid,bhjd->bhij', Q, K)` compute?**
   - _Answer_: Batched multi-head dot products between query and key vectors. For each batch (b) and head (h), it computes the dot product (sum over d) between every query position (i) and key position (j). This produces the attention score matrix. It's equivalent to `Q @ K.transpose(-2, -1)` but also handles arbitrary batch dims.

2. **Why is Einstein convention useful for ML?**
   - _Answer_: It provides a single, universal notation for all tensor operations. Instead of remembering separate functions for matmul, bmm, dot, outer product, trace, etc., everything becomes a subscript pattern. It also makes the mathematical structure transparent — you can see exactly which dimensions are contracted.

3. **What's the difference between `'ij,jk->ik'` and `'ij,jk'`?**
   - _Answer_: They compute the same thing (matrix multiplication). The implicit form (without `->`) infers the output indices as the sorted alphabetical free indices. Both sum over j and keep i,k in the output.

4. **How would you implement multi-head attention using only einsum?**
   - _Answer_: Three einsum calls:
     1. `'bhid,bhjd->bhij'` for QK^T scores
     2. softmax on the last dimension
     3. `'bhij,bhjd->bhid'` for weighted sum over values

5. **Why does contraction order matter for einsum?**
   - _Answer_: Different contraction orders have different computational costs. For $A_{ij}B_{jk}C_{kl}$, contracting AB first or BC first can differ by orders of magnitude depending on tensor shapes. Libraries like `opt_einsum` find the optimal path.

---

## Companion Notebooks

| Notebook                           | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| [theory.ipynb](theory.ipynb)       | Interactive examples: index notation, einsum operations, ML applications |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions                                         |

---

## 10. Summary

### Quick Reference Card

```
EINSTEIN SUMMATION CHEAT SHEET
═══════════════════════════════════════════════════════════════════════

RULE: Repeated index → summed.  Single index → free (in output).

VECTORS:
  Dot product:        aᵢbᵢ       'i,i->'
  Outer product:      aᵢbⱼ       'i,j->ij'
  Element-wise:       —           'i,i->i'

MATRICES:
  Matrix multiply:    AᵢₖBₖⱼ     'ik,kj->ij'
  Matrix-vector:      Aᵢⱼxⱼ      'ij,j->i'
  Trace:              Aᵢᵢ        'ii->'
  Transpose:          —           'ij->ji'
  Frobenius:          AᵢⱼBᵢⱼ     'ij,ij->'

BATCHED:
  Batch matmul:       —           'bik,bkj->bij'
  Attention scores:   —           'bhid,bhjd->bhij'
  Attention output:   —           'bhij,bhjd->bhid'

HIGHER-ORDER:
  Bilinear:           xᵢWᵢⱼₖzⱼ   'i,ijk,j->k'
  CP decomp:          aᵢᵣbⱼᵣcₖᵣ   'ir,jr,kr->ijk'
```

### Key Concepts

| Concept                 | Definition                                              |
| ----------------------- | ------------------------------------------------------- |
| **Einstein convention** | Repeated indices are implicitly summed                  |
| **Free index**          | Appears once → labels output component                  |
| **Dummy index**         | Appears twice → summed over (contracted)                |
| **Contraction**         | Summing over a shared index pair                        |
| **Kronecker delta**     | $\delta_{ij}$ = identity matrix in index form           |
| **einsum**              | NumPy/PyTorch function implementing Einstein convention |

---

## 11. Further Reading

### Books

1. **"Mathematics for Machine Learning"** (Deisenroth, Faisal, Ong) — Ch.2: tensors and index notation
2. **"Concrete Mathematics"** (Graham, Knuth, Patashnik) — summation mastery
3. **"An Introduction to Tensors and Group Theory for Physicists"** (Jeevanjee) — deep tensor foundations

### Papers & Resources

- 📄 [opt_einsum: Optimizing Tensor Contractions](https://github.com/dgasmith/opt_einsum)
- 📄 [Einsum Is All You Need](https://rockt.github.io/2018/04/30/einsum) — Tim Rocktäschel's visual guide
- 📄 [The Tensor Algebra Compiler](https://doi.org/10.1145/3133901) — TACO project
- 📄 [Attention Is All You Need (Vaswani 2017)](https://arxiv.org/abs/1706.03762) — einsum in action

---

[← Summation and Product Notation](../04-Summation-and-Product-Notation/notes.md) | [Home](../../README.md) | [Proof Techniques →](../06-Proof-Techniques/notes.md)
