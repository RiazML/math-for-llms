[Previous: Vectors and Spaces](../01-Vectors-and-Spaces/notes.md) | [Home](../../README.md) | [Next: Systems of Equations](../03-Systems-of-Equations/notes.md)

---

# Matrix Operations

> _"If vectors are the language of representation, matrices are the language of computation. A modern neural network is what happens when you apply a very large number of matrix operations, very quickly, to very large collections of vectors."_

## Overview

The previous chapter introduced vectors, subspaces, bases, norms, projections, and the geometry of linear algebra. This chapter turns that geometry into computation. A matrix is not only a rectangular array of numbers. It is the concrete representation of a linear map between finite-dimensional vector spaces, and matrix operations are the computational procedures that make linear algebra operational.

This matters immediately in machine learning. A dense layer is a matrix-vector multiply plus bias. A batch of activations passing through a layer is a matrix-matrix multiply. Self-attention is built from matrix multiplies, elementwise operations, transposes, and softmax. Backpropagation is dominated by transposed matrix multiplies and outer products. Numerical stability, hardware utilization, low-rank adaptation, quantization, and efficient inference are all matrix-operation questions.

This chapter therefore moves in three directions at once:

- the algebra of matrices as mathematical objects
- the numerical algorithms used to compute with them
- the AI interpretation of those operations in modern models

The goal is not only to know the rules of matrix arithmetic, but to understand why those rules explain how transformers run on GPUs, why certain decompositions are preferred in practice, and why low-rank structure appears so often in large-scale deep learning.

## Prerequisites

- Basic algebra
- Familiarity with vectors in `R^n`
- Basic knowledge of linear combinations, basis, rank, and subspaces
- Comfort with summation notation

## Learning Objectives

After completing this chapter, you should be able to:

- Read and manipulate matrix notation fluently
- Distinguish elementwise, algebraic, and decomposition-based matrix operations
- Carry out matrix addition, scalar multiplication, transpose, trace, and matrix multiply correctly
- Reason about shape compatibility and debug dimension errors quickly
- Explain matrix multiplication from row, column, outer-product, and block perspectives
- Understand when inverses exist and when pseudo-inverses are required
- Know when an inverse exists and when to use a pseudo-inverse instead
- Connect matrix operations directly to forward passes, backpropagation, attention, LoRA, quantization, and hardware efficiency
- Interpret matrix conditioning and numerical stability in AI systems
- Recognise which matrix decomposition is appropriate for a given problem and know where to find its full treatment

---

## Table of Contents

- [Matrix Operations](#matrix-operations)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Are Matrix Operations?](#11-what-are-matrix-operations)
    - [1.2 Why Matrix Operations Are Central to AI](#12-why-matrix-operations-are-central-to-ai)
    - [1.3 The Computational Centrality of Matrix Multiply](#13-the-computational-centrality-of-matrix-multiply)
    - [1.4 The Map from Linear Algebra to Computation](#14-the-map-from-linear-algebra-to-computation)
    - [1.5 Taxonomy of Matrix Operations](#15-taxonomy-of-matrix-operations)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Matrix Basics and Notation](#2-matrix-basics-and-notation)
    - [2.1 Definition and Notation](#21-definition-and-notation)
    - [2.2 Special Matrices](#22-special-matrices)
    - [2.3 Matrix Shape and Compatibility](#23-matrix-shape-and-compatibility)
    - [2.4 Memory Layout](#24-memory-layout)
  - [3. Elementwise and Scalar Operations](#3-elementwise-and-scalar-operations)
    - [3.1 Matrix Addition and Subtraction](#31-matrix-addition-and-subtraction)
    - [3.2 Scalar Multiplication](#32-scalar-multiplication)
    - [3.3 Hadamard Elementwise Product](#33-hadamard-elementwise-product)
    - [3.4 Trace](#34-trace)
    - [3.5 Transpose](#35-transpose)
  - [4. Matrix Multiplication](#4-matrix-multiplication)
    - [4.1 Definition](#41-definition)
    - [4.2 Four Interpretations of Matrix Multiply](#42-four-interpretations-of-matrix-multiply)
    - [4.3 Properties of Matrix Multiplication](#43-properties-of-matrix-multiplication)
    - [4.4 Non-Commutativity and Consequences](#44-non-commutativity-and-consequences)
    - [4.5 Computational Complexity](#45-computational-complexity)
    - [4.6 GEMM on GPU: Hardware Perspective](#46-gemm-on-gpu-hardware-perspective)
    - [4.7 Matrix-Vector Multiply GEMV](#47-matrix-vector-multiply-gemv)
  - [5. Special Matrix Products](#5-special-matrix-products)
    - [5.1 Outer Product](#51-outer-product)
    - [5.2 Kronecker Product](#52-kronecker-product)
    - [5.3 Khatri-Rao Product](#53-khatri-rao-product)
    - [5.4 Block Matrix Multiplication](#54-block-matrix-multiplication)
  - [6. Matrix Inverse](#6-matrix-inverse)
    - [6.1 Definition and Existence](#61-definition-and-existence)
    - [6.2 Properties of Inverse](#62-properties-of-inverse)
    - [6.3 Explicit Formula for 2x2 Inverse](#63-explicit-formula-for-2x2-inverse)
    - [6.4 Adjugate Formula General Case](#64-adjugate-formula-general-case)
    - [6.5 Computing Inverses in Practice](#65-computing-inverses-in-practice)
    - [6.6 Numerical Stability and Condition Number](#66-numerical-stability-and-condition-number)
  - [7. Moore-Penrose Pseudo-Inverse](#7-moore-penrose-pseudo-inverse)
    - [7.1 Motivation](#71-motivation)
    - [7.2 Definition via SVD](#72-definition-via-svd)
    - [7.3 Moore-Penrose Conditions](#73-moore-penrose-conditions)
    - [7.4 Least Squares Solution](#74-least-squares-solution)
    - [7.5 Pseudo-Inverse Properties](#75-pseudo-inverse-properties)
  - [8. Matrix Decompositions — Preview](#8-matrix-decompositions--preview)
    - [8.1 Determinant](#81-determinant)
    - [8.2 LU Decomposition](#82-lu-decomposition)
    - [8.3 QR Decomposition](#83-qr-decomposition)
    - [8.4 Eigendecomposition](#84-eigendecomposition)
    - [8.5 Singular Value Decomposition](#85-singular-value-decomposition)
    - [8.6 Cholesky Decomposition](#86-cholesky-decomposition)
  - [9. Common Mistakes](#9-common-mistakes)
  - [10. Exercises](#10-exercises)
  - [11. Why This Matters for AI](#11-why-this-matters-for-ai)
  - [12. Conceptual Bridge](#12-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Are Matrix Operations?

A matrix is a rectangular array of numbers arranged in rows and columns:

$$
A =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}
\in \mathbb{R}^{m \times n}.
$$

But that description is only the surface. The deeper point is that every linear map

$$
T : \mathbb{R}^n \to \mathbb{R}^m
$$

is represented by a unique matrix $A \in \mathbb{R}^{m \times n}$ such that

$$
T(x) = Ax.
$$

So matrix operations are the arithmetic of linear maps:

- matrix addition adds linear transformations
- matrix multiplication composes linear transformations
- identity represents "do nothing"
- inverse represents "undo the transformation"
- transpose reorganizes rows and columns and, in the real inner-product case, corresponds to the adjoint
- decompositions expose hidden structure inside the transformation

If vectors are the states of a linear system, matrices are the machines that act on those states.

```text
Vectors are states.
Matrices are transformations.

x in R^n  --A-->  y in R^m

Matrix operations tell us how transformations
combine, invert, factor, and approximate each other.
```

### 1.2 Why Matrix Operations Are Central to AI

In deep learning, almost every major computation is a matrix operation.

For a dense layer with weights $W \in \mathbb{R}^{m \times n}$, bias $b \in \mathbb{R}^m$, and input $x \in \mathbb{R}^n$:

$$
y = Wx + b.
$$

For a batch of inputs $X \in \mathbb{R}^{B \times n}$:

$$
Y = XW^\top + b.
$$

That is matrix-matrix multiply plus broadcasting.

In scaled dot-product attention (Vaswani et al., 2017):

$$
S = \frac{QK^\top}{\sqrt{d_k}},
\qquad
A = \mathrm{softmax}(S),
\qquad
O = AV,
$$

where $Q, K, V$ are matrices. Two matrix multiplies dominate the work: $QK^\top$ and $AV$.

Backpropagation is also matrix arithmetic. If $Y = XW^\top$, then:

$$
\frac{\partial L}{\partial W} =
\left(\frac{\partial L}{\partial Y}\right)^\top X,
\qquad
\frac{\partial L}{\partial X} =
\frac{\partial L}{\partial Y}W.
$$

Weight updates are elementwise matrix operations:

$$
W \leftarrow W - \eta \nabla_W.
$$

So the forward pass, backward pass, and optimization step are all matrix-operation stories. Once models become large enough, performance becomes a question of how well those operations map to GPU kernels, memory hierarchies, and accelerator hardware.

### 1.3 The Computational Centrality of Matrix Multiply

The single most important primitive in deep learning systems is GEMM: general matrix-matrix multiplication. Vendor libraries such as cuBLAS on NVIDIA GPUs and oneDNN on CPUs exist largely to make GEMM fast and hardware-efficient. Dense neural workloads spend much of their time doing repeated multiply-accumulate operations over large arrays, so hardware is built around this pattern:

- NVIDIA Tensor Cores specialize in tiled matrix multiply
- TPUs organize compute as dense systolic-array linear algebra
- many convolutions are lowered to GEMM-style operations
- attention implementations are optimized around matmul and memory reuse

FlashAttention makes this especially clear. The original FlashAttention paper (Dao et al., 2022) did not change the mathematical definition of attention; it reorganized the computation to reduce IO between high-bandwidth memory and on-chip SRAM by tiling the operation better. FlashAttention-3 (Dao et al., 2024) pushed this further on H100 hardware, reporting FP16 throughput up to about 740 TFLOPs/s, roughly 75% utilization of peak tensor-core performance on that kernel.

This is one of the main systems lessons of modern AI:

- if your model is slow, the issue is often not "too much math"
- it is often "the matrix math is scheduled poorly"
- bad tiling, awkward shapes, poor layout, or excessive memory traffic can waste expensive hardware

```text
Why matrix multiply dominates

model layer
   -> matrix multiply
   -> nonlinearity
   -> matrix multiply
   -> normalization
   -> matrix multiply

Most of the FLOPs live in the multiply stages.
That is why hardware and libraries are optimized around GEMM.
```

### 1.4 The Map from Linear Algebra to Computation

It helps to translate abstract language directly into computational language.

| Abstract concept | Computational object |
| --- | --- |
| Linear map $T : \mathbb{R}^n \to \mathbb{R}^m$ | Matrix $A \in \mathbb{R}^{m \times n}$ |
| Composition $T_2 \circ T_1$ | Matrix product $A_2 A_1$ |
| Identity map | Identity matrix $I$ |
| Inverse map | Matrix inverse $A^{-1}$ |
| Adjoint in the real case | Transpose $A^\top$ |
| Projection onto a subspace | Projection matrix $P$ |
| Change of basis | Similarity transform $B^{-1}AB$ |
| Orthonormal basis | Orthogonal matrix $Q$ with $Q^\top Q = I$ |
| Spectral decomposition | Eigendecomposition $A = V \Lambda V^{-1}$ |
| Best rank-$r$ approximation | Truncated SVD |

This dictionary is the bridge between theory and implementation. It explains why so many later algorithms are specialized ways of multiplying, factoring, or solving with matrices.

### 1.5 Taxonomy of Matrix Operations

The subject is easier to navigate if matrix operations are grouped by what they preserve or compute.

```text
MATRIX OPERATIONS
|
|-- Elementwise operations
|   |-- addition / subtraction
|   |-- scalar multiplication
|   `-- Hadamard product
|
|-- Structure-preserving operations
|   |-- transpose
|   |-- conjugate transpose
|   `-- trace
|
|-- Algebraic operations
|   |-- matrix multiplication
|   |-- matrix-vector multiply
|   `-- outer product
|
|-- Solving and inverting
|   |-- inverse
|   |-- pseudo-inverse
|   `-- linear system solvers
|
|-- Decompositions
|   |-- LU
|   |-- QR
|   |-- eigendecomposition
|   |-- SVD
|   `-- Cholesky
|
`-- Derived quantities
    |-- determinant
    |-- rank
    |-- norms
    `-- condition number
```

These categories overlap, but the split is useful. It separates simple entrywise arithmetic from the deeper operations that reveal structure or solve numerical problems.

### 1.6 Historical Timeline

Matrix operations did not appear all at once. The modern subject is the result of several centuries of algebra, geometry, and numerical computation.

| Period | Development | Why it matters |
| --- | --- | --- |
| Ancient China | Elimination methods in *Nine Chapters on the Mathematical Art* | Proto-Gaussian elimination |
| 17th-18th centuries | Determinants, Cramer's rule, linear systems | Early algebraic treatment of systems |
| 1809 | Gauss formalizes least squares and elimination | Linear systems and approximation become computational sciences |
| 1850s | Cayley and Sylvester develop matrix algebra | Matrices become algebraic objects |
| Late 19th century | Frobenius and spectral ideas | Eigenvalues and structure become central |
| Early 20th century | Gram-Schmidt and orthogonalization | Stable basis construction emerges |
| Mid 20th century | Householder, Golub, Reinsch, Wilkinson | Modern numerical linear algebra is born |
| 1990s | LAPACK | Reference dense linear algebra software stack |
| 2000s onward | cuBLAS and GPU linear algebra | Matrix operations move to accelerators |
| 2017 onward | Transformers and attention | GEMM becomes the dominant workload in AI |
| 2020s | FlashAttention, Tensor Cores, FP8 | Matrix arithmetic is co-designed with hardware |

The historical arc is instructive. Matrix operations began as abstract mathematics, matured into numerical algorithms, and then became the central workload of modern AI hardware.

---

## 2. Matrix Basics and Notation

### 2.1 Definition and Notation

A matrix $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns:

$$
A =
\begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{pmatrix}.
$$

Standard notation:

- $A_{ij}$ or $(A)_{ij}$ is the entry in row $i$, column $j$
- the $i$-th row is written $A_{i \cdot}$
- the $j$-th column is written $A_{\cdot j}$
- each row is an element of $\mathbb{R}^n$
- each column is an element of $\mathbb{R}^m$

That distinction matters. Matrix multiplication can be understood as rows meeting columns through dot products.

```text
Shape and indexing

          columns
        j=1  j=2      j=n
      +--------------------+
i=1   | a11  a12  ... a1n |
i=2   | a21  a22  ... a2n |
 ...  | ...  ...  ... ... |
i=m   | am1  am2  ... amn |
      +--------------------+
           rows
```

Block notation is also important. A matrix can be partitioned into submatrices:

$$
A =
\begin{pmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{pmatrix}.
$$

This is not just notation. Block structure is how large matrix computations are actually implemented on modern hardware.

### 2.2 Special Matrices

Many algorithms simplify once a matrix has special structure.

**Square matrix**  
A matrix with the same number of rows and columns.

**Zero matrix**  
The additive identity; every entry is zero.

**Identity matrix**

$$
I_n =
\begin{pmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{pmatrix}.
$$

It satisfies $AI = IA = A$ when shapes are compatible.

**Diagonal matrix**

$$
D = \mathrm{diag}(d_1, \ldots, d_n).
$$

Multiplying by a diagonal matrix is coordinatewise scaling.

**Upper and lower triangular matrices**  
Upper triangular matrices have zeros below the diagonal; lower triangular matrices have zeros above it.

**Symmetric matrix**

$$
A^\top = A.
$$

Symmetric matrices have real eigenvalues and orthogonal eigenspaces for distinct eigenvalues.

**Skew-symmetric matrix**

$$
A^\top = -A.
$$

Its diagonal entries must be zero.

**Orthogonal matrix**

$$
Q^\top Q = QQ^\top = I.
$$

Its columns form an orthonormal basis, and $Q^{-1} = Q^\top$.

**Positive definite matrix**

$$
x^\top A x > 0 \quad \text{for all } x \neq 0.
$$

All eigenvalues are positive.

**Positive semidefinite matrix**

$$
x^\top A x \geq 0 \quad \text{for all } x.
$$

All eigenvalues are nonnegative.

These classes are not merely labels. They determine which algorithms are valid:

- SPD matrices admit Cholesky decomposition
- orthogonal matrices make inversion cheap
- triangular matrices make solving systems cheap
- symmetric matrices make spectral analysis easier

```text
Some important matrix shapes

Identity           Diagonal           Upper triangular

[1 0 0]            [d1 0  0 ]         [a b c]
[0 1 0]            [0  d2 0 ]         [0 d e]
[0 0 1]            [0  0  d3]         [0 0 f]

Symmetric          Lower triangular

[a b c]            [a 0 0]
[b d e]            [b c 0]
[c e f]            [d e f]
```

### 2.3 Matrix Shape and Compatibility

Shape checking is the first line of defense in matrix reasoning.

Addition requires identical shape:

$$
A, B \in \mathbb{R}^{m \times n}
\quad \Longrightarrow \quad
A + B \in \mathbb{R}^{m \times n}.
$$

Multiplication requires matching inner dimensions:

$$
A \in \mathbb{R}^{m \times p},
\qquad
B \in \mathbb{R}^{p \times n}
\quad \Longrightarrow \quad
AB \in \mathbb{R}^{m \times n}.
$$

The inner dimension $p$ disappears because it is the contracted summation index.

```text
Compatibility rule for multiplication

(m x p)  times  (p x n)  ->  (m x n)

   A               B              AB

inner dimensions must match
outer dimensions survive
```

This sounds elementary, but in practice it is one of the most important debugging tools in machine learning. If a tensor program fails, a shape mismatch is one of the first things to check.

Broadcasting is different from matrix multiplication. In libraries such as NumPy and PyTorch, dimensions of size 1 can be expanded logically to match other shapes. That is a convenience rule for elementwise operations, not a new form of matrix algebra.

### 2.4 Memory Layout

A matrix is conceptually two-dimensional, but in memory it is stored as a linear sequence of numbers. The layout matters for performance.

In **row-major order** (common in C-style conventions), entries of the same row are stored contiguously. For a matrix with shape $(m,n)$, the offset of entry $(i,j)$ is approximately

$$
\text{offset} = i \cdot n + j.
$$

In **column-major order** (common in Fortran and classical BLAS/LAPACK conventions), entries of the same column are stored contiguously:

$$
\text{offset} = i + j \cdot m.
$$

```text
2 x 3 matrix

[a b c]
[d e f]

row-major memory:    a b c d e f
column-major memory: a d b e c f
```

This matters because contiguous memory access is cache-friendly. A mathematically identical algorithm can run much faster or much slower depending on layout and traversal order.

The transpose illustrates the difference between mathematics and storage. Mathematically,

$$
(A^\top)_{ij} = A_{ji}.
$$

But in high-performance libraries, a transpose often does not require moving data immediately. It can be represented by changing shape and stride metadata, so the same underlying memory is interpreted with swapped indexing. In PyTorch, a 2D `.T` operation typically returns a view with transposed strides rather than a full data copy.

This matters in AI because operations such as $QK^\top$ in attention rely heavily on fast transposed access patterns. Optimized systems try to get the mathematical effect of transpose without paying the full physical cost of reshuffling memory.

---

## 3. Elementwise and Scalar Operations

### 3.1 Matrix Addition and Subtraction

If $A, B \in \mathbb{R}^{m \times n}$, then matrix addition and subtraction are defined entrywise:

$$
(A \pm B)_{ij} = A_{ij} \pm B_{ij}.
$$

The rules mirror vector addition:

- commutativity: $A + B = B + A$
- associativity: $(A + B) + C = A + (B + C)$
- additive identity: $A + 0 = A$
- additive inverse: $A + (-A) = 0$

The operation cost is $O(mn)$ because every entry must be touched once. On modern GPUs, matrix addition is usually memory-bandwidth bound rather than compute-bound. There is very little arithmetic per byte moved.

In deep learning, matrix addition appears constantly:

- adding biases
- adding residual connections
- updating weights
- combining branches in computation graphs

### 3.2 Scalar Multiplication

If $\alpha \in \mathbb{R}$ and $A \in \mathbb{R}^{m \times n}$, then

$$
(\alpha A)_{ij} = \alpha A_{ij}.
$$

Every entry is scaled by the same scalar.

The familiar linearity properties hold:

- $\alpha(A + B) = \alpha A + \alpha B$
- $(\alpha + \beta)A = \alpha A + \beta A$
- $\alpha(\beta A) = (\alpha \beta)A$
- $1 \cdot A = A$

Together with addition, scalar multiplication defines affine updates such as

$$
\alpha A + \beta B.
$$

In numerical linear algebra, the operation

$$
Y \leftarrow \alpha X + Y
$$

is so common that BLAS gives it a dedicated name: AXPY. Gradient descent is exactly this pattern:

$$
\theta \leftarrow \theta - \eta g,
$$

where $\eta$ is a scalar learning rate and $g$ is a gradient matrix or tensor.

### 3.3 Hadamard Elementwise Product

The Hadamard product of two same-shaped matrices $A, B \in \mathbb{R}^{m \times n}$ is

$$
(A \odot B)_{ij} = A_{ij} B_{ij}.
$$

This is elementwise multiplication. It is not matrix multiplication.

```text
Hadamard product vs matrix product

same shapes required              inner dimensions required

A (m x n)                         A (m x p)
B (m x n)                         B (p x n)

A odot B  ->  (m x n)             AB      ->  (m x n)

no summation                      sum over shared index
entry by entry                    row-column dot products
```

Properties:

- commutative: $A \odot B = B \odot A$
- associative: $(A \odot B) \odot C = A \odot (B \odot C)$
- distributive over addition: $A \odot (B + C) = A \odot B + A \odot C$

But it does **not** obey the same algebra as matrix multiplication.

The Hadamard product is central in AI because modern networks use gates all over the place:

- LSTM and GRU gates
- dropout masks
- attention masks and padding masks
- SwiGLU and related gated feed-forward activations
- elementwise modulation in conditional models

### 3.4 Trace

For a square matrix $A \in \mathbb{R}^{n \times n}$, the trace is

$$
\mathrm{tr}(A) = \sum_{i=1}^n A_{ii}.
$$

Key properties:

**Linearity**

$$
\mathrm{tr}(\alpha A + \beta B)
=
\alpha \mathrm{tr}(A) + \beta \mathrm{tr}(B).
$$

**Transpose invariance**

$$
\mathrm{tr}(A^\top) = \mathrm{tr}(A).
$$

**Cyclic permutation**

For compatible matrices:

$$
\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB).
$$

This does **not** mean arbitrary reordering is allowed.

**Frobenius inner product**

$$
\mathrm{tr}(A^\top B)
=
\sum_{i,j} A_{ij} B_{ij}
=
\langle A, B \rangle_F.
$$

**Eigenvalue sum**

If $A$ is diagonalizable, then

$$
\mathrm{tr}(A) = \sum_i \lambda_i.
$$

In AI, trace appears in Frobenius norms, covariance identities, kernel formulas, and matrix-calculus derivations.

### 3.5 Transpose

The transpose of $A \in \mathbb{R}^{m \times n}$ is the matrix $A^\top \in \mathbb{R}^{n \times m}$ defined by

$$
(A^\top)_{ij} = A_{ji}.
$$

Rows become columns and columns become rows.

Basic properties:

- $(A^\top)^\top = A$
- $(A + B)^\top = A^\top + B^\top$
- $(\alpha A)^\top = \alpha A^\top$
- $(AB)^\top = B^\top A^\top$

The reverse-order rule is fundamental:

$$
((AB)^\top)_{ij}
=
(AB)_{ji}
=
\sum_k A_{jk} B_{ki}
=
\sum_k (B^\top)_{ik}(A^\top)_{kj}
=
(B^\top A^\top)_{ij}.
$$

So

$$
(AB)^\top = B^\top A^\top.
$$

This is the algebraic reason backpropagation has the structure it does. Reverse-mode differentiation repeatedly moves backward through compositions, and transposes inherit the same reverse-order behavior.

```text
Transpose flips orientation

A:      m rows, n cols
A^T:    n rows, m cols

row of A      -> column of A^T
column of A   -> row of A^T
```

---

## 4. Matrix Multiplication

### 4.1 Definition

If

$$
A \in \mathbb{R}^{m \times p},
\qquad
B \in \mathbb{R}^{p \times n},
$$

then the matrix product $C = AB \in \mathbb{R}^{m \times n}$ is defined by

$$
C_{ij} = \sum_{k=1}^p A_{ik} B_{kj}.
$$

Equivalently, the $(i,j)$ entry is the dot product of row $i$ of $A$ with column $j$ of $B$:

$$
C_{ij} = A_{i \cdot} \cdot B_{\cdot j}.
$$

```text
How one entry is formed

row i of A:      [ a_i1  a_i2  ...  a_ip ]
column j of B:   [ b_1j
                   b_2j
                   ...
                   b_pj ]

dot product -> c_ij
```

This definition is exactly what composition of linear maps requires. If $x$ is first transformed by $B$ and then by $A$, the combined transformation is $(AB)x$.

### 4.2 Four Interpretations of Matrix Multiply

One of the best ways to understand matrix multiply is to see four different but equivalent interpretations.

**Row perspective**

$$
C_{i \cdot} = A_{i \cdot} B.
$$

Each row of $C$ is the corresponding row of $A$ acting on $B$.

**Column perspective**

$$
C_{\cdot j} = A B_{\cdot j}.
$$

Each column of $C$ is $A$ applied to the corresponding column of $B$.

**Outer-product perspective**

Let $a_k = A_{\cdot k}$ be the $k$-th column of $A$ and let $b_k^\top = B_{k \cdot}$ be the $k$-th row of $B$. Then

$$
AB = \sum_{k=1}^p a_k b_k^\top.
$$

This is one of the most important views for low-rank structure and gradients.

**Block perspective**

If compatible blocks are used, block multiplication works exactly like scalar algebra with block products in place of scalar products.

```text
One multiply, four views

rows meet columns
columns get transformed
outer products get accumulated
blocks combine hierarchically
```

### 4.3 Properties of Matrix Multiplication

Matrix multiplication obeys several essential rules.

**Associativity**

$$
(AB)C = A(BC).
$$

**Distributivity**

$$
A(B + C) = AB + AC,
\qquad
(A + B)C = AC + BC.
$$

**Identity**

$$
AI = IA = A.
$$

**Zero behavior**

$$
A0 = 0,
\qquad
0A = 0.
$$

But

$$
AB = 0
\centernot\implies
A = 0 \text{ or } B = 0.
$$

**Transpose reversal**

$$
(AB)^\top = B^\top A^\top.
$$

**Rank inequality**

$$
\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B)).
$$

The rank cannot exceed the bottleneck rank of the factors.

### 4.4 Non-Commutativity and Consequences

In general,

$$
AB \neq BA.
$$

Sometimes one product is defined and the other is not. Even when both are defined and square, they are usually different.

A standard counterexample is

$$
A =
\begin{pmatrix}
0 & 1 \\
0 & 0
\end{pmatrix},
\qquad
B =
\begin{pmatrix}
0 & 0 \\
1 & 0
\end{pmatrix}.
$$

Then

$$
AB =
\begin{pmatrix}
1 & 0 \\
0 & 0
\end{pmatrix},
\qquad
BA =
\begin{pmatrix}
0 & 0 \\
0 & 1
\end{pmatrix}.
$$

This has many consequences:

- the order of layer operations matters
- transpose and inverse formulas reverse order
- gradient derivations must preserve order carefully
- symmetric times symmetric is not automatically symmetric
- changing parenthesization changes computational cost even if the answer is unchanged

For example, in transformer notation:

$$
(XW_Q)(XW_K)^\top
=
XW_QW_K^\top X^\top,
$$

and that identity only works because transpose reverses the order inside the second factor.

### 4.5 Computational Complexity

For

$$
C = AB,
\qquad
A \in \mathbb{R}^{m \times p},
\qquad
B \in \mathbb{R}^{p \times n},
$$

the naive algorithm performs about

$$
2mnp - mn
$$

floating-point operations if multiply-adds are counted separately.

For square matrices, that becomes $O(n^3)$.

Strassen's algorithm reduces the exponent to about

$$
O(n^{2.807}),
$$

and later algorithms reduce it further in theory. But these methods bring larger constants, more complicated memory behavior, and weaker numerical properties. They are mathematically important, but they are not the default engine of deep learning workloads.

Another practical optimization problem is matrix-chain order. Because multiplication is associative, the result does not change when the parenthesization changes, but the cost can change drastically.

### 4.6 GEMM on GPU: Hardware Perspective

On accelerators, matrix multiply is implemented by tiling the problem into fragments that fit into fast on-chip memory and specialized multiply-accumulate units.

At a high level:

1. load a tile of $A$ and a tile of $B$ from HBM
2. stage them into shared memory or registers
3. run many fused multiply-add operations on the tile
4. accumulate partial sums
5. write the output tile back

```text
Tiled GEMM

HBM -> shared memory -> registers -> multiply-accumulate -> write back

Do more arithmetic per byte loaded.
That is the whole performance game.
```

NVIDIA's H100 product specifications list up to 989 TFLOPS tensor-core TF32 performance, 1,979 TFLOPS tensor-core BF16/FP16 performance, 3,958 TFLOPS tensor-core FP8 performance, and 3.35 TB/s HBM bandwidth for the SXM variant. That gap between compute throughput and memory bandwidth is why naive memory movement can destroy performance.

Large dense GEMMs usually become compute-bound if tiled well. Small or awkwardly shaped matrix multiplies often do not.

FlashAttention is a direct application of this principle. The algorithm is faster not because it changes the mathematics of attention, but because it reorganizes the matrix operations and reductions to reduce expensive memory traffic between HBM and SRAM while keeping arithmetic dense and local (Dao et al., 2022; Dao et al., 2024).

### 4.7 Matrix-Vector Multiply GEMV

Matrix-vector multiply

$$
y = Av,
\qquad
A \in \mathbb{R}^{m \times n},
\qquad
v \in \mathbb{R}^n,
$$

is a special case of matrix multiplication where one dimension is 1.

It costs about $2mn$ floating-point operations, but unlike large GEMM it has poor arithmetic intensity. You read the matrix, read the vector, and produce one output vector. There is much less reuse.

This is why GEMV is usually memory-bandwidth bound.

The AI consequence is important:

- batched training prefers GEMM because batching increases reuse
- autoregressive decoding often degenerates into many GEMV-like operations because the batch or active sequence dimension is small
- single-sequence decoding therefore achieves far lower hardware utilization than large-batch training

Batching converts many GEMV workloads back toward GEMM, which is why serving systems aggressively batch requests whenever latency constraints allow it.

---

## 5. Special Matrix Products

### 5.1 Outer Product

If $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$, then the outer product is

$$
uv^\top \in \mathbb{R}^{m \times n},
\qquad
(uv^\top)_{ij} = u_i v_j.
$$

Every column of $uv^\top$ is a scalar multiple of $u$, and every row is a scalar multiple of $v^\top$. Therefore, if neither vector is zero, the matrix has rank 1.

```text
Outer product

u = [u1]          v^T = [v1 v2 v3]
    [u2]
    [u3]

uv^T =

[u1v1 u1v2 u1v3]
[u2v1 u2v2 u2v3]
[u3v1 u3v2 u3v3]
```

Outer products matter because they are the atoms of low-rank matrix structure. Any rank-$r$ matrix can be written as a sum of $r$ outer products.

They also appear directly in AI:

- dense-layer gradient for one sample: $\nabla_W = \delta x^\top$
- LoRA updates as sums of rank-1 terms
- covariance and second-moment estimates as sums of outer products

### 5.2 Kronecker Product

If

$$
A \in \mathbb{R}^{m \times n},
\qquad
B \in \mathbb{R}^{p \times q},
$$

then the Kronecker product is the block matrix

$$
A \otimes B \in \mathbb{R}^{mp \times nq}
$$

defined by

$$
A \otimes B =
\begin{pmatrix}
a_{11} B & a_{12} B & \cdots & a_{1n} B \\
a_{21} B & a_{22} B & \cdots & a_{2n} B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} B & a_{m2} B & \cdots & a_{mn} B
\end{pmatrix}.
$$

Important properties:

$$
(A \otimes B)(C \otimes D) = (AC) \otimes (BD),
$$

$$
(A \otimes B)^\top = A^\top \otimes B^\top,
$$

$$
\mathrm{rank}(A \otimes B) = \mathrm{rank}(A)\mathrm{rank}(B),
$$

$$
\mathrm{tr}(A \otimes B) = \mathrm{tr}(A)\mathrm{tr}(B).
$$

The Kronecker product appears whenever a large linear structure decomposes across independent dimensions. In AI it is especially important in second-order optimization approximations such as K-FAC.

### 5.3 Khatri-Rao Product

The Khatri-Rao product is the columnwise Kronecker product. If

$$
A = [a_1 \,|\, a_2 \,|\, \cdots \,|\, a_n],
\qquad
B = [b_1 \,|\, b_2 \,|\, \cdots \,|\, b_n],
$$

with the same number of columns, then

$$
A \odot_{KR} B
=
[a_1 \otimes b_1 \,|\, a_2 \otimes b_2 \,|\, \cdots \,|\, a_n \otimes b_n].
$$

It is more specialized than the Kronecker product, but it shows up in multilinear algebra, tensor factorization, and certain bilinear model derivations.

### 5.4 Block Matrix Multiplication

Block matrices let us reason about large matrices hierarchically. If compatible blocks are used, multiplication works exactly like scalar algebra except that scalar multiplication is replaced by matrix multiplication.

For example:

$$
\begin{pmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{pmatrix}
\begin{pmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{pmatrix}
=
\begin{pmatrix}
A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\
A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22}
\end{pmatrix}.
$$

This is the correct abstraction for tiled GPU GEMM, distributed matrix multiply, tensor parallelism, and block-structured model design.

---

## 6. Matrix Inverse

### 6.1 Definition and Existence

A square matrix $A \in \mathbb{R}^{n \times n}$ is invertible if there exists a matrix $A^{-1}$ such that

$$
AA^{-1} = A^{-1}A = I.
$$

This means $A^{-1}$ undoes the transformation performed by $A$.

```text
x --A--> y
y --A^{-1}--> x

Applying A and then A^{-1} returns the original vector.
```

For a square matrix $A$, the following are equivalent:

- $A$ is invertible
- $\det(A) \neq 0$
- $\mathrm{rank}(A) = n$
- the columns of $A$ are linearly independent
- the rows of $A$ are linearly independent
- $Ax = 0$ has only the trivial solution
- 0 is not an eigenvalue of $A$

### 6.2 Properties of Inverse

If $A$ is invertible, then the inverse is unique. Other core properties are:

$$
(A^{-1})^{-1} = A,
$$

$$
(AB)^{-1} = B^{-1} A^{-1},
$$

$$
(A^\top)^{-1} = (A^{-1})^\top,
$$

$$
(\alpha A)^{-1} = \frac{1}{\alpha} A^{-1}
\quad \text{for } \alpha \neq 0,
$$

$$
\det(A^{-1}) = \frac{1}{\det(A)}.
$$

Again, inverse reverses order just like transpose.

### 6.3 Explicit Formula for 2x2 Inverse

For

$$
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix},
$$

the inverse exists if and only if

$$
ad - bc \neq 0.
$$

When it exists:

$$
A^{-1}
=
\frac{1}{ad - bc}
\begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix}.
$$

This is worth memorizing for intuition, but it is not how serious matrix inversion is done for large matrices.

### 6.4 Adjugate Formula General Case

For a general square matrix, one classical formula is

$$
A^{-1} = \frac{\mathrm{adj}(A)}{\det(A)},
$$

where $\mathrm{adj}(A)$ is the adjugate matrix obtained by transposing the cofactor matrix.

This formula is theoretically elegant but computationally poor. It explains structure; it is not the preferred practical algorithm.

### 6.5 Computing Inverses in Practice

There are several standard ways to compute with inverses or inverse-like effects.

**Gaussian elimination**

Augment the matrix with the identity:

$$
[A \,|\, I]
$$

and row-reduce until it becomes

$$
[I \,|\, A^{-1}].
$$

This is conceptually direct and leads to $O(n^3)$ cost.

**LU-based methods**

Factor

$$
A = LU
$$

or, with pivoting,

$$
PA = LU.
$$

Then solve the triangular systems needed to reconstruct the inverse column by column.

But in practice, the most important rule is:

> To solve $Ax = b$, do not compute $x = A^{-1}b$ explicitly unless you have a very unusual reason.

Instead, solve the linear system directly using LU, QR, or Cholesky.

Why?

- fewer operations
- better numerical stability
- better library support
- avoids forming an object you usually do not need

### 6.6 Numerical Stability and Condition Number

Some matrices are easy to invert numerically. Others are technically invertible but catastrophically unstable in floating-point arithmetic.

The condition number measures this sensitivity:

$$
\kappa(A) = \|A\| \, \|A^{-1}\|.
$$

In the 2-norm,

$$
\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.
$$

Interpretation:

- $\kappa(A) \approx 1$: well-conditioned
- $\kappa(A) \gg 1$: ill-conditioned

If $A$ is ill-conditioned, small perturbations in data or rounding error can produce large perturbations in the solution.

This matters directly in AI:

- poorly conditioned Hessians slow optimization
- badly scaled feature matrices make regression unstable
- covariance matrices can become numerically troublesome
- second-order methods require careful conditioning

---

## 7. Moore-Penrose Pseudo-Inverse

### 7.1 Motivation

Ordinary inverse is only defined for square, full-rank matrices. But many practical matrices are rectangular or rank-deficient. So we need a generalized inverse.

The Moore-Penrose pseudo-inverse, written $A^+$, extends inverse-like behavior to every matrix $A \in \mathbb{R}^{m \times n}$.

It solves two kinds of problems especially well:

- least-squares solutions when $Ax = b$ has no exact solution
- minimum-norm solutions when there are infinitely many exact solutions

### 7.2 Definition via SVD

Let

$$
A = U \Sigma V^\top
$$

be the SVD of $A$, where the nonzero singular values are $\sigma_1, \ldots, \sigma_r$. Then

$$
A^+ = V \Sigma^+ U^\top,
$$

where

$$
\Sigma^+_{ii}
=
\begin{cases}
1/\sigma_i & \text{if } \sigma_i \neq 0 \\
0 & \text{if } \sigma_i = 0.
\end{cases}
$$

If $A$ is square and invertible, then $A^+ = A^{-1}$.

### 7.3 Moore-Penrose Conditions

The pseudo-inverse is characterized uniquely by the four Penrose conditions:

1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^\top = AA^+$
4. $(A^+A)^\top = A^+A$

Two especially important consequences are:

- $AA^+$ is the orthogonal projector onto $\mathrm{col}(A)$
- $A^+A$ is the orthogonal projector onto $\mathrm{row}(A)$

### 7.4 Least Squares Solution

Suppose $A \in \mathbb{R}^{m \times n}$ and $b \in \mathbb{R}^m$.

If $m > n$, the system is overdetermined and there is usually no exact solution. The pseudo-inverse returns the least-squares solution:

$$
x^\star = A^+ b.
$$

If $A$ has full column rank, this becomes

$$
x^\star = (A^\top A)^{-1} A^\top b.
$$

This solves

$$
\min_x \|Ax - b\|_2^2.
$$

Geometrically, $Ax^\star$ is the projection of $b$ onto the column space of $A$.

If $m < n$, there are typically infinitely many exact solutions. Then $A^+b$ returns the minimum-norm solution.

```text
Least squares picture

          b
         /|
        / |
       /  | residual
      /   |
-----*----+----------> col(A)
   A x_star

A x_star is the closest reachable vector to b.
```

### 7.5 Pseudo-Inverse Properties

Important identities:

$$
(A^+)^+ = A,
$$

$$
(A^\top)^+ = (A^+)^\top,
$$

$$
(\alpha A)^+ = \frac{1}{\alpha} A^+
\quad \text{for } \alpha \neq 0,
$$

$$
\mathrm{rank}(A^+) = \mathrm{rank}(A).
$$

In general,

$$
(AB)^+ \neq B^+A^+.
$$

The pseudo-inverse behaves like inverse in some ways, but not all.


---

## 8. Matrix Decompositions — Preview

Matrix decompositions factor $A$ into structured products that expose different kinds of mathematical structure. Each decomposition in this section has a dedicated home elsewhere in the curriculum where it is treated in full — with proofs, algorithms, and worked examples. What follows is a navigational preview: enough context to understand which decomposition applies and why, plus explicit forward links to the full treatment.

> **Decomposition decision table**
>
> | Goal | Use |
> | ---- | --- |
> | Solve $Ax = b$ for general square $A$ | LU (Gaussian elimination) |
> | Solve $Ax = b$ for symmetric positive definite $A$ | Cholesky |
> | Least squares: $\min_x \\|Ax - b\\|$ | QR |
> | Eigenvalues and eigenvectors of square $A$ | Eigendecomposition |
> | Best rank-$k$ approximation, pseudo-inverse, all cases | SVD |

---

### 8.1 Determinant

The determinant $\det(A)$ of a square matrix $A \in \mathbb{R}^{n \times n}$ is a scalar measuring the signed volume-scaling factor of the linear map. When $|\det(A)| > 1$ the map expands volume; when $|\det(A)| < 1$ it compresses; when $\det(A) = 0$ the map is singular — it collapses the full space onto a lower-dimensional subspace, meaning $Ax = b$ has no unique solution.

The determinant is also the product of all eigenvalues and the reciprocal of the product of the eigenvalues of $A^{-1}$. In practice, explicit determinant computation is rarely done by direct expansion (which costs $O(n!)$) — instead it is read off from an LU or Cholesky factorization as a product of diagonal entries.

**For AI:** Log-determinants appear in Gaussian log-likelihoods $\log p(x) \propto -\tfrac{1}{2}\log\det\Sigma - \tfrac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)$, in the ELBO of variational inference, and in normalizing flow log-likelihood as the log absolute Jacobian determinant.

> → _Full treatment: [Determinants](../04-Determinants/notes.md)_

---

### 8.2 LU Decomposition

LU factorization writes $PA = LU$, where $P$ is a permutation (row-reordering) matrix, $L$ is lower triangular with ones on the diagonal, and $U$ is upper triangular. It is the computational form of Gaussian elimination. Once computed in $O(n^3)$, each new right-hand side $b$ can be solved in $O(n^2)$ via forward substitution on $Ly = Pb$ then back substitution on $Ux = y$.

**For AI:** LU appears in second-order optimizers (K-FAC, Shampoo), Kalman filter updates, and continuous normalizing flows that require solving stiff ODEs. It is also how `numpy.linalg.solve` and `scipy.linalg.solve` are implemented internally.

> → _Full treatment: [Matrix Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md)_

---

### 8.3 QR Decomposition

QR factorization writes $A = QR$ where $Q \in \mathbb{R}^{m \times m}$ is orthogonal ($Q^\top Q = I$) and $R \in \mathbb{R}^{m \times n}$ is upper triangular. It is numerically more stable than LU for least-squares problems because $Q$ preserves norms. The main algorithms are Gram–Schmidt orthogonalization (conceptually clear), Householder reflectors (numerically stable, standard in practice), and Givens rotations (good for sparse matrices).

**For AI:** Orthogonal weight initialization (Saxe et al. 2013) uses QR. The QR algorithm is the standard method for computing all eigenvalues of a dense matrix. Weight orthogonalization regularizers (Brock et al.) apply QR repeatedly during training.

> → _Full treatment: [Matrix Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md)_

---

### 8.4 Eigendecomposition

For a diagonalizable matrix $A \in \mathbb{R}^{n \times n}$, the eigendecomposition is $A = V\Lambda V^{-1}$, where columns of $V$ are eigenvectors and $\Lambda = \mathrm{diag}(\lambda_1, \ldots, \lambda_n)$ holds the eigenvalues. It reveals the invariant directions of the linear map and makes matrix powers trivial: $A^k = V\Lambda^k V^{-1}$.

Not every square matrix is diagonalizable. Defective matrices (with repeated eigenvalues and insufficient eigenvectors) require the Jordan normal form. For symmetric real matrices, the spectral theorem guarantees real eigenvalues and an orthonormal eigenbasis, making $V^{-1} = V^\top$ and giving $A = V\Lambda V^\top$.

**For AI:** Eigenspectrum analysis of the Hessian explains sharpness and flat minima. Adam's diagonal preconditioning approximates the Fisher information matrix eigenspectrum. Vanishing and exploding gradients in RNNs are a direct consequence of the dominant eigenvalue of the recurrence matrix exceeding or falling below 1.

> → _Full treatment: [Eigenvalues and Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)_

---

### 8.5 Singular Value Decomposition

The SVD is the universal matrix factorization: for any $A \in \mathbb{R}^{m \times n}$,

$$A = U \Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ are orthogonal, and $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$. Unlike eigendecomposition, SVD exists for every matrix — rectangular, rank-deficient, or otherwise.

Key consequences: the best rank-$k$ approximation is $A_k = U_k \Sigma_k V_k^\top$ (Eckart–Young theorem); the pseudo-inverse is $A^+ = V \Sigma^+ U^\top$; the condition number is $\sigma_1 / \sigma_{\min}$; and the four fundamental subspaces (column space, null space, row space, left null space) are read directly from $U$ and $V$.

**For AI:** LoRA decomposes parameter updates as $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$ — a direct consequence of the SVD low-rank viewpoint. Mechanistic interpretability uses SVD to decompose attention OV circuits into rank-1 components. Randomised SVD (Halko et al. 2011) makes low-rank approximation tractable for billion-parameter weight matrices.

> → _Full treatment: [Singular Value Decomposition](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md)_

---

### 8.6 Cholesky Decomposition

For a symmetric positive definite (SPD) matrix $A$, Cholesky gives $A = LL^\top$ where $L$ is lower triangular with positive diagonal. It is approximately twice as fast as LU for SPD systems (exploiting symmetry), requires no pivoting, and serves as a numerical test of positive definiteness: Cholesky fails if and only if $A$ is not SPD.

**For AI:** Gaussian process inference, Bayesian neural network covariance updates, Kalman filter covariance propagation, and second-order optimizer preconditioning (natural gradient methods) all require efficient SPD linear solves — Cholesky is the standard tool. It also appears in sampling from multivariate Gaussians: to draw $x \sim \mathcal{N}(\mu, \Sigma)$, compute $L = \mathrm{chol}(\Sigma)$ then return $\mu + Lz$ where $z \sim \mathcal{N}(0, I)$.

> → _Full treatment: [Matrix Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md)_

## 9. Common Mistakes

| Mistake | Why it is wrong | Better rule |
| --- | --- | --- |
| "Matrix multiplication is commutative." | In general $AB \neq BA$, and sometimes only one product is even defined. | Track order carefully. |
| "$AB = 0$ means one factor must be zero." | Matrices have zero divisors. | Zero product does not imply zero factor. |
| "$(AB)^{-1} = A^{-1}B^{-1}$." | Inverse reverses order. | $(AB)^{-1} = B^{-1}A^{-1}$. |
| "If $A^2 = 0$, then $A=0$." | Nilpotent matrices exist. | Matrix powers do not behave like scalar powers. |
| "Product of symmetric matrices is symmetric." | $(AB)^\top = BA$, which equals $AB$ only if $A$ and $B$ commute. | $AB$ is symmetric iff $AB=BA$ when both are symmetric. |
| "Every square matrix has an inverse." | A square matrix can still be singular. | Check rank, determinant, or null space first. |
| "To solve $Ax=b$, compute $A^{-1}b$." | Explicit inversion is usually slower and less stable than solving directly. | Use a linear solver or factorization. |
| "Row rank and column rank may differ." | They are always equal. | Rank is one number, not two competing ones. |
| "Every square matrix has an eigendecomposition." | Some matrices are defective and not diagonalizable. | SVD always exists; eigendecomposition does not. |
| "Hadamard and matrix product are basically the same." | One is entrywise, one contracts over an index. | Treat them as different operations. |

---

## 10. Exercises

These exercises are designed to force both algebraic fluency and computational interpretation.

1. **Matrix arithmetic and properties**  
   Let

   $$
   A =
   \begin{pmatrix}
   1 & 2 \\
   3 & 4
   \end{pmatrix},
   \qquad
   B =
   \begin{pmatrix}
   0 & 1 \\
   1 & 0
   \end{pmatrix}.
   $$

   Compute:
   - $(a)$ $AB$ and $BA$, and verify they differ
   - $(b)$ $(AB)^\top$ and $B^\top A^\top$
   - $(c)$ $A^{-1}$ using the $2 \times 2$ formula
   - $(d)$ $\det(A)$, $\det(B)$, and $\det(AB)$
   - $(e)$ $\mathrm{tr}(AB)$ and $\mathrm{tr}(BA)$

2. **Shape compatibility and broadcasting**
   Let $A \in \mathbb{R}^{3 \times 4}$, $B \in \mathbb{R}^{4 \times 2}$, $C \in \mathbb{R}^{3 \times 2}$, $u \in \mathbb{R}^4$, $v \in \mathbb{R}^3$.
   - $(a)$ which of $AB$, $BA$, $AC$, $CA$, $BC$, $CB$ are defined? State each resulting shape.
   - $(b)$ compute the shape of $(AB)^\top$ and verify it equals $B^\top A^\top$
   - $(c)$ is $A^\top v$ defined? What about $Au$?
   - $(d)$ compute $Au$ — what kind of object is the result?
   - $(e)$ write $AB$ as a sum of $4$ rank-1 outer products (column-of-$A$ times row-of-$B$)

3. **Trace and inner-product identities**
   Let $A, B \in \mathbb{R}^{n \times n}$.
   - $(a)$ prove $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ from the definition $\mathrm{tr}(M) = \sum_i M_{ii}$
   - $(b)$ show that $\mathrm{tr}(A^\top B) = \sum_{i,j} A_{ij} B_{ij}$ (the Frobenius inner product)
   - $(c)$ use $(b)$ to write $\|A\|_F^2 = \mathrm{tr}(A^\top A)$
   - $(d)$ for $A = \begin{pmatrix}1&2\\3&4\end{pmatrix}$, $B = \begin{pmatrix}0&1\\1&0\end{pmatrix}$, verify $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ numerically
   - $(e)$ is $\mathrm{tr}(ABC) = \mathrm{tr}(BAC)$ in general? Prove or give a counterexample.

4. **Block matrix multiply and the outer-product form**
   Partition $A \in \mathbb{R}^{4 \times 4}$ and $B \in \mathbb{R}^{4 \times 4}$ as $2 \times 2$ blocks of size $2 \times 2$.
   - $(a)$ write the block matrix product formula $(AB)_{IJ} = \sum_K A_{IK} B_{KJ}$
   - $(b)$ for $A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$ with $A_{11} = I_2$, $A_{12} = 0$, $A_{21} = 0$, $A_{22} = 2I_2$ and $B = I_4$, compute $AB$ by block multiply and verify it equals $A$
   - $(c)$ express a rank-2 matrix $M = u_1 v_1^\top + u_2 v_2^\top$ as a matrix product $[u_1\ u_2][v_1\ v_2]^\top$ and state the shapes
   - $(d)$ explain why every rank-$r$ matrix can be written as a product of an $m \times r$ matrix and an $r \times n$ matrix
   - $(e)$ for a LoRA update $\Delta W = BA$ with $B \in \mathbb{R}^{512 \times 8}$ and $A \in \mathbb{R}^{8 \times 512}$, compute the parameter count versus full $W \in \mathbb{R}^{512 \times 512}$

5. **Pseudo-inverse and least squares**  
   Let

   $$
   A =
   \begin{pmatrix}
   1 & 2 \\
   3 & 4 \\
   5 & 6
   \end{pmatrix},
   \qquad
   b =
   \begin{pmatrix}
   1 \\ 2 \\ 3
   \end{pmatrix}.
   $$

   Solve:
   - $(a)$ show why the system is overdetermined
   - $(b)$ compute $(A^\top A)^{-1}A^\top b$
   - $(c)$ compute the residual norm
   - $(d)$ verify the pseudo-inverse formula using SVD
   - $(e)$ explain why $AA^+$ is a projection matrix

6. **Attention as matrix operations**  
   Let $Q, K, V \in \mathbb{R}^{4 \times 2}$ for a single attention head.
   - $(a)$ write down the shapes in $S = QK^\top / \sqrt{2}$
   - $(b)$ explain why $S$ is $4 \times 4$
   - $(c)$ apply row-wise softmax conceptually to obtain attention weights
   - $(d)$ compute the output shape of $AV$
   - $(e)$ explain why each row of the attention matrix sums to 1
   - $(f)$ if $Q=K$, determine whether $S$ is symmetric and whether the softmaxed matrix must still be symmetric

7. **Conditioning and numerical stability**
   Let $A = \begin{pmatrix}1 & 1 \\ 1 & 1+\epsilon\end{pmatrix}$ for small $\epsilon > 0$.
   - $(a)$ compute $\det(A)$ symbolically as a function of $\epsilon$
   - $(b)$ compute $A^{-1}$ explicitly using the $2 \times 2$ inverse formula
   - $(c)$ compute the condition number $\kappa(A) = \|A\|\|A^{-1}\|$ using the spectral norm (largest singular value) — describe what happens as $\epsilon \to 0$
   - $(d)$ explain why a large condition number means that a small perturbation $\delta b$ in $b$ causes a large perturbation in the solution $x$ of $Ax = b$
   - $(e)$ for $\epsilon = 10^{-8}$, estimate the number of digits of precision lost in the solution relative to 64-bit floating-point (which has $\approx 16$ significant digits)

8. **LoRA and low-rank structure**  
   Let $W \in \mathbb{R}^{512 \times 512}$ and let a LoRA update use rank $r=8$.
   - $(a)$ compute the number of parameters in full $W$
   - $(b)$ compute the number of parameters in a rank-8 factorization
   - $(c)$ express the update as a sum of outer products
   - $(d)$ interpret a singular-value spectrum such as $[5.2, 4.8, 0.1, 0.05, 0.02, 0.01, 0.003, 0.001]$
   - $(e)$ explain how truncated SVD could compress the learned update further

---

## 11. Why This Matters for AI

| Aspect | Why matrix operations matter |
| --- | --- |
| Every forward pass | Transformer layers are dominated by matrix multiplies: projections, attention scores, attention outputs, and feed-forward blocks. |
| Hardware efficiency | GPU and accelerator utilization is largely a GEMM scheduling problem. Batching, tiling, and layout determine whether tensor units are actually used well. |
| Backpropagation | Gradient formulas are matrix operations with transposes, outer products, and reductions. Reverse-order transpose rules are built directly into reverse-mode differentiation. |
| Low-rank adaptation | LoRA is matrix factorization applied to parameter updates. Its practicality is a direct consequence of low-rank structure and the SVD viewpoint. |
| MLA-style attention | DeepSeek-V2 reports that Multi-head Latent Attention compresses KV cache heavily and increases generation throughput by caching a low-rank latent representation instead of full key-value states. |
| Numerical stability | Condition number, decomposition choice, and matrix norms affect whether optimization remains stable or blows up. |
| Quantization | Post-training quantization, compensation, and calibration all rely on matrix norms, least squares, and low-rank corrections. |
| Optimizers | Methods such as K-FAC and Shampoo are explicitly matrix-based, using structured curvature approximations, matrix roots, or Kronecker factors. |
| Interpretability | SVD and eigenspectral analysis of weight matrices, OV circuits, and activations expose dominant directions and effective rank. |
| Inference latency | Training is GEMM-heavy and compute-dense, while single-sequence decoding often behaves more like bandwidth-limited GEMV. Understanding the matrix regime explains the performance difference. |

The short version is that matrix operations are not a chapter you pass through on the way to AI. They are the operational core of AI itself.

---

## 12. Conceptual Bridge

Vectors and spaces gave us the geometry of linear algebra. Matrix operations gave us the computational rules for acting on that geometry. Every abstract idea from the previous chapter now has an executable form:

- subspaces become column spaces, row spaces, null spaces, and projections
- orthogonality becomes QR and Cholesky
- low-dimensional approximation becomes SVD truncation
- invertibility becomes solving linear systems
- composition becomes matrix multiplication

This is the bridge to the next topics:

```text
vectors and spaces
    -> matrix operations
    -> spectral structure and decompositions
    -> probability and statistics
    -> optimization and dynamics
    -> deep learning systems
```

If this chapter leaves you with one durable mental model, let it be this:

> neural networks are not mysterious collections of tensor APIs; they are structured compositions of matrix operations, and the quality, speed, and stability of those operations determine what the model can learn and how efficiently it can run.

---

## References

### Core Linear Algebra and Numerical Linear Algebra

- Sheldon Axler, *Linear Algebra Done Right*: [https://linear.axler.net/](https://linear.axler.net/)
- MIT 18.06 Linear Algebra materials: [https://web.mit.edu/18.06/www/](https://web.mit.edu/18.06/www/)
- LAPACK official site: [https://www.netlib.org/lapack/](https://www.netlib.org/lapack/)
- NVIDIA cuBLAS documentation: [https://docs.nvidia.com/cuda/cublas/](https://docs.nvidia.com/cuda/cublas/)
- Halko, Martinsson, and Tropp (2011), *Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions*: [https://authors.library.caltech.edu/27399/](https://authors.library.caltech.edu/27399/)

### Transformers, Attention, and Low-Rank Adaptation

- Ashish Vaswani et al. (2017), *Attention Is All You Need*: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Tri Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- Tri Dao et al. (2024), *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*: [https://arxiv.org/abs/2407.08608](https://arxiv.org/abs/2407.08608)
- Edward J. Hu et al. (2021/2022), *LoRA: Low-Rank Adaptation of Large Language Models*: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

### Large-Model Systems and Hardware

- NVIDIA H100 product specifications: [https://www.nvidia.com/en-eu/data-center/h100/](https://www.nvidia.com/en-eu/data-center/h100/)
- NVIDIA Hopper architecture overview: [https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- DeepSeek-AI (2024), *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*: [https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-AI (2024/2025), *DeepSeek-V3 Technical Report*: [https://arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)

---

[Back to Home](../../README.md) | [Next: Systems of Equations](../03-Systems-of-Equations/notes.md)
