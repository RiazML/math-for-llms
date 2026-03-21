[Previous: Determinants](../04-Determinants/notes.md) | [Home](../../README.md) | [Next: Vector Spaces and Subspaces](../06-Vector-Spaces-Subspaces/notes.md)

---

# Matrix Rank

> _"Rank is the number of directions a matrix can actually use. Everything else is bookkeeping, redundancy, or collapse."_

## Overview

If determinants answer the question "is this matrix invertible?", rank answers the more general and more useful question:

$$
\text{How much independent information does this matrix really contain?}
$$

A matrix can have thousands or millions of entries and still behave as though only a few independent directions matter. That is the idea of rank. Rank measures the dimension of the image of a linear map, the number of independent columns, the number of pivots in row reduction, and the number of non-zero singular values. These are all the same quantity viewed from different angles.

This chapter treats rank as both:

- a classical linear algebra invariant
- a live design variable in modern AI systems

That second point matters. By 2026, low-rank structure is not a niche idea. It is built directly into:

- LoRA and other PEFT methods
- low-rank KV compression in DeepSeek-style latent attention
- truncated SVD model compression
- effective-rank analysis of trained weight matrices
- collapse diagnostics in self-supervised learning
- second-order optimization approximations

So rank is not merely one more chapter after determinants. It is the language of information bottlenecks in linear maps and therefore the language of compression, capacity, and structure in neural networks.

## Prerequisites

- Vector spaces, span, linear independence, basis, and dimension
- Matrix multiplication, transpose, and inverse
- Systems of linear equations and row reduction
- Determinants, especially their connection to invertibility
- Basic familiarity with SVD and singular values is helpful, but not required

## Companion Notebooks

| Notebook                           | Description                                                                                                    |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive exploration of rank, row reduction, SVD-based rank, low-rank approximation, and AI examples        |
| [exercises.ipynb](exercises.ipynb) | Practice on rank computation, null spaces, low-rank approximation, LoRA analysis, and attention-rank reasoning |

## Learning Objectives

After completing this chapter, you should be able to:

- Explain rank as the true dimensionality of a matrix and of the image of a linear map
- Compute rank from RREF, QR with pivoting, and singular values
- Use the rank-nullity theorem fluently
- Prove and apply basic rank inequalities
- Connect rank to solvability of linear systems and least-squares structure
- Understand rank factorisation and low-rank approximation via SVD
- Interpret low-rank structure in LoRA, attention, embeddings, and model compression
- Distinguish exact rank, numerical rank, stable rank, and effective rank
- Recognize how rank controls both expressiveness and information loss in neural layers

---

## Table of Contents

- [Matrix Rank](#matrix-rank)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Matrix Rank?](#11-what-is-matrix-rank)
    - [1.2 Why Rank Is Central to AI](#12-why-rank-is-central-to-ai)
    - [1.3 Rank as Dimensionality of Information Flow](#13-rank-as-dimensionality-of-information-flow)
    - [1.4 Rank and the Geometry of Linear Maps](#14-rank-and-the-geometry-of-linear-maps)
    - [1.5 Intuitive Examples](#15-intuitive-examples)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Row Rank and Column Rank](#21-row-rank-and-column-rank)
    - [2.2 Rank via Linear Independence](#22-rank-via-linear-independence)
    - [2.3 Rank via Row Reduction](#23-rank-via-row-reduction)
    - [2.4 Rank via Singular Values](#24-rank-via-singular-values)
    - [2.5 Rank via Determinants (Minor Rank)](#25-rank-via-determinants-minor-rank)
    - [2.6 Rank-Nullity Theorem](#26-rank-nullity-theorem)
  - [3. Computing Rank](#3-computing-rank)
    - [3.1 Rank via Row Reduction - Step by Step](#31-rank-via-row-reduction---step-by-step)
    - [3.2 Rank via QR with Column Pivoting](#32-rank-via-qr-with-column-pivoting)
    - [3.3 Rank via SVD (Most Reliable)](#33-rank-via-svd-most-reliable)
    - [3.4 Rank of Specific Matrices](#34-rank-of-specific-matrices)
    - [3.5 Rank Determination for Large Matrices](#35-rank-determination-for-large-matrices)
  - [4. Fundamental Properties of Rank](#4-fundamental-properties-of-rank)
    - [4.1 Row Rank Equals Column Rank - Proof](#41-row-rank-equals-column-rank---proof)
    - [4.2 Rank Inequalities](#42-rank-inequalities)
    - [4.3 Rank of Transpose and Gram Matrices](#43-rank-of-transpose-and-gram-matrices)
    - [4.4 Rank of Special Structures](#44-rank-of-special-structures)
    - [4.5 Rank and Linear Systems](#45-rank-and-linear-systems)
  - [5. Rank and Matrix Decompositions](#5-rank-and-matrix-decompositions)
    - [5.1 Rank Factorisation](#51-rank-factorisation)
    - [5.2 SVD and Rank](#52-svd-and-rank)
    - [5.3 Rank-Revealing QR Decomposition](#53-rank-revealing-qr-decomposition)
    - [5.4 Eigendecomposition and Rank](#54-eigendecomposition-and-rank)
    - [5.5 LU Decomposition and Rank](#55-lu-decomposition-and-rank)
  - [6. Low-Rank Approximation](#6-low-rank-approximation)
    - [6.1 The Eckart-Young-Mirsky Theorem](#61-the-eckart-young-mirsky-theorem)
    - [6.2 Fraction of Variance Explained](#62-fraction-of-variance-explained)
    - [6.3 Optimal Rank Selection](#63-optimal-rank-selection)
    - [6.4 Nuclear Norm as Convex Relaxation of Rank](#64-nuclear-norm-as-convex-relaxation-of-rank)
    - [6.5 Robust PCA (RPCA)](#65-robust-pca-rpca)
  - [7. Rank in Linear Systems](#7-rank-in-linear-systems)
    - [7.1 Rank and Solvability](#71-rank-and-solvability)
    - [7.2 Rank and the Four Fundamental Subspaces](#72-rank-and-the-four-fundamental-subspaces)
    - [7.3 Rank in Least Squares](#73-rank-in-least-squares)
    - [7.4 Rank and Pseudo-Inverse](#74-rank-and-pseudo-inverse)
  - [8. Rank in Neural Networks](#8-rank-in-neural-networks)
    - [8.1 Rank of Weight Matrices in Trained Models](#81-rank-of-weight-matrices-in-trained-models)
    - [8.2 LoRA - Low-Rank Adaptation](#82-lora---low-rank-adaptation)
    - [8.3 Rank in Attention Mechanisms](#83-rank-in-attention-mechanisms)
    - [8.4 MLA - Multi-head Latent Attention](#84-mla---multi-head-latent-attention)
    - [8.5 Rank in Embedding Matrices](#85-rank-in-embedding-matrices)
    - [8.6 Rank and Expressiveness](#86-rank-and-expressiveness)
  - [9. Rank, Regularisation, and Generalisation](#9-rank-regularisation-and-generalisation)
    - [9.1 Rank as a Complexity Measure](#91-rank-as-a-complexity-measure)
    - [9.2 Implicit Low-Rank Bias of Gradient Descent](#92-implicit-low-rank-bias-of-gradient-descent)
    - [9.3 Explicit Rank Regularisation](#93-explicit-rank-regularisation)
    - [9.4 The Double Descent Phenomenon and Rank](#94-the-double-descent-phenomenon-and-rank)
    - [9.5 Rank in Knowledge Distillation](#95-rank-in-knowledge-distillation)
  - [10. Numerical Rank and Stability](#10-numerical-rank-and-stability)
    - [10.1 Exact vs Numerical Rank](#101-exact-vs-numerical-rank)
    - [10.2 Rank Deficiency and Ill-Conditioning](#102-rank-deficiency-and-ill-conditioning)
    - [10.3 Rank, Conditioning, and Numerical Stability](#103-rank-conditioning-and-numerical-stability)
    - [10.4 Rank Monitoring During Training](#104-rank-monitoring-during-training)
    - [10.5 Rank in Mixed Precision Training](#105-rank-in-mixed-precision-training)
  - [11. Rank in Information Theory and Geometry](#11-rank-in-information-theory-and-geometry)
    - [11.1 Rank and Information Capacity](#111-rank-and-information-capacity)
    - [11.2 Rank and Mutual Information](#112-rank-and-mutual-information)
    - [11.3 Rank and the Geometry of Transformations](#113-rank-and-the-geometry-of-transformations)
    - [11.4 Grassmannian and the Space of Rank-r Matrices](#114-grassmannian-and-the-space-of-rank-r-matrices)
    - [11.5 Rank and Generalisation Bounds](#115-rank-and-generalisation-bounds)
  - [12. Structured Low-Rank Matrices](#12-structured-low-rank-matrices)
    - [12.1 Toeplitz and Hankel Matrices](#121-toeplitz-and-hankel-matrices)
    - [12.2 Butterfly Matrices](#122-butterfly-matrices)
    - [12.3 Hierarchical Low-Rank (H-matrices)](#123-hierarchical-low-rank-h-matrices)
    - [12.4 Tensor Train / Matrix Product States](#124-tensor-train--matrix-product-states)
    - [12.5 Block Low-Rank for Attention](#125-block-low-rank-for-attention)
  - [13. Rank Decompositions in Practice](#13-rank-decompositions-in-practice)
    - [13.1 Computing Low-Rank Factors](#131-computing-low-rank-factors)
    - [13.2 Incremental/Online Low-Rank Approximation](#132-incrementalonline-low-rank-approximation)
    - [13.3 Structured Rank Decompositions in AI Frameworks](#133-structured-rank-decompositions-in-ai-frameworks)
    - [13.4 Choosing the Right Rank](#134-choosing-the-right-rank)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI (2026 Perspective)](#16-why-this-matters-for-ai-2026-perspective)
  - [17. Conceptual Bridge](#17-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Is Matrix Rank?

The rank of a matrix is the number of genuinely independent directions that matrix contains. Equivalently:

- the maximum number of linearly independent rows
- the maximum number of linearly independent columns
- the dimension of the image of the associated linear map
- the number of pivots in row-reduced form
- the number of non-zero singular values

All of these definitions describe the same invariant.

That is already a strong hint about what rank really measures. Rank is not about how many entries are non-zero. It is not about how large the matrix looks on paper. It is about how much of the matrix is **new information** and how much is repetition.

Consider three extremes:

- **Rank 0**: the zero matrix. No row carries information. No column carries information. Everything maps to zero.
- **Rank 1**: every row is a multiple of one row, and every column is a multiple of one column. The matrix knows only one direction.
- **Full rank**: the maximum possible rank, namely $\min(m,n)$ for an $m \times n$ matrix. No redundant row/column directions remain.

This makes rank the right answer to a subtle but practical question:

```text
How many dimensions of structure does this matrix actually use?
```

That is why rank belongs next to determinants, not far away from them. Determinants tell you whether a square matrix collapses volume completely. Rank tells you **by how many dimensions** a general matrix collapses space.

### 1.2 Why Rank Is Central to AI

Matrix rank is not just classical theory with a few AI examples attached afterward. It is structurally central to modern deep learning.

**LoRA**

The defining equation of LoRA is

$$
\Delta W = BA
$$

with $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{r \times n}$. The whole point is that

$$
\operatorname{rank}(\Delta W) \le r.
$$

So the key LoRA hyperparameter is literally a rank budget.

**Attention**

If a head projects from dimension $d$ to $d_k$, then its induced bilinear form

$$
W_Q W_K^T
$$

has rank at most $d_k$. That means the head "sees" the world through a low-rank lens, even if $d$ is much larger.

**Model compression**

Truncated SVD replaces a dense matrix $W$ by a rank-$r$ approximation

$$
W_r = U_r \Sigma_r V_r^T.
$$

This is rank as compression.

**Generalisation**

Empirically, trained neural networks often behave as though their useful structure has much lower effective rank than their nominal parameter dimension would suggest. That is why stable rank, effective rank, and singular value decay are now common diagnostic tools.

So rank is the right abstraction for:

- information bottlenecks
- compression
- redundancy
- capacity control
- structured efficiency

### 1.3 Rank as Dimensionality of Information Flow

Let a linear layer be

$$
x \mapsto Wx
$$

with $W \in \mathbb{R}^{m \times n}$.

Then the output is always constrained to lie in the column space of $W$. If

$$
\operatorname{rank}(W)=r,
$$

then every possible output lies in an $r$-dimensional subspace of $\mathbb{R}^m$.

So even if:

- inputs live in $n=4096$ dimensions
- outputs live in $m=4096$ dimensions

if the rank is only $r=128$, then the layer only transmits 128 independent output directions.

At the same time, the null space has dimension

$$
n-r.
$$

Those are directions in input space that the layer destroys completely.

This gives the right information-flow picture:

```text
input directions split into:

  surviving directions  -> mapped into the image
  null directions       -> erased

rank(W)      = how much survives
nullity(W)   = how much is lost
```

This is why rank is best thought of as the dimensionality of information passage through a linear map.

### 1.4 Rank and the Geometry of Linear Maps

Every matrix $A \in \mathbb{R}^{m \times n}$ defines a map from $\mathbb{R}^n$ to $\mathbb{R}^m$. Rank describes how this map decomposes the spaces on both sides.

```text
Input space R^n
    |
    |-- row space      (dim = r)      -> directions that matter
    |
    |-- null space     (dim = n-r)    -> directions sent to 0
    |
    v
Output space R^m
    |
    |-- column space   (dim = r)      -> reachable outputs
    |
    |-- left null      (dim = m-r)    -> unreachable output directions
```

Restricted to the row space, the map is effectively a bijection onto the column space. Restricted to the null space, the map is zero.

That geometric decomposition is the real content behind many theorems about rank.

### 1.5 Intuitive Examples

Some matrices are worth keeping in mind as rank prototypes.

**Identity matrix**

$$
I_n
$$

has rank $n$. Every direction survives.

**All-ones matrix**

$$
\mathbf{1}\mathbf{1}^T
$$

has rank $1$. Every row is the same. Every column is the same up to scale. Only one direction remains.

**Projection matrix**

For a unit vector $u$,

$$
P = uu^T
$$

has rank $1$. Everything gets projected onto the line spanned by $u$.

**Random Gaussian matrix**

A random dense Gaussian matrix in $\mathbb{R}^{m \times n}$ has full rank with probability $1$. Accidental exact dependencies almost never occur in exact continuous models.

**Embedding matrix**

If $E \in \mathbb{R}^{|V| \times d}$ with vocabulary size $|V| \gg d$, then

$$
\operatorname{rank}(E) \le d.
$$

So no matter how large the vocabulary is, the embedding space dimension is the bottleneck.

### 1.6 Historical Timeline

- **Grassmann (1844):** abstract higher-dimensional linear structure begins to take shape
- **Frobenius (late 19th century):** matrix rank formalised in the context of linear systems
- **Sylvester:** nullity and dimensional relationships clarified
- **Eckart and Young (1936):** best low-rank approximation via SVD
- **Modern numerical linear algebra:** QR with pivoting and SVD become standard practical rank tools
- **Compressed sensing / matrix completion era (2000s):** low-rank structure becomes an optimisation object
- **Word embedding era (2010s):** low-rank factorisation ideas become standard in NLP representation learning
- **LoRA era (2021 onward):** low-rank adaptation becomes a default engineering primitive in LLM fine-tuning
- **DeepSeek-style latent attention (2024 onward):** low-rank bottlenecks move into core architecture design

---

## 2. Formal Definitions

### 2.1 Row Rank and Column Rank

For a matrix $A \in \mathbb{R}^{m \times n}$:

- the **row space** is the span of its rows inside $\mathbb{R}^n$
- the **column space** is the span of its columns inside $\mathbb{R}^m$

The **row rank** is

$$
\dim(\operatorname{row}(A)),
$$

and the **column rank** is

$$
\dim(\operatorname{col}(A)).
$$

The fundamental theorem is that these two numbers are always equal.

This equality is not obvious. Rows live in $\mathbb{R}^n$ and columns live in $\mathbb{R}^m$, so there is no immediate reason they should match. But they do, and that common value is what we call:

$$
\operatorname{rank}(A).
$$

### 2.2 Rank via Linear Independence

Rank can be defined directly in the language of linear independence.

We say $\operatorname{rank}(A)=r$ if:

- there exist $r$ linearly independent rows, and every set of $r+1$ rows is dependent

or equivalently:

- there exist $r$ linearly independent columns, and every set of $r+1$ columns is dependent

This is conceptually clean but computationally awkward, because checking all subsets is not scalable. It is better as a definition than as an algorithm.

### 2.3 Rank via Row Reduction

If you row-reduce a matrix to reduced row echelon form, the rank is the number of pivots.

That gives the practical rule:

$$
\operatorname{rank}(A)
=
\text{number of non-zero rows in RREF}
=
\text{number of pivots}.
$$

This works because elementary row operations preserve the row space dimension.

So RREF turns the vague question

```text
How many independent rows are really here?
```

into the concrete question

```text
How many pivots survived elimination?
```

### 2.4 Rank via Singular Values

If

$$
A = U \Sigma V^T
$$

is the singular value decomposition, then the rank is exactly the number of non-zero singular values.

If the singular values are

$$
\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(m,n)} \ge 0,
$$

then

$$
\operatorname{rank}(A)=\#\{i : \sigma_i > 0\}.
$$

This is the best definition numerically, because singular values are stable under perturbations and reveal approximate rank structure naturally.

In floating-point computation, one often uses a threshold rather than exact zero:

$$
\operatorname{rank}_\varepsilon(A)
=
\#\{i : \sigma_i > \varepsilon\}.
$$

### 2.5 Rank via Determinants (Minor Rank)

Rank can also be characterised through minors.

A matrix has rank at least $r$ if and only if it has some $r \times r$ minor with non-zero determinant.

It has rank exactly $r$ if:

- some $r \times r$ minor has non-zero determinant
- every $(r+1) \times (r+1)$ minor has determinant zero

This is important theoretically because it connects rank to determinants and algebraic geometry. But it is not how one computes rank in practice for large matrices.

### 2.6 Rank-Nullity Theorem

For $A \in \mathbb{R}^{m \times n}$,

$$
\operatorname{rank}(A) + \operatorname{nullity}(A) = n.
$$

This theorem says every input direction does exactly one of two things:

- it survives into the image
- it disappears into the null space

There is no third option.

So rank and nullity partition the domain:

```text
domain dimension
  =
  surviving dimensions
  +
  annihilated dimensions
```

This theorem is one of the fastest ways to turn rank information into concrete statements about solution sets, degrees of freedom, and information loss.

---

## 3. Computing Rank

### 3.1 Rank via Row Reduction - Step by Step

Take

$$
A=
\begin{pmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
1 & 3 & 4
\end{pmatrix}.
$$

Row-reduce:

$$
R_2 \leftarrow R_2 - 2R_1
$$

gives

$$
\begin{pmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
1 & 3 & 4
\end{pmatrix}.
$$

Then

$$
R_3 \leftarrow R_3 - R_1
$$

gives

$$
\begin{pmatrix}
1 & 2 & 3 \\
0 & 0 & 0 \\
0 & 1 & 1
\end{pmatrix}.
$$

Swap the second and third rows, then eliminate above the pivot:

$$
\begin{pmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
0 & 0 & 0
\end{pmatrix}.
$$

There are two pivots, so

$$
\operatorname{rank}(A)=2.
$$

That means:

- the row space is 2-dimensional
- the column space is 2-dimensional
- the null space in $\mathbb{R}^3$ is 1-dimensional

### 3.2 Rank via QR with Column Pivoting

In exact algebra, RREF is enough. In numerical work, QR with column pivoting is often preferred.

The factorisation is

$$
AP = QR,
$$

where $P$ is a permutation matrix chosen to expose the most important columns first.

The diagonal entries of $R$ usually decay in magnitude. A sharp drop suggests the effective rank.

This is rank determination through orthogonalisation rather than through exact symbolic elimination. It is more stable in floating-point arithmetic and is standard in serious numerical libraries.

### 3.3 Rank via SVD (Most Reliable)

SVD is the gold standard for numerical rank.

Why?

- singular values are stable
- near-dependence appears explicitly as tiny singular values
- exact rank and approximate rank are treated in one framework

So if a matrix is close to rank-deficient, SVD makes that visible immediately.

This is why `numpy.linalg.matrix_rank` uses singular values rather than RREF internally.

### 3.4 Rank of Specific Matrices

Some classes have immediate rank formulas.

**Zero matrix**

$$
\operatorname{rank}(0)=0.
$$

**Outer product**

If $u \neq 0$ and $v \neq 0$, then

$$
uv^T
$$

has rank $1$.

**Diagonal matrix**

The rank is the number of non-zero diagonal entries.

**Projection matrix**

Its rank is the dimension of the subspace onto which it projects.

**Gram matrix**

$$
\operatorname{rank}(A^TA)=\operatorname{rank}(A).
$$

This is extremely important in least squares, kernels, and attention-style Gram constructions.

### 3.5 Rank Determination for Large Matrices

For very large matrices, exact full SVD can be too expensive. Then one uses approximate tools.

**Randomised SVD**

Sketch the matrix with random projections, compress the action into a smaller subspace, and compute the SVD there. This gives high-quality approximate singular values at much lower cost when the matrix is approximately low rank.

**Stable rank**

Defined by

$$
\operatorname{sr}(A)=\frac{\|A\|_F^2}{\|A\|_2^2},
$$

stable rank is a smooth proxy for rank. It is never larger than the true rank and is less brittle under perturbations.

**Effective rank**

Effective rank is entropy-based: instead of asking "how many singular values are exactly non-zero?", it asks "how spread out is the singular value mass?"

This is often a more useful question in deep learning, where exact rank is usually full but effective structure is much smaller.

---

## 4. Fundamental Properties of Rank

### 4.1 Row Rank Equals Column Rank - Proof

The equality of row rank and column rank is fundamental enough that it deserves more than a slogan.

One clean proof uses the four-subspace viewpoint.

Let $r$ be the row rank of $A$. Then the row space has dimension $r$. The null space is the orthogonal complement of the row space inside $\mathbb{R}^n$, so by rank-nullity the null space has dimension $n-r$.

Now restrict $A$ to the row space. On that restricted space, $A$ is injective:

- if a vector lies in the row space and also in the null space, it must be zero

Therefore the image of the row space under $A$ has the same dimension as the row space itself, namely $r$.

But the image of the row space is exactly the column space.

So

$$
\dim(\operatorname{col}(A)) = \dim(\operatorname{row}(A)) = r.
$$

That is why there is only one rank.

### 4.2 Rank Inequalities

Rank obeys a small collection of extremely useful inequalities.

**Multiplication cannot increase information**

$$
\operatorname{rank}(AB)\le \min(\operatorname{rank}(A), \operatorname{rank}(B)).
$$

This is a profound structural statement: multiplying matrices cannot create new independent directions out of nowhere.

**Subadditivity**

$$
\operatorname{rank}(A+B)\le \operatorname{rank}(A)+\operatorname{rank}(B).
$$

Adding two low-rank matrices can increase rank, but not arbitrarily.

**Sylvester's inequality**

For compatible $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$,

$$
\operatorname{rank}(A)+\operatorname{rank}(B)-n
\le
\operatorname{rank}(AB).
$$

So the product rank is trapped between a lower and upper bound.

**Invariance under invertible change of basis**

If $P$ and $Q$ are invertible, then

$$
\operatorname{rank}(PAQ)=\operatorname{rank}(A).
$$

Rank is therefore an intrinsic property of the linear map, not of a particular coordinate representation.

### 4.3 Rank of Transpose and Gram Matrices

Several rank identities appear constantly in applications.

$$
\operatorname{rank}(A^T)=\operatorname{rank}(A).
$$

Also

$$
\operatorname{rank}(A^TA)=\operatorname{rank}(A)
$$

and

$$
\operatorname{rank}(AA^T)=\operatorname{rank}(A).
$$

The key reason is that

$$
\operatorname{null}(A^TA)=\operatorname{null}(A),
$$

since

$$
x^TA^TAx = \|Ax\|^2.
$$

So the Gram matrix does not change the essential dimensionality. It only repackages it.

This identity is central in:

- least squares
- covariance and kernel matrices
- attention score analysis
- SVD derivations

### 4.4 Rank of Special Structures

Rank behaves differently under different structured operations.

**Block diagonal**

$$
\operatorname{rank}
\begin{pmatrix}
A & 0 \\
0 & B
\end{pmatrix}
=
\operatorname{rank}(A)+\operatorname{rank}(B).
$$

The blocks contribute independently.

**Outer product sums**

If

$$
A = \sum_{k=1}^r u_k v_k^T,
$$

then

$$
\operatorname{rank}(A)\le r.
$$

This is the linear-algebraic heart of low-rank modelling.

**Kronecker product**

$$
\operatorname{rank}(A \otimes B)=\operatorname{rank}(A)\operatorname{rank}(B).
$$

This exact multiplicative rule makes Kronecker-structured approximations attractive in optimisation and second-order methods.

### 4.5 Rank and Linear Systems

Rank completely governs the structural classification of

$$
Ax=b.
$$

The system is:

- **consistent** iff $\operatorname{rank}(A)=\operatorname{rank}([A|b])$
- **uniquely solvable** iff that common rank equals $n$
- **infinitely solvable** iff the common rank is less than $n$
- **inconsistent** iff $\operatorname{rank}(A)<\operatorname{rank}([A|b])$

So rank is not just about the matrix in isolation. It decides what kinds of solutions are even possible.

---

## 5. Rank and Matrix Decompositions

### 5.1 Rank Factorisation

One of the cleanest structural theorems about rank is the rank factorisation theorem:

> If $\operatorname{rank}(A)=r$, then $A$ can be written as
>
> $$
> A = BC
> $$
>
> with $B \in \mathbb{R}^{m \times r}$ and $C \in \mathbb{R}^{r \times n}$.

This says a rank-$r$ matrix is exactly a matrix that factors through an $r$-dimensional latent space.

That point is worth slowing down for:

```text
R^n  --C-->  R^r  --B-->  R^m
```

Instead of mapping directly from $\mathbb{R}^n$ to $\mathbb{R}^m$, the matrix passes through an $r$-dimensional bottleneck.

That is not just a theorem. It is the blueprint of low-rank modelling.

If you write

$$
A = \sum_{k=1}^r b_k c_k^T,
$$

where the $b_k$ are columns of $B$ and the $c_k^T$ are rows of $C$, then the matrix is a sum of $r$ rank-1 outer products. Rank counts the minimum number of such outer products needed.

This is exactly why rank is the right quantity for parameter-efficient adaptation.

### 5.2 SVD and Rank

The singular value decomposition of $A \in \mathbb{R}^{m \times n}$ is

$$
A = U \Sigma V^T.
$$

If $\operatorname{rank}(A)=r$, then exactly $r$ singular values are non-zero, so

$$
A = \sum_{i=1}^r \sigma_i u_i v_i^T.
$$

This is the most canonical rank decomposition of all:

- the $u_i$ are orthonormal output directions
- the $v_i$ are orthonormal input directions
- the $\sigma_i$ tell you how strongly each direction is transmitted

So SVD does more than count rank. It orders the rank contributions by importance.

That is why SVD is both:

- a rank-revealing decomposition
- the foundation of optimal low-rank approximation

### 5.3 Rank-Revealing QR Decomposition

Ordinary QR decomposition is useful for solving systems, but it is not always explicit enough about rank structure.

Rank-revealing QR introduces column pivoting:

$$
AP = QR,
$$

where $P$ permutes columns so that important directions appear first.

In a numerically rank-$r$ matrix, the upper-triangular factor often takes the form

$$
R =
\begin{pmatrix}
R_{11} & R_{12} \\
0 & R_{22}
\end{pmatrix},
$$

with:

- $R_{11}$ well-conditioned
- $R_{22}$ small in norm

This makes the approximate rank visible without computing a full SVD.

QR with column pivoting is therefore a practical middle ground:

- cheaper than full SVD
- more stable than naive elimination
- often good enough for rank detection

### 5.4 Eigendecomposition and Rank

If $A$ is diagonalisable,

$$
A = V \Lambda V^{-1},
$$

then the rank equals the number of non-zero eigenvalues.

For symmetric matrices, this becomes particularly clean:

$$
A = Q \Lambda Q^T,
$$

and

$$
\operatorname{rank}(A)=\#\{i : \lambda_i \neq 0\}.
$$

This is especially important for symmetric positive semidefinite matrices such as:

- covariance matrices
- Gram matrices
- kernel matrices
- Hessian approximations

For such matrices, the eigenvalue story and the singular value story align perfectly.

### 5.5 LU Decomposition and Rank

If Gaussian elimination yields

$$
PA = LU,
$$

then in exact arithmetic the rank is the number of non-zero diagonal pivots in $U$.

This is fast and useful for well-conditioned problems, but less robust than SVD near rank deficiency.

So the hierarchy is:

- LU: fast and useful
- QR with pivoting: more robust
- SVD: most reliable

That hierarchy shows up repeatedly in numerical linear algebra and in large-scale model analysis.

---

## 6. Low-Rank Approximation

### 6.1 The Eckart-Young-Mirsky Theorem

The most important theorem in low-rank approximation is the Eckart-Young-Mirsky theorem.

If

$$
A = U \Sigma V^T
$$

is the SVD of $A$, then the best rank-$r$ approximation in both Frobenius norm and spectral norm is

$$
A_r = \sum_{i=1}^r \sigma_i u_i v_i^T.
$$

In other words, if you are only allowed rank $r$, the optimal thing to do is:

- keep the top $r$ singular directions
- throw away the rest

This is a theorem of exceptional practical importance because it says truncated SVD is not just a good heuristic. It is provably optimal.

The approximation errors are explicit:

$$
\|A-A_r\|_F^2 = \sum_{i=r+1}^{\min(m,n)} \sigma_i^2
$$

and

$$
\|A-A_r\|_2 = \sigma_{r+1}.
$$

### 6.2 Fraction of Variance Explained

The Frobenius norm of a matrix decomposes over singular values:

$$
\|A\|_F^2 = \sum_i \sigma_i^2.
$$

So the fraction captured by a rank-$r$ approximation is

$$
\frac{\|A_r\|_F^2}{\|A\|_F^2}
=
\frac{\sum_{i=1}^r \sigma_i^2}{\sum_i \sigma_i^2}.
$$

This is the matrix analogue of explained variance in PCA.

In practice, this matters because the usefulness of a low-rank approximation depends much more on singular value decay than on rank ratio alone.

A rank-8 approximation can be:

- terrible for one matrix
- almost perfect for another

depending entirely on the spectrum.

### 6.3 Optimal Rank Selection

The theorem tells you the best rank-$r$ approximation **for a chosen $r$**. But how do you choose $r$?

Common strategies include:

- choose the smallest $r$ explaining 90%, 95%, or 99% of Frobenius energy
- inspect the singular value scree plot and locate an elbow
- use an explicit numerical threshold for noise-dominated singular values
- tune $r$ as a compression-quality tradeoff hyperparameter

In modern ML practice, rank selection is often not purely mathematical. It is a budget decision:

```text
more rank  -> more parameters / more compute / more expressiveness
less rank  -> stronger compression / stronger bottleneck / less flexibility
```

This is exactly the LoRA tradeoff.

### 6.4 Nuclear Norm as Convex Relaxation of Rank

Rank is a difficult optimisation objective because it is discrete and non-convex.

The standard convex surrogate is the nuclear norm:

$$
\|A\|_* = \sum_i \sigma_i.
$$

This plays the same role for rank that the $\ell_1$ norm plays for sparsity.

Why is that useful? Because problems of the form

$$
\min \operatorname{rank}(A) \quad \text{subject to constraints}
$$

are usually hard, while

$$
\min \|A\|_* \quad \text{subject to constraints}
$$

is convex and therefore much more tractable.

This is the mathematical foundation of:

- matrix completion
- low-rank denoising
- convex low-rank regularisation

### 6.5 Robust PCA (RPCA)

Classical PCA assumes the data matrix is approximately low-rank with small dense noise.

Robust PCA assumes instead that

$$
A = L + S,
$$

where:

- $L$ is low-rank
- $S$ is sparse

The goal is to recover both.

The ideal optimisation problem is

$$
\min \operatorname{rank}(L) + \lambda \|S\|_0
$$

subject to

$$
A = L + S.
$$

Its convex relaxation is

$$
\min \|L\|_* + \lambda \|S\|_1
$$

subject to the same constraint.

This is important because it separates structured low-dimensional signal from sparse corruptions, which is exactly the type of decomposition often desired in model diagnostics and anomaly detection.

---

## 7. Rank in Linear Systems

### 7.1 Rank and Solvability

For a system

$$
Ax=b,
$$

rank determines the entire structural classification.

The system is solvable exactly when

$$
\operatorname{rank}(A)=\operatorname{rank}([A|b]).
$$

If that common rank equals the number of unknowns $n$, the solution is unique.

If it is smaller than $n$, there are infinitely many solutions, with solution set dimension

$$
n-\operatorname{rank}(A).
$$

So rank is the quantity that tells you how constrained the system really is.

### 7.2 Rank and the Four Fundamental Subspaces

For $A \in \mathbb{R}^{m \times n}$ with rank $r$:

- column space has dimension $r$
- row space has dimension $r$
- null space has dimension $n-r$
- left null space has dimension $m-r$

All four spaces are determined by that single number.

This is one reason rank is so powerful. A single integer determines the dimension bookkeeping of the entire map.

In Strang's language:

```text
R^n = row(A)  (+)  null(A)
R^m = col(A)  (+)  null(A^T)
```

where the sums are direct and orthogonal in the standard Euclidean setting.

### 7.3 Rank in Least Squares

If $A$ has full column rank, the least-squares problem

$$
\min_x \|Ax-b\|_2^2
$$

has a unique minimiser:

$$
x^* = (A^TA)^{-1}A^Tb.
$$

But if $A$ is rank-deficient, then $A^TA$ is singular, and uniqueness is lost unless one imposes an additional criterion.

That criterion is usually minimum norm, which leads directly to the pseudo-inverse.

So rank is what decides whether the normal equations are well-posed or degenerate.

### 7.4 Rank and Pseudo-Inverse

The Moore-Penrose pseudo-inverse handles all rank cases uniformly.

If

$$
A = U \Sigma V^T,
$$

then

$$
A^+ = V \Sigma^+ U^T,
$$

where non-zero singular values are inverted and zero singular values remain zero.

The pseudo-inverse gives:

- the exact inverse when $A$ is invertible
- the minimum-norm least-squares solution otherwise

And importantly,

$$
\operatorname{rank}(A^+) = \operatorname{rank}(A).
$$

So the pseudo-inverse does not invent new information. It works entirely inside the same effective rank structure.

---

## 8. Rank in Neural Networks

### 8.1 Rank of Weight Matrices in Trained Models

A dense weight matrix in a transformer layer may be nominally full size, but that does not mean all of its directions are equally important.

Empirically, trained neural network weight matrices often show:

- rapid singular value decay
- low stable rank relative to ambient size
- effective rank far below nominal rank

This means that while the matrix may be full rank in exact arithmetic, most of its action is concentrated in a smaller number of dominant directions.

That observation matters because it explains why low-rank methods work as well as they do. They are not compressing a perfectly isotropic object. They are exploiting genuine spectral redundancy.

### 8.2 LoRA - Low-Rank Adaptation

LoRA parameterises the update to a weight matrix as

$$
\Delta W = BA
$$

with

$$
B \in \mathbb{R}^{m \times r},
\qquad
A \in \mathbb{R}^{r \times n}.
$$

Therefore

$$
\operatorname{rank}(\Delta W)\le r.
$$

This is the central reason LoRA is parameter-efficient: instead of learning $mn$ independent parameters, it learns only

$$
r(m+n)
$$

parameters.

If $m=n=4096$ and $r=8$, that is:

$$
8(4096+4096)=65{,}536
$$

parameters instead of

$$
4096^2=16{,}777{,}216.
$$

So the update is over 250 times smaller.

The deeper point is not just compression. It is the hypothesis that task-specific adaptation lives in a low-dimensional subspace. Rank is therefore acting as:

- a memory budget
- an optimization constraint
- a prior on task structure

### 8.3 Rank in Attention Mechanisms

In a transformer head, the query and key projections are typically

$$
W_Q, W_K \in \mathbb{R}^{d \times d_k}.
$$

Then the induced bilinear form

$$
W_Q W_K^T
$$

is a $d \times d$ matrix of rank at most $d_k$.

So even before softmax, the attention mechanism is structurally low-rank.

If the sequence matrix is $X \in \mathbb{R}^{n \times d}$, then

$$
Q = XW_Q,
\qquad
K = XW_K,
$$

and the score matrix is

$$
S = QK^T.
$$

Its rank satisfies

$$
\operatorname{rank}(S)\le d_k.
$$

This is a remarkable fact: an $n \times n$ attention score matrix may have $n^2$ entries, but it is generated through a much smaller subspace constraint.

That is why it is accurate to say attention operates through a low-rank lens.

### 8.4 MLA - Multi-head Latent Attention

DeepSeek-style Multi-head Latent Attention makes rank structure architectural rather than merely emergent.

Instead of storing full key and value representations, MLA compresses them through a latent bottleneck:

$$
c = X W_{KV}^{\downarrow},
$$

with latent width $r \ll d$, followed by separate up-projections for keys and values.

So the effective key/value maps factor through an $r$-dimensional latent space. That means their rank is at most $r$.

This gives a direct memory benefit:

- smaller KV cache
- lower bandwidth pressure
- preserved expressive power when the needed structure is genuinely low-rank

In other words, MLA treats low-rank structure as a first-class systems design principle rather than as a post-hoc compression trick.

### 8.5 Rank in Embedding Matrices

An embedding matrix

$$
E \in \mathbb{R}^{|V| \times d}
$$

for vocabulary size $|V|$ and embedding width $d$ satisfies

$$
\operatorname{rank}(E)\le d.
$$

So if $|V| \gg d$, the vocabulary is forced through a $d$-dimensional bottleneck.

This has two implications:

1. the embedding width, not the vocabulary size, is the fundamental linear bottleneck
2. spectral structure of the embedding matrix can reveal how semantic information is distributed across directions

This is part of why embedding dimension is such a consequential architectural decision.

### 8.6 Rank and Expressiveness

A purely linear network cannot create rank that is not already permitted by its bottlenecks.

If

$$
W = W_L W_{L-1}\cdots W_1,
$$

then

$$
\operatorname{rank}(W)\le \min_\ell \operatorname{rank}(W_\ell).
$$

So depth alone does not remove rank bottlenecks in linear models.

What changes the story is nonlinearity. ReLU, GELU, and other nonlinearities mean the whole network is no longer a single linear map, and effective representational complexity can exceed what any one linear rank statement would suggest.

But the rank of each linear component still matters, because it shapes:

- local information flow
- bottlenecks between nonlinear stages
- capacity of heads and projections

---

## 9. Rank, Regularisation, and Generalisation

### 9.1 Rank as a Complexity Measure

Rank is a natural complexity measure for matrices because it counts how many independent directions are actually used.

A rank-$r$ matrix does not need $mn$ independent degrees of freedom in the way an arbitrary dense matrix does. Its structure is constrained to a much smaller manifold.

That is why lower rank is associated with:

- fewer effective degrees of freedom
- stronger inductive bias
- less risk of memorising arbitrary noise

This is the matrix analogue of preferring simpler models over more complicated ones.

### 9.2 Implicit Low-Rank Bias of Gradient Descent

A major line of modern theory shows that gradient-based optimisation often has an implicit bias toward low-complexity solutions.

In matrix factorisation settings, when one optimises over factors rather than directly over a dense matrix, the induced bias can favour low nuclear norm and therefore low-rank structure.

This matters for deep learning because it offers a mathematical explanation for why overparameterised models do not always behave as though all their nominal degrees of freedom are equally used.

In practical terms:

- many learned updates are much lower-dimensional than their ambient space
- fine-tuning often occupies a small subspace
- low-rank adaptation methods work because optimisation itself already tends to prefer such structure

### 9.3 Explicit Rank Regularisation

There are also explicit ways to push matrices toward low-rank structure.

**Nuclear norm penalties**

Add

$$
\lambda \|W\|_*
$$

to the objective.

**Hard rank constraints**

Optimise subject to

$$
\operatorname{rank}(W)\le r.
$$

**Singular value thresholding**

Shrink small singular values toward zero during iterative optimisation.

These methods turn low-rank structure from an emergent phenomenon into a deliberate design choice.

### 9.4 The Double Descent Phenomenon and Rank

In modern interpolation regimes, one often sees double descent rather than the classical U-shaped bias-variance curve.

Rank offers one useful lens on this:

- before interpolation, the effective model rank may be too small to fit the data well
- near interpolation, the system becomes delicate
- beyond interpolation, minimum-norm or low-complexity interpolating solutions can still generalise well

This perspective does not explain all of deep learning, but it does connect overparameterisation to low-complexity solution selection more concretely than vague "capacity" language alone.

### 9.5 Rank in Knowledge Distillation

Distillation often compresses a high-capacity teacher into a smaller student.

One way to understand that process is spectrally:

- the teacher may have a richer singular spectrum
- the student is asked to preserve the most important directions
- distillation is therefore partly a low-rank approximation problem in function space and representation space

This is why singular vectors, effective rank, and principal subspaces are natural diagnostic tools in distillation pipelines.

---

## 10. Numerical Rank and Stability

### 10.1 Exact vs Numerical Rank

Exact rank is a theorem-level object:

$$
\operatorname{rank}(A)=\#\{i : \sigma_i > 0\}.
$$

Numerical rank is a computational object:

$$
\operatorname{rank}_\varepsilon(A)=\#\{i : \sigma_i > \varepsilon\}.
$$

In exact arithmetic these coincide when there is a clear gap. In floating-point arithmetic they may not.

This matters because almost every learned matrix in deep learning is full rank numerically if you perturb it even slightly. But that does not mean full rank is the right structural description.

So one often asks:

```text
How many singular values are meaningfully above noise level?
```

rather than:

```text
How many singular values are literally non-zero?
```

### 10.2 Rank Deficiency and Ill-Conditioning

Exact rank deficiency means some singular values are truly zero.

Near rank deficiency means some singular values are tiny but non-zero.

That difference is mathematically small but numerically huge.

If $\sigma_{\min}$ is tiny, then

$$
\kappa(A)=\frac{\sigma_{\max}}{\sigma_{\min}}
$$

is large, and the matrix behaves as if it were almost singular for computational purposes.

So dropping effective rank and worsening conditioning are deeply connected.

### 10.3 Rank, Conditioning, and Numerical Stability

A square matrix can be full rank and still be numerically dangerous.

That is why determinant and exact rank are not enough in serious numerical analysis. One must also ask how separated the important singular directions are.

Regularisation helps by lifting the floor of the spectrum. For example, Tikhonov regularisation replaces

$$
A^TA
$$

by

$$
A^TA+\lambda I,
$$

which improves conditioning by preventing the smallest singular directions from collapsing entirely.

This is one reason regularisation is both:

- a statistical tool
- a numerical stability tool

### 10.4 Rank Monitoring During Training

In modern ML systems, one often tracks spectral quantities during training:

- singular value decay
- stable rank
- effective rank
- covariance rank of activations

Why?

Because representational collapse often appears spectrally before it appears in top-line metrics.

If a representation matrix or batch covariance becomes nearly rank-1, that is a strong warning sign that the model is learning redundancy instead of diversity.

This is particularly relevant in:

- self-supervised learning
- contrastive learning
- representation alignment objectives

### 10.5 Rank in Mixed Precision Training

Low-precision arithmetic introduces perturbations, and perturbations blur rank boundaries.

In exact arithmetic, a matrix may be rank $r$. In BF16 or FP16 computation, tiny singular directions may be drowned in rounding noise, making the matrix appear numerically full rank or numerically lower rank depending on the threshold and scaling.

This is why serious spectral analysis of trained models is usually done using higher-precision copies of the weights rather than the lowest-precision training representation.

Rank is therefore not only an algebraic concept. In practice it is a precision-sensitive diagnostic.

---

## 11. Rank in Information Theory and Geometry

### 11.1 Rank and Information Capacity

In communications theory, the rank of a channel matrix determines how many independent streams can be transmitted simultaneously.

That analogy carries directly into neural networks.

If a weight matrix has rank $r$, then it supports at most $r$ independent linear output directions. The matrix may be large, but its information-carrying capacity as a linear map is constrained by rank.

This makes rank a natural notion of channel capacity for linear layers.

### 11.2 Rank and Mutual Information

For Gaussian models, information quantities often reduce to log-determinant expressions involving matrices such as

$$
I + A \Sigma_X A^T \Sigma_{\text{noise}}^{-1}.
$$

Only the non-zero singular directions of $A$ contribute to these expressions. So while mutual information is not "equal to rank", rank determines how many independent signal directions can participate.

That is why low-rank channels are information-limited channels.

### 11.3 Rank and the Geometry of Transformations

A rank-$r$ map from $\mathbb{R}^n$ to $\mathbb{R}^m$ can be understood geometrically in two stages:

1. collapse the domain onto an $r$-dimensional subspace by killing the null directions
2. map that surviving $r$-dimensional space into an $r$-dimensional subspace of the codomain

So the map behaves like an isomorphism between two $r$-dimensional spaces plus total annihilation on the complement.

This is the real geometric meaning of rank.

### 11.4 Grassmannian and the Space of Rank-r Matrices

The collection of all $r$-dimensional subspaces of $\mathbb{R}^n$ forms the Grassmannian $\mathrm{Gr}(r,n)$.

A rank-$r$ matrix is determined by:

- an $r$-dimensional row space
- an $r$-dimensional column space
- an isomorphism between them

So low-rank optimisation is not just "optimisation with fewer numbers". It is optimisation over a curved structured space of matrices with fixed-rank geometry.

This viewpoint matters in modern optimisation because many low-rank methods are implicitly or explicitly moving on that manifold.

### 11.5 Rank and Generalisation Bounds

Rank-based model classes often admit tighter complexity bounds than unrestricted dense classes.

This is intuitive: if a hypothesis class is restricted to low-rank matrices, then it cannot represent arbitrary dense behaviour in every direction. Fewer effective degrees of freedom means fewer ways to overfit.

So rank is one of the cleanest structural complexity measures available for linear and bilinear models.

---

## 12. Structured Low-Rank Matrices

### 12.1 Toeplitz and Hankel Matrices

Toeplitz and Hankel matrices are not necessarily low rank themselves, but they often have low **structured** rank in the sense of displacement rank.

That means they are close to highly structured operators and therefore admit fast algorithms. Convolution is the most familiar example: it can be represented by a Toeplitz matrix but is usually applied with FFT-style or direct local methods rather than dense matrix multiplication.

This is an important lesson:

```text
Low rank is one kind of structure.
It is not the only kind of structure.
```

### 12.2 Butterfly Matrices

Butterfly factorisations provide sparse structured products with fast $O(n \log n)$ application cost.

These matrices may be full rank, but they still have dramatically reduced algorithmic complexity due to structural factorisation.

This matters because efficient AI layers are often built not from low rank alone, but from hybrid structure:

- sparse
- low-rank
- butterfly / FFT-like
- block-structured

### 12.3 Hierarchical Low-Rank (H-matrices)

Hierarchical low-rank methods approximate a matrix blockwise:

- near-diagonal or near-field blocks may be dense
- far-field blocks are approximated as low-rank

This is the right abstraction for large kernel matrices and long-range interactions where global dense structure exists, but most of it is compressible away from the diagonal.

It is one of the conceptual ancestors of many long-context attention approximations.

### 12.4 Tensor Train / Matrix Product States

Tensor-train decompositions generalise low-rank ideas to higher-order arrays.

Instead of approximating a matrix by low-rank factors, one approximates a tensor by chained low-dimensional cores. The analogue of rank becomes a sequence of bond dimensions.

This matters because many large parameter objects in ML are more naturally tensor-shaped than matrix-shaped, and compression ideas must scale accordingly.

### 12.5 Block Low-Rank for Attention

Long-context attention is often too expensive in its naive dense form.

One family of approximations treats the attention matrix as block low-rank:

- nearby interactions are kept accurately
- distant interactions are compressed

This makes rank a systems-level scalability tool:

- lower memory
- lower compute
- controllable approximation quality

So rank is not merely a post-training analysis object. It is an architectural design primitive.

---

## 13. Rank Decompositions in Practice

### 13.1 Computing Low-Rank Factors

There are several practical ways to compute low-rank factors:

- full SVD then truncate
- randomised SVD
- alternating minimisation over low-rank factors
- incremental / streaming sketch methods

The right choice depends on whether:

- the matrix is explicitly stored
- only matrix-vector products are available
- one needs exact approximation quality or just a useful sketch

### 13.2 Incremental/Online Low-Rank Approximation

In online settings, data or gradients arrive sequentially. Recomputing a full SVD every time is infeasible.

That motivates incremental methods such as:

- streaming PCA
- sketch-based subspace tracking
- rank-adaptive updates

These methods matter in training-time monitoring and in continual-learning settings, where one wants to estimate evolving low-rank structure without constant full decompositions.

### 13.3 Structured Rank Decompositions in AI Frameworks

Modern ML tooling exposes low-rank structure directly:

- approximate SVD routines
- LoRA / PEFT libraries
- quantisation-plus-adapter stacks
- rank-adaptive PEFT variants

This is evidence that low-rank modelling is not just theory. It is now part of the standard engineering interface for large-model training and deployment.

### 13.4 Choosing the Right Rank

There is no universal best rank.

One chooses rank based on:

- approximation error tolerance
- memory and compute budget
- task complexity
- observed singular value decay

In practice, the rank question is:

```text
What is the smallest subspace that still captures the behaviour I care about?
```

That is the right engineering formulation.

---

## 14. Common Mistakes

| Mistake                                                      | Why It's Wrong                                                | Fix                                                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| "Rank equals the number of non-zero entries"                 | A matrix can have many non-zero entries and still have rank 1 | Count independent rows/columns, not non-zero positions                                   |
| "Full rank means invertible"                                 | Only true for square matrices                                 | For rectangular matrices, full rank means max possible rank, not two-sided invertibility |
| "rank(A+B) = rank(A)+rank(B)"                                | Rank is subadditive, not additive in general                  | Use $\operatorname{rank}(A+B)\le \operatorname{rank}(A)+\operatorname{rank}(B)$          |
| "If A and B have rank r, then AB has rank r"                 | Product rank can drop drastically                             | Use $\operatorname{rank}(AB)\le \min(\operatorname{rank}(A),\operatorname{rank}(B))$     |
| "Tiny determinant means tiny rank"                           | Determinant and rank are different objects                    | Use singular values or pivots to assess rank                                             |
| "Exact rank is the right practical notion in floating point" | Near-zero singular values blur exact rank                     | Use numerical rank with thresholds                                                       |
| "More LoRA rank is always better"                            | Extra rank can waste budget or overfit                        | Match rank to task complexity and spectral decay                                         |
| "A matrix with large Frobenius norm must have high rank"     | Magnitude and rank are different concepts                     | Separate scale from dimensionality                                                       |
| "rank(A^TA) can exceed rank(A)"                              | They are always equal                                         | Remember $\operatorname{null}(A^TA)=\operatorname{null}(A)$                              |
| "Low-rank means useless"                                     | Many important signals live in low-dimensional structure      | Judge approximation quality by spectral decay, not by rank alone                         |

---

## 15. Exercises

1. **Computing rank**
   For each matrix, compute the rank by row reduction, then find a basis for the row space, column space, and null space:
   - $A=\begin{pmatrix}1&2&3\\2&4&6\\3&6&9\end{pmatrix}$
   - $B=\begin{pmatrix}1&0&2&1\\0&1&1&2\\1&1&3&3\end{pmatrix}$
   - $C=\begin{pmatrix}1&2&0&-1\\2&4&1&1\\-1&-2&1&3\\0&0&2&4\end{pmatrix}$

2. **Rank inequalities**
   Construct explicit examples showing:
   - two non-zero matrices whose product has rank 0
   - two rank-1 matrices whose sum has rank 2
   - a case where Sylvester's lower bound is tight

3. **SVD and low-rank approximation**
   For

   $$
   A=\begin{pmatrix}3&2&2\\2&3&-2\end{pmatrix},
   $$

   compute the SVD and the best rank-1 approximation. Measure the Frobenius error and compare it with the discarded singular value.

4. **LoRA parameter analysis**
   For a $4096 \times 4096$ weight matrix, compute the parameter counts for LoRA ranks $r=1,4,8,16,32,64$. Express each as a fraction of full fine-tuning.

5. **Rank and linear systems**
   Let $A \in \mathbb{R}^{4 \times 5}$ have rank 3.
   - What is the nullity?
   - If $Ax=b$ is consistent, what is the dimension of the solution set?
   - What changes if $b \notin \operatorname{col}(A)$?

6. **Attention rank**
   If a transformer head uses $d_k=64$ and sequence length $n=2048$, what is the maximum possible rank of the score matrix $QK^T$? Explain why this means the attention matrix is structurally constrained despite being $2048 \times 2048$.

7. **Stable rank**
   Compute the exact rank, stable rank, and condition number of:
   - $\operatorname{diag}(10,5,1,0.1,0.01)$
   - $I_5$
   - a rank-1 outer product $uv^T$ with $\|u\|=\|v\|=10$

8. **Numerical rank**
   Build a matrix with singular values $(10,1,10^{-2},10^{-6})$ and study how the detected numerical rank changes as the threshold varies.

---

## 16. Why This Matters for AI (2026 Perspective)

| Aspect                    | Impact                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| LoRA and PEFT             | Rank is the core budget variable controlling memory, compute, and expressiveness           |
| MLA and KV compression    | Low-rank bottlenecks reduce cache cost without full dense storage                          |
| Attention subspaces       | Rank explains why attention operates through a limited projection lens                     |
| Model compression         | Truncated SVD and related methods are rank engineering in practice                         |
| Representation collapse   | Stable rank and covariance rank expose collapse early                                      |
| Generalisation            | Low-rank structure acts as both explicit and implicit complexity control                   |
| Second-order optimisation | Curvature approximations often exploit low-rank or Kronecker structure                     |
| Spectral diagnostics      | Singular value decay, stable rank, and effective rank are now routine model-analysis tools |
| Embedding bottlenecks     | Embedding width sets a hard linear rank cap regardless of vocabulary size                  |
| Future architectures      | Low-rank structure is increasingly designed in, not merely discovered later                |

Rank matters for AI because deep learning is full of large matrices that are far less independent than they first appear. The question is almost never "how big is this matrix?" It is "how many directions inside it actually matter?" Rank is the mathematically precise version of that question.

---

## 17. Conceptual Bridge

Rank is the dimensionality of linear information.

- determinants told us whether a square matrix collapses volume completely
- rank tells us how many dimensions survive even when the map is rectangular or singular
- SVD turns rank into an ordered spectrum of importance

So rank is the natural bridge between:

- algebraic structure
- numerical approximation
- information bottlenecks
- AI efficiency methods

The next natural step is spectral structure in full detail: eigenvalues, eigenvectors, orthogonal decompositions, and singular value geometry.

```text
Matrix entries
    ->
rank
    ->
image / null space / bottleneck dimension
    ->
singular values and eigen-structure
    ->
compression, stability, and spectral analysis
```

---

## References

- Gilbert Strang, _Introduction to Linear Algebra_, Wellesley-Cambridge Press.
- Lloyd N. Trefethen and David Bau III, _Numerical Linear Algebra_, SIAM.
- Gene H. Golub and Charles F. Van Loan, _Matrix Computations_, Johns Hopkins University Press.
- [MIT 18.06 Linear Algebra](https://web.mit.edu/18.06/www/)
- [Stanford EE263](https://stanford.edu/class/ee263/)
- [Hu et al. (2021), "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Aghajanyan et al. (2020), "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"](https://arxiv.org/abs/2012.13255)
- [Halko, Martinsson, and Tropp (2011), "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions"](https://arxiv.org/abs/0909.4061)
- [Roy and Vetterli (2007), "The Effective Rank: A Measure of Effective Dimensionality"](https://infoscience.epfl.ch/entities/publication/f3c74b8f-1cad-43ed-8696-04b318d58703)
- [Gavish and Donoho (2014), "The Optimal Hard Threshold for Singular Values is 4/sqrt(3)"](https://arxiv.org/abs/1305.5870)
- [Candes and Recht (2009), "Exact Matrix Completion via Convex Optimization"](https://arxiv.org/abs/0805.4471)
- [Candes et al. (2011), "Robust Principal Component Analysis?"](https://arxiv.org/abs/0912.3599)
- [Mikolov et al. (2013), "Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/abs/1301.3781)
- [Pennington, Socher, and Manning (2014), "GloVe: Global Vectors for Word Representation"](https://aclanthology.org/D14-1162/)
- [Levy and Goldberg (2014), "Neural Word Embedding as Implicit Matrix Factorization"](https://papers.nips.cc/paper_files/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)
- [AdaLoRA (2023), "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"](https://arxiv.org/abs/2303.10512)
