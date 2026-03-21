[← Back to Advanced Linear Algebra](../README.md) | [← PCA](../03-Principal-Component-Analysis/notes.md) | [Orthogonality →](../05-Orthogonality-and-Orthonormality/notes.md)

---

# Linear Transformations

> _"The matrix is not the territory. It is a coordinate representation of a linear map — and the map exists independently of any basis you choose to describe it."_

## Overview

Linear transformations are the language in which all of machine learning is written. Every forward pass through a neural network is a sequence of linear maps punctuated by nonlinearities. Every attention head projects queries, keys, and values through learned linear transformations. Every gradient computation traces the chain rule backward through compositions of Jacobians — which are themselves linear maps. Understanding linear transformations at the abstract level — beyond matrices, beyond coordinates — is what separates a practitioner who can apply formulas from one who can reason about why those formulas work.

This section develops the full theory of linear transformations between vector spaces. We begin with the geometric intuition: what does a linear map *do* to space? We then build the formal machinery — kernel, image, rank-nullity theorem, matrix representation in arbitrary bases, change of basis, isomorphisms, and composition — that makes this intuition precise. We extend to affine maps (which include bias terms), to the Jacobian as the canonical linear approximation of a nonlinear function, and to the dual space, where gradients live.

Throughout, the AI connections are not decorative but structural: LoRA is a rank-$r$ linear map composition; attention is a sequence of linear projections; backpropagation is reverse-mode accumulation of Jacobians. By the end of this section, you will see matrix multiplication not as arithmetic, but as function composition — the most powerful organizing principle in applied mathematics.

## Prerequisites

- Vector spaces, subspaces, basis, and dimension — [Chapter 2 §06: Vector Spaces and Subspaces](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md)
- Matrix multiplication, inverse, transpose — [Chapter 2 §02: Matrix Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- Rank and null space — [Chapter 2 §05: Matrix Rank](../../02-Linear-Algebra-Basics/05-Matrix-Rank/notes.md)
- Eigenvalues and eigenvectors (used in §3.4, §5.1) — [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive exploration of linear maps: kernel/image computation, change-of-basis visualization, Jacobian experiments, LoRA rank analysis |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from rank-nullity proofs through softmax Jacobian and LoRA fine-tuning analysis |

## Learning Objectives

After completing this section, you will be able to:

- State the two axioms of a linear transformation and derive all immediate consequences
- Compute the kernel and image of any linear map and verify the rank-nullity theorem
- Construct the matrix representation of a linear map in any pair of bases
- Derive the change-of-basis formula $[T]_{\mathcal{B}'} = P^{-1}[T]_{\mathcal{B}} P$ and use it to diagonalize operators
- Identify and characterize projection, rotation, reflection, shear, and low-rank maps geometrically
- Extend linear maps to affine maps via homogeneous coordinates
- Compute the Jacobian of a vector-valued function and interpret backpropagation as Jacobian composition
- Explain why LoRA, attention, and linear probing are instances of linear map theory

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Three Views of a Matrix](#11-three-views-of-a-matrix)
  - [1.2 Geometric Action of Linear Maps](#12-geometric-action-of-linear-maps)
  - [1.3 Why Linearity is Special](#13-why-linearity-is-special)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Two Axioms and All Their Consequences](#21-the-two-axioms-and-all-their-consequences)
  - [2.2 Kernel of a Linear Map](#22-kernel-of-a-linear-map)
  - [2.3 Image of a Linear Map](#23-image-of-a-linear-map)
  - [2.4 The Rank-Nullity Theorem](#24-the-rank-nullity-theorem)
  - [2.5 Examples and Non-Examples](#25-examples-and-non-examples)
- [3. Matrix Representation of a Linear Map](#3-matrix-representation-of-a-linear-map)
  - [3.1 The Standard Basis Construction](#31-the-standard-basis-construction)
  - [3.2 Representation in Arbitrary Bases](#32-representation-in-arbitrary-bases)
  - [3.3 The Change-of-Basis Matrix](#33-the-change-of-basis-matrix)
  - [3.4 Similarity Transformations](#34-similarity-transformations)
- [4. Composition, Invertibility, and Isomorphisms](#4-composition-invertibility-and-isomorphisms)
  - [4.1 Composition is Matrix Multiplication](#41-composition-is-matrix-multiplication)
  - [4.2 Injectivity, Surjectivity, and Bijectivity](#42-injectivity-surjectivity-and-bijectivity)
  - [4.3 Isomorphisms](#43-isomorphisms)
  - [4.4 The Inverse Map](#44-the-inverse-map)
  - [4.5 The Four Fundamental Subspaces via Linear Maps](#45-the-four-fundamental-subspaces-via-linear-maps)
- [5. Special Classes of Linear Transformations](#5-special-classes-of-linear-transformations)
  - [5.1 Projection Operators](#51-projection-operators)
  - [5.2 Rotations and Reflections](#52-rotations-and-reflections)
  - [5.3 Shear and Scaling Maps](#53-shear-and-scaling-maps)
  - [5.4 The Geometry of Low-Rank Maps](#54-the-geometry-of-low-rank-maps)
- [6. Dual Spaces and Transposes](#6-dual-spaces-and-transposes)
  - [6.1 The Dual Space](#61-the-dual-space)
  - [6.2 The Dual Map and the Transpose](#62-the-dual-map-and-the-transpose)
  - [6.3 Gradients Live in the Dual Space](#63-gradients-live-in-the-dual-space)
- [7. Affine Transformations](#7-affine-transformations)
  - [7.1 Beyond Linearity: Affine Maps](#71-beyond-linearity-affine-maps)
  - [7.2 Homogeneous Coordinates](#72-homogeneous-coordinates)
  - [7.3 Neural Network Layers as Affine Maps](#73-neural-network-layers-as-affine-maps)
- [8. The Jacobian as Linear Approximation](#8-the-jacobian-as-linear-approximation)
  - [8.1 Linearization of Nonlinear Maps](#81-linearization-of-nonlinear-maps)
  - [8.2 The Jacobian Matrix](#82-the-jacobian-matrix)
  - [8.3 Chain Rule = Composition of Jacobians](#83-chain-rule--composition-of-jacobians)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Attention as Linear Projections](#91-attention-as-linear-projections)
  - [9.2 LoRA: Fine-Tuning via Low-Rank Composition](#92-lora-fine-tuning-via-low-rank-composition)
  - [9.3 Linear Probes and the Linear Representation Hypothesis](#93-linear-probes-and-the-linear-representation-hypothesis)
  - [9.4 Embedding and Unembedding](#94-embedding-and-unembedding)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 Three Views of a Matrix

A matrix $A \in \mathbb{R}^{m \times n}$ admits three distinct but equivalent interpretations, and fluent practitioners shift between them effortlessly.

**View 1: Data container.** The matrix is a rectangular array of numbers — $m$ rows, $n$ columns. This is the computational view. We use it when we care about entries: $A_{ij}$, the element in row $i$, column $j$.

**View 2: Column geometry.** The matrix is a collection of $n$ column vectors in $\mathbb{R}^m$. The product $A\mathbf{x}$ is then a linear combination: $A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$, where $\mathbf{a}_i$ is the $i$-th column. This view makes the column space visible and illuminates when $A\mathbf{x} = \mathbf{b}$ has solutions.

**View 3: Linear transformation.** The matrix defines a function $T: \mathbb{R}^n \to \mathbb{R}^m$ by $T(\mathbf{x}) = A\mathbf{x}$. This is the abstract, coordinate-free view. The *map* $T$ exists independently of any coordinate representation — the matrix $A$ is merely its description in a particular pair of bases (standard basis for both $\mathbb{R}^n$ and $\mathbb{R}^m$).

```
THREE VIEWS OF THE MATRIX A
════════════════════════════════════════════════════════════════════════

  [2  1]         Column geometry:          Linear map:
  [0  3]         A = [a₁ | a₂]            T: ℝ² → ℝ²
  [1 -1]                                  T(x) = Ax

  Data container      Span{a₁, a₂}         Sends basis vectors
  A₂₁ = 0            = column space        e₁ → col 1 of A
  A₃₂ = -1                                e₂ → col 2 of A

  All three describe the same mathematical object.

════════════════════════════════════════════════════════════════════════
```

**For AI:** In a transformer, the weight matrix $W_Q \in \mathbb{R}^{d_k \times d}$ is simultaneously all three: raw parameters to be optimized (View 1), a basis for the query subspace (View 2), and a linear projection from the residual stream to query space (View 3). Understanding which view you're using prevents confusion about what "the attention head is doing."

### 1.2 Geometric Action of Linear Maps

What does $T(\mathbf{x}) = A\mathbf{x}$ *do* to space? The two axioms — additivity and homogeneity — impose strong geometric constraints:

1. **The origin is fixed.** $T(\mathbf{0}) = A\mathbf{0} = \mathbf{0}$ always. A linear map cannot translate; it cannot move the origin. This is the most important geometric fact about linear transformations.

2. **Lines through the origin stay lines.** If $\mathbf{x}(t) = t\mathbf{v}$ is a line, then $T(\mathbf{x}(t)) = tT(\mathbf{v})$ is a line through the origin in the output space.

3. **Parallel lines stay parallel.** If $\mathbf{y}_1 - \mathbf{y}_2 = \mathbf{v}$ (same direction), then $T(\mathbf{y}_1) - T(\mathbf{y}_2) = T(\mathbf{v})$ (still same direction).

4. **Grid lines go to grid lines, equally spaced.** This is the classic 3Blue1Brown visualization: apply $A$ to the integer grid of $\mathbb{R}^2$ and you get a (possibly skewed) grid in $\mathbb{R}^2$.

5. **The unit square maps to a parallelogram.** The parallelogram spanned by the columns of $A$ is the image of the unit square under $T$.

**What can a linear map do?** Depending on the matrix:

- **Rotate** space (orthogonal matrix, $\det = 1$)
- **Reflect** space (orthogonal matrix, $\det = -1$)
- **Scale** dimensions (diagonal matrix)
- **Shear** space (triangular matrix)
- **Project** onto a subspace (idempotent: $P^2 = P$)
- **Collapse** space to a lower dimension (rank-deficient matrix)

What a linear map **cannot** do: translate (no origin-shifting), apply nonlinear distortion (no bending, folding, or curving of lines).

### 1.3 Why Linearity is Special

The superposition principle is the reason linear algebra is tractable. For a linear map $T$:

$$T\!\left(\sum_{i=1}^n c_i \mathbf{v}_i\right) = \sum_{i=1}^n c_i T(\mathbf{v}_i)$$

This means: to know $T$ on **all** of $V$, you only need to know $T$ on a **basis**. A basis has $\dim(V)$ elements. The map on an infinite space is encoded in finitely many column vectors. This is extraordinary compression: infinite → finite.

**Linearity vs nonlinearity in neural networks.** A deep neural network with no nonlinearities is just a single linear transformation: $W_L W_{L-1} \cdots W_1 \mathbf{x} = W_{\text{eff}} \mathbf{x}$. Multiple linear layers collapse to one. The nonlinearities (ReLU, GELU, SiLU) are what allow composition to create genuinely new representational power. The *linear* layers provide the parameterized directions; the *nonlinear* activations provide the expressive capacity.

**Linearity in analysis.** Many of the hardest problems in mathematics and ML become tractable when restricted to linear functions: linear regression has a closed-form solution, linear systems have complete theory, spectral analysis of linear operators is well-understood. The strategy of "linearize, solve, interpret" recurs throughout calculus, optimization, and signal processing.

**For AI:** The linear representation hypothesis (Elhage et al. 2022, Park et al. 2023) conjectures that many high-level features in LLMs are encoded as *directions* in representation space — i.e., they are linear features. If true, this means the crucial structure of LLM computation is linear, and all the machinery of this section applies directly to understanding what models are doing.

### 1.4 Historical Timeline

| Year | Person | Contribution |
| --- | --- | --- |
| 1844 | Hermann Grassmann | *Die lineale Ausdehnungslehre* — first abstract treatment of linear spaces |
| 1855 | Arthur Cayley | Matrix algebra as formal system; composition of matrices |
| 1888 | Giuseppe Peano | First rigorous axiomatization of vector spaces |
| 1902 | Henri Lebesgue | Integration as a linear functional on function spaces |
| 1904–1910 | Hilbert, Riesz, Fischer | Infinite-dimensional linear algebra; Hilbert spaces; spectral theory |
| 1929 | John von Neumann | Operator theory; linear transformations on Hilbert spaces |
| 1940s | Alan Turing, von Neumann | Linear algebra for numerical computation; matrix algorithms |
| 1986 | Rumelhart, Hinton, Williams | Backpropagation — chains of Jacobians as the training algorithm |
| 2017 | Vaswani et al. | Attention = linear projections + scaled dot product; transformers |
| 2021 | Hu et al. (LoRA) | Low-rank linear map updates for efficient fine-tuning |
| 2022– | Elhage, Park et al. | Linear representation hypothesis; mechanistic interpretability |

---

## 2. Formal Definitions

### 2.1 The Two Axioms and All Their Consequences

**Definition (Linear Transformation).** Let $V$ and $W$ be vector spaces over the same field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$). A function $T: V \to W$ is a **linear transformation** (also called a **linear map** or **homomorphism**) if it satisfies:

1. **Additivity:** $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ for all $\mathbf{u}, \mathbf{v} \in V$
2. **Homogeneity:** $T(c\mathbf{v}) = cT(\mathbf{v})$ for all $\mathbf{v} \in V$, $c \in \mathbb{F}$

These two axioms are often combined into the single condition:
$$T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v}) \quad \text{for all } \mathbf{u}, \mathbf{v} \in V, \; a, b \in \mathbb{F}$$

**Immediate consequences** (each follows directly from the axioms):

**Proposition 2.1.1 (Zero maps to zero).** $T(\mathbf{0}_V) = \mathbf{0}_W$.

*Proof:* $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$ for any $\mathbf{v} \in V$. $\square$

*This is a universal test:* if $T(\mathbf{0}) \neq \mathbf{0}$, then $T$ is not linear. Translation ($T(\mathbf{x}) = \mathbf{x} + \mathbf{b}$, $\mathbf{b} \neq \mathbf{0}$) fails immediately.

**Proposition 2.1.2 (Negatives are preserved).** $T(-\mathbf{v}) = -T(\mathbf{v})$.

*Proof:* $T(-\mathbf{v}) = T((-1)\mathbf{v}) = (-1)T(\mathbf{v}) = -T(\mathbf{v})$. $\square$

**Proposition 2.1.3 (General linear combinations).** For any $\mathbf{v}_1, \ldots, \mathbf{v}_k \in V$ and $c_1, \ldots, c_k \in \mathbb{F}$:
$$T\!\left(\sum_{i=1}^k c_i \mathbf{v}_i\right) = \sum_{i=1}^k c_i T(\mathbf{v}_i)$$

*Proof:* By induction using additivity and homogeneity. $\square$

**Proposition 2.1.4 (Determined by basis images).** If $\{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ is a basis for $V$, then $T$ is completely determined by $T(\mathbf{b}_1), \ldots, T(\mathbf{b}_n)$. Moreover, for any choice of $\mathbf{w}_1, \ldots, \mathbf{w}_n \in W$, there exists a unique linear map $T$ with $T(\mathbf{b}_i) = \mathbf{w}_i$.

*This is the fundamental construction theorem.* It means the matrix of $T$ (in standard coordinates) is:
$$A = \begin{bmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{bmatrix}$$

**The set of all linear maps $T: V \to W$** is denoted $\mathcal{L}(V, W)$ or $\operatorname{Hom}(V, W)$. It is itself a vector space under pointwise operations: $(S + T)(\mathbf{v}) = S(\mathbf{v}) + T(\mathbf{v})$ and $(cT)(\mathbf{v}) = cT(\mathbf{v})$.

### 2.2 Kernel of a Linear Map

**Definition (Kernel).** The **kernel** (or **null space**) of a linear map $T: V \to W$ is:
$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}_W\}$$

It is the set of all inputs that $T$ maps to zero — the "lost information" of the map.

**Theorem 2.2.1.** $\ker(T)$ is a subspace of $V$.

*Proof:*
- *Zero:* $T(\mathbf{0}) = \mathbf{0}$, so $\mathbf{0} \in \ker(T)$.
- *Closure under addition:* If $T(\mathbf{u}) = T(\mathbf{v}) = \mathbf{0}$, then $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0} + \mathbf{0} = \mathbf{0}$.
- *Closure under scaling:* If $T(\mathbf{v}) = \mathbf{0}$, then $T(c\mathbf{v}) = cT(\mathbf{v}) = c\mathbf{0} = \mathbf{0}$.
$\square$

**Geometric meaning.** $\ker(T)$ is the subspace that $T$ "collapses to zero." If $T$ is the projection onto the $xy$-plane, then $\ker(T)$ is the $z$-axis. If $T$ is differentiation of polynomials, $\ker(T)$ is the constant polynomials.

**Theorem 2.2.2 (Injectivity criterion).** $T$ is injective (one-to-one) if and only if $\ker(T) = \{\mathbf{0}\}$.

*Proof.* ($\Rightarrow$) If $T$ is injective and $T(\mathbf{v}) = \mathbf{0} = T(\mathbf{0})$, then $\mathbf{v} = \mathbf{0}$. ($\Leftarrow$) If $\ker(T) = \{\mathbf{0}\}$ and $T(\mathbf{u}) = T(\mathbf{v})$, then $T(\mathbf{u} - \mathbf{v}) = \mathbf{0}$, so $\mathbf{u} - \mathbf{v} = \mathbf{0}$, i.e., $\mathbf{u} = \mathbf{v}$. $\square$

The dimension of the kernel, $\dim(\ker(T))$, is called the **nullity** of $T$.

### 2.3 Image of a Linear Map

**Definition (Image).** The **image** (or **range**) of a linear map $T: V \to W$ is:
$$\operatorname{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\} = T(V)$$

It is the set of all possible outputs — the "reachable" part of $W$.

**Theorem 2.3.1.** $\operatorname{im}(T)$ is a subspace of $W$.

*Proof:*
- *Zero:* $T(\mathbf{0}) = \mathbf{0} \in \operatorname{im}(T)$.
- *Closure under addition:* $T(\mathbf{u}) + T(\mathbf{v}) = T(\mathbf{u} + \mathbf{v}) \in \operatorname{im}(T)$.
- *Closure under scaling:* $cT(\mathbf{v}) = T(c\mathbf{v}) \in \operatorname{im}(T)$.
$\square$

**Theorem 2.3.2.** $\operatorname{im}(T)$ is the column space of the matrix $A$ representing $T$.

*Proof:* $T(\mathbf{x}) = A\mathbf{x} = x_1\mathbf{a}_1 + \cdots + x_n\mathbf{a}_n$, which is exactly the span of the columns of $A$. $\square$

**Theorem 2.3.3 (Surjectivity criterion).** $T: V \to W$ is surjective (onto) if and only if $\operatorname{im}(T) = W$.

The dimension of the image, $\dim(\operatorname{im}(T))$, is called the **rank** of $T$, denoted $\operatorname{rank}(T)$.

### 2.4 The Rank-Nullity Theorem

This is one of the most elegant and useful results in linear algebra, connecting the three fundamental dimensions of a linear map.

**Theorem 2.4.1 (Rank-Nullity Theorem).** Let $T: V \to W$ be a linear map with $V$ finite-dimensional. Then:
$$\dim(V) = \dim(\ker(T)) + \dim(\operatorname{im}(T)) = \operatorname{nullity}(T) + \operatorname{rank}(T)$$

**Proof.** Let $\{\mathbf{k}_1, \ldots, \mathbf{k}_p\}$ be a basis for $\ker(T)$ (so $p = \operatorname{nullity}(T)$). Extend this to a basis for all of $V$: $\{\mathbf{k}_1, \ldots, \mathbf{k}_p, \mathbf{b}_1, \ldots, \mathbf{b}_q\}$ where $p + q = \dim(V)$.

We claim $\{T(\mathbf{b}_1), \ldots, T(\mathbf{b}_q)\}$ is a basis for $\operatorname{im}(T)$.

*Spanning:* For any $\mathbf{w} \in \operatorname{im}(T)$, write $\mathbf{w} = T(\mathbf{v})$ for some $\mathbf{v} = \sum c_i \mathbf{k}_i + \sum d_j \mathbf{b}_j$. Then $\mathbf{w} = T(\mathbf{v}) = \sum c_i T(\mathbf{k}_i) + \sum d_j T(\mathbf{b}_j) = \sum d_j T(\mathbf{b}_j)$.

*Linear independence:* If $\sum d_j T(\mathbf{b}_j) = \mathbf{0}$, then $T(\sum d_j \mathbf{b}_j) = \mathbf{0}$, so $\sum d_j \mathbf{b}_j \in \ker(T) = \operatorname{span}\{\mathbf{k}_1, \ldots, \mathbf{k}_p\}$. But $\{\mathbf{k}_1, \ldots, \mathbf{k}_p, \mathbf{b}_1, \ldots, \mathbf{b}_q\}$ is linearly independent, so all $d_j = 0$. $\square$

**Intuition:** The rank-nullity theorem says $T$ "uses" its input dimensions in two ways: some dimensions are collapsed to zero (nullity), and the rest are faithfully transmitted to the output (rank). Total in = lost + kept.

```
RANK-NULLITY THEOREM
════════════════════════════════════════════════════════════════════════

  T: V ──────────────────────────────────→ W
       │                                  │
       ├── ker(T) ──→ {0}                 │
       │   (nullity)   collapsed           │
       │                                  │
       └── "rest" ──→ im(T) ⊆ W          │
           (rank)      transmitted         │

  dim(V)    =    nullity(T)    +    rank(T)
  [total]     [collapsed dims]  [surviving dims]

  Example: T: ℝ⁵ → ℝ³, rank 2
  → nullity = 5 - 2 = 3
  → 3 dimensions collapse, 2 survive

════════════════════════════════════════════════════════════════════════
```

**Example.** $T: \mathbb{R}^4 \to \mathbb{R}^3$ defined by $A = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$. Rank = 3 (full row rank). Nullity = $4 - 3 = 1$. The null space is $\ker(T) = \operatorname{span}\{(-1, -1, 1, 0)^\top\}$.

**For AI:** In LoRA, a weight update $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ has rank at most $r$. By rank-nullity, its kernel has dimension at least $k - r$. When $r \ll k$, the update leaves most of the input space unchanged — it only "speaks to" an $r$-dimensional subspace.

### 2.5 Examples and Non-Examples

**Linear transformations:**

| Map | Domain → Codomain | Kernel | Image |
| --- | --- | --- | --- |
| $T(\mathbf{x}) = A\mathbf{x}$ (matrix mult.) | $\mathbb{R}^n \to \mathbb{R}^m$ | null space of $A$ | column space of $A$ |
| $T(f) = f'$ (differentiation) | $\mathcal{P}_n \to \mathcal{P}_{n-1}$ | constants | all polynomials of degree $\leq n-1$ |
| $T(f) = \int_0^x f(t)\,dt$ (integration) | $\mathcal{C}[0,1] \to \mathcal{C}[0,1]$ | $\{\mathbf{0}\}$ | functions vanishing at 0 |
| $T(\mathbf{x}) = \mathbf{0}$ (zero map) | $V \to W$ | all of $V$ | $\{\mathbf{0}\}$ |
| $T(\mathbf{x}) = \mathbf{x}$ (identity) | $V \to V$ | $\{\mathbf{0}\}$ | all of $V$ |
| $T(x, y) = (x, 0)$ (projection) | $\mathbb{R}^2 \to \mathbb{R}^2$ | $y$-axis | $x$-axis |
| $T(A) = \operatorname{tr}(A)$ (trace) | $\mathbb{R}^{n\times n} \to \mathbb{R}$ | trace-zero matrices | $\mathbb{R}$ |

**Non-linear maps (and why they fail):**

| Map | Linearity failure | Test |
| --- | --- | --- |
| $T(\mathbf{x}) = \mathbf{x} + \mathbf{b}$ ($\mathbf{b} \neq \mathbf{0}$) | $T(\mathbf{0}) = \mathbf{b} \neq \mathbf{0}$ | Zero test |
| $T(x) = x^2$ | $T(1+1) = 4 \neq T(1) + T(1) = 2$ | Additivity |
| $T(\mathbf{x}) = \lVert\mathbf{x}\rVert$ (norm) | $T(2\mathbf{e}_1) = 2 \neq 2T(\mathbf{e}_1) = 2$... wait, this passes? No: $T(-\mathbf{e}_1) = 1 \neq -T(\mathbf{e}_1) = -1$ | Homogeneity |
| $T(\mathbf{x}) = \operatorname{softmax}(\mathbf{x})$ | $T(2\mathbf{x}) \neq 2T(\mathbf{x})$; softmax is scale-invariant | Homogeneity |
| $T(\mathbf{x}) = \operatorname{ReLU}(\mathbf{x})$ | $T(\mathbf{u} + \mathbf{v}) \neq T(\mathbf{u}) + T(\mathbf{v})$ in general | Additivity |
| $T(\mathbf{x}) = \mathbf{x} \odot \mathbf{x}$ (elementwise square) | $T(c\mathbf{x}) = c^2 \mathbf{x} \odot \mathbf{x} \neq c(\mathbf{x} \odot \mathbf{x})$ | Homogeneity |

**Note on ReLU:** Though ReLU is not linear, it is **piecewise linear** — linear on each orthant. This means neural networks with ReLU are piecewise linear functions, which is a key fact for understanding their behavior.

---

## 3. Matrix Representation of a Linear Map

### 3.1 The Standard Basis Construction

Over $\mathbb{R}^n$ and $\mathbb{R}^m$ with standard bases, every linear map $T: \mathbb{R}^n \to \mathbb{R}^m$ corresponds to a unique matrix $A \in \mathbb{R}^{m \times n}$.

**Construction.** The $j$-th column of $A$ is $T(\mathbf{e}_j)$, the image of the $j$-th standard basis vector:
$$A = \begin{bmatrix} T(\mathbf{e}_1) & T(\mathbf{e}_2) & \cdots & T(\mathbf{e}_n) \end{bmatrix}$$

**Why this works:** For any $\mathbf{x} = \sum_{j=1}^n x_j \mathbf{e}_j$:
$$T(\mathbf{x}) = T\!\left(\sum_j x_j \mathbf{e}_j\right) = \sum_j x_j T(\mathbf{e}_j) = A\mathbf{x}$$

The matrix is the complete, coordinate-encoded description of $T$.

**Example.** Find the matrix for the $2D$ counterclockwise rotation by angle $\theta$.

$T(\mathbf{e}_1) = (\cos\theta, \sin\theta)^\top$ and $T(\mathbf{e}_2) = (-\sin\theta, \cos\theta)^\top$, giving:
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Example.** Find the matrix for differentiation $D: \mathcal{P}_3 \to \mathcal{P}_2$ using the bases $\{1, x, x^2, x^3\}$ and $\{1, x, x^2\}$.

$D(1) = 0$, $D(x) = 1$, $D(x^2) = 2x$, $D(x^3) = 3x^2$. In the output basis:
$$[D]_{\mathcal{B}}^{\mathcal{C}} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}$$

This is a $3 \times 4$ matrix, reflecting $D: \mathcal{P}_3 \to \mathcal{P}_2$.

### 3.2 Representation in Arbitrary Bases

When $V$ and $W$ have non-standard bases, the matrix of $T$ depends on those bases.

**Setup.** Let $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ be a basis for $V$ and $\mathcal{C} = \{\mathbf{c}_1, \ldots, \mathbf{c}_m\}$ be a basis for $W$.

**Coordinate vectors.** For $\mathbf{v} \in V$, write $\mathbf{v} = \sum_j \alpha_j \mathbf{b}_j$. The **coordinate vector** is $[\mathbf{v}]_{\mathcal{B}} = (\alpha_1, \ldots, \alpha_n)^\top \in \mathbb{R}^n$.

**The matrix of $T$ in bases $(\mathcal{B}, \mathcal{C})$.** Express each $T(\mathbf{b}_j)$ in the basis $\mathcal{C}$:
$$T(\mathbf{b}_j) = \sum_{i=1}^m a_{ij} \mathbf{c}_i$$

The matrix $[T]_{\mathcal{B}}^{\mathcal{C}} = (a_{ij}) \in \mathbb{R}^{m \times n}$ satisfies:
$$[T(\mathbf{v})]_{\mathcal{C}} = [T]_{\mathcal{B}}^{\mathcal{C}} \, [\mathbf{v}]_{\mathcal{B}}$$

The commutative diagram:

```
  v ∈ V  ────────────────T────────────────→  T(v) ∈ W
     │                                            │
   [·]_B (coord in B)                       [·]_C (coord in C)
     │                                            │
     ↓                                            ↓
  [v]_B ─────────[T]_B^C (matrix mult.)────→  [T(v)]_C
```

The matrix $[T]_{\mathcal{B}}^{\mathcal{C}}$ is the bridge between coordinates: it takes $\mathcal{B}$-coordinates of input to $\mathcal{C}$-coordinates of output.

**Example.** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ be the map $T(x, y) = (x + y, x - y)$. In the non-standard basis $\mathcal{B} = \{(1, 1), (1, -1)\}$:

$T(1,1) = (2, 0) = 1 \cdot (1,1) + 1 \cdot (1,-1)$, so column 1 is $(1, 1)^\top$.
$T(1,-1) = (0, 2) = 1 \cdot (1,1) + (-1) \cdot (1,-1)$, so column 2 is $(1, -1)^\top$.

$$[T]_{\mathcal{B}}^{\mathcal{B}} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

This is diagonal-like: in the basis $\mathcal{B}$, $T$ acts as a simple scaling on each basis vector — exactly the diagonalization idea.

### 3.3 The Change-of-Basis Matrix

**Definition.** Let $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ and $\mathcal{B}' = \{\mathbf{b}'_1, \ldots, \mathbf{b}'_n\}$ be two bases for the same space $V$. The **change-of-basis matrix** from $\mathcal{B}'$ to $\mathcal{B}$ is:
$$P = \begin{bmatrix} [\mathbf{b}'_1]_{\mathcal{B}} & [\mathbf{b}'_2]_{\mathcal{B}} & \cdots & [\mathbf{b}'_n]_{\mathcal{B}} \end{bmatrix}$$

Each column is the $\mathcal{B}$-coordinate vector of the corresponding new basis vector.

**Key property:** If $[\mathbf{v}]_{\mathcal{B}'}$ are the $\mathcal{B}'$-coordinates of $\mathbf{v}$, then the $\mathcal{B}$-coordinates are:
$$[\mathbf{v}]_{\mathcal{B}} = P \, [\mathbf{v}]_{\mathcal{B}'}$$

**The change-of-basis formula for $T$.** If $[T]_{\mathcal{B}}$ is the matrix of $T$ in the basis $\mathcal{B}$, and $P$ is the change-of-basis matrix from $\mathcal{B}'$ to $\mathcal{B}$, then:

$$[T]_{\mathcal{B}'} = P^{-1} [T]_{\mathcal{B}} P$$

**Derivation:**
$$[T(\mathbf{v})]_{\mathcal{B}'} = P^{-1} [T(\mathbf{v})]_{\mathcal{B}} = P^{-1} [T]_{\mathcal{B}} [\mathbf{v}]_{\mathcal{B}} = P^{-1} [T]_{\mathcal{B}} P [\mathbf{v}]_{\mathcal{B}'}$$

**Worked example.** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ have matrix $A = \begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}$ in the standard basis. In the basis $\mathcal{B}' = \{(1, 0), (1, 1)\}$:

$$P = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad P^{-1} = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix}$$

$$P^{-1} A P = \begin{bmatrix} 1 & -1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 3 & -1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 0 & 3 \end{bmatrix}$$

### 3.4 Similarity Transformations

**Definition.** Two matrices $A, B \in \mathbb{R}^{n \times n}$ are **similar** ($A \sim B$) if there exists an invertible $P$ such that $B = P^{-1}AP$.

Geometrically: $A$ and $B$ represent the **same linear map** in **different bases**.

**Invariants under similarity** (properties that don't change when you change basis):

| Invariant | Formula | Meaning |
| --- | --- | --- |
| Eigenvalues | $\lambda$ unchanged | Spectrum is basis-independent |
| Determinant | $\det(P^{-1}AP) = \det(A)$ | Volume scaling is basis-independent |
| Trace | $\operatorname{tr}(P^{-1}AP) = \operatorname{tr}(A)$ | Sum of eigenvalues |
| Rank | $\operatorname{rank}(P^{-1}AP) = \operatorname{rank}(A)$ | Dimension of image |
| Characteristic polynomial | $\det(\lambda I - P^{-1}AP) = \det(\lambda I - A)$ | Entire spectrum |

**Diagonalization is change of basis.** When $A$ is diagonalizable with eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$, the matrix $P = [\mathbf{v}_1 | \cdots | \mathbf{v}_n]$ gives $P^{-1}AP = \Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$. We are choosing the basis of eigenvectors, in which the map acts by simple scaling.

> **Forward reference: Eigenvalues and Eigenvectors**
>
> The full theory of diagonalization, the spectral theorem, and when a matrix is diagonalizable is developed in [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md). Here we note only that diagonalization is a special case of change of basis.

---

## 4. Composition, Invertibility, and Isomorphisms

### 4.1 Composition is Matrix Multiplication

Let $T: U \to V$ and $S: V \to W$ be linear maps. Their **composition** $S \circ T: U \to W$ defined by $(S \circ T)(\mathbf{u}) = S(T(\mathbf{u}))$ is also linear.

**Theorem 4.1.1.** If $T$ has matrix $A$ (in standard bases) and $S$ has matrix $B$, then $S \circ T$ has matrix $BA$.

*Proof:* $(S \circ T)(\mathbf{x}) = S(T(\mathbf{x})) = S(A\mathbf{x}) = B(A\mathbf{x}) = (BA)\mathbf{x}$. $\square$

This is the fundamental theorem connecting composition of functions to matrix multiplication. It explains:
- **Non-commutativity:** $S \circ T \neq T \circ S$ in general (different domains/codomains, or $BA \neq AB$ for square matrices).
- **Associativity:** $(R \circ S) \circ T = R \circ (S \circ T)$ — matches matrix associativity.
- **Deep learning:** A network $f = f_L \circ \cdots \circ f_1$ is a composition. The forward pass computes this product. The backward pass (backprop) uses the chain rule, which is exactly the composition of Jacobians.

**Collapse of linear-only networks.** If $f_i(\mathbf{x}) = W_i \mathbf{x}$ (all layers linear, no activations):
$$f(\mathbf{x}) = W_L \cdots W_2 W_1 \mathbf{x} = W_{\text{eff}} \mathbf{x}$$

where $W_{\text{eff}} = W_L \cdots W_1$ is a single matrix. No depth benefit without nonlinearity.

### 4.2 Injectivity, Surjectivity, and Bijectivity

**Definitions.** A linear map $T: V \to W$ is:
- **Injective** (one-to-one) if $T(\mathbf{u}) = T(\mathbf{v}) \Rightarrow \mathbf{u} = \mathbf{v}$
- **Surjective** (onto) if for every $\mathbf{w} \in W$, $\exists\, \mathbf{v} \in V$ with $T(\mathbf{v}) = \mathbf{w}$
- **Bijective** if both injective and surjective

**Criteria via rank-nullity:**

| Property | Condition | Matrix equivalent |
| --- | --- | --- |
| Injective | $\ker(T) = \{\mathbf{0}\}$, i.e., nullity = 0 | Columns of $A$ are linearly independent |
| Surjective | $\operatorname{im}(T) = W$, i.e., rank = $\dim(W)$ | Rows of $A$ span $\mathbb{R}^m$; full row rank |
| Bijective | Both; requires $\dim(V) = \dim(W)$ and full rank | $A$ is square and invertible |

**Dimension constraints:**
- If $\dim(V) < \dim(W)$: $T$ cannot be surjective (rank $\leq \dim(V) < \dim(W)$).
- If $\dim(V) > \dim(W)$: $T$ cannot be injective (nullity $= \dim(V) - \text{rank} \geq \dim(V) - \dim(W) > 0$).
- If $\dim(V) = \dim(W)$: injective $\Leftrightarrow$ surjective $\Leftrightarrow$ bijective.

### 4.3 Isomorphisms

**Definition.** A bijective linear map $T: V \to W$ is called an **isomorphism**. If such a map exists, $V$ and $W$ are **isomorphic**, written $V \cong W$.

Isomorphic spaces are "the same" as vector spaces — they have the same algebraic structure. An isomorphism is a relabeling of elements that preserves all linear operations.

**Theorem 4.3.1 (Fundamental Classification).** Two finite-dimensional vector spaces over the same field are isomorphic if and only if they have the same dimension.

*Consequence:* Every $n$-dimensional vector space over $\mathbb{R}$ is isomorphic to $\mathbb{R}^n$. The space $\mathcal{P}_n$ of polynomials of degree $\leq n$ (dimension $n+1$) is isomorphic to $\mathbb{R}^{n+1}$. The space $\mathbb{R}^{2 \times 2}$ (dimension 4) is isomorphic to $\mathbb{R}^4$.

**For AI:** The residual stream of a transformer (a vector in $\mathbb{R}^d$) is isomorphic to any other $d$-dimensional space. Features are not inherently "in $\mathbb{R}^d$" — they live in an abstract vector space and $\mathbb{R}^d$ is just one coordinate representation. The linear representation hypothesis says this space has meaningful geometric structure independent of the coordinate system.

**Properties of isomorphisms:**
- The inverse $T^{-1}: W \to V$ is also an isomorphism.
- Isomorphism preserves all vector space structure: subspaces, linear independence, bases, dimension.
- Isomorphisms form a group under composition.

### 4.4 The Inverse Map

**Theorem 4.4.1.** If $T: V \to W$ is an isomorphism (bijective linear map), then $T^{-1}: W \to V$ is also a linear map.

*Proof:* For $\mathbf{w}_1, \mathbf{w}_2 \in W$, let $\mathbf{v}_i = T^{-1}(\mathbf{w}_i)$, so $T(\mathbf{v}_i) = \mathbf{w}_i$. Then:
$$T(a\mathbf{v}_1 + b\mathbf{v}_2) = aT(\mathbf{v}_1) + bT(\mathbf{v}_2) = a\mathbf{w}_1 + b\mathbf{w}_2$$
So $T^{-1}(a\mathbf{w}_1 + b\mathbf{w}_2) = a\mathbf{v}_1 + b\mathbf{v}_2 = aT^{-1}(\mathbf{w}_1) + bT^{-1}(\mathbf{w}_2)$. $\square$

**Matrix inverse.** If $T$ has matrix $A$ (square, full rank), then $T^{-1}$ has matrix $A^{-1}$.

**Left and right inverses for non-square maps.** When $T: \mathbb{R}^n \to \mathbb{R}^m$ is injective but not surjective ($n < m$), there is no full inverse. However:
- **Left inverse:** $L$ such that $L \circ T = I_V$. Exists iff $T$ is injective. Formula: $L = (A^\top A)^{-1} A^\top$ (left pseudo-inverse).
- **Right inverse:** $R$ such that $T \circ R = I_W$. Exists iff $T$ is surjective. Formula: $R = A^\top (A A^\top)^{-1}$ (right pseudo-inverse).

> **Forward reference: Moore-Penrose Pseudo-Inverse**
>
> The general pseudo-inverse $A^+$, defined via SVD as $A^+ = V\Sigma^+ U^\top$, handles all cases uniformly and gives the least-squares solution when an exact solution doesn't exist. Full treatment in [§02: Singular Value Decomposition](../02-Singular-Value-Decomposition/notes.md#pseudo-inverse).

### 4.5 The Four Fundamental Subspaces via Linear Maps

Every linear map $T: V \to W$ (with $V = \mathbb{R}^n$, $W = \mathbb{R}^m$, matrix $A$) defines four fundamental subspaces:

| Subspace | Definition | Lives in | Dimension |
| --- | --- | --- | --- |
| Column space $\operatorname{col}(A)$ | $\operatorname{im}(T)$ | $\mathbb{R}^m$ | $r = \operatorname{rank}(A)$ |
| Null space $\operatorname{null}(A)$ | $\ker(T)$ | $\mathbb{R}^n$ | $n - r$ |
| Row space $\operatorname{row}(A)$ | $\operatorname{im}(T^\top)$ | $\mathbb{R}^n$ | $r$ |
| Left null space $\operatorname{null}(A^\top)$ | $\ker(T^\top)$ | $\mathbb{R}^m$ | $m - r$ |

**Orthogonality relations** (proven using the dual map — see §6):
$$\operatorname{null}(A) \perp \operatorname{row}(A), \quad \operatorname{null}(A^\top) \perp \operatorname{col}(A)$$

This splits $\mathbb{R}^n = \operatorname{row}(A) \oplus \operatorname{null}(A)$ and $\mathbb{R}^m = \operatorname{col}(A) \oplus \operatorname{null}(A^\top)$.

```
THE FOUR FUNDAMENTAL SUBSPACES
════════════════════════════════════════════════════════════════════════

  ℝⁿ (domain)                    ℝᵐ (codomain)
  ┌────────────────────┐         ┌────────────────────┐
  │ row space          │──T──→   │ column space       │
  │ (dim r)            │         │ (dim r)            │
  ├────────────────────┤         ├────────────────────┤
  │ null space         │──T──→   │ left null space    │
  │ (dim n-r)          │ {0}     │ (dim m-r)          │
  └────────────────────┘         └────────────────────┘
        ↕ ⊥                               ↕ ⊥
     orthogonal complement            orthogonal complement

════════════════════════════════════════════════════════════════════════
```

---

## 5. Special Classes of Linear Transformations

### 5.1 Projection Operators

**Definition.** A linear map $P: V \to V$ is a **projection** if it is **idempotent**: $P^2 = P$.

Idempotency means "doing it twice is the same as doing it once." Once you project onto a subspace, you're already there — projecting again does nothing.

**Theorem 5.1.1.** If $P$ is a projection, then:
- $\operatorname{im}(P) = \ker(I - P)$: the image of $P$ is the fixed-point set.
- $\ker(P) = \operatorname{im}(I - P)$: the kernel of $P$ is the image of the complementary projection.
- $V = \operatorname{im}(P) \oplus \ker(P)$ (direct sum decomposition).

**Proof of last claim:** Any $\mathbf{v} \in V$ decomposes as $\mathbf{v} = P\mathbf{v} + (I-P)\mathbf{v}$, where $P\mathbf{v} \in \operatorname{im}(P)$ and $(I-P)\mathbf{v} \in \ker(P)$ (since $P(I-P)\mathbf{v} = (P-P^2)\mathbf{v} = \mathbf{0}$). Uniqueness follows from: if $\mathbf{v} = \mathbf{u} + \mathbf{w}$ with $\mathbf{u} \in \operatorname{im}(P)$, $\mathbf{w} \in \ker(P)$, then $P\mathbf{v} = P\mathbf{u} + P\mathbf{w} = \mathbf{u}$. $\square$

**Rank and trace.** For a projection: $\operatorname{rank}(P) = \operatorname{tr}(P)$ (eigenvalues are only 0 and 1; trace = sum of eigenvalues = number of 1s = rank).

**Orthogonal vs oblique projections.** An **orthogonal projection** satisfies additionally $P = P^\top$ (i.e., $P$ is symmetric). In this case:
- $\ker(P) = \operatorname{im}(P)^\perp$ — the kernel is the orthogonal complement of the image.
- Formula: $P = A(A^\top A)^{-1}A^\top$ for projection onto $\operatorname{col}(A)$.

An **oblique projection** has $P^2 = P$ but $P \neq P^\top$ — it projects along a subspace that is not orthogonal to the target.

> **Forward reference: Orthogonal Projections**
>
> The full theory of orthogonal projections — including the projection formula, orthogonal decompositions, and the relationship to least squares — is in [§05: Orthogonality and Orthonormality](../05-Orthogonality-and-Orthonormality/notes.md).

**For AI:** Attention heads in transformers apply projections onto query/key/value subspaces. The head-specific projection matrices $W_Q, W_K, W_V$ define low-dimensional subspaces of the residual stream. The attention mechanism then computes weighted combinations within these projected spaces. Projection is also central to layer normalization (projecting onto the unit sphere in the mean-subtracted subspace).

### 5.2 Rotations and Reflections

**Orthogonal transformations** are linear maps $T: \mathbb{R}^n \to \mathbb{R}^n$ that preserve inner products:
$$\langle T(\mathbf{u}), T(\mathbf{v}) \rangle = \langle \mathbf{u}, \mathbf{v} \rangle \quad \text{for all } \mathbf{u}, \mathbf{v}$$

Equivalently, their matrices satisfy $A^\top A = I$, i.e., $A^{-1} = A^\top$.

The set of all $n \times n$ orthogonal matrices forms the **orthogonal group** $O(n)$.

**Determinant splits them into two classes:**
- $\det(A) = +1$: **proper rotations** — preserve orientation. These form $SO(n)$, the **special orthogonal group**.
- $\det(A) = -1$: **improper rotations** (reflections, or rotation + reflection) — reverse orientation.

**2D rotation by angle $\theta$:**
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \in SO(2)$$

Properties: $R_\theta R_\phi = R_{\theta+\phi}$, $R_\theta^{-1} = R_{-\theta} = R_\theta^\top$.

**3D rotations** are parametrized by an axis $\hat{\mathbf{n}}$ and angle $\theta$ (Rodrigues' formula):
$$R = I + \sin(\theta)\, K + (1 - \cos\theta)\, K^2$$
where $K$ is the skew-symmetric matrix with $K\mathbf{v} = \hat{\mathbf{n}} \times \mathbf{v}$.

**Householder reflections.** Given a unit vector $\hat{\mathbf{n}}$, the reflection through the hyperplane orthogonal to $\hat{\mathbf{n}}$ is:
$$H = I - 2\hat{\mathbf{n}}\hat{\mathbf{n}}^\top$$

Properties: $H = H^\top$, $H^2 = I$, $\det(H) = -1$.

**For AI:** Rotary Position Embedding (RoPE), used in LLaMA, GPT-NeoX, and Gemma, encodes positional information via rotation matrices applied blockwise to query and key vectors. The rotation angle depends on position, and relative positions appear as relative rotations, which interact cleanly with the dot-product attention score.

### 5.3 Shear and Scaling Maps

**Diagonal scaling:** $T(\mathbf{x}) = D\mathbf{x}$ where $D = \operatorname{diag}(d_1, \ldots, d_n)$. Scales each coordinate independently. Kernel = $\{\mathbf{0}\}$ if all $d_i \neq 0$. $\det(D) = \prod d_i$.

**Shear maps** in $\mathbb{R}^2$: the horizontal shear by factor $k$ is:
$$S_k = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}, \quad S_k(x, y) = (x + ky, y)$$

$\det(S_k) = 1$ — shear preserves area. $S_k^{-1} = S_{-k}$.

Geometrically: the $x$-axis is fixed; each horizontal line slides by a distance proportional to its height. Parallelograms are tilted but area is preserved.

**Elementary matrices** (from row operations) are all either shears, row swaps, or scalings:
- Row scaling by $c$: $E_{ii}(c)$ — multiply row $i$ by $c$
- Row swap: $E_{ij}$ — swap rows $i$ and $j$
- Row addition: $E_{ij}(c)$ — add $c$ times row $j$ to row $i$ (this is a shear)

Every invertible matrix is a product of elementary matrices — Gaussian elimination as composition of linear maps.

### 5.4 The Geometry of Low-Rank Maps

A rank-$k$ linear map $T: \mathbb{R}^n \to \mathbb{R}^m$ with $k < n$ **collapses** the domain: it maps all of $\mathbb{R}^n$ onto a $k$-dimensional subspace of $\mathbb{R}^m$, while sending an $(n-k)$-dimensional subspace (the null space) to zero.

**Decomposition via SVD preview.** Any rank-$k$ matrix admits:
$$A = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

This is a sum of $k$ rank-1 outer products. Each rank-1 term $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ maps everything in the direction $\mathbf{v}_i$ to the direction $\mathbf{u}_i$, scaled by $\sigma_i$.

> **Forward reference: SVD and Low-Rank Approximation**
>
> The full theory of SVD — including the Eckart-Young theorem (best rank-$k$ approximation) and the geometric interpretation of singular values — is in [§02: Singular Value Decomposition](../02-Singular-Value-Decomposition/notes.md).

**LoRA preview.** A rank-$r$ update $\Delta W = BA$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$) is a composition of two low-rank linear maps:
1. $A: \mathbb{R}^k \to \mathbb{R}^r$ compresses the input to $r$ dimensions.
2. $B: \mathbb{R}^r \to \mathbb{R}^d$ expands back to $d$ dimensions.

The effective map $\Delta W = BA$ has rank $\leq r$, so it only modifies the network's behavior along $r$ directions in the input space. The hypothesis underlying LoRA is that the task-relevant weight changes lie in a low-dimensional subspace — a statement directly about the geometry of linear maps.

---

## 6. Dual Spaces and Transposes

### 6.1 The Dual Space

**Definition.** Given a vector space $V$ over $\mathbb{F}$, the **dual space** $V^*$ is the set of all linear maps from $V$ to $\mathbb{F}$:
$$V^* = \mathcal{L}(V, \mathbb{F}) = \{f: V \to \mathbb{F} \mid f \text{ is linear}\}$$

Elements of $V^*$ are called **linear functionals** or **covectors** or **one-forms**.

$V^*$ is itself a vector space under pointwise addition $(f+g)(\mathbf{v}) = f(\mathbf{v}) + g(\mathbf{v})$ and scalar multiplication $(cf)(\mathbf{v}) = c \cdot f(\mathbf{v})$.

**The dual basis.** If $\mathcal{B} = \{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ is a basis for $V$, then the **dual basis** $\mathcal{B}^* = \{e^1, \ldots, e^n\}$ is defined by:
$$e^i(\mathbf{e}_j) = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

The dual basis is a basis for $V^*$, so $\dim(V^*) = \dim(V)$.

**Key example: Row vectors.** In $\mathbb{R}^n$, a linear functional is any map $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$ for some fixed $\mathbf{a} \in \mathbb{R}^n$. So $(\mathbb{R}^n)^* \cong \mathbb{R}^n$, but the identification $\mathbf{a}^\top$ (row vector) $\leftrightarrow$ $\mathbf{a}$ (column vector) is coordinate-dependent. A **row vector** $\mathbf{a}^\top$ is intrinsically an element of $(\mathbb{R}^n)^*$, not of $\mathbb{R}^n$.

**Why the distinction matters.** When you write $\mathbf{a}^\top \mathbf{x}$, you're applying the dual vector $\mathbf{a}^\top \in (\mathbb{R}^n)^*$ to the vector $\mathbf{x} \in \mathbb{R}^n$. This is a pairing between a space and its dual — not a dot product of two vectors in the same space. In Riemannian geometry and general relativity, this distinction is essential. In ML, it matters for understanding gradients.

### 6.2 The Dual Map and the Transpose

**Definition.** Given $T: V \to W$, the **dual map** (or **transpose**) $T^\top: W^* \to V^*$ is defined by:
$$(T^\top f)(\mathbf{v}) = f(T(\mathbf{v})) \quad \text{for } f \in W^*, \mathbf{v} \in V$$

In words: to apply $T^\top f$ to $\mathbf{v}$, first apply $T$ to $\mathbf{v}$, then apply $f$ to the result.

**Matrix of the dual map.** If $T$ has matrix $A$ in standard coordinates, then $T^\top$ has matrix $A^\top$ (the matrix transpose). The notation "$T^\top$" for the dual map is intentional and consistent.

**Domain and codomain switch.** $T: V \to W$ means $T^\top: W^* \to V^*$. The transpose reverses the direction. This is why:
- Composing on the left: $(ST)^\top = T^\top S^\top$ (order reversal).
- In backpropagation: if the forward pass multiplies by $W$, the backward pass multiplies by $W^\top$.

**Kernel of the transpose = left null space:**
$$\ker(T^\top) = \{\mathbf{w} \in W : T^\top \mathbf{w} = \mathbf{0}\} = \operatorname{null}(A^\top)$$

This is the **left null space** of $A$, which is orthogonal to $\operatorname{col}(A)$.

**Annihilators.** The **annihilator** of a subspace $S \subseteq V$ is $S^0 = \{f \in V^* : f(\mathbf{s}) = 0 \, \forall \mathbf{s} \in S\}$. Key identities:
$$\ker(T^\top) = (\operatorname{im}(T))^0, \qquad \operatorname{im}(T^\top) = (\ker(T))^0$$

### 6.3 Gradients Live in the Dual Space

This connection between linear maps and gradients is one of the deepest in all of applied mathematics, and is directly relevant to understanding backpropagation.

**The gradient as a linear functional.** For a differentiable function $f: \mathbb{R}^n \to \mathbb{R}$, the derivative at $\mathbf{x}$ is a linear functional $Df_{\mathbf{x}}: \mathbb{R}^n \to \mathbb{R}$, defined by:
$$Df_{\mathbf{x}}(\mathbf{h}) = \lim_{t \to 0} \frac{f(\mathbf{x} + t\mathbf{h}) - f(\mathbf{x})}{t} = \nabla f(\mathbf{x})^\top \mathbf{h}$$

The derivative $Df_{\mathbf{x}}$ is an element of $(\mathbb{R}^n)^*$ — a **covector**, represented as a row vector $\nabla f(\mathbf{x})^\top$.

The **gradient vector** $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ is obtained by identifying $(\mathbb{R}^n)^*$ with $\mathbb{R}^n$ via the standard inner product. This identification is coordinate-dependent: on a curved manifold (like a constraint surface or the space of probability distributions), gradients and vectors must be treated differently.

**Backpropagation via dual maps.** Consider a composed network $\mathbf{y} = W\mathbf{x}$ (one linear layer, loss $\ell = L(\mathbf{y})$). The gradient with respect to $\mathbf{x}$ is:
$$\frac{\partial \ell}{\partial \mathbf{x}} = W^\top \frac{\partial \ell}{\partial \mathbf{y}}$$

This is exactly the dual map $T^\top$ applied to the incoming gradient $\frac{\partial \ell}{\partial \mathbf{y}}$. Backpropagation propagates gradients backward through the transpose of each weight matrix. The chain of transposes in the backward pass is the dual map composition:
$$(W_L \cdots W_1)^\top = W_1^\top \cdots W_L^\top$$

**For AI:** Modern deep learning frameworks (PyTorch, JAX) compute gradients using automatic differentiation, which is precisely the evaluation of dual maps via the chain rule. Every `.backward()` call accumulates contributions via transpose weight matrices — it is applying the adjoint (dual) of the forward linear map.

---

## 7. Affine Transformations

### 7.1 Beyond Linearity: Affine Maps

A linear map fixes the origin: $T(\mathbf{0}) = \mathbf{0}$. But many practical transformations in geometry and ML need to shift the origin — they include a **translation** component.

**Definition.** A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is an **affine transformation** if it has the form:
$$f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$$
where $A \in \mathbb{R}^{m \times n}$ is a matrix (the **linear part**) and $\mathbf{b} \in \mathbb{R}^m$ is a vector (the **translation**).

Affine maps are linear maps composed with a translation. They are NOT linear (unless $\mathbf{b} = \mathbf{0}$) — they fail the zero test: $f(\mathbf{0}) = \mathbf{b} \neq \mathbf{0}$ when $\mathbf{b} \neq \mathbf{0}$.

**Affine subspaces.** The image of $\mathbb{R}^n$ under an affine map is an **affine subspace** — a translate of a linear subspace. Solutions to $A\mathbf{x} = \mathbf{b}$ (when they exist) form an affine subspace: $\mathbf{x}_0 + \ker(A)$.

**Composition of affine maps.** If $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ and $g(\mathbf{y}) = C\mathbf{y} + \mathbf{d}$, then:
$$(g \circ f)(\mathbf{x}) = g(A\mathbf{x} + \mathbf{b}) = C(A\mathbf{x} + \mathbf{b}) + \mathbf{d} = CA\mathbf{x} + (C\mathbf{b} + \mathbf{d})$$

The composition is affine: linear part $CA$, translation $C\mathbf{b} + \mathbf{d}$.

### 7.2 Homogeneous Coordinates

The elegant trick to make affine maps linear is to lift to one higher dimension.

**Definition.** The **homogeneous coordinates** of $\mathbf{x} \in \mathbb{R}^n$ are $\tilde{\mathbf{x}} = \begin{pmatrix} \mathbf{x} \\ 1 \end{pmatrix} \in \mathbb{R}^{n+1}$.

**The augmented matrix.** The affine map $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ becomes:
$$\tilde{f}(\tilde{\mathbf{x}}) = \begin{bmatrix} A & \mathbf{b} \\ \mathbf{0}^\top & 1 \end{bmatrix} \begin{pmatrix} \mathbf{x} \\ 1 \end{pmatrix} = \begin{pmatrix} A\mathbf{x} + \mathbf{b} \\ 1 \end{pmatrix}$$

Now $\tilde{f}$ is a **linear map** in $\mathbb{R}^{n+1}$. The last coordinate is always 1 for "proper" points.

**Key benefit: composition becomes matrix multiplication.** Two affine maps $\tilde{M}_1, \tilde{M}_2$ compose as:
$$\tilde{M}_2 \tilde{M}_1 = \begin{bmatrix} A_2 & \mathbf{b}_2 \\ \mathbf{0}^\top & 1 \end{bmatrix} \begin{bmatrix} A_1 & \mathbf{b}_1 \\ \mathbf{0}^\top & 1 \end{bmatrix} = \begin{bmatrix} A_2 A_1 & A_2\mathbf{b}_1 + \mathbf{b}_2 \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

No special cases needed — composition is just matrix multiplication.

**Inverse of an affine map:**
$$\begin{bmatrix} A & \mathbf{b} \\ \mathbf{0}^\top & 1 \end{bmatrix}^{-1} = \begin{bmatrix} A^{-1} & -A^{-1}\mathbf{b} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$

(Valid when $A$ is invertible.)

**In computer graphics:** Every rigid body transformation (rotation + translation) is an affine map, and sequences of transformations are composed by matrix multiplication in homogeneous coordinates. This is the foundation of 3D graphics pipelines (OpenGL, Vulkan).

### 7.3 Neural Network Layers as Affine Maps

A single fully-connected layer computes:
$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad \mathbf{a} = \sigma(\mathbf{z})$$

The pre-activation $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ is an **affine map** from $\mathbb{R}^{d_{\text{in}}}$ to $\mathbb{R}^{d_{\text{out}}}$. In homogeneous coordinates:
$$\tilde{\mathbf{z}} = \begin{bmatrix} W & \mathbf{b} \\ \mathbf{0}^\top & 1 \end{bmatrix} \tilde{\mathbf{x}}$$

**Why bias matters.** Without bias ($\mathbf{b} = \mathbf{0}$), each layer is linear and the hyperplanes separating classes must pass through the origin. With bias, the decision boundary can be placed anywhere. This is the geometric reason bias terms dramatically increase expressive power.

**Multi-layer without activations.** A network $\mathbf{z}_L = W_L(\cdots W_2(W_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 \cdots) + \mathbf{b}_L$ is still affine:
$$\mathbf{z}_L = W_{\text{eff}}\mathbf{x} + \mathbf{b}_{\text{eff}}$$

Multiple affine layers compose to a single affine layer. Depth without nonlinearity gives no expressive benefit.

**BatchNorm as affine rescaling.** After normalizing to zero mean and unit variance, BatchNorm applies a learned affine transformation $\gamma \mathbf{z} + \beta$ (elementwise). This is an affine map on the normalized activations, restoring the capacity to represent any desired scale and shift.

**Embedding layers.** An embedding layer $E \in \mathbb{R}^{|V| \times d}$ maps token indices to vectors. Selecting the $i$-th embedding is equivalent to multiplying by the one-hot vector $\mathbf{e}_i$: $E\mathbf{e}_i = \mathbf{e}_{i,:}$ (the $i$-th row). This is linear in the one-hot representation. The unembedding $W_U \in \mathbb{R}^{d \times |V|}$ maps representations to logits: $\mathbf{l} = W_U \mathbf{h}$, a pure linear map.

---

## 8. The Jacobian as Linear Approximation

### 8.1 Linearization of Nonlinear Maps

Every differentiable function is "locally linear" — at any point, it looks like a linear map to first order. This principle underlies calculus, optimization, and all of numerical analysis.

**Definition (Total Derivative).** A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **differentiable** at $\mathbf{x}$ if there exists a linear map $Df_{\mathbf{x}}: \mathbb{R}^n \to \mathbb{R}^m$ such that:
$$\lim_{\lVert\mathbf{h}\rVert \to 0} \frac{\lVert f(\mathbf{x} + \mathbf{h}) - f(\mathbf{x}) - Df_{\mathbf{x}}(\mathbf{h})\rVert}{\lVert\mathbf{h}\rVert} = 0$$

The linear map $Df_{\mathbf{x}}$ is the **total derivative** or **Fréchet derivative** of $f$ at $\mathbf{x}$. The first-order approximation is:
$$f(\mathbf{x} + \mathbf{h}) \approx f(\mathbf{x}) + Df_{\mathbf{x}}(\mathbf{h})$$

For small $\mathbf{h}$, the function looks like an affine map: constant $f(\mathbf{x})$ plus linear correction $Df_{\mathbf{x}}(\mathbf{h})$.

**Geometric meaning.** At each point $\mathbf{x}$, the total derivative $Df_{\mathbf{x}}$ is the best linear approximation to $f$ near $\mathbf{x}$. It maps directions (tangent vectors at $\mathbf{x}$) to directions (tangent vectors at $f(\mathbf{x})$). This is the **pushforward** of vectors.

### 8.2 The Jacobian Matrix

**Definition.** The matrix representation of $Df_{\mathbf{x}}$ in standard coordinates is the **Jacobian matrix**:
$$J_f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Entry $(i, j)$: $J_{ij} = \frac{\partial f_i}{\partial x_j}$ — how the $i$-th output changes with the $j$-th input.

**Shape mnemonic:** output dimension $\times$ input dimension: $(m \times n)$ for $f: \mathbb{R}^n \to \mathbb{R}^m$.

**Special cases:**
- $f: \mathbb{R}^n \to \mathbb{R}$ (scalar function): Jacobian is the row gradient $\nabla f^\top \in \mathbb{R}^{1 \times n}$.
- $f: \mathbb{R} \to \mathbb{R}^m$ (curve): Jacobian is the column tangent vector $f'(t) \in \mathbb{R}^m$.
- $f: \mathbb{R}^n \to \mathbb{R}^n$ (same dim): $J_f$ is square; $\det(J_f)$ is the local volume scaling.

**The Jacobian of softmax.** For $\mathbf{s} = \operatorname{softmax}(\mathbf{z})$ where $s_i = e^{z_i} / \sum_k e^{z_k}$:
$$J_{\operatorname{softmax}}(\mathbf{z}) = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

This is a rank-deficient matrix (rank $n-1$) because softmax outputs sum to 1: the Jacobian has a constant vector $\mathbf{1}$ in its null space.

**The Jacobian of ReLU.** For $\mathbf{a} = \operatorname{ReLU}(\mathbf{z})$ (elementwise):
$$J_{\operatorname{ReLU}}(\mathbf{z}) = \operatorname{diag}(\mathbf{1}[\mathbf{z} > 0])$$

A diagonal matrix with 1s where $z_i > 0$ and 0s elsewhere. ReLU's Jacobian is a projection — it zeroes out the gradient for dead neurons.

**The Jacobian of layer normalization.** More complex: layer norm applies mean subtraction, variance normalization, and an affine transformation. Its Jacobian is a projection onto a specific hyperplane (orthogonal to the constant vector), scaled and shifted.

### 8.3 Chain Rule = Composition of Jacobians

The **chain rule** for vector-valued functions states: if $f: \mathbb{R}^n \to \mathbb{R}^m$ and $g: \mathbb{R}^m \to \mathbb{R}^p$, then:
$$J_{g \circ f}(\mathbf{x}) = J_g(f(\mathbf{x})) \cdot J_f(\mathbf{x})$$

This is **matrix multiplication** of Jacobians. The composition of two differentiable maps has a Jacobian equal to the matrix product of their individual Jacobians (evaluated at the appropriate points).

**Backpropagation is reverse-mode Jacobian accumulation.** For a network $\mathbf{y} = f_L \circ \cdots \circ f_1(\mathbf{x})$ with loss $\ell = L(\mathbf{y})$:

$$\frac{\partial \ell}{\partial \mathbf{x}} = J_{f_1}^\top \cdots J_{f_L}^\top \nabla_{\mathbf{y}} L$$

Reading right to left: start with $\nabla_{\mathbf{y}} L$, multiply by $J_{f_L}^\top$, then $J_{f_{L-1}}^\top$, etc. At each step, we multiply by the **transpose Jacobian** of a layer — which is the **dual map** of that layer's linear approximation.

**Computational graph view.** Each node in the computation graph stores its local Jacobian. Forward pass evaluates the functions; backward pass multiplies the transpose Jacobians (right to left). This is the mathematical content of `autograd`.

```
BACKPROPAGATION AS JACOBIAN CHAIN
════════════════════════════════════════════════════════════════════════

  Forward:  x ──J₁──→ z₁ ──J₂──→ z₂ ── ... ──J_L──→ y ──→ ℓ

  Backward: ∂ℓ/∂x ←──J₁ᵀ── ∂ℓ/∂z₁ ←──J₂ᵀ── ∂ℓ/∂z₂ ←── ... ←──J_Lᵀ── ∇y L

  Each backward step:  (grad at input) = Jᵀ × (grad at output)
                     = (dual map) applied to incoming gradient

════════════════════════════════════════════════════════════════════════
```

**Why Jacobians matter for training dynamics:**
- Large Jacobian singular values → exploding gradients
- Small Jacobian singular values → vanishing gradients
- The spectral norm of $J_f$ measures how much the linear approximation of $f$ can amplify inputs

**For AI:** Gradient clipping, careful weight initialization (Xavier, He), and residual connections all address the Jacobian conditioning problem. Residual connections add an identity Jacobian contribution: $J_{x + F(x)} = I + J_F$, which prevents the singular values from collapsing to zero (vanishing gradients).

---

## 9. Applications in Machine Learning

### 9.1 Attention as Linear Projections

The scaled dot-product attention mechanism (Vaswani et al., 2017) is built entirely from linear transformations applied to a sequence of token representations.

**Setup.** Given an input sequence $X \in \mathbb{R}^{L \times d}$ (L tokens, d-dimensional residual stream), three learned projection matrices $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$ define:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Each is a linear transformation: $Q$ projects each token into "query space", $K$ into "key space", $V$ into "value space".

**Attention scores.** The attention pattern $\alpha = \operatorname{softmax}(QK^\top / \sqrt{d_k})$ is a soft selection matrix. For a fixed query vector $\mathbf{q}$, the attention scores $\mathbf{q}^\top K^\top$ are dot products in key space — a linear functional applied to each key.

**Output.** $\operatorname{Attn}(Q,K,V) = \alpha V$. The output is a **weighted linear combination** of value vectors — a linear operation parameterized by $\alpha$. The output projection $W_O \in \mathbb{R}^{d_v \times d}$ maps back to the residual stream: another linear map.

**Multi-head attention.** Each head $h$ applies its own projection matrices $(W_Q^h, W_K^h, W_V^h)$, computes attention independently, and the results are concatenated and projected:
$$\operatorname{MHA}(X) = \operatorname{Concat}(\operatorname{head}_1, \ldots, \operatorname{head}_H) W_O$$

This is a composition and concatenation of linear maps. The expressivity comes from the multiplicative interaction $QK^\top$ (which is bilinear, not linear) — but conditioned on fixed $\alpha$, the rest is linear.

**For AI:** The "OV circuit" and "QK circuit" decomposition (Elhage et al., 2021) analyzes transformer attention by studying the linear maps $W_OW_V$ (value writing) and $W_QW_K^\top$ (key-query matching) separately. This is possible precisely because attention is compositionally linear.

### 9.2 LoRA: Fine-Tuning via Low-Rank Composition

Low-Rank Adaptation (Hu et al., 2021) is one of the most important parameter-efficient fine-tuning methods, and its design is a direct application of linear map theory.

**Setup.** Freeze the pretrained weight $W_0 \in \mathbb{R}^{d \times k}$. Add a trainable low-rank update:
$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

**Rank-nullity interpretation.** The map $\Delta W = BA$ has rank at most $r$. By rank-nullity:
$$\dim(\ker(\Delta W)) \geq k - r$$

The update only changes the network's behavior along at most $r$ directions in the $k$-dimensional input space. When $r = 8$ and $k = 4096$, only $8/4096 \approx 0.2\%$ of input directions are affected — the update is extremely sparse in "direction space."

**The two-layer interpretation.** The update $\Delta W = BA$ is a composition of two maps:
1. $A: \mathbb{R}^k \to \mathbb{R}^r$ — compression to rank-$r$ space
2. $B: \mathbb{R}^r \to \mathbb{R}^d$ — expansion back to full dimension

Initializing $A$ randomly and $B = 0$ ensures $\Delta W = 0$ at the start of fine-tuning, so the model starts from the pretrained weights.

**Parameter count.** $\Delta W$ would have $dk$ parameters. $BA$ uses $dr + rk = r(d+k)$ parameters. Savings ratio: $\frac{r(d+k)}{dk} = \frac{r}{d} + \frac{r}{k} \approx \frac{2r}{\min(d,k)}$. For $r=8$, $d=k=4096$: savings factor of $\approx 256\times$.

**Extensions:** DoRA (Weight-Decomposition Low-Rank Adaptation, 2024) decomposes $W$ into magnitude and direction, applying LoRA only to the direction component. GaLore (2024) applies LoRA-style updates to gradients rather than weights.

### 9.3 Linear Probes and the Linear Representation Hypothesis

**Linear probing** tests whether a feature is linearly decodable from a model's representations. Given representations $\{\mathbf{h}_i\}$ and labels $\{y_i\}$, train a linear classifier:
$$\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{h} + b)$$

If the probe achieves high accuracy, the feature is **linearly represented** — it corresponds to a direction $\mathbf{w}$ in representation space.

**The Linear Representation Hypothesis** (Mikolov et al., 2013; Elhage et al., 2022; Park et al., 2023) states that high-level features (sentiment, syntax, factual attributes, world models) are encoded as **directions** in the residual stream — i.e., as linear features.

**Evidence:**
- Word2vec arithmetic: $\mathbf{v}(\text{king}) - \mathbf{v}(\text{man}) + \mathbf{v}(\text{woman}) \approx \mathbf{v}(\text{queen})$. Semantic relationships are linear offsets.
- Steering vectors: adding $c\mathbf{d}$ to all residual stream activations controls model behavior (e.g., "banana" direction, sentiment directions).
- Probing studies: most tested syntactic and semantic features are linearly decodable.

**Superposition.** When there are more features than dimensions, the model stores features in near-orthogonal directions that partially overlap (superposition). This is still linear representation — just with interference.

**For AI:** If the linear representation hypothesis holds broadly, then:
- Linear algebra provides the right toolkit for model interpretability.
- Interventions on model behavior reduce to vector addition in representation space.
- Feature extraction is a linear map — PCA/SVD on activations finds meaningful directions.

### 9.4 Embedding and Unembedding

**Token embeddings.** The embedding matrix $W_E \in \mathbb{R}^{|V| \times d}$ maps vocabulary indices to $d$-dimensional vectors. Indexing row $i$ of $W_E$ is equivalent to the linear map $\mathbf{e}_i^\top W_E$ (one-hot selection). The embedding layer is linear in the one-hot representation.

**Unembedding.** The unembedding matrix $W_U \in \mathbb{R}^{d \times |V|}$ maps residual stream vectors to logits over the vocabulary:
$$\mathbf{l} = W_U \mathbf{h} \in \mathbb{R}^{|V|}$$

This is a pure linear map. The logit for token $v$ is $l_v = \mathbf{w}_{U,v}^\top \mathbf{h}$ — a dot product (linear functional) between the unembedding direction $\mathbf{w}_{U,v}$ and the residual stream.

**Logit lens.** Applying $W_U$ to intermediate residual stream states (before the final layer) gives "early predictions" — showing what the model is computing at each layer. This technique (Nostalgebraist, 2020) is possible because unembedding is linear.

**Tied embeddings.** Many models (GPT-2, LLaMA variants) use $W_U = W_E^\top$ — the same matrix for both embedding and unembedding. This enforces consistency: the most likely next token $v$ after seeing context $\mathbf{h}$ is the one whose embedding $W_E[v]$ has the highest dot product with $\mathbf{h}$ — i.e., $\operatorname{argmax}_v W_E[v] \cdot \mathbf{h}$.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | **Assuming $T(\mathbf{0}) \neq \mathbf{0}$ is possible for a linear map** | Homogeneity immediately gives $T(\mathbf{0}) = T(0\cdot\mathbf{v}) = 0\cdot T(\mathbf{v}) = \mathbf{0}$. Any map with $T(\mathbf{0}) \neq \mathbf{0}$ is affine or nonlinear. | The zero-test is the fastest way to rule out linearity. |
| 2 | **Treating translation as linear** | $f(\mathbf{x}) = \mathbf{x} + \mathbf{b}$ fails additivity: $f(\mathbf{u}+\mathbf{v}) = \mathbf{u}+\mathbf{v}+\mathbf{b} \neq f(\mathbf{u})+f(\mathbf{v}) = \mathbf{u}+\mathbf{v}+2\mathbf{b}$ | Neural network layers are affine ($W\mathbf{x}+\mathbf{b}$), not linear. This matters for composing and inverting. |
| 3 | **Confusing rank of a map with rank of a matrix** | Rank is basis-independent (it's the dimension of the image) but the matrix depends on the basis. | $\operatorname{rank}(T) = \operatorname{rank}([T]_{\mathcal{B}}^{\mathcal{C}})$ for any bases — rank is a map invariant, not a matrix property. |
| 4 | **Misapplying rank-nullity** | Forgetting that rank-nullity applies to the domain dimension, not the codomain. For $T: \mathbb{R}^5 \to \mathbb{R}^3$, rank + nullity = 5, not 3. | Identify $\dim(V)$ explicitly before applying the theorem. |
| 5 | **Assuming $AB = BA$** | Matrix multiplication (= composition of linear maps) is generally not commutative. Even for square matrices, $AB \neq BA$ in general. | Non-commutativity is the default. Commutativity (e.g., diagonal matrices, polynomial functions of a matrix) is the special case. |
| 6 | **Confusing similar matrices with equal matrices** | $A \sim B$ means they represent the same map in different bases — same eigenvalues, trace, determinant. But $A \neq B$ in general. | Similar matrices are equal only if the change-of-basis matrix $P$ is the identity. |
| 7 | **Thinking the kernel is trivial when $A$ is tall** | A tall matrix $A \in \mathbb{R}^{m \times n}$ with $m > n$ can still have a non-trivial null space if its columns are linearly dependent. | Compute $\operatorname{rank}(A)$. Nullity = $n - \operatorname{rank}(A)$, regardless of whether $m > n$. |
| 8 | **Applying the inverse when only a one-sided inverse exists** | A non-square $A \in \mathbb{R}^{m \times n}$ ($m \neq n$) cannot have a two-sided inverse. Attempting $A^{-1}$ for such matrices is undefined. | Use the pseudo-inverse $A^+$ for the least-squares solution. |
| 9 | **Forgetting that the Jacobian is a linear map, not just partial derivatives** | The Jacobian matrix is the coordinate representation of the total derivative $Df_{\mathbf{x}}$. Partial derivatives individually give the columns but don't by themselves prove differentiability. | Differentiability requires the linear approximation error to go to zero — this requires the partials to exist AND be continuous. |
| 10 | **Treating the gradient as a vector in the same space** | Strictly, $\nabla f(\mathbf{x}) \in (\mathbb{R}^n)^*$ (the dual space). In Euclidean space with standard basis, the identification $(\mathbb{R}^n)^* \cong \mathbb{R}^n$ makes this invisible. But for optimization on manifolds or with non-Euclidean metrics, treating gradients as primal vectors gives wrong answers. | Use the gradient as a covector when working with Fisher information, natural gradient, or Riemannian optimization. |
| 11 | **Assuming all bijective functions are linear isomorphisms** | A function can be bijective but not linear. E.g., $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^3$ is bijective but not linear ($f(1+1) = 8 \neq f(1) + f(1) = 2$). | Isomorphisms must be both bijective AND linear. Check both conditions. |
| 12 | **Forgetting that $\operatorname{im}(T)$ is in $W$, not in $V$** | The image is a subspace of the *codomain* $W$, not the domain $V$. The null space is in $V$. | Draw the diagram: $T: V \to W$. $\ker(T) \subseteq V$. $\operatorname{im}(T) \subseteq W$. |

---

## 11. Exercises

### Exercise 1 ★ — Kernel and Image Computation

**Goal:** Given explicit matrices, compute kernel and image and verify rank-nullity.

**(a)** For $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$: find a basis for $\ker(A)$, a basis for $\operatorname{im}(A)$, and verify rank + nullity = 3.

**(b)** For $B = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$ (projection to first two coordinates): identify kernel and image without computation.

**(c)** For the differentiation map $D: \mathcal{P}_3 \to \mathcal{P}_2$ with matrix as in §3.1: find kernel and image dimensions.

### Exercise 2 ★ — Matrix of a Linear Map in Non-Standard Basis

**Goal:** Construct the matrix of a given transformation in a specified basis.

Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ be reflection across the line $y = x$ (i.e., $T(x,y) = (y,x)$). Basis $\mathcal{B} = \{(1,1)/\sqrt{2}, (1,-1)/\sqrt{2}\}$.

**(a)** Write the standard matrix $A$ of $T$ in the standard basis.

**(b)** Compute the change-of-basis matrix $P$ from $\mathcal{B}$ to the standard basis.

**(c)** Compute $[T]_{\mathcal{B}} = P^{-1}AP$. What is notable about this result?

**(d)** Explain geometrically why $T$ is diagonal in the basis $\mathcal{B}$.

### Exercise 3 ★ — Rank-Nullity and Linear Systems

**Goal:** Use rank-nullity to understand solution sets of linear systems.

**(a)** $A \in \mathbb{R}^{4 \times 6}$ has rank 3. What is the nullity? How many free variables are there in $A\mathbf{x} = \mathbf{0}$?

**(b)** $B \in \mathbb{R}^{6 \times 4}$ has rank 4. Is $B\mathbf{x} = \mathbf{b}$ always solvable? What is the nullity of $B$?

**(c)** Prove: if $T: V \to V$ is a linear map on a finite-dimensional space, then $T$ is injective if and only if $T$ is surjective.

### Exercise 4 ★★ — Projection Operator Construction

**Goal:** Build an orthogonal projection onto a given subspace and verify idempotency.

Let $S = \operatorname{span}\{(1, 2, 0), (0, 1, 1)\} \subseteq \mathbb{R}^3$.

**(a)** Orthonormalize the spanning set using Gram-Schmidt.

**(b)** Construct the projection matrix $P = QQ^\top$ where $Q$ has the orthonormal basis as columns.

**(c)** Verify: $P^2 = P$, $P = P^\top$, and $\operatorname{rank}(P) = \operatorname{tr}(P) = 2$.

**(d)** Compute $(I-P)$ and verify it is also a projection. What does it project onto?

### Exercise 5 ★★ — Jacobian of Softmax

**Goal:** Derive and verify the Jacobian of the softmax function.

**(a)** Derive $\frac{\partial s_i}{\partial z_j}$ for $\mathbf{s} = \operatorname{softmax}(\mathbf{z})$, handling the cases $i = j$ and $i \neq j$ separately.

**(b)** Show that the Jacobian is $J = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$.

**(c)** Verify that $J\mathbf{1} = \mathbf{0}$ (the Jacobian kills constant vectors). Why does this make sense?

**(d)** Show that the rank of $J$ is at most $n-1$.

### Exercise 6 ★★ — Affine Map Composition via Homogeneous Coordinates

**Goal:** Compose affine transformations using augmented matrices.

In $\mathbb{R}^2$, let:
- $f_1$: rotation by $45°$ then translate by $(1, 0)$
- $f_2$: scale by $2$ in both dimensions then translate by $(0, 1)$

**(a)** Write augmented $3 \times 3$ matrices $\tilde{M}_1$ and $\tilde{M}_2$ for $f_1$ and $f_2$.

**(b)** Compute $\tilde{M}_2 \tilde{M}_1$ (apply $f_1$ then $f_2$). What are the effective rotation, scale, and translation?

**(c)** Apply the composition to the point $(1, 0)$. Verify by applying $f_1$ then $f_2$ directly.

### Exercise 7 ★★★ — LoRA Rank Analysis

**Goal:** Analyze the geometry of low-rank weight updates.

Let $W_0 \in \mathbb{R}^{512 \times 512}$ be a pretrained weight. A LoRA update with rank $r = 4$ is $\Delta W = BA$ where $B \in \mathbb{R}^{512 \times 4}$, $A \in \mathbb{R}^{4 \times 512}$.

**(a)** What is $\operatorname{rank}(\Delta W)$? What is the nullity of $\Delta W$?

**(b)** Generate random $B$ and $A$ and numerically verify: $\operatorname{rank}(BA) \leq 4$.

**(c)** For a fixed input $\mathbf{x} \in \mathbb{R}^{512}$, compute $\Delta W \mathbf{x} = BA\mathbf{x}$. Show this is in $\operatorname{col}(B)$.

**(d)** Compute the singular values of $\Delta W$. How many are nonzero? What does this tell you about the effective dimension of the update?

**(e)** Compare the number of trainable parameters: $512 \times 512$ vs $r(512 + 512)$ for $r = 4, 8, 16, 32$.

### Exercise 8 ★★★ — Linear Probing and the Linear Representation Hypothesis

**Goal:** Empirically test whether a concept is linearly represented in embedding space.

**(a)** Generate synthetic embeddings for binary sentiment: positive embeddings clustered near $\mathbf{d}$, negative near $-\mathbf{d}$ for a fixed direction $\mathbf{d}$, plus noise.

**(b)** Train a linear probe (logistic regression on top of embeddings) and measure accuracy.

**(c)** Apply PCA to the embeddings. Show that the first principal component aligns with $\mathbf{d}$.

**(d)** Add a "superposition" condition: embed two independent binary features in $d = 3$ dimensions. Show that both features can be linearly decoded but with interference.

**(e)** Compute the mutual coherence of the feature directions. How does it relate to probe accuracy?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI Application | Why It Matters |
| --- | --- | --- |
| Linear map axioms | Neural layer computation | Every forward pass is a composition of linear maps + activations; understanding linearity separates what the layer does from what nonlinearity adds |
| Kernel and image | Information compression | The null space of a weight matrix is "dead information" — inputs in $\ker(W)$ produce no signal. Attention heads have low-rank structure that defines their effective null space |
| Rank-nullity theorem | LoRA, model compression | LoRA exploits rank-nullity: a rank-$r$ update has $(k-r)$-dimensional null space, leaving most of the input space unaffected — this is why it works with few parameters |
| Change of basis | Diagonalization, eigenfeatures | The eigenvalue decomposition of weight matrices (studied in mechanistic interpretability) is a change to the "natural basis" in which the layer acts by simple scaling |
| Composition = multiplication | Deep network analysis | The effective weight of a $k$-layer linear network is one matrix $W_L\cdots W_1$ — depth without nonlinearity has zero benefit. All depth benefits require nonlinearity |
| Projection operators | Attention heads, layer norm | Attention heads project onto query/key/value subspaces; layer norm projects onto the hyperplane of mean-zero vectors; understanding projections clarifies what information is preserved |
| Affine maps + bias | Universal approximation | Bias terms are essential for shifting decision boundaries. Without bias, the model cannot represent affine functions — only linear ones. Universal approximation requires affine layers |
| Jacobian and chain rule | Backpropagation | Every `.backward()` call is Jacobian-matrix multiplication via the chain rule. Gradient explosion/vanishing is about Jacobian singular value growth/decay through layers |
| Dual maps and transposes | Gradient computation | The backward pass uses transpose weight matrices — these are the dual maps. Natural gradient and Fisher information matrix methods exploit the geometry of the dual space |
| Linear representation hypothesis | Mechanistic interpretability | If features are linear, activation patching, steering vectors, and linear probing all work. This is why "linear algebra for interpretability" (e.g., representation engineering, logit lens) is a coherent research program |

---

## 13. Conceptual Bridge

### Looking Back

This section builds on two foundational pillars from earlier in the curriculum.

**From [Chapter 2 §06: Vector Spaces and Subspaces](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md):** We studied the axioms of abstract vector spaces (closure, associativity, identity, inverse, distributivity) and their subspaces (spans, null spaces, column spaces). Linear transformations are the *maps between* these abstract structures — the morphisms of the category of vector spaces. The four fundamental subspaces (column space, null space, row space, left null space) that were defined for matrices are now understood as $\operatorname{im}(T)$, $\ker(T)$, $\operatorname{im}(T^\top)$, and $\ker(T^\top)$ — intrinsic properties of the map, not the matrix.

**From [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md) and [§02: SVD](../02-Singular-Value-Decomposition/notes.md):** Both eigendecomposition and SVD are studied as **decompositions of linear maps into simple pieces**. Eigendecomposition finds a basis in which $T$ acts by scaling. SVD finds two orthonormal bases (for domain and codomain) in which $T$ acts by scaling. These are special cases of the general change-of-basis machinery developed here in §3.

**From [§03: PCA](../03-Principal-Component-Analysis/notes.md):** PCA uses the linear structure of the data covariance matrix — the covariance $C = X^\top X / (n-1)$ is built from linear maps — to find the principal directions via SVD. The whitening transform and PCA projection are linear maps, and their geometry (dimension reduction, preserved variance) follows directly from the rank-nullity theorem.

### Looking Forward

The abstract machinery of linear transformations will appear throughout the rest of the curriculum in concrete technical forms.

**[§05: Orthogonality and Orthonormality](../05-Orthogonality-and-Orthonormality/notes.md):** Gram-Schmidt is an algorithm that constructs a specific linear map (change of basis to an orthonormal basis). QR decomposition $A = QR$ factors a linear map as an orthogonal map $Q$ followed by a triangular map $R$. Orthogonal projections (§5.1 here) are studied in depth there.

**[§06: Matrix Norms](../06-Matrix-Norms/notes.md):** The spectral norm $\lVert A \rVert_2 = \sigma_1(A)$ measures how much a linear map can amplify vectors. The nuclear norm $\lVert A \rVert_* = \sum \sigma_i$ measures the "effective rank." These norms quantify properties of the linear map that are invisible from the matrix entries.

**[Chapter 4: Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md):** The Jacobian (§8 here) is the bridge between linear algebra and calculus. Multivariable calculus is essentially the study of how linear maps (Jacobians) approximate smooth nonlinear maps. The Hessian is the second-order analogue — a bilinear map that measures curvature. The implicit function theorem, inverse function theorem, and change-of-variables formula all rely on the Jacobian being an invertible linear map at the point of interest.

### Curriculum Position

```
POSITION IN THE MATH FOR LLMS CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 1: Mathematical Foundations
  Chapter 2: Linear Algebra Basics
    ├── Vectors, Matrix Operations, Systems of Equations
    ├── Determinants, Matrix Rank
    └── Vector Spaces ────────────────────────────┐
                                                  │ prerequisite
  Chapter 3: Advanced Linear Algebra              │
    ├── 01-Eigenvalues ──────────────────────────┤
    ├── 02-SVD ─────────────────────────────────┤
    ├── 03-PCA ─────────────────────────────────┤
    │                                             │
    ├── 04-Linear Transformations ◄──────────────┘
    │      (YOU ARE HERE)
    │      Kernel, Image, Rank-Nullity
    │      Matrix Representation, Change of Basis
    │      Composition, Isomorphisms, Jacobian
    │      Affine Maps, Dual Spaces, AI Applications
    │                                             │
    ├── 05-Orthogonality ←─────────────────────┤
    ├── 06-Matrix Norms ←──────────────────────┤
    ├── 07-Positive Definite Matrices           │
    └── 08-Matrix Decompositions               │
                                               ↓
  Chapter 4: Calculus Fundamentals
    (Jacobians and chain rule developed further)

════════════════════════════════════════════════════════════════════════
```

**The unifying theme:** Every major algorithm in deep learning is a composition of linear maps and nonlinearities. Understanding this section means understanding the *language* in which every neural network, every attention mechanism, every gradient computation, and every interpretability method is written. The matrix is not the territory — but knowing how to move between coordinate representations (change of basis), how to measure what a map collapses (kernel and rank), how to compose maps (matrix multiplication), and how to invert them (isomorphisms and pseudo-inverses) gives you the full algebraic toolkit to reason about any linear system you encounter in machine learning.

---

[← PCA](../03-Principal-Component-Analysis/notes.md) | [Back to Advanced Linear Algebra](../README.md) | [Orthogonality →](../05-Orthogonality-and-Orthonormality/notes.md)

---

## Appendix A: Extended Examples and Computations

### A.1 Computing the Matrix of Differentiation

We work out the differentiation operator in full detail to build intuition for linear maps on function spaces.

**Setting.** Let $V = \mathcal{P}_3 = \{a_0 + a_1 x + a_2 x^2 + a_3 x^3\}$ with basis $\mathcal{B}_V = \{1, x, x^2, x^3\}$ and $W = \mathcal{P}_2$ with basis $\mathcal{B}_W = \{1, x, x^2\}$.

Define $D: V \to W$ by $D(p) = p'$ (differentiation).

**Step 1: Apply $D$ to each basis vector of $V$.**
- $D(1) = 0 = 0 \cdot 1 + 0 \cdot x + 0 \cdot x^2$ → coordinate vector $(0, 0, 0)^\top$
- $D(x) = 1 = 1 \cdot 1 + 0 \cdot x + 0 \cdot x^2$ → coordinate vector $(1, 0, 0)^\top$
- $D(x^2) = 2x = 0 \cdot 1 + 2 \cdot x + 0 \cdot x^2$ → coordinate vector $(0, 2, 0)^\top$
- $D(x^3) = 3x^2 = 0 \cdot 1 + 0 \cdot x + 3 \cdot x^2$ → coordinate vector $(0, 0, 3)^\top$

**Step 2: Assemble into matrix.**
$$[D]_{\mathcal{B}_V}^{\mathcal{B}_W} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix}$$

**Step 3: Use the matrix.** Differentiate $p(x) = 2 + 3x - x^2 + 4x^3$. In $\mathcal{B}_V$-coordinates: $[p]_{\mathcal{B}_V} = (2, 3, -1, 4)^\top$.

$$[D(p)]_{\mathcal{B}_W} = \begin{bmatrix} 0 & 1 & 0 & 0 \\ 0 & 0 & 2 & 0 \\ 0 & 0 & 0 & 3 \end{bmatrix} \begin{pmatrix} 2 \\ 3 \\ -1 \\ 4 \end{pmatrix} = \begin{pmatrix} 3 \\ -2 \\ 12 \end{pmatrix}$$

So $D(p) = 3 - 2x + 12x^2$. Check: $(2 + 3x - x^2 + 4x^3)' = 3 - 2x + 12x^2$. ✓

**Kernel of $D$:** The null space is $\{p : p' = 0\} = \{\text{constants}\} = \operatorname{span}\{1\}$. Dimension 1.

**Image of $D$:** All polynomials of degree $\leq 2$ (since every $q \in \mathcal{P}_2$ equals $(p)'$ for some $p \in \mathcal{P}_3$). Dimension 3.

**Rank-nullity check:** $\dim(\mathcal{P}_3) = 4 = 1 + 3 = \text{nullity} + \text{rank}$. ✓

### A.2 Change of Basis: Full Worked Example

**Problem.** Let $T: \mathbb{R}^2 \to \mathbb{R}^2$ rotate vectors by $30°$ counterclockwise. Express $T$ in the rotated basis $\mathcal{B}' = \{R_{45°}\mathbf{e}_1, R_{45°}\mathbf{e}_2\}$.

**Step 1: Standard matrix of $T$.**
$$A = R_{30°} = \begin{bmatrix} \cos 30° & -\sin 30° \\ \sin 30° & \cos 30° \end{bmatrix} = \begin{bmatrix} \sqrt{3}/2 & -1/2 \\ 1/2 & \sqrt{3}/2 \end{bmatrix}$$

**Step 2: Change-of-basis matrix.** The new basis vectors, in standard coordinates:
$$\mathbf{b}'_1 = R_{45°}\mathbf{e}_1 = (\cos 45°, \sin 45°)^\top = (1/\sqrt{2}, 1/\sqrt{2})^\top$$
$$\mathbf{b}'_2 = R_{45°}\mathbf{e}_2 = (-\sin 45°, \cos 45°)^\top = (-1/\sqrt{2}, 1/\sqrt{2})^\top$$

$$P = \begin{bmatrix} 1/\sqrt{2} & -1/\sqrt{2} \\ 1/\sqrt{2} & 1/\sqrt{2} \end{bmatrix} = R_{45°}$$

**Step 3: New matrix.**
$$[T]_{\mathcal{B}'} = P^{-1} A P = R_{-45°} R_{30°} R_{45°}$$

Since rotation matrices commute (they all rotate around the same axis in 2D): $R_{-45°} R_{30°} R_{45°} = R_{30°}$.

**Key insight:** A rotation in 2D has the same matrix in every orthonormal basis (since all such matrices are just $R_\theta$). This is because $R_\theta$ commutes with all rotations: $R_\alpha R_\theta R_\alpha^{-1} = R_\theta$.

### A.3 The Null Space as a Subspace: Visualization

For $A = \begin{bmatrix} 1 & 2 & -1 \\ 2 & 4 & -2 \end{bmatrix}$ (rows are multiples), let's find $\ker(A)$.

Row reduce: $R_2 \leftarrow R_2 - 2R_1$:
$$\begin{bmatrix} 1 & 2 & -1 \\ 0 & 0 & 0 \end{bmatrix}$$

Free variables: $x_2 = s$, $x_3 = t$ (free). Back-substitute: $x_1 = -2s + t$.

$$\ker(A) = \left\{ s \begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix} + t \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} : s, t \in \mathbb{R} \right\}$$

This is a **plane** through the origin in $\mathbb{R}^3$. The map $T_A: \mathbb{R}^3 \to \mathbb{R}^2$ collapses this entire plane to zero. By rank-nullity: rank $= 1$, nullity $= 2$, and $1 + 2 = 3 = \dim(\mathbb{R}^3)$. ✓

The image of $T_A$ is $\operatorname{span}\{(1,2)^\top\}$ — a line in $\mathbb{R}^2$ — since the two columns of $A$ are $(1,2)^\top$ and $(2,4)^\top = 2 \cdot (1,2)^\top$, so rank = 1.

---

## Appendix B: Abstract Linear Algebra Perspective

### B.1 The Category of Vector Spaces

In the language of category theory (which provides a unifying framework for all of mathematics):
- **Objects:** Vector spaces over a field $\mathbb{F}$
- **Morphisms:** Linear maps between them
- **Composition:** Function composition (= matrix multiplication)
- **Identity morphism:** The identity map $\operatorname{id}_V(\mathbf{v}) = \mathbf{v}$ (= identity matrix $I$)

This forms a **category** denoted $\mathbf{Vect}_\mathbb{F}$.

**Isomorphisms in the category** are exactly the invertible linear maps — this matches our definition of isomorphism in §4.3.

**Functors** are maps between categories that preserve the categorical structure. The "matrix representation" is a functor from $\mathbf{Vect}_\mathbb{F}$ (with chosen bases) to the category $\mathbf{Mat}_\mathbb{F}$ of matrices. Changing bases corresponds to a natural transformation.

This perspective matters for ML because: neural network architectures are themselves categorical structures (composition of morphisms), and categorical thinking helps reason about when two architectures are "equivalent" (related by isomorphisms).

### B.2 Infinite-Dimensional Extensions

The theory extends to infinite-dimensional spaces, where the familiar finite-dimensional results require modification.

**Bounded linear operators.** On Hilbert spaces (infinite-dimensional inner product spaces), the right notion of "linear transformation" is a **bounded linear operator** $T: H \to H$ satisfying $\lVert T\mathbf{v}\rVert \leq M\lVert\mathbf{v}\rVert$ for some $M < \infty$. Unbounded operators (like differentiation on $L^2$) require careful domain specification.

**The spectral theorem for compact operators.** A compact self-adjoint operator on a Hilbert space has a countable set of eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \to 0$ and an orthonormal basis of eigenvectors. This is the infinite-dimensional analogue of diagonalization.

**Functional analysis.** The study of linear maps on infinite-dimensional spaces is called functional analysis. Key results (Hahn-Banach, open mapping theorem, closed graph theorem) parallel finite-dimensional results but require additional technical hypotheses.

**For AI:** Neural network function classes are subsets of infinite-dimensional function spaces ($L^2$ or Sobolev spaces). Understanding the "size" and "complexity" of these classes uses infinite-dimensional linear algebra — e.g., kernel methods and Gaussian processes operate in reproducing kernel Hilbert spaces (RKHS), and neural tangent kernel theory analyzes infinitely wide networks using spectral theory of linear operators.

### B.3 The Tensor Product and Multilinear Maps

**Bilinear maps.** A map $B: V \times W \to U$ is **bilinear** if it is linear in each argument separately. The attention score $B(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$ is bilinear (linear in $\mathbf{q}$, linear in $\mathbf{k}$, but not linear jointly: $B(c\mathbf{q}, c\mathbf{k}) = c^2 B(\mathbf{q}, \mathbf{k}) \neq c B(\mathbf{q}, \mathbf{k})$ in general).

**The tensor product $V \otimes W$** is the universal space for bilinear maps: every bilinear map $B: V \times W \to U$ factors through a unique linear map $\tilde{B}: V \otimes W \to U$. This is why tensors (in the ML sense: multi-dimensional arrays) are called tensors — they represent multilinear maps.

**For AI:** The key operation in self-attention, $\mathbf{q}^\top \mathbf{k} = \langle \mathbf{q}, \mathbf{k} \rangle$, is a bilinear form. The matrix $W$ in the "weight matrix" formulation of attention ($\mathbf{q}^\top W_{\text{QK}} \mathbf{k}$) makes this explicit: $W_{\text{QK}} = W_Q^\top W_K$ is the matrix of the bilinear form. This is the reason attention is more expressive than standard linear transformations — it computes a bilinear (quadratic) function of the input.

---


---

## Appendix C: The Geometry of Linear Maps — A Deep Dive

### C.1 How Linear Maps Deform Space

To deeply understand a linear map $T: \mathbb{R}^n \to \mathbb{R}^m$, we track how it deforms geometric objects.

**Ellipsoids to ellipsoids.** The image of the unit sphere $S^{n-1} = \{\mathbf{x} : \lVert\mathbf{x}\rVert = 1\}$ under a full-rank map $A$ is an ellipsoid whose semi-axes have lengths equal to the singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n > 0$ of $A$, pointing in the directions of the left singular vectors $\mathbf{u}_1, \ldots, \mathbf{u}_m$.

This is the geometric content of the SVD: $A = U\Sigma V^\top$ means:
1. $V^\top$: rotate the input so the "natural input directions" $\mathbf{v}_i$ align with the coordinate axes.
2. $\Sigma$: stretch each axis $i$ by $\sigma_i$.
3. $U$: rotate the output to place the stretched axes along the $\mathbf{u}_i$ directions.

The sphere becomes an ellipsoid. The "shape" of the ellipsoid is completely described by the singular values.

**Volume scaling.** The volume of the image of a set $S \subseteq \mathbb{R}^n$ under $A$ is $|\det(A)| \cdot \operatorname{vol}(S)$ (when $A$ is square). More precisely, $|\det(A)| = \prod \sigma_i$ = product of all singular values = volume of the image of the unit cube.

For a rank-deficient map ($\det(A) = 0$), the image has lower dimension — and $m$-dimensional volume = 0. The map "collapses" $n$-dimensional space to a lower-dimensional flat object.

**Angles.** Unless $A$ is orthogonal, linear maps change angles. The angle $\theta$ between $\mathbf{u}$ and $\mathbf{v}$ satisfies:
$$\cos\theta = \frac{\mathbf{u}^\top\mathbf{v}}{\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert}$$
but $\cos\angle(A\mathbf{u}, A\mathbf{v}) = \frac{(A\mathbf{u})^\top(A\mathbf{v})}{\lVert A\mathbf{u}\rVert\lVert A\mathbf{v}\rVert} = \frac{\mathbf{u}^\top A^\top A\mathbf{v}}{\lVert A\mathbf{u}\rVert\lVert A\mathbf{v}\rVert}$.

The matrix $G = A^\top A$ (the **Gram matrix**) determines how $A$ distorts inner products. Eigenvalues of $G$ are $\sigma_i^2$ (squared singular values).

### C.2 Interpreting the Four Fundamental Subspaces Geometrically

Given $T: \mathbb{R}^n \to \mathbb{R}^m$ with matrix $A$ and rank $r$:

**The row space $\operatorname{row}(A)$:** This is the "input directions that survive" — the $r$-dimensional subspace of $\mathbb{R}^n$ that $T$ maps faithfully (injectively) onto the column space. Any $\mathbf{x} \in \operatorname{row}(A)$ is "noticed" by $T$.

**The null space $\operatorname{null}(A)$:** This is the "input directions that are killed" — the $(n-r)$-dimensional subspace of $\mathbb{R}^n$ that $T$ maps to zero. Any $\mathbf{x} \in \operatorname{null}(A)$ is "invisible" to $T$.

**The decomposition $\mathbb{R}^n = \operatorname{row}(A) \oplus \operatorname{null}(A)$:** Every input $\mathbf{x}$ splits uniquely as $\mathbf{x} = \mathbf{x}_r + \mathbf{x}_n$ where $\mathbf{x}_r$ is in the row space (the "signal" part) and $\mathbf{x}_n$ is in the null space (the "noise" invisible to $T$).

**The column space $\operatorname{col}(A)$:** The $r$-dimensional subspace of $\mathbb{R}^m$ that $T$ can actually reach. Solutions to $A\mathbf{x} = \mathbf{b}$ exist iff $\mathbf{b} \in \operatorname{col}(A)$.

**The left null space $\operatorname{null}(A^\top)$:** The $(m-r)$-dimensional complement of $\operatorname{col}(A)$ in $\mathbb{R}^m$. Directions in the left null space are unreachable by $T$.

```
COMPLETE PICTURE OF THE FOUR FUNDAMENTAL SUBSPACES
════════════════════════════════════════════════════════════════════════

  ℝⁿ (domain)                           ℝᵐ (codomain)
  ─────────────────────────────────────────────────────────

  ┌─────────────────────┐    T(x) = Ax    ┌─────────────────────┐
  │    row space        │ ─────────────→  │    column space     │
  │   (dim = r)         │  isomorphism    │    (dim = r)        │
  │                     │                 │                     │
  │   ─────────────     │                 │   ─────────────     │
  │                     │                 │                     │
  │    null space       │ ─────────────→  │    left null space  │
  │   (dim = n-r)       │    maps to 0    │    (dim = m-r)      │
  └─────────────────────┘                 └─────────────────────┘
         ↑ ⊥ complement                          ↑ ⊥ complement

  Every x = (row space part) + (null space part)
  T sees only the row space part.

════════════════════════════════════════════════════════════════════════
```

**For AI (linear systems / least squares):** When fitting a model $A\mathbf{w} = \mathbf{b}$ with more constraints than parameters ($m > n$), the system is overdetermined. A solution exists only if $\mathbf{b} \in \operatorname{col}(A)$. If not, the least-squares solution minimizes $\lVert A\mathbf{w} - \mathbf{b}\rVert^2$ — finding the projection of $\mathbf{b}$ onto $\operatorname{col}(A)$ and then solving in the row space.

### C.3 Linear Maps and Information Theory

The rank-nullity theorem has an information-theoretic interpretation.

**Rank = information preserved.** A linear map of rank $r$ preserves at most $r$ "dimensions" of information from the input. The remaining $n - r$ dimensions are destroyed.

**Mutual information.** For a Gaussian input $\mathbf{x} \sim \mathcal{N}(\mathbf{0}, I_n)$ and output $\mathbf{y} = A\mathbf{x}$, the mutual information:
$$I(\mathbf{x}; \mathbf{y}) = \frac{1}{2} \sum_{i=1}^r \log(1 + \sigma_i^2)$$

depends only on the singular values — not on the specific directions. The null space of $A$ contributes zero mutual information.

**Compression.** If we want to compress $\mathbf{x} \in \mathbb{R}^n$ to $\mathbf{z} \in \mathbb{R}^r$ via a linear map $C \in \mathbb{R}^{r \times n}$, the maximum mutual information $I(\mathbf{x}; C\mathbf{x})$ is achieved when $C$ projects onto the top-$r$ right singular vectors of... itself (the row space). For structured data with covariance $\Sigma$, the optimal compression is PCA — projecting onto the top eigenvectors of $\Sigma$.

This is why PCA is the optimal linear compressor under mean-squared error: it maximizes the retained variance (information) for any fixed rank $r$.

### C.4 Generalization of Linear Maps: Tensors

The concept of a linear map generalizes to multilinear maps and tensors in ways that are directly relevant to deep learning.

**Bilinear maps and matrices of bilinear forms.** A bilinear form $B: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ can be written as $B(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top M \mathbf{y}$ for a matrix $M \in \mathbb{R}^{n \times m}$. The bilinear form is:
- **Symmetric** if $M = M^\top$ (and $n = m$): $B(\mathbf{x}, \mathbf{y}) = B(\mathbf{y}, \mathbf{x})$.
- **Positive definite** if $B(\mathbf{x}, \mathbf{x}) > 0$ for $\mathbf{x} \neq \mathbf{0}$: inner products are positive definite symmetric bilinear forms.

**Multilinear maps.** A $k$-linear map $T: V_1 \times \cdots \times V_k \to W$ is linear in each argument separately. The space of $k$-linear maps on $\mathbb{R}^n$ is the space of **tensors** of order $k$.

**For AI:** The multi-head attention score $\mathbf{q}^\top W_{\text{QK}} \mathbf{k}$ is a bilinear form parameterized by $W_{\text{QK}} = W_Q^\top W_K$. Understanding bilinear forms via their eigendecomposition ($W_{\text{QK}} = U\Lambda V^\top$ by SVD) reveals what "patterns" each attention head is sensitive to: the left singular vectors $U$ are "what queries to look for" and the right singular vectors $V$ are "what keys are being matched against."

---


---

## Appendix D: Computational Methods and Numerical Considerations

### D.1 Computing the Kernel via Row Reduction

Given $A \in \mathbb{R}^{m \times n}$, finding a basis for $\ker(A)$ requires solving $A\mathbf{x} = \mathbf{0}$.

**Algorithm (Gaussian Elimination to RREF):**

1. Apply row operations to reduce $A$ to **reduced row echelon form (RREF)**.
2. Identify **pivot columns** (columns with leading 1s in RREF) and **free columns** (all other columns).
3. For each free variable, set it to 1 and all other free variables to 0, then solve for pivot variables.
4. Each such solution is one basis vector for $\ker(A)$.

**Example.** $A = \begin{bmatrix} 1 & 2 & -1 & 0 \\ 2 & 4 & -2 & 3 \\ -1 & -2 & 1 & 2 \end{bmatrix}$.

RREF: $R_2 \leftarrow R_2 - 2R_1$, $R_3 \leftarrow R_3 + R_1$:
$$\begin{bmatrix} 1 & 2 & -1 & 0 \\ 0 & 0 & 0 & 3 \\ 0 & 0 & 0 & 2 \end{bmatrix}$$
$R_3 \leftarrow R_3 - \frac{2}{3} R_2$, $R_2 \leftarrow \frac{1}{3}R_2$:
$$\begin{bmatrix} 1 & 2 & -1 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$
$R_1 \leftarrow R_1 - 0 \cdot R_2$: already done.

Pivot columns: 1 and 4. Free variables: $x_2$ and $x_3$.

Setting $x_2 = 1, x_3 = 0$: $x_4 = 0$, $x_1 + 2(1) - 1(0) = 0 \Rightarrow x_1 = -2$. Basis vector: $(-2, 1, 0, 0)^\top$.

Setting $x_2 = 0, x_3 = 1$: $x_4 = 0$, $x_1 + 0 - 1 = 0 \Rightarrow x_1 = 1$. Basis vector: $(1, 0, 1, 0)^\top$.

**Null space basis:** $\ker(A) = \operatorname{span}\left\{(-2,1,0,0)^\top, (1,0,1,0)^\top\right\}$. Nullity = 2.

Rank-nullity: $n = 4$, rank = 2 (two pivots), nullity = 2. Check: $2 + 2 = 4$. ✓

### D.2 Numerical Stability of Basis Computations

Computing the null space or column space numerically requires care because floating-point arithmetic can introduce small errors.

**The SVD approach (recommended).** Instead of row reduction, compute the SVD: $A = U\Sigma V^\top$. Then:
- $\ker(A)$ = span of columns of $V$ corresponding to zero singular values (or singular values below a threshold $\epsilon$).
- $\operatorname{col}(A)$ = span of columns of $U$ corresponding to nonzero singular values.

The SVD-based approach is numerically stable because orthonormal bases ($U$ and $V$) are well-conditioned.

**Numerical rank.** For floating-point matrices, "zero" singular values appear as small but nonzero values. The **numerical rank** with threshold $\epsilon$ is:
$$\operatorname{rank}_\epsilon(A) = |\{i : \sigma_i > \epsilon \cdot \sigma_1\}|$$

A common choice is $\epsilon = n \cdot \text{machine epsilon}$ (about $10^{-13}$ for double precision). `numpy.linalg.matrix_rank` uses a default threshold based on machine epsilon.

**Why this matters:** In practice, a matrix with theoretical rank $r$ may appear to have rank $r + 5$ due to measurement noise. The SVD reveals the "intrinsic" rank through the gap in singular values.

### D.3 Efficient Change of Basis Computations

**Naive approach:** Compute $P^{-1}AP$ directly. For $n \times n$ matrices, this costs $O(n^3)$ (two matrix multiplications plus one matrix inversion).

**Better approach when $P$ is orthogonal:** If $P$ is orthogonal ($P^{-1} = P^\top$), then $P^\top AP$ costs only $O(n^3)$ but with better constants than general $P^{-1}AP$ (no matrix inversion needed).

**Eigendecomposition case:** When $A = P\Lambda P^{-1}$, computing $A^k = P\Lambda^k P^{-1}$ requires only $O(n)$ operations to compute $\Lambda^k$ (raise each diagonal entry to the $k$-th power), plus two $O(n^2)$ matrix-vector multiplications for $P$ and $P^{-1}$.

**For AI:** Computing $\exp(At)$ (matrix exponential, important for continuous-time state space models like Mamba/S4) is done by diagonalizing: $\exp(At) = P\exp(\Lambda t)P^{-1}$, where $\exp(\Lambda t)$ is diagonal with entries $e^{\lambda_i t}$.

### D.4 The Rank-Revealing QR Decomposition

Standard QR decomposition $A = QR$ doesn't directly reveal rank. The **rank-revealing QR** (RRQR) uses column pivoting:

$$AP = QR = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_{11} & R_{12} \\ 0 & R_{22} \end{bmatrix}$$

where $P$ is a permutation matrix, and $R_{22}$ is "small" (its Frobenius norm bounds how far $A$ is from rank-$r$). The columns of $Q_1$ form a basis for $\operatorname{col}(A)$.

RRQR is preferred over SVD when only a basis for the column space (not the singular values themselves) is needed, as it is about $3\times$ faster.

---

## Appendix E: Connections to Other Fields

### E.1 Linear Maps in Physics

In quantum mechanics, **operators** on Hilbert spaces are infinite-dimensional linear maps. The Hamiltonian $\hat{H}$, momentum $\hat{p}$, and position $\hat{x}$ are linear operators. The Schrödinger equation $i\hbar \partial_t |\psi\rangle = \hat{H}|\psi\rangle$ is a linear ODE on the Hilbert space of quantum states.

The **spectral theorem** for self-adjoint operators (the quantum generalization of symmetric matrix diagonalization) guarantees that observables have real eigenvalues (the possible measurement outcomes) and that the eigenfunctions form a complete orthonormal basis.

**For AI:** Transformers share surprising mathematical parallels with quantum mechanics: both involve attention-like mechanisms (inner products of states), superposition (linear combinations of basis states), and entanglement-like correlations. The linear algebra of quantum mechanics and of transformers both live in the framework of linear maps on Hilbert spaces.

### E.2 Linear Maps in Topology: Homomorphisms

In algebraic topology, **chain complexes** are sequences of vector spaces connected by linear maps:
$$\cdots \xrightarrow{\partial_3} C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0 \xrightarrow{\partial_0} 0$$

where $\partial_{k-1} \circ \partial_k = 0$ (boundary of a boundary is zero — exactly the condition $\ker(\partial_{k-1}) \supseteq \operatorname{im}(\partial_k)$). The **homology groups** $H_k = \ker(\partial_k) / \operatorname{im}(\partial_{k+1})$ measure "holes" in topological spaces.

**Persistent homology**, used in topological data analysis (TDA), applies this to point cloud data to find features that persist across scales. It's used in ML for analyzing data manifolds, protein structure prediction, and understanding neural network loss landscapes.

### E.3 Linear Maps in Signal Processing

The **Discrete Fourier Transform** (DFT) is a linear map $F: \mathbb{C}^n \to \mathbb{C}^n$ with matrix entries $F_{kj} = e^{-2\pi i jk/n}$. The DFT matrix is unitary ($F^* F = nI$).

Convolution is linear — convolving a signal with a kernel is a linear map — and in the Fourier domain it becomes pointwise multiplication. This is the key to making CNNs efficient: convolution is a structured linear map with shared weights (translation equivariance), and the Fourier transform diagonalizes the convolution operator.

**For AI:** The fast Fourier transform (FFT) is $O(n \log n)$ instead of $O(n^2)$ for the full DFT matrix multiply, by exploiting the structure (sparsity in a different basis) of the DFT linear map. Similarly, FlashAttention speeds up attention by exploiting the structure of the attention linear map to minimize memory bandwidth.

---


---

## Appendix F: The Algebra of Linear Maps — Structural Results

### F.1 The Space of Linear Maps is a Vector Space

We noted briefly that $\mathcal{L}(V, W)$ is a vector space. Let's make this precise and compute its dimension.

**Operations:** For $S, T \in \mathcal{L}(V, W)$ and $c \in \mathbb{F}$:
- $(S + T)(\mathbf{v}) = S(\mathbf{v}) + T(\mathbf{v})$
- $(cT)(\mathbf{v}) = c \cdot T(\mathbf{v})$
- The zero element is the zero map $O(\mathbf{v}) = \mathbf{0}$ for all $\mathbf{v}$.

**Dimension.** If $\dim(V) = n$ and $\dim(W) = m$, then:
$$\dim(\mathcal{L}(V, W)) = mn$$

*Proof:* Every linear map $T: V \to W$ is determined by $n$ vectors in $W$ (images of basis vectors), each in a $m$-dimensional space. The natural isomorphism is $\mathcal{L}(V, W) \cong W^n \cong \mathbb{R}^{mn}$ (as vector spaces), which corresponds to the identification with $m \times n$ matrices. $\square$

**Basis for $\mathcal{L}(\mathbb{R}^n, \mathbb{R}^m)$.** The standard basis consists of the $mn$ maps $E_{ij}: \mathbb{R}^n \to \mathbb{R}^m$ defined by $E_{ij}(\mathbf{e}_k) = \delta_{jk}\mathbf{e}_i$. In matrix form, $E_{ij}$ is the matrix with 1 in position $(i,j)$ and 0 elsewhere.

### F.2 Composition Gives $\mathcal{L}(V, V)$ an Algebra Structure

When $V = W$, linear maps $T: V \to V$ can be composed. The space $\mathcal{L}(V, V)$ (endomorphisms of $V$) is a **ring** under composition (it is also a vector space — together, an **algebra**).

**Properties of composition in $\mathcal{L}(V, V)$:**
- **Associative:** $(RS)T = R(ST)$
- **Identity:** $I \circ T = T \circ I = T$
- **Distributive:** $R(S + T) = RS + RT$ and $(R+S)T = RT + ST$
- **NOT commutative:** $RS \neq SR$ in general

**Matrix polynomials.** For $T \in \mathcal{L}(V,V)$, we can form $p(T) = a_0 I + a_1 T + a_2 T^2 + \cdots + a_k T^k$ for any polynomial $p$. This is well-defined because we can add and compose linear maps.

**Cayley-Hamilton theorem.** Every linear operator $T$ satisfies its own characteristic polynomial: $p_T(T) = 0$, where $p_T(\lambda) = \det(\lambda I - T)$.

**For AI:** The spectral approach to recurrent networks analyzes the long-run behavior of $T^k \mathbf{v}$ as $k \to \infty$. If $T$ has eigenvalues $|\lambda_i| < 1$, then $T^k \to 0$ (stable memory decay). If any $|\lambda_i| > 1$, the recurrence explodes. This spectral stability analysis is the foundation of designing stable RNNs (LSTM, GRU use gating to control the effective eigenvalue spectrum of the recurrence).

### F.3 Quotient Maps and Projections

**The quotient space.** Given $T: V \to W$ with kernel $K = \ker(T)$, the **quotient space** $V/K$ consists of equivalence classes $[\mathbf{v}] = \mathbf{v} + K = \{\mathbf{v} + \mathbf{k} : \mathbf{k} \in K\}$.

$V/K$ is a vector space of dimension $\dim(V) - \dim(K) = \operatorname{rank}(T)$.

**The first isomorphism theorem.** Every linear map $T: V \to W$ factors as:
$$V \xrightarrow{\pi} V/\ker(T) \xrightarrow{\tilde{T}} \operatorname{im}(T) \hookrightarrow W$$

where $\pi$ is the quotient map ($\pi(\mathbf{v}) = [\mathbf{v}]$) and $\tilde{T}$ is an **isomorphism** from $V/\ker(T)$ to $\operatorname{im}(T)$. 

This is the coordinate-free statement of the rank-nullity theorem: $V/\ker(T) \cong \operatorname{im}(T)$.

**Geometric meaning.** The quotient map $\pi$ "collapses" the null space to a point, then $\tilde{T}$ acts faithfully (injectively) on the resulting space. Any linear map splits into: collapse (project out the null space) + inject faithfully into the codomain.

**For AI:** In contrastive learning (SimCLR, MoCo), the projection head maps representations to a lower-dimensional space. This is a linear (or nonlinear) quotient map — it deliberately collapses some dimensions (those corresponding to nuisance factors like image augmentation) while preserving the semantically meaningful directions. The first isomorphism theorem says: what survives in the image is exactly what was not collapsed.

### F.4 Dual Bases and the Canonical Isomorphism

We saw that $\dim(V^*) = \dim(V)$, so $V \cong V^*$. But this isomorphism is **non-canonical** — it depends on the choice of basis.

**With an inner product.** When $V$ is an inner product space (like $\mathbb{R}^n$ with the standard dot product), there is a **canonical** isomorphism $\phi: V \to V^*$ defined by:
$$\phi(\mathbf{v}) = \langle \mathbf{v}, \cdot \rangle$$

That is, $\phi(\mathbf{v})$ is the linear functional that takes $\mathbf{w} \mapsto \langle \mathbf{v}, \mathbf{w}\rangle$.

This isomorphism is canonical because it doesn't depend on any choice of basis — it uses only the inner product structure. When we identify $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ as a column vector (primal) rather than a row vector (dual), we are implicitly using this canonical isomorphism via the standard inner product.

**For AI:** On non-Euclidean spaces (manifolds of probability distributions, manifolds of neural network weights under the Fisher metric), the identification $V \cong V^*$ is NO longer trivial — gradients and velocity vectors live in different spaces. The **natural gradient** method corrects for this by using the Fisher information matrix $F$ as the metric: $\tilde{\nabla} \theta = F^{-1} \nabla \theta$. This maps the gradient (a covector) to a tangent vector using the Riemannian metric instead of the Euclidean metric.

---

## Appendix G: Linear Maps in Practice — Worked Problems

### G.1 Verifying Linearity: Systematic Approach

**Problem:** Is $T: \mathbb{R}^{2\times 2} \to \mathbb{R}$ defined by $T(A) = \operatorname{tr}(A)$ linear?

**Check additivity:** $T(A + B) = \operatorname{tr}(A + B) = \operatorname{tr}(A) + \operatorname{tr}(B) = T(A) + T(B)$. ✓

**Check homogeneity:** $T(cA) = \operatorname{tr}(cA) = c\operatorname{tr}(A) = cT(A)$. ✓

**Conclusion:** $T$ is linear. Its matrix (viewing $\mathbb{R}^{2\times 2}$ with basis $\{E_{11}, E_{12}, E_{21}, E_{22}\}$):
$$\operatorname{tr}(E_{11}) = 1, \quad \operatorname{tr}(E_{12}) = 0, \quad \operatorname{tr}(E_{21}) = 0, \quad \operatorname{tr}(E_{22}) = 1$$

So $[T] = (1, 0, 0, 1)$ as a $1 \times 4$ matrix. Kernel = $\{A : \operatorname{tr}(A) = 0\}$ (trace-zero matrices), dimension 3.

**Problem:** Is $T: \mathbb{R}^n \to \mathbb{R}$ defined by $T(\mathbf{x}) = \lVert\mathbf{x}\rVert_2$ linear?

**Check homogeneity:** $T(c\mathbf{x}) = \lVert c\mathbf{x}\rVert = |c| \lVert\mathbf{x}\rVert$. For $c = -1$: $T(-\mathbf{x}) = \lVert\mathbf{x}\rVert = T(\mathbf{x})$, but linearity requires $T(-\mathbf{x}) = -T(\mathbf{x})$.

For $\mathbf{x} \neq \mathbf{0}$: $T(\mathbf{x}) > 0$ but $-T(\mathbf{x}) < 0$. So $T(-\mathbf{x}) \neq -T(\mathbf{x})$.

**Conclusion:** $T$ is NOT linear (the norm fails homogeneity due to the absolute value).

### G.2 Finding the Kernel: Four Approaches

For $A = \begin{bmatrix} 1 & -1 & 2 \\ 2 & -2 & 4 \end{bmatrix}$ (rows are multiples), find $\ker(A)$:

**Approach 1: Row reduction.** RREF: $\begin{bmatrix} 1 & -1 & 2 \\ 0 & 0 & 0 \end{bmatrix}$. Free variables $x_2 = s$, $x_3 = t$. Then $x_1 = s - 2t$.

$\ker(A) = \operatorname{span}\{(1,1,0)^\top, (-2,0,1)^\top\}$.

**Approach 2: Inspection.** The columns satisfy $\mathbf{a}_2 = -\mathbf{a}_1$ and $\mathbf{a}_3 = 2\mathbf{a}_1$. So $A(1,-1,0)^\top = \mathbf{a}_1 + \mathbf{a}_1 = 0$... no, wait: $A(1,-1,0)^\top = 1\mathbf{a}_1 + (-1)(-\mathbf{a}_1) + 0 = \mathbf{a}_1 + \mathbf{a}_1 = 2\mathbf{a}_1 \neq \mathbf{0}$.

Correcting: $\mathbf{a}_2 = -\mathbf{a}_1$ means $\mathbf{a}_1 + \mathbf{a}_2 = \mathbf{0}$, so $(1,1,0)^\top \in \ker(A)$. And $2\mathbf{a}_1 + 0\mathbf{a}_2 + (-1)\mathbf{a}_3 = 2\mathbf{a}_1 - 2\mathbf{a}_1 = \mathbf{0}$ since $\mathbf{a}_3 = 2\mathbf{a}_1$, so $(2,0,-1)^\top \in \ker(A)$ — or equivalently $(-2,0,1)^\top$.

**Approach 3: SVD.** Compute SVD of $A$; null space vectors are the right singular vectors with zero (or near-zero) singular values.

**Approach 4: Random sampling + orthogonalization.** Sample many vectors, project out the row space, keep those with zero image (useful when $A$ is very large).

### G.3 Composition of Transforms in a Graphics Pipeline

A 3D object is processed through a graphics pipeline using compositions of affine maps:

1. **Model matrix** $M$: transform from object coordinates to world coordinates (rotation, scale, translation).
2. **View matrix** $V$: transform from world coordinates to camera coordinates (rotation + translation).
3. **Projection matrix** $P$: from camera coordinates to clip coordinates (perspective projection).

The combined transform: $\mathbf{p}_{\text{clip}} = P \cdot V \cdot M \cdot \mathbf{p}_{\text{object}}$ (in homogeneous coordinates).

**Composition order matters.** Reading left to right: first apply $M$, then $V$, then $P$. The matrix product $PVM$ can be precomputed once per frame (not per vertex), saving $O(|\text{vertices}|)$ matrix multiplications.

This is the same principle as "avoid recomputing shared prefixes" in transformer KV-caching: the $KV$ cache stores the linear maps $K = XW_K$, $V = XW_V$ for all past tokens, so they don't need to be recomputed when generating each new token.

---


---

## Appendix H: Linear Maps in Optimization and Training

### H.1 The Gradient as a Linear Map

In optimization, we minimize a loss function $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$. The gradient $\nabla \mathcal{L}(\mathbf{w})$ tells us the direction of steepest ascent. But more precisely:

The **directional derivative** of $\mathcal{L}$ at $\mathbf{w}$ in direction $\mathbf{d}$ is:
$$D_{\mathbf{d}} \mathcal{L}(\mathbf{w}) = \lim_{t \to 0} \frac{\mathcal{L}(\mathbf{w} + t\mathbf{d}) - \mathcal{L}(\mathbf{w})}{t} = \nabla \mathcal{L}(\mathbf{w})^\top \mathbf{d}$$

This is a **linear functional** in $\mathbf{d}$: it is the dual vector $\nabla \mathcal{L}(\mathbf{w})^\top \in (\mathbb{R}^n)^*$.

**Gradient descent** in its pure form: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$. This uses the Euclidean identification of the gradient (covector) with a primal vector.

**Natural gradient descent**: $\mathbf{w}_{t+1} = \mathbf{w}_t - \eta F(\mathbf{w}_t)^{-1} \nabla \mathcal{L}(\mathbf{w}_t)$, where $F$ is the Fisher information matrix. This uses the correct metric on the manifold of probability distributions (the Fisher-Rao metric) to convert the covector gradient to a tangent vector.

### H.2 The Hessian as a Bilinear Map

The **Hessian** $H\mathcal{L}(\mathbf{w}) \in \mathbb{R}^{n \times n}$ is the matrix of second derivatives:
$$H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}$$

But more abstractly, the Hessian is a **bilinear form** $B: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$:
$$B(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top H \mathbf{v} = \text{rate of change of directional derivative of } \mathcal{L} \text{ in direction } \mathbf{v}, \text{ in direction } \mathbf{u}$$

The Hessian determines the **curvature** of the loss landscape:
- Positive definite Hessian ($\mathbf{u}^\top H\mathbf{u} > 0$ for all $\mathbf{u} \neq 0$): the point is a local minimum.
- Indefinite Hessian (has both positive and negative eigenvalues): the point is a saddle point.
- The ratio of largest to smallest eigenvalue is the **condition number** $\kappa(H)$ — it governs how slowly gradient descent converges.

**For AI:** Modern optimizers (Adam, AdaGrad) approximate Hessian-related quantities. Adam's second moment estimate $\hat{v}_t \approx \operatorname{diag}(H)$ approximates the diagonal of the Hessian. Dividing the gradient by $\sqrt{\hat{v}_t}$ is an approximation to Newton's method (which divides by $H$). This is why Adam often converges much faster than SGD on ill-conditioned problems.

### H.3 Weight Matrices as Linear Maps: Training Dynamics

The **neural tangent kernel** (NTK) theory (Jacot et al., 2018) analyzes infinitely wide neural networks and shows that their training dynamics under gradient flow are governed by a **linear system**:

$$\dot{\mathbf{y}}_t = -\eta \, \Theta(\mathbf{X}, \mathbf{X}) (\mathbf{y}_t - \mathbf{y}^*)$$

where $\Theta(\mathbf{X}, \mathbf{X})$ is the NTK matrix (constant in the infinite-width limit). This is an ODE with a constant **linear map** $\Theta$ — so its solution is $\mathbf{y}_t = \mathbf{y}^* + e^{-\eta \Theta t}(\mathbf{y}_0 - \mathbf{y}^*)$.

The eigenvalues of $\Theta$ determine which output directions are learned quickly (large eigenvalues → fast convergence) and which slowly (small eigenvalues → slow convergence). This is linear algebra — specifically, the spectral decomposition of a positive semidefinite linear map.

### H.4 Gradient Flow through Linear Layers

Consider a linear layer $\mathbf{y} = W\mathbf{x}$ with loss $\mathcal{L}$. The gradient with respect to the weight matrix:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \mathbf{x}^\top = \boldsymbol{\delta} \mathbf{x}^\top$$

where $\boldsymbol{\delta} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \in \mathbb{R}^m$ is the "error signal" (upstream gradient). The gradient $\frac{\partial \mathcal{L}}{\partial W}$ is an **outer product** — a rank-1 matrix.

**This means gradient updates are always rank-1.** For a mini-batch of $B$ samples, the gradient is:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{B}\sum_{b=1}^B \boldsymbol{\delta}_b \mathbf{x}_b^\top$$

A sum of $B$ rank-1 matrices — the gradient has rank at most $B$. For large models with batch size $B \ll n$, the gradient is a very low-rank update to the weight matrix. This low-rank structure of gradients is the empirical justification for gradient low-rank projection methods (GaLore, 2024).

---

## Appendix I: Reference Tables

### I.1 Linear Map Properties at a Glance

| Property | Condition | Matrix Equivalent | Geometric Meaning |
| --- | --- | --- | --- |
| Linear | $T(a\mathbf{u}+b\mathbf{v}) = aT\mathbf{u}+bT\mathbf{v}$ | Any $m \times n$ matrix | Preserves addition and scaling |
| Injective | $\ker(T) = \{\mathbf{0}\}$ | Full column rank | No two inputs map to same output |
| Surjective | $\operatorname{im}(T) = W$ | Full row rank | Every output is reachable |
| Bijective (isomorphism) | Both injective and surjective | Square, full rank, invertible | Perfect correspondence |
| Orthogonal | $T^\top T = I$ | $A^\top A = I$, orthogonal columns | Preserves lengths and angles |
| Unitary | $T^* T = I$ (complex) | $A^* A = I$, unitary columns | Complex analogue of orthogonal |
| Projection | $T^2 = T$ | $A^2 = A$ | Idempotent: applying twice = once |
| Symmetric | $T = T^\top$ | $A = A^\top$ | Self-adjoint; diagonalizable by spectral theorem |
| Positive definite | $\langle T\mathbf{v}, \mathbf{v}\rangle > 0$ | $\mathbf{x}^\top A\mathbf{x} > 0$ for $\mathbf{x} \neq \mathbf{0}$ | All eigenvalues positive; curvature at min |
| Normal | $T T^\top = T^\top T$ | $A A^\top = A^\top A$ | Diagonalizable by unitary matrix |
| Nilpotent | $T^k = 0$ for some $k$ | $A^k = 0$ | Powers eventually vanish; all eigenvalues 0 |
| Involution | $T^2 = I$ | $A^2 = I$ | Self-inverse (like Householder reflections) |

### I.2 Rank and Dimension Formulas

| Formula | Statement |
| --- | --- |
| $\operatorname{rank}(T) + \operatorname{nullity}(T) = \dim(V)$ | Rank-nullity theorem for $T: V \to W$ |
| $\operatorname{rank}(A) = \operatorname{rank}(A^\top)$ | Row rank equals column rank |
| $\operatorname{rank}(AB) \leq \min(\operatorname{rank}(A), \operatorname{rank}(B))$ | Rank cannot increase under composition |
| $\operatorname{rank}(A + B) \leq \operatorname{rank}(A) + \operatorname{rank}(B)$ | Subadditivity of rank |
| $\dim(S + T) = \dim(S) + \dim(T) - \dim(S \cap T)$ | Inclusion-exclusion for subspaces |
| $\dim(V/W) = \dim(V) - \dim(W)$ (for subspace $W$) | Dimension of quotient space |
| $\operatorname{rank}(A^\top A) = \operatorname{rank}(A)$ | Gram matrix has same rank |
| $\operatorname{rank}(P) = \operatorname{tr}(P)$ (for projection $P$) | Rank = trace for idempotent matrices |

### I.3 AI Applications Cross-Reference

| Linear Map Concept | Where It Appears in AI | Mathematical Role |
| --- | --- | --- |
| $W\mathbf{x} + \mathbf{b}$ (affine map) | Every neural layer | Pre-activation computation |
| $Q = XW_Q$ (linear projection) | Attention mechanism | Projects to query subspace |
| $\Delta W = BA$ (low-rank) | LoRA fine-tuning | Rank-$r$ weight update |
| $J_f = \frac{\partial f}{\partial \mathbf{x}}$ (Jacobian) | Backpropagation | Chain rule at each layer |
| $W^\top \boldsymbol{\delta}$ (transpose map) | Backward pass | Dual map of forward |
| $P = QQ^\top$ (projection) | Layer norm, attention | Projects onto subspace |
| $W_U \mathbf{h}$ (linear map) | Unembedding (logit computation) | Representation to vocabulary |
| $e^{i\theta} \cdot \mathbf{q}$ (rotation in $\mathbb{C}$) | RoPE positional encoding | Positional rotation |
| $F^{-1}\nabla\mathcal{L}$ (metric-adjusted gradient) | Natural gradient / Adam | Riemannian gradient |
| $\Theta = J^\top J$ (Gram matrix of Jacobian) | Neural tangent kernel | Training dynamics |

---


---

## Appendix J: Proofs of Key Results

### J.1 Proof: $\operatorname{rank}(A) = \operatorname{rank}(A^\top)$

This is a fundamental result that deserves a careful proof.

**Theorem.** For any matrix $A \in \mathbb{R}^{m \times n}$, the column rank (dimension of the column space) equals the row rank (dimension of the row space).

**Proof (via RREF).** Let $A$ have RREF $R$ (obtained by row operations, which don't change the row space but can change the column space). In $R$:
- The nonzero rows are linearly independent (each has a leading 1 not shared by any other row).
- The number of nonzero rows = number of pivot columns = rank.

So row rank = column rank = number of pivots in RREF. $\square$

**Alternative proof (via SVD).** The SVD $A = U\Sigma V^\top$ has $\operatorname{rank}(A)$ nonzero singular values. The column space of $A$ is spanned by $\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ (first $r$ left singular vectors), dimension $r$. The row space of $A$ (= column space of $A^\top$) is spanned by $\{\mathbf{v}_1, \ldots, \mathbf{v}_r\}$ (first $r$ right singular vectors), dimension $r$. Both have dimension $r$ = number of nonzero singular values. $\square$

### J.2 Proof: Kernel and Image are Subspaces

**Theorem.** For any linear map $T: V \to W$, both $\ker(T)$ and $\operatorname{im}(T)$ are subspaces (of $V$ and $W$ respectively).

**Proof for $\ker(T)$:**
1. **Non-empty:** $T(\mathbf{0}_V) = \mathbf{0}_W$, so $\mathbf{0}_V \in \ker(T)$.
2. **Closed under addition:** Let $\mathbf{u}, \mathbf{v} \in \ker(T)$. Then $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}) = \mathbf{0} + \mathbf{0} = \mathbf{0}$, so $\mathbf{u} + \mathbf{v} \in \ker(T)$.
3. **Closed under scalar multiplication:** Let $\mathbf{v} \in \ker(T)$, $c \in \mathbb{F}$. Then $T(c\mathbf{v}) = cT(\mathbf{v}) = c\mathbf{0} = \mathbf{0}$, so $c\mathbf{v} \in \ker(T)$. $\square$

**Proof for $\operatorname{im}(T)$:**
1. **Non-empty:** $T(\mathbf{0}_V) = \mathbf{0}_W \in \operatorname{im}(T)$.
2. **Closed under addition:** Let $\mathbf{w}_1, \mathbf{w}_2 \in \operatorname{im}(T)$, so $\mathbf{w}_i = T(\mathbf{v}_i)$ for some $\mathbf{v}_i \in V$. Then $\mathbf{w}_1 + \mathbf{w}_2 = T(\mathbf{v}_1) + T(\mathbf{v}_2) = T(\mathbf{v}_1 + \mathbf{v}_2) \in \operatorname{im}(T)$.
3. **Closed under scalar multiplication:** Let $\mathbf{w} = T(\mathbf{v}) \in \operatorname{im}(T)$, $c \in \mathbb{F}$. Then $c\mathbf{w} = cT(\mathbf{v}) = T(c\mathbf{v}) \in \operatorname{im}(T)$. $\square$

### J.3 Proof: Composition of Linear Maps is Linear

**Theorem.** If $S: V \to W$ and $T: U \to V$ are linear, then $S \circ T: U \to W$ is linear.

**Proof:**
$$(S \circ T)(a\mathbf{u}_1 + b\mathbf{u}_2) = S(T(a\mathbf{u}_1 + b\mathbf{u}_2))$$
$$= S(aT(\mathbf{u}_1) + bT(\mathbf{u}_2)) \quad \text{(T is linear)}$$
$$= aS(T(\mathbf{u}_1)) + bS(T(\mathbf{u}_2)) \quad \text{(S is linear)}$$
$$= a(S \circ T)(\mathbf{u}_1) + b(S \circ T)(\mathbf{u}_2) \quad \square$$

### J.4 Proof: The Dual Map is Linear

**Theorem.** If $T: V \to W$ is linear, then $T^\top: W^* \to V^*$ defined by $(T^\top f)(\mathbf{v}) = f(T(\mathbf{v}))$ is linear.

**Proof:**
- $(T^\top(af + bg))(\mathbf{v}) = (af + bg)(T(\mathbf{v})) = af(T(\mathbf{v})) + bg(T(\mathbf{v}))$
- $= a(T^\top f)(\mathbf{v}) + b(T^\top g)(\mathbf{v}) = (aT^\top f + bT^\top g)(\mathbf{v})$

So $T^\top(af + bg) = aT^\top f + bT^\top g$. $\square$

### J.5 Proof: Invertibility Criterion for Finite-Dimensional Spaces

**Theorem.** Let $T: V \to V$ be a linear map on a finite-dimensional space $V$. Then the following are equivalent:
1. $T$ is injective (one-to-one)
2. $T$ is surjective (onto)
3. $T$ is bijective (invertible)
4. $\ker(T) = \{\mathbf{0}\}$
5. $\operatorname{rank}(T) = \dim(V)$

**Proof:**
$(1) \Leftrightarrow (4)$: $T$ injective iff $\ker(T) = \{\mathbf{0}\}$ (standard).

$(4) \Leftrightarrow (5)$: By rank-nullity: $\operatorname{rank}(T) + \operatorname{nullity}(T) = \dim(V)$. Nullity = 0 iff rank = $\dim(V)$.

$(5) \Leftrightarrow (2)$: Rank = $\dim(\operatorname{im}(T))$. Since $\operatorname{im}(T) \subseteq V$ and both are finite-dimensional, $\dim(\operatorname{im}(T)) = \dim(V)$ iff $\operatorname{im}(T) = V$ (a subspace of equal dimension must be the whole space).

$(3) \Leftrightarrow (1) \& (2)$: by definition of bijective. $\square$

**Important:** This equivalence only holds for maps $T: V \to V$ with the **same** domain and codomain. For $T: \mathbb{R}^m \to \mathbb{R}^n$ with $m \neq n$, injective and surjective are NOT equivalent (one is impossible given the dimension constraint).

---


---

## Appendix K: Additional AI Case Studies

### K.1 Mechanistic Interpretability via Linear Maps

Mechanistic interpretability (MI) aims to reverse-engineer neural networks by understanding what computation each component performs. Linear map theory is central to this enterprise.

**The residual stream as a communication bus.** In transformer models, each layer reads from and writes to a shared "residual stream" $\mathbf{x} \in \mathbb{R}^d$. Each attention head and MLP layer contributes an additive update:

$$\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \underbrace{\text{Attn}_\ell(\mathbf{x}_\ell)}_{\text{attn. update}} + \underbrace{\text{MLP}_\ell(\mathbf{x}_\ell)}_{\text{MLP update}}$$

Each update is (approximately) a low-rank linear map from the residual stream back to itself. The attention update's linear part is $W_O W_V$ (the "OV circuit"); the MLP's linear part is $W_{\text{out}} W_{\text{in}}$ after linearizing the activation.

**SVD of the OV circuit.** The matrix $W_O W_V \in \mathbb{R}^{d \times d}$ can be analyzed via SVD. Its singular values reveal how strongly the attention head modifies the residual stream, and its singular vectors reveal which directions it reads from and writes to. Heads with near-zero singular values are "inattentive" — they barely modify the residual stream regardless of attention pattern.

**Subspace decomposition.** The full set of $L \times H$ attention heads (for $L$ layers, $H$ heads per layer) collectively form a large linear map from the input to the residual stream updates. The total update is a sum of $LH$ low-rank linear maps. Understanding the structure of this sum — which heads are redundant, which are essential — is a central goal of circuit-level MI.

### K.2 Linear Algebra of Diffusion Models

Diffusion models (DDPM, Score matching) add Gaussian noise to data and learn to denoise. The forward process is an affine map:

$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\, \mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\, \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

This is an affine interpolation between the data $\mathbf{x}_0$ and pure noise. The coefficient $\sqrt{\bar\alpha_t}$ scales the data, and $\sqrt{1-\bar\alpha_t}$ scales the noise.

**The denoising objective.** The neural network $\epsilon_\theta(\mathbf{x}_t, t)$ estimates $\boldsymbol{\varepsilon}$ (the noise) from the noisy input. Near a data point $\mathbf{x}_0$, this estimator is approximately a **linear function** of $\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0$ — the Tweedie formula gives the optimal estimator as:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta(\mathbf{x}_t, t)}{\sqrt{\bar\alpha_t}}$$

which is a linear function of $\mathbf{x}_t$ and $\epsilon_\theta(\mathbf{x}_t, t)$. The diffusion process itself is a composition of affine maps in the forward direction, and the learned reverse process approximately inverts these affine maps.

### K.3 State Space Models as Linear Dynamical Systems

Structured State Space Models (S4, Mamba, RWKV) compute their state updates via linear recurrences:

$$\mathbf{h}_{t+1} = A\mathbf{h}_t + B\mathbf{x}_t$$
$$\mathbf{y}_t = C\mathbf{h}_t + D\mathbf{x}_t$$

where $A \in \mathbb{R}^{N \times N}$, $B \in \mathbb{R}^{N \times d}$, $C \in \mathbb{R}^{d \times N}$, $D \in \mathbb{R}^{d \times d}$ are (possibly input-dependent) matrices.

The state transition $\mathbf{h} \mapsto A\mathbf{h} + B\mathbf{x}$ is a **linear dynamical system** — the fundamental object of study in control theory and signal processing.

**Key linear algebra results for SSMs:**

1. **Eigenvalues of $A$ determine memory.** If $|\lambda_i(A)| < 1$ for all $i$, the system has bounded memory decay. If any $|\lambda_i| > 1$, the state can grow unboundedly.

2. **Diagonalization for efficiency.** If $A = P\Lambda P^{-1}$, the recurrence decouples into $N$ independent scalar recurrences — each computable independently. S4 uses diagonal $A$ (DPLR structure) for $O(N\log N)$ parallel computation via convolution.

3. **The convolution view.** Unrolling the recurrence: $\mathbf{y}_t = \sum_{\tau=0}^t C A^{t-\tau} B \mathbf{x}_\tau + D\mathbf{x}_t$. The impulse response $C A^k B$ is a sequence of matrix powers — analyzable by the spectral decomposition of $A$.

4. **Mamba's selectivity.** Mamba makes $A, B, C$ input-dependent: $A(\mathbf{x}), B(\mathbf{x}), C(\mathbf{x})$. The recurrence becomes **bilinear** in $(\mathbf{h}, \mathbf{x})$, not purely linear. The linearization (around typical inputs) gives a locally linear system analyzable by the tools of this section.

---

## Appendix L: Further Reading

### Primary References

1. **Axler, S. (2015).** *Linear Algebra Done Right* (3rd ed.). Springer. — The definitive abstract treatment of linear maps. Goes from axioms to spectral theory without matrices until chapter 10. Highly recommended for conceptual depth.

2. **Strang, G. (2016).** *Introduction to Linear Algebra* (5th ed.). Wellesley-Cambridge Press. — Computational and applied focus. Excellent for four fundamental subspaces and applications.

3. **Horn, R. & Johnson, C. (2013).** *Matrix Analysis* (2nd ed.). Cambridge University Press. — Comprehensive advanced reference. Proofs of all major results, including Cayley-Hamilton, spectral theorems, singular values.

4. **Trefethen, L. & Bau, D. (1997).** *Numerical Linear Algebra.* SIAM. — Gold standard for computational linear algebra and stability.

### AI-Focused References

5. **Vaswani, A. et al. (2017).** "Attention is All You Need." NeurIPS. — The original transformer paper; read the attention mechanism as linear projections.

6. **Hu, E. et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. — LoRA rank-nullity argument.

7. **Elhage, N. et al. (2021).** "A Mathematical Framework for Transformer Circuits." Anthropic. — OV and QK circuits as linear maps; residual stream as communication bus.

8. **Park, K. et al. (2023).** "The Linear Representation Hypothesis and the Geometry of Large Language Models." — Linear features in transformer representations.

9. **Jacot, A. et al. (2018).** "Neural Tangent Kernel: Convergence and Generalization in Neural Networks." NeurIPS. — Training dynamics via linear maps (NTK theory).

10. **Gu, A. et al. (2022).** "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR. — SSMs as linear dynamical systems.

---

*This section is part of the [Math for LLMs](../../README.md) curriculum — a systematic treatment of the mathematics underlying modern large language models.*


---

## Appendix M: Self-Assessment Checklist

After completing this section, you should be able to answer the following questions without notes.

### Conceptual Understanding

- [ ] **Q1.** State the two axioms of a linear map. What is the fastest way to show a map is NOT linear?

- [ ] **Q2.** What is the kernel of a linear map? Prove it is a subspace.

- [ ] **Q3.** State the rank-nullity theorem. Give an example where rank = 2, nullity = 3. What can you say about the domain and codomain dimensions?

- [ ] **Q4.** Why is a linear map from $V$ to $V$ injective if and only if it is surjective? Why does this fail for maps between spaces of different dimensions?

- [ ] **Q5.** What is the change-of-basis formula? If $A = P^{-1}BP$, what relationship does that establish between the maps represented by $A$ and $B$?

- [ ] **Q6.** What is an orthogonal projection? How do you verify that a matrix $P$ is a projection? What two extra properties make it an orthogonal projection?

- [ ] **Q7.** What is the Jacobian matrix? For $f: \mathbb{R}^3 \to \mathbb{R}^2$, what is the shape of $J_f$?

- [ ] **Q8.** In the backward pass of backpropagation, why do we multiply by $W^\top$ rather than $W$?

- [ ] **Q9.** What makes an affine map different from a linear map? How do you represent an affine map as a linear map (in one higher dimension)?

### Computational Skills

- [ ] **C1.** Given a matrix $A$, find a basis for $\ker(A)$ using row reduction.

- [ ] **C2.** Given a linear map $T$ defined on a non-standard basis, write its matrix in that basis.

- [ ] **C3.** Given two bases $\mathcal{B}$ and $\mathcal{B}'$, compute the change-of-basis matrix $P$ and use it to transform the matrix of $T$.

- [ ] **C4.** For a rank-$r$ update $\Delta W = BA$, compute the null space dimension and verify it numerically.

- [ ] **C5.** Compute the Jacobian of a given vector-valued function (e.g., softmax, elementwise ReLU, an affine map composed with sigmoid).

### AI Connections

- [ ] **AI1.** Explain why LoRA (low-rank adaptation) is more parameter-efficient than full fine-tuning, using the language of rank and nullity.

- [ ] **AI2.** Describe the forward pass of a multi-head attention layer as a sequence of linear maps. Which operations are linear, which are bilinear, and which are nonlinear?

- [ ] **AI3.** What is the linear representation hypothesis? Why does it matter for interpretability research?

- [ ] **AI4.** Why does a purely linear deep network (no activations) collapse to a single linear map, regardless of depth?

- [ ] **AI5.** How does the dual map relate to backpropagation? What mathematical object is the "gradient" in the strict sense?

---

## Appendix N: Notation Summary for This Section

| Symbol | Meaning |
| --- | --- |
| $T: V \to W$ | Linear transformation from $V$ to $W$ |
| $\ker(T)$ | Kernel (null space) of $T$ |
| $\operatorname{im}(T)$ | Image (range) of $T$ |
| $\operatorname{rank}(T)$ | Dimension of $\operatorname{im}(T)$ |
| $\operatorname{nullity}(T)$ | Dimension of $\ker(T)$ |
| $[T]_{\mathcal{B}}^{\mathcal{C}}$ | Matrix of $T$ from basis $\mathcal{B}$ to basis $\mathcal{C}$ |
| $P$ | Change-of-basis matrix |
| $T^\top$ | Dual (transpose) map |
| $V^*$ | Dual space of $V$ |
| $Df_{\mathbf{x}}$ | Total derivative (Fréchet derivative) of $f$ at $\mathbf{x}$ |
| $J_f(\mathbf{x})$ | Jacobian matrix of $f$ at $\mathbf{x}$ |
| $\mathcal{L}(V, W)$ | Space of all linear maps from $V$ to $W$ |
| $V \cong W$ | $V$ and $W$ are isomorphic |
| $V/K$ | Quotient space of $V$ modulo subspace $K$ |
| $A \sim B$ | $A$ and $B$ are similar matrices ($B = P^{-1}AP$) |
| $P^2 = P$ | Projection (idempotent) |
| $A^\top A = I$ | Orthogonal matrix |
| $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ | Affine map |
| $\Delta W = BA$ | LoRA low-rank update, $\operatorname{rank} \leq r$ |


---

## Appendix O: Linear Maps and Symmetry

### O.1 Equivariant Maps

A linear map $T: V \to W$ is **equivariant** with respect to a group $G$ if, for every $g \in G$ and every $\mathbf{v} \in V$:
$$T(\rho_V(g)\mathbf{v}) = \rho_W(g) T(\mathbf{v})$$

where $\rho_V: G \to GL(V)$ and $\rho_W: G \to GL(W)$ are representations of $G$ on $V$ and $W$.

Intuitively: "applying the group action then the map = applying the map then the group action." The map commutes with the symmetry.

**Examples:**
- **Translation equivariance:** $T(v + c) = T(v) + T(c)$... but this is additivity — every linear map is equivariant to the translation group on vector spaces.
- **Rotation equivariance:** $T(R\mathbf{v}) = RT(\mathbf{v})$ for all rotations $R$. In 3D: implies $T = \lambda I$ for some scalar $\lambda$ (Schur's lemma for the rotation representation).
- **Permutation equivariance:** $T(P\mathbf{v}) = PT(\mathbf{v})$ for all permutation matrices $P$. Implies $T$ is a sum of a "same-position" term and a "mean-field" term — this is why mean pooling and attention with tied weights are permutation equivariant.

**For AI:** CNNs achieve translation equivariance by using convolutional (shared-weight) linear maps. Equivariant graph neural networks use permutation-equivariant maps. Geometric deep learning is the systematic study of building neural networks as compositions of equivariant linear maps. Transformer attention (without positional encoding) is permutation equivariant — adding positional encodings explicitly breaks this symmetry.

### O.2 Schur's Lemma and Irreducible Representations

**Schur's Lemma.** Let $T: V \to V$ be a linear map that commutes with all maps in an irreducible representation of a group $G$ (i.e., $T\rho(g) = \rho(g)T$ for all $g \in G$). Then $T = \lambda I$ for some scalar $\lambda$.

This powerful result says: the only linear maps that commute with all symmetries of an irreducible representation are scalar multiples of the identity. This constrains the form of equivariant maps.

**Application:** If attention weights must be equivariant to the representation of a certain symmetry group acting on the heads, Schur's lemma constrains the possible attention patterns.

### O.3 Representation Theory Preview

**Representation theory** studies how groups act on vector spaces via linear maps. Every group representation $\rho: G \to GL(V)$ is a group homomorphism — a map that takes group elements to invertible linear maps, preserving the group structure:
$$\rho(gh) = \rho(g)\rho(h) \quad \text{(composition respects group multiplication)}$$

This is the language in which equivariant neural networks (E(3)-equivariant networks for molecular property prediction, SE(3)-equivariant networks for robotics, permutation-equivariant networks for sets) are designed. The "weights" of an equivariant linear layer are constrained to be equivariant — and representation theory tells you exactly what form these weights can take.

---

## Appendix P: Quick Reference — Common Linear Maps in $\mathbb{R}^2$ and $\mathbb{R}^3$

### Common $2 \times 2$ Linear Maps

| Transformation | Matrix | Properties |
| --- | --- | --- |
| Rotation by $\theta$ | $\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ | Orthogonal, $\det=1$, $|\lambda|=1$ |
| Reflection across $x$-axis | $\begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}$ | Symmetric, $\det=-1$, $\lambda = \pm 1$ |
| Reflection across $y=x$ | $\begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}$ | Symmetric, $\det=-1$, $\lambda = \pm 1$ |
| Horizontal shear by $k$ | $\begin{pmatrix}1 & k \\ 0 & 1\end{pmatrix}$ | $\det=1$, $\lambda = 1$ (double) |
| Scaling by $(a, b)$ | $\begin{pmatrix}a & 0 \\ 0 & b\end{pmatrix}$ | Symmetric, $\det=ab$, $\lambda = a, b$ |
| Projection onto $x$-axis | $\begin{pmatrix}1 & 0 \\ 0 & 0\end{pmatrix}$ | Symmetric, idempotent, $\det=0$, $\lambda=0,1$ |
| Projection onto $y=x$ | $\frac{1}{2}\begin{pmatrix}1 & 1 \\ 1 & 1\end{pmatrix}$ | Symmetric, idempotent, $\lambda=0,1$ |
| Zero map | $\begin{pmatrix}0 & 0 \\ 0 & 0\end{pmatrix}$ | Rank 0, $\det=0$, $\lambda=0$ |
| Identity | $\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$ | All eigenvalues 1, $\det=1$ |

All these are linear maps. To make them affine (include translation), append a row and column in homogeneous form.


### Common $3 \times 3$ Linear Maps

| Transformation | Description | Key Properties |
| --- | --- | --- |
| Rotation around $z$-axis | $R_z(\theta)$: rotates $xy$-plane, fixes $z$ | Orthogonal, $\det=1$ |
| Reflection across $xy$-plane | $\operatorname{diag}(1,1,-1)$ | Symmetric, $\det=-1$ |
| Projection onto $xy$-plane | $\operatorname{diag}(1,1,0)$ | Symmetric, idempotent, rank 2 |
| Householder reflection | $I - 2\mathbf{n}\mathbf{n}^\top$ | Symmetric, $\det=-1$, $\lambda = 1$ (mult. 2) and $-1$ |
| Scaling | $\operatorname{diag}(a,b,c)$ | Diagonal; eigenvalues are $a,b,c$ |
| Shear | $I + s\,\mathbf{e}_i\mathbf{e}_j^\top$ ($i \neq j$) | $\det=1$; all eigenvalues 1 |

---

*End of Linear Transformations section. Continue to [§05: Orthogonality and Orthonormality](../05-Orthogonality-and-Orthonormality/notes.md).*
