[← Back to Linear Algebra Basics](../README.md) | [Next: Eigenvalues and Decompositions →](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)

---

# Vector Spaces and Subspaces

> _"A vector space is not a collection of arrows. It is a collection of anything that can be added together and scaled — and whose arithmetic obeys eight simple rules. The moment you verify those eight rules, every theorem ever proved about vectors becomes available for free."_

## Overview

A vector space is the central object of linear algebra. The abstract definition — a set with two operations satisfying eight axioms — is deceptively simple. Its power lies in universality: the same framework that governs arrows in the plane also governs polynomials, matrices, functions, probability distributions, and the weight spaces of neural networks. Every result proved once for abstract vector spaces applies simultaneously to all of these.

Subspaces are the geometric heart of the theory. A subspace is a flat, linear subset of a vector space that passes through the origin and is closed under the same operations. The column space of a weight matrix, the null space of a linear layer, the span of a set of gradient vectors, the low-rank subspace targeted by LoRA — all are subspaces. Understanding their structure is understanding how information flows, is stored, and is transformed in deep learning systems.

This chapter develops vector spaces and subspaces from intuition through rigorous definition to computational tools and AI applications. It is the conceptual foundation for everything that follows in linear algebra.

## Prerequisites

- Basic vector arithmetic (addition, scalar multiplication)
- Matrix multiplication and linear systems (Sections 02–05)
- Familiarity with ℝⁿ as a coordinate space
- Exposure to the concept of rank (Section 05)

## Companion Notebooks

| Notebook                           | Description                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demonstrations: subspace visualisation, Gram-Schmidt, four fundamental subspaces, PCA    |
| [exercises.ipynb](exercises.ipynb) | Practice problems: axiom verification, subspace proofs, basis computation, Gram-Schmidt, projections |

## Learning Objectives

After completing this section, you will:

- Verify that a given set with operations satisfies (or fails) the eight vector space axioms
- Identify and prove subspaces using the three-condition test
- Compute span, test linear independence, and extract bases via row reduction
- Identify and compute all four fundamental subspaces of a matrix
- Apply Gram-Schmidt orthogonalisation and construct projection matrices
- Understand direct sums, orthogonal complements, and quotient spaces
- Connect vector space structure to transformer architecture, LoRA, PCA, and mechanistic interpretability
- Recognise invariant subspaces and state the spectral theorem in subspace language

---

## Table of Contents

- [Vector Spaces and Subspaces](#vector-spaces-and-subspaces)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is a Vector Space?](#11-what-is-a-vector-space)
    - [1.2 Why Vector Spaces Are the Language of Deep Learning](#12-why-vector-spaces-are-the-language-of-deep-learning)
    - [1.3 The Hierarchy of Structure](#13-the-hierarchy-of-structure)
    - [1.4 The Intuition of a Subspace](#14-the-intuition-of-a-subspace)
    - [1.5 Subspaces That Appear Constantly in AI](#15-subspaces-that-appear-constantly-in-ai)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The Axioms of a Vector Space](#21-the-axioms-of-a-vector-space)
    - [2.2 Immediate Consequences of the Axioms](#22-immediate-consequences-of-the-axioms)
    - [2.3 Standard Examples of Vector Spaces](#23-standard-examples-of-vector-spaces)
    - [2.4 Non-Examples of Vector Spaces](#24-non-examples-of-vector-spaces)
    - [2.5 Vector Spaces over Other Fields](#25-vector-spaces-over-other-fields)
  - [3. Subspaces](#3-subspaces)
    - [3.1 Definition and the Three Conditions](#31-definition-and-the-three-conditions)
    - [3.2 Why These Three Conditions Suffice](#32-why-these-three-conditions-suffice)
    - [3.3 Standard Subspace Examples in ℝⁿ](#33-standard-subspace-examples-in-ℝⁿ)
    - [3.4 Non-Subspace Examples](#34-non-subspace-examples)
    - [3.5 The Two Trivial Subspaces](#35-the-two-trivial-subspaces)
  - [4. Span and Linear Combinations](#4-span-and-linear-combinations)
    - [4.1 Linear Combinations](#41-linear-combinations)
    - [4.2 The Span](#42-the-span)
    - [4.3 Geometric Pictures of Span](#43-geometric-pictures-of-span)
    - [4.4 Spanning Sets](#44-spanning-sets)
    - [4.5 Span in Terms of Matrices](#45-span-in-terms-of-matrices)
  - [5. Linear Independence](#5-linear-independence)
    - [5.1 Definition of Linear Independence](#51-definition-of-linear-independence)
    - [5.2 The Dependence Relation](#52-the-dependence-relation)
    - [5.3 Checking Linear Independence](#53-checking-linear-independence)
    - [5.4 Key Properties](#54-key-properties)
    - [5.5 Linear Independence in AI Contexts](#55-linear-independence-in-ai-contexts)
  - [6. Basis and Dimension](#6-basis-and-dimension)
    - [6.1 Definition of a Basis](#61-definition-of-a-basis)
    - [6.2 The Unique Representation Theorem](#62-the-unique-representation-theorem)
    - [6.3 Dimension](#63-dimension)
    - [6.4 Standard Bases](#64-standard-bases)
    - [6.5 Change of Basis](#65-change-of-basis)
    - [6.6 Constructing a Basis](#66-constructing-a-basis)
    - [6.7 Dimension Counting Theorems](#67-dimension-counting-theorems)
  - [7. The Four Fundamental Subspaces](#7-the-four-fundamental-subspaces)
    - [7.1 Definition of All Four Subspaces](#71-definition-of-all-four-subspaces)
    - [7.2 The Orthogonality Relations](#72-the-orthogonality-relations)
    - [7.3 Strang's Big Picture](#73-strangs-big-picture)
    - [7.4 Computing the Four Subspaces](#74-computing-the-four-subspaces)
    - [7.5 AI Interpretation of the Four Subspaces](#75-ai-interpretation-of-the-four-subspaces)
  - [8. Subspace Operations](#8-subspace-operations)
    - [8.1 Sum of Subspaces](#81-sum-of-subspaces)
    - [8.2 Intersection of Subspaces](#82-intersection-of-subspaces)
    - [8.3 Direct Sum](#83-direct-sum)
    - [8.4 Projection onto Subspaces](#84-projection-onto-subspaces)
    - [8.5 Subspace Angles and Principal Angles](#85-subspace-angles-and-principal-angles)
  - [9. Inner Product Spaces and Orthogonality](#9-inner-product-spaces-and-orthogonality)
    - [9.1 Inner Products](#91-inner-products)
    - [9.2 Orthogonality](#92-orthogonality)
    - [9.3 Gram-Schmidt Orthogonalisation](#93-gram-schmidt-orthogonalisation)
    - [9.4 The Orthogonal Complement](#94-the-orthogonal-complement)
    - [9.5 Orthogonal Bases for AI](#95-orthogonal-bases-for-ai)
  - [10. Affine Subspaces and Quotient Spaces](#10-affine-subspaces-and-quotient-spaces)
    - [10.1 Affine Subspaces](#101-affine-subspaces)
    - [10.2 Affine Combinations](#102-affine-combinations)
    - [10.3 Quotient Spaces](#103-quotient-spaces)
    - [10.4 Cosets and Their Structure](#104-cosets-and-their-structure)
  - [11. Subspaces in Functional Analysis](#11-subspaces-in-functional-analysis)
    - [11.1 Infinite-Dimensional Spaces and Closed Subspaces](#111-infinite-dimensional-spaces-and-closed-subspaces)
    - [11.2 Function Spaces Relevant to AI](#112-function-spaces-relevant-to-ai)
    - [11.3 Neural Networks as Subspaces of Function Spaces](#113-neural-networks-as-subspaces-of-function-spaces)
    - [11.4 Krylov Subspaces](#114-krylov-subspaces)
  - [12. Subspace Methods in Machine Learning](#12-subspace-methods-in-machine-learning)
    - [12.1 PCA as Subspace Finding](#121-pca-as-subspace-finding)
    - [12.2 Subspace Tracking During Training](#122-subspace-tracking-during-training)
    - [12.3 Representation Subspaces](#123-representation-subspaces)
    - [12.4 Mechanistic Interpretability Through Subspaces](#124-mechanistic-interpretability-through-subspaces)
    - [12.5 Subspace Fine-Tuning Methods](#125-subspace-fine-tuning-methods)
  - [13. Invariant Subspaces and Spectral Theory](#13-invariant-subspaces-and-spectral-theory)
    - [13.1 Invariant Subspaces](#131-invariant-subspaces)
    - [13.2 The Spectral Theorem via Invariant Subspaces](#132-the-spectral-theorem-via-invariant-subspaces)
    - [13.3 Singular Value Decomposition as Subspace Decomposition](#133-singular-value-decomposition-as-subspace-decomposition)
    - [13.4 Schur Decomposition](#134-schur-decomposition)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI (2026 Perspective)](#16-why-this-matters-for-ai-2026-perspective)
  - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Vector Space?

A vector space is a collection of objects — called **vectors** — together with two operations (addition and scalar multiplication) that behave exactly the way arrows in the plane behave. You can add any two vectors and get another vector in the same collection. You can stretch any vector by any scalar and stay in the collection. And these two operations interact with each other in the exact way you would expect from ordinary arithmetic.

The power of the abstract definition is this: once you verify that a collection satisfies the vector space axioms, every theorem ever proved about vector spaces applies immediately — whether the "vectors" are geometric arrows, polynomials, matrices, functions, probability distributions, or sequences of real numbers. You do not reprove theorems for each new setting; you simply check the axioms once and inherit a century of results for free.

The word "space" is deliberate. A vector space has geometric structure. You can talk about directions, linear combinations, independence, and dimensionality — even when the "vectors" are abstract objects that look nothing like arrows. The polynomial $1 + 2t + 3t^2$ lives in a vector space. So does the $3 \times 3$ identity matrix. So does the constant function $f(x) = 7$. In each case, the "geometry" is real and consequential, even if it is not immediately visible.

For AI, this abstraction is not philosophical — it is practical:

```text
VECTOR SPACES THAT APPEAR IN EVERY NEURAL NETWORK
════════════════════════════════════════════════════════════════════

  ℝᵈ  — embedding space
  ──── tokens, sentences, images, users are all vectors here
       arithmetic on vectors (king − man + woman ≈ queen)
       only makes sense because ℝᵈ is a vector space

  ℝᵖ  — parameter space
  ──── all weights of a neural network form a single vector in ℝᵖ
       gradient descent navigates this space
       LoRA restricts updates to a subspace of ℝᵖ

  ℝᵐˣⁿ — weight matrix space
  ──── each weight matrix is a "vector" in a higher-dimensional space
       adding matrices (as in LoRA: W + ΔW) is vector addition
       scaling a matrix by a learning rate is scalar multiplication

  ∞-dim — function space
  ──── the set of all functions a neural architecture can represent
       approximation theory asks which subspace of all functions
       a given architecture can reach
```

The abstract definition captures all of these simultaneously. That is the point.

### 1.2 Why Vector Spaces Are the Language of Deep Learning

Every major architectural concept in modern deep learning can be phrased as a statement about vector spaces or their subspaces. This is not a coincidence — it reflects the fundamental linearity of most operations in a transformer.

**Embedding spaces.** Tokens, sentences, images, and users are all represented as vectors in $\mathbb{R}^d$. This only makes sense because $\mathbb{R}^d$ is a vector space: you can meaningfully add embedding vectors, scale them, and take linear combinations. The famous word embedding arithmetic $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ is a statement about the linear structure of the embedding space. Cosine similarity — the standard measure of semantic relatedness — is an inner product operation. None of this would be coherent if the embedding space were not a vector space.

**Parameter space.** All parameters of a neural network — every weight matrix, every bias vector, every LayerNorm scale — can be concatenated into a single vector $\theta \in \mathbb{R}^p$. Gradient descent navigates this $p$-dimensional vector space. The loss landscape $\mathcal{L}: \mathbb{R}^p \to \mathbb{R}$ lives over this space. The geometry of the parameter space — its curvature, the directions of steepest descent, the flat directions — shapes every aspect of optimisation. Gradient clipping, momentum, and weight decay are all operations in the vector space $\mathbb{R}^p$.

**Residual stream.** In transformer architectures, the central computational object is the residual stream: a vector $\mathbf{x} \in \mathbb{R}^d$ that is passed from layer to layer and updated by each component:

$$\mathbf{x} \leftarrow \mathbf{x} + \text{Attention}(\mathbf{x}) + \text{MLP}(\mathbf{x})$$

The update rule makes sense only because $\mathbb{R}^d$ is a vector space — you can add the attention output (a vector in $\mathbb{R}^d$) and the MLP output (also a vector in $\mathbb{R}^d$) to the residual stream. Each layer writes a new vector into the same shared space. All components communicate through this one $d$-dimensional vector space. The subspace structure of $\mathbb{R}^d$ — which directions are "owned" by which components — is what mechanistic interpretability studies.

**Function spaces.** Neural networks represent functions. The set of all functions representable by a given architecture with bounded parameters is a subset of the infinite-dimensional function space $L^2$. Approximation theory studies which subspaces of $L^2$ can be well-approximated by which architectures. The universal approximation theorem is a statement about density in a function space. Understanding which functions a model can and cannot represent requires understanding subspaces of infinite-dimensional vector spaces.

**Gradient space.** At each training step, the gradient $\nabla \mathcal{L}(\theta)$ is a vector in the same parameter space $\mathbb{R}^p$. Gradient accumulation (summing gradients across microbatches) is vector addition. Gradient clipping (bounding the norm) is a scaling operation. Momentum (maintaining a weighted sum of past gradients) is a linear combination. All of these rely on the vector space structure of $\mathbb{R}^p$. Empirically, the gradient vectors observed during training lie in a much lower-dimensional subspace of $\mathbb{R}^p$ — a fact that motivates LoRA and related methods.

**Tangent spaces.** The loss surface is a manifold embedded in $\mathbb{R}^p$. At each point $\theta$, the tangent space — the set of directions one can move while remaining (to first order) on the manifold — is a vector space. Natural gradient methods use the geometry of these tangent spaces; the Fisher information matrix defines a metric on the parameter manifold that is fundamentally a statement about the inner product structure of the tangent space.

### 1.3 The Hierarchy of Structure

Vector spaces occupy a specific level in a hierarchy of mathematical structures. Each level adds additional structure on top of the previous, enabling more refined geometric notions:

```text
HIERARCHY OF MATHEMATICAL STRUCTURE
════════════════════════════════════════════════════════════════════

  Set  (no structure)
      ↓
      + addition and scalar multiplication obeying 8 axioms
      ↓
  Vector Space  (linear structure: add, scale)
      — can speak of linear combinations, span, independence, basis
      — can speak of dimension
      ↓
      + norm  ‖v‖  (a function measuring "length")
      ↓
  Normed Space  (metric structure: can measure distances)
      — can speak of convergence, continuity, open/closed sets
      — can speak of the "size" of a vector
      ↓
      + inner product  ⟨u, v⟩  (a function measuring "alignment")
      ↓
  Inner Product Space  (angular structure: can measure angles)
      — can speak of orthogonality, projections, cosine similarity
      — can speak of the "angle" between two vectors
      ↓
      + completeness  (every Cauchy sequence converges in the space)
      ↓
  Hilbert Space  (complete inner product space)
      — can speak of infinite sums that converge
      — can speak of Fourier decompositions
      — the correct setting for quantum mechanics and function spaces

  ────────────────────────────────────────────────────────────────
  Key facts:
  • ℝⁿ (Euclidean space) satisfies ALL levels simultaneously.
  • Every finite-dimensional inner product space is a Hilbert space
    (completeness is automatic in finite dimensions).
  • Infinite-dimensional function spaces (L², Sobolev spaces)
    require the Hilbert structure to behave well.
  • AI embedding spaces are finite-dimensional: all levels apply.
  • RKHS for kernel methods: infinite-dimensional Hilbert spaces.
```

The hierarchy matters because each additional level of structure enables new theorems. In a bare vector space, you can talk about linear combinations and dimension. Add a norm: you can measure convergence. Add an inner product: you can define orthogonality and projections. Add completeness: you can take limits of infinite sequences. For AI, finite-dimensional vector spaces with inner products (Hilbert spaces) are the default working environment. Function spaces require infinite-dimensional Hilbert structure.

### 1.4 The Intuition of a Subspace

A subspace is a vector space living inside a larger vector space — a flat, linear subset that passes through the origin and is closed under the same two operations.

**"Flat" and "through the origin"** are the two defining geometric properties:

- A line through the origin in $\mathbb{R}^3$ is a subspace. A line not through the origin is not.
- A plane through the origin is a subspace. A plane not through the origin is not.
- A sphere, a circle, a parabola, a curved surface — none are subspaces.
- The positive quadrant $\{(x,y) : x \geq 0, y \geq 0\}$ is not a subspace (not closed under scalar multiplication by $-1$).

**Why "through the origin"?** The zero vector must be in any subspace. This follows from closure under scalar multiplication: if $\mathbf{v}$ is in the subspace, then $0 \cdot \mathbf{v} = \mathbf{0}$ must also be in the subspace. A subset that doesn't contain $\mathbf{0}$ cannot be a subspace.

**Why "flat"?** Take any two vectors in the subspace. Their sum must also be in the subspace. Multiply either by any scalar — must still be in the subspace. This forces the subset to be "straight". There can be no curves, bends, or boundaries. If the subspace contains a vector $\mathbf{v}$, it must contain all scalar multiples $\alpha\mathbf{v}$ for every $\alpha \in \mathbb{R}$ — the entire line through the origin in the direction of $\mathbf{v}$.

```text
SUBSPACES AND NON-SUBSPACES IN ℝ² AND ℝ³
════════════════════════════════════════════════════════════════════

  ℝ²:                         ℝ³:

  ✓  The whole plane ℝ²       ✓  All of ℝ³
  ✓  Any line through origin  ✓  Any plane through origin
  ✓  {0}                      ✓  Any line through origin
                              ✓  {0}

  ✗  Line not through origin  ✗  Plane not through origin
  ✗  Half-plane               ✗  Sphere / cylinder / cone
  ✗  Unit circle              ✗  Positive octant
  ✗  Filled disk              ✗  Affine subspace (offset from origin)
  ✗  Upper half-plane

  Test: does it contain 0? is it closed under + and scaling?
  If any answer is NO → not a subspace.
```

**For AI, the subspace concept is concrete.** The column space of a weight matrix is a subspace of the output space. The null space is a subspace of the input space. The set of all weight updates targeted by LoRA is a subspace of the weight matrix space. The set of all probability vectors (probabilities summing to 1) is _not_ a subspace (it does not contain the zero vector and is not closed under vector addition). This distinction matters: arithmetic that is meaningful in a vector space may be meaningless in a non-subspace.

### 1.5 Subspaces That Appear Constantly in AI

The following subspaces recur throughout deep learning theory, practice, and interpretability. Each is worth understanding geometrically before encountering it formally.

**Column space col(W).** For a weight matrix $W \in \mathbb{R}^{m \times n}$, the column space is the set of all vectors $W\mathbf{x}$ as $\mathbf{x}$ ranges over $\mathbb{R}^n$. It is the set of all reachable outputs of the linear layer. If $\text{rank}(W) = r < m$, then only an $r$-dimensional subspace of $\mathbb{R}^m$ can be produced by this layer regardless of the input. The remaining $m - r$ dimensions of the output space are always zero from this layer's contribution.

**Null space null(W).** The null space is the set of all inputs $\mathbf{x}$ that produce zero output: $W\mathbf{x} = \mathbf{0}$. It is the "invisible" subspace — directions in the input that this layer completely ignores. A vector in the null space carries no information through the layer. Its dimension is $n - \text{rank}(W)$. Mechanistically: if a component of the residual stream lies in the null space of all query and key projections of some head, that component is invisible to that head.

**Row space row(W).** The row space is the span of the rows of $W$, equivalently the column space of $W^\top$. It is the set of input directions that $W$ "listens to". The projection of an input onto the row space determines the output; the projection onto the null space is discarded.

**Attention subspace.** Each attention head in a transformer has query and key projection matrices $W_Q, W_K \in \mathbb{R}^{d \times d_k}$. The attention scores are computed in the $d_k$-dimensional subspace defined by these projections. The head is completely blind to directions in $\mathbb{R}^d$ orthogonal to the column spaces of $W_Q$ and $W_K$. The value and output projections $W_V, W_O$ similarly define the subspace that the head writes to. Attention is fundamentally a subspace operation: it reads from one subspace and writes to another.

**Gradient subspace.** The gradient $\nabla \mathcal{L}(\theta_t)$ at step $t$ is a vector in $\mathbb{R}^p$. Over the course of training, these gradient vectors empirically lie in a subspace of $\mathbb{R}^p$ whose dimension is much smaller than $p$. Gur-Ari et al. (2018) showed that the top eigenspace of the gradient covariance matrix — the "gradient subspace" — has effective dimension in the hundreds or thousands, not in the hundreds of millions. This is the empirical foundation for LoRA: if gradients live in a low-dimensional subspace, it may suffice to restrict weight updates to that subspace.

**Residual stream subspace.** Mechanistic interpretability reveals that different components of a transformer write to different subspaces of the $d$-dimensional residual stream. Each attention head "owns" a subspace (defined by its OV circuit) and communicates with other components through that subspace. The residual stream is a shared message-passing space, and the subspace structure determines which components can communicate with which.

**LoRA update subspace.** Low-Rank Adaptation restricts the weight update $\Delta W = B A^\top$ where $B \in \mathbb{R}^{m \times r}$ and $A \in \mathbb{R}^{n \times r}$. This constrains $\Delta W$ to a rank-$r$ subspace of $\mathbb{R}^{m \times n}$: the subspace of matrices with column space contained in $\text{col}(B)$. The rank $r$ is literally the dimension of the subspace being searched. Choosing $r$ is choosing the dimension of the subspace.

### 1.6 Historical Timeline

The concept of a vector space did not emerge from a single insight — it crystallised gradually over seven decades as mathematicians in different countries, working on different problems, all needed the same abstract structure:

| Year        | Person / Event                                           | Significance                                                                                                                                                                                                                 |
| ----------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1827        | **Möbius** — barycentric coordinates                     | Linear combinations with fixed coefficient sum; early subspace thinking embedded in geometry                                                                                                                                 |
| 1844        | **Grassmann** — _Ausdehnungslehre_ (Theory of Extension) | Abstract $n$-dimensional linear spaces, vector algebra, exterior products; 50 years ahead of its time; largely ignored until the 1880s                                                                                       |
| 1858        | **Cayley** — matrix algebra                              | Matrices as the natural representation of linear maps; implicit use of vector space structure in $\mathbb{R}^n$                                                                                                              |
| 1888        | **Peano** — _Calcolo Geometrico_                         | First rigorous axiomatic definition of a vector space (Peano called them "linear systems"); the modern eight axioms are essentially unchanged from his formulation                                                           |
| 1900–10     | **Hilbert** — infinite-dimensional spaces                | Spectral theory of operators; sequences as vectors; laid the foundation for quantum mechanics and functional analysis; introduced Hilbert spaces                                                                             |
| 1920s       | **Banach** — normed complete spaces                      | Generalised Hilbert spaces to settings without an inner product; functional analysis matures as a discipline                                                                                                                 |
| 1929        | **von Neumann** — Hilbert space formalism                | Rigorous foundation for quantum mechanics as operators on Hilbert space; spectral theorem in full generality                                                                                                                 |
| 1930s–60s   | **Bourbaki** — abstract algebra                          | Vector spaces as modules over fields; categorical perspective; the modern textbook treatment of linear algebra                                                                                                               |
| 1935        | **Whitney** — tangent bundles                            | A vector space (tangent space) attached to every point of a smooth manifold; modern differential geometry                                                                                                                    |
| 1950s–90s   | **Krylov subspace methods**                              | CG, GMRES, Lanczos: iterative solvers that expand a nested sequence of subspaces; subspace geometry as a computational algorithm                                                                                             |
| 1970s–90s   | **LAPACK / numerical linear algebra**                    | Practical algorithms for computing with subspaces (QR, SVD, Gram-Schmidt); subspace methods become usable at scale                                                                                                           |
| 1901 / 1991 | **Pearson / Turk–Pentland** — PCA                        | Pearson (1901) defines principal component analysis; Turk and Pentland (1991) apply PCA to face recognition ("eigenfaces"); subspace learning as a machine learning tool                                                     |
| 2013        | **Mikolov et al.** — word2vec                            | Words as vectors in $\mathbb{R}^d$; semantic arithmetic; gender direction as a 1D subspace; demonstrated that natural language semantics has linear subspace structure                                                       |
| 2018        | **Gur-Ari et al.** — gradient subspace                   | The "gradient subspace" of LLM training has dimension $\ll p$; gradient updates live in a low-dimensional subspace of parameter space                                                                                        |
| 2020        | **Aghajanyan et al.** — intrinsic dimensionality         | Fine-tuning gradient updates for GPT-2 lie in a subspace of dimension $\approx 200$ out of $p \approx 1.5 \times 10^8$; formal evidence for low-dimensional fine-tuning                                                      |
| 2021        | **Hu et al.** — LoRA                                     | Explicit restriction of weight updates to a rank-$r$ subspace; subspace constraint as an inductive bias; state-of-the-art parameter-efficient fine-tuning                                                                    |
| 2022        | **Elhage et al.** — superposition hypothesis             | Features in LLMs are represented as directions (1D subspaces) in the residual stream; features outnumber dimensions, forcing non-orthogonal superposition; subspace geometry is the language of mechanistic interpretability |
| 2024        | **DeepSeek-V2/V3** — MLA                                 | Multi-head Latent Attention: KV projections compressed to a rank-$r$ subspace; the KV information is architecturally constrained to a low-dimensional subspace; 5.75× memory reduction                                       |

The arc of this timeline is clear: what began as abstract geometry became, over 180 years, the core language of AI architecture and efficiency.

---

## 2. Formal Definitions

> **Recall:** [Vectors and Spaces §3](../01-Vectors-and-Spaces/notes.md#3-abstract-vector-spaces) introduced the vector space axioms with concrete geometric intuition ($\mathbb{R}^n$, function spaces, polynomial spaces). This section re-derives them in full axiomatic rigour — the goal here is _provability_, not intuition. The two sections are complementary: §01 tells you _what_ the axioms describe; this section tells you _what follows_ from them.

### 2.1 The Axioms of a Vector Space

A **vector space** over a field $F$ (we use $F = \mathbb{R}$ throughout unless stated otherwise) is a set $V$ together with two operations:

- **Addition**: $+: V \times V \to V$ — takes two vectors, produces a vector
- **Scalar multiplication**: $\cdot: \mathbb{R} \times V \to V$ — takes a scalar and a vector, produces a vector

satisfying the following eight axioms for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and all $\alpha, \beta \in \mathbb{R}$:

**Axioms for addition:**

| #   | Name                  | Statement                                                                                                        |
| --- | --------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1   | **Commutativity**     | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$                                                              |
| 2   | **Associativity**     | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$                                |
| 3   | **Additive identity** | $\exists\, \mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v}$               |
| 4   | **Additive inverse**  | For each $\mathbf{v} \in V$, $\exists\, (-\mathbf{v}) \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |

**Axioms for scalar multiplication:**

| #   | Name                        | Statement                                            |
| --- | --------------------------- | ---------------------------------------------------- |
| 5   | **Multiplicative identity** | $1 \cdot \mathbf{v} = \mathbf{v}$                    |
| 6   | **Compatibility**           | $\alpha(\beta \mathbf{v}) = (\alpha\beta)\mathbf{v}$ |

**Distributivity (linking the two operations):**

| #   | Name                       | Statement                                                               |
| --- | -------------------------- | ----------------------------------------------------------------------- |
| 7   | **Scalar over vector sum** | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ |
| 8   | **Vector over scalar sum** | $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$       |

These eight axioms are the **complete** definition. Anything satisfying all eight is a vector space; anything failing even one is not. The axioms are minimal: each is independent of the others; removing any one produces a strictly weaker structure.

Notice what the axioms do and do not say:

- They say nothing about what the "vectors" look like. Arrows, polynomials, matrices, functions — all are equally valid.
- They say nothing about angles, lengths, or orthogonality. Those require additional structure (inner product).
- They say nothing about convergence. That requires a topology (norm or metric).
- They require closure under both operations: the result of adding two vectors must be back in $V$; scaling a vector must stay in $V$.

```text
THE AXIOMS AS CLOSURE CONDITIONS
════════════════════════════════════════════════════════════════════

  Axioms 1–4: addition is a well-behaved operation on V
              V is an abelian group under +

  Axioms 5–6: scalar multiplication interacts correctly with itself

  Axioms 7–8: scalar multiplication distributes over both kinds of +
              (over vector sums and over scalar sums)

  Together: V behaves like "an ℝⁿ that we haven't yet coordinatised"
```

### 2.2 Immediate Consequences of the Axioms

The eight axioms imply several useful facts that are not axioms themselves. These are theorems — they must be proved from the axioms — but they follow quickly:

**Uniqueness of the zero vector.** If $\mathbf{0}$ and $\mathbf{0}'$ both satisfy Axiom 3, then:
$$\mathbf{0} = \mathbf{0} + \mathbf{0}' = \mathbf{0}'$$
The first equality uses $\mathbf{0}'$ as the identity; the second uses $\mathbf{0}$. So the zero vector is unique.

**Uniqueness of additive inverses.** If $\mathbf{v} + \mathbf{w} = \mathbf{0}$ and $\mathbf{v} + \mathbf{w}' = \mathbf{0}$, then:
$$\mathbf{w} = \mathbf{w} + \mathbf{0} = \mathbf{w} + (\mathbf{v} + \mathbf{w}') = (\mathbf{w} + \mathbf{v}) + \mathbf{w}' = \mathbf{0} + \mathbf{w}' = \mathbf{w}'$$
So the additive inverse $-\mathbf{v}$ is unique for each $\mathbf{v}$.

**$0 \cdot \mathbf{v} = \mathbf{0}$** (the zero scalar times any vector = zero vector):
$$0 \cdot \mathbf{v} = (0 + 0) \cdot \mathbf{v} = 0 \cdot \mathbf{v} + 0 \cdot \mathbf{v}$$
Subtract $0 \cdot \mathbf{v}$ from both sides (add the additive inverse of $0 \cdot \mathbf{v}$):
$$\mathbf{0} = 0 \cdot \mathbf{v}$$

**$\alpha \cdot \mathbf{0} = \mathbf{0}$** (any scalar times the zero vector = zero vector):
$$\alpha \cdot \mathbf{0} = \alpha(\mathbf{0} + \mathbf{0}) = \alpha \cdot \mathbf{0} + \alpha \cdot \mathbf{0}$$
Subtract $\alpha \cdot \mathbf{0}$: $\mathbf{0} = \alpha \cdot \mathbf{0}$.

**$(-1) \cdot \mathbf{v} = -\mathbf{v}$** (multiplying by $-1$ gives the additive inverse):
$$\mathbf{v} + (-1)\mathbf{v} = 1 \cdot \mathbf{v} + (-1)\mathbf{v} = (1 + (-1))\mathbf{v} = 0 \cdot \mathbf{v} = \mathbf{0}$$
So $(-1)\mathbf{v}$ is the additive inverse of $\mathbf{v}$, i.e., $(-1)\mathbf{v} = -\mathbf{v}$.

**Cancellation law for scalars.** If $\alpha\mathbf{v} = \mathbf{0}$, then either $\alpha = 0$ or $\mathbf{v} = \mathbf{0}$:

_Proof:_ Suppose $\alpha \neq 0$. Then $\alpha$ has a multiplicative inverse $\alpha^{-1}$ in $\mathbb{R}$. Multiply both sides:
$$\mathbf{v} = 1 \cdot \mathbf{v} = (\alpha^{-1}\alpha)\mathbf{v} = \alpha^{-1}(\alpha\mathbf{v}) = \alpha^{-1} \cdot \mathbf{0} = \mathbf{0}$$

This last fact is particularly useful in linear independence arguments: if a scalar multiple of a vector is zero, either the scalar is zero or the vector is zero — no third option.

### 2.3 Standard Examples of Vector Spaces

Understanding which objects form vector spaces — and why — is as important as knowing the axioms. Here are the key examples, from most concrete to most abstract.

---

**$\mathbb{R}^n$ (Euclidean space)**

- **Vectors:** $n$-tuples of real numbers $(x_1, x_2, \ldots, x_n)$
- **Addition:** $(x_1,\ldots,x_n) + (y_1,\ldots,y_n) = (x_1+y_1,\ldots,x_n+y_n)$ (componentwise)
- **Scalar multiplication:** $\alpha(x_1,\ldots,x_n) = (\alpha x_1,\ldots,\alpha x_n)$ (componentwise)
- **Zero vector:** $(0, 0, \ldots, 0)$
- **Additive inverse:** $-(x_1,\ldots,x_n) = (-x_1,\ldots,-x_n)$
- **Dimension:** $n$

This is the prototype. Every finite-dimensional real vector space is isomorphic to $\mathbb{R}^n$ for some $n$. All eight axioms follow directly from the arithmetic of real numbers applied componentwise. The AI embedding space $\mathbb{R}^d$ is exactly this.

---

**$\mathbb{R}^{m \times n}$ (space of $m \times n$ real matrices)**

- **Vectors:** $m \times n$ matrices $A$ with real entries
- **Addition:** entrywise addition $(A + B)_{ij} = A_{ij} + B_{ij}$
- **Scalar multiplication:** $(\alpha A)_{ij} = \alpha A_{ij}$
- **Zero:** zero matrix $\mathbf{0}_{m \times n}$
- **Dimension:** $mn$ (each of the $mn$ entries is an independent coordinate)

This is the vector space of all weight matrices of a fixed shape. Gradient descent in a single layer's weight space is gradient descent in $\mathbb{R}^{m \times n}$. When LoRA writes $W \leftarrow W + BA^\top$, it is performing vector addition in $\mathbb{R}^{m \times n}$. The learning rate times the gradient, $\eta \nabla_W \mathcal{L}$, is a scalar multiplication in $\mathbb{R}^{m \times n}$.

---

**$\mathcal{P}_n$ (polynomials of degree $\leq n$)**

- **Vectors:** polynomials $p(t) = a_0 + a_1 t + a_2 t^2 + \cdots + a_n t^n$
- **Addition:** standard polynomial addition: $(p+q)(t) = p(t) + q(t)$
- **Scalar multiplication:** $(\alpha p)(t) = \alpha p(t)$
- **Zero:** the zero polynomial $p(t) = 0$ (all coefficients zero)
- **Dimension:** $n + 1$ (the coefficients $a_0, a_1, \ldots, a_n$ are the $n+1$ coordinates)

The vectors here "look like" arrows in $\mathbb{R}^{n+1}$ (just replace each polynomial with its coefficient vector), but the polynomial structure is richer — you can evaluate $p(t)$ at any point, differentiate, integrate, and compose. The vector space structure is the foundation; the polynomial structure is additional.

---

**$\mathcal{P}$ (all polynomials, any degree)**

- **Vectors:** polynomials of any (finite) degree
- **Operations:** same as $\mathcal{P}_n$
- **Zero:** zero polynomial
- **Dimension:** $\infty$ (countably infinite; basis $= \{1, t, t^2, t^3, \ldots\}$)

Note that the sum of two polynomials of degree $\leq n$ is again a polynomial of degree $\leq n$, so $\mathcal{P}_n$ is a subspace of $\mathcal{P}$. This is our first example of a subspace inside a larger space.

---

**$C([a, b])$ (continuous functions on $[a,b]$)**

- **Vectors:** continuous functions $f: [a,b] \to \mathbb{R}$
- **Addition:** $(f + g)(t) = f(t) + g(t)$ (pointwise)
- **Scalar multiplication:** $(\alpha f)(t) = \alpha f(t)$ (pointwise)
- **Zero:** constant zero function $f(t) = 0$
- **Dimension:** $\infty$ (uncountably infinite dimensional)

The sum of two continuous functions is continuous; a scalar multiple of a continuous function is continuous. All eight axioms follow from corresponding facts about real numbers applied pointwise. The space $C([a,b])$ is the natural setting for physics-informed neural networks (PINNs), neural ODEs, and any application where the model represents a continuous function.

---

**$L^2([a,b])$ (square-integrable functions)**

- **Vectors:** (equivalence classes of) functions $f: [a,b] \to \mathbb{R}$ with $\int_a^b f(t)^2 \, dt < \infty$
- **Operations:** pointwise, same as $C([a,b])$
- **Inner product:** $\langle f, g \rangle = \int_a^b f(t) g(t) \, dt$; induced norm $\|f\| = \sqrt{\langle f, f \rangle}$
- **Dimension:** $\infty$; a separable Hilbert space

$L^2$ is the correct setting for Fourier analysis (the Fourier basis $\{1, \sin(t), \cos(t), \sin(2t), \ldots\}$ is an orthonormal basis for $L^2([0, 2\pi])$), for quantum mechanics (wavefunctions are unit vectors in $L^2$), and for kernel methods (the RKHS of a translation-invariant kernel is a subspace of $L^2$). Every finite-dimensional vector space with an inner product is a Hilbert space; $L^2$ is the prototypical infinite-dimensional Hilbert space.

---

**$\ell^2$ (square-summable sequences)**

- **Vectors:** infinite sequences $(x_1, x_2, x_3, \ldots)$ with $\sum_{i=1}^\infty x_i^2 < \infty$
- **Operations:** componentwise
- **Inner product:** $\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^\infty x_i y_i$
- **Dimension:** $\infty$; a separable Hilbert space; isomorphic to $L^2([0,1])$

The infinite-dimensional analogue of $\mathbb{R}^n$. Relevant as the theoretical limit of embedding spaces as vocabulary size $|V| \to \infty$.

---

**$\{\mathbf{0}\}$ (trivial vector space)**

- The set containing only the zero vector
- Addition: $\mathbf{0} + \mathbf{0} = \mathbf{0}$; scalar multiplication: $\alpha \cdot \mathbf{0} = \mathbf{0}$
- The smallest vector space; dimension 0
- It is a subspace of every vector space

### 2.4 Non-Examples of Vector Spaces

Understanding which sets fail to be vector spaces — and precisely which axiom they violate — sharpens intuition and prevents common errors in AI applications.

**$\mathbb{R}_{+}^n = \{(x_1,\ldots,x_n) : x_i > 0\}$ (positive reals)**

Fails closure under scalar multiplication and additive inverse: $(-1) \cdot (1, 2, 3) = (-1, -2, -3) \notin \mathbb{R}_{+}^n$. Also fails to contain the zero vector. This is why you cannot do linear algebra directly in probability space (where all entries must be positive).

**$S^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| = 1\}$ (unit sphere)**

Fails closure under addition ($\|\mathbf{u} + \mathbf{v}\| \neq 1$ in general) and scalar multiplication ($\|2\mathbf{u}\| = 2 \neq 1$). Fails to contain the zero vector. Normalised embedding vectors live on the unit sphere — a non-subspace. This is why arithmetic on normalised vectors (like averaging two unit vectors) is not straightforward; the result must be renormalised.

**$\Delta^{n-1} = \{\mathbf{p} \in \mathbb{R}^n : p_i \geq 0,\ \sum_i p_i = 1\}$ (probability simplex)**

Fails closure under addition: $\mathbf{p} + \mathbf{q}$ has $\sum_i(p_i + q_i) = 2 \neq 1$. Fails to contain the zero vector. Fails scalar multiplication: $2\mathbf{p}$ has $\sum_i 2p_i = 2 \neq 1$. The simplex is an _affine_ subspace (it is a translate of a linear subspace), not a vector subspace.

This is a crucial AI pitfall: softmax outputs live in the interior of $\Delta^{|V|}$. Averaging two softmax output vectors does not produce a valid probability distribution in the same parameterisation. Interpolation, averaging, and arithmetic must be done in **logit space** (pre-softmax), which _is_ a vector space.

**$\mathbb{Z}^n$ (integer vectors)**

Closed under addition (integer + integer = integer) but fails closure under scalar multiplication over $\mathbb{R}$: $\pi \cdot (1, 0, \ldots, 0) = (\pi, 0, \ldots, 0) \notin \mathbb{Z}^n$. The integers form a module over $\mathbb{Z}$ (a more general structure), but not a vector space over $\mathbb{R}$.

**$\{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{b}\}$ with $\mathbf{b} \neq \mathbf{0}$ (inhomogeneous linear system)**

Does not contain the zero vector ($A \cdot \mathbf{0} = \mathbf{0} \neq \mathbf{b}$). Not closed under addition: $A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{b} + \mathbf{b} = 2\mathbf{b} \neq \mathbf{b}$. This is an affine subspace (a coset of the null space of $A$), not a vector subspace. The general solution to $A\mathbf{x} = \mathbf{b}$ is $\mathbf{x}_p + \text{null}(A)$, where $\mathbf{x}_p$ is any particular solution — a coset structure, not a subspace structure.

```text
SUMMARY: WHY COMMON AI SETS FAIL TO BE VECTOR SPACES
════════════════════════════════════════════════════════════════════

  Softmax outputs (interior of Δ):
      ✗ no zero vector   ✗ not closed under +   ✗ not closed under scaling
      → Do arithmetic in logit space, not probability space

  Normalised embeddings (unit sphere):
      ✗ no zero vector   ✗ not closed under +   ✗ not closed under scaling
      → Spherical interpolation (slerp) for moving on the sphere

  Integer weight tensors (quantised models):
      ✓ closed under +   ✗ not closed under ℝ-scaling
      → Module over ℤ; must dequantise before doing real linear algebra

  Solution set Ax = b, b ≠ 0:
      ✗ no zero vector   ✗ not closed under +
      → Affine subspace = coset of null(A); add particular solution last
```

### 2.5 Vector Spaces over Other Fields

While we work over $\mathbb{R}$ throughout, the definition of a vector space makes sense over any field $F$. The most relevant alternatives for AI:

**$\mathbb{C}^n$ (complex vector space).** Each scalar is complex; each component is complex. Used in quantum computing, signal processing, and complex-valued Fourier analysis. Dimension $n$ over $\mathbb{C}$, but $2n$ over $\mathbb{R}$ (since each complex number requires 2 real numbers). The inner product becomes sesquilinear: $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\dagger \mathbf{v} = \sum_i \overline{u_i} v_i$ (conjugate of first argument). Complex exponentials $e^{i\omega t}$ are the natural eigenfunctions of differentiation; Fourier analysis over $\mathbb{C}$ is cleaner than over $\mathbb{R}$.

**$\mathbb{F}_2^n$ (binary vectors).** Field $\mathbb{F}_2 = \{0, 1\}$ with arithmetic modulo 2: $1 + 1 = 0$. Addition = XOR; multiplication = AND. Used in error-correcting codes (Hamming codes, LDPC codes, turbo codes) and cryptography. All of linear algebra carries over: basis, span, null space, rank — all computable over $\mathbb{F}_2$ by the same row reduction algorithm, with arithmetic mod 2.

**$\mathbb{F}_p^n$ (vectors over finite fields).** For prime $p$, $\mathbb{F}_p = \{0, 1, \ldots, p-1\}$ with arithmetic mod $p$. Used in coding theory, cryptography (elliptic curve cryptography over finite fields), and randomised algorithms. Reed-Solomon codes are linear codes over $\mathbb{F}_p$ for large $p$; the generator matrix spans the code as a subspace of $\mathbb{F}_p^n$.

**Practical AI note.** Transformers use real vector spaces $\mathbb{R}^d$ (or approximately real, since floating-point arithmetic is finite-precision). Quantum transformers (theoretical) would use $\mathbb{C}^d$. Reliable communication protocols for distributed training use $\mathbb{F}_2$ codes for error correction. GPU hardware operates over approximately $\mathbb{R}$ with format-specific precision (BF16, FP8, INT4).

---

## 3. Subspaces

### 3.1 Definition and the Three Conditions

A subset $W \subseteq V$ is a **subspace** of $V$ if $W$ is itself a vector space under the same operations inherited from $V$.

Checking all eight axioms would be redundant. The operations themselves are inherited from $V$: the addition of two elements of $W$ uses the same addition formula as in $V$, and similarly for scalar multiplication. Because these operations are already defined and well-behaved in $V$, the axioms involving equalities (commutativity, associativity, distributivity) are automatically satisfied for elements of $W$ — they hold for all elements of $V$, so in particular they hold for elements of $W \subseteq V$.

What can fail is **closure**: the result of an operation on elements of $W$ might fall outside $W$. And the zero vector might not be in $W$. So only three conditions must be verified:

> **Subspace Test.** A subset $W \subseteq V$ is a subspace of $V$ if and only if:
>
> 1. **Non-empty / contains zero:** $\mathbf{0} \in W$
> 2. **Closed under addition:** for all $\mathbf{u}, \mathbf{w} \in W$: $\mathbf{u} + \mathbf{w} \in W$
> 3. **Closed under scalar multiplication:** for all $\mathbf{w} \in W$ and $\alpha \in \mathbb{R}$: $\alpha\mathbf{w} \in W$

Equivalently, conditions 2 and 3 can be combined into a single condition:

> **Combined condition:** For all $\mathbf{u}, \mathbf{w} \in W$ and all $\alpha, \beta \in \mathbb{R}$: $\alpha\mathbf{u} + \beta\mathbf{w} \in W$

This single condition says that $W$ is closed under all linear combinations of its elements. It is often the fastest way to verify that something is a subspace: show that every linear combination of elements of $W$ remains in $W$.

**Strategy for proving subspace:**

1. Show $W$ is non-empty (usually by showing $\mathbf{0} \in W$ directly)
2. Take arbitrary $\mathbf{u}, \mathbf{w} \in W$ and arbitrary $\alpha, \beta \in \mathbb{R}$
3. Compute $\alpha\mathbf{u} + \beta\mathbf{w}$ and verify it satisfies the defining property of $W$

**Strategy for disproving subspace:**

- Find a specific counterexample: either $\mathbf{0} \notin W$, or two elements whose sum is not in $W$, or an element whose scalar multiple is not in $W$.

### 3.2 Why These Three Conditions Suffice

Each condition handles a different potential failure:

**Condition 1 (contains zero):** Without $\mathbf{0}$, $W$ has no additive identity, so Axiom 3 fails. More fundamentally: if $\mathbf{0} \notin W$, then $W$ cannot be a vector space under the same operations as $V$, because the additive identity is uniquely determined by the operations (we proved this in §2.2).

Note that showing $\mathbf{0} \in W$ also shows $W$ is non-empty, which is required for $W$ to be a set with meaningful structure. In practice, checking $\mathbf{0} \in W$ first is the quickest way to rule out non-subspaces: if the zero vector fails to satisfy the defining property of $W$, we are done.

**Condition 2 (closed under addition):** Without this, addition is not even a well-defined operation on $W$ — adding two "vectors" in $W$ might produce something outside $W$. If the operation leaves the set, the set cannot be a vector space under that operation.

**Condition 3 (closed under scalar multiplication):** Without this, scalar multiplication is not well-defined on $W$ — scaling a vector in $W$ might produce something outside $W$. This failure also implies Axiom 4 (additive inverse) fails: since $(-1) \cdot \mathbf{v}$ must be in $W$ for each $\mathbf{v} \in W$, closure under scalar multiplication is required for inverses to exist.

**The remaining axioms are inherited automatically.** For example, commutativity: for any $\mathbf{u}, \mathbf{v} \in W \subseteq V$, we have $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ because this equality holds in $V$ for all vectors, and $\mathbf{u}, \mathbf{v}$ are in particular vectors in $V$. The same argument applies to associativity and both distributivity laws. These properties are free.

### 3.3 Standard Subspace Examples in ℝⁿ

**Lines through the origin in $\mathbb{R}^2$ and $\mathbb{R}^3$**

$W = \{(t, 2t) : t \in \mathbb{R}\} \subseteq \mathbb{R}^2$ — the line $y = 2x$.

- Contains $\mathbf{0}$: set $t = 0$ → $(0, 0) \in W$ ✓
- Closed under $+$: $(t, 2t) + (s, 2s) = (t+s, 2(t+s)) \in W$ ✓
- Closed under $\cdot$: $\alpha(t, 2t) = (\alpha t, 2\alpha t) \in W$ ✓

Dimension: 1. Basis: $\{(1, 2)^\top\}$. This is a 1-dimensional subspace of $\mathbb{R}^2$ — a line through the origin.

---

**Planes through the origin in $\mathbb{R}^3$**

$W = \{(x, y, z) \in \mathbb{R}^3 : ax + by + cz = 0\}$ — a homogeneous linear equation.

- Contains $\mathbf{0}$: $a \cdot 0 + b \cdot 0 + c \cdot 0 = 0$ ✓
- Closed under $+$: if $ax + by + cz = 0$ and $ax' + by' + cz' = 0$, then $a(x+x') + b(y+y') + c(z+z') = 0$ ✓ (by linearity of the equation)
- Closed under $\cdot$: if $ax + by + cz = 0$, then $a(\alpha x) + b(\alpha y) + c(\alpha z) = \alpha(ax+by+cz) = 0$ ✓

Dimension: 2 (one homogeneous linear constraint on 3 variables leaves 2 free variables).

---

**Null space of a matrix (solution set of a homogeneous system)**

$W = \text{null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$ for $A \in \mathbb{R}^{m \times n}$.

- Contains $\mathbf{0}$: $A \cdot \mathbf{0} = \mathbf{0}$ ✓
- Closed under $+$: $A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} = \mathbf{0} + \mathbf{0} = \mathbf{0}$ ✓
- Closed under $\cdot$: $A(\alpha \mathbf{x}) = \alpha A\mathbf{x} = \alpha \cdot \mathbf{0} = \mathbf{0}$ ✓

The null space is always a subspace of $\mathbb{R}^n$, regardless of $A$. By the Rank-Nullity Theorem (§6.7): $\dim(\text{null}(A)) = n - \text{rank}(A)$.

---

**Column space of a matrix**

$W = \text{col}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$ for $A \in \mathbb{R}^{m \times n}$.

- Contains $\mathbf{0}$: $A \cdot \mathbf{0} = \mathbf{0}$ ✓
- Closed under $+$: $A\mathbf{x} + A\mathbf{y} = A(\mathbf{x} + \mathbf{y}) \in \text{col}(A)$ ✓
- Closed under $\cdot$: $\alpha(A\mathbf{x}) = A(\alpha\mathbf{x}) \in \text{col}(A)$ ✓

The column space is always a subspace of $\mathbb{R}^m$. Its dimension equals $\text{rank}(A)$.

---

**Symmetric matrices**

$W = \{A \in \mathbb{R}^{n \times n} : A = A^\top\} \subseteq \mathbb{R}^{n \times n}$.

- Contains $\mathbf{0}$: $\mathbf{0}^\top = \mathbf{0}$ ✓
- Closed under $+$: $(A + B)^\top = A^\top + B^\top = A + B$ ✓
- Closed under $\cdot$: $(\alpha A)^\top = \alpha A^\top = \alpha A$ ✓

Dimension: $n(n+1)/2$. A basis consists of all matrices $E_{ii}$ (diagonal) and $E_{ij} + E_{ji}$ (off-diagonal symmetric pairs) for $i < j$. Symmetric matrices appear constantly in AI: covariance matrices, Gram matrices, Hessians, and attention score matrices (in some formulations) are all symmetric.

---

**Traceless matrices**

$W = \{A \in \mathbb{R}^{n \times n} : \text{tr}(A) = 0\} \subseteq \mathbb{R}^{n \times n}$.

- Contains $\mathbf{0}$: $\text{tr}(\mathbf{0}) = 0$ ✓
- Closed under $+$: $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B) = 0$ ✓
- Closed under $\cdot$: $\text{tr}(\alpha A) = \alpha \, \text{tr}(A) = 0$ ✓

Dimension: $n^2 - 1$ (one linear constraint on $n^2$ free entries). Relevant to normalisation in transformers: the centering step of LayerNorm projects activations onto the traceless (mean-zero) subspace.

---

**Upper triangular matrices**

$W = \{A \in \mathbb{R}^{n \times n} : A_{ij} = 0 \text{ for all } i > j\}$.

The defining condition $A_{ij} = 0$ for $i > j$ is a collection of linear constraints; closure under $+$ and $\cdot$ follows because the zero entries stay zero under linear combinations. Dimension: $n(n+1)/2$. The space of upper triangular matrices is a subspace of $\mathbb{R}^{n \times n}$.

---

**Polynomials vanishing at a point**

$W = \{p \in \mathcal{P}_n : p(1) = 0\}$ — polynomials of degree $\leq n$ with $p(1) = 0$.

- Contains $\mathbf{0}$: zero polynomial satisfies $0(1) = 0$ ✓
- Closed: if $p(1) = 0$ and $q(1) = 0$, then $(\alpha p + \beta q)(1) = \alpha p(1) + \beta q(1) = 0$ ✓

Dimension: $n$ (one linear constraint on $n+1$ coefficients). A basis: $\{t - 1,\ t^2 - 1,\ t^3 - 1,\ \ldots,\ t^n - 1\}$.

### 3.4 Non-Subspace Examples

For each of the following, we identify the specific condition that fails:

**Affine subspace $\{\mathbf{x} : A\mathbf{x} = \mathbf{b}\}$ with $\mathbf{b} \neq \mathbf{0}$.**

Condition 1 fails: $A \cdot \mathbf{0} = \mathbf{0} \neq \mathbf{b}$, so $\mathbf{0} \notin W$. Additionally, if $\mathbf{x}, \mathbf{y}$ are both solutions, $A(\mathbf{x} + \mathbf{y}) = 2\mathbf{b} \neq \mathbf{b}$, so condition 2 also fails. This set is a coset of $\text{null}(A)$: the general solution is $\mathbf{x}_p + \text{null}(A)$ for any particular solution $\mathbf{x}_p$.

**Unit ball $\{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| \leq 1\}$.**

Condition 1 holds: $\mathbf{0} \in B$. Condition 3 fails: take $\mathbf{x}$ with $\|\mathbf{x}\| = 1$ and $\alpha = 2$: then $\|\alpha\mathbf{x}\| = 2 > 1$, so $\alpha\mathbf{x} \notin B$. The unit ball is bounded; a subspace is never bounded (it contains all scalar multiples of its vectors, so it extends to infinity in every direction it contains).

**First quadrant $\{(x, y) \in \mathbb{R}^2 : x \geq 0,\ y \geq 0\}$.**

Conditions 1 and 2 hold: $\mathbf{0} \in W$; sum of non-negative vectors is non-negative. Condition 3 fails: $(-1) \cdot (1, 1) = (-1, -1) \notin W$. A subspace must be closed under all scalars, including negative ones — which forces it to be symmetric about the origin.

**Nonzero vectors $\{\mathbf{v} \in V : \mathbf{v} \neq \mathbf{0}\}$.**

Condition 1 fails immediately: $\mathbf{0}$ is excluded by definition. Since the zero vector is always required, this cannot be a subspace.

**The probability simplex $\Delta^{|V|-1}$ (softmax outputs).**

All three conditions fail: no zero vector; not closed under $+$; not closed under scaling. The AI implication: you cannot average, interpolate, or take linear combinations of probability distributions and expect to get another valid probability distribution. Arithmetic must happen in logit space (pre-softmax), where the vector space structure is intact.

```text
QUICK NON-SUBSPACE CHECK
════════════════════════════════════════════════════════════════════

  Given a candidate set W, the fastest refutation is usually:

  1. Check 0 ∈ W — if the zero vector fails the defining property,
     stop immediately. W is not a subspace.

  2. Check negative scalar — find v ∈ W and test −v ∈ W.
     This catches all sets that require positivity or a fixed norm.

  3. Check sum of two "extreme" elements — if W has a boundary
     condition, vectors near the boundary often sum outside W.
```

### 3.5 The Two Trivial Subspaces

Every vector space $V$ has exactly two subspaces that are always present, regardless of what $V$ is:

1. **$\{\mathbf{0}\}$** — the trivial subspace, containing only the zero vector. Dimension 0.
2. **$V$ itself** — the whole space. Dimension $= \dim(V)$.

Any subspace $W \subseteq V$ satisfies $\{\mathbf{0}\} \subseteq W \subseteq V$.

A **proper subspace** is any subspace $W \subsetneq V$ (not the whole space). A **non-trivial subspace** is any subspace $W \neq \{\mathbf{0}\}$ (not just the zero vector). The interesting subspaces are the non-trivial proper ones — the lines, planes, hyperplanes, and higher-dimensional flats that live inside $V$ without being all of $V$.

For AI, interpreting these extremes:

- A layer that writes only the zero vector to the residual stream corresponds to the trivial subspace $\{\mathbf{0}\}$ — it contributes nothing.
- A layer with full rank that can write anything to $\mathbb{R}^d$ corresponds to the full space $V$ — it has maximum expressive power.
- Most layers operate in a proper non-trivial subspace: they can produce outputs in a $r$-dimensional subspace of $\mathbb{R}^d$ where $0 < r < d$.

---

## 4. Span and Linear Combinations

### 4.1 Linear Combinations

A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k \in V$ is any vector of the form:

$$\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \sum_{i=1}^k \alpha_i \mathbf{v}_i$$

where $\alpha_1, \alpha_2, \ldots, \alpha_k \in \mathbb{R}$ are arbitrary scalars called the **coefficients** of the linear combination.

A linear combination uses only the two vector space operations: scalar multiplication (to produce each $\alpha_i \mathbf{v}_i$) and addition (to sum them). By the closure axioms, any linear combination of vectors in $V$ produces another vector in $V$. By the closure conditions for subspaces, any linear combination of vectors in a subspace $W$ produces another vector in $W$.

**Special cases of linear combinations:**

| Coefficients                                | Name               | Meaning                                     |
| ------------------------------------------- | ------------------ | ------------------------------------------- |
| Any $\alpha_i \in \mathbb{R}$               | Linear combination | General; unrestricted                       |
| $\sum_i \alpha_i = 1$                       | Affine combination | Preserves "position" (stays in affine hull) |
| $\alpha_i \geq 0$ and $\sum_i \alpha_i = 1$ | Convex combination | Stays in convex hull (interpolation)        |
| $\alpha_i \in \{0, 1\}$                     | Subset sum         | Used in combinatorics over $\mathbb{F}_2$   |

For AI, the most important special case is the unconstrained linear combination — it defines the span of a set of vectors, which in turn defines the reachable outputs of a linear layer.

### 4.2 The Span

The **span** of a set $S = \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\} \subseteq V$ is the set of all linear combinations of vectors in $S$:

$$\text{span}(S) = \left\{ \sum_{i=1}^k \alpha_i \mathbf{v}_i \;:\; \alpha_1, \ldots, \alpha_k \in \mathbb{R} \right\}$$

**Theorem:** $\text{span}(S)$ is always a subspace of $V$.

_Proof:_

1. Contains $\mathbf{0}$: set all $\alpha_i = 0$ → $\sum \alpha_i \mathbf{v}_i = \mathbf{0} \in \text{span}(S)$ ✓
2. Closed under addition: if $\mathbf{u} = \sum \alpha_i \mathbf{v}_i$ and $\mathbf{w} = \sum \beta_i \mathbf{v}_i$, then $\mathbf{u} + \mathbf{w} = \sum (\alpha_i + \beta_i) \mathbf{v}_i \in \text{span}(S)$ ✓
3. Closed under scalar multiplication: if $\mathbf{u} = \sum \alpha_i \mathbf{v}_i$ then $\gamma \mathbf{u} = \sum (\gamma\alpha_i) \mathbf{v}_i \in \text{span}(S)$ ✓

**Minimality:** $\text{span}(S)$ is the _smallest_ subspace containing all vectors in $S$. Any subspace $W$ that contains all vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ must also contain all their linear combinations (by closure), and hence must contain $\text{span}(S)$.

**Conventions:**

- $\text{span}(\emptyset) = \{\mathbf{0}\}$ — the span of the empty set is the trivial subspace. (An empty sum equals zero.)
- The span grows as $S$ grows: if $S \subseteq T$ then $\text{span}(S) \subseteq \text{span}(T)$.
- Adding a vector already in $\text{span}(S)$ does not increase the span.

### 4.3 Geometric Pictures of Span

Geometric intuition for span is crucial and carries directly into AI:

```text
GEOMETRIC MEANING OF SPAN
════════════════════════════════════════════════════════════════════

  span{v}  (one non-zero vector in ℝⁿ):
  ─────────────────────────────────────
  A line through the origin in the direction of v.
  All scalar multiples of v: {αv : α ∈ ℝ}.
  Always a 1-dimensional subspace.

  span{u, v}  (two non-parallel vectors in ℝⁿ):
  ───────────────────────────────────────────────
  A plane through the origin containing both directions.
  All combinations αu + βv: a 2-dimensional subspace.
  (If u and v ARE parallel: same as span{u} — 1-dimensional.)

  span{u, v, w}  (in ℝ³):
  ─────────────────────────
  If u, v, w are coplanar → a plane (2D subspace).
  If not coplanar → all of ℝ³ (3D subspace = whole space).

  span{v₁, …, vₖ}  (in ℝⁿ):
  ────────────────────────────
  Dimension = rank of the matrix [v₁ | v₂ | … | vₖ].
  The "flat hull" of the vectors, passing through the origin.
  It is the column space of the matrix with columns v₁, …, vₖ.
```

**The key dimension count.** Given $k$ vectors in $\mathbb{R}^n$:

- If $k < n$: span has dimension at most $k$ (a proper subspace)
- If $k = n$: span has dimension $n$ iff the vectors are independent (iff the matrix is invertible)
- If $k > n$: span has dimension at most $n$; the vectors must be linearly dependent

### 4.4 Spanning Sets

A set $S$ **spans** $V$ (or is a **spanning set** for $V$) if $\text{span}(S) = V$ — equivalently, if every vector in $V$ can be expressed as a linear combination of vectors in $S$.

**Standard examples:**

- $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$ spans $\mathbb{R}^n$ (standard basis, by definition of coordinates).
- $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_1 + \mathbf{e}_2\}$ spans $\mathbb{R}^2$ — but is redundant; $\mathbf{e}_1 + \mathbf{e}_2 \in \text{span}\{\mathbf{e}_1, \mathbf{e}_2\}$ already.
- $\{(1,1), (1,-1)\}$ spans $\mathbb{R}^2$: every $(x,y)$ can be written as $\frac{x+y}{2}(1,1) + \frac{x-y}{2}(1,-1)$.
- $\{1, t, t^2, \ldots, t^n\}$ spans $\mathcal{P}_n$ (every polynomial is a linear combination of monomials).
- The columns of a matrix $A \in \mathbb{R}^{m \times n}$ span $\text{col}(A) \subseteq \mathbb{R}^m$; they span all of $\mathbb{R}^m$ iff $A$ has full row rank ($\text{rank}(A) = m$).

**AI application.** The columns of a weight matrix $W \in \mathbb{R}^{m \times n}$ span $\text{col}(W) \subseteq \mathbb{R}^m$. If $\text{rank}(W) = r < m$, then only an $r$-dimensional subspace of $\mathbb{R}^m$ is reachable: regardless of the input, the layer's output can never leave the $r$-dimensional column space. Outputs in the orthogonal complement $\text{null}(W^\top) = \text{col}(W)^\perp$ are permanently inaccessible from this layer. They must come from other components (residual connections, other layers).

A spanning set need not be a basis — it may contain redundant vectors. The minimal spanning sets are the bases (Section 6).

### 4.5 Span in Terms of Matrices

Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k \in \mathbb{R}^m$, form the matrix:

$$V = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k] \in \mathbb{R}^{m \times k}$$

Then:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{V\boldsymbol{\alpha} : \boldsymbol{\alpha} \in \mathbb{R}^k\} = \text{col}(V)$$

This connects span to:

- **Column space:** $\text{span}$ of the vectors = column space of the matrix they form
- **Rank:** $\dim(\text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}) = \text{rank}(V)$
- **Linear system solvability:** $\mathbf{b} \in \text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ if and only if the system $V\boldsymbol{\alpha} = \mathbf{b}$ is consistent, which by the Rouché–Capelli theorem holds if and only if $\text{rank}(V) = \text{rank}([V \mid \mathbf{b}])$.

**Algorithm for membership testing.** To check whether $\mathbf{b} \in \text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$:

1. Form the augmented matrix $[V \mid \mathbf{b}]$
2. Row reduce to RREF
3. If no row of the form $[0\ 0\ \cdots\ 0 \mid *]$ with $* \neq 0$ appears: $\mathbf{b} \in \text{span}$ ✓
4. If such a row appears: $\mathbf{b} \notin \text{span}$ ✗

If $\mathbf{b} \in \text{span}$, the RREF also yields the coordinates $\boldsymbol{\alpha}$ expressing $\mathbf{b}$ as a linear combination.

---

## 5. Linear Independence

### 5.1 Definition of Linear Independence

Vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are **linearly independent** if the only linear combination that equals the zero vector is the trivial one — all coefficients zero:

$$\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0} \implies \alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$$

Equivalently: no vector in the set is a linear combination of the others; each vector contributes a genuinely new direction that cannot be reached from the others.

The vectors are **linearly dependent** if the negation holds: there exist coefficients $\alpha_1, \ldots, \alpha_k$, not all zero, such that $\sum_i \alpha_i \mathbf{v}_i = \mathbf{0}$. In this case, at least one vector is redundant — it can be expressed as a linear combination of the others.

The contrast is worth making explicit:

```text
INDEPENDENT vs DEPENDENT
════════════════════════════════════════════════════════════════════

  Independent: each vector adds a new direction.
               The set is "lean" — nothing is redundant.
               Remove any one vector: the span strictly shrinks.
               dim(span) = k  (maximum possible given k vectors)

  Dependent:   at least one vector is a combination of the others.
               The set contains redundancy.
               Remove the dependent vector: span unchanged.
               dim(span) < k  (less than the number of vectors)
```

**Why it matters.** Linear independence is what makes a basis a basis. Without it, some vectors in a spanning set are superfluous — they do not add to the span. A basis is the most efficient spanning set: independent and spanning.

### 5.2 The Dependence Relation

If $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are linearly dependent, there exist coefficients $\alpha_1, \ldots, \alpha_k$ (not all zero) with $\sum_{i=1}^k \alpha_i \mathbf{v}_i = \mathbf{0}$. Let $j$ be any index with $\alpha_j \neq 0$. Then:

$$\mathbf{v}_j = -\frac{1}{\alpha_j} \sum_{i \neq j} \alpha_i \mathbf{v}_i = \sum_{i \neq j} \left(-\frac{\alpha_i}{\alpha_j}\right) \mathbf{v}_i$$

So $\mathbf{v}_j$ is a linear combination of the other vectors. Consequently, removing $\mathbf{v}_j$ from the set does not reduce the span:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \text{span}\{\mathbf{v}_1, \ldots, \hat{\mathbf{v}}_j, \ldots, \mathbf{v}_k\}$$

where $\hat{\mathbf{v}}_j$ means $\mathbf{v}_j$ is omitted. This is the operational meaning of dependence: you can identify and remove dependent vectors without losing span. Row reduction does exactly this.

### 5.3 Checking Linear Independence

**Matrix method (universal).**

Form the matrix $A = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_k] \in \mathbb{R}^{m \times k}$ and row reduce to RREF.

- **Linearly independent** iff every column of $A$ has a pivot iff $\text{rank}(A) = k$ iff $\text{null}(A) = \{\mathbf{0}\}$
- **Linearly dependent** iff some column has no pivot iff $\text{rank}(A) < k$ iff $\text{null}(A) \neq \{\mathbf{0}\}$

When dependent, the RREF reveals which vectors are dependent and expresses them in terms of the pivot columns.

**Determinant test** (square matrices only, $k = n$).

$\mathbf{v}_1, \ldots, \mathbf{v}_n \in \mathbb{R}^n$ are linearly independent iff $\det([\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n]) \neq 0$.

**SVD / singular value test** (numerical).

$\sigma_{\min}(A) > \varepsilon$ for some tolerance $\varepsilon$ iff the columns are linearly independent to numerical precision $\varepsilon$. The smallest singular value $\sigma_{\min}$ measures "how far from dependent" the columns are. When $\sigma_{\min} \approx 0$, the columns are nearly dependent (nearly lying in a lower-dimensional subspace).

**Gram matrix test.**

The Gram matrix $G = A^\top A \in \mathbb{R}^{k \times k}$ with $G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle$. The columns $\mathbf{v}_1,\ldots,\mathbf{v}_k$ are linearly independent iff $G$ is positive definite (all eigenvalues $> 0$) iff $\det(G) > 0$.

### 5.4 Key Properties

**Any set containing $\mathbf{0}$ is linearly dependent.**

$1 \cdot \mathbf{0} + 0 \cdot \mathbf{v}_2 + \cdots = \mathbf{0}$ — a non-trivial combination. So $\mathbf{0}$ is always dependent on any other set of vectors. This is consistent with the subspace requirement: $\mathbf{0}$ carries no directional information.

**A single non-zero vector is always linearly independent.**

$\alpha \mathbf{v} = \mathbf{0}$ with $\mathbf{v} \neq \mathbf{0}$ forces $\alpha = 0$ (by the cancellation law proved in §2.2). So $\{\mathbf{v}\}$ is independent whenever $\mathbf{v} \neq \mathbf{0}$.

**Two vectors are linearly dependent iff one is a scalar multiple of the other.**

$\alpha \mathbf{u} + \beta \mathbf{v} = \mathbf{0}$ with $(\alpha, \beta) \neq (0,0)$. If $\beta \neq 0$: $\mathbf{v} = -(\alpha/\beta)\mathbf{u}$. If $\beta = 0$: $\alpha \neq 0$ so $\mathbf{u} = \mathbf{0}$. So dependence of two vectors = one is a scalar multiple of the other = they point in the same (or opposite) direction.

**More vectors than dimensions forces dependence.**

If $k > n$ for vectors in $\mathbb{R}^n$, they are always linearly dependent. The matrix $A = [\mathbf{v}_1|\cdots|\mathbf{v}_k] \in \mathbb{R}^{n \times k}$ has rank at most $n < k$; by rank-nullity, $\text{null}(A)$ is non-trivial, giving a non-trivial dependence relation.

This has a profound implication: _in $\mathbb{R}^d$, you cannot have more than $d$ linearly independent vectors_. In a $d$-dimensional embedding space, there are at most $d$ independent directions. Any collection of more than $d$ feature vectors must be linearly dependent.

**Subsets of independent sets are independent.**

If $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are linearly independent, so is every subset. (Removing vectors from an independent set cannot introduce a new dependence — the remaining vectors were already independent when part of the larger set.)

**Supersets of dependent sets are dependent.**

If $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are linearly dependent, so is every superset. (The same non-trivial dependence relation, with zero coefficients for the added vectors, still witnesses dependence.)

### 5.5 Linear Independence in AI Contexts

**Attention heads.** If the $H$ attention heads write to linearly independent subspaces of $\mathbb{R}^d$ (their OV circuits have mutually orthogonal column spaces), then each head provides genuinely independent information. Dependent heads write to overlapping subspaces; they interfere and are candidates for pruning. Head redundancy is exactly the linear dependence of the subspaces they write to.

**The superposition hypothesis.** Elhage et al. (2022) observe that large language models appear to represent more features than the embedding dimension $d$ allows. Since more than $d$ linearly independent directions cannot exist in $\mathbb{R}^d$, the features must be stored as _linearly dependent_ (non-orthogonal) directions. This is the superposition hypothesis: features are packed into the embedding space at densities exceeding the dimension, using non-orthogonal directions. Each feature "leaks" into the directions of nearby features — a phenomenon called polysemanticity. Linear independence is the threshold: below $d$ features, they can be orthogonal and non-interfering; beyond $d$, interference is inevitable.

**LoRA rank selection.** In LoRA, the update is $\Delta W = BA^\top$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{n \times r}$. The rank of $\Delta W$ equals the number of linearly independent columns in $B$ (equivalently, in $A$). If the columns of $B$ are linearly dependent, the effective rank is less than $r$; some of the $r$ rank budget is wasted. Choosing $r$ large helps only if the corresponding columns of $B$ actually end up spanning an $r$-dimensional space after training.

**Gradient directions across batches.** If gradients from different batches are linearly independent, each batch provides new information about the loss landscape. If gradients are nearly parallel (nearly dependent), the batches are redundant — they are teaching the model the same thing. Gradient diversity, measured by the rank or effective dimension of the gradient matrix $[\nabla\mathcal{L}_1 \mid \cdots \mid \nabla\mathcal{L}_B]$, is a measure of how much information each batch contributes.

**Feature directions in representation learning.** Each probing direction in an embedding space is a vector in $\mathbb{R}^d$. For $k$ concepts to be probed without interference, their associated directions should be linearly independent (ideally orthogonal). The maximum number of perfectly non-interfering linear features is $d$. Beyond that, interference is structural — a consequence of linear independence breaking down.

---

## 6. Basis and Dimension

### 6.1 Definition of a Basis

A **basis** for a vector space $V$ is a set $\mathcal{B} = \{\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n\}$ satisfying two conditions simultaneously:

1. **Linearly independent:** the vectors $\mathbf{b}_i$ are linearly independent
2. **Spans $V$:** every vector in $V$ is a linear combination of the $\mathbf{b}_i$

A basis is the most efficient possible spanning set: it spans $V$ with no redundancy. Equivalently:

- A basis is a **minimal spanning set**: remove any single vector and the set no longer spans $V$
- A basis is a **maximal independent set**: add any vector from $V$ and the set becomes linearly dependent

These three characterisations — (independent + spanning), (minimal spanning), (maximal independent) — are all equivalent. Each one defines the same objects.

```text
THE ROLE OF A BASIS
════════════════════════════════════════════════════════════════════

  A basis gives V a coordinate system.

  Once you fix a basis {b₁, …, bₙ} for V:
  • Every vector v ∈ V corresponds to a unique n-tuple (α₁,…,αₙ)
  • The n-tuple is the vector's coordinates in this basis
  • V becomes "identified" with ℝⁿ via this correspondence

  Different bases = different coordinate systems for the same space.
  The space itself is unchanged; only the coordinatisation differs.

  In AI: the "basis" of the residual stream is not canonical.
  Attention heads and MLP layers operate in their own "preferred"
  bases, related to the standard basis by learned rotations.
```

### 6.2 The Unique Representation Theorem

**Theorem.** If $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ is a basis for $V$, then every vector $\mathbf{v} \in V$ has a **unique** representation:

$$\mathbf{v} = \alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2 + \cdots + \alpha_n \mathbf{b}_n$$

The scalars $(\alpha_1, \ldots, \alpha_n)$ are called the **coordinates** (or **components**) of $\mathbf{v}$ with respect to basis $\mathcal{B}$, written $[\mathbf{v}]_{\mathcal{B}} = (\alpha_1, \ldots, \alpha_n)$.

**Proof of uniqueness.** Suppose $\mathbf{v} = \sum_i \alpha_i \mathbf{b}_i$ and $\mathbf{v} = \sum_i \beta_i \mathbf{b}_i$. Subtract:

$$\mathbf{0} = \mathbf{v} - \mathbf{v} = \sum_{i=1}^n (\alpha_i - \beta_i) \mathbf{b}_i$$

By linear independence of the basis vectors, all coefficients are zero: $\alpha_i - \beta_i = 0$, so $\alpha_i = \beta_i$ for all $i$. ∎

**Why uniqueness matters.** Without uniqueness, coordinates would be ambiguous — the same vector could have multiple different "addresses" in the same coordinate system. The combination of independence (ensures uniqueness) and spanning (ensures existence) is precisely what makes a basis the right tool for establishing a coordinate system.

**The isomorphism.** The unique representation theorem says that every $n$-dimensional vector space $V$ is **isomorphic** to $\mathbb{R}^n$: the map $\mathbf{v} \mapsto [\mathbf{v}]_{\mathcal{B}} \in \mathbb{R}^n$ is a bijection that preserves both operations (addition and scalar multiplication). This is why working in coordinates is always valid: the coordinate map translates the abstract vector space structure faithfully into $\mathbb{R}^n$.

### 6.3 Dimension

**Theorem.** All bases for a given vector space $V$ have the same number of vectors. This number is the **dimension** of $V$:

$$\dim(V) = \text{number of vectors in any basis of } V$$

**Proof sketch.** Suppose $\mathcal{B}_1 = \{\mathbf{b}_1, \ldots, \mathbf{b}_m\}$ and $\mathcal{B}_2 = \{\mathbf{c}_1, \ldots, \mathbf{c}_n\}$ are both bases for $V$. Since $\mathcal{B}_1$ spans $V$ and $\mathcal{B}_2$ is independent, by the Steinitz Exchange Lemma (or the Replacement Theorem), we must have $n \leq m$. By symmetry, $m \leq n$. Hence $m = n$. ∎

Dimension is the fundamental measure of the "size" of a vector space — the number of independent directions it contains. It is an intrinsic property of the space, independent of any particular basis.

**Dimensions of standard spaces:**

| Space                                   | Dimension  | Basis                                               |
| --------------------------------------- | ---------- | --------------------------------------------------- |
| $\mathbb{R}^n$                          | $n$        | $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ (standard) |
| $\mathbb{R}^{m \times n}$               | $mn$       | $\{E_{ij} : 1 \leq i \leq m,\ 1 \leq j \leq n\}$    |
| $\mathcal{P}_n$                         | $n+1$      | $\{1, t, t^2, \ldots, t^n\}$                        |
| $\text{Sym}_n$ (symmetric $n \times n$) | $n(n+1)/2$ | $\{E_{ii}\} \cup \{E_{ij}+E_{ji} : i < j\}$         |
| $\text{Skew}_n$ (skew-symmetric)        | $n(n-1)/2$ | $\{E_{ij} - E_{ji} : i < j\}$                       |
| $\text{Diag}_n$ (diagonal)              | $n$        | $\{E_{ii}\}$                                        |
| $\{\mathbf{0}\}$                        | $0$        | $\emptyset$                                         |
| $C([a,b])$, $L^2([a,b])$, $\mathcal{P}$ | $\infty$   | Countably infinite bases exist                      |

**Dimension inequalities.** For subspace $W \subseteq V$:

- $\dim(W) \leq \dim(V)$
- $\dim(W) = \dim(V)$ and $W \subseteq V$ together imply $W = V$ (in finite dimensions)

This last point is crucial: if you find a subspace with the same dimension as the whole space, it must equal the whole space. In particular, a subspace of $\mathbb{R}^n$ with dimension $n$ is all of $\mathbb{R}^n$.

### 6.4 Standard Bases

**Standard basis for $\mathbb{R}^n$:**

$$\mathbf{e}_i = (0, \ldots, 0, \underbrace{1}_{i\text{-th}}, 0, \ldots, 0)$$

The coordinates of any vector $\mathbf{v} = (v_1, \ldots, v_n)$ in the standard basis are simply its components: $[\mathbf{v}]_{\text{std}} = (v_1, \ldots, v_n)$. This is why the standard basis is the "natural" choice for $\mathbb{R}^n$ — coordinates and components coincide.

**Standard basis for $\mathbb{R}^{m \times n}$:**

$E_{ij}$ is the matrix with a 1 in position $(i,j)$ and 0 elsewhere. There are $mn$ such matrices, and they form a basis for $\mathbb{R}^{m \times n}$. The coordinate of a matrix $A$ in this basis is simply $A_{ij}$.

**Standard basis for $\mathcal{P}_n$:**

$\{1, t, t^2, \ldots, t^n\}$ — the monomials. The coordinates of $p(t) = a_0 + a_1 t + \cdots + a_n t^n$ are its coefficients $(a_0, a_1, \ldots, a_n)$.

**Fourier basis for $L^2([0, 2\pi])$:**

$$\left\{\frac{1}{\sqrt{2\pi}},\ \frac{\cos(t)}{\sqrt{\pi}},\ \frac{\sin(t)}{\sqrt{\pi}},\ \frac{\cos(2t)}{\sqrt{\pi}},\ \frac{\sin(2t)}{\sqrt{\pi}},\ \ldots\right\}$$

An orthonormal basis. The coordinates of $f$ in this basis are the Fourier coefficients. This is the infinite-dimensional analogue of an orthonormal basis in $\mathbb{R}^n$.

**Eigenbasis.** For a linear map $T: V \to V$ with $n$ linearly independent eigenvectors $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$, this set forms a basis called the **eigenbasis** of $T$. In the eigenbasis, $T$ acts diagonally: $T\mathbf{q}_i = \lambda_i \mathbf{q}_i$. This diagonalises $T$, making its action transparent. PCA uses the eigenbasis of the covariance matrix; the action of the covariance matrix in its eigenbasis is pure scaling by eigenvalues (variances).

### 6.5 Change of Basis

The same vector can be described by different coordinate tuples in different bases. Understanding how coordinates transform under a change of basis is essential for working in multiple representations simultaneously — which is exactly what happens in a transformer, where the residual stream has a "standard" basis but each component may operate in a different preferred basis.

**Setup.** Let $\mathcal{B} = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ and $\mathcal{C} = \{\mathbf{c}_1, \ldots, \mathbf{c}_n\}$ be two bases for $V$. The **change-of-basis matrix** $P_{\mathcal{B} \to \mathcal{C}}$ is the $n \times n$ matrix whose $j$-th column is the coordinate representation of $\mathbf{b}_j$ in basis $\mathcal{C}$:

$$P_{\mathcal{B} \to \mathcal{C}} = \big[[\mathbf{b}_1]_{\mathcal{C}} \mid [\mathbf{b}_2]_{\mathcal{C}} \mid \cdots \mid [\mathbf{b}_n]_{\mathcal{C}}\big]$$

**Coordinate transformation.** For any $\mathbf{v} \in V$:

$$[\mathbf{v}]_{\mathcal{C}} = P_{\mathcal{B} \to \mathcal{C}}\, [\mathbf{v}]_{\mathcal{B}}$$

**Properties:**

- $P_{\mathcal{B} \to \mathcal{C}}$ is always invertible (since both $\mathcal{B}$ and $\mathcal{C}$ are bases)
- $P_{\mathcal{C} \to \mathcal{B}} = P_{\mathcal{B} \to \mathcal{C}}^{-1}$ — the inverse change-of-basis goes the other way
- Composition: $P_{\mathcal{A} \to \mathcal{C}} = P_{\mathcal{B} \to \mathcal{C}}\, P_{\mathcal{A} \to \mathcal{B}}$

**Linear maps under change of basis.** If a linear map $T: V \to V$ has matrix $M$ in basis $\mathcal{B}$, its matrix in basis $\mathcal{C}$ is:

$$[T]_{\mathcal{C}} = P_{\mathcal{B} \to \mathcal{C}}\, M\, P_{\mathcal{B} \to \mathcal{C}}^{-1}$$

This is a **similarity transformation**: $M$ and $P M P^{-1}$ represent the same linear map in different bases. They have the same eigenvalues, rank, trace, and determinant — these are basis-independent properties of the map.

**AI application: the residual stream basis.** In a transformer, the residual stream has dimension $d$ and technically lives in $\mathbb{R}^d$ with the standard basis. But each attention head's OV circuit is parameterised by $W_V \in \mathbb{R}^{d \times d_k}$ and $W_O \in \mathbb{R}^{d_k \times d}$: the head reads in a $d_k$-dimensional subspace and writes in another. The "natural basis" for head $h$ is different from the standard basis of $\mathbb{R}^d$. Change-of-basis matrices connect these representations. When we say a head "rotates" the residual stream, we mean it applies a change-of-basis transformation to the component it reads.

### 6.6 Constructing a Basis

**From a spanning set (row reduction).**

Given a spanning set $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ for some subspace $W$:

1. Form the matrix $A = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k]$
2. Row reduce $A$ to RREF
3. The **pivot columns** of $A$ (not the RREF, but the original $A$) form a basis for $\text{col}(A) = W$

The non-pivot columns correspond to dependent vectors that can be removed without reducing the span.

**From a null space (free variables).**

To find a basis for $\text{null}(A)$:

1. Row reduce $A$ to RREF; identify free variables $x_{j_1}, \ldots, x_{j_k}$
2. For each free variable $x_{j_\ell}$, set $x_{j_\ell} = 1$ and all other free variables $= 0$; solve for pivot variables
3. The resulting vector is one basis vector for $\text{null}(A)$
4. Repeat for each free variable; the $k$ vectors form a basis for $\text{null}(A)$

**Gram-Schmidt (from independent vectors to orthonormal basis).**

Given $k$ linearly independent vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$, produce an orthonormal basis $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ for $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$. See Section 9.3 for the full procedure.

**Extending a basis.**

Given a basis $\mathcal{B}_W = \{\mathbf{b}_1, \ldots, \mathbf{b}_k\}$ for a subspace $W \subsetneq V$, extend to a basis for all of $V$:

1. Start with $S = \mathcal{B}_W$
2. Find any $\mathbf{v} \in V \setminus \text{span}(S)$ and add it: $S \leftarrow S \cup \{\mathbf{v}\}$
3. Repeat until $\text{span}(S) = V$

This process terminates in at most $\dim(V) - \dim(W)$ steps.

### 6.7 Dimension Counting Theorems

These theorems relate the dimensions of related subspaces. They are among the most useful tools in linear algebra.

---

**Rank-Nullity Theorem.** For any matrix $A \in \mathbb{R}^{m \times n}$:

$$\underbrace{\dim(\text{null}(A))}_{\text{nullity}(A)} + \underbrace{\dim(\text{col}(A))}_{\text{rank}(A)} = n$$

In words: the number of "invisible" input directions plus the number of "active" input directions equals the total number of input dimensions.

_Proof idea._ Row reduce $A$ to RREF. Let $r = \text{rank}(A)$. There are $r$ pivot columns (corresponding to $r$ independent output directions → $\dim(\text{col}(A)) = r$) and $n - r$ free columns (corresponding to $n - r$ null space basis vectors → $\dim(\text{null}(A)) = n - r$). Total: $r + (n - r) = n$. ∎

_AI application._ For a weight matrix $W \in \mathbb{R}^{m \times n}$ with rank $r$:

- $r$ input dimensions "pass through" to the output (row space directions)
- $n - r$ input dimensions are silenced (null space directions)
- The layer uses $r$ out of $n$ available input dimensions

---

**Dimension of a Sum.** For subspaces $W_1, W_2 \subseteq V$:

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$$

This is the **Grassmann formula** or **inclusion-exclusion for dimensions**. It is the direct analogue of $|A \cup B| = |A| + |B| - |A \cap B|$ for finite sets, but for dimensions of subspaces.

_Key consequence._ If $W_1 \cap W_2 = \{\mathbf{0}\}$ (intersection is trivial):

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2)$$

and the sum is a **direct sum**: $W_1 + W_2 = W_1 \oplus W_2$. Dimensions add when the subspaces are independent (share only the origin).

---

**Dimension bounds for intersection.** Given $\dim(W_1) = k_1$, $\dim(W_2) = k_2$, $\dim(V) = n$:

$$\max(0,\ k_1 + k_2 - n) \leq \dim(W_1 \cap W_2) \leq \min(k_1, k_2)$$

The upper bound: the intersection is contained in both $W_1$ and $W_2$, so its dimension cannot exceed either. The lower bound: from the Grassmann formula and $\dim(W_1 + W_2) \leq n$.

_AI application._ Two attention heads with $d_k$-dimensional subspaces in $\mathbb{R}^d$: their subspace intersection has dimension at least $\max(0, 2d_k - d)$. If $d_k > d/2$, the heads must share at least $2d_k - d$ dimensions; their communication channels overlap inevitably.

---

**Complement dimension.** For subspace $W \subseteq V$ with $\dim(V) = n$:

$$\dim(W^\perp) = n - \dim(W)$$

So $\dim(W) + \dim(W^\perp) = n$. The orthogonal complement "fills in" the remaining dimensions. The fundamental subspaces obey:

$$\text{rank}(A) + \text{nullity}(A) = n \quad \Leftrightarrow \quad \dim(\text{row}(A)) + \dim(\text{null}(A)) = n$$

which is just the Rank-Nullity Theorem re-expressed using the fact that $\text{null}(A) = \text{row}(A)^\perp$.

---

## 7. The Four Fundamental Subspaces

> **Recall:** Earlier sections used these subspaces informally:
>
> - [Systems of Equations §4.3](../03-Systems-of-Equations/notes.md#43-the-fundamental-theorem-of-linear-algebra-applied-to-systems) — used col($A$) and null($A^\top$) to characterise when $Ax = b$ is solvable
> - [Matrix Rank §4](../05-Matrix-Rank/notes.md) — used rank and null space dimension to state the rank-nullity theorem
>
> This section is the canonical home: rigorous definitions, dimensional identities, bases computed from RREF, the orthogonality relationships, and their geometric interpretation as the complete decomposition of $\mathbb{R}^m$ and $\mathbb{R}^n$.

For any matrix $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$, Gilbert Strang identified four fundamental subspaces that together give a complete picture of how $A$ acts as a linear map $\mathbb{R}^n \to \mathbb{R}^m$. Understanding all four — their definitions, dimensions, bases, and mutual relationships — is the complete theory of linear systems.

### 7.1 Definition of All Four Subspaces

**1. Column space** $\text{col}(A) \subseteq \mathbb{R}^m$ (also called the image or range):

$$\text{col}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$$

- The set of all possible outputs of $A$; the "reachable" subspace in $\mathbb{R}^m$
- $\dim(\text{col}(A)) = r$
- Basis: pivot columns of $A$ (the original $A$, not its RREF)
- The system $A\mathbf{x} = \mathbf{b}$ is consistent if and only if $\mathbf{b} \in \text{col}(A)$

---

**2. Null space** $\text{null}(A) \subseteq \mathbb{R}^n$ (also called the kernel):

$$\text{null}(A) = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

- The set of all inputs mapped to zero; the "invisible" directions in $\mathbb{R}^n$
- $\dim(\text{null}(A)) = n - r$ (by Rank-Nullity)
- Basis: free variable solution vectors from the RREF of $A$
- $A$ cannot distinguish $\mathbf{x}$ from $\mathbf{x} + \mathbf{z}$ for any $\mathbf{z} \in \text{null}(A)$: both produce the same output

---

**3. Row space** $\text{row}(A) = \text{col}(A^\top) \subseteq \mathbb{R}^n$:

$$\text{row}(A) = \{A^\top \mathbf{y} : \mathbf{y} \in \mathbb{R}^m\} = \text{span of rows of } A$$

- The set of all linear combinations of the rows of $A$; the directions in $\mathbb{R}^n$ that $A$ "listens to"
- $\dim(\text{row}(A)) = r$ (same as column space — both equal the rank)
- Basis: non-zero rows of the RREF of $A$
- The projection of any input $\mathbf{x}$ onto $\text{row}(A)$ determines the output; the null space component of $\mathbf{x}$ is discarded

---

**4. Left null space** $\text{null}(A^\top) \subseteq \mathbb{R}^m$:

$$\text{null}(A^\top) = \{\mathbf{y} \in \mathbb{R}^m : A^\top \mathbf{y} = \mathbf{0}\}$$

- The set of all output-space vectors orthogonal to the column space of $A$
- $\dim(\text{null}(A^\top)) = m - r$ (by Rank-Nullity applied to $A^\top$)
- Basis: free variable solution vectors from the RREF of $A^\top$
- If $\mathbf{b} \in \text{null}(A^\top)$ then $\mathbf{b} \perp \text{col}(A)$, which means $A\mathbf{x} = \mathbf{b}$ has no solution

**Dimension summary:**

| Subspace              | Ambient space  | Dimension | Complement            |
| --------------------- | -------------- | --------- | --------------------- |
| $\text{row}(A)$       | $\mathbb{R}^n$ | $r$       | $\text{null}(A)$      |
| $\text{null}(A)$      | $\mathbb{R}^n$ | $n-r$     | $\text{row}(A)$       |
| $\text{col}(A)$       | $\mathbb{R}^m$ | $r$       | $\text{null}(A^\top)$ |
| $\text{null}(A^\top)$ | $\mathbb{R}^m$ | $m-r$     | $\text{col}(A)$       |

### 7.2 The Orthogonality Relations

The four subspaces pair into two orthogonal decompositions:

$$\mathbb{R}^n = \text{row}(A) \oplus \text{null}(A) \qquad \text{(orthogonal direct sum in } \mathbb{R}^n\text{)}$$

$$\mathbb{R}^m = \text{col}(A) \oplus \text{null}(A^\top) \qquad \text{(orthogonal direct sum in } \mathbb{R}^m\text{)}$$

**Proof that $\text{row}(A) \perp \text{null}(A)$.**

Take any $\mathbf{x} \in \text{null}(A)$ and any $\mathbf{y} \in \text{row}(A) = \text{col}(A^\top)$. Then $\mathbf{y} = A^\top \mathbf{v}$ for some $\mathbf{v} \in \mathbb{R}^m$. Compute:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \mathbf{x}^\top A^\top \mathbf{v} = (A\mathbf{x})^\top \mathbf{v} = \mathbf{0}^\top \mathbf{v} = 0$$

So every null space vector is orthogonal to every row space vector. ∎

Since $\dim(\text{row}(A)) + \dim(\text{null}(A)) = r + (n-r) = n = \dim(\mathbb{R}^n)$ and the two subspaces are orthogonal (hence their intersection is $\{\mathbf{0}\}$), they form an orthogonal direct sum decomposition of $\mathbb{R}^n$. Every vector $\mathbf{x} \in \mathbb{R}^n$ decomposes uniquely as:

$$\mathbf{x} = \underbrace{\mathbf{x}_r}_{\in \text{row}(A)} + \underbrace{\mathbf{x}_n}_{\in \text{null}(A)}$$

And $A\mathbf{x} = A\mathbf{x}_r + A\mathbf{x}_n = A\mathbf{x}_r + \mathbf{0} = A\mathbf{x}_r$. The null space component is silenced; only the row space component matters.

### 7.3 Strang's Big Picture

The complete action of $A$ as a linear map $\mathbb{R}^n \to \mathbb{R}^m$ is captured in the following diagram:

```text
STRANG'S FOUR FUNDAMENTAL SUBSPACES
════════════════════════════════════════════════════════════════════

            ℝⁿ (input space)                    ℝᵐ (output space)
  ┌──────────────────────────┐        ┌──────────────────────────┐
  │                          │        │                          │
  │   row(A)                 │        │   col(A)                 │
  │   dim = r                │──A────▶│   dim = r                │
  │                          │        │                          │
  │   "the listening space"  │        │   "the reachable space"  │
  │   A is sensitive here    │        │   A can write here       │
  │                          │        │                          │
  │         ⊕                │        │         ⊕                │
  │                          │        │                          │
  │   null(A)                │        │   null(Aᵀ)               │
  │   dim = n − r            │──A────▶│   dim = m − r            │
  │                          │  (→0)  │                          │
  │   "the silent space"     │        │   "the unreachable       │
  │   A maps this to 0       │        │    space"                │
  │                          │        │                          │
  └──────────────────────────┘        └──────────────────────────┘

  Key facts:
  • A is a bijection from row(A) to col(A) — restricted to row(A),
    A is invertible.
  • A maps all of null(A) to the single point {0} in ℝᵐ.
  • No input — however large — can produce an output in null(Aᵀ).
  • The orthogonal complements pair up:
    row(A)⊥ = null(A)  and  col(A)⊥ = null(Aᵀ)
```

**The action of $A$ completely described.** For any input $\mathbf{x} = \mathbf{x}_r + \mathbf{x}_n$ (row space + null space decomposition):

$$A\mathbf{x} = A\mathbf{x}_r \in \text{col}(A)$$

The null space component vanishes; the row space component is mapped bijectively into the column space. The left null space is entirely inaccessible from the input side.

### 7.4 Computing the Four Subspaces

The systematic procedure to find bases for all four fundamental subspaces from a matrix $A \in \mathbb{R}^{m \times n}$:

**Step 1: Row reduce $A$ to RREF.**

$$A \xrightarrow{\text{row ops}} R = \text{RREF}(A)$$

Identify the $r$ pivot positions (row $i$, column $j$ pairs where $R_{ij} = 1$ is the leading 1 of row $i$). Let $r = \text{rank}(A)$.

**Step 2: null(A).**

- Free variables: columns of $R$ without a pivot (say columns $j_1, \ldots, j_{n-r}$)
- For each free variable $x_{j_\ell}$: set $x_{j_\ell} = 1$, all other free variables $= 0$, solve for the pivot variables from $R\mathbf{x} = \mathbf{0}$
- The resulting vector $\mathbf{n}_\ell \in \mathbb{R}^n$ is the $\ell$-th null space basis vector
- Repeat for $\ell = 1, \ldots, n-r$; these $n-r$ vectors form a basis for $\text{null}(A)$

**Step 3: col(A).**

- The pivot columns of the **original** matrix $A$ (not $R$) form a basis for $\text{col}(A)$
- Specifically: if pivots appear in columns $p_1, p_2, \ldots, p_r$, then $\{\mathbf{a}_{p_1}, \mathbf{a}_{p_2}, \ldots, \mathbf{a}_{p_r}\}$ (columns of $A$) is a basis

**Why original $A$, not $R$?** Row operations change the column space (they change the linear dependencies between columns). The pivot positions identify which columns are independent, but the actual column vectors must be taken from the original $A$ to get a basis for the original column space.

**Step 4: row(A).**

- The **non-zero rows of $R$** (the RREF of $A$) form a basis for $\text{row}(A)$
- Unlike for the column space, row operations preserve the row space; the non-zero rows of $R$ are particular convenient linear combinations of the rows of $A$ that happen to be in reduced form

**Step 5: null(A^T).**

- Row reduce $A^\top$ to its RREF, then apply the null space procedure (Step 2) to $A^\top$
- Alternatively: if you have already computed the four dimensions (and bases for three of the four subspaces), use the complementary dimension argument
- Basis vectors of $\text{null}(A^\top)$ can also be read off from certain row operations applied to $[A \mid I_m]$

### 7.5 AI Interpretation of the Four Subspaces

For a weight matrix $W \in \mathbb{R}^{m \times n}$ in a neural network layer $\mathbf{y} = W\mathbf{x}$, the four subspaces tell the complete story of what this layer can and cannot do:

**Column space col(W) — the "reachable" subspace.**

Only outputs in $\text{col}(W) \subseteq \mathbb{R}^m$ can be produced by this layer. If $\text{rank}(W) = r < m$, then $m - r$ dimensions of the output space receive exactly zero contribution from this layer, regardless of input. In a transformer residual stream: each attention head and MLP block writes to a specific subspace of $\mathbb{R}^d$; the column space of the output projection $W_O$ is the subspace the head writes to. Dimensions orthogonal to $\text{col}(W_O)$ are untouched by this head.

**Null space null(W) — the "invisible" subspace.**

Input directions in $\text{null}(W)$ are completely ignored by the layer. If an input vector has all its "mass" in the null space, the output is zero — the layer cannot see it. This is both a limitation and a design feature: in mechanistic interpretability, the null space of a query projection $W_Q$ is the set of residual stream directions that this attention head does not use to form its queries. These directions are invisible to the head's query computation.

**Row space row(W) — the "listening" subspace.**

The row space is the complement of the null space: input directions in $\text{row}(W)$ are the ones the layer actually "listens to". The projection of the input onto $\text{row}(W)$ determines the output; the null space projection vanishes. In a transformer, the row space of $W_K$ determines which directions of the residual stream the key computation is sensitive to. Keys "live in" the row space of $W_K$.

**Left null space null(W^T) — the "unreachable" output subspace.**

The left null space $\text{null}(W^\top) \subseteq \mathbb{R}^m$ is orthogonal to $\text{col}(W)$. No input, however crafted, can produce an output with a component in this subspace from this layer. In a transformer with residual connections, the left null space of one layer's output projection is the subspace that must be written by other layers. This is why residual connections are essential: they allow information to flow "around" layers whose column spaces don't reach certain directions.

```text
AI INTERPRETATION: WEIGHT MATRIX SUBSPACES
════════════════════════════════════════════════════════════════════

  W ∈ ℝᵐˣⁿ, rank r

  ℝⁿ (input):                        ℝᵐ (output):
  ┌──────────────────┐               ┌──────────────────┐
  │  row(W)          │               │  col(W)          │
  │  dim r           │───W──────────▶│  dim r           │
  │  "W listens here"│               │  "W writes here" │
  │        ⊕         │               │        ⊕         │
  │  null(W)         │               │  null(Wᵀ)        │
  │  dim n−r         │──W──▶{0}      │  dim m−r         │
  │  "W ignores this"│               │  "W never here"  │
  └──────────────────┘               └──────────────────┘

  Transformer layer (y = Wx + b, residual):
  • Attention head h reads from row(W_Q^h), row(W_K^h), row(W_V^h)
  • Head h writes to col(W_O^h)
  • Information in null(W_Q^h) is invisible to head h's queries
  • Information in null(W_V^h) is not read by head h's values
  • The residual stream preserves ALL d dimensions; individual
    layers each modify only their respective column space subspaces
```

---

## 8. Subspace Operations

### 8.1 Sum of Subspaces

For subspaces $W_1, W_2 \subseteq V$, their **sum** is:

$$W_1 + W_2 = \{\mathbf{w}_1 + \mathbf{w}_2 : \mathbf{w}_1 \in W_1,\ \mathbf{w}_2 \in W_2\}$$

**Theorem.** $W_1 + W_2$ is a subspace of $V$.

_Proof._

1. Contains $\mathbf{0}$: $\mathbf{0} = \mathbf{0} + \mathbf{0} \in W_1 + W_2$ ✓
2. Closed under $+$: $(\mathbf{w}_1 + \mathbf{w}_2) + (\mathbf{w}_1' + \mathbf{w}_2') = (\mathbf{w}_1 + \mathbf{w}_1') + (\mathbf{w}_2 + \mathbf{w}_2') \in W_1 + W_2$ ✓ (since $W_1$, $W_2$ are closed)
3. Closed under $\cdot$: $\alpha(\mathbf{w}_1 + \mathbf{w}_2) = \alpha\mathbf{w}_1 + \alpha\mathbf{w}_2 \in W_1 + W_2$ ✓ ∎

$W_1 + W_2$ is the **smallest** subspace containing both $W_1$ and $W_2$. Any subspace $U$ that contains both $W_1$ and $W_2$ must contain all sums $\mathbf{w}_1 + \mathbf{w}_2$ (by closure), so $W_1 + W_2 \subseteq U$.

**Grassmann dimension formula:**

$$\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$$

**Important distinction:** $W_1 \cup W_2$ (set union) is **not** a subspace in general. Take $W_1 = \text{span}\{(1,0)\}$ (x-axis) and $W_2 = \text{span}\{(0,1)\}$ (y-axis) in $\mathbb{R}^2$. Then $(1,0) \in W_1 \cup W_2$ and $(0,1) \in W_1 \cup W_2$, but $(1,0) + (0,1) = (1,1) \notin W_1 \cup W_2$. The union is not closed under addition. The sum $W_1 + W_2 = \mathbb{R}^2$ (the whole plane) is a subspace; the union (just the two axes) is not.

### 8.2 Intersection of Subspaces

For subspaces $W_1, W_2 \subseteq V$, their **intersection** is:

$$W_1 \cap W_2 = \{\mathbf{v} \in V : \mathbf{v} \in W_1 \text{ and } \mathbf{v} \in W_2\}$$

**Theorem.** $W_1 \cap W_2$ is a subspace of $V$.

_Proof._

1. Contains $\mathbf{0}$: $\mathbf{0} \in W_1$ and $\mathbf{0} \in W_2$, so $\mathbf{0} \in W_1 \cap W_2$ ✓
2. Closed under $+$: if $\mathbf{v}, \mathbf{w} \in W_1 \cap W_2$, then $\mathbf{v} + \mathbf{w} \in W_1$ (since $W_1$ closed) and $\mathbf{v} + \mathbf{w} \in W_2$ (since $W_2$ closed), so $\mathbf{v} + \mathbf{w} \in W_1 \cap W_2$ ✓
3. Closed under $\cdot$: $\alpha\mathbf{v} \in W_1$ and $\alpha\mathbf{v} \in W_2$, so $\alpha\mathbf{v} \in W_1 \cap W_2$ ✓ ∎

$W_1 \cap W_2$ is the **largest** subspace contained in both $W_1$ and $W_2$.

**Computing $W_1 \cap W_2$:** If $W_1 = \text{null}(A_1)$ and $W_2 = \text{null}(A_2)$, then:

$$W_1 \cap W_2 = \text{null}\left(\begin{bmatrix} A_1 \\ A_2 \end{bmatrix}\right)$$

For general subspaces given by spanning sets: find vectors in $\text{span}\{\mathbf{u}_1, \ldots\}$ that also lie in $\text{span}\{\mathbf{v}_1, \ldots\}$ by setting up a linear system and solving.

### 8.3 Direct Sum

Subspaces $W_1$ and $W_2$ are **complementary** in $V$ (equivalently, $V$ is their **direct sum**) if:

1. $W_1 + W_2 = V$ — they together span $V$
2. $W_1 \cap W_2 = \{\mathbf{0}\}$ — they share only the origin

We write $V = W_1 \oplus W_2$.

**Theorem.** $V = W_1 \oplus W_2$ if and only if every $\mathbf{v} \in V$ has a **unique** decomposition $\mathbf{v} = \mathbf{w}_1 + \mathbf{w}_2$ with $\mathbf{w}_1 \in W_1$ and $\mathbf{w}_2 \in W_2$.

_Proof of uniqueness._ If $\mathbf{v} = \mathbf{w}_1 + \mathbf{w}_2 = \mathbf{w}_1' + \mathbf{w}_2'$, then $\mathbf{w}_1 - \mathbf{w}_1' = \mathbf{w}_2' - \mathbf{w}_2$. The left side is in $W_1$; the right side is in $W_2$. So both sides lie in $W_1 \cap W_2 = \{\mathbf{0}\}$. Hence $\mathbf{w}_1 = \mathbf{w}_1'$ and $\mathbf{w}_2 = \mathbf{w}_2'$. ∎

**Dimension:** $\dim(W_1 \oplus W_2) = \dim(W_1) + \dim(W_2)$. This follows from the Grassmann formula with $\dim(W_1 \cap W_2) = 0$.

**Complements are not unique.** For a given subspace $W \subseteq V$, there are many subspaces $W'$ such that $V = W \oplus W'$. The **orthogonal complement** $W^\perp$ is the canonical, unique choice:

$$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}$$

**Theorem** (for finite-dimensional inner product spaces). $V = W \oplus W^\perp$, and $\dim(W^\perp) = \dim(V) - \dim(W)$.

The fundamental subspace decompositions from Section 7 are orthogonal direct sums:

$$\mathbb{R}^n = \text{row}(A) \oplus \text{null}(A) = \text{row}(A) \oplus \text{row}(A)^\perp$$
$$\mathbb{R}^m = \text{col}(A) \oplus \text{null}(A^\top) = \text{col}(A) \oplus \text{col}(A)^\perp$$

**Multiple direct sums.** The direct sum generalises: $V = W_1 \oplus W_2 \oplus \cdots \oplus W_k$ if the $W_i$ are pairwise "independent" (each $W_i \cap (W_1 + \cdots + \hat{W}_i + \cdots + W_k) = \{\mathbf{0}\}$) and together span $V$. Then every $\mathbf{v}$ has a unique decomposition $\mathbf{v} = \mathbf{w}_1 + \cdots + \mathbf{w}_k$ and $\dim(V) = \sum_i \dim(W_i)$.

In the spectral theorem (Section 13), the orthogonal decomposition into eigenspaces is a multiple orthogonal direct sum: $\mathbb{R}^n = E(\lambda_1) \oplus E(\lambda_2) \oplus \cdots \oplus E(\lambda_k)$.

### 8.4 Projection onto Subspaces

Given a subspace $W \subseteq V$, the **orthogonal projection** $\text{Proj}_W: V \to W$ maps each vector $\mathbf{v}$ to the closest point in $W$:

$$\text{Proj}_W(\mathbf{v}) = \arg\min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|$$

The unique minimiser exists because $W$ is a closed convex set (any subspace is convex, and in finite dimensions, closed).

**Characterisation.** $\hat{\mathbf{v}} = \text{Proj}_W(\mathbf{v})$ is the unique vector in $W$ such that $\mathbf{v} - \hat{\mathbf{v}} \perp W$, i.e., $\langle \mathbf{v} - \hat{\mathbf{v}}, \mathbf{w} \rangle = 0$ for all $\mathbf{w} \in W$.

**Formula 1: orthonormal basis.**

If $W$ has orthonormal basis $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$, then:

$$\text{Proj}_W(\mathbf{v}) = \sum_{i=1}^k \langle \mathbf{v}, \mathbf{q}_i \rangle \mathbf{q}_i$$

In matrix form, with $Q = [\mathbf{q}_1 \mid \cdots \mid \mathbf{q}_k] \in \mathbb{R}^{n \times k}$ (orthonormal columns):

$$\text{Proj}_W(\mathbf{v}) = QQ^\top \mathbf{v}$$

The **projection matrix** is $P = QQ^\top \in \mathbb{R}^{n \times n}$.

**Formula 2: general (non-orthonormal) basis.**

If $W = \text{col}(A)$ for $A \in \mathbb{R}^{n \times k}$ with linearly independent columns (full column rank):

$$\text{Proj}_W(\mathbf{v}) = A(A^\top A)^{-1} A^\top \mathbf{v}$$

The projection matrix is $P = A(A^\top A)^{-1}A^\top$. The matrix $(A^\top A)^{-1} A^\top$ is the **Moore-Penrose pseudoinverse** $A^\dagger$ when $A$ has full column rank.

**Properties of an orthogonal projection matrix $P$:**

| Property    | Expression                 | Meaning                                                                    |
| ----------- | -------------------------- | -------------------------------------------------------------------------- |
| Idempotent  | $P^2 = P$                  | Projecting twice = projecting once                                         |
| Symmetric   | $P^\top = P$               | Projection is self-adjoint                                                 |
| Eigenvalues | $\lambda \in \{0, 1\}$     | Vectors in $W$ map to themselves; vectors in $W^\perp$ map to $\mathbf{0}$ |
| Rank        | $\text{rank}(P) = \dim(W)$ | Rank = dimension of the subspace projected onto                            |
| Trace       | $\text{tr}(P) = \dim(W)$   | Since eigenvalues are 0s and 1s; trace = sum of eigenvalues                |
| Complement  | $I - P$                    | Projection onto $W^\perp$                                                  |

**Proof that $P^2 = P$:** $P^2 = (QQ^\top)(QQ^\top) = Q(Q^\top Q)Q^\top = QIQ^\top = QQ^\top = P$ (using $Q^\top Q = I_k$ for orthonormal columns). ∎

**Decomposition.** Every $\mathbf{v} \in V$ decomposes as:

$$\mathbf{v} = \underbrace{P\mathbf{v}}_{\in W} + \underbrace{(I-P)\mathbf{v}}_{\in W^\perp}$$

The residual $(I-P)\mathbf{v} = \mathbf{v} - P\mathbf{v}$ is the component of $\mathbf{v}$ orthogonal to $W$, and $(I-P)$ is itself an orthogonal projection matrix (onto $W^\perp$).

**AI applications of projection:**

- **Least squares:** the best-fit solution $\hat{\mathbf{x}} = (A^\top A)^{-1} A^\top \mathbf{b}$ projects $\mathbf{b}$ onto $\text{col}(A)$; least squares is orthogonal projection
- **PCA:** projecting data onto the top-$r$ principal component subspace is a rank-$r$ projection $P = U_r U_r^\top$ where $U_r$ contains the top $r$ left singular vectors
- **Attention:** (soft) attention can be viewed as a weighted projection of value vectors onto query-determined subspaces
- **Concept erasure:** removing a concept from an embedding by projecting onto the orthogonal complement of the concept subspace: $P_{\perp\text{concept}} = I - P_{\text{concept}}$; used in LEACE and related methods

### 8.5 Subspace Angles and Principal Angles

The angle between two vectors is well-defined: $\cos\theta = \langle \mathbf{u}, \mathbf{v} \rangle / (\|\mathbf{u}\| \|\mathbf{v}\|)$. The "angle" between two subspaces is more subtle — it requires a collection of angles called **principal angles**.

**Definition.** For subspaces $W_1, W_2 \subseteq V$ with $\dim(W_1) = p$, $\dim(W_2) = q$, $k = \min(p, q)$, the **principal angles** $0 \leq \theta_1 \leq \theta_2 \leq \cdots \leq \theta_k \leq \pi/2$ are defined recursively:

$$\cos\theta_j = \max_{\substack{\mathbf{u} \in W_1 \\ \mathbf{v} \in W_2}} \langle \mathbf{u}, \mathbf{v} \rangle \quad \text{subject to } \|\mathbf{u}\| = \|\mathbf{v}\| = 1,\ \mathbf{u} \perp \mathbf{u}_i,\ \mathbf{v} \perp \mathbf{v}_i \text{ for } i < j$$

**Computation via SVD.** If $Q_1 \in \mathbb{R}^{n \times p}$ and $Q_2 \in \mathbb{R}^{n \times q}$ are orthonormal bases for $W_1$ and $W_2$:

$$Q_1^\top Q_2 = U \Sigma V^\top \qquad (\text{SVD})$$

Then $\cos\theta_i = \sigma_i$ (the $i$-th singular value of $Q_1^\top Q_2$). The principal angles are the arc-cosines of the singular values.

**Interpretation:**

- $\theta_1 = 0$: the subspaces share a common direction (their intersection is non-trivial)
- $\theta_k = \pi/2$: the subspaces have a pair of orthogonal directions (they are "partially orthogonal")
- All $\theta_i = \pi/2$: the subspaces are orthogonal ($W_1 \perp W_2$, i.e., $W_1 \subseteq W_2^\perp$)
- All $\theta_i = 0$: $W_1 \subseteq W_2$ or $W_2 \subseteq W_1$ (one contains the other)

**AI applications:**

- **Head overlap:** principal angles between two attention heads' OV subspaces measure how much they overlap. If $\theta_1 \approx 0$, the heads share a direction and are partially redundant. If all $\theta_i \approx \pi/2$, the heads write to orthogonal subspaces — they are fully independent.
- **Gradient similarity across layers:** principal angles between gradient subspaces at different layers measure gradient diversity during training.
- **Representation similarity:** the CKA (Centered Kernel Alignment) measure of representation similarity between two networks is related to principal angles between their representation subspaces.
- **LoRA subspace alignment:** after fine-tuning, the principal angles between the LoRA update subspace and the top-$r$ gradient subspace reveal how well LoRA captured the gradient's preferred directions.

---

## 9. Inner Product Spaces and Orthogonality

> **Recall:** [Vectors and Spaces §5–§6](../01-Vectors-and-Spaces/notes.md#6-inner-product-spaces) developed norms and inner products concretely in $\mathbb{R}^n$ — dot product, angle, Cauchy-Schwarz, and orthogonal projection. This section extends those ideas to abstract inner product spaces: general Hilbert spaces, orthonormal bases, Gram-Schmidt orthogonalization, and orthogonal complements. The abstract setting is what makes these concepts applicable to function spaces (Fourier analysis, kernels) and not just $\mathbb{R}^n$.

### 9.1 Inner Products

An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:

1. **Symmetry:** $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **Linearity in the first argument:** $\langle \alpha\mathbf{u} + \beta\mathbf{w}, \mathbf{v} \rangle = \alpha\langle\mathbf{u},\mathbf{v}\rangle + \beta\langle\mathbf{w},\mathbf{v}\rangle$
3. **Positive definiteness:** $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality if and only if $\mathbf{v} = \mathbf{0}$

Together, properties 1 and 2 imply linearity in the second argument as well (by symmetry), making the inner product **bilinear**: linear in each argument separately.

The **induced norm** is $\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$, and the induced **metric** (distance) is $d(\mathbf{u},\mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$.

**Standard inner products:**

| Space                     | Inner product                                                                          | Induced norm                                       |
| ------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------- |
| $\mathbb{R}^n$            | $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = \sum_i u_i v_i$ | $\|\mathbf{u}\| = \sqrt{\sum_i u_i^2}$ (Euclidean) |
| $\mathbb{R}^{m \times n}$ | $\langle A, B \rangle = \text{tr}(A^\top B) = \sum_{i,j} A_{ij} B_{ij}$                | $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$ (Frobenius) |
| $C([a,b])$                | $\langle f, g \rangle = \int_a^b f(t) g(t)\, dt$                                       | $\|f\| = \sqrt{\int_a^b f(t)^2\, dt}$ ($L^2$ norm) |

**Weighted inner product.** For a symmetric positive definite matrix $M \in \mathbb{R}^{n \times n}$:

$$\langle \mathbf{u}, \mathbf{v} \rangle_M = \mathbf{u}^\top M \mathbf{v}$$

This defines a different geometry on $\mathbb{R}^n$: the unit ball $\{\mathbf{v} : \langle\mathbf{v},\mathbf{v}\rangle_M \leq 1\}$ is an ellipsoid rather than a sphere. The natural gradient in optimisation uses the Fisher information matrix $F$ as the weight matrix: $\langle \mathbf{g}, \mathbf{g} \rangle_F = \mathbf{g}^\top F \mathbf{g}$ measures gradient magnitude in the geometry of the statistical model, not the flat geometry of parameter space.

**Cauchy-Schwarz inequality.** For any inner product:

$$|\langle \mathbf{u}, \mathbf{v} \rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$

with equality if and only if $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent ($\mathbf{u} = \alpha\mathbf{v}$ for some scalar $\alpha$).

This is one of the most important inequalities in mathematics. It implies:

- Cosine similarity $\frac{\langle \mathbf{u},\mathbf{v}\rangle}{\|\mathbf{u}\|\|\mathbf{v}\|}$ is always in $[-1, 1]$ — well-defined as a cosine
- Triangle inequality $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ (from Cauchy-Schwarz applied to $\langle\mathbf{u},\mathbf{v}\rangle \leq \|\mathbf{u}\|\|\mathbf{v}\|$)
- The law of cosines: $\|\mathbf{u}-\mathbf{v}\|^2 = \|\mathbf{u}\|^2 - 2\langle\mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2$

### 9.2 Orthogonality

Vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal**, written $\mathbf{u} \perp \mathbf{v}$, if $\langle \mathbf{u}, \mathbf{v} \rangle = 0$.

Orthogonality generalises perpendicularity from Euclidean geometry to any inner product space. In the Fourier inner product on $L^2$, $\sin(mt)$ and $\cos(nt)$ are orthogonal for all integers $m, n$. In the Frobenius inner product on matrices, symmetric and skew-symmetric matrices are orthogonal (since $\text{tr}(A^\top B) = 0$ when $A = A^\top$ and $B = -B^\top$).

**Pythagorean theorem.** If $\mathbf{u} \perp \mathbf{v}$, then:

$$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$

_Proof:_
$$\|\mathbf{u} + \mathbf{v}\|^2 = \langle\mathbf{u}+\mathbf{v},\mathbf{u}+\mathbf{v}\rangle = \langle\mathbf{u},\mathbf{u}\rangle + 2\langle\mathbf{u},\mathbf{v}\rangle + \langle\mathbf{v},\mathbf{v}\rangle = \|\mathbf{u}\|^2 + 0 + \|\mathbf{v}\|^2 \quad \checkmark$$

More generally, if $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are pairwise orthogonal:

$$\left\|\sum_{i=1}^k \mathbf{v}_i\right\|^2 = \sum_{i=1}^k \|\mathbf{v}_i\|^2$$

**Orthogonal sets and independence.** Any set of non-zero pairwise orthogonal vectors is linearly independent.

_Proof._ Suppose $\sum_{i=1}^k \alpha_i \mathbf{v}_i = \mathbf{0}$ with $\mathbf{v}_i \neq \mathbf{0}$ pairwise orthogonal. Take the inner product of both sides with $\mathbf{v}_j$:

$$0 = \left\langle \sum_i \alpha_i \mathbf{v}_i,\ \mathbf{v}_j \right\rangle = \sum_i \alpha_i \langle \mathbf{v}_i, \mathbf{v}_j \rangle = \alpha_j \langle \mathbf{v}_j, \mathbf{v}_j \rangle = \alpha_j \|\mathbf{v}_j\|^2$$

Since $\mathbf{v}_j \neq \mathbf{0}$, we have $\|\mathbf{v}_j\|^2 > 0$, so $\alpha_j = 0$. This holds for all $j$. ∎

This is a powerful observation: orthogonality implies independence. An orthogonal set is automatically independent, so an orthogonal spanning set is automatically a basis.

### 9.3 Gram-Schmidt Orthogonalisation

Given $k$ linearly independent vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ in an inner product space, the **Gram-Schmidt process** produces an orthonormal set $\{\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_k\}$ such that:

$$\text{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_j\} = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_j\} \quad \text{for all } j = 1, \ldots, k$$

The key invariant is that at each step, the orthonormal basis spans the same subspace as the original vectors up to that index.

**Algorithm:**

```text
Step 1:  u₁ = v₁
         q₁ = u₁ / ‖u₁‖

Step 2:  u₂ = v₂ − ⟨v₂, q₁⟩ q₁
         q₂ = u₂ / ‖u₂‖

Step 3:  u₃ = v₃ − ⟨v₃, q₁⟩ q₁ − ⟨v₃, q₂⟩ q₂
         q₃ = u₃ / ‖u₃‖

         ⋮

Step j:  uⱼ = vⱼ − ∑ᵢ₌₁^{j-1} ⟨vⱼ, qᵢ⟩ qᵢ
         qⱼ = uⱼ / ‖uⱼ‖
```

**What each step does.** At step $j$, we subtract from $\mathbf{v}_j$ all of its projections onto the previously constructed orthonormal vectors $\mathbf{q}_1, \ldots, \mathbf{q}_{j-1}$. The result $\mathbf{u}_j$ is the component of $\mathbf{v}_j$ not explained by the previous vectors — the "new" direction that $\mathbf{v}_j$ adds. Normalising gives $\mathbf{q}_j$.

**Why $\mathbf{u}_j \neq \mathbf{0}$:** Since $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are linearly independent, $\mathbf{v}_j \notin \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_{j-1}\} = \text{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_{j-1}\}$. Therefore $\mathbf{u}_j = \mathbf{v}_j - \sum_{i<j}\langle\mathbf{v}_j,\mathbf{q}_i\rangle\mathbf{q}_i \neq \mathbf{0}$ (if it were zero, $\mathbf{v}_j$ would be in the span of the previous $\mathbf{q}$'s, contradicting independence).

**Verification of orthonormality.** For $i < j$:
$$\langle \mathbf{q}_j, \mathbf{q}_i \rangle = \frac{1}{\|\mathbf{u}_j\|} \left\langle \mathbf{v}_j - \sum_{\ell < j} \langle \mathbf{v}_j, \mathbf{q}_\ell \rangle \mathbf{q}_\ell,\ \mathbf{q}_i \right\rangle = \frac{1}{\|\mathbf{u}_j\|}\left(\langle \mathbf{v}_j, \mathbf{q}_i \rangle - \langle \mathbf{v}_j, \mathbf{q}_i \rangle \langle \mathbf{q}_i, \mathbf{q}_i \rangle\right) = 0$$

**Connection to QR decomposition.** The Gram-Schmidt process is the algorithm underlying QR decomposition. For a matrix $A = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k]$ with independent columns:

$$A = QR$$

where $Q = [\mathbf{q}_1 \mid \cdots \mid \mathbf{q}_k]$ has orthonormal columns and $R$ is upper triangular with positive diagonal:

$$R_{ij} = \begin{cases} \langle \mathbf{v}_j, \mathbf{q}_i \rangle & \text{if } i \leq j \\ 0 & \text{if } i > j \end{cases}$$

The diagonal entries are $R_{jj} = \|\mathbf{u}_j\| > 0$.

**Numerical issues.** The classical Gram-Schmidt algorithm as stated is numerically unstable for nearly dependent vectors: rounding errors accumulate and the resulting vectors may not be accurately orthogonal. The **Modified Gram-Schmidt** algorithm reorders the operations to reduce error propagation. For production use, **Householder QR** (using Householder reflections rather than projections) is numerically stable and preferred.

**Worked example.** Let $\mathbf{v}_1 = (1, 1, 0)^\top$, $\mathbf{v}_2 = (1, 0, 1)^\top$ in $\mathbb{R}^3$.

Step 1: $\mathbf{u}_1 = (1,1,0)^\top$, $\|\mathbf{u}_1\| = \sqrt{2}$, $\mathbf{q}_1 = (1/\sqrt{2},\ 1/\sqrt{2},\ 0)^\top$

Step 2: $\langle \mathbf{v}_2, \mathbf{q}_1 \rangle = (1)(1/\sqrt{2}) + (0)(1/\sqrt{2}) + (1)(0) = 1/\sqrt{2}$

$\mathbf{u}_2 = (1,0,1)^\top - \frac{1}{\sqrt{2}}(1/\sqrt{2}, 1/\sqrt{2}, 0)^\top = (1,0,1)^\top - (1/2, 1/2, 0)^\top = (1/2, -1/2, 1)^\top$

$\|\mathbf{u}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}$

$\mathbf{q}_2 = \frac{1}{\sqrt{3/2}}(1/2, -1/2, 1)^\top = (1/\sqrt{6},\ -1/\sqrt{6},\ 2/\sqrt{6})^\top$

Verify: $\langle\mathbf{q}_1,\mathbf{q}_2\rangle = (1/\sqrt{2})(1/\sqrt{6}) + (1/\sqrt{2})(-1/\sqrt{6}) + 0 = 1/\sqrt{12} - 1/\sqrt{12} = 0$ ✓

### 9.4 The Orthogonal Complement

For a subspace $W \subseteq V$, the **orthogonal complement** is:

$$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}$$

**$W^\perp$ is a subspace** (easy check):

1. $\langle \mathbf{0}, \mathbf{w} \rangle = 0$ for all $\mathbf{w}$ ✓
2. If $\langle \mathbf{v}, \mathbf{w} \rangle = 0$ and $\langle \mathbf{u}, \mathbf{w} \rangle = 0$ for all $\mathbf{w} \in W$, then $\langle \mathbf{v}+\mathbf{u}, \mathbf{w} \rangle = 0$ ✓
3. $\langle \alpha\mathbf{v}, \mathbf{w} \rangle = \alpha \langle \mathbf{v},\mathbf{w} \rangle = 0$ ✓

**Properties** (for finite-dimensional inner product spaces):

| Property                 | Statement                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------- |
| Double complement        | $(W^\perp)^\perp = W$                                                               |
| Dimension                | $\dim(W) + \dim(W^\perp) = \dim(V)$                                                 |
| Trivial intersection     | $W \cap W^\perp = \{\mathbf{0}\}$                                                   |
| Direct sum decomposition | $V = W \oplus W^\perp$                                                              |
| Fundamental subspaces    | $\text{null}(A) = \text{row}(A)^\perp$; $\text{null}(A^\top) = \text{col}(A)^\perp$ |

**Why $(W^\perp)^\perp = W$:** Clearly $W \subseteq (W^\perp)^\perp$ (every vector in $W$ is orthogonal to everything in $W^\perp$). By the dimension formula: $\dim((W^\perp)^\perp) = \dim(V) - \dim(W^\perp) = \dim(V) - (\dim(V) - \dim(W)) = \dim(W)$. Same dimension and one contains the other → they are equal.

**Computing $W^\perp$.** If $W = \text{span}\{\mathbf{w}_1, \ldots, \mathbf{w}_k\}$ and we form $A = [\mathbf{w}_1 \mid \cdots \mid \mathbf{w}_k]^\top$ (rows are the $\mathbf{w}_i$), then:

$$W^\perp = \text{null}(A)$$

because $\langle \mathbf{v}, \mathbf{w}_i \rangle = 0$ for all $i$ is equivalent to $A\mathbf{v} = \mathbf{0}$.

### 9.5 Orthogonal Bases for AI

The choice of basis — orthogonal vs arbitrary — has concrete consequences in AI applications.

**Interpretability and orthogonal bases.** When the basis vectors of an embedding space are orthogonal (as in the standard basis of $\mathbb{R}^d$), each dimension corresponds to an independent direction. Inner products between embeddings measure similarity in a well-defined way. Projections onto individual basis directions give clean, independent "components". In practice, the features of a learned representation are rarely aligned with the standard basis — but if they were, each feature could be read off by looking at one coordinate.

**The superposition hypothesis and basis independence.** Elhage et al. (2022) argue that transformers represent features as directions in $\mathbb{R}^d$, not as individual coordinates. The model does not commit to any particular orthonormal basis; instead, features are placed as arbitrary unit vectors in $\mathbb{R}^d$. When there are fewer features than dimensions, they can be nearly orthogonal (low interference). When features exceed $d$, they must be non-orthogonal (superposed). The interference between features is measured by the inner products between their representing directions — exactly the off-diagonal entries of the Gram matrix of feature vectors.

**Cosine similarity.** The dominant similarity measure in embedding spaces:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

This is the cosine of the angle between $\mathbf{u}$ and $\mathbf{v}$, ranging from $-1$ (anti-parallel) to $+1$ (parallel). Orthogonal vectors have cosine similarity 0 — they are "unrelated" by this measure. The cosine similarity is basis-dependent: it changes if we apply a non-orthogonal change of basis to the space.

**Orthogonal vs orthonormal bases.** Orthogonal bases (pairwise orthogonal, but not necessarily unit length) are often more natural than orthonormal ones for intermediate computations. An orthonormal basis (each vector unit length AND pairwise orthogonal) is the "gold standard" for numerical stability and interpretability. The Gram-Schmidt process converts any independent set into an orthonormal one.

**PCA as an orthonormal basis for data.** Principal Component Analysis finds the orthonormal basis of $\mathbb{R}^d$ that best aligns with the directions of maximal variance in a dataset. In the PCA basis, the covariance matrix is diagonal (its off-diagonal elements are zero), meaning the principal components are statistically uncorrelated. PCA is precisely the process of finding the orthonormal eigenbasis of the covariance matrix and using it as the new coordinate system.

**Layer normalisation and orthogonality.** Layer normalisation in transformers includes a learnable scale $\gamma$ and shift $\beta$:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

The mean-subtraction step projects $\mathbf{x}$ onto the orthogonal complement of the all-ones vector $\mathbf{1}/\sqrt{d}$. This is an explicit orthogonal projection: $\mathbf{x} - \mu\mathbf{1} = (I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top)\mathbf{x}$, where $P = \frac{1}{d}\mathbf{1}\mathbf{1}^\top$ is the projection onto the 1-dimensional "mean direction" and $I - P$ is the projection onto its orthogonal complement.

---

## 10. Affine Subspaces and Quotient Spaces

### 10.1 Affine Subspaces

An **affine subspace** (also called an **affine flat** or **coset**) is a translate of a linear subspace:

$$W + \mathbf{v}_0 = \{\mathbf{w} + \mathbf{v}_0 : \mathbf{w} \in W\}$$

for a fixed offset vector $\mathbf{v}_0 \in V$ and a linear subspace $W \subseteq V$.

**Geometric picture:**

- If $W$ is a line through the origin, $W + \mathbf{v}_0$ is a parallel line through $\mathbf{v}_0$
- If $W$ is a plane through the origin, $W + \mathbf{v}_0$ is a parallel plane through $\mathbf{v}_0$
- The affine subspace is a "shifted" copy of the linear subspace; it has the same "shape" and dimension as $W$, but it generally does not pass through the origin (unless $\mathbf{v}_0 \in W$, in which case $W + \mathbf{v}_0 = W$)

**Affine subspaces are NOT vector subspaces.** Unless $\mathbf{v}_0 \in W$, the affine subspace $W + \mathbf{v}_0$ does not contain $\mathbf{0}$ and is not closed under addition (the sum of two elements is offset by $2\mathbf{v}_0$, not $\mathbf{v}_0$).

**Key example: solution sets of linear systems.** The solution set of $A\mathbf{x} = \mathbf{b}$ with $\mathbf{b} \neq \mathbf{0}$ is an affine subspace:

$$\{\mathbf{x} : A\mathbf{x} = \mathbf{b}\} = \mathbf{x}_p + \text{null}(A)$$

where $\mathbf{x}_p$ is any particular solution and $\text{null}(A)$ is the null space (a linear subspace). The solution set is a translate of the null space. All solutions lie in the same affine subspace parallel to $\text{null}(A)$, offset by $\mathbf{x}_p$.

**Other examples:**

- A line not through the origin in $\mathbb{R}^2$: $\{(1,0) + t(1,2) : t \in \mathbb{R}\}$ — a translate of the span of $(1,2)$
- The probability simplex $\Delta^{n-1}$: an $(n-1)$-dimensional affine subspace of $\mathbb{R}^n$, defined by $\sum_i p_i = 1$ — a translate of the hyperplane $\sum_i x_i = 0$
- Batch normalisation output space: after fixing the batch statistics, the normalised output lives in an affine subspace determined by the running mean and variance

**Dimension of an affine subspace.** The affine subspace $W + \mathbf{v}_0$ has the same dimension as the underlying linear subspace $W$. A $k$-dimensional affine subspace in $\mathbb{R}^n$ is sometimes called a **$k$-flat**.

```text
AFFINE VS LINEAR SUBSPACES
════════════════════════════════════════════════════════════════════

  Linear subspace W:            Affine subspace W + v₀:
  ─ passes through origin       ─ passes through v₀ (not 0)
  ─ closed under + and scaling  ─ closed under affine combinations
  ─ IS a vector space           ─ is NOT a vector space
  ─ examples: lines/planes/     ─ examples: lines/planes/
    hyperplanes through 0         hyperplanes NOT through 0

  In AI:
  Linear: null(W), col(W), LoRA update subspace
  Affine: solution set of Ax=b, probability simplex,
          offset embeddings before centering
```

### 10.2 Affine Combinations

An **affine combination** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is a linear combination $\sum_{i=1}^k \alpha_i \mathbf{v}_i$ where the coefficients sum to one:

$$\sum_{i=1}^k \alpha_i = 1$$

Affine combinations "stay within" affine subspaces: if $\mathbf{v}_1, \ldots, \mathbf{v}_k \in W + \mathbf{v}_0$, then any affine combination of them also lies in $W + \mathbf{v}_0$.

**Convex combinations** are affine combinations with the additional constraint $\alpha_i \geq 0$:

$$\sum_{i=1}^k \alpha_i \mathbf{v}_i \quad \text{with } \alpha_i \geq 0 \text{ and } \sum_i \alpha_i = 1$$

Convex combinations stay within the **convex hull** of $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$.

**Why the constraint $\sum \alpha_i = 1$ matters.** Without it, scaling the vectors by a common factor would scale the combination — the result would depend on "how large" the vectors are. With $\sum \alpha_i = 1$, the combination is "location-aware" rather than "direction-aware" — it picks a point in the affine hull regardless of scale.

**AI applications of affine and convex combinations:**

- **Embedding interpolation.** Given two embedding vectors $\mathbf{u}$ and $\mathbf{v}$, the linear interpolation $\alpha\mathbf{u} + (1-\alpha)\mathbf{v}$ for $\alpha \in [0,1]$ is a convex combination. This is the operation underlying latent space arithmetic: "find the midpoint of 'cat' and 'dog' in embedding space". Note: this is NOT generally a valid probability-weighted average of the corresponding tokens' distributions — that arithmetic must happen in logit space.

- **Spherical linear interpolation (slerp).** For unit vectors (on the unit sphere), linear interpolation does not stay on the sphere. Slerp uses trigonometric weights to interpolate along the great circle: $\text{slerp}(\mathbf{u}, \mathbf{v}, t) = \frac{\sin((1-t)\theta)}{\sin\theta}\mathbf{u} + \frac{\sin(t\theta)}{\sin\theta}\mathbf{v}$ where $\cos\theta = \langle\mathbf{u},\mathbf{v}\rangle$. The weights $\frac{\sin((1-t)\theta)}{\sin\theta}$ and $\frac{\sin(t\theta)}{\sin\theta}$ sum to... almost 1 (they are affine-like weights for the sphere geometry).

- **Model averaging and interpolation.** Averaging two neural networks $\theta_1$ and $\theta_2$ by $\theta = \frac{1}{2}(\theta_1 + \theta_2)$ is a convex combination in parameter space. "Model soup" (Wortsman et al. 2022) averages fine-tuned models; WiSE-FT interpolates between pretrained and fine-tuned weights. These are affine combinations in $\mathbb{R}^p$.

- **Probability distributions.** A mixture distribution $p_{\text{mix}}(x) = \alpha p_1(x) + (1-\alpha) p_2(x)$ is a convex combination of two distributions. This is valid because: $p_{\text{mix}}(x) \geq 0$ (convex combination of non-negatives) and $\int p_{\text{mix}} dx = \alpha + (1-\alpha) = 1$ (the constraint $\sum \alpha_i = 1$ ensures the mixture is still a probability distribution). This is why mixture models work: affine combinations with unit-sum weights preserve the probability simplex.

### 10.3 Quotient Spaces

The **quotient space** $V / W$ (read "V mod W") captures the idea of "ignoring the $W$ directions" — it identifies all vectors that differ by an element of $W$.

**Definition.** For a subspace $W \subseteq V$, define the equivalence relation: $\mathbf{u} \sim \mathbf{v}$ iff $\mathbf{u} - \mathbf{v} \in W$. The **coset** of $\mathbf{v}$ is:

$$[\mathbf{v}] = \mathbf{v} + W = \{\mathbf{v} + \mathbf{w} : \mathbf{w} \in W\}$$

This is an affine subspace — a translate of $W$ passing through $\mathbf{v}$. Note that $[\mathbf{u}] = [\mathbf{v}]$ iff $\mathbf{u} - \mathbf{v} \in W$ (the cosets are identical iff the vectors are equivalent).

The **quotient space** $V / W = \{[\mathbf{v}] : \mathbf{v} \in V\}$ is the collection of all cosets of $W$ in $V$.

**Operations on $V/W$:**

- **Addition:** $[\mathbf{u}] + [\mathbf{v}] = [\mathbf{u} + \mathbf{v}]$ (well-defined: the result does not depend on which representatives $\mathbf{u}, \mathbf{v}$ we choose, as long as the cosets are the same)
- **Scalar multiplication:** $\alpha[\mathbf{v}] = [\alpha\mathbf{v}]$ (well-defined by the same argument)

These operations make $V/W$ a vector space (the quotient space). The zero vector of $V/W$ is $[\mathbf{0}] = W$ (the coset containing $\mathbf{0}$, which is $W$ itself).

**Dimension:** $\dim(V/W) = \dim(V) - \dim(W)$.

_Intuition._ $V/W$ "collapses" the $W$ directions to a point (the zero element $[W]$) and retains only the directions perpendicular to $W$. The quotient space has dimension $= \dim(V) - \dim(W)$ because it "forgets" $\dim(W)$ directions.

**First Isomorphism Theorem.** For a linear map $T: V \to U$:

$$V / \text{null}(T) \cong \text{col}(T)$$

The quotient space $V / \text{null}(T)$ is isomorphic to the image of $T$. Intuitively: the null space is exactly the "ambiguity" in $T$ — different inputs mapping to the same output; the quotient space removes this ambiguity, leaving a bijection between cosets and outputs.

**AI applications:**

- **Layer normalisation.** The mean-subtraction in LayerNorm — subtracting the mean of all $d$ coordinates from each coordinate — is equivalent to projecting onto the orthogonal complement of the all-ones direction. The "normalised" space can be viewed as the quotient $\mathbb{R}^d / \text{span}\{\mathbf{1}\}$, where the direction $\mathbf{1} = (1,1,\ldots,1)^\top / \sqrt{d}$ is "divided out". Two activations that differ only by a constant shift (equal means) are identified in this quotient space.

- **Residual connections.** The residual connection $\mathbf{x} \leftarrow \mathbf{x} + f(\mathbf{x})$ adds a vector to the current residual stream. From the quotient space perspective: the "content" of the residual stream modulo the current layer's contribution is what gets passed to the next layer. Each layer writes to its column-space subspace; the rest of $\mathbb{R}^d$ is preserved as-is.

- **Equivalence classes in training.** If $W = \text{null}(A)$ for a data matrix $A$, then all weight vectors in the same coset $\mathbf{w} + \text{null}(A)$ produce the same predictions on the training data. The space of distinct prediction functions is the quotient $\mathbb{R}^n / \text{null}(A) \cong \text{row}(A)$. Gradient descent with squared loss finds the minimum-norm representative of the coset $[\mathbf{w}^*] = \mathbf{w}^* + \text{null}(A)$ — the vector in the coset closest to the origin.

### 10.4 Cosets and Their Structure

The cosets $\mathbf{v} + W$ partition $V$ into disjoint, equally-shaped affine subspaces:

- **Partition:** every $\mathbf{v} \in V$ belongs to exactly one coset; cosets are either identical or disjoint
- **Uniform shape:** every coset is a translate of $W$, so all cosets have the same dimension ($= \dim(W)$) and the same "shape"
- **No overlap:** two cosets $\mathbf{u} + W$ and $\mathbf{v} + W$ are either equal (when $\mathbf{u} - \mathbf{v} \in W$) or disjoint

**Solution structure for linear systems.** This is the most immediate application. For $A\mathbf{x} = \mathbf{b}$:

- The solution set (if non-empty) is the coset $\mathbf{x}_p + \text{null}(A)$ for any particular solution $\mathbf{x}_p$
- All solutions in this coset produce the same output: $A(\mathbf{x}_p + \mathbf{z}) = A\mathbf{x}_p + A\mathbf{z} = \mathbf{b} + \mathbf{0} = \mathbf{b}$ for any $\mathbf{z} \in \text{null}(A)$
- The minimum-norm solution (the "pseudoinverse solution") is $\mathbf{x}^+ = A^\dagger \mathbf{b} = A^\top(AA^\top)^{-1}\mathbf{b}$, which lies in $\text{row}(A)$ (the orthogonal complement of $\text{null}(A)$ within the coset)

**Implicit bias as coset selection.** In overparameterised neural networks ($n > m$, more parameters than constraints), gradient descent with zero initialisation and MSE loss converges to the minimum-norm solution in $\text{row}(A)$ (the unique solution in $\text{row}(A) \cap (\mathbf{x}_p + \text{null}(A))$). This "implicit bias" towards minimum-norm solutions is a coset-selection phenomenon: gradient descent selects the minimum-$\ell^2$-norm representative from the coset of all equally good solutions.

---

## 11. Subspaces in Functional Analysis

### 11.1 Infinite-Dimensional Spaces and Closed Subspaces

In finite dimensions, every subspace is automatically closed (in the topological sense). This ceases to be true in infinite dimensions, where one must distinguish between **algebraic subspaces** (satisfying the three conditions of Section 3) and **closed subspaces** (additionally closed under limits).

**Closed subspace.** A subspace $W$ of a Hilbert space $H$ is **closed** if every Cauchy sequence in $W$ converges to a limit that remains in $W$. Equivalently, $W$ is closed if it equals its topological closure $\bar{W}$.

**Why closedness matters.** In infinite dimensions:

- The projection theorem ($H = W \oplus W^\perp$) holds only for **closed** subspaces
- The best approximation in $W$ exists only if $W$ is closed
- The spectral theorem and other key results require closed invariant subspaces

**Examples of closed subspaces:**

- $\mathcal{P}_n \subset L^2([0,1])$: polynomials of degree $\leq n$; finite-dimensional, hence closed
- $\text{null}(T)$ for a bounded linear operator $T$: always closed (because $T$ is continuous and $\{0\}$ is closed)
- $L^2_{\text{even}}([−\pi,\pi]) = \{f \in L^2 : f(-x) = f(x)\}$: the even functions; closed under limits
- The span of any finite set of vectors in a Hilbert space: always closed (finite-dimensional subspace)

**Examples of non-closed subspaces in infinite dimensions:**

- $\mathcal{P} \subset L^2([0,1])$: all polynomials (of any degree); algebraically a subspace, but not closed — the sequence $f_n(x) = \sum_{k=0}^n x^k / k!$ converges in $L^2$ to $e^x$, which is not a polynomial. The closure $\overline{\mathcal{P}} = L^2([0,1])$ (by the Weierstrass approximation theorem in $L^2$)
- Finitely supported sequences in $\ell^2$: sequences with only finitely many non-zero entries; algebraically a subspace, but limit of $\mathbf{e}_1, \mathbf{e}_1 + (1/2)\mathbf{e}_2, \mathbf{e}_1 + (1/2)\mathbf{e}_2 + (1/3)\mathbf{e}_3, \ldots$ is $(1, 1/2, 1/3, \ldots) \in \ell^2$ but not finitely supported

**Projection in infinite dimensions.** The projection theorem extends to Hilbert spaces: if $W$ is a **closed** subspace of a Hilbert space $H$, then $H = W \oplus W^\perp$ and every $\mathbf{v} \in H$ has a unique best approximation $\hat{\mathbf{v}} \in W$. This is the foundation of least squares in infinite dimensions and of the representer theorem in kernel methods.

### 11.2 Function Spaces Relevant to AI

**$L^2(\mathbb{R}^d)$ — square-integrable functions on $\mathbb{R}^d$.**

The space of functions $f: \mathbb{R}^d \to \mathbb{R}$ with $\int_{\mathbb{R}^d} f(\mathbf{x})^2 \, d\mathbf{x} < \infty$, equipped with inner product $\langle f, g \rangle = \int f(\mathbf{x}) g(\mathbf{x}) \, d\mathbf{x}$. This is a separable Hilbert space. It is the natural setting for:

- Kernel methods: the RKHS of a translation-invariant kernel is a subspace of $L^2$
- Probability: densities of probability distributions with finite second moment live here
- Signal processing: finite-energy signals are in $L^2(\mathbb{R})$
- Neural networks: functions representable by a network can be analysed as elements of $L^2$

**Sobolev spaces $W^{k,p}(\Omega)$.**

Functions with $k$ weak derivatives all lying in $L^p(\Omega)$, equipped with norms that account for function values and derivative values. They specify "smoothness":

- $W^{0,2} = L^2$ (no smoothness condition beyond square integrability)
- $W^{1,2} = H^1$ (square-integrable function with square-integrable first derivative): relevant for PDEs and PINNs
- $W^{2,2} = H^2$ (second derivatives in $L^2$): relevant for spline regression, Gaussian processes with Matérn kernels

Sobolev spaces are used in physics-informed neural networks (PINNs): the solution to a PDE is sought in a Sobolev space, and the network is trained to minimise a loss that penalises PDE residuals in the $L^2$ norm. The constraint "solution in $H^1$" is a subspace constraint on the function space.

**Reproducing Kernel Hilbert Spaces (RKHS).**

An RKHS is a Hilbert space $\mathcal{H}$ of functions $f: \mathcal{X} \to \mathbb{R}$ such that **point evaluation is a bounded linear functional**: for each $\mathbf{x} \in \mathcal{X}$, the map $f \mapsto f(\mathbf{x})$ is continuous in the $\mathcal{H}$-norm.

By the Riesz representation theorem, there exists a unique function $k(\cdot, \mathbf{x}) \in \mathcal{H}$ such that:

$$f(\mathbf{x}) = \langle f, k(\cdot, \mathbf{x}) \rangle_{\mathcal{H}} \quad \text{for all } f \in \mathcal{H}$$

The function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ defined by $k(\mathbf{x}, \mathbf{x}') = \langle k(\cdot, \mathbf{x}), k(\cdot, \mathbf{x}') \rangle_{\mathcal{H}}$ is the **reproducing kernel**. It is symmetric and positive semi-definite.

**Representer theorem.** For any regularised learning problem of the form:

$$\min_{f \in \mathcal{H}_k} \sum_{i=1}^n \ell(y_i, f(\mathbf{x}_i)) + \lambda \|f\|_{\mathcal{H}_k}^2$$

the optimal $f^*$ lies in the **finite-dimensional subspace** $\text{span}\{k(\cdot, \mathbf{x}_1), \ldots, k(\cdot, \mathbf{x}_n)\} \subseteq \mathcal{H}_k$. This is a subspace result: despite the infinite-dimensional function space, the optimal solution lies in an $n$-dimensional subspace spanned by the kernel functions at the training points. SVMs, Gaussian process regression, and kernel ridge regression all instantiate this theorem.

**Common kernels and their RKHS subspaces:**

- **Linear kernel** $k(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$: RKHS = $\mathbb{R}^d$ (linear functions); finite-dimensional subspace
- **RBF / Gaussian kernel** $k(\mathbf{x}, \mathbf{x}') = \exp(-\|\mathbf{x}-\mathbf{x}'\|^2 / 2\sigma^2)$: RKHS is a dense subspace of $L^2(\mathbb{R}^d)$; infinite-dimensional
- **Matérn kernel**: RKHS is the Sobolev space $W^{\nu+d/2, 2}$; smoothness parameter $\nu$ controls which Sobolev subspace
- **Polynomial kernel** $k(\mathbf{x}, \mathbf{x}') = (1 + \mathbf{x}^\top\mathbf{x}')^p$: RKHS = polynomials of degree $\leq p$; $(d+p)$-choose-$p$-dimensional

### 11.3 Neural Networks as Subspaces of Function Spaces

A neural network architecture with parameter space $\mathbb{R}^p$ defines a parametric family of functions:

$$\mathcal{F}_\theta = \{f_\theta : \theta \in \mathbb{R}^p\} \subset L^2(\mathcal{X})$$

This family is **not** a subspace of $L^2$. In general, $f_{\theta_1} + f_{\theta_2}$ is not equal to $f_\theta$ for any $\theta$ (the sum of two networks is not itself a network with the same architecture). Similarly for scalar multiples. The non-linearity of the architecture means the function family is a non-linear manifold embedded in $L^2$, not a subspace.

**However, the tangent space at any parameter $\theta$ IS a subspace.** The tangent space to the manifold $\mathcal{F}_\theta$ at the point $f_\theta$ is:

$$T_{f_\theta}\mathcal{F}_\theta = \text{span}\left\{\frac{\partial f_\theta}{\partial \theta_i} : i = 1, \ldots, p\right\} \subseteq L^2(\mathcal{X})$$

This is the span of $p$ functions — a (at most) $p$-dimensional linear subspace of $L^2$. When you take a gradient step $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$, you are moving in the direction that lies in this tangent space. First-order optimisation operates in the tangent subspace.

**Neural Tangent Kernel (NTK).** In the infinite-width limit (network width $\to \infty$ with appropriate scaling), the evolution of the network function during gradient flow is governed by a linear equation in function space:

$$\dot{f}_{t}(\mathbf{x}) = -\sum_{\mathbf{x}'} K(\mathbf{x}, \mathbf{x}') \frac{\partial \mathcal{L}}{\partial f(\mathbf{x}')}$$

where $K(\mathbf{x}, \mathbf{x}') = \langle \nabla_\theta f_\theta(\mathbf{x}), \nabla_\theta f_\theta(\mathbf{x}') \rangle_{\mathbb{R}^p}$ is the NTK. In the infinite-width limit, $K$ becomes constant during training. The trained network converges to kernel regression with the NTK as kernel — meaning the trained function lies in the RKHS defined by the NTK, which is a subspace of $L^2$. The NTK result says: in the lazy training regime, the neural network effectively performs a linear projection onto a specific infinite-dimensional subspace of function space.

**Universal approximation and subspace density.** The universal approximation theorem says that the set of functions representable by a neural network with bounded width and arbitrary depth (or arbitrary width and one hidden layer) is **dense** in $C([0,1]^n)$ (continuous functions on the unit hypercube) under the uniform norm. "Dense" means: for any continuous function $f$ and any $\varepsilon > 0$, there is a network function $f_\theta$ with $\|f - f_\theta\|_\infty < \varepsilon$. This is a subspace density statement: the parametric family $\mathcal{F}$ is dense in the function space $C([0,1]^n)$. Depth (or width) determines which subspace is reachable; nonlinearity is what allows density.

### 11.4 Krylov Subspaces

Krylov subspaces are the foundation of the most practical iterative linear algebra algorithms. They connect the abstract geometry of subspaces to the computational efficiency of matrix-vector products.

**Definition.** For a matrix $A \in \mathbb{R}^{n \times n}$ and a vector $\mathbf{b} \in \mathbb{R}^n$, the **Krylov subspace** of order $k$ is:

$$\mathcal{K}_k(A, \mathbf{b}) = \text{span}\{\mathbf{b},\ A\mathbf{b},\ A^2\mathbf{b},\ \ldots,\ A^{k-1}\mathbf{b}\}$$

This is the span of the first $k$ vectors in the sequence $\mathbf{b}, A\mathbf{b}, A^2\mathbf{b}, \ldots$ — the orbit of $\mathbf{b}$ under repeated application of $A$.

**Nested structure.** The Krylov subspaces form a nested sequence:

$$\mathcal{K}_1 \subseteq \mathcal{K}_2 \subseteq \mathcal{K}_3 \subseteq \cdots \subseteq \mathcal{K}_n$$

The sequence stabilises at some $r \leq n$: $\mathcal{K}_r = \mathcal{K}_{r+1} = \cdots$ At that point, $A \cdot \mathcal{K}_r \subseteq \mathcal{K}_r$ (the subspace is invariant under $A$), and $r$ equals the degree of the minimal polynomial of $A$ with respect to $\mathbf{b}$.

**Krylov methods.** Iterative solvers based on Krylov subspaces find the best approximate solution within $\mathcal{K}_k$ at step $k$, then expand the subspace:

| Method                       | Problem                                 | Optimisation in $\mathcal{K}_k$                                  |
| ---------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| **Conjugate Gradients (CG)** | $A\mathbf{x} = \mathbf{b}$, $A$ SPD     | Minimises $\|\mathbf{x} - \mathbf{x}^*\|_A$ over $\mathcal{K}_k$ |
| **GMRES**                    | $A\mathbf{x} = \mathbf{b}$, general $A$ | Minimises $\|A\mathbf{x} - \mathbf{b}\|_2$ over $\mathcal{K}_k$  |
| **Lanczos**                  | Eigenvalues of $A$ symmetric            | Finds best rank-$k$ approximation to spectrum                    |
| **Arnoldi**                  | Eigenvalues of general $A$              | Orthogonalises $\mathcal{K}_k$ via Gram-Schmidt                  |

Each step costs one matrix-vector product with $A$ and $O(k \cdot n)$ work for orthogonalisation. For a sparse $A$ with $\text{nnz}$ non-zeros, each step costs $O(\text{nnz})$. Contrast with direct methods (Gaussian elimination): $O(n^3)$ cost regardless of sparsity.

**AI applications of Krylov methods:**

- **Second-order optimisation.** Methods like K-FAC (Kronecker-Factored Approximate Curvature) need to solve linear systems involving the Fisher information matrix $F$ to compute the natural gradient. Krylov methods can solve $F\mathbf{d} = \mathbf{g}$ (where $\mathbf{g}$ is the gradient) without explicitly forming $F$ — only matrix-vector products $F\mathbf{v}$ are needed, which can be computed efficiently.
- **Eigenvalue computation for interpretability.** Computing the top eigenvalues of the Hessian $\nabla^2 \mathcal{L}$ or the covariance matrix of gradients is done via Lanczos, which builds a Krylov subspace using Hessian-vector products. These products are available cheaply via the "pearlmutter trick" (forward-over-backward AD). The Krylov subspace approach gives the dominant eigenspace in $O(k)$ Hessian-vector products.
- **Linear attention and state space models.** SSMs (S4, Mamba) compute recurrences of the form $\mathbf{h}_{t+1} = A\mathbf{h}_t + B\mathbf{x}_t$ whose outputs lie in Krylov-like subspaces. The efficiency of convolutional SSM computation is related to the structure of the Krylov subspace generated by the state matrix $A$ and input matrix $B$.

---

## 12. Subspace Methods in Machine Learning

### 12.1 PCA as Subspace Finding

Principal Component Analysis is fundamentally a subspace problem: find the rank-$r$ subspace of $\mathbb{R}^d$ that best explains the variance in a dataset.

**Setup.** Given data $X = [\mathbf{x}_1 \mid \cdots \mid \mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ with centred rows ($\bar{\mathbf{x}} = \frac{1}{n}\sum_i \mathbf{x}_i = \mathbf{0}$), the **sample covariance matrix** is:

$$C = \frac{1}{n} X^\top X = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \mathbf{x}_i^\top \in \mathbb{R}^{d \times d}$$

$C$ is symmetric positive semidefinite. Its eigendecomposition $C = Q \Lambda Q^\top$ (with $Q$ orthogonal, $\Lambda$ diagonal with $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$) defines the **principal components**.

**PCA as orthogonal projection.** The optimal rank-$r$ subspace $W^*$ minimises the mean squared reconstruction error:

$$W^* = \arg\min_{\substack{W \subseteq \mathbb{R}^d \\ \dim(W) = r}} \frac{1}{n} \sum_{i=1}^n \|\mathbf{x}_i - \text{Proj}_W(\mathbf{x}_i)\|^2$$

**Theorem** (Eckart-Young for PCA). The solution is $W^* = \text{span}\{\mathbf{q}_1, \ldots, \mathbf{q}_r\}$, the span of the top-$r$ eigenvectors of $C$ (equivalently, the top-$r$ left singular vectors of $\frac{1}{\sqrt{n}}X$).

The minimum reconstruction error equals the sum of the discarded eigenvalues: $\frac{1}{n}\sum_{i=1}^n \|\mathbf{x}_i - P_{W^*}\mathbf{x}_i\|^2 = \sum_{j=r+1}^d \lambda_j$.

**Equivalences.** PCA can be described four equivalent ways, all pointing to the same subspace:

1. **Maximum variance:** find the $r$-dimensional subspace on which projected data has maximum variance
2. **Minimum reconstruction error:** find the $r$-dimensional subspace minimising projection error (the formulation above)
3. **SVD:** compute $\frac{1}{\sqrt{n}}X = U\Sigma V^\top$; the top-$r$ right singular vectors $\{v_1,\ldots,v_r\}$ span $W^*$
4. **Eigendecomposition:** compute eigenvectors of $C$; top-$r$ eigenvectors span $W^*$

**AI applications of PCA as subspace finding:**

- **Embedding analysis.** PCA of a word embedding matrix $E \in \mathbb{R}^{|V| \times d}$ reveals the dominant semantic axes. The top principal component often corresponds to a frequency direction; later components correspond to semantic distinctions. PCA on embeddings is a way of finding the most informative subspace of the $d$-dimensional embedding space.
- **Activation subspaces.** PCA on the activations of a layer across many inputs reveals the effective dimensionality of the layer's representation. If 95% of variance is explained by the top-$k$ principal components, the layer effectively operates in a $k$-dimensional subspace of $\mathbb{R}^{d_{\text{hidden}}}$, even though nominally $d_{\text{hidden}}$-dimensional.
- **Weight matrix analysis.** PCA on the rows or columns of a weight matrix reveals which directions the matrix emphasises. The top singular vectors of $W$ are the principal directions — the row space direction that $W$ maps to the largest column space direction.
- **Collapse detection.** In self-supervised learning (SimCLR, BYOL, VICReg), representation collapse means all embeddings converge to a low-dimensional subspace (or a single point). Tracking the rank (effective number of significant singular values) of the representation matrix over training detects and diagnoses collapse.

### 12.2 Subspace Tracking During Training

The gradient subspace evolves over the course of training, and its structure is a key diagnostic for understanding optimisation.

**The gradient subspace.** At training step $t$, the gradient $\nabla_\theta \mathcal{L}(\theta_t) \in \mathbb{R}^p$ is a single vector. Over $T$ steps, the gradients $\{\nabla_\theta \mathcal{L}(\theta_t)\}_{t=1}^T$ lie in some subspace of $\mathbb{R}^p$. The **gradient subspace** at step $T$ is approximately:

$$G_T = \text{span}\{\nabla_\theta \mathcal{L}(\theta_1), \nabla_\theta \mathcal{L}(\theta_2), \ldots, \nabla_\theta \mathcal{L}(\theta_T)\}$$

or more precisely, the leading eigenspace of the gradient covariance matrix $\sum_{t=1}^T \nabla_t \nabla_t^\top$.

**Empirical finding: the gradient subspace is small.** Gur-Ari, Bar-On, and Shashua (2018) showed that for large neural networks, the gradient vectors observed during training effectively lie in a subspace of dimension much smaller than $p$. For models with $p \sim 10^7$ to $10^9$ parameters, the effective gradient subspace has dimension in the thousands or tens of thousands. This means:

- Gradient descent traverses a low-dimensional "groove" in the high-dimensional parameter space
- The $p - k$ directions orthogonal to the gradient subspace are never updated
- A $k$-dimensional update rule (where $k \ll p$) can achieve nearly the same performance

**Intrinsic dimensionality of fine-tuning** (Aghajanyan, Zettlemoyer, Gupta, 2020). They parameterise the fine-tuning update as $\theta = \theta_0 + P\delta$ where $P \in \mathbb{R}^{p \times d}$ is a fixed random projection matrix and $\delta \in \mathbb{R}^d$ is optimised. The smallest $d$ for which fine-tuning achieves 90% of full-fine-tuning performance is the **intrinsic dimension**. For GPT-2 fine-tuned on MRPC (a sentence similarity task), the intrinsic dimension is $\approx 200$ out of $p \approx 1.5 \times 10^8$ parameters. Fine-tuning happens in a subspace of effective dimension $\approx 200$.

**LoRA as gradient subspace approximation.** LoRA restricts $\Delta W = BA^\top$ to a rank-$r$ subspace of $\mathbb{R}^{m \times n}$. The justification is that the gradient $\nabla_W \mathcal{L}$ during fine-tuning lives in a low-rank subspace. By restricting updates to rank $r$, LoRA approximates this gradient subspace with $r(m+n)$ parameters instead of $mn$. The choice of $r$ balances: larger $r$ = larger subspace = more expressive but more parameters; smaller $r$ = tighter constraint but fewer parameters. The "right" $r$ is approximately the intrinsic dimension of the task.

**GaLore** (Zhao et al. 2024). Rather than permanently fixing the subspace (as LoRA does), GaLore periodically updates the projection subspace by computing the top-$r$ singular vectors of the gradient matrix. The gradient is projected onto the current subspace before the optimiser step; optimizer state is maintained in the subspace. Memory savings come from keeping optimizer state in $\mathbb{R}^r$ rather than $\mathbb{R}^{m \times n}$.

### 12.3 Representation Subspaces

The linear representation hypothesis proposes that concepts are encoded as linear subspaces of the residual stream in large language models. This is a strong structural claim with substantial empirical support.

**Concept directions.** A **concept direction** for a binary concept $c$ (e.g., "positive sentiment", "female gender", "medical domain") is a unit vector $\hat{\mathbf{d}} \in \mathbb{R}^d$ such that $\langle \mathbf{x}, \hat{\mathbf{d}} \rangle > 0$ indicates the concept is present and $\langle \mathbf{x}, \hat{\mathbf{d}} \rangle < 0$ indicates it is absent. The concept is represented as a 1-dimensional subspace (a line through the origin) in $\mathbb{R}^d$.

Evidence: probing classifiers (linear models predicting a concept from an embedding) work well for many concepts, suggesting the concept is linearly decodable. The existence of a linear probe is evidence that the concept has a direction in the embedding space — a 1D subspace.

**Multi-dimensional concept subspaces.** Some concepts are not well-represented by a single direction. "Colour" might involve multiple hues; "grammatical role" might involve multiple positions. In these cases, the concept corresponds to a subspace of dimension $> 1$. The **concept subspace** $C \subseteq \mathbb{R}^d$ contains all directions relevant to the concept.

**LEACE (Least-squares Concept Erasure)** (Belrose et al. 2023). Given two sets of representations — one with concept present, one absent — LEACE finds the minimal subspace $W$ such that projecting onto $W^\perp$ makes the concept undetectable by any linear probe. This is explicitly a subspace computation:

1. Compute the within-class covariance $\Sigma_W$ and between-class difference $\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0$
2. Find the projection direction that maximally separates classes (Fisher discriminant direction)
3. Project onto the orthogonal complement: $\mathbf{x} \leftarrow (I - P_{\text{concept}})\mathbf{x}$

Concept erasure = projection onto the orthogonal complement of the concept subspace.

**Distributed representations.** A single concept may activate many neurons, and a single neuron may respond to many concepts. This is distributed representation, and it is the normal state in large networks:

- **Distributed over neurons:** each neuron contributes a little to many concepts; the concept is "smeared" across many dimensions
- **Superposition:** many concepts share the same dimensions (polysemanticity); the concept subspaces are non-orthogonal

The superposition hypothesis (Elhage et al. 2022) says that in the regime where the number of features $F > d$ (more features than dimensions), the optimal strategy is to store features as nearly-orthogonal, non-orthogonal unit vectors. Interference is minimised by choosing feature directions to be as orthogonal as possible, but it cannot be entirely eliminated when $F > d$. This is the geometry of packing more vectors into a space than the dimension allows.

### 12.4 Mechanistic Interpretability Through Subspaces

Mechanistic interpretability analyses the internal computations of neural networks at the level of circuits — sequences of operations that implement specific algorithmic behaviours. Subspace geometry is the natural language for this analysis.

**Residual stream decomposition.** At layer $\ell$, the residual stream $\mathbf{x}^\ell \in \mathbb{R}^d$ is updated by:

$$\mathbf{x}^{\ell+1} = \mathbf{x}^\ell + \text{Attn}^\ell(\mathbf{x}^\ell) + \text{MLP}^\ell(\mathbf{x}^\ell)$$

Each component (attention head $h$, MLP layer $\ell$) writes to a specific subspace of $\mathbb{R}^d$ (its column space) and reads from another (its row space). The residual stream is a shared highway; every component adds its contribution to the same $d$-dimensional space. The subspace written to by component $c$ is $\text{col}(W_{O}^c)$ for attention heads or determined by the MLP weight matrices for MLP blocks.

**OV and QK circuits.** For attention head $h$ with projections $W_Q^h, W_K^h, W_V^h \in \mathbb{R}^{d \times d_k}$ and $W_O^h \in \mathbb{R}^{d_k \times d}$:

- **QK circuit:** $W_Q^h (W_K^h)^\top \in \mathbb{R}^{d \times d}$ but has rank $\leq d_k$; it maps the residual stream to a $d_k$-dimensional subspace for query-key comparison; the head computes attention scores using only $d_k$ dimensions of the $d$-dimensional residual stream
- **OV circuit:** $W_O^h W_V^h \in \mathbb{R}^{d \times d}$ but has rank $\leq d_k$; the head reads from a $d_k$-dimensional subspace (via $W_V^h$) and writes to a $d_k$-dimensional subspace (via $W_O^h$); the composition is a rank-$d_k$ map from $\mathbb{R}^d$ to $\mathbb{R}^d$

Both the QK and OV circuits are low-rank (rank $\leq d_k$) operations in $\mathbb{R}^d$, even though $\mathbb{R}^d$ is $d$-dimensional. Each head has access to $d$-dimensional representations but can only "look at" or "write to" $d_k$-dimensional subspaces. When $d_k \ll d$ (typical: $d_k = d/H$ for $H$ heads), each head is a dimensionality-reducing probe and writer.

**Induction heads.** An induction head (Olsson et al. 2022) is an attention head that implements in-context learning by attending to previous occurrences of the current token. It works via two components: a "previous token head" that shifts information one position back (writing to a specific subspace), and an "induction head" that looks for matches in that subspace. The circuit is: head A writes "what came before position $t$" to a subspace $S_A$; head B reads from $S_A$ and attends accordingly. The communication between heads A and B happens through a shared subspace of the residual stream.

**Superposition and polysemanticity.** Toy models of superposition (Elhage et al. 2022) show that:

- With $F \leq d$ features: each feature can be stored in its own orthogonal direction; no interference; polysemanticity unnecessary
- With $F > d$ features (superposition regime): features are stored as nearly-orthogonal non-orthogonal directions; each direction contains a weighted sum of multiple features; individual neurons are polysemantic (they respond to multiple distinct features)
- The "geometry" of superposition: features are arranged as vertices of polytopes inscribed in the unit sphere; the maximum number of nearly-orthogonal unit vectors in $\mathbb{R}^d$ that have pairwise inner products $\leq \epsilon$ is approximately $e^{c \cdot d}$ for some constant $c > 0$ depending on $\epsilon$ — exponential in the dimension

### 12.5 Subspace Fine-Tuning Methods

The empirical low-dimensionality of fine-tuning gradients motivates a family of methods that explicitly restrict weight updates to specific subspaces. Here is a comparative overview:

**LoRA** (Hu et al. 2021).

Parameterise the weight update as $\Delta W = BA^\top$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{n \times r}$, rank $r \ll \min(m,n)$. The update lives in the subspace of rank-$\leq r$ matrices in $\mathbb{R}^{m \times n}$. Only $r(m+n)$ parameters are needed instead of $mn$. The pre-trained weights $W_0$ are frozen; only $A$ and $B$ are trained. At inference, the update is merged: $W = W_0 + BA^\top$, adding no latency.

**AdaLoRA** (Zhang et al. 2023).

Parameterise $\Delta W = P \Lambda Q^\top$ where $P \in \mathbb{R}^{m \times r}$, $Q \in \mathbb{R}^{n \times r}$ are updated by gradient descent and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_r)$ is a diagonal matrix of learned singular values. Singular values that shrink toward zero can be pruned during training, giving adaptive rank allocation: some layers get high rank, others get low rank, depending on the task's needs. The total parameter count is bounded, but rank is allocated where the gradient signal is strongest.

**IA³** (Liu et al. 2022).

Rather than adding a low-rank matrix, IA³ multiplies activations by learned scaling vectors: $\mathbf{h} \leftarrow \mathbf{l} \odot \mathbf{h}$ for a learned vector $\mathbf{l}$. Equivalently, this rescales the columns of weight matrices: $W_{\text{eff}} = W \cdot \text{diag}(\mathbf{l})$. Each learned vector is a 1D subspace modulation — a rank-1 multiplicative update to a direction in activation space. Extremely parameter-efficient ($\leq d$ parameters per layer).

**Prefix Tuning** (Li and Liang 2021).

Prepend $k$ trainable "prefix" token embeddings to the input sequence. At each attention layer, the prefix tokens contribute additional keys and values that the original tokens can attend to. The prefix embeddings inject task-specific information into the attention computation. The "prefix subspace" is $\text{span}\{\mathbf{p}_1, \ldots, \mathbf{p}_k\} \subseteq \mathbb{R}^d$ — the $k$-dimensional subspace of information injected at each layer.

**DoRA** (Liu et al. 2024).

Decompose the weight matrix as $W = \|W\|_c \cdot (W / \|W\|_c)$ (magnitude times direction, computed column-wise). Fine-tune magnitude and direction separately: magnitude is a scalar per column (1D), direction uses LoRA (rank-$r$ subspace). The decomposition reflects the observation that weight magnitude and weight direction play different roles in learning: magnitude controls the "strength" of a feature; direction controls "which feature". DoRA consistently outperforms LoRA at the same rank by freeing the direction update from the magnitude constraint.

**DARE** (Yu et al. 2024).

After standard fine-tuning, randomly prune $\delta W = W_{\text{FT}} - W_0$ with probability $p$ (set to zero) and rescale the remainder by $1/(1-p)$ to maintain expectation. The effect is to project the fine-tuning update onto a sparse "subspace" (sparse in the coordinate basis). Pruned updates can be summed across multiple fine-tuned models (model merging) with reduced interference, since the non-zero coordinates are less likely to overlap.

```text
SUBSPACE FINE-TUNING: COMPARISON
════════════════════════════════════════════════════════════════════

  Method      Subspace type        Parameters        Memory saving
  ────────    ─────────────────    ──────────────    ─────────────
  Full FT     all of ℝᵐˣⁿ         mn                1×  (baseline)
  LoRA r      rank-r subspace      r(m+n)            mn/r(m+n)
  AdaLoRA     adaptive rank-r      r(m+n) + r        similar to LoRA
  IA³         column scaling       m or n            mn/d ≈ n
  Prefix k    k-dim prefix         k·d per layer     varies
  DoRA        magnitude + LoRA     r(m+n) + n        similar to LoRA
  DARE        sparse random        mn (train), 0 (prune)  0 (post-hoc)

  All methods exploit: fine-tuning gradient lives in low-dim subspace
```

---

## 13. Invariant Subspaces and Spectral Theory

### 13.1 Invariant Subspaces

A subspace $W \subseteq V$ is **invariant** under a linear map $T: V \to V$ if $T$ maps $W$ back into $W$:

$$T(W) \subseteq W \quad \text{i.e., for all } \mathbf{w} \in W:\ T(\mathbf{w}) \in W$$

Invariant subspaces are the subspaces that $T$ "respects" — it does not mix $W$ with its complement. Restricted to $W$, the map $T|_W: W \to W$ is a linear map from $W$ to itself.

**Trivial invariant subspaces.** Every linear map has two trivial invariant subspaces: $\{\mathbf{0}\}$ and $V$ itself. Any map $T$ maps $\mathbf{0}$ to $\mathbf{0}$ and maps $V$ to $T(V) \subseteq V$.

**Eigenspaces are 1-dimensional invariant subspaces.** If $T\mathbf{v} = \lambda\mathbf{v}$ (eigenvector equation), then $\text{span}\{\mathbf{v}\}$ is invariant: for any $\mathbf{w} = \alpha\mathbf{v} \in \text{span}\{\mathbf{v}\}$:

$$T(\mathbf{w}) = T(\alpha\mathbf{v}) = \alpha T(\mathbf{v}) = \alpha\lambda\mathbf{v} = \lambda\mathbf{w} \in \text{span}\{\mathbf{v}\}$$

Conversely, any 1-dimensional invariant subspace $\text{span}\{\mathbf{v}\}$ must have $T(\mathbf{v}) = \lambda\mathbf{v}$ for some scalar $\lambda$ (since $T(\mathbf{v})$ must lie in $\text{span}\{\mathbf{v}\}$). So eigenvectors and 1-dimensional invariant subspaces are the same thing.

**Generalised eigenspaces.** For eigenvalue $\lambda$, the **generalised eigenspace** is:

$$G(\lambda) = \text{null}((T - \lambda I)^k) \quad \text{for large enough } k$$

(Specifically, $k = n$ always works, but often a smaller $k$ suffices — the smallest $k$ with $\text{null}((T-\lambda I)^k) = \text{null}((T-\lambda I)^{k+1})$ is the **index** of $\lambda$.) The generalised eigenspace is invariant under $T$: $T(G(\lambda)) \subseteq G(\lambda)$.

**Invariant subspace decomposition.** The ideal scenario is when $V$ decomposes as a direct sum of invariant subspaces:

$$V = W_1 \oplus W_2 \oplus \cdots \oplus W_k \quad \text{with each } W_i \text{ invariant under } T$$

In this case, $T$ acts independently on each $W_i$: the restriction $T|_{W_i}$ is a linear map from $W_i$ to $W_i$, and the overall action of $T$ is the "direct sum" of these restricted maps. In a basis that respects this decomposition, the matrix of $T$ is **block diagonal**:

$$[T]_{\mathcal{B}} = \begin{pmatrix} [T|_{W_1}] & & \\ & \ddots & \\ & & [T|_{W_k}] \end{pmatrix}$$

Block diagonalisation is the goal of spectral theory: decompose $V$ into invariant subspaces so that $T$ is maximally "decoupled".

### 13.2 The Spectral Theorem via Invariant Subspaces

The **real spectral theorem** is the most important result about invariant subspaces. It applies to symmetric matrices (or more generally, self-adjoint operators on inner product spaces).

**Theorem (Real Spectral Theorem).** For a symmetric matrix $A = A^\top \in \mathbb{R}^{n \times n}$:

1. All eigenvalues of $A$ are real
2. Eigenvectors corresponding to distinct eigenvalues are orthogonal
3. $\mathbb{R}^n$ decomposes as an orthogonal direct sum of eigenspaces:
   $$\mathbb{R}^n = E(\lambda_1) \oplus E(\lambda_2) \oplus \cdots \oplus E(\lambda_k)$$
4. $A = Q \Lambda Q^\top$ where $Q$ is orthogonal (eigenvectors as columns) and $\Lambda$ is diagonal (eigenvalues)

**Why eigenvalues are real.** For symmetric $A$: $\langle A\mathbf{u}, \mathbf{v} \rangle = (A\mathbf{u})^\top \mathbf{v} = \mathbf{u}^\top A^\top \mathbf{v} = \mathbf{u}^\top A \mathbf{v} = \langle \mathbf{u}, A\mathbf{v} \rangle$. Such an operator is called **self-adjoint**. If $A\mathbf{v} = \lambda\mathbf{v}$ (over $\mathbb{C}$):
$$\lambda \langle \mathbf{v}, \mathbf{v} \rangle = \langle \lambda\mathbf{v}, \mathbf{v} \rangle = \langle A\mathbf{v}, \mathbf{v} \rangle = \langle \mathbf{v}, A\mathbf{v} \rangle = \langle \mathbf{v}, \lambda\mathbf{v} \rangle = \bar{\lambda} \langle \mathbf{v}, \mathbf{v} \rangle$$
Since $\|\mathbf{v}\|^2 > 0$: $\lambda = \bar{\lambda}$, so $\lambda \in \mathbb{R}$.

**Why eigenvectors for distinct eigenvalues are orthogonal.** If $A\mathbf{u} = \lambda\mathbf{u}$ and $A\mathbf{v} = \mu\mathbf{v}$ with $\lambda \neq \mu$:
$$\lambda \langle \mathbf{u}, \mathbf{v} \rangle = \langle A\mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{u}, A\mathbf{v} \rangle = \mu \langle \mathbf{u}, \mathbf{v} \rangle$$
So $(\lambda - \mu)\langle\mathbf{u},\mathbf{v}\rangle = 0$, and since $\lambda \neq \mu$: $\langle\mathbf{u},\mathbf{v}\rangle = 0$.

**Spectral decomposition.** Let $P_i$ be the orthogonal projection onto the eigenspace $E(\lambda_i)$. Then:

$$A = \sum_{i=1}^k \lambda_i P_i, \qquad I = \sum_{i=1}^k P_i, \qquad P_i P_j = \delta_{ij} P_i$$

The action of $A$ decomposes into independent scalings of orthogonal subspaces: $A$ scales $E(\lambda_i)$ by $\lambda_i$ and acts on each eigenspace independently.

**AI relevance.** Symmetric matrices appear constantly in deep learning:

- **Covariance matrices** $C = \frac{1}{n}X^\top X$: PCA = spectral decomposition of $C$; eigenspaces = principal component subspaces; eigenvalues = variances in each direction
- **Gram matrices** $G = XX^\top$ (dual of covariance): non-zero eigenvalues of $G$ = non-zero eigenvalues of $C$; eigenvectors related by $X$
- **Hessian** $H = \nabla^2 \mathcal{L}(\theta)$: curvature of loss landscape; eigenspaces = directions of different curvature; positive eigenvalues = directions curving up (need small learning rate); near-zero eigenvalues = flat directions (optimisation easy in these directions)
- **Laplacian of a graph** $L = D - A$: symmetric positive semidefinite; eigenvectors = graph Fourier basis; used in spectral clustering and graph neural networks

### 13.3 Singular Value Decomposition as Subspace Decomposition

The SVD extends the spectral theorem to non-square (and non-symmetric) matrices. It is the complete subspace decomposition of any linear map.

**Theorem (SVD).** For any matrix $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$, there exist orthogonal matrices $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$ and a matrix $\Sigma \in \mathbb{R}^{m \times n}$ with non-negative entries only on the main diagonal $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots$ such that:

$$A = U \Sigma V^\top$$

The $\sigma_i$ are the **singular values**, $\mathbf{u}_i$ (columns of $U$) are the **left singular vectors**, and $\mathbf{v}_i$ (columns of $V$) are the **right singular vectors**.

**SVD as complete subspace decomposition:**

- **Domain $\mathbb{R}^n$:** decomposes as $\text{row}(A) \oplus \text{null}(A)$
  - $\text{row}(A) = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_r\}$ (right singular vectors for non-zero singular values)
  - $\text{null}(A) = \text{span}\{\mathbf{v}_{r+1}, \ldots, \mathbf{v}_n\}$ (right singular vectors for zero singular values)

- **Codomain $\mathbb{R}^m$:** decomposes as $\text{col}(A) \oplus \text{null}(A^\top)$
  - $\text{col}(A) = \text{span}\{\mathbf{u}_1, \ldots, \mathbf{u}_r\}$ (left singular vectors for non-zero singular values)
  - $\text{null}(A^\top) = \text{span}\{\mathbf{u}_{r+1}, \ldots, \mathbf{u}_m\}$ (left singular vectors for zero singular values)

- **The action of $A$:** $A$ maps the $i$-th right singular direction to the $i$-th left singular direction, with scaling $\sigma_i$:
  $$A \mathbf{v}_i = \sigma_i \mathbf{u}_i \quad \text{for } i = 1, \ldots, r, \qquad A \mathbf{v}_i = \mathbf{0} \quad \text{for } i > r$$

**Rank-$k$ approximation.** The best rank-$k$ approximation to $A$ (in Frobenius or operator norm) is:

$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top$$

where $U_k, V_k$ contain the first $k$ left/right singular vectors and $\Sigma_k$ the top-$k$ singular values. This is the **Eckart-Young theorem**: among all rank-$k$ matrices, $A_k$ is closest to $A$.

The approximation error: $\|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$ (discarded singular values).

**AI applications of SVD as subspace decomposition:**

- **Weight matrix compression.** A weight matrix $W \in \mathbb{R}^{m \times n}$ can be approximated as $W \approx U_k \Sigma_k V_k^\top$ using the top-$k$ singular components. This compresses $mn$ parameters to $k(m+n)$. The column space of the compressed weight uses only the top-$k$ left singular directions; the null space grows accordingly.

- **LoRA in SVD language.** LoRA's $\Delta W = BA^\top$ is a rank-$r$ matrix, hence its SVD has at most $r$ non-zero singular values. The columns of $B$ span the column space of $\Delta W$; the columns of $A$ span the row space. Initialising $B = \mathbf{0}$ (as in standard LoRA) means $\Delta W = \mathbf{0}$ at the start, which is correct for starting from the pre-trained weights.

- **Activation space analysis.** The SVD of the activation matrix $H \in \mathbb{R}^{n \times d}$ (rows = activations for $n$ inputs) reveals the effective dimensionality of the representation. If many singular values are near zero, the activations lie in a low-dimensional subspace of $\mathbb{R}^d$. This is a direct measure of representation collapse or dimensionality.

- **Principal angle computation.** As noted in §8.5, the principal angles between two subspaces spanned by $Q_1$ and $Q_2$ are given by the singular values of $Q_1^\top Q_2$. SVD gives the geometry of subspace-to-subspace relationships.

### 13.4 Schur Decomposition

**Theorem (Schur).** Every square matrix $A \in \mathbb{R}^{n \times n}$ (working over $\mathbb{C}$, or over $\mathbb{R}$ for real Schur form) can be written as:

$$A = Q T Q^*$$

where $Q$ is unitary ($Q^* Q = I$) and $T$ is upper triangular. The diagonal entries of $T$ are the eigenvalues of $A$ (possibly complex, in some order).

**Real Schur form.** Over $\mathbb{R}$, $A = QTQ^\top$ where $T$ is **quasi-upper triangular**: block upper triangular with $1 \times 1$ blocks (real eigenvalues) and $2 \times 2$ blocks (complex conjugate eigenvalue pairs). The columns of $Q$ form an orthonormal basis.

**Invariant subspace connection.** The first $k$ columns of $Q$ span a $k$-dimensional **invariant subspace** of $A$:

$$A \, \text{span}\{Q_1, \ldots, Q_k\} \subseteq \text{span}\{Q_1, \ldots, Q_k\}$$

This follows because $AQ_k = QT$ and $T$ is upper triangular: $(AQ)_{j} = \sum_{i=1}^j T_{ij} Q_i$, so $AQ_j$ is a linear combination of $Q_1, \ldots, Q_j$. The Schur decomposition gives a nested sequence of invariant subspaces:

$$\{0\} \subset \text{span}\{Q_1\} \subset \text{span}\{Q_1, Q_2\} \subset \cdots \subset \mathbb{R}^n$$

Each subspace in the chain is invariant under $A$. The Schur form is the "staircase" of all invariant subspaces simultaneously.

**Relationship to spectral theorem.** For symmetric $A$: the Schur form has $T = \Lambda$ (diagonal), since eigenvectors are orthogonal and $T$ being upper triangular with orthogonal columns forces it diagonal. The Schur decomposition of a symmetric matrix is exactly the spectral decomposition.

**Practical relevance.** The QR algorithm (the standard algorithm for computing eigenvalues) works by iteratively applying QR decompositions to drive $A$ toward its Schur form. Each QR step refines the invariant subspace structure. The Schur form is numerically stable (unitary transformations do not amplify errors) and can always be computed, unlike the eigendecomposition (which may not exist for non-diagonalisable matrices).

---

## 14. Common Mistakes

| #   | Mistake                                                                                                                                               | Why It's Wrong                                                                                                                                                                                                                                                                                                                                                     | Correct Understanding                                                                                                                                                                                                                                     |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **"The union of two subspaces is a subspace"**                                                                                                        | $W_1 \cup W_2$ fails closure under addition: take $\mathbf{u} \in W_1 \setminus W_2$ and $\mathbf{v} \in W_2 \setminus W_1$; their sum $\mathbf{u} + \mathbf{v}$ need not lie in $W_1$ or $W_2$. Concrete example: $W_1 = \text{span}\{(1,0)\}$ (x-axis) and $W_2 = \text{span}\{(0,1)\}$ (y-axis) in $\mathbb{R}^2$; $(1,0) + (0,1) = (1,1) \notin W_1 \cup W_2$. | The **sum** $W_1 + W_2$ is a subspace; the union is generally not. Use $W_1 + W_2$ when you need the subspace generated by both.                                                                                                                          |
| 2   | **"An affine subspace is a subspace"**                                                                                                                | The solution set $\{A\mathbf{x} = \mathbf{b}\}$ with $\mathbf{b} \neq \mathbf{0}$ does not contain the zero vector ($A \cdot \mathbf{0} = \mathbf{0} \neq \mathbf{b}$) and is not closed under addition ($A(\mathbf{x}+\mathbf{y}) = 2\mathbf{b} \neq \mathbf{b}$). The probability simplex $\Delta^n$ is an affine subspace — not a vector subspace.              | An affine subspace = coset of a linear subspace = linear subspace shifted off the origin. It shares the same dimension and "shape" but is NOT a vector space. Subspaces must contain $\mathbf{0}$.                                                        |
| 3   | **"Closure under addition is sufficient for a subspace"**                                                                                             | Closure under addition alone (without scalar multiplication) is insufficient. The set $\mathbb{Z}^n \subset \mathbb{R}^n$ is closed under addition but $\pi \cdot (1,0,\ldots,0) = (\pi, 0, \ldots, 0) \notin \mathbb{Z}^n$, so it fails the scalar multiplication axiom.                                                                                          | All three conditions are required: (1) contains $\mathbf{0}$, (2) closed under addition, (3) closed under scalar multiplication. All three are independent of each other.                                                                                 |
| 4   | **"$\text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\} = \text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}\}$ implies $\mathbf{v}_{k+1} = \mathbf{0}$"** | The spans being equal means $\mathbf{v}_{k+1}$ is linearly dependent on $\mathbf{v}_1,\ldots,\mathbf{v}_k$ — it can be expressed as a linear combination of the earlier vectors. But it need not be zero. For example: $\text{span}\{(1,0),(0,1)\} = \text{span}\{(1,0),(0,1),(1,1)\}$; $(1,1)$ is not zero.                                                       | Equal spans means $\mathbf{v}_{k+1} \in \text{span}\{\mathbf{v}_1,\ldots,\mathbf{v}_k\}$ — redundant, not zero.                                                                                                                                           |
| 5   | **"$\dim(W_1 \cap W_2) = \dim(W_1) + \dim(W_2) - \dim(V)$"**                                                                                          | This formula is wrong. The correct formula involves the sum: $\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$. The dimension of $W_1 \cap W_2$ ranges between $\max(0, \dim(W_1)+\dim(W_2)-n)$ and $\min(\dim(W_1),\dim(W_2))$.                                                                                                                      | Correct formula: $\dim(W_1 \cap W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 + W_2)$. You cannot determine $\dim(W_1 \cap W_2)$ from dimensions alone without knowing $\dim(W_1 + W_2)$.                                                                       |
| 6   | **"The complement of a subspace is a subspace"**                                                                                                      | The **set complement** $V \setminus W$ is never a subspace: it does not contain $\mathbf{0}$ (since $\mathbf{0} \in W$) and is not closed under addition or scalar multiplication.                                                                                                                                                                                 | The **orthogonal complement** $W^\perp$ IS a subspace. Always say "orthogonal complement $W^\perp$", never just "complement". The orthogonal complement is the canonical linear-algebraic complement of $W$.                                              |
| 7   | **"All bases for $V$ have the same vectors"**                                                                                                         | Bases are not unique. Any set of $n$ linearly independent spanning vectors is a basis for an $n$-dimensional space. For $\mathbb{R}^2$, $\{(1,0),(0,1)\}$, $\{(1,1),(1,-1)\}$, and $\{(2,3),(1,2)\}$ are three distinct bases.                                                                                                                                     | All bases for $V$ have the same **number** of vectors (= $\dim(V)$). The actual vectors can be anything linearly independent and spanning. The choice of basis is a choice of coordinate system.                                                          |
| 8   | **"$\text{null}(AB) = \text{null}(B)$"**                                                                                                              | $\text{null}(B) \subseteq \text{null}(AB)$ always (if $B\mathbf{x} = \mathbf{0}$ then $AB\mathbf{x} = \mathbf{0}$). But $\text{null}(AB)$ can be strictly larger: if $A$ maps some $B\mathbf{x} \neq \mathbf{0}$ to zero ($B\mathbf{x} \in \text{null}(A)$), then $\mathbf{x} \in \text{null}(AB)$ but $\mathbf{x} \notin \text{null}(B)$.                         | $\text{null}(AB) \supseteq \text{null}(B)$. Equality holds iff $\text{null}(A) \cap \text{col}(B) = \{\mathbf{0}\}$, i.e., $A$ is injective on the column space of $B$.                                                                                   |
| 9   | **"Two subspaces with the same dimension are equal"**                                                                                                 | Dimension only describes the "size" of a subspace, not its orientation. The x-axis $\text{span}\{(1,0)\}$ and y-axis $\text{span}\{(0,1)\}$ in $\mathbb{R}^2$ both have dimension 1 but are completely different subspaces (they share only the origin).                                                                                                           | Equal dimension means **isomorphic** as abstract vector spaces. Actual equality as subsets requires showing every vector in one is in the other — or equivalently, that each spans the other.                                                             |
| 10  | **"Gram-Schmidt always produces a basis from a spanning set"**                                                                                        | Gram-Schmidt fails (division by zero) when it encounters a vector already in the span of the previously constructed orthonormal vectors. The intermediate vector $\mathbf{u}_j = \mathbf{v}_j - \sum_{i<j}\langle\mathbf{v}_j,\mathbf{q}_i\rangle\mathbf{q}_i$ becomes $\mathbf{0}$, which cannot be normalised.                                                   | Apply Gram-Schmidt **only to linearly independent vectors**. For a spanning set, first extract an independent subset via row reduction, then apply Gram-Schmidt. A zero intermediate vector signals linear dependence — discard that vector and continue. |
| 11  | **"The pivot columns of the RREF give a basis for the column space"**                                                                                 | Row operations change the column space. The RREF of $A$ has different columns than $A$ itself. The pivot positions (column indices) are correct, but the actual basis vectors must be taken from the **original** matrix $A$, not from the RREF.                                                                                                                   | Identify pivot positions from the RREF, but extract the corresponding columns from the **original** $A$. The pivot columns of $A$ (not its RREF) form a basis for $\text{col}(A)$.                                                                        |
| 12  | **"$A\mathbf{x} = \mathbf{0}$ has only the trivial solution implies $A$ is invertible"**                                                              | For $A \in \mathbb{R}^{m \times n}$: if $m \neq n$, $A$ cannot be invertible regardless. The trivial null space means $A$ is injective ($\text{rank}(A) = n$), but invertibility requires square and $\text{rank}(A) = n = m$.                                                                                                                                     | For square $A \in \mathbb{R}^{n \times n}$: $\text{null}(A) = \{\mathbf{0}\}$ iff $A$ is invertible. For rectangular $A$: $\text{null}(A) = \{\mathbf{0}\}$ means $A$ is injective (left-invertible) but not invertible.                                  |

---

## 15. Exercises

**Exercise 1: Verifying Vector Space Axioms**

For each of the following, determine whether it is a vector space with the given operations. If not, identify which axiom(s) fail. For each valid vector space, identify the zero vector and describe the additive inverses.

(a) $V = \mathbb{R}^2$, addition: $(u_1, u_2) + (v_1, v_2) = (u_1 + v_1,\ u_2 + v_2)$, scalar mult: $\alpha(u_1, u_2) = (\alpha u_1,\ u_2)$ — scalar multiplication only affects the first coordinate.

(b) $V = \mathbb{R}_{+} = \{x \in \mathbb{R} : x > 0\}$, addition defined as $u \oplus v = uv$ (multiplication of positive reals), scalar multiplication $\alpha \odot u = u^\alpha$.

(c) $V = \{(x, y, z) \in \mathbb{R}^3 : x + y + z = 1\}$ with the standard addition and scalar multiplication inherited from $\mathbb{R}^3$.

(d) $V = \{f : \mathbb{R} \to \mathbb{R} : f(0) = 0\}$ with standard pointwise addition $(f+g)(t) = f(t)+g(t)$ and scalar multiplication $(\alpha f)(t) = \alpha f(t)$.

(e) $V = \{f : \mathbb{R} \to \mathbb{R} : f(0) = 1\}$ with the same standard operations.

_Hints for (b):_ Verify all eight axioms carefully with the non-standard operations. The zero element of the vector space (if it exists) is the identity for $\oplus$, not the number 0. For (d) and (e), think about which evaluation constraint is compatible with the zero function.

---

**Exercise 2: Subspace Verification**

For each subset, determine whether it is a subspace of the given vector space. Prove it is a subspace (using the three-condition test) or find an explicit counterexample showing it is not.

(a) $W = \{(x, y, z) \in \mathbb{R}^3 : 2x - y + 3z = 0\}$ inside $\mathbb{R}^3$

(b) $W = \{(x, y) \in \mathbb{R}^2 : xy = 0\}$ (the two coordinate axes together) inside $\mathbb{R}^2$

(c) $W = \{A \in \mathbb{R}^{2 \times 2} : \text{tr}(A) = 0\}$ (traceless $2 \times 2$ matrices) inside $\mathbb{R}^{2 \times 2}$

(d) $W = \{p \in \mathcal{P}_3 : p(1) = 0\}$ (polynomials of degree $\leq 3$ that vanish at $t = 1$) inside $\mathcal{P}_3$

(e) $W = \{(x, y, z) \in \mathbb{R}^3 : x^2 + y^2 = z^2\}$ (double cone) inside $\mathbb{R}^3$

For each valid subspace: find a basis and state the dimension.

---

**Exercise 3: The Four Fundamental Subspaces**

Let $A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 0 & 1 & 3 \\ 1 & 2 & 1 & 4 \end{pmatrix}$.

(a) Row-reduce $A$ to RREF. Identify the pivot columns and free columns.

(b) Find a basis for $\text{null}(A)$. Verify the dimension via the Rank-Nullity Theorem.

(c) Find a basis for $\text{col}(A)$ using the pivot columns of the **original** $A$ (not the RREF). State the dimension.

(d) Find a basis for $\text{row}(A)$ using the non-zero rows of the RREF. State the dimension.

(e) Find a basis for $\text{null}(A^\top)$ by row-reducing $A^\top$ and finding its null space. Alternatively, argue from the dimension formula.

(f) Verify the dimension counts: $\dim(\text{col}(A)) + \dim(\text{null}(A^\top)) = 3$ and $\dim(\text{row}(A)) + \dim(\text{null}(A)) = 4$.

(g) Verify the orthogonality $\text{row}(A) \perp \text{null}(A)$: compute the dot product of every basis vector of $\text{row}(A)$ with every basis vector of $\text{null}(A)$ and confirm it equals zero.

---

**Exercise 4: Span, Independence, and Basis in $\mathbb{R}^4$**

Let $\mathbf{v}_1 = (1,2,0,1)^\top$, $\mathbf{v}_2 = (0,1,1,2)^\top$, $\mathbf{v}_3 = (1,3,1,3)^\top$, $\mathbf{v}_4 = (2,1,-1,-1)^\top$.

(a) Form the matrix $A = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \mathbf{v}_3 \mid \mathbf{v}_4]$ and row-reduce. Are $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\}$ linearly independent?

(b) What is $\dim(\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\})$?

(c) Identify a subset $\{\mathbf{v}_{i_1}, \mathbf{v}_{i_2}, \ldots\}$ of the given vectors that forms a basis for $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\}$.

(d) Express any dependent vectors as explicit linear combinations of your basis vectors.

(e) Is $\mathbf{w} = (3, 5, 1, 4)^\top$ in $\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3, \mathbf{v}_4\}$? If yes, find coordinates $(\alpha_1, \alpha_2, \ldots)$ such that $\mathbf{w} = \alpha_1 \mathbf{v}_{i_1} + \alpha_2 \mathbf{v}_{i_2} + \cdots$. If no, explain why.

---

**Exercise 5: Gram-Schmidt and Orthogonal Projection**

Work in $\mathbb{R}^3$ with the standard inner product.

(a) Starting from $\mathbf{v}_1 = (1,1,0)^\top$, $\mathbf{v}_2 = (1,0,1)^\top$, $\mathbf{v}_3 = (0,1,1)^\top$, verify that these three vectors are linearly independent (compute the determinant of the matrix they form).

(b) Apply Gram-Schmidt to produce an orthonormal basis $\{\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3\}$ for $\mathbb{R}^3$.

(c) Verify your result: check that $\|\mathbf{q}_i\| = 1$ for each $i$ and $\langle \mathbf{q}_i, \mathbf{q}_j \rangle = 0$ for $i \neq j$.

(d) Express $\mathbf{w} = (1, 2, 3)^\top$ in your orthonormal basis: find $\alpha_i = \langle \mathbf{w}, \mathbf{q}_i \rangle$ for each $i$. Verify that $\mathbf{w} = \sum_i \alpha_i \mathbf{q}_i$.

(e) Construct the orthogonal projection matrix $P$ onto $W = \text{span}\{\mathbf{v}_1, \mathbf{v}_2\} = \text{span}\{\mathbf{q}_1, \mathbf{q}_2\}$ using $P = Q_2 Q_2^\top$ where $Q_2 = [\mathbf{q}_1 \mid \mathbf{q}_2]$. Verify: (i) $P^2 = P$, (ii) $P^\top = P$, (iii) $\text{rank}(P) = 2$. Compute $P\mathbf{w}$ and the residual $\mathbf{r} = \mathbf{w} - P\mathbf{w}$. Verify that $\mathbf{r} \perp \mathbf{v}_1$ and $\mathbf{r} \perp \mathbf{v}_2$.

---

**Exercise 6: Subspace Operations and Dimension Formula**

(a) In $\mathbb{R}^3$, let $W_1 = \text{span}\{(1,0,0)^\top, (0,1,0)^\top\}$ (the $xy$-plane) and $W_2 = \text{span}\{(1,1,0)^\top, (0,0,1)^\top\}$. Find a basis for $W_1 + W_2$. Is $W_1 + W_2 = \mathbb{R}^3$? Find $W_1 \cap W_2$ and verify the Grassmann formula: $\dim(W_1 + W_2) = \dim(W_1) + \dim(W_2) - \dim(W_1 \cap W_2)$.

(b) Is $\mathbb{R}^3 = W_1 \oplus W_2$ (direct sum)? If not, find a subspace $W_3 \subseteq \mathbb{R}^3$ such that $\mathbb{R}^3 = W_1 \oplus W_3$.

(c) In $\mathbb{R}^4$, let $W_1 = \{\mathbf{x} : x_1 + x_2 = 0\}$ and $W_2 = \{\mathbf{x} : x_3 + x_4 = 0\}$. Find $\dim(W_1)$, $\dim(W_2)$, $\dim(W_1 \cap W_2)$, and $\dim(W_1 + W_2)$. Verify the Grassmann formula.

(d) For $W_1 \cap W_2$ from part (c): find an explicit basis.

---

**Exercise 7: Change of Basis and Coordinates**

In $\mathbb{R}^2$, let $\mathcal{B} = \{\mathbf{b}_1, \mathbf{b}_2\}$ with $\mathbf{b}_1 = (1,2)^\top$ and $\mathbf{b}_2 = (1,-1)^\top$, and let $\mathcal{C} = \{\mathbf{c}_1, \mathbf{c}_2\}$ with $\mathbf{c}_1 = (2,1)^\top$ and $\mathbf{c}_2 = (-1,1)^\top$.

(a) Verify that both $\mathcal{B}$ and $\mathcal{C}$ are bases for $\mathbb{R}^2$ (compute the determinants of the corresponding matrices).

(b) Find the change-of-basis matrix $P_{\mathcal{B} \to \mathcal{C}}$: express each $\mathbf{b}_i$ as a linear combination of $\mathbf{c}_1, \mathbf{c}_2$, and use these as the columns of $P$.

(c) For the vector $\mathbf{v}$ with $[\mathbf{v}]_{\mathcal{B}} = (3, -1)^\top$ (coordinates $3\mathbf{b}_1 - \mathbf{b}_2$ in basis $\mathcal{B}$), compute $[\mathbf{v}]_{\mathcal{C}} = P_{\mathcal{B} \to \mathcal{C}} [\mathbf{v}]_{\mathcal{B}}$.

(d) Compute $\mathbf{v}$ explicitly in the standard basis. Then verify your answer to (c) by directly expressing $\mathbf{v}$ as a linear combination of $\mathbf{c}_1$ and $\mathbf{c}_2$.

(e) If a linear map $T: \mathbb{R}^2 \to \mathbb{R}^2$ has matrix $M = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$ in the standard basis, find its matrix representation in basis $\mathcal{B}$.

---

**Exercise 8: AI Application — Subspace Analysis of a Weight Matrix**

Let $W = \begin{pmatrix} 3 & 1 & 2 \\ 1 & 2 & 1 \\ 2 & 1 & 3 \end{pmatrix}$.

(a) Compute $\det(W)$ and $\text{rank}(W)$. Is $W$ full rank?

(b) Find a basis for $\text{null}(W)$ and $\text{col}(W)$. Verify $\dim(\text{null}(W)) + \dim(\text{col}(W)) = 3$.

(c) A LoRA adapter has rank $r = 1$ with update $\Delta W = \mathbf{b}\mathbf{a}^\top$ where $\mathbf{b} = (1, 0, 1)^\top$ and $\mathbf{a} = (1, 1, 0)^\top$. Explicitly compute $\Delta W$. What is $\text{col}(\Delta W)$? What is $\text{rank}(\Delta W)$?

(d) Consider the updated weight $W' = W + \Delta W$. Without computing $W'$ explicitly, state the upper and lower bounds on $\text{rank}(W')$. Under what condition on the relationship between $\text{col}(\Delta W)$ and $\text{null}(W^\top)$ (i.e., the left null space of $W$) would $\text{rank}(W') = \text{rank}(W) + 1$? Under what condition would $\text{rank}(W') = \text{rank}(W)$?

(e) Now compute $W'$ explicitly. Verify your predictions from (d) by computing $\text{rank}(W')$.

---

**Exercise 9 (Challenge): The Superposition Geometry**

This exercise develops the geometry of the superposition hypothesis.

(a) In $\mathbb{R}^2$ (a 2-dimensional embedding space), suppose we want to store $F = 3$ features as unit vectors with maximum pairwise orthogonality. Show that it is impossible for all three feature vectors to be pairwise orthogonal. (Hint: three pairwise orthogonal unit vectors in $\mathbb{R}^2$ cannot exist.)

(b) Instead, place three unit vectors at angles $0°$, $120°$, $240°$ from the positive $x$-axis: $\mathbf{f}_1 = (1,0)^\top$, $\mathbf{f}_2 = (-1/2, \sqrt{3}/2)^\top$, $\mathbf{f}_3 = (-1/2, -\sqrt{3}/2)^\top$. Compute all pairwise inner products $\langle \mathbf{f}_i, \mathbf{f}_j \rangle$ for $i \neq j$. What is the interference level?

(c) The "reconstruction loss" for feature $i$ when all features have activation $x_i \in \{0,1\}$ is: the residual after reading off the $i$-th feature and subtracting the contribution of the feature direction. Specifically, if the residual stream stores $\mathbf{h} = \sum_j x_j \mathbf{f}_j$, and we estimate $\hat{x}_i = \langle \mathbf{h}, \mathbf{f}_i \rangle$, show that $\hat{x}_i = x_i + \sum_{j \neq i} x_j \langle \mathbf{f}_j, \mathbf{f}_i \rangle$. The error $\hat{x}_i - x_i$ is the interference from other features.

(d) Compute the expected interference for the configuration in (b) assuming each $x_j \in \{0, 1\}$ independently with probability $p$ of being active. What does the interference approach as $p \to 0$?

---

**Exercise 10 (Challenge): Krylov Subspaces**

Let $A = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$ and $\mathbf{b} = (1, 1)^\top$.

(a) Compute $\mathcal{K}_1(A, \mathbf{b}) = \text{span}\{\mathbf{b}\}$ and $\mathcal{K}_2(A, \mathbf{b}) = \text{span}\{\mathbf{b}, A\mathbf{b}\}$.

(b) Does $\mathcal{K}_2(A, \mathbf{b}) = \mathbb{R}^2$? (Check whether $\mathbf{b}$ and $A\mathbf{b}$ are linearly independent.)

(c) Apply one step of the Lanczos algorithm to produce an orthonormal basis for $\mathcal{K}_2(A, \mathbf{b})$ (Gram-Schmidt applied to $\{\mathbf{b}, A\mathbf{b}\}$).

(d) In this orthonormal basis $\{\mathbf{q}_1, \mathbf{q}_2\}$, compute the tridiagonal matrix $T = Q^\top A Q$ where $Q = [\mathbf{q}_1 \mid \mathbf{q}_2]$. Verify that $T$ is symmetric and (nearly) tridiagonal. The eigenvalues of $T$ approximate the eigenvalues of $A$ — verify this by comparing with the actual eigenvalues of $A$.

---

## 16. Why This Matters for AI (2026 Perspective)

| Aspect                                       | Impact                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Residual stream as shared vector space**   | The residual stream $\mathbf{x} \in \mathbb{R}^d$ in a transformer is the central shared vector space. Every attention head, MLP layer, and positional encoding adds vectors to this space via the residual connection. All components communicate through this one $d$-dimensional space — nothing else is shared. Understanding the subspace decomposition of $\mathbb{R}^d$ (which subspaces are written by which components, which are read by which) is literally the same as understanding how information flows through a transformer. There is no higher-level description.                                                                                                                |
| **LoRA rank = subspace dimension**           | LoRA's entire design is a subspace constraint. The update $\Delta W = BA^\top$ is a rank-$r$ matrix, living in an $r(m+n)/(mn)$-fraction of the full parameter space. Choosing $r$ is choosing the dimension of the subspace to search in. Too small: the subspace doesn't contain the optimal update direction, and performance suffers. Too large: you are searching a subspace larger than necessary, wasting parameters and potentially overfitting. The right $r$ is the intrinsic dimension of the fine-tuning task — a subspace dimension.                                                                                                                                                  |
| **Superposition and polysemanticity**        | The superposition hypothesis says LLMs represent more features than their embedding dimension allows. Since more than $d$ linearly independent directions cannot exist in $\mathbb{R}^d$, features beyond $d$ must share dimensions through non-orthogonal superposition. Each neuron becomes polysemantic — it responds to multiple features. This limits interpretability (you cannot read off features from individual dimensions) and causes interference between features. Solving superposition is one of the central unsolved problems in mechanistic interpretability, and the solution must be phrased in the language of subspace geometry.                                              |
| **MLA and KV subspace compression**          | DeepSeek-V3's Multi-head Latent Attention compresses KV projections to a rank-$r$ subspace with $r \ll d$. The KV cache stores only the $r$-dimensional compressed representation; at inference, it is decompressed back to $\mathbb{R}^d$. The rank $r$ is the architectural bottleneck that enables a $5.75\times$ reduction in KV cache memory. The subspace dimension $r$ is the design variable that trades memory against expressiveness. This is subspace thinking at the architecture level: the architecture is designed around a low-dimensional subspace constraint.                                                                                                                    |
| **Mechanistic interpretability circuits**    | Every circuit found in mechanistic interpretability is a composition of subspace operations. A "previous token head" reads from a specific subspace of the residual stream (via its $W_Q$, $W_K$ row spaces) and writes to a specific subspace (via its $W_O$ column space). An "induction head" communicates with the previous token head through a shared subspace. Superposition of features happens in specific subspaces. The "language" of mechanistic interpretability is entirely the language of subspaces: read, write, rotate, project, in $\mathbb{R}^d$.                                                                                                                              |
| **Representation collapse prevention**       | In self-supervised learning, representation collapse means all embeddings converge to a low-dimensional (or 0-dimensional) subspace — the network learns to output the same or similar vectors for all inputs. VICReg, Barlow Twins, and BYOL all prevent collapse by adding losses that penalise low-dimensional representations. VICReg's variance loss requires that the covariance matrix of batch embeddings has high trace (high average variance), which prevents the embeddings from collapsing to a subspace. Collapse = subspace dimension $\to 0$; the loss explicitly maximises subspace dimension.                                                                                    |
| **Implicit bias and minimum-norm solutions** | Gradient descent on overparameterised linear models (and, empirically, on large neural networks) converges to minimum-norm solutions. The minimum-norm solution lies in the row space of the data matrix — it is the unique solution in the subspace $\text{row}(X)$, the orthogonal complement of the null space. The implicit bias of gradient descent is a subspace selection: it selects the solution in $\text{row}(X)$ rather than any other coset representative. This subspace selection is what enables generalisation in overparameterised models.                                                                                                                                       |
| **Orthogonality and head diversity**         | If attention heads write to mutually orthogonal subspaces of the residual stream, they do not interfere with each other — each head has an independent information channel. Head redundancy is equivalent to subspace overlap: if head $A$'s column space is a subspace of head $B$'s column space, head $A$ is redundant. Pruning heads that write to subspaces already covered by remaining heads preserves expressiveness while reducing computation. Orthogonality between head subspaces is the precise mathematical criterion for head independence.                                                                                                                                         |
| **Function space universality**              | The universal approximation theorem says that neural networks with nonlinear activations can approximate any continuous function to arbitrary accuracy. In subspace language: the set of functions representable by a sufficiently large network is dense in $C([0,1]^n)$ — it is not a subspace (the network family is non-linear), but its closure is the entire function space. The depth and nonlinearity of the architecture determine which function-space "directions" are accessible. Width and depth determine the "dimensionality" of the accessible function subspace.                                                                                                                  |
| **Second-order optimisation**                | The Hessian $H = \nabla^2 \mathcal{L}(\theta)$ of the training loss is a $p \times p$ matrix with most of its spectral mass concentrated in a low-dimensional subspace (the "bulk" subspace where curvature is large). The remaining $p - k$ directions have near-zero curvature (flat directions). K-FAC, Shampoo, and SOAP all identify and exploit this curvature subspace by approximating the inverse Hessian in the curved subspace and using the identity in the flat directions. Natural gradient preconditions updates by the inverse Fisher (a positive semidefinite matrix): it aligns updates with the geometry of the curved subspace and suppresses movement in the flat directions. |

---

## Conceptual Bridge

Vector spaces and subspaces are the geometry underlying all of linear algebra. Every concept — linear independence, span, basis, dimension, rank, orthogonality, projections, eigenvalues — is ultimately a statement about the structure of subspaces. The eight axioms that define a vector space are simple; the richness emerges from the subspace hierarchy they generate.

The abstract framework is what makes the theory universal. The same theorems that govern arrows in a plane govern polynomials, matrices, functions, and probability distributions. Verifying the eight axioms once grants access to a century of results — for free, without reproof. This universality is not just mathematical elegance; it is practical power. When you recognise that the gradient vectors during training form a set of vectors in $\mathbb{R}^p$ and that $\mathbb{R}^p$ is a vector space, you immediately know that all of linear algebra applies: span, independence, subspaces, projections, and dimension all become tools for understanding training dynamics.

For AI in 2026, subspaces are not abstract. They are the architectural primitives of transformers (the residual stream, the attention subspace, the MLP subspace), the design variables of efficient fine-tuning (LoRA rank = subspace dimension), the diagnostic language of interpretability (which subspace does this head write to?), and the theoretical foundation of generalisation (gradient updates live in a low-dimensional subspace; the implicit bias selects the minimum-norm solution in the row space).

```text
WHERE THIS MODULE SITS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════

  Sets and Logic → Functions → Summation
          ↓
  Proof Techniques
          ↓
  Vectors and Spaces (geometry: what vectors are)
          ↓
  Matrix Operations (computation: how to multiply matrices)
          ↓
  Systems of Equations (solving: Gaussian elimination)
          ↓
  Determinants (volume: the signed volume of a parallelepiped)
          ↓
  Matrix Rank (dimensionality: the rank-nullity connection)
          ↓
  Vector Spaces and Subspaces (structure) ← THIS MODULE
  ════════════════════════════════════════
  The payoff: everything from earlier modules
  is now understood geometrically as a statement
  about subspaces. Rank = dimension of col space.
  Null space = kernel. Solutions of Ax=b live in
  a coset of null(A). Determinant = 0 iff the
  column space is a proper subspace of ℝⁿ.
          ↓
  Eigenvalues and Decompositions (spectrum of a linear map)
  — invariant subspaces of T: revealed through its eigenvalues
  — eigenvector: 1D invariant subspace
  — spectral theorem: orthogonal direct sum of eigenspaces
  — SVD: complete subspace decomposition of any matrix
          ↓
  Probability and Information Theory
  — probability distributions as vectors in function space
  — KL divergence as a geometry on the simplex
          ↓
  Calculus and Optimisation
  — gradient lives in ℝᵖ, a vector space
  — Hessian's eigenspaces = directions of curvature
  — natural gradient uses curved subspace geometry
          ↓
          LLM Mathematics
```

**What comes next: Eigenvalues, Eigenvectors, and Matrix Decompositions.**

The next module reveals the _internal_ subspace structure of a linear map — the invariant subspaces that a matrix preserves, revealed through its eigenvalues and eigenvectors. An eigenvector spans a 1-dimensional invariant subspace; the spectral theorem decomposes any symmetric matrix into orthogonal 1-dimensional invariant subspaces; the SVD extends this to the complete subspace structure of any rectangular matrix. The four fundamental subspaces are decomposed even further — into singular subspaces aligned with singular values. All of this is the continuation of the subspace story begun in this module.

---

[← Back to Linear Algebra Basics](../README.md) | [Next: Eigenvalues and Decompositions →](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)
