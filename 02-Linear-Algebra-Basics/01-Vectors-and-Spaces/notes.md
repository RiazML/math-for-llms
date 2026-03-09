[Previous: Mathematical Foundations](../../01-Mathematical-Foundations/06-Proof-Techniques/notes.md) | [Home](../../README.md) | [Next: Matrix Operations](../02-Matrix-Operations/notes.md)

---

# Vectors and Spaces

> _"Deep learning runs on arrays of floating-point numbers, but the right mental model is not 'array processing.' It is geometry inside vector spaces."_

## Overview

Vectors and spaces are the basic language of modern machine learning. An embedding is a vector. A batch of activations is a matrix whose rows or columns are vectors. A transformer's attention mechanism compares vectors with inner products, combines vectors with weighted sums, and moves information through learned subspaces. Optimization updates a parameter vector in a very high-dimensional space. Even when the underlying object is not a list of numbers, the mathematics usually places it inside a vector space: matrices, polynomials, functions, sequences, kernels, and distributions all become easier to reason about once their linear structure is exposed.

This chapter is deliberately broader than a minimal first course treatment. It starts with concrete vectors in $\mathbb{R}^n$, then moves to abstract vector spaces, norms, inner products, projections, convex geometry, high-dimensional effects, linear maps, and the AI-specific spaces that appear in transformers and large language models. The goal is not just to define terms, but to build the geometric intuition that later chapters on matrices, orthogonality, SVD, optimization, embeddings, and attention depend on.

Where possible, the discussion connects classical linear algebra to modern AI research. Word vectors made semantic geometry operational in NLP (Mikolov et al., 2013). The Transformer made dot-product geometry the core computational primitive of sequence modeling (Vaswani et al., 2017). Low-rank adaptation methods such as LoRA rely directly on rank, subspace, and projection ideas (Hu et al., 2022). Neural tangent kernel theory studies training as dynamics in function space (Jacot et al., 2018). Understanding vectors and spaces is therefore not background decoration. It is the medium in which AI computation happens.

## Prerequisites

- Basic algebra and coordinate geometry
- Familiarity with sets and functions
- Comfort with summation notation
- Basic proof ideas (especially direct proof and contradiction)

## Learning Objectives

After completing this chapter, you should be able to:

- Explain vectors geometrically, algebraically, and abstractly
- Work fluently with vector addition, scalar multiplication, span, and linear independence
- Verify whether a set is a vector space or a subspace
- Find bases, compute dimensions, and interpret coordinate systems
- Use norms, inner products, orthogonality, and projections
- Recognize affine and convex structure in optimization and probabilistic modeling
- Understand why high-dimensional geometry differs sharply from low-dimensional intuition
- Interpret kernels, images, null spaces, and rank-nullity for linear maps
- Connect embedding spaces, attention subspaces, parameter space, and function space to concrete AI systems
- Reason about regularization geometrically rather than as a bag of tricks

---

## Table of Contents

- [Vectors and Spaces](#vectors-and-spaces)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Are Vectors and Spaces?](#11-what-are-vectors-and-spaces)
    - [1.2 Why Vectors and Spaces Are Central to AI](#12-why-vectors-and-spaces-are-central-to-ai)
    - [1.3 Geometric, Algebraic, and Abstract Views](#13-geometric-algebraic-and-abstract-views)
    - [1.4 Where Vectors and Spaces Appear in AI](#14-where-vectors-and-spaces-appear-in-ai)
    - [1.5 The Hierarchy of Structure](#15-the-hierarchy-of-structure)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Vectors in R^n](#2-vectors-in-rn)
    - [2.1 Definition and Notation](#21-definition-and-notation)
    - [2.2 Vector Addition](#22-vector-addition)
    - [2.3 Scalar Multiplication](#23-scalar-multiplication)
    - [2.4 Linear Combinations](#24-linear-combinations)
    - [2.5 Linear Independence](#25-linear-independence)
    - [2.6 Span](#26-span)
  - [3. Abstract Vector Spaces](#3-abstract-vector-spaces)
    - [3.1 The Vector Space Axioms](#31-the-vector-space-axioms)
    - [3.2 Consequences of the Axioms](#32-consequences-of-the-axioms)
    - [3.3 Examples of Vector Spaces](#33-examples-of-vector-spaces)
    - [3.4 Non-Examples](#34-non-examples)
    - [3.5 Subspaces](#35-subspaces)
  - [4. Basis and Dimension](#4-basis-and-dimension)
    - [4.1 Basis Definition](#41-basis-definition)
    - [4.2 Standard Basis of R^n](#42-standard-basis-of-rn)
    - [4.3 Dimension](#43-dimension)
    - [4.4 Dimension and Subspaces](#44-dimension-and-subspaces)
    - [4.5 Coordinates and Change of Basis](#45-coordinates-and-change-of-basis)
    - [4.6 Rank-Nullity Theorem](#46-rank-nullity-theorem)
  - [5. Norms and Metric Spaces](#5-norms-and-metric-spaces)
    - [5.1 Norms on Vector Spaces](#51-norms-on-vector-spaces)
    - [5.2 The p-Norms](#52-the-p-norms)
    - [5.3 Norm Equivalence](#53-norm-equivalence)
    - [5.4 Matrix Norms](#54-matrix-norms)
    - [5.5 Metric Spaces](#55-metric-spaces)
    - [5.6 Convergence in Normed Spaces](#56-convergence-in-normed-spaces)
  - [6. Inner Product Spaces](#6-inner-product-spaces)
    - [6.1 Inner Products](#61-inner-products)
    - [6.2 Standard Inner Product on R^n](#62-standard-inner-product-on-rn)
    - [6.3 Geometric Interpretation](#63-geometric-interpretation)
    - [6.4 Cauchy-Schwarz Inequality](#64-cauchy-schwarz-inequality)
    - [6.5 Orthogonality](#65-orthogonality)
    - [6.6 Gram-Schmidt Orthogonalization](#66-gram-schmidt-orthogonalization)
    - [6.7 Orthogonal Complement](#67-orthogonal-complement)
    - [6.8 Hilbert Spaces](#68-hilbert-spaces)
  - [7. Orthogonal Projections](#7-orthogonal-projections)
    - [7.1 Projection onto a Subspace](#71-projection-onto-a-subspace)
    - [7.2 Projection Matrix Properties](#72-projection-matrix-properties)
    - [7.3 Projection onto Column Space](#73-projection-onto-column-space)
    - [7.4 Gram Matrix](#74-gram-matrix)
    - [7.5 Best Approximation and Least Squares](#75-best-approximation-and-least-squares)
  - [8. Affine Subspaces and Convexity](#8-affine-subspaces-and-convexity)
    - [8.1 Affine Subspaces](#81-affine-subspaces)
    - [8.2 Convex Sets](#82-convex-sets)
    - [8.3 Hyperplanes and Half-Spaces](#83-hyperplanes-and-half-spaces)
    - [8.4 Convex Functions and Sublevel Sets](#84-convex-functions-and-sublevel-sets)
  - [9. High-Dimensional Geometry](#9-high-dimensional-geometry)
    - [9.1 The Curse of Dimensionality](#91-the-curse-of-dimensionality)
    - [9.2 Concentration of Measure](#92-concentration-of-measure)
    - [9.3 Johnson-Lindenstrauss Lemma](#93-johnson-lindenstrauss-lemma)
    - [9.4 Random Vectors in High Dimensions](#94-random-vectors-in-high-dimensions)
    - [9.5 Geometry of Embedding Spaces](#95-geometry-of-embedding-spaces)
    - [9.6 Softmax Temperature and Simplex Geometry](#96-softmax-temperature-and-simplex-geometry)
  - [10. Linear Maps Between Spaces](#10-linear-maps-between-spaces)
    - [10.1 Definition and Properties](#101-definition-and-properties)
    - [10.2 Kernel and Image](#102-kernel-and-image)
    - [10.3 The Four Fundamental Subspaces](#103-the-four-fundamental-subspaces)
    - [10.4 Isomorphisms](#104-isomorphisms)
    - [10.5 Dual Spaces](#105-dual-spaces)
  - [11. Special Vector Spaces in AI](#11-special-vector-spaces-in-ai)
    - [11.1 Embedding Space Geometry](#111-embedding-space-geometry)
    - [11.2 Representation Space Through Layers](#112-representation-space-through-layers)
    - [11.3 Attention Subspaces](#113-attention-subspaces)
    - [11.4 The Probability Simplex](#114-the-probability-simplex)
    - [11.5 Parameter Space](#115-parameter-space)
    - [11.6 Function Spaces and the Neural Tangent Kernel](#116-function-spaces-and-the-neural-tangent-kernel)
  - [12. Direct Sums and Quotient Spaces](#12-direct-sums-and-quotient-spaces)
    - [12.1 Direct Sum](#121-direct-sum)
    - [12.2 Decomposing R^n](#122-decomposing-rn)
    - [12.3 Quotient Spaces](#123-quotient-spaces)
    - [12.4 Tensor Product](#124-tensor-product)
  - [13. Norms, Regularization, and Geometry](#13-norms-regularization-and-geometry)
    - [13.1 L2 Regularization as a Geometric Constraint](#131-l2-regularization-as-a-geometric-constraint)
    - [13.2 L1 Regularization and Sparsity](#132-l1-regularization-and-sparsity)
    - [13.3 Spectral Regularization](#133-spectral-regularization)
    - [13.4 The Geometry of Gradient Descent](#134-the-geometry-of-gradient-descent)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI](#16-why-this-matters-for-ai)
  - [17. Conceptual Bridge](#17-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Are Vectors and Spaces?

A vector is first encountered as an arrow with magnitude and direction, but that picture is only the beginning. In modern linear algebra, a vector is any element of a set that supports two operations:

1. addition of vectors
2. multiplication of a vector by a scalar

If those operations satisfy the right axioms, the set is called a vector space. The power of the abstraction is that the "things" inside the space do not need to be arrows in physical space. They may be coordinate tuples, polynomials, matrices, functions, sequences, or even gradient fields.

Concrete examples help fix the idea:

- $(3, -1, 2) \in \mathbb{R}^3$ is a vector
- a polynomial such as $1 + 2x - 5x^2$ is a vector in a polynomial space
- a matrix in $\mathbb{R}^{m \times n}$ is a vector in a matrix space
- a continuous function $f:[0,1] \to \mathbb{R}$ is a vector in a function space

The unifying idea is linear structure. If you can sensibly add two objects of the same kind and scale the result by numbers, then linear algebra is likely available.

### 1.2 Why Vectors and Spaces Are Central to AI

Nearly every important object in deep learning is a vector or lives in a vector space:

- token embeddings are vectors in $\mathbb{R}^d$
- hidden states are vectors in a residual stream
- query, key, and value representations are vectors in learned subspaces
- parameter sets are points in a very large Euclidean space
- gradients are vectors in the same parameter space
- logits are vectors in vocabulary space
- probability outputs live in the probability simplex, which sits inside a vector space but is not itself one

The Transformer made this especially explicit. Attention compares a query vector $q$ with key vectors $k_j$ using dot products, then produces an output as a weighted sum of value vectors $v_j$:

$$
\mathrm{Attention}(q, K, V) = \sum_{j=1}^{n} \alpha_j v_j,
\qquad
\alpha_j = \frac{\exp(q^\top k_j / \sqrt{d_k})}{\sum_{\ell=1}^{n}\exp(q^\top k_\ell / \sqrt{d_k})}.
$$

That is pure vector-space computation: inner products, scaling, exponentiation of scores, and linear combination. The semantic geometry of embeddings became operational in NLP through word2vec (Mikolov et al., 2013), and the geometry of projected subspaces became central to model architecture through self-attention (Vaswani et al., 2017). In large open models such as Llama 3, parameter space itself has hundreds of billions of coordinates at the frontier scale (Dubey et al., 2024), so geometric reasoning is not optional; it is the only way to think clearly about what the model can represent.

### 1.3 Geometric, Algebraic, and Abstract Views

There are three complementary ways to think about vectors.

| View | Core idea | Strength | Typical AI use |
| --- | --- | --- | --- |
| Geometric | A vector is an arrow with length and direction | Builds intuition for angle, distance, projection | Similarity, attention, clustering |
| Algebraic | A vector is an ordered tuple of numbers | Easy to compute with componentwise rules | Arrays, tensors, embeddings, gradients |
| Abstract | A vector is any element of a vector space | Generalizes linear algebra beyond coordinates | Function spaces, kernels, NTK, RKHS |

The geometric view tells you why cosine similarity matters. The algebraic view tells you how to implement it. The abstract view tells you why the same mathematics reappears for functions, distributions, and operators.

```text
One idea, three views

Geometric:             Algebraic:              Abstract:

    y                     [v1, v2, ... , vn]      v in V
    ^                                              where V has
    |   / v                                         vector addition
    |  /                                            and scalar multiplication
    | /
----+--------------> x
```

### 1.4 Where Vectors and Spaces Appear in AI

The flow through a language model can be read as a sequence of moves between vector spaces:

```text
Token IDs
  -> embedding vectors in R^d
  -> sequence matrix X in R^(n x d)
  -> projected query/key/value spaces
  -> attention outputs as weighted sums
  -> feed-forward maps R^d -> R^d_ff -> R^d
  -> logits in R^|V|
  -> probabilities in the simplex Delta^(|V|-1)
```

Even when the last object is not a vector space, it usually sits inside one. The simplex of probability vectors is an affine slice of $\mathbb{R}^{|V|}$ with nonnegativity constraints. That pattern is common in AI: model objects often live in structured subsets of ambient vector spaces.

### 1.5 The Hierarchy of Structure

A useful way to organize the subject is by how much extra structure is available.

```text
set
  -> abelian group (addition)
  -> vector space (addition + scalar multiplication)
     -> normed space (size)
        -> Banach space (complete normed space)
     -> inner product space (angles and lengths)
        -> Hilbert space (complete inner product space)
```

This branching picture is more accurate than a single chain. A normed space need not come from an inner product. An inner product space automatically induces a norm, and a Hilbert space is therefore also a Banach space, but not every Banach space is Hilbert. Most day-to-day ML uses finite-dimensional Euclidean spaces, where these distinctions collapse because all norms are equivalent and completeness is automatic. In theoretical ML and functional analysis, they matter a great deal.

### 1.6 Historical Timeline

The language of vectors and spaces was assembled gradually.

| Period | Figure or development | Importance |
| --- | --- | --- |
| Ancient mechanics | Directed quantities in geometry and physics | Proto-vector intuition |
| 17th century | Galileo and Newton | Composition of velocity and force |
| 1843-1844 | Hamilton and Grassmann | Algebraic extension beyond ordinary 3D geometry |
| Late 19th century | Peano and axiomatic linear algebra | Abstract vector space formulation |
| Early 20th century | Hilbert | Infinite-dimensional inner product spaces |
| 1920s-1930s | Banach and functional analysis | Normed complete spaces and operator theory |
| 20th century computing | Numerical linear algebra | Matrix computation becomes practical at scale |
| 2013 | word2vec | Semantic geometry becomes an engineering object |
| 2017 | Transformer | Dot-product geometry becomes core architecture |
| 2020s | Foundation models | Representation geometry becomes a first-class research topic |

The modern lesson is simple: vectors started as geometry, became algebra, and now serve as the organizing language of computation.

---

## 2. Vectors in R^n

### 2.1 Definition and Notation

A vector in $\mathbb{R}^n$ is an ordered $n$-tuple of real numbers:

$$
\mathbf{v} = (v_1, v_2, \ldots, v_n) \in \mathbb{R}^n.
$$

In linear algebra, the default convention is usually the column vector:

$$
\mathbf{v} =
\begin{pmatrix}
v_1 \\
v_2 \\
\vdots \\
v_n
\end{pmatrix}.
$$

Key notation:

- $v_i$ is the $i$-th component or coordinate
- $\mathbf{0} = (0,\ldots,0)$ is the zero vector
- $\mathbf{e}_i$ is the $i$-th standard basis vector, with a 1 in position $i$ and 0 elsewhere
- $\mathbf{v}^\top$ denotes the transpose, turning a column into a row

Every vector in $\mathbb{R}^n$ can be written as

$$
\mathbf{v} = v_1 \mathbf{e}_1 + v_2 \mathbf{e}_2 + \cdots + v_n \mathbf{e}_n.
$$

That formula is already telling you something profound: coordinates are not the vector itself; they are the coefficients of the vector relative to a chosen basis.

### 2.2 Vector Addition

Vectors in $\mathbb{R}^n$ add componentwise:

$$
(\mathbf{u} + \mathbf{v})_i = u_i + v_i.
$$

So if

$$
\mathbf{u} =
\begin{pmatrix}
1 \\ -2 \\ 5
\end{pmatrix},
\qquad
\mathbf{v} =
\begin{pmatrix}
3 \\ 4 \\ -1
\end{pmatrix},
$$

then

$$
\mathbf{u} + \mathbf{v} =
\begin{pmatrix}
4 \\ 2 \\ 4
\end{pmatrix}.
$$

Vector addition satisfies the familiar algebraic laws:

- commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- associativity: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- identity: $\mathbf{v} + \mathbf{0} = \mathbf{v}$
- inverse: $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

Geometrically, addition follows the parallelogram law. Place the tail of one vector at the head of another; the resulting diagonal is the sum. That picture becomes important later when we interpret residual connections as additive updates to a representation vector.

```text
Head-to-tail view of vector addition

O ----u----> A ----v----> B
O ----------------------> B
            u + v
```

### 2.3 Scalar Multiplication

If $\alpha \in \mathbb{R}$ and $\mathbf{v} \in \mathbb{R}^n$, then scalar multiplication is also componentwise:

$$
(\alpha \mathbf{v})_i = \alpha v_i.
$$

For example,

$$
2
\begin{pmatrix}
1 \\ -3 \\ 4
\end{pmatrix}
=
\begin{pmatrix}
2 \\ -6 \\ 8
\end{pmatrix},
\qquad
-
\begin{pmatrix}
1 \\ -3 \\ 4
\end{pmatrix}
=
\begin{pmatrix}
-1 \\ 3 \\ -4
\end{pmatrix}.
$$

The main properties are:

- $\alpha(\mathbf{u} + \mathbf{v}) = \alpha \mathbf{u} + \alpha \mathbf{v}$
- $(\alpha + \beta)\mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}$
- $\alpha(\beta \mathbf{v}) = (\alpha \beta)\mathbf{v}$
- $1 \mathbf{v} = \mathbf{v}$

Geometrically, positive scaling stretches or shrinks the vector, while negative scaling also reverses direction.

```text
Scalar multiplication changes length and possibly direction

alpha > 1           0 < alpha < 1         alpha < 0

O----->             O-->                  O<-----
stretch             shrink                flip + scale
```

### 2.4 Linear Combinations

A linear combination of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is any expression of the form

$$
\alpha_1 \mathbf{v}_1 + \alpha_2 \mathbf{v}_2 + \cdots + \alpha_k \mathbf{v}_k
=
\sum_{i=1}^{k} \alpha_i \mathbf{v}_i
$$

with scalars $\alpha_1, \ldots, \alpha_k$.

This idea is central because most linear algebra questions reduce to asking which linear combinations are possible. If you know the allowable combinations, you know what a system can generate.

In AI, linear combinations are everywhere:

- a dense layer computes weighted sums of input coordinates before adding bias
- attention outputs are weighted sums of value vectors
- residual streams combine contributions from multiple components additively
- concept arithmetic in embedding space uses vector addition and subtraction as semantic approximations

The famous analogy

$$
\text{king} - \text{man} + \text{woman} \approx \text{queen}
$$

is a statement that some semantic relations behave approximately linearly in embedding space (Mikolov et al., 2013). It is not an exact algebraic law, but it is a powerful geometric clue.

### 2.5 Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are linearly independent if the only way to make the zero vector from them is the trivial way:

$$
\alpha_1 \mathbf{v}_1 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0}
\quad \Longrightarrow \quad
\alpha_1 = \cdots = \alpha_k = 0.
$$

If a nontrivial combination gives zero, the vectors are linearly dependent. Dependence means redundancy: at least one vector can be expressed in terms of the others.

Examples:

- $(1,0)$ and $(0,1)$ are independent in $\mathbb{R}^2$
- $(1,2)$ and $(2,4)$ are dependent because $(2,4) = 2(1,2)$
- any set containing $\mathbf{0}$ is automatically dependent

Testing independence computationally is a rank question. Put the vectors as columns of a matrix $A$. Then:

- if $\mathrm{rank}(A) = k$, the columns are independent
- if $\mathrm{rank}(A) < k$, they are dependent

In model analysis, approximate dependence matters as much as exact dependence. If different heads or features occupy nearly the same direction, they carry overlapping information and may be prunable.

```text
Independent vs dependent in R^2

Independent                     Dependent

      ^ y                            ^ y
      |   w                          |  2v
      |  /                           | /
      | /                            |/
------+------> x             -------+------> x
     /                               /
    v                               v

two distinct directions         same line, one is redundant
```

### 2.6 Span

The span of a set of vectors is the set of all linear combinations of those vectors:

$$
\mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}
=
\left\{
\sum_{i=1}^{k} \alpha_i \mathbf{v}_i \; \middle| \; \alpha_i \in \mathbb{R}
\right\}.
$$

The span tells you every direction reachable from the given generating set.

Typical cases:

- $\mathrm{span}\{\mathbf{v}\}$ is a line through the origin
- $\mathrm{span}\{\mathbf{v}, \mathbf{w}\}$ is a plane through the origin if $\mathbf{v}$ and $\mathbf{w}$ are independent
- $\mathrm{span}\{\mathbf{e}_1, \ldots, \mathbf{e}_n\} = \mathbb{R}^n$

Two facts are worth memorizing:

1. span is always a subspace
2. adding a vector outside the current span increases the dimension by one

This is why span is the right language for representational capacity. A low-rank matrix can only output vectors inside a low-dimensional span. A collection of value vectors can only produce attention outputs inside their span.

```text
What span looks like geometrically

span{v}              span{v, w}               span{e1, ..., en}

   line                 plane                    whole ambient space

    /                    __________
   /                    /        /|
--+--->                /________/ |
                      |        |  |
                      |        | /
                      |________|/
```

---

## 3. Abstract Vector Spaces

### 3.1 The Vector Space Axioms

Let $F$ be a field, usually $\mathbb{R}$ or $\mathbb{C}$. A vector space over $F$ is a set $V$ equipped with:

- vector addition: $+: V \times V \to V$
- scalar multiplication: $\cdot: F \times V \to V$

such that, for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $\alpha, \beta \in F$, the following laws hold.

| Axiom | Statement |
| --- | --- |
| Additive closure | $\mathbf{u} + \mathbf{v} \in V$ |
| Commutativity | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| Associativity | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| Zero vector | There exists $\mathbf{0} \in V$ with $\mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| Additive inverse | For each $\mathbf{v}$ there exists $-\mathbf{v}$ with $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| Scalar closure | $\alpha \mathbf{v} \in V$ |
| Distributivity over vector addition | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha \mathbf{u} + \alpha \mathbf{v}$ |
| Distributivity over scalar addition | $(\alpha + \beta)\mathbf{v} = \alpha \mathbf{v} + \beta \mathbf{v}$ |
| Scalar associativity | $\alpha(\beta \mathbf{v}) = (\alpha \beta)\mathbf{v}$ |
| Unit scalar | $1 \mathbf{v} = \mathbf{v}$ |

Some texts count eight axioms by folding closure into the operation definitions. Listing ten is often pedagogically clearer.

### 3.2 Consequences of the Axioms

The axioms are minimal, but they already imply a lot.

**The zero vector is unique.**  
If $\mathbf{0}$ and $\mathbf{0}'$ both behave like additive identities, then

$$
\mathbf{0} = \mathbf{0} + \mathbf{0}' = \mathbf{0}'.
$$

**Additive inverses are unique.**  
If $\mathbf{w}$ and $\mathbf{w}'$ both satisfy $\mathbf{v} + \mathbf{w} = \mathbf{0}$ and $\mathbf{v} + \mathbf{w}' = \mathbf{0}$, then

$$
\mathbf{w} = \mathbf{w} + \mathbf{0}
= \mathbf{w} + (\mathbf{v} + \mathbf{w}')
= (\mathbf{w} + \mathbf{v}) + \mathbf{w}'
= \mathbf{0} + \mathbf{w}'
= \mathbf{w}'.
$$

**Zero scalar kills every vector.**

$$
0 \mathbf{v} = (0 + 0)\mathbf{v} = 0 \mathbf{v} + 0 \mathbf{v}.
$$

Add the additive inverse of $0\mathbf{v}$ to both sides to get $0\mathbf{v} = \mathbf{0}$.

**Every scalar kills the zero vector.**

$$
\alpha \mathbf{0} = \alpha(\mathbf{0} + \mathbf{0}) = \alpha \mathbf{0} + \alpha \mathbf{0},
$$

so again $\alpha \mathbf{0} = \mathbf{0}$.

**Multiplying by $-1$ gives the additive inverse.**

$$
\mathbf{v} + (-1)\mathbf{v} = (1 + -1)\mathbf{v} = 0\mathbf{v} = \mathbf{0},
$$

so $(-1)\mathbf{v} = -\mathbf{v}$.

These are simple but useful. Many later proofs quietly depend on them.

### 3.3 Examples of Vector Spaces

The same axioms govern many different mathematical objects.

**1. Euclidean space $\mathbb{R}^n$**  
This is the standard example. Addition and scalar multiplication are componentwise.

**2. Polynomial space $P_n$**  
The set of polynomials of degree at most $n$:

$$
P_n = \{a_0 + a_1 x + \cdots + a_n x^n : a_i \in \mathbb{R}\}.
$$

Addition is polynomial addition, and scalar multiplication rescales coefficients.

**3. Matrix space $\mathbb{R}^{m \times n}$**  
All $m \times n$ real matrices form a vector space with componentwise addition and scaling.

**4. Continuous function space $C([a,b])$**  
All continuous real-valued functions on $[a,b]$ form a vector space under pointwise operations:

$$
(f+g)(x) = f(x)+g(x),
\qquad
(\alpha f)(x) = \alpha f(x).
$$

**5. Sequence spaces**  
Infinite sequences can also form vector spaces. Important examples include $\ell^2$, the square-summable sequences, and $\ell^1$, the absolutely summable sequences.

**6. $L^2([0,1])$**  
Square-integrable functions form an infinite-dimensional vector space that becomes a Hilbert space once we equip it with the usual inner product.

This last example matters in machine learning theory because kernels, Fourier methods, and approximation theory are often most naturally phrased in function spaces rather than coordinate spaces.

### 3.4 Non-Examples

Students usually learn vector spaces faster by seeing what fails.

| Set | Why it is not a vector space |
| --- | --- |
| $\{x \in \mathbb{R}^n : x_i > 0 \text{ for all } i\}$ | Not closed under additive inverse |
| $\{x \in \mathbb{R}^n : \|x\| = 1\}$ | Not closed under addition or scaling |
| Probability simplex $\Delta^{n-1}$ | Not closed under arbitrary addition or scaling |
| $\mathbb{Z}^n$ over $\mathbb{R}$ | Not closed under real scalar multiplication |
| A line not through the origin | Does not contain the zero vector |

The probability simplex is especially important in AI. It lives inside a vector space, but it is not itself a vector space because probabilities must remain nonnegative and sum to one.

### 3.5 Subspaces

A subset $W \subseteq V$ is a subspace if it is itself a vector space under the same operations. The practical test is short:

1. $\mathbf{0} \in W$
2. if $\mathbf{u}, \mathbf{v} \in W$, then $\mathbf{u} + \mathbf{v} \in W$
3. if $\mathbf{v} \in W$ and $\alpha \in F$, then $\alpha \mathbf{v} \in W$

Equivalently, a nonempty set is a subspace if it is closed under linear combinations.

Examples inside $\mathbb{R}^3$:

- $\{\mathbf{0}\}$ is the trivial subspace
- any line through the origin is a one-dimensional subspace
- any plane through the origin is a two-dimensional subspace
- $\mathbb{R}^3$ itself is a subspace

Non-examples:

- a translated line or plane not passing through the origin
- the unit sphere
- the set of vectors with first coordinate equal to 1

Subspaces are how linear algebra talks about structure. The column space of a matrix, the null space of a map, the subspace spanned by a collection of attention heads, and the tangent space of a model family all follow the same logic: identify the directions that are allowed, closed, and linearly stable.

```text
Subspace vs non-subspace in R^2

Subspace (through origin)      Not a subspace (shifted)

      /                             /
     /                             /
----+----> x                  -----/----> x
   /                             /
  /

contains 0                     misses 0
```

---

## 4. Basis and Dimension

### 4.1 Basis Definition

A basis for a vector space $V$ is a set of vectors

$$
B = \{\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_n\}
$$

that satisfies two conditions:

1. the vectors are linearly independent
2. the vectors span the whole space

These two requirements are exact opposites that must hold simultaneously. Independence prevents redundancy; spanning prevents missing directions.

If $B$ is a basis, then every vector $\mathbf{v} \in V$ has a unique expansion

$$
\mathbf{v} = \alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2 + \cdots + \alpha_n \mathbf{b}_n.
$$

The coefficients $\alpha_1, \ldots, \alpha_n$ are the coordinates of $\mathbf{v}$ relative to $B$.

Uniqueness matters. If the same vector had two different coordinate descriptions in the same basis, then subtracting those descriptions would produce a nontrivial linear combination of basis vectors equal to zero, contradicting independence.

```text
In R^2:

one vector          two non-collinear vectors       three vectors
not enough          just enough = basis             spanning but redundant

   /                    ^ y                           ^ y
  /                     |  /                          |  /|
 /                      | /                           | / |
-----------------> x    +--------> x                 +--------> x
                       /                             /   /
```

### 4.2 Standard Basis of R^n

The standard basis of $\mathbb{R}^n$ is

$$
\mathbf{e}_1 = (1,0,\ldots,0), \;
\mathbf{e}_2 = (0,1,\ldots,0), \;
\ldots, \;
\mathbf{e}_n = (0,0,\ldots,1).
$$

In the standard basis, the coordinate vector is just the familiar list of components:

$$
\mathbf{v} =
\begin{pmatrix}
v_1 \\
\vdots \\
v_n
\end{pmatrix}
=
v_1 \mathbf{e}_1 + \cdots + v_n \mathbf{e}_n.
$$

But the standard basis is not sacred. Many problems become simpler after a change of basis:

- PCA chooses a basis aligned with data variance
- Fourier analysis chooses sinusoidal basis functions
- diagonalization chooses an eigenbasis when available
- orthonormal bases simplify coordinates, projections, and energy calculations

The vector does not change when the basis changes. Only its coordinate description changes.

### 4.3 Dimension

For finite-dimensional vector spaces, every basis has the same number of elements. That common number is the dimension of the space:

$$
\dim(V) = \text{number of vectors in any basis of } V.
$$

Examples:

- $\dim(\mathbb{R}^n) = n$
- $\dim(\mathbb{R}^{m \times n}) = mn$
- $\dim(P_n) = n+1$ because $\{1, x, x^2, \ldots, x^n\}$ is a basis
- $\dim(C([a,b])) = \infty$

Dimension is a count of independent directions, not a count of how many vectors happen to be mentioned in a description. You can describe a plane in $\mathbb{R}^3$ using ten spanning vectors if you want; the dimension is still 2.

In AI practice, dimension is often both a modeling choice and a computational budget. Embedding dimension, head dimension, hidden width, and bottleneck rank are all dimension decisions.

### 4.4 Dimension and Subspaces

If $W$ is a subspace of a finite-dimensional vector space $V$, then

$$
\dim(W) \leq \dim(V).
$$

Moreover:

- if $\dim(W) = \dim(V)$, then $W = V$
- if $\dim(W) = 0$, then $W = \{\mathbf{0}\}$
- in $\mathbb{R}^2$, the only subspaces are $\{\mathbf{0}\}$, lines through the origin, and $\mathbb{R}^2$
- in $\mathbb{R}^3$, the only subspaces are $\{\mathbf{0}\}$, lines through the origin, planes through the origin, and $\mathbb{R}^3$

The codimension of $W$ in $V$ is

$$
\mathrm{codim}(W) = \dim(V) - \dim(W).
$$

Codimension tells you how many independent directions are missing.

This is the right language for low-rank modeling. If a matrix maps a 4096-dimensional residual stream into a 64-dimensional attention head space, then its image has dimension at most 64, so most of the ambient directions are not expressible in that head.

### 4.5 Coordinates and Change of Basis

Let $B = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ be a basis for $V$. The coordinate vector of $\mathbf{v}$ relative to $B$ is

$$
[\mathbf{v}]_B =
\begin{pmatrix}
\alpha_1 \\
\vdots \\
\alpha_n
\end{pmatrix}
\quad \text{where} \quad
\mathbf{v} = \alpha_1 \mathbf{b}_1 + \cdots + \alpha_n \mathbf{b}_n.
$$

If $B'$ is another basis, then coordinates transform by an invertible matrix:

$$
[\mathbf{v}]_{B'} = P [\mathbf{v}]_B.
$$

The change-of-basis matrix is invertible because coordinates in each basis are unique. If it were not invertible, some nonzero coordinate vector would collapse to zero, contradicting uniqueness.

Suppose a linear map $T$ is represented by matrix $A$ in one basis and by matrix $A'$ in another basis. Then the relationship is

$$
A' = P^{-1} A P.
$$

This is similarity transformation. It does not change the underlying map; it changes the coordinates in which the map is described.

That idea appears all over ML:

- whitening and PCA rotate data into more convenient coordinates
- orthogonal parameterizations change basis while preserving norms
- query, key, and value projections choose learned coordinate systems inside the embedding dimension

```text
Same vector, different coordinates

standard basis:              rotated basis:

      y                           b2
      ^                          /
      |    v                    /
      |   /                    /   v
      |  /                    /
------+------> x        -----/----------> b1

The geometric arrow is the same.
Only the coordinate description changes.
```

### 4.6 Rank-Nullity Theorem

Let $T: V \to W$ be a linear map with finite-dimensional domain. Then

$$
\dim(\ker T) + \dim(\mathrm{im}\,T) = \dim(V).
$$

The two terms are named:

- **nullity**: $\dim(\ker T)$
- **rank**: $\dim(\mathrm{im}\,T)$

For a matrix $A \in \mathbb{R}^{m \times n}$, this becomes

$$
\mathrm{rank}(A) + \mathrm{nullity}(A) = n.
$$

Interpretation:

- rank counts how many independent directions survive the map
- nullity counts how many independent directions are lost completely

If $A$ has rank $r < n$, then every output lies in an $r$-dimensional subspace of $\mathbb{R}^m$, and an $(n-r)$-dimensional family of input directions gets sent to zero.

This theorem explains a great deal of ML engineering:

- low-rank adaptation works because useful updates often live in small image spaces (Hu et al., 2022)
- bottleneck layers deliberately compress information into lower-dimensional subspaces
- overparameterized models have large null spaces in parameter space, which helps explain why many parameter settings realize similar functions

---

## 5. Norms and Metric Spaces

### 5.1 Norms on Vector Spaces

A norm on a vector space $V$ is a function

$$
\|\cdot\|: V \to \mathbb{R}
$$

satisfying, for all $\mathbf{u}, \mathbf{v} \in V$ and all scalars $\alpha$:

1. **positive definiteness**: $\|\mathbf{v}\| \geq 0$, with equality iff $\mathbf{v} = \mathbf{0}$
2. **homogeneity**: $\|\alpha \mathbf{v}\| = |\alpha| \, \|\mathbf{v}\|$
3. **triangle inequality**: $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$

A norm measures size. Once a norm is available, we can talk about boundedness, convergence, approximation error, and regularization.

### 5.2 The p-Norms

For $\mathbf{v} \in \mathbb{R}^n$ and $1 \leq p < \infty$, the $p$-norm is

$$
\|\mathbf{v}\|_p =
\left(
\sum_{i=1}^{n} |v_i|^p
\right)^{1/p}.
$$

The limiting case is

$$
\|\mathbf{v}\|_\infty = \max_i |v_i|.
$$

The most important examples are:

**L1 norm**

$$
\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i|.
$$

This encourages sparsity in optimization because its unit ball has corners aligned with coordinate axes.

**L2 norm**

$$
\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}.
$$

This is the Euclidean length and the most common norm in deep learning.

**L-infinity norm**

$$
\|\mathbf{v}\|_\infty = \max_i |v_i|.
$$

This measures worst-case component size and is common in adversarial robustness.

The geometry of the unit ball depends on $p$:

- $p=1$: diamond or cross-polytope
- $p=2$: sphere
- $p=\infty$: cube

Those shapes are not cosmetic. They explain different optimization behavior.

```text
Unit balls in 2D

L1 ball              L2 ball              L_inf ball

   /\                  ____               +------+
  /  \               /      \             |      |
  \  /               \______/             |      |
   \/                                       +------+

corners              smooth                flat faces
```

### 5.3 Norm Equivalence

In finite-dimensional spaces, all norms induce the same notion of convergence. More precisely, if $\|\cdot\|_a$ and $\|\cdot\|_b$ are norms on $\mathbb{R}^n$, then there exist constants $c_1, c_2 > 0$ such that

$$
c_1 \|\mathbf{v}\|_a \leq \|\mathbf{v}\|_b \leq c_2 \|\mathbf{v}\|_a
\qquad
\text{for all } \mathbf{v} \in \mathbb{R}^n.
$$

Useful special inequalities:

$$
\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1,
$$

$$
\|\mathbf{v}\|_1 \leq \sqrt{n}\,\|\mathbf{v}\|_2,
\qquad
\|\mathbf{v}\|_2 \leq \sqrt{n}\,\|\mathbf{v}\|_\infty,
\qquad
\|\mathbf{v}\|_1 \leq n \|\mathbf{v}\|_\infty.
$$

This means that in finite dimensions, saying "a sequence converges" does not depend on whether you measure error with L1, L2, or L-infinity. In infinite dimensions, this is false. Different norms can induce genuinely different topologies.

### 5.4 Matrix Norms

Matrices also admit norms. Important ones include:

**Frobenius norm**

$$
\|A\|_F =
\sqrt{\sum_{i,j} A_{ij}^2}
=
\sqrt{\mathrm{tr}(A^\top A)}.
$$

This is the Euclidean norm of the matrix viewed as one long vector.

**Spectral norm**

$$
\|A\|_2 = \sigma_{\max}(A).
$$

This is the maximum stretch factor of the linear map:

$$
\|A\|_2 = \sup_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2.
$$

**Nuclear norm**

$$
\|A\|_* = \sum_i \sigma_i(A).
$$

This is the sum of singular values. It is a convex surrogate for rank and appears in matrix completion and low-rank learning (Candes and Recht, 2009).

**Induced 1-norm and infinity-norm**

$$
\|A\|_1 = \max_j \sum_i |A_{ij}|,
\qquad
\|A\|_\infty = \max_i \sum_j |A_{ij}|.
$$

In ML:

- Frobenius norm corresponds to standard weight decay on matrix entries
- spectral norm controls the Lipschitz constant of a linear layer
- nuclear norm promotes low-rank structure

### 5.5 Metric Spaces

A metric space is a set $X$ equipped with a distance function

$$
d: X \times X \to \mathbb{R}
$$

such that for all $x,y,z \in X$:

1. $d(x,y) \geq 0$, with equality iff $x=y$
2. $d(x,y) = d(y,x)$
3. $d(x,z) \leq d(x,y) + d(y,z)$

Every norm induces a metric by

$$
d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|.
$$

Not every metric comes from a norm. Edit distance on strings is a metric, but there is no vector subtraction of strings that generates it as a norm.

AI uses many distances:

- Euclidean distance for retrieval and clustering
- cosine distance for directional similarity
- Hamming distance for binary codes
- edit distance for token sequences
- Wasserstein distance for distributions
- KL divergence as a useful non-metric divergence

The important point is conceptual: vector-space geometry gives one family of distances, but machine learning uses broader metric ideas too.

### 5.6 Convergence in Normed Spaces

A sequence $(\mathbf{v}_n)$ converges to $\mathbf{v}$ in norm if

$$
\|\mathbf{v}_n - \mathbf{v}\| \to 0
\qquad \text{as } n \to \infty.
$$

A sequence is Cauchy if its terms eventually become arbitrarily close to each other:

$$
\forall \varepsilon > 0, \exists N \text{ such that } m,n \geq N
\implies
\|\mathbf{v}_n - \mathbf{v}_m\| < \varepsilon.
$$

A normed space is complete if every Cauchy sequence converges to a limit that still belongs to the space. A complete normed space is called a Banach space.

Important examples:

- $\mathbb{R}^n$ with any norm is complete
- matrix spaces with standard norms are complete
- $C([a,b])$ with the sup norm is complete
- $L^p$ spaces are complete for $1 \leq p \leq \infty$

Completeness matters because many learning algorithms are iterative. Gradient methods, fixed-point solvers, and optimization routines generate sequences. Completeness is the condition that prevents the limit from "falling out of the space."

---

## 6. Inner Product Spaces

### 6.1 Inner Products

An inner product on a real vector space $V$ is a function

$$
\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}
$$

satisfying, for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $\alpha, \beta$:

1. **symmetry**: $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **linearity**: $\langle \alpha \mathbf{u} + \beta \mathbf{v}, \mathbf{w} \rangle = \alpha \langle \mathbf{u}, \mathbf{w} \rangle + \beta \langle \mathbf{v}, \mathbf{w} \rangle$
3. **positive definiteness**: $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$, with equality iff $\mathbf{v} = \mathbf{0}$

An inner product upgrades a vector space with geometry. Once it is available, we can measure lengths, define angles, talk about orthogonality, and project onto subspaces.

For complex vector spaces, the definition is slightly modified: $\langle \mathbf{u}, \mathbf{v} \rangle = \overline{\langle \mathbf{v}, \mathbf{u} \rangle}$ and one argument is conjugate-linear. The core geometric story remains the same.

### 6.2 Standard Inner Product on R^n

The standard inner product on $\mathbb{R}^n$ is the dot product:

$$
\langle \mathbf{u}, \mathbf{v} \rangle
=
\mathbf{u}^\top \mathbf{v}
=
\sum_{i=1}^{n} u_i v_i.
$$

This inner product induces the Euclidean norm:

$$
\|\mathbf{v}\|_2 = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}.
$$

So in Euclidean space, angle and length are not independent notions. They are both derived from the same primitive object.

### 6.3 Geometric Interpretation

The angle $\theta$ between nonzero vectors $\mathbf{u}$ and $\mathbf{v}$ is defined by

$$
\cos \theta =
\frac{\langle \mathbf{u}, \mathbf{v} \rangle}
{\|\mathbf{u}\| \, \|\mathbf{v}\|}.
$$

Interpretation:

- $\theta = 0$: same direction, maximal alignment
- $\theta = \pi/2$: orthogonal, zero alignment
- $\theta = \pi$: opposite directions

Cosine similarity is exactly this normalized inner product. It is widely used for embeddings because it measures direction rather than raw magnitude.

For attention, the score $q^\top k / \sqrt{d_k}$ combines both angular alignment and magnitude. If vectors are normalized, the score is proportional to cosine similarity. If they are not, norm effects matter too.

```text
Angle and alignment

same direction          orthogonal            opposite direction

-----> ----->           ----->                <----- ----->
                        |
                        |
                        v

cos(theta) = 1          cos(theta) = 0        cos(theta) = -1
```

### 6.4 Cauchy-Schwarz Inequality

The fundamental inequality of inner-product geometry is

$$
|\langle \mathbf{u}, \mathbf{v} \rangle|
\leq
\|\mathbf{u}\| \, \|\mathbf{v}\|.
$$

Equality holds exactly when $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent.

A classic proof uses positivity. For any real $t$,

$$
0 \leq \|\mathbf{u} + t \mathbf{v}\|^2
=
\langle \mathbf{u} + t\mathbf{v}, \mathbf{u} + t\mathbf{v} \rangle
=
\|\mathbf{u}\|^2 + 2t \langle \mathbf{u}, \mathbf{v} \rangle + t^2 \|\mathbf{v}\|^2.
$$

This quadratic in $t$ must have nonpositive discriminant, so

$$
4\langle \mathbf{u}, \mathbf{v} \rangle^2 - 4\|\mathbf{u}\|^2 \|\mathbf{v}\|^2 \leq 0,
$$

which implies the result.

Cauchy-Schwarz guarantees that cosine similarity always lies in $[-1,1]$. It also gives quick upper bounds on correlations, dot products, and approximation error.

### 6.5 Orthogonality

Vectors are orthogonal if their inner product is zero:

$$
\mathbf{u} \perp \mathbf{v}
\quad \Longleftrightarrow \quad
\langle \mathbf{u}, \mathbf{v} \rangle = 0.
$$

An orthogonal set is a collection of pairwise orthogonal nonzero vectors. Such a set is automatically linearly independent.

Proof sketch: if

$$
\alpha_1 \mathbf{v}_1 + \cdots + \alpha_k \mathbf{v}_k = \mathbf{0},
$$

take the inner product with $\mathbf{v}_j$. All cross-terms vanish, leaving

$$
\alpha_j \|\mathbf{v}_j\|^2 = 0,
$$

so $\alpha_j = 0$ for every $j$.

An orthonormal set is orthogonal and normalized:

$$
\langle \mathbf{u}_i, \mathbf{u}_j \rangle = \delta_{ij}.
$$

In an orthonormal basis, coordinates are exceptionally simple:

$$
\mathbf{v} = \sum_{i=1}^{n} \langle \mathbf{v}, \mathbf{u}_i \rangle \mathbf{u}_i.
$$

You get coefficients by taking inner products. No matrix inversion is needed.

Orthogonality is central in ML because it reduces interference:

- orthogonal initializations help preserve signal norms through depth (Saxe et al., 2014)
- orthogonal features are easier to disentangle
- orthogonal subspaces make decomposition interpretable

### 6.6 Gram-Schmidt Orthogonalization

Given linearly independent vectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$, the Gram-Schmidt procedure constructs an orthonormal basis $\mathbf{u}_1, \ldots, \mathbf{u}_n$ spanning the same subspace.

First set

$$
\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}.
$$

Then recursively subtract the components already accounted for:

$$
\widetilde{\mathbf{u}}_k =
\mathbf{v}_k - \sum_{j=1}^{k-1}
\langle \mathbf{v}_k, \mathbf{u}_j \rangle \mathbf{u}_j,
$$

and normalize:

$$
\mathbf{u}_k =
\frac{\widetilde{\mathbf{u}}_k}{\|\widetilde{\mathbf{u}}_k\|}.
$$

Each step removes the part of $\mathbf{v}_k$ already explained by the previous orthonormal vectors. What remains is orthogonal to the earlier directions.

Conceptually, Gram-Schmidt says:

1. start with a spanning description
2. strip away redundancy direction by direction
3. normalize what survives

Algorithmically, Gram-Schmidt underlies QR decomposition. Numerically, modified Gram-Schmidt and Householder methods are often preferred for stability.

```text
Gram-Schmidt idea for v2

u1 -------------------->

v2 ------------------------>

split v2 into:

proj_u1(v2) -------------->
remainder                  ^
                           |
                           |

Keep the remainder, then normalize it.
```

### 6.7 Orthogonal Complement

If $W$ is a subspace of an inner-product space $V$, its orthogonal complement is

$$
W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}.
$$

Important facts:

- $W^\perp$ is always a subspace
- in finite dimensions, $\dim(W) + \dim(W^\perp) = \dim(V)$
- every vector decomposes uniquely as

$$
\mathbf{v} = \mathbf{w} + \mathbf{w}^\perp
\qquad
\text{with }
\mathbf{w} \in W,\;
\mathbf{w}^\perp \in W^\perp
$$

This decomposition is one of the most useful ideas in linear algebra. It is how least squares, residuals, and projection error are understood.

For matrices, the row space is orthogonal to the null space, and the column space is orthogonal to the left null space. Those are special cases of the same principle.

```text
Orthogonal decomposition

               v
              /|
             / |
            /  |  v_perp
-----------*---+-----------------> W
          proj_W(v)

v = proj_W(v) + v_perp
```

### 6.8 Hilbert Spaces

A Hilbert space is a complete inner-product space. It has both geometry and completeness.

Finite-dimensional inner-product spaces are automatically Hilbert spaces. The interesting cases are infinite-dimensional:

- $\ell^2$: square-summable sequences
- $L^2([a,b])$: square-integrable functions
- reproducing kernel Hilbert spaces (RKHS), central to kernel methods

In a Hilbert space, orthonormal bases and projection theory still work, though bases may now be infinite. Parseval-type identities and Fourier expansions become available:

$$
\|f\|^2 = \sum_{i=1}^{\infty} |\langle f, e_i \rangle|^2
$$

for suitable orthonormal bases $\{e_i\}$.

Hilbert spaces matter for AI because many learning-theoretic objects are not finite vectors at all; they are functions. Kernel methods, Gaussian processes, and parts of neural network theory live more naturally in Hilbert spaces than in $\mathbb{R}^n$.

---

## 7. Orthogonal Projections

### 7.1 Projection onto a Subspace

Given a subspace $W$ of an inner-product space and a vector $\mathbf{v}$, the orthogonal projection of $\mathbf{v}$ onto $W$ is the vector in $W$ closest to $\mathbf{v}$:

$$
\mathrm{Proj}_W(\mathbf{v})
=
\arg\min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|.
$$

The key characterization is:

- $\mathrm{Proj}_W(\mathbf{v}) \in W$
- $\mathbf{v} - \mathrm{Proj}_W(\mathbf{v}) \in W^\perp$

For a one-dimensional subspace spanned by a nonzero vector $\mathbf{u}$,

$$
\mathrm{Proj}_{\mathbf{u}}(\mathbf{v})
=
\frac{\langle \mathbf{v}, \mathbf{u} \rangle}
{\langle \mathbf{u}, \mathbf{u} \rangle}
\mathbf{u}.
$$

If $\mathbf{u}$ is already unit length, this simplifies to

$$
\mathrm{Proj}_{\mathbf{u}}(\mathbf{v}) = \langle \mathbf{v}, \mathbf{u} \rangle \mathbf{u}.
$$

Projection is the formal version of "keep the part of the vector that points in the subspace, discard the perpendicular remainder."

```text
Projection onto a line or subspace

          v
         /|
        / |
       /  |  residual = v - Proj_W(v)
------*---+---------------------- W
     Proj_W(v)
```

### 7.2 Projection Matrix Properties

If $P$ is the matrix of an orthogonal projection, then:

- $P^2 = P$ (idempotence)
- $P^\top = P$ (symmetry)

The first identity says projecting twice is the same as projecting once. The second says the projection is orthogonal rather than oblique.

For projection onto the line spanned by a nonzero vector $\mathbf{u}$,

$$
P = \frac{\mathbf{u}\mathbf{u}^\top}{\mathbf{u}^\top \mathbf{u}}.
$$

Its eigenvalues are 0 and 1 only:

- eigenvalue 1 corresponds to directions already in the target subspace
- eigenvalue 0 corresponds to directions annihilated by projection

The complementary projector is $I-P$, which projects onto the orthogonal complement.

### 7.3 Projection onto Column Space

Let $A \in \mathbb{R}^{m \times n}$ have full column rank. The orthogonal projector onto the column space of $A$ is

$$
P_A = A(A^\top A)^{-1}A^\top.
$$

Why this formula works:

1. every projected vector has the form $A\mathbf{x}$, so it lies in $\mathrm{col}(A)$
2. the residual must be orthogonal to every column of $A$
3. orthogonality of the residual gives the normal equations

If the columns of $A$ are orthonormal, then $A^\top A = I$ and the formula simplifies to

$$
P_A = AA^\top.
$$

This expression appears in least squares, regression, PCA, and low-dimensional approximation.

In Transformer language, learned projections $W_Q$, $W_K$, and $W_V$ are not orthogonal projectors in the strict algebraic sense, but they do map activations into lower-dimensional subspaces where later computations occur. Orthogonal projection is therefore the clean mathematical ideal behind many approximate representation moves.

```text
Projecting b onto col(A)

          b
         /|
        / |
       /  | residual
      /   |
-----*----+----------> col(A)
   A x_hat

Best-fit output lies inside col(A).
The leftover error is orthogonal to col(A).
```

### 7.4 Gram Matrix

Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ in $\mathbb{R}^d$, the Gram matrix is

$$
G_{ij} = \langle \mathbf{v}_i, \mathbf{v}_j \rangle.
$$

If $V$ is the matrix whose rows are the vectors, then

$$
G = VV^\top.
$$

Gram matrices have two key properties:

1. they are symmetric
2. they are positive semidefinite

Indeed, for any coefficient vector $\mathbf{x}$,

$$
\mathbf{x}^\top G \mathbf{x}
=
\mathbf{x}^\top VV^\top \mathbf{x}
=
\|V^\top \mathbf{x}\|^2
\geq 0.
$$

The Gram matrix is positive definite exactly when the vectors are linearly independent.

In AI:

- attention score matrices are built from dot products, so they are Gram-like objects before softmax scaling and masking
- kernel matrices are Gram matrices in feature space
- covariance and similarity matrices are Gram constructions with centering or normalization added

### 7.5 Best Approximation and Least Squares

Orthogonal projection solves the best approximation problem:

$$
\min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|_2.
$$

If $\mathbf{p} = \mathrm{Proj}_W(\mathbf{v})$, then the residual $\mathbf{r} = \mathbf{v} - \mathbf{p}$ is orthogonal to $W$, and Pythagoras gives

$$
\|\mathbf{v}\|^2 = \|\mathbf{p}\|^2 + \|\mathbf{r}\|^2.
$$

Least squares is exactly this geometry in matrix form. Given an overdetermined system $A\mathbf{x} \approx \mathbf{b}$, the least-squares solution minimizes

$$
\|A\mathbf{x} - \mathbf{b}\|_2^2.
$$

The residual must be orthogonal to the column space of $A$:

$$
A^\top (A\mathbf{x} - \mathbf{b}) = 0,
$$

so

$$
A^\top A \mathbf{x} = A^\top \mathbf{b}.
$$

When $A$ has full column rank,

$$
\mathbf{x}^* = (A^\top A)^{-1} A^\top \mathbf{b}.
$$

Geometrically, $A\mathbf{x}^*$ is the projection of $\mathbf{b}$ onto the column space of $A$.

This is why projection is not just a geometric curiosity. It is the linear algebra under classical regression, representation compression, denoising, and many approximation arguments used throughout machine learning.

---

## 8. Affine Subspaces and Convexity

### 8.1 Affine Subspaces

A linear subspace must pass through the origin. Many important geometric sets do not. An affine subspace is a translated subspace:

$$
A = \mathbf{v}_0 + W = \{\mathbf{v}_0 + \mathbf{w} : \mathbf{w} \in W\}.
$$

Here $\mathbf{v}_0$ is a base point and $W$ is a linear subspace called the direction space.

Examples:

- a line not through the origin in $\mathbb{R}^2$
- a plane $ax + by + cz = d$ in $\mathbb{R}^3$ with $d \neq 0$
- the probability simplex $\Delta^{n-1}$ before nonnegativity is imposed, because $\sum_i p_i = 1$ is an affine constraint

Affine spaces are not vector spaces unless they happen to pass through the origin. You can subtract two points in an affine space to get a direction vector, but you cannot usually add two points and stay in the set.

```text
Linear subspace vs affine subspace

subspace W                  affine set v0 + W

      /                           /
     /                           /
----+----> x                ----/----> x
   /                           /
  /

through 0                    shifted away from 0
```

### 8.2 Convex Sets

A set $C$ is convex if, whenever $\mathbf{u}, \mathbf{v} \in C$ and $\lambda \in [0,1]$,

$$
\lambda \mathbf{u} + (1-\lambda)\mathbf{v} \in C.
$$

This means the entire line segment joining any two points of the set stays inside the set.

Important convex sets:

- every subspace
- every affine subspace
- every norm ball $\{\|\mathbf{v}\| \leq r\}$ for a true norm
- every half-space $\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} \leq b\}$
- the probability simplex

The convex hull of a set $S$, written $\mathrm{conv}(S)$, is the set of all convex combinations of points in $S$. It is the smallest convex set containing $S$.

This matters directly for attention. The output of a single attention head at one position is

$$
\sum_j \alpha_j \mathbf{v}_j
\quad \text{with } \alpha_j \geq 0,\; \sum_j \alpha_j = 1,
$$

so it lies in the convex hull of the value vectors at that position.

```text
Convex vs non-convex

Convex: segment stays inside

*-----------*

Non-convex: segment leaves the set

*---- gap ----*
```

### 8.3 Hyperplanes and Half-Spaces

A hyperplane in $\mathbb{R}^n$ is a set of the form

$$
\{\mathbf{x} \in \mathbb{R}^n : \mathbf{w}^\top \mathbf{x} = b\}
$$

for some nonzero normal vector $\mathbf{w}$.

Geometrically, a hyperplane is an $(n-1)$-dimensional affine subspace. It cuts space into two half-spaces:

$$
\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} \leq b\},
\qquad
\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} \geq b\}.
$$

Hyperplanes are the basic decision surfaces of linear models. A neuron computes

$$
\mathbf{w}^\top \mathbf{x} + b.
$$

Thresholding that expression defines a half-space. A ReLU network can therefore be viewed as composing piecewise-linear maps defined by many hyperplane arrangements.

Convex analysis adds a deeper theorem: disjoint convex sets can often be separated by hyperplanes. This is the geometric basis for max-margin classification, linear probes, and many duality arguments.

```text
Hyperplane and half-spaces

half-space        hyperplane         half-space

xxxxx             |                  .....
xxxxx             |  w^T x = b       .....
xxxxx             |                  .....
                  ^
                  normal direction w
```

### 8.4 Convex Functions and Sublevel Sets

A function $f: V \to \mathbb{R}$ is convex if for all $\mathbf{u}, \mathbf{v}$ and $\lambda \in [0,1]$,

$$
f(\lambda \mathbf{u} + (1-\lambda)\mathbf{v})
\leq
\lambda f(\mathbf{u}) + (1-\lambda) f(\mathbf{v}).
$$

Convex functions are important because their sublevel sets

$$
\{\mathbf{v} : f(\mathbf{v}) \leq c\}
$$

are convex. This makes optimization far more tractable.

Examples:

- $f(\mathbf{x}) = \|\mathbf{x}\|_2^2$ is convex
- $f(\mathbf{x}) = \|\mathbf{x}\|_1$ is convex
- logistic loss and cross-entropy are convex in suitable linear-model settings
- deep network training objectives are generally not globally convex

The contrast matters. Classical optimization often uses convex structure globally. Deep learning rarely gets global convexity, but convex sets and convex penalties still appear everywhere locally and architecturally.

---

## 9. High-Dimensional Geometry

### 9.1 The Curse of Dimensionality

Low-dimensional intuition is unreliable in high dimensions. This broad phenomenon is often called the curse of dimensionality.

One form is volume concentration. For a $d$-dimensional ball of radius $r$, the fraction of volume inside the inner ball of radius $r-\varepsilon$ is

$$
\left(1 - \frac{\varepsilon}{r}\right)^d.
$$

As $d$ grows, this shrinks rapidly. Most of the volume moves near the boundary.

Another form is distance concentration. Random points in high-dimensional spaces tend to be at similar distances from each other, so nearest-neighbor structure becomes less contrastive unless data has meaningful low-dimensional structure.

For embeddings, this means random vectors in high dimension are usually nearly orthogonal. Meaningful similarity is therefore a signal against a strong null background of near-right angles.

```text
Volume concentration intuition

low dimension:            high dimension:

[ core | shell ]          [tiny core | shell shell shell shell]

As dimension grows, most volume moves into the shell.
```

### 9.2 Concentration of Measure

Concentration of measure says that in high-dimensional spaces, many reasonably smooth functions vary much less than low-dimensional intuition suggests. Informally: a Lipschitz function on a high-dimensional sphere is almost constant on most of the sphere.

One representative result is that if $\mathbf{x}$ is random on the unit sphere $S^{d-1}$ and $\mathbf{u}$ is a fixed unit vector, then the inner product $\langle \mathbf{x}, \mathbf{u} \rangle$ is sharply concentrated near 0 as $d$ grows.

Consequences:

- random directions are almost orthogonal
- norms and averages become more stable than naive geometric pictures suggest
- random projections preserve more structure than one might expect

These ideas are central to modern high-dimensional probability (Vershynin, 2018) and are one reason random features and approximate nearest neighbor methods work surprisingly well.

### 9.3 Johnson-Lindenstrauss Lemma

The Johnson-Lindenstrauss lemma formalizes dimension reduction with bounded distortion (Johnson and Lindenstrauss, 1984; Dasgupta and Gupta, 2003).

Given $n$ points in a high-dimensional Euclidean space and an accuracy parameter $\varepsilon \in (0,1)$, there exists a linear map

$$
f: \mathbb{R}^d \to \mathbb{R}^k
$$

with

$$
k = O\!\left(\frac{\log n}{\varepsilon^2}\right)
$$

such that all pairwise distances are preserved up to multiplicative error $(1 \pm \varepsilon)$:

$$
(1-\varepsilon)\|\mathbf{v}_i - \mathbf{v}_j\|_2^2
\leq
\|f(\mathbf{v}_i) - f(\mathbf{v}_j)\|_2^2
\leq
(1+\varepsilon)\|\mathbf{v}_i - \mathbf{v}_j\|_2^2.
$$

Remarkably, the target dimension depends logarithmically on the number of points, not on the original ambient dimension.

Practical message for ML:

- random projections can compress data while preserving geometry
- retrieval systems can work in reduced spaces
- head dimension can be much smaller than model dimension without destroying all pairwise structure

### 9.4 Random Vectors in High Dimensions

If $\mathbf{x}$ and $\mathbf{y}$ are independent random unit vectors in $\mathbb{R}^d$, then for large $d$,

$$
\mathbb{E}[\langle \mathbf{x}, \mathbf{y} \rangle] = 0,
\qquad
\mathrm{Var}(\langle \mathbf{x}, \mathbf{y} \rangle) \approx \frac{1}{d}.
$$

So the inner product typically has size about $1/\sqrt{d}$, which goes to zero with dimension.

A related fact is

$$
\|\mathbf{x} - \mathbf{y}\|_2^2
=
\|\mathbf{x}\|_2^2 + \|\mathbf{y}\|_2^2 - 2\langle \mathbf{x}, \mathbf{y} \rangle
\approx
2.
$$

Thus random unit vectors are not only almost orthogonal; they are also all about the same Euclidean distance apart.

This helps explain why high-dimensional spaces can pack many nearly independent directions. It also explains why cosine similarity is a more informative baseline than raw Euclidean distance in many embedding applications.

### 9.5 Geometry of Embedding Spaces

Learned embeddings are not random vectors, but high-dimensional geometry strongly shapes how they behave.

Important empirical observations:

- simple semantic relations can appear as approximately linear directions (Mikolov et al., 2013)
- contextual embedding spaces are often anisotropic rather than uniformly spread on a sphere (Ethayarajh, 2019)
- nominal dimension may be large while intrinsic dimension is much smaller
- singular values of embedding or feature matrices often decay quickly, indicating effective low-rank structure

Anisotropy matters because cosine similarity becomes biased if almost all vectors occupy a narrow cone. Various centering, whitening, and contrastive training methods attempt to improve isotropy for retrieval and representation quality.

The lesson is subtle: high-dimensional space provides enormous capacity, but learned models do not use that capacity uniformly. They carve out structured manifolds, cones, and low-rank directions inside the ambient space.

```text
Embedding geometry: isotropic vs anisotropic

roughly isotropic              anisotropic / narrow cone

     .   .                           . .
   .   o   .                          . .
     .   .                            . .
   .       .                           . .

spread in many directions         packed into a small angular region
```

### 9.6 Softmax Temperature and Simplex Geometry

Softmax maps a logit vector $\mathbf{z} \in \mathbb{R}^n$ to a probability vector:

$$
p_i =
\frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)},
$$

where $\tau > 0$ is temperature.

Geometrically:

- as $\tau \to 0$, the distribution concentrates near a vertex of the simplex
- as $\tau \to \infty$, the distribution approaches the center of the simplex

So temperature controls how close the output lies to an extreme point versus the interior of the simplex.

This is not just a sampling trick. It is a geometric control knob for how sharply a model commits to one direction in probability space.

```text
Temperature moves probability inside the simplex

high tau                         low tau

   more uniform                   more peaked

   [0.33 0.33 0.34]  ------->     [0.97 0.02 0.01]
        near center                    near a vertex
```

---

## 10. Linear Maps Between Spaces

### 10.1 Definition and Properties

A map $T: V \to W$ is linear if for all vectors $\mathbf{u}, \mathbf{v}$ and scalars $\alpha, \beta$,

$$
T(\alpha \mathbf{u} + \beta \mathbf{v})
=
\alpha T(\mathbf{u}) + \beta T(\mathbf{v}).
$$

Equivalently, it preserves addition and scalar multiplication separately.

Immediate consequences:

- $T(\mathbf{0}) = \mathbf{0}$
- $T(-\mathbf{v}) = -T(\mathbf{v})$
- $T$ is determined completely by its action on a basis

For finite-dimensional spaces, every linear map $\mathbb{R}^n \to \mathbb{R}^m$ is represented by a unique matrix $A \in \mathbb{R}^{m \times n}$ such that

$$
T(\mathbf{x}) = A\mathbf{x}.
$$

Composition of linear maps corresponds to matrix multiplication.

### 10.2 Kernel and Image

The kernel of a linear map is

$$
\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}.
$$

The image is

$$
\mathrm{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}.
$$

Both are subspaces.

Interpretation:

- the kernel contains directions completely ignored by the map
- the image contains all outputs the map can possibly produce

Injectivity and surjectivity are controlled by these spaces:

- $T$ is injective iff $\ker(T) = \{\mathbf{0}\}$
- $T$ is surjective iff $\mathrm{im}(T) = W$

This is a simple but powerful way to reason about architecture. If a projection matrix has a large kernel, then many input directions are guaranteed to be invisible downstream.

### 10.3 The Four Fundamental Subspaces

For a matrix $A \in \mathbb{R}^{m \times n}$, the four fundamental subspaces are:

1. **column space** $\mathrm{col}(A) \subseteq \mathbb{R}^m$
2. **null space** $\mathrm{null}(A) \subseteq \mathbb{R}^n$
3. **row space** $\mathrm{row}(A) = \mathrm{col}(A^\top) \subseteq \mathbb{R}^n$
4. **left null space** $\mathrm{null}(A^\top) \subseteq \mathbb{R}^m$

They satisfy the orthogonal decompositions

$$
\mathbb{R}^n = \mathrm{row}(A) \oplus \mathrm{null}(A),
$$

$$
\mathbb{R}^m = \mathrm{col}(A) \oplus \mathrm{null}(A^\top).
$$

Dimensional relations:

- $\dim(\mathrm{col}(A)) = \dim(\mathrm{row}(A)) = \mathrm{rank}(A)$
- $\dim(\mathrm{null}(A)) = n - \mathrm{rank}(A)$
- $\dim(\mathrm{null}(A^\top)) = m - \mathrm{rank}(A)$

These decompositions provide a precise vocabulary for information flow:

- the row space describes which input combinations matter
- the null space describes which input directions are lost
- the column space describes what outputs are expressible
- the left null space describes output directions orthogonal to everything the map can produce

```text
Four fundamental subspaces

Domain R^n                          Codomain R^m

[ row(A) | null(A) ] --A--> [ col(A) | 0 ]
                               ^
                               |
                     left-null directions live here,
                     orthogonal to every reachable output
```

### 10.4 Isomorphisms

A linear map $T: V \to W$ is an isomorphism if it is bijective. Then $V$ and $W$ are isomorphic vector spaces.

For finite-dimensional real spaces, the classification is simple:

$$
V \cong W
\quad \Longleftrightarrow \quad
\dim(V) = \dim(W).
$$

So all $n$-dimensional real vector spaces are structurally equivalent to $\mathbb{R}^n$, even if their elements look different.

Important caution: isomorphic does not mean equal. The space of quadratic polynomials and $\mathbb{R}^3$ are not the same set, but they are isomorphic because both have dimension 3.

This is why linear algebra can move freely between concrete coordinates and abstract objects. Once a basis is chosen, every finite-dimensional space looks like ordinary Euclidean coordinate space.

### 10.5 Dual Spaces

The dual space of $V$ is the space of linear functionals on $V$:

$$
V^* = \{f: V \to \mathbb{R} \mid f \text{ is linear}\}.
$$

Elements of the dual space eat vectors and return scalars.

If $V$ is finite-dimensional, then

$$
\dim(V^*) = \dim(V).
$$

With an inner product, every linear functional can be represented as an inner product against a fixed vector:

$$
f(\mathbf{x}) = \langle \mathbf{x}, \mathbf{v} \rangle.
$$

This is the finite-dimensional form of the Riesz representation idea.

In coordinates, row vectors behave like dual vectors:

$$
\mathbf{a}^\top \mathbf{x}
$$

is a linear functional of $\mathbf{x}$.

This perspective is useful in attention. A query can be viewed as defining a linear functional on key space: it "asks" how much each key aligns with a certain direction. The score $q^\top k$ is then the action of a dual vector on a primal vector.

---

## 11. Special Vector Spaces in AI

### 11.1 Embedding Space Geometry

Let $E \in \mathbb{R}^{|V| \times d}$ be a token embedding matrix. Each row $E_t$ is the embedding of token $t$.

This space is where discrete symbols become continuous geometry. The hope is that semantically or syntactically related tokens occupy nearby or meaningfully aligned regions. Word2vec made this idea famous by showing that certain analogical relations behave approximately linearly (Mikolov et al., 2013).

Several geometric questions matter in practice:

- **local similarity**: are related tokens close under cosine or Euclidean distance?
- **directional structure**: do directions correspond to interpretable relations?
- **isotropy**: are embeddings spread broadly or collapsed into a narrow cone?
- **intrinsic dimension**: how many effective degrees of freedom are really used?

Modern contextual embeddings complicate the picture because a token does not have a single representation; it acquires one through context-dependent computation. Even so, the geometric language remains the same. Probing methods ask whether a property is linearly readable from a representation, which is a question about whether a suitable separating hyperplane exists in the embedding space.

### 11.2 Representation Space Through Layers

In a Transformer, each layer maps a sequence representation

$$
X^{(\ell)} \in \mathbb{R}^{n \times d}
$$

to a new one

$$
X^{(\ell+1)} = X^{(\ell)} + f^{(\ell)}(X^{(\ell)}).
$$

The residual connection means layers do not overwrite the representation from scratch. They add a vector-valued update inside a shared ambient space.

This suggests a productive geometric picture:

- the residual stream is a shared communication space
- attention heads read from and write to particular directions or subspaces
- MLP blocks add structured nonlinear updates in that same space
- representation learning is partly the art of organizing information so it can be linearly extracted or manipulated later

Mechanistic interpretability often phrases model behavior in exactly these terms: directions, features, subspaces, superposition, and read/write operations into a shared residual stream.

```text
Residual stream picture

X^(l) --------> [ layer computes update f^(l)(X^(l)) ] ----+
  |                                                        |
  +-------------------------- add --------------------------+
                                                           |
                                                           v
                                                        X^(l+1)
```

### 11.3 Attention Subspaces

Attention projections are linear maps

$$
W_Q, W_K \in \mathbb{R}^{d \times d_k},
\qquad
W_V \in \mathbb{R}^{d \times d_v}.
$$

For token representations $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$,

$$
q = W_Q^\top \mathbf{x},
\qquad
k = W_K^\top \mathbf{y},
\qquad
v = W_V^\top \mathbf{y}.
$$

The raw attention score is

$$
q^\top k
=
\mathbf{x}^\top W_Q W_K^\top \mathbf{y}.
$$

So each head defines a bilinear form on the original embedding space. Because $W_Q W_K^\top$ has rank at most $d_k$, each head uses a low-rank interaction relative to the full $d \times d$ ambient space.

This is one of the cleanest examples of linear-algebraic structure in a large model:

- attention does not compare full vectors directly
- it compares them after moving them into learned low-dimensional subspaces
- different heads can be understood as different learned geometric lenses

```text
Attention subspace flow

x ----W_Q^T----> q
                  \
                   \   score = q^T k
                    \
y ----W_K^T----> k
|
----W_V^T----> v

output = weighted sum of v vectors
```

### 11.4 The Probability Simplex

The probability simplex in $\mathbb{R}^n$ is

$$
\Delta^{n-1}
=
\left\{
\mathbf{p} \in \mathbb{R}^n :
p_i \geq 0,\;
\sum_{i=1}^{n} p_i = 1
\right\}.
$$

It is not a vector space, but it is a highly structured convex set:

- its vertices are the one-hot vectors
- its center is the uniform distribution
- it lies in an affine hyperplane defined by $\sum_i p_i = 1$

Softmax maps logits from $\mathbb{R}^n$ into the interior of this simplex. Sampling, calibration, entropy, KL divergence, and decoding strategies all live on this geometric object.

A deeper geometric view uses the Fisher information metric, which turns the simplex into a curved statistical manifold. That perspective becomes important in information geometry and natural-gradient methods.

```text
Probability simplex in 3 classes

          e2
         /\
        /  \
       / p  \
      /      \
     /________\
   e1          e3

vertices = one-hot distributions
center = uniform distribution
```

### 11.5 Parameter Space

If a model has $p$ trainable parameters, then its parameter space is

$$
\Theta = \mathbb{R}^p.
$$

For contemporary language models, $p$ can be enormous. A 7B model has roughly $7 \times 10^9$ coordinates, and frontier open models reach hundreds of billions (Dubey et al., 2024).

Key objects in parameter space:

- parameter vector $\theta$
- loss function $L(\theta)$
- gradient $\nabla L(\theta)$
- Hessian $\nabla^2 L(\theta)$

This space is flat as a Euclidean manifold, but the loss defined on it can be highly nonconvex. Optimization therefore studies geometry not of the ambient space alone, but of the scalar field drawn on top of it.

Mode connectivity, low-loss valleys, and flat-versus-sharp minima are all geometric statements about subsets of parameter space.

### 11.6 Function Spaces and the Neural Tangent Kernel

A neural network with parameters $\theta$ defines a function

$$
f_\theta : \mathcal{X} \to \mathcal{Y}.
$$

As $\theta$ varies, the model traces a family of functions inside a function space. Locally, infinitesimal parameter changes move the model through the tangent directions

$$
\frac{\partial f_\theta}{\partial \theta_j}.
$$

The neural tangent kernel (NTK) is

$$
K(x, x')
=
\sum_{j=1}^{p}
\frac{\partial f_\theta(x)}{\partial \theta_j}
\frac{\partial f_\theta(x')}{\partial \theta_j}
=
\left\langle
\nabla_\theta f_\theta(x),
\nabla_\theta f_\theta(x')
\right\rangle.
$$

So the NTK is an inner product in parameter-gradient space. Jacot et al. (2018) showed that in the infinite-width limit, gradient descent can be analyzed using this kernel viewpoint, making neural network training resemble kernel regression.

This is a good example of why abstract vector spaces matter. The relevant geometry is no longer primarily in input space or parameter space, but in a function space whose coordinates are themselves derivatives.

---

## 12. Direct Sums and Quotient Spaces

### 12.1 Direct Sum

If $W_1$ and $W_2$ are subspaces of $V$, then

$$
V = W_1 \oplus W_2
$$

means:

1. every vector in $V$ can be written as $\mathbf{w}_1 + \mathbf{w}_2$
2. the decomposition is unique

Equivalently:

- $W_1 + W_2 = V$
- $W_1 \cap W_2 = \{\mathbf{0}\}$

Dimensions add:

$$
\dim(W_1 \oplus W_2) = \dim(W_1) + \dim(W_2).
$$

The most important example in inner-product spaces is

$$
V = W \oplus W^\perp.
$$

Direct sums say that a space can be assembled from independent coordinate subsystems.

```text
Direct sum intuition

      w2
      ^
      |
O---->* 
  w1   \
        \
         v = w1 + w2

Two independent pieces add to one vector.
```

### 12.2 Decomposing R^n

For a matrix $A$, one fundamental decomposition is

$$
\mathbb{R}^n = \mathrm{row}(A) \oplus \mathrm{null}(A).
$$

Likewise,

$$
\mathbb{R}^m = \mathrm{col}(A) \oplus \mathrm{null}(A^\top).
$$

For symmetric matrices, eigenspaces corresponding to distinct eigenvalues are orthogonal, allowing spectral decompositions of the form

$$
\mathbb{R}^n = E_{\lambda_1} \oplus \cdots \oplus E_{\lambda_k}.
$$

In AI, decompositions of hidden spaces into feature subspaces, head subspaces, or signal-versus-noise components are often informal versions of direct-sum thinking.

### 12.3 Quotient Spaces

Given a subspace $W \subseteq V$, the quotient space $V/W$ identifies vectors that differ by an element of $W$:

$$
\mathbf{v}_1 \sim \mathbf{v}_2
\quad \Longleftrightarrow \quad
\mathbf{v}_1 - \mathbf{v}_2 \in W.
$$

The elements of the quotient are cosets

$$
\mathbf{v} + W = \{\mathbf{v} + \mathbf{w} : \mathbf{w} \in W\}.
$$

Dimension behaves cleanly:

$$
\dim(V/W) = \dim(V) - \dim(W).
$$

Conceptually, quotienting collapses all directions in $W$ to zero. If a model is invariant to a set of directions, the meaningful representation can often be thought of as living in a quotient space.

This is the right abstraction for "ignore all variation of type X." It appears in invariant learning, symmetry reduction, and representation factorization.

```text
Quotient-space intuition

Before quotient by W:          After collapsing W-directions:

*----*----*                    [o]   [o]   [o]
*----*----*        ---->          each class = one coset
*----*----*

Points that differ only along W become the same object.
```

### 12.4 Tensor Product

The tensor product $V \otimes W$ is the vector space generated by formal bilinear pairs $v \otimes w$ subject to bilinearity rules. For finite-dimensional spaces,

$$
\dim(V \otimes W) = \dim(V)\dim(W).
$$

If $\{\mathbf{e}_i\}$ is a basis for $V$ and $\{\mathbf{f}_j\}$ is a basis for $W$, then $\{\mathbf{e}_i \otimes \mathbf{f}_j\}$ is a basis for $V \otimes W$.

In coordinates, matrices can be viewed as tensor-product objects:

$$
\mathbb{R}^{m \times n} \cong \mathbb{R}^m \otimes \mathbb{R}^n.
$$

A rank-1 matrix is a simple tensor:

$$
\mathbf{u}\mathbf{v}^\top \leftrightarrow \mathbf{u} \otimes \mathbf{v}.
$$

This is relevant for model compression and LoRA. A low-rank update

$$
\Delta W = BA
$$

is a sum of a small number of rank-1 outer products, so it lives in a low-dimensional tensorially structured subset of matrix space.

```text
Outer product / simple tensor

column u           row v^T            matrix u v^T

[u1]               [v1 v2 v3]         [u1v1 u1v2 u1v3]
[u2]       x                          [u2v1 u2v2 u2v3]

One column times one row creates a rank-1 matrix.
```

---

## 13. Norms, Regularization, and Geometry

### 13.1 L2 Regularization as a Geometric Constraint

L2 regularization adds a penalty

$$
\lambda \|\theta\|_2^2
$$

to the loss. Geometrically, this prefers solutions near the origin and can be viewed through the constrained problem

$$
\min_\theta L(\theta)
\quad \text{subject to} \quad
\|\theta\|_2 \leq r.
$$

The L2 ball is smooth and rotationally symmetric. As a result, L2 regularization shrinks parameters continuously rather than forcing exact zeros.

This is the modern descendant of ridge regression (Hoerl and Kennard, 1970). In neural networks it appears as weight decay.

```text
L2 constraint set in 2D

      ____ 
    /      \
   |   .    |
    \______/

smooth boundary -> shrink smoothly
```

### 13.2 L1 Regularization and Sparsity

L1 regularization uses

$$
\lambda \|\theta\|_1.
$$

Its geometry is different. The L1 ball has corners along coordinate axes, so when an objective first touches the constraint boundary, it is disproportionately likely to do so at a sparse point.

That geometric fact explains why L1 promotes sparsity (Tibshirani, 1996). The corresponding proximal operator is soft thresholding:

$$
\mathrm{prox}_{\lambda \|\cdot\|_1}(x_i)
=
\mathrm{sign}(x_i)\max(0, |x_i| - \lambda).
$$

Small coordinates are driven exactly to zero.

This matters for:

- feature selection
- sparse coding
- pruning
- sparse MoE routing and sparse attention variants

```text
L1 constraint set in 2D

        /\
       /  \
       \  /
        \/

corners on coordinate axes -> sparse solutions are favored
```

### 13.3 Spectral Regularization

For a matrix-valued parameter $W$, several geometry-aware penalties are common.

**Frobenius penalty**

$$
\|W\|_F^2
$$

which is entrywise L2 regularization.

**Spectral norm control**

$$
\|W\|_2 = \sigma_{\max}(W)
$$

which bounds the largest one-step amplification of the layer.

**Nuclear norm**

$$
\|W\|_*
$$

which promotes low rank.

Spectral normalization rescales a weight matrix by an estimate of its top singular value so that the layer has bounded operator norm (Miyato et al., 2018). This is a geometric way of controlling Lipschitz constants and stabilizing training.

```text
What spectral control means geometrically

unit ball              after W                after spectral normalization

   ()                    (------)                  (----)
                         long axis = sigma_max     longest stretch capped

Spectral norm controls the biggest stretching direction.
```

### 13.4 The Geometry of Gradient Descent

The gradient

$$
\nabla L(\theta)
$$

is a vector in parameter space pointing in the direction of steepest increase of the loss under the Euclidean metric. Gradient descent updates by

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t).
$$

Several geometric facts follow:

- the gradient is orthogonal to level sets of $L$
- ill-conditioned curvature creates narrow valleys and zig-zagging updates
- preconditioning changes the metric of parameter space to make descent more isotropic

If the Hessian has eigenvalues $\lambda_{\max}$ and $\lambda_{\min}$ with large ratio

$$
\kappa = \frac{\lambda_{\max}}{\lambda_{\min}},
$$

then the local landscape is elongated and naive gradient descent is inefficient.

Adaptive methods and natural gradient can be understood as changes of geometry. Natural gradient replaces the Euclidean metric with the Fisher information metric, aligning steepest descent with the geometry of the model's predictive distribution (Amari, 1998).

This is the right mental model: optimization is not just arithmetic on parameters; it is motion through a geometric space whose metric determines what "steepest" even means.

```text
Gradient descent in a narrow valley

\                /
 \   x ->      /
  \           /
  /   <- x   \
 /           \
/_____________\

Poor conditioning makes updates zig-zag.
```

---

## 14. Common Mistakes

| Mistake | Why it is wrong | Better statement |
| --- | --- | --- |
| "A set closed under addition is automatically a subspace." | It must also contain the zero vector and be closed under scalar multiplication. | Check all subspace conditions, not just one. |
| "Independent vectors are orthogonal." | Orthogonality implies independence for nonzero vectors, but independence does not imply orthogonality. | Use Gram-Schmidt when you want orthogonality. |
| "The zero vector is not in a proper subspace." | Every subspace must contain the zero vector. | If $\mathbf{0} \notin W$, then $W$ is not a subspace. |
| "Cosine similarity tells you Euclidean closeness." | High cosine can coexist with large Euclidean distance when norms differ a lot. | Cosine measures angle; Euclidean distance also depends on magnitude. |
| "All vectors in high dimension are similar." | Random high-dimensional vectors are typically nearly orthogonal, not highly aligned. | Meaningful similarity is rare relative to the random baseline. |
| "A spanning set has as many vectors as the dimension." | Spanning sets can be redundant. Dimension counts basis vectors, not all listed vectors. | Reduce to a linearly independent spanning set to find dimension. |
| "The simplex is a vector space because it contains vectors." | It is not closed under arbitrary scaling or addition. | The simplex is a convex subset of a vector space. |
| "Projection leaves every vector unchanged." | It leaves only vectors already in the target subspace unchanged. | Projection keeps the in-subspace component and removes the orthogonal remainder. |
| "Two spaces of the same dimension are equal." | They are isomorphic, not literally the same set. | Same dimension means same linear structure up to coordinate relabeling. |
| "Null$(A)$ and Null$(A^\top)$ are the same thing." | They live in different ambient spaces unless $A$ is square and special. | Right null space and left null space are distinct objects. |

---

## 15. Exercises

These are designed to force both symbolic fluency and geometric interpretation.

1. **Vector space verification.** Determine whether each set is a vector space under the stated operations. If it is, justify all axioms; if not, identify a failing axiom.
   - $(a)$ $\{(x,y) \in \mathbb{R}^2 : x \geq 0\}$ with standard operations
   - $(b)$ $\{f:\mathbb{R}\to\mathbb{R} : f(0)=0\}$ with pointwise operations
   - $(c)$ $\{A \in \mathbb{R}^{2 \times 2} : \det(A)=1\}$ with matrix addition
   - $(d)$ $\{\mathbf{p} \in \mathbb{R}^3 : p_1+p_2+p_3=1\}$ with componentwise addition

2. **Subspaces and dimension.**
   - $(a)$ Show that $\{(x,y,z)\in\mathbb{R}^3 : x+2y-z=0\}$ is a subspace, find a basis, and compute its dimension.
   - $(b)$ Find the dimension of the intersection of the planes $x+y+z=0$ and $x-y+2z=0$.
   - $(c)$ Let $W=\mathrm{span}\{(1,1,0),(1,0,1),(0,1,-1)\}$. Find a basis for $W$ and determine whether $W=\mathbb{R}^3$.

3. **Inner products and orthogonality.** Let $\mathbf{u}=(1,2,-1)$ and $\mathbf{v}=(3,0,1)$.
   - $(a)$ Compute $\langle \mathbf{u}, \mathbf{v} \rangle$.
   - $(b)$ Compute $\|\mathbf{u}\|_2$ and $\|\mathbf{v}\|_2$.
   - $(c)$ Find the angle between the vectors.
   - $(d)$ Compute the projection of $\mathbf{u}$ onto $\mathbf{v}$.
   - $(e)$ Verify the Pythagorean decomposition.
   - $(f)$ Apply Gram-Schmidt to produce an orthonormal basis for $\mathrm{span}\{\mathbf{u}, \mathbf{v}\}$.

4. **Rank-nullity.** For

$$
A =
\begin{pmatrix}
1 & 2 & 3 \\
2 & 4 & 6 \\
1 & 1 & 2
\end{pmatrix},
$$

find:
   - $(a)$ a basis for $\mathrm{null}(A)$
   - $(b)$ a basis for $\mathrm{col}(A)$
   - $(c)$ a basis for $\mathrm{row}(A)$
   - $(d)$ the rank and nullity, verifying rank-nullity
   - $(e)$ a nonzero vector in $\mathrm{null}(A^\top)$

5. **Projection matrices.**
   - $(a)$ Compute the orthogonal projector onto the line spanned by $(1,2,2)^\top$.
   - $(b)$ Verify $P^2=P$ and $P^\top=P$.
   - $(c)$ Compute $I-P$ and identify its target subspace.
   - $(d)$ For

$$
A =
\begin{pmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{pmatrix},
$$

compute $P_A = A(A^\top A)^{-1}A^\top$ and interpret the result geometrically.

6. **High-dimensional geometry.**
   - $(a)$ Suppose $\mathbf{x}, \mathbf{y}$ are independent random unit vectors in $\mathbb{R}^{100}$. What are the expected value and approximate variance of $\langle \mathbf{x}, \mathbf{y} \rangle$?
   - $(b)$ For 500 points and $\varepsilon=0.1$, estimate a Johnson-Lindenstrauss target dimension $k$ up to constants.
   - $(c)$ Show that all distinct standard basis vectors in $\mathbb{R}^d$ are at Euclidean distance $\sqrt{2}$ from one another.

7. **Attention geometry.** Let $W_Q \in \mathbb{R}^{512 \times 64}$.
   - $(a)$ What is the maximum possible rank of $W_Q$?
   - $(b)$ What is the dimension of its null space if the rank is maximal?
   - $(c)$ What kinds of input directions are lost under this projection?
   - $(d)$ Why is $W_Q W_K^\top$ low rank?

8. **Norms and distances.**
   - $(a)$ For $\mathbf{v}=(3,-4,0,1)^\top$, compute $\|\mathbf{v}\|_1$, $\|\mathbf{v}\|_2$, and $\|\mathbf{v}\|_\infty$.
   - $(b)$ Verify $\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1$.
   - $(c)$ For

$$
A =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix},
$$

compute $\|A\|_F$ and $\|A\|_2$.

9. **Quotient-space thinking.** Suppose a representation is unchanged when you add any vector from a subspace $W$. Explain why the meaningful information lives in $V/W$ rather than in $V$ itself.

10. **Regularization geometry.** Sketch or describe the L1 and L2 unit balls in $\mathbb{R}^2$, then explain geometrically why L1 tends to produce sparse solutions and L2 typically does not.

---

## 16. Why This Matters for AI

| Aspect | Why vectors and spaces matter |
| --- | --- |
| Embeddings | Every token, position, and feature is represented as a vector in a learned space. |
| Attention | Attention is built from inner products, low-rank projections, and weighted sums. |
| Representation learning | Features live in subspaces, not just in scalar coordinates. |
| Compression | Low-rank methods, adapters, and LoRA rely on dimension, span, and rank. |
| Optimization | Gradients are vectors in parameter space, and regularization is geometric constraint design. |
| Generalization theory | RKHS, NTK, margin bounds, and norm-based complexity all use vector-space language. |
| Numerical stability | Orthogonality, conditioning, and norm control determine whether training remains stable. |
| Interpretability | Linear probes, concept vectors, activation steering, and superposition are all geometric analyses. |
| Retrieval and search | Similarity search depends on metrics, angle, and concentration in high dimensions. |
| Probabilistic modeling | Logits live in vector spaces and probabilities live in simplices embedded in them. |

The short version is that vectors and spaces are not one topic among many. They are the common substrate of model architecture, learning dynamics, representation quality, and theoretical analysis.

---

## 17. Conceptual Bridge

Scalars are the atoms of computation. Vectors are organized collections of scalars. Vector spaces tell us which collections can be added and scaled consistently. Norms and inner products add geometry. Linear maps move us between spaces. Convexity constrains what combinations are allowed. High-dimensional geometry tells us why intuition breaks and why large models can still work.

That is the bridge to the next chapters:

```text
sets and functions
  -> vectors and spaces
  -> matrices and linear maps
  -> orthogonality, eigenvalues, and SVD
  -> optimization, probability, and information
  -> neural networks and transformers
```

If this chapter gives you one durable mental model, let it be this: deep learning is geometry performed by linear maps inside structured high-dimensional spaces, with nonlinearities deciding which geometric regions remain active.

---

## References

### Core Linear Algebra and Geometry

- Sheldon Axler, *Linear Algebra Done Right* (official site): [https://linear.axler.net/](https://linear.axler.net/)
- Roman Vershynin, *High-Dimensional Probability* (author page): [https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html](https://www.math.uci.edu/~rvershyn/papers/HDP-book/HDP-book.html)
- William B. Johnson and Joram Lindenstrauss (1984), "Extensions of Lipschitz mappings into a Hilbert space": [https://web.stanford.edu/class/cs114/readings/JL-Johnson.pdf](https://web.stanford.edu/class/cs114/readings/JL-Johnson.pdf)
- Sanjoy Dasgupta and Anupam Gupta (2003), "An Elementary Proof of the Johnson-Lindenstrauss Lemma": [https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=bcd7068ee41305cb4dc5f133379cc22801cf744d](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=bcd7068ee41305cb4dc5f133379cc22801cf744d)
- Nachman Aronszajn (1950), "Theory of Reproducing Kernels": [https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/](https://www.ams.org/journals/tran/1950-068-03/S0002-9947-1950-0051437-7/)

### AI and Representation Geometry

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean (2013), "Efficient Estimation of Word Representations in Vector Space": [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
- Ashish Vaswani et al. (2017), "Attention Is All You Need": [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Kawin Ethayarajh (2019), "How Contextual Are Contextualized Word Representations?": [https://aclanthology.org/D19-1006/](https://aclanthology.org/D19-1006/)
- Abhimanyu Dubey et al. (2024), "The Llama 3 Herd of Models": [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

### Low Rank, Kernels, and Optimization Geometry

- Edward J. Hu et al. (2022), "LoRA: Low-Rank Adaptation of Large Language Models": [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Arthur Jacot, Franck Gabriel, and Clement Hongler (2018), "Neural Tangent Kernel: Convergence and Generalization in Neural Networks": [https://proceedings.mlr.press/v80/jacot18a.html](https://proceedings.mlr.press/v80/jacot18a.html)
- Emmanuel J. Candes and Benjamin Recht (2009), "Exact Matrix Completion via Convex Optimization": [https://arxiv.org/abs/0805.4471](https://arxiv.org/abs/0805.4471)
- Takeru Miyato et al. (2018), "Spectral Normalization for Generative Adversarial Networks": [https://openreview.net/forum?id=B1QRgziT-](https://openreview.net/forum?id=B1QRgziT-)
- Andrew M. Saxe, James L. McClelland, and Surya Ganguli (2014), "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks": [https://arxiv.org/abs/1312.6120](https://arxiv.org/abs/1312.6120)

### Regularization and Information Geometry

- Robert Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso": [https://www.jstor.org/stable/2346178](https://www.jstor.org/stable/2346178)
- Arthur E. Hoerl and Robert W. Kennard (1970), "Ridge Regression: Biased Estimation for Nonorthogonal Problems." Original Technometrics paper.
- Shun-ichi Amari (1998), "Natural Gradient Works Efficiently in Learning": [https://papers.nips.cc/paper_files/paper/1998/hash/4b22aa9464f5a138cb5c51cab4093b7b-Abstract.html](https://papers.nips.cc/paper_files/paper/1998/hash/4b22aa9464f5a138cb5c51cab4093b7b-Abstract.html)

---

[Back to Home](../../README.md) | [Next: Matrix Operations](../02-Matrix-Operations/notes.md)
