# Vectors and Spaces

## Overview

Vectors are the **fundamental building blocks of machine learning**. Every piece of data—images, text, audio, tabular data—is represented as vectors before being processed by ML algorithms. This chapter provides complete coverage of vectors and vector spaces, with all the theory, examples, and intuitions you need.

## Prerequisites

- Basic algebra (solving equations)
- Coordinate geometry basics

## Learning Objectives

By the end of this section, you will:
- Understand vectors from geometric, algebraic, and abstract perspectives
- Master all vector operations (addition, scaling, dot product, norms)
- Compute angles, projections, and similarities between vectors
- Understand vector spaces, linear independence, span, and basis
- Apply vector concepts to ML problems

---

## Table of Contents

1. [What is a Vector?](#what-is-a-vector)
2. [Vector Operations](#vector-operations)
3. [Vector Norms (Length)](#vector-norms-length)
4. [Angles and Similarity](#angles-and-similarity)
5. [Vector Projection](#vector-projection)
6. [Orthogonality](#orthogonality)
7. [Vector Spaces](#vector-spaces)
8. [Linear Independence, Span, and Basis](#linear-independence-span-and-basis)
9. [Gram-Schmidt Orthogonalization](#gram-schmidt-orthogonalization)
10. [ML Applications](#ml-applications)
11. [Complete Worked Examples](#complete-worked-examples)

---

## What is a Vector?

A vector can be understood from three perspectives:

### 1. Geometric View (Physics)

An arrow with **magnitude** (length) and **direction**.

```
                    ↑ y
                    │
                    │    → v = (3, 2)
                    │   ╱
                  2 │  ╱
                    │ ╱ 
                    │╱  θ = 33.7°
        ────────────┼────────────→ x
                    │     3
```

- **Magnitude**: $\|v\| = \sqrt{3^2 + 2^2} = \sqrt{13} \approx 3.61$
- **Direction**: angle θ from x-axis

### 2. Coordinate View (Computer Science)

An **ordered list** of numbers (components/elements):

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} = (v_1, v_2, \ldots, v_n)$$

**Examples:**
```
2D vector: v = [3, 2]          → Point in a plane
3D vector: v = [1, 2, 3]       → Point in 3D space  
784D vector: v = [p₁, p₂, ..., p₇₈₄]  → Flattened 28×28 image
```

### 3. Abstract View (Mathematics)

An element of a **vector space** — any object that can be added and scaled according to certain rules (axioms).

**Why This Matters:** Functions, matrices, and even signals can be treated as vectors!

### Vector Notation Reference

| Notation | Meaning |
|----------|---------|
| $\mathbf{v}$, $\vec{v}$, $\boldsymbol{v}$ | Vector (bold, arrow, or bold italic) |
| $v_i$ or $[\mathbf{v}]_i$ | $i$-th component (1-indexed or 0-indexed) |
| $\mathbf{v} \in \mathbb{R}^n$ | $n$-dimensional real vector |
| $\mathbf{0}$ | Zero vector: $[0, 0, \ldots, 0]$ |
| $\mathbf{e}_i$ | Standard basis vector ($i$-th position is 1, rest 0) |
| $\mathbf{v}^T$ | Transpose (column → row) |
| $\|\mathbf{v}\|$ | Magnitude/norm of vector |
| $\hat{\mathbf{v}}$ | Unit vector in direction of $\mathbf{v}$ |

---

## Vector Operations

### 1. Vector Addition

Add vectors **component-wise**:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

**Example:**
$$\begin{bmatrix} 3 \\ 1 \end{bmatrix} + \begin{bmatrix} 1 \\ 4 \end{bmatrix} = \begin{bmatrix} 3+1 \\ 1+4 \end{bmatrix} = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$$

**Geometric Interpretation — Two Methods:**

```
METHOD 1: Tip-to-Tail              METHOD 2: Parallelogram
                                   
       u + v                              u + v
         ↗                               ↗
        ╱                               ╱ │
       ╱ v                             ╱  │ v
      ╱                               ╱   │
     ╱                               ╱    │
    ↗───────→                       ↗─────┘
    u                               u
```

**Properties of Vector Addition:**

| Property | Formula | Meaning |
|----------|---------|---------|
| Commutative | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ | Order doesn't matter |
| Associative | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ | Grouping doesn't matter |
| Identity | $\mathbf{v} + \mathbf{0} = \mathbf{v}$ | Adding zero does nothing |
| Inverse | $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ | Every vector has a negative |

### 2. Scalar Multiplication

Multiply **each component** by a scalar (real number):

$$c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

**Example:**
$$3 \cdot \begin{bmatrix} 2 \\ -1 \end{bmatrix} = \begin{bmatrix} 6 \\ -3 \end{bmatrix}$$

**Geometric Interpretation:**

```
c > 1: Stretches          0 < c < 1: Shrinks        c < 0: Reverses
                                                    
    2v                        0.5v                     -v
────────→                   ───→                    ←───
   v                          v                        v
────→                       ────→                   ────→
```

| Scalar Value | Effect |
|--------------|--------|
| $c > 1$ | Stretches (longer) |
| $c = 1$ | No change |
| $0 < c < 1$ | Shrinks (shorter) |
| $c = 0$ | Becomes zero vector |
| $c < 0$ | Reverses direction + scales |

### 3. Vector Subtraction

Subtraction is adding the negative:

$$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v}) = \begin{bmatrix} u_1 - v_1 \\ u_2 - v_2 \end{bmatrix}$$

**Geometric Meaning:** The vector from $\mathbf{v}$ to $\mathbf{u}$.

```
          u
         ↗
        ╱
       ╱  u - v (points from v to u)
      ╱   ↗
     ╱   ╱
    ↗───╱
    v
```

### 4. Linear Combination

A **linear combination** of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ with scalars $c_1, c_2, \ldots, c_k$:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$

**Example:** Express $\begin{bmatrix} 7 \\ 4 \end{bmatrix}$ as a linear combination of $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$:

$$7 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 4 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 7 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 4 \end{bmatrix} = \begin{bmatrix} 7 \\ 4 \end{bmatrix} \checkmark$$

**ML Connection:** A neural network layer computes a linear combination of inputs plus bias!

---

## Dot Product (Inner Product)

The **dot product** is the most important vector operation in ML.

### Definition (Algebraic)

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

**Example:**
$$\begin{bmatrix} 3 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} 2 \\ 5 \end{bmatrix} = 3 \times 2 + 4 \times 5 = 6 + 20 = 26$$

### Definition (Geometric)

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between the vectors.

### Geometric Interpretation

The dot product measures **how much two vectors point in the same direction**:

```
Same direction (θ = 0°)      Perpendicular (θ = 90°)    Opposite (θ = 180°)
cos(0°) = 1                  cos(90°) = 0               cos(180°) = -1

    u                            u                          u
    ↗                            ↗                          ↗
   ╱                            │                          ╲
  ╱                             │ v                         ╲ v
 ↗                              ↓                            ↘
 v                              
                              
u · v > 0 (max)              u · v = 0                   u · v < 0 (min)
```

### Properties of Dot Product

| Property | Formula | 
|----------|---------|
| Commutative | $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$ |
| Distributive | $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$ |
| Scalar | $(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})$ |
| Self | $\mathbf{v} \cdot \mathbf{v} = \|\mathbf{v}\|^2$ |
| Zero | $\mathbf{v} \cdot \mathbf{0} = 0$ |

### Why Dot Product Matters in ML

| Application | How Dot Product is Used |
|-------------|------------------------|
| **Linear Layers** | $y = \mathbf{w}^T\mathbf{x} + b$ computes dot product of weights and inputs |
| **Similarity** | Higher dot product = more similar vectors |
| **Attention** | Query-Key dot products determine attention weights |
| **Projections** | Project vectors onto subspaces |
| **Activation** | Input to neurons is weighted sum (dot product) |

---

## Vector Norms (Length)

A **norm** measures the "size" or "length" of a vector.

### L2 Norm (Euclidean)

The most common norm — the straight-line distance from origin:

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Example:**
$$\left\|\begin{bmatrix} 3 \\ 4 \end{bmatrix}\right\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**Connection to Dot Product:**
$$\|\mathbf{v}\|_2 = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

### L1 Norm (Manhattan)

Sum of absolute values — "city block" distance:

$$\|\mathbf{v}\|_1 = \sum_{i=1}^{n} |v_i| = |v_1| + |v_2| + \cdots + |v_n|$$

**Example:**
$$\left\|\begin{bmatrix} 3 \\ -4 \end{bmatrix}\right\|_1 = |3| + |-4| = 3 + 4 = 7$$

### L∞ Norm (Max/Chebyshev)

Maximum absolute component:

$$\|\mathbf{v}\|_\infty = \max_i |v_i|$$

**Example:**
$$\left\|\begin{bmatrix} 3 \\ -4 \\ 2 \end{bmatrix}\right\|_\infty = \max(|3|, |-4|, |2|) = 4$$

### General Lp Norm

$$\|\mathbf{v}\|_p = \left(\sum_{i=1}^{n} |v_i|^p\right)^{1/p}$$

### Visual Comparison: Unit Balls

The "unit ball" for a norm is all vectors with norm ≤ 1:

```
         L1 (Diamond)         L2 (Circle)          L∞ (Square)
            
              ↑                    ↑                    ↑
              │                    │               ┌────┼────┐
              ◇                    │               │    │    │
             ╱│╲               ╭───┼───╮           │    │    │
        ────◇─┼─◇────      ────│───┼───│────   ────┼────┼────┼────
             ╲│╱               ╰───┼───╯           │    │    │
              ◇                    │               │    │    │
              │                    │               └────┼────┘
```

### Norms in Machine Learning

| Norm | ML Application | Why? |
|------|----------------|------|
| **L1** | Lasso regression, sparse models | Encourages zeros (sparsity) |
| **L2** | Ridge regression, weight decay | Encourages small weights |
| **L∞** | Adversarial robustness | Bounds max perturbation |

### Unit Vectors

A **unit vector** has norm 1. To create a unit vector in the direction of $\mathbf{v}$:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Example:**
$$\hat{\mathbf{v}} = \frac{\begin{bmatrix} 3 \\ 4 \end{bmatrix}}{5} = \begin{bmatrix} 0.6 \\ 0.8 \end{bmatrix}$$

Verify: $\sqrt{0.6^2 + 0.8^2} = \sqrt{0.36 + 0.64} = 1$ ✓

---

## Angles and Similarity

### Computing the Angle Between Vectors

From the geometric dot product formula:

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

$$\theta = \arccos\left(\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}\right)$$

**Example:** Find the angle between $\mathbf{u} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$

1. Dot product: $\mathbf{u} \cdot \mathbf{v} = 1 \times 1 + 0 \times 1 = 1$
2. Norms: $\|\mathbf{u}\| = 1$, $\|\mathbf{v}\| = \sqrt{2}$
3. $\cos\theta = \frac{1}{1 \times \sqrt{2}} = \frac{1}{\sqrt{2}} = 0.707$
4. $\theta = \arccos(0.707) = 45°$

### Cosine Similarity

$$\text{cosine\_similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos\theta$$

**Range:** $[-1, 1]$

| Value | Meaning |
|-------|---------|
| 1 | Identical direction (θ = 0°) |
| 0 | Perpendicular (θ = 90°) |
| -1 | Opposite direction (θ = 180°) |

**Why Cosine Similarity?**
- **Magnitude-invariant:** Only cares about direction
- Comparing $[1, 2]$ and $[100, 200]$ gives similarity = 1 (same direction)
- Essential for comparing documents of different lengths, embeddings, etc.

### Euclidean Distance

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$

**Example:**
$$d\left(\begin{bmatrix} 1 \\ 2 \end{bmatrix}, \begin{bmatrix} 4 \\ 6 \end{bmatrix}\right) = \sqrt{(1-4)^2 + (2-6)^2} = \sqrt{9 + 16} = 5$$

### Similarity Measures Comparison

| Measure | Formula | Range | Best For |
|---------|---------|-------|----------|
| Dot Product | $\mathbf{u}^T\mathbf{v}$ | $(-\infty, +\infty)$ | When magnitude matters |
| Cosine Similarity | $\frac{\mathbf{u}^T\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | $[-1, 1]$ | When only direction matters |
| Euclidean Distance | $\|\mathbf{u} - \mathbf{v}\|_2$ | $[0, +\infty)$ | KNN, clustering (smaller = more similar) |
| Manhattan Distance | $\|\mathbf{u} - \mathbf{v}\|_1$ | $[0, +\infty)$ | Grid-based problems |

---

## Vector Projection

Projecting one vector onto another is fundamental for regression, PCA, and more.

### Scalar Projection

The **scalar projection** (or component) of $\mathbf{u}$ onto $\mathbf{v}$:

$$\text{comp}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|} = \|\mathbf{u}\| \cos\theta$$

This is the "length" of $\mathbf{u}$ in the direction of $\mathbf{v}$.

### Vector Projection

The **vector projection** of $\mathbf{u}$ onto $\mathbf{v}$:

$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\mathbf{v} \cdot \mathbf{v}} \mathbf{v} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$$

**Alternative form using unit vector:**

$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = (\mathbf{u} \cdot \hat{\mathbf{v}}) \hat{\mathbf{v}}$$

### Geometric Picture

```
                u
               ↗
              ╱│
             ╱ │ perpendicular
            ╱  │ component
           ╱   │
          ╱    ↓
    ─────●━━━━━━━━━●─────────→ v
         ↑         ↑
         └─────────┘
          projection
         (parallel to v)
```

The vector $\mathbf{u}$ decomposes into:
- **Parallel component** (projection): $\text{proj}_{\mathbf{v}}(\mathbf{u})$
- **Perpendicular component**: $\mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u})$

### Worked Example

Project $\mathbf{u} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ onto $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ (the x-axis):

**Step 1:** Compute dot products
- $\mathbf{u} \cdot \mathbf{v} = 3 \times 1 + 4 \times 0 = 3$
- $\mathbf{v} \cdot \mathbf{v} = 1 \times 1 + 0 \times 0 = 1$

**Step 2:** Apply formula
$$\text{proj}_{\mathbf{v}}(\mathbf{u}) = \frac{3}{1} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$$

**Step 3:** Perpendicular component
$$\mathbf{u} - \text{proj}_{\mathbf{v}}(\mathbf{u}) = \begin{bmatrix} 3 \\ 4 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 4 \end{bmatrix}$$

**Verify:** Check perpendicularity
$$\begin{bmatrix} 3 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 4 \end{bmatrix} = 0 \checkmark$$

---

## Orthogonality

### Definition

Two vectors are **orthogonal** (perpendicular) if their dot product is zero:

$$\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0$$

**Examples:**
- $\begin{bmatrix} 1 \\ 0 \end{bmatrix} \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix} = 0$ ✓ (standard basis vectors)
- $\begin{bmatrix} 3 \\ 4 \end{bmatrix} \cdot \begin{bmatrix} -4 \\ 3 \end{bmatrix} = -12 + 12 = 0$ ✓

### Orthonormal Vectors

Vectors are **orthonormal** if they are:
1. **Orthogonal:** pairwise dot product = 0
2. **Normalized:** each has unit length

**Standard Basis in $\mathbb{R}^3$:**

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

### Why Orthogonality Matters

1. **Independence:** Orthogonal features provide non-redundant information
2. **Computation:** Orthogonal matrices have easy inverses: $Q^{-1} = Q^T$
3. **Numerical Stability:** Orthogonal transformations preserve lengths
4. **PCA:** Principal components are orthogonal
5. **Fourier Transform:** Basis functions are orthogonal

---

## Vector Spaces

A **vector space** is a collection of objects (vectors) where addition and scalar multiplication are defined and follow specific rules.

### Formal Definition

A **vector space** $V$ over a field $F$ (usually $\mathbb{R}$) is a set with two operations:
1. **Vector addition:** $+: V \times V \to V$
2. **Scalar multiplication:** $\cdot: F \times V \to V$

Satisfying the **8 axioms:**

| # | Axiom | Formula |
|---|-------|---------|
| 1 | Associativity of + | $\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}$ |
| 2 | Commutativity of + | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 3 | Additive identity | $\exists \mathbf{0}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 4 | Additive inverse | $\exists (-\mathbf{v}): \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| 5 | Compatibility of · | $a(b\mathbf{v}) = (ab)\mathbf{v}$ |
| 6 | Scalar identity | $1\mathbf{v} = \mathbf{v}$ |
| 7 | Distributivity (vector) | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| 8 | Distributivity (scalar) | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |

### Examples of Vector Spaces

| Space | Elements | Addition | Scalar Mult | ML Example |
|-------|----------|----------|-------------|------------|
| $\mathbb{R}^n$ | n-tuples of reals | component-wise | component-wise | Feature vectors |
| $\mathbb{R}^{m \times n}$ | m×n matrices | element-wise | element-wise | Weight matrices |
| $P_n$ | Polynomials deg ≤ n | poly addition | coefficient mult | Polynomial features |
| $C[a,b]$ | Continuous functions | function addition | function scaling | Signal processing |

### Subspaces

A **subspace** W of V is a subset that is itself a vector space.

**Requirements:**
1. Contains zero vector: $\mathbf{0} \in W$
2. Closed under addition: $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$
3. Closed under scaling: $\mathbf{v} \in W, c \in \mathbb{R} \Rightarrow c\mathbf{v} \in W$

**Example:** The xy-plane is a subspace of $\mathbb{R}^3$:
$$W = \left\{ \begin{bmatrix} x \\ y \\ 0 \end{bmatrix} : x, y \in \mathbb{R} \right\}$$

---

## Linear Independence, Span, and Basis

### Linear Independence

Vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ are **linearly independent** if the only solution to:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

is $c_1 = c_2 = \cdots = c_k = 0$.

**Intuition:** No vector can be written as a combination of the others.

**Example — Independent:**
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Neither is a multiple of the other ✓

**Example — Dependent:**
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$

$\mathbf{v}_2 = 2\mathbf{v}_1$, so $2\mathbf{v}_1 - \mathbf{v}_2 = \mathbf{0}$ has non-trivial solution ✗

**How to Check:** Form matrix and compute rank. If rank = number of vectors → independent.

### Span

The **span** of vectors is the set of all their linear combinations:

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

**Example:**
$$\text{span}\left(\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}\right) = \mathbb{R}^2$$

All 2D vectors can be reached!

### Basis

A **basis** for a vector space V is a set of vectors that:
1. **Spans V:** Every vector in V can be expressed as a linear combination
2. **Is linearly independent:** No redundant vectors

**Standard Basis for $\mathbb{R}^n$:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ \vdots \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ \vdots \\ 0 \end{bmatrix}, \quad \ldots, \quad \mathbf{e}_n = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 1 \end{bmatrix}$$

**Key Property:** The number of vectors in any basis is the same — this is the **dimension**.

### Dimension

$$\dim(V) = \text{number of vectors in any basis}$$

**Examples:**
- $\dim(\mathbb{R}^n) = n$
- $\dim(\mathbb{R}^{m \times n}) = mn$
- $\dim(P_n) = n + 1$ (polynomials of degree ≤ n)

### ML Connection

- **Feature space dimension** = number of features
- **Redundant features** = linearly dependent → don't add information
- **PCA** finds a lower-dimensional basis that captures most variance

---

## Gram-Schmidt Orthogonalization

Convert any linearly independent set into an **orthonormal basis**.

### Algorithm

Given vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$:

**Step 1:** Start with first vector
$$\mathbf{u}_1 = \mathbf{v}_1$$

**Step 2:** For each subsequent vector, subtract projections onto previous vectors
$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \text{proj}_{\mathbf{u}_j}(\mathbf{v}_k)$$

**Step 3:** Normalize
$$\mathbf{e}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

### Worked Example

Orthogonalize $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$

**Step 1:** First orthogonal vector
$$\mathbf{u}_1 = \mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$

**Step 2:** Second orthogonal vector
$$\text{proj}_{\mathbf{u}_1}(\mathbf{v}_2) = \frac{\mathbf{v}_2 \cdot \mathbf{u}_1}{\mathbf{u}_1 \cdot \mathbf{u}_1} \mathbf{u}_1 = \frac{1}{2} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.5 \\ 0 \end{bmatrix}$$

$$\mathbf{u}_2 = \mathbf{v}_2 - \text{proj}_{\mathbf{u}_1}(\mathbf{v}_2) = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \begin{bmatrix} 0.5 \\ 0.5 \\ 0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ -0.5 \\ 1 \end{bmatrix}$$

**Step 3:** Normalize
$$\mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$

$$\mathbf{e}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|} = \frac{1}{\sqrt{1.5}} \begin{bmatrix} 0.5 \\ -0.5 \\ 1 \end{bmatrix}$$

**Verify:** $\mathbf{e}_1 \cdot \mathbf{e}_2 = \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{1.5}} (0.5 - 0.5 + 0) = 0$ ✓

---

## ML Applications

### 1. Feature Vectors

Every data point is a vector where each dimension is a feature:

```python
# Image (28×28 pixels, flattened)
image_vector = [p1, p2, ..., p784]  # ∈ ℝ^784

# Word embedding
word_vector = [0.2, -0.5, 0.8, ...]  # ∈ ℝ^300 (Word2Vec, GloVe)

# Customer profile
customer = [age, income, score, ...]  # ∈ ℝ^n
```

### 2. Neural Network Layers

A linear layer is a weighted sum (dot product):

$$y = \mathbf{w}^T\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

```python
# Single neuron
output = np.dot(weights, inputs) + bias
```

### 3. Word Embeddings and Analogies

Word embeddings capture semantic relationships as vector arithmetic:

```
king - man + woman ≈ queen

v_king - v_man + v_woman ≈ v_queen
```

### 4. Similarity Search

Finding similar items using vector similarity:

```python
# Find most similar items to query
similarities = [cosine_similarity(query, item) for item in database]
most_similar = items[np.argmax(similarities)]
```

### 5. Regularization

L1 and L2 norms control model complexity:

$$L_{\text{Ridge}} = \text{MSE} + \lambda \|\mathbf{w}\|_2^2$$
$$L_{\text{Lasso}} = \text{MSE} + \lambda \|\mathbf{w}\|_1$$

### 6. Distance-Based Algorithms

| Algorithm | Uses Vector Distance For |
|-----------|-------------------------|
| **k-NN** | Finding nearest neighbors |
| **K-means** | Assigning points to clusters |
| **DBSCAN** | Density-based clustering |
| **t-SNE/UMAP** | Dimensionality reduction |

---

## Complete Worked Examples

### Example 1: Customer Similarity

**Problem:** Find which customers are most similar to a target customer.

```
Customer features: [age_normalized, income_normalized, purchases_normalized]

Alice:   [0.30, 0.70, 0.80]
Bob:     [0.35, 0.65, 0.75]
Charlie: [0.80, 0.30, 0.20]

Find who is most similar to Alice.
```

**Solution using Cosine Similarity:**

$$\text{sim}(\text{Alice}, \text{Bob}) = \frac{[0.30, 0.70, 0.80] \cdot [0.35, 0.65, 0.75]}{\|\text{Alice}\| \|\text{Bob}\|}$$

Numerator: $0.30 \times 0.35 + 0.70 \times 0.65 + 0.80 \times 0.75 = 0.105 + 0.455 + 0.6 = 1.16$

$\|\text{Alice}\| = \sqrt{0.09 + 0.49 + 0.64} = \sqrt{1.22} = 1.105$
$\|\text{Bob}\| = \sqrt{0.1225 + 0.4225 + 0.5625} = \sqrt{1.1075} = 1.052$

$$\text{sim}(\text{Alice}, \text{Bob}) = \frac{1.16}{1.105 \times 1.052} = \frac{1.16}{1.163} = 0.997$$

Similarly: $\text{sim}(\text{Alice}, \text{Charlie}) \approx 0.71$

**Answer:** Bob is much more similar to Alice (0.997 vs 0.71).

### Example 2: Linear Independence Check

**Problem:** Are these vectors linearly independent?

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 0 \\ 3 \\ 2 \end{bmatrix}$$

**Solution:** Form matrix and find rank:

$$A = \begin{bmatrix} 1 & 2 & 0 \\ 2 & 1 & 3 \\ 1 & 0 & 2 \end{bmatrix}$$

Compute determinant: $\det(A) = 1(2-0) - 2(4-3) + 0 = 2 - 2 = 0$

Since det = 0, rank < 3, so vectors are **linearly dependent**.

### Example 3: Vector Projection for Regression

**Problem:** Project data point $\mathbf{y} = \begin{bmatrix} 4 \\ 5 \end{bmatrix}$ onto the line spanned by $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.

**Solution:**
$$\text{proj}_{\mathbf{x}}(\mathbf{y}) = \frac{\mathbf{y} \cdot \mathbf{x}}{\mathbf{x} \cdot \mathbf{x}} \mathbf{x}$$

$$= \frac{4 \times 1 + 5 \times 2}{1^2 + 2^2} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \frac{14}{5} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 2.8 \\ 5.6 \end{bmatrix}$$

The best fit scalar is $\frac{14}{5} = 2.8$.

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| **Vector Addition** | $\mathbf{u} + \mathbf{v} = [u_1+v_1, u_2+v_2, \ldots]$ |
| **Scalar Multiplication** | $c\mathbf{v} = [cv_1, cv_2, \ldots]$ |
| **Dot Product** | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$ |
| **L2 Norm** | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ |
| **L1 Norm** | $\|\mathbf{v}\|_1 = \sum_i |v_i|$ |
| **Unit Vector** | $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$ |
| **Cosine Similarity** | $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ |
| **Euclidean Distance** | $d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2$ |
| **Projection** | $\text{proj}_\mathbf{v}\mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$ |
| **Orthogonality** | $\mathbf{u} \perp \mathbf{v} \iff \mathbf{u} \cdot \mathbf{v} = 0$ |

---

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Confusing row/column vectors | Convention: $\mathbf{x}$ is column, $\mathbf{x}^T$ is row |
| Dimension mismatch | Check dimensions before operations |
| Forgetting to normalize | Use cosine similarity for direction-only comparison |
| Zero vector division | Always check $\|\mathbf{v}\| \neq 0$ before normalizing |
| Numerical precision | Use `np.allclose()` instead of `==` for floating point |

---

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [examples.ipynb](examples.ipynb) | Interactive examples with visualizations covering vector operations, norms, similarity, projection, and ML applications |
| [exercises.ipynb](exercises.ipynb) | 10 hands-on exercises implementing vector operations from scratch (⭐ to ⭐⭐⭐ difficulty) |

---

## Interview Questions

**Q1: What's the geometric meaning of the dot product?**

The dot product measures how much two vectors point in the same direction. Mathematically, $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$. 
- Positive → same general direction
- Zero → perpendicular
- Negative → opposite directions

**Q2: Why use cosine similarity instead of dot product?**

Cosine similarity is **normalized**, so it only measures angle (direction), not magnitude. This is essential when comparing items of different scales—e.g., documents of different lengths, or embeddings with different magnitudes.

**Q3: How are vectors used in neural networks?**

- **Inputs:** Data points are feature vectors
- **Weights:** Each layer has weight vectors/matrices
- **Forward pass:** Matrix-vector multiplications (dot products)
- **Embeddings:** Map discrete tokens to continuous vectors
- **Attention:** Query-key dot products determine attention weights

**Q4: What's the difference between L1 and L2 regularization?**

- **L1 (Lasso):** $\|\mathbf{w}\|_1$ encourages sparsity (many weights become exactly 0)
- **L2 (Ridge):** $\|\mathbf{w}\|_2^2$ encourages small weights (but rarely exactly 0)

L1 is useful for feature selection; L2 is more numerically stable.

**Q5: What does it mean for vectors to be linearly independent?**

No vector can be written as a linear combination of the others. In ML terms, each independent feature provides unique information that can't be derived from the others.

---

## Next Steps

→ [Matrix Operations](../02-Matrix-Operations/README.md): Learn how matrices transform vectors and represent linear maps

