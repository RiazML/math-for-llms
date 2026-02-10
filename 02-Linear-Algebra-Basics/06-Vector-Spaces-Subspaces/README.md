# Vector Spaces and Subspaces

## Overview

Vector spaces are the foundational algebraic structures of linear algebra, providing a rigorous framework for understanding linear transformations, dimensionality, and the structure of solutions to linear systems. A vector space is a set equipped with addition and scalar multiplication that satisfies specific axioms, enabling the powerful machinery of linear algebra.

Subspaces are vector spaces within vector spaces—subsets that retain all vector space properties. Understanding subspaces is crucial for decomposing problems, analyzing transformations, and building intuition about dimensionality.

In machine learning and AI, vector space concepts are everywhere:
- **Feature spaces** where data lives as vectors
- **Embedding spaces** learned by neural networks
- **Kernel spaces** in support vector machines
- **Attention mechanisms** projecting to query/key/value subspaces
- **Latent spaces** in variational autoencoders

## Learning Objectives

By the end of this section, you will be able to:

- **State and verify** the 10 vector space axioms
- **Test** whether a subset is a subspace (three conditions)
- **Compute** span of a set of vectors
- **Determine** linear independence using rank
- **Find** bases and compute dimension
- **Identify** the four fundamental subspaces of a matrix
- **Apply** vector space concepts to ML problems

## Prerequisites

- Vector operations (addition, scalar multiplication)
- Matrix operations and rank
- Systems of linear equations
- Null space basics

---

## 1. Vector Space Definition

### 1.1 Formal Definition

A **vector space** $V$ over a field $F$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a set with two operations:

- **Vector addition**: $V \times V \to V$
- **Scalar multiplication**: $F \times V \to V$

satisfying ten axioms for all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $a, b \in F$:

### Addition Axioms

| # | Axiom | Statement |
|---|-------|-----------|
| 1 | Closure | $\mathbf{u} + \mathbf{v} \in V$ |
| 2 | Commutativity | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 3 | Associativity | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| 4 | Zero vector | $\exists \mathbf{0} \in V : \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 5 | Additive inverse | $\forall \mathbf{v}, \exists (-\mathbf{v}) : \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |

### Scalar Multiplication Axioms

| # | Axiom | Statement |
|---|-------|-----------|
| 6 | Closure | $a\mathbf{v} \in V$ |
| 7 | Distributivity (scalar) | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ |
| 8 | Distributivity (vector) | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ |
| 9 | Associativity | $a(b\mathbf{v}) = (ab)\mathbf{v}$ |
| 10 | Identity | $1\mathbf{v} = \mathbf{v}$ |

### 1.2 Common Examples of Vector Spaces

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      STANDARD VECTOR SPACES                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ℝⁿ: n-dimensional real vectors                                         │
│      (x₁, x₂, ..., xₙ) with component-wise operations                   │
│      Examples: ℝ², ℝ³, ℝ¹⁰⁰ (feature vectors)                           │
│                                                                          │
│  ℂⁿ: n-dimensional complex vectors                                       │
│      Used in quantum computing, signal processing                        │
│                                                                          │
│  M_{m×n}: All m×n matrices over ℝ                                        │
│      dim(M_{m×n}) = m·n                                                  │
│                                                                          │
│  Pₙ: Polynomials of degree ≤ n                                           │
│      p(x) = a₀ + a₁x + ... + aₙxⁿ                                        │
│      dim(Pₙ) = n + 1                                                     │
│                                                                          │
│  C[a,b]: Continuous functions on [a,b]                                   │
│      Infinite-dimensional!                                               │
│                                                                          │
│  ℓ²: Square-summable sequences                                           │
│      {(x₁, x₂, ...) : Σxᵢ² < ∞}                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Non-Examples

Not everything is a vector space:

| Set | Why Not a Vector Space |
|-----|------------------------|
| Positive reals $\mathbb{R}^+$ | Not closed under scalar mult: $(-1) \cdot 5 = -5 \notin \mathbb{R}^+$ |
| Integers $\mathbb{Z}$ | Not closed under scalar mult: $\frac{1}{2} \cdot 3 = 1.5 \notin \mathbb{Z}$ |
| Unit circle | Not closed under addition or scalar mult |

---

## 2. Subspaces

### 2.1 Definition

A subset $W \subseteq V$ is a **subspace** of $V$ if $W$ is itself a vector space under the same operations.

### 2.2 Subspace Test (Three Conditions)

$W$ is a subspace of $V$ if and only if:

$$\begin{aligned}
&\text{1. } \mathbf{0} \in W \quad \text{(contains zero vector)} \\
&\text{2. } \mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W \quad \text{(closed under addition)} \\
&\text{3. } \mathbf{v} \in W, c \in F \Rightarrow c\mathbf{v} \in W \quad \text{(closed under scalar multiplication)}
\end{aligned}$$

**Equivalent One-Condition Test:**
$W$ is a subspace iff for all $\mathbf{u}, \mathbf{v} \in W$ and scalars $a, b$:
$$a\mathbf{u} + b\mathbf{v} \in W$$

### 2.3 Visualizing Subspaces in ℝ³

```
                              z
                              │
                 Plane        │      Line through origin
                (dim=2)       │        (dim=1)
                   ╲          │         ╱
                    ╲         │        ╱
       ┌─────────────╲────────│───────╱──────────┐
       │              ╲       │      ╱           │
       │               ╲      │     ╱            │
       │                ╲     │    ╱             │
       │                 ╲    │   ╱              │
       │──────────────────────●──────────────────▶ y
       │                 ╱    │   ╲              │
       │                ╱     │    ╲             │
       │               ╱      │     ╲            │
       │              ╱       │      ╲           │
       └─────────────╱────────│───────╲──────────┘
                    ╱         │        ╲
                   ╱          │         ╲
                  x

    ALL subspaces of ℝ³:
    • {0}: Just the origin (dim = 0)
    • Lines through origin (dim = 1)
    • Planes through origin (dim = 2)
    • ℝ³ itself (dim = 3)

    NOT subspaces:
    • Lines not through origin
    • Planes not through origin
    • Spheres, cubes, etc.
```

### 2.4 Examples: Subspace or Not?

| Set | Subspace? | Reason |
|-----|-----------|--------|
| {(x, y) : y = 2x} | ✓ Yes | Line through origin |
| {(x, y) : y = 2x + 1} | ✗ No | Doesn't contain (0,0) |
| {(x, y, z) : x + y + z = 0} | ✓ Yes | Plane through origin |
| {(x, y) : x ≥ 0} | ✗ No | (-1)·(1,0) = (-1,0), x < 0 |
| Symmetric 2×2 matrices | ✓ Yes | Closed under both ops |
| Invertible 2×2 matrices | ✗ No | Zero matrix not invertible |
| Upper triangular matrices | ✓ Yes | Closed under both ops |
| Matrices with trace = 0 | ✓ Yes | tr(A+B) = tr(A)+tr(B) |
| Matrices with det = 0 | ✗ No | Not closed under addition |

---

## 3. Span

### 3.1 Definition

The **span** of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is the set of all linear combinations:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k : c_i \in F\}$$

### 3.2 Key Properties

- $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is always a subspace
- It's the **smallest** subspace containing all the vectors
- Adding a vector outside the span increases dimension by 1

### 3.3 Geometric Interpretation

```
Example: span{(1,0), (0,1)} in ℝ²

       y
       │
     1 ┼───●  (0,1)
       │   │
       │   │  span = entire ℝ² plane
       ├───┼───────▶ x
       0   1
           ●  (1,0)


Example: span{(1,2), (2,4)} in ℝ²

       y
       │    ╱
     4 ┼───●  (2,4)
       │  ╱
     2 ┼─●     (1,2)
       │╱
       ├────────────▶ x

span = line through origin (y = 2x)
       (vectors are proportional: (2,4) = 2·(1,2))
```

### 3.4 Testing if Vector is in Span

To check if $\mathbf{w} \in \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$:

1. Form matrix $A = [\mathbf{v}_1 | \cdots | \mathbf{v}_k]$
2. Augment: $[A | \mathbf{w}]$
3. Compare ranks: $\mathbf{w} \in \text{span}$ iff $\text{rank}(A) = \text{rank}([A|\mathbf{w}])$

---

## 4. Linear Independence

### 4.1 Definition

Vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ are **linearly independent** if:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

**Linearly dependent**: At least one vector can be written as a linear combination of the others.

### 4.2 Testing Independence

For vectors as columns of matrix $A$:

| Condition | Result |
|-----------|--------|
| $\text{rank}(A) = k$ (# vectors) | Independent |
| $\text{rank}(A) < k$ | Dependent |
| $\ker(A) = \{\mathbf{0}\}$ | Independent |
| $A\mathbf{x} = \mathbf{0}$ has only trivial solution | Independent |
| $\det(A) \neq 0$ (if square) | Independent |

### 4.3 Geometric Interpretation

```
Independent vectors in ℝ²:          Dependent vectors in ℝ²:
         y                                   y
         │                                   │    ╱ v₂
         │    v₂                             │   ╱
         │   ╱                               │  ╱
         │  ╱                                │ ╱
         │ ╱                                 │╱
    ─────┼──────▶ x                     ─────┼──────▶ x
         │╲     v₁                           │ ╲
         │ ╲                                 │  ╲ v₁
         │  ╲                                │
         │                                   │
    (span entire ℝ²)                    (span only a line)
```

### 4.4 Key Facts

- In $\mathbb{R}^n$, at most $n$ vectors can be linearly independent
- More than $n$ vectors in $\mathbb{R}^n$ are always dependent
- A set containing $\mathbf{0}$ is always dependent

---

## 5. Basis and Dimension

### 5.1 Basis Definition

A **basis** for vector space $V$ is a set $B = \{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ such that:

1. $B$ is linearly independent
2. $B$ spans $V$

Every vector in $V$ can be written **uniquely** as a linear combination of basis vectors.

### 5.2 Dimension

The **dimension** of $V$, denoted $\dim(V)$, is the number of vectors in any basis.

$$\dim(V) = \text{number of vectors in any basis of } V$$

**Key theorem**: All bases of a vector space have the same number of vectors.

### 5.3 Standard Bases

```
Standard basis for ℝⁿ:
┌─────────────────────────────────────────────────────────────┐
│ e₁ = (1, 0, 0, ..., 0)                                      │
│ e₂ = (0, 1, 0, ..., 0)                                      │
│ e₃ = (0, 0, 1, ..., 0)                                      │
│ ⋮                                                            │
│ eₙ = (0, 0, 0, ..., 1)                                      │
└─────────────────────────────────────────────────────────────┘

Standard basis for M₂ₓ₂ (dim = 4):
E₁₁ = [1 0]   E₁₂ = [0 1]   E₂₁ = [0 0]   E₂₂ = [0 0]
      [0 0]         [0 0]         [1 0]         [0 1]

Standard basis for P₂ (dim = 3):
{1, x, x²}

Symmetric 2×2 matrices (dim = 3):
[1 0]   [0 1]   [0 0]
[0 0]   [1 0]   [0 1]
```

### 5.4 Non-Standard Bases

Any set of $n$ linearly independent vectors in $\mathbb{R}^n$ forms a basis:

```
Non-standard basis for ℝ²:
B = {(1, 1), (1, -1)}

To express (3, 1) in basis B:
(3, 1) = c₁(1, 1) + c₂(1, -1)
       = 2(1, 1) + 1(1, -1)

Coordinates in B: [(3,1)]_B = (2, 1)
```

### 5.5 Change of Basis

If $B = [\mathbf{b}_1 | \cdots | \mathbf{b}_n]$ is a matrix whose columns are basis vectors:

- **Standard → B**: $[\mathbf{v}]_B = B^{-1}\mathbf{v}$
- **B → Standard**: $\mathbf{v} = B[\mathbf{v}]_B$

---

## 6. The Four Fundamental Subspaces

For matrix $A \in \mathbb{R}^{m \times n}$ with rank $r$:

```
                    A : ℝⁿ → ℝᵐ

       ┌─────────────────────────────────────────────┐
       │                                             │
       │     Domain (ℝⁿ)          Codomain (ℝᵐ)      │
       │                                             │
       │     ┌───────────┐        ┌───────────┐      │
       │     │           │   A    │           │      │
       │     │ Row Space │ ─────▶ │  Column   │      │
       │     │  C(Aᵀ)    │        │   Space   │      │
       │     │  dim = r  │        │   C(A)    │      │
       │     │           │        │  dim = r  │      │
       │     └───────────┘        └───────────┘      │
       │           ⊥                    ⊥            │
       │     ┌───────────┐        ┌───────────┐      │
       │     │           │        │           │      │
       │     │Null Space │        │ Left Null │      │
       │     │   N(A)    │        │   Space   │      │
       │     │ dim = n-r │        │  N(Aᵀ)    │      │
       │     │           │        │ dim = m-r │      │
       │     └───────────┘        └───────────┘      │
       │                                             │
       └─────────────────────────────────────────────┘
```

### 6.1 Column Space (Range)

$$C(A) = \{\mathbf{b} : A\mathbf{x} = \mathbf{b} \text{ has a solution}\} = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$$

- Subspace of $\mathbb{R}^m$
- $\dim(C(A)) = r$ (rank)
- Spanned by columns of $A$
- **Basis**: Pivot columns of original $A$

### 6.2 Row Space

$$C(A^T) = \{\mathbf{x}^T A : \mathbf{x} \in \mathbb{R}^m\}$$

- Subspace of $\mathbb{R}^n$
- $\dim(C(A^T)) = r$ (rank)
- Spanned by rows of $A$
- **Basis**: Non-zero rows of RREF

### 6.3 Null Space (Kernel)

$$N(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$

- Subspace of $\mathbb{R}^n$
- $\dim(N(A)) = n - r$ (nullity)
- Solution set of homogeneous system
- **Basis**: Found from free variables in RREF

### 6.4 Left Null Space

$$N(A^T) = \{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\} = \{\mathbf{y} : \mathbf{y}^T A = \mathbf{0}\}$$

- Subspace of $\mathbb{R}^m$
- $\dim(N(A^T)) = m - r$

### 6.5 Orthogonal Complements

The four subspaces come in orthogonal pairs:

$$\mathbb{R}^n = C(A^T) \oplus N(A) \quad \text{(orthogonal direct sum)}$$
$$\mathbb{R}^m = C(A) \oplus N(A^T)$$

This means:
- Every $\mathbf{x} \in \mathbb{R}^n$ can be uniquely written as $\mathbf{x} = \mathbf{r} + \mathbf{n}$ where $\mathbf{r} \in C(A^T)$, $\mathbf{n} \in N(A)$
- Row space vectors are perpendicular to null space vectors

---

## 7. Finding Bases for Fundamental Subspaces

### Example

$$A = \begin{bmatrix} 1 & 2 & 1 & 0 \\ 2 & 4 & 3 & 1 \\ 3 & 6 & 4 & 1 \end{bmatrix}$$

### Step 1: Row Reduce

$$A \xrightarrow{\text{RREF}} \begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

$\text{rank}(A) = 2$, Pivot columns: 1, 3

### Step 2: Column Space Basis

Take pivot columns from **original** $A$:

$$\text{Basis for } C(A) = \left\{ \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix} \right\}$$

### Step 3: Row Space Basis

Take non-zero rows from RREF:

$$\text{Basis for } C(A^T) = \{(1, 2, 0, -1), (0, 0, 1, 1)\}$$

### Step 4: Null Space Basis

From RREF, free variables are $x_2, x_4$:
- Set $x_2 = 1, x_4 = 0$: $\mathbf{x} = (-2, 1, 0, 0)$
- Set $x_2 = 0, x_4 = 1$: $\mathbf{x} = (1, 0, -1, 1)$

$$\text{Basis for } N(A) = \left\{ \begin{bmatrix} -2 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ -1 \\ 1 \end{bmatrix} \right\}$$

### Verification

- $\dim(C(A)) + \dim(N(A^T)) = 2 + 1 = 3 = m$ ✓
- $\dim(C(A^T)) + \dim(N(A)) = 2 + 2 = 4 = n$ ✓

---

## 8. Applications in Machine Learning

### 8.1 Feature Space

Data lives in a vector space where:
- Features define dimensions
- Samples are vectors
- Linear models operate on this space

```
Feature Vector Space (ℝᵈ):

       Feature 2
           │
           │    ● sample 3
           │   ╱
    ●──────┼──╱───────● sample 2
 sample 1  │ ╱
           │╱
           └──────────────▶ Feature 1
          ╱
         ╱
      Feature 3

Each sample = point in d-dimensional space
```

### 8.2 Null Space and Degenerate Solutions

In linear regression with rank-deficient $X$:

$$X\mathbf{w} = \mathbf{y}$$

- If $N(X) \neq \{\mathbf{0}\}$: infinitely many solutions
- Adding $\mathbf{n} \in N(X)$ to any solution gives another solution
- **Regularization** selects a unique solution:
  - Ridge: minimum norm solution
  - Lasso: sparse solution

### 8.3 Dimensionality Reduction

PCA finds a lower-dimensional subspace:

```
Original space (ℝ³)              Subspace (ℝ²)
       z                              │
       │   ╱ data                     │  data
       │  ╱  scattered               ─┼──────────
       │ ╱                            │
    ───┼────▶ y        ════════▶    PC₂│
      ╱│                              │
     ╱ │                              └──────────▶ PC₁
    x

• Find subspace that captures most variance
• Project data onto this subspace
• Principal components form a basis
```

### 8.4 Kernel Methods

Feature maps $\phi: X \to H$ (high-dimensional space):

- $H$ is a vector space (often infinite-dimensional)
- Kernel trick: $K(x, y) = \langle \phi(x), \phi(y) \rangle$
- Subspaces in $H$ define decision boundaries

### 8.5 Attention Mechanisms in Transformers

Query, Key, Value are projections to subspaces:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q = XW_Q$: projection to query subspace
- $K = XW_K$: projection to key subspace
- $V = XW_V$: projection to value subspace
- Each projection is a linear map to a subspace

### 8.6 Embeddings

Word/image embeddings live in learned subspaces:

```
Embedding Space:

    "king" - "man" + "woman" ≈ "queen"

         ● queen
        ╱
       ╱  gender
      ╱   direction
     ╱
    ● king ─────────────────▶ royalty direction
    │
    │
    ● man
```

---

## 9. Advanced Topics

### 9.1 Direct Sum

Two subspaces $U$ and $W$ form a **direct sum** $V = U \oplus W$ if:
- $U \cap W = \{\mathbf{0}\}$
- $\dim(U) + \dim(W) = \dim(V)$

Every vector can be uniquely written as $\mathbf{u} + \mathbf{w}$.

### 9.2 Dimension Formula

For subspaces $U$ and $W$:

$$\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$$

### 9.3 Quotient Spaces

For subspace $W \subseteq V$, the quotient space $V/W$ consists of cosets:

$$\mathbf{v} + W = \{\mathbf{v} + \mathbf{w} : \mathbf{w} \in W\}$$

$$\dim(V/W) = \dim(V) - \dim(W)$$

### 9.4 Isomorphisms

Two vector spaces are **isomorphic** if they have the same dimension.

Examples:
- $\mathbb{R}^3 \cong P_2$ (both dimension 3)
- $M_{2 \times 2} \cong \mathbb{R}^4$ (both dimension 4)

---

## 10. Summary

### Key Concepts

| Concept | Definition | How to Find |
|---------|------------|-------------|
| Vector Space | Set with + and · satisfying 10 axioms | Verify axioms |
| Subspace | Subset that's also a vector space | Test 3 conditions |
| Span | All linear combinations | Row reduce |
| Linear Independence | Only trivial solution to $\sum c_i v_i = 0$ | Check rank = # vectors |
| Basis | Independent spanning set | Find pivot columns |
| Dimension | Size of any basis | = rank |
| Column Space | $\{A\mathbf{x}\}$ | Pivot columns of $A$ |
| Null Space | $\{\mathbf{x} : A\mathbf{x} = 0\}$ | Solve homogeneous system |

### Dimension Relationships

$$\dim(C(A)) + \dim(N(A)) = n \quad \text{(Rank-Nullity)}$$
$$\dim(C(A^T)) + \dim(N(A^T)) = m$$
$$\dim(C(A)) = \dim(C(A^T)) = \text{rank}(A)$$

### ML Connections

| Concept | ML Application |
|---------|----------------|
| Feature space | Where data lives |
| Subspace | Dimensionality reduction |
| Null space | Degenerate solutions, regularization |
| Basis | Representation, embeddings |
| Orthogonality | Decorrelated features |
| Projection | Feature extraction, attention |

---

## 11. Practice Problems

See the accompanying Jupyter notebooks:
- **[examples.ipynb](examples.ipynb)**: Worked examples with visualizations
- **[exercises.ipynb](exercises.ipynb)**: Practice problems with solutions

Key exercises include:
1. Testing whether sets are subspaces
2. Computing span and testing membership
3. Determining linear independence
4. Finding bases for fundamental subspaces
5. Change of basis computations
6. Direct sum and dimension formula

---

## 12. References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Axler, S. - "Linear Algebra Done Right"
3. MIT 18.06 - Linear Algebra (Gilbert Strang)
4. 3Blue1Brown - "Essence of Linear Algebra"

---

## Navigation

[← Previous: Matrix Rank](../05-Matrix-Rank/README.md) | [Next: Advanced Linear Algebra →](../../03-Advanced-Linear-Algebra/README.md)

[↑ Back to Linear Algebra Basics](../README.md) | [↑↑ Back to Main](../../README.md)
