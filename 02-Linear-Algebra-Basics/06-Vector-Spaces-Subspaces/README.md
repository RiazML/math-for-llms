# Vector Spaces and Subspaces

## Introduction

Vector spaces are the foundational algebraic structures of linear algebra, providing a framework for understanding linear transformations, dimensionality, and the structure of solutions to linear systems. Subspaces are vector spaces within vector spaces, representing constrained sets that retain vector space properties.

## Prerequisites

- Vectors and vector operations
- Matrix operations
- Systems of linear equations
- Matrix rank and null space basics

## Learning Objectives

1. Understand the axioms defining vector spaces
2. Identify and work with subspaces
3. Understand linear independence and spanning sets
4. Master the concept of basis and dimension
5. Work with the four fundamental subspaces

---

## 1. Vector Space Definition

### Formal Definition

A **vector space** V over a field F (usually ℝ or ℂ) is a set with two operations:

- **Vector addition**: V × V → V
- **Scalar multiplication**: F × V → V

satisfying these axioms for all **u**, **v**, **w** ∈ V and scalars a, b ∈ F:

### Addition Axioms

1. **Closure**: **u** + **v** ∈ V
2. **Commutativity**: **u** + **v** = **v** + **u**
3. **Associativity**: (**u** + **v**) + **w** = **u** + (**v** + **w**)
4. **Zero vector**: ∃ **0** ∈ V such that **v** + **0** = **v**
5. **Additive inverse**: ∀**v**, ∃(-**v**) such that **v** + (-**v**) = **0**

### Scalar Multiplication Axioms

6. **Closure**: a**v** ∈ V
7. **Distributivity (scalar)**: a(**u** + **v**) = a**u** + a**v**
8. **Distributivity (vector)**: (a + b)**v** = a**v** + b**v**
9. **Associativity**: a(b**v**) = (ab)**v**
10. **Identity**: 1**v** = **v**

### Examples of Vector Spaces

```
Standard Examples:
┌──────────────────────────────────────────────────────────┐
│ ℝⁿ: n-dimensional real vectors                          │
│     (x₁, x₂, ..., xₙ) with standard operations          │
│                                                          │
│ ℂⁿ: n-dimensional complex vectors                        │
│                                                          │
│ M_{m×n}: All m×n matrices over ℝ                         │
│                                                          │
│ P_n: Polynomials of degree ≤ n                           │
│      p(x) = a₀ + a₁x + ... + aₙxⁿ                        │
│                                                          │
│ C[a,b]: Continuous functions on [a,b]                    │
└──────────────────────────────────────────────────────────┘
```

---

## 2. Subspaces

### Definition

A subset W ⊆ V is a **subspace** of V if W is itself a vector space under the same operations.

### Subspace Test (Three Conditions)

W is a subspace of V if and only if:

$$
\begin{aligned}
&\text{1. } \mathbf{0} \in W \text{ (contains zero vector)} \\
&\text{2. } \mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W \text{ (closed under addition)} \\
&\text{3. } \mathbf{v} \in W, c \in F \Rightarrow c\mathbf{v} \in W \text{ (closed under scalar multiplication)}
\end{aligned}
$$

**Equivalent One-Condition Test:**
W is a subspace if and only if for all **u**, **v** ∈ W and scalars a, b:
$$a\mathbf{u} + b\mathbf{v} \in W$$

### Visualizing Subspaces in ℝ³

```
                        z
                        │
              Plane     │    Line through origin
             (dim=2)    │      (dim=1)
                ╲       │       ╱
                 ╲      │      ╱
    ┌─────────────╲─────│─────╱───────┐
    │              ╲    │    ╱        │
    │               ╲   │   ╱         │
    │                ╲  │  ╱          │
    │                 ╲ │ ╱           │
    │──────────────────·────────────▶ y
    │                 ╱ │ ╲           │
    │                ╱  │  ╲          │
    │               ╱   │   ╲         │
    │              ╱    │    ╲        │
    └─────────────╱─────│─────╲───────┘
                 ╱      │      ╲
                ╱       │       ╲
               ╱        │
              x

Subspaces of ℝ³:
- {0}: The zero vector (dim = 0)
- Lines through origin (dim = 1)
- Planes through origin (dim = 2)
- ℝ³ itself (dim = 3)

NOT subspaces:
- Lines not through origin
- Planes not through origin
- Spheres, cubes, etc.
```

### Examples: Subspace or Not?

| Set                         | Subspace? | Reason                       |
| --------------------------- | --------- | ---------------------------- |
| {(x, y) : y = 2x}           | ✓ Yes     | Line through origin          |
| {(x, y) : y = 2x + 1}       | ✗ No      | Doesn't contain (0,0)        |
| {(x, y, z) : x + y + z = 0} | ✓ Yes     | Plane through origin         |
| {(x, y) : x ≥ 0}            | ✗ No      | Not closed under scalar mult |
| All 2×2 symmetric matrices  | ✓ Yes     | Closed under both operations |
| All 2×2 invertible matrices | ✗ No      | Zero matrix not included     |

---

## 3. Span

### Definition

The **span** of vectors {**v**₁, **v**₂, ..., **v**ₖ} is the set of all linear combinations:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k : c_i \in F\}$$

### Key Properties

- span{**v**₁, ..., **v**ₖ} is always a subspace
- The smallest subspace containing all the vectors

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
(Vectors are proportional)
```

---

## 4. Linear Independence

### Definition

Vectors {**v**₁, **v**₂, ..., **v**ₖ} are **linearly independent** if:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

**Linearly dependent**: At least one vector can be written as a linear combination of others.

### Testing Independence

For vectors as columns of matrix A:

- **Independent** if and only if rank(A) = number of vectors
- **Independent** if and only if Null(A) = {**0**}
- **Independent** if and only if A**x** = **0** has only trivial solution

```
Independent vectors in ℝ²:        Dependent vectors in ℝ²:
         y                                 y
         │                                 │    ╱ v₂
         │    v₂                           │   ╱
         │   ╱                             │  ╱
         │  ╱                              │ ╱
         │ ╱                               │╱
    ─────┼──────▶ x                   ─────┼──────▶ x
         │╲     v₁                         │ ╲
         │ ╲                               │  ╲ v₁
         │  ╲                              │
         │                                 │
    (span entire ℝ²)                  (span only a line)
```

### Maximum Independent Vectors

In ℝⁿ:

- At most n vectors can be linearly independent
- Exactly n independent vectors span all of ℝⁿ

---

## 5. Basis and Dimension

### Basis Definition

A **basis** for vector space V is a set B = {**v**₁, ..., **v**ₙ} such that:

1. B is linearly independent
2. B spans V

Every vector in V can be written **uniquely** as a linear combination of basis vectors.

### Dimension

The **dimension** of V, denoted dim(V), is the number of vectors in any basis.

$$\dim(V) = \text{number of vectors in any basis of } V$$

### Standard Bases

```
Standard basis for ℝⁿ:
┌─────────────────────────────────────────┐
│ e₁ = (1, 0, 0, ..., 0)                  │
│ e₂ = (0, 1, 0, ..., 0)                  │
│ e₃ = (0, 0, 1, ..., 0)                  │
│ ⋮                                        │
│ eₙ = (0, 0, 0, ..., 1)                  │
└─────────────────────────────────────────┘

Standard basis for M₂ₓ₂ (4-dimensional):
E₁₁ = [1 0]  E₁₂ = [0 1]  E₂₁ = [0 0]  E₂₂ = [0 0]
      [0 0]        [0 0]        [1 0]        [0 1]

Standard basis for P₂ (3-dimensional):
{1, x, x²}
```

### Non-Standard Bases

Any set of n linearly independent vectors in ℝⁿ forms a basis:

```
Non-standard basis for ℝ²:
B = {(1, 1), (1, -1)}

To express (3, 1) in this basis:
(3, 1) = c₁(1, 1) + c₂(1, -1)
       = 2(1, 1) + 1(1, -1)

Coordinates: [2, 1]_B
```

---

## 6. The Four Fundamental Subspaces

For matrix A (m×n):

```
              A : ℝⁿ → ℝᵐ

    ┌─────────────────────────────────────┐
    │                                     │
    │   Domain (ℝⁿ)      Codomain (ℝᵐ)    │
    │   ┌───────┐          ┌───────┐      │
    │   │       │    A     │       │      │
    │   │ Row   │ ───────▶ │Column │      │
    │   │ Space │          │ Space │      │
    │   │       │          │       │      │
    │   └───────┘          └───────┘      │
    │   ┌───────┐          ┌───────┐      │
    │   │       │          │       │      │
    │   │ Null  │          │ Left  │      │
    │   │ Space │          │ Null  │      │
    │   │       │          │ Space │      │
    │   └───────┘          └───────┘      │
    │                                     │
    └─────────────────────────────────────┘
```

### 1. Column Space (Range)

$$C(A) = \{\mathbf{b} : A\mathbf{x} = \mathbf{b} \text{ has a solution}\}$$

- Subspace of ℝᵐ
- dim(C(A)) = rank(A) = r
- Spanned by columns of A

### 2. Row Space

$$C(A^T) = \{\mathbf{x}^T A : \mathbf{x} \in \mathbb{R}^m\}$$

- Subspace of ℝⁿ
- dim(C(A^T)) = rank(A) = r
- Spanned by rows of A

### 3. Null Space (Kernel)

$$N(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$

- Subspace of ℝⁿ
- dim(N(A)) = n - r (nullity)
- Solution set of homogeneous system

### 4. Left Null Space

$$N(A^T) = \{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\} = \{\mathbf{y} : \mathbf{y}^T A = \mathbf{0}\}$$

- Subspace of ℝᵐ
- dim(N(A^T)) = m - r

### Orthogonal Complements

```
In ℝⁿ:                        In ℝᵐ:
┌─────────────────────┐       ┌─────────────────────┐
│                     │       │                     │
│   Row Space         │       │   Column Space      │
│      C(Aᵀ)          │       │      C(A)           │
│      dim = r        │       │      dim = r        │
│         ⊥           │       │         ⊥           │
│   Null Space        │       │   Left Null Space   │
│      N(A)           │       │      N(Aᵀ)          │
│      dim = n-r      │       │      dim = m-r      │
│                     │       │                     │
└─────────────────────┘       └─────────────────────┘

C(Aᵀ) ⊕ N(A) = ℝⁿ             C(A) ⊕ N(Aᵀ) = ℝᵐ
```

---

## 7. Finding Bases for Fundamental Subspaces

### Example Matrix

$$A = \begin{bmatrix} 1 & 2 & 1 & 0 \\ 2 & 4 & 3 & 1 \\ 3 & 6 & 4 & 1 \end{bmatrix}$$

### Row Reduce to Find Rank

$$A \xrightarrow{RREF} \begin{bmatrix} 1 & 2 & 0 & -1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

rank(A) = 2, Pivot columns: 1, 3

### Column Space Basis

Take pivot columns from **original** A:
$$\text{Basis for } C(A) = \left\{ \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix} \right\}$$

### Row Space Basis

Take non-zero rows from RREF:
$$\text{Basis for } C(A^T) = \{(1, 2, 0, -1), (0, 0, 1, 1)\}$$

### Null Space Basis

Solve A**x** = **0** using RREF:

- Free variables: x₂, x₄
- Set x₂ = 1, x₄ = 0: **x** = (-2, 1, 0, 0)
- Set x₂ = 0, x₄ = 1: **x** = (1, 0, -1, 1)

$$\text{Basis for } N(A) = \left\{ \begin{bmatrix} -2 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ -1 \\ 1 \end{bmatrix} \right\}$$

---

## 8. Applications in ML/AI

### 1. Feature Space

Data lives in a vector space where:

- Features define dimensions
- Samples are vectors
- Linear models operate on this space

### 2. Null Space and Degenerate Solutions

In linear regression with rank-deficient X:
$$X\mathbf{w} = \mathbf{y}$$

- If Null(X) ≠ {0}, infinitely many solutions
- Regularization selects unique solution
- Ridge: minimum norm solution
- Lasso: sparse solution

### 3. Dimensionality Reduction

PCA finds a lower-dimensional subspace:

- Maximize variance in the subspace
- Principal components form a basis
- Data projected onto this subspace

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
```

### 4. Kernel Methods

Feature maps φ: X → H (high-dimensional space)

- H is a vector space (often infinite-dimensional)
- Kernel trick: work with inner products only
- Subspaces in H define decision boundaries

### 5. Attention in Transformers

Query, Key, Value spaces:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q, K, V are linear projections (subspaces)
- Attention weights operate in these subspaces

---

## 9. Summary

### Key Concepts

| Concept      | Definition                                        | Dimension             |
| ------------ | ------------------------------------------------- | --------------------- |
| Vector Space | Set with addition & scalar mult satisfying axioms | n (if ℝⁿ)             |
| Subspace     | Subset that's also a vector space                 | ≤ n                   |
| Span         | All linear combinations                           | # independent vectors |
| Basis        | Independent spanning set                          | = dim(V)              |
| Column Space | {Ax : x ∈ ℝⁿ}                                     | rank(A)               |
| Null Space   | {x : Ax = 0}                                      | n - rank(A)           |

### Dimension Relationships

$$\dim(C(A)) + \dim(N(A)) = n \quad \text{(Rank-Nullity)}$$

$$\dim(C(A^T)) + \dim(N(A^T)) = m$$

### ML Connections

- **Feature engineering**: Working in/defining the right vector space
- **Regularization**: Constraining to subspaces
- **Embeddings**: Learning low-dimensional subspaces
- **Attention**: Projecting to query/key/value subspaces

---

## Exercises

1. Verify that P₂ (polynomials of degree ≤ 2) is a vector space
2. Determine if {(x, y, z) : x² + y² + z² = 1} is a subspace of ℝ³
3. Find a basis for span{(1,2,3), (4,5,6), (2,1,0)}
4. Find all four fundamental subspaces for a given matrix
5. Prove that the intersection of two subspaces is a subspace
6. Show that the union of two subspaces is NOT generally a subspace

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Axler, S. - "Linear Algebra Done Right"
3. MIT 18.06 - Linear Algebra (Gilbert Strang)
4. 3Blue1Brown - "Essence of Linear Algebra"
