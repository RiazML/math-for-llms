# Vectors and Spaces

## Overview

Vectors are the fundamental building blocks of machine learning. Every piece of data—images, text, audio—is represented as vectors. Understanding vectors and vector spaces is essential for grasping how ML algorithms work.

## Prerequisites

- Number systems
- Basic algebra

## Learning Objectives

- Understand vectors geometrically and algebraically
- Master vector operations (addition, scaling, dot product)
- Grasp the concept of vector spaces
- Connect vectors to ML feature representations

---

## Theory

### What is a Vector?

A vector can be thought of in three ways:

#### 1. Geometric View (Physics)

An arrow with magnitude (length) and direction.

```
        ↑
        │  v⃗
        │ ╱
        │╱ θ
    ────┼────→
        │
        │
```

#### 2. Coordinate View (Computer Science)

An ordered list of numbers.

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

#### 3. Abstract View (Mathematics)

An element of a vector space.

**ML Perspective:** In machine learning, we typically use the coordinate view, where each number represents a feature.

### Vector Notation

| Notation                      | Meaning                     |
| ----------------------------- | --------------------------- |
| $\mathbf{v}$, $\vec{v}$       | Vector (bold or arrow)      |
| $v_i$ or $[\mathbf{v}]_i$     | $i$-th component            |
| $\mathbf{v} \in \mathbb{R}^n$ | $n$-dimensional real vector |
| $\mathbf{0}$                  | Zero vector                 |
| $\mathbf{e}_i$                | Standard basis vector       |

---

## Vector Operations

### 1. Vector Addition

Add component-wise:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

**Geometric Interpretation:** Parallelogram law or tip-to-tail addition.

```
        v⃗
    ┌───────→
    │       │
u⃗  │       │ u⃗
    │       │
    └───────┘
        v⃗

    u⃗ + v⃗ = diagonal of parallelogram
```

### 2. Scalar Multiplication

Multiply each component by a scalar:

$$c \cdot \mathbf{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

**Geometric Interpretation:**

- $c > 1$: Stretches the vector
- $0 < c < 1$: Shrinks the vector
- $c < 0$: Reverses direction

### 3. Dot Product (Inner Product)

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

**Alternative formula:**
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

**Properties:**
| Property | Formula |
|----------|---------|
| Commutative | $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$ |
| Distributive | $\mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w}$ |
| Scalar | $(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})$ |

**Why Dot Product Matters in ML:**

1. **Similarity measurement**: High dot product = similar direction
2. **Linear layers**: $y = \mathbf{w}^T\mathbf{x} + b$ is a dot product!
3. **Attention mechanisms**: Query-Key dot products
4. **Projections**: Compute projections using dot products

### 4. Vector Norm (Length/Magnitude)

**Euclidean Norm (L2):**
$$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{\mathbf{v} \cdot \mathbf{v}}$$

**Other Common Norms:**

| Norm           | Formula                                  | Use in ML             |
| -------------- | ---------------------------------------- | --------------------- |
| L1 (Manhattan) | $\|\mathbf{v}\|_1 = \sum_i \|v_i\|$      | Sparse regularization |
| L2 (Euclidean) | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | Ridge regularization  |
| L∞ (Max)       | $\|\mathbf{v}\|_\infty = \max_i \|v_i\|$ | Adversarial bounds    |

### 5. Unit Vectors

A unit vector has magnitude 1:

$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**In ML:** Normalizing vectors to unit length (e.g., cosine similarity, word embeddings).

### 6. Angle Between Vectors

$$\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

**Cosine Similarity:**
$$\text{similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} \in [-1, 1]$$

---

## Vector Spaces

### Definition

A **vector space** $V$ over a field $F$ is a set with two operations:

1. Vector addition: $+: V \times V \to V$
2. Scalar multiplication: $\cdot: F \times V \to V$

Satisfying these axioms:

| Axiom             | Formula                                                                           |
| ----------------- | --------------------------------------------------------------------------------- |
| Associativity (+) | $\mathbf{u} + (\mathbf{v} + \mathbf{w}) = (\mathbf{u} + \mathbf{v}) + \mathbf{w}$ |
| Commutativity (+) | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$                               |
| Identity (+)      | $\mathbf{v} + \mathbf{0} = \mathbf{v}$                                            |
| Inverse (+)       | $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$                                         |
| Compatibility (·) | $a(b\mathbf{v}) = (ab)\mathbf{v}$                                                 |
| Identity (·)      | $1\mathbf{v} = \mathbf{v}$                                                        |
| Distributivity    | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$                          |
| Distributivity    | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$                                   |

### Common Vector Spaces

| Space                     | Description                | ML Example          |
| ------------------------- | -------------------------- | ------------------- |
| $\mathbb{R}^n$            | n-dimensional real vectors | Feature vectors     |
| $\mathbb{R}^{m \times n}$ | Matrices                   | Weight matrices     |
| $C[a,b]$                  | Continuous functions       | Signal processing   |
| $P_n$                     | Polynomials of degree ≤ n  | Polynomial features |

### Linear Independence

Vectors $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_k$ are **linearly independent** if:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

**In ML:** Linearly independent features provide non-redundant information.

### Span

The **span** of vectors $\mathbf{v}_1, ..., \mathbf{v}_k$ is the set of all linear combinations:

$$\text{span}(\mathbf{v}_1, ..., \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{R}\}$$

### Basis

A **basis** is a set of linearly independent vectors that spans the space.

**Standard Basis for $\mathbb{R}^3$:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

### Dimension

The **dimension** of a vector space is the number of vectors in any basis.

$$\dim(\mathbb{R}^n) = n$$

---

## ML Applications

### 1. Feature Vectors

Each data point is represented as a vector:

```python
# Image (28x28 pixels, flattened)
image = np.array([pixel_1, pixel_2, ..., pixel_784])  # R^784

# Text (word embeddings)
word_embedding = np.array([0.2, -0.5, 0.8, ...])  # R^300 (e.g., Word2Vec)

# Tabular data
customer = np.array([age, income, credit_score, ...])  # R^n
```

### 2. Weight Vectors in Linear Models

Linear regression:
$$\hat{y} = \mathbf{w}^T\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

This is just a dot product plus bias!

### 3. Similarity Measures

| Measure            | Formula                                                       | Use Case        |
| ------------------ | ------------------------------------------------------------- | --------------- |
| Dot Product        | $\mathbf{u}^T\mathbf{v}$                                      | Raw similarity  |
| Cosine Similarity  | $\frac{\mathbf{u}^T\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ | Text similarity |
| Euclidean Distance | $\|\mathbf{u} - \mathbf{v}\|_2$                               | KNN, clustering |

### 4. Embeddings as Vectors

Word embeddings capture semantic relationships:

```
king - man + woman ≈ queen

As vectors:
v_king - v_man + v_woman ≈ v_queen
```

---

## Key Formulas

| Concept           | Formula                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| Dot Product       | $\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i$                                                      |
| Magnitude         | $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$                                                              |
| Unit Vector       | $\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|$                                                    |
| Cosine Similarity | $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$                     |
| Projection        | $\text{proj}_\mathbf{v}\mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2}\mathbf{v}$ |

---

## Common Pitfalls

1. **Confusing row and column vectors**
   - Convention: $\mathbf{x}$ usually means column vector
   - $\mathbf{x}^T$ is row vector
2. **Forgetting to normalize**
   - Dot product depends on magnitude
   - Use cosine similarity for direction-only comparison

3. **Dimension mismatch**
   - Can only add vectors of same dimension
   - Dot product requires same dimension

---

## Interview Questions

1. **Q: What's the geometric meaning of the dot product?**
   A: It measures how much two vectors point in the same direction. $\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$. Positive = same direction, zero = perpendicular, negative = opposite.

2. **Q: Why use cosine similarity instead of dot product?**
   A: Cosine similarity is normalized, so it only measures angle (direction), not magnitude. This is useful when you care about orientation, not scale (e.g., comparing documents of different lengths).

3. **Q: How are vectors used in neural networks?**
   A: Input features are vectors, weights are vectors/matrices, activations are vectors. The forward pass is largely vector/matrix operations. Embeddings map discrete items to continuous vectors.

---

## Further Reading

- 📺 [3Blue1Brown - Vectors | Essence of Linear Algebra, Ch. 1](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- 📺 [3Blue1Brown - Linear Combinations, Span, and Basis | Ch. 2](https://www.youtube.com/watch?v=k7RM-ot2NWY)
- 📖 Mathematics for Machine Learning, Ch. 2.1-2.4

---

## Next Steps

→ [Matrix Operations](../02-Matrix-Operations/README.md)
