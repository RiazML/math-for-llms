# Vector Spaces for Machine Learning

[← Previous: Graph Theory](../../11-Graph-Theory) | [Next: Normed Spaces →](../02-Normed-Spaces)

## Overview

Vector spaces form the mathematical foundation for virtually all ML algorithms, from linear regression to deep neural networks. Understanding their properties is essential for analyzing model behavior and designing algorithms.

## Why This Matters for Machine Learning

Vector spaces are not just abstract mathematical constructs—they are the language in which modern machine learning "thinks." Every word embedding in a transformer model lives in a high-dimensional vector space where semantic relationships become geometric ones: "king - man + woman ≈ queen" is a vector space operation. The latent spaces of variational autoencoders and GANs are vector spaces where smooth interpolation between data points becomes possible.

The concept of basis and dimension directly translates to model capacity and representational power. When we talk about the "embedding dimension" of a transformer or the "hidden size" of an LSTM, we're specifying the dimension of vector spaces. Understanding linear independence helps explain why certain features are redundant, while span tells us what predictions are reachable by a model.

Projections—perhaps the most powerful concept in this chapter—underpin everything from PCA to attention mechanisms. The query-key-value attention in transformers can be viewed through the lens of projections and dual spaces. Mastering these concepts provides the geometric intuition needed to design, debug, and improve neural architectures.

## Chapter Roadmap

- **Section 1-2**: Foundations—axioms, common vector spaces in ML (Euclidean, function, matrix, sequence spaces)
- **Section 3-5**: Structure—subspaces, linear independence, basis, and dimension with rank-nullity theorem
- **Section 6-7**: Transformations—linear maps, their matrix representations, and eigenspaces
- **Section 8-10**: Geometry—inner products, orthogonality, and projections (key for least squares)
- **Section 11-12**: Advanced—direct sums, quotient spaces, and dual spaces (connecting to attention)

## Files in This Section

| File | Description |
|------|-------------|
| [examples.ipynb](examples.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## 1. Definition and Axioms

### Vector Space

A **vector space** over field $\mathbb{F}$ (typically $\mathbb{R}$ or $\mathbb{C}$) is a set $V$ with two operations:

- **Addition**: $V \times V \to V$
- **Scalar multiplication**: $\mathbb{F} \times V \to V$

### Vector Space Axioms

For all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $a, b \in \mathbb{F}$:

| Axiom | Property                                                                                          |
| ----- | ------------------------------------------------------------------------------------------------- |
| A1    | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ (Commutativity)                               |
| A2    | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ (Associativity) |
| A3    | $\exists \mathbf{0}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ (Zero vector)                          |
| A4    | $\exists (-\mathbf{v}): \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ (Additive inverse)               |
| M1    | $a(b\mathbf{v}) = (ab)\mathbf{v}$ (Compatibility)                                                 |
| M2    | $1 \cdot \mathbf{v} = \mathbf{v}$ (Identity)                                                      |
| D1    | $a(\mathbf{u} + \mathbf{v}) = a\mathbf{u} + a\mathbf{v}$ (Distributivity)                         |
| D2    | $(a + b)\mathbf{v} = a\mathbf{v} + b\mathbf{v}$ (Distributivity)                                  |

## 2. Common Vector Spaces in ML

### Euclidean Space $\mathbb{R}^n$

$$\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) : x_i \in \mathbb{R}\}$$

**Applications**:

- Feature vectors
- Model parameters
- Embeddings

### Function Spaces

**Continuous functions** $C[a, b]$:
$$C[a, b] = \{f: [a, b] \to \mathbb{R} : f \text{ is continuous}\}$$

**$L^p$ spaces**:
$$L^p = \left\{ f : \int |f(x)|^p dx < \infty \right\}$$

**Applications**:

- Kernel methods
- Functional data analysis
- Neural network function spaces

### Matrix Spaces

$$\mathbb{R}^{m \times n} = \{A : A \text{ is an } m \times n \text{ real matrix}\}$$

**Applications**:

- Weight matrices in neural networks
- Covariance matrices
- Linear transformations

### Sequence Spaces

$$\ell^2 = \left\{ (x_n) : \sum_{n=1}^{\infty} |x_n|^2 < \infty \right\}$$

**Applications**:

- Time series
- RNNs (hidden states over time)
- Infinite-dimensional models

## 3. Subspaces

### Definition

A **subspace** $W$ of $V$ is a subset that is itself a vector space:

1. $\mathbf{0} \in W$
2. $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$ (closed under addition)
3. $\mathbf{v} \in W, c \in \mathbb{F} \Rightarrow c\mathbf{v} \in W$ (closed under scalar multiplication)

### Important Subspaces

**Column Space** (Range):
$$\text{Col}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$$

**Null Space** (Kernel):
$$\text{Null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$$

**Row Space**:
$$\text{Row}(A) = \text{Col}(A^T)$$

### ML Applications

| Subspace           | Application           |
| ------------------ | --------------------- |
| Column space       | Reachable predictions |
| Null space         | Parameter redundancy  |
| Principal subspace | PCA                   |
| Krylov subspace    | Iterative solvers     |

## 4. Linear Independence and Basis

### Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if:
$$c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \Rightarrow c_1 = \cdots = c_k = 0$$

### Span

$$\text{span}(\mathbf{v}_1, \ldots, \mathbf{v}_k) = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{F}\}$$

### Basis

A **basis** is a linearly independent spanning set.

**Properties**:

- Every vector has unique representation
- All bases have same cardinality (dimension)

### Standard Bases

**Euclidean**: $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ where $\mathbf{e}_i$ has 1 in position $i$

**Polynomial**: $\{1, x, x^2, \ldots, x^n\}$ for polynomials of degree $\leq n$

> 💡 **Insight:** Word embeddings like Word2Vec or GloVe learn a basis for semantic space. The "meaning" of a word is its coordinates in this learned basis. When embeddings capture analogies (king - man + woman = queen), they reveal that semantic relationships are encoded as directions in the vector space—a profound connection between geometry and meaning.

## 5. Dimension

### Definition

$$\dim(V) = |B| \text{ where } B \text{ is any basis of } V$$

### Dimension Theorems

**Rank-Nullity Theorem**:
$$\dim(\text{Col}(A)) + \dim(\text{Null}(A)) = n$$

For $A \in \mathbb{R}^{m \times n}$:
$$\text{rank}(A) + \text{nullity}(A) = n$$

### ML Implications

- **Underdetermined systems**: More parameters than constraints
- **Overdetermined systems**: More constraints than parameters
- **Model capacity**: Dimension of hypothesis space

## 6. Linear Transformations

### Definition

$T: V \to W$ is **linear** if:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{v}) = cT(\mathbf{v})$

### Matrix Representation

Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ has matrix representation:
$$T(\mathbf{x}) = A\mathbf{x}$$

where columns of $A$ are $T(\mathbf{e}_1), \ldots, T(\mathbf{e}_n)$.

### Kernel and Image

$$\ker(T) = \{\mathbf{v} : T(\mathbf{v}) = \mathbf{0}\}$$
$$\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

### Important Transformations in ML

| Transformation | Matrix                 | Effect                  |
| -------------- | ---------------------- | ----------------------- |
| Rotation       | Orthogonal             | Preserves lengths       |
| Scaling        | Diagonal               | Stretches axes          |
| Projection     | Idempotent ($P^2 = P$) | Maps to subspace        |
| Reflection     | Orthogonal, det = -1   | Flips across hyperplane |

## 7. Eigenspaces

### Eigenvalue Problem

$$A\mathbf{v} = \lambda\mathbf{v}$$

**Eigenspace** for $\lambda$:
$$E_\lambda = \ker(A - \lambda I) = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\}$$

### Properties

- Eigenspaces are subspaces
- Eigenspaces for distinct eigenvalues are linearly independent
- $\dim(E_\lambda) \leq$ algebraic multiplicity of $\lambda$

### Diagonalization

$A$ is **diagonalizable** if:
$$A = PDP^{-1}$$

where $D$ is diagonal (eigenvalues) and $P$ contains eigenvectors.

**Condition**: Sum of geometric multiplicities equals $n$.

## 8. Inner Product Spaces

### Inner Product

A function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{F}$ satisfying:

1. $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ and $= 0$ iff $\mathbf{v} = \mathbf{0}$ (Positive definite)
2. $\langle \mathbf{u}, \mathbf{v} \rangle = \overline{\langle \mathbf{v}, \mathbf{u} \rangle}$ (Conjugate symmetric)
3. $\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$ (Linearity)

### Standard Inner Products

**Euclidean**:
$$\langle \mathbf{x}, \mathbf{y} \rangle = \sum_{i=1}^{n} x_i y_i = \mathbf{x}^T \mathbf{y}$$

**Weighted**:
$$\langle \mathbf{x}, \mathbf{y} \rangle_M = \mathbf{x}^T M \mathbf{y}$$

where $M$ is positive definite (e.g., Mahalanobis distance).

**Function space**:
$$\langle f, g \rangle = \int_a^b f(x) g(x) dx$$

### Induced Norm

$$\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$$

## 9. Orthogonality

### Orthogonal Vectors

$$\mathbf{u} \perp \mathbf{v} \Leftrightarrow \langle \mathbf{u}, \mathbf{v} \rangle = 0$$

### Orthogonal Complement

$$W^\perp = \{\mathbf{v} \in V : \langle \mathbf{v}, \mathbf{w} \rangle = 0 \text{ for all } \mathbf{w} \in W\}$$

**Property**: $V = W \oplus W^\perp$ (direct sum)

### Orthonormal Basis

Basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$ where:
$$\langle \mathbf{e}_i, \mathbf{e}_j \rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

### Gram-Schmidt Process

Convert basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$ to orthonormal:

$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \text{proj}_{\mathbf{u}_j}(\mathbf{v}_k)$$
$$\mathbf{e}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

> 💡 **Insight:** Layer normalization in transformers is essentially Gram-Schmidt in disguise! It orthogonalizes the representation by removing the mean (centering) and scaling (normalizing). This prevents the "internal covariate shift" problem by ensuring activations don't drift into extreme regions of the space.

## 10. Projections

### Orthogonal Projection

Projection of $\mathbf{v}$ onto subspace $W$:
$$\text{proj}_W(\mathbf{v}) = \arg\min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|$$

### Projection onto Span

For orthonormal basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_k\}$ of $W$:
$$\text{proj}_W(\mathbf{v}) = \sum_{i=1}^{k} \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i$$

**Matrix form**: $P = E E^T$ where $E = [\mathbf{e}_1 | \cdots | \mathbf{e}_k]$

### Projection onto Column Space

$$\text{proj}_{\text{Col}(A)}(\mathbf{b}) = A(A^TA)^{-1}A^T\mathbf{b}$$

**Projection matrix**: $P = A(A^TA)^{-1}A^T$

**Properties**:

- $P^2 = P$ (idempotent)
- $P^T = P$ (symmetric)
- Eigenvalues: 0 or 1

### ML Application: Least Squares

$$\hat{\mathbf{x}} = \arg\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|^2 = (A^TA)^{-1}A^T\mathbf{b}$$

This projects $\mathbf{b}$ onto $\text{Col}(A)$.

## 11. Direct Sums and Quotient Spaces

### Direct Sum

$$V = W_1 \oplus W_2 \Leftrightarrow V = W_1 + W_2 \text{ and } W_1 \cap W_2 = \{\mathbf{0}\}$$

Every $\mathbf{v} \in V$ has unique decomposition: $\mathbf{v} = \mathbf{w}_1 + \mathbf{w}_2$

### Quotient Space

$$V / W = \{\mathbf{v} + W : \mathbf{v} \in V\}$$

Equivalence classes (cosets) form a vector space.

**Dimension**: $\dim(V/W) = \dim(V) - \dim(W)$

## 12. Dual Spaces

### Dual Space

$$V^* = \{f: V \to \mathbb{F} : f \text{ is linear}\}$$

Elements called **linear functionals** or **covectors**.

### Dual Basis

For basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_n\}$, dual basis $\{\mathbf{e}^1, \ldots, \mathbf{e}^n\}$:
$$\mathbf{e}^i(\mathbf{e}_j) = \delta_{ij}$$

> 💡 **Insight:** The key-query mechanism in attention is a dual space pairing! Queries live in the dual space of keys—they're designed to "measure" keys. The dot product $\mathbf{q}^T \mathbf{k}$ is the dual pairing, and softmax normalizes these measurements into a probability distribution. This explains why increasing embedding dimension improves attention quality: higher-dimensional dual spaces can represent more discriminative functionals.

### ML Application: Attention

Attention can be viewed as dual space operation:
$$\text{Attention}(\mathbf{q}, K, V) = \sum_i \text{softmax}(\mathbf{q}^T \mathbf{k}_i) \mathbf{v}_i$$

Query $\mathbf{q}$ acts as functional on keys.

## Summary: Vector Space Concepts in ML

| Concept       | ML Application           |
| ------------- | ------------------------ |
| Basis         | Feature representation   |
| Dimension     | Model capacity           |
| Subspace      | Latent space, PCA        |
| Projection    | Dimensionality reduction |
| Orthogonality | Independent features     |
| Dual space    | Attention mechanisms     |
| Eigenspace    | Spectral methods         |

## Key Takeaways

- **Vector spaces provide the unified language for ML**: Features, parameters, embeddings, and activations all live in vector spaces with the same algebraic rules.

- **Dimension equals capacity**: The dimension of your hypothesis space determines what functions you can represent. Underfitting often means your space is too low-dimensional.

- **Subspaces capture structure**: PCA finds the principal subspace; autoencoders learn nonlinear subspaces (manifolds); attention selects relevant subspaces dynamically.

- **Projections are everywhere**: Least squares, PCA, layer normalization, and residual connections all involve projections onto subspaces.

- **Linear independence prevents redundancy**: Dropout works partly by forcing the network to learn more independent (less redundant) features.

- **The rank-nullity theorem constrains learning**: A weight matrix with low rank has a large null space—many different inputs produce the same output, limiting discriminative power.

- **Dual spaces explain attention**: Queries as linear functionals on keys provides deep geometric insight into why attention works and how to improve it.

## Key Theorems

1. **Rank-Nullity**: $\text{rank}(A) + \text{nullity}(A) = n$
2. **Dimension formula**: $\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$
3. **Orthogonal decomposition**: $V = W \oplus W^\perp$
4. **Isomorphism**: Finite-dimensional spaces of same dimension are isomorphic

## Exercises

1. **Subspace Verification**: Prove that the set of all $n \times n$ symmetric matrices forms a subspace of $\mathbb{R}^{n \times n}$. What is its dimension?

2. **Linear Independence**: Given vectors $\mathbf{v}_1 = (1, 2, 3)$, $\mathbf{v}_2 = (4, 5, 6)$, $\mathbf{v}_3 = (7, 8, 9)$, determine if they are linearly independent. Find a basis for their span.

3. **Rank-Nullity Application**: For a neural network layer with weight matrix $W \in \mathbb{R}^{100 \times 784}$, if $\text{rank}(W) = 50$, what is the dimension of the null space? What does this mean for the network's representational capacity?

4. **Projection Computation**: Compute the orthogonal projection of $\mathbf{b} = (1, 2, 3, 4)$ onto the column space of $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$. Verify that the residual is orthogonal to the column space.

5. **Dual Space in Attention**: Show that the attention mechanism $\text{Attention}(\mathbf{q}, K, V) = \sum_i \text{softmax}(\mathbf{q}^T \mathbf{k}_i) \mathbf{v}_i$ can be interpreted as a weighted linear functional. What role does the query vector play in the dual space interpretation?
