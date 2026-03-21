[Previous: Systems of Equations](../03-Systems-of-Equations/notes.md) | [Home](../../README.md) | [Next: Matrix Rank](../05-Matrix-Rank/notes.md)

---

# Determinants

> _"A determinant turns an entire linear transformation into one number without throwing away its most important geometry: invertibility, orientation, and volume change."_

## Overview

Among all the quantities attached to a square matrix, the determinant is the most compressed and the most deceptive. It is only one scalar, but it simultaneously encodes whether a matrix is invertible, whether it preserves or reverses orientation, how it scales area or volume, and how its eigenvalues multiply together. That is why determinants feel both elementary and deep: the formulas look concrete, but the ideas connect linear algebra, multivariable calculus, probability, geometry, and modern machine learning.

At a geometric level, the determinant answers a simple question:

$$
\text{What happens to volume when the linear map } x \mapsto Ax \text{ acts on space?}
$$

If $A$ maps the unit square, unit cube, or unit hypercube to a parallelogram, parallelepiped, or higher-dimensional analogue, the signed volume of that image is exactly $\det(A)$. The absolute value tells you the volume scaling factor. The sign tells you whether the transformation preserves or flips orientation.

At an algebraic level, the same number answers equally fundamental questions:

- Is the matrix invertible?
- Are its columns linearly independent?
- What is the constant term of its characteristic polynomial?
- What is the product of its eigenvalues?

For machine learning, determinants are not decorative theory. They appear operationally in:

- normalising flows through $\log|\det J|$
- multivariate Gaussian likelihoods through $\log\det(\Sigma)$
- Gaussian process marginal likelihoods through covariance log-determinants
- information geometry through Fisher-metric volume terms
- stability analysis through eigenvalue products and Jacobian determinants
- structured matrix updates through determinant identities such as the matrix determinant lemma

This chapter therefore treats determinants in four intertwined ways:

- geometric meaning
- formal definitions
- efficient computation
- AI-relevant applications

The goal is not to memorize formulas in isolation. It is to understand why all determinant formulas are really statements about the same object seen from different angles.

## Prerequisites

- Matrix multiplication, transpose, and inverse
- Systems of linear equations and row reduction
- Rank, linear dependence, and eigenvalue basics
- Comfort with basic multivariable calculus notation

## Companion Notebooks

| Notebook                           | Description                                                                                                                   |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive determinant computation, geometric volume intuition, log-det examples, and AI-motivated demos                     |
| [exercises.ipynb](exercises.ipynb) | Guided practice on cofactor expansion, characteristic polynomials, determinant identities, log-determinants, and applications |

## Learning Objectives

After completing this chapter, you should be able to:

- Explain the determinant as signed volume scaling and orientation change
- Compute determinants using the Leibniz formula, cofactor expansion, and LU-based elimination
- Use determinant properties correctly under row operations, products, transpose, similarity, and scaling
- Connect determinants to invertibility, rank, eigenvalues, and characteristic polynomials
- Derive and use the adjugate identity and Cramer's rule
- Compute stable log-determinants for SPD matrices and general square matrices
- Explain why triangular Jacobians make normalising flows tractable
- Use determinant identities such as the matrix determinant lemma, Sylvester's theorem, and Schur complements
- Interpret determinant-based quantities in Gaussian models, GPs, DPPs, and information geometry

---

## Table of Contents

- [Determinants](#determinants)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is a Determinant?](#11-what-is-a-determinant)
    - [1.2 The Geometric Picture - Volume and Orientation](#12-the-geometric-picture---volume-and-orientation)
    - [1.3 Why Determinants Matter for AI](#13-why-determinants-matter-for-ai)
    - [1.4 The Determinant as a Function](#14-the-determinant-as-a-function)
    - [1.5 Historical Timeline](#15-historical-timeline)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The Leibniz Formula](#21-the-leibniz-formula)
    - [2.2 The Permutation Group and Signs](#22-the-permutation-group-and-signs)
    - [2.3 The Axiomatic Definition](#23-the-axiomatic-definition)
    - [2.4 Equivalent Characterisations](#24-equivalent-characterisations)
    - [2.5 Cofactor Definition (Recursive)](#25-cofactor-definition-recursive)
  - [3. Computing Determinants](#3-computing-determinants)
    - [3.1 The 2x2 Determinant](#31-the-2x2-determinant)
    - [3.2 The 3x3 Determinant - Sarrus' Rule](#32-the-3x3-determinant---sarrus-rule)
    - [3.3 Cofactor Expansion - Worked Example (4x4)](#33-cofactor-expansion---worked-example-4x4)
    - [3.4 Gaussian Elimination Method (Practical)](#34-gaussian-elimination-method-practical)
    - [3.5 Determinant of Triangular Matrices](#35-determinant-of-triangular-matrices)
    - [3.6 Special Formulas](#36-special-formulas)
  - [4. Properties of Determinants](#4-properties-of-determinants)
    - [4.1 Multiplicativity](#41-multiplicativity)
    - [4.2 Transpose Invariance](#42-transpose-invariance)
    - [4.3 Row and Column Operations](#43-row-and-column-operations)
    - [4.4 Determinant of Products of Special Matrices](#44-determinant-of-products-of-special-matrices)
    - [4.5 Linear Dependence Test](#45-linear-dependence-test)
    - [4.6 Determinant and Eigenvalues](#46-determinant-and-eigenvalues)
  - [5. The Characteristic Polynomial and Eigenvalues](#5-the-characteristic-polynomial-and-eigenvalues)
    - [5.1 Definition of the Characteristic Polynomial](#51-definition-of-the-characteristic-polynomial)
    - [5.2 Finding Eigenvalues via the Characteristic Polynomial](#52-finding-eigenvalues-via-the-characteristic-polynomial)
    - [5.3 The Cayley-Hamilton Theorem](#53-the-cayley-hamilton-theorem)
    - [5.4 Characteristic Polynomial Examples](#54-characteristic-polynomial-examples)
    - [5.5 Resolvent and Green's Function](#55-resolvent-and-greens-function)
  - [6. Cofactor Matrix and Adjugate](#6-cofactor-matrix-and-adjugate)
    - [6.1 The Cofactor Matrix](#61-the-cofactor-matrix)
    - [6.2 The Adjugate (Classical Adjoint)](#62-the-adjugate-classical-adjoint)
    - [6.3 Cramer's Rule](#63-cramers-rule)
    - [6.4 Derivative of the Determinant](#64-derivative-of-the-determinant)
  - [7. Determinants and Geometric Transformations](#7-determinants-and-geometric-transformations)
    - [7.1 Area and Volume](#71-area-and-volume)
    - [7.2 Orientation](#72-orientation)
    - [7.3 Specific Transformations and Their Determinants](#73-specific-transformations-and-their-determinants)
    - [7.4 The Cross Product via Determinants](#74-the-cross-product-via-determinants)
    - [7.5 Gram Determinant](#75-gram-determinant)
  - [8. Determinants in Special Matrix Classes](#8-determinants-in-special-matrix-classes)
    - [8.1 Diagonal and Triangular Matrices](#81-diagonal-and-triangular-matrices)
    - [8.2 Orthogonal Matrices](#82-orthogonal-matrices)
    - [8.3 Symmetric Positive Definite Matrices](#83-symmetric-positive-definite-matrices)
    - [8.4 Vandermonde Matrix](#84-vandermonde-matrix)
    - [8.5 Circulant Matrices](#85-circulant-matrices)
    - [8.6 Tridiagonal Matrices](#86-tridiagonal-matrices)
  - [9. Determinantal Identities](#9-determinantal-identities)
    - [9.1 The Matrix Determinant Lemma](#91-the-matrix-determinant-lemma)
    - [9.2 Sylvester's Determinant Theorem](#92-sylvesters-determinant-theorem)
    - [9.3 Weinstein-Aronszajn Identity](#93-weinstein-aronszajn-identity)
    - [9.4 Schur Complement and Block Determinants](#94-schur-complement-and-block-determinants)
    - [9.5 Cauchy-Binet Formula](#95-cauchy-binet-formula)
  - [10. Log-Determinants in Machine Learning](#10-log-determinants-in-machine-learning)
    - [10.1 Why Log-Determinant?](#101-why-log-determinant)
    - [10.2 Normalising Flows](#102-normalising-flows)
    - [10.3 Architectures Enabling Efficient Log-Det](#103-architectures-enabling-efficient-log-det)
    - [10.4 Multivariate Gaussian Log-Likelihood](#104-multivariate-gaussian-log-likelihood)
    - [10.5 Gaussian Process Marginal Likelihood](#105-gaussian-process-marginal-likelihood)
    - [10.6 Information-Theoretic Role of Log-Det](#106-information-theoretic-role-of-log-det)
  - [11. Determinants in Advanced Topics](#11-determinants-in-advanced-topics)
    - [11.1 Jacobian Determinant in Calculus](#111-jacobian-determinant-in-calculus)
    - [11.2 Functional Determinants](#112-functional-determinants)
    - [11.3 Determinantal Point Processes](#113-determinantal-point-processes)
    - [11.4 Random Matrix Theory and Determinants](#114-random-matrix-theory-and-determinants)
    - [11.5 Determinants in Stability Analysis](#115-determinants-in-stability-analysis)
  - [12. Computational Considerations](#12-computational-considerations)
    - [12.1 Algorithms Comparison](#121-algorithms-comparison)
    - [12.2 Log-Determinant Computation](#122-log-determinant-computation)
    - [12.3 Gradient of Log-Determinant in Autograd](#123-gradient-of-log-determinant-in-autograd)
    - [12.4 Stochastic Log-Det Estimation](#124-stochastic-log-det-estimation)
    - [12.5 Determinants with Low-Rank Structure](#125-determinants-with-low-rank-structure)
  - [13. Determinants and Linear System Theory](#13-determinants-and-linear-system-theory)
    - [13.1 Invertibility and the Determinant](#131-invertibility-and-the-determinant)
    - [13.2 Cramer's Rule and Explicit Formulas](#132-cramers-rule-and-explicit-formulas)
    - [13.3 Determinant Conditions for Solution Uniqueness](#133-determinant-conditions-for-solution-uniqueness)
    - [13.4 Characteristic Polynomial and Eigenvalue Systems](#134-characteristic-polynomial-and-eigenvalue-systems)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI (2026 Edition)](#16-why-this-matters-for-ai-2026-edition)
  - [17. Conceptual Bridge](#17-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Is a Determinant?

The determinant is a function

$$
\det : \mathbb{R}^{n \times n} \to \mathbb{R}
$$

that assigns a single scalar to every square matrix. The remarkable fact is not that such a function exists. The remarkable fact is how much it knows.

From one number, we can tell:

- whether the matrix is invertible
- whether its columns are linearly independent
- how it scales $n$-dimensional volume
- whether it preserves or flips orientation
- what the product of its eigenvalues is

So while a matrix has $n^2$ entries, the determinant distills its most global linear effect into one scalar.

The determinant should be thought of as answering this geometric question:

```text
Take a unit box in n dimensions.
Apply the linear map A.

How much does its signed volume change?
```

If the answer is zero, the transformation crushes space into a lower-dimensional object. If the answer is non-zero, the transformation preserves dimension and therefore remains invertible.

This is why

$$
\det(A) = 0 \iff A \text{ is singular}
$$

is not an isolated theorem. It is a geometric inevitability.

### 1.2 The Geometric Picture - Volume and Orientation

In two dimensions, the determinant of

$$
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
$$

is

$$
\det(A) = ad - bc.
$$

If the columns of $A$ are the vectors

$$
u =
\begin{pmatrix}
a \\
c
\end{pmatrix},
\qquad
v =
\begin{pmatrix}
b \\
d
\end{pmatrix},
$$

then $|\det(A)|$ is exactly the area of the parallelogram spanned by $u$ and $v$.

```text
2D picture

v
^
|      /
|     /
|    /    parallelogram area = |det([u v])|
|   /___
|  /   /
| /   /
|/___/------> u
```

The sign matters too.

- $\det(A) > 0$: orientation is preserved
- $\det(A) < 0$: orientation is reversed
- $\det(A) = 0$: the two column vectors are parallel, so the parallelogram collapses to a line

In three dimensions, the same idea becomes the signed volume of the parallelepiped spanned by the three columns.

In $n$ dimensions, nothing conceptually changes. The determinant is the signed $n$-dimensional volume scaling factor of the linear map.

This is one of the most important cases where geometric intuition scales cleanly from low dimension to high dimension.

### 1.3 Why Determinants Matter for AI

Determinants are not just a classical linear algebra topic that happens to show up occasionally in machine learning. They sit inside several major AI computations.

**Normalising flows**

The change-of-variables formula uses the Jacobian determinant:

$$
\log p_X(x)
=
\log p_Z(f^{-1}(x))
+
\log \left| \det \frac{\partial f^{-1}}{\partial x} \right|.
$$

The entire architecture design of coupling flows, autoregressive flows, Glow-style invertible convolutions, and CNFs is about making this determinant or log-determinant tractable.

**Multivariate Gaussian models**

For

$$
x \sim \mathcal{N}(\mu, \Sigma),
$$

the density contains the factor

$$
\det(\Sigma)^{-1/2}.
$$

This is the normalisation term that makes the density integrate to one. Gaussian processes, Bayesian linear regression, Kalman filtering, and many variational models depend on this.

**Spectral structure**

Eigenvalues are defined by the equation

$$
\det(\lambda I - A) = 0.
$$

So the entry point to eigenvalue theory is itself determinant theory.

**Optimization and stability**

The Hessian determinant appears in second-derivative tests. Jacobian determinants help diagnose local invertibility, singularity, and stability in implicit or dynamical models.

### 1.4 The Determinant as a Function

The determinant is best understood not just by formulas, but by its defining properties.

It is the unique function satisfying:

1. **Multilinearity in the columns**
2. **Alternating behaviour under column swaps**
3. **Normalization on the identity**

That is:

- linear in each column separately
- zero if two columns coincide
- sign flips when two columns are swapped
- $\det(I)=1$

These properties are so strong that they determine the determinant uniquely.

This perspective is powerful because it explains why so many facts about determinants are inevitable:

- swapping rows changes sign
- triangular determinants are products of diagonal entries
- equal or dependent columns force determinant zero
- elimination operations preserve or track determinant in predictable ways

So instead of thinking "the determinant is a complicated formula," it is better to think:

```text
The determinant is the unique alternating multilinear volume form
normalised to be 1 on the identity basis.
```

### 1.5 Historical Timeline

- Seki Takakazu and Leibniz both developed determinant-like expressions in the late 17th century.
- Cramer gave the first widely recognised determinant-based explicit rule for solving linear systems.
- Vandermonde and Laplace developed systematic formulas and expansions.
- Cauchy established crucial algebraic properties such as multiplicativity.
- Jacobi connected determinants to calculus through the Jacobian.
- The modern matrix viewpoint then made determinants part of a broader algebraic theory of linear transformations.
- In the 20th and 21st centuries, determinants moved into operator theory, probability, random matrix theory, and computational ML through log-determinants and Jacobian-based likelihood models.

Historically, determinants were first used to solve systems. Only later did their true geometric meaning become central. In modern ML, both roles are alive at once.

---

## 2. Formal Definitions

### 2.1 The Leibniz Formula

For a square matrix $A \in \mathbb{R}^{n \times n}$, the determinant can be defined by the Leibniz formula:

$$
\det(A)
=
\sum_{\sigma \in S_n}
\operatorname{sgn}(\sigma)
\prod_{i=1}^n a_{i,\sigma(i)}.
$$

This looks intimidating at first, but the pattern is precise:

- choose one entry from each row
- choose one entry from each column
- multiply them
- assign a sign based on the parity of the corresponding permutation
- sum over all permutations

For $n=2$, there are only two permutations, so we recover

$$
\det
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
=
ad - bc.
$$

For $n=3$, there are six permutations, producing the usual six-term formula.

The Leibniz formula is exact and conceptually complete, but computationally terrible for large $n$, since it has $n!$ terms.

### 2.2 The Permutation Group and Signs

The sign in the Leibniz formula comes from permutation parity.

The symmetric group $S_n$ contains all permutations of $\{1,\dots,n\}$. Each permutation can be written as a product of transpositions, and its sign is

$$
\operatorname{sgn}(\sigma) = (-1)^k,
$$

where $k$ is the number of transpositions in such a decomposition.

What matters is not the decomposition itself but its parity. Even permutations always have sign $+1$, odd permutations always have sign $-1$.

For $n=3$, the six permutations split into:

- three even permutations
- three odd permutations

This is why the $3 \times 3$ determinant formula has three positive and three negative terms.

The determinant therefore depends not only on which row-column products are chosen, but on the parity structure of those choices.

### 2.3 The Axiomatic Definition

The cleanest abstract definition is:

The determinant is the unique function

$$
\det : \mathbb{R}^{n \times n} \to \mathbb{R}
$$

such that:

1. **Multilinearity**  
   For each column separately,

   $$
   \det(\dots, \alpha u + \beta v, \dots)
   =
   \alpha \det(\dots, u, \dots)
   +
   \beta \det(\dots, v, \dots)
   $$

2. **Alternating property**  
   Swapping any two columns changes the sign:

   $$
   \det(\dots, c_i, \dots, c_j, \dots)
   =
   -\det(\dots, c_j, \dots, c_i, \dots)
   $$

3. **Normalization**
   $$
   \det(I) = 1
   $$

This definition is mathematically elegant because all the familiar determinant formulas follow from it.

It also makes uniqueness believable: expand every column in the standard basis, apply multilinearity, observe that alternating kills every term with repeated basis vectors, and only permutation terms survive.

### 2.4 Equivalent Characterisations

The determinant can be described in several equivalent ways:

- $\det(A)=0$ iff the columns of $A$ are linearly dependent
- $\det(A)\neq 0$ iff $A$ is invertible
- $\det(A)$ is the signed volume scaling of the linear map
- $\det(A)$ is the product of the eigenvalues, counting algebraic multiplicity
- for triangular matrices, $\det(A)$ is the product of diagonal entries

These are not unrelated facts. They are different manifestations of the same underlying object.

### 2.5 Cofactor Definition (Recursive)

Another exact definition is recursive.

Delete row $i$ and column $j$ from $A$ to obtain the minor matrix $M_{ij}$. Its determinant is the minor associated with $(i,j)$. The cofactor is

$$
C_{ij} = (-1)^{i+j}\det(M_{ij}).
$$

Then expansion along row $i$ gives

$$
\det(A) = \sum_{j=1}^n a_{ij} C_{ij},
$$

and expansion along column $j$ gives

$$
\det(A) = \sum_{i=1}^n a_{ij} C_{ij}.
$$

The sign pattern is the familiar checkerboard:

```text
+  -  +  -  ...
-  +  -  +  ...
+  -  +  -  ...
-  +  -  +  ...
```

This definition is theoretically useful and perfect for symbolic manipulation or small matrices, but again computationally poor for large $n$.

---

## 3. Computing Determinants

### 3.1 The 2x2 Determinant

For

$$
A =
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix},
$$

the determinant is

$$
\det(A)=ad-bc.
$$

This is the simplest nontrivial determinant and already captures all the core geometry:

- if $ad-bc=0$, the columns are parallel
- if $ad-bc>0$, orientation is preserved
- if $ad-bc<0$, orientation is reversed

Geometrically, this is the signed area of the parallelogram spanned by the two columns.

### 3.2 The 3x3 Determinant - Sarrus' Rule

For a $3 \times 3$ matrix,

$$
A =
\begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix},
$$

the determinant is

$$
\det(A)=aei+bfg+cdh-ceg-bdi-afh.
$$

Sarrus' rule is a mnemonic for this formula:

```text
a b c | a b
d e f | d e
g h i | g h

positive diagonals:  aei + bfg + cdh
negative diagonals:  ceg + afh + bdi
```

Important warning:

```text
Sarrus' rule works only for 3x3 matrices.
It does not generalise.
```

### 3.3 Cofactor Expansion - Worked Example (4x4)

For larger symbolic determinants, cofactor expansion is practical only when the matrix has a good row or column with many zeros.

Suppose

$$
A =
\begin{pmatrix}
1 & 2 & 0 & 1 \\
0 & 3 & 0 & 2 \\
0 & 0 & 4 & 1 \\
0 & 0 & 0 & 5
\end{pmatrix}.
$$

This matrix is already upper triangular, so we should not expand at all; we should use the triangular rule. But if we did cofactor-expand along the first column, only one term would survive.

This example teaches the real lesson:

```text
The best determinant method depends on structure.
Zeros are opportunities.
Triangular form is the goal.
```

### 3.4 Gaussian Elimination Method (Practical)

For numerical work, the practical determinant algorithm is elimination.

If Gaussian elimination with partial pivoting gives

$$
PA = LU,
$$

then

$$
\det(A)
=
\det(P^{-1})\det(L)\det(U).
$$

Now:

- $\det(L)=1$ for unit lower triangular $L$
- $\det(U)=\prod_i U_{ii}$
- $\det(P)=(-1)^k$, where $k$ is the number of row swaps

Therefore

$$
\det(A) = (-1)^k \prod_i U_{ii}.
$$

This reduces determinant computation from factorial cost to cubic cost:

$$
O(n!) \quad \longrightarrow \quad O(n^3).
$$

This is why every serious determinant routine for moderate or large dense matrices is LU-based.

### 3.5 Determinant of Triangular Matrices

If $A$ is triangular, then

$$
\det(A)=\prod_{i=1}^n a_{ii}.
$$

The reason is simple. In the Leibniz formula, any non-identity permutation must pick at least one entry above or below the diagonal where the triangular matrix has a zero. So only the identity permutation survives.

This makes diagonal, upper triangular, lower triangular, and block triangular matrices especially nice.

### 3.6 Special Formulas

Several determinant formulas recur constantly in applications.

**Block triangular**

$$
\det
\begin{pmatrix}
A & B \\
0 & D
\end{pmatrix}
=
\det(A)\det(D).
$$

**Block diagonal**

$$
\det(\operatorname{diag}(A_1,\dots,A_k))
=
\prod_{j=1}^k \det(A_j).
$$

**Matrix determinant lemma**

For invertible $A$ and vectors $u,v$,

$$
\det(A+uv^\top)
=
(1+v^\top A^{-1}u)\det(A).
$$

**Schur complement formula**

If $A$ is invertible, then

$$
\det
\begin{pmatrix}
A & B \\
C & D
\end{pmatrix}
=
\det(A)\det(D-CA^{-1}B).
$$

These formulas matter because they turn large determinants into smaller ones and are central in GP updates, low-rank corrections, block systems, and structured models.

---

## 4. Properties of Determinants

### 4.1 Multiplicativity

The defining algebraic property is

$$
\det(AB)=\det(A)\det(B).
$$

This is one of the most powerful identities in linear algebra.

Immediate consequences:

$$
\det(A^{-1}) = \frac{1}{\det(A)},
$$

$$
\det(A^k) = \det(A)^k,
$$

and for scalar $\alpha$,

$$
\det(\alpha A)=\alpha^n \det(A).
$$

The last formula is often misremembered. The exponent $n$ appears because scaling the entire matrix by $\alpha$ scales each of the $n$ columns by $\alpha$.

### 4.2 Transpose Invariance

Determinant is unchanged by transpose:

$$
\det(A^\top)=\det(A).
$$

This means every row-based statement has a corresponding column-based statement and vice versa.

So:

- multilinearity holds in the rows as well as the columns
- swapping two rows also changes sign
- cofactor expansion works along any row or any column

### 4.3 Row and Column Operations

Determinants respond to elementary operations in a very controlled way.

**Swap two rows**

- determinant changes sign

**Scale one row by $\alpha$**

- determinant is multiplied by $\alpha$

**Add a multiple of one row to another**

- determinant is unchanged

The last fact is what makes elimination so useful for determinant computation. Row replacement simplifies the matrix without altering the determinant.

The same statements hold for column operations by transpose invariance.

### 4.4 Determinant of Products of Special Matrices

If $Q$ is orthogonal, then

$$
\det(Q)=\pm 1.
$$

So orthogonal matrices preserve volume magnitude, though not necessarily orientation.

If $P$ is invertible, then

$$
\det(P^{-1}AP)=\det(A).
$$

So determinant is similarity-invariant. It depends on the linear map itself, not on the particular basis representation.

This matters conceptually for ML: changing basis in representation space does not change the determinant of the underlying linear operator.

### 4.5 Linear Dependence Test

The determinant detects full-rank failure:

$$
\det(A)=0
\iff
\text{columns of } A \text{ are linearly dependent}
\iff
\operatorname{rank}(A)<n.
$$

This is one reason determinants became historically tied to system solving. A zero determinant means the square system cannot have a unique solution.

But there is also a numerical warning:

```text
det(A) close to 0 does not reliably mean "numerically close to singular."
```

A tiny determinant may simply come from global scaling or large dimension. Condition number, not determinant magnitude, is the correct practical test for near-singularity.

### 4.6 Determinant and Eigenvalues

For any square matrix,

$$
\det(A)=\prod_{i=1}^n \lambda_i,
$$

where the eigenvalues are counted with algebraic multiplicity over $\mathbb{C}$.

This is one of the deepest bridges in the subject:

- determinant is defined from entries
- eigenvalues are defined spectrally
- the product formula connects the two exactly

For symmetric positive definite matrices, all eigenvalues are positive, so the determinant is positive. For orthogonal matrices, the eigenvalues lie on the unit circle, so the determinant has absolute value $1$.

## 5. The Characteristic Polynomial and Eigenvalues

### 5.1 Definition of the Characteristic Polynomial

Given a square matrix $A \in \mathbb{R}^{n \times n}$, its characteristic polynomial is

$$
p_A(\lambda)=\det(\lambda I-A).
$$

This is a degree-$n$ polynomial in the scalar variable $\lambda$. It is monic, meaning the coefficient of $\lambda^n$ is $1$.

Expanding it gives

$$
p_A(\lambda)
=
\lambda^n - \operatorname{tr}(A)\lambda^{n-1} + \cdots + (-1)^n \det(A).
$$

Several facts are packed into that one line:

- the trace is the sum of eigenvalues
- the determinant is the product of eigenvalues
- the intermediate coefficients are symmetric polynomials in the eigenvalues

So determinants do not merely help with eigenvalues. They define the polynomial whose roots are the eigenvalues.

### 5.2 Finding Eigenvalues via the Characteristic Polynomial

A scalar $\lambda$ is an eigenvalue of $A$ exactly when there exists a nonzero vector $v$ such that

$$
Av=\lambda v.
$$

Rearranging,

$$
(A-\lambda I)v=0.
$$

This homogeneous system has a nontrivial solution exactly when $A-\lambda I$ is singular, so

$$
\lambda \text{ is an eigenvalue}
\iff
\det(A-\lambda I)=0.
$$

That determinant equation is called the **characteristic equation**.

```text
Matrix entries
    ->
det(lambda I - A)
    ->
characteristic polynomial
    ->
roots = eigenvalues
```

For $2 \times 2$ matrices, this leads to a quadratic. For $3 \times 3$, a cubic. Beyond that, the polynomial remains theoretically central, but direct symbolic root-finding quickly becomes the wrong computational tool.

In practical numerical linear algebra, one does not compute eigenvalues by first expanding the characteristic polynomial. One uses QR-like iterative algorithms. The determinant remains conceptually foundational, even when it is not computationally front-and-center.

### 5.3 The Cayley-Hamilton Theorem

One of the most beautiful consequences of the characteristic polynomial is the Cayley-Hamilton theorem:

> Every square matrix satisfies its own characteristic polynomial.

If

$$
p_A(\lambda)=\lambda^n + c_{n-1}\lambda^{n-1}+\cdots + c_1\lambda + c_0,
$$

then

$$
p_A(A)=A^n + c_{n-1}A^{n-1}+\cdots + c_1A + c_0I = 0.
$$

For a $2 \times 2$ matrix,

$$
p_A(\lambda)=\lambda^2-\operatorname{tr}(A)\lambda+\det(A),
$$

so Cayley-Hamilton becomes

$$
A^2-\operatorname{tr}(A)A+\det(A)I=0.
$$

If $\det(A)\neq 0$, this can be rearranged to express the inverse:

$$
A^{-1}=\frac{\operatorname{tr}(A)I-A}{\det(A)}.
$$

That formula is rarely the best numerical method, but it is conceptually revealing. It says the determinant is not merely a scalar summary. It also enters explicit polynomial identities satisfied by the matrix itself.

### 5.4 Characteristic Polynomial Examples

Some standard examples are worth memorising because they calibrate intuition.

**Identity matrix**

$$
p_I(\lambda)=\det(\lambda I-I)=(\lambda-1)^n.
$$

All eigenvalues are $1$, so $\det(I)=1$.

**Zero matrix**

$$
p_0(\lambda)=\lambda^n.
$$

All eigenvalues are $0$, so $\det(0)=0$.

**Projection matrix**

If $P^2=P$ and $\operatorname{rank}(P)=r$, the eigenvalues are $0$ and $1$, so

$$
p_P(\lambda)=\lambda^{n-r}(\lambda-1)^r
$$

and

$$
\det(P)=0
$$

unless $P=I$.

**Rotation matrix in 2D**

For

$$
R_\theta=
\begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix},
$$

the determinant is

$$
\cos^2\theta+\sin^2\theta=1.
$$

So rotations preserve area and orientation.

### 5.5 Resolvent and Green's Function

The **resolvent** of a matrix is

$$
R(\lambda)=(\lambda I-A)^{-1},
$$

defined whenever $\lambda I-A$ is invertible.

That means

$$
R(\lambda) \text{ exists }
\iff
\det(\lambda I-A)\neq 0
\iff
\lambda \text{ is not an eigenvalue of } A.
$$

So the determinant detects exactly where the resolvent breaks down.

This matters in spectral analysis because the poles of the resolvent occur at eigenvalues. In PDEs and operator theory, resolvents lead to Green's functions. In matrix analysis, they give a clean way to think about spectral separation: if $\lambda$ is close to an eigenvalue, the resolvent norm tends to become large.

In machine learning, this viewpoint appears indirectly in:

- stability of recurrent and iterative models
- spectral filtering methods
- graph diffusion operators
- continuous-time linear systems

The determinant is the scalar object that tells you when the resolvent is allowed to exist.

> **Scope of this section:** Section 5 covers the _determinantal_ side of eigenvalue theory — how the characteristic polynomial is defined, why its roots are eigenvalues, and what Cayley-Hamilton says about matrix polynomials. The full eigenvalue story (computation, geometric multiplicity, diagonalization, spectral theorem, Jordan form, applications in gradient dynamics and transformers) is the canonical subject of the next chapter.
>
> → _Full treatment: [Eigenvalues and Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)_

## 6. Cofactor Matrix and Adjugate

### 6.1 The Cofactor Matrix

For each entry $a_{ij}$ of an $n \times n$ matrix $A$, delete row $i$ and column $j$. The determinant of what remains is the **minor** $M_{ij}$.

The corresponding **cofactor** is

$$
C_{ij}=(-1)^{i+j}M_{ij}.
$$

The alternating sign pattern is the familiar checkerboard:

```text
+  -  +  -  ...
-  +  -  +  ...
+  -  +  -  ...
-  +  -  +  ...
```

The matrix of these cofactors is the **cofactor matrix**.

Why does this matter? Because cofactors package every cofactor expansion at once. They are not just bookkeeping devices. They are the entries of the gradient of the determinant and the building blocks of the inverse formula.

### 6.2 The Adjugate (Classical Adjoint)

The **adjugate** of $A$, written $\operatorname{adj}(A)$, is the transpose of the cofactor matrix:

$$
\operatorname{adj}(A)_{ij}=C_{ji}.
$$

Its key identity is

$$
A \operatorname{adj}(A)=\operatorname{adj}(A)A=\det(A)I.
$$

This is one of the most important identities in the chapter.

Why does it work?

- on the diagonal, you recover the cofactor expansion of the determinant
- off the diagonal, you get the determinant of a matrix with two equal rows, which vanishes

So when $\det(A)\neq 0$,

$$
A^{-1}=\frac{\operatorname{adj}(A)}{\det(A)}.
$$

For a $2 \times 2$ matrix,

$$
\operatorname{adj}
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
=
\begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix},
$$

which reproduces the standard inverse formula.

### 6.3 Cramer's Rule

Suppose $Ax=b$ and $\det(A)\neq 0$. Let $A_i$ be the matrix obtained by replacing column $i$ of $A$ with the right-hand side vector $b$. Then

$$
x_i=\frac{\det(A_i)}{\det(A)}.
$$

This is **Cramer's rule**.

Its computational value is low for large systems, but its theoretical value is high:

- it gives an explicit formula for each coordinate of the solution
- it proves uniqueness immediately when $\det(A)\neq 0$
- it shows solutions depend rationally on the data

In modern numerical work, Cramer's rule is almost never used for solving systems. LU or QR is the right tool. But Cramer's rule remains important in theory, symbolic algebra, and derivations involving parameter dependence.

### 6.4 Derivative of the Determinant

A major reason determinants matter in machine learning is that they differentiate cleanly.

For each entry,

$$
\frac{\partial \det(A)}{\partial A_{ij}}=C_{ij}.
$$

In matrix form,

$$
\nabla_A \det(A)=\operatorname{adj}(A)^T.
$$

If $A$ is invertible, using $\operatorname{adj}(A)=\det(A)A^{-1}$ gives

$$
\nabla_A \det(A)=\det(A)A^{-T}.
$$

The log-determinant is even cleaner:

$$
\nabla_A \log|\det(A)|=A^{-T}.
$$

This formula appears constantly in:

- normalising flow training
- Gaussian process hyperparameter optimisation
- covariance learning
- information geometry

There is also a useful scalar derivative identity with respect to parameters:

$$
\frac{d}{d\theta}\log\det A(\theta)
=
\operatorname{tr}\!\left(A(\theta)^{-1}\frac{dA(\theta)}{d\theta}\right),
$$

assuming $A(\theta)$ stays invertible.

This converts a difficult-looking derivative of a determinant into a trace of a matrix product, which is much easier to manipulate analytically and computationally.

## 7. Determinants and Geometric Transformations

### 7.1 Area and Volume

The cleanest geometric interpretation of the determinant is volume scaling.

If the columns of $A$ are the vectors $v_1,\dots,v_n$, then

$$
|\det(A)|
$$

is the volume of the parallelepiped spanned by those columns.

```text
In 2D:
columns -> parallelogram
|det|   -> area

In 3D:
columns -> parallelepiped
|det|   -> volume

In nD:
columns -> n-dimensional parallelotope
|det|   -> n-volume
```

This interpretation is not just intuition. It is the reason the determinant appears in the change-of-variables theorem from multivariable calculus.

### 7.2 Orientation

Absolute value gives size change. The sign gives orientation.

Two ordered bases of $\mathbb{R}^n$ have the same orientation if the change-of-basis matrix between them has positive determinant, and opposite orientation if the determinant is negative.

So:

- $\det(A)>0$: orientation preserved
- $\det(A)<0$: orientation reversed
- $\det(A)=0$: orientation is no longer meaningful because the map collapses dimension

```text
det > 0   preserve handedness
det < 0   flip handedness
det = 0   flatten space
```

This is why reflections have determinant $-1$ while rotations have determinant $+1$.

### 7.3 Specific Transformations and Their Determinants

Some standard transformations are worth learning as templates.

**Uniform scaling**

For $\alpha I$ in $\mathbb{R}^{n \times n}$,

$$
\det(\alpha I)=\alpha^n.
$$

Scaling each axis by $\alpha$ scales volume by $\alpha^n$.

**Rotation**

Any proper rotation has determinant $+1$. It preserves both volume and orientation.

**Reflection**

A reflection has determinant $-1$. It preserves volume magnitude but flips orientation.

**Shear**

A shear matrix typically has determinant $1$. It distorts shape but preserves volume.

**Projection**

A nontrivial projection has determinant $0$, since it collapses at least one dimension.

These examples are operationally useful because they let you interpret determinants before calculating them.

### 7.4 The Cross Product via Determinants

In $\mathbb{R}^3$, the cross product can be written formally as a determinant:

$$
u \times v
=
\begin{vmatrix}
e_1 & e_2 & e_3 \\
u_1 & u_2 & u_3 \\
v_1 & v_2 & v_3
\end{vmatrix}.
$$

Expanding along the first row yields

$$
u \times v
=
(u_2v_3-u_3v_2)e_1
-
(u_1v_3-u_3v_1)e_2
+
(u_1v_2-u_2v_1)e_3.
$$

The magnitude satisfies

$$
\|u \times v\| = \|u\|\,\|v\|\sin\theta,
$$

which is the area of the parallelogram spanned by $u$ and $v$.

So determinant structure is hiding inside the cross product too. In three dimensions, oriented area and determinant algebra become the same story told in two different languages.

### 7.5 Gram Determinant

If $v_1,\dots,v_k \in \mathbb{R}^n$, their Gram matrix is

$$
G_{ij}=\langle v_i,v_j\rangle.
$$

If $V=[v_1 \ \cdots \ v_k]$, then

$$
G=V^TV.
$$

The determinant of $G$ satisfies:

- $\det(G)\geq 0$
- $\det(G)=0$ iff the vectors are linearly dependent
- $\sqrt{\det(G)}$ is the $k$-dimensional volume of the parallelepiped spanned by the vectors

This is a subtle but important extension:

- ordinary determinant measures volume when the spanning vectors live in the same dimension as the space
- Gram determinant measures the intrinsic volume of $k$ vectors inside a possibly larger ambient space

That distinction matters in high-dimensional ML, where one often studies a small set of vectors inside a very large representation space.

## 8. Determinants in Special Matrix Classes

### 8.1 Diagonal and Triangular Matrices

For diagonal and triangular matrices, determinant computation collapses to the simplest possible formula:

$$
\det(A)=\prod_{i=1}^n a_{ii}.
$$

For diagonal matrices this is obvious from the Leibniz formula: only the identity permutation contributes.

For triangular matrices the same reasoning applies. Any non-identity permutation must select at least one off-diagonal zero, so every non-identity term vanishes.

This fact is why LU factorisation is so powerful. Once a matrix has been reduced to triangular form, determinant computation becomes just a signed product of pivots.

```text
A --elimination--> U

det(A) = sign_from_swaps * product(diagonal of U)
```

### 8.2 Orthogonal Matrices

If $Q$ is orthogonal, then

$$
Q^TQ=I.
$$

Taking determinants gives

$$
\det(Q)^2=\det(I)=1,
$$

so

$$
\det(Q)\in\{+1,-1\}.
$$

This means orthogonal matrices preserve volume magnitude exactly.

- $\det(Q)=+1$: rotation-type behaviour
- $\det(Q)=-1$: reflection-type behaviour

This is one reason orthogonal initialisation is so useful in deep learning. A matrix with singular values near $1$ avoids exploding or vanishing signal magnitude, and the determinant provides the most global version of that statement: no overall volume collapse or explosion occurs when $|\det(Q)|=1$.

### 8.3 Symmetric Positive Definite Matrices

If $A$ is symmetric positive definite (SPD), then all eigenvalues are positive, so

$$
\det(A)=\prod_{i=1}^n \lambda_i > 0.
$$

This ensures the log-determinant is real:

$$
\log\det(A)=\sum_{i=1}^n \log\lambda_i.
$$

If $A=LL^T$ is the Cholesky factorisation, then

$$
\det(A)=\det(L)^2=\left(\prod_{i=1}^n L_{ii}\right)^2
$$

and therefore

$$
\log\det(A)=2\sum_{i=1}^n \log L_{ii}.
$$

This identity is central in:

- Gaussian likelihoods
- Gaussian process marginal likelihoods
- kernel methods
- covariance estimation

It is numerically far better than computing the determinant directly and then taking a logarithm.

### 8.4 Vandermonde Matrix

The Vandermonde matrix associated with numbers $x_1,\dots,x_n$ is

$$
V=
\begin{pmatrix}
1 & 1 & \cdots & 1 \\
x_1 & x_2 & \cdots & x_n \\
x_1^2 & x_2^2 & \cdots & x_n^2 \\
\vdots & \vdots & \ddots & \vdots \\
x_1^{n-1} & x_2^{n-1} & \cdots & x_n^{n-1}
\end{pmatrix}.
$$

Its determinant is

$$
\det(V)=\prod_{1\leq i<j\leq n}(x_j-x_i).
$$

So:

- it is zero exactly when two nodes coincide
- it is nonzero exactly when polynomial interpolation on distinct nodes is unique

This is one of the great closed-form determinant formulas in classical linear algebra.

### 8.5 Circulant Matrices

A circulant matrix is determined entirely by its first row, and each later row is a cyclic shift of the previous one.

These matrices are diagonalised by the discrete Fourier transform (DFT) matrix, so their eigenvalues are given by the Fourier transform of the first row. Therefore

$$
\det(C)=\prod_{k=1}^n \lambda_k,
$$

where those $\lambda_k$ are Fourier-domain quantities.

This is a useful example because it shows how structure converts determinant computation from generic $O(n^3)$ work into something closer to FFT cost.

In ML, circulant and convolution-like structure appears in:

- convolutional kernels
- FFT-based linear layers
- structured state-space models
- fast kernel methods

### 8.6 Tridiagonal Matrices

For a tridiagonal matrix, determinants satisfy a recurrence relation rather than requiring full elimination.

If the diagonal entries are $a_i$, upper diagonal entries $b_i$, and lower diagonal entries $c_i$, then the determinant $d_n$ of the leading $n \times n$ block satisfies

$$
d_n = a_n d_{n-1} - b_{n-1}c_{n-1} d_{n-2},
$$

with appropriate initial conditions.

This reduces the cost from cubic to linear time for that special structure.

That matters in PDE discretisations, Kalman-style banded systems, and any structured model where nearest-neighbour interactions dominate.

## 9. Determinantal Identities

### 9.1 The Matrix Determinant Lemma

For invertible $A$ and vectors $u,v$,

$$
\det(A+uv^T)=\left(1+v^TA^{-1}u\right)\det(A).
$$

This is the **matrix determinant lemma**.

It says a rank-1 perturbation of a matrix changes the determinant by a scalar correction factor rather than requiring a full recomputation.

That is already useful on its own, but the deeper lesson is structural:

```text
full n x n determinant
    +
low-rank update
        ->
small correction problem
```

For low-rank updates $UV^T$ with $U,V \in \mathbb{R}^{n \times k}$, the identity generalises to

$$
\det(A+UV^T)=\det(A)\det(I_k+V^TA^{-1}U).
$$

Now an $n \times n$ determinant becomes a $k \times k$ determinant, which is a massive computational win when $k \ll n$.

### 9.2 Sylvester's Determinant Theorem

If $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times m}$, then

$$
\det(I_m+AB)=\det(I_n+BA).
$$

The matrices on the two sides do not even have the same size, yet the determinants agree.

This is a profoundly useful identity because it allows you to move the determinant to the smaller side.

If $m \ll n$, compute the left side. If $n \ll m$, compute the right side.

In ML this matters whenever a covariance, Hessian approximation, or low-rank adapter can be written as "identity plus low-rank product".

### 9.3 Weinstein-Aronszajn Identity

Closely related identities let us factor determinant changes under perturbation:

$$
\det(A-B)=\det(A)\det(I-A^{-1}B),
$$

whenever $A$ is invertible.

This is conceptually the same move:

- pull out the large, known matrix
- reduce the new determinant to a perturbation around identity

The identity is especially useful when $A^{-1}B$ is low-rank, small in norm, or has special structure.

### 9.4 Schur Complement and Block Determinants

For a block matrix

$$
M=
\begin{pmatrix}
A & B \\
C & D
\end{pmatrix},
$$

if $A$ is invertible, then

$$
\det(M)=\det(A)\det(D-CA^{-1}B).
$$

The matrix

$$
D-CA^{-1}B
$$

is the **Schur complement** of $A$ in $M$.

This identity is everywhere in applied mathematics because it converts a large determinant into:

- determinant of a block
- determinant of a smaller corrected block

It underlies block Gaussian elimination, conditional Gaussians, saddle-point systems, and many structured probabilistic models.

### 9.5 Cauchy-Binet Formula

The Cauchy-Binet formula generalises $\det(AB)=\det(A)\det(B)$ to rectangular matrices.

If $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times m}$ with $m \leq n$, then

$$
\det(AB)
=
\sum_{S \subseteq \{1,\dots,n\}, |S|=m}
\det(A_{:,S})\det(B_{S,:}).
$$

This looks technical, but its meaning is geometric: the determinant of the composed map can be decomposed into contributions from all $m$-dimensional coordinate selections.

It appears naturally in:

- volume identities
- exterior algebra
- determinantal point process theory
- low-rank approximation arguments

## 10. Log-Determinants in Machine Learning

### 10.1 Why Log-Determinant?

Determinants grow or shrink exponentially in dimension. That makes raw determinant values numerically fragile.

For example,

$$
\det(2I_{1000})=2^{1000},
\qquad
\det(0.5I_{1000})=0.5^{1000}.
$$

One overflows; the other underflows.

The log-determinant fixes this:

$$
\log|\det(2I_{1000})| = 1000\log 2,
$$

which is perfectly manageable.

This is why modern probabilistic ML almost always uses $\log\det$ rather than $\det$ directly.

There is also an optimization reason:

$$
\nabla_A \log|\det(A)|=A^{-T}
$$

is much cleaner than differentiating the determinant itself.

### 10.2 Normalising Flows

Normalising flows define an invertible map

$$
x=f(z)
$$

that transforms a simple base distribution into a more complex one.

The change-of-variables formula says

$$
\log p_X(x)=\log p_Z(z)-\log\left|\det\frac{\partial f}{\partial z}\right|.
$$

So every flow model lives or dies by the cost of computing

$$
\log|\det J_f(z)|.
$$

This is not an implementation detail. It is the central architectural constraint.

If the Jacobian is dense and unstructured, the cost is generically cubic in dimension. That is too expensive for large models. Therefore flow architectures are designed so the Jacobian is:

- triangular
- block triangular
- diagonal plus structured corrections
- tractable via traces in continuous-time settings

### 10.3 Architectures Enabling Efficient Log-Det

There are several standard design patterns.

**Autoregressive flows**

Each output depends only on earlier inputs, so the Jacobian is triangular. For triangular matrices,

$$
\log|\det J| = \sum_i \log|J_{ii}|.
$$

This turns an $O(n^3)$ problem into an $O(n)$ one.

**Coupling layers**

Part of the input is copied, while the rest is scaled and shifted using functions of the copied part. The Jacobian becomes block triangular, so again the log-determinant is just a sum over easy diagonal terms.

**Invertible 1x1 convolutions**

Glow-style models use learned invertible channel mixing. If $W$ is parameterised with LU structure, then

$$
\log|\det W|
$$

can be computed from the diagonal of the triangular factor.

**Continuous normalising flows**

Instead of computing a full determinant of a Jacobian, one uses the instantaneous identity

$$
\frac{d}{dt}\log p(z(t)) = -\operatorname{tr}\left(\frac{\partial v}{\partial z}\right),
$$

and estimates traces stochastically.

The pattern is always the same:

```text
generic Jacobian -> too expensive
structured Jacobian -> cheap log-det
```

### 10.4 Multivariate Gaussian Log-Likelihood

For

$$
x \sim \mathcal{N}(\mu,\Sigma),
$$

the log-density is

$$
\log p(x)
=
-\frac{n}{2}\log(2\pi)
-\frac{1}{2}\log\det(\Sigma)
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu).
$$

The determinant term is the normalization factor. Geometrically, it measures how spread out the Gaussian ellipsoid is.

Large determinant:

- covariance ellipsoid has large volume
- density is more diffuse

Small determinant:

- covariance ellipsoid is narrow or nearly degenerate
- density is more concentrated

For SPD covariance matrices, Cholesky gives

$$
\log\det(\Sigma)=2\sum_i \log L_{ii}.
$$

That is the standard numerically stable implementation.

### 10.5 Gaussian Process Marginal Likelihood

Gaussian processes require the log marginal likelihood

$$
\log p(y|X,\theta)
=
-\frac{1}{2}y^TK_\theta^{-1}y
-\frac{1}{2}\log\det(K_\theta)
-\frac{n}{2}\log(2\pi),
$$

where $K_\theta$ is the kernel matrix plus observation noise.

This creates two hard matrix tasks:

- solve a linear system with $K_\theta$
- compute $\log\det(K_\theta)$

For exact GP inference, Cholesky is the classical answer. For large-scale approximate GP methods, stochastic trace and Lanczos-style log-det estimators become essential.

This is one of the cleanest places in modern ML where determinant theory, numerical linear algebra, and probabilistic modelling meet directly.

### 10.6 Information-Theoretic Role of Log-Det

For a Gaussian random vector,

$$
H(X)=\frac{1}{2}\log\det(2\pi e\,\Sigma).
$$

So differential entropy is directly controlled by the log-determinant of the covariance.

This gives log-det a genuine information-theoretic meaning:

- larger log-det -> more spread -> larger entropy
- smaller log-det -> less spread -> lower entropy

Related quantities also appear in:

- mutual information formulas for Gaussians
- Bayesian experimental design
- feature diversity regularisation
- Fisher information geometry

So when ML objectives contain a log-determinant, they are often measuring some combination of volume, uncertainty, diversity, or information content.

## 11. Determinants in Advanced Topics

### 11.1 Jacobian Determinant in Calculus

For a differentiable map

$$
f:\mathbb{R}^n \to \mathbb{R}^n,
$$

the Jacobian matrix is

$$
J_f(x)=\left[\frac{\partial f_i}{\partial x_j}\right].
$$

Its determinant measures the local volume scaling of the map near the point $x$.

- $|\det J_f(x)| > 1$: local expansion
- $|\det J_f(x)| < 1$: local contraction
- $\det J_f(x)=0$: local singularity

The inverse function theorem says:

$$
\det J_f(x)\neq 0
\implies
f \text{ is locally invertible near } x.
$$

That theorem is the nonlinear analogue of the fact that a square matrix is invertible exactly when its determinant is nonzero.

### 11.2 Functional Determinants

In infinite-dimensional analysis, determinants generalise to operators.

One important example is the **Fredholm determinant**, written formally as

$$
\det(I+K)=\prod_i (1+\lambda_i),
$$

for suitable trace-class operators $K$.

This idea appears in:

- PDE and operator theory
- statistical physics
- quantum field theory
- continuous-time probabilistic models

In ML, the finite-dimensional determinant story survives in approximate form through Jacobian traces, spectral sums, and operator-inspired kernels.

### 11.3 Determinantal Point Processes

A determinantal point process (DPP) is a probability distribution over subsets where

$$
P(Y=S)\propto \det(K_S)
$$

for a positive semidefinite kernel matrix $K$ and principal submatrix $K_S$.

Why determinant? Because $\det(K_S)$ measures the volume spanned by the feature embeddings of the selected items. Large determinant means the selected items are both high-quality and diverse.

This creates **repulsion**:

- redundant items have similar feature vectors
- similar vectors reduce the determinant
- diverse sets get higher probability

That makes DPPs natural for:

- diverse retrieval
- extractive summarisation
- representative subset selection
- active learning

### 11.4 Random Matrix Theory and Determinants

Random matrix theory studies eigenvalue distributions of random matrices, and determinants appear all over the subject because the joint eigenvalue density often contains Vandermonde-type determinant factors.

This matters for modern ML because spectra of trained weight matrices often show structured deviations from classical random baselines. Those deviations are informative about:

- effective rank
- heavy-tailed structure
- noise versus signal
- generalisation-related geometry

Even when one is not computing determinants directly, determinant identities live in the background of spectral density theory.

### 11.5 Determinants in Stability Analysis

For linear dynamics

$$
\dot{x}=Ax
$$

or discrete dynamics

$$
x_{t+1}=Ax_t,
$$

the eigenvalues of $A$ control stability, and the determinant gives their product.

That makes determinant a coarse but meaningful summary of total expansion or contraction.

In recurrent models, one often cares more directly about singular values than determinants, but the determinant still carries interpretable global information:

- $|\det(A)| \gg 1$: strong global volume expansion
- $|\det(A)| \ll 1$: strong global contraction
- $|\det(A)| = 1$: overall volume preserved

This is why orthogonal and unitary constructions are associated with stable signal propagation.

## 12. Computational Considerations

### 12.1 Algorithms Comparison

In practice, determinant computation is not about formulas first. It is about choosing the right factorisation for the matrix class.

| Method             | Cost                                | Stability                       | Best use                       |
| ------------------ | ----------------------------------- | ------------------------------- | ------------------------------ |
| Leibniz formula    | $O(n!)$                             | Exact but combinatorial         | Only tiny symbolic cases       |
| Cofactor expansion | Exponential in general              | Fine for hand work              | Small matrices, many zeros     |
| LU factorisation   | $O(n^3)$                            | Good with pivoting              | General dense square matrices  |
| Cholesky           | $O(n^3)$ setup, cheap log-det after | Excellent for SPD               | Covariance and kernel matrices |
| Eigenvalue product | $O(n^3)$                            | Fine if spectrum already needed | Spectral analysis              |

The practical rule is simple:

- hand computation -> cofactor / structure
- code -> LU or Cholesky

### 12.2 Log-Determinant Computation

Never compute a determinant and then take its logarithm if numerical stability matters.

Instead use:

- LU-based `slogdet` for general matrices
- Cholesky-based formulas for SPD matrices

Conceptually:

```text
det(A)       -> can overflow / underflow
log|det(A)|  -> stable scale
sign + logabsdet -> safest representation
```

That is why libraries such as NumPy, SciPy, PyTorch, and JAX expose sign-and-log-determinant APIs rather than encouraging raw determinant use in probabilistic objectives.

### 12.3 Gradient of Log-Determinant in Autograd

Autodiff frameworks implement

$$
\nabla_A \log|\det(A)|=A^{-T}
$$

through stable matrix factorizations rather than symbolic expansion.

This matters because a naive determinant implementation would be:

- slow
- unstable
- disastrous for gradients

In practice, gradient flow through log-det is usually routed through LU, QR, or Cholesky internals depending on matrix structure.

### 12.4 Stochastic Log-Det Estimation

For very large SPD matrices, exact $O(n^3)$ log-det is too expensive.

A standard trick is

$$
\log\det(A)=\operatorname{tr}(\log A).
$$

Then instead of forming $\log A$ explicitly, one estimates the trace stochastically using random probe vectors and polynomial or Lanczos approximations.

This leads to methods such as:

- Hutchinson trace estimation
- stochastic Lanczos quadrature
- Chebyshev-based trace approximations

These methods are critical in scalable Gaussian process toolkits because they replace dense factorisations with repeated matrix-vector products.

### 12.5 Determinants with Low-Rank Structure

Low-rank structure is determinant gold.

If

$$
A \mapsto A + UV^T
$$

with rank $k \ll n$, then the determinant lemma turns an $n \times n$ problem into a $k \times k$ one.

That is the same general efficiency principle behind many ML approximations:

- low-rank covariance updates
- inducing-point approximations
- LoRA-style matrix updates
- adapter-style structured perturbations

The chapter theme is repeating itself:

```text
generic matrix -> expensive
structured matrix -> determinant becomes tractable
```

## 13. Determinants and Linear System Theory

### 13.1 Invertibility and the Determinant

For a square matrix,

$$
A \text{ invertible } \iff \det(A)\neq 0.
$$

This is the determinant's most famous theorem, but it should be understood as the synthesis of many equivalent statements:

$$
\det(A)\neq 0
\iff
\operatorname{rank}(A)=n
\iff
\operatorname{null}(A)=\{0\}
\iff
A^{-1}\text{ exists}.
$$

So the determinant is not one test among many. It is one gateway into the entire equivalence class of invertibility statements.

### 13.2 Cramer's Rule and Explicit Formulas

Cramer's rule gives explicit formulas for the coordinates of the solution to

$$
Ax=b.
$$

That makes determinant theory historically inseparable from linear systems. Before modern numerical linear algebra, determinants were studied partly because they gave exact symbolic solution formulas.

Today the computational message is different:

- Cramer's rule explains
- LU solves

The determinant remains conceptually central even when it is not the fastest numerical tool.

### 13.3 Determinant Conditions for Solution Uniqueness

If $A$ is square:

- $\det(A)\neq 0$ -> unique solution for every $b$
- $\det(A)=0$ -> not uniquely solvable for every $b$

But note the subtlety:

$$
\det(A)=0
$$

does **not** by itself tell you whether a particular system has:

- no solution
- infinitely many solutions

For that, one must compare the rank of $A$ and the augmented matrix.

So determinants are decisive for invertibility, but not sufficient by themselves to classify every singular system.

### 13.4 Characteristic Polynomial and Eigenvalue Systems

The eigenvalue equation

$$
\det(A-\lambda I)=0
$$

is itself a determinant-based system condition.

That is the bridge to the next chapter:

- determinants tell you when a shifted matrix becomes singular
- those singular shifts are exactly the eigenvalues
- decomposition theory begins there

## 14. Common Mistakes

| Mistake                                                               | Why it is wrong                                                      | Fix                                               |
| --------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------- | ---------------------- | ------------------------------------- |
| `det(A + B) = det(A) + det(B)`                                        | Determinant is not linear in the whole matrix                        | Use multilinearity one row/column at a time only  |
| `det(AB) = det(A) + det(B)`                                           | Determinant is multiplicative, not additive                          | Remember `det(AB) = det(A)det(B)`                 |
| `det(2A) = 2 det(A)`                                                  | Scaling every row by 2 scales determinant by $2^n$                   | Use `det(alpha A) = alpha^n det(A)`               |
| `det(A) approximately 0 means numerically singular`                   | Determinant magnitude depends on scale and dimension                 | Use condition number to diagnose near-singularity |
| `Sarrus' rule works for 4x4`                                          | It only works for 3x3 matrices                                       | Use cofactor expansion or LU beyond 3x3           |
| `det(A) > 0 means all eigenvalues are positive`                       | Only the product is positive; negative eigenvalues can come in pairs | Check all eigenvalues or SPD criteria             |
| `Adding a multiple of one row changes determinant`                    | Row replacement leaves determinant unchanged                         | Track only swaps and scalings                     |
| `det(A) = 0 tells me whether Ax=b has no solution or infinitely many` | It only tells you the matrix is singular                             | Use the ranks of $A$ and $[A \mid b]$ for classification |
| `log(det(A))` is always real                                          | Not if determinant is negative or matrix is not SPD                  | Use $\log |\det(A)|$ or `slogdet` unless SPD is guaranteed |
| `A large determinant always means good conditioning`                  | Conditioning depends on singular value ratio, not product alone      | Use singular values or condition number           |

## 15. Exercises

1. **Computing determinants**
   Compute the determinant of each matrix using the most efficient method and justify the method choice:
   - $A=\begin{pmatrix}5&3\\2&4\end{pmatrix}$
   - $B=\begin{pmatrix}2&1&0\\0&3&1\\0&0&4\end{pmatrix}$
   - $C=\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$
   - $D=\begin{pmatrix}1&0&0&0\\2&3&0&0\\4&5&6&0\\7&8&9&10\end{pmatrix}$
   - $E=\operatorname{diag}(2,-3,1,4,-1)$

2. **Property verification**
   Let

   $$
   A=\begin{pmatrix}2&1\\1&3\end{pmatrix},
   \qquad
   B=\begin{pmatrix}1&-1\\2&0\end{pmatrix}.
   $$

   Verify:
   - $\det(AB)=\det(A)\det(B)=\det(BA)$
   - $\det(A+B)\neq \det(A)+\det(B)$
   - $\det(3A)=3^2\det(A)$
   - $\det(A^T)=\det(A)$

3. **Characteristic polynomial**
   For

   $$
   A=\begin{pmatrix}4&2\\1&3\end{pmatrix},
   $$

   compute:
   - the characteristic polynomial
   - the eigenvalues
   - eigenvectors
   - a direct verification of Cayley-Hamilton

4. **Geometric interpretation**
   In $\mathbb{R}^2$, let

   $$
   u=\begin{pmatrix}3\\1\end{pmatrix},\qquad
   v=\begin{pmatrix}1\\2\end{pmatrix}.
   $$

   Find the area of the spanned parallelogram, then apply:
   - a rotation
   - a reflection
   - a scaling by factor 3
     and track what happens to the determinant and orientation in each case.

5. **Cofactors and adjugate**
   For

   $$
   A=\begin{pmatrix}1&2&0\\3&1&1\\0&2&1\end{pmatrix},
   $$

   compute:
   - $\det(A)$ by two different cofactor expansions
   - the full cofactor matrix
   - $\operatorname{adj}(A)$
   - $A^{-1}$ from the adjugate identity

6. **Determinant identities**
   - Use the matrix determinant lemma on a diagonal matrix plus a rank-1 update
   - Verify Sylvester's theorem on a small rectangular example
   - Verify the block determinant formula on a block triangular matrix
   - Compute a Schur complement determinant directly and by formula

7. **Log-det for flows**
   Consider a 3D coupling transformation with triangular Jacobian. Write its Jacobian explicitly, identify the diagonal entries, and derive the formula for $\log|\det J|$.

8. **SPD and Gaussian computation**
   For a symmetric positive definite covariance matrix:
   - compute a Cholesky factor
   - derive $\log\det(\Sigma)$ from the diagonal of the factor
   - evaluate the Gaussian log-likelihood for a sample vector

9. **Numerical instability**
   Compare:
   - direct determinant computation of $0.1 I_{50}$
   - stable sign/log-determinant computation
     Explain why the raw determinant is a poor numerical representation.

## 16. Why This Matters for AI (2026 Edition)

| Aspect                 | Impact                                                                             |
| ---------------------- | ---------------------------------------------------------------------------------- |
| Normalising flows      | The log-determinant of the Jacobian is the core term in the likelihood             |
| Multivariate Gaussians | Covariance normalization and entropy both depend on log-det                        |
| Gaussian processes     | Marginal likelihood combines linear solves and log-determinants                    |
| Eigenvalue theory      | The characteristic equation is a determinant equation                              |
| Invertible networks    | Local and global invertibility are determinant statements                          |
| Information geometry   | Fisher-metric volume terms involve determinants and log-determinants               |
| DPP-based diversity    | Determinants score subset diversity via spanned volume                             |
| Stability analysis     | Determinants summarize total volume expansion or contraction of dynamics           |
| Low-rank updates       | Determinant lemmas turn expensive recomputation into small auxiliary problems      |
| Numerical ML systems   | Stable `slogdet`, Cholesky log-det, and stochastic estimators are production tools |

The bigger picture is that determinants are one of the places where algebra and probability truly lock together. In deep learning you often care less about a matrix entry-by-entry and more about what the matrix does globally: does it preserve information, collapse dimensions, distort probability mass, or amplify uncertainty? Determinants answer precisely those questions.

## 17. Conceptual Bridge

The determinant is where matrix theory stops being just arithmetic and becomes geometry.

- In linear algebra, it detects invertibility and dependence.
- In geometry, it measures volume and orientation.
- In calculus, it becomes the Jacobian factor in change of variables.
- In probability, it becomes the normalization term for Gaussian families and invertible generative models.

That is why this chapter sits exactly between matrix operations and spectral theory. Determinants summarize a matrix globally, but they also open the door to a finer structural story:

$$
\det(\lambda I-A)=0.
$$

That single equation launches the study of eigenvalues, eigenspaces, diagonalisation, spectral decompositions, PCA, SVD, and much of modern representation analysis in ML.

```text
Matrix entries
    ->
determinant
    ->
invertibility / volume / orientation
    ->
det(lambda I - A)
    ->
eigenvalues and decomposition theory
```

Next: **Eigenvalues, Eigenvectors, and Matrix Decompositions**.

## References

- Gilbert Strang, _Introduction to Linear Algebra_, Wellesley-Cambridge Press.
- Lloyd N. Trefethen and David Bau III, _Numerical Linear Algebra_, SIAM.
- Gene H. Golub and Charles F. Van Loan, _Matrix Computations_, Johns Hopkins University Press.
- [MIT 18.06 Linear Algebra](https://web.mit.edu/18.06/www/)
- [Stanford EE263: Introduction to Linear Dynamical Systems](https://stanford.edu/class/ee263/)
- [Vaswani et al. (2017), "Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- [Rezende and Mohamed (2015), "Variational Inference with Normalizing Flows"](https://arxiv.org/abs/1505.05770)
- [Dinh, Sohl-Dickstein, and Bengio (2017), "Density Estimation using Real NVP"](https://arxiv.org/abs/1605.08803)
- [Kingma and Dhariwal (2018), "Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)
- [Grathwohl et al. (2018), "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models"](https://arxiv.org/abs/1810.01367)
- [Gardner et al. (2018), "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration"](https://proceedings.neurips.cc/paper/2018/hash/27e8e17134dd7083b050476733207ea1-Abstract.html)
- [GPyTorch documentation: stochastic log-likelihood and Lanczos settings](https://docs.gpytorch.ai/en/v1.8.1/settings.html)
- [GPyTorch `StochasticLQ` implementation notes](https://docs.gpytorch.ai/en/v1.7.0/_modules/gpytorch/utils/stochastic_lq.html)
- [Kulesza and Taskar (2012), _Determinantal Point Processes for Machine Learning_](https://www.nowpublishers.com/article/Details/MAL-044)
- [Chen, Trogdon, and Ubaru (2021), "Analysis of stochastic Lanczos quadrature for spectrum approximation"](https://proceedings.mlr.press/v139/chen21s.html)
