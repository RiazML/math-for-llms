[Previous: Matrix Operations](../02-Matrix-Operations/notes.md) | [Home](../../README.md) | [Next: Determinants](../04-Determinants/notes.md)

---

# Systems of Equations

> _"A model is useful only when its constraints can be satisfied. Systems of equations are the language in which those constraints become explicit, and solving them is the point where mathematical structure turns into actual computation."_

## Overview

A single equation imposes a single condition. A system of equations imposes many conditions at once. That shift from "one condition" to "all of these conditions simultaneously" is the core step from elementary algebra to modern computational mathematics. It is also the step that makes linear algebra operational in AI.

In geometric language, each equation cuts down the set of allowable points. In algebraic language, each equation adds a constraint. In computational language, the entire problem becomes: find the variable assignment that satisfies the full constraint set, or determine that no such assignment exists. Everything that follows in numerical linear algebra, optimization, probabilistic inference, and deep learning depends on this viewpoint.

For machine learning, systems of equations appear everywhere:

- linear regression solves the normal equations
- second-order optimization solves Hessian systems
- Gaussian processes solve large symmetric positive definite systems
- constrained optimization solves KKT systems
- deep equilibrium models solve nonlinear fixed-point systems
- training itself searches for solutions to the gradient equations $\nabla L(\theta) = 0$

This chapter therefore treats systems of equations in three linked ways:

- as geometric intersections of constraints
- as algebraic objects described by rank, null spaces, and consistency
- as computational problems requiring direct or iterative solvers

The linear case is the foundation because it is both completely understood and universally useful. The nonlinear case matters because every realistic optimization problem eventually becomes nonlinear. Together they explain why solving equations is central to both classical scientific computing and modern AI.

## Prerequisites

- Basic algebra and multivariable notation
- Familiarity with vectors, matrices, rank, null spaces, and subspaces
- Comfort with matrix multiplication and transpose
- Basic understanding of norms and least squares

## Companion Notebooks

| Notebook                           | Description                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [theory.ipynb](theory.ipynb)       | Interactive examples for row reduction, least squares, conditioning, and iterative solvers                   |
| [exercises.ipynb](exercises.ipynb) | Guided practice on consistency, Gaussian elimination, least squares, Newton updates, and AI-flavored systems |

## Learning Objectives

After completing this chapter, you should be able to:

- Translate a system of equations into matrix form $Ax = b$
- Diagnose whether a system has zero, one, or infinitely many solutions
- Use rank conditions to characterize consistency and uniqueness
- Perform Gaussian elimination and interpret REF / RREF correctly
- Separate homogeneous and inhomogeneous solution structure
- Understand least squares as projection onto a column space
- Distinguish direct solvers from iterative solvers and know when each is appropriate
- Explain Newton's method as repeated solution of linearized systems
- Interpret constrained optimization through KKT systems and saddle-point structure
- Connect all of the above directly to linear regression, attention, Gaussian processes, DEQs, and second-order optimization

---

## Table of Contents

- [Systems of Equations](#systems-of-equations)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Are Systems of Equations?](#11-what-are-systems-of-equations)
    - [1.2 Why Systems of Equations Are Central to AI](#12-why-systems-of-equations-are-central-to-ai)
    - [1.3 The Geometric Picture](#13-the-geometric-picture)
    - [1.4 Linear vs Nonlinear Systems](#14-linear-vs-nonlinear-systems)
    - [1.5 The Three Cases and Their Geometry](#15-the-three-cases-and-their-geometry)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Linear Systems - Formulation](#2-linear-systems---formulation)
    - [2.1 Standard Form](#21-standard-form)
    - [2.2 Augmented Matrix](#22-augmented-matrix)
    - [2.3 Existence and Uniqueness - The Fundamental Criterion](#23-existence-and-uniqueness---the-fundamental-criterion)
    - [2.4 Homogeneous vs Inhomogeneous](#24-homogeneous-vs-inhomogeneous)
    - [2.5 Types by Shape of A](#25-types-by-shape-of-a)
  - [3. Gaussian Elimination and Row Reduction](#3-gaussian-elimination-and-row-reduction)
    - [3.1 Elementary Row Operations](#31-elementary-row-operations)
    - [3.2 Row Echelon Form REF](#32-row-echelon-form-ref)
    - [3.3 Reduced Row Echelon Form RREF](#33-reduced-row-echelon-form-rref)
    - [3.4 Gaussian Elimination Algorithm](#34-gaussian-elimination-algorithm)
    - [3.5 Partial and Complete Pivoting](#35-partial-and-complete-pivoting)
    - [3.6 Reading Solutions from RREF](#36-reading-solutions-from-rref)
    - [3.7 Worked Example - Underdetermined System](#37-worked-example---underdetermined-system)
  - [4. Existence, Uniqueness, and the Rank Conditions](#4-existence-uniqueness-and-the-rank-conditions)
    - [4.1 Rank and Solution Structure - Complete Characterisation](#41-rank-and-solution-structure---complete-characterisation)
    - [4.2 Geometric Interpretation of Each Case](#42-geometric-interpretation-of-each-case)
    - [4.3 The Fundamental Theorem of Linear Algebra Applied to Systems](#43-the-fundamental-theorem-of-linear-algebra-applied-to-systems)
    - [4.4 The Fredholm Alternative](#44-the-fredholm-alternative)
    - [4.5 Sensitivity and Condition Number](#45-sensitivity-and-condition-number)
  - [5. Special Linear Systems](#5-special-linear-systems)
    - [5.1 Triangular Systems](#51-triangular-systems)
    - [5.2 Symmetric Positive Definite Systems](#52-symmetric-positive-definite-systems)
    - [5.3 Diagonal and Block Diagonal Systems](#53-diagonal-and-block-diagonal-systems)
    - [5.4 Sparse Linear Systems](#54-sparse-linear-systems)
    - [5.5 Toeplitz and Structured Systems](#55-toeplitz-and-structured-systems)
  - [6. Least Squares Problems](#6-least-squares-problems)
    - [6.1 Motivation and Formulation](#61-motivation-and-formulation)
    - [6.2 Normal Equations](#62-normal-equations)
    - [6.3 QR-Based Least Squares](#63-qr-based-least-squares)
    - [6.4 Regularised Least Squares](#64-regularised-least-squares)
    - [6.5 Weighted Least Squares](#65-weighted-least-squares)
    - [6.6 Total Least Squares](#66-total-least-squares)
  - [7. Iterative Methods for Linear Systems](#7-iterative-methods-for-linear-systems)
    - [7.1 Why Iterative Methods?](#71-why-iterative-methods)
    - [7.2 Stationary Iterative Methods](#72-stationary-iterative-methods)
    - [7.3 Conjugate Gradient Method](#73-conjugate-gradient-method)
    - [7.4 Preconditioning](#74-preconditioning)
    - [7.5 GMRES for Non-Symmetric Systems](#75-gmres-for-non-symmetric-systems)
    - [7.6 Krylov Subspace Methods Summary](#76-krylov-subspace-methods-summary)
  - [8. Nonlinear Systems of Equations](#8-nonlinear-systems-of-equations)
    - [8.1 Formulation](#81-formulation)
    - [8.2 Newton's Method for Nonlinear Systems](#82-newtons-method-for-nonlinear-systems)
    - [8.3 Quasi-Newton Methods](#83-quasi-newton-methods)
    - [8.4 Fixed-Point Iteration](#84-fixed-point-iteration)
    - [8.5 Continuation and Homotopy Methods](#85-continuation-and-homotopy-methods)
    - [8.6 The Gradient Equations as a Nonlinear System](#86-the-gradient-equations-as-a-nonlinear-system)
  - [9. Constrained Systems and Lagrangians](#9-constrained-systems-and-lagrangians)
    - [9.1 Equality-Constrained Systems](#91-equality-constrained-systems)
    - [9.2 Saddle-Point Systems](#92-saddle-point-systems)
    - [9.3 KKT Systems in AI](#93-kkt-systems-in-ai)
    - [9.4 Inequality Constraints and Linear Programming](#94-inequality-constraints-and-linear-programming)
  - [10. Systems Arising Directly in AI](#10-systems-arising-directly-in-ai)
    - [10.1 Normal Equations in Linear Regression](#101-normal-equations-in-linear-regression)
    - [10.2 Attention Normalisation as Constraint System](#102-attention-normalisation-as-constraint-system)
    - [10.3 Deep Equilibrium Models DEQ](#103-deep-equilibrium-models-deq)
    - [10.4 Gaussian Processes and Linear System Solves](#104-gaussian-processes-and-linear-system-solves)
    - [10.5 Newton's Method for Transformer Training](#105-newtons-method-for-transformer-training)
    - [10.6 Physics-Informed Neural Networks PINNs](#106-physics-informed-neural-networks-pinns)
  - [11. Overdetermined and Underdetermined Systems in Practice](#11-overdetermined-and-underdetermined-systems-in-practice)
    - [11.1 The Underdetermined Case and Implicit Bias](#111-the-underdetermined-case-and-implicit-bias)
    - [11.2 Minimum Norm Solutions and Regularisation](#112-minimum-norm-solutions-and-regularisation)
    - [11.3 Structured Underdetermined Systems in Language Models](#113-structured-underdetermined-systems-in-language-models)
    - [11.4 The System Perspective on Self-Attention](#114-the-system-perspective-on-self-attention)
  - [12. Numerical Methods and Stability](#12-numerical-methods-and-stability)
    - [12.1 Floating-Point Arithmetic and Rounding Errors](#121-floating-point-arithmetic-and-rounding-errors)
    - [12.2 Iterative Refinement](#122-iterative-refinement)
    - [12.3 Pivoting Strategies and Numerical Stability](#123-pivoting-strategies-and-numerical-stability)
    - [12.4 Preconditioning for Ill-Conditioned Systems](#124-preconditioning-for-ill-conditioned-systems)
    - [12.5 Condition Number and Practical Impact](#125-condition-number-and-practical-impact)
  - [13. Systems of Equations in Optimisation](#13-systems-of-equations-in-optimisation)
    - [13.1 Gradient Descent as Approximate System Solving](#131-gradient-descent-as-approximate-system-solving)
    - [13.2 Newton's Method and the Hessian System](#132-newtons-method-and-the-hessian-system)
    - [13.3 The Role of Linear System Solves in Second-Order Methods](#133-the-role-of-linear-system-solves-in-second-order-methods)
    - [13.4 Convergence Analysis via Linear Systems](#134-convergence-analysis-via-linear-systems)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI (2026 Perspective)](#16-why-this-matters-for-ai-2026-perspective)
  - [17. Conceptual Bridge](#17-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Are Systems of Equations?

A system of equations is a collection of two or more equations that must all be satisfied at the same time. The word "simultaneously" matters more than anything else. One equation restricts the variables somewhat. Many equations restrict them together, and the solution is whatever survives all of those restrictions at once.

For example, in two variables:

$$
\begin{aligned}
x + y &= 3 \\
2x - y &= 0
\end{aligned}
$$

the first equation alone allows infinitely many points, namely every point on a line. The second equation also allows infinitely many points, again a line. Solving the system means finding the point or points that lie on both lines at the same time.

This generalises immediately:

- one equation in two unknowns gives a curve or a line in $\mathbb{R}^2$
- one equation in three unknowns gives a surface or a plane in $\mathbb{R}^3$
- one linear equation in $n$ unknowns gives a hyperplane in $\mathbb{R}^n$
- a full system gives the intersection of all those constraint sets

Linear systems are special because each equation has degree $1$. Their geometry is flat, their algebra is complete, and their computational methods are highly developed. Nonlinear systems are harder because the constraints can curve, fold, branch, or fail to intersect in complicated ways.

In AI, both matter:

- linear systems appear directly in least squares, Gaussian processes, Kalman filters, and second-order methods
- nonlinear systems appear in optimization, fixed-point models, equilibrium conditions, KKT systems, and PDE-constrained learning

### 1.2 Why Systems of Equations Are Central to AI

Many AI computations that seem unrelated are really system-solving problems in disguise.

**Least squares and linear regression**

If we want to minimize

$$
\|Ax - b\|_2^2,
$$

then the minimizer satisfies the normal equations

$$
A^\top A x = A^\top b.
$$

So even the most basic linear regression problem ends as a linear system solve.

**Optimization**

At a stationary point of a loss $L(\theta)$,

$$
\nabla L(\theta) = 0.
$$

This is a system of $p$ equations in $p$ unknowns when $\theta \in \mathbb{R}^p$. Newton's method solves this nonlinear system by repeatedly solving linear systems involving the Hessian.

**Probabilistic inference**

Gaussian processes, Kalman filters, and many Bayesian update formulas require solving systems such as

$$
K \alpha = y,
$$

where $K$ is a covariance or kernel matrix. The main computational bottleneck is often not "doing Bayesian inference" in the abstract, but solving this linear system accurately and efficiently.

**Attention and normalization**

Softmax enforces constraints: non-negativity and sum-to-one. That is not a linear system, but it is still a simultaneous constraint-satisfaction problem at every query position.

**Constrained generation and RLHF**

Grammar-constrained decoding, schema following, and KL-constrained policy updates are all system-like problems. They involve satisfying many conditions at once, exactly the mentality of equation systems and constrained optimization.

### 1.3 The Geometric Picture

The geometry is the fastest way to build intuition.

One linear equation in two variables,

$$
a_1 x_1 + a_2 x_2 = b,
$$

defines a line in $\mathbb{R}^2$.

One linear equation in three variables,

$$
a_1 x_1 + a_2 x_2 + a_3 x_3 = b,
$$

defines a plane in $\mathbb{R}^3$.

In $\mathbb{R}^n$, one equation

$$
a^\top x = b
$$

defines a hyperplane, which is an affine subspace of codimension $1$.

```text
One equation cuts space once.

R^2:  line
R^3:  plane
R^n:  hyperplane

Many equations = many cuts
Solution = common intersection
```

The three basic outcomes are:

```text
TWO-LINE PICTURE IN R^2

1. Unique solution

   \  /
    \/
    /\
   /  \

2. No solution

   -----
   -----

3. Infinitely many solutions

   =====
   =====

The same logic scales to hyperplanes in higher dimension.
```

This same geometry survives in abstract form in higher dimensions:

- unique solution means the constraints cut down to a single point
- infinitely many solutions means the constraints leave a whole direction or family of directions unconstrained
- no solution means the constraints are incompatible

### 1.4 Linear vs Nonlinear Systems

The distinction between linear and nonlinear systems is structural, not cosmetic.

```text
LINEAR SYSTEMS                    NONLINEAR SYSTEMS
-------------------------------------------------------------
Constraints: hyperplanes          Constraints: curves/surfaces/manifolds
Superposition: yes                Superposition: no
Theory: complete                  Theory: problem-dependent
Algorithms: elimination, QR,      Algorithms: Newton, fixed-point,
            Cholesky, CG                     homotopy, continuation
Solution set: affine subspace     Solution set: isolated roots,
            or empty                          branches, manifolds, empty
```

Linearity means two things happen:

1. the geometry is flat
2. the algebra closes under addition and scalar multiplication in a controlled way

That is why the linear case admits the clean rank-based theory:

- consistency is determined by column space membership
- uniqueness is determined by null-space triviality
- general solutions are particular solution plus null-space directions

None of this survives in the same complete form for nonlinear systems. But nonlinear methods almost always depend on linearization, which is why linear systems remain the local language of almost all nonlinear computation.

### 1.5 The Three Cases and Their Geometry

For a linear system $Ax = b$, there are only three possible outcomes.

**Case 1: unique solution**

- the system is consistent
- the constraints are independent enough to determine every variable
- algebraically, the null space is trivial: $\mathrm{null}(A) = \{0\}$
- when $A$ is square, this means $\mathrm{rank}(A) = n$

**Case 2: infinitely many solutions**

- the system is consistent
- but there are free directions left
- algebraically, $\mathrm{null}(A)$ is nontrivial
- the full solution set is

$$
x = x_p + x_h, \qquad x_h \in \mathrm{null}(A),
$$

so it is an affine subspace

**Case 3: no solution**

- the right-hand side $b$ is incompatible with the column space of $A$
- equivalently, $b \notin \mathrm{col}(A)$
- geometrically, the hyperplanes do not all meet

```text
SOLUTION SET SHAPE

unique solution         infinite solutions        no solution
      point               affine family             empty set

      o                    o----o----o                (none)
                           |    |    |
                           line / plane / etc.
```

The entire chapter is really about characterising these three cases precisely, then solving them efficiently.

### 1.6 Historical Timeline

- Babylonian mathematics already solved small systems by elimination-like reasoning in commercial and geometric contexts.
- The Chinese _Nine Chapters on the Mathematical Art_ described the **fangcheng** method, a tabular elimination scheme that is recognisably proto-Gaussian elimination.
- Cramer introduced determinant-based formulas for square systems in the 18th century.
- Gauss systematised elimination and least squares in astronomical computation.
- Jordan refined the full reduction process now called Gauss-Jordan elimination.
- The 19th century matrix viewpoint, due to Cayley, Sylvester, Frobenius, and others, turned system solving into matrix algebra.
- The 20th century added numerical stability, pivoting, factorisation methods, and large-scale iterative solvers.
- Modern AI inherits all of this: dense solvers for medium-scale exact problems, sparse / iterative solvers for large systems, and structured approximations for trillion-parameter optimization.

The historical pattern is important: the subject developed first from geometry and astronomy, then from algebra, then from numerical analysis, and now from large-scale machine computation. AI sits in the fourth stage, but it still depends on the first three.

---

## 2. Linear Systems - Formulation

### 2.1 Standard Form

A system of $m$ linear equations in $n$ unknowns has the form

$$
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1
$$

$$
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2
$$

$$
\vdots
$$

$$
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m.
$$

The compact matrix form is

$$
Ax = b,
$$

where

- $A \in \mathbb{R}^{m \times n}$ is the coefficient matrix
- $x \in \mathbb{R}^n$ is the unknown vector
- $b \in \mathbb{R}^m$ is the right-hand side

```text
Ax = b

  m x n      n x 1    m x 1

[coeffs]  [unknowns] = [targets]
```

This notation compresses the entire system into one object. It also reveals the key question immediately:

$$
\text{Is } b \text{ in the column space of } A?
$$

If yes, the system is consistent. If no, it is not.

### 2.2 Augmented Matrix

The augmented matrix packages the coefficients and the right-hand side together:

$$
[A \mid b] =
\begin{pmatrix}
a_{11} & \cdots & a_{1n} & b_1 \\
\vdots & \ddots & \vdots & \vdots \\
a_{m1} & \cdots & a_{mn} & b_m
\end{pmatrix}.
$$

Why do this? Because row operations affect the equations and the right-hand side at the same time. Working with $[A \mid b]$ keeps the bookkeeping honest.

```text
[A | b]

coefficient columns | right-hand side
-------------------------------------
a11 a12 ... a1n    | b1
a21 a22 ... a2n    | b2
... ... ... ...    | ...
am1 am2 ... amn    | bm
```

In practice:

- row-reduce $[A \mid b]$
- inspect the pivots and any contradiction rows
- read off the solution type directly

### 2.3 Existence and Uniqueness - The Fundamental Criterion

The basic theorem is the Rouché-Capelli criterion:

- the system is **consistent** iff

$$
\mathrm{rank}(A) = \mathrm{rank}([A \mid b])
$$

- the system has a **unique** solution iff

$$
\mathrm{rank}(A) = \mathrm{rank}([A \mid b]) = n
$$

- the system has **infinitely many** solutions iff

$$
\mathrm{rank}(A) = \mathrm{rank}([A \mid b]) < n
$$

- the system is **inconsistent** iff

$$
\mathrm{rank}(A) < \mathrm{rank}([A \mid b]).
$$

This is the complete classification.

Why it works:

- $\mathrm{rank}(A)$ counts how many independent coefficient constraints there are
- $\mathrm{rank}([A \mid b])$ counts how many independent constraints remain after including the targets
- if the augmented rank is larger, then $b$ introduced an extra independent demand that the columns of $A$ cannot satisfy

### 2.4 Homogeneous vs Inhomogeneous

A system is **homogeneous** if $b = 0$:

$$
Ax = 0.
$$

It is **inhomogeneous** if $b \neq 0$:

$$
Ax = b.
$$

The homogeneous case is special because it is always consistent: $x = 0$ always works.

Its solution set is exactly the null space:

$$
\{x : Ax = 0\} = \mathrm{null}(A).
$$

This is a subspace, not merely an affine set. Its dimension is

$$
\dim(\mathrm{null}(A)) = n - \mathrm{rank}(A),
$$

by rank-nullity.

The inhomogeneous case is built from the homogeneous one. If $x_p$ is any particular solution to $Ax = b$, then every solution has the form

$$
x = x_p + x_h,
\qquad x_h \in \mathrm{null}(A).
$$

This is one of the most important formulas in the chapter. It says:

```text
general solution
= one concrete solution
+ every homogeneous direction that changes nothing
```

So the homogeneous system describes the freedom left over after one particular solution is found.

### 2.5 Types by Shape of A

The shape of $A$ already tells you a lot.

**Square system**: $m = n$

- same number of equations as unknowns
- generically unique when full rank
- singular if rank drops

**Overdetermined system**: $m > n$

- more equations than unknowns
- generically inconsistent
- solved approximately by least squares

**Underdetermined system**: $m < n$

- fewer equations than unknowns
- generically infinitely many solutions if consistent
- often choose the minimum-norm solution

**Exactly determined full-rank square system**

- the classical "nice" case
- unique exact solution
- direct methods like LU are natural

In ML, overdetermined and underdetermined cases are at least as important as square ones:

- regression and curve fitting are overdetermined
- overparameterized learning is underdetermined
- inverse problems are often ill-posed even when square

---

## 3. Gaussian Elimination and Row Reduction

### 3.1 Elementary Row Operations

There are three row operations that preserve the solution set:

1. swap two rows: $R_i \leftrightarrow R_j$
2. scale a row by a non-zero scalar: $R_i \leftarrow \alpha R_i$
3. replace a row with itself plus a multiple of another row:
   $R_i \leftarrow R_i + \alpha R_j$

These are not arbitrary tricks. Each corresponds to a legitimate equation manipulation:

- swapping changes equation order
- scaling keeps the same solution set because dividing back is possible
- adding one equation to another creates an equivalent equation

Each elementary row operation can also be represented as left multiplication by an elementary matrix. That matters later because elimination is really matrix multiplication in disguise.

### 3.2 Row Echelon Form REF

A matrix is in row echelon form if:

1. all zero rows are at the bottom
2. each pivot is to the right of the pivot above it
3. everything below each pivot is zero

Example:

$$
\begin{pmatrix}
1 & 2 & 0 & 3 \\
0 & 1 & 4 & -1 \\
0 & 0 & 2 & 5 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

is in REF.

The pivots tell you which variables are constrained directly. The non-pivot columns are free-variable columns.

```text
Pivot columns  -> determined variables
Free columns   -> free variables / parameters
```

Forward elimination always produces an REF. It is not unique, but the pivot structure encodes the rank and the free-variable count.

### 3.3 Reduced Row Echelon Form RREF

RREF imposes two extra conditions:

1. each pivot equals $1$
2. each pivot is the only non-zero entry in its column

Example:

$$
\begin{pmatrix}
1 & 0 & 3 & 0 \\
0 & 1 & -2 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

This form is stronger than REF and, crucially, unique for a given matrix.

That uniqueness makes RREF a conceptual gold standard: once you have it, the solution structure can be read off immediately. In practice, though, solving via REF plus back substitution is usually cheaper than carrying the full matrix all the way to RREF.

### 3.4 Gaussian Elimination Algorithm

Gaussian elimination has two conceptual stages.

**Forward elimination**

- choose a pivot column
- select a pivot row
- eliminate entries below the pivot
- move to the next submatrix

**Back substitution**

- solve the last pivot variable first
- substitute upward one row at a time

```text
FORWARD ELIMINATION

* * * *
* * * *   ->   * * * *
* * * *        0 * * *
* * * *        0 0 * *
                0 0 0 *

BACK SUBSTITUTION

solve bottom row first, then move upward
```

For an $n \times n$ dense system:

- forward elimination costs on the order of $(1/3)n^3$ arithmetic units
- back substitution costs on the order of $n^2$
- total cost is therefore $O(n^3)$

This cubic cost is why direct dense methods become expensive at scale, and why iterative methods matter for huge sparse systems.

### 3.5 Partial and Complete Pivoting

In exact arithmetic, any non-zero pivot is fine. In floating-point arithmetic, that is false. Choosing a tiny pivot can amplify rounding error badly.

That is why practical elimination uses pivoting.

**No pivoting**

- simplest conceptually
- dangerous numerically
- may divide by a very small number and create huge multipliers

**Partial pivoting**

- in the current column, swap in the row with the largest absolute pivot candidate
- this is the default in most dense direct solvers
- it gives strong practical stability at low extra cost

**Complete pivoting**

- search over the remaining submatrix, not just the current column
- swap both rows and columns
- slightly more stable in theory, but more expensive and less common in practice

```text
PIVOTING IDEA

Bad pivot:

0.000001   *   *        divide by tiny number -> big multipliers -> error growth
  5        *   *
  7        *   *

After partial pivoting:

  7        *   *
  5        *   *
0.000001   *   *
```

For AI, pivoting matters whenever system solves occur inside a larger algorithm:

- Gaussian process regression
- second-order optimization
- implicit differentiation
- structured probabilistic inference

It is easy to think "the math says invertible, so we are safe." Numerical analysis says otherwise. Invertible is not the same as well-conditioned.

### 3.6 Reading Solutions from RREF

Once $[A \mid b]$ is in RREF, the solution set can be read directly.

**Inconsistency**

If a row becomes

$$
[0 \ \ 0 \ \ \cdots \ \ 0 \mid c], \qquad c \neq 0,
$$

then the system says $0 = c$, which is impossible. No solution exists.

**Pivot variables**

Variables corresponding to pivot columns are determined by the equations.

**Free variables**

Variables corresponding to non-pivot columns can be chosen freely. They become parameters.

The general solution is then

$$
x = x_p + t_1 v_1 + \cdots + t_k v_k,
$$

where

- $x_p$ is one particular solution
- $v_1, \ldots, v_k$ form a basis for $\mathrm{null}(A)$
- $k = n - \mathrm{rank}(A)$

This is not just a computational trick. It is the structural theorem of linear systems expressed in coordinates.

### 3.7 Worked Example - Underdetermined System

Consider

$$
\begin{pmatrix}
1 & 2 & 0 & 1 \\
2 & 4 & 1 & 3 \\
0 & 0 & 1 & 1
\end{pmatrix}
x
=
\begin{pmatrix}
1 \\
3 \\
1
\end{pmatrix}.
$$

Its augmented matrix is

$$
\left[
\begin{array}{cccc|c}
1 & 2 & 0 & 1 & 1 \\
2 & 4 & 1 & 3 & 3 \\
0 & 0 & 1 & 1 & 1
\end{array}
\right].
$$

Apply

$$
R_2 \leftarrow R_2 - 2R_1
$$

to get

$$
\left[
\begin{array}{cccc|c}
1 & 2 & 0 & 1 & 1 \\
0 & 0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 & 1
\end{array}
\right].
$$

Then

$$
R_3 \leftarrow R_3 - R_2
$$

gives

$$
\left[
\begin{array}{cccc|c}
1 & 2 & 0 & 1 & 1 \\
0 & 0 & 1 & 1 & 1 \\
0 & 0 & 0 & 0 & 0
\end{array}
\right].
$$

So the pivot columns are $1$ and $3$, while columns $2$ and $4$ are free.

Let

$$
x_2 = t, \qquad x_4 = s.
$$

Then row $2$ gives

$$
x_3 + x_4 = 1 \Rightarrow x_3 = 1 - s,
$$

and row $1$ gives

$$
x_1 + 2x_2 + x_4 = 1 \Rightarrow x_1 = 1 - 2t - s.
$$

Therefore

$$
x =
\begin{pmatrix}
1 - 2t - s \\
t \\
1 - s \\
s
\end{pmatrix}
=
\begin{pmatrix}
1 \\
0 \\
1 \\
0
\end{pmatrix}
+
t
\begin{pmatrix}
-2 \\
1 \\
0 \\
0
\end{pmatrix}
+
s
\begin{pmatrix}
-1 \\
0 \\
-1 \\
1
\end{pmatrix}.
$$

This displays the exact structure:

- one particular solution
- plus a $2$-dimensional null-space family

That is the canonical shape of a consistent underdetermined linear system.

---

## 4. Existence, Uniqueness, and the Rank Conditions

### 4.1 Rank and Solution Structure - Complete Characterisation

For $Ax = b$ with $A \in \mathbb{R}^{m \times n}$ and rank $r$, the possibilities can be summarised as follows.

| rank(A) | rank([A\|b]) | relation to n                 | solution set              |
| ------- | ------------ | ----------------------------- | ------------------------- |
| $r = n$ | $r = n$      | full column rank              | unique solution           |
| $r < n$ | $r < n$      | consistent but rank-deficient | infinitely many solutions |
| $r$     | $r+1$        | augmented rank larger         | no solution               |

More explicitly:

- if $\mathrm{rank}(A) = \mathrm{rank}([A \mid b]) = n$, then there are no free variables
- if $\mathrm{rank}(A) = \mathrm{rank}([A \mid b]) < n$, then there are $n-r$ free variables
- if $\mathrm{rank}(A) < \mathrm{rank}([A \mid b])$, then the system is inconsistent

This is complete. There are no other cases.

### 4.2 Geometric Interpretation of Each Case

It is useful to combine algebra and geometry explicitly.

**Square full-rank system**

- $m = n$, rank $= n$
- hyperplanes meet in a single point
- algebraically: no null directions remain

**Consistent overdetermined system**

- $m > n$
- possible only if extra equations are redundant or implied
- geometrically: many hyperplanes still share the same point

**Generic overdetermined system**

- more constraints than degrees of freedom
- usually inconsistent
- motivates least squares rather than exact solving

**Underdetermined system**

- too few independent constraints to determine every variable
- solutions form an affine family
- dimension of the family is $n-r$

```text
DIMENSION COUNT

n unknown directions
- r independent constraints
--------------------------
n-r remaining free directions
```

This count is one of the simplest and most useful geometric heuristics in all of linear algebra.

### 4.3 The Fundamental Theorem of Linear Algebra Applied to Systems

The four fundamental subspaces of $A \in \mathbb{R}^{m \times n}$ give the cleanest picture of when $Ax = b$ is solvable:

- $\mathrm{col}(A) \subseteq \mathbb{R}^m$ — the set of achievable right-hand sides
- $\mathrm{null}(A) \subseteq \mathbb{R}^n$ — the set of solutions to $Ax = 0$
- $\mathrm{row}(A) \subseteq \mathbb{R}^n$ — the input directions that are not annihilated
- $\mathrm{null}(A^\top) \subseteq \mathbb{R}^m$ — the left null space; directions $b$ cannot lie in for solvability

**Reading solvability from these subspaces:**

- $Ax = b$ is consistent $\iff$ $b \in \mathrm{col}(A)$
- if $x_p$ is any particular solution, every solution is $x_p + v$ for some $v \in \mathrm{null}(A)$
- $b$ has a component in $\mathrm{null}(A^\top)$ $\iff$ the system is inconsistent

The direct-sum decomposition $\mathbb{R}^m = \mathrm{col}(A) \oplus \mathrm{null}(A^\top)$ is what makes the least-squares projection of §6 well-defined: we project $b$ onto $\mathrm{col}(A)$ and discard the $\mathrm{null}(A^\top)$ component.

> **Scope note:** This section uses the four fundamental subspaces as a practical tool for understanding $Ax = b$. Their rigorous definitions, orthogonality relationships, and dimensional identities are the canonical subject of a later section.
>
> → _Full treatment: [Vector Spaces and Subspaces §7](../06-Vector-Spaces-Subspaces/notes.md#7-the-four-fundamental-subspaces)_

### 4.4 The Fredholm Alternative

The Fredholm alternative packages the previous paragraph into a clean either-or statement:

For a given $A$ and $b$, exactly one of the following is true:

1. $Ax = b$ has a solution, or
2. there exists $y$ such that

$$
A^\top y = 0
\qquad \text{and} \qquad
y^\top b \neq 0.
$$

Interpretation:

- if $A^\top y = 0$, then $y$ lies in the left null space
- if $y^\top b \neq 0$, then $b$ has a component orthogonal to the column space
- therefore $b$ cannot lie entirely in $\mathrm{col}(A)$

This theorem is conceptually deeper than it first appears. It says inconsistency is not mysterious. It is witnessed by an explicit certificate $y$ that proves the right-hand side points outside the achievable subspace.

### 4.5 Sensitivity and Condition Number

Not all solvable systems are equally safe to solve numerically.

Suppose $Ax = b$ and the right-hand side is perturbed to $b + \delta b$. Then the solution changes to $x + \delta x$ satisfying

$$
A(x + \delta x) = b + \delta b.
$$

Subtracting $Ax = b$ gives

$$
A \delta x = \delta b,
$$

so

$$
\delta x = A^{-1} \delta b
$$

when $A$ is invertible.

This leads to the standard error bound

$$
\frac{\|\delta x\|}{\|x\|}
\leq
\kappa(A)
\frac{\|\delta b\|}{\|b\|},
$$

where

$$
\kappa(A) = \|A\| \, \|A^{-1}\|
$$

is the condition number in the chosen norm.

Meaning:

- $\kappa(A) \approx 1$ means the system is well-conditioned
- large $\kappa(A)$ means the system strongly amplifies perturbations

```text
small data error  --times-->  condition number  --gives-->  solution error
```

In ML this matters constantly:

- Hessian systems can be very ill-conditioned
- kernel matrices can be nearly singular
- normal equations square the condition number
- regularization often helps more by improving conditioning than by changing the exact minimizer

---

## 5. Special Linear Systems

### 5.1 Triangular Systems

Triangular systems are the cheap end of linear solving.

For a lower triangular system $Lx = b$:

$$
x_i =
\frac{1}{\ell_{ii}}
\left(
b_i - \sum_{j=1}^{i-1} \ell_{ij} x_j
\right),
\qquad i = 1,2,\ldots,n.
$$

This is **forward substitution**.

For an upper triangular system $Ux = b$:

$$
x_i =
\frac{1}{u_{ii}}
\left(
b_i - \sum_{j=i+1}^{n} u_{ij} x_j
\right),
\qquad i = n,n-1,\ldots,1.
$$

This is **back substitution**.

Each costs $O(n^2)$, much cheaper than factorisation. That is why methods such as LU and Cholesky are so useful: they reduce the hard problem to triangular solves.

### 5.2 Symmetric Positive Definite Systems

If $A$ is symmetric positive definite (SPD), then

$$
A^\top = A
\qquad \text{and} \qquad
x^\top A x > 0 \text{ for all } x \neq 0.
$$

This is an especially pleasant class:

- eigenvalues are positive
- the system has a unique solution
- Cholesky factorisation applies:

$$
A = LL^\top
$$

with $L$ lower triangular

Cholesky is faster and more stable than generic LU in this setting, and it avoids pivoting.

SPD systems appear everywhere in ML:

- Gram matrices
- kernel matrices (with regularization)
- covariance matrices
- normal equations when $A$ has full column rank
- local quadratic models near well-behaved minima

### 5.3 Diagonal and Block Diagonal Systems

If $D$ is diagonal, then solving

$$
Dx = b
$$

is trivial:

$$
x_i = \frac{b_i}{d_i}.
$$

This is $O(n)$.

If the matrix is block diagonal, then each block can be solved independently.

```text
[B1  0   0 ] [x1]   [b1]
[0   B2  0 ] [x2] = [b2]
[0   0   B3] [x3]   [b3]

solve B1 x1 = b1, B2 x2 = b2, B3 x3 = b3 separately
```

This separability is a major computational win. Many preconditioners and structured models try to approximate a difficult system by something diagonal or block diagonal precisely because those systems are cheap to solve.

### 5.4 Sparse Linear Systems

A sparse matrix has relatively few non-zero entries. For large systems, sparsity changes everything.

Dense viewpoint:

- storage: $O(n^2)$
- factorisation: $O(n^3)$

Sparse viewpoint:

- storage: $O(\mathrm{nnz})$
- matrix-vector multiply: often $O(\mathrm{nnz})$
- iterative methods become viable

The main subtlety is **fill-in**: elimination may create new non-zero entries where zeros previously existed. Sparse direct solvers therefore spend significant effort on reordering the matrix to reduce fill-in before factorisation.

In AI, sparsity appears in:

- graph neural networks
- sparse attention patterns
- recommender systems
- PDE discretisations in scientific ML
- structured factor graphs and graphical models

### 5.5 Toeplitz and Structured Systems

Some matrices are not sparse but are still highly structured.

**Toeplitz matrix**

- constant along diagonals
- $A_{ij}$ depends only on $i-j$

**Circulant matrix**

- special Toeplitz form
- diagonalised by the discrete Fourier transform
- solvable via FFT in near-linear time

**Vandermonde matrix**

- arises in polynomial interpolation
- numerically delicate but structurally special

Convolution is a good AI example. A 1D convolution can be written as multiplication by a Toeplitz matrix. In practice we do not materialise that matrix, but the structured-system viewpoint is still correct and useful.

---

## 6. Least Squares Problems

### 6.1 Motivation and Formulation

When $Ax = b$ is overdetermined, exact consistency is unlikely. The next best question is:

$$
\text{Which } x \text{ makes } Ax \text{ as close as possible to } b?
$$

This gives the least-squares problem:

$$
x^\star = \arg\min_{x \in \mathbb{R}^n} \|Ax - b\|_2^2.
$$

Geometrically:

- $Ax$ ranges over the column space of $A$
- we want the point in $\mathrm{col}(A)$ closest to $b$
- therefore $Ax^\star$ is the orthogonal projection of $b$ onto $\mathrm{col}(A)$

The residual

$$
r^\star = b - Ax^\star
$$

is orthogonal to the column space at the optimum.

```text
b
|\
| \
|  \   residual r*
|   \
|    o  = projection of b onto col(A)
|
+--------------------------> col(A)
```

Least squares is therefore not merely "approximate solving." It is orthogonal projection in disguise.

### 6.2 Normal Equations

Differentiate the least-squares objective:

$$
f(x) = \|Ax - b\|_2^2 = (Ax-b)^\top (Ax-b).
$$

Then

$$
\nabla f(x) = 2A^\top (Ax-b).
$$

At the minimizer,

$$
A^\top(Ax-b) = 0,
$$

which gives the normal equations

$$
A^\top A x = A^\top b.
$$

If $A$ has full column rank, then $A^\top A$ is SPD and

$$
x^\star = (A^\top A)^{-1} A^\top b.
$$

This formula is exact and important, but also numerically dangerous when $A$ is ill-conditioned because

$$
\kappa(A^\top A) = \kappa(A)^2.
$$

That squared condition number is the main reason serious numerical work often avoids solving least squares through normal equations directly.

### 6.3 QR-Based Least Squares

If $A = QR$ is the thin QR factorisation, with

- $Q \in \mathbb{R}^{m \times n}$ having orthonormal columns
- $R \in \mathbb{R}^{n \times n}$ upper triangular

then

$$
\|Ax - b\|_2
=
\|QRx - b\|_2.
$$

Because $Q$ preserves lengths on its column space, the least-squares problem reduces to

$$
Rx = Q^\top b.
$$

This is an upper triangular solve.

Advantages over normal equations:

- avoids forming $A^\top A$
- avoids squaring the condition number
- numerically more stable

For practical least squares, QR is usually the safer default.

### 6.4 Regularised Least Squares

When $A$ is nearly singular, rank-deficient, or noisy, add an $L^2$ penalty:

$$
x_\lambda^\star
=
\arg\min_x \|Ax-b\|_2^2 + \lambda \|x\|_2^2.
$$

This is Tikhonov regularisation or ridge regression.

The optimality condition becomes

$$
(A^\top A + \lambda I)x = A^\top b.
$$

This does two things at once:

- shrinks the solution toward the origin
- improves conditioning by shifting small eigenvalues upward

In ML, this is not a side topic. Weight decay is exactly this geometry applied to parameter fitting.

### 6.5 Weighted Least Squares

If some observations are more reliable than others, use a positive diagonal weight matrix $W$:

$$
x^\star
=
\arg\min_x (Ax-b)^\top W (Ax-b).
$$

This produces the weighted normal equations

$$
A^\top W A x = A^\top W b.
$$

If $W = \mathrm{diag}(w_1,\ldots,w_m)$, then each residual is penalised according to its importance.

This is natural when:

- measurement errors have unequal variances
- some samples are trusted more than others
- robust procedures iteratively reweight points

### 6.6 Total Least Squares

Ordinary least squares assumes $A$ is exact and only $b$ is noisy. That is often unrealistic.

Total least squares allows perturbations to both:

$$
(A + \delta A)x = b + \delta b.
$$

The goal is to minimise the size of the joint perturbation

$$
\|[\delta A \mid \delta b]\|_F.
$$

The classical solution uses the SVD of the augmented data matrix $[A \mid b]$. The smallest singular direction identifies the smallest correction that makes the system exactly consistent.

This matters when the design matrix itself was measured or estimated with error, which is common in scientific inference and noisy representation learning.

---

## 7. Iterative Methods for Linear Systems

### 7.1 Why Iterative Methods?

Direct solvers are excellent when matrices are moderate in size and dense. But they do not scale indefinitely.

If $n = 10^6$, then:

- storing a dense matrix would require on the order of $10^{12}$ entries
- a dense direct factorisation would require on the order of $10^{18}$ arithmetic operations

That is not realistic.

Iterative methods change the question. Instead of computing the exact answer in finitely many elimination steps, they produce a sequence

$$
x_0, x_1, x_2, \ldots
$$

that ideally converges to the true solution.

The stopping rule is usually based on the residual:

$$
r_k = b - A x_k.
$$

When $\|r_k\|$ is small enough, stop.

The key computational benefit is that many iterative methods only need:

- matrix-vector products
- vector inner products
- inexpensive preconditioner applications

That makes them ideal for large sparse systems where factorising $A$ would be too expensive.

### 7.2 Stationary Iterative Methods

A broad stationary-method template begins by splitting

$$
A = M - N,
$$

where $M$ is chosen so that solving $My = z$ is easy.

Then iterate:

$$
M x_{k+1} = N x_k + b,
\qquad
x_{k+1} = M^{-1} N x_k + M^{-1} b.
$$

Convergence depends on the spectral radius:

$$
\rho(M^{-1}N) < 1.
$$

**Jacobi**

Take $M = D$, the diagonal of $A$.

$$
x_i^{(k+1)}
=
\frac{1}{a_{ii}}
\left(
b_i - \sum_{j \neq i} a_{ij} x_j^{(k)}
\right).
$$

All components use only old values, so Jacobi is naturally parallel.

**Gauss-Seidel**

Use the lower triangular part as the easy system. Updated values are used immediately:

$$
x_i^{(k+1)}
=
\frac{1}{a_{ii}}
\left(
b_i
- \sum_{j < i} a_{ij} x_j^{(k+1)}
- \sum_{j > i} a_{ij} x_j^{(k)}
\right).
$$

This often converges faster than Jacobi but is less parallel-friendly.

**SOR**

Successive over-relaxation adds a parameter $\omega$:

$$
x^{(k+1)} = (1-\omega)x^{(k)} + \omega x_{\text{GS}}^{(k+1)}.
$$

When $\omega$ is well chosen, convergence can improve dramatically.

Stationary methods are conceptually important, but for large modern workloads Krylov methods usually dominate in practice.

### 7.3 Conjugate Gradient Method

For SPD systems, conjugate gradient (CG) is the central iterative method.

It solves

$$
Ax = b
$$

by minimising the quadratic

$$
f(x) = \frac{1}{2}x^\top A x - b^\top x
$$

over increasingly rich Krylov subspaces

$$
\mathcal{K}_k(A, r_0) = \mathrm{span}\{r_0, Ar_0, A^2 r_0, \ldots, A^{k-1}r_0\}.
$$

The algorithm is:

$$
r_0 = b - A x_0, \qquad p_0 = r_0
$$

and then for each step:

$$
\alpha_k = \frac{r_k^\top r_k}{p_k^\top A p_k},
$$

$$
x_{k+1} = x_k + \alpha_k p_k,
$$

$$
r_{k+1} = r_k - \alpha_k A p_k,
$$

$$
\beta_k = \frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k},
$$

$$
p_{k+1} = r_{k+1} + \beta_k p_k.
$$

Important facts:

- in exact arithmetic, CG terminates in at most $n$ steps
- in practice it is useful because it often converges much sooner
- convergence improves when eigenvalues are clustered

The classical bound is

$$
\|x_k - x^\star\|_A
\le
2
\left(
\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}
\right)^k
\|x_0 - x^\star\|_A,
$$

where $\kappa = \kappa(A)$.

So conditioning directly controls iteration count.

### 7.4 Preconditioning

If conditioning controls convergence, the natural question is:

$$
\text{Can we transform the system into one with smaller condition number?}
$$

That is the purpose of preconditioning.

Choose $M \approx A$ such that:

- applying $M^{-1}$ is cheap
- $M^{-1}A$ is better conditioned than $A$

Then solve the transformed system instead.

Common examples:

- Jacobi preconditioner: $M = \mathrm{diag}(A)$
- incomplete LU (ILU)
- incomplete Cholesky for SPD systems
- multigrid preconditioners for PDE-like problems

Preconditioning is often the difference between an iterative solver that is unusably slow and one that converges in a practical number of steps.

In AI terms, normalization layers and adaptive optimizers play a closely related geometric role: they reshape the effective conditioning of the optimization problem.

### 7.5 GMRES for Non-Symmetric Systems

Conjugate gradient requires SPD structure. For general non-symmetric systems, a standard choice is GMRES: Generalised Minimal Residual.

GMRES constructs an orthonormal basis for the Krylov space using the Arnoldi process and chooses the iterate whose residual norm is minimal over that subspace.

Conceptually:

- build a basis for $\mathcal{K}_k(A, r_0)$
- solve a small least-squares problem each iteration
- obtain the best residual achievable in that subspace

Benefits:

- applies to very general systems
- robust in many settings

Costs:

- memory grows with iteration count because the Krylov basis grows
- restarted GMRES($m$) is often used to limit storage

GMRES is a good example of a recurring pattern in numerical linear algebra: turn a hard large system into a sequence of small projection problems.

### 7.6 Krylov Subspace Methods Summary

| Method       | Matrix type                 | Cost per iteration          | Memory  | Typical use                                    |
| ------------ | --------------------------- | --------------------------- | ------- | ---------------------------------------------- |
| Jacobi       | broad but fragile           | sparse matvec + vector ops  | low     | simple baseline, parallel smoothing            |
| Gauss-Seidel | broad but sequential        | sparse matvec-like updates  | low     | small systems, relaxation                      |
| CG           | SPD                         | matvec + a few dot products | low     | kernel systems, normal equations, SPD problems |
| MINRES       | symmetric indefinite        | matvec + vector ops         | low     | saddle-like symmetric systems                  |
| GMRES        | general non-singular        | matvec + orthogonalisation  | growing | non-symmetric large systems                    |
| LSQR         | least squares / rectangular | bidiagonalisation-based     | low     | large sparse least squares                     |

The main lesson is not to memorise solver names. It is to match system structure to algorithm structure.

---

## 8. Nonlinear Systems of Equations

### 8.1 Formulation

A nonlinear system has the form

$$
F(x) = 0,
\qquad
F : \mathbb{R}^n \to \mathbb{R}^n,
$$

or in components

$$
F_1(x_1,\ldots,x_n) = 0,
$$

$$
F_2(x_1,\ldots,x_n) = 0,
$$

$$
\vdots
$$

$$
F_n(x_1,\ldots,x_n) = 0.
$$

Unlike the linear case, the solution set may be:

- empty
- a finite set of isolated points
- a curve or surface of solutions
- multiple disconnected branches

There is no complete analogue of the rank classification above. Structure still matters, but now local differential information becomes central.

### 8.2 Newton's Method for Nonlinear Systems

Newton's method linearises the system around the current iterate $x_k$:

$$
F(x) \approx F(x_k) + J_F(x_k)(x-x_k),
$$

where $J_F(x_k)$ is the Jacobian.

Set the linear approximation to zero:

$$
F(x_k) + J_F(x_k)\Delta x_k \approx 0.
$$

So the Newton step solves the linear system

$$
J_F(x_k)\Delta x_k = -F(x_k),
$$

and updates

$$
x_{k+1} = x_k + \Delta x_k.
$$

This is one of the great unifying ideas of applied mathematics:

```text
nonlinear problem
   -> local linearisation
   -> solve linear system
   -> update
   -> repeat
```

Near a simple root, Newton convergence is quadratic, which is extremely fast. But it can fail badly if:

- the initial guess is poor
- the Jacobian is singular or nearly singular
- the function is not smooth enough

### 8.3 Quasi-Newton Methods

Newton's method is powerful but expensive because each step requires the full Jacobian and a system solve.

Quasi-Newton methods approximate the Jacobian (or the Hessian in optimization form) using update formulas instead of recomputing it exactly.

For root-finding, Broyden's method updates an approximate Jacobian with a rank-one correction chosen to satisfy the latest secant relation.

In optimization, the best-known analogues are BFGS and L-BFGS. They trade exact second-order information for lower cost while preserving much of Newton's geometric advantage.

The recurring theme is the same as elsewhere in AI systems engineering:

- exact structure is too expensive
- structured approximation is often the practical win

### 8.4 Fixed-Point Iteration

Sometimes we rewrite the system

$$
F(x) = 0
$$

as

$$
x = G(x)
$$

and iterate

$$
x_{k+1} = G(x_k).
$$

If $G$ is a contraction, Banach's fixed-point theorem guarantees convergence to a unique fixed point.

This sounds abstract, but the idea is everywhere:

- value iteration in reinforcement learning
- expectation-maximization style self-consistency updates
- implicit layers and deep equilibrium models
- iterative normalisation or message-passing updates

Fixed-point iteration is usually slower than Newton near a solution, but it may be simpler, cheaper, or more robust globally.

### 8.5 Continuation and Homotopy Methods

What if the target nonlinear system is too hard to solve directly?

Continuation methods embed it in a family:

$$
H(x,t) = 0,
\qquad t \in [0,1],
$$

such that:

- $H(x,0)=0$ is easy
- $H(x,1)=F(x)=0$ is the true problem

Then track the solution path as $t$ moves from $0$ to $1$.

This is especially powerful for:

- polynomial systems with many roots
- bifurcation analysis
- problems where direct Newton iteration fails from naive initialisation

The conceptual point is valuable even if one never implements homotopy directly: difficult problems can often be solved by deforming from an easier one.

### 8.6 The Gradient Equations as a Nonlinear System

Training a model means minimising $L(\theta)$. The stationary condition is

$$
\nabla_\theta L(\theta^\star) = 0.
$$

That is a nonlinear system in the parameters.

Gradient descent is therefore not separate from system solving. It is one particular iterative scheme for approximately solving the gradient equations.

Newton's method for optimization goes one step further:

$$
H(\theta_k)\Delta \theta_k = -\nabla L(\theta_k),
$$

where $H$ is the Hessian.

So each Newton step is a linear-system solve inside a nonlinear-system solve. This nesting is fundamental to modern optimization theory.

---

## 9. Constrained Systems and Lagrangians

### 9.1 Equality-Constrained Systems

Suppose we want to minimise $f(x)$ subject to constraints

$$
g(x) = 0,
\qquad g : \mathbb{R}^n \to \mathbb{R}^m.
$$

Introduce Lagrange multipliers $\lambda \in \mathbb{R}^m$ and define

$$
\mathcal{L}(x,\lambda) = f(x) + \lambda^\top g(x).
$$

The first-order conditions are

$$
\nabla_x \mathcal{L}(x,\lambda) = 0,
\qquad
g(x) = 0.
$$

This is itself a larger system of equations in the primal variables and the multipliers.

This is the central message:

```text
optimization with constraints
   ->
system of equations in (x, lambda)
```

### 9.2 Saddle-Point Systems

Linearised KKT systems often take the block form

$$
\begin{pmatrix}
A & B^\top \\
B & 0
\end{pmatrix}
\begin{pmatrix}
x \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
f \\
g
\end{pmatrix}.
$$

These are saddle-point systems:

- not positive definite
- often indefinite
- structured rather than arbitrary

They require specialised linear algebra.

A common strategy is block elimination using the Schur complement. If $A$ is invertible, eliminate $x$ first and solve a reduced system for $\lambda$.

From

$$
Ax + B^\top \lambda = f,
$$

we get

$$
x = A^{-1}(f - B^\top \lambda).
$$

Substituting into

$$
Bx = g
$$

gives

$$
BA^{-1}(f - B^\top \lambda) = g,
$$

so

$$
(BA^{-1}B^\top)\lambda = BA^{-1}f - g.
$$

The matrix

$$
S = BA^{-1}B^\top
$$

is the Schur complement. Once $\lambda$ is known, recover $x$ from the first block equation.

This block viewpoint is essential in large constrained problems because it turns one large structure into smaller pieces that can be interpreted and solved more intelligently.

### 9.3 KKT Systems in AI

Constraint systems appear in many AI settings:

- support vector machines through quadratic programming
- KL-constrained policy optimization and RLHF-style objectives
- simplex-constrained probability vectors
- structured prediction with equality or inequality constraints

Even when an ML paper does not say "system of equations," KKT conditions are often doing the hidden work.

For example, the softmax distribution can be derived from an entropy-regularised constrained optimization problem with the simplex constraint

$$
\sum_i p_i = 1.
$$

The multiplier associated with this normalisation is exactly the kind of object introduced by the Lagrangian formalism.

### 9.4 Inequality Constraints and Linear Programming

If the constraints are inequalities,

$$
Ax \le b,
\qquad
Cx = d,
$$

then feasibility itself becomes a constraint system.

Linear programming studies

$$
\min_x c^\top x \quad \text{subject to } Ax \le b, \ Cx=d.
$$

The geometry changes from affine subspaces to convex polytopes, but the system perspective remains:

- equations describe faces
- inequalities carve out feasible regions
- KKT conditions add complementary slackness

This viewpoint matters in AI for decoding constraints, sparse optimization, and relaxations of combinatorial prediction problems.

---

## 10. Systems Arising Directly in AI

### 10.1 Normal Equations in Linear Regression

Given data matrix $X \in \mathbb{R}^{n \times d}$ and target vector $y \in \mathbb{R}^n$, least squares solves

$$
\min_w \|Xw-y\|_2^2.
$$

The normal equations are

$$
X^\top X w = X^\top y.
$$

This is the canonical ML example of a system emerging from an optimization problem. Ridge regression modifies it to

$$
(X^\top X + \lambda I)w = X^\top y,
$$

which is both statistically and numerically better behaved when the feature matrix is close to rank deficient.

Linear probes on frozen embeddings are exactly this calculation.

### 10.2 Attention Normalisation as Constraint System

For one attention row, raw scores $s_j$ are transformed into weights

$$
\alpha_j = \frac{e^{s_j}}{\sum_k e^{s_k}}.
$$

The resulting vector $\alpha$ satisfies:

- $\alpha_j \ge 0$
- $\sum_j \alpha_j = 1$

So every attention row is a constrained weighting problem over the value vectors. Softmax is the smooth mechanism that enforces those constraints.

The final output is

$$
o = \sum_j \alpha_j v_j,
$$

which is a convex combination. System language clarifies what attention is doing: it solves for admissible weights under normalization constraints, then forms a structured linear combination.

### 10.3 Deep Equilibrium Models DEQ

A deep equilibrium model defines its hidden state implicitly through

$$
z^\star = f_\theta(z^\star, x).
$$

This is a nonlinear fixed-point system.

Forward pass:

- solve for $z^\star$ iteratively

Backward pass:

- use implicit differentiation, which again requires solving a linear system involving $(I - J_f^\top)$ or a related operator

More concretely, if

$$
z^\star = f_\theta(z^\star, x),
$$

then differentiating implicitly gives

$$
\left(I - \frac{\partial f_\theta}{\partial z}(z^\star, x)\right)\frac{\partial z^\star}{\partial \theta}
=
\frac{\partial f_\theta}{\partial \theta}(z^\star, x).
$$

So the backward pass avoids storing an explicit deep stack of activations, but only because it replaces that memory cost with a linear solve in the implicit Jacobian operator.

DEQs are a clean example of how system solving is not only an analysis tool but the actual definition of the model computation.

### 10.4 Gaussian Processes and Linear System Solves

In Gaussian process regression, the posterior mean involves

$$
(K + \sigma^2 I)^{-1} y,
$$

where $K$ is the kernel matrix.

Rather than explicitly invert the matrix, one solves

$$
(K + \sigma^2 I)\alpha = y.
$$

Because the matrix is SPD, Cholesky is the natural direct method. For larger problems, iterative methods such as CG with kernel-structure-aware approximations become important.

In practical Bayesian optimization and uncertainty-aware modelling, the system solve is often the dominant cost.

### 10.5 Newton's Method for Transformer Training

A full Newton step for a large model would solve

$$
H \Delta \theta = -g,
$$

with

- $g = \nabla L(\theta)$
- $H = \nabla^2 L(\theta)$

This is not feasible at LLM scale because:

- $H$ is far too large to store
- even matrix-vector access can be expensive

Modern second-order or quasi-second-order optimizers therefore approximate the system structure:

- diagonal approximations such as Adam-like methods
- Kronecker-factored approximations such as K-FAC
- matrix preconditioners such as Shampoo and related variants

The common idea is always the same: do not ignore the linear system, approximate it intelligently.

For example:

- K-FAC approximates layerwise curvature by Kronecker factors, turning one huge inverse into smaller matrix inverses or Cholesky solves
- Shampoo maintains matrix preconditioners for row and column structure, effectively solving a block-shaped system in a better metric
- recent methods such as SOAP stabilise these preconditioners further while keeping their cost tractable at scale

This is why second-order methods should be understood as linear-system design problems rather than as isolated optimizer tricks.

### 10.6 Physics-Informed Neural Networks PINNs

Physics-informed neural networks train a network $u_\theta$ so that it satisfies a differential equation system approximately.

Typical ingredients:

- PDE residual equations
- boundary conditions
- initial conditions

The loss is built from these residuals, so training attempts to solve the underlying equation system through function approximation.

PINNs matter here because they make the analogy literal: neural networks are being used as approximate solvers for systems of equations coming from physics.

---

## 11. Overdetermined and Underdetermined Systems in Practice

### 11.1 The Underdetermined Case and Implicit Bias

In modern machine learning, underdetermined systems are not pathological. They are normal.

If a model has many more parameters than effectively independent constraints, then the stationary equations admit many solutions. The question is no longer:

$$
\text{Is there a solution?}
$$

It becomes:

$$
\text{Which solution does the algorithm choose?}
$$

For linear models trained from small or zero initialisation, gradient descent often converges to the minimum-norm interpolating solution. That is an example of **implicit bias**: the algorithm selects one point from a large affine family without that preference being written explicitly in the objective.

This matters for generalisation. In overparameterized learning, the geometry of the solution set is not enough; the algorithm's path through that set matters too.

### 11.2 Minimum Norm Solutions and Regularisation

For an underdetermined but consistent system with full row rank,

$$
Ax = b,
\qquad
A \in \mathbb{R}^{m \times n},
\qquad
m < n,
$$

the minimum Euclidean norm solution is

$$
x^\star = A^\top (AA^\top)^{-1} b = A^+ b.
$$

This is the Moore-Penrose pseudo-inverse solution.

It is the special member of the solution family

$$
x = x_p + x_h
$$

that is orthogonal to the null space.

Other norms lead to other preferences:

- $L^1$ minimisation promotes sparsity
- nuclear norm minimisation promotes low rank
- explicit penalties reshape which member of a feasible family is preferred

The practical lesson is simple: once exact uniqueness is gone, the norm or regulariser you choose becomes part of the definition of the problem.

### 11.3 Structured Underdetermined Systems in Language Models

Language-model generation contains many structured underdetermined choices.

At each decoding step:

- many token distributions satisfy the simplex constraint
- many sequences satisfy a grammar or schema
- many continuation paths have similar local likelihood

The system is not linear, but the principle is similar: there are many admissible solutions, and the algorithm plus constraints determine which one is selected.

Examples:

- beam search chooses among multiple high-scoring structured candidates
- constrained decoding solves a combinatorial feasibility problem over token sequences
- watermarking schemes add extra statistical constraints to the output distribution

### 11.4 The System Perspective on Self-Attention

For one query position $i$, attention computes

$$
O_i = \sum_j \alpha_{ij} V_j,
$$

with

$$
\alpha_{ij} \ge 0,
\qquad
\sum_j \alpha_{ij} = 1.
$$

So the output lies in the convex hull of the value vectors. This is a constrained coefficient-selection problem:

- unconstrained linear combination would allow arbitrary signed weights
- attention allows only simplex-constrained weights

That restriction is part of why attention is stable and interpretable as "allocation of focus" rather than arbitrary recombination.

---

## 12. Numerical Methods and Stability

### 12.1 Floating-Point Arithmetic and Rounding Errors

Every real computation on a machine is finite-precision computation. Solving systems of equations therefore means solving them with rounding error.

Each arithmetic operation introduces a relative perturbation on the order of machine epsilon:

- FP64: about $10^{-16}$
- FP32: about $10^{-7}$
- BF16: about $10^{-3}$ in relative mantissa precision

The actual solution error depends on both:

- the stability of the algorithm
- the conditioning of the problem

This distinction is foundational:

- a stable algorithm can still produce a poor answer on an ill-conditioned problem
- an unstable algorithm can fail even on a well-conditioned problem

### 12.2 Iterative Refinement

Suppose we computed an approximate solution $\tilde{x}$. We can form the residual

$$
r = b - A \tilde{x}
$$

and solve a correction system

$$
Ad = r.
$$

Then update

$$
\tilde{x} \leftarrow \tilde{x} + d.
$$

This is iterative refinement.

Why it works:

- the factorisation of $A$ can be reused
- the residual often reveals the remaining error accurately
- low-precision factorisation plus higher-precision residual computation can recover surprisingly high accuracy

In mixed precision form, a typical pattern is:

1. factorise or approximately solve in low precision
2. compute the residual in higher precision
3. solve the correction equation using the stored factorisation
4. update the solution and repeat

This is mathematically classical and computationally modern at the same time.

Modern mixed-precision solvers rely heavily on this principle. The same mentality appears in mixed-precision training: cheap low-precision main computation, with strategically preserved higher-precision corrective structure.

### 12.3 Pivoting Strategies and Numerical Stability

Pivoting is not merely a convenience; it is a stability mechanism.

If a matrix is diagonally dominant or structurally well behaved, elimination may be stable even without pivoting. But in general, partial pivoting is the practical baseline because it controls growth in the elimination process.

The high-level rule is:

```text
algebraic solvability
!=
numerical reliability
```

This distinction explains why numerically aware implementations almost never use the classroom version of Gaussian elimination verbatim.

### 12.4 Preconditioning for Ill-Conditioned Systems

Ill-conditioning hurts twice:

- it amplifies perturbations in direct solving
- it slows iterative convergence

Preconditioning attacks the geometry. Instead of accepting the original coordinate system of the problem, it rescales or reshapes the system into one that behaves better numerically.

Left preconditioning:

$$
M^{-1} A x = M^{-1} b.
$$

Right preconditioning:

$$
A M y = b,
\qquad x = M y.
$$

Split preconditioning applies transforms on both sides.

The point is never to find a perfect inverse. It is to find a cheap approximation that dramatically improves the effective spectrum.

### 12.5 Condition Number and Practical Impact

Condition number translates geometry into operational risk.

If $\kappa(A)$ is large, then:

- small perturbations in data can produce large perturbations in the solution
- low-precision arithmetic becomes dangerous
- iterative methods need more steps
- regularization becomes more attractive

In AI this explains many engineering interventions:

- damping Hessian systems in second-order methods
- adding ridge terms to kernel matrices
- scaling attention scores by $\sqrt{d_k}$
- normalizing activations and gradients

All of these can be interpreted as attempts to avoid badly conditioned computational subproblems.

---

## 13. Systems of Equations in Optimisation

### 13.1 Gradient Descent as Approximate System Solving

Optimization can be read as equation solving:

$$
\nabla L(\theta) = 0.
$$

Gradient descent iterates

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t).
$$

A fixed point of this update satisfies

$$
\theta_{t+1} = \theta_t
\quad \Longleftrightarrow \quad
\nabla L(\theta_t) = 0.
$$

So gradient descent is one way of seeking a root of the gradient field.

For quadratic losses, this connection becomes explicit: gradient descent is an iterative solver for the normal equations.

### 13.2 Newton's Method and the Hessian System

Newton's method in optimization solves

$$
H(\theta)\Delta \theta = -\nabla L(\theta)
$$

and updates

$$
\theta \leftarrow \theta + \Delta \theta.
$$

Near a well-behaved optimum, this gives quadratic convergence. But for large neural networks it is limited by the same obstacle as every other second-order method: the Hessian system is enormous.

That is why exact Newton is rare at scale, while approximate curvature methods are common.

### 13.3 The Role of Linear System Solves in Second-Order Methods

Second-order methods differ mainly in how they approximate or factor the curvature system.

**K-FAC**

- approximates curvature blockwise with Kronecker structure
- converts one huge system into many smaller structured solves
- for a layer gradient $G$, the preconditioned update has the schematic form
  $$
  \Delta W \approx G_{\text{left}}^{-1} \, G \, G_{\text{right}}^{-1}
  $$
  where each side is estimated from activations or gradient covariances

**Shampoo**

- keeps matrix-valued preconditioners per parameter block
- uses matrix roots and inverse roots to precondition updates
- replaces scalar learning-rate rescaling with geometry-aware matrix rescaling

**SOAP and related recent preconditioners**

- push matrix preconditioning further while retaining practical scalability
- aim to capture more curvature structure than diagonal methods without paying full Hessian cost

These methods are not mysterious if you view them through the system lens. Each is a different compromise between:

- fidelity to the true Hessian system
- memory cost
- solve cost
- numerical stability

### 13.4 Convergence Analysis via Linear Systems

For the quadratic loss

$$
L(\theta) = \frac{1}{2}\|A\theta - b\|_2^2,
$$

we have

$$
\nabla L(\theta) = A^\top(A\theta - b),
$$

so the stationary condition is

$$
A^\top A \theta = A^\top b.
$$

Thus the local convergence of first-order methods is governed by the spectrum of $A^\top A$.

This is why eigenvalues, singular values, and condition numbers are not separate subjects from optimization. They govern the difficulty of the linear systems sitting underneath it.

---

## 14. Common Mistakes

| Mistake                                                           | Why it is wrong                                                                                                                       | Fix                                                                               |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| "If $m=n$, the system must have a unique solution."               | Square only means the matrix is square. It does not imply full rank. Singular square systems may have no solution or infinitely many. | Check rank or determinant; uniqueness requires full rank.                         |
| "Gaussian elimination is only for square systems."                | Row reduction works for any $m \times n$ matrix and is one of the best tools for understanding rectangular systems.                   | Use elimination on any augmented matrix to reveal rank and solution structure.    |
| "Homogeneous systems might have no solution."                     | $x=0$ always solves $Ax=0$.                                                                                                           | The real question is whether non-zero solutions exist.                            |
| "Least squares solves the original system exactly."               | Least squares solves the closest projection problem, not the exact system unless $b \in \mathrm{col}(A)$.                             | Always inspect the residual.                                                      |
| "Normal equations are always fine because the formula is simple." | Forming $A^\top A$ squares the condition number and can destroy accuracy.                                                             | Prefer QR or SVD for harder least-squares problems.                               |
| "A small residual means a highly accurate solution."              | For ill-conditioned systems, the residual can be small while the actual solution error is large.                                      | Consider conditioning as well as residual size.                                   |
| "Free variables can be ignored after finding one solution."       | Free variables encode the entire family of exact solutions.                                                                           | Write the full affine solution: particular plus null-space basis.                 |
| "Iterative methods always converge if you run them long enough."  | Convergence depends on matrix structure and the chosen method. Some iterations diverge.                                               | Check assumptions such as SPD, diagonal dominance, or spectral-radius conditions. |
| "The inverse is the natural way to solve every linear system."    | Explicit inversion is usually slower and less stable than factorisation-based solving.                                                | Solve $Ax=b$ directly using LU, QR, or Cholesky as appropriate.                   |
| "Optimization is separate from systems of equations."             | Stationarity, KKT conditions, Newton steps, and fixed points are all equation systems.                                                | Reframe optimization algorithms through their associated systems.                 |

---

## 15. Exercises

1. **Row reduction and solution types**: for each augmented matrix, reduce to RREF, classify the system, and write the full solution set:
   (a) $\begin{bmatrix} 1 & 2 & -1 & \vline & 4 \\ 2 & 4 & -2 & \vline & 8 \\ 3 & 6 & -3 & \vline & 12 \end{bmatrix}$
   (b) $\begin{bmatrix} 1 & 0 & 2 & \vline & 3 \\ 0 & 1 & -1 & \vline & 2 \\ 1 & 1 & 1 & \vline & 6 \end{bmatrix}$
   (c) $\begin{bmatrix} 1 & 2 & \vline & 3 \\ 2 & 4 & \vline & 7 \\ 0 & 1 & \vline & 1 \end{bmatrix}$
   (d) $\begin{bmatrix} 1 & 1 & 1 & \vline & 6 \\ 0 & 1 & 2 & \vline & 7 \\ 1 & 0 & -1 & \vline & 1 \end{bmatrix}$

2. **Particular solution and null space**: for

   $$
   A =
   \begin{pmatrix}
   1 & 2 & 0 & 1 \\
   0 & 0 & 1 & 1 \\
   1 & 2 & 1 & 2
   \end{pmatrix},
   $$

   (a) find the RREF, (b) identify pivot and free variables, (c) find a particular solution to $Ax = (1,1,2)^\top$, (d) find a basis for $\mathrm{null}(A)$, and (e) verify rank-nullity.

3. **Least squares**: with

   $$
   X =
   \begin{pmatrix}
   1 & 1 \\
   1 & 2 \\
   1 & 3 \\
   1 & 4
   \end{pmatrix},
   \qquad
   y =
   \begin{pmatrix}
   2 \\ 3 \\ 3.5 \\ 5
   \end{pmatrix},
   $$

   derive the normal equations, solve for $w^\star$, compute the residual, verify orthogonality to the column space, and then compare with ridge regularisation using $\lambda = 0.5$.

4. **Conditioning**: for

   $$
   A =
   \begin{pmatrix}
   1000 & 999 \\
   999 & 998
   \end{pmatrix},
   $$

   compute the determinant, solve $Ax=b$, perturb $b$ slightly, and quantify how much the solution changes. Interpret the result using the condition number.

5. **Newton's method for a nonlinear system**: solve

   $$
   F(x,y) = (x^2 + y^2 - 1,\ x - y^2) = 0
   $$

   by deriving the Jacobian and performing two Newton steps from $(1,1)$.

6. **Iterative methods**: for

   $$
   A =
   \begin{pmatrix}
   4 & 1 \\
   1 & 3
   \end{pmatrix},
   \qquad
   b =
   \begin{pmatrix}
   1 \\ 2
   \end{pmatrix},
   $$

   solve using Cholesky, then compare Jacobi, Gauss-Seidel, and two steps of CG.

7. **Constrained system**: minimise $f(x,y)=x^2 + 2y^2$ subject to $x+y=3$. Write the Lagrangian, derive the KKT system, solve it, and then repeat with the extra constraint $x-y=1$.

8. **AI application - attention as a constrained coefficient system**: with
   $$
   Q =
   \begin{pmatrix}
   1 & 0 \\
   0 & 1 \\
   1 & 1
   \end{pmatrix},
   \quad
   K =
   \begin{pmatrix}
   1 & 1 \\
   1 & 0 \\
   0 & 1
   \end{pmatrix},
   \quad
   V =
   \begin{pmatrix}
   1 & 0 \\
   0 & 1 \\
   1 & 1
   \end{pmatrix},
   $$
   compute the score matrix, the softmax weights, and the output matrix. Interpret each attention row as a simplex-constrained weighting problem.

---

## 16. Why This Matters for AI (2026 Perspective)

| Aspect                                    | Impact                                                                                                                                         |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear regression and probing             | Fitting linear probes, calibration heads, and readout layers reduces to exact or regularised linear systems.                                   |
| Newton and second-order optimisation      | K-FAC, Shampoo, SOAP, and related optimizers are structured approximations to very large curvature-system solves.                              |
| Overparameterisation                      | Modern deep networks often operate in underdetermined regimes, so the geometry of the solution set and the implicit bias of the solver matter. |
| Deep equilibrium models                   | Forward pass solves a nonlinear fixed-point system; backward pass solves an implicit linear system.                                            |
| Gaussian processes and Bayesian inference | Kernel methods are fundamentally system-solving pipelines, usually with SPD matrices.                                                          |
| Constrained decoding and RLHF             | Normalization, KL constraints, and structured generation can all be expressed as coupled constraint systems.                                   |
| Numerical stability                       | Condition numbers explain why damping, normalization, scaling, and mixed precision are operational necessities rather than cosmetic choices.   |
| PINNs and scientific AI                   | Solving PDE systems with neural surrogates makes equation-solving central to modern AI-for-science workflows.                                  |
| Sparse and efficient attention            | Approximate attention mechanisms preserve or relax the original attention system structure to gain scale.                                      |
| Interpretability                          | Many attribution and decomposition techniques boil down to solving linear systems for component contributions.                                 |

---

## 17. Conceptual Bridge

Systems of equations are where abstract algebra, geometry, and computation become inseparable.

- vector spaces told us what the objects are
- matrix operations told us how those objects are transformed
- systems of equations tell us how to recover unknowns from constraints

The linear case gives the complete reference model:

- rank tells us how many independent constraints exist
- null spaces tell us what freedom remains
- elimination and factorisation tell us how to solve
- least squares tells us what to do when exact compatibility fails

The nonlinear case adds the genuinely hard phenomena:

- multiple solutions
- bifurcations
- local linearisation
- fixed points
- constrained stationary conditions

This is exactly the bridge needed for the next major topic. Once a system has been written as $Ax=b$, the next question is often:

$$
\text{What does the structure of } A \text{ tell us about the solve?}
$$

That is the spectral question, and it leads naturally to eigenvalues, eigenvectors, and decompositions.

```text
Vectors and Spaces      -> geometry
Matrix Operations       -> computation
Systems of Equations    -> solving
Eigenvalues / SVD       -> structure
Probability             -> uncertainty
Optimisation            -> dynamics
LLM Mathematics         -> all of them together
```

---

## References

- Gilbert Strang, _MIT 18.06 Linear Algebra_ course materials and notes: [https://web.mit.edu/18.06/www/](https://web.mit.edu/18.06/www/)
- Richard Barrett et al., _Templates for the Solution of Linear Systems_: [https://www.netlib.org/templates/templates.html](https://www.netlib.org/templates/templates.html)
- Lloyd N. Trefethen and David Bau III, _Numerical Linear Algebra_, SIAM, 1997.
- Gene H. Golub and Charles F. Van Loan, _Matrix Computations_, 4th ed., Johns Hopkins University Press, 2013.
- Jonathan Richard Shewchuk, _An Introduction to the Conjugate Gradient Method Without the Agonizing Pain_: [https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
- Y. Saad and M. H. Schultz, "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems": [https://epubs.siam.org/doi/10.1137/0907058](https://epubs.siam.org/doi/10.1137/0907058)
- Ashish Vaswani et al., "Attention Is All You Need" (2017): [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Shaojie Bai, J. Zico Kolter, and Vladlen Koltun, "Deep Equilibrium Models" (2019): [https://arxiv.org/abs/1909.01377](https://arxiv.org/abs/1909.01377)
- James Martens and Roger Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (ICML 2015): [https://proceedings.mlr.press/v37/martens15.html](https://proceedings.mlr.press/v37/martens15.html)
- Vineet Gupta et al., "Shampoo: Preconditioned Stochastic Tensor Optimization" (ICML 2018): [https://proceedings.mlr.press/v80/gupta18a.html](https://proceedings.mlr.press/v80/gupta18a.html)
- Nikhil Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam" (2024): [https://arxiv.org/abs/2409.11321](https://arxiv.org/abs/2409.11321)
- A. S. Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023): [https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)
- Maziar Raissi, Paris Perdikaris, and George E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (2019): [https://www.sciencedirect.com/science/article/pii/S0021999118307125](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- NVIDIA cuSOLVER documentation: [https://docs.nvidia.com/cuda/cusolver/](https://docs.nvidia.com/cuda/cusolver/)
- NVIDIA cuSPARSE documentation: [https://docs.nvidia.com/cuda/cusparse/](https://docs.nvidia.com/cuda/cusparse/)
