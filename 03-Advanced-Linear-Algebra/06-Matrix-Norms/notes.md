[← Back to Advanced Linear Algebra](../README.md) | [← Orthogonality](../05-Orthogonality-and-Orthonormality/notes.md) | [Next: Positive Definite Matrices →](../07-Positive-Definite-Matrices/notes.md)

---

# Matrix Norms and Condition Numbers

> *"A norm is a lens that reveals the geometry hidden inside a matrix."*

## Overview

Matrix norms answer a deceptively simple question: **how big is this matrix?** The answer depends on what we mean by "big" — the largest entry, the total energy, the most a vector can be stretched — and each choice of measurement gives a different norm, each useful in a different context.

This section builds a complete theory of matrix norms from first principles. We develop the three central families — Frobenius norms (entry-wise energy), induced operator norms (worst-case amplification), and Schatten norms (singular-value aggregates) — and the unifying concept of unitarily invariant norms. The condition number, which measures how much a matrix amplifies errors in linear systems, emerges as the ratio of the largest to smallest singular value.

For machine learning practitioners, matrix norms are not abstract tools. Spectral normalization (the key to stable GAN training) clips the spectral norm to 1 at each layer. Weight decay penalizes the Frobenius norm. Nuclear norm regularization promotes low-rank structure. Gradient clipping bounds the global L2 norm. Lipschitz constants of deep networks are products of per-layer spectral norms. Understanding these connections requires the precise definitions and inequalities developed here.

## Prerequisites

- Vector norms: L1, L2, L∞, Lp — axioms and geometry
- SVD: $A = U\Sigma V^\top$, singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$ — see [§02: SVD](../02-Singular-Value-Decomposition/notes.md)
- Eigenvalues and spectral theory — see [§01: Eigenvalues](../01-Eigenvalues-and-Eigenvectors/notes.md)
- Orthogonal matrices, inner products — see [§05: Orthogonality](../05-Orthogonality-and-Orthonormality/notes.md)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Computational verification of all norm definitions, inequalities, condition number, and ML applications |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from norm axiom verification to spectral normalization |

## Learning Objectives

After completing this section, you will be able to:

- State the three norm axioms and verify them for any proposed norm
- Compute Frobenius, spectral (operator 2-norm), nuclear, matrix 1-norm, and matrix ∞-norm
- Derive the formula $\|A\|_2 = \sigma_1(A)$ and $\|A\|_* = \sum_i \sigma_i$ from the SVD
- Prove that all induced norms are submultiplicative and explain the consequence for deep networks
- Define unitarily invariant norms and classify Schatten and Ky Fan norms as examples
- Compute and interpret the condition number $\kappa(A) = \sigma_1/\sigma_n$ and apply the perturbation bound
- Explain why Tikhonov regularization improves conditioning and quantify the improvement
- State Weyl's inequality for singular value perturbations and apply it to error analysis
- Connect spectral normalization, nuclear norm regularization, and Frobenius weight decay to their mathematical foundations

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why We Need to Measure Matrices](#11-why-we-need-to-measure-matrices)
  - [1.2 Geometric Picture: Amplification and Shrinkage](#12-geometric-picture-amplification-and-shrinkage)
  - [1.3 Why Matrix Norms Matter for AI](#13-why-matrix-norms-matter-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Norm Axioms](#2-norm-axioms)
  - [2.1 Abstract Norm Definition](#21-abstract-norm-definition)
  - [2.2 Standard Vector Norms](#22-standard-vector-norms)
  - [2.3 Equivalence of Norms in Finite Dimensions](#23-equivalence-of-norms-in-finite-dimensions)
  - [2.4 Dual Norms](#24-dual-norms)
- [3. The Frobenius Norm](#3-the-frobenius-norm)
  - [3.1 Definition and Basic Properties](#31-definition-and-basic-properties)
  - [3.2 Frobenius as an Inner Product Space](#32-frobenius-as-an-inner-product-space)
  - [3.3 Frobenius and Singular Values](#33-frobenius-and-singular-values)
  - [3.4 Submultiplicativity and Mixed Norm Inequalities](#34-submultiplicativity-and-mixed-norm-inequalities)
  - [3.5 Frobenius in ML: Weight Decay](#35-frobenius-in-ml-weight-decay)
- [4. Induced (Operator) Norms](#4-induced-operator-norms)
  - [4.1 Definition and Geometric Meaning](#41-definition-and-geometric-meaning)
  - [4.2 The Spectral Norm](#42-the-spectral-norm)
  - [4.3 The Matrix 1-Norm and ∞-Norm](#43-the-matrix-1-norm-and--norm)
  - [4.4 Submultiplicativity](#44-submultiplicativity)
  - [4.5 Consistency Between Matrix and Vector Norms](#45-consistency-between-matrix-and-vector-norms)
- [5. Schatten Norms and the Nuclear Norm](#5-schatten-norms-and-the-nuclear-norm)
  - [5.1 Schatten p-Norms](#51-schatten-p-norms)
  - [5.2 The Nuclear Norm](#52-the-nuclear-norm)
  - [5.3 Nuclear/Spectral Duality](#53-nuclearspectral-duality)
  - [5.4 Nuclear Norm as Convex Relaxation of Rank](#54-nuclear-norm-as-convex-relaxation-of-rank)
- [6. Unitarily Invariant Norms](#6-unitarily-invariant-norms)
  - [6.1 Definition and Examples](#61-definition-and-examples)
  - [6.2 Ky Fan Norms](#62-ky-fan-norms)
  - [6.3 Von Neumann's Characterization](#63-von-neumanns-characterization)
  - [6.4 Fan Dominance and Majorization](#64-fan-dominance-and-majorization)
- [7. Condition Number](#7-condition-number)
  - [7.1 Definition and Interpretations](#71-definition-and-interpretations)
  - [7.2 Conditioning of Linear Systems](#72-conditioning-of-linear-systems)
  - [7.3 Condition Number and Convergence Rates](#73-condition-number-and-convergence-rates)
  - [7.4 Ill-Conditioning: Diagnosis and Examples](#74-ill-conditioning-diagnosis-and-examples)
  - [7.5 Regularization as Condition Improvement](#75-regularization-as-condition-improvement)
- [8. Perturbation Theory and Error Bounds](#8-perturbation-theory-and-error-bounds)
  - [8.1 Forward and Backward Error](#81-forward-and-backward-error)
  - [8.2 Weyl's Inequality](#82-weyls-inequality)
  - [8.3 Bauer-Fike Theorem](#83-bauer-fike-theorem)
  - [8.4 Practical Implications](#84-practical-implications)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Spectral Normalization](#91-spectral-normalization)
  - [9.2 Nuclear Norm Regularization and Low-Rank Models](#92-nuclear-norm-regularization-and-low-rank-models)
  - [9.3 Gradient Clipping via Norm Bounds](#93-gradient-clipping-via-norm-bounds)
  - [9.4 Frobenius and Weight Decay](#94-frobenius-and-weight-decay)
  - [9.5 Lipschitz Constants and Robustness](#95-lipschitz-constants-and-robustness)
  - [9.6 Norm-Based Generalization Bounds](#96-norm-based-generalization-bounds)
  - [9.7 Mixed Norms in Modern Architectures](#97-mixed-norms-in-modern-architectures)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---
## 1. Intuition

### 1.1 Why We Need to Measure Matrices

Before introducing any formula, consider five concrete scenarios where "how big is this matrix?" is not a vague question but a precise computational need:

**Scenario 1 — Bounding errors in linear systems.** You solve $A\mathbf{x} = \mathbf{b}$ on a computer. The computed $\hat{\mathbf{x}}$ satisfies $(A+E)\hat{\mathbf{x}} = \mathbf{b}$ for some rounding error matrix $E$. How wrong is $\hat{\mathbf{x}}$? The answer involves $\|E\|$ and $\|A^{-1}\|$ — you need norms to bound the damage.

**Scenario 2 — Regularization.** Your neural network overfits. You add a penalty $\lambda\|W\|^2$ to the loss. Which norm $\|\cdot\|$? The Frobenius norm penalizes every weight equally. The spectral norm penalizes the largest singular value. The nuclear norm encourages low rank. Each choice embeds different inductive bias.

**Scenario 3 — Lipschitz constants.** Your network $f$ maps inputs to outputs. How much can a small input perturbation change the output? The Lipschitz constant of a linear layer is exactly $\|W\|_2$. The global Lipschitz constant of a deep network is bounded by $\prod_l \|W^{[l]}\|_2$.

**Scenario 4 — Convergence of gradient descent.** Gradient descent on $\min_\mathbf{x} \frac{1}{2}\mathbf{x}^\top A\mathbf{x} - \mathbf{b}^\top\mathbf{x}$ converges geometrically with rate $1 - 2/(\kappa+1)$ where $\kappa = \|A\|_2\|A^{-1}\|_2$ is the condition number. A large condition number means slow convergence.

**Scenario 5 — Low-rank approximation quality.** The Eckart-Young theorem says the best rank-$k$ approximation to $A$ has error $\|A - A_k\|_2 = \sigma_{k+1}$ in spectral norm and $\|A - A_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$ in Frobenius norm. Without norms, we cannot even state what "best approximation" means.

In every scenario, the choice of norm carries meaning. This section develops the machinery to make those choices deliberate and principled.

### 1.2 Geometric Picture: Amplification and Shrinkage

A matrix $A \in \mathbb{R}^{m \times n}$ is a linear map. Applied to the unit sphere $\{\mathbf{x} : \|\mathbf{x}\|_2 = 1\}$ in $\mathbb{R}^n$, it produces an **ellipsoid** in $\mathbb{R}^m$. This follows directly from the SVD:

$$A = U\Sigma V^\top \quad \Rightarrow \quad A\mathbf{x} = U(\Sigma(V^\top\mathbf{x}))$$

$V^\top$ rotates the sphere (orthogonal maps preserve spheres). $\Sigma$ stretches along coordinate axes by factors $\sigma_1, \ldots, \sigma_r$ (with $\sigma_i > 0$). $U$ rotates the result. The outcome: the unit sphere maps to an ellipsoid whose semi-axes have lengths $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

```
MATRIX NORMS AS ELLIPSOID GEOMETRY
════════════════════════════════════════════════════════════════════════

  Unit sphere in ℝⁿ         →  A maps to →  Ellipsoid in ℝᵐ

  ·····                                      ╭───────────╮
 · · · ·          A = UΣVᵀ                ╭─╯           ╰─╮
· · · · ·    ───────────────▶            ╭─╯               ╰─╮
 · · · ·                                ╰──╮               ╭──╯
  ·····                                    ╰─╮           ╭─╯  ← σ₂
                                              ╰───────────╯
                               ←────────────────────────────────────→
                                              2σ₁  (longest axis)

  Spectral norm  ‖A‖₂  = σ₁  = length of longest semi-axis
  Frobenius norm ‖A‖_F = √(σ₁²+σ₂²+…) = "total energy" of axes
  Nuclear norm   ‖A‖_*  = σ₁+σ₂+… = sum of all semi-axis lengths

════════════════════════════════════════════════════════════════════════
```

**Each norm captures a different aspect of this ellipsoid:**
- $\|A\|_2 = \sigma_1$: the worst-case amplification — how long is the longest axis?
- $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$: root-mean-square amplification across all directions
- $\|A\|_* = \sum_i \sigma_i$: total "size" of all directions (related to rank)

**For AI:** Weight matrices in neural networks define such ellipsoids. A spectrally normalized layer has $\sigma_1 = 1$ — the unit sphere maps to an ellipsoid whose longest axis has length exactly 1. A Frobenius-regularized layer has bounded total energy across all singular values.

### 1.3 Why Matrix Norms Matter for AI

The table below shows each major AI technique and the norm that is its mathematical core:

| AI Technique | Norm Used | Mathematical Role |
|-------------|-----------|-------------------|
| **Weight decay / L2 reg** | Frobenius $\|W\|_F^2$ | Penalizes total squared weight energy |
| **Spectral normalization** (GANs) | Spectral $\|W\|_2 = \sigma_1$ | Enforces 1-Lipschitz discriminator |
| **Nuclear norm reg** | Nuclear $\|W\|_* = \sum\sigma_i$ | Promotes low-rank weight matrices |
| **Gradient clipping** | Frobenius or L2 of gradient | Prevents exploding gradients |
| **LoRA / low-rank adaptation** | Implicit nuclear norm | Low-rank $\Delta W = BA$ has bounded $\|.\|_*$ |
| **Certified robustness** | Per-layer spectral $\prod\|W^{[l]}\|_2$ | Bounds Lipschitz constant of network |
| **Condition number in Adam** | $\kappa = \sigma_1/\sigma_n$ of curvature | Diagonal preconditioning approximates $H^{-1}$ |
| **Attention stability** | $\|QK^\top\|_2 / \sqrt{d_k}$ | Prevents softmax saturation |
| **Generalization bounds** | $\|W\|_F$ products | PAC-Bayes and Rademacher complexity |

### 1.4 Historical Timeline

| Year | Contributor | Contribution |
|------|-------------|-------------|
| 1878 | Frobenius | Introduced the entry-sum norm (now Frobenius) in matrix theory |
| 1911 | Hilbert | Trace-class operators in infinite-dimensional spaces |
| 1937 | von Neumann | Schatten/trace norms; characterization of unitarily invariant norms |
| 1951 | Ky Fan | Ky Fan k-norms; Fan dominance theorem |
| 1960s | Golub, Kahan | Numerical algorithms for condition numbers; QR iteration |
| 1976 | Rudin | Functional analysis formalization; duality of norms |
| 1987 | Candès, Recht | Nuclear norm relaxation of rank minimization |
| 1992 | Trefethen & Bau | Numerical linear algebra framing of condition number for ML |
| 2009 | Candès & Recht | Matrix completion via nuclear norm minimization (Netflix Prize theory) |
| 2018 | Miyato et al. | Spectral normalization for GANs — spectral norm goes mainstream in DL |
| 2021–2024 | LoRA, DoRA, MuP | Mixed-norm methods for efficient fine-tuning of LLMs |

---

## 2. Norm Axioms

### 2.1 Abstract Norm Definition

**Definition.** A **norm** on a vector space $V$ over $\mathbb{R}$ is a function $\|\cdot\|: V \to \mathbb{R}_{\geq 0}$ satisfying three axioms for all $\mathbf{u}, \mathbf{v} \in V$ and $\alpha \in \mathbb{R}$:

| Axiom | Formula | Name |
|-------|---------|------|
| N1 | $\|\mathbf{v}\| \geq 0$ and $\|\mathbf{v}\| = 0 \Leftrightarrow \mathbf{v} = \mathbf{0}$ | Positive definiteness |
| N2 | $\|\alpha\mathbf{v}\| = |\alpha|\|\mathbf{v}\|$ | Absolute homogeneity |
| N3 | $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ | Triangle inequality |

A function satisfying N1 (without the zero condition), N2, and N3 is a **semi-norm**. A semi-norm can be zero on nonzero vectors.

**Matrix space as a vector space.** The space $\mathbb{R}^{m \times n}$ of $m \times n$ real matrices is a vector space of dimension $mn$ under entry-wise addition and scalar multiplication. Any norm on $\mathbb{R}^{mn}$ (after reshaping) gives a valid matrix norm. But not every matrix norm is "compatible" with matrix multiplication — additional properties (submultiplicativity) are often required.

**Definition (submultiplicative norm).** A matrix norm $\|\cdot\|$ on $\mathbb{R}^{n \times n}$ is **submultiplicative** (or **consistent**) if:
$$\|AB\| \leq \|A\|\|B\| \quad \text{for all } A, B \in \mathbb{R}^{n \times n}$$

Submultiplicativity is essential for analyzing products of matrices (as arise in deep network Jacobians) and for bounding errors that propagate through multiplications.

**Non-example of submultiplicativity.** The max-entry norm $\|A\|_{\max} = \max_{ij}|A_{ij}|$ satisfies the three norm axioms but is NOT submultiplicative in general. For $A = B = \mathbf{1}\mathbf{1}^\top/n$ (all-ones matrix divided by $n$): $\|A\|_{\max} = 1/n$, $\|AB\|_{\max} = 1$, so $\|AB\|_{\max} = 1 > 1/n^2 = \|A\|_{\max}\|B\|_{\max}$.

### 2.2 Standard Vector Norms

For $\mathbf{x} \in \mathbb{R}^n$, the standard norms are:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| \qquad \|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} \qquad \|\mathbf{x}\|_\infty = \max_i |x_i|$$

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p} \quad (p \geq 1)$$

The unit balls $\{\mathbf{x} : \|\mathbf{x}\|_p \leq 1\}$ have characteristic shapes:

```
UNIT BALLS IN 2D FOR DIFFERENT p-NORMS
════════════════════════════════════════════════════════════════════════

  p=1 (diamond)     p=2 (circle)     p=∞ (square)     p=1/2 (star)
                                                        [not a norm!]
      *                 ___               ┌───┐
     /|\               /   \             │   │               *
    / | \             /     \            │   │              /|\
  *──┼──*            │   ·   │           │   │            */   \*
    \ | /             \     /            │   │              \   /
     \|/               \___/             └───┘               \|/
      *                                                        *

  p < 1: concave, violates triangle inequality → NOT a norm
  p ≥ 1: convex, satisfies all axioms → valid norm

════════════════════════════════════════════════════════════════════════
```

**Hölder's inequality.** For conjugate exponents $1/p + 1/q = 1$ (with $p = 1 \Leftrightarrow q = \infty$):
$$|\langle\mathbf{x},\mathbf{y}\rangle| = \left|\sum_i x_i y_i\right| \leq \|\mathbf{x}\|_p\|\mathbf{y}\|_q$$

This generalizes the Cauchy-Schwarz inequality ($p = q = 2$) and is the foundation for dual norm theory.

### 2.3 Equivalence of Norms in Finite Dimensions

**Theorem.** On $\mathbb{R}^n$ (or any finite-dimensional normed space), all norms are **equivalent**: for any two norms $\|\cdot\|_\alpha$ and $\|\cdot\|_\beta$, there exist constants $c, C > 0$ such that:
$$c\|\mathbf{x}\|_\alpha \leq \|\mathbf{x}\|_\beta \leq C\|\mathbf{x}\|_\alpha \quad \text{for all } \mathbf{x} \in \mathbb{R}^n$$

**Proof sketch.** The unit sphere $\{\mathbf{x} : \|\mathbf{x}\|_\alpha = 1\}$ is compact. A norm is continuous. A continuous function on a compact set attains its extrema. These extrema give the constants $c$ and $C$. $\square$

**Specific bounds** relating standard norms on $\mathbb{R}^n$:

| Inequality | Bound | Tight when |
|-----------|-------|-----------|
| $\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$ | factor 1 | $\mathbf{x}$ has one nonzero entry |
| $\|\mathbf{x}\|_1 \leq \sqrt{n}\|\mathbf{x}\|_2$ | factor $\sqrt{n}$ | all entries equal |
| $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n}\|\mathbf{x}\|_\infty$ | factor $\sqrt{n}$ | one / all entries equal |
| $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_1 \leq n\|\mathbf{x}\|_\infty$ | factor $n$ | one / all entries equal |

**Why this matters:** Norm equivalence guarantees that convergence in one norm implies convergence in all others — so the choice of norm for convergence analysis is flexible. However, the **constants** matter for quantitative bounds, and choosing the right norm can tighten bounds by factors of $\sqrt{n}$ or $n$.

**Warning:** Norm equivalence fails in infinite dimensions. In $L^p$ function spaces, $L^2 \not\simeq L^\infty$ — this distinction is crucial in functional analysis and infinite-width neural network theory.

### 2.4 Dual Norms

**Definition.** The **dual norm** of $\|\cdot\|$ on $\mathbb{R}^n$ is:
$$\|\mathbf{y}\|_* = \max_{\|\mathbf{x}\| \leq 1} \langle\mathbf{x}, \mathbf{y}\rangle = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\langle\mathbf{x},\mathbf{y}\rangle}{\|\mathbf{x}\|}$$

**Key dual pairs:**
- $\|\cdot\|_1$ and $\|\cdot\|_\infty$ are dual: $\|\mathbf{y}\|_\infty = \max_{\|\mathbf{x}\|_1 \leq 1}\langle\mathbf{x},\mathbf{y}\rangle$
- $\|\cdot\|_2$ is self-dual: $\|\mathbf{y}\|_2 = \max_{\|\mathbf{x}\|_2 \leq 1}\langle\mathbf{x},\mathbf{y}\rangle$ (Cauchy-Schwarz with equality)
- $\|\cdot\|_p$ and $\|\cdot\|_q$ are dual when $1/p + 1/q = 1$ (Hölder conjugates)

**Dual norms for matrices** (treating them as vectors): the dual of the spectral norm is the nuclear norm:
$$\|A\|_* = \max_{\|B\|_2 \leq 1}\langle A, B\rangle_F = \max_{\|B\|_2 \leq 1}\operatorname{tr}(A^\top B)$$

This duality appears in optimization: the subdifferential of $\|A\|_*$ involves matrices with spectral norm $\leq 1$.

**For AI:** The dual norm appears naturally in proximal gradient methods. The proximal operator of $\lambda\|A\|_*$ (nuclear norm) is soft-thresholding of singular values — a key step in matrix completion algorithms and low-rank optimization.

---
## 3. The Frobenius Norm

### 3.1 Definition and Geometric Meaning

The **Frobenius norm** treats a matrix as a long vector and applies the Euclidean norm:

$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} = \sqrt{\operatorname{tr}(A^\top A)}$$

For $A \in \mathbb{R}^{m \times n}$, it is the $\ell^2$ norm on $\mathbb{R}^{mn}$. This gives the Frobenius norm full access to Euclidean geometry: inner products, projections, and angles between matrices.

**The matrix inner product.** The Frobenius norm is induced by the inner product:
$$\langle A, B \rangle_F = \operatorname{tr}(A^\top B) = \sum_{i,j} a_{ij} b_{ij}$$

This turns the space $\mathbb{R}^{m \times n}$ into a Hilbert space. Cauchy-Schwarz holds: $|\langle A, B\rangle_F| \leq \|A\|_F \|B\|_F$.

**Geometric interpretation.** $\|A\|_F$ is the "size" of the linear map $A$ when averaged uniformly over the unit sphere. More precisely, if $\mathbf{x}$ is drawn uniformly from the unit sphere $S^{n-1}$, then:
$$\mathbb{E}[\|A\mathbf{x}\|^2] = \frac{\|A\|_F^2}{n}$$

This is the mean squared output energy — the Frobenius norm measures average amplification over all directions.

### 3.2 SVD Relationship

The deepest formula for the Frobenius norm comes from the singular value decomposition:

$$\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2} = \|\boldsymbol{\sigma}\|_2$$

where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$ are the nonzero singular values and $r = \operatorname{rank}(A)$.

**Proof.** $\|A\|_F^2 = \operatorname{tr}(A^\top A)$. Since $A = U\Sigma V^\top$, we have $A^\top A = V\Sigma^2 V^\top$. Trace is invariant under similarity, so $\operatorname{tr}(A^\top A) = \operatorname{tr}(\Sigma^2) = \sum_i \sigma_i^2$.

**Consequences:**
- $\|A\|_F^2$ equals the sum of squared singular values — a measure of total "energy"
- $\|A\|_F \geq \|A\|_2 = \sigma_1$ (Frobenius norm is at least as large as spectral norm)
- $\|A\|_F \leq \sqrt{r}\|A\|_2$ (Frobenius norm bounded by $\sqrt{\text{rank}}$ times spectral norm)
- For rank-1 matrices: $\|A\|_F = \|A\|_2 = \sigma_1$ (single singular value)

**For AI:** The Frobenius norm of a weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ equals $\|\boldsymbol{\sigma}(W)\|_2$ — the $\ell^2$ norm of singular values. Weight decay (L2 regularization) penalizes $\|W\|_F^2 = \sum_i \sigma_i^2$, encouraging all singular values to shrink uniformly. This is weaker than nuclear norm regularization ($\|\boldsymbol{\sigma}\|_1$), which promotes low-rank structure.

### 3.3 Key Properties and Inequalities

**Submultiplicativity:**
$$\|AB\|_F \leq \|A\|_F \|B\|_F$$

**Proof.** $(AB)_{ij} = \mathbf{a}_i^\top \mathbf{b}_j$ where $\mathbf{a}_i$ is the $i$-th row of $A$ and $\mathbf{b}_j$ is the $j$-th column of $B$. By Cauchy-Schwarz: $|(AB)_{ij}|^2 \leq \|\mathbf{a}_i\|^2 \|\mathbf{b}_j\|^2$. Summing over $i,j$: $\|AB\|_F^2 \leq (\sum_i \|\mathbf{a}_i\|^2)(\sum_j \|\mathbf{b}_j\|^2) = \|A\|_F^2 \|B\|_F^2$.

**Unitary invariance:**
$$\|UAV\|_F = \|A\|_F \quad \text{for unitary } U, V$$

This follows from the SVD formula: unitary transformations permute singular values but don't change them.

**Comparison with other norms:**

| Norm | Symbol | Relationship to $\|A\|_F$ |
|------|--------|--------------------------|
| Spectral | $\|A\|_2$ | $\|A\|_2 \leq \|A\|_F \leq \sqrt{r}\|A\|_2$ |
| Nuclear | $\|A\|_*$ | $\|A\|_* \leq \sqrt{r}\|A\|_F$ |
| Max entry | $\|A\|_{\max}$ | $\|A\|_{\max} \leq \|A\|_F \leq \sqrt{mn}\|A\|_{\max}$ |
| Row-sum | $\|A\|_{1,\infty}$ | $\|A\|_{1,\infty} \leq \|A\|_F \leq \sqrt{m}\|A\|_{1,\infty}$ |

### 3.4 Best Low-Rank Approximation

The Eckart-Young theorem — the fundamental result about low-rank approximation — is stated in terms of both the Frobenius and spectral norms:

**Theorem (Eckart-Young, 1936).** Let $A = U\Sigma V^\top$ and $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$. Then:
$$\min_{\operatorname{rank}(B) \leq k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$$
$$\min_{\operatorname{rank}(B) \leq k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}$$

In both norms, the optimal rank-$k$ approximation is obtained by truncating the SVD.

**For AI:** This theorem is the mathematical foundation of PCA (→ [§03-PCA](../03-PCA/notes.md)), LoRA, and attention matrix compression. When training a LoRA adapter $W = W_0 + BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, the Eckart-Young theorem guarantees this is the optimal rank-$r$ perturbation in the Frobenius norm.

### 3.5 Computing and Differentiating the Frobenius Norm

**Computation.** In NumPy: `np.linalg.norm(A, 'fro')` or equivalently `np.sqrt(np.sum(A**2))`.

**Gradient.** The Frobenius norm is differentiable everywhere. Its gradient with respect to $A$ is:
$$\frac{\partial \|A\|_F^2}{\partial A} = 2A, \qquad \frac{\partial \|A\|_F}{\partial A} = \frac{A}{\|A\|_F}$$

This makes the Frobenius norm easy to differentiate in backpropagation. Weight decay adds $\lambda \|W\|_F^2$ to the loss, contributing gradient $2\lambda W$ — the standard L2 regularization term.

**For AI:** Spectral normalization (Miyato 2018) normalizes weight matrices by their spectral norm, not Frobenius. The gradient of the spectral norm is:
$$\frac{\partial \|A\|_2}{\partial A} = \mathbf{u}_1 \mathbf{v}_1^\top$$
the rank-1 outer product of the top left and right singular vectors. This is non-smooth when $\sigma_1$ has multiplicity $> 1$.

---

## 4. Induced (Operator) Norms

### 4.1 The General Definition

A matrix norm is **induced** (or **operator**) if it measures the worst-case amplification of a vector norm:

$$\|A\|_{p \to q} = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_q}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_q$$

The $p \to q$ notation specifies the norms on input ($p$) and output ($q$) spaces.

**Why induced norms matter.** The definition captures the most important property of a linear map: its maximum stretch factor. For stability analysis, control theory, neural network Lipschitz bounds, and generalization theory, the induced norm is the right tool.

**Submultiplicativity is automatic.** Any induced norm satisfies $\|AB\|_{p \to q} \leq \|A\|_{r \to q}\|B\|_{p \to r}$ because:
$$\|AB\mathbf{x}\|_q \leq \|A\|_{r\to q}\|B\mathbf{x}\|_r \leq \|A\|_{r\to q}\|B\|_{p\to r}\|\mathbf{x}\|_p$$

### 4.2 The Spectral Norm ($p = q = 2$)

The most important induced norm is the **spectral norm** (also called **operator 2-norm**):
$$\|A\|_2 = \|A\|_{2 \to 2} = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2 = \sigma_1(A)$$

**Proof that $\|A\|_2 = \sigma_1$.** Write $A = U\Sigma V^\top$. Then:
$$\|A\mathbf{x}\|_2^2 = \|U\Sigma V^\top \mathbf{x}\|_2^2 = \|\Sigma V^\top \mathbf{x}\|_2^2 = \|\Sigma \mathbf{y}\|_2^2 = \sum_i \sigma_i^2 y_i^2$$
where $\mathbf{y} = V^\top \mathbf{x}$ has $\|\mathbf{y}\|_2 = \|\mathbf{x}\|_2 = 1$. This is maximized by $\mathbf{y} = \mathbf{e}_1$ (put all weight on $\sigma_1$), giving $\|A\mathbf{x}\|_2^2 = \sigma_1^2$. The maximizer is $\mathbf{x} = \mathbf{v}_1$ (the top right singular vector).

**Computing the spectral norm:**
- Exact: via SVD, $\|A\|_2 = \sigma_1$
- Approximate: power iteration $\mathbf{x}_{k+1} = A^\top A \mathbf{x}_k / \|A^\top A \mathbf{x}_k\|$, converging as $(\sigma_2/\sigma_1)^k$
- For symmetric PSD matrices: $\|A\|_2 = \lambda_{\max}(A)$

**For AI:** The spectral norm of a weight matrix $W_l$ in a neural network is the Lipschitz constant of that layer (for activations with Lipschitz constant 1). The overall network Lipschitz constant is $\prod_l \|W_l\|_2$. Spectral normalization divides each weight matrix by its spectral norm at every step, bounding this product and stabilizing GAN training.

### 4.3 The Matrix 1-Norm and ∞-Norm

**Matrix 1-norm** ($p = q = 1$, input and output use $\ell^1$):
$$\|A\|_1 = \max_j \sum_i |a_{ij}| = \text{maximum absolute column sum}$$

**Matrix ∞-norm** ($p = q = \infty$, input and output use $\ell^\infty$):
$$\|A\|_\infty = \max_i \sum_j |a_{ij}| = \text{maximum absolute row sum}$$

**Why these formulas.** For the 1-norm: $\|A\mathbf{x}\|_1 = \|\sum_j x_j \mathbf{a}_j\|_1 \leq \sum_j |x_j| \|\mathbf{a}_j\|_1$. Since $\sum_j |x_j| = \|\mathbf{x}\|_1 = 1$, the worst case is to put all weight ($x_j = 1$) on the column with maximum $\ell^1$ norm.

**Examples.** For $A = \begin{pmatrix} 1 & 2 \\ -3 & 4 \end{pmatrix}$:
- Column sums: $|1| + |-3| = 4$, $|2| + |4| = 6$. So $\|A\|_1 = 6$.
- Row sums: $|1| + |2| = 3$, $|-3| + |4| = 7$. So $\|A\|_\infty = 7$.

**Duality.** $\|A\|_\infty = \|A^\top\|_1$. These two norms are transposes of each other.

### 4.4 The 2→1 and 1→2 Norms (NP-Hard)

The **nuclear norm** is the induced $2 \to 1$ norm's dual... actually, let's be precise: computing $\|A\|_{1 \to 2}$ and $\|A\|_{2 \to 1}$ is **NP-hard** in general. This is a fundamental computational barrier.

$$\|A\|_{2 \to 1} = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_1$$

No polynomial-time algorithm is known for this. However, there are useful bounds:
$$\frac{1}{\sqrt{n}}\|A\|_F \leq \|A\|_{2\to 1} \leq \sqrt{m}\|A\|_F$$

**For AI:** The NP-hardness of $\|A\|_{1\to 2}$ connects to problems in statistics (robustness of PCA) and compressed sensing. Approximate algorithms using semidefinite programming can compute $\|A\|_{2\to 1}$ up to a $O(\log n)$ factor.

### 4.5 Submultiplicativity and Consistency

**Definition.** A matrix norm $\|\cdot\|$ is:
- **Submultiplicative**: $\|AB\| \leq \|A\|\|B\|$ (all induced norms satisfy this)
- **Consistent** with vector norm $\|\cdot\|_v$: $\|A\mathbf{x}\|_v \leq \|A\|\|\mathbf{x}\|_v$ (induced norms satisfy this by construction)
- **Compatible** or **subordinate**: same as consistent

**The Frobenius norm is submultiplicative but NOT induced.** Every induced norm must satisfy $\|I\| = 1$, but $\|I\|_F = \sqrt{n}$. The Frobenius norm is sometimes called a "compatible" norm because $\|A\mathbf{x}\|_2 \leq \|A\|_F\|\mathbf{x}\|_2$.

**Non-example.** The max-entry norm $\|A\|_{\max} = \max_{i,j}|a_{ij}|$ satisfies the norm axioms but is NOT submultiplicative: $\|AB\|_{\max}$ can exceed $\|A\|_{\max}\|B\|_{\max}$.

---
## 5. Schatten Norms and the Nuclear Norm

### 5.1 The Schatten p-Norm Family

The **Schatten $p$-norm** unifies the spectral, Frobenius, and nuclear norms by applying the vector $\ell^p$ norm to the vector of singular values:

$$\|A\|_{S_p} = \|\boldsymbol{\sigma}(A)\|_p = \left(\sum_{i=1}^r \sigma_i^p\right)^{1/p}$$

where $r = \operatorname{rank}(A)$ and $\sigma_1 \geq \cdots \geq \sigma_r > 0$.

**The three canonical Schatten norms:**

| $p$ | Name | Formula | Interpretation |
|-----|------|---------|----------------|
| $1$ | Nuclear (trace) | $\sum_i \sigma_i = \operatorname{tr}(\sqrt{A^\top A})$ | Sum of singular values |
| $2$ | Frobenius | $\sqrt{\sum_i \sigma_i^2} = \sqrt{\operatorname{tr}(A^\top A)}$ | RMS singular value |
| $\infty$ | Spectral (operator) | $\max_i \sigma_i = \sigma_1$ | Maximum singular value |

**Ordering.** For $p \leq q$: $\|A\|_{S_p} \geq \|A\|_{S_q}$. In particular:
$$\|A\|_2 = \|A\|_{S_\infty} \leq \|A\|_F = \|A\|_{S_2} \leq \|A\|_* = \|A\|_{S_1}$$

with inequalities sharp for full-rank matrices with equal singular values.

**Duality.** Schatten norms are dual in pairs: $(S_p)^* = S_q$ where $1/p + 1/q = 1$. In particular:
$$\langle A, B\rangle_F \leq \|A\|_{S_p}\|B\|_{S_q} \qquad \left(\frac{1}{p}+\frac{1}{q}=1\right)$$

This is Hölder's inequality for singular values.

### 5.2 The Nuclear Norm

The **nuclear norm** (also called **trace norm** or **Ky Fan $r$-norm** when applied to all $r$ singular values) is:
$$\|A\|_* = \sum_{i=1}^r \sigma_i = \operatorname{tr}(\sqrt{A^\top A}) = \operatorname{tr}(|A|)$$

where $|A| = \sqrt{A^\top A}$ is the matrix square root (symmetric PSD). For square PSD matrices, $\|A\|_* = \operatorname{tr}(A)$ — hence "trace norm."

**The nuclear norm as the convex envelope of rank.** This is the key motivation:

$$\|A\|_* = \operatorname{conv}\{\operatorname{rank}(A) : \|A\|_2 \leq 1\}$$

More precisely, on the set $\{A : \|A\|_2 \leq 1\}$, the nuclear norm is the tightest convex lower bound on rank. This makes it the **convex relaxation of rank minimization** — replacing the NP-hard rank minimization with a convex nuclear norm minimization.

**Dual norm.** The nuclear norm is dual to the spectral norm:
$$\|A\|_* = \max_{\|B\|_2 \leq 1} \langle A, B\rangle_F = \max_{\|B\|_2 \leq 1} \operatorname{tr}(A^\top B)$$

This duality is exploited in semidefinite programming formulations of nuclear norm minimization.

### 5.3 Computing the Nuclear Norm and Proximal Operator

**Computation.** $\|A\|_* = \sum_i \sigma_i$, computed via SVD. In NumPy: `np.linalg.norm(A, 'nuc')`.

**Subdifferential.** The nuclear norm is convex but non-smooth (when $A$ has repeated singular values). Its subdifferential at $A = U\Sigma V^\top$ is:
$$\partial\|A\|_* = \{UV^\top + W : \|W\|_2 \leq 1,\, U^\top W = 0,\, WV = 0\}$$

For strictly positive singular values, $UV^\top$ is the unique subgradient.

**Proximal operator.** The proximal operator of $\lambda\|\cdot\|_*$ (needed for nuclear norm regularization) is **singular value soft-thresholding**:
$$\operatorname{prox}_{\lambda\|\cdot\|_*}(A) = U \operatorname{diag}(\max(\sigma_i - \lambda, 0)) V^\top$$

This is the key computational step in matrix completion, robust PCA, and spectral recovery algorithms.

**For AI:** Matrix completion (Netflix Prize formulation) seeks the minimum-rank matrix consistent with observed entries. The convex relaxation minimizes the nuclear norm, solvable with alternating direction method of multipliers (ADMM) using the proximal operator above. LoRA implicitly regularizes toward low nuclear norm through the factored parameterization $W = W_0 + BA$.

### 5.4 Ky Fan Norms

The **Ky Fan $k$-norm** is the sum of the top $k$ singular values:
$$\|A\|_{(k)} = \sum_{i=1}^k \sigma_i = \sigma_1 + \sigma_2 + \cdots + \sigma_k$$

**Special cases:**
- $k = 1$: $\|A\|_{(1)} = \sigma_1 = \|A\|_2$ (spectral norm)
- $k = r$: $\|A\|_{(r)} = \sum_i \sigma_i = \|A\|_*$ (nuclear norm)

Ky Fan norms form a chain: $\|A\|_{(1)} \leq \|A\|_{(2)} \leq \cdots \leq \|A\|_{(r)}$, but this is NOT a "norm" ordering — a matrix can have larger $\|A\|_{(1)}$ than another while having smaller $\|A\|_{(k)}$ for larger $k$.

**Variational formula.** The Ky Fan $k$-norm has a beautiful characterization:
$$\|A\|_{(k)} = \max_{U_k, V_k \text{ orthonormal}} \operatorname{tr}(U_k^\top A V_k)$$

where the max is over $n \times k$ orthonormal matrices $U_k, V_k$. The optimal choice is $U_k = [\mathbf{u}_1|\cdots|\mathbf{u}_k]$ and $V_k = [\mathbf{v}_1|\cdots|\mathbf{v}_k]$.

---

## 6. Unitarily Invariant Norms

### 6.1 Definition and Characterization

A matrix norm $\|\cdot\|$ is **unitarily invariant** if:
$$\|UAV\| = \|A\| \quad \text{for all unitary } U, V$$

This means the norm depends only on the singular values of $A$, not on the specific singular vectors.

**Von Neumann's characterization.** A norm is unitarily invariant if and only if it is a **symmetric gauge function** applied to the vector of singular values:
$$\|A\| = g(\sigma_1(A), \sigma_2(A), \ldots, \sigma_r(A))$$

where $g: \mathbb{R}^r \to \mathbb{R}$ is a **symmetric gauge function** — a norm on $\mathbb{R}^r$ that is:
1. A norm (positive definite, homogeneous, triangle inequality)
2. Symmetric: $g(\sigma_{\pi(1)}, \ldots, \sigma_{\pi(r)}) = g(\sigma_1, \ldots, \sigma_r)$ for any permutation $\pi$
3. Monotone: $g(\sigma_1, \ldots, \sigma_r) \geq g(\sigma_1', \ldots, \sigma_r')$ when $|\sigma_i| \geq |\sigma_i'|$ for all $i$

**Examples of unitarily invariant norms:**
- Frobenius: $g = \ell^2$ on singular values
- Spectral: $g = \ell^\infty$ on singular values
- Nuclear: $g = \ell^1$ on singular values
- Ky Fan $k$: $g(\boldsymbol{\sigma}) = \sigma_1 + \cdots + \sigma_k$
- Schatten $p$: $g = \ell^p$ on singular values

**Non-unitarily-invariant norms:**
- Matrix 1-norm $\|A\|_1$ (depends on column structure)
- Matrix $\infty$-norm $\|A\|_\infty$ (depends on row structure)
- Entry-max norm $\|A\|_{\max}$

### 6.2 Fan Dominance and Majorization

**Majorization.** A vector $\mathbf{x} \in \mathbb{R}^n$ **majorizes** $\mathbf{y} \in \mathbb{R}^n$ (written $\mathbf{x} \succ \mathbf{y}$) if:
$$\sum_{i=1}^k x_{[i]} \geq \sum_{i=1}^k y_{[i]} \text{ for all } k = 1, \ldots, n-1, \qquad \sum_{i=1}^n x_{[i]} = \sum_{i=1}^n y_{[i]}$$

where $x_{[1]} \geq \cdots \geq x_{[n]}$ are the decreasing order statistics.

**Fan's dominance theorem.** For unitarily invariant norms:
$$\|A\| \leq \|B\| \iff \sigma(A) \prec \sigma(B) \text{ (in some sense)}$$

More precisely: $\|A\| \leq \|B\|$ for ALL unitarily invariant norms if and only if $\sigma(B)$ majorizes $\sigma(A)$:
$$\sum_{i=1}^k \sigma_i(A) \leq \sum_{i=1}^k \sigma_i(B) \quad \text{for all } k$$

**This connects matrix norm comparison to singular value majorization.**

### 6.3 Weyl's Inequality (Preview of Perturbation Theory)

For unitarily invariant norms, Weyl's inequality gives bounds on singular value perturbation:

**Theorem (Weyl).** If $A, E \in \mathbb{R}^{m \times n}$, then for all $i$:
$$|\sigma_i(A + E) - \sigma_i(A)| \leq \sigma_1(E) = \|E\|_2$$

Each singular value can change by at most $\|E\|_2$ under perturbation $E$. This will be developed fully in §8.

### 6.4 The Role in Optimization

Unitarily invariant norms appear naturally in matrix optimization because they respect the singular value geometry of the problem. The **Mirsky theorem** states that for unitarily invariant norms, the best rank-$k$ approximation (in ANY unitarily invariant norm) is the truncated SVD — Eckart-Young generalizes beyond Frobenius.

**For AI:** The invariance to unitary transformations aligns with key AI structures. Transformers apply learned projections $Q = XW_Q$, $K = XW_K$; if $W_Q, W_K$ undergo orthogonal reparametrization, unitarily invariant norms capture the effective capacity independently of parameterization choice. This is exploited in analyses of transformer expressivity.

---

## 7. Condition Number

### 7.1 Definition and Intuition

The **condition number** of a nonsingular matrix $A \in \mathbb{R}^{n \times n}$ (with respect to the $p$-norm) is:

$$\kappa_p(A) = \|A\|_p \|A^{-1}\|_p$$

For $p = 2$ (using spectral norms):
$$\kappa_2(A) = \|A\|_2 \|A^{-1}\|_2 = \sigma_1(A) \cdot \frac{1}{\sigma_n(A)} = \frac{\sigma_1(A)}{\sigma_n(A)}$$

the ratio of the largest to smallest singular value.

**Geometric intuition.** The condition number measures the **eccentricity** of the ellipsoid $\{A\mathbf{x} : \|\mathbf{x}\|_2 \leq 1\}$. A sphere (all $\sigma_i$ equal) gives $\kappa = 1$. A very flat pancake (one $\sigma_n \approx 0$) gives $\kappa \to \infty$.

```
CONDITION NUMBER GEOMETRY
════════════════════════════════════════════════════════════════════════

  κ(A) = 1 (identity)        κ(A) = 10 (moderate)       κ(A) → ∞ (near-singular)

      ●●●                        ●●●●●●●                    ─────────────────────
     ●   ●                      ●       ●                   ─────────────────────
      ●●●                        ●●●●●●●                    ─────────────────────

  σ₁/σ₂ = 1                  σ₁/σ₂ = 10                 σ₁/σ₂ → ∞
  (sphere)                    (ellipse)                   (flat disk → singular)

════════════════════════════════════════════════════════════════════════
```

### 7.2 Condition Number and Numerical Stability

The condition number quantifies how sensitive the solution of $A\mathbf{x} = \mathbf{b}$ is to perturbations.

**Forward error bound.** If $\hat{\mathbf{x}}$ solves $(A + \delta A)\hat{\mathbf{x}} = \mathbf{b} + \delta\mathbf{b}$, then:
$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A)\left(\frac{\|\delta A\|}{\|A\|} + \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}\right)$$

The **relative error in the solution is at most $\kappa(A)$ times the relative error in the data.** If the input has $t$ digits of precision, the output has roughly $t - \log_{10}\kappa(A)$ reliable digits.

**Example.** For $\kappa(A) = 10^6$ and double-precision arithmetic ($\approx 16$ correct digits), the solution has $\approx 10$ reliable digits. For $\kappa(A) = 10^{12}$, only 4 reliable digits remain.

**For AI:** The Hessian of the loss $H = \nabla^2 \mathcal{L}$ has condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$. High $\kappa(H)$ means:
1. Gradient descent requires tiny step sizes (limited by $1/\lambda_{\max}$) but needs many steps (to make progress along $\lambda_{\min}$ directions)
2. The loss landscape has steep valleys — the classic "ill-conditioned optimization" problem
3. Adam and other adaptive optimizers implicitly precondition to reduce effective $\kappa$

### 7.3 Properties and Bounds

**Key properties:**
1. $\kappa(A) \geq 1$ for all nonsingular $A$ (since $1 = \|I\| = \|AA^{-1}\| \leq \|A\|\|A^{-1}\|$)
2. $\kappa(A) = 1$ iff $A$ is a scalar multiple of a unitary matrix
3. $\kappa(AB) \leq \kappa(A)\kappa(B)$ — condition numbers multiply
4. $\kappa(A) = \kappa(A^{-1})$
5. $\kappa(\alpha A) = \kappa(A)$ for any scalar $\alpha \neq 0$

**Norm dependence.** Different norms give different condition numbers, but they're equivalent up to polynomial factors in $n$.

**For singular matrices:** We define $\kappa(A) = \infty$ (singular matrices are maximally ill-conditioned).

**Distance to singularity.** The condition number relates to how far $A$ is from the nearest singular matrix:
$$\min\{\|E\| : A + E \text{ is singular}\} = \frac{\|A\|}{\kappa(A)} = \sigma_n(A)$$

(using the spectral norm). A matrix with $\kappa(A) = 10^6$ is "close to singular" in a relative sense.

### 7.4 Tikhonov Regularization

When $\kappa(A)$ is too large for reliable computation, **Tikhonov regularization** artificially improves conditioning:

$$(A^\top A + \lambda I)\hat{\mathbf{x}} = A^\top \mathbf{b}$$

**Condition number of the regularized system:**
$$\kappa_2(A^\top A + \lambda I) = \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda}$$

Since $\sigma_n^2 + \lambda \geq \lambda > 0$, the regularized system has finite condition number even when $A$ is singular ($\sigma_n = 0$). As $\lambda$ increases, $\kappa$ decreases toward 1.

**Solution bias.** The Tikhonov solution has smaller norm than the minimum-norm least-squares solution: $\|\hat{\mathbf{x}}_\lambda\| < \|\hat{\mathbf{x}}_{LS}\|$. This is the bias-variance tradeoff: regularization introduces bias to reduce variance (sensitivity to noise).

**For AI:** L2 regularization (weight decay) in neural networks is essentially Tikhonov regularization applied to the loss landscape. Adding $\lambda\|W\|_F^2$ to the loss shifts the Hessian eigenvalues by $2\lambda$, bounding $\kappa(H + 2\lambda I) \leq (\lambda_{\max} + 2\lambda)/(2\lambda)$ — improving the condition number of the optimization problem.

### 7.5 Estimating Condition Numbers

Computing $\kappa(A)$ exactly requires full SVD ($O(mn^2)$ for $m \geq n$). For large matrices, approximations are needed.

**Power method / Lanczos.** Estimate $\sigma_1$ by power iteration on $A^\top A$. Estimate $\sigma_n$ by inverse power iteration (more expensive, requires solving linear systems).

**LAPACK routines.** `scipy.linalg.svd` + ratio; or `np.linalg.cond(A, p)` for various $p$-norms.

**Randomized condition number estimation.** For $A \in \mathbb{R}^{n \times n}$: generate $\mathbf{g} \sim \mathcal{N}(0, I)$, compute $\|A\mathbf{g}\|/\|A^{-1}\mathbf{g}\|$. This estimates $\kappa(A)$ with high probability using just two matrix-vector products.

---

## 8. Perturbation Theory

### 8.1 Motivation and Setup

**The fundamental question of numerical analysis:** If we perturb the data slightly, how much do the outputs change?

For matrix computations, we study:
- How much do eigenvalues change when $A \to A + E$?
- How much do singular values change when $A \to A + E$?
- How much does the solution $\mathbf{x} = A^{-1}\mathbf{b}$ change when $A$ or $\mathbf{b}$ are perturbed?

All answers involve matrix norms.

**Setup.** Let $A \in \mathbb{R}^{m \times n}$ with SVD $A = U\Sigma V^\top$, singular values $\sigma_1 \geq \cdots \geq \sigma_p$ ($p = \min(m,n)$). Let $\tilde{A} = A + E$ be the perturbed matrix with singular values $\tilde{\sigma}_1 \geq \cdots \geq \tilde{\sigma}_p$.

### 8.2 Weyl's Inequality for Singular Values

**Theorem (Weyl's Inequality).** For any $i$:
$$|\tilde{\sigma}_i - \sigma_i| \leq \|E\|_2 = \sigma_1(E)$$

**Proof sketch.** By the variational formula for singular values (Courant-Fischer), $\sigma_i(A) = \min_{\dim U = i} \max_{\mathbf{x} \perp U} \|A\mathbf{x}\|/\|\mathbf{x}\|$. Adding $E$ shifts this by at most $\|E\mathbf{x}\|/\|\mathbf{x}\| \leq \|E\|_2$.

**Interpretation:** Singular values are **Lipschitz** functions of the matrix with Lipschitz constant 1 (with respect to the spectral norm):
$$\|\boldsymbol{\sigma}(A+E) - \boldsymbol{\sigma}(A)\|_2 \leq \|E\|_F \leq \sqrt{p}\|E\|_2$$

This makes singular values numerically stable — small matrix perturbations cause small changes.

**Stronger form (Mirsky's theorem):**
$$\|\boldsymbol{\sigma}(A+E) - \boldsymbol{\sigma}(A)\|_F \leq \|E\|_F$$

In Frobenius norm, the vector of singular values is also Lipschitz-1.

### 8.3 Eigenvalue Perturbation: Bauer-Fike

For symmetric matrices $A$ with eigenvalues $\lambda_i$, the analog of Weyl is:
$$|\tilde{\lambda}_i - \lambda_i| \leq \|E\|_2$$

For **non-symmetric** matrices, the situation is more complex. The **Bauer-Fike theorem** bounds eigenvalue perturbation for diagonalizable $A = PDP^{-1}$:
$$\min_j |\tilde{\lambda} - \lambda_j| \leq \kappa_2(P) \|E\|_2$$

where $\kappa_2(P) = \|P\|_2\|P^{-1}\|_2$ is the condition number of the eigenvector matrix.

**Key insight:** Eigenvalues of non-normal matrices can be extremely sensitive to perturbation. The condition number $\kappa_2(P)$ — which can be astronomically large for non-normal matrices — amplifies the perturbation. This is why direct eigenvalue computation for non-normal matrices is numerically dangerous.

**For AI:** The sensitivity of eigenvalues of non-normal operators appears in RNN/LSTM gradient flow analysis. The eigenvalues of the recurrent weight matrix $W_{hh}$ determine long-term memory, but if $W_{hh}$ is non-normal, small weight perturbations (from noise, finite-precision, or gradient updates) can dramatically change eigenvalues and hence gradient behavior.

### 8.4 Forward and Backward Error Analysis

**Forward error:** $\|\hat{x} - x\|$ — how wrong is the computed answer?
**Backward error:** $\min\{\|\delta A\| : (A + \delta A)\hat{x} = b\}$ — how much do we need to perturb the input for $\hat{x}$ to be exact?

**Fundamental relationship:**
$$\text{Forward error} \leq \kappa(A) \times \text{Backward error}$$

An algorithm is **backward stable** if it produces output $\hat{x}$ with small backward error. For backward-stable algorithms on problems with small condition number, the forward error is also small.

**Example.** Gaussian elimination with partial pivoting is backward stable: the computed solution $\hat{x}$ satisfies $(A + E)\hat{x} = b$ where $\|E\|_F / \|A\|_F \leq c n \epsilon_{mach}$ (small backward error). The forward error is bounded by $\kappa(A) \cdot cn\epsilon_{mach}$.

---
## 9. Applications in Machine Learning

### 9.1 Weight Regularization: Frobenius vs Nuclear

Neural network training minimizes a loss with a regularizer:
$$\mathcal{L}(W) = \text{task loss} + \lambda R(W)$$

**Frobenius (L2/weight decay):** $R(W) = \|W\|_F^2 = \sum_i \sigma_i^2$

- Gradient: $\nabla R = 2W$
- Effect: shrinks all singular values uniformly
- Induces: no sparsity in singular values; doesn't promote low rank
- Used in: essentially all deep learning (Adam, SGD with weight decay)

**Nuclear norm:** $R(W) = \|W\|_* = \sum_i \sigma_i$

- Subgradient: $UV^\top$ (top singular vectors)
- Effect: promotes sparsity in singular values → low-rank solutions
- Proximal step: singular value soft-thresholding
- Used in: matrix completion, robust PCA, some transformer pruning methods

**Spectral norm:** $R(W) = \|W\|_2 = \sigma_1$

- Subgradient: $\mathbf{u}_1\mathbf{v}_1^\top$ (rank-1 outer product)
- Effect: limits largest singular value; controls Lipschitz constant
- Used in: spectral normalization for GAN training (Miyato 2018)

**LoRA and nuclear norm.** The LoRA reparameterization $W = W_0 + BA$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$) has an interesting norm structure:
$$\|BA\|_* \leq \|B\|_F\|A\|_F$$
The product parameterization implicitly regularizes the nuclear norm of the weight increment, even without explicit regularization.

### 9.2 Spectral Normalization for GANs

Generative Adversarial Networks (GANs) require the discriminator $D: \mathbb{R}^d \to \mathbb{R}$ to be Lipschitz:
$$|D(\mathbf{x}) - D(\mathbf{y})| \leq L\|\mathbf{x} - \mathbf{y}\|_2 \quad \forall \mathbf{x}, \mathbf{y}$$

For a neural network $D = D_L \circ \cdots \circ D_1$ where $D_l(\mathbf{x}) = \phi(W_l\mathbf{x} + \mathbf{b}_l)$, and the activation $\phi$ has Lipschitz constant 1 (ReLU, tanh, etc.):
$$\text{Lip}(D) \leq \prod_{l=1}^L \|W_l\|_2$$

**Spectral normalization** (Miyato et al., 2018) replaces each $W_l$ with $\hat{W}_l = W_l / \|W_l\|_2$, ensuring each layer has spectral norm exactly 1, hence:
$$\text{Lip}(D_{\text{SN}}) \leq 1$$

**Implementation.** The spectral norm is estimated via power iteration — a single step per training iteration suffices in practice:
```
v ← W^T u / ||W^T u||
u ← W v / ||W v||
σ̂ ← u^T W v
Ŵ ← W / σ̂
```

This adds minimal overhead and dramatically stabilizes GAN training.

### 9.3 Gradient Clipping and Norm Bounds

**Gradient clipping** prevents gradient explosion by rescaling gradients when their norm exceeds a threshold:
$$\hat{g} = \begin{cases} g & \text{if } \|g\| \leq \tau \\ \tau \cdot g / \|g\| & \text{if } \|g\| > \tau \end{cases}$$

**Choice of norm matters:**
- **Global L2 clipping:** $\|g\| = \|\nabla_\theta \mathcal{L}\|_2$ (all parameters concatenated). Standard in transformers (Vaswani 2017): clip at $\tau = 1.0$.
- **Per-tensor clipping:** clip each weight matrix's gradient separately using its spectral norm. More expensive but more principled.
- **Frobenius clipping:** clip by $\|\nabla_W \mathcal{L}\|_F$ per layer. Common in modern optimizers.

**For AI:** The choice of gradient clipping norm affects which singular value components of the gradient are preserved. Clipping by spectral norm $\|G\|_2$ ensures the maximum singular value doesn't exceed $\tau$, while Frobenius clipping preserves the relative structure of all singular values but scales them uniformly.

### 9.4 Low-Rank Approximation and Model Compression

The Eckart-Young theorem justifies truncated SVD for model compression:

**Weight matrix compression.** For a weight matrix $W \in \mathbb{R}^{d_1 \times d_2}$, store the rank-$k$ approximation $W_k = U_k \Sigma_k V_k^\top$ using $k(d_1 + d_2)$ parameters instead of $d_1 d_2$. The approximation error:
$$\|W - W_k\|_F = \sqrt{\sum_{i>k} \sigma_i^2}, \qquad \|W - W_k\|_2 = \sigma_{k+1}$$

**Compression ratio.** For compression ratio $r = d_1 d_2 / k(d_1 + d_2)$, we need $k < d_1 d_2 / (d_1 + d_2)$.

**LoRA revisited.** Instead of post-hoc compression, LoRA trains the low-rank increments directly:
$$W = W_{\text{pretrained}} + \underbrace{B}_{\mathbb{R}^{d \times r}} \underbrace{A}_{\mathbb{R}^{r \times d}}$$

The parameter savings are dramatic: fine-tuning $r \ll d$ parameters per layer instead of $d^2$.

### 9.5 Lipschitz Bounds and Generalization

**PAC-Bayes generalization bounds** connect norms of weight matrices to the generalization gap. For a feedforward network with $L$ layers:

$$\text{Generalization gap} \lesssim \frac{1}{\sqrt{n}} \cdot \kappa_{\text{network}} \cdot \prod_{l=1}^L \|W_l\|_F$$

where the precise bound depends on the norm choice. Bartlett et al. (2017) proved a spectrally-normalized margin bound:

$$\text{Generalization gap} \lesssim \frac{B^2}{\gamma^2} \sqrt{\frac{\left(\sum_l \|W_l\|_F^2 / \|W_l\|_2^2\right) \cdot \prod_l \|W_l\|_2^2}{n}}$$

where $B$ is the input bound, $\gamma$ is the margin, and $n$ is the training set size.

**Key insight:** The ratio $\|W\|_F^2 / \|W\|_2^2 = \sum_i (\sigma_i/\sigma_1)^2 \leq r$ (the stable rank) measures effective dimensionality. Matrices with low stable rank generalize better.

### 9.6 Attention Mechanisms and Low-Rank Structure

The attention mechanism in transformers computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

The matrix $S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{T \times T}$ (for sequence length $T$) is the **attention logit matrix**.

**Nuclear norm of attention.** Empirically, trained attention matrices have low nuclear norm relative to their Frobenius norm — they exhibit low-rank structure. Dong et al. (2021) ("Attention is not All You Need") showed that purely attention-based models without MLP layers collapse to rank-1, connecting to the spectral norm of the attention matrix.

**Multi-head Low-Rank Attention (MLA).** DeepSeek's MLA architecture (2024) explicitly parameterizes key-value caches in low-rank form to reduce memory:
$$K = C_{KV} W^K, \quad V = C_{KV} W^V$$
where $C_{KV} \in \mathbb{R}^{T \times d_c}$ with $d_c \ll d_{model}$. This is exactly SVD-based compression of the KV cache, motivated by the low-rank structure of attention matrices.

### 9.7 Implicit Regularization in Deep Learning

Deep learning optimizers (SGD, Adam) exhibit **implicit regularization** — they converge to solutions with small norms even without explicit regularization.

**Observation (Gunasekar et al., 2017).** Gradient flow on matrix factorization $\min_{U,V} \|UV^\top - A\|_F^2$ converges to the minimum nuclear norm solution. This is because the dynamics $\dot{U} = -(UV^\top - A)V$, $\dot{V} = -(UV^\top - A)^\top U$ implicitly minimize $\|UV^\top\|_*$.

**Spectral norm implicit regularization.** For overparameterized linear networks (depth $\geq 2$), gradient descent with small step size finds solutions with small spectral norm of the end-to-end product $W_L \cdots W_1$.

**For AI (2026).** Modern foundation models benefit from understanding these implicit biases:
- LoRA fine-tuning implicitly penalizes nuclear norm of the weight increment
- Adam with weight decay (decoupled) implicitly regularizes toward solutions with small $\|W\|_F$
- Gradient clipping implicitly regularizes the spectral norm of gradient steps

---
## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Confusing $\|A\|_1$ (matrix 1-norm = max column sum) with $\|A\|_F$ (Frobenius) | The notation $\|A\|_1$ for matrices means max column sum, not the sum of absolute entries | Use `np.linalg.norm(A, 1)` for max col sum; `np.linalg.norm(A, 'fro')` for Frobenius |
| 2 | Assuming the Frobenius norm is induced | $\|I\|_F = \sqrt{n} \neq 1$; no induced norm can have $\|I\| \neq 1$ | Remember: Frobenius = Euclidean on vectorized matrix; consistent but not induced |
| 3 | Confusing $\|A\|_2$ (spectral norm = $\sigma_1$) with $\|A\|_F$ | For a general matrix, $\|A\|_2 \leq \|A\|_F$; they equal only for rank-1 matrices | $\|A\|_2 = \sigma_1$; $\|A\|_F = \|\boldsymbol{\sigma}\|_2$ — very different |
| 4 | Thinking $\|AB\|_F = \|A\|_F\|B\|_F$ | This equality fails in general; it's an upper bound | The submultiplicativity $\|AB\|_F \leq \|A\|_F\|B\|_F$ is an inequality, not equality |
| 5 | Applying the spectral norm formula $\kappa = \sigma_1/\sigma_n$ to singular (or non-square) matrices | Singular matrices have $\sigma_n = 0$, giving $\kappa = \infty$ | For non-square or rank-deficient matrices, use the pseudocondition $\sigma_1/\sigma_r$ where $\sigma_r$ is the smallest nonzero singular value |
| 6 | Forgetting that condition number depends on the norm choice | $\kappa_1(A)$, $\kappa_2(A)$, $\kappa_\infty(A)$ can differ by polynomial factors in $n$ | Specify the norm; use $\kappa_2$ (spectral) unless another norm is more natural |
| 7 | Assuming small Frobenius norm implies small condition number | A matrix can have $\|A\|_F = 1$ but $\kappa(A) = 10^{15}$ | $\kappa$ depends on the ratio $\sigma_1/\sigma_n$, not the absolute values |
| 8 | Misidentifying the matrix 1-norm as the entry-wise 1-norm (sum of all entries) | $\|A\|_1$ (induced) is max column sum; $\sum_{i,j}|a_{ij}|$ is sometimes called the "entry norm" but is NOT the induced 1-norm | The induced $\ell^1 \to \ell^1$ norm = max col sum; the Schatten $S_1$ norm = nuclear norm — entirely different objects |
| 9 | Assuming Weyl's inequality gives tight bounds on eigenvalue changes for non-symmetric matrices | For non-symmetric matrices, Weyl doesn't apply; use Bauer-Fike, which involves $\kappa(P)$ | Non-normal matrices can have extreme eigenvalue sensitivity; use pseudo-spectrum for analysis |
| 10 | Thinking nuclear norm minimization always recovers low-rank matrices | Recovery requires incoherence conditions (columns/rows not aligned with standard basis) | Nuclear norm minimization provably recovers rank-$r$ matrices under RIP or incoherence; check conditions before claiming recovery |
| 11 | Confusing spectral normalization (normalizes by $\|W\|_2$) with batch normalization (normalizes activations) | Spectral normalization is on weight matrices; batch normalization is on hidden layer activations | They target different quantities: spectral norm controls Lipschitz constant; batch norm controls activation statistics |
| 12 | Assuming stable rank = rank | Stable rank $\rho(A) = \|A\|_F^2 / \|A\|_2^2 = \sum(\sigma_i/\sigma_1)^2$; it's a real number, not an integer, and $\rho(A) \leq r = \operatorname{rank}(A)$ | Stable rank is a smooth proxy for rank that appears in generalization bounds |

---

## 11. Exercises

**Exercise 1** ★ — Norm computation and verification

Let $A = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 2 & -1 \\ 1 & 0 & 4 \end{pmatrix}$.

(a) Compute $\|A\|_F$ (Frobenius norm) by hand and verify using the SVD formula $\|A\|_F = \|\boldsymbol{\sigma}\|_2$.

(b) Compute $\|A\|_1$ (max absolute column sum) and $\|A\|_\infty$ (max absolute row sum). Verify $\|A^\top\|_1 = \|A\|_\infty$.

(c) Compute $\|A\|_2 = \sigma_1(A)$ numerically. Verify the inequalities $\|A\|_2 \leq \|A\|_F \leq \sqrt{3}\|A\|_2$.

(d) Compute $\|A\|_*$ (nuclear norm). Verify $\|A\|_* \leq \sqrt{3}\|A\|_F$ and $\|A\|_* \geq \|A\|_F$ (when checking your numbers!).

---

**Exercise 2** ★ — Frobenius norm properties

(a) Prove that $\|A\|_F^2 = \operatorname{tr}(A^\top A)$.

(b) Show that the Frobenius inner product $\langle A, B\rangle_F = \operatorname{tr}(A^\top B)$ satisfies all inner product axioms.

(c) For $A = \mathbf{u}\mathbf{v}^\top$ (rank-1), show $\|A\|_F = \|\mathbf{u}\|_2\|\mathbf{v}\|_2 = \|A\|_2$. Verify numerically.

(d) Verify submultiplicativity $\|AB\|_F \leq \|A\|_F\|B\|_F$ for random matrices. Show it fails as equality for most pairs.

---

**Exercise 3** ★ — Condition number analysis

(a) Construct a $3 \times 3$ matrix with condition number $\kappa_2 \approx 100$ and one with $\kappa_2 \approx 10^6$.

(b) For each matrix, solve $A\mathbf{x} = \mathbf{b}$ with $\mathbf{b} = A\mathbf{x}_0$ for known $\mathbf{x}_0 = [1,1,1]^\top$. Add noise $\delta\mathbf{b} = \epsilon \cdot \text{rand}$ with $\epsilon = 10^{-6}$. Measure the relative error $\|\hat{\mathbf{x}} - \mathbf{x}_0\| / \|\mathbf{x}_0\|$.

(c) Verify the bound: relative forward error $\leq \kappa(A) \cdot$ relative backward error.

(d) Apply Tikhonov regularization with $\lambda = 10^{-3}$ to the ill-conditioned matrix. Compute the new condition number and solution accuracy.

---

**Exercise 4** ★★ — Nuclear norm and low-rank approximation

(a) Generate a random rank-3 matrix $A = UV^\top$ with $U \in \mathbb{R}^{8 \times 3}$, $V \in \mathbb{R}^{10 \times 3}$. Confirm $\|A\|_* = \sum_{i=1}^3 \sigma_i$ and $\operatorname{rank}(A) = 3$.

(b) Add Gaussian noise $E$ with $\|E\|_F = 0.1\|A\|_F$. Compute the best rank-$k$ approximations $\tilde{A}_k$ for $k = 1, 2, 3, 5, 8$. Plot $\|A - \tilde{A}_k\|_F$ vs $k$.

(c) Verify the Eckart-Young theorem: $\|A - \tilde{A}_k\|_F^2 = \sum_{i>k}\sigma_i^2$ and $\|A - \tilde{A}_k\|_2 = \sigma_{k+1}$.

(d) Implement the proximal operator for the nuclear norm (singular value soft-thresholding) and verify it on a $5 \times 5$ random matrix with threshold $\lambda = 2$.

---

**Exercise 5** ★★ — Weyl's inequality and perturbation

(a) Verify Weyl's inequality: construct $A, E \in \mathbb{R}^{5 \times 5}$ and verify $|\sigma_i(A+E) - \sigma_i(A)| \leq \|E\|_2$ for all $i$.

(b) Show (numerically) that Weyl's bound is tight: construct $A$ and $E$ such that equality is achieved for $i=1$.

(c) For a non-symmetric matrix $A$, compute eigenvalues of $A + E$ for several random perturbations $\|E\|_2 = \epsilon$. Compare the eigenvalue scatter to the singular value scatter — demonstrate that eigenvalues of non-normal matrices are more sensitive.

(d) Bauer-Fike: compute $\kappa_2(P)$ where $P$ is the eigenvector matrix of your non-symmetric $A$. Verify the bound $\min_j|\tilde{\lambda} - \lambda_j| \leq \kappa_2(P)\|E\|_2$.

---

**Exercise 6** ★★ — Dual norms and optimization

(a) Verify the duality $\|A\|_* = \max_{\|B\|_2 \leq 1}\operatorname{tr}(A^\top B)$ numerically for a random $4 \times 4$ matrix.

(b) The dual of $\|A\|_F$ is itself. Verify: $\|A\|_F = \max_{\|B\|_F \leq 1}\operatorname{tr}(A^\top B)$.

(c) The Ky Fan $k$-norm has a variational formula: $\|A\|_{(k)} = \max_{U_k, V_k \text{ orthonormal}}\operatorname{tr}(U_k^\top A V_k)$. Verify for $k=2$ on a $5 \times 5$ matrix.

(d) Write the subgradient of $\|A\|_*$ at a rank-2 matrix $A = \mathbf{u}_1\mathbf{v}_1^\top + \mathbf{u}_2\mathbf{v}_2^\top$. Verify it numerically by finite differences.

---

**Exercise 7** ★★★ — Spectral normalization for neural networks

Implement spectral normalization for a simple 2-layer network on a toy binary classification problem.

(a) Implement power iteration to estimate $\sigma_1(W)$ for a weight matrix. Compare to `np.linalg.svd(W)[1][0]`. Measure convergence rate.

(b) Train a 2-layer network with and without spectral normalization on a toy dataset. Track $\|W_l\|_2$ during training.

(c) Estimate the Lipschitz constant $\prod_l \|W_l\|_2$ during training. Compare the two settings.

(d) Demonstrate that spectral normalization stabilizes training: use a larger learning rate that causes instability without SN but converges with SN.

---

**Exercise 8** ★★★ — Implicit regularization in matrix factorization

(a) Solve $\min_{U,V} \|UV^\top - A\|_F^2$ for a random $6 \times 6$ matrix $A$ with gradient flow (small step gradient descent). Initialize $U, V$ close to zero.

(b) Track $\|UV^\top\|_*$ during optimization. Compare to the minimum nuclear norm completion $\min\{\|X\|_* : X = A\}$ (which equals $\|A\|_*$ if $A$ is full rank).

(c) Now let $A$ be rank-2. Show that gradient flow recovers the rank-2 structure even without explicit rank regularization: plot the singular values of $UV^\top$ over iterations.

(d) Compare the implicit nuclear norm regularization of matrix factorization to explicit nuclear norm minimization using the proximal gradient method (from Exercise 4(d)). Which finds a solution faster? Which has smaller $\|X\|_*$?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Application | Impact |
|---------|-------------------|--------|
| Frobenius norm | Weight decay (L2 regularization): penalizes $\|W\|_F^2$ | Controls parameter magnitudes; implicit bias toward small-norm solutions; universal in all optimizers with `weight_decay` |
| Spectral norm ($\sigma_1$) | Spectral normalization in GANs; Lipschitz bound for discriminators; RoPE analysis | Guarantees Lipschitz-1 layers; stabilizes adversarial training; bounds gradient flow |
| Nuclear norm | Low-rank regularization; matrix completion; LoRA implicit regularization; DoRA | Promotes sparse singular values; the convex proxy for rank; connects fine-tuning to compressed sensing |
| Condition number | Loss landscape analysis; preconditioned optimizers (Adam); convergence theory | High $\kappa$ → slow convergence; Adam approximates $\kappa(H)^{-1}$ preconditioning; Shampoo computes exact preconditioning |
| Weyl's inequality | Stability of trained models under weight noise; pruning/quantization error bounds | Guarantees singular value stability; bounds the effect of INT8 quantization on model behavior |
| Eckart-Young theorem | LoRA, SVD compression, attention matrix analysis | Mathematical foundation of parameter-efficient fine-tuning; optimal rank-$k$ approximation in both F and spectral norms |
| Stable rank | PAC-Bayes generalization bounds; network complexity measures | $\rho(W) = \|W\|_F^2/\|W\|_2^2$ predicts generalization better than raw rank |
| Dual norms | Duality in optimization (nuclear ↔ spectral); proximal methods; online learning | Fenchel duality underlies mirror descent, natural gradient, and FTRL algorithms |
| Bauer-Fike theorem | RNN gradient flow analysis; eigenvalue sensitivity of recurrent matrices | Non-normal recurrent weight matrices can have extreme eigenvalue sensitivity even with bounded $\|W\|_2$ |
| Proximal gradient | Nuclear norm minimization algorithms; ADMM for matrix completion | Singular value soft-thresholding is the key computational primitive for nuclear norm optimization |
| Low-rank structure | MLA (DeepSeek); attention compression; GQA/MQA | Empirical discovery that trained attention has low nuclear/stable rank; motivates linear attention approximations |
| Tikhonov regularization | Ridge regression; L2 penalty in fine-tuning; conditioning of Gram matrices in kernel methods | Converts ill-conditioned problems into well-conditioned ones; connects to weight decay |

---

## 13. Conceptual Bridge

Matrix norms are the measuring instruments of linear algebra. Without them, we cannot quantify stability, sensitivity, approximation quality, or regularization strength. Every major result in numerical linear algebra — condition number theory, perturbation bounds, error analysis — is stated in terms of matrix norms. And in machine learning, norms appear at every level: they define regularizers, measure approximation error, bound generalization, and determine convergence rates.

**What came before.** This section builds on the SVD (→ [Singular Value Decomposition](../02-Singular-Value-Decomposition/notes.md)), which provides the singular values that define the spectral, Frobenius, nuclear, and Schatten norms. It uses eigenvalue theory (→ [Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md)) for the spectral norm of symmetric PSD matrices and for condition number of symmetric positive definite systems. It uses orthogonality (→ [Orthogonality](../05-Orthogonality-and-Orthonormality/notes.md)) for the unitary invariance of Schatten norms.

**What comes after.** Condition number analysis is essential for numerical methods in the next chapter. Matrix norm theory enables gradient analysis in optimization (→ Chapter 8: Optimization). The nuclear norm and low-rank regularization connect directly to dimensionality reduction (→ [PCA](../03-PCA/notes.md)) and the mathematical foundations of LoRA and other PEFT methods in the models chapters. Perturbation theory connects to the stability analysis of RNNs and LSTMs (→ [RNN and LSTM Math](../../14-Math-for-Specific-Models/04-RNN-and-LSTM-Math/notes.md)).

```
MATRIX NORMS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  02-Linear-Algebra-Basics              03-Advanced-Linear-Algebra
  ───────────────────────               ────────────────────────────

  ┌──────────────────────┐             ┌───────────────────────────────┐
  │ 05-Matrix-Rank       │  →  rank    │ 01-Eigenvalues-Eigenvectors   │
  │ 04-Determinants      │  →  det     │ 02-SVD  ──┐                   │
  └──────────────────────┘             │           │ singular values   │
                                       │           ↓                   │
                                       │ ┌─────────────────────────┐  │
                                       │ │  06-Matrix-Norms  ★     │  │
                                       │ │                         │  │
                                       │ │  ‣ Frobenius (F-norm)   │  │
                                       │ │  ‣ Spectral (σ₁)        │  │
                                       │ │  ‣ Nuclear (Σσᵢ)        │  │
                                       │ │  ‣ Schatten (ℓᵖ of σ)  │  │
                                       │ │  ‣ Condition number     │  │
                                       │ │  ‣ Perturbation theory  │  │
                                       │ └─────────────┬───────────┘  │
                                       │               │               │
                                       │               ↓               │
                                       │ 07-Linear-Transformations     │
                                       │ 08-Matrix-Decompositions      │
                                       └───────────────────────────────┘
                                                       │
                                     ┌─────────────────┼────────────────┐
                                     ↓                 ↓                ↓
                              08-Optimization    ML-Applications    14-Specific
                              (gradient flow,   (LoRA, spectral    (RNN stability,
                               convergence)      normalization,     attention rank,
                                                 PAC-Bayes)        MLA compression)

════════════════════════════════════════════════════════════════════════
```

The key insight unifying this section: **matrix norms reduce questions about linear operators to scalar quantities.** The spectral norm reduces Lipschitz analysis to a number. The nuclear norm reduces rank minimization to a convex program. The condition number reduces numerical stability to a ratio. This reduction from functional complexity to scalar measurement is the fundamental technical move that makes both theory and computation tractable.

---

## Appendix A: Proofs and Derivations

### A.1 Proof that $\|A\|_F = \sqrt{\operatorname{tr}(A^\top A)}$

Let $A \in \mathbb{R}^{m \times n}$. The $(j,k)$-entry of $A^\top A$ is:
$$(A^\top A)_{jk} = \sum_i (A^\top)_{ji} A_{ik} = \sum_i A_{ij} A_{ik}$$

The trace sums the diagonal ($j = k$):
$$\operatorname{tr}(A^\top A) = \sum_j (A^\top A)_{jj} = \sum_j \sum_i A_{ij}^2 = \sum_{i,j} a_{ij}^2 = \|A\|_F^2 \qquad \square$$

**Corollary.** For any unitary $U \in \mathbb{R}^{m \times m}$ and $V \in \mathbb{R}^{n \times n}$:
$$\|UAV\|_F^2 = \operatorname{tr}((UAV)^\top (UAV)) = \operatorname{tr}(V^\top A^\top U^\top UAV) = \operatorname{tr}(V^\top A^\top AV) = \operatorname{tr}(A^\top A) = \|A\|_F^2$$

using $U^\top U = I$ and cyclic invariance of the trace $\operatorname{tr}(VXV^\top) = \operatorname{tr}(X)$.

### A.2 Proof of Submultiplicativity of the Frobenius Norm

We prove $\|AB\|_F \leq \|A\|_F \|B\|_F$.

Let $A \in \mathbb{R}^{m \times p}$ and $B \in \mathbb{R}^{p \times n}$. Denote the rows of $A$ as $\mathbf{a}_i^\top$ and columns of $B$ as $\mathbf{b}_j$. Then $(AB)_{ij} = \mathbf{a}_i^\top \mathbf{b}_j$.

By Cauchy-Schwarz: $|(AB)_{ij}|^2 \leq \|\mathbf{a}_i\|_2^2 \|\mathbf{b}_j\|_2^2$. Summing over all $i,j$:
$$\|AB\|_F^2 = \sum_{i,j} |(AB)_{ij}|^2 \leq \sum_{i,j} \|\mathbf{a}_i\|^2 \|\mathbf{b}_j\|^2 = \left(\sum_i \|\mathbf{a}_i\|^2\right)\left(\sum_j \|\mathbf{b}_j\|^2\right) = \|A\|_F^2 \|B\|_F^2 \qquad \square$$

### A.3 Proof that the Matrix 1-Norm Equals Maximum Column Sum

**Claim.** $\|A\|_{1 \to 1} = \max_j \sum_i |a_{ij}|$.

**Upper bound.** For $\|\mathbf{x}\|_1 = 1$:
$$\|A\mathbf{x}\|_1 = \left\|\sum_j x_j \mathbf{a}_j\right\|_1 \leq \sum_j |x_j| \|\mathbf{a}_j\|_1 \leq \max_j \|\mathbf{a}_j\|_1 \cdot \sum_j |x_j| = \max_j \|\mathbf{a}_j\|_1$$

**Attainment.** Let $j^* = \arg\max_j \|\mathbf{a}_{j}\|_1$. Set $\mathbf{x} = \mathbf{e}_{j^*}$. Then $A\mathbf{x} = \mathbf{a}_{j^*}$, so $\|A\mathbf{x}\|_1 = \max_j\|\mathbf{a}_j\|_1$. $\square$

### A.4 Proof of Weyl's Inequality

**Theorem.** For $A, E \in \mathbb{R}^{m \times n}$ with $m \geq n$, and all $i = 1, \ldots, n$:
$$|\sigma_i(A+E) - \sigma_i(A)| \leq \sigma_1(E) = \|E\|_2$$

**Proof using the minimax characterization.** By Courant-Fischer:
$$\sigma_i(A) = \min_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = n-i+1}} \max_{\mathbf{x} \in S, \|\mathbf{x}\|=1} \|A\mathbf{x}\|$$

Let $S^*$ achieve the minimax for $\sigma_i(A)$. Then:
$$\sigma_i(A+E) \leq \max_{\mathbf{x} \in S^*, \|\mathbf{x}\|=1} \|(A+E)\mathbf{x}\| \leq \max_{\mathbf{x} \in S^*, \|\mathbf{x}\|=1} \left(\|A\mathbf{x}\| + \|E\mathbf{x}\|\right) \leq \sigma_i(A) + \|E\|_2$$

By symmetry (apply to $A+E$ with perturbation $-E$): $\sigma_i(A) \leq \sigma_i(A+E) + \|E\|_2$.

Combining both inequalities: $|\sigma_i(A+E) - \sigma_i(A)| \leq \|E\|_2$. $\square$

### A.5 Proof that the Nuclear Norm is Dual to the Spectral Norm

**Claim.** $\|A\|_* = \max_{\|B\|_2 \leq 1} \operatorname{tr}(A^\top B)$.

**Achieving the bound.** Let $A = U\Sigma V^\top$ and set $B = UV^\top$. Then $\|B\|_2 = 1$ (product of unitary-like matrices with unit singular values) and:
$$\operatorname{tr}(A^\top B) = \operatorname{tr}(V\Sigma U^\top \cdot UV^\top) = \operatorname{tr}(V\Sigma V^\top) = \operatorname{tr}(\Sigma) = \sum_i \sigma_i = \|A\|_*$$

**Upper bound for any $\|B\|_2 \leq 1$.** By the Von Neumann trace inequality:
$$|\operatorname{tr}(A^\top B)| \leq \sum_i \sigma_i(A)\sigma_i(B) \leq \sum_i \sigma_i(A) \cdot 1 = \|A\|_*$$

(since $\sigma_i(B) \leq \|B\|_2 \leq 1$). Therefore $\max_{\|B\|_2\leq 1}\operatorname{tr}(A^\top B) = \|A\|_*$. $\square$

---

## Appendix B: Advanced Topics

### B.1 The Von Neumann Trace Inequality

A fundamental result connecting norms to the trace:

**Theorem (Von Neumann, 1937).** For $A, B \in \mathbb{R}^{m \times n}$:
$$|\operatorname{tr}(A^\top B)| \leq \sum_{i=1}^{\min(m,n)} \sigma_i(A)\sigma_i(B)$$

with equality when $A$ and $B$ share the same left and right singular vectors (i.e., $A$ and $B$ are simultaneously diagonalizable in the SVD sense).

**Consequences:**

1. Nuclear-spectral Hölder: $|\langle A,B\rangle_F| \leq \|A\|_*\|B\|_2$
2. Cauchy-Schwarz (Frobenius): $|\langle A,B\rangle_F| \leq \|A\|_F\|B\|_F$ (using $\sigma_i(A)\sigma_i(B) \leq (\sigma_i(A)^2+\sigma_i(B)^2)/2$)
3. Duality: $\|A\|_* = \max_{\|B\|_2\leq 1}\operatorname{tr}(A^\top B)$ (proved in §A.5 using this inequality)

**For AI:** This inequality bounds approximation errors in attention: $|\operatorname{tr}(\hat{S}^\top V^\top V)| \leq \|\hat{S}\|_* \|V\|_2^2$, connecting attention approximation quality to the nuclear norm of the approximate attention matrix.

### B.2 Pseudo-Spectrum and Non-Normal Matrices

For a non-normal matrix $A$ ($AA^\top \neq A^\top A$), eigenvalues can be extremely sensitive to perturbation even when the matrix has bounded spectral norm. The **$\epsilon$-pseudo-spectrum** reveals this sensitivity:

$$\Lambda_\epsilon(A) = \{\lambda \in \mathbb{C} : \sigma_{\min}(A - \lambda I) \leq \epsilon\} = \{z : \|(A - zI)^{-1}\|_2 \geq \epsilon^{-1}\}$$

This is the set of complex numbers that are eigenvalues of some perturbation $A + E$ with $\|E\|_2 \leq \epsilon$.

**Normal vs. non-normal.** For normal matrices ($A^\top A = AA^\top$), $\Lambda_\epsilon(A) = \bigcup_j D(\lambda_j, \epsilon)$ — just disks of radius $\epsilon$ around each eigenvalue. For non-normal matrices, the pseudo-spectrum can extend far beyond these disks.

**Example: Jordan block.** $A = \begin{pmatrix}0&1\\0&0\end{pmatrix}$ has eigenvalue $0$ with multiplicity 2. Yet $\Lambda_\epsilon(A) \supset \{|z| < \sqrt{\epsilon}\}$ — a disk of radius $\sqrt{\epsilon}$, much larger than $\epsilon$ for small $\epsilon$.

**For AI:** Non-normal recurrent weight matrices $W_{hh}$ in RNNs can transiently amplify signals even when $\|W_{hh}\|_2 < 1$. The pseudo-spectrum shows why the simple criterion "$\|W_{hh}\|_2 < 1$ implies stability" is insufficient — transient growth can lead to effective instability in finite-depth computations (finite unrolling of time steps).

### B.3 Stable Rank and Generalization

The **stable rank** of $A$ is:
$$\rho(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2} = \sum_i \left(\frac{\sigma_i}{\sigma_1}\right)^2$$

**Properties:**

- $1 \leq \rho(A) \leq \operatorname{rank}(A)$
- $\rho(A) = 1$ iff $A$ is rank-1
- $\rho(A) = r$ iff $A$ has $r$ equal nonzero singular values (flat singular value spectrum)
- $\rho(A)$ is continuous in $A$ (unlike rank, which is discontinuous)

**Generalization bound.** For a linear predictor $\mathbf{f}(\mathbf{x}) = W\mathbf{x}$ learned from $n$ samples:
$$\text{Generalization gap} \leq O\left(\sqrt{\frac{\rho(W)\|W\|_2^2}{n}}\right) = O\left(\sqrt{\frac{\|W\|_F^2}{n}}\right)$$

This shows that the Frobenius norm (not rank) controls generalization for linear models — justifying Frobenius regularization (weight decay).

**For deep networks.** Bartlett et al. (2017) proved:
$$\text{Generalization gap} \leq O\left(\frac{\prod_l \|W_l\|_2 \cdot \sqrt{\sum_l \rho(W_l)}}{\gamma\sqrt{n}}\right)$$

where $\gamma$ is the margin. The stable rank $\rho(W_l)$ of each layer's weight matrix appears, not its true rank.

### B.4 Matrix Norms in Optimization: Proximal Methods

Many regularized optimization problems of the form:
$$\min_W \mathcal{L}(W) + \lambda R(W)$$

can be solved with **proximal gradient descent**:
$$W^{(t+1)} = \operatorname{prox}_{\alpha\lambda R}(W^{(t)} - \alpha \nabla \mathcal{L}(W^{(t)}))$$

The proximal operator $\operatorname{prox}_{\tau R}(V) = \arg\min_W \frac{1}{2}\|W-V\|_F^2 + \tau R(W)$ depends on the choice of $R$:

| Regularizer $R(W)$ | Proximal Operator | Effect |
| --- | --- | --- |
| $\|W\|_F^2$ (L2) | $W/(1+2\tau)$ | Scale all entries uniformly |
| $\|W\|_*$ (nuclear) | SVT: soft-threshold singular values | Zero small singular values |
| $\|W\|_2$ (spectral) | Cap all $\sigma_i$ at $\tau$; project onto spectral ball | Bound largest singular value |
| $\|W\|_1$ (entry) | $\operatorname{sign}(V_{ij})\max(\lvert V_{ij}\rvert-\tau,0)$ | Soft-threshold each entry |

**For AI:** The proximal operator for nuclear norm regularization is singular value soft-thresholding (SVT), the key subroutine in matrix completion algorithms (Cai-Candès-Shen, 2010). Modern deep learning rarely uses explicit nuclear norm regularization because the factored parameterization in LoRA provides implicit nuclear norm regularization via the dynamics of gradient descent on $\min_{B,A}\mathcal{L}(W_0+BA)$.

### B.5 Norm Balls and Geometry

The geometry of norm balls illuminates the differences between norms.

```
UNIT BALLS IN 2D (for matrix norms, visualized on singular value vectors)
════════════════════════════════════════════════════════════════════════

  Spectral (ℓ∞ on σ)    Frobenius (ℓ² on σ)   Nuclear (ℓ¹ on σ)

    σ₂                     σ₂                     σ₂
    │                      │                      │
  1─┼───────┐            1─┼    ╭───╮           1─┤
    │       │              │   ╱   ╲              │╲
    │       │              │  │     │             │  ╲
    │       │              │  │     │             │    ╲
  ──┼───────┼──σ₁       ──┼──╰─────╯──σ₁      ──┼─────┼──σ₁
    │       1              │        1             │     1
    │                      │                      │
    └ square               └ circle               └ diamond

  All σ ≤ 1               σ₁²+σ₂² ≤ 1          σ₁+σ₂ ≤ 1

════════════════════════════════════════════════════════════════════════
```

**The nuclear norm ball is a polytope** in singular value space (it's the $\ell^1$ ball), making nuclear norm minimization a linear program in singular value space — hence solvable efficiently.

**The spectral norm ball is a hypercube** in singular value space ($\ell^\infty$ ball), with flat faces. This geometry means that spectral norm projection (projecting onto $\|W\|_2 \leq 1$) simply caps all singular values at 1.

**For AI:** The different geometries explain why nuclear norm regularization promotes low-rank solutions (the corners of the $\ell^1$ diamond are at axis-aligned points = rank-1 matrices) while Frobenius regularization does not (the smooth $\ell^2$ ball has no corners to attract the solution toward).

---

## Appendix C: Computational Reference

### C.1 NumPy/SciPy Norm API

```python
import numpy as np
import scipy.linalg as la

A = np.random.randn(4, 5)   # example matrix

# Frobenius norm
np.linalg.norm(A, 'fro')           # direct
np.sqrt(np.sum(A**2))              # elementwise
np.sqrt(np.trace(A.T @ A))        # trace formula

# Spectral norm = sigma_1
np.linalg.norm(A, 2)               # direct (uses SVD internally)
la.svdvals(A)[0]                   # explicit first singular value

# Nuclear norm = sum of singular values
np.linalg.norm(A, 'nuc')           # direct
np.sum(la.svdvals(A))              # explicit sum

# Matrix 1-norm = max absolute column sum
np.linalg.norm(A, 1)               # induced 1-norm

# Matrix infinity-norm = max absolute row sum
np.linalg.norm(A, np.inf)          # induced inf-norm

# Condition number
np.linalg.cond(A)                  # spectral: sigma_1 / sigma_n
np.linalg.cond(A, 'fro')          # Frobenius-based
np.linalg.cond(A, 1)              # 1-norm condition number
```

### C.2 Power Iteration for Spectral Norm

```python
def spectral_norm_power_iter(A, n_iter=20, seed=42):
    """Estimate ||A||_2 = sigma_1(A) via power iteration."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        u = A @ v;  u /= np.linalg.norm(u)
        v = A.T @ u
        sigma = np.linalg.norm(v)
        v /= sigma

    return sigma   # converges at rate (sigma_2 / sigma_1)^(2*n_iter)
```

This is used in spectral normalization: one step per training iteration is enough (the estimate from the previous step is a warm start).

### C.3 Singular Value Soft-Thresholding

```python
def svt(A, threshold):
    """Proximal operator of threshold * ||.||_* applied to A."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_thresh = np.maximum(s - threshold, 0)
    return U @ np.diag(s_thresh) @ Vt

# Returns argmin_X  (1/2)||X - A||_F^2 + threshold * ||X||_*
# Singular values below threshold are zeroed → rank reduction
```

### C.4 Stable Rank

```python
def stable_rank(A):
    """Stable rank rho(A) = ||A||_F^2 / ||A||_2^2."""
    s = np.linalg.svd(A, compute_uv=False)
    return np.sum(s**2) / s[0]**2

# Always in [1, rank(A)]
# = 1 iff rank-1; = rank(A) iff flat singular value spectrum
```

### C.5 Tikhonov Solution via SVD

```python
def tikhonov_svd(A, b, lam):
    """Tikhonov-regularized least squares: min ||Ax-b||^2 + lam||x||^2.
    Uses SVD for numerical stability even when A is near-singular."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    d = s / (s**2 + lam)     # modified singular value filter
    return Vt.T @ (d * (U.T @ b))

# Condition number of regularized system: (s[0]^2 + lam) / (s[-1]^2 + lam)
```

---

## Appendix D: Extended Examples and Worked Problems

### D.1 A Complete Norm Taxonomy on One Matrix

Let $A = \begin{pmatrix} 4 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}$ (diagonal with singular values $\sigma_1=4, \sigma_2=2, \sigma_3=1$).

**All norms in one place:**

| Norm | Formula | Value | Computation |
| --- | --- | --- | --- |
| Frobenius $\|A\|_F$ | $\sqrt{\sum\sigma_i^2}$ | $\sqrt{21} \approx 4.58$ | $\sqrt{16+4+1}$ |
| Spectral $\|A\|_2$ | $\sigma_1$ | $4$ | Largest singular value |
| Nuclear $\|A\|_*$ | $\sum\sigma_i$ | $7$ | $4+2+1$ |
| Schatten $S_3$ | $(\sum\sigma_i^3)^{1/3}$ | $(64+8+1)^{1/3} \approx 4.17$ | $73^{1/3}$ |
| Matrix 1-norm $\|A\|_1$ | max col sum | $4$ | Max of $\{4,2,1\}$ |
| Matrix $\infty$-norm | max row sum | $4$ | Max of $\{4,2,1\}$ |
| Condition number $\kappa_2$ | $\sigma_1/\sigma_3$ | $4$ | $4/1$ |
| Stable rank $\rho$ | $\|A\|_F^2/\|A\|_2^2$ | $21/16 \approx 1.31$ | $21/16$ |

**Ordering verification:** $\|A\|_2 = 4 \leq \|A\|_F \approx 4.58 \leq \|A\|_* = 7$ ✓

For rank-$r$ bounds: $\|A\|_F \leq \sqrt{3}\|A\|_2$ gives $4.58 \leq 6.93$ ✓ and $\|A\|_* \leq \sqrt{3}\|A\|_F$ gives $7 \leq 7.94$ ✓.

### D.2 Worked Example: Spectral Norm via Power Iteration

Let $A = \begin{pmatrix} 3 & 1 \\ 2 & 4 \end{pmatrix}$.

**Exact:** The singular values satisfy $\sigma^2 = \text{eigenvalues of } A^\top A = \begin{pmatrix}13&11\\11&17\end{pmatrix}$.

$\text{tr}(A^\top A) = 30$, $\det(A^\top A) = 13\cdot17 - 121 = 221 - 121 = 100$.

Eigenvalues: $\lambda = (30 \pm \sqrt{900 - 400})/2 = (30 \pm \sqrt{500})/2 = 15 \pm 5\sqrt{5}$.

So $\sigma_1 = \sqrt{15 + 5\sqrt{5}} \approx \sqrt{26.18} \approx 5.12$ and $\|A\|_2 \approx 5.12$.

**Power iteration** (starting with $\mathbf{v}_0 = (1,0)^\top$):

| Iter | $A\mathbf{v}$ | $\mathbf{u}$ | $A^\top\mathbf{u}$ | $\mathbf{v}$ | $\sigma$ estimate |
| --- | --- | --- | --- | --- | --- |
| 0 | — | — | — | $(1,0)^\top$ | — |
| 1 | $(3,2)^\top$ | $(0.832,0.555)^\top$ | $(3.607,4.052)^\top$ | $(0.664,0.747)^\top$ | $5.43$ |
| 2 | $(2.74,4.29)^\top$ | $(0.538,0.843)^\top$ | $(3.460,4.110)^\top$ | $(0.644,0.765)^\top$ | $5.37$ |
| 3 | $(2.70,4.35)^\top$ | $(0.529,0.849)^\top$ | $(3.437,4.124)^\top$ | $(0.641,0.768)^\top$ | $5.37$ |

Converging to $\sigma_1 \approx 5.12$ (the remaining gap is because we stopped early).

**For AI:** This is exactly how PyTorch implements `torch.nn.utils.spectral_norm` — one step of power iteration per training step, with the previous estimate $\mathbf{v}$ stored as a buffer.

### D.3 Worked Example: Condition Number and Error Amplification

Consider the **Hilbert matrix** $H_n$ with $(H_n)_{ij} = 1/(i+j-1)$ — a canonical example of an ill-conditioned matrix.

For $n = 5$:
$$H_5 = \begin{pmatrix} 1 & 1/2 & 1/3 & 1/4 & 1/5 \\ 1/2 & 1/3 & 1/4 & 1/5 & 1/6 \\ 1/3 & 1/4 & 1/5 & 1/6 & 1/7 \\ 1/4 & 1/5 & 1/6 & 1/7 & 1/8 \\ 1/5 & 1/6 & 1/7 & 1/8 & 1/9 \end{pmatrix}$$

The condition number grows as $\kappa_2(H_n) \approx e^{3.5n}$:

- $\kappa_2(H_5) \approx 4.77 \times 10^5$
- $\kappa_2(H_{10}) \approx 1.60 \times 10^{13}$
- $\kappa_2(H_{12}) \approx 1.83 \times 10^{16}$ (at the limit of double precision!)

**Error analysis.** Solve $H_5 \mathbf{x} = \mathbf{b}$ where $\mathbf{b} = H_5 \mathbf{1}$ (exact solution $\mathbf{x}_0 = (1,1,1,1,1)^\top$). With $\mathbf{b}$ perturbed by machine epsilon $\epsilon \approx 10^{-16}$:

Relative forward error $\leq \kappa(H_5) \cdot 10^{-16} \approx 4.77 \times 10^5 \cdot 10^{-16} \approx 5 \times 10^{-11}$

So we lose about 5 decimal places of precision. For $H_{10}$, we'd lose 13 places — near total loss of information.

**Tikhonov fix:** With $\lambda = 10^{-5}$, $\kappa(H_5 + \lambda I) \approx 8 \times 10^4$ — a 6× improvement at the cost of introducing systematic bias $\approx \lambda\|\mathbf{x}\|$.

### D.4 The Spectral Norm of the Attention Matrix

In a transformer with sequence length $T = 512$, dimension $d_k = 64$, and queries and keys drawn from a standard Gaussian:

**Expected spectral norm.** For $Q, K \in \mathbb{R}^{T \times d_k}$ with i.i.d. $\mathcal{N}(0, 1/d_k)$ entries:
$$\mathbb{E}[\|QK^\top\|_2] \approx \frac{T + d_k}{d_k/T + 1} \cdot \frac{\sigma^2}{d_k}$$

More concretely, by random matrix theory (Marchenko-Pastur law), $\|QK^\top\|_2 \approx 2\sqrt{Td_k}/d_k \cdot \sqrt{d_k} = 2\sqrt{T}$.

The factor $1/\sqrt{d_k}$ in attention ($\text{softmax}(QK^\top/\sqrt{d_k})$) is a **spectral normalization**: it scales the attention logits so that $\|QK^\top/\sqrt{d_k}\|_2 \approx 2\sqrt{T/d_k}$ — preventing the softmax from collapsing to one-hot distributions when $T \gg d_k$.

**Nuclear norm and attention rank.** In trained models, attention matrices exhibit low stable rank. Dong et al. (2021) measured stable rank $\rho(A) = \|A\|_F^2/\|A\|_2^2$ averaging 4-8 in BERT's attention heads (out of possible 512), suggesting the effective attention rank is much lower than the sequence length.

### D.5 LoRA Norm Analysis

LoRA fine-tuning adds $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, typically $r \in \{4, 8, 16, 32\}$ vs. $d \in \{768, 1024, 4096\}$.

**Norm bounds on $\Delta W$:**

- $\|\Delta W\|_F = \|BA\|_F \leq \|B\|_F\|A\|_F$ (submultiplicativity)
- $\|\Delta W\|_2 = \|BA\|_2 \leq \|B\|_2\|A\|_2$
- $\|\Delta W\|_* = \|BA\|_* \leq \min(\|B\|_*\|A\|_2, \|B\|_2\|A\|_*)$

**Implicit nuclear norm regularization.** The factored parameterization $\Delta W = BA$ has at most $r$ nonzero singular values (since $\text{rank}(BA) \leq r$). The nuclear norm $\|BA\|_* = \sum_{i=1}^r \sigma_i(BA)$ is automatically bounded by:
$$\|BA\|_* \leq \sqrt{r}\|BA\|_F \leq \sqrt{r}\|B\|_F\|A\|_F$$

During optimization, gradient descent on $\min_{B,A}\mathcal{L}(W_0+BA)$ implicitly minimizes $\|BA\|_*$ — this is the Gunasekar et al. (2017) implicit regularization result.

**DoRA (weight decomposition).** DoRA (Liu et al., 2024) reparameterizes as $W = m\cdot(V+BA)/\|V+BA\|_{col}$ where $m$ controls the magnitude and $V+BA/\|...\|$ controls the direction. The column-wise normalization is related to the nuclear norm: $\|W\|_*$ is invariant to column-wise rescaling.

### D.6 Gradient Flow and Norm Dynamics

During training, norms of weight matrices evolve. Understanding this evolution helps diagnose training pathologies.

**Stable training.** For a well-trained transformer layer with weight matrix $W \in \mathbb{R}^{4096 \times 4096}$:

- $\|W\|_2 \approx 1-5$ (spectral norm; bounded by weight decay + gradient clipping)
- $\|W\|_F \approx 30-100$ (Frobenius norm; $\approx \sqrt{d}\|W\|_2$ for flat spectrum)
- $\kappa(W) \approx 10^2-10^4$ (condition number; not too large for well-initialized models)
- $\rho(W) \approx 100-500$ (stable rank; much less than 4096)

**Warning signs:**

- $\|W\|_2$ growing rapidly → potential gradient explosion; increase weight decay
- $\|W\|_2 \to 0$ → vanishing gradients or overly aggressive weight decay
- $\kappa(W) \to 10^6$ → poorly conditioned optimization; consider preconditioning
- $\rho(W) \to 1$ → the layer is becoming rank-1 (mode collapse); increase model capacity

### D.7 Norm-Based Pruning

Neural network pruning can be guided by matrix norms:

**Magnitude pruning (Frobenius).** Remove weight matrices (or rows/columns) with smallest Frobenius norm. The remaining error $\|W_{\text{pruned}} - W\|_F$ equals the Frobenius norm of the pruned components.

**Spectral pruning.** Prune singular value components below a threshold $\tau$: $\hat{W} = \sum_{\sigma_i > \tau} \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$. Error: $\|W - \hat{W}\|_2 = \sigma_{k+1}$ (spectral) and $\|W-\hat{W}\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$ (Frobenius).

**Nuclear norm pruning.** Regularize the fine-tuned model with $\lambda\|\Delta W\|_*$ during training, encouraging the weight update to use fewer singular value "directions." After training, prune near-zero singular values.

**Comparison:**

| Method | Error controlled | Norm minimized | Computational cost |
| --- | --- | --- | --- |
| Magnitude pruning | $\|W_{\text{pruned}}\|_F$ | None explicitly | $O(1)$ |
| SVD truncation | $\|W-\hat{W}\|_2$ and $\|W-\hat{W}\|_F$ | None (deterministic) | $O(mn\min(m,n))$ |
| Nuclear norm reg. | $\|\hat{W}\|_*$ | Nuclear norm | $O(mn\min(m,n))$ per step |
| Spectral norm SN | $\|W\|_2$ | Spectral norm | $O(k)$ power iter per step |

---

*← [Back to Advanced Linear Algebra](../README.md) | [Next: Linear Transformations →](../07-Linear-Transformations/notes.md)*

## Appendix E: Connections to Other Fields

### E.1 Matrix Norms in Statistics

Matrix norms appear throughout multivariate statistics.

**Covariance matrices.** For a sample covariance matrix $\hat{\Sigma} = \frac{1}{n}X^\top X$ (where $X \in \mathbb{R}^{n \times p}$):
- $\|\hat{\Sigma}\|_2 = \lambda_{\max}(\hat{\Sigma})$: the variance in the direction of maximum spread (principal component 1)
- $\|\hat{\Sigma}\|_*= \operatorname{tr}(\hat{\Sigma}) = \sum_j \hat{\sigma}_j^2$: total variance
- $\kappa(\hat{\Sigma}) = \lambda_{\max}/\lambda_{\min}$: measures how "spherical" the distribution is

**Condition number and regression.** For linear regression $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}$:
$$\kappa(X^\top X) = \frac{\lambda_{\max}(X^\top X)}{\lambda_{\min}(X^\top X)} = \frac{\sigma_1(X)^2}{\sigma_p(X)^2} = \kappa(X)^2$$

Multicollinearity in $X$ gives large $\kappa(X^\top X)$, making OLS estimates numerically unstable. Ridge regression adds $\lambda I$ to improve conditioning: $\kappa((X^\top X + \lambda I)) = (\sigma_1^2 + \lambda)/(\sigma_p^2 + \lambda)$.

**PCA error bounds.** Truncating PCA at $k$ components introduces Frobenius error $\sqrt{\sum_{i>k}\lambda_i}$ where $\lambda_i$ are eigenvalues of $\hat{\Sigma}$. This is exactly the Eckart-Young theorem applied to the covariance matrix.

**Sample covariance concentration.** For $n$ samples from $\mathcal{N}(0, \Sigma)$:
$$\mathbb{E}\|\hat{\Sigma} - \Sigma\|_2 \lesssim \|\Sigma\|_2 \sqrt{\frac{p}{n}} \quad (p \leq n)$$

The spectral norm of the estimation error scales as $\|\Sigma\|_2\sqrt{p/n}$ — one needs $n \gg p$ samples to estimate $\Sigma$ well in spectral norm.

### E.2 Matrix Norms in Control Theory

Control systems use matrix norms to quantify system gain and stability margins.

**System norms.** For a linear system $\dot{\mathbf{x}} = A\mathbf{x} + B\mathbf{u}$, $\mathbf{y} = C\mathbf{x}$:

- **$H_\infty$ norm**: $\|G\|_\infty = \sup_\omega \sigma_{\max}(G(i\omega))$ — worst-case gain over all frequencies; the spectral norm of the frequency response
- **$H_2$ norm**: $\|G\|_2 = \sqrt{\operatorname{tr}(B^\top P B)}$ where $P$ solves a Lyapunov equation; related to Frobenius norm in the frequency domain

**Stability.** A discrete-time system $\mathbf{x}_{t+1} = A\mathbf{x}_t$ is stable iff $\rho(A) < 1$ (spectral radius < 1). But for finite-time analysis, $\|A^t\|_2 \leq \|A\|_2^t$ — the spectral norm controls the growth of powers.

**For AI:** Recurrent networks are discrete-time dynamical systems with $\mathbf{h}_{t+1} = \sigma(W_{hh}\mathbf{h}_t + W_{xh}\mathbf{x}_t)$. The condition $\|W_{hh}\|_2 < 1$ (spectral normalization) guarantees contractivity: gradients decay as $\|W_{hh}\|_2^t$ through time — preventing explosion but also potentially causing vanishing gradients in long sequences.

### E.3 Matrix Norms in Quantum Information

Quantum states are represented by density matrices $\rho$ (PSD, $\operatorname{tr}(\rho) = 1$). Operations are described by quantum channels. Matrix norms quantify distances between quantum states and operations.

**Trace norm distance.** The trace distance between quantum states $\rho, \sigma$:
$$D(\rho, \sigma) = \frac{1}{2}\|\rho - \sigma\|_* = \frac{1}{2}\sum_i |\lambda_i(\rho - \sigma)|$$

(For Hermitian matrices, the nuclear norm equals the sum of absolute eigenvalues, not singular values.)

**Diamond norm.** For quantum channels $\mathcal{E}, \mathcal{F}$:
$$\|\mathcal{E} - \mathcal{F}\|_\diamond = \max_\rho \|(\mathcal{E} \otimes I)(\rho) - (\mathcal{F} \otimes I)(\rho)\|_*$$

This is the quantum analog of the $2 \to 1$ norm — measuring the maximum distinguishability of the channels.

**For AI:** Large language models in principle implement quantum-inspired computations when trained on quantum data. More concretely, tensor network approximations of quantum states use matrix norms (Frobenius) to bound truncation error in tensor decompositions — the same mathematics as SVD-based model compression.

### E.4 Matrix Norms in Graph Theory

For a graph $G$ on $n$ vertices with adjacency matrix $A \in \{0,1\}^{n \times n}$:

**Spectral radius.** $\|A\|_2 = \lambda_{\max}(A)$ (for symmetric $A$, since $\|A\|_2 = \rho(A)$ for symmetric PSD).

**Graph conductance and mixing.** The second eigenvalue $\lambda_2$ of the normalized Laplacian controls the mixing time of random walks. Small $\lambda_2$ means slow mixing (large $\kappa$).

**Graph neural networks.** A GNN layer applies $H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)})$ where $\hat{A}$ is the normalized adjacency. The spectral norm $\|\hat{A}\|_2 \leq 1$ (by design of normalization) ensures the graph propagation is non-expansive — a Lipschitz constraint built into the architecture.

**Over-smoothing and rank collapse.** After many GNN layers, the representations $H^{(L)}$ converge to a low-rank matrix. The nuclear norm of $H^{(L)}$ decreases, reflecting that all nodes' features converge to the same value (weighted by PageRank). This is the GNN analog of the attention rank collapse phenomenon.

---

## Appendix F: Historical Development

The theory of matrix norms developed in parallel with the needs of numerical analysis and functional analysis.

```
HISTORICAL TIMELINE: MATRIX NORMS
════════════════════════════════════════════════════════════════════════

  1844  Cauchy introduces the "module" of a complex matrix — early norm idea

  1878  Frobenius studies bilinear forms; Frobenius norm implicit in his work

  1905  Hilbert introduces "Hilbert space" — inner product → norm framework

  1929  Von Neumann: abstract operator theory in Hilbert space;
        nuclear operators (trace class) defined

  1937  Von Neumann trace inequality proved; singular values for operators

  1939  Ky Fan: Ky Fan k-norms, matrix inequalities

  1946  Eckart & Young: best low-rank approximation theorem (SVD)

  1950s Development of "matrix analysis" as a field (Bauer, Fike, etc.)
        Bauer-Fike theorem (1960): eigenvalue perturbation bound

  1960s Gershgorin circle theorem; norm-based stability conditions
        for numerical ODE solvers and control systems

  1970s  Numerical linear algebra matures (Golub, Van Loan)
        Condition number becomes central concept in floating-point analysis

  1988  Grothendieck's "résumé" translated; connections to operator spaces

  1990s  Nuclear norm in matrix completion; compressed sensing foundations

  2004  Candès & Tao: compressed sensing; nuclear norm as convex
        proxy for rank in matrix recovery

  2010  Cai, Candès, Shen: singular value thresholding algorithm (SVT)
        for nuclear norm minimization

  2018  Miyato et al.: Spectral Normalization for GANs
        (ICLR 2018 paper)

  2021  Dong et al.: attention rank collapse analysis via matrix norms

  2021  Hu et al.: LoRA — implicit nuclear norm regularization via
        low-rank factored parameterization

  2024  DeepSeek MLA: explicit low-rank KV cache via nuclear norm
        motivated matrix compression

════════════════════════════════════════════════════════════════════════
```

**Key insight from the historical development.** Matrix norms began as abstract tools in functional analysis (Von Neumann, Ky Fan) and numerical analysis (Frobenius, Bauer, Fike). Their entry into machine learning came through three channels:
1. **Optimization theory** (nuclear norm = convex proxy for rank, 2004-2010)
2. **Generalization theory** (Frobenius and spectral norms in PAC-Bayes bounds, 2017)
3. **Architecture design** (spectral normalization, LoRA, MLA, 2018-2024)

The mathematical tools developed over 80 years now routinely appear in the training pipelines of the largest language models.

---

## Appendix G: Summary Tables

### G.1 Complete Norm Reference Table

| Norm | Symbol | Formula | SVD formula | Computation | Notes |
| --- | --- | --- | --- | --- | --- |
| Frobenius | $\|A\|_F$ | $\sqrt{\sum a_{ij}^2}$ | $\|\boldsymbol{\sigma}\|_2$ | `norm(A,'fro')` | Euclidean on entries; not induced |
| Spectral | $\|A\|_2$ | $\max_{\|x\|=1}\|Ax\|$ | $\sigma_1$ | `norm(A,2)` | Largest singular value; induced |
| Nuclear | $\|A\|_*$ | $\operatorname{tr}(\sqrt{A^\top A})$ | $\sum\sigma_i$ | `norm(A,'nuc')` | Dual of spectral; convex rank proxy |
| Matrix-1 | $\|A\|_1$ | $\max_j\sum_i\|a_{ij}\|$ | — | `norm(A,1)` | Max col sum; induced |
| Matrix-∞ | $\|A\|_\infty$ | $\max_i\sum_j\|a_{ij}\|$ | — | `norm(A,inf)` | Max row sum; induced |
| Schatten-$p$ | $\|A\|_{S_p}$ | $(\sum\sigma_i^p)^{1/p}$ | $\|\boldsymbol{\sigma}\|_p$ | Custom (use SVD) | Unifies F, spec, nuclear |
| Ky Fan-$k$ | $\|A\|_{(k)}$ | $\sigma_1+\cdots+\sigma_k$ | $\|\boldsymbol{\sigma}_{1:k}\|_1$ | `sum(svdvals[:k])` | Sum of top-$k$ singular values |
| Max entry | $\|A\|_{\max}$ | $\max_{ij}\|a_{ij}\|$ | — | `np.max(abs(A))` | NOT submultiplicative |

### G.2 Norm Inequality Summary

For $A \in \mathbb{R}^{m \times n}$ with rank $r$ and singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$:

$$\|A\|_2 \leq \|A\|_F \leq \sqrt{r}\|A\|_2$$

$$\|A\|_F \leq \|A\|_* \leq \sqrt{r}\|A\|_F$$

$$\|A\|_2 \leq \|A\|_* \leq r\|A\|_2$$

$$\frac{1}{\sqrt{n}}\|A\|_1 \leq \|A\|_2 \leq \sqrt{m}\|A\|_1$$

$$\frac{1}{\sqrt{m}}\|A\|_\infty \leq \|A\|_2 \leq \sqrt{n}\|A\|_\infty$$

$$\|A\|_{\max} \leq \|A\|_2 \leq \sqrt{mn}\|A\|_{\max}$$

All inequalities are tight (achieved by appropriate matrices).

### G.3 Which Norm to Use When

| Goal | Recommended norm | Reason |
| --- | --- | --- |
| Measure approximation error $\|A-B\|$ | Frobenius | Euclidean geometry; easy to compute and differentiate |
| Bound amplification of a linear map | Spectral | Exactly measures worst-case gain |
| Promote low-rank solutions | Nuclear | Convex proxy for rank; SVT proximal operator |
| Measure GAN discriminator Lipschitz | Spectral | Product of spectral norms bounds network Lipschitz |
| Analyze numerical stability | Condition number $\kappa = \|A\|_2\|A^{-1}\|_2$ | Quantifies sensitivity to perturbations |
| Regularize neural network weights | Frobenius ($\|W\|_F^2$) | Gradient is $2W$; easy to implement; weight decay |
| Compress weight matrices | Spectral + Frobenius (Eckart-Young) | Optimal low-rank approximation uses SVD |
| Certify network robustness | Spectral norm per layer | Bound on overall Lipschitz constant |
| Matrix completion / recommendation | Nuclear | Nuclear norm minimization recovers under incoherence |
| Quantum state discrimination | Trace norm (nuclear for Hermitian) | Operational meaning in quantum measurement |

### G.4 Condition Number Thresholds (Practical Guide)

| $\kappa_2(A)$ | Precision loss | Status | Action |
| --- | --- | --- | --- |
| $1$ | None | Perfectly conditioned | — |
| $< 10^3$ | $< 3$ digits | Well conditioned | Standard algorithms |
| $10^3$–$10^6$ | 3-6 digits | Moderately ill-conditioned | Watch for rounding |
| $10^6$–$10^{10}$ | 6-10 digits | Ill-conditioned | Use Tikhonov; double-check |
| $10^{10}$–$10^{14}$ | 10-14 digits | Severely ill-conditioned | Regularize aggressively |
| $> 10^{14}$ | > 14 digits | Near-singular | Essentially singular in double precision |
| $\infty$ | All digits | Singular | Problem is underdetermined |

*In double precision arithmetic ($\epsilon_{\text{mach}} \approx 10^{-16}$), effective precision = $16 - \log_{10}\kappa$ digits.*

---

## Appendix H: Deep Dives

### H.1 The Spectral Norm in Full Detail

The spectral norm $\|A\|_2 = \sigma_1(A)$ deserves extended treatment because it appears in so many different contexts.

**Variational characterizations.** All of the following equal $\sigma_1(A)$:

$$\sigma_1(A) = \max_{\|\mathbf{x}\|_2 = 1}\|A\mathbf{x}\|_2 \qquad \text{(definition)}$$

$$= \max_{\|\mathbf{x}\|_2 = \|\mathbf{y}\|_2 = 1} \mathbf{y}^\top A\mathbf{x} \qquad \text{(bilinear form)}$$

$$= \sqrt{\lambda_{\max}(A^\top A)} \qquad \text{(eigenvalue of Gram matrix)}$$

$$= \sqrt{\lambda_{\max}(AA^\top)} \qquad \text{(left Gram matrix)}$$

$$= \max_{\|B\|_* \leq 1}\langle A, B\rangle_F \qquad \text{(dual norm formula)}$$

Each characterization reveals a different aspect:
- The first shows it as maximum stretching
- The second shows it as maximum inner product over the unit sphere (bilinear)
- The third and fourth show it as square root of a largest eigenvalue
- The fifth shows it as dual of nuclear norm

**Computing $\sigma_1$ for structured matrices:**

For $A = \mathbf{u}\mathbf{v}^\top$ (rank-1): $\sigma_1 = \|\mathbf{u}\|\|\mathbf{v}\|$ (exact, no SVD needed).

For $A$ symmetric PSD: $\sigma_1 = \lambda_{\max}(A)$ (Lanczos iteration converges fast).

For $A$ sparse: Power iteration on $A^\top A$ with sparse matrix-vector products, $O(\text{nnz}(A) \cdot k)$ for $k$ iterations.

For $A$ a convolution matrix (CNN weights): The spectral norm equals the maximum frequency response $\|\hat{a}\|_\infty$ where $\hat{a}$ is the Fourier transform of the filter — computable in $O(n\log n)$ via FFT.

**For AI:** The "singular value of a convolution" insight means spectral normalization of CNN layers can be done in $O(n\log n)$ using FFT instead of $O(n^2)$ SVD. This is exploited in efficient implementations of spectral normalization for CNNs.

**Spectral norm of attention.** For scaled dot-product attention with keys $K \in \mathbb{R}^{T\times d_k}$, queries $Q \in \mathbb{R}^{T\times d_k}$, the attention logit matrix $S = QK^\top/\sqrt{d_k}$ satisfies:
$$\|S\|_2 \leq \|Q\|_2\|K\|_2/\sqrt{d_k}$$

In practice, if $Q, K$ are normalized (unit-norm rows), then $\|Q\|_2 \leq \sqrt{T}$ and $\|K\|_2 \leq \sqrt{T}$, giving $\|S\|_2 \leq T/\sqrt{d_k}$. The $1/\sqrt{d_k}$ factor prevents this from growing with $d_k$, maintaining bounded spectral norm for fixed $T$.

**Spectral norm through training.** In typical training of GPT-like models:
- At initialization (Xavier/Kaiming): $\|W\|_2 \approx \sqrt{2/d}$ (by design)
- After training without regularization: $\|W\|_2$ can grow by 10-100× as features strengthen
- With weight decay $\lambda$: equilibrium at $\|W\|_2 \approx \|\nabla_W\mathcal{L}\|_2/\lambda$ (gradient norm divided by decay rate)
- With spectral normalization: $\|W\|_2 = 1$ exactly (enforced each step)

### H.2 The Nuclear Norm in Full Detail

The nuclear norm $\|A\|_* = \sum_i\sigma_i$ has a rich structure worth examining closely.

**Characterizations.** All of the following equal $\|A\|_*$:

$$\|A\|_* = \operatorname{tr}(\sqrt{A^\top A}) = \operatorname{tr}(|A|)$$

$$= \min_{A = UV^\top} \frac{1}{2}(\|U\|_F^2 + \|V\|_F^2) \qquad \text{(factorization formula)}$$

$$= \max_{\|B\|_2 \leq 1}\operatorname{tr}(A^\top B) \qquad \text{(duality)}$$

$$= \min_{\substack{A = B + C \\ B\text{ rank-1}}} \|B\|_* + \|C\|_* \qquad \text{(triangle inequality applied optimally)}$$

**The factorization formula** $\|A\|_* = \min_{A=UV^\top}\frac{1}{2}(\|U\|_F^2+\|V\|_F^2)$ is particularly useful:

It shows that the nuclear norm equals half the squared Frobenius norm at the optimal factorization — the one where the singular values split evenly: $U = U_0\Sigma^{1/2}$ and $V = V_0\Sigma^{1/2}$ in the SVD $A = U_0\Sigma V_0^\top$.

**For AI:** This is exactly why LoRA works. The optimization $\min_{B,A}\mathcal{L}(W_0 + BA)$ with $B \in \mathbb{R}^{d\times r}$ and $A \in \mathbb{R}^{r\times d}$ implicitly minimizes $\|BA\|_* = \min_{\Delta W = BA}\|BA\|_*$ over rank-$r$ weight increments. The factored form is the "unfolded" version of nuclear norm minimization.

**Nuclear norm SDP representation.** The nuclear norm can be computed via semidefinite programming:
$$\|A\|_* = \min_{W_1, W_2} \frac{1}{2}(\operatorname{tr}(W_1) + \operatorname{tr}(W_2)) \quad\text{s.t.}\quad \begin{pmatrix}W_1 & A \\ A^\top & W_2\end{pmatrix} \succeq 0$$

This shows nuclear norm minimization is a semidefinite program (SDP) — solvable in polynomial time (in principle). For large-scale problems, proximal methods (using SVT) are much faster than interior-point SDP solvers.

**Subdifferential.** The subdifferential of $\|\cdot\|_*$ at $A$ with compact SVD $A = U_r\Sigma_r V_r^\top$ (only positive singular values) is:
$$\partial\|A\|_* = \{U_r V_r^\top + P_\perp W P_\perp : \|W\|_2 \leq 1\}$$

where $P_\perp = I - U_rU_r^\top$ projects onto the orthogonal complement of the column space. The unique subgradient when all singular values are distinct is $U_r V_r^\top$ — the "direction matrix."

### H.3 Condition Number and Optimization Landscape

The condition number of the Hessian $H = \nabla^2\mathcal{L}$ at a critical point determines the local convergence rate of gradient-based methods.

**Gradient descent convergence.** For $\mathcal{L}$ with $L$-smooth and $\mu$-strongly convex Hessian ($\mu I \preceq H \preceq LI$):
$$\mathcal{L}(\mathbf{x}^{(t)}) - \mathcal{L}^* \leq \left(1 - \frac{\mu}{L}\right)^t (\mathcal{L}(\mathbf{x}^{(0)}) - \mathcal{L}^*)$$

The convergence rate $(1-\mu/L) = 1 - 1/\kappa(H)$ depends directly on the condition number. For $\kappa(H) = 1000$, we need $\sim 1000$ steps to reduce error by $1/e$.

**Newton's method.** Newton steps use the Hessian as a preconditioner: $\mathbf{x}^{(t+1)} = \mathbf{x}^{(t)} - H^{-1}\nabla\mathcal{L}$. This achieves **quadratic convergence** near the optimum, independent of $\kappa(H)$.

**Adam as approximate preconditioning.** Adam maintains estimates of $\mathbb{E}[g_i^2]$ for each parameter. Dividing the gradient by $\sqrt{\mathbb{E}[g_i^2]}$ approximates dividing by $\sqrt{H_{ii}}$ — a diagonal preconditioning. This reduces the effective condition number:
$$\kappa_{\text{eff}} = \frac{\max_i H_{ii}/\mathbb{E}[g_i^2]^{1/2}}{\min_i H_{ii}/\mathbb{E}[g_i^2]^{1/2}} \ll \kappa(H)$$

when $H$ is diagonally dominant. For non-diagonal $H$, Adam's approximation is crude, which is why Shampoo (full matrix preconditioning) converges faster.

**Shampoo optimizer.** Shampoo (Gupta et al., 2018) computes full matrix preconditioners:
$$G_{t+1} = G_t + \text{grad}_t \text{grad}_t^\top, \quad \hat{G}_t = G_t^{-1/4}$$

The preconditioned gradient $\hat{G}_t \text{grad}_t \hat{G}_t^\top$ (for a 2D weight matrix) approximates the natural gradient, achieving condition number $\kappa_{\text{eff}} \approx 1$ in the limit. The cost is computing $G_t^{-1/4}$ (matrix $p$-th root) periodically.

### H.4 Norms in Transformer Architecture Design

Modern transformer architectures are designed with matrix norm considerations in mind.

**Layer normalization.** LayerNorm normalizes hidden representations $\mathbf{h}$ so that $\|\mathbf{h}\|_2 \approx \sqrt{d}$ (unit variance per coordinate). This bounds the spectral norm of the Jacobian of subsequent layers: if $\|\mathbf{h}\|_2 \leq \sqrt{d}$ and $\|W\|_F \leq C$, then $\|W\mathbf{h}\|_2 \leq \|W\|_2\|\mathbf{h}\|_2 \leq \|W\|_F\sqrt{d}$.

**Residual connections.** The residual architecture $\mathbf{h}_{l+1} = \mathbf{h}_l + F_l(\mathbf{h}_l)$ has Jacobian $I + J_{F_l}$. The spectral norm:
$$\|I + J_{F_l}\|_2 \leq 1 + \|J_{F_l}\|_2$$

This prevents extreme gradient flow: even if $\|J_{F_l}\|_2$ is large, the identity term stabilizes backward propagation. Empirically, $\|J_{F_l}\|_2 \approx 0.5-2$ in trained transformers.

**Pre-norm vs. post-norm.** In pre-norm transformers (LayerNorm before the sublayer), the effective Jacobian has better conditioning than post-norm. Specifically, pre-norm ensures $\|\mathbf{h}\|_2$ remains bounded before each attention/MLP, giving more stable Hessian conditioning.

**RoPE (Rotary Position Embeddings).** RoPE applies a position-dependent rotation to queries and keys:
$$\tilde{\mathbf{q}} = R_m \mathbf{q}, \quad \tilde{\mathbf{k}} = R_n \mathbf{k}$$

where $R_m$ is a rotation matrix depending on position $m$. Since $R_m$ is unitary ($\|R_m\|_2 = 1$), this preserves the spectral norm of the query/key vectors: $\|\tilde{\mathbf{q}}\|_2 = \|\mathbf{q}\|_2$. The inner product $\tilde{\mathbf{q}}^\top\tilde{\mathbf{k}} = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m}\mathbf{k}$ — only the relative position $n-m$ matters.

**MLA (Multi-Head Latent Attention).** DeepSeek's MLA compresses the KV cache from $2Td_{kv}$ to $Td_c$ (with $d_c \ll 2d_{kv}$) by decomposing:
$$K = C_{KV}W^K, \quad V = C_{KV}W^V$$

where $C_{KV} \in \mathbb{R}^{T\times d_c}$ is the "latent" KV. The approximation error (measured in nuclear norm) is bounded by the nuclear norm of the difference $\|KV^\top - \hat{K}\hat{V}^\top\|_*$ — exactly the type of bound from Eckart-Young applied to the KV product matrix.

---
## Appendix I: Practice Problems and Solutions

### I.1 Conceptual Checks

**Check 1.** Explain why $\|I_n\|_F = \sqrt{n}$ while $\|I_n\|_2 = 1$. What does this say about whether the Frobenius norm is induced?

*Solution.* $\|I_n\|_F = \sqrt{\sum_{i,j}\delta_{ij}^2} = \sqrt{n}$. Every induced norm must satisfy $\|I\| = 1$ (since $\|I\mathbf{x}\| = \|\mathbf{x}\|$ for all $\mathbf{x}$, so $\max_{\|\mathbf{x}\|=1}\|I\mathbf{x}\| = 1$). Since $\|I_n\|_F = \sqrt{n} \neq 1$ for $n > 1$, the Frobenius norm cannot be induced by any vector norm.

**Check 2.** A student claims: "Since $\|A\|_F \leq \sqrt{r}\|A\|_2$, I can upper bound the Frobenius norm by computing only the spectral norm." When is this bound tight and when is it loose?

*Solution.* The bound $\|A\|_F \leq \sqrt{r}\|A\|_2$ is tight when all $r$ singular values are equal ($\sigma_1 = \cdots = \sigma_r$), e.g., a scalar multiple of a unitary matrix. It is loose when singular values decay rapidly (e.g., $\sigma_i = \sigma_1/i$), in which case $\|A\|_F \approx \sigma_1\sqrt{\sum 1/i^2} \approx \sigma_1\cdot\pi/\sqrt{6}$ while the bound gives $\sigma_1\sqrt{r}$. For matrices with low stable rank, the bound is tight.

**Check 3.** Is the condition number invariant under matrix addition? I.e., does $\kappa(A+B) = \kappa(A) + \kappa(B)$?

*Solution.* No. Condition number is not additive. Consider $A = \text{diag}(100, 1)$ (so $\kappa(A) = 100$) and $B = \text{diag}(0, 99)$ (so $\kappa(B) = \infty$ since $B$ is singular). Then $A + B = \text{diag}(100, 100)$ with $\kappa(A+B) = 1$. The sum of two ill-conditioned matrices can be perfectly conditioned.

**Check 4.** Prove or disprove: $\|A+B\|_2 = \|A\|_2 + \|B\|_2$ implies $A$ and $B$ have the same top right singular vector.

*Solution.* True. The spectral norm satisfies the triangle inequality $\|A+B\|_2 \leq \|A\|_2 + \|B\|_2$ with equality iff $A$ and $B$ are "aligned" in the spectral sense. Specifically, equality holds iff there exist unit vectors $\mathbf{u}, \mathbf{v}$ such that $\mathbf{u}^\top A\mathbf{v} = \|A\|_2$ and $\mathbf{u}^\top B\mathbf{v} = \|B\|_2$ — meaning $\mathbf{v}$ is the top right singular vector of both $A$ and $B$, and $\mathbf{u}$ is the corresponding top left singular vector. (Same structure as Cauchy-Schwarz equality condition.)

### I.2 Computational Exercises with Answers

**Problem 1.** Let $A = \begin{pmatrix}2 & 1 \\ 0 & 3\end{pmatrix}$.

**(a)** Find all five norms: $\|A\|_F$, $\|A\|_2$, $\|A\|_*$, $\|A\|_1$, $\|A\|_\infty$.

**(b)** Find $\kappa_2(A)$ and the condition number of the equation $A\mathbf{x} = \mathbf{b}$.

**Solution.**

First compute $A^\top A = \begin{pmatrix}4&2\\2&10\end{pmatrix}$. Eigenvalues: $\lambda = (14 \pm \sqrt{196-36-16})/2$... Let us directly:

$\det(A^\top A - \lambda I) = (4-\lambda)(10-\lambda) - 4 = \lambda^2 - 14\lambda + 36 = 0$

$\lambda = (14 \pm \sqrt{196-144})/2 = (14 \pm \sqrt{52})/2 = 7 \pm \sqrt{13}$

So $\sigma_1 = \sqrt{7+\sqrt{13}} \approx \sqrt{10.606} \approx 3.257$ and $\sigma_2 = \sqrt{7-\sqrt{13}} \approx \sqrt{3.394} \approx 1.842$.

**(a) Results:**
- $\|A\|_F = \sqrt{4+1+0+9} = \sqrt{14} \approx 3.742$
- $\|A\|_2 = \sigma_1 \approx 3.257$
- $\|A\|_* = \sigma_1 + \sigma_2 \approx 3.257 + 1.842 = 5.099$
- $\|A\|_1 = \max(|2|+|0|, |1|+|3|) = \max(2, 4) = 4$
- $\|A\|_\infty = \max(|2|+|1|, |0|+|3|) = \max(3, 3) = 3$

**(b)** $\kappa_2(A) = \sigma_1/\sigma_2 \approx 3.257/1.842 \approx 1.768$.

This is a well-conditioned matrix. For $A\mathbf{x}=\mathbf{b}$ with relative perturbation $\epsilon$ in $\mathbf{b}$, the relative error in $\mathbf{x}$ is at most $\kappa \epsilon \approx 1.77\epsilon$.

**Problem 2.** Rank-1 approximation.

For $A = \begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix}$, find the best rank-1 approximation in the Frobenius norm and compute the approximation error.

**Solution.** Compute SVD. The singular values are $\sigma_1 \approx 9.525$ and $\sigma_2 \approx 0.514$. The best rank-1 approximation $A_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top$ has:
$$\|A - A_1\|_F = \sigma_2 \approx 0.514, \quad \|A - A_1\|_2 = \sigma_2 \approx 0.514$$

The approximation captures $\sigma_1^2/(\sigma_1^2+\sigma_2^2) = 90.7/91.0 \approx 99.7\%$ of the Frobenius-squared energy.

**Problem 3.** Tikhonov regularization effect on condition number.

Let $A = \text{diag}(10, 1, 0.01)$ (diagonal, well-structured). Compute $\kappa_2(A^\top A + \lambda I)$ for $\lambda \in \{0, 0.001, 0.01, 0.1, 1\}$ and observe the trade-off.

**Solution.** $A^\top A = \text{diag}(100, 1, 10^{-4})$, so $\kappa(A^\top A) = 10^6$.

$\kappa(A^\top A + \lambda I) = (100+\lambda)/(10^{-4}+\lambda)$:

| $\lambda$ | $\kappa_2$ | Notes |
| --- | --- | --- |
| $0$ | $10^6$ | Original: very ill-conditioned |
| $0.001$ | $(100.001)/(1.001\times10^{-3}) \approx 99,901$ | Modest improvement |
| $0.01$ | $(100.01)/(0.0101) \approx 9,902$ | 100× improvement |
| $0.1$ | $(100.1)/(0.1001) \approx 1,000$ | 1000× improvement |
| $1$ | $(101)/(1.0001) \approx 101$ | 10,000× improvement |

Each decade of $\lambda$ improves $\kappa$ dramatically but increases bias. The optimal $\lambda$ balances condition improvement against solution bias.

### I.3 Connection to Linear Programming

The nuclear norm can be computed via a semidefinite program (SDP) or, for special cases, via linear programming.

**Ky Fan norm via LP.** The Ky Fan $k$-norm has the dual representation:
$$\|A\|_{(k)} = \min\{k\mu + \operatorname{tr}(S) : S \succeq 0,\, S \geq 0,\, S + \mu I \geq \text{diag}(\boldsymbol{\sigma}),\, \mu \geq 0\}$$

(where the inequality is in the PSD sense on the diagonal). This is a semidefinite program in $(\mu, S)$.

**Nuclear norm via SDP.**
$$\|A\|_* = \min_{W_1,W_2}\frac{1}{2}(\operatorname{tr}(W_1)+\operatorname{tr}(W_2)) \quad \text{s.t.} \quad \begin{pmatrix}W_1&A\\A^\top&W_2\end{pmatrix} \succeq 0$$

This is a standard SDP with $O(m+n)$ variables. Interior-point methods solve it in $O((m+n)^3)$ time.

**For AI:** While these SDP representations are theoretically important (they prove polynomial solvability of nuclear norm problems), they are impractical for large matrices. Instead, proximal gradient descent with SVT (the practical algorithm) exploits the simple proximal operator structure.

---

## Appendix J: Quick Reference — Key Theorems

### J.1 The Four Fundamental Theorems of Matrix Norms

**Theorem 1 (Equivalence of Matrix Norms).** On $\mathbb{R}^{m\times n}$ (finite-dimensional), all matrix norms are equivalent: for any two norms $\|\cdot\|_\alpha$ and $\|\cdot\|_\beta$, there exist constants $c_1, c_2 > 0$ such that:
$$c_1\|A\|_\alpha \leq \|A\|_\beta \leq c_2\|A\|_\alpha \quad \text{for all } A \in \mathbb{R}^{m\times n}$$

*Implication:* Convergence (or divergence) in one norm implies convergence (or divergence) in all norms. Norms differ quantitatively, not qualitatively, in finite dimensions.

**Theorem 2 (Eckart-Young).** The best rank-$k$ approximation to $A$ in BOTH the Frobenius and spectral norms is the truncated SVD $A_k = \sum_{i=1}^k\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. The approximation errors are $\sigma_{k+1}$ (spectral) and $\sqrt{\sum_{i>k}\sigma_i^2}$ (Frobenius).

*Implication:* Low-rank approximation is solved optimally by SVD. This underpins PCA, LoRA, MLA, and all SVD-based compression methods.

**Theorem 3 (Weyl's Inequality).** Singular values are Lipschitz-1 functions of the matrix: $|\sigma_i(A+E) - \sigma_i(A)| \leq \sigma_1(E) = \|E\|_2$ for all $i$.

*Implication:* Singular values are numerically stable — guaranteed small changes for small perturbations. Eigenvalues of non-symmetric matrices do NOT have this guarantee.

**Theorem 4 (Von Neumann Trace Inequality).** $|\operatorname{tr}(A^\top B)| \leq \sum_i\sigma_i(A)\sigma_i(B)$, with equality when $A, B$ share singular vector bases.

*Implication:* The trace inner product $\langle A,B\rangle_F = \operatorname{tr}(A^\top B)$ is bounded by singular value inner products. This is the fundamental inequality behind nuclear-spectral duality and all Hölder-type bounds for matrix norms.

### J.2 Key Formulas at a Glance

**Norm relations for $A \in \mathbb{R}^{m\times n}$ of rank $r$:**

$$\|A\|_2 \leq \|A\|_F \leq \sqrt{r}\|A\|_2 \leq \sqrt{r}\|A\|_* \leq r\|A\|_2$$

**Condition number:** $\kappa_2(A) = \sigma_1/\sigma_n$ (for square nonsingular); each decimal digit of $\log_{10}\kappa$ costs one digit of precision in double arithmetic.

**Proximal operators:**

- $\operatorname{prox}_{\tau\|\cdot\|_*}(A) = U\operatorname{diag}(\max(\sigma_i-\tau,0))V^\top$ (nuclear norm → SVT)
- $\operatorname{prox}_{\tau\|\cdot\|_F^2}(A) = A/(1+2\tau)$ (Frobenius squared → scaling)

**Dual norm pairs:** spectral $\leftrightarrow$ nuclear; $\ell^p \leftrightarrow \ell^q$ (matrix 1-norm $\leftrightarrow$ $\infty$-norm); Frobenius is self-dual.

**Stable rank:** $\rho(A) = \|A\|_F^2/\|A\|_2^2 \in [1, r]$ — a smooth proxy for rank that controls generalization bounds.

### J.3 Notation Reference

Following the project notation guide (`docs/NOTATION_GUIDE.md`):

| Symbol | Meaning |
| --- | --- |
| $\|A\|_F$ | Frobenius norm of matrix $A$ |
| $\|A\|_2$ | Spectral (operator 2-) norm of $A$; equals $\sigma_1(A)$ |
| $\|A\|_*$ | Nuclear (trace) norm of $A$; equals $\sum_i\sigma_i(A)$ |
| $\|A\|_1$ | Matrix 1-norm (max absolute column sum) |
| $\|A\|_\infty$ | Matrix $\infty$-norm (max absolute row sum) |
| $\|A\|_{S_p}$ | Schatten $p$-norm; $\ell^p$ applied to singular values |
| $\|A\|_{(k)}$ | Ky Fan $k$-norm; sum of top-$k$ singular values |
| $\kappa_p(A)$ | Condition number in $p$-norm: $\|A\|_p\|A^{-1}\|_p$ |
| $\rho(A)$ | Stable rank: $\|A\|_F^2/\|A\|_2^2$ |
| $\sigma_i(A)$ | $i$-th singular value (decreasing order) |
| $\boldsymbol{\sigma}(A)$ | Vector of all singular values |
| $\langle A,B\rangle_F$ | Frobenius inner product: $\operatorname{tr}(A^\top B)$ |

---

*← [Back to Advanced Linear Algebra](../README.md) | [Next: Linear Transformations →](../07-Linear-Transformations/notes.md)*

## Appendix K: Further Reading

### K.1 Textbooks

1. **Golub & Van Loan** — *Matrix Computations* (4th ed., 2013). The definitive numerical linear algebra reference. Chapters 2-3 cover matrix norms and condition numbers in full depth.

2. **Horn & Johnson** — *Matrix Analysis* (2nd ed., 2013). Comprehensive theoretical treatment of matrix norms, singular values, and inequalities.

3. **Bhatia** — *Matrix Analysis* (1997). Advanced treatment of matrix inequalities, Schatten norms, and majorization. Chapter IV covers unitarily invariant norms.

4. **Trefethen & Bau** — *Numerical Linear Algebra* (1997). Excellent pedagogical treatment. Lectures 4-5 cover norms; Lectures 11-12 cover condition number theory.

### K.2 Foundational Papers

1. **Eckart & Young (1936)** — "The approximation of one matrix by another of lower rank." *Psychometrika*. The Eckart-Young theorem.

2. **Mirsky (1960)** — "Symmetric gauge functions and unitarily invariant norms." *Quarterly Journal of Mathematics*. Unitarily invariant norm characterization via symmetric gauge functions.

3. **Candès & Recht (2009)** — "Exact matrix completion via convex optimization." *Foundations of Computational Mathematics*. Nuclear norm for matrix recovery under incoherence.

4. **Cai, Candès & Shen (2010)** — "A singular value thresholding algorithm for matrix completion." *SIAM Journal on Optimization*. The SVT proximal algorithm.

### K.3 Machine Learning Papers

1. **Miyato et al. (2018)** — "Spectral Normalization for Generative Adversarial Networks." ICLR 2018. Spectral norm in GAN training; power iteration for $\sigma_1$.

2. **Bartlett, Foster & Telgarsky (2017)** — "Spectrally-normalized margin bounds for neural networks." NeurIPS 2017. PAC-Bayes generalization bounds via spectral and Frobenius norms.

3. **Gunasekar et al. (2017)** — "Implicit Regularization in Matrix Factorization." NeurIPS 2017. Gradient flow on factored parameterization implicitly minimizes nuclear norm.

4. **Hu et al. (2021)** — "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. Low-rank weight adaptation; implicit nuclear norm regularization.

5. **Dong et al. (2021)** — "Attention is Not All You Need: Pure Attention Loses Rank Doubly Exponentially with Depth." ICML 2021. Attention matrix rank analysis via spectral/nuclear norms.

6. **DeepSeek-AI (2024)** — "DeepSeek-V2." MLA architecture and nuclear norm-motivated KV cache compression.

### K.4 Online Resources

- **Gilbert Strang's Linear Algebra lectures** (MIT OpenCourseWare 18.06): Lectures 29-33 cover SVD, norms, and condition numbers with worked examples.
- **Numerical Linear Algebra** (Trefethen, Oxford): Freely available course notes that complement the textbook above.
- **Matrix Cookbook** (Petersen & Pedersen, 2012): Dense reference for matrix identities and norm formulas. Freely available as a PDF.
- **Convex Optimization** (Boyd & Vandenberghe, Stanford): Chapter 6 covers matrix norms as convex functions; Chapter 11 covers proximal methods including SVT.

---

*This section is part of the Math for LLMs curriculum. All notation follows `docs/NOTATION_GUIDE.md`. For visualization standards, see `docs/VISUALIZATION_GUIDE.md`.*

*← [Back to Advanced Linear Algebra](../README.md) | [Next: Linear Transformations →](../07-Linear-Transformations/notes.md)*

---

## Summary

Matrix norms are the central quantitative tools of matrix analysis. This section covered:

**Core norm families** — The five principal matrix norms (Frobenius, spectral, nuclear, matrix-1, matrix-$\infty$) and the broader Schatten family that unifies them. Each norm captures a different geometric property of a linear map: the Frobenius norm measures RMS stretching; the spectral norm measures maximum stretching; the nuclear norm measures total stretching (sum of singular values).

**Induced vs. non-induced** — The spectral, matrix-1, and matrix-$\infty$ norms are induced (they arise from vector norms on input/output spaces). The Frobenius norm is not induced but is compatible. Every induced norm is submultiplicative; submultiplicativity is the key property for stability analysis.

**Unitarily invariant norms** — Norms that depend only on singular values (Frobenius, spectral, nuclear, Schatten, Ky Fan) have a unified theory via symmetric gauge functions. The Von Neumann trace inequality and Mirsky's theorem are the central results.

**Condition number** — The ratio $\sigma_1/\sigma_n$ quantifies sensitivity to perturbations. Every decimal digit of $\log_{10}\kappa$ costs one digit of precision. Tikhonov regularization reduces $\kappa$ at the cost of introducing systematic bias.

**Perturbation theory** — Weyl's inequality (singular values are Lipschitz-1 w.r.t. spectral norm) guarantees stability of SVD computations. The Bauer-Fike theorem shows that non-normal eigenvalues can be much less stable.

**Machine learning connections** — Every major modern ML technique has a matrix norm interpretation: weight decay (Frobenius), spectral normalization (spectral), LoRA (nuclear via factored form), gradient clipping (spectral or Frobenius of gradients), PAC-Bayes bounds (stable rank and Frobenius/spectral), MLA compression (Eckart-Young), and attention analysis (nuclear norm rank collapse).

The key unifying insight: **matrix norms reduce infinite-dimensional questions about linear operators to finite-dimensional scalar measurements.** This reduction is the mathematical move that makes analysis tractable, computation feasible, and regularization principled.


This understanding of norms as measurement instruments prepares us for the next section on Linear Transformations, where we use norms to study how maps between vector spaces can be classified, composed, and analyzed. The spectral norm of a transformation matrix is precisely its operator norm — the fundamental quantity controlling how much the transformation can stretch vectors. Every topic in the remaining curriculum — optimization convergence (Chapter 8), probabilistic models (Chapter 10), and the mathematics of specific architectures (Chapter 14) — uses matrix norms as a core tool.

The journey from abstract norm axioms to practical tools — power iteration, singular value thresholding, Tikhonov regularization, spectral normalization — illustrates how pure mathematics becomes engineering. Matrix norms are not just theoretical objects but the computational primitives of modern AI.


Matrix norms connect every part of linear algebra — the SVD gives their values, orthogonality explains their invariance, eigenvalues govern condition numbers — and every part of machine learning — norms define regularizers, bound generalization, and stabilize training. They are the language in which the theory and practice of modern AI is written.
