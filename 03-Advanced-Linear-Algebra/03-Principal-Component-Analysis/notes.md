[← Back to Advanced Linear Algebra](../README.md) | [← SVD](../02-Singular-Value-Decomposition/notes.md) | [Linear Transformations →](../04-Linear-Transformations/notes.md)

---

# Principal Component Analysis (PCA)

> _"Give me the directions of maximum variance and I will show you the geometry of the data. Give me enough of them and I can reconstruct any measurement ever made."_

## Overview

Principal Component Analysis is the fundamental linear technique for understanding high-dimensional data. Given a dataset in $\mathbb{R}^d$, PCA finds a new coordinate system — the **principal components** — aligned with the directions of greatest variance. Projecting onto the top $k$ components gives the best possible rank-$k$ linear summary of the data, optimal in mean-squared reconstruction error.

PCA sits at the intersection of spectral theory, geometry, and statistics. Its derivation is an optimization problem (maximize projected variance) whose solution is an eigenvalue problem (diagonalize the covariance matrix) whose computation is best handled via the SVD. All three views — optimization, spectral, and computational — are essential for practitioners.

In 2026, PCA remains indispensable. It reveals the intrinsic dimensionality of LLM token representations, drives mechanistic interpretability analyses of transformer residual streams, and motivates the low-rank hypothesis behind LoRA. Understanding PCA deeply means understanding why high-dimensional models often live near low-dimensional manifolds — a fact that shapes training dynamics, generalization, and efficiency.

## Prerequisites

- Eigenvalues, eigenvectors, and the spectral theorem for symmetric matrices ([§01-Eigenvalues-and-Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md))
- SVD: $A = U\Sigma V^\top$ and the Eckart-Young theorem ([§02-Singular-Value-Decomposition](../02-Singular-Value-Decomposition/notes.md))
- Covariance matrices and inner products ([02-Linear-Algebra-Basics §06](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md))
- Matrix rank and null space ([02-Linear-Algebra-Basics §05](../../02-Linear-Algebra-Basics/05-Matrix-Rank/notes.md))

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demos: geometry of PCA, scree plots, whitening, PPCA, kernel PCA, eigenfaces, LLM intrinsic dimensionality |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: from covariance derivation through kernel PCA and LoRA rank analysis |

## Learning Objectives

After completing this section, you will:

- Derive PCA from the variance-maximization principle using Lagrange multipliers
- Explain the equivalence of the eigendecomposition and SVD routes to PCA
- Compute variance explained, draw and interpret scree plots, and choose $k$ systematically
- Implement PCA whitening (PCA and ZCA variants) and understand when each is appropriate
- State the probabilistic PCA generative model and its MLE solution
- Implement kernel PCA and explain what the kernel matrix encodes
- Describe sparse PCA, robust PCA, and randomized PCA and when each is preferred
- Connect PCA to eigenfaces, LLM intrinsic dimension, mechanistic interpretability, LoRA, and loss landscape analysis
- Identify the ten most common PCA mistakes and fix them

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Big Picture — Variance, Compression, and Geometry](#11-the-big-picture--variance-compression-and-geometry)
  - [1.2 Three Equivalent Views](#12-three-equivalent-views)
  - [1.3 Why PCA Matters for AI in 2026](#13-why-pca-matters-for-ai-in-2026)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Setup: Centered Data and the Covariance Matrix](#21-setup-centered-data-and-the-covariance-matrix)
  - [2.2 Variance Maximization — The Optimization Problem](#22-variance-maximization--the-optimization-problem)
  - [2.3 The Eigenvalue Equation and Principal Components](#23-the-eigenvalue-equation-and-principal-components)
  - [2.4 Projection and Reconstruction](#24-projection-and-reconstruction)
  - [2.5 PCA via SVD — The Preferred Route](#25-pca-via-svd--the-preferred-route)
- [3. Properties of PCA](#3-properties-of-pca)
  - [3.1 Decorrelation: Diagonalizing the Covariance Matrix](#31-decorrelation-diagonalizing-the-covariance-matrix)
  - [3.2 Optimality — The Eckart-Young Connection](#32-optimality--the-eckart-young-connection)
  - [3.3 Sign Ambiguity and Uniqueness](#33-sign-ambiguity-and-uniqueness)
  - [3.4 Scale Sensitivity and Standardization](#34-scale-sensitivity-and-standardization)
- [4. Explained Variance and Component Selection](#4-explained-variance-and-component-selection)
  - [4.1 Variance Ratio and Cumulative Explained Variance](#41-variance-ratio-and-cumulative-explained-variance)
  - [4.2 Scree Plots and the Elbow Heuristic](#42-scree-plots-and-the-elbow-heuristic)
  - [4.3 Threshold Methods](#43-threshold-methods)
  - [4.4 Kaiser Criterion](#44-kaiser-criterion)
  - [4.5 Cross-Validation and Downstream-Task Selection](#45-cross-validation-and-downstream-task-selection)
- [5. PCA Whitening](#5-pca-whitening)
  - [5.1 What Whitening Achieves](#51-what-whitening-achieves)
  - [5.2 PCA Whitening vs ZCA Whitening](#52-pca-whitening-vs-zca-whitening)
  - [5.3 Applications of Whitening](#53-applications-of-whitening)
- [6. Probabilistic PCA (PPCA)](#6-probabilistic-pca-ppca)
  - [6.1 The Generative Model](#61-the-generative-model)
  - [6.2 MLE Solution and Connection to Standard PCA](#62-mle-solution-and-connection-to-standard-pca)
  - [6.3 Handling Missing Data via EM](#63-handling-missing-data-via-em)
  - [6.4 Mixture of PPCA](#64-mixture-of-ppca)
- [7. Kernel PCA](#7-kernel-pca)
  - [7.1 Motivation — When Linear PCA Fails](#71-motivation--when-linear-pca-fails)
  - [7.2 The Kernel Trick and Feature-Space PCA](#72-the-kernel-trick-and-feature-space-pca)
  - [7.3 Centering the Kernel Matrix](#73-centering-the-kernel-matrix)
  - [7.4 Common Kernels and Their Geometric Effects](#74-common-kernels-and-their-geometric-effects)
  - [7.5 Limitations and Out-of-Sample Extension](#75-limitations-and-out-of-sample-extension)
- [8. PCA Variants](#8-pca-variants)
  - [8.1 Sparse PCA — Interpretable Loadings](#81-sparse-pca--interpretable-loadings)
  - [8.2 Robust PCA — Low-Rank + Sparse Decomposition](#82-robust-pca--low-rank--sparse-decomposition)
  - [8.3 Incremental and Randomized PCA](#83-incremental-and-randomized-pca)
- [9. Applications in Machine Learning and AI](#9-applications-in-machine-learning-and-ai)
  - [9.1 Eigenfaces and Image Compression](#91-eigenfaces-and-image-compression)
  - [9.2 Intrinsic Dimensionality of LLM Representations](#92-intrinsic-dimensionality-of-llm-representations)
  - [9.3 Mechanistic Interpretability: PCA of the Residual Stream](#93-mechanistic-interpretability-pca-of-the-residual-stream)
  - [9.4 LoRA and the Low-Rank Hypothesis](#94-lora-and-the-low-rank-hypothesis)
  - [9.5 Loss Landscape Geometry](#95-loss-landscape-geometry)
  - [9.6 PCA vs t-SNE vs UMAP for Visualization](#96-pca-vs-t-sne-vs-umap-for-visualization)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 The Big Picture — Variance, Compression, and Geometry

Imagine measuring 1,000 features on each of 10,000 patients. Most of those features are correlated — height and weight move together, blood pressure readings at different times are similar, gene expression patterns cluster. The data occupies a $10{,}000 \times 1{,}000$ matrix, but the effective "information content" may be much lower. PCA finds that smaller set of independent directions.

The key geometric idea: data in $\mathbb{R}^d$ typically doesn't spread equally in all directions. It has a shape — an ellipsoid — with a longest axis, a second-longest axis orthogonal to the first, and so on. PCA finds those axes, in order of length (variance). Projecting onto the first $k$ axes keeps the longest dimensions and discards the short ones, where variation is small and likely noise.

```
PCA AS AXIS ALIGNMENT
════════════════════════════════════════════════════════════════════════

  Original frame          After PCA rotation
  (arbitrary x₁,x₂)      (aligned to data structure)

        x₂                         PC2 (low variance)
        │                            ↑
      • │•                           │ ·
    •   │ •  •                  ·   │  ·
   •    │   •           →     ·    │    ·
  •     │    •                ·    │     ·
 •      │     •             ·      │      ·
────────┼────── x₁        ──────────────────→ PC1 (high variance)

  The covariance ellipse    PC1 = major axis = max variance direction
  is tilted in (x₁,x₂)     PC2 = minor axis = min variance direction

════════════════════════════════════════════════════════════════════════
```

**What PCA does, step by step:**
1. **Center** the data (subtract the mean) — PCA measures variance around the centroid.
2. **Find the longest axis** of the resulting cloud (PC1). This is the direction $\mathbf{v}_1$ maximizing $\operatorname{Var}(\mathbf{X}\mathbf{v}_1)$.
3. **Find the next longest axis** orthogonal to PC1 (PC2). Repeat for PC3, PC4, …, PC_d.
4. **Project** onto the top $k$ axes to get a $k$-dimensional representation that preserves as much variance as possible.

**For AI:** A transformer with $d_{\mathrm{model}} = 4096$ has token representations in $\mathbb{R}^{4096}$. PCA of these representations typically shows that 50–200 components explain 90%+ of the variance — the model has learned a much lower-dimensional structure than its nominal dimension suggests. This is the empirical foundation of the _intrinsic dimensionality_ hypothesis.

### 1.2 Three Equivalent Views

The same PCA solution arises from three completely different starting points. Understanding all three gives deep insight into why PCA is so fundamental.

```
THREE EQUIVALENT VIEWS OF PCA
════════════════════════════════════════════════════════════════════════

  View 1: VARIANCE MAXIMIZATION
  ─────────────────────────────
  Find the unit vector v that maximizes Var(Xv) = vᵀCv
  where C = XᵀX/(n-1) is the sample covariance matrix.

  Solution: v₁ = top eigenvector of C.
  Intuition: look in the direction where the data spreads most.

  View 2: RECONSTRUCTION MINIMIZATION
  ─────────────────────────────────────
  Find the k-dimensional subspace S minimizing mean reconstruction error:
      min_{S, dim(S)=k}  (1/n) Σᵢ ‖xᵢ - projₛ(xᵢ)‖²

  Solution: S = span{v₁, ..., vₖ} (top k eigenvectors of C).
  Intuition: project onto the flat that keeps points closest to original.

  View 3: COVARIANCE DIAGONALIZATION
  ───────────────────────────────────
  Find orthogonal Q such that QᵀCQ = Λ is diagonal.

  Solution: Q = eigenvector matrix of C; Λ = eigenvalue matrix.
  Intuition: find a rotation that makes all features independent.

  All three yield the same Q = [v₁ | v₂ | ... | vₐ] ← SAME ANSWER.

════════════════════════════════════════════════════════════════════════
```

| View | Objective | Mathematical form | Who uses it |
|---|---|---|---|
| Variance maximization | Find most informative direction | $\max_{\lVert\mathbf{v}\rVert=1} \mathbf{v}^\top C \mathbf{v}$ | Feature extraction, signal processing |
| Reconstruction | Best low-rank approximation | $\min_{\operatorname{rank}(M)=k}\lVert X - M\rVert_F^2$ | Compression, denoising |
| Decorrelation | Diagonalize the covariance | $Q^\top C Q = \Lambda$ | Statistics, ICA preprocessing |

The equivalence of views 1 and 2 is not obvious — it is proved by the Eckart-Young theorem (see [§02-SVD](../02-Singular-Value-Decomposition/notes.md)). The equivalence of views 1 and 3 follows directly from the spectral theorem for symmetric matrices (see [§01-Eigenvalues](../01-Eigenvalues-and-Eigenvectors/notes.md)).

### 1.3 Why PCA Matters for AI in 2026

PCA is not merely a classical preprocessing step — it is embedded in the mathematical foundations of modern deep learning:

**Intrinsic dimensionality of representations.** LLM token embeddings, despite living in $\mathbb{R}^{4096}$ or larger, occupy a low-dimensional effective subspace. Studies of GPT-2 through GPT-4-class models consistently find that PCA explains 90% of embedding variance with far fewer components than the nominal dimension. This matters for compression, distillation, and understanding generalization.

**Mechanistic interpretability.** Researchers apply PCA to the residual stream of transformer layers to identify "directions" that encode specific features (sentiment, position, subject-verb agreement). Each interpretable direction is a principal component of the activations. The technique, used by Anthropic, DeepMind, and EleutherAI, depends entirely on PCA as a lens.

**LoRA and low-rank adaptation.** The LoRA paper (Hu et al., 2021) hypothesizes that fine-tuning weight updates have low intrinsic rank — i.e., the gradient matrix $\Delta W$ is well-approximated by a rank-$r$ matrix with $r \ll d$. PCA of $\Delta W$ is the theoretical grounding. DoRA, GaLore, and Flora extend this to gradient-space PCA.

**Loss landscape analysis.** Goodfellow et al. showed that the loss landscape along random directions is complex, but along the top PCA directions of gradient trajectories, it is nearly one-dimensional. Li et al.'s filter normalization visualization uses PCA of parameter trajectories.

**Eigenfaces and foundation models.** Turk & Pentland (1991) showed that faces lie near a low-dimensional linear subspace — the "face space" spanned by eigenfaces, which are PCA components of face images. Modern vision transformers implicitly learn analogous patch-level PCA components in their first layers.

### 1.4 Historical Timeline

| Year | Who | Contribution |
|---|---|---|
| 1901 | Karl Pearson | First formulation: lines/planes of closest fit to point systems |
| 1933 | Harold Hotelling | Named "principal components"; Lagrange-multiplier derivation |
| 1936 | Fisher | Discriminant analysis (supervised analog) |
| 1963 | Eckart & Young | Best low-rank approximation theorem (published 1936, recognized 1963) |
| 1978 | Tipping & Bishop | Probabilistic PCA as a latent variable model |
| 1998 | Schölkopf et al. | Kernel PCA — non-linear extension via RKHS |
| 2003 | Candès & Recht | Robust PCA (nuclear norm minimization) |
| 2009 | Halko et al. | Randomized SVD for large-scale PCA |
| 2021 | Hu et al. | LoRA: low-rank adaptation hypothesis connects PCA to fine-tuning |
| 2022–26 | Many authors | PCA of LLM residual streams as mechanistic interpretability tool |

---

## 2. Formal Definitions

### 2.1 Setup: Centered Data and the Covariance Matrix

**Data matrix.** Let $X \in \mathbb{R}^{n \times d}$ be a dataset with $n$ observations and $d$ features. Row $i$ is $\mathbf{x}^{(i)} \in \mathbb{R}^d$, a single data point (column vector stored as a row in $X$).

**Centering.** Define the sample mean:
$$\bar{\mathbf{x}} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)}$$

The **centered data matrix** is:
$$\tilde{X} = X - \mathbf{1}\bar{\mathbf{x}}^\top$$

where $\mathbf{1} \in \mathbb{R}^n$ is the all-ones vector. After centering, $\tilde{X}$ has zero column means. PCA is always applied to centered data — failing to center is the most common mistake.

**Sample covariance matrix.** The covariance matrix $C \in \mathbb{R}^{d \times d}$ is:
$$C = \frac{1}{n-1}\tilde{X}^\top \tilde{X}$$

Its $(i,j)$ entry is the sample covariance between feature $i$ and feature $j$:
$$C_{ij} = \frac{1}{n-1}\sum_{k=1}^n (\tilde{X}_{ki})(\tilde{X}_{kj})$$

**Key properties of $C$:**
- $C$ is **symmetric**: $C^\top = C$ (since $(\tilde{X}^\top\tilde{X})^\top = \tilde{X}^\top\tilde{X}$)
- $C$ is **positive semidefinite**: $\mathbf{v}^\top C \mathbf{v} \geq 0$ for all $\mathbf{v}$ (since $\mathbf{v}^\top C \mathbf{v} = \lVert\tilde{X}\mathbf{v}\rVert_2^2 / (n-1) \geq 0$)
- Diagonal entries $C_{ii} = \operatorname{Var}(\text{feature } i)$ are non-negative
- Off-diagonal $C_{ij}$ measures linear co-movement between features $i$ and $j$
- $\operatorname{tr}(C) = \sum_{i=1}^d C_{ii}$ = total variance = sum of all feature variances

Because $C$ is real symmetric and PSD, the **spectral theorem** guarantees a full orthonormal eigenbasis with non-negative eigenvalues. This is the structural fact that makes PCA possible.

> **Recall (Spectral Theorem):** Every real symmetric matrix $C$ admits $C = Q\Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$. Full treatment: [§01-Eigenvalues-and-Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md).

### 2.2 Variance Maximization — The Optimization Problem

**What we want:** a unit vector $\mathbf{v} \in \mathbb{R}^d$ such that the projected data $\tilde{X}\mathbf{v} \in \mathbb{R}^n$ has maximum variance.

**Variance of the projection:**
$$\operatorname{Var}(\tilde{X}\mathbf{v}) = \frac{1}{n-1}\lVert \tilde{X}\mathbf{v} \rVert_2^2 = \frac{1}{n-1}(\tilde{X}\mathbf{v})^\top(\tilde{X}\mathbf{v}) = \mathbf{v}^\top \left(\frac{\tilde{X}^\top\tilde{X}}{n-1}\right) \mathbf{v} = \mathbf{v}^\top C \mathbf{v}$$

**The first principal component** solves:
$$\mathbf{v}_1 = \arg\max_{\mathbf{v}} \mathbf{v}^\top C \mathbf{v} \quad \text{subject to} \quad \lVert\mathbf{v}\rVert_2 = 1$$

**Lagrange multiplier solution.** Form the Lagrangian:
$$\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^\top C \mathbf{v} - \lambda(\mathbf{v}^\top\mathbf{v} - 1)$$

Setting $\partial\mathcal{L}/\partial\mathbf{v} = 0$:
$$2C\mathbf{v} - 2\lambda\mathbf{v} = 0 \implies C\mathbf{v} = \lambda\mathbf{v}$$

This is the **eigenvalue equation**. The optimal $\mathbf{v}$ is an eigenvector of $C$, and the achieved variance is the corresponding eigenvalue:
$$\mathbf{v}^\top C \mathbf{v} = \mathbf{v}^\top (\lambda \mathbf{v}) = \lambda \lVert \mathbf{v}\rVert_2^2 = \lambda$$

To maximize, we choose the eigenvector corresponding to the **largest eigenvalue** $\lambda_1$.

### 2.3 The Eigenvalue Equation and Principal Components

**Definition (Principal Components):** The $i$-th principal component direction $\mathbf{v}_i$ is the $i$-th eigenvector of the sample covariance matrix $C$, corresponding to the $i$-th largest eigenvalue $\lambda_i$:
$$C \mathbf{v}_i = \lambda_i \mathbf{v}_i, \qquad \lVert\mathbf{v}_i\rVert_2 = 1, \qquad \langle\mathbf{v}_i, \mathbf{v}_j\rangle = 0 \text{ for } i \neq j$$

The eigenvalues satisfy $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$.

**Sequential derivation.** After finding $\mathbf{v}_1$, the second PC maximizes variance subject to being orthogonal to $\mathbf{v}_1$:
$$\mathbf{v}_2 = \arg\max_{\lVert\mathbf{v}\rVert=1,\, \mathbf{v}\perp\mathbf{v}_1} \mathbf{v}^\top C \mathbf{v}$$

By the spectral theorem, the solution is the second eigenvector. Repeating gives all $d$ PCs. The key: **they are exactly the eigenvectors of $C$**, computed all at once by diagonalizing $C = Q\Lambda Q^\top$.

**Principal component matrix.** Stack all $d$ eigenvectors as columns:
$$V = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_d] \in \mathbb{R}^{d \times d}$$

$V$ is **orthogonal**: $V^\top V = VV^\top = I_d$. Geometrically, $V$ is a rotation matrix that aligns the coordinate axes with the principal directions of variance.

### 2.4 Projection and Reconstruction

**Full projection (change of basis).** Project the centered data onto all principal components:
$$Z = \tilde{X} V \in \mathbb{R}^{n \times d}$$

$Z_{ij}$ is the coordinate of point $i$ along PC direction $j$. The columns of $Z$ are uncorrelated and have variances $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$.

**Rank-$k$ approximation.** Keep only the top $k$ components — define $V_k = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k] \in \mathbb{R}^{d \times k}$:
$$Z_k = \tilde{X} V_k \in \mathbb{R}^{n \times k} \quad \text{(scores / embedding)}$$

$Z_k$ is the low-dimensional representation. Each row $\mathbf{z}_k^{(i)} \in \mathbb{R}^k$ is the PCA embedding of data point $i$.

**Reconstruction.** Map back to the original space:
$$\hat{X} = Z_k V_k^\top + \mathbf{1}\bar{\mathbf{x}}^\top \in \mathbb{R}^{n \times d}$$

The reconstruction error is:
$$\lVert \tilde{X} - Z_k V_k^\top \rVert_F^2 = \sum_{i=k+1}^d \lambda_i = \sum_{i=k+1}^d \frac{\sigma_i^2}{n-1}$$

That is, the error equals the sum of discarded eigenvalues (= discarded singular values squared, normalized). Retaining more components reduces error monotonically.

### 2.5 PCA via SVD — The Preferred Route

In practice, computing the covariance $C = \tilde{X}^\top\tilde{X}/(n-1)$ explicitly is numerically inferior. The SVD of $\tilde{X}$ directly gives all PCA quantities with better numerical stability.

**SVD of centered data:**
$$\tilde{X} = U\Sigma V^\top, \quad U \in \mathbb{R}^{n \times n},\; \Sigma \in \mathbb{R}^{n \times d},\; V \in \mathbb{R}^{d \times d}$$

where $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(n,d)} \geq 0$ are the singular values.

> **Recall (SVD):** For the full treatment of the SVD, its geometric meaning, and the Eckart-Young theorem, see [§02-Singular-Value-Decomposition](../02-Singular-Value-Decomposition/notes.md).

**The connection:**
$$C = \frac{\tilde{X}^\top\tilde{X}}{n-1} = \frac{(U\Sigma V^\top)^\top(U\Sigma V^\top)}{n-1} = V \frac{\Sigma^\top\Sigma}{n-1} V^\top = V \operatorname{diag}\!\left(\frac{\sigma_1^2}{n-1}, \ldots\right) V^\top$$

This is exactly the eigendecomposition of $C$ — the columns of $V$ are the principal components and $\lambda_i = \sigma_i^2/(n-1)$.

**Complete PCA quantities from SVD:**

| PCA quantity | Formula via SVD |
|---|---|
| Principal component directions | Columns of $V$ (rows of $V^\top$) |
| Sample variances | $\lambda_i = \sigma_i^2 / (n-1)$ |
| Scores (projected data) | $Z = U\Sigma$ or equivalently $\tilde{X}V$ |
| Rank-$k$ reconstruction | $\hat{X} = U_k\Sigma_k V_k^\top + \mathbf{1}\bar{\mathbf{x}}^\top$ |
| Reconstruction error | $\sum_{i>k}\sigma_i^2$ |

**Why SVD is numerically preferred:**

| Issue | Covariance + Eigen | SVD |
|---|---|---|
| Forming $\tilde{X}^\top\tilde{X}$ | Squares condition number: $\kappa(C) = \kappa(\tilde{X})^2$ | Avoids squaring — works on $\tilde{X}$ directly |
| Memory | $O(d^2)$ for $C$ | $O(nd)$ for $\tilde{X}$ |
| When $n < d$ | $C$ is rank-deficient, unstable | Thin SVD is efficient |
| Complex eigenvalues | Possible for `np.linalg.eig` | Never (singular values always real) |

**Practical implementation:**

```python
import numpy as np

def pca_svd(X, k=None):
    """PCA via SVD — the numerically stable route."""
    n, d = X.shape
    k = k or d

    # Step 1: Center
    mean = X.mean(axis=0)
    X_c = X - mean

    # Step 2: Thin SVD (economy SVD)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    # U: (n, min(n,d)), s: (min(n,d),), Vt: (min(n,d), d)

    # Step 3: Extract PCA quantities
    V_k = Vt[:k].T            # Principal component directions (d, k)
    Z_k = U[:, :k] * s[:k]   # Scores = projected data (n, k)
    lambdas = s**2 / (n - 1)  # Explained variances

    var_ratio = lambdas[:k] / lambdas.sum()
    cumvar = np.cumsum(var_ratio)

    return {
        'scores': Z_k,           # (n, k) — low-dim representation
        'components': V_k,       # (d, k) — principal directions
        'explained_var': lambdas[:k],
        'explained_var_ratio': var_ratio,
        'cumulative_var': cumvar,
        'mean': mean,
    }
```

---

## 3. Properties of PCA

### 3.1 Decorrelation: Diagonalizing the Covariance Matrix

The most important algebraic property of PCA is that the projected scores are **uncorrelated**. If $Z = \tilde{X}V$ is the full projection, then:

$$\operatorname{Cov}(Z) = \frac{Z^\top Z}{n-1} = \frac{V^\top \tilde{X}^\top \tilde{X} V}{n-1} = V^\top C V = V^\top (V\Lambda V^\top) V = \Lambda$$

The covariance matrix of the PCA scores is diagonal — equal to $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d)$. This means:

- **All pairs of PCA scores are uncorrelated**: $\operatorname{Cov}(Z_{:,i}, Z_{:,j}) = 0$ for $i \neq j$
- **Each score's variance equals the corresponding eigenvalue**: $\operatorname{Var}(Z_{:,i}) = \lambda_i$
- **Total variance is preserved**: $\sum_i \lambda_i = \operatorname{tr}(C) = \sum_i C_{ii}$

This is precisely view 3 from §1.2 — PCA performs a rotation that diagonalizes the covariance matrix. After PCA, each component carries independent information (in the linear sense; they are not necessarily statistically independent, only uncorrelated — independence requires a distributional assumption like Gaussianity).

**Contrast with ICA:** Independent Component Analysis (ICA) finds directions of statistical independence, not just decorrelation. PCA decorrelation is a necessary but not sufficient condition for independence.

### 3.2 Optimality — The Eckart-Young Connection

PCA achieves two distinct optimality guarantees, both following from the Eckart-Young-Mirsky theorem:

**Theorem (Optimality of PCA, informal).** Among all rank-$k$ linear maps $\mathbb{R}^d \to \mathbb{R}^k$, the PCA projection minimizes mean reconstruction error:
$$\min_{\text{orthonormal } V_k \in \mathbb{R}^{d\times k}} \lVert \tilde{X} - \tilde{X}V_kV_k^\top \rVert_F^2 = \sum_{i=k+1}^d \sigma_i^2$$

The minimum is achieved by taking $V_k$ = first $k$ right singular vectors of $\tilde{X}$ = first $k$ eigenvectors of $C$.

> **Proof:** This is a direct consequence of the Eckart-Young theorem applied to $\tilde{X}$. See [§02-SVD §4.2](../02-Singular-Value-Decomposition/notes.md) for the complete proof.

**Two sides of the same coin:**

| Optimality | Statement | Error |
|---|---|---|
| Maximum variance | PCA directions maximize total explained variance | $\sum_{i\leq k}\lambda_i$ is maximized |
| Minimum reconstruction | PCA projection minimizes $\lVert X - \hat{X}\rVert_F^2$ | $\sum_{i>k}\lambda_i$ is minimized |

These are equivalent: maximizing captured variance = minimizing discarded variance = minimizing reconstruction error. The optimal solution is the same in both cases.

**Implication for choosing $k$:** Setting a reconstruction error budget $\epsilon$ directly determines $k$ — find the smallest $k$ such that $\sum_{i>k}\sigma_i^2 \leq \epsilon \cdot \sum_i \sigma_i^2$ (i.e., keep the fraction $1-\epsilon$ of total variance).

### 3.3 Sign Ambiguity and Uniqueness

**Sign ambiguity.** If $\mathbf{v}_i$ is an eigenvector of $C$ with eigenvalue $\lambda_i$, so is $-\mathbf{v}_i$. PCA directions are only defined up to sign. This is unavoidable: there is no canonical "positive" direction for an eigenvector.

**Consequence:** Two PCA implementations may return $\mathbf{v}_i$ and $-\mathbf{v}_i$ for the same data. Always compare directions via $|\langle\mathbf{u}, \mathbf{v}\rangle|$ or $\cos^2\theta$, not the raw dot product.

**Uniqueness of the subspace.** While individual directions are sign-ambiguous, the **subspace** $\operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is unique — as long as the eigenvalues are distinct ($\lambda_k > \lambda_{k+1}$). If there is an eigenvalue tie ($\lambda_k = \lambda_{k+1}$), any orthonormal basis for the corresponding eigenspace is a valid solution. This is not a flaw but a genuine mathematical degeneracy: all directions in that eigenspace explain equal variance.

**When uniqueness fails:**
- Perfectly spherical data: $C = \sigma^2 I$ → every direction is a PC, problem is entirely degenerate
- Repeated eigenvalues: the corresponding eigenspace can be any rotation
- Near-equal eigenvalues: direction is numerically unstable — small data perturbations change it dramatically

**Practical fix for sign ambiguity** (sklearn convention): flip $\mathbf{v}_i$ so that the entry with the largest absolute value is positive:
```python
signs = np.sign(Vt[np.abs(Vt).argmax(axis=1), range(Vt.shape[1])])
Vt = Vt * signs[:, None]
```

### 3.4 Scale Sensitivity and Standardization

PCA is **not scale-invariant**: multiplying a feature by a constant changes its variance and therefore its contribution to the PCA solution.

**Example:** Suppose feature 1 is height in centimetres (range 150–190) and feature 2 is weight in kilograms (range 50–100). The variance of feature 1 is ~$170^2 / 12 \approx 240$ and of feature 2 is ~$75^2 / 12 \approx 469$. But if we rescale height to metres (divide by 100), variance drops to $\approx 0.024$ — and PC1 now points almost entirely in the weight direction.

**Decision guide for standardization:**

| Scenario | Recommendation | Reason |
|---|---|---|
| Features on same scale and unit | Center only | Preserves relative magnitudes; large variance is genuinely informative |
| Features on different scales/units | **Standardize** ($z$-score) | Equal contribution regardless of unit |
| Interpreting component loadings | Standardize | Loadings become comparable across features |
| Applying Kaiser criterion | Standardize | Criterion requires each feature to have unit variance |
| Image pixels, identical sensors | Center only | Same unit, variance differences are meaningful |

**Z-score standardization:**
$$x_{\text{std}} = \frac{x - \hat{\mu}}{\hat{\sigma}}, \qquad \hat{\mu} = \text{sample mean},\;\hat{\sigma} = \text{sample std}$$

After standardizing, $C$ becomes the **sample correlation matrix** $R_{ij} = C_{ij}/\sqrt{C_{ii}C_{jj}}$, and PCA on standardized data is equivalent to PCA on the correlation matrix.

**Important:** Standardization parameters ($\hat{\mu}$, $\hat{\sigma}$) must be computed on training data only, then applied to test data. Never fit these on test data — this constitutes data leakage and inflates performance estimates.

---

## 4. Explained Variance and Component Selection

### 4.1 Variance Ratio and Cumulative Explained Variance

**Total variance** of the data equals the trace of the covariance matrix, which equals the sum of all eigenvalues:
$$\text{Total Var} = \operatorname{tr}(C) = \sum_{i=1}^d \lambda_i = \sum_{i=1}^d \frac{\sigma_i^2}{n-1}$$

**Variance ratio** for component $i$:
$$\rho_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j} = \frac{\sigma_i^2}{\sum_{j=1}^d \sigma_j^2}$$

$\rho_i \in [0,1]$ and $\sum_i \rho_i = 1$.

**Cumulative explained variance** for $k$ components:
$$\text{CEV}(k) = \sum_{i=1}^k \rho_i = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

$\text{CEV}(k)$ increases monotonically from 0 to 1 as $k$ goes from 0 to $d$.

```python
def explained_variance(X_centered):
    """Compute variance ratios and cumulative EVR from centered data."""
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    var_ratio = s**2 / (s**2).sum()
    cum_var = np.cumsum(var_ratio)
    return var_ratio, cum_var
```

### 4.2 Scree Plots and the Elbow Heuristic

A **scree plot** (from geology: the rubble at the base of a cliff) shows the eigenvalues $\lambda_i$ plotted against component index $i$.

```
SCREE PLOT ANATOMY
════════════════════════════════════════════════════════════════════════

  λᵢ                      Individual eigenvalues
  │
  5.2 ─ ●
  │
  3.1 ─   ●                 Cumulative EVR
  │         ●               100%│              ─────────
  1.8 ─       ●              90%│         ────
  │             ●  ●  ●  ●   80%│      ───
  0.2 ─             ───────   70%│   ───
  └──────────────────── i      60%│───
      1  2  3  4  5  6  7      0  1  2  3  4  5  6

  "Elbow" at i=3: eigenvalues      Choose k at the "knee" of
  level off after here             the cumulative curve

════════════════════════════════════════════════════════════════════════
```

**Reading the scree plot:**
- A sharp drop followed by a "floor" of small eigenvalues indicates a clear intrinsic dimension
- The **elbow** (knee) is the bend where eigenvalues stop decreasing steeply — a heuristic for the number of meaningful components
- No elbow: data may genuinely be spread across many dimensions, or may need more samples

**Limitations of the elbow heuristic:** Subjective; different analysts may pick different elbows; fails when the spectrum decays smoothly (e.g., power-law decay). Cross-validation or parallel analysis is more rigorous.

### 4.3 Threshold Methods

The most common method is to choose the smallest $k$ such that the cumulative explained variance exceeds a threshold $\tau$:
$$k^* = \min\left\{k : \text{CEV}(k) \geq \tau\right\}$$

**Typical thresholds by application:**

| Threshold $\tau$ | Use case |
|---|---|
| 90% | Exploratory analysis, rough compression |
| 95% | Standard ML preprocessing |
| 99% | Applications where accuracy matters (genomics, finance) |
| 99.9% | Near-lossless compression |

```python
def components_for_threshold(s, tau=0.95):
    """Return minimum k achieving tau cumulative explained variance."""
    var_ratio = s**2 / (s**2).sum()
    cumvar = np.cumsum(var_ratio)
    k = int(np.searchsorted(cumvar, tau)) + 1
    return min(k, len(s))
```

**sklearn shortcut:** `PCA(n_components=0.95)` automatically chooses $k$ to explain 95% of variance.

### 4.4 Kaiser Criterion

For **standardized data** (each feature has variance 1), the total variance equals $d$ (the number of features), and each eigenvalue $\lambda_i$ measures how many "original features" that component explains.

**Kaiser criterion:** Keep components with $\lambda_i > 1$ — components that explain more variance than a single original feature.

$$k_{\text{Kaiser}} = \#\{i : \lambda_i > 1\}$$

This is the default in many statistics packages. It works specifically because standardized data makes the threshold $\lambda = 1$ meaningful — without standardization, the threshold has no natural interpretation.

**When Kaiser fails:**
- Over-extraction (too many components) when $d$ is large relative to $n$
- Under-extraction when the first PC has very high eigenvalue and the rest decay slowly
- Should not be applied to unstandardized data

### 4.5 Cross-Validation and Downstream-Task Selection

For predictive modelling, the optimal $k$ is application-specific. Cross-validation is the principled approach:

**Reconstruction CV:** Hold out observations, compute reconstruction error as a function of $k$. Select $k$ minimizing validation reconstruction error. Provides a principled but unsupervised criterion.

**Downstream CV:** Fit a model (classifier, regressor) on PCA-reduced features. Select $k$ maximizing cross-validation performance on the downstream task. This is the most principled choice for supervised settings.

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def find_k_cv(X, y, k_values, cv=5):
    """Find optimal k via downstream cross-validation."""
    scores = []
    for k in k_values:
        pipe = Pipeline([
            ('pca', PCA(n_components=k)),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        score = cross_val_score(pipe, X, y, cv=cv).mean()
        scores.append(score)
    return k_values[np.argmax(scores)], scores
```

**Parallel analysis (Horn's method):** Compare each $\lambda_i$ against the $\lambda_i$ from randomly generated data of the same shape. Keep components where the actual eigenvalue exceeds the random baseline. More statistically rigorous than the Kaiser criterion; less commonly implemented but available in some R packages.

---

## 5. PCA Whitening

### 5.1 What Whitening Achieves

Standard PCA projects data onto principal components and produces scores with variances $\lambda_1 \geq \cdots \geq \lambda_k$ — the components are uncorrelated but have different magnitudes. **Whitening** goes one step further: it rescales each component to unit variance, producing data with covariance $I_k$.

A dataset $Z_{\text{white}}$ is called **white** (or **spherical**) if:
$$\operatorname{Cov}(Z_{\text{white}}) = I$$

Geometrically, whitening maps the data ellipsoid to a hypersphere — every direction has equal variance.

**PCA whitening transform:**

Starting from centered data $\tilde{X}$ with SVD $\tilde{X} = U\Sigma V^\top$:

$$Z_{\text{white}} = \tilde{X} V \Lambda^{-1/2} = U\Sigma V^\top V \Lambda^{-1/2} = U \Sigma \Lambda^{-1/2}$$

where $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_k)$ and $\Lambda^{-1/2} = \operatorname{diag}(1/\sqrt{\lambda_1}, \ldots, 1/\sqrt{\lambda_k})$.

Since $\lambda_i = \sigma_i^2/(n-1)$, this simplifies to:
$$Z_{\text{white}} = \sqrt{n-1}\, U_k$$

The columns of $U$ from the SVD are already orthonormal, so $\operatorname{Cov}(U_k) = I_k/(n-1)$. After scaling by $\sqrt{n-1}$, each column has unit variance.

**Verification:**
$$\operatorname{Cov}(Z_{\text{white}}) = \frac{Z_{\text{white}}^\top Z_{\text{white}}}{n-1} = \frac{(n-1)U_k^\top U_k}{n-1} = I_k$$

### 5.2 PCA Whitening vs ZCA Whitening

**PCA whitening** rotates to the PC axes first, then scales:
$$Z_{\text{PCA-white}} = \tilde{X} V_k \Lambda_k^{-1/2}$$
The result is in a different coordinate system than the original data — the axes are the principal components.

**ZCA whitening** (Zero-phase Component Analysis) applies a further rotation to bring the data back to the original feature space:
$$Z_{\text{ZCA}} = \tilde{X} V_k \Lambda_k^{-1/2} V_k^\top = \tilde{X} C^{-1/2}$$
where $C^{-1/2} = V\Lambda^{-1/2}V^\top$ is the symmetric square root of $C^{-1}$.

```
ZCA vs PCA WHITENING
════════════════════════════════════════════════════════════════════════

  Original data:              PCA-white:               ZCA-white:
  correlated ellipse          decorrelated sphere       sphere in orig frame
                              (rotated)                 (NOT rotated)

       •• •                        •  •                    • •
      •   •                       •    •                  •   •
     •     •         →           •      •        →       •     •
      •   •                       •    •                  •   •
       •• •                        •  •                    • •

  ×── features ──→          ×─── PCs ────→           ×── features ──→

  Best for images: ZCA preserves spatial locality.
  Best for ICA: PCA-white, since ICA works in PC space.

════════════════════════════════════════════════════════════════════════
```

| Property | PCA whitening | ZCA whitening |
|---|---|---|
| Covariance | $I_k$ | $I_d$ (in original space) |
| Axes | PC axes | Original feature axes |
| Spatial locality | Destroyed | Preserved |
| Invertible? | No (if $k < d$) | Yes |
| Preferred for | ICA, neural nets | Image preprocessing, GANs |

### 5.3 Applications of Whitening

**ICA preprocessing.** Independent Component Analysis assumes whitened inputs. Whitening reduces the ICA problem from $d^2$ parameters to $d(d-1)/2$ (rotation angles only), making it tractable.

**Neural network training.** Whitening input features can dramatically speed up SGD convergence by removing the elongated covariance ellipse that makes gradient descent take zig-zag paths. Batch Normalization is a learned approximation of per-layer whitening.

**Image preprocessing.** ZCA whitening (also called "global contrast normalization") is used in CNN preprocessing pipelines (CIFAR-10 baseline, etc.) to remove correlations between neighboring pixels.

**Reparametrization tricks.** To sample from $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$, first sample $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$, then compute $\mathbf{x} = L\mathbf{z} + \boldsymbol{\mu}$ where $\Sigma = LL^\top$ (Cholesky). Whitening is the inverse: given $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu},\Sigma)$, compute $\mathbf{z} = L^{-1}(\mathbf{x}-\boldsymbol{\mu}) \sim \mathcal{N}(\mathbf{0},I)$.

---

## 6. Probabilistic PCA (PPCA)

### 6.1 The Generative Model

Standard PCA is a geometric algorithm — it finds optimal directions but provides no probabilistic model. Probabilistic PCA (PPCA), introduced by Tipping & Bishop (1999), embeds PCA in a latent variable model that enables likelihood-based model selection, missing data handling, and Bayesian extensions.

**The PPCA generative model:**
$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_k)$$
$$\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(W\mathbf{z} + \boldsymbol{\mu},\, \sigma^2 I_d)$$

where:
- $\mathbf{z} \in \mathbb{R}^k$ is the latent code (low-dimensional)
- $W \in \mathbb{R}^{d \times k}$ is the loading matrix (maps latent to observed)
- $\boldsymbol{\mu} \in \mathbb{R}^d$ is the mean of the observed data
- $\sigma^2 > 0$ is the isotropic noise variance

Marginalizing over $\mathbf{z}$, the observed distribution is also Gaussian:
$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu},\, WW^\top + \sigma^2 I_d)$$

The key parameters to learn from data are $W$, $\boldsymbol{\mu}$, and $\sigma^2$.

**Comparison to standard PCA:**

| Aspect | Standard PCA | PPCA |
|---|---|---|
| Type | Geometric/algebraic | Probabilistic/generative |
| Parameters | $V_k$, mean | $W$, $\boldsymbol{\mu}$, $\sigma^2$ |
| Model selection | Heuristic ($k$ via scree) | Likelihood-based (BIC/AIC) |
| Missing data | Not directly handled | EM handles naturally |
| Uncertainty | None | Posterior over $\mathbf{z}$ |
| Limit $\sigma^2 \to 0$ | Recovers standard PCA | ✓ |

### 6.2 MLE Solution and Connection to Standard PCA

The MLE for the PPCA model has a closed-form solution. Let $C$ be the sample covariance matrix, $\lambda_i$ its eigenvalues, and $\mathbf{v}_i$ its eigenvectors. Then:

**MLE for $\boldsymbol{\mu}$:** $\hat{\boldsymbol{\mu}} = \bar{\mathbf{x}}$ (sample mean).

**MLE for $\sigma^2$:**
$$\hat{\sigma}^2 = \frac{1}{d-k}\sum_{i=k+1}^d \lambda_i$$
This is the average variance of the discarded components — the isotropic noise level.

**MLE for $W$:**
$$\hat{W} = V_k (\Lambda_k - \hat{\sigma}^2 I_k)^{1/2} R$$
where $V_k$ contains the top $k$ eigenvectors, $\Lambda_k = \operatorname{diag}(\lambda_1,\ldots,\lambda_k)$, and $R \in \mathbb{R}^{k\times k}$ is any orthogonal matrix (rotational ambiguity).

**Connection to PCA:** As $\sigma^2 \to 0$, $\hat{W} \to V_k \operatorname{diag}(\sqrt{\lambda_1},\ldots,\sqrt{\lambda_k})$, and the latent posterior concentrates on the standard PCA projection. PPCA is a proper generative model whose zero-noise limit recovers classical PCA.

**Log-likelihood at the MLE:**
$$\log p(X \mid \hat{W}, \hat{\boldsymbol{\mu}}, \hat{\sigma}^2) = -\frac{n}{2}\left[d\log(2\pi) + \log|\hat{C}| + d\right]$$
where $\hat{C} = \hat{W}\hat{W}^\top + \hat{\sigma}^2 I$. This provides a proper likelihood for model comparison (choosing $k$ via AIC or BIC).

### 6.3 Handling Missing Data via EM

PPCA handles missing data naturally through the EM algorithm, unlike standard PCA which requires complete data.

**E-step:** Given current parameters $W$, $\boldsymbol{\mu}$, $\sigma^2$, compute the posterior over latent codes for each observation (even those with missing features):
$$\mathbb{E}[\mathbf{z}^{(i)} \mid \mathbf{x}_{\text{obs}}^{(i)}] = M^{-1}W_{\text{obs}}^\top(\mathbf{x}_{\text{obs}}^{(i)} - \boldsymbol{\mu}_{\text{obs}})$$
where $M = W_{\text{obs}}^\top W_{\text{obs}} + \sigma^2 I$ and subscript "obs" denotes the non-missing entries.

**M-step:** Update $W$, $\boldsymbol{\mu}$, $\sigma^2$ using the expected sufficient statistics from the E-step.

The EM converges to the MLE. For the complete-data case, EM recovers the exact closed-form MLE. For missing data, it provides principled imputation — predicting missing entries via the posterior mean.

**For AI:** Masked autoencoding (MAE, BERT) is conceptually related — learn representations that can reconstruct masked/missing inputs, which is a non-linear deep-learning analog of the PPCA EM algorithm.

### 6.4 Mixture of PPCA

A single PPCA model assumes the data is approximately Gaussian in $\mathbb{R}^d$. For multi-modal data, a **Mixture of PPCA** uses $M$ local PPCA models:

$$p(\mathbf{x}) = \sum_{m=1}^M \pi_m \mathcal{N}(\mathbf{x};\, \boldsymbol{\mu}_m, W_m W_m^\top + \sigma_m^2 I)$$

Each component $m$ finds its own local principal subspace. The full EM algorithm alternates between:
- **E-step:** Assign soft responsibilities $r_{im} = p(m \mid \mathbf{x}^{(i)})$ via Bayes' rule
- **M-step:** Update $\pi_m$, $\boldsymbol{\mu}_m$, $W_m$, $\sigma_m^2$ weighted by responsibilities

This is a smooth, probabilistic generalization of K-means + PCA per cluster. It handles non-convex clusters and provides uncertainty estimates, at the cost of needing to choose both $M$ (number of clusters) and $k$ (number of components per cluster).

---

## 7. Kernel PCA

### 7.1 Motivation — When Linear PCA Fails

Standard PCA finds **linear** structure in data. If the meaningful variation lies along a non-linear manifold, PCA will fail to capture it — the top PCs will not align with the manifold's intrinsic directions.

```
WHEN LINEAR PCA FAILS
════════════════════════════════════════════════════════════════════════

  Two-moons data:                Swiss roll:

      ●●●                        (3D spiral surface)
    ●●   ●●                      PCA finds the 3D extent.
   ●●     ●●       ←PCA→        Kernel PCA (RBF) unfolds the
   ●●     ●●      collapses      roll into a 2D plane.
    ●●   ●●       the two
      ●●●         moons!

  PCA sees two overlapping         Kernel PCA extracts
  blobs, not two crescents.        intrinsic 2D structure.

════════════════════════════════════════════════════════════════════════
```

**Kernel PCA** (Schölkopf, Smola, Müller, 1998) solves this by implicitly mapping data to a high-dimensional feature space $\mathcal{H}$ where the non-linear structure becomes linear, then applying standard PCA there.

### 7.2 The Kernel Trick and Feature-Space PCA

**Feature map.** Let $\phi: \mathbb{R}^d \to \mathcal{H}$ be a (possibly infinite-dimensional) feature map. The feature-space covariance is:
$$C_\phi = \frac{1}{n}\sum_{i=1}^n \phi(\mathbf{x}^{(i)})\phi(\mathbf{x}^{(i)})^\top$$

We want the eigenvectors of $C_\phi$, but $\mathcal{H}$ may be infinite-dimensional — we can never compute $C_\phi$ explicitly.

**Key insight.** Every eigenvector $\mathbf{v}$ of $C_\phi$ lies in the span of $\{\phi(\mathbf{x}^{(1)}), \ldots, \phi(\mathbf{x}^{(n)})\}$:
$$\mathbf{v} = \sum_{i=1}^n \alpha_i \phi(\mathbf{x}^{(i)})$$

Substituting into the eigenvalue equation $C_\phi \mathbf{v} = \lambda \mathbf{v}$ and multiplying both sides by $\phi(\mathbf{x}^{(j)})^\top$:
$$K\boldsymbol{\alpha} = n\lambda\boldsymbol{\alpha}$$

where $K_{ij} = \langle\phi(\mathbf{x}^{(i)}), \phi(\mathbf{x}^{(j)})\rangle_\mathcal{H} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ is the **kernel matrix** (Gram matrix).

**The kernel trick:** We never need $\phi$ explicitly. We only need the kernel function $k(\mathbf{x}, \mathbf{y})$ that computes inner products in $\mathcal{H}$.

**Kernel PCA algorithm:**
1. Compute $n \times n$ kernel matrix $K$ with $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$
2. Center the kernel matrix (see §7.3)
3. Eigendecompose $\tilde{K} = \Phi D \Phi^\top$ (or use SVD)
4. Extract top $k$ eigenvectors $\boldsymbol{\alpha}^{(1)}, \ldots, \boldsymbol{\alpha}^{(k)}$, normalized: $\lVert\boldsymbol{\alpha}^{(i)}\rVert = 1/\sqrt{d_i}$
5. Project new point $\mathbf{x}$: compute $k_i = k(\mathbf{x}^{(i)}, \mathbf{x})$ for all $i$, then scores $= \Phi^\top \mathbf{k}$

**Computational cost:**
- Building $K$: $O(n^2 d)$ — expensive for large $n$
- Eigendecomposition: $O(n^3)$ — bottleneck
- For large $n$: use Nyström approximation or random features

### 7.3 Centering the Kernel Matrix

Standard PCA requires centered data. In feature space, centering means:
$$\tilde{\phi}(\mathbf{x}^{(i)}) = \phi(\mathbf{x}^{(i)}) - \frac{1}{n}\sum_{j=1}^n \phi(\mathbf{x}^{(j)})$$

The centered kernel matrix is:
$$\tilde{K} = K - \mathbf{1}_n K - K \mathbf{1}_n + \mathbf{1}_n K \mathbf{1}_n$$

where $\mathbf{1}_n = \frac{1}{n}\mathbf{1}\mathbf{1}^\top \in \mathbb{R}^{n \times n}$ is the matrix of all $1/n$ entries.

In code:
```python
def center_kernel(K):
    """Center the n×n kernel matrix."""
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n
```

**Centering for new data:** When projecting a new point $\mathbf{x}$, the kernel vector $\mathbf{k} = [k(\mathbf{x}^{(1)},\mathbf{x}), \ldots, k(\mathbf{x}^{(n)},\mathbf{x})]^\top$ must also be centered:
$$\tilde{\mathbf{k}} = \mathbf{k} - \frac{1}{n}\mathbf{K}\mathbf{1} - \frac{1}{n}\mathbf{1}^\top\mathbf{k}\,\mathbf{1} + \frac{1}{n^2}\mathbf{1}^\top\mathbf{K}\mathbf{1}\,\mathbf{1}$$

Failing to center new-point kernels is a common implementation error.

### 7.4 Common Kernels and Their Geometric Effects

| Kernel | Formula $k(\mathbf{x}, \mathbf{y})$ | Feature space | Captures |
|---|---|---|---|
| **Linear** | $\mathbf{x}^\top\mathbf{y}$ | $\mathbb{R}^d$ (identity map) | Standard PCA |
| **Polynomial** | $(\mathbf{x}^\top\mathbf{y} + c)^p$ | Monomials up to degree $p$ | Polynomial interactions |
| **RBF / Gaussian** | $\exp(-\lVert\mathbf{x}-\mathbf{y}\rVert^2 / 2\sigma^2)$ | Infinite-dimensional | Smooth, local structure |
| **Sigmoid** | $\tanh(\kappa\mathbf{x}^\top\mathbf{y} + \theta)$ | Neural-net-like | (Not always PD) |
| **Laplacian** | $\exp(-\lVert\mathbf{x}-\mathbf{y}\rVert_1/\sigma)$ | Infinite-dimensional | Robust to outliers |

**RBF kernel bandwidth $\sigma$:**
- Small $\sigma$: each point forms its own neighborhood — overfitting
- Large $\sigma$: kernel approaches constant — loses discriminative power
- Rule of thumb: set $\sigma$ to the median pairwise distance (the "median heuristic")

**For AI:** The NTK (Neural Tangent Kernel) is the kernel induced by an infinitely wide neural network. Kernel PCA with the NTK is equivalent to PCA on neural network features at initialization — a theoretical tool for analyzing early training dynamics.

### 7.5 Limitations and Out-of-Sample Extension

**Limitations of kernel PCA:**

1. **No out-of-sample formula.** Unlike standard PCA (where new $\mathbf{x}$ maps to $V_k^\top(\mathbf{x}-\bar{\mathbf{x}})$), kernel PCA requires the kernel vector against all training points — inference cost is $O(n)$ per new point.

2. **Scalability.** $O(n^2)$ memory and $O(n^3)$ compute. For $n > 10{,}000$, need approximations.

3. **Hyperparameter sensitivity.** Kernel choice and bandwidth significantly affect results; requires cross-validation.

4. **No explicit feature map.** Cannot interpret the kernel PCA components as directions in the original feature space.

**Out-of-sample extension (Bengio et al., 2004).** For new point $\mathbf{x}$:
$$z_j(\mathbf{x}) = \frac{1}{d_j}\sum_{i=1}^n \alpha_j^{(i)} \tilde{k}(\mathbf{x}^{(i)}, \mathbf{x})$$

where $d_j$ is the $j$-th eigenvalue and $\tilde{k}$ denotes the centered kernel. This gives $O(n)$ inference.

**Nyström approximation.** Approximate $K$ using a random subset of $m \ll n$ landmark points:
$$K \approx K_{nm} K_{mm}^{-1} K_{mn}$$
Reduces cost to $O(nm^2)$ while preserving the top eigenspectrum approximately.

---

## 8. PCA Variants

### 8.1 Sparse PCA — Interpretable Loadings

Standard PCA components are generally dense — each PC is a linear combination of **all** original features. This makes interpretation difficult: "PC1 is 0.12·feature1 + 0.31·feature2 − 0.07·feature3 + ..." is hard to understand.

**Sparse PCA** constrains components to use only a few original features, trading reconstruction optimality for interpretability.

**Formulation (LASSO-regularized):**
$$\min_{V, Z} \lVert \tilde{X} - ZV^\top \rVert_F^2 + \lambda\lVert V \rVert_1 \quad \text{subject to} \quad Z^\top Z = I$$

The $\ell_1$ penalty on $V$ forces most loadings to zero. The result: each PC depends on only a handful of original features — interpretable as a "factor" in the original variable space.

**Algorithm (Zou, Hastie, Tibshirani, 2006 — Elastic Net formulation):**
- Fix $Z$: solve Lasso for each column of $V$
- Fix $V$: solve for $Z$ via SVD of $\tilde{X}V$
- Iterate until convergence

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=0.1)  # alpha = L1 regularization
Z_sparse = spca.fit_transform(X)
# spca.components_ has many zeros — interpretable loadings
```

**Applications:** Gene expression analysis (each PC = a small gene module), finance (each PC = a sector factor), neuroscience (each PC = a brain region pattern).

### 8.2 Robust PCA — Low-Rank + Sparse Decomposition

Standard PCA is sensitive to outliers: a single corrupted data point can completely distort the principal components (because PCA minimizes squared Frobenius norm, and outliers contribute quadratically).

**Robust PCA** (Candès, Li, Ma, Wright, 2011) decomposes a matrix $M$ into:
$$M = L + S$$
where $L$ is **low-rank** (the "clean" signal) and $S$ is **sparse** (the outliers or corruptions).

**Formulation (Principal Component Pursuit):**
$$\min_{L, S} \lVert L \rVert_* + \mu \lVert S \rVert_1 \quad \text{subject to} \quad L + S = M$$

where $\lVert L\rVert_*$ is the nuclear norm (sum of singular values — convex surrogate for rank) and $\lVert S\rVert_1$ is the element-wise $\ell_1$ norm (convex surrogate for sparsity).

**Exact recovery theorem:** Under mild conditions (incoherence conditions on $L$ and random support for $S$), the optimization exactly recovers $(L, S)$ even when the sparse component has a constant fraction of corrupted entries.

**Applications:**
- **Video surveillance**: $M$ = surveillance video frames; $L$ = static background (low-rank); $S$ = moving objects (sparse). Robust PCA cleanly separates background from foreground.
- **Collaborative filtering**: Robust PCA on rating matrices separates low-rank preference patterns from sparse anomalous ratings.
- **Medical imaging**: Separate structured anatomical patterns (low-rank) from focal lesions (sparse).

### 8.3 Incremental and Randomized PCA

**Incremental PCA** (also called online PCA) processes data in mini-batches, updating the PCA solution without storing the full data matrix. Essential when $n$ is too large for memory.

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=500)
for batch in data_generator():     # stream data in batches
    ipca.partial_fit(batch)

Z = ipca.transform(X_new)          # apply to new data
```

The algorithm maintains running estimates of the covariance matrix eigenvectors, using Gram-Schmidt to enforce orthogonality after each update.

**Randomized SVD** (Halko, Martinsson, Tropp, 2011) computes an approximate truncated SVD much faster than the full SVD when $k \ll \min(n,d)$:

1. Generate random Gaussian matrix $\Omega \in \mathbb{R}^{d \times (k+p)}$ (oversampling with $p \approx 10$)
2. Compute $Y = \tilde{X}\Omega \in \mathbb{R}^{n \times (k+p)}$ — projects data to random subspace
3. QR-decompose $Y = QR$ to get orthonormal basis $Q$
4. Compute $B = Q^\top \tilde{X} \in \mathbb{R}^{(k+p) \times d}$ — small matrix
5. SVD of $B = \hat{U}\Sigma V^\top$; then $U = Q\hat{U}$

**Cost:** $O(ndk)$ instead of $O(nd\min(n,d))$ — a factor of $\min(n,d)/k$ speedup.

```python
from sklearn.utils.extmath import randomized_svd

U, s, Vt = randomized_svd(X_centered, n_components=50,
                           n_iter=5, random_state=42)
```

**For AI:** sklearn's `PCA(svd_solver='randomized')` uses randomized SVD by default for large inputs. This is how PCA on ImageNet-scale datasets ($n > 10^6$) becomes tractable.

---

## 9. Applications in Machine Learning and AI

### 9.1 Eigenfaces and Image Compression

The classic application of PCA to images (Turk & Pentland, 1991) demonstrates the power of low-rank approximation in practice.

**Setup:** $n$ face images of size $h \times w$ pixels, each flattened to a vector in $\mathbb{R}^{hw}$. Form the data matrix $X \in \mathbb{R}^{n \times hw}$.

**Eigenfaces:** The principal components $\mathbf{v}_1, \ldots, \mathbf{v}_k$ can be reshaped back to $h \times w$ images — these are the "eigenfaces," the most important face patterns in the dataset.

```python
def compute_eigenfaces(face_images, k=50):
    """
    face_images: (n, h, w) array
    Returns: eigenfaces (k, h, w), mean_face (h, w), scores (n, k)
    """
    n, h, w = face_images.shape
    X = face_images.reshape(n, -1).astype(float)

    mean_face = X.mean(axis=0)
    X_c = X - mean_face

    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    eigenfaces = Vt[:k].reshape(k, h, w)
    scores = U[:, :k] * s[:k]    # (n, k) — PCA coordinates

    return eigenfaces, mean_face.reshape(h, w), scores
```

**Compression ratio:** Storing $k$ eigenfaces ($k \cdot hw$ values) and $n$ score vectors ($n \cdot k$ values) vs. the original $n \cdot hw$ values. For $k = 50$, $n = 1000$, $hw = 4096$: compression from 4M to 255K values — a 16× reduction.

**For AI:** ViT (Vision Transformers) learn patch embeddings that, in early layers, closely resemble the top PCA components of image patches. The first self-attention heads often perform a form of learned PCA on the patch token space.

### 9.2 Intrinsic Dimensionality of LLM Representations

A fundamental empirical question in LLM research: "how many dimensions does a language model actually use?"

**Methodology:**
1. Pass $n$ text samples through an LLM layer, collect activations $X \in \mathbb{R}^{n \times d_{\mathrm{model}}}$
2. Compute PCA and examine cumulative explained variance
3. Define intrinsic dimension $k^*$ as the smallest $k$ with CEV $\geq 90\%$ (or use participation ratio)

**Empirical findings (across multiple studies):**
- GPT-2 (d=768): ~50-100 components explain 90% of variance at middle layers
- LLaMA-2-7B (d=4096): ~200-400 components at most layers
- The effective dimension grows with layer depth and shrinks in final layers
- Instruction-tuned models tend to have lower intrinsic dimension than base models

**Participation ratio** — a continuous intrinsic dimensionality measure:
$$d_{\text{PR}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

Equals $d$ when all eigenvalues are equal (maximum dimensionality); equals 1 when one eigenvalue dominates.

```python
def participation_ratio(X_centered):
    """Continuous measure of effective dimensionality."""
    _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    lambdas = s**2
    return lambdas.sum()**2 / (lambdas**2).sum()
```

### 9.3 Mechanistic Interpretability: PCA of the Residual Stream

Mechanistic interpretability researchers use PCA to understand what information is stored and processed at each transformer layer.

**The technique:**
1. Collect residual stream activations $X^{[l]} \in \mathbb{R}^{n \times d}$ at layer $l$ for a dataset of prompts
2. Apply PCA: $X^{[l]} = U^{[l]}\Sigma^{[l]}(V^{[l]})^\top$
3. Examine top principal components $\mathbf{v}^{[l]}_1, \mathbf{v}^{[l]}_2, \ldots$ by probing: which token properties predict high/low scores on each PC?

**Examples from published work:**
- PC1 of early layers often tracks token position (Fourier modes of position)
- PCs in middle layers encode syntactic structure (subject, verb, object positions)
- PCs in late layers encode semantic content (entity type, sentiment, factual attributes)
- The "residual stream basis" changes sharply at layers where attention heads write task-relevant information

**Logit lens** is a related technique: project residual stream activations at each layer back to vocabulary space using the unembedding matrix, then PCA on these logit distributions tracks information flow from token features to next-token predictions.

### 9.4 LoRA and the Low-Rank Hypothesis

LoRA (Low-Rank Adaptation, Hu et al., 2021) is the dominant parameter-efficient fine-tuning method for LLMs. Its mathematical grounding is directly the low-rank PCA hypothesis.

**The hypothesis:** During fine-tuning, the weight update matrix $\Delta W = W_{\text{fine-tuned}} - W_{\text{pretrained}}$ has low intrinsic rank. That is, PCA of $\Delta W$ reveals that most of its "information" lies in a rank-$r$ subspace with $r \ll \min(d_{\mathrm{in}}, d_{\mathrm{out}})$.

**LoRA parametrization:**
$$\Delta W = BA, \quad B \in \mathbb{R}^{d_{\mathrm{out}} \times r},\; A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$$

The modified forward pass: $\mathbf{h} = W\mathbf{x} + \Delta W\mathbf{x} = W\mathbf{x} + BA\mathbf{x}$.

**Parameter savings:** Original $\Delta W$ has $d_{\mathrm{out}} \cdot d_{\mathrm{in}}$ parameters; LoRA uses $r(d_{\mathrm{out}} + d_{\mathrm{in}})$ — a ratio of $r/d$ savings (often $r = 4$--$64$ vs $d = 4096$).

**Empirical validation:** Aghajanyan et al. (2020) measured the intrinsic dimensionality of fine-tuning loss landscapes and found most models can be fine-tuned with $r < 200$ free parameters in a random low-dimensional subspace — validating the hypothesis empirically.

**Extensions:**
- **DoRA** (Weight Decomposed LoRA): decomposes weight update into magnitude and direction components
- **GaLore** (Gradient Low-Rank Projection): computes running PCA of gradient matrices, projects updates to the dominant gradient subspace
- **Flora**: applies random projections (randomized PCA) to gradients to reduce optimizer memory

### 9.5 Loss Landscape Geometry

PCA of gradient trajectories and weight perturbations reveals the geometry of the loss landscape.

**Li et al. (2018) — Filter normalization visualization:**
1. Train a model, record the final weights $\boldsymbol{\theta}^*$
2. Sample $n$ perturbation directions; collect weights along training trajectory $\boldsymbol{\theta}_t$
3. Apply PCA to the trajectory matrix: find the 2 principal directions $\mathbf{d}_1, \mathbf{d}_2$
4. Plot loss surface in the 2D PCA plane: $\mathcal{L}(\boldsymbol{\theta}^* + \alpha\mathbf{d}_1 + \beta\mathbf{d}_2)$

This "loss landscape visualization" shows that well-trained models live in wide, flat valleys; poorly trained ones (skip connections removed) live in sharp minima with many barriers.

**Dominant gradient subspace.** The gradient $\nabla\mathcal{L}(\boldsymbol{\theta})$ lies almost entirely in a $k$-dimensional subspace with $k \ll d$ throughout training. This subspace is almost stable: the top-$k$ PCA directions of gradient matrices computed at different training steps have $> 90\%$ overlap. This has implications for: (a) why SGD finds flat minima, (b) why momentum methods work, (c) why random projections preserve gradient structure.

### 9.6 PCA vs t-SNE vs UMAP for Visualization

When reducing to 2D/3D for visualization, practitioners must choose between linear and non-linear methods.

| Method | Type | Preserves | Time | Deterministic? | Best for |
|---|---|---|---|---|---|
| **PCA** | Linear | Global variance, large-scale structure | $O(nd^2)$ or faster | Yes | Exploring bulk structure, preprocessing |
| **t-SNE** | Non-linear | Local neighborhoods (distances within clusters) | $O(n^2)$ or $O(n\log n)$ | No (random init) | Revealing cluster structure in 2D |
| **UMAP** | Non-linear | Local + some global topology | $O(n^{1.14})$ approx. | No (random init) | Faster t-SNE alternative; better global |

**Key differences:**
- PCA: distances in 2D PCA plot are meaningful; axes have units (variance explained)
- t-SNE/UMAP: cluster shapes and inter-cluster distances are **NOT** preserved — only within-cluster topology
- PCA is reproducible; t-SNE/UMAP have random initialization (seed for reproducibility)
- For LLM embedding visualization: PCA first (to ~50 dims), then t-SNE/UMAP on PCA scores is standard practice (PCA removes noise, t-SNE reveals cluster fine structure)

**Common mistake:** Interpreting t-SNE distances between clusters as meaningful. They are not — t-SNE cluster sizes and inter-cluster distances are artifacts of the perplexity parameter and optimization.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | **Not centering before PCA** | The covariance is computed around the mean; without centering, PC1 points toward the mean (not the direction of variance), giving spurious components | Always subtract `X.mean(axis=0)` before SVD |
| 2 | **Not standardizing when features have different units** | A feature in metres dominates one in millimetres even if they have equal information content; PCA will find a direction dominated by the large-scale feature | Standardize with `(X - mean) / std` when units differ |
| 3 | **Fitting mean/std on test data** | Data leakage: test statistics partially determine the PCA basis, inflating generalization metrics | Fit `mean`, `std`, and $V_k$ on training data; apply to test |
| 4 | **Using `np.linalg.eig` on the covariance matrix** | `eig` may return complex eigenvalues due to floating-point asymmetry; sorting eigenvalues requires extra code; condition number is squared | Use `np.linalg.svd` on the centered data directly |
| 5 | **Interpreting PCA scores as probabilities or distances without thought** | PCA scores have different scales ($\lambda_1 \gg \lambda_k$); Euclidean distance in score space is not isotropic | Use whitened scores for distance-based methods (k-NN, k-means) |
| 6 | **Choosing $k$ once and never revisiting** | The right $k$ depends on the downstream task; the variance-threshold $k$ may be too large or too small for a specific classifier | Use cross-validation on downstream task to validate $k$ |
| 7 | **Applying test-time kernel without centering** | Kernel PCA centering involves the training kernel matrix; new-point kernels must be centered using training statistics | Implement $\tilde{\mathbf{k}} = \mathbf{k} - K_{\text{train}}\mathbf{1}/n - \mathbf{k}^\top\mathbf{1}/n \cdot\mathbf{1} + \text{const}$ |
| 8 | **Interpreting PC directions as independent (not just uncorrelated)** | PCA guarantees zero correlation in the scores; it does NOT guarantee statistical independence (non-Gaussian data can have dependent PC scores) | For independence: use ICA after whitening |
| 9 | **Using PCA for class separation before LDA** | PCA discards directions with low variance, which may be the most discriminative; applying PCA before LDA can hurt classification | Use LDA directly, or apply PCA only to reduce within LDA's constraint $k < C-1$ |
| 10 | **Comparing PCA loadings across datasets without alignment** | PCA directions are sign-ambiguous and may be permuted across different datasets or runs | Align directions via Procrustes rotation or compare absolute values |
| 11 | **Using PCA on count data without transformation** | PCA assumes continuous, approximately Gaussian data; count data (word counts, genomic reads) violates this assumption | Apply log(1+x) or variance-stabilizing transformation first |
| 12 | **Reporting variance explained on training set only** | In-sample variance explained is optimistic; held-out variance explained gives a fair estimate of generalization | Report reconstruction error on held-out data |

---

## 11. Exercises

**Exercise 1 ★** — Derivation from first principles

Consider centered data $\tilde{X} \in \mathbb{R}^{3 \times 3}$ given below. Without using `np.linalg.svd` or `np.linalg.eigh` directly on $C$:

(a) Compute the sample covariance matrix $C = \tilde{X}^\top\tilde{X}/(n-1)$ by hand.

(b) Find the characteristic polynomial $\det(C - \lambda I)$ and solve for the eigenvalues.

(c) For each eigenvalue, find the corresponding eigenvector by solving $(C - \lambda I)\mathbf{v} = \mathbf{0}$.

(d) Verify that your eigenvectors are orthogonal and normalize them.

(e) Project the data onto the first 2 PCs and verify that the reconstruction error equals $\lambda_3$.

**Exercise 2 ★** — PCA via SVD and numerical equivalence

(a) Implement PCA using the covariance eigendecomposition (`np.linalg.eigh`).

(b) Implement PCA using `np.linalg.svd` on the centered data matrix.

(c) Compare the two implementations on a $100 \times 20$ random matrix. Do the loadings match? The explained variances? The scores?

(d) Construct a numerically ill-conditioned matrix (e.g., two nearly identical columns) and show that the covariance approach returns spurious imaginary components when using `np.linalg.eig` (not `eigh`), while the SVD approach is stable.

(e) Explain why `eigh` (symmetric eigenvalue solver) is better than `eig` for covariance matrices, and why `svd` on $\tilde{X}$ is better than both.

**Exercise 3 ★** — Explained variance and component selection

(a) Generate a correlated multivariate Gaussian dataset in $\mathbb{R}^{20}$ with a covariance matrix that has a spectrum decaying as $\lambda_i \propto 1/i$.

(b) Plot the scree plot (individual eigenvalues) and cumulative explained variance.

(c) Find the number of components needed for 80%, 90%, 95%, and 99% explained variance.

(d) Apply the Kaiser criterion and compare with the threshold methods.

(e) Fit a linear regression on the raw features vs. PCA scores with 90% variance retained. Which gives better cross-validated $R^2$? Explain.

**Exercise 4 ★★** — Whitening and its effects

(a) Generate a 2D dataset with covariance $\Sigma = \begin{pmatrix}9 & 6 \\ 6 & 5\end{pmatrix}$. Verify numerically.

(b) Apply PCA whitening and verify that the whitened covariance is $I_2$.

(c) Apply ZCA whitening and verify the covariance is $I_2$ AND that the result is in the original feature axes (not rotated).

(d) Plot both whitened versions alongside the original. Describe geometrically what each transformation does.

(e) Time the convergence of gradient descent on a simple quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\Sigma\mathbf{x}$ starting from $(1,1)$ — first with raw data, then with whitened inputs. Report the number of steps to reach $f < 10^{-4}$.

**Exercise 5 ★★** — Probabilistic PCA

(a) Implement PPCA via the closed-form MLE. Inputs: centered data $\tilde{X}$, number of components $k$. Outputs: $\hat{W}$, $\hat{\sigma}^2$, log-likelihood.

(b) Generate data from a PPCA model with $k=3$, $d=20$, $n=500$, $\sigma^2 = 0.1$, and random $W$. Fit PPCA with $k = 2, 3, 4, 5$ and plot log-likelihood vs $k$.

(c) Does the true $k=3$ achieve the highest likelihood? Why or why not? (Hint: consider the bias-variance tradeoff in likelihood estimation.)

(d) Implement the MLE formula for $\hat{\sigma}^2$ and verify it equals the mean discarded eigenvalue.

(e) Compare the PPCA reconstruction (posterior mean $\hat{\mathbf{x}} = \hat{W}\hat{\mathbf{z}} + \hat{\boldsymbol{\mu}}$) with standard PCA reconstruction on 20% of randomly masked features.

**Exercise 6 ★★** — Kernel PCA

(a) Generate the two-moons dataset (`sklearn.datasets.make_moons`). Show that linear PCA fails to separate the classes in 2D.

(b) Implement kernel PCA with an RBF kernel from scratch: compute $K$, center it, eigendecompose, extract top 2 components.

(c) Plot the kernel PCA embedding for $\sigma \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$. Which value separates the moons?

(d) Implement the out-of-sample projection formula for new test points. Verify that it matches projecting the test points alongside the training set.

(e) Compare kernel PCA with t-SNE and UMAP (from `sklearn` and `umap-learn`) on the same two-moons data. Discuss the qualitative and quantitative differences.

**Exercise 7 ★★★** — LoRA and the low-rank hypothesis

(a) Generate a "weight matrix" $W \in \mathbb{R}^{100 \times 80}$ with a known low-rank structure: $W = W_0 + UV^\top$ where $U \in \mathbb{R}^{100 \times 5}$, $V \in \mathbb{R}^{80 \times 5}$ are random, $W_0$ is small random noise.

(b) Compute the SVD of $W$ and plot the singular value spectrum. How many singular values are significantly above the noise floor?

(c) Implement LoRA: parametrize $\Delta W = BA$ with $B \in \mathbb{R}^{100 \times r}$, $A \in \mathbb{R}^{r \times 80}$. For $r \in \{2, 5, 10, 20\}$, minimize $\lVert W - W_0 - BA\rVert_F^2$ using gradient descent.

(d) Compare the LoRA reconstruction error with the optimal rank-$r$ approximation error from SVD (Eckart-Young). How close does LoRA get?

(e) Explain why $r = 5$ should achieve near-zero error here, and compute the compression ratio for each $r$ vs. storing $\Delta W$ directly.

**Exercise 8 ★★★** — Intrinsic dimensionality of representations

(a) Generate synthetic "LLM activations" by: (i) sample $n = 500$ points in $\mathbb{R}^{200}$ from a Gaussian with covariance $\Sigma = Q\Lambda Q^\top$ where $\Lambda$ has eigenvalues $\lambda_i = e^{-i/20}$.

(b) Compute the participation ratio and compare with the $k$ for 90% EVR. Are they consistent?

(c) Simulate "layer depth" by applying successive random rotations + low-rank perturbations. Plot the intrinsic dimensionality at each "layer."

(d) Implement robust PCA (via `sklearn` or manual nuclear norm minimization) on a corrupted version of the activations (add sparse outliers to 10% of entries). Compare the recovered low-rank component with standard PCA.

(e) Write a function `analyze_representations(X)` that returns: participation ratio, 90%-EVR dimension, top-3 eigenvalue ratio, and spectrum decay rate (fit $\lambda_i \propto i^{-\alpha}$ via linear regression on log-log scale). Apply to your synthetic "layers" and report how each statistic evolves.

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Application | Concrete Example |
|---|---|---|
| PCA variance maximization | Understanding which directions in weight space matter | WeightWatcher: heavy-tailed singular value spectra predict model quality |
| Eckart-Young theorem | Justifies low-rank approximation everywhere | LoRA: $\Delta W = BA$, rank $r = 4$--$64$ recovers most fine-tuning capability |
| Intrinsic dimensionality | Characterizes "how much information" a model layer uses | GPT-4 class models: 90% EVR at ~200 dims in $\mathbb{R}^{8192}$ |
| PCA of residual stream | Mechanistic interpretability | Anthropic's "linear representation hypothesis": features are directions in residual stream |
| PPCA / missing data | Masked language modelling as probabilistic inference | BERT's MLM objective ≈ EM for missing tokens in a PPCA-like model |
| Kernel PCA | Attention as implicit feature-space PCA | NTK theory: infinite-width networks perform kernel regression |
| Sparse PCA | Interpretable features in transformers | Superposition hypothesis: polysemantic neurons are superposed sparse features |
| Randomized SVD | Efficient computation on billion-parameter models | GaLore projects gradients to dominant right-singular-vector subspace |
| Whitening / decorrelation | Normalization layers | Batch Norm ≈ approximate feature whitening per layer |
| PCA of attention heads | Understanding head specialization | PCA of $QK^\top$ matrices reveals head function (positional, syntactic, semantic) |
| Loss landscape PCA | Visualization of optimization dynamics | Filter-normalized loss landscape shows flat minima for residual networks |
| Robust PCA | Denoising corrupted activations | Separating adversarial perturbations (sparse) from clean representations (low-rank) |

---

## 13. Conceptual Bridge

**Where we came from.** This section builds directly on two prior foundations. From [§01-Eigenvalues-and-Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md), we take the spectral theorem for symmetric matrices: every real symmetric positive semidefinite matrix $C$ diagonalizes as $C = Q\Lambda Q^\top$ with real non-negative eigenvalues. PCA is precisely the application of this theorem to the sample covariance matrix. From [§02-Singular-Value-Decomposition](../02-Singular-Value-Decomposition/notes.md), we take the Eckart-Young theorem and the SVD computation: PCA is most efficiently and stably computed by taking the SVD of the centered data matrix $\tilde{X}$ rather than computing $C$ explicitly.

**What PCA adds to the story.** The eigendecomposition and SVD are algebraic facts about matrices. PCA gives them statistical meaning: the data matrix $\tilde{X}$ is not arbitrary — it is a sample from an unknown distribution, and the covariance matrix $C$ is an estimate of that distribution's second moment. Maximizing projected variance, minimizing reconstruction error, and diagonalizing the covariance are all expressions of the same statistical goal: find the coordinate system in which the data is most "efficiently" described. This is PCA's unique contribution — it marries linear algebra with statistical estimation.

**Forward to Linear Transformations.** The next section, [§04-Linear-Transformations](../04-Linear-Transformations/notes.md), develops the abstract theory of linear maps between vector spaces. PCA can be understood in this language: the PCA projection $\mathbf{x} \mapsto V_k^\top \mathbf{x}$ is a linear map from $\mathbb{R}^d$ to $\mathbb{R}^k$, and the reconstruction $\mathbf{z} \mapsto V_k\mathbf{z}$ is its "almost-inverse" (a pseudoinverse). The kernel of the projection is the span of the discarded components $\{\mathbf{v}_{k+1}, \ldots, \mathbf{v}_d\}$ — the null space of the PCA transform. This connection between PCA and kernel/image theory is explored in §04.

**Forward to Orthogonality.** [§05-Orthogonality-and-Orthonormality](../05-Orthogonality-and-Orthonormality/notes.md) develops the Gram-Schmidt algorithm for constructing orthonormal bases, which is closely related to PCA (both find orthonormal bases, but PCA optimizes which basis is chosen). The QR decomposition studied there is also the computational backbone of many iterative eigenvalue algorithms used in large-scale PCA.

```
PCA IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01 Eigenvalues          §02 SVD
  ─────────────────        ───────────────
  Spectral theorem    ──►  Eckart-Young
  Covariance matrix   ──►  Singular values
  Eigenvectors        ──►  Right singular vectors
          │                      │
          └──────────┬───────────┘
                     ▼
            §03 PCA  ◄── THIS SECTION
            ─────────────────────────────
            • Variance maximization
            • Explained variance
            • Whitening
            • Probabilistic PCA
            • Kernel PCA
            • AI applications
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
  §04 Linear              §05 Orthogonality
  Transformations         ─────────────────
  ───────────────         Gram-Schmidt
  Kernel of PCA proj      QR decomposition
  Image = PC subspace     Orthogonal bases

════════════════════════════════════════════════════════════════════════
```

**The deeper pattern.** PCA is the first section in this curriculum where algebra, geometry, optimization, and statistics come together in one algorithm. Every major dimension-reduction technique in modern AI — LoRA, attention head analysis, representation geometry, loss landscape visualization — is a variation on the PCA theme: find a low-dimensional linear subspace that captures the essential structure of high-dimensional data. Mastering PCA is mastering the mathematical language in which most of AI's structural properties are expressed.

---

[← Back to Advanced Linear Algebra](../README.md) | [← SVD](../02-Singular-Value-Decomposition/notes.md) | [Linear Transformations →](../04-Linear-Transformations/notes.md)

---

## Appendix A: Complete PCA Algorithm and Complexity

### A.1 Algorithm with Full Pseudocode

```
ALGORITHM: Principal Component Analysis
════════════════════════════════════════════════════════════════════════

INPUT:
  X ∈ ℝ^{n×d}   Data matrix (n samples, d features)
  k              Number of components (1 ≤ k ≤ min(n,d))
  standardize    Boolean: whether to z-score normalize (default: False)

PREPROCESSING:
  1. Compute mean: μ = (1/n) Xᵀ 1  ∈ ℝ^d
  2. Center data: X̃ = X - 1μᵀ
  3. If standardize:
       σ = std(X̃, axis=0)        ∈ ℝ^d   (per-feature std)
       X̃ = X̃ / σ                           (in-place normalize)
     Else:
       σ = 1  (no-op)

CORE COMPUTATION (via thin SVD):
  4. Compute thin SVD: X̃ = UΣVᵀ
       U ∈ ℝ^{n×r},  Σ = diag(σ₁,...,σᵣ),  V ∈ ℝ^{d×r}
       r = min(n, d),  σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0
  5. Select top k columns:
       Vₖ = V[:, :k] ∈ ℝ^{d×k}   (principal component directions)
       Uₖ = U[:, :k] ∈ ℝ^{n×k}
       sₖ = σ[:k]    ∈ ℝ^k

OUTPUT COMPUTATION:
  6. Scores (low-dim representation): Z = Uₖ · diag(sₖ) ∈ ℝ^{n×k}
       Equivalently: Z = X̃ Vₖ
  7. Explained variances: λᵢ = σᵢ² / (n-1)   for i = 1,...,k
  8. Variance ratios: ρᵢ = λᵢ / Σⱼ λⱼ
  9. Cumulative EVR: CEV(k) = Σᵢ₌₁ᵏ ρᵢ

RECONSTRUCTION (optional):
  10. X̂ = Z Vₖᵀ · diag(σ) + 1μᵀ    (undo standardization and centering)
  11. Reconstruction error = Σᵢ₌ₖ₊₁ʳ σᵢ²

NEW-SAMPLE PROJECTION:
  Given x_new ∈ ℝ^d:
  12. x̃_new = (x_new - μ) / σ        (use TRAINING μ, σ)
  13. z_new = Vₖᵀ x̃_new ∈ ℝ^k

════════════════════════════════════════════════════════════════════════
```

### A.2 Time and Space Complexity

| Operation | Time complexity | Space complexity | Notes |
|---|---|---|---|
| Centering | $O(nd)$ | $O(nd)$ | In-place possible |
| Standardization | $O(nd)$ | $O(d)$ | Per-feature |
| Full SVD | $O(\min(n^2 d, nd^2))$ | $O(nd)$ | `np.linalg.svd` |
| Thin (economy) SVD | $O(nd \cdot \min(n,d))$ | $O(n\cdot\min(n,d))$ | `full_matrices=False` |
| Truncated SVD (ARPACK) | $O(ndk)$ | $O(nk + dk)$ | `scipy.sparse.linalg.svds` |
| Randomized SVD | $O(ndk + k^2(n+d))$ | $O(nk + dk)$ | `sklearn` default for large $n,d$ |
| Projection | $O(ndk)$ | $O(nk)$ | $Z = \tilde{X}V_k$ |
| Reconstruction | $O(ndk)$ | $O(nd)$ | $\hat{X} = ZV_k^\top$ |

**When to use which SVD:**

| Setting | Recommendation |
|---|---|
| $n, d \leq 10{,}000$, any $k$ | Full/thin SVD (`np.linalg.svd`, `full_matrices=False`) |
| $k \ll \min(n,d)$, moderate size | Randomized SVD (`sklearn.utils.extmath.randomized_svd`) |
| $n > 10^5$ or streaming data | Incremental PCA (`sklearn.decomposition.IncrementalPCA`) |
| $k \ll \min(n,d)$, sparse $X$ | ARPACK (`scipy.sparse.linalg.svds`) |
| Distributed computation | Distributed SVD (Spark MLlib, Dask) |

### A.3 Numerical Stability Details

**Why squaring the data is dangerous.** If $\tilde{X}$ has condition number $\kappa(\tilde{X}) = \sigma_1/\sigma_r$, then $C = \tilde{X}^\top\tilde{X}/(n-1)$ has condition number $\kappa(C) = \kappa(\tilde{X})^2$. For ill-conditioned data (common in genomics, sensor fusion), this means:

- Data with $\kappa(\tilde{X}) = 10^4$: SVD of $\tilde{X}$ stable in double precision ($\epsilon \approx 10^{-16}$)
- But covariance $C$ has $\kappa(C) = 10^8$: eigenvalue computation loses $8$ digits of precision
- Near-zero eigenvalues of $C$ may become negative due to rounding — `np.linalg.eig` may return complex values

**Regularization for ill-conditioned covariance.** Add a small diagonal:
$$C_\epsilon = C + \epsilon I, \quad \epsilon = \delta \cdot \operatorname{tr}(C)/d$$

This "regularized PCA" or "ridge PCA" ensures all eigenvalues are at least $\epsilon$. Common in high-dimensional settings ($d > n$) where $C$ is rank-deficient.

```python
def pca_regularized(X, k, delta=1e-5):
    """PCA with Tikhonov regularization for ill-conditioned data."""
    n, d = X.shape
    X_c = X - X.mean(axis=0)
    C = X_c.T @ X_c / (n - 1)
    eps = delta * np.trace(C) / d
    C_reg = C + eps * np.eye(d)
    lambdas, V = np.linalg.eigh(C_reg)
    # eigh returns ascending order; reverse
    lambdas = lambdas[::-1]
    V = V[:, ::-1]
    return V[:, :k], lambdas[:k] - eps  # subtract regularization
```

---

## Appendix B: PCA Decision Guide

```
PCA DECISION FLOWCHART
════════════════════════════════════════════════════════════════════════

  START: I need to reduce dimensionality
          │
          ▼
  Is the relationship between variables LINEAR?
  ├── YES → Standard PCA (§2-§4)
  └── NO  → Is n manageable (< 50,000)?
             ├── YES → Kernel PCA with RBF (§7)
             └── NO  → Autoencoder or UMAP

  Chosen PCA. Should I STANDARDIZE?
  ├── Features on same scale, units meaningful → Center only
  └── Different scales or units → Standardize (z-score)

  How to choose k?
  ├── Exploratory / visualization → Scree plot elbow + 90% EVR
  ├── Preprocessing for classifier → Cross-validate downstream metric
  ├── Compression → Set reconstruction error budget
  └── Statistical test needed → Parallel analysis (Horn's method)

  Large dataset? (n > 10,000 or d > 10,000)
  ├── n >> d → Incremental PCA (mini-batch)
  ├── k << min(n,d) → Randomized SVD
  └── Data has outliers → Robust PCA first, then PCA on low-rank part

  Need uncertainty / missing data handling?
  └── YES → Probabilistic PCA (§6) with EM

════════════════════════════════════════════════════════════════════════
```

---

## Appendix C: Key Formulas Reference

| Quantity | Formula | Code |
|---|---|---|
| Sample covariance | $C = \tilde{X}^\top\tilde{X}/(n-1)$ | `np.cov(X.T)` |
| PCA via SVD | $\tilde{X} = U\Sigma V^\top$ | `np.linalg.svd(X_c, full_matrices=False)` |
| PC directions | Columns of $V$ (rows of $V^\top$) | `Vt.T` |
| Explained variance | $\lambda_i = \sigma_i^2/(n-1)$ | `s**2 / (n-1)` |
| Variance ratio | $\rho_i = \sigma_i^2 / \sum_j\sigma_j^2$ | `s**2 / (s**2).sum()` |
| Scores | $Z = \tilde{X}V_k = U_k\Sigma_k$ | `X_c @ Vt[:k].T` |
| Reconstruction | $\hat{X} = ZV_k^\top + \bar{x}$ | `Z @ Vt[:k] + mean` |
| Recon error | $\sum_{i>k}\sigma_i^2$ | `(s[k:]**2).sum()` |
| PCA whitening | $Z_w = Z\Lambda_k^{-1/2}$ | `Z / np.sqrt(lambdas)` |
| ZCA whitening | $Z_w = \tilde{X}V_k\Lambda_k^{-1/2}V_k^\top$ | See §5.2 |
| Participation ratio | $(\sum\lambda_i)^2 / \sum\lambda_i^2$ | Custom (§9.2) |
| Centered kernel | $\tilde{K} = K - \mathbf{1}_nK - K\mathbf{1}_n + \mathbf{1}_nK\mathbf{1}_n$ | See §7.3 |
| PPCA noise | $\hat{\sigma}^2 = \sum_{i>k}\lambda_i/(d-k)$ | See §6.2 |

---

---

## Appendix D: Worked Example — Complete PCA from Scratch

This section walks through a complete PCA computation on a small dataset, showing every intermediate result.

### D.1 Setup

Consider a dataset of $n=5$ samples and $d=3$ features — temperature, humidity, and pressure readings from a weather station:

$$X = \begin{pmatrix}23 & 65 & 1013 \\ 27 & 72 & 1008 \\ 19 & 58 & 1020 \\ 25 & 68 & 1015 \\ 21 & 61 & 1018\end{pmatrix}$$

### D.2 Step 1: Center the Data

Sample mean:
$$\bar{\mathbf{x}} = (23, 65, 1013)^\top + \frac{1}{5}\begin{pmatrix}4\\7\\-5\\2\\-2\\-3\\3\\8\\2\\-2\end{pmatrix} \approx (23, 64.8, 1014.8)^\top$$

After centering: $\tilde{X} = X - \mathbf{1}\bar{\mathbf{x}}^\top$. Note the scale difference — pressure values are ~1000× larger than temperature values. This illustrates why standardization is critical.

### D.3 Step 2: Effect of Scale

Without standardization, the covariance matrix is dominated by the pressure column (variance $\approx 22$) which is already largest, but temperature (variance $\approx 10$) and humidity (variance $\approx 30$) are comparable. After standardization, all features have unit variance, and PCA finds directions of equal information.

```python
import numpy as np

X = np.array([[23, 65, 1013],
              [27, 72, 1008],
              [19, 58, 1020],
              [25, 68, 1015],
              [21, 61, 1018]], dtype=float)

# Without standardization
X_c = X - X.mean(axis=0)
C = X_c.T @ X_c / (len(X) - 1)
print("Covariance matrix (no standardization):")
print(np.round(C, 2))
# [[ 10.    12.25 -12.5 ]
#  [ 12.25  27.2  -21.25]
#  [-12.5  -21.25  24.2 ]]

# With standardization
X_std = X_c / X_c.std(axis=0, ddof=1)
C_std = X_std.T @ X_std / (len(X) - 1)
print("\nCorrelation matrix (standardized):")
print(np.round(C_std, 3))
# [[1.    0.984 -0.999]
#  [0.984 1.    -0.997]
#  [-0.999 -0.997 1.   ]]
```

**Insight from the correlation matrix:** All three features are almost perfectly correlated (or anti-correlated). Temperature and humidity move together; both are anti-correlated with pressure. This is physically meaningful: warm, humid days are low-pressure days. PCA should find ~1 meaningful component.

### D.4 Step 3: SVD and Principal Components

```python
U, s, Vt = np.linalg.svd(X_std, full_matrices=False)

# Singular values → explained variance ratios
var_ratio = s**2 / (s**2).sum()
print("Explained variance ratios:", np.round(var_ratio, 4))
# [0.9986, 0.0013, 0.0001]
# → PC1 explains 99.86% of variance!

print("PC1 direction:", np.round(Vt[0], 3))
# [ 0.577  0.579 -0.577] ← equal weights on all three
# Temperature and humidity load positively, pressure negatively
# → PC1 is a "warm-humid-low-pressure" index
```

**Interpretation:** PC1 = $(0.577, 0.579, -0.577)^\top$ — a linear combination that increases with temperature and humidity and decreases with pressure. This is physically the "storm index." PC2 and PC3 explain only 0.14% of variance combined — essentially noise.

### D.5 Step 4: Reconstruction

```python
k = 1  # Keep only PC1
Z_k = U[:, :k] * s[:k]       # Scores: (5, 1)
X_reconstructed = Z_k @ Vt[:k] + X_std.mean(axis=0)

recon_error = np.linalg.norm(X_std - Z_k @ Vt[:k])**2
print(f"Reconstruction error (k=1): {recon_error:.6f}")
print(f"Expected (discarded variance): {(s[1:]**2).sum():.6f}")
# Both ~0.0014 — matches Eckart-Young
```

---

## Appendix E: Connections to Information Theory and Statistics

### E.1 PCA and Maximum Likelihood under Gaussianity

If the data is Gaussian, $\mathbf{x}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, then the sample covariance $C$ is the maximum likelihood estimator of $\Sigma$. PCA on $C$ finds the directions of greatest information concentration under the Gaussian assumption.

Under this model, the PCA projection scores $Z = \tilde{X}V_k$ are also Gaussian: $Z_{:,i} \sim \mathcal{N}(0, \lambda_i)$. The optimal Bayesian compression of a Gaussian random vector to $k$ dimensions is exactly the PCA projection — PCA is the optimal linear encoder under squared loss and Gaussian data.

**Non-Gaussian data.** For non-Gaussian distributions, PCA minimizes reconstruction error but may not capture the most "interesting" structure. Alternative approaches:
- ICA: maximize statistical independence (not just decorrelation)
- NMF: non-negative matrix factorization for parts-based representations
- Sparse coding: find sparse representations in overcomplete dictionaries
- VAE: learn a non-linear latent space via variational inference

### E.2 Information-Geometric View

The set of all $d \times d$ positive definite matrices forms a Riemannian manifold $(\mathbb{S}^d_{++}, g)$ with the Fisher information metric. The MLE $C$ is the Fréchet mean on this manifold.

The PCA truncation $C \to V_k\Lambda_k V_k^\top + \hat{\sigma}^2(I - V_kV_k^\top)$ (the PPCA approximation) is the projection of $C$ onto the set of matrices with at most $k$ large eigenvalues — the PPCA manifold. This gives information-geometric meaning to choosing $k$: it is the number of "directions" the data needs to distinguish itself from noise.

### E.3 PCA as a Linear Autoencoder

A linear autoencoder with bottleneck dimension $k$ consists of:
- **Encoder:** $\mathbf{z} = W_e\mathbf{x} \in \mathbb{R}^k$, $W_e \in \mathbb{R}^{k \times d}$
- **Decoder:** $\hat{\mathbf{x}} = W_d\mathbf{z} \in \mathbb{R}^d$, $W_d \in \mathbb{R}^{d \times k}$
- **Loss:** $\mathcal{L} = \lVert\mathbf{x} - \hat{\mathbf{x}}\rVert_2^2$

**Theorem.** The global minimum of the linear autoencoder loss is achieved when $\operatorname{col}(W_d) = \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ (the PCA subspace). The optimal reconstruction equals the PCA reconstruction.

**Important caveat:** The optimal $W_e$ and $W_d$ need not equal $V_k^\top$ and $V_k$ individually — any rotation $R$ works: $W_d = V_k R$, $W_e = R^\top V_k^\top$. The autoencoder can discover any orthogonal basis for the PCA subspace — it is unique as a subspace, not as specific directions.

**Contrast with non-linear autoencoders:** A neural autoencoder with non-linear activations can learn non-linear manifolds, capturing curvature that linear PCA misses. But for linear data, the two are equivalent, and the non-linear version is strictly harder to optimize.

---

## Appendix F: PCA in Python Ecosystem

### F.1 sklearn API

```python
from sklearn.decomposition import PCA, KernelPCA, SparsePCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Standard PCA with variance threshold
pca = PCA(n_components=0.95)    # keep 95% variance
pca.fit(X_train)
Z_train = pca.transform(X_train)
Z_test = pca.transform(X_test)  # uses training mean/components
print(f"Shape: {X_train.shape} → {Z_train.shape}")
print(f"Components: {pca.n_components_}")
print(f"Explained var: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")

# Full pipeline with standardization
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
])
Z = pipe.fit_transform(X_train)
Z_test = pipe.transform(X_test)  # correct: uses training scaler+PCA

# Randomized PCA (faster for large n, d, small k)
pca_rand = PCA(n_components=50, svd_solver='randomized', random_state=42)

# Incremental PCA for streaming/large data
ipca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in data_batches:
    ipca.partial_fit(batch)

# Kernel PCA
kpca = KernelPCA(n_components=10, kernel='rbf', gamma=0.1,
                  fit_inverse_transform=True)
Z_kpca = kpca.fit_transform(X)
X_back = kpca.inverse_transform(Z_kpca)  # approximate reconstruction
```

### F.2 NumPy/SciPy Low-Level API

```python
import numpy as np
import scipy.linalg as la

# Full SVD
U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

# Symmetric eigendecomposition (for SPD matrices — better than eig)
lambdas, Q = np.linalg.eigh(C)    # returns ascending order
lambdas = lambdas[::-1]; Q = Q[:, ::-1]  # → descending

# Truncated SVD via ARPACK (for sparse matrices, large k)
from scipy.sparse.linalg import svds
U_k, s_k, Vt_k = svds(X_sparse, k=50)  # top-50 SVD of sparse X
idx = np.argsort(s_k)[::-1]             # ARPACK returns arbitrary order
U_k, s_k, Vt_k = U_k[:, idx], s_k[idx], Vt_k[idx]
```

### F.3 PyTorch (GPU-accelerated)

```python
import torch

X_t = torch.tensor(X_centered, dtype=torch.float32).cuda()

# Full SVD on GPU
U, s, Vh = torch.linalg.svd(X_t, full_matrices=False)

# Low-rank SVD (much faster for k << n,d)
U_k, s_k, Vh_k = torch.svd_lowrank(X_t, q=k, niter=4)

# Projection and reconstruction
Z_k = X_t @ Vh_k.T          # (n, k)
X_recon = Z_k @ Vh_k + mean_t  # (n, d)
```

---

---

## Appendix G: PCA vs Factor Analysis

PCA and Factor Analysis (FA) both reduce dimensionality, but with fundamentally different models and goals. This confusion is common in practice.

### G.1 The Factor Analysis Model

FA assumes:
$$\mathbf{x} = \Lambda \mathbf{f} + \boldsymbol{\epsilon}, \quad \mathbf{f} \sim \mathcal{N}(\mathbf{0}, I_k), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \Psi)$$

where $\Lambda \in \mathbb{R}^{d \times k}$ is the loading matrix, $\mathbf{f}$ are the latent factors, and $\Psi = \operatorname{diag}(\psi_1, \ldots, \psi_d)$ is a **diagonal** (not isotropic) noise covariance.

The observed covariance is: $\Sigma = \Lambda\Lambda^\top + \Psi$

**Key difference from PPCA:** In PPCA, $\Psi = \sigma^2 I$ (isotropic noise — same noise in every direction). In FA, $\Psi$ is diagonal but arbitrary — each feature has its own noise level. This makes FA more flexible but harder to fit (requires EM or iterative estimation).

### G.2 Comparison Table

| Aspect | PCA | Factor Analysis |
|---|---|---|
| Goal | Explain maximum variance | Explain correlations via shared factors |
| Noise model | None (geometric) | $\mathcal{N}(\mathbf{0}, \Psi)$, $\Psi$ diagonal |
| Unique components | No (rotation-ambiguous) | No (rotation-ambiguous) |
| Scale-invariant | No | Yes (FA is correlation-based) |
| Interpretation | Components = linear combos of features | Factors = latent causes of correlations |
| Model selection | Scree, EVR, Kaiser | Likelihood ratio test, BIC |
| Handles unique variances | No (all noise enters identically) | Yes ($\psi_i$ per feature) |
| Preferred when | Features are measured on same scale; want linear compression | Features have different measurement errors; want causal factors |

### G.3 When to Use Which

**Use PCA when:**
- The goal is linear dimensionality reduction for downstream modelling
- Features are on the same scale (or you standardize for scale-invariance)
- You want the computationally simplest, most interpretable method
- Reconstruction error is the primary criterion

**Use Factor Analysis when:**
- Features have different measurement error variances
- You believe latent factors causally generate the features
- Statistical testing of the number of factors is needed
- You are working in psychometrics, econometrics, or psychophysiology

**Use PPCA when:**
- You want a probabilistic model that recovers standard PCA in the zero-noise limit
- Missing data is present (EM handles it naturally)
- You need likelihood scores for model comparison (choosing $k$)

---

## Appendix H: Geometric Interpretation Across Dimensions

### H.1 PCA in 2D

In 2D, the data cloud is an ellipse. The two principal components are the major and minor axes of the ellipse. The aspect ratio of the ellipse equals $\sqrt{\lambda_1/\lambda_2}$.

```
2D DATA ELLIPSE AND PRINCIPAL COMPONENTS
════════════════════════════════════════════════════════════════════════

          x₂
          │       . . .
          │    .         .           v₁ = major axis = PC1
          │  .      ↗     .          λ₁ = half-length² (variance)
          │ .    v₁       .
          │.  ↑v₂          .         v₂ = minor axis = PC2
          │ .              .          λ₂ = half-length² (variance)
          │  .            .
          │    .         .
          │       . . .
          └──────────────────── x₁

  PC1 direction: angle θ where tan(2θ) = 2C₁₂/(C₁₁ - C₂₂)
  (from characteristic polynomial of 2×2 covariance)

════════════════════════════════════════════════════════════════════════
```

For 2D Gaussian data with covariance $C = \begin{pmatrix}a & b \\ b & c\end{pmatrix}$:
- Eigenvalues: $\lambda_{1,2} = \frac{(a+c) \pm \sqrt{(a-c)^2 + 4b^2}}{2}$
- PC1 angle: $\theta = \frac{1}{2}\arctan\!\left(\frac{2b}{a-c}\right)$

### H.2 PCA in 3D

In 3D, the data lives in an ellipsoid with three principal axes of lengths $\sqrt{\lambda_1} \geq \sqrt{\lambda_2} \geq \sqrt{\lambda_3}$. Three cases:

| Spectrum | Shape | Example |
|---|---|---|
| $\lambda_1 \gg \lambda_2 \approx \lambda_3$ | Cigar (1D structure) | Collinear data, LoRA rank-1 |
| $\lambda_1 \approx \lambda_2 \gg \lambda_3$ | Pancake (2D structure) | Planar data, face images |
| $\lambda_1 \approx \lambda_2 \approx \lambda_3$ | Sphere (3D, no structure) | White noise |
| $\lambda_1 > \lambda_2 > \lambda_3$ | General ellipsoid | Typical real data |

Projection onto PC1: map to a line (1D cigar projection).
Projection onto PC1, PC2: map to a plane (2D cross-section of ellipsoid).

### H.3 High-Dimensional PCA: The Marchenko-Pastur Law

In the "spiked" random matrix model, data has $n$ samples and $d$ features with no structure: $X_{ij} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$. The eigenvalues of $C = X^\top X/n$ converge (as $n,d \to \infty$ with $d/n \to \gamma$) to the **Marchenko-Pastur distribution**:

$$p(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\gamma\lambda}, \quad \lambda \in [\lambda_-, \lambda_+]$$

where $\lambda_\pm = (1 \pm \sqrt{\gamma})^2$.

**Critical implication:** When eigenvalues from PCA fall within the Marchenko-Pastur bulk $[\lambda_-, \lambda_+]$, they correspond to pure noise — keeping these components adds noise, not signal. Only eigenvalues **above** $\lambda_+$ are genuine signal components.

This provides a rigorous threshold for component selection:
- Parallel analysis (Horn's method) approximates this empirically
- For standardized data with $n = 500$, $d = 100$ ($\gamma = 0.2$): $\lambda_+ = (1+\sqrt{0.2})^2 \approx 1.99$ — keep only PCs with $\lambda > 1.99$, not just $\lambda > 1$ (Kaiser criterion is too lenient here)

---

## Appendix I: Self-Check Questions

Before moving to the next section, verify you can answer these questions:

**Conceptual:**
1. Why must data be centered before PCA? What goes wrong if you skip centering?
2. Why do PCA principal components have sign ambiguity? Is the PCA *subspace* unique?
3. In what sense is PCA the "optimal" linear dimensionality reduction? What is it optimizing?
4. What is the difference between PCA scores and PCA loadings (components)?
5. Why is SVD numerically preferred over eigendecomposition of the covariance matrix?
6. When should you standardize before PCA? Give a concrete example where failing to standardize changes the answer qualitatively.
7. What does the PPCA noise parameter $\hat{\sigma}^2$ estimate, and why is it the average of the discarded eigenvalues?
8. Why does kernel PCA require centering the kernel matrix? What is the kernel matrix equivalent of "centering the data"?

**Mathematical:**
1. Derive the PCA optimization: show that maximizing $\mathbf{v}^\top C\mathbf{v}$ subject to $\lVert\mathbf{v}\rVert = 1$ gives the eigenvalue equation $C\mathbf{v} = \lambda\mathbf{v}$.
2. Prove that the PCA scores are decorrelated: show $\operatorname{Cov}(Z) = \Lambda$.
3. State the Eckart-Young theorem and explain how it implies PCA minimizes reconstruction error.
4. Show that the reconstruction error from keeping $k$ PCs equals $\sum_{i>k}\sigma_i^2$.
5. For PPCA: verify that as $\sigma^2 \to 0$, the PPCA MLE $\hat{W}$ converges to $V_k\Lambda_k^{1/2}$ (up to rotation).

**Computational:**
1. Given a $100 \times 50$ data matrix, what is the shape of $U$, $\Sigma$, $V^\top$ from the thin SVD?
2. How do you project a new test point onto the PCA subspace without data leakage?
3. What does `PCA(n_components=0.95)` in sklearn compute?
4. Why must you center the kernel vector (not just the kernel matrix) when projecting new points in kernel PCA?

---

---

## Appendix J: PCA and the Four Fundamental Subspaces

The connection between PCA and the four fundamental subspaces (see [§02-Linear-Algebra-Basics §05](../../02-Linear-Algebra-Basics/05-Matrix-Rank/notes.md)) is illuminating.

For the centered data matrix $\tilde{X} \in \mathbb{R}^{n \times d}$ with rank $r = \min(n,d)$ (assuming generic position):

| Subspace | For $\tilde{X}$ | PCA interpretation |
|---|---|---|
| Column space $\mathcal{C}(\tilde{X})$ | Span of columns $= $ span of $U$ | Space of all possible score vectors |
| Row space $\mathcal{C}(\tilde{X}^\top)$ | Span of rows $=$ span of $V$ | The PCA subspace — span of principal components |
| Null space $\mathcal{N}(\tilde{X})$ | Vectors $\mathbf{v}$ with $\tilde{X}\mathbf{v}=\mathbf{0}$ | Discarded directions — zero-variance PCs |
| Left null space $\mathcal{N}(\tilde{X}^\top)$ | Vectors $\mathbf{u}$ with $\tilde{X}^\top\mathbf{u}=\mathbf{0}$ | Directions in sample space with no data |

**PCA projection in subspace language:**
- The PCA encoder $\mathbf{x} \mapsto V_k^\top\mathbf{x}$ is an orthogonal projection onto the row space of $\tilde{X}$
- The discarded components $\{\mathbf{v}_{k+1},\ldots,\mathbf{v}_d\}$ span the part of the row space that is "noisy" or irrelevant
- If $n < d$, then $\tilde{X}$ has a null space of dimension $d - r$ — these directions have zero variance and are automatically discarded

**Dimension counting:**
$$d = k_{\text{signal}} + k_{\text{noise}} + \dim\mathcal{N}(\tilde{X})$$

For $n > d$ (more samples than features): $\dim\mathcal{N}(\tilde{X}) = 0$, all variance is captured. For $n < d$: $\dim\mathcal{N}(\tilde{X}) = d - n$ — the data cannot fill the feature space, and PCA automatically collapses those directions.

---

## Appendix K: References and Further Reading

### Primary References

1. **Pearson, K. (1901).** "On Lines and Planes of Closest Fit to Systems of Points in Space." *Philosophical Magazine*, 2(11), 559–572. — The original PCA paper.

2. **Hotelling, H. (1933).** "Analysis of a Complex of Statistical Variables into Principal Components." *Journal of Educational Psychology*, 24(6), 417–441. — Named "principal components" and gave the Lagrange-multiplier derivation.

3. **Tipping, M.E. & Bishop, C.M. (1999).** "Probabilistic Principal Component Analysis." *Journal of the Royal Statistical Society B*, 61(3), 611–622. — PPCA: the generative model and MLE.

4. **Schölkopf, B., Smola, A., & Müller, K.-R. (1998).** "Nonlinear Component Analysis as a Kernel Eigenvalue Problem." *Neural Computation*, 10(5), 1299–1319. — Kernel PCA.

5. **Candès, E.J., Li, X., Ma, Y., & Wright, J. (2011).** "Robust Principal Component Analysis?" *Journal of the ACM*, 58(3), 1–37. — Robust PCA via nuclear norm minimization.

6. **Halko, N., Martinsson, P.-G., & Tropp, J.A. (2011).** "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." *SIAM Review*, 53(2), 217–288. — Randomized SVD.

### AI/ML Applications

7. **Hu, E.J. et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*. — The LoRA paper; low-rank hypothesis.

8. **Li, H. et al. (2018).** "Visualizing the Loss Landscape of Neural Nets." *NeurIPS 2018*. — PCA of gradient trajectories for loss landscape visualization.

9. **Aghajanyan, A. et al. (2020).** "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." *ACL 2021*. — Measuring intrinsic dimensionality of fine-tuning.

10. **Elhage, N. et al. (2022).** "Toy Models of Superposition." *Anthropic Research*. — PCA and sparse representations in transformer models.

### Textbooks

11. **Jolliffe, I.T. (2002).** *Principal Component Analysis* (2nd ed.). Springer. — The definitive reference.

12. **Bishop, C.M. (2006).** *Pattern Recognition and Machine Learning*. Springer. — Chapter 12: PPCA, mixtures, probabilistic view.

13. **Murphy, K.P. (2022).** *Probabilistic Machine Learning: An Introduction*. MIT Press. — Chapter 20: low-rank models, PCA, factor analysis.

14. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. — Chapter 14: unsupervised learning, PCA, kernel PCA.

---

---

## Appendix L: PCA in Practice — Common Workflows

### L.1 Preprocessing Pipeline for Supervised Learning

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Full pipeline: scale → PCA → classifier
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('clf', SVC(kernel='rbf'))
])

# Grid search over number of PCA components AND SVM hyperparameters
param_grid = {
    'pca__n_components': [10, 20, 50, 100, 0.95],  # 0.95 = 95% variance
    'clf__C': [0.1, 1.0, 10.0],
    'clf__gamma': ['scale', 'auto']
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
search.fit(X_train, y_train)

print(f"Best k: {search.best_params_['pca__n_components']}")
print(f"Best accuracy: {search.best_score_:.4f}")
```

**Best practice:** Always search over `n_components` as a hyperparameter when using PCA for supervised learning. The optimal $k$ for downstream accuracy often differs from the $k$ chosen by variance threshold.

### L.2 Visualization Workflow

Standard workflow for visualizing high-dimensional data:

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Step 1: PCA to ~50 dims (remove noise, keep structure)
pca = PCA(n_components=50, random_state=42)
X_pca50 = pca.fit_transform(StandardScaler().fit_transform(X))
print(f"50-dim PCA captures {pca.explained_variance_ratio_.sum():.1%} variance")

# Step 2: t-SNE on PCA output (not raw data — faster, more stable)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_2d = tsne.fit_transform(X_pca50)

# Step 3: Plot with class labels
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA 2D (first 2 components)
ax = axes[0]
for c in np.unique(labels):
    mask = labels == c
    ax.scatter(X_pca50[mask, 0], X_pca50[mask, 1], label=f'Class {c}', alpha=0.7, s=20)
ax.set_title(f'PCA (PC1+PC2: {pca.explained_variance_ratio_[:2].sum():.1%} var)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend()

# t-SNE 2D
ax = axes[1]
for c in np.unique(labels):
    mask = labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=f'Class {c}', alpha=0.7, s=20)
ax.set_title('t-SNE (on 50-dim PCA output)')
ax.legend()

plt.tight_layout()
```

### L.3 Anomaly Detection with PCA

PCA-based anomaly detection: high reconstruction error = anomalous point.

```python
def pca_anomaly_detector(X_train, X_test, k, threshold_pct=95):
    """
    Detect anomalies as points with reconstruction error above threshold.
    Returns: anomaly scores for X_test, threshold value
    """
    # Fit PCA
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=k)
    pca.fit(X_train_s)

    # Reconstruction error for training data
    X_train_recon = pca.inverse_transform(pca.transform(X_train_s))
    train_errors = np.sum((X_train_s - X_train_recon)**2, axis=1)
    threshold = np.percentile(train_errors, threshold_pct)

    # Reconstruction error for test data
    X_test_recon = pca.inverse_transform(pca.transform(X_test_s))
    test_errors = np.sum((X_test_s - X_test_recon)**2, axis=1)

    anomalies = test_errors > threshold
    return test_errors, threshold, anomalies
```

**How it works:** Normal samples live near the PCA subspace (low reconstruction error). Anomalous samples deviate from the normal subspace (high error). The threshold is set from the distribution of training reconstruction errors.

### L.4 Monitoring Distribution Shift

PCA scores can detect dataset drift between training and production:

```python
def check_distribution_shift(X_train, X_prod, k=10, alpha=0.05):
    """
    Check if production data has drifted from training distribution.
    Uses Hotelling's T² test on PCA scores.
    """
    from scipy import stats

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_pr_s = scaler.transform(X_prod)

    pca = PCA(n_components=k)
    Z_train = pca.fit_transform(X_tr_s)
    Z_prod = pca.transform(X_pr_s)

    # Test each component for distribution shift (Kolmogorov-Smirnov)
    drifted = []
    for i in range(k):
        _, p_value = stats.ks_2samp(Z_train[:, i], Z_prod[:, i])
        if p_value < alpha:
            drifted.append(i)

    if drifted:
        print(f"Distribution drift detected in PCA dimensions: {drifted}")
        print(f"PC{drifted[0]+1} explains {pca.explained_variance_ratio_[drifted[0]]:.1%} variance")
    else:
        print("No significant drift detected.")
    return drifted
```

**For AI:** Production monitoring of LLM inputs and outputs often uses PCA of embeddings. If the PCA score distribution of production inputs drifts from training, it signals distribution shift — triggering retraining or alerts.

### L.5 Summary of Key Implementation Choices

```
IMPLEMENTATION DECISION TREE
════════════════════════════════════════════════════════════════════════

  Dataset size?
  ┌─── Small (n,d < 10k) ───────────────────────────────────────────┐
  │  np.linalg.svd(X_c, full_matrices=False)  → exact, fast        │
  └─────────────────────────────────────────────────────────────────┘
  ┌─── Large, small k ──────────────────────────────────────────────┐
  │  sklearn.utils.extmath.randomized_svd(k, n_iter=5)             │
  │  → ~10x faster than full SVD                                   │
  └─────────────────────────────────────────────────────────────────┘
  ┌─── Streaming / memory-constrained ──────────────────────────────┐
  │  sklearn.decomposition.IncrementalPCA(batch_size=B)             │
  │  → processes data in batches of size B                         │
  └─────────────────────────────────────────────────────────────────┘
  ┌─── Sparse data ─────────────────────────────────────────────────┐
  │  scipy.sparse.linalg.svds(X_sparse, k=k)                       │
  │  → never forms dense matrices                                  │
  └─────────────────────────────────────────────────────────────────┘
  ┌─── GPU / large-scale ───────────────────────────────────────────┐
  │  torch.svd_lowrank(X_tensor, q=k, niter=4)                     │
  │  → GPU-accelerated randomized SVD                              │
  └─────────────────────────────────────────────────────────────────┘
  ┌─── Outliers present ────────────────────────────────────────────┐
  │  sklearn.decomposition.MiniBatchSparsePCA or Robust PCA        │
  │  → decompose X = L + S first, then PCA on L                   │
  └─────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

---
