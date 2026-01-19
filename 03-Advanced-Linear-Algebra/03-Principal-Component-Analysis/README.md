# Principal Component Analysis (PCA)

## Introduction

Principal Component Analysis (PCA) is one of the most fundamental dimensionality reduction techniques in machine learning. It finds new orthogonal axes (principal components) that maximize variance in the data, enabling compression, visualization, noise reduction, and feature extraction.

## Prerequisites

- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Covariance matrices
- Matrix operations

## Learning Objectives

1. Understand PCA geometrically and algebraically
2. Derive PCA from covariance eigendecomposition
3. Implement PCA using SVD
4. Choose the number of components
5. Apply PCA to real-world problems

---

## 1. The Goal of PCA

### Problem Statement

Given high-dimensional data $X \in \mathbb{R}^{n \times d}$ (n samples, d features):

- Find a lower-dimensional representation $Z \in \mathbb{R}^{n \times k}$ where $k < d$
- Preserve as much variance (information) as possible
- Find orthogonal directions that capture maximum variance

```
Original Data (d dimensions)          PCA           Reduced (k dimensions)
                                      →
    [x₁, x₂, ..., x_d]                            [z₁, z₂, ..., z_k]
         ↓                                              ↓
    n samples × d features                        n samples × k features
```

### Geometric Intuition

```
High variance direction (PC1):        Low variance direction (PC2):

        •  •                                    • •
       • ← →  •                                • • •
      •   PC1   •                              • • •
       •       •                                ↑
        •    •                                 PC2

Keep this direction!                 Can discard with little loss
```

---

## 2. Mathematical Formulation

### Covariance Matrix

For centered data $X$ (mean-subtracted):

$$C = \frac{1}{n-1} X^T X$$

The covariance matrix $C \in \mathbb{R}^{d \times d}$ captures:

- Diagonal: variance of each feature
- Off-diagonal: covariance between features

### PCA as Eigenvalue Problem

Principal components are **eigenvectors of the covariance matrix**:

$$C v_i = \lambda_i v_i$$

where:

- $v_i$ is the $i$-th principal component direction
- $\lambda_i$ is the variance along that direction
- Eigenvectors are sorted: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

### Projection

Project data onto first $k$ principal components:

$$Z = X V_k$$

where $V_k \in \mathbb{R}^{d \times k}$ contains top $k$ eigenvectors as columns.

---

## 3. PCA via SVD

### The Connection

For centered data matrix $X$:
$$X = U \Sigma V^T$$

Then:
$$X^T X = V \Sigma^2 V^T$$

This means:

- **Principal components** = columns of $V$ (right singular vectors)
- **Variances** = $\sigma_i^2 / (n-1)$ (squared singular values, normalized)
- **Projections** = $U \Sigma$ (scaled left singular vectors)

### Why SVD is Preferred

| Method             | Computation                                            |
| ------------------ | ------------------------------------------------------ |
| Covariance + Eigen | Form $X^TX$ (loses precision), then eigendecomposition |
| SVD                | Direct factorization, numerically more stable          |

---

## 4. Algorithm

### Step-by-Step PCA

```
Input: Data matrix X (n × d), number of components k

1. Center the data:
   X_centered = X - mean(X)

2. Compute SVD (or eigendecomposition):
   U, Σ, Vᵀ = SVD(X_centered)

3. Select top k components:
   V_k = first k columns of V

4. Project data:
   Z = X_centered @ V_k

5. (Optional) Reconstruct:
   X_reconstructed = Z @ V_k.T + mean(X)

Output: Reduced data Z (n × k), principal components V_k
```

### Complexity

| Step          | Complexity            |
| ------------- | --------------------- |
| Centering     | $O(nd)$               |
| Full SVD      | $O(\min(n^2d, nd^2))$ |
| Truncated SVD | $O(ndk)$              |
| Projection    | $O(ndk)$              |

---

## 5. Variance Explained

### Total Variance

$$\text{Total Variance} = \sum_{i=1}^{d} \lambda_i = \sum_{i=1}^{d} \frac{\sigma_i^2}{n-1}$$

### Variance Explained by Component

$$\text{Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

### Cumulative Variance

$$\text{Cumulative}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

```
Scree Plot:

Var  │ ▌
     │ ▌▌
     │ ▌▌▌
     │ ▌▌▌▌
     │ ▌▌▌▌▌_________
     └──────────────────
       1 2 3 4 5 6 7 8  Component

Look for "elbow" - where variance drops off
```

---

## 6. Choosing the Number of Components

### Methods

| Method             | Rule                                                         |
| ------------------ | ------------------------------------------------------------ |
| Variance threshold | Keep components until 90%, 95%, or 99% variance              |
| Scree plot / Elbow | Visual inspection for drop-off                               |
| Kaiser criterion   | Keep components with $\lambda_i > 1$ (for standardized data) |
| Cross-validation   | Evaluate downstream task performance                         |

### Reconstruction Error

$$\text{Error} = \|X - X_{\text{reconstructed}}\|_F^2 = \sum_{i=k+1}^{d} \lambda_i$$

---

## 7. Properties of PCA

### What PCA Does

1. **Decorrelates** features (covariance of Z is diagonal)
2. **Orders** components by importance (variance)
3. **Provides** orthonormal basis for data
4. **Minimizes** reconstruction error for given dimensionality

### What PCA Does NOT Do

1. Does NOT handle non-linear relationships (use kernel PCA)
2. Does NOT preserve distances (use MDS)
3. Does NOT model probability (use probabilistic PCA)
4. Does NOT work well with categorical data
5. Is sensitive to feature scaling

---

## 8. Standardization

### When to Standardize

| Scenario                        | Recommendation        |
| ------------------------------- | --------------------- |
| Features on same scale          | Centering only        |
| Features on different scales    | Standardize (z-score) |
| Want to preserve magnitude info | Centering only        |

### Z-Score Standardization

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

```python
X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)
```

**Important**: After standardization, each feature has variance 1, so:

- Kaiser criterion ($\lambda > 1$) makes sense
- All features contribute equally to PCA

---

## 9. Applications in ML/AI

### 1. Dimensionality Reduction

```
Original: 10,000 features → PCA → 100 components (99% variance)

Benefits:
- Faster training
- Reduced overfitting
- Lower storage
```

### 2. Data Visualization

Project to 2D or 3D for visualization:

```
High-D Data → PCA → 2D Plot

     •  • •           Class A
   •  • • •
              ○ ○     Class B
            ○ ○ ○
```

### 3. Noise Reduction

Keep only top components (signal), discard rest (noise):

$$X_{\text{denoised}} = X V_k V_k^T$$

### 4. Feature Extraction

PCA components as new features:

- Often more informative than original features
- Uncorrelated (no redundancy)

### 5. Eigenfaces (Face Recognition)

```
Training faces → PCA → Eigenfaces (principal components)
New face → Project onto eigenfaces → Compare with database
```

### 6. Preprocessing for Other Algorithms

PCA + Algorithm:

- PCA + Logistic Regression
- PCA + k-NN (faster in lower dimensions)
- PCA + Neural Networks (reduce input size)

---

## 10. Variants and Extensions

### Kernel PCA

Non-linear extension using kernel trick:
$$K_{ij} = k(x_i, x_j)$$

Applies PCA in high-dimensional feature space.

### Incremental PCA

For large datasets that don't fit in memory:

- Process data in mini-batches
- Update components incrementally

### Sparse PCA

Add sparsity constraint to loadings:

- More interpretable components
- Each component uses few original features

### Probabilistic PCA

Generative model:
$$x = Wz + \mu + \epsilon$$

where $z \sim \mathcal{N}(0, I)$ and $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

---

## 11. PCA vs. Other Methods

| Method       | Goal                  | Linear? | Preserves           |
| ------------ | --------------------- | ------- | ------------------- |
| PCA          | Max variance          | Yes     | Global structure    |
| LDA          | Max class separation  | Yes     | Discriminative info |
| t-SNE        | Neighbor preservation | No      | Local structure     |
| UMAP         | Manifold structure    | No      | Local + global      |
| Autoencoders | Reconstruction        | No      | Learned features    |

---

## 12. Summary

### Key Formulas

| Concept              | Formula                              |
| -------------------- | ------------------------------------ |
| Covariance matrix    | $C = \frac{1}{n-1}X^TX$              |
| Eigenvalue problem   | $Cv = \lambda v$                     |
| Projection           | $Z = XV_k$                           |
| Reconstruction       | $\hat{X} = ZV_k^T$                   |
| Variance explained   | $\frac{\lambda_i}{\sum_j \lambda_j}$ |
| Reconstruction error | $\sum_{i>k} \lambda_i$               |

### Algorithm Summary

```
PCA Algorithm:
1. Center (and optionally standardize) data
2. Compute covariance matrix C = XᵀX / (n-1)
3. Eigendecompose: C = VΛVᵀ (or use SVD)
4. Sort by eigenvalue descending
5. Select top k eigenvectors → V_k
6. Project: Z = X @ V_k
```

### Quick Decision Guide

```
When to use PCA:
✓ High-dimensional data
✓ Features are correlated
✓ Linear relationships dominate
✓ Need interpretable components

When NOT to use PCA:
✗ Non-linear relationships (use kernel PCA)
✗ Categorical data
✗ Need to preserve specific distances
✗ Interpretability of original features required
```

---

## Exercises

1. Implement PCA from scratch using eigendecomposition
2. Compare PCA via SVD vs eigendecomposition
3. Determine the number of components for 95% variance
4. Apply PCA to visualize the Iris dataset
5. Implement incremental PCA for streaming data

---

## References

1. Jolliffe, I.T. - "Principal Component Analysis"
2. Bishop, C.M. - "Pattern Recognition and Machine Learning"
3. Murphy, K.P. - "Machine Learning: A Probabilistic Perspective"
