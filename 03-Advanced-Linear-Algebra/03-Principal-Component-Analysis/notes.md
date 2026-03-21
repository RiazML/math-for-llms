[← Back to Advanced Linear Algebra](../README.md) | [← SVD](../02-Singular-Value-Decomposition/notes.md) | [Linear Transformations →](../04-Linear-Transformations/notes.md)

---

# Principal Component Analysis (PCA)

## Introduction

Principal Component Analysis (PCA) is the most fundamental dimensionality reduction technique in machine learning. It finds orthogonal axes (principal components) that capture maximum variance, enabling compression, visualization, noise reduction, and feature extraction.

**Why PCA Matters for AI/ML:**
- Reduces computational cost for downstream models
- Mitigates curse of dimensionality
- Removes multicollinearity
- Provides interpretable linear combinations of features
- Foundation for many advanced techniques (sparse PCA, kernel PCA, probabilistic PCA)

## Prerequisites

- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Covariance matrices
- Matrix operations

## Learning Objectives

1. Understand PCA geometrically and algebraically
2. Derive PCA from covariance eigendecomposition
3. Implement PCA using SVD
4. Choose the optimal number of components
5. Apply PCA to real-world ML problems
6. Understand when PCA fails and alternatives

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
2D Point Cloud:                    After PCA rotation:

       •  •                               •  •
      • \ •  •                           •  •  •
     •   \   •                          •  •  •  •
    •     \   •        →               •  •  •  •  •
   •    PC1\   •                      ───────────────
  •    ─────\──•                          PC1 axis
       PC2↑   •                     (max variance direction)

The data ellipse's major axis = PC1
               minor axis = PC2
```

**Key Insight**: PCA finds a rotation that aligns data with its principal axes of variation.

### Three Equivalent Views

| View | Goal | Result |
|------|------|--------|
| **Variance Maximization** | Find direction maximizing projected variance | First eigenvector |
| **Reconstruction Error** | Find projection minimizing reconstruction loss | Same eigenvector |
| **Covariance Diagonalization** | Find rotation decorrelating features | Eigendecomposition |

---

## 2. Mathematical Formulation

### Covariance Matrix

For centered data $X$ (mean-subtracted):

$$C = \frac{1}{n-1} X^T X$$

The covariance matrix $C \in \mathbb{R}^{d \times d}$ captures:
- **Diagonal**: variance of each feature
- **Off-diagonal**: covariance between feature pairs

```
       Feature 1  Feature 2  Feature 3
          │          │          │
          ▼          ▼          ▼
C = [ var(f1)    cov(f1,f2) cov(f1,f3) ]
    [ cov(f2,f1)   var(f2)  cov(f2,f3) ]
    [ cov(f3,f1) cov(f3,f2)   var(f3)  ]
```

### Variance Maximization Derivation

**Goal**: Find unit vector $v$ maximizing variance of projected data.

Projected data: $z = Xv$

Variance of projection: $\text{Var}(z) = v^T C v$

**Optimization problem**:
$$\max_v v^T C v \quad \text{subject to} \quad \|v\|_2 = 1$$

**Solution** (via Lagrange multipliers):
$$Cv = \lambda v$$

This is the eigenvalue equation! The optimal $v$ is the eigenvector with largest eigenvalue.

### PCA as Eigenvalue Problem

Principal components are **eigenvectors of the covariance matrix**:

$$C v_i = \lambda_i v_i$$

where:
- $v_i$ is the $i$-th principal component direction (unit vector)
- $\lambda_i$ is the variance along that direction
- Sorted: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$

### Projection

Project data onto first $k$ principal components:

$$Z = X_{centered} V_k$$

where $V_k \in \mathbb{R}^{d \times k}$ contains top $k$ eigenvectors as columns.

---

## 3. PCA via SVD

### The Connection

For centered data matrix $X$ with SVD:
$$X = U \Sigma V^T$$

The covariance matrix becomes:
$$C = \frac{1}{n-1}X^T X = \frac{1}{n-1}V \Sigma^T \Sigma V^T = V \frac{\Sigma^2}{n-1} V^T$$

This means:
- **Principal components** = columns of $V$ (right singular vectors)
- **Variances** = $\sigma_i^2 / (n-1)$ (squared singular values, normalized)
- **Projections** = $U \Sigma$ (or equivalently $X V$)

### Complete Relationship

| PCA Quantity | From Eigen | From SVD |
|--------------|------------|----------|
| Principal components | Eigenvectors of $C$ | Columns of $V$ |
| Variances | Eigenvalues $\lambda_i$ | $\sigma_i^2 / (n-1)$ |
| Projected data | $X V$ | $U \Sigma$ |
| Reconstruction | $Z V^T$ | $U_k \Sigma_k V_k^T$ |

### Why SVD is Preferred

| Aspect | Covariance + Eigen | SVD |
|--------|-------------------|-----|
| Numerical stability | Lower (forms $X^TX$) | Higher (direct) |
| Memory | $O(d^2)$ for $C$ | Works on $X$ directly |
| For $n < d$ | Wasteful | More efficient |
| Sign consistency | May vary | More stable |

### NumPy Implementation Comparison

```python
# Method 1: Eigendecomposition
X_centered = X - X.mean(axis=0)
C = (X_centered.T @ X_centered) / (n - 1)
eigenvalues, eigenvectors = np.linalg.eig(C)
idx = np.argsort(eigenvalues)[::-1]
V = eigenvectors[:, idx].real

# Method 2: SVD (preferred)
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T
variances = s**2 / (n - 1)
```

---

## 4. Algorithm

### Step-by-Step PCA

```
Input: Data matrix X (n × d), number of components k

1. Center the data:
   mean = X.mean(axis=0)
   X_centered = X - mean

2. (Optional) Standardize:
   std = X.std(axis=0)
   X_std = X_centered / std

3. Compute SVD:
   U, Σ, Vᵀ = SVD(X_centered)

4. Select top k components:
   V_k = Vᵀ[:k, :].T   # (d × k)

5. Project data:
   Z = X_centered @ V_k  # (n × k)

6. (Optional) Reconstruct:
   X_reconstructed = Z @ V_k.T + mean

Output: Reduced data Z, principal components V_k, variances s**2/(n-1)
```

### Complete NumPy Implementation

```python
def pca(X, k=None, standardize=False):
    """
    Principal Component Analysis.
    
    Parameters:
    - X: (n, d) data matrix
    - k: number of components (default: all)
    - standardize: whether to z-score normalize
    
    Returns:
    - Z: projected data (n, k)
    - V_k: principal components (d, k)
    - var_ratio: variance explained ratio
    """
    n, d = X.shape
    k = k or d
    
    # Center
    mean = X.mean(axis=0)
    X_c = X - mean
    
    # Optionally standardize
    if standardize:
        std = X.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        X_c = X_c / std
    
    # SVD
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    
    # Select k components
    V_k = Vt[:k, :].T
    
    # Project
    Z = X_c @ V_k
    
    # Variance explained
    var_ratio = (s[:k]**2) / np.sum(s**2)
    
    return Z, V_k, var_ratio
```

### Complexity Analysis

| Step | Time Complexity | Space Complexity |
|------|-----------------|------------------|
| Centering | $O(nd)$ | $O(nd)$ |
| Full SVD | $O(\min(n^2d, nd^2))$ | $O(\min(n, d)^2)$ |
| Truncated SVD | $O(ndk)$ | $O(nk + dk)$ |
| Projection | $O(ndk)$ | $O(nk)$ |

For large datasets, use **truncated/randomized SVD** (sklearn's `TruncatedSVD`).

---

## 5. Variance Explained

### Total Variance

$$\text{Total Variance} = \sum_{i=1}^{d} \lambda_i = \sum_{i=1}^{d} \frac{\sigma_i^2}{n-1} = \text{trace}(C)$$

### Variance Explained by Component

$$\text{Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j} = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

### Cumulative Variance

$$\text{Cumulative}(k) = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

### Scree Plot

```
Individual                          Cumulative
Variance %                          Variance %

40% │▌                              100%│           --------
30% │▌▌                              80%│      ----/
20% │▌▌▌                             60%│    -/
10% │▌▌▌▌▌___                        40%│  -/      95% threshold
    └─────────────                      │ /        ───────────
      1 2 3 4 5 6 7 8                   └/────────────────────
            PC                            1 2 3 4 5 6 7 8 PC
              ↑
         "Elbow" point
```

### NumPy Implementation

```python
def variance_analysis(X):
    """Analyze variance explained by each component."""
    X_c = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(X_c, full_matrices=False)
    
    variance = s**2 / (len(X) - 1)
    var_ratio = variance / variance.sum()
    cumulative = np.cumsum(var_ratio)
    
    return var_ratio, cumulative
```

---

## 6. Choosing the Number of Components

### Methods Comparison

| Method | Rule | Best For |
|--------|------|----------|
| **Variance threshold** | Keep until 90%, 95%, 99% | General use |
| **Scree plot / Elbow** | Visual drop in eigenvalues | Exploratory analysis |
| **Kaiser criterion** | Keep $\lambda_i > 1$ (standardized) | Standardized data |
| **Cross-validation** | Optimize downstream metric | Supervised learning |
| **Parallel analysis** | Compare to random data | Statistical rigor |

### Variance Threshold Implementation

```python
def components_for_variance(s, threshold=0.95):
    """Find k for given cumulative variance threshold."""
    var_ratio = s**2 / np.sum(s**2)
    cumulative = np.cumsum(var_ratio)
    k = np.searchsorted(cumulative, threshold) + 1
    return k
```

### Kaiser Criterion

For **standardized data**, each original feature has variance 1.
Keep components where $\lambda_i > 1$ (explains more than single feature).

```python
def kaiser_criterion(X_standardized):
    """Apply Kaiser criterion."""
    n = len(X_standardized)
    _, s, _ = np.linalg.svd(X_standardized, full_matrices=False)
    eigenvalues = s**2 / (n - 1)
    k = np.sum(eigenvalues > 1)
    return k
```

### Reconstruction Error Perspective

$$\text{Reconstruction Error} = \|X - \hat{X}_k\|_F^2 = \sum_{i=k+1}^{d} \sigma_i^2$$

Choose $k$ where error is "acceptable" for your application.

---

## 7. Properties of PCA

### What PCA Achieves

| Property | Explanation |
|----------|-------------|
| **Decorrelation** | $\text{Cov}(Z) = \text{diag}(\lambda_1, \ldots, \lambda_k)$ |
| **Ordering** | PC1 has most variance, PC2 second most, etc. |
| **Orthonormal basis** | Principal components are orthonormal |
| **Optimal reconstruction** | Minimizes MSE for linear projection to $k$ dims |
| **Maximum variance** | Each PC captures max remaining variance |

### Mathematical Guarantees

**Eckart-Young-Mirsky Theorem**: The rank-$k$ approximation from SVD minimizes Frobenius norm error:

$$\hat{X}_k = \underset{\text{rank}(M) = k}{\arg\min} \|X - M\|_F$$

### What PCA Does NOT Do

| Limitation | Explanation | Alternative |
|------------|-------------|-------------|
| Non-linear patterns | Only captures linear relationships | Kernel PCA, t-SNE, UMAP |
| Distance preservation | Distances may distort | MDS, Isomap |
| Probability modeling | No generative model | Probabilistic PCA |
| Categorical data | Requires numerical input | MCA, embeddings |
| Scale sensitivity | Large features dominate | Standardization |
| Class separation | Unsupervised (ignores labels) | LDA |

---

## 8. Standardization and Preprocessing

### When to Standardize

| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Features on same scale | Centering only | Preserve relative magnitudes |
| Different scales/units | **Standardize** | Equal contribution |
| Interpretability needed | Standardize | Loadings comparable |
| Kaiser criterion | Standardize | Makes eigenvalue threshold meaningful |

### Z-Score Standardization

$$x_{\text{std}} = \frac{x - \mu}{\sigma}$$

```python
# Standardization
mean = X.mean(axis=0)
std = X.std(axis=0)
X_std = (X - mean) / std
```

### Effect of Not Standardizing

```
Original:                           Without standardization:
  Feature 1: range [0, 1000]          PC1 ≈ [1, 0, 0]  (dominated by F1!)
  Feature 2: range [0, 10]            PC2 ≈ [0, 1, 0]
  Feature 3: range [0, 0.1]           PC3 ≈ [0, 0, 1]

With standardization:
  All features: mean=0, std=1         PC1, PC2, PC3 reflect actual correlations
```

---

## 9. PCA Whitening

### What is Whitening?

Whitening transforms data to have:
- **Zero mean** (centering)
- **Unit variance** in all PC directions
- **Zero correlation** (decorrelation)

### Whitening Transform

$$X_{white} = U = X V \Sigma^{-1}$$

Or equivalently: project to PCA space, then divide by standard deviation:

$$Z_{white} = Z \cdot \text{diag}(1/\sqrt{\lambda_1}, \ldots, 1/\sqrt{\lambda_k})$$

### Properties

After whitening:
$$\text{Cov}(X_{white}) = I$$

### NumPy Implementation

```python
def whiten(X):
    """Apply PCA whitening."""
    X_c = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    
    # Whitened data is just U (orthonormal)
    return U
```

### Applications of Whitening

- **Independent Component Analysis (ICA)**: Preprocessing step
- **Neural Networks**: Can speed up training
- **Computer Vision**: ZCA whitening for images
- **Sphering**: Makes data "spherical" (isotropic)

---

## 10. Applications in ML/AI

### 1. Dimensionality Reduction

```
Original: 10,000 features → PCA → 100 components (99% variance)

Benefits:
✓ Faster training (O(nk) vs O(nd))
✓ Reduced overfitting
✓ Lower storage requirements
✓ Removes multicollinearity
```

### 2. Data Visualization

Project to 2D/3D for visualization:

```python
Z_2d = X @ V[:, :2]  # Project to first 2 PCs
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=labels)
```

### 3. Noise Reduction / Denoising

```python
# Keep top k components (signal), discard rest (noise)
def denoise(X, k):
    mean = X.mean(axis=0)
    X_c = X - mean
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_denoised = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :] + mean
    return X_denoised
```

### 4. Feature Extraction

PCA components as interpretable features:
- Bank data: PC1 might capture "wealth", PC2 "spending behavior"
- Images: Eigenfaces capture facial variations
- Sensors: PC1 might capture overall activity level

### 5. Compression

Store only:
- Mean vector: $\bar{x}$ ($d$ values)
- Top $k$ components: $V_k$ ($dk$ values)
- Projections: $Z$ ($nk$ values)

Compression ratio: $\frac{nd}{d + dk + nk} \approx \frac{d}{k}$ for large $n$

### 6. Preprocessing for ML Models

```python
# Common pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('classifier', LogisticRegression())
])
```

---

## 11. Variants and Extensions

### Kernel PCA

**Problem**: Standard PCA only captures linear relationships.

**Solution**: Apply PCA in a high-dimensional feature space via kernel trick.

$$K_{ij} = k(x_i, x_j) = \phi(x_i)^T \phi(x_j)$$

**Common kernels**:
- RBF: $k(x, y) = \exp(-\gamma \|x-y\|^2)$
- Polynomial: $k(x, y) = (x^T y + c)^d$

**Algorithm**:
1. Compute kernel matrix $K$
2. Center kernel: $\tilde{K} = (I - \frac{1}{n}11^T) K (I - \frac{1}{n}11^T)$
3. Eigendecompose $\tilde{K}$
4. Project: $\alpha_i = \frac{1}{\sqrt{\lambda_i}} v_i$

### Sparse PCA

**Goal**: Find principal components with sparse loadings (many zeros).

**Benefit**: More interpretable—each PC uses only a few original features.

**Formulation**: Add L1 penalty:
$$\min_V \|X - XV V^T\|_F^2 + \lambda \|V\|_1$$

### Incremental PCA

**Problem**: Data too large for memory.

**Solution**: Process in mini-batches, update components incrementally.

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=50, batch_size=1000)
for batch in data_generator:
    ipca.partial_fit(batch)
```

### Probabilistic PCA (PPCA)

**Generative model**:
$$x = Wz + \mu + \epsilon$$

where $z \sim \mathcal{N}(0, I)$, $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

**Benefits**:
- Handles missing data
- Provides likelihood scores
- Enables mixture models (Mixture of PPCA)

### Robust PCA

**Problem**: Outliers corrupt standard PCA.

**Solution**: Decompose $X = L + S$ where $L$ is low-rank and $S$ is sparse (outliers).

---

## 12. PCA vs. Other Dimensionality Reduction

### Comparison Table

| Method | Type | Preserves | Best For |
|--------|------|-----------|----------|
| **PCA** | Linear | Global variance | General purpose, fast |
| **LDA** | Linear, supervised | Class separation | Classification pre-processing |
| **t-SNE** | Non-linear | Local neighborhoods | Visualization (2D/3D) |
| **UMAP** | Non-linear | Local + global | Visualization + analysis |
| **Autoencoders** | Non-linear | Learned features | Complex patterns |
| **ICA** | Linear | Statistical independence | Signal separation |
| **MDS** | Various | Pairwise distances | Distance-based data |

### When to Choose PCA

**Use PCA when**:
- Data has linear relationships
- Speed is important
- Interpretability of components matters
- Need preprocessing for other algorithms
- Working with high-dimensional numerical data

**Avoid PCA when**:
- Data lies on a non-linear manifold
- Local structure is more important than global
- Need to separate classes (use LDA instead)
- Data is categorical

---

## 13. Practical Tips and Common Pitfalls

### Best Practices

1. **Always center** (subtract mean) before PCA
2. **Standardize** if features have different scales
3. **Use SVD** instead of eigendecomposition
4. **Check for outliers** before PCA (consider robust PCA)
5. **Visualize** scree plot and cumulative variance
6. **Cross-validate** component count if using for prediction

### Common Mistakes

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Not centering | Shifts origin, wrong covariance | Always center |
| Ignoring scale | Large features dominate | Standardize |
| Using covariance eigendecomposition | Less stable numerically | Use SVD |
| Applying training mean to test data | Data leakage | Save and reuse training statistics |
| Choosing k arbitrarily | May lose information or include noise | Use variance threshold |
| Interpreting after rotation | PCs can flip sign | Focus on absolute loadings |

### Handling New Data

```python
# Training
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_std = (X_train - mean) / std
U, s, Vt = np.linalg.svd(X_train_std, full_matrices=False)
V_k = Vt[:k].T

# Inference (use TRAINING statistics!)
X_test_std = (X_test - mean) / std
Z_test = X_test_std @ V_k
```

---

## 14. Connection to Neural Networks

### PCA as a Linear Autoencoder

```
Input (d)  →  Encoder (d→k)  →  Latent (k)  →  Decoder (k→d)  →  Output (d)
   X           Z = XW              Z              X̂ = ZWᵀ           X̂

With linear activations and MSE loss:
- Optimal W spans same subspace as top k PCs
- Actual W may differ from V by rotation
```

### Why Non-linear Autoencoders Are More Powerful

- PCA: Linear compression only
- Autoencoders: Can learn non-linear manifolds
- But: Non-linear = harder to train, less interpretable

### Relationship to Other Concepts

| Neural Network Concept | PCA Equivalent |
|------------------------|----------------|
| Linear autoencoder | PCA |
| Encoder weights | Principal components |
| Latent representation | Projected data |
| Reconstruction loss | PCA objective |

---

## 15. Summary

### Key Formulas

| Concept | Formula | NumPy |
|---------|---------|-------|
| Covariance | $C = \frac{1}{n-1}X^TX$ | `np.cov(X.T)` |
| Eigenvalue problem | $Cv = \lambda v$ | `np.linalg.eig(C)` |
| SVD | $X = U\Sigma V^T$ | `np.linalg.svd(X)` |
| Projection | $Z = XV_k$ | `X @ Vt[:k].T` |
| Reconstruction | $\hat{X} = ZV_k^T$ | `Z @ Vt[:k]` |
| Variance ratio | $\lambda_i / \sum \lambda_j$ | `(s**2) / sum(s**2)` |
| Whitening | $X_{white} = U$ | `U` from SVD |

### Algorithm Summary

```
PCA Algorithm:
1. Center: X_c = X - mean(X)
2. (Optional) Standardize: X_c = X_c / std(X)
3. SVD: U, s, Vt = svd(X_c)
4. Select k components: V_k = Vt[:k].T
5. Project: Z = X_c @ V_k
6. Variance explained: var_ratio = s[:k]**2 / sum(s**2)
```

### Decision Guide

```
When to use PCA:
✓ High-dimensional numerical data
✓ Features are correlated
✓ Linear relationships dominate
✓ Need fast, interpretable dimensionality reduction
✓ Preprocessing for other algorithms

When to consider alternatives:
→ Non-linear relationships: Kernel PCA, autoencoders
→ Class separation: LDA
→ Visualization: t-SNE, UMAP
→ Outliers present: Robust PCA
→ Sparse components wanted: Sparse PCA
```

---

## Exercises

1. Implement PCA from scratch using eigendecomposition
2. Compare PCA via SVD vs eigendecomposition numerically
3. Determine the number of components for 95% variance
4. Implement and compare PCA with/without standardization
5. Use PCA + nearest centroid for a classification task
6. Verify reconstruction error equals discarded eigenvalues
7. Implement Kernel PCA with RBF kernel
8. Compare PCA vs random projection for dimensionality reduction
9. Analyze and interpret principal component loadings
10. Implement incremental PCA for batch processing

---

## References

1. Jolliffe, I.T. - "Principal Component Analysis" (2002)
2. Bishop, C.M. - "Pattern Recognition and Machine Learning" (2006)
3. Murphy, K.P. - "Machine Learning: A Probabilistic Perspective" (2012)
4. Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning" (2009)
5. Tipping, M.E. & Bishop, C.M. - "Probabilistic Principal Component Analysis" (1999)

---

[← Back to Advanced Linear Algebra](../README.md) | [← SVD](../02-Singular-Value-Decomposition/notes.md) | [Linear Transformations →](../04-Linear-Transformations/notes.md)
