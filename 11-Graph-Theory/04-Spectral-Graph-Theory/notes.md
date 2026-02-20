# Spectral Graph Theory

[← Previous: Graph Algorithms](../03-Graph-Algorithms) | [Next: Graph Neural Networks →](../05-Graph-Neural-Networks)

## Overview

Spectral graph theory studies graphs through the eigenvalues and eigenvectors of matrices associated with graphs. This mathematical framework is foundational for Graph Neural Networks, spectral clustering, and understanding graph structure.

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Learning Objectives

- Understand graph matrices and their spectra
- Apply spectral methods to graph problems
- Connect spectral theory to GNNs
- Implement spectral clustering

## 1. Graph Matrices

### Adjacency Matrix

For graph $G = (V, E)$ with $n$ vertices:

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**Properties:**

- Symmetric for undirected graphs
- $A^k_{ij}$ = number of walks of length $k$ from $i$ to $j$
- Eigenvalues are real for undirected graphs

### Degree Matrix

$$D = \text{diag}(d_1, d_2, ..., d_n)$$

where $d_i = \sum_j A_{ij}$ is the degree of vertex $i$.

### Laplacian Matrix

**Unnormalized Laplacian:**
$$L = D - A$$

$$L_{ij} = \begin{cases} d_i & \text{if } i = j \\ -1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

**Properties:**

- Symmetric positive semi-definite
- All eigenvalues $\geq 0$
- Smallest eigenvalue is always 0
- Number of zero eigenvalues = number of connected components

### Normalized Laplacians

**Symmetric normalized:**
$$L_{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

**Random walk normalized:**
$$L_{rw} = D^{-1} L = I - D^{-1} A$$

**Properties:**

- Eigenvalues in $[0, 2]$
- $L_{rw}$ relates to random walk transition matrix

## 2. Spectral Properties

### Eigenvalue Bounds

For adjacency matrix $A$:

- $|\lambda| \leq d_{max}$ (maximum degree)
- $\lambda_1 = d$ for $d$-regular graphs

For Laplacian $L$:

- $0 = \lambda_1 \leq \lambda_2 \leq ... \leq \lambda_n \leq 2d_{max}$

### Cheeger's Inequality

The **algebraic connectivity** $\lambda_2(L)$ (second smallest eigenvalue) relates to graph connectivity:

$$\frac{h^2}{2} \leq \lambda_2 \leq 2h$$

where $h$ is the Cheeger constant (minimum normalized cut).

**Interpretation:**

- Small $\lambda_2$ → graph has bottleneck (easy to cut)
- Large $\lambda_2$ → graph is well-connected

### Spectral Gap

$$\text{gap} = \lambda_2 - \lambda_1$$

For connected graphs, $\lambda_1 = 0$, so gap = $\lambda_2$.

**Applications:**

- Mixing time of random walks: $\propto 1/\text{gap}$
- Graph expansion properties
- Clustering quality

## 3. Graph Fourier Transform

### Classical Fourier Transform

For signals on $\mathbb{R}^n$, Fourier transform decomposes into complex exponentials (eigenfunctions of Laplacian on $\mathbb{R}^n$).

### Graph Fourier Transform

For signal $\mathbf{x} \in \mathbb{R}^n$ on graph:

$$\hat{\mathbf{x}} = U^T \mathbf{x}$$

where $U = [\mathbf{u}_1, ..., \mathbf{u}_n]$ are eigenvectors of $L$.

**Inverse transform:**
$$\mathbf{x} = U \hat{\mathbf{x}}$$

### Frequency Interpretation

- **Low frequency** (small $\lambda$): Smooth signals, vary slowly across edges
- **High frequency** (large $\lambda$): Signals that vary rapidly

**Graph smoothness:**
$$\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E} (x_i - x_j)^2$$

This is the **Dirichlet energy** - measures how much signal varies across edges.

## 4. Spectral Filtering

### Filtering in Spectral Domain

For filter function $g(\lambda)$:

$$\mathbf{y} = g(L) \mathbf{x} = U g(\Lambda) U^T \mathbf{x}$$

where $\Lambda = \text{diag}(\lambda_1, ..., \lambda_n)$.

**Low-pass filter:** $g(\lambda) = e^{-\alpha \lambda}$ (smoothing)

**High-pass filter:** $g(\lambda) = 1 - e^{-\alpha \lambda}$ (sharpening)

### Polynomial Filters

$$g(L) = \sum_{k=0}^{K} \theta_k L^k$$

**Advantages:**

- No eigendecomposition needed
- $K$-localized (depends on $K$-hop neighborhood)
- $O(K|E|)$ computation

### Chebyshev Polynomials

$$g(L) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L})$$

where $\tilde{L} = 2L/\lambda_{max} - I$ (scaled to $[-1, 1]$).

**Recurrence:**

- $T_0(x) = 1$
- $T_1(x) = x$
- $T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$

## 5. Spectral Clustering

### The Algorithm

1. Compute Laplacian $L$ (or $L_{sym}$)
2. Find $k$ smallest eigenvectors $\mathbf{u}_1, ..., \mathbf{u}_k$
3. Form matrix $U \in \mathbb{R}^{n \times k}$
4. Normalize rows (for $L_{sym}$)
5. Apply k-means to rows

### Why It Works

**Relaxed min-cut:** Minimizing:
$$\min_{\mathbf{f}} \mathbf{f}^T L \mathbf{f} \quad \text{s.t. } \mathbf{f} \perp \mathbf{1}, ||\mathbf{f}|| = 1$$

gives $\mathbf{f} = \mathbf{u}_2$ (Fiedler vector).

**Multi-way cut:** Use $k$ eigenvectors for $k$ clusters.

### Normalized Cuts

**RatioCut:**
$$\text{RatioCut}(A_1, ..., A_k) = \sum_{i=1}^k \frac{|E(A_i, \bar{A}_i)|}{|A_i|}$$

Use unnormalized Laplacian $L$.

**Normalized Cut:**
$$\text{NCut}(A_1, ..., A_k) = \sum_{i=1}^k \frac{|E(A_i, \bar{A}_i)|}{\text{vol}(A_i)}$$

Use normalized Laplacian $L_{sym}$.

## 6. Connection to GNNs

### Spectral GNNs

Original spectral convolution:
$$\mathbf{y} = g_\theta(L) \mathbf{x} = U g_\theta(\Lambda) U^T \mathbf{x}$$

**Problems:**

- $O(n^2)$ for eigendecomposition
- Not transferable (eigenvectors graph-specific)
- $O(n)$ parameters

### ChebNet

Use Chebyshev polynomials:
$$\mathbf{y} = \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) \mathbf{x}$$

**Advantages:**

- $O(K|E|)$ computation
- $K$-localized
- Only $K$ parameters

### GCN as Spectral Filter

GCN uses first-order approximation:
$$\mathbf{Y} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$$

where $\tilde{A} = A + I$ (self-loops).

This is equivalent to:
$$g(\lambda) = 2 - \lambda$$

A low-pass filter!

### Spatial vs Spectral

| Aspect          | Spectral                          | Spatial                  |
| --------------- | --------------------------------- | ------------------------ |
| Definition      | Filter in frequency domain        | Aggregate from neighbors |
| Basis           | Graph Laplacian eigenvectors      | Local neighborhood       |
| Computation     | Eigen-decomp or polynomial approx | Direct aggregation       |
| Transferability | Limited (eigenvectors vary)       | Better                   |

**Modern GNNs:** Mostly spatial, but inspired by spectral theory.

## 7. Important Theorems

### Courant-Fischer

$$\lambda_k = \min_{\dim(S)=k} \max_{\mathbf{x} \in S, ||\mathbf{x}||=1} \mathbf{x}^T L \mathbf{x}$$

### Interlacing

If $H$ is induced subgraph of $G$:
$$\lambda_i(G) \leq \lambda_i(H) \leq \lambda_{n-m+i}(G)$$

### Perron-Frobenius

For connected graph, adjacency matrix has:

- Unique largest eigenvalue $\lambda_1$
- Corresponding eigenvector has all positive entries

## 8. Applications Summary

| Application        | Key Spectral Property           |
| ------------------ | ------------------------------- |
| Clustering         | Small eigenvectors of $L$       |
| Graph partitioning | Fiedler vector ($\mathbf{u}_2$) |
| Random walk mixing | Spectral gap                    |
| GNN filters        | Polynomial approximation        |
| Graph embedding    | Laplacian eigenmaps             |
| Anomaly detection  | Spectral deviation              |

## Key Takeaways

1. **Laplacian eigenvalues** encode graph connectivity
2. **Eigenvectors** provide natural embedding coordinates
3. **$\lambda_2$** (algebraic connectivity) measures bottlenecks
4. **Spectral clustering** = eigenvector clustering
5. **GCN** is a first-order spectral filter
6. **Polynomial filters** avoid expensive eigendecomposition

## Practice Problems

1. Prove that $L = D - A$ is positive semi-definite
2. Compute the spectrum of a cycle graph $C_n$
3. Implement spectral clustering from scratch
4. Show that GCN aggregation is equivalent to a specific spectral filter
5. Compare Laplacian eigenmaps to PCA for graph embedding

## Exercises

1. **Laplacian Properties**: Verify the properties of unnormalized and normalized Laplacians on different graph types (complete, cycle, path)
2. **Spectral Clustering**: Implement k-way spectral clustering and compare results with k-means on graph node features
3. **Graph Fourier Analysis**: Compute the Graph Fourier Transform of a signal and visualize low vs high frequency components
4. **Chebyshev Filters**: Implement polynomial spectral filtering using Chebyshev polynomials without eigendecomposition
5. **Algebraic Connectivity**: Analyze the relationship between the Fiedler value and graph connectivity for random graphs
