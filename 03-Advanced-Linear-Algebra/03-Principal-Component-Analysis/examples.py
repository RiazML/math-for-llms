"""
Principal Component Analysis (PCA) - Examples
==============================================
Practical demonstrations of PCA concepts.
"""

import numpy as np
from numpy.linalg import eig, svd, norm
import matplotlib.pyplot as plt


def example_basic_pca():
    """Demonstrate basic PCA computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic PCA Computation")
    print("=" * 60)
    
    # 2D correlated data
    np.random.seed(42)
    n = 100
    
    # Generate correlated data
    t = np.random.randn(n)
    X = np.column_stack([
        2*t + 0.3*np.random.randn(n),
        t + 0.3*np.random.randn(n)
    ])
    
    print(f"Data shape: {X.shape}")
    print(f"Feature means: {X.mean(axis=0)}")
    
    # Step 1: Center the data
    X_centered = X - X.mean(axis=0)
    print(f"\nCentered means: {X_centered.mean(axis=0)}")
    
    # Step 2: Compute covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)
    print(f"\nCovariance matrix:\n{np.round(cov, 4)}")
    
    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = eig(cov)
    
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    eigenvectors = eigenvectors[:, idx].real
    
    print(f"\nEigenvalues (variances): {eigenvalues}")
    print(f"\nPrincipal components:\n{eigenvectors}")
    
    # Step 4: Project data
    Z = X_centered @ eigenvectors
    print(f"\nProjected data shape: {Z.shape}")
    print(f"Projected covariance:\n{np.round(np.cov(Z.T), 4)}")
    print("Note: Off-diagonal is ~0 (decorrelated!)")


def example_pca_via_svd():
    """Demonstrate PCA using SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: PCA via SVD")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    
    # Generate data
    t = np.random.randn(n)
    X = np.column_stack([
        2*t + 0.3*np.random.randn(n),
        t + 0.3*np.random.randn(n)
    ])
    
    # Center
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    print("SVD Results:")
    print(f"  Singular values: {s}")
    print(f"  V (principal components):\n{np.round(Vt.T, 4)}")
    
    # Compare with eigendecomposition
    cov = (X_centered.T @ X_centered) / (n - 1)
    eigenvalues, eigenvectors = eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx].real
    
    print(f"\nVariance from SVD: σ²/(n-1) = {s**2 / (n-1)}")
    print(f"Variance from eigen: {eigenvalues}")
    
    # Projection using SVD
    Z_svd = U * s  # or X_centered @ Vt.T
    Z_eigen = X_centered @ eigenvectors[:, idx].real
    
    print(f"\nProjections match: {np.allclose(np.abs(Z_svd), np.abs(Z_eigen))}")
    print("(May differ by sign)")


def example_variance_explained():
    """Demonstrate variance explained calculation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Variance Explained")
    print("=" * 60)
    
    # Higher dimensional data
    np.random.seed(42)
    n = 200
    d = 10
    
    # Create data with decreasing variance in each dimension
    X = np.random.randn(n, d)
    scales = np.array([10, 5, 3, 2, 1, 0.5, 0.3, 0.2, 0.1, 0.05])
    X = X * scales
    
    # Add some correlation
    X[:, 1] += 0.8 * X[:, 0]
    X[:, 2] += 0.5 * X[:, 0]
    
    # Center
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Variance explained
    variance = s**2 / (n - 1)
    total_variance = variance.sum()
    variance_ratio = variance / total_variance
    cumulative_variance = np.cumsum(variance_ratio)
    
    print("Variance Explained by Each Component:")
    print(f"{'PC':>4} {'Variance':>12} {'Ratio':>10} {'Cumulative':>12}")
    for i in range(d):
        print(f"{i+1:4d} {variance[i]:12.4f} {100*variance_ratio[i]:9.2f}% {100*cumulative_variance[i]:11.2f}%")
    
    # Find number of components for thresholds
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cumulative_variance, threshold) + 1
        print(f"\nComponents for {100*threshold:.0f}% variance: {k}")


def example_dimensionality_reduction():
    """Demonstrate dimensionality reduction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Dimensionality Reduction")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    d = 50  # High dimensional
    
    # Create low-rank data (intrinsically 3D) embedded in 50D
    true_dim = 3
    latent = np.random.randn(n, true_dim)
    mixing = np.random.randn(true_dim, d)
    X = latent @ mixing + 0.1 * np.random.randn(n, d)
    
    print(f"Original data: {X.shape}")
    print(f"True intrinsic dimension: {true_dim}")
    
    # Center and SVD
    X_centered = X - X.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Variance analysis
    variance_ratio = (s**2) / np.sum(s**2)
    print(f"\nTop 10 singular values: {np.round(s[:10], 4)}")
    print(f"Top 10 variance ratios: {np.round(100*variance_ratio[:10], 2)}%")
    
    # Reduce to k dimensions
    k = 3
    V_k = Vt[:k, :].T
    Z = X_centered @ V_k
    
    print(f"\nReduced to {k} dimensions: {Z.shape}")
    
    # Reconstruction error
    X_reconstructed = Z @ V_k.T + X.mean(axis=0)
    error = norm(X - X_reconstructed, 'fro') / norm(X, 'fro')
    print(f"Relative reconstruction error: {error:.4f}")
    print(f"Variance captured: {100*np.sum(variance_ratio[:k]):.2f}%")


def example_whitening():
    """Demonstrate PCA whitening (decorrelation + scaling)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: PCA Whitening")
    print("=" * 60)
    
    np.random.seed(42)
    n = 500
    
    # Generate correlated data
    cov_true = np.array([[4, 2],
                         [2, 3]])
    X = np.random.multivariate_normal([0, 0], cov_true, n)
    
    print("Original data:")
    print(f"  Shape: {X.shape}")
    print(f"  Covariance:\n{np.round(np.cov(X.T), 4)}")
    
    # Center
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Standard PCA projection
    Z_pca = U * s
    print("\nAfter PCA projection:")
    print(f"  Covariance:\n{np.round(np.cov(Z_pca.T), 4)}")
    print("  (Decorrelated but different variances)")
    
    # Whitening: divide by singular values (normalize variance)
    Z_white = U  # Just U, without scaling by s
    # Or equivalently: Z_white = X_centered @ Vt.T @ np.diag(1/s)
    
    print("\nAfter whitening:")
    cov_white = np.cov(Z_white.T) * (n-1)  # Adjust for sample cov
    print(f"  Covariance:\n{np.round(cov_white, 4)}")
    print("  (Identity matrix - decorrelated and unit variance)")


def example_noise_reduction():
    """Demonstrate noise reduction using PCA."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Noise Reduction")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    d = 20
    
    # Create clean low-rank signal (rank 3)
    true_rank = 3
    U_true = np.random.randn(n, true_rank)
    V_true = np.random.randn(true_rank, d)
    X_clean = U_true @ V_true
    
    # Add noise
    noise_level = 0.5
    noise = noise_level * np.random.randn(n, d)
    X_noisy = X_clean + noise
    
    print(f"Clean signal rank: {true_rank}")
    print(f"Noise level: {noise_level}")
    print(f"SNR: {norm(X_clean, 'fro') / norm(noise, 'fro'):.2f}")
    
    # Center and SVD
    X_centered = X_noisy - X_noisy.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    print(f"\nTop 10 singular values: {np.round(s[:10], 4)}")
    
    # Denoise by keeping top k components
    k = true_rank
    X_denoised = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :] + X_noisy.mean(axis=0)
    
    print(f"\nUsing k={k} components:")
    print(f"  Noisy error: ||X_noisy - X_clean||_F = {norm(X_noisy - X_clean, 'fro'):.4f}")
    print(f"  Denoised error: ||X_denoised - X_clean||_F = {norm(X_denoised - X_clean, 'fro'):.4f}")
    print(f"  Improvement: {100*(1 - norm(X_denoised - X_clean, 'fro') / norm(X_noisy - X_clean, 'fro')):.1f}%")


def example_reconstruction():
    """Demonstrate reconstruction from principal components."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Reconstruction")
    print("=" * 60)
    
    np.random.seed(42)
    n = 50
    d = 10
    
    # Generate data
    X = np.random.randn(n, d)
    X[:, 0] *= 3  # First feature has more variance
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n)  # Correlated
    
    # Center
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # SVD
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    print("Reconstruction with different numbers of components:")
    print(f"{'k':>4} {'Rel. Error':>12} {'Variance Kept':>14}")
    
    total_var = np.sum(s**2)
    
    for k in [1, 2, 3, 5, 10]:
        # Project and reconstruct
        Z_k = U[:, :k] * s[:k]
        X_reconstructed = Z_k @ Vt[:k, :] + mean
        
        rel_error = norm(X - X_reconstructed, 'fro') / norm(X, 'fro')
        var_kept = np.sum(s[:k]**2) / total_var
        
        print(f"{k:4d} {rel_error:12.4f} {100*var_kept:13.2f}%")
    
    # Show a sample reconstruction
    print("\nSample point reconstruction (k=2):")
    sample_idx = 0
    print(f"  Original:      {np.round(X[sample_idx], 2)}")
    
    k = 2
    Z_k = U[:, :k] * s[:k]
    X_reconstructed = Z_k @ Vt[:k, :] + mean
    print(f"  Reconstructed: {np.round(X_reconstructed[sample_idx], 2)}")


def example_standardization_effect():
    """Show effect of standardization on PCA."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Effect of Standardization")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    
    # Features on very different scales
    X = np.column_stack([
        1000 + 100 * np.random.randn(n),  # Large scale
        10 + 1 * np.random.randn(n),       # Small scale
        0.1 + 0.01 * np.random.randn(n)    # Tiny scale
    ])
    
    print("Feature statistics:")
    print(f"  Means: {X.mean(axis=0)}")
    print(f"  Stds:  {X.std(axis=0)}")
    
    # PCA without standardization
    X_centered = X - X.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    var_ratio = s**2 / np.sum(s**2)
    
    print("\nPCA without standardization:")
    print(f"  Variance ratios: {np.round(100*var_ratio, 2)}%")
    print("  (First PC dominates due to large scale!)")
    
    # PCA with standardization
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    U_std, s_std, Vt_std = svd(X_std, full_matrices=False)
    var_ratio_std = s_std**2 / np.sum(s_std**2)
    
    print("\nPCA with standardization:")
    print(f"  Variance ratios: {np.round(100*var_ratio_std, 2)}%")
    print("  (More balanced contribution from all features)")


def example_incremental_pca():
    """Demonstrate incremental PCA for large datasets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Incremental PCA")
    print("=" * 60)
    
    np.random.seed(42)
    n = 1000
    d = 50
    k = 5  # Number of components
    
    # Generate full dataset
    X = np.random.randn(n, d)
    X[:, 0] *= 5  # Add structure
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n)
    
    # Full PCA for reference
    X_centered = X - X.mean(axis=0)
    _, s_full, Vt_full = svd(X_centered, full_matrices=False)
    
    print("Full PCA (for reference):")
    print(f"  Top {k} singular values: {np.round(s_full[:k], 4)}")
    
    # Incremental PCA simulation (simplified version)
    # In practice, use sklearn.decomposition.IncrementalPCA
    batch_size = 100
    n_batches = n // batch_size
    
    print(f"\nIncremental PCA with batch_size={batch_size}:")
    
    # Running mean
    running_mean = np.zeros(d)
    running_count = 0
    
    # Process first batch for initialization
    batch = X[:batch_size]
    running_mean = batch.mean(axis=0)
    running_count = batch_size
    
    batch_centered = batch - running_mean
    U_inc, s_inc, Vt_inc = svd(batch_centered, full_matrices=False)
    s_inc = s_inc[:k]
    Vt_inc = Vt_inc[:k, :]
    
    # Process remaining batches
    for i in range(1, n_batches):
        batch = X[i*batch_size:(i+1)*batch_size]
        
        # Update mean
        batch_mean = batch.mean(axis=0)
        new_count = running_count + batch_size
        running_mean = (running_count * running_mean + batch_size * batch_mean) / new_count
        running_count = new_count
        
        # Center batch
        batch_centered = batch - running_mean
        
        # Simplified update (in practice, more sophisticated)
        # Combine old components with new batch
        combined = np.vstack([s_inc.reshape(1, -1) * Vt_inc, batch_centered @ Vt_inc.T])
        _, s_inc, Vh = svd(combined, full_matrices=False)
        s_inc = s_inc[:k]
    
    print(f"  Incremental top {k} singular values: {np.round(s_inc, 4)}")
    print(f"  Full PCA top {k}: {np.round(s_full[:k], 4)}")
    print("\nNote: Real incremental PCA uses more sophisticated updates")


def example_kernel_pca_preview():
    """Preview of Kernel PCA for non-linear data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Kernel PCA Preview")
    print("=" * 60)
    
    np.random.seed(42)
    n = 200
    
    # Generate concentric circles (non-linearly separable)
    theta = np.random.uniform(0, 2*np.pi, n)
    r_inner = 1 + 0.1 * np.random.randn(n//2)
    r_outer = 3 + 0.1 * np.random.randn(n//2)
    
    X_inner = np.column_stack([r_inner * np.cos(theta[:n//2]), 
                                r_inner * np.sin(theta[:n//2])])
    X_outer = np.column_stack([r_outer * np.cos(theta[n//2:]), 
                                r_outer * np.sin(theta[n//2:])])
    
    X = np.vstack([X_inner, X_outer])
    y = np.array([0] * (n//2) + [1] * (n//2))  # Labels
    
    print("Concentric circles dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: 2 (inner and outer circles)")
    
    # Standard PCA
    X_centered = X - X.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    Z_pca = U * s
    
    print("\nStandard PCA:")
    print(f"  Variance ratios: {np.round(100 * s**2 / np.sum(s**2), 2)}%")
    print("  Cannot separate circles in 2D!")
    
    # Kernel PCA with RBF kernel (simplified)
    gamma = 0.5
    
    # Compute kernel matrix
    from scipy.spatial.distance import cdist
    dists = cdist(X, X, 'sqeuclidean')
    K = np.exp(-gamma * dists)
    
    # Center kernel matrix
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    
    # Eigendecomposition of centered kernel
    eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Project to kernel space
    Z_kpca = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    print("\nKernel PCA (RBF kernel, γ=0.5):")
    print(f"  Top 2 eigenvalues: {np.round(eigenvalues[:2], 4)}")
    print("  Can now separate circles in projected space!")
    
    # Check separation
    inner_mean = Z_kpca[:n//2, 0].mean()
    outer_mean = Z_kpca[n//2:, 0].mean()
    print(f"\n  PC1 mean (inner): {inner_mean:.4f}")
    print(f"  PC1 mean (outer): {outer_mean:.4f}")
    print(f"  Separation: {abs(inner_mean - outer_mean):.4f}")


def visualize_pca_2d():
    """Visualize PCA on 2D data."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: PCA on 2D Data")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    
    # Generate correlated 2D data
    cov = [[2, 1.5], [1.5, 2]]
    X = np.random.multivariate_normal([0, 0], cov, n)
    
    # PCA
    X_centered = X - X.mean(axis=0)
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original data with PC directions
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5)
    
    # Plot principal components
    mean = X.mean(axis=0)
    for i in range(2):
        pc = Vt[i, :] * s[i] / np.sqrt(n)
        ax.arrow(mean[0], mean[1], pc[0]*2, pc[1]*2, 
                head_width=0.2, head_length=0.1, fc=f'C{i+1}', ec=f'C{i+1}',
                linewidth=2, label=f'PC{i+1} (var={s[i]**2/(n-1):.2f})')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_title('Original Data with Principal Components')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Projected data
    ax = axes[1]
    Z = X_centered @ Vt.T
    ax.scatter(Z[:, 0], Z[:, 1], alpha=0.5)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Data in PC Space (Decorrelated)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Variance explained
    ax = axes[2]
    variance_ratio = (s**2) / np.sum(s**2)
    ax.bar([1, 2], variance_ratio * 100, color=['C1', 'C2'])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Scree Plot')
    ax.set_xticks([1, 2])
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: pca_visualization.png")


if __name__ == "__main__":
    example_basic_pca()
    example_pca_via_svd()
    example_variance_explained()
    example_dimensionality_reduction()
    example_whitening()
    example_noise_reduction()
    example_reconstruction()
    example_standardization_effect()
    example_incremental_pca()
    example_kernel_pca_preview()
    
    # Uncomment to generate visualization
    # visualize_pca_2d()
