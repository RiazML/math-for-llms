"""
Singular Value Decomposition (SVD) - Examples
==============================================
Practical demonstrations of SVD concepts.
"""

import numpy as np
from numpy.linalg import svd, norm, matrix_rank, pinv
import matplotlib.pyplot as plt


def example_basic_svd():
    """Demonstrate basic SVD computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic SVD Computation")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    
    print(f"Matrix A (3×2):\n{A}")
    
    # Compute SVD
    U, s, Vt = svd(A, full_matrices=True)
    
    print(f"\nFull SVD:")
    print(f"U (3×3):\n{np.round(U, 4)}")
    print(f"\nSingular values: {np.round(s, 4)}")
    print(f"\nVᵀ (2×2):\n{np.round(Vt, 4)}")
    
    # Reconstruct
    S = np.zeros((3, 2))
    S[:2, :2] = np.diag(s)
    reconstructed = U @ S @ Vt
    
    print(f"\nReconstruction U @ Σ @ Vᵀ:\n{np.round(reconstructed, 4)}")
    print(f"Matches A: {np.allclose(A, reconstructed)}")


def example_thin_svd():
    """Demonstrate thin (economy) SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Thin (Economy) SVD")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    
    print(f"Matrix A (3×2):\n{A}")
    
    # Full SVD
    U_full, s, Vt = svd(A, full_matrices=True)
    
    # Thin SVD
    U_thin, s_thin, Vt_thin = svd(A, full_matrices=False)
    
    print(f"\nFull SVD: U is {U_full.shape}")
    print(f"Thin SVD: U is {U_thin.shape}")
    
    print(f"\nThin U (3×2):\n{np.round(U_thin, 4)}")
    print(f"\nThin reconstruction:")
    S_thin = np.diag(s_thin)
    reconstructed = U_thin @ S_thin @ Vt_thin
    print(f"{np.round(reconstructed, 4)}")
    print(f"Matches A: {np.allclose(A, reconstructed)}")


def example_svd_from_eigendecomposition():
    """Show relationship between SVD and eigendecomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: SVD from Eigendecomposition")
    print("=" * 60)
    
    A = np.array([[3, 1],
                  [1, 3]])
    
    print(f"Matrix A:\n{A}")
    
    # Method 1: Direct SVD
    U, s, Vt = svd(A)
    print(f"\nDirect SVD:")
    print(f"Singular values: {s}")
    
    # Method 2: From A^T A
    AtA = A.T @ A
    print(f"\nA^T A:\n{AtA}")
    
    eigenvalues, V = np.linalg.eig(AtA)
    # Sort by eigenvalue descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]
    
    print(f"Eigenvalues of A^T A: {eigenvalues}")
    print(f"√eigenvalues (singular values): {np.sqrt(eigenvalues)}")
    
    # Compute U from Av = σu
    singular_values = np.sqrt(eigenvalues)
    U_computed = np.zeros_like(A, dtype=float)
    for i in range(len(singular_values)):
        U_computed[:, i] = A @ V[:, i] / singular_values[i]
    
    print(f"\nComputed U:\n{np.round(U_computed, 4)}")
    print(f"Direct SVD U:\n{np.round(U, 4)}")


def example_low_rank_approximation():
    """Demonstrate low-rank approximation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Low-Rank Approximation")
    print("=" * 60)
    
    # Create a 4×4 matrix
    A = np.array([[1, 2, 3, 4],
                  [2, 4, 6, 8],
                  [1, 1, 1, 1],
                  [3, 3, 3, 3]])
    
    print(f"Original matrix A:\n{A}")
    print(f"Rank of A: {matrix_rank(A)}")
    
    U, s, Vt = svd(A)
    print(f"\nSingular values: {np.round(s, 4)}")
    
    # Rank-1 approximation
    A1 = s[0] * np.outer(U[:, 0], Vt[0, :])
    print(f"\nRank-1 approximation A₁:\n{np.round(A1, 2)}")
    print(f"Error ||A - A₁||_F: {norm(A - A1, 'fro'):.4f}")
    
    # Rank-2 approximation
    A2 = s[0] * np.outer(U[:, 0], Vt[0, :]) + s[1] * np.outer(U[:, 1], Vt[1, :])
    print(f"\nRank-2 approximation A₂:\n{np.round(A2, 2)}")
    print(f"Error ||A - A₂||_F: {norm(A - A2, 'fro'):.4f}")
    
    # Theoretical error bound
    print(f"\nTheoretical error bound √(σ₃² + σ₄²) = {np.sqrt(s[2]**2 + s[3]**2):.4f}")


def example_outer_product_form():
    """Demonstrate SVD as sum of outer products."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Outer Product Form")
    print("=" * 60)
    
    A = np.array([[4, 0],
                  [3, -5]])
    
    print(f"Matrix A:\n{A}")
    
    U, s, Vt = svd(A)
    
    print(f"\nSingular values: σ₁={s[0]:.4f}, σ₂={s[1]:.4f}")
    print(f"U:\n{np.round(U, 4)}")
    print(f"Vᵀ:\n{np.round(Vt, 4)}")
    
    # Outer product form
    print("\nA = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ")
    
    term1 = s[0] * np.outer(U[:, 0], Vt[0, :])
    term2 = s[1] * np.outer(U[:, 1], Vt[1, :])
    
    print(f"\nσ₁u₁v₁ᵀ = {s[0]:.4f} × \n{np.round(np.outer(U[:, 0], Vt[0, :]), 4)}")
    print(f"= \n{np.round(term1, 4)}")
    
    print(f"\nσ₂u₂v₂ᵀ = {s[1]:.4f} × \n{np.round(np.outer(U[:, 1], Vt[1, :]), 4)}")
    print(f"= \n{np.round(term2, 4)}")
    
    print(f"\nSum:\n{np.round(term1 + term2, 4)}")
    print(f"Matches A: {np.allclose(A, term1 + term2)}")


def example_matrix_norms():
    """Demonstrate matrix norms via SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Matrix Norms via SVD")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4]])
    
    print(f"Matrix A:\n{A}")
    
    U, s, Vt = svd(A)
    print(f"\nSingular values: {s}")
    
    # Spectral norm (2-norm)
    spectral_norm = s[0]
    print(f"\nSpectral norm ||A||₂ = σ₁ = {spectral_norm:.4f}")
    print(f"NumPy norm(A, 2): {norm(A, 2):.4f}")
    
    # Frobenius norm
    frobenius_svd = np.sqrt(np.sum(s**2))
    print(f"\nFrobenius norm ||A||_F = √(Σσᵢ²) = {frobenius_svd:.4f}")
    print(f"NumPy norm(A, 'fro'): {norm(A, 'fro'):.4f}")
    
    # Nuclear norm (trace norm)
    nuclear = np.sum(s)
    print(f"\nNuclear norm ||A||_* = Σσᵢ = {nuclear:.4f}")
    print(f"NumPy nuclear norm: {np.linalg.norm(A, 'nuc'):.4f}")


def example_condition_number():
    """Demonstrate condition number."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Condition Number")
    print("=" * 60)
    
    # Well-conditioned matrix
    A = np.array([[2, 1],
                  [1, 2]])
    
    U, s, Vt = svd(A)
    cond_A = s[0] / s[-1]
    
    print("Well-conditioned matrix:")
    print(f"A:\n{A}")
    print(f"Singular values: {s}")
    print(f"Condition number κ(A) = σ₁/σ_r = {cond_A:.4f}")
    
    # Ill-conditioned matrix
    B = np.array([[1, 1],
                  [1, 1.0001]])
    
    U, s, Vt = svd(B)
    cond_B = s[0] / s[-1]
    
    print("\nIll-conditioned matrix:")
    print(f"B:\n{B}")
    print(f"Singular values: {s}")
    print(f"Condition number κ(B) = {cond_B:.1f}")
    
    # Effect on linear system
    print("\nEffect on solving Ax = b:")
    print("Well-conditioned: small errors in b → small errors in x")
    print(f"Ill-conditioned: errors amplified by up to {cond_B:.1f}×")


def example_pseudoinverse():
    """Demonstrate pseudoinverse via SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Pseudoinverse via SVD")
    print("=" * 60)
    
    # Overdetermined system (more rows than columns)
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    
    print(f"Matrix A (3×2):\n{A}")
    
    U, s, Vt = svd(A)
    
    # Compute pseudoinverse: A⁺ = V Σ⁺ Uᵀ
    S_pinv = np.zeros((2, 3))
    for i in range(len(s)):
        if s[i] > 1e-10:
            S_pinv[i, i] = 1 / s[i]
    
    A_pinv = Vt.T @ S_pinv @ U.T
    
    print(f"\nPseudoinverse A⁺ (via SVD):\n{np.round(A_pinv, 4)}")
    print(f"NumPy pinv:\n{np.round(pinv(A), 4)}")
    
    # Verify properties
    print(f"\nVerification of pseudoinverse properties:")
    print(f"AA⁺A ≈ A: {np.allclose(A @ A_pinv @ A, A)}")
    print(f"A⁺AA⁺ ≈ A⁺: {np.allclose(A_pinv @ A @ A_pinv, A_pinv)}")
    
    # Solving least squares
    b = np.array([1, 2, 3])
    x_lstsq = A_pinv @ b
    print(f"\nLeast squares solution for Ax ≈ b = {b}:")
    print(f"x = A⁺b = {x_lstsq}")
    print(f"Residual ||Ax - b||: {norm(A @ x_lstsq - b):.4f}")


def example_image_compression():
    """Demonstrate image compression using SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Image Compression Simulation")
    print("=" * 60)
    
    # Create a sample "image" matrix
    np.random.seed(42)
    m, n = 50, 50
    
    # Create structured image (not pure noise)
    x = np.linspace(0, 4*np.pi, m)
    y = np.linspace(0, 4*np.pi, n)
    X, Y = np.meshgrid(x, y)
    image = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(m, n)
    
    print(f"Original 'image' size: {m}×{n} = {m*n} values")
    
    # SVD
    U, s, Vt = svd(image)
    
    print(f"\nSingular values (first 10): {np.round(s[:10], 4)}")
    
    # Compression at different ranks
    ranks = [1, 5, 10, 20, 50]
    
    print("\nCompression results:")
    print(f"{'Rank k':>8} {'Storage':>12} {'Compression':>12} {'Error':>12}")
    
    for k in ranks:
        # Reconstruct with rank k
        image_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        # Storage: k columns of U + k values + k rows of Vt
        storage = k * (m + n + 1)
        compression = 100 * (1 - storage / (m * n))
        error = norm(image - image_k, 'fro') / norm(image, 'fro')
        
        print(f"{k:8d} {storage:12d} {compression:11.1f}% {error:12.4f}")


def example_pca_via_svd():
    """Demonstrate PCA using SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: PCA via SVD")
    print("=" * 60)
    
    # Generate correlated 2D data
    np.random.seed(42)
    n = 100
    
    # Create correlated data
    t = np.random.randn(n)
    X = np.column_stack([
        2*t + 0.1*np.random.randn(n),
        t + 0.1*np.random.randn(n)
    ])
    
    print(f"Data shape: {X.shape}")
    
    # Center the data
    X_centered = X - X.mean(axis=0)
    
    # SVD of centered data
    U, s, Vt = svd(X_centered, full_matrices=False)
    
    # Principal components are rows of Vt (or columns of V)
    print(f"\nSingular values: {s}")
    print(f"\nPrincipal components (rows of Vᵀ):\n{Vt}")
    
    # Variance explained
    variance_explained = s**2 / (n - 1)
    total_variance = variance_explained.sum()
    
    print(f"\nVariance explained by each PC:")
    for i, (var, pct) in enumerate(zip(variance_explained, 
                                        100 * variance_explained / total_variance)):
        print(f"  PC{i+1}: {var:.4f} ({pct:.1f}%)")
    
    # Project to 1D
    X_1d = X_centered @ Vt[0, :]  # Project onto first PC
    print(f"\nOriginal data: {X.shape}")
    print(f"Projected to 1D: {X_1d.shape}")


def example_recommender_system():
    """Demonstrate SVD for recommender systems."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Recommender System (Matrix Factorization)")
    print("=" * 60)
    
    # User-Item rating matrix (0 = missing)
    # 4 users, 5 items
    R = np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 0, 1, 0],
        [1, 1, 0, 5, 4],
        [0, 1, 5, 4, 0]
    ], dtype=float)
    
    print("Rating matrix (0 = missing):")
    print(R)
    
    # For simplicity, fill missing with row means (basic approach)
    R_filled = R.copy()
    for i in range(R.shape[0]):
        row_mean = R[i, R[i] > 0].mean()
        R_filled[i, R[i] == 0] = row_mean
    
    print(f"\nFilled matrix:\n{np.round(R_filled, 2)}")
    
    # SVD
    U, s, Vt = svd(R_filled)
    print(f"\nSingular values: {np.round(s, 4)}")
    
    # Low-rank approximation (k=2 latent factors)
    k = 2
    R_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    print(f"\nRank-{k} approximation:")
    print(np.round(R_approx, 2))
    
    # Predictions for missing entries
    print("\nPredicted ratings for originally missing entries:")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if R[i, j] == 0:
                print(f"  User {i+1}, Item {j+1}: {R_approx[i, j]:.2f}")


def example_noise_reduction():
    """Demonstrate noise reduction using SVD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Noise Reduction")
    print("=" * 60)
    
    # Create clean signal (low-rank)
    np.random.seed(42)
    m, n = 20, 20
    
    # Low-rank clean matrix (rank 2)
    u1 = np.sin(np.linspace(0, np.pi, m))
    v1 = np.cos(np.linspace(0, np.pi, n))
    u2 = np.cos(np.linspace(0, 2*np.pi, m))
    v2 = np.sin(np.linspace(0, 2*np.pi, n))
    
    clean = 5 * np.outer(u1, v1) + 3 * np.outer(u2, v2)
    
    # Add noise
    noise = 0.5 * np.random.randn(m, n)
    noisy = clean + noise
    
    print(f"Clean matrix rank: {matrix_rank(clean)}")
    print(f"Noisy matrix rank: {matrix_rank(noisy)}")
    
    # SVD of noisy data
    U, s, Vt = svd(noisy)
    
    print(f"\nSingular values (first 5): {np.round(s[:5], 4)}")
    print("Note: First 2 are much larger (signal), rest is noise")
    
    # Denoise by keeping top k components
    k = 2
    denoised = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    print(f"\nError comparison:")
    print(f"  ||noisy - clean||_F = {norm(noisy - clean, 'fro'):.4f}")
    print(f"  ||denoised - clean||_F = {norm(denoised - clean, 'fro'):.4f}")
    print(f"  Improvement: {100*(1 - norm(denoised-clean,'fro')/norm(noisy-clean,'fro')):.1f}%")


def visualize_svd_geometry():
    """Visualize SVD as rotation-scale-rotation."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: SVD Geometry")
    print("=" * 60)
    
    A = np.array([[3, 1],
                  [1, 3]])
    
    U, s, Vt = svd(A)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Generate unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    # Step 0: Original circle
    ax = axes[0]
    ax.plot(circle[0], circle[1], 'b-', linewidth=2)
    ax.set_title('1. Unit Circle')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    # Step 1: After Vᵀ (rotation)
    after_Vt = Vt @ circle
    ax = axes[1]
    ax.plot(after_Vt[0], after_Vt[1], 'g-', linewidth=2)
    ax.set_title('2. After Vᵀ (rotation)')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    # Step 2: After Σ (scaling)
    S = np.diag(s)
    after_S = S @ after_Vt
    ax = axes[2]
    ax.plot(after_S[0], after_S[1], 'orange', linewidth=2)
    ax.set_title(f'3. After Σ (scale by {s[0]:.2f}, {s[1]:.2f})')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    # Step 3: After U (rotation)
    after_U = U @ after_S
    ax = axes[3]
    ax.plot(after_U[0], after_U[1], 'r-', linewidth=2)
    ax.set_title('4. After U (rotation) = A × circle')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('svd_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: svd_geometry.png")


if __name__ == "__main__":
    example_basic_svd()
    example_thin_svd()
    example_svd_from_eigendecomposition()
    example_low_rank_approximation()
    example_outer_product_form()
    example_matrix_norms()
    example_condition_number()
    example_pseudoinverse()
    example_image_compression()
    example_pca_via_svd()
    example_recommender_system()
    example_noise_reduction()
    
    # Uncomment to generate visualization
    # visualize_svd_geometry()
