"""
Spectral Graph Theory - Examples
================================

Practical implementations of spectral methods for graphs,
connecting mathematical theory to Graph Neural Networks.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


# =============================================================================
# Example 1: Computing Graph Matrices
# =============================================================================

def example_graph_matrices():
    """
    Compute various matrices associated with graphs.
    
    - Adjacency matrix A
    - Degree matrix D
    - Laplacian L = D - A
    - Normalized Laplacians
    """
    print("=" * 60)
    print("Example 1: Graph Matrices")
    print("=" * 60)
    
    def compute_all_matrices(adj_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all standard graph matrices."""
        n = len(adj_matrix)
        A = adj_matrix.astype(float)
        
        # Degree matrix
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        
        # Unnormalized Laplacian
        L = D - A
        
        # Symmetric normalized Laplacian: L_sym = D^(-1/2) L D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        L_sym = D_inv_sqrt @ L @ D_inv_sqrt
        
        # Random walk Laplacian: L_rw = D^(-1) L = I - D^(-1) A
        D_inv = np.diag(1.0 / (degrees + 1e-10))
        L_rw = np.eye(n) - D_inv @ A
        
        # Transition matrix (for random walk)
        P = D_inv @ A
        
        return {
            'A': A,
            'D': D,
            'L': L,
            'L_sym': L_sym,
            'L_rw': L_rw,
            'P': P
        }
    
    # Create simple graph (triangle with extra node)
    #   0 --- 1
    #   |   / |
    #   | /   |
    #   2 --- 3
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    
    matrices = compute_all_matrices(A)
    
    print("Adjacency matrix A:")
    print(matrices['A'].astype(int))
    
    print("\nDegree matrix D:")
    print(matrices['D'].astype(int))
    
    print("\nLaplacian L = D - A:")
    print(matrices['L'].astype(int))
    
    print("\nSymmetric normalized Laplacian L_sym:")
    print(np.round(matrices['L_sym'], 3))
    
    print("\nRandom walk transition matrix P = D^(-1)A:")
    print(np.round(matrices['P'], 3))
    
    return compute_all_matrices


# =============================================================================
# Example 2: Laplacian Eigendecomposition
# =============================================================================

def example_laplacian_spectrum():
    """
    Compute and analyze Laplacian eigenvalues and eigenvectors.
    
    Key properties:
    - All eigenvalues >= 0
    - Smallest eigenvalue = 0
    - Number of zero eigenvalues = connected components
    """
    print("\n" + "=" * 60)
    print("Example 2: Laplacian Spectrum")
    print("=" * 60)
    
    def compute_spectrum(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and eigenvectors of Laplacian."""
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def count_connected_components(L: np.ndarray, tol: float = 1e-10) -> int:
        """Count connected components using eigenvalues."""
        eigenvalues, _ = compute_spectrum(L)
        return np.sum(eigenvalues < tol)
    
    # Example 1: Connected graph
    A_connected = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    D = np.diag(A_connected.sum(axis=1))
    L_connected = D - A_connected
    
    eigenvalues, eigenvectors = compute_spectrum(L_connected)
    
    print("Connected graph Laplacian spectrum:")
    print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"  Connected components: {count_connected_components(L_connected)}")
    print(f"  Algebraic connectivity (λ₂): {eigenvalues[1]:.4f}")
    
    # Example 2: Disconnected graph (two triangles)
    A_disconnected = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    D = np.diag(A_disconnected.sum(axis=1))
    L_disconnected = D - A_disconnected
    
    eigenvalues_disc, _ = compute_spectrum(L_disconnected)
    
    print("\nDisconnected graph (2 triangles):")
    print(f"  Eigenvalues: {np.round(eigenvalues_disc, 4)}")
    print(f"  Connected components: {count_connected_components(L_disconnected)}")
    
    # Fiedler vector (second eigenvector) for partitioning
    print(f"\nFiedler vector (u₂) for connected graph:")
    print(f"  {np.round(eigenvectors[:, 1], 4)}")
    print("  Sign indicates partition: positive vs negative")
    
    return compute_spectrum


# =============================================================================
# Example 3: Graph Fourier Transform
# =============================================================================

def example_graph_fourier():
    """
    Graph Fourier Transform: Decompose signal into graph frequencies.
    
    GFT: x̂ = U^T x
    Inverse: x = U x̂
    """
    print("\n" + "=" * 60)
    print("Example 3: Graph Fourier Transform")
    print("=" * 60)
    
    def graph_fourier_transform(x: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Transform signal to spectral domain."""
        return U.T @ x
    
    def inverse_gft(x_hat: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Transform from spectral to vertex domain."""
        return U @ x_hat
    
    def dirichlet_energy(x: np.ndarray, L: np.ndarray) -> float:
        """
        Measure signal smoothness: x^T L x = Σ (x_i - x_j)²
        
        Low energy = smooth signal
        High energy = varying signal
        """
        return float(x.T @ L @ x)
    
    # Create graph
    A = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ])  # Cycle graph
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Get eigenvectors
    eigenvalues, U = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    print("Cycle graph C₅ spectrum:")
    print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
    
    # Example signals
    # Signal 1: Smooth (constant)
    x_smooth = np.ones(5)
    x_smooth_hat = graph_fourier_transform(x_smooth, U)
    
    print(f"\nSmooth signal (constant):")
    print(f"  Vertex domain: {x_smooth}")
    print(f"  Spectral domain: {np.round(x_smooth_hat, 4)}")
    print(f"  Dirichlet energy: {dirichlet_energy(x_smooth, L):.4f}")
    
    # Signal 2: Varying
    x_varying = np.array([1, -1, 1, -1, 1])
    x_varying_hat = graph_fourier_transform(x_varying, U)
    
    print(f"\nVarying signal (alternating):")
    print(f"  Vertex domain: {x_varying}")
    print(f"  Spectral domain: {np.round(x_varying_hat, 4)}")
    print(f"  Dirichlet energy: {dirichlet_energy(x_varying, L):.4f}")
    
    # Reconstruct
    x_reconstructed = inverse_gft(x_varying_hat, U)
    print(f"  Reconstructed: {np.round(x_reconstructed, 4)}")
    
    return graph_fourier_transform, inverse_gft


# =============================================================================
# Example 4: Spectral Filtering
# =============================================================================

def example_spectral_filtering():
    """
    Apply filters in the spectral domain.
    
    y = g(L) x = U g(Λ) U^T x
    """
    print("\n" + "=" * 60)
    print("Example 4: Spectral Filtering")
    print("=" * 60)
    
    def spectral_filter(x: np.ndarray, L: np.ndarray, 
                       g: callable) -> np.ndarray:
        """
        Apply spectral filter.
        
        Args:
            x: Input signal
            L: Laplacian matrix
            g: Filter function g(λ)
        """
        eigenvalues, U = np.linalg.eigh(L)
        
        # Apply filter in spectral domain
        x_hat = U.T @ x
        g_lambda = np.array([g(lam) for lam in eigenvalues])
        y_hat = g_lambda * x_hat
        
        return U @ y_hat
    
    # Create graph
    n = 10
    # Path graph
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Noisy signal
    np.random.seed(42)
    x_clean = np.sin(np.linspace(0, 2 * np.pi, n))
    x_noisy = x_clean + 0.5 * np.random.randn(n)
    
    print("Path graph with noisy sinusoidal signal")
    print(f"Clean signal: {np.round(x_clean, 3)}")
    print(f"Noisy signal: {np.round(x_noisy, 3)}")
    
    # Low-pass filter (smoothing): g(λ) = exp(-αλ)
    def low_pass(lam, alpha=1.0):
        return np.exp(-alpha * lam)
    
    x_smoothed = spectral_filter(x_noisy, L, lambda lam: low_pass(lam, alpha=0.5))
    
    print(f"\nAfter low-pass filter (α=0.5):")
    print(f"  Filtered: {np.round(x_smoothed, 3)}")
    print(f"  MSE to clean: {np.mean((x_smoothed - x_clean)**2):.4f}")
    print(f"  MSE noisy to clean: {np.mean((x_noisy - x_clean)**2):.4f}")
    
    # High-pass filter: g(λ) = 1 - exp(-αλ)
    def high_pass(lam, alpha=1.0):
        return 1 - np.exp(-alpha * lam)
    
    x_highpass = spectral_filter(x_noisy, L, lambda lam: high_pass(lam, alpha=0.5))
    
    print(f"\nAfter high-pass filter:")
    print(f"  Filtered: {np.round(x_highpass, 3)}")
    
    return spectral_filter


# =============================================================================
# Example 5: Polynomial Filters (No Eigendecomposition)
# =============================================================================

def example_polynomial_filters():
    """
    Polynomial filters avoid expensive eigendecomposition.
    
    g(L) = Σ θ_k L^k
    
    This is K-localized: only depends on K-hop neighborhood.
    """
    print("\n" + "=" * 60)
    print("Example 5: Polynomial Filters")
    print("=" * 60)
    
    def polynomial_filter(x: np.ndarray, L: np.ndarray, 
                         coeffs: List[float]) -> np.ndarray:
        """
        Apply polynomial filter without eigendecomposition.
        
        g(L) = θ_0 I + θ_1 L + θ_2 L² + ...
        
        Efficient: O(K * edges)
        """
        result = np.zeros_like(x)
        L_power = np.eye(len(L))  # L^0 = I
        
        for theta in coeffs:
            result += theta * (L_power @ x)
            L_power = L_power @ L
        
        return result
    
    def chebyshev_filter(x: np.ndarray, L: np.ndarray,
                        coeffs: List[float]) -> np.ndarray:
        """
        Chebyshev polynomial filter.
        
        More numerically stable than monomial basis.
        Uses recurrence: T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)
        """
        n = len(x)
        
        # Scale L to [-1, 1]: L̃ = 2L/λ_max - I
        lambda_max = np.max(np.linalg.eigvalsh(L))
        L_scaled = 2 * L / lambda_max - np.eye(n)
        
        # T_0(L̃)x = x
        T_0 = x.copy()
        result = coeffs[0] * T_0
        
        if len(coeffs) == 1:
            return result
        
        # T_1(L̃)x = L̃x
        T_1 = L_scaled @ x
        result += coeffs[1] * T_1
        
        # Recurrence for higher orders
        T_prev, T_curr = T_0, T_1
        for k in range(2, len(coeffs)):
            T_next = 2 * L_scaled @ T_curr - T_prev
            result += coeffs[k] * T_next
            T_prev, T_curr = T_curr, T_next
        
        return result
    
    # Create graph
    n = 8
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Test signal
    x = np.zeros(n)
    x[0] = 1  # Impulse at node 0
    
    print("Impulse signal on cycle graph C₈:")
    print(f"  Input: {x}")
    
    # Apply polynomial filter
    # Approximating low-pass: g(λ) ≈ 1 - 0.5λ + 0.1λ²
    coeffs = [1.0, -0.5, 0.1]
    y_poly = polynomial_filter(x, L, coeffs)
    
    print(f"\nMonomial filter [1, -0.5, 0.1]:")
    print(f"  Output: {np.round(y_poly, 4)}")
    
    # Chebyshev filter
    cheb_coeffs = [0.5, 0.3, 0.1]
    y_cheb = chebyshev_filter(x, L, cheb_coeffs)
    
    print(f"\nChebyshev filter [0.5, 0.3, 0.1]:")
    print(f"  Output: {np.round(y_cheb, 4)}")
    
    # Localization: K-th order filter only affects K-hop neighborhood
    print(f"\nLocalization property:")
    print(f"  K=2 filter only affects 2-hop neighbors of impulse")
    
    return polynomial_filter, chebyshev_filter


# =============================================================================
# Example 6: Spectral Clustering
# =============================================================================

def example_spectral_clustering():
    """
    Spectral clustering using Laplacian eigenvectors.
    
    1. Compute L (or L_sym)
    2. Find k smallest eigenvectors
    3. Apply k-means to embedded points
    """
    print("\n" + "=" * 60)
    print("Example 6: Spectral Clustering")
    print("=" * 60)
    
    def spectral_clustering(A: np.ndarray, k: int, 
                           normalized: bool = True) -> np.ndarray:
        """
        Spectral clustering.
        
        Args:
            A: Adjacency matrix
            k: Number of clusters
            normalized: Use normalized Laplacian
            
        Returns:
            Cluster labels
        """
        n = len(A)
        D = np.diag(A.sum(axis=1))
        
        if normalized:
            # Symmetric normalized Laplacian
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
            L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            L = D - A
        
        # Get k smallest eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        U = eigenvectors[:, idx[:k]]
        
        # Normalize rows (for normalized cut)
        if normalized:
            row_norms = np.linalg.norm(U, axis=1, keepdims=True) + 1e-10
            U = U / row_norms
        
        # Simple k-means
        labels = simple_kmeans(U, k)
        
        return labels
    
    def simple_kmeans(X: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(X)
        
        # Initialize centroids randomly
        np.random.seed(42)
        centroid_idx = np.random.choice(n, k, replace=False)
        centroids = X[centroid_idx].copy()
        
        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.zeros((n, k))
            for i in range(k):
                distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = X[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels
    
    # Create graph with clear cluster structure
    # Two cliques connected by a few edges
    n1, n2 = 6, 6
    n = n1 + n2
    A = np.zeros((n, n))
    
    # Clique 1
    for i in range(n1):
        for j in range(i + 1, n1):
            A[i, j] = A[j, i] = 1
    
    # Clique 2
    for i in range(n1, n):
        for j in range(i + 1, n):
            A[i, j] = A[j, i] = 1
    
    # Bridge edges
    A[0, n1] = A[n1, 0] = 1
    A[n1 - 1, n - 1] = A[n - 1, n1 - 1] = 1
    
    print("Graph: Two cliques with bridge edges")
    print(f"  Clique 1: nodes 0-{n1-1}")
    print(f"  Clique 2: nodes {n1}-{n-1}")
    
    # Compute Laplacian spectrum
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    
    print(f"\nLaplacian eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"  Note: One near-zero eigenvalue (connected)")
    print(f"  Spectral gap: {eigenvalues[1]:.4f} (small = clear cut)")
    
    # Cluster
    labels = spectral_clustering(A, k=2)
    
    print(f"\nSpectral clustering result:")
    print(f"  Labels: {labels}")
    print(f"  Cluster 0: {np.where(labels == 0)[0].tolist()}")
    print(f"  Cluster 1: {np.where(labels == 1)[0].tolist()}")
    
    return spectral_clustering


# =============================================================================
# Example 7: Fiedler Vector for Graph Partitioning
# =============================================================================

def example_fiedler_vector():
    """
    Fiedler vector: Second eigenvector of Laplacian.
    
    Sign of Fiedler vector gives 2-way partition.
    Optimal for minimizing RatioCut.
    """
    print("\n" + "=" * 60)
    print("Example 7: Fiedler Vector for Partitioning")
    print("=" * 60)
    
    def fiedler_partition(A: np.ndarray) -> Tuple[List[int], List[int]]:
        """Partition graph using Fiedler vector."""
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        
        fiedler_vector = eigenvectors[:, idx[1]]
        
        # Partition by sign
        part_0 = np.where(fiedler_vector <= 0)[0].tolist()
        part_1 = np.where(fiedler_vector > 0)[0].tolist()
        
        return part_0, part_1, fiedler_vector
    
    def compute_cut_size(A: np.ndarray, part_0: List[int], 
                        part_1: List[int]) -> int:
        """Count edges crossing the partition."""
        cut = 0
        for i in part_0:
            for j in part_1:
                cut += A[i, j]
        return int(cut)
    
    # Create barbell graph (two cliques connected by path)
    n = 10
    A = np.zeros((n, n))
    
    # Left clique (0-3)
    for i in range(4):
        for j in range(i + 1, 4):
            A[i, j] = A[j, i] = 1
    
    # Right clique (6-9)
    for i in range(6, 10):
        for j in range(i + 1, 10):
            A[i, j] = A[j, i] = 1
    
    # Path connecting them (3-4-5-6)
    A[3, 4] = A[4, 3] = 1
    A[4, 5] = A[5, 4] = 1
    A[5, 6] = A[6, 5] = 1
    
    print("Barbell graph: two cliques connected by path")
    
    part_0, part_1, fiedler = fiedler_partition(A)
    cut_size = compute_cut_size(A, part_0, part_1)
    
    print(f"\nFiedler vector: {np.round(fiedler, 3)}")
    print(f"\nPartition by sign:")
    print(f"  Part 0: {part_0}")
    print(f"  Part 1: {part_1}")
    print(f"  Cut size: {cut_size}")
    
    # Algebraic connectivity
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    print(f"\nAlgebraic connectivity λ₂: {eigenvalues[1]:.4f}")
    
    return fiedler_partition


# =============================================================================
# Example 8: GCN as Spectral Filter
# =============================================================================

def example_gcn_spectral():
    """
    Show that GCN is equivalent to a first-order spectral filter.
    
    GCN: Y = σ(Ã X W) where Ã = D̃^(-1/2) (A + I) D̃^(-1/2)
    
    Spectral interpretation: g(λ) = 2 - λ (low-pass filter)
    """
    print("\n" + "=" * 60)
    print("Example 8: GCN as Spectral Filter")
    print("=" * 60)
    
    def gcn_propagation(A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        GCN propagation (without learnable weights).
        
        Ã = D̃^(-1/2) (A + I) D̃^(-1/2)
        """
        n = len(A)
        A_tilde = A + np.eye(n)  # Add self-loops
        D_tilde = np.diag(A_tilde.sum(axis=1))
        D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
        
        A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        return A_norm @ X
    
    def spectral_equivalent(A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Equivalent spectral filter: g(λ) = 1 - λ (approximately).
        
        For normalized Laplacian L_sym, GCN uses:
        I - L_sym = D^(-1/2) A D^(-1/2)
        """
        n = len(A)
        D = np.diag(A.sum(axis=1) + 1e-10)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
        
        # Normalized adjacency
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        # L_sym = I - A_norm, so A_norm = I - L_sym
        # This applies filter g(λ) = 1 - λ
        return A_norm @ X
    
    # Create graph
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    # Node features
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
        [0.0, 0.0]
    ])
    
    print("Input features:")
    print(X)
    
    # GCN propagation
    Y_gcn = gcn_propagation(A, X)
    
    print("\nAfter GCN propagation:")
    print(np.round(Y_gcn, 4))
    
    # Analyze as spectral filter
    n = len(A)
    A_tilde = A + np.eye(n)
    D_tilde = np.diag(A_tilde.sum(axis=1))
    D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    
    # Eigenvalues of normalized adjacency
    eigenvalues = np.sort(np.linalg.eigvalsh(A_norm))[::-1]
    
    print(f"\nEigenvalues of Ã (normalized adjacency with self-loops):")
    print(f"  {np.round(eigenvalues, 4)}")
    print("  Range: [0, 2] - this is low-pass filtering")
    
    # Compare with multiple GCN layers (over-smoothing)
    print("\nMultiple GCN layers (over-smoothing demonstration):")
    Y = X.copy()
    for k in range(1, 6):
        Y = gcn_propagation(A, Y)
        print(f"  Layer {k}: {np.round(Y[:, 0], 3)}")
    print("  Notice: Features converge (over-smoothing)")
    
    return gcn_propagation


# =============================================================================
# Example 9: Laplacian Eigenmaps Embedding
# =============================================================================

def example_laplacian_eigenmaps():
    """
    Laplacian Eigenmaps: Embed graph nodes using Laplacian eigenvectors.
    
    Minimize: Σ_ij w_ij ||y_i - y_j||²
    Solution: Use smallest eigenvectors of L
    """
    print("\n" + "=" * 60)
    print("Example 9: Laplacian Eigenmaps")
    print("=" * 60)
    
    def laplacian_eigenmaps(A: np.ndarray, dim: int) -> np.ndarray:
        """
        Compute Laplacian eigenmaps embedding.
        
        Args:
            A: Adjacency matrix (can be weighted)
            dim: Embedding dimension
            
        Returns:
            Embedding matrix (n x dim)
        """
        n = len(A)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Generalized eigenvalue problem: L y = λ D y
        # Or use normalized Laplacian
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        
        # Skip first eigenvector (constant)
        embedding = eigenvectors[:, idx[1:dim + 1]]
        
        return embedding
    
    # Create graph with natural 2D structure
    # Grid-like graph
    n = 9  # 3x3 grid
    A = np.zeros((n, n))
    
    # Connect neighbors
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            if j < 2:  # Right neighbor
                A[idx, idx + 1] = 1
                A[idx + 1, idx] = 1
            if i < 2:  # Bottom neighbor
                A[idx, idx + 3] = 1
                A[idx + 3, idx] = 1
    
    print("3x3 grid graph")
    
    # Compute 2D embedding
    embedding = laplacian_eigenmaps(A, dim=2)
    
    print(f"\n2D Laplacian Eigenmaps embedding:")
    for i in range(n):
        row, col = i // 3, i % 3
        print(f"  Node {i} (grid pos {row},{col}): [{embedding[i, 0]:.3f}, {embedding[i, 1]:.3f}]")
    
    # The embedding should preserve the grid structure!
    print("\nNote: Embedding preserves spatial relationships")
    
    return laplacian_eigenmaps


# =============================================================================
# Example 10: Heat Kernel on Graphs
# =============================================================================

def example_heat_kernel():
    """
    Heat diffusion on graphs.
    
    ∂u/∂t = -L u
    
    Solution: u(t) = exp(-tL) u(0)
    """
    print("\n" + "=" * 60)
    print("Example 10: Heat Kernel on Graphs")
    print("=" * 60)
    
    def heat_kernel(L: np.ndarray, t: float) -> np.ndarray:
        """Compute heat kernel H_t = exp(-tL)."""
        eigenvalues, U = np.linalg.eigh(L)
        return U @ np.diag(np.exp(-t * eigenvalues)) @ U.T
    
    def heat_diffusion(L: np.ndarray, u0: np.ndarray, 
                      t: float) -> np.ndarray:
        """Diffuse signal u0 for time t."""
        H_t = heat_kernel(L, t)
        return H_t @ u0
    
    def heat_kernel_signature(L: np.ndarray, times: List[float]) -> np.ndarray:
        """
        Heat Kernel Signature (HKS): Multi-scale node descriptor.
        
        HKS(i, t) = H_t[i, i]
        """
        n = len(L)
        hks = np.zeros((n, len(times)))
        
        for j, t in enumerate(times):
            H_t = heat_kernel(L, t)
            hks[:, j] = np.diag(H_t)
        
        return hks
    
    # Create graph
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Initial heat at node 0
    u0 = np.zeros(6)
    u0[0] = 1.0
    
    print("Heat diffusion from node 0:")
    print(f"  t=0: {u0}")
    
    for t in [0.5, 1.0, 2.0, 5.0]:
        u_t = heat_diffusion(L, u0, t)
        print(f"  t={t}: {np.round(u_t, 4)}")
    
    print("\nNote: Heat spreads and eventually equilibrates")
    
    # Heat Kernel Signature
    times = [0.1, 0.5, 1.0, 2.0, 5.0]
    hks = heat_kernel_signature(L, times)
    
    print(f"\nHeat Kernel Signature (node descriptors):")
    print(f"  Times: {times}")
    for i in range(6):
        print(f"  Node {i}: {np.round(hks[i], 4)}")
    
    return heat_kernel, heat_diffusion


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    example_graph_matrices()
    example_laplacian_spectrum()
    example_graph_fourier()
    example_spectral_filtering()
    example_polynomial_filters()
    example_spectral_clustering()
    example_fiedler_vector()
    example_gcn_spectral()
    example_laplacian_eigenmaps()
    example_heat_kernel()
    
    print("\n" + "=" * 60)
    print("All Spectral Graph Theory Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
