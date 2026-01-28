"""
Eigenvalues and Eigenvectors - Examples
=======================================
Practical demonstrations of eigenvalue concepts.
"""

import numpy as np
from numpy.linalg import eig, det, inv, matrix_power
import matplotlib.pyplot as plt


def example_basic_eigenvalues():
    """Demonstrate basic eigenvalue computation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Eigenvalues and Eigenvectors")
    print("=" * 60)
    
    A = np.array([[4, 1],
                  [2, 3]])
    
    print(f"Matrix A:\n{A}")
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eig(A)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"\nEigenvectors (as columns):\n{eigenvectors}")
    
    # Verify Av = λv
    print("\nVerification (Av = λv):")
    for i in range(len(eigenvalues)):
        lam = eigenvalues[i]
        v = eigenvectors[:, i]
        Av = A @ v
        lam_v = lam * v
        print(f"\nλ_{i+1} = {lam:.4f}")
        print(f"v_{i+1} = {v}")
        print(f"Av = {Av}")
        print(f"λv = {lam_v}")
        print(f"Match: {np.allclose(Av, lam_v)}")


def example_trace_determinant():
    """Demonstrate trace and determinant relationships."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Trace and Determinant Relationships")
    print("=" * 60)
    
    A = np.array([[4, 1],
                  [2, 3]])
    
    eigenvalues, _ = eig(A)
    
    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues: {eigenvalues}")
    
    # Trace
    trace_A = np.trace(A)
    sum_eigenvalues = np.sum(eigenvalues)
    print(f"\ntr(A) = {trace_A}")
    print(f"λ₁ + λ₂ = {sum_eigenvalues.real:.4f}")
    print(f"Equal: {np.isclose(trace_A, sum_eigenvalues.real)}")
    
    # Determinant
    det_A = det(A)
    prod_eigenvalues = np.prod(eigenvalues)
    print(f"\ndet(A) = {det_A}")
    print(f"λ₁ × λ₂ = {prod_eigenvalues.real:.4f}")
    print(f"Equal: {np.isclose(det_A, prod_eigenvalues.real)}")


def example_eigenvalue_properties():
    """Demonstrate eigenvalue properties under transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Eigenvalue Properties")
    print("=" * 60)
    
    A = np.array([[3, 1],
                  [0, 2]])
    
    eigenvalues_A, _ = eig(A)
    print(f"Matrix A:\n{A}")
    print(f"Eigenvalues of A: {eigenvalues_A}")
    
    # A^2
    A2 = A @ A
    eigenvalues_A2, _ = eig(A2)
    print(f"\nA²:\n{A2}")
    print(f"Eigenvalues of A²: {eigenvalues_A2}")
    print(f"λ² values: {eigenvalues_A**2}")
    
    # A^(-1)
    A_inv = inv(A)
    eigenvalues_inv, _ = eig(A_inv)
    print(f"\nA⁻¹:\n{A_inv}")
    print(f"Eigenvalues of A⁻¹: {eigenvalues_inv}")
    print(f"1/λ values: {1/eigenvalues_A}")
    
    # A + 2I
    A_shift = A + 2 * np.eye(2)
    eigenvalues_shift, _ = eig(A_shift)
    print(f"\nA + 2I:\n{A_shift}")
    print(f"Eigenvalues of A + 2I: {eigenvalues_shift}")
    print(f"λ + 2 values: {eigenvalues_A + 2}")
    
    # 3A
    A_scaled = 3 * A
    eigenvalues_scaled, _ = eig(A_scaled)
    print(f"\n3A:\n{A_scaled}")
    print(f"Eigenvalues of 3A: {eigenvalues_scaled}")
    print(f"3λ values: {3 * eigenvalues_A}")


def example_symmetric_matrix():
    """Demonstrate properties of symmetric matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Symmetric Matrix Properties")
    print("=" * 60)
    
    # Symmetric matrix
    A = np.array([[4, 2, 0],
                  [2, 5, 3],
                  [0, 3, 6]])
    
    print(f"Symmetric matrix A:\n{A}")
    print(f"A = Aᵀ: {np.allclose(A, A.T)}")
    
    eigenvalues, eigenvectors = eig(A)
    
    print(f"\nEigenvalues: {eigenvalues.real}")
    print("Note: All eigenvalues are real!")
    
    print(f"\nEigenvectors:\n{eigenvectors.real}")
    
    # Check orthogonality
    print("\nOrthogonality check (vᵢᵀvⱼ for i≠j):")
    for i in range(3):
        for j in range(i+1, 3):
            dot = np.dot(eigenvectors[:, i], eigenvectors[:, j])
            print(f"  v{i+1}·v{j+1} = {dot.real:.6f}")
    
    # Verify Q^T Q = I
    Q = eigenvectors.real
    print(f"\nQᵀQ (should be I):\n{np.round(Q.T @ Q, 6)}")


def example_diagonalization():
    """Demonstrate matrix diagonalization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Diagonalization")
    print("=" * 60)
    
    A = np.array([[4, 1],
                  [2, 3]])
    
    eigenvalues, P = eig(A)
    D = np.diag(eigenvalues)
    
    print(f"Matrix A:\n{A}")
    print(f"\nEigenvalues → Diagonal matrix D:\n{D.real}")
    print(f"\nEigenvector matrix P:\n{P.real}")
    
    # Verify A = PDP^(-1)
    P_inv = inv(P)
    reconstructed = P @ D @ P_inv
    
    print(f"\nP⁻¹:\n{P_inv.real}")
    print(f"\nPDP⁻¹:\n{reconstructed.real}")
    print(f"\nA = PDP⁻¹: {np.allclose(A, reconstructed)}")


def example_matrix_powers():
    """Demonstrate computing matrix powers via diagonalization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Matrix Powers via Diagonalization")
    print("=" * 60)
    
    A = np.array([[2, 1],
                  [0, 3]])
    
    print(f"Matrix A:\n{A}")
    
    # Get eigendecomposition
    eigenvalues, P = eig(A)
    D = np.diag(eigenvalues)
    P_inv = inv(P)
    
    # Compute A^5 two ways
    k = 5
    
    # Method 1: Direct computation
    A_k_direct = matrix_power(A, k)
    
    # Method 2: Using diagonalization
    D_k = np.diag(eigenvalues ** k)
    A_k_diag = P @ D_k @ P_inv
    
    print(f"\nA^{k} via direct multiplication:\n{A_k_direct}")
    print(f"\nA^{k} via PD^{k}P⁻¹:\n{A_k_diag.real}")
    print(f"\nResults match: {np.allclose(A_k_direct, A_k_diag)}")
    
    # Show D^k is easy
    print(f"\nD^{k} = diag(λ₁^{k}, λ₂^{k}):")
    print(f"  λ₁^{k} = {eigenvalues[0]**k:.0f}")
    print(f"  λ₂^{k} = {eigenvalues[1]**k:.0f}")


def example_spectral_decomposition():
    """Demonstrate spectral decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Spectral Decomposition")
    print("=" * 60)
    
    # Symmetric matrix
    A = np.array([[3, 1],
                  [1, 3]])
    
    print(f"Symmetric matrix A:\n{A}")
    
    eigenvalues, eigenvectors = eig(A)
    
    # Normalize eigenvectors
    v1 = eigenvectors[:, 0].real
    v2 = eigenvectors[:, 1].real
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    lam1, lam2 = eigenvalues.real
    
    print(f"\nEigenvalues: λ₁ = {lam1}, λ₂ = {lam2}")
    print(f"Eigenvectors: v₁ = {v1}, v₂ = {v2}")
    
    # Spectral decomposition: A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ
    term1 = lam1 * np.outer(v1, v1)
    term2 = lam2 * np.outer(v2, v2)
    
    print(f"\nλ₁v₁v₁ᵀ:\n{term1}")
    print(f"\nλ₂v₂v₂ᵀ:\n{term2}")
    
    reconstructed = term1 + term2
    print(f"\nλ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ:\n{reconstructed}")
    print(f"\nMatches A: {np.allclose(A, reconstructed)}")


def example_power_method():
    """Demonstrate the power method for finding dominant eigenvalue."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Power Method")
    print("=" * 60)
    
    A = np.array([[4, 1],
                  [2, 3]])
    
    print(f"Matrix A:\n{A}")
    
    # True eigenvalues
    true_eigenvalues, true_eigenvectors = eig(A)
    print(f"\nTrue eigenvalues: {true_eigenvalues}")
    print(f"Dominant eigenvalue: {max(abs(true_eigenvalues))}")
    
    # Power method
    np.random.seed(42)
    v = np.random.rand(2)
    v = v / np.linalg.norm(v)
    
    print(f"\nPower Method Iterations:")
    print(f"{'Iter':>4} {'Estimated λ':>15} {'Error':>15}")
    
    for i in range(10):
        w = A @ v
        eigenvalue_estimate = np.dot(w, v) / np.dot(v, v)  # Rayleigh quotient
        v = w / np.linalg.norm(w)
        error = abs(eigenvalue_estimate - max(true_eigenvalues.real))
        print(f"{i+1:4d} {eigenvalue_estimate:15.10f} {error:15.10f}")
    
    print(f"\nFinal eigenvector estimate: {v}")
    print(f"True dominant eigenvector: {true_eigenvectors[:, 0].real / np.linalg.norm(true_eigenvectors[:, 0])}")


def example_covariance_eigenanalysis():
    """Demonstrate eigenanalysis of covariance matrix (PCA preview)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Covariance Matrix Eigenanalysis (PCA)")
    print("=" * 60)
    
    # Generate correlated 2D data
    np.random.seed(42)
    n = 100
    
    # Create correlation
    mean = [0, 0]
    cov = [[3, 1.5],
           [1.5, 1]]
    
    data = np.random.multivariate_normal(mean, cov, n)
    
    print(f"Generated {n} data points with covariance:\n{np.array(cov)}")
    
    # Compute sample covariance
    X = data - data.mean(axis=0)  # Center the data
    C = (X.T @ X) / (n - 1)
    
    print(f"\nSample covariance matrix:\n{np.round(C, 4)}")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eig(C)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues (variances along principal axes):")
    for i, lam in enumerate(eigenvalues):
        print(f"  PC{i+1}: λ = {lam.real:.4f}")
    
    print(f"\nPrincipal components (eigenvectors):")
    for i in range(2):
        print(f"  PC{i+1}: {eigenvectors[:, i].real}")
    
    # Variance explained
    total_var = np.sum(eigenvalues)
    print(f"\nVariance explained:")
    for i, lam in enumerate(eigenvalues):
        pct = 100 * lam.real / total_var.real
        print(f"  PC{i+1}: {pct:.1f}%")


def example_markov_chain():
    """Demonstrate eigenvalues in Markov chains."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Markov Chain Stationary Distribution")
    print("=" * 60)
    
    # Transition matrix (rows sum to 1)
    # State 1: Sunny, State 2: Rainy
    P = np.array([[0.8, 0.2],   # From Sunny
                  [0.4, 0.6]])  # From Rainy
    
    print("Weather Markov Chain")
    print(f"Transition matrix P:\n{P}")
    print("  P[i,j] = P(tomorrow=j | today=i)")
    
    # Find eigenvalues
    eigenvalues, eigenvectors = eig(P.T)  # Note: using P^T
    
    print(f"\nEigenvalues of Pᵀ: {eigenvalues}")
    
    # Find eigenvector for λ = 1
    idx = np.argmin(np.abs(eigenvalues - 1))
    stationary = eigenvectors[:, idx].real
    stationary = stationary / np.sum(stationary)  # Normalize to probability
    
    print(f"\nStationary distribution π (eigenvector for λ=1):")
    print(f"  π = {stationary}")
    print(f"  P(Sunny) = {stationary[0]:.4f}")
    print(f"  P(Rainy) = {stationary[1]:.4f}")
    
    # Verify: π P = π
    pi_P = stationary @ P
    print(f"\nVerification: πP = {pi_P}")
    print(f"πP = π: {np.allclose(pi_P, stationary)}")
    
    # Long-run behavior
    P_100 = matrix_power(P, 100)
    print(f"\nP¹⁰⁰ (converges to stationary):\n{np.round(P_100, 4)}")


def example_complex_eigenvalues():
    """Demonstrate complex eigenvalues (rotation matrices)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Complex Eigenvalues (Rotation)")
    print("=" * 60)
    
    # Rotation matrix by 90 degrees
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    print(f"Rotation matrix (90°):\n{np.round(R, 4)}")
    
    eigenvalues, eigenvectors = eig(R)
    
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"  λ₁ = {eigenvalues[0]:.4f} = e^(iπ/2) = i")
    print(f"  λ₂ = {eigenvalues[1]:.4f} = e^(-iπ/2) = -i")
    
    print(f"\n|λ₁| = {abs(eigenvalues[0]):.4f}")
    print(f"|λ₂| = {abs(eigenvalues[1]):.4f}")
    print("Note: |λ| = 1 for rotation matrices!")
    
    # General rotation
    print("\n\nFor general rotation by angle θ:")
    print("Eigenvalues are e^(±iθ) = cos(θ) ± i·sin(θ)")
    print("This explains why rotations don't preserve any direction in 2D")
    print("(except at θ = 0 or π)")


def example_positive_definite():
    """Demonstrate positive definite matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Positive Definite Matrices")
    print("=" * 60)
    
    # Positive definite matrix
    A = np.array([[4, 2],
                  [2, 3]])
    
    print(f"Matrix A:\n{A}")
    
    eigenvalues, _ = eig(A)
    print(f"\nEigenvalues: {eigenvalues.real}")
    print(f"All positive: {all(eigenvalues.real > 0)}")
    
    # Test x^T A x > 0
    print("\nQuadratic form xᵀAx for various x:")
    test_vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([-1, 2]),
        np.array([0.5, -0.5])
    ]
    
    for x in test_vectors:
        quad_form = x.T @ A @ x
        print(f"  x = {x}, xᵀAx = {quad_form:.4f} > 0: {quad_form > 0}")
    
    # Cholesky decomposition exists for positive definite
    L = np.linalg.cholesky(A)
    print(f"\nCholesky decomposition exists (A = LLᵀ):")
    print(f"L:\n{L}")
    print(f"LLᵀ:\n{L @ L.T}")


def visualize_eigenvectors():
    """Visualize eigenvectors and their transformation."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Eigenvector Transformation")
    print("=" * 60)
    
    A = np.array([[2, 1],
                  [1, 2]])
    
    eigenvalues, eigenvectors = eig(A)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Original vectors and transformed
    ax = axes[0]
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    transformed = A @ circle
    
    ax.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit circle')
    ax.plot(transformed[0], transformed[1], 'r-', alpha=0.3, label='Transformed')
    
    # Eigenvectors
    colors = ['green', 'purple']
    for i in range(2):
        v = eigenvectors[:, i].real
        v = v / np.linalg.norm(v)
        Av = A @ v
        
        ax.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.05, 
                fc=colors[i], ec=colors[i], linewidth=2)
        ax.arrow(0, 0, Av[0], Av[1], head_width=0.1, head_length=0.05,
                fc=colors[i], ec=colors[i], linewidth=2, linestyle='--', alpha=0.5)
        ax.annotate(f'v{i+1}', v + 0.1, fontsize=12, color=colors[i])
        ax.annotate(f'Av{i+1}=λ{i+1}v{i+1}', Av + 0.1, fontsize=10, color=colors[i], alpha=0.7)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title(f'Eigenvectors under transformation\nλ₁={eigenvalues[0].real:.2f}, λ₂={eigenvalues[1].real:.2f}')
    ax.legend()
    
    # Plot 2: Eigenvalue spectrum
    ax = axes[1]
    ax.bar(range(1, len(eigenvalues)+1), eigenvalues.real, color=['green', 'purple'])
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue Spectrum')
    ax.set_xticks([1, 2])
    
    plt.tight_layout()
    plt.savefig('eigenvectors_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: eigenvectors_visualization.png")


if __name__ == "__main__":
    example_basic_eigenvalues()
    example_trace_determinant()
    example_eigenvalue_properties()
    example_symmetric_matrix()
    example_diagonalization()
    example_matrix_powers()
    example_spectral_decomposition()
    example_power_method()
    example_covariance_eigenanalysis()
    example_markov_chain()
    example_complex_eigenvalues()
    example_positive_definite()
    
    # Uncomment to generate visualization
    # visualize_eigenvectors()
