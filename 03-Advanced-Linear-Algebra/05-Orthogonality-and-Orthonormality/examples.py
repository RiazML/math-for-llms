"""
Orthogonality and Orthonormality - Examples
===========================================
Practical demonstrations of orthogonality concepts.
"""

import numpy as np
from numpy.linalg import norm, inv, qr, det
import matplotlib.pyplot as plt


def example_orthogonality_basics():
    """Demonstrate basic orthogonality concepts."""
    print("=" * 60)
    print("EXAMPLE 1: Orthogonality Basics")
    print("=" * 60)
    
    # Check if vectors are orthogonal
    u = np.array([1, 0, 0])
    v = np.array([0, 1, 0])
    w = np.array([1, 1, 0])
    
    print(f"u = {u}")
    print(f"v = {v}")
    print(f"w = {w}")
    
    print(f"\nu · v = {np.dot(u, v)}")
    print(f"u ⊥ v? {np.dot(u, v) == 0}")
    
    print(f"\nu · w = {np.dot(u, w)}")
    print(f"u ⊥ w? {np.dot(u, w) == 0}")
    
    # Pythagorean theorem for orthogonal vectors
    print("\nPythagorean Theorem:")
    print(f"||u||² + ||v||² = {norm(u)**2} + {norm(v)**2} = {norm(u)**2 + norm(v)**2}")
    print(f"||u + v||² = ||{u + v}||² = {norm(u + v)**2}")
    print(f"Equal? {np.isclose(norm(u)**2 + norm(v)**2, norm(u + v)**2)}")


def example_orthonormal_vectors():
    """Demonstrate orthonormal vectors."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Orthonormal Vectors")
    print("=" * 60)
    
    # Standard basis (orthonormal)
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    
    print("Standard basis vectors:")
    print(f"e1 = {e1}")
    print(f"e2 = {e2}")
    print(f"e3 = {e3}")
    
    print("\nOrthonormality check:")
    print(f"e1 · e2 = {np.dot(e1, e2)}")
    print(f"e1 · e3 = {np.dot(e1, e3)}")
    print(f"e2 · e3 = {np.dot(e2, e3)}")
    print(f"||e1|| = {norm(e1)}, ||e2|| = {norm(e2)}, ||e3|| = {norm(e3)}")
    
    # Non-standard orthonormal basis
    print("\nCustom orthonormal basis (45° rotation):")
    u1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    u2 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
    u3 = np.array([0, 0, 1])
    
    print(f"u1 = {np.round(u1, 4)}")
    print(f"u2 = {np.round(u2, 4)}")
    print(f"u3 = {u3}")
    
    print(f"\nu1 · u2 = {np.round(np.dot(u1, u2), 10)}")
    print(f"||u1|| = {norm(u1):.4f}")


def example_orthogonal_matrices():
    """Demonstrate orthogonal matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Orthogonal Matrices")
    print("=" * 60)
    
    # Rotation matrix (orthogonal)
    theta = np.pi / 4  # 45 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    print(f"Rotation matrix (45°):")
    print(f"R = \n{np.round(R, 4)}")
    
    # Verify orthogonality
    print(f"\nR^T @ R = \n{np.round(R.T @ R, 4)}")
    print(f"Is orthogonal: {np.allclose(R.T @ R, np.eye(2))}")
    
    # Property 1: R^-1 = R^T
    print(f"\nR⁻¹ = R^T? {np.allclose(inv(R), R.T)}")
    
    # Property 2: Preserves length
    x = np.array([3, 4])
    Rx = R @ x
    print(f"\nLength preservation:")
    print(f"||x|| = ||{x}|| = {norm(x)}")
    print(f"||Rx|| = ||{np.round(Rx, 4)}|| = {norm(Rx):.4f}")
    
    # Property 3: Determinant = ±1
    print(f"\ndet(R) = {det(R):.4f} (rotation)")
    
    # Reflection matrix
    M = np.array([[1, 0],
                  [0, -1]])
    print(f"\nReflection matrix:")
    print(f"M = \n{M}")
    print(f"det(M) = {det(M):.0f} (reflection)")
    print(f"Is orthogonal: {np.allclose(M.T @ M, np.eye(2))}")


def example_gram_schmidt():
    """Demonstrate Gram-Schmidt orthogonalization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Gram-Schmidt Process")
    print("=" * 60)
    
    # Original vectors (linearly independent but not orthogonal)
    v1 = np.array([1, 1, 0], dtype=float)
    v2 = np.array([1, 0, 1], dtype=float)
    v3 = np.array([0, 1, 1], dtype=float)
    
    print("Original vectors:")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    
    print(f"\nv1 · v2 = {np.dot(v1, v2)} (not orthogonal)")
    
    # Gram-Schmidt process
    print("\n--- Gram-Schmidt Process ---")
    
    # Step 1: Normalize v1
    u1 = v1 / norm(v1)
    print(f"\nStep 1: u1 = v1 / ||v1|| = {np.round(u1, 4)}")
    
    # Step 2: Make v2 orthogonal to u1, then normalize
    w2 = v2 - np.dot(v2, u1) * u1
    u2 = w2 / norm(w2)
    print(f"\nStep 2:")
    print(f"  proj_{'{u1}'}(v2) = {np.round(np.dot(v2, u1) * u1, 4)}")
    print(f"  w2 = v2 - proj = {np.round(w2, 4)}")
    print(f"  u2 = w2 / ||w2|| = {np.round(u2, 4)}")
    
    # Step 3: Make v3 orthogonal to u1 and u2, then normalize
    w3 = v3 - np.dot(v3, u1) * u1 - np.dot(v3, u2) * u2
    u3 = w3 / norm(w3)
    print(f"\nStep 3:")
    print(f"  w3 = v3 - proj_{'{u1}'}(v3) - proj_{'{u2}'}(v3)")
    print(f"  w3 = {np.round(w3, 4)}")
    print(f"  u3 = {np.round(u3, 4)}")
    
    # Verify orthonormality
    print("\nVerification:")
    print(f"u1 · u2 = {np.round(np.dot(u1, u2), 10)}")
    print(f"u1 · u3 = {np.round(np.dot(u1, u3), 10)}")
    print(f"u2 · u3 = {np.round(np.dot(u2, u3), 10)}")
    print(f"||u1|| = {norm(u1):.4f}")
    print(f"||u2|| = {norm(u2):.4f}")
    print(f"||u3|| = {norm(u3):.4f}")


def gram_schmidt(V):
    """
    Gram-Schmidt orthogonalization.
    
    Parameters:
        V: Matrix where columns are vectors to orthogonalize
    
    Returns:
        Q: Matrix with orthonormal columns
    """
    n, k = V.shape
    Q = np.zeros((n, k))
    
    for j in range(k):
        v = V[:, j].copy()
        for i in range(j):
            v -= np.dot(Q[:, i], V[:, j]) * Q[:, i]
        Q[:, j] = v / norm(v)
    
    return Q


def modified_gram_schmidt(V):
    """
    Modified Gram-Schmidt (more numerically stable).
    
    Parameters:
        V: Matrix where columns are vectors to orthogonalize
    
    Returns:
        Q: Matrix with orthonormal columns
    """
    n, k = V.shape
    Q = V.copy().astype(float)
    
    for j in range(k):
        Q[:, j] = Q[:, j] / norm(Q[:, j])
        for i in range(j + 1, k):
            Q[:, i] -= np.dot(Q[:, j], Q[:, i]) * Q[:, j]
    
    return Q


def example_projection():
    """Demonstrate orthogonal projection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Orthogonal Projection")
    print("=" * 60)
    
    # Projection onto a vector
    u = np.array([1, 2])
    v = np.array([3, 1])
    
    print("Projection onto a vector:")
    print(f"u = {u}, v = {v}")
    
    # proj_u(v)
    proj = (np.dot(v, u) / np.dot(u, u)) * u
    print(f"\nproj_u(v) = (v·u / u·u) × u")
    print(f"         = ({np.dot(v, u)} / {np.dot(u, u)}) × {u}")
    print(f"         = {proj}")
    
    # Orthogonal component
    perp = v - proj
    print(f"\nOrthogonal component: v - proj = {perp}")
    print(f"Verify: proj · perp = {np.dot(proj, perp):.10f}")
    
    # Projection matrix
    print("\nProjection matrix P = uu^T / (u^T u):")
    P = np.outer(u, u) / np.dot(u, u)
    print(f"P = \n{P}")
    print(f"\nP @ v = {P @ v}")
    
    # Properties
    print("\nProjection matrix properties:")
    print(f"P² = P? {np.allclose(P @ P, P)}")
    print(f"P^T = P? {np.allclose(P.T, P)}")


def example_projection_subspace():
    """Demonstrate projection onto a subspace."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Projection onto a Subspace")
    print("=" * 60)
    
    # Subspace spanned by columns of A
    A = np.array([[1, 0],
                  [1, 1],
                  [0, 1]], dtype=float)
    
    print(f"Subspace basis (columns of A):\n{A}")
    
    # Vector to project
    b = np.array([1, 2, 3], dtype=float)
    print(f"\nVector b = {b}")
    
    # Projection matrix P = A(A^T A)^-1 A^T
    P = A @ inv(A.T @ A) @ A.T
    print(f"\nProjection matrix P = A(A^TA)⁻¹A^T:")
    print(f"P = \n{np.round(P, 4)}")
    
    # Project b
    proj_b = P @ b
    print(f"\nProjection of b: P @ b = {np.round(proj_b, 4)}")
    
    # Error vector
    error = b - proj_b
    print(f"Error vector: b - Pb = {np.round(error, 4)}")
    
    # Verify error is orthogonal to column space
    print(f"\nError orthogonal to columns of A?")
    print(f"A^T @ error = {np.round(A.T @ error, 10)}")
    
    # Using orthonormal basis (simpler)
    Q, R = qr(A)
    print(f"\n--- Using QR decomposition ---")
    print(f"Q (orthonormal columns):\n{np.round(Q[:, :2], 4)}")
    
    P_qr = Q[:, :2] @ Q[:, :2].T
    print(f"\nP = QQ^T:\n{np.round(P_qr, 4)}")
    print(f"Same as before? {np.allclose(P, P_qr)}")


def example_qr_decomposition():
    """Demonstrate QR decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: QR Decomposition")
    print("=" * 60)
    
    A = np.array([[1, 1],
                  [1, 0],
                  [0, 1]], dtype=float)
    
    print(f"Matrix A:\n{A}")
    
    # Manual QR using Gram-Schmidt
    print("\n--- Manual QR via Gram-Schmidt ---")
    
    a1, a2 = A[:, 0], A[:, 1]
    
    # Orthonormalize
    q1 = a1 / norm(a1)
    w2 = a2 - np.dot(a2, q1) * q1
    q2 = w2 / norm(w2)
    
    Q = np.column_stack([q1, q2])
    print(f"\nQ (orthonormal columns):\n{np.round(Q, 4)}")
    
    # R = Q^T A
    R = Q.T @ A
    print(f"\nR (upper triangular) = Q^T A:\n{np.round(R, 4)}")
    
    # Verify
    print(f"\nVerify Q @ R = A:\n{np.round(Q @ R, 4)}")
    
    # NumPy QR
    print("\n--- NumPy QR ---")
    Q_np, R_np = qr(A)
    print(f"Q:\n{np.round(Q_np[:, :2], 4)}")
    print(f"R:\n{np.round(R_np[:2, :], 4)}")


def example_least_squares():
    """Demonstrate least squares via orthogonality."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Least Squares via QR")
    print("=" * 60)
    
    # Overdetermined system
    A = np.array([[1, 1],
                  [1, 2],
                  [1, 3],
                  [1, 4]], dtype=float)
    b = np.array([2, 3, 4, 5.5], dtype=float)
    
    print("Overdetermined system Ax = b:")
    print(f"A:\n{A}")
    print(f"b: {b}")
    
    # Via normal equations
    print("\n--- Normal equations: A^TAx = A^Tb ---")
    x_normal = inv(A.T @ A) @ A.T @ b
    print(f"x = {np.round(x_normal, 4)}")
    
    # Via QR
    print("\n--- QR method: Rx = Q^Tb ---")
    Q, R = qr(A)
    Q = Q[:, :2]
    R = R[:2, :]
    
    x_qr = inv(R) @ Q.T @ b
    print(f"x = {np.round(x_qr, 4)}")
    
    # Verify
    print(f"\nBoth methods agree: {np.allclose(x_normal, x_qr)}")
    
    # Residual
    residual = b - A @ x_qr
    print(f"\nResidual: b - Ax = {np.round(residual, 4)}")
    print(f"||residual|| = {norm(residual):.4f}")
    
    # Verify residual is orthogonal to column space
    print(f"Residual ⊥ columns of A? A^T(b - Ax) = {np.round(A.T @ residual, 10)}")


def example_orthogonal_neural_init():
    """Demonstrate orthogonal weight initialization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Orthogonal Weight Initialization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Random initialization
    W_random = np.random.randn(4, 4) * 0.1
    
    # Orthogonal initialization (via QR)
    M = np.random.randn(4, 4)
    Q, R = qr(M)
    W_orth = Q  # Orthogonal matrix
    
    print("Random initialization:")
    print(f"W @ W^T:\n{np.round(W_random @ W_random.T, 3)}")
    print(f"Singular values: {np.round(np.linalg.svd(W_random)[1], 3)}")
    
    print("\nOrthogonal initialization:")
    print(f"W @ W^T:\n{np.round(W_orth @ W_orth.T, 3)}")
    print(f"Singular values: {np.round(np.linalg.svd(W_orth)[1], 3)}")
    
    # Signal propagation through layers
    x = np.random.randn(4)
    
    print("\n--- Signal propagation (10 layers) ---")
    
    signal_random = x.copy()
    signal_orth = x.copy()
    
    for i in range(10):
        signal_random = W_random @ signal_random
        signal_orth = W_orth @ signal_orth
    
    print(f"Random: ||x|| = {norm(x):.2f} → ||W^10 x|| = {norm(signal_random):.6f}")
    print(f"Orthogonal: ||x|| = {norm(x):.2f} → ||W^10 x|| = {norm(signal_orth):.2f}")
    print("\nOrthogonal preserves signal magnitude!")


def example_orthogonal_complement():
    """Demonstrate orthogonal complement."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Orthogonal Complement")
    print("=" * 60)
    
    # Subspace W: xy-plane (z = 0)
    print("W = xy-plane (z = 0)")
    
    # Basis for W
    w1 = np.array([1, 0, 0])
    w2 = np.array([0, 1, 0])
    print(f"Basis for W: w1 = {w1}, w2 = {w2}")
    
    # Orthogonal complement W^perp
    print(f"\nW^⊥ = z-axis")
    w_perp = np.array([0, 0, 1])
    print(f"Basis for W^⊥: {w_perp}")
    
    # Verify orthogonality
    print(f"\nw1 · w^⊥ = {np.dot(w1, w_perp)}")
    print(f"w2 · w^⊥ = {np.dot(w2, w_perp)}")
    
    # Decompose a vector
    v = np.array([3, 4, 5])
    print(f"\nDecompose v = {v}:")
    
    v_W = np.array([v[0], v[1], 0])  # In W
    v_perp = np.array([0, 0, v[2]])  # In W^perp
    
    print(f"v_W (in xy-plane) = {v_W}")
    print(f"v_⊥ (along z-axis) = {v_perp}")
    print(f"v = v_W + v_⊥? {np.allclose(v, v_W + v_perp)}")
    print(f"v_W ⊥ v_⊥? {np.dot(v_W, v_perp) == 0}")
    
    # Dimension check
    print(f"\ndim(W) + dim(W^⊥) = 2 + 1 = 3 = dim(ℝ³) ✓")


def example_pca_orthogonality():
    """Demonstrate orthogonality in PCA."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: PCA and Orthogonality")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate correlated data
    n = 100
    X = np.random.randn(n, 2)
    # Add correlation
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * X[:, 1]
    X = X - X.mean(axis=0)  # Center
    
    # Covariance matrix
    C = X.T @ X / (n - 1)
    print("Covariance matrix:")
    print(f"C = \n{np.round(C, 4)}")
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nEigenvalues: {np.round(eigenvalues, 4)}")
    print(f"\nPrincipal components (eigenvectors):")
    print(f"PC1 = {np.round(eigenvectors[:, 0], 4)}")
    print(f"PC2 = {np.round(eigenvectors[:, 1], 4)}")
    
    # Verify orthogonality
    print(f"\nPC1 · PC2 = {np.round(np.dot(eigenvectors[:, 0], eigenvectors[:, 1]), 10)}")
    print("Principal components are orthogonal!")
    
    # Transform data
    X_pca = X @ eigenvectors
    C_pca = np.cov(X_pca.T)
    print(f"\nCovariance after PCA (should be diagonal):")
    print(f"{np.round(C_pca, 4)}")


def visualize_gram_schmidt():
    """Visualize Gram-Schmidt process in 2D."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Gram-Schmidt Process")
    print("=" * 60)
    
    # Original vectors
    v1 = np.array([2, 1])
    v2 = np.array([1, 2])
    
    # Gram-Schmidt
    u1 = v1 / norm(v1)
    proj = np.dot(v2, u1) * u1
    w2 = v2 - proj
    u2 = w2 / norm(w2)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original
    ax = axes[0]
    ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Original (not orthogonal)')
    
    # Step 1
    ax = axes[1]
    ax.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1, color='blue', linewidth=2, label='u1')
    ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5, label='v2')
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='green', label='proj')
    ax.quiver(proj[0], proj[1], w2[0], w2[1], angles='xy', scale_units='xy', scale=1, color='orange', label='w2')
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Step 1: Subtract projection')
    
    # Final
    ax = axes[2]
    ax.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1, color='blue', linewidth=2, label='u1')
    ax.quiver(0, 0, u2[0], u2[1], angles='xy', scale_units='xy', scale=1, color='red', linewidth=2, label='u2')
    ax.set_xlim(-1, 3)
    ax.set_ylim(-1, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Result (orthonormal)')
    
    plt.tight_layout()
    plt.savefig('gram_schmidt_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gram_schmidt_visualization.png")


if __name__ == "__main__":
    example_orthogonality_basics()
    example_orthonormal_vectors()
    example_orthogonal_matrices()
    example_gram_schmidt()
    example_projection()
    example_projection_subspace()
    example_qr_decomposition()
    example_least_squares()
    example_orthogonal_neural_init()
    example_orthogonal_complement()
    example_pca_orthogonality()
    
    # Uncomment to generate visualization
    # visualize_gram_schmidt()
