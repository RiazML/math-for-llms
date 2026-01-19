"""
Matrix Rank - Examples
======================
Practical demonstrations of matrix rank computation and applications.
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def example_basic_rank():
    """Compute rank of various matrices."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Rank Computation")
    print("=" * 60)
    
    # Full rank matrix
    A = np.array([[1, 2],
                  [3, 4]])
    print(f"Matrix A:\n{A}")
    print(f"rank(A) = {np.linalg.matrix_rank(A)}")
    print("Full rank: columns are independent")
    
    # Rank-deficient matrix
    B = np.array([[1, 2],
                  [2, 4]])  # Row 2 = 2 * Row 1
    print(f"\nMatrix B:\n{B}")
    print(f"rank(B) = {np.linalg.matrix_rank(B)}")
    print("Rank-deficient: rows are dependent")
    
    # Rectangular matrix
    C = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print(f"\nMatrix C (2×3):\n{C}")
    print(f"rank(C) = {np.linalg.matrix_rank(C)}")
    print(f"Upper bound: min(2, 3) = 2")
    
    # Zero matrix
    Z = np.zeros((3, 3))
    print(f"\nZero matrix: rank = {np.linalg.matrix_rank(Z)}")
    
    # Identity matrix
    I = np.eye(4)
    print(f"Identity (4×4): rank = {np.linalg.matrix_rank(I)}")


def example_rank_via_svd():
    """Understand rank through singular values."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Rank via SVD")
    print("=" * 60)
    
    # Create a rank-2 matrix (3×4)
    A = np.array([[1, 2, 3, 4],
                  [2, 4, 6, 8],
                  [1, 0, 1, 2]])
    
    print(f"Matrix A (3×4):\n{A}")
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    
    print(f"\nSingular values: {S}")
    print(f"Non-zero singular values: {np.sum(S > 1e-10)}")
    print(f"Rank (np.linalg.matrix_rank): {np.linalg.matrix_rank(A)}")
    
    # Verify first row is 2× second row
    print(f"\nNote: Row 2 = 2 × Row 1")
    print(f"Row 1: {A[0]}")
    print(f"Row 2: {A[1]}")
    print(f"2 × Row 1: {2 * A[0]}")


def example_rank_deficiency():
    """Demonstrate effects of rank deficiency."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Effects of Rank Deficiency")
    print("=" * 60)
    
    # Full rank system: unique solution
    A_full = np.array([[1, 2],
                       [3, 4]])
    b = np.array([5, 11])
    
    print("Full rank system:")
    print(f"A = {A_full.tolist()}")
    print(f"b = {b.tolist()}")
    print(f"rank(A) = {np.linalg.matrix_rank(A_full)}")
    x = np.linalg.solve(A_full, b)
    print(f"Unique solution: x = {x}")
    
    # Rank-deficient system: no unique solution
    A_def = np.array([[1, 2],
                      [2, 4]])
    b1 = np.array([3, 6])   # Consistent
    b2 = np.array([3, 7])   # Inconsistent
    
    print("\nRank-deficient system:")
    print(f"A = {A_def.tolist()}")
    print(f"rank(A) = {np.linalg.matrix_rank(A_def)}")
    
    # Consistent case
    print(f"\nConsistent b = {b1.tolist()}:")
    print(f"rank([A|b]) = {np.linalg.matrix_rank(np.column_stack([A_def, b1]))}")
    x_lstsq = np.linalg.lstsq(A_def, b1, rcond=None)[0]
    print(f"Least squares solution: {x_lstsq}")
    print("(Infinite solutions exist along the null space)")
    
    # Inconsistent case
    print(f"\nInconsistent b = {b2.tolist()}:")
    print(f"rank([A|b]) = {np.linalg.matrix_rank(np.column_stack([A_def, b2]))}")
    print("rank(A) < rank([A|b]) → No exact solution exists")


def example_rank_nullity():
    """Demonstrate the rank-nullity theorem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Rank-Nullity Theorem")
    print("=" * 60)
    
    # Create a 3×5 matrix with rank 2
    A = np.array([[1, 2, 1, 0, 1],
                  [2, 4, 3, 1, 3],
                  [3, 6, 4, 1, 4]])
    
    print(f"Matrix A (3×5):\n{A}")
    
    n = A.shape[1]  # Number of columns
    rank = np.linalg.matrix_rank(A)
    nullity = n - rank
    
    print(f"\nn (columns) = {n}")
    print(f"rank(A) = {rank}")
    print(f"nullity(A) = n - rank = {nullity}")
    print(f"Rank-Nullity: {rank} + {nullity} = {n} ✓")
    
    # Find null space basis
    null_space = linalg.null_space(A)
    print(f"\nNull space dimension: {null_space.shape[1]}")
    print(f"Null space basis:\n{null_space}")
    
    # Verify null space vectors
    for i in range(null_space.shape[1]):
        v = null_space[:, i]
        Av = A @ v
        print(f"A @ null_vector_{i+1} = {Av} (should be ~0)")


def example_row_echelon_rank():
    """Compute rank using row reduction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Rank via Row Echelon Form")
    print("=" * 60)
    
    A = np.array([[1, 2, 3, 4],
                  [2, 4, 7, 9],
                  [3, 6, 10, 13]], dtype=float)
    
    print(f"Original matrix A:\n{A}")
    
    # Manual row reduction (simplified)
    U = A.copy()
    
    # Eliminate column 1
    U[1] = U[1] - 2 * U[0]
    U[2] = U[2] - 3 * U[0]
    print(f"\nAfter eliminating column 1:\n{U}")
    
    # Eliminate column 3 (column 2 is all zeros below pivot)
    U[2] = U[2] - U[1]
    print(f"\nAfter eliminating column 3:\n{U}")
    
    # Count non-zero rows (pivots)
    non_zero_rows = np.sum(~np.allclose(U, 0, axis=1))
    print(f"\nNumber of non-zero rows (pivots): {non_zero_rows}")
    print(f"rank(A) = {np.linalg.matrix_rank(A)}")


def example_feature_redundancy():
    """Demonstrate rank and feature redundancy in ML."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Feature Redundancy in ML")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 100
    
    # Create features with redundancy
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    x3 = 2 * x1 + 3 * x2  # Redundant: linear combination
    x4 = np.random.randn(n_samples)
    
    X = np.column_stack([x1, x2, x3, x4])
    
    print(f"Data matrix X: {X.shape}")
    print(f"rank(X) = {np.linalg.matrix_rank(X)}")
    print(f"Expected rank: 3 (feature x3 = 2*x1 + 3*x2)")
    
    # Check XᵀX
    XtX = X.T @ X
    print(f"\nXᵀX (Gram matrix):")
    print(f"rank(XᵀX) = {np.linalg.matrix_rank(XtX)}")
    print(f"Condition number: {np.linalg.cond(XtX):.2e}")
    
    # Singular values reveal the redundancy
    _, S, _ = np.linalg.svd(X)
    print(f"\nSingular values of X: {S}")
    print("Note: One singular value is much smaller (near-zero)")
    
    # Impact on linear regression
    y = x1 + x2 + x4 + np.random.randn(n_samples) * 0.1
    
    try:
        w = np.linalg.solve(XtX, X.T @ y)
        print(f"\nDirect solution: {w}")
    except np.linalg.LinAlgError:
        print("\nDirect solution failed (singular matrix)")
    
    # Use pseudo-inverse instead
    w_lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"Least squares solution: {w_lstsq}")


def example_low_rank_approximation():
    """Low-rank matrix approximation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Low-Rank Approximation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create a full-rank matrix
    A = np.random.randn(10, 8)
    
    print(f"Original matrix A: {A.shape}")
    print(f"Full rank: {np.linalg.matrix_rank(A)}")
    
    # SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"\nSingular values: {np.round(S, 3)}")
    
    # Rank-k approximations
    for k in [1, 2, 4, 8]:
        # Truncated SVD
        A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        
        error = np.linalg.norm(A - A_k, 'fro')
        relative_error = error / np.linalg.norm(A, 'fro')
        
        print(f"\nRank-{k} approximation:")
        print(f"  Frobenius error: {error:.4f}")
        print(f"  Relative error: {relative_error:.4f}")
        print(f"  Compression: {A.size} → {k * (A.shape[0] + A.shape[1])} elements")


def example_matrix_completion():
    """Matrix completion (collaborative filtering intuition)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Low-Rank Matrix for Recommendations")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simulate user-item ratings (low-rank structure)
    n_users, n_items, k = 20, 15, 3
    
    # True low-rank factors
    U_true = np.random.randn(n_users, k)
    V_true = np.random.randn(n_items, k)
    
    # True rating matrix (low-rank)
    R_true = U_true @ V_true.T
    
    print(f"Rating matrix: {R_true.shape}")
    print(f"True rank: {k}")
    print(f"Verified rank: {np.linalg.matrix_rank(R_true)}")
    
    # Add noise (simulating real ratings)
    R_noisy = R_true + 0.5 * np.random.randn(n_users, n_items)
    print(f"\nNoisy matrix rank: {np.linalg.matrix_rank(R_noisy)}")
    
    # Recover low-rank structure via SVD
    U, S, Vt = np.linalg.svd(R_noisy, full_matrices=False)
    
    print(f"\nSingular values: {np.round(S[:6], 2)}...")
    print("Note: First few singular values are dominant")
    
    # Rank-3 approximation
    R_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    recovery_error = np.linalg.norm(R_approx - R_true, 'fro')
    print(f"\nRecovery error (vs true low-rank): {recovery_error:.4f}")


def example_numerical_rank():
    """Numerical rank and tolerance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Numerical Rank Issues")
    print("=" * 60)
    
    # Theoretically rank-deficient
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=float)
    
    print(f"Matrix A:\n{A}")
    print(f"Theoretical rank: 2 (row 3 = 2*row2 - row1)")
    
    # Check with SVD
    _, S, _ = np.linalg.svd(A)
    print(f"\nSingular values: {S}")
    print(f"Computed rank: {np.linalg.matrix_rank(A)}")
    
    # Add tiny perturbation
    epsilon = 1e-10
    A_perturbed = A + epsilon * np.random.randn(3, 3)
    _, S_perturbed, _ = np.linalg.svd(A_perturbed)
    
    print(f"\nPerturbed singular values: {S_perturbed}")
    print(f"Perturbed rank (default tol): {np.linalg.matrix_rank(A_perturbed)}")
    print(f"Perturbed rank (tol=1e-8): {np.linalg.matrix_rank(A_perturbed, tol=1e-8)}")
    
    # Condition number
    cond = S[0] / S[-1] if S[-1] > 0 else np.inf
    print(f"\nCondition number: {cond:.2e}")


def example_rank_operations():
    """Rank under various operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Rank Under Operations")
    print("=" * 60)
    
    np.random.seed(42)
    
    A = np.random.randn(4, 3)
    B = np.random.randn(3, 5)
    
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    
    print(f"A: {A.shape}, rank = {rank_A}")
    print(f"B: {B.shape}, rank = {rank_B}")
    
    # Product
    AB = A @ B
    rank_AB = np.linalg.matrix_rank(AB)
    print(f"\nAB: {AB.shape}, rank = {rank_AB}")
    print(f"rank(AB) ≤ min(rank(A), rank(B)) = min({rank_A}, {rank_B}) = {min(rank_A, rank_B)} ✓")
    
    # Transpose
    print(f"\nrank(A) = {rank_A}")
    print(f"rank(Aᵀ) = {np.linalg.matrix_rank(A.T)}")
    print("rank(A) = rank(Aᵀ) ✓")
    
    # Gram matrix
    print(f"\nrank(AᵀA) = {np.linalg.matrix_rank(A.T @ A)}")
    print(f"rank(AAᵀ) = {np.linalg.matrix_rank(A @ A.T)}")
    print(f"All equal to rank(A) = {rank_A} ✓")
    
    # Outer product
    u = np.random.randn(4)
    v = np.random.randn(3)
    outer = np.outer(u, v)
    print(f"\nOuter product uvᵀ: {outer.shape}")
    print(f"rank(uvᵀ) = {np.linalg.matrix_rank(outer)}")


def visualize_rank_geometry():
    """Visualize column space dimension."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Rank Geometry")
    print("=" * 60)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Rank 2 (full rank for 2D)
    ax1 = fig.add_subplot(131)
    A1 = np.array([[1, 0.5], [0, 1]])
    
    # Transform unit square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    transformed = A1 @ square
    
    ax1.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax1.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
    ax1.set_xlim(-0.5, 2)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    ax1.set_title(f'Full Rank (rank=2)\nArea preserved')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Rank 1 (collapses to line)
    ax2 = fig.add_subplot(132)
    A2 = np.array([[1, 0.5], [2, 1]])  # Columns are dependent
    
    transformed2 = A2 @ square
    
    ax2.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax2.plot(transformed2[0], transformed2[1], 'r-', linewidth=3, label='Transformed (line)')
    ax2.set_xlim(-0.5, 3)
    ax2.set_ylim(-0.5, 4)
    ax2.set_aspect('equal')
    ax2.set_title(f'Rank 1\nCollapses to line')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Rank 0 (collapses to point)
    ax3 = fig.add_subplot(133)
    A3 = np.zeros((2, 2))
    
    transformed3 = A3 @ square
    
    ax3.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
    ax3.plot(0, 0, 'ro', markersize=15, label='Transformed (point)')
    ax3.set_xlim(-0.5, 2)
    ax3.set_ylim(-0.5, 2)
    ax3.set_aspect('equal')
    ax3.set_title(f'Rank 0\nCollapses to point')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rank_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: rank_geometry.png")


if __name__ == "__main__":
    example_basic_rank()
    example_rank_via_svd()
    example_rank_deficiency()
    example_rank_nullity()
    example_row_echelon_rank()
    example_feature_redundancy()
    example_low_rank_approximation()
    example_matrix_completion()
    example_numerical_rank()
    example_rank_operations()
    
    # Uncomment to generate visualization
    # visualize_rank_geometry()
