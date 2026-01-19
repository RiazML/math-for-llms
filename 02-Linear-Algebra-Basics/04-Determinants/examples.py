"""
Determinants - Examples
=======================
Practical demonstrations of determinant computations and applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def example_basic_determinants():
    """Compute determinants of small matrices."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Determinant Computation")
    print("=" * 60)
    
    # 2×2 matrix
    A = np.array([[3, 2],
                  [1, 4]])
    det_A = np.linalg.det(A)
    det_A_manual = 3*4 - 2*1
    
    print(f"2×2 Matrix A:\n{A}")
    print(f"det(A) = ad - bc = {det_A_manual}")
    print(f"np.linalg.det(A) = {det_A:.4f}")
    
    # 3×3 matrix
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])
    det_B = np.linalg.det(B)
    
    print(f"\n3×3 Matrix B:\n{B}")
    print(f"det(B) = {det_B:.4f}")
    
    # Singular matrix (det = 0)
    C = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])  # Row 3 = Row 1 + Row 2
    det_C = np.linalg.det(C)
    
    print(f"\nSingular Matrix C:\n{C}")
    print(f"det(C) = {det_C:.4f} ≈ 0 (singular)")
    print(f"Note: Row 3 = Row 1 + Row 2, so columns are dependent")


def example_determinant_properties():
    """Demonstrate determinant properties."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Determinant Properties")
    print("=" * 60)
    
    np.random.seed(42)
    A = np.random.randn(3, 3)
    B = np.random.randn(3, 3)
    
    det_A = np.linalg.det(A)
    det_B = np.linalg.det(B)
    
    print(f"det(A) = {det_A:.4f}")
    print(f"det(B) = {det_B:.4f}")
    
    # Property 1: det(AB) = det(A) * det(B)
    det_AB = np.linalg.det(A @ B)
    print(f"\n1. Product rule:")
    print(f"   det(AB) = {det_AB:.4f}")
    print(f"   det(A) × det(B) = {det_A * det_B:.4f}")
    print(f"   Equal: {np.isclose(det_AB, det_A * det_B)}")
    
    # Property 2: det(A^T) = det(A)
    det_AT = np.linalg.det(A.T)
    print(f"\n2. Transpose:")
    print(f"   det(Aᵀ) = {det_AT:.4f}")
    print(f"   det(A) = {det_A:.4f}")
    print(f"   Equal: {np.isclose(det_AT, det_A)}")
    
    # Property 3: det(A^-1) = 1/det(A)
    if np.abs(det_A) > 1e-10:
        A_inv = np.linalg.inv(A)
        det_A_inv = np.linalg.det(A_inv)
        print(f"\n3. Inverse:")
        print(f"   det(A⁻¹) = {det_A_inv:.4f}")
        print(f"   1/det(A) = {1/det_A:.4f}")
        print(f"   Equal: {np.isclose(det_A_inv, 1/det_A)}")
    
    # Property 4: det(cA) = c^n * det(A)
    c = 2
    n = A.shape[0]
    det_cA = np.linalg.det(c * A)
    print(f"\n4. Scalar multiplication (c={c}, n={n}):")
    print(f"   det(cA) = {det_cA:.4f}")
    print(f"   c^n × det(A) = {(c**n) * det_A:.4f}")
    print(f"   Equal: {np.isclose(det_cA, (c**n) * det_A)}")


def example_row_operations():
    """Show effect of row operations on determinant."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Row Operations and Determinants")
    print("=" * 60)
    
    A = np.array([[2., 1., 3.],
                  [4., 2., 1.],
                  [1., 5., 2.]])
    
    det_A = np.linalg.det(A)
    print(f"Original matrix A:\n{A}")
    print(f"det(A) = {det_A:.4f}")
    
    # Operation 1: Swap rows
    B = A.copy()
    B[[0, 1]] = B[[1, 0]]  # Swap row 0 and row 1
    det_B = np.linalg.det(B)
    print(f"\nAfter swapping rows 0 and 1:")
    print(f"det(B) = {det_B:.4f} = -det(A)")
    
    # Operation 2: Multiply row by constant
    C = A.copy()
    C[0] = 3 * C[0]  # Multiply row 0 by 3
    det_C = np.linalg.det(C)
    print(f"\nAfter multiplying row 0 by 3:")
    print(f"det(C) = {det_C:.4f} = 3 × det(A)")
    
    # Operation 3: Add multiple of one row to another
    D = A.copy()
    D[1] = D[1] - 2 * D[0]  # R1 - 2*R0
    det_D = np.linalg.det(D)
    print(f"\nAfter R₁ ← R₁ - 2R₀:")
    print(f"det(D) = {det_D:.4f} = det(A) (unchanged)")


def example_special_matrices():
    """Determinants of special matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Special Matrix Determinants")
    print("=" * 60)
    
    # Identity
    I = np.eye(4)
    print(f"Identity I (4×4):")
    print(f"det(I) = {np.linalg.det(I):.4f}")
    
    # Diagonal
    D = np.diag([2, 3, 5, 7])
    print(f"\nDiagonal D = diag(2, 3, 5, 7):")
    print(f"det(D) = 2×3×5×7 = {np.linalg.det(D):.4f}")
    
    # Upper triangular
    U = np.array([[1, 2, 3],
                  [0, 4, 5],
                  [0, 0, 6]])
    print(f"\nUpper triangular U:\n{U}")
    print(f"det(U) = 1×4×6 = {np.linalg.det(U):.4f}")
    
    # Orthogonal (rotation)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    print(f"\nRotation matrix R (45°):\n{R}")
    print(f"det(R) = {np.linalg.det(R):.4f} (preserves orientation)")
    
    # Orthogonal (reflection)
    F = np.array([[1, 0],
                  [0, -1]])  # Reflection across x-axis
    print(f"\nReflection matrix F:\n{F}")
    print(f"det(F) = {np.linalg.det(F):.4f} (flips orientation)")


def example_geometric_interpretation():
    """Geometric meaning of determinant."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Geometric Interpretation")
    print("=" * 60)
    
    # Area of parallelogram
    u = np.array([2, 0])
    v = np.array([1, 3])
    
    # Determinant = area of parallelogram
    area_matrix = np.array([u, v]).T
    area = np.abs(np.linalg.det(area_matrix))
    
    print(f"Vectors: u = {u}, v = {v}")
    print(f"Area of parallelogram = |det([u|v])| = {area:.4f}")
    
    # Cross product formula for 2D
    cross_2d = u[0] * v[1] - u[1] * v[0]
    print(f"Cross product (2D) = u₁v₂ - u₂v₁ = {cross_2d}")
    print(f"|Cross product| = {np.abs(cross_2d)} = Area")
    
    # Volume of parallelepiped
    u3 = np.array([1, 0, 0])
    v3 = np.array([0, 2, 0])
    w3 = np.array([0, 0, 3])
    
    volume_matrix = np.array([u3, v3, w3]).T
    volume = np.abs(np.linalg.det(volume_matrix))
    
    print(f"\n3D vectors: u={u3}, v={v3}, w={w3}")
    print(f"Volume of parallelepiped = |det([u|v|w])| = {volume:.4f}")


def example_transformation_scaling():
    """How transformations scale area/volume."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Transformation Scaling")
    print("=" * 60)
    
    # Original unit square has area 1
    print("Unit square: Area = 1")
    
    # Scaling transformation
    T1 = np.array([[2, 0],
                   [0, 3]])
    det_T1 = np.linalg.det(T1)
    print(f"\n1. Scaling [2×, 3×]:")
    print(f"   T = {T1.tolist()}")
    print(f"   det(T) = {det_T1:.4f}")
    print(f"   New area = {np.abs(det_T1):.4f}")
    
    # Shear transformation
    T2 = np.array([[1, 2],
                   [0, 1]])
    det_T2 = np.linalg.det(T2)
    print(f"\n2. Shear:")
    print(f"   T = {T2.tolist()}")
    print(f"   det(T) = {det_T2:.4f}")
    print(f"   New area = {np.abs(det_T2):.4f} (area preserved!)")
    
    # Rotation
    theta = np.pi / 6
    T3 = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])
    det_T3 = np.linalg.det(T3)
    print(f"\n3. Rotation (30°):")
    print(f"   det(T) = {det_T3:.4f}")
    print(f"   New area = {np.abs(det_T3):.4f} (area preserved!)")
    
    # Projection (singular)
    T4 = np.array([[1, 0],
                   [0, 0]])  # Project onto x-axis
    det_T4 = np.linalg.det(T4)
    print(f"\n4. Projection onto x-axis:")
    print(f"   T = {T4.tolist()}")
    print(f"   det(T) = {det_T4:.4f}")
    print(f"   New 'area' = 0 (collapsed to a line)")


def example_cramers_rule():
    """Solve system using Cramer's rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Cramer's Rule")
    print("=" * 60)
    
    # System: 2x + y = 5
    #         x + 3y = 6
    A = np.array([[2, 1],
                  [1, 3]])
    b = np.array([5, 6])
    
    print(f"System: 2x + y = 5")
    print(f"        x + 3y = 6")
    print(f"\nA = {A.tolist()}")
    print(f"b = {b.tolist()}")
    
    det_A = np.linalg.det(A)
    print(f"\ndet(A) = {det_A:.4f}")
    
    # Replace column 1 with b
    A1 = A.copy()
    A1[:, 0] = b
    det_A1 = np.linalg.det(A1)
    x1 = det_A1 / det_A
    
    # Replace column 2 with b
    A2 = A.copy()
    A2[:, 1] = b
    det_A2 = np.linalg.det(A2)
    x2 = det_A2 / det_A
    
    print(f"\nCramer's rule:")
    print(f"A₁ (column 1 replaced):\n{A1}")
    print(f"x = det(A₁)/det(A) = {det_A1:.4f}/{det_A:.4f} = {x1:.4f}")
    
    print(f"\nA₂ (column 2 replaced):\n{A2}")
    print(f"y = det(A₂)/det(A) = {det_A2:.4f}/{det_A:.4f} = {x2:.4f}")
    
    # Verify
    x_solve = np.linalg.solve(A, b)
    print(f"\nVerification (np.linalg.solve): {x_solve}")


def example_covariance_determinant():
    """Determinant as generalized variance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Covariance Matrix Determinant")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Dataset 1: Uncorrelated, high variance
    data1 = np.random.randn(1000, 2) * np.array([3, 2])
    cov1 = np.cov(data1.T)
    det1 = np.linalg.det(cov1)
    
    # Dataset 2: Correlated
    L = np.array([[1, 0], [0.8, 0.6]])
    data2 = np.random.randn(1000, 2) @ L.T
    cov2 = np.cov(data2.T)
    det2 = np.linalg.det(cov2)
    
    print("Dataset 1 (uncorrelated, spread out):")
    print(f"Covariance matrix:\n{cov1}")
    print(f"Determinant (generalized variance): {det1:.4f}")
    
    print("\nDataset 2 (correlated):")
    print(f"Covariance matrix:\n{cov2}")
    print(f"Determinant (generalized variance): {det2:.4f}")
    print("\nLarger determinant = more spread/uncertainty")
    
    # Relationship to eigenvalues
    eigenvalues1 = np.linalg.eigvalsh(cov1)
    eigenvalues2 = np.linalg.eigvalsh(cov2)
    
    print(f"\nEigenvalues of Cov1: {eigenvalues1}")
    print(f"Product: {np.prod(eigenvalues1):.4f} = det(Cov1)")
    
    print(f"\nEigenvalues of Cov2: {eigenvalues2}")
    print(f"Product: {np.prod(eigenvalues2):.4f} = det(Cov2)")


def example_log_determinant():
    """Using log-determinant for numerical stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Log-Determinant for Stability")
    print("=" * 60)
    
    # Large covariance matrix
    n = 100
    np.random.seed(42)
    X = np.random.randn(200, n)
    cov = np.cov(X.T)
    
    # Regular determinant
    det_cov = np.linalg.det(cov)
    print(f"Covariance matrix size: {cov.shape}")
    print(f"Regular det: {det_cov}")
    
    # Log-determinant (numerically stable)
    sign, logdet = np.linalg.slogdet(cov)
    print(f"\nLog-determinant approach:")
    print(f"sign = {sign}")
    print(f"log|det| = {logdet:.4f}")
    print(f"Reconstructed: sign × exp(log|det|) = {sign * np.exp(logdet)}")
    
    # Using Cholesky for positive definite
    try:
        L = np.linalg.cholesky(cov)
        logdet_chol = 2 * np.sum(np.log(np.diag(L)))
        print(f"\nVia Cholesky: log|det| = 2Σlog(L_ii) = {logdet_chol:.4f}")
    except np.linalg.LinAlgError:
        print("\nNote: Cholesky failed (matrix not positive definite)")


def example_jacobian_determinant():
    """Jacobian determinant in change of variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Jacobian Determinant")
    print("=" * 60)
    
    print("Polar to Cartesian transformation:")
    print("x = r cos(θ)")
    print("y = r sin(θ)")
    
    print("\nJacobian matrix:")
    print("J = [∂x/∂r  ∂x/∂θ] = [cos(θ)   -r sin(θ)]")
    print("    [∂y/∂r  ∂y/∂θ]   [sin(θ)    r cos(θ)]")
    
    print("\n|det(J)| = |r cos²(θ) + r sin²(θ)| = |r| = r (for r > 0)")
    
    # Example calculation
    r, theta = 2, np.pi/4
    J = np.array([[np.cos(theta), -r * np.sin(theta)],
                  [np.sin(theta), r * np.cos(theta)]])
    
    print(f"\nAt r={r}, θ=π/4:")
    print(f"Jacobian:\n{J}")
    print(f"|det(J)| = {np.abs(np.linalg.det(J)):.4f}")
    print(f"r = {r:.4f}")
    
    print("\nIn integration: dxdy = |det(J)| dr dθ = r dr dθ")
    print("This is why the area element in polar is r dr dθ!")


def visualize_determinant_geometry():
    """Visualize geometric meaning of determinant."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Determinant Geometry")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Unit square vertices
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    
    transformations = [
        (np.array([[2, 0], [0, 1.5]]), "Scaling: det = 3"),
        (np.array([[1, 0.5], [0, 1]]), "Shear: det = 1"),
        (np.array([[0.7071, -0.7071], [0.7071, 0.7071]]), "Rotation: det = 1"),
        (np.array([[1, 0], [0.5, 0]]), "Projection: det = 0")
    ]
    
    for ax, (T, title) in zip(axes.flat, transformations):
        # Original
        ax.fill(square[0], square[1], alpha=0.3, color='blue', label='Original')
        ax.plot(square[0], square[1], 'b-', linewidth=2)
        
        # Transformed
        transformed = T @ square
        ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2)
        
        det = np.linalg.det(T)
        ax.set_title(f'{title}\nOriginal Area: 1, New Area: |{det:.2f}| = {abs(det):.2f}')
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('determinant_geometry.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: determinant_geometry.png")


if __name__ == "__main__":
    example_basic_determinants()
    example_determinant_properties()
    example_row_operations()
    example_special_matrices()
    example_geometric_interpretation()
    example_transformation_scaling()
    example_cramers_rule()
    example_covariance_determinant()
    example_log_determinant()
    example_jacobian_determinant()
    
    # Uncomment to generate visualization
    # visualize_determinant_geometry()
