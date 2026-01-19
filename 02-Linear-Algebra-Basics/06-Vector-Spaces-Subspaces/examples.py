"""
Vector Spaces and Subspaces - Examples
======================================
Practical demonstrations of vector space concepts.
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def example_vector_space_axioms():
    """Demonstrate vector space axioms in ℝⁿ."""
    print("=" * 60)
    print("EXAMPLE 1: Vector Space Axioms in ℝ³")
    print("=" * 60)
    
    u = np.array([1, 2, 3])
    v = np.array([4, 5, 6])
    w = np.array([7, 8, 9])
    a, b = 2, 3
    
    print("Vectors: u =", u, ", v =", v, ", w =", w)
    print("Scalars: a =", a, ", b =", b)
    
    # Addition axioms
    print("\n--- Addition Axioms ---")
    print(f"1. Closure: u + v = {u + v} ∈ ℝ³ ✓")
    print(f"2. Commutativity: u + v = {u + v}, v + u = {v + u} ✓")
    print(f"3. Associativity: (u+v)+w = {(u+v)+w}, u+(v+w) = {u+(v+w)} ✓")
    
    zero = np.zeros(3)
    print(f"4. Zero vector: u + 0 = {u + zero} = u ✓")
    print(f"5. Additive inverse: u + (-u) = {u + (-u)} = 0 ✓")
    
    # Scalar multiplication axioms
    print("\n--- Scalar Multiplication Axioms ---")
    print(f"6. Closure: a*u = {a*u} ∈ ℝ³ ✓")
    print(f"7. Distributivity (scalar): a(u+v) = {a*(u+v)}, au+av = {a*u + a*v} ✓")
    print(f"8. Distributivity (vector): (a+b)u = {(a+b)*u}, au+bu = {a*u + b*u} ✓")
    print(f"9. Associativity: a(bu) = {a*(b*u)}, (ab)u = {(a*b)*u} ✓")
    print(f"10. Identity: 1*u = {1*u} = u ✓")


def example_subspace_test():
    """Test if sets are subspaces."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Subspace Tests")
    print("=" * 60)
    
    # Test 1: Plane through origin (x + y + z = 0)
    print("\n--- Test 1: W = {(x,y,z) : x + y + z = 0} ---")
    
    def in_W1(v):
        return np.isclose(v[0] + v[1] + v[2], 0)
    
    # 1. Zero vector
    zero = np.array([0, 0, 0])
    print(f"Zero test: 0+0+0 = 0? {in_W1(zero)} ✓")
    
    # 2. Closure under addition
    v1 = np.array([1, 2, -3])  # 1 + 2 - 3 = 0
    v2 = np.array([2, -1, -1])  # 2 - 1 - 1 = 0
    print(f"v1 = {v1}, in W? {in_W1(v1)}")
    print(f"v2 = {v2}, in W? {in_W1(v2)}")
    print(f"v1 + v2 = {v1 + v2}, in W? {in_W1(v1 + v2)} ✓")
    
    # 3. Closure under scalar multiplication
    c = 5
    print(f"c*v1 = {c*v1}, in W? {in_W1(c*v1)} ✓")
    
    print("Conclusion: W is a SUBSPACE ✓")
    
    # Test 2: Line not through origin (y = 2x + 1)
    print("\n--- Test 2: W = {(x,y) : y = 2x + 1} ---")
    
    def in_W2(v):
        return np.isclose(v[1], 2*v[0] + 1)
    
    zero_2d = np.array([0, 0])
    print(f"Zero test: 0 = 2(0) + 1? {in_W2(zero_2d)} ✗")
    print("Conclusion: W is NOT a subspace ✗")
    
    # Test 3: Non-negative vectors
    print("\n--- Test 3: W = {(x,y) : x ≥ 0, y ≥ 0} ---")
    v = np.array([1, 2])
    c = -1
    cv = c * v
    print(f"v = {v} (in W)")
    print(f"-1 * v = {cv} (x = {cv[0]} < 0)")
    print("Not closed under scalar multiplication")
    print("Conclusion: W is NOT a subspace ✗")


def example_span():
    """Demonstrate span of vectors."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Span of Vectors")
    print("=" * 60)
    
    # Two independent vectors span ℝ²
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    
    print("Vectors: v1 =", v1, ", v2 =", v2)
    
    # Express arbitrary vector as linear combination
    target = np.array([3, 5])
    print(f"\nExpress {target} as c1*v1 + c2*v2:")
    print(f"  {target[0]}*{v1} + {target[1]}*{v2} = {target}")
    print("span{v1, v2} = ℝ² (all 2D vectors)")
    
    # Two dependent vectors span only a line
    print("\n--- Dependent vectors ---")
    w1 = np.array([1, 2])
    w2 = np.array([2, 4])  # w2 = 2*w1
    
    print("Vectors: w1 =", w1, ", w2 =", w2)
    print("Note: w2 = 2*w1 (linearly dependent)")
    print("span{w1, w2} = line through origin with direction (1, 2)")
    
    # Check if (3, 5) is in span
    # c1(1,2) + c2(2,4) = (c1+2c2, 2c1+4c2) = (3, 5)
    # c1 + 2c2 = 3 and 2c1 + 4c2 = 5
    # 2(c1 + 2c2) = 6 but 2c1 + 4c2 = 5: contradiction!
    print(f"\nIs (3, 5) in span{{w1, w2}}? No!")
    print("  Would need: c1 + 2c2 = 3 AND 2c1 + 4c2 = 5")
    print("  But 2(c1 + 2c2) = 6 ≠ 5 → contradiction")


def example_linear_independence():
    """Test linear independence of vectors."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Linear Independence")
    print("=" * 60)
    
    # Independent vectors
    A = np.array([[1, 2, 3],
                  [0, 1, 2],
                  [0, 0, 1]])
    
    print("Vectors as columns of A:")
    print(A)
    
    rank = np.linalg.matrix_rank(A)
    n_vectors = A.shape[1]
    
    print(f"\nNumber of vectors: {n_vectors}")
    print(f"Rank of A: {rank}")
    print(f"Independent? {rank == n_vectors} ✓")
    
    # Dependent vectors
    B = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    print("\n--- Dependent vectors ---")
    print("Vectors as columns of B:")
    print(B)
    
    rank_B = np.linalg.matrix_rank(B)
    print(f"\nRank of B: {rank_B}")
    print(f"Number of vectors: {B.shape[1]}")
    print(f"Independent? {rank_B == B.shape[1]} (dependent)")
    
    # Find dependency relation
    # Solve Bx = 0
    null_space = linalg.null_space(B)
    if null_space.size > 0:
        c = null_space[:, 0]
        print(f"\nDependency relation: {c[0]:.3f}*v1 + {c[1]:.3f}*v2 + {c[2]:.3f}*v3 = 0")


def example_basis():
    """Demonstrate basis concepts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Basis and Dimension")
    print("=" * 60)
    
    # Standard basis
    print("Standard basis for ℝ³:")
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    print(f"  e1 = {e1}")
    print(f"  e2 = {e2}")
    print(f"  e3 = {e3}")
    
    # Express vector in standard basis
    v = np.array([3, -2, 5])
    print(f"\nVector v = {v}")
    print(f"  = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")
    
    # Non-standard basis
    print("\n--- Non-standard basis ---")
    b1 = np.array([1, 1, 0])
    b2 = np.array([1, 0, 1])
    b3 = np.array([0, 1, 1])
    
    B = np.column_stack([b1, b2, b3])
    print("Basis B:")
    print(f"  b1 = {b1}")
    print(f"  b2 = {b2}")
    print(f"  b3 = {b3}")
    
    # Check it's a valid basis
    print(f"\nRank of [b1 b2 b3]: {np.linalg.matrix_rank(B)}")
    print(f"Determinant: {np.linalg.det(B):.2f} ≠ 0 → basis ✓")
    
    # Express v in new basis
    # v = c1*b1 + c2*b2 + c3*b3
    # [b1 b2 b3][c1 c2 c3]^T = v
    coords = np.linalg.solve(B, v)
    print(f"\nCoordinates of v = {v} in basis B:")
    print(f"  [v]_B = {coords}")
    print(f"  Verification: {coords[0]:.2f}*b1 + {coords[1]:.2f}*b2 + {coords[2]:.2f}*b3")
    print(f"            = {coords[0]*b1 + coords[1]*b2 + coords[2]*b3}")


def example_four_subspaces():
    """Demonstrate the four fundamental subspaces."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Four Fundamental Subspaces")
    print("=" * 60)
    
    A = np.array([[1, 2, 1, 0],
                  [2, 4, 3, 1],
                  [3, 6, 4, 1]], dtype=float)
    
    m, n = A.shape
    print(f"Matrix A ({m}×{n}):")
    print(A)
    
    # Compute rank
    rank = np.linalg.matrix_rank(A)
    print(f"\nrank(A) = {rank}")
    
    # 1. Column Space C(A)
    print("\n--- 1. Column Space C(A) ---")
    print(f"Subspace of ℝ^{m}")
    print(f"dim(C(A)) = rank = {rank}")
    
    # Find basis: pivot columns
    # RREF gives pivot positions
    # For this matrix, columns 1 and 3 are pivot columns
    print("Basis: columns 1 and 3 of original A")
    print(f"  c1 = {A[:, 0]}")
    print(f"  c3 = {A[:, 2]}")
    
    # 2. Row Space C(A^T)
    print("\n--- 2. Row Space C(Aᵀ) ---")
    print(f"Subspace of ℝ^{n}")
    print(f"dim(C(Aᵀ)) = rank = {rank}")
    
    # Get RREF (simplified)
    # Using scipy
    from scipy.linalg import lu
    P, L, U = lu(A)
    print("Basis from row echelon form:")
    # Non-zero rows of reduced matrix
    for i in range(rank):
        print(f"  r{i+1} = {U[i, :]}")
    
    # 3. Null Space N(A)
    print("\n--- 3. Null Space N(A) ---")
    print(f"Subspace of ℝ^{n}")
    nullity = n - rank
    print(f"dim(N(A)) = nullity = {n} - {rank} = {nullity}")
    
    null_basis = linalg.null_space(A)
    print(f"Null space basis ({null_basis.shape[1]} vectors):")
    for i in range(null_basis.shape[1]):
        print(f"  n{i+1} = {null_basis[:, i]}")
    
    # Verify Ax = 0
    if null_basis.size > 0:
        print("Verification A @ n1 =", A @ null_basis[:, 0])
    
    # 4. Left Null Space N(A^T)
    print("\n--- 4. Left Null Space N(Aᵀ) ---")
    print(f"Subspace of ℝ^{m}")
    left_nullity = m - rank
    print(f"dim(N(Aᵀ)) = {m} - {rank} = {left_nullity}")
    
    left_null = linalg.null_space(A.T)
    print(f"Left null space basis ({left_null.shape[1]} vectors):")
    for i in range(left_null.shape[1]):
        print(f"  l{i+1} = {left_null[:, i]}")
    
    # Verify y^T A = 0
    if left_null.size > 0:
        print("Verification l1ᵀ @ A =", left_null[:, 0] @ A)
    
    # Summary
    print("\n--- Summary ---")
    print(f"rank(A) = {rank}")
    print(f"dim(C(A)) = dim(C(Aᵀ)) = {rank}")
    print(f"dim(N(A)) = {nullity}, dim(N(Aᵀ)) = {left_nullity}")
    print(f"Check: {rank} + {nullity} = {rank + nullity} = n = {n} ✓")
    print(f"Check: {rank} + {left_nullity} = {rank + left_nullity} = m = {m} ✓")


def example_orthogonal_complements():
    """Demonstrate orthogonality of fundamental subspaces."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Orthogonal Complements")
    print("=" * 60)
    
    A = np.array([[1, 2, 1],
                  [2, 4, 3]])
    
    print(f"Matrix A (2×3):\n{A}")
    
    # Row space basis
    print("\n--- Row Space vs Null Space ---")
    row_space = A  # Rows of A span the row space
    null_space = linalg.null_space(A)
    
    print("Row space is spanned by rows of A")
    print(f"Null space basis:\n{null_space}")
    
    # Check orthogonality
    for i, row in enumerate(A):
        for j in range(null_space.shape[1]):
            dot = np.dot(row, null_space[:, j])
            print(f"Row {i+1} · NullVec {j+1} = {dot:.6f} ≈ 0 ✓")
    
    # Column space vs left null space
    print("\n--- Column Space vs Left Null Space ---")
    left_null = linalg.null_space(A.T)
    
    print("Column space is spanned by columns of A")
    print(f"Left null space basis:\n{left_null}")
    
    # Check orthogonality
    for i, col in enumerate(A.T):
        for j in range(left_null.shape[1]):
            if left_null.shape[1] > 0:
                dot = np.dot(col, left_null[:, j])
                print(f"Col {i+1} · LeftNullVec {j+1} = {dot:.6f} ≈ 0 ✓")


def example_change_of_basis():
    """Demonstrate change of basis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Change of Basis")
    print("=" * 60)
    
    # Standard basis
    E = np.eye(2)
    print("Standard basis E:")
    print(E)
    
    # New basis
    B = np.array([[1, 1],
                  [1, -1]])
    print("\nNew basis B:")
    print(B)
    
    # Vector in standard coordinates
    v_std = np.array([3, 1])
    print(f"\nVector v in standard coords: {v_std}")
    
    # Change of basis: [v]_B = B^(-1) @ v
    B_inv = np.linalg.inv(B)
    v_B = B_inv @ v_std
    print(f"Vector v in B coords: {v_B}")
    
    # Verify
    v_reconstructed = B @ v_B
    print(f"\nVerification: B @ [v]_B = {v_reconstructed}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print(f"  v = {v_B[0]}*(1,1) + {v_B[1]}*(1,-1)")
    print(f"    = {v_B[0]*B[:,0]} + {v_B[1]*B[:,1]}")
    print(f"    = {v_B[0]*B[:,0] + v_B[1]*B[:,1]}")


def example_dimension_examples():
    """Examples of dimensions of various spaces."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Dimension Examples")
    print("=" * 60)
    
    print("1. ℝⁿ: dim = n")
    print("   Standard basis: e₁, e₂, ..., eₙ")
    
    print("\n2. M_{m×n} (all m×n matrices): dim = m×n")
    print("   Basis: matrices with single 1, rest 0s")
    m, n = 2, 3
    print(f"   Example M_{m}×{n}: dim = {m*n}")
    
    print("\n3. Symmetric n×n matrices: dim = n(n+1)/2")
    n = 3
    print(f"   Example Sym_{n}×{n}: dim = {n*(n+1)//2}")
    
    print("\n4. Skew-symmetric n×n matrices: dim = n(n-1)/2")
    print(f"   Example SkewSym_{n}×{n}: dim = {n*(n-1)//2}")
    
    print("\n5. Diagonal n×n matrices: dim = n")
    print(f"   Example Diag_{n}×{n}: dim = {n}")
    
    print("\n6. Polynomials of degree ≤ n: dim = n+1")
    n = 4
    print(f"   Example P_{n}: dim = {n+1}")
    print(f"   Basis: {{1, x, x², ..., x^{n}}}")
    
    print("\n7. Solutions to Ax = 0 where A is m×n, rank r:")
    print(f"   dim = n - r (nullity)")


def example_ml_application():
    """ML application: feature space analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: ML Application - Feature Space Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create data with redundant features
    n_samples = 100
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)
    
    # Feature matrix with redundancy
    X = np.column_stack([
        x1,
        x2,
        x1 + x2,      # Redundant
        2*x1 - x2,    # Adds new direction
        x1 + 2*x2     # Redundant with above
    ])
    
    print(f"Data matrix X: {X.shape}")
    print(f"rank(X) = {np.linalg.matrix_rank(X)}")
    print("Only 3 linearly independent directions despite 5 features")
    
    # SVD analysis
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"\nSingular values: {np.round(S, 2)}")
    
    # Effective dimensionality
    explained = S**2 / np.sum(S**2)
    cumulative = np.cumsum(explained)
    print("\nExplained variance ratio:")
    for i, (exp, cum) in enumerate(zip(explained, cumulative)):
        print(f"  PC{i+1}: {exp:.4f} (cumulative: {cum:.4f})")
    
    # Null space (redundancy structure)
    null = linalg.null_space(X.T)
    print(f"\nNull space dimension: {null.shape[1]}")
    print("These directions have zero variance in the data")


def visualize_subspaces():
    """Visualize subspaces in ℝ³."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Subspaces in ℝ³")
    print("=" * 60)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Line through origin
    ax1 = fig.add_subplot(131, projection='3d')
    t = np.linspace(-2, 2, 100)
    direction = np.array([1, 1, 1]) / np.sqrt(3)
    line = np.outer(t, direction)
    ax1.plot(line[:, 0], line[:, 1], line[:, 2], 'b-', linewidth=2)
    ax1.scatter([0], [0], [0], c='r', s=100, label='Origin')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('1D Subspace (Line)\ndim = 1')
    ax1.legend()
    
    # 2. Plane through origin
    ax2 = fig.add_subplot(132, projection='3d')
    xx, yy = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
    # Plane: x + y - z = 0 → z = x + y
    zz = xx + yy
    ax2.plot_surface(xx, yy, zz, alpha=0.5, color='blue')
    ax2.scatter([0], [0], [0], c='r', s=100, label='Origin')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('2D Subspace (Plane)\nx + y - z = 0, dim = 2')
    
    # 3. Not a subspace (plane not through origin)
    ax3 = fig.add_subplot(133, projection='3d')
    zz_shifted = xx + yy + 1  # z = x + y + 1
    ax3.plot_surface(xx, yy, zz_shifted, alpha=0.5, color='red')
    ax3.scatter([0], [0], [0], c='r', s=100, label='Origin')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('NOT a Subspace\nx + y - z = -1\n(misses origin)')
    
    plt.tight_layout()
    plt.savefig('subspaces_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: subspaces_visualization.png")


if __name__ == "__main__":
    example_vector_space_axioms()
    example_subspace_test()
    example_span()
    example_linear_independence()
    example_basis()
    example_four_subspaces()
    example_orthogonal_complements()
    example_change_of_basis()
    example_dimension_examples()
    example_ml_application()
    
    # Uncomment to generate visualization
    # visualize_subspaces()
