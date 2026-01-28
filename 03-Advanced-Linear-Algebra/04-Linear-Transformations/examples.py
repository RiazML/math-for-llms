"""
Linear Transformations - Examples
=================================
Practical demonstrations of linear transformation concepts.
"""

import numpy as np
from numpy.linalg import det, inv, eig, norm
import matplotlib.pyplot as plt


def example_verify_linearity():
    """Verify linearity of a transformation."""
    print("=" * 60)
    print("EXAMPLE 1: Verifying Linearity")
    print("=" * 60)
    
    # Define transformation T(x, y) = (2x + y, x - y)
    def T(v):
        x, y = v
        return np.array([2*x + y, x - y])
    
    # Corresponding matrix
    A = np.array([[2, 1],
                  [1, -1]])
    
    print("Transformation: T(x, y) = (2x + y, x - y)")
    print(f"Matrix A:\n{A}")
    
    # Test vectors
    u = np.array([3, 1])
    v = np.array([2, 4])
    a, b = 2, 3
    
    print(f"\nTest vectors: u = {u}, v = {v}")
    print(f"Scalars: a = {a}, b = {b}")
    
    # Check additivity: T(u + v) = T(u) + T(v)
    print("\n1. Additivity: T(u + v) = T(u) + T(v)?")
    T_u_plus_v = T(u + v)
    T_u_plus_T_v = T(u) + T(v)
    print(f"   T(u + v) = T({u + v}) = {T_u_plus_v}")
    print(f"   T(u) + T(v) = {T(u)} + {T(v)} = {T_u_plus_T_v}")
    print(f"   Equal: {np.allclose(T_u_plus_v, T_u_plus_T_v)}")
    
    # Check homogeneity: T(av) = aT(v)
    print("\n2. Homogeneity: T(av) = aT(v)?")
    T_av = T(a * v)
    a_Tv = a * T(v)
    print(f"   T({a}v) = T({a * v}) = {T_av}")
    print(f"   {a}T(v) = {a} × {T(v)} = {a_Tv}")
    print(f"   Equal: {np.allclose(T_av, a_Tv)}")
    
    # Combined check
    print("\n3. Combined: T(au + bv) = aT(u) + bT(v)?")
    left = T(a*u + b*v)
    right = a*T(u) + b*T(v)
    print(f"   T({a}u + {b}v) = {left}")
    print(f"   {a}T(u) + {b}T(v) = {right}")
    print(f"   Equal: {np.allclose(left, right)}")


def example_basic_transformations():
    """Demonstrate basic 2D transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Basic 2D Transformations")
    print("=" * 60)
    
    # Test point
    x = np.array([1, 0])
    print(f"Original point: {x}")
    
    # 1. Scaling
    S = np.array([[2, 0],
                  [0, 3]])
    print(f"\n1. Scaling (2x, 3y):\n{S}")
    print(f"   {x} → {S @ x}")
    
    # 2. Rotation by 90 degrees
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    print(f"\n2. Rotation 90°:\n{np.round(R, 4)}")
    print(f"   {x} → {np.round(R @ x, 4)}")
    
    # 3. Reflection across y-axis
    M = np.array([[-1, 0],
                  [0, 1]])
    print(f"\n3. Reflection (y-axis):\n{M}")
    print(f"   {x} → {M @ x}")
    
    # 4. Shear
    k = 0.5
    H = np.array([[1, k],
                  [0, 1]])
    print(f"\n4. Horizontal shear (k={k}):\n{H}")
    y = np.array([0, 1])
    print(f"   {y} → {H @ y}")
    
    # 5. Projection onto x-axis
    P = np.array([[1, 0],
                  [0, 0]])
    print(f"\n5. Projection onto x-axis:\n{P}")
    z = np.array([3, 4])
    print(f"   {z} → {P @ z}")


def example_composition():
    """Demonstrate composition of transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Composition of Transformations")
    print("=" * 60)
    
    # Scale by 2, then rotate 90 degrees
    S = np.array([[2, 0],
                  [0, 2]])
    
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    print("Transformations:")
    print(f"S (scale by 2):\n{S}")
    print(f"\nR (rotate 90°):\n{np.round(R, 4)}")
    
    # Composition: first S, then R
    RS = R @ S  # R(S(x)) = (RS)x
    print(f"\nComposition R ∘ S (first scale, then rotate):")
    print(f"RS = \n{np.round(RS, 4)}")
    
    # Composition: first R, then S
    SR = S @ R
    print(f"\nComposition S ∘ R (first rotate, then scale):")
    print(f"SR = \n{np.round(SR, 4)}")
    
    # They're the same for this case! (uniform scaling commutes)
    print(f"\nRS = SR (uniform scaling commutes): {np.allclose(RS, SR)}")
    
    # Non-uniform scaling doesn't commute
    S2 = np.array([[2, 0],
                   [0, 1]])
    RS2 = R @ S2
    S2R = S2 @ R
    print(f"\nNon-uniform scaling:")
    print(f"RS2 = \n{np.round(RS2, 4)}")
    print(f"S2R = \n{np.round(S2R, 4)}")
    print(f"RS2 = S2R: {np.allclose(RS2, S2R)}")


def example_kernel_and_image():
    """Demonstrate kernel and image of transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Kernel and Image")
    print("=" * 60)
    
    # Projection onto x-axis
    P = np.array([[1, 0],
                  [0, 0]])
    
    print("Projection onto x-axis:")
    print(f"P = \n{P}")
    
    print("\nKernel (null space):")
    print("  Vectors v where Pv = 0")
    print("  Solution: v = (0, y) for any y")
    print("  Kernel = y-axis (dimension 1)")
    
    # Verify
    v_kernel = np.array([0, 5])
    print(f"  Check: P @ {v_kernel} = {P @ v_kernel}")
    
    print("\nImage (range):")
    print("  All possible outputs Pv")
    print("  Image = x-axis (dimension 1)")
    
    # Verify
    v = np.array([3, 7])
    print(f"  Check: P @ {v} = {P @ v} (always on x-axis)")
    
    print("\nRank-Nullity Theorem:")
    print(f"  dim(kernel) + dim(image) = dim(domain)")
    print(f"  1 + 1 = 2 ✓")
    
    # Another example
    print("\n" + "-" * 40)
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    print(f"\nMatrix A:\n{A}")
    
    # Rank
    rank = np.linalg.matrix_rank(A)
    print(f"Rank (dim of image): {rank}")
    
    # Nullity
    nullity = A.shape[1] - rank
    print(f"Nullity (dim of kernel): {nullity}")
    
    print(f"Rank + Nullity = {rank + nullity} = {A.shape[1]} (# columns) ✓")


def example_invertibility():
    """Demonstrate invertibility of transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Invertibility")
    print("=" * 60)
    
    # Invertible transformation (rotation)
    theta = np.pi / 4  # 45 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    print("Rotation by 45°:")
    print(f"R = \n{np.round(R, 4)}")
    print(f"det(R) = {det(R):.4f} ≠ 0 → Invertible!")
    
    R_inv = inv(R)
    print(f"\nR⁻¹ (rotation by -45°):\n{np.round(R_inv, 4)}")
    print(f"R @ R⁻¹ = \n{np.round(R @ R_inv, 4)}")
    
    # Non-invertible transformation (projection)
    P = np.array([[1, 0],
                  [0, 0]])
    print("\n" + "-" * 40)
    print("\nProjection onto x-axis:")
    print(f"P = \n{P}")
    print(f"det(P) = {det(P):.4f} = 0 → Not invertible!")
    print("Reason: Information lost (y-coordinate discarded)")
    
    # Verify: multiple inputs give same output
    v1 = np.array([3, 0])
    v2 = np.array([3, 5])
    print(f"\nP @ {v1} = {P @ v1}")
    print(f"P @ {v2} = {P @ v2}")
    print("Same output, different inputs → Cannot invert!")


def example_orthogonal_transformation():
    """Demonstrate orthogonal transformations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Orthogonal Transformations")
    print("=" * 60)
    
    # Rotation (orthogonal)
    theta = np.pi / 3  # 60 degrees
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    print("Rotation by 60°:")
    print(f"R = \n{np.round(R, 4)}")
    
    # Property 1: R^T R = I
    print(f"\nR^T @ R = \n{np.round(R.T @ R, 4)}")
    print(f"Is orthogonal (R^T R = I): {np.allclose(R.T @ R, np.eye(2))}")
    
    # Property 2: Preserves length
    v = np.array([3, 4])
    Rv = R @ v
    print(f"\nLength preservation:")
    print(f"  ||v|| = ||{v}|| = {norm(v):.4f}")
    print(f"  ||Rv|| = ||{np.round(Rv, 4)}|| = {norm(Rv):.4f}")
    
    # Property 3: Preserves angles
    u = np.array([1, 0])
    print(f"\nAngle preservation:")
    cos_angle_before = np.dot(u, v) / (norm(u) * norm(v))
    Ru, Rv = R @ u, R @ v
    cos_angle_after = np.dot(Ru, Rv) / (norm(Ru) * norm(Rv))
    print(f"  cos(angle before) = {cos_angle_before:.4f}")
    print(f"  cos(angle after) = {cos_angle_after:.4f}")
    
    # Property 4: Eigenvalues have |λ| = 1
    eigenvalues, _ = eig(R)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"|λ| = {np.abs(eigenvalues)}")


def example_change_of_basis():
    """Demonstrate change of basis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Change of Basis")
    print("=" * 60)
    
    # Transformation in standard basis
    A = np.array([[4, 1],
                  [2, 3]])
    
    print("Transformation in standard basis:")
    print(f"A = \n{A}")
    
    # Find eigenvectors (new basis)
    eigenvalues, P = eig(A)
    eigenvalues = eigenvalues.real
    P = P.real
    
    print(f"\nEigenvectors (new basis P):\n{np.round(P, 4)}")
    print(f"Eigenvalues: {eigenvalues}")
    
    # Transform to eigenbasis
    A_new = inv(P) @ A @ P
    print(f"\nIn eigenvector basis:")
    print(f"A' = P⁻¹AP = \n{np.round(A_new, 4)}")
    print("(Diagonal matrix! Just scaling along each axis)")
    
    # Visualize: transformation of a vector
    v = np.array([1, 1])
    print(f"\nTransforming v = {v}:")
    
    # Standard basis
    Av = A @ v
    print(f"  Standard basis: Av = {Av}")
    
    # Convert to eigenbasis, transform, convert back
    v_eig = inv(P) @ v
    print(f"  In eigenbasis: v' = {np.round(v_eig, 4)}")
    Av_eig = A_new @ v_eig  # Just scaling!
    print(f"  Transformed: A'v' = {np.round(Av_eig, 4)}")
    Av_back = P @ Av_eig
    print(f"  Back to standard: PAv' = {np.round(Av_back, 4)}")


def example_neural_network_layers():
    """Demonstrate linear transformations in neural networks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Neural Network Layers")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Input
    x = np.array([1, 2, 3])
    print(f"Input x (3D): {x}")
    
    # Layer 1: 3 → 4
    W1 = np.random.randn(4, 3) * 0.5
    b1 = np.random.randn(4) * 0.1
    
    # Layer 2: 4 → 2
    W2 = np.random.randn(2, 4) * 0.5
    b2 = np.random.randn(2) * 0.1
    
    print(f"\nLayer 1: W1 ({W1.shape}), b1 ({b1.shape})")
    print(f"Layer 2: W2 ({W2.shape}), b2 ({b2.shape})")
    
    # Forward pass (linear only)
    h1_linear = W1 @ x + b1
    print(f"\nAfter Layer 1 (linear): h1 = W1x + b1 = {np.round(h1_linear, 4)}")
    
    # With ReLU
    h1 = np.maximum(0, h1_linear)
    print(f"After ReLU: h1 = max(0, h1) = {np.round(h1, 4)}")
    
    h2_linear = W2 @ h1 + b2
    print(f"\nAfter Layer 2 (linear): h2 = W2h1 + b2 = {np.round(h2_linear, 4)}")
    
    output = np.maximum(0, h2_linear)
    print(f"After ReLU: output = {np.round(output, 4)}")
    
    # Without nonlinearity: equivalent to single layer!
    print("\n" + "-" * 40)
    print("\nWithout nonlinearities:")
    W_combined = W2 @ W1
    b_combined = W2 @ b1 + b2
    print(f"Combined: W = W2 @ W1 ({W_combined.shape})")
    output_linear = W_combined @ x + b_combined
    print(f"Output = Wx + b = {np.round(output_linear, 4)}")
    print("This is just one linear transformation!")
    print("Nonlinearities are essential for deep learning!")


def example_attention_transformation():
    """Demonstrate linear transformations in attention."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Attention Mechanism Transformations")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Input: 3 tokens, 4-dimensional embeddings
    X = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0]], dtype=float)
    
    print(f"Input X (3 tokens × 4 dims):\n{X}")
    
    # Query, Key, Value transformations (to 3D)
    d_k = 3
    W_Q = np.random.randn(4, d_k) * 0.5
    W_K = np.random.randn(4, d_k) * 0.5
    W_V = np.random.randn(4, d_k) * 0.5
    
    # Apply transformations
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    print(f"\nQuery Q = X @ W_Q:\n{np.round(Q, 3)}")
    print(f"\nKey K = X @ W_K:\n{np.round(K, 3)}")
    print(f"\nValue V = X @ W_V:\n{np.round(V, 3)}")
    
    # Attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    print(f"\nAttention scores (Q @ K^T / √d_k):\n{np.round(scores, 3)}")
    
    # Softmax
    def softmax(x):
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
    
    attention = softmax(scores)
    print(f"\nAttention weights (softmax):\n{np.round(attention, 3)}")
    
    # Output
    output = attention @ V
    print(f"\nOutput (attention @ V):\n{np.round(output, 3)}")


def example_homogeneous_coordinates():
    """Demonstrate homogeneous coordinates for translation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Homogeneous Coordinates")
    print("=" * 60)
    
    # Point in 2D
    x = np.array([2, 3])
    print(f"Original point: {x}")
    
    # Translation (not linear in standard coords)
    t = np.array([5, -1])
    print(f"Translation: t = {t}")
    
    print(f"\nStandard: x + t = {x + t} (not a matrix multiplication!)")
    
    # Homogeneous coordinates
    x_h = np.array([x[0], x[1], 1])  # Add 1 as third coordinate
    
    # Translation matrix
    T = np.array([[1, 0, t[0]],
                  [0, 1, t[1]],
                  [0, 0, 1]])
    
    print(f"\nHomogeneous coordinates:")
    print(f"x_h = {x_h}")
    print(f"\nTranslation matrix T:\n{T}")
    
    x_translated = T @ x_h
    print(f"\nT @ x_h = {x_translated}")
    print(f"Result (first 2 coords): [{x_translated[0]}, {x_translated[1]}]")
    
    # Combine with other transformations
    print("\n" + "-" * 40)
    print("\nComposition: Scale, Rotate, Translate")
    
    # Scale
    s = 2
    S = np.array([[s, 0, 0],
                  [0, s, 0],
                  [0, 0, 1]])
    
    # Rotate
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    
    # Combined: first scale, then rotate, then translate
    # Order: T @ R @ S @ x
    M = T @ R @ S
    print(f"\nCombined matrix M = T @ R @ S:\n{np.round(M, 4)}")
    
    result = M @ x_h
    print(f"\nM @ x_h = {np.round(result, 4)}")


def visualize_transformations():
    """Visualize various 2D transformations."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: 2D Transformations")
    print("=" * 60)
    
    # Unit square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    
    # Transformations
    transformations = {
        'Original': np.eye(2),
        'Scale (2, 0.5)': np.array([[2, 0], [0, 0.5]]),
        'Rotate 45°': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                [np.sin(np.pi/4), np.cos(np.pi/4)]]),
        'Shear': np.array([[1, 0.5], [0, 1]]),
        'Reflect (y-axis)': np.array([[-1, 0], [0, 1]]),
        'Project (x-axis)': np.array([[1, 0], [0, 0]])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for ax, (name, A) in zip(axes, transformations.items()):
        transformed = A @ square
        
        ax.fill(square[0], square[1], alpha=0.3, label='Original')
        ax.plot(square[0], square[1], 'b-', linewidth=2)
        
        ax.fill(transformed[0], transformed[1], alpha=0.3, color='red', label='Transformed')
        ax.plot(transformed[0], transformed[1], 'r-', linewidth=2)
        
        ax.set_xlim(-2, 3)
        ax.set_ylim(-1.5, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(name)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('transformations_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: transformations_visualization.png")


if __name__ == "__main__":
    example_verify_linearity()
    example_basic_transformations()
    example_composition()
    example_kernel_and_image()
    example_invertibility()
    example_orthogonal_transformation()
    example_change_of_basis()
    example_neural_network_layers()
    example_attention_transformation()
    example_homogeneous_coordinates()
    
    # Uncomment to generate visualization
    # visualize_transformations()
