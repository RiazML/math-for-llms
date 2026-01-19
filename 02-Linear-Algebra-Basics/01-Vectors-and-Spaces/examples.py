"""
Vectors and Spaces - Examples
=============================
Python implementations demonstrating vector concepts.

Requirements: numpy, matplotlib
Run: python examples.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


# =============================================================================
# EXAMPLE 1: Basic Vector Operations
# =============================================================================

def example_vector_operations():
    """
    Demonstrate basic vector operations using NumPy.
    
    These operations are the building blocks of all ML algorithms.
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Vector Operations")
    print("=" * 60)
    
    # Creating vectors
    u = np.array([3, 4])
    v = np.array([1, 2])
    
    print(f"\nVectors:")
    print(f"u = {u}")
    print(f"v = {v}")
    
    # Addition
    print(f"\nVector Addition:")
    print(f"u + v = {u + v}")
    
    # Subtraction
    print(f"\nVector Subtraction:")
    print(f"u - v = {u - v}")
    
    # Scalar multiplication
    c = 2.5
    print(f"\nScalar Multiplication (c = {c}):")
    print(f"c * u = {c * u}")
    
    # Dot product
    print(f"\nDot Product:")
    print(f"u · v = {np.dot(u, v)}")
    print(f"Using @ operator: {u @ v}")
    
    # Magnitude (L2 norm)
    print(f"\nMagnitude (L2 Norm):")
    print(f"||u|| = {np.linalg.norm(u)}")
    print(f"||v|| = {np.linalg.norm(v)}")
    
    # Unit vector
    print(f"\nUnit Vectors:")
    u_hat = u / np.linalg.norm(u)
    print(f"û = u / ||u|| = {u_hat}")
    print(f"||û|| = {np.linalg.norm(u_hat)}")  # Should be 1


# =============================================================================
# EXAMPLE 2: Dot Product and Angles
# =============================================================================

def example_dot_product_geometry():
    """
    Demonstrate the geometric interpretation of dot product.
    
    Key insight: u · v = ||u|| ||v|| cos(θ)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Dot Product and Angles")
    print("=" * 60)
    
    # Create vectors at different angles
    u = np.array([1, 0])  # Along x-axis
    
    angles = [0, 45, 90, 135, 180]
    
    print(f"\nu = {u}")
    print(f"\nVector v at different angles from u:")
    print(f"{'Angle':<10} {'v':<20} {'u·v':<10} {'cos(θ)':<10}")
    print("-" * 50)
    
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        v = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        
        dot_product = np.dot(u, v)
        cos_theta = np.cos(angle_rad)
        
        print(f"{angle_deg}°{'':<7} [{v[0]:6.3f}, {v[1]:6.3f}]   {dot_product:8.3f}   {cos_theta:8.3f}")
    
    print("\nObservation:")
    print("- θ = 0°: vectors aligned, dot product = 1 (maximum)")
    print("- θ = 90°: vectors perpendicular, dot product = 0")
    print("- θ = 180°: vectors opposite, dot product = -1 (minimum)")


# =============================================================================
# EXAMPLE 3: Different Norms
# =============================================================================

def example_norms():
    """
    Compare different vector norms used in ML.
    
    L1: Sum of absolute values (encourages sparsity)
    L2: Euclidean length (encourages small weights)
    L∞: Maximum absolute value (bounds all values)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Different Norms")
    print("=" * 60)
    
    v = np.array([3, -4, 5, -2])
    
    print(f"\nVector v = {v}")
    print(f"\nDifferent norms:")
    
    # L1 norm
    l1 = np.linalg.norm(v, ord=1)
    print(f"L1 (Manhattan): ||v||₁ = |3| + |-4| + |5| + |-2| = {l1}")
    
    # L2 norm
    l2 = np.linalg.norm(v, ord=2)
    print(f"L2 (Euclidean): ||v||₂ = √(3² + 4² + 5² + 2²) = {l2:.4f}")
    
    # L∞ norm
    linf = np.linalg.norm(v, ord=np.inf)
    print(f"L∞ (Max):       ||v||∞ = max(|3|, |4|, |5|, |2|) = {linf}")
    
    # Visualize unit balls
    print("\nML Applications:")
    print("- L1: Lasso regression (sparse solutions)")
    print("- L2: Ridge regression (small weights)")
    print("- L∞: Adversarial robustness bounds")


# =============================================================================
# EXAMPLE 4: Similarity Measures
# =============================================================================

def example_similarity():
    """
    Compare different similarity/distance measures.
    
    These are fundamental for:
    - KNN classification
    - Clustering (K-means)
    - Recommendation systems
    - Semantic search
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Similarity Measures")
    print("=" * 60)
    
    # Example: Word vectors (simplified)
    # In practice, these would be 300-dimensional
    king = np.array([0.8, 0.6, 0.2])
    queen = np.array([0.7, 0.7, 0.3])
    apple = np.array([0.1, 0.2, 0.9])
    
    print("\nSimulated word vectors:")
    print(f"king = {king}")
    print(f"queen = {queen}")
    print(f"apple = {apple}")
    
    def cosine_similarity(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    def euclidean_distance(u, v):
        return np.linalg.norm(u - v)
    
    print("\nCosine Similarity (higher = more similar):")
    print(f"sim(king, queen) = {cosine_similarity(king, queen):.4f}")
    print(f"sim(king, apple) = {cosine_similarity(king, apple):.4f}")
    
    print("\nEuclidean Distance (lower = more similar):")
    print(f"dist(king, queen) = {euclidean_distance(king, queen):.4f}")
    print(f"dist(king, apple) = {euclidean_distance(king, apple):.4f}")
    
    # Word analogy
    print("\nWord Analogy: king - man + woman ≈ queen")
    man = np.array([0.9, 0.4, 0.1])
    woman = np.array([0.6, 0.8, 0.2])
    
    result = king - man + woman
    print(f"king - man + woman = {result}")
    print(f"Similarity to queen: {cosine_similarity(result, queen):.4f}")


# =============================================================================
# EXAMPLE 5: Projection
# =============================================================================

def example_projection():
    """
    Demonstrate vector projection.
    
    Projection is used in:
    - Least squares regression
    - PCA
    - Gram-Schmidt orthogonalization
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Vector Projection")
    print("=" * 60)
    
    # Vector to project
    u = np.array([3, 4])
    # Vector to project onto
    v = np.array([4, 0])
    
    print(f"\nProject u onto v:")
    print(f"u = {u}")
    print(f"v = {v}")
    
    # Projection formula: proj_v(u) = (u·v / v·v) * v
    scalar = np.dot(u, v) / np.dot(v, v)
    proj = scalar * v
    
    print(f"\nScalar component: (u·v)/(v·v) = {scalar}")
    print(f"Projection: proj_v(u) = {proj}")
    
    # The component perpendicular to v
    perp = u - proj
    print(f"Perpendicular component: u - proj = {perp}")
    
    # Verify: proj and perp should be orthogonal
    print(f"\nVerification: proj · perp = {np.dot(proj, perp):.10f} (should be ~0)")


# =============================================================================
# EXAMPLE 6: Linear Independence
# =============================================================================

def example_linear_independence():
    """
    Demonstrate linear independence and dependence.
    
    Linear independence is crucial for:
    - Understanding feature redundancy
    - Basis vectors
    - Matrix rank
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Linear Independence")
    print("=" * 60)
    
    # Independent vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([0, 0, 1])
    
    print("\nLinearly Independent Vectors (Standard Basis):")
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v3 = {v3}")
    
    # Check: form matrix and compute rank
    A = np.column_stack([v1, v2, v3])
    rank = np.linalg.matrix_rank(A)
    print(f"Rank of [v1, v2, v3] = {rank} (= 3, so independent)")
    
    # Dependent vectors
    u1 = np.array([1, 2])
    u2 = np.array([2, 4])  # u2 = 2 * u1
    
    print("\nLinearly Dependent Vectors:")
    print(f"u1 = {u1}")
    print(f"u2 = {u2} = 2 * u1")
    
    B = np.column_stack([u1, u2])
    rank = np.linalg.matrix_rank(B)
    print(f"Rank of [u1, u2] = {rank} (< 2, so dependent)")
    
    print("\nML Insight: Dependent features don't add information!")


# =============================================================================
# EXAMPLE 7: Visualization
# =============================================================================

def example_visualization():
    """
    Visualize vectors and their operations.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Vector Visualization")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Plot 1: Vector Addition
    ax = axes[0, 0]
    u = np.array([3, 1])
    v = np.array([1, 3])
    
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label='u', width=0.02)
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color='red', label='v', width=0.02)
    ax.quiver(0, 0, u[0]+v[0], u[1]+v[1], angles='xy', scale_units='xy', scale=1, 
              color='green', label='u + v', width=0.02)
    # Parallelogram
    ax.plot([u[0], u[0]+v[0]], [u[1], u[1]+v[1]], 'g--', alpha=0.5)
    ax.plot([v[0], u[0]+v[0]], [v[1], u[1]+v[1]], 'g--', alpha=0.5)
    
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Vector Addition')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 2: Scalar Multiplication
    ax = axes[0, 1]
    v = np.array([2, 1])
    scalars = [0.5, 1, 2, -1]
    colors = ['lightblue', 'blue', 'darkblue', 'red']
    
    for c, color in zip(scalars, colors):
        sv = c * v
        ax.quiver(0, 0, sv[0], sv[1], angles='xy', scale_units='xy', scale=1, 
                  color=color, label=f'{c}v', width=0.02)
    
    ax.set_xlim(-3, 5)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Scalar Multiplication')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 3: Projection
    ax = axes[1, 0]
    u = np.array([3, 4])
    v = np.array([5, 0])
    
    # Compute projection
    proj = (np.dot(u, v) / np.dot(v, v)) * v
    perp = u - proj
    
    ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label='u', width=0.02)
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, 
              color='gray', label='v', width=0.02, alpha=0.5)
    ax.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, 
              color='green', label='proj_v(u)', width=0.02)
    ax.quiver(proj[0], proj[1], perp[0], perp[1], angles='xy', scale_units='xy', scale=1, 
              color='red', label='perpendicular', width=0.02)
    
    # Right angle marker
    ax.plot([proj[0], proj[0]], [proj[1], proj[1]+0.3], 'r-', linewidth=1)
    ax.plot([proj[0], proj[0]+0.3], [proj[1], proj[1]], 'r-', linewidth=1)
    
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Projection')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    # Plot 4: Unit Circles (Norm Balls)
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 100)
    
    # L2 unit circle
    x_l2 = np.cos(theta)
    y_l2 = np.sin(theta)
    ax.plot(x_l2, y_l2, 'b-', label='L2 (circle)', linewidth=2)
    
    # L1 unit "circle" (diamond)
    t = np.linspace(0, 1, 50)
    x_l1 = np.concatenate([t, 1-t, -t, t-1])
    y_l1 = np.concatenate([1-t, -t, t-1, t])
    ax.plot(x_l1, y_l1, 'r-', label='L1 (diamond)', linewidth=2)
    
    # L∞ unit "circle" (square)
    ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], 'g-', label='L∞ (square)', linewidth=2)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Unit Balls for Different Norms')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('vectors.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved as 'vectors.png'")


# =============================================================================
# EXAMPLE 8: ML Application - Feature Similarity
# =============================================================================

def example_ml_application():
    """
    Demonstrate vectors in a real ML context.
    
    Scenario: Finding similar customers based on features.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 8: ML Application - Customer Similarity")
    print("=" * 60)
    
    # Customer features: [age, income (normalized), purchase_frequency, account_age]
    customers = {
        'Alice': np.array([0.3, 0.7, 0.8, 0.5]),
        'Bob': np.array([0.35, 0.65, 0.75, 0.55]),  # Similar to Alice
        'Charlie': np.array([0.8, 0.3, 0.2, 0.9]),  # Different
        'Diana': np.array([0.32, 0.72, 0.78, 0.48]),  # Very similar to Alice
    }
    
    print("\nCustomer Feature Vectors (normalized):")
    for name, features in customers.items():
        print(f"  {name}: {features}")
    
    def cosine_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    # Find most similar to Alice
    print("\nSimilarity to Alice (cosine similarity):")
    alice = customers['Alice']
    similarities = []
    
    for name, features in customers.items():
        if name != 'Alice':
            sim = cosine_sim(alice, features)
            similarities.append((name, sim))
            print(f"  Alice ↔ {name}: {sim:.4f}")
    
    # Rank by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    print(f"\nMost similar to Alice: {similarities[0][0]}")
    
    # Using Euclidean distance
    print("\nDistance to Alice (Euclidean):")
    for name, features in customers.items():
        if name != 'Alice':
            dist = np.linalg.norm(alice - features)
            print(f"  Alice ↔ {name}: {dist:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("       VECTORS AND SPACES - EXAMPLES")
    print("=" * 60)
    
    example_vector_operations()
    example_dot_product_geometry()
    example_norms()
    example_similarity()
    example_projection()
    example_linear_independence()
    example_visualization()
    example_ml_application()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
