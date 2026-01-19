"""
Matrix Norms and Condition Numbers - Examples
=============================================
Practical demonstrations of matrix norm concepts.
"""

import numpy as np
from numpy.linalg import norm, cond, svd, inv


def example_vector_norms():
    """Demonstrate vector norms."""
    print("=" * 60)
    print("EXAMPLE 1: Vector Norms")
    print("=" * 60)
    
    x = np.array([3, -4, 0, 2])
    
    print(f"x = {x}")
    
    # L1 norm (Manhattan)
    l1 = norm(x, 1)
    print(f"\nL¹ norm (Manhattan): ||x||₁ = Σ|xᵢ| = {l1}")
    print(f"  = |3| + |-4| + |0| + |2| = {abs(3) + abs(-4) + abs(0) + abs(2)}")
    
    # L2 norm (Euclidean)
    l2 = norm(x, 2)
    print(f"\nL² norm (Euclidean): ||x||₂ = √(Σxᵢ²) = {l2:.4f}")
    print(f"  = √(9 + 16 + 0 + 4) = √29 = {np.sqrt(29):.4f}")
    
    # L∞ norm (Max)
    linf = norm(x, np.inf)
    print(f"\nL∞ norm (Max): ||x||_∞ = max|xᵢ| = {linf}")
    
    # General p-norm
    p = 3
    lp = norm(x, p)
    print(f"\nL{p} norm: ||x||_{p} = (Σ|xᵢ|^{p})^(1/{p}) = {lp:.4f}")


def example_matrix_norms():
    """Demonstrate matrix norms."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Matrix Norms")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4]])
    
    print(f"A = \n{A}")
    
    # L1 norm (max column sum)
    l1_norm = norm(A, 1)
    print(f"\nL¹ norm (max column sum):")
    print(f"  Column 1: |1| + |3| = 4")
    print(f"  Column 2: |2| + |4| = 6")
    print(f"  ||A||₁ = max(4, 6) = {l1_norm}")
    
    # L∞ norm (max row sum)
    linf_norm = norm(A, np.inf)
    print(f"\nL∞ norm (max row sum):")
    print(f"  Row 1: |1| + |2| = 3")
    print(f"  Row 2: |3| + |4| = 7")
    print(f"  ||A||_∞ = max(3, 7) = {linf_norm}")
    
    # Spectral norm (largest singular value)
    U, S, Vt = svd(A)
    l2_norm = norm(A, 2)
    print(f"\nSpectral norm (largest singular value):")
    print(f"  Singular values: {np.round(S, 4)}")
    print(f"  ||A||₂ = σ_max = {l2_norm:.4f}")
    
    # Frobenius norm
    fro_norm = norm(A, 'fro')
    print(f"\nFrobenius norm:")
    print(f"  ||A||_F = √(Σaᵢⱼ²) = √(1 + 4 + 9 + 16) = √30 = {fro_norm:.4f}")
    print(f"  Also: ||A||_F = √(Σσᵢ²) = √({S[0]**2:.4f} + {S[1]**2:.4f}) = {np.sqrt(S[0]**2 + S[1]**2):.4f}")


def example_norm_properties():
    """Demonstrate properties of matrix norms."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Norm Properties")
    print("=" * 60)
    
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[2, 0], [1, 1]])
    c = 3
    
    print(f"A = \n{A}")
    print(f"B = \n{B}")
    print(f"c = {c}")
    
    # Homogeneity
    print(f"\n1. Homogeneity: ||cA|| = |c|||A||")
    print(f"   ||{c}A||_F = {norm(c * A, 'fro'):.4f}")
    print(f"   |{c}| × ||A||_F = {c} × {norm(A, 'fro'):.4f} = {c * norm(A, 'fro'):.4f}")
    
    # Triangle inequality
    print(f"\n2. Triangle inequality: ||A + B|| ≤ ||A|| + ||B||")
    print(f"   ||A + B||_F = {norm(A + B, 'fro'):.4f}")
    print(f"   ||A||_F + ||B||_F = {norm(A, 'fro'):.4f} + {norm(B, 'fro'):.4f} = {norm(A, 'fro') + norm(B, 'fro'):.4f}")
    print(f"   {norm(A + B, 'fro'):.4f} ≤ {norm(A, 'fro') + norm(B, 'fro'):.4f} ✓")
    
    # Submultiplicativity
    print(f"\n3. Submultiplicativity: ||AB|| ≤ ||A|| ||B||")
    print(f"   ||AB||_F = {norm(A @ B, 'fro'):.4f}")
    print(f"   ||A||_F × ||B||_F = {norm(A, 'fro'):.4f} × {norm(B, 'fro'):.4f} = {norm(A, 'fro') * norm(B, 'fro'):.4f}")
    print(f"   {norm(A @ B, 'fro'):.4f} ≤ {norm(A, 'fro') * norm(B, 'fro'):.4f} ✓")


def example_condition_number():
    """Demonstrate condition number."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Condition Number")
    print("=" * 60)
    
    # Well-conditioned matrix
    A = np.array([[1, 0],
                  [0, 1]])
    
    print("Well-conditioned (Identity):")
    print(f"A = \n{A}")
    print(f"κ(A) = {cond(A):.4f}")
    
    # Slightly ill-conditioned
    B = np.array([[1, 0],
                  [0, 0.1]])
    
    print("\n" + "-" * 40)
    print("\nMildly ill-conditioned:")
    print(f"B = \n{B}")
    U, S, Vt = svd(B)
    print(f"Singular values: {S}")
    print(f"κ(B) = σ_max/σ_min = {S[0]}/{S[1]} = {cond(B):.4f}")
    
    # Very ill-conditioned
    C = np.array([[1, 1],
                  [1, 1.0001]])
    
    print("\n" + "-" * 40)
    print("\nVery ill-conditioned:")
    print(f"C = \n{C}")
    U, S, Vt = svd(C)
    print(f"Singular values: {S}")
    print(f"κ(C) = {cond(C):.1f}")
    
    # Effect on solution
    print("\n--- Effect on solution ---")
    b = np.array([2, 2])
    x = np.linalg.solve(C, b)
    print(f"Solving Cx = b where b = {b}")
    print(f"Solution x = {x}")
    
    # Perturb b slightly
    b_perturbed = np.array([2.001, 2])
    x_perturbed = np.linalg.solve(C, b_perturbed)
    print(f"\nPerturbed b = {b_perturbed} (0.05% change)")
    print(f"New solution = {x_perturbed}")
    print(f"Change in x: {100 * norm(x_perturbed - x) / norm(x):.1f}%")
    print("Small input change → Large output change!")


def example_hilbert_matrix():
    """Demonstrate ill-conditioning of Hilbert matrix."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Hilbert Matrix (Classic Ill-Conditioning)")
    print("=" * 60)
    
    def hilbert(n):
        """Generate n×n Hilbert matrix."""
        return np.array([[1/(i + j + 1) for j in range(n)] for i in range(n)])
    
    print("Hilbert matrix H_n where H_ij = 1/(i + j - 1)")
    
    for n in [3, 5, 7, 10]:
        H = hilbert(n)
        kappa = cond(H)
        print(f"\nn = {n}: κ(H) = {kappa:.2e}")
        
        # Demonstrate error
        if n == 5:
            print("\n  Solving Hx = b:")
            x_true = np.ones(n)
            b = H @ x_true
            x_computed = np.linalg.solve(H, b)
            error = norm(x_computed - x_true) / norm(x_true)
            print(f"  True x = ones, Computed x error = {error:.2e}")
    
    print("\nNote: Hilbert matrices become exponentially ill-conditioned!")


def example_regularization():
    """Demonstrate regularization and condition number."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Regularization Effect on Condition Number")
    print("=" * 60)
    
    # Create ill-conditioned matrix
    X = np.array([[1, 1, 1],
                  [1, 1.001, 1],
                  [1, 1, 1.001]])
    
    XTX = X.T @ X
    print(f"X^T X = \n{np.round(XTX, 4)}")
    print(f"κ(X^T X) = {cond(XTX):.2e}")
    
    # Add L2 regularization
    print("\n--- Adding L2 regularization (ridge) ---")
    
    for lam in [0.001, 0.01, 0.1, 1.0]:
        XTX_reg = XTX + lam * np.eye(3)
        print(f"\nλ = {lam}:")
        print(f"  κ(X^T X + λI) = {cond(XTX_reg):.2e}")
    
    print("\nRegularization improves conditioning!")


def example_spectral_normalization():
    """Demonstrate spectral normalization for neural networks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Spectral Normalization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Random weight matrix
    W = np.random.randn(4, 4) * 2
    
    print("Original weight matrix W:")
    print(f"||W||₂ (spectral norm) = {norm(W, 2):.4f}")
    
    # Spectral normalization: W / ||W||_2
    W_normalized = W / norm(W, 2)
    
    print("\nAfter spectral normalization:")
    print(f"||W_normalized||₂ = {norm(W_normalized, 2):.4f}")
    
    # Lipschitz constant for layer
    print("\n--- Lipschitz Analysis ---")
    x = np.random.randn(4)
    y = np.random.randn(4)
    
    print(f"For inputs x, y:")
    print(f"||Wx - Wy|| / ||x - y|| = {norm(W @ x - W @ y) / norm(x - y):.4f}")
    print(f"This is bounded by ||W||₂ = {norm(W, 2):.4f}")
    
    print(f"\nWith normalized W:")
    print(f"||W_n x - W_n y|| / ||x - y|| = {norm(W_normalized @ x - W_normalized @ y) / norm(x - y):.4f}")
    print(f"This is bounded by ||W_n||₂ = 1.0")


def example_nuclear_norm():
    """Demonstrate nuclear norm for low-rank."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Nuclear Norm")
    print("=" * 60)
    
    # Full rank matrix
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])  # Slightly modified to be full rank
    
    U, S, Vt = svd(A)
    
    print(f"Matrix A:\n{A}")
    print(f"\nSingular values: {np.round(S, 4)}")
    print(f"Nuclear norm ||A||_* = Σσᵢ = {np.sum(S):.4f}")
    print(f"Frobenius norm ||A||_F = √(Σσᵢ²) = {norm(A, 'fro'):.4f}")
    print(f"Spectral norm ||A||₂ = σ_max = {S[0]:.4f}")
    
    # Low-rank approximation
    print("\n--- Low-Rank Approximation ---")
    for rank in [1, 2]:
        A_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        print(f"\nRank-{rank} approximation:")
        print(f"||A||_* = {np.sum(S[:rank]):.4f}")
        print(f"||A - A_approx||_F = {norm(A - A_approx, 'fro'):.4f}")


def example_norm_comparison():
    """Compare different norms."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Norm Comparison")
    print("=" * 60)
    
    A = np.array([[3, 0, 0],
                  [0, 2, 0],
                  [0, 0, 1]])
    
    print(f"Diagonal matrix A:\n{A}")
    
    norms = {
        '||A||₁': norm(A, 1),
        '||A||₂': norm(A, 2),
        '||A||_∞': norm(A, np.inf),
        '||A||_F': norm(A, 'fro'),
        '||A||_*': np.sum(svd(A)[1])
    }
    
    print("\nAll norms:")
    for name, value in norms.items():
        print(f"  {name} = {value:.4f}")
    
    print("\nRelationships:")
    print(f"||A||₂ ≤ ||A||_F: {norms['||A||₂']:.4f} ≤ {norms['||A||_F']:.4f} ✓")
    print(f"||A||_F ≤ √rank × ||A||₂: {norms['||A||_F']:.4f} ≤ {np.sqrt(3) * norms['||A||₂']:.4f} ✓")


def example_gradient_condition():
    """Demonstrate condition number effect on optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Condition Number and Optimization")
    print("=" * 60)
    
    # Well-conditioned Hessian
    H_good = np.array([[1, 0],
                       [0, 1]])
    
    # Ill-conditioned Hessian
    H_bad = np.array([[1, 0],
                      [0, 100]])
    
    print("Quadratic: f(x) = 0.5 x^T H x")
    print("\nWell-conditioned:")
    print(f"H = \n{H_good}")
    print(f"κ(H) = {cond(H_good)}")
    print("Gradient descent: Fast convergence in all directions")
    
    print("\nIll-conditioned:")
    print(f"H = \n{H_bad}")
    print(f"κ(H) = {cond(H_bad)}")
    print("Gradient descent: Slow (zigzag) in direction with small eigenvalue")
    
    # Simulate gradient descent
    print("\n--- Gradient Descent Simulation ---")
    
    def gradient_descent(H, x0, lr, steps):
        x = x0.copy()
        for _ in range(steps):
            grad = H @ x
            x = x - lr * grad
        return x
    
    x0 = np.array([10.0, 10.0])
    
    # Well-conditioned
    x_good = gradient_descent(H_good, x0, 0.1, 50)
    print(f"H_good: After 50 steps with lr=0.1, x = {np.round(x_good, 6)}")
    
    # Ill-conditioned (need smaller lr for stability)
    x_bad = gradient_descent(H_bad, x0, 0.01, 50)
    print(f"H_bad: After 50 steps with lr=0.01, x = {np.round(x_bad, 4)}")
    print("  (Needs smaller learning rate and more iterations!)")


def example_numerical_stability():
    """Demonstrate numerical stability analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Numerical Stability")
    print("=" * 60)
    
    def solve_with_error(A, b, epsilon=1e-10):
        """Solve Ax = b with small perturbation."""
        # Add small noise to A
        A_noisy = A + epsilon * np.random.randn(*A.shape)
        
        x_true = np.linalg.solve(A, b)
        x_noisy = np.linalg.solve(A_noisy, b)
        
        rel_change_A = norm(A_noisy - A) / norm(A)
        rel_change_x = norm(x_noisy - x_true) / norm(x_true)
        
        return rel_change_A, rel_change_x
    
    np.random.seed(42)
    
    # Well-conditioned
    A_good = np.array([[2, 1],
                       [1, 2]])
    b = np.array([1, 1])
    
    print(f"Well-conditioned A:\n{A_good}")
    print(f"κ(A) = {cond(A_good):.2f}")
    
    change_A, change_x = solve_with_error(A_good, b)
    print(f"Relative change in A: {change_A:.2e}")
    print(f"Relative change in x: {change_x:.2e}")
    print(f"Amplification: {change_x / change_A:.1f}x")
    
    # Ill-conditioned
    print("\n" + "-" * 40)
    A_bad = np.array([[1, 1],
                      [1, 1.001]])
    
    print(f"\nIll-conditioned A:\n{A_bad}")
    print(f"κ(A) = {cond(A_bad):.0f}")
    
    change_A, change_x = solve_with_error(A_bad, b)
    print(f"Relative change in A: {change_A:.2e}")
    print(f"Relative change in x: {change_x:.2e}")
    print(f"Amplification: {change_x / change_A:.0f}x")
    
    print("\nRule: Error can be amplified by factor κ(A)!")


def visualize_condition_number():
    """Visualize condition number geometrically."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Condition Number Geometry")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    
    # Create matrices with different condition numbers
    matrices = {
        'κ = 1 (Identity)': np.eye(2),
        'κ = 3': np.array([[3, 0], [0, 1]]),
        'κ = 10': np.array([[10, 0], [0, 1]])
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])
    
    for ax, (name, A) in zip(axes, matrices.items()):
        # Transform unit circle
        ellipse = A @ circle
        
        ax.plot(circle[0], circle[1], 'b--', alpha=0.3, label='Unit circle')
        ax.fill(ellipse[0], ellipse[1], alpha=0.3, color='red')
        ax.plot(ellipse[0], ellipse[1], 'r-', linewidth=2, label='A × (unit circle)')
        
        ax.set_xlim(-11, 11)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_title(name)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('condition_number_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: condition_number_visualization.png")
    print("\nIll-conditioned matrices stretch the unit circle into thin ellipses.")


if __name__ == "__main__":
    example_vector_norms()
    example_matrix_norms()
    example_norm_properties()
    example_condition_number()
    example_hilbert_matrix()
    example_regularization()
    example_spectral_normalization()
    example_nuclear_norm()
    example_norm_comparison()
    example_gradient_condition()
    example_numerical_stability()
    
    # Uncomment to generate visualization
    # visualize_condition_number()
