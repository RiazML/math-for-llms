"""
Systems of Linear Equations - Examples
======================================
Practical demonstrations of solving linear systems using NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def example_basic_system():
    """Solve a simple 3x3 system."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Linear System")
    print("=" * 60)
    
    # System:
    # x + 2y + z = 9
    # 2x - y + 3z = 8
    # 3x + y - z = 3
    
    A = np.array([[1, 2, 1],
                  [2, -1, 3],
                  [3, 1, -1]], dtype=float)
    
    b = np.array([9, 8, 3], dtype=float)
    
    print("System Ax = b:")
    print(f"A =\n{A}")
    print(f"b = {b}")
    
    # Method 1: np.linalg.solve (recommended)
    x = np.linalg.solve(A, b)
    print(f"\nSolution using np.linalg.solve: x = {x}")
    
    # Verify
    print(f"Verification: Ax = {A @ x}")
    print(f"Residual ||Ax - b|| = {np.linalg.norm(A @ x - b)}")


def example_different_solution_types():
    """Demonstrate unique, infinite, and no solutions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Different Solution Types")
    print("=" * 60)
    
    # 1. Unique solution
    print("\n1. UNIQUE SOLUTION")
    A1 = np.array([[2, 1], [1, 3]])
    b1 = np.array([5, 5])
    x1 = np.linalg.solve(A1, b1)
    print(f"   A = {A1.tolist()}, b = {b1.tolist()}")
    print(f"   Solution: x = {x1}")
    print(f"   rank(A) = {np.linalg.matrix_rank(A1)}")
    
    # 2. No solution (inconsistent)
    print("\n2. NO SOLUTION (Parallel lines)")
    A2 = np.array([[1, 1], [1, 1]])
    b2 = np.array([2, 3])  # Different RHS
    print(f"   A = {A2.tolist()}, b = {b2.tolist()}")
    print(f"   rank(A) = {np.linalg.matrix_rank(A2)}")
    print(f"   rank([A|b]) = {np.linalg.matrix_rank(np.column_stack([A2, b2]))}")
    print("   Since rank(A) < rank([A|b]), no solution exists")
    
    # Least squares "solution"
    x2, residuals, rank, s = np.linalg.lstsq(A2, b2, rcond=None)
    print(f"   Least squares approximation: {x2}")
    
    # 3. Infinite solutions (dependent equations)
    print("\n3. INFINITE SOLUTIONS (Same line)")
    A3 = np.array([[1, 2], [2, 4]])
    b3 = np.array([3, 6])
    print(f"   A = {A3.tolist()}, b = {b3.tolist()}")
    print(f"   rank(A) = {np.linalg.matrix_rank(A3)} < 2 unknowns")
    print("   Infinite solutions: x + 2y = 3, parameterized by y")
    print("   Example solutions: (3, 0), (1, 1), (-1, 2), ...")


def example_gaussian_elimination():
    """Demonstrate Gaussian elimination step by step."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Gaussian Elimination (Manual)")
    print("=" * 60)
    
    # Augmented matrix [A | b]
    aug = np.array([[1., 2., 1., 9.],
                    [2., -1., 3., 8.],
                    [3., 1., -1., 3.]])
    
    print("Initial augmented matrix [A|b]:")
    print(aug)
    
    # Step 1: Eliminate first column below pivot
    print("\nStep 1: R₂ ← R₂ - 2R₁, R₃ ← R₃ - 3R₁")
    aug[1] = aug[1] - 2 * aug[0]
    aug[2] = aug[2] - 3 * aug[0]
    print(aug)
    
    # Step 2: Eliminate second column below pivot
    print("\nStep 2: R₃ ← R₃ - R₂")
    aug[2] = aug[2] - aug[1]
    print(aug)
    
    print("\nRow Echelon Form achieved!")
    
    # Back substitution
    z = aug[2, 3] / aug[2, 2]
    y = (aug[1, 3] - aug[1, 2] * z) / aug[1, 1]
    x = (aug[0, 3] - aug[0, 2] * z - aug[0, 1] * y) / aug[0, 0]
    
    print(f"\nBack substitution:")
    print(f"z = {aug[2,3]}/{aug[2,2]} = {z}")
    print(f"y = ({aug[1,3]} - {aug[1,2]}*{z})/{aug[1,1]} = {y}")
    print(f"x = ({aug[0,3]} - {aug[0,2]}*{z} - {aug[0,1]}*{y})/{aug[0,0]} = {x}")


def example_lu_decomposition():
    """LU decomposition for solving systems."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: LU Decomposition")
    print("=" * 60)
    
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    
    print(f"Matrix A:\n{A}")
    
    # LU decomposition
    P, L, U = linalg.lu(A)
    
    print(f"\nPermutation P:\n{P}")
    print(f"\nLower triangular L:\n{L}")
    print(f"\nUpper triangular U:\n{U}")
    print(f"\nVerify: P @ L @ U =\n{P @ L @ U}")
    print(f"Equals A: {np.allclose(P @ L @ U, A)}")
    
    # Solve Ax = b using LU
    b = np.array([4, 10, 24])
    
    # Method: Solve P @ L @ U @ x = b
    # 1. y = L⁻¹ @ P.T @ b (forward substitution)
    # 2. x = U⁻¹ @ y (back substitution)
    
    lu, piv = linalg.lu_factor(A)
    x = linalg.lu_solve((lu, piv), b)
    
    print(f"\nb = {b}")
    print(f"Solution x = {x}")
    print(f"Verify: Ax = {A @ x}")


def example_cholesky():
    """Cholesky decomposition for symmetric positive definite matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Cholesky Decomposition")
    print("=" * 60)
    
    # Create a symmetric positive definite matrix
    # A = X.T @ X is always positive semi-definite
    X = np.array([[2, 1],
                  [1, 2],
                  [1, 1]])
    A = X.T @ X
    
    print(f"Matrix A = XᵀX:\n{A}")
    print(f"Is symmetric: {np.allclose(A, A.T)}")
    print(f"Eigenvalues: {np.linalg.eigvalsh(A)} (all positive)")
    
    # Cholesky decomposition: A = L @ L.T
    L = np.linalg.cholesky(A)
    
    print(f"\nCholesky factor L:\n{L}")
    print(f"Verify: L @ Lᵀ =\n{L @ L.T}")
    
    # Solve using Cholesky
    b = np.array([10, 8])
    
    # Forward solve: L @ y = b
    y = linalg.solve_triangular(L, b, lower=True)
    # Backward solve: L.T @ x = y
    x = linalg.solve_triangular(L.T, y, lower=False)
    
    print(f"\nb = {b}")
    print(f"Solution x = {x}")
    print(f"Verify: Ax = {A @ x}")


def example_linear_regression():
    """Linear regression as a system of equations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Linear Regression (Normal Equations)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data: y = 2 + 3*x + noise
    n = 50
    x = np.random.uniform(0, 10, n)
    y = 2 + 3 * x + np.random.randn(n) * 2
    
    # Design matrix [1, x]
    X = np.column_stack([np.ones(n), x])
    
    print(f"Data: {n} points")
    print(f"Design matrix X shape: {X.shape}")
    print(f"True parameters: w₀=2, w₁=3")
    
    # Normal equations: (XᵀX)w = Xᵀy
    XtX = X.T @ X
    Xty = X.T @ y
    
    print(f"\nXᵀX =\n{XtX}")
    print(f"Xᵀy = {Xty}")
    
    # Solve
    w = np.linalg.solve(XtX, Xty)
    print(f"\nEstimated parameters: w₀={w[0]:.4f}, w₁={w[1]:.4f}")
    
    # Compare with numpy's lstsq
    w_lstsq, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    print(f"Using lstsq: w₀={w_lstsq[0]:.4f}, w₁={w_lstsq[1]:.4f}")


def example_ridge_regression():
    """Ridge regression for ill-conditioned systems."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Ridge Regression (Regularization)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Create an ill-conditioned system
    # Features are nearly collinear
    n, d = 100, 5
    X = np.random.randn(n, d)
    X[:, 1] = X[:, 0] + 0.01 * np.random.randn(n)  # Nearly collinear
    X[:, 2] = X[:, 0] + X[:, 1] + 0.01 * np.random.randn(n)
    
    true_w = np.array([1, 2, 3, 4, 5])
    y = X @ true_w + np.random.randn(n) * 0.1
    
    # Check condition number
    XtX = X.T @ X
    cond = np.linalg.cond(XtX)
    print(f"Condition number of XᵀX: {cond:.2e}")
    print("(Large condition number = ill-conditioned)")
    
    # Ordinary least squares
    try:
        w_ols = np.linalg.solve(XtX, X.T @ y)
        print(f"\nOLS solution: {w_ols}")
    except np.linalg.LinAlgError:
        print("\nOLS failed (singular matrix)")
        w_ols = None
    
    # Ridge regression: (XᵀX + λI)w = Xᵀy
    lambda_reg = 0.1
    XtX_reg = XtX + lambda_reg * np.eye(d)
    w_ridge = np.linalg.solve(XtX_reg, X.T @ y)
    
    print(f"\nRidge (λ={lambda_reg}) solution: {w_ridge}")
    print(f"True weights: {true_w}")
    print(f"Regularized condition number: {np.linalg.cond(XtX_reg):.2e}")


def example_overdetermined_system():
    """Least squares solution for overdetermined systems."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Overdetermined System (Least Squares)")
    print("=" * 60)
    
    # Fit a quadratic: y = a + bx + cx²
    # to 10 data points (more equations than unknowns)
    
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1.0, 1.8, 1.3, 3.4, 4.2, 5.8, 7.1, 7.9, 9.4, 10.2])
    
    # Design matrix for quadratic fit
    X = np.column_stack([np.ones_like(x), x, x**2])
    
    print(f"Points: {len(x)}")
    print(f"Design matrix shape: {X.shape} (overdetermined)")
    
    # Least squares solution
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    
    print(f"\nFitted coefficients: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}")
    print(f"Model: y = {coeffs[0]:.4f} + {coeffs[1]:.4f}x + {coeffs[2]:.4f}x²")
    
    # Prediction
    y_pred = X @ coeffs
    mse = np.mean((y - y_pred) ** 2)
    print(f"Mean Squared Error: {mse:.4f}")


def example_underdetermined_system():
    """Minimum norm solution for underdetermined systems."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Underdetermined System")
    print("=" * 60)
    
    # 2 equations, 4 unknowns
    A = np.array([[1, 2, 1, 0],
                  [0, 1, 1, 1]])
    b = np.array([4, 3])
    
    print(f"System: 2 equations, 4 unknowns")
    print(f"A =\n{A}")
    print(f"b = {b}")
    
    # Minimum norm solution (pseudo-inverse)
    x_min_norm = np.linalg.lstsq(A, b, rcond=None)[0]
    
    print(f"\nMinimum norm solution: {x_min_norm}")
    print(f"||x||² = {np.linalg.norm(x_min_norm)**2:.4f}")
    print(f"Verify: Ax = {A @ x_min_norm}")
    
    # Alternative solutions (add null space vectors)
    null_space = linalg.null_space(A)
    print(f"\nNull space dimension: {null_space.shape[1]}")
    
    # Another valid solution
    x_alt = x_min_norm + null_space @ np.array([1, 1])
    print(f"\nAlternative solution: {x_alt}")
    print(f"||x_alt||² = {np.linalg.norm(x_alt)**2:.4f} (larger)")
    print(f"Verify: Ax_alt = {A @ x_alt}")


def example_condition_number():
    """Demonstrate the effect of condition number on stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Condition Number and Stability")
    print("=" * 60)
    
    # Well-conditioned matrix
    A_good = np.array([[1, 0], [0, 1]], dtype=float)
    b_good = np.array([1, 1], dtype=float)
    
    # Ill-conditioned matrix (Hilbert matrix)
    A_bad = np.array([[1, 1/2], [1/2, 1/3]], dtype=float)
    b_bad = np.array([1, 1], dtype=float)
    
    print("Well-conditioned system:")
    print(f"  Condition number: {np.linalg.cond(A_good):.2f}")
    
    print("\nIll-conditioned system (Hilbert):")
    print(f"  Condition number: {np.linalg.cond(A_bad):.2f}")
    
    # Solve both
    x_good = np.linalg.solve(A_good, b_good)
    x_bad = np.linalg.solve(A_bad, b_bad)
    
    print(f"\nSolutions:")
    print(f"  Well-conditioned: {x_good}")
    print(f"  Ill-conditioned: {x_bad}")
    
    # Perturb b slightly
    epsilon = 0.01
    b_good_perturbed = b_good + epsilon * np.array([1, 0])
    b_bad_perturbed = b_bad + epsilon * np.array([1, 0])
    
    x_good_perturbed = np.linalg.solve(A_good, b_good_perturbed)
    x_bad_perturbed = np.linalg.solve(A_bad, b_bad_perturbed)
    
    print(f"\nAfter perturbing b by {epsilon}:")
    print(f"  Well-conditioned solution change: {np.linalg.norm(x_good_perturbed - x_good):.4f}")
    print(f"  Ill-conditioned solution change: {np.linalg.norm(x_bad_perturbed - x_bad):.4f}")
    print("\n  Ill-conditioned systems amplify small errors!")


def visualize_linear_systems():
    """Visualize 2D linear systems."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: 2D Linear Systems")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x_range = np.linspace(-2, 4, 100)
    
    # 1. Unique solution
    ax = axes[0]
    # x + y = 3  ->  y = 3 - x
    # 2x - y = 0  ->  y = 2x
    ax.plot(x_range, 3 - x_range, 'b-', label='x + y = 3')
    ax.plot(x_range, 2 * x_range, 'r-', label='2x - y = 0')
    ax.plot(1, 2, 'go', markersize=10, label='Solution (1, 2)')
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 6)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Unique Solution')
    ax.legend()
    
    # 2. No solution (parallel lines)
    ax = axes[1]
    # x + y = 2  ->  y = 2 - x
    # x + y = 4  ->  y = 4 - x
    ax.plot(x_range, 2 - x_range, 'b-', label='x + y = 2')
    ax.plot(x_range, 4 - x_range, 'r-', label='x + y = 4')
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 6)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('No Solution (Parallel Lines)')
    ax.legend()
    
    # 3. Infinite solutions (same line)
    ax = axes[2]
    # x + 2y = 4  ->  y = (4 - x) / 2
    # 2x + 4y = 8  ->  y = (8 - 2x) / 4 = (4 - x) / 2
    ax.plot(x_range, (4 - x_range) / 2, 'b-', linewidth=4, alpha=0.5, label='x + 2y = 4')
    ax.plot(x_range, (4 - x_range) / 2, 'r--', linewidth=2, label='2x + 4y = 8 (same line)')
    ax.set_xlim(-2, 4)
    ax.set_ylim(-2, 6)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title('Infinite Solutions (Same Line)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('linear_systems_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: linear_systems_2d.png")


if __name__ == "__main__":
    example_basic_system()
    example_different_solution_types()
    example_gaussian_elimination()
    example_lu_decomposition()
    example_cholesky()
    example_linear_regression()
    example_ridge_regression()
    example_overdetermined_system()
    example_underdetermined_system()
    example_condition_number()
    
    # Uncomment to generate visualization
    # visualize_linear_systems()
