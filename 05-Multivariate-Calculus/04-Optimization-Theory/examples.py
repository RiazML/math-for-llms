"""
Optimization Theory - Examples
==============================
Demonstrations of optimization concepts and algorithms.
"""

import numpy as np


def example_unconstrained_optimization():
    """Find minimum of unconstrained function."""
    print("=" * 60)
    print("EXAMPLE 1: Unconstrained Optimization")
    print("=" * 60)
    
    print("f(x, y) = x² + y² + 2x + 4y")
    
    print("\nFirst-order condition: ∇f = 0")
    print("  ∂f/∂x = 2x + 2 = 0 → x = -1")
    print("  ∂f/∂y = 2y + 4 = 0 → y = -2")
    
    print("\nCritical point: (-1, -2)")
    
    print("\nSecond-order check: Hessian")
    print("  H = [2  0]")
    print("      [0  2]")
    print("  Eigenvalues: 2, 2 (both positive)")
    print("  → Positive definite → LOCAL MINIMUM")
    
    # Verify
    x_opt, y_opt = -1, -2
    f_opt = x_opt**2 + y_opt**2 + 2*x_opt + 4*y_opt
    print(f"\nf(-1, -2) = {f_opt}")


def example_saddle_point():
    """Identify saddle point using Hessian."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Saddle Point Detection")
    print("=" * 60)
    
    print("f(x, y) = x² - y²")
    
    print("\n∇f = [2x, -2y]")
    print("Critical point at (0, 0)")
    
    print("\nHessian:")
    print("  H = [2   0 ]")
    print("      [0  -2]")
    
    print("\nEigenvalues: 2 and -2")
    print("Mixed signs → SADDLE POINT")
    
    print("\nVisualization:")
    print("  Along x-axis: f(x, 0) = x² (bowl shape, minimum)")
    print("  Along y-axis: f(0, y) = -y² (inverted bowl, maximum)")


def example_lagrange_circle():
    """Lagrange multipliers for constraint on circle."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Lagrange Multipliers - Circle Constraint")
    print("=" * 60)
    
    print("Maximize f(x, y) = x + y")
    print("Subject to: g(x, y) = x² + y² - 1 = 0")
    
    print("\nLagrangian: L = x + y + λ(x² + y² - 1)")
    
    print("\nConditions:")
    print("  ∂L/∂x = 1 + 2λx = 0  →  x = -1/(2λ)")
    print("  ∂L/∂y = 1 + 2λy = 0  →  y = -1/(2λ)")
    print("  ∂L/∂λ = x² + y² - 1 = 0")
    
    print("\nFrom x = y, substitute into constraint:")
    print("  2x² = 1  →  x = ±1/√2")
    
    print("\nSolutions:")
    sqrt2 = np.sqrt(2)
    print(f"  Maximum: (1/√2, 1/√2) ≈ ({1/sqrt2:.4f}, {1/sqrt2:.4f})")
    print(f"  f_max = √2 ≈ {sqrt2:.4f}")
    print(f"  Minimum: (-1/√2, -1/√2) ≈ ({-1/sqrt2:.4f}, {-1/sqrt2:.4f})")
    print(f"  f_min = -√2 ≈ {-sqrt2:.4f}")


def example_lagrange_multiple_constraints():
    """Lagrange multipliers with multiple constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multiple Equality Constraints")
    print("=" * 60)
    
    print("Minimize f(x, y, z) = x² + y² + z²")
    print("Subject to:")
    print("  g₁: x + y + z = 1")
    print("  g₂: x + 2y + 3z = 2")
    
    print("\nL = x² + y² + z² + λ₁(x + y + z - 1) + λ₂(x + 2y + 3z - 2)")
    
    print("\nOptimality conditions:")
    print("  2x + λ₁ + λ₂ = 0")
    print("  2y + λ₁ + 2λ₂ = 0")
    print("  2z + λ₁ + 3λ₂ = 0")
    print("  x + y + z = 1")
    print("  x + 2y + 3z = 2")
    
    # Solve system
    # From first 3 equations:
    # x = -(λ₁ + λ₂)/2
    # y = -(λ₁ + 2λ₂)/2
    # z = -(λ₁ + 3λ₂)/2
    
    # Substitute into constraints:
    # -(λ₁ + λ₂)/2 - (λ₁ + 2λ₂)/2 - (λ₁ + 3λ₂)/2 = 1
    # -3λ₁/2 - 6λ₂/2 = 1  →  -3λ₁ - 6λ₂ = 2 ... (A)
    
    # -(λ₁ + λ₂)/2 - 2(λ₁ + 2λ₂)/2 - 3(λ₁ + 3λ₂)/2 = 2
    # -(λ₁ + λ₂ + 2λ₁ + 4λ₂ + 3λ₁ + 9λ₂)/2 = 2
    # -6λ₁ - 14λ₂ = 4 ... (B)
    
    # Solve: From (A): λ₁ = (-2 - 6λ₂)/3
    # Substitute into (B): -6(-2 - 6λ₂)/3 - 14λ₂ = 4
    # 4 + 12λ₂ - 14λ₂ = 4  →  λ₂ = 0
    # λ₁ = -2/3
    
    lambda1 = -2/3
    lambda2 = 0
    
    x = -(lambda1 + lambda2)/2
    y = -(lambda1 + 2*lambda2)/2
    z = -(lambda1 + 3*lambda2)/2
    
    print(f"\nSolution:")
    print(f"  λ₁ = {lambda1:.4f}, λ₂ = {lambda2:.4f}")
    print(f"  (x, y, z) = ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # Verify constraints
    print(f"\nVerification:")
    print(f"  x + y + z = {x + y + z:.4f}")
    print(f"  x + 2y + 3z = {x + 2*y + 3*z:.4f}")
    print(f"  f(x,y,z) = {x**2 + y**2 + z**2:.4f}")


def example_kkt_inequality():
    """KKT conditions with inequality constraint."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: KKT Conditions - Inequality Constraint")
    print("=" * 60)
    
    print("Minimize f(x) = (x - 2)²")
    print("Subject to: h(x) = x - 1 ≤ 0  (i.e., x ≤ 1)")
    
    print("\nLagrangian: L = (x - 2)² + μ(x - 1)")
    
    print("\nKKT conditions:")
    print("  1. Stationarity: 2(x - 2) + μ = 0")
    print("  2. Primal feasibility: x ≤ 1")
    print("  3. Dual feasibility: μ ≥ 0")
    print("  4. Complementary slackness: μ(x - 1) = 0")
    
    print("\nCase 1: Constraint inactive (μ = 0)")
    print("  From stationarity: x = 2")
    print("  Check primal: 2 ≤ 1? NO - infeasible")
    
    print("\nCase 2: Constraint active (x = 1)")
    print("  From stationarity: 2(1-2) + μ = 0  →  μ = 2")
    print("  Check dual: μ = 2 ≥ 0? YES")
    
    print("\nSolution: x* = 1, μ* = 2")
    print(f"f(1) = {(1-2)**2}")


def example_convexity_check():
    """Check convexity of functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Convexity Verification")
    print("=" * 60)
    
    print("Check convexity by examining Hessian eigenvalues.")
    
    print("\n--- f(x, y) = x² + y² (Quadratic) ---")
    print("H = [2 0]")
    print("    [0 2]")
    print("Eigenvalues: 2, 2 (all positive)")
    print("→ CONVEX (and strongly convex)")
    
    print("\n--- f(x, y) = e^x + y² ---")
    print("H = [e^x  0]")
    print("    [0    2]")
    print("Eigenvalues: e^x > 0, 2 > 0 (all positive for all x)")
    print("→ CONVEX")
    
    print("\n--- f(x, y) = x² - y² ---")
    print("H = [2   0]")
    print("    [0  -2]")
    print("Eigenvalues: 2, -2 (mixed signs)")
    print("→ NOT CONVEX (indefinite)")
    
    print("\n--- f(x) = log(x) for x > 0 ---")
    print("f'(x) = 1/x, f''(x) = -1/x² < 0")
    print("→ CONCAVE (not convex)")


def example_ridge_regression():
    """Ridge regression as constrained optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Ridge Regression")
    print("=" * 60)
    
    print("Constrained form:")
    print("  min ||Xw - y||²  s.t. ||w||² ≤ t")
    
    print("\nLagrangian form (equivalent):")
    print("  min ||Xw - y||² + λ||w||²")
    
    print("\nSolution:")
    print("  ∇[||Xw - y||² + λ||w||²] = 0")
    print("  2Xᵀ(Xw - y) + 2λw = 0")
    print("  (XᵀX + λI)w = Xᵀy")
    print("  w* = (XᵀX + λI)⁻¹Xᵀy")
    
    # Example
    np.random.seed(42)
    n, d = 10, 3
    X = np.random.randn(n, d)
    w_true = np.array([1, 2, 3])
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    lam = 0.1
    
    # Ridge solution
    w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    
    # OLS solution for comparison
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    
    print(f"\nExample with λ = {lam}:")
    print(f"  True w:  {w_true}")
    print(f"  OLS w:   {w_ols.round(3)}")
    print(f"  Ridge w: {w_ridge.round(3)}")
    print(f"  ||w_ols||² = {np.sum(w_ols**2):.4f}")
    print(f"  ||w_ridge||² = {np.sum(w_ridge**2):.4f}")


def example_svm_dual():
    """SVM dual problem derivation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: SVM Dual Problem")
    print("=" * 60)
    
    print("Hard-margin SVM primal:")
    print("  min ½||w||²")
    print("  s.t. yᵢ(wᵀxᵢ + b) ≥ 1")
    
    print("\nLagrangian:")
    print("  L = ½||w||² - Σᵢ αᵢ[yᵢ(wᵀxᵢ + b) - 1]")
    
    print("\nStationarity conditions:")
    print("  ∂L/∂w = w - Σᵢ αᵢyᵢxᵢ = 0  →  w = Σᵢ αᵢyᵢxᵢ")
    print("  ∂L/∂b = -Σᵢ αᵢyᵢ = 0  →  Σᵢ αᵢyᵢ = 0")
    
    print("\nSubstitute back to get dual:")
    print("  max Σᵢ αᵢ - ½ Σᵢⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ")
    print("  s.t. αᵢ ≥ 0, Σᵢ αᵢyᵢ = 0")
    
    print("\nKernel trick: Replace xᵢᵀxⱼ with K(xᵢ, xⱼ)")
    
    # Simple example
    print("\n--- Simple 2D Example ---")
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 2]])
    y = np.array([1, 1, -1, -1])
    
    print(f"Data points: {X.shape[0]} samples")
    print(f"Labels: {y}")
    
    # Compute Gram matrix (kernel)
    K = X @ X.T
    print(f"\nGram matrix K = XXᵀ:")
    print(K)


def example_pca_as_optimization():
    """PCA as constrained optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: PCA as Constrained Optimization")
    print("=" * 60)
    
    print("First principal component:")
    print("  max vᵀCv  s.t. ||v||² = 1")
    print("  (C = data covariance matrix)")
    
    print("\nLagrangian: L = vᵀCv - λ(vᵀv - 1)")
    print("\nStationarity: ∂L/∂v = 2Cv - 2λv = 0")
    print("  → Cv = λv")
    print("\nThis is an eigenvalue problem!")
    print("v* is the eigenvector with largest eigenvalue")
    
    # Example
    np.random.seed(42)
    n = 100
    
    # Data with clear principal direction
    X = np.random.randn(n, 2)
    X[:, 0] *= 3  # Stretch in first direction
    
    # Covariance matrix
    C = (X.T @ X) / n
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"\nExample:")
    print(f"  Covariance matrix:\n{C.round(3)}")
    print(f"  Eigenvalues: {eigenvalues.round(3)}")
    print(f"  First PC: {eigenvectors[:, 0].round(3)}")
    print(f"  Variance explained: {eigenvalues[0]/sum(eigenvalues)*100:.1f}%")


def example_gradient_descent():
    """Gradient descent optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Gradient Descent")
    print("=" * 60)
    
    print("Minimize f(x, y) = (x - 1)² + 4(y - 2)²")
    
    def f(x):
        return (x[0] - 1)**2 + 4*(x[1] - 2)**2
    
    def grad_f(x):
        return np.array([2*(x[0] - 1), 8*(x[1] - 2)])
    
    # Gradient descent
    x = np.array([0.0, 0.0])
    lr = 0.1
    history = [x.copy()]
    
    print(f"\nInitial point: {x}")
    print(f"Learning rate: {lr}")
    
    for i in range(20):
        grad = grad_f(x)
        x = x - lr * grad
        history.append(x.copy())
        if i < 5 or i == 19:
            print(f"Iter {i+1:2d}: x = {x.round(4)}, f(x) = {f(x):.6f}")
    
    print(f"\nOptimum found: {x.round(6)}")
    print(f"True optimum: [1, 2]")


def example_newton_method():
    """Newton's method for optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Newton's Method")
    print("=" * 60)
    
    print("Minimize f(x, y) = (x - 1)² + 4(y - 2)²")
    
    def f(x):
        return (x[0] - 1)**2 + 4*(x[1] - 2)**2
    
    def grad_f(x):
        return np.array([2*(x[0] - 1), 8*(x[1] - 2)])
    
    def hessian_f(x):
        return np.array([[2, 0], [0, 8]])
    
    # Newton's method
    x = np.array([0.0, 0.0])
    
    print(f"\nInitial point: {x}")
    
    for i in range(5):
        grad = grad_f(x)
        H = hessian_f(x)
        H_inv = np.linalg.inv(H)
        x = x - H_inv @ grad
        print(f"Iter {i+1}: x = {x.round(6)}, f(x) = {f(x):.6f}")
    
    print("\nNewton's method converges in ONE step for quadratic!")
    print("(Because the quadratic approximation is exact)")


def example_projected_gradient():
    """Projected gradient descent for constrained optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Projected Gradient Descent")
    print("=" * 60)
    
    print("Minimize f(x) = (x₁ - 2)² + (x₂ - 2)²")
    print("Subject to: x ≥ 0 (non-negative)")
    
    def f(x):
        return (x[0] - 2)**2 + (x[1] - 2)**2
    
    def grad_f(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 2)])
    
    def project(x):
        """Project onto non-negative orthant."""
        return np.maximum(x, 0)
    
    x = np.array([5.0, 0.5])  # Start infeasible-ish
    lr = 0.1
    
    print(f"\nInitial: x = {x}, f = {f(x):.4f}")
    
    for i in range(20):
        grad = grad_f(x)
        x = x - lr * grad
        x = project(x)  # Project to feasible set
        if i < 5 or i == 19:
            print(f"Iter {i+1:2d}: x = {x.round(4)}, f = {f(x):.4f}")
    
    print(f"\nSolution: {x.round(4)}")
    print("True unconstrained optimum is [2, 2], which is feasible!")


if __name__ == "__main__":
    example_unconstrained_optimization()
    example_saddle_point()
    example_lagrange_circle()
    example_lagrange_multiple_constraints()
    example_kkt_inequality()
    example_convexity_check()
    example_ridge_regression()
    example_svm_dual()
    example_pca_as_optimization()
    example_gradient_descent()
    example_newton_method()
    example_projected_gradient()
