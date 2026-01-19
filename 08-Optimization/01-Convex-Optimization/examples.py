"""
Convex Optimization - Examples
==============================
Practical demonstrations of convexity in ML.
"""

import numpy as np
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


def example_convex_sets():
    """Verify convexity of sets."""
    print("=" * 60)
    print("EXAMPLE 1: Convex Sets")
    print("=" * 60)
    
    # Check if line segment between two points stays in set
    
    # 1. Ball/Circle
    print("\n1. Ball: {x : ||x|| <= r}")
    x = np.array([0.5, 0.0])
    y = np.array([0.0, 0.8])
    r = 1.0
    
    print(f"   x = {x}, ||x|| = {np.linalg.norm(x):.2f}")
    print(f"   y = {y}, ||y|| = {np.linalg.norm(y):.2f}")
    
    # Check midpoint
    midpoint = 0.5 * x + 0.5 * y
    print(f"   midpoint = {midpoint}, ||mid|| = {np.linalg.norm(midpoint):.2f}")
    print(f"   Midpoint in set? {np.linalg.norm(midpoint) <= r}")
    
    # 2. Halfspace
    print("\n2. Halfspace: {x : a'x <= b}")
    a = np.array([1, 2])
    b = 3
    x = np.array([1, 0])  # a'x = 1 <= 3 ✓
    y = np.array([0, 1])  # a'y = 2 <= 3 ✓
    
    print(f"   a = {a}, b = {b}")
    print(f"   x = {x}, a'x = {a @ x}")
    print(f"   y = {y}, a'y = {a @ y}")
    
    midpoint = 0.5 * x + 0.5 * y
    print(f"   midpoint = {midpoint}, a'mid = {a @ midpoint}")
    print(f"   Midpoint in halfspace? {a @ midpoint <= b}")
    
    # 3. Intersection preserves convexity
    print("\n3. Intersection of convex sets is convex")
    print("   Polyhedron = intersection of halfspaces")


def example_convex_functions():
    """Verify convexity of functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Convex Functions")
    print("=" * 60)
    
    # Check f(θx + (1-θ)y) <= θf(x) + (1-θ)f(y)
    
    functions = [
        ("x²", lambda x: x**2),
        ("e^x", lambda x: np.exp(x)),
        ("-log(x)", lambda x: -np.log(x)),
        ("|x|", lambda x: np.abs(x)),
    ]
    
    x, y = 1.0, 3.0
    theta = 0.4
    
    print(f"\nChecking convexity at x={x}, y={y}, θ={theta}")
    print(f"Convex combination: {theta}*{x} + {1-theta}*{y} = {theta*x + (1-theta)*y}")
    
    print(f"\n{'Function':<10} {'f(θx+(1-θ)y)':<15} {'θf(x)+(1-θ)f(y)':<18} {'Convex?':<8}")
    print("-" * 55)
    
    for name, f in functions:
        z = theta * x + (1 - theta) * y  # Convex combination
        left = f(z)  # f(convex comb)
        right = theta * f(x) + (1 - theta) * f(y)  # Convex comb of f values
        
        is_convex = left <= right + 1e-10
        print(f"{name:<10} {left:<15.6f} {right:<18.6f} {'✓' if is_convex else '✗':<8}")
    
    print("\nConvex: f(θx+(1-θ)y) <= θf(x)+(1-θ)f(y)")


def example_second_order_condition():
    """Check convexity via Hessian."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Second-Order Condition")
    print("=" * 60)
    
    print("\nConvex iff Hessian H(x) ≽ 0 (positive semidefinite)")
    
    # 1D case: second derivative >= 0
    print("\n1. f(x) = x² + 2x + 1")
    print("   f''(x) = 2 > 0 → Convex ✓")
    
    # 2D case
    print("\n2. f(x,y) = x² + xy + y²")
    print("   ∇f = [2x + y, x + 2y]")
    print("   H = [[2, 1], [1, 2]]")
    
    H = np.array([[2, 1], [1, 2]])
    eigenvalues = np.linalg.eigvalsh(H)
    print(f"   Eigenvalues of H: {eigenvalues}")
    print(f"   All eigenvalues >= 0? {all(eigenvalues >= 0)}")
    print("   → Convex ✓")
    
    # 3. Quadratic form
    print("\n3. f(x) = x'Ax + b'x + c")
    print("   Convex iff A ≽ 0")
    
    A = np.array([[4, 2], [2, 3]])
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"   A = [[4,2],[2,3]]")
    print(f"   Eigenvalues: {eigenvalues}")
    print(f"   A ≽ 0? {all(eigenvalues >= 0)} → Convex ✓")


def example_ml_loss_convexity():
    """Show ML losses are convex."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: ML Loss Functions Convexity")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate simple data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])
    
    # 1. MSE Loss
    print("\n1. MSE Loss: L(w) = ||y - Xw||²")
    print("   ∇²L(w) = 2X'X ≽ 0 → Convex ✓")
    
    # Hessian
    H_mse = 2 * X.T @ X
    print(f"   Hessian = 2X'X = {H_mse.flatten()}")
    print(f"   Eigenvalues: {np.linalg.eigvalsh(H_mse)}")
    
    # 2. Ridge Loss
    print("\n2. Ridge: L(w) = ||y - Xw||² + λ||w||²")
    lambda_reg = 1.0
    H_ridge = 2 * X.T @ X + 2 * lambda_reg * np.eye(1)
    print(f"   Hessian = 2X'X + 2λI")
    print(f"   Adding regularization makes Hessian more positive definite")
    print(f"   Eigenvalues: {np.linalg.eigvalsh(H_ridge)}")
    
    # 3. Logistic Loss
    print("\n3. Logistic Loss: L(w) = Σ log(1 + e^(-y*w'x))")
    print("   Hessian = Σ σ(z)(1-σ(z)) x x' ≽ 0")
    print("   σ(z)(1-σ(z)) >= 0 always → Convex ✓")


def example_operations_preserving_convexity():
    """Operations that preserve convexity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Operations Preserving Convexity")
    print("=" * 60)
    
    # 1. Non-negative sum
    print("\n1. Non-negative weighted sum")
    print("   f₁(x) = x², f₂(x) = e^x both convex")
    print("   g(x) = 2*x² + 3*e^x is convex ✓")
    
    # 2. Pointwise maximum
    print("\n2. Pointwise maximum")
    print("   f₁(x) = 2x - 1, f₂(x) = -x + 2 (both affine)")
    print("   max(f₁, f₂) is convex")
    
    x_range = np.linspace(-1, 4, 100)
    f1 = 2 * x_range - 1
    f2 = -x_range + 2
    f_max = np.maximum(f1, f2)
    
    print(f"   At x=0: f₁={f1[50]:.1f}, f₂={f2[50]:.1f}, max={f_max[50]:.1f}")
    print("   This is how hinge loss works!")
    
    # 3. Composition
    print("\n3. Composition rules")
    print("   g(x) = e^x convex and increasing")
    print("   h(x) = x² convex")
    print("   f(x) = g(h(x)) = e^(x²) is convex ✓")
    
    # Verify numerically
    def f_composed(x):
        return np.exp(x**2)
    
    x1, x2 = 0.5, 1.5
    theta = 0.3
    mid = theta * x1 + (1 - theta) * x2
    
    left = f_composed(mid)
    right = theta * f_composed(x1) + (1 - theta) * f_composed(x2)
    print(f"   f(θx₁+(1-θ)x₂) = {left:.4f}")
    print(f"   θf(x₁)+(1-θ)f(x₂) = {right:.4f}")
    print(f"   Convex? {left <= right + 1e-10} ✓")


def example_local_global_minimum():
    """Local = global for convex functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Local = Global Minimum")
    print("=" * 60)
    
    # Convex function: any local min is global
    print("\n1. Convex: f(x) = x² + 2x + 1 = (x+1)²")
    print("   f'(x) = 2x + 2 = 0 → x = -1")
    print("   f''(x) = 2 > 0 → local min is global min ✓")
    
    # Find numerically
    result = optimize.minimize_scalar(lambda x: x**2 + 2*x + 1)
    print(f"   Numerical: x* = {result.x:.4f}, f(x*) = {result.fun:.4f}")
    
    # Non-convex: may have multiple local minima
    print("\n2. Non-convex: g(x) = sin(x) + 0.1*x²")
    print("   Has multiple local minima!")
    
    # Find from different starting points
    print("\n   Starting from different points:")
    for x0 in [-5, 0, 5]:
        result = optimize.minimize(
            lambda x: np.sin(x[0]) + 0.1*x[0]**2, 
            [x0],
            method='BFGS'
        )
        print(f"   x₀={x0:>3}: x*={result.x[0]:.4f}, g(x*)={result.fun:.4f}")
    
    print("\n   Different starting points → different local minima")
    print("   For non-convex, must try multiple initializations!")


def example_kkt_conditions():
    """KKT conditions example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: KKT Conditions")
    print("=" * 60)
    
    print("""
Problem: min x² + y²
         s.t. x + y >= 1  (or equivalently: -x - y + 1 <= 0)
    """)
    
    print("Lagrangian: L = x² + y² + λ(-x - y + 1)")
    print("where λ >= 0\n")
    
    print("KKT Conditions:")
    print("1. Stationarity: ∇L = 0")
    print("   ∂L/∂x = 2x - λ = 0  →  x = λ/2")
    print("   ∂L/∂y = 2y - λ = 0  →  y = λ/2")
    
    print("\n2. Primal feasibility: x + y >= 1")
    print("3. Dual feasibility: λ >= 0")
    print("4. Complementary slackness: λ(1 - x - y) = 0")
    
    print("\nSolving:")
    print("From stationarity: x = y = λ/2")
    print("If λ = 0: x = y = 0, but violates x + y >= 1")
    print("So λ > 0, which means x + y = 1 (constraint active)")
    print("λ/2 + λ/2 = 1 → λ = 1")
    print("x* = y* = 0.5")
    
    # Verify numerically
    from scipy.optimize import minimize
    
    def objective(z):
        return z[0]**2 + z[1]**2
    
    constraint = {'type': 'ineq', 'fun': lambda z: z[0] + z[1] - 1}
    
    result = minimize(objective, [0, 0], constraints=constraint)
    print(f"\nNumerical solution: x*={result.x[0]:.4f}, y*={result.x[1]:.4f}")
    print(f"Optimal value: {result.fun:.4f}")


def example_duality():
    """Lagrangian duality example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Lagrangian Duality")
    print("=" * 60)
    
    print("""
Primal: min  x²
        s.t. x >= 2
        
Lagrangian: L(x, λ) = x² + λ(2 - x), λ >= 0
    """)
    
    print("Dual function d(λ):")
    print("  d(λ) = min_x [x² + λ(2 - x)]")
    print("  ∂L/∂x = 2x - λ = 0 → x = λ/2")
    print("  d(λ) = (λ/2)² + λ(2 - λ/2) = λ²/4 + 2λ - λ²/2 = 2λ - λ²/4")
    
    print("\nMaximizing dual:")
    print("  d'(λ) = 2 - λ/2 = 0 → λ* = 4")
    print("  d(λ*) = 2(4) - 16/4 = 8 - 4 = 4")
    
    print("\nPrimal solution:")
    print("  x* = 2 (constraint binding)")
    print("  f(x*) = 4")
    
    print("\nStrong duality holds: p* = d* = 4")
    print("Duality gap = 0")


def example_svm_duality():
    """SVM dual problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: SVM Dual Problem")
    print("=" * 60)
    
    print("""
Hard-margin SVM:

Primal: min  (1/2)||w||²
        s.t. yᵢ(w'xᵢ + b) >= 1

Dual:   max  Σαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼxᵢ'xⱼ
        s.t. αᵢ >= 0, Σαᵢyᵢ = 0
    """)
    
    # Simple 2D example
    np.random.seed(42)
    
    # Two separable classes
    X = np.array([[1, 1], [2, 1], [1, 2],  # Class +1
                  [-1, -1], [-2, -1], [-1, -2]])  # Class -1
    y = np.array([1, 1, 1, -1, -1, -1])
    
    print("Data points:")
    for i, (xi, yi) in enumerate(zip(X, y)):
        print(f"  x{i+1} = {xi}, y{i+1} = {yi:+d}")
    
    # Compute kernel matrix (linear kernel)
    K = X @ X.T
    print(f"\nKernel matrix K = X X':")
    print(K)
    
    # Solve dual using simple gradient ascent
    # Dual objective: Σα - 0.5 * α' diag(y) K diag(y) α
    
    from scipy.optimize import minimize
    
    def dual_objective(alpha):
        """Negative dual (for minimization)."""
        return -np.sum(alpha) + 0.5 * alpha @ (y[:, None] * K * y) @ alpha
    
    n = len(y)
    constraints = [
        {'type': 'eq', 'fun': lambda a: np.dot(a, y)},  # Σαy = 0
    ]
    bounds = [(0, None) for _ in range(n)]  # α >= 0
    
    result = minimize(dual_objective, np.zeros(n), 
                     method='SLSQP', bounds=bounds, constraints=constraints)
    
    alpha = result.x
    print(f"\nOptimal α: {np.round(alpha, 4)}")
    print(f"Support vectors (α > 0): {np.where(alpha > 0.01)[0]}")
    
    # Recover w from dual
    w = np.sum((alpha * y)[:, None] * X, axis=0)
    print(f"Recovered w = Σαᵢyᵢxᵢ = {np.round(w, 4)}")


def example_ridge_as_constrained():
    """Ridge as constrained optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Ridge Regression - Two Views")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n, p = 50, 3
    X = np.random.randn(n, p)
    true_w = np.array([2.0, -1.0, 0.5])
    y = X @ true_w + np.random.randn(n) * 0.5
    
    print("Ridge regression has two equivalent formulations:\n")
    
    # Formulation 1: Penalized
    print("1. Penalized (Lagrangian) form:")
    print("   min ||y - Xw||² + λ||w||²")
    
    lambda_val = 1.0
    w_penalized = np.linalg.solve(X.T @ X + lambda_val * np.eye(p), X.T @ y)
    print(f"   λ = {lambda_val}")
    print(f"   w* = {np.round(w_penalized, 4)}")
    
    # Formulation 2: Constrained
    print("\n2. Constrained (primal) form:")
    print("   min ||y - Xw||²  s.t. ||w||² <= t")
    
    # Find t that gives same solution
    t = np.sum(w_penalized**2)
    
    from scipy.optimize import minimize
    
    def mse(w):
        return np.sum((y - X @ w)**2)
    
    constraint = {'type': 'ineq', 'fun': lambda w: t - np.sum(w**2)}
    result = minimize(mse, np.zeros(p), constraints=constraint)
    
    w_constrained = result.x
    print(f"   t = {t:.4f}")
    print(f"   w* = {np.round(w_constrained, 4)}")
    
    print("\nBoth give same solution (by duality)!")
    print(f"Difference: {np.linalg.norm(w_penalized - w_constrained):.2e}")


def example_convexity_visualization():
    """Visualize convexity concepts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Convexity Visualization (Numerical)")
    print("=" * 60)
    
    # Generate points to check convexity
    x_range = np.linspace(-2, 2, 100)
    
    def check_convexity(f, f_name, x_range):
        """Check if function appears convex by sampling."""
        n_tests = 50
        np.random.seed(42)
        
        is_convex = True
        for _ in range(n_tests):
            # Random points
            i, j = np.random.choice(len(x_range), 2, replace=False)
            x1, x2 = x_range[i], x_range[j]
            theta = np.random.uniform(0, 1)
            
            # Convex combination
            x_mid = theta * x1 + (1 - theta) * x2
            
            left = f(x_mid)
            right = theta * f(x1) + (1 - theta) * f(x2)
            
            if left > right + 1e-10:
                is_convex = False
                break
        
        return is_convex
    
    functions = [
        ("x²", lambda x: x**2, True),
        ("x⁴", lambda x: x**4, True),
        ("e^x", lambda x: np.exp(x), True),
        ("|x|", lambda x: np.abs(x), True),
        ("x³", lambda x: x**3, False),
        ("sin(x)", lambda x: np.sin(x), False),
    ]
    
    print(f"{'Function':<12} {'Expected':<12} {'Verified':<12}")
    print("-" * 36)
    
    for name, f, expected in functions:
        verified = check_convexity(f, name, x_range)
        match = "✓" if verified == expected else "✗"
        print(f"{name:<12} {'Convex' if expected else 'Non-convex':<12} {'Convex' if verified else 'Non-convex':<12} {match}")


def example_condition_number():
    """Condition number and optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Condition Number and Optimization")
    print("=" * 60)
    
    print("Condition number affects convergence of gradient descent\n")
    
    # Well-conditioned quadratic
    A_good = np.array([[2, 0], [0, 2]])
    
    # Ill-conditioned quadratic
    A_bad = np.array([[10, 0], [0, 0.1]])
    
    print("f(x) = x'Ax (quadratic)")
    print(f"\nA₁ = diag(2, 2)")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(A_good)}")
    print(f"  Condition number κ = λmax/λmin = {np.linalg.cond(A_good):.1f}")
    
    print(f"\nA₂ = diag(10, 0.1)")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(A_bad)}")
    print(f"  Condition number κ = {np.linalg.cond(A_bad):.1f}")
    
    # Gradient descent simulation
    def gradient_descent(A, x0, lr, n_steps):
        x = x0.copy()
        trajectory = [x.copy()]
        for _ in range(n_steps):
            grad = 2 * A @ x
            x = x - lr * grad
            trajectory.append(x.copy())
        return np.array(trajectory)
    
    x0 = np.array([1.0, 1.0])
    
    # Well-conditioned: can use larger learning rate
    traj_good = gradient_descent(A_good, x0, lr=0.2, n_steps=20)
    
    # Ill-conditioned: must use smaller learning rate
    traj_bad = gradient_descent(A_bad, x0, lr=0.05, n_steps=20)
    
    print(f"\nGradient Descent Progress (distance to optimum):")
    print(f"{'Step':<6} {'Well-cond (κ=1)':<20} {'Ill-cond (κ=100)':<20}")
    print("-" * 46)
    
    for i in [0, 5, 10, 15, 20]:
        dist_good = np.linalg.norm(traj_good[i])
        dist_bad = np.linalg.norm(traj_bad[i])
        print(f"{i:<6} {dist_good:<20.6f} {dist_bad:<20.6f}")
    
    print("\nIll-conditioned problems converge slower!")
    print("Preconditioning/adaptive methods help.")


if __name__ == "__main__":
    example_convex_sets()
    example_convex_functions()
    example_second_order_condition()
    example_ml_loss_convexity()
    example_operations_preserving_convexity()
    example_local_global_minimum()
    example_kkt_conditions()
    example_duality()
    example_svm_duality()
    example_ridge_as_constrained()
    example_convexity_visualization()
    example_condition_number()
