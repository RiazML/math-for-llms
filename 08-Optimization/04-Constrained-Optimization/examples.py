"""
Constrained Optimization - Examples
===================================
Implementations of constrained optimization methods.
"""

import numpy as np
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


def example_lagrange_multipliers():
    """Basic Lagrange multipliers example."""
    print("=" * 60)
    print("EXAMPLE 1: Lagrange Multipliers")
    print("=" * 60)
    
    print("Minimize f(x,y) = x² + y²")
    print("Subject to: x + y = 1")
    
    print("\nSolution by Lagrange multipliers:")
    print("L = x² + y² + λ(x + y - 1)")
    print("\n∂L/∂x = 2x + λ = 0  →  x = -λ/2")
    print("∂L/∂y = 2y + λ = 0  →  y = -λ/2")
    print("x + y = 1  →  -λ = 1  →  λ = -1")
    
    x_opt, y_opt = 0.5, 0.5
    f_opt = x_opt**2 + y_opt**2
    
    print(f"\nOptimal: x* = {x_opt}, y* = {y_opt}")
    print(f"f* = {f_opt}")
    print(f"λ* = -1 (shadow price: rate of change of f* w.r.t. constraint)")
    
    # Verify with scipy
    result = optimize.minimize(
        lambda x: x[0]**2 + x[1]**2,
        [0, 0],
        constraints={'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}
    )
    print(f"\nScipy verification: {result.x}")


def example_kkt_conditions():
    """KKT conditions with inequality constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: KKT Conditions")
    print("=" * 60)
    
    print("Minimize f(x) = (x-2)²")
    print("Subject to: x ≤ 1")
    print("(equivalently: g(x) = x - 1 ≤ 0)")
    
    print("\nKKT Conditions:")
    print("1. Stationarity: 2(x-2) + μ = 0")
    print("2. Primal feasibility: x ≤ 1")
    print("3. Dual feasibility: μ ≥ 0")
    print("4. Complementary slackness: μ(x-1) = 0")
    
    print("\nCase Analysis:")
    print("\nCase 1: μ = 0 (constraint inactive)")
    print("  2(x-2) = 0 → x = 2")
    print("  But x = 2 violates x ≤ 1. ✗")
    
    print("\nCase 2: x = 1 (constraint active)")
    print("  2(1-2) + μ = 0 → μ = 2")
    print("  μ = 2 ≥ 0. ✓")
    
    x_opt = 1
    f_opt = (x_opt - 2)**2
    mu_opt = 2
    
    print(f"\nOptimal: x* = {x_opt}, f* = {f_opt}, μ* = {mu_opt}")
    print("Constraint is active (binding)")


def example_svm_dual():
    """SVM primal-dual formulation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: SVM Dual Problem")
    print("=" * 60)
    
    # Simple linearly separable data
    np.random.seed(42)
    
    # Class +1
    X_pos = np.array([[2, 2], [2, 3], [3, 2]])
    # Class -1
    X_neg = np.array([[0, 0], [0, 1], [1, 0]])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1, 1, 1, -1, -1, -1])
    
    print("Data:")
    print(f"  Class +1: {X_pos.tolist()}")
    print(f"  Class -1: {X_neg.tolist()}")
    
    print("\nSVM Primal:")
    print("  min_{w,b} (1/2)||w||²")
    print("  s.t. y_i(w'x_i + b) ≥ 1")
    
    print("\nSVM Dual:")
    print("  max_α Σαᵢ - (1/2)Σᵢⱼ αᵢαⱼyᵢyⱼxᵢ'xⱼ")
    print("  s.t. αᵢ ≥ 0, Σαᵢyᵢ = 0")
    
    # Solve dual using quadratic programming
    n = len(y)
    K = X @ X.T  # Kernel matrix (linear kernel)
    
    # Dual: max Σα - (1/2)α'Qα where Q_ij = y_i y_j x_i'x_j
    Q = np.outer(y, y) * K
    
    # Use scipy to solve
    from scipy.optimize import minimize
    
    def dual_objective(alpha):
        return 0.5 * alpha @ Q @ alpha - np.sum(alpha)
    
    def dual_grad(alpha):
        return Q @ alpha - 1
    
    constraints = [
        {'type': 'eq', 'fun': lambda a: a @ y},  # Σα_i y_i = 0
    ]
    bounds = [(0, None) for _ in range(n)]  # α_i ≥ 0
    
    result = minimize(dual_objective, np.zeros(n), jac=dual_grad,
                     constraints=constraints, bounds=bounds, method='SLSQP')
    
    alpha = result.x
    
    # Recover primal variables
    w = np.sum(alpha[:, None] * y[:, None] * X, axis=0)
    
    # Find support vectors (α > 0)
    sv_mask = alpha > 1e-5
    
    # Compute b from support vectors
    b = np.mean(y[sv_mask] - X[sv_mask] @ w)
    
    print(f"\nDual solution:")
    print(f"  α = {np.round(alpha, 4)}")
    print(f"\nSupport vectors at indices: {np.where(sv_mask)[0]}")
    
    print(f"\nRecovered primal:")
    print(f"  w = {np.round(w, 4)}")
    print(f"  b = {round(b, 4)}")


def example_quadratic_penalty():
    """Quadratic penalty method."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Quadratic Penalty Method")
    print("=" * 60)
    
    print("Minimize f(x,y) = x + y")
    print("Subject to: x² + y² = 1")
    
    def objective(x):
        return x[0] + x[1]
    
    def constraint(x):
        return x[0]**2 + x[1]**2 - 1
    
    def penalized_objective(x, rho):
        return objective(x) + rho * constraint(x)**2
    
    print("\nPenalty method: min f(x) + ρ × h(x)²")
    print("\n{'ρ':>10} {'x':>20} {'f(x)':>10} {'|h(x)|':>12}")
    print("-" * 55)
    
    x = np.array([0.0, 0.0])
    
    for rho in [1, 10, 100, 1000, 10000]:
        result = optimize.minimize(
            lambda z: penalized_objective(z, rho),
            x
        )
        x = result.x
        
        print(f"{rho:>10} {str(np.round(x, 6)):>20} {objective(x):>10.6f} "
              f"{abs(constraint(x)):>12.2e}")
    
    # True optimum
    x_true = np.array([-1/np.sqrt(2), -1/np.sqrt(2)])
    print(f"\nTrue optimum: {np.round(x_true, 6)}")
    print(f"As ρ → ∞, penalty solution → true optimum")


def example_barrier_method():
    """Log barrier method."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Log Barrier Method")
    print("=" * 60)
    
    print("Minimize f(x) = -log(x)")
    print("Subject to: x ≤ 1")
    print("(equivalently: g(x) = x - 1 ≤ 0)")
    
    def objective(x):
        return -np.log(x)
    
    def barrier(x, t):
        """Barrier: -log(-g(x)) = -log(1-x)"""
        if x >= 1:
            return np.inf
        return objective(x) - (1/t) * np.log(1 - x)
    
    print("\nBarrier method: min f(x) - (1/t)×log(-g(x))")
    print(f"\n{'t':>10} {'x*':>15} {'f(x*)':>15} {'1-x*':>15}")
    print("-" * 60)
    
    for t in [1, 10, 100, 1000]:
        # Analytical solution for this simple case
        # d/dx[-log(x) - (1/t)log(1-x)] = -1/x + (1/t)/(1-x) = 0
        # -1/x + (1/t)/(1-x) = 0
        # (1-x)/x = 1/t → x = t/(t+1)
        x_opt = t / (t + 1)
        
        print(f"{t:>10} {x_opt:>15.10f} {objective(x_opt):>15.10f} {1-x_opt:>15.10f}")
    
    print("\nAs t → ∞, x* → 1 (boundary)")
    print("True optimum: x* = 1, f* = 0")


def example_projected_gradient():
    """Projected gradient descent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Projected Gradient Descent")
    print("=" * 60)
    
    print("Minimize f(x) = (x-2)²")
    print("Subject to: x ∈ [0, 1]")
    
    def gradient(x):
        return 2 * (x - 2)
    
    def project(x, lo=0, hi=1):
        return np.clip(x, lo, hi)
    
    x = 0.5
    eta = 0.1
    
    print(f"\nProjected GD: x ← Π[x - η∇f(x)]")
    print(f"\n{'Step':>4} {'x':>15} {'f(x)':>15} {'∇f':>15}")
    print("-" * 55)
    
    for i in range(15):
        f_val = (x - 2)**2
        g = gradient(x)
        print(f"{i:>4} {x:>15.6f} {f_val:>15.6f} {g:>15.6f}")
        
        # Gradient step
        x_unconstrained = x - eta * g
        # Project
        x = project(x_unconstrained)
    
    print(f"\nOptimal: x* = 1 (boundary of [0,1])")
    print("Unconstrained optimum would be x = 2")


def example_projection_simplex():
    """Project onto probability simplex."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Projection onto Simplex")
    print("=" * 60)
    
    def project_simplex(v):
        """Project v onto probability simplex."""
        n = len(v)
        # Sort in descending order
        u = np.sort(v)[::-1]
        
        # Find threshold
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > cssv - 1)[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        
        return np.maximum(v - theta, 0)
    
    print("Simplex: {x : Σxᵢ = 1, xᵢ ≥ 0}")
    
    test_vectors = [
        np.array([2.0, 1.0, 0.0]),
        np.array([-1.0, 2.0, 3.0]),
        np.array([0.3, 0.3, 0.4])
    ]
    
    for v in test_vectors:
        p = project_simplex(v)
        print(f"\n  v = {np.round(v, 4)}")
        print(f"  Π(v) = {np.round(p, 4)}")
        print(f"  Sum: {np.sum(p):.6f}, All ≥ 0: {np.all(p >= -1e-10)}")


def example_proximal_lasso():
    """Proximal gradient for LASSO."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Proximal Gradient for LASSO")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n, d = 50, 10
    X = np.random.randn(n, d)
    true_w = np.array([3, -2, 0, 0, 0, 0, 0, 0, 1.5, 0])  # Sparse
    y = X @ true_w + 0.1 * np.random.randn(n)
    
    lam = 0.5
    
    print("LASSO: min (1/2)||y - Xw||² + λ||w||₁")
    print(f"λ = {lam}")
    print(f"True w (sparse): {true_w}")
    
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
    
    w = np.zeros(d)
    eta = 1 / np.linalg.norm(X.T @ X, 2)  # Step size
    
    print(f"\nProximal GD: w ← soft_λη(w - η X'(Xw - y))")
    
    for i in range(200):
        # Gradient of smooth part
        grad = X.T @ (X @ w - y)
        
        # Proximal step (soft thresholding)
        w = soft_threshold(w - eta * grad, eta * lam)
    
    print(f"\nEstimated w:")
    print(f"  {np.round(w, 4)}")
    
    print(f"\nNonzero entries:")
    print(f"  True: {np.nonzero(true_w)[0]}")
    print(f"  Estimated: {np.nonzero(np.abs(w) > 0.1)[0]}")


def example_admm_lasso():
    """ADMM for LASSO."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: ADMM for LASSO")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 50, 10
    X = np.random.randn(n, d)
    true_w = np.array([3, -2, 0, 0, 0, 0, 0, 0, 1.5, 0])
    y = X @ true_w + 0.1 * np.random.randn(n)
    
    lam = 0.5
    rho = 1.0
    
    print("LASSO via ADMM:")
    print("  Reformulate: min (1/2)||y-Xw||² + λ||z||₁")
    print("  s.t. w = z")
    
    def soft_threshold(v, threshold):
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)
    
    # Precompute
    XTX = X.T @ X
    XTy = X.T @ y
    factor = np.linalg.inv(XTX + rho * np.eye(d))
    
    # Initialize
    w = np.zeros(d)
    z = np.zeros(d)
    u = np.zeros(d)  # Scaled dual variable
    
    print(f"\n{'Iter':>4} {'||w - z||':>12} {'||w||₁':>12} {'Loss':>12}")
    print("-" * 45)
    
    for i in range(100):
        # w-update: (X'X + ρI)w = X'y + ρ(z - u)
        w = factor @ (XTy + rho * (z - u))
        
        # z-update: soft thresholding
        z = soft_threshold(w + u, lam / rho)
        
        # u-update
        u = u + w - z
        
        primal_residual = np.linalg.norm(w - z)
        loss = 0.5 * np.linalg.norm(y - X @ w)**2 + lam * np.linalg.norm(z, 1)
        
        if i % 20 == 0:
            print(f"{i:>4} {primal_residual:>12.6f} {np.linalg.norm(z, 1):>12.4f} {loss:>12.4f}")
    
    print(f"\nFinal w (ADMM): {np.round(z, 4)}")
    print(f"True w:         {true_w}")


def example_quadratic_with_constraints():
    """Quadratic programming example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Quadratic Programming")
    print("=" * 60)
    
    print("Minimize f(x) = x² + y² - xy")
    print("Subject to:")
    print("  x + y ≥ 1")
    print("  x ≥ 0")
    print("  y ≥ 0")
    
    def objective(x):
        return x[0]**2 + x[1]**2 - x[0]*x[1]
    
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},  # x + y ≥ 1
    ]
    bounds = [(0, None), (0, None)]  # x, y ≥ 0
    
    result = optimize.minimize(
        objective, [1, 1],
        constraints=constraints,
        bounds=bounds,
        method='SLSQP'
    )
    
    x_opt = result.x
    
    print(f"\nOptimal solution: x = {x_opt[0]:.6f}, y = {x_opt[1]:.6f}")
    print(f"Optimal value: f* = {result.fun:.6f}")
    
    # Check constraints
    print(f"\nConstraint check:")
    print(f"  x + y = {x_opt[0] + x_opt[1]:.6f} (≥ 1)")
    print(f"  x = {x_opt[0]:.6f} (≥ 0)")
    print(f"  y = {x_opt[1]:.6f} (≥ 0)")


def example_portfolio_optimization():
    """Portfolio optimization with constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Portfolio Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 4 assets
    expected_returns = np.array([0.12, 0.10, 0.07, 0.03])
    cov_matrix = np.array([
        [0.10, 0.01, 0.02, 0.00],
        [0.01, 0.08, 0.01, 0.00],
        [0.02, 0.01, 0.06, 0.01],
        [0.00, 0.00, 0.01, 0.02]
    ])
    
    print("Mean-Variance Optimization:")
    print("  min (1/2) w'Σw  (minimize variance)")
    print("  s.t. μ'w ≥ r_target  (minimum return)")
    print("       Σwᵢ = 1  (fully invested)")
    print("       wᵢ ≥ 0  (no short selling)")
    
    def portfolio_variance(w):
        return 0.5 * w @ cov_matrix @ w
    
    target_return = 0.08
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum = 1
        {'type': 'ineq', 'fun': lambda w: expected_returns @ w - target_return}  # Return ≥ target
    ]
    bounds = [(0, 1) for _ in range(4)]  # No short selling
    
    result = optimize.minimize(
        portfolio_variance,
        np.array([0.25, 0.25, 0.25, 0.25]),
        constraints=constraints,
        bounds=bounds,
        method='SLSQP'
    )
    
    w_opt = result.x
    
    print(f"\nTarget return: {target_return:.1%}")
    print(f"\nOptimal weights:")
    for i, w in enumerate(w_opt):
        print(f"  Asset {i+1}: {w:.2%}")
    
    print(f"\nPortfolio statistics:")
    print(f"  Expected return: {expected_returns @ w_opt:.2%}")
    print(f"  Variance: {w_opt @ cov_matrix @ w_opt:.4f}")
    print(f"  Std dev: {np.sqrt(w_opt @ cov_matrix @ w_opt):.2%}")


def example_constrained_regression():
    """Regression with constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Constrained Regression")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n, d = 100, 5
    X = np.random.randn(n, d)
    true_w = np.array([0.3, 0.4, 0.3, 0.0, 0.0])
    y = X @ true_w + 0.1 * np.random.randn(n)
    
    print("Non-negative least squares:")
    print("  min ||y - Xw||²")
    print("  s.t. w ≥ 0")
    
    def loss(w):
        return np.sum((y - X @ w)**2)
    
    bounds = [(0, None) for _ in range(d)]
    
    # Unconstrained
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Constrained
    result = optimize.minimize(
        loss,
        np.zeros(d),
        bounds=bounds,
        method='L-BFGS-B'
    )
    w_nnls = result.x
    
    print(f"\nTrue w:        {np.round(true_w, 4)}")
    print(f"OLS (unconstrained): {np.round(w_ols, 4)}")
    print(f"NNLS (w ≥ 0): {np.round(w_nnls, 4)}")
    
    print(f"\nOLS loss: {loss(w_ols):.4f}")
    print(f"NNLS loss: {loss(w_nnls):.4f}")


if __name__ == "__main__":
    example_lagrange_multipliers()
    example_kkt_conditions()
    example_svm_dual()
    example_quadratic_penalty()
    example_barrier_method()
    example_projected_gradient()
    example_projection_simplex()
    example_proximal_lasso()
    example_admm_lasso()
    example_quadratic_with_constraints()
    example_portfolio_optimization()
    example_constrained_regression()
