"""
Second-Order Optimization Methods - Examples
============================================
Implementations of Newton, quasi-Newton, and related methods.
"""

import numpy as np
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


def example_newton_quadratic():
    """Newton's method on quadratic (exact in one step)."""
    print("=" * 60)
    print("EXAMPLE 1: Newton's Method on Quadratic")
    print("=" * 60)
    
    # f(x) = 0.5 * x'Ax - b'x
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    
    def grad(x):
        return A @ x - b
    
    def hessian(x):
        return A
    
    x_optimal = np.linalg.solve(A, b)
    
    # Newton's method
    x = np.array([0.0, 0.0])
    
    print(f"f(x) = 0.5 x'Ax - b'x")
    print(f"Optimal: {x_optimal}")
    print(f"\nNewton's method from x₀ = {x}:")
    
    print(f"\n{'Step':>4} {'x':>20} {'f(x)':>12} {'||grad||':>12}")
    print("-" * 55)
    
    for i in range(5):
        g = grad(x)
        H = hessian(x)
        print(f"{i:>4} {str(np.round(x, 6)):>20} {f(x):>12.6f} {np.linalg.norm(g):>12.6f}")
        
        if np.linalg.norm(g) < 1e-10:
            break
        
        # Newton step
        x = x - np.linalg.solve(H, g)
    
    print(f"\nNewton converges in 1 step for quadratics!")


def example_newton_nonquadratic():
    """Newton's method on non-quadratic function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Newton's Method - Non-quadratic")
    print("=" * 60)
    
    # Rosenbrock function
    def f(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    
    def grad(x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dy = 200*(x[1] - x[0]**2)
        return np.array([dx, dy])
    
    def hessian(x):
        h11 = 2 - 400*x[1] + 1200*x[0]**2
        h12 = -400*x[0]
        h22 = 200
        return np.array([[h11, h12], [h12, h22]])
    
    x = np.array([-1.0, 1.0])
    
    print(f"Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²")
    print(f"Optimal: (1, 1)")
    print(f"Initial: {x}")
    
    print(f"\n{'Step':>4} {'x':>20} {'f(x)':>15}")
    print("-" * 45)
    
    for i in range(20):
        print(f"{i:>4} {str(np.round(x, 6)):>20} {f(x):>15.6f}")
        
        g = grad(x)
        if np.linalg.norm(g) < 1e-8:
            print("Converged!")
            break
        
        H = hessian(x)
        
        # Check if Hessian is positive definite
        eigvals = np.linalg.eigvalsh(H)
        if np.min(eigvals) <= 0:
            # Use gradient descent step instead
            x = x - 0.001 * g
        else:
            # Newton step with line search
            delta = np.linalg.solve(H, g)
            
            # Simple backtracking line search
            alpha = 1.0
            while f(x - alpha * delta) > f(x) - 0.1 * alpha * g @ delta:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
            
            x = x - alpha * delta
    
    print(f"\nNewton with line search converges faster than GD")


def example_gd_vs_newton():
    """Compare GD and Newton convergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: GD vs Newton Convergence")
    print("=" * 60)
    
    # Ill-conditioned quadratic
    A = np.diag([100, 1])  # Condition number = 100
    b = np.array([1, 1])
    
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    
    def grad(x):
        return A @ x - b
    
    x_optimal = np.linalg.solve(A, b)
    
    # GD
    x_gd = np.array([0.0, 0.0])
    eta = 0.01  # Safe learning rate
    
    gd_errors = []
    for _ in range(100):
        gd_errors.append(np.linalg.norm(x_gd - x_optimal))
        x_gd = x_gd - eta * grad(x_gd)
    
    # Newton
    x_newton = np.array([0.0, 0.0])
    
    newton_errors = []
    for _ in range(100):
        newton_errors.append(np.linalg.norm(x_newton - x_optimal))
        g = grad(x_newton)
        if np.linalg.norm(g) < 1e-12:
            newton_errors.extend([newton_errors[-1]] * (100 - len(newton_errors)))
            break
        x_newton = x_newton - np.linalg.solve(A, g)
    
    print(f"Ill-conditioned quadratic: κ = {np.linalg.cond(A)}")
    print(f"Optimal: {x_optimal}")
    
    print(f"\n{'Step':>6} {'GD ||x-x*||':>15} {'Newton ||x-x*||':>18}")
    print("-" * 45)
    
    for i in [0, 1, 2, 5, 10, 50, 99]:
        print(f"{i:>6} {gd_errors[i]:>15.6e} {newton_errors[min(i, len(newton_errors)-1)]:>18.6e}")
    
    print(f"\nNewton: 1 step. GD: still converging at step 100")


def example_gauss_newton():
    """Gauss-Newton for nonlinear least squares."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Gauss-Newton for Curve Fitting")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True model: y = a * exp(b * x)
    true_a, true_b = 2.5, -0.5
    
    # Generate data
    x_data = np.linspace(0, 4, 20)
    y_data = true_a * np.exp(true_b * x_data) + np.random.randn(20) * 0.1
    
    def residuals(params):
        a, b = params
        return y_data - a * np.exp(b * x_data)
    
    def jacobian(params):
        a, b = params
        J = np.zeros((len(x_data), 2))
        J[:, 0] = -np.exp(b * x_data)  # dr/da
        J[:, 1] = -a * x_data * np.exp(b * x_data)  # dr/db
        return J
    
    # Gauss-Newton
    params = np.array([1.0, -0.1])  # Initial guess
    
    print(f"Model: y = a * exp(b * x)")
    print(f"True: a = {true_a}, b = {true_b}")
    print(f"Initial: a = {params[0]}, b = {params[1]}")
    
    print(f"\n{'Step':>4} {'a':>10} {'b':>10} {'||r||²':>15}")
    print("-" * 45)
    
    for i in range(10):
        r = residuals(params)
        J = jacobian(params)
        loss = 0.5 * np.sum(r**2)
        
        print(f"{i:>4} {params[0]:>10.4f} {params[1]:>10.4f} {loss:>15.6f}")
        
        if loss < 1e-6:
            break
        
        # Gauss-Newton step: delta = (J'J)^{-1} J' r
        delta = np.linalg.solve(J.T @ J, J.T @ r)
        params = params - delta
    
    print(f"\nFinal: a = {params[0]:.4f}, b = {params[1]:.4f}")


def example_levenberg_marquardt():
    """Levenberg-Marquardt algorithm."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Levenberg-Marquardt")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Same nonlinear least squares
    true_a, true_b = 2.5, -0.5
    x_data = np.linspace(0, 4, 20)
    y_data = true_a * np.exp(true_b * x_data) + np.random.randn(20) * 0.1
    
    def residuals(params):
        a, b = params
        return y_data - a * np.exp(b * x_data)
    
    def jacobian(params):
        a, b = params
        J = np.zeros((len(x_data), 2))
        J[:, 0] = -np.exp(b * x_data)
        J[:, 1] = -a * x_data * np.exp(b * x_data)
        return J
    
    # LM algorithm
    params = np.array([1.0, -0.1])
    lam = 0.01  # Damping parameter
    
    print(f"Levenberg-Marquardt with adaptive λ")
    print(f"\n{'Step':>4} {'a':>10} {'b':>10} {'||r||²':>12} {'λ':>10}")
    print("-" * 55)
    
    for i in range(15):
        r = residuals(params)
        J = jacobian(params)
        loss = 0.5 * np.sum(r**2)
        
        print(f"{i:>4} {params[0]:>10.4f} {params[1]:>10.4f} {loss:>12.6f} {lam:>10.4f}")
        
        if loss < 1e-6:
            break
        
        # LM step: delta = (J'J + λI)^{-1} J' r
        JTJ = J.T @ J
        delta = np.linalg.solve(JTJ + lam * np.eye(2), J.T @ r)
        
        new_params = params - delta
        new_loss = 0.5 * np.sum(residuals(new_params)**2)
        
        # Adaptive damping
        if new_loss < loss:
            params = new_params
            lam *= 0.5  # Decrease λ (more Newton-like)
        else:
            lam *= 2.0  # Increase λ (more GD-like)
    
    print(f"\nFinal: a = {params[0]:.4f}, b = {params[1]:.4f}")


def example_bfgs_update():
    """Demonstrate BFGS Hessian approximation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: BFGS Update")
    print("=" * 60)
    
    # Quadratic function
    A_true = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    
    def f(x):
        return 0.5 * x @ A_true @ x - b @ x
    
    def grad(x):
        return A_true @ x - b
    
    # BFGS
    x = np.array([0.0, 0.0])
    H = np.eye(2)  # Initial inverse Hessian approximation
    
    print(f"True Hessian:\n{A_true}")
    print(f"\nBFGS builds inverse Hessian approximation:")
    
    print(f"\n{'Step':>4} {'||H^{-1} - A^{-1}||':>20} {'f(x)':>12}")
    print("-" * 45)
    
    A_inv_true = np.linalg.inv(A_true)
    
    for i in range(10):
        error = np.linalg.norm(H - A_inv_true)
        print(f"{i:>4} {error:>20.6f} {f(x):>12.6f}")
        
        g = grad(x)
        if np.linalg.norm(g) < 1e-10:
            break
        
        # Search direction
        p = -H @ g
        
        # Line search (exact for quadratic)
        alpha = - (g @ p) / (p @ A_true @ p)
        
        # Update x
        s = alpha * p
        x_new = x + s
        
        # Gradient difference
        y = grad(x_new) - g
        
        # BFGS update for inverse Hessian
        rho = 1.0 / (y @ s)
        I = np.eye(2)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        x = x_new
    
    print(f"\nFinal H^{{-1}} approximation:\n{np.round(H, 4)}")
    print(f"\nTrue A^{{-1}}:\n{np.round(A_inv_true, 4)}")


def example_lbfgs():
    """L-BFGS implementation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: L-BFGS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Larger problem
    d = 50
    A = np.random.randn(d, d)
    A = A.T @ A + 0.1 * np.eye(d)  # Make positive definite
    b = np.random.randn(d)
    
    def f(x):
        return 0.5 * x @ A @ x - b @ x
    
    def grad(x):
        return A @ x - b
    
    x_optimal = np.linalg.solve(A, b)
    
    # L-BFGS
    def lbfgs_two_loop(g, s_list, y_list):
        """Two-loop recursion for L-BFGS."""
        q = g.copy()
        m = len(s_list)
        
        if m == 0:
            return g
        
        alpha = np.zeros(m)
        rho = np.zeros(m)
        
        for i in range(m - 1, -1, -1):
            rho[i] = 1.0 / (y_list[i] @ s_list[i])
            alpha[i] = rho[i] * (s_list[i] @ q)
            q = q - alpha[i] * y_list[i]
        
        # Initial Hessian approximation
        gamma = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
        r = gamma * q
        
        for i in range(m):
            beta = rho[i] * (y_list[i] @ r)
            r = r + s_list[i] * (alpha[i] - beta)
        
        return r
    
    x = np.zeros(d)
    m = 10  # Memory
    s_list, y_list = [], []
    
    print(f"Problem dimension: d = {d}")
    print(f"L-BFGS memory: m = {m}")
    print(f"\n{'Step':>4} {'||x - x*||':>15} {'f(x)':>15} {'||grad||':>15}")
    print("-" * 55)
    
    for i in range(50):
        g = grad(x)
        error = np.linalg.norm(x - x_optimal)
        
        if i % 10 == 0:
            print(f"{i:>4} {error:>15.6e} {f(x):>15.6f} {np.linalg.norm(g):>15.6e}")
        
        if np.linalg.norm(g) < 1e-10:
            break
        
        # Compute direction
        if len(s_list) == 0:
            p = -g
        else:
            p = -lbfgs_two_loop(g, s_list, y_list)
        
        # Line search (simple backtracking)
        alpha = 1.0
        c1 = 1e-4
        while f(x + alpha * p) > f(x) + c1 * alpha * (g @ p):
            alpha *= 0.5
        
        s = alpha * p
        x_new = x + s
        y = grad(x_new) - g
        
        # Store
        if len(s_list) >= m:
            s_list.pop(0)
            y_list.pop(0)
        s_list.append(s)
        y_list.append(y)
        
        x = x_new
    
    print(f"\nL-BFGS: O(md) storage instead of O(d²)")


def example_scipy_optimizers():
    """Compare scipy optimizers."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Scipy Optimizers Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Rosenbrock
    def rosenbrock(x):
        return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    
    def rosenbrock_grad(x):
        grad = np.zeros_like(x)
        grad[:-1] = -400*x[:-1]*(x[1:] - x[:-1]**2) - 2*(1 - x[:-1])
        grad[1:] += 200*(x[1:] - x[:-1]**2)
        return grad
    
    x0 = np.zeros(10)
    
    methods = ['CG', 'BFGS', 'L-BFGS-B', 'Newton-CG']
    
    print(f"Rosenbrock function, d = {len(x0)}")
    print(f"Optimal: all ones")
    
    print(f"\n{'Method':>12} {'Iterations':>12} {'f(x*)':>15} {'||x* - 1||':>12}")
    print("-" * 55)
    
    for method in methods:
        result = optimize.minimize(
            rosenbrock, x0, 
            method=method,
            jac=rosenbrock_grad if method != 'L-BFGS-B' else None,
            options={'maxiter': 1000}
        )
        
        dist = np.linalg.norm(result.x - 1)
        print(f"{method:>12} {result.nit:>12} {result.fun:>15.6e} {dist:>12.6e}")


def example_hessian_free():
    """Hessian-free optimization (conjugate gradient)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Hessian-Free Optimization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Large quadratic
    d = 100
    A = np.random.randn(d, d)
    A = A.T @ A + 0.1 * np.eye(d)
    b = np.random.randn(d)
    
    def hessian_vector_product(x, v):
        """Compute Hv without forming H explicitly."""
        return A @ v  # In practice, use automatic differentiation
    
    def grad(x):
        return A @ x - b
    
    x_optimal = np.linalg.solve(A, b)
    
    # Hessian-free Newton (using CG to solve H*delta = g)
    x = np.zeros(d)
    
    print(f"Hessian-free Newton on d={d} problem")
    print("Uses CG to solve Newton system without forming Hessian")
    
    print(f"\n{'Outer':>6} {'CG iters':>10} {'||x-x*||':>15}")
    print("-" * 35)
    
    for outer in range(10):
        g = grad(x)
        error = np.linalg.norm(x - x_optimal)
        
        if error < 1e-10:
            break
        
        # Solve H*delta = g using CG
        delta = np.zeros(d)
        r = g.copy()
        p = -r.copy()
        
        for cg_iter in range(min(50, d)):
            Hp = hessian_vector_product(x, p)
            alpha = (r @ r) / (p @ Hp)
            delta = delta + alpha * p
            r_new = r + alpha * Hp
            
            if np.linalg.norm(r_new) < 1e-6 * np.linalg.norm(g):
                break
            
            beta = (r_new @ r_new) / (r @ r)
            p = -r_new + beta * p
            r = r_new
        
        print(f"{outer:>6} {cg_iter+1:>10} {error:>15.6e}")
        
        x = x - delta
    
    print(f"\nHessian-free: O(d) per CG iteration")


def example_natural_gradient():
    """Natural gradient concept."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Natural Gradient")
    print("=" * 60)
    
    print("""
Natural Gradient uses Fisher Information Matrix:
    
    θ_{t+1} = θ_t - η F^{-1} ∇L(θ)
    
where F = E[∇log p(x|θ) ∇log p(x|θ)']

For Gaussian with mean μ, variance σ²:
    F = diag(1/σ², 2/σ⁴)
    
Standard gradient treats μ and σ² equally,
Natural gradient accounts for different curvatures.
""")
    
    # Example: fitting Gaussian
    np.random.seed(42)
    data = np.random.normal(5.0, 2.0, 100)  # True: μ=5, σ=2
    
    def neg_log_likelihood(params):
        mu, log_var = params
        var = np.exp(log_var)
        return 0.5 * (np.log(var) + ((data - mu)**2 / var)).sum()
    
    def grad_nll(params):
        mu, log_var = params
        var = np.exp(log_var)
        grad_mu = -np.sum(data - mu) / var
        grad_log_var = 0.5 * (len(data) - np.sum((data - mu)**2) / var)
        return np.array([grad_mu, grad_log_var])
    
    def fisher_matrix(params):
        mu, log_var = params
        var = np.exp(log_var)
        return np.diag([len(data) / var, 0.5 * len(data)])
    
    # Standard gradient descent
    params_gd = np.array([0.0, 0.0])  # [mu, log_var]
    eta = 0.01
    
    for _ in range(100):
        params_gd = params_gd - eta * grad_nll(params_gd)
    
    # Natural gradient descent
    params_ng = np.array([0.0, 0.0])
    eta_ng = 0.1
    
    for _ in range(100):
        g = grad_nll(params_ng)
        F = fisher_matrix(params_ng)
        params_ng = params_ng - eta_ng * np.linalg.solve(F, g)
    
    print(f"Fitting Gaussian to data (true: μ=5, σ²=4)")
    print(f"\nStandard GD: μ = {params_gd[0]:.4f}, σ² = {np.exp(params_gd[1]):.4f}")
    print(f"Natural GD:  μ = {params_ng[0]:.4f}, σ² = {np.exp(params_ng[1]):.4f}")


def example_condition_number():
    """Effect of condition number on methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Condition Number Effects")
    print("=" * 60)
    
    def run_optimization(kappa, n_steps=50):
        """Run GD and Newton on ill-conditioned problem."""
        A = np.diag([kappa, 1])
        b = np.array([1, 1])
        
        def grad(x):
            return A @ x - b
        
        x_opt = np.linalg.solve(A, b)
        
        # GD
        x_gd = np.zeros(2)
        eta = 1.0 / kappa  # Safe learning rate
        
        for _ in range(n_steps):
            x_gd = x_gd - eta * grad(x_gd)
        
        # Newton
        x_newton = np.zeros(2)
        for _ in range(min(5, n_steps)):
            x_newton = x_newton - np.linalg.solve(A, grad(x_newton))
        
        return np.linalg.norm(x_gd - x_opt), np.linalg.norm(x_newton - x_opt)
    
    print(f"{'κ':>10} {'GD error (50 steps)':>25} {'Newton error (5 steps)':>25}")
    print("-" * 65)
    
    for kappa in [1, 10, 100, 1000, 10000]:
        gd_err, newton_err = run_optimization(kappa)
        print(f"{kappa:>10} {gd_err:>25.6e} {newton_err:>25.6e}")
    
    print("\nNewton is invariant to condition number!")
    print("GD convergence degrades with condition number")


def example_trust_region():
    """Trust region method concept."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Trust Region Method")
    print("=" * 60)
    
    def f(x):
        return (x[0] - 1)**2 + 10*(x[1] - x[0]**2)**2
    
    def grad(x):
        g0 = 2*(x[0] - 1) - 40*x[0]*(x[1] - x[0]**2)
        g1 = 20*(x[1] - x[0]**2)
        return np.array([g0, g1])
    
    def hessian(x):
        h00 = 2 - 40*x[1] + 120*x[0]**2
        h01 = -40*x[0]
        h11 = 20
        return np.array([[h00, h01], [h01, h11]])
    
    x = np.array([-1.0, 1.0])
    Delta = 1.0  # Trust region radius
    
    print(f"Trust region method")
    print(f"Initial radius: Δ = {Delta}")
    
    print(f"\n{'Step':>4} {'x':>20} {'f(x)':>12} {'Δ':>8}")
    print("-" * 50)
    
    for i in range(20):
        print(f"{i:>4} {str(np.round(x, 4)):>20} {f(x):>12.6f} {Delta:>8.4f}")
        
        g = grad(x)
        H = hessian(x)
        
        if np.linalg.norm(g) < 1e-6:
            break
        
        # Solve trust region subproblem (simplified: just clip Newton step)
        try:
            p_newton = -np.linalg.solve(H, g)
        except:
            p_newton = -g
        
        if np.linalg.norm(p_newton) > Delta:
            p = Delta * p_newton / np.linalg.norm(p_newton)
        else:
            p = p_newton
        
        # Evaluate ratio of actual vs predicted reduction
        actual = f(x) - f(x + p)
        predicted = -(g @ p + 0.5 * p @ H @ p)
        
        if predicted > 0:
            rho = actual / predicted
        else:
            rho = 0
        
        # Update trust region
        if rho > 0.75 and np.linalg.norm(p) > 0.9 * Delta:
            Delta = min(2 * Delta, 10)
        elif rho < 0.25:
            Delta = 0.5 * Delta
        
        # Accept step if improvement
        if rho > 0:
            x = x + p
    
    print(f"\nFinal: {x}, f = {f(x):.6f}")


if __name__ == "__main__":
    example_newton_quadratic()
    example_newton_nonquadratic()
    example_gd_vs_newton()
    example_gauss_newton()
    example_levenberg_marquardt()
    example_bfgs_update()
    example_lbfgs()
    example_scipy_optimizers()
    example_hessian_free()
    example_natural_gradient()
    example_condition_number()
    example_trust_region()
