"""
Numerical Optimization - Exercises
==================================
Practice implementing optimization algorithms.
"""

import numpy as np


def exercise_1_gradient_descent():
    """
    EXERCISE 1: Implement Gradient Descent with Line Search
    =======================================================
    
    Implement GD with backtracking line search (Armijo condition).
    
    Tasks:
    a) Implement backtracking_line_search(f, grad_f, x, p, c=0.1, rho=0.5)
    b) Integrate into gradient descent
    c) Compare with fixed learning rate
    """
    print("=" * 60)
    print("EXERCISE 1: GD with Line Search")
    print("=" * 60)
    
    # YOUR CODE HERE
    def backtracking_line_search(f, grad_f, x, p, c=0.1, rho=0.5):
        """
        Armijo backtracking line search.
        
        Find α such that f(x + αp) ≤ f(x) + c·α·∇f(x)ᵀp
        
        Start with α = 1, multiply by ρ until condition holds.
        """
        # TODO: Implement
        pass
    
    def gd_with_line_search(f, grad_f, x0, max_iter=100, tol=1e-6):
        """Gradient descent with backtracking line search."""
        # TODO: Implement
        pass


def exercise_1_solution():
    """Solution for Exercise 1."""
    print("=" * 60)
    print("SOLUTION 1: GD with Line Search")
    print("=" * 60)
    
    def backtracking_line_search(f, grad_f, x, p, c=0.1, rho=0.5, max_iter=50):
        alpha = 1.0
        fx = f(x)
        gx = grad_f(x)
        
        for _ in range(max_iter):
            if f(x + alpha * p) <= fx + c * alpha * (gx @ p):
                return alpha
            alpha *= rho
        
        return alpha
    
    def gd_with_line_search(f, grad_f, x0, max_iter=100, tol=1e-6):
        x = x0.copy()
        history = [(x.copy(), f(x))]
        
        for i in range(max_iter):
            g = grad_f(x)
            if np.linalg.norm(g) < tol:
                break
            
            p = -g  # Descent direction
            alpha = backtracking_line_search(f, grad_f, x, p)
            x = x + alpha * p
            history.append((x.copy(), f(x)))
        
        return x, history
    
    # Test on Rosenbrock
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
    
    x0 = np.array([-1.0, 1.0])
    x_opt, history = gd_with_line_search(rosenbrock, rosenbrock_grad, x0, max_iter=1000)
    
    print(f"Starting: {x0}")
    print(f"Found: {x_opt}")
    print(f"Steps: {len(history)}")
    print(f"Final f(x): {rosenbrock(x_opt):.6f}")


def exercise_2_momentum():
    """
    EXERCISE 2: Compare Momentum Methods
    ====================================
    
    Implement and compare different momentum variants.
    
    Tasks:
    a) Implement classical momentum
    b) Implement Nesterov momentum
    c) Compare on ill-conditioned quadratic
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Momentum Methods")
    print("=" * 60)
    
    # YOUR CODE HERE
    def classical_momentum(grad_f, x0, alpha, beta, max_iter, tol):
        """
        v_t = β·v_{t-1} + ∇f(x_t)
        x_{t+1} = x_t - α·v_t
        """
        # TODO: Implement
        pass
    
    def nesterov_momentum(grad_f, x0, alpha, beta, max_iter, tol):
        """
        v_t = β·v_{t-1} + ∇f(x_t - α·β·v_{t-1})
        x_{t+1} = x_t - α·v_t
        """
        # TODO: Implement
        pass


def exercise_2_solution():
    """Solution for Exercise 2."""
    print("\n" + "=" * 60)
    print("SOLUTION 2: Momentum Methods")
    print("=" * 60)
    
    def classical_momentum(grad_f, x0, alpha, beta, max_iter, tol):
        x = x0.copy()
        v = np.zeros_like(x)
        n_iter = 0
        
        for i in range(max_iter):
            g = grad_f(x)
            if np.linalg.norm(g) < tol:
                n_iter = i
                break
            v = beta * v + g
            x = x - alpha * v
            n_iter = i + 1
        
        return x, n_iter
    
    def nesterov_momentum(grad_f, x0, alpha, beta, max_iter, tol):
        x = x0.copy()
        v = np.zeros_like(x)
        n_iter = 0
        
        for i in range(max_iter):
            g = grad_f(x - alpha * beta * v)  # Look-ahead
            if np.linalg.norm(g) < tol:
                n_iter = i
                break
            v = beta * v + g
            x = x - alpha * v
            n_iter = i + 1
        
        return x, n_iter
    
    def basic_gd(grad_f, x0, alpha, max_iter, tol):
        x = x0.copy()
        n_iter = 0
        
        for i in range(max_iter):
            g = grad_f(x)
            if np.linalg.norm(g) < tol:
                n_iter = i
                break
            x = x - alpha * g
            n_iter = i + 1
        
        return x, n_iter
    
    # Ill-conditioned quadratic
    A = np.diag([100, 1])  # Condition number = 100
    b = np.array([1, 1])
    
    grad_f = lambda x: A @ x - b
    x_opt = np.linalg.solve(A, b)
    
    x0 = np.array([0.0, 0.0])
    alpha = 0.01
    beta = 0.9
    
    x_gd, n_gd = basic_gd(grad_f, x0, alpha, 1000, 1e-6)
    x_cm, n_cm = classical_momentum(grad_f, x0, alpha, beta, 1000, 1e-6)
    x_nag, n_nag = nesterov_momentum(grad_f, x0, alpha, beta, 1000, 1e-6)
    
    print(f"Ill-conditioned quadratic (κ = 100):")
    print(f"  Optimal: {x_opt}")
    print(f"\n{'Method':<20} {'Iterations':<15} {'Error':<15}")
    print("-" * 50)
    print(f"{'Basic GD':<20} {n_gd:<15} {np.linalg.norm(x_gd - x_opt):.2e}")
    print(f"{'Classical Momentum':<20} {n_cm:<15} {np.linalg.norm(x_cm - x_opt):.2e}")
    print(f"{'Nesterov':<20} {n_nag:<15} {np.linalg.norm(x_nag - x_opt):.2e}")


def exercise_3_adam():
    """
    EXERCISE 3: Implement Adam Optimizer
    ====================================
    
    Implement Adam with all its components.
    
    Tasks:
    a) Implement first and second moment updates
    b) Add bias correction
    c) Test on non-convex function
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Adam Optimizer")
    print("=" * 60)
    
    # YOUR CODE HERE
    def adam(grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
             max_iter=1000, tol=1e-6):
        """
        Adam optimizer.
        
        m_t = β₁·m_{t-1} + (1-β₁)·g_t
        v_t = β₂·v_{t-1} + (1-β₂)·g_t²
        m̂_t = m_t / (1 - β₁ᵗ)
        v̂_t = v_t / (1 - β₂ᵗ)
        x_{t+1} = x_t - α·m̂_t / (√v̂_t + ε)
        """
        # TODO: Implement
        pass


def exercise_3_solution():
    """Solution for Exercise 3."""
    print("\n" + "=" * 60)
    print("SOLUTION 3: Adam Optimizer")
    print("=" * 60)
    
    def adam(grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
             max_iter=1000, tol=1e-6):
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for t in range(1, max_iter + 1):
            g = grad_f(x)
            
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            x_new = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
            
            if np.linalg.norm(x_new - x) < tol:
                break
            
            x = x_new
            history.append(x.copy())
        
        return x, history
    
    # Rastrigin function (many local minima)
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    def rastrigin_grad(x):
        A = 10
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    
    x0 = np.array([2.5, -2.5])
    x_opt, history = adam(rastrigin_grad, x0, alpha=0.1, max_iter=500)
    
    print(f"Rastrigin function (global min at origin):")
    print(f"  Starting: {x0}")
    print(f"  Found: {x_opt}")
    print(f"  f(x): {rastrigin(x_opt):.4f}")
    print(f"  Steps: {len(history)}")


def exercise_4_newton():
    """
    EXERCISE 4: Newton's Method with Modifications
    ==============================================
    
    Implement modified Newton's method that handles:
    - Non-positive definite Hessians
    - Line search for globalization
    
    Tasks:
    a) Implement basic Newton
    b) Add Hessian modification (ensure PD)
    c) Add line search
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Modified Newton")
    print("=" * 60)
    
    # YOUR CODE HERE
    def modified_newton(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
        """
        Newton's method with Hessian modification.
        
        If H is not PD, use H + τI where τ makes it PD.
        """
        # TODO: Implement
        pass


def exercise_4_solution():
    """Solution for Exercise 4."""
    print("\n" + "=" * 60)
    print("SOLUTION 4: Modified Newton")
    print("=" * 60)
    
    def make_pd(H, eps=1e-6):
        """Ensure H is positive definite by adding to diagonal."""
        eigvals = np.linalg.eigvalsh(H)
        min_eig = np.min(eigvals)
        
        if min_eig < eps:
            tau = eps - min_eig
            return H + tau * np.eye(len(H))
        return H
    
    def modified_newton(f, grad_f, hess_f, x0, max_iter=100, tol=1e-6):
        x = x0.copy()
        history = [(x.copy(), f(x))]
        
        for i in range(max_iter):
            g = grad_f(x)
            H = hess_f(x)
            
            if np.linalg.norm(g) < tol:
                break
            
            # Modify Hessian to be PD
            H_mod = make_pd(H)
            
            # Newton step
            p = -np.linalg.solve(H_mod, g)
            
            # Backtracking line search
            alpha = 1.0
            c = 0.1
            rho = 0.5
            while f(x + alpha * p) > f(x) + c * alpha * (g @ p):
                alpha *= rho
                if alpha < 1e-10:
                    break
            
            x = x + alpha * p
            history.append((x.copy(), f(x)))
        
        return x, history
    
    # Test on function with indefinite Hessian at some points
    def f(x):
        return x[0]**4 - 2*x[0]**2 + x[0]*x[1] + x[1]**2
    
    def grad_f(x):
        return np.array([
            4*x[0]**3 - 4*x[0] + x[1],
            x[0] + 2*x[1]
        ])
    
    def hess_f(x):
        return np.array([
            [12*x[0]**2 - 4, 1],
            [1, 2]
        ])
    
    x0 = np.array([2.0, 2.0])
    x_opt, history = modified_newton(f, grad_f, hess_f, x0)
    
    print(f"f(x) = x₁⁴ - 2x₁² + x₁x₂ + x₂²")
    print(f"Starting: {x0}")
    print(f"Found: {x_opt}")
    print(f"f(x): {f(x_opt):.6f}")
    print(f"||∇f||: {np.linalg.norm(grad_f(x_opt)):.2e}")


def exercise_5_sgd_variants():
    """
    EXERCISE 5: SGD with Variance Reduction
    =======================================
    
    Compare different SGD variants.
    
    Tasks:
    a) Implement basic SGD
    b) Implement mini-batch SGD
    c) Compare variance and convergence
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: SGD Variants")
    print("=" * 60)
    
    # YOUR CODE HERE
    def sgd(grad_fn, x0, alpha, n_samples, max_iter):
        """
        Basic SGD: sample one gradient per step.
        """
        # TODO: Implement
        pass
    
    def minibatch_sgd(grad_fn, x0, alpha, n_samples, batch_size, max_iter):
        """
        Mini-batch SGD: average gradients over batch.
        """
        # TODO: Implement
        pass


def exercise_5_solution():
    """Solution for Exercise 5."""
    print("\n" + "=" * 60)
    print("SOLUTION 5: SGD Variants")
    print("=" * 60)
    
    # Linear regression setup
    np.random.seed(42)
    n_samples, n_features = 500, 10
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y) ** 2)
    
    def grad_single(w, i):
        """Gradient for single sample."""
        return X[i:i+1].T @ (X[i:i+1] @ w - y[i:i+1])
    
    def grad_batch(w, indices):
        """Gradient for batch."""
        X_b = X[indices]
        y_b = y[indices]
        return X_b.T @ (X_b @ w - y_b) / len(indices)
    
    def sgd(x0, alpha, max_iter):
        x = x0.copy()
        losses = [loss(x)]
        
        for _ in range(max_iter):
            i = np.random.randint(n_samples)
            x = x - alpha * grad_single(x, i).flatten()
            losses.append(loss(x))
        
        return x, losses
    
    def minibatch_sgd(x0, alpha, batch_size, max_iter):
        x = x0.copy()
        losses = [loss(x)]
        
        for _ in range(max_iter):
            indices = np.random.choice(n_samples, batch_size, replace=False)
            x = x - alpha * grad_batch(x, indices)
            losses.append(loss(x))
        
        return x, losses
    
    x0 = np.zeros(n_features)
    max_iter = 1000
    
    x_sgd, losses_sgd = sgd(x0, 0.001, max_iter)
    x_mb32, losses_mb32 = minibatch_sgd(x0, 0.01, 32, max_iter)
    x_mb128, losses_mb128 = minibatch_sgd(x0, 0.02, 128, max_iter)
    
    print(f"Linear regression: {n_samples} samples, {n_features} features")
    print(f"\n{'Method':<20} {'Final Loss':<15} {'Error ||w-w*||':<15}")
    print("-" * 50)
    print(f"{'SGD (n=1)':<20} {losses_sgd[-1]:<15.6f} {np.linalg.norm(x_sgd - w_true):<15.4f}")
    print(f"{'Mini-batch (n=32)':<20} {losses_mb32[-1]:<15.6f} {np.linalg.norm(x_mb32 - w_true):<15.4f}")
    print(f"{'Mini-batch (n=128)':<20} {losses_mb128[-1]:<15.6f} {np.linalg.norm(x_mb128 - w_true):<15.4f}")


def exercise_6_proximal():
    """
    EXERCISE 6: Proximal Gradient for LASSO
    =======================================
    
    Implement proximal gradient method for L1 regularization.
    
    Tasks:
    a) Implement soft thresholding
    b) Implement proximal gradient descent
    c) Test sparsity recovery
    """
    print("\n" + "=" * 60)
    print("EXERCISE 6: Proximal Gradient")
    print("=" * 60)
    
    # YOUR CODE HERE
    def soft_threshold(x, lam):
        """
        Proximal operator for L1:
        prox(x) = sign(x) * max(|x| - λ, 0)
        """
        # TODO: Implement
        pass
    
    def proximal_gd_lasso(X, y, lam, alpha, max_iter):
        """
        Solve: min 0.5||Xw - y||² + λ||w||₁
        """
        # TODO: Implement
        pass


def exercise_6_solution():
    """Solution for Exercise 6."""
    print("\n" + "=" * 60)
    print("SOLUTION 6: Proximal Gradient")
    print("=" * 60)
    
    def soft_threshold(x, lam):
        return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
    
    def proximal_gd_lasso(X, y, lam, alpha, max_iter, tol=1e-6):
        n, p = X.shape
        w = np.zeros(p)
        
        for _ in range(max_iter):
            grad = X.T @ (X @ w - y) / n
            w_new = soft_threshold(w - alpha * grad, alpha * lam)
            
            if np.linalg.norm(w_new - w) < tol:
                break
            w = w_new
        
        return w
    
    # Sparse problem
    np.random.seed(42)
    n, p = 100, 50
    k = 5  # True sparsity
    
    X = np.random.randn(n, p)
    w_true = np.zeros(p)
    w_true[:k] = np.random.randn(k) * 3
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    print(f"Sparse regression: n={n}, p={p}, k={k}")
    print(f"True non-zeros: {k}")
    
    for lam in [0.01, 0.1, 0.5, 1.0]:
        w_lasso = proximal_gd_lasso(X, y, lam, 0.001, 5000)
        n_nonzero = np.sum(np.abs(w_lasso) > 1e-4)
        
        print(f"\nλ = {lam}:")
        print(f"  Non-zeros: {n_nonzero}")
        print(f"  MSE: {np.mean((w_lasso - w_true)**2):.4f}")


def exercise_7_projected_gd():
    """
    EXERCISE 7: Projected Gradient Descent
    ======================================
    
    Implement projection onto different constraint sets.
    
    Tasks:
    a) Projection onto box constraints
    b) Projection onto L2 ball
    c) Projection onto probability simplex
    """
    print("\n" + "=" * 60)
    print("EXERCISE 7: Projected GD")
    print("=" * 60)
    
    # YOUR CODE HERE
    def project_box(x, lo, hi):
        """Project onto [lo, hi]^n."""
        # TODO: Implement
        pass
    
    def project_l2_ball(x, r):
        """Project onto {x : ||x|| ≤ r}."""
        # TODO: Implement
        pass
    
    def project_simplex(x):
        """Project onto {x : sum(x)=1, x≥0}."""
        # TODO: Implement
        pass


def exercise_7_solution():
    """Solution for Exercise 7."""
    print("\n" + "=" * 60)
    print("SOLUTION 7: Projected GD")
    print("=" * 60)
    
    def project_box(x, lo, hi):
        return np.clip(x, lo, hi)
    
    def project_l2_ball(x, r):
        norm = np.linalg.norm(x)
        if norm <= r:
            return x
        return x * r / norm
    
    def project_simplex(x):
        n = len(x)
        u = np.sort(x)[::-1]
        cssv = np.cumsum(u)
        rho = np.sum(u > (cssv - 1) / np.arange(1, n + 1)) - 1
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(x - theta, 0)
    
    def projected_gd(grad_f, project, x0, alpha, max_iter):
        x = project(x0)
        
        for _ in range(max_iter):
            x = project(x - alpha * grad_f(x))
        
        return x
    
    # Test projections
    print("Projection tests:")
    
    x_test = np.array([2.0, -0.5, 1.5])
    print(f"\nOriginal: {x_test}")
    print(f"Box [0,1]: {project_box(x_test, 0, 1)}")
    print(f"L2 ball r=1: {project_l2_ball(x_test, 1)}")
    print(f"Simplex: {project_simplex(x_test)}")
    print(f"  Sum: {np.sum(project_simplex(x_test)):.4f}")


def exercise_8_conjugate_gradient():
    """
    EXERCISE 8: Nonlinear Conjugate Gradient
    ========================================
    
    Implement CG for general optimization.
    
    Tasks:
    a) Implement Fletcher-Reeves update
    b) Implement Polak-Ribière update
    c) Add restart strategy
    """
    print("\n" + "=" * 60)
    print("EXERCISE 8: Conjugate Gradient")
    print("=" * 60)
    
    # YOUR CODE HERE
    def conjugate_gradient_fr(f, grad_f, x0, max_iter, tol):
        """
        Fletcher-Reeves CG.
        
        β_k = ||∇f(x_k)||² / ||∇f(x_{k-1})||²
        """
        # TODO: Implement
        pass


def exercise_8_solution():
    """Solution for Exercise 8."""
    print("\n" + "=" * 60)
    print("SOLUTION 8: Conjugate Gradient")
    print("=" * 60)
    
    def line_search(f, x, p, max_iter=20):
        alpha = 1.0
        c = 0.1
        rho = 0.5
        
        for _ in range(max_iter):
            if f(x + alpha * p) < f(x):
                return alpha
            alpha *= rho
        
        return alpha
    
    def conjugate_gradient_fr(f, grad_f, x0, max_iter, tol):
        x = x0.copy()
        g = grad_f(x)
        d = -g  # Initial direction
        history = [(x.copy(), f(x))]
        
        for k in range(max_iter):
            if np.linalg.norm(g) < tol:
                break
            
            alpha = line_search(f, x, d)
            x_new = x + alpha * d
            g_new = grad_f(x_new)
            
            # Fletcher-Reeves
            beta = (g_new @ g_new) / (g @ g)
            d = -g_new + beta * d
            
            x = x_new
            g = g_new
            history.append((x.copy(), f(x)))
        
        return x, history
    
    # Test
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        return np.array([
            -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
            200*(x[1] - x[0]**2)
        ])
    
    x0 = np.array([-1.0, 1.0])
    x_cg, history = conjugate_gradient_fr(rosenbrock, rosenbrock_grad, x0, 1000, 1e-6)
    
    print(f"Rosenbrock with CG (Fletcher-Reeves):")
    print(f"  Starting: {x0}")
    print(f"  Found: {x_cg}")
    print(f"  Steps: {len(history)}")


def exercise_9_learning_rate_finder():
    """
    EXERCISE 9: Learning Rate Range Test
    ====================================
    
    Implement Leslie Smith's learning rate finder.
    
    Tasks:
    a) Exponentially increase learning rate
    b) Track loss
    c) Find optimal range
    """
    print("\n" + "=" * 60)
    print("EXERCISE 9: Learning Rate Finder")
    print("=" * 60)
    
    # YOUR CODE HERE
    def lr_range_test(loss_fn, grad_fn, x0, lr_min, lr_max, n_steps):
        """
        Increase lr exponentially from lr_min to lr_max.
        Return (lrs, losses) for plotting.
        """
        # TODO: Implement
        pass


def exercise_9_solution():
    """Solution for Exercise 9."""
    print("\n" + "=" * 60)
    print("SOLUTION 9: Learning Rate Finder")
    print("=" * 60)
    
    def lr_range_test(loss_fn, grad_fn, x0, lr_min, lr_max, n_steps):
        x = x0.copy()
        lrs = []
        losses = []
        
        mult = (lr_max / lr_min) ** (1 / n_steps)
        lr = lr_min
        
        for _ in range(n_steps):
            current_loss = loss_fn(x)
            lrs.append(lr)
            losses.append(current_loss)
            
            if not np.isfinite(current_loss):
                break
            
            g = grad_fn(x)
            x = x - lr * g
            lr *= mult
        
        return lrs, losses
    
    # Test
    A = np.diag([10, 1])
    b = np.array([1, 1])
    
    loss_fn = lambda x: 0.5 * x @ A @ x - b @ x
    grad_fn = lambda x: A @ x - b
    
    x0 = np.array([0.0, 0.0])
    lrs, losses = lr_range_test(loss_fn, grad_fn, x0, 1e-4, 1.0, 50)
    
    print(f"Learning rate range test:")
    print(f"\n{'LR':>12} {'Loss':>12}")
    print("-" * 26)
    
    for i in range(0, len(lrs), 10):
        print(f"{lrs[i]:>12.6f} {losses[i]:>12.4f}")
    
    # Find best lr (steepest descent in loss)
    loss_changes = np.diff(losses)
    best_idx = np.argmin(loss_changes)
    best_lr = lrs[best_idx]
    
    print(f"\nSuggested LR: {best_lr:.6f}")


def exercise_10_optimizer_comparison():
    """
    EXERCISE 10: Comprehensive Optimizer Comparison
    ===============================================
    
    Compare all optimizers on the same problem.
    
    Tasks:
    a) Implement benchmark function
    b) Run each optimizer
    c) Compare iterations and final loss
    """
    print("\n" + "=" * 60)
    print("EXERCISE 10: Optimizer Comparison")
    print("=" * 60)
    
    # Beale's function (challenging test function)
    def beale(x):
        return ((1.5 - x[0] + x[0]*x[1])**2 + 
                (2.25 - x[0] + x[0]*x[1]**2)**2 +
                (2.625 - x[0] + x[0]*x[1]**3)**2)
    
    def beale_grad(x):
        t1 = 1.5 - x[0] + x[0]*x[1]
        t2 = 2.25 - x[0] + x[0]*x[1]**2
        t3 = 2.625 - x[0] + x[0]*x[1]**3
        
        dx0 = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
        dx1 = 2*t1*x[0] + 2*t2*2*x[0]*x[1] + 2*t3*3*x[0]*x[1]**2
        
        return np.array([dx0, dx1])
    
    x0 = np.array([0.0, 0.0])
    x_opt = np.array([3.0, 0.5])
    
    print(f"Beale's function:")
    print(f"  Starting: {x0}")
    print(f"  Optimal: {x_opt}, f(x*) = 0")
    
    # TODO: Run each optimizer and compare


def exercise_10_solution():
    """Solution for Exercise 10."""
    print("\n" + "=" * 60)
    print("SOLUTION 10: Optimizer Comparison")
    print("=" * 60)
    
    def beale(x):
        return ((1.5 - x[0] + x[0]*x[1])**2 + 
                (2.25 - x[0] + x[0]*x[1]**2)**2 +
                (2.625 - x[0] + x[0]*x[1]**3)**2)
    
    def beale_grad(x):
        t1 = 1.5 - x[0] + x[0]*x[1]
        t2 = 2.25 - x[0] + x[0]*x[1]**2
        t3 = 2.625 - x[0] + x[0]*x[1]**3
        
        dx0 = 2*t1*(-1 + x[1]) + 2*t2*(-1 + x[1]**2) + 2*t3*(-1 + x[1]**3)
        dx1 = 2*t1*x[0] + 2*t2*2*x[0]*x[1] + 2*t3*3*x[0]*x[1]**2
        
        return np.array([dx0, dx1])
    
    def gd(grad_f, x0, alpha, max_iter):
        x = x0.copy()
        for _ in range(max_iter):
            x = x - alpha * grad_f(x)
        return x
    
    def momentum(grad_f, x0, alpha, beta, max_iter):
        x = x0.copy()
        v = np.zeros_like(x)
        for _ in range(max_iter):
            v = beta * v + grad_f(x)
            x = x - alpha * v
        return x
    
    def adam(grad_f, x0, alpha, max_iter):
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        for t in range(1, max_iter + 1):
            g = grad_f(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        
        return x
    
    x0 = np.array([0.0, 0.0])
    x_opt = np.array([3.0, 0.5])
    max_iter = 5000
    
    results = {
        'GD': gd(beale_grad, x0, 0.0001, max_iter),
        'Momentum': momentum(beale_grad, x0, 0.0001, 0.9, max_iter),
        'Adam': adam(beale_grad, x0, 0.01, max_iter)
    }
    
    print(f"Beale's function optimization:")
    print(f"  Optimal: {x_opt}")
    print(f"\n{'Method':<15} {'x found':<25} {'f(x)':<15} {'Error':<15}")
    print("-" * 70)
    
    for name, x in results.items():
        f_val = beale(x)
        error = np.linalg.norm(x - x_opt)
        print(f"{name:<15} [{x[0]:.4f}, {x[1]:.4f}]{'':>10} {f_val:<15.6f} {error:<15.4f}")


def run_all_exercises():
    """Run all exercise solutions."""
    exercise_1_solution()
    exercise_2_solution()
    exercise_3_solution()
    exercise_4_solution()
    exercise_5_solution()
    exercise_6_solution()
    exercise_7_solution()
    exercise_8_solution()
    exercise_9_solution()
    exercise_10_solution()


if __name__ == "__main__":
    run_all_exercises()
