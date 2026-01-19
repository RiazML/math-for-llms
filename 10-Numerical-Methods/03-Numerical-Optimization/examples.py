"""
Numerical Optimization - Examples
=================================
Implementation of optimization algorithms.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
import time


def example_1_gradient_descent():
    """
    Example 1: Basic Gradient Descent
    =================================
    Minimize a quadratic function.
    """
    print("=" * 60)
    print("Example 1: Gradient Descent")
    print("=" * 60)
    
    def gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=100, tol=1e-6):
        """Basic gradient descent."""
        x = x0.copy()
        history = [x.copy()]
        
        for i in range(max_iter):
            g = grad_f(x)
            x_new = x - alpha * g
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                print(f"Converged in {i+1} iterations")
                break
            x = x_new
        
        return x, history
    
    # Quadratic: f(x) = 0.5 * x^T A x - b^T x
    A = np.array([[3, 1], [1, 2]], dtype=float)
    b = np.array([1, 1], dtype=float)
    
    f = lambda x: 0.5 * x @ A @ x - b @ x
    grad_f = lambda x: A @ x - b
    
    # Optimal solution: x* = A^{-1} b
    x_opt = np.linalg.solve(A, b)
    print(f"Optimal solution: {x_opt}")
    
    # Gradient descent
    x0 = np.array([0.0, 0.0])
    x_gd, history = gradient_descent(f, grad_f, x0, alpha=0.2)
    
    print(f"GD solution: {x_gd}")
    print(f"Error: {np.linalg.norm(x_gd - x_opt):.2e}")
    
    # Show convergence
    print(f"\nConvergence history (first 5):")
    for i, h in enumerate(history[:5]):
        print(f"  Iter {i}: x = {h}, f(x) = {f(h):.4f}")


def example_2_learning_rate_effect():
    """
    Example 2: Effect of Learning Rate
    ==================================
    Too small, too large, and optimal.
    """
    print("\n" + "=" * 60)
    print("Example 2: Learning Rate Effect")
    print("=" * 60)
    
    # Rosenbrock function (challenging optimization landscape)
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def grad_rosenbrock(x):
        dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dx1 = 200*(x[1] - x[0]**2)
        return np.array([dx0, dx1])
    
    def gradient_descent_with_history(f, grad_f, x0, alpha, max_iter):
        x = x0.copy()
        f_history = [f(x)]
        
        for _ in range(max_iter):
            g = grad_f(x)
            x = x - alpha * g
            f_history.append(f(x))
            
            if not np.isfinite(f(x)):
                break
        
        return x, f_history
    
    x0 = np.array([-1.0, 1.0])
    
    print(f"Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
    print(f"Optimal: x* = (1, 1), f(x*) = 0")
    print(f"Starting point: {x0}")
    
    learning_rates = [0.0001, 0.001, 0.005]
    
    for alpha in learning_rates:
        x_final, history = gradient_descent_with_history(
            rosenbrock, grad_rosenbrock, x0, alpha, 1000
        )
        print(f"\nα = {alpha}:")
        print(f"  Final x: {x_final}")
        print(f"  Final f(x): {history[-1]:.4f}")
        print(f"  Steps: {len(history)}")


def example_3_momentum():
    """
    Example 3: Gradient Descent with Momentum
    =========================================
    Accelerating convergence.
    """
    print("\n" + "=" * 60)
    print("Example 3: Momentum")
    print("=" * 60)
    
    def gd_momentum(grad_f, x0, alpha=0.01, beta=0.9, max_iter=1000, tol=1e-6):
        """Gradient descent with momentum."""
        x = x0.copy()
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for i in range(max_iter):
            g = grad_f(x)
            v = beta * v + g
            x_new = x - alpha * v
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    def nesterov_momentum(grad_f, x0, alpha=0.01, beta=0.9, max_iter=1000, tol=1e-6):
        """Nesterov accelerated gradient."""
        x = x0.copy()
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for i in range(max_iter):
            # Look-ahead gradient
            g = grad_f(x - alpha * beta * v)
            v = beta * v + g
            x_new = x - alpha * v
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    # Ill-conditioned quadratic
    A = np.array([[100, 0], [0, 1]], dtype=float)
    b = np.array([1, 1], dtype=float)
    
    grad_f = lambda x: A @ x - b
    x_opt = np.linalg.solve(A, b)
    
    print(f"Condition number: {np.linalg.cond(A)}")
    print(f"Optimal: {x_opt}")
    
    x0 = np.array([0.0, 0.0])
    
    # Compare methods
    def basic_gd(grad_f, x0, alpha, max_iter, tol):
        x = x0.copy()
        history = [x.copy()]
        for _ in range(max_iter):
            x_new = x - alpha * grad_f(x)
            history.append(x_new.copy())
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        return x, history
    
    x_gd, hist_gd = basic_gd(grad_f, x0, 0.01, 1000, 1e-6)
    x_mom, hist_mom = gd_momentum(grad_f, x0, 0.01, 0.9, 1000, 1e-6)
    x_nag, hist_nag = nesterov_momentum(grad_f, x0, 0.01, 0.9, 1000, 1e-6)
    
    print(f"\nBasic GD: {len(hist_gd)} steps, error = {np.linalg.norm(x_gd - x_opt):.2e}")
    print(f"Momentum: {len(hist_mom)} steps, error = {np.linalg.norm(x_mom - x_opt):.2e}")
    print(f"Nesterov: {len(hist_nag)} steps, error = {np.linalg.norm(x_nag - x_opt):.2e}")


def example_4_adagrad_rmsprop():
    """
    Example 4: AdaGrad and RMSprop
    ==============================
    Adaptive learning rate methods.
    """
    print("\n" + "=" * 60)
    print("Example 4: AdaGrad and RMSprop")
    print("=" * 60)
    
    def adagrad(grad_f, x0, alpha=0.1, eps=1e-8, max_iter=1000, tol=1e-6):
        """AdaGrad optimizer."""
        x = x0.copy()
        G = np.zeros_like(x)  # Accumulated squared gradients
        history = [x.copy()]
        
        for _ in range(max_iter):
            g = grad_f(x)
            G += g ** 2
            x_new = x - alpha * g / (np.sqrt(G) + eps)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    def rmsprop(grad_f, x0, alpha=0.01, beta=0.9, eps=1e-8, max_iter=1000, tol=1e-6):
        """RMSprop optimizer."""
        x = x0.copy()
        v = np.zeros_like(x)  # Moving average of squared gradients
        history = [x.copy()]
        
        for _ in range(max_iter):
            g = grad_f(x)
            v = beta * v + (1 - beta) * g ** 2
            x_new = x - alpha * g / (np.sqrt(v) + eps)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    # Function with different scales in different dimensions
    A = np.diag([100, 1])
    b = np.array([1, 1])
    
    grad_f = lambda x: A @ x - b
    x_opt = np.linalg.solve(A, b)
    
    x0 = np.array([0.0, 0.0])
    
    x_ada, hist_ada = adagrad(grad_f, x0, alpha=1.0)
    x_rms, hist_rms = rmsprop(grad_f, x0, alpha=0.1)
    
    print(f"Optimal: {x_opt}")
    print(f"\nAdaGrad: {len(hist_ada)} steps, error = {np.linalg.norm(x_ada - x_opt):.2e}")
    print(f"RMSprop: {len(hist_rms)} steps, error = {np.linalg.norm(x_rms - x_opt):.2e}")
    
    # Show adaptive learning rate effect
    print(f"\nAdaGrad adapts per-dimension learning rates:")
    print(f"  Dimension 1 (high curvature): effective α decreases faster")
    print(f"  Dimension 2 (low curvature): effective α stays larger")


def example_5_adam():
    """
    Example 5: Adam Optimizer
    =========================
    The workhorse of deep learning optimization.
    """
    print("\n" + "=" * 60)
    print("Example 5: Adam Optimizer")
    print("=" * 60)
    
    def adam(grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
             max_iter=1000, tol=1e-6):
        """Adam optimizer with bias correction."""
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = [x.copy()]
        
        for t in range(1, max_iter + 1):
            g = grad_f(x)
            
            # Update moments
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update
            x_new = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    # Rosenbrock function
    def rosenbrock_grad(x):
        dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dx1 = 200*(x[1] - x[0]**2)
        return np.array([dx0, dx1])
    
    x0 = np.array([-1.0, 1.0])
    x_opt = np.array([1.0, 1.0])
    
    x_adam, hist_adam = adam(rosenbrock_grad, x0, alpha=0.01, max_iter=5000)
    
    print(f"Rosenbrock optimization with Adam:")
    print(f"  Starting point: {x0}")
    print(f"  Optimal: {x_opt}")
    print(f"  Found: {x_adam}")
    print(f"  Error: {np.linalg.norm(x_adam - x_opt):.4f}")
    print(f"  Steps: {len(hist_adam)}")


def example_6_newton_method():
    """
    Example 6: Newton's Method
    ==========================
    Second-order optimization for fast convergence.
    """
    print("\n" + "=" * 60)
    print("Example 6: Newton's Method")
    print("=" * 60)
    
    def newton_method(grad_f, hess_f, x0, max_iter=100, tol=1e-10):
        """Newton's method for optimization."""
        x = x0.copy()
        history = [x.copy()]
        
        for i in range(max_iter):
            g = grad_f(x)
            H = hess_f(x)
            
            # Newton step: x_new = x - H^{-1} g
            try:
                p = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                print("Singular Hessian, stopping")
                break
            
            x_new = x - p
            history.append(x_new.copy())
            
            if np.linalg.norm(g) < tol:
                print(f"Converged in {i+1} iterations")
                break
            x = x_new
        
        return x, history
    
    # Quadratic function
    A = np.array([[4, 1], [1, 3]], dtype=float)
    b = np.array([1, 2], dtype=float)
    
    f = lambda x: 0.5 * x @ A @ x - b @ x
    grad_f = lambda x: A @ x - b
    hess_f = lambda x: A  # Constant Hessian for quadratic
    
    x_opt = np.linalg.solve(A, b)
    
    x0 = np.array([10.0, 10.0])
    
    x_newton, hist_newton = newton_method(grad_f, hess_f, x0)
    
    print(f"Quadratic function:")
    print(f"  Starting point: {x0}")
    print(f"  Optimal: {x_opt}")
    print(f"  Newton found: {x_newton}")
    print(f"  Error: {np.linalg.norm(x_newton - x_opt):.2e}")
    
    # Newton converges in 1 step for quadratics!
    print(f"\nNote: Newton converges in 1 step for quadratic functions!")


def example_7_bfgs():
    """
    Example 7: BFGS Quasi-Newton Method
    ===================================
    Approximating the Hessian from gradient information.
    """
    print("\n" + "=" * 60)
    print("Example 7: BFGS")
    print("=" * 60)
    
    def bfgs(f, grad_f, x0, max_iter=1000, tol=1e-6):
        """BFGS quasi-Newton method."""
        n = len(x0)
        x = x0.copy()
        H = np.eye(n)  # Initial inverse Hessian approximation
        history = [x.copy()]
        
        g = grad_f(x)
        
        for i in range(max_iter):
            if np.linalg.norm(g) < tol:
                print(f"Converged in {i} iterations")
                break
            
            # Search direction
            p = -H @ g
            
            # Line search (simple backtracking)
            alpha = 1.0
            c = 0.1
            rho = 0.5
            while f(x + alpha * p) > f(x) + c * alpha * g @ p:
                alpha *= rho
            
            # Update
            x_new = x + alpha * p
            g_new = grad_f(x_new)
            
            # BFGS update
            s = x_new - x
            y = g_new - g
            
            rho_k = 1.0 / (y @ s) if y @ s > 1e-10 else 0
            
            if rho_k > 0:
                I = np.eye(n)
                H = (I - rho_k * np.outer(s, y)) @ H @ (I - rho_k * np.outer(y, s))
                H += rho_k * np.outer(s, s)
            
            x = x_new
            g = g_new
            history.append(x.copy())
        
        return x, history
    
    # Rosenbrock function
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    
    def rosenbrock_grad(x):
        dx0 = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dx1 = 200*(x[1] - x[0]**2)
        return np.array([dx0, dx1])
    
    x0 = np.array([-1.0, 1.0])
    x_opt = np.array([1.0, 1.0])
    
    x_bfgs, hist_bfgs = bfgs(rosenbrock, rosenbrock_grad, x0)
    
    print(f"Rosenbrock optimization with BFGS:")
    print(f"  Starting point: {x0}")
    print(f"  Found: {x_bfgs}")
    print(f"  Optimal: {x_opt}")
    print(f"  Error: {np.linalg.norm(x_bfgs - x_opt):.2e}")
    print(f"  Steps: {len(hist_bfgs)}")


def example_8_sgd():
    """
    Example 8: Stochastic Gradient Descent
    ======================================
    For large-scale optimization.
    """
    print("\n" + "=" * 60)
    print("Example 8: Stochastic Gradient Descent")
    print("=" * 60)
    
    # Linear regression: min_w ||Xw - y||^2
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.random.randn(n_features)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    def full_gradient(w):
        return X.T @ (X @ w - y) / n_samples
    
    def stochastic_gradient(w, batch_indices):
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        return X_batch.T @ (X_batch @ w - y_batch) / len(batch_indices)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y) ** 2)
    
    # Full batch GD
    def full_gd(w0, alpha, max_iter):
        w = w0.copy()
        history = [loss(w)]
        for _ in range(max_iter):
            w = w - alpha * full_gradient(w)
            history.append(loss(w))
        return w, history
    
    # Mini-batch SGD
    def mini_batch_sgd(w0, alpha, batch_size, max_iter):
        w = w0.copy()
        history = [loss(w)]
        for _ in range(max_iter):
            indices = np.random.choice(n_samples, batch_size, replace=False)
            w = w - alpha * stochastic_gradient(w, indices)
            history.append(loss(w))
        return w, history
    
    w0 = np.zeros(n_features)
    
    # Compare
    start = time.time()
    w_gd, hist_gd = full_gd(w0, 0.1, 100)
    time_gd = time.time() - start
    
    start = time.time()
    w_sgd, hist_sgd = mini_batch_sgd(w0, 0.01, 32, 1000)
    time_sgd = time.time() - start
    
    print(f"Linear regression: {n_samples} samples, {n_features} features")
    print(f"\nFull batch GD (100 iters):")
    print(f"  Time: {time_gd*1000:.2f} ms")
    print(f"  Final loss: {hist_gd[-1]:.6f}")
    print(f"  Error ||w - w_true||: {np.linalg.norm(w_gd - w_true):.4f}")
    
    print(f"\nMini-batch SGD (1000 iters, batch=32):")
    print(f"  Time: {time_sgd*1000:.2f} ms")
    print(f"  Final loss: {hist_sgd[-1]:.6f}")
    print(f"  Error ||w - w_true||: {np.linalg.norm(w_sgd - w_true):.4f}")


def example_9_learning_rate_schedule():
    """
    Example 9: Learning Rate Scheduling
    ===================================
    Different decay strategies.
    """
    print("\n" + "=" * 60)
    print("Example 9: Learning Rate Scheduling")
    print("=" * 60)
    
    def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=10):
        return initial_lr * (drop ** (epoch // epochs_drop))
    
    def exponential_decay(epoch, initial_lr=0.1, decay_rate=0.95):
        return initial_lr * (decay_rate ** epoch)
    
    def cosine_annealing(epoch, initial_lr=0.1, total_epochs=100):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
    
    def warmup_linear(epoch, initial_lr=0.1, warmup_epochs=10, total_epochs=100):
        if epoch < warmup_epochs:
            return initial_lr * epoch / warmup_epochs
        else:
            return initial_lr * (1 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
    
    epochs = 100
    print(f"Learning rate at different epochs:\n")
    print(f"{'Epoch':<10} {'Step':<12} {'Exponential':<12} {'Cosine':<12} {'Warmup':<12}")
    print("-" * 60)
    
    for epoch in [0, 5, 10, 20, 50, 80, 99]:
        lr_step = step_decay(epoch)
        lr_exp = exponential_decay(epoch)
        lr_cos = cosine_annealing(epoch, total_epochs=epochs)
        lr_warm = warmup_linear(epoch, total_epochs=epochs)
        
        print(f"{epoch:<10} {lr_step:<12.4f} {lr_exp:<12.4f} {lr_cos:<12.4f} {lr_warm:<12.4f}")


def example_10_projected_gradient():
    """
    Example 10: Projected Gradient Descent
    ======================================
    Constrained optimization.
    """
    print("\n" + "=" * 60)
    print("Example 10: Projected Gradient Descent")
    print("=" * 60)
    
    def project_onto_box(x, lower, upper):
        """Project onto box constraints."""
        return np.clip(x, lower, upper)
    
    def project_onto_simplex(x):
        """Project onto probability simplex (sum to 1, all >= 0)."""
        n = len(x)
        sorted_x = np.sort(x)[::-1]
        cumsum = np.cumsum(sorted_x)
        
        t = (cumsum - 1) / np.arange(1, n + 1)
        rho = np.sum(sorted_x > t)
        theta = (cumsum[rho-1] - 1) / rho
        
        return np.maximum(x - theta, 0)
    
    def projected_gd(grad_f, project, x0, alpha=0.1, max_iter=100, tol=1e-6):
        """Projected gradient descent."""
        x = project(x0)
        history = [x.copy()]
        
        for _ in range(max_iter):
            g = grad_f(x)
            x_new = project(x - alpha * g)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    # Example 1: Box constraints
    # min ||x - c||^2 subject to 0 <= x <= 1
    c = np.array([2.0, -0.5, 0.5])
    grad_f = lambda x: 2 * (x - c)
    
    x0 = np.array([0.5, 0.5, 0.5])
    
    x_box, _ = projected_gd(
        grad_f, 
        lambda x: project_onto_box(x, 0, 1),
        x0
    )
    
    print("Box constraints: min ||x - c||² s.t. 0 ≤ x ≤ 1")
    print(f"  c = {c}")
    print(f"  Optimal x = clip(c, 0, 1) = {np.clip(c, 0, 1)}")
    print(f"  Found x = {x_box}")
    
    # Example 2: Simplex constraints
    # min ||x - c||^2 subject to sum(x) = 1, x >= 0
    c = np.array([0.5, 0.3, 0.1, 0.4])
    grad_f = lambda x: 2 * (x - c)
    
    x0 = np.ones(4) / 4
    
    x_simplex, _ = projected_gd(
        grad_f,
        project_onto_simplex,
        x0,
        alpha=0.5
    )
    
    print(f"\nSimplex constraints: min ||x - c||² s.t. sum(x)=1, x≥0")
    print(f"  c = {c}")
    print(f"  Found x = {x_simplex.round(4)}")
    print(f"  sum(x) = {sum(x_simplex):.6f}")


def example_11_proximal_gradient():
    """
    Example 11: Proximal Gradient Method
    ====================================
    For composite optimization (L1 regularization).
    """
    print("\n" + "=" * 60)
    print("Example 11: Proximal Gradient (LASSO)")
    print("=" * 60)
    
    def soft_threshold(x, threshold):
        """Proximal operator for L1 norm."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def proximal_gradient(X, y, lambda_reg, alpha=0.01, max_iter=1000, tol=1e-6):
        """
        Solve LASSO: min_w 0.5||Xw - y||^2 + lambda||w||_1
        """
        n, p = X.shape
        w = np.zeros(p)
        history = []
        
        for _ in range(max_iter):
            # Gradient of smooth part
            grad = X.T @ (X @ w - y) / n
            
            # Gradient step then proximal step
            w_new = soft_threshold(w - alpha * grad, alpha * lambda_reg)
            
            # Loss
            loss = 0.5 * np.mean((X @ w_new - y)**2) + lambda_reg * np.sum(np.abs(w_new))
            history.append(loss)
            
            if np.linalg.norm(w_new - w) < tol:
                break
            w = w_new
        
        return w, history
    
    # Generate sparse problem
    np.random.seed(42)
    n_samples, n_features = 100, 20
    n_nonzero = 5
    
    X = np.random.randn(n_samples, n_features)
    w_true = np.zeros(n_features)
    w_true[:n_nonzero] = np.random.randn(n_nonzero)
    y = X @ w_true + 0.1 * np.random.randn(n_samples)
    
    print(f"Sparse linear regression:")
    print(f"  {n_samples} samples, {n_features} features")
    print(f"  True sparsity: {n_nonzero} non-zero coefficients")
    
    # Solve with different regularization
    for lambda_reg in [0.01, 0.1, 0.5]:
        w_lasso, _ = proximal_gradient(X, y, lambda_reg, alpha=0.001, max_iter=5000)
        n_selected = np.sum(np.abs(w_lasso) > 1e-4)
        mse = np.mean((w_lasso - w_true) ** 2)
        
        print(f"\n  λ = {lambda_reg}:")
        print(f"    Selected features: {n_selected}")
        print(f"    MSE vs true w: {mse:.4f}")


def example_12_adamw():
    """
    Example 12: AdamW (Adam with Decoupled Weight Decay)
    ===================================================
    Modern optimization for deep learning.
    """
    print("\n" + "=" * 60)
    print("Example 12: AdamW")
    print("=" * 60)
    
    def adamw(grad_f, x0, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
              weight_decay=0.01, max_iter=1000, tol=1e-6):
        """AdamW: Adam with decoupled weight decay."""
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = [x.copy()]
        
        for t in range(1, max_iter + 1):
            g = grad_f(x)
            
            # Update moments
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update with decoupled weight decay
            x_new = x - alpha * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)
            history.append(x_new.copy())
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        return x, history
    
    # Compare Adam vs AdamW on regularized problem
    A = np.array([[2, 0], [0, 10]], dtype=float)
    b = np.array([1, 1], dtype=float)
    
    # L2-regularized loss: f(x) = 0.5 x^T A x - b^T x + 0.5 λ ||x||^2
    lambda_reg = 0.1
    
    # For Adam, include L2 in gradient
    grad_f_adam = lambda x: A @ x - b + lambda_reg * x
    
    # For AdamW, don't include L2 in gradient (handled separately)
    grad_f_adamw = lambda x: A @ x - b
    
    x0 = np.array([5.0, 5.0])
    
    # Adam (L2 in gradient)
    def adam(grad_f, x0, alpha=0.1, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        x = x0.copy()
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        for t in range(1, max_iter + 1):
            g = grad_f(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            x = x - alpha * m_hat / (np.sqrt(v_hat) + eps)
        
        return x
    
    x_adam = adam(grad_f_adam, x0, alpha=0.1)
    x_adamw, _ = adamw(grad_f_adamw, x0, alpha=0.1, weight_decay=lambda_reg)
    
    # Optimal solution with L2: (A + λI)^{-1} b
    x_opt = np.linalg.solve(A + lambda_reg * np.eye(2), b)
    
    print(f"L2-regularized quadratic (λ = {lambda_reg})")
    print(f"Optimal: {x_opt}")
    print(f"Adam:    {x_adam}")
    print(f"AdamW:   {x_adamw}")
    print(f"\nAdamW applies weight decay directly to parameters,")
    print(f"which is more principled than L2 in gradient.")


def run_all_examples():
    """Run all examples."""
    example_1_gradient_descent()
    example_2_learning_rate_effect()
    example_3_momentum()
    example_4_adagrad_rmsprop()
    example_5_adam()
    example_6_newton_method()
    example_7_bfgs()
    example_8_sgd()
    example_9_learning_rate_schedule()
    example_10_projected_gradient()
    example_11_proximal_gradient()
    example_12_adamw()


if __name__ == "__main__":
    run_all_examples()
