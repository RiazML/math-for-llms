"""
Gradient Descent Methods - Examples
===================================
Implementations and comparisons of gradient descent variants.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def example_vanilla_gradient_descent():
    """Basic gradient descent on quadratic."""
    print("=" * 60)
    print("EXAMPLE 1: Vanilla Gradient Descent")
    print("=" * 60)
    
    # Minimize f(x) = (x - 3)^2 + (y - 2)^2
    # Gradient: [2(x-3), 2(y-2)]
    
    def f(w):
        return (w[0] - 3)**2 + (w[1] - 2)**2
    
    def grad_f(w):
        return np.array([2*(w[0] - 3), 2*(w[1] - 2)])
    
    # Initial point
    w = np.array([0.0, 0.0])
    eta = 0.1  # Learning rate
    
    print(f"Objective: f(x,y) = (x-3)² + (y-2)²")
    print(f"Optimal: (3, 2)")
    print(f"Initial: {w}, f = {f(w):.4f}")
    print(f"Learning rate: {eta}\n")
    
    print(f"{'Step':>4} {'x':>10} {'y':>10} {'f(x,y)':>12} {'||grad||':>12}")
    print("-" * 50)
    
    for i in range(15):
        grad = grad_f(w)
        print(f"{i:>4} {w[0]:>10.4f} {w[1]:>10.4f} {f(w):>12.6f} {np.linalg.norm(grad):>12.6f}")
        w = w - eta * grad
    
    print(f"\nFinal: {w}, f = {f(w):.6f}")


def example_learning_rate_effects():
    """Show effects of different learning rates."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Learning Rate Effects")
    print("=" * 60)
    
    # f(x) = x^2, optimal at x = 0
    def f(x):
        return x**2
    
    def grad_f(x):
        return 2*x
    
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.0, 1.1]
    x0 = 10.0
    n_steps = 20
    
    print(f"f(x) = x², optimal at x = 0")
    print(f"Starting at x = {x0}\n")
    
    for eta in learning_rates:
        x = x0
        trajectory = [x]
        
        for _ in range(n_steps):
            x = x - eta * grad_f(x)
            trajectory.append(x)
            if abs(x) > 1e10:  # Diverged
                break
        
        final_x = trajectory[-1]
        if abs(final_x) > 1e10:
            status = "DIVERGED"
        elif abs(final_x) < 0.01:
            status = f"converged to {final_x:.6f}"
        else:
            status = f"slow: x = {final_x:.4f}"
        
        print(f"η = {eta}: {status}")
    
    print("\nNote: For f(x)=x², optimal η < 1 (since Hessian = 2)")


def example_batch_vs_sgd():
    """Compare batch GD vs SGD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch GD vs SGD vs Mini-batch")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Linear regression: y = 2x + 1 + noise
    n = 1000
    X = np.random.randn(n, 1)
    y = 2 * X.flatten() + 1 + np.random.randn(n) * 0.5
    
    # Add bias term
    X_b = np.hstack([np.ones((n, 1)), X])
    
    def mse_loss(w, X, y):
        return np.mean((X @ w - y)**2)
    
    def grad_mse(w, X, y):
        return 2/len(y) * X.T @ (X @ w - y)
    
    # Batch GD
    w_batch = np.zeros(2)
    eta = 0.1
    losses_batch = []
    
    for _ in range(50):
        losses_batch.append(mse_loss(w_batch, X_b, y))
        w_batch = w_batch - eta * grad_mse(w_batch, X_b, y)
    
    # SGD (single sample)
    w_sgd = np.zeros(2)
    eta_sgd = 0.01
    losses_sgd = []
    
    for epoch in range(50):
        losses_sgd.append(mse_loss(w_sgd, X_b, y))
        for i in np.random.permutation(n):
            Xi = X_b[i:i+1]
            yi = y[i:i+1]
            w_sgd = w_sgd - eta_sgd * grad_mse(w_sgd, Xi, yi)
    
    # Mini-batch
    w_mini = np.zeros(2)
    batch_size = 32
    eta_mini = 0.1
    losses_mini = []
    
    for epoch in range(50):
        losses_mini.append(mse_loss(w_mini, X_b, y))
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start+batch_size]
            Xi = X_b[batch_idx]
            yi = y[batch_idx]
            w_mini = w_mini - eta_mini * grad_mse(w_mini, Xi, yi)
    
    print(f"True parameters: [1, 2]")
    print(f"\nFinal parameters after 50 epochs:")
    print(f"  Batch GD:    {np.round(w_batch, 4)}")
    print(f"  SGD:         {np.round(w_sgd, 4)}")
    print(f"  Mini-batch:  {np.round(w_mini, 4)}")
    
    print(f"\nFinal losses:")
    print(f"  Batch:     {losses_batch[-1]:.6f}")
    print(f"  SGD:       {losses_sgd[-1]:.6f}")
    print(f"  Mini-batch: {losses_mini[-1]:.6f}")


def example_momentum():
    """Gradient descent with momentum."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Momentum")
    print("=" * 60)
    
    # Rosenbrock-like function with narrow valley
    def f(w):
        return (1 - w[0])**2 + 10*(w[1] - w[0]**2)**2
    
    def grad_f(w):
        dx = -2*(1 - w[0]) - 40*w[0]*(w[1] - w[0]**2)
        dy = 20*(w[1] - w[0]**2)
        return np.array([dx, dy])
    
    # Without momentum
    w_no_mom = np.array([-1.0, 1.0])
    eta = 0.001
    
    for _ in range(1000):
        w_no_mom = w_no_mom - eta * grad_f(w_no_mom)
    
    # With momentum
    w_mom = np.array([-1.0, 1.0])
    v = np.zeros(2)
    gamma = 0.9
    
    for _ in range(1000):
        v = gamma * v + eta * grad_f(w_mom)
        w_mom = w_mom - v
    
    print(f"Minimizing Rosenbrock-like function")
    print(f"Optimal: (1, 1)")
    print(f"Initial: (-1, 1)")
    print(f"\nAfter 1000 steps (η={eta}):")
    print(f"  Without momentum: {np.round(w_no_mom, 4)}, f = {f(w_no_mom):.6f}")
    print(f"  With momentum (γ={gamma}): {np.round(w_mom, 4)}, f = {f(w_mom):.6f}")
    
    print("\nMomentum helps navigate narrow valleys!")


def example_nesterov():
    """Nesterov accelerated gradient."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Nesterov Accelerated Gradient")
    print("=" * 60)
    
    # Quadratic with different curvatures
    A = np.array([[10, 0], [0, 1]])  # Condition number = 10
    b = np.array([1, 1])
    
    def f(w):
        return 0.5 * w @ A @ w - b @ w
    
    def grad_f(w):
        return A @ w - b
    
    w_optimal = np.linalg.solve(A, b)
    
    # Classical momentum
    w_mom = np.zeros(2)
    v_mom = np.zeros(2)
    eta = 0.1
    gamma = 0.9
    
    dist_mom = []
    for _ in range(100):
        dist_mom.append(np.linalg.norm(w_mom - w_optimal))
        v_mom = gamma * v_mom + eta * grad_f(w_mom)
        w_mom = w_mom - v_mom
    
    # Nesterov momentum
    w_nest = np.zeros(2)
    v_nest = np.zeros(2)
    
    dist_nest = []
    for _ in range(100):
        dist_nest.append(np.linalg.norm(w_nest - w_optimal))
        # Look ahead
        w_lookahead = w_nest - gamma * v_nest
        v_nest = gamma * v_nest + eta * grad_f(w_lookahead)
        w_nest = w_nest - v_nest
    
    print(f"Minimizing f(w) = 0.5 w'Aw - b'w")
    print(f"Condition number κ = {np.linalg.cond(A)}")
    print(f"Optimal: {np.round(w_optimal, 4)}")
    
    print(f"\n{'Step':>6} {'Classical ||w-w*||':>20} {'Nesterov ||w-w*||':>20}")
    print("-" * 50)
    for i in [0, 10, 20, 50, 99]:
        print(f"{i:>6} {dist_mom[i]:>20.6f} {dist_nest[i]:>20.6f}")
    
    print("\nNesterov often converges faster!")


def example_adagrad():
    """AdaGrad optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: AdaGrad")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Sparse features simulation
    # Some features have large gradients, others small
    n, p = 100, 5
    X = np.random.randn(n, p)
    X[:, 0] *= 10  # Feature 0 has large values
    X[:, 4] *= 0.1  # Feature 4 has small values
    
    true_w = np.array([1, 2, 3, 4, 5])
    y = X @ true_w + np.random.randn(n) * 0.5
    
    def loss(w):
        return np.mean((X @ w - y)**2)
    
    def grad(w):
        return 2/n * X.T @ (X @ w - y)
    
    # Vanilla SGD
    w_sgd = np.zeros(p)
    eta_sgd = 0.001  # Small due to large gradients on feature 0
    
    for _ in range(1000):
        w_sgd = w_sgd - eta_sgd * grad(w_sgd)
    
    # AdaGrad
    w_ada = np.zeros(p)
    G = np.zeros(p)
    eta_ada = 0.5
    eps = 1e-8
    
    for _ in range(1000):
        g = grad(w_ada)
        G = G + g**2
        w_ada = w_ada - eta_ada * g / (np.sqrt(G) + eps)
    
    print(f"Features with varying scales")
    print(f"True w: {true_w}")
    
    print(f"\nAfter 1000 steps:")
    print(f"  SGD (η={eta_sgd}): {np.round(w_sgd, 3)}")
    print(f"  AdaGrad (η={eta_ada}): {np.round(w_ada, 3)}")
    
    print(f"\nFinal losses:")
    print(f"  SGD: {loss(w_sgd):.6f}")
    print(f"  AdaGrad: {loss(w_ada):.6f}")
    
    print("\nAdaGrad adapts learning rate per feature!")


def example_rmsprop():
    """RMSProp optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: RMSProp")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Non-stationary objective (changing gradients)
    n = 100
    
    def objective(w, t):
        """Objective changes over time."""
        target = np.array([np.sin(t/10), np.cos(t/10)])
        return np.sum((w - target)**2)
    
    def grad_objective(w, t):
        target = np.array([np.sin(t/10), np.cos(t/10)])
        return 2 * (w - target)
    
    # AdaGrad (learning rate diminishes)
    w_ada = np.zeros(2)
    G = np.zeros(2)
    eta = 0.5
    eps = 1e-8
    
    errors_ada = []
    for t in range(500):
        target = np.array([np.sin(t/10), np.cos(t/10)])
        errors_ada.append(np.linalg.norm(w_ada - target))
        g = grad_objective(w_ada, t)
        G = G + g**2
        w_ada = w_ada - eta * g / (np.sqrt(G) + eps)
    
    # RMSProp (learning rate adapts)
    w_rms = np.zeros(2)
    E_g2 = np.zeros(2)
    rho = 0.9
    
    errors_rms = []
    for t in range(500):
        target = np.array([np.sin(t/10), np.cos(t/10)])
        errors_rms.append(np.linalg.norm(w_rms - target))
        g = grad_objective(w_rms, t)
        E_g2 = rho * E_g2 + (1 - rho) * g**2
        w_rms = w_rms - eta * g / (np.sqrt(E_g2) + eps)
    
    print("Non-stationary objective (target moves)")
    print(f"\n{'Time':>6} {'AdaGrad Error':>15} {'RMSProp Error':>15}")
    print("-" * 40)
    for t in [0, 100, 200, 300, 400]:
        print(f"{t:>6} {errors_ada[t]:>15.6f} {errors_rms[t]:>15.6f}")
    
    print("\nAdaGrad learning rate keeps decreasing")
    print("RMSProp maintains adaptive learning rate")


def example_adam():
    """Adam optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Adam Optimizer")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Noisy quadratic
    A = np.array([[10, 2], [2, 5]])
    b = np.array([3, 2])
    
    def f(w):
        return 0.5 * w @ A @ w - b @ w
    
    def noisy_grad(w):
        return A @ w - b + np.random.randn(2) * 0.5
    
    w_optimal = np.linalg.solve(A, b)
    
    # Adam
    w = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    eta = 0.1
    eps = 1e-8
    
    trajectory = [w.copy()]
    
    for t in range(1, 201):
        g = noisy_grad(w)
        
        # Update biased moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update
        w = w - eta * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(w.copy())
    
    trajectory = np.array(trajectory)
    
    print(f"Noisy quadratic optimization")
    print(f"Optimal: {np.round(w_optimal, 4)}")
    print(f"Adam (β₁={beta1}, β₂={beta2}, η={eta})")
    
    print(f"\n{'Step':>6} {'w':>20} {'||w-w*||':>15}")
    print("-" * 45)
    for t in [0, 10, 50, 100, 200]:
        dist = np.linalg.norm(trajectory[t] - w_optimal)
        print(f"{t:>6} {str(np.round(trajectory[t], 4)):>20} {dist:>15.6f}")
    
    print("\nAdam combines momentum (noise reduction)")
    print("and adaptive learning rates (per-parameter)")


def example_adam_bias_correction():
    """Importance of bias correction in Adam."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Adam Bias Correction")
    print("=" * 60)
    
    # Show why bias correction is needed
    beta1, beta2 = 0.9, 0.999
    true_mean = 5.0
    true_var = 2.0
    
    # Simulate gradient stream
    np.random.seed(42)
    gradients = np.random.normal(true_mean, np.sqrt(true_var), 100)
    
    # Without bias correction
    m_no_corr = 0
    v_no_corr = 0
    
    # With bias correction
    m_corr = 0
    v_corr = 0
    
    print(f"True gradient mean: {true_mean}")
    print(f"True gradient variance: {true_var}")
    print(f"\n{'Step':>6} {'m (no corr)':>12} {'m (corrected)':>15} {'True':>8}")
    print("-" * 45)
    
    for t in range(1, 21):
        g = gradients[t-1]
        
        m_no_corr = beta1 * m_no_corr + (1 - beta1) * g
        m_corr_val = m_no_corr / (1 - beta1**t)
        
        if t in [1, 2, 5, 10, 20]:
            print(f"{t:>6} {m_no_corr:>12.4f} {m_corr_val:>15.4f} {true_mean:>8.1f}")
    
    print("\nWithout correction, early estimates are biased toward 0")
    print("Bias correction fixes this initialization bias")


def example_optimizer_comparison():
    """Compare all optimizers on same problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Optimizer Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Logistic regression
    n, p = 500, 10
    X = np.random.randn(n, p)
    true_w = np.random.randn(p)
    prob = 1 / (1 + np.exp(-X @ true_w))
    y = (np.random.rand(n) < prob).astype(float)
    
    def logistic_loss(w):
        z = X @ w
        return -np.mean(y * z - np.log(1 + np.exp(z)))
    
    def grad_logistic(w):
        prob = 1 / (1 + np.exp(-X @ w))
        return -X.T @ (y - prob) / n
    
    # SGD
    w_sgd = np.zeros(p)
    eta_sgd = 0.1
    losses_sgd = []
    for _ in range(200):
        losses_sgd.append(logistic_loss(w_sgd))
        w_sgd = w_sgd - eta_sgd * grad_logistic(w_sgd)
    
    # SGD + Momentum
    w_mom = np.zeros(p)
    v = np.zeros(p)
    losses_mom = []
    for _ in range(200):
        losses_mom.append(logistic_loss(w_mom))
        v = 0.9 * v + eta_sgd * grad_logistic(w_mom)
        w_mom = w_mom - v
    
    # Adam
    w_adam = np.zeros(p)
    m = np.zeros(p)
    v_adam = np.zeros(p)
    eta_adam = 0.01
    losses_adam = []
    for t in range(1, 201):
        losses_adam.append(logistic_loss(w_adam))
        g = grad_logistic(w_adam)
        m = 0.9 * m + 0.1 * g
        v_adam = 0.999 * v_adam + 0.001 * g**2
        m_hat = m / (1 - 0.9**t)
        v_hat = v_adam / (1 - 0.999**t)
        w_adam = w_adam - eta_adam * m_hat / (np.sqrt(v_hat) + 1e-8)
    
    print("Logistic Regression Optimization")
    print(f"\n{'Step':>6} {'SGD':>12} {'SGD+Mom':>12} {'Adam':>12}")
    print("-" * 50)
    for i in [0, 10, 50, 100, 199]:
        print(f"{i:>6} {losses_sgd[i]:>12.6f} {losses_mom[i]:>12.6f} {losses_adam[i]:>12.6f}")
    
    print(f"\nFinal losses:")
    print(f"  SGD: {losses_sgd[-1]:.6f}")
    print(f"  SGD+Momentum: {losses_mom[-1]:.6f}")
    print(f"  Adam: {losses_adam[-1]:.6f}")


def example_learning_rate_schedule():
    """Learning rate scheduling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Learning Rate Schedules")
    print("=" * 60)
    
    T = 100  # Total steps
    eta_0 = 0.1
    
    # Step decay
    def step_decay(t, eta_0=0.1, drop=0.5, epochs_drop=20):
        return eta_0 * (drop ** (t // epochs_drop))
    
    # Exponential decay
    def exp_decay(t, eta_0=0.1, decay_rate=0.05):
        return eta_0 * np.exp(-decay_rate * t)
    
    # Cosine annealing
    def cosine_anneal(t, eta_0=0.1, eta_min=0.001, T=100):
        return eta_min + 0.5 * (eta_0 - eta_min) * (1 + np.cos(np.pi * t / T))
    
    # Warmup + decay
    def warmup_decay(t, eta_0=0.1, warmup=10, decay_rate=0.01):
        if t < warmup:
            return eta_0 * t / warmup
        return eta_0 * np.exp(-decay_rate * (t - warmup))
    
    print(f"Initial learning rate: {eta_0}")
    print(f"\n{'Step':>6} {'Step':>10} {'Exponential':>12} {'Cosine':>12} {'Warmup':>12}")
    print("-" * 55)
    
    for t in [0, 10, 20, 50, 75, 100]:
        print(f"{t:>6} {step_decay(t):>10.4f} {exp_decay(t):>12.4f} "
              f"{cosine_anneal(t):>12.4f} {warmup_decay(t):>12.4f}")


def example_gradient_clipping():
    """Gradient clipping for stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Gradient Clipping")
    print("=" * 60)
    
    # Simulate exploding gradients
    np.random.seed(42)
    
    gradients = [np.random.randn(5) * scale 
                 for scale in [1, 1, 10, 1, 100, 1, 1]]
    
    max_norm = 5.0
    
    print(f"Max gradient norm: {max_norm}")
    print(f"\n{'Step':>6} {'Original ||g||':>15} {'Clipped ||g||':>15}")
    print("-" * 40)
    
    for i, g in enumerate(gradients):
        orig_norm = np.linalg.norm(g)
        
        # Clip by norm
        if orig_norm > max_norm:
            g_clipped = g * max_norm / orig_norm
        else:
            g_clipped = g
        
        clipped_norm = np.linalg.norm(g_clipped)
        print(f"{i:>6} {orig_norm:>15.4f} {clipped_norm:>15.4f}")
    
    print("\nGradient clipping prevents exploding gradients")
    print("Essential for training RNNs and Transformers")


if __name__ == "__main__":
    example_vanilla_gradient_descent()
    example_learning_rate_effects()
    example_batch_vs_sgd()
    example_momentum()
    example_nesterov()
    example_adagrad()
    example_rmsprop()
    example_adam()
    example_adam_bias_correction()
    example_optimizer_comparison()
    example_learning_rate_schedule()
    example_gradient_clipping()
