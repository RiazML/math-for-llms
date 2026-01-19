"""
Adaptive Learning Rate Methods - Examples
==========================================
Implementations of momentum, AdaGrad, RMSProp, Adam, and variants.
"""

import numpy as np
from typing import Tuple, Callable, Dict, List


def example_momentum():
    """Compare SGD with and without momentum."""
    print("=" * 60)
    print("EXAMPLE 1: Momentum")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Ill-conditioned quadratic
    A = np.array([[10, 0], [0, 1]])  # Condition number = 10
    b = np.array([1, 1])
    
    def loss(w):
        return 0.5 * w @ A @ w - b @ w
    
    def gradient(w):
        return A @ w - b
    
    w_opt = np.linalg.solve(A, b)
    
    eta = 0.1
    beta = 0.9
    
    # SGD without momentum
    w_sgd = np.array([0.0, 0.0])
    sgd_losses = [loss(w_sgd)]
    
    # SGD with momentum
    w_mom = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    mom_losses = [loss(w_mom)]
    
    for _ in range(50):
        # Vanilla SGD
        g_sgd = gradient(w_sgd)
        w_sgd = w_sgd - eta * g_sgd
        sgd_losses.append(loss(w_sgd))
        
        # Momentum
        g_mom = gradient(w_mom)
        v = beta * v + g_mom
        w_mom = w_mom - eta * v
        mom_losses.append(loss(w_mom))
    
    print(f"Quadratic with condition number κ = 10")
    print(f"η = {eta}, β = {beta}")
    print(f"\nOptimal w* = {w_opt}")
    print(f"\n{'Step':>5} {'SGD Loss':>15} {'Momentum Loss':>15}")
    print("-" * 40)
    
    for i in [0, 5, 10, 20, 50]:
        print(f"{i:>5} {sgd_losses[i]:>15.6f} {mom_losses[i]:>15.6f}")
    
    print(f"\nFinal errors:")
    print(f"  SGD: {np.linalg.norm(w_sgd - w_opt):.6e}")
    print(f"  Momentum: {np.linalg.norm(w_mom - w_opt):.6e}")


def example_nesterov():
    """Nesterov accelerated gradient."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Nesterov Momentum")
    print("=" * 60)
    
    np.random.seed(42)
    
    A = np.array([[10, 0], [0, 1]])
    b = np.array([1, 1])
    
    def gradient(w):
        return A @ w - b
    
    def loss(w):
        return 0.5 * w @ A @ w - b @ w
    
    eta = 0.1
    beta = 0.9
    
    # Standard momentum
    w_mom = np.array([0.0, 0.0])
    v_mom = np.array([0.0, 0.0])
    
    # Nesterov momentum
    w_nes = np.array([0.0, 0.0])
    v_nes = np.array([0.0, 0.0])
    
    mom_losses = []
    nes_losses = []
    
    for _ in range(50):
        # Standard momentum
        g_mom = gradient(w_mom)
        v_mom = beta * v_mom + g_mom
        w_mom = w_mom - eta * v_mom
        mom_losses.append(loss(w_mom))
        
        # Nesterov: gradient at look-ahead position
        lookahead = w_nes - eta * beta * v_nes
        g_nes = gradient(lookahead)
        v_nes = beta * v_nes + g_nes
        w_nes = w_nes - eta * v_nes
        nes_losses.append(loss(w_nes))
    
    print("Nesterov computes gradient at 'look-ahead' position")
    print(f"\n{'Step':>5} {'Momentum':>15} {'Nesterov':>15}")
    print("-" * 40)
    
    for i in [0, 5, 10, 20, 49]:
        print(f"{i+1:>5} {mom_losses[i]:>15.6f} {nes_losses[i]:>15.6f}")


def example_adagrad():
    """AdaGrad optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: AdaGrad")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Problem with sparse gradients
    d = 10
    n_samples = 100
    
    # Sparse data: each sample only affects a few features
    X = np.zeros((n_samples, d))
    for i in range(n_samples):
        # Only 2 random features are non-zero
        idx = np.random.choice(d, 2, replace=False)
        X[i, idx] = np.random.randn(2)
    
    w_true = np.random.randn(d)
    y = X @ w_true
    
    def gradient(w, idx):
        """Gradient for single sample."""
        return (X[idx] @ w - y[idx]) * X[idx]
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    eta = 1.0  # AdaGrad can use larger learning rate
    eps = 1e-8
    
    # AdaGrad
    w_ada = np.zeros(d)
    G = np.zeros(d)  # Sum of squared gradients
    
    # SGD for comparison
    w_sgd = np.zeros(d)
    
    ada_losses = []
    sgd_losses = []
    
    for t in range(500):
        idx = np.random.randint(n_samples)
        
        g = gradient(w_ada, idx)
        G = G + g**2
        w_ada = w_ada - eta * g / (np.sqrt(G) + eps)
        
        g_sgd = gradient(w_sgd, idx)
        w_sgd = w_sgd - 0.01 * g_sgd  # Smaller LR for SGD
        
        if t % 50 == 0:
            ada_losses.append(loss(w_ada))
            sgd_losses.append(loss(w_sgd))
    
    print("Sparse data: each sample affects only 2 features")
    print(f"AdaGrad η = {eta}, SGD η = 0.01")
    
    print(f"\n{'Step':>5} {'AdaGrad Loss':>15} {'SGD Loss':>15}")
    print("-" * 40)
    
    for i, (a, s) in enumerate(zip(ada_losses, sgd_losses)):
        print(f"{i*50:>5} {a:>15.6f} {s:>15.6f}")
    
    print("\nAdaGrad adapts learning rate per feature:")
    print(f"  Max G[i]: {np.max(G):.2f}")
    print(f"  Min G[i]: {np.min(G):.2f}")
    print("  Rare features get larger updates!")


def example_rmsprop():
    """RMSProp optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: RMSProp")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Rosenbrock function (non-convex)
    def rosenbrock(w):
        return (1 - w[0])**2 + 100 * (w[1] - w[0]**2)**2
    
    def gradient(w):
        dw0 = -2*(1 - w[0]) - 400*w[0]*(w[1] - w[0]**2)
        dw1 = 200*(w[1] - w[0]**2)
        return np.array([dw0, dw1])
    
    eta = 0.001
    rho = 0.9  # Decay rate
    eps = 1e-8
    
    # RMSProp
    w_rms = np.array([-1.0, 1.0])
    v = np.zeros(2)
    
    # AdaGrad for comparison
    w_ada = np.array([-1.0, 1.0])
    G = np.zeros(2)
    
    rms_losses = [rosenbrock(w_rms)]
    ada_losses = [rosenbrock(w_ada)]
    
    for _ in range(5000):
        # RMSProp
        g_rms = gradient(w_rms)
        v = rho * v + (1 - rho) * g_rms**2
        w_rms = w_rms - eta * g_rms / (np.sqrt(v) + eps)
        rms_losses.append(rosenbrock(w_rms))
        
        # AdaGrad
        g_ada = gradient(w_ada)
        G = G + g_ada**2
        w_ada = w_ada - eta * g_ada / (np.sqrt(G) + eps)
        ada_losses.append(rosenbrock(w_ada))
    
    print("Rosenbrock function (minimum at [1, 1])")
    print(f"η = {eta}, ρ = {rho}")
    
    print(f"\n{'Step':>6} {'RMSProp':>15} {'AdaGrad':>15}")
    print("-" * 40)
    
    for i in [0, 100, 500, 1000, 5000]:
        print(f"{i:>6} {rms_losses[i]:>15.4f} {ada_losses[i]:>15.4f}")
    
    print(f"\nFinal positions:")
    print(f"  RMSProp: {np.round(w_rms, 4)}")
    print(f"  AdaGrad: {np.round(w_ada, 4)}")
    print("\nRMSProp doesn't let learning rate decay to zero")


def example_adam():
    """Adam optimizer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Adam")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Logistic regression
    n, d = 100, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = np.sign(X @ w_true)
    
    def loss(w):
        z = np.clip(X @ w, -500, 500)
        return np.mean(np.log(1 + np.exp(-y * z)))
    
    def gradient(w, batch_idx):
        X_b = X[batch_idx]
        y_b = y[batch_idx]
        z = np.clip(X_b @ w, -500, 500)
        p = 1 / (1 + np.exp(-y_b * z))
        return -X_b.T @ (y_b * (1 - p)) / len(batch_idx)
    
    # Adam parameters
    eta = 0.01
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    batch_size = 20
    
    # Initialize
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    
    losses = []
    
    for t in range(1, 501):
        batch_idx = np.random.choice(n, batch_size, replace=False)
        g = gradient(w, batch_idx)
        
        # Update moments
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # Update
        w = w - eta * m_hat / (np.sqrt(v_hat) + eps)
        
        if t % 50 == 0:
            losses.append(loss(w))
    
    print("Logistic regression with Adam")
    print(f"η = {eta}, β₁ = {beta1}, β₂ = {beta2}")
    
    print(f"\n{'Step':>5} {'Loss':>15}")
    print("-" * 25)
    
    for i, l in enumerate(losses):
        print(f"{(i+1)*50:>5} {l:>15.6f}")
    
    print("\nAdam combines momentum and adaptive LR with bias correction")


def example_bias_correction():
    """Importance of bias correction in Adam."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Bias Correction Importance")
    print("=" * 60)
    
    beta1, beta2 = 0.9, 0.999
    
    # Simulate constant gradient
    g = 1.0
    
    print("Simulating constant gradient g = 1.0")
    print(f"β₁ = {beta1}, β₂ = {beta2}")
    
    m = 0
    v = 0
    
    print(f"\n{'t':>4} {'m_t':>10} {'m̂_t':>10} {'v_t':>12} {'v̂_t':>12}")
    print("-" * 55)
    
    for t in range(1, 11):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        print(f"{t:>4} {m:>10.4f} {m_hat:>10.4f} {v:>12.6f} {v_hat:>12.6f}")
    
    print(f"\nTrue values: E[g] = 1.0, E[g²] = 1.0")
    print("Without correction, early estimates are biased toward 0")
    print("Bias correction fixes this!")


def example_adamw():
    """AdamW vs Adam with L2 regularization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: AdamW (Decoupled Weight Decay)")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 50, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w, lamb=0):
        return 0.5 * np.mean((X @ w - y)**2) + 0.5 * lamb * np.sum(w**2)
    
    def gradient(w):
        return X.T @ (X @ w - y) / n
    
    eta = 0.1
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    lamb = 0.1  # Weight decay
    
    # Adam with L2 in gradient
    w_adam = np.zeros(d)
    m_adam = np.zeros(d)
    v_adam = np.zeros(d)
    
    # AdamW with decoupled weight decay
    w_adamw = np.zeros(d)
    m_adamw = np.zeros(d)
    v_adamw = np.zeros(d)
    
    adam_losses = []
    adamw_losses = []
    
    for t in range(1, 201):
        # Adam with L2: add λw to gradient
        g_adam = gradient(w_adam) + lamb * w_adam
        m_adam = beta1 * m_adam + (1 - beta1) * g_adam
        v_adam = beta2 * v_adam + (1 - beta2) * g_adam**2
        m_hat = m_adam / (1 - beta1**t)
        v_hat = v_adam / (1 - beta2**t)
        w_adam = w_adam - eta * m_hat / (np.sqrt(v_hat) + eps)
        
        # AdamW: gradient without L2, then apply weight decay
        g_adamw = gradient(w_adamw)  # No L2 in gradient
        m_adamw = beta1 * m_adamw + (1 - beta1) * g_adamw
        v_adamw = beta2 * v_adamw + (1 - beta2) * g_adamw**2
        m_hat = m_adamw / (1 - beta1**t)
        v_hat = v_adamw / (1 - beta2**t)
        w_adamw = w_adamw - eta * (m_hat / (np.sqrt(v_hat) + eps) + lamb * w_adamw)
        
        if t % 20 == 0:
            adam_losses.append(loss(w_adam))
            adamw_losses.append(loss(w_adamw))
    
    print("Adam with L2: includes λw in gradient (affected by adaptive LR)")
    print("AdamW: applies weight decay after Adam step (decoupled)")
    
    print(f"\n{'Step':>5} {'Adam+L2':>15} {'AdamW':>15}")
    print("-" * 40)
    
    for i, (a, aw) in enumerate(zip(adam_losses, adamw_losses)):
        print(f"{(i+1)*20:>5} {a:>15.6f} {aw:>15.6f}")
    
    print(f"\nFinal ||w||:")
    print(f"  Adam+L2: {np.linalg.norm(w_adam):.4f}")
    print(f"  AdamW:   {np.linalg.norm(w_adamw):.4f}")


def example_lr_schedules():
    """Learning rate schedules."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Learning Rate Schedules")
    print("=" * 60)
    
    T = 100  # Total steps
    eta_max = 0.1
    eta_min = 0.001
    
    def constant(t):
        return eta_max
    
    def step_decay(t, gamma=0.5, step_size=30):
        return eta_max * (gamma ** (t // step_size))
    
    def exponential(t, lamb=0.03):
        return eta_max * np.exp(-lamb * t)
    
    def cosine(t):
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / T))
    
    def warmup_cosine(t, warmup=10):
        if t < warmup:
            return eta_max * t / warmup
        return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (t - warmup) / (T - warmup)))
    
    schedules = {
        'Constant': constant,
        'Step': step_decay,
        'Exponential': exponential,
        'Cosine': cosine,
        'Warmup+Cosine': warmup_cosine
    }
    
    print(f"{'t':>4}", end='')
    for name in schedules:
        print(f"{name:>15}", end='')
    print()
    print("-" * 80)
    
    for t in [0, 10, 25, 50, 75, 100]:
        print(f"{t:>4}", end='')
        for name, sched in schedules.items():
            print(f"{sched(t):>15.4f}", end='')
        print()


def example_optimizer_comparison():
    """Compare all optimizers on same problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Optimizer Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 100, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def gradient(w):
        return X.T @ (X @ w - y) / n
    
    n_steps = 100
    
    # SGD
    def run_sgd(eta=0.1):
        w = np.zeros(d)
        losses = [loss(w)]
        for _ in range(n_steps):
            w = w - eta * gradient(w)
            losses.append(loss(w))
        return losses
    
    # Momentum
    def run_momentum(eta=0.1, beta=0.9):
        w = np.zeros(d)
        v = np.zeros(d)
        losses = [loss(w)]
        for _ in range(n_steps):
            v = beta * v + gradient(w)
            w = w - eta * v
            losses.append(loss(w))
        return losses
    
    # AdaGrad
    def run_adagrad(eta=0.5):
        w = np.zeros(d)
        G = np.zeros(d)
        losses = [loss(w)]
        for _ in range(n_steps):
            g = gradient(w)
            G = G + g**2
            w = w - eta * g / (np.sqrt(G) + 1e-8)
            losses.append(loss(w))
        return losses
    
    # RMSProp
    def run_rmsprop(eta=0.01, rho=0.9):
        w = np.zeros(d)
        v = np.zeros(d)
        losses = [loss(w)]
        for _ in range(n_steps):
            g = gradient(w)
            v = rho * v + (1 - rho) * g**2
            w = w - eta * g / (np.sqrt(v) + 1e-8)
            losses.append(loss(w))
        return losses
    
    # Adam
    def run_adam(eta=0.01, beta1=0.9, beta2=0.999):
        w = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        losses = [loss(w)]
        for t in range(1, n_steps + 1):
            g = gradient(w)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            w = w - eta * m_hat / (np.sqrt(v_hat) + 1e-8)
            losses.append(loss(w))
        return losses
    
    results = {
        'SGD': run_sgd(),
        'Momentum': run_momentum(),
        'AdaGrad': run_adagrad(),
        'RMSProp': run_rmsprop(),
        'Adam': run_adam()
    }
    
    print(f"{'Step':>5}", end='')
    for name in results:
        print(f"{name:>12}", end='')
    print()
    print("-" * 70)
    
    for step in [0, 10, 25, 50, 100]:
        print(f"{step:>5}", end='')
        for name, losses in results.items():
            print(f"{losses[step]:>12.6f}", end='')
        print()
    
    print("\nFinal losses:")
    for name, losses in results.items():
        print(f"  {name}: {losses[-1]:.6f}")


def example_one_cycle():
    """One-cycle learning rate policy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: One-Cycle Learning Rate")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 100, 10
    X = np.random.randn(n, d)
    y = np.sign(X @ np.random.randn(d))
    
    def loss(w):
        z = np.clip(X @ w, -500, 500)
        return np.mean(np.log(1 + np.exp(-y * z)))
    
    def gradient(w):
        z = np.clip(X @ w, -500, 500)
        p = 1 / (1 + np.exp(-y * z))
        return -X.T @ (y * (1 - p)) / n
    
    T = 200
    
    # One-cycle: increase then decrease
    def one_cycle_lr(t, max_lr=0.1, div_factor=25, final_div=1e4):
        mid = T // 2
        if t < mid:
            # Warmup phase
            return max_lr / div_factor + (max_lr - max_lr/div_factor) * t / mid
        else:
            # Annealing phase
            return max_lr - (max_lr - max_lr/final_div) * (t - mid) / (T - mid)
    
    # Constant LR for comparison
    def constant_lr(t):
        return 0.01
    
    # One-cycle training
    w_oc = np.zeros(d)
    oc_losses = [loss(w_oc)]
    oc_lrs = []
    
    for t in range(T):
        lr = one_cycle_lr(t)
        oc_lrs.append(lr)
        w_oc = w_oc - lr * gradient(w_oc)
        oc_losses.append(loss(w_oc))
    
    # Constant LR training
    w_const = np.zeros(d)
    const_losses = [loss(w_const)]
    
    for t in range(T):
        lr = constant_lr(t)
        w_const = w_const - lr * gradient(w_const)
        const_losses.append(loss(w_const))
    
    print("One-cycle: warmup → high LR → anneal to very low")
    print(f"\n{'Step':>5} {'LR':>10} {'One-Cycle':>15} {'Constant':>15}")
    print("-" * 50)
    
    for t in [0, 50, 100, 150, 200]:
        lr = one_cycle_lr(min(t, T-1)) if t < T else oc_lrs[-1]
        print(f"{t:>5} {lr:>10.4f} {oc_losses[t]:>15.6f} {const_losses[t]:>15.6f}")
    
    print("\nOne-cycle can achieve faster convergence!")


def example_gradient_clipping():
    """Gradient clipping for stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Gradient Clipping")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Problem with potentially large gradients
    def gradient(w):
        return 100 * w + 10 * np.random.randn(len(w))  # Large gradient
    
    def clip_by_norm(g, max_norm):
        norm = np.linalg.norm(g)
        if norm > max_norm:
            return g * max_norm / norm
        return g
    
    def clip_by_value(g, clip_value):
        return np.clip(g, -clip_value, clip_value)
    
    d = 5
    w_init = np.ones(d)
    
    # Simulate one step
    g = gradient(w_init)
    
    print(f"Original gradient: {np.round(g, 2)}")
    print(f"||g|| = {np.linalg.norm(g):.2f}")
    
    print("\nGradient clipping methods:")
    
    # Clip by norm
    max_norm = 10
    g_norm = clip_by_norm(g, max_norm)
    print(f"\n1. Clip by norm (max_norm={max_norm}):")
    print(f"   {np.round(g_norm, 2)}")
    print(f"   ||g|| = {np.linalg.norm(g_norm):.2f}")
    
    # Clip by value
    clip_value = 20
    g_value = clip_by_value(g, clip_value)
    print(f"\n2. Clip by value (clip_value={clip_value}):")
    print(f"   {np.round(g_value, 2)}")
    print(f"   ||g|| = {np.linalg.norm(g_value):.2f}")
    
    print("\nClip-by-norm preserves direction, clip-by-value doesn't")


def example_lookahead():
    """Lookahead optimizer wrapper."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Lookahead Optimizer")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 100, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def gradient(w):
        return X.T @ (X @ w - y) / n
    
    # Lookahead wraps any optimizer
    # Fast weights (inner loop) + slow weights (outer loop)
    
    eta = 0.1  # Inner learning rate
    alpha = 0.5  # Slow weights step size
    k = 5  # Synchronization period
    
    # Slow weights
    w_slow = np.zeros(d)
    # Fast weights
    w_fast = np.zeros(d)
    
    la_losses = []
    
    for t in range(100):
        # Inner loop: standard SGD on fast weights
        w_fast = w_fast - eta * gradient(w_fast)
        
        # Synchronization every k steps
        if (t + 1) % k == 0:
            # Update slow weights
            w_slow = w_slow + alpha * (w_fast - w_slow)
            # Reset fast weights
            w_fast = w_slow.copy()
        
        la_losses.append(loss(w_slow))
    
    # Standard Adam for comparison
    w_adam = np.zeros(d)
    m, v = np.zeros(d), np.zeros(d)
    beta1, beta2 = 0.9, 0.999
    
    adam_losses = []
    for t in range(1, 101):
        g = gradient(w_adam)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        w_adam = w_adam - 0.1 * (m / (1-beta1**t)) / (np.sqrt(v / (1-beta2**t)) + 1e-8)
        adam_losses.append(loss(w_adam))
    
    print("Lookahead: maintains slow and fast weights")
    print(f"  Inner LR: {eta}, Outer LR: {alpha}, Sync period: {k}")
    
    print(f"\n{'Step':>5} {'Lookahead':>15} {'Adam':>15}")
    print("-" * 40)
    
    for t in [9, 24, 49, 74, 99]:
        print(f"{t+1:>5} {la_losses[t]:>15.6f} {adam_losses[t]:>15.6f}")


if __name__ == "__main__":
    example_momentum()
    example_nesterov()
    example_adagrad()
    example_rmsprop()
    example_adam()
    example_bias_correction()
    example_adamw()
    example_lr_schedules()
    example_optimizer_comparison()
    example_one_cycle()
    example_gradient_clipping()
    example_lookahead()
