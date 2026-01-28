"""
Stochastic Optimization - Examples
==================================
Implementations of stochastic optimization methods.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


def example_sgd_basics():
    """Basic SGD implementation and comparison with GD."""
    print("=" * 60)
    print("EXAMPLE 1: SGD Basics")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate linear regression data
    n, d = 1000, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def full_gradient(w):
        return X.T @ (X @ w - y) / n
    
    def stochastic_gradient(w, idx):
        return X[idx].T * (X[idx] @ w - y[idx])
    
    # Full gradient descent
    w_gd = np.zeros(d)
    eta_gd = 0.1
    gd_losses = []
    
    for _ in range(100):
        gd_losses.append(loss(w_gd))
        w_gd = w_gd - eta_gd * full_gradient(w_gd)
    
    # SGD
    w_sgd = np.zeros(d)
    eta_sgd = 0.01
    sgd_losses = []
    
    for t in range(100 * n):  # Same number of gradient evaluations
        if t % n == 0:
            sgd_losses.append(loss(w_sgd))
        idx = np.random.randint(n)
        w_sgd = w_sgd - eta_sgd * stochastic_gradient(w_sgd, idx)
    
    print(f"Linear regression: n={n}, d={d}")
    print(f"\n{'Epoch':>6} {'GD Loss':>15} {'SGD Loss':>15}")
    print("-" * 40)
    
    for i in [0, 10, 50, 99]:
        print(f"{i:>6} {gd_losses[i]:>15.6f} {sgd_losses[min(i, len(sgd_losses)-1)]:>15.6f}")
    
    print(f"\nGD: {n} gradient evals per epoch")
    print(f"SGD: 1 gradient eval per step, {n} steps per epoch")


def example_minibatch_variance():
    """Effect of mini-batch size on gradient variance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Mini-batch Variance Reduction")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 1000, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.5 * np.random.randn(n)
    
    w = np.random.randn(d)  # Fixed point for variance calculation
    
    # True gradient
    true_grad = X.T @ (X @ w - y) / n
    
    def minibatch_gradient(w, batch_size):
        idx = np.random.choice(n, batch_size, replace=False)
        return X[idx].T @ (X[idx] @ w - y[idx]) / batch_size
    
    print(f"Gradient variance at a fixed point")
    print(f"\n{'Batch Size':>12} {'Variance':>15} {'Reduction':>15}")
    print("-" * 45)
    
    variances = {}
    for batch_size in [1, 5, 10, 50, 100, 500]:
        # Estimate variance with many samples
        grads = [minibatch_gradient(w, batch_size) for _ in range(1000)]
        variance = np.mean([np.linalg.norm(g - true_grad)**2 for g in grads])
        variances[batch_size] = variance
        
        reduction = variances[1] / variance if batch_size > 1 else 1.0
        print(f"{batch_size:>12} {variance:>15.6f} {reduction:>15.2f}x")
    
    print(f"\nVariance ∝ 1/batch_size (approximately)")


def example_learning_rate_schedules():
    """Different learning rate schedules."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Learning Rate Schedules")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 500, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def stochastic_gradient(w, idx):
        return X[idx].T * (X[idx] @ w - y[idx])
    
    schedules = {
        'Constant': lambda t: 0.01,
        '1/t': lambda t: 0.1 / (1 + 0.01 * t),
        '1/sqrt(t)': lambda t: 0.1 / np.sqrt(1 + t),
        'Step decay': lambda t: 0.05 * (0.5 ** (t // 2000)),
        'Cosine': lambda t: 0.001 + 0.049 * (1 + np.cos(np.pi * t / 10000)) / 2
    }
    
    results = {}
    T = 10000
    
    for name, schedule in schedules.items():
        w = np.zeros(d)
        losses = []
        
        for t in range(T):
            if t % 500 == 0:
                losses.append(loss(w))
            
            eta = schedule(t)
            idx = np.random.randint(n)
            w = w - eta * stochastic_gradient(w, idx)
        
        results[name] = losses
    
    print(f"Loss after T iterations:")
    print(f"\n{'Schedule':>15}", end="")
    for t in [0, 2500, 5000, 7500, 9500]:
        print(f"{t:>10}", end="")
    print()
    print("-" * 70)
    
    for name, losses in results.items():
        print(f"{name:>15}", end="")
        for i in [0, 5, 10, 15, 19]:
            if i < len(losses):
                print(f"{losses[i]:>10.4f}", end="")
        print()


def example_svrg():
    """SVRG implementation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: SVRG (Variance Reduced SGD)")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 500, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def full_gradient(w):
        return X.T @ (X @ w - y) / n
    
    def stochastic_gradient(w, idx):
        return X[idx].T * (X[idx] @ w - y[idx])
    
    # Standard SGD
    w_sgd = np.zeros(d)
    eta_sgd = 0.01
    sgd_losses = [loss(w_sgd)]
    
    for epoch in range(20):
        for _ in range(n):
            idx = np.random.randint(n)
            w_sgd = w_sgd - eta_sgd * stochastic_gradient(w_sgd, idx)
        sgd_losses.append(loss(w_sgd))
    
    # SVRG
    w_svrg = np.zeros(d)
    eta_svrg = 0.1
    m = 2 * n  # Inner loop size
    svrg_losses = [loss(w_svrg)]
    
    for epoch in range(20):
        # Compute full gradient at snapshot
        w_snapshot = w_svrg.copy()
        mu = full_gradient(w_snapshot)
        
        # Inner loop
        for _ in range(m):
            idx = np.random.randint(n)
            g_i = stochastic_gradient(w_svrg, idx)
            g_i_snapshot = stochastic_gradient(w_snapshot, idx)
            
            # Variance-reduced gradient
            g = g_i - g_i_snapshot + mu
            w_svrg = w_svrg - eta_svrg * g
        
        svrg_losses.append(loss(w_svrg))
    
    print(f"SGD vs SVRG convergence:")
    print(f"\n{'Epoch':>6} {'SGD Loss':>15} {'SVRG Loss':>15}")
    print("-" * 40)
    
    for i in [0, 5, 10, 15, 20]:
        print(f"{i:>6} {sgd_losses[i]:>15.6e} {svrg_losses[i]:>15.6e}")
    
    print(f"\nSVRG achieves linear convergence!")


def example_saga():
    """SAGA implementation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: SAGA")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 200, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    # SAGA
    w = np.zeros(d)
    eta = 0.05
    
    # Store gradients for each sample
    stored_grads = np.zeros((n, d))
    grad_sum = np.zeros(d)
    
    # Initialize stored gradients
    for i in range(n):
        stored_grads[i] = X[i].T * (X[i] @ w - y[i])
        grad_sum += stored_grads[i]
    
    losses = [loss(w)]
    
    for epoch in range(20):
        for _ in range(n):
            # Sample uniformly
            i = np.random.randint(n)
            
            # Compute new gradient
            new_grad = X[i].T * (X[i] @ w - y[i])
            
            # SAGA gradient
            g = new_grad - stored_grads[i] + grad_sum / n
            
            # Update stored gradient
            grad_sum += new_grad - stored_grads[i]
            stored_grads[i] = new_grad
            
            # Update w
            w = w - eta * g
        
        losses.append(loss(w))
    
    print(f"SAGA convergence:")
    print(f"\n{'Epoch':>6} {'Loss':>15}")
    print("-" * 25)
    
    for i in [0, 5, 10, 15, 20]:
        print(f"{i:>6} {losses[i]:>15.6e}")
    
    print(f"\nStorage: O(nd) = O({n*d})")


def example_importance_sampling():
    """Importance sampling in SGD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Importance Sampling")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 500, 5
    X = np.random.randn(n, d)
    # Make some samples have larger gradients
    X[:50] *= 5
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    # Compute Lipschitz constants (approximation)
    L = np.array([np.linalg.norm(X[i])**2 for i in range(n)])
    p = L / L.sum()  # Importance sampling probabilities
    
    # Uniform SGD
    w_uniform = np.zeros(d)
    eta = 0.001
    uniform_losses = [loss(w_uniform)]
    
    for epoch in range(30):
        for _ in range(n):
            i = np.random.randint(n)
            g = X[i].T * (X[i] @ w_uniform - y[i])
            w_uniform = w_uniform - eta * g
        uniform_losses.append(loss(w_uniform))
    
    # Importance sampling SGD
    w_is = np.zeros(d)
    is_losses = [loss(w_is)]
    
    for epoch in range(30):
        for _ in range(n):
            i = np.random.choice(n, p=p)
            # Reweight gradient
            g = (X[i].T * (X[i] @ w_is - y[i])) / (n * p[i])
            w_is = w_is - eta * g
        is_losses.append(loss(w_is))
    
    print(f"Uniform vs Importance Sampling:")
    print(f"\n{'Epoch':>6} {'Uniform':>15} {'Importance':>15}")
    print("-" * 40)
    
    for i in [0, 5, 10, 20, 30]:
        print(f"{i:>6} {uniform_losses[i]:>15.6f} {is_losses[i]:>15.6f}")
    
    print(f"\nImportance sampling helps when samples have varying 'importance'")


def example_sgd_momentum():
    """SGD with momentum."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: SGD with Momentum")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 500, 10
    X = np.random.randn(n, d)
    # Ill-conditioned problem
    X = X @ np.diag(np.linspace(1, 100, d))
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def stochastic_gradient(w, idx):
        return X[idx].T * (X[idx] @ w - y[idx])
    
    # Vanilla SGD
    w_sgd = np.zeros(d)
    eta = 0.0001
    sgd_losses = [loss(w_sgd)]
    
    for epoch in range(50):
        for _ in range(n):
            idx = np.random.randint(n)
            w_sgd = w_sgd - eta * stochastic_gradient(w_sgd, idx)
        sgd_losses.append(loss(w_sgd))
    
    # SGD with momentum
    w_mom = np.zeros(d)
    v = np.zeros(d)
    beta = 0.9
    mom_losses = [loss(w_mom)]
    
    for epoch in range(50):
        for _ in range(n):
            idx = np.random.randint(n)
            g = stochastic_gradient(w_mom, idx)
            v = beta * v + g
            w_mom = w_mom - eta * v
        mom_losses.append(loss(w_mom))
    
    print(f"Ill-conditioned problem")
    print(f"\n{'Epoch':>6} {'SGD':>15} {'SGD+Momentum':>15}")
    print("-" * 40)
    
    for i in [0, 10, 25, 40, 50]:
        print(f"{i:>6} {sgd_losses[i]:>15.4f} {mom_losses[i]:>15.4f}")
    
    print(f"\nMomentum helps on ill-conditioned problems")


def example_parallel_sgd():
    """Simulate parallel SGD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Parallel SGD (Simulated)")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 1000, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def compute_gradient_batch(w, indices):
        return X[indices].T @ (X[indices] @ w - y[indices]) / len(indices)
    
    # Sequential SGD
    w_seq = np.zeros(d)
    eta = 0.1
    seq_losses = [loss(w_seq)]
    
    for epoch in range(20):
        g = compute_gradient_batch(w_seq, np.arange(n))
        w_seq = w_seq - eta * g
        seq_losses.append(loss(w_seq))
    
    # Synchronous parallel SGD (4 workers)
    num_workers = 4
    w_sync = np.zeros(d)
    sync_losses = [loss(w_sync)]
    
    for epoch in range(20):
        # Each worker computes gradient on n/4 samples
        gradients = []
        for worker in range(num_workers):
            start = worker * (n // num_workers)
            end = start + (n // num_workers)
            g = compute_gradient_batch(w_sync, np.arange(start, end))
            gradients.append(g)
        
        # Average gradients
        avg_g = np.mean(gradients, axis=0)
        w_sync = w_sync - eta * avg_g
        sync_losses.append(loss(w_sync))
    
    # Local SGD (4 workers, sync every 5 steps)
    w_local = np.zeros(d)
    local_losses = [loss(w_local)]
    sync_interval = 5
    
    for epoch in range(20):
        # Local models
        workers_w = [w_local.copy() for _ in range(num_workers)]
        
        for step in range(sync_interval):
            for worker in range(num_workers):
                start = worker * (n // num_workers)
                end = start + (n // num_workers)
                idx = np.random.randint(start, end)
                g = X[idx].T * (X[idx] @ workers_w[worker] - y[idx])
                workers_w[worker] = workers_w[worker] - 0.01 * g
        
        # Synchronize: average models
        w_local = np.mean(workers_w, axis=0)
        local_losses.append(loss(w_local))
    
    print(f"Parallel SGD strategies ({num_workers} workers):")
    print(f"\n{'Epoch':>6} {'Sequential':>15} {'Sync-SGD':>15} {'Local-SGD':>15}")
    print("-" * 55)
    
    for i in [0, 5, 10, 15, 20]:
        print(f"{i:>6} {seq_losses[i]:>15.6f} {sync_losses[i]:>15.6f} {local_losses[i]:>15.6f}")


def example_noise_and_generalization():
    """SGD noise and generalization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: SGD Noise and Generalization")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate polynomial regression data
    n_train, n_test = 50, 200
    
    X_train = np.random.uniform(-1, 1, n_train)
    y_train = np.sin(3 * X_train) + 0.2 * np.random.randn(n_train)
    
    X_test = np.random.uniform(-1, 1, n_test)
    y_test = np.sin(3 * X_test)
    
    # Create polynomial features (degree 15 - overparameterized)
    degree = 15
    
    def poly_features(x):
        return np.column_stack([x**i for i in range(degree + 1)])
    
    Phi_train = poly_features(X_train)
    Phi_test = poly_features(X_test)
    
    def train_loss(w):
        return 0.5 * np.mean((Phi_train @ w - y_train)**2)
    
    def test_loss(w):
        return 0.5 * np.mean((Phi_test @ w - y_test)**2)
    
    results = {}
    
    # Different batch sizes
    for batch_size in [1, 5, 10, 50]:
        w = np.zeros(degree + 1)
        eta = 0.01 if batch_size == 1 else 0.1
        
        train_losses = []
        test_losses = []
        
        for epoch in range(200):
            # Shuffle
            perm = np.random.permutation(n_train)
            
            for start in range(0, n_train, batch_size):
                idx = perm[start:start+batch_size]
                g = Phi_train[idx].T @ (Phi_train[idx] @ w - y_train[idx]) / len(idx)
                w = w - eta * g
            
            if epoch % 20 == 0:
                train_losses.append(train_loss(w))
                test_losses.append(test_loss(w))
        
        results[batch_size] = (train_losses[-1], test_losses[-1])
    
    print(f"Overparameterized model (degree {degree}, n={n_train})")
    print(f"\n{'Batch Size':>12} {'Train Loss':>15} {'Test Loss':>15}")
    print("-" * 45)
    
    for bs, (tr, te) in results.items():
        print(f"{bs:>12} {tr:>15.4f} {te:>15.4f}")
    
    print("\nSmaller batches (more noise) may generalize better!")


def example_gradient_noise_scale():
    """Gradient noise scale analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Gradient Noise Scale")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 1000, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    w = np.zeros(d)  # Fixed point for analysis
    
    # Full gradient
    full_grad = X.T @ (X @ w - y) / n
    
    # Compute gradient variance for different batch sizes
    print("Gradient Noise Scale = η × Var(g) / batch_size")
    print("\nFor fixed learning rate η = 0.01:")
    print(f"\n{'Batch':>8} {'Var(g)':>15} {'Noise Scale':>15}")
    print("-" * 45)
    
    eta = 0.01
    for batch_size in [1, 10, 50, 100, 500, 1000]:
        variances = []
        for _ in range(500):
            idx = np.random.choice(n, batch_size, replace=False)
            g = X[idx].T @ (X[idx] @ w - y[idx]) / batch_size
            variances.append(np.linalg.norm(g - full_grad)**2)
        
        var_g = np.mean(variances)
        noise_scale = eta * var_g / batch_size
        
        print(f"{batch_size:>8} {var_g:>15.6f} {noise_scale:>15.6e}")
    
    print("\nNoise scale controls implicit regularization")
    print("Same noise scale with different (η, batch) pairs → similar dynamics")


def example_shuffling_strategies():
    """Compare different data shuffling strategies."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Shuffling Strategies")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 200, 5
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    def sgd_step(w, idx, eta):
        g = X[idx].T * (X[idx] @ w - y[idx])
        return w - eta * g
    
    strategies = {
        'With replacement': 'replace',
        'Without replacement (reshuffle)': 'reshuffle',
        'Without replacement (fixed order)': 'fixed'
    }
    
    results = {}
    eta = 0.01
    epochs = 30
    
    for name, strategy in strategies.items():
        np.random.seed(42)
        w = np.zeros(d)
        losses = [loss(w)]
        
        if strategy == 'fixed':
            fixed_order = np.random.permutation(n)
        
        for epoch in range(epochs):
            if strategy == 'replace':
                # Sample with replacement
                for _ in range(n):
                    idx = np.random.randint(n)
                    w = sgd_step(w, idx, eta)
            elif strategy == 'reshuffle':
                # Shuffle each epoch
                perm = np.random.permutation(n)
                for idx in perm:
                    w = sgd_step(w, idx, eta)
            else:  # fixed
                for idx in fixed_order:
                    w = sgd_step(w, idx, eta)
            
            losses.append(loss(w))
        
        results[name] = losses
    
    print(f"{'Epoch':>6}", end="")
    for name in strategies.keys():
        print(f"{name[:15]:>18}", end="")
    print()
    print("-" * 65)
    
    for i in [0, 10, 20, 30]:
        print(f"{i:>6}", end="")
        for name in strategies.keys():
            print(f"{results[name][i]:>18.6f}", end="")
        print()
    
    print("\nReshuffling each epoch often works best in practice")


def example_gradient_accumulation():
    """Gradient accumulation for large effective batch size."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Gradient Accumulation")
    print("=" * 60)
    
    np.random.seed(42)
    
    n, d = 500, 10
    X = np.random.randn(n, d)
    w_true = np.random.randn(d)
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def loss(w):
        return 0.5 * np.mean((X @ w - y)**2)
    
    # Simulate memory constraint: can only compute gradients for 10 samples at a time
    micro_batch = 10
    accumulation_steps = 5  # Effective batch = 50
    effective_batch = micro_batch * accumulation_steps
    
    # Without accumulation (small batch)
    w_small = np.zeros(d)
    eta = 0.05
    small_losses = [loss(w_small)]
    
    for epoch in range(20):
        perm = np.random.permutation(n)
        for start in range(0, n, micro_batch):
            idx = perm[start:start+micro_batch]
            g = X[idx].T @ (X[idx] @ w_small - y[idx]) / micro_batch
            w_small = w_small - eta * g
        small_losses.append(loss(w_small))
    
    # With accumulation (larger effective batch)
    w_accum = np.zeros(d)
    accum_losses = [loss(w_accum)]
    
    for epoch in range(20):
        perm = np.random.permutation(n)
        for start in range(0, n, effective_batch):
            accumulated_g = np.zeros(d)
            
            for step in range(accumulation_steps):
                micro_start = start + step * micro_batch
                if micro_start >= n:
                    break
                idx = perm[micro_start:micro_start+micro_batch]
                g = X[idx].T @ (X[idx] @ w_accum - y[idx]) / micro_batch
                accumulated_g += g
            
            accumulated_g /= accumulation_steps
            w_accum = w_accum - eta * accumulated_g
        
        accum_losses.append(loss(w_accum))
    
    print(f"Micro-batch: {micro_batch}, Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch}")
    
    print(f"\n{'Epoch':>6} {'Small batch':>15} {'Accumulated':>15}")
    print("-" * 40)
    
    for i in [0, 5, 10, 15, 20]:
        print(f"{i:>6} {small_losses[i]:>15.6f} {accum_losses[i]:>15.6f}")
    
    print("\nGradient accumulation achieves larger batch effect")
    print("with limited memory")


if __name__ == "__main__":
    example_sgd_basics()
    example_minibatch_variance()
    example_learning_rate_schedules()
    example_svrg()
    example_saga()
    example_importance_sampling()
    example_sgd_momentum()
    example_parallel_sgd()
    example_noise_and_generalization()
    example_gradient_noise_scale()
    example_shuffling_strategies()
    example_gradient_accumulation()
