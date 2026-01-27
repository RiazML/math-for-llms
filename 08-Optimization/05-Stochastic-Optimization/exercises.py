"""
Stochastic Optimization - Exercises
===================================
Practice problems for stochastic optimization.
"""

import numpy as np


class StochasticOptimizationExercises:
    """Exercises for stochastic optimization methods."""
    
    def exercise_1_sgd_unbiasedness(self):
        """
        Exercise 1: Verify SGD Unbiasedness
        
        Show that stochastic gradient is unbiased.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: SGD Unbiasedness")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 100, 5
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        w = np.random.randn(d)  # Test point
        
        # True (full) gradient
        full_grad = X.T @ (X @ w - y) / n
        
        # Estimate E[stochastic gradient] by averaging many samples
        n_samples = 10000
        stochastic_grads = []
        
        for _ in range(n_samples):
            i = np.random.randint(n)
            g_i = X[i].T * (X[i] @ w - y[i])
            stochastic_grads.append(g_i)
        
        estimated_expectation = np.mean(stochastic_grads, axis=0)
        
        print("E[∇f_i(w)] should equal ∇F(w)")
        print(f"\nFull gradient: {np.round(full_grad, 6)}")
        print(f"E[stochastic]:  {np.round(estimated_expectation, 6)}")
        print(f"\nDifference: {np.linalg.norm(full_grad - estimated_expectation):.6f}")
        print("\nStochastic gradient is unbiased ✓")
    
    def exercise_2_variance_analysis(self):
        """
        Exercise 2: Analyze Gradient Variance
        
        Compute and analyze variance of stochastic gradients.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Gradient Variance Analysis")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 100, 5
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + np.random.randn(n) * 0.5  # More noise
        
        w = np.zeros(d)
        
        # Full gradient
        full_grad = X.T @ (X @ w - y) / n
        
        # Compute variance for each sample
        individual_variances = []
        for i in range(n):
            g_i = X[i].T * (X[i] @ w - y[i])
            individual_variances.append(np.linalg.norm(g_i - full_grad)**2)
        
        total_variance = np.mean(individual_variances)
        
        print("Gradient variance = E[||∇f_i - ∇F||²]")
        print(f"\nEstimated variance σ² = {total_variance:.4f}")
        
        # Verify mini-batch variance reduction
        print("\nMini-batch variance reduction:")
        print(f"{'Batch Size':>12} {'Var(g)':>15} {'Var/b':>15}")
        print("-" * 45)
        
        for b in [1, 5, 10, 20, 50]:
            batch_variances = []
            for _ in range(1000):
                idx = np.random.choice(n, b, replace=False)
                g_batch = X[idx].T @ (X[idx] @ w - y[idx]) / b
                batch_variances.append(np.linalg.norm(g_batch - full_grad)**2)
            
            batch_var = np.mean(batch_variances)
            print(f"{b:>12} {batch_var:>15.4f} {total_variance/b:>15.4f}")
        
        print("\nVar(mini-batch) ≈ σ²/b")
    
    def exercise_3_convergence_rate(self):
        """
        Exercise 3: Measure Convergence Rate
        
        Compare SGD convergence on convex vs strongly convex.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Convergence Rate")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 200, 5
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        
        # Strongly convex (with regularization)
        lam = 0.1
        
        def loss_sc(w):
            return 0.5 * np.mean((X @ w - X @ w_true)**2) + 0.5 * lam * np.sum(w**2)
        
        # SGD with decreasing step size
        w = np.zeros(d)
        errors_sc = []
        
        for t in range(1, 5001):
            idx = np.random.randint(n)
            g = X[idx].T * (X[idx] @ w - X[idx] @ w_true) + lam * w
            eta = 1.0 / (lam * t)  # O(1/t) for strongly convex
            w = w - eta * g
            
            if t % 500 == 0:
                errors_sc.append((t, np.linalg.norm(w - w_true)**2))
        
        print("Strongly convex (μ = 0.1): E[||w - w*||²] = O(1/T)")
        print(f"\n{'T':>8} {'||w - w*||²':>15} {'Theory O(1/T)':>15}")
        print("-" * 45)
        
        C = errors_sc[0][1] * errors_sc[0][0]  # Estimate constant
        for t, err in errors_sc:
            theory = C / t
            print(f"{t:>8} {err:>15.6f} {theory:>15.6f}")
    
    def exercise_4_implement_svrg(self):
        """
        Exercise 4: Implement SVRG
        
        Implement variance-reduced SGD.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: SVRG Implementation")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 300, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        def full_gradient(w):
            return X.T @ (X @ w - y) / n
        
        # SVRG
        w = np.zeros(d)
        eta = 0.1
        m = 2 * n  # Inner loop iterations
        
        print("SVRG Algorithm:")
        print(f"  Inner loop m = {m}")
        print(f"  Step size η = {eta}")
        
        print(f"\n{'Outer Iter':>12} {'Loss':>15} {'||w - w*||':>15}")
        print("-" * 45)
        
        for s in range(10):
            # Full gradient at snapshot
            w_tilde = w.copy()
            mu = full_gradient(w_tilde)
            
            print(f"{s:>12} {loss(w):>15.6e} {np.linalg.norm(w - w_true):>15.6e}")
            
            # Inner loop
            for t in range(m):
                i = np.random.randint(n)
                
                # Variance-reduced gradient
                g_i = X[i].T * (X[i] @ w - y[i])
                g_i_tilde = X[i].T * (X[i] @ w_tilde - y[i])
                g = g_i - g_i_tilde + mu
                
                w = w - eta * g
        
        print(f"\nFinal loss: {loss(w):.6e}")
        print("SVRG achieves linear convergence!")
    
    def exercise_5_learning_rate_finder(self):
        """
        Exercise 5: Learning Rate Finder
        
        Implement learning rate range test.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Learning Rate Finder")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 500, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        def stochastic_gradient(w, batch_idx):
            return X[batch_idx].T @ (X[batch_idx] @ w - y[batch_idx]) / len(batch_idx)
        
        print("Learning Rate Range Test:")
        print("  Increase LR exponentially while tracking loss")
        
        # LR range test
        w = np.zeros(d)
        batch_size = 32
        
        lr_min, lr_max = 1e-5, 1.0
        num_steps = 100
        
        lrs = np.exp(np.linspace(np.log(lr_min), np.log(lr_max), num_steps))
        losses = []
        
        for i, eta in enumerate(lrs):
            batch_idx = np.random.choice(n, batch_size, replace=False)
            g = stochastic_gradient(w, batch_idx)
            w = w - eta * g
            losses.append(loss(w))
        
        # Find best LR (steepest descent)
        smoothed_losses = np.convolve(losses, np.ones(5)/5, mode='valid')
        gradients = np.diff(smoothed_losses)
        best_idx = np.argmin(gradients) + 2
        best_lr = lrs[best_idx]
        
        print(f"\n{'LR':>12} {'Loss':>15}")
        print("-" * 30)
        
        for i in [0, 25, 50, 75, 99]:
            print(f"{lrs[i]:>12.6f} {losses[i]:>15.4f}")
        
        print(f"\nSuggested LR: {best_lr:.6f}")
        print("(Where loss decreases fastest)")
    
    def exercise_6_mini_batch_optimal(self):
        """
        Exercise 6: Optimal Batch Size
        
        Find optimal batch size for given compute budget.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Optimal Batch Size")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 1000, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        # Fixed compute budget: total gradient evaluations
        budget = 10000  # e.g., 10 epochs with n=1000
        
        print(f"Fixed compute budget: {budget} gradient evaluations")
        print("\nCompare different batch sizes:")
        print(f"{'Batch':>8} {'Steps':>8} {'Final Loss':>15}")
        print("-" * 35)
        
        results = {}
        for batch_size in [1, 10, 50, 100, 250, 500, 1000]:
            np.random.seed(42)
            w = np.zeros(d)
            
            num_steps = budget // batch_size
            eta = 0.1 * np.sqrt(batch_size / 100)  # Scale LR with batch
            
            for _ in range(num_steps):
                idx = np.random.choice(n, batch_size, replace=False)
                g = X[idx].T @ (X[idx] @ w - y[idx]) / batch_size
                w = w - eta * g
            
            final_loss = loss(w)
            results[batch_size] = final_loss
            print(f"{batch_size:>8} {num_steps:>8} {final_loss:>15.6f}")
        
        best_batch = min(results, key=results.get)
        print(f"\nBest batch size: {best_batch}")
    
    def exercise_7_importance_weights(self):
        """
        Exercise 7: Compute Importance Weights
        
        Derive and apply importance sampling.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Importance Sampling Weights")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 200, 5
        X = np.random.randn(n, d)
        # Make some samples have larger Lipschitz constant
        X[:20] *= 5
        
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        # Compute Lipschitz constants
        L_i = np.array([np.linalg.norm(X[i])**2 for i in range(n)])
        
        print("Lipschitz constants L_i = ||x_i||²:")
        print(f"  Min: {L_i.min():.4f}")
        print(f"  Max: {L_i.max():.4f}")
        print(f"  Ratio: {L_i.max()/L_i.min():.2f}")
        
        # Importance weights
        p = L_i / L_i.sum()
        
        print("\nImportance weights p_i ∝ L_i:")
        print(f"  Min: {p.min():.6f}")
        print(f"  Max: {p.max():.6f}")
        
        # Compare variance
        w = np.zeros(d)
        full_grad = X.T @ (X @ w - y) / n
        
        # Uniform sampling variance
        uniform_vars = []
        for _ in range(1000):
            i = np.random.randint(n)
            g = X[i].T * (X[i] @ w - y[i])
            uniform_vars.append(np.linalg.norm(g - full_grad)**2)
        
        # Importance sampling variance
        is_vars = []
        for _ in range(1000):
            i = np.random.choice(n, p=p)
            g = (X[i].T * (X[i] @ w - y[i])) / (n * p[i])
            is_vars.append(np.linalg.norm(g - full_grad)**2)
        
        print(f"\nGradient variance comparison:")
        print(f"  Uniform:    {np.mean(uniform_vars):.4f}")
        print(f"  Importance: {np.mean(is_vars):.4f}")
        print(f"  Reduction:  {np.mean(uniform_vars)/np.mean(is_vars):.2f}x")
    
    def exercise_8_sgd_momentum(self):
        """
        Exercise 8: Implement SGD with Momentum
        
        Implement and analyze momentum.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: SGD with Momentum")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 500, 10
        X = np.random.randn(n, d)
        # Ill-conditioned
        X = X @ np.diag(np.linspace(1, 50, d))
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        eta = 0.0001
        
        # Vanilla SGD
        w_sgd = np.zeros(d)
        sgd_losses = []
        
        for epoch in range(30):
            for _ in range(n):
                i = np.random.randint(n)
                g = X[i].T * (X[i] @ w_sgd - y[i])
                w_sgd = w_sgd - eta * g
            sgd_losses.append(loss(w_sgd))
        
        # SGD + Momentum
        w_mom = np.zeros(d)
        v = np.zeros(d)
        beta = 0.9
        mom_losses = []
        
        for epoch in range(30):
            for _ in range(n):
                i = np.random.randint(n)
                g = X[i].T * (X[i] @ w_mom - y[i])
                v = beta * v + g
                w_mom = w_mom - eta * v
            mom_losses.append(loss(w_mom))
        
        # Nesterov Momentum
        w_nes = np.zeros(d)
        v_nes = np.zeros(d)
        nes_losses = []
        
        for epoch in range(30):
            for _ in range(n):
                i = np.random.randint(n)
                w_look = w_nes - beta * v_nes  # Look ahead
                g = X[i].T * (X[i] @ w_look - y[i])
                v_nes = beta * v_nes + eta * g
                w_nes = w_nes - v_nes
            nes_losses.append(loss(w_nes))
        
        print(f"Ill-conditioned problem (κ ≈ 50)")
        print(f"\n{'Epoch':>6} {'SGD':>12} {'Momentum':>12} {'Nesterov':>12}")
        print("-" * 45)
        
        for i in [0, 10, 20, 29]:
            print(f"{i:>6} {sgd_losses[i]:>12.4f} {mom_losses[i]:>12.4f} {nes_losses[i]:>12.4f}")
    
    def exercise_9_local_sgd(self):
        """
        Exercise 9: Local SGD Simulation
        
        Simulate federated/local SGD.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Local SGD")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 400, 5
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        num_workers = 4
        local_n = n // num_workers
        eta = 0.01
        
        # Split data among workers
        worker_data = [(X[i*local_n:(i+1)*local_n], 
                       y[i*local_n:(i+1)*local_n]) 
                      for i in range(num_workers)]
        
        print(f"Local SGD with {num_workers} workers")
        print(f"Data per worker: {local_n}")
        
        # Compare sync intervals
        for H in [1, 5, 10, 20]:
            np.random.seed(42)
            w = np.zeros(d)
            losses = []
            
            for epoch in range(20):
                # Each worker runs H local steps
                local_ws = [w.copy() for _ in range(num_workers)]
                
                for _ in range(H):
                    for k in range(num_workers):
                        X_k, y_k = worker_data[k]
                        i = np.random.randint(local_n)
                        g = X_k[i].T * (X_k[i] @ local_ws[k] - y_k[i])
                        local_ws[k] = local_ws[k] - eta * g
                
                # Synchronize (average)
                w = np.mean(local_ws, axis=0)
                losses.append(loss(w))
            
            print(f"\nSync every H={H} steps: Final loss = {losses[-1]:.6f}")
    
    def exercise_10_noise_injection(self):
        """
        Exercise 10: Analyze Gradient Noise
        
        Study the effect of noise on optimization.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Gradient Noise Analysis")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Simple 2D problem with multiple minima
        def loss(w):
            # Non-convex: multiple local minima
            return (w[0]**2 - 1)**2 + w[1]**2 + 0.1*np.sin(5*w[0])
        
        def gradient(w):
            g0 = 4*w[0]*(w[0]**2 - 1) + 0.5*np.cos(5*w[0])
            g1 = 2*w[1]
            return np.array([g0, g1])
        
        print("Non-convex loss with noise injection")
        print("Starting from w = (-1.5, 0)")
        
        results = {}
        
        for noise_level in [0, 0.1, 0.5, 1.0]:
            np.random.seed(42)
            w = np.array([-1.5, 0.0])
            eta = 0.01
            
            trajectory = [w.copy()]
            
            for _ in range(500):
                g = gradient(w)
                # Add noise
                g_noisy = g + noise_level * np.random.randn(2)
                w = w - eta * g_noisy
                trajectory.append(w.copy())
            
            results[noise_level] = (w.copy(), loss(w))
        
        print(f"\n{'Noise':>8} {'Final w':>25} {'Loss':>15}")
        print("-" * 55)
        
        for noise, (w_final, l_final) in results.items():
            print(f"{noise:>8.2f} {str(np.round(w_final, 4)):>25} {l_final:>15.6f}")
        
        print("\nNoise can help escape local minima!")
        print("But too much noise hurts convergence")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = StochasticOptimizationExercises()
    
    print("STOCHASTIC OPTIMIZATION EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    exercises.solution_2()
    exercises.solution_3()
    exercises.solution_4()
    exercises.solution_5()
    exercises.solution_6()
    exercises.solution_7()
    exercises.solution_8()
    exercises.solution_9()
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
