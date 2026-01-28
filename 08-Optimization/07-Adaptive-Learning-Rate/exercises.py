"""
Adaptive Learning Rate Methods - Exercises
==========================================
Practice problems for adaptive optimizers.
"""

import numpy as np


class AdaptiveLearningRateExercises:
    """Exercises for adaptive learning rate methods."""
    
    def exercise_1_momentum_derivation(self):
        """
        Exercise 1: Momentum Effective Update
        
        Show that momentum accumulates gradient over time.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Momentum Effective Update")
        print("=" * 60)
        
        print("Momentum update: v_t = βv_{t-1} + g_t")
        print("\nExpanding recursively:")
        print("  v_t = βv_{t-1} + g_t")
        print("      = β(βv_{t-2} + g_{t-1}) + g_t")
        print("      = β²v_{t-2} + βg_{t-1} + g_t")
        print("      = β^t v_0 + Σᵢ β^{t-i} g_i")
        print("\nWith v_0 = 0:")
        print("  v_t = Σᵢ₌₁ᵗ β^{t-i} g_i")
        
        print("\nFor constant gradient g:")
        print("  v_∞ = g(1 + β + β² + ...) = g/(1-β)")
        print("  With β = 0.9: v_∞ = 10g")
        
        # Numerical verification
        beta = 0.9
        g = 1.0
        v = 0
        
        print("\nNumerical verification (g=1, β=0.9):")
        print(f"{'t':>3} {'v_t':>10} {'g/(1-β)':>10}")
        print("-" * 25)
        
        for t in range(1, 51):
            v = beta * v + g
            if t in [1, 5, 10, 20, 50]:
                print(f"{t:>3} {v:>10.4f} {g/(1-beta):>10.4f}")
        
        print("\nMomentum amplifies consistent gradients!")
    
    def exercise_2_adagrad_analysis(self):
        """
        Exercise 2: AdaGrad Learning Rate Decay
        
        Analyze AdaGrad's effective learning rate over time.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: AdaGrad Learning Rate Decay")
        print("=" * 60)
        
        print("AdaGrad: G_t = G_{t-1} + g_t²")
        print("Effective LR: η_eff = η / √(G_t + ε)")
        print("\nFor constant gradient g:")
        print("  G_t = t·g²")
        print("  η_eff ≈ η / (g√t)")
        
        eta = 1.0
        g = 1.0
        eps = 1e-8
        
        print(f"\nNumerical (η={eta}, g={g}):")
        print(f"{'t':>5} {'G_t':>10} {'η_eff':>10} {'η/(g√t)':>10}")
        print("-" * 40)
        
        G = 0
        for t in range(1, 101):
            G = G + g**2
            eta_eff = eta / (np.sqrt(G) + eps)
            eta_theory = eta / (g * np.sqrt(t))
            
            if t in [1, 10, 25, 50, 100]:
                print(f"{t:>5} {G:>10.1f} {eta_eff:>10.4f} {eta_theory:>10.4f}")
        
        print("\nAdaGrad LR decays as O(1/√t)")
        print("Problem: LR can become too small!")
    
    def exercise_3_rmsprop_vs_adagrad(self):
        """
        Exercise 3: RMSProp vs AdaGrad
        
        Compare the two methods.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: RMSProp vs AdaGrad")
        print("=" * 60)
        
        print("AdaGrad: G_t = G_{t-1} + g_t²")
        print("RMSProp: v_t = ρv_{t-1} + (1-ρ)g_t²")
        
        eta = 1.0
        g = 1.0
        rho = 0.9
        eps = 1e-8
        
        G_ada = 0
        v_rms = 0
        
        print(f"\nWith constant gradient g={g}, ρ={rho}:")
        print(f"{'t':>5} {'AdaGrad LR':>15} {'RMSProp LR':>15}")
        print("-" * 40)
        
        for t in range(1, 101):
            G_ada = G_ada + g**2
            v_rms = rho * v_rms + (1 - rho) * g**2
            
            lr_ada = eta / (np.sqrt(G_ada) + eps)
            lr_rms = eta / (np.sqrt(v_rms) + eps)
            
            if t in [1, 10, 25, 50, 100]:
                print(f"{t:>5} {lr_ada:>15.6f} {lr_rms:>15.6f}")
        
        print("\nRMSProp converges to stable LR: η/√((1-ρ)g²/(1-ρ)) = η/g")
        print(f"Theoretical RMSProp LR: {eta/g:.4f}")
        print("RMSProp prevents learning rate from decaying to zero!")
    
    def exercise_4_adam_bias(self):
        """
        Exercise 4: Adam Bias Correction
        
        Derive the bias correction formula.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Adam Bias Correction")
        print("=" * 60)
        
        print("First moment: m_t = β₁m_{t-1} + (1-β₁)g_t")
        print("\nFor constant gradient g, m_0 = 0:")
        print("  m_t = (1-β₁)Σᵢ₌₁ᵗ β₁^{t-i} g")
        print("      = (1-β₁)g · (1-β₁ᵗ)/(1-β₁)")
        print("      = g(1-β₁ᵗ)")
        
        print("\nExpected value: E[m_t] = (1-β₁ᵗ)E[g]")
        print("Bias: m_t underestimates E[g] by factor (1-β₁ᵗ)")
        print("Correction: m̂_t = m_t / (1-β₁ᵗ)")
        
        beta1 = 0.9
        g = 1.0
        m = 0
        
        print(f"\nNumerical (β₁={beta1}, g={g}):")
        print(f"{'t':>4} {'m_t':>10} {'m̂_t':>10} {'E[g]':>10}")
        print("-" * 40)
        
        for t in range(1, 11):
            m = beta1 * m + (1 - beta1) * g
            m_hat = m / (1 - beta1**t)
            print(f"{t:>4} {m:>10.4f} {m_hat:>10.4f} {g:>10.4f}")
        
        print("\nBias correction ensures m̂_t ≈ E[g] from the start")
    
    def exercise_5_implement_adam(self):
        """
        Exercise 5: Implement Adam from Scratch
        
        Full Adam implementation.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Adam Implementation")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Problem setup
        n, d = 100, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        def gradient(w):
            return X.T @ (X @ w - y) / n
        
        class Adam:
            def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
                self.eta = eta
                self.beta1 = beta1
                self.beta2 = beta2
                self.eps = eps
                self.m = None
                self.v = None
                self.t = 0
            
            def step(self, w, g):
                if self.m is None:
                    self.m = np.zeros_like(w)
                    self.v = np.zeros_like(w)
                
                self.t += 1
                
                # Update moments
                self.m = self.beta1 * self.m + (1 - self.beta1) * g
                self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
                
                # Bias correction
                m_hat = self.m / (1 - self.beta1**self.t)
                v_hat = self.v / (1 - self.beta2**self.t)
                
                # Update
                return w - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Training
        w = np.zeros(d)
        optimizer = Adam(eta=0.1)
        
        print("Adam training:")
        print(f"{'Step':>5} {'Loss':>15}")
        print("-" * 25)
        
        for step in range(200):
            g = gradient(w)
            w = optimizer.step(w, g)
            
            if step % 20 == 0:
                print(f"{step:>5} {loss(w):>15.6f}")
        
        print(f"\nFinal loss: {loss(w):.6f}")
    
    def exercise_6_adamw_comparison(self):
        """
        Exercise 6: Adam vs AdamW
        
        Compare weight decay implementations.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Adam vs AdamW")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 50, 20
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        def gradient(w):
            return X.T @ (X @ w - y) / n
        
        eta = 0.1
        beta1, beta2 = 0.9, 0.999
        lamb = 0.1
        eps = 1e-8
        
        # Adam with L2 regularization (wrong way)
        w_adam = np.zeros(d)
        m_adam = np.zeros(d)
        v_adam = np.zeros(d)
        
        # AdamW (correct way)
        w_adamw = np.zeros(d)
        m_adamw = np.zeros(d)
        v_adamw = np.zeros(d)
        
        adam_norms = []
        adamw_norms = []
        
        for t in range(1, 201):
            # Adam: L2 in gradient
            g_adam = gradient(w_adam) + lamb * w_adam
            m_adam = beta1 * m_adam + (1 - beta1) * g_adam
            v_adam = beta2 * v_adam + (1 - beta2) * g_adam**2
            m_hat = m_adam / (1 - beta1**t)
            v_hat = v_adam / (1 - beta2**t)
            w_adam = w_adam - eta * m_hat / (np.sqrt(v_hat) + eps)
            adam_norms.append(np.linalg.norm(w_adam))
            
            # AdamW: decoupled weight decay
            g_adamw = gradient(w_adamw)
            m_adamw = beta1 * m_adamw + (1 - beta1) * g_adamw
            v_adamw = beta2 * v_adamw + (1 - beta2) * g_adamw**2
            m_hat = m_adamw / (1 - beta1**t)
            v_hat = v_adamw / (1 - beta2**t)
            w_adamw = w_adamw - eta * m_hat / (np.sqrt(v_hat) + eps) - eta * lamb * w_adamw
            adamw_norms.append(np.linalg.norm(w_adamw))
        
        print("Adam+L2: includes λw in gradient (affected by adaptive LR)")
        print("AdamW: decoupled weight decay after update")
        
        print(f"\n{'Step':>5} {'||w_adam||':>12} {'||w_adamw||':>12}")
        print("-" * 35)
        
        for t in [19, 49, 99, 149, 199]:
            print(f"{t+1:>5} {adam_norms[t]:>12.4f} {adamw_norms[t]:>12.4f}")
        
        print(f"\nFinal ||w||:")
        print(f"  Adam+L2: {adam_norms[-1]:.4f}")
        print(f"  AdamW:   {adamw_norms[-1]:.4f}")
    
    def exercise_7_lr_finder(self):
        """
        Exercise 7: Learning Rate Finder
        
        Implement LR range test.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Learning Rate Finder")
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
        
        # LR range test: exponentially increase LR
        lr_min, lr_max = 1e-5, 1.0
        n_steps = 100
        
        w = np.zeros(d)
        lrs = np.geomspace(lr_min, lr_max, n_steps)
        losses = []
        
        for lr in lrs:
            g = gradient(w)
            w = w - lr * g
            losses.append(loss(w))
        
        # Find best LR (before loss starts increasing rapidly)
        best_idx = 0
        for i in range(1, len(losses) - 1):
            if losses[i+1] > losses[i] * 1.5:  # Loss increasing
                break
            if losses[i] < losses[best_idx]:
                best_idx = i
        
        print("LR Range Test: exponentially increase LR, track loss")
        print(f"\n{'LR':>12} {'Loss':>12}")
        print("-" * 30)
        
        for i in [0, 25, 50, 75, 99]:
            print(f"{lrs[i]:>12.6f} {losses[i]:>12.4f}")
        
        print(f"\nSuggested LR: {lrs[best_idx]:.4f}")
        print("Use LR where loss is still decreasing but before explosion")
    
    def exercise_8_warmup(self):
        """
        Exercise 8: Learning Rate Warmup
        
        Implement and test warmup strategies.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Learning Rate Warmup")
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
        
        T = 100
        warmup = 20
        max_lr = 0.5
        
        def no_warmup_lr(t):
            return max_lr
        
        def linear_warmup_lr(t):
            if t < warmup:
                return max_lr * (t + 1) / warmup
            return max_lr
        
        def cosine_warmup_decay(t):
            if t < warmup:
                return max_lr * (t + 1) / warmup
            return max_lr * 0.5 * (1 + np.cos(np.pi * (t - warmup) / (T - warmup)))
        
        schedules = {
            'No warmup': no_warmup_lr,
            'Linear warmup': linear_warmup_lr,
            'Warmup + cosine': cosine_warmup_decay
        }
        
        results = {}
        
        for name, schedule in schedules.items():
            w = np.zeros(d)
            losses = [loss(w)]
            
            for t in range(T):
                lr = schedule(t)
                w = w - lr * gradient(w)
                losses.append(loss(w))
            
            results[name] = losses
        
        print("Comparing warmup strategies:")
        print(f"\n{'Step':>5}", end='')
        for name in results:
            print(f"{name:>18}", end='')
        print()
        print("-" * 65)
        
        for t in [0, 10, 25, 50, 100]:
            print(f"{t:>5}", end='')
            for name, losses in results.items():
                print(f"{losses[t]:>18.6f}", end='')
            print()
        
        print("\nWarmup helps when using large learning rates!")
    
    def exercise_9_compare_optimizers(self):
        """
        Exercise 9: Optimizer Comparison
        
        Compare optimizers on different problem types.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Optimizer Comparison")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Problem 1: Well-conditioned
        A_good = np.eye(10)
        
        # Problem 2: Ill-conditioned
        A_bad = np.diag(np.geomspace(0.1, 100, 10))
        
        b = np.ones(10)
        
        def run_experiment(A, name):
            print(f"\n{name} (κ = {np.linalg.cond(A):.1f}):")
            
            def gradient(w):
                return A @ w - b
            
            def loss(w):
                return 0.5 * w @ A @ w - b @ w
            
            results = {}
            
            # SGD
            w = np.zeros(10)
            for _ in range(200):
                w = w - 0.01 * gradient(w)
            results['SGD'] = loss(w)
            
            # Momentum
            w = np.zeros(10)
            v = np.zeros(10)
            for _ in range(200):
                v = 0.9 * v + gradient(w)
                w = w - 0.01 * v
            results['Momentum'] = loss(w)
            
            # Adam
            w = np.zeros(10)
            m, v = np.zeros(10), np.zeros(10)
            for t in range(1, 201):
                g = gradient(w)
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g**2
                w = w - 0.1 * (m / (1-0.9**t)) / (np.sqrt(v / (1-0.999**t)) + 1e-8)
            results['Adam'] = loss(w)
            
            optimal_loss = loss(np.linalg.solve(A, b))
            
            print(f"  Optimal loss: {optimal_loss:.6f}")
            for name, final_loss in results.items():
                gap = final_loss - optimal_loss
                print(f"  {name}: {final_loss:.6f} (gap: {gap:.6e})")
        
        run_experiment(A_good, "Well-conditioned")
        run_experiment(A_bad, "Ill-conditioned")
    
    def exercise_10_gradient_accumulation(self):
        """
        Exercise 10: Gradient Accumulation
        
        Simulate large batch with small memory.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Gradient Accumulation")
        print("=" * 60)
        
        np.random.seed(42)
        
        n, d = 100, 10
        X = np.random.randn(n, d)
        w_true = np.random.randn(d)
        y = X @ w_true + 0.1 * np.random.randn(n)
        
        def gradient(w, idx):
            X_b = X[idx]
            y_b = y[idx]
            return X_b.T @ (X_b @ w - y_b) / len(idx)
        
        def loss(w):
            return 0.5 * np.mean((X @ w - y)**2)
        
        # Target effective batch size
        effective_batch = 32
        # Actual batch size (memory constrained)
        micro_batch = 8
        # Accumulation steps
        accum_steps = effective_batch // micro_batch
        
        eta = 0.1
        
        print(f"Effective batch: {effective_batch}")
        print(f"Micro batch: {micro_batch}")
        print(f"Accumulation steps: {accum_steps}")
        
        # With accumulation
        w_accum = np.zeros(d)
        accum_losses = [loss(w_accum)]
        
        # Without accumulation (small batch)
        w_small = np.zeros(d)
        small_losses = [loss(w_small)]
        
        for step in range(50):
            # Gradient accumulation
            g_accum = np.zeros(d)
            for _ in range(accum_steps):
                idx = np.random.choice(n, micro_batch, replace=False)
                g_accum += gradient(w_accum, idx)
            g_accum /= accum_steps
            w_accum = w_accum - eta * g_accum
            accum_losses.append(loss(w_accum))
            
            # Small batch only
            idx = np.random.choice(n, micro_batch, replace=False)
            w_small = w_small - eta * gradient(w_small, idx)
            small_losses.append(loss(w_small))
        
        print(f"\n{'Step':>5} {'Accumulated':>15} {'Small batch':>15}")
        print("-" * 40)
        
        for t in [0, 10, 25, 50]:
            print(f"{t:>5} {accum_losses[t]:>15.6f} {small_losses[t]:>15.6f}")
        
        print("\nGradient accumulation simulates larger batch size!")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = AdaptiveLearningRateExercises()
    
    print("ADAPTIVE LEARNING RATE EXERCISES")
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
