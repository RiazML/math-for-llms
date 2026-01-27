"""
Gradient Descent Methods - Exercises
====================================
Practice problems for gradient descent optimization.
"""

import numpy as np


class GradientDescentExercises:
    """Exercises for gradient descent methods."""
    
    def exercise_1_implement_gd(self):
        """
        Exercise 1: Implement Basic Gradient Descent
        
        Minimize f(x) = x^4 - 3x^3 + 2
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Gradient Descent")
        print("=" * 60)
        
        def f(x):
            return x**4 - 3*x**3 + 2
        
        def grad_f(x):
            return 4*x**3 - 9*x**2
        
        print("f(x) = x⁴ - 3x³ + 2")
        print("f'(x) = 4x³ - 9x²")
        print("Setting f'(x) = 0: x²(4x - 9) = 0")
        print("Critical points: x = 0, x = 9/4 = 2.25")
        
        # Gradient descent
        x = 3.0  # Start
        eta = 0.01
        
        print(f"\nGradient descent from x₀ = {x}, η = {eta}")
        print(f"\n{'Step':>6} {'x':>12} {'f(x)':>12} {'|f\'(x)|':>12}")
        print("-" * 45)
        
        for i in range(50):
            if i % 10 == 0:
                print(f"{i:>6} {x:>12.6f} {f(x):>12.6f} {abs(grad_f(x)):>12.6f}")
            x = x - eta * grad_f(x)
        
        print(f"\nConverged to x = {x:.6f}")
        print(f"This is the local minimum at x = 2.25")
        print("Note: x = 0 is a saddle point (f''(0) = 0)")
    
    def exercise_2_convergence_rate(self):
        """
        Exercise 2: Analyze Convergence Rate
        
        Compare convergence on well-conditioned vs ill-conditioned.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Convergence Rate Analysis")
        print("=" * 60)
        
        # Well-conditioned: f(x) = x² + y²
        # Ill-conditioned: f(x) = 100x² + y²
        
        def gd_quadratic(A, x0, eta, n_steps):
            """GD on f(x) = 0.5 x'Ax."""
            x = x0.copy()
            distances = []
            for _ in range(n_steps):
                distances.append(np.linalg.norm(x))
                x = x - eta * A @ x
            return distances
        
        # Well-conditioned (κ = 1)
        A1 = np.eye(2)
        
        # Ill-conditioned (κ = 100)
        A2 = np.diag([100, 1])
        
        x0 = np.array([1.0, 1.0])
        
        print("f(x) = 0.5 x'Ax, optimal at x = 0")
        print(f"\nCondition numbers:")
        print(f"  A₁ = I: κ = {np.linalg.cond(A1):.1f}")
        print(f"  A₂ = diag(100,1): κ = {np.linalg.cond(A2):.1f}")
        
        # Optimal learning rate: η = 2/(λ_max + λ_min)
        eta1 = 2 / (1 + 1)
        eta2 = 2 / (100 + 1)
        
        dist1 = gd_quadratic(A1, x0, eta1, 50)
        dist2 = gd_quadratic(A2, x0, eta2, 50)
        
        print(f"\nOptimal learning rates: η₁ = {eta1:.4f}, η₂ = {eta2:.4f}")
        print(f"\n{'Step':>6} {'||x|| (κ=1)':>15} {'||x|| (κ=100)':>15}")
        print("-" * 40)
        
        for i in [0, 5, 10, 20, 30, 49]:
            print(f"{i:>6} {dist1[i]:>15.6f} {dist2[i]:>15.6f}")
        
        print("\nIll-conditioned problems converge much slower!")
        print("Convergence rate: (κ-1)/(κ+1) per step")
    
    def exercise_3_implement_momentum(self):
        """
        Exercise 3: Implement Momentum
        
        Add momentum to gradient descent.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Implement Momentum")
        print("=" * 60)
        
        # Narrow valley problem
        def f(w):
            return 0.5 * (w[0]**2 + 10*w[1]**2)
        
        def grad_f(w):
            return np.array([w[0], 10*w[1]])
        
        # GD without momentum
        w = np.array([10.0, 1.0])
        eta = 0.1
        
        trajectory_gd = [w.copy()]
        for _ in range(50):
            w = w - eta * grad_f(w)
            trajectory_gd.append(w.copy())
        
        # GD with momentum
        w = np.array([10.0, 1.0])
        v = np.zeros(2)
        gamma = 0.9
        
        trajectory_mom = [w.copy()]
        for _ in range(50):
            v = gamma * v + eta * grad_f(w)
            w = w - v
            trajectory_mom.append(w.copy())
        
        print("f(x,y) = 0.5(x² + 10y²)")
        print("Narrow valley: curvature ratio = 10")
        print(f"\nStarting at (10, 1), η = {eta}, γ = {gamma}")
        
        print(f"\n{'Step':>6} {'GD ||w||':>15} {'Momentum ||w||':>15}")
        print("-" * 40)
        
        for i in [0, 5, 10, 20, 50]:
            print(f"{i:>6} {np.linalg.norm(trajectory_gd[i]):>15.6f} "
                  f"{np.linalg.norm(trajectory_mom[i]):>15.6f}")
        
        print("\nMomentum converges faster in narrow valleys!")
    
    def exercise_4_implement_adam(self):
        """
        Exercise 4: Implement Adam
        
        Full Adam optimizer from scratch.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Implement Adam")
        print("=" * 60)
        
        class Adam:
            def __init__(self, eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
                self.eta = eta
                self.beta1 = beta1
                self.beta2 = beta2
                self.eps = eps
                self.m = None
                self.v = None
                self.t = 0
            
            def step(self, w, grad):
                if self.m is None:
                    self.m = np.zeros_like(w)
                    self.v = np.zeros_like(w)
                
                self.t += 1
                
                # Update moments
                self.m = self.beta1 * self.m + (1 - self.beta1) * grad
                self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
                
                # Bias correction
                m_hat = self.m / (1 - self.beta1**self.t)
                v_hat = self.v / (1 - self.beta2**self.t)
                
                # Update
                return w - self.eta * m_hat / (np.sqrt(v_hat) + self.eps)
        
        # Test on Rosenbrock
        def f(w):
            return (1 - w[0])**2 + 100*(w[1] - w[0]**2)**2
        
        def grad_f(w):
            dx = -2*(1 - w[0]) - 400*w[0]*(w[1] - w[0]**2)
            dy = 200*(w[1] - w[0]**2)
            return np.array([dx, dy])
        
        optimizer = Adam(eta=0.01)
        w = np.array([-1.0, 1.0])
        
        print("Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
        print("Optimal: (1, 1)")
        print(f"Initial: {w}")
        
        print(f"\n{'Step':>6} {'x':>10} {'y':>10} {'f(x,y)':>15}")
        print("-" * 45)
        
        for i in range(2000):
            if i % 400 == 0:
                print(f"{i:>6} {w[0]:>10.4f} {w[1]:>10.4f} {f(w):>15.6f}")
            grad = grad_f(w)
            w = optimizer.step(w, grad)
        
        print(f"\nFinal: ({w[0]:.4f}, {w[1]:.4f}), f = {f(w):.6f}")
    
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
        
        # Generate data
        n, p = 200, 5
        X = np.random.randn(n, p)
        true_w = np.random.randn(p)
        y = X @ true_w + np.random.randn(n) * 0.5
        
        def loss(w):
            return np.mean((X @ w - y)**2)
        
        def grad(w):
            return 2/n * X.T @ (X @ w - y)
        
        print("Learning Rate Range Test")
        print("Increase LR exponentially, track loss")
        print("\nLooking for:")
        print("  - Loss decreasing: LR too small")
        print("  - Loss stable/decreasing: Good LR range")
        print("  - Loss increasing: LR too large")
        
        # Range test
        lr_min, lr_max = 1e-5, 10
        n_steps = 100
        
        w = np.zeros(p)
        lr = lr_min
        lr_mult = (lr_max / lr_min) ** (1 / n_steps)
        
        lrs = []
        losses = []
        
        for _ in range(n_steps):
            lrs.append(lr)
            losses.append(loss(w))
            
            w = w - lr * grad(w)
            lr *= lr_mult
        
        print(f"\n{'LR':>12} {'Loss':>15} {'Status':>15}")
        print("-" * 45)
        
        prev_loss = float('inf')
        for i in range(0, n_steps, 10):
            status = ""
            if losses[i] < prev_loss * 0.99:
                status = "decreasing ✓"
            elif losses[i] > prev_loss * 1.1:
                status = "increasing ✗"
            else:
                status = "stable"
            
            print(f"{lrs[i]:>12.6f} {losses[i]:>15.6f} {status:>15}")
            prev_loss = losses[i]
        
        # Find best LR (where loss decreases fastest)
        loss_changes = np.diff(losses)
        best_idx = np.argmin(loss_changes)
        
        print(f"\nSuggested LR: {lrs[best_idx]:.6f}")
        print("(Where loss decreased fastest)")
    
    def exercise_6_sgd_noise(self):
        """
        Exercise 6: SGD Noise Analysis
        
        Analyze variance of SGD gradients.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: SGD Noise Analysis")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Data
        n = 1000
        X = np.random.randn(n, 3)
        true_w = np.array([1, 2, 3])
        y = X @ true_w + np.random.randn(n) * 0.5
        
        def full_grad(w):
            return 2/n * X.T @ (X @ w - y)
        
        def stochastic_grad(w, idx):
            Xi = X[idx:idx+1]
            yi = y[idx:idx+1]
            return 2 * Xi.T @ (Xi @ w - yi)
        
        # At a fixed point, compare full vs stochastic gradients
        w_test = np.array([0.5, 1.5, 2.5])
        
        true_grad = full_grad(w_test)
        
        # Sample many stochastic gradients
        n_samples = 500
        stoch_grads = []
        
        for _ in range(n_samples):
            idx = np.random.randint(n)
            stoch_grads.append(stochastic_grad(w_test, idx))
        
        stoch_grads = np.array(stoch_grads)
        
        print(f"Analyzing SGD gradient at w = {w_test}")
        print(f"\nTrue gradient: {np.round(true_grad, 4)}")
        print(f"Mean of SGD gradients: {np.round(stoch_grads.mean(axis=0), 4)}")
        print(f"Std of SGD gradients: {np.round(stoch_grads.std(axis=0), 4)}")
        
        print("\nSGD is unbiased (mean ≈ true gradient)")
        print("But has high variance!")
        
        # Variance reduction with mini-batches
        print("\nVariance vs batch size:")
        for batch_size in [1, 8, 32, 128, 512]:
            batch_grads = []
            for _ in range(100):
                idx = np.random.choice(n, batch_size, replace=False)
                Xi = X[idx]
                yi = y[idx]
                batch_grad = 2/batch_size * Xi.T @ (Xi @ w_test - yi)
                batch_grads.append(batch_grad)
            
            batch_std = np.mean(np.std(batch_grads, axis=0))
            print(f"  Batch size {batch_size:>3}: std = {batch_std:.4f}")
    
    def exercise_7_warmup(self):
        """
        Exercise 7: Learning Rate Warmup
        
        Implement warmup schedule.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Learning Rate Warmup")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Large batch training scenario
        n, p = 500, 10
        X = np.random.randn(n, p)
        true_w = np.random.randn(p)
        y = X @ true_w + np.random.randn(n) * 0.5
        
        def loss(w):
            return np.mean((X @ w - y)**2)
        
        def grad(w):
            return 2/n * X.T @ (X @ w - y)
        
        # Without warmup
        w_no_warmup = np.zeros(p)
        eta = 0.5  # Aggressive LR
        
        losses_no_warmup = []
        for _ in range(100):
            losses_no_warmup.append(loss(w_no_warmup))
            w_no_warmup = w_no_warmup - eta * grad(w_no_warmup)
        
        # With warmup
        w_warmup = np.zeros(p)
        warmup_steps = 10
        
        losses_warmup = []
        for t in range(100):
            losses_warmup.append(loss(w_warmup))
            
            if t < warmup_steps:
                current_eta = eta * (t + 1) / warmup_steps
            else:
                current_eta = eta
            
            w_warmup = w_warmup - current_eta * grad(w_warmup)
        
        print("Large learning rate (η=0.5) training")
        print(f"\n{'Step':>6} {'No Warmup':>15} {'With Warmup':>15}")
        print("-" * 40)
        
        for i in [0, 2, 5, 10, 20, 50, 99]:
            print(f"{i:>6} {losses_no_warmup[i]:>15.6f} {losses_warmup[i]:>15.6f}")
        
        print("\nWarmup stabilizes early training")
        print("Especially important for large batch / high LR")
    
    def exercise_8_convergence_proof(self):
        """
        Exercise 8: Convergence Analysis
        
        Verify theoretical convergence rate.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Convergence Analysis")
        print("=" * 60)
        
        # Strongly convex: f(x) = 0.5 * x'Ax
        # Convergence: ||x_t - x*||² <= (1 - μ/L)^t ||x_0 - x*||²
        
        L = 10  # Smoothness (max eigenvalue)
        mu = 1   # Strong convexity (min eigenvalue)
        
        A = np.diag([L, mu])
        x_star = np.zeros(2)
        
        # Optimal learning rate
        eta = 2 / (L + mu)
        
        # Theoretical rate
        rho = (L - mu) / (L + mu)  # Contraction factor
        
        print(f"f(x) = 0.5 x'Ax where A = diag({L}, {mu})")
        print(f"Condition number κ = L/μ = {L/mu}")
        print(f"Optimal η = 2/(L+μ) = {eta:.4f}")
        print(f"Contraction factor ρ = (L-μ)/(L+μ) = {rho:.4f}")
        
        x = np.array([10.0, 10.0])
        x0_dist = np.linalg.norm(x - x_star)
        
        print(f"\n{'Step':>6} {'Actual ||x-x*||':>18} {'Theory bound':>15} {'Ratio':>10}")
        print("-" * 55)
        
        for t in range(20):
            actual_dist = np.linalg.norm(x - x_star)
            theory_bound = x0_dist * (rho ** t)
            ratio = actual_dist / theory_bound
            
            if t % 4 == 0:
                print(f"{t:>6} {actual_dist:>18.6f} {theory_bound:>15.6f} {ratio:>10.4f}")
            
            x = x - eta * A @ x
        
        print("\nActual convergence follows theoretical bound")
    
    def exercise_9_gradient_clipping(self):
        """
        Exercise 9: Implement Gradient Clipping
        
        Both by value and by norm.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Gradient Clipping")
        print("=" * 60)
        
        def clip_by_value(grad, clip_value):
            """Clip each element to [-clip_value, clip_value]."""
            return np.clip(grad, -clip_value, clip_value)
        
        def clip_by_norm(grad, max_norm):
            """Scale gradient if norm exceeds max_norm."""
            norm = np.linalg.norm(grad)
            if norm > max_norm:
                return grad * max_norm / norm
            return grad
        
        # Test gradients
        gradients = [
            np.array([1.0, 2.0, 0.5]),
            np.array([10.0, 20.0, 5.0]),
            np.array([100.0, 0.0, -50.0]),
        ]
        
        max_norm = 5.0
        clip_value = 3.0
        
        print(f"Clipping parameters: max_norm={max_norm}, clip_value={clip_value}")
        
        for i, g in enumerate(gradients):
            print(f"\nGradient {i+1}: {g}")
            print(f"  Original norm: {np.linalg.norm(g):.4f}")
            
            g_value = clip_by_value(g, clip_value)
            print(f"  Clip by value: {g_value}, norm: {np.linalg.norm(g_value):.4f}")
            
            g_norm = clip_by_norm(g, max_norm)
            print(f"  Clip by norm:  {np.round(g_norm, 4)}, norm: {np.linalg.norm(g_norm):.4f}")
        
        print("\nValue clipping: changes direction")
        print("Norm clipping: preserves direction (usually preferred)")
    
    def exercise_10_optimizer_from_scratch(self):
        """
        Exercise 10: Complete Optimizer Class
        
        Build a configurable optimizer.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Complete Optimizer Class")
        print("=" * 60)
        
        class Optimizer:
            """Configurable optimizer supporting SGD, Momentum, Adam."""
            
            def __init__(self, params, lr=0.01, method='sgd', 
                        momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8,
                        weight_decay=0):
                self.params = params
                self.lr = lr
                self.method = method
                self.momentum = momentum
                self.beta1 = beta1
                self.beta2 = beta2
                self.eps = eps
                self.weight_decay = weight_decay
                self.t = 0
                
                # State
                self.v = np.zeros_like(params)  # Velocity/first moment
                self.s = np.zeros_like(params)  # Second moment
            
            def step(self, grad):
                self.t += 1
                
                # Weight decay
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * self.params
                
                if self.method == 'sgd':
                    update = self.lr * grad
                
                elif self.method == 'momentum':
                    self.v = self.momentum * self.v + grad
                    update = self.lr * self.v
                
                elif self.method == 'adam':
                    self.v = self.beta1 * self.v + (1 - self.beta1) * grad
                    self.s = self.beta2 * self.s + (1 - self.beta2) * grad**2
                    
                    v_hat = self.v / (1 - self.beta1**self.t)
                    s_hat = self.s / (1 - self.beta2**self.t)
                    
                    update = self.lr * v_hat / (np.sqrt(s_hat) + self.eps)
                
                self.params = self.params - update
                return self.params
        
        # Test on quadratic
        np.random.seed(42)
        
        A = np.array([[5, 1], [1, 3]])
        b = np.array([1, 2])
        optimal = np.linalg.solve(A, b)
        
        def loss(w):
            return 0.5 * w @ A @ w - b @ w
        
        def grad(w):
            return A @ w - b
        
        methods = ['sgd', 'momentum', 'adam']
        
        print(f"Minimizing f(x) = 0.5 x'Ax - b'x")
        print(f"Optimal: {np.round(optimal, 4)}")
        
        print(f"\n{'Method':>12} {'Final w':>20} {'||w-w*||':>12}")
        print("-" * 50)
        
        for method in methods:
            w = np.zeros(2)
            opt = Optimizer(w, lr=0.1, method=method)
            
            for _ in range(100):
                g = grad(opt.params)
                opt.step(g)
            
            dist = np.linalg.norm(opt.params - optimal)
            print(f"{method:>12} {str(np.round(opt.params, 4)):>20} {dist:>12.6f}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = GradientDescentExercises()
    
    print("GRADIENT DESCENT EXERCISES")
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
