"""
Constrained Optimization - Exercises
====================================
Practice problems for constrained optimization.
"""

import numpy as np
from scipy import optimize


class ConstrainedOptimizationExercises:
    """Exercises for constrained optimization."""
    
    def exercise_1_lagrange_multipliers(self):
        """
        Exercise 1: Lagrange Multipliers
        
        Solve using Lagrange multipliers.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Lagrange Multipliers")
        print("=" * 60)
        
        print("Maximize f(x,y) = xy")
        print("Subject to: x + 2y = 6")
        
        print("\nLagrangian: L = xy + λ(x + 2y - 6)")
        
        print("\n∂L/∂x = y + λ = 0  →  λ = -y")
        print("∂L/∂y = x + 2λ = 0  →  λ = -x/2")
        print("-y = -x/2  →  x = 2y")
        print("x + 2y = 6  →  2y + 2y = 6  →  y = 3/2")
        print("x = 2(3/2) = 3")
        
        x_opt, y_opt = 3, 1.5
        f_opt = x_opt * y_opt
        lambda_opt = -y_opt
        
        print(f"\nOptimal: x* = {x_opt}, y* = {y_opt}")
        print(f"f* = {f_opt}")
        print(f"λ* = {lambda_opt}")
        
        # Verify
        result = optimize.minimize(
            lambda x: -x[0]*x[1],  # Minimize negative for maximization
            [1, 1],
            constraints={'type': 'eq', 'fun': lambda x: x[0] + 2*x[1] - 6}
        )
        print(f"\nVerification: x = {result.x}")
    
    def exercise_2_kkt_analysis(self):
        """
        Exercise 2: KKT Conditions
        
        Solve using KKT conditions.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: KKT Conditions")
        print("=" * 60)
        
        print("Minimize f(x,y) = x² + y²")
        print("Subject to: x + y ≥ 2")
        print("(Standard form: g(x,y) = 2 - x - y ≤ 0)")
        
        print("\nLagrangian: L = x² + y² + μ(2 - x - y)")
        
        print("\nKKT Conditions:")
        print("1. ∂L/∂x = 2x - μ = 0  →  x = μ/2")
        print("2. ∂L/∂y = 2y - μ = 0  →  y = μ/2")
        print("3. g(x,y) ≤ 0: x + y ≥ 2")
        print("4. μ ≥ 0")
        print("5. μ(2-x-y) = 0")
        
        print("\nCase: μ = 0")
        print("  x = y = 0, but 0 + 0 < 2. Violates constraint. ✗")
        
        print("\nCase: μ > 0 (constraint active: x + y = 2)")
        print("  x = y = μ/2")
        print("  μ/2 + μ/2 = 2  →  μ = 2")
        print("  x = y = 1. ✓")
        
        x_opt, y_opt, mu_opt = 1, 1, 2
        f_opt = x_opt**2 + y_opt**2
        
        print(f"\nOptimal: x* = {x_opt}, y* = {y_opt}, μ* = {mu_opt}")
        print(f"f* = {f_opt}")
    
    def exercise_3_penalty_method(self):
        """
        Exercise 3: Implement Penalty Method
        
        Solve constrained problem using quadratic penalty.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Penalty Method")
        print("=" * 60)
        
        print("Minimize f(x,y) = x² + y²")
        print("Subject to: x + y = 2")
        
        def objective(x):
            return x[0]**2 + x[1]**2
        
        def constraint(x):
            return x[0] + x[1] - 2
        
        def penalized(x, rho):
            return objective(x) + rho * constraint(x)**2
        
        print(f"\n{'ρ':>10} {'x':>10} {'y':>10} {'f':>10} {'|h|':>12}")
        print("-" * 55)
        
        x = np.array([0.0, 0.0])
        
        for rho in [0.1, 1, 10, 100, 1000]:
            result = optimize.minimize(
                lambda z: penalized(z, rho),
                x
            )
            x = result.x
            
            print(f"{rho:>10} {x[0]:>10.6f} {x[1]:>10.6f} "
                  f"{objective(x):>10.6f} {abs(constraint(x)):>12.2e}")
        
        print(f"\nTrue optimum: x = y = 1, f* = 2")
    
    def exercise_4_projected_gd(self):
        """
        Exercise 4: Implement Projected Gradient Descent
        
        For constrained optimization.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Projected Gradient Descent")
        print("=" * 60)
        
        print("Minimize f(x,y) = (x-3)² + (y-3)²")
        print("Subject to: x² + y² ≤ 1 (unit ball)")
        
        def gradient(x):
            return np.array([2*(x[0]-3), 2*(x[1]-3)])
        
        def project_ball(x):
            """Project onto unit ball."""
            norm = np.linalg.norm(x)
            if norm <= 1:
                return x
            return x / norm
        
        x = np.array([0.0, 0.0])
        eta = 0.1
        
        print(f"\n{'Iter':>4} {'x':>10} {'y':>10} {'f':>12} {'||x||':>10}")
        print("-" * 50)
        
        for i in range(20):
            f_val = (x[0]-3)**2 + (x[1]-3)**2
            
            if i % 5 == 0:
                print(f"{i:>4} {x[0]:>10.6f} {x[1]:>10.6f} "
                      f"{f_val:>12.6f} {np.linalg.norm(x):>10.6f}")
            
            # Gradient step
            x_unconstrained = x - eta * gradient(x)
            # Project
            x = project_ball(x_unconstrained)
        
        print(f"\nFinal: x = {x[0]:.6f}, y = {x[1]:.6f}")
        print(f"||x|| = {np.linalg.norm(x):.6f} (on boundary)")
        print(f"\nOptimal: (1/√2, 1/√2) ≈ (0.707, 0.707)")
    
    def exercise_5_dual_problem(self):
        """
        Exercise 5: Derive and Solve Dual
        
        Find dual problem and solve.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Dual Problem")
        print("=" * 60)
        
        print("Primal: min x² + y²")
        print("        s.t. x + y ≥ 2")
        
        print("\nLagrangian: L = x² + y² + μ(2 - x - y)")
        
        print("\nDual function g(μ) = min_x,y L(x,y,μ)")
        print("∂L/∂x = 2x - μ = 0  →  x = μ/2")
        print("∂L/∂y = 2y - μ = 0  →  y = μ/2")
        
        print("\ng(μ) = (μ/2)² + (μ/2)² + μ(2 - μ/2 - μ/2)")
        print("     = μ²/4 + μ²/4 + 2μ - μ²")
        print("     = μ²/2 + 2μ - μ²")
        print("     = 2μ - μ²/2")
        
        print("\nDual: max μ(2 - μ/2)")
        print("      s.t. μ ≥ 0")
        
        print("\nd g/d μ = 2 - μ = 0  →  μ* = 2")
        print("g(2) = 2(2) - 4/2 = 2")
        
        print("\nStrong duality: primal optimal = dual optimal = 2")
    
    def exercise_6_soft_thresholding(self):
        """
        Exercise 6: Derive Soft Thresholding
        
        Prove proximal operator of L1.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Soft Thresholding")
        print("=" * 60)
        
        print("Proximal operator of ||x||₁:")
        print("prox_λ||·||₁(v) = argmin_x (||x||₁ + (1/2λ)||x-v||²)")
        
        print("\nFor scalar case:")
        print("prox(v) = argmin_x (|x| + (1/2λ)(x-v)²)")
        
        print("\nCase 1: x > 0")
        print("  d/dx (x + (1/2λ)(x-v)²) = 1 + (x-v)/λ = 0")
        print("  x = v - λ")
        print("  Valid when x > 0, i.e., v > λ")
        
        print("\nCase 2: x < 0")
        print("  d/dx (-x + (1/2λ)(x-v)²) = -1 + (x-v)/λ = 0")
        print("  x = v + λ")
        print("  Valid when x < 0, i.e., v < -λ")
        
        print("\nCase 3: |v| ≤ λ")
        print("  x = 0 minimizes (by checking)")
        
        print("\nSoft thresholding:")
        print("S_λ(v) = sign(v) × max(|v| - λ, 0)")
        
        # Demonstration
        def soft_threshold(v, lam):
            return np.sign(v) * np.maximum(np.abs(v) - lam, 0)
        
        lam = 0.5
        print(f"\nλ = {lam}")
        for v in [-1.5, -0.3, 0, 0.3, 1.5]:
            print(f"  S_{lam}({v:>5.1f}) = {soft_threshold(v, lam):>5.2f}")
    
    def exercise_7_barrier_analysis(self):
        """
        Exercise 7: Analyze Barrier Method
        
        Study central path.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Barrier Method Analysis")
        print("=" * 60)
        
        print("Minimize f(x) = x")
        print("Subject to: x ≥ 1 (i.e., 1 - x ≤ 0)")
        
        print("\nBarrier problem: min x - (1/t)log(x - 1)")
        print("Domain: x > 1")
        
        print("\nOptimality: 1 - (1/t)/(x-1) = 0")
        print("x - 1 = 1/t")
        print("x*(t) = 1 + 1/t")
        
        print(f"\n{'t':>8} {'x*(t)':>15} {'f(x*)':>15} {'Gap':>15}")
        print("-" * 60)
        
        for t in [1, 10, 100, 1000, 10000]:
            x_t = 1 + 1/t
            f_t = x_t  # Objective
            gap = x_t - 1  # Distance from true optimum
            
            print(f"{t:>8} {x_t:>15.10f} {f_t:>15.10f} {gap:>15.10f}")
        
        print("\nCentral path: x*(t) = 1 + 1/t")
        print("As t → ∞, x*(t) → 1 (true optimum)")
    
    def exercise_8_admm(self):
        """
        Exercise 8: ADMM Updates
        
        Derive ADMM updates for specific problem.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: ADMM")
        print("=" * 60)
        
        print("Solve: min (1/2)||Ax - b||² + λ||z||₁")
        print("       s.t. x = z")
        
        print("\nAugmented Lagrangian:")
        print("L_ρ = (1/2)||Ax-b||² + λ||z||₁ + y'(x-z) + (ρ/2)||x-z||²")
        
        print("\nADMM updates:")
        print("1. x ← (A'A + ρI)⁻¹(A'b + ρz - y)")
        print("2. z ← S_{λ/ρ}(x + y/ρ)  [soft thresholding]")
        print("3. y ← y + ρ(x - z)")
        
        # Implementation
        np.random.seed(42)
        n, d = 30, 10
        A = np.random.randn(n, d)
        x_true = np.array([2, -1, 0, 0, 0, 0, 0, 0, 1, 0])
        b = A @ x_true + 0.1 * np.random.randn(n)
        
        lam = 0.5
        rho = 1.0
        
        # Precompute
        ATA = A.T @ A
        ATb = A.T @ b
        factor = np.linalg.inv(ATA + rho * np.eye(d))
        
        x = np.zeros(d)
        z = np.zeros(d)
        y = np.zeros(d)
        
        def soft_threshold(v, t):
            return np.sign(v) * np.maximum(np.abs(v) - t, 0)
        
        print(f"\n{'Iter':>4} {'||x-z||':>12} {'nnz(z)':>8}")
        print("-" * 30)
        
        for i in range(100):
            # x-update
            x = factor @ (ATb + rho * z - y)
            
            # z-update
            z = soft_threshold(x + y/rho, lam/rho)
            
            # y-update
            y = y + rho * (x - z)
            
            if i % 20 == 0:
                print(f"{i:>4} {np.linalg.norm(x-z):>12.6f} "
                      f"{np.sum(np.abs(z) > 0.01):>8}")
        
        print(f"\nTrue nonzeros: {np.sum(np.abs(x_true) > 0)}")
        print(f"Recovered z: {np.round(z, 3)}")
    
    def exercise_9_svm_from_scratch(self):
        """
        Exercise 9: SVM Optimization
        
        Solve SVM dual problem.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: SVM from Dual")
        print("=" * 60)
        
        # Simple dataset
        np.random.seed(42)
        X_pos = np.array([[1, 2], [2, 1], [1.5, 1.5]]) + 1
        X_neg = np.array([[1, 2], [2, 1], [1.5, 1.5]]) - 1
        X = np.vstack([X_pos, X_neg])
        y = np.array([1, 1, 1, -1, -1, -1])
        
        print("Dual: max Σαᵢ - (1/2)ΣΣ αᵢαⱼyᵢyⱼxᵢ'xⱼ")
        print("s.t. αᵢ ≥ 0, Σαᵢyᵢ = 0")
        
        n = len(y)
        K = X @ X.T
        Q = np.outer(y, y) * K
        
        # Solve with quadratic programming
        def dual_obj(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)
        
        constraints = [
            {'type': 'eq', 'fun': lambda a: a @ y}
        ]
        bounds = [(0, 100) for _ in range(n)]  # C = 100 (soft margin)
        
        result = optimize.minimize(
            dual_obj, np.ones(n),
            constraints=constraints,
            bounds=bounds,
            method='SLSQP'
        )
        
        alpha = result.x
        
        # Recover w
        w = np.sum((alpha * y)[:, None] * X, axis=0)
        
        # Find support vectors and compute b
        sv = alpha > 1e-4
        b = np.mean(y[sv] - X[sv] @ w)
        
        print(f"\nDual variables α: {np.round(alpha, 4)}")
        print(f"Support vector indices: {np.where(sv)[0]}")
        print(f"\nw = {np.round(w, 4)}")
        print(f"b = {b:.4f}")
        
        # Test
        predictions = np.sign(X @ w + b)
        accuracy = np.mean(predictions == y)
        print(f"\nTraining accuracy: {accuracy:.0%}")
    
    def exercise_10_portfolio(self):
        """
        Exercise 10: Portfolio Optimization
        
        Mean-variance with constraints.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Portfolio Optimization")
        print("=" * 60)
        
        # Asset data
        returns = np.array([0.10, 0.15, 0.08, 0.12])  # Expected returns
        cov = np.array([
            [0.04, 0.01, 0.00, 0.02],
            [0.01, 0.09, 0.01, 0.03],
            [0.00, 0.01, 0.02, 0.00],
            [0.02, 0.03, 0.00, 0.05]
        ])  # Covariance matrix
        
        print("Min variance portfolio:")
        print("  min w'Σw")
        print("  s.t. w'μ ≥ r_target")
        print("       Σwᵢ = 1")
        print("       wᵢ ≥ 0")
        
        def variance(w):
            return w @ cov @ w
        
        print(f"\n{'r_target':>10} {'Variance':>12} {'Std':>10} {'Weights':>30}")
        print("-" * 70)
        
        for r_target in [0.08, 0.10, 0.12, 0.14]:
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w, r=r_target: returns @ w - r}
            ]
            bounds = [(0, 1) for _ in range(4)]
            
            result = optimize.minimize(
                variance,
                np.ones(4) / 4,
                constraints=constraints,
                bounds=bounds,
                method='SLSQP'
            )
            
            w = result.x
            var = variance(w)
            std = np.sqrt(var)
            
            print(f"{r_target:>10.0%} {var:>12.4f} {std:>10.2%} "
                  f"{str(np.round(w, 3)):>30}")
        
        print("\nEfficient frontier: higher return → higher variance")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = ConstrainedOptimizationExercises()
    
    print("CONSTRAINED OPTIMIZATION EXERCISES")
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
