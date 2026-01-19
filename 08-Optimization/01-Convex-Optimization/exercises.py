"""
Convex Optimization - Exercises
===============================
Practice problems for convexity in ML.
"""

import numpy as np
from scipy import optimize
from scipy.optimize import minimize


class ConvexOptimizationExercises:
    """Exercises for convex optimization."""
    
    def exercise_1_verify_convexity(self):
        """
        Exercise 1: Verify Convexity
        
        Determine if the function is convex.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Verify Convexity")
        print("=" * 60)
        
        print("\nDetermine if each function is convex:\n")
        
        # 1. f(x) = 3x² + 2x + 1
        print("1. f(x) = 3x² + 2x + 1")
        print("   f''(x) = 6 > 0 → Convex ✓")
        
        # 2. f(x) = log(1 + e^x)
        print("\n2. f(x) = log(1 + e^x)  (softplus)")
        print("   f'(x) = e^x / (1 + e^x) = σ(x)")
        print("   f''(x) = σ(x)(1 - σ(x)) > 0 → Convex ✓")
        
        # 3. f(x) = x log(x)
        print("\n3. f(x) = x log(x)  for x > 0")
        print("   f'(x) = log(x) + 1")
        print("   f''(x) = 1/x > 0 for x > 0 → Convex ✓")
        
        # 4. f(x,y) = x² - y²
        print("\n4. f(x,y) = x² - y²")
        print("   H = [[2, 0], [0, -2]]")
        print("   Eigenvalues: 2, -2")
        print("   Not all >= 0 → NOT Convex ✗")
        
        # 5. f(x,y) = x² + xy + y² + x + y
        print("\n5. f(x,y) = x² + xy + y² + x + y")
        H = np.array([[2, 1], [1, 2]])
        eigs = np.linalg.eigvalsh(H)
        print(f"   H = [[2, 1], [1, 2]]")
        print(f"   Eigenvalues: {eigs}")
        print(f"   All >= 0 → Convex ✓")
    
    def exercise_2_convex_set(self):
        """
        Exercise 2: Convex Sets
        
        Determine if sets are convex.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Convex Sets")
        print("=" * 60)
        
        print("\nDetermine if each set is convex:\n")
        
        print("1. S = {x ∈ R² : x₁² + x₂² ≤ 1}  (unit disk)")
        print("   Ball is convex ✓")
        
        print("\n2. S = {x ∈ R² : x₁² + x₂² ≥ 1}  (outside unit disk)")
        print("   Take x = (1,0), y = (-1,0), midpoint = (0,0)")
        print("   ||midpoint|| = 0 < 1, not in set")
        print("   NOT Convex ✗")
        
        print("\n3. S = {x ∈ R² : x₁ + x₂ ≤ 1, x₁ ≥ 0, x₂ ≥ 0}")
        print("   Intersection of halfspaces → Convex ✓")
        print("   (This is a triangle/simplex)")
        
        print("\n4. S = {x ∈ R² : |x₁| + |x₂| ≤ 1}  (L1 ball)")
        print("   L1 ball is convex ✓")
        print("   (Diamond/rhombus shape)")
        
        print("\n5. S = {A ∈ R^(n×n) : A = A', eigenvalues ≥ 0}")
        print("   Positive semidefinite cone is convex ✓")
    
    def exercise_3_operations(self):
        """
        Exercise 3: Operations Preserving Convexity
        
        Use composition rules.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Operations Preserving Convexity")
        print("=" * 60)
        
        print("\nDetermine convexity using composition rules:\n")
        
        print("1. f(x) = e^(x²)")
        print("   e^t is convex and increasing")
        print("   x² is convex")
        print("   Convex ∘ increasing ∘ convex → Convex ✓")
        
        print("\n2. f(x) = log(e^x + e^(-x))")
        print("   log(t) is concave and increasing for t > 0")
        print("   e^x + e^(-x) = 2cosh(x) is convex")
        print("   Concave ∘ increasing ∘ convex → NOT necessarily convex")
        print("   Actually: f''(x) = sech²(x) > 0 → Convex ✓")
        print("   (Need to verify directly)")
        
        print("\n3. f(x) = max(x², (x-1)²)")
        print("   x² is convex")
        print("   (x-1)² is convex")
        print("   Max of convex functions → Convex ✓")
        
        print("\n4. f(x) = ||Ax - b||₂²")
        print("   ||·||₂² is convex")
        print("   Ax - b is affine")
        print("   Convex ∘ affine → Convex ✓")
        
        print("\n5. f(x,y) = x²/y for y > 0")
        print("   This is a perspective function")
        print("   Convex (can verify via Hessian)")
        
        # Verify numerically
        def f5(z):
            x, y = z
            if y <= 0:
                return np.inf
            return x**2 / y
        
        # Check at a few points
        x1 = np.array([1.0, 2.0])
        x2 = np.array([2.0, 1.0])
        theta = 0.5
        mid = theta * x1 + (1 - theta) * x2
        
        print(f"\n   Numerical check:")
        print(f"   f(midpoint) = {f5(mid):.4f}")
        print(f"   Convex combination of f values = {theta*f5(x1) + (1-theta)*f5(x2):.4f}")
    
    def exercise_4_kkt(self):
        """
        Exercise 4: KKT Conditions
        
        Apply KKT to constrained problem.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: KKT Conditions")
        print("=" * 60)
        
        print("""
Problem: min  x² + y²
         s.t. x + 2y = 1
    """)
        
        print("Lagrangian: L = x² + y² + ν(x + 2y - 1)")
        
        print("\nKKT Conditions (equality constraint only):")
        print("1. Stationarity:")
        print("   ∂L/∂x = 2x + ν = 0  →  x = -ν/2")
        print("   ∂L/∂y = 2y + 2ν = 0  →  y = -ν")
        
        print("\n2. Primal feasibility:")
        print("   x + 2y = 1")
        print("   -ν/2 + 2(-ν) = 1")
        print("   -5ν/2 = 1")
        print("   ν = -2/5")
        
        print("\nSolution:")
        nu = -2/5
        x_opt = -nu/2
        y_opt = -nu
        print(f"   x* = {x_opt:.4f}")
        print(f"   y* = {y_opt:.4f}")
        print(f"   ν* = {nu:.4f}")
        print(f"   f(x*,y*) = {x_opt**2 + y_opt**2:.4f}")
        
        # Verify
        result = minimize(
            lambda z: z[0]**2 + z[1]**2,
            [0, 0],
            constraints={'type': 'eq', 'fun': lambda z: z[0] + 2*z[1] - 1}
        )
        print(f"\nNumerical verification: x*={result.x[0]:.4f}, y*={result.x[1]:.4f}")
    
    def exercise_5_duality(self):
        """
        Exercise 5: Lagrangian Duality
        
        Derive and solve dual problem.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Lagrangian Duality")
        print("=" * 60)
        
        print("""
Primal: min  x₁ + x₂
        s.t. x₁² + x₂² ≤ 1
    """)
        
        print("Lagrangian: L = x₁ + x₂ + λ(x₁² + x₂² - 1)")
        print("where λ ≥ 0")
        
        print("\nDual function d(λ):")
        print("  d(λ) = min_x [x₁ + x₂ + λ(x₁² + x₂² - 1)]")
        print("  ∂L/∂x₁ = 1 + 2λx₁ = 0  →  x₁ = -1/(2λ)")
        print("  ∂L/∂x₂ = 1 + 2λx₂ = 0  →  x₂ = -1/(2λ)")
        
        print("\n  Substituting:")
        print("  d(λ) = -1/(2λ) - 1/(2λ) + λ(1/(4λ²) + 1/(4λ²) - 1)")
        print("       = -1/λ + λ(1/(2λ²) - 1)")
        print("       = -1/λ + 1/(2λ) - λ")
        print("       = -1/(2λ) - λ")
        
        print("\nMaximizing dual:")
        print("  d'(λ) = 1/(2λ²) - 1 = 0")
        print("  λ² = 1/2  →  λ* = 1/√2 ≈ 0.707")
        
        lambda_opt = 1/np.sqrt(2)
        x_opt = -1/(2*lambda_opt)
        d_opt = -1/(2*lambda_opt) - lambda_opt
        
        print(f"\n  λ* = {lambda_opt:.4f}")
        print(f"  x₁* = x₂* = {x_opt:.4f}")
        print(f"  d(λ*) = {d_opt:.4f}")
        
        print(f"\nPrimal optimal: p* = x₁* + x₂* = {2*x_opt:.4f}")
        print(f"Strong duality: p* = d* = {d_opt:.4f} ✓")
    
    def exercise_6_ridge_kkt(self):
        """
        Exercise 6: Ridge Regression via KKT
        
        Derive ridge solution using KKT.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Ridge Regression via KKT")
        print("=" * 60)
        
        print("""
Problem: min  ||y - Xw||² + λ||w||²
    """)
        
        print("This is unconstrained, so optimality condition is just ∇f = 0")
        
        print("\n∇f = -2X'(y - Xw) + 2λw = 0")
        print("X'Xw - X'y + λw = 0")
        print("(X'X + λI)w = X'y")
        print("w* = (X'X + λI)⁻¹X'y")
        
        # Numerical example
        np.random.seed(42)
        n, p = 20, 3
        X = np.random.randn(n, p)
        true_w = np.array([1, 2, 3])
        y = X @ true_w + np.random.randn(n) * 0.5
        
        lambda_val = 1.0
        
        # Analytical solution
        w_ridge = np.linalg.solve(X.T @ X + lambda_val * np.eye(p), X.T @ y)
        
        print(f"\nNumerical example:")
        print(f"  True w = {true_w}")
        print(f"  λ = {lambda_val}")
        print(f"  w* = {np.round(w_ridge, 4)}")
        
        # Verify by numerical optimization
        result = minimize(
            lambda w: np.sum((y - X @ w)**2) + lambda_val * np.sum(w**2),
            np.zeros(p)
        )
        print(f"  Numerical opt: {np.round(result.x, 4)}")
    
    def exercise_7_lasso(self):
        """
        Exercise 7: LASSO Convexity
        
        Analyze LASSO optimization.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: LASSO Convexity")
        print("=" * 60)
        
        print("""
LASSO: min  ||y - Xw||² + λ||w||₁
    """)
        
        print("1. Is this problem convex?")
        print("   ||y - Xw||² is convex (composition of convex with affine)")
        print("   ||w||₁ = Σ|wᵢ| is convex (sum of convex functions)")
        print("   Sum of convex is convex → YES, convex ✓")
        
        print("\n2. Is the objective differentiable?")
        print("   ||w||₁ is not differentiable at wᵢ = 0")
        print("   Need subgradient methods")
        
        print("\n3. Subdifferential of |w|:")
        print("   ∂|w| = {-1}  if w < 0")
        print("        = [-1,1] if w = 0")
        print("        = {1}   if w > 0")
        
        print("\n4. Optimality condition (componentwise):")
        print("   0 ∈ 2X'(Xw - y) + λ∂||w||₁")
        print("   For each j:")
        print("   If wⱼ ≠ 0: rⱼ = λ sign(wⱼ)")
        print("   If wⱼ = 0: |rⱼ| ≤ λ")
        print("   where rⱼ = 2(X'(Xw - y))ⱼ")
        
        print("\n5. Soft thresholding interpretation:")
        print("   wⱼ = soft_threshold(zⱼ, λ)")
        print("   where soft_threshold(z, λ) = sign(z)max(|z| - λ, 0)")
    
    def exercise_8_logistic_convexity(self):
        """
        Exercise 8: Logistic Regression Convexity
        
        Prove logistic loss is convex.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Logistic Regression Convexity")
        print("=" * 60)
        
        print("""
Logistic loss: L(w) = Σᵢ log(1 + exp(-yᵢw'xᵢ))
    """)
        
        print("Let zᵢ = yᵢw'xᵢ")
        print("For one sample: ℓ(z) = log(1 + e^(-z))")
        
        print("\nFirst derivative:")
        print("  ℓ'(z) = -e^(-z)/(1 + e^(-z)) = -1/(1 + e^z) = σ(z) - 1")
        print("  where σ(z) = 1/(1 + e^(-z))")
        
        print("\nSecond derivative:")
        print("  ℓ''(z) = e^z/(1 + e^z)² = σ(z)(1 - σ(z))")
        print("  Since 0 < σ(z) < 1, we have ℓ''(z) > 0")
        print("  → ℓ(z) is convex ✓")
        
        print("\nHessian of L(w):")
        print("  H = Σᵢ σ(zᵢ)(1-σ(zᵢ)) xᵢxᵢ'")
        print("  = X' diag(σ(1-σ)) X")
        print("  Since σ(1-σ) > 0, H is positive semidefinite")
        print("  → L(w) is convex ✓")
        
        # Numerical verification
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        y = np.sign(np.random.randn(n))
        
        def logistic_loss(w):
            z = y * (X @ w)
            return np.sum(np.log(1 + np.exp(-z)))
        
        # Check convexity at two points
        w1 = np.random.randn(p)
        w2 = np.random.randn(p)
        theta = 0.3
        w_mid = theta * w1 + (1 - theta) * w2
        
        left = logistic_loss(w_mid)
        right = theta * logistic_loss(w1) + (1 - theta) * logistic_loss(w2)
        
        print(f"\nNumerical verification:")
        print(f"  L(θw₁ + (1-θ)w₂) = {left:.4f}")
        print(f"  θL(w₁) + (1-θ)L(w₂) = {right:.4f}")
        print(f"  Convex? {left <= right + 1e-10} ✓")
    
    def exercise_9_constrained_ml(self):
        """
        Exercise 9: Constrained ML Problem
        
        Solve constrained optimization.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Constrained ML Problem")
        print("=" * 60)
        
        print("""
Problem: Find weights that minimize MSE 
         subject to weights summing to 1 and being non-negative
         (portfolio optimization / mixture weights)

         min  ||y - Xw||²
         s.t. Σwⱼ = 1
              wⱼ ≥ 0
    """)
        
        np.random.seed(42)
        n, p = 50, 4
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        # Solve using scipy
        def objective(w):
            return np.sum((y - X @ w)**2)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        bounds = [(0, None) for _ in range(p)]
        
        result = minimize(objective, np.ones(p)/p, 
                         constraints=constraints, bounds=bounds)
        
        print(f"Optimal weights: {np.round(result.x, 4)}")
        print(f"Sum of weights: {np.sum(result.x):.4f}")
        print(f"All non-negative: {all(result.x >= -1e-10)}")
        print(f"Optimal MSE: {result.fun:.4f}")
        
        # Compare with unconstrained
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"\nUnconstrained OLS: {np.round(w_ols, 4)}")
        print(f"OLS MSE: {np.sum((y - X @ w_ols)**2):.4f}")
        print("\nConstraints increase MSE but may be required!")
    
    def exercise_10_projection(self):
        """
        Exercise 10: Projection onto Convex Set
        
        Project point onto convex set.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Projection onto Convex Set")
        print("=" * 60)
        
        print("""
Projection of point y onto convex set C:
    proj_C(y) = argmin_{x ∈ C} ||x - y||²
    """)
        
        print("\n1. Projection onto ball {x : ||x|| ≤ r}")
        y = np.array([3, 4])
        r = 2
        
        norm_y = np.linalg.norm(y)
        if norm_y <= r:
            proj = y
        else:
            proj = r * y / norm_y
        
        print(f"   y = {y}, ||y|| = {norm_y:.2f}")
        print(f"   r = {r}")
        print(f"   proj(y) = {np.round(proj, 4)}")
        print(f"   ||proj|| = {np.linalg.norm(proj):.4f}")
        
        print("\n2. Projection onto simplex {x : Σxᵢ = 1, xᵢ ≥ 0}")
        
        def project_simplex(y):
            """Project onto probability simplex."""
            n = len(y)
            u = np.sort(y)[::-1]
            cssv = np.cumsum(u)
            rho = np.where(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
            theta = (cssv[rho] - 1) / (rho + 1)
            return np.maximum(y - theta, 0)
        
        y2 = np.array([2, 0.5, -0.5, 0])
        proj2 = project_simplex(y2)
        
        print(f"   y = {y2}")
        print(f"   proj(y) = {np.round(proj2, 4)}")
        print(f"   Sum: {np.sum(proj2):.4f}, Min: {np.min(proj2):.4f}")
        
        print("\n3. Projection onto non-negative orthant {x : x ≥ 0}")
        y3 = np.array([2, -1, 3, -2])
        proj3 = np.maximum(y3, 0)
        
        print(f"   y = {y3}")
        print(f"   proj(y) = {proj3}")
        print("   (Just clip negative values to 0)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = ConvexOptimizationExercises()
    
    print("CONVEX OPTIMIZATION EXERCISES")
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
