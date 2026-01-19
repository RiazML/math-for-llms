"""
Optimization Theory - Exercises
===============================
Practice problems for optimization theory.
"""

import numpy as np


class OptimizationExercises:
    """Exercises for optimization theory."""
    
    def exercise_1_unconstrained_optimum(self):
        """
        Exercise 1: Find Unconstrained Optimum
        
        Find all critical points of f(x, y) = x³ - 3xy + y³
        and classify them.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Unconstrained Critical Points")
        
        print("\nf(x, y) = x³ - 3xy + y³")
        
        print("\nGradient:")
        print("  ∂f/∂x = 3x² - 3y = 0  →  y = x²")
        print("  ∂f/∂y = -3x + 3y² = 0  →  x = y²")
        
        print("\nSubstitute y = x² into x = y²:")
        print("  x = (x²)² = x⁴")
        print("  x⁴ - x = 0")
        print("  x(x³ - 1) = 0")
        print("  x = 0 or x = 1")
        
        print("\nCritical points:")
        print("  (0, 0) and (1, 1)")
        
        print("\nHessian:")
        print("  H = [6x   -3]")
        print("      [-3   6y]")
        
        print("\nAt (0, 0):")
        print("  H = [0  -3]")
        print("      [-3  0]")
        print("  det(H) = -9 < 0 → SADDLE POINT")
        
        print("\nAt (1, 1):")
        print("  H = [6  -3]")
        print("      [-3  6]")
        print("  det(H) = 36 - 9 = 27 > 0")
        print("  trace(H) = 12 > 0")
        print("  → POSITIVE DEFINITE → LOCAL MINIMUM")
        
        # Verify
        f = lambda x, y: x**3 - 3*x*y + y**3
        print(f"\nf(0, 0) = {f(0, 0)}")
        print(f"f(1, 1) = {f(1, 1)}")
    
    def exercise_2_lagrange_box(self):
        """
        Exercise 2: Lagrange Multipliers - Box Volume
        
        Find the maximum volume rectangular box inscribed in 
        the ellipsoid x²/a² + y²/b² + z²/c² = 1.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Maximum Volume Box in Ellipsoid")
        
        print("\nMaximize V = 8xyz (volume of box with corners at (±x, ±y, ±z))")
        print("Subject to: x²/a² + y²/b² + z²/c² = 1")
        
        print("\nLagrangian:")
        print("  L = 8xyz + λ(x²/a² + y²/b² + z²/c² - 1)")
        
        print("\nConditions:")
        print("  ∂L/∂x = 8yz + 2λx/a² = 0")
        print("  ∂L/∂y = 8xz + 2λy/b² = 0")
        print("  ∂L/∂z = 8xy + 2λz/c² = 0")
        
        print("\nFrom these equations:")
        print("  8yz = -2λx/a²  →  4a²yz = -λx")
        print("  8xz = -2λy/b²  →  4b²xz = -λy")
        print("  8xy = -2λz/c²  →  4c²xy = -λz")
        
        print("\nDividing:")
        print("  a²y/x = b²x/y  →  a²y² = b²x²  →  y/x = b/a")
        print("  b²z/y = c²y/z  →  b²z² = c²y²  →  z/y = c/b")
        
        print("\nSo: x : y : z = a : b : c")
        
        print("\nSubstitute into constraint:")
        print("  3x²/a² = 1  →  x = a/√3")
        
        print("\nSolution:")
        print("  x = a/√3, y = b/√3, z = c/√3")
        
        print("\nMaximum volume:")
        print("  V = 8 · (a/√3)(b/√3)(c/√3) = 8abc/(3√3)")
        
        # Numerical example
        a, b, c = 3, 2, 1
        V_max = 8 * a * b * c / (3 * np.sqrt(3))
        print(f"\nFor a={a}, b={b}, c={c}: V_max = {V_max:.4f}")
    
    def exercise_3_kkt_quadratic(self):
        """
        Exercise 3: KKT Conditions
        
        Solve: min x² + y²
               s.t. x + y ≥ 1
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: KKT for Quadratic Problem")
        
        print("\nmin x² + y²")
        print("s.t. -x - y + 1 ≤ 0  (rewritten h(x,y) ≤ 0)")
        
        print("\nLagrangian: L = x² + y² + μ(-x - y + 1)")
        
        print("\nKKT conditions:")
        print("  1. 2x - μ = 0  →  x = μ/2")
        print("  2. 2y - μ = 0  →  y = μ/2")
        print("  3. -x - y + 1 ≤ 0  (primal feasibility)")
        print("  4. μ ≥ 0  (dual feasibility)")
        print("  5. μ(-x - y + 1) = 0  (complementary slackness)")
        
        print("\nCase 1: μ = 0 (constraint inactive)")
        print("  x = y = 0")
        print("  Check: -0 - 0 + 1 = 1 > 0  VIOLATED!")
        
        print("\nCase 2: -x - y + 1 = 0 (constraint active)")
        print("  x + y = 1")
        print("  From x = y = μ/2: μ/2 + μ/2 = 1  →  μ = 1")
        print("  So x = y = 0.5")
        print("  Check μ ≥ 0: 1 ≥ 0 ✓")
        
        print("\nSolution: x* = y* = 0.5, μ* = 1")
        print(f"f* = 0.5² + 0.5² = {0.5**2 + 0.5**2}")
    
    def exercise_4_convexity(self):
        """
        Exercise 4: Prove Convexity
        
        Show that f(x) = ||Ax - b||² is convex for any matrix A.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Convexity of Least Squares")
        
        print("\nf(x) = ||Ax - b||² = (Ax - b)ᵀ(Ax - b)")
        print("     = xᵀAᵀAx - 2bᵀAx + bᵀb")
        
        print("\nGradient:")
        print("  ∇f = 2AᵀAx - 2Aᵀb = 2Aᵀ(Ax - b)")
        
        print("\nHessian:")
        print("  H = 2AᵀA")
        
        print("\nFor any vector v:")
        print("  vᵀHv = 2vᵀAᵀAv = 2||Av||² ≥ 0")
        
        print("\nTherefore H = 2AᵀA is positive semi-definite")
        print("→ f(x) is CONVEX")
        
        print("\nNote: If A has full column rank, AᵀA is positive DEFINITE")
        print("→ f(x) is STRONGLY CONVEX → unique minimum")
        
        # Verify
        np.random.seed(42)
        A = np.random.randn(5, 3)
        H = 2 * A.T @ A
        eigenvalues = np.linalg.eigvalsh(H)
        print(f"\nExample eigenvalues of H: {eigenvalues.round(4)}")
        print(f"All non-negative: {all(eigenvalues >= -1e-10)}")
    
    def exercise_5_dual_norm(self):
        """
        Exercise 5: Lagrangian Dual
        
        Find the dual of:
        min cᵀx
        s.t. ||x||₂ ≤ 1
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Dual of Norm-Constrained Problem")
        
        print("\nPrimal:")
        print("  min cᵀx")
        print("  s.t. ||x||₂ ≤ 1")
        
        print("\nRewrite constraint: ||x||₂² - 1 ≤ 0")
        
        print("\nLagrangian:")
        print("  L(x, μ) = cᵀx + μ(xᵀx - 1)")
        
        print("\nDual function:")
        print("  g(μ) = min_x [cᵀx + μxᵀx - μ]")
        
        print("\nStationarity: c + 2μx = 0  →  x = -c/(2μ)")
        
        print("\nSubstitute back:")
        print("  g(μ) = cᵀ(-c/(2μ)) + μ·||c||²/(4μ²) - μ")
        print("       = -||c||²/(2μ) + ||c||²/(4μ) - μ")
        print("       = -||c||²/(4μ) - μ")
        
        print("\nDual problem:")
        print("  max -||c||²/(4μ) - μ")
        print("  s.t. μ ≥ 0")
        
        print("\nOptimize over μ:")
        print("  d/dμ[-||c||²/(4μ) - μ] = ||c||²/(4μ²) - 1 = 0")
        print("  μ* = ||c||/2")
        
        print("\nDual optimal value:")
        print("  g(μ*) = -||c||²/(4·||c||/2) - ||c||/2 = -||c||")
        
        print("\nPrimal optimal (by inspection): x* = -c/||c||")
        print("  f* = cᵀ(-c/||c||) = -||c||")
        
        print("\nStrong duality holds! d* = p* = -||c||")
    
    def exercise_6_lasso(self):
        """
        Exercise 6: LASSO Problem
        
        Derive the KKT conditions for:
        min ½||Ax - b||² + λ||x||₁
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: LASSO KKT Conditions")
        
        print("\nmin f(x) = ½||Ax - b||² + λΣ|xᵢ|")
        
        print("\nSubdifferential of |xᵢ|:")
        print("  ∂|xᵢ| = { 1    if xᵢ > 0")
        print("         { [-1,1] if xᵢ = 0")
        print("         { -1   if xᵢ < 0")
        
        print("\nKKT condition (using subgradient):")
        print("  0 ∈ Aᵀ(Ax - b) + λ∂||x||₁")
        
        print("\nFor each coordinate i:")
        print("  [Aᵀ(Ax - b)]ᵢ + λsᵢ = 0, where sᵢ ∈ ∂|xᵢ|")
        
        print("\nCases:")
        print("  1. If xᵢ > 0: [Aᵀ(Ax-b)]ᵢ = -λ")
        print("  2. If xᵢ < 0: [Aᵀ(Ax-b)]ᵢ = λ")
        print("  3. If xᵢ = 0: |[Aᵀ(Ax-b)]ᵢ| ≤ λ")
        
        print("\nThis implies sparsity: xᵢ = 0 when |gradient| ≤ λ")
        
        # Soft thresholding for simple case (A = I)
        print("\n--- Soft Thresholding (A = I) ---")
        print("Solution: xᵢ = sign(bᵢ) · max(|bᵢ| - λ, 0)")
        
        b = np.array([2.0, 0.5, -1.0, 0.3])
        lam = 0.7
        
        def soft_threshold(b, lam):
            return np.sign(b) * np.maximum(np.abs(b) - lam, 0)
        
        x = soft_threshold(b, lam)
        print(f"\nb = {b}")
        print(f"λ = {lam}")
        print(f"x* = {x}")
    
    def exercise_7_svm_margin(self):
        """
        Exercise 7: SVM Margin Derivation
        
        Show that the margin in hard-margin SVM is 2/||w||.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: SVM Margin")
        
        print("\nDecision boundary: wᵀx + b = 0")
        print("Support vectors satisfy: wᵀx + b = ±1")
        
        print("\nDistance from point x₀ to hyperplane wᵀx + b = 0:")
        print("  d = |wᵀx₀ + b| / ||w||")
        
        print("\nFor positive support vector (wᵀx₊ + b = 1):")
        print("  d₊ = 1 / ||w||")
        
        print("\nFor negative support vector (wᵀx₋ + b = -1):")
        print("  d₋ = 1 / ||w||")
        
        print("\nTotal margin = d₊ + d₋ = 2 / ||w||")
        
        print("\nMaximizing margin = minimizing ||w||")
        print("Equivalently: minimize ½||w||² (convex!)")
        
        # Visual
        print("\n--- Visual ---")
        print("       wᵀx + b = 1    wᵀx + b = 0    wᵀx + b = -1")
        print("          │              │              │")
        print("    ●     │              │              │     ○")
        print("          │              │              │")
        print("    ●     │      +       │      -       │     ○")
        print("          │              │              │")
        print("          │←── 1/||w|| ──│── 1/||w|| ──→│")
        print("          │←────── margin = 2/||w|| ────→│")
    
    def exercise_8_newton_convergence(self):
        """
        Exercise 8: Newton's Method Convergence
        
        Implement Newton's method for f(x) = x⁴ - 2x² + 1
        and observe convergence rate.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Newton's Method Convergence")
        
        print("\nf(x) = x⁴ - 2x² + 1 = (x² - 1)²")
        print("f'(x) = 4x³ - 4x = 4x(x² - 1)")
        print("f''(x) = 12x² - 4")
        
        print("\nCritical points: x = 0, ±1")
        print("Global minima at x = ±1 (f = 0)")
        
        def f(x):
            return x**4 - 2*x**2 + 1
        
        def f_prime(x):
            return 4*x**3 - 4*x
        
        def f_double_prime(x):
            return 12*x**2 - 4
        
        x = 2.0  # Initial guess
        print(f"\nStarting from x₀ = {x}")
        
        errors = []
        x_star = 1.0
        
        for i in range(10):
            err = abs(x - x_star)
            errors.append(err)
            
            if abs(f_double_prime(x)) < 1e-10:
                print("  Warning: f''(x) ≈ 0")
                break
            
            x_new = x - f_prime(x) / f_double_prime(x)
            print(f"  Iter {i+1}: x = {x:.10f}, error = {err:.2e}")
            
            if err < 1e-12:
                break
            x = x_new
        
        print("\nConvergence rate analysis:")
        for i in range(1, len(errors)-1):
            if errors[i] > 1e-15 and errors[i-1] > 1e-15:
                rate = np.log(errors[i+1]) / np.log(errors[i])
                print(f"  err_{i+1}/err_{i}^r ≈ r = {rate:.2f}")
        
        print("\nNear a simple root, Newton has quadratic convergence (r ≈ 2)")
    
    def exercise_9_coordinate_descent(self):
        """
        Exercise 9: Coordinate Descent
        
        Implement coordinate descent for:
        min f(x, y) = x² + y² + xy
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Coordinate Descent")
        
        print("\nf(x, y) = x² + y² + xy")
        
        print("\nCoordinate descent:")
        print("  Fix y, minimize over x: ∂f/∂x = 2x + y = 0 → x = -y/2")
        print("  Fix x, minimize over y: ∂f/∂y = 2y + x = 0 → y = -x/2")
        
        def f(x, y):
            return x**2 + y**2 + x*y
        
        x, y = 3.0, 4.0
        print(f"\nInitial: ({x}, {y}), f = {f(x,y):.4f}")
        
        for i in range(10):
            # Minimize over x
            x = -y/2
            # Minimize over y
            y = -x/2
            print(f"Iter {i+1}: ({x:.6f}, {y:.6f}), f = {f(x,y):.6f}")
            
            if abs(x) < 1e-10 and abs(y) < 1e-10:
                break
        
        print("\nOptimum: (0, 0)")
    
    def exercise_10_barrier_method(self):
        """
        Exercise 10: Barrier Method
        
        Solve: min x² 
               s.t. x ≥ 1
        using log barrier.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Barrier Method")
        
        print("\nOriginal problem:")
        print("  min x²  s.t. x ≥ 1")
        
        print("\nBarrier problem:")
        print("  min x² - (1/t)log(x - 1)")
        
        print("\nAs t → ∞, solution approaches constrained optimum")
        
        def barrier_obj(x, t):
            if x <= 1:
                return float('inf')
            return x**2 - (1/t) * np.log(x - 1)
        
        def barrier_grad(x, t):
            return 2*x - 1/(t * (x - 1))
        
        # Solve for increasing t
        x = 3.0  # Start in interior
        
        for t in [1, 10, 100, 1000, 10000]:
            # Newton's method for each t
            for _ in range(20):
                g = barrier_grad(x, t)
                # Approximate Hessian
                h = 2 + 1/(t * (x-1)**2)
                x = x - g/h
                x = max(x, 1 + 1e-10)  # Stay feasible
            
            print(f"t = {t:5d}: x = {x:.6f}, f(x) = {x**2:.6f}")
        
        print("\nTrue optimum: x* = 1, f* = 1")
        print("Barrier method approaches this as t → ∞")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = OptimizationExercises()
    
    print("OPTIMIZATION THEORY EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    print("\n" + "=" * 70)
    
    exercises.solution_2()
    print("\n" + "=" * 70)
    
    exercises.solution_3()
    print("\n" + "=" * 70)
    
    exercises.solution_4()
    print("\n" + "=" * 70)
    
    exercises.solution_5()
    print("\n" + "=" * 70)
    
    exercises.solution_6()
    print("\n" + "=" * 70)
    
    exercises.solution_7()
    print("\n" + "=" * 70)
    
    exercises.solution_8()
    print("\n" + "=" * 70)
    
    exercises.solution_9()
    print("\n" + "=" * 70)
    
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
