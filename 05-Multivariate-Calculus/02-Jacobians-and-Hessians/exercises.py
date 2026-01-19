"""
Jacobians and Hessians - Exercises
==================================
Practice problems for Jacobian and Hessian matrices.
"""

import numpy as np


class JacobianHessianExercises:
    """Exercises for Jacobians and Hessians."""
    
    def exercise_1_jacobian_basic(self):
        """
        Exercise 1: Basic Jacobian
        
        Compute the Jacobian of:
        f(x, y) = [xy + y², x² - y, e^x]ᵀ
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Jacobian")
        
        print("\nf(x, y) = [xy + y², x² - y, eˣ]ᵀ")
        print("\nPartial derivatives:")
        print("∂f₁/∂x = y,     ∂f₁/∂y = x + 2y")
        print("∂f₂/∂x = 2x,    ∂f₂/∂y = -1")
        print("∂f₃/∂x = eˣ,    ∂f₃/∂y = 0")
        
        print("\nJacobian J (3×2):")
        print("J = [  y     x+2y  ]")
        print("    [ 2x      -1   ]")
        print("    [ eˣ       0   ]")
        
        # Verify at specific point
        x, y = 1.0, 2.0
        J = np.array([
            [y, x + 2*y],
            [2*x, -1],
            [np.exp(x), 0]
        ])
        
        print(f"\nAt (x, y) = ({x}, {y}):")
        print(f"J = \n{J.round(4)}")
    
    def exercise_2_jacobian_chain(self):
        """
        Exercise 2: Jacobian Chain Rule
        
        Given g(t) = [t², t³] and f(u, v) = [u + v, uv]
        
        Find the Jacobian of f ∘ g
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Jacobian Chain Rule")
        
        print("\ng(t) = [t², t³]  (ℝ → ℝ²)")
        print("f(u, v) = [u + v, uv]  (ℝ² → ℝ²)")
        
        print("\nJ_g = [2t]   (2×1)")
        print("      [3t²]")
        
        print("\nJ_f = [ 1   1 ]   (2×2)")
        print("      [ v   u ]")
        
        print("\nAt u = t², v = t³:")
        print("J_f = [ 1    1  ]")
        print("      [ t³   t² ]")
        
        print("\nJ_{f∘g} = J_f × J_g")
        print("       = [ 1    1  ] [ 2t  ]")
        print("         [ t³   t² ] [ 3t² ]")
        print("       = [ 2t + 3t²     ]")
        print("         [ 2t⁴ + 3t⁴   ]")
        print("       = [ 2t + 3t²    ]")
        print("         [ 5t⁴         ]")
        
        # Verify
        t = 2.0
        J_chain = np.array([2*t + 3*t**2, 5*t**4]).reshape(-1, 1)
        print(f"\nAt t = {t}: J = {J_chain.flatten()}")
        
        # Direct derivative
        # (f∘g)(t) = [t² + t³, t²·t³] = [t² + t³, t⁵]
        # d/dt = [2t + 3t², 5t⁴]
        print(f"Direct: d/dt[t²+t³, t⁵] = [{2*t + 3*t**2}, {5*t**4}] ✓")
    
    def exercise_3_hessian_computation(self):
        """
        Exercise 3: Hessian Computation
        
        Find the Hessian of f(x, y) = x³ - 3xy + y³
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Hessian Computation")
        
        print("\nf(x, y) = x³ - 3xy + y³")
        
        print("\nFirst derivatives:")
        print("∂f/∂x = 3x² - 3y")
        print("∂f/∂y = -3x + 3y²")
        
        print("\nSecond derivatives:")
        print("∂²f/∂x² = 6x")
        print("∂²f/∂y² = 6y")
        print("∂²f/∂x∂y = -3")
        
        print("\nH = [ 6x   -3 ]")
        print("    [ -3   6y ]")
        
        # At specific point
        x, y = 1.0, 2.0
        H = np.array([[6*x, -3], [-3, 6*y]])
        
        print(f"\nAt ({x}, {y}): H = \n{H}")
        print(f"Eigenvalues: {np.linalg.eigvals(H).round(4)}")
    
    def exercise_4_critical_points(self):
        """
        Exercise 4: Critical Point Classification
        
        For f(x, y) = x² + y² - 2x - 4y + 5:
        a) Find critical points
        b) Classify using Hessian
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Critical Point Classification")
        
        print("\nf(x, y) = x² + y² - 2x - 4y + 5")
        
        print("\na) Finding critical points:")
        print("∂f/∂x = 2x - 2 = 0  →  x = 1")
        print("∂f/∂y = 2y - 4 = 0  →  y = 2")
        print("Critical point: (1, 2)")
        
        print("\nb) Classification using Hessian:")
        print("∂²f/∂x² = 2")
        print("∂²f/∂y² = 2")
        print("∂²f/∂x∂y = 0")
        print("\nH = [ 2   0 ]")
        print("    [ 0   2 ]")
        
        H = np.array([[2, 0], [0, 2]])
        eigenvalues = np.linalg.eigvals(H)
        
        print(f"\nEigenvalues: {eigenvalues}")
        print("Both positive → LOCAL MINIMUM")
        
        # Verify
        def f(x, y):
            return x**2 + y**2 - 2*x - 4*y + 5
        
        print(f"\nf(1, 2) = {f(1, 2)} (minimum value)")
    
    def exercise_5_newton_2d(self):
        """
        Exercise 5: Newton's Method
        
        Use Newton's method to find minimum of:
        f(x, y) = x² + 4y² + 2x - 8y
        Starting from (0, 0)
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Newton's Method")
        
        print("\nf(x, y) = x² + 4y² + 2x - 8y")
        print("∇f = [2x + 2, 8y - 8]ᵀ")
        print("H = [ 2   0 ]  (constant)")
        print("    [ 0   8 ]")
        
        def grad_f(x):
            return np.array([2*x[0] + 2, 8*x[1] - 8])
        
        H = np.array([[2, 0], [0, 8]])
        H_inv = np.linalg.inv(H)
        
        x = np.array([0.0, 0.0])
        
        print(f"\nStarting: x = {x}")
        print("\nNewton iteration: x_new = x - H⁻¹∇f")
        
        for i in range(3):
            g = grad_f(x)
            delta = H_inv @ g
            x_new = x - delta
            print(f"Iter {i}: x = {x}, ∇f = {g}, Δx = {-delta}, x_new = {x_new}")
            x = x_new
        
        print(f"\nMinimum at: {x}")
        print("(Setting ∇f = 0: x = -1, y = 1)")
    
    def exercise_6_softmax_jacobian(self):
        """
        Exercise 6: Softmax Jacobian
        
        For 3-class softmax, verify that:
        a) J is symmetric
        b) Each row sums to 0
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Softmax Jacobian Properties")
        
        def softmax(z):
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        
        def softmax_jacobian(p):
            return np.diag(p) - np.outer(p, p)
        
        z = np.array([1.0, 2.0, 3.0])
        p = softmax(z)
        J = softmax_jacobian(p)
        
        print(f"z = {z}")
        print(f"p = softmax(z) = {p.round(4)}")
        print(f"\nJacobian J:\n{J.round(6)}")
        
        print("\na) Symmetric check:")
        print(f"   J = Jᵀ? {np.allclose(J, J.T)}")
        print(f"   ||J - Jᵀ|| = {np.linalg.norm(J - J.T):.2e}")
        
        print("\nb) Row sums:")
        row_sums = J.sum(axis=1)
        print(f"   Row sums = {row_sums.round(10)}")
        print("   (All ≈ 0 because Σpᵢ = 1, so Σ ∂pᵢ/∂zⱼ = 0)")
        
        print("\nc) Column sums:")
        col_sums = J.sum(axis=0)
        print(f"   Column sums = {col_sums.round(10)}")
        print("   (Also ≈ 0 by symmetry)")
    
    def exercise_7_hessian_quadratic(self):
        """
        Exercise 7: Hessian of Quadratic Form
        
        Show that for f(x) = xᵀAx where A is symmetric:
        H = 2A
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Hessian of Quadratic Form")
        
        print("\nf(x) = xᵀAx where A is symmetric")
        print("\nComponent form:")
        print("f = Σᵢ Σⱼ xᵢ Aᵢⱼ xⱼ")
        
        print("\n∂f/∂xₖ = Σⱼ Aₖⱼ xⱼ + Σᵢ xᵢ Aᵢₖ")
        print("       = (Ax)ₖ + (Aᵀx)ₖ")
        print("       = 2(Ax)ₖ  (since A = Aᵀ)")
        print("\nSo ∇f = 2Ax")
        
        print("\n∂²f/∂xₖ∂xₗ = 2Aₖₗ")
        print("\nTherefore H = 2A")
        
        # Verify numerically
        A = np.array([[2, 1], [1, 3]])  # Symmetric
        
        def f(x):
            return x @ A @ x
        
        def numerical_hessian(x, h=1e-5):
            n = len(x)
            H = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
                    x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
                    x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
                    x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
                    H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h*h)
            return H
        
        x = np.array([1.0, 2.0])
        H_numerical = numerical_hessian(x)
        H_analytical = 2 * A
        
        print(f"\nA = \n{A}")
        print(f"\n2A = \n{H_analytical}")
        print(f"\nNumerical H = \n{H_numerical.round(4)}")
        print(f"\nMatch: {np.allclose(H_numerical, H_analytical)}")
    
    def exercise_8_saddle_point(self):
        """
        Exercise 8: Saddle Point Detection
        
        Show that f(x, y) = x³ - 3xy² has a saddle point at origin
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Saddle Point Detection")
        
        print("\nf(x, y) = x³ - 3xy²")
        
        print("\n∇f = [3x² - 3y², -6xy]ᵀ")
        print("At (0, 0): ∇f = [0, 0]ᵀ ✓ (critical point)")
        
        print("\nH = [ 6x   -6y ]")
        print("    [ -6y  -6x ]")
        
        print("\nAt (0, 0):")
        print("H = [ 0   0 ]")
        print("    [ 0   0 ]")
        
        print("\nThe Hessian is zero! Need higher-order analysis.")
        
        print("\nAlternative: Check function values near origin")
        
        def f(x, y):
            return x**3 - 3*x*y**2
        
        # Along x-axis (y=0): f(x, 0) = x³
        print("\nAlong x-axis (y=0): f(x, 0) = x³")
        print("  x > 0: f > 0 (increasing)")
        print("  x < 0: f < 0 (decreasing)")
        
        # Along line y=x: f(x, x) = x³ - 3x³ = -2x³
        print("\nAlong y=x: f(x, x) = -2x³")
        print("  x > 0: f < 0 (decreasing)")
        print("  x < 0: f > 0 (increasing)")
        
        print("\nFunction increases in some directions, decreases in others")
        print("→ SADDLE POINT at origin")
    
    def exercise_9_layer_jacobian(self):
        """
        Exercise 9: Neural Network Layer Jacobian
        
        For a layer y = σ(Wx + b), compute ∂y/∂x
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Neural Network Layer Jacobian")
        
        print("\ny = σ(Wx + b) = σ(z) where z = Wx + b")
        print("\nBy chain rule:")
        print("∂y/∂x = (∂y/∂z)(∂z/∂x)")
        print("      = diag(σ'(z)) · W")
        
        print("\nFor sigmoid σ: σ'(z) = σ(z)(1 - σ(z))")
        
        print("\nSo ∂y/∂x = diag(y ⊙ (1-y)) · W")
        
        # Example
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        W = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3
        b = np.array([0.1, 0.2])
        x = np.array([1.0, 2.0, 3.0])
        
        z = W @ x + b
        y = sigmoid(z)
        
        # Jacobian
        sigma_prime = y * (1 - y)
        J = np.diag(sigma_prime) @ W
        
        print(f"\nW (2×3):\n{W}")
        print(f"x = {x}")
        print(f"z = Wx + b = {z.round(4)}")
        print(f"y = σ(z) = {y.round(4)}")
        print(f"σ'(z) = {sigma_prime.round(4)}")
        print(f"\n∂y/∂x (2×3):\n{J.round(4)}")
        
        # Verify numerically
        h = 1e-7
        J_num = np.zeros((2, 3))
        for j in range(3):
            x_plus = x.copy()
            x_plus[j] += h
            y_plus = sigmoid(W @ x_plus + b)
            J_num[:, j] = (y_plus - y) / h
        
        print(f"\nNumerical Jacobian:\n{J_num.round(4)}")
    
    def exercise_10_hessian_eigenvalues(self):
        """
        Exercise 10: Hessian Eigenvalue Analysis
        
        For f(x, y) = x⁴ - 2x²y² + y⁴:
        a) Find all critical points
        b) Analyze Hessian eigenvalues at each
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Hessian Eigenvalue Analysis")
        
        print("\nf(x, y) = x⁴ - 2x²y² + y⁴ = (x² - y²)²")
        
        print("\n∂f/∂x = 4x³ - 4xy² = 4x(x² - y²)")
        print("∂f/∂y = -4x²y + 4y³ = 4y(y² - x²)")
        
        print("\na) Critical points where ∇f = 0:")
        print("  4x(x² - y²) = 0 → x = 0 or x² = y²")
        print("  4y(y² - x²) = 0 → y = 0 or y² = x²")
        print("\nCritical points: (0, 0) and all (t, t), (t, -t) for any t")
        
        print("\nb) Hessian:")
        print("∂²f/∂x² = 12x² - 4y²")
        print("∂²f/∂y² = -4x² + 12y²")
        print("∂²f/∂x∂y = -8xy")
        
        def hessian(x, y):
            return np.array([
                [12*x**2 - 4*y**2, -8*x*y],
                [-8*x*y, -4*x**2 + 12*y**2]
            ])
        
        points = [(0, 0), (1, 1), (1, -1), (2, 2)]
        
        print("\nAnalysis at key points:")
        for pt in points:
            x, y = pt
            H = hessian(x, y)
            eigs = np.linalg.eigvals(H)
            
            if np.all(eigs > 0):
                classification = "Local Minimum"
            elif np.all(eigs < 0):
                classification = "Local Maximum"
            elif np.any(eigs == 0):
                classification = "Degenerate"
            else:
                classification = "Saddle Point"
            
            print(f"\n  ({x}, {y}):")
            print(f"    H = \n    {H}")
            print(f"    Eigenvalues: {eigs.round(4)}")
            print(f"    Classification: {classification}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = JacobianHessianExercises()
    
    print("JACOBIANS AND HESSIANS EXERCISES")
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
