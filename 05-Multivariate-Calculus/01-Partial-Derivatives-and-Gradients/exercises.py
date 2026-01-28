"""
Partial Derivatives and Gradients - Exercises
==============================================
Practice problems for multivariate calculus concepts.
"""

import numpy as np


class GradientExercises:
    """Exercises for partial derivatives and gradients."""
    
    def exercise_1_partial_derivatives(self):
        """
        Exercise 1: Partial Derivatives
        
        Find all partial derivatives of:
        a) f(x, y) = x²y³ + 2xy - y
        b) f(x, y, z) = xyz + e^(xy)
        c) f(x, y) = ln(x² + y²)
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Partial Derivatives")
        
        print("\na) f(x, y) = x²y³ + 2xy - y")
        print("   ∂f/∂x = 2xy³ + 2y")
        print("   ∂f/∂y = 3x²y² + 2x - 1")
        
        print("\nb) f(x, y, z) = xyz + e^(xy)")
        print("   ∂f/∂x = yz + ye^(xy)")
        print("   ∂f/∂y = xz + xe^(xy)")
        print("   ∂f/∂z = xy")
        
        print("\nc) f(x, y) = ln(x² + y²)")
        print("   ∂f/∂x = 2x/(x² + y²)")
        print("   ∂f/∂y = 2y/(x² + y²)")
        
        # Numerical verification for (a)
        def f_a(x, y):
            return x**2 * y**3 + 2*x*y - y
        
        x, y, h = 2.0, 3.0, 1e-7
        df_dx_num = (f_a(x+h, y) - f_a(x-h, y)) / (2*h)
        df_dy_num = (f_a(x, y+h) - f_a(x, y-h)) / (2*h)
        
        df_dx_analytic = 2*x*y**3 + 2*y
        df_dy_analytic = 3*x**2*y**2 + 2*x - 1
        
        print(f"\n--- Verification at (2, 3) ---")
        print(f"∂f/∂x: analytic = {df_dx_analytic}, numerical = {df_dx_num:.4f}")
        print(f"∂f/∂y: analytic = {df_dy_analytic}, numerical = {df_dy_num:.4f}")
    
    def exercise_2_gradient(self):
        """
        Exercise 2: Gradient Computation
        
        Find the gradient ∇f for:
        a) f(x, y) = x² + 4y² - 4x + 8y
        b) f(x, y, z) = x² + y² + z² (distance squared from origin)
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Gradient Computation")
        
        print("\na) f(x, y) = x² + 4y² - 4x + 8y")
        print("   ∂f/∂x = 2x - 4")
        print("   ∂f/∂y = 8y + 8")
        print("   ∇f = [2x - 4, 8y + 8]ᵀ")
        
        print("\n   Setting ∇f = 0:")
        print("   2x - 4 = 0 → x = 2")
        print("   8y + 8 = 0 → y = -1")
        print("   Critical point: (2, -1)")
        
        print("\nb) f(x, y, z) = x² + y² + z²")
        print("   ∇f = [2x, 2y, 2z]ᵀ = 2[x, y, z]ᵀ = 2r")
        print("   Gradient points radially outward from origin")
        print("   Minimum at origin: ∇f(0,0,0) = 0")
    
    def exercise_3_directional_derivative(self):
        """
        Exercise 3: Directional Derivative
        
        For f(x, y) = x² - xy + y²:
        a) Find ∇f at (1, 1)
        b) Find D_u f at (1, 1) in direction u = (1, 1)/√2
        c) Find the direction of maximum increase
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Directional Derivative")
        
        print("f(x, y) = x² - xy + y²")
        print("∇f = [2x - y, -x + 2y]ᵀ")
        
        print("\na) At (1, 1):")
        grad = np.array([2*1 - 1, -1 + 2*1])
        print(f"   ∇f(1,1) = {grad}")
        
        print("\nb) Directional derivative in u = (1,1)/√2:")
        u = np.array([1, 1]) / np.sqrt(2)
        D_u = np.dot(grad, u)
        print(f"   u = {u}")
        print(f"   D_u f = ∇f · u = {D_u:.4f}")
        
        print("\nc) Direction of maximum increase:")
        max_dir = grad / np.linalg.norm(grad)
        print(f"   Direction: ∇f/||∇f|| = {max_dir}")
        print(f"   Maximum rate: ||∇f|| = {np.linalg.norm(grad):.4f}")
    
    def exercise_4_gradient_descent(self):
        """
        Exercise 4: Gradient Descent
        
        Implement gradient descent to minimize:
        f(x, y) = (x - 3)² + 4(y + 1)²
        
        Start from (0, 0), use learning rate 0.1
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Gradient Descent")
        
        print("f(x, y) = (x - 3)² + 4(y + 1)²")
        print("∇f = [2(x - 3), 8(y + 1)]ᵀ")
        print("Minimum at (3, -1)")
        
        def f(x):
            return (x[0] - 3)**2 + 4*(x[1] + 1)**2
        
        def grad_f(x):
            return np.array([2*(x[0] - 3), 8*(x[1] + 1)])
        
        x = np.array([0.0, 0.0])
        lr = 0.1
        
        print(f"\nStarting: x = {x}, f(x) = {f(x)}")
        print("\nGradient descent iterations:")
        
        for i in range(10):
            g = grad_f(x)
            x = x - lr * g
            print(f"  k={i+1}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {f(x):.6f}")
        
        print(f"\nFinal: x = {x}, Expected: [3, -1]")
    
    def exercise_5_linear_regression(self):
        """
        Exercise 5: Linear Regression Gradient
        
        For MSE loss L(w, b) = (1/n) Σ(wx_i + b - y_i)²
        
        Derive ∂L/∂w and ∂L/∂b
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Linear Regression Gradient")
        
        print("L(w, b) = (1/n) Σ(wx_i + b - y_i)²")
        
        print("\nLet r_i = wx_i + b - y_i (residual)")
        print("L = (1/n) Σ r_i²")
        
        print("\n∂L/∂w = (1/n) Σ 2r_i · ∂r_i/∂w")
        print("      = (1/n) Σ 2r_i · x_i")
        print("      = (2/n) Σ (wx_i + b - y_i) · x_i")
        
        print("\n∂L/∂b = (1/n) Σ 2r_i · ∂r_i/∂b")
        print("      = (1/n) Σ 2r_i · 1")
        print("      = (2/n) Σ (wx_i + b - y_i)")
        
        # Example
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2*x + 3 + 0.1*np.random.randn(n)
        
        w, b = 0.0, 0.0
        lr = 0.1
        
        print("\n--- Implementation ---")
        for i in range(100):
            pred = w*x + b
            residual = pred - y
            dw = (2/n) * np.sum(residual * x)
            db = (2/n) * np.sum(residual)
            w -= lr * dw
            b -= lr * db
        
        print(f"Final w = {w:.4f} (true: 2)")
        print(f"Final b = {b:.4f} (true: 3)")
    
    def exercise_6_softmax_gradient(self):
        """
        Exercise 6: Softmax Gradient
        
        For softmax: p_i = exp(z_i) / Σ exp(z_j)
        
        Show that ∂p_i/∂z_j = p_i(δ_ij - p_j)
        where δ_ij is Kronecker delta
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Softmax Gradient")
        
        print("Softmax: p_i = exp(z_i) / S where S = Σ exp(z_j)")
        
        print("\nCase 1: i = j")
        print("∂p_i/∂z_i = [exp(z_i)·S - exp(z_i)·exp(z_i)] / S²")
        print("         = exp(z_i)/S - exp(z_i)²/S²")
        print("         = p_i - p_i²")
        print("         = p_i(1 - p_i)")
        print("         = p_i(δ_ii - p_i) ✓")
        
        print("\nCase 2: i ≠ j")
        print("∂p_i/∂z_j = [0 - exp(z_i)·exp(z_j)] / S²")
        print("         = -exp(z_i)·exp(z_j) / S²")
        print("         = -p_i · p_j")
        print("         = p_i(0 - p_j)")
        print("         = p_i(δ_ij - p_j) ✓")
        
        print("\nCombined: ∂p_i/∂z_j = p_i(δ_ij - p_j)")
        
        # Numerical verification
        def softmax(z):
            exp_z = np.exp(z - np.max(z))
            return exp_z / np.sum(exp_z)
        
        z = np.array([1.0, 2.0, 3.0])
        p = softmax(z)
        
        print("\n--- Numerical verification ---")
        print(f"z = {z}, p = {p.round(4)}")
        
        h = 1e-7
        for i in range(3):
            for j in range(3):
                z_plus = z.copy()
                z_plus[j] += h
                z_minus = z.copy()
                z_minus[j] -= h
                numerical = (softmax(z_plus)[i] - softmax(z_minus)[i]) / (2*h)
                analytic = p[i] * ((1 if i==j else 0) - p[j])
                print(f"  ∂p_{i}/∂z_{j}: analytic = {analytic:.6f}, numerical = {numerical:.6f}")
    
    def exercise_7_chain_rule_multivar(self):
        """
        Exercise 7: Multivariate Chain Rule
        
        Let f(x, y) = x² + y², where x = r cos(θ), y = r sin(θ)
        
        Find ∂f/∂r and ∂f/∂θ
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Multivariate Chain Rule")
        
        print("f(x, y) = x² + y²")
        print("x = r cos(θ), y = r sin(θ)")
        
        print("\nUsing chain rule:")
        print("∂f/∂r = (∂f/∂x)(∂x/∂r) + (∂f/∂y)(∂y/∂r)")
        print("      = 2x·cos(θ) + 2y·sin(θ)")
        print("      = 2r·cos²(θ) + 2r·sin²(θ)")
        print("      = 2r")
        
        print("\n∂f/∂θ = (∂f/∂x)(∂x/∂θ) + (∂f/∂y)(∂y/∂θ)")
        print("      = 2x·(-r·sin(θ)) + 2y·(r·cos(θ))")
        print("      = -2r²·cos(θ)·sin(θ) + 2r²·sin(θ)·cos(θ)")
        print("      = 0")
        
        print("\nVerification: f = x² + y² = r², so:")
        print("∂f/∂r = 2r ✓")
        print("∂f/∂θ = 0 ✓ (f doesn't depend on θ)")
    
    def exercise_8_gradient_checking(self):
        """
        Exercise 8: Gradient Checking
        
        Implement gradient checking for:
        f(x, y, z) = x·sin(y) + z²
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Gradient Checking")
        
        def f(x):
            return x[0] * np.sin(x[1]) + x[2]**2
        
        def grad_f_analytic(x):
            return np.array([
                np.sin(x[1]),        # ∂f/∂x
                x[0] * np.cos(x[1]), # ∂f/∂y
                2 * x[2]             # ∂f/∂z
            ])
        
        def grad_f_numerical(x, h=1e-7):
            grad = np.zeros(3)
            for i in range(3):
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
            return grad
        
        x = np.array([2.0, np.pi/4, 1.0])
        
        g_analytic = grad_f_analytic(x)
        g_numerical = grad_f_numerical(x)
        
        print(f"Point: x = {x}")
        print(f"\nAnalytical gradient: {g_analytic}")
        print(f"Numerical gradient:  {g_numerical}")
        
        rel_error = np.linalg.norm(g_analytic - g_numerical) / (
            np.linalg.norm(g_analytic) + np.linalg.norm(g_numerical) + 1e-10
        )
        print(f"\nRelative error: {rel_error:.2e}")
        print(f"Check {'PASSED' if rel_error < 1e-5 else 'FAILED'}")
    
    def exercise_9_mse_matrix_gradient(self):
        """
        Exercise 9: MSE Matrix Gradient
        
        For L(w) = ||Xw - y||²
        
        Derive ∇_w L using matrix calculus
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: MSE Matrix Gradient")
        
        print("L(w) = ||Xw - y||² = (Xw - y)ᵀ(Xw - y)")
        print("\nExpanding:")
        print("L = wᵀXᵀXw - 2yᵀXw + yᵀy")
        
        print("\nUsing matrix calculus rules:")
        print("  ∂(wᵀAw)/∂w = (A + Aᵀ)w")
        print("  ∂(bᵀw)/∂w = b")
        
        print("\n∇_w L = 2XᵀXw - 2Xᵀy")
        print("      = 2Xᵀ(Xw - y)")
        
        print("\nSetting ∇_w L = 0:")
        print("XᵀXw = Xᵀy")
        print("w* = (XᵀX)⁻¹Xᵀy  (Normal equation)")
        
        # Numerical verification
        np.random.seed(42)
        X = np.random.randn(100, 3)
        w_true = np.array([1, 2, 3])
        y = X @ w_true + 0.1*np.random.randn(100)
        
        # Analytical solution
        w_closed = np.linalg.solve(X.T @ X, X.T @ y)
        print(f"\n--- Verification ---")
        print(f"True w: {w_true}")
        print(f"Estimated w (closed form): {w_closed.round(4)}")
    
    def exercise_10_neural_layer_gradient(self):
        """
        Exercise 10: Neural Network Layer Gradient
        
        For a layer: y = σ(Wx + b)
        Loss: L = ||y - t||²
        
        Derive ∂L/∂W and ∂L/∂b
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Neural Network Layer Gradient")
        
        print("Forward: z = Wx + b")
        print("         y = σ(z)")
        print("         L = ||y - t||²")
        
        print("\nBackward (chain rule):")
        print("∂L/∂y = 2(y - t)")
        print("∂y/∂z = σ'(z) = σ(z)(1 - σ(z))  [for sigmoid]")
        print("∂L/∂z = ∂L/∂y ⊙ ∂y/∂z = 2(y - t) ⊙ σ'(z)")
        
        print("\nLet δ = ∂L/∂z (error signal)")
        
        print("\n∂z/∂W: z_i = Σ_j W_ij x_j + b_i")
        print("       ∂z_i/∂W_ij = x_j")
        print("∂L/∂W_ij = δ_i · x_j")
        print("∂L/∂W = δ · xᵀ  (outer product)")
        
        print("\n∂L/∂b = δ")
        
        # Implementation
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        
        np.random.seed(42)
        x = np.array([1.0, 2.0])
        t = np.array([0.8, 0.2])
        W = np.random.randn(2, 2) * 0.1
        b = np.zeros(2)
        
        # Forward
        z = W @ x + b
        y = sigmoid(z)
        L = np.sum((y - t)**2)
        
        # Backward
        dL_dy = 2 * (y - t)
        dy_dz = sigmoid(z) * (1 - sigmoid(z))
        delta = dL_dy * dy_dz
        
        dL_dW = np.outer(delta, x)
        dL_db = delta
        
        print("\n--- Implementation ---")
        print(f"x = {x}")
        print(f"y = {y.round(4)}")
        print(f"t = {t}")
        print(f"Loss = {L:.4f}")
        print(f"∂L/∂W =\n{dL_dW.round(4)}")
        print(f"∂L/∂b = {dL_db.round(4)}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = GradientExercises()
    
    print("PARTIAL DERIVATIVES AND GRADIENTS EXERCISES")
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
