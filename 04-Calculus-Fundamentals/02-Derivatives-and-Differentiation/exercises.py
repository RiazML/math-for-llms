"""
Derivatives and Differentiation - Exercises
==========================================
Practice problems for differentiation concepts.
"""

import numpy as np


class DerivativeExercises:
    """Exercises for derivatives and differentiation."""
    
    def exercise_1_basic_derivatives(self):
        """
        Exercise 1: Basic Derivatives
        
        Find the derivatives:
        a) f(x) = 5x³ - 2x² + 4x - 7
        b) g(x) = √x + 1/x²
        c) h(x) = (x² + 1)⁵
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Derivatives")
        
        print("\na) f(x) = 5x³ - 2x² + 4x - 7")
        print("   f'(x) = 15x² - 4x + 4")
        
        def df_a(x): return 15*x**2 - 4*x + 4
        print(f"   Check at x=2: f'(2) = 15(4) - 4(2) + 4 = {df_a(2)}")
        
        print("\nb) g(x) = √x + 1/x² = x^(1/2) + x^(-2)")
        print("   g'(x) = (1/2)x^(-1/2) + (-2)x^(-3)")
        print("        = 1/(2√x) - 2/x³")
        
        def dg_b(x): return 0.5*x**(-0.5) - 2*x**(-3)
        print(f"   Check at x=4: g'(4) = 1/(2·2) - 2/64 = 0.25 - 0.03125 = {dg_b(4):.5f}")
        
        print("\nc) h(x) = (x² + 1)⁵")
        print("   Using chain rule: h'(x) = 5(x² + 1)⁴ · (2x)")
        print("                          = 10x(x² + 1)⁴")
        
        def dh_c(x): return 10*x*(x**2 + 1)**4
        print(f"   Check at x=1: h'(1) = 10(1)(2)⁴ = 10·16 = {dh_c(1)}")
    
    def exercise_2_chain_rule(self):
        """
        Exercise 2: Chain Rule
        
        Find the derivatives:
        a) f(x) = e^(x²)
        b) g(x) = ln(sin(x))
        c) h(x) = sin²(3x)
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Chain Rule")
        
        print("\na) f(x) = e^(x²)")
        print("   f'(x) = e^(x²) · d/dx[x²]")
        print("        = e^(x²) · 2x")
        print("        = 2x·e^(x²)")
        
        def df_a(x): return 2*x*np.exp(x**2)
        x = 1
        h = 0.0001
        numerical = (np.exp((x+h)**2) - np.exp(x**2)) / h
        print(f"   Verify at x={x}: analytical={df_a(x):.6f}, numerical={numerical:.6f}")
        
        print("\nb) g(x) = ln(sin(x))")
        print("   g'(x) = (1/sin(x)) · cos(x)")
        print("        = cos(x)/sin(x)")
        print("        = cot(x)")
        
        def dg_b(x): return np.cos(x)/np.sin(x)
        x = np.pi/4
        print(f"   At x=π/4: g'(π/4) = cot(π/4) = {dg_b(x):.6f}")
        
        print("\nc) h(x) = sin²(3x) = [sin(3x)]²")
        print("   h'(x) = 2·sin(3x) · cos(3x) · 3")
        print("        = 6·sin(3x)·cos(3x)")
        print("        = 3·sin(6x)  [using sin(2θ) = 2sin(θ)cos(θ)]")
        
        def dh_c(x): return 3*np.sin(6*x)
        x = np.pi/12
        print(f"   At x=π/12: h'(π/12) = 3·sin(π/2) = {dh_c(x):.6f}")
    
    def exercise_3_product_quotient(self):
        """
        Exercise 3: Product and Quotient Rules
        
        Find the derivatives:
        a) f(x) = x·e^x
        b) g(x) = x²·sin(x)
        c) h(x) = (x² + 1)/(x - 1)
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Product and Quotient Rules")
        
        print("\na) f(x) = x·e^x")
        print("   Product rule: f'(x) = 1·e^x + x·e^x = e^x(1 + x)")
        
        def f_a(x): return x*np.exp(x)
        def df_a(x): return np.exp(x)*(1 + x)
        x = 2
        h = 0.0001
        numerical = (f_a(x+h) - f_a(x)) / h
        print(f"   At x={x}: analytical={df_a(x):.6f}, numerical={numerical:.6f}")
        
        print("\nb) g(x) = x²·sin(x)")
        print("   g'(x) = 2x·sin(x) + x²·cos(x)")
        
        def g_b(x): return x**2 * np.sin(x)
        def dg_b(x): return 2*x*np.sin(x) + x**2*np.cos(x)
        x = np.pi/2
        numerical = (g_b(x+h) - g_b(x)) / h
        print(f"   At x=π/2: analytical={dg_b(x):.6f}, numerical={numerical:.6f}")
        
        print("\nc) h(x) = (x² + 1)/(x - 1)")
        print("   Quotient rule: h'(x) = [2x(x-1) - (x²+1)·1] / (x-1)²")
        print("                       = [2x² - 2x - x² - 1] / (x-1)²")
        print("                       = (x² - 2x - 1) / (x-1)²")
        
        def h_c(x): return (x**2 + 1) / (x - 1)
        def dh_c(x): return (x**2 - 2*x - 1) / (x - 1)**2
        x = 3
        numerical = (h_c(x+h) - h_c(x)) / h
        print(f"   At x={x}: analytical={dh_c(x):.6f}, numerical={numerical:.6f}")
    
    def exercise_4_sigmoid_derivation(self):
        """
        Exercise 4: Sigmoid Derivative
        
        Prove that if σ(x) = 1/(1 + e^(-x)), then:
        σ'(x) = σ(x)(1 - σ(x))
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Sigmoid Derivative")
        
        print("σ(x) = 1/(1 + e^(-x)) = (1 + e^(-x))^(-1)")
        print("\nUsing chain rule:")
        print("σ'(x) = -1·(1 + e^(-x))^(-2) · d/dx[1 + e^(-x)]")
        print("     = -(1 + e^(-x))^(-2) · (-e^(-x))")
        print("     = e^(-x) / (1 + e^(-x))²")
        
        print("\nNow show this equals σ(x)(1 - σ(x)):")
        print("\nσ(x) = 1/(1 + e^(-x))")
        print("\n1 - σ(x) = 1 - 1/(1 + e^(-x))")
        print("         = [(1 + e^(-x)) - 1] / (1 + e^(-x))")
        print("         = e^(-x) / (1 + e^(-x))")
        
        print("\nσ(x)(1 - σ(x)) = [1/(1 + e^(-x))] · [e^(-x)/(1 + e^(-x))]")
        print("              = e^(-x) / (1 + e^(-x))²")
        print("              = σ'(x) ✓")
        
        # Numerical verification
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        print("\nNumerical verification:")
        for x in [-1, 0, 1]:
            h = 0.0001
            numerical = (sigmoid(x+h) - sigmoid(x)) / h
            s = sigmoid(x)
            analytical = s * (1 - s)
            print(f"  x={x:2d}: σ(1-σ) = {analytical:.6f}, numerical = {numerical:.6f}")
    
    def exercise_5_implicit(self):
        """
        Exercise 5: Implicit Differentiation
        
        Find dy/dx for x² + y² = 25 (circle).
        Then find the slope of the tangent at (3, 4).
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Implicit Differentiation")
        
        print("x² + y² = 25")
        print("\nDifferentiate both sides with respect to x:")
        print("d/dx[x²] + d/dx[y²] = d/dx[25]")
        print("2x + 2y·(dy/dx) = 0")
        print("\nSolve for dy/dx:")
        print("dy/dx = -2x/(2y) = -x/y")
        
        print("\nAt point (3, 4):")
        x, y = 3, 4
        slope = -x/y
        print(f"dy/dx = -{x}/{y} = {slope}")
        
        print("\nThe tangent line at (3, 4) has slope -3/4")
        print("Equation: y - 4 = -3/4(x - 3)")
        print("         y = -3x/4 + 9/4 + 4 = -3x/4 + 25/4")
    
    def exercise_6_optimization(self):
        """
        Exercise 6: Finding Extrema
        
        Find the local maxima and minima of:
        f(x) = x⁴ - 8x² + 16
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Finding Extrema")
        
        print("f(x) = x⁴ - 8x² + 16")
        
        print("\n--- First derivative ---")
        print("f'(x) = 4x³ - 16x = 4x(x² - 4) = 4x(x-2)(x+2)")
        print("Critical points: x = 0, x = 2, x = -2")
        
        print("\n--- Second derivative ---")
        print("f''(x) = 12x² - 16")
        
        print("\n--- Classification ---")
        def f(x): return x**4 - 8*x**2 + 16
        def d2f(x): return 12*x**2 - 16
        
        critical_points = [-2, 0, 2]
        for x in critical_points:
            second_deriv = d2f(x)
            if second_deriv > 0:
                classification = "Local minimum"
            elif second_deriv < 0:
                classification = "Local maximum"
            else:
                classification = "Inconclusive"
            print(f"  x = {x:2d}: f''(x) = {second_deriv:3d}, f(x) = {f(x):2.0f}, {classification}")
        
        print("\nConclusion:")
        print("  x = ±2: Local minima with f(±2) = 0")
        print("  x = 0:  Local maximum with f(0) = 16")
    
    def exercise_7_gradient_descent(self):
        """
        Exercise 7: Gradient Descent
        
        Use gradient descent to find the minimum of:
        f(x) = x² - 4x + 5
        
        Start at x = 0, use learning rate α = 0.1
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Gradient Descent")
        
        print("f(x) = x² - 4x + 5")
        print("f'(x) = 2x - 4")
        print("\nAnalytical minimum: f'(x) = 0 → x = 2")
        
        def f(x): return x**2 - 4*x + 5
        def df(x): return 2*x - 4
        
        x = 0
        alpha = 0.1
        
        print(f"\nGradient descent from x₀ = {x}, α = {alpha}")
        print("\nStep | x        | f(x)    | f'(x)")
        print("-" * 42)
        
        for i in range(15):
            print(f" {i:2d}  | {x:8.5f} | {f(x):7.5f} | {df(x):7.5f}")
            x = x - alpha * df(x)
        
        print(f"\nConverged to x ≈ {x:.5f} (analytical: x = 2)")
    
    def exercise_8_second_derivative(self):
        """
        Exercise 8: Second Derivative and Concavity
        
        For f(x) = x³ - 3x:
        a) Find intervals where f is concave up/down
        b) Find inflection points
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Second Derivative and Concavity")
        
        print("f(x) = x³ - 3x")
        print("\nf'(x) = 3x² - 3")
        print("f''(x) = 6x")
        
        print("\n--- Concavity ---")
        print("f''(x) > 0 when 6x > 0 → x > 0: Concave UP")
        print("f''(x) < 0 when 6x < 0 → x < 0: Concave DOWN")
        
        print("\n--- Inflection point ---")
        print("f''(x) = 0 when x = 0")
        print("Sign changes at x = 0 → Inflection point at (0, f(0)) = (0, 0)")
        
        def f(x): return x**3 - 3*x
        
        print("\nVerification with values:")
        print("  x = -1: f''(-1) = -6 < 0 (concave down)")
        print("  x = 0:  f''(0) = 0 (inflection)")
        print("  x = 1:  f''(1) = 6 > 0 (concave up)")
    
    def exercise_9_leaky_relu(self):
        """
        Exercise 9: Leaky ReLU Derivative
        
        Leaky ReLU: f(x) = { x      if x > 0
                          { αx     if x ≤ 0
        
        For α = 0.01, find f'(x) and verify numerically.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Leaky ReLU Derivative")
        
        alpha = 0.01
        
        print(f"Leaky ReLU (α = {alpha}):")
        print(f"  f(x) = x     if x > 0")
        print(f"  f(x) = {alpha}x  if x ≤ 0")
        
        print(f"\nDerivative:")
        print(f"  f'(x) = 1     if x > 0")
        print(f"  f'(x) = {alpha}  if x ≤ 0")
        
        def leaky_relu(x, alpha=0.01):
            return np.where(x > 0, x, alpha * x)
        
        def leaky_relu_derivative(x, alpha=0.01):
            return np.where(x > 0, 1.0, alpha)
        
        print("\nNumerical verification:")
        h = 0.0001
        for x in [-2, -1, -0.5, 0.5, 1, 2]:
            numerical = (leaky_relu(x + h) - leaky_relu(x)) / h
            analytical = leaky_relu_derivative(x)
            print(f"  x = {x:4.1f}: f'(x) = {analytical:.4f}, numerical = {numerical:.4f}")
        
        print(f"\nAdvantage over ReLU: gradient is {alpha} (not 0) for x < 0")
        print("This helps prevent 'dying ReLU' problem!")
    
    def exercise_10_loss_derivative(self):
        """
        Exercise 10: MSE Loss Derivative
        
        For MSE loss: L = (1/n)Σ(yᵢ - ŷᵢ)²
        
        If ŷᵢ = wxᵢ (linear prediction), find ∂L/∂w.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: MSE Loss Derivative")
        
        print("MSE Loss: L = (1/n)Σ(yᵢ - ŷᵢ)²")
        print("Prediction: ŷᵢ = w·xᵢ")
        print("\nSo: L = (1/n)Σ(yᵢ - w·xᵢ)²")
        
        print("\n--- Finding ∂L/∂w ---")
        print("∂L/∂w = (1/n)Σ ∂/∂w[(yᵢ - w·xᵢ)²]")
        print("     = (1/n)Σ 2(yᵢ - w·xᵢ)·(-xᵢ)")
        print("     = -(2/n)Σ xᵢ(yᵢ - w·xᵢ)")
        print("     = (2/n)Σ xᵢ(ŷᵢ - yᵢ)")
        
        print("\n--- Numerical verification ---")
        np.random.seed(42)
        n = 5
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([2, 4, 5, 4, 5], dtype=float)
        w = 1.0
        
        def mse(w, x, y):
            y_hat = w * x
            return np.mean((y - y_hat)**2)
        
        def dmse_dw(w, x, y):
            y_hat = w * x
            return (2/len(x)) * np.sum(x * (y_hat - y))
        
        print(f"x = {x}")
        print(f"y = {y}")
        print(f"w = {w}")
        
        h = 0.0001
        numerical = (mse(w + h, x, y) - mse(w, x, y)) / h
        analytical = dmse_dw(w, x, y)
        
        print(f"\n∂L/∂w analytical: {analytical:.6f}")
        print(f"∂L/∂w numerical:  {numerical:.6f}")
        
        print("\n--- Gradient descent update ---")
        print(f"w_new = w - α·∂L/∂w")
        print(f"This moves w in the direction that decreases MSE!")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = DerivativeExercises()
    
    print("DERIVATIVES AND DIFFERENTIATION EXERCISES")
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
