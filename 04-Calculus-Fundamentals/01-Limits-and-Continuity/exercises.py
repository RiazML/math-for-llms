"""
Limits and Continuity - Exercises
=================================
Practice problems for limits and continuity concepts.
"""

import numpy as np


class LimitExercises:
    """Exercises for limits and continuity."""
    
    def exercise_1_basic_limits(self):
        """
        Exercise 1: Basic Limits
        
        Evaluate the following limits:
        a) lim(x→3) (x² - 9)/(x - 3)
        b) lim(x→0) sin(5x)/x
        c) lim(x→∞) (3x² + 2x)/(x² + 1)
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Limits")
        
        print("\na) lim(x→3) (x² - 9)/(x - 3)")
        print("   Factor: (x² - 9) = (x-3)(x+3)")
        print("   = lim(x→3) (x-3)(x+3)/(x-3)")
        print("   = lim(x→3) (x+3)")
        print("   = 3 + 3 = 6")
        
        # Verify numerically
        def f_a(x):
            return (x**2 - 9) / (x - 3)
        print(f"   Numerical check: f(3.0001) = {f_a(3.0001):.4f}")
        
        print("\nb) lim(x→0) sin(5x)/x")
        print("   = lim(x→0) 5 · sin(5x)/(5x)")
        print("   = 5 · lim(u→0) sin(u)/u  [where u = 5x]")
        print("   = 5 · 1 = 5")
        
        # Verify
        def f_b(x):
            return np.sin(5*x) / x
        print(f"   Numerical check: f(0.0001) = {f_b(0.0001):.4f}")
        
        print("\nc) lim(x→∞) (3x² + 2x)/(x² + 1)")
        print("   Divide numerator and denominator by x²:")
        print("   = lim(x→∞) (3 + 2/x)/(1 + 1/x²)")
        print("   = (3 + 0)/(1 + 0) = 3")
        
        def f_c(x):
            return (3*x**2 + 2*x) / (x**2 + 1)
        print(f"   Numerical check: f(10000) = {f_c(10000):.6f}")
    
    def exercise_2_one_sided(self):
        """
        Exercise 2: One-Sided Limits
        
        For f(x) = (x - 1)/|x - 1|:
        a) Find lim(x→1⁻) f(x)
        b) Find lim(x→1⁺) f(x)
        c) Does lim(x→1) f(x) exist?
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: One-Sided Limits")
        
        def f(x):
            return (x - 1) / abs(x - 1)
        
        print("\nf(x) = (x - 1)/|x - 1|")
        
        print("\na) lim(x→1⁻) f(x):")
        print("   For x < 1: |x-1| = -(x-1)")
        print("   f(x) = (x-1)/(-(x-1)) = -1")
        for x in [0.9, 0.99, 0.999]:
            print(f"   f({x}) = {f(x)}")
        print("   Left limit = -1")
        
        print("\nb) lim(x→1⁺) f(x):")
        print("   For x > 1: |x-1| = (x-1)")
        print("   f(x) = (x-1)/(x-1) = 1")
        for x in [1.1, 1.01, 1.001]:
            print(f"   f({x}) = {f(x)}")
        print("   Right limit = +1")
        
        print("\nc) Since left limit ≠ right limit,")
        print("   lim(x→1) f(x) does NOT exist")
    
    def exercise_3_lhopital(self):
        """
        Exercise 3: L'Hôpital's Rule
        
        Use L'Hôpital's Rule to evaluate:
        a) lim(x→0) (1 - cos(x))/x²
        b) lim(x→∞) x·e^(-x)
        c) lim(x→0⁺) x·ln(x)
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: L'Hôpital's Rule")
        
        print("\na) lim(x→0) (1 - cos(x))/x²")
        print("   Form: 0/0, apply L'Hôpital")
        print("   = lim(x→0) sin(x)/(2x)")
        print("   Still 0/0, apply again:")
        print("   = lim(x→0) cos(x)/2")
        print("   = 1/2")
        
        def f_a(x):
            return (1 - np.cos(x)) / x**2
        print(f"   Check: f(0.001) = {f_a(0.001):.6f}")
        
        print("\nb) lim(x→∞) x·e^(-x)")
        print("   Rewrite: lim(x→∞) x/e^x (form ∞/∞)")
        print("   L'Hôpital: = lim(x→∞) 1/e^x")
        print("   = 0")
        
        def f_b(x):
            return x * np.exp(-x)
        print(f"   Check: f(100) = {f_b(100):.6e}")
        
        print("\nc) lim(x→0⁺) x·ln(x)")
        print("   Form: 0·(-∞)")
        print("   Rewrite: lim(x→0⁺) ln(x)/(1/x) (form -∞/∞)")
        print("   L'Hôpital: = lim(x→0⁺) (1/x)/(-1/x²)")
        print("   = lim(x→0⁺) (-x)")
        print("   = 0")
        
        def f_c(x):
            return x * np.log(x)
        print(f"   Check: f(0.0001) = {f_c(0.0001):.6f}")
    
    def exercise_4_continuity(self):
        """
        Exercise 4: Continuity Analysis
        
        Determine where each function is discontinuous:
        a) f(x) = (x² - 4)/(x + 2)
        b) f(x) = 1/(x² - 1)
        c) f(x) = floor(x) (greatest integer function)
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Continuity Analysis")
        
        print("\na) f(x) = (x² - 4)/(x + 2)")
        print("   Undefined when x + 2 = 0, i.e., x = -2")
        print("   But: (x² - 4)/(x + 2) = (x-2)(x+2)/(x+2) = x - 2")
        print("   lim(x→-2) f(x) = -4 exists")
        print("   Discontinuity type: Removable (hole at x = -2)")
        
        print("\nb) f(x) = 1/(x² - 1)")
        print("   Undefined when x² - 1 = 0, i.e., x = ±1")
        print("   At x = 1: lim(x→1) = ±∞")
        print("   At x = -1: lim(x→-1) = ±∞")
        print("   Discontinuity type: Infinite (vertical asymptotes)")
        
        print("\nc) f(x) = floor(x)")
        print("   At every integer n:")
        print("   lim(x→n⁻) = n - 1")
        print("   lim(x→n⁺) = n")
        print("   Left ≠ Right")
        print("   Discontinuity type: Jump at every integer")
    
    def exercise_5_squeeze(self):
        """
        Exercise 5: Squeeze Theorem
        
        Use the Squeeze Theorem to find:
        lim(x→∞) sin(x)/x
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Squeeze Theorem")
        
        print("Find: lim(x→∞) sin(x)/x")
        
        print("\nWe know: -1 ≤ sin(x) ≤ 1 for all x")
        print("\nDivide by x (positive for large x):")
        print("-1/x ≤ sin(x)/x ≤ 1/x")
        
        print("\nAs x → ∞:")
        print("lim(x→∞) (-1/x) = 0")
        print("lim(x→∞) (1/x) = 0")
        
        print("\nBy Squeeze Theorem:")
        print("lim(x→∞) sin(x)/x = 0")
        
        # Verify
        def f(x):
            return np.sin(x) / x
        
        print("\nNumerical verification:")
        for x in [10, 100, 1000, 10000]:
            print(f"  f({x}) = {f(x):.6f}")
    
    def exercise_6_piecewise(self):
        """
        Exercise 6: Piecewise Function Continuity
        
        For what value of k is f continuous everywhere?
        
        f(x) = { x² + k,  if x < 2
               { 3x,      if x ≥ 2
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Piecewise Continuity")
        
        print("f(x) = { x² + k,  if x < 2")
        print("       { 3x,      if x ≥ 2")
        
        print("\nFor continuity at x = 2:")
        print("lim(x→2⁻) f(x) = lim(x→2⁺) f(x) = f(2)")
        
        print("\nFrom the left: lim(x→2⁻) (x² + k) = 4 + k")
        print("From the right: lim(x→2⁺) 3x = 6")
        print("At x = 2: f(2) = 3(2) = 6")
        
        print("\nFor continuity: 4 + k = 6")
        print("Therefore: k = 2")
        
        def f(x, k=2):
            if x < 2:
                return x**2 + k
            else:
                return 3*x
        
        print("\nVerification with k = 2:")
        print(f"  f(1.999) = {f(1.999):.4f}")
        print(f"  f(2.000) = {f(2.0):.4f}")
        print(f"  f(2.001) = {f(2.001):.4f}")
    
    def exercise_7_relu_continuity(self):
        """
        Exercise 7: ReLU Continuity (ML)
        
        Show that ReLU(x) = max(0, x) is continuous everywhere.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: ReLU Continuity")
        
        print("ReLU(x) = max(0, x) = { 0,  if x < 0")
        print("                      { x,  if x ≥ 0")
        
        print("\n--- Check continuity at x = 0 ---")
        print("1. f(0) = 0 (defined)")
        print("2. lim(x→0⁻) 0 = 0")
        print("3. lim(x→0⁺) x = 0")
        print("4. Left = Right = f(0) = 0")
        print("Therefore: ReLU is continuous at x = 0 ✓")
        
        print("\n--- Check everywhere else ---")
        print("For x < 0: f(x) = 0 (constant, continuous)")
        print("For x > 0: f(x) = x (linear, continuous)")
        
        print("\nConclusion: ReLU is continuous on all of ℝ")
        
        def relu(x):
            return np.maximum(0, x)
        
        print("\nNumerical verification near x = 0:")
        for x in [-0.001, -0.0001, 0, 0.0001, 0.001]:
            print(f"  ReLU({x:7.4f}) = {relu(x):.4f}")
    
    def exercise_8_exponential_limit(self):
        """
        Exercise 8: Exponential Limit
        
        Prove: lim(n→∞) (1 + r/n)^n = e^r
        
        Verify numerically for r = 2.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Exponential Limit")
        
        print("lim(n→∞) (1 + r/n)^n = e^r")
        
        print("\n--- Proof sketch ---")
        print("Let y = (1 + r/n)^n")
        print("ln(y) = n·ln(1 + r/n)")
        print("\nAs n → ∞, r/n → 0")
        print("Using ln(1+x) ≈ x for small x:")
        print("ln(y) ≈ n·(r/n) = r")
        print("Therefore y → e^r")
        
        r = 2
        print(f"\n--- Numerical verification for r = {r} ---")
        print(f"e^{r} = {np.exp(r):.10f}")
        
        for n in [10, 100, 1000, 10000, 100000]:
            val = (1 + r/n)**n
            error = abs(val - np.exp(r))
            print(f"  n = {n:6d}: (1 + {r}/n)^n = {val:.10f}, error = {error:.2e}")
    
    def exercise_9_gradient_limit(self):
        """
        Exercise 9: Gradient as Limit (ML)
        
        The derivative is defined as:
        f'(x) = lim(h→0) [f(x+h) - f(x)]/h
        
        For f(x) = x³, compute f'(2) using:
        a) The limit definition numerically
        b) The analytical formula
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Gradient as Limit")
        
        def f(x):
            return x**3
        
        x = 2
        
        print(f"f(x) = x³, find f'({x})")
        
        print("\na) Numerical approximation using limit definition:")
        print("   f'(x) = lim(h→0) [f(x+h) - f(x)]/h")
        
        for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
            derivative_approx = (f(x + h) - f(x)) / h
            print(f"   h = {h}: [f({x}+h) - f({x})]/h = {derivative_approx:.10f}")
        
        print("\nb) Analytical formula:")
        print("   f(x) = x³")
        print("   f'(x) = 3x²")
        print(f"   f'({x}) = 3·{x}² = {3*x**2}")
        
        print(f"\nThe numerical limit approaches {3*x**2} ✓")
    
    def exercise_10_cross_entropy_limit(self):
        """
        Exercise 10: Cross-Entropy Limit (ML)
        
        Cross-entropy: H(y, p) = -y·log(p) - (1-y)·log(1-p)
        
        Find lim(p→0⁺) H(1, p) and interpret.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Cross-Entropy Limit")
        
        print("Cross-entropy: H(y, p) = -y·log(p) - (1-y)·log(1-p)")
        
        print("\nFor y = 1 (true label is 1):")
        print("H(1, p) = -1·log(p) - 0·log(1-p)")
        print("       = -log(p)")
        
        print("\nAs p → 0⁺ (predicting probability near 0 for true class 1):")
        print("lim(p→0⁺) H(1, p) = lim(p→0⁺) -log(p) = +∞")
        
        def cross_entropy(y, p):
            return -y * np.log(p) - (1-y) * np.log(1-p)
        
        print("\nNumerical verification:")
        for p in [0.1, 0.01, 0.001, 0.0001]:
            H = cross_entropy(1, p)
            print(f"  p = {p}: H(1, p) = {H:.4f}")
        
        print("\n--- Interpretation ---")
        print("When true label is 1 but model predicts p ≈ 0,")
        print("the cross-entropy loss → ∞ (severe penalty).")
        print("This is why cross-entropy is effective for classification!")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = LimitExercises()
    
    print("LIMITS AND CONTINUITY EXERCISES")
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
