"""
Series and Sequences - Exercises
================================
Practice problems for sequences, series, and Taylor expansions.
"""

import numpy as np
from scipy import special


class SeriesExercises:
    """Exercises for sequences and series."""
    
    def exercise_1_sequence_limits(self):
        """
        Exercise 1: Sequence Limits
        
        Determine if the following sequences converge and find their limits:
        a) a_n = (2n + 1)/(3n - 1)
        b) a_n = n²/e^n
        c) a_n = (1 + 2/n)^n
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Sequence Limits")
        
        print("\na) a_n = (2n + 1)/(3n - 1)")
        print("   Divide numerator and denominator by n:")
        print("   = (2 + 1/n)/(3 - 1/n)")
        print("   As n → ∞: → (2 + 0)/(3 - 0) = 2/3")
        print("   CONVERGES to 2/3")
        
        n = 10000
        a_n = (2*n + 1)/(3*n - 1)
        print(f"   Numerical check: a_{n} = {a_n:.6f}")
        
        print("\nb) a_n = n²/e^n")
        print("   Exponential grows faster than polynomial")
        print("   lim n²/e^n = 0 (L'Hôpital twice)")
        print("   CONVERGES to 0")
        
        a_n_b = n**2 / np.exp(n)
        print(f"   a_100 = {100**2 / np.exp(100):.2e}")
        
        print("\nc) a_n = (1 + 2/n)^n")
        print("   Using lim (1 + x/n)^n = e^x with x = 2:")
        print("   = e^2 ≈ 7.389")
        print("   CONVERGES to e²")
        
        a_n_c = (1 + 2/n)**n
        print(f"   a_{n} = {a_n_c:.6f}, e² = {np.e**2:.6f}")
    
    def exercise_2_geometric_series(self):
        """
        Exercise 2: Geometric Series
        
        a) Find the sum: 1 + 1/2 + 1/4 + 1/8 + ...
        b) Find the sum: Σ (2/3)^n for n = 0 to ∞
        c) Express 0.333... as a fraction using geometric series
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Geometric Series")
        
        print("\na) 1 + 1/2 + 1/4 + 1/8 + ...")
        print("   This is Σ (1/2)^n for n = 0 to ∞")
        print("   = 1/(1 - 1/2) = 1/(1/2) = 2")
        
        partial = sum((1/2)**n for n in range(50))
        print(f"   Partial sum (50 terms): {partial:.10f}")
        
        print("\nb) Σ (2/3)^n for n = 0 to ∞")
        print("   = 1/(1 - 2/3) = 1/(1/3) = 3")
        
        partial_b = sum((2/3)**n for n in range(100))
        print(f"   Partial sum (100 terms): {partial_b:.10f}")
        
        print("\nc) 0.333... = 3/10 + 3/100 + 3/1000 + ...")
        print("   = 3(1/10 + 1/100 + 1/1000 + ...)")
        print("   = 3 · Σ (1/10)^n for n = 1 to ∞")
        print("   = 3 · (1/10)/(1 - 1/10)")
        print("   = 3 · (1/10)/(9/10)")
        print("   = 3 · 1/9 = 1/3")
    
    def exercise_3_convergence_tests(self):
        """
        Exercise 3: Convergence Tests
        
        Determine if the following series converge:
        a) Σ n/2^n (ratio test)
        b) Σ 1/n² (p-series)
        c) Σ (-1)^n / √n (alternating series test)
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Convergence Tests")
        
        print("\na) Σ n/2^n - Ratio Test")
        print("   a_n = n/2^n")
        print("   a_(n+1)/a_n = [(n+1)/2^(n+1)] / [n/2^n]")
        print("              = (n+1)/(2n)")
        print("   L = lim (n+1)/(2n) = 1/2 < 1")
        print("   CONVERGES (by ratio test)")
        
        total = sum(n/2**n for n in range(1, 100))
        print(f"   Sum ≈ {total:.6f}")
        
        print("\nb) Σ 1/n² - p-Series")
        print("   This is a p-series with p = 2")
        print("   p > 1, so series CONVERGES")
        print("   (This equals π²/6 ≈ 1.6449)")
        
        total_b = sum(1/n**2 for n in range(1, 10000))
        print(f"   Partial sum: {total_b:.6f}, π²/6 = {np.pi**2/6:.6f}")
        
        print("\nc) Σ (-1)^n / √n - Alternating Series Test")
        print("   b_n = 1/√n")
        print("   1) b_(n+1) = 1/√(n+1) < 1/√n = b_n ✓")
        print("   2) lim b_n = lim 1/√n = 0 ✓")
        print("   CONVERGES (conditionally)")
    
    def exercise_4_taylor_series(self):
        """
        Exercise 4: Taylor Series
        
        a) Find the Maclaurin series for f(x) = cos(x)
        b) Write the first 4 terms of Taylor series for e^x about x = 1
        c) Approximate sin(0.1) using 3 terms of Maclaurin series
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Taylor Series")
        
        print("\na) Maclaurin series for cos(x)")
        print("   f(x) = cos(x)   → f(0) = 1")
        print("   f'(x) = -sin(x) → f'(0) = 0")
        print("   f''(x) = -cos(x) → f''(0) = -1")
        print("   f'''(x) = sin(x) → f'''(0) = 0")
        print("   f''''(x) = cos(x) → f''''(0) = 1")
        print("\n   cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...")
        print("         = Σ (-1)^n x^(2n)/(2n)!")
        
        print("\nb) Taylor series for e^x about x = 1")
        print("   f(x) = e^x, f^(n)(1) = e for all n")
        print("   e^x = e + e(x-1) + e(x-1)²/2! + e(x-1)³/3! + ...")
        print("       = e[1 + (x-1) + (x-1)²/2 + (x-1)³/6 + ...]")
        
        print("\nc) Approximate sin(0.1)")
        print("   sin(x) ≈ x - x³/3! + x⁵/5!")
        x = 0.1
        approx = x - x**3/6 + x**5/120
        exact = np.sin(x)
        print(f"   sin(0.1) ≈ 0.1 - 0.001/6 + 0.00001/120")
        print(f"            = {approx:.10f}")
        print(f"   Exact:    = {exact:.10f}")
        print(f"   Error:    = {abs(approx - exact):.2e}")
    
    def exercise_5_radius_convergence(self):
        """
        Exercise 5: Radius of Convergence
        
        Find the radius of convergence for:
        a) Σ x^n/n
        b) Σ n! x^n
        c) Σ x^n/n!
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Radius of Convergence")
        
        print("\na) Σ x^n/n")
        print("   c_n = 1/n")
        print("   R = lim |c_n/c_(n+1)| = lim |n/(n+1)·n| = lim n/(n+1)")
        print("   Wait, using ratio test on series:")
        print("   |a_(n+1)/a_n| = |x^(n+1)/(n+1)| / |x^n/n| = |x| · n/(n+1)")
        print("   lim = |x|")
        print("   Converges when |x| < 1")
        print("   R = 1")
        
        print("\nb) Σ n! x^n")
        print("   |a_(n+1)/a_n| = |(n+1)! x^(n+1)| / |n! x^n|")
        print("                = (n+1)|x|")
        print("   lim = ∞ for any x ≠ 0")
        print("   R = 0 (converges only at x = 0)")
        
        print("\nc) Σ x^n/n!")
        print("   |a_(n+1)/a_n| = |x^(n+1)/(n+1)!| / |x^n/n!|")
        print("                = |x|/(n+1)")
        print("   lim = 0 for any x")
        print("   R = ∞ (converges for all x)")
        print("   This is the series for e^x!")
    
    def exercise_6_sigmoid_taylor(self):
        """
        Exercise 6: Sigmoid Taylor Series (ML)
        
        Derive the first 3 non-zero terms of the Taylor series 
        for σ(x) = 1/(1 + e^(-x)) about x = 0.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Sigmoid Taylor Series")
        
        print("σ(x) = 1/(1 + e^(-x))")
        
        print("\n--- Computing derivatives at x = 0 ---")
        print("σ(0) = 1/(1 + 1) = 1/2")
        
        print("\nσ'(x) = σ(x)(1 - σ(x)) = σ(x) - σ(x)²")
        print("σ'(0) = 1/2 - (1/2)² = 1/2 - 1/4 = 1/4")
        
        print("\nσ''(x) = σ'(x) - 2σ(x)σ'(x) = σ'(x)(1 - 2σ(x))")
        print("σ''(0) = (1/4)(1 - 2·1/2) = (1/4)(0) = 0")
        
        print("\nσ'''(x) = σ''(x)(1-2σ) + σ'(x)(-2σ')")
        print("        = σ''(x)(1-2σ) - 2(σ')²")
        print("σ'''(0) = 0·(0) - 2·(1/4)² = -2/16 = -1/8")
        
        print("\n--- Taylor series ---")
        print("σ(x) = σ(0) + σ'(0)x + σ''(0)x²/2! + σ'''(0)x³/3! + ...")
        print("     = 1/2 + (1/4)x + 0 + (-1/8)x³/6 + ...")
        print("     = 1/2 + x/4 - x³/48 + O(x⁵)")
        
        print("\n--- Verification ---")
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def taylor_approx(x):
            return 0.5 + x/4 - x**3/48
        
        for x in [-0.5, 0, 0.5]:
            s = sigmoid(x)
            t = taylor_approx(x)
            print(f"x = {x:5.2f}: σ(x) = {s:.6f}, Taylor = {t:.6f}, error = {abs(s-t):.6f}")
    
    def exercise_7_exponential_decay(self):
        """
        Exercise 7: Exponential Learning Rate Decay
        
        For α(t) = α₀ · e^(-λt):
        a) Find the first 3 terms of Taylor expansion for small λt
        b) For α₀ = 0.01, λ = 0.1, approximate α(1) using Taylor series
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Exponential Learning Rate Decay")
        
        print("α(t) = α₀ · e^(-λt)")
        
        print("\na) Taylor expansion of e^(-λt) for small λt:")
        print("   e^x = 1 + x + x²/2! + x³/3! + ...")
        print("   e^(-λt) = 1 + (-λt) + (-λt)²/2 + (-λt)³/6 + ...")
        print("          = 1 - λt + (λt)²/2 - (λt)³/6 + ...")
        print("\n   α(t) ≈ α₀[1 - λt + (λt)²/2]")
        
        print("\nb) For α₀ = 0.01, λ = 0.1, t = 1:")
        alpha0, lam, t = 0.01, 0.1, 1
        lt = lam * t
        
        exact = alpha0 * np.exp(-lt)
        taylor = alpha0 * (1 - lt + lt**2/2)
        
        print(f"   λt = {lt}")
        print(f"   Exact: α(1) = {alpha0}·e^(-0.1) = {exact:.8f}")
        print(f"   Taylor: α(1) ≈ {alpha0}·(1 - 0.1 + 0.005) = {taylor:.8f}")
        print(f"   Error: {abs(exact - taylor):.2e}")
    
    def exercise_8_power_series(self):
        """
        Exercise 8: Power Series Operations
        
        Using known series, find the series for:
        a) f(x) = x · e^x
        b) f(x) = sin(x)/x  (for x ≠ 0)
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Power Series Operations")
        
        print("\na) f(x) = x · e^x")
        print("   e^x = 1 + x + x²/2! + x³/3! + ...")
        print("   x·e^x = x(1 + x + x²/2! + x³/3! + ...)")
        print("        = x + x² + x³/2! + x⁴/3! + ...")
        print("        = Σ x^(n+1)/n! = Σ x^n/(n-1)! for n ≥ 1")
        
        print("\n--- Verification ---")
        x = 0.5
        exact = x * np.exp(x)
        series = sum(x**(n+1)/np.math.factorial(n) for n in range(10))
        print(f"   x = 0.5: exact = {exact:.6f}, series = {series:.6f}")
        
        print("\nb) f(x) = sin(x)/x")
        print("   sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...")
        print("   sin(x)/x = 1 - x²/3! + x⁴/5! - x⁶/7! + ...")
        print("            = Σ (-1)^n x^(2n)/(2n+1)!")
        print("\n   Note: lim(x→0) sin(x)/x = 1 (from series)")
        
        print("\n--- Verification ---")
        x = 0.5
        exact_sinc = np.sin(x)/x if x != 0 else 1
        series_sinc = sum((-1)**n * x**(2*n)/np.math.factorial(2*n+1) for n in range(10))
        print(f"   x = 0.5: exact = {exact_sinc:.6f}, series = {series_sinc:.6f}")
    
    def exercise_9_numerical_series(self):
        """
        Exercise 9: Computing π Using Series
        
        The Leibniz formula: π/4 = 1 - 1/3 + 1/5 - 1/7 + ...
        
        a) How many terms are needed for 2 decimal places of π?
        b) Implement and verify
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Computing π")
        
        print("Leibniz formula: π/4 = 1 - 1/3 + 1/5 - 1/7 + ...")
        print("                     = Σ (-1)^n / (2n+1)")
        
        print("\na) Finding number of terms for 2 decimal places:")
        print("   Need error < 0.005 for π accurate to 0.01")
        print("   For alternating series, error < |next term|")
        print("   Need π/4 accurate to 0.00125")
        
        target_error = 0.005 / 4
        n = 0
        while 1/(2*n+1) > target_error:
            n += 1
        print(f"   Need n ≥ {n} terms (1/(2n+1) < {target_error:.5f})")
        
        print("\nb) Verification:")
        for N in [10, 100, 1000, 10000]:
            partial = sum((-1)**n / (2*n + 1) for n in range(N))
            pi_approx = 4 * partial
            error = abs(pi_approx - np.pi)
            print(f"   N = {N:5d}: π ≈ {pi_approx:.6f}, error = {error:.6f}")
        
        print("\n   Note: This series converges VERY slowly!")
        print("   Better methods exist (Machin's formula, etc.)")
    
    def exercise_10_loss_taylor(self):
        """
        Exercise 10: Loss Function Taylor Expansion (ML)
        
        For a loss function L(θ), write the second-order Taylor expansion
        about θ* (the optimum), and explain its use in optimization.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Loss Function Taylor Expansion")
        
        print("Taylor expansion of L(θ) about θ*:")
        print("\nL(θ) = L(θ*) + ∇L(θ*)ᵀ(θ - θ*) + ½(θ - θ*)ᵀH(θ*)(θ - θ*) + O(||θ - θ*||³)")
        
        print("\nAt the optimum θ*:")
        print("  - ∇L(θ*) = 0 (first-order condition)")
        print("  - H(θ*) ≻ 0 (positive definite for minimum)")
        
        print("\nSimplified at optimum:")
        print("  L(θ) ≈ L(θ*) + ½(θ - θ*)ᵀH(θ*)(θ - θ*)")
        print("\nThis is a quadratic approximation!")
        
        print("\n--- Applications ---")
        print("1. Newton's Method:")
        print("   Setting ∇L ≈ ∇L(θₖ) + H(θₖ)(θ - θₖ) = 0")
        print("   θₖ₊₁ = θₖ - H⁻¹∇L")
        
        print("\n2. Learning Rate Selection:")
        print("   For gradient descent with step η:")
        print("   Optimal η relates to eigenvalues of H")
        
        print("\n3. Convergence Analysis:")
        print("   Near optimum, loss decreases like:")
        print("   L(θₖ) - L(θ*) ~ (1 - 2ηλ_min + η²λ_max²)^k")
        
        print("\n--- Example: Quadratic Loss ---")
        print("L(θ) = ½θᵀAθ - bᵀθ")
        print("∇L = Aθ - b")
        print("H = A")
        print("θ* = A⁻¹b")
        print("L(θ*) = -½bᵀA⁻¹b")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = SeriesExercises()
    
    print("SERIES AND SEQUENCES EXERCISES")
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
