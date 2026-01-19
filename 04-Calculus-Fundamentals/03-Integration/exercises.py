"""
Integration - Exercises
=======================
Practice problems for integration concepts.
"""

import numpy as np
from scipy import integrate


class IntegrationExercises:
    """Exercises for integration."""
    
    def exercise_1_basic_integrals(self):
        """
        Exercise 1: Basic Integrals
        
        Evaluate:
        a) ∫ (3x² + 2x - 1) dx
        b) ∫ (e^x + 1/x) dx
        c) ∫ (sin(x) + cos(x)) dx
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic Integrals")
        
        print("\na) ∫ (3x² + 2x - 1) dx")
        print("   = 3·(x³/3) + 2·(x²/2) - x + C")
        print("   = x³ + x² - x + C")
        
        print("\nb) ∫ (e^x + 1/x) dx")
        print("   = e^x + ln|x| + C")
        
        print("\nc) ∫ (sin(x) + cos(x)) dx")
        print("   = -cos(x) + sin(x) + C")
        
        print("\n--- Verification by differentiation ---")
        print("d/dx[x³ + x² - x] = 3x² + 2x - 1 ✓")
        print("d/dx[e^x + ln|x|] = e^x + 1/x ✓")
        print("d/dx[-cos(x) + sin(x)] = sin(x) + cos(x) ✓")
    
    def exercise_2_definite_integrals(self):
        """
        Exercise 2: Definite Integrals
        
        Evaluate:
        a) ∫[0 to 1] x³ dx
        b) ∫[0 to π] sin(x) dx
        c) ∫[1 to e] 1/x dx
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Definite Integrals")
        
        print("\na) ∫[0 to 1] x³ dx")
        print("   = [x⁴/4] from 0 to 1")
        print("   = 1/4 - 0 = 1/4 = 0.25")
        
        result_a, _ = integrate.quad(lambda x: x**3, 0, 1)
        print(f"   Numerical: {result_a}")
        
        print("\nb) ∫[0 to π] sin(x) dx")
        print("   = [-cos(x)] from 0 to π")
        print("   = -cos(π) - (-cos(0))")
        print("   = -(-1) - (-1) = 1 + 1 = 2")
        
        result_b, _ = integrate.quad(np.sin, 0, np.pi)
        print(f"   Numerical: {result_b}")
        
        print("\nc) ∫[1 to e] 1/x dx")
        print("   = [ln|x|] from 1 to e")
        print("   = ln(e) - ln(1)")
        print("   = 1 - 0 = 1")
        
        result_c, _ = integrate.quad(lambda x: 1/x, 1, np.e)
        print(f"   Numerical: {result_c}")
    
    def exercise_3_substitution(self):
        """
        Exercise 3: U-Substitution
        
        Use substitution to evaluate:
        a) ∫ 2x/(x² + 1) dx
        b) ∫ cos(x)·e^(sin(x)) dx
        c) ∫[0 to 1] x·(1 - x²)³ dx
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: U-Substitution")
        
        print("\na) ∫ 2x/(x² + 1) dx")
        print("   Let u = x² + 1, du = 2x dx")
        print("   = ∫ 1/u du")
        print("   = ln|u| + C")
        print("   = ln(x² + 1) + C")
        
        print("\nb) ∫ cos(x)·e^(sin(x)) dx")
        print("   Let u = sin(x), du = cos(x) dx")
        print("   = ∫ e^u du")
        print("   = e^u + C")
        print("   = e^(sin(x)) + C")
        
        print("\nc) ∫[0 to 1] x·(1 - x²)³ dx")
        print("   Let u = 1 - x², du = -2x dx")
        print("   x dx = -du/2")
        print("   When x=0: u=1, When x=1: u=0")
        print("   = ∫[1 to 0] u³·(-1/2) du")
        print("   = (1/2)∫[0 to 1] u³ du")
        print("   = (1/2)·[u⁴/4] from 0 to 1")
        print("   = (1/2)·(1/4) = 1/8")
        
        result_c, _ = integrate.quad(lambda x: x*(1-x**2)**3, 0, 1)
        print(f"   Numerical: {result_c}")
    
    def exercise_4_by_parts(self):
        """
        Exercise 4: Integration by Parts
        
        Evaluate using integration by parts:
        a) ∫ x·cos(x) dx
        b) ∫ x²·e^x dx
        c) ∫ ln(x) dx
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Integration by Parts")
        
        print("\na) ∫ x·cos(x) dx")
        print("   u = x, dv = cos(x) dx")
        print("   du = dx, v = sin(x)")
        print("   = x·sin(x) - ∫ sin(x) dx")
        print("   = x·sin(x) + cos(x) + C")
        
        print("\nb) ∫ x²·e^x dx")
        print("   Apply by parts twice:")
        print("   First: u = x², dv = e^x dx")
        print("   = x²·e^x - ∫ 2x·e^x dx")
        print("   Second: u = 2x, dv = e^x dx")
        print("   = x²·e^x - (2x·e^x - ∫ 2e^x dx)")
        print("   = x²·e^x - 2x·e^x + 2e^x + C")
        print("   = e^x(x² - 2x + 2) + C")
        
        print("\nc) ∫ ln(x) dx")
        print("   u = ln(x), dv = dx")
        print("   du = 1/x dx, v = x")
        print("   = x·ln(x) - ∫ x·(1/x) dx")
        print("   = x·ln(x) - ∫ 1 dx")
        print("   = x·ln(x) - x + C")
        print("   = x(ln(x) - 1) + C")
    
    def exercise_5_expected_value(self):
        """
        Exercise 5: Expected Value
        
        For X ~ Uniform(0, 1), calculate:
        a) E[X]
        b) E[X²]
        c) Var(X)
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Expected Value")
        
        print("X ~ Uniform(0, 1): p(x) = 1 for x ∈ [0, 1]")
        
        print("\na) E[X] = ∫[0 to 1] x·1 dx")
        print("        = [x²/2] from 0 to 1")
        print("        = 1/2")
        
        E_X, _ = integrate.quad(lambda x: x, 0, 1)
        print(f"   Numerical: {E_X}")
        
        print("\nb) E[X²] = ∫[0 to 1] x²·1 dx")
        print("         = [x³/3] from 0 to 1")
        print("         = 1/3")
        
        E_X2, _ = integrate.quad(lambda x: x**2, 0, 1)
        print(f"   Numerical: {E_X2}")
        
        print("\nc) Var(X) = E[X²] - (E[X])²")
        print("         = 1/3 - (1/2)²")
        print("         = 1/3 - 1/4")
        print("         = 4/12 - 3/12")
        print("         = 1/12 ≈ 0.0833")
        
        print(f"   Computed: {1/3 - (1/2)**2:.6f}")
    
    def exercise_6_gaussian(self):
        """
        Exercise 6: Gaussian Distribution
        
        For X ~ N(0, 1) with PDF p(x) = (1/√(2π))e^(-x²/2):
        
        a) Verify ∫ p(x) dx = 1
        b) Calculate E[X]
        c) Calculate E[X²] = Var(X)
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Gaussian Distribution")
        
        def gaussian_pdf(x):
            return np.exp(-x**2/2) / np.sqrt(2*np.pi)
        
        print("Standard Normal: p(x) = (1/√(2π))·e^(-x²/2)")
        
        print("\na) Verify normalization:")
        integral, _ = integrate.quad(gaussian_pdf, -np.inf, np.inf)
        print(f"   ∫ p(x) dx = {integral:.10f} ≈ 1 ✓")
        
        print("\nb) E[X] = ∫ x·p(x) dx")
        E_X, _ = integrate.quad(lambda x: x * gaussian_pdf(x), -np.inf, np.inf)
        print(f"   = {E_X:.10f} ≈ 0")
        print("   (By symmetry, the mean is 0)")
        
        print("\nc) E[X²] = ∫ x²·p(x) dx")
        E_X2, _ = integrate.quad(lambda x: x**2 * gaussian_pdf(x), -np.inf, np.inf)
        print(f"   = {E_X2:.10f} ≈ 1")
        print("   For N(0, 1): Var(X) = 1 ✓")
    
    def exercise_7_monte_carlo(self):
        """
        Exercise 7: Monte Carlo Integration
        
        Use Monte Carlo to estimate:
        ∫[0 to 1] e^(-x²) dx
        
        Compare with exact value.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Monte Carlo Integration")
        
        def f(x):
            return np.exp(-x**2)
        
        exact, _ = integrate.quad(f, 0, 1)
        print(f"Exact value: {exact:.10f}")
        
        print("\nMonte Carlo estimation:")
        np.random.seed(42)
        
        for n in [100, 1000, 10000, 100000]:
            samples = np.random.uniform(0, 1, n)
            estimate = np.mean(f(samples))  # (b-a) = 1
            error = abs(estimate - exact)
            print(f"  n = {n:6d}: estimate = {estimate:.6f}, error = {error:.6f}")
        
        print("\nNote: Error decreases as O(1/√n)")
    
    def exercise_8_probability(self):
        """
        Exercise 8: Probability via Integration
        
        For X ~ Exponential(λ=1) with PDF p(x) = e^(-x) for x ≥ 0:
        
        Calculate P(1 ≤ X ≤ 2)
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Probability via Integration")
        
        print("X ~ Exp(λ=1): p(x) = e^(-x) for x ≥ 0")
        print("\nP(1 ≤ X ≤ 2) = ∫[1 to 2] e^(-x) dx")
        print("             = [-e^(-x)] from 1 to 2")
        print("             = -e^(-2) - (-e^(-1))")
        print("             = e^(-1) - e^(-2)")
        print(f"             = {np.exp(-1):.6f} - {np.exp(-2):.6f}")
        print(f"             = {np.exp(-1) - np.exp(-2):.6f}")
        
        result, _ = integrate.quad(lambda x: np.exp(-x), 1, 2)
        print(f"\nNumerical: {result:.6f}")
    
    def exercise_9_improper(self):
        """
        Exercise 9: Improper Integrals
        
        Evaluate:
        a) ∫[1 to ∞] 1/x² dx
        b) ∫[0 to ∞] e^(-x) dx
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Improper Integrals")
        
        print("\na) ∫[1 to ∞] 1/x² dx")
        print("   = lim(t→∞) ∫[1 to t] x^(-2) dx")
        print("   = lim(t→∞) [-x^(-1)] from 1 to t")
        print("   = lim(t→∞) (-1/t - (-1))")
        print("   = lim(t→∞) (1 - 1/t)")
        print("   = 1")
        
        result_a, _ = integrate.quad(lambda x: 1/x**2, 1, np.inf)
        print(f"   Numerical: {result_a:.6f}")
        
        print("\nb) ∫[0 to ∞] e^(-x) dx")
        print("   = lim(t→∞) ∫[0 to t] e^(-x) dx")
        print("   = lim(t→∞) [-e^(-x)] from 0 to t")
        print("   = lim(t→∞) (-e^(-t) - (-1))")
        print("   = lim(t→∞) (1 - e^(-t))")
        print("   = 1 - 0 = 1")
        
        result_b, _ = integrate.quad(lambda x: np.exp(-x), 0, np.inf)
        print(f"   Numerical: {result_b:.6f}")
    
    def exercise_10_kl_divergence(self):
        """
        Exercise 10: KL Divergence (ML)
        
        For two normal distributions:
        p(x) = N(0, 1)
        q(x) = N(μ, 1)
        
        Show that KL(p||q) = μ²/2
        
        Verify numerically for μ = 1.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: KL Divergence")
        
        print("KL(p||q) = ∫ p(x) log(p(x)/q(x)) dx")
        print("\nFor p = N(0,1) and q = N(μ,1):")
        print("log(p(x)/q(x)) = log(p(x)) - log(q(x))")
        print("               = -x²/2 - (-（x-μ)²/2)")
        print("               = -(x²)/2 + (x-μ)²/2")
        print("               = (x² - 2xμ + μ² - x²)/2")
        print("               = (-2xμ + μ²)/2")
        print("               = μ²/2 - xμ")
        
        print("\nKL(p||q) = ∫ p(x) (μ²/2 - xμ) dx")
        print("        = μ²/2 · ∫p(x)dx - μ·∫x·p(x)dx")
        print("        = μ²/2 · 1 - μ · E[X]")
        print("        = μ²/2 - μ · 0")
        print("        = μ²/2")
        
        print("\n--- Numerical verification for μ = 1 ---")
        mu = 1
        
        def p(x):
            return np.exp(-x**2/2) / np.sqrt(2*np.pi)
        
        def q(x, mu=mu):
            return np.exp(-(x-mu)**2/2) / np.sqrt(2*np.pi)
        
        def kl_integrand(x):
            px = p(x)
            qx = q(x)
            if px < 1e-10 or qx < 1e-10:
                return 0
            return px * np.log(px / qx)
        
        kl_numerical, _ = integrate.quad(kl_integrand, -10, 10)
        kl_analytical = mu**2 / 2
        
        print(f"Analytical: KL = μ²/2 = {mu}²/2 = {kl_analytical}")
        print(f"Numerical:  KL = {kl_numerical:.6f}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = IntegrationExercises()
    
    print("INTEGRATION EXERCISES")
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
