"""
Integration - Examples
======================
Practical demonstrations of integration concepts.
"""

import numpy as np
from scipy import integrate


def example_definite_integral():
    """Demonstrate definite integral as area."""
    print("=" * 60)
    print("EXAMPLE 1: Definite Integral as Area")
    print("=" * 60)
    
    print("Evaluate: ∫[0 to 2] x² dx")
    print("\nUsing Fundamental Theorem of Calculus:")
    print("Antiderivative of x² is x³/3")
    print("∫[0 to 2] x² dx = [x³/3] from 0 to 2")
    print("              = 2³/3 - 0³/3")
    print("              = 8/3 - 0")
    print("              = 8/3 ≈ 2.667")
    
    # Numerical verification using scipy
    result, error = integrate.quad(lambda x: x**2, 0, 2)
    print(f"\nNumerical verification: {result:.6f}")


def example_riemann_sum():
    """Demonstrate Riemann sum approximation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Riemann Sum Approximation")
    print("=" * 60)
    
    def f(x):
        return x**2
    
    a, b = 0, 2
    
    print(f"Approximate ∫[{a} to {b}] x² dx using Riemann sums")
    print(f"\nExact value: {8/3:.6f}")
    
    for n in [4, 10, 100, 1000]:
        dx = (b - a) / n
        # Right endpoint Riemann sum
        x_points = np.linspace(a + dx, b, n)
        riemann_sum = np.sum(f(x_points)) * dx
        error = abs(riemann_sum - 8/3)
        print(f"  n = {n:4d}: Sum = {riemann_sum:.6f}, Error = {error:.6f}")


def example_antiderivatives():
    """Demonstrate basic antiderivatives."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Basic Antiderivatives")
    print("=" * 60)
    
    print("Power Rule: ∫ x^n dx = x^(n+1)/(n+1) + C")
    
    examples = [
        ("∫ x³ dx", "x⁴/4 + C"),
        ("∫ x^(-2) dx", "-x^(-1) + C = -1/x + C"),
        ("∫ √x dx = ∫ x^(1/2) dx", "x^(3/2)/(3/2) + C = (2/3)x^(3/2) + C"),
        ("∫ 1 dx", "x + C"),
    ]
    
    for integral, result in examples:
        print(f"\n  {integral} = {result}")
    
    print("\n--- Special cases ---")
    print("  ∫ e^x dx = e^x + C")
    print("  ∫ 1/x dx = ln|x| + C")
    print("  ∫ sin(x) dx = -cos(x) + C")
    print("  ∫ cos(x) dx = sin(x) + C")


def example_substitution():
    """Demonstrate u-substitution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: U-Substitution")
    print("=" * 60)
    
    print("Evaluate: ∫ 2x·e^(x²) dx")
    print("\n--- Method ---")
    print("Let u = x²")
    print("Then du = 2x dx")
    print("\nSubstitute:")
    print("∫ 2x·e^(x²) dx = ∫ e^u du = e^u + C = e^(x²) + C")
    
    print("\n--- Verification ---")
    print("d/dx[e^(x²)] = e^(x²)·2x = 2x·e^(x²) ✓")
    
    # Numerical check for definite integral
    print("\n--- Definite integral: ∫[0 to 1] 2x·e^(x²) dx ---")
    print("= [e^(x²)] from 0 to 1")
    print(f"= e¹ - e⁰ = e - 1 = {np.e - 1:.6f}")
    
    result, _ = integrate.quad(lambda x: 2*x*np.exp(x**2), 0, 1)
    print(f"Numerical: {result:.6f}")


def example_integration_by_parts():
    """Demonstrate integration by parts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Integration by Parts")
    print("=" * 60)
    
    print("Formula: ∫ u dv = uv - ∫ v du")
    
    print("\n--- Example: ∫ x·e^x dx ---")
    print("Using LIATE rule: u = x (Algebraic), dv = e^x dx")
    print("Then: du = dx, v = e^x")
    print("\n∫ x·e^x dx = x·e^x - ∫ e^x dx")
    print("          = x·e^x - e^x + C")
    print("          = e^x(x - 1) + C")
    
    # Verify by differentiation
    print("\n--- Verification ---")
    print("d/dx[e^x(x-1)] = e^x(x-1) + e^x·1 = e^x·x - e^x + e^x = x·e^x ✓")
    
    print("\n--- Definite integral: ∫[0 to 1] x·e^x dx ---")
    print("= [e^x(x-1)] from 0 to 1")
    print(f"= e¹(1-1) - e⁰(0-1) = 0 - (-1) = 1")
    
    result, _ = integrate.quad(lambda x: x*np.exp(x), 0, 1)
    print(f"Numerical: {result:.6f}")


def example_gaussian_integral():
    """Demonstrate the Gaussian integral."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Gaussian Integral")
    print("=" * 60)
    
    print("The Gaussian integral:")
    print("∫[-∞ to ∞] e^(-x²) dx = √π")
    
    print(f"\n√π = {np.sqrt(np.pi):.10f}")
    
    # Numerical approximation
    result, error = integrate.quad(lambda x: np.exp(-x**2), -np.inf, np.inf)
    print(f"Numerical: {result:.10f}")
    
    print("\n--- Standard Normal Distribution ---")
    print("∫[-∞ to ∞] (1/√(2π)) e^(-x²/2) dx = 1")
    
    result_normal, _ = integrate.quad(
        lambda x: np.exp(-x**2/2) / np.sqrt(2*np.pi), 
        -np.inf, np.inf
    )
    print(f"Numerical: {result_normal:.10f}")


def example_expected_value():
    """Calculate expected value via integration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Expected Value (Probability)")
    print("=" * 60)
    
    print("For continuous random variable X with PDF p(x):")
    print("E[X] = ∫ x·p(x) dx")
    
    print("\n--- Example: Uniform(0, 1) ---")
    print("p(x) = 1 for x ∈ [0, 1], 0 otherwise")
    print("\nE[X] = ∫[0 to 1] x·1 dx")
    print("     = [x²/2] from 0 to 1")
    print("     = 1/2 - 0 = 0.5")
    
    print("\n--- Example: Exponential(λ=1) ---")
    print("p(x) = e^(-x) for x ≥ 0")
    print("\nE[X] = ∫[0 to ∞] x·e^(-x) dx")
    print("(Integration by parts)")
    print("     = 1")
    
    result, _ = integrate.quad(lambda x: x * np.exp(-x), 0, np.inf)
    print(f"Numerical: {result:.6f}")
    
    print("\n--- E[X²] for Uniform(0, 1) ---")
    print("E[X²] = ∫[0 to 1] x²·1 dx")
    print("      = [x³/3] from 0 to 1")
    print("      = 1/3")
    
    result_x2, _ = integrate.quad(lambda x: x**2, 0, 1)
    print(f"Numerical: {result_x2:.6f}")


def example_variance():
    """Calculate variance via integration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Variance")
    print("=" * 60)
    
    print("Var(X) = E[(X - μ)²] = E[X²] - (E[X])²")
    
    print("\n--- Uniform(0, 1) ---")
    print("E[X] = 1/2")
    print("E[X²] = 1/3")
    print("Var(X) = 1/3 - (1/2)² = 1/3 - 1/4 = 4/12 - 3/12 = 1/12")
    print(f"       = {1/12:.6f}")
    
    print("\n--- Exponential(λ=1) ---")
    print("E[X] = 1")
    print("E[X²] = ∫[0 to ∞] x²·e^(-x) dx = 2 (using gamma function)")
    
    E_X2, _ = integrate.quad(lambda x: x**2 * np.exp(-x), 0, np.inf)
    print(f"E[X²] numerical: {E_X2:.6f}")
    
    var = E_X2 - 1**2
    print(f"Var(X) = E[X²] - (E[X])² = 2 - 1 = 1")


def example_monte_carlo():
    """Demonstrate Monte Carlo integration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Monte Carlo Integration")
    print("=" * 60)
    
    def f(x):
        return np.exp(-x**2)
    
    a, b = 0, 1
    
    print(f"Estimate: ∫[{a} to {b}] e^(-x²) dx")
    
    # Exact value
    exact, _ = integrate.quad(f, a, b)
    print(f"\nExact value: {exact:.10f}")
    
    print("\nMonte Carlo estimates:")
    np.random.seed(42)
    
    for n in [100, 1000, 10000, 100000]:
        x_samples = np.random.uniform(a, b, n)
        mc_estimate = (b - a) * np.mean(f(x_samples))
        error = abs(mc_estimate - exact)
        print(f"  n = {n:6d}: Estimate = {mc_estimate:.6f}, Error = {error:.6f}")
    
    print("\nMonte Carlo: O(1/√n) convergence regardless of dimension!")
    print("This is why it's useful for high-dimensional integrals in ML.")


def example_numerical_methods():
    """Compare numerical integration methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Numerical Integration Methods")
    print("=" * 60)
    
    def f(x):
        return np.sin(x)
    
    a, b = 0, np.pi
    exact = 2.0  # ∫[0 to π] sin(x) dx = 2
    
    print(f"∫[0 to π] sin(x) dx = 2 (exact)")
    
    for n in [4, 8, 16, 32]:
        x = np.linspace(a, b, n + 1)
        dx = (b - a) / n
        
        # Rectangle rule (left endpoint)
        rect = np.sum(f(x[:-1])) * dx
        
        # Trapezoidal rule
        trap = np.trapz(f(x), x)
        
        # Simpson's rule (requires even number of intervals)
        if n % 2 == 0:
            simp = integrate.simpson(f(x), x=x)
        else:
            simp = float('nan')
        
        print(f"\nn = {n}:")
        print(f"  Rectangle:   {rect:.8f}, Error: {abs(rect - exact):.2e}")
        print(f"  Trapezoidal: {trap:.8f}, Error: {abs(trap - exact):.2e}")
        if not np.isnan(simp):
            print(f"  Simpson:     {simp:.8f}, Error: {abs(simp - exact):.2e}")


def example_improper_integral():
    """Demonstrate improper integrals."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Improper Integrals")
    print("=" * 60)
    
    print("Type 1: Infinite limits")
    print("∫[1 to ∞] 1/x² dx")
    print("= lim(t→∞) ∫[1 to t] x^(-2) dx")
    print("= lim(t→∞) [-1/x] from 1 to t")
    print("= lim(t→∞) (-1/t - (-1))")
    print("= 0 + 1 = 1")
    
    result, _ = integrate.quad(lambda x: 1/x**2, 1, np.inf)
    print(f"Numerical: {result:.6f}")
    
    print("\n--- Divergent example ---")
    print("∫[1 to ∞] 1/x dx = lim(t→∞) ln(t) = ∞ (diverges)")
    
    print("\n--- p-test ---")
    print("∫[1 to ∞] 1/x^p dx converges iff p > 1")
    
    for p in [0.5, 1.0, 1.5, 2.0]:
        try:
            result, _ = integrate.quad(lambda x, p=p: 1/x**p, 1, 1000)
            if p > 1:
                expected = 1 / (p - 1)
                print(f"  p = {p}: ≈ {result:.4f} (converges to {expected:.4f})")
            else:
                print(f"  p = {p}: ≈ {result:.4f} (diverges)")
        except:
            print(f"  p = {p}: computation error")


if __name__ == "__main__":
    example_definite_integral()
    example_riemann_sum()
    example_antiderivatives()
    example_substitution()
    example_integration_by_parts()
    example_gaussian_integral()
    example_expected_value()
    example_variance()
    example_monte_carlo()
    example_numerical_methods()
    example_improper_integral()
