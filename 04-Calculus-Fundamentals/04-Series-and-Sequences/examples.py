"""
Series and Sequences - Examples
===============================
Practical demonstrations of sequences, series, and Taylor expansions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


def example_sequence_convergence():
    """Demonstrate sequence convergence."""
    print("=" * 60)
    print("EXAMPLE 1: Sequence Convergence")
    print("=" * 60)
    
    print("Several sequences and their limits:\n")
    
    n_values = np.arange(1, 21)
    
    # a_n = 1/n → 0
    seq1 = 1 / n_values
    print(f"a_n = 1/n: {seq1[-5:].round(4)} → 0")
    
    # a_n = (1 + 1/n)^n → e
    seq2 = (1 + 1/n_values) ** n_values
    print(f"a_n = (1+1/n)^n: {seq2[-5:].round(4)} → e = {np.e:.4f}")
    
    # a_n = n/(n+1) → 1
    seq3 = n_values / (n_values + 1)
    print(f"a_n = n/(n+1): {seq3[-5:].round(4)} → 1")
    
    # a_n = (-1)^n / n → 0 (oscillating but converging)
    seq4 = ((-1) ** n_values) / n_values
    print(f"a_n = (-1)^n/n: {seq4[-5:].round(4)} → 0")
    
    print("\n--- Key sequence limits ---")
    print("lim n→∞ (1 + x/n)^n = e^x")
    for x in [1, 2, -1]:
        large_n = 10000
        approx = (1 + x/large_n) ** large_n
        print(f"  x = {x:2d}: (1 + {x}/10000)^10000 = {approx:.6f}, e^{x} = {np.exp(x):.6f}")


def example_geometric_series():
    """Demonstrate geometric series."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Geometric Series")
    print("=" * 60)
    
    print("Geometric series: Σ r^n = 1/(1-r) for |r| < 1")
    
    for r in [0.5, 0.9, 0.99, -0.5]:
        # Partial sums
        N = 100
        terms = np.array([r**n for n in range(N)])
        partial_sums = np.cumsum(terms)
        
        if abs(r) < 1:
            exact = 1 / (1 - r)
            print(f"\nr = {r}")
            print(f"  Exact sum: 1/(1-{r}) = {exact:.6f}")
            print(f"  Partial sum (N=10): {partial_sums[9]:.6f}")
            print(f"  Partial sum (N=100): {partial_sums[99]:.6f}")
        else:
            print(f"\nr = {r}: Series diverges")


def example_ratio_test():
    """Demonstrate the ratio test."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Ratio Test for Convergence")
    print("=" * 60)
    
    print("Ratio Test: If L = lim |a_(n+1)/a_n| < 1, series converges")
    
    print("\n--- Example 1: Σ n/2^n ---")
    print("a_n = n/2^n")
    print("a_(n+1)/a_n = [(n+1)/2^(n+1)] / [n/2^n]")
    print("           = (n+1)/(2n)")
    print("L = lim (n+1)/(2n) = 1/2 < 1")
    print("Series CONVERGES")
    
    # Numerical verification
    N = 50
    terms = np.array([n / 2**n for n in range(1, N+1)])
    total = np.sum(terms)
    print(f"Sum ≈ {total:.6f}")
    
    print("\n--- Example 2: Σ n!/n^n ---")
    print("Using Stirling's approximation: n! ≈ √(2πn)(n/e)^n")
    print("a_n ≈ √(2πn) / e^n → 0 very fast")
    print("Series CONVERGES")
    
    terms2 = [special.factorial(n) / n**n for n in range(1, 20)]
    print(f"First 5 terms: {[f'{t:.6f}' for t in terms2[:5]]}")
    print(f"Sum ≈ {sum(terms2):.6f}")


def example_taylor_exponential():
    """Demonstrate Taylor series for e^x."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Taylor Series for e^x")
    print("=" * 60)
    
    print("e^x = Σ x^n/n! = 1 + x + x²/2! + x³/3! + ...")
    
    x = 1.0
    print(f"\nApproximating e^{x} = {np.exp(x):.10f}")
    
    approx = 0
    print("\nPartial sums:")
    for n in range(11):
        term = x**n / np.math.factorial(n)
        approx += term
        error = abs(approx - np.exp(x))
        print(f"  n = {n:2d}: sum = {approx:.10f}, error = {error:.2e}")
    
    print("\n--- Different x values ---")
    for x in [0.1, 0.5, 2.0, -1.0]:
        approx = sum(x**n / np.math.factorial(n) for n in range(20))
        print(f"  x = {x:4.1f}: approx = {approx:.8f}, exact = {np.exp(x):.8f}")


def example_taylor_trig():
    """Demonstrate Taylor series for sin and cos."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Taylor Series for Trigonometric Functions")
    print("=" * 60)
    
    print("sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...")
    print("cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...")
    
    x = np.pi / 4  # 45 degrees
    
    print(f"\nApproximating sin(π/4) = {np.sin(x):.10f}")
    
    sin_approx = 0
    for n in range(6):
        term = ((-1)**n * x**(2*n+1)) / np.math.factorial(2*n+1)
        sin_approx += term
        error = abs(sin_approx - np.sin(x))
        print(f"  n = {n}: sum = {sin_approx:.10f}, error = {error:.2e}")
    
    print(f"\nApproximating cos(π/4) = {np.cos(x):.10f}")
    
    cos_approx = 0
    for n in range(6):
        term = ((-1)**n * x**(2*n)) / np.math.factorial(2*n)
        cos_approx += term
        error = abs(cos_approx - np.cos(x))
        print(f"  n = {n}: sum = {cos_approx:.10f}, error = {error:.2e}")


def example_ln_series():
    """Demonstrate Taylor series for ln(1+x)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Taylor Series for ln(1+x)")
    print("=" * 60)
    
    print("ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...")
    print("Valid for -1 < x ≤ 1")
    
    for x in [0.1, 0.5, 1.0]:
        print(f"\nln(1 + {x}) = {np.log(1+x):.10f}")
        
        approx = 0
        for n in range(1, 16):
            term = ((-1)**(n+1) * x**n) / n
            approx += term
        
        error = abs(approx - np.log(1+x))
        print(f"  15-term approx: {approx:.10f}, error: {error:.2e}")
    
    print("\nNote: Convergence is slower as x approaches 1")


def example_sigmoid_taylor():
    """Taylor series approximation for sigmoid."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Taylor Series for Sigmoid (ML)")
    print("=" * 60)
    
    print("σ(x) = 1/(1 + e^(-x))")
    print("\nTaylor expansion around x = 0:")
    print("σ(0) = 1/2")
    print("σ'(x) = σ(x)(1 - σ(x)) → σ'(0) = 1/4")
    print("σ''(x) = σ(x)(1 - σ(x))(1 - 2σ(x)) → σ''(0) = 0")
    print("σ'''(0) = -1/8")
    
    print("\nσ(x) ≈ 1/2 + x/4 - x³/48 + O(x⁵)")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_taylor(x):
        return 0.5 + x/4 - x**3/48
    
    print("\nComparison:")
    for x in [-1, -0.5, 0, 0.5, 1]:
        exact = sigmoid(x)
        approx = sigmoid_taylor(x)
        error = abs(exact - approx)
        print(f"  x = {x:5.2f}: exact = {exact:.6f}, Taylor = {approx:.6f}, error = {error:.6f}")
    
    print("\nThis approximation is useful for:")
    print("  - Analyzing behavior near origin")
    print("  - Understanding why sigmoid ≈ linear for small inputs")


def example_softplus_series():
    """Taylor series for softplus."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Softplus Approximations (ML)")
    print("=" * 60)
    
    print("softplus(x) = ln(1 + e^x)")
    
    def softplus(x):
        return np.log(1 + np.exp(np.clip(x, -100, 100)))
    
    print("\nFor large x: softplus(x) ≈ x")
    print("For small x: Using ln(1+y) ≈ y for small y")
    print("            softplus(x) ≈ e^x for large negative x")
    print("For x near 0: softplus(x) ≈ ln(2) + x/2 + x²/8 - x⁴/192 + ...")
    
    print("\nComparison:")
    print("x\t\tExact\t\tLinear (x)\tln(2)+x/2")
    for x in [-5, -2, 0, 2, 5]:
        exact = softplus(x)
        linear = max(0, x)  # ReLU
        quadratic = np.log(2) + x/2 + x**2/8
        print(f"{x:2d}\t\t{exact:.4f}\t\t{linear:.4f}\t\t{quadratic:.4f}")


def example_convergence_rates():
    """Demonstrate convergence rate analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Convergence Rate Analysis (ML)")
    print("=" * 60)
    
    print("Comparing convergence rates of different sequences:")
    
    n_vals = np.arange(1, 21)
    
    # Sublinear O(1/n)
    sublinear = 1 / n_vals
    
    # Linear O(ρ^n) where ρ = 0.9
    rho = 0.9
    linear = rho ** n_vals
    
    # Superlinear O(ρ^(n²))
    superlinear = rho ** (n_vals ** 1.5)
    
    print("\nError after n iterations:")
    print("n\tO(1/n)\t\tO(0.9^n)\tO(0.9^n^1.5)")
    for n in [1, 5, 10, 15, 20]:
        print(f"{n}\t{sublinear[n-1]:.6f}\t{linear[n-1]:.6f}\t{superlinear[n-1]:.6f}")
    
    print("\n--- Gradient Descent Convergence ---")
    print("For L-smooth convex function:")
    print("  f(x_k) - f(x*) ≤ L||x_0 - x*||² / (2k)")
    print("\nThis is O(1/k) - sublinear convergence")


def example_euler_number():
    """Different ways to compute e."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Computing Euler's Number e")
    print("=" * 60)
    
    print(f"Exact: e = {np.e:.15f}")
    
    print("\n--- Method 1: Limit definition ---")
    print("e = lim (1 + 1/n)^n")
    for n in [10, 100, 1000, 10000, 100000]:
        approx = (1 + 1/n) ** n
        error = abs(approx - np.e)
        print(f"  n = {n:6d}: e ≈ {approx:.10f}, error = {error:.2e}")
    
    print("\n--- Method 2: Series ---")
    print("e = Σ 1/n!")
    approx = 0
    for n in range(15):
        approx += 1 / np.math.factorial(n)
    error = abs(approx - np.e)
    print(f"  15 terms: e ≈ {approx:.15f}, error = {error:.2e}")
    
    print("\n--- Method 3: Continued fraction ---")
    print("e = 2 + 1/(1 + 1/(2 + 1/(1 + 1/(1 + ...))))")


def example_radius_convergence():
    """Demonstrate radius of convergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Radius of Convergence")
    print("=" * 60)
    
    print("For power series Σ c_n x^n, radius R = lim |c_n/c_(n+1)|")
    
    print("\n--- Example 1: e^x = Σ x^n/n! ---")
    print("c_n = 1/n!")
    print("R = lim |c_n/c_(n+1)| = lim |(n+1)!/n!| = lim (n+1) = ∞")
    print("Converges for ALL x")
    
    print("\n--- Example 2: ln(1+x) = Σ (-1)^(n+1) x^n/n ---")
    print("c_n = (-1)^(n+1)/n")
    print("R = lim |c_n/c_(n+1)| = lim |n/(n+1)| = 1")
    print("Converges for |x| < 1, check endpoints:")
    print("  x = 1: Σ (-1)^(n+1)/n = ln(2) ✓ (alternating series)")
    print("  x = -1: Σ -1/n = -∞ ✗ (harmonic series)")
    
    print("\n--- Example 3: Σ n! x^n ---")
    print("c_n = n!")
    print("R = lim |n!/(n+1)!| = lim 1/(n+1) = 0")
    print("Converges ONLY at x = 0")


def example_numerical_stability():
    """Series for numerical stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Series for Numerical Stability")
    print("=" * 60)
    
    print("Computing log(1 + x) for small x:")
    
    x = 1e-10
    
    # Direct computation (problematic)
    direct = np.log(1 + x)
    print(f"\nDirect: log(1 + {x}) = {direct}")
    
    # Using series
    series_approx = x - x**2/2 + x**3/3
    print(f"Series: x - x²/2 + x³/3 = {series_approx}")
    
    # Using numpy's stable function
    stable = np.log1p(x)
    print(f"log1p: {stable}")
    
    print("\nFor small x, direct computation loses precision!")
    print("numpy.log1p uses series internally for stability.")
    
    print("\n--- expm1(x) = e^x - 1 ---")
    x = 1e-10
    direct_exp = np.exp(x) - 1
    stable_exp = np.expm1(x)
    series_exp = x + x**2/2 + x**3/6
    
    print(f"Direct (e^x - 1): {direct_exp}")
    print(f"Stable (expm1):   {stable_exp}")
    print(f"Series:           {series_exp}")


if __name__ == "__main__":
    example_sequence_convergence()
    example_geometric_series()
    example_ratio_test()
    example_taylor_exponential()
    example_taylor_trig()
    example_ln_series()
    example_sigmoid_taylor()
    example_softplus_series()
    example_convergence_rates()
    example_euler_number()
    example_radius_convergence()
    example_numerical_stability()
