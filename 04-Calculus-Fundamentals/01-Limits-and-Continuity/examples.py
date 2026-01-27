"""
Limits and Continuity - Examples
================================
Practical demonstrations of limits and continuity concepts.
"""

import numpy as np
import matplotlib.pyplot as plt


def example_limit_intuition():
    """Demonstrate limits intuitively with numerical approach."""
    print("=" * 60)
    print("EXAMPLE 1: Limit Intuition")
    print("=" * 60)
    
    print("Evaluate: lim(x→2) (x² - 4)/(x - 2)")
    print("\nApproaching x = 2 from both sides:")
    
    def f(x):
        return (x**2 - 4) / (x - 2)
    
    # Approach from left
    print("\nFrom left (x < 2):")
    for x in [1.9, 1.99, 1.999, 1.9999]:
        print(f"  f({x}) = {f(x):.6f}")
    
    # Approach from right
    print("\nFrom right (x > 2):")
    for x in [2.1, 2.01, 2.001, 2.0001]:
        print(f"  f({x}) = {f(x):.6f}")
    
    print("\nBoth sides approach 4!")
    print("Therefore: lim(x→2) (x² - 4)/(x - 2) = 4")
    
    print("\n--- Algebraic verification ---")
    print("(x² - 4)/(x - 2) = (x-2)(x+2)/(x-2) = x + 2")
    print("lim(x→2) (x + 2) = 4 ✓")


def example_one_sided_limits():
    """Demonstrate one-sided limits."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: One-Sided Limits")
    print("=" * 60)
    
    print("Consider f(x) = |x|/x")
    print("(Sign function)")
    
    def f(x):
        if x == 0:
            return np.nan
        return np.abs(x) / x
    
    print("\nFrom left (x → 0⁻):")
    for x in [-0.1, -0.01, -0.001]:
        print(f"  f({x}) = {f(x)}")
    print("  Left-hand limit = -1")
    
    print("\nFrom right (x → 0⁺):")
    for x in [0.1, 0.01, 0.001]:
        print(f"  f({x}) = {f(x)}")
    print("  Right-hand limit = +1")
    
    print("\nSince left ≠ right, lim(x→0) |x|/x does NOT exist!")


def example_fundamental_limits():
    """Demonstrate fundamental limits."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Fundamental Limits")
    print("=" * 60)
    
    print("1. lim(x→0) sin(x)/x = 1")
    print("\nNumerical verification:")
    for x in [0.1, 0.01, 0.001, 0.0001]:
        val = np.sin(x) / x
        print(f"   sin({x})/{x} = {val:.10f}")
    
    print("\n2. lim(x→0) (eˣ - 1)/x = 1")
    for x in [0.1, 0.01, 0.001, 0.0001]:
        val = (np.exp(x) - 1) / x
        print(f"   (e^{x} - 1)/{x} = {val:.10f}")
    
    print("\n3. lim(x→∞) (1 + 1/x)ˣ = e")
    for x in [10, 100, 1000, 10000]:
        val = (1 + 1/x)**x
        print(f"   (1 + 1/{x})^{x} = {val:.10f}")
    print(f"   e = {np.e:.10f}")


def example_lhopital():
    """Demonstrate L'Hôpital's Rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: L'Hôpital's Rule")
    print("=" * 60)
    
    print("Find: lim(x→0) (eˣ - 1 - x)/x²")
    print("\nDirect substitution gives 0/0 (indeterminate)")
    
    print("\n--- Applying L'Hôpital ---")
    print("f(x) = eˣ - 1 - x, f'(x) = eˣ - 1")
    print("g(x) = x², g'(x) = 2x")
    
    print("\nlim(x→0) (eˣ - 1)/(2x) still gives 0/0")
    
    print("\n--- Apply L'Hôpital again ---")
    print("f''(x) = eˣ")
    print("g''(x) = 2")
    
    print("\nlim(x→0) eˣ/2 = 1/2")
    
    print("\n--- Numerical verification ---")
    def f(x):
        return (np.exp(x) - 1 - x) / x**2
    
    for x in [0.1, 0.01, 0.001, 0.0001]:
        print(f"   f({x}) = {f(x):.10f}")
    print("   Approaches 0.5 ✓")


def example_limits_at_infinity():
    """Demonstrate limits at infinity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Limits at Infinity")
    print("=" * 60)
    
    print("Polynomial ratios:")
    
    print("\n1. lim(x→∞) (2x² + 3x)/(x² - 1)")
    print("   Divide by x²: lim (2 + 3/x)/(1 - 1/x²) = 2/1 = 2")
    
    def f1(x):
        return (2*x**2 + 3*x) / (x**2 - 1)
    
    for x in [10, 100, 1000, 10000]:
        print(f"   f({x}) = {f1(x):.6f}")
    
    print("\n2. lim(x→∞) x/(x² + 1)")
    print("   Degree of numerator < denominator → 0")
    
    def f2(x):
        return x / (x**2 + 1)
    
    for x in [10, 100, 1000, 10000]:
        print(f"   f({x}) = {f2(x):.6f}")


def example_continuity():
    """Demonstrate continuity concepts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Continuity")
    print("=" * 60)
    
    print("Checking continuity at x = 0 for different functions:")
    
    print("\n1. f(x) = x² (continuous everywhere)")
    print("   f(0) = 0")
    print("   lim(x→0) x² = 0")
    print("   f(0) = lim → Continuous ✓")
    
    print("\n2. f(x) = |x|/x (discontinuous at 0)")
    print("   f(0) is undefined")
    print("   Left limit = -1, Right limit = +1")
    print("   Jump discontinuity at x = 0")
    
    print("\n3. f(x) = (x² - 1)/(x - 1) at x = 1")
    print("   f(1) is undefined (0/0)")
    print("   But lim(x→1) = lim(x→1)(x+1) = 2")
    print("   Removable discontinuity (hole)")
    
    print("\n4. f(x) = 1/x at x = 0")
    print("   f(0) is undefined")
    print("   lim(x→0⁺) = +∞, lim(x→0⁻) = -∞")
    print("   Infinite discontinuity (vertical asymptote)")


def example_squeeze_theorem():
    """Demonstrate Squeeze Theorem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Squeeze Theorem")
    print("=" * 60)
    
    print("Find: lim(x→0) x² sin(1/x)")
    
    print("\nKey insight:")
    print("-1 ≤ sin(1/x) ≤ 1 for all x ≠ 0")
    print("Therefore: -x² ≤ x² sin(1/x) ≤ x²")
    
    print("\nSince lim(x→0) (-x²) = 0 and lim(x→0) x² = 0")
    print("By Squeeze Theorem: lim(x→0) x² sin(1/x) = 0")
    
    print("\n--- Numerical verification ---")
    def f(x):
        return x**2 * np.sin(1/x)
    
    for x in [0.1, 0.01, 0.001, 0.0001]:
        val = f(x)
        bound = x**2
        print(f"   x={x}: f(x)={val:.2e}, bound=±{bound:.2e}")


def example_softmax_temperature():
    """Demonstrate softmax temperature limit."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Softmax Temperature Limit (ML)")
    print("=" * 60)
    
    def softmax(z, T=1.0):
        z_scaled = z / T
        exp_z = np.exp(z_scaled - np.max(z_scaled))
        return exp_z / np.sum(exp_z)
    
    z = np.array([1.0, 2.0, 3.0])
    print(f"Logits z = {z}")
    
    print("\nSoftmax at different temperatures:")
    temperatures = [10.0, 1.0, 0.1, 0.01]
    
    for T in temperatures:
        probs = softmax(z, T)
        print(f"  T = {T:5.2f}: {np.round(probs, 4)}")
    
    print("\nAs T → 0:")
    print("  Softmax becomes 'hard' max: [0, 0, 1]")
    print("  (All probability on largest logit)")
    
    print("\nAs T → ∞:")
    print("  Softmax becomes uniform: [0.333, 0.333, 0.333]")


def example_sigmoid_limits():
    """Demonstrate sigmoid saturation limits."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Sigmoid Saturation (ML)")
    print("=" * 60)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    print("σ(x) = 1/(1 + e^(-x))")
    
    print("\nAs x → +∞:")
    for x in [1, 5, 10, 50, 100]:
        print(f"  σ({x}) = {sigmoid(x):.10f}")
    print("  → 1")
    
    print("\nAs x → -∞:")
    for x in [-1, -5, -10, -50, -100]:
        print(f"  σ({x}) = {sigmoid(x):.10e}")
    print("  → 0")
    
    print("\n--- Implication for gradient ---")
    print("σ'(x) = σ(x)(1 - σ(x))")
    print("At saturation: σ'(x) ≈ 0 → vanishing gradients!")


def example_learning_rate_decay():
    """Demonstrate learning rate decay convergence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Learning Rate Decay (ML)")
    print("=" * 60)
    
    print("For convergence, we need:")
    print("1. Σ αₜ = ∞ (can reach any point)")
    print("2. Σ αₜ² < ∞ (variance goes to zero)")
    
    print("\n--- Example: αₜ = 1/t ---")
    
    def alpha(t):
        return 1 / t
    
    print("α(t) = 1/t")
    
    # Partial sums
    print("\nPartial sums Σ αₜ:")
    for n in [10, 100, 1000, 10000]:
        harmonic = sum(1/t for t in range(1, n+1))
        print(f"  Σ(t=1 to {n}) 1/t = {harmonic:.4f}")
    print("  → ∞ (harmonic series diverges) ✓")
    
    print("\nPartial sums Σ αₜ²:")
    for n in [10, 100, 1000, 10000]:
        sum_sq = sum(1/t**2 for t in range(1, n+1))
        print(f"  Σ(t=1 to {n}) 1/t² = {sum_sq:.6f}")
    print(f"  → π²/6 ≈ {np.pi**2/6:.6f} (converges) ✓")


def example_numerical_stability():
    """Demonstrate numerical limits and stability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Numerical Stability")
    print("=" * 60)
    
    print("Computing (e^x - 1)/x as x → 0")
    print("Limit is 1, but naive computation fails:")
    
    print("\n--- Naive computation ---")
    for x in [1e-5, 1e-10, 1e-15, 1e-16]:
        naive = (np.exp(x) - 1) / x
        stable = np.expm1(x) / x if x != 0 else 1.0
        print(f"  x = {x:.0e}: naive = {naive:.10f}, stable = {stable:.10f}")
    
    print("\nNaive fails due to catastrophic cancellation!")
    print("Use np.expm1(x) which computes e^x - 1 accurately for small x.")


if __name__ == "__main__":
    example_limit_intuition()
    example_one_sided_limits()
    example_fundamental_limits()
    example_lhopital()
    example_limits_at_infinity()
    example_continuity()
    example_squeeze_theorem()
    example_softmax_temperature()
    example_sigmoid_limits()
    example_learning_rate_decay()
    example_numerical_stability()
