"""
Proof Techniques - Examples
===========================
Demonstrations of mathematical proof methods.
"""

import numpy as np
from fractions import Fraction
from math import gcd, sqrt


def example_direct_proof():
    """Demonstrate direct proof technique."""
    print("=" * 60)
    print("EXAMPLE 1: Direct Proof")
    print("=" * 60)
    
    print("\n--- Theorem: Sum of two even numbers is even ---")
    print("\nProof:")
    print("1. Let a and b be even numbers")
    print("2. By definition: a = 2m, b = 2n for some integers m, n")
    print("3. a + b = 2m + 2n = 2(m + n)")
    print("4. Since (m + n) is an integer, a + b is even ∎")
    
    print("\nVerification with examples:")
    examples = [(4, 6), (10, 22), (100, 200)]
    for a, b in examples:
        result = a + b
        print(f"  {a} + {b} = {result} (even: {result % 2 == 0})")
    
    print("\n--- Theorem: Product of two odd numbers is odd ---")
    print("\nProof:")
    print("1. Let a and b be odd numbers")
    print("2. Then a = 2m + 1, b = 2n + 1 for some integers m, n")
    print("3. a · b = (2m + 1)(2n + 1)")
    print("4.      = 4mn + 2m + 2n + 1")
    print("5.      = 2(2mn + m + n) + 1")
    print("6. This has form 2k + 1, so a · b is odd ∎")
    
    print("\nVerification:")
    examples = [(3, 5), (7, 9), (11, 13)]
    for a, b in examples:
        result = a * b
        print(f"  {a} × {b} = {result} (odd: {result % 2 == 1})")


def example_proof_by_contradiction():
    """Demonstrate proof by contradiction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Proof by Contradiction")
    print("=" * 60)
    
    print("\n--- Theorem: √2 is irrational ---")
    print("\nProof by contradiction:")
    print("1. Assume √2 is rational")
    print("2. Then √2 = p/q where gcd(p,q) = 1 (lowest terms)")
    print("3. Squaring: 2 = p²/q², so p² = 2q²")
    print("4. Therefore p² is even, so p is even")
    print("5. Let p = 2k, then 4k² = 2q², so q² = 2k²")
    print("6. Therefore q² is even, so q is even")
    print("7. Both p and q even contradicts gcd(p,q) = 1")
    print("8. Contradiction! Therefore √2 is irrational ∎")
    
    print("\nNumerical evidence (√2 has no repeating decimal):")
    sqrt2 = sqrt(2)
    print(f"  √2 ≈ {sqrt2:.20f}...")
    
    print("\n--- Theorem: There is no largest integer ---")
    print("\nProof by contradiction:")
    print("1. Assume N is the largest integer")
    print("2. Consider N + 1")
    print("3. N + 1 > N and N + 1 is an integer")
    print("4. Contradiction! N is not the largest")
    print("5. Therefore no largest integer exists ∎")


def example_proof_by_contrapositive():
    """Demonstrate proof by contrapositive."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Proof by Contrapositive")
    print("=" * 60)
    
    print("\n--- Theorem: If n² is odd, then n is odd ---")
    print("\nOriginal: n² odd ⟹ n odd")
    print("Contrapositive: n even ⟹ n² even")
    print("\nProof of contrapositive:")
    print("1. Assume n is even")
    print("2. Then n = 2k for some integer k")
    print("3. n² = (2k)² = 4k² = 2(2k²)")
    print("4. Since 2k² is an integer, n² is even ∎")
    
    print("\nVerification:")
    for n in range(1, 11):
        n_odd = n % 2 == 1
        n2_odd = (n**2) % 2 == 1
        arrow = "⟹" if n2_odd else "  "
        print(f"  n={n}: n²={n**2:3d}, n² odd: {n2_odd}, n odd: {n_odd} {arrow}")
    
    print("\n--- Theorem: If xy is odd, then both x and y are odd ---")
    print("\nContrapositive: If x is even OR y is even, then xy is even")
    print("\nProof:")
    print("1. Assume x is even (WLOG)")
    print("2. Then x = 2k for some integer k")
    print("3. xy = 2ky")
    print("4. Since ky is an integer, xy is even ∎")


def example_mathematical_induction():
    """Demonstrate mathematical induction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Mathematical Induction")
    print("=" * 60)
    
    print("\n--- Theorem: Σᵢ₌₁ⁿ i = n(n+1)/2 ---")
    print("\nProof by induction:")
    
    print("\nBase case (n = 1):")
    print("  LHS: Σᵢ₌₁¹ i = 1")
    print("  RHS: 1(1+1)/2 = 1 ✓")
    
    print("\nInductive hypothesis:")
    print("  Assume Σᵢ₌₁ᵏ i = k(k+1)/2 for some k ≥ 1")
    
    print("\nInductive step (show for k+1):")
    print("  Σᵢ₌₁ᵏ⁺¹ i = (Σᵢ₌₁ᵏ i) + (k+1)")
    print("           = k(k+1)/2 + (k+1)  [by hypothesis]")
    print("           = [k(k+1) + 2(k+1)]/2")
    print("           = (k+1)(k+2)/2")
    print("  This is the formula with n = k+1 ✓")
    
    print("\nBy induction, the formula holds for all n ≥ 1 ∎")
    
    print("\nVerification:")
    for n in [1, 5, 10, 100]:
        computed = sum(range(1, n + 1))
        formula = n * (n + 1) // 2
        print(f"  n={n:3d}: Σi = {computed:5d}, n(n+1)/2 = {formula:5d}")


def example_strong_induction():
    """Demonstrate strong induction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Strong Induction")
    print("=" * 60)
    
    print("\n--- Theorem: Every integer n > 1 has a prime factorization ---")
    print("\nProof by strong induction:")
    
    print("\nBase case (n = 2):")
    print("  2 is prime, factorization: 2 ✓")
    
    print("\nStrong inductive hypothesis:")
    print("  Assume all integers from 2 to k have prime factorizations")
    
    print("\nInductive step (show for k+1):")
    print("  Case 1: k+1 is prime")
    print("    → Factorization is just (k+1) itself ✓")
    print("  Case 2: k+1 is composite")
    print("    → k+1 = a × b where 1 < a, b < k+1")
    print("    → By hypothesis, a and b have prime factorizations")
    print("    → Product of their factorizations gives k+1's factorization ✓")
    
    print("\nBy strong induction, all n > 1 have prime factorizations ∎")
    
    print("\nExamples of prime factorization:")
    
    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    for n in [12, 28, 100, 127, 1024]:
        factors = prime_factors(n)
        print(f"  {n} = {' × '.join(map(str, factors))}")


def example_proof_by_cases():
    """Demonstrate proof by cases."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Proof by Cases")
    print("=" * 60)
    
    print("\n--- Theorem: |xy| = |x| × |y| for all real x, y ---")
    print("\nProof by cases:")
    
    print("\nCase 1: x ≥ 0 and y ≥ 0")
    print("  |xy| = xy = |x| × |y| ✓")
    
    print("\nCase 2: x ≥ 0 and y < 0")
    print("  |xy| = |-xy| = -xy = x × (-y) = |x| × |y| ✓")
    
    print("\nCase 3: x < 0 and y ≥ 0")
    print("  |xy| = |-xy| = -xy = (-x) × y = |x| × |y| ✓")
    
    print("\nCase 4: x < 0 and y < 0")
    print("  |xy| = xy = (-x) × (-y) = |x| × |y| ✓")
    
    print("\nAll cases covered ∎")
    
    print("\nVerification:")
    test_cases = [(3, 4), (3, -4), (-3, 4), (-3, -4), (0, 5), (-2, 0)]
    for x, y in test_cases:
        lhs = abs(x * y)
        rhs = abs(x) * abs(y)
        print(f"  x={x:2d}, y={y:2d}: |xy| = {lhs:2d}, |x||y| = {rhs:2d}, equal: {lhs == rhs}")


def example_existence_uniqueness():
    """Demonstrate existence and uniqueness proofs."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Existence and Uniqueness")
    print("=" * 60)
    
    print("\n--- Theorem: Division Algorithm ---")
    print("For integers a ≥ 0 and d > 0, there exist UNIQUE q and r such that:")
    print("  a = qd + r, where 0 ≤ r < d")
    
    print("\nExistence proof (constructive):")
    print("  q = ⌊a/d⌋ (floor division)")
    print("  r = a - qd (remainder)")
    print("  By construction: a = qd + r")
    print("  Since q = ⌊a/d⌋: qd ≤ a < (q+1)d")
    print("  So: 0 ≤ a - qd < d, meaning 0 ≤ r < d ✓")
    
    print("\nUniqueness proof:")
    print("  Suppose a = q₁d + r₁ = q₂d + r₂ with 0 ≤ r₁, r₂ < d")
    print("  Then (q₁ - q₂)d = r₂ - r₁")
    print("  Since -d < r₂ - r₁ < d and d|(r₂ - r₁)")
    print("  We must have r₂ - r₁ = 0, so r₁ = r₂")
    print("  Therefore q₁ = q₂ ∎")
    
    print("\nExamples:")
    test_cases = [(17, 5), (100, 7), (25, 4), (0, 3)]
    for a, d in test_cases:
        q, r = divmod(a, d)
        print(f"  {a} = {q} × {d} + {r}")
        print(f"    Verify: {q}×{d}+{r} = {q*d + r}, 0 ≤ {r} < {d}: {0 <= r < d}")


def example_induction_inequality():
    """Demonstrate induction for inequalities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Induction for Inequalities")
    print("=" * 60)
    
    print("\n--- Theorem: For n ≥ 3, 2ⁿ > 2n + 1 ---")
    
    print("\nBase case (n = 3):")
    print(f"  LHS: 2³ = 8")
    print(f"  RHS: 2(3) + 1 = 7")
    print(f"  8 > 7 ✓")
    
    print("\nInductive hypothesis:")
    print("  Assume 2ᵏ > 2k + 1 for some k ≥ 3")
    
    print("\nInductive step:")
    print("  2ᵏ⁺¹ = 2 × 2ᵏ > 2(2k + 1) = 4k + 2  [by hypothesis]")
    print("  Need to show: 4k + 2 > 2(k+1) + 1 = 2k + 3")
    print("  4k + 2 - (2k + 3) = 2k - 1")
    print("  For k ≥ 3: 2k - 1 ≥ 5 > 0 ✓")
    print("  So 2ᵏ⁺¹ > 4k + 2 > 2(k+1) + 1 ∎")
    
    print("\nVerification:")
    for n in range(3, 12):
        lhs = 2 ** n
        rhs = 2 * n + 1
        print(f"  n={n:2d}: 2ⁿ = {lhs:4d}, 2n+1 = {rhs:2d}, ratio: {lhs/rhs:.2f}")


def example_ml_convergence_proof():
    """Demonstrate a convergence proof structure."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: ML Convergence Proof (Sketch)")
    print("=" * 60)
    
    print("\n--- Theorem: Gradient Descent Convergence for Convex Functions ---")
    print("\nSetup:")
    print("  f: convex, L-smooth (|∇f(x) - ∇f(y)| ≤ L|x - y|)")
    print("  Update: xₜ₊₁ = xₜ - η∇f(xₜ) with learning rate η = 1/L")
    
    print("\nProof sketch:")
    print("\n1. By L-smoothness (direct proof):")
    print("   f(xₜ₊₁) ≤ f(xₜ) + ⟨∇f(xₜ), xₜ₊₁ - xₜ⟩ + (L/2)||xₜ₊₁ - xₜ||²")
    
    print("\n2. Substituting xₜ₊₁ - xₜ = -η∇f(xₜ):")
    print("   f(xₜ₊₁) ≤ f(xₜ) - η||∇f(xₜ)||² + (Lη²/2)||∇f(xₜ)||²")
    
    print("\n3. With η = 1/L:")
    print("   f(xₜ₊₁) ≤ f(xₜ) - (1/2L)||∇f(xₜ)||²")
    
    print("\n4. By convexity:")
    print("   f(xₜ) - f(x*) ≤ ⟨∇f(xₜ), xₜ - x*⟩ ≤ ||∇f(xₜ)|| ||xₜ - x*||")
    
    print("\n5. Telescoping sum (by induction):")
    print("   f(xₜ) - f(x*) ≤ L||x₀ - x*||² / (2T)")
    
    print("\nConclusion: Convergence rate O(1/T) ∎")
    
    # Numerical demonstration
    print("\nNumerical demonstration with f(x) = x²:")
    
    def f(x):
        return x ** 2
    
    def grad_f(x):
        return 2 * x
    
    L = 2  # Lipschitz constant for f(x) = x²
    eta = 1 / L
    x = 10.0
    x_star = 0.0
    
    print(f"\n  L = {L}, η = 1/L = {eta}")
    print(f"  x₀ = {x}, x* = {x_star}")
    print(f"\n  {'T':>4} {'xₜ':>10} {'f(xₜ)':>10} {'f(xₜ)-f(x*)':>12}")
    
    for t in range(11):
        f_diff = f(x) - f(x_star)
        print(f"  {t:4d} {x:10.4f} {f(x):10.4f} {f_diff:12.4f}")
        x = x - eta * grad_f(x)


def example_well_ordering_principle():
    """Demonstrate the well-ordering principle."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Well-Ordering Principle")
    print("=" * 60)
    
    print("\n--- Well-Ordering Principle ---")
    print("Every non-empty set of positive integers has a smallest element.")
    
    print("\n--- Application: Prove √2 is irrational (alternative proof) ---")
    print("\nAssume √2 = p/q where p, q are positive integers.")
    print("Let S = {n ∈ ℤ⁺ : n√2 ∈ ℤ⁺}")
    print("Then q ∈ S (since q√2 = p is a positive integer)")
    print("By well-ordering, S has a smallest element m.")
    print("")
    print("Let m' = m√2 - m = m(√2 - 1)")
    print("Since 0 < √2 - 1 < 1, we have 0 < m' < m")
    print("")
    print("Now m'√2 = m(√2 - 1)√2 = 2m - m√2")
    print("Both 2m and m√2 are integers, so m'√2 is an integer.")
    print("")
    print("But then m' ∈ S with m' < m, contradicting minimality of m!")
    print("Therefore our assumption was wrong, and √2 is irrational ∎")


if __name__ == "__main__":
    example_direct_proof()
    example_proof_by_contradiction()
    example_proof_by_contrapositive()
    example_mathematical_induction()
    example_strong_induction()
    example_proof_by_cases()
    example_existence_uniqueness()
    example_induction_inequality()
    example_ml_convergence_proof()
    example_well_ordering_principle()
