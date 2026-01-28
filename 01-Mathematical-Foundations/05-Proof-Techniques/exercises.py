"""
Proof Techniques - Exercises
============================
Practice problems for mathematical proofs.
"""

import numpy as np
from math import sqrt, factorial


class ProofExercises:
    """Exercises for proof techniques."""
    
    # ==================== DIRECT PROOF EXERCISES ====================
    
    def exercise_1_direct_proof(self):
        """
        Exercise 1: Direct Proofs
        
        Prove the following using direct proof:
        a) The sum of an even number and an odd number is odd.
        b) The square of any odd number is odd.
        c) If n is an integer, then n² + n is even.
        d) The sum of any three consecutive integers is divisible by 3.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solutions: Direct Proofs")
        
        print("\na) Sum of even and odd is odd")
        print("   Let a be even: a = 2m")
        print("   Let b be odd:  b = 2n + 1")
        print("   a + b = 2m + 2n + 1 = 2(m + n) + 1")
        print("   This has form 2k + 1, so a + b is odd ∎")
        
        # Verify
        examples = [(4, 3), (10, 7), (2, 1)]
        for even, odd in examples:
            print(f"   Verify: {even} + {odd} = {even + odd} (odd: {(even+odd) % 2 == 1})")
        
        print("\nb) Square of odd number is odd")
        print("   Let n be odd: n = 2k + 1")
        print("   n² = (2k + 1)² = 4k² + 4k + 1 = 2(2k² + 2k) + 1")
        print("   This has form 2m + 1, so n² is odd ∎")
        
        print("\nc) n² + n is even for any integer n")
        print("   n² + n = n(n + 1)")
        print("   n and n+1 are consecutive, so one is even")
        print("   Product of even with anything is even ∎")
        
        # Verify
        for n in range(-3, 6):
            result = n**2 + n
            print(f"   n={n:2d}: n² + n = {result:2d} (even: {result % 2 == 0})")
        
        print("\nd) Sum of three consecutive integers divisible by 3")
        print("   Let integers be n, n+1, n+2")
        print("   Sum = n + (n+1) + (n+2) = 3n + 3 = 3(n + 1)")
        print("   This is clearly divisible by 3 ∎")
    
    # ==================== CONTRAPOSITIVE EXERCISES ====================
    
    def exercise_2_contrapositive(self):
        """
        Exercise 2: Proof by Contrapositive
        
        Prove using contrapositive:
        a) If n² is even, then n is even.
        b) If x² - 2x + 1 ≠ 0, then x ≠ 1.
        c) If n³ is odd, then n is odd.
        d) If n is an integer such that n² is divisible by 4, then n is even.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("Exercise 2 Solutions: Contrapositive")
        
        print("\na) If n² is even, then n is even")
        print("   Contrapositive: If n is odd, then n² is odd")
        print("   Proof: Let n = 2k + 1")
        print("   n² = 4k² + 4k + 1 = 2(2k² + 2k) + 1 (odd) ∎")
        
        print("\nb) If x² - 2x + 1 ≠ 0, then x ≠ 1")
        print("   Contrapositive: If x = 1, then x² - 2x + 1 = 0")
        print("   Proof: 1² - 2(1) + 1 = 1 - 2 + 1 = 0 ✓ ∎")
        
        print("\nc) If n³ is odd, then n is odd")
        print("   Contrapositive: If n is even, then n³ is even")
        print("   Proof: Let n = 2k")
        print("   n³ = 8k³ = 2(4k³) (even) ∎")
        
        print("\nd) If n² divisible by 4, then n is even")
        print("   Contrapositive: If n is odd, then n² not divisible by 4")
        print("   Proof: Let n = 2k + 1")
        print("   n² = 4k² + 4k + 1 = 4(k² + k) + 1")
        print("   n² ≡ 1 (mod 4), so not divisible by 4 ∎")
        
        # Verify
        print("\n   Verification:")
        for n in range(1, 8):
            n2 = n ** 2
            div4 = n2 % 4 == 0
            n_even = n % 2 == 0
            print(f"   n={n}: n²={n2:2d}, n²÷4: {div4}, n even: {n_even}")
    
    # ==================== CONTRADICTION EXERCISES ====================
    
    def exercise_3_contradiction(self):
        """
        Exercise 3: Proof by Contradiction
        
        Prove using contradiction:
        a) There is no smallest positive rational number.
        b) √3 is irrational.
        c) The set of prime numbers is infinite.
        d) If a² is even, then a is even.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("Exercise 3 Solutions: Contradiction")
        
        print("\na) No smallest positive rational number")
        print("   Assume r is the smallest positive rational")
        print("   Consider r/2: this is positive and rational")
        print("   r/2 < r, contradicting r being smallest ∎")
        
        print("\nb) √3 is irrational")
        print("   Assume √3 = p/q with gcd(p,q) = 1")
        print("   Then 3 = p²/q², so p² = 3q²")
        print("   Therefore 3 | p², so 3 | p")
        print("   Let p = 3k, then 9k² = 3q², so q² = 3k²")
        print("   Therefore 3 | q², so 3 | q")
        print("   But 3|p and 3|q contradicts gcd(p,q) = 1 ∎")
        
        print(f"\n   √3 ≈ {sqrt(3):.10f}... (non-repeating)")
        
        print("\nc) Infinitely many primes")
        print("   Assume only finitely many primes: p₁, p₂, ..., pₙ")
        print("   Let N = p₁ · p₂ · ... · pₙ + 1")
        print("   N is not divisible by any pᵢ (remainder 1)")
        print("   Either N is prime, or has a prime factor not in our list")
        print("   Contradiction! ∎")
        
        print("\nd) a² even ⟹ a even (by contradiction)")
        print("   Assume a² even but a odd")
        print("   Then a = 2k + 1 for some integer k")
        print("   a² = 4k² + 4k + 1 = 2(2k² + 2k) + 1 (odd)")
        print("   But a² is even - contradiction! ∎")
    
    # ==================== INDUCTION EXERCISES ====================
    
    def exercise_4_induction(self):
        """
        Exercise 4: Mathematical Induction
        
        Prove by induction:
        a) Σᵢ₌₁ⁿ i² = n(n+1)(2n+1)/6
        b) Σᵢ₌₁ⁿ (2i-1) = n² (sum of first n odd numbers)
        c) n! > 2ⁿ for n ≥ 4
        d) 1 + 2 + 4 + ... + 2ⁿ = 2ⁿ⁺¹ - 1
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("Exercise 4 Solutions: Induction")
        
        print("\na) Σᵢ₌₁ⁿ i² = n(n+1)(2n+1)/6")
        
        print("\n   Base case (n=1):")
        print("   LHS: 1² = 1")
        print("   RHS: 1(2)(3)/6 = 1 ✓")
        
        print("\n   Inductive step:")
        print("   Assume Σᵢ₌₁ᵏ i² = k(k+1)(2k+1)/6")
        print("   Σᵢ₌₁ᵏ⁺¹ i² = k(k+1)(2k+1)/6 + (k+1)²")
        print("   = (k+1)[k(2k+1)/6 + (k+1)]")
        print("   = (k+1)[(2k² + k + 6k + 6)/6]")
        print("   = (k+1)(2k² + 7k + 6)/6")
        print("   = (k+1)(k+2)(2k+3)/6")
        print("   = (k+1)((k+1)+1)(2(k+1)+1)/6 ✓ ∎")
        
        print("\n   Verification:")
        for n in [1, 5, 10]:
            computed = sum(i**2 for i in range(1, n+1))
            formula = n * (n+1) * (2*n+1) // 6
            print(f"   n={n:2d}: Σi² = {computed:4d}, formula = {formula:4d}")
        
        print("\nb) Σᵢ₌₁ⁿ (2i-1) = n²")
        
        print("\n   Base case (n=1):")
        print("   LHS: 2(1)-1 = 1")
        print("   RHS: 1² = 1 ✓")
        
        print("\n   Inductive step:")
        print("   Assume Σᵢ₌₁ᵏ (2i-1) = k²")
        print("   Σᵢ₌₁ᵏ⁺¹ (2i-1) = k² + (2(k+1)-1)")
        print("   = k² + 2k + 1 = (k+1)² ✓ ∎")
        
        print("\nc) n! > 2ⁿ for n ≥ 4")
        
        print("\n   Base case (n=4):")
        print(f"   4! = {factorial(4)} > 2⁴ = {2**4} ✓")
        
        print("\n   Inductive step:")
        print("   Assume k! > 2ᵏ for some k ≥ 4")
        print("   (k+1)! = (k+1) · k! > (k+1) · 2ᵏ")
        print("   Since k+1 ≥ 5 > 2: (k+1)! > 2 · 2ᵏ = 2ᵏ⁺¹ ✓ ∎")
        
        print("\n   Verification:")
        for n in range(4, 10):
            print(f"   {n}! = {factorial(n):6d} > 2^{n} = {2**n:5d}: {factorial(n) > 2**n}")
        
        print("\nd) 1 + 2 + 4 + ... + 2ⁿ = 2ⁿ⁺¹ - 1")
        
        print("\n   Base case (n=0):")
        print("   LHS: 2⁰ = 1")
        print("   RHS: 2¹ - 1 = 1 ✓")
        
        print("\n   Inductive step:")
        print("   Assume Σᵢ₌₀ᵏ 2ⁱ = 2ᵏ⁺¹ - 1")
        print("   Σᵢ₌₀ᵏ⁺¹ 2ⁱ = (2ᵏ⁺¹ - 1) + 2ᵏ⁺¹")
        print("   = 2·2ᵏ⁺¹ - 1 = 2ᵏ⁺² - 1 ✓ ∎")
    
    # ==================== STRONG INDUCTION EXERCISES ====================
    
    def exercise_5_strong_induction(self):
        """
        Exercise 5: Strong Induction
        
        Prove using strong induction:
        a) Every amount of postage ≥ 12 cents can be made with 4¢ and 5¢ stamps.
        b) Every integer n ≥ 2 can be written as a product of primes.
        c) Fibonacci: Fₙ < 2ⁿ for all n ≥ 1.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("Exercise 5 Solutions: Strong Induction")
        
        print("\na) Postage ≥ 12 cents with 4¢ and 5¢ stamps")
        
        print("\n   Base cases:")
        print("   12 = 3×4 ✓")
        print("   13 = 2×4 + 1×5 ✓")
        print("   14 = 1×4 + 2×5 ✓")
        print("   15 = 0×4 + 3×5 ✓")
        
        print("\n   Strong inductive step:")
        print("   For n ≥ 16, by hypothesis n-4 can be made (since n-4 ≥ 12)")
        print("   Add one more 4¢ stamp to get n ∎")
        
        print("\n   Verification:")
        def make_postage(n):
            for fours in range(n // 4 + 1):
                remainder = n - 4 * fours
                if remainder >= 0 and remainder % 5 == 0:
                    return (fours, remainder // 5)
            return None
        
        for n in range(12, 21):
            result = make_postage(n)
            if result:
                print(f"   {n}¢ = {result[0]}×4¢ + {result[1]}×5¢")
        
        print("\nb) Every n ≥ 2 is a product of primes")
        print("   (See Example 5 in examples.py)")
        
        print("\nc) Fₙ < 2ⁿ for n ≥ 1")
        
        print("\n   Base cases:")
        print("   F₁ = 1 < 2¹ = 2 ✓")
        print("   F₂ = 1 < 2² = 4 ✓")
        
        print("\n   Strong inductive step:")
        print("   Assume Fⱼ < 2ʲ for all j ≤ k")
        print("   Fₖ₊₁ = Fₖ + Fₖ₋₁ < 2ᵏ + 2ᵏ⁻¹ (by hypothesis)")
        print("   = 2ᵏ⁻¹(2 + 1) = 3·2ᵏ⁻¹ < 4·2ᵏ⁻¹ = 2ᵏ⁺¹ ✓ ∎")
        
        print("\n   Verification:")
        fib = [0, 1, 1]
        for n in range(3, 15):
            fib.append(fib[-1] + fib[-2])
        for n in range(1, 12):
            print(f"   F_{n:2d} = {fib[n]:4d} < 2^{n} = {2**n:4d}: {fib[n] < 2**n}")
    
    # ==================== MIXED EXERCISES ====================
    
    def exercise_6_choose_technique(self):
        """
        Exercise 6: Choose the Appropriate Technique
        
        For each statement, choose the best proof technique and prove:
        a) If 3n + 2 is odd, then n is odd.
        b) There are no integers a and b such that a² - b² = 10.
        c) For all n ≥ 1: Σᵢ₌₁ⁿ 1/(i(i+1)) = n/(n+1)
        d) For all x ∈ ℝ: x² - 6x + 10 > 0
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solutions: Choose Technique")
        
        print("\na) If 3n + 2 is odd, then n is odd")
        print("   Best technique: CONTRAPOSITIVE")
        print("\n   Contrapositive: If n is even, then 3n + 2 is even")
        print("   Let n = 2k")
        print("   3n + 2 = 6k + 2 = 2(3k + 1) (even) ∎")
        
        print("\nb) No integers a, b with a² - b² = 10")
        print("   Best technique: CASES / CONTRADICTION")
        print("\n   Note: a² - b² = (a+b)(a-b) = 10")
        print("   Factors of 10: 1×10, 2×5")
        print("\n   Case 1: a+b = 10, a-b = 1")
        print("   → 2a = 11, a = 5.5 (not integer)")
        print("\n   Case 2: a+b = 5, a-b = 2")
        print("   → 2a = 7, a = 3.5 (not integer)")
        print("\n   Case 3: a+b = -10, a-b = -1")
        print("   → 2a = -11, a = -5.5 (not integer)")
        print("\n   Case 4: a+b = -5, a-b = -2")
        print("   → 2a = -7, a = -3.5 (not integer)")
        print("\n   No integer solutions exist ∎")
        
        print("\nc) Σᵢ₌₁ⁿ 1/(i(i+1)) = n/(n+1)")
        print("   Best technique: INDUCTION (or telescoping)")
        print("\n   Note: 1/(i(i+1)) = 1/i - 1/(i+1) (partial fractions)")
        print("   This telescopes!")
        print("   Σ = (1 - 1/2) + (1/2 - 1/3) + ... + (1/n - 1/(n+1))")
        print("   = 1 - 1/(n+1) = n/(n+1) ∎")
        
        print("\n   Verification:")
        for n in [1, 5, 10]:
            computed = sum(1/(i*(i+1)) for i in range(1, n+1))
            formula = n / (n + 1)
            print(f"   n={n:2d}: Σ = {computed:.6f}, n/(n+1) = {formula:.6f}")
        
        print("\nd) x² - 6x + 10 > 0 for all x ∈ ℝ")
        print("   Best technique: DIRECT (complete the square)")
        print("\n   x² - 6x + 10 = (x² - 6x + 9) + 1")
        print("   = (x - 3)² + 1")
        print("   ≥ 0 + 1 = 1 > 0 ∎")
        
        print("\n   Verification (minimum at x = 3):")
        for x in [0, 1, 2, 3, 4, 5, 6]:
            val = x**2 - 6*x + 10
            print(f"   x={x}: x² - 6x + 10 = {val}")


def example_epsilon_delta():
    """Bonus: Epsilon-delta proof structure."""
    print("\n" + "=" * 70)
    print("BONUS: Epsilon-Delta Proof (Limits)")
    print("=" * 70)
    
    print("\nDefinition of limit:")
    print("lim(x→a) f(x) = L means:")
    print("∀ε > 0, ∃δ > 0 such that 0 < |x - a| < δ ⟹ |f(x) - L| < ε")
    
    print("\n--- Prove: lim(x→2) (3x - 1) = 5 ---")
    print("\nScratch work:")
    print("  We need |f(x) - L| = |(3x - 1) - 5| = |3x - 6| = 3|x - 2| < ε")
    print("  This requires |x - 2| < ε/3")
    print("  So we choose δ = ε/3")
    
    print("\nFormal proof:")
    print("  Let ε > 0 be given.")
    print("  Choose δ = ε/3.")
    print("  Suppose 0 < |x - 2| < δ.")
    print("  Then |(3x - 1) - 5| = |3x - 6| = 3|x - 2| < 3δ = 3(ε/3) = ε ∎")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = ProofExercises()
    
    print("PROOF TECHNIQUES EXERCISES")
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
    
    example_epsilon_delta()


if __name__ == "__main__":
    run_all_exercises()
