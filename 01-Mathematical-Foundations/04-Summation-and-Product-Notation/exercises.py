"""
Summation and Product Notation - Exercises
==========================================
Practice problems for sigma and pi notation.
"""

import numpy as np


class SummationExercises:
    """Exercises for summation and product notation."""
    
    # ==================== BASIC EXERCISES ====================
    
    def exercise_1_expand_summation(self):
        """
        Exercise 1: Expand Summations
        
        Expand and compute:
        a) Σᵢ₌₁⁴ i
        b) Σᵢ₌₁⁵ i²
        c) Σₖ₌₀³ 2ᵏ
        d) Σⱼ₌₂⁴ (3j - 1)
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution:")
        
        print("\na) Σᵢ₌₁⁴ i")
        terms = [i for i in range(1, 5)]
        print(f"   = {' + '.join(map(str, terms))}")
        print(f"   = {sum(terms)}")
        
        print("\nb) Σᵢ₌₁⁵ i²")
        terms = [i**2 for i in range(1, 6)]
        print(f"   = {' + '.join(map(str, terms))}")
        print(f"   = {sum(terms)}")
        
        print("\nc) Σₖ₌₀³ 2ᵏ")
        terms = [2**k for k in range(4)]
        print(f"   = {' + '.join(map(str, terms))}")
        print(f"   = {sum(terms)}")
        
        print("\nd) Σⱼ₌₂⁴ (3j - 1)")
        terms = [3*j - 1 for j in range(2, 5)]
        print(f"   = {' + '.join(map(str, terms))}")
        print(f"   = {sum(terms)}")
    
    def exercise_2_write_in_notation(self):
        """
        Exercise 2: Write in Summation Notation
        
        Express using Σ notation:
        a) 1 + 2 + 3 + 4 + 5 + 6
        b) 2 + 4 + 6 + 8 + 10
        c) 1 + 4 + 9 + 16 + 25
        d) 1 + 1/2 + 1/3 + 1/4 + 1/5
        e) 1 - 2 + 3 - 4 + 5 - 6
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("Exercise 2 Solution:")
        
        print("\na) 1 + 2 + 3 + 4 + 5 + 6")
        print("   = Σᵢ₌₁⁶ i")
        
        print("\nb) 2 + 4 + 6 + 8 + 10")
        print("   = Σᵢ₌₁⁵ 2i")
        
        print("\nc) 1 + 4 + 9 + 16 + 25")
        print("   = Σᵢ₌₁⁵ i²")
        
        print("\nd) 1 + 1/2 + 1/3 + 1/4 + 1/5")
        print("   = Σᵢ₌₁⁵ (1/i)")
        
        print("\ne) 1 - 2 + 3 - 4 + 5 - 6")
        print("   = Σᵢ₌₁⁶ (-1)^(i+1) × i")
        print("   Verify:", sum((-1)**(i+1) * i for i in range(1, 7)))
    
    def exercise_3_use_formulas(self):
        """
        Exercise 3: Apply Summation Formulas
        
        Compute using formulas (not expansion):
        a) Σᵢ₌₁¹⁰⁰ i
        b) Σᵢ₌₁⁵⁰ i²
        c) Σᵢ₌₀¹⁰ 2ⁱ
        d) Σᵢ₌₀^∞ (1/2)ⁱ
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("Exercise 3 Solution:")
        
        print("\na) Σᵢ₌₁¹⁰⁰ i using formula n(n+1)/2")
        n = 100
        result = n * (n + 1) // 2
        print(f"   = 100 × 101 / 2 = {result}")
        print(f"   Verify by sum: {sum(range(1, 101))}")
        
        print("\nb) Σᵢ₌₁⁵⁰ i² using formula n(n+1)(2n+1)/6")
        n = 50
        result = n * (n + 1) * (2*n + 1) // 6
        print(f"   = 50 × 51 × 101 / 6 = {result}")
        print(f"   Verify: {sum(i**2 for i in range(1, 51))}")
        
        print("\nc) Σᵢ₌₀¹⁰ 2ⁱ using formula (1 - r^(n+1))/(1-r)")
        r, n = 2, 10
        result = (1 - r**(n+1)) // (1 - r)
        print(f"   = (1 - 2¹¹)/(1-2) = (1 - 2048)/(-1) = {result}")
        print(f"   Verify: {sum(2**i for i in range(11))}")
        
        print("\nd) Σᵢ₌₀^∞ (1/2)ⁱ using formula 1/(1-r) for |r|<1")
        r = 0.5
        result = 1 / (1 - r)
        print(f"   = 1/(1-0.5) = 1/0.5 = {result}")
        partial = sum(0.5**i for i in range(100))
        print(f"   Verify (partial sum, 100 terms): {partial:.6f}")
    
    def exercise_4_products(self):
        """
        Exercise 4: Product Notation
        
        Compute:
        a) Πᵢ₌₁⁵ i (i.e., 5!)
        b) Πᵢ₌₁⁴ 2i
        c) Πᵢ₌₁³ iⁱ
        d) Πᵢ₌₂⁵ (i-1)
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("Exercise 4 Solution:")
        
        print("\na) Πᵢ₌₁⁵ i = 5!")
        result = 1
        for i in range(1, 6):
            result *= i
        print(f"   = 1 × 2 × 3 × 4 × 5 = {result}")
        
        print("\nb) Πᵢ₌₁⁴ 2i")
        result = 1
        terms = []
        for i in range(1, 5):
            terms.append(str(2*i))
            result *= 2*i
        print(f"   = {' × '.join(terms)} = {result}")
        print(f"   = 2⁴ × 4! = 16 × 24 = {16 * 24}")
        
        print("\nc) Πᵢ₌₁³ iⁱ")
        result = 1
        terms = []
        for i in range(1, 4):
            terms.append(f"{i}^{i}")
            result *= i**i
        print(f"   = {' × '.join(terms)}")
        print(f"   = 1 × 4 × 27 = {result}")
        
        print("\nd) Πᵢ₌₂⁵ (i-1)")
        result = 1
        terms = []
        for i in range(2, 6):
            terms.append(str(i-1))
            result *= (i-1)
        print(f"   = {' × '.join(terms)} = {result}")
        print(f"   = 4! = {result}")
    
    # ==================== INTERMEDIATE EXERCISES ====================
    
    def exercise_5_simplify(self):
        """
        Exercise 5: Simplify Expressions
        
        Simplify:
        a) Σᵢ₌₁ⁿ (3xᵢ + 2)
        b) Σᵢ₌₁ⁿ (xᵢ - x̄)  where x̄ = (1/n)Σxᵢ
        c) Σᵢ₌₁ⁿ (aᵢ - bᵢ)²
        d) Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ 1
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("Exercise 5 Solution:")
        
        print("\na) Σᵢ₌₁ⁿ (3xᵢ + 2)")
        print("   = Σᵢ₌₁ⁿ 3xᵢ + Σᵢ₌₁ⁿ 2")
        print("   = 3 Σᵢ₌₁ⁿ xᵢ + 2n")
        
        print("\nb) Σᵢ₌₁ⁿ (xᵢ - x̄)")
        print("   = Σᵢ₌₁ⁿ xᵢ - Σᵢ₌₁ⁿ x̄")
        print("   = Σᵢ₌₁ⁿ xᵢ - n·x̄")
        print("   = Σᵢ₌₁ⁿ xᵢ - n·(1/n)Σᵢ₌₁ⁿ xᵢ")
        print("   = Σᵢ₌₁ⁿ xᵢ - Σᵢ₌₁ⁿ xᵢ")
        print("   = 0")
        print("   (Deviations from mean sum to zero!)")
        
        print("\nc) Σᵢ₌₁ⁿ (aᵢ - bᵢ)²")
        print("   = Σᵢ₌₁ⁿ (aᵢ² - 2aᵢbᵢ + bᵢ²)")
        print("   = Σᵢ₌₁ⁿ aᵢ² - 2 Σᵢ₌₁ⁿ aᵢbᵢ + Σᵢ₌₁ⁿ bᵢ²")
        
        print("\nd) Σᵢ₌₁ⁿ Σⱼ₌₁ⁿ 1")
        print("   = Σᵢ₌₁ⁿ n")
        print("   = n × n")
        print("   = n²")
    
    def exercise_6_double_sum(self):
        """
        Exercise 6: Double Summation
        
        Compute:
        a) Σᵢ₌₁³ Σⱼ₌₁² (i + j)
        b) Σᵢ₌₁³ Σⱼ₌₁ⁱ j
        c) Σᵢ₌₁² Σⱼ₌₁² ij  (relate to matrix multiplication)
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solution:")
        
        print("\na) Σᵢ₌₁³ Σⱼ₌₁² (i + j)")
        total = 0
        for i in range(1, 4):
            inner = sum(i + j for j in range(1, 3))
            print(f"   i={i}: Σⱼ₌₁² ({i}+j) = {inner}")
            total += inner
        print(f"   Total = {total}")
        
        print("\nb) Σᵢ₌₁³ Σⱼ₌₁ⁱ j (triangular sum)")
        total = 0
        for i in range(1, 4):
            inner = sum(range(1, i + 1))
            print(f"   i={i}: Σⱼ₌₁^{i} j = {inner}")
            total += inner
        print(f"   Total = {total}")
        
        print("\nc) Σᵢ₌₁² Σⱼ₌₁² ij")
        total = 0
        print("   Expanding:")
        for i in range(1, 3):
            for j in range(1, 3):
                print(f"     i={i}, j={j}: {i}×{j} = {i*j}")
                total += i * j
        print(f"   Total = {total}")
        print("\n   Alternative: (Σi)(Σj) = (1+2)(1+2) = 3×3 = 9")
    
    def exercise_7_telescoping(self):
        """
        Exercise 7: Telescoping Sum
        
        Compute using telescoping:
        a) Σᵢ₌₁ⁿ [1/i - 1/(i+1)]
        b) Σᵢ₌₁ⁿ (aᵢ - aᵢ₋₁) given a₀ = 0
        c) Σᵢ₌₁¹⁰ [(i+1)² - i²]
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("Exercise 7 Solution:")
        
        print("\na) Σᵢ₌₁ⁿ [1/i - 1/(i+1)]")
        print("   This telescopes!")
        print("   = (1/1 - 1/2) + (1/2 - 1/3) + (1/3 - 1/4) + ... + (1/n - 1/(n+1))")
        print("   = 1 - 1/(n+1)")
        print("   = n/(n+1)")
        n = 10
        computed = sum(1/i - 1/(i+1) for i in range(1, n+1))
        formula = n/(n+1)
        print(f"   For n=10: computed = {computed:.6f}, formula = {formula:.6f}")
        
        print("\nb) Σᵢ₌₁ⁿ (aᵢ - aᵢ₋₁) with a₀ = 0")
        print("   = (a₁ - a₀) + (a₂ - a₁) + (a₃ - a₂) + ... + (aₙ - aₙ₋₁)")
        print("   = aₙ - a₀")
        print("   = aₙ")
        
        print("\nc) Σᵢ₌₁¹⁰ [(i+1)² - i²]")
        print("   Telescopes: = 11² - 1² = 121 - 1 = 120")
        computed = sum((i+1)**2 - i**2 for i in range(1, 11))
        print(f"   Computed: {computed}")
        
        print("\n   Alternative: (i+1)² - i² = 2i + 1")
        print("   Σᵢ₌₁¹⁰ (2i+1) = 2·55 + 10 = 120 ✓")
    
    def exercise_8_log_product(self):
        """
        Exercise 8: Product-Sum Conversion
        
        Convert between products and sums:
        a) Write log(Πᵢ₌₁ⁿ xᵢ) as a sum
        b) Write Πᵢ₌₁ⁿ eˣⁱ using a single exponential
        c) Compute Πᵢ₌₁⁵ (i+1)/i using telescoping
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("Exercise 8 Solution:")
        
        print("\na) log(Πᵢ₌₁ⁿ xᵢ)")
        print("   = log(x₁ · x₂ · ... · xₙ)")
        print("   = log(x₁) + log(x₂) + ... + log(xₙ)")
        print("   = Σᵢ₌₁ⁿ log(xᵢ)")
        
        # Verify numerically
        x = np.array([2, 3, 4, 5])
        print(f"\n   Verify with x = {x}:")
        print(f"   log(Π xᵢ) = log({np.prod(x)}) = {np.log(np.prod(x)):.4f}")
        print(f"   Σ log(xᵢ) = {np.sum(np.log(x)):.4f}")
        
        print("\nb) Πᵢ₌₁ⁿ eˣⁱ")
        print("   = eˣ¹ · eˣ² · ... · eˣⁿ")
        print("   = e^(x₁ + x₂ + ... + xₙ)")
        print("   = e^(Σᵢ₌₁ⁿ xᵢ)")
        
        print(f"\n   Verify with x = {x}:")
        print(f"   Π eˣⁱ = {np.prod(np.exp(x)):.4f}")
        print(f"   e^(Σxᵢ) = e^{np.sum(x)} = {np.exp(np.sum(x)):.4f}")
        
        print("\nc) Πᵢ₌₁⁵ (i+1)/i")
        print("   = (2/1)(3/2)(4/3)(5/4)(6/5)")
        print("   = 6/1 = 6")
        print("   (Telescoping product!)")
        computed = np.prod([(i+1)/i for i in range(1, 6)])
        print(f"   Computed: {computed}")
    
    # ==================== ADVANCED EXERCISES ====================
    
    def exercise_9_ml_formulas(self):
        """
        Exercise 9: ML Formulas
        
        Given data: x = [2, 4, 4, 4, 5, 5, 7, 9]
        
        Compute using summation notation:
        a) Mean: x̄ = (1/n) Σᵢ₌₁ⁿ xᵢ
        b) Variance: σ² = (1/n) Σᵢ₌₁ⁿ (xᵢ - x̄)²
        c) Alternative variance formula: σ² = (1/n) Σxᵢ² - x̄²
        d) Show both variance formulas are equivalent
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("Exercise 9 Solution:")
        
        x = np.array([2, 4, 4, 4, 5, 5, 7, 9])
        n = len(x)
        
        print(f"Data: x = {x}")
        print(f"n = {n}")
        
        # a) Mean
        mean = np.sum(x) / n
        print(f"\na) Mean:")
        print(f"   x̄ = (1/{n}) × {np.sum(x)} = {mean}")
        
        # b) Variance (definition)
        var_def = np.sum((x - mean)**2) / n
        print(f"\nb) Variance (definition):")
        print(f"   σ² = (1/n) Σ(xᵢ - x̄)²")
        print(f"   Deviations: {x - mean}")
        print(f"   Squared: {(x - mean)**2}")
        print(f"   σ² = {var_def}")
        
        # c) Alternative formula
        var_alt = np.sum(x**2) / n - mean**2
        print(f"\nc) Alternative formula:")
        print(f"   σ² = (1/n)Σxᵢ² - x̄²")
        print(f"   Σxᵢ² = {np.sum(x**2)}")
        print(f"   (1/{n})×{np.sum(x**2)} - {mean}² = {var_alt}")
        
        # d) Equivalence
        print(f"\nd) Equivalence proof:")
        print("   (1/n)Σ(xᵢ - x̄)²")
        print("   = (1/n)Σ(xᵢ² - 2xᵢx̄ + x̄²)")
        print("   = (1/n)Σxᵢ² - (2x̄/n)Σxᵢ + (1/n)·n·x̄²")
        print("   = (1/n)Σxᵢ² - 2x̄·x̄ + x̄²")
        print("   = (1/n)Σxᵢ² - 2x̄² + x̄²")
        print("   = (1/n)Σxᵢ² - x̄²")
        print(f"\n   Both give: {var_def:.4f} ✓")
    
    def exercise_10_change_of_index(self):
        """
        Exercise 10: Change of Index
        
        Transform:
        a) Convert Σᵢ₌₁ⁿ f(i) to a sum starting at i=0
        b) Convert Σᵢ₌₀^(n-1) aᵢ to a sum starting at i=1
        c) Show: Σᵢ₌₁ⁿ i = Σⱼ₌₀^(n-1) (n-j)
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("Exercise 10 Solution:")
        
        print("\na) Convert Σᵢ₌₁ⁿ f(i) to start at i=0")
        print("   Let j = i - 1, so i = j + 1")
        print("   When i = 1, j = 0; when i = n, j = n-1")
        print("   Σᵢ₌₁ⁿ f(i) = Σⱼ₌₀^(n-1) f(j+1)")
        
        # Verify
        n = 5
        f = lambda i: i**2
        sum1 = sum(f(i) for i in range(1, n+1))
        sum2 = sum(f(j+1) for j in range(0, n))
        print(f"\n   Verify with f(i) = i², n = 5:")
        print(f"   Σᵢ₌₁⁵ i² = {sum1}")
        print(f"   Σⱼ₌₀⁴ (j+1)² = {sum2}")
        
        print("\nb) Convert Σᵢ₌₀^(n-1) aᵢ to start at i=1")
        print("   Let j = i + 1, so i = j - 1")
        print("   When i = 0, j = 1; when i = n-1, j = n")
        print("   Σᵢ₌₀^(n-1) aᵢ = Σⱼ₌₁ⁿ aⱼ₋₁")
        
        print("\nc) Show: Σᵢ₌₁ⁿ i = Σⱼ₌₀^(n-1) (n-j)")
        print("   LHS: Σᵢ₌₁ⁿ i = 1 + 2 + ... + n")
        print("   RHS: Substitute i = n - j")
        print("        When j = 0, i = n; when j = n-1, i = 1")
        print("   Σⱼ₌₀^(n-1) (n-j) = n + (n-1) + ... + 1")
        print("   Both equal n(n+1)/2 ✓")
        
        n = 5
        lhs = sum(range(1, n+1))
        rhs = sum(n - j for j in range(n))
        print(f"\n   For n = 5: LHS = {lhs}, RHS = {rhs}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = SummationExercises()
    
    print("SUMMATION AND PRODUCT NOTATION EXERCISES")
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
