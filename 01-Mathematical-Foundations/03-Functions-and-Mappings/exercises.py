"""
Functions and Mappings - Exercises
==================================
Practice problems for function concepts.
"""

import numpy as np


class FunctionExercises:
    """Exercises for functions and mappings."""
    
    # ==================== BASIC EXERCISES ====================
    
    def exercise_1_function_evaluation(self):
        """
        Exercise 1: Function Evaluation
        
        Given:
        f(x) = x² - 3x + 2
        g(x) = (x + 1)/(x - 1)
        h(x) = √(x - 1)
        
        Evaluate:
        a) f(0), f(1), f(2)
        b) g(2), g(0), g(-1)
        c) h(1), h(2), h(5)
        d) Find the domain of each function
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution:")
        
        f = lambda x: x**2 - 3*x + 2
        g = lambda x: (x + 1)/(x - 1)
        h = lambda x: np.sqrt(x - 1)
        
        print("\na) f(x) = x² - 3x + 2")
        print(f"   f(0) = 0 - 0 + 2 = {f(0)}")
        print(f"   f(1) = 1 - 3 + 2 = {f(1)}")
        print(f"   f(2) = 4 - 6 + 2 = {f(2)}")
        
        print("\nb) g(x) = (x+1)/(x-1)")
        print(f"   g(2) = 3/1 = {g(2)}")
        print(f"   g(0) = 1/(-1) = {g(0)}")
        print(f"   g(-1) = 0/(-2) = {g(-1)}")
        
        print("\nc) h(x) = √(x-1)")
        print(f"   h(1) = √0 = {h(1)}")
        print(f"   h(2) = √1 = {h(2)}")
        print(f"   h(5) = √4 = {h(5)}")
        
        print("\nd) Domains:")
        print("   f(x): ℝ (all real numbers)")
        print("   g(x): ℝ \\ {1} (x ≠ 1)")
        print("   h(x): [1, ∞) (x ≥ 1)")
    
    def exercise_2_injective_surjective(self):
        """
        Exercise 2: Determine Function Properties
        
        For each function, determine if it is:
        - Injective (one-to-one)
        - Surjective (onto)
        - Bijective
        
        a) f: ℝ → ℝ, f(x) = 3x + 5
        b) f: ℝ → ℝ, f(x) = x²
        c) f: [0,∞) → [0,∞), f(x) = x²
        d) f: ℝ → ℝ, f(x) = eˣ
        e) f: ℝ → (0,∞), f(x) = eˣ
        f) f: ℝ → [-1,1], f(x) = sin(x)
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("Exercise 2 Solution:")
        
        print("\na) f: ℝ → ℝ, f(x) = 3x + 5")
        print("   Injective? YES (linear with non-zero slope)")
        print("   Surjective? YES (every y = 3x+5 has solution x = (y-5)/3)")
        print("   BIJECTIVE ✓")
        
        print("\nb) f: ℝ → ℝ, f(x) = x²")
        print("   Injective? NO (f(-2) = f(2) = 4)")
        print("   Surjective? NO (no x gives f(x) = -1)")
        print("   NOT bijective")
        
        print("\nc) f: [0,∞) → [0,∞), f(x) = x²")
        print("   Injective? YES (x₁² = x₂² and x₁,x₂ ≥ 0 ⟹ x₁ = x₂)")
        print("   Surjective? YES (every y ≥ 0 has x = √y)")
        print("   BIJECTIVE ✓")
        
        print("\nd) f: ℝ → ℝ, f(x) = eˣ")
        print("   Injective? YES (strictly increasing)")
        print("   Surjective? NO (eˣ > 0 always, can't hit negatives)")
        print("   NOT bijective")
        
        print("\ne) f: ℝ → (0,∞), f(x) = eˣ")
        print("   Injective? YES")
        print("   Surjective? YES (every y > 0 has x = ln(y))")
        print("   BIJECTIVE ✓")
        
        print("\nf) f: ℝ → [-1,1], f(x) = sin(x)")
        print("   Injective? NO (sin(0) = sin(2π) = 0)")
        print("   Surjective? YES (sin covers all of [-1,1])")
        print("   NOT bijective")
    
    def exercise_3_composition(self):
        """
        Exercise 3: Function Composition
        
        Given:
        f(x) = 2x + 1
        g(x) = x²
        h(x) = 1/x
        
        Find:
        a) (g ∘ f)(x)
        b) (f ∘ g)(x)
        c) (g ∘ f)(3)
        d) (f ∘ g)(3)
        e) (h ∘ g ∘ f)(x)
        f) (f ∘ f)(x)
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("Exercise 3 Solution:")
        
        f = lambda x: 2*x + 1
        g = lambda x: x**2
        h = lambda x: 1/x
        
        print("\nf(x) = 2x + 1")
        print("g(x) = x²")
        print("h(x) = 1/x")
        
        print("\na) (g ∘ f)(x) = g(f(x)) = g(2x + 1) = (2x + 1)²")
        gf = lambda x: g(f(x))
        print(f"   = 4x² + 4x + 1")
        
        print("\nb) (f ∘ g)(x) = f(g(x)) = f(x²) = 2x² + 1")
        fg = lambda x: f(g(x))
        
        print(f"\nc) (g ∘ f)(3) = g(f(3)) = g(7) = 49")
        print(f"   Computed: {gf(3)}")
        
        print(f"\nd) (f ∘ g)(3) = f(g(3)) = f(9) = 19")
        print(f"   Computed: {fg(3)}")
        
        print("\ne) (h ∘ g ∘ f)(x) = h(g(f(x))) = h((2x+1)²) = 1/(2x+1)²")
        hgf = lambda x: h(g(f(x)))
        print(f"   At x=1: {hgf(1):.4f} = 1/9")
        
        print("\nf) (f ∘ f)(x) = f(f(x)) = f(2x + 1) = 2(2x + 1) + 1 = 4x + 3")
        ff = lambda x: f(f(x))
        print(f"   At x=2: {ff(2)} = 4(2) + 3")
    
    def exercise_4_inverse(self):
        """
        Exercise 4: Finding Inverses
        
        Find the inverse of each function (if it exists):
        
        a) f(x) = 5x - 3
        b) f(x) = (x + 2)/(x - 1)
        c) f(x) = x³
        d) f(x) = eˣ⁺¹
        e) f(x) = ln(2x)
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("Exercise 4 Solution:")
        
        print("\na) f(x) = 5x - 3")
        print("   y = 5x - 3")
        print("   5x = y + 3")
        print("   x = (y + 3)/5")
        print("   f⁻¹(x) = (x + 3)/5")
        
        # Verify
        f = lambda x: 5*x - 3
        f_inv = lambda x: (x + 3)/5
        x = 7
        print(f"   Verify: f(7) = {f(7)}, f⁻¹({f(7)}) = {f_inv(f(7))}")
        
        print("\nb) f(x) = (x + 2)/(x - 1)")
        print("   y = (x + 2)/(x - 1)")
        print("   y(x - 1) = x + 2")
        print("   yx - y = x + 2")
        print("   yx - x = y + 2")
        print("   x(y - 1) = y + 2")
        print("   x = (y + 2)/(y - 1)")
        print("   f⁻¹(x) = (x + 2)/(x - 1)")
        print("   Note: f = f⁻¹ (self-inverse!)")
        
        print("\nc) f(x) = x³")
        print("   y = x³")
        print("   x = ∛y")
        print("   f⁻¹(x) = ∛x = x^(1/3)")
        
        print("\nd) f(x) = e^(x+1)")
        print("   y = e^(x+1)")
        print("   ln(y) = x + 1")
        print("   x = ln(y) - 1")
        print("   f⁻¹(x) = ln(x) - 1")
        
        # Verify
        f = lambda x: np.exp(x + 1)
        f_inv = lambda x: np.log(x) - 1
        x = 2
        print(f"   Verify: f(2) = {f(2):.4f}, f⁻¹({f(2):.4f}) = {f_inv(f(2)):.4f}")
        
        print("\ne) f(x) = ln(2x)")
        print("   y = ln(2x)")
        print("   e^y = 2x")
        print("   x = e^y/2")
        print("   f⁻¹(x) = eˣ/2")
    
    # ==================== INTERMEDIATE EXERCISES ====================
    
    def exercise_5_bijection_proof(self):
        """
        Exercise 5: Prove Bijection
        
        Prove that f: ℝ → ℝ, f(x) = 2x + 7 is a bijection.
        Find its inverse.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("Exercise 5 Solution:")
        print("\nProve f(x) = 2x + 7 is a bijection:")
        
        print("\n1. Prove Injective:")
        print("   Assume f(x₁) = f(x₂)")
        print("   2x₁ + 7 = 2x₂ + 7")
        print("   2x₁ = 2x₂")
        print("   x₁ = x₂")
        print("   Therefore f is injective ✓")
        
        print("\n2. Prove Surjective:")
        print("   Let y ∈ ℝ be arbitrary")
        print("   Find x such that f(x) = y")
        print("   2x + 7 = y")
        print("   x = (y - 7)/2")
        print("   Since (y-7)/2 ∈ ℝ for all y ∈ ℝ, f is surjective ✓")
        
        print("\n3. Since f is injective and surjective, f is BIJECTIVE")
        
        print("\n4. Inverse:")
        print("   f⁻¹(x) = (x - 7)/2")
        
        # Verify
        f = lambda x: 2*x + 7
        f_inv = lambda x: (x - 7)/2
        print(f"\n   Verification:")
        print(f"   f(5) = {f(5)}")
        print(f"   f⁻¹(f(5)) = f⁻¹({f(5)}) = {f_inv(f(5))}")
        print(f"   f⁻¹(10) = {f_inv(10)}")
        print(f"   f(f⁻¹(10)) = f({f_inv(10)}) = {f(f_inv(10))}")
    
    def exercise_6_composition_properties(self):
        """
        Exercise 6: Composition Properties
        
        Prove or disprove:
        a) If f and g are both injective, then g ∘ f is injective
        b) If f and g are both surjective, then g ∘ f is surjective
        c) If g ∘ f is injective, then f is injective
        d) If g ∘ f is surjective, then g is surjective
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solution:")
        
        print("\na) f, g injective ⟹ g ∘ f injective")
        print("   Proof: Assume (g ∘ f)(x₁) = (g ∘ f)(x₂)")
        print("   Then g(f(x₁)) = g(f(x₂))")
        print("   Since g is injective: f(x₁) = f(x₂)")
        print("   Since f is injective: x₁ = x₂")
        print("   TRUE ✓")
        
        print("\nb) f, g surjective ⟹ g ∘ f surjective")
        print("   Proof: Let z be in codomain of g ∘ f")
        print("   Since g is surjective: ∃y such that g(y) = z")
        print("   Since f is surjective: ∃x such that f(x) = y")
        print("   Therefore (g ∘ f)(x) = g(f(x)) = g(y) = z")
        print("   TRUE ✓")
        
        print("\nc) g ∘ f injective ⟹ f injective")
        print("   Proof: Assume f(x₁) = f(x₂)")
        print("   Then g(f(x₁)) = g(f(x₂))")
        print("   So (g ∘ f)(x₁) = (g ∘ f)(x₂)")
        print("   Since g ∘ f is injective: x₁ = x₂")
        print("   TRUE ✓")
        
        print("\nd) g ∘ f surjective ⟹ g surjective")
        print("   Proof: Let z be in codomain of g")
        print("   Since g ∘ f is surjective: ∃x such that (g ∘ f)(x) = z")
        print("   So g(f(x)) = z")
        print("   Let y = f(x), then g(y) = z")
        print("   TRUE ✓")
    
    def exercise_7_activation_analysis(self):
        """
        Exercise 7: Activation Function Analysis
        
        For each activation function, determine:
        - Domain and range
        - Is it injective? surjective? bijective?
        - Find the inverse if it exists
        
        a) Sigmoid: σ(x) = 1/(1 + e⁻ˣ)
        b) ReLU: f(x) = max(0, x)
        c) Tanh: f(x) = tanh(x)
        d) Leaky ReLU: f(x) = max(0.1x, x)
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("Exercise 7 Solution:")
        
        print("\na) Sigmoid: σ(x) = 1/(1 + e⁻ˣ)")
        print("   Domain: ℝ")
        print("   Range: (0, 1)")
        print("   Injective? YES (strictly increasing)")
        print("   Surjective onto ℝ? NO")
        print("   Surjective onto (0,1)? YES")
        print("   Bijective onto (0,1)? YES")
        print("\n   Inverse (logit function):")
        print("   y = 1/(1 + e⁻ˣ)")
        print("   1 + e⁻ˣ = 1/y")
        print("   e⁻ˣ = 1/y - 1 = (1-y)/y")
        print("   -x = ln((1-y)/y)")
        print("   x = ln(y/(1-y))")
        print("   σ⁻¹(y) = ln(y/(1-y))")
        
        # Verify
        x = 2
        sigma = 1/(1 + np.exp(-x))
        sigma_inv = np.log(sigma/(1-sigma))
        print(f"\n   Verify: σ(2) = {sigma:.4f}")
        print(f"   σ⁻¹({sigma:.4f}) = {sigma_inv:.4f}")
        
        print("\nb) ReLU: f(x) = max(0, x)")
        print("   Domain: ℝ")
        print("   Range: [0, ∞)")
        print("   Injective? NO (all x < 0 map to 0)")
        print("   Inverse: Does not exist")
        
        print("\nc) Tanh: f(x) = tanh(x)")
        print("   Domain: ℝ")
        print("   Range: (-1, 1)")
        print("   Injective? YES (strictly increasing)")
        print("   Bijective onto (-1,1)? YES")
        print("   Inverse: tanh⁻¹(y) = arctanh(y) = (1/2)ln((1+y)/(1-y))")
        
        print("\nd) Leaky ReLU: f(x) = max(0.1x, x)")
        print("   = x if x ≥ 0")
        print("   = 0.1x if x < 0")
        print("   Domain: ℝ")
        print("   Range: ℝ")
        print("   Injective? YES (different slopes, but still one-to-one)")
        print("   Surjective? YES")
        print("   Bijective? YES")
        print("   Inverse:")
        print("   f⁻¹(y) = y if y ≥ 0")
        print("   f⁻¹(y) = 10y if y < 0")
    
    # ==================== ADVANCED EXERCISES ====================
    
    def exercise_8_fixed_points(self):
        """
        Exercise 8: Fixed Points
        
        A fixed point of f is a value x where f(x) = x.
        
        Find all fixed points of:
        a) f(x) = x²
        b) f(x) = cos(x) (approximately)
        c) f(x) = 3x - 2
        d) σ(x) = 1/(1 + e⁻ˣ)
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("Exercise 8 Solution:")
        
        print("\na) f(x) = x²")
        print("   Solve: x² = x")
        print("   x² - x = 0")
        print("   x(x - 1) = 0")
        print("   Fixed points: x = 0, x = 1")
        
        print("\nb) f(x) = cos(x)")
        print("   Solve: cos(x) = x (numerically)")
        # Fixed point iteration
        x = 0.5
        for _ in range(50):
            x = np.cos(x)
        print(f"   Fixed point: x ≈ {x:.6f}")
        print(f"   Verify: cos({x:.6f}) = {np.cos(x):.6f}")
        
        print("\nc) f(x) = 3x - 2")
        print("   Solve: 3x - 2 = x")
        print("   2x = 2")
        print("   x = 1")
        print("   Fixed point: x = 1")
        
        print("\nd) σ(x) = 1/(1 + e⁻ˣ)")
        print("   Solve: 1/(1 + e⁻ˣ) = x")
        # Numerical solution
        from scipy.optimize import fsolve
        sigma = lambda x: 1/(1 + np.exp(-x))
        f = lambda x: sigma(x) - x
        x0 = fsolve(f, 0.5)[0]
        print(f"   Fixed point: x ≈ {x0:.6f}")
        print(f"   Verify: σ({x0:.6f}) = {sigma(x0):.6f}")
    
    def exercise_9_function_iteration(self):
        """
        Exercise 9: Function Iteration
        
        For f(x) = x/2 + 1:
        a) Compute f¹(x) = f(x), f²(x) = f(f(x)), f³(x), f⁴(x)
        b) Find a pattern and conjecture fⁿ(x)
        c) Find lim(n→∞) fⁿ(x) for any starting x
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("Exercise 9 Solution:")
        
        f = lambda x: x/2 + 1
        
        print("\nf(x) = x/2 + 1")
        
        print("\na) Iterations:")
        x = 8  # Starting value
        print(f"   Starting x = {x}")
        print(f"   f¹(x) = x/2 + 1 = {f(x)}")
        print(f"   f²(x) = f(f(x)) = (x/2 + 1)/2 + 1 = x/4 + 1/2 + 1 = x/4 + 3/2")
        print(f"         = {f(f(x))}")
        print(f"   f³(x) = x/8 + 3/4 + 1 = x/8 + 7/4 = {f(f(f(x)))}")
        print(f"   f⁴(x) = x/16 + 7/8 + 1 = x/16 + 15/8 = {f(f(f(f(x))))}")
        
        print("\nb) Pattern:")
        print("   fⁿ(x) = x/2ⁿ + (2ⁿ - 1)/2^(n-1)")
        print("         = x/2ⁿ + 2 - 1/2^(n-1)")
        print("         = x/2ⁿ + 2(1 - 1/2ⁿ)")
        
        print("\nc) Limit as n → ∞:")
        print("   lim fⁿ(x) = lim [x/2ⁿ + 2(1 - 1/2ⁿ)]")
        print("             = 0 + 2(1 - 0)")
        print("             = 2")
        print("   The fixed point is x = 2 (verify: f(2) = 2/2 + 1 = 2)")
        
        # Verify numerically
        x = 100
        for i in range(20):
            x = f(x)
        print(f"\n   Verification: After 20 iterations from x=100: {x:.6f}")
    
    def exercise_10_ml_function_chain(self):
        """
        Exercise 10: ML Function Chain
        
        Consider a simple neural network layer:
        h(x) = σ(Wx + b)
        
        where σ is sigmoid, W = [[0.5, 0.3], [-0.2, 0.4]], b = [0.1, -0.1]
        
        a) Compute h([1, 2])
        b) What is the domain and range of h?
        c) Is h injective? Why or why not?
        d) Compute the Jacobian of h at [1, 2]
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("Exercise 10 Solution:")
        
        W = np.array([[0.5, 0.3],
                      [-0.2, 0.4]])
        b = np.array([0.1, -0.1])
        
        sigma = lambda x: 1/(1 + np.exp(-x))
        h = lambda x: sigma(W @ x + b)
        
        print("\nh(x) = σ(Wx + b)")
        print(f"W = \n{W}")
        print(f"b = {b}")
        
        # a)
        x = np.array([1, 2])
        z = W @ x + b
        output = h(x)
        
        print(f"\na) h([1, 2]):")
        print(f"   z = Wx + b = {z}")
        print(f"   h(x) = σ(z) = {output}")
        
        # b)
        print("\nb) Domain and Range:")
        print("   Domain: ℝ² (any 2D vector)")
        print("   Range: (0,1)² (each component in (0,1))")
        
        # c)
        print("\nc) Is h injective?")
        print("   No, h is NOT injective.")
        print("   Reason: The matrix W maps ℝ² to ℝ², but:")
        print(f"   - det(W) = {np.linalg.det(W):.4f}")
        if np.abs(np.linalg.det(W)) < 1e-10:
            print("   - det(W) ≈ 0, so W is not invertible")
            print("   - Multiple inputs can give same output")
        else:
            print("   - W is invertible, but sigmoid compresses values")
            print("   - For very large/small inputs, outputs are nearly same")
        
        # d) Jacobian
        print("\nd) Jacobian of h at [1, 2]:")
        # h(x) = σ(Wx + b)
        # ∂h/∂x = diag(σ'(z)) @ W
        # where σ'(z) = σ(z)(1 - σ(z))
        
        sigma_deriv = output * (1 - output)  # σ'(z) at z = Wx + b
        jacobian = np.diag(sigma_deriv) @ W
        
        print("   J = diag(σ'(z)) @ W")
        print(f"   σ'(z) = σ(z)(1-σ(z)) = {sigma_deriv}")
        print(f"   J = \n{jacobian}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = FunctionExercises()
    
    print("FUNCTIONS AND MAPPINGS EXERCISES")
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
