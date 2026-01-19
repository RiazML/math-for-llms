"""
Entropy - Exercises
===================
Practice problems for entropy concepts.
"""

import numpy as np
from collections import Counter


class EntropyExercises:
    """Exercises for entropy and information theory basics."""
    
    def exercise_1_compute_entropy(self):
        """
        Exercise 1: Compute Entropy
        
        Calculate entropy for various distributions.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Computing Entropy")
        print("=" * 60)
        
        def entropy(p):
            p = np.array(p)
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        
        print("H(X) = -Σ p(x) log₂ p(x)")
        
        # a) Fair 6-sided die
        p_die = np.ones(6) / 6
        H_die = entropy(p_die)
        print(f"\na) Fair 6-sided die:")
        print(f"   p = {p_die}")
        print(f"   H = log₂(6) = {H_die:.4f} bits")
        
        # b) Biased die
        p_biased = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        H_biased = entropy(p_biased)
        print(f"\nb) Biased die p = {p_biased}:")
        print(f"   H = {H_biased:.4f} bits")
        
        # c) Two fair coins
        p_coins = np.array([0.25, 0.25, 0.25, 0.25])  # HH, HT, TH, TT
        H_coins = entropy(p_coins)
        print(f"\nc) Two fair coins:")
        print(f"   H = log₂(4) = {H_coins:.4f} bits")
        
        # d) English letter frequencies (simplified)
        p_letters = np.array([0.12, 0.09, 0.08, 0.07, 0.07, 0.06, 0.06, 
                             0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.03,
                             0.03, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01])
        # Normalize
        p_letters = p_letters / p_letters.sum()
        H_letters = entropy(p_letters)
        print(f"\nd) Approximate English letter distribution:")
        print(f"   H ≈ {H_letters:.4f} bits")
        print(f"   Max possible (uniform 26): {np.log2(26):.4f} bits")
    
    def exercise_2_binary_entropy(self):
        """
        Exercise 2: Binary Entropy Properties
        
        Analyze the binary entropy function.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Binary Entropy Properties")
        print("=" * 60)
        
        def H_b(p):
            if p == 0 or p == 1:
                return 0
            return -p * np.log2(p) - (1-p) * np.log2(1-p)
        
        def H_b_derivative(p):
            """d/dp H_b(p) = log₂((1-p)/p)"""
            if p == 0 or p == 1:
                return float('inf') if p == 0 else float('-inf')
            return np.log2((1-p)/p)
        
        print("H_b(p) = -p log₂(p) - (1-p) log₂(1-p)")
        
        # a) Find maximum
        print("\na) Finding maximum:")
        print("   d/dp H_b = log₂((1-p)/p)")
        print("   Setting = 0: (1-p)/p = 1 → p = 0.5")
        print(f"   H_b(0.5) = {H_b(0.5)} bit")
        
        # b) Second derivative test
        print("\nb) Second derivative at p=0.5:")
        print("   d²/dp² H_b = -1/(p ln(2)) - 1/((1-p) ln(2))")
        d2 = -1/(0.5 * np.log(2)) - 1/(0.5 * np.log(2))
        print(f"   At p=0.5: {d2:.4f} < 0 (confirms maximum)")
        
        # c) Symmetry
        print("\nc) Symmetry: H_b(p) = H_b(1-p)")
        for p in [0.1, 0.3, 0.4]:
            print(f"   H_b({p}) = {H_b(p):.4f} = H_b({1-p}) = {H_b(1-p):.4f}")
    
    def exercise_3_joint_conditional(self):
        """
        Exercise 3: Joint and Conditional Entropy
        
        Work with joint distributions.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Joint and Conditional Entropy")
        print("=" * 60)
        
        # Joint distribution
        P_XY = np.array([
            [0.2, 0.1, 0.1],
            [0.1, 0.2, 0.3]
        ])
        
        print("Joint distribution P(X,Y):")
        print(P_XY)
        
        # Marginals
        P_X = P_XY.sum(axis=1)
        P_Y = P_XY.sum(axis=0)
        
        print(f"\nP(X) = {P_X}")
        print(f"P(Y) = {P_Y}")
        
        def entropy(p):
            p = p.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        
        H_X = entropy(P_X)
        H_Y = entropy(P_Y)
        H_XY = entropy(P_XY)
        
        print(f"\na) Marginal entropies:")
        print(f"   H(X) = {H_X:.4f} bits")
        print(f"   H(Y) = {H_Y:.4f} bits")
        
        print(f"\nb) Joint entropy:")
        print(f"   H(X,Y) = {H_XY:.4f} bits")
        
        # Conditional entropy H(Y|X)
        H_Y_given_X = 0
        for i in range(P_XY.shape[0]):
            if P_X[i] > 0:
                P_Y_given_xi = P_XY[i, :] / P_X[i]
                H_Y_given_X += P_X[i] * entropy(P_Y_given_xi)
        
        print(f"\nc) Conditional entropy:")
        print(f"   H(Y|X) = {H_Y_given_X:.4f} bits")
        
        # Chain rule verification
        print(f"\nd) Chain rule verification:")
        print(f"   H(X,Y) = {H_XY:.4f}")
        print(f"   H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f}")
        
        # Mutual information
        I_XY = H_Y - H_Y_given_X
        print(f"\ne) Mutual information:")
        print(f"   I(X;Y) = H(Y) - H(Y|X) = {I_XY:.4f} bits")
    
    def exercise_4_max_entropy(self):
        """
        Exercise 4: Maximum Entropy
        
        Find max entropy distributions under constraints.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Maximum Entropy")
        print("=" * 60)
        
        print("Problem: Find distribution maximizing entropy subject to:")
        print("  - Σ p(x) = 1")
        print("  - Additional constraints")
        
        print("\na) No constraints (discrete, n outcomes):")
        print("   Solution: Uniform p(x) = 1/n")
        print("   H_max = log(n)")
        
        for n in [2, 4, 8, 16]:
            print(f"   n={n}: H_max = {np.log2(n):.2f} bits")
        
        print("\nb) Fixed mean μ (continuous, x ≥ 0):")
        print("   Solution: Exponential p(x) = λe^{-λx}, λ = 1/μ")
        
        print("\nc) Fixed mean μ and variance σ² (continuous, x ∈ ℝ):")
        print("   Solution: Gaussian N(μ, σ²)")
        print(f"   h(X) = ½ log(2πeσ²)")
        
        for sigma in [1, 2, 5]:
            h = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
            print(f"   σ={sigma}: h = {h:.4f} bits")
        
        print("\nd) Why maximum entropy?")
        print("   - Least biased/most objective")
        print("   - Only uses available information")
        print("   - Leads to exponential family distributions")
    
    def exercise_5_information_gain(self):
        """
        Exercise 5: Information Gain Calculation
        
        Compute information gain for feature selection.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Information Gain")
        print("=" * 60)
        
        # Simple dataset
        # Features: A (binary), B (binary)
        # Label: Y
        
        data = {
            'A': [0, 0, 0, 0, 1, 1, 1, 1],
            'B': [0, 0, 1, 1, 0, 0, 1, 1],
            'Y': [0, 0, 0, 1, 0, 1, 1, 1]
        }
        
        Y = np.array(data['Y'])
        A = np.array(data['A'])
        B = np.array(data['B'])
        
        def entropy(labels):
            n = len(labels)
            counts = Counter(labels)
            probs = np.array([c/n for c in counts.values()])
            return -np.sum(probs * np.log2(probs + 1e-10))
        
        H_Y = entropy(Y)
        print(f"Dataset: {len(Y)} samples")
        print(f"Y distribution: {Counter(Y)}")
        print(f"H(Y) = {H_Y:.4f} bits")
        
        # Information gain for A
        print("\nInformation Gain for feature A:")
        
        for val in [0, 1]:
            mask = A == val
            Y_subset = Y[mask]
            print(f"  A={val}: {len(Y_subset)} samples, Y={Counter(Y_subset)}")
            print(f"         H(Y|A={val}) = {entropy(Y_subset):.4f}")
        
        H_Y_given_A = (np.sum(A==0)/len(A)) * entropy(Y[A==0]) + \
                      (np.sum(A==1)/len(A)) * entropy(Y[A==1])
        IG_A = H_Y - H_Y_given_A
        
        print(f"  H(Y|A) = {H_Y_given_A:.4f}")
        print(f"  IG(A) = H(Y) - H(Y|A) = {IG_A:.4f} bits")
        
        # Information gain for B
        print("\nInformation Gain for feature B:")
        
        for val in [0, 1]:
            mask = B == val
            Y_subset = Y[mask]
            print(f"  B={val}: {len(Y_subset)} samples, Y={Counter(Y_subset)}")
        
        H_Y_given_B = (np.sum(B==0)/len(B)) * entropy(Y[B==0]) + \
                      (np.sum(B==1)/len(B)) * entropy(Y[B==1])
        IG_B = H_Y - H_Y_given_B
        
        print(f"  H(Y|B) = {H_Y_given_B:.4f}")
        print(f"  IG(B) = {IG_B:.4f} bits")
        
        print(f"\nBest feature: {'A' if IG_A > IG_B else 'B'}")
    
    def exercise_6_differential_entropy(self):
        """
        Exercise 6: Differential Entropy
        
        Calculate differential entropy for continuous distributions.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Differential Entropy")
        print("=" * 60)
        
        print("h(X) = -∫ f(x) log f(x) dx")
        
        print("\na) Uniform[0, a]:")
        print("   f(x) = 1/a for x ∈ [0, a]")
        print("   h(X) = -∫₀ᵃ (1/a) log(1/a) dx")
        print("        = log(a)")
        
        for a in [1, 2, 5, 10]:
            print(f"   a={a}: h = {np.log2(a):.4f} bits")
        
        print("\nb) Exponential(λ):")
        print("   f(x) = λe^{-λx} for x ≥ 0")
        print("   h(X) = 1 - log(λ) = 1 + log(1/λ)")
        
        for lam in [0.5, 1, 2]:
            h = 1 - np.log2(lam)
            print(f"   λ={lam}: h = {h:.4f} bits")
        
        print("\nc) Gaussian N(μ, σ²):")
        print("   h(X) = ½ log(2πeσ²)")
        
        for sigma in [0.5, 1, 2, 5]:
            h = 0.5 * np.log2(2 * np.pi * np.e * sigma**2)
            print(f"   σ={sigma}: h = {h:.4f} bits")
        
        print("\nd) Key difference from discrete:")
        print("   - Can be negative (e.g., Uniform[0, 0.5])")
        h_small = np.log2(0.5)
        print(f"   - Uniform[0, 0.5]: h = {h_small:.4f} bits")
    
    def exercise_7_entropy_estimation(self):
        """
        Exercise 7: Entropy from Samples
        
        Estimate entropy from empirical data.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Entropy Estimation")
        print("=" * 60)
        
        np.random.seed(42)
        
        # True distribution
        true_p = np.array([0.5, 0.3, 0.15, 0.05])
        true_H = -np.sum(true_p * np.log2(true_p))
        
        print(f"True distribution: {true_p}")
        print(f"True entropy: {true_H:.4f} bits")
        
        def plugin_entropy(samples, n_classes):
            counts = Counter(samples)
            probs = np.array([counts.get(i, 0) for i in range(n_classes)])
            probs = probs / probs.sum()
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        
        def miller_madow(samples, n_classes):
            H_plugin = plugin_entropy(samples, n_classes)
            counts = Counter(samples)
            m = len([c for c in counts.values() if c > 0])
            return H_plugin + (m - 1) / (2 * len(samples) * np.log(2))
        
        print(f"\n{'n':>6} {'Plugin':>12} {'M-M Corrected':>15} {'Bias':>12}")
        print("-" * 50)
        
        for n in [10, 50, 100, 500, 1000]:
            samples = np.random.choice(4, n, p=true_p)
            H_plugin = plugin_entropy(samples, 4)
            H_mm = miller_madow(samples, 4)
            bias = H_plugin - true_H
            
            print(f"{n:>6} {H_plugin:>12.4f} {H_mm:>15.4f} {bias:>12.4f}")
        
        print("\nPlugin estimator is biased low (underestimates entropy)")
    
    def exercise_8_chain_rule(self):
        """
        Exercise 8: Chain Rule Application
        
        Apply chain rule to multivariate distributions.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Chain Rule")
        print("=" * 60)
        
        print("Chain rule: H(X₁, X₂, ..., Xₙ) = Σᵢ H(Xᵢ | X₁, ..., Xᵢ₋₁)")
        
        # Example: 3 binary variables
        # X₁, X₂, X₃ with specific dependencies
        
        # Joint distribution as 2x2x2 tensor
        P = np.zeros((2, 2, 2))
        P[0, 0, 0] = 0.2
        P[0, 0, 1] = 0.1
        P[0, 1, 0] = 0.1
        P[0, 1, 1] = 0.1
        P[1, 0, 0] = 0.1
        P[1, 0, 1] = 0.1
        P[1, 1, 0] = 0.1
        P[1, 1, 1] = 0.2
        
        def entropy(p):
            p = p.flatten()
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        
        # Joint entropy
        H_123 = entropy(P)
        print(f"\nH(X₁, X₂, X₃) = {H_123:.4f} bits")
        
        # Marginals
        P_1 = P.sum(axis=(1, 2))
        P_2 = P.sum(axis=(0, 2))
        P_3 = P.sum(axis=(0, 1))
        
        H_1 = entropy(P_1)
        print(f"\nH(X₁) = {H_1:.4f} bits")
        
        # H(X₂ | X₁)
        P_12 = P.sum(axis=2)
        H_12 = entropy(P_12)
        H_2_given_1 = H_12 - H_1
        print(f"H(X₂ | X₁) = H(X₁,X₂) - H(X₁) = {H_2_given_1:.4f} bits")
        
        # H(X₃ | X₁, X₂)
        H_3_given_12 = H_123 - H_12
        print(f"H(X₃ | X₁,X₂) = H(X₁,X₂,X₃) - H(X₁,X₂) = {H_3_given_12:.4f} bits")
        
        # Verify chain rule
        chain_sum = H_1 + H_2_given_1 + H_3_given_12
        print(f"\nVerification:")
        print(f"H(X₁) + H(X₂|X₁) + H(X₃|X₁,X₂) = {chain_sum:.4f}")
        print(f"H(X₁, X₂, X₃) = {H_123:.4f}")
    
    def exercise_9_entropy_bounds(self):
        """
        Exercise 9: Entropy Bounds
        
        Prove and verify entropy inequalities.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Entropy Bounds")
        print("=" * 60)
        
        print("Key inequalities:")
        print("\n1. H(X) ≥ 0 (non-negativity)")
        print("   Proof: -p log p ≥ 0 for p ∈ [0,1]")
        
        print("\n2. H(X) ≤ log|X| (maximum entropy)")
        print("   Equality when X is uniform")
        
        n = 8
        p_uniform = np.ones(n) / n
        p_peaked = np.array([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01])
        
        def entropy(p):
            p = p[p > 0]
            return -np.sum(p * np.log2(p))
        
        print(f"\n   For n={n} outcomes:")
        print(f"   log({n}) = {np.log2(n):.4f} bits")
        print(f"   H(uniform) = {entropy(p_uniform):.4f} bits")
        print(f"   H(peaked) = {entropy(p_peaked):.4f} bits")
        
        print("\n3. H(X,Y) ≤ H(X) + H(Y) (subadditivity)")
        print("   Equality iff X, Y independent")
        
        # Independent
        P_XY_ind = np.outer([0.3, 0.7], [0.4, 0.6])
        P_X = P_XY_ind.sum(axis=1)
        P_Y = P_XY_ind.sum(axis=0)
        
        H_XY = entropy(P_XY_ind)
        H_X = entropy(P_X)
        H_Y = entropy(P_Y)
        
        print(f"\n   Independent X, Y:")
        print(f"   H(X,Y) = {H_XY:.4f}")
        print(f"   H(X) + H(Y) = {H_X + H_Y:.4f}")
        
        # Dependent
        P_XY_dep = np.array([[0.4, 0.1], [0.1, 0.4]])
        
        H_XY_dep = entropy(P_XY_dep)
        H_X_dep = entropy(P_XY_dep.sum(axis=1))
        H_Y_dep = entropy(P_XY_dep.sum(axis=0))
        
        print(f"\n   Dependent X, Y:")
        print(f"   H(X,Y) = {H_XY_dep:.4f}")
        print(f"   H(X) + H(Y) = {H_X_dep + H_Y_dep:.4f}")
        
        print("\n4. H(Y|X) ≤ H(Y) (conditioning reduces entropy)")
        print(f"   H(Y|X) = H(X,Y) - H(X) = {H_XY_dep - H_X_dep:.4f}")
        print(f"   H(Y) = {H_Y_dep:.4f}")
    
    def exercise_10_cross_entropy(self):
        """
        Exercise 10: Cross-Entropy and Classification
        
        Understand cross-entropy loss.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Cross-Entropy Loss")
        print("=" * 60)
        
        print("Cross-entropy: H(p, q) = -Σ p(x) log q(x)")
        print("               = H(p) + D_KL(p || q)")
        
        print("\nIn classification:")
        print("  p = true distribution (one-hot)")
        print("  q = predicted distribution")
        
        # Binary classification
        print("\na) Binary cross-entropy:")
        print("   L = -[y log(q) + (1-y) log(1-q)]")
        
        y = 1  # True label
        print(f"\n   True label y = {y}")
        print(f"   {'q':>8} {'Loss':>12} {'-log(q)':>12}")
        print("   " + "-" * 35)
        
        for q in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            loss = -y * np.log(q) - (1-y) * np.log(1-q)
            print(f"   {q:>8.2f} {loss:>12.4f} {-np.log(q):>12.4f}")
        
        # Multiclass
        print("\nb) Multiclass cross-entropy:")
        print("   L = -Σₖ yₖ log(qₖ) = -log(q_true_class)")
        
        y_true = np.array([0, 0, 1])  # Class 2
        
        predictions = [
            [0.1, 0.1, 0.8],   # Good
            [0.2, 0.3, 0.5],   # Okay
            [0.33, 0.33, 0.34], # Bad
        ]
        
        print(f"\n   True class: 2")
        print(f"   {'Prediction':>25} {'Loss':>12}")
        print("   " + "-" * 40)
        
        for q in predictions:
            q = np.array(q)
            loss = -np.sum(y_true * np.log(q + 1e-10))
            print(f"   {str(q):>25} {loss:>12.4f}")
        
        print("\nc) Relationship to entropy:")
        print("   H(p, q) = H(p) + D_KL(p || q)")
        print("   For one-hot p: H(p) = 0")
        print("   So: H(p, q) = D_KL(p || q) = -log(q_true_class)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = EntropyExercises()
    
    print("ENTROPY EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    exercises.solution_2()
    exercises.solution_3()
    exercises.solution_4()
    exercises.solution_5()
    exercises.solution_6()
    exercises.solution_7()
    exercises.solution_8()
    exercises.solution_9()
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
