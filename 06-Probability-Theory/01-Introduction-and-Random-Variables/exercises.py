"""
Introduction to Probability and Random Variables - Exercises
=============================================================
Practice problems for probability fundamentals.
"""

import numpy as np
from scipy import stats


class ProbabilityExercises:
    """Exercises for probability and random variables."""
    
    def exercise_1_union_rule(self):
        """
        Exercise 1: Prove the Union Rule
        
        Prove: P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Union Rule Proof")
        
        print("\nProof using set decomposition:")
        print("  A ∪ B can be written as disjoint union:")
        print("  A ∪ B = A ∪ (B ∩ Aᶜ)")
        
        print("\n  Since A and (B ∩ Aᶜ) are disjoint:")
        print("  P(A ∪ B) = P(A) + P(B ∩ Aᶜ)  ... (1)")
        
        print("\n  Also, B = (B ∩ A) ∪ (B ∩ Aᶜ) (disjoint)")
        print("  P(B) = P(B ∩ A) + P(B ∩ Aᶜ)")
        print("  Therefore: P(B ∩ Aᶜ) = P(B) - P(A ∩ B)  ... (2)")
        
        print("\n  Substituting (2) into (1):")
        print("  P(A ∪ B) = P(A) + P(B) - P(A ∩ B)  □")
        
        # Numerical verification
        print("\n--- Numerical Verification ---")
        np.random.seed(42)
        n = 100000
        
        # Events defined by random thresholds
        U = np.random.uniform(0, 1, n)
        A = U < 0.6  # P(A) ≈ 0.6
        B = U < 0.5  # P(B) ≈ 0.5
        
        P_A = A.mean()
        P_B = B.mean()
        P_A_and_B = (A & B).mean()
        P_A_or_B = (A | B).mean()
        
        print(f"  P(A) = {P_A:.4f}")
        print(f"  P(B) = {P_B:.4f}")
        print(f"  P(A ∩ B) = {P_A_and_B:.4f}")
        print(f"  P(A ∪ B) measured = {P_A_or_B:.4f}")
        print(f"  P(A) + P(B) - P(A ∩ B) = {P_A + P_B - P_A_and_B:.4f}")
    
    def exercise_2_binomial_moments(self):
        """
        Exercise 2: Binomial Moments
        
        Derive E[X] and Var(X) for X ~ Binomial(n, p).
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Binomial Moments")
        
        print("\nX ~ Binomial(n, p)")
        print("X = Σᵢ Xᵢ where Xᵢ ~ Bernoulli(p) are independent")
        
        print("\n--- E[X] ---")
        print("  E[Xᵢ] = 0·(1-p) + 1·p = p")
        print("  E[X] = E[Σᵢ Xᵢ] = Σᵢ E[Xᵢ] = np")
        
        print("\n--- Var(X) ---")
        print("  Var(Xᵢ) = E[Xᵢ²] - E[Xᵢ]²")
        print("          = p - p² = p(1-p)")
        print("  Var(X) = Var(Σᵢ Xᵢ)")
        print("         = Σᵢ Var(Xᵢ)  (independence)")
        print("         = np(1-p)")
        
        # Verify
        n, p = 20, 0.3
        print(f"\n--- Verification: n={n}, p={p} ---")
        print(f"  Theoretical: E[X] = {n*p}, Var(X) = {n*p*(1-p)}")
        
        np.random.seed(42)
        samples = np.random.binomial(n, p, size=100000)
        print(f"  Simulated:   E[X] = {samples.mean():.4f}, Var(X) = {samples.var():.4f}")
    
    def exercise_3_normal_second_moment(self):
        """
        Exercise 3: Normal Second Moment
        
        Show that for X ~ N(0, 1), E[X²] = 1.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: E[X²] for Standard Normal")
        
        print("\nX ~ N(0, 1)")
        print("E[X²] = ∫_{-∞}^{∞} x² · (1/√(2π)) e^{-x²/2} dx")
        
        print("\nMethod 1: Integration by parts")
        print("  Let u = x, dv = x·e^{-x²/2} dx")
        print("  Then du = dx, v = -e^{-x²/2}")
        
        print("\n  E[X²] = [-x·e^{-x²/2}]_{-∞}^{∞} + ∫ e^{-x²/2} dx")
        print("        = 0 + √(2π) · (1/√(2π))")
        print("        = 1")
        
        print("\nMethod 2: Use Var(X) = E[X²] - E[X]²")
        print("  For N(0,1): Var(X) = 1, E[X] = 0")
        print("  Therefore: E[X²] = Var(X) + E[X]² = 1 + 0 = 1")
        
        # Verify
        print("\n--- Verification ---")
        np.random.seed(42)
        samples = np.random.standard_normal(100000)
        print(f"  E[X²] from simulation: {(samples**2).mean():.4f}")
    
    def exercise_4_poisson_mle(self):
        """
        Exercise 4: Poisson MLE
        
        Derive the MLE for λ in Poisson(λ).
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Poisson MLE")
        
        print("\nData: x₁, x₂, ..., xₙ ~ Poisson(λ)")
        
        print("\nLikelihood:")
        print("  L(λ) = ∏ᵢ P(X = xᵢ)")
        print("       = ∏ᵢ (λ^{xᵢ} e^{-λ}) / xᵢ!")
        print("       = λ^{Σxᵢ} e^{-nλ} / ∏ᵢxᵢ!")
        
        print("\nLog-likelihood:")
        print("  ℓ(λ) = (Σxᵢ) log(λ) - nλ - Σ log(xᵢ!)")
        
        print("\nDerivative:")
        print("  dℓ/dλ = (Σxᵢ)/λ - n = 0")
        
        print("\nSolve:")
        print("  λ̂ = (Σxᵢ)/n = x̄")
        
        print("\nThe MLE is the sample mean!")
        
        # Verify
        np.random.seed(42)
        true_lambda = 5.0
        data = np.random.poisson(true_lambda, size=100)
        
        print(f"\n--- Verification ---")
        print(f"  True λ = {true_lambda}")
        print(f"  MLE λ̂ = x̄ = {data.mean():.4f}")
    
    def exercise_5_bayes_spam(self):
        """
        Exercise 5: Naive Bayes
        
        Update spam probability given multiple word occurrences.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Naive Bayes for Spam")
        
        print("\nSetup:")
        print("  P(spam) = 0.3")
        print("  Words: 'free', 'money', 'meeting'")
        print("  P(word|spam) and P(word|not spam):")
        
        P_spam = 0.3
        P_not_spam = 0.7
        
        # Likelihoods: P(word|spam), P(word|not spam)
        words = {
            'free': (0.8, 0.1),
            'money': (0.7, 0.1),
            'meeting': (0.2, 0.5)
        }
        
        print("  Word      P(w|spam)  P(w|not spam)")
        for word, (p_spam, p_not_spam) in words.items():
            print(f"  {word:10s} {p_spam:.1f}        {p_not_spam:.1f}")
        
        print("\nEmail contains: 'free', 'money'")
        observed = ['free', 'money']
        
        print("\nNaive Bayes assumes conditional independence:")
        print("  P(words|spam) = ∏ P(word|spam)")
        
        # Compute likelihoods
        P_words_given_spam = 1.0
        P_words_given_not_spam = 1.0
        for word in observed:
            P_words_given_spam *= words[word][0]
            P_words_given_not_spam *= words[word][1]
        
        print(f"\n  P(free, money | spam) = {words['free'][0]} × {words['money'][0]} = {P_words_given_spam}")
        print(f"  P(free, money | not spam) = {words['free'][1]} × {words['money'][1]} = {P_words_given_not_spam}")
        
        # Evidence
        P_words = P_words_given_spam * P_spam + P_words_given_not_spam * P_not_spam
        
        # Posterior
        P_spam_given_words = (P_words_given_spam * P_spam) / P_words
        
        print(f"\n  P(spam | free, money) = P(free,money|spam)P(spam) / P(free,money)")
        print(f"                        = ({P_words_given_spam} × {P_spam}) / {P_words:.4f}")
        print(f"                        = {P_spam_given_words:.4f}")
        
        print(f"\n  Classification: {'SPAM' if P_spam_given_words > 0.5 else 'NOT SPAM'}")
    
    def exercise_6_variance_sum(self):
        """
        Exercise 6: Variance of Sum
        
        Show Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y).
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Variance of Sum")
        
        print("\nVar(X + Y) = E[(X + Y - E[X+Y])²]")
        print("          = E[((X - μₓ) + (Y - μᵧ))²]")
        print("          = E[(X - μₓ)² + 2(X - μₓ)(Y - μᵧ) + (Y - μᵧ)²]")
        print("          = E[(X - μₓ)²] + 2E[(X - μₓ)(Y - μᵧ)] + E[(Y - μᵧ)²]")
        print("          = Var(X) + 2Cov(X, Y) + Var(Y)")
        
        print("\nSpecial case: X, Y independent")
        print("  Cov(X, Y) = 0")
        print("  Therefore: Var(X + Y) = Var(X) + Var(Y)")
        
        # Numerical example
        print("\n--- Numerical Example ---")
        np.random.seed(42)
        n = 100000
        
        # Correlated variables
        cov_matrix = [[1, 0.5], [0.5, 2]]
        samples = np.random.multivariate_normal([0, 0], cov_matrix, size=n)
        X, Y = samples[:, 0], samples[:, 1]
        
        var_X = X.var()
        var_Y = Y.var()
        cov_XY = np.cov(X, Y)[0, 1]
        var_sum = (X + Y).var()
        
        print(f"  Var(X) = {var_X:.4f}")
        print(f"  Var(Y) = {var_Y:.4f}")
        print(f"  Cov(X,Y) = {cov_XY:.4f}")
        print(f"  Var(X+Y) measured = {var_sum:.4f}")
        print(f"  Var(X) + Var(Y) + 2Cov(X,Y) = {var_X + var_Y + 2*cov_XY:.4f}")
    
    def exercise_7_exponential_memoryless(self):
        """
        Exercise 7: Exponential Memoryless Property
        
        Show P(X > s+t | X > s) = P(X > t) for Exponential.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Exponential Memoryless Property")
        
        print("\nX ~ Exponential(λ)")
        print("CDF: F(x) = 1 - e^{-λx}")
        print("P(X > x) = e^{-λx}")
        
        print("\nProve: P(X > s+t | X > s) = P(X > t)")
        
        print("\nLHS = P(X > s+t | X > s)")
        print("    = P(X > s+t AND X > s) / P(X > s)")
        print("    = P(X > s+t) / P(X > s)")
        print("    = e^{-λ(s+t)} / e^{-λs}")
        print("    = e^{-λt}")
        
        print("\nRHS = P(X > t) = e^{-λt}")
        
        print("\nTherefore LHS = RHS □")
        
        print("\nInterpretation:")
        print("  Given that we've waited s time units,")
        print("  the remaining wait time has same distribution as original!")
        print("  The process has 'no memory' of past waiting.")
        
        # Verify
        print("\n--- Simulation Verification ---")
        np.random.seed(42)
        lam = 0.5
        n = 100000
        X = np.random.exponential(1/lam, size=n)
        
        s, t = 2.0, 1.0
        
        P_gt_spt_given_gt_s = (X[X > s] > s + t).mean()
        P_gt_t = (X > t).mean()
        
        print(f"  P(X > {s+t} | X > {s}) = {P_gt_spt_given_gt_s:.4f}")
        print(f"  P(X > {t}) = {P_gt_t:.4f}")
    
    def exercise_8_cdf_to_pdf(self):
        """
        Exercise 8: CDF to PDF
        
        Given F(x) = 1 - e^{-x²/2} for x ≥ 0, find PDF.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: CDF to PDF")
        
        print("\nGiven: F(x) = 1 - e^{-x²/2} for x ≥ 0")
        print("       F(x) = 0 for x < 0")
        
        print("\nPDF: f(x) = dF/dx")
        print("     = d/dx [1 - e^{-x²/2}]")
        print("     = -e^{-x²/2} · d/dx[-x²/2]")
        print("     = -e^{-x²/2} · (-x)")
        print("     = x · e^{-x²/2}")
        
        print("\nThis is a Rayleigh distribution with σ = 1!")
        
        # Verify
        print("\n--- Verification ---")
        print("  Check: ∫₀^∞ f(x) dx = 1?")
        
        from scipy import integrate
        f = lambda x: x * np.exp(-x**2/2)
        integral, _ = integrate.quad(f, 0, np.inf)
        print(f"  ∫₀^∞ x·e^{{-x²/2}} dx = {integral:.4f}")
        
        # Mean
        mean_func = lambda x: x * x * np.exp(-x**2/2)
        E_X, _ = integrate.quad(mean_func, 0, np.inf)
        print(f"  E[X] = ∫₀^∞ x·f(x) dx = {E_X:.4f}")
        print(f"  (Rayleigh mean = √(π/2) = {np.sqrt(np.pi/2):.4f})")
    
    def exercise_9_transformation(self):
        """
        Exercise 9: Transformation of Random Variables
        
        If X ~ Uniform(0,1), find PDF of Y = -ln(X).
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Transformation of Variables")
        
        print("\nX ~ Uniform(0, 1)")
        print("Y = -ln(X)")
        
        print("\nMethod: CDF technique")
        print("  F_Y(y) = P(Y ≤ y)")
        print("         = P(-ln(X) ≤ y)")
        print("         = P(ln(X) ≥ -y)")
        print("         = P(X ≥ e^{-y})")
        print("         = 1 - e^{-y}  for y ≥ 0")
        
        print("\n  f_Y(y) = dF_Y/dy = e^{-y}")
        
        print("\nThis is Exponential(1)!")
        
        print("\nThis is the basis of inverse transform sampling:")
        print("  To generate Exp(1), compute -ln(U) where U ~ Uniform(0,1)")
        
        # Verify
        print("\n--- Verification ---")
        np.random.seed(42)
        X = np.random.uniform(0, 1, 100000)
        Y = -np.log(X)
        
        print(f"  Mean of Y: {Y.mean():.4f} (Exp(1) mean = 1)")
        print(f"  Var of Y:  {Y.var():.4f} (Exp(1) var = 1)")
    
    def exercise_10_law_large_numbers(self):
        """
        Exercise 10: Law of Large Numbers
        
        Demonstrate LLN with simulation.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Law of Large Numbers")
        
        print("\nWeak Law of Large Numbers:")
        print("  For IID X₁, X₂, ... with E[Xᵢ] = μ:")
        print("  X̄ₙ = (1/n)Σᵢ Xᵢ → μ in probability")
        
        print("\nMeaning: Sample mean converges to population mean")
        
        np.random.seed(42)
        
        # Population: Exponential(0.5) with mean = 2
        true_mean = 2.0
        
        print(f"\nPopulation: Exponential with mean = {true_mean}")
        print("\nSample means for increasing n:")
        
        max_n = 10000
        X = np.random.exponential(true_mean, max_n)
        
        ns = [10, 50, 100, 500, 1000, 5000, 10000]
        
        print(f"\n  {'n':>6s}  {'X̄':>8s}  {'|X̄ - μ|':>10s}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*10}")
        
        for n in ns:
            sample_mean = X[:n].mean()
            error = abs(sample_mean - true_mean)
            print(f"  {n:6d}  {sample_mean:8.4f}  {error:10.4f}")
        
        print(f"\n  True mean μ = {true_mean}")
        print("\nAs n increases, X̄ₙ gets closer to μ")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = ProbabilityExercises()
    
    print("PROBABILITY AND RANDOM VARIABLES EXERCISES")
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
