"""
Common Probability Distributions - Exercises
============================================
Practice problems for probability distributions.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func


class DistributionExercises:
    """Exercises for probability distributions."""
    
    def exercise_1_poisson_mgf(self):
        """
        Exercise 1: Poisson MGF
        
        Derive the moment generating function of Poisson(λ).
        Use it to find E[X] and Var(X).
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Poisson MGF")
        
        print("\nMGF: M(t) = E[e^{tX}]")
        
        print("\nFor Poisson(λ):")
        print("  M(t) = Σₖ e^{tk} · λᵏe^{-λ}/k!")
        print("       = e^{-λ} Σₖ (λe^t)ᵏ/k!")
        print("       = e^{-λ} · e^{λe^t}")
        print("       = exp(λ(e^t - 1))")
        
        print("\nMean from MGF:")
        print("  E[X] = M'(0)")
        print("  M'(t) = λe^t · exp(λ(e^t - 1))")
        print("  M'(0) = λ · 1 = λ")
        
        print("\nVariance from MGF:")
        print("  E[X²] = M''(0)")
        print("  M''(t) = [λe^t + (λe^t)²] · exp(λ(e^t - 1))")
        print("  M''(0) = λ + λ² = λ(1 + λ)")
        print("  Var(X) = E[X²] - E[X]² = λ + λ² - λ² = λ")
        
        # Verify
        lam = 5.0
        print(f"\n--- Verification: λ = {lam} ---")
        np.random.seed(42)
        samples = np.random.poisson(lam, 100000)
        print(f"  Theoretical: E[X] = {lam}, Var(X) = {lam}")
        print(f"  Simulated:   E[X] = {samples.mean():.3f}, Var(X) = {samples.var():.3f}")
    
    def exercise_2_sum_poissons(self):
        """
        Exercise 2: Sum of Poissons
        
        Prove that sum of independent Poisson(λ₁) and Poisson(λ₂)
        is Poisson(λ₁ + λ₂).
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Sum of Independent Poissons")
        
        print("\nLet X ~ Poisson(λ₁), Y ~ Poisson(λ₂) independent")
        print("Z = X + Y")
        
        print("\nProof using MGF:")
        print("  M_X(t) = exp(λ₁(e^t - 1))")
        print("  M_Y(t) = exp(λ₂(e^t - 1))")
        
        print("\n  For independent RVs: M_Z(t) = M_X(t) · M_Y(t)")
        print("  M_Z(t) = exp(λ₁(e^t - 1)) · exp(λ₂(e^t - 1))")
        print("         = exp((λ₁ + λ₂)(e^t - 1))")
        
        print("\n  This is the MGF of Poisson(λ₁ + λ₂)!")
        print("  Therefore: X + Y ~ Poisson(λ₁ + λ₂) □")
        
        # Verify
        lambda1, lambda2 = 3.0, 5.0
        print(f"\n--- Verification: λ₁={lambda1}, λ₂={lambda2} ---")
        np.random.seed(42)
        X = np.random.poisson(lambda1, 100000)
        Y = np.random.poisson(lambda2, 100000)
        Z = X + Y
        
        print(f"  X + Y should be Poisson({lambda1 + lambda2})")
        print(f"  E[Z] = {Z.mean():.3f} (theory: {lambda1 + lambda2})")
        print(f"  Var(Z) = {Z.var():.3f} (theory: {lambda1 + lambda2})")
    
    def exercise_3_beta_binomial(self):
        """
        Exercise 3: Beta-Binomial Posterior
        
        Derive the posterior distribution for Beta-Binomial model.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Beta-Binomial Posterior")
        
        print("\nSetup:")
        print("  Prior: p ~ Beta(α, β)")
        print("  Likelihood: k | p ~ Binomial(n, p)")
        
        print("\nPrior PDF:")
        print("  π(p) ∝ p^{α-1}(1-p)^{β-1}")
        
        print("\nLikelihood:")
        print("  P(k|p) ∝ p^k (1-p)^{n-k}")
        
        print("\nPosterior (Bayes' theorem):")
        print("  π(p|k) ∝ π(p) · P(k|p)")
        print("         ∝ p^{α-1}(1-p)^{β-1} · p^k(1-p)^{n-k}")
        print("         = p^{(α+k)-1}(1-p)^{(β+n-k)-1}")
        
        print("\nThis is Beta(α + k, β + n - k)!")
        
        print("\nPosterior mean:")
        print("  E[p|k] = (α + k)/(α + β + n)")
        
        # Example
        alpha, beta_param = 2, 5
        n, k = 20, 12
        
        print(f"\n--- Example ---")
        print(f"  Prior: Beta({alpha}, {beta_param})")
        print(f"  Data: {k} successes in {n} trials")
        
        alpha_post = alpha + k
        beta_post = beta_param + n - k
        
        print(f"  Posterior: Beta({alpha_post}, {beta_post})")
        
        prior_mean = alpha / (alpha + beta_param)
        posterior_mean = alpha_post / (alpha_post + beta_post)
        mle = k / n
        
        print(f"\n  Prior mean: {prior_mean:.4f}")
        print(f"  MLE: {mle:.4f}")
        print(f"  Posterior mean: {posterior_mean:.4f}")
        print("  (Posterior is between prior and MLE)")
    
    def exercise_4_normal_transformation(self):
        """
        Exercise 4: Normal Linear Transformation
        
        Show that if X ~ N(μ, σ²), then aX + b ~ N(aμ + b, a²σ²).
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Normal Linear Transformation")
        
        print("\nLet X ~ N(μ, σ²), Y = aX + b")
        
        print("\nMethod 1: MGF")
        print("  M_X(t) = exp(μt + σ²t²/2)")
        print("  M_Y(t) = E[e^{tY}] = E[e^{t(aX+b)}]")
        print("         = e^{tb} · E[e^{(at)X}]")
        print("         = e^{tb} · M_X(at)")
        print("         = e^{tb} · exp(μ(at) + σ²(at)²/2)")
        print("         = exp(tb + aμt + a²σ²t²/2)")
        print("         = exp((aμ+b)t + (a²σ²)t²/2)")
        
        print("\n  This is MGF of N(aμ + b, a²σ²) ✓")
        
        print("\nMethod 2: Direct calculation")
        print("  E[Y] = E[aX + b] = aE[X] + b = aμ + b")
        print("  Var(Y) = Var(aX + b) = a²Var(X) = a²σ²")
        
        # Verify
        mu, sigma = 5, 2
        a, b = 3, -2
        
        print(f"\n--- Verification ---")
        print(f"  X ~ N({mu}, {sigma**2})")
        print(f"  Y = {a}X + ({b})")
        print(f"  Y should be N({a*mu + b}, {(a*sigma)**2})")
        
        np.random.seed(42)
        X = np.random.normal(mu, sigma, 100000)
        Y = a * X + b
        
        print(f"  Simulated: Mean = {Y.mean():.3f}, Var = {Y.var():.3f}")
    
    def exercise_5_entropy(self):
        """
        Exercise 5: Bernoulli Entropy
        
        Compute the entropy of Bernoulli(p).
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Bernoulli Entropy")
        
        print("\nEntropy: H(X) = -Σᵢ P(xᵢ) log P(xᵢ)")
        
        print("\nFor Bernoulli(p):")
        print("  H = -[p log(p) + (1-p) log(1-p)]")
        
        print("\nThis is called the binary entropy function H(p)")
        
        print("\nProperties:")
        print("  H(0) = H(1) = 0 (no uncertainty)")
        print("  H(0.5) = log(2) = 1 bit (maximum uncertainty)")
        
        print("\n  p      H(p)")
        print("  ────   ────")
        for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            if p == 0 or p == 1:
                H = 0
            else:
                H = -(p * np.log2(p) + (1-p) * np.log2(1-p))
            bar = '█' * int(H * 30) if H > 0 else ''
            print(f"  {p:.1f}   {H:.4f} {bar}")
        
        print("\nMaximum at p = 0.5 (uniform distribution has max entropy)")
    
    def exercise_6_gamma_chi_squared(self):
        """
        Exercise 6: Gamma and Chi-Squared
        
        Show that Chi-squared(k) = Gamma(k/2, 1/2).
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Gamma and Chi-Squared")
        
        print("\nChi-squared(k) is distribution of Σᵢ Zᵢ² where Zᵢ ~ N(0,1)")
        
        print("\nPDF of χ²(k):")
        print("  f(x) = x^{k/2 - 1} e^{-x/2} / (2^{k/2} Γ(k/2))")
        
        print("\nPDF of Gamma(α, β):")
        print("  f(x) = β^α x^{α-1} e^{-βx} / Γ(α)")
        
        print("\nSubstitute α = k/2, β = 1/2:")
        print("  f(x) = (1/2)^{k/2} x^{k/2 - 1} e^{-x/2} / Γ(k/2)")
        print("       = x^{k/2 - 1} e^{-x/2} / (2^{k/2} Γ(k/2))")
        
        print("\nThis matches χ²(k) PDF! ✓")
        
        # Verify
        k = 5
        print(f"\n--- Verification: k = {k} ---")
        np.random.seed(42)
        
        # Chi-squared from normal
        Z = np.random.standard_normal((100000, k))
        chi_sq = (Z**2).sum(axis=1)
        
        # Gamma
        gamma_samples = np.random.gamma(k/2, 2, 100000)  # scale = 1/rate = 2
        
        print(f"  χ²({k}): Mean = {chi_sq.mean():.3f}, Var = {chi_sq.var():.3f}")
        print(f"  Gamma({k/2}, 1/2): Mean = {gamma_samples.mean():.3f}, Var = {gamma_samples.var():.3f}")
        print(f"  Theory: Mean = {k}, Var = {2*k}")
    
    def exercise_7_t_distribution(self):
        """
        Exercise 7: Student's t Definition
        
        Show that Z/√(V/ν) ~ t(ν) where Z~N(0,1) and V~χ²(ν) independent.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Student's t Definition")
        
        print("\nDefinition: If Z ~ N(0,1) and V ~ χ²(ν) independent,")
        print("then T = Z / √(V/ν) has t-distribution with ν degrees of freedom.")
        
        print("\nWhy this matters:")
        print("  In estimating population mean from sample,")
        print("  (X̄ - μ) / (S/√n) ~ t(n-1)")
        print("  where S is sample standard deviation.")
        
        print("\nThis is used for confidence intervals and hypothesis tests")
        print("when population σ is unknown.")
        
        # Verify
        nu = 5
        print(f"\n--- Verification: ν = {nu} ---")
        np.random.seed(42)
        
        Z = np.random.standard_normal(100000)
        V = np.random.chisquare(nu, 100000)
        T = Z / np.sqrt(V / nu)
        
        # Compare to scipy t
        theoretical_var = nu / (nu - 2) if nu > 2 else np.inf
        
        print(f"  Constructed T: Mean = {T.mean():.4f}, Var = {T.var():.4f}")
        print(f"  Theory: Mean = 0, Var = ν/(ν-2) = {theoretical_var:.4f}")
        
        # Quantiles
        print("\n  Quantile comparison:")
        for q in [0.025, 0.5, 0.975]:
            emp_q = np.percentile(T, q * 100)
            th_q = stats.t.ppf(q, nu)
            print(f"    q={q}: Empirical = {emp_q:.4f}, Theory = {th_q:.4f}")
    
    def exercise_8_exponential_minimum(self):
        """
        Exercise 8: Minimum of Exponentials
        
        Show that min of n independent Exp(λ) is Exp(nλ).
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Minimum of Exponentials")
        
        print("\nLet X₁, ..., Xₙ ~ Exp(λ) independent")
        print("M = min(X₁, ..., Xₙ)")
        
        print("\nP(M > t) = P(X₁ > t, ..., Xₙ > t)")
        print("         = P(X₁ > t) · ... · P(Xₙ > t)  (independence)")
        print("         = e^{-λt} · ... · e^{-λt}")
        print("         = e^{-nλt}")
        
        print("\nThis is P(Y > t) for Y ~ Exp(nλ)")
        print("Therefore M ~ Exp(nλ)")
        
        print("\nInterpretation:")
        print("  If n processes each have rate λ,")
        print("  the rate of first completion is nλ.")
        
        # Verify
        n = 5
        lam = 2.0
        
        print(f"\n--- Verification: n={n}, λ={lam} ---")
        np.random.seed(42)
        
        X = np.random.exponential(1/lam, (100000, n))
        M = X.min(axis=1)
        
        theoretical_mean = 1 / (n * lam)
        
        print(f"  min(X₁,...,X₅) should be Exp({n*lam})")
        print(f"  Mean: {M.mean():.4f} (theory: {theoretical_mean:.4f})")
        print(f"  Var:  {M.var():.4f} (theory: {theoretical_mean**2:.4f})")
    
    def exercise_9_normal_squared(self):
        """
        Exercise 9: Square of Normal
        
        Find the distribution of X² where X ~ N(0,1).
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Square of Standard Normal")
        
        print("\nLet X ~ N(0,1), Y = X²")
        
        print("\nCDF of Y:")
        print("  F_Y(y) = P(Y ≤ y) = P(X² ≤ y)")
        print("         = P(-√y ≤ X ≤ √y)  for y ≥ 0")
        print("         = Φ(√y) - Φ(-√y)")
        print("         = 2Φ(√y) - 1")
        
        print("\nPDF of Y:")
        print("  f_Y(y) = d/dy [2Φ(√y) - 1]")
        print("         = 2φ(√y) · 1/(2√y)")
        print("         = φ(√y)/√y")
        print("         = (1/√(2π)) e^{-y/2} / √y")
        print("         = y^{-1/2} e^{-y/2} / √(2π)")
        
        print("\nRecognize this as Gamma(1/2, 1/2) = χ²(1)")
        
        print("\nTherefore: X² ~ χ²(1)")
        
        # Verify
        print("\n--- Verification ---")
        np.random.seed(42)
        X = np.random.standard_normal(100000)
        Y = X**2
        
        # Chi-squared(1) has mean=1, var=2
        print(f"  X² has: Mean = {Y.mean():.4f} (χ²(1) mean = 1)")
        print(f"          Var = {Y.var():.4f} (χ²(1) var = 2)")
    
    def exercise_10_mixture_distribution(self):
        """
        Exercise 10: Mixture of Normals
        
        Compute mean and variance of Gaussian Mixture Model.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Gaussian Mixture Model")
        
        print("\nGMM: X ~ Σᵢ πᵢ N(μᵢ, σᵢ²)")
        print("where Σᵢ πᵢ = 1 (mixing weights)")
        
        print("\nMean of mixture:")
        print("  E[X] = Σᵢ πᵢ E[Xᵢ] = Σᵢ πᵢ μᵢ")
        
        print("\nVariance of mixture:")
        print("  E[X²] = Σᵢ πᵢ E[Xᵢ²] = Σᵢ πᵢ (σᵢ² + μᵢ²)")
        print("  Var(X) = E[X²] - E[X]²")
        print("         = Σᵢ πᵢ(σᵢ² + μᵢ²) - (Σᵢ πᵢ μᵢ)²")
        
        print("\nThis can be written as:")
        print("  Var(X) = Σᵢ πᵢ σᵢ² + Σᵢ πᵢ (μᵢ - μ̄)²")
        print("         = within-cluster var + between-cluster var")
        
        # Example
        print("\n--- Example: 2-component mixture ---")
        pi = [0.3, 0.7]
        mu = [-2, 3]
        sigma = [1, 1.5]
        
        print(f"  π = {pi}, μ = {mu}, σ = {sigma}")
        
        # Mean
        mixture_mean = sum(p * m for p, m in zip(pi, mu))
        print(f"\n  E[X] = {pi[0]}×{mu[0]} + {pi[1]}×{mu[1]} = {mixture_mean:.2f}")
        
        # Variance
        E_X2 = sum(p * (s**2 + m**2) for p, m, s in zip(pi, mu, sigma))
        mixture_var = E_X2 - mixture_mean**2
        print(f"  E[X²] = {E_X2:.4f}")
        print(f"  Var(X) = {mixture_var:.4f}")
        
        # Simulate
        np.random.seed(42)
        n_samples = 100000
        
        # Choose component
        components = np.random.choice(2, n_samples, p=pi)
        samples = np.where(
            components == 0,
            np.random.normal(mu[0], sigma[0], n_samples),
            np.random.normal(mu[1], sigma[1], n_samples)
        )
        
        print(f"\n  Simulation:")
        print(f"    Mean = {samples.mean():.4f}")
        print(f"    Var = {samples.var():.4f}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = DistributionExercises()
    
    print("PROBABILITY DISTRIBUTIONS EXERCISES")
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
