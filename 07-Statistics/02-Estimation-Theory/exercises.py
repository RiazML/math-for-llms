"""
Estimation Theory - Exercises
=============================
Practice problems for statistical estimation.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize


class EstimationExercises:
    """Exercises for estimation theory."""
    
    def exercise_1_bias_calculation(self):
        """
        Exercise 1: Calculate Bias
        
        Show that the MLE for variance (÷n) is biased.
        Derive the bias analytically and verify numerically.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Bias of Variance MLE")
        
        print("\nAnalytical derivation:")
        print("  MLE: σ̂² = (1/n)Σ(Xᵢ - X̄)²")
        print("  E[σ̂²] = E[(1/n)Σ(Xᵢ - X̄)²]")
        print("        = (1/n)E[Σ(Xᵢ - μ)² - n(X̄ - μ)²]")
        print("        = (1/n)[nσ² - σ²]")
        print("        = (n-1)σ²/n")
        print("  Bias = E[σ̂²] - σ² = -σ²/n")
        
        # Numerical verification
        np.random.seed(42)
        true_sigma2 = 10.0
        n = 20
        n_simulations = 50000
        
        mle_estimates = []
        for _ in range(n_simulations):
            data = np.random.normal(0, np.sqrt(true_sigma2), n)
            mle_estimates.append(data.var(ddof=0))
        
        expected_mle = np.mean(mle_estimates)
        expected_bias = expected_mle - true_sigma2
        theoretical_bias = -true_sigma2 / n
        
        print(f"\nNumerical verification (σ² = {true_sigma2}, n = {n}):")
        print(f"  E[σ̂²_MLE] = {expected_mle:.4f}")
        print(f"  Bias = {expected_bias:.4f}")
        print(f"  Theoretical bias = -σ²/n = {theoretical_bias:.4f}")
        print(f"  Match: {np.isclose(expected_bias, theoretical_bias, rtol=0.05)}")
    
    def exercise_2_mle_derivation(self):
        """
        Exercise 2: Derive MLE for Poisson
        
        Given X₁,...,Xₙ ~ Poisson(λ), derive the MLE for λ.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: MLE for Poisson")
        
        print("\nDerivation:")
        print("  PMF: P(X = k) = e^(-λ)λ^k / k!")
        print("  Likelihood: L(λ) = Π e^(-λ)λ^(xᵢ) / xᵢ!")
        print("  Log-likelihood:")
        print("    ℓ(λ) = Σ[-λ + xᵢ log(λ) - log(xᵢ!)]")
        print("         = -nλ + log(λ)Σxᵢ - Σlog(xᵢ!)")
        print("  ∂ℓ/∂λ = -n + (1/λ)Σxᵢ = 0")
        print("  λ̂_MLE = Σxᵢ/n = X̄")
        
        # Verify
        np.random.seed(42)
        true_lambda = 4.5
        n = 100
        
        data = np.random.poisson(true_lambda, n)
        lambda_mle = data.mean()
        
        print(f"\nNumerical verification:")
        print(f"  True λ = {true_lambda}")
        print(f"  Sample: Σxᵢ = {data.sum()}, n = {n}")
        print(f"  λ̂_MLE = X̄ = {lambda_mle:.4f}")
    
    def exercise_3_fisher_information(self):
        """
        Exercise 3: Fisher Information for Poisson
        
        Compute Fisher Information and Cramér-Rao bound.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Fisher Information for Poisson")
        
        print("\nDerivation:")
        print("  ℓ(λ) = -λ + x log(λ) - log(x!)")
        print("  ∂ℓ/∂λ = -1 + x/λ")
        print("  ∂²ℓ/∂λ² = -x/λ²")
        print("  I(λ) = -E[∂²ℓ/∂λ²] = E[X]/λ² = λ/λ² = 1/λ")
        print("\nCramér-Rao bound: Var(λ̂) ≥ 1/(nI(λ)) = λ/n")
        
        # Verify numerically
        np.random.seed(42)
        true_lambda = 5.0
        n_values = [10, 50, 100, 500]
        n_simulations = 10000
        
        print(f"\nNumerical verification (λ = {true_lambda}):")
        print(f"{'n':>6} {'Var(λ̂)':>12} {'CR bound':>12} {'Efficient':>10}")
        print("-" * 45)
        
        for n in n_values:
            estimates = [np.random.poisson(true_lambda, n).mean() 
                        for _ in range(n_simulations)]
            actual_var = np.var(estimates)
            cr_bound = true_lambda / n
            efficient = np.isclose(actual_var, cr_bound, rtol=0.1)
            
            print(f"{n:>6} {actual_var:>12.6f} {cr_bound:>12.6f} {str(efficient):>10}")
        
        print("\nMLE achieves Cramér-Rao bound → efficient estimator")
    
    def exercise_4_method_of_moments(self):
        """
        Exercise 4: Method of Moments for Beta Distribution
        
        Derive MoM estimators for Beta(α, β).
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Method of Moments for Beta")
        
        print("\nBeta distribution moments:")
        print("  E[X] = α/(α+β)")
        print("  Var(X) = αβ/[(α+β)²(α+β+1)]")
        print("\nLet m₁ = X̄, m₂ = (1/n)ΣXᵢ² (sample moments)")
        print("Sample variance: s² = m₂ - m₁²")
        
        print("\nSolving for α, β:")
        print("  From E[X] = m₁: α = m₁(α+β)")
        print("  Let k = α + β")
        print("  Var(X) = m₁(1-m₁)/(k+1) = s²")
        print("  k = m₁(1-m₁)/s² - 1")
        print("  α̂ = m₁ × k, β̂ = (1-m₁) × k")
        
        # Numerical example
        np.random.seed(42)
        true_alpha, true_beta = 2.0, 5.0
        n = 500
        
        data = np.random.beta(true_alpha, true_beta, n)
        
        m1 = data.mean()
        s2 = data.var()
        
        k = m1 * (1 - m1) / s2 - 1
        alpha_mom = m1 * k
        beta_mom = (1 - m1) * k
        
        print(f"\nNumerical verification (α={true_alpha}, β={true_beta}):")
        print(f"  m₁ = {m1:.4f}, s² = {s2:.6f}")
        print(f"  k = {k:.4f}")
        print(f"  α̂_MoM = {alpha_mom:.4f}")
        print(f"  β̂_MoM = {beta_mom:.4f}")
    
    def exercise_5_mse_comparison(self):
        """
        Exercise 5: Compare Estimators by MSE
        
        Compare sample variance (unbiased) vs MLE variance.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: MSE Comparison of Variance Estimators")
        
        print("\nTwo estimators for σ²:")
        print("  S² = Σ(Xᵢ-X̄)²/(n-1)  [unbiased]")
        print("  σ̂² = Σ(Xᵢ-X̄)²/n      [MLE, biased]")
        
        print("\nMSE = Variance + Bias²")
        
        np.random.seed(42)
        true_sigma2 = 10.0
        n_values = [5, 10, 20, 50, 100]
        n_simulations = 20000
        
        print(f"\n{'n':>4} {'MSE(S²)':>12} {'MSE(σ̂²)':>12} {'Better':>10}")
        print("-" * 42)
        
        for n in n_values:
            unbiased_est = []
            mle_est = []
            
            for _ in range(n_simulations):
                data = np.random.normal(0, np.sqrt(true_sigma2), n)
                unbiased_est.append(data.var(ddof=1))
                mle_est.append(data.var(ddof=0))
            
            mse_unbiased = np.mean((np.array(unbiased_est) - true_sigma2)**2)
            mse_mle = np.mean((np.array(mle_est) - true_sigma2)**2)
            better = "MLE" if mse_mle < mse_unbiased else "Unbiased"
            
            print(f"{n:>4} {mse_unbiased:>12.4f} {mse_mle:>12.4f} {better:>10}")
        
        print("\nConclusion:")
        print("  MLE has lower MSE for small n (bias-variance tradeoff)")
        print("  As n increases, both converge and difference diminishes")
    
    def exercise_6_map_estimation(self):
        """
        Exercise 6: MAP Estimation
        
        Compare MLE and MAP for Bernoulli with Beta prior.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: MAP Estimation for Bernoulli")
        
        print("\nSetup:")
        print("  Data: k successes in n trials")
        print("  Prior: p ~ Beta(α₀, β₀)")
        print("  Posterior: p|data ~ Beta(α₀+k, β₀+n-k)")
        
        print("\nEstimators:")
        print("  MLE: p̂ = k/n")
        print("  MAP: p̂ = (α₀+k-1)/(α₀+β₀+n-2)")
        print("  Posterior mean: E[p|data] = (α₀+k)/(α₀+β₀+n)")
        
        # Example with small sample
        np.random.seed(42)
        true_p = 0.7
        n = 5
        data = np.random.binomial(1, true_p, n)
        k = data.sum()
        
        print(f"\nExample: {k} successes in {n} trials (true p = {true_p})")
        
        # MLE
        p_mle = k / n
        print(f"\nMLE: p̂ = {p_mle:.4f}")
        
        # Different priors
        priors = [(1, 1), (2, 2), (5, 5), (10, 10)]
        
        print(f"\n{'Prior':>12} {'MAP':>10} {'Post Mean':>12}")
        print("-" * 38)
        
        for alpha0, beta0 in priors:
            alpha_post = alpha0 + k
            beta_post = beta0 + (n - k)
            
            p_map = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else p_mle
            p_post_mean = alpha_post / (alpha_post + beta_post)
            
            print(f"Beta({alpha0},{beta0}):   {p_map:>10.4f} {p_post_mean:>12.4f}")
        
        print("\nObservation: Stronger prior → more shrinkage toward prior mean (0.5)")
    
    def exercise_7_asymptotic_normality(self):
        """
        Exercise 7: Asymptotic Normality of MLE
        
        Verify that MLE is asymptotically normal.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Asymptotic Normality of MLE")
        
        print("\nTheory: √n(θ̂_MLE - θ) → N(0, 1/I(θ))")
        
        np.random.seed(42)
        
        # Exponential distribution
        true_lambda = 2.0
        fisher_info = 1 / true_lambda**2  # I(λ) = 1/λ² for Exp
        
        n_values = [30, 100, 500, 2000]
        n_simulations = 5000
        
        print(f"\nExponential(λ={true_lambda}), I(λ) = 1/λ² = {fisher_info}")
        print(f"\n{'n':>6} {'Mean':>10} {'Std':>10} {'Theory Std':>12}")
        print("-" * 42)
        
        for n in n_values:
            # Collect MLEs
            mles = []
            for _ in range(n_simulations):
                data = np.random.exponential(1/true_lambda, n)
                mles.append(1 / data.mean())  # MLE for λ
            
            mles = np.array(mles)
            
            # Standardized: √n(λ̂ - λ)
            standardized = np.sqrt(n) * (mles - true_lambda)
            
            theory_std = 1 / np.sqrt(fisher_info)  # √(1/I(λ)) = λ
            
            print(f"{n:>6} {standardized.mean():>10.4f} {standardized.std():>10.4f} {theory_std:>12.4f}")
        
        print("\nAs n increases, distribution approaches N(0, λ²)")
        
        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(standardized[:5000])
        print(f"\nShapiro-Wilk test (n={n_values[-1]}): p = {p_value:.4f}")
        print(f"Normal at α=0.05: {p_value > 0.05}")
    
    def exercise_8_mle_multiparameter(self):
        """
        Exercise 8: MLE for Normal (both parameters)
        
        Derive MLEs for μ and σ² simultaneously.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: MLE for Normal (μ and σ²)")
        
        print("\nLog-likelihood:")
        print("  ℓ(μ,σ²) = -(n/2)log(2π) - (n/2)log(σ²) - (1/2σ²)Σ(xᵢ-μ)²")
        
        print("\nFirst-order conditions:")
        print("  ∂ℓ/∂μ = (1/σ²)Σ(xᵢ-μ) = 0  →  μ̂ = x̄")
        print("  ∂ℓ/∂σ² = -n/(2σ²) + (1/2σ⁴)Σ(xᵢ-μ)² = 0")
        print("        →  σ̂² = (1/n)Σ(xᵢ-x̄)²")
        
        print("\nSecond-order conditions (Hessian):")
        print("  ∂²ℓ/∂μ² = -n/σ² < 0")
        print("  ∂²ℓ/∂(σ²)² = n/(2σ⁴) - (1/σ⁶)Σ(xᵢ-μ)²")
        print("  At MLE: = n/(2σ̂⁴) - nσ̂²/σ̂⁶ = -n/(2σ̂⁴) < 0")
        print("  Hessian is negative definite → maximum")
        
        # Numerical verification
        np.random.seed(42)
        true_mu, true_sigma = 5.0, 2.0
        n = 100
        
        data = np.random.normal(true_mu, true_sigma, n)
        
        mu_mle = data.mean()
        sigma2_mle = data.var(ddof=0)
        
        print(f"\nNumerical verification (μ={true_mu}, σ={true_sigma}):")
        print(f"  μ̂_MLE = {mu_mle:.4f}")
        print(f"  σ̂²_MLE = {sigma2_mle:.4f}")
        print(f"  σ̂_MLE = {np.sqrt(sigma2_mle):.4f}")
    
    def exercise_9_efficiency_comparison(self):
        """
        Exercise 9: Efficiency of Estimators
        
        Compare sample mean vs sample median for normal data.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Efficiency Comparison")
        
        print("\nFor N(μ, σ²):")
        print("  Var(X̄) = σ²/n")
        print("  Var(median) ≈ πσ²/(2n) for large n")
        print("  Relative efficiency = Var(X̄)/Var(median) ≈ 2/π ≈ 0.637")
        
        np.random.seed(42)
        true_mu = 10.0
        true_sigma = 3.0
        n = 50
        n_simulations = 10000
        
        means = []
        medians = []
        
        for _ in range(n_simulations):
            data = np.random.normal(true_mu, true_sigma, n)
            means.append(data.mean())
            medians.append(np.median(data))
        
        var_mean = np.var(means)
        var_median = np.var(medians)
        efficiency = var_mean / var_median
        
        print(f"\nSimulation (n={n}, σ={true_sigma}):")
        print(f"  Var(mean) = {var_mean:.6f}")
        print(f"  Var(median) = {var_median:.6f}")
        print(f"  Efficiency = {efficiency:.4f}")
        print(f"  Theoretical ≈ {2/np.pi:.4f}")
        
        print("\nConclusion:")
        print("  Mean is more efficient for normal data")
        print("  Median wastes ~36% of information")
        print("  BUT median is robust to outliers!")
        
        # With outliers
        print("\n--- With outliers ---")
        means_outlier = []
        medians_outlier = []
        
        for _ in range(n_simulations):
            data = np.random.normal(true_mu, true_sigma, n)
            data[0] = 100  # Add outlier
            means_outlier.append(data.mean())
            medians_outlier.append(np.median(data))
        
        mse_mean = np.mean((np.array(means_outlier) - true_mu)**2)
        mse_median = np.mean((np.array(medians_outlier) - true_mu)**2)
        
        print(f"  MSE(mean) = {mse_mean:.4f}")
        print(f"  MSE(median) = {mse_median:.4f}")
        print(f"  Median wins with outliers!")
    
    def exercise_10_regularization_prior(self):
        """
        Exercise 10: Connect Regularization to Prior
        
        Show Ridge/Lasso are MAP with Gaussian/Laplace priors.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Regularization as MAP")
        
        print("\nLinear regression: y = Xβ + ε, ε ~ N(0, σ²I)")
        print("\nNegative log-likelihood:")
        print("  -log L(β) ∝ ||y - Xβ||²/(2σ²)")
        
        print("\n" + "=" * 50)
        print("Ridge Regression (L2)")
        print("=" * 50)
        print("Prior: β ~ N(0, τ²I)")
        print("-log prior ∝ ||β||²/(2τ²)")
        print("\nMAP objective:")
        print("  min ||y - Xβ||² + (σ²/τ²)||β||²")
        print("  = min ||y - Xβ||² + λ||β||²  where λ = σ²/τ²")
        
        print("\n" + "=" * 50)
        print("Lasso Regression (L1)")
        print("=" * 50)
        print("Prior: βⱼ ~ Laplace(0, b)")
        print("  p(βⱼ) = (1/2b)exp(-|βⱼ|/b)")
        print("-log prior ∝ Σ|βⱼ|/b")
        print("\nMAP objective:")
        print("  min ||y - Xβ||² + (σ²/b)||β||₁")
        print("  = min ||y - Xβ||² + λ||β||₁")
        
        # Numerical demonstration
        np.random.seed(42)
        n, p = 100, 20
        X = np.random.randn(n, p)
        true_beta = np.array([3, -2, 0, 0, 1] + [0] * (p - 5))
        y = X @ true_beta + np.random.randn(n) * 0.5
        
        # OLS
        beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Ridge
        lambda_ridge = 1.0
        beta_ridge = np.linalg.solve(X.T @ X + lambda_ridge * np.eye(p), X.T @ y)
        
        print(f"\nNumerical comparison (first 5 coefficients):")
        print(f"  True:  {true_beta[:5]}")
        print(f"  OLS:   {beta_ols[:5].round(3)}")
        print(f"  Ridge: {beta_ridge[:5].round(3)}")
        
        print(f"\n||β||² comparison:")
        print(f"  OLS:   {np.sum(beta_ols**2):.4f}")
        print(f"  Ridge: {np.sum(beta_ridge**2):.4f}")
        print("  Ridge shrinks coefficients toward 0 (prior mean)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = EstimationExercises()
    
    print("ESTIMATION THEORY EXERCISES")
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
