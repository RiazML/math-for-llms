"""
Bayesian Inference - Exercises
==============================
Practice problems for Bayesian inference.
"""

import numpy as np
from scipy import stats
from scipy.special import beta as beta_func, comb


class BayesianExercises:
    """Exercises for Bayesian inference."""
    
    def exercise_1_beta_binomial(self):
        """
        Exercise 1: Beta-Binomial
        
        Compute the posterior for a coin bias.
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Beta-Binomial")
        
        print("\nProblem: A coin is flipped 20 times, showing 14 heads.")
        print("Prior: Beta(3, 3) (slight belief in fair coin)")
        print("Find the posterior and 95% credible interval.")
        
        # Prior parameters
        alpha_0 = 3
        beta_0 = 3
        
        # Data
        n = 20
        k = 14  # heads
        
        # Posterior
        alpha_n = alpha_0 + k
        beta_n = beta_0 + (n - k)
        
        print(f"\nPrior: Beta({alpha_0}, {beta_0})")
        print(f"  Prior mean: {alpha_0/(alpha_0+beta_0):.4f}")
        
        print(f"\nData: {k} heads in {n} flips")
        print(f"MLE: {k/n:.4f}")
        
        print(f"\nPosterior: Beta({alpha_n}, {beta_n})")
        post_mean = alpha_n / (alpha_n + beta_n)
        post_mode = (alpha_n - 1) / (alpha_n + beta_n - 2)
        post_var = (alpha_n * beta_n) / ((alpha_n + beta_n)**2 * (alpha_n + beta_n + 1))
        
        print(f"  Posterior mean: {post_mean:.4f}")
        print(f"  Posterior mode (MAP): {post_mode:.4f}")
        print(f"  Posterior std: {np.sqrt(post_var):.4f}")
        
        # 95% credible interval
        ci = stats.beta.ppf([0.025, 0.975], alpha_n, beta_n)
        print(f"  95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        # Probability coin is biased toward heads
        p_biased = 1 - stats.beta.cdf(0.5, alpha_n, beta_n)
        print(f"\n  P(p > 0.5 | data) = {p_biased:.4f}")
    
    def exercise_2_normal_conjugate(self):
        """
        Exercise 2: Normal-Normal Conjugacy
        
        Update belief about mean with normal prior.
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Normal-Normal Conjugacy")
        
        print("\nProblem: Estimate mean test score.")
        print("Prior: μ ~ N(70, 100) (from historical data)")
        print("Known σ² = 225 (σ = 15)")
        print("Data: Sample of 25 students, mean = 75")
        
        # Prior
        mu_0 = 70
        tau_0_sq = 100
        
        # Known variance
        sigma_sq = 225
        
        # Data
        n = 25
        x_bar = 75
        
        # Posterior parameters
        precision_0 = 1 / tau_0_sq
        precision_data = n / sigma_sq
        precision_n = precision_0 + precision_data
        
        tau_n_sq = 1 / precision_n
        mu_n = (precision_0 * mu_0 + precision_data * x_bar) / precision_n
        
        print(f"\nPrior: N({mu_0}, {tau_0_sq})")
        print(f"  Prior weight: {precision_0/precision_n:.4f}")
        
        print(f"\nData: n={n}, x̄={x_bar}")
        print(f"  Data weight: {precision_data/precision_n:.4f}")
        
        print(f"\nPosterior: N({mu_n:.4f}, {tau_n_sq:.4f})")
        print(f"  Posterior mean: {mu_n:.4f}")
        print(f"  Posterior std: {np.sqrt(tau_n_sq):.4f}")
        
        # 95% CI
        ci = stats.norm.ppf([0.025, 0.975], mu_n, np.sqrt(tau_n_sq))
        print(f"  95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        print(f"\nNote: Posterior mean is weighted average of prior mean and data mean")
    
    def exercise_3_prior_sensitivity(self):
        """
        Exercise 3: Prior Sensitivity Analysis
        
        Examine how posterior changes with different priors.
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Prior Sensitivity Analysis")
        
        # Data: 6 successes in 10 trials
        k, n = 6, 10
        
        print(f"Data: {k}/{n} successes")
        print(f"MLE: {k/n:.4f}")
        print("\nComparing different priors:\n")
        
        priors = [
            (0.5, 0.5, "Jeffreys (non-informative)"),
            (1, 1, "Uniform"),
            (2, 2, "Weakly informative (fair)"),
            (5, 5, "Moderately informative"),
            (2, 8, "Skeptical (low p)"),
            (8, 2, "Optimistic (high p)")
        ]
        
        print(f"{'Prior':^30} {'Prior Mean':>12} {'Post Mean':>12} {'95% CI':>20}")
        print("-" * 78)
        
        for a, b, name in priors:
            prior_mean = a / (a + b)
            
            # Posterior
            a_post = a + k
            b_post = b + (n - k)
            post_mean = a_post / (a_post + b_post)
            
            ci = stats.beta.ppf([0.025, 0.975], a_post, b_post)
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            
            print(f"Beta({a},{b}) {name:^18} {prior_mean:>12.4f} {post_mean:>12.4f} {ci_str:>20}")
        
        print("\nObservation: With limited data, prior has substantial influence")
        print("Sensitivity analysis is crucial when priors are informative!")
    
    def exercise_4_posterior_predictive(self):
        """
        Exercise 4: Posterior Predictive Distribution
        
        Compute probability of next observation.
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Posterior Predictive")
        
        print("Posterior: p | data ~ Beta(10, 5)")
        print("Question: What's P(next observation = success)?")
        
        alpha, beta = 10, 5
        
        # Posterior predictive for Bernoulli with Beta posterior
        # P(success | data) = E[p | data] = alpha / (alpha + beta)
        
        p_success = alpha / (alpha + beta)
        
        print(f"\nP(next = success | data) = E[p | data] = α/(α+β)")
        print(f"                         = {alpha}/({alpha}+{beta})")
        print(f"                         = {p_success:.4f}")
        
        print("\nThis is NOT the same as plugging in MAP estimate!")
        map_estimate = (alpha - 1) / (alpha + beta - 2)
        print(f"MAP estimate of p: {map_estimate:.4f}")
        
        print("\nPosterior predictive accounts for parameter uncertainty")
        
        # For multiple future observations
        print("\nPredicting next 5 observations:")
        n_future = 5
        for k in range(n_future + 1):
            # Beta-Binomial distribution
            prob = comb(n_future, k) * beta_func(alpha + k, beta + n_future - k) / beta_func(alpha, beta)
            print(f"  P({k} successes in 5) = {prob:.4f}")
    
    def exercise_5_bayes_factor(self):
        """
        Exercise 5: Model Comparison with Bayes Factors
        
        Compare two competing models.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Bayes Factor")
        
        print("Data: 3 successes in 10 trials")
        print("M1: Fair coin (p = 0.5)")
        print("M2: Biased coin (p ~ Beta(1,1) = Uniform)")
        
        k, n = 3, 10
        
        # Evidence for M1 (fair coin)
        p1_evidence = stats.binom.pmf(k, n, 0.5)
        
        # Evidence for M2 (uniform prior)
        # Marginal likelihood: integral of likelihood × prior
        # For Beta(1,1) prior: n choose k × Beta(1+k, 1+n-k) / Beta(1,1)
        p2_evidence = comb(n, k) * beta_func(1 + k, 1 + n - k) / beta_func(1, 1)
        
        print(f"\nModel evidences:")
        print(f"  P(data | M1) = {p1_evidence:.6f}")
        print(f"  P(data | M2) = {p2_evidence:.6f}")
        
        # Bayes Factor
        bf_12 = p1_evidence / p2_evidence
        bf_21 = p2_evidence / p1_evidence
        
        print(f"\nBayes Factor:")
        print(f"  BF₁₂ = P(data|M1)/P(data|M2) = {bf_12:.4f}")
        print(f"  BF₂₁ = P(data|M2)/P(data|M1) = {bf_21:.4f}")
        
        # Interpretation
        print("\nInterpretation (BF₁₂):")
        if bf_12 > 10:
            print("  Strong evidence for M1 (fair coin)")
        elif bf_12 > 3:
            print("  Moderate evidence for M1")
        elif bf_12 > 1:
            print("  Weak evidence for M1")
        elif bf_12 > 1/3:
            print("  Inconclusive")
        elif bf_12 > 1/10:
            print("  Weak evidence for M2")
        else:
            print("  Strong evidence for M2 (biased coin)")
        
        # Posterior model probabilities (equal prior)
        p_m1_post = p1_evidence / (p1_evidence + p2_evidence)
        print(f"\nPosterior model probabilities (equal priors):")
        print(f"  P(M1 | data) = {p_m1_post:.4f}")
        print(f"  P(M2 | data) = {1-p_m1_post:.4f}")
    
    def exercise_6_credible_interval(self):
        """
        Exercise 6: Credible Interval Types
        
        Compare equal-tailed and HPD intervals.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Credible Interval Types")
        
        # Skewed posterior: Beta(2, 8)
        alpha, beta = 2, 8
        
        print(f"Posterior: Beta({alpha}, {beta})")
        print(f"  Mean: {alpha/(alpha+beta):.4f}")
        print(f"  Mode: {(alpha-1)/(alpha+beta-2):.4f}")
        
        # Equal-tailed 95% CI
        et_ci = stats.beta.ppf([0.025, 0.975], alpha, beta)
        et_width = et_ci[1] - et_ci[0]
        
        print(f"\n1. Equal-tailed 95% CI:")
        print(f"   [{et_ci[0]:.4f}, {et_ci[1]:.4f}]")
        print(f"   Width: {et_width:.4f}")
        
        # HPD interval (highest posterior density)
        # For Beta, this requires numerical optimization
        from scipy.optimize import minimize_scalar
        
        def hpd_objective(lower, alpha, beta, prob=0.95):
            """Find HPD by minimizing width given coverage."""
            upper = stats.beta.ppf(stats.beta.cdf(lower, alpha, beta) + prob, alpha, beta)
            if np.isnan(upper):
                return np.inf
            return upper - lower
        
        result = minimize_scalar(hpd_objective, bounds=(0, 0.1), 
                                args=(alpha, beta), method='bounded')
        hpd_lower = result.x
        hpd_upper = stats.beta.ppf(stats.beta.cdf(hpd_lower, alpha, beta) + 0.95, alpha, beta)
        hpd_width = hpd_upper - hpd_lower
        
        print(f"\n2. HPD 95% CI (approximate):")
        print(f"   [{hpd_lower:.4f}, {hpd_upper:.4f}]")
        print(f"   Width: {hpd_width:.4f}")
        
        print(f"\nHPD is narrower ({hpd_width:.4f} vs {et_width:.4f})")
        print("For skewed distributions, HPD is often preferred")
    
    def exercise_7_sequential_update(self):
        """
        Exercise 7: Sequential Bayesian Updating
        
        Update beliefs as data arrives.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Sequential Updating")
        
        np.random.seed(42)
        
        # True parameter
        true_p = 0.7
        
        # Start with uniform prior
        alpha, beta = 1, 1
        
        print(f"True p = {true_p}")
        print("Starting with uniform prior Beta(1,1)")
        print("Observing data one at a time...\n")
        
        # Generate 20 observations
        observations = np.random.binomial(1, true_p, 20)
        
        print(f"{'Obs':>4} {'Value':>6} {'Posterior':>15} {'Mean':>8} {'95% CI Width':>12}")
        print("-" * 50)
        
        # Initial state
        mean = alpha / (alpha + beta)
        ci = stats.beta.ppf([0.025, 0.975], alpha, beta)
        print(f"{'':>4} {'':>6} Beta({alpha:>2},{beta:>2}){mean:>12.4f} {ci[1]-ci[0]:>12.4f}")
        
        for i, obs in enumerate(observations):
            # Update
            alpha += obs
            beta += (1 - obs)
            
            mean = alpha / (alpha + beta)
            ci = stats.beta.ppf([0.025, 0.975], alpha, beta)
            
            if (i + 1) % 5 == 0:  # Print every 5th
                print(f"{i+1:>4} {obs:>6} Beta({alpha:>2},{beta:>2}){mean:>12.4f} {ci[1]-ci[0]:>12.4f}")
        
        print(f"\nFinal posterior: Beta({alpha}, {beta})")
        print(f"Final mean: {alpha/(alpha+beta):.4f} (true: {true_p})")
        print("\nSequential updating gives same result as batch update!")
        
        # Verify batch
        alpha_batch = 1 + observations.sum()
        beta_batch = 1 + len(observations) - observations.sum()
        print(f"Batch update: Beta({alpha_batch}, {beta_batch})")
    
    def exercise_8_map_ridge(self):
        """
        Exercise 8: MAP and Ridge Regression Connection
        
        Show Ridge = MAP with Gaussian prior.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: MAP = Ridge Regression")
        
        np.random.seed(42)
        
        # Generate regression data
        n, p = 100, 10
        X = np.random.randn(n, p)
        true_w = np.array([2, -1, 0.5, 0, 0, 0, 0, 0, 0, 0])
        sigma = 0.5
        y = X @ true_w + np.random.randn(n) * sigma
        
        print("Linear regression: y = Xw + ε")
        print(f"True w: {true_w}")
        print(f"\nBayesian model:")
        print("  Likelihood: y|w ~ N(Xw, σ²I)")
        print("  Prior: w ~ N(0, τ²I)")
        
        # OLS (flat prior / MLE)
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Ridge (MAP with Gaussian prior)
        # Prior: w ~ N(0, τ²I) where τ² = σ²/λ
        lambda_val = 1.0
        w_ridge = np.linalg.solve(X.T @ X + lambda_val * np.eye(p), X.T @ y)
        
        # Bayesian with explicit prior
        sigma_sq = sigma**2
        tau_sq = sigma_sq / lambda_val  # Prior variance
        
        # Posterior covariance and mean
        S_post_inv = X.T @ X / sigma_sq + np.eye(p) / tau_sq
        S_post = np.linalg.inv(S_post_inv)
        m_post = S_post @ X.T @ y / sigma_sq  # Posterior mean
        
        print(f"\nλ = {lambda_val}, τ² = σ²/λ = {tau_sq:.4f}")
        
        print(f"\n{'':>12}{'OLS':>10}{'Ridge':>10}{'MAP':>10}{'True':>10}")
        print("-" * 55)
        for i in range(5):
            print(f"w{i+1:>10}{w_ols[i]:>10.4f}{w_ridge[i]:>10.4f}{m_post[i]:>10.4f}{true_w[i]:>10.4f}")
        
        print("\nRidge and MAP are identical!")
        print(f"Max difference: {np.max(np.abs(w_ridge - m_post)):.2e}")
    
    def exercise_9_prediction(self):
        """
        Exercise 9: Bayesian Prediction
        
        Compute predictive distribution.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Bayesian Prediction")
        
        print("Setup: Estimating IQ scores")
        print("  Population: N(100, 225)  [σ² = 225 known]")
        print("  Prior on μ: N(100, 100)")
        print("  Data: 10 observations with mean 110")
        
        # Parameters
        sigma_sq = 225
        mu_0 = 100
        tau_sq_0 = 100
        n = 10
        x_bar = 110
        
        # Posterior
        precision_0 = 1 / tau_sq_0
        precision_data = n / sigma_sq
        precision_post = precision_0 + precision_data
        tau_sq_post = 1 / precision_post
        mu_post = (precision_0 * mu_0 + precision_data * x_bar) / precision_post
        
        print(f"\nPosterior for μ: N({mu_post:.2f}, {tau_sq_post:.4f})")
        
        # Predictive distribution for new observation
        # y_new | data ~ N(mu_post, sigma_sq + tau_sq_post)
        pred_var = sigma_sq + tau_sq_post
        pred_std = np.sqrt(pred_var)
        
        print(f"\nPredictive distribution for new observation:")
        print(f"  y_new | data ~ N({mu_post:.2f}, {pred_var:.2f})")
        print(f"  Predictive std: {pred_std:.2f}")
        
        # 95% prediction interval
        pred_ci = stats.norm.ppf([0.025, 0.975], mu_post, pred_std)
        print(f"  95% prediction interval: [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}]")
        
        # Compare with just using point estimate
        print(f"\nCompare: Using only posterior mean (ignoring uncertainty)")
        print(f"  Would predict: N({mu_post:.2f}, {sigma_sq})")
        print(f"  Prediction interval: [{mu_post - 1.96*15:.2f}, {mu_post + 1.96*15:.2f}]")
        print("\nBayesian prediction interval is wider - accounts for uncertainty in μ!")
    
    def exercise_10_conjugate_table(self):
        """
        Exercise 10: Conjugate Prior Table
        
        Derive posteriors for common conjugate pairs.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Conjugate Prior Summary")
        
        print("=" * 70)
        print("CONJUGATE PRIOR TABLE")
        print("=" * 70)
        
        print("""
Likelihood          | Conjugate Prior      | Posterior
--------------------+----------------------+---------------------------
Bernoulli(p)        | Beta(α, β)           | Beta(α + Σx, β + n - Σx)
Binomial(n, p)      | Beta(α, β)           | Beta(α + k, β + n - k)
Poisson(λ)          | Gamma(α, β)          | Gamma(α + Σx, β + n)
Exponential(λ)      | Gamma(α, β)          | Gamma(α + n, β + Σx)
Normal(μ, σ²) [σ²]  | Normal(μ₀, τ²)       | Normal(μₙ, τₙ²)
Normal(μ, σ²) [μ]   | Inv-Gamma(α, β)      | Inv-Gamma(α + n/2, β + SSE/2)
Multinomial         | Dirichlet(α)         | Dirichlet(α + counts)
""")
        
        print("\nNormal-Normal posterior formulas:")
        print("  τₙ² = 1/(1/τ₀² + n/σ²)")
        print("  μₙ = τₙ² × (μ₀/τ₀² + nx̄/σ²)")
        
        print("\nKey insight: Posterior hyperparameters are")
        print("prior hyperparameters + sufficient statistics")
        
        # Numerical example for Poisson-Gamma
        print("\n" + "-" * 70)
        print("Example: Poisson-Gamma")
        print("-" * 70)
        
        # Prior: λ ~ Gamma(2, 1)
        alpha_0, beta_0 = 2, 1
        
        # Data: observations 3, 5, 4, 2, 6
        data = np.array([3, 5, 4, 2, 6])
        n = len(data)
        
        # Posterior
        alpha_n = alpha_0 + data.sum()
        beta_n = beta_0 + n
        
        print(f"Prior: Gamma({alpha_0}, {beta_0})")
        print(f"  Prior mean: α/β = {alpha_0/beta_0:.4f}")
        print(f"\nData: {data}")
        print(f"  Sum: {data.sum()}, n: {n}")
        print(f"\nPosterior: Gamma({alpha_n}, {beta_n})")
        print(f"  Posterior mean: {alpha_n/beta_n:.4f}")
        print(f"  MLE: {data.mean():.4f}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = BayesianExercises()
    
    print("BAYESIAN INFERENCE EXERCISES")
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
