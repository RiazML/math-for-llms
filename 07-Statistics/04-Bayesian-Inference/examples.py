"""
Bayesian Inference - Examples
=============================
Practical demonstrations of Bayesian inference.
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
import warnings
warnings.filterwarnings('ignore')


def example_beta_binomial():
    """Beta-Binomial conjugate pair."""
    print("=" * 60)
    print("EXAMPLE 1: Beta-Binomial Conjugacy")
    print("=" * 60)
    
    # Coin flip example
    np.random.seed(42)
    
    # Prior belief: symmetric, slight preference for fair coin
    alpha_prior, beta_prior = 2, 2
    
    # Observe data: 7 heads in 10 flips
    n_flips = 10
    n_heads = 7
    
    print("Estimating coin bias p")
    print(f"\nPrior: Beta({alpha_prior}, {beta_prior})")
    print(f"  Prior mean: {alpha_prior/(alpha_prior + beta_prior):.4f}")
    print(f"  Prior mode: {(alpha_prior-1)/(alpha_prior + beta_prior - 2):.4f}")
    
    print(f"\nData: {n_heads} heads in {n_flips} flips")
    print(f"MLE: {n_heads/n_flips:.4f}")
    
    # Posterior
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + (n_flips - n_heads)
    
    print(f"\nPosterior: Beta({alpha_post}, {beta_post})")
    print(f"  Posterior mean: {alpha_post/(alpha_post + beta_post):.4f}")
    print(f"  Posterior mode (MAP): {(alpha_post-1)/(alpha_post + beta_post - 2):.4f}")
    
    # 95% credible interval
    ci = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
    print(f"  95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Posterior probability that p > 0.5
    prob_biased = 1 - stats.beta.cdf(0.5, alpha_post, beta_post)
    print(f"\n  P(p > 0.5 | data) = {prob_biased:.4f}")


def example_normal_normal():
    """Normal-Normal conjugate pair."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Normal-Normal Conjugacy")
    print("=" * 60)
    
    # Estimating mean with known variance
    sigma_sq = 4.0  # Known variance
    sigma = np.sqrt(sigma_sq)
    
    # Prior on mean
    mu_prior = 0
    tau_sq_prior = 10  # Prior variance (uncertain)
    
    print(f"Known σ² = {sigma_sq}")
    print(f"Prior: μ ~ N({mu_prior}, {tau_sq_prior})")
    
    # Observed data
    data = np.array([2.5, 3.1, 2.8, 3.5, 2.9])
    n = len(data)
    x_bar = data.mean()
    
    print(f"\nData: {data}")
    print(f"Sample mean: {x_bar:.4f}")
    print(f"MLE: {x_bar:.4f}")
    
    # Posterior parameters
    precision_prior = 1 / tau_sq_prior
    precision_data = n / sigma_sq
    precision_post = precision_prior + precision_data
    tau_sq_post = 1 / precision_post
    
    mu_post = (precision_prior * mu_prior + precision_data * x_bar) / precision_post
    
    print(f"\nPosterior: μ | data ~ N({mu_post:.4f}, {tau_sq_post:.4f})")
    print(f"  Posterior mean: {mu_post:.4f}")
    print(f"  Posterior std: {np.sqrt(tau_sq_post):.4f}")
    
    # 95% credible interval
    ci = stats.norm.ppf([0.025, 0.975], mu_post, np.sqrt(tau_sq_post))
    print(f"  95% credible interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Posterior is precision-weighted average
    print(f"\nPosterior mean is weighted average:")
    print(f"  Prior weight: {precision_prior/precision_post:.4f}")
    print(f"  Data weight: {precision_data/precision_post:.4f}")


def example_prior_influence():
    """Effect of prior strength on posterior."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Prior Influence on Posterior")
    print("=" * 60)
    
    # Same data with different priors
    n_heads, n_flips = 7, 10
    
    print(f"Data: {n_heads}/{n_flips} heads")
    print(f"MLE: {n_heads/n_flips:.4f}\n")
    
    priors = [
        (1, 1, "Uniform (non-informative)"),
        (2, 2, "Weakly informative"),
        (10, 10, "Moderately informative (p≈0.5)"),
        (50, 50, "Strongly informative (p≈0.5)"),
        (2, 10, "Informative (p≈0.2)")
    ]
    
    print(f"{'Prior':^35} {'Prior Mean':>12} {'Post Mean':>12} {'Post Mode':>12}")
    print("-" * 75)
    
    for alpha, beta, name in priors:
        prior_mean = alpha / (alpha + beta)
        
        alpha_post = alpha + n_heads
        beta_post = beta + (n_flips - n_heads)
        
        post_mean = alpha_post / (alpha_post + beta_post)
        post_mode = (alpha_post - 1) / (alpha_post + beta_post - 2)
        
        print(f"Beta({alpha},{beta}) {name:^20} {prior_mean:>12.4f} {post_mean:>12.4f} {post_mode:>12.4f}")
    
    print("\nWith weak priors, posterior ≈ MLE")
    print("With strong priors, posterior pulled toward prior")


def example_sequential_updating():
    """Sequential Bayesian updating."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Sequential Bayesian Updating")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True probability
    true_p = 0.6
    
    # Start with uniform prior
    alpha, beta = 1, 1
    
    print(f"True p = {true_p}")
    print(f"Starting prior: Beta({alpha}, {beta})")
    print(f"\n{'Flip':>4} {'Outcome':>8} {'Posterior':>15} {'Mean':>8} {'95% CI Width':>12}")
    print("-" * 55)
    
    for i in range(10):
        # Generate observation
        outcome = np.random.binomial(1, true_p)
        
        # Update posterior
        alpha += outcome
        beta += (1 - outcome)
        
        mean = alpha / (alpha + beta)
        ci = stats.beta.ppf([0.025, 0.975], alpha, beta)
        ci_width = ci[1] - ci[0]
        
        outcome_str = "Head" if outcome else "Tail"
        print(f"{i+1:>4} {outcome_str:>8} Beta({alpha:>2},{beta:>2}){mean:>12.4f} {ci_width:>12.4f}")
    
    print(f"\nFinal estimate: {alpha/(alpha+beta):.4f}")
    print("CI gets narrower with more data (uncertainty decreases)")


def example_map_vs_mle():
    """MAP vs MLE comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: MAP vs MLE")
    print("=" * 60)
    
    # Sparse data scenario
    n_flips = 5
    n_heads = 4
    
    print(f"Data: {n_heads}/{n_flips} heads")
    
    # MLE
    mle = n_heads / n_flips
    print(f"\nMLE: {mle:.4f}")
    
    # MAP with different priors
    priors = [(1, 1), (2, 2), (5, 5)]
    
    for alpha, beta in priors:
        alpha_post = alpha + n_heads
        beta_post = beta + (n_flips - n_heads)
        
        map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
        post_mean = alpha_post / (alpha_post + beta_post)
        
        print(f"\nPrior Beta({alpha},{beta}):")
        print(f"  MAP: {map_estimate:.4f}")
        print(f"  Posterior mean: {post_mean:.4f}")
    
    print("\nMAP shrinks toward prior mode")
    print("Equivalent to adding pseudocounts!")


def example_credible_vs_confidence():
    """Credible intervals vs confidence intervals."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Credible vs Confidence Intervals")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Data
    true_mu = 5.0
    sigma = 2.0  # Known
    n = 20
    
    data = np.random.normal(true_mu, sigma, n)
    x_bar = data.mean()
    
    print(f"True μ = {true_mu}")
    print(f"Sample mean = {x_bar:.4f}")
    
    # Frequentist 95% CI
    se = sigma / np.sqrt(n)
    freq_ci = (x_bar - 1.96*se, x_bar + 1.96*se)
    
    print(f"\nFrequentist 95% CI: [{freq_ci[0]:.4f}, {freq_ci[1]:.4f}]")
    print("  Interpretation: If we repeated, 95% of CIs would contain μ")
    print("  CANNOT say: 'μ is in this interval with 95% probability'")
    
    # Bayesian with flat prior (approximately)
    mu_prior, tau_sq_prior = 0, 1000  # Very diffuse
    precision_prior = 1 / tau_sq_prior
    precision_data = n / sigma**2
    
    mu_post = (precision_prior * mu_prior + precision_data * x_bar) / (precision_prior + precision_data)
    tau_sq_post = 1 / (precision_prior + precision_data)
    
    bayes_ci = stats.norm.ppf([0.025, 0.975], mu_post, np.sqrt(tau_sq_post))
    
    print(f"\nBayesian 95% credible interval: [{bayes_ci[0]:.4f}, {bayes_ci[1]:.4f}]")
    print("  CAN say: 'P(μ in interval | data) = 0.95'")
    
    print(f"\nWith flat prior, intervals are nearly identical!")
    print("Difference is in interpretation, not computation")


def example_bayesian_regression():
    """Bayesian linear regression."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Bayesian Linear Regression")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data: y = 2x + 1 + noise
    n = 50
    X = np.random.randn(n)
    true_w = 2.0
    true_b = 1.0
    noise_std = 0.5
    y = true_w * X + true_b + np.random.randn(n) * noise_std
    
    print(f"True model: y = {true_w}x + {true_b}")
    print(f"Noise std: {noise_std}")
    
    # Design matrix
    Phi = np.column_stack([np.ones(n), X])
    
    # Prior: w ~ N(0, alpha^-1 * I)
    alpha = 1.0  # Prior precision
    beta = 1 / noise_std**2  # Noise precision (known)
    
    print(f"\nPrior: w ~ N(0, {1/alpha:.2f}I)")
    
    # Posterior covariance and mean
    S_N_inv = alpha * np.eye(2) + beta * Phi.T @ Phi
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N @ Phi.T @ y
    
    print(f"\nPosterior mean: [{m_N[0]:.4f}, {m_N[1]:.4f}]")
    print(f"  (True: [1.0, 2.0])")
    
    print(f"\nPosterior covariance:\n{S_N.round(4)}")
    
    # 95% credible intervals for weights
    stds = np.sqrt(np.diag(S_N))
    for i, (name, true_val) in enumerate([("Intercept", true_b), ("Slope", true_w)]):
        ci = (m_N[i] - 1.96*stds[i], m_N[i] + 1.96*stds[i])
        print(f"\n{name}:")
        print(f"  Posterior mean: {m_N[i]:.4f}")
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Contains true value ({true_val}): {ci[0] <= true_val <= ci[1]}")


def example_bayesian_prediction():
    """Predictive distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Bayesian Prediction")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Beta-binomial predictive
    alpha_prior, beta_prior = 2, 2
    n_heads, n_flips = 7, 10
    
    alpha_post = alpha_prior + n_heads
    beta_post = beta_prior + (n_flips - n_heads)
    
    print("Predicting outcome of next flip")
    print(f"Posterior: Beta({alpha_post}, {beta_post})")
    
    # Predictive probability of head
    # P(next=head | data) = E[p | data] = alpha_post / (alpha_post + beta_post)
    p_head = alpha_post / (alpha_post + beta_post)
    
    print(f"\nP(next flip = head | data) = {p_head:.4f}")
    print(f"(Posterior mean of p)")
    
    # Compare with MLE prediction
    mle = n_heads / n_flips
    print(f"\nMLE would predict: {mle:.4f}")
    print("Bayesian is more conservative (shrunk toward 0.5)")
    
    # Normal case: predictive distribution
    print("\n" + "-" * 40)
    print("Normal predictive distribution")
    
    # Suppose posterior for mean is N(3.0, 0.5)
    mu_post = 3.0
    sigma_post = np.sqrt(0.5)
    sigma_data = 1.0  # Known observation noise
    
    # Predictive: y_new | data ~ N(mu_post, sigma_data^2 + sigma_post^2)
    sigma_pred = np.sqrt(sigma_data**2 + sigma_post**2)
    
    print(f"\nPosterior on μ: N({mu_post}, {sigma_post**2})")
    print(f"Observation noise: σ = {sigma_data}")
    print(f"\nPredictive: y_new ~ N({mu_post}, {sigma_pred**2:.4f})")
    print(f"Predictive std: {sigma_pred:.4f}")
    print("Includes both parameter uncertainty and observation noise!")


def example_model_comparison():
    """Bayesian model comparison."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Bayesian Model Comparison")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Compare two models for coin
    # M1: Fair coin (p = 0.5)
    # M2: Unfair coin (p has Beta(1,1) prior)
    
    n_heads, n_flips = 8, 10
    
    print(f"Data: {n_heads}/{n_flips} heads")
    print("\nM1: Fair coin (p = 0.5 fixed)")
    print("M2: Biased coin (p ~ Beta(1,1))")
    
    # Evidence for M1: P(data | M1) = binomial(k=8, n=10, p=0.5)
    evidence_m1 = stats.binom.pmf(n_heads, n_flips, 0.5)
    
    # Evidence for M2: P(data | M2) = integral of binomial * beta
    # This is Beta-Binomial: n choose k * B(α+k, β+n-k) / B(α, β)
    from scipy.special import comb, beta as beta_func
    alpha, beta = 1, 1
    evidence_m2 = comb(n_flips, n_heads) * beta_func(alpha + n_heads, beta + n_flips - n_heads) / beta_func(alpha, beta)
    
    print(f"\nEvidence:")
    print(f"  P(data | M1) = {evidence_m1:.6f}")
    print(f"  P(data | M2) = {evidence_m2:.6f}")
    
    # Bayes factor
    bf_21 = evidence_m2 / evidence_m1
    print(f"\nBayes Factor BF(M2/M1) = {bf_21:.4f}")
    
    if bf_21 > 3:
        print("  Moderate evidence for M2 (biased coin)")
    elif bf_21 > 1:
        print("  Weak evidence for M2")
    else:
        print("  Evidence favors M1 (fair coin)")
    
    # Posterior model probabilities (equal priors)
    p_m1_post = evidence_m1 / (evidence_m1 + evidence_m2)
    p_m2_post = evidence_m2 / (evidence_m1 + evidence_m2)
    
    print(f"\nPosterior model probabilities:")
    print(f"  P(M1 | data) = {p_m1_post:.4f}")
    print(f"  P(M2 | data) = {p_m2_post:.4f}")


def example_regularization_prior():
    """Regularization as Bayesian prior."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Regularization as Bayesian Prior")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n = 50
    X = np.random.randn(n, 5)
    true_w = np.array([3, -2, 0, 0, 0])  # Sparse weights
    y = X @ true_w + np.random.randn(n) * 0.5
    
    print("True weights: [3, -2, 0, 0, 0]")
    
    # OLS (no prior / flat prior)
    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Ridge (Gaussian prior)
    lambda_ridge = 1.0
    w_ridge = np.linalg.solve(X.T @ X + lambda_ridge * np.eye(5), X.T @ y)
    
    print(f"\n{'Method':<15} {'w1':>8} {'w2':>8} {'w3':>8} {'w4':>8} {'w5':>8}")
    print("-" * 60)
    print(f"{'True':<15}" + "".join(f"{w:>8.3f}" for w in true_w))
    print(f"{'OLS (flat)':<15}" + "".join(f"{w:>8.3f}" for w in w_ols))
    print(f"{'Ridge (N(0,1))':<15}" + "".join(f"{w:>8.3f}" for w in w_ridge))
    
    print("\nConnection:")
    print("  Ridge: w ~ N(0, 1/λ)  prior")
    print("  MAP with this prior = Ridge solution")
    print("  λ controls prior precision (1/variance)")
    
    # Posterior uncertainty (approximate)
    sigma_sq = 0.25  # Known noise variance
    S_post = np.linalg.inv(X.T @ X / sigma_sq + lambda_ridge * np.eye(5))
    
    print(f"\nPosterior std for each weight (Ridge/Bayesian):")
    print("  " + "  ".join(f"w{i+1}: {np.sqrt(S_post[i,i]):.3f}" for i in range(5)))


def example_mcmc_concept():
    """Conceptual MCMC demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: MCMC Concept (Metropolis-Hastings)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Target: posterior for p given data
    # Beta(9, 5) posterior from earlier example
    alpha_post, beta_post = 9, 5
    
    print(f"Target posterior: Beta({alpha_post}, {beta_post})")
    print(f"True posterior mean: {alpha_post/(alpha_post + beta_post):.4f}")
    
    # Simple Metropolis-Hastings
    def log_posterior(p):
        if p <= 0 or p >= 1:
            return -np.inf
        return stats.beta.logpdf(p, alpha_post, beta_post)
    
    # MCMC sampling
    n_samples = 10000
    samples = np.zeros(n_samples)
    samples[0] = 0.5  # Initial value
    proposal_std = 0.1
    
    accepted = 0
    for i in range(1, n_samples):
        # Propose
        proposal = samples[i-1] + np.random.normal(0, proposal_std)
        
        # Accept/reject
        log_alpha = log_posterior(proposal) - log_posterior(samples[i-1])
        
        if np.log(np.random.rand()) < log_alpha:
            samples[i] = proposal
            accepted += 1
        else:
            samples[i] = samples[i-1]
    
    # Discard burn-in
    burn_in = 1000
    samples = samples[burn_in:]
    
    print(f"\nMCMC results ({n_samples} samples, {burn_in} burn-in):")
    print(f"  Acceptance rate: {accepted/n_samples:.2%}")
    print(f"  Estimated mean: {samples.mean():.4f}")
    print(f"  Estimated std: {samples.std():.4f}")
    print(f"  95% credible interval: [{np.percentile(samples, 2.5):.4f}, {np.percentile(samples, 97.5):.4f}]")
    
    # Compare with analytical
    true_mean = alpha_post / (alpha_post + beta_post)
    true_std = np.sqrt(alpha_post * beta_post / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))
    print(f"\nAnalytical:")
    print(f"  True mean: {true_mean:.4f}")
    print(f"  True std: {true_std:.4f}")


def example_prior_predictive():
    """Prior predictive checking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Prior Predictive Checking")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("Prior predictive: What data would our prior predict?")
    print("Use to check if prior is sensible\n")
    
    # Model: height ~ N(μ, σ²)
    # Prior: μ ~ N(170, 10²), σ ~ HalfNormal(20)
    
    n_prior_samples = 5000
    
    # Prior predictive distribution
    mu_samples = np.random.normal(170, 10, n_prior_samples)
    sigma_samples = np.abs(np.random.normal(0, 20, n_prior_samples))
    
    # Generate one data point from each prior sample
    y_prior_pred = np.random.normal(mu_samples, sigma_samples)
    
    print("Model: height ~ N(μ, σ²)")
    print("Prior: μ ~ N(170, 10²), σ ~ HalfNormal(20)")
    
    print(f"\nPrior predictive distribution for height:")
    print(f"  Mean: {y_prior_pred.mean():.1f} cm")
    print(f"  Std: {y_prior_pred.std():.1f} cm")
    print(f"  95% range: [{np.percentile(y_prior_pred, 2.5):.1f}, {np.percentile(y_prior_pred, 97.5):.1f}] cm")
    
    # Check for unreasonable predictions
    n_negative = (y_prior_pred < 0).sum()
    n_over_300 = (y_prior_pred > 300).sum()
    
    print(f"\n  P(height < 0): {n_negative/n_prior_samples:.4f}")
    print(f"  P(height > 300cm): {n_over_300/n_prior_samples:.4f}")
    
    print("\nPrior seems reasonable if these probabilities are low!")


if __name__ == "__main__":
    example_beta_binomial()
    example_normal_normal()
    example_prior_influence()
    example_sequential_updating()
    example_map_vs_mle()
    example_credible_vs_confidence()
    example_bayesian_regression()
    example_bayesian_prediction()
    example_model_comparison()
    example_regularization_prior()
    example_mcmc_concept()
    example_prior_predictive()
