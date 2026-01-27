"""
Estimation Theory - Examples
============================
Practical demonstrations of statistical estimation.
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def example_bias_variance():
    """Demonstrate bias-variance tradeoff."""
    print("=" * 60)
    print("EXAMPLE 1: Bias-Variance Tradeoff")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True parameter
    true_sigma2 = 25.0  # True variance
    n_samples = 10
    n_simulations = 10000
    
    # Two estimators for variance
    # 1. MLE: divide by n (biased)
    # 2. Unbiased: divide by n-1
    
    mle_estimates = []
    unbiased_estimates = []
    
    for _ in range(n_simulations):
        data = np.random.normal(0, np.sqrt(true_sigma2), n_samples)
        mle_estimates.append(data.var(ddof=0))  # divide by n
        unbiased_estimates.append(data.var(ddof=1))  # divide by n-1
    
    mle_estimates = np.array(mle_estimates)
    unbiased_estimates = np.array(unbiased_estimates)
    
    print(f"\nTrue σ² = {true_sigma2}")
    print(f"Sample size n = {n_samples}")
    print(f"Simulations = {n_simulations}")
    
    print("\nMLE Estimator (÷n):")
    mle_bias = mle_estimates.mean() - true_sigma2
    mle_var = mle_estimates.var()
    mle_mse = mle_bias**2 + mle_var
    print(f"  E[σ̂²] = {mle_estimates.mean():.4f}")
    print(f"  Bias = {mle_bias:.4f}")
    print(f"  Variance = {mle_var:.4f}")
    print(f"  MSE = {mle_mse:.4f}")
    
    print("\nUnbiased Estimator (÷(n-1)):")
    unb_bias = unbiased_estimates.mean() - true_sigma2
    unb_var = unbiased_estimates.var()
    unb_mse = unb_bias**2 + unb_var
    print(f"  E[σ̂²] = {unbiased_estimates.mean():.4f}")
    print(f"  Bias = {unb_bias:.4f}")
    print(f"  Variance = {unb_var:.4f}")
    print(f"  MSE = {unb_mse:.4f}")
    
    print("\nConclusion:")
    print(f"  MLE has bias ≈ -σ²/n = -{true_sigma2/n_samples:.2f}")
    print(f"  Unbiased has zero bias but higher variance")
    print(f"  MLE actually has lower MSE! (bias-variance tradeoff)")


def example_mle_normal():
    """MLE for normal distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MLE for Normal Distribution")
    print("=" * 60)
    
    np.random.seed(42)
    
    # True parameters
    true_mu = 5.0
    true_sigma = 2.0
    n = 100
    
    data = np.random.normal(true_mu, true_sigma, n)
    
    print(f"\nTrue parameters: μ = {true_mu}, σ = {true_sigma}")
    print(f"Sample size: n = {n}")
    
    # MLE estimates
    mu_mle = data.mean()
    sigma2_mle = data.var(ddof=0)  # MLE uses n
    
    print(f"\nMLE estimates:")
    print(f"  μ̂ = {mu_mle:.4f} (true: {true_mu})")
    print(f"  σ̂² = {sigma2_mle:.4f} (true: {true_sigma**2})")
    print(f"  σ̂ = {np.sqrt(sigma2_mle):.4f} (true: {true_sigma})")
    
    # Log-likelihood at MLE
    def neg_log_likelihood(params, data):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        return -np.sum(stats.norm.logpdf(data, mu, sigma))
    
    ll_at_mle = -neg_log_likelihood([mu_mle, np.sqrt(sigma2_mle)], data)
    print(f"\nLog-likelihood at MLE: {ll_at_mle:.4f}")


def example_mle_bernoulli():
    """MLE for Bernoulli/Binomial."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: MLE for Bernoulli")
    print("=" * 60)
    
    np.random.seed(42)
    
    true_p = 0.3
    n = 200
    
    data = np.random.binomial(1, true_p, n)
    
    print(f"\nTrue p = {true_p}")
    print(f"Sample: {n} Bernoulli trials")
    print(f"Successes: {data.sum()}")
    
    # MLE for p
    p_mle = data.mean()
    
    print(f"\nMLE: p̂ = {p_mle:.4f}")
    
    # Fisher Information
    fisher_info = 1 / (p_mle * (1 - p_mle))
    se = 1 / np.sqrt(n * fisher_info)
    
    print(f"\nFisher Information: I(p) = 1/(p(1-p)) = {fisher_info:.4f}")
    print(f"Standard Error: {se:.4f}")
    print(f"95% CI: [{p_mle - 1.96*se:.4f}, {p_mle + 1.96*se:.4f}]")


def example_mle_exponential():
    """MLE for exponential distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: MLE for Exponential Distribution")
    print("=" * 60)
    
    np.random.seed(42)
    
    # X ~ Exp(λ) with E[X] = 1/λ
    true_lambda = 0.5
    n = 100
    
    data = np.random.exponential(1/true_lambda, n)
    
    print(f"\nTrue λ = {true_lambda} (rate)")
    print(f"True E[X] = 1/λ = {1/true_lambda}")
    print(f"Sample size: n = {n}")
    
    # MLE derivation:
    # L(λ) = λ^n exp(-λ Σxᵢ)
    # ℓ(λ) = n log(λ) - λ Σxᵢ
    # ∂ℓ/∂λ = n/λ - Σxᵢ = 0
    # λ̂ = n / Σxᵢ = 1/x̄
    
    lambda_mle = 1 / data.mean()
    
    print(f"\nSample mean: {data.mean():.4f}")
    print(f"MLE: λ̂ = 1/x̄ = {lambda_mle:.4f}")
    print(f"Error: {abs(lambda_mle - true_lambda):.4f}")


def example_method_of_moments():
    """Method of moments estimation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Method of Moments - Gamma Distribution")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Gamma(α, β) with E[X] = α/β, Var(X) = α/β²
    true_alpha = 3.0
    true_beta = 0.5
    n = 500
    
    data = np.random.gamma(true_alpha, 1/true_beta, n)
    
    print(f"\nTrue parameters: α = {true_alpha}, β = {true_beta}")
    print(f"Sample size: n = {n}")
    
    # Sample moments
    m1 = data.mean()  # x̄
    m2 = (data**2).mean()  # x̄²
    
    print(f"\nSample moments:")
    print(f"  m₁ = {m1:.4f}")
    print(f"  m₂ = {m2:.4f}")
    
    # Method of moments:
    # m₁ = α/β
    # m₂ - m₁² = α/β² (variance)
    # Solving: β = m₁/(m₂ - m₁²), α = m₁β
    
    sample_var = m2 - m1**2
    beta_mom = m1 / sample_var
    alpha_mom = m1 * beta_mom
    
    print(f"\nMethod of Moments estimates:")
    print(f"  α̂ = {alpha_mom:.4f} (true: {true_alpha})")
    print(f"  β̂ = {beta_mom:.4f} (true: {true_beta})")
    
    # Compare with MLE (scipy)
    alpha_mle, _, scale_mle = stats.gamma.fit(data, floc=0)
    beta_mle = 1 / scale_mle
    
    print(f"\nMLE estimates (scipy):")
    print(f"  α̂ = {alpha_mle:.4f}")
    print(f"  β̂ = {beta_mle:.4f}")


def example_mle_vs_map():
    """Compare MLE and MAP estimation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: MLE vs MAP Estimation")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Estimate success probability with small sample
    true_p = 0.7
    n = 10
    
    data = np.random.binomial(1, true_p, n)
    successes = data.sum()
    
    print(f"\nTrue p = {true_p}")
    print(f"Sample: {successes} successes in {n} trials")
    
    # MLE
    p_mle = successes / n
    
    print(f"\nMLE: p̂ = {p_mle:.4f}")
    
    # MAP with Beta prior
    # Prior: Beta(α₀, β₀)
    # Posterior: Beta(α₀ + successes, β₀ + failures)
    # MAP = (α₀ + successes - 1) / (α₀ + β₀ + n - 2)
    
    # Weak prior (Beta(2, 2))
    alpha0, beta0 = 2, 2
    alpha_post = alpha0 + successes
    beta_post = beta0 + (n - successes)
    p_map_weak = (alpha_post - 1) / (alpha_post + beta_post - 2)
    
    print(f"\nMAP with weak prior Beta(2,2):")
    print(f"  Posterior: Beta({alpha_post}, {beta_post})")
    print(f"  p̂_MAP = {p_map_weak:.4f}")
    
    # Strong prior towards 0.5 (Beta(10, 10))
    alpha0, beta0 = 10, 10
    alpha_post = alpha0 + successes
    beta_post = beta0 + (n - successes)
    p_map_strong = (alpha_post - 1) / (alpha_post + beta_post - 2)
    
    print(f"\nMAP with strong prior Beta(10,10):")
    print(f"  Posterior: Beta({alpha_post}, {beta_post})")
    print(f"  p̂_MAP = {p_map_strong:.4f}")
    
    print("\nObservation:")
    print("  MLE purely driven by data")
    print("  MAP pulls toward prior (regularization effect)")


def example_fisher_information():
    """Fisher information and Cramér-Rao bound."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Fisher Information and Cramér-Rao Bound")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Normal distribution, known variance
    true_mu = 5.0
    true_sigma = 2.0
    
    print(f"X ~ N(μ, σ²) with σ = {true_sigma} known")
    print(f"Estimating μ")
    
    # Fisher Information for μ in Normal
    # I(μ) = 1/σ²
    fisher_info = 1 / true_sigma**2
    print(f"\nFisher Information: I(μ) = 1/σ² = {fisher_info:.4f}")
    
    # Cramér-Rao lower bound
    for n in [10, 50, 100, 500]:
        crb = 1 / (n * fisher_info)
        print(f"\nn = {n}:")
        print(f"  Cramér-Rao bound: Var(μ̂) ≥ {crb:.6f}")
        print(f"  SE lower bound: {np.sqrt(crb):.6f}")
        
        # Simulate to verify
        estimates = [np.random.normal(true_mu, true_sigma, n).mean() 
                     for _ in range(5000)]
        actual_var = np.var(estimates)
        print(f"  Actual Var(x̄): {actual_var:.6f}")
        print(f"  MLE achieves bound: {np.isclose(actual_var, crb, rtol=0.1)}")


def example_consistency():
    """Demonstrate consistency of MLE."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Consistency of MLE")
    print("=" * 60)
    
    np.random.seed(42)
    
    true_lambda = 2.0
    sample_sizes = [10, 50, 100, 500, 1000, 5000]
    n_simulations = 1000
    
    print(f"\nPoisson(λ) with true λ = {true_lambda}")
    print(f"MLE: λ̂ = x̄")
    
    print(f"\n{'n':>6} {'Mean(λ̂)':>10} {'Std(λ̂)':>10} {'MSE':>10}")
    print("-" * 40)
    
    for n in sample_sizes:
        mle_estimates = []
        for _ in range(n_simulations):
            data = np.random.poisson(true_lambda, n)
            mle_estimates.append(data.mean())
        
        mle_estimates = np.array(mle_estimates)
        mean_mle = mle_estimates.mean()
        std_mle = mle_estimates.std()
        mse = ((mle_estimates - true_lambda)**2).mean()
        
        print(f"{n:>6} {mean_mle:>10.4f} {std_mle:>10.4f} {mse:>10.6f}")
    
    print("\nAs n → ∞: Mean → λ, Std → 0, MSE → 0")
    print("This is consistency!")


def example_mle_numerical():
    """Numerical MLE optimization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Numerical MLE (Beta Distribution)")
    print("=" * 60)
    
    np.random.seed(42)
    
    true_alpha, true_beta = 2.5, 5.0
    n = 200
    
    data = np.random.beta(true_alpha, true_beta, n)
    
    print(f"\nBeta({true_alpha}, {true_beta}) - no closed-form MLE")
    print(f"Sample size: n = {n}")
    
    # Negative log-likelihood
    def neg_log_likelihood(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e10
        return -np.sum(stats.beta.logpdf(data, a, b))
    
    # Optimize
    from scipy.optimize import minimize
    
    initial_guess = [1.0, 1.0]
    result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B',
                      bounds=[(0.01, None), (0.01, None)])
    
    alpha_mle, beta_mle = result.x
    
    print(f"\nNumerical MLE:")
    print(f"  α̂ = {alpha_mle:.4f} (true: {true_alpha})")
    print(f"  β̂ = {beta_mle:.4f} (true: {true_beta})")
    
    # Compare with scipy's built-in
    alpha_scipy, beta_scipy, _, _ = stats.beta.fit(data, floc=0, fscale=1)
    print(f"\nSciPy fit:")
    print(f"  α̂ = {alpha_scipy:.4f}")
    print(f"  β̂ = {beta_scipy:.4f}")


def example_ml_connection():
    """Connection between MLE and ML loss functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: MLE and ML Loss Functions")
    print("=" * 60)
    
    np.random.seed(42)
    
    print("\n1. Regression: MSE ↔ Gaussian MLE")
    print("-" * 40)
    
    # y = 2x + 1 + noise
    n = 100
    X = np.random.randn(n, 1)
    y = 2 * X.squeeze() + 1 + np.random.randn(n) * 0.5
    
    # OLS (minimizes MSE)
    X_aug = np.column_stack([np.ones(n), X])
    beta_ols = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    
    print(f"True: y = 2x + 1")
    print(f"OLS: y = {beta_ols[1]:.4f}x + {beta_ols[0]:.4f}")
    print("OLS = MLE under Gaussian noise assumption")
    
    print("\n2. Classification: Cross-Entropy ↔ Bernoulli MLE")
    print("-" * 40)
    
    # Binary classification
    n = 200
    true_w = 2.0
    X = np.random.randn(n)
    p_true = 1 / (1 + np.exp(-true_w * X))
    y = (np.random.rand(n) < p_true).astype(float)
    
    def neg_log_likelihood_logistic(w):
        p = 1 / (1 + np.exp(-w * X))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    result = minimize(neg_log_likelihood_logistic, [0.0])
    w_mle = result.x[0]
    
    print(f"True w = {true_w}")
    print(f"MLE w = {w_mle:.4f}")
    print("Minimizing cross-entropy = MLE for logistic regression")
    
    print("\n3. Regularization ↔ MAP Estimation")
    print("-" * 40)
    
    # L2 regularization = Gaussian prior
    lambda_reg = 1.0
    
    # MAP with Gaussian prior on w
    def neg_log_posterior(w, lambda_reg):
        nll = neg_log_likelihood_logistic(w)
        prior = lambda_reg * w**2 / 2  # -log of Gaussian prior
        return nll + prior
    
    result = minimize(neg_log_posterior, [0.0], args=(lambda_reg,))
    w_map = result.x[0]
    
    print(f"MLE w = {w_mle:.4f}")
    print(f"MAP w (λ={lambda_reg}) = {w_map:.4f}")
    print("Regularization shrinks estimate toward 0 (prior mean)")


def example_ridge_as_map():
    """Ridge regression as MAP estimation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Ridge Regression as MAP")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n, p = 50, 20
    X = np.random.randn(n, p)
    true_beta = np.array([3, -2, 1, 0.5] + [0] * (p - 4))
    y = X @ true_beta + np.random.randn(n) * 0.5
    
    print(f"Data: n={n}, p={p}")
    print(f"True β has only 4 non-zero coefficients")
    
    # OLS (MLE)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Ridge (MAP with Gaussian prior)
    lambda_values = [0.1, 1.0, 10.0]
    
    print(f"\n{'Method':<15} {'||β||²':<10} {'||β - β_true||²':<15}")
    print("-" * 45)
    print(f"{'OLS (MLE)':<15} {np.sum(beta_ols**2):<10.4f} {np.sum((beta_ols - true_beta)**2):<15.4f}")
    
    for lam in lambda_values:
        # Ridge: (X'X + λI)^(-1) X'y
        beta_ridge = np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
        print(f"{'Ridge λ=' + str(lam):<15} {np.sum(beta_ridge**2):<10.4f} {np.sum((beta_ridge - true_beta)**2):<15.4f}")
    
    print("\nRidge shrinks coefficients → MAP with N(0, 1/λ) prior")


def example_mle_invariance():
    """MLE invariance property."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: MLE Invariance Property")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Exponential with rate λ
    true_lambda = 0.5
    n = 100
    data = np.random.exponential(1/true_lambda, n)
    
    lambda_mle = 1 / data.mean()
    
    print(f"X ~ Exp(λ) with λ = {true_lambda}")
    print(f"\nMLE of λ: λ̂ = {lambda_mle:.4f}")
    
    # Want to estimate mean = 1/λ
    mean_mle = 1 / lambda_mle  # Invariance!
    
    print(f"\nBy invariance, MLE of 1/λ = 1/λ̂ = {mean_mle:.4f}")
    print(f"This equals sample mean: {data.mean():.4f}")
    
    # Want to estimate P(X > 2) = exp(-2λ)
    prob_true = np.exp(-2 * true_lambda)
    prob_mle = np.exp(-2 * lambda_mle)
    
    print(f"\nP(X > 2) = exp(-2λ)")
    print(f"True: {prob_true:.4f}")
    print(f"MLE: exp(-2λ̂) = {prob_mle:.4f}")
    
    # Verify with empirical
    prob_empirical = (data > 2).mean()
    print(f"Empirical: {prob_empirical:.4f}")


if __name__ == "__main__":
    example_bias_variance()
    example_mle_normal()
    example_mle_bernoulli()
    example_mle_exponential()
    example_method_of_moments()
    example_mle_vs_map()
    example_fisher_information()
    example_consistency()
    example_mle_numerical()
    example_ml_connection()
    example_ridge_as_map()
    example_mle_invariance()
