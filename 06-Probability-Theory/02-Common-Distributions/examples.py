"""
Common Probability Distributions - Examples
==========================================
Practical demonstrations of probability distributions.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')


def example_bernoulli_binomial():
    """Bernoulli and Binomial distributions."""
    print("=" * 60)
    print("EXAMPLE 1: Bernoulli and Binomial")
    print("=" * 60)
    
    p = 0.7
    
    print(f"Bernoulli(p={p}):")
    print(f"  P(X=1) = {p}")
    print(f"  P(X=0) = {1-p}")
    print(f"  E[X] = {p}")
    print(f"  Var(X) = {p*(1-p):.4f}")
    
    # Binomial as sum of Bernoulli
    n = 20
    print(f"\nBinomial(n={n}, p={p}):")
    print("  'Sum of n independent Bernoulli trials'")
    
    # Theoretical
    mean_th = n * p
    var_th = n * p * (1-p)
    print(f"  E[X] = np = {mean_th}")
    print(f"  Var(X) = np(1-p) = {var_th:.2f}")
    
    # Simulation
    np.random.seed(42)
    samples = np.random.binomial(n, p, size=10000)
    print(f"\n  Simulation (10000 samples):")
    print(f"    Mean = {samples.mean():.3f}")
    print(f"    Var  = {samples.var():.3f}")
    
    # PMF visualization
    print("\n  PMF:")
    for k in range(n+1):
        prob = stats.binom.pmf(k, n, p)
        if prob > 0.01:
            bar = '█' * int(prob * 40)
            print(f"    P(X={k:2d}) = {prob:.4f} {bar}")


def example_poisson():
    """Poisson distribution and its relationship to Binomial."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Poisson Distribution")
    print("=" * 60)
    
    lam = 5.0
    
    print(f"Poisson(λ={lam}):")
    print(f"  'λ = average number of events per unit time'")
    
    print("\n  PMF:")
    for k in range(16):
        prob = stats.poisson.pmf(k, lam)
        if prob > 0.005:
            bar = '█' * int(prob * 40)
            print(f"    P(X={k:2d}) = {prob:.4f} {bar}")
    
    print(f"\n  E[X] = λ = {lam}")
    print(f"  Var(X) = λ = {lam}")
    
    # Poisson as limit of Binomial
    print("\n--- Poisson as Binomial limit ---")
    print(f"  Binomial(n, p=λ/n) → Poisson(λ) as n → ∞")
    
    k = 5
    poisson_prob = stats.poisson.pmf(k, lam)
    
    print(f"\n  P(X={k}) comparison:")
    for n in [10, 50, 100, 500, 1000]:
        p = lam / n
        binom_prob = stats.binom.pmf(k, n, p)
        print(f"    Binomial({n:4d}, {p:.4f}): {binom_prob:.6f}")
    print(f"    Poisson({lam}):           {poisson_prob:.6f}")


def example_geometric():
    """Geometric distribution - trials until first success."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Geometric Distribution")
    print("=" * 60)
    
    p = 0.2
    
    print(f"Geometric(p={p}):")
    print(f"  'Number of trials until first success'")
    print(f"  P(success per trial) = {p}")
    
    print("\n  PMF: P(X=k) = (1-p)^(k-1) × p")
    for k in range(1, 16):
        prob = stats.geom.pmf(k, p)
        if prob > 0.005:
            bar = '█' * int(prob * 40)
            print(f"    P(X={k:2d}) = {prob:.4f} {bar}")
    
    print(f"\n  E[X] = 1/p = {1/p:.2f}")
    print(f"  Var(X) = (1-p)/p² = {(1-p)/p**2:.2f}")
    
    # Memoryless property
    print("\n--- Memoryless Property ---")
    np.random.seed(42)
    samples = np.random.geometric(p, size=100000)
    
    s, t = 5, 3
    # P(X > s+t | X > s) should equal P(X > t)
    P_gt_spt_given_gt_s = (samples[samples > s] > s + t).mean()
    P_gt_t = (samples > t).mean()
    
    print(f"  P(X > {s+t} | X > {s}) = {P_gt_spt_given_gt_s:.4f}")
    print(f"  P(X > {t}) = {P_gt_t:.4f}")
    print("  These should be approximately equal (memoryless!)")


def example_normal():
    """Normal (Gaussian) distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Normal Distribution")
    print("=" * 60)
    
    mu, sigma = 100, 15
    
    print(f"Normal(μ={mu}, σ²={sigma**2}):")
    print(f"  (Example: IQ scores)")
    
    print("\n  Properties:")
    print(f"    Mean = μ = {mu}")
    print(f"    Variance = σ² = {sigma**2}")
    print(f"    Std Dev = σ = {sigma}")
    
    # Probability calculations
    print("\n  Probability regions:")
    P_below_100 = stats.norm.cdf(100, mu, sigma)
    P_above_130 = 1 - stats.norm.cdf(130, mu, sigma)
    P_between = stats.norm.cdf(115, mu, sigma) - stats.norm.cdf(85, mu, sigma)
    
    print(f"    P(X < 100) = {P_below_100:.4f}")
    print(f"    P(X > 130) = {P_above_130:.4f}")
    print(f"    P(85 < X < 115) = {P_between:.4f}")
    
    # 68-95-99.7 rule
    print("\n--- 68-95-99.7 Rule (Standard Normal) ---")
    for k in [1, 2, 3]:
        prob = stats.norm.cdf(k) - stats.norm.cdf(-k)
        print(f"    P(|Z| < {k}) = {prob:.4f} ({prob*100:.1f}%)")
    
    # Standardization
    print("\n--- Standardization ---")
    x = 130
    z = (x - mu) / sigma
    print(f"  X = {x}")
    print(f"  Z = (X - μ)/σ = ({x} - {mu})/{sigma} = {z:.2f}")
    print(f"  P(X > {x}) = P(Z > {z}) = {1 - stats.norm.cdf(z):.4f}")


def example_exponential():
    """Exponential distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Exponential Distribution")
    print("=" * 60)
    
    lam = 0.5  # Rate parameter
    
    print(f"Exponential(λ={lam}):")
    print(f"  'Time between events that occur at rate {lam} per unit time'")
    print(f"  Mean wait time = 1/λ = {1/lam}")
    
    print("\n  PDF: f(x) = λe^(-λx)")
    print("  CDF: F(x) = 1 - e^(-λx)")
    
    # Probabilities
    print("\n  Probabilities:")
    for t in [1, 2, 4]:
        prob_less = stats.expon.cdf(t, scale=1/lam)
        print(f"    P(X < {t}) = {prob_less:.4f}")
    
    # Relation to Poisson
    print("\n--- Relation to Poisson ---")
    print("  If events occur as Poisson process with rate λ,")
    print("  then time between events is Exponential(λ)")
    
    # Simulation
    np.random.seed(42)
    arrivals = np.random.exponential(1/lam, size=1000)
    
    print(f"\n  Simulation (1000 inter-arrival times):")
    print(f"    Mean = {arrivals.mean():.3f} (expected: {1/lam})")
    print(f"    Variance = {arrivals.var():.3f} (expected: {1/lam**2})")


def example_gamma():
    """Gamma distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Gamma Distribution")
    print("=" * 60)
    
    print("Gamma(α, β): α = shape, β = rate")
    print("  PDF: f(x) = β^α x^(α-1) e^(-βx) / Γ(α)")
    
    print("\n--- Special Cases ---")
    print("  Gamma(1, β) = Exponential(β)")
    print("  Gamma(k, 1/2) = Chi-squared(2k)")
    
    # Sum of exponentials
    print("\n--- Sum of Exponentials ---")
    print("  Sum of n Exponential(λ) = Gamma(n, λ)")
    
    np.random.seed(42)
    n = 5
    lam = 2.0
    
    # Sum of exponentials
    exp_samples = np.random.exponential(1/lam, size=(10000, n))
    sum_samples = exp_samples.sum(axis=1)
    
    # Compare to Gamma
    gamma_samples = np.random.gamma(n, 1/lam, size=10000)
    
    print(f"\n  Sum of {n} Exp({lam}):")
    print(f"    Mean: {sum_samples.mean():.3f} (theory: {n/lam:.2f})")
    print(f"    Var:  {sum_samples.var():.3f} (theory: {n/lam**2:.2f})")
    
    print(f"\n  Gamma({n}, {lam}):")
    print(f"    Mean: {gamma_samples.mean():.3f}")
    print(f"    Var:  {gamma_samples.var():.3f}")


def example_beta():
    """Beta distribution - distribution on [0,1]."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Beta Distribution")
    print("=" * 60)
    
    print("Beta(α, β): Distribution on [0, 1]")
    print("  Useful as prior for probabilities!")
    
    print("\n--- Different Shapes ---")
    shapes = [(1, 1), (2, 2), (5, 1), (1, 5), (0.5, 0.5), (5, 5)]
    
    for alpha, beta in shapes:
        mean = alpha / (alpha + beta)
        mode = (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else None
        var = (alpha * beta) / ((alpha + beta)**2 * (alpha + beta + 1))
        
        mode_str = f"{mode:.3f}" if mode is not None else "N/A"
        print(f"  Beta({alpha}, {beta}): Mean={mean:.3f}, Mode={mode_str}, Var={var:.4f}")
    
    # Beta-Binomial conjugacy
    print("\n--- Beta-Binomial Conjugacy ---")
    print("  Prior: p ~ Beta(α, β)")
    print("  Likelihood: k|p ~ Binomial(n, p)")
    print("  Posterior: p|k ~ Beta(α + k, β + n - k)")
    
    alpha_prior, beta_prior = 2, 2
    n, k = 10, 7  # 7 successes in 10 trials
    
    alpha_post = alpha_prior + k
    beta_post = beta_prior + n - k
    
    print(f"\n  Prior: Beta({alpha_prior}, {beta_prior})")
    print(f"  Data: {k} successes in {n} trials")
    print(f"  Posterior: Beta({alpha_post}, {beta_post})")
    
    prior_mean = alpha_prior / (alpha_prior + beta_prior)
    post_mean = alpha_post / (alpha_post + beta_post)
    mle = k / n
    
    print(f"\n  Prior mean: {prior_mean:.3f}")
    print(f"  MLE: {mle:.3f}")
    print(f"  Posterior mean: {post_mean:.3f} (shrunk toward prior)")


def example_student_t():
    """Student's t-distribution - heavy tails."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Student's t-Distribution")
    print("=" * 60)
    
    print("Student's t with ν degrees of freedom:")
    print("  - Heavier tails than Normal")
    print("  - As ν → ∞, t → N(0,1)")
    
    print("\n--- Tail Probabilities ---")
    x = 3.0
    normal_prob = 2 * (1 - stats.norm.cdf(x))
    
    print(f"  P(|Z| > {x}) for different distributions:")
    print(f"    Normal:    {normal_prob:.6f}")
    
    for nu in [1, 3, 5, 10, 30]:
        t_prob = 2 * (1 - stats.t.cdf(x, nu))
        print(f"    t(ν={nu:2d}):  {t_prob:.6f}")
    
    # t(1) is Cauchy - no mean!
    print("\n--- Special Cases ---")
    print("  t(ν=1) = Cauchy distribution (no mean exists!)")
    print("  t(ν=2): Var = ∞")
    print("  t(ν>2): Var = ν/(ν-2)")
    
    # Robust regression
    print("\n--- Application: Robust Regression ---")
    np.random.seed(42)
    n = 100
    x = np.random.randn(n)
    y_clean = 2 * x + 1 + 0.5 * np.random.randn(n)
    
    # Add outliers
    y_outlier = y_clean.copy()
    y_outlier[:5] += 10  # 5 outliers
    
    # OLS
    coeffs_ols = np.polyfit(x, y_outlier, 1)
    
    print(f"  True slope: 2.0")
    print(f"  OLS slope (with outliers): {coeffs_ols[0]:.3f}")
    print("  (t-distribution errors would be more robust!)")


def example_multivariate_normal():
    """Multivariate Normal distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Multivariate Normal")
    print("=" * 60)
    
    print("N(μ, Σ): μ = mean vector, Σ = covariance matrix")
    
    mu = np.array([1, 2])
    Sigma = np.array([[2, 0.8],
                      [0.8, 1]])
    
    print(f"\n  μ = {mu}")
    print(f"  Σ = {Sigma[0]}")
    print(f"      {Sigma[1]}")
    
    # Correlation
    corr = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
    print(f"\n  Correlation: ρ = {corr:.3f}")
    
    # Marginals
    print("\n--- Marginal Distributions ---")
    print(f"  X₁ ~ N({mu[0]}, {Sigma[0,0]})")
    print(f"  X₂ ~ N({mu[1]}, {Sigma[1,1]})")
    
    # Conditional
    print("\n--- Conditional Distribution ---")
    print("  X₁|X₂=x₂ is Normal with:")
    
    x2_given = 3.0
    mu_cond = mu[0] + Sigma[0,1]/Sigma[1,1] * (x2_given - mu[1])
    var_cond = Sigma[0,0] - Sigma[0,1]**2/Sigma[1,1]
    
    print(f"  Given X₂ = {x2_given}:")
    print(f"    E[X₁|X₂] = {mu_cond:.3f}")
    print(f"    Var(X₁|X₂) = {var_cond:.3f}")
    
    # Sample
    np.random.seed(42)
    samples = np.random.multivariate_normal(mu, Sigma, size=1000)
    
    print(f"\n--- Simulation (1000 samples) ---")
    print(f"  Sample mean: {samples.mean(axis=0).round(3)}")
    print(f"  Sample cov: {np.cov(samples.T).round(3)}")


def example_dirichlet():
    """Dirichlet distribution - distribution on simplex."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Dirichlet Distribution")
    print("=" * 60)
    
    print("Dirichlet(α₁, ..., αₖ): Distribution on probability vectors")
    print("  Σᵢ xᵢ = 1, xᵢ ≥ 0")
    
    print("\n--- Conjugate prior for Categorical ---")
    print("  Prior: θ ~ Dirichlet(α)")
    print("  Likelihood: counts ~ Multinomial(θ)")
    print("  Posterior: θ ~ Dirichlet(α + counts)")
    
    # Example: 3-sided die
    print("\n--- Example: Estimating biased die ---")
    alpha_prior = np.array([1, 1, 1])  # Uniform prior
    counts = np.array([10, 15, 25])    # Observed counts
    alpha_post = alpha_prior + counts
    
    print(f"  Prior: Dirichlet({alpha_prior})")
    print(f"  Observed counts: {counts}")
    print(f"  Posterior: Dirichlet({alpha_post})")
    
    prior_mean = alpha_prior / alpha_prior.sum()
    post_mean = alpha_post / alpha_post.sum()
    mle = counts / counts.sum()
    
    print(f"\n  Prior mean: {prior_mean.round(3)}")
    print(f"  MLE: {mle.round(3)}")
    print(f"  Posterior mean: {post_mean.round(3)}")
    
    # Sample from Dirichlet
    np.random.seed(42)
    samples = np.random.dirichlet(alpha_post, size=5)
    
    print(f"\n  Posterior samples:")
    for i, sample in enumerate(samples):
        print(f"    {sample.round(3)}")


def example_distribution_selection():
    """Guide to selecting distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Distribution Selection Guide")
    print("=" * 60)
    
    print("Decision tree for choosing distributions:")
    print("""
    Data type?
    │
    ├── Discrete
    │   ├── Binary (0/1) → Bernoulli
    │   │   └── Sum of n trials → Binomial
    │   ├── Count (rare events) → Poisson  
    │   ├── Trials until success → Geometric
    │   └── K categories → Categorical/Multinomial
    │
    └── Continuous
        ├── Bounded [0,1]
        │   └── Probability values → Beta
        │
        ├── Bounded [a,b]
        │   └── No preference → Uniform
        │
        ├── Positive only (x > 0)
        │   ├── Wait times → Exponential
        │   ├── Flexible shape → Gamma
        │   └── Multiplicative → Log-normal
        │
        └── Unbounded (all real)
            ├── Light tails → Normal
            └── Heavy tails → Student's t
    """)
    
    print("ML Application Examples:")
    print("  - Classification probabilities: Bernoulli, Categorical")
    print("  - Regression errors: Normal, Student's t")
    print("  - Bayesian priors: Beta, Gamma, Dirichlet")
    print("  - Latent variables: Normal (VAEs), Categorical (GMMs)")


if __name__ == "__main__":
    example_bernoulli_binomial()
    example_poisson()
    example_geometric()
    example_normal()
    example_exponential()
    example_gamma()
    example_beta()
    example_student_t()
    example_multivariate_normal()
    example_dirichlet()
    example_distribution_selection()
