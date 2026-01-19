"""
Joint Distributions and Independence - Examples
===============================================
Practical demonstrations of joint probability distributions.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def example_discrete_joint():
    """Joint, marginal, and conditional distributions for discrete RVs."""
    print("=" * 60)
    print("EXAMPLE 1: Discrete Joint Distribution")
    print("=" * 60)
    
    # Joint PMF table
    joint = np.array([
        [0.10, 0.15, 0.05],  # X=0
        [0.20, 0.30, 0.20]   # X=1
    ])
    
    print("Joint PMF P(X, Y):")
    print("       Y=0    Y=1    Y=2")
    print(f"  X=0  {joint[0,0]:.2f}   {joint[0,1]:.2f}   {joint[0,2]:.2f}")
    print(f"  X=1  {joint[1,0]:.2f}   {joint[1,1]:.2f}   {joint[1,2]:.2f}")
    print(f"  Sum = {joint.sum():.2f}")
    
    # Marginals
    p_X = joint.sum(axis=1)  # Sum over Y
    p_Y = joint.sum(axis=0)  # Sum over X
    
    print("\nMarginal P(X):")
    print(f"  P(X=0) = {p_X[0]:.2f}")
    print(f"  P(X=1) = {p_X[1]:.2f}")
    
    print("\nMarginal P(Y):")
    print(f"  P(Y=0) = {p_Y[0]:.2f}")
    print(f"  P(Y=1) = {p_Y[1]:.2f}")
    print(f"  P(Y=2) = {p_Y[2]:.2f}")
    
    # Conditional P(Y|X)
    print("\nConditional P(Y|X=0):")
    p_Y_given_X0 = joint[0, :] / p_X[0]
    for j in range(3):
        print(f"  P(Y={j}|X=0) = {joint[0,j]:.2f}/{p_X[0]:.2f} = {p_Y_given_X0[j]:.3f}")
    
    print("\nConditional P(Y|X=1):")
    p_Y_given_X1 = joint[1, :] / p_X[1]
    for j in range(3):
        print(f"  P(Y={j}|X=1) = {joint[1,j]:.2f}/{p_X[1]:.2f} = {p_Y_given_X1[j]:.3f}")
    
    # Check independence: joint = marginal_X * marginal_Y?
    print("\n--- Independence Check ---")
    product = np.outer(p_X, p_Y)
    print("P(X) × P(Y):")
    print(f"       Y=0    Y=1    Y=2")
    print(f"  X=0  {product[0,0]:.2f}   {product[0,1]:.3f}   {product[0,2]:.3f}")
    print(f"  X=1  {product[1,0]:.2f}   {product[1,1]:.3f}   {product[1,2]:.3f}")
    
    is_independent = np.allclose(joint, product)
    print(f"\nJoint = Product of marginals? {is_independent}")
    print("X and Y are NOT independent!")


def example_continuous_joint():
    """Joint distribution for continuous random variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Continuous Joint Distribution")
    print("=" * 60)
    
    print("Example: f(x,y) = 2 for 0 < x < y < 1")
    print("         (Uniform over upper triangle)")
    
    # Verify it integrates to 1
    print("\nVerification: ∫∫ f(x,y) dx dy")
    print("  = ∫₀¹ ∫₀ʸ 2 dx dy")
    print("  = ∫₀¹ 2y dy")
    print("  = y² |₀¹ = 1 ✓")
    
    # Marginals
    print("\nMarginal f_X(x):")
    print("  f_X(x) = ∫ₓ¹ 2 dy = 2(1-x) for 0 < x < 1")
    
    print("\nMarginal f_Y(y):")
    print("  f_Y(y) = ∫₀ʸ 2 dx = 2y for 0 < y < 1")
    
    # Conditional
    print("\nConditional f(X|Y=y):")
    print("  f_{X|Y}(x|y) = f(x,y)/f_Y(y) = 2/(2y) = 1/y")
    print("  for 0 < x < y (Uniform on [0,y])")
    
    # Simulate
    np.random.seed(42)
    n = 100000
    
    # Sample using rejection or transformation
    # Use: Y ~ Beta(2,1), X|Y ~ Uniform(0,Y)
    Y = np.random.beta(2, 1, n)  # f_Y(y) = 2y
    X = np.random.uniform(0, Y)
    
    print("\n--- Simulation Verification ---")
    print(f"  E[X] = ∫x·2(1-x)dx = [x² - 2x³/3]₀¹ = 1/3")
    print(f"  Simulated E[X] = {X.mean():.4f}")
    
    print(f"  E[Y] = ∫y·2y dy = [2y³/3]₀¹ = 2/3")
    print(f"  Simulated E[Y] = {Y.mean():.4f}")


def example_independence():
    """Testing and understanding independence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Independence")
    print("=" * 60)
    
    # Independent case
    print("Case 1: Independent dice rolls")
    np.random.seed(42)
    n = 100000
    X = np.random.randint(1, 7, n)  # Die 1
    Y = np.random.randint(1, 7, n)  # Die 2
    
    print(f"  P(X=1) = {(X == 1).mean():.4f}")
    print(f"  P(Y=2) = {(Y == 2).mean():.4f}")
    print(f"  P(X=1, Y=2) = {((X == 1) & (Y == 2)).mean():.4f}")
    print(f"  P(X=1) × P(Y=2) = {(X == 1).mean() * (Y == 2).mean():.4f}")
    print("  ≈ Equal, so independent ✓")
    
    # Dependent case
    print("\nCase 2: Dependent - Y = X + noise")
    Z = X + np.random.randint(-1, 2, n)  # X + {-1, 0, 1}
    
    print(f"  P(X=3) = {(X == 3).mean():.4f}")
    print(f"  P(Z=4) = {(Z == 4).mean():.4f}")
    print(f"  P(X=3, Z=4) = {((X == 3) & (Z == 4)).mean():.4f}")
    print(f"  P(X=3) × P(Z=4) = {(X == 3).mean() * (Z == 4).mean():.4f}")
    print("  ≠ Equal, so dependent!")
    
    # Conditional gives more info
    print(f"\n  P(Z=4 | X=3) = {(Z[X == 3] == 4).mean():.4f}")
    print(f"  P(Z=4 | X=6) = {(Z[X == 6] == 4).mean():.4f}")
    print("  Conditional ≠ marginal confirms dependence")


def example_conditional_independence():
    """Conditional independence demonstration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Conditional Independence")
    print("=" * 60)
    
    print("Scenario: X and Y share a common cause Z")
    print("  Z = weather (0=bad, 1=good)")
    print("  X = traffic delay")
    print("  Y = outdoor event attendance")
    
    np.random.seed(42)
    n = 100000
    
    # Z = weather (prior)
    Z = np.random.binomial(1, 0.6, n)
    
    # X depends on Z
    X = np.random.normal(30 - 20*Z, 10)  # Bad weather → more delay
    
    # Y depends on Z
    Y = np.random.normal(100 + 50*Z, 20)  # Good weather → more attendance
    
    # Unconditional correlation
    corr_uncond = np.corrcoef(X, Y)[0, 1]
    print(f"\nCorr(X, Y) = {corr_uncond:.4f}")
    print("  Negative! Bad weather → high delay, low attendance")
    
    # Conditional correlation (given Z=0)
    mask_z0 = (Z == 0)
    corr_z0 = np.corrcoef(X[mask_z0], Y[mask_z0])[0, 1]
    
    mask_z1 = (Z == 1)
    corr_z1 = np.corrcoef(X[mask_z1], Y[mask_z1])[0, 1]
    
    print(f"\nCorr(X, Y | Z=0) = {corr_z0:.4f}")
    print(f"Corr(X, Y | Z=1) = {corr_z1:.4f}")
    print("  ≈ 0! Given weather, X and Y are nearly independent")
    
    print("\nThis illustrates X ⊥ Y | Z")
    print("(X and Y are conditionally independent given Z)")


def example_covariance_correlation():
    """Computing covariance and correlation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Covariance and Correlation")
    print("=" * 60)
    
    np.random.seed(42)
    n = 10000
    
    # Generate correlated data
    mu = [5, 10]
    cov_matrix = [[4, 3],
                  [3, 9]]
    
    data = np.random.multivariate_normal(mu, cov_matrix, n)
    X, Y = data[:, 0], data[:, 1]
    
    print("Theoretical parameters:")
    print(f"  E[X] = {mu[0]}, E[Y] = {mu[1]}")
    print(f"  Var(X) = {cov_matrix[0][0]}, Var(Y) = {cov_matrix[1][1]}")
    print(f"  Cov(X,Y) = {cov_matrix[0][1]}")
    
    # Compute from samples
    print("\nSample estimates:")
    print(f"  Mean X = {X.mean():.4f}")
    print(f"  Mean Y = {Y.mean():.4f}")
    print(f"  Var(X) = {X.var():.4f}")
    print(f"  Var(Y) = {Y.var():.4f}")
    
    # Covariance formula
    cov_xy = ((X - X.mean()) * (Y - Y.mean())).mean()
    print(f"  Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)] = {cov_xy:.4f}")
    
    # Alternative formula
    cov_xy_alt = (X * Y).mean() - X.mean() * Y.mean()
    print(f"  Cov(X,Y) = E[XY] - E[X]E[Y] = {cov_xy_alt:.4f}")
    
    # Correlation
    corr = cov_xy / (X.std() * Y.std())
    print(f"\n  ρ(X,Y) = Cov(X,Y)/(σ_X σ_Y) = {corr:.4f}")
    
    # Verify with numpy
    print(f"  np.corrcoef: {np.corrcoef(X, Y)[0,1]:.4f}")


def example_covariance_zero_not_independent():
    """Showing that zero covariance doesn't imply independence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Zero Covariance ≠ Independence")
    print("=" * 60)
    
    print("Classic counterexample: X ~ N(0,1), Y = X²")
    
    np.random.seed(42)
    n = 100000
    X = np.random.standard_normal(n)
    Y = X**2
    
    print("\nTheoretical calculation:")
    print("  Cov(X, X²) = E[X · X²] - E[X]·E[X²]")
    print("             = E[X³] - 0·E[X²]")
    print("             = 0  (odd moments of N(0,1) are 0)")
    
    cov = np.cov(X, Y)[0, 1]
    corr = np.corrcoef(X, Y)[0, 1]
    
    print(f"\nSimulation:")
    print(f"  Cov(X, Y) = {cov:.6f} ≈ 0")
    print(f"  Corr(X, Y) = {corr:.6f} ≈ 0")
    
    print("\nBut are they independent?")
    print("  If independent: P(Y < 1 | X = 0.5) = P(Y < 1)")
    
    # Check independence
    mask = np.abs(X - 0.5) < 0.1
    p_y_lt_1_given_x = (Y[mask] < 1).mean()
    p_y_lt_1 = (Y < 1).mean()
    
    print(f"  P(Y < 1) = {p_y_lt_1:.4f}")
    print(f"  P(Y < 1 | X ≈ 0.5) = {p_y_lt_1_given_x:.4f}")
    print("\n  These are different! → NOT independent")
    print("  Y is completely determined by X!")


def example_multivariate_normal():
    """Multivariate normal distribution properties."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Multivariate Normal")
    print("=" * 60)
    
    mu = np.array([1, 2])
    Sigma = np.array([[2, 1.5],
                      [1.5, 3]])
    
    print(f"μ = {mu}")
    print(f"Σ = \n{Sigma}")
    
    # Marginals
    print("\n--- Marginal Distributions ---")
    print(f"  X₁ ~ N({mu[0]}, {Sigma[0,0]})")
    print(f"  X₂ ~ N({mu[1]}, {Sigma[1,1]})")
    
    # Conditional
    print("\n--- Conditional Distribution X₁|X₂=x₂ ---")
    x2_given = 4.0
    
    # μ_{1|2} = μ₁ + Σ₁₂/Σ₂₂ × (x₂ - μ₂)
    mu_cond = mu[0] + Sigma[0,1]/Sigma[1,1] * (x2_given - mu[1])
    # Σ_{1|2} = Σ₁₁ - Σ₁₂²/Σ₂₂
    var_cond = Sigma[0,0] - Sigma[0,1]**2/Sigma[1,1]
    
    print(f"  Given X₂ = {x2_given}:")
    print(f"  E[X₁|X₂] = μ₁ + Σ₁₂/Σ₂₂(x₂-μ₂)")
    print(f"           = {mu[0]} + {Sigma[0,1]}/{Sigma[1,1]}×({x2_given}-{mu[1]})")
    print(f"           = {mu_cond:.4f}")
    print(f"  Var(X₁|X₂) = Σ₁₁ - Σ₁₂²/Σ₂₂")
    print(f"             = {Sigma[0,0]} - {Sigma[0,1]}²/{Sigma[1,1]}")
    print(f"             = {var_cond:.4f}")
    
    # Verify by simulation
    np.random.seed(42)
    samples = np.random.multivariate_normal(mu, Sigma, 100000)
    
    mask = np.abs(samples[:, 1] - x2_given) < 0.1
    conditional_samples = samples[mask, 0]
    
    print(f"\n  Simulation (samples where X₂ ≈ {x2_given}):")
    print(f"    Mean: {conditional_samples.mean():.4f}")
    print(f"    Var:  {conditional_samples.var():.4f}")


def example_linear_transformation():
    """Linear transformation of multivariate normal."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Linear Transformation of MVN")
    print("=" * 60)
    
    mu = np.array([1, 2])
    Sigma = np.array([[2, 0.5],
                      [0.5, 1]])
    
    A = np.array([[2, 1],
                  [0, 1]])
    b = np.array([1, -1])
    
    print("X ~ N(μ, Σ)")
    print(f"  μ = {mu}")
    print(f"  Σ = \n{Sigma}")
    
    print(f"\nY = AX + b")
    print(f"  A = \n{A}")
    print(f"  b = {b}")
    
    # Theoretical result
    mu_Y = A @ mu + b
    Sigma_Y = A @ Sigma @ A.T
    
    print("\nTheoretical: Y ~ N(Aμ+b, AΣAᵀ)")
    print(f"  E[Y] = {mu_Y}")
    print(f"  Cov(Y) = \n{Sigma_Y}")
    
    # Simulation
    np.random.seed(42)
    X = np.random.multivariate_normal(mu, Sigma, 100000)
    Y = (A @ X.T).T + b
    
    print("\nSimulation:")
    print(f"  E[Y] = {Y.mean(axis=0).round(4)}")
    print(f"  Cov(Y) = \n{np.cov(Y.T).round(4)}")


def example_bayes_theorem():
    """Bayes' theorem with random variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Bayes' Theorem for Classification")
    print("=" * 60)
    
    print("Scenario: Two classes with Gaussian features")
    print("  Class 0: X|Y=0 ~ N(μ₀, σ²)")
    print("  Class 1: X|Y=1 ~ N(μ₁, σ²)")
    print("  Prior: P(Y=1) = π")
    
    mu_0, mu_1 = 2, 5
    sigma = 1.5
    prior_1 = 0.3
    
    print(f"\n  μ₀={mu_0}, μ₁={mu_1}, σ={sigma}")
    print(f"  P(Y=1) = {prior_1}")
    
    print("\nBayes' theorem:")
    print("  P(Y=1|X=x) = P(X=x|Y=1)P(Y=1) / P(X=x)")
    
    # For a specific observation
    x_obs = 3.5
    
    # Likelihoods
    lik_0 = stats.norm.pdf(x_obs, mu_0, sigma)
    lik_1 = stats.norm.pdf(x_obs, mu_1, sigma)
    
    print(f"\nFor observation x = {x_obs}:")
    print(f"  P(X={x_obs}|Y=0) = {lik_0:.4f}")
    print(f"  P(X={x_obs}|Y=1) = {lik_1:.4f}")
    
    # Evidence
    evidence = lik_0 * (1 - prior_1) + lik_1 * prior_1
    print(f"  P(X={x_obs}) = {evidence:.4f}")
    
    # Posterior
    post_1 = lik_1 * prior_1 / evidence
    post_0 = lik_0 * (1 - prior_1) / evidence
    
    print(f"\nPosterior:")
    print(f"  P(Y=0|X={x_obs}) = {post_0:.4f}")
    print(f"  P(Y=1|X={x_obs}) = {post_1:.4f}")
    
    print(f"\nClassify as: Class {1 if post_1 > 0.5 else 0}")


def example_naive_bayes():
    """Naive Bayes classifier."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Naive Bayes Classifier")
    print("=" * 60)
    
    print("Assumes: Features conditionally independent given class")
    print("  P(X₁, X₂, ..., Xₙ | Y) = ∏ᵢ P(Xᵢ | Y)")
    
    # Training data statistics
    print("\nSpam classification example:")
    print("  Features: word presence (binary)")
    
    # P(word|spam) and P(word|not spam)
    p_free_spam = 0.8
    p_free_notspam = 0.1
    p_win_spam = 0.7
    p_win_notspam = 0.05
    p_meeting_spam = 0.1
    p_meeting_notspam = 0.6
    
    p_spam = 0.3  # Prior
    
    print(f"\n  Prior P(spam) = {p_spam}")
    print("\n  P(word|class):")
    print(f"    P('free'|spam) = {p_free_spam}, P('free'|not spam) = {p_free_notspam}")
    print(f"    P('win'|spam) = {p_win_spam}, P('win'|not spam) = {p_win_notspam}")
    print(f"    P('meeting'|spam) = {p_meeting_spam}, P('meeting'|not spam) = {p_meeting_notspam}")
    
    # New email: contains "free" and "win", no "meeting"
    print("\nNew email: contains 'free', 'win', not 'meeting'")
    
    # Likelihood × Prior
    lik_spam = p_free_spam * p_win_spam * (1 - p_meeting_spam) * p_spam
    lik_notspam = p_free_notspam * p_win_notspam * (1 - p_meeting_notspam) * (1 - p_spam)
    
    print(f"\n  P(features|spam) × P(spam) = {lik_spam:.6f}")
    print(f"  P(features|not spam) × P(not spam) = {lik_notspam:.6f}")
    
    # Normalize
    p_spam_given_email = lik_spam / (lik_spam + lik_notspam)
    print(f"\n  P(spam|email) = {p_spam_given_email:.4f}")
    print(f"  Classification: {'SPAM' if p_spam_given_email > 0.5 else 'NOT SPAM'}")


def example_covariance_matrix():
    """Working with covariance matrices."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Covariance Matrix Operations")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    n = 5000
    
    # True covariance
    Sigma_true = np.array([[1.0, 0.7, 0.3],
                           [0.7, 2.0, 0.5],
                           [0.3, 0.5, 1.5]])
    mu_true = np.array([0, 1, 2])
    
    X = np.random.multivariate_normal(mu_true, Sigma_true, n)
    
    print("True covariance matrix:")
    print(Sigma_true)
    
    # Sample covariance
    Sigma_sample = np.cov(X.T)
    print("\nSample covariance matrix:")
    print(Sigma_sample.round(3))
    
    # Properties
    print("\n--- Properties ---")
    
    # Symmetric
    print(f"Symmetric: {np.allclose(Sigma_sample, Sigma_sample.T)}")
    
    # Positive semi-definite (all eigenvalues ≥ 0)
    eigenvalues = np.linalg.eigvals(Sigma_sample)
    print(f"Eigenvalues: {eigenvalues.round(4)}")
    print(f"Positive semi-definite: {all(eigenvalues >= -1e-10)}")
    
    # Correlation matrix
    D = np.diag(1/np.sqrt(np.diag(Sigma_sample)))
    corr_matrix = D @ Sigma_sample @ D
    
    print("\nCorrelation matrix:")
    print(corr_matrix.round(3))
    print("(Diagonal = 1, off-diagonal = correlations)")


if __name__ == "__main__":
    example_discrete_joint()
    example_continuous_joint()
    example_independence()
    example_conditional_independence()
    example_covariance_correlation()
    example_covariance_zero_not_independent()
    example_multivariate_normal()
    example_linear_transformation()
    example_bayes_theorem()
    example_naive_bayes()
    example_covariance_matrix()
