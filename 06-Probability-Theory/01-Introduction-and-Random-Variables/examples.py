"""
Introduction to Probability and Random Variables - Examples
===========================================================
Practical examples of probability concepts.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def example_basic_probability():
    """Basic probability rules."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Probability Rules")
    print("=" * 60)
    
    print("Rolling a fair die:")
    print("  Sample space Ω = {1, 2, 3, 4, 5, 6}")
    
    # Event probabilities
    P_even = 3/6  # {2, 4, 6}
    P_gt4 = 2/6   # {5, 6}
    P_even_and_gt4 = 1/6  # {6}
    
    print(f"\n  P(even) = 3/6 = {P_even:.4f}")
    print(f"  P(>4) = 2/6 = {P_gt4:.4f}")
    print(f"  P(even AND >4) = 1/6 = {P_even_and_gt4:.4f}")
    
    # Union rule
    P_even_or_gt4 = P_even + P_gt4 - P_even_and_gt4
    print(f"\n  P(even OR >4) = P(even) + P(>4) - P(even AND >4)")
    print(f"               = {P_even} + {P_gt4} - {P_even_and_gt4} = {P_even_or_gt4:.4f}")
    
    # Verify by counting: {2, 4, 5, 6}
    print(f"  Verify: {{2, 4, 5, 6}} = 4/6 = {4/6:.4f}")


def example_conditional_probability():
    """Conditional probability computation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Conditional Probability")
    print("=" * 60)
    
    print("Medical test scenario:")
    print("  - Disease prevalence: P(D) = 0.01 (1%)")
    print("  - Test sensitivity: P(+|D) = 0.99 (true positive rate)")
    print("  - Test specificity: P(-|D') = 0.95 (true negative rate)")
    
    P_D = 0.01
    P_pos_given_D = 0.99
    P_neg_given_notD = 0.95
    P_notD = 1 - P_D
    P_pos_given_notD = 1 - P_neg_given_notD  # False positive rate
    
    # P(+) using total probability
    P_pos = P_pos_given_D * P_D + P_pos_given_notD * P_notD
    
    print(f"\n  P(+) = P(+|D)P(D) + P(+|D')P(D')")
    print(f"       = {P_pos_given_D} × {P_D} + {P_pos_given_notD} × {P_notD}")
    print(f"       = {P_pos:.4f}")
    
    # Bayes' theorem
    P_D_given_pos = (P_pos_given_D * P_D) / P_pos
    
    print(f"\n  P(D|+) = P(+|D)P(D) / P(+)")
    print(f"        = ({P_pos_given_D} × {P_D}) / {P_pos:.4f}")
    print(f"        = {P_D_given_pos:.4f}")
    
    print(f"\n  Interpretation: If test is positive, only {P_D_given_pos*100:.1f}% chance")
    print(f"  of actually having the disease! (due to low prevalence)")


def example_independence():
    """Testing independence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Independence")
    print("=" * 60)
    
    print("Two fair coins:")
    # Joint distribution
    joint = np.array([[0.25, 0.25],  # First coin: H
                      [0.25, 0.25]]) # First coin: T
    
    print("Joint distribution P(X, Y):")
    print("         Y=H    Y=T")
    print(f"  X=H   {joint[0,0]:.2f}   {joint[0,1]:.2f}")
    print(f"  X=T   {joint[1,0]:.2f}   {joint[1,1]:.2f}")
    
    # Marginals
    P_X = joint.sum(axis=1)
    P_Y = joint.sum(axis=0)
    
    print(f"\n  P(X=H) = {P_X[0]:.2f}, P(X=T) = {P_X[1]:.2f}")
    print(f"  P(Y=H) = {P_Y[0]:.2f}, P(Y=T) = {P_Y[1]:.2f}")
    
    # Check independence: P(X,Y) = P(X)P(Y)?
    print("\n  Check P(X,Y) = P(X)P(Y):")
    print(f"    P(H,H) = {joint[0,0]:.2f} = {P_X[0]:.2f} × {P_Y[0]:.2f} = {P_X[0]*P_Y[0]:.2f} ✓")
    
    # Dependent example
    print("\n--- Dependent case: sum of two dice ---")
    print("  Let X = first die, Y = sum of both dice")
    print("  P(Y=7|X=1) = 1/6 (need second die = 6)")
    print("  P(Y=7|X=3) = 1/6 (need second die = 4)")
    print("  P(Y=7) = 6/36 = 1/6 (many ways to get 7)")
    print("  Since P(Y=7|X) = P(Y=7) for all X, X and Y are independent!")
    print("  (Surprising but true!)")


def example_bernoulli_binomial():
    """Bernoulli and Binomial distributions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Bernoulli and Binomial")
    print("=" * 60)
    
    p = 0.3  # Probability of success
    
    print(f"Bernoulli(p={p}):")
    print(f"  P(X=1) = {p}")
    print(f"  P(X=0) = {1-p}")
    print(f"  E[X] = {p}")
    print(f"  Var(X) = p(1-p) = {p*(1-p):.4f}")
    
    # Binomial
    n = 10
    print(f"\nBinomial(n={n}, p={p}):")
    print(f"  '{n} independent trials, each with P(success)={p}'")
    
    for k in range(n+1):
        prob = stats.binom.pmf(k, n, p)
        bar = '█' * int(prob * 50)
        print(f"  P(X={k:2d}) = {prob:.4f} {bar}")
    
    print(f"\n  E[X] = np = {n*p:.2f}")
    print(f"  Var(X) = np(1-p) = {n*p*(1-p):.2f}")
    
    # Simulation
    np.random.seed(42)
    samples = np.random.binomial(n, p, size=10000)
    print(f"\n  Simulation (10000 samples):")
    print(f"    Mean = {samples.mean():.3f} (theoretical: {n*p:.2f})")
    print(f"    Var  = {samples.var():.3f} (theoretical: {n*p*(1-p):.2f})")


def example_poisson():
    """Poisson distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Poisson Distribution")
    print("=" * 60)
    
    lam = 4  # Average rate
    
    print(f"Poisson(λ={lam}):")
    print(f"  'Average of {lam} events per unit time'")
    
    print("\nPMF:")
    for k in range(15):
        prob = stats.poisson.pmf(k, lam)
        bar = '█' * int(prob * 50)
        print(f"  P(X={k:2d}) = {prob:.4f} {bar}")
    
    print(f"\n  E[X] = λ = {lam}")
    print(f"  Var(X) = λ = {lam}")
    
    # Binomial approximation to Poisson
    print("\n--- Binomial → Poisson (n→∞, np=λ) ---")
    for n in [10, 50, 100, 500]:
        p = lam / n
        binom_prob = stats.binom.pmf(4, n, p)
        poisson_prob = stats.poisson.pmf(4, lam)
        print(f"  n={n:3d}: Binomial({n},{lam/n:.4f}) P(X=4) = {binom_prob:.4f}, Poisson({lam}) = {poisson_prob:.4f}")


def example_uniform():
    """Uniform distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Continuous Uniform Distribution")
    print("=" * 60)
    
    a, b = 2, 8
    
    print(f"Uniform({a}, {b}):")
    print(f"  PDF: f(x) = 1/(b-a) = 1/{b-a} = {1/(b-a):.4f} for x ∈ [{a}, {b}]")
    
    print(f"\n  E[X] = (a+b)/2 = ({a}+{b})/2 = {(a+b)/2:.2f}")
    print(f"  Var(X) = (b-a)²/12 = ({b-a})²/12 = {(b-a)**2/12:.4f}")
    
    # Probabilities
    print(f"\n  P(X ≤ 5) = (5 - {a})/({b} - {a}) = {(5-a)/(b-a):.4f}")
    print(f"  P(3 ≤ X ≤ 6) = (6-3)/({b}-{a}) = {(6-3)/(b-a):.4f}")
    
    # Simulation
    np.random.seed(42)
    samples = np.random.uniform(a, b, size=10000)
    print(f"\n  Simulation:")
    print(f"    Mean = {samples.mean():.3f}")
    print(f"    Var  = {samples.var():.3f}")


def example_gaussian():
    """Gaussian/Normal distribution."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Gaussian (Normal) Distribution")
    print("=" * 60)
    
    mu, sigma = 5, 2
    
    print(f"Normal(μ={mu}, σ²={sigma**2}):")
    print(f"  PDF: f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))")
    
    print(f"\n  E[X] = μ = {mu}")
    print(f"  Var(X) = σ² = {sigma**2}")
    print(f"  Std(X) = σ = {sigma}")
    
    # Standard normal probabilities
    print("\n--- Standard Normal Z ~ N(0,1) ---")
    print(f"  P(Z ≤ 0) = {stats.norm.cdf(0):.4f}")
    print(f"  P(Z ≤ 1) = {stats.norm.cdf(1):.4f}")
    print(f"  P(Z ≤ 2) = {stats.norm.cdf(2):.4f}")
    print(f"  P(|Z| ≤ 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f} (68% rule)")
    print(f"  P(|Z| ≤ 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f} (95% rule)")
    print(f"  P(|Z| ≤ 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f} (99.7% rule)")
    
    # Visualization
    print("\n  PDF shape (ASCII):")
    x_vals = np.linspace(mu-4*sigma, mu+4*sigma, 41)
    for x in x_vals[::2]:
        pdf_val = stats.norm.pdf(x, mu, sigma)
        bar = '█' * int(pdf_val * 80)
        print(f"  x={x:5.1f}: {bar}")


def example_expectation_variance():
    """Computing expectation and variance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Expectation and Variance")
    print("=" * 60)
    
    # Discrete: custom die
    print("Custom die with P(X=x) proportional to x:")
    probs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    probs = probs / probs.sum()
    values = np.array([1, 2, 3, 4, 5, 6])
    
    print("  x:", values)
    print("  P:", probs.round(4))
    
    # E[X]
    E_X = np.sum(values * probs)
    print(f"\n  E[X] = Σ x·P(X=x) = {E_X:.4f}")
    
    # E[X²]
    E_X2 = np.sum(values**2 * probs)
    print(f"  E[X²] = Σ x²·P(X=x) = {E_X2:.4f}")
    
    # Var(X)
    Var_X = E_X2 - E_X**2
    print(f"  Var(X) = E[X²] - E[X]² = {E_X2:.4f} - {E_X**2:.4f} = {Var_X:.4f}")
    
    # Linearity
    print("\n--- Linearity of Expectation ---")
    print(f"  Let Y = 2X + 3")
    E_Y = 2 * E_X + 3
    print(f"  E[Y] = 2E[X] + 3 = 2×{E_X:.4f} + 3 = {E_Y:.4f}")
    
    Var_Y = 4 * Var_X  # Var(aX+b) = a²Var(X)
    print(f"  Var(Y) = 4·Var(X) = 4×{Var_X:.4f} = {Var_Y:.4f}")


def example_sum_of_random_variables():
    """Sum of independent random variables."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Sum of Random Variables")
    print("=" * 60)
    
    print("X₁, X₂ independent with:")
    print("  E[X₁] = 3, Var(X₁) = 4")
    print("  E[X₂] = 5, Var(X₂) = 9")
    
    E_X1, Var_X1 = 3, 4
    E_X2, Var_X2 = 5, 9
    
    print("\nFor S = X₁ + X₂:")
    E_S = E_X1 + E_X2
    Var_S = Var_X1 + Var_X2  # Independent!
    print(f"  E[S] = E[X₁] + E[X₂] = {E_S}")
    print(f"  Var(S) = Var(X₁) + Var(X₂) = {Var_S} (independence!)")
    
    print("\nFor D = X₁ - X₂:")
    E_D = E_X1 - E_X2
    Var_D = Var_X1 + Var_X2  # Var(X-Y) = Var(X) + Var(Y) for independent
    print(f"  E[D] = E[X₁] - E[X₂] = {E_D}")
    print(f"  Var(D) = Var(X₁) + Var(X₂) = {Var_D}")
    print("  (Note: variance ADDS even for subtraction!)")
    
    # Central Limit Theorem
    print("\n--- Central Limit Theorem ---")
    print("Sum of n IID variables → Normal as n → ∞")
    
    np.random.seed(42)
    
    # Sum of uniform random variables
    for n in [1, 2, 5, 10, 30]:
        samples = np.sum(np.random.uniform(0, 1, size=(10000, n)), axis=1)
        # Standardize
        samples_std = (samples - n*0.5) / np.sqrt(n/12)
        
        # Compare to standard normal
        ks_stat, p_value = stats.kstest(samples_std, 'norm')
        print(f"  n={n:2d}: KS test p-value = {p_value:.4f}")


def example_bayes_ml():
    """Bayes' theorem in ML context."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Bayes' Theorem in ML")
    print("=" * 60)
    
    print("Naive Bayes for spam detection:")
    print("  Classify email as spam (S=1) or not (S=0)")
    print("  Feature: contains word 'free' (W=1) or not (W=0)")
    
    # Prior
    P_spam = 0.4
    P_not_spam = 0.6
    
    # Likelihoods
    P_free_given_spam = 0.7
    P_free_given_not_spam = 0.1
    
    print(f"\n  Prior: P(spam) = {P_spam}, P(not spam) = {P_not_spam}")
    print(f"  P('free'|spam) = {P_free_given_spam}")
    print(f"  P('free'|not spam) = {P_free_given_not_spam}")
    
    # Evidence
    P_free = P_free_given_spam * P_spam + P_free_given_not_spam * P_not_spam
    print(f"\n  P('free') = {P_free_given_spam}×{P_spam} + {P_free_given_not_spam}×{P_not_spam} = {P_free}")
    
    # Posterior
    P_spam_given_free = (P_free_given_spam * P_spam) / P_free
    P_not_spam_given_free = (P_free_given_not_spam * P_not_spam) / P_free
    
    print(f"\n  P(spam|'free') = {P_spam_given_free:.4f}")
    print(f"  P(not spam|'free') = {P_not_spam_given_free:.4f}")
    
    print(f"\n  Classification: {'SPAM' if P_spam_given_free > 0.5 else 'NOT SPAM'}")


def example_mle():
    """Maximum Likelihood Estimation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Maximum Likelihood Estimation")
    print("=" * 60)
    
    print("MLE for Bernoulli parameter:")
    print("  Data: [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]")
    
    data = np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 1])
    n = len(data)
    k = data.sum()
    
    print(f"  n = {n} trials, k = {k} successes")
    
    print("\n  Likelihood: L(p) = p^k (1-p)^(n-k)")
    print("  Log-likelihood: ℓ(p) = k log(p) + (n-k) log(1-p)")
    print("  Derivative: dℓ/dp = k/p - (n-k)/(1-p) = 0")
    print("  Solution: p̂ = k/n")
    
    p_mle = k / n
    print(f"\n  MLE: p̂ = {k}/{n} = {p_mle:.4f}")
    
    # Visualize likelihood
    print("\n  Likelihood function:")
    p_vals = np.linspace(0.1, 0.9, 9)
    for p in p_vals:
        likelihood = (p ** k) * ((1-p) ** (n-k))
        bar = '█' * int(likelihood * 1000)
        marker = ' ← MLE' if abs(p - p_mle) < 0.05 else ''
        print(f"  p={p:.1f}: L={likelihood:.6f} {bar}{marker}")


if __name__ == "__main__":
    example_basic_probability()
    example_conditional_probability()
    example_independence()
    example_bernoulli_binomial()
    example_poisson()
    example_uniform()
    example_gaussian()
    example_expectation_variance()
    example_sum_of_random_variables()
    example_bayes_ml()
    example_mle()
