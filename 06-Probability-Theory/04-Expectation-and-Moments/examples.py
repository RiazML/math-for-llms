"""
Expectation, Variance, and Moments - Examples
=============================================
Practical demonstrations of expectation and variance concepts.
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def example_expectation_basics():
    """Basic expectation calculations."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Expectation Calculations")
    print("=" * 60)
    
    # Discrete case: Die roll
    print("Example: Fair die roll")
    values = [1, 2, 3, 4, 5, 6]
    probs = [1/6] * 6
    
    E_X = sum(x * p for x, p in zip(values, probs))
    print(f"  E[X] = Σ x·P(X=x) = (1+2+3+4+5+6)/6 = {E_X:.4f}")
    
    # E[X²]
    E_X2 = sum(x**2 * p for x, p in zip(values, probs))
    print(f"  E[X²] = Σ x²·P(X=x) = {E_X2:.4f}")
    
    # Continuous case: Uniform(0, 1)
    print("\nExample: Uniform(0, 1)")
    print("  E[X] = ∫₀¹ x·1 dx = [x²/2]₀¹ = 0.5")
    print("  E[X²] = ∫₀¹ x²·1 dx = [x³/3]₀¹ = 1/3")
    
    # Verify with simulation
    np.random.seed(42)
    uniform_samples = np.random.uniform(0, 1, 100000)
    print(f"\n  Simulation: E[X] = {uniform_samples.mean():.4f}")
    print(f"              E[X²] = {(uniform_samples**2).mean():.4f}")


def example_expectation_properties():
    """Demonstrate linearity of expectation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Linearity of Expectation")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100000
    
    X = np.random.normal(5, 2, n)
    Y = np.random.normal(3, 1, n)
    
    print("X ~ N(5, 4), Y ~ N(3, 1)")
    print(f"  E[X] = {X.mean():.4f}")
    print(f"  E[Y] = {Y.mean():.4f}")
    
    # E[aX + b]
    a, b = 2, 3
    print(f"\n  E[{a}X + {b}] = {a}·E[X] + {b} = {a*5 + b}")
    print(f"  Simulation: {(a*X + b).mean():.4f}")
    
    # E[X + Y] - works even for DEPENDENT variables!
    print(f"\n  E[X + Y] = E[X] + E[Y] = {5 + 3}")
    print(f"  Simulation: {(X + Y).mean():.4f}")
    
    # Make dependent variables
    Z = X + np.random.randn(n)  # Z depends on X
    print(f"\n  Z = X + noise (dependent on X)")
    print(f"  E[X + Z] = E[X] + E[Z] still works!")
    print(f"  E[Z] = {Z.mean():.4f}")
    print(f"  E[X + Z] = {(X + Z).mean():.4f} ≈ {X.mean() + Z.mean():.4f}")


def example_expectation_of_function():
    """E[g(X)] is NOT g(E[X]) in general."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Expectation of Function")
    print("=" * 60)
    
    print("WARNING: E[g(X)] ≠ g(E[X]) in general!")
    
    np.random.seed(42)
    n = 100000
    X = np.random.uniform(1, 3, n)
    
    E_X = X.mean()
    
    print(f"\nX ~ Uniform(1, 3)")
    print(f"E[X] = {E_X:.4f}")
    
    # Square function
    print("\ng(x) = x²")
    print(f"  E[X²] = {(X**2).mean():.4f}")
    print(f"  E[X]² = {E_X**2:.4f}")
    print(f"  E[X²] ≠ E[X]² (Jensen: E[X²] ≥ E[X]² for convex x²)")
    
    # Log function
    print("\ng(x) = log(x)")
    print(f"  E[log(X)] = {np.log(X).mean():.4f}")
    print(f"  log(E[X]) = {np.log(E_X):.4f}")
    print(f"  E[log(X)] ≤ log(E[X]) (Jensen: log is concave)")
    
    # 1/x function
    print("\ng(x) = 1/x")
    print(f"  E[1/X] = {(1/X).mean():.4f}")
    print(f"  1/E[X] = {1/E_X:.4f}")
    print(f"  E[1/X] ≥ 1/E[X] (Jensen: 1/x is convex)")


def example_variance_basics():
    """Variance calculations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Variance Calculations")
    print("=" * 60)
    
    print("Definition: Var(X) = E[(X-μ)²] = E[X²] - E[X]²")
    
    # Die roll
    print("\nExample: Fair die")
    values = [1, 2, 3, 4, 5, 6]
    probs = [1/6] * 6
    
    E_X = sum(x * p for x, p in zip(values, probs))
    E_X2 = sum(x**2 * p for x, p in zip(values, probs))
    Var_X = E_X2 - E_X**2
    
    print(f"  E[X] = {E_X:.4f}")
    print(f"  E[X²] = {E_X2:.4f}")
    print(f"  Var(X) = {E_X2:.4f} - {E_X:.4f}² = {Var_X:.4f}")
    print(f"  σ = √Var(X) = {np.sqrt(Var_X):.4f}")
    
    # Simulate
    np.random.seed(42)
    die_rolls = np.random.randint(1, 7, 100000)
    print(f"\n  Simulation: Var = {die_rolls.var():.4f}")


def example_variance_properties():
    """Properties of variance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Properties of Variance")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100000
    X = np.random.normal(5, 2, n)  # Var = 4
    
    print("X ~ N(5, 4)")
    print(f"  Var(X) = {X.var():.4f}")
    
    # Var(aX)
    a = 3
    print(f"\nVar({a}X) = {a}²·Var(X) = {a**2 * 4}")
    print(f"  Simulation: {(a*X).var():.4f}")
    
    # Var(X + b)
    b = 100
    print(f"\nVar(X + {b}) = Var(X) = 4")
    print(f"  Simulation: {(X + b).var():.4f}")
    
    # Var(aX + b)
    print(f"\nVar({a}X + {b}) = {a}²·Var(X) = {a**2 * 4}")
    print(f"  Simulation: {(a*X + b).var():.4f}")
    
    # Sum of independent
    Y = np.random.normal(0, 3, n)  # Var = 9
    print(f"\nY ~ N(0, 9) independent of X")
    print(f"  Var(X) + Var(Y) = 4 + 9 = 13")
    print(f"  Var(X + Y) = {(X + Y).var():.4f}")


def example_covariance_correlation():
    """Covariance and correlation calculations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Covariance and Correlation")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100000
    
    # Generate correlated data
    X = np.random.normal(0, 1, n)
    Y = 0.7 * X + np.sqrt(1 - 0.7**2) * np.random.normal(0, 1, n)
    
    print("X ~ N(0,1), Y = 0.7X + 0.714Z where Z ~ N(0,1)")
    
    # Covariance
    cov_formula = (X * Y).mean() - X.mean() * Y.mean()
    cov_direct = ((X - X.mean()) * (Y - Y.mean())).mean()
    
    print(f"\nCov(X,Y):")
    print(f"  Formula: E[XY] - E[X]E[Y] = {cov_formula:.4f}")
    print(f"  Direct: E[(X-μX)(Y-μY)] = {cov_direct:.4f}")
    print(f"  NumPy: {np.cov(X, Y)[0,1]:.4f}")
    
    # Correlation
    corr = cov_formula / (X.std() * Y.std())
    print(f"\nCorr(X,Y):")
    print(f"  Cov(X,Y)/(σX·σY) = {corr:.4f}")
    print(f"  NumPy: {np.corrcoef(X, Y)[0,1]:.4f}")
    print(f"  (True correlation: 0.7)")


def example_sum_variance():
    """Variance of sum with correlation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Variance of Sum")
    print("=" * 60)
    
    print("Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X,Y)")
    
    np.random.seed(42)
    n = 100000
    
    # Positive correlation
    X = np.random.normal(0, 2, n)
    Y = 0.5 * X + np.random.normal(0, 1.5, n)
    
    var_X = X.var()
    var_Y = Y.var()
    cov_XY = np.cov(X, Y)[0, 1]
    
    print(f"\nPositive correlation case:")
    print(f"  Var(X) = {var_X:.4f}")
    print(f"  Var(Y) = {var_Y:.4f}")
    print(f"  Cov(X,Y) = {cov_XY:.4f}")
    print(f"  Var(X) + Var(Y) + 2Cov(X,Y) = {var_X + var_Y + 2*cov_XY:.4f}")
    print(f"  Actual Var(X+Y) = {(X+Y).var():.4f}")
    
    # Negative correlation
    Z = -0.5 * X + np.random.normal(0, 1.5, n)
    var_Z = Z.var()
    cov_XZ = np.cov(X, Z)[0, 1]
    
    print(f"\nNegative correlation case:")
    print(f"  Cov(X,Z) = {cov_XZ:.4f}")
    print(f"  Var(X) + Var(Z) + 2Cov(X,Z) = {var_X + var_Z + 2*cov_XZ:.4f}")
    print(f"  Actual Var(X+Z) = {(X+Z).var():.4f}")
    
    print("\nNote: Negative correlation reduces variance of sum!")
    print("      (Portfolio diversification uses this!)")


def example_conditional_expectation():
    """Conditional expectation and law of total expectation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Conditional Expectation")
    print("=" * 60)
    
    print("Law of Total Expectation: E[X] = E[E[X|Y]]")
    
    np.random.seed(42)
    n = 100000
    
    # Y = group (0 or 1)
    Y = np.random.binomial(1, 0.6, n)
    # X | Y=0 ~ N(2, 1), X | Y=1 ~ N(5, 2)
    X = np.where(Y == 0,
                 np.random.normal(2, 1, n),
                 np.random.normal(5, np.sqrt(2), n))
    
    print("\nY ~ Bernoulli(0.6)")
    print("X | Y=0 ~ N(2, 1)")
    print("X | Y=1 ~ N(5, 2)")
    
    # Conditional expectations
    E_X_given_Y0 = X[Y == 0].mean()
    E_X_given_Y1 = X[Y == 1].mean()
    
    print(f"\nE[X|Y=0] = {E_X_given_Y0:.4f} (theory: 2)")
    print(f"E[X|Y=1] = {E_X_given_Y1:.4f} (theory: 5)")
    
    # Law of total expectation
    E_X_via_total = 0.4 * 2 + 0.6 * 5
    print(f"\nE[E[X|Y]] = P(Y=0)·E[X|Y=0] + P(Y=1)·E[X|Y=1]")
    print(f"         = 0.4×2 + 0.6×5 = {E_X_via_total}")
    print(f"E[X] (simulation) = {X.mean():.4f}")


def example_mgf():
    """Moment generating functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Moment Generating Functions")
    print("=" * 60)
    
    print("MGF: M(t) = E[e^{tX}]")
    print("Moments: E[X^n] = M^{(n)}(0)")
    
    print("\n--- Normal(μ, σ²) MGF ---")
    print("M(t) = exp(μt + σ²t²/2)")
    
    mu, sigma = 3, 2
    
    print(f"\nFor N({mu}, {sigma**2}):")
    print("  M'(t) = (μ + σ²t)·M(t)")
    print(f"  M'(0) = μ = {mu} = E[X]")
    
    print("  M''(t) = σ²·M(t) + (μ + σ²t)²·M(t)")
    print(f"  M''(0) = σ² + μ² = {sigma**2} + {mu**2} = {sigma**2 + mu**2} = E[X²]")
    print(f"  Var(X) = E[X²] - E[X]² = {sigma**2 + mu**2} - {mu**2} = {sigma**2}")
    
    # Verify
    np.random.seed(42)
    X = np.random.normal(mu, sigma, 100000)
    print(f"\nSimulation:")
    print(f"  E[X] = {X.mean():.4f}")
    print(f"  E[X²] = {(X**2).mean():.4f}")
    print(f"  Var(X) = {X.var():.4f}")


def example_higher_moments():
    """Skewness and kurtosis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Higher Moments - Skewness & Kurtosis")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100000
    
    # Normal (symmetric, normal tails)
    normal = np.random.normal(0, 1, n)
    
    # Exponential (right skewed)
    exp_shifted = np.random.exponential(1, n) - 1
    
    # Student's t (heavy tails)
    student = np.random.standard_t(5, n)
    
    # Uniform (light tails)
    uniform = np.random.uniform(-np.sqrt(3), np.sqrt(3), n)
    
    def moments(x, name):
        mean = x.mean()
        std = x.std()
        skew = ((x - mean)**3).mean() / std**3
        kurt = ((x - mean)**4).mean() / std**4
        excess_kurt = kurt - 3
        
        print(f"\n{name}:")
        print(f"  Mean = {mean:.4f}")
        print(f"  Std = {std:.4f}")
        print(f"  Skewness = {skew:.4f}")
        print(f"  Kurtosis = {kurt:.4f}")
        print(f"  Excess Kurtosis = {excess_kurt:.4f}")
    
    moments(normal, "Normal (reference)")
    moments(exp_shifted, "Exponential (right skewed)")
    moments(student, "Student's t(5) (heavy tails)")
    moments(uniform, "Uniform (light tails)")
    
    print("\nInterpretation:")
    print("  Skewness: 0 = symmetric, >0 = right skew, <0 = left skew")
    print("  Excess Kurtosis: 0 = normal, >0 = heavy tails, <0 = light tails")


def example_inequalities():
    """Markov, Chebyshev, and Jensen inequalities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Probability Inequalities")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100000
    X = np.random.exponential(2, n)  # Mean = 2
    
    print("X ~ Exponential(rate=1/2), so E[X] = 2")
    
    # Markov's inequality
    a = 4
    markov_bound = 2 / a
    actual_prob = (X >= a).mean()
    
    print(f"\n--- Markov's Inequality ---")
    print(f"P(X ≥ {a}) ≤ E[X]/{a} = 2/{a} = {markov_bound:.4f}")
    print(f"Actual P(X ≥ {a}) = {actual_prob:.4f}")
    print(f"Bound is correct: {actual_prob <= markov_bound}")
    
    # Chebyshev's inequality
    Y = np.random.normal(10, 3, n)  # μ=10, σ=3
    
    print(f"\n--- Chebyshev's Inequality ---")
    print("Y ~ N(10, 9), so σ = 3")
    
    for k in [2, 3, 4]:
        chebyshev_bound = 1 / k**2
        actual = (np.abs(Y - 10) >= k * 3).mean()
        print(f"  P(|Y-μ| ≥ {k}σ) ≤ 1/{k}² = {chebyshev_bound:.4f}, Actual = {actual:.4f}")
    
    # Jensen's inequality
    print(f"\n--- Jensen's Inequality ---")
    Z = np.random.uniform(1, 5, n)
    
    print(f"Z ~ Uniform(1, 5), E[Z] = {Z.mean():.4f}")
    print(f"Convex g(x) = x²:  E[Z²] = {(Z**2).mean():.4f} ≥ E[Z]² = {Z.mean()**2:.4f}")
    print(f"Concave g(x) = log: E[log Z] = {np.log(Z).mean():.4f} ≤ log(E[Z]) = {np.log(Z.mean()):.4f}")


def example_sample_statistics():
    """Sample mean and variance properties."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Sample Mean and Variance")
    print("=" * 60)
    
    # True parameters
    mu_true = 10
    sigma_true = 3
    
    np.random.seed(42)
    
    print(f"Population: N({mu_true}, {sigma_true**2})")
    print(f"True μ = {mu_true}, True σ² = {sigma_true**2}")
    
    # Multiple samples
    n_samples = 10000
    sample_size = 30
    
    sample_means = []
    sample_vars_n = []    # Divide by n
    sample_vars_n1 = []   # Divide by n-1
    
    for _ in range(n_samples):
        sample = np.random.normal(mu_true, sigma_true, sample_size)
        sample_means.append(sample.mean())
        sample_vars_n.append(sample.var())           # np default is n
        sample_vars_n1.append(sample.var(ddof=1))    # Unbiased
    
    sample_means = np.array(sample_means)
    sample_vars_n = np.array(sample_vars_n)
    sample_vars_n1 = np.array(sample_vars_n1)
    
    print(f"\n--- Sample Mean X̄ ---")
    print(f"  E[X̄] = μ = {mu_true}")
    print(f"  Simulated E[X̄] = {sample_means.mean():.4f}")
    print(f"  Var(X̄) = σ²/n = {sigma_true**2/sample_size:.4f}")
    print(f"  Simulated Var(X̄) = {sample_means.var():.4f}")
    
    print(f"\n--- Sample Variance ---")
    print(f"  True σ² = {sigma_true**2}")
    print(f"  S²_n (divide by n): E[S²_n] = {sample_vars_n.mean():.4f} (biased!)")
    print(f"  S²_{n-1} (divide by n-1): E[S²_{n-1}] = {sample_vars_n1.mean():.4f} (unbiased)")
    print(f"\n  Bias factor: E[S²_n]/σ² = {sample_vars_n.mean()/sigma_true**2:.4f} ≈ (n-1)/n = {(sample_size-1)/sample_size:.4f}")


if __name__ == "__main__":
    example_expectation_basics()
    example_expectation_properties()
    example_expectation_of_function()
    example_variance_basics()
    example_variance_properties()
    example_covariance_correlation()
    example_sum_variance()
    example_conditional_expectation()
    example_mgf()
    example_higher_moments()
    example_inequalities()
    example_sample_statistics()
