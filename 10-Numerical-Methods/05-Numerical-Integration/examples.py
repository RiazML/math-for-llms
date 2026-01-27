"""
Numerical Integration - Examples
================================
Practical implementations of quadrature methods.
"""

import numpy as np
from scipy import integrate
from scipy.special import roots_legendre, roots_hermite


def example_1_newton_cotes():
    """
    Example 1: Newton-Cotes Formulas
    ================================
    Basic quadrature rules.
    """
    print("=" * 60)
    print("Example 1: Newton-Cotes Formulas")
    print("=" * 60)
    
    # Test function: integrate sin(x) from 0 to pi
    # Exact value = 2
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    
    # Midpoint rule
    midpoint = (b - a) * f((a + b) / 2)
    
    # Trapezoidal rule
    trapezoid = (b - a) / 2 * (f(a) + f(b))
    
    # Simpson's rule
    simpson = (b - a) / 6 * (f(a) + 4*f((a+b)/2) + f(b))
    
    print(f"Integral: ∫₀^π sin(x) dx = {exact}")
    print(f"\nSingle interval approximations:")
    print(f"  Midpoint:    {midpoint:.6f}, error = {abs(midpoint - exact):.6f}")
    print(f"  Trapezoidal: {trapezoid:.6f}, error = {abs(trapezoid - exact):.6f}")
    print(f"  Simpson's:   {simpson:.6f}, error = {abs(simpson - exact):.6f}")


def example_2_composite_rules():
    """
    Example 2: Composite Quadrature
    ===============================
    Multiple subintervals for better accuracy.
    """
    print("\n" + "=" * 60)
    print("Example 2: Composite Rules")
    print("=" * 60)
    
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    
    def composite_trapezoid(f, a, b, n):
        """Composite trapezoidal rule with n subintervals."""
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    def composite_simpson(f, a, b, n):
        """Composite Simpson's rule (n must be even)."""
        if n % 2 != 0:
            n += 1
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    
    print(f"{'n':<8} {'Trapezoidal':<15} {'Error':<12} {'Simpson':<15} {'Error':<12}")
    print("-" * 65)
    
    for n in [2, 4, 8, 16, 32, 64]:
        trap = composite_trapezoid(f, a, b, n)
        simp = composite_simpson(f, a, b, n)
        
        print(f"{n:<8} {trap:<15.10f} {abs(trap-exact):<12.2e} "
              f"{simp:<15.10f} {abs(simp-exact):<12.2e}")
    
    print(f"\nNote: Simpson's error decreases ~16x when n doubles (O(h⁴))")
    print(f"      Trapezoidal error decreases ~4x when n doubles (O(h²))")


def example_3_gauss_legendre():
    """
    Example 3: Gauss-Legendre Quadrature
    ====================================
    Optimal node placement for polynomials.
    """
    print("\n" + "=" * 60)
    print("Example 3: Gauss-Legendre Quadrature")
    print("=" * 60)
    
    def gauss_legendre(f, a, b, n):
        """Gauss-Legendre quadrature with n points."""
        # Get nodes and weights for [-1, 1]
        nodes, weights = roots_legendre(n)
        
        # Transform to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        scale = 0.5 * (b - a)
        
        return scale * np.sum(weights * f(x))
    
    # Test: ∫₀^π sin(x) dx = 2
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    
    print(f"Integral: ∫₀^π sin(x) dx = {exact}")
    print(f"\n{'Points (n)':<15} {'Approximation':<18} {'Error':<15}")
    print("-" * 50)
    
    for n in [1, 2, 3, 4, 5, 10]:
        approx = gauss_legendre(f, a, b, n)
        print(f"{n:<15} {approx:<18.12f} {abs(approx - exact):<15.2e}")
    
    # Compare with Simpson at same number of function evaluations
    print(f"\nComparison at 5 function evaluations:")
    gl_5 = gauss_legendre(f, a, b, 5)
    
    # Simpson's with 5 points (n=4 intervals)
    h = (b - a) / 4
    x = np.linspace(a, b, 5)
    simp = h / 3 * (f(x[0]) + 4*f(x[1]) + 2*f(x[2]) + 4*f(x[3]) + f(x[4]))
    
    print(f"  Gauss-Legendre (5 pts): error = {abs(gl_5 - exact):.2e}")
    print(f"  Simpson's (5 pts):      error = {abs(simp - exact):.2e}")


def example_4_gauss_hermite():
    """
    Example 4: Gauss-Hermite for Gaussian Integrals
    ===============================================
    Essential for computing expectations under Gaussian distributions.
    """
    print("\n" + "=" * 60)
    print("Example 4: Gauss-Hermite Quadrature")
    print("=" * 60)
    
    def gauss_hermite_expectation(g, mu, sigma, n):
        """
        Compute E[g(X)] where X ~ N(mu, sigma²) using Gauss-Hermite.
        
        Transform: ∫ g(x) N(x|μ,σ²) dx = (1/√π) ∫ g(√2·σ·t + μ) e^(-t²) dt
        """
        nodes, weights = roots_hermite(n)
        
        # Transform nodes
        x = np.sqrt(2) * sigma * nodes + mu
        
        # Gauss-Hermite uses weight e^(-t²), so adjust
        return np.sum(weights * g(x)) / np.sqrt(np.pi)
    
    mu, sigma = 2.0, 1.5
    
    # Test 1: E[X] = μ
    g1 = lambda x: x
    exp_x = gauss_hermite_expectation(g1, mu, sigma, 5)
    print(f"X ~ N({mu}, {sigma}²)")
    print(f"\nE[X]:")
    print(f"  True:     {mu}")
    print(f"  GH (n=5): {exp_x:.10f}")
    
    # Test 2: E[X²] = μ² + σ²
    g2 = lambda x: x**2
    exp_x2 = gauss_hermite_expectation(g2, mu, sigma, 5)
    true_x2 = mu**2 + sigma**2
    print(f"\nE[X²]:")
    print(f"  True:     {true_x2}")
    print(f"  GH (n=5): {exp_x2:.10f}")
    
    # Test 3: E[exp(X)] - important for log-normal
    g3 = lambda x: np.exp(x)
    exp_exp = gauss_hermite_expectation(g3, mu, sigma, 20)
    true_exp = np.exp(mu + sigma**2 / 2)  # MGF at t=1
    print(f"\nE[exp(X)]:")
    print(f"  True:      {true_exp:.6f}")
    print(f"  GH (n=20): {exp_exp:.6f}")
    print(f"  Error:     {abs(exp_exp - true_exp):.2e}")


def example_5_monte_carlo():
    """
    Example 5: Monte Carlo Integration
    ==================================
    Essential for high-dimensional integrals.
    """
    print("\n" + "=" * 60)
    print("Example 5: Monte Carlo Integration")
    print("=" * 60)
    
    np.random.seed(42)
    
    def monte_carlo_integrate(f, a, b, n_samples):
        """Basic Monte Carlo integration on [a,b]."""
        x = np.random.uniform(a, b, n_samples)
        fx = f(x)
        
        estimate = (b - a) * np.mean(fx)
        std_error = (b - a) * np.std(fx) / np.sqrt(n_samples)
        
        return estimate, std_error
    
    # Test: ∫₀^π sin(x) dx = 2
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    
    print(f"Integral: ∫₀^π sin(x) dx = {exact}")
    print(f"\n{'Samples':<12} {'Estimate':<15} {'Std Error':<12} {'Actual Error':<12}")
    print("-" * 55)
    
    for n in [100, 1000, 10000, 100000, 1000000]:
        estimate, std_err = monte_carlo_integrate(f, a, b, n)
        actual_err = abs(estimate - exact)
        print(f"{n:<12} {estimate:<15.6f} {std_err:<12.4f} {actual_err:<12.4f}")
    
    print(f"\nNote: Error decreases as O(1/√N) - doubles samples halves std error")


def example_6_importance_sampling():
    """
    Example 6: Importance Sampling
    ==============================
    Reduce variance by sampling from better distribution.
    """
    print("\n" + "=" * 60)
    print("Example 6: Importance Sampling")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Compute E[exp(-X)] where X ~ Uniform(0, 10)
    # True value: (1/10) * ∫₀^10 e^(-x) dx = (1 - e^(-10))/10 ≈ 0.09999546
    
    true_value = (1 - np.exp(-10)) / 10
    n_samples = 10000
    
    # Method 1: Naive Monte Carlo
    x_uniform = np.random.uniform(0, 10, n_samples)
    naive_estimate = np.mean(np.exp(-x_uniform))
    naive_var = np.var(np.exp(-x_uniform))
    
    # Method 2: Importance Sampling with Exponential(λ) proposal
    # Sample from Exp(λ), truncated to [0, 10]
    lam = 0.5  # Rate parameter
    
    # Sample from truncated exponential
    u = np.random.uniform(0, 1 - np.exp(-lam * 10), n_samples)
    x_exp = -np.log(1 - u) / lam
    
    # Importance weights: p(x)/q(x) where p(x) = 1/10, q(x) = λe^(-λx) / (1-e^(-10λ))
    normalizer = 1 - np.exp(-lam * 10)
    proposal_density = lam * np.exp(-lam * x_exp) / normalizer
    target_density = 0.1  # Uniform on [0,10]
    weights = target_density / proposal_density
    
    is_estimate = np.mean(weights * np.exp(-x_exp))
    is_var = np.var(weights * np.exp(-x_exp))
    
    print(f"Estimating E[e^(-X)] where X ~ Uniform(0, 10)")
    print(f"True value: {true_value:.6f}")
    print(f"\n{'Method':<25} {'Estimate':<12} {'Variance':<15} {'Error':<10}")
    print("-" * 65)
    print(f"{'Naive MC':<25} {naive_estimate:<12.6f} {naive_var:<15.6f} {abs(naive_estimate-true_value):<10.4f}")
    print(f"{'Importance Sampling':<25} {is_estimate:<12.6f} {is_var:<15.6f} {abs(is_estimate-true_value):<10.4f}")
    print(f"\nVariance reduction factor: {naive_var/is_var:.2f}x")


def example_7_multidimensional_mc():
    """
    Example 7: Multi-dimensional Monte Carlo
    ========================================
    MC becomes advantageous in high dimensions.
    """
    print("\n" + "=" * 60)
    print("Example 7: Multi-dimensional Integration")
    print("=" * 60)
    
    np.random.seed(42)
    
    def monte_carlo_nd(f, dim, n_samples):
        """Monte Carlo integration over [0,1]^dim."""
        x = np.random.uniform(0, 1, (n_samples, dim))
        fx = np.array([f(xi) for xi in x])
        return np.mean(fx), np.std(fx) / np.sqrt(n_samples)
    
    # Test function: ∫_{[0,1]^d} exp(-||x||²) dx
    def integrand(x):
        return np.exp(-np.sum(x**2))
    
    print(f"Integral: ∫_[0,1]^d exp(-||x||²) dx")
    print(f"\n{'Dimension':<12} {'MC Estimate':<15} {'Std Error':<12}")
    print("-" * 40)
    
    n_samples = 100000
    
    for dim in [1, 2, 5, 10, 20]:
        estimate, std_err = monte_carlo_nd(integrand, dim, n_samples)
        print(f"{dim:<12} {estimate:<15.6f} {std_err:<12.6f}")
    
    # Compare with tensor product rule for low dimensions
    print(f"\nComparison with tensor product rule (2D):")
    
    # Tensor product Simpson (5 points per dimension = 25 evaluations)
    from scipy import integrate
    result_2d, _ = integrate.dblquad(lambda y, x: np.exp(-x**2 - y**2),
                                      0, 1, lambda x: 0, lambda x: 1)
    print(f"  scipy.integrate: {result_2d:.8f}")
    
    mc_2d, mc_err = monte_carlo_nd(integrand, 2, 100000)
    print(f"  MC (100K samples): {mc_2d:.8f} ± {mc_err:.6f}")


def example_8_adaptive_quadrature():
    """
    Example 8: Adaptive Quadrature
    ==============================
    Automatically refine where needed.
    """
    print("\n" + "=" * 60)
    print("Example 8: Adaptive Quadrature")
    print("=" * 60)
    
    # Function with rapid variation in one region
    def challenging_func(x):
        return np.sin(10 / (x + 0.1))
    
    a, b = 0, 1
    
    # Using scipy's adaptive quadrature
    result, error = integrate.quad(challenging_func, a, b)
    
    print(f"Integrating sin(10/(x+0.1)) from 0 to 1")
    print(f"(Oscillates rapidly near x=0)")
    print(f"\nAdaptive quadrature result: {result:.10f}")
    print(f"Estimated error: {error:.2e}")
    
    # Compare with fixed-step methods
    for n in [10, 100, 1000]:
        x = np.linspace(a, b, n+1)
        h = (b - a) / n
        y = challenging_func(x)
        trap = h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])
        print(f"Trapezoidal (n={n}): {trap:.10f}, error ≈ {abs(trap-result):.2e}")


def example_9_romberg_integration():
    """
    Example 9: Romberg Integration
    ==============================
    Extrapolation for improved accuracy.
    """
    print("\n" + "=" * 60)
    print("Example 9: Romberg Integration")
    print("=" * 60)
    
    def romberg(f, a, b, max_iter=10, tol=1e-12):
        """Romberg integration using Richardson extrapolation."""
        R = np.zeros((max_iter, max_iter))
        
        # First estimate: single trapezoid
        h = b - a
        R[0, 0] = h / 2 * (f(a) + f(b))
        
        for i in range(1, max_iter):
            # Composite trapezoidal with 2^i intervals
            h = (b - a) / (2**i)
            
            # Add new midpoints
            new_points = np.sum([f(a + (2*k + 1) * h) for k in range(2**(i-1))])
            R[i, 0] = 0.5 * R[i-1, 0] + h * new_points
            
            # Richardson extrapolation
            for j in range(1, i + 1):
                R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
            # Check convergence
            if i > 0 and abs(R[i, i] - R[i-1, i-1]) < tol:
                return R[i, i], i
        
        return R[max_iter-1, max_iter-1], max_iter
    
    # Test: ∫₀^π sin(x) dx = 2
    f = np.sin
    a, b = 0, np.pi
    exact = 2.0
    
    result, iterations = romberg(f, a, b)
    
    print(f"Integral: ∫₀^π sin(x) dx = {exact}")
    print(f"\nRomberg result: {result:.15f}")
    print(f"Error: {abs(result - exact):.2e}")
    print(f"Iterations: {iterations}")
    
    # Show convergence
    print(f"\nConvergence of Romberg method:")
    R = np.zeros((6, 6))
    h = b - a
    R[0, 0] = h / 2 * (f(a) + f(b))
    
    for i in range(1, 6):
        h = (b - a) / (2**i)
        new_points = np.sum([f(a + (2*k + 1) * h) for k in range(2**(i-1))])
        R[i, 0] = 0.5 * R[i-1, 0] + h * new_points
        
        for j in range(1, i + 1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
    
    print(f"\n{'k':<4}", end="")
    for j in range(6):
        print(f"{'R[k,'+str(j)+']':<15}", end="")
    print()
    print("-" * 95)
    
    for i in range(6):
        print(f"{i:<4}", end="")
        for j in range(i + 1):
            print(f"{R[i,j]:<15.10f}", end="")
        print()


def example_10_variance_reduction():
    """
    Example 10: Variance Reduction Techniques
    =========================================
    Methods to improve Monte Carlo efficiency.
    """
    print("\n" + "=" * 60)
    print("Example 10: Variance Reduction")
    print("=" * 60)
    
    np.random.seed(42)
    n_samples = 10000
    
    # Estimate ∫₀^1 e^x dx = e - 1 ≈ 1.71828
    exact = np.exp(1) - 1
    
    # Method 1: Simple MC
    x_simple = np.random.uniform(0, 1, n_samples)
    simple_estimate = np.mean(np.exp(x_simple))
    simple_var = np.var(np.exp(x_simple))
    
    # Method 2: Antithetic variates
    # Use x and 1-x (negatively correlated)
    x_anti = np.random.uniform(0, 1, n_samples // 2)
    anti_estimate = 0.5 * np.mean(np.exp(x_anti) + np.exp(1 - x_anti))
    anti_var = np.var(0.5 * (np.exp(x_anti) + np.exp(1 - x_anti)))
    
    # Method 3: Control variates
    # Use known integral: ∫₀^1 x dx = 0.5
    # Correlated with e^x
    x_cv = np.random.uniform(0, 1, n_samples)
    # Optimal c ≈ Cov(e^x, x) / Var(x)
    cov_est = np.cov(np.exp(x_cv), x_cv)[0, 1]
    var_x = np.var(x_cv)
    c_opt = cov_est / var_x
    cv_estimate = np.mean(np.exp(x_cv) - c_opt * (x_cv - 0.5))
    cv_var = np.var(np.exp(x_cv) - c_opt * (x_cv - 0.5))
    
    print(f"Estimating ∫₀^1 e^x dx = {exact:.6f}")
    print(f"\n{'Method':<25} {'Estimate':<12} {'Variance':<15} {'Var. Reduction':<15}")
    print("-" * 70)
    print(f"{'Simple MC':<25} {simple_estimate:<12.6f} {simple_var:<15.6f} {'1x (baseline)':<15}")
    print(f"{'Antithetic Variates':<25} {anti_estimate:<12.6f} {anti_var:<15.6f} {simple_var/anti_var:<15.2f}x")
    print(f"{'Control Variates':<25} {cv_estimate:<12.6f} {cv_var:<15.6f} {simple_var/cv_var:<15.2f}x")


def example_11_bayesian_evidence():
    """
    Example 11: Computing Bayesian Evidence
    =======================================
    Marginal likelihood via numerical integration.
    """
    print("\n" + "=" * 60)
    print("Example 11: Bayesian Evidence Computation")
    print("=" * 60)
    
    # Simple example: Gaussian likelihood, Gaussian prior
    # Data: x₁, ..., xₙ ~ N(μ, σ²)
    # Prior: μ ~ N(μ₀, τ²)
    # Evidence: p(data) = ∫ p(data|μ) p(μ) dμ
    
    # Generate data
    np.random.seed(42)
    true_mu = 2.0
    sigma = 1.0  # Known variance
    n_data = 10
    data = np.random.normal(true_mu, sigma, n_data)
    x_bar = np.mean(data)
    
    # Prior parameters
    mu_0 = 0.0
    tau = 2.0  # Prior std
    
    # Exact evidence (conjugate case)
    posterior_var = 1 / (n_data / sigma**2 + 1 / tau**2)
    posterior_mean = posterior_var * (n_data * x_bar / sigma**2 + mu_0 / tau**2)
    
    # Evidence formula for conjugate case
    exact_log_evidence = (
        -n_data / 2 * np.log(2 * np.pi * sigma**2)
        - np.sum((data - x_bar)**2) / (2 * sigma**2)
        - 0.5 * np.log(n_data * tau**2 / sigma**2 + 1)
        - (x_bar - mu_0)**2 / (2 * (sigma**2 / n_data + tau**2))
    )
    
    print(f"Data: {n_data} observations from N({true_mu}, {sigma}²)")
    print(f"Prior: μ ~ N({mu_0}, {tau}²)")
    print(f"\nAnalytical log-evidence: {exact_log_evidence:.6f}")
    
    # Numerical integration
    def integrand(mu):
        """p(data|μ) × p(μ)"""
        log_likelihood = -0.5 * np.sum((data - mu)**2) / sigma**2 - n_data/2 * np.log(2*np.pi*sigma**2)
        log_prior = -0.5 * (mu - mu_0)**2 / tau**2 - 0.5 * np.log(2*np.pi*tau**2)
        return np.exp(log_likelihood + log_prior)
    
    # Gauss-Hermite centered at posterior mean
    n_points = 20
    nodes, weights = roots_hermite(n_points)
    
    # Scale nodes to cover posterior
    post_std = np.sqrt(posterior_var)
    mu_points = posterior_mean + np.sqrt(2) * post_std * 3 * nodes
    
    numerical_evidence = 0
    for mu, w in zip(mu_points, weights):
        numerical_evidence += w * integrand(mu) * np.sqrt(2) * post_std * 3 / np.sqrt(np.pi)
    
    numerical_log_evidence = np.log(numerical_evidence)
    
    print(f"Numerical log-evidence: {numerical_log_evidence:.6f}")
    print(f"Difference: {abs(numerical_log_evidence - exact_log_evidence):.2e}")


def example_12_ml_expectation():
    """
    Example 12: Computing Expectations in ML
    ========================================
    Common integrals in machine learning.
    """
    print("\n" + "=" * 60)
    print("Example 12: ML Expectations")
    print("=" * 60)
    
    # Example: Expected activation in neural network
    # Input X ~ N(0, 1), compute E[ReLU(X)]
    
    # Analytical: E[max(0, X)] = φ(0) + 0·Φ(0) = 1/√(2π)
    exact_relu = 1 / np.sqrt(2 * np.pi)
    
    # Gauss-Hermite
    nodes, weights = roots_hermite(20)
    x = np.sqrt(2) * nodes  # Transform to N(0,1)
    relu_values = np.maximum(0, x)
    gh_estimate = np.sum(weights * relu_values) / np.sqrt(np.pi)
    
    print(f"E[ReLU(X)] where X ~ N(0, 1)")
    print(f"  Analytical: {exact_relu:.8f}")
    print(f"  Gauss-Hermite (n=20): {gh_estimate:.8f}")
    
    # E[sigmoid(X)] for X ~ N(μ, σ²)
    mu, sigma = 1.0, 2.0
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Monte Carlo
    np.random.seed(42)
    x_samples = np.random.normal(mu, sigma, 100000)
    mc_sigmoid = np.mean(sigmoid(x_samples))
    
    # Gauss-Hermite
    nodes, weights = roots_hermite(30)
    x_gh = np.sqrt(2) * sigma * nodes + mu
    gh_sigmoid = np.sum(weights * sigmoid(x_gh)) / np.sqrt(np.pi)
    
    print(f"\nE[σ(X)] where X ~ N({mu}, {sigma}²)")
    print(f"  Monte Carlo (100K): {mc_sigmoid:.8f}")
    print(f"  Gauss-Hermite (n=30): {gh_sigmoid:.8f}")
    
    # Softmax normalization
    print(f"\nE[exp(X)] / Σ E[exp(X_i)] - softmax output expectation")
    print("  Commonly computed via Gauss-Hermite or MC in variational inference")


def run_all_examples():
    """Run all examples."""
    example_1_newton_cotes()
    example_2_composite_rules()
    example_3_gauss_legendre()
    example_4_gauss_hermite()
    example_5_monte_carlo()
    example_6_importance_sampling()
    example_7_multidimensional_mc()
    example_8_adaptive_quadrature()
    example_9_romberg_integration()
    example_10_variance_reduction()
    example_11_bayesian_evidence()
    example_12_ml_expectation()


if __name__ == "__main__":
    run_all_examples()
