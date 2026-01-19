"""
Numerical Integration - Exercises
=================================
Practice problems for quadrature methods.
"""

import numpy as np
from scipy import integrate
from scipy.special import roots_legendre, roots_hermite, roots_laguerre


class Exercise1:
    """
    Exercise 1: Composite Quadrature Implementation
    ===============================================
    
    Implement composite Newton-Cotes rules:
    1. Composite trapezoidal rule
    2. Composite Simpson's rule
    3. Convergence analysis
    """
    
    @staticmethod
    def composite_trapezoidal(f, a, b, n):
        """
        Composite trapezoidal rule.
        
        ∫_a^b f(x)dx ≈ h/2 [f(a) + 2Σf(x_i) + f(b)]
        """
        # YOUR CODE HERE
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        
        result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
        return result
    
    @staticmethod
    def composite_simpson(f, a, b, n):
        """
        Composite Simpson's rule (n must be even).
        
        ∫_a^b f(x)dx ≈ h/3 [f(a) + 4Σf(odd) + 2Σf(even) + f(b)]
        """
        # YOUR CODE HERE
        if n % 2 != 0:
            n += 1
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        
        result = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
        return result
    
    @staticmethod
    def convergence_analysis(f, a, b, exact, rule_func, n_values):
        """Analyze convergence rate of quadrature rule."""
        # YOUR CODE HERE
        errors = []
        for n in n_values:
            approx = rule_func(f, a, b, n)
            errors.append(abs(approx - exact))
        
        # Estimate convergence rate: error ≈ C * h^p
        rates = []
        for i in range(1, len(errors)):
            if errors[i] > 0 and errors[i-1] > 0:
                rate = np.log(errors[i-1] / errors[i]) / np.log(n_values[i] / n_values[i-1])
                rates.append(rate)
        
        return errors, np.mean(rates) if rates else 0
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("Exercise 1: Composite Quadrature")
        print("-" * 40)
        
        # Test function: ∫₀^π sin(x) dx = 2
        f = np.sin
        a, b = 0, np.pi
        exact = 2.0
        
        n_values = [4, 8, 16, 32, 64, 128]
        
        print(f"{'n':<8} {'Trapezoidal':<15} {'Simpson':<15}")
        print("-" * 40)
        
        for n in n_values:
            trap = Exercise1.composite_trapezoidal(f, a, b, n)
            simp = Exercise1.composite_simpson(f, a, b, n)
            print(f"{n:<8} {abs(trap-exact):<15.2e} {abs(simp-exact):<15.2e}")
        
        # Convergence rates
        trap_errors, trap_rate = Exercise1.convergence_analysis(
            f, a, b, exact, Exercise1.composite_trapezoidal, n_values)
        simp_errors, simp_rate = Exercise1.convergence_analysis(
            f, a, b, exact, Exercise1.composite_simpson, n_values)
        
        print(f"\nConvergence rates:")
        print(f"  Trapezoidal: {trap_rate:.2f} (expected: 2)")
        print(f"  Simpson's:   {simp_rate:.2f} (expected: 4)")


class Exercise2:
    """
    Exercise 2: Gauss-Legendre Quadrature
    =====================================
    
    Implement Gauss-Legendre quadrature from scratch:
    1. Compute nodes (roots of Legendre polynomials)
    2. Compute weights
    3. General interval transformation
    """
    
    @staticmethod
    def legendre_polynomial(n, x):
        """Evaluate Legendre polynomial P_n(x) using recursion."""
        # YOUR CODE HERE
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x
        
        P_prev = np.ones_like(x)
        P_curr = x
        
        for k in range(2, n + 1):
            P_next = ((2*k - 1) * x * P_curr - (k - 1) * P_prev) / k
            P_prev = P_curr
            P_curr = P_next
        
        return P_curr
    
    @staticmethod
    def legendre_derivative(n, x):
        """Compute P'_n(x)."""
        # YOUR CODE HERE
        if n == 0:
            return np.zeros_like(x)
        
        # P'_n(x) = n/(x² - 1) * [x*P_n(x) - P_{n-1}(x)]
        P_n = Exercise2.legendre_polynomial(n, x)
        P_nm1 = Exercise2.legendre_polynomial(n - 1, x)
        
        return n / (x**2 - 1) * (x * P_n - P_nm1)
    
    @staticmethod
    def gauss_legendre_nodes_weights(n):
        """
        Compute Gauss-Legendre nodes and weights.
        """
        # YOUR CODE HERE
        # Initial guesses for roots
        nodes = np.cos(np.pi * (np.arange(n) + 0.75) / (n + 0.5))
        
        # Newton-Raphson iteration
        for _ in range(20):
            P_n = Exercise2.legendre_polynomial(n, nodes)
            P_n_prime = Exercise2.legendre_derivative(n, nodes)
            nodes = nodes - P_n / P_n_prime
        
        # Compute weights
        P_n_prime = Exercise2.legendre_derivative(n, nodes)
        weights = 2 / ((1 - nodes**2) * P_n_prime**2)
        
        return np.sort(nodes), weights[np.argsort(nodes)]
    
    @staticmethod
    def gauss_legendre_integrate(f, a, b, n):
        """Integrate f over [a, b] using n-point Gauss-Legendre."""
        # YOUR CODE HERE
        nodes, weights = Exercise2.gauss_legendre_nodes_weights(n)
        
        # Transform from [-1, 1] to [a, b]
        x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
        scale = 0.5 * (b - a)
        
        return scale * np.sum(weights * f(x))
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 2: Gauss-Legendre Quadrature")
        print("-" * 40)
        
        # Compare with scipy
        for n in [3, 5, 7]:
            our_nodes, our_weights = Exercise2.gauss_legendre_nodes_weights(n)
            scipy_nodes, scipy_weights = roots_legendre(n)
            
            print(f"\nn = {n}:")
            print(f"  Nodes match: {np.allclose(our_nodes, scipy_nodes)}")
            print(f"  Weights match: {np.allclose(our_weights, scipy_weights)}")
        
        # Test integration
        f = np.sin
        a, b = 0, np.pi
        exact = 2.0
        
        print(f"\nIntegrating sin(x) from 0 to π:")
        for n in [2, 3, 5, 10]:
            result = Exercise2.gauss_legendre_integrate(f, a, b, n)
            print(f"  n = {n}: {result:.12f}, error = {abs(result - exact):.2e}")


class Exercise3:
    """
    Exercise 3: Gauss-Hermite for ML
    ================================
    
    Apply Gauss-Hermite quadrature to machine learning problems:
    1. Expected activation value
    2. Expected loss
    3. Variational inference
    """
    
    @staticmethod
    def gauss_hermite_expectation(g, mu, sigma, n):
        """
        Compute E[g(X)] where X ~ N(mu, sigma²).
        
        Transform: ∫ g(x) N(x|μ,σ²) dx = (1/√π) ∫ g(√2·σ·t + μ) e^(-t²) dt
        """
        # YOUR CODE HERE
        nodes, weights = roots_hermite(n)
        
        x = np.sqrt(2) * sigma * nodes + mu
        
        return np.sum(weights * g(x)) / np.sqrt(np.pi)
    
    @staticmethod
    def expected_relu(mu, sigma, n=30):
        """E[ReLU(X)] where X ~ N(mu, sigma²)."""
        # YOUR CODE HERE
        relu = lambda x: np.maximum(0, x)
        return Exercise3.gauss_hermite_expectation(relu, mu, sigma, n)
    
    @staticmethod
    def expected_softplus(mu, sigma, n=30):
        """E[softplus(X)] where X ~ N(mu, sigma²)."""
        # YOUR CODE HERE
        softplus = lambda x: np.log1p(np.exp(np.clip(x, -500, 500)))
        return Exercise3.gauss_hermite_expectation(softplus, mu, sigma, n)
    
    @staticmethod
    def expected_cross_entropy(mu, sigma, y, n=30):
        """
        E[-y*log(σ(X)) - (1-y)*log(1-σ(X))] where X ~ N(mu, sigma²).
        
        This is the expected binary cross-entropy loss.
        """
        # YOUR CODE HERE
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def bce(x):
            s = sigmoid(x)
            return -y * np.log(s + 1e-10) - (1 - y) * np.log(1 - s + 1e-10)
        
        return Exercise3.gauss_hermite_expectation(bce, mu, sigma, n)
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 3: Gauss-Hermite for ML")
        print("-" * 40)
        
        np.random.seed(42)
        
        # Test E[X] and E[X²]
        mu, sigma = 1.5, 0.8
        
        exp_x = Exercise3.gauss_hermite_expectation(lambda x: x, mu, sigma, 20)
        exp_x2 = Exercise3.gauss_hermite_expectation(lambda x: x**2, mu, sigma, 20)
        
        print(f"X ~ N({mu}, {sigma}²)")
        print(f"E[X] = {exp_x:.6f} (true: {mu})")
        print(f"E[X²] = {exp_x2:.6f} (true: {mu**2 + sigma**2})")
        
        # Expected ReLU
        relu_exp = Exercise3.expected_relu(mu, sigma)
        
        # Monte Carlo verification
        samples = np.random.normal(mu, sigma, 100000)
        mc_relu = np.mean(np.maximum(0, samples))
        
        print(f"\nE[ReLU(X)]:")
        print(f"  Gauss-Hermite: {relu_exp:.6f}")
        print(f"  Monte Carlo:   {mc_relu:.6f}")
        
        # Expected cross-entropy
        for y in [0, 1]:
            bce = Exercise3.expected_cross_entropy(mu, sigma, y)
            mc_bce = np.mean(-y * np.log(1/(1+np.exp(-samples)) + 1e-10) 
                           - (1-y) * np.log(1 - 1/(1+np.exp(-samples)) + 1e-10))
            print(f"\nE[BCE(y={y})]:")
            print(f"  Gauss-Hermite: {bce:.6f}")
            print(f"  Monte Carlo:   {mc_bce:.6f}")


class Exercise4:
    """
    Exercise 4: Monte Carlo Integration
    ===================================
    
    Implement Monte Carlo integration with:
    1. Basic Monte Carlo
    2. Stratified sampling
    3. Confidence intervals
    """
    
    @staticmethod
    def basic_monte_carlo(f, a, b, n_samples):
        """
        Basic Monte Carlo integration.
        
        Returns (estimate, standard_error).
        """
        # YOUR CODE HERE
        np.random.seed()  # Allow different results each call
        x = np.random.uniform(a, b, n_samples)
        fx = f(x)
        
        estimate = (b - a) * np.mean(fx)
        std_error = (b - a) * np.std(fx) / np.sqrt(n_samples)
        
        return estimate, std_error
    
    @staticmethod
    def stratified_monte_carlo(f, a, b, n_samples, n_strata):
        """
        Stratified Monte Carlo integration.
        
        Divide [a,b] into n_strata equal intervals.
        Sample equally from each stratum.
        """
        # YOUR CODE HERE
        samples_per_stratum = n_samples // n_strata
        stratum_width = (b - a) / n_strata
        
        total = 0
        var_sum = 0
        
        for i in range(n_strata):
            stratum_a = a + i * stratum_width
            stratum_b = stratum_a + stratum_width
            
            x = np.random.uniform(stratum_a, stratum_b, samples_per_stratum)
            fx = f(x)
            
            total += stratum_width * np.mean(fx)
            var_sum += (stratum_width * np.std(fx) / np.sqrt(samples_per_stratum))**2
        
        estimate = total
        std_error = np.sqrt(var_sum)
        
        return estimate, std_error
    
    @staticmethod
    def confidence_interval(estimate, std_error, confidence=0.95):
        """Compute confidence interval."""
        # YOUR CODE HERE
        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        
        lower = estimate - z * std_error
        upper = estimate + z * std_error
        
        return lower, upper
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 4: Monte Carlo Integration")
        print("-" * 40)
        
        np.random.seed(42)
        
        # Test function: ∫₀^1 exp(x) dx = e - 1
        f = np.exp
        a, b = 0, 1
        exact = np.exp(1) - 1
        
        n_samples = 10000
        
        # Basic MC
        basic_est, basic_se = Exercise4.basic_monte_carlo(f, a, b, n_samples)
        basic_ci = Exercise4.confidence_interval(basic_est, basic_se)
        
        # Stratified MC
        strat_est, strat_se = Exercise4.stratified_monte_carlo(f, a, b, n_samples, 10)
        strat_ci = Exercise4.confidence_interval(strat_est, strat_se)
        
        print(f"Integral: ∫₀^1 e^x dx = {exact:.6f}")
        print(f"\nBasic MC ({n_samples} samples):")
        print(f"  Estimate: {basic_est:.6f} ± {basic_se:.4f}")
        print(f"  95% CI: ({basic_ci[0]:.6f}, {basic_ci[1]:.6f})")
        print(f"  Contains true: {basic_ci[0] <= exact <= basic_ci[1]}")
        
        print(f"\nStratified MC (10 strata):")
        print(f"  Estimate: {strat_est:.6f} ± {strat_se:.4f}")
        print(f"  95% CI: ({strat_ci[0]:.6f}, {strat_ci[1]:.6f})")
        print(f"  Variance reduction: {(basic_se/strat_se)**2:.2f}x")


class Exercise5:
    """
    Exercise 5: Importance Sampling
    ===============================
    
    Implement importance sampling for:
    1. Tail probability estimation
    2. Rare event simulation
    3. Optimal proposal distribution
    """
    
    @staticmethod
    def importance_sampling(f, target_pdf, proposal_sampler, proposal_pdf, n_samples):
        """
        Importance sampling estimate of E_target[f(X)].
        
        Returns (estimate, effective_sample_size).
        """
        # YOUR CODE HERE
        x = proposal_sampler(n_samples)
        
        weights = target_pdf(x) / proposal_pdf(x)
        
        estimate = np.sum(weights * f(x)) / np.sum(weights)
        
        # Effective sample size
        ess = np.sum(weights)**2 / np.sum(weights**2)
        
        return estimate, ess
    
    @staticmethod
    def tail_probability_estimation(threshold, n_samples):
        """
        Estimate P(X > threshold) where X ~ N(0,1).
        
        Use importance sampling with shifted Gaussian.
        """
        # YOUR CODE HERE
        from scipy.stats import norm
        
        # Naive MC
        np.random.seed(42)
        x_naive = np.random.randn(n_samples)
        naive_estimate = np.mean(x_naive > threshold)
        
        # Importance sampling with N(threshold, 1)
        x_is = np.random.randn(n_samples) + threshold
        
        # Weights: φ(x) / φ(x - threshold)
        weights = np.exp(-0.5 * x_is**2) / np.exp(-0.5 * (x_is - threshold)**2)
        
        is_estimate = np.sum(weights * (x_is > threshold)) / n_samples
        
        # True value
        true_prob = 1 - norm.cdf(threshold)
        
        return naive_estimate, is_estimate, true_prob
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 5: Importance Sampling")
        print("-" * 40)
        
        # Tail probability
        n_samples = 100000
        
        print("Tail probability P(X > c) for X ~ N(0,1):")
        print(f"\n{'c':<8} {'True':<15} {'Naive MC':<15} {'IS':<15}")
        print("-" * 55)
        
        for threshold in [2, 3, 4]:
            naive, is_est, true_prob = Exercise5.tail_probability_estimation(threshold, n_samples)
            print(f"{threshold:<8} {true_prob:<15.8f} {naive:<15.8f} {is_est:<15.8f}")
        
        print(f"\nNote: IS works much better for rare events (large c)")


class Exercise6:
    """
    Exercise 6: Adaptive Quadrature
    ===============================
    
    Implement adaptive Simpson's rule:
    1. Error estimation
    2. Recursive subdivision
    3. Error tolerance control
    """
    
    @staticmethod
    def simpson_rule(f, a, b):
        """Single Simpson's rule on [a, b]."""
        return (b - a) / 6 * (f(a) + 4 * f((a + b) / 2) + f(b))
    
    @staticmethod
    def adaptive_simpson(f, a, b, tol=1e-10, max_depth=50):
        """
        Adaptive Simpson's rule with error control.
        
        Returns (integral, function_evaluations).
        """
        # YOUR CODE HERE
        def adaptive_helper(a, b, tol, S_ab, depth, f_evals):
            c = (a + b) / 2
            
            S_ac = Exercise6.simpson_rule(f, a, c)
            S_cb = Exercise6.simpson_rule(f, c, b)
            f_evals[0] += 4  # New function evaluations
            
            # Error estimate
            error = abs(S_ab - (S_ac + S_cb)) / 15
            
            if error < tol or depth >= max_depth:
                return S_ac + S_cb + (S_ac + S_cb - S_ab) / 15
            else:
                left = adaptive_helper(a, c, tol/2, S_ac, depth + 1, f_evals)
                right = adaptive_helper(c, b, tol/2, S_cb, depth + 1, f_evals)
                return left + right
        
        S_ab = Exercise6.simpson_rule(f, a, b)
        f_evals = [3]  # Initial evaluations
        
        result = adaptive_helper(a, b, tol, S_ab, 0, f_evals)
        
        return result, f_evals[0]
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 6: Adaptive Quadrature")
        print("-" * 40)
        
        # Difficult function: oscillates rapidly near 0
        def challenging(x):
            return np.sin(10 / (x + 0.1))
        
        a, b = 0, 1
        
        # scipy reference
        scipy_result, _ = integrate.quad(challenging, a, b)
        
        print(f"Integrating sin(10/(x+0.1)) from 0 to 1")
        print(f"\n{'Tolerance':<15} {'Result':<18} {'Error':<12} {'Func Evals'}")
        print("-" * 60)
        
        for tol in [1e-4, 1e-6, 1e-8, 1e-10]:
            result, f_evals = Exercise6.adaptive_simpson(challenging, a, b, tol)
            error = abs(result - scipy_result)
            print(f"{tol:<15.0e} {result:<18.12f} {error:<12.2e} {f_evals}")
        
        print(f"\nSciPy reference: {scipy_result:.12f}")


class Exercise7:
    """
    Exercise 7: Multi-dimensional Integration
    =========================================
    
    Compare methods for multi-dimensional integrals:
    1. Tensor product quadrature
    2. Monte Carlo
    3. Curse of dimensionality
    """
    
    @staticmethod
    def tensor_product_gauss(f, dim, n_points):
        """
        Tensor product Gauss-Legendre over [0,1]^dim.
        """
        # YOUR CODE HERE
        nodes_1d, weights_1d = roots_legendre(n_points)
        
        # Transform to [0, 1]
        nodes_1d = 0.5 * (nodes_1d + 1)
        weights_1d = 0.5 * weights_1d
        
        # Create tensor product grid
        from itertools import product
        
        grid_indices = list(product(range(n_points), repeat=dim))
        
        total = 0
        for idx in grid_indices:
            x = np.array([nodes_1d[i] for i in idx])
            w = np.prod([weights_1d[i] for i in idx])
            total += w * f(x)
        
        return total, n_points**dim
    
    @staticmethod
    def monte_carlo_nd(f, dim, n_samples):
        """Monte Carlo over [0,1]^dim."""
        # YOUR CODE HERE
        x = np.random.rand(n_samples, dim)
        fx = np.array([f(xi) for xi in x])
        
        return np.mean(fx), n_samples
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 7: Multi-dimensional Integration")
        print("-" * 40)
        
        np.random.seed(42)
        
        # Test function: ∫_{[0,1]^d} exp(-||x||²) dx
        def integrand(x):
            return np.exp(-np.sum(x**2))
        
        print(f"Integral: ∫_[0,1]^d exp(-||x||²) dx")
        print(f"\n{'Dim':<6} {'Tensor (n=5)':<15} {'Evals':<10} {'MC (10K)':<15}")
        print("-" * 50)
        
        for dim in [2, 3, 4, 5]:
            if 5**dim <= 100000:
                tensor_result, tensor_evals = Exercise7.tensor_product_gauss(integrand, dim, 5)
            else:
                tensor_result, tensor_evals = np.nan, "N/A"
            
            mc_result, mc_evals = Exercise7.monte_carlo_nd(integrand, dim, 10000)
            
            if np.isnan(tensor_result):
                print(f"{dim:<6} {'Too expensive':<15} {tensor_evals:<10} {mc_result:<15.6f}")
            else:
                print(f"{dim:<6} {tensor_result:<15.6f} {tensor_evals:<10} {mc_result:<15.6f}")
        
        print(f"\nNote: Tensor product cost explodes (5^d), MC stays at 10K")


class Exercise8:
    """
    Exercise 8: Romberg Integration
    ===============================
    
    Implement Romberg integration with Richardson extrapolation.
    """
    
    @staticmethod
    def romberg_integration(f, a, b, max_iter=10, tol=1e-12):
        """
        Romberg integration.
        
        Returns (result, Romberg_table, iterations).
        """
        # YOUR CODE HERE
        R = np.zeros((max_iter, max_iter))
        
        # First estimate
        h = b - a
        R[0, 0] = h / 2 * (f(a) + f(b))
        
        for i in range(1, max_iter):
            h = (b - a) / (2**i)
            
            # Add new midpoints
            new_sum = sum(f(a + (2*k + 1) * h) for k in range(2**(i-1)))
            R[i, 0] = 0.5 * R[i-1, 0] + h * new_sum
            
            # Richardson extrapolation
            for j in range(1, i + 1):
                R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4**j - 1)
            
            # Check convergence
            if abs(R[i, i] - R[i-1, i-1]) < tol:
                return R[i, i], R[:i+1, :i+1], i + 1
        
        return R[max_iter-1, max_iter-1], R, max_iter
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 8: Romberg Integration")
        print("-" * 40)
        
        # Test: ∫₀^1 e^x dx = e - 1
        f = np.exp
        a, b = 0, 1
        exact = np.exp(1) - 1
        
        result, R, iterations = Exercise8.romberg_integration(f, a, b)
        
        print(f"Integral: ∫₀^1 e^x dx = {exact:.12f}")
        print(f"Romberg result: {result:.12f}")
        print(f"Error: {abs(result - exact):.2e}")
        print(f"Iterations: {iterations}")
        
        print(f"\nRomberg table:")
        for i in range(min(6, R.shape[0])):
            print(f"  ", end="")
            for j in range(i + 1):
                print(f"{R[i, j]:.10f}  ", end="")
            print()


class Exercise9:
    """
    Exercise 9: Gauss-Laguerre for Semi-infinite Integrals
    ======================================================
    
    Use Gauss-Laguerre for integrals of form ∫₀^∞ f(x) e^(-x) dx.
    """
    
    @staticmethod
    def gauss_laguerre_integrate(f, n):
        """
        Compute ∫₀^∞ f(x) e^(-x) dx using Gauss-Laguerre.
        """
        # YOUR CODE HERE
        nodes, weights = roots_laguerre(n)
        return np.sum(weights * f(nodes))
    
    @staticmethod
    def gamma_function(a, n=30):
        """
        Compute Γ(a) = ∫₀^∞ x^(a-1) e^(-x) dx using Gauss-Laguerre.
        """
        # YOUR CODE HERE
        return Exercise9.gauss_laguerre_integrate(lambda x: x**(a-1), n)
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 9: Gauss-Laguerre Quadrature")
        print("-" * 40)
        
        from scipy.special import gamma
        
        print("Computing Gamma function:")
        print(f"\n{'a':<8} {'GL Estimate':<15} {'True':<15} {'Error'}")
        print("-" * 50)
        
        for a in [0.5, 1, 2, 5, 10]:
            gl_result = Exercise9.gamma_function(a)
            true_val = gamma(a)
            error = abs(gl_result - true_val)
            print(f"{a:<8.1f} {gl_result:<15.8f} {true_val:<15.8f} {error:.2e}")
        
        # Also test ∫₀^∞ x e^(-x) dx = 1
        integral = Exercise9.gauss_laguerre_integrate(lambda x: x, 10)
        print(f"\n∫₀^∞ x e^(-x) dx = {integral:.10f} (true: 1)")


class Exercise10:
    """
    Exercise 10: Computing Normalizing Constants
    ============================================
    
    Compute partition functions and normalizing constants
    commonly needed in probabilistic ML.
    """
    
    @staticmethod
    def log_sum_exp(log_values):
        """Compute log(Σ exp(log_values)) stably."""
        # YOUR CODE HERE
        max_val = np.max(log_values)
        return max_val + np.log(np.sum(np.exp(log_values - max_val)))
    
    @staticmethod
    def log_normalizing_constant_1d(log_unnormalized, a, b, n_points=100):
        """
        Compute log of normalizing constant Z = ∫_a^b exp(f(x)) dx.
        
        Uses composite trapezoidal with log-sum-exp for stability.
        """
        # YOUR CODE HERE
        x = np.linspace(a, b, n_points)
        h = (b - a) / (n_points - 1)
        
        log_f = log_unnormalized(x)
        
        # Trapezoidal: h/2 * [f(a) + 2*Σf(x_i) + f(b)]
        # In log: log(h/2) + log_sum_exp of terms
        
        # Adjust for trapezoidal weights
        log_f[0] -= np.log(2)
        log_f[-1] -= np.log(2)
        
        log_Z = np.log(h) + Exercise10.log_sum_exp(log_f)
        
        return log_Z
    
    @staticmethod
    def gaussian_partition_function(mu, sigma):
        """
        Compute partition function of Gaussian.
        
        Z = ∫ exp(-0.5 * (x-μ)²/σ²) dx = σ√(2π)
        """
        # YOUR CODE HERE
        # Numerically (for demonstration)
        log_unnorm = lambda x: -0.5 * ((x - mu) / sigma)**2
        
        # Integrate over wide range
        a, b = mu - 10*sigma, mu + 10*sigma
        log_Z_numerical = Exercise10.log_normalizing_constant_1d(log_unnorm, a, b, 1000)
        
        # Analytical
        log_Z_analytical = np.log(sigma) + 0.5 * np.log(2 * np.pi)
        
        return log_Z_numerical, log_Z_analytical
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 10: Normalizing Constants")
        print("-" * 40)
        
        # Test log_sum_exp
        log_values = np.array([1000, 1001, 999])  # Would overflow with naive approach
        lse = Exercise10.log_sum_exp(log_values)
        print(f"log_sum_exp([1000, 1001, 999]) = {lse:.6f}")
        print(f"  Expected ≈ 1001.41")
        
        # Gaussian partition function
        print(f"\nGaussian partition functions:")
        for mu, sigma in [(0, 1), (5, 2), (0, 0.5)]:
            numerical, analytical = Exercise10.gaussian_partition_function(mu, sigma)
            print(f"  N({mu}, {sigma}²): numerical = {numerical:.6f}, "
                  f"analytical = {analytical:.6f}, diff = {abs(numerical-analytical):.2e}")


def run_all_exercises():
    """Run all exercises."""
    Exercise1.verify()
    Exercise2.verify()
    Exercise3.verify()
    Exercise4.verify()
    Exercise5.verify()
    Exercise6.verify()
    Exercise7.verify()
    Exercise8.verify()
    Exercise9.verify()
    Exercise10.verify()


if __name__ == "__main__":
    run_all_exercises()
