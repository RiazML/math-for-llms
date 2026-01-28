"""
Interpolation and Approximation - Exercises
============================================
Practice problems for interpolation methods.
"""

import numpy as np
from scipy import interpolate


class Exercise1:
    """
    Exercise 1: Lagrange Interpolation from Scratch
    ===============================================
    
    Implement Lagrange interpolation with the following features:
    1. lagrange_basis(x_points, i, x) - compute i-th basis polynomial
    2. lagrange_interpolation(x_points, y_points, x) - evaluate at x
    3. lagrange_polynomial(x_points, y_points) - return callable
    
    Test on the points (0, 1), (1, 3), (2, 2), (3, 5).
    """
    
    @staticmethod
    def lagrange_basis(x_points, i, x):
        """
        Compute the i-th Lagrange basis polynomial L_i(x).
        
        L_i(x) = ∏_{j≠i} (x - x_j) / (x_i - x_j)
        """
        # YOUR CODE HERE
        n = len(x_points)
        L_i = 1.0
        
        for j in range(n):
            if j != i:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        
        return L_i
    
    @staticmethod
    def lagrange_interpolation(x_points, y_points, x):
        """
        Evaluate Lagrange interpolating polynomial at x.
        
        P(x) = Σ_i y_i * L_i(x)
        """
        # YOUR CODE HERE
        n = len(x_points)
        result = 0.0
        
        for i in range(n):
            result += y_points[i] * Exercise1.lagrange_basis(x_points, i, x)
        
        return result
    
    @staticmethod
    def lagrange_polynomial(x_points, y_points):
        """Return a callable polynomial function."""
        # YOUR CODE HERE
        def P(x):
            if np.isscalar(x):
                return Exercise1.lagrange_interpolation(x_points, y_points, x)
            else:
                return np.array([Exercise1.lagrange_interpolation(x_points, y_points, xi) 
                                for xi in x])
        return P
    
    @staticmethod
    def verify():
        """Test the implementation."""
        x_points = np.array([0, 1, 2, 3])
        y_points = np.array([1, 3, 2, 5])
        
        P = Exercise1.lagrange_polynomial(x_points, y_points)
        
        print("Exercise 1: Lagrange Interpolation")
        print("-" * 40)
        
        # Check interpolation at data points
        for x, y in zip(x_points, y_points):
            interp = P(x)
            check = "✓" if np.isclose(interp, y) else "✗"
            print(f"P({x}) = {interp:.4f}, expected {y} {check}")
        
        # Evaluate at intermediate point
        print(f"\nP(1.5) = {P(1.5):.4f}")
        
        return P


class Exercise2:
    """
    Exercise 2: Newton's Divided Differences
    ========================================
    
    Implement Newton interpolation:
    1. divided_differences(x, y) - compute divided difference table
    2. newton_interpolation(x_points, coeffs, x) - evaluate
    3. add_point(x_points, y_points, table, x_new, y_new) - add new point efficiently
    """
    
    @staticmethod
    def divided_differences(x, y):
        """
        Compute the divided difference table.
        
        Returns n x n matrix where first row contains coefficients.
        """
        # YOUR CODE HERE
        n = len(x)
        table = np.zeros((n, n))
        table[:, 0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x[i+j] - x[i])
        
        return table
    
    @staticmethod
    def newton_interpolation(x_points, coeffs, x):
        """
        Evaluate Newton form of interpolating polynomial.
        
        P(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...
        """
        # YOUR CODE HERE
        n = len(coeffs)
        result = coeffs[0]
        product = 1.0
        
        for i in range(1, n):
            product *= (x - x_points[i-1])
            result += coeffs[i] * product
        
        return result
    
    @staticmethod
    def add_point(x_points, y_points, table, x_new, y_new):
        """
        Efficiently add a new point to existing interpolation.
        
        Returns updated x_points, y_points, and table.
        """
        # YOUR CODE HERE
        n = len(x_points)
        x_extended = np.append(x_points, x_new)
        y_extended = np.append(y_points, y_new)
        
        # Extend table
        new_table = np.zeros((n + 1, n + 1))
        new_table[:n, :n] = table
        new_table[n, 0] = y_new
        
        # Compute new divided differences
        for j in range(1, n + 1):
            i = n - j
            new_table[i, j] = (new_table[i+1, j-1] - new_table[i, j-1]) / (x_extended[i+j] - x_extended[i])
        
        return x_extended, y_extended, new_table
    
    @staticmethod
    def verify():
        """Test the implementation."""
        x = np.array([0, 1, 2, 3], dtype=float)
        y = np.array([1, 2, 4, 8], dtype=float)  # Exponential-like
        
        print("\nExercise 2: Newton's Divided Differences")
        print("-" * 40)
        
        table = Exercise2.divided_differences(x, y)
        coeffs = table[0, :]
        
        print(f"Coefficients: {coeffs.round(4)}")
        
        # Verify
        for xi, yi in zip(x, y):
            interp = Exercise2.newton_interpolation(x, coeffs, xi)
            check = "✓" if np.isclose(interp, yi) else "✗"
            print(f"P({xi}) = {interp:.4f}, expected {yi} {check}")
        
        # Add a new point
        x_new, y_new, new_table = Exercise2.add_point(x, y, table, 4, 16)
        print(f"\nAdded point (4, 16)")
        print(f"New coefficients: {new_table[0, :].round(4)}")


class Exercise3:
    """
    Exercise 3: Cubic Spline from Scratch
    =====================================
    
    Implement natural cubic spline interpolation:
    1. Solve the tridiagonal system for second derivatives
    2. Compute spline coefficients for each interval
    3. Evaluate the spline at arbitrary points
    """
    
    @staticmethod
    def solve_tridiagonal(a, b, c, d):
        """
        Solve tridiagonal system Ax = d using Thomas algorithm.
        a: sub-diagonal, b: main diagonal, c: super-diagonal
        """
        # YOUR CODE HERE
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        
        # Forward sweep
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom if i < n - 1 else 0
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
        # Back substitution
        x = np.zeros(n)
        x[-1] = d_prime[-1]
        
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    @staticmethod
    def natural_cubic_spline(x, y):
        """
        Compute natural cubic spline coefficients.
        
        Returns dict with coefficients a, b, c, d for each interval.
        S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)² + d_i(x-x_i)³
        """
        # YOUR CODE HERE
        n = len(x) - 1  # Number of intervals
        h = np.diff(x)
        
        # Set up tridiagonal system for second derivatives
        # Natural spline: M_0 = M_n = 0
        sub = np.zeros(n + 1)
        main = np.zeros(n + 1)
        sup = np.zeros(n + 1)
        rhs = np.zeros(n + 1)
        
        main[0] = 1
        main[-1] = 1
        
        for i in range(1, n):
            sub[i] = h[i-1]
            main[i] = 2 * (h[i-1] + h[i])
            sup[i] = h[i]
            rhs[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
        # Solve for M (second derivatives at knots)
        M = Exercise3.solve_tridiagonal(sub, main, sup, rhs)
        
        # Compute spline coefficients
        a = y[:-1].copy()
        c = M[:-1] / 2
        d = (M[1:] - M[:-1]) / (6 * h)
        b = (y[1:] - y[:-1]) / h - h * (M[1:] + 2 * M[:-1]) / 6
        
        return {'a': a, 'b': b, 'c': c, 'd': d, 'x': x}
    
    @staticmethod
    def evaluate_spline(coeffs, x_eval):
        """Evaluate cubic spline at points x_eval."""
        # YOUR CODE HERE
        x = coeffs['x']
        a, b, c, d = coeffs['a'], coeffs['b'], coeffs['c'], coeffs['d']
        
        if np.isscalar(x_eval):
            x_eval = np.array([x_eval])
            scalar_input = True
        else:
            x_eval = np.asarray(x_eval)
            scalar_input = False
        
        result = np.zeros_like(x_eval, dtype=float)
        
        for j, xe in enumerate(x_eval):
            # Find interval
            i = np.searchsorted(x, xe, side='right') - 1
            i = max(0, min(i, len(a) - 1))
            
            dx = xe - x[i]
            result[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
        
        return result[0] if scalar_input else result
    
    @staticmethod
    def verify():
        """Test the implementation."""
        x = np.array([0, 1, 2, 3, 4], dtype=float)
        y = np.array([0, 0.5, 2, 1.5, 0], dtype=float)
        
        print("\nExercise 3: Natural Cubic Spline")
        print("-" * 40)
        
        coeffs = Exercise3.natural_cubic_spline(x, y)
        
        # Compare with scipy
        cs = interpolate.CubicSpline(x, y, bc_type='natural')
        
        test_points = [0.5, 1.5, 2.5, 3.5]
        print(f"{'x':<8} {'Our Spline':<15} {'SciPy':<15} {'Match'}")
        print("-" * 50)
        
        for xp in test_points:
            our = Exercise3.evaluate_spline(coeffs, xp)
            scipy_val = cs(xp)
            match = "✓" if np.isclose(our, scipy_val, rtol=1e-5) else "✗"
            print(f"{xp:<8.1f} {our:<15.6f} {scipy_val:<15.6f} {match}")


class Exercise4:
    """
    Exercise 4: RBF Interpolation Network
    =====================================
    
    Implement RBF interpolation as a network layer:
    1. Various RBF kernels (Gaussian, multiquadric, inverse multiquadric)
    2. Trainable width parameters
    3. Condition number analysis
    """
    
    @staticmethod
    def gaussian_rbf(r, epsilon=1.0):
        """Gaussian RBF: φ(r) = exp(-(εr)²)"""
        return np.exp(-(epsilon * r)**2)
    
    @staticmethod
    def multiquadric_rbf(r, epsilon=1.0):
        """Multiquadric RBF: φ(r) = √(1 + (εr)²)"""
        return np.sqrt(1 + (epsilon * r)**2)
    
    @staticmethod
    def inverse_multiquadric_rbf(r, epsilon=1.0):
        """Inverse multiquadric RBF: φ(r) = 1/√(1 + (εr)²)"""
        return 1 / np.sqrt(1 + (epsilon * r)**2)
    
    @staticmethod
    def build_interpolation_matrix(x_points, rbf_func, epsilon=1.0):
        """Build the RBF interpolation matrix."""
        # YOUR CODE HERE
        n = len(x_points)
        Phi = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                r = np.abs(x_points[i] - x_points[j])
                Phi[i, j] = rbf_func(r, epsilon)
        
        return Phi
    
    @staticmethod
    def rbf_interpolate(x_train, y_train, x_eval, rbf_func, epsilon=1.0):
        """
        Perform RBF interpolation.
        
        Returns interpolated values and condition number.
        """
        # YOUR CODE HERE
        n = len(x_train)
        
        # Build interpolation matrix
        Phi = Exercise4.build_interpolation_matrix(x_train, rbf_func, epsilon)
        cond_num = np.linalg.cond(Phi)
        
        # Solve for weights
        weights = np.linalg.solve(Phi, y_train)
        
        # Evaluate at new points
        y_eval = np.zeros(len(x_eval))
        for i, x in enumerate(x_eval):
            for j in range(n):
                r = np.abs(x - x_train[j])
                y_eval[i] += weights[j] * rbf_func(r, epsilon)
        
        return y_eval, cond_num
    
    @staticmethod
    def find_optimal_epsilon(x_train, y_train, rbf_func, epsilon_range):
        """Find optimal epsilon using leave-one-out cross-validation."""
        # YOUR CODE HERE
        n = len(x_train)
        best_epsilon = epsilon_range[0]
        best_error = float('inf')
        
        for epsilon in epsilon_range:
            loocv_error = 0
            
            for i in range(n):
                # Leave one out
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                x_loo = x_train[mask]
                y_loo = y_train[mask]
                
                # Interpolate
                y_pred, _ = Exercise4.rbf_interpolate(x_loo, y_loo, [x_train[i]], rbf_func, epsilon)
                loocv_error += (y_pred[0] - y_train[i])**2
            
            loocv_error /= n
            
            if loocv_error < best_error:
                best_error = loocv_error
                best_epsilon = epsilon
        
        return best_epsilon, best_error
    
    @staticmethod
    def verify():
        """Test the implementation."""
        np.random.seed(42)
        x_train = np.array([0, 1, 2, 3, 4], dtype=float)
        y_train = np.sin(x_train)
        
        print("\nExercise 4: RBF Interpolation")
        print("-" * 40)
        
        x_eval = np.linspace(0, 4, 9)
        y_true = np.sin(x_eval)
        
        print(f"{'RBF Type':<20} {'ε':<8} {'Max Error':<12} {'Cond #':<12}")
        print("-" * 55)
        
        for name, rbf in [('Gaussian', Exercise4.gaussian_rbf),
                          ('Multiquadric', Exercise4.multiquadric_rbf),
                          ('Inv. Multiquadric', Exercise4.inverse_multiquadric_rbf)]:
            y_interp, cond = Exercise4.rbf_interpolate(x_train, y_train, x_eval, rbf, epsilon=1.0)
            max_err = np.max(np.abs(y_interp - y_true))
            print(f"{name:<20} {1.0:<8.2f} {max_err:<12.6f} {cond:<12.2e}")


class Exercise5:
    """
    Exercise 5: Chebyshev Nodes and Interpolation
    =============================================
    
    Demonstrate optimal node placement:
    1. Generate Chebyshev nodes
    2. Compare interpolation error with equally spaced nodes
    3. Lebesgue constant computation
    """
    
    @staticmethod
    def chebyshev_nodes(n, a=-1, b=1):
        """Generate n Chebyshev nodes on [a, b]."""
        # YOUR CODE HERE
        k = np.arange(n)
        # Chebyshev nodes of the first kind
        x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
        # Scale to [a, b]
        x_scaled = 0.5 * (a + b) + 0.5 * (b - a) * x_cheb
        return x_scaled
    
    @staticmethod
    def lebesgue_constant(x_nodes, n_samples=1000):
        """
        Compute Lebesgue constant for given nodes.
        
        Λ_n = max_x Σ |L_i(x)|
        """
        # YOUR CODE HERE
        a, b = np.min(x_nodes), np.max(x_nodes)
        x_fine = np.linspace(a, b, n_samples)
        n = len(x_nodes)
        
        max_sum = 0
        for x in x_fine:
            lambda_sum = 0
            for i in range(n):
                L_i = 1.0
                for j in range(n):
                    if j != i:
                        L_i *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                lambda_sum += np.abs(L_i)
            max_sum = max(max_sum, lambda_sum)
        
        return max_sum
    
    @staticmethod
    def compare_interpolation(f, n, a=-1, b=1):
        """
        Compare interpolation with Chebyshev vs equally spaced nodes.
        """
        # YOUR CODE HERE
        x_fine = np.linspace(a, b, 500)
        y_true = f(x_fine)
        
        # Chebyshev nodes
        x_cheb = Exercise5.chebyshev_nodes(n, a, b)
        y_cheb = f(x_cheb)
        
        # Equally spaced nodes
        x_equal = np.linspace(a, b, n)
        y_equal = f(x_equal)
        
        # Lagrange interpolation
        def lagrange_interp(x_nodes, y_nodes, x_eval):
            n = len(x_nodes)
            result = np.zeros_like(x_eval)
            for i in range(n):
                L_i = np.ones_like(x_eval)
                for j in range(n):
                    if j != i:
                        L_i *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
                result += y_nodes[i] * L_i
            return result
        
        y_interp_cheb = lagrange_interp(x_cheb, y_cheb, x_fine)
        y_interp_equal = lagrange_interp(x_equal, y_equal, x_fine)
        
        error_cheb = np.max(np.abs(y_true - y_interp_cheb))
        error_equal = np.max(np.abs(y_true - y_interp_equal))
        
        return {
            'chebyshev_error': error_cheb,
            'equally_spaced_error': error_equal,
            'improvement_factor': error_equal / error_cheb if error_cheb > 0 else np.inf
        }
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 5: Chebyshev Nodes")
        print("-" * 40)
        
        # Runge function
        runge = lambda x: 1 / (1 + 25 * x**2)
        
        print("Interpolating Runge function f(x) = 1/(1+25x²)")
        print(f"\n{'n':<5} {'Chebyshev Err':<15} {'Equal Space Err':<18} {'Improvement':<12}")
        print("-" * 55)
        
        for n in [5, 9, 15, 21]:
            result = Exercise5.compare_interpolation(runge, n)
            print(f"{n:<5} {result['chebyshev_error']:<15.6f} {result['equally_spaced_error']:<18.4f} {result['improvement_factor']:<12.1f}x")
        
        print("\nLebesgue constants:")
        for n in [5, 10, 15]:
            x_cheb = Exercise5.chebyshev_nodes(n)
            x_equal = np.linspace(-1, 1, n)
            leb_cheb = Exercise5.lebesgue_constant(x_cheb)
            leb_equal = Exercise5.lebesgue_constant(x_equal)
            print(f"  n={n}: Chebyshev={leb_cheb:.2f}, Equal Spaced={leb_equal:.2f}")


class Exercise6:
    """
    Exercise 6: B-Spline Basis Functions
    ====================================
    
    Implement B-spline basis computation:
    1. De Boor's algorithm for basis functions
    2. Derivative computation
    3. Knot insertion
    """
    
    @staticmethod
    def bspline_basis(t, i, k, x):
        """
        Compute B-spline basis function B_{i,k}(x) using Cox-de Boor recursion.
        
        t: knot vector
        i: basis function index
        k: degree
        x: evaluation point
        """
        # YOUR CODE HERE
        # Base case: k=0 (constant)
        if k == 0:
            return 1.0 if t[i] <= x < t[i+1] else 0.0
        
        # Recursive case
        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]
        
        term1 = 0.0
        if denom1 > 0:
            term1 = (x - t[i]) / denom1 * Exercise6.bspline_basis(t, i, k-1, x)
        
        term2 = 0.0
        if denom2 > 0:
            term2 = (t[i+k+1] - x) / denom2 * Exercise6.bspline_basis(t, i+1, k-1, x)
        
        return term1 + term2
    
    @staticmethod
    def bspline_basis_derivative(t, i, k, x):
        """Compute derivative of B-spline basis."""
        # YOUR CODE HERE
        if k == 0:
            return 0.0
        
        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]
        
        term1 = 0.0
        if denom1 > 0:
            term1 = k / denom1 * Exercise6.bspline_basis(t, i, k-1, x)
        
        term2 = 0.0
        if denom2 > 0:
            term2 = -k / denom2 * Exercise6.bspline_basis(t, i+1, k-1, x)
        
        return term1 + term2
    
    @staticmethod
    def evaluate_bspline(t, c, k, x):
        """Evaluate B-spline curve at x."""
        # YOUR CODE HERE
        n = len(c)  # Number of control points
        result = 0.0
        
        for i in range(n):
            result += c[i] * Exercise6.bspline_basis(t, i, k, x)
        
        return result
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 6: B-Spline Basis Functions")
        print("-" * 40)
        
        # Uniform cubic B-spline
        k = 3  # Cubic
        t = np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3])  # Clamped
        c = np.array([0, 1, 0, 1, 0, 1])  # Control points
        
        print(f"Cubic B-spline with {len(c)} control points")
        print(f"Knots: {t}")
        
        # Evaluate basis functions
        print(f"\nBasis function values at x=1.5:")
        for i in range(len(c)):
            val = Exercise6.bspline_basis(t, i, k, 1.5)
            print(f"  B_{i},{k}(1.5) = {val:.4f}")
        
        # Sum should be 1 (partition of unity)
        total = sum(Exercise6.bspline_basis(t, i, k, 1.5) for i in range(len(c)))
        print(f"  Sum = {total:.6f} (should be 1)")
        
        # Compare with scipy
        print(f"\nSpline values:")
        scipy_spline = interpolate.BSpline(t, c, k)
        for x in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
            our_val = Exercise6.evaluate_bspline(t, c, k, x) if x < 3 else c[-1]
            scipy_val = scipy_spline(x)
            match = "✓" if np.isclose(our_val, scipy_val, atol=1e-4) else "✗"
            print(f"  S({x}) = {our_val:.4f}, scipy = {scipy_val:.4f} {match}")


class Exercise7:
    """
    Exercise 7: Fourier Interpolation
    =================================
    
    Implement trigonometric interpolation:
    1. DFT-based interpolation
    2. Interpolation error for smooth vs non-smooth functions
    3. Gibbs phenomenon demonstration
    """
    
    @staticmethod
    def fourier_interpolation(y_samples, n_eval):
        """
        Interpolate using DFT.
        
        Given n equally spaced samples, interpolate to n_eval points.
        """
        # YOUR CODE HERE
        n = len(y_samples)
        coeffs = np.fft.fft(y_samples)
        
        # Create interpolated coefficients
        n_half = n // 2
        new_coeffs = np.zeros(n_eval, dtype=complex)
        
        # Copy frequencies
        new_coeffs[:n_half] = coeffs[:n_half]
        new_coeffs[-n_half:] = coeffs[-n_half:]
        
        # Scale by ratio
        new_coeffs *= n_eval / n
        
        # Inverse FFT
        y_interp = np.fft.ifft(new_coeffs).real
        
        return y_interp
    
    @staticmethod
    def truncated_fourier_series(f, n_terms, x_eval, period=2*np.pi):
        """
        Compute truncated Fourier series approximation.
        """
        # YOUR CODE HERE
        n_samples = 1000
        x_samples = np.linspace(0, period, n_samples, endpoint=False)
        y_samples = f(x_samples)
        
        # Compute coefficients
        a0 = np.mean(y_samples)
        a = np.zeros(n_terms)
        b = np.zeros(n_terms)
        
        for k in range(1, n_terms + 1):
            a[k-1] = 2 * np.mean(y_samples * np.cos(2 * np.pi * k * x_samples / period))
            b[k-1] = 2 * np.mean(y_samples * np.sin(2 * np.pi * k * x_samples / period))
        
        # Evaluate series
        result = np.ones_like(x_eval) * a0 / 2
        for k in range(1, n_terms + 1):
            result += a[k-1] * np.cos(2 * np.pi * k * x_eval / period)
            result += b[k-1] * np.sin(2 * np.pi * k * x_eval / period)
        
        return result
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 7: Fourier Interpolation")
        print("-" * 40)
        
        # Smooth function
        f_smooth = lambda x: np.sin(x) + 0.5 * np.cos(3*x)
        
        # Step function (discontinuous)
        f_step = lambda x: np.where(x < np.pi, 1.0, -1.0)
        
        x_eval = np.linspace(0, 2*np.pi, 100, endpoint=False)
        
        print("Fourier series approximation error:")
        print(f"{'Function':<15} {'Terms':<8} {'Max Error':<12}")
        print("-" * 35)
        
        for name, f in [('Smooth', f_smooth), ('Step', f_step)]:
            y_true = f(x_eval)
            for n_terms in [3, 10, 30]:
                y_approx = Exercise7.truncated_fourier_series(f, n_terms, x_eval)
                max_error = np.max(np.abs(y_true - y_approx))
                print(f"{name:<15} {n_terms:<8} {max_error:<12.6f}")
        
        print("\nNote: Step function shows Gibbs phenomenon - error doesn't go to 0")


class Exercise8:
    """
    Exercise 8: Least Squares Polynomial with Regularization
    ========================================================
    
    Implement regularized polynomial fitting:
    1. Standard least squares
    2. Ridge regression (L2 regularization)
    3. Cross-validation for regularization parameter
    """
    
    @staticmethod
    def polynomial_least_squares(x, y, degree, regularization=0.0):
        """
        Fit polynomial using least squares with optional regularization.
        
        Returns polynomial coefficients (highest degree first).
        """
        # YOUR CODE HERE
        # Vandermonde matrix
        V = np.vander(x, degree + 1)
        
        if regularization > 0:
            # Ridge regression
            coeffs = np.linalg.solve(
                V.T @ V + regularization * np.eye(degree + 1),
                V.T @ y
            )
        else:
            # Standard least squares
            coeffs, _, _, _ = np.linalg.lstsq(V, y, rcond=None)
        
        return coeffs
    
    @staticmethod
    def evaluate_polynomial(coeffs, x):
        """Evaluate polynomial at x."""
        return np.polyval(coeffs, x)
    
    @staticmethod
    def cross_validate_regularization(x, y, degree, reg_values, n_folds=5):
        """
        Find best regularization parameter using k-fold cross-validation.
        """
        # YOUR CODE HERE
        n = len(x)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        
        best_reg = reg_values[0]
        best_error = float('inf')
        
        for reg in reg_values:
            total_error = 0
            
            for fold in range(n_folds):
                # Split data
                val_start = fold * fold_size
                val_end = val_start + fold_size
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                x_train, y_train = x[train_idx], y[train_idx]
                x_val, y_val = x[val_idx], y[val_idx]
                
                # Fit and evaluate
                coeffs = Exercise8.polynomial_least_squares(x_train, y_train, degree, reg)
                y_pred = Exercise8.evaluate_polynomial(coeffs, x_val)
                total_error += np.mean((y_pred - y_val)**2)
            
            avg_error = total_error / n_folds
            if avg_error < best_error:
                best_error = avg_error
                best_reg = reg
        
        return best_reg, best_error
    
    @staticmethod
    def verify():
        """Test the implementation."""
        np.random.seed(42)
        
        print("\nExercise 8: Regularized Polynomial Fitting")
        print("-" * 40)
        
        # Generate noisy data
        n = 30
        x = np.linspace(0, 1, n)
        y_true = np.sin(4 * np.pi * x)
        y_noisy = y_true + 0.3 * np.random.randn(n)
        
        print(f"Data: {n} noisy samples from sin(4πx)")
        
        # Fit with different regularizations
        degree = 15
        x_test = np.linspace(0, 1, 100)
        y_test = np.sin(4 * np.pi * x_test)
        
        print(f"\nDegree {degree} polynomial fitting:")
        print(f"{'Regularization':<18} {'Train MSE':<12} {'Test MSE':<12}")
        print("-" * 45)
        
        for reg in [0, 1e-6, 1e-4, 1e-2, 1]:
            coeffs = Exercise8.polynomial_least_squares(x, y_noisy, degree, reg)
            train_mse = np.mean((Exercise8.evaluate_polynomial(coeffs, x) - y_noisy)**2)
            test_mse = np.mean((Exercise8.evaluate_polynomial(coeffs, x_test) - y_test)**2)
            print(f"{reg:<18.0e} {train_mse:<12.6f} {test_mse:<12.6f}")
        
        # Cross-validation
        reg_values = np.logspace(-8, 0, 20)
        best_reg, cv_error = Exercise8.cross_validate_regularization(
            x, y_noisy, degree, reg_values, n_folds=5
        )
        print(f"\nBest regularization from CV: {best_reg:.2e}")


class Exercise9:
    """
    Exercise 9: Multivariate Interpolation
    ======================================
    
    Implement 2D interpolation methods:
    1. Bilinear interpolation
    2. Bicubic interpolation
    3. Thin-plate spline interpolation
    """
    
    @staticmethod
    def bilinear_interpolation(x_grid, y_grid, Z, x_eval, y_eval):
        """
        Bilinear interpolation on regular grid.
        """
        # YOUR CODE HERE
        # Find enclosing cell
        i = np.searchsorted(x_grid, x_eval) - 1
        j = np.searchsorted(y_grid, y_eval) - 1
        
        i = max(0, min(i, len(x_grid) - 2))
        j = max(0, min(j, len(y_grid) - 2))
        
        x1, x2 = x_grid[i], x_grid[i+1]
        y1, y2 = y_grid[j], y_grid[j+1]
        
        # Interpolation weights
        wx = (x_eval - x1) / (x2 - x1)
        wy = (y_eval - y1) / (y2 - y1)
        
        # Bilinear formula
        z = (1 - wx) * (1 - wy) * Z[j, i] + \
            wx * (1 - wy) * Z[j, i+1] + \
            (1 - wx) * wy * Z[j+1, i] + \
            wx * wy * Z[j+1, i+1]
        
        return z
    
    @staticmethod
    def thin_plate_spline_2d(x_train, y_train, z_train, x_eval, y_eval):
        """
        Thin-plate spline interpolation for scattered 2D data.
        """
        # YOUR CODE HERE
        def tps_kernel(r):
            """TPS kernel: r² log(r)"""
            return np.where(r > 0, r**2 * np.log(r), 0)
        
        n = len(x_train)
        
        # Build system matrix [K P; P^T 0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                r = np.sqrt((x_train[i] - x_train[j])**2 + (y_train[i] - y_train[j])**2)
                K[i, j] = tps_kernel(r)
        
        P = np.column_stack([np.ones(n), x_train, y_train])
        
        # Full system
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        
        # Right-hand side
        b = np.zeros(n + 3)
        b[:n] = z_train
        
        # Solve
        params = np.linalg.solve(A, b)
        w = params[:n]
        a = params[n:]
        
        # Evaluate
        result = a[0] + a[1] * x_eval + a[2] * y_eval
        for i in range(n):
            r = np.sqrt((x_eval - x_train[i])**2 + (y_eval - y_train[i])**2)
            result += w[i] * tps_kernel(r)
        
        return result
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 9: 2D Interpolation")
        print("-" * 40)
        
        # Test function
        f = lambda x, y: np.sin(x) * np.cos(y)
        
        # Grid data
        x_grid = np.linspace(0, 2*np.pi, 10)
        y_grid = np.linspace(0, 2*np.pi, 10)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = f(X, Y)
        
        # Test points
        test_points = [(1.5, 1.5), (3.0, 2.0), (5.0, 4.5)]
        
        print("Bilinear interpolation:")
        print(f"{'Point':<15} {'True':<12} {'Bilinear':<12} {'Error'}")
        print("-" * 50)
        
        for x, y in test_points:
            true_val = f(x, y)
            interp_val = Exercise9.bilinear_interpolation(x_grid, y_grid, Z, x, y)
            error = abs(true_val - interp_val)
            print(f"({x:.1f}, {y:.1f}){'':>5} {true_val:<12.6f} {interp_val:<12.6f} {error:.6f}")
        
        # TPS test with scattered data
        np.random.seed(42)
        n_scatter = 30
        x_scatter = np.random.rand(n_scatter) * 2 * np.pi
        y_scatter = np.random.rand(n_scatter) * 2 * np.pi
        z_scatter = f(x_scatter, y_scatter)
        
        print("\nThin-plate spline (scattered data):")
        for x, y in test_points[:2]:  # Fewer points for speed
            true_val = f(x, y)
            interp_val = Exercise9.thin_plate_spline_2d(x_scatter, y_scatter, z_scatter, x, y)
            error = abs(true_val - interp_val)
            print(f"({x:.1f}, {y:.1f}){'':>5} {true_val:<12.6f} {interp_val:<12.6f} {error:.6f}")


class Exercise10:
    """
    Exercise 10: Positional Encoding Design
    =======================================
    
    Explore positional encodings for Transformers:
    1. Sinusoidal encoding (original Transformer)
    2. Learned encoding
    3. Relative positional encoding
    """
    
    @staticmethod
    def sinusoidal_encoding(max_len, d_model):
        """
        Create sinusoidal positional encoding.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        # YOUR CODE HERE
        PE = np.zeros((max_len, d_model))
        positions = np.arange(max_len)[:, np.newaxis]
        
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        PE[:, 0::2] = np.sin(positions * div_term)
        PE[:, 1::2] = np.cos(positions * div_term)
        
        return PE
    
    @staticmethod
    def relative_positional_encoding(max_len, d_model):
        """
        Compute relative positional encodings.
        
        Returns matrix where entry [i,j] encodes position i-j.
        """
        # YOUR CODE HERE
        PE_abs = Exercise10.sinusoidal_encoding(2 * max_len - 1, d_model)
        
        # Relative positions from -max_len+1 to max_len-1
        RPE = np.zeros((max_len, max_len, d_model))
        
        for i in range(max_len):
            for j in range(max_len):
                rel_pos = i - j + max_len - 1  # Shift to non-negative index
                RPE[i, j] = PE_abs[rel_pos]
        
        return RPE
    
    @staticmethod
    def rotary_positional_encoding(positions, d_model, base=10000):
        """
        Rotary positional encoding (RoPE).
        
        Used in LLaMA and other modern models.
        """
        # YOUR CODE HERE
        half_d = d_model // 2
        
        # Frequency for each dimension pair
        theta = base ** (-np.arange(half_d) / half_d)
        
        # Position-dependent angles
        angles = np.outer(positions, theta)
        
        # Return sin and cos separately (to apply rotation)
        return np.sin(angles), np.cos(angles)
    
    @staticmethod
    def verify():
        """Test the implementation."""
        print("\nExercise 10: Positional Encodings")
        print("-" * 40)
        
        max_len = 10
        d_model = 8
        
        # Sinusoidal
        PE = Exercise10.sinusoidal_encoding(max_len, d_model)
        
        print(f"Sinusoidal encoding ({max_len} positions, d={d_model})")
        print(f"\nPosition similarity (dot products):")
        
        similarity = PE @ PE.T
        print("   ", end="")
        for j in range(5):
            print(f"{j:>6}", end="")
        print()
        
        for i in range(5):
            print(f"{i}: ", end="")
            for j in range(5):
                print(f"{similarity[i,j]:>6.2f}", end="")
            print()
        
        print("\nNote: Nearby positions have higher similarity")
        
        # Relative encoding
        RPE = Exercise10.relative_positional_encoding(5, d_model)
        print(f"\nRelative encoding: RPE[2,4] = position (2-4=-2)")
        print(f"  Same as RPE[0,2] (also -2): {np.allclose(RPE[2,4], RPE[0,2])}")
        
        # RoPE
        positions = np.arange(5)
        sin_vals, cos_vals = Exercise10.rotary_positional_encoding(positions, d_model)
        print(f"\nRoPE sin values shape: {sin_vals.shape}")
        print(f"Position 0 sin: {sin_vals[0].round(3)}")
        print(f"Position 1 sin: {sin_vals[1].round(3)}")


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
