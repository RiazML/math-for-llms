"""
Interpolation and Approximation - Examples
==========================================
Practical implementations of interpolation methods.
"""

import numpy as np
from scipy import interpolate
from scipy.fft import fft, ifft


def example_1_lagrange_interpolation():
    """
    Example 1: Lagrange Interpolation
    =================================
    Classic polynomial interpolation.
    """
    print("=" * 60)
    print("Example 1: Lagrange Interpolation")
    print("=" * 60)
    
    def lagrange_basis(x_points, i, x):
        """Compute i-th Lagrange basis polynomial at x."""
        n = len(x_points)
        L_i = 1.0
        for j in range(n):
            if j != i:
                L_i *= (x - x_points[j]) / (x_points[i] - x_points[j])
        return L_i
    
    def lagrange_interpolation(x_points, y_points, x):
        """Lagrange interpolation at point x."""
        n = len(x_points)
        result = 0.0
        for i in range(n):
            result += y_points[i] * lagrange_basis(x_points, i, x)
        return result
    
    # Interpolation points
    x_points = np.array([0, 1, 2, 3])
    y_points = np.array([1, 2, 0, 5])
    
    print(f"Data points: {list(zip(x_points, y_points))}")
    
    # Evaluate at new points
    x_eval = np.linspace(-0.5, 3.5, 9)
    y_eval = [lagrange_interpolation(x_points, y_points, x) for x in x_eval]
    
    print(f"\nInterpolated values:")
    for x, y in zip(x_eval, y_eval):
        print(f"  P({x:.2f}) = {y:.4f}")
    
    # Verify at data points
    print(f"\nVerification at data points:")
    for x, y_true in zip(x_points, y_points):
        y_interp = lagrange_interpolation(x_points, y_points, x)
        print(f"  P({x}) = {y_interp:.6f}, expected = {y_true}")


def example_2_newton_divided_differences():
    """
    Example 2: Newton's Divided Differences
    =======================================
    Efficient polynomial interpolation.
    """
    print("\n" + "=" * 60)
    print("Example 2: Newton's Divided Differences")
    print("=" * 60)
    
    def divided_differences(x, y):
        """Compute divided difference table."""
        n = len(x)
        table = np.zeros((n, n))
        table[:, 0] = y
        
        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i+1, j-1] - table[i, j-1]) / (x[i+j] - x[i])
        
        return table
    
    def newton_interpolation(x_points, table, x):
        """Evaluate Newton interpolating polynomial."""
        n = len(x_points)
        result = table[0, 0]
        product = 1.0
        
        for i in range(1, n):
            product *= (x - x_points[i-1])
            result += table[0, i] * product
        
        return result
    
    # Data points
    x_points = np.array([0, 1, 2, 3, 4], dtype=float)
    y_points = np.array([1, 1, 2, 6, 24], dtype=float)  # Factorial-like
    
    print(f"Data points: {list(zip(x_points, y_points))}")
    
    # Compute divided differences
    table = divided_differences(x_points, y_points)
    
    print(f"\nDivided difference table (first row = coefficients):")
    print(f"  {table[0, :].round(4)}")
    
    # Evaluate
    x_test = 2.5
    y_interp = newton_interpolation(x_points, table, x_test)
    
    print(f"\nP({x_test}) = {y_interp:.4f}")
    
    # Easy to add a new point
    x_new = 5
    y_new = 120  # 5!
    
    print(f"\nAdding point ({x_new}, {y_new}):")
    x_extended = np.append(x_points, x_new)
    y_extended = np.append(y_points, y_new)
    table_ext = divided_differences(x_extended, y_extended)
    
    print(f"  New coefficient: {table_ext[0, 5]:.4f}")


def example_3_runge_phenomenon():
    """
    Example 3: Runge's Phenomenon
    =============================
    Why high-degree polynomials can fail.
    """
    print("\n" + "=" * 60)
    print("Example 3: Runge's Phenomenon")
    print("=" * 60)
    
    # Runge function
    def runge(x):
        return 1 / (1 + 25 * x**2)
    
    def lagrange_interpolation(x_points, y_points, x_eval):
        """Vectorized Lagrange interpolation."""
        n = len(x_points)
        result = np.zeros_like(x_eval)
        
        for i in range(n):
            L_i = np.ones_like(x_eval)
            for j in range(n):
                if j != i:
                    L_i *= (x_eval - x_points[j]) / (x_points[i] - x_points[j])
            result += y_points[i] * L_i
        
        return result
    
    x_fine = np.linspace(-1, 1, 200)
    y_true = runge(x_fine)
    
    print(f"Runge function: f(x) = 1/(1 + 25x²)")
    print(f"\nMax interpolation error with equally spaced nodes:")
    
    for n in [5, 9, 15, 21]:
        # Equally spaced nodes
        x_equal = np.linspace(-1, 1, n)
        y_equal = runge(x_equal)
        
        y_interp = lagrange_interpolation(x_equal, y_equal, x_fine)
        max_error = np.max(np.abs(y_true - y_interp))
        
        print(f"  n = {n:2d}: max error = {max_error:.4f}")
    
    print(f"\nWith Chebyshev nodes:")
    for n in [5, 9, 15, 21]:
        # Chebyshev nodes
        k = np.arange(n)
        x_cheb = np.cos((2*k + 1) * np.pi / (2*n))
        y_cheb = runge(x_cheb)
        
        y_interp = lagrange_interpolation(x_cheb, y_cheb, x_fine)
        max_error = np.max(np.abs(y_true - y_interp))
        
        print(f"  n = {n:2d}: max error = {max_error:.6f}")


def example_4_cubic_spline():
    """
    Example 4: Cubic Spline Interpolation
    =====================================
    Smooth piecewise polynomial.
    """
    print("\n" + "=" * 60)
    print("Example 4: Cubic Spline")
    print("=" * 60)
    
    # Data points
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1.0])
    
    print(f"Data points: {list(zip(x, y))}")
    
    # Natural cubic spline (scipy)
    cs_natural = interpolate.CubicSpline(x, y, bc_type='natural')
    
    # Clamped cubic spline
    cs_clamped = interpolate.CubicSpline(x, y, bc_type='clamped')
    
    # Not-a-knot (default)
    cs_nak = interpolate.CubicSpline(x, y)
    
    # Evaluate
    x_fine = np.linspace(0, 5, 50)
    
    print(f"\nSpline values at x = 2.5:")
    print(f"  Natural:     {cs_natural(2.5):.4f}")
    print(f"  Clamped:     {cs_clamped(2.5):.4f}")
    print(f"  Not-a-knot:  {cs_nak(2.5):.4f}")
    
    # Derivatives
    print(f"\nFirst derivative at x = 2.5:")
    print(f"  Natural:     {cs_natural(2.5, 1):.4f}")
    print(f"  Clamped:     {cs_clamped(2.5, 1):.4f}")
    
    print(f"\nSecond derivative at x = 2.5:")
    print(f"  Natural:     {cs_natural(2.5, 2):.4f}")
    
    # Spline coefficients
    print(f"\nCubic spline coefficients (natural):")
    print(f"  Format: S_i(x) = c[i,0] + c[i,1](x-x_i) + c[i,2](x-x_i)² + c[i,3](x-x_i)³")
    print(f"  Coefficients shape: {cs_natural.c.shape}")


def example_5_b_splines():
    """
    Example 5: B-Splines
    ====================
    Basis for spline functions.
    """
    print("\n" + "=" * 60)
    print("Example 5: B-Splines")
    print("=" * 60)
    
    # Knot vector for cubic B-splines
    t = np.array([0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4])  # Clamped
    k = 3  # Cubic (order 4, degree 3)
    
    # Control points
    c = np.array([0, 1, 0, 1, 0, 1, 0])  # 7 control points
    
    # Create B-spline
    spline = interpolate.BSpline(t, c, k)
    
    print(f"Cubic B-spline:")
    print(f"  Knots: {t}")
    print(f"  Control points: {c}")
    
    # Evaluate
    x_eval = np.linspace(0, 4, 41)
    y_eval = spline(x_eval)
    
    print(f"\nB-spline values:")
    for x in [0, 0.5, 1, 2, 3, 4]:
        print(f"  S({x}) = {spline(x):.4f}")
    
    # Basis functions
    print(f"\nB-spline basis functions at x = 1.5:")
    basis_values = interpolate.BSpline.basis_element(t[3:8])(1.5)
    print(f"  B_3,3(1.5) = {basis_values:.4f}")


def example_6_rbf_interpolation():
    """
    Example 6: Radial Basis Function Interpolation
    ==============================================
    For scattered data in any dimension.
    """
    print("\n" + "=" * 60)
    print("Example 6: RBF Interpolation")
    print("=" * 60)
    
    def gaussian_rbf(r, epsilon=1.0):
        return np.exp(-(epsilon * r)**2)
    
    def multiquadric_rbf(r, epsilon=1.0):
        return np.sqrt(1 + (epsilon * r)**2)
    
    def rbf_interpolation(x_train, y_train, x_eval, rbf_func, epsilon=1.0):
        """RBF interpolation."""
        n_train = len(x_train)
        
        # Build interpolation matrix
        Phi = np.zeros((n_train, n_train))
        for i in range(n_train):
            for j in range(n_train):
                r = np.abs(x_train[i] - x_train[j])
                Phi[i, j] = rbf_func(r, epsilon)
        
        # Solve for coefficients
        c = np.linalg.solve(Phi, y_train)
        
        # Evaluate at new points
        y_eval = np.zeros(len(x_eval))
        for k, x in enumerate(x_eval):
            for i in range(n_train):
                r = np.abs(x - x_train[i])
                y_eval[k] += c[i] * rbf_func(r, epsilon)
        
        return y_eval
    
    # Data points
    x_train = np.array([0, 1, 3, 4, 6])
    y_train = np.array([0, 1, 0.5, 0.8, 0])
    
    print(f"Data points: {list(zip(x_train, y_train))}")
    
    x_eval = np.linspace(0, 6, 7)
    
    # Compare RBF types
    print(f"\nInterpolated values:")
    print(f"{'x':<8} {'Gaussian':<12} {'Multiquadric':<12}")
    print("-" * 32)
    
    y_gauss = rbf_interpolation(x_train, y_train, x_eval, gaussian_rbf, epsilon=1.0)
    y_mq = rbf_interpolation(x_train, y_train, x_eval, multiquadric_rbf, epsilon=1.0)
    
    for x, yg, ym in zip(x_eval, y_gauss, y_mq):
        print(f"{x:<8.1f} {yg:<12.4f} {ym:<12.4f}")


def example_7_least_squares_polynomial():
    """
    Example 7: Polynomial Least Squares
    ===================================
    Fitting polynomials to noisy data.
    """
    print("\n" + "=" * 60)
    print("Example 7: Least Squares Polynomial Fitting")
    print("=" * 60)
    
    # Generate noisy data
    np.random.seed(42)
    n_points = 50
    x = np.linspace(0, 2*np.pi, n_points)
    y_true = np.sin(x)
    y_noisy = y_true + 0.2 * np.random.randn(n_points)
    
    print(f"Fitting polynomial to noisy sin(x) data ({n_points} points)")
    
    # Fit polynomials of different degrees
    print(f"\n{'Degree':<10} {'Training MSE':<15} {'Max Error':<15}")
    print("-" * 40)
    
    for degree in [1, 3, 5, 10, 20]:
        # Polynomial fit
        coeffs = np.polyfit(x, y_noisy, degree)
        p = np.poly1d(coeffs)
        
        y_pred = p(x)
        mse = np.mean((y_pred - y_noisy)**2)
        max_err = np.max(np.abs(y_pred - y_true))
        
        print(f"{degree:<10} {mse:<15.6f} {max_err:<15.4f}")
    
    # Compare with interpolation (exact fit but overfitting)
    print(f"\nNote: Higher degree fits training data better but")
    print(f"may overfit - degree 5 gives best approximation to true sin(x)")


def example_8_fourier_approximation():
    """
    Example 8: Fourier Series Approximation
    =======================================
    Trigonometric polynomials for periodic functions.
    """
    print("\n" + "=" * 60)
    print("Example 8: Fourier Approximation")
    print("=" * 60)
    
    # Square wave (periodic)
    def square_wave(x):
        return np.sign(np.sin(x))
    
    def fourier_coefficients(f, n_terms, n_samples=1000):
        """Compute Fourier coefficients numerically."""
        x = np.linspace(-np.pi, np.pi, n_samples, endpoint=False)
        y = f(x)
        
        a0 = np.mean(y)
        a = np.zeros(n_terms)
        b = np.zeros(n_terms)
        
        for k in range(1, n_terms + 1):
            a[k-1] = 2 * np.mean(y * np.cos(k * x))
            b[k-1] = 2 * np.mean(y * np.sin(k * x))
        
        return a0, a, b
    
    def fourier_series(x, a0, a, b):
        """Evaluate truncated Fourier series."""
        result = a0 / 2
        for k in range(len(a)):
            result += a[k] * np.cos((k+1) * x) + b[k] * np.sin((k+1) * x)
        return result
    
    print("Square wave Fourier approximation:")
    print("(Only odd harmonics contribute)")
    
    x_eval = np.linspace(-np.pi, np.pi, 100)
    y_true = square_wave(x_eval)
    
    print(f"\n{'Terms':<10} {'Max Error':<15}")
    print("-" * 25)
    
    for n_terms in [1, 3, 5, 11, 21]:
        a0, a, b = fourier_coefficients(square_wave, n_terms)
        y_approx = fourier_series(x_eval, a0, a, b)
        max_error = np.max(np.abs(y_true - y_approx))
        
        print(f"{n_terms:<10} {max_error:<15.4f}")
    
    # Show coefficients for square wave
    print(f"\nFourier sine coefficients for square wave (first 5 odd):")
    a0, a, b = fourier_coefficients(square_wave, 10)
    for k in [1, 3, 5, 7, 9]:
        # Theory: b_k = 4/(π*k) for odd k
        theory = 4 / (np.pi * k) if k % 2 == 1 else 0
        print(f"  b_{k} = {b[k-1]:.4f}, theory = {theory:.4f}")


def example_9_2d_interpolation():
    """
    Example 9: 2D Interpolation
    ===========================
    Bilinear and bicubic methods.
    """
    print("\n" + "=" * 60)
    print("Example 9: 2D Interpolation")
    print("=" * 60)
    
    # Create sample data on grid
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3])
    X, Y = np.meshgrid(x, y)
    
    # Sample function
    def f(x, y):
        return np.sin(x) * np.cos(y)
    
    Z = f(X, Y)
    
    print(f"Grid data: {len(x)} x {len(y)} points")
    print(f"Function: sin(x) * cos(y)")
    
    # Create interpolators
    interp_linear = interpolate.RegularGridInterpolator((y, x), Z, method='linear')
    interp_cubic = interpolate.RegularGridInterpolator((y, x), Z, method='cubic')
    
    # Test points
    test_points = np.array([
        [0.5, 0.5],
        [1.5, 1.5],
        [2.5, 2.5]
    ])
    
    print(f"\nInterpolation at test points:")
    print(f"{'Point':<15} {'True':<12} {'Linear':<12} {'Cubic':<12}")
    print("-" * 55)
    
    for pt in test_points:
        true_val = f(pt[1], pt[0])  # Note: x, y order
        linear_val = interp_linear(pt)[0]
        cubic_val = interp_cubic(pt)[0]
        
        print(f"({pt[1]:.1f}, {pt[0]:.1f}){'':>5} {true_val:<12.6f} {linear_val:<12.6f} {cubic_val:<12.6f}")


def example_10_scattered_data():
    """
    Example 10: Scattered Data Interpolation
    ========================================
    RBF for non-grid data.
    """
    print("\n" + "=" * 60)
    print("Example 10: Scattered Data Interpolation")
    print("=" * 60)
    
    # Random scattered points
    np.random.seed(42)
    n_points = 20
    x_scatter = np.random.rand(n_points) * 4
    y_scatter = np.random.rand(n_points) * 4
    z_scatter = np.sin(x_scatter) + np.cos(y_scatter)
    
    print(f"Scattered data: {n_points} random points in [0,4] x [0,4]")
    print(f"Function: sin(x) + cos(y)")
    
    # RBF interpolation
    rbf_interpolator = interpolate.RBFInterpolator(
        np.column_stack([x_scatter, y_scatter]),
        z_scatter,
        kernel='thin_plate_spline'
    )
    
    # Evaluate on grid
    x_grid = np.linspace(0, 4, 5)
    y_grid = np.linspace(0, 4, 5)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    points_eval = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    Z_interp = rbf_interpolator(points_eval).reshape(X_grid.shape)
    Z_true = np.sin(X_grid) + np.cos(Y_grid)
    
    error = np.abs(Z_interp - Z_true)
    
    print(f"\nRBF interpolation error on 5x5 grid:")
    print(f"  Mean error: {np.mean(error):.4f}")
    print(f"  Max error:  {np.max(error):.4f}")
    
    # Compare kernels
    print(f"\nComparison of RBF kernels:")
    print(f"{'Kernel':<20} {'Mean Error':<12} {'Max Error':<12}")
    print("-" * 45)
    
    for kernel in ['thin_plate_spline', 'cubic', 'gaussian', 'multiquadric']:
        try:
            rbf = interpolate.RBFInterpolator(
                np.column_stack([x_scatter, y_scatter]),
                z_scatter,
                kernel=kernel
            )
            Z_interp = rbf(points_eval).reshape(X_grid.shape)
            error = np.abs(Z_interp - Z_true)
            print(f"{kernel:<20} {np.mean(error):<12.4f} {np.max(error):<12.4f}")
        except Exception as e:
            print(f"{kernel:<20} Failed: {str(e)[:30]}")


def example_11_positional_encoding():
    """
    Example 11: Positional Encoding (Transformers)
    ==============================================
    Fourier features for position representation.
    """
    print("\n" + "=" * 60)
    print("Example 11: Positional Encoding")
    print("=" * 60)
    
    def positional_encoding(positions, d_model):
        """
        Compute positional encodings as in "Attention is All You Need".
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        PE = np.zeros((len(positions), d_model))
        
        for i in range(d_model // 2):
            div_term = 10000 ** (2 * i / d_model)
            PE[:, 2*i] = np.sin(positions / div_term)
            PE[:, 2*i+1] = np.cos(positions / div_term)
        
        return PE
    
    # Example
    positions = np.arange(10)
    d_model = 8
    
    PE = positional_encoding(positions, d_model)
    
    print(f"Positional encoding for positions 0-9, d_model=8")
    print(f"\nEncoding matrix shape: {PE.shape}")
    
    print(f"\nFirst 5 positions:")
    print(f"{'Pos':<5}", end="")
    for d in range(d_model):
        print(f"{'PE['+str(d)+']':<10}", end="")
    print()
    print("-" * (5 + 10 * d_model))
    
    for pos in range(5):
        print(f"{pos:<5}", end="")
        for d in range(d_model):
            print(f"{PE[pos, d]:<10.4f}", end="")
        print()
    
    # Show how positions can be distinguished
    print(f"\nDot products between position encodings:")
    for i in range(5):
        for j in range(5):
            dot = PE[i] @ PE[j]
            print(f"{dot:6.2f}", end=" ")
        print()
    
    print(f"\nNote: Nearby positions have higher dot products")


def example_12_neural_network_approximation():
    """
    Example 12: Neural Network as Universal Approximator
    ====================================================
    ReLU networks and piecewise linear approximation.
    """
    print("\n" + "=" * 60)
    print("Example 12: Neural Network Approximation")
    print("=" * 60)
    
    def relu(x):
        return np.maximum(0, x)
    
    def simple_nn(x, W1, b1, W2, b2):
        """Two-layer ReLU network."""
        h = relu(x.reshape(-1, 1) @ W1 + b1)
        return h @ W2 + b2
    
    # Approximate sin(x) with small network
    np.random.seed(42)
    
    # Training data
    x_train = np.linspace(0, 2*np.pi, 100)
    y_train = np.sin(x_train)
    
    # Initialize network (2 hidden units)
    n_hidden = 10
    W1 = np.random.randn(1, n_hidden) * 0.5
    b1 = np.random.randn(n_hidden) * 0.5
    W2 = np.random.randn(n_hidden, 1) * 0.5
    b2 = np.random.randn(1) * 0.1
    
    # Simple gradient descent
    lr = 0.001
    
    for epoch in range(1000):
        # Forward
        h = relu(x_train.reshape(-1, 1) @ W1 + b1)
        y_pred = (h @ W2 + b2).flatten()
        
        # Loss
        loss = np.mean((y_pred - y_train)**2)
        
        # Backward (simplified, not full backprop)
        # Just demonstrating the concept
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: MSE = {loss:.6f}")
    
    print(f"\nNote: With proper training, neural networks can approximate any continuous function")
    print(f"ReLU networks produce piecewise linear approximations")
    print(f"Number of linear pieces ≤ number of hidden units")


def run_all_examples():
    """Run all examples."""
    example_1_lagrange_interpolation()
    example_2_newton_divided_differences()
    example_3_runge_phenomenon()
    example_4_cubic_spline()
    example_5_b_splines()
    example_6_rbf_interpolation()
    example_7_least_squares_polynomial()
    example_8_fourier_approximation()
    example_9_2d_interpolation()
    example_10_scattered_data()
    example_11_positional_encoding()
    example_12_neural_network_approximation()


if __name__ == "__main__":
    run_all_examples()
