"""
Partial Derivatives and Gradients - Examples
============================================
Practical demonstrations of multivariate calculus concepts.
"""

import numpy as np


def example_partial_derivatives():
    """Demonstrate partial derivatives."""
    print("=" * 60)
    print("EXAMPLE 1: Partial Derivatives")
    print("=" * 60)
    
    print("For f(x, y) = x²y + 3xy² - 2y")
    print("\n∂f/∂x: Treat y as constant, differentiate w.r.t. x")
    print("     = 2xy + 3y² + 0 = 2xy + 3y²")
    
    print("\n∂f/∂y: Treat x as constant, differentiate w.r.t. y")
    print("     = x² + 6xy - 2")
    
    # Numerical verification
    def f(x, y):
        return x**2 * y + 3*x*y**2 - 2*y
    
    def df_dx_analytic(x, y):
        return 2*x*y + 3*y**2
    
    def df_dy_analytic(x, y):
        return x**2 + 6*x*y - 2
    
    x, y = 2.0, 3.0
    h = 1e-7
    
    # Numerical partial derivatives
    df_dx_num = (f(x+h, y) - f(x-h, y)) / (2*h)
    df_dy_num = (f(x, y+h) - f(x, y-h)) / (2*h)
    
    print(f"\nAt point ({x}, {y}):")
    print(f"∂f/∂x: analytic = {df_dx_analytic(x, y):.6f}, numerical = {df_dx_num:.6f}")
    print(f"∂f/∂y: analytic = {df_dy_analytic(x, y):.6f}, numerical = {df_dy_num:.6f}")


def example_gradient_computation():
    """Demonstrate gradient computation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Gradient Computation")
    print("=" * 60)
    
    print("For f(x, y) = x² + y²")
    print("\n∂f/∂x = 2x")
    print("∂f/∂y = 2y")
    print("\n∇f = [2x, 2y]ᵀ")
    
    def f(x):
        return x[0]**2 + x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 2*x[1]])
    
    points = [(1, 2), (0, 0), (-1, 1), (3, 4)]
    
    print("\nGradient at various points:")
    for point in points:
        x = np.array(point)
        g = grad_f(x)
        mag = np.linalg.norm(g)
        print(f"  Point {point}: ∇f = {g}, ||∇f|| = {mag:.4f}")
    
    print("\nNote: Gradient points away from origin (minimum)")


def example_directional_derivative():
    """Demonstrate directional derivative."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Directional Derivative")
    print("=" * 60)
    
    print("For f(x, y) = x² - xy + y²")
    print("∇f = [2x - y, -x + 2y]ᵀ")
    
    def grad_f(x, y):
        return np.array([2*x - y, -x + 2*y])
    
    x, y = 1, 2
    g = grad_f(x, y)
    print(f"\nAt point (1, 2): ∇f = {g}")
    
    # Various directions
    directions = [
        (1, 0, "x-direction"),
        (0, 1, "y-direction"),
        (1, 1, "diagonal"),
        (g[0], g[1], "gradient direction"),
        (-g[0], -g[1], "negative gradient"),
    ]
    
    print("\nDirectional derivatives:")
    for dx, dy, name in directions:
        u = np.array([dx, dy])
        u = u / np.linalg.norm(u)  # Normalize
        D_u = np.dot(g, u)
        print(f"  {name:20s}: D_u f = {D_u:.4f}")
    
    print(f"\n||∇f|| = {np.linalg.norm(g):.4f} (maximum directional derivative)")


def example_gradient_descent_2d():
    """Demonstrate gradient descent in 2D."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Gradient Descent (2D)")
    print("=" * 60)
    
    print("Minimize f(x, y) = (x - 1)² + (y - 2)²")
    print("Minimum at (1, 2)")
    
    def f(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    def grad_f(x):
        return np.array([2*(x[0] - 1), 2*(x[1] - 2)])
    
    # Gradient descent
    x = np.array([0.0, 0.0])  # Starting point
    learning_rate = 0.1
    
    print(f"\nStarting point: {x}")
    print(f"Learning rate: {learning_rate}")
    print("\nIteration history:")
    
    for i in range(10):
        loss = f(x)
        grad = grad_f(x)
        print(f"  k={i}: x = [{x[0]:.4f}, {x[1]:.4f}], f(x) = {loss:.6f}, ||∇f|| = {np.linalg.norm(grad):.6f}")
        x = x - learning_rate * grad
    
    print(f"\nFinal: x = [{x[0]:.4f}, {x[1]:.4f}]")
    print("Expected: [1.0000, 2.0000]")


def example_gradient_descent_rosenbrock():
    """Gradient descent on Rosenbrock function."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Gradient Descent (Rosenbrock Function)")
    print("=" * 60)
    
    print("Rosenbrock: f(x, y) = (1 - x)² + 100(y - x²)²")
    print("Minimum at (1, 1)")
    
    def rosenbrock(x):
        return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    
    def grad_rosenbrock(x):
        dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
        dy = 200*(x[1] - x[0]**2)
        return np.array([dx, dy])
    
    x = np.array([-1.0, 1.0])
    learning_rate = 0.001  # Small due to ill-conditioning
    
    print(f"\nStarting: {x}, f(x) = {rosenbrock(x):.4f}")
    
    for i in range(5000):
        grad = grad_rosenbrock(x)
        x = x - learning_rate * grad
    
    print(f"After 5000 iterations: [{x[0]:.4f}, {x[1]:.4f}], f(x) = {rosenbrock(x):.6f}")
    print("\nNote: Rosenbrock is a challenging optimization problem!")


def example_linear_regression_gradient():
    """Gradient for linear regression."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Linear Regression Gradient")
    print("=" * 60)
    
    print("MSE Loss: L(w) = (1/n)||Xw - y||²")
    print("Gradient: ∇L = (2/n)Xᵀ(Xw - y)")
    
    # Generate data
    np.random.seed(42)
    n, d = 100, 2
    X = np.random.randn(n, d)
    w_true = np.array([2.0, -1.0])
    y = X @ w_true + 0.1 * np.random.randn(n)
    
    def mse_loss(w):
        residual = X @ w - y
        return np.mean(residual**2)
    
    def grad_mse(w):
        residual = X @ w - y
        return (2/n) * X.T @ residual
    
    # Gradient descent
    w = np.zeros(d)
    learning_rate = 0.1
    
    print(f"\nTrue weights: {w_true}")
    print(f"Initial weights: {w}")
    
    for i in range(100):
        if i % 20 == 0:
            print(f"  Iter {i}: w = {w.round(4)}, Loss = {mse_loss(w):.6f}")
        grad = grad_mse(w)
        w = w - learning_rate * grad
    
    print(f"\nFinal weights: {w.round(4)}")
    print(f"True weights:  {w_true}")


def example_logistic_regression_gradient():
    """Gradient for logistic regression."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Logistic Regression Gradient")
    print("=" * 60)
    
    print("Cross-entropy loss gradient: ∇L = (1/n)Xᵀ(σ(Xw) - y)")
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    # Generate binary classification data
    np.random.seed(42)
    n, d = 100, 2
    X = np.random.randn(n, d)
    w_true = np.array([2.0, -1.0])
    probs = sigmoid(X @ w_true)
    y = (np.random.rand(n) < probs).astype(float)
    
    def cross_entropy_loss(w):
        z = X @ w
        p = sigmoid(z)
        # Clip for numerical stability
        p = np.clip(p, 1e-10, 1-1e-10)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
    
    def grad_ce(w):
        z = X @ w
        p = sigmoid(z)
        return (1/n) * X.T @ (p - y)
    
    # Gradient descent
    w = np.zeros(d)
    learning_rate = 1.0
    
    print(f"\nInitial weights: {w}")
    
    for i in range(100):
        if i % 20 == 0:
            print(f"  Iter {i}: w = {w.round(4)}, Loss = {cross_entropy_loss(w):.6f}")
        grad = grad_ce(w)
        w = w - learning_rate * grad
    
    print(f"\nFinal weights: {w.round(4)}")
    print(f"(True direction: {w_true})")


def example_gradient_checking():
    """Demonstrate gradient checking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Gradient Checking")
    print("=" * 60)
    
    print("Verify analytical gradient with numerical approximation")
    
    def f(x):
        return x[0]**3 + x[0]*x[1]**2 + np.sin(x[1])
    
    def grad_f_analytic(x):
        return np.array([
            3*x[0]**2 + x[1]**2,
            2*x[0]*x[1] + np.cos(x[1])
        ])
    
    def grad_f_numerical(x, h=1e-7):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
        return grad
    
    x = np.array([1.0, 2.0])
    
    grad_analytic = grad_f_analytic(x)
    grad_numerical = grad_f_numerical(x)
    
    print(f"\nAt point x = {x}")
    print(f"Analytical gradient:  {grad_analytic}")
    print(f"Numerical gradient:   {grad_numerical}")
    
    relative_error = np.linalg.norm(grad_analytic - grad_numerical) / (
        np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numerical) + 1e-10
    )
    print(f"\nRelative error: {relative_error:.2e}")
    print(f"Gradient check {'PASSED' if relative_error < 1e-5 else 'FAILED'}")


def example_higher_order_partials():
    """Demonstrate higher-order partial derivatives."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Higher-Order Partial Derivatives")
    print("=" * 60)
    
    print("For f(x, y) = x³y + xy³")
    print("\nFirst partial derivatives:")
    print("  ∂f/∂x = 3x²y + y³")
    print("  ∂f/∂y = x³ + 3xy²")
    
    print("\nSecond partial derivatives:")
    print("  ∂²f/∂x² = 6xy")
    print("  ∂²f/∂y² = 6xy")
    print("  ∂²f/∂x∂y = 3x² + 3y²")
    print("  ∂²f/∂y∂x = 3x² + 3y²")
    
    print("\nClairaut's Theorem: Mixed partials are equal!")
    print("∂²f/∂x∂y = ∂²f/∂y∂x = 3x² + 3y²")
    
    # Numerical verification
    def f(x, y):
        return x**3 * y + x * y**3
    
    x, y, h = 1.0, 2.0, 1e-5
    
    # Mixed partial ∂²f/∂x∂y
    f_xy = (f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h)) / (4*h*h)
    
    analytic = 3*x**2 + 3*y**2
    print(f"\nAt (1, 2):")
    print(f"  Analytic ∂²f/∂x∂y = {analytic}")
    print(f"  Numerical ≈ {f_xy:.4f}")


def example_neural_network_gradient():
    """Simple neural network gradient computation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Simple Neural Network Gradient")
    print("=" * 60)
    
    print("Single neuron: y = σ(w·x + b)")
    print("Loss: L = (y - t)² where t is target")
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(z):
        s = sigmoid(z)
        return s * (1 - s)
    
    # Forward pass
    x = np.array([1.0, 2.0])  # Input
    w = np.array([0.5, -0.3])  # Weights
    b = 0.1  # Bias
    t = 1.0  # Target
    
    z = np.dot(w, x) + b
    y = sigmoid(z)
    loss = (y - t)**2
    
    print(f"\nInput x = {x}")
    print(f"Weights w = {w}, bias b = {b}")
    print(f"z = w·x + b = {z:.4f}")
    print(f"y = σ(z) = {y:.4f}")
    print(f"Loss = (y - t)² = {loss:.4f}")
    
    # Backward pass (gradients)
    dL_dy = 2 * (y - t)  # ∂L/∂y
    dy_dz = sigmoid_derivative(z)  # ∂y/∂z
    dz_dw = x  # ∂z/∂w
    dz_db = 1  # ∂z/∂b
    
    # Chain rule
    dL_dz = dL_dy * dy_dz
    dL_dw = dL_dz * dz_dw
    dL_db = dL_dz * dz_db
    
    print("\n--- Backpropagation ---")
    print(f"∂L/∂y = {dL_dy:.4f}")
    print(f"∂y/∂z = {dy_dz:.4f}")
    print(f"∂L/∂z = {dL_dz:.4f} (chain rule)")
    print(f"∂L/∂w = {dL_dw}")
    print(f"∂L/∂b = {dL_db:.4f}")


def example_batch_gradient():
    """Batch vs stochastic gradient."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Batch vs Stochastic Gradient")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 1)
    y = 2 * X.squeeze() + 1 + 0.1 * np.random.randn(n)
    
    def loss(w, b, X, y):
        pred = X.squeeze() * w + b
        return np.mean((pred - y)**2)
    
    def batch_gradient(w, b, X, y):
        pred = X.squeeze() * w + b
        residual = pred - y
        dw = 2 * np.mean(residual * X.squeeze())
        db = 2 * np.mean(residual)
        return dw, db
    
    def sgd_gradient(w, b, xi, yi):
        pred = xi * w + b
        residual = pred - yi
        dw = 2 * residual * xi
        db = 2 * residual
        return dw, db
    
    # Compare
    w, b = 0.0, 0.0
    
    print("Full batch gradient (using all data):")
    dw_batch, db_batch = batch_gradient(w, b, X, y)
    print(f"  ∂L/∂w = {dw_batch:.4f}, ∂L/∂b = {db_batch:.4f}")
    
    print("\nStochastic gradients (single samples):")
    for i in range(3):
        dw_sgd, db_sgd = sgd_gradient(w, b, X[i, 0], y[i])
        print(f"  Sample {i}: ∂L/∂w = {dw_sgd:.4f}, ∂L/∂b = {db_sgd:.4f}")
    
    print("\nNote: SGD gradients are noisy but unbiased!")


if __name__ == "__main__":
    example_partial_derivatives()
    example_gradient_computation()
    example_directional_derivative()
    example_gradient_descent_2d()
    example_gradient_descent_rosenbrock()
    example_linear_regression_gradient()
    example_logistic_regression_gradient()
    example_gradient_checking()
    example_higher_order_partials()
    example_neural_network_gradient()
    example_batch_gradient()
