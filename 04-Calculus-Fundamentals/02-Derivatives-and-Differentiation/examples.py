"""
Derivatives and Differentiation - Examples
==========================================
Practical demonstrations of differentiation concepts.
"""

import numpy as np


def example_derivative_definition():
    """Compute derivative from definition."""
    print("=" * 60)
    print("EXAMPLE 1: Derivative from Definition")
    print("=" * 60)
    
    print("f(x) = x², find f'(x) using the limit definition")
    print("\nf'(x) = lim(h→0) [f(x+h) - f(x)] / h")
    
    def f(x):
        return x**2
    
    x = 3
    print(f"\nCompute f'({x}) numerically:")
    
    for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        derivative_approx = (f(x + h) - f(x)) / h
        print(f"  h = {h}: [f({x}+h) - f({x})]/h = {derivative_approx:.10f}")
    
    print(f"\nAnalytical: f'(x) = 2x, so f'({x}) = {2*x}")


def example_power_rule():
    """Demonstrate the power rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Power Rule")
    print("=" * 60)
    
    print("Power Rule: d/dx[x^n] = n·x^(n-1)")
    
    examples = [
        ("x³", 3, lambda x: x**3, lambda x: 3*x**2),
        ("x⁵", 5, lambda x: x**5, lambda x: 5*x**4),
        ("x^(-1) = 1/x", -1, lambda x: x**(-1), lambda x: -x**(-2)),
        ("x^(1/2) = √x", 0.5, lambda x: x**0.5, lambda x: 0.5*x**(-0.5)),
    ]
    
    x = 2
    print(f"\nEvaluate at x = {x}:")
    
    for name, n, f, df in examples:
        print(f"\n  f(x) = {name}")
        print(f"  f'(x) = {n}·x^({n-1}) = {n}·x^{n-1}")
        print(f"  f'({x}) = {df(x):.6f}")


def example_chain_rule():
    """Demonstrate the chain rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Chain Rule")
    print("=" * 60)
    
    print("Chain Rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)")
    
    print("\n--- Example: f(x) = (3x + 1)⁴ ---")
    print("Let u = g(x) = 3x + 1")
    print("Then f(x) = u⁴")
    print("\nf'(x) = d/du[u⁴] · d/dx[3x + 1]")
    print("     = 4u³ · 3")
    print("     = 12(3x + 1)³")
    
    def f(x):
        return (3*x + 1)**4
    
    def df(x):
        return 12 * (3*x + 1)**3
    
    x = 1
    print(f"\nAt x = {x}:")
    print(f"  f({x}) = {f(x)}")
    print(f"  f'({x}) = 12·(3·{x} + 1)³ = 12·4³ = {df(x)}")
    
    # Verify numerically
    h = 0.0001
    numerical = (f(x + h) - f(x)) / h
    print(f"  Numerical verification: {numerical:.4f}")


def example_product_rule():
    """Demonstrate the product rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Product Rule")
    print("=" * 60)
    
    print("Product Rule: d/dx[f(x)·g(x)] = f'(x)g(x) + f(x)g'(x)")
    
    print("\n--- Example: h(x) = x² · eˣ ---")
    print("f(x) = x², f'(x) = 2x")
    print("g(x) = eˣ, g'(x) = eˣ")
    print("\nh'(x) = 2x·eˣ + x²·eˣ = eˣ(2x + x²) = eˣ·x(2 + x)")
    
    def h(x):
        return x**2 * np.exp(x)
    
    def dh(x):
        return np.exp(x) * (2*x + x**2)
    
    x = 2
    print(f"\nAt x = {x}:")
    print(f"  h({x}) = {h(x):.4f}")
    print(f"  h'({x}) = e^{x}·(2·{x} + {x}²) = {dh(x):.4f}")


def example_sigmoid_derivative():
    """Derive and verify sigmoid derivative."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Sigmoid Derivative")
    print("=" * 60)
    
    print("σ(x) = 1/(1 + e^(-x))")
    print("\n--- Derivation ---")
    print("Let u = 1 + e^(-x)")
    print("σ(x) = u^(-1)")
    print("\nσ'(x) = -u^(-2) · d/dx[1 + e^(-x)]")
    print("      = -1/(1 + e^(-x))² · (-e^(-x))")
    print("      = e^(-x)/(1 + e^(-x))²")
    
    print("\n--- Simplify to σ(1-σ) ---")
    print("σ(x) = 1/(1 + e^(-x))")
    print("1 - σ(x) = e^(-x)/(1 + e^(-x))")
    print("\nσ(x)(1-σ(x)) = 1/(1 + e^(-x)) · e^(-x)/(1 + e^(-x))")
    print("             = e^(-x)/(1 + e^(-x))²")
    print("             = σ'(x) ✓")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)
    
    print("\n--- Numerical verification ---")
    for x in [-2, -1, 0, 1, 2]:
        # Numerical derivative
        h = 0.0001
        numerical = (sigmoid(x + h) - sigmoid(x)) / h
        analytical = sigmoid_derivative(x)
        print(f"  x = {x:2d}: σ'(x) = {analytical:.6f}, numerical = {numerical:.6f}")


def example_relu_derivative():
    """Examine ReLU derivative."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: ReLU Derivative")
    print("=" * 60)
    
    print("ReLU(x) = max(0, x)")
    print("\nReLU'(x) = { 0 if x < 0")
    print("          { 1 if x > 0")
    print("          { undefined if x = 0")
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    print("\n--- Values ---")
    x_vals = np.array([-2, -1, 0, 1, 2])
    for x in x_vals:
        print(f"  x = {x:2d}: ReLU(x) = {relu(x)}, ReLU'(x) = {relu_derivative(x):.0f}")
    
    print("\n--- At x = 0 (special case) ---")
    print("Left derivative: lim(h→0⁻) [ReLU(0+h) - ReLU(0)]/h = 0/h = 0")
    print("Right derivative: lim(h→0⁺) [ReLU(0+h) - ReLU(0)]/h = h/h = 1")
    print("Left ≠ Right → ReLU is not differentiable at x = 0")
    print("\nIn practice, we use subgradient (often 0 or 0.5 at x = 0)")


def example_tanh_derivative():
    """Demonstrate tanh derivative."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Tanh Derivative")
    print("=" * 60)
    
    print("tanh(x) = (eˣ - e^(-x))/(eˣ + e^(-x))")
    print("\ntanh'(x) = 1 - tanh²(x) = sech²(x)")
    
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    print("\n--- Values ---")
    for x in [-2, -1, 0, 1, 2]:
        h = 0.0001
        numerical = (np.tanh(x + h) - np.tanh(x)) / h
        analytical = tanh_derivative(x)
        print(f"  x = {x:2d}: tanh'(x) = {analytical:.6f}, numerical = {numerical:.6f}")
    
    print("\nNote: tanh'(0) = 1 (maximum), tanh'(±∞) → 0 (saturation)")


def example_softplus():
    """Demonstrate softplus as smooth ReLU."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Softplus (Smooth ReLU)")
    print("=" * 60)
    
    print("softplus(x) = ln(1 + eˣ)")
    print("softplus'(x) = eˣ/(1 + eˣ) = σ(x)")
    print("\nSoftplus is a smooth approximation to ReLU!")
    
    def softplus(x):
        return np.log1p(np.exp(np.clip(x, -500, 500)))
    
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    print("\nComparison:")
    print("  x   | ReLU(x) | softplus(x) | softplus'(x)=σ(x)")
    print("  " + "-" * 50)
    for x in [-2, -1, 0, 1, 2, 5]:
        r = relu(x)
        sp = softplus(x)
        dsp = sigmoid(x)
        print(f"  {x:2d}  |  {r:5.2f}  |   {sp:6.3f}   |     {dsp:.4f}")


def example_higher_derivatives():
    """Demonstrate higher-order derivatives."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Higher-Order Derivatives")
    print("=" * 60)
    
    print("f(x) = x⁴ - 2x³ + x²")
    print("\nFirst derivative:")
    print("f'(x) = 4x³ - 6x² + 2x")
    print("\nSecond derivative:")
    print("f''(x) = 12x² - 12x + 2")
    print("\nThird derivative:")
    print("f'''(x) = 24x - 12")
    print("\nFourth derivative:")
    print("f''''(x) = 24")
    
    def f(x): return x**4 - 2*x**3 + x**2
    def df(x): return 4*x**3 - 6*x**2 + 2*x
    def d2f(x): return 12*x**2 - 12*x + 2
    def d3f(x): return 24*x - 12
    def d4f(x): return 24
    
    x = 1
    print(f"\nAt x = {x}:")
    print(f"  f({x}) = {f(x)}")
    print(f"  f'({x}) = {df(x)}")
    print(f"  f''({x}) = {d2f(x)}")
    print(f"  f'''({x}) = {d3f(x)}")
    print(f"  f''''({x}) = {d4f(x)}")


def example_critical_points():
    """Find and classify critical points."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Critical Points and Extrema")
    print("=" * 60)
    
    print("f(x) = x³ - 3x")
    print("\n--- Find critical points ---")
    print("f'(x) = 3x² - 3 = 3(x² - 1) = 3(x-1)(x+1)")
    print("f'(x) = 0 when x = ±1")
    
    print("\n--- Classify using second derivative ---")
    print("f''(x) = 6x")
    print("\nAt x = -1: f''(-1) = -6 < 0 → Local MAXIMUM")
    print("At x = 1:  f''(1) = 6 > 0 → Local MINIMUM")
    
    def f(x): return x**3 - 3*x
    def df(x): return 3*x**2 - 3
    def d2f(x): return 6*x
    
    print("\n--- Values ---")
    for x in [-1, 1]:
        print(f"  x = {x:2d}: f(x) = {f(x):2d}, f'(x) = {df(x)}, f''(x) = {d2f(x):2d}")


def example_gradient_descent_1d():
    """Demonstrate 1D gradient descent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Gradient Descent in 1D")
    print("=" * 60)
    
    print("Minimize: f(x) = (x - 3)²")
    print("f'(x) = 2(x - 3)")
    print("\nGradient descent: x_{t+1} = x_t - α·f'(x_t)")
    
    def f(x):
        return (x - 3)**2
    
    def df(x):
        return 2 * (x - 3)
    
    # Gradient descent
    x = 0  # Starting point
    alpha = 0.1  # Learning rate
    
    print(f"\nStarting at x = {x}, learning rate α = {alpha}")
    print("\nIteration | x       | f(x)    | f'(x)")
    print("-" * 45)
    
    for i in range(10):
        print(f"    {i:2d}    | {x:7.4f} | {f(x):7.4f} | {df(x):7.4f}")
        x = x - alpha * df(x)
    
    print(f"\nConverged to x ≈ 3 (the minimum)")


def example_numerical_differentiation():
    """Compare numerical differentiation methods."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Numerical Differentiation Methods")
    print("=" * 60)
    
    def f(x):
        return np.sin(x)
    
    def df_analytical(x):
        return np.cos(x)
    
    x = 1.0
    true_derivative = df_analytical(x)
    
    print(f"f(x) = sin(x), f'(x) = cos(x)")
    print(f"True f'({x}) = cos({x}) = {true_derivative:.10f}")
    
    print("\n--- Forward Difference: [f(x+h) - f(x)] / h ---")
    print("h          | Approx     | Error")
    print("-" * 40)
    for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        forward = (f(x + h) - f(x)) / h
        error = abs(forward - true_derivative)
        print(f"{h:.5f}    | {forward:.8f} | {error:.2e}")
    
    print("\n--- Central Difference: [f(x+h) - f(x-h)] / (2h) ---")
    print("h          | Approx     | Error")
    print("-" * 40)
    for h in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        central = (f(x + h) - f(x - h)) / (2 * h)
        error = abs(central - true_derivative)
        print(f"{h:.5f}    | {central:.8f} | {error:.2e}")
    
    print("\nCentral difference is more accurate (O(h²) vs O(h))!")


if __name__ == "__main__":
    example_derivative_definition()
    example_power_rule()
    example_chain_rule()
    example_product_rule()
    example_sigmoid_derivative()
    example_relu_derivative()
    example_tanh_derivative()
    example_softplus()
    example_higher_derivatives()
    example_critical_points()
    example_gradient_descent_1d()
    example_numerical_differentiation()
