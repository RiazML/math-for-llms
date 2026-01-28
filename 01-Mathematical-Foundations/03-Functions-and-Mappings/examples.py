"""
Functions and Mappings - Examples
=================================
Practical demonstrations of function concepts.
"""

import numpy as np
import matplotlib.pyplot as plt


def example_basic_functions():
    """Demonstrate basic function concepts."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Functions")
    print("=" * 60)
    
    # Define functions
    def f(x):
        return x ** 2
    
    def g(x):
        return 2 * x + 1
    
    def h(x):
        return np.exp(x)
    
    # Evaluate at points
    x_vals = np.array([-2, -1, 0, 1, 2])
    
    print("x values:", x_vals)
    print(f"\nf(x) = x²")
    print(f"f(x): {f(x_vals)}")
    
    print(f"\ng(x) = 2x + 1")
    print(f"g(x): {g(x_vals)}")
    
    print(f"\nh(x) = eˣ")
    print(f"h(x): {np.round(h(x_vals), 4)}")


def example_domain_range():
    """Demonstrate domain, codomain, and range."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Domain, Codomain, and Range")
    print("=" * 60)
    
    print("f(x) = x²")
    print("- Domain: ℝ (all real numbers)")
    print("- Codomain: ℝ")
    print("- Range: [0, ∞) (only non-negative)")
    
    print("\ng(x) = √x")
    print("- Domain: [0, ∞)")
    print("- Codomain: ℝ")
    print("- Range: [0, ∞)")
    
    print("\nh(x) = 1/x")
    print("- Domain: ℝ \\ {0}")
    print("- Codomain: ℝ")
    print("- Range: ℝ \\ {0}")
    
    print("\nσ(x) = 1/(1 + e⁻ˣ) (sigmoid)")
    print("- Domain: ℝ")
    print("- Codomain: ℝ")
    print("- Range: (0, 1)")
    
    # Verify range of sigmoid
    x = np.linspace(-10, 10, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    print(f"\nSigmoid range verification:")
    print(f"  min(σ(x)) ≈ {sigmoid.min():.6f} (approaches 0)")
    print(f"  max(σ(x)) ≈ {sigmoid.max():.6f} (approaches 1)")


def example_injective():
    """Demonstrate injective (one-to-one) functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Injective Functions")
    print("=" * 60)
    
    print("Testing if functions are injective (one-to-one):")
    
    # f(x) = x² on ℝ
    print("\n1. f(x) = x² on ℝ")
    print("   f(-2) = 4")
    print("   f(2) = 4")
    print("   Different inputs (-2, 2) give same output")
    print("   NOT INJECTIVE ✗")
    
    # f(x) = x² on [0, ∞)
    print("\n2. f(x) = x² on [0, ∞)")
    print("   For x₁, x₂ ≥ 0: x₁² = x₂² ⟹ x₁ = x₂")
    print("   INJECTIVE ✓")
    
    # g(x) = 2x + 1
    print("\n3. g(x) = 2x + 1")
    print("   If g(x₁) = g(x₂):")
    print("   2x₁ + 1 = 2x₂ + 1")
    print("   x₁ = x₂")
    print("   INJECTIVE ✓")
    
    # sin(x)
    print("\n4. sin(x) on ℝ")
    print("   sin(0) = 0")
    print("   sin(π) = 0")
    print("   Different inputs give same output")
    print("   NOT INJECTIVE ✗")
    
    # sin(x) restricted
    print("\n5. sin(x) on [-π/2, π/2]")
    print("   Strictly increasing on this interval")
    print("   INJECTIVE ✓")


def example_surjective():
    """Demonstrate surjective (onto) functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Surjective Functions")
    print("=" * 60)
    
    print("Testing if functions are surjective (onto):")
    
    # f(x) = x² from ℝ to ℝ
    print("\n1. f: ℝ → ℝ, f(x) = x²")
    print("   Can we hit y = -1?")
    print("   x² = -1 has no real solution")
    print("   NOT SURJECTIVE onto ℝ ✗")
    
    # f(x) = x² from ℝ to [0, ∞)
    print("\n2. f: ℝ → [0, ∞), f(x) = x²")
    print("   Every y ≥ 0 is hit by x = √y")
    print("   SURJECTIVE onto [0, ∞) ✓")
    
    # g(x) = eˣ from ℝ to ℝ
    print("\n3. g: ℝ → ℝ, g(x) = eˣ")
    print("   Can we hit y = -1?")
    print("   eˣ = -1 has no solution")
    print("   NOT SURJECTIVE onto ℝ ✗")
    
    # g(x) = eˣ from ℝ to (0, ∞)
    print("\n4. g: ℝ → (0, ∞), g(x) = eˣ")
    print("   Every y > 0 is hit by x = ln(y)")
    print("   SURJECTIVE onto (0, ∞) ✓")
    
    # h(x) = x³
    print("\n5. h: ℝ → ℝ, h(x) = x³")
    print("   Every y ∈ ℝ is hit by x = ∛y")
    print("   SURJECTIVE onto ℝ ✓")


def example_bijective():
    """Demonstrate bijective functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Bijective Functions")
    print("=" * 60)
    
    print("Bijective = Injective + Surjective")
    print("Bijective functions have inverses")
    
    print("\n1. f(x) = 2x + 3 (ℝ → ℝ)")
    print("   Injective: Different x give different 2x + 3 ✓")
    print("   Surjective: Every y = 2x + 3 ⟹ x = (y-3)/2 ✓")
    print("   BIJECTIVE ✓")
    print("   Inverse: f⁻¹(x) = (x - 3)/2")
    
    print("\n2. f(x) = x³ (ℝ → ℝ)")
    print("   Injective: x₁³ = x₂³ ⟹ x₁ = x₂ ✓")
    print("   Surjective: Every y has x = ∛y ✓")
    print("   BIJECTIVE ✓")
    print("   Inverse: f⁻¹(x) = ∛x")
    
    print("\n3. σ(x) = 1/(1+e⁻ˣ) (ℝ → (0,1))")
    print("   Injective: Strictly increasing ✓")
    print("   Surjective: Covers all (0, 1) ✓")
    print("   BIJECTIVE onto (0, 1) ✓")
    print("   Inverse: σ⁻¹(y) = ln(y/(1-y)) (logit function)")
    
    # Verify sigmoid inverse
    x = 2.0
    sigma_x = 1 / (1 + np.exp(-x))
    inverse = np.log(sigma_x / (1 - sigma_x))
    print(f"\n   Verification: σ(2) = {sigma_x:.4f}")
    print(f"   σ⁻¹({sigma_x:.4f}) = {inverse:.4f} ✓")


def example_composition():
    """Demonstrate function composition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Function Composition")
    print("=" * 60)
    
    def f(x):
        return x + 1
    
    def g(x):
        return x ** 2
    
    def h(x):
        return np.sin(x)
    
    x = 2
    
    print(f"f(x) = x + 1")
    print(f"g(x) = x²")
    print(f"h(x) = sin(x)")
    print(f"\nx = {x}")
    
    # g ∘ f
    print(f"\n(g ∘ f)(x) = g(f(x)) = g(x + 1) = (x + 1)²")
    print(f"(g ∘ f)({x}) = g(f({x})) = g({f(x)}) = {g(f(x))}")
    
    # f ∘ g
    print(f"\n(f ∘ g)(x) = f(g(x)) = f(x²) = x² + 1")
    print(f"(f ∘ g)({x}) = f(g({x})) = f({g(x)}) = {f(g(x))}")
    
    print(f"\nNote: (g ∘ f)({x}) = {g(f(x))} ≠ {f(g(x))} = (f ∘ g)({x})")
    print("Composition is NOT commutative!")
    
    # Triple composition
    print(f"\n(h ∘ g ∘ f)(x) = h(g(f(x)))")
    print(f"(h ∘ g ∘ f)({x}) = h(g(f({x}))) = h(g({f(x)})) = h({g(f(x))}) = {h(g(f(x))):.4f}")


def example_inverse():
    """Demonstrate inverse functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Inverse Functions")
    print("=" * 60)
    
    # Linear function
    print("1. f(x) = 3x - 2")
    print("   Finding inverse:")
    print("   y = 3x - 2")
    print("   y + 2 = 3x")
    print("   x = (y + 2)/3")
    print("   f⁻¹(x) = (x + 2)/3")
    
    f = lambda x: 3*x - 2
    f_inv = lambda x: (x + 2)/3
    
    x = 5
    print(f"\n   Verification: f({x}) = {f(x)}")
    print(f"   f⁻¹({f(x)}) = {f_inv(f(x))}")
    print(f"   f⁻¹(f({x})) = {x} ✓")
    
    # Exponential
    print("\n2. f(x) = eˣ")
    print("   f⁻¹(x) = ln(x)")
    
    x = 2
    print(f"\n   f({x}) = e^{x} = {np.exp(x):.4f}")
    print(f"   f⁻¹({np.exp(x):.4f}) = ln({np.exp(x):.4f}) = {np.log(np.exp(x)):.4f}")
    
    # More complex
    print("\n3. f(x) = (x + 1)/(x - 1)")
    print("   Finding inverse:")
    print("   y = (x + 1)/(x - 1)")
    print("   y(x - 1) = x + 1")
    print("   yx - y = x + 1")
    print("   yx - x = y + 1")
    print("   x(y - 1) = y + 1")
    print("   x = (y + 1)/(y - 1)")
    print("   f⁻¹(x) = (x + 1)/(x - 1)")
    print("   Note: f = f⁻¹! (self-inverse function)")
    
    f = lambda x: (x + 1)/(x - 1)
    x = 3
    print(f"\n   f({x}) = {f(x)}")
    print(f"   f(f({x})) = f({f(x)}) = {f(f(x))} = {x} ✓")


def example_multivariate():
    """Demonstrate multivariate functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Multivariate Functions")
    print("=" * 60)
    
    # Scalar-valued multivariate function
    print("1. Scalar-valued: f(x, y) = x² + y²")
    
    def f(x, y):
        return x**2 + y**2
    
    points = [(1, 0), (0, 1), (1, 1), (3, 4)]
    print("   (x, y) → f(x, y)")
    for x, y in points:
        print(f"   ({x}, {y}) → {f(x, y)}")
    
    # Vector-valued function
    print("\n2. Vector-valued: g(t) = (cos(t), sin(t))")
    print("   Maps ℝ → ℝ² (parametric circle)")
    
    def g(t):
        return np.array([np.cos(t), np.sin(t)])
    
    t_vals = [0, np.pi/4, np.pi/2, np.pi]
    print("   t → g(t)")
    for t in t_vals:
        result = g(t)
        print(f"   {t:.4f} → ({result[0]:.4f}, {result[1]:.4f})")
    
    # Matrix function (linear transformation)
    print("\n3. Linear transformation: f(x) = Ax")
    A = np.array([[2, 1],
                  [1, 3]])
    print(f"   A = \n{A}")
    
    x = np.array([1, 2])
    print(f"   x = {x}")
    print(f"   f(x) = Ax = {A @ x}")


def example_activation_functions():
    """Demonstrate common ML activation functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: ML Activation Functions")
    print("=" * 60)
    
    x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    # Sigmoid
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    print("1. Sigmoid: σ(x) = 1/(1 + e⁻ˣ)")
    print(f"   Domain: ℝ, Range: (0, 1)")
    print(f"   x:    {x}")
    print(f"   σ(x): {np.round(sigmoid(x), 4)}")
    
    # Tanh
    def tanh(x):
        return np.tanh(x)
    
    print("\n2. Tanh: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)")
    print(f"   Domain: ℝ, Range: (-1, 1)")
    print(f"   x:      {x}")
    print(f"   tanh(x): {np.round(tanh(x), 4)}")
    
    # ReLU
    def relu(x):
        return np.maximum(0, x)
    
    print("\n3. ReLU: f(x) = max(0, x)")
    print(f"   Domain: ℝ, Range: [0, ∞)")
    print(f"   x:      {x}")
    print(f"   ReLU(x): {relu(x)}")
    print("   NOT injective (all negatives map to 0)")
    
    # Leaky ReLU
    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)
    
    print("\n4. Leaky ReLU: f(x) = max(αx, x) with α = 0.1")
    print(f"   x:          {x}")
    print(f"   LeakyReLU(x): {leaky_relu(x)}")
    print("   IS injective (different slopes for + and -)")
    
    # Softmax
    def softmax(z):
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / exp_z.sum()
    
    print("\n5. Softmax: σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ")
    z = np.array([1.0, 2.0, 3.0])
    print(f"   Input z: {z}")
    print(f"   Softmax(z): {np.round(softmax(z), 4)}")
    print(f"   Sum: {softmax(z).sum():.4f} = 1 ✓")


def example_loss_functions():
    """Demonstrate loss functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Loss Functions")
    print("=" * 60)
    
    # MSE
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.2, 2.8, 4.1])
    
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    print("1. Mean Squared Error (MSE)")
    print(f"   y_true: {y_true}")
    print(f"   y_pred: {y_pred}")
    print(f"   MSE = (1/n) Σ(yᵢ - ŷᵢ)² = {mse(y_true, y_pred):.4f}")
    
    # MAE
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    print(f"\n2. Mean Absolute Error (MAE)")
    print(f"   MAE = (1/n) Σ|yᵢ - ŷᵢ| = {mae(y_true, y_pred):.4f}")
    
    # Binary Cross-Entropy
    y_true_binary = np.array([1, 0, 1, 1])
    y_pred_prob = np.array([0.9, 0.1, 0.8, 0.7])
    
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15  # Prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    print(f"\n3. Binary Cross-Entropy")
    print(f"   y_true: {y_true_binary}")
    print(f"   y_pred: {y_pred_prob}")
    print(f"   BCE = -mean[y·log(p) + (1-y)·log(1-p)] = {binary_cross_entropy(y_true_binary, y_pred_prob):.4f}")
    
    # Categorical Cross-Entropy
    y_true_cat = np.array([1, 0, 0])  # One-hot encoded
    y_pred_cat = np.array([0.7, 0.2, 0.1])
    
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1)
        return -np.sum(y_true * np.log(y_pred))
    
    print(f"\n4. Categorical Cross-Entropy")
    print(f"   y_true (one-hot): {y_true_cat}")
    print(f"   y_pred (probs):   {y_pred_cat}")
    print(f"   CCE = -Σ yᵢ·log(pᵢ) = {categorical_cross_entropy(y_true_cat, y_pred_cat):.4f}")


def example_composition_ml():
    """Demonstrate function composition in neural networks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Neural Network as Function Composition")
    print("=" * 60)
    
    # Simple 2-layer network
    np.random.seed(42)
    
    # Layer functions
    def linear(W, b):
        return lambda x: W @ x + b
    
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Network architecture
    # Input: 2D, Hidden: 3D, Output: 1D
    W1 = np.array([[0.5, 0.3],
                   [-0.2, 0.4],
                   [0.1, -0.3]])
    b1 = np.array([0.1, -0.1, 0.2])
    
    W2 = np.array([[0.4, -0.3, 0.2]])
    b2 = np.array([0.1])
    
    print("Network Architecture:")
    print("  Input (2D) → Linear → ReLU → Linear → Sigmoid → Output (1D)")
    print(f"\n  W1 shape: {W1.shape}, b1 shape: {b1.shape}")
    print(f"  W2 shape: {W2.shape}, b2 shape: {b2.shape}")
    
    # Forward pass as composition
    x = np.array([1.0, 2.0])
    print(f"\nInput x = {x}")
    
    # Step by step
    z1 = W1 @ x + b1
    print(f"z1 = W1·x + b1 = {z1}")
    
    a1 = relu(z1)
    print(f"a1 = ReLU(z1) = {a1}")
    
    z2 = W2 @ a1 + b2
    print(f"z2 = W2·a1 + b2 = {z2}")
    
    output = sigmoid(z2)
    print(f"output = σ(z2) = {output}")
    
    # As single composition
    def neural_net(x):
        return sigmoid(W2 @ relu(W1 @ x + b1) + b2)
    
    print(f"\nAs composition: f(x) = σ(W2·ReLU(W1·x + b1) + b2)")
    print(f"f({x}) = {neural_net(x)}")


def example_jacobian():
    """Demonstrate Jacobian matrix."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Jacobian Matrix")
    print("=" * 60)
    
    print("For f: ℝ² → ℝ³ with:")
    print("  f₁(x, y) = x² + y")
    print("  f₂(x, y) = xy")
    print("  f₃(x, y) = sin(x)")
    
    print("\nJacobian matrix J:")
    print("  ∂f₁/∂x = 2x    ∂f₁/∂y = 1")
    print("  ∂f₂/∂x = y     ∂f₂/∂y = x")
    print("  ∂f₃/∂x = cos(x) ∂f₃/∂y = 0")
    
    def jacobian(x, y):
        return np.array([
            [2*x, 1],
            [y, x],
            [np.cos(x), 0]
        ])
    
    x, y = 1.0, 2.0
    J = jacobian(x, y)
    
    print(f"\nAt (x, y) = ({x}, {y}):")
    print(f"J = \n{J}")
    
    print("\nJacobian gives the best linear approximation:")
    print("f(x + Δx) ≈ f(x) + J·Δx")


def visualize_functions():
    """Visualize various functions."""
    print("\n" + "=" * 60)
    print("VISUALIZATION: Common Functions")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    x = np.linspace(-3, 3, 100)
    
    # 1. Quadratic
    ax = axes[0, 0]
    ax.plot(x, x**2, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title('f(x) = x²\n(Not injective)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    # 2. Cubic
    ax = axes[0, 1]
    ax.plot(x, x**3, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title('f(x) = x³\n(Bijective)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    
    # 3. Exponential
    ax = axes[0, 2]
    ax.plot(x, np.exp(x), 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=1, color='r', linewidth=0.5, linestyle='--')
    ax.set_title('f(x) = eˣ\n(Injective, Range: (0, ∞))')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_ylim([-0.5, 10])
    
    # 4. Sigmoid
    ax = axes[1, 0]
    sigmoid = 1 / (1 + np.exp(-x))
    ax.plot(x, sigmoid, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='r', linewidth=0.5, linestyle='--')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=1, color='k', linewidth=0.5)
    ax.set_title('σ(x) = 1/(1+e⁻ˣ)\n(Sigmoid, Range: (0,1))')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('σ(x)')
    
    # 5. ReLU
    ax = axes[1, 1]
    relu = np.maximum(0, x)
    ax.plot(x, relu, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_title('ReLU(x) = max(0, x)\n(Not injective)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('ReLU(x)')
    
    # 6. Tanh
    ax = axes[1, 2]
    ax.plot(x, np.tanh(x), 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=1, color='r', linewidth=0.5, linestyle='--')
    ax.axhline(y=-1, color='r', linewidth=0.5, linestyle='--')
    ax.set_title('tanh(x)\n(Range: (-1, 1))')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('tanh(x)')
    
    plt.tight_layout()
    plt.savefig('functions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: functions_visualization.png")


if __name__ == "__main__":
    example_basic_functions()
    example_domain_range()
    example_injective()
    example_surjective()
    example_bijective()
    example_composition()
    example_inverse()
    example_multivariate()
    example_activation_functions()
    example_loss_functions()
    example_composition_ml()
    example_jacobian()
    
    # Uncomment to generate visualization
    # visualize_functions()
