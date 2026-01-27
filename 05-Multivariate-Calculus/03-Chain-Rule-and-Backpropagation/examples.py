"""
Chain Rule and Backpropagation - Examples
=========================================
Practical demonstrations of backpropagation and chain rule.
"""

import numpy as np


def example_scalar_chain_rule():
    """Demonstrate scalar chain rule."""
    print("=" * 60)
    print("EXAMPLE 1: Scalar Chain Rule")
    print("=" * 60)
    
    print("y = sin(x²)")
    print("Let u = x², then y = sin(u)")
    print("\ndy/dx = dy/du · du/dx")
    print("      = cos(u) · 2x")
    print("      = 2x · cos(x²)")
    
    x = 1.0
    
    # Analytical
    dy_dx_analytical = 2 * x * np.cos(x**2)
    
    # Numerical
    h = 1e-7
    y_plus = np.sin((x + h)**2)
    y_minus = np.sin((x - h)**2)
    dy_dx_numerical = (y_plus - y_minus) / (2 * h)
    
    print(f"\nAt x = {x}:")
    print(f"  Analytical: dy/dx = {dy_dx_analytical:.6f}")
    print(f"  Numerical:  dy/dx = {dy_dx_numerical:.6f}")


def example_vector_chain_rule():
    """Demonstrate vector chain rule."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Vector Chain Rule")
    print("=" * 60)
    
    print("L(y(x)) where:")
    print("  x ∈ ℝ²")
    print("  y = [x₁², x₁x₂]ᵀ ∈ ℝ²")
    print("  L = y₁ + y₂ ∈ ℝ")
    
    print("\n∂L/∂y = [1, 1]ᵀ")
    print("\nJacobian of y:")
    print("J_y = [2x₁  0  ]")
    print("      [x₂   x₁ ]")
    
    print("\n∂L/∂x = J_yᵀ · ∂L/∂y")
    print("      = [2x₁  x₂] [1]   [2x₁ + x₂]")
    print("        [0    x₁] [1] = [x₁      ]")
    
    x = np.array([2.0, 3.0])
    
    def y_func(x):
        return np.array([x[0]**2, x[0]*x[1]])
    
    def L_func(x):
        y = y_func(x)
        return y[0] + y[1]
    
    # Analytical gradient
    grad_analytical = np.array([2*x[0] + x[1], x[0]])
    
    # Numerical gradient
    grad_numerical = np.zeros(2)
    h = 1e-7
    for i in range(2):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad_numerical[i] = (L_func(x_plus) - L_func(x_minus)) / (2*h)
    
    print(f"\nAt x = {x}:")
    print(f"  Analytical: ∂L/∂x = {grad_analytical}")
    print(f"  Numerical:  ∂L/∂x = {grad_numerical.round(6)}")


def example_simple_backprop():
    """Simple backpropagation example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Simple Backpropagation")
    print("=" * 60)
    
    print("Network: x → Linear(w,b) → Sigmoid → MSE Loss")
    print("  z = wx + b")
    print("  y = σ(z)")
    print("  L = (y - t)²")
    
    # Parameters
    w = 0.5
    b = 0.1
    x = 2.0
    t = 0.8  # Target
    
    # Forward pass
    z = w * x + b
    y = 1 / (1 + np.exp(-z))
    L = (y - t)**2
    
    print(f"\n--- Forward Pass ---")
    print(f"  x = {x}, w = {w}, b = {b}")
    print(f"  z = wx + b = {z}")
    print(f"  y = σ(z) = {y:.4f}")
    print(f"  L = (y - t)² = {L:.4f}")
    
    # Backward pass
    dL_dy = 2 * (y - t)
    dy_dz = y * (1 - y)  # sigmoid derivative
    dz_dw = x
    dz_db = 1
    
    # Chain rule
    dL_dz = dL_dy * dy_dz
    dL_dw = dL_dz * dz_dw
    dL_db = dL_dz * dz_db
    
    print(f"\n--- Backward Pass ---")
    print(f"  ∂L/∂y = 2(y - t) = {dL_dy:.4f}")
    print(f"  ∂y/∂z = y(1-y) = {dy_dz:.4f}")
    print(f"  ∂L/∂z = ∂L/∂y · ∂y/∂z = {dL_dz:.4f}")
    print(f"  ∂L/∂w = ∂L/∂z · x = {dL_dw:.4f}")
    print(f"  ∂L/∂b = ∂L/∂z = {dL_db:.4f}")
    
    # Numerical verification
    h = 1e-7
    def compute_loss(w, b):
        z = w * x + b
        y = 1 / (1 + np.exp(-z))
        return (y - t)**2
    
    dL_dw_num = (compute_loss(w + h, b) - compute_loss(w - h, b)) / (2*h)
    dL_db_num = (compute_loss(w, b + h) - compute_loss(w, b - h)) / (2*h)
    
    print(f"\n--- Numerical Verification ---")
    print(f"  ∂L/∂w: analytical = {dL_dw:.6f}, numerical = {dL_dw_num:.6f}")
    print(f"  ∂L/∂b: analytical = {dL_db:.6f}, numerical = {dL_db_num:.6f}")


def example_two_layer_network():
    """Backpropagation through 2-layer network."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Two-Layer Network Backpropagation")
    print("=" * 60)
    
    print("Network: x → [W₁,b₁] → ReLU → [W₂,b₂] → MSE")
    
    np.random.seed(42)
    
    # Dimensions
    n_input = 3
    n_hidden = 4
    n_output = 2
    
    # Initialize
    W1 = np.random.randn(n_hidden, n_input) * 0.1
    b1 = np.zeros(n_hidden)
    W2 = np.random.randn(n_output, n_hidden) * 0.1
    b2 = np.zeros(n_output)
    
    x = np.random.randn(n_input)
    t = np.random.randn(n_output)
    
    def relu(z):
        return np.maximum(0, z)
    
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    # Forward pass
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    L = np.sum((z2 - t)**2)
    
    print(f"\n--- Forward Pass ---")
    print(f"  x shape: {x.shape}")
    print(f"  z1 shape: {z1.shape}, a1 shape: {a1.shape}")
    print(f"  z2 shape: {z2.shape}")
    print(f"  Loss L = {L:.4f}")
    
    # Backward pass
    dL_dz2 = 2 * (z2 - t)                    # (n_output,)
    dL_dW2 = np.outer(dL_dz2, a1)            # (n_output, n_hidden)
    dL_db2 = dL_dz2                          # (n_output,)
    
    dL_da1 = W2.T @ dL_dz2                   # (n_hidden,)
    dL_dz1 = dL_da1 * relu_derivative(z1)   # (n_hidden,)
    dL_dW1 = np.outer(dL_dz1, x)            # (n_hidden, n_input)
    dL_db1 = dL_dz1                         # (n_hidden,)
    
    print(f"\n--- Backward Pass ---")
    print(f"  ∂L/∂z₂ shape: {dL_dz2.shape}")
    print(f"  ∂L/∂W₂ shape: {dL_dW2.shape}")
    print(f"  ∂L/∂a₁ shape: {dL_da1.shape}")
    print(f"  ∂L/∂z₁ shape: {dL_dz1.shape}")
    print(f"  ∂L/∂W₁ shape: {dL_dW1.shape}")
    
    # Numerical gradient check for W1
    h = 1e-5
    dL_dW1_numerical = np.zeros_like(W1)
    
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = W1.copy()
            W1_plus[i, j] += h
            z1_plus = W1_plus @ x + b1
            a1_plus = relu(z1_plus)
            z2_plus = W2 @ a1_plus + b2
            L_plus = np.sum((z2_plus - t)**2)
            
            W1_minus = W1.copy()
            W1_minus[i, j] -= h
            z1_minus = W1_minus @ x + b1
            a1_minus = relu(z1_minus)
            z2_minus = W2 @ a1_minus + b2
            L_minus = np.sum((z2_minus - t)**2)
            
            dL_dW1_numerical[i, j] = (L_plus - L_minus) / (2 * h)
    
    rel_error = np.linalg.norm(dL_dW1 - dL_dW1_numerical) / (
        np.linalg.norm(dL_dW1) + np.linalg.norm(dL_dW1_numerical) + 1e-10
    )
    
    print(f"\n--- Gradient Check ---")
    print(f"  Relative error for W₁: {rel_error:.2e}")
    print(f"  Check {'PASSED' if rel_error < 1e-5 else 'FAILED'}")


def example_softmax_crossentropy():
    """Backprop through softmax + cross-entropy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Softmax + Cross-Entropy Backprop")
    print("=" * 60)
    
    print("Common trick: Combined gradient is simple!")
    print("  p = softmax(z)")
    print("  L = -Σ yᵢ log(pᵢ)  (cross-entropy)")
    print("  ∂L/∂z = p - y")
    
    def softmax(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    z = np.array([1.0, 2.0, 3.0])
    y = np.array([0, 0, 1])  # One-hot label (class 2)
    
    # Forward
    p = softmax(z)
    L = -np.sum(y * np.log(p + 1e-10))
    
    print(f"\nForward:")
    print(f"  z = {z}")
    print(f"  p = softmax(z) = {p.round(4)}")
    print(f"  y = {y}")
    print(f"  L = {L:.4f}")
    
    # Simple combined gradient
    dL_dz_simple = p - y
    
    print(f"\nCombined gradient:")
    print(f"  ∂L/∂z = p - y = {dL_dz_simple.round(4)}")
    
    # Verify by computing separately
    # ∂L/∂p_k = -y_k / p_k
    # ∂p_i/∂z_j = p_i(δ_ij - p_j)
    # ∂L/∂z_j = Σ_i (∂L/∂p_i)(∂p_i/∂z_j)
    #         = Σ_i (-y_i/p_i) p_i(δ_ij - p_j)
    #         = -Σ_i y_i(δ_ij - p_j)
    #         = -y_j + p_j Σ_i y_i
    #         = p_j - y_j  (since Σy_i = 1)
    
    print("\nDerivation confirms: ∂L/∂z = p - y")
    
    # Numerical check
    h = 1e-7
    dL_dz_numerical = np.zeros(3)
    for j in range(3):
        z_plus = z.copy()
        z_plus[j] += h
        p_plus = softmax(z_plus)
        L_plus = -np.sum(y * np.log(p_plus + 1e-10))
        
        z_minus = z.copy()
        z_minus[j] -= h
        p_minus = softmax(z_minus)
        L_minus = -np.sum(y * np.log(p_minus + 1e-10))
        
        dL_dz_numerical[j] = (L_plus - L_minus) / (2*h)
    
    print(f"\nNumerical: {dL_dz_numerical.round(4)}")


def example_gradient_checking():
    """Comprehensive gradient checking example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Gradient Checking")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Simple function
    def f(theta):
        x, y, z = theta
        return x**2 * y + np.sin(z) * y
    
    def grad_f(theta):
        x, y, z = theta
        return np.array([
            2*x*y,           # ∂f/∂x
            x**2 + np.sin(z), # ∂f/∂y
            np.cos(z) * y    # ∂f/∂z
        ])
    
    theta = np.array([1.0, 2.0, 3.0])
    
    # Analytical gradient
    g_analytical = grad_f(theta)
    
    # Numerical gradient
    def numerical_gradient(f, theta, h=1e-7):
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += h
            theta_minus = theta.copy()
            theta_minus[i] -= h
            grad[i] = (f(theta_plus) - f(theta_minus)) / (2*h)
        return grad
    
    g_numerical = numerical_gradient(f, theta)
    
    print(f"θ = {theta}")
    print(f"\nAnalytical gradient: {g_analytical.round(6)}")
    print(f"Numerical gradient:  {g_numerical.round(6)}")
    
    # Relative error
    rel_error = np.linalg.norm(g_analytical - g_numerical) / (
        np.linalg.norm(g_analytical) + np.linalg.norm(g_numerical) + 1e-10
    )
    
    print(f"\nRelative error: {rel_error:.2e}")
    print(f"Check {'PASSED ✓' if rel_error < 1e-5 else 'FAILED ✗'}")
    
    # Element-wise comparison
    print("\nElement-wise comparison:")
    for i, (a, n) in enumerate(zip(g_analytical, g_numerical)):
        rel = abs(a - n) / (abs(a) + abs(n) + 1e-10)
        status = '✓' if rel < 1e-5 else '✗'
        print(f"  θ[{i}]: analytical={a:.6f}, numerical={n:.6f}, rel_err={rel:.2e} {status}")


def example_batch_gradient():
    """Gradient computation with batched data."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Batched Gradient Computation")
    print("=" * 60)
    
    print("For batch of inputs X (batch_size × features):")
    print("  Z = XW + b")
    print("  Loss averaged over batch")
    
    np.random.seed(42)
    
    batch_size = 4
    n_features = 3
    n_outputs = 2
    
    X = np.random.randn(batch_size, n_features)
    W = np.random.randn(n_features, n_outputs) * 0.1
    b = np.zeros(n_outputs)
    Y_true = np.random.randn(batch_size, n_outputs)
    
    # Forward pass
    Z = X @ W + b
    L = np.mean((Z - Y_true)**2)
    
    print(f"\nX shape: {X.shape}")
    print(f"W shape: {W.shape}")
    print(f"Z shape: {Z.shape}")
    print(f"Loss: {L:.4f}")
    
    # Backward pass
    dL_dZ = 2 * (Z - Y_true) / batch_size  # (batch, output)
    dL_dW = X.T @ dL_dZ                     # (features, output)
    dL_db = np.sum(dL_dZ, axis=0)           # (output,)
    
    print(f"\n∂L/∂Z shape: {dL_dZ.shape}")
    print(f"∂L/∂W shape: {dL_dW.shape}")
    print(f"∂L/∂b shape: {dL_db.shape}")
    
    # Verify
    h = 1e-5
    dL_dW_numerical = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += h
            L_plus = np.mean((X @ W_plus + b - Y_true)**2)
            
            W_minus = W.copy()
            W_minus[i, j] -= h
            L_minus = np.mean((X @ W_minus + b - Y_true)**2)
            
            dL_dW_numerical[i, j] = (L_plus - L_minus) / (2*h)
    
    rel_error = np.linalg.norm(dL_dW - dL_dW_numerical) / (
        np.linalg.norm(dL_dW) + np.linalg.norm(dL_dW_numerical) + 1e-10
    )
    print(f"\nGradient check for W: rel_error = {rel_error:.2e}")


def example_computational_graph():
    """Visualize computational graph and backprop."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Computational Graph")
    print("=" * 60)
    
    print("Expression: L = (a * b + c)²")
    print("")
    print("Computational graph:")
    print("  a ───┐")
    print("       ├──→ [×] ──→ d ─┐")
    print("  b ───┘               │")
    print("                       ├──→ [+] ──→ e ──→ [²] ──→ L")
    print("  c ───────────────────┘")
    
    a, b, c = 2.0, 3.0, 1.0
    
    # Forward pass
    d = a * b      # multiplication
    e = d + c      # addition
    L = e ** 2     # square
    
    print(f"\nForward pass:")
    print(f"  a = {a}, b = {b}, c = {c}")
    print(f"  d = a × b = {d}")
    print(f"  e = d + c = {e}")
    print(f"  L = e² = {L}")
    
    # Backward pass
    dL_de = 2 * e           # ∂L/∂e = 2e
    de_dd = 1               # ∂e/∂d = 1
    de_dc = 1               # ∂e/∂c = 1
    dd_da = b               # ∂d/∂a = b
    dd_db = a               # ∂d/∂b = a
    
    dL_dd = dL_de * de_dd   # Chain rule
    dL_dc = dL_de * de_dc
    dL_da = dL_dd * dd_da
    dL_db = dL_dd * dd_db
    
    print(f"\nBackward pass:")
    print(f"  ∂L/∂e = 2e = {dL_de}")
    print(f"  ∂L/∂d = ∂L/∂e × 1 = {dL_dd}")
    print(f"  ∂L/∂c = ∂L/∂e × 1 = {dL_dc}")
    print(f"  ∂L/∂a = ∂L/∂d × b = {dL_da}")
    print(f"  ∂L/∂b = ∂L/∂d × a = {dL_db}")
    
    # Verify
    print("\nVerification (analytical):")
    print(f"  L = (ab + c)²")
    print(f"  ∂L/∂a = 2(ab + c) × b = 2 × {e} × {b} = {2*e*b}")
    print(f"  ∂L/∂b = 2(ab + c) × a = 2 × {e} × {a} = {2*e*a}")
    print(f"  ∂L/∂c = 2(ab + c) × 1 = 2 × {e} = {2*e}")


def example_vanishing_gradient():
    """Demonstrate vanishing gradient problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Vanishing Gradient Problem")
    print("=" * 60)
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    print("Deep network with sigmoid activations:")
    print("x → σ → σ → σ → σ → ... → σ → L")
    print("\nσ'(z) = σ(z)(1-σ(z)) ≤ 0.25")
    print("After n layers: gradient × (0.25)ⁿ")
    
    n_layers = 10
    z = 0.5  # Typical value
    
    sigma_prime = sigmoid(z) * (1 - sigmoid(z))
    
    print(f"\nσ'({z}) = {sigma_prime:.4f}")
    
    gradient_factor = sigma_prime ** n_layers
    print(f"\nAfter {n_layers} layers:")
    print(f"  Gradient factor: {sigma_prime}^{n_layers} = {gradient_factor:.2e}")
    
    print("\n--- ReLU solves this ---")
    print("ReLU'(z) = 1 for z > 0")
    print("Gradient doesn't shrink through layers!")


def example_layer_class():
    """Implement layer as class with forward/backward."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Layer Class Implementation")
    print("=" * 60)
    
    class Linear:
        def __init__(self, in_features, out_features):
            self.W = np.random.randn(out_features, in_features) * 0.1
            self.b = np.zeros(out_features)
            self.grad_W = None
            self.grad_b = None
        
        def forward(self, x):
            self.x = x  # Cache for backward
            return self.W @ x + self.b
        
        def backward(self, grad_output):
            self.grad_W = np.outer(grad_output, self.x)
            self.grad_b = grad_output
            return self.W.T @ grad_output
    
    class Sigmoid:
        def forward(self, z):
            self.out = 1 / (1 + np.exp(-z))
            return self.out
        
        def backward(self, grad_output):
            return grad_output * self.out * (1 - self.out)
    
    class MSELoss:
        def forward(self, pred, target):
            self.pred = pred
            self.target = target
            return np.sum((pred - target)**2)
        
        def backward(self):
            return 2 * (self.pred - self.target)
    
    # Build network
    linear1 = Linear(3, 4)
    sigmoid = Sigmoid()
    linear2 = Linear(4, 2)
    loss_fn = MSELoss()
    
    # Data
    x = np.array([1.0, 2.0, 3.0])
    target = np.array([0.5, 0.5])
    
    # Forward pass
    z1 = linear1.forward(x)
    a1 = sigmoid.forward(z1)
    z2 = linear2.forward(a1)
    loss = loss_fn.forward(z2, target)
    
    print(f"Forward pass:")
    print(f"  x → z1 → a1 → z2 → Loss")
    print(f"  {x.shape} → {z1.shape} → {a1.shape} → {z2.shape} → scalar")
    print(f"  Loss = {loss:.4f}")
    
    # Backward pass
    grad_z2 = loss_fn.backward()
    grad_a1 = linear2.backward(grad_z2)
    grad_z1 = sigmoid.backward(grad_a1)
    grad_x = linear1.backward(grad_z1)
    
    print(f"\nBackward pass:")
    print(f"  ∂L/∂z2 = {grad_z2.round(4)}")
    print(f"  ∂L/∂W2 shape: {linear2.grad_W.shape}")
    print(f"  ∂L/∂W1 shape: {linear1.grad_W.shape}")


if __name__ == "__main__":
    example_scalar_chain_rule()
    example_vector_chain_rule()
    example_simple_backprop()
    example_two_layer_network()
    example_softmax_crossentropy()
    example_gradient_checking()
    example_batch_gradient()
    example_computational_graph()
    example_vanishing_gradient()
    example_layer_class()
