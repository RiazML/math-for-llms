"""
Activation Functions - Exercises
================================

Practice exercises for understanding and implementing activation functions.
"""

import numpy as np
from typing import Tuple, Callable, Dict, List, Optional


# =============================================================================
# Exercise 1: Implement All ReLU Variants
# =============================================================================

def exercise_relu_variants():
    """
    Exercise: Implement ReLU, Leaky ReLU, PReLU, ELU, and SELU from scratch.
    Each should have both forward and backward (gradient) functions.
    
    Tasks:
    1. Implement ReLU with gradient
    2. Implement Leaky ReLU with configurable slope
    3. Implement PReLU with learnable alpha and its gradient
    4. Implement ELU with gradient
    5. Implement SELU with the exact constants
    6. Test on sample data and verify gradients
    """
    print("=" * 70)
    print("Exercise 1: ReLU Variants Implementation")
    print("=" * 70)
    
    # YOUR CODE HERE
    # Implement the following functions:
    
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU: max(0, x)"""
        pass
    
    def relu_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of ReLU"""
        pass
    
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU: x if x > 0 else alpha * x"""
        pass
    
    def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Gradient of Leaky ReLU"""
        pass
    
    class PReLU:
        """Parametric ReLU with learnable alpha."""
        def __init__(self, alpha: float = 0.25):
            self.alpha = alpha
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass"""
            pass
        
        def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
            """Backward pass returning (grad_x, grad_alpha)"""
            pass
    
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU: x if x > 0 else alpha * (exp(x) - 1)"""
        pass
    
    def elu_grad(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Gradient of ELU"""
        pass
    
    def selu(x: np.ndarray) -> np.ndarray:
        """SELU with self-normalizing constants"""
        pass
    
    def selu_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of SELU"""
        pass
    
    # Test your implementations
    x_test = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    # print("Testing implementations...")
    # print(f"x: {x_test}")
    # print(f"ReLU: {relu(x_test)}")
    # print(f"Leaky ReLU (0.1): {leaky_relu(x_test, 0.1)}")
    # print(f"ELU: {elu(x_test)}")
    # print(f"SELU: {selu(x_test)}")


def solution_relu_variants():
    """Reference solution for ReLU variants."""
    print("\n--- Solution ---\n")
    
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)
    
    class PReLU:
        def __init__(self, alpha: float = 0.25):
            self.alpha = alpha
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            self.x = x
            self.mask = x > 0
            return np.where(self.mask, x, self.alpha * x)
        
        def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
            grad_x = np.where(x > 0, grad_output, self.alpha * grad_output)
            grad_alpha = np.sum(grad_output * np.where(x > 0, 0, x))
            return grad_x, grad_alpha
    
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_grad(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    def selu(x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def selu_grad(x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, 1.0, alpha * np.exp(x))
    
    # Test
    x_test = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    print(f"x: {x_test}")
    print(f"ReLU: {relu(x_test)}")
    print(f"ReLU grad: {relu_grad(x_test)}")
    print(f"Leaky ReLU (α=0.1): {np.round(leaky_relu(x_test, 0.1), 4)}")
    print(f"Leaky ReLU grad: {np.round(leaky_relu_grad(x_test, 0.1), 4)}")
    print(f"ELU: {np.round(elu(x_test), 4)}")
    print(f"ELU grad: {np.round(elu_grad(x_test), 4)}")
    print(f"SELU: {np.round(selu(x_test), 4)}")
    print(f"SELU grad: {np.round(selu_grad(x_test), 4)}")
    
    # Verify PReLU gradients numerically
    print("\nPReLU gradient verification:")
    prelu = PReLU(alpha=0.25)
    x = np.array([-1.0, 0.5, -0.5, 1.0])
    output = prelu.forward(x)
    grad_output = np.ones_like(x)
    grad_x, grad_alpha = prelu.backward(grad_output, x)
    print(f"  x: {x}")
    print(f"  output: {output}")
    print(f"  grad_x: {grad_x}")
    print(f"  grad_alpha: {grad_alpha}")


# =============================================================================
# Exercise 2: GELU Implementation
# =============================================================================

def exercise_gelu():
    """
    Exercise: Implement GELU activation function.
    
    Tasks:
    1. Implement exact GELU using error function
    2. Implement GELU approximation using tanh
    3. Implement the derivative
    4. Compare accuracy of approximation
    5. Show that GELU can be interpreted as expected dropout
    """
    print("\n" + "=" * 70)
    print("Exercise 2: GELU Implementation")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def gelu_exact(x: np.ndarray) -> np.ndarray:
        """
        Exact GELU: x * Φ(x) where Φ is the CDF of standard normal.
        Hint: Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
        """
        pass
    
    def gelu_approx(x: np.ndarray) -> np.ndarray:
        """
        GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        """
        pass
    
    def gelu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of GELU (approximation)."""
        pass
    
    # Test and compare
    x_test = np.linspace(-3, 3, 7)
    # print(f"x: {x_test}")
    # print(f"GELU exact: {gelu_exact(x_test)}")
    # print(f"GELU approx: {gelu_approx(x_test)}")


def solution_gelu():
    """Reference solution for GELU."""
    print("\n--- Solution ---\n")
    
    from scipy.special import erf
    
    def gelu_exact(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    
    def gelu_approx(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def gelu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of GELU using approximation."""
        c = np.sqrt(2/np.pi)
        inner = c * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        sech2 = 1 - tanh_inner**2
        inner_deriv = c * (1 + 3 * 0.044715 * x**2)
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv
    
    # Test
    x_test = np.linspace(-3, 3, 7)
    
    print(f"x: {np.round(x_test, 2)}")
    print(f"GELU exact: {np.round(gelu_exact(x_test), 4)}")
    print(f"GELU approx: {np.round(gelu_approx(x_test), 4)}")
    print(f"GELU derivative: {np.round(gelu_derivative(x_test), 4)}")
    
    # Approximation error
    x_dense = np.linspace(-5, 5, 1000)
    error = np.abs(gelu_exact(x_dense) - gelu_approx(x_dense))
    print(f"\nApproximation error - max: {np.max(error):.6f}, mean: {np.mean(error):.6f}")
    
    # GELU as expected dropout
    print("\nGELU as Expected Dropout:")
    print("  GELU(x) = x * P(X ≤ x) where X ~ N(0,1)")
    print("  Interpretation: x is kept with probability Φ(x)")
    print("  Larger x → more likely to be kept")
    print("  Smaller/negative x → more likely to be zeroed")
    
    x_probs = np.array([-2, -1, 0, 1, 2])
    keep_probs = 0.5 * (1 + erf(x_probs / np.sqrt(2)))
    print(f"\n  x values: {x_probs}")
    print(f"  Keep prob: {np.round(keep_probs, 4)}")


# =============================================================================
# Exercise 3: Softmax with Numerical Stability
# =============================================================================

def exercise_softmax():
    """
    Exercise: Implement numerically stable softmax and its Jacobian.
    
    Tasks:
    1. Implement stable softmax (subtract max)
    2. Implement log-softmax directly (more stable)
    3. Compute the full Jacobian matrix
    4. Implement softmax with temperature
    5. Show gradient for cross-entropy loss simplifies to (p - y)
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Softmax Implementation")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        pass
    
    def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable log-softmax."""
        pass
    
    def softmax_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Compute full Jacobian of softmax.
        J[i,j] = ∂softmax_i/∂x_j = s_i(δ_ij - s_j)
        """
        pass
    
    def softmax_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax with temperature scaling."""
        pass
    
    # Test
    logits = np.array([2.0, 1.0, 0.1])
    # print(f"Softmax: {softmax(logits)}")
    # print(f"Sum: {np.sum(softmax(logits))}")


def solution_softmax():
    """Reference solution for softmax."""
    print("\n--- Solution ---\n")
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
        return x_shifted - log_sum_exp
    
    def softmax_jacobian(x: np.ndarray) -> np.ndarray:
        s = softmax(x)
        n = len(s)
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i, j] = s[i] * (1 - s[i])
                else:
                    jacobian[i, j] = -s[i] * s[j]
        return jacobian
    
    def softmax_temperature(x: np.ndarray, temperature: float) -> np.ndarray:
        return softmax(x / temperature)
    
    # Test
    logits = np.array([2.0, 1.0, 0.1])
    
    print(f"Logits: {logits}")
    print(f"Softmax: {np.round(softmax(logits), 4)}")
    print(f"Sum: {np.sum(softmax(logits)):.6f}")
    print(f"Log-softmax: {np.round(log_softmax(logits), 4)}")
    print(f"Exp(log-softmax): {np.round(np.exp(log_softmax(logits)), 4)}")
    
    print("\nJacobian:")
    J = softmax_jacobian(logits)
    for row in J:
        print(f"  {np.round(row, 4)}")
    
    print("\nTemperature effects:")
    for temp in [0.5, 1.0, 2.0, 5.0]:
        probs = softmax_temperature(logits, temp)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        print(f"  T={temp}: {np.round(probs, 3)}, entropy={entropy:.3f}")
    
    # Cross-entropy gradient simplification
    print("\nCross-entropy gradient simplification:")
    print("  L = -Σ y_i log(p_i)  where p = softmax(z)")
    print("  ∂L/∂z_j = Σ_i ∂L/∂p_i * ∂p_i/∂z_j")
    print("          = Σ_i (-y_i/p_i) * p_i(δ_ij - p_j)")
    print("          = -y_j + p_j * Σ_i y_i")
    print("          = p_j - y_j  (since Σ y_i = 1)")


# =============================================================================
# Exercise 4: Gated Activations (SwiGLU)
# =============================================================================

def exercise_gated_activations():
    """
    Exercise: Implement gated activation units used in modern transformers.
    
    Tasks:
    1. Implement GLU (Gated Linear Unit)
    2. Implement GEGLU (GELU-gated)
    3. Implement SwiGLU (Swish-gated)
    4. Implement a feed-forward network with gated activation
    5. Compare parameter counts and expressivity
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Gated Activations")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def glu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """GLU: x_a * sigmoid(x_b)"""
        pass
    
    def geglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """GEGLU: x_a * gelu(x_b)"""
        pass
    
    def swiglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """SwiGLU: x_a * swish(x_b)"""
        pass
    
    class GatedFFN:
        """Feed-forward with gated activation."""
        def __init__(self, d_model: int, d_ff: int, gate_type: str = 'swiglu'):
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass


def solution_gated_activations():
    """Reference solution for gated activations."""
    print("\n--- Solution ---\n")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def swish(x):
        return x * sigmoid(x)
    
    def glu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * sigmoid(x_b)
    
    def geglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * gelu(x_b)
    
    def swiglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * swish(x_b)
    
    class GatedFFN:
        def __init__(self, d_model: int, d_ff: int, gate_type: str = 'swiglu', seed: int = 42):
            np.random.seed(seed)
            self.gate_type = gate_type
            scale = np.sqrt(2.0 / d_model)
            
            # Note: for gated activations, we project to 2*d_ff
            self.W1 = np.random.randn(d_model, d_ff * 2) * scale
            self.W2 = np.random.randn(d_ff, d_model) * scale
            
            self.gate_fn = {'glu': glu, 'geglu': geglu, 'swiglu': swiglu}[gate_type]
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            # Up projection with gate
            hidden = x @ self.W1
            # Apply gated activation
            gated = self.gate_fn(hidden)
            # Down projection
            return gated @ self.W2
    
    # Test
    x = np.array([1.0, 0.5, -0.5, 2.0, 1.5, -1.0])
    
    print(f"Input (6D): {x}")
    print(f"GLU output (3D): {np.round(glu(x), 4)}")
    print(f"GEGLU output: {np.round(geglu(x), 4)}")
    print(f"SwiGLU output: {np.round(swiglu(x), 4)}")
    
    # FFN comparison
    d_model, d_ff = 512, 2048
    
    print(f"\nGated FFN (d_model={d_model}, d_ff={d_ff}):")
    ffn = GatedFFN(d_model, d_ff, 'swiglu')
    
    x_batch = np.random.randn(2, 8, d_model)
    output = ffn.forward(x_batch)
    
    print(f"  Input shape: {x_batch.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Parameter count comparison
    standard_params = d_model * d_ff + d_ff * d_model  # W1 + W2
    gated_params = d_model * 2 * d_ff + d_ff * d_model  # W1 (2x) + W2
    
    print(f"\nParameter comparison:")
    print(f"  Standard FFN (ReLU): {standard_params:,} params")
    print(f"  Gated FFN (SwiGLU): {gated_params:,} params")
    print(f"  Overhead: {100*(gated_params/standard_params - 1):.1f}%")
    print("  Note: Gated FFNs typically use smaller d_ff to match param count")


# =============================================================================
# Exercise 5: Dying ReLU Analysis
# =============================================================================

def exercise_dying_relu():
    """
    Exercise: Analyze and mitigate the dying ReLU problem.
    
    Tasks:
    1. Simulate a network and count dead neurons
    2. Analyze how initialization affects dead neurons
    3. Compare dead neuron rates for ReLU variants
    4. Implement a resurrection strategy
    5. Show how batch normalization helps
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Dying ReLU Analysis")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def count_dead_neurons(W: np.ndarray, b: np.ndarray, 
                           X: np.ndarray, activation: str = 'relu') -> float:
        """
        Count fraction of neurons that are always inactive (dead).
        A neuron is dead if it outputs 0 for all inputs in X.
        """
        pass
    
    def analyze_initialization(input_dim: int, hidden_dim: int,
                               n_samples: int = 1000) -> dict:
        """
        Analyze dead neuron rate for different initialization schemes.
        """
        pass


def solution_dying_relu():
    """Reference solution for dying ReLU analysis."""
    print("\n--- Solution ---\n")
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def count_dead_neurons(W: np.ndarray, b: np.ndarray, 
                           X: np.ndarray, activation: str = 'relu') -> float:
        """Count fraction of neurons that are always inactive."""
        # Compute pre-activation
        z = X @ W + b  # (n_samples, hidden_dim)
        
        # Apply activation
        if activation == 'relu':
            a = relu(z)
        elif activation == 'leaky_relu':
            a = leaky_relu(z)
        elif activation == 'elu':
            a = elu(z)
        else:
            a = z
        
        # Count neurons that are always zero/inactive
        always_zero = np.all(a <= 0, axis=0)
        return np.mean(always_zero)
    
    def analyze_initialization(input_dim: int, hidden_dim: int,
                               n_samples: int = 1000, seed: int = 42) -> dict:
        """Analyze dead neuron rate for different initialization schemes."""
        np.random.seed(seed)
        
        # Generate data (standard normal)
        X = np.random.randn(n_samples, input_dim)
        
        results = {}
        
        # Different initializations
        inits = {
            'small': 0.01,
            'xavier': np.sqrt(2.0 / (input_dim + hidden_dim)),
            'he': np.sqrt(2.0 / input_dim),
            'large': 1.0
        }
        
        for name, scale in inits.items():
            W = np.random.randn(input_dim, hidden_dim) * scale
            b = np.zeros(hidden_dim)
            
            dead_rate = count_dead_neurons(W, b, X, 'relu')
            results[name] = dead_rate
        
        return results
    
    # Run analysis
    print("Dead Neuron Rate by Initialization (input=100, hidden=100):")
    results = analyze_initialization(100, 100)
    for name, rate in results.items():
        print(f"  {name:10s}: {rate:.2%}")
    
    # Compare activations
    print("\nDead/Inactive Rate by Activation (He init):")
    np.random.seed(42)
    input_dim, hidden_dim = 100, 100
    X = np.random.randn(1000, input_dim)
    W = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b = np.zeros(hidden_dim)
    
    for act_name in ['relu', 'leaky_relu', 'elu']:
        rate = count_dead_neurons(W, b, X, act_name)
        print(f"  {act_name:12s}: {rate:.2%}")
    
    # Effect of bias initialization
    print("\nEffect of Bias Initialization (ReLU):")
    for bias_init in [0.0, 0.1, 0.5, 1.0]:
        b = np.ones(hidden_dim) * bias_init
        rate = count_dead_neurons(W, b, X, 'relu')
        print(f"  bias={bias_init}: {rate:.2%}")
    
    # Batch normalization effect
    print("\nBatch Normalization Effect:")
    print("  Pre-BN dead rate: ~50% (for balanced data)")
    z = X @ W  # Pre-activation
    z_bn = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-8)
    z_bn_shifted = z_bn * 1.0 + 0.0  # gamma=1, beta=0
    
    pre_bn_dead = np.mean(np.all(relu(z) <= 0, axis=0))
    post_bn_dead = np.mean(np.all(relu(z_bn_shifted) <= 0, axis=0))
    
    print(f"  Without BN: {pre_bn_dead:.2%} dead neurons")
    print(f"  With BN:    {post_bn_dead:.2%} dead neurons")


# =============================================================================
# Exercise 6: Activation Function Properties
# =============================================================================

def exercise_activation_properties():
    """
    Exercise: Analyze mathematical properties of activation functions.
    
    Tasks:
    1. Compute Lipschitz constants
    2. Check monotonicity
    3. Analyze zero-centeredness
    4. Compute mean activation for standard normal input
    5. Analyze saturation regions
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Activation Function Properties")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def compute_lipschitz(activation: Callable, x_range: Tuple[float, float] = (-10, 10),
                          n_points: int = 10000) -> float:
        """Compute the Lipschitz constant (max |f'(x)|)."""
        pass
    
    def check_monotonicity(activation: Callable, x_range: Tuple[float, float] = (-10, 10),
                           n_points: int = 10000) -> bool:
        """Check if activation is monotonically increasing."""
        pass
    
    def compute_mean_std(activation: Callable, n_samples: int = 100000) -> Tuple[float, float]:
        """Compute mean and std of activation for N(0,1) input."""
        pass


def solution_activation_properties():
    """Reference solution for activation properties."""
    print("\n--- Solution ---\n")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def swish(x):
        return x * sigmoid(x)
    
    activations = {
        'sigmoid': sigmoid,
        'tanh': np.tanh,
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
        'gelu': gelu,
        'swish': swish,
    }
    
    def compute_lipschitz(activation: Callable, x_range: Tuple[float, float] = (-10, 10),
                          n_points: int = 10000) -> float:
        x = np.linspace(x_range[0], x_range[1], n_points)
        dx = x[1] - x[0]
        y = activation(x)
        derivatives = np.abs(np.diff(y) / dx)
        return np.max(derivatives)
    
    def check_monotonicity(activation: Callable, x_range: Tuple[float, float] = (-10, 10),
                           n_points: int = 10000) -> bool:
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = activation(x)
        return np.all(np.diff(y) >= -1e-10)  # Small tolerance for numerical error
    
    def compute_mean_std(activation: Callable, n_samples: int = 100000, 
                         seed: int = 42) -> Tuple[float, float]:
        np.random.seed(seed)
        x = np.random.randn(n_samples)
        y = activation(x)
        return np.mean(y), np.std(y)
    
    def check_zero_centered(activation: Callable, n_samples: int = 100000) -> float:
        """Return mean activation for N(0,1) input."""
        mean, _ = compute_mean_std(activation, n_samples)
        return mean
    
    # Analyze all activations
    print("Activation Function Properties:")
    print("-" * 80)
    print(f"{'Name':12s} | {'Lipschitz':10s} | {'Monotonic':10s} | {'Mean(N(0,1))':12s} | {'Std(N(0,1))':12s}")
    print("-" * 80)
    
    for name, func in activations.items():
        lip = compute_lipschitz(func)
        mono = "Yes" if check_monotonicity(func) else "No"
        mean, std = compute_mean_std(func)
        
        print(f"{name:12s} | {lip:10.4f} | {mono:10s} | {mean:12.4f} | {std:12.4f}")
    
    # Saturation analysis
    print("\nSaturation Analysis (|f'(x)| < 0.01):")
    x = np.linspace(-5, 5, 1000)
    dx = x[1] - x[0]
    
    for name, func in activations.items():
        y = func(x)
        derivs = np.abs(np.gradient(y, dx))
        saturated = np.mean(derivs < 0.01)
        print(f"  {name:12s}: {saturated:.1%} of range [-5, 5] is saturated")
    
    # Non-monotonic activations
    print("\nNon-monotonic Region Analysis:")
    for name in ['gelu', 'swish']:
        func = activations[name]
        y = func(x)
        non_mono_mask = np.diff(y) < 0
        non_mono_x = x[:-1][non_mono_mask]
        if len(non_mono_x) > 0:
            print(f"  {name}: non-monotonic in [{non_mono_x[0]:.2f}, {non_mono_x[-1]:.2f}]")
        else:
            print(f"  {name}: monotonic in [-5, 5]")


# =============================================================================
# Exercise 7: Neural Tangent Kernel Activation
# =============================================================================

def exercise_ntk_activation():
    """
    Exercise: Explore the Neural Tangent Kernel perspective on activations.
    
    Tasks:
    1. Compute the dual activation (for NTK kernel)
    2. Show that ReLU's dual is arc-cosine kernel
    3. Compute expected gradients for different activations
    4. Analyze infinite-width limit behavior
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Neural Tangent Kernel Activation")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def arc_cosine_kernel(x1: np.ndarray, x2: np.ndarray, degree: int = 1) -> float:
        """
        Arc-cosine kernel (NTK for ReLU).
        K_n(x, y) = (1/π) * ||x|| * ||y|| * J_n(θ)
        where θ = arccos(x·y / (||x|| * ||y||))
        """
        pass
    
    def relu_ntk_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """Neural Tangent Kernel for ReLU activation."""
        pass


def solution_ntk_activation():
    """Reference solution for NTK activation analysis."""
    print("\n--- Solution ---\n")
    
    def arc_cosine_kernel(x1: np.ndarray, x2: np.ndarray, degree: int = 1) -> float:
        """Arc-cosine kernel K_n(x, y) for ReLU networks."""
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        cos_theta = np.clip(np.dot(x1, x2) / (norm1 * norm2), -1, 1)
        theta = np.arccos(cos_theta)
        
        if degree == 0:
            # K_0: step function kernel
            return 1 - theta / np.pi
        elif degree == 1:
            # K_1: ReLU kernel
            J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            return (norm1 * norm2 / np.pi) * J
        elif degree == 2:
            # K_2: ReLU^2 kernel
            J = 3 * np.sin(theta) * np.cos(theta) + (np.pi - theta) * (1 + 2 * np.cos(theta)**2)
            return (norm1**2 * norm2**2 / np.pi) * J
        else:
            raise ValueError(f"Degree {degree} not implemented")
    
    def expected_relu_derivative_product(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        E[ReLU'(w·x1) * ReLU'(w·x2)] for w ~ N(0, I)
        This equals (π - θ) / (2π) where θ = angle between x1 and x2.
        """
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        cos_theta = np.clip(np.dot(x1, x2) / (norm1 * norm2), -1, 1)
        theta = np.arccos(cos_theta)
        
        return (np.pi - theta) / (2 * np.pi)
    
    # Test arc-cosine kernel
    print("Arc-Cosine Kernel (ReLU NTK):")
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.0, 1.0, 0.0])
    x3 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    
    print(f"  x1 = {x1}")
    print(f"  x2 = {x2} (orthogonal to x1)")
    print(f"  x3 = {np.round(x3, 3)} (45° from x1)")
    
    print(f"\n  K_1(x1, x1) = {arc_cosine_kernel(x1, x1, 1):.4f} (self-similarity)")
    print(f"  K_1(x1, x2) = {arc_cosine_kernel(x1, x2, 1):.4f} (orthogonal)")
    print(f"  K_1(x1, x3) = {arc_cosine_kernel(x1, x3, 1):.4f} (45°)")
    
    # Different degrees
    print("\nKernel by Degree for x1 and x3:")
    for degree in [0, 1, 2]:
        k = arc_cosine_kernel(x1, x3, degree)
        print(f"  K_{degree}(x1, x3) = {k:.4f}")
    
    # Expected derivative products
    print("\nExpected Derivative Products E[σ'(w·x1) * σ'(w·x2)]:")
    print(f"  x1, x1 (θ=0°):   {expected_relu_derivative_product(x1, x1):.4f}")
    print(f"  x1, x2 (θ=90°):  {expected_relu_derivative_product(x1, x2):.4f}")
    print(f"  x1, x3 (θ=45°):  {expected_relu_derivative_product(x1, x3):.4f}")
    
    # Monte Carlo verification
    print("\nMonte Carlo Verification:")
    np.random.seed(42)
    n_samples = 100000
    
    def relu_deriv(x):
        return (x > 0).astype(float)
    
    # Sample random weights
    W = np.random.randn(n_samples, 3)
    
    # Compute empirical expectation
    z1 = W @ x1
    z3 = W @ x3
    empirical = np.mean(relu_deriv(z1) * relu_deriv(z3))
    theoretical = expected_relu_derivative_product(x1, x3)
    
    print(f"  E[ReLU'(w·x1) * ReLU'(w·x3)]:")
    print(f"    Theoretical: {theoretical:.4f}")
    print(f"    Empirical:   {empirical:.4f}")


# =============================================================================
# Exercise 8: Custom Activation Design
# =============================================================================

def exercise_custom_activation():
    """
    Exercise: Design and evaluate a custom activation function.
    
    Tasks:
    1. Design an activation with specific properties
    2. Implement forward and backward passes
    3. Verify gradient correctness numerically
    4. Test in a simple network
    5. Compare against standard activations
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Custom Activation Design")
    print("=" * 70)
    
    # YOUR CODE HERE
    # Design requirements:
    # - Zero-centered (approximately)
    # - Non-saturating for large positive inputs
    # - Smooth and differentiable everywhere
    # - Computationally efficient
    
    class CustomActivation:
        """Your custom activation function."""
        
        @staticmethod
        def forward(x: np.ndarray) -> np.ndarray:
            pass
        
        @staticmethod
        def backward(x: np.ndarray) -> np.ndarray:
            pass


def solution_custom_activation():
    """Reference solution for custom activation design."""
    print("\n--- Solution ---\n")
    
    class CustomActivation:
        """
        Custom activation: combination of good properties.
        f(x) = x * sigmoid(x) - 0.17 (Swish shifted to be ~zero-centered)
        or
        f(x) = x * tanh(softplus(x)) - c (Mish shifted)
        
        Here we implement a "Shifted Swish" for simplicity.
        """
        
        @staticmethod
        def sigmoid(x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        @staticmethod
        def forward(x: np.ndarray) -> np.ndarray:
            # Swish shifted to be approximately zero-centered
            return x * CustomActivation.sigmoid(x) - 0.17
        
        @staticmethod
        def backward(x: np.ndarray) -> np.ndarray:
            sig = CustomActivation.sigmoid(x)
            return sig + x * sig * (1 - sig)
    
    def verify_gradient(activation_class, x: np.ndarray, eps: float = 1e-5) -> float:
        """Verify gradient using finite differences."""
        numerical_grad = (activation_class.forward(x + eps) - 
                         activation_class.forward(x - eps)) / (2 * eps)
        analytical_grad = activation_class.backward(x)
        return np.max(np.abs(numerical_grad - analytical_grad))
    
    # Test custom activation
    act = CustomActivation()
    x_test = np.linspace(-3, 3, 7)
    
    print("Custom Activation (Shifted Swish):")
    print(f"  x: {np.round(x_test, 2)}")
    print(f"  f(x): {np.round(act.forward(x_test), 4)}")
    print(f"  f'(x): {np.round(act.backward(x_test), 4)}")
    
    # Verify gradients
    x_verify = np.random.randn(100)
    max_error = verify_gradient(CustomActivation, x_verify)
    print(f"\n  Gradient verification error: {max_error:.2e}")
    
    # Properties
    np.random.seed(42)
    samples = np.random.randn(100000)
    outputs = act.forward(samples)
    
    print(f"\nProperties for N(0,1) input:")
    print(f"  Mean output: {np.mean(outputs):.4f} (target: ~0)")
    print(f"  Std output: {np.std(outputs):.4f}")
    
    # Compare in a simple network
    print("\nComparison in 2-layer network (XOR problem):")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train_simple_network(activation_fn, activation_grad, n_epochs=100, lr=0.5):
        np.random.seed(42)
        
        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        
        # Initialize
        W1 = np.random.randn(2, 4) * 0.5
        b1 = np.zeros(4)
        W2 = np.random.randn(4, 1) * 0.5
        b2 = np.zeros(1)
        
        for epoch in range(n_epochs):
            # Forward
            z1 = X @ W1 + b1
            a1 = activation_fn(z1)
            z2 = a1 @ W2 + b2
            pred = sigmoid(z2)
            
            # Backward
            d2 = pred - y
            dW2 = a1.T @ d2 / 4
            db2 = np.mean(d2, axis=0)
            
            d1 = (d2 @ W2.T) * activation_grad(z1)
            dW1 = X.T @ d1 / 4
            db1 = np.mean(d1, axis=0)
            
            # Update
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1
        
        # Final prediction
        z1 = X @ W1 + b1
        a1 = activation_fn(z1)
        z2 = a1 @ W2 + b2
        pred = sigmoid(z2)
        
        return np.round(pred.flatten(), 2)
    
    # Test different activations
    relu_pred = train_simple_network(
        lambda x: np.maximum(0, x),
        lambda x: (x > 0).astype(float)
    )
    
    custom_pred = train_simple_network(
        CustomActivation.forward,
        CustomActivation.backward
    )
    
    print(f"  Target: [0, 1, 1, 0]")
    print(f"  ReLU predictions: {relu_pred}")
    print(f"  Custom predictions: {custom_pred}")


# =============================================================================
# Exercise 9: Activation Visualization
# =============================================================================

def exercise_activation_visualization():
    """
    Exercise: Create visualization data for activation functions.
    
    Tasks:
    1. Generate activation curves for all major functions
    2. Generate derivative curves
    3. Compute statistics over different input distributions
    4. Create comparison tables
    """
    print("\n" + "=" * 70)
    print("Exercise 9: Activation Visualization Data")
    print("=" * 70)
    
    # YOUR CODE HERE
    pass


def solution_activation_visualization():
    """Reference solution for activation visualization."""
    print("\n--- Solution ---\n")
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def swish(x):
        return x * sigmoid(x)
    
    def mish(x):
        return x * np.tanh(np.log1p(np.exp(np.clip(x, -500, 20))))
    
    activations = {
        'sigmoid': (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
        'tanh': (np.tanh, lambda x: 1 - np.tanh(x)**2),
        'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
        'leaky_relu': (lambda x: np.where(x > 0, x, 0.01*x), lambda x: np.where(x > 0, 1., 0.01)),
        'elu': (lambda x: np.where(x > 0, x, np.exp(x)-1), lambda x: np.where(x > 0, 1., np.exp(x))),
        'gelu': (gelu, None),
        'swish': (swish, None),
        'mish': (mish, None),
    }
    
    # Generate curve data
    x = np.linspace(-4, 4, 100)
    
    print("Activation Function Values (sampled):")
    print("-" * 70)
    sample_points = [-3, -1, 0, 1, 3]
    
    header = f"{'Activation':12s}"
    for p in sample_points:
        header += f" | x={p:2d}"
    print(header)
    print("-" * 70)
    
    for name, (func, _) in activations.items():
        row = f"{name:12s}"
        for p in sample_points:
            val = func(np.array([p]))[0]
            row += f" | {val:5.2f}"
        print(row)
    
    # Statistics for different input distributions
    print("\nOutput Statistics by Input Distribution:")
    print("-" * 70)
    
    distributions = {
        'N(0, 1)': lambda: np.random.randn(10000),
        'N(0, 2)': lambda: np.random.randn(10000) * 2,
        'U[-3, 3]': lambda: np.random.uniform(-3, 3, 10000),
    }
    
    np.random.seed(42)
    
    for dist_name, sampler in distributions.items():
        print(f"\n{dist_name}:")
        print(f"  {'Activation':12s} | {'Mean':8s} | {'Std':8s} | {'Min':8s} | {'Max':8s}")
        print("  " + "-" * 52)
        
        samples = sampler()
        
        for act_name, (func, _) in activations.items():
            outputs = func(samples)
            print(f"  {act_name:12s} | {np.mean(outputs):8.4f} | {np.std(outputs):8.4f} | "
                  f"{np.min(outputs):8.4f} | {np.max(outputs):8.4f}")
    
    # Recommendation summary
    print("\n" + "=" * 70)
    print("Activation Function Recommendations")
    print("=" * 70)
    print("""
    | Use Case                  | Recommended Activation |
    |---------------------------|------------------------|
    | General hidden layers     | ReLU, GELU             |
    | Transformer FFN           | GELU, SwiGLU           |
    | CNNs                      | ReLU, LeakyReLU        |
    | RNN/LSTM gates            | Sigmoid, Tanh          |
    | Self-normalizing networks | SELU                   |
    | GANs                      | LeakyReLU              |
    | Mobile/efficient          | ReLU, Hard Swish       |
    | Binary classification     | Sigmoid (output)       |
    | Multi-class classification| Softmax (output)       |
    | Regression                | None (linear output)   |
    """)


# =============================================================================
# Exercise 10: Complete Activation Module
# =============================================================================

def exercise_activation_module():
    """
    Exercise: Build a complete activation function module.
    
    Tasks:
    1. Create a unified Activation class
    2. Support all major activation functions
    3. Include forward and backward methods
    4. Add gradient checking utility
    5. Support learnable parameters (PReLU)
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Activation Module")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class Activation:
        """Unified activation function module."""
        
        def __init__(self, name: str, **kwargs):
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass
        
        def backward(self, grad_output: np.ndarray) -> np.ndarray:
            pass
        
        @staticmethod
        def check_gradient(activation, x: np.ndarray, eps: float = 1e-5) -> float:
            pass


def solution_activation_module():
    """Reference solution for complete activation module."""
    print("\n--- Solution ---\n")
    
    class Activation:
        """Unified activation function module with forward/backward."""
        
        SUPPORTED = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'prelu', 'elu', 
                     'selu', 'gelu', 'swish', 'mish', 'softplus', 'softmax']
        
        def __init__(self, name: str, **kwargs):
            if name not in self.SUPPORTED:
                raise ValueError(f"Unknown activation: {name}. Supported: {self.SUPPORTED}")
            
            self.name = name
            self.kwargs = kwargs
            self.cache = {}
            
            # Learnable parameters
            if name == 'prelu':
                self.alpha = kwargs.get('alpha', 0.25)
                self.requires_grad = True
            else:
                self.requires_grad = False
        
        def _sigmoid(self, x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass, stores cache for backward."""
            self.cache['x'] = x
            
            if self.name == 'sigmoid':
                out = self._sigmoid(x)
                self.cache['out'] = out
                return out
            
            elif self.name == 'tanh':
                out = np.tanh(x)
                self.cache['out'] = out
                return out
            
            elif self.name == 'relu':
                return np.maximum(0, x)
            
            elif self.name == 'leaky_relu':
                alpha = self.kwargs.get('alpha', 0.01)
                return np.where(x > 0, x, alpha * x)
            
            elif self.name == 'prelu':
                mask = x > 0
                self.cache['mask'] = mask
                return np.where(mask, x, self.alpha * x)
            
            elif self.name == 'elu':
                alpha = self.kwargs.get('alpha', 1.0)
                return np.where(x > 0, x, alpha * (np.exp(x) - 1))
            
            elif self.name == 'selu':
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
            
            elif self.name == 'gelu':
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
            
            elif self.name == 'swish':
                sig = self._sigmoid(x)
                self.cache['sig'] = sig
                return x * sig
            
            elif self.name == 'mish':
                sp = np.log1p(np.exp(np.clip(x, -500, 20)))
                tanh_sp = np.tanh(sp)
                self.cache['tanh_sp'] = tanh_sp
                self.cache['sp'] = sp
                return x * tanh_sp
            
            elif self.name == 'softplus':
                return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
            
            elif self.name == 'softmax':
                axis = self.kwargs.get('axis', -1)
                x_shift = x - np.max(x, axis=axis, keepdims=True)
                exp_x = np.exp(x_shift)
                out = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
                self.cache['out'] = out
                return out
        
        def backward(self, grad_output: np.ndarray) -> np.ndarray:
            """Backward pass using cached values."""
            x = self.cache.get('x')
            
            if self.name == 'sigmoid':
                out = self.cache['out']
                return grad_output * out * (1 - out)
            
            elif self.name == 'tanh':
                out = self.cache['out']
                return grad_output * (1 - out ** 2)
            
            elif self.name == 'relu':
                return grad_output * (x > 0).astype(float)
            
            elif self.name == 'leaky_relu':
                alpha = self.kwargs.get('alpha', 0.01)
                return grad_output * np.where(x > 0, 1.0, alpha)
            
            elif self.name == 'prelu':
                mask = self.cache['mask']
                grad_x = grad_output * np.where(mask, 1.0, self.alpha)
                self.grad_alpha = np.sum(grad_output * np.where(mask, 0, x))
                return grad_x
            
            elif self.name == 'elu':
                alpha = self.kwargs.get('alpha', 1.0)
                return grad_output * np.where(x > 0, 1.0, alpha * np.exp(x))
            
            elif self.name == 'selu':
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                return grad_output * scale * np.where(x > 0, 1.0, alpha * np.exp(x))
            
            elif self.name == 'gelu':
                c = np.sqrt(2/np.pi)
                inner = c * (x + 0.044715 * x**3)
                tanh_inner = np.tanh(inner)
                sech2 = 1 - tanh_inner**2
                inner_deriv = c * (1 + 3 * 0.044715 * x**2)
                return grad_output * (0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv)
            
            elif self.name == 'swish':
                sig = self.cache['sig']
                return grad_output * (sig + x * sig * (1 - sig))
            
            elif self.name == 'softmax':
                # Softmax backward is typically combined with cross-entropy
                out = self.cache['out']
                # For general gradient (not with CE loss)
                sum_term = np.sum(grad_output * out, axis=-1, keepdims=True)
                return out * (grad_output - sum_term)
            
            else:
                raise NotImplementedError(f"Backward not implemented for {self.name}")
        
        @staticmethod
        def check_gradient(activation: 'Activation', x: np.ndarray, 
                          eps: float = 1e-5) -> float:
            """Numerically verify gradient computation."""
            # Forward passes with perturbation
            x_plus = x + eps
            x_minus = x - eps
            
            y_plus = activation.forward(x_plus)
            y_minus = activation.forward(x_minus)
            numerical_grad = (y_plus - y_minus) / (2 * eps)
            
            # Analytical gradient
            activation.forward(x)
            analytical_grad = activation.backward(np.ones_like(x))
            
            return np.max(np.abs(numerical_grad - analytical_grad))
    
    # Test the module
    print("Testing Activation Module:")
    print("-" * 50)
    
    x_test = np.array([-2., -1., 0., 1., 2.])
    
    for act_name in ['relu', 'leaky_relu', 'elu', 'gelu', 'swish', 'sigmoid', 'tanh']:
        act = Activation(act_name)
        output = act.forward(x_test)
        grad = act.backward(np.ones_like(x_test))
        
        # Check gradient
        error = Activation.check_gradient(Activation(act_name), x_test)
        
        print(f"{act_name:12s} - output: {np.round(output, 3)}, grad_check: {error:.2e}")
    
    # Test PReLU with learnable parameter
    print("\nPReLU with learnable alpha:")
    prelu = Activation('prelu', alpha=0.25)
    x_prelu = np.array([-1., 0., 1.])
    out = prelu.forward(x_prelu)
    grad = prelu.backward(np.ones_like(x_prelu))
    print(f"  alpha: {prelu.alpha}")
    print(f"  output: {out}")
    print(f"  grad_x: {grad}")
    print(f"  grad_alpha: {prelu.grad_alpha}")
    
    # Test softmax
    print("\nSoftmax:")
    softmax = Activation('softmax')
    logits = np.array([[2., 1., 0.1], [1., 2., 1.]])
    probs = softmax.forward(logits)
    print(f"  logits: {logits}")
    print(f"  probs: {np.round(probs, 4)}")
    print(f"  row sums: {np.sum(probs, axis=1)}")


def main():
    """Run all exercises with solutions."""
    print("ACTIVATION FUNCTIONS - EXERCISES")
    print("=" * 70)
    
    # Exercise 1
    exercise_relu_variants()
    solution_relu_variants()
    
    # Exercise 2
    exercise_gelu()
    solution_gelu()
    
    # Exercise 3
    exercise_softmax()
    solution_softmax()
    
    # Exercise 4
    exercise_gated_activations()
    solution_gated_activations()
    
    # Exercise 5
    exercise_dying_relu()
    solution_dying_relu()
    
    # Exercise 6
    exercise_activation_properties()
    solution_activation_properties()
    
    # Exercise 7
    exercise_ntk_activation()
    solution_ntk_activation()
    
    # Exercise 8
    exercise_custom_activation()
    solution_custom_activation()
    
    # Exercise 9
    exercise_activation_visualization()
    solution_activation_visualization()
    
    # Exercise 10
    exercise_activation_module()
    solution_activation_module()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
