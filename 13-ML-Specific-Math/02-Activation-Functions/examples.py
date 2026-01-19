"""
Activation Functions - Examples
==============================

Comprehensive examples of activation functions for neural networks.
"""

import numpy as np
from typing import Tuple, Callable, Dict, List, Optional
import warnings


# =============================================================================
# Example 1: Classic Activation Functions
# =============================================================================

def example_classic_activations():
    """
    Implement sigmoid, tanh, and their derivatives.
    Analyze gradient flow properties.
    """
    print("=" * 70)
    print("Example 1: Classic Activation Functions")
    print("=" * 70)
    
    # Sigmoid activation
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid."""
        s = sigmoid(x)
        return s * (1 - s)
    
    # Tanh activation
    def tanh(x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent."""
        return np.tanh(x)
    
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh."""
        return 1 - np.tanh(x) ** 2
    
    # Analyze over range
    x = np.linspace(-5, 5, 100)
    
    print("\nSigmoid Analysis:")
    print(f"  σ(0) = {sigmoid(np.array([0]))[0]:.4f}")
    print(f"  σ'(0) = {sigmoid_derivative(np.array([0]))[0]:.4f} (maximum)")
    print(f"  σ(-5) = {sigmoid(np.array([-5]))[0]:.6f} (near saturation)")
    print(f"  σ'(-5) = {sigmoid_derivative(np.array([-5]))[0]:.6f} (vanishing)")
    
    print("\nTanh Analysis:")
    print(f"  tanh(0) = {tanh(np.array([0]))[0]:.4f}")
    print(f"  tanh'(0) = {tanh_derivative(np.array([0]))[0]:.4f} (maximum)")
    print(f"  tanh(-5) = {tanh(np.array([-5]))[0]:.6f} (near saturation)")
    print(f"  tanh'(-5) = {tanh_derivative(np.array([-5]))[0]:.6f} (vanishing)")
    
    # Compare gradient flow through multiple layers
    print("\nGradient Flow Through 5 Layers:")
    print("  (Product of derivatives at x=1)")
    
    sigmoid_grad_product = sigmoid_derivative(np.array([1])) ** 5
    tanh_grad_product = tanh_derivative(np.array([1])) ** 5
    
    print(f"  Sigmoid: {sigmoid_grad_product[0]:.6f}")
    print(f"  Tanh: {tanh_grad_product[0]:.6f}")
    
    # Show relationship between sigmoid and tanh
    print("\nRelationship: tanh(x) = 2σ(2x) - 1")
    test_x = np.array([0.5, 1.0, 2.0])
    tanh_direct = np.tanh(test_x)
    tanh_from_sigmoid = 2 * sigmoid(2 * test_x) - 1
    print(f"  tanh(x):        {tanh_direct}")
    print(f"  2σ(2x) - 1:     {tanh_from_sigmoid}")
    
    return sigmoid, tanh


# =============================================================================
# Example 2: ReLU Family
# =============================================================================

def example_relu_family():
    """
    Implement ReLU and its variants: Leaky ReLU, PReLU, ELU, SELU.
    """
    print("\n" + "=" * 70)
    print("Example 2: ReLU Family Activations")
    print("=" * 70)
    
    # Standard ReLU
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    # Leaky ReLU
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha)
    
    # Parametric ReLU (learnable alpha)
    class PReLU:
        def __init__(self, alpha: float = 0.25):
            self.alpha = alpha
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            self.mask = x > 0
            return np.where(self.mask, x, self.alpha * x)
        
        def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
            grad_x = np.where(self.mask, grad_output, self.alpha * grad_output)
            grad_alpha = np.sum(grad_output * np.where(self.mask, 0, x))
            return grad_x, grad_alpha
    
    # ELU (Exponential Linear Unit)
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, 1.0, alpha * np.exp(x))
    
    # SELU (Scaled ELU) - self-normalizing
    def selu(x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def selu_derivative(x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(x > 0, 1.0, alpha * np.exp(x))
    
    # Test activations
    x = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    
    print("\nActivation Values for x = [-2, -1, -0.5, 0, 0.5, 1, 2]:")
    print(f"  ReLU:       {relu(x)}")
    print(f"  LeakyReLU:  {np.round(leaky_relu(x), 4)}")
    print(f"  ELU:        {np.round(elu(x), 4)}")
    print(f"  SELU:       {np.round(selu(x), 4)}")
    
    print("\nDerivatives at x = [-2, -1, -0.5, 0, 0.5, 1, 2]:")
    print(f"  ReLU:       {relu_derivative(x)}")
    print(f"  LeakyReLU:  {np.round(leaky_relu_derivative(x), 4)}")
    print(f"  ELU:        {np.round(elu_derivative(x), 4)}")
    print(f"  SELU:       {np.round(selu_derivative(x), 4)}")
    
    # Demonstrate dying ReLU problem
    print("\nDying ReLU Demonstration:")
    np.random.seed(42)
    x_samples = np.random.randn(1000) - 1  # Shifted to have more negatives
    relu_outputs = relu(x_samples)
    active_fraction = np.mean(relu_outputs > 0)
    print(f"  Input mean: {np.mean(x_samples):.2f}")
    print(f"  Fraction of active ReLU neurons: {active_fraction:.2%}")
    print(f"  (Dead neurons: {1 - active_fraction:.2%})")
    
    return relu, leaky_relu, elu, selu


# =============================================================================
# Example 3: GELU and Swish
# =============================================================================

def example_gelu_swish():
    """
    Implement GELU and Swish activations used in modern transformers.
    """
    print("\n" + "=" * 70)
    print("Example 3: GELU and Swish Activations")
    print("=" * 70)
    
    # Error function approximation
    def erf(x: np.ndarray) -> np.ndarray:
        """Approximate error function."""
        # Using tanh approximation
        a = 8 / (3 * np.pi) * (np.pi - 3) / (4 - np.pi)
        x2 = x * x
        return np.sign(x) * np.sqrt(1 - np.exp(-x2 * (4/np.pi + a*x2) / (1 + a*x2)))
    
    # GELU - exact
    def gelu_exact(x: np.ndarray) -> np.ndarray:
        """GELU using scipy's error function for accuracy."""
        from scipy.special import erf as scipy_erf
        return 0.5 * x * (1 + scipy_erf(x / np.sqrt(2)))
    
    # GELU - tanh approximation (used in practice)
    def gelu_approx(x: np.ndarray) -> np.ndarray:
        """GELU with tanh approximation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def gelu_derivative_approx(x: np.ndarray) -> np.ndarray:
        """Derivative of GELU (approximation)."""
        c = np.sqrt(2/np.pi)
        inner = c * (x + 0.044715 * x**3)
        tanh_inner = np.tanh(inner)
        sech2 = 1 - tanh_inner**2
        inner_deriv = c * (1 + 3 * 0.044715 * x**2)
        return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv
    
    # Sigmoid Linear Unit
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    # Swish / SiLU
    def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Swish activation: x * sigmoid(beta * x)."""
        return x * sigmoid(beta * x)
    
    def swish_derivative(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Derivative of swish."""
        sig = sigmoid(beta * x)
        return sig + beta * x * sig * (1 - sig)
    
    # Mish
    def softplus(x: np.ndarray) -> np.ndarray:
        """Softplus: log(1 + exp(x))."""
        return np.where(x > 20, x, np.log1p(np.exp(np.clip(x, -500, 20))))
    
    def mish(x: np.ndarray) -> np.ndarray:
        """Mish: x * tanh(softplus(x))."""
        return x * np.tanh(softplus(x))
    
    # Compare activations
    x = np.linspace(-3, 3, 7)
    
    print("\nActivation Comparison:")
    print(f"  x:          {np.round(x, 2)}")
    print(f"  GELU:       {np.round(gelu_approx(x), 4)}")
    print(f"  Swish:      {np.round(swish(x), 4)}")
    print(f"  Mish:       {np.round(mish(x), 4)}")
    print(f"  ReLU:       {np.maximum(0, x)}")
    
    print("\nDerivatives:")
    print(f"  x:          {np.round(x, 2)}")
    print(f"  GELU':      {np.round(gelu_derivative_approx(x), 4)}")
    print(f"  Swish':     {np.round(swish_derivative(x), 4)}")
    
    # Show GELU approximation accuracy
    print("\nGELU Approximation Accuracy:")
    test_x = np.array([-2, -1, 0, 1, 2])
    try:
        exact = gelu_exact(test_x)
        approx = gelu_approx(test_x)
        print(f"  Exact:  {np.round(exact, 6)}")
        print(f"  Approx: {np.round(approx, 6)}")
        print(f"  Max absolute error: {np.max(np.abs(exact - approx)):.6f}")
    except ImportError:
        print("  (scipy not available for exact comparison)")
    
    # Swish behavior with different beta
    print("\nSwish with Different Beta Values:")
    x_test = np.array([-1, 0, 1])
    for beta in [0.5, 1.0, 2.0, 10.0]:
        print(f"  β={beta:4.1f}: {np.round(swish(x_test, beta), 4)}")
    print("  β→∞:   ReLU behavior")
    
    return gelu_approx, swish, mish


# =============================================================================
# Example 4: Softmax and Temperature
# =============================================================================

def example_softmax():
    """
    Implement softmax with numerical stability and temperature scaling.
    """
    print("\n" + "=" * 70)
    print("Example 4: Softmax and Temperature Scaling")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable log-softmax."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=axis, keepdims=True))
    
    def softmax_temperature(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Softmax with temperature scaling."""
        return softmax(x / temperature)
    
    # Basic softmax
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    
    print("\nBasic Softmax:")
    print(f"  Logits: {logits}")
    print(f"  Probabilities: {np.round(probs, 4)}")
    print(f"  Sum: {np.sum(probs):.4f}")
    
    # Temperature effects
    print("\nTemperature Scaling Effects:")
    print(f"  Logits: {logits}")
    for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
        probs_temp = softmax_temperature(logits, temp)
        print(f"  τ={temp:4.1f}: {np.round(probs_temp, 4)} (entropy: {-np.sum(probs_temp * np.log(probs_temp + 1e-10)):.3f})")
    
    # Jacobian of softmax
    def softmax_jacobian(x: np.ndarray) -> np.ndarray:
        """Compute the Jacobian of softmax."""
        s = softmax(x)
        n = len(x)
        jacobian = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    jacobian[i, j] = s[i] * (1 - s[i])
                else:
                    jacobian[i, j] = -s[i] * s[j]
        return jacobian
    
    print("\nSoftmax Jacobian for logits [2, 1, 0.1]:")
    jacobian = softmax_jacobian(logits)
    print(f"  ∂softmax/∂logits:")
    for row in jacobian:
        print(f"    {np.round(row, 4)}")
    
    # Numerical stability demonstration
    print("\nNumerical Stability:")
    large_logits = np.array([1000, 1001, 1002])
    
    # Unstable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            unstable = np.exp(large_logits) / np.sum(np.exp(large_logits))
            print(f"  Unstable: {unstable} (may have inf/nan)")
        except:
            print(f"  Unstable: overflow error")
    
    # Stable
    stable = softmax(large_logits)
    print(f"  Stable:   {np.round(stable, 4)}")
    
    # Batch processing
    print("\nBatch Softmax:")
    batch_logits = np.array([
        [2.0, 1.0, 0.1],
        [0.1, 2.0, 1.0],
        [1.0, 0.1, 2.0]
    ])
    batch_probs = softmax(batch_logits, axis=1)
    print(f"  Input shape: {batch_logits.shape}")
    print(f"  Each row sums to: {np.sum(batch_probs, axis=1)}")
    
    return softmax, softmax_temperature


# =============================================================================
# Example 5: Gated Activations (GLU, SwiGLU, GEGLU)
# =============================================================================

def example_gated_activations():
    """
    Implement gated linear units used in modern language models.
    """
    print("\n" + "=" * 70)
    print("Example 5: Gated Activations (GLU, SwiGLU, GEGLU)")
    print("=" * 70)
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def swish(x: np.ndarray) -> np.ndarray:
        return x * sigmoid(x)
    
    # GLU: Gated Linear Unit
    def glu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Gated Linear Unit: x_a * sigmoid(x_b)
        Splits input along axis and applies gating.
        """
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * sigmoid(x_b)
    
    # ReGLU: ReLU-gated
    def reglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """ReGLU: x_a * relu(x_b)"""
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * np.maximum(0, x_b)
    
    # GEGLU: GELU-gated
    def geglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """GEGLU: x_a * gelu(x_b)"""
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * gelu(x_b)
    
    # SwiGLU: Swish-gated
    def swiglu(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """SwiGLU: x_a * swish(x_b)"""
        x_a, x_b = np.split(x, 2, axis=axis)
        return x_a * swish(x_b)
    
    # Demonstrate gated activations
    x = np.array([1.0, 0.5, -0.5, 2.0, 1.5, -1.0])  # 6D, will be split to 3D each
    
    print("\nGated Activation Comparison:")
    print(f"  Input: {x}")
    print(f"  x_a (first half): {x[:3]}")
    print(f"  x_b (second half): {x[3:]}")
    print(f"\n  GLU(x):    {np.round(glu(x), 4)}")
    print(f"  ReGLU(x):  {np.round(reglu(x), 4)}")
    print(f"  GEGLU(x):  {np.round(geglu(x), 4)}")
    print(f"  SwiGLU(x): {np.round(swiglu(x), 4)}")
    
    # FFN with gated activation (as in PaLM, LLaMA)
    class GatedFFN:
        """Feed-forward network with gated activation (SwiGLU)."""
        
        def __init__(self, d_model: int, d_ff: int, seed: int = 42):
            np.random.seed(seed)
            # Note: d_ff is the hidden size, but we need 2*d_ff for the gate
            scale = np.sqrt(2.0 / d_model)
            self.W1 = np.random.randn(d_model, d_ff * 2) * scale  # Project to 2*d_ff
            self.W2 = np.random.randn(d_ff, d_model) * scale      # Project back
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            x: (batch, seq_len, d_model)
            """
            # Project up with gate
            hidden = x @ self.W1  # (batch, seq_len, 2*d_ff)
            # Apply SwiGLU
            gated = swiglu(hidden, axis=-1)  # (batch, seq_len, d_ff)
            # Project down
            return gated @ self.W2  # (batch, seq_len, d_model)
    
    # Demonstrate FFN
    print("\nGated FFN Example:")
    d_model, d_ff = 8, 16
    ffn = GatedFFN(d_model, d_ff)
    x_input = np.random.randn(2, 4, d_model)  # batch=2, seq_len=4
    output = ffn.forward(x_input)
    print(f"  Input shape: {x_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  W1 shape: {ffn.W1.shape} (projects to 2*d_ff for gating)")
    print(f"  W2 shape: {ffn.W2.shape}")
    
    return glu, geglu, swiglu


# =============================================================================
# Example 6: Maxout and Adaptive Activations
# =============================================================================

def example_maxout_adaptive():
    """
    Implement Maxout networks and adaptive activation functions.
    """
    print("\n" + "=" * 70)
    print("Example 6: Maxout and Adaptive Activations")
    print("=" * 70)
    
    class MaxoutLayer:
        """
        Maxout layer: max over k linear pieces.
        Can learn any piecewise linear activation.
        """
        
        def __init__(self, input_dim: int, output_dim: int, k: int = 2, seed: int = 42):
            np.random.seed(seed)
            self.k = k
            scale = np.sqrt(2.0 / input_dim)
            # k weight matrices
            self.W = np.random.randn(k, input_dim, output_dim) * scale
            self.b = np.zeros((k, output_dim))
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            x: (batch_size, input_dim)
            Returns: (batch_size, output_dim)
            """
            # Compute all k linear transformations
            # z[i] = x @ W[i] + b[i] for each piece i
            z = np.einsum('bi,kio->bko', x, self.W) + self.b  # (batch, k, output_dim)
            
            # Take maximum over k pieces
            self.max_idx = np.argmax(z, axis=1)  # For backward pass
            return np.max(z, axis=1)  # (batch, output_dim)
    
    # Demonstrate Maxout
    print("\nMaxout Layer:")
    layer = MaxoutLayer(input_dim=4, output_dim=3, k=4)
    x = np.random.randn(5, 4)  # batch of 5
    output = layer.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of linear pieces: {layer.k}")
    print(f"  Note: Maxout with k=2 generalizes ReLU")
    
    # Adaptive Activation (Learnable)
    class PAU:
        """
        Padé Activation Unit: learnable rational function.
        f(x) = (a0 + a1*x + ... + am*x^m) / (1 + |b1*x + ... + bn*x^n|)
        """
        
        def __init__(self, m: int = 5, n: int = 4, init: str = 'relu'):
            self.m = m
            self.n = n
            
            # Initialize to approximate ReLU
            if init == 'relu':
                self.a = np.zeros(m + 1)
                self.a[0] = 0.0
                self.a[1] = 0.5
                self.b = np.zeros(n)
            else:
                self.a = np.random.randn(m + 1) * 0.1
                self.b = np.random.randn(n) * 0.1
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            # Numerator: polynomial in x
            powers = np.array([x ** i for i in range(self.m + 1)])
            numerator = np.sum(self.a[:, None] * powers, axis=0)
            
            # Denominator: 1 + |polynomial|
            if self.n > 0:
                b_powers = np.array([x ** (i + 1) for i in range(self.n)])
                denominator = 1 + np.abs(np.sum(self.b[:, None] * b_powers, axis=0))
            else:
                denominator = 1.0
            
            return numerator / denominator
    
    # Demonstrate PAU
    print("\nPadé Activation Unit (PAU):")
    pau = PAU(m=5, n=4, init='relu')
    x_test = np.linspace(-2, 2, 5)
    pau_output = pau.forward(x_test)
    relu_output = np.maximum(0, x_test)
    print(f"  x:     {np.round(x_test, 2)}")
    print(f"  PAU:   {np.round(pau_output, 4)}")
    print(f"  ReLU:  {relu_output}")
    print("  (PAU initialized to approximate ReLU)")
    
    # Adaptive Piecewise Linear (APL)
    class APL:
        """
        Adaptive Piecewise Linear: sum of hinge functions.
        f(x) = max(0, x) + sum_i a_i * max(0, -x + b_i)
        """
        
        def __init__(self, n_hinges: int = 3, seed: int = 42):
            np.random.seed(seed)
            self.n_hinges = n_hinges
            self.a = np.random.randn(n_hinges) * 0.1
            self.b = np.random.randn(n_hinges)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            result = np.maximum(0, x)
            for i in range(self.n_hinges):
                result = result + self.a[i] * np.maximum(0, -x + self.b[i])
            return result
    
    print("\nAdaptive Piecewise Linear (APL):")
    apl = APL(n_hinges=3)
    apl_output = apl.forward(x_test)
    print(f"  x:    {np.round(x_test, 2)}")
    print(f"  APL:  {np.round(apl_output, 4)}")
    
    return MaxoutLayer, PAU


# =============================================================================
# Example 7: Hard/Efficient Activations
# =============================================================================

def example_hard_activations():
    """
    Implement computationally efficient activation approximations
    for mobile and embedded deployment.
    """
    print("\n" + "=" * 70)
    print("Example 7: Hard/Efficient Activations")
    print("=" * 70)
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def swish(x: np.ndarray) -> np.ndarray:
        return x * sigmoid(x)
    
    # Hard Sigmoid
    def hard_sigmoid(x: np.ndarray) -> np.ndarray:
        """Piecewise linear approximation of sigmoid."""
        return np.clip(x / 6 + 0.5, 0, 1)
    
    # Hard Swish (used in MobileNetV3)
    def hard_swish(x: np.ndarray) -> np.ndarray:
        """Piecewise linear approximation of swish."""
        return x * hard_sigmoid(x)
    
    # Hard Tanh
    def hard_tanh(x: np.ndarray, min_val: float = -1, max_val: float = 1) -> np.ndarray:
        """Clipped linear function."""
        return np.clip(x, min_val, max_val)
    
    # ReLU6 (used in MobileNets)
    def relu6(x: np.ndarray) -> np.ndarray:
        """ReLU capped at 6."""
        return np.minimum(np.maximum(0, x), 6)
    
    # Compare soft vs hard versions
    x = np.linspace(-4, 4, 9)
    
    print("\nSoft vs Hard Activation Comparison:")
    print(f"  x:           {np.round(x, 2)}")
    print(f"  sigmoid:     {np.round(sigmoid(x), 4)}")
    print(f"  hard_sigmoid:{np.round(hard_sigmoid(x), 4)}")
    print()
    print(f"  swish:       {np.round(swish(x), 4)}")
    print(f"  hard_swish:  {np.round(hard_swish(x), 4)}")
    
    # Approximation error
    print("\nApproximation Error Analysis:")
    x_dense = np.linspace(-4, 4, 1000)
    
    sigmoid_error = np.abs(sigmoid(x_dense) - hard_sigmoid(x_dense))
    swish_error = np.abs(swish(x_dense) - hard_swish(x_dense))
    
    print(f"  Hard Sigmoid - max error: {np.max(sigmoid_error):.4f}, mean: {np.mean(sigmoid_error):.4f}")
    print(f"  Hard Swish   - max error: {np.max(swish_error):.4f}, mean: {np.mean(swish_error):.4f}")
    
    # Computational cost comparison (conceptual)
    print("\nComputational Cost (operations):")
    print("  Sigmoid:      1 exp, 1 add, 1 div = ~30 cycles")
    print("  Hard Sigmoid: 1 add, 1 mul, 2 clip = ~5 cycles")
    print("  Speedup:      ~6x")
    
    # ReLU6 demonstration
    print("\nReLU6 (for quantization-friendly networks):")
    x_relu6 = np.array([-2, 0, 3, 6, 8])
    print(f"  x:     {x_relu6}")
    print(f"  ReLU:  {np.maximum(0, x_relu6)}")
    print(f"  ReLU6: {relu6(x_relu6)}")
    print("  Bounded output [0,6] helps with fixed-point quantization")
    
    return hard_sigmoid, hard_swish, relu6


# =============================================================================
# Example 8: Activation Gradient Flow Analysis
# =============================================================================

def example_gradient_flow():
    """
    Analyze gradient flow through different activations
    in a deep network simulation.
    """
    print("\n" + "=" * 70)
    print("Example 8: Activation Gradient Flow Analysis")
    print("=" * 70)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    activations = {
        'sigmoid': (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
        'tanh': (np.tanh, lambda x: 1 - np.tanh(x) ** 2),
        'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
        'leaky_relu': (
            lambda x: np.where(x > 0, x, 0.01 * x),
            lambda x: np.where(x > 0, 1.0, 0.01)
        ),
        'elu': (
            lambda x: np.where(x > 0, x, np.exp(x) - 1),
            lambda x: np.where(x > 0, 1.0, np.exp(x))
        ),
    }
    
    def simulate_gradient_flow(activation_name: str, n_layers: int, n_samples: int = 1000):
        """Simulate gradient magnitude through a deep network."""
        _, deriv = activations[activation_name]
        
        np.random.seed(42)
        
        # Sample pre-activations (standard normal)
        grad_magnitudes = []
        
        for _ in range(n_samples):
            grad = 1.0  # Start with unit gradient
            
            for layer in range(n_layers):
                # Random pre-activation
                z = np.random.randn()
                # Gradient through activation
                grad *= deriv(z)
            
            grad_magnitudes.append(np.abs(grad))
        
        return np.mean(grad_magnitudes), np.std(grad_magnitudes)
    
    # Compare gradient flow
    n_layers = 10
    
    print(f"\nGradient Flow Through {n_layers} Layers:")
    print(f"  (Starting with unit gradient)")
    print()
    
    for name in activations:
        mean_grad, std_grad = simulate_gradient_flow(name, n_layers)
        status = "OK" if mean_grad > 0.01 else "VANISHING" if mean_grad < 0.001 else "LOW"
        print(f"  {name:12s}: mean = {mean_grad:.6f}, std = {std_grad:.6f} [{status}]")
    
    # Analysis with varying depth
    print("\n  Gradient magnitude vs depth (ReLU vs Sigmoid):")
    for depth in [5, 10, 20, 50]:
        relu_grad, _ = simulate_gradient_flow('relu', depth)
        sigmoid_grad, _ = simulate_gradient_flow('sigmoid', depth)
        print(f"    Depth {depth:2d}: ReLU = {relu_grad:.6f}, Sigmoid = {sigmoid_grad:.10f}")
    
    # Effect of initialization
    print("\nEffect of Activation Range on Gradient:")
    x_range = np.linspace(-3, 3, 1000)
    
    for name, (act, deriv) in activations.items():
        derivs = deriv(x_range)
        mean_deriv = np.mean(np.abs(derivs))
        max_deriv = np.max(np.abs(derivs))
        print(f"  {name:12s}: E[|f'(x)|] = {mean_deriv:.4f}, max|f'(x)| = {max_deriv:.4f}")


# =============================================================================
# Example 9: Neural Network with Different Activations
# =============================================================================

def example_nn_activations():
    """
    Compare training dynamics with different activation functions.
    """
    print("\n" + "=" * 70)
    print("Example 9: Neural Network with Different Activations")
    print("=" * 70)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    class SimpleNN:
        """Simple 2-layer neural network for comparison."""
        
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                     activation: str = 'relu', seed: int = 42):
            np.random.seed(seed)
            self.activation_name = activation
            
            # He initialization for ReLU family, Xavier for others
            if activation in ['relu', 'leaky_relu', 'elu']:
                scale1 = np.sqrt(2.0 / input_dim)
                scale2 = np.sqrt(2.0 / hidden_dim)
            else:
                scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
                scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))
            
            self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
            self.b2 = np.zeros(output_dim)
        
        def activation(self, x: np.ndarray) -> np.ndarray:
            if self.activation_name == 'relu':
                return np.maximum(0, x)
            elif self.activation_name == 'leaky_relu':
                return np.where(x > 0, x, 0.01 * x)
            elif self.activation_name == 'tanh':
                return np.tanh(x)
            elif self.activation_name == 'sigmoid':
                return sigmoid(x)
            elif self.activation_name == 'gelu':
                return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        
        def activation_deriv(self, x: np.ndarray) -> np.ndarray:
            if self.activation_name == 'relu':
                return (x > 0).astype(float)
            elif self.activation_name == 'leaky_relu':
                return np.where(x > 0, 1.0, 0.01)
            elif self.activation_name == 'tanh':
                return 1 - np.tanh(x) ** 2
            elif self.activation_name == 'sigmoid':
                s = sigmoid(x)
                return s * (1 - s)
            elif self.activation_name == 'gelu':
                c = np.sqrt(2/np.pi)
                inner = c * (x + 0.044715 * x**3)
                tanh_inner = np.tanh(inner)
                sech2 = 1 - tanh_inner**2
                inner_deriv = c * (1 + 3 * 0.044715 * x**2)
                return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_deriv
        
        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
            self.z1 = x @ self.W1 + self.b1
            self.a1 = self.activation(self.z1)
            self.z2 = self.a1 @ self.W2 + self.b2
            # Softmax output
            exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
            self.probs = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
            return self.probs
        
        def backward(self, x: np.ndarray, y: np.ndarray) -> dict:
            batch_size = x.shape[0]
            
            # Output gradient (softmax + cross-entropy)
            dz2 = self.probs - y
            
            # Layer 2 gradients
            dW2 = self.a1.T @ dz2 / batch_size
            db2 = np.mean(dz2, axis=0)
            
            # Hidden layer gradient
            da1 = dz2 @ self.W2.T
            dz1 = da1 * self.activation_deriv(self.z1)
            
            # Layer 1 gradients
            dW1 = x.T @ dz1 / batch_size
            db1 = np.mean(dz1, axis=0)
            
            return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
        
        def update(self, grads: dict, lr: float):
            self.W1 -= lr * grads['W1']
            self.b1 -= lr * grads['b1']
            self.W2 -= lr * grads['W2']
            self.b2 -= lr * grads['b2']
        
        def loss(self, y: np.ndarray) -> float:
            return -np.mean(np.sum(y * np.log(self.probs + 1e-10), axis=1))
    
    # Generate XOR-like data
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 2)
    y_labels = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    y = np.eye(2)[y_labels]  # One-hot
    
    # Train with different activations
    print("\nTraining on XOR-like Data:")
    print(f"  Samples: {n_samples}, Features: 2, Classes: 2")
    
    activations = ['relu', 'leaky_relu', 'tanh', 'gelu']
    n_epochs = 100
    lr = 0.5
    
    results = {}
    
    for act_name in activations:
        nn = SimpleNN(2, 16, 2, activation=act_name)
        losses = []
        
        for epoch in range(n_epochs):
            probs = nn.forward(X)
            loss = nn.loss(y)
            grads = nn.backward(X, y)
            nn.update(grads, lr)
            losses.append(loss)
        
        # Compute accuracy
        preds = np.argmax(nn.forward(X), axis=1)
        accuracy = np.mean(preds == y_labels)
        
        results[act_name] = {
            'final_loss': losses[-1],
            'accuracy': accuracy,
            'loss_history': losses
        }
    
    print(f"\nResults after {n_epochs} epochs (lr={lr}):")
    for act_name, res in results.items():
        print(f"  {act_name:12s}: loss = {res['final_loss']:.4f}, accuracy = {res['accuracy']:.2%}")
    
    return results


# =============================================================================
# Example 10: Activation Function Visualization Data
# =============================================================================

def example_visualization_data():
    """
    Generate data for visualizing activation functions and their properties.
    """
    print("\n" + "=" * 70)
    print("Example 10: Activation Function Properties Summary")
    print("=" * 70)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    def swish(x):
        return x * sigmoid(x)
    
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    def selu(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * elu(x, alpha)
    
    # Define activations with properties
    activations = {
        'sigmoid': {
            'func': sigmoid,
            'range': (0, 1),
            'zero_centered': False,
            'unbounded': False,
            'dead_neurons': False,
            'smooth': True,
        },
        'tanh': {
            'func': np.tanh,
            'range': (-1, 1),
            'zero_centered': True,
            'unbounded': False,
            'dead_neurons': False,
            'smooth': True,
        },
        'relu': {
            'func': lambda x: np.maximum(0, x),
            'range': (0, 'inf'),
            'zero_centered': False,
            'unbounded': True,
            'dead_neurons': True,
            'smooth': False,
        },
        'leaky_relu': {
            'func': lambda x: np.where(x > 0, x, 0.01 * x),
            'range': ('-inf', 'inf'),
            'zero_centered': False,
            'unbounded': True,
            'dead_neurons': False,
            'smooth': False,
        },
        'elu': {
            'func': elu,
            'range': (-1, 'inf'),
            'zero_centered': 'approx',
            'unbounded': True,
            'dead_neurons': False,
            'smooth': True,
        },
        'selu': {
            'func': selu,
            'range': ('-λα', 'inf'),
            'zero_centered': True,
            'unbounded': True,
            'dead_neurons': False,
            'smooth': True,
        },
        'gelu': {
            'func': gelu,
            'range': ('~-0.17', 'inf'),
            'zero_centered': 'approx',
            'unbounded': True,
            'dead_neurons': False,
            'smooth': True,
        },
        'swish': {
            'func': swish,
            'range': ('~-0.28', 'inf'),
            'zero_centered': False,
            'unbounded': True,
            'dead_neurons': False,
            'smooth': True,
        },
    }
    
    # Generate data for plotting
    x = np.linspace(-4, 4, 1000)
    
    print("\nActivation Function Properties:")
    print("-" * 75)
    print(f"{'Name':12s} | {'Range':15s} | {'Zero-cent':9s} | {'Unbounded':9s} | {'Dead':5s} | {'Smooth':6s}")
    print("-" * 75)
    
    for name, props in activations.items():
        range_str = f"({props['range'][0]}, {props['range'][1]})"
        zero = str(props['zero_centered'])
        unbnd = 'Yes' if props['unbounded'] else 'No'
        dead = 'Yes' if props['dead_neurons'] else 'No'
        smooth = 'Yes' if props['smooth'] else 'No'
        print(f"{name:12s} | {range_str:15s} | {zero:9s} | {unbnd:9s} | {dead:5s} | {smooth:6s}")
    
    print("\nActivation Values at Key Points:")
    key_points = [-2, -1, 0, 1, 2]
    print(f"  {'Activation':12s}", end="")
    for p in key_points:
        print(f" | x={p:2d}", end="")
    print()
    print("-" * 60)
    
    for name, props in activations.items():
        values = props['func'](np.array(key_points))
        print(f"  {name:12s}", end="")
        for v in values:
            print(f" | {v:5.2f}", end="")
        print()
    
    # Statistics over standard normal distribution
    print("\nStatistics over N(0,1) Inputs:")
    samples = np.random.randn(10000)
    
    print(f"  {'Activation':12s} | {'Mean':8s} | {'Std':8s} | {'% Active':10s}")
    print("-" * 50)
    
    for name, props in activations.items():
        outputs = props['func'](samples)
        mean_out = np.mean(outputs)
        std_out = np.std(outputs)
        # For bounded activations, "active" means |output| > 0.1
        # For unbounded, "active" means output > 0
        if name in ['sigmoid', 'tanh']:
            active = np.mean(np.abs(outputs) > 0.1)
        else:
            active = np.mean(outputs != 0)
        print(f"  {name:12s} | {mean_out:8.4f} | {std_out:8.4f} | {active:10.2%}")
    
    # Lipschitz constants
    print("\nLipschitz Constants (max |f'(x)|):")
    for name, props in activations.items():
        # Numerical derivative
        dx = 0.0001
        f = props['func']
        derivs = (f(x + dx) - f(x - dx)) / (2 * dx)
        lipschitz = np.max(np.abs(derivs))
        print(f"  {name:12s}: {lipschitz:.4f}")
    
    return activations


def main():
    """Run all activation function examples."""
    print("ACTIVATION FUNCTIONS FOR NEURAL NETWORKS")
    print("=" * 70)
    
    # Run all examples
    example_classic_activations()
    example_relu_family()
    example_gelu_swish()
    example_softmax()
    example_gated_activations()
    example_maxout_adaptive()
    example_hard_activations()
    example_gradient_flow()
    example_nn_activations()
    example_visualization_data()
    
    print("\n" + "=" * 70)
    print("All activation function examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
