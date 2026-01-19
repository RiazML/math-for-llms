"""
Neural Networks - Exercises
===========================

Practice exercises for neural network fundamentals.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict


# =============================================================================
# Exercise 1: Dense Layer with Backpropagation
# =============================================================================

def exercise_dense_layer():
    """
    Exercise: Implement a dense layer with forward and backward pass.
    
    Tasks:
    1. Forward: y = xW + b
    2. Backward: compute gradients
    3. Test gradient computation
    """
    print("=" * 70)
    print("Exercise 1: Dense Layer")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class Dense:
        def __init__(self, in_features: int, out_features: int):
            self.W = np.random.randn(in_features, out_features) * 0.01
            self.b = np.zeros(out_features)
            self._cache = None
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Compute y = xW + b"""
            pass
        
        def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Returns: (grad_x, grad_W, grad_b)
            """
            pass


def solution_dense_layer():
    """Reference solution for dense layer."""
    print("\n--- Solution ---\n")
    
    class Dense:
        def __init__(self, in_features: int, out_features: int):
            self.W = np.random.randn(in_features, out_features) * 0.01
            self.b = np.zeros(out_features)
            self._cache = None
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            self._cache = x
            return x @ self.W + self.b
        
        def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = self._cache
            grad_x = grad_output @ self.W.T
            grad_W = x.T @ grad_output
            grad_b = np.sum(grad_output, axis=0)
            return grad_x, grad_W, grad_b
    
    # Test
    np.random.seed(42)
    
    layer = Dense(4, 3)
    x = np.random.randn(2, 4)
    
    # Forward
    y = layer.forward(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Backward
    grad_y = np.random.randn(2, 3)
    grad_x, grad_W, grad_b = layer.backward(grad_y)
    
    print(f"grad_x shape: {grad_x.shape}")
    print(f"grad_W shape: {grad_W.shape}")
    print(f"grad_b shape: {grad_b.shape}")
    
    # Numerical gradient check
    eps = 1e-5
    
    numerical_grad_W = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            layer.W[i, j] += eps
            y_plus = layer.forward(x)
            layer.W[i, j] -= 2 * eps
            y_minus = layer.forward(x)
            layer.W[i, j] += eps
            
            numerical_grad_W[i, j] = np.sum(grad_y * (y_plus - y_minus)) / (2 * eps)
    
    print(f"\nGradient check (should be ~0):")
    print(f"  Max diff: {np.max(np.abs(grad_W - numerical_grad_W)):.6e}")


# =============================================================================
# Exercise 2: Activation Functions
# =============================================================================

def exercise_activations():
    """
    Exercise: Implement activation functions with derivatives.
    
    Tasks:
    1. ReLU and its derivative
    2. Sigmoid and its derivative
    3. Softmax (stable)
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Activation Functions")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def relu(x: np.ndarray) -> np.ndarray:
        pass
    
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        pass
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        pass
    
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        pass
    
    def softmax(x: np.ndarray) -> np.ndarray:
        pass


def solution_activations():
    """Reference solution for activations."""
    print("\n--- Solution ---\n")
    
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))
    
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = sigmoid(x)
        return s * (1 - s)
    
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Test
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    print(f"Input: {x}")
    print(f"ReLU: {relu(x)}")
    print(f"ReLU': {relu_derivative(x)}")
    print(f"Sigmoid: {sigmoid(x).round(4)}")
    print(f"Sigmoid': {sigmoid_derivative(x).round(4)}")
    
    # Softmax
    logits = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    probs = softmax(logits)
    print(f"\nSoftmax:")
    print(f"  Input: {logits}")
    print(f"  Output: {probs.round(4)}")
    print(f"  Sum: {np.sum(probs, axis=-1)}")


# =============================================================================
# Exercise 3: Cross-Entropy Loss
# =============================================================================

def exercise_cross_entropy():
    """
    Exercise: Implement cross-entropy loss and gradient.
    
    Tasks:
    1. Binary cross-entropy
    2. Categorical cross-entropy
    3. Gradient w.r.t. logits
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Cross-Entropy Loss")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass
    
    def categorical_cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
        """y_true is one-hot encoded"""
        pass
    
    def softmax_cross_entropy_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient of CE loss w.r.t. logits"""
        pass


def solution_cross_entropy():
    """Reference solution for cross-entropy."""
    print("\n--- Solution ---\n")
    
    def binary_cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def categorical_cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
        probs = softmax(logits)
        return -np.mean(np.sum(y_true * np.log(probs + 1e-7), axis=-1))
    
    def softmax_cross_entropy_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        probs = softmax(logits)
        return (probs - y_true) / len(logits)
    
    # Test
    np.random.seed(42)
    
    # Binary
    y_pred = np.array([0.9, 0.1, 0.8])
    y_true = np.array([1.0, 0.0, 1.0])
    
    bce = binary_cross_entropy(y_pred, y_true)
    print(f"Binary cross-entropy: {bce:.4f}")
    
    # Categorical
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
    y_one_hot = np.array([[1, 0, 0], [0, 1, 0]])
    
    cce = categorical_cross_entropy(logits, y_one_hot)
    print(f"Categorical cross-entropy: {cce:.4f}")
    
    grad = softmax_cross_entropy_gradient(logits, y_one_hot)
    print(f"Gradient shape: {grad.shape}")
    print(f"Gradient:\n{grad.round(4)}")


# =============================================================================
# Exercise 4: Convolutional Layer
# =============================================================================

def exercise_conv_layer():
    """
    Exercise: Implement 2D convolution forward pass.
    
    Tasks:
    1. Basic 2D convolution
    2. With padding and stride
    3. Batch processing
    """
    print("\n" + "=" * 70)
    print("Exercise 4: 2D Convolution")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def conv2d(x: np.ndarray, kernel: np.ndarray, 
               stride: int = 1, padding: int = 0) -> np.ndarray:
        """
        x: (batch, channels, H, W)
        kernel: (out_channels, in_channels, kH, kW)
        """
        pass


def solution_conv_layer():
    """Reference solution for convolution."""
    print("\n--- Solution ---\n")
    
    def conv2d(x: np.ndarray, kernel: np.ndarray,
               stride: int = 1, padding: int = 0) -> np.ndarray:
        batch, in_ch, H, W = x.shape
        out_ch, _, kH, kW = kernel.shape
        
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        
        H_out = (H + 2 * padding - kH) // stride + 1
        W_out = (W + 2 * padding - kW) // stride + 1
        
        output = np.zeros((batch, out_ch, H_out, W_out))
        
        for b in range(batch):
            for c_out in range(out_ch):
                for i in range(H_out):
                    for j in range(W_out):
                        i_start = i * stride
                        j_start = j * stride
                        
                        region = x[b, :, i_start:i_start + kH, j_start:j_start + kW]
                        output[b, c_out, i, j] = np.sum(region * kernel[c_out])
        
        return output
    
    # Test
    np.random.seed(42)
    
    x = np.random.randn(2, 3, 8, 8)  # 2 images, 3 channels, 8x8
    kernel = np.random.randn(4, 3, 3, 3)  # 4 output channels, 3x3 kernel
    
    # No padding
    out1 = conv2d(x, kernel, stride=1, padding=0)
    print(f"Input: {x.shape}")
    print(f"Kernel: {kernel.shape}")
    print(f"Output (no pad): {out1.shape}")
    
    # With padding
    out2 = conv2d(x, kernel, stride=1, padding=1)
    print(f"Output (pad=1): {out2.shape}")
    
    # With stride
    out3 = conv2d(x, kernel, stride=2, padding=1)
    print(f"Output (stride=2): {out3.shape}")


# =============================================================================
# Exercise 5: LSTM Cell
# =============================================================================

def exercise_lstm():
    """
    Exercise: Implement LSTM cell forward pass.
    
    Tasks:
    1. Gate computations (forget, input, output)
    2. Cell state update
    3. Hidden state computation
    """
    print("\n" + "=" * 70)
    print("Exercise 5: LSTM Cell")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class LSTMCell:
        def __init__(self, input_size: int, hidden_size: int):
            self.input_size = input_size
            self.hidden_size = hidden_size
            # Initialize weights
            pass
        
        def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                    c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            pass


def solution_lstm():
    """Reference solution for LSTM."""
    print("\n--- Solution ---\n")
    
    class LSTMCell:
        def __init__(self, input_size: int, hidden_size: int):
            self.H = hidden_size
            std = np.sqrt(1.0 / hidden_size)
            
            self.W_ih = std * np.random.randn(4 * hidden_size, input_size)
            self.W_hh = std * np.random.randn(4 * hidden_size, hidden_size)
            self.bias = np.zeros(4 * hidden_size)
            self.bias[hidden_size:2*hidden_size] = 1.0  # Forget gate bias
        
        def _sigmoid(self, x):
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        def forward(self, x: np.ndarray, h_prev: np.ndarray,
                    c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            H = self.H
            
            gates = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.bias
            
            f = self._sigmoid(gates[:, 0:H])
            i = self._sigmoid(gates[:, H:2*H])
            o = self._sigmoid(gates[:, 2*H:3*H])
            g = np.tanh(gates[:, 3*H:4*H])
            
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            
            return h, c
    
    # Test
    np.random.seed(42)
    
    batch, input_size, hidden_size = 4, 8, 16
    
    cell = LSTMCell(input_size, hidden_size)
    
    x = np.random.randn(batch, input_size)
    h = np.zeros((batch, hidden_size))
    c = np.zeros((batch, hidden_size))
    
    h_new, c_new = cell.forward(x, h, c)
    
    print(f"Input: {x.shape}")
    print(f"h_prev: {h.shape}")
    print(f"h_new: {h_new.shape}")
    print(f"c_new: {c_new.shape}")
    
    # Process sequence
    seq_len = 10
    x_seq = np.random.randn(batch, seq_len, input_size)
    
    outputs = []
    h, c = np.zeros((batch, hidden_size)), np.zeros((batch, hidden_size))
    
    for t in range(seq_len):
        h, c = cell.forward(x_seq[:, t], h, c)
        outputs.append(h)
    
    print(f"\nSequence output shape: ({len(outputs)}, {outputs[0].shape})")


# =============================================================================
# Exercise 6: Attention Mechanism
# =============================================================================

def exercise_attention():
    """
    Exercise: Implement scaled dot-product attention.
    
    Tasks:
    1. Compute attention scores
    2. Apply softmax
    3. Weighted sum of values
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, 
                                     V: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Q, K, V: (batch, seq_len, d_k)
        mask: (batch, seq_len, seq_len) or None
        """
        pass


def solution_attention():
    """Reference solution for attention."""
    print("\n--- Solution ---\n")
    
    def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                     V: np.ndarray, mask: np.ndarray = None):
        d_k = Q.shape[-1]
        
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        
        return attn @ V, attn
    
    # Test
    np.random.seed(42)
    
    batch, seq_len, d_k = 2, 8, 16
    
    Q = np.random.randn(batch, seq_len, d_k)
    K = np.random.randn(batch, seq_len, d_k)
    V = np.random.randn(batch, seq_len, d_k)
    
    output, attn = scaled_dot_product_attention(Q, K, V)
    
    print(f"Q, K, V shape: ({batch}, {seq_len}, {d_k})")
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"Attention sums to 1: {np.allclose(attn.sum(axis=-1), 1)}")
    
    # With causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) == 0
    causal_mask = np.broadcast_to(causal_mask, (batch, seq_len, seq_len))
    
    output_masked, attn_masked = scaled_dot_product_attention(Q, K, V, causal_mask)
    
    print(f"\nCausal attention (position 0):")
    print(f"  {attn_masked[0, 0].round(3)}")
    print(f"Causal attention (position 7):")
    print(f"  {attn_masked[0, 7].round(3)}")


# =============================================================================
# Exercise 7: Layer Normalization
# =============================================================================

def exercise_layer_norm():
    """
    Exercise: Implement layer normalization.
    
    Tasks:
    1. Compute mean and variance per sample
    2. Normalize
    3. Scale and shift
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Layer Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class LayerNorm:
        def __init__(self, features: int, eps: float = 1e-5):
            self.gamma = np.ones(features)
            self.beta = np.zeros(features)
            self.eps = eps
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass


def solution_layer_norm():
    """Reference solution for layer norm."""
    print("\n--- Solution ---\n")
    
    class LayerNorm:
        def __init__(self, features: int, eps: float = 1e-5):
            self.gamma = np.ones(features)
            self.beta = np.zeros(features)
            self.eps = eps
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            return self.gamma * x_norm + self.beta
    
    # Test
    np.random.seed(42)
    
    batch, seq_len, features = 4, 8, 64
    x = np.random.randn(batch, seq_len, features) * 5 + 3
    
    ln = LayerNorm(features)
    y = ln.forward(x)
    
    print(f"Input statistics:")
    print(f"  Mean: {np.mean(x):.4f}")
    print(f"  Std: {np.std(x):.4f}")
    
    print(f"\nAfter LayerNorm:")
    print(f"  Per-position mean: {np.mean(np.mean(y, axis=-1)):.6f}")
    print(f"  Per-position std: {np.mean(np.std(y, axis=-1)):.4f}")


# =============================================================================
# Exercise 8: Adam Optimizer
# =============================================================================

def exercise_adam():
    """
    Exercise: Implement Adam optimizer.
    
    Tasks:
    1. First moment (momentum)
    2. Second moment (RMSprop)
    3. Bias correction
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Adam Optimizer")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class Adam:
        def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                     beta2: float = 0.999, eps: float = 1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.t = 0
            self.m = None
            self.v = None
        
        def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            """Update and return new parameter."""
            pass


def solution_adam():
    """Reference solution for Adam."""
    print("\n--- Solution ---\n")
    
    class Adam:
        def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                     beta2: float = 0.999, eps: float = 1e-8):
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.t = 0
            self.m = None
            self.v = None
        
        def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            self.t += 1
            
            if self.m is None:
                self.m = np.zeros_like(param)
                self.v = np.zeros_like(param)
            
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            
            return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    # Test: minimize x^2 + y^2
    np.random.seed(42)
    
    x = np.array([5.0, -3.0])
    opt = Adam(lr=0.1)
    
    print("Minimizing f(x,y) = x² + y²:")
    for i in range(100):
        grad = 2 * x
        x = opt.step(x, grad)
        
        if i < 5 or i % 20 == 0:
            print(f"  Step {i}: x={x.round(4)}, f={np.sum(x**2):.6f}")


# =============================================================================
# Exercise 9: Residual Connection
# =============================================================================

def exercise_residual():
    """
    Exercise: Implement residual block.
    
    Tasks:
    1. Pre-activation residual
    2. Post-activation residual
    3. Gradient flow analysis
    """
    print("\n" + "=" * 70)
    print("Exercise 9: Residual Connections")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class ResidualBlock:
        def __init__(self, features: int):
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass


def solution_residual():
    """Reference solution for residual."""
    print("\n--- Solution ---\n")
    
    class ResidualBlock:
        def __init__(self, features: int):
            std = np.sqrt(2.0 / features)
            self.W1 = std * np.random.randn(features, features)
            self.b1 = np.zeros(features)
            self.W2 = std * np.random.randn(features, features)
            self.b2 = np.zeros(features)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            # F(x) = W2 * ReLU(W1 * x + b1) + b2
            h = np.maximum(0, x @ self.W1 + self.b1)
            F_x = h @ self.W2 + self.b2
            return x + F_x  # Residual connection
    
    # Test
    np.random.seed(42)
    
    batch, features = 32, 64
    n_blocks = 20
    
    # Compare with and without residuals
    x_resid = np.random.randn(batch, features)
    x_plain = x_resid.copy()
    
    resid_blocks = [ResidualBlock(features) for _ in range(n_blocks)]
    
    print(f"Passing through {n_blocks} blocks:")
    
    for i, block in enumerate(resid_blocks):
        x_resid = block.forward(x_resid)
        x_plain = np.maximum(0, x_plain @ block.W1 + block.b1) @ block.W2 + block.b2
        
        if i == 0 or i == n_blocks - 1:
            print(f"  Block {i}: resid std={np.std(x_resid):.4f}, plain std={np.std(x_plain):.4f}")
    
    print(f"\nResidual preserves signal, plain network may vanish/explode")


# =============================================================================
# Exercise 10: Complete Training Loop
# =============================================================================

def exercise_training_loop():
    """
    Exercise: Implement complete training loop.
    
    Tasks:
    1. Forward pass
    2. Backward pass
    3. Parameter update
    4. Evaluation
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Training Loop")
    print("=" * 70)
    
    # YOUR CODE HERE
    pass


def solution_training_loop():
    """Reference solution for training loop."""
    print("\n--- Solution ---\n")
    
    class SimpleNN:
        def __init__(self, layer_sizes: List[int]):
            self.weights = []
            self.biases = []
            
            for i in range(len(layer_sizes) - 1):
                std = np.sqrt(2.0 / layer_sizes[i])
                self.weights.append(std * np.random.randn(layer_sizes[i], layer_sizes[i+1]))
                self.biases.append(np.zeros(layer_sizes[i+1]))
            
            # Adam state
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0
        
        def forward(self, x):
            self.activations = [x]
            self.zs = []
            
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = self.activations[-1] @ W + b
                self.zs.append(z)
                
                if i < len(self.weights) - 1:
                    a = np.maximum(0, z)  # ReLU
                else:
                    # Softmax for output
                    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
                    a = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
                
                self.activations.append(a)
            
            return self.activations[-1]
        
        def backward(self, y):
            n = len(y)
            grad_w = [np.zeros_like(w) for w in self.weights]
            grad_b = [np.zeros_like(b) for b in self.biases]
            
            # Output layer (softmax + CE)
            delta = (self.activations[-1] - y) / n
            grad_w[-1] = self.activations[-2].T @ delta
            grad_b[-1] = np.sum(delta, axis=0)
            
            # Hidden layers
            for l in range(len(self.weights) - 2, -1, -1):
                delta = (delta @ self.weights[l+1].T) * (self.zs[l] > 0)
                grad_w[l] = self.activations[l].T @ delta
                grad_b[l] = np.sum(delta, axis=0)
            
            return grad_w, grad_b
        
        def update(self, grad_w, grad_b, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
            self.t += 1
            
            for i in range(len(self.weights)):
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grad_w[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * grad_w[i]**2
                
                m_hat = self.m_w[i] / (1 - beta1**self.t)
                v_hat = self.v_w[i] / (1 - beta2**self.t)
                
                self.weights[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grad_b[i]
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * grad_b[i]**2
                
                m_hat = self.m_b[i] / (1 - beta1**self.t)
                v_hat = self.v_b[i] / (1 - beta2**self.t)
                
                self.biases[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        def train_step(self, X, y, lr=0.001):
            probs = self.forward(X)
            loss = -np.mean(np.sum(y * np.log(probs + 1e-7), axis=-1))
            grad_w, grad_b = self.backward(y)
            self.update(grad_w, grad_b, lr)
            return loss
        
        def predict(self, X):
            return np.argmax(self.forward(X), axis=-1)
    
    # Generate classification data
    np.random.seed(42)
    
    n_samples = 1000
    n_classes = 3
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    true_W = np.random.randn(n_features, n_classes)
    logits = X @ true_W
    y_idx = np.argmax(logits, axis=-1)
    y = np.eye(n_classes)[y_idx]
    
    # Split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    y_test_idx = y_idx[800:]
    
    # Train
    model = SimpleNN([n_features, 32, 16, n_classes])
    
    print("Training:")
    for epoch in range(100):
        # Mini-batch
        idx = np.random.permutation(len(X_train))
        total_loss = 0
        
        for i in range(0, len(X_train), 32):
            batch_idx = idx[i:i+32]
            loss = model.train_step(X_train[batch_idx], y_train[batch_idx])
            total_loss += loss
        
        if epoch % 20 == 0:
            train_acc = np.mean(model.predict(X_train) == np.argmax(y_train, axis=-1))
            test_acc = np.mean(model.predict(X_test) == y_test_idx)
            print(f"  Epoch {epoch}: loss={total_loss:.4f}, train_acc={train_acc:.2%}, test_acc={test_acc:.2%}")
    
    final_acc = np.mean(model.predict(X_test) == y_test_idx)
    print(f"\nFinal test accuracy: {final_acc:.2%}")


def main():
    """Run all exercises with solutions."""
    print("NEURAL NETWORKS - EXERCISES")
    print("=" * 70)
    
    exercise_dense_layer()
    solution_dense_layer()
    
    exercise_activations()
    solution_activations()
    
    exercise_cross_entropy()
    solution_cross_entropy()
    
    exercise_conv_layer()
    solution_conv_layer()
    
    exercise_lstm()
    solution_lstm()
    
    exercise_attention()
    solution_attention()
    
    exercise_layer_norm()
    solution_layer_norm()
    
    exercise_adam()
    solution_adam()
    
    exercise_residual()
    solution_residual()
    
    exercise_training_loop()
    solution_training_loop()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
