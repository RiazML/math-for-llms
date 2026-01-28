"""
Neural Networks - Examples
==========================

Comprehensive examples of neural network fundamentals from scratch.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict


# =============================================================================
# Example 1: Multi-Layer Perceptron from Scratch
# =============================================================================

def example_mlp():
    """
    Multi-layer perceptron with backpropagation.
    """
    print("=" * 70)
    print("Example 1: Multi-Layer Perceptron")
    print("=" * 70)
    
    class MLP:
        """Multi-layer perceptron with configurable architecture."""
        
        def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
            self.layer_sizes = layer_sizes
            self.activation = activation
            self.weights = []
            self.biases = []
            
            # Xavier initialization
            for i in range(len(layer_sizes) - 1):
                fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                self.weights.append(std * np.random.randn(fan_in, fan_out))
                self.biases.append(np.zeros(fan_out))
        
        def _activate(self, z: np.ndarray) -> np.ndarray:
            if self.activation == 'relu':
                return np.maximum(0, z)
            elif self.activation == 'tanh':
                return np.tanh(z)
            elif self.activation == 'sigmoid':
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return z
        
        def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
            if self.activation == 'relu':
                return (z > 0).astype(float)
            elif self.activation == 'tanh':
                return 1 - np.tanh(z)**2
            elif self.activation == 'sigmoid':
                s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                return s * (1 - s)
            return np.ones_like(z)
        
        def forward(self, X: np.ndarray) -> Tuple[List, List]:
            """Forward pass, returns activations and pre-activations."""
            activations = [X]
            zs = []
            
            a = X
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                z = a @ W + b
                zs.append(z)
                
                if i < len(self.weights) - 1:
                    a = self._activate(z)
                else:
                    a = z  # No activation on output
                
                activations.append(a)
            
            return activations, zs
        
        def backward(self, activations: List, zs: List, 
                     y: np.ndarray) -> Tuple[List, List]:
            """Backward pass, returns gradients."""
            n = len(y)
            grad_W = [np.zeros_like(W) for W in self.weights]
            grad_b = [np.zeros_like(b) for b in self.biases]
            
            # Output layer gradient (MSE loss)
            delta = (activations[-1] - y) / n
            grad_W[-1] = activations[-2].T @ delta
            grad_b[-1] = np.sum(delta, axis=0)
            
            # Hidden layers
            for l in range(len(self.weights) - 2, -1, -1):
                delta = (delta @ self.weights[l + 1].T) * self._activate_derivative(zs[l])
                grad_W[l] = activations[l].T @ delta
                grad_b[l] = np.sum(delta, axis=0)
            
            return grad_W, grad_b
        
        def fit(self, X: np.ndarray, y: np.ndarray, 
                epochs: int = 1000, lr: float = 0.01) -> List[float]:
            """Train the network."""
            losses = []
            
            for epoch in range(epochs):
                # Forward
                activations, zs = self.forward(X)
                
                # Loss
                loss = 0.5 * np.mean((activations[-1] - y)**2)
                losses.append(loss)
                
                # Backward
                grad_W, grad_b = self.backward(activations, zs, y)
                
                # Update
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * grad_W[i]
                    self.biases[i] -= lr * grad_b[i]
            
            return losses
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            activations, _ = self.forward(X)
            return activations[-1]
    
    # Test: Learn XOR function
    np.random.seed(42)
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    mlp = MLP([2, 8, 1], activation='relu')
    losses = mlp.fit(X, y, epochs=5000, lr=0.1)
    
    print(f"\nXOR Problem:")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    
    print(f"\nPredictions:")
    for i in range(4):
        pred = mlp.predict(X[i:i+1])[0, 0]
        print(f"  Input: {X[i]} -> Predicted: {pred:.4f}, Target: {y[i, 0]}")


# =============================================================================
# Example 2: Convolutional Neural Network
# =============================================================================

def example_cnn():
    """
    Convolutional layer implementation.
    """
    print("\n" + "=" * 70)
    print("Example 2: Convolutional Neural Network")
    print("=" * 70)
    
    class Conv2D:
        """2D Convolutional layer."""
        
        def __init__(self, in_channels: int, out_channels: int,
                     kernel_size: int, stride: int = 1, padding: int = 0):
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            
            # He initialization
            std = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
            self.weight = std * np.random.randn(out_channels, in_channels, 
                                                 kernel_size, kernel_size)
            self.bias = np.zeros(out_channels)
            
            self._cache = None
        
        def _pad(self, x: np.ndarray) -> np.ndarray:
            if self.padding == 0:
                return x
            return np.pad(x, ((0, 0), (0, 0), 
                             (self.padding, self.padding),
                             (self.padding, self.padding)))
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            x: (batch, in_channels, H, W)
            output: (batch, out_channels, H_out, W_out)
            """
            batch_size = x.shape[0]
            x_pad = self._pad(x)
            
            H, W = x.shape[2], x.shape[3]
            H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
            W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
            
            output = np.zeros((batch_size, self.out_channels, H_out, W_out))
            
            for b in range(batch_size):
                for c_out in range(self.out_channels):
                    for i in range(H_out):
                        for j in range(W_out):
                            i_start = i * self.stride
                            j_start = j * self.stride
                            
                            region = x_pad[b, :, 
                                          i_start:i_start + self.kernel_size,
                                          j_start:j_start + self.kernel_size]
                            
                            output[b, c_out, i, j] = (
                                np.sum(region * self.weight[c_out]) + self.bias[c_out]
                            )
            
            self._cache = x
            return output
    
    class MaxPool2D:
        """Max pooling layer."""
        
        def __init__(self, kernel_size: int = 2, stride: int = 2):
            self.kernel_size = kernel_size
            self.stride = stride
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            batch_size, channels, H, W = x.shape
            H_out = (H - self.kernel_size) // self.stride + 1
            W_out = (W - self.kernel_size) // self.stride + 1
            
            output = np.zeros((batch_size, channels, H_out, W_out))
            
            for b in range(batch_size):
                for c in range(channels):
                    for i in range(H_out):
                        for j in range(W_out):
                            i_start = i * self.stride
                            j_start = j * self.stride
                            
                            region = x[b, c,
                                       i_start:i_start + self.kernel_size,
                                       j_start:j_start + self.kernel_size]
                            
                            output[b, c, i, j] = np.max(region)
            
            return output
    
    # Test
    np.random.seed(42)
    
    # Create simple "image" batch
    x = np.random.randn(2, 1, 8, 8)  # 2 images, 1 channel, 8x8
    
    conv = Conv2D(1, 4, kernel_size=3, padding=1)
    pool = MaxPool2D(2)
    
    conv_out = conv.forward(x)
    pool_out = pool.forward(conv_out)
    
    print(f"\nInput shape: {x.shape}")
    print(f"After conv (3x3, pad=1): {conv_out.shape}")
    print(f"After max pool (2x2): {pool_out.shape}")
    
    # Output dimensions
    print(f"\nOutput dimension formula:")
    print(f"  H_out = (H + 2*P - K) / S + 1")
    print(f"  Conv: (8 + 2*1 - 3) / 1 + 1 = 8")
    print(f"  Pool: (8 - 2) / 2 + 1 = 4")


# =============================================================================
# Example 3: LSTM Cell
# =============================================================================

def example_lstm():
    """
    Long Short-Term Memory implementation.
    """
    print("\n" + "=" * 70)
    print("Example 3: LSTM Cell")
    print("=" * 70)
    
    class LSTMCell:
        """Single LSTM cell."""
        
        def __init__(self, input_size: int, hidden_size: int):
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # Combined weights for efficiency
            # Gates: forget, input, output, cell candidate
            std = np.sqrt(1.0 / hidden_size)
            
            self.W_ih = std * np.random.randn(4 * hidden_size, input_size)
            self.W_hh = std * np.random.randn(4 * hidden_size, hidden_size)
            self.bias = np.zeros(4 * hidden_size)
            
            # Initialize forget gate bias to 1
            self.bias[hidden_size:2*hidden_size] = 1.0
        
        def forward(self, x: np.ndarray, h_prev: np.ndarray, 
                    c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            x: (batch, input_size)
            h_prev: (batch, hidden_size)
            c_prev: (batch, hidden_size)
            """
            batch_size = x.shape[0]
            H = self.hidden_size
            
            # Combined linear transformation
            gates = x @ self.W_ih.T + h_prev @ self.W_hh.T + self.bias
            
            # Split gates
            f = self._sigmoid(gates[:, 0:H])          # Forget
            i = self._sigmoid(gates[:, H:2*H])        # Input
            o = self._sigmoid(gates[:, 2*H:3*H])      # Output
            g = np.tanh(gates[:, 3*H:4*H])            # Cell candidate
            
            # Cell state update
            c = f * c_prev + i * g
            
            # Hidden state
            h = o * np.tanh(c)
            
            return h, c
        
        def _sigmoid(self, x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    class LSTM:
        """Full LSTM layer processing sequences."""
        
        def __init__(self, input_size: int, hidden_size: int):
            self.cell = LSTMCell(input_size, hidden_size)
            self.hidden_size = hidden_size
        
        def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple]:
            """
            x: (batch, seq_len, input_size)
            Returns: outputs (batch, seq_len, hidden_size), (h_n, c_n)
            """
            batch_size, seq_len, _ = x.shape
            
            h = np.zeros((batch_size, self.hidden_size))
            c = np.zeros((batch_size, self.hidden_size))
            
            outputs = []
            
            for t in range(seq_len):
                h, c = self.cell.forward(x[:, t, :], h, c)
                outputs.append(h)
            
            return np.stack(outputs, axis=1), (h, c)
    
    # Test
    np.random.seed(42)
    
    batch_size = 4
    seq_len = 10
    input_size = 8
    hidden_size = 16
    
    x = np.random.randn(batch_size, seq_len, input_size)
    
    lstm = LSTM(input_size, hidden_size)
    outputs, (h_n, c_n) = lstm.forward(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Final hidden state shape: {h_n.shape}")
    print(f"Final cell state shape: {c_n.shape}")
    
    # Analyze gate activations
    cell = lstm.cell
    h = np.zeros((batch_size, hidden_size))
    c = np.zeros((batch_size, hidden_size))
    
    gates = x[:, 0] @ cell.W_ih.T + h @ cell.W_hh.T + cell.bias
    H = hidden_size
    f = 1 / (1 + np.exp(-gates[:, 0:H]))
    i = 1 / (1 + np.exp(-gates[:, H:2*H]))
    
    print(f"\nGate analysis (first timestep):")
    print(f"  Forget gate mean: {np.mean(f):.4f}")
    print(f"  Input gate mean: {np.mean(i):.4f}")


# =============================================================================
# Example 4: Self-Attention Mechanism
# =============================================================================

def example_self_attention():
    """
    Self-attention (transformer building block).
    """
    print("\n" + "=" * 70)
    print("Example 4: Self-Attention")
    print("=" * 70)
    
    class SelfAttention:
        """Scaled dot-product self-attention."""
        
        def __init__(self, d_model: int, d_k: int = None):
            self.d_model = d_model
            self.d_k = d_k if d_k else d_model
            
            std = np.sqrt(1.0 / d_model)
            self.W_Q = std * np.random.randn(d_model, self.d_k)
            self.W_K = std * np.random.randn(d_model, self.d_k)
            self.W_V = std * np.random.randn(d_model, self.d_k)
        
        def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
            """
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) or None
            """
            Q = x @ self.W_Q
            K = x @ self.W_K
            V = x @ self.W_V
            
            # Scaled dot-product attention
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
            
            if mask is not None:
                scores = np.where(mask, scores, -1e9)
            
            attn_weights = self._softmax(scores)
            output = attn_weights @ V
            
            return output, attn_weights
        
        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    class MultiHeadAttention:
        """Multi-head attention."""
        
        def __init__(self, d_model: int, n_heads: int):
            assert d_model % n_heads == 0
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            std = np.sqrt(1.0 / d_model)
            self.W_Q = std * np.random.randn(d_model, d_model)
            self.W_K = std * np.random.randn(d_model, d_model)
            self.W_V = std * np.random.randn(d_model, d_model)
            self.W_O = std * np.random.randn(d_model, d_model)
        
        def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
            batch_size, seq_len, _ = x.shape
            
            # Linear projections
            Q = x @ self.W_Q
            K = x @ self.W_K
            V = x @ self.W_V
            
            # Reshape for multi-head
            Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            
            # Attention
            scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
            
            if mask is not None:
                scores = np.where(mask[:, np.newaxis, :, :], scores, -1e9)
            
            attn = self._softmax(scores)
            context = attn @ V
            
            # Concatenate heads
            context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            
            # Output projection
            return context @ self.W_O
        
        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    # Test
    np.random.seed(42)
    
    batch_size = 2
    seq_len = 8
    d_model = 64
    n_heads = 4
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Single head
    single_attn = SelfAttention(d_model)
    single_out, attn_weights = single_attn.forward(x)
    
    print(f"\nSingle-head attention:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {single_out.shape}")
    print(f"  Attention weights: {attn_weights.shape}")
    
    # Multi-head
    multi_attn = MultiHeadAttention(d_model, n_heads)
    multi_out = multi_attn.forward(x)
    
    print(f"\nMulti-head attention ({n_heads} heads):")
    print(f"  Output: {multi_out.shape}")
    
    # Causal mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1) == 0
    causal_mask = np.broadcast_to(causal_mask, (batch_size, seq_len, seq_len))
    
    _, masked_weights = single_attn.forward(x, mask=causal_mask)
    print(f"\nCausal attention pattern (position 0 attends to):")
    print(f"  {masked_weights[0, 0, :].round(3)}")
    print(f"Causal attention pattern (position 7 attends to):")
    print(f"  {masked_weights[0, 7, :].round(3)}")


# =============================================================================
# Example 5: Positional Encoding
# =============================================================================

def example_positional_encoding():
    """
    Sinusoidal positional encoding for transformers.
    """
    print("\n" + "=" * 70)
    print("Example 5: Positional Encoding")
    print("=" * 70)
    
    def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
        """
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def rotary_positional_embedding(x: np.ndarray, seq_len: int) -> np.ndarray:
        """RoPE: Rotary Position Embedding."""
        d = x.shape[-1]
        
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d, 2) * (-np.log(10000.0) / d))
        
        angles = position * div_term
        
        # Split into pairs and rotate
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        x_rot = np.zeros_like(x)
        x_rot[..., 0::2] = x1 * cos_angles - x2 * sin_angles
        x_rot[..., 1::2] = x1 * sin_angles + x2 * cos_angles
        
        return x_rot
    
    # Test sinusoidal
    seq_len = 100
    d_model = 64
    
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    
    print(f"\nSinusoidal Positional Encoding:")
    print(f"  Shape: {pe.shape}")
    print(f"  Range: [{pe.min():.2f}, {pe.max():.2f}]")
    
    # Properties
    print(f"\nProperties:")
    
    # PE can represent relative positions via dot product
    pos_10 = pe[10]
    pos_15 = pe[15]
    pos_20 = pe[20]
    pos_25 = pe[25]
    
    sim_5 = np.dot(pos_10, pos_15) / (np.linalg.norm(pos_10) * np.linalg.norm(pos_15))
    sim_5_alt = np.dot(pos_20, pos_25) / (np.linalg.norm(pos_20) * np.linalg.norm(pos_25))
    
    print(f"  Similarity(pos10, pos15): {sim_5:.4f}")
    print(f"  Similarity(pos20, pos25): {sim_5_alt:.4f}")
    print(f"  (Same relative distance → similar dot product)")
    
    # Test RoPE
    x = np.random.randn(2, seq_len, d_model)
    x_rot = rotary_positional_embedding(x, seq_len)
    
    print(f"\nRotary Position Embedding:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_rot.shape}")


# =============================================================================
# Example 6: Batch Normalization
# =============================================================================

def example_batch_norm():
    """
    Batch normalization implementation.
    """
    print("\n" + "=" * 70)
    print("Example 6: Batch Normalization")
    print("=" * 70)
    
    class BatchNorm1d:
        """Batch normalization for 1D inputs."""
        
        def __init__(self, num_features: int, eps: float = 1e-5, 
                     momentum: float = 0.1):
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)
            
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            
            self.training = True
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            if self.training:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
                
                # Update running stats
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            return self.gamma * x_norm + self.beta
    
    class LayerNorm:
        """Layer normalization."""
        
        def __init__(self, normalized_shape: int, eps: float = 1e-5):
            self.normalized_shape = normalized_shape
            self.eps = eps
            
            self.gamma = np.ones(normalized_shape)
            self.beta = np.zeros(normalized_shape)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            return self.gamma * x_norm + self.beta
    
    # Test
    np.random.seed(42)
    
    batch_size = 32
    features = 64
    
    x = 3.0 * np.random.randn(batch_size, features) + 5.0  # Mean=5, Std=3
    
    print(f"\nInput statistics:")
    print(f"  Mean: {np.mean(x):.4f}")
    print(f"  Std: {np.std(x):.4f}")
    
    # Batch norm
    bn = BatchNorm1d(features)
    x_bn = bn.forward(x)
    
    print(f"\nAfter Batch Normalization:")
    print(f"  Mean: {np.mean(x_bn):.4f}")
    print(f"  Std: {np.std(x_bn):.4f}")
    
    # Layer norm
    ln = LayerNorm(features)
    x_ln = ln.forward(x)
    
    print(f"\nAfter Layer Normalization:")
    print(f"  Per-sample mean: {np.mean(np.mean(x_ln, axis=-1)):.4f}")
    print(f"  Per-sample std: {np.mean(np.std(x_ln, axis=-1)):.4f}")
    
    # Comparison
    print(f"\nKey difference:")
    print(f"  BatchNorm normalizes across batch (same feature stats)")
    print(f"  LayerNorm normalizes across features (same sample stats)")


# =============================================================================
# Example 7: Dropout
# =============================================================================

def example_dropout():
    """
    Dropout regularization.
    """
    print("\n" + "=" * 70)
    print("Example 7: Dropout")
    print("=" * 70)
    
    class Dropout:
        """Dropout layer."""
        
        def __init__(self, p: float = 0.5):
            self.p = p
            self.training = True
            self._mask = None
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            if not self.training:
                return x
            
            self._mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self._mask
        
        def backward(self, grad_output: np.ndarray) -> np.ndarray:
            return grad_output * self._mask
    
    # Test
    np.random.seed(42)
    
    x = np.ones((100, 50))
    
    for p in [0.0, 0.2, 0.5, 0.8]:
        dropout = Dropout(p)
        
        # Multiple forward passes
        outputs = [dropout.forward(x.copy()) for _ in range(100)]
        
        mean_output = np.mean([np.mean(o) for o in outputs])
        std_output = np.std([np.mean(o) for o in outputs])
        zero_frac = np.mean([np.mean(o == 0) for o in outputs])
        
        print(f"\nDropout p={p}:")
        print(f"  Mean output: {mean_output:.4f} (expected: 1.0)")
        print(f"  Std across runs: {std_output:.4f}")
        print(f"  Zero fraction: {zero_frac:.2%} (expected: {p:.0%})")


# =============================================================================
# Example 8: Optimizers
# =============================================================================

def example_optimizers():
    """
    SGD, Momentum, and Adam optimizers.
    """
    print("\n" + "=" * 70)
    print("Example 8: Optimizers")
    print("=" * 70)
    
    class SGD:
        def __init__(self, params: List[np.ndarray], lr: float = 0.01):
            self.params = params
            self.lr = lr
        
        def step(self, grads: List[np.ndarray]):
            for param, grad in zip(self.params, grads):
                param -= self.lr * grad
    
    class SGDMomentum:
        def __init__(self, params: List[np.ndarray], lr: float = 0.01, 
                     momentum: float = 0.9):
            self.params = params
            self.lr = lr
            self.momentum = momentum
            self.velocities = [np.zeros_like(p) for p in params]
        
        def step(self, grads: List[np.ndarray]):
            for i, (param, grad) in enumerate(zip(self.params, grads)):
                self.velocities[i] = self.momentum * self.velocities[i] + grad
                param -= self.lr * self.velocities[i]
    
    class Adam:
        def __init__(self, params: List[np.ndarray], lr: float = 0.001,
                     beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
            self.params = params
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self.t = 0
        
        def step(self, grads: List[np.ndarray]):
            self.t += 1
            
            for i, (param, grad) in enumerate(zip(self.params, grads)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
                
                m_hat = self.m[i] / (1 - self.beta1**self.t)
                v_hat = self.v[i] / (1 - self.beta2**self.t)
                
                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
    
    # Test on Rosenbrock function
    def rosenbrock(x, y):
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_grad(x, y):
        dx = -2 * (1 - x) - 400 * x * (y - x**2)
        dy = 200 * (y - x**2)
        return np.array([dx, dy])
    
    np.random.seed(42)
    
    print(f"\nOptimizing Rosenbrock function (minimum at (1,1)):")
    
    for OptClass, name, lr in [(SGD, "SGD", 0.0001),
                                (SGDMomentum, "SGD+Momentum", 0.0001),
                                (Adam, "Adam", 0.01)]:
        x = np.array([-1.0, -1.0])
        params = [x]
        
        if name == "Adam":
            opt = OptClass(params, lr=lr)
        elif name == "SGD+Momentum":
            opt = OptClass(params, lr=lr, momentum=0.9)
        else:
            opt = OptClass(params, lr=lr)
        
        for i in range(10000):
            grad = rosenbrock_grad(x[0], x[1])
            opt.step([grad])
        
        final_loss = rosenbrock(x[0], x[1])
        print(f"  {name:15s}: x={x.round(4)}, loss={final_loss:.6f}")


# =============================================================================
# Example 9: Weight Initialization
# =============================================================================

def example_initialization():
    """
    Weight initialization strategies.
    """
    print("\n" + "=" * 70)
    print("Example 9: Weight Initialization")
    print("=" * 70)
    
    def xavier_uniform(shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot uniform initialization."""
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)
    
    def xavier_normal(shape: Tuple[int, int]) -> np.ndarray:
        """Xavier/Glorot normal initialization."""
        fan_in, fan_out = shape
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * std
    
    def he_normal(shape: Tuple[int, int]) -> np.ndarray:
        """He/Kaiming initialization for ReLU."""
        fan_in = shape[0]
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * std
    
    def orthogonal(shape: Tuple[int, int]) -> np.ndarray:
        """Orthogonal initialization."""
        a = np.random.randn(*shape)
        q, r = np.linalg.qr(a)
        return q
    
    # Compare signal propagation
    np.random.seed(42)
    
    layer_size = 256
    n_layers = 10
    
    print(f"\nSignal propagation through {n_layers} layers:")
    print(f"{'Init':15s} {'Mean':>12s} {'Std':>12s} {'Range':>15s}")
    print("-" * 55)
    
    for init_fn, name, activation in [
        (xavier_normal, "Xavier+tanh", np.tanh),
        (he_normal, "He+ReLU", lambda x: np.maximum(0, x)),
        (orthogonal, "Orthogonal", lambda x: x),
    ]:
        x = np.random.randn(32, layer_size)
        
        for _ in range(n_layers):
            W = init_fn((layer_size, layer_size))
            x = activation(x @ W)
        
        print(f"{name:15s} {np.mean(x):>12.4f} {np.std(x):>12.4f} "
              f"[{np.min(x):.2f}, {np.max(x):.2f}]")
    
    # Bad initialization
    x = np.random.randn(32, layer_size)
    for _ in range(n_layers):
        W = np.random.randn(layer_size, layer_size)  # No scaling!
        x = np.maximum(0, x @ W)
    
    print(f"{'Bad init':15s} {np.mean(x):>12.4f} {np.std(x):>12.4f} "
          f"[{np.min(x):.2f}, {np.max(x):.2f}]")


# =============================================================================
# Example 10: Complete Mini Transformer
# =============================================================================

def example_mini_transformer():
    """
    Complete transformer encoder block.
    """
    print("\n" + "=" * 70)
    print("Example 10: Mini Transformer Encoder")
    print("=" * 70)
    
    class TransformerEncoderLayer:
        """Single transformer encoder layer."""
        
        def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                     dropout: float = 0.1):
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            # Multi-head attention
            std = np.sqrt(1.0 / d_model)
            self.W_Q = std * np.random.randn(d_model, d_model)
            self.W_K = std * np.random.randn(d_model, d_model)
            self.W_V = std * np.random.randn(d_model, d_model)
            self.W_O = std * np.random.randn(d_model, d_model)
            
            # Feed-forward
            self.W1 = np.sqrt(2.0 / d_model) * np.random.randn(d_model, d_ff)
            self.b1 = np.zeros(d_ff)
            self.W2 = np.sqrt(2.0 / d_ff) * np.random.randn(d_ff, d_model)
            self.b2 = np.zeros(d_model)
            
            # Layer norm
            self.ln1_gamma = np.ones(d_model)
            self.ln1_beta = np.zeros(d_model)
            self.ln2_gamma = np.ones(d_model)
            self.ln2_beta = np.zeros(d_model)
            
            self.dropout = dropout
        
        def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, 
                        beta: np.ndarray) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta
        
        def _softmax(self, x: np.ndarray) -> np.ndarray:
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
        def _mha(self, x: np.ndarray) -> np.ndarray:
            batch_size, seq_len, _ = x.shape
            
            Q = x @ self.W_Q
            K = x @ self.W_K
            V = x @ self.W_V
            
            # Reshape for multi-head
            Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            
            scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
            attn = self._softmax(scores)
            context = attn @ V
            
            context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            return context @ self.W_O
        
        def _ffn(self, x: np.ndarray) -> np.ndarray:
            h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
            return h @ self.W2 + self.b2
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            # Self-attention with residual
            attn_out = self._mha(x)
            x = self._layer_norm(x + attn_out, self.ln1_gamma, self.ln1_beta)
            
            # Feed-forward with residual
            ff_out = self._ffn(x)
            x = self._layer_norm(x + ff_out, self.ln2_gamma, self.ln2_beta)
            
            return x
    
    # Test
    np.random.seed(42)
    
    batch_size = 4
    seq_len = 16
    d_model = 64
    n_heads = 4
    d_ff = 256
    
    x = np.random.randn(batch_size, seq_len, d_model)
    
    encoder = TransformerEncoderLayer(d_model, n_heads, d_ff)
    output = encoder.forward(x)
    
    print(f"\nTransformer Encoder Layer:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Check statistics
    print(f"\nOutput statistics (after LayerNorm):")
    print(f"  Mean: {np.mean(output):.4f}")
    print(f"  Std: {np.std(output):.4f}")
    
    # Parameter count
    n_params = (
        4 * d_model * d_model +  # Q, K, V, O
        d_model * d_ff + d_ff +  # FFN W1, b1
        d_ff * d_model + d_model +  # FFN W2, b2
        4 * d_model  # LayerNorm
    )
    print(f"\nTotal parameters: {n_params:,}")


def main():
    """Run all examples."""
    print("NEURAL NETWORKS - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    example_mlp()
    example_cnn()
    example_lstm()
    example_self_attention()
    example_positional_encoding()
    example_batch_norm()
    example_dropout()
    example_optimizers()
    example_initialization()
    example_mini_transformer()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
