"""
Normalization Techniques - Exercises
====================================

Practice exercises for implementing and understanding normalization methods.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Exercise 1: Implement Batch Normalization
# =============================================================================

def exercise_batch_normalization():
    """
    Exercise: Implement Batch Normalization from scratch.
    
    Tasks:
    1. Implement forward pass with running statistics
    2. Implement backward pass
    3. Handle training vs inference modes
    """
    print("=" * 70)
    print("Exercise 1: Batch Normalization Implementation")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class BatchNorm1d:
        def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
            """
            Initialize batch normalization layer.
            
            num_features: number of features (channels)
            eps: small constant for numerical stability
            momentum: momentum for running statistics
            """
            pass
        
        def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
            """
            Forward pass.
            x: (batch_size, num_features)
            """
            pass
        
        def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Backward pass.
            Returns: (dx, dgamma, dbeta)
            """
            pass


def solution_batch_normalization():
    """Reference solution for Batch Normalization."""
    print("\n--- Solution ---\n")
    
    class BatchNorm1d:
        def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            
            self.cache = {}
        
        def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
            if training:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            std = np.sqrt(var + self.eps)
            x_norm = (x - mean) / std
            out = self.gamma * x_norm + self.beta
            
            if training:
                self.cache = {'x': x, 'x_norm': x_norm, 'mean': mean, 'var': var, 'std': std}
            
            return out
        
        def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = self.cache['x']
            x_norm = self.cache['x_norm']
            mean = self.cache['mean']
            std = self.cache['std']
            m = x.shape[0]
            
            dgamma = np.sum(dout * x_norm, axis=0)
            dbeta = np.sum(dout, axis=0)
            
            dx_norm = dout * self.gamma
            dvar = np.sum(dx_norm * (x - mean) * -0.5 * std**-3, axis=0)
            dmean = np.sum(dx_norm * -1/std, axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
            dx = dx_norm / std + dvar * 2 * (x - mean) / m + dmean / m
            
            return dx, dgamma, dbeta
    
    # Test
    np.random.seed(42)
    batch_size, features = 32, 64
    
    bn = BatchNorm1d(features)
    x = np.random.randn(batch_size, features) * 5 + 10
    
    out = bn.forward(x, training=True)
    print(f"Input mean: {np.mean(x, axis=0)[:3].round(3)}")
    print(f"Output mean: {np.mean(out, axis=0)[:3].round(6)}")
    print(f"Output var: {np.var(out, axis=0)[:3].round(6)}")
    
    # Test backward
    dout = np.random.randn(batch_size, features)
    dx, dgamma, dbeta = bn.backward(dout)
    print(f"\nBackward pass:")
    print(f"  dx shape: {dx.shape}")
    print(f"  dgamma shape: {dgamma.shape}")
    
    # Numerical gradient check
    eps_check = 1e-5
    x_plus = x.copy()
    x_plus[0, 0] += eps_check
    x_minus = x.copy()
    x_minus[0, 0] -= eps_check
    
    bn_check = BatchNorm1d(features)
    bn_check.gamma = bn.gamma.copy()
    bn_check.beta = bn.beta.copy()
    
    out_plus = bn_check.forward(x_plus, training=True)
    out_minus = bn_check.forward(x_minus, training=True)
    
    numerical_grad = np.sum(dout * (out_plus - out_minus)) / (2 * eps_check)
    print(f"\n  Numerical gradient check (dx[0,0]):")
    print(f"    Analytical: {dx[0, 0]:.6f}")
    print(f"    Numerical:  {numerical_grad:.6f}")


# =============================================================================
# Exercise 2: Layer Normalization
# =============================================================================

def exercise_layer_normalization():
    """
    Exercise: Implement Layer Normalization.
    
    Tasks:
    1. Normalize over feature dimensions
    2. Support different normalized shapes
    3. Implement backward pass
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Layer Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class LayerNorm:
        def __init__(self, normalized_shape: Tuple[int, ...], eps: float = 1e-5):
            """
            normalized_shape: shape over which to normalize
            """
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass
        
        def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            pass


def solution_layer_normalization():
    """Reference solution for Layer Normalization."""
    print("\n--- Solution ---\n")
    
    class LayerNorm:
        def __init__(self, normalized_shape, eps: float = 1e-5):
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = normalized_shape
            self.eps = eps
            
            self.gamma = np.ones(normalized_shape)
            self.beta = np.zeros(normalized_shape)
            self.cache = {}
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            axes = tuple(range(-len(self.normalized_shape), 0))
            
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            std = np.sqrt(var + self.eps)
            
            x_norm = (x - mean) / std
            out = self.gamma * x_norm + self.beta
            
            self.cache = {'x': x, 'x_norm': x_norm, 'mean': mean, 'std': std, 'axes': axes}
            return out
        
        def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = self.cache['x']
            x_norm = self.cache['x_norm']
            std = self.cache['std']
            axes = self.cache['axes']
            
            # Number of elements in normalized dimensions
            n = np.prod([x.shape[ax] for ax in axes])
            
            dgamma = np.sum(dout * x_norm, axis=tuple(range(len(x.shape) - len(self.normalized_shape))))
            dbeta = np.sum(dout, axis=tuple(range(len(x.shape) - len(self.normalized_shape))))
            
            dx_norm = dout * self.gamma
            dx = (1.0 / n) * (1.0 / std) * (
                n * dx_norm - 
                np.sum(dx_norm, axis=axes, keepdims=True) - 
                x_norm * np.sum(dx_norm * x_norm, axis=axes, keepdims=True)
            )
            
            return dx, dgamma, dbeta
    
    # Test
    np.random.seed(42)
    batch_size, seq_len, d_model = 2, 5, 32
    
    ln = LayerNorm(d_model)
    x = np.random.randn(batch_size, seq_len, d_model) * 3 + 2
    
    out = ln.forward(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Normalized shape: {ln.normalized_shape}")
    
    print(f"\nPer-position statistics (sample 0):")
    for pos in range(seq_len):
        print(f"  Position {pos}: mean={np.mean(out[0, pos]):.6f}, var={np.var(out[0, pos]):.6f}")
    
    # Backward test
    dout = np.random.randn(batch_size, seq_len, d_model)
    dx, dgamma, dbeta = ln.backward(dout)
    print(f"\ndx shape: {dx.shape}")
    print(f"dgamma shape: {dgamma.shape}")


# =============================================================================
# Exercise 3: Group Normalization
# =============================================================================

def exercise_group_normalization():
    """
    Exercise: Implement Group Normalization.
    
    Tasks:
    1. Divide channels into groups
    2. Normalize within each group
    3. Show connection to LayerNorm (G=1) and InstanceNorm (G=C)
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Group Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class GroupNorm:
        def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """x: (N, C, H, W)"""
            pass


def solution_group_normalization():
    """Reference solution for Group Normalization."""
    print("\n--- Solution ---\n")
    
    class GroupNorm:
        def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
            assert num_channels % num_groups == 0
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            
            self.gamma = np.ones(num_channels)
            self.beta = np.zeros(num_channels)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            N, C, H, W = x.shape
            G = self.num_groups
            
            # Reshape: (N, G, C/G, H, W)
            x = x.reshape(N, G, C // G, H, W)
            
            # Compute stats over (C/G, H, W)
            mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
            var = np.var(x, axis=(2, 3, 4), keepdims=True)
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Reshape back
            x_norm = x_norm.reshape(N, C, H, W)
            
            # Apply gamma, beta
            out = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
            
            return out
    
    # Test
    np.random.seed(42)
    N, C, H, W = 2, 32, 8, 8
    x = np.random.randn(N, C, H, W) * 3 + 5
    
    print(f"Input shape: {x.shape}")
    
    # Different group sizes
    for G in [1, 4, 8, 32]:
        gn = GroupNorm(G, C)
        out = gn.forward(x)
        
        name = "LayerNorm" if G == 1 else "InstanceNorm" if G == C else f"GroupNorm(G={G})"
        
        # Check stats within groups
        out_groups = out[0].reshape(G, C // G, H, W)
        group_means = [np.mean(out_groups[g]) for g in range(G)]
        group_vars = [np.var(out_groups[g]) for g in range(G)]
        
        print(f"\n{name}:")
        print(f"  Groups: {G}, Channels/group: {C // G}")
        print(f"  Group means: {np.round(group_means[:4], 6)}")
        print(f"  Group vars:  {np.round(group_vars[:4], 6)}")


# =============================================================================
# Exercise 4: RMS Normalization
# =============================================================================

def exercise_rms_normalization():
    """
    Exercise: Implement RMS Normalization.
    
    Tasks:
    1. Implement RMSNorm without mean centering
    2. Compare to LayerNorm
    3. Analyze when they're equivalent
    """
    print("\n" + "=" * 70)
    print("Exercise 4: RMS Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class RMSNorm:
        def __init__(self, dim: int, eps: float = 1e-6):
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass


def solution_rms_normalization():
    """Reference solution for RMS Normalization."""
    print("\n--- Solution ---\n")
    
    class RMSNorm:
        def __init__(self, dim: int, eps: float = 1e-6):
            self.dim = dim
            self.eps = eps
            self.gamma = np.ones(dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return self.gamma * x / rms
    
    class LayerNorm:
        def __init__(self, dim: int, eps: float = 1e-6):
            self.dim = dim
            self.eps = eps
            self.gamma = np.ones(dim)
            self.beta = np.zeros(dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
    
    # Test
    np.random.seed(42)
    batch_size, seq_len, dim = 2, 10, 64
    
    rms = RMSNorm(dim)
    ln = LayerNorm(dim)
    
    # Case 1: Non-centered data
    x1 = np.random.randn(batch_size, seq_len, dim) * 2 + 5
    out_rms1 = rms.forward(x1)
    out_ln1 = ln.forward(x1)
    
    print(f"Case 1: Non-centered data (mean ≈ 5)")
    print(f"  RMSNorm mean: {np.mean(out_rms1[0, 0]):.4f}")
    print(f"  LayerNorm mean: {np.mean(out_ln1[0, 0]):.6f}")
    print(f"  Difference: {np.mean(np.abs(out_rms1 - out_ln1)):.4f}")
    
    # Case 2: Centered data
    x2 = np.random.randn(batch_size, seq_len, dim) * 2
    out_rms2 = rms.forward(x2)
    out_ln2 = ln.forward(x2)
    
    print(f"\nCase 2: Centered data (mean ≈ 0)")
    print(f"  RMSNorm mean: {np.mean(out_rms2[0, 0]):.6f}")
    print(f"  LayerNorm mean: {np.mean(out_ln2[0, 0]):.6f}")
    print(f"  Difference: {np.mean(np.abs(out_rms2 - out_ln2)):.4f}")
    
    print(f"\nConclusion: RMSNorm ≈ LayerNorm when input is zero-centered")
    print(f"Advantage: RMSNorm saves the mean computation")


# =============================================================================
# Exercise 5: Spectral Normalization
# =============================================================================

def exercise_spectral_normalization():
    """
    Exercise: Implement Spectral Normalization.
    
    Tasks:
    1. Implement power iteration for spectral norm
    2. Apply to weight matrix
    3. Verify 1-Lipschitz property
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Spectral Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def power_iteration(W: np.ndarray, u: np.ndarray, 
                        n_iterations: int = 1) -> Tuple[float, np.ndarray]:
        """
        Estimate spectral norm using power iteration.
        
        Returns: (sigma, updated_u)
        """
        pass
    
    def spectral_normalize(W: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Apply spectral normalization to weight matrix."""
        pass


def solution_spectral_normalization():
    """Reference solution for Spectral Normalization."""
    print("\n--- Solution ---\n")
    
    def power_iteration(W: np.ndarray, u: np.ndarray, 
                        n_iterations: int = 1) -> Tuple[float, np.ndarray]:
        for _ in range(n_iterations):
            v = W.T @ u
            v = v / (np.linalg.norm(v) + 1e-12)
            u = W @ v
            u = u / (np.linalg.norm(u) + 1e-12)
        
        sigma = u @ W @ v
        return sigma, u
    
    def spectral_normalize(W: np.ndarray, u: np.ndarray) -> np.ndarray:
        sigma, _ = power_iteration(W, u, n_iterations=1)
        return W / sigma
    
    # Test
    np.random.seed(42)
    out_features, in_features = 64, 32
    
    W = np.random.randn(out_features, in_features)
    u = np.random.randn(out_features)
    u = u / np.linalg.norm(u)
    
    # Exact spectral norm via SVD
    _, S, _ = np.linalg.svd(W)
    exact_sigma = S[0]
    
    # Power iteration estimate
    print(f"Spectral Norm Estimation:")
    for n_iter in [1, 5, 10, 20]:
        est_sigma, _ = power_iteration(W, u.copy(), n_iter)
        error = abs(est_sigma - exact_sigma) / exact_sigma * 100
        print(f"  {n_iter:2d} iterations: {est_sigma:.4f} (error: {error:.2f}%)")
    
    print(f"  Exact (SVD): {exact_sigma:.4f}")
    
    # Spectral normalization
    W_sn = spectral_normalize(W, u)
    _, S_sn, _ = np.linalg.svd(W_sn)
    print(f"\nAfter normalization, spectral norm: {S_sn[0]:.4f}")
    
    # Lipschitz verification
    print(f"\nLipschitz verification:")
    for _ in range(5):
        x1 = np.random.randn(in_features)
        x2 = np.random.randn(in_features)
        
        y1 = W_sn @ x1
        y2 = W_sn @ x2
        
        ratio = np.linalg.norm(y1 - y2) / np.linalg.norm(x1 - x2)
        print(f"  ||f(x1)-f(x2)|| / ||x1-x2|| = {ratio:.4f} ≤ 1")


# =============================================================================
# Exercise 6: Weight Normalization
# =============================================================================

def exercise_weight_normalization():
    """
    Exercise: Implement Weight Normalization.
    
    Tasks:
    1. Reparameterize weight as g * v / ||v||
    2. Compute gradients for g and v
    3. Compare with standard parameterization
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Weight Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class WeightNormLinear:
        def __init__(self, in_features: int, out_features: int):
            pass
        
        @property
        def weight(self):
            """Compute effective weight from g and v."""
            pass
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            pass


def solution_weight_normalization():
    """Reference solution for Weight Normalization."""
    print("\n--- Solution ---\n")
    
    class WeightNormLinear:
        def __init__(self, in_features: int, out_features: int, seed: int = 42):
            np.random.seed(seed)
            self.v = np.random.randn(out_features, in_features) * 0.01
            self.g = np.linalg.norm(self.v, axis=1, keepdims=True)
            self.bias = np.zeros(out_features)
        
        @property
        def weight(self):
            v_norm = np.linalg.norm(self.v, axis=1, keepdims=True)
            return self.g * self.v / (v_norm + 1e-8)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            return x @ self.weight.T + self.bias
    
    # Test
    np.random.seed(42)
    in_features, out_features = 32, 16
    batch_size = 8
    
    layer = WeightNormLinear(in_features, out_features)
    x = np.random.randn(batch_size, in_features)
    
    print(f"Weight Normalization: W = g * v / ||v||")
    print(f"\nParameters:")
    print(f"  v shape: {layer.v.shape}")
    print(f"  g shape: {layer.g.shape}")
    
    # Weight norms
    weight_norms = np.linalg.norm(layer.weight, axis=1)
    print(f"\nWeight row norms: {weight_norms[:5].round(6)}")
    print(f"  These equal g: {layer.g[:5, 0].round(6)}")
    
    # Scaling g
    original_g = layer.g.copy()
    layer.g *= 2
    new_weight_norms = np.linalg.norm(layer.weight, axis=1)
    print(f"\nAfter doubling g:")
    print(f"  Weight norms: {new_weight_norms[:5].round(6)}")
    
    # Verify direction unchanged
    layer.g = original_g
    direction_before = layer.weight / np.linalg.norm(layer.weight, axis=1, keepdims=True)
    layer.g *= 2
    direction_after = layer.weight / np.linalg.norm(layer.weight, axis=1, keepdims=True)
    
    print(f"\nDirection change: {np.max(np.abs(direction_before - direction_after)):.10f}")
    print(f"  g only affects magnitude, not direction")


# =============================================================================
# Exercise 7: Pre-Norm vs Post-Norm Analysis
# =============================================================================

def exercise_pre_post_norm():
    """
    Exercise: Analyze Pre-Norm vs Post-Norm.
    
    Tasks:
    1. Implement both architectures
    2. Compare gradient flow
    3. Analyze stability for deep networks
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Pre-Norm vs Post-Norm")
    print("=" * 70)
    
    # YOUR CODE HERE - Implement PostNorm and PreNorm blocks
    # Analyze gradient magnitudes through layers
    pass


def solution_pre_post_norm():
    """Reference solution for Pre-Norm vs Post-Norm."""
    print("\n--- Solution ---\n")
    
    def layer_norm(x, gamma=None, beta=None, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        if gamma is not None:
            x_norm = gamma * x_norm + beta
        return x_norm
    
    def feedforward(x, W1, W2):
        return np.maximum(0, x @ W1) @ W2  # ReLU activation
    
    def simulate_gradient_flow(n_layers, norm_type='post', dim=64, ff_dim=256):
        np.random.seed(42)
        
        # Initialize weights
        weights = []
        for _ in range(n_layers):
            W1 = np.random.randn(dim, ff_dim) * np.sqrt(2 / dim)
            W2 = np.random.randn(ff_dim, dim) * np.sqrt(2 / ff_dim)
            weights.append((W1, W2))
        
        # Forward pass
        x = np.random.randn(2, 10, dim)
        activations = [x]
        
        for i in range(n_layers):
            W1, W2 = weights[i]
            
            if norm_type == 'post':
                out = layer_norm(x + feedforward(x, W1, W2))
            else:  # pre
                out = x + feedforward(layer_norm(x), W1, W2)
            
            x = out
            activations.append(x)
        
        return [np.linalg.norm(a) for a in activations]
    
    # Compare
    n_layers = 12
    
    post_norms = simulate_gradient_flow(n_layers, 'post')
    pre_norms = simulate_gradient_flow(n_layers, 'pre')
    
    print(f"Activation Norms ({n_layers} layers):")
    print(f"{'Layer':>6} | {'Post-Norm':>12} | {'Pre-Norm':>12} | {'Ratio':>8}")
    print("-" * 50)
    for i in range(n_layers + 1):
        ratio = post_norms[i] / pre_norms[i] if pre_norms[i] > 0 else float('inf')
        print(f"{i:>6} | {post_norms[i]:>12.4f} | {pre_norms[i]:>12.4f} | {ratio:>8.4f}")
    
    print(f"\nAnalysis:")
    print(f"  Post-Norm: Normalization dampens residual signal")
    print(f"  Pre-Norm: Residual path preserved → better gradient flow")
    print(f"  Pre-Norm often preferred for very deep models")


# =============================================================================
# Exercise 8: Adaptive Instance Normalization (AdaIN)
# =============================================================================

def exercise_adaptive_norm():
    """
    Exercise: Implement Adaptive Instance Normalization.
    
    Tasks:
    1. Implement AdaIN for style transfer
    2. Implement Conditional BatchNorm
    3. Show style statistics transfer
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Adaptive Normalization")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class AdaIN:
        def __init__(self, eps: float = 1e-5):
            pass
        
        def forward(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
            """Transfer style statistics to content features."""
            pass


def solution_adaptive_norm():
    """Reference solution for Adaptive Normalization."""
    print("\n--- Solution ---\n")
    
    class AdaIN:
        def __init__(self, eps: float = 1e-5):
            self.eps = eps
        
        def forward(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
            # Compute instance statistics
            content_mean = np.mean(content, axis=(2, 3), keepdims=True)
            content_std = np.std(content, axis=(2, 3), keepdims=True) + self.eps
            
            style_mean = np.mean(style, axis=(2, 3), keepdims=True)
            style_std = np.std(style, axis=(2, 3), keepdims=True) + self.eps
            
            # Normalize content, apply style statistics
            content_norm = (content - content_mean) / content_std
            return style_std * content_norm + style_mean
    
    # Test
    np.random.seed(42)
    N, C, H, W = 1, 8, 16, 16
    
    adain = AdaIN()
    
    # Different content and style
    content = np.random.randn(N, C, H, W) * 1 + 0
    style = np.random.randn(N, C, H, W) * 3 + 5
    
    output = adain.forward(content, style)
    
    print(f"Statistics per channel:")
    print(f"{'Channel':>8} | {'Content μ':>10} | {'Style μ':>10} | {'Output μ':>10}")
    print("-" * 50)
    for c in range(min(4, C)):
        c_mean = np.mean(content[0, c])
        s_mean = np.mean(style[0, c])
        o_mean = np.mean(output[0, c])
        print(f"{c:>8} | {c_mean:>10.4f} | {s_mean:>10.4f} | {o_mean:>10.4f}")
    
    print(f"\nOutput inherits style statistics while preserving content structure")
    
    # Conditional BatchNorm
    print(f"\n\nConditional BatchNorm:")
    print(f"  y = γ(c) * norm(x) + β(c)")
    print(f"  where γ(c), β(c) are class-dependent")
    
    class ConditionalBatchNorm:
        def __init__(self, num_features, num_classes):
            self.gamma = np.random.randn(num_classes, num_features) * 0.02 + 1
            self.beta = np.random.randn(num_classes, num_features) * 0.02
        
        def forward(self, x, class_id):
            mean = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)
            x_norm = (x - mean) / np.sqrt(var + 1e-5)
            return self.gamma[class_id] * x_norm + self.beta[class_id]
    
    cbn = ConditionalBatchNorm(num_features=32, num_classes=10)
    x = np.random.randn(8, 32)
    
    out_class0 = cbn.forward(x, class_id=0)
    out_class1 = cbn.forward(x, class_id=1)
    
    print(f"\n  Same input, different classes:")
    print(f"    Class 0 mean: {np.mean(out_class0):.4f}")
    print(f"    Class 1 mean: {np.mean(out_class1):.4f}")


# =============================================================================
# Exercise 9: Numerical Stability Analysis
# =============================================================================

def exercise_numerical_stability():
    """
    Exercise: Analyze numerical stability of normalization.
    
    Tasks:
    1. Compare naive vs stable variance computation
    2. Analyze behavior with extreme values
    3. Test epsilon sensitivity
    """
    print("\n" + "=" * 70)
    print("Exercise 9: Numerical Stability")
    print("=" * 70)
    
    # YOUR CODE HERE
    pass


def solution_numerical_stability():
    """Reference solution for numerical stability analysis."""
    print("\n--- Solution ---\n")
    
    def naive_variance(x):
        """E[X²] - E[X]² (numerically unstable)"""
        return np.mean(x ** 2) - np.mean(x) ** 2
    
    def stable_variance(x):
        """Welford's algorithm (numerically stable)"""
        n = len(x)
        mean = 0
        M2 = 0
        for val in x:
            delta = val - mean
            mean += delta / (x.tolist().index(val) + 1)
            M2 += delta * (val - mean)
        return M2 / n if n > 1 else 0
    
    def numpy_variance(x):
        """NumPy's implementation"""
        return np.var(x)
    
    # Test with normal values
    print("Test 1: Normal values")
    x_normal = np.random.randn(1000)
    print(f"  Naive:  {naive_variance(x_normal):.10f}")
    print(f"  NumPy:  {numpy_variance(x_normal):.10f}")
    
    # Test with large offset
    print("\nTest 2: Large offset (x + 1e8)")
    x_offset = x_normal + 1e8
    print(f"  Naive:  {naive_variance(x_offset):.10f}")
    print(f"  NumPy:  {numpy_variance(x_offset):.10f}")
    print(f"  True:   {numpy_variance(x_normal):.10f}")
    
    # Epsilon sensitivity
    print("\nTest 3: Epsilon sensitivity")
    x_tiny_var = np.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10])
    var = np.var(x_tiny_var)
    
    for eps in [1e-5, 1e-8, 1e-12, 0]:
        try:
            result = 1 / np.sqrt(var + eps)
            print(f"  eps={eps}: 1/sqrt(var+eps) = {result:.6f}")
        except:
            print(f"  eps={eps}: Division error!")
    
    # Layer norm stability
    print("\nTest 4: LayerNorm with extreme values")
    
    def stable_layernorm(x, eps=1e-5):
        # Shift to prevent overflow in x²
        x_shifted = x - np.mean(x)
        var = np.mean(x_shifted ** 2)
        return (x - np.mean(x)) / np.sqrt(var + eps)
    
    x_large = np.array([1e38, 1e38 + 1, 1e38 - 1])
    
    try:
        naive_result = (x_large - np.mean(x_large)) / np.sqrt(np.var(x_large) + 1e-5)
        print(f"  Naive LayerNorm: {naive_result}")
    except:
        print(f"  Naive LayerNorm: OVERFLOW")
    
    stable_result = stable_layernorm(x_large)
    print(f"  Stable LayerNorm: {stable_result}")


# =============================================================================
# Exercise 10: Complete Normalization Module
# =============================================================================

def exercise_complete_module():
    """
    Exercise: Build a unified normalization module.
    
    Tasks:
    1. Support multiple normalization types
    2. Unified interface
    3. Automatic selection based on input
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Normalization Module")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class NormalizationLayer:
        """Unified normalization layer supporting multiple types."""
        
        SUPPORTED_TYPES = ['batch', 'layer', 'instance', 'group', 'rms']
        
        def __init__(self, norm_type: str, num_features: int, **kwargs):
            pass
        
        def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
            pass


def solution_complete_module():
    """Reference solution for complete normalization module."""
    print("\n--- Solution ---\n")
    
    class NormalizationLayer:
        SUPPORTED_TYPES = ['batch', 'layer', 'instance', 'group', 'rms']
        
        def __init__(self, norm_type: str, num_features: int, 
                     num_groups: int = 32, eps: float = 1e-5, 
                     momentum: float = 0.1, affine: bool = True):
            
            assert norm_type in self.SUPPORTED_TYPES
            self.norm_type = norm_type
            self.num_features = num_features
            self.num_groups = num_groups
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            
            # Learnable parameters
            if affine:
                self.gamma = np.ones(num_features)
                self.beta = np.zeros(num_features) if norm_type != 'rms' else None
            
            # Running stats for batch norm
            if norm_type == 'batch':
                self.running_mean = np.zeros(num_features)
                self.running_var = np.ones(num_features)
        
        def _batch_norm(self, x, training):
            # x: (N, C) or (N, C, H, W)
            if x.ndim == 2:
                axis = 0
            else:
                axis = (0, 2, 3)
            
            if training:
                mean = np.mean(x, axis=axis, keepdims=True)
                var = np.var(x, axis=axis, keepdims=True)
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*np.squeeze(mean)
                self.running_var = (1-self.momentum)*self.running_var + self.momentum*np.squeeze(var)
            else:
                mean = self.running_mean.reshape([1, -1] + [1]*(x.ndim-2))
                var = self.running_var.reshape([1, -1] + [1]*(x.ndim-2))
            
            return (x - mean) / np.sqrt(var + self.eps)
        
        def _layer_norm(self, x):
            # Normalize over last axis
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return (x - mean) / np.sqrt(var + self.eps)
        
        def _instance_norm(self, x):
            # x: (N, C, H, W)
            mean = np.mean(x, axis=(2, 3), keepdims=True)
            var = np.var(x, axis=(2, 3), keepdims=True)
            return (x - mean) / np.sqrt(var + self.eps)
        
        def _group_norm(self, x):
            N, C, H, W = x.shape
            G = min(self.num_groups, C)
            x = x.reshape(N, G, C // G, H, W)
            mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
            var = np.var(x, axis=(2, 3, 4), keepdims=True)
            x = (x - mean) / np.sqrt(var + self.eps)
            return x.reshape(N, C, H, W)
        
        def _rms_norm(self, x):
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            return x / rms
        
        def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
            # Normalize
            if self.norm_type == 'batch':
                x_norm = self._batch_norm(x, training)
            elif self.norm_type == 'layer':
                x_norm = self._layer_norm(x)
            elif self.norm_type == 'instance':
                x_norm = self._instance_norm(x)
            elif self.norm_type == 'group':
                x_norm = self._group_norm(x)
            else:  # rms
                x_norm = self._rms_norm(x)
            
            # Apply affine
            if self.affine:
                if x.ndim == 2:
                    out = self.gamma * x_norm + (self.beta if self.beta is not None else 0)
                elif x.ndim == 4:
                    gamma = self.gamma[None, :, None, None]
                    beta = self.beta[None, :, None, None] if self.beta is not None else 0
                    out = gamma * x_norm + beta
                else:  # 3D for transformers
                    out = self.gamma * x_norm + (self.beta if self.beta is not None else 0)
            else:
                out = x_norm
            
            return out
    
    # Test all types
    np.random.seed(42)
    
    print("Testing all normalization types:")
    print("-" * 60)
    
    # 2D input (dense layer)
    x_2d = np.random.randn(32, 64) * 5 + 10
    for norm_type in ['batch', 'layer', 'rms']:
        layer = NormalizationLayer(norm_type, 64)
        out = layer.forward(x_2d)
        print(f"  {norm_type:8s} (2D): mean={np.mean(out):.6f}, std={np.std(out):.4f}")
    
    # 4D input (conv layer)
    x_4d = np.random.randn(8, 32, 16, 16) * 5 + 10
    for norm_type in ['batch', 'instance', 'group']:
        layer = NormalizationLayer(norm_type, 32, num_groups=8)
        out = layer.forward(x_4d)
        print(f"  {norm_type:8s} (4D): mean={np.mean(out):.6f}, std={np.std(out):.4f}")
    
    # 3D input (transformer)
    x_3d = np.random.randn(2, 10, 64) * 5 + 10
    for norm_type in ['layer', 'rms']:
        layer = NormalizationLayer(norm_type, 64)
        out = layer.forward(x_3d)
        print(f"  {norm_type:8s} (3D): mean={np.mean(out):.6f}, std={np.std(out):.4f}")


def main():
    """Run all exercises with solutions."""
    print("NORMALIZATION TECHNIQUES - EXERCISES")
    print("=" * 70)
    
    exercise_batch_normalization()
    solution_batch_normalization()
    
    exercise_layer_normalization()
    solution_layer_normalization()
    
    exercise_group_normalization()
    solution_group_normalization()
    
    exercise_rms_normalization()
    solution_rms_normalization()
    
    exercise_spectral_normalization()
    solution_spectral_normalization()
    
    exercise_weight_normalization()
    solution_weight_normalization()
    
    exercise_pre_post_norm()
    solution_pre_post_norm()
    
    exercise_adaptive_norm()
    solution_adaptive_norm()
    
    exercise_numerical_stability()
    solution_numerical_stability()
    
    exercise_complete_module()
    solution_complete_module()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
