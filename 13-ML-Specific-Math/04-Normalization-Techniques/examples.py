"""
Normalization Techniques - Examples
===================================

Comprehensive examples demonstrating various normalization methods in ML.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# =============================================================================
# Example 1: Batch Normalization
# =============================================================================

def example_batch_normalization():
    """Demonstrate Batch Normalization for training and inference."""
    print("=" * 70)
    print("Example 1: Batch Normalization")
    print("=" * 70)
    
    class BatchNorm1d:
        """Batch Normalization for 1D data (dense layers)."""
        
        def __init__(self, num_features: int, eps: float = 1e-5, 
                     momentum: float = 0.1):
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            
            # Learnable parameters
            self.gamma = np.ones(num_features)
            self.beta = np.zeros(num_features)
            
            # Running statistics for inference
            self.running_mean = np.zeros(num_features)
            self.running_var = np.ones(num_features)
            
            # Cache for backward pass
            self.cache = {}
        
        def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
            """
            Forward pass.
            x: (batch_size, num_features)
            """
            if training:
                # Compute batch statistics
                batch_mean = np.mean(x, axis=0)
                batch_var = np.var(x, axis=0)
                
                # Update running statistics
                self.running_mean = (1 - self.momentum) * self.running_mean + \
                                    self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + \
                                   self.momentum * batch_var
                
                mean, var = batch_mean, batch_var
            else:
                mean, var = self.running_mean, self.running_var
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            # Cache for backward
            if training:
                self.cache = {
                    'x': x, 'x_norm': x_norm, 'mean': mean, 
                    'var': var, 'std': np.sqrt(var + self.eps)
                }
            
            return out
        
        def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Backward pass."""
            x = self.cache['x']
            x_norm = self.cache['x_norm']
            std = self.cache['std']
            m = x.shape[0]
            
            # Gradients for learnable parameters
            dgamma = np.sum(dout * x_norm, axis=0)
            dbeta = np.sum(dout, axis=0)
            
            # Gradient for input
            dx_norm = dout * self.gamma
            dvar = np.sum(dx_norm * (x - self.cache['mean']) * -0.5 * std**-3, axis=0)
            dmean = np.sum(dx_norm * -1/std, axis=0) + dvar * np.mean(-2 * (x - self.cache['mean']), axis=0)
            dx = dx_norm / std + dvar * 2 * (x - self.cache['mean']) / m + dmean / m
            
            return dx, dgamma, dbeta
    
    # Test
    np.random.seed(42)
    batch_size, features = 32, 64
    
    bn = BatchNorm1d(features)
    
    # Training mode
    x = np.random.randn(batch_size, features) * 3 + 5  # Non-zero mean, large var
    
    print(f"Input statistics:")
    print(f"  Mean: {np.mean(x, axis=0)[:5].round(3)}")
    print(f"  Var:  {np.var(x, axis=0)[:5].round(3)}")
    
    out_train = bn.forward(x, training=True)
    
    print(f"\nOutput statistics (training):")
    print(f"  Mean: {np.mean(out_train, axis=0)[:5].round(6)}")
    print(f"  Var:  {np.var(out_train, axis=0)[:5].round(6)}")
    
    # Multiple batches to accumulate running stats
    for _ in range(100):
        x_batch = np.random.randn(batch_size, features) * 3 + 5
        bn.forward(x_batch, training=True)
    
    print(f"\nRunning statistics after 100 batches:")
    print(f"  Running mean: {bn.running_mean[:5].round(3)}")
    print(f"  Running var:  {bn.running_var[:5].round(3)}")
    
    # Inference mode
    x_test = np.random.randn(1, features) * 3 + 5  # Single sample
    out_test = bn.forward(x_test, training=False)
    
    print(f"\nInference with batch_size=1:")
    print(f"  Works correctly (uses running stats)")


# =============================================================================
# Example 2: Layer Normalization
# =============================================================================

def example_layer_normalization():
    """Demonstrate Layer Normalization for sequence models."""
    print("\n" + "=" * 70)
    print("Example 2: Layer Normalization")
    print("=" * 70)
    
    class LayerNorm:
        """Layer Normalization."""
        
        def __init__(self, normalized_shape: Tuple[int, ...], eps: float = 1e-5):
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = normalized_shape
            self.eps = eps
            
            # Learnable parameters
            self.gamma = np.ones(normalized_shape)
            self.beta = np.zeros(normalized_shape)
            
            self.cache = {}
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            Normalizes over the last len(normalized_shape) dimensions.
            """
            # Compute normalization axes
            axes = tuple(range(-len(self.normalized_shape), 0))
            
            # Compute statistics
            mean = np.mean(x, axis=axes, keepdims=True)
            var = np.var(x, axis=axes, keepdims=True)
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            out = self.gamma * x_norm + self.beta
            
            self.cache = {'x': x, 'x_norm': x_norm, 'mean': mean, 'var': var}
            
            return out
    
    # Test with Transformer-style input
    np.random.seed(42)
    batch_size, seq_len, d_model = 2, 10, 64
    
    ln = LayerNorm(d_model)
    x = np.random.randn(batch_size, seq_len, d_model) * 2 + 1
    
    print(f"Input shape: {x.shape}")
    print(f"Input statistics (sample 0, position 0):")
    print(f"  Mean: {np.mean(x[0, 0]):.4f}")
    print(f"  Var:  {np.var(x[0, 0]):.4f}")
    
    out = ln.forward(x)
    
    print(f"\nOutput shape: {out.shape}")
    print(f"Output statistics (sample 0, position 0):")
    print(f"  Mean: {np.mean(out[0, 0]):.6f}")
    print(f"  Var:  {np.var(out[0, 0]):.6f}")
    
    # Each position normalized independently
    print(f"\nEach position normalized independently:")
    for pos in [0, 5, 9]:
        mean = np.mean(out[0, pos])
        var = np.var(out[0, pos])
        print(f"  Position {pos}: mean={mean:.6f}, var={var:.6f}")


# =============================================================================
# Example 3: Group Normalization
# =============================================================================

def example_group_normalization():
    """Demonstrate Group Normalization for CNNs with small batches."""
    print("\n" + "=" * 70)
    print("Example 3: Group Normalization")
    print("=" * 70)
    
    class GroupNorm:
        """Group Normalization for convolutional layers."""
        
        def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
            assert num_channels % num_groups == 0
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.channels_per_group = num_channels // num_groups
            self.eps = eps
            
            # Learnable parameters
            self.gamma = np.ones(num_channels)
            self.beta = np.zeros(num_channels)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            x: (N, C, H, W)
            """
            N, C, H, W = x.shape
            
            # Reshape into groups
            x_groups = x.reshape(N, self.num_groups, self.channels_per_group, H, W)
            
            # Compute statistics per group
            mean = np.mean(x_groups, axis=(2, 3, 4), keepdims=True)
            var = np.var(x_groups, axis=(2, 3, 4), keepdims=True)
            
            # Normalize
            x_norm = (x_groups - mean) / np.sqrt(var + self.eps)
            
            # Reshape back
            x_norm = x_norm.reshape(N, C, H, W)
            
            # Scale and shift (broadcast gamma and beta)
            out = self.gamma[np.newaxis, :, np.newaxis, np.newaxis] * x_norm + \
                  self.beta[np.newaxis, :, np.newaxis, np.newaxis]
            
            return out
    
    # Test
    np.random.seed(42)
    batch_size = 2
    num_channels = 32
    height, width = 8, 8
    num_groups = 8
    
    gn = GroupNorm(num_groups, num_channels)
    x = np.random.randn(batch_size, num_channels, height, width) * 3 + 2
    
    print(f"Input shape: {x.shape}")
    print(f"Number of groups: {num_groups}")
    print(f"Channels per group: {num_channels // num_groups}")
    
    out = gn.forward(x)
    
    print(f"\nInput statistics (sample 0):")
    print(f"  Mean: {np.mean(x[0]):.4f}")
    print(f"  Var:  {np.var(x[0]):.4f}")
    
    print(f"\nOutput statistics per group (sample 0):")
    out_groups = out[0].reshape(num_groups, -1)
    for g in range(min(4, num_groups)):
        mean = np.mean(out_groups[g])
        var = np.var(out_groups[g])
        print(f"  Group {g}: mean={mean:.6f}, var={var:.6f}")
    
    # Comparison with different group counts
    print(f"\nSpecial cases:")
    
    # G=1: Layer Norm
    gn_layer = GroupNorm(1, num_channels)
    out_layer = gn_layer.forward(x)
    print(f"  G=1 (Layer Norm): normalizes all {num_channels} channels together")
    
    # G=C: Instance Norm
    gn_instance = GroupNorm(num_channels, num_channels)
    out_instance = gn_instance.forward(x)
    print(f"  G=C (Instance Norm): normalizes each channel separately")


# =============================================================================
# Example 4: Instance Normalization
# =============================================================================

def example_instance_normalization():
    """Demonstrate Instance Normalization for style transfer."""
    print("\n" + "=" * 70)
    print("Example 4: Instance Normalization")
    print("=" * 70)
    
    class InstanceNorm:
        """Instance Normalization."""
        
        def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
            self.num_features = num_features
            self.eps = eps
            self.affine = affine
            
            if affine:
                self.gamma = np.ones(num_features)
                self.beta = np.zeros(num_features)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            x: (N, C, H, W)
            """
            N, C, H, W = x.shape
            
            # Compute statistics per (N, C) pair
            mean = np.mean(x, axis=(2, 3), keepdims=True)
            var = np.var(x, axis=(2, 3), keepdims=True)
            
            # Normalize
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            
            # Scale and shift
            if self.affine:
                out = self.gamma[np.newaxis, :, np.newaxis, np.newaxis] * x_norm + \
                      self.beta[np.newaxis, :, np.newaxis, np.newaxis]
            else:
                out = x_norm
            
            return out
    
    # Test
    np.random.seed(42)
    batch_size = 2
    num_channels = 16
    height, width = 32, 32
    
    inorm = InstanceNorm(num_channels)
    x = np.random.randn(batch_size, num_channels, height, width) * 5 + 3
    
    print(f"Input shape: {x.shape}")
    
    out = inorm.forward(x)
    
    print(f"\nOutput statistics per channel (sample 0):")
    for c in range(min(4, num_channels)):
        mean = np.mean(out[0, c])
        var = np.var(out[0, c])
        print(f"  Channel {c}: mean={mean:.6f}, var={var:.6f}")
    
    print(f"\nUse case: Style Transfer")
    print(f"  Instance Norm removes style-specific statistics")
    print(f"  Each feature map normalized independently")


# =============================================================================
# Example 5: RMS Normalization
# =============================================================================

def example_rms_normalization():
    """Demonstrate RMS Normalization (used in LLaMA, etc.)."""
    print("\n" + "=" * 70)
    print("Example 5: RMS Normalization")
    print("=" * 70)
    
    class RMSNorm:
        """Root Mean Square Layer Normalization."""
        
        def __init__(self, dim: int, eps: float = 1e-6):
            self.dim = dim
            self.eps = eps
            self.gamma = np.ones(dim)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass."""
            # Compute RMS
            rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
            
            # Normalize and scale
            return self.gamma * (x / rms)
    
    class LayerNormComparison:
        """Standard Layer Norm for comparison."""
        
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
    
    rms_norm = RMSNorm(dim)
    layer_norm = LayerNormComparison(dim)
    
    x = np.random.randn(batch_size, seq_len, dim)
    
    out_rms = rms_norm.forward(x)
    out_ln = layer_norm.forward(x)
    
    print(f"Input shape: {x.shape}")
    
    print(f"\nRMSNorm output statistics:")
    print(f"  Mean: {np.mean(out_rms[0, 0]):.6f}")
    print(f"  RMS:  {np.sqrt(np.mean(out_rms[0, 0]**2)):.6f}")
    
    print(f"\nLayerNorm output statistics:")
    print(f"  Mean: {np.mean(out_ln[0, 0]):.6f}")
    print(f"  Std:  {np.std(out_ln[0, 0]):.6f}")
    
    # Computational comparison
    print(f"\nComputational comparison:")
    print(f"  LayerNorm: mean + variance = 2 passes")
    print(f"  RMSNorm:   square mean only = 1 pass + no beta")
    print(f"  ~10-15% speedup in practice")
    
    # Show equivalence when mean is ~0
    x_centered = x - np.mean(x, axis=-1, keepdims=True)
    out_rms_centered = rms_norm.forward(x_centered)
    out_ln_centered = layer_norm.forward(x_centered)
    
    diff = np.mean(np.abs(out_rms_centered - out_ln_centered))
    print(f"\nDifference when input is centered: {diff:.6f}")


# =============================================================================
# Example 6: Weight Normalization
# =============================================================================

def example_weight_normalization():
    """Demonstrate Weight Normalization."""
    print("\n" + "=" * 70)
    print("Example 6: Weight Normalization")
    print("=" * 70)
    
    class WeightNormalizedLinear:
        """Linear layer with weight normalization."""
        
        def __init__(self, in_features: int, out_features: int):
            # Initialize direction and magnitude separately
            self.v = np.random.randn(out_features, in_features)
            self.g = np.linalg.norm(self.v, axis=1, keepdims=True)
            self.bias = np.zeros(out_features)
        
        @property
        def weight(self):
            """Compute normalized weight."""
            v_norm = np.linalg.norm(self.v, axis=1, keepdims=True)
            return self.g * self.v / (v_norm + 1e-8)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass."""
            return x @ self.weight.T + self.bias
    
    # Test
    np.random.seed(42)
    in_features, out_features = 64, 32
    batch_size = 8
    
    layer = WeightNormalizedLinear(in_features, out_features)
    x = np.random.randn(batch_size, in_features)
    
    print(f"Weight Normalization: w = g * v / ||v||")
    print(f"\nInitial state:")
    print(f"  v shape: {layer.v.shape}")
    print(f"  g shape: {layer.g.shape}")
    print(f"  Effective weight norms: {np.linalg.norm(layer.weight, axis=1)[:5].round(4)}")
    
    # Show that g controls magnitude
    layer.g *= 2
    print(f"\nAfter doubling g:")
    print(f"  Effective weight norms: {np.linalg.norm(layer.weight, axis=1)[:5].round(4)}")
    
    # Forward pass
    out = layer.forward(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    print(f"\nAdvantages:")
    print(f"  - Decouples direction (v) from magnitude (g)")
    print(f"  - No running statistics needed")
    print(f"  - Better conditioning for optimization")


# =============================================================================
# Example 7: Spectral Normalization
# =============================================================================

def example_spectral_normalization():
    """Demonstrate Spectral Normalization for GANs."""
    print("\n" + "=" * 70)
    print("Example 7: Spectral Normalization")
    print("=" * 70)
    
    class SpectralNormalizedLinear:
        """Linear layer with spectral normalization."""
        
        def __init__(self, in_features: int, out_features: int, n_power_iterations: int = 1):
            self.in_features = in_features
            self.out_features = out_features
            self.n_power_iterations = n_power_iterations
            
            # Initialize weight
            self.weight = np.random.randn(out_features, in_features) / np.sqrt(in_features)
            self.bias = np.zeros(out_features)
            
            # Initialize u, v for power iteration
            self.u = np.random.randn(out_features)
            self.u = self.u / np.linalg.norm(self.u)
        
        def spectral_norm(self) -> float:
            """Estimate spectral norm using power iteration."""
            u = self.u.copy()
            
            for _ in range(self.n_power_iterations):
                # v = W^T u / ||W^T u||
                v = self.weight.T @ u
                v = v / (np.linalg.norm(v) + 1e-12)
                
                # u = W v / ||W v||
                u = self.weight @ v
                u = u / (np.linalg.norm(u) + 1e-12)
            
            # Update stored u
            self.u = u
            
            # σ(W) ≈ u^T W v
            sigma = u @ self.weight @ v
            
            return sigma
        
        @property
        def normalized_weight(self):
            """Get spectrally normalized weight."""
            sigma = self.spectral_norm()
            return self.weight / sigma
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            """Forward pass with normalized weights."""
            W_sn = self.normalized_weight
            return x @ W_sn.T + self.bias
    
    # Test
    np.random.seed(42)
    in_features, out_features = 64, 32
    batch_size = 8
    
    layer = SpectralNormalizedLinear(in_features, out_features)
    
    print(f"Spectral Normalization: W_SN = W / σ(W)")
    
    # Compare spectral norms
    print(f"\nSpectral norm of original weight:")
    U, S, Vh = np.linalg.svd(layer.weight)
    print(f"  Exact (SVD): {S[0]:.4f}")
    print(f"  Estimated (power iter): {layer.spectral_norm():.4f}")
    
    # After normalization
    W_sn = layer.normalized_weight
    _, S_sn, _ = np.linalg.svd(W_sn)
    print(f"\nSpectral norm after normalization: {S_sn[0]:.4f}")
    
    # 1-Lipschitz property
    x1 = np.random.randn(batch_size, in_features)
    x2 = np.random.randn(batch_size, in_features)
    
    y1 = layer.forward(x1)
    y2 = layer.forward(x2)
    
    input_dist = np.linalg.norm(x1 - x2, axis=1)
    output_dist = np.linalg.norm(y1 - y2, axis=1)
    
    print(f"\nLipschitz constraint verification:")
    print(f"  ||f(x1) - f(x2)|| / ||x1 - x2|| ≤ 1")
    ratios = output_dist / (input_dist + 1e-10)
    print(f"  Ratios: {ratios[:5].round(4)}")
    print(f"  Max ratio: {ratios.max():.4f}")


# =============================================================================
# Example 8: Pre-Norm vs Post-Norm
# =============================================================================

def example_pre_post_norm():
    """Compare Pre-Norm and Post-Norm architectures."""
    print("\n" + "=" * 70)
    print("Example 8: Pre-Norm vs Post-Norm")
    print("=" * 70)
    
    class LayerNorm:
        def __init__(self, dim):
            self.gamma = np.ones(dim)
            self.beta = np.zeros(dim)
            self.eps = 1e-5
        
        def __call__(self, x):
            mean = np.mean(x, axis=-1, keepdims=True)
            var = np.var(x, axis=-1, keepdims=True)
            return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
    
    def feedforward(x, W1, W2, activation=lambda x: np.maximum(0, x)):
        """Simple feedforward sublayer."""
        return activation(x @ W1) @ W2
    
    class PostNormBlock:
        """Post-LN: LayerNorm(x + Sublayer(x))"""
        def __init__(self, dim, ff_dim):
            self.ln = LayerNorm(dim)
            self.W1 = np.random.randn(dim, ff_dim) * 0.02
            self.W2 = np.random.randn(ff_dim, dim) * 0.02
        
        def forward(self, x):
            residual = x
            out = feedforward(x, self.W1, self.W2)
            return self.ln(residual + out)
    
    class PreNormBlock:
        """Pre-LN: x + Sublayer(LayerNorm(x))"""
        def __init__(self, dim, ff_dim):
            self.ln = LayerNorm(dim)
            self.W1 = np.random.randn(dim, ff_dim) * 0.02
            self.W2 = np.random.randn(ff_dim, dim) * 0.02
        
        def forward(self, x):
            residual = x
            out = feedforward(self.ln(x), self.W1, self.W2)
            return residual + out
    
    # Test gradient flow
    np.random.seed(42)
    dim, ff_dim = 64, 256
    batch_size, seq_len = 2, 10
    
    x = np.random.randn(batch_size, seq_len, dim)
    
    # Stack multiple blocks
    n_layers = 6
    
    post_blocks = [PostNormBlock(dim, ff_dim) for _ in range(n_layers)]
    pre_blocks = [PreNormBlock(dim, ff_dim) for _ in range(n_layers)]
    
    print(f"Comparing {n_layers}-layer networks:")
    
    # Forward pass and track statistics
    out_post = x.copy()
    out_pre = x.copy()
    
    post_norms = [np.linalg.norm(x)]
    pre_norms = [np.linalg.norm(x)]
    
    for i in range(n_layers):
        out_post = post_blocks[i].forward(out_post)
        out_pre = pre_blocks[i].forward(out_pre)
        post_norms.append(np.linalg.norm(out_post))
        pre_norms.append(np.linalg.norm(out_pre))
    
    print(f"\nActivation norms through layers:")
    print(f"  Layer | Post-Norm | Pre-Norm")
    print(f"  ------|-----------|----------")
    for i in range(n_layers + 1):
        print(f"  {i:5d} | {post_norms[i]:9.4f} | {pre_norms[i]:9.4f}")
    
    print(f"\nKey differences:")
    print(f"  Post-Norm: Normalization after residual → can have gradient issues")
    print(f"  Pre-Norm:  Direct residual path → better gradient flow")
    print(f"  Pre-Norm often more stable for very deep models")


# =============================================================================
# Example 9: Adaptive Normalization (AdaIN, SPADE)
# =============================================================================

def example_adaptive_normalization():
    """Demonstrate Adaptive Instance Normalization."""
    print("\n" + "=" * 70)
    print("Example 9: Adaptive Normalization")
    print("=" * 70)
    
    class AdaIN:
        """Adaptive Instance Normalization for style transfer."""
        
        def __init__(self, eps: float = 1e-5):
            self.eps = eps
        
        def instance_stats(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Compute instance-wise mean and std."""
            # x: (N, C, H, W)
            mean = np.mean(x, axis=(2, 3), keepdims=True)
            std = np.std(x, axis=(2, 3), keepdims=True) + self.eps
            return mean, std
        
        def forward(self, content: np.ndarray, style: np.ndarray) -> np.ndarray:
            """
            Apply AdaIN: align content statistics to style statistics.
            
            content: (N, C, H, W) - content features
            style: (N, C, H, W) - style features
            """
            content_mean, content_std = self.instance_stats(content)
            style_mean, style_std = self.instance_stats(style)
            
            # Normalize content and apply style statistics
            content_norm = (content - content_mean) / content_std
            out = style_std * content_norm + style_mean
            
            return out
    
    # Test
    np.random.seed(42)
    batch_size = 1
    channels = 16
    height, width = 32, 32
    
    adain = AdaIN()
    
    # Simulated content and style features
    content = np.random.randn(batch_size, channels, height, width) * 1 + 0
    style = np.random.randn(batch_size, channels, height, width) * 3 + 5
    
    print(f"Content statistics (channel 0):")
    print(f"  Mean: {np.mean(content[0, 0]):.4f}")
    print(f"  Std:  {np.std(content[0, 0]):.4f}")
    
    print(f"\nStyle statistics (channel 0):")
    print(f"  Mean: {np.mean(style[0, 0]):.4f}")
    print(f"  Std:  {np.std(style[0, 0]):.4f}")
    
    output = adain.forward(content, style)
    
    print(f"\nOutput statistics (channel 0):")
    print(f"  Mean: {np.mean(output[0, 0]):.4f}")
    print(f"  Std:  {np.std(output[0, 0]):.4f}")
    print(f"  → Matches style statistics!")
    
    # Conditional BatchNorm
    print(f"\n\nConditional Batch Normalization:")
    print(f"  γ(c), β(c) depend on class c")
    print(f"  Used in class-conditional generation")


# =============================================================================
# Example 10: Comprehensive Normalization Comparison
# =============================================================================

def example_normalization_comparison():
    """Compare all normalization methods."""
    print("\n" + "=" * 70)
    print("Example 10: Comprehensive Comparison")
    print("=" * 70)
    
    # Unified implementation for comparison
    def batch_norm(x, axis=(0,)):
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)
    
    def layer_norm(x, axis=(-1,)):
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)
    
    def instance_norm(x):
        # For (N, C, H, W)
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)
    
    def group_norm(x, num_groups):
        N, C, H, W = x.shape
        x = x.reshape(N, num_groups, C // num_groups, H, W)
        mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
        var = np.var(x, axis=(2, 3, 4), keepdims=True)
        x = (x - mean) / np.sqrt(var + 1e-5)
        return x.reshape(N, C, H, W)
    
    def rms_norm(x, axis=-1):
        rms = np.sqrt(np.mean(x ** 2, axis=axis, keepdims=True) + 1e-5)
        return x / rms
    
    # Create test data (CNN format: N, C, H, W)
    np.random.seed(42)
    N, C, H, W = 4, 32, 8, 8
    x = np.random.randn(N, C, H, W) * 5 + 10
    
    print(f"Input shape: {x.shape} (N={N}, C={C}, H={H}, W={W})")
    print(f"Input mean: {np.mean(x):.4f}, std: {np.std(x):.4f}")
    
    # Apply different normalizations
    results = {}
    
    # Batch Norm: normalize over N, H, W (keep C)
    results['BatchNorm'] = batch_norm(x, axis=(0, 2, 3))
    
    # Layer Norm: normalize over C, H, W (keep N)
    results['LayerNorm'] = layer_norm(x, axis=(1, 2, 3))
    
    # Instance Norm: normalize over H, W (keep N, C)
    results['InstanceNorm'] = instance_norm(x)
    
    # Group Norm: normalize over C/G, H, W (keep N, G)
    results['GroupNorm(G=8)'] = group_norm(x, num_groups=8)
    
    # RMS Norm: normalize over C, H, W
    results['RMSNorm'] = rms_norm(x, axis=(1, 2, 3))
    
    print(f"\nNormalization comparison:")
    print(f"{'Method':<20} | {'Mean':>10} | {'Std':>10} | Stats Computed Over")
    print("-" * 70)
    
    for name, out in results.items():
        mean = np.mean(out)
        std = np.std(out)
        
        if name == 'BatchNorm':
            scope = "N, H, W → per channel"
        elif name == 'LayerNorm':
            scope = "C, H, W → per sample"
        elif name == 'InstanceNorm':
            scope = "H, W → per (sample, channel)"
        elif name.startswith('GroupNorm'):
            scope = "C/G, H, W → per (sample, group)"
        else:
            scope = "C, H, W (RMS only)"
        
        print(f"{name:<20} | {mean:>10.6f} | {std:>10.6f} | {scope}")
    
    # Use case summary
    print(f"\n\nRecommended Use Cases:")
    print("-" * 50)
    use_cases = [
        ("BatchNorm", "CNN training with large batches (≥16)"),
        ("LayerNorm", "Transformers, RNNs, any batch size"),
        ("InstanceNorm", "Style transfer, image generation"),
        ("GroupNorm", "Object detection, small batch training"),
        ("RMSNorm", "LLMs (LLaMA, etc.) for efficiency"),
    ]
    for name, use_case in use_cases:
        print(f"  {name:<15}: {use_case}")


def main():
    """Run all examples."""
    print("NORMALIZATION TECHNIQUES - EXAMPLES")
    print("=" * 70)
    
    example_batch_normalization()
    example_layer_normalization()
    example_group_normalization()
    example_instance_normalization()
    example_rms_normalization()
    example_weight_normalization()
    example_spectral_normalization()
    example_pre_post_norm()
    example_adaptive_normalization()
    example_normalization_comparison()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
