"""
Attention Mechanisms - Exercises
================================

Practice exercises for understanding and implementing attention mechanisms.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable


# =============================================================================
# Exercise 1: Implement Scaled Dot-Product Attention
# =============================================================================

def exercise_scaled_attention():
    """
    Exercise: Implement scaled dot-product attention from scratch.
    
    Tasks:
    1. Implement softmax with numerical stability
    2. Implement scaled dot-product attention
    3. Handle optional masking
    4. Verify shapes and properties
    """
    print("=" * 70)
    print("Exercise 1: Scaled Dot-Product Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        pass
    
    def scaled_dot_product_attention(
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled Dot-Product Attention.
        
        Args:
            Q: Queries (..., n_q, d_k)
            K: Keys (..., n_k, d_k)
            V: Values (..., n_k, d_v)
            mask: Optional mask with -inf for masked positions
            
        Returns:
            output: (..., n_q, d_v)
            attention_weights: (..., n_q, n_k)
        """
        pass
    
    # Test
    Q = np.random.randn(2, 4, 8)  # batch=2, n_q=4, d_k=8
    K = np.random.randn(2, 6, 8)  # batch=2, n_k=6, d_k=8
    V = np.random.randn(2, 6, 8)  # batch=2, n_k=6, d_v=8


def solution_scaled_attention():
    """Reference solution for scaled dot-product attention."""
    print("\n--- Solution ---\n")
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def scaled_dot_product_attention(
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_k = K.shape[-1]
        
        # Compute scores
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores + mask
        
        # Softmax over keys
        attention_weights = softmax(scores, axis=-1)
        
        # Weighted sum of values
        output = attention_weights @ V
        
        return output, attention_weights
    
    # Test
    np.random.seed(42)
    Q = np.random.randn(2, 4, 8)
    K = np.random.randn(2, 6, 8)
    V = np.random.randn(2, 6, 8)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum (should be 1): {weights[0, 0].sum():.6f}")
    
    # Test with mask
    mask = np.zeros((2, 4, 6))
    mask[:, :, 4:] = -np.inf  # Mask last 2 positions
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    print(f"\nWith mask (last 2 positions masked):")
    print(f"Weights[0,0]: {np.round(weights_masked[0, 0], 4)}")
    print(f"Last 2 weights are ~0: {weights_masked[0, 0, 4:]}")


# =============================================================================
# Exercise 2: Multi-Head Attention
# =============================================================================

def exercise_multi_head():
    """
    Exercise: Implement multi-head attention.
    
    Tasks:
    1. Implement head splitting and combining
    2. Implement projection matrices
    3. Apply attention to each head
    4. Concatenate and project output
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Multi-Head Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class MultiHeadAttention:
        def __init__(self, d_model: int, n_heads: int, seed: int = 42):
            """
            Initialize multi-head attention.
            
            d_model: model dimension
            n_heads: number of attention heads
            """
            pass
        
        def split_heads(self, x: np.ndarray) -> np.ndarray:
            """
            Split last dimension into (n_heads, d_k).
            x: (batch, seq_len, d_model)
            returns: (batch, n_heads, seq_len, d_k)
            """
            pass
        
        def combine_heads(self, x: np.ndarray) -> np.ndarray:
            """
            Inverse of split_heads.
            x: (batch, n_heads, seq_len, d_k)
            returns: (batch, seq_len, d_model)
            """
            pass
        
        def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            """Forward pass."""
            pass


def solution_multi_head():
    """Reference solution for multi-head attention."""
    print("\n--- Solution ---\n")
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    class MultiHeadAttention:
        def __init__(self, d_model: int, n_heads: int, seed: int = 42):
            assert d_model % n_heads == 0
            np.random.seed(seed)
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            scale = np.sqrt(2.0 / d_model)
            self.W_q = np.random.randn(d_model, d_model) * scale
            self.W_k = np.random.randn(d_model, d_model) * scale
            self.W_v = np.random.randn(d_model, d_model) * scale
            self.W_o = np.random.randn(d_model, d_model) * scale
        
        def split_heads(self, x: np.ndarray) -> np.ndarray:
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            return x.transpose(0, 2, 1, 3)
        
        def combine_heads(self, x: np.ndarray) -> np.ndarray:
            batch_size, _, seq_len, _ = x.shape
            x = x.transpose(0, 2, 1, 3)
            return x.reshape(batch_size, seq_len, self.d_model)
        
        def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            # Project
            Q_proj = Q @ self.W_q
            K_proj = K @ self.W_k
            V_proj = V @ self.W_v
            
            # Split heads
            Q_heads = self.split_heads(Q_proj)
            K_heads = self.split_heads(K_proj)
            V_heads = self.split_heads(V_proj)
            
            # Attention
            scores = Q_heads @ K_heads.swapaxes(-2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores + mask
            
            attention_weights = softmax(scores, axis=-1)
            head_outputs = attention_weights @ V_heads
            
            # Combine and project
            concat = self.combine_heads(head_outputs)
            output = concat @ self.W_o
            
            return output, attention_weights
    
    # Test
    np.random.seed(42)
    batch_size, seq_len, d_model, n_heads = 2, 6, 64, 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    X = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = mha.forward(X, X, X)
    
    print(f"d_model: {d_model}, n_heads: {n_heads}, d_k: {mha.d_k}")
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")
    
    # Analyze attention patterns per head
    print("\nAttention entropy per head (first query):")
    for h in range(n_heads):
        w = weights[0, h, 0]
        entropy = -np.sum(w * np.log(w + 1e-10))
        print(f"  Head {h}: {entropy:.4f}")


# =============================================================================
# Exercise 3: Causal and Padding Masks
# =============================================================================

def exercise_attention_masks():
    """
    Exercise: Implement different attention masks.
    
    Tasks:
    1. Implement causal (look-ahead) mask
    2. Implement padding mask
    3. Combine masks
    4. Test with attention
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Attention Masks")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create lower-triangular causal mask.
        Returns mask with 0 for allowed, -inf for blocked.
        """
        pass
    
    def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
        """
        Create padding mask for variable-length sequences.
        
        lengths: actual length of each sequence in batch
        max_len: padded sequence length
        
        Returns: (batch, 1, 1, max_len) mask
        """
        pass
    
    def combine_masks(causal_mask: np.ndarray, 
                      padding_mask: np.ndarray) -> np.ndarray:
        """Combine causal and padding masks."""
        pass


def solution_attention_masks():
    """Reference solution for attention masks."""
    print("\n--- Solution ---\n")
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def create_causal_mask(seq_len: int) -> np.ndarray:
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        return np.where(mask == 1, -np.inf, 0)
    
    def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
        batch_size = len(lengths)
        mask = np.zeros((batch_size, max_len))
        for i, length in enumerate(lengths):
            mask[i, length:] = 1
        mask = np.where(mask == 1, -np.inf, 0)
        return mask[:, np.newaxis, np.newaxis, :]
    
    def combine_masks(causal_mask: np.ndarray, 
                      padding_mask: np.ndarray) -> np.ndarray:
        # Broadcasting handles shape differences
        return causal_mask + padding_mask
    
    # Test causal mask
    seq_len = 5
    causal = create_causal_mask(seq_len)
    print(f"Causal Mask (seq_len={seq_len}):")
    print(f"  (0=allow, -inf=block)")
    for i, row in enumerate(causal):
        symbols = ['.' if v == 0 else 'X' for v in row]
        print(f"  {i}: {''.join(symbols)}")
    
    # Test padding mask
    lengths = np.array([4, 3, 5])
    max_len = 5
    padding = create_padding_mask(lengths, max_len)
    print(f"\nPadding Mask (lengths={lengths.tolist()}):")
    for i, row in enumerate(padding[:, 0, 0, :]):
        symbols = ['.' if v == 0 else 'X' for v in row]
        print(f"  Batch {i}: {''.join(symbols)}")
    
    # Combined mask
    print("\nCombined Mask (causal + padding) for batch 1:")
    combined = combine_masks(causal, padding[1:2])
    for i, row in enumerate(combined[0, 0]):
        symbols = ['.' if v == 0 else 'X' for v in row]
        print(f"  {i}: {''.join(symbols)}")
    
    # Test with attention
    print("\nAttention with combined mask:")
    d_k = 4
    Q = np.random.randn(1, seq_len, d_k)
    K = np.random.randn(1, seq_len, d_k)
    V = np.random.randn(1, seq_len, d_k)
    
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    scores_masked = scores + combined
    weights = softmax(scores_masked, axis=-1)
    
    print(f"  Weights (batch 1, which has length 3):")
    for row in np.round(weights[0], 3):
        print(f"  {row}")


# =============================================================================
# Exercise 4: Positional Encodings
# =============================================================================

def exercise_positional_encoding():
    """
    Exercise: Implement different positional encodings.
    
    Tasks:
    1. Sinusoidal encoding (original Transformer)
    2. Learned positional embeddings
    3. Rotary position embedding (simplified)
    4. Compare properties
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Positional Encodings")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
        """
        Sinusoidal positional encoding.
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        """
        pass
    
    class LearnedPositionalEncoding:
        def __init__(self, max_len: int, d_model: int):
            pass
        
        def forward(self, seq_len: int) -> np.ndarray:
            pass
    
    def apply_rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Apply Rotary Position Embedding.
        Rotates pairs of dimensions based on position.
        """
        pass


def solution_positional_encoding():
    """Reference solution for positional encodings."""
    print("\n--- Solution ---\n")
    
    def sinusoidal_encoding(max_len: int, d_model: int) -> np.ndarray:
        PE = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        PE[:, 0::2] = np.sin(position * div_term)
        PE[:, 1::2] = np.cos(position * div_term)
        
        return PE
    
    class LearnedPositionalEncoding:
        def __init__(self, max_len: int, d_model: int, seed: int = 42):
            np.random.seed(seed)
            self.embeddings = np.random.randn(max_len, d_model) * 0.02
        
        def forward(self, seq_len: int) -> np.ndarray:
            return self.embeddings[:seq_len]
    
    def apply_rope(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Apply RoPE to input tensor."""
        seq_len, d = x.shape
        assert d % 2 == 0
        
        # Frequencies
        theta = 10000 ** (-2 * np.arange(d // 2) / d)
        angles = positions[:, np.newaxis] * theta[np.newaxis, :]
        
        # Reshape to pairs
        x_pairs = x.reshape(seq_len, d // 2, 2)
        
        # Apply rotation
        cos_a = np.cos(angles)[:, :, np.newaxis]
        sin_a = np.sin(angles)[:, :, np.newaxis]
        
        x_rot = np.zeros_like(x_pairs)
        x_rot[:, :, 0] = x_pairs[:, :, 0] * cos_a[:, :, 0] - x_pairs[:, :, 1] * sin_a[:, :, 0]
        x_rot[:, :, 1] = x_pairs[:, :, 0] * sin_a[:, :, 0] + x_pairs[:, :, 1] * cos_a[:, :, 0]
        
        return x_rot.reshape(seq_len, d)
    
    # Test sinusoidal
    max_len, d_model = 100, 32
    PE = sinusoidal_encoding(max_len, d_model)
    
    print(f"Sinusoidal Encoding (max_len={max_len}, d_model={d_model}):")
    print(f"  Shape: {PE.shape}")
    print(f"  PE[0, :8]: {np.round(PE[0, :8], 4)}")
    print(f"  PE[1, :8]: {np.round(PE[1, :8], 4)}")
    
    # Distance property
    print("\n  Cosine similarity between positions:")
    for i, j in [(0, 1), (0, 5), (0, 10), (0, 50)]:
        sim = np.dot(PE[i], PE[j]) / (np.linalg.norm(PE[i]) * np.linalg.norm(PE[j]))
        print(f"    pos {i} vs {j}: {sim:.4f}")
    
    # Test learned
    print("\nLearned Positional Encoding:")
    learned = LearnedPositionalEncoding(max_len, d_model)
    print(f"  Embeddings shape: {learned.embeddings.shape}")
    print(f"  Mean: {np.mean(learned.embeddings):.4f}")
    print(f"  Std: {np.std(learned.embeddings):.4f}")
    
    # Test RoPE
    print("\nRotary Position Embedding (RoPE):")
    seq_len, d = 5, 8
    x = np.random.randn(seq_len, d)
    positions = np.arange(seq_len)
    x_rope = apply_rope(x, positions)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {x_rope.shape}")
    print(f"  Norms preserved: {np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(x_rope, axis=1))}")


# =============================================================================
# Exercise 5: Attention Analysis
# =============================================================================

def exercise_attention_analysis():
    """
    Exercise: Analyze attention patterns.
    
    Tasks:
    1. Compute attention entropy
    2. Compute effective context size
    3. Analyze attention distances
    4. Detect pattern types
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Attention Analysis")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def attention_entropy(weights: np.ndarray) -> np.ndarray:
        """
        Compute entropy of attention distribution.
        Higher = more diffuse, Lower = more focused.
        """
        pass
    
    def effective_context_size(weights: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Count positions with weight > threshold."""
        pass
    
    def mean_attention_distance(weights: np.ndarray) -> np.ndarray:
        """Average distance of attended positions."""
        pass
    
    def detect_pattern_type(weights: np.ndarray) -> str:
        """
        Detect attention pattern type:
        - 'diagonal': attends mainly to self
        - 'local': attends to nearby positions
        - 'global': attends uniformly
        - 'peaked': attends to few specific positions
        """
        pass


def solution_attention_analysis():
    """Reference solution for attention analysis."""
    print("\n--- Solution ---\n")
    
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def attention_entropy(weights: np.ndarray) -> np.ndarray:
        return -np.sum(weights * np.log(weights + 1e-10), axis=-1)
    
    def effective_context_size(weights: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        return np.sum(weights > threshold, axis=-1)
    
    def mean_attention_distance(weights: np.ndarray) -> np.ndarray:
        seq_len = weights.shape[-1]
        positions = np.arange(seq_len)
        
        result = np.zeros(weights.shape[:-1])
        for i in range(seq_len):
            distances = np.abs(positions - i)
            result[..., i] = np.sum(weights[..., i, :] * distances, axis=-1)
        
        return result
    
    def detect_pattern_type(weights: np.ndarray) -> str:
        seq_len = weights.shape[-1]
        
        # Check diagonal (self-attention)
        diag = np.diag(weights)
        if np.mean(diag) > 0.5:
            return 'diagonal'
        
        # Check local (nearby positions)
        local_sum = 0
        for i in range(seq_len):
            start, end = max(0, i-2), min(seq_len, i+3)
            local_sum += np.sum(weights[i, start:end])
        if local_sum / seq_len > 0.7:
            return 'local'
        
        # Check global (uniform)
        entropy = attention_entropy(weights)
        max_entropy = np.log(seq_len)
        if np.mean(entropy) > 0.8 * max_entropy:
            return 'global'
        
        # Check peaked
        if np.mean(effective_context_size(weights, 0.1)) < 3:
            return 'peaked'
        
        return 'mixed'
    
    # Create different patterns
    np.random.seed(42)
    seq_len = 8
    
    # Diagonal pattern
    diag_weights = np.eye(seq_len) * 0.7 + np.ones((seq_len, seq_len)) * 0.3 / seq_len
    diag_weights = diag_weights / diag_weights.sum(axis=-1, keepdims=True)
    
    # Local pattern
    local_scores = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            local_scores[i, j] = -abs(i - j)
    local_weights = softmax(local_scores)
    
    # Global pattern
    global_weights = np.ones((seq_len, seq_len)) / seq_len
    
    # Peaked pattern
    peaked_scores = np.full((seq_len, seq_len), -10.0)
    peaked_scores[:, 0] = 0  # All attend to first position
    peaked_weights = softmax(peaked_scores)
    
    print("Pattern Analysis:")
    for name, weights in [
        ("Diagonal", diag_weights),
        ("Local", local_weights),
        ("Global", global_weights),
        ("Peaked", peaked_weights),
    ]:
        entropy = np.mean(attention_entropy(weights))
        eff_ctx = np.mean(effective_context_size(weights))
        mean_dist = np.mean(mean_attention_distance(weights))
        pattern = detect_pattern_type(weights)
        
        print(f"\n  {name}:")
        print(f"    Detected type: {pattern}")
        print(f"    Mean entropy: {entropy:.4f} (max={np.log(seq_len):.4f})")
        print(f"    Effective context: {eff_ctx:.2f}")
        print(f"    Mean distance: {mean_dist:.2f}")


# =============================================================================
# Exercise 6: Linear Attention
# =============================================================================

def exercise_linear_attention():
    """
    Exercise: Implement linear attention approximation.
    
    Tasks:
    1. Implement feature map for linear attention
    2. Compute attention in O(nd²) instead of O(n²d)
    3. Compare accuracy with standard attention
    4. Implement causal linear attention
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Linear Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def elu_feature(x: np.ndarray) -> np.ndarray:
        """ELU feature map: elu(x) + 1"""
        pass
    
    def linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         feature_fn: Callable = None) -> np.ndarray:
        """
        Linear attention using kernel feature maps.
        
        Attention(Q, K, V) ≈ φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K).sum(0))
        
        Complexity: O(n d²) instead of O(n² d)
        """
        pass
    
    def causal_linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                feature_fn: Callable = None) -> np.ndarray:
        """
        Causal linear attention using cumulative sums.
        
        For each position i, only attend to positions <= i.
        """
        pass


def solution_linear_attention():
    """Reference solution for linear attention."""
    print("\n--- Solution ---\n")
    
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def elu_feature(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x + 1, np.exp(x))
    
    def linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         feature_fn: Callable = None) -> np.ndarray:
        if feature_fn is None:
            feature_fn = elu_feature
        
        Q_prime = feature_fn(Q)
        K_prime = feature_fn(K)
        
        # K^T @ V: (d, d_v)
        KTV = K_prime.T @ V
        
        # Q @ (K^T @ V): (n, d_v)
        numerator = Q_prime @ KTV
        
        # Normalization
        K_sum = K_prime.sum(axis=0)
        denominator = Q_prime @ K_sum
        
        return numerator / (denominator[:, np.newaxis] + 1e-10)
    
    def causal_linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                feature_fn: Callable = None) -> np.ndarray:
        if feature_fn is None:
            feature_fn = elu_feature
        
        Q_prime = feature_fn(Q)
        K_prime = feature_fn(K)
        
        n, d = Q.shape
        d_v = V.shape[1]
        
        output = np.zeros((n, d_v))
        
        # Cumulative K^T @ V
        KTV_cumsum = np.zeros((d, d_v))
        K_sum_cumsum = np.zeros(d)
        
        for i in range(n):
            # Update cumulative sums
            KTV_cumsum += np.outer(K_prime[i], V[i])
            K_sum_cumsum += K_prime[i]
            
            # Compute output for position i
            numerator = Q_prime[i] @ KTV_cumsum
            denominator = Q_prime[i] @ K_sum_cumsum + 1e-10
            output[i] = numerator / denominator
        
        return output
    
    def standard_attention(Q, K, V):
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores)
        return weights @ V
    
    # Test
    np.random.seed(42)
    
    print("Linear Attention vs Standard:")
    for n in [32, 128, 512]:
        d = 32
        Q = np.random.randn(n, d) * 0.5
        K = np.random.randn(n, d) * 0.5
        V = np.random.randn(n, d)
        
        out_std = standard_attention(Q, K, V)
        out_lin = linear_attention(Q, K, V)
        
        mse = np.mean((out_std - out_lin) ** 2)
        
        std_ops = n * n * d * 2
        lin_ops = n * d * d * 2
        
        print(f"  n={n:3d}: MSE={mse:.6f}, Speedup={std_ops/lin_ops:.1f}x")
    
    # Test causal
    print("\nCausal Linear Attention:")
    n, d = 16, 8
    Q = np.random.randn(n, d) * 0.5
    K = np.random.randn(n, d) * 0.5
    V = np.random.randn(n, d)
    
    out_causal = causal_linear_attention(Q, K, V)
    
    # Compare with standard causal
    mask = np.triu(np.full((n, n), -np.inf), k=1)
    scores = Q @ K.T / np.sqrt(d) + mask
    weights = softmax(scores)
    out_std_causal = weights @ V
    
    mse_causal = np.mean((out_std_causal - out_causal) ** 2)
    print(f"  Causal MSE: {mse_causal:.6f}")


# =============================================================================
# Exercise 7: Sparse Attention Patterns
# =============================================================================

def exercise_sparse_attention():
    """
    Exercise: Implement sparse attention patterns.
    
    Tasks:
    1. Local (sliding window) attention
    2. Strided attention
    3. Longformer-style (local + global)
    4. Analyze sparsity and compute savings
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Sparse Attention Patterns")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def local_attention_mask(seq_len: int, window: int) -> np.ndarray:
        """Create local attention mask with given window size."""
        pass
    
    def strided_attention_mask(seq_len: int, stride: int) -> np.ndarray:
        """Create strided attention mask."""
        pass
    
    def longformer_mask(seq_len: int, window: int, 
                        global_indices: List[int]) -> np.ndarray:
        """Create Longformer-style attention mask."""
        pass
    
    def compute_sparsity(mask: np.ndarray) -> float:
        """Compute fraction of non-masked positions."""
        pass


def solution_sparse_attention():
    """Reference solution for sparse attention patterns."""
    print("\n--- Solution ---\n")
    
    def local_attention_mask(seq_len: int, window: int) -> np.ndarray:
        mask = np.full((seq_len, seq_len), -np.inf)
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            mask[i, start:end] = 0
        return mask
    
    def strided_attention_mask(seq_len: int, stride: int) -> np.ndarray:
        mask = np.full((seq_len, seq_len), -np.inf)
        for i in range(seq_len):
            mask[i, ::stride] = 0
            mask[i, i] = 0  # Always attend to self
        return mask
    
    def longformer_mask(seq_len: int, window: int, 
                        global_indices: List[int]) -> np.ndarray:
        mask = local_attention_mask(seq_len, window)
        
        for idx in global_indices:
            mask[idx, :] = 0
            mask[:, idx] = 0
        
        return mask
    
    def compute_sparsity(mask: np.ndarray) -> float:
        return np.mean(mask > -np.inf)
    
    # Test patterns
    seq_len = 16
    
    print("Sparse Attention Patterns:")
    
    patterns = [
        ("Local (w=2)", local_attention_mask(seq_len, 2)),
        ("Local (w=4)", local_attention_mask(seq_len, 4)),
        ("Strided (s=4)", strided_attention_mask(seq_len, 4)),
        ("Longformer (w=2, g=[0,8])", longformer_mask(seq_len, 2, [0, 8])),
    ]
    
    for name, mask in patterns:
        sparsity = compute_sparsity(mask)
        print(f"\n  {name}:")
        print(f"    Density: {sparsity:.2%}")
        print(f"    Pattern (first 8 rows):")
        for i in range(min(8, seq_len)):
            row = ['.' if v > -np.inf else ' ' for v in mask[i]]
            print(f"      {i}: {''.join(row)}")
    
    # Complexity comparison
    print("\nComplexity for seq_len=1024, d=64:")
    seq_len = 1024
    d = 64
    full_ops = seq_len * seq_len * d
    
    for name, sparsity in [
        ("Full", 1.0),
        ("Local (w=64)", (2 * 64 + 1) / seq_len),
        ("Strided (s=32)", seq_len / 32 / seq_len),
        ("Longformer (w=64, g=4)", ((2 * 64 + 1) + 4 * 2) / seq_len),
    ]:
        ops = full_ops * sparsity
        print(f"  {name:25s}: {sparsity:6.2%} density, {ops/1e6:6.2f}M ops")


# =============================================================================
# Exercise 8: Cross-Attention Implementation
# =============================================================================

def exercise_cross_attention():
    """
    Exercise: Implement cross-attention for encoder-decoder.
    
    Tasks:
    1. Implement cross-attention where Q from decoder, K/V from encoder
    2. Handle different sequence lengths
    3. Implement encoder-decoder attention with masking
    4. Show alignment interpretation
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Cross-Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def cross_attention(
        decoder_state: np.ndarray,
        encoder_output: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        encoder_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cross-attention from decoder to encoder.
        
        decoder_state: (batch, dec_len, d_model)
        encoder_output: (batch, enc_len, d_model)
        
        Returns: (context, attention_weights)
        """
        pass


def solution_cross_attention():
    """Reference solution for cross-attention."""
    print("\n--- Solution ---\n")
    
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def cross_attention(
        decoder_state: np.ndarray,
        encoder_output: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        encoder_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        d_k = W_q.shape[1]
        
        # Project
        Q = decoder_state @ W_q  # (batch, dec_len, d_k)
        K = encoder_output @ W_k  # (batch, enc_len, d_k)
        V = encoder_output @ W_v  # (batch, enc_len, d_v)
        
        # Attention scores
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)  # (batch, dec_len, enc_len)
        
        if encoder_mask is not None:
            scores = scores + encoder_mask
        
        # Weights and output
        attention_weights = softmax(scores, axis=-1)
        context = attention_weights @ V
        
        return context, attention_weights
    
    # Test
    np.random.seed(42)
    batch_size = 2
    enc_len = 6
    dec_len = 4
    d_model = 32
    d_k = 16
    
    encoder_output = np.random.randn(batch_size, enc_len, d_model)
    decoder_state = np.random.randn(batch_size, dec_len, d_model)
    
    W_q = np.random.randn(d_model, d_k) * 0.1
    W_k = np.random.randn(d_model, d_k) * 0.1
    W_v = np.random.randn(d_model, d_k) * 0.1
    
    context, weights = cross_attention(decoder_state, encoder_output, W_q, W_k, W_v)
    
    print(f"Cross-Attention (Encoder-Decoder):")
    print(f"  Encoder output: {encoder_output.shape}")
    print(f"  Decoder state: {decoder_state.shape}")
    print(f"  Context: {context.shape}")
    print(f"  Weights: {weights.shape}")
    
    print(f"\n  Attention weights (batch 0):")
    print(f"  Rows=decoder positions, Cols=encoder positions")
    for i, row in enumerate(weights[0]):
        print(f"    Dec {i} → Enc: {np.round(row, 3)}")
    
    # With encoder padding mask
    print("\n  With encoder mask (positions 4,5 masked):")
    encoder_mask = np.zeros((batch_size, 1, enc_len))
    encoder_mask[:, :, 4:] = -np.inf
    
    context_masked, weights_masked = cross_attention(
        decoder_state, encoder_output, W_q, W_k, W_v, encoder_mask
    )
    
    print(f"  Masked weights (batch 0, dec pos 0):")
    print(f"    {np.round(weights_masked[0, 0], 3)}")
    print(f"  Positions 4,5 have ~0 weight")


# =============================================================================
# Exercise 9: Relative Position Attention
# =============================================================================

def exercise_relative_attention():
    """
    Exercise: Implement relative position attention.
    
    Tasks:
    1. Add relative position bias to attention
    2. Implement ALiBi (Attention with Linear Biases)
    3. Compare absolute vs relative position
    """
    print("\n" + "=" * 70)
    print("Exercise 9: Relative Position Attention")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def relative_position_bias(seq_len: int, n_heads: int) -> np.ndarray:
        """
        Create learnable relative position bias.
        Returns: (n_heads, seq_len, seq_len)
        """
        pass
    
    def alibi_bias(seq_len: int, n_heads: int) -> np.ndarray:
        """
        ALiBi: Add linear bias based on distance.
        slope_h * |i - j| subtracted from attention scores.
        """
        pass


def solution_relative_attention():
    """Reference solution for relative position attention."""
    print("\n--- Solution ---\n")
    
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def relative_position_bias(seq_len: int, n_heads: int, seed: int = 42) -> np.ndarray:
        """Learnable relative position bias."""
        np.random.seed(seed)
        
        # Bias for each relative position (-seq_len+1 to seq_len-1)
        n_relative = 2 * seq_len - 1
        bias_table = np.random.randn(n_heads, n_relative) * 0.02
        
        # Create bias matrix
        bias = np.zeros((n_heads, seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                rel_pos = j - i + seq_len - 1  # Offset to positive index
                bias[:, i, j] = bias_table[:, rel_pos]
        
        return bias
    
    def alibi_bias(seq_len: int, n_heads: int) -> np.ndarray:
        """ALiBi: linear penalty for distance."""
        # Slopes for each head (geometric sequence)
        slopes = 2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)
        
        # Distance matrix
        positions = np.arange(seq_len)
        distances = np.abs(positions[:, np.newaxis] - positions[np.newaxis, :])
        
        # Apply slopes
        bias = -slopes[:, np.newaxis, np.newaxis] * distances[np.newaxis, :, :]
        
        return bias
    
    # Test
    seq_len = 8
    n_heads = 4
    d_k = 8
    
    # Relative position bias
    rel_bias = relative_position_bias(seq_len, n_heads)
    print(f"Relative Position Bias:")
    print(f"  Shape: {rel_bias.shape}")
    print(f"  Head 0, row 0: {np.round(rel_bias[0, 0], 3)}")
    print(f"  Note: Bias depends on (i-j), not absolute i,j")
    
    # ALiBi
    alibi = alibi_bias(seq_len, n_heads)
    print(f"\nALiBi Bias:")
    print(f"  Shape: {alibi.shape}")
    print(f"  Slopes: {2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)}")
    print(f"  Head 0, row 0: {np.round(alibi[0, 0], 2)}")
    print(f"  Head 0, row 4: {np.round(alibi[0, 4], 2)}")
    
    # Compare attention with/without ALiBi
    print("\nAttention With/Without ALiBi:")
    Q = np.random.randn(n_heads, seq_len, d_k)
    K = np.random.randn(n_heads, seq_len, d_k)
    
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    
    weights_no_alibi = softmax(scores, axis=-1)
    weights_alibi = softmax(scores + alibi, axis=-1)
    
    print(f"  Without ALiBi (head 0, pos 4):")
    print(f"    {np.round(weights_no_alibi[0, 4], 3)}")
    print(f"  With ALiBi (head 0, pos 4):")
    print(f"    {np.round(weights_alibi[0, 4], 3)}")
    print(f"  Note: ALiBi promotes attending to nearby positions")


# =============================================================================
# Exercise 10: Complete Attention Module
# =============================================================================

def exercise_attention_module():
    """
    Exercise: Build a complete, flexible attention module.
    
    Tasks:
    1. Support different attention types (self, cross)
    2. Support different position encodings
    3. Support sparse patterns
    4. Include gradient computation
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Attention Module")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class AttentionModule:
        """Complete attention module with multiple options."""
        
        def __init__(self, d_model: int, n_heads: int, 
                     attention_type: str = 'self',
                     position_encoding: str = 'none',
                     sparse_pattern: str = 'full'):
            pass
        
        def forward(self, x: np.ndarray, 
                    context: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
            pass


def solution_attention_module():
    """Reference solution for complete attention module."""
    print("\n--- Solution ---\n")
    
    def softmax(x, axis=-1):
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    class AttentionModule:
        """Complete attention module."""
        
        ATTENTION_TYPES = ['self', 'cross']
        POSITION_ENCODINGS = ['none', 'sinusoidal', 'alibi']
        SPARSE_PATTERNS = ['full', 'local', 'causal']
        
        def __init__(self, d_model: int, n_heads: int,
                     attention_type: str = 'self',
                     position_encoding: str = 'none',
                     sparse_pattern: str = 'full',
                     window_size: int = 64,
                     max_len: int = 512,
                     seed: int = 42):
            
            assert d_model % n_heads == 0
            np.random.seed(seed)
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.attention_type = attention_type
            self.position_encoding = position_encoding
            self.sparse_pattern = sparse_pattern
            self.window_size = window_size
            
            # Projections
            scale = np.sqrt(2.0 / d_model)
            self.W_q = np.random.randn(d_model, d_model) * scale
            self.W_k = np.random.randn(d_model, d_model) * scale
            self.W_v = np.random.randn(d_model, d_model) * scale
            self.W_o = np.random.randn(d_model, d_model) * scale
            
            # Position encoding
            if position_encoding == 'sinusoidal':
                self.pos_enc = self._sinusoidal(max_len, d_model)
            elif position_encoding == 'alibi':
                self.alibi_slopes = 2 ** (-8 * np.arange(1, n_heads + 1) / n_heads)
        
        def _sinusoidal(self, max_len, d_model):
            PE = np.zeros((max_len, d_model))
            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            PE[:, 0::2] = np.sin(position * div_term)
            PE[:, 1::2] = np.cos(position * div_term)
            return PE
        
        def _get_sparse_mask(self, q_len, k_len):
            if self.sparse_pattern == 'full':
                return None
            
            mask = np.full((q_len, k_len), -np.inf)
            
            if self.sparse_pattern == 'local':
                for i in range(q_len):
                    start = max(0, i - self.window_size)
                    end = min(k_len, i + self.window_size + 1)
                    mask[i, start:end] = 0
            
            elif self.sparse_pattern == 'causal':
                for i in range(q_len):
                    mask[i, :i+1] = 0
            
            return mask
        
        def _get_alibi_bias(self, q_len, k_len):
            positions_q = np.arange(q_len)
            positions_k = np.arange(k_len)
            distances = np.abs(positions_q[:, np.newaxis] - positions_k[np.newaxis, :])
            return -self.alibi_slopes[:, np.newaxis, np.newaxis] * distances[np.newaxis, :, :]
        
        def forward(self, x: np.ndarray,
                    context: Optional[np.ndarray] = None,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            
            batch_size, q_len, _ = x.shape
            
            # Add positional encoding
            if self.position_encoding == 'sinusoidal':
                x = x + self.pos_enc[:q_len]
            
            # Determine K, V source
            if self.attention_type == 'self':
                kv_source = x
                k_len = q_len
            else:  # cross
                assert context is not None
                kv_source = context
                k_len = context.shape[1]
            
            # Project
            Q = x @ self.W_q
            K = kv_source @ self.W_k
            V = kv_source @ self.W_v
            
            # Reshape for heads
            Q = Q.reshape(batch_size, q_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            K = K.reshape(batch_size, k_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            V = V.reshape(batch_size, k_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            
            # Attention scores
            scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)
            
            # Add ALiBi bias
            if self.position_encoding == 'alibi':
                scores = scores + self._get_alibi_bias(q_len, k_len)
            
            # Add sparse mask
            sparse_mask = self._get_sparse_mask(q_len, k_len)
            if sparse_mask is not None:
                scores = scores + sparse_mask
            
            # Add custom mask
            if mask is not None:
                scores = scores + mask
            
            # Softmax and output
            weights = softmax(scores, axis=-1)
            output = weights @ V
            
            # Combine heads
            output = output.transpose(0, 2, 1, 3).reshape(batch_size, q_len, self.d_model)
            output = output @ self.W_o
            
            return output, weights
    
    # Test
    np.random.seed(42)
    batch_size = 2
    seq_len = 16
    d_model = 64
    n_heads = 4
    
    print("Testing Complete Attention Module:\n")
    
    # Self-attention with sinusoidal
    print("1. Self-attention with sinusoidal position encoding:")
    module = AttentionModule(d_model, n_heads, 'self', 'sinusoidal', 'full')
    x = np.random.randn(batch_size, seq_len, d_model)
    out, weights = module.forward(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    
    # Self-attention with causal + ALiBi
    print("\n2. Causal self-attention with ALiBi:")
    module = AttentionModule(d_model, n_heads, 'self', 'alibi', 'causal')
    out, weights = module.forward(x)
    print(f"   Weights (head 0, pos 8): {np.round(weights[0, 0, 8], 3)}")
    print(f"   Note: Only attends to positions 0-8, with nearby bias")
    
    # Cross-attention
    print("\n3. Cross-attention:")
    module = AttentionModule(d_model, n_heads, 'cross', 'none', 'full')
    encoder_out = np.random.randn(batch_size, 20, d_model)
    decoder_state = np.random.randn(batch_size, 8, d_model)
    out, weights = module.forward(decoder_state, context=encoder_out)
    print(f"   Encoder: {encoder_out.shape}, Decoder: {decoder_state.shape}")
    print(f"   Output: {out.shape}, Weights: {weights.shape}")
    
    # Local sparse attention
    print("\n4. Local sparse attention (window=3):")
    module = AttentionModule(d_model, n_heads, 'self', 'none', 'local', window_size=3)
    out, weights = module.forward(x)
    density = np.mean(weights[0, 0] > 0.001)
    print(f"   Effective density: {density:.2%}")


def main():
    """Run all exercises with solutions."""
    print("ATTENTION MECHANISMS - EXERCISES")
    print("=" * 70)
    
    # Exercise 1
    exercise_scaled_attention()
    solution_scaled_attention()
    
    # Exercise 2
    exercise_multi_head()
    solution_multi_head()
    
    # Exercise 3
    exercise_attention_masks()
    solution_attention_masks()
    
    # Exercise 4
    exercise_positional_encoding()
    solution_positional_encoding()
    
    # Exercise 5
    exercise_attention_analysis()
    solution_attention_analysis()
    
    # Exercise 6
    exercise_linear_attention()
    solution_linear_attention()
    
    # Exercise 7
    exercise_sparse_attention()
    solution_sparse_attention()
    
    # Exercise 8
    exercise_cross_attention()
    solution_cross_attention()
    
    # Exercise 9
    exercise_relative_attention()
    solution_relative_attention()
    
    # Exercise 10
    exercise_attention_module()
    solution_attention_module()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
