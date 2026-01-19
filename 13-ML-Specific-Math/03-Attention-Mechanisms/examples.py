"""
Attention Mechanisms - Examples
===============================

Comprehensive examples of attention mechanisms for deep learning.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Callable
import warnings


# =============================================================================
# Example 1: Basic Attention Mechanism
# =============================================================================

def example_basic_attention():
    """
    Implement basic attention with different scoring functions.
    """
    print("=" * 70)
    print("Example 1: Basic Attention Mechanism")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    # Dot-product attention score
    def dot_product_score(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """
        query: (d_k,) or (n_q, d_k)
        keys: (n_k, d_k)
        returns: (n_k,) or (n_q, n_k)
        """
        return query @ keys.T
    
    # Scaled dot-product score
    def scaled_dot_product_score(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        d_k = keys.shape[-1]
        return (query @ keys.T) / np.sqrt(d_k)
    
    # Additive (Bahdanau) attention score
    def additive_score(query: np.ndarray, keys: np.ndarray,
                       W_q: np.ndarray, W_k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        query: (d_q,)
        keys: (n_k, d_k)
        W_q: (d_hidden, d_q)
        W_k: (d_hidden, d_k)
        v: (d_hidden,)
        """
        # Transform query and keys
        q_transformed = W_q @ query  # (d_hidden,)
        k_transformed = keys @ W_k.T  # (n_k, d_hidden)
        
        # Add and apply tanh
        combined = np.tanh(q_transformed + k_transformed)  # (n_k, d_hidden)
        
        # Project to scalar
        return combined @ v  # (n_k,)
    
    # General attention score
    def general_score(query: np.ndarray, keys: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        query: (d_q,)
        keys: (n_k, d_k)
        W: (d_q, d_k)
        """
        return query @ W @ keys.T
    
    # Basic attention computation
    def attention(query: np.ndarray, keys: np.ndarray, values: np.ndarray,
                  score_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute attention output and weights.
        
        query: (d_k,)
        keys: (n_k, d_k)
        values: (n_k, d_v)
        
        returns: (output, weights)
        """
        scores = score_fn(query, keys)  # (n_k,)
        weights = softmax(scores)  # (n_k,)
        output = weights @ values  # (d_v,)
        return output, weights
    
    # Example: simple sentence attention
    np.random.seed(42)
    n_tokens = 5  # Number of tokens in sequence
    d_k = 4  # Key/Query dimension
    d_v = 3  # Value dimension
    
    # Create keys, values (from encoder)
    keys = np.random.randn(n_tokens, d_k)
    values = np.random.randn(n_tokens, d_v)
    
    # Query (from decoder)
    query = np.random.randn(d_k)
    
    print("\nInput Shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")
    
    # Compute attention with different scoring
    print("\nAttention with Different Scoring Functions:")
    
    output_dot, weights_dot = attention(query, keys, values, dot_product_score)
    print(f"\n  Dot Product:")
    print(f"    Weights: {np.round(weights_dot, 4)}")
    print(f"    Output: {np.round(output_dot, 4)}")
    
    output_scaled, weights_scaled = attention(query, keys, values, scaled_dot_product_score)
    print(f"\n  Scaled Dot Product:")
    print(f"    Weights: {np.round(weights_scaled, 4)}")
    print(f"    Output: {np.round(output_scaled, 4)}")
    
    # Additive attention
    d_hidden = 8
    W_q = np.random.randn(d_hidden, d_k) * 0.1
    W_k = np.random.randn(d_hidden, d_k) * 0.1
    v = np.random.randn(d_hidden) * 0.1
    
    additive_fn = lambda q, k: additive_score(q, k, W_q, W_k, v)
    output_add, weights_add = attention(query, keys, values, additive_fn)
    print(f"\n  Additive (Bahdanau):")
    print(f"    Weights: {np.round(weights_add, 4)}")
    print(f"    Output: {np.round(output_add, 4)}")
    
    return attention, softmax


# =============================================================================
# Example 2: Scaled Dot-Product Attention
# =============================================================================

def example_scaled_dot_product():
    """
    Implement scaled dot-product attention as used in Transformers.
    """
    print("\n" + "=" * 70)
    print("Example 2: Scaled Dot-Product Attention")
    print("=" * 70)
    
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
        """
        Scaled Dot-Product Attention.
        
        Q: Queries (..., n_q, d_k)
        K: Keys (..., n_k, d_k)
        V: Values (..., n_k, d_v)
        mask: Optional mask (..., n_q, n_k), -inf for masked positions
        
        Returns: (output, attention_weights)
        """
        d_k = K.shape[-1]
        
        # Compute attention scores
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)  # (..., n_q, n_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Softmax over keys
        attention_weights = softmax(scores, axis=-1)  # (..., n_q, n_k)
        
        # Weighted sum of values
        output = attention_weights @ V  # (..., n_q, d_v)
        
        return output, attention_weights
    
    # Example usage
    np.random.seed(42)
    batch_size = 2
    n_q = 4  # Query sequence length
    n_k = 6  # Key sequence length
    d_k = 8  # Key dimension
    d_v = 8  # Value dimension
    
    Q = np.random.randn(batch_size, n_q, d_k)
    K = np.random.randn(batch_size, n_k, d_k)
    V = np.random.randn(batch_size, n_k, d_v)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nScaled Dot-Product Attention:")
    print(f"  Q shape: {Q.shape}")
    print(f"  K shape: {K.shape}")
    print(f"  V shape: {V.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights sum (should be 1): {np.round(weights[0, 0].sum(), 4)}")
    
    # Demonstrate why scaling matters
    print("\nWhy Scaling Matters (d_k effect):")
    
    for d in [8, 64, 512]:
        q = np.random.randn(d)
        k = np.random.randn(d)
        
        dot = np.dot(q, k)
        scaled = dot / np.sqrt(d)
        
        print(f"  d_k={d:3d}: dot={dot:8.2f}, scaled={scaled:8.2f}")
    
    print("\n  Large d_k → large dot products → saturated softmax → tiny gradients")
    print("  Scaling keeps variance ~1, preventing saturation")
    
    return scaled_dot_product_attention


# =============================================================================
# Example 3: Multi-Head Attention
# =============================================================================

def example_multi_head_attention():
    """
    Implement multi-head attention from Transformer architecture.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multi-Head Attention")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    class MultiHeadAttention:
        """
        Multi-Head Attention module.
        
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
        """
        
        def __init__(self, d_model: int, n_heads: int, seed: int = 42):
            assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
            
            np.random.seed(seed)
            
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            self.d_v = d_model // n_heads
            
            # Initialize projection matrices
            scale = np.sqrt(2.0 / d_model)
            
            self.W_q = np.random.randn(d_model, d_model) * scale
            self.W_k = np.random.randn(d_model, d_model) * scale
            self.W_v = np.random.randn(d_model, d_model) * scale
            self.W_o = np.random.randn(d_model, d_model) * scale
        
        def split_heads(self, x: np.ndarray) -> np.ndarray:
            """
            Split the last dimension into (n_heads, d_k).
            x: (batch, seq_len, d_model)
            returns: (batch, n_heads, seq_len, d_k)
            """
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size, seq_len, self.n_heads, self.d_k)
            return x.transpose(0, 2, 1, 3)
        
        def combine_heads(self, x: np.ndarray) -> np.ndarray:
            """
            Inverse of split_heads.
            x: (batch, n_heads, seq_len, d_k)
            returns: (batch, seq_len, d_model)
            """
            batch_size, _, seq_len, _ = x.shape
            x = x.transpose(0, 2, 1, 3)
            return x.reshape(batch_size, seq_len, self.d_model)
        
        def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                    mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            """
            Forward pass of multi-head attention.
            
            Q, K, V: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len) or broadcastable
            
            Returns: (output, attention_weights)
            """
            batch_size = Q.shape[0]
            
            # Linear projections
            Q_proj = Q @ self.W_q  # (batch, seq_len, d_model)
            K_proj = K @ self.W_k
            V_proj = V @ self.W_v
            
            # Split into heads
            Q_heads = self.split_heads(Q_proj)  # (batch, n_heads, seq_len, d_k)
            K_heads = self.split_heads(K_proj)
            V_heads = self.split_heads(V_proj)
            
            # Scaled dot-product attention for each head
            scores = Q_heads @ K_heads.swapaxes(-2, -1) / np.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores + mask
            
            attention_weights = softmax(scores, axis=-1)
            
            # Weighted sum
            head_outputs = attention_weights @ V_heads  # (batch, n_heads, seq_len, d_v)
            
            # Combine heads
            concat = self.combine_heads(head_outputs)  # (batch, seq_len, d_model)
            
            # Final projection
            output = concat @ self.W_o
            
            return output, attention_weights
    
    # Example usage
    np.random.seed(42)
    batch_size = 2
    seq_len = 6
    d_model = 64
    n_heads = 8
    
    mha = MultiHeadAttention(d_model, n_heads)
    
    # Create input (self-attention: Q=K=V=X)
    X = np.random.randn(batch_size, seq_len, d_model)
    
    output, weights = mha.forward(X, X, X)
    
    print("\nMulti-Head Attention:")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_k (per head): {mha.d_k}")
    print(f"\n  Input shape: {X.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    
    # Analyze attention patterns
    print("\nAttention Pattern Analysis:")
    print(f"  First head, first batch:")
    print(f"  {np.round(weights[0, 0], 3)}")
    
    # Show each head captures different patterns
    print("\n  Entropy per head (first batch, first query):")
    for h in range(n_heads):
        w = weights[0, h, 0]  # First batch, head h, first query
        entropy = -np.sum(w * np.log(w + 1e-10))
        print(f"    Head {h}: entropy = {entropy:.4f}")
    
    return MultiHeadAttention


# =============================================================================
# Example 4: Self-Attention and Cross-Attention
# =============================================================================

def example_self_cross_attention():
    """
    Demonstrate self-attention and cross-attention patterns.
    """
    print("\n" + "=" * 70)
    print("Example 4: Self-Attention and Cross-Attention")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = softmax(scores, axis=-1)
        return weights @ V, weights
    
    np.random.seed(42)
    d_model = 16
    
    # Self-attention example
    print("\n1. Self-Attention:")
    print("   Q, K, V all come from the same sequence")
    
    seq_len = 5
    X = np.random.randn(seq_len, d_model)
    
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    output, weights = attention(Q, K, V)
    
    print(f"\n   Input sequence: {X.shape}")
    print(f"   Self-attention weights (how each position attends to others):")
    print(f"   {np.round(weights, 3)}")
    print(f"   Output: {output.shape}")
    
    # Cross-attention example
    print("\n2. Cross-Attention:")
    print("   Q from one sequence, K/V from another (encoder-decoder)")
    
    encoder_len = 4
    decoder_len = 3
    
    encoder_output = np.random.randn(encoder_len, d_model)  # Source
    decoder_state = np.random.randn(decoder_len, d_model)   # Target
    
    Q = decoder_state @ W_q  # Queries from decoder
    K = encoder_output @ W_k  # Keys from encoder
    V = encoder_output @ W_v  # Values from encoder
    
    output, weights = attention(Q, K, V)
    
    print(f"\n   Encoder (source): {encoder_output.shape}")
    print(f"   Decoder (target): {decoder_state.shape}")
    print(f"   Cross-attention weights (decoder → encoder):")
    print(f"   {np.round(weights, 3)}")
    print(f"   Shape: {weights.shape} (decoder_len × encoder_len)")
    
    # Interpretation
    print("\n   Interpretation:")
    print("   - Row i: which encoder positions decoder position i attends to")
    print("   - Useful for alignment in translation")
    
    return attention


# =============================================================================
# Example 5: Attention Masking
# =============================================================================

def example_attention_masking():
    """
    Implement padding and causal masking for attention.
    """
    print("\n" + "=" * 70)
    print("Example 5: Attention Masking")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def attention_with_mask(Q, K, V, mask=None):
        d_k = K.shape[-1]
        scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = softmax(scores, axis=-1)
        return weights @ V, weights
    
    # Causal mask (for autoregressive models)
    def create_causal_mask(seq_len: int) -> np.ndarray:
        """
        Create lower-triangular mask for causal attention.
        Position i can only attend to positions <= i.
        """
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)  # Upper triangular
        mask = np.where(mask == 1, -np.inf, 0)
        return mask
    
    # Padding mask
    def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
        """
        Create mask for padded sequences.
        
        lengths: actual lengths for each sequence in batch
        max_len: padded length
        
        Returns: (batch, 1, 1, max_len) mask
        """
        batch_size = len(lengths)
        mask = np.zeros((batch_size, max_len))
        for i, length in enumerate(lengths):
            mask[i, length:] = 1
        mask = np.where(mask == 1, -np.inf, 0)
        return mask[:, np.newaxis, np.newaxis, :]  # Broadcast shape
    
    np.random.seed(42)
    seq_len = 5
    d_k = 8
    
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    # No mask (bidirectional)
    print("\n1. No Mask (Bidirectional Attention):")
    _, weights_bi = attention_with_mask(Q, K, V)
    print(f"   Attention weights:")
    for row in np.round(weights_bi, 3):
        print(f"   {row}")
    
    # Causal mask
    print("\n2. Causal Mask (Autoregressive):")
    causal_mask = create_causal_mask(seq_len)
    print(f"   Causal mask (0=allow, -inf=block):")
    print(f"   {np.where(causal_mask == -np.inf, 'X', '0')}")
    
    _, weights_causal = attention_with_mask(Q, K, V, causal_mask)
    print(f"\n   Attention weights (causal):")
    for row in np.round(weights_causal, 3):
        print(f"   {row}")
    print("   Note: upper triangle is zero (can't attend to future)")
    
    # Padding mask
    print("\n3. Padding Mask:")
    batch_size = 2
    max_len = 6
    lengths = np.array([4, 3])  # Actual lengths
    
    Q_batch = np.random.randn(batch_size, max_len, d_k)
    K_batch = np.random.randn(batch_size, max_len, d_k)
    V_batch = np.random.randn(batch_size, max_len, d_k)
    
    padding_mask = create_padding_mask(lengths, max_len)
    print(f"   Lengths: {lengths}")
    print(f"   Padding mask shape: {padding_mask.shape}")
    print(f"   Batch 0 mask (pos 4,5 masked): {padding_mask[0, 0, 0]}")
    print(f"   Batch 1 mask (pos 3,4,5 masked): {padding_mask[1, 0, 0]}")
    
    _, weights_padded = attention_with_mask(Q_batch, K_batch, V_batch, padding_mask)
    print(f"\n   Batch 0, first query (attends to pos 0-3):")
    print(f"   {np.round(weights_padded[0, 0], 4)}")
    print(f"   Batch 1, first query (attends to pos 0-2):")
    print(f"   {np.round(weights_padded[1, 0], 4)}")
    
    return create_causal_mask, create_padding_mask


# =============================================================================
# Example 6: Positional Encoding
# =============================================================================

def example_positional_encoding():
    """
    Implement sinusoidal and learned positional encodings.
    """
    print("\n" + "=" * 70)
    print("Example 6: Positional Encoding")
    print("=" * 70)
    
    def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
        """
        Sinusoidal positional encoding from "Attention is All You Need".
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        PE = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        PE[:, 0::2] = np.sin(position * div_term)
        PE[:, 1::2] = np.cos(position * div_term)
        
        return PE
    
    class LearnedPositionalEncoding:
        """Learned positional embeddings."""
        
        def __init__(self, max_len: int, d_model: int, seed: int = 42):
            np.random.seed(seed)
            self.embeddings = np.random.randn(max_len, d_model) * 0.02
        
        def forward(self, seq_len: int) -> np.ndarray:
            return self.embeddings[:seq_len]
    
    # Rotary Position Embedding (RoPE) - simplified 2D version
    def rope_encoding(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Apply Rotary Position Embedding.
        x: (seq_len, d) where d is even
        positions: (seq_len,) position indices
        
        For each pair (x_{2i}, x_{2i+1}), apply rotation by position * theta_i
        """
        seq_len, d = x.shape
        theta = 10000 ** (-2 * np.arange(d // 2) / d)
        
        # Angles for each position
        angles = positions[:, np.newaxis] * theta[np.newaxis, :]  # (seq_len, d/2)
        
        # Reshape x into pairs
        x_pairs = x.reshape(seq_len, d // 2, 2)
        
        # Apply rotation to each pair
        cos_angles = np.cos(angles)[:, :, np.newaxis]
        sin_angles = np.sin(angles)[:, :, np.newaxis]
        
        # Rotation matrix [[cos, -sin], [sin, cos]]
        x_rotated = np.zeros_like(x_pairs)
        x_rotated[:, :, 0] = x_pairs[:, :, 0] * cos_angles[:, :, 0] - x_pairs[:, :, 1] * sin_angles[:, :, 0]
        x_rotated[:, :, 1] = x_pairs[:, :, 0] * sin_angles[:, :, 0] + x_pairs[:, :, 1] * cos_angles[:, :, 0]
        
        return x_rotated.reshape(seq_len, d)
    
    # Demonstrate sinusoidal encoding
    max_len = 100
    d_model = 64
    
    PE = sinusoidal_positional_encoding(max_len, d_model)
    
    print("\nSinusoidal Positional Encoding:")
    print(f"  Shape: {PE.shape}")
    print(f"  PE[0, :8]: {np.round(PE[0, :8], 4)}")
    print(f"  PE[1, :8]: {np.round(PE[1, :8], 4)}")
    
    # Show periodicity
    print("\n  Periodicity (dimension 0 has period 2π, dimension 6 has longer period):")
    print(f"    Dim 0: PE[0,0]={PE[0,0]:.4f}, PE[3,0]={PE[3,0]:.4f}, PE[6,0]={PE[6,0]:.4f}")
    print(f"    Dim 6: PE[0,6]={PE[0,6]:.4f}, PE[3,6]={PE[3,6]:.4f}, PE[6,6]={PE[6,6]:.4f}")
    
    # Relative position property
    print("\n  Relative Position Property:")
    print("  PE[pos+k] can be expressed as linear transform of PE[pos]")
    k = 5
    pos = 10
    
    # For each dimension pair, rotation by k
    for dim in [0, 2, 4]:
        freq = 10000 ** (-dim / d_model)
        rotation = np.array([
            [np.cos(k * freq), np.sin(k * freq)],
            [-np.sin(k * freq), np.cos(k * freq)]
        ])
        pe_pos = PE[pos, dim:dim+2]
        pe_pos_k_actual = PE[pos + k, dim:dim+2]
        pe_pos_k_computed = rotation @ pe_pos
        error = np.linalg.norm(pe_pos_k_actual - pe_pos_k_computed)
        print(f"    Dim {dim},{dim+1}: error = {error:.6f}")
    
    # Learned encoding
    print("\nLearned Positional Encoding:")
    learned_pe = LearnedPositionalEncoding(max_len, d_model)
    print(f"  Shape: {learned_pe.embeddings.shape}")
    print(f"  Note: Can't extrapolate beyond max_len")
    
    # RoPE demonstration
    print("\nRotary Position Embedding (RoPE):")
    seq_len = 5
    d = 8
    x = np.random.randn(seq_len, d)
    positions = np.arange(seq_len)
    
    x_rope = rope_encoding(x, positions)
    print(f"  Input shape: {x.shape}")
    print(f"  After RoPE: {x_rope.shape}")
    print("  Note: Encodes relative position in dot products")
    
    return sinusoidal_positional_encoding, rope_encoding


# =============================================================================
# Example 7: Attention Analysis and Visualization
# =============================================================================

def example_attention_analysis():
    """
    Analyze attention patterns and their properties.
    """
    print("\n" + "=" * 70)
    print("Example 7: Attention Analysis")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def attention_entropy(weights: np.ndarray) -> np.ndarray:
        """
        Compute entropy of attention distribution.
        Low entropy = focused, High entropy = diffuse.
        """
        return -np.sum(weights * np.log(weights + 1e-10), axis=-1)
    
    def attention_effective_context(weights: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Count positions with attention weight above threshold.
        """
        return np.sum(weights > threshold, axis=-1)
    
    def attention_mean_distance(weights: np.ndarray) -> np.ndarray:
        """
        Compute mean attended distance from each position.
        """
        seq_len = weights.shape[-1]
        positions = np.arange(seq_len)
        
        # For each query position, compute weighted average distance
        mean_dist = np.zeros(weights.shape[:-1])
        
        for i in range(seq_len):
            distances = np.abs(positions - i)
            mean_dist[..., i] = np.sum(weights[..., i, :] * distances, axis=-1)
        
        return mean_dist
    
    # Create attention patterns
    np.random.seed(42)
    seq_len = 10
    d_k = 16
    
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    
    # Different attention patterns
    scores = Q @ K.T / np.sqrt(d_k)
    weights_uniform = softmax(scores)
    
    # More focused attention (larger scores)
    weights_focused = softmax(scores * 2)
    
    # Very focused (almost hard)
    weights_hard = softmax(scores * 10)
    
    print("\nAttention Pattern Analysis:")
    
    for name, weights in [
        ("Normal", weights_uniform),
        ("Focused (×2)", weights_focused),
        ("Hard (×10)", weights_hard)
    ]:
        entropy = attention_entropy(weights)
        eff_context = attention_effective_context(weights)
        mean_dist = attention_mean_distance(weights)
        
        print(f"\n  {name}:")
        print(f"    Mean entropy: {np.mean(entropy):.4f}")
        print(f"    Mean effective context: {np.mean(eff_context):.2f} positions")
        print(f"    Mean attention distance: {np.mean(mean_dist):.2f}")
    
    # Analyze specific patterns
    print("\nSpecific Pattern Examples:")
    
    # Local attention (window)
    local_scores = np.full((seq_len, seq_len), -np.inf)
    window = 3
    for i in range(seq_len):
        start = max(0, i - window)
        end = min(seq_len, i + window + 1)
        local_scores[i, start:end] = 0
    weights_local = softmax(local_scores)
    
    print(f"\n  Local Attention (window={window}):")
    print(f"    Pattern (first 5 rows):")
    for row in np.round(weights_local[:5], 2):
        print(f"    {row}")
    
    # Causal attention analysis
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    weights_causal = softmax(scores + causal_mask)
    
    entropy_causal = attention_entropy(weights_causal)
    print(f"\n  Causal Attention Entropy by Position:")
    print(f"    {np.round(entropy_causal, 3)}")
    print("    Note: Earlier positions have lower entropy (fewer options)")
    
    return attention_entropy, attention_effective_context


# =============================================================================
# Example 8: Linear Attention
# =============================================================================

def example_linear_attention():
    """
    Implement linear attention for efficient long sequences.
    """
    print("\n" + "=" * 70)
    print("Example 8: Linear Attention")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Standard scaled dot-product attention.
        Complexity: O(n² d)
        """
        d_k = K.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)  # O(n² d)
        weights = softmax(scores, axis=-1)  # O(n²)
        return weights @ V  # O(n² d)
    
    def elu_feature(x: np.ndarray) -> np.ndarray:
        """ELU feature map for linear attention."""
        return np.where(x > 0, x + 1, np.exp(x))
    
    def linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         feature_fn: callable = None) -> np.ndarray:
        """
        Linear attention using feature maps.
        
        Instead of: softmax(QK^T)V
        Compute: φ(Q)(φ(K)^T V) using associativity
        
        Complexity: O(n d²) instead of O(n² d)
        """
        if feature_fn is None:
            feature_fn = elu_feature
        
        # Apply feature map
        Q_prime = feature_fn(Q)  # (n_q, d)
        K_prime = feature_fn(K)  # (n_k, d)
        
        # Compute K^T V first: O(n_k d d_v)
        KTV = K_prime.T @ V  # (d, d_v)
        
        # Then Q @ (K^T V): O(n_q d d_v)
        numerator = Q_prime @ KTV  # (n_q, d_v)
        
        # Normalization
        K_sum = K_prime.sum(axis=0)  # (d,)
        denominator = Q_prime @ K_sum  # (n_q,)
        
        return numerator / (denominator[:, np.newaxis] + 1e-10)
    
    def random_fourier_features(Q: np.ndarray, K: np.ndarray, 
                                n_features: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random Fourier Features approximation for softmax attention.
        exp(QK^T / √d) ≈ φ(Q)φ(K)^T
        """
        np.random.seed(seed)
        d_k = Q.shape[-1]
        
        # Random projection
        W = np.random.randn(d_k, n_features) / np.sqrt(d_k)
        
        # Feature map
        Q_proj = Q @ W  # (n_q, n_features)
        K_proj = K @ W  # (n_k, n_features)
        
        # Approximate softmax kernel
        Q_features = np.concatenate([np.cos(Q_proj), np.sin(Q_proj)], axis=-1) / np.sqrt(n_features)
        K_features = np.concatenate([np.cos(K_proj), np.sin(K_proj)], axis=-1) / np.sqrt(n_features)
        
        return Q_features, K_features
    
    # Compare standard vs linear attention
    np.random.seed(42)
    
    print("\nComplexity Comparison:")
    for seq_len in [64, 256, 1024]:
        d = 64
        Q = np.random.randn(seq_len, d)
        K = np.random.randn(seq_len, d)
        V = np.random.randn(seq_len, d)
        
        standard_ops = seq_len * seq_len * d + seq_len * seq_len * d  # QK^T + AV
        linear_ops = seq_len * d * d + seq_len * d * d  # K^T V + Q(K^T V)
        
        print(f"  n={seq_len:4d}, d={d}: Standard={standard_ops/1e6:.1f}M, Linear={linear_ops/1e6:.1f}M, Ratio={standard_ops/linear_ops:.1f}x")
    
    # Accuracy comparison
    seq_len = 32
    d = 16
    Q = np.random.randn(seq_len, d) * 0.5
    K = np.random.randn(seq_len, d) * 0.5
    V = np.random.randn(seq_len, d)
    
    output_standard = standard_attention(Q, K, V)
    output_linear = linear_attention(Q, K, V)
    
    print(f"\nAccuracy Comparison (seq_len={seq_len}, d={d}):")
    print(f"  Standard output mean: {np.mean(output_standard):.4f}")
    print(f"  Linear output mean: {np.mean(output_linear):.4f}")
    print(f"  MSE: {np.mean((output_standard - output_linear)**2):.6f}")
    
    # RFF approximation
    print("\nRandom Fourier Features Approximation:")
    for n_features in [16, 32, 64, 128]:
        Q_feat, K_feat = random_fourier_features(Q, K, n_features)
        
        # Approximate attention matrix
        approx_attn = Q_feat @ K_feat.T
        exact_attn = np.exp(Q @ K.T / np.sqrt(d))
        exact_attn = exact_attn / exact_attn.sum(axis=-1, keepdims=True)
        approx_attn = approx_attn / (approx_attn.sum(axis=-1, keepdims=True) + 1e-10)
        
        error = np.mean((exact_attn - approx_attn)**2)
        print(f"  n_features={n_features:3d}: MSE={error:.6f}")
    
    return linear_attention, random_fourier_features


# =============================================================================
# Example 9: Sparse Attention Patterns
# =============================================================================

def example_sparse_attention():
    """
    Implement various sparse attention patterns.
    """
    print("\n" + "=" * 70)
    print("Example 9: Sparse Attention Patterns")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def create_local_mask(seq_len: int, window: int) -> np.ndarray:
        """
        Local (sliding window) attention mask.
        Each position attends to positions within window.
        """
        mask = np.full((seq_len, seq_len), -np.inf)
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            mask[i, start:end] = 0
        return mask
    
    def create_strided_mask(seq_len: int, stride: int) -> np.ndarray:
        """
        Strided attention: attend to every stride-th position.
        """
        mask = np.full((seq_len, seq_len), -np.inf)
        for i in range(seq_len):
            mask[i, ::stride] = 0
            mask[i, i] = 0  # Always attend to self
        return mask
    
    def create_block_sparse_mask(seq_len: int, block_size: int) -> np.ndarray:
        """
        Block-sparse attention: attend within blocks.
        """
        mask = np.full((seq_len, seq_len), -np.inf)
        n_blocks = seq_len // block_size
        
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            mask[start:end, start:end] = 0
        
        # Handle remainder
        remainder_start = n_blocks * block_size
        if remainder_start < seq_len:
            mask[remainder_start:, remainder_start:] = 0
        
        return mask
    
    def create_dilated_mask(seq_len: int, dilation: int, window: int) -> np.ndarray:
        """
        Dilated attention: local attention with gaps.
        """
        mask = np.full((seq_len, seq_len), -np.inf)
        for i in range(seq_len):
            for j in range(-window, window + 1):
                pos = i + j * dilation
                if 0 <= pos < seq_len:
                    mask[i, pos] = 0
        return mask
    
    def create_longformer_mask(seq_len: int, window: int, 
                               global_indices: List[int]) -> np.ndarray:
        """
        Longformer-style: local + global attention.
        - Most positions: local window attention
        - Global positions: attend to/from all positions
        """
        # Start with local
        mask = create_local_mask(seq_len, window)
        
        # Add global positions
        for idx in global_indices:
            mask[idx, :] = 0  # Global attends to all
            mask[:, idx] = 0  # All attend to global
        
        return mask
    
    seq_len = 16
    
    print("\nSparse Attention Patterns:")
    
    # Visualize patterns
    patterns = [
        ("Local (w=2)", create_local_mask(seq_len, window=2)),
        ("Strided (s=4)", create_strided_mask(seq_len, stride=4)),
        ("Block (b=4)", create_block_sparse_mask(seq_len, block_size=4)),
        ("Dilated (d=2, w=2)", create_dilated_mask(seq_len, dilation=2, window=2)),
        ("Longformer (w=2, g=[0,8])", create_longformer_mask(seq_len, window=2, global_indices=[0, 8])),
    ]
    
    for name, mask in patterns:
        density = np.mean(mask > -np.inf)
        print(f"\n  {name}:")
        print(f"    Density: {density:.2%}")
        
        # Show pattern (. = attend, x = blocked)
        pattern_str = np.where(mask > -np.inf, '.', 'x')
        for i, row in enumerate(pattern_str):
            if i < 8:  # Show first 8 rows
                print(f"    {''.join(row[:8])}...")
        if seq_len > 8:
            print(f"    ...")
    
    # Complexity comparison
    print("\nComplexity Comparison (seq_len=1024):")
    seq_len = 1024
    
    patterns_complexity = [
        ("Full attention", 1.0),
        ("Local (w=64)", 2 * 64 / seq_len),
        ("Strided (s=32)", 1 / 32 + 1 / seq_len),
        ("Block (b=64)", 64 / seq_len),
        ("Longformer (w=64, g=4)", (2 * 64 + 4) / seq_len + 4 / seq_len),
    ]
    
    for name, density in patterns_complexity:
        ops = seq_len * seq_len * density
        print(f"  {name:30s}: {density:6.2%} density, {ops/1e6:.2f}M ops")
    
    return create_local_mask, create_longformer_mask


# =============================================================================
# Example 10: Complete Transformer Block
# =============================================================================

def example_transformer_block():
    """
    Implement a complete transformer encoder block.
    """
    print("\n" + "=" * 70)
    print("Example 10: Complete Transformer Block")
    print("=" * 70)
    
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    class LayerNorm:
        """Layer normalization."""
        
        def __init__(self, d_model: int, eps: float = 1e-6):
            self.eps = eps
            self.gamma = np.ones(d_model)
            self.beta = np.zeros(d_model)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            mean = np.mean(x, axis=-1, keepdims=True)
            std = np.std(x, axis=-1, keepdims=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    class MultiHeadAttention:
        """Multi-head attention."""
        
        def __init__(self, d_model: int, n_heads: int, seed: int = 42):
            np.random.seed(seed)
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            scale = np.sqrt(2.0 / d_model)
            self.W_q = np.random.randn(d_model, d_model) * scale
            self.W_k = np.random.randn(d_model, d_model) * scale
            self.W_v = np.random.randn(d_model, d_model) * scale
            self.W_o = np.random.randn(d_model, d_model) * scale
        
        def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
            batch_size, seq_len, _ = x.shape
            
            # Project
            Q = x @ self.W_q
            K = x @ self.W_k
            V = x @ self.W_v
            
            # Reshape for heads
            Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
            
            # Attention
            scores = Q @ K.swapaxes(-2, -1) / np.sqrt(self.d_k)
            if mask is not None:
                scores = scores + mask
            
            weights = softmax(scores, axis=-1)
            attn_output = weights @ V
            
            # Combine heads
            attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            
            return attn_output @ self.W_o
    
    class FeedForward:
        """Position-wise feed-forward network."""
        
        def __init__(self, d_model: int, d_ff: int, seed: int = 42):
            np.random.seed(seed)
            scale = np.sqrt(2.0 / d_model)
            self.W1 = np.random.randn(d_model, d_ff) * scale
            self.b1 = np.zeros(d_ff)
            self.W2 = np.random.randn(d_ff, d_model) * scale
            self.b2 = np.zeros(d_model)
        
        def forward(self, x: np.ndarray) -> np.ndarray:
            hidden = gelu(x @ self.W1 + self.b1)
            return hidden @ self.W2 + self.b2
    
    class TransformerEncoderBlock:
        """
        Complete transformer encoder block:
        - Multi-head self-attention
        - Add & Norm
        - Feed-forward
        - Add & Norm
        """
        
        def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                     dropout: float = 0.1, seed: int = 42):
            self.attention = MultiHeadAttention(d_model, n_heads, seed)
            self.norm1 = LayerNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, seed + 1)
            self.norm2 = LayerNorm(d_model)
            self.dropout = dropout
        
        def forward(self, x: np.ndarray, mask: np.ndarray = None,
                    training: bool = False) -> np.ndarray:
            # Self-attention with residual
            attn_output = self.attention.forward(x, mask)
            x = self.norm1.forward(x + attn_output)
            
            # Feed-forward with residual
            ffn_output = self.ffn.forward(x)
            x = self.norm2.forward(x + ffn_output)
            
            return x
    
    class TransformerEncoder:
        """Stack of transformer encoder blocks."""
        
        def __init__(self, n_layers: int, d_model: int, n_heads: int, 
                     d_ff: int, max_len: int, seed: int = 42):
            self.layers = [
                TransformerEncoderBlock(d_model, n_heads, d_ff, seed=seed + i)
                for i in range(n_layers)
            ]
            
            # Positional encoding
            self.pos_encoding = self._sinusoidal_encoding(max_len, d_model)
        
        def _sinusoidal_encoding(self, max_len: int, d_model: int) -> np.ndarray:
            PE = np.zeros((max_len, d_model))
            position = np.arange(max_len)[:, np.newaxis]
            div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            PE[:, 0::2] = np.sin(position * div_term)
            PE[:, 1::2] = np.cos(position * div_term)
            return PE
        
        def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
            seq_len = x.shape[1]
            
            # Add positional encoding
            x = x + self.pos_encoding[:seq_len]
            
            # Pass through layers
            for layer in self.layers:
                x = layer.forward(x, mask)
            
            return x
    
    # Create and test transformer
    np.random.seed(42)
    
    # Configuration
    n_layers = 4
    d_model = 128
    n_heads = 8
    d_ff = 512
    max_len = 256
    
    encoder = TransformerEncoder(n_layers, d_model, n_heads, d_ff, max_len)
    
    # Test input
    batch_size = 2
    seq_len = 32
    x = np.random.randn(batch_size, seq_len, d_model)
    
    output = encoder.forward(x)
    
    print("\nTransformer Encoder Configuration:")
    print(f"  Layers: {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  n_heads: {n_heads}")
    print(f"  d_ff: {d_ff}")
    print(f"  max_len: {max_len}")
    
    print(f"\nInput/Output:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Parameter count
    params_attention = 4 * d_model * d_model  # Q, K, V, O
    params_ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model
    params_norm = 2 * d_model * 2  # 2 layer norms with gamma, beta
    params_per_layer = params_attention + params_ffn + params_norm
    total_params = n_layers * params_per_layer
    
    print(f"\nParameter Count:")
    print(f"  Per layer: {params_per_layer:,}")
    print(f"  Total: {total_params:,}")
    
    return TransformerEncoder


def main():
    """Run all attention mechanism examples."""
    print("ATTENTION MECHANISMS IN DEEP LEARNING")
    print("=" * 70)
    
    # Run all examples
    example_basic_attention()
    example_scaled_dot_product()
    example_multi_head_attention()
    example_self_cross_attention()
    example_attention_masking()
    example_positional_encoding()
    example_attention_analysis()
    example_linear_attention()
    example_sparse_attention()
    example_transformer_block()
    
    print("\n" + "=" * 70)
    print("All attention mechanism examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
