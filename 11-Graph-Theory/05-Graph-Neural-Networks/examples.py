"""
Graph Neural Networks - Examples
================================

Implementations of core GNN architectures from scratch,
demonstrating the mathematical foundations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


# =============================================================================
# Example 1: Graph Convolutional Network (GCN) Layer
# =============================================================================

def example_1_gcn_layer():
    """
    GCN Layer: h^{(l+1)} = σ(Â H^{(l)} W^{(l)})
    
    Where Â = D̃^{-1/2} Ã D̃^{-1/2}, Ã = A + I
    """
    print("Example 1: GCN Layer")
    print("=" * 60)
    
    class GCNLayer:
        """Single GCN layer."""
        
        def __init__(self, in_features: int, out_features: int):
            self.in_features = in_features
            self.out_features = out_features
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * scale
            self.b = np.zeros(out_features)
        
        def forward(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            
            Args:
                A: Adjacency matrix (n, n)
                H: Node features (n, in_features)
            
            Returns:
                Output features (n, out_features)
            """
            n = len(A)
            
            # Add self-loops
            A_tilde = A + np.eye(n)
            
            # Degree matrix
            D_tilde = np.diag(A_tilde.sum(axis=1))
            
            # Symmetric normalization
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde) + 1e-10))
            A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
            
            # Propagate and transform
            H_out = A_hat @ H @ self.W + self.b
            
            # ReLU activation
            return np.maximum(0, H_out)
    
    # Create graph
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    # Node features
    np.random.seed(42)
    X = np.random.randn(5, 4)  # 5 nodes, 4 features
    
    # Create and apply layer
    layer = GCNLayer(in_features=4, out_features=8)
    H = layer.forward(A, X)
    
    print(f"Input features shape: {X.shape}")
    print(f"Output features shape: {H.shape}")
    print(f"Sample output:\n{H[:2, :4]}")
    
    return GCNLayer


# =============================================================================
# Example 2: Multi-Layer GCN
# =============================================================================

def example_2_multilayer_gcn():
    """
    Stack multiple GCN layers for deeper representations.
    """
    print("\nExample 2: Multi-Layer GCN")
    print("=" * 60)
    
    class GCN:
        """Multi-layer GCN."""
        
        def __init__(self, layer_sizes: List[int], dropout: float = 0.5):
            self.layers = []
            self.dropout = dropout
            
            for i in range(len(layer_sizes) - 1):
                scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i+1]))
                W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
                b = np.zeros(layer_sizes[i+1])
                self.layers.append((W, b))
        
        def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
            """Compute normalized adjacency."""
            n = len(A)
            A_tilde = A + np.eye(n)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1) + 1e-10))
            return D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
        def forward(self, A: np.ndarray, X: np.ndarray, 
                   training: bool = True) -> np.ndarray:
            """Forward pass through all layers."""
            A_hat = self._normalize_adjacency(A)
            H = X
            
            for i, (W, b) in enumerate(self.layers):
                H = A_hat @ H @ W + b
                
                # Apply ReLU except last layer
                if i < len(self.layers) - 1:
                    H = np.maximum(0, H)
                    
                    # Dropout
                    if training and self.dropout > 0:
                        mask = np.random.rand(*H.shape) > self.dropout
                        H = H * mask / (1 - self.dropout)
            
            return H
        
        def predict(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Predict with softmax."""
            logits = self.forward(A, X, training=False)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
    
    # Create GCN
    gcn = GCN(layer_sizes=[4, 16, 8, 3])  # 4 input, 3 output classes
    
    # Example data
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    X = np.random.randn(5, 4)
    
    # Forward pass
    probs = gcn.predict(A, X)
    print(f"Prediction probabilities:\n{probs}")
    print(f"Predicted classes: {probs.argmax(axis=1)}")
    
    return GCN


# =============================================================================
# Example 3: GraphSAGE Layer
# =============================================================================

def example_3_graphsage():
    """
    GraphSAGE: Sample and Aggregate for inductive learning.
    
    h_v = σ(W · CONCAT(h_v, AGG({h_u : u ∈ N(v)})))
    """
    print("\nExample 3: GraphSAGE Layer")
    print("=" * 60)
    
    class GraphSAGELayer:
        """GraphSAGE layer with different aggregators."""
        
        def __init__(self, in_features: int, out_features: int,
                    aggregator: str = 'mean'):
            self.aggregator = aggregator
            
            # Weight for concatenated features
            scale = np.sqrt(2.0 / (2 * in_features + out_features))
            self.W = np.random.randn(2 * in_features, out_features) * scale
            self.b = np.zeros(out_features)
            
            # For pool aggregator
            if aggregator == 'pool':
                self.W_pool = np.random.randn(in_features, in_features) * scale
                self.b_pool = np.zeros(in_features)
        
        def aggregate(self, neighbor_features: List[np.ndarray]) -> np.ndarray:
            """Aggregate neighbor features."""
            if len(neighbor_features) == 0:
                return np.zeros(self.W.shape[0] // 2)
            
            stacked = np.stack(neighbor_features)
            
            if self.aggregator == 'mean':
                return stacked.mean(axis=0)
            elif self.aggregator == 'max':
                return stacked.max(axis=0)
            elif self.aggregator == 'pool':
                # Apply MLP then max pool
                transformed = np.maximum(0, stacked @ self.W_pool + self.b_pool)
                return transformed.max(axis=0)
            else:
                raise ValueError(f"Unknown aggregator: {self.aggregator}")
        
        def forward(self, adj_list: Dict[int, List[int]], 
                   H: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            
            Args:
                adj_list: Adjacency list {node: [neighbors]}
                H: Node features (n, in_features)
            
            Returns:
                Updated features (n, out_features)
            """
            n = len(H)
            H_out = []
            
            for v in range(n):
                # Get self features
                h_v = H[v]
                
                # Aggregate neighbor features
                neighbors = adj_list.get(v, [])
                neighbor_feats = [H[u] for u in neighbors]
                h_N = self.aggregate(neighbor_feats)
                
                # Concatenate and transform
                h_concat = np.concatenate([h_v, h_N])
                h_new = h_concat @ self.W + self.b
                
                # Normalize (important for stability)
                h_new = h_new / (np.linalg.norm(h_new) + 1e-10)
                
                H_out.append(h_new)
            
            return np.maximum(0, np.array(H_out))  # ReLU
    
    # Create adjacency list
    adj_list = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3, 4],
        3: [1, 2, 4],
        4: [2, 3]
    }
    
    # Node features
    np.random.seed(42)
    H = np.random.randn(5, 4)
    
    # Test different aggregators
    for agg in ['mean', 'max', 'pool']:
        layer = GraphSAGELayer(in_features=4, out_features=8, aggregator=agg)
        H_out = layer.forward(adj_list, H)
        print(f"{agg.upper()} aggregator output shape: {H_out.shape}")
    
    return GraphSAGELayer


# =============================================================================
# Example 4: Graph Attention Network (GAT) Layer
# =============================================================================

def example_4_gat():
    """
    GAT: Learn attention weights for neighbor aggregation.
    
    α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    h'_i = σ(Σ_j α_ij W h_j)
    """
    print("\nExample 4: GAT Layer")
    print("=" * 60)
    
    class GATLayer:
        """Single-head Graph Attention layer."""
        
        def __init__(self, in_features: int, out_features: int,
                    negative_slope: float = 0.2):
            self.negative_slope = negative_slope
            
            # Feature transformation
            scale = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * scale
            
            # Attention weights
            self.a = np.random.randn(2 * out_features) * scale
        
        def leaky_relu(self, x: np.ndarray) -> np.ndarray:
            return np.where(x > 0, x, self.negative_slope * x)
        
        def forward(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            
            Args:
                A: Adjacency matrix (used for masking)
                H: Node features (n, in_features)
            
            Returns:
                Updated features (n, out_features)
            """
            n = len(H)
            
            # Transform features
            Wh = H @ self.W  # (n, out_features)
            
            # Compute attention scores for all pairs
            # a^T [Wh_i || Wh_j] = a_1^T Wh_i + a_2^T Wh_j
            a_1 = self.a[:len(self.a)//2]
            a_2 = self.a[len(self.a)//2:]
            
            e1 = Wh @ a_1  # (n,) - self attention scores
            e2 = Wh @ a_2  # (n,) - neighbor attention scores
            
            # Pairwise attention: e_ij = e1_i + e2_j
            e = e1.reshape(-1, 1) + e2.reshape(1, -1)  # (n, n)
            e = self.leaky_relu(e)
            
            # Mask with adjacency (including self-loops)
            A_with_self = A + np.eye(n)
            e = np.where(A_with_self > 0, e, -1e9)
            
            # Softmax
            e_max = e.max(axis=1, keepdims=True)
            exp_e = np.exp(e - e_max)
            attention = exp_e / (exp_e.sum(axis=1, keepdims=True) + 1e-10)
            
            # Aggregate
            H_out = attention @ Wh
            
            return np.maximum(0, H_out)  # ReLU
        
        def get_attention_weights(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
            """Get attention weights for visualization."""
            n = len(H)
            Wh = H @ self.W
            
            a_1 = self.a[:len(self.a)//2]
            a_2 = self.a[len(self.a)//2:]
            
            e1 = Wh @ a_1
            e2 = Wh @ a_2
            e = e1.reshape(-1, 1) + e2.reshape(1, -1)
            e = self.leaky_relu(e)
            
            A_with_self = A + np.eye(n)
            e = np.where(A_with_self > 0, e, -1e9)
            
            e_max = e.max(axis=1, keepdims=True)
            exp_e = np.exp(e - e_max)
            attention = exp_e / (exp_e.sum(axis=1, keepdims=True) + 1e-10)
            
            return attention
    
    # Test
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    np.random.seed(42)
    H = np.random.randn(5, 4)
    
    layer = GATLayer(in_features=4, out_features=8)
    H_out = layer.forward(A, H)
    attention = layer.get_attention_weights(A, H)
    
    print(f"Output shape: {H_out.shape}")
    print(f"Attention weights for node 0: {np.round(attention[0], 3)}")
    print(f"  (Sums to 1: {attention[0].sum():.4f})")
    
    return GATLayer


# =============================================================================
# Example 5: Multi-Head Attention GAT
# =============================================================================

def example_5_multihead_gat():
    """
    Multi-head GAT: Concatenate (or average) multiple attention heads.
    """
    print("\nExample 5: Multi-Head GAT")
    print("=" * 60)
    
    class MultiHeadGATLayer:
        """Multi-head Graph Attention layer."""
        
        def __init__(self, in_features: int, out_features: int,
                    num_heads: int = 4, concat: bool = True):
            self.num_heads = num_heads
            self.concat = concat
            self.head_dim = out_features // num_heads if not concat else out_features
            
            # Initialize heads
            self.heads = []
            for _ in range(num_heads):
                scale = np.sqrt(2.0 / (in_features + self.head_dim))
                W = np.random.randn(in_features, self.head_dim) * scale
                a = np.random.randn(2 * self.head_dim) * scale
                self.heads.append((W, a))
        
        def forward(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
            """Forward with multiple heads."""
            n = len(H)
            head_outputs = []
            
            for W, a in self.heads:
                Wh = H @ W
                
                a_1 = a[:len(a)//2]
                a_2 = a[len(a)//2:]
                
                e = (Wh @ a_1).reshape(-1, 1) + (Wh @ a_2).reshape(1, -1)
                e = np.where(e > 0, e, 0.2 * e)  # LeakyReLU
                
                A_with_self = A + np.eye(n)
                e = np.where(A_with_self > 0, e, -1e9)
                
                exp_e = np.exp(e - e.max(axis=1, keepdims=True))
                attention = exp_e / (exp_e.sum(axis=1, keepdims=True) + 1e-10)
                
                head_outputs.append(attention @ Wh)
            
            if self.concat:
                return np.concatenate(head_outputs, axis=1)
            else:
                return np.mean(head_outputs, axis=0)
    
    # Test
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    np.random.seed(42)
    H = np.random.randn(5, 8)
    
    # Concatenation (hidden layers)
    layer_concat = MultiHeadGATLayer(in_features=8, out_features=4, 
                                     num_heads=4, concat=True)
    H_concat = layer_concat.forward(A, H)
    print(f"Concat mode output shape: {H_concat.shape}")
    
    # Averaging (final layer)
    layer_avg = MultiHeadGATLayer(in_features=8, out_features=4,
                                  num_heads=4, concat=False)
    H_avg = layer_avg.forward(A, H)
    print(f"Average mode output shape: {H_avg.shape}")
    
    return MultiHeadGATLayer


# =============================================================================
# Example 6: Graph Isomorphism Network (GIN)
# =============================================================================

def example_6_gin():
    """
    GIN: Maximally expressive GNN (equivalent to 1-WL test).
    
    h_v^{(k)} = MLP((1 + ε) h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)})
    """
    print("\nExample 6: Graph Isomorphism Network (GIN)")
    print("=" * 60)
    
    class GINLayer:
        """GIN layer with learnable epsilon."""
        
        def __init__(self, in_features: int, hidden_features: int,
                    out_features: int, learn_eps: bool = True):
            # MLP
            scale1 = np.sqrt(2.0 / (in_features + hidden_features))
            scale2 = np.sqrt(2.0 / (hidden_features + out_features))
            
            self.W1 = np.random.randn(in_features, hidden_features) * scale1
            self.b1 = np.zeros(hidden_features)
            self.W2 = np.random.randn(hidden_features, out_features) * scale2
            self.b2 = np.zeros(out_features)
            
            # Epsilon (learnable or fixed)
            self.learn_eps = learn_eps
            self.eps = 0.0
        
        def mlp(self, x: np.ndarray) -> np.ndarray:
            """Two-layer MLP."""
            h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
            return h @ self.W2 + self.b2
        
        def forward(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
            """
            Forward pass.
            
            h_v = MLP((1 + ε) h_v + Σ h_u)
            """
            # Neighbor sum
            neighbor_sum = A @ H
            
            # Update
            H_combined = (1 + self.eps) * H + neighbor_sum
            
            return np.maximum(0, self.mlp(H_combined))
    
    class GIN:
        """Full GIN model for graph classification."""
        
        def __init__(self, input_dim: int, hidden_dim: int, 
                    output_dim: int, num_layers: int):
            self.layers = []
            
            # First layer
            self.layers.append(GINLayer(input_dim, hidden_dim, hidden_dim))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                self.layers.append(GINLayer(hidden_dim, hidden_dim, hidden_dim))
            
            # Readout MLP
            scale = np.sqrt(2.0 / (hidden_dim * num_layers + output_dim))
            self.W_out = np.random.randn(hidden_dim * num_layers, output_dim) * scale
        
        def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Forward with concatenated layer outputs (jump knowledge)."""
            layer_outputs = []
            H = X
            
            for layer in self.layers:
                H = layer.forward(A, H)
                layer_outputs.append(H)
            
            # Graph readout: sum pooling for each layer
            graph_features = []
            for H_layer in layer_outputs:
                graph_features.append(H_layer.sum(axis=0))
            
            # Concatenate layer features
            graph_embed = np.concatenate(graph_features)
            
            return graph_embed @ self.W_out
    
    # Test
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    np.random.seed(42)
    X = np.random.randn(5, 4)
    
    gin = GIN(input_dim=4, hidden_dim=16, output_dim=3, num_layers=3)
    logits = gin.forward(A, X)
    
    print(f"Graph embedding logits: {logits}")
    print(f"Predicted class: {logits.argmax()}")
    
    return GIN


# =============================================================================
# Example 7: Message Passing Framework
# =============================================================================

def example_7_mpnn():
    """
    General Message Passing Neural Network framework.
    
    h_v = UPDATE(h_v, AGG({MESSAGE(h_v, h_u, e_uv) : u ∈ N(v)}))
    """
    print("\nExample 7: Message Passing Framework")
    print("=" * 60)
    
    class MPNN:
        """Configurable Message Passing Neural Network."""
        
        def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int,
                    message_type: str = 'edge', 
                    aggregate_type: str = 'sum',
                    update_type: str = 'gru'):
            
            self.message_type = message_type
            self.aggregate_type = aggregate_type
            self.update_type = update_type
            
            # Message network
            if message_type == 'edge':
                # Message depends on edge features
                scale = np.sqrt(2.0 / (2 * node_dim + edge_dim + hidden_dim))
                self.M_W = np.random.randn(2 * node_dim + edge_dim, hidden_dim) * scale
            else:
                # Simple linear
                scale = np.sqrt(2.0 / (node_dim + hidden_dim))
                self.M_W = np.random.randn(node_dim, hidden_dim) * scale
            
            # Update network (simplified GRU)
            if update_type == 'gru':
                scale = np.sqrt(2.0 / (node_dim + hidden_dim + hidden_dim))
                self.U_z = np.random.randn(node_dim + hidden_dim, hidden_dim) * scale
                self.U_r = np.random.randn(node_dim + hidden_dim, hidden_dim) * scale
                self.U_h = np.random.randn(node_dim + hidden_dim, hidden_dim) * scale
            else:
                scale = np.sqrt(2.0 / (node_dim + hidden_dim + hidden_dim))
                self.U_W = np.random.randn(node_dim + hidden_dim, hidden_dim) * scale
        
        def message(self, h_v: np.ndarray, h_u: np.ndarray, 
                   e_uv: Optional[np.ndarray] = None) -> np.ndarray:
            """Compute message from u to v."""
            if self.message_type == 'edge' and e_uv is not None:
                x = np.concatenate([h_v, h_u, e_uv])
            else:
                x = h_u
            return np.maximum(0, x @ self.M_W)
        
        def aggregate(self, messages: List[np.ndarray]) -> np.ndarray:
            """Aggregate neighbor messages."""
            if len(messages) == 0:
                return np.zeros(self.M_W.shape[1])
            
            stacked = np.stack(messages)
            
            if self.aggregate_type == 'sum':
                return stacked.sum(axis=0)
            elif self.aggregate_type == 'mean':
                return stacked.mean(axis=0)
            elif self.aggregate_type == 'max':
                return stacked.max(axis=0)
        
        def update(self, h_v: np.ndarray, m_v: np.ndarray) -> np.ndarray:
            """Update node state."""
            x = np.concatenate([h_v, m_v])
            
            if self.update_type == 'gru':
                z = 1 / (1 + np.exp(-x @ self.U_z))  # Update gate
                r = 1 / (1 + np.exp(-x @ self.U_r))  # Reset gate
                
                h_v_padded = np.concatenate([r * h_v, m_v])
                h_tilde = np.tanh(h_v_padded @ self.U_h[:len(h_v_padded)])
                
                return (1 - z) * h_v[:len(z)] + z * h_tilde
            else:
                return np.maximum(0, x @ self.U_W)
        
        def forward(self, adj_list: Dict[int, List[int]], 
                   H: np.ndarray,
                   E: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
                   num_steps: int = 3) -> np.ndarray:
            """
            Run message passing for multiple steps.
            
            Args:
                adj_list: {node: [neighbors]}
                H: Node features (n, node_dim)
                E: Edge features {(u, v): features}
                num_steps: Number of message passing steps
            """
            n = len(H)
            
            for _ in range(num_steps):
                H_new = []
                
                for v in range(n):
                    # Collect messages
                    messages = []
                    for u in adj_list.get(v, []):
                        e_uv = E.get((u, v)) if E else None
                        msg = self.message(H[v], H[u], e_uv)
                        messages.append(msg)
                    
                    # Aggregate
                    m_v = self.aggregate(messages)
                    
                    # Update
                    h_new = self.update(H[v], m_v)
                    H_new.append(h_new)
                
                H = np.array(H_new)
            
            return H
    
    # Test
    adj_list = {0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 4], 4: [2, 3]}
    
    np.random.seed(42)
    H = np.random.randn(5, 4)
    
    # Edge features
    E = {}
    for v, neighbors in adj_list.items():
        for u in neighbors:
            E[(u, v)] = np.random.randn(2)
    
    mpnn = MPNN(node_dim=4, edge_dim=2, hidden_dim=8,
               message_type='edge', aggregate_type='sum', update_type='gru')
    
    H_out = mpnn.forward(adj_list, H, E, num_steps=3)
    print(f"Output shape: {H_out.shape}")
    
    return MPNN


# =============================================================================
# Example 8: Graph Pooling Methods
# =============================================================================

def example_8_graph_pooling():
    """
    Graph-level readout and hierarchical pooling.
    """
    print("\nExample 8: Graph Pooling Methods")
    print("=" * 60)
    
    class GlobalPooling:
        """Global graph pooling methods."""
        
        @staticmethod
        def sum_pool(H: np.ndarray) -> np.ndarray:
            """Sum all node features."""
            return H.sum(axis=0)
        
        @staticmethod
        def mean_pool(H: np.ndarray) -> np.ndarray:
            """Mean of node features."""
            return H.mean(axis=0)
        
        @staticmethod
        def max_pool(H: np.ndarray) -> np.ndarray:
            """Element-wise max."""
            return H.max(axis=0)
        
        @staticmethod
        def attention_pool(H: np.ndarray, 
                          W_gate: np.ndarray) -> np.ndarray:
            """Attention-weighted sum."""
            # Compute attention scores
            scores = H @ W_gate  # (n, 1)
            attention = np.exp(scores) / np.exp(scores).sum()
            return (attention * H).sum(axis=0)
    
    class TopKPooling:
        """Top-K node selection pooling."""
        
        def __init__(self, in_features: int, ratio: float = 0.5):
            self.ratio = ratio
            self.score_W = np.random.randn(in_features, 1)
        
        def forward(self, A: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            Select top-k nodes based on learned scores.
            
            Returns:
                pooled_A: Reduced adjacency
                pooled_H: Selected node features
            """
            n = len(H)
            k = max(1, int(n * self.ratio))
            
            # Compute scores
            scores = (H @ self.score_W).flatten()
            
            # Select top-k
            top_indices = np.argsort(scores)[-k:]
            
            # Create pooled adjacency
            pooled_A = A[np.ix_(top_indices, top_indices)]
            
            # Gate features by score
            gate = np.tanh(scores[top_indices])
            pooled_H = H[top_indices] * gate.reshape(-1, 1)
            
            return pooled_A, pooled_H
    
    # Test global pooling
    np.random.seed(42)
    H = np.random.randn(5, 4)
    
    print("Global Pooling:")
    print(f"  Sum: {GlobalPooling.sum_pool(H).shape}")
    print(f"  Mean: {GlobalPooling.mean_pool(H).shape}")
    print(f"  Max: {GlobalPooling.max_pool(H).shape}")
    
    W_gate = np.random.randn(4, 1)
    print(f"  Attention: {GlobalPooling.attention_pool(H, W_gate).shape}")
    
    # Test hierarchical pooling
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)
    
    pooler = TopKPooling(in_features=4, ratio=0.6)
    A_pooled, H_pooled = pooler.forward(A, H)
    
    print(f"\nTop-K Pooling (ratio=0.6):")
    print(f"  Original: {H.shape[0]} nodes")
    print(f"  Pooled: {H_pooled.shape[0]} nodes")
    
    return GlobalPooling, TopKPooling


# =============================================================================
# Example 9: Over-Smoothing Demonstration
# =============================================================================

def example_9_oversmoothing():
    """
    Demonstrate and measure over-smoothing in deep GNNs.
    """
    print("\nExample 9: Over-Smoothing Demonstration")
    print("=" * 60)
    
    def compute_dirichlet_energy(H: np.ndarray, L: np.ndarray) -> float:
        """Measure feature smoothness."""
        return float(np.trace(H.T @ L @ H))
    
    def compute_mad(H: np.ndarray) -> float:
        """Mean Average Distance between node features."""
        n = len(H)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += np.linalg.norm(H[i] - H[j])
                count += 1
        return total / count if count > 0 else 0
    
    # Create graph
    A = np.array([
        [0, 1, 0, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # GCN propagation matrix
    n = len(A)
    A_tilde = A + np.eye(n)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    # Initial random features
    np.random.seed(42)
    H = np.random.randn(n, 4)
    
    # Track metrics over layers
    print("Layer | Dirichlet Energy | MAD")
    print("-" * 40)
    
    for layer in range(20):
        energy = compute_dirichlet_energy(H, L)
        mad = compute_mad(H)
        if layer < 10 or layer % 5 == 0:
            print(f"{layer:5d} | {energy:16.4f} | {mad:.4f}")
        
        # GCN propagation (no learned weights, just propagation)
        H = A_hat @ H
    
    print("\nObservation: Energy and MAD decrease → features become similar")
    
    return compute_dirichlet_energy, compute_mad


# =============================================================================
# Example 10: Relational GCN for Heterogeneous Graphs
# =============================================================================

def example_10_rgcn():
    """
    R-GCN: Separate weight matrices per relation type.
    
    h_v = σ(W_0 h_v + Σ_r Σ_{u∈N_r(v)} (1/|N_r(v)|) W_r h_u)
    """
    print("\nExample 10: Relational GCN (R-GCN)")
    print("=" * 60)
    
    class RGCNLayer:
        """Relational GCN layer."""
        
        def __init__(self, in_features: int, out_features: int,
                    num_relations: int, regularization: str = 'basis',
                    num_bases: int = 2):
            self.num_relations = num_relations
            self.regularization = regularization
            
            scale = np.sqrt(2.0 / (in_features + out_features))
            
            # Self-loop weight
            self.W_0 = np.random.randn(in_features, out_features) * scale
            
            if regularization == 'basis':
                # Basis decomposition: W_r = Σ_b c_rb V_b
                self.bases = [np.random.randn(in_features, out_features) * scale 
                             for _ in range(num_bases)]
                self.coefficients = np.random.randn(num_relations, num_bases)
            else:
                # Separate weights per relation
                self.W_r = [np.random.randn(in_features, out_features) * scale
                           for _ in range(num_relations)]
        
        def get_relation_weight(self, r: int) -> np.ndarray:
            """Get weight matrix for relation r."""
            if self.regularization == 'basis':
                W = np.zeros_like(self.bases[0])
                for b, V_b in enumerate(self.bases):
                    W += self.coefficients[r, b] * V_b
                return W
            else:
                return self.W_r[r]
        
        def forward(self, H: np.ndarray, 
                   relation_adj: Dict[int, np.ndarray]) -> np.ndarray:
            """
            Forward pass.
            
            Args:
                H: Node features (n, in_features)
                relation_adj: {relation_id: adjacency_matrix}
            
            Returns:
                Updated features (n, out_features)
            """
            n = len(H)
            
            # Self-loop contribution
            H_out = H @ self.W_0
            
            # Relation contributions
            for r, A_r in relation_adj.items():
                W_r = self.get_relation_weight(r)
                
                # Normalize by in-degree
                degree = A_r.sum(axis=1, keepdims=True) + 1e-10
                A_r_norm = A_r / degree
                
                H_out += A_r_norm @ H @ W_r
            
            return np.maximum(0, H_out)
    
    # Create heterogeneous graph
    n = 6
    
    # Relation 0: "follows"
    A_follows = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0]
    ], dtype=float)
    
    # Relation 1: "friends"
    A_friends = np.array([
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0]
    ], dtype=float)
    
    relation_adj = {0: A_follows, 1: A_friends}
    
    np.random.seed(42)
    H = np.random.randn(n, 4)
    
    # Create and apply R-GCN
    rgcn = RGCNLayer(in_features=4, out_features=8, 
                    num_relations=2, regularization='basis', num_bases=2)
    
    H_out = rgcn.forward(H, relation_adj)
    print(f"Input shape: {H.shape}")
    print(f"Output shape: {H_out.shape}")
    print(f"Num relations: {len(relation_adj)}")
    
    return RGCNLayer


# =============================================================================
# Example 11: Mini-Batch Training with Neighbor Sampling
# =============================================================================

def example_11_minibatch_training():
    """
    Efficient mini-batch training with neighbor sampling (GraphSAGE style).
    """
    print("\nExample 11: Mini-Batch Training")
    print("=" * 60)
    
    def sample_neighbors(adj_list: Dict[int, List[int]], 
                        nodes: List[int],
                        num_samples: int) -> Dict[int, List[int]]:
        """Sample fixed number of neighbors for each node."""
        sampled = {}
        for v in nodes:
            neighbors = adj_list.get(v, [])
            if len(neighbors) <= num_samples:
                sampled[v] = neighbors
            else:
                sampled[v] = list(np.random.choice(neighbors, num_samples, 
                                                   replace=False))
        return sampled
    
    def build_computation_graph(adj_list: Dict[int, List[int]],
                               batch_nodes: List[int],
                               num_layers: int,
                               samples_per_layer: List[int]) -> List[Dict]:
        """
        Build multi-layer computation graph for batch.
        
        Returns list of layer info: {nodes, adj}
        """
        layers = []
        current_nodes = set(batch_nodes)
        
        for layer in range(num_layers):
            # Sample neighbors
            sampled_adj = sample_neighbors(
                adj_list, list(current_nodes), samples_per_layer[layer])
            
            # Collect all nodes needed
            all_neighbors = set()
            for neighbors in sampled_adj.values():
                all_neighbors.update(neighbors)
            
            layers.append({
                'nodes': list(current_nodes),
                'adj': sampled_adj
            })
            
            # Expand to include neighbors for next layer
            current_nodes = current_nodes.union(all_neighbors)
        
        # Reverse so layer 0 is closest to input
        return layers[::-1]
    
    # Create larger graph
    np.random.seed(42)
    n = 100
    
    # Random sparse graph
    adj_list = {i: [] for i in range(n)}
    for i in range(n):
        num_neighbors = np.random.randint(3, 10)
        neighbors = np.random.choice([j for j in range(n) if j != i], 
                                    num_neighbors, replace=False)
        adj_list[i] = list(neighbors)
    
    # Batch of target nodes
    batch_nodes = [0, 5, 10, 15, 20]
    
    # Build computation graph
    comp_graph = build_computation_graph(
        adj_list, batch_nodes, 
        num_layers=2, 
        samples_per_layer=[10, 5])
    
    print(f"Target batch size: {len(batch_nodes)}")
    for i, layer in enumerate(comp_graph):
        print(f"Layer {i}: {len(layer['nodes'])} nodes")
    
    # Count unique nodes needed
    all_nodes = set()
    for layer in comp_graph:
        all_nodes.update(layer['nodes'])
    
    print(f"\nTotal unique nodes needed: {len(all_nodes)}")
    print(f"Full graph size: {n}")
    print(f"Savings: {100 * (1 - len(all_nodes)/n):.1f}%")
    
    return build_computation_graph


# =============================================================================
# Example 12: Complete GNN Pipeline
# =============================================================================

def example_12_complete_pipeline():
    """
    Complete node classification pipeline.
    """
    print("\nExample 12: Complete GNN Pipeline")
    print("=" * 60)
    
    class NodeClassificationGNN:
        """End-to-end GNN for node classification."""
        
        def __init__(self, input_dim: int, hidden_dim: int, 
                    output_dim: int, num_layers: int = 2,
                    dropout: float = 0.5):
            self.dropout = dropout
            self.layers = []
            
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            
            for i in range(num_layers):
                scale = np.sqrt(2.0 / (dims[i] + dims[i+1]))
                W = np.random.randn(dims[i], dims[i+1]) * scale
                self.layers.append(W)
        
        def _normalize_adj(self, A: np.ndarray) -> np.ndarray:
            """GCN normalization."""
            n = len(A)
            A_tilde = A + np.eye(n)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1) + 1e-10))
            return D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
        def forward(self, A: np.ndarray, X: np.ndarray, 
                   training: bool = True) -> np.ndarray:
            """Forward pass."""
            A_hat = self._normalize_adj(A)
            H = X
            
            for i, W in enumerate(self.layers):
                H = A_hat @ H @ W
                
                if i < len(self.layers) - 1:
                    H = np.maximum(0, H)  # ReLU
                    if training and self.dropout > 0:
                        mask = np.random.rand(*H.shape) > self.dropout
                        H = H * mask / (1 - self.dropout)
            
            return H
        
        def predict(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Predict class probabilities."""
            logits = self.forward(A, X, training=False)
            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        def loss(self, probs: np.ndarray, labels: np.ndarray, 
                mask: np.ndarray) -> float:
            """Cross-entropy loss on masked nodes."""
            n_masked = mask.sum()
            if n_masked == 0:
                return 0.0
            
            # Gather predictions for labeled nodes
            log_probs = np.log(probs[mask] + 1e-10)
            
            # One-hot encode labels
            num_classes = probs.shape[1]
            one_hot = np.zeros((n_masked, num_classes))
            one_hot[np.arange(n_masked), labels[mask]] = 1
            
            return -np.sum(one_hot * log_probs) / n_masked
        
        def accuracy(self, probs: np.ndarray, labels: np.ndarray,
                    mask: np.ndarray) -> float:
            """Classification accuracy on masked nodes."""
            preds = probs[mask].argmax(axis=1)
            return (preds == labels[mask]).mean()
    
    # Create synthetic data
    np.random.seed(42)
    n = 20
    num_classes = 3
    
    # Block-structured graph (community structure)
    A = np.zeros((n, n))
    for c in range(num_classes):
        start = c * (n // num_classes)
        end = (c + 1) * (n // num_classes)
        # Within-community edges (dense)
        for i in range(start, end):
            for j in range(i + 1, end):
                if np.random.rand() < 0.7:
                    A[i, j] = A[j, i] = 1
        # Between-community edges (sparse)
        if c < num_classes - 1:
            for _ in range(2):
                i = np.random.randint(start, end)
                j = np.random.randint(end, min(end + n // num_classes, n))
                A[i, j] = A[j, i] = 1
    
    # Features: class-informative with noise
    X = np.random.randn(n, 8)
    labels = np.repeat(np.arange(num_classes), n // num_classes)
    for i, c in enumerate(labels):
        X[i, :2] += c  # First 2 features are class-informative
    
    # Train/val/test split
    train_mask = np.zeros(n, dtype=bool)
    train_mask[::3] = True  # Every 3rd node
    
    val_mask = np.zeros(n, dtype=bool)
    val_mask[1::3] = True
    
    test_mask = np.zeros(n, dtype=bool)
    test_mask[2::3] = True
    
    # Create model
    model = NodeClassificationGNN(
        input_dim=8, hidden_dim=16, output_dim=num_classes, 
        num_layers=2, dropout=0.5)
    
    # "Training" (just showing forward pass - real training needs backprop)
    probs = model.predict(A, X)
    
    print(f"Graph: {n} nodes, {int(A.sum()/2)} edges")
    print(f"Classes: {num_classes}")
    print(f"Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    print(f"\nInitial metrics (random weights):")
    print(f"  Train accuracy: {model.accuracy(probs, labels, train_mask):.2%}")
    print(f"  Val accuracy: {model.accuracy(probs, labels, val_mask):.2%}")
    print(f"  Test accuracy: {model.accuracy(probs, labels, test_mask):.2%}")
    
    return NodeClassificationGNN


def run_all_examples():
    """Run all GNN examples."""
    print("=" * 70)
    print("GRAPH NEURAL NETWORKS - EXAMPLES")
    print("=" * 70)
    
    example_1_gcn_layer()
    example_2_multilayer_gcn()
    example_3_graphsage()
    example_4_gat()
    example_5_multihead_gat()
    example_6_gin()
    example_7_mpnn()
    example_8_graph_pooling()
    example_9_oversmoothing()
    example_10_rgcn()
    example_11_minibatch_training()
    example_12_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All GNN examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
