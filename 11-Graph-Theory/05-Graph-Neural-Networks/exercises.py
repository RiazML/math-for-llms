"""
Graph Neural Networks - Exercises
=================================

Hands-on exercises implementing GNN architectures and understanding
their mathematical foundations.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict


class GNNExercises:
    """Exercises for Graph Neural Networks."""
    
    # =========================================================================
    # Exercise 1: GCN Layer with Gradient Computation
    # =========================================================================
    
    @staticmethod
    def exercise_1_gcn_with_gradients():
        """
        Implement GCN layer with backward pass.
        
        Forward: H' = σ(Â H W)
        Need gradients for training.
        """
        
        class GCNLayerWithGrad:
            """GCN layer with gradient computation."""
            
            def __init__(self, in_features: int, out_features: int):
                scale = np.sqrt(2.0 / (in_features + out_features))
                self.W = np.random.randn(in_features, out_features) * scale
                self.b = np.zeros(out_features)
                
                # Cache for backward
                self.cache = {}
            
            def forward(self, A_hat: np.ndarray, H: np.ndarray) -> np.ndarray:
                """
                Forward pass with caching.
                
                TODO: Implement forward and cache intermediate values.
                """
                # TODO: Implement
                # 1. Z = Â H W + b
                # 2. H' = ReLU(Z)
                # 3. Cache A_hat, H, Z for backward
                pass
            
            def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, 
                                                                  np.ndarray, 
                                                                  np.ndarray]:
                """
                Backward pass.
                
                TODO: Compute gradients w.r.t. H, W, b
                
                Returns:
                    grad_H: Gradient w.r.t. input features
                    grad_W: Gradient w.r.t. weights
                    grad_b: Gradient w.r.t. bias
                """
                # TODO: Implement
                # 1. Backprop through ReLU
                # 2. Compute grad_W, grad_b
                # 3. Backprop through matrix multiplications
                pass
        
        def verify_gradient(layer, A_hat, H, eps=1e-5):
            """Numerical gradient verification."""
            # TODO: Implement gradient checking
            pass
        
        # Test
        print("Exercise 1: GCN Layer with Gradients")
        print("-" * 40)
        
        # Create test data
        n, d_in, d_out = 5, 4, 8
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.5).astype(float)
        np.fill_diagonal(A, 0)
        
        # Normalize adjacency
        A_tilde = A + np.eye(n)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(A_tilde.sum(axis=1)))
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
        
        H = np.random.randn(n, d_in)
        
        # Uncomment to test:
        # layer = GCNLayerWithGrad(d_in, d_out)
        # H_out = layer.forward(A_hat, H)
        # print(f"  Forward output shape: {H_out.shape}")
        
        return GCNLayerWithGrad
    
    # =========================================================================
    # Exercise 2: Attention Mechanism Variants
    # =========================================================================
    
    @staticmethod
    def exercise_2_attention_variants():
        """
        Implement different attention mechanisms for graphs.
        
        1. Additive attention (GAT style)
        2. Scaled dot-product attention (Transformer style)
        3. General attention
        """
        
        def additive_attention(Q: np.ndarray, K: np.ndarray,
                              A: np.ndarray,
                              W_att: np.ndarray) -> np.ndarray:
            """
            Additive attention: a^T tanh(W_q q + W_k k)
            
            Args:
                Q: Query features (n, d)
                K: Key features (n, d)
                A: Adjacency mask
                W_att: Attention parameters
            
            Returns:
                Attention weights (n, n)
            
            TODO: Implement
            """
            pass
        
        def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                        A: np.ndarray) -> np.ndarray:
            """
            Scaled dot-product: softmax(QK^T / sqrt(d))
            
            With adjacency masking.
            
            TODO: Implement
            """
            pass
        
        def general_attention(Q: np.ndarray, K: np.ndarray,
                            A: np.ndarray,
                            W: np.ndarray) -> np.ndarray:
            """
            General bilinear attention: softmax(Q W K^T)
            
            TODO: Implement
            """
            pass
        
        def compare_attention_mechanisms(Q, K, A):
            """Compare all attention types."""
            # TODO: Implement comparison
            pass
        
        # Test
        print("\nExercise 2: Attention Variants")
        print("-" * 40)
        
        n, d = 5, 4
        Q = np.random.randn(n, d)
        K = np.random.randn(n, d)
        A = np.random.randint(0, 2, (n, n)).astype(float)
        A = (A + A.T > 0).astype(float)
        np.fill_diagonal(A, 1)
        
        # Uncomment to test:
        # att_dot = scaled_dot_product_attention(Q, K, A)
        # print(f"  Attention shape: {att_dot.shape}")
        # print(f"  Row sums (should be 1): {att_dot.sum(axis=1)}")
        
        return additive_attention, scaled_dot_product_attention
    
    # =========================================================================
    # Exercise 3: Neighbor Sampling Implementation
    # =========================================================================
    
    @staticmethod
    def exercise_3_neighbor_sampling():
        """
        Implement efficient neighbor sampling for mini-batch training.
        """
        
        def uniform_sample(adj_list: Dict[int, List[int]],
                          node: int,
                          num_samples: int) -> List[int]:
            """
            Uniform random sampling without replacement.
            
            TODO: Implement
            """
            pass
        
        def importance_sample(adj_list: Dict[int, List[int]],
                            node: int,
                            num_samples: int,
                            node_degrees: Dict[int, int]) -> Tuple[List[int], 
                                                                    np.ndarray]:
            """
            Sample neighbors with probability proportional to degree.
            
            Returns sampled neighbors and importance weights.
            
            TODO: Implement
            """
            pass
        
        def layer_wise_sampling(adj_list: Dict[int, List[int]],
                               batch_nodes: List[int],
                               fanouts: List[int]) -> List[Dict[int, List[int]]]:
            """
            Sample fixed-size neighborhood for multiple layers.
            
            Args:
                adj_list: Full adjacency list
                batch_nodes: Target nodes for batch
                fanouts: [num_neighbors_layer1, num_neighbors_layer2, ...]
            
            Returns:
                List of sampled adjacency lists, one per layer
            
            TODO: Implement
            """
            pass
        
        def compute_sampling_variance(adj_list, node, num_samples, 
                                     num_trials=1000):
            """Estimate variance of different sampling strategies."""
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 3: Neighbor Sampling")
        print("-" * 40)
        
        # Create adjacency list
        adj_list = {
            0: [1, 2, 3, 4, 5],
            1: [0, 2, 6],
            2: [0, 1, 3],
            3: [0, 2, 4, 7],
            4: [0, 3, 5],
            5: [0, 4, 8],
            6: [1, 7],
            7: [3, 6, 8, 9],
            8: [5, 7, 9],
            9: [7, 8]
        }
        
        # Uncomment to test:
        # samples = uniform_sample(adj_list, node=0, num_samples=3)
        # print(f"  Sampled neighbors of node 0: {samples}")
        
        return layer_wise_sampling
    
    # =========================================================================
    # Exercise 4: Graph Isomorphism Network (GIN)
    # =========================================================================
    
    @staticmethod
    def exercise_4_gin_implementation():
        """
        Implement GIN with MLPs for maximal expressiveness.
        
        h_v = MLP((1 + ε) h_v + Σ h_u)
        """
        
        class MLP:
            """Simple MLP for GIN."""
            
            def __init__(self, layers: List[int]):
                """
                Args:
                    layers: [input_dim, hidden_dim, ..., output_dim]
                
                TODO: Initialize weights
                """
                pass
            
            def forward(self, x: np.ndarray) -> np.ndarray:
                """
                Forward pass with ReLU activations.
                
                TODO: Implement
                """
                pass
        
        class GINConv:
            """GIN convolution layer."""
            
            def __init__(self, in_dim: int, out_dim: int, 
                        eps: float = 0.0, train_eps: bool = True):
                """
                TODO: Initialize MLP and epsilon
                """
                pass
            
            def forward(self, A: np.ndarray, H: np.ndarray) -> np.ndarray:
                """
                Forward: MLP((1 + ε) h_v + Σ h_u)
                
                TODO: Implement
                """
                pass
        
        def test_wl_equivalence(gin, graphs: List[Tuple[np.ndarray, np.ndarray]]):
            """
            Test if GIN can distinguish WL-distinguishable graphs.
            
            TODO: Implement test
            """
            pass
        
        # Test
        print("\nExercise 4: GIN Implementation")
        print("-" * 40)
        
        # Two non-isomorphic graphs
        A1 = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ], dtype=float)
        
        A2 = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=float)
        
        # Uncomment to test:
        # gin = GINConv(in_dim=4, out_dim=4)
        # Test distinguishing ability
        
        return GINConv
    
    # =========================================================================
    # Exercise 5: Graph Pooling Methods
    # =========================================================================
    
    @staticmethod
    def exercise_5_graph_pooling():
        """
        Implement various graph pooling methods for graph-level tasks.
        """
        
        class DiffPool:
            """
            Differentiable Pooling.
            
            Learns soft cluster assignments.
            """
            
            def __init__(self, in_dim: int, out_dim: int, 
                        num_clusters: int):
                """
                TODO: Initialize GNNs for embedding and assignment.
                """
                pass
            
            def forward(self, A: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, 
                                                                      np.ndarray]:
                """
                Compute pooled graph.
                
                S = softmax(GNN_pool(A, H))  # Assignment matrix
                H' = S^T GNN_embed(A, H)     # Pooled features
                A' = S^T A S                  # Pooled adjacency
                
                TODO: Implement
                
                Returns:
                    A_pooled, H_pooled
                """
                pass
            
            def link_loss(self, A: np.ndarray, S: np.ndarray) -> float:
                """
                Link prediction loss: ||A - SS^T||_F
                
                Encourages nearby nodes to be in same cluster.
                
                TODO: Implement
                """
                pass
            
            def entropy_loss(self, S: np.ndarray) -> float:
                """
                Entropy loss: -Σ S log S
                
                Encourages sharp assignments.
                
                TODO: Implement
                """
                pass
        
        class SAGPooling:
            """Self-Attention Graph Pooling."""
            
            def __init__(self, in_dim: int, ratio: float = 0.5):
                """
                TODO: Initialize score computation.
                """
                pass
            
            def forward(self, A: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray,
                                                                      np.ndarray,
                                                                      np.ndarray]:
                """
                Select top-k nodes based on self-attention scores.
                
                TODO: Implement
                
                Returns:
                    A_pooled, H_pooled, indices of selected nodes
                """
                pass
        
        # Test
        print("\nExercise 5: Graph Pooling")
        print("-" * 40)
        
        n = 10
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.6).astype(float)
        np.fill_diagonal(A, 0)
        H = np.random.randn(n, 8)
        
        # Uncomment to test:
        # diffpool = DiffPool(in_dim=8, out_dim=8, num_clusters=4)
        # A_pooled, H_pooled = diffpool.forward(A, H)
        # print(f"  Original: {n} nodes, Pooled: {len(H_pooled)} nodes")
        
        return DiffPool, SAGPooling
    
    # =========================================================================
    # Exercise 6: Over-Smoothing Mitigation
    # =========================================================================
    
    @staticmethod
    def exercise_6_oversmoothing_mitigation():
        """
        Implement techniques to address over-smoothing in deep GNNs.
        """
        
        class ResidualGCN:
            """GCN with residual connections."""
            
            def __init__(self, in_dim: int, hidden_dim: int, 
                        out_dim: int, num_layers: int):
                """
                TODO: Initialize layers with residual connections.
                """
                pass
            
            def forward(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
                """
                H^{l+1} = H^l + GCN(H^l)
                
                TODO: Implement
                """
                pass
        
        class JumpingKnowledge:
            """
            Jumping Knowledge Networks.
            
            Aggregate representations from all layers.
            """
            
            def __init__(self, hidden_dim: int, num_layers: int,
                        mode: str = 'concat'):  # 'concat', 'max', 'lstm'
                """
                TODO: Initialize.
                """
                pass
            
            def aggregate(self, layer_outputs: List[np.ndarray]) -> np.ndarray:
                """
                Aggregate features from all layers.
                
                TODO: Implement different aggregation modes.
                """
                pass
        
        class PairNorm:
            """
            PairNorm: Prevents over-smoothing through pairwise normalization.
            
            Center then scale to unit variance.
            """
            
            def forward(self, H: np.ndarray) -> np.ndarray:
                """
                H_centered = H - mean(H)
                H_normed = H_centered / sqrt(mean(||H_centered||^2))
                
                TODO: Implement
                """
                pass
        
        def compare_methods(A: np.ndarray, X: np.ndarray, 
                           num_layers: int = 20):
            """Compare over-smoothing across methods."""
            # TODO: Implement comparison
            pass
        
        # Test
        print("\nExercise 6: Over-Smoothing Mitigation")
        print("-" * 40)
        
        n = 20
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        A = (A > 0.7).astype(float)
        np.fill_diagonal(A, 0)
        X = np.random.randn(n, 8)
        
        # Uncomment to test:
        # res_gcn = ResidualGCN(in_dim=8, hidden_dim=16, out_dim=8, num_layers=10)
        # H_out = res_gcn.forward(A, X)
        
        return ResidualGCN, JumpingKnowledge, PairNorm
    
    # =========================================================================
    # Exercise 7: Heterogeneous GNN
    # =========================================================================
    
    @staticmethod
    def exercise_7_heterogeneous_gnn():
        """
        Implement a heterogeneous graph neural network.
        """
        
        class HeteroGNNLayer:
            """
            GNN layer for heterogeneous graphs.
            
            Different transformations for different node/edge types.
            """
            
            def __init__(self, node_types: List[str],
                        edge_types: List[Tuple[str, str, str]],
                        in_dims: Dict[str, int],
                        out_dim: int):
                """
                Args:
                    node_types: List of node type names
                    edge_types: List of (src_type, edge_type, dst_type)
                    in_dims: {node_type: input_dim}
                    out_dim: Output dimension (same for all types)
                
                TODO: Initialize per-type weights.
                """
                pass
            
            def forward(self, H: Dict[str, np.ndarray],
                       edges: Dict[Tuple[str, str, str], 
                                  Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
                """
                Forward pass with type-specific message passing.
                
                Args:
                    H: {node_type: features}
                    edges: {(src_type, edge_type, dst_type): (src_idx, dst_idx)}
                
                Returns:
                    Updated features for each node type.
                
                TODO: Implement
                """
                pass
        
        def create_bipartite_graph():
            """Create example user-item bipartite graph."""
            # TODO: Create heterogeneous graph data
            pass
        
        # Test
        print("\nExercise 7: Heterogeneous GNN")
        print("-" * 40)
        
        # Example: User-Item graph
        node_types = ['user', 'item']
        edge_types = [('user', 'buys', 'item'), ('item', 'bought_by', 'user')]
        
        # Uncomment to test:
        # het_gnn = HeteroGNNLayer(
        #     node_types=node_types,
        #     edge_types=edge_types,
        #     in_dims={'user': 8, 'item': 16},
        #     out_dim=32
        # )
        
        return HeteroGNNLayer
    
    # =========================================================================
    # Exercise 8: Temporal GNN
    # =========================================================================
    
    @staticmethod
    def exercise_8_temporal_gnn():
        """
        Implement temporal/dynamic graph neural network.
        """
        
        class TimeEncode:
            """Learnable time encoding."""
            
            def __init__(self, dim: int):
                """
                Fourier-like encoding: [cos(ω_1 t), sin(ω_1 t), ...]
                
                TODO: Initialize
                """
                pass
            
            def forward(self, t: np.ndarray) -> np.ndarray:
                """
                Encode timestamps.
                
                TODO: Implement
                """
                pass
        
        class TemporalGNNLayer:
            """GNN layer with temporal awareness."""
            
            def __init__(self, node_dim: int, time_dim: int, out_dim: int):
                """
                TODO: Initialize with temporal components.
                """
                pass
            
            def forward(self, H: np.ndarray,
                       edges: np.ndarray,
                       timestamps: np.ndarray,
                       current_time: float) -> np.ndarray:
                """
                Process graph with temporal decay.
                
                More recent interactions should have higher weight.
                
                Args:
                    H: Node features (n, node_dim)
                    edges: Edge list (num_edges, 2)
                    timestamps: Edge timestamps (num_edges,)
                    current_time: Current time for decay
                
                TODO: Implement
                """
                pass
        
        # Test
        print("\nExercise 8: Temporal GNN")
        print("-" * 40)
        
        # Time encoder test
        # Uncomment to test:
        # time_enc = TimeEncode(dim=16)
        # t = np.array([0.0, 1.0, 2.0, 10.0])
        # encoded = time_enc.forward(t)
        # print(f"  Time encoding shape: {encoded.shape}")
        
        return TimeEncode, TemporalGNNLayer
    
    # =========================================================================
    # Exercise 9: GNN Explainability
    # =========================================================================
    
    @staticmethod
    def exercise_9_explainability():
        """
        Implement methods to explain GNN predictions.
        """
        
        def gradient_based_importance(model, A: np.ndarray, X: np.ndarray,
                                     target_node: int) -> np.ndarray:
            """
            Compute node importance using gradients.
            
            Importance(v) = ||∂output / ∂h_v||
            
            TODO: Implement
            """
            pass
        
        def attention_based_explanation(attention_weights: np.ndarray,
                                       target_node: int) -> Dict[int, float]:
            """
            Extract important neighbors from attention weights.
            
            TODO: Implement
            """
            pass
        
        def gnnexplainer_like(model, A: np.ndarray, X: np.ndarray,
                            target_node: int,
                            num_edges_to_select: int) -> np.ndarray:
            """
            Find minimal subgraph that maintains prediction.
            
            Simplified GNNExplainer: greedy edge selection.
            
            TODO: Implement
            """
            pass
        
        def counterfactual_explanation(model, A: np.ndarray, X: np.ndarray,
                                      target_node: int,
                                      target_class: int) -> np.ndarray:
            """
            Find minimal changes to flip prediction.
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 9: GNN Explainability")
        print("-" * 40)
        
        # Create simple graph
        A = np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0]
        ], dtype=float)
        X = np.random.randn(5, 4)
        
        # Uncomment to test:
        # importance = gradient_based_importance(model, A, X, target_node=0)
        # print(f"  Node importance: {importance}")
        
        return gnnexplainer_like
    
    # =========================================================================
    # Exercise 10: Complete GNN Training Loop
    # =========================================================================
    
    @staticmethod
    def exercise_10_training_loop():
        """
        Implement complete training loop for node classification.
        """
        
        class GNNTrainer:
            """Complete training loop for GNN."""
            
            def __init__(self, model, learning_rate: float = 0.01,
                        weight_decay: float = 5e-4):
                """
                TODO: Initialize optimizer state.
                """
                self.model = model
                self.lr = learning_rate
                self.wd = weight_decay
            
            def compute_loss(self, logits: np.ndarray, labels: np.ndarray,
                           mask: np.ndarray) -> float:
                """
                Cross-entropy loss on masked nodes.
                
                TODO: Implement
                """
                pass
            
            def compute_gradients(self, A: np.ndarray, X: np.ndarray,
                                labels: np.ndarray, mask: np.ndarray):
                """
                Compute gradients w.r.t. model parameters.
                
                TODO: Implement backpropagation
                """
                pass
            
            def update_parameters(self):
                """
                Update parameters with gradient descent + weight decay.
                
                TODO: Implement
                """
                pass
            
            def train_epoch(self, A: np.ndarray, X: np.ndarray,
                          labels: np.ndarray, 
                          train_mask: np.ndarray) -> float:
                """
                One training epoch.
                
                TODO: Implement
                """
                pass
            
            def evaluate(self, A: np.ndarray, X: np.ndarray,
                        labels: np.ndarray, mask: np.ndarray) -> float:
                """
                Evaluate accuracy on masked nodes.
                
                TODO: Implement
                """
                pass
            
            def train(self, A: np.ndarray, X: np.ndarray,
                     labels: np.ndarray,
                     train_mask: np.ndarray,
                     val_mask: np.ndarray,
                     num_epochs: int = 200,
                     patience: int = 10):
                """
                Full training with early stopping.
                
                TODO: Implement training loop with validation.
                """
                pass
        
        def create_synthetic_dataset(n: int = 100, 
                                    num_classes: int = 3) -> Tuple:
            """
            Create synthetic node classification dataset.
            
            TODO: Generate graph with community structure.
            """
            pass
        
        # Test
        print("\nExercise 10: Complete Training Loop")
        print("-" * 40)
        
        # Uncomment to test:
        # A, X, labels, train_mask, val_mask, test_mask = create_synthetic_dataset()
        # trainer = GNNTrainer(model, learning_rate=0.01)
        # trainer.train(A, X, labels, train_mask, val_mask, num_epochs=100)
        
        return GNNTrainer


def verify_implementations():
    """Run all exercise stubs."""
    print("=" * 60)
    print("Graph Neural Networks - Exercise Verification")
    print("=" * 60)
    
    exercises = GNNExercises()
    
    exercises.exercise_1_gcn_with_gradients()
    exercises.exercise_2_attention_variants()
    exercises.exercise_3_neighbor_sampling()
    exercises.exercise_4_gin_implementation()
    exercises.exercise_5_graph_pooling()
    exercises.exercise_6_oversmoothing_mitigation()
    exercises.exercise_7_heterogeneous_gnn()
    exercises.exercise_8_temporal_gnn()
    exercises.exercise_9_explainability()
    exercises.exercise_10_training_loop()
    
    print("\n" + "=" * 60)
    print("Complete the TODO sections in each exercise!")
    print("=" * 60)


# =============================================================================
# Solutions (Reference Implementation)
# =============================================================================

class Solutions:
    """Reference solutions for exercises."""
    
    @staticmethod
    def solution_1_gcn_gradients():
        """Solution for Exercise 1."""
        
        class GCNLayerWithGrad:
            def __init__(self, in_features: int, out_features: int):
                scale = np.sqrt(2.0 / (in_features + out_features))
                self.W = np.random.randn(in_features, out_features) * scale
                self.b = np.zeros(out_features)
                self.cache = {}
            
            def forward(self, A_hat: np.ndarray, H: np.ndarray) -> np.ndarray:
                Z = A_hat @ H @ self.W + self.b
                H_out = np.maximum(0, Z)  # ReLU
                
                self.cache = {'A_hat': A_hat, 'H': H, 'Z': Z}
                return H_out
            
            def backward(self, grad_output: np.ndarray):
                A_hat = self.cache['A_hat']
                H = self.cache['H']
                Z = self.cache['Z']
                
                # Backprop through ReLU
                grad_Z = grad_output * (Z > 0)
                
                # Gradients
                grad_W = (A_hat @ H).T @ grad_Z
                grad_b = grad_Z.sum(axis=0)
                grad_H = A_hat.T @ grad_Z @ self.W.T
                
                return grad_H, grad_W, grad_b
        
        return GCNLayerWithGrad
    
    @staticmethod
    def solution_2_attention():
        """Solution for Exercise 2."""
        
        def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                        A: np.ndarray) -> np.ndarray:
            d = Q.shape[1]
            scores = Q @ K.T / np.sqrt(d)
            
            # Mask non-edges
            A_with_self = A + np.eye(len(A))
            scores = np.where(A_with_self > 0, scores, -1e9)
            
            # Softmax
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            attention = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-10)
            
            return attention
        
        return scaled_dot_product_attention
    
    @staticmethod
    def solution_3_sampling():
        """Solution for Exercise 3."""
        
        def layer_wise_sampling(adj_list, batch_nodes, fanouts):
            layers = []
            current_nodes = set(batch_nodes)
            
            for fanout in reversed(fanouts):
                sampled_adj = {}
                next_nodes = set()
                
                for v in current_nodes:
                    neighbors = adj_list.get(v, [])
                    if len(neighbors) <= fanout:
                        sampled = neighbors
                    else:
                        sampled = list(np.random.choice(neighbors, fanout, 
                                                        replace=False))
                    sampled_adj[v] = sampled
                    next_nodes.update(sampled)
                
                layers.append({
                    'target_nodes': list(current_nodes),
                    'sampled_adj': sampled_adj,
                    'source_nodes': list(next_nodes)
                })
                
                current_nodes = current_nodes.union(next_nodes)
            
            return layers[::-1]
        
        return layer_wise_sampling
    
    @staticmethod
    def solution_6_pairnorm():
        """Solution for Exercise 6 - PairNorm."""
        
        class PairNorm:
            def forward(self, H: np.ndarray) -> np.ndarray:
                # Center
                H_centered = H - H.mean(axis=0, keepdims=True)
                
                # Scale
                norm_sq = (H_centered ** 2).sum() / len(H)
                H_normed = H_centered / (np.sqrt(norm_sq) + 1e-10)
                
                return H_normed
        
        return PairNorm


if __name__ == "__main__":
    verify_implementations()
