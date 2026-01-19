"""
Graph Representations - Examples
================================

Practical implementations showing different ways to represent graphs
and their trade-offs for machine learning applications.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Set, Optional, Any


# =============================================================================
# Example 1: Adjacency Matrix Representation
# =============================================================================

def example_adjacency_matrix():
    """
    Adjacency matrix: A[i,j] = 1 if edge (i,j) exists.
    
    Properties:
    - Space: O(n²)
    - Edge check: O(1)
    - Neighbors: O(n)
    """
    print("=" * 60)
    print("Example 1: Adjacency Matrix Representation")
    print("=" * 60)
    
    class AdjacencyMatrix:
        """Graph using adjacency matrix."""
        
        def __init__(self, n: int, directed: bool = False):
            self.n = n
            self.directed = directed
            self.matrix = np.zeros((n, n), dtype=np.float32)
        
        def add_edge(self, u: int, v: int, weight: float = 1.0):
            self.matrix[u, v] = weight
            if not self.directed:
                self.matrix[v, u] = weight
        
        def has_edge(self, u: int, v: int) -> bool:
            return self.matrix[u, v] != 0
        
        def get_neighbors(self, v: int) -> List[int]:
            return list(np.where(self.matrix[v] != 0)[0])
        
        def degree(self, v: int) -> int:
            return int(np.sum(self.matrix[v] != 0))
        
        def num_edges(self) -> int:
            count = np.sum(self.matrix != 0)
            return int(count if self.directed else count // 2)
        
        def get_matrix(self) -> np.ndarray:
            return self.matrix.copy()
    
    # Create graph
    #   0 --- 1
    #   |     |
    #   3 --- 2
    g = AdjacencyMatrix(4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for u, v in edges:
        g.add_edge(u, v)
    
    print("Graph: 0-1-2-3-0 (cycle)")
    print(f"\nAdjacency Matrix:\n{g.matrix.astype(int)}")
    print(f"\nEdge (0,1) exists: {g.has_edge(0, 1)}")
    print(f"Edge (0,2) exists: {g.has_edge(0, 2)}")
    print(f"Neighbors of 1: {g.get_neighbors(1)}")
    print(f"Degree of vertex 2: {g.degree(2)}")
    
    # Matrix powers for path counting
    A = g.get_matrix()
    A2 = A @ A
    A3 = A @ A @ A
    
    print(f"\nA² (2-hop paths):\n{A2.astype(int)}")
    print(f"Number of 2-hop paths from 0 to 2: {int(A2[0, 2])}")
    
    return g


# =============================================================================
# Example 2: Adjacency List Representation
# =============================================================================

def example_adjacency_list():
    """
    Adjacency list: Each vertex stores list of neighbors.
    
    Properties:
    - Space: O(n + m)
    - Edge check: O(deg) or O(1) with set
    - Neighbors: O(1) access
    """
    print("\n" + "=" * 60)
    print("Example 2: Adjacency List Representation")
    print("=" * 60)
    
    class AdjacencyList:
        """Graph using adjacency list."""
        
        def __init__(self, directed: bool = False):
            self.adj = defaultdict(list)
            self.directed = directed
            self._edges = 0
        
        def add_vertex(self, v: int):
            if v not in self.adj:
                self.adj[v] = []
        
        def add_edge(self, u: int, v: int, weight: float = 1.0):
            self.adj[u].append((v, weight))
            if not self.directed:
                self.adj[v].append((u, weight))
            self._edges += 1
        
        def has_edge(self, u: int, v: int) -> bool:
            return any(neighbor == v for neighbor, _ in self.adj[u])
        
        def get_neighbors(self, v: int) -> List[int]:
            return [neighbor for neighbor, _ in self.adj[v]]
        
        def get_weighted_neighbors(self, v: int) -> List[Tuple[int, float]]:
            return self.adj[v].copy()
        
        def degree(self, v: int) -> int:
            return len(self.adj[v])
        
        def vertices(self) -> List[int]:
            return list(self.adj.keys())
        
        def num_vertices(self) -> int:
            return len(self.adj)
        
        def num_edges(self) -> int:
            return self._edges
    
    # Create weighted graph
    g = AdjacencyList()
    weighted_edges = [
        (0, 1, 1.0), (0, 2, 4.0), (1, 2, 2.0),
        (1, 3, 5.0), (2, 3, 1.0)
    ]
    
    for u, v, w in weighted_edges:
        g.add_edge(u, v, w)
    
    print("Weighted graph adjacency list:")
    for v in sorted(g.vertices()):
        neighbors = g.get_weighted_neighbors(v)
        print(f"  {v}: {neighbors}")
    
    print(f"\nNeighbors of vertex 1: {g.get_neighbors(1)}")
    print(f"Degree of vertex 2: {g.degree(2)}")
    print(f"Total vertices: {g.num_vertices()}")
    print(f"Total edges: {g.num_edges()}")
    
    return g


# =============================================================================
# Example 3: Edge List Representation
# =============================================================================

def example_edge_list():
    """
    Edge list: Simple list of (source, target) pairs.
    
    Properties:
    - Space: O(m)
    - Edge check: O(m)
    - Simple but inefficient for queries
    """
    print("\n" + "=" * 60)
    print("Example 3: Edge List Representation")
    print("=" * 60)
    
    class EdgeList:
        """Graph using edge list."""
        
        def __init__(self, directed: bool = False):
            self.edges: List[Tuple[int, int, float]] = []
            self.directed = directed
        
        def add_edge(self, u: int, v: int, weight: float = 1.0):
            self.edges.append((u, v, weight))
        
        def has_edge(self, u: int, v: int) -> bool:
            for src, dst, _ in self.edges:
                if src == u and dst == v:
                    return True
                if not self.directed and src == v and dst == u:
                    return True
            return False
        
        def get_neighbors(self, v: int) -> List[int]:
            neighbors = []
            for src, dst, _ in self.edges:
                if src == v:
                    neighbors.append(dst)
                elif not self.directed and dst == v:
                    neighbors.append(src)
            return neighbors
        
        def vertices(self) -> Set[int]:
            verts = set()
            for u, v, _ in self.edges:
                verts.add(u)
                verts.add(v)
            return verts
        
        def to_adjacency_matrix(self) -> np.ndarray:
            n = max(max(u, v) for u, v, _ in self.edges) + 1
            A = np.zeros((n, n))
            for u, v, w in self.edges:
                A[u, v] = w
                if not self.directed:
                    A[v, u] = w
            return A
    
    # Create graph from edge list
    g = EdgeList()
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
    
    for u, v in edges:
        g.add_edge(u, v)
    
    print(f"Edge list: {[(u, v) for u, v, _ in g.edges]}")
    print(f"Vertices: {sorted(g.vertices())}")
    print(f"Neighbors of 0: {g.get_neighbors(0)}")
    print(f"\nAdjacency matrix from edge list:")
    print(g.to_adjacency_matrix().astype(int))
    
    return g


# =============================================================================
# Example 4: COO Format (Coordinate List)
# =============================================================================

def example_coo_format():
    """
    COO format: Parallel arrays for edges.
    
    This is the standard format for PyTorch Geometric.
    edge_index[0] = source nodes
    edge_index[1] = target nodes
    """
    print("\n" + "=" * 60)
    print("Example 4: COO Format (PyTorch Geometric Style)")
    print("=" * 60)
    
    class COOGraph:
        """Graph in COO format."""
        
        def __init__(self, num_nodes: int):
            self.num_nodes = num_nodes
            self.edge_index = [[], []]  # [sources, targets]
            self.edge_attr = []  # Optional edge attributes
        
        def add_edge(self, u: int, v: int, attr: Any = None):
            self.edge_index[0].append(u)
            self.edge_index[1].append(v)
            if attr is not None:
                self.edge_attr.append(attr)
        
        def add_undirected_edge(self, u: int, v: int, attr: Any = None):
            self.add_edge(u, v, attr)
            self.add_edge(v, u, attr)
        
        def get_edge_index(self) -> np.ndarray:
            return np.array(self.edge_index)
        
        def num_edges(self) -> int:
            return len(self.edge_index[0])
        
        def to_adjacency_matrix(self) -> np.ndarray:
            A = np.zeros((self.num_nodes, self.num_nodes))
            for i in range(self.num_edges()):
                u, v = self.edge_index[0][i], self.edge_index[1][i]
                A[u, v] = 1
            return A
        
        def neighbors(self, v: int) -> List[int]:
            """Get outgoing neighbors."""
            neighbors = []
            for i, src in enumerate(self.edge_index[0]):
                if src == v:
                    neighbors.append(self.edge_index[1][i])
            return neighbors
    
    # Create graph
    g = COOGraph(num_nodes=4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for u, v in edges:
        g.add_undirected_edge(u, v)
    
    edge_index = g.get_edge_index()
    print(f"Edge index (COO format):")
    print(f"  Source: {edge_index[0].tolist()}")
    print(f"  Target: {edge_index[1].tolist()}")
    print(f"\nNumber of edges: {g.num_edges()}")
    print(f"Adjacency matrix:\n{g.to_adjacency_matrix().astype(int)}")
    
    # Simulating message passing (like in GNNs)
    print("\n--- Message Passing Example ---")
    node_features = np.array([[1.0], [2.0], [3.0], [4.0]])  # 4 nodes, 1 feature
    print(f"Node features: {node_features.flatten()}")
    
    # Aggregate neighbors (sum aggregation)
    aggregated = np.zeros_like(node_features)
    for i in range(g.num_edges()):
        src, dst = edge_index[0, i], edge_index[1, i]
        aggregated[dst] += node_features[src]
    
    print(f"After sum aggregation: {aggregated.flatten()}")
    
    return g


# =============================================================================
# Example 5: CSR Format (Compressed Sparse Row)
# =============================================================================

def example_csr_format():
    """
    CSR format: Efficient for row-wise operations.
    
    row_ptr[i] : start index for row i's edges
    col_idx    : column indices
    data       : edge values
    """
    print("\n" + "=" * 60)
    print("Example 5: CSR Format (Compressed Sparse Row)")
    print("=" * 60)
    
    class CSRGraph:
        """Graph in CSR format."""
        
        def __init__(self, n: int, edges: List[Tuple[int, int]], 
                     weights: Optional[List[float]] = None):
            # Sort edges by source
            if weights is None:
                weights = [1.0] * len(edges)
            
            edge_data = sorted(zip(edges, weights), key=lambda x: x[0][0])
            
            self.n = n
            self.row_ptr = [0] * (n + 1)
            self.col_idx = []
            self.data = []
            
            for (u, v), w in edge_data:
                self.row_ptr[u + 1] += 1
                self.col_idx.append(v)
                self.data.append(w)
            
            # Cumulative sum for row_ptr
            for i in range(1, n + 1):
                self.row_ptr[i] += self.row_ptr[i - 1]
            
            self.row_ptr = np.array(self.row_ptr)
            self.col_idx = np.array(self.col_idx)
            self.data = np.array(self.data)
        
        def get_neighbors(self, v: int) -> np.ndarray:
            start, end = self.row_ptr[v], self.row_ptr[v + 1]
            return self.col_idx[start:end]
        
        def get_weighted_neighbors(self, v: int) -> Tuple[np.ndarray, np.ndarray]:
            start, end = self.row_ptr[v], self.row_ptr[v + 1]
            return self.col_idx[start:end], self.data[start:end]
        
        def degree(self, v: int) -> int:
            return self.row_ptr[v + 1] - self.row_ptr[v]
        
        def spmv(self, x: np.ndarray) -> np.ndarray:
            """Sparse matrix-vector multiply: A @ x"""
            result = np.zeros(self.n)
            for i in range(self.n):
                start, end = self.row_ptr[i], self.row_ptr[i + 1]
                for j in range(start, end):
                    result[i] += self.data[j] * x[self.col_idx[j]]
            return result
        
        def to_dense(self) -> np.ndarray:
            A = np.zeros((self.n, self.n))
            for i in range(self.n):
                start, end = self.row_ptr[i], self.row_ptr[i + 1]
                for j in range(start, end):
                    A[i, self.col_idx[j]] = self.data[j]
            return A
    
    # Create symmetric adjacency
    edges = [(0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3), (3, 0), (3, 2)]
    g = CSRGraph(4, edges)
    
    print("CSR representation:")
    print(f"  row_ptr: {g.row_ptr.tolist()}")
    print(f"  col_idx: {g.col_idx.tolist()}")
    print(f"  data:    {g.data.tolist()}")
    
    print(f"\nNeighbors of vertex 0: {g.get_neighbors(0).tolist()}")
    print(f"Neighbors of vertex 2: {g.get_neighbors(2).tolist()}")
    print(f"Degree of vertex 1: {g.degree(1)}")
    
    # Sparse matrix-vector multiply
    x = np.array([1.0, 2.0, 3.0, 4.0])
    Ax = g.spmv(x)
    print(f"\nSparse matrix-vector multiply:")
    print(f"  x = {x}")
    print(f"  A @ x = {Ax}")
    
    # Verify with dense
    A_dense = g.to_dense()
    print(f"  Dense verification: {A_dense @ x}")
    
    return g


# =============================================================================
# Example 6: SciPy Sparse Matrices
# =============================================================================

def example_scipy_sparse():
    """
    Using SciPy's sparse matrix implementations.
    
    Efficient for large graphs and linear algebra operations.
    """
    print("\n" + "=" * 60)
    print("Example 6: SciPy Sparse Matrices")
    print("=" * 60)
    
    from scipy import sparse
    
    # Create sparse matrix from COO data
    n = 5
    row = np.array([0, 0, 1, 2, 2, 3, 4])
    col = np.array([1, 3, 2, 1, 4, 4, 3])
    data = np.ones(len(row))
    
    # COO matrix
    A_coo = sparse.coo_matrix((data, (row, col)), shape=(n, n))
    print(f"COO matrix:\n{A_coo.toarray().astype(int)}")
    
    # Convert to CSR for efficient operations
    A_csr = A_coo.tocsr()
    print(f"\nCSR format:")
    print(f"  indptr: {A_csr.indptr}")
    print(f"  indices: {A_csr.indices}")
    print(f"  data: {A_csr.data}")
    
    # Make symmetric for undirected graph
    A_sym = A_csr + A_csr.T
    print(f"\nSymmetric (undirected) matrix:\n{A_sym.toarray().astype(int)}")
    
    # Operations
    x = np.ones(n)
    Ax = A_sym @ x  # SpMV
    print(f"\nA @ ones = {Ax}")  # Degree of each vertex
    
    # Memory comparison
    A_dense = A_sym.toarray()
    print(f"\nMemory comparison for {n}x{n} matrix:")
    print(f"  Dense: {A_dense.nbytes} bytes")
    print(f"  Sparse (CSR): {A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes} bytes")
    
    return A_csr


# =============================================================================
# Example 7: Incidence Matrix
# =============================================================================

def example_incidence_matrix():
    """
    Incidence matrix: B[v, e] indicates vertex v is incident to edge e.
    
    For undirected: B[v,e] = 1 if vertex v is endpoint of edge e
    For directed: B[v,e] = +1 (head), -1 (tail)
    """
    print("\n" + "=" * 60)
    print("Example 7: Incidence Matrix")
    print("=" * 60)
    
    class IncidenceGraph:
        """Graph using incidence matrix."""
        
        def __init__(self, n: int):
            self.n = n
            self.edges = []  # List of (u, v) pairs
        
        def add_edge(self, u: int, v: int):
            self.edges.append((u, v))
        
        def get_incidence_matrix_undirected(self) -> np.ndarray:
            """B[v, e] = 1 if vertex v is endpoint of edge e."""
            m = len(self.edges)
            B = np.zeros((self.n, m))
            for e, (u, v) in enumerate(self.edges):
                B[u, e] = 1
                B[v, e] = 1
            return B
        
        def get_incidence_matrix_directed(self) -> np.ndarray:
            """B[v, e] = -1 (tail), +1 (head)."""
            m = len(self.edges)
            B = np.zeros((self.n, m))
            for e, (u, v) in enumerate(self.edges):
                B[u, e] = -1  # Tail (source)
                B[v, e] = 1   # Head (target)
            return B
        
        def laplacian_from_incidence(self) -> np.ndarray:
            """L = B @ B^T for undirected graphs."""
            B = self.get_incidence_matrix_directed()
            return B @ B.T
    
    # Create graph
    g = IncidenceGraph(4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Square
    for u, v in edges:
        g.add_edge(u, v)
    
    print("Graph edges: 0→1, 1→2, 2→3, 3→0")
    
    B_undir = g.get_incidence_matrix_undirected()
    print(f"\nUndirected incidence matrix:\n{B_undir.astype(int)}")
    print("(Rows = vertices, Cols = edges)")
    
    B_dir = g.get_incidence_matrix_directed()
    print(f"\nDirected incidence matrix:\n{B_dir.astype(int)}")
    
    # Laplacian from incidence
    L = g.laplacian_from_incidence()
    print(f"\nLaplacian L = B @ B^T:\n{L.astype(int)}")
    
    # Verify: L = D - A
    A = np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    D = np.diag(np.sum(A, axis=1))
    L_verify = D - A
    print(f"L = D - A verification:\n{L_verify.astype(int)}")
    
    return g


# =============================================================================
# Example 8: Format Conversions
# =============================================================================

def example_format_conversions():
    """
    Converting between different graph representations.
    """
    print("\n" + "=" * 60)
    print("Example 8: Format Conversions")
    print("=" * 60)
    
    def edge_list_to_adj_list(edges, directed=False):
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            if not directed:
                adj[v].append(u)
        return dict(adj)
    
    def adj_list_to_matrix(adj_list, n):
        A = np.zeros((n, n))
        for u, neighbors in adj_list.items():
            for v in neighbors:
                A[u, v] = 1
        return A
    
    def matrix_to_edge_list(A):
        edges = []
        n = len(A)
        for i in range(n):
            for j in range(i + 1, n):  # Upper triangle for undirected
                if A[i, j] != 0:
                    edges.append((i, j))
        return edges
    
    def edge_list_to_coo(edges, n):
        row = [u for u, v in edges]
        col = [v for u, v in edges]
        # Add reverse for undirected
        row_sym = row + col
        col_sym = col + row
        return np.array([row_sym, col_sym])
    
    def coo_to_csr(edge_index, n):
        from scipy import sparse
        data = np.ones(edge_index.shape[1])
        return sparse.coo_matrix(
            (data, (edge_index[0], edge_index[1])),
            shape=(n, n)
        ).tocsr()
    
    # Start with edge list
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    n = 4
    
    print(f"Original edge list: {edges}")
    
    # Convert to adjacency list
    adj_list = edge_list_to_adj_list(edges)
    print(f"\n→ Adjacency list: {dict(adj_list)}")
    
    # Convert to matrix
    A = adj_list_to_matrix(adj_list, n)
    print(f"\n→ Adjacency matrix:\n{A.astype(int)}")
    
    # Back to edge list
    edges_back = matrix_to_edge_list(A)
    print(f"\n→ Back to edge list: {edges_back}")
    
    # To COO (PyTorch style)
    coo = edge_list_to_coo(edges, n)
    print(f"\n→ COO edge_index:\n  {coo}")
    
    # To CSR
    csr = coo_to_csr(coo, n)
    print(f"\n→ CSR format:")
    print(f"   indptr: {csr.indptr}")
    print(f"   indices: {csr.indices}")


# =============================================================================
# Example 9: Memory Comparison
# =============================================================================

def example_memory_comparison():
    """
    Compare memory usage of different representations.
    """
    print("\n" + "=" * 60)
    print("Example 9: Memory Comparison")
    print("=" * 60)
    
    import sys
    from scipy import sparse
    
    def analyze_memory(n, m, density=None):
        """Analyze memory for graph with n nodes and m edges."""
        if density is not None:
            m = int(density * n * n)
        
        print(f"\nGraph: {n} nodes, {m} edges, density={m/(n*n):.4f}")
        
        # Dense adjacency matrix
        dense_bytes = n * n * 8  # float64
        print(f"  Dense matrix:      {dense_bytes:>12,} bytes ({dense_bytes/1e6:.2f} MB)")
        
        # Adjacency list (approximate)
        # Each edge stored twice for undirected, ~16 bytes per edge (int + pointer)
        adj_list_bytes = n * 56 + m * 2 * 16  # dict overhead + entries
        print(f"  Adjacency list:    {adj_list_bytes:>12,} bytes ({adj_list_bytes/1e6:.2f} MB)")
        
        # Edge list
        edge_list_bytes = m * 2 * 8  # Two ints per edge
        print(f"  Edge list:         {edge_list_bytes:>12,} bytes ({edge_list_bytes/1e6:.2f} MB)")
        
        # COO
        coo_bytes = m * 2 * 8 + m * 8  # row, col, data arrays
        print(f"  COO:               {coo_bytes:>12,} bytes ({coo_bytes/1e6:.2f} MB)")
        
        # CSR
        csr_bytes = (n + 1) * 8 + m * 8 + m * 8  # indptr, indices, data
        print(f"  CSR:               {csr_bytes:>12,} bytes ({csr_bytes/1e6:.2f} MB)")
        
        return dense_bytes, adj_list_bytes, coo_bytes, csr_bytes
    
    # Small dense graph
    print("\n--- Small Dense Graph ---")
    analyze_memory(n=100, m=5000, density=0.5)
    
    # Medium sparse graph (social network like)
    print("\n--- Medium Sparse Graph (Social Network) ---")
    analyze_memory(n=10000, m=100000)  # Avg degree 20
    
    # Large sparse graph
    print("\n--- Large Sparse Graph ---")
    analyze_memory(n=1000000, m=10000000)  # 1M nodes, 10M edges
    
    # Conclusion
    print("\n" + "-" * 50)
    print("Conclusion:")
    print("- Dense: Only for small graphs (< 10K nodes)")
    print("- Sparse (CSR/COO): Essential for real-world graphs")
    print("- Memory savings: 100-10000x for sparse graphs")


# =============================================================================
# Example 10: Sparse Operations for GNNs
# =============================================================================

def example_gnn_sparse_operations():
    """
    Sparse operations commonly used in Graph Neural Networks.
    """
    print("\n" + "=" * 60)
    print("Example 10: Sparse Operations for GNNs")
    print("=" * 60)
    
    from scipy import sparse
    
    # Create a small graph
    n = 5
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (2, 4)]
    
    # Build symmetric adjacency
    row = [u for u, v in edges] + [v for u, v in edges]
    col = [v for u, v in edges] + [u for u, v in edges]
    data = np.ones(len(row))
    
    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    
    # Add self-loops: Ã = A + I
    A_tilde = A + sparse.eye(n)
    
    # Degree matrix
    degrees = np.array(A_tilde.sum(axis=1)).flatten()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
    
    # GCN normalization: D^(-1/2) @ Ã @ D^(-1/2)
    A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    
    print("Adjacency matrix A:")
    print(A.toarray().astype(int))
    
    print(f"\nDegrees (with self-loops): {degrees}")
    
    print("\nGCN normalized adjacency:")
    print(np.round(A_norm.toarray(), 3))
    
    # Simulate one GCN layer: H' = σ(Ã_norm @ H @ W)
    np.random.seed(42)
    d_in, d_out = 3, 2
    H = np.random.randn(n, d_in)  # Node features
    W = np.random.randn(d_in, d_out)  # Weights
    
    # Forward pass
    AH = A_norm @ H  # Aggregate
    AHW = AH @ W     # Transform
    H_out = np.maximum(0, AHW)  # ReLU
    
    print(f"\nGCN layer forward pass:")
    print(f"  Input features shape: {H.shape}")
    print(f"  Output features shape: {H_out.shape}")
    print(f"  Output:\n{np.round(H_out, 3)}")


# =============================================================================
# Example 11: Batched Graphs
# =============================================================================

def example_batched_graphs():
    """
    Combining multiple graphs into a batch for parallel processing.
    
    Standard approach in PyTorch Geometric.
    """
    print("\n" + "=" * 60)
    print("Example 11: Batched Graphs")
    print("=" * 60)
    
    class BatchedGraph:
        """Batch multiple graphs into one disconnected graph."""
        
        def __init__(self):
            self.edge_index = [[], []]
            self.node_features = []
            self.batch = []  # Which graph each node belongs to
            self.graph_sizes = []
            self.num_nodes = 0
        
        def add_graph(self, edge_index, node_features):
            """Add a graph to the batch."""
            graph_id = len(self.graph_sizes)
            n = len(node_features)
            
            # Offset edges
            for src in edge_index[0]:
                self.edge_index[0].append(src + self.num_nodes)
            for dst in edge_index[1]:
                self.edge_index[1].append(dst + self.num_nodes)
            
            # Add features
            self.node_features.extend(node_features)
            
            # Track batch membership
            self.batch.extend([graph_id] * n)
            self.graph_sizes.append(n)
            self.num_nodes += n
        
        def get_edge_index(self):
            return np.array(self.edge_index)
        
        def get_node_features(self):
            return np.array(self.node_features)
        
        def get_batch(self):
            return np.array(self.batch)
        
        def unbatch(self, node_outputs):
            """Split batched node outputs back to individual graphs."""
            results = []
            offset = 0
            for size in self.graph_sizes:
                results.append(node_outputs[offset:offset + size])
                offset += size
            return results
        
        def global_pool(self, node_features, method='mean'):
            """Pool node features to graph-level."""
            batch = self.get_batch()
            num_graphs = len(self.graph_sizes)
            
            if method == 'mean':
                pooled = np.zeros((num_graphs, node_features.shape[1]))
                counts = np.zeros(num_graphs)
                for i, b in enumerate(batch):
                    pooled[b] += node_features[i]
                    counts[b] += 1
                return pooled / counts[:, None]
            elif method == 'sum':
                pooled = np.zeros((num_graphs, node_features.shape[1]))
                for i, b in enumerate(batch):
                    pooled[b] += node_features[i]
                return pooled
            elif method == 'max':
                pooled = np.full((num_graphs, node_features.shape[1]), -np.inf)
                for i, b in enumerate(batch):
                    pooled[b] = np.maximum(pooled[b], node_features[i])
                return pooled
    
    # Create three small graphs
    # Graph 0: Triangle (3 nodes)
    g0_edges = [[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]]
    g0_features = [[1, 0], [0, 1], [1, 1]]
    
    # Graph 1: Line (2 nodes)
    g1_edges = [[0, 1], [1, 0]]
    g1_features = [[2, 0], [0, 2]]
    
    # Graph 2: Square (4 nodes)
    g2_edges = [[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 3, 0, 1, 2]]
    g2_features = [[1, 1], [2, 2], [3, 3], [4, 4]]
    
    # Batch them
    batch = BatchedGraph()
    batch.add_graph(g0_edges, g0_features)
    batch.add_graph(g1_edges, g1_features)
    batch.add_graph(g2_edges, g2_features)
    
    print(f"Batched graph:")
    print(f"  Total nodes: {batch.num_nodes}")
    print(f"  Graph sizes: {batch.graph_sizes}")
    print(f"  Batch indices: {batch.get_batch()}")
    print(f"  Edge index shape: {batch.get_edge_index().shape}")
    
    # Global pooling
    features = batch.get_node_features()
    pooled_mean = batch.global_pool(features, 'mean')
    pooled_sum = batch.global_pool(features, 'sum')
    
    print(f"\nNode features:\n{features}")
    print(f"\nMean pooling (graph-level):\n{pooled_mean}")
    print(f"Sum pooling (graph-level):\n{pooled_sum}")
    
    return batch


# =============================================================================
# Example 12: Efficient Neighbor Sampling
# =============================================================================

def example_neighbor_sampling():
    """
    Neighbor sampling for mini-batch training on large graphs.
    
    Key technique for scaling GNNs to large graphs.
    """
    print("\n" + "=" * 60)
    print("Example 12: Neighbor Sampling for Mini-Batch GNNs")
    print("=" * 60)
    
    class NeighborSampler:
        """Sample neighbors for mini-batch training."""
        
        def __init__(self, edge_index, num_nodes):
            # Build adjacency list for fast sampling
            self.adj_list = defaultdict(list)
            for i in range(len(edge_index[0])):
                src, dst = edge_index[0][i], edge_index[1][i]
                self.adj_list[src].append(dst)
            self.num_nodes = num_nodes
        
        def sample(self, batch_nodes, num_samples, num_layers):
            """
            Sample subgraph for batch_nodes.
            
            Args:
                batch_nodes: Target nodes to predict
                num_samples: Max neighbors to sample per layer
                num_layers: Number of GNN layers
            
            Returns:
                sampled_nodes: All nodes in computation graph
                edge_index: Edges in sampled subgraph
                layer_sizes: Nodes per layer
            """
            sampled_nodes = list(batch_nodes)
            sampled_set = set(batch_nodes)
            layer_sizes = [len(batch_nodes)]
            edge_index = [[], []]
            
            current_layer = list(batch_nodes)
            
            for layer in range(num_layers):
                next_layer = []
                for node in current_layer:
                    neighbors = self.adj_list[node]
                    if len(neighbors) <= num_samples:
                        sampled = neighbors
                    else:
                        sampled = list(np.random.choice(
                            neighbors, num_samples, replace=False
                        ))
                    
                    for neighbor in sampled:
                        edge_index[0].append(neighbor)
                        edge_index[1].append(node)
                        if neighbor not in sampled_set:
                            sampled_set.add(neighbor)
                            sampled_nodes.append(neighbor)
                            next_layer.append(neighbor)
                
                layer_sizes.append(len(next_layer))
                current_layer = next_layer
            
            return sampled_nodes, edge_index, layer_sizes
    
    # Create a larger graph
    np.random.seed(42)
    n = 100
    avg_degree = 10
    m = n * avg_degree // 2
    
    # Random edges
    edge_index = [[], []]
    for _ in range(m):
        u, v = np.random.randint(0, n, 2)
        if u != v:
            edge_index[0].extend([u, v])
            edge_index[1].extend([v, u])
    
    sampler = NeighborSampler(edge_index, n)
    
    # Sample for a mini-batch
    batch_nodes = [0, 1, 2, 3, 4]  # 5 target nodes
    num_samples = 5  # Sample up to 5 neighbors
    num_layers = 2   # 2-hop neighborhood
    
    sampled_nodes, sampled_edges, layer_sizes = sampler.sample(
        batch_nodes, num_samples, num_layers
    )
    
    print(f"Full graph: {n} nodes, {len(edge_index[0])} edges")
    print(f"\nMini-batch sampling:")
    print(f"  Target nodes: {batch_nodes}")
    print(f"  Sampled nodes: {len(sampled_nodes)}")
    print(f"  Sampled edges: {len(sampled_edges[0])}")
    print(f"  Layer sizes: {layer_sizes}")
    print(f"  Compression: {len(sampled_nodes)/n*100:.1f}% of nodes")
    
    # This subgraph can now be used for a forward pass
    print(f"\nComputation graph contains:")
    print(f"  Layer 0 (targets): {layer_sizes[0]} nodes")
    print(f"  Layer 1 (1-hop): {layer_sizes[1]} nodes")
    print(f"  Layer 2 (2-hop): {layer_sizes[2]} nodes")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    example_adjacency_matrix()
    example_adjacency_list()
    example_edge_list()
    example_coo_format()
    example_csr_format()
    example_scipy_sparse()
    example_incidence_matrix()
    example_format_conversions()
    example_memory_comparison()
    example_gnn_sparse_operations()
    example_batched_graphs()
    example_neighbor_sampling()
    
    print("\n" + "=" * 60)
    print("All Graph Representation Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
