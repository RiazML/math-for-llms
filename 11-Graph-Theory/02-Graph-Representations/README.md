# Graph Representations

## Overview

Choosing the right graph representation is crucial for algorithm efficiency. Different representations offer trade-offs between memory usage, query time, and update time. Understanding these is essential for implementing graph neural networks and large-scale graph processing.

## Learning Objectives

- Understand different graph representations
- Analyze space-time trade-offs
- Convert between representations
- Choose appropriate representations for ML tasks

## 1. Adjacency Matrix

### Definition

For a graph $G = (V, E)$ with $n = |V|$ vertices, the **adjacency matrix** $A \in \{0,1\}^{n \times n}$:

$$A_{ij} = \begin{cases} 1 & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

```
Graph:          Adjacency Matrix:
   1 ─── 2           1  2  3  4
   │     │      1 [  0  1  0  1 ]
   │     │      2 [  1  0  1  0 ]
   4 ─── 3      3 [  0  1  0  1 ]
                4 [  1  0  1  0 ]
```

### Properties

**Undirected graph:** $A$ is symmetric ($A = A^T$)

**Weighted graph:** $A_{ij} = w_{ij}$ (edge weight)

**Self-loops:** $A_{ii} = 1$ if self-loop at vertex $i$

### Complexity

| Operation        | Complexity |
| ---------------- | ---------- |
| Space            | $O(n^2)$   |
| Check edge (i,j) | $O(1)$     |
| Get neighbors    | $O(n)$     |
| Add/remove edge  | $O(1)$     |
| Add vertex       | $O(n^2)$   |

### Advantages

1. **Matrix operations:** Powers, eigenvalues, spectral analysis
2. **O(1) edge lookup:** Instant edge existence check
3. **Graph products:** Easy to compute

### Disadvantages

1. **Quadratic space:** Impractical for large sparse graphs
2. **Slow neighbor enumeration:** Must scan entire row
3. **Difficult dynamic updates:** Adding vertices is expensive

### ML Applications

- **Spectral methods:** Eigendecomposition of $A$ or Laplacian
- **Message passing:** $A \cdot X$ aggregates neighbor features
- **GCN:** Uses normalized adjacency $\tilde{A} = D^{-1/2}AD^{-1/2}$

## 2. Adjacency List

### Definition

Each vertex stores a list of its neighbors:

```python
adj_list = {
    1: [2, 4],
    2: [1, 3],
    3: [2, 4],
    4: [1, 3]
}
```

```
Graph:          Adjacency List:
   1 ─── 2      1 → [2, 4]
   │     │      2 → [1, 3]
   │     │      3 → [2, 4]
   4 ─── 3      4 → [1, 3]
```

### Complexity

| Operation        | Complexity                               |
| ---------------- | ---------------------------------------- |
| Space            | $O(n + m)$                               |
| Check edge (i,j) | $O(\deg(i))$                             |
| Get neighbors    | $O(1)$ to access, $O(\deg)$ to enumerate |
| Add edge         | $O(1)$                                   |
| Remove edge      | $O(\deg)$                                |

### Advantages

1. **Space efficient for sparse graphs:** Only stores existing edges
2. **Fast neighbor enumeration:** Direct access to neighbors
3. **Easy to extend:** Adding vertices is trivial

### Disadvantages

1. **Edge lookup:** Linear in degree (can use set for O(1))
2. **No matrix operations:** Must convert for linear algebra

### ML Applications

- **Graph sampling:** Random neighbor selection
- **Mini-batching:** Subgraph extraction
- **Message passing implementations**

## 3. Edge List

### Definition

List of all edges as tuples:

```python
edge_list = [(1, 2), (1, 4), (2, 3), (3, 4)]
```

For weighted graphs:

```python
weighted_edges = [(1, 2, 0.5), (1, 4, 1.2), ...]
```

### Complexity

| Operation     | Complexity |
| ------------- | ---------- |
| Space         | $O(m)$     |
| Check edge    | $O(m)$     |
| Get neighbors | $O(m)$     |
| Add edge      | $O(1)$     |

### Advantages

1. **Minimal space:** Just stores edges
2. **Easy I/O:** Simple to read/write from files
3. **Edge-centric operations:** Good for edge sampling

### Disadvantages

1. **Slow queries:** Everything requires scanning
2. **No structure:** Poor for most algorithms

### ML Applications

- **Data loading:** Common input format
- **Edge sampling:** For link prediction
- **Batch processing:** GPU edge operations

## 4. COO Format (Coordinate List)

### Definition

Three parallel arrays storing edges:

- `row`: Source vertices
- `col`: Target vertices
- `data`: Edge weights (optional)

```python
# Edges: (0,1), (0,3), (1,2), (2,3)
row  = [0, 0, 1, 2]
col  = [1, 3, 2, 3]
data = [1, 1, 1, 1]  # Optional weights
```

### PyTorch Geometric Format

```python
edge_index = torch.tensor([
    [0, 0, 1, 2],  # Source nodes
    [1, 3, 2, 3]   # Target nodes
])
```

### Complexity

Same as edge list but with array access patterns.

### ML Applications

- **PyTorch Geometric:** Standard format
- **GPU computation:** Efficient memory layout
- **Sparse operations:** Easy conversion to CSR/CSC

## 5. CSR Format (Compressed Sparse Row)

### Definition

Three arrays for efficient row access:

- `row_ptr`: Start of each row's edges
- `col_idx`: Column indices of edges
- `data`: Edge values

```
Matrix:             CSR:
[0 1 0 1]          row_ptr = [0, 2, 4, 6, 8]
[1 0 1 0]          col_idx = [1, 3, 0, 2, 1, 3, 0, 2]
[0 1 0 1]          data    = [1, 1, 1, 1, 1, 1, 1, 1]
[1 0 1 0]
```

```
Row 0: edges in col_idx[0:2] = [1, 3]
Row 1: edges in col_idx[2:4] = [0, 2]
Row 2: edges in col_idx[4:6] = [1, 3]
Row 3: edges in col_idx[6:8] = [0, 2]
```

### Complexity

| Operation              | Complexity                         |
| ---------------------- | ---------------------------------- |
| Space                  | $O(n + m)$                         |
| Get row neighbors      | $O(1)$ access, $O(\deg)$ enumerate |
| Matrix-vector multiply | $O(m)$                             |

### Advantages

1. **Efficient row access:** Fast neighbor enumeration
2. **Memory efficient:** Optimal for sparse matrices
3. **Fast SpMV:** Sparse matrix-vector multiplication

### ML Applications

- **GNN implementations:** Efficient message passing
- **Sparse linear algebra:** sciPy sparse matrices
- **Large-scale graphs:** Billions of edges

## 6. CSC Format (Compressed Sparse Column)

### Definition

Same as CSR but organized by columns:

- `col_ptr`: Start of each column
- `row_idx`: Row indices
- `data`: Values

### Use Case

- Efficient column access
- Transposed operations
- Some GNN variants need incoming edges

## 7. Incidence Matrix

### Definition

$B \in \{-1, 0, 1\}^{n \times m}$ where rows = vertices, columns = edges:

$$B_{ve} = \begin{cases} 1 & \text{if } v \text{ is head of } e \\ -1 & \text{if } v \text{ is tail of } e \\ 0 & \text{otherwise} \end{cases}$$

For undirected graphs, use $\{0, 1\}$.

```
Graph:           Incidence Matrix:
   1 ─e1─ 2           e1 e2 e3 e4
   │      │      1 [   1  0  0  1 ]
  e4     e2      2 [   1  1  0  0 ]
   │      │      3 [   0  1  1  0 ]
   4 ─e3─ 3      4 [   0  0  1  1 ]
```

### Properties

- **Laplacian:** $L = BB^T$ (for undirected)
- **Space:** $O(nm)$ - often worse than adjacency matrix

### ML Applications

- **Hypergraphs:** Generalizes to hyperedges
- **Network flow:** Natural for flow problems

## 8. Comparison Summary

```
Representation Comparison:

           Space    Edge     Neighbors   Matrix   ML Use
           Check    Query    Query       Ops

Matrix     O(n²)    O(1)     O(n)        Yes      Spectral
Adj List   O(n+m)   O(deg)   O(1)        No       Sampling
Edge List  O(m)     O(m)     O(m)        No       I/O
COO        O(m)     O(m)     O(m)        Convert  PyG
CSR        O(n+m)   O(log)   O(1)        Yes      GNN
CSC        O(n+m)   O(1)     O(log)      Yes      GNN
```

## 9. Conversions

### Edge List to Adjacency Matrix

```python
def edge_list_to_matrix(edges, n):
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected
    return A
```

### Adjacency Matrix to CSR

```python
from scipy.sparse import csr_matrix

A_sparse = csr_matrix(A_dense)
```

### COO to CSR (PyTorch Sparse)

```python
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
adj = torch.sparse_coo_tensor(edge_index,
                               torch.ones(3),
                               size=(3, 3))
adj_csr = adj.to_sparse_csr()
```

## 10. Choosing a Representation

### Decision Guide

```
                    Use Matrix
                        │
    Is graph dense? ────┤
           │            │
          No           Yes ──→ Adjacency Matrix
           │
           ↓
    Need spectral ops? ──Yes──→ Sparse Matrix (CSR)
           │
          No
           │
           ↓
    Need fast neighbor ──Yes──→ Adjacency List
    enumeration?
           │
          No
           │
           ↓
    GPU computation? ────Yes──→ COO/Edge Index
           │
          No
           │
           ↓
    Just storing ────────────→ Edge List
```

### By Graph Size

| Graph Size | Edges  | Recommended           |
| ---------- | ------ | --------------------- |
| < 1K nodes | Any    | Matrix or List        |
| 1K - 100K  | Sparse | Adjacency List or CSR |
| 100K - 1M  | Sparse | CSR/CSC               |
| > 1M       | Sparse | CSR + edge list files |

## Summary

| Representation   | Best For                             |
| ---------------- | ------------------------------------ |
| Adjacency Matrix | Small dense graphs, spectral methods |
| Adjacency List   | Medium graphs, sampling, traversal   |
| Edge List        | Data I/O, edge operations            |
| COO              | GPU operations, PyTorch Geometric    |
| CSR/CSC          | Large sparse graphs, GNN, SpMV       |
| Incidence        | Hypergraphs, flow problems           |

## Key Takeaways

1. **Space-time trade-off:** Matrix is fast but memory-heavy
2. **Sparse formats:** Essential for real-world graphs
3. **ML frameworks:** Each has preferred format (COO for PyG)
4. **Convert as needed:** Different ops need different formats
5. **Profile your workload:** Choose based on dominant operations

## Practice Problems

1. Convert an edge list to CSR format manually
2. Implement sparse matrix-vector multiplication using CSR
3. Compare memory usage of matrix vs CSR for a 10K node graph with 50K edges
4. Convert between PyTorch Geometric edge_index and scipy CSR
