# Graph Algorithms

## Overview

Graph algorithms are fundamental to many ML applications: shortest paths for network routing, minimum spanning trees for clustering, flow algorithms for assignment problems, and traversal algorithms for feature propagation in GNNs.

## Learning Objectives

- Implement classic graph algorithms from scratch
- Understand algorithmic complexity and trade-offs
- Apply algorithms to ML problems
- Recognize when to use each algorithm

## 1. Traversal Algorithms

### Breadth-First Search (BFS)

Explores graph level by level. Key properties:

- Finds **shortest path** (in unweighted graphs)
- Time: $O(V + E)$
- Space: $O(V)$

```
BFS from vertex 0:
Level 0: [0]
Level 1: [1, 3]
Level 2: [2, 4]
Level 3: [5]
```

**Applications in ML:**

- Computing k-hop neighborhoods
- Graph diameter estimation
- Connected component detection
- Feature propagation in GNNs

### Depth-First Search (DFS)

Explores as deep as possible before backtracking.

- Time: $O(V + E)$
- Space: $O(V)$

```
DFS from vertex 0: 0 → 1 → 2 → 4 → 3 → 5
```

**Applications in ML:**

- Topological sort for DAGs
- Cycle detection
- Strongly connected components
- Articulation points/bridges

### DFS Timestamps

```
Pre-order: [0, 1, 2, 4, 3, 5]   # Discovery order
Post-order: [4, 2, 1, 5, 3, 0]  # Finish order
```

## 2. Shortest Path Algorithms

### Single-Source Shortest Path (SSSP)

| Algorithm    | Graph Type   | Complexity          | Notes            |
| ------------ | ------------ | ------------------- | ---------------- |
| BFS          | Unweighted   | $O(V + E)$          | Simplest         |
| Dijkstra     | Non-negative | $O((V + E) \log V)$ | Most common      |
| Bellman-Ford | Any weights  | $O(VE)$             | Handles negative |
| A\*          | Any          | Varies              | With heuristic   |

### Dijkstra's Algorithm

Greedy algorithm for non-negative edge weights.

$$d[v] = \min_{u \in N^-(v)} (d[u] + w(u, v))$$

**Relaxation step:**

```
if d[u] + w(u,v) < d[v]:
    d[v] = d[u] + w(u,v)
    parent[v] = u
```

### Bellman-Ford Algorithm

Handles negative weights, detects negative cycles.

Repeat $|V| - 1$ times:

```
for each edge (u, v):
    relax(u, v)
```

### All-Pairs Shortest Path (APSP)

**Floyd-Warshall:** $O(V^3)$

$$d_{ij}^{(k)} = \min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)} + d_{kj}^{(k-1)})$$

**Applications in ML:**

- Graph kernels (shortest-path kernel)
- Network centrality measures
- Embedding distances

## 3. Minimum Spanning Tree (MST)

A tree connecting all vertices with minimum total edge weight.

### Properties

- Has exactly $|V| - 1$ edges
- Unique if all weights are distinct
- Cut property: Minimum edge crossing any cut is in MST

### Algorithms

| Algorithm | Complexity    | Approach                   |
| --------- | ------------- | -------------------------- |
| Kruskal   | $O(E \log E)$ | Sort edges, use Union-Find |
| Prim      | $O(E \log V)$ | Grow tree from vertex      |
| Borůvka   | $O(E \log V)$ | Parallel-friendly          |

### Kruskal's Algorithm

```
1. Sort edges by weight
2. For each edge (u, v) in sorted order:
   If u and v are in different components:
      Add edge to MST
      Union(u, v)
```

### Prim's Algorithm

```
1. Start from arbitrary vertex
2. Maintain priority queue of edges to unexplored vertices
3. Always add minimum-weight edge to new vertex
```

### Applications in ML

- **Hierarchical clustering:** Single-linkage clustering = MST
- **Feature selection:** Build MST on correlation graph
- **Image segmentation:** Pixel similarity graph

## 4. Flow Algorithms

### Maximum Flow

Find maximum flow from source $s$ to sink $t$.

**Max-Flow Min-Cut Theorem:**
$$\max \text{flow} = \min \text{cut capacity}$$

### Ford-Fulkerson Method

```
While there exists augmenting path from s to t:
    Find path with BFS/DFS
    Augment flow along path
```

**Edmonds-Karp:** Use BFS, $O(VE^2)$

### Bipartite Matching

Maximum matching in bipartite graph = Maximum flow with unit capacities.

**Applications in ML:**

- Assignment problems
- Resource allocation
- Recommendation systems

## 5. Strongly Connected Components (SCC)

For directed graphs: maximal subgraphs where every vertex is reachable from every other.

### Kosaraju's Algorithm

1. Run DFS, record finish times
2. Transpose graph
3. Run DFS in reverse finish time order
4. Each DFS tree is an SCC

### Tarjan's Algorithm

Single DFS with lowlink values. $O(V + E)$

**Applications in ML:**

- Finding cyclic dependencies
- DAG conversion for neural architecture
- Identifying tightly connected communities

## 6. Topological Sort

Linear ordering of vertices in DAG where $(u, v) \in E \Rightarrow u$ before $v$.

### Kahn's Algorithm (BFS)

```
1. Find all vertices with in-degree 0
2. Remove vertex, add to result, update in-degrees
3. Repeat until empty
```

### DFS-based

```
Post-order of DFS gives reverse topological order
```

**Applications in ML:**

- Computation graphs (backprop order)
- Dependency resolution
- Task scheduling

## 7. Graph Coloring

Assign colors to vertices such that no adjacent vertices share a color.

### Chromatic Number

$\chi(G)$ = minimum number of colors needed

**Bounds:**

- $\chi(G) \leq \Delta(G) + 1$ (maximum degree + 1)
- $\chi(G) = 2$ iff $G$ is bipartite

### Greedy Coloring

```
For each vertex v in some order:
    Assign smallest color not used by neighbors
```

**Applications in ML:**

- Feature assignment
- Parallel scheduling
- Register allocation in compilers

## 8. PageRank

Importance score based on link structure.

$$PR(v) = \frac{1-d}{n} + d \sum_{u \in N^-(v)} \frac{PR(u)}{|N^+(u)|}$$

where $d \approx 0.85$ is damping factor.

**Matrix form:**
$$\mathbf{r} = (1-d)\frac{\mathbf{1}}{n} + d \cdot M \cdot \mathbf{r}$$

**Power iteration:**

```
r = [1/n, 1/n, ..., 1/n]
repeat until convergence:
    r = (1-d)/n + d * M @ r
```

**Applications in ML:**

- Node importance in GNNs
- Feature weighting
- Recommendation ranking

## 9. Community Detection

### Modularity

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

where $k_i$ = degree of $i$, $c_i$ = community of $i$.

### Algorithms

| Algorithm           | Approach                 | Complexity    |
| ------------------- | ------------------------ | ------------- |
| Louvain             | Greedy modularity        | $O(n \log n)$ |
| Label Propagation   | Iterative voting         | $O(m)$        |
| Spectral Clustering | Laplacian eigenvectors   | $O(n^3)$      |
| Girvan-Newman       | Edge betweenness removal | $O(m^2 n)$    |

**Applications in ML:**

- Customer segmentation
- Protein function prediction
- Social network analysis

## 10. Random Walks

### Simple Random Walk

At each step, move to random neighbor with probability $1/\deg(v)$.

**Transition matrix:**
$$P_{ij} = \frac{A_{ij}}{\deg(i)}$$

### Hitting Time

$h(i, j)$ = expected steps to reach $j$ from $i$

### Stationary Distribution

For connected, non-bipartite graph:
$$\pi_v = \frac{\deg(v)}{2|E|}$$

**Applications in ML:**

- Graph embeddings (DeepWalk, Node2Vec)
- Semi-supervised learning (label propagation)
- Recommendation (random walk with restart)

## 11. Algorithm Summary

| Problem           | Best Algorithm  | Complexity        |
| ----------------- | --------------- | ----------------- |
| Traversal         | BFS/DFS         | $O(V + E)$        |
| SSSP (unweighted) | BFS             | $O(V + E)$        |
| SSSP (weighted)   | Dijkstra        | $O((V+E) \log V)$ |
| APSP              | Floyd-Warshall  | $O(V^3)$          |
| MST               | Kruskal/Prim    | $O(E \log V)$     |
| Max Flow          | Edmonds-Karp    | $O(VE^2)$         |
| SCC               | Tarjan          | $O(V + E)$        |
| Topological Sort  | Kahn/DFS        | $O(V + E)$        |
| PageRank          | Power iteration | $O(k(V + E))$     |

## Key Takeaways

1. **BFS/DFS** are building blocks for many algorithms
2. **Dijkstra** is the go-to for shortest paths with non-negative weights
3. **MST algorithms** directly connect to clustering
4. **PageRank** and random walks underlie many graph ML methods
5. **Community detection** is essentially graph clustering
6. **Topological sort** is essential for computation graphs

## Practice Problems

1. Implement Dijkstra with a min-heap
2. Find all bridges in a graph using DFS
3. Implement the Louvain algorithm for community detection
4. Compute PageRank using power iteration
5. Design an algorithm to find the shortest path that visits all nodes (TSP variant)
