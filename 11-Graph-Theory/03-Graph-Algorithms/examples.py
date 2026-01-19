"""
Graph Algorithms - Examples
===========================

Practical implementations of classic graph algorithms
with applications to machine learning problems.
"""

import numpy as np
from collections import defaultdict, deque
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any


# =============================================================================
# Example 1: Breadth-First Search (BFS)
# =============================================================================

def example_bfs():
    """
    BFS: Level-by-level exploration.
    
    Properties:
    - Finds shortest path in unweighted graphs
    - Time: O(V + E)
    - Space: O(V)
    """
    print("=" * 60)
    print("Example 1: Breadth-First Search (BFS)")
    print("=" * 60)
    
    def bfs(adj_list: Dict[int, List[int]], start: int) -> Tuple[Dict, Dict]:
        """
        BFS traversal returning distances and parents.
        
        Returns:
            distances: Shortest distance from start to each vertex
            parents: Parent in BFS tree (for path reconstruction)
        """
        distances = {start: 0}
        parents = {start: None}
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj_list.get(u, []):
                if v not in distances:
                    distances[v] = distances[u] + 1
                    parents[v] = u
                    queue.append(v)
        
        return distances, parents
    
    def reconstruct_path(parents: Dict, start: int, end: int) -> List[int]:
        """Reconstruct shortest path from start to end."""
        if end not in parents:
            return []
        
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parents[current]
        return path[::-1]
    
    def bfs_levels(adj_list: Dict[int, List[int]], start: int) -> List[List[int]]:
        """Return nodes grouped by level (distance from start)."""
        distances, _ = bfs(adj_list, start)
        
        max_dist = max(distances.values()) if distances else 0
        levels = [[] for _ in range(max_dist + 1)]
        
        for node, dist in distances.items():
            levels[dist].append(node)
        
        return levels
    
    # Create graph
    adj_list = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 5],
        3: [1],
        4: [1, 5],
        5: [2, 4]
    }
    
    print("Graph adjacency list:")
    for v, neighbors in sorted(adj_list.items()):
        print(f"  {v}: {neighbors}")
    
    # BFS from vertex 0
    distances, parents = bfs(adj_list, 0)
    print(f"\nBFS from vertex 0:")
    print(f"  Distances: {distances}")
    
    # Levels
    levels = bfs_levels(adj_list, 0)
    print(f"  Levels: {levels}")
    
    # Shortest path
    path = reconstruct_path(parents, 0, 5)
    print(f"  Shortest path 0 → 5: {path}")
    
    # K-hop neighbors (important for GNNs)
    def k_hop_neighbors(adj_list, start, k):
        distances, _ = bfs(adj_list, start)
        return [v for v, d in distances.items() if d <= k and v != start]
    
    print(f"\n  2-hop neighbors of 0: {k_hop_neighbors(adj_list, 0, 2)}")
    
    return bfs


# =============================================================================
# Example 2: Depth-First Search (DFS)
# =============================================================================

def example_dfs():
    """
    DFS: Explore as deep as possible before backtracking.
    
    Applications:
    - Topological sort
    - Cycle detection
    - Connected components
    - Articulation points
    """
    print("\n" + "=" * 60)
    print("Example 2: Depth-First Search (DFS)")
    print("=" * 60)
    
    def dfs_recursive(adj_list: Dict[int, List[int]], start: int) -> List[int]:
        """Simple DFS returning visited order."""
        visited = set()
        order = []
        
        def dfs(u):
            visited.add(u)
            order.append(u)
            for v in adj_list.get(u, []):
                if v not in visited:
                    dfs(v)
        
        dfs(start)
        return order
    
    def dfs_iterative(adj_list: Dict[int, List[int]], start: int) -> List[int]:
        """Iterative DFS using stack."""
        visited = set()
        order = []
        stack = [start]
        
        while stack:
            u = stack.pop()
            if u not in visited:
                visited.add(u)
                order.append(u)
                # Add neighbors in reverse order for consistent ordering
                for v in reversed(adj_list.get(u, [])):
                    if v not in visited:
                        stack.append(v)
        
        return order
    
    def dfs_with_timestamps(adj_list: Dict[int, List[int]]) -> Tuple[Dict, Dict]:
        """
        DFS with discovery and finish times.
        Useful for topological sort and SCC detection.
        """
        discovery = {}
        finish = {}
        time = [0]  # Using list for mutable counter
        
        def dfs(u):
            time[0] += 1
            discovery[u] = time[0]
            
            for v in adj_list.get(u, []):
                if v not in discovery:
                    dfs(v)
            
            time[0] += 1
            finish[u] = time[0]
        
        # Visit all vertices (handles disconnected graphs)
        for v in adj_list:
            if v not in discovery:
                dfs(v)
        
        return discovery, finish
    
    # Create graph
    adj_list = {
        0: [1, 2],
        1: [3],
        2: [3, 4],
        3: [5],
        4: [5],
        5: []
    }
    
    print("Graph (directed):")
    for v, neighbors in sorted(adj_list.items()):
        print(f"  {v} → {neighbors}")
    
    # DFS traversal
    order_rec = dfs_recursive(adj_list, 0)
    order_iter = dfs_iterative(adj_list, 0)
    
    print(f"\nDFS order (recursive): {order_rec}")
    print(f"DFS order (iterative): {order_iter}")
    
    # Timestamps
    discovery, finish = dfs_with_timestamps(adj_list)
    print(f"\nDFS timestamps:")
    for v in sorted(discovery.keys()):
        print(f"  Vertex {v}: discover={discovery[v]}, finish={finish[v]}")
    
    return dfs_recursive, dfs_with_timestamps


# =============================================================================
# Example 3: Dijkstra's Shortest Path
# =============================================================================

def example_dijkstra():
    """
    Dijkstra's algorithm for single-source shortest paths.
    
    Requirements: Non-negative edge weights
    Time: O((V + E) log V) with binary heap
    """
    print("\n" + "=" * 60)
    print("Example 3: Dijkstra's Shortest Path Algorithm")
    print("=" * 60)
    
    def dijkstra(adj_list: Dict[int, List[Tuple[int, float]]], 
                 start: int) -> Tuple[Dict, Dict]:
        """
        Dijkstra's algorithm using min-heap.
        
        Args:
            adj_list: {vertex: [(neighbor, weight), ...]}
            start: Source vertex
            
        Returns:
            distances: Shortest distance to each vertex
            parents: Parent in shortest path tree
        """
        distances = {start: 0}
        parents = {start: None}
        pq = [(0, start)]  # (distance, vertex)
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            for v, weight in adj_list.get(u, []):
                new_dist = d + weight
                if v not in distances or new_dist < distances[v]:
                    distances[v] = new_dist
                    parents[v] = u
                    heapq.heappush(pq, (new_dist, v))
        
        return distances, parents
    
    # Create weighted graph
    adj_list = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [(4, 3)],
        4: []
    }
    
    print("Weighted graph:")
    for v, edges in sorted(adj_list.items()):
        print(f"  {v}: {edges}")
    
    # Run Dijkstra
    distances, parents = dijkstra(adj_list, 0)
    
    print(f"\nShortest distances from vertex 0:")
    for v in sorted(distances.keys()):
        print(f"  To {v}: {distances[v]}")
    
    # Reconstruct path
    def get_path(parents, start, end):
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parents.get(current)
        return path[::-1]
    
    path_to_4 = get_path(parents, 0, 4)
    print(f"\nShortest path 0 → 4: {path_to_4}")
    print(f"Path cost: {distances[4]}")
    
    return dijkstra


# =============================================================================
# Example 4: Bellman-Ford Algorithm
# =============================================================================

def example_bellman_ford():
    """
    Bellman-Ford: Handles negative weights, detects negative cycles.
    
    Time: O(VE)
    """
    print("\n" + "=" * 60)
    print("Example 4: Bellman-Ford Algorithm")
    print("=" * 60)
    
    def bellman_ford(vertices: List[int], 
                     edges: List[Tuple[int, int, float]],
                     start: int) -> Tuple[Dict, Dict, bool]:
        """
        Bellman-Ford algorithm.
        
        Args:
            vertices: List of vertex IDs
            edges: List of (source, target, weight)
            start: Source vertex
            
        Returns:
            distances, parents, has_negative_cycle
        """
        distances = {v: float('inf') for v in vertices}
        parents = {v: None for v in vertices}
        distances[start] = 0
        
        # Relax all edges |V| - 1 times
        for _ in range(len(vertices) - 1):
            for u, v, w in edges:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    parents[v] = u
        
        # Check for negative cycles
        has_negative_cycle = False
        for u, v, w in edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                has_negative_cycle = True
                break
        
        return distances, parents, has_negative_cycle
    
    # Graph with some negative weights (but no negative cycle)
    vertices = [0, 1, 2, 3, 4]
    edges = [
        (0, 1, 4), (0, 2, 1),
        (1, 3, 1),
        (2, 1, -2), (2, 3, 5),  # Note: negative weight edge (2, 1)
        (3, 4, 3)
    ]
    
    print("Graph with negative weights:")
    for u, v, w in edges:
        print(f"  {u} --({w})--> {v}")
    
    distances, parents, has_neg_cycle = bellman_ford(vertices, edges, 0)
    
    print(f"\nShortest distances from vertex 0:")
    for v in sorted(distances.keys()):
        print(f"  To {v}: {distances[v]}")
    print(f"Has negative cycle: {has_neg_cycle}")
    
    # Example with negative cycle
    print("\n--- Graph with negative cycle ---")
    edges_neg_cycle = edges + [(4, 2, -8)]  # Creates negative cycle
    _, _, has_neg_cycle = bellman_ford(vertices, edges_neg_cycle, 0)
    print(f"Has negative cycle: {has_neg_cycle}")
    
    return bellman_ford


# =============================================================================
# Example 5: Floyd-Warshall All-Pairs Shortest Path
# =============================================================================

def example_floyd_warshall():
    """
    Floyd-Warshall: All-pairs shortest paths.
    
    Time: O(V³)
    Space: O(V²)
    """
    print("\n" + "=" * 60)
    print("Example 5: Floyd-Warshall All-Pairs Shortest Path")
    print("=" * 60)
    
    def floyd_warshall(adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Floyd-Warshall algorithm.
        
        Args:
            adj_matrix: Adjacency matrix (inf for no edge)
            
        Returns:
            dist: Distance matrix
            next_hop: Next vertex matrix for path reconstruction
        """
        n = len(adj_matrix)
        dist = adj_matrix.copy()
        next_hop = np.zeros((n, n), dtype=int)
        
        # Initialize next_hop
        for i in range(n):
            for j in range(n):
                if i != j and dist[i, j] < np.inf:
                    next_hop[i, j] = j
                else:
                    next_hop[i, j] = -1
        
        # Main algorithm
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
                        next_hop[i, j] = next_hop[i, k]
        
        return dist, next_hop
    
    def reconstruct_path(next_hop: np.ndarray, i: int, j: int) -> List[int]:
        """Reconstruct shortest path from i to j."""
        if next_hop[i, j] == -1:
            return []
        
        path = [i]
        while i != j:
            i = next_hop[i, j]
            path.append(i)
        return path
    
    # Create adjacency matrix
    INF = np.inf
    adj = np.array([
        [0, 3, INF, 7],
        [8, 0, 2, INF],
        [5, INF, 0, 1],
        [2, INF, INF, 0]
    ])
    
    print("Adjacency matrix (inf = no edge):")
    print(np.where(adj == INF, '∞', adj.astype(int)))
    
    dist, next_hop = floyd_warshall(adj)
    
    print("\nAll-pairs shortest distances:")
    print(dist.astype(int))
    
    # Reconstruct some paths
    print("\nShortest paths:")
    for i in range(4):
        for j in range(4):
            if i != j:
                path = reconstruct_path(next_hop, i, j)
                if path:
                    print(f"  {i} → {j}: {path} (cost: {int(dist[i, j])})")
    
    return floyd_warshall


# =============================================================================
# Example 6: Kruskal's MST Algorithm
# =============================================================================

def example_kruskal():
    """
    Kruskal's algorithm for Minimum Spanning Tree.
    
    Uses Union-Find data structure.
    Time: O(E log E) dominated by sorting
    """
    print("\n" + "=" * 60)
    print("Example 6: Kruskal's MST Algorithm")
    print("=" * 60)
    
    class UnionFind:
        """Union-Find with path compression and union by rank."""
        
        def __init__(self, n: int):
            self.parent = list(range(n))
            self.rank = [0] * n
        
        def find(self, x: int) -> int:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x: int, y: int) -> bool:
            """Union x and y. Returns True if they were in different sets."""
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            
            return True
    
    def kruskal(n: int, edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """
        Kruskal's algorithm.
        
        Args:
            n: Number of vertices
            edges: List of (u, v, weight)
            
        Returns:
            List of edges in MST
        """
        # Sort edges by weight
        sorted_edges = sorted(edges, key=lambda x: x[2])
        
        uf = UnionFind(n)
        mst = []
        
        for u, v, w in sorted_edges:
            if uf.union(u, v):
                mst.append((u, v, w))
                if len(mst) == n - 1:
                    break
        
        return mst
    
    # Create graph
    edges = [
        (0, 1, 4), (0, 2, 3),
        (1, 2, 1), (1, 3, 2),
        (2, 3, 4), (2, 4, 3),
        (3, 4, 2), (3, 5, 1),
        (4, 5, 6)
    ]
    n = 6
    
    print("Graph edges (u, v, weight):")
    for e in edges:
        print(f"  {e}")
    
    mst = kruskal(n, edges)
    total_weight = sum(w for _, _, w in mst)
    
    print(f"\nMinimum Spanning Tree:")
    for u, v, w in mst:
        print(f"  ({u}, {v}) weight={w}")
    print(f"Total MST weight: {total_weight}")
    
    return kruskal


# =============================================================================
# Example 7: Prim's MST Algorithm
# =============================================================================

def example_prim():
    """
    Prim's algorithm for Minimum Spanning Tree.
    
    Grows tree from a starting vertex.
    Time: O(E log V) with binary heap
    """
    print("\n" + "=" * 60)
    print("Example 7: Prim's MST Algorithm")
    print("=" * 60)
    
    def prim(n: int, adj_list: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
        """
        Prim's algorithm using min-heap.
        
        Args:
            n: Number of vertices
            adj_list: {vertex: [(neighbor, weight), ...]}
            
        Returns:
            List of edges in MST
        """
        mst = []
        visited = set([0])
        
        # Add all edges from vertex 0
        edges = [(w, 0, v) for v, w in adj_list.get(0, [])]
        heapq.heapify(edges)
        
        while edges and len(mst) < n - 1:
            w, u, v = heapq.heappop(edges)
            
            if v in visited:
                continue
            
            visited.add(v)
            mst.append((u, v, w))
            
            # Add edges from new vertex
            for neighbor, weight in adj_list.get(v, []):
                if neighbor not in visited:
                    heapq.heappush(edges, (weight, v, neighbor))
        
        return mst
    
    # Create adjacency list from edges
    edges = [
        (0, 1, 4), (0, 2, 3),
        (1, 2, 1), (1, 3, 2),
        (2, 3, 4), (2, 4, 3),
        (3, 4, 2), (3, 5, 1),
        (4, 5, 6)
    ]
    
    adj_list = defaultdict(list)
    for u, v, w in edges:
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    
    mst = prim(6, adj_list)
    total_weight = sum(w for _, _, w in mst)
    
    print("Prim's MST (starting from vertex 0):")
    for u, v, w in mst:
        print(f"  ({u}, {v}) weight={w}")
    print(f"Total MST weight: {total_weight}")
    
    return prim


# =============================================================================
# Example 8: Topological Sort
# =============================================================================

def example_topological_sort():
    """
    Topological sort for DAGs.
    
    Linear ordering where u comes before v for all edges (u, v).
    Essential for computation graphs in neural networks.
    """
    print("\n" + "=" * 60)
    print("Example 8: Topological Sort")
    print("=" * 60)
    
    def topological_sort_kahn(adj_list: Dict[int, List[int]], 
                              vertices: List[int]) -> List[int]:
        """
        Kahn's algorithm (BFS-based).
        
        Returns empty list if graph has cycle.
        """
        # Compute in-degrees
        in_degree = defaultdict(int)
        for v in vertices:
            in_degree[v] = 0
        for u in adj_list:
            for v in adj_list[u]:
                in_degree[v] += 1
        
        # Start with vertices having in-degree 0
        queue = deque([v for v in vertices if in_degree[v] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            
            for v in adj_list.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        # Check for cycle
        if len(result) != len(vertices):
            return []  # Cycle detected
        
        return result
    
    def topological_sort_dfs(adj_list: Dict[int, List[int]], 
                            vertices: List[int]) -> List[int]:
        """
        DFS-based topological sort.
        
        Returns reverse post-order.
        """
        visited = set()
        result = []
        has_cycle = [False]
        in_stack = set()
        
        def dfs(u):
            if has_cycle[0]:
                return
            
            visited.add(u)
            in_stack.add(u)
            
            for v in adj_list.get(u, []):
                if v in in_stack:
                    has_cycle[0] = True
                    return
                if v not in visited:
                    dfs(v)
            
            in_stack.remove(u)
            result.append(u)
        
        for v in vertices:
            if v not in visited:
                dfs(v)
        
        if has_cycle[0]:
            return []
        
        return result[::-1]
    
    # Neural network computation graph
    # 0: input, 1: conv1, 2: relu1, 3: conv2, 4: relu2, 5: fc, 6: output
    adj_list = {
        0: [1],      # input → conv1
        1: [2],      # conv1 → relu1
        2: [3],      # relu1 → conv2
        3: [4],      # conv2 → relu2
        4: [5],      # relu2 → fc
        5: [6],      # fc → output
        6: []
    }
    vertices = list(range(7))
    
    print("Neural network computation graph:")
    layer_names = ['input', 'conv1', 'relu1', 'conv2', 'relu2', 'fc', 'output']
    for v in vertices:
        deps = adj_list.get(v, [])
        if deps:
            print(f"  {layer_names[v]} → {[layer_names[d] for d in deps]}")
    
    order_kahn = topological_sort_kahn(adj_list, vertices)
    order_dfs = topological_sort_dfs(adj_list, vertices)
    
    print(f"\nTopological order (Kahn): {[layer_names[v] for v in order_kahn]}")
    print(f"Topological order (DFS):  {[layer_names[v] for v in order_dfs]}")
    
    return topological_sort_kahn, topological_sort_dfs


# =============================================================================
# Example 9: Strongly Connected Components
# =============================================================================

def example_scc():
    """
    Find Strongly Connected Components using Kosaraju's algorithm.
    
    SCC: Maximal subgraph where every vertex reaches every other.
    Time: O(V + E)
    """
    print("\n" + "=" * 60)
    print("Example 9: Strongly Connected Components")
    print("=" * 60)
    
    def kosaraju_scc(adj_list: Dict[int, List[int]], 
                     vertices: List[int]) -> List[List[int]]:
        """
        Kosaraju's algorithm for SCCs.
        
        1. DFS on original graph, record finish order
        2. Create transpose graph
        3. DFS on transpose in reverse finish order
        """
        # Step 1: DFS to get finish order
        visited = set()
        finish_order = []
        
        def dfs1(u):
            visited.add(u)
            for v in adj_list.get(u, []):
                if v not in visited:
                    dfs1(v)
            finish_order.append(u)
        
        for v in vertices:
            if v not in visited:
                dfs1(v)
        
        # Step 2: Create transpose graph
        transpose = defaultdict(list)
        for u in adj_list:
            for v in adj_list[u]:
                transpose[v].append(u)
        
        # Step 3: DFS on transpose in reverse finish order
        visited = set()
        sccs = []
        
        def dfs2(u, component):
            visited.add(u)
            component.append(u)
            for v in transpose.get(u, []):
                if v not in visited:
                    dfs2(v, component)
        
        for v in reversed(finish_order):
            if v not in visited:
                component = []
                dfs2(v, component)
                sccs.append(component)
        
        return sccs
    
    # Graph with 3 SCCs
    adj_list = {
        0: [1],
        1: [2],
        2: [0, 3],  # 0,1,2 form an SCC
        3: [4],
        4: [5],
        5: [3],     # 3,4,5 form an SCC
        6: [5, 7],
        7: [6]      # 6,7 form an SCC
    }
    vertices = list(range(8))
    
    print("Directed graph with SCCs:")
    for v in sorted(adj_list.keys()):
        print(f"  {v} → {adj_list[v]}")
    
    sccs = kosaraju_scc(adj_list, vertices)
    
    print(f"\nStrongly Connected Components:")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i + 1}: {sorted(scc)}")
    
    return kosaraju_scc


# =============================================================================
# Example 10: PageRank Algorithm
# =============================================================================

def example_pagerank():
    """
    PageRank: Node importance based on link structure.
    
    PR(v) = (1-d)/n + d * Σ PR(u)/out_degree(u)
    
    Used in GNNs for node importance weighting.
    """
    print("\n" + "=" * 60)
    print("Example 10: PageRank Algorithm")
    print("=" * 60)
    
    def pagerank(adj_list: Dict[int, List[int]], 
                 n: int,
                 damping: float = 0.85,
                 max_iter: int = 100,
                 tol: float = 1e-6) -> np.ndarray:
        """
        Compute PageRank using power iteration.
        
        Args:
            adj_list: {vertex: [outgoing neighbors]}
            n: Number of vertices
            damping: Damping factor (typically 0.85)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            PageRank scores for each vertex
        """
        # Initialize
        pr = np.ones(n) / n
        
        # Compute out-degrees
        out_degree = np.zeros(n)
        for u in range(n):
            out_degree[u] = len(adj_list.get(u, []))
        
        # Handle dangling nodes (no outgoing edges)
        dangling = out_degree == 0
        
        # Build incoming edges for efficiency
        incoming = defaultdict(list)
        for u in adj_list:
            for v in adj_list[u]:
                incoming[v].append(u)
        
        # Power iteration
        for iteration in range(max_iter):
            pr_new = np.ones(n) * (1 - damping) / n
            
            # Dangling node contribution
            dangling_sum = pr[dangling].sum()
            pr_new += damping * dangling_sum / n
            
            # Regular contribution
            for v in range(n):
                for u in incoming[v]:
                    pr_new[v] += damping * pr[u] / out_degree[u]
            
            # Check convergence
            diff = np.abs(pr_new - pr).sum()
            pr = pr_new
            
            if diff < tol:
                print(f"  Converged in {iteration + 1} iterations")
                break
        
        return pr
    
    def pagerank_matrix(adj_matrix: np.ndarray, 
                       damping: float = 0.85,
                       max_iter: int = 100) -> np.ndarray:
        """PageRank using matrix form (for comparison)."""
        n = len(adj_matrix)
        
        # Create transition matrix
        out_degree = adj_matrix.sum(axis=1)
        out_degree[out_degree == 0] = 1  # Avoid division by zero
        M = adj_matrix.T / out_degree
        
        # Handle dangling nodes
        dangling = adj_matrix.sum(axis=1) == 0
        M[:, dangling] = 1 / n
        
        # Power iteration
        pr = np.ones(n) / n
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * M @ pr
            if np.abs(pr_new - pr).sum() < 1e-6:
                break
            pr = pr_new
        
        return pr
    
    # Web graph example
    adj_list = {
        0: [1, 2],    # Page 0 links to 1, 2
        1: [2],       # Page 1 links to 2
        2: [0],       # Page 2 links to 0
        3: [2]        # Page 3 links to 2
    }
    n = 4
    
    print("Web graph (link structure):")
    for v in range(n):
        print(f"  Page {v} → {adj_list.get(v, [])}")
    
    pr = pagerank(adj_list, n)
    
    print(f"\nPageRank scores:")
    for v in range(n):
        print(f"  Page {v}: {pr[v]:.4f}")
    
    print(f"\nMost important page: {np.argmax(pr)}")
    
    return pagerank


# =============================================================================
# Example 11: Graph Coloring (Greedy)
# =============================================================================

def example_graph_coloring():
    """
    Greedy graph coloring.
    
    Assign colors so no adjacent vertices share a color.
    Applications: scheduling, register allocation.
    """
    print("\n" + "=" * 60)
    print("Example 11: Greedy Graph Coloring")
    print("=" * 60)
    
    def greedy_coloring(adj_list: Dict[int, List[int]], 
                        vertices: List[int]) -> Dict[int, int]:
        """
        Greedy graph coloring.
        
        Assigns smallest available color to each vertex.
        Uses at most Δ + 1 colors (Δ = max degree).
        """
        colors = {}
        
        for v in vertices:
            # Find colors used by neighbors
            neighbor_colors = set()
            for u in adj_list.get(v, []):
                if u in colors:
                    neighbor_colors.add(colors[u])
            
            # Find smallest available color
            color = 0
            while color in neighbor_colors:
                color += 1
            
            colors[v] = color
        
        return colors
    
    def is_valid_coloring(adj_list: Dict[int, List[int]], 
                          colors: Dict[int, int]) -> bool:
        """Check if coloring is valid."""
        for u in adj_list:
            for v in adj_list[u]:
                if colors.get(u) == colors.get(v):
                    return False
        return True
    
    # Create graph (Petersen-like structure)
    adj_list = {
        0: [1, 4, 5],
        1: [0, 2, 6],
        2: [1, 3, 7],
        3: [2, 4, 8],
        4: [3, 0, 9],
        5: [0, 7, 8],
        6: [1, 8, 9],
        7: [2, 5, 9],
        8: [3, 5, 6],
        9: [4, 6, 7]
    }
    vertices = list(range(10))
    
    print("Graph (vertices and neighbors):")
    max_degree = 0
    for v in vertices:
        degree = len(adj_list.get(v, []))
        max_degree = max(max_degree, degree)
        print(f"  {v}: degree={degree}, neighbors={adj_list.get(v, [])}")
    
    colors = greedy_coloring(adj_list, vertices)
    num_colors = len(set(colors.values()))
    
    print(f"\nColoring result:")
    for v in vertices:
        print(f"  Vertex {v}: Color {colors[v]}")
    
    print(f"\nColors used: {num_colors}")
    print(f"Max degree Δ: {max_degree}")
    print(f"Valid coloring: {is_valid_coloring(adj_list, colors)}")
    
    return greedy_coloring


# =============================================================================
# Example 12: Random Walk and DeepWalk Sampling
# =============================================================================

def example_random_walks():
    """
    Random walks on graphs.
    
    Foundation for graph embeddings (DeepWalk, Node2Vec).
    """
    print("\n" + "=" * 60)
    print("Example 12: Random Walks for Graph Embeddings")
    print("=" * 60)
    
    def random_walk(adj_list: Dict[int, List[int]], 
                    start: int,
                    walk_length: int) -> List[int]:
        """Simple random walk starting from a vertex."""
        walk = [start]
        current = start
        
        for _ in range(walk_length - 1):
            neighbors = adj_list.get(current, [])
            if not neighbors:
                break
            current = np.random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def random_walk_with_restart(adj_list: Dict[int, List[int]],
                                 start: int,
                                 walk_length: int,
                                 restart_prob: float = 0.15) -> List[int]:
        """Random walk with probability of teleporting back to start."""
        walk = [start]
        current = start
        
        for _ in range(walk_length - 1):
            if np.random.random() < restart_prob:
                current = start
            else:
                neighbors = adj_list.get(current, [])
                if neighbors:
                    current = np.random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def generate_walks(adj_list: Dict[int, List[int]],
                      num_walks: int,
                      walk_length: int) -> List[List[int]]:
        """Generate multiple random walks (DeepWalk style)."""
        vertices = list(adj_list.keys())
        walks = []
        
        for _ in range(num_walks):
            for v in vertices:
                walk = random_walk(adj_list, v, walk_length)
                walks.append(walk)
        
        return walks
    
    def node2vec_walk(adj_list: Dict[int, List[int]],
                      start: int,
                      walk_length: int,
                      p: float = 1.0,
                      q: float = 1.0) -> List[int]:
        """
        Node2Vec biased random walk.
        
        p: Return parameter (higher = less likely to return)
        q: In-out parameter (q > 1 favors local, q < 1 favors exploration)
        """
        walk = [start]
        
        # First step
        neighbors = adj_list.get(start, [])
        if not neighbors:
            return walk
        current = np.random.choice(neighbors)
        walk.append(current)
        
        # Subsequent steps with bias
        for _ in range(walk_length - 2):
            prev = walk[-2]
            curr = walk[-1]
            neighbors = adj_list.get(curr, [])
            
            if not neighbors:
                break
            
            # Compute transition probabilities
            prev_neighbors = set(adj_list.get(prev, []))
            probs = []
            
            for neighbor in neighbors:
                if neighbor == prev:
                    probs.append(1 / p)  # Return
                elif neighbor in prev_neighbors:
                    probs.append(1)      # Same distance from prev
                else:
                    probs.append(1 / q)  # Move away
            
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            next_node = np.random.choice(neighbors, p=probs)
            walk.append(next_node)
        
        return walk
    
    # Create social network-like graph
    np.random.seed(42)
    adj_list = {
        0: [1, 2, 3],
        1: [0, 2, 4],
        2: [0, 1, 3, 4],
        3: [0, 2, 5],
        4: [1, 2, 5, 6],
        5: [3, 4, 6],
        6: [4, 5]
    }
    
    print("Social network graph:")
    for v in sorted(adj_list.keys()):
        print(f"  {v}: {adj_list[v]}")
    
    # Generate walks
    print("\nRandom walks from vertex 0:")
    for i in range(3):
        walk = random_walk(adj_list, 0, 10)
        print(f"  Walk {i + 1}: {walk}")
    
    print("\nNode2Vec walks (p=0.5, q=2 - local exploration):")
    for i in range(3):
        walk = node2vec_walk(adj_list, 0, 10, p=0.5, q=2)
        print(f"  Walk {i + 1}: {walk}")
    
    print("\nNode2Vec walks (p=2, q=0.5 - global exploration):")
    for i in range(3):
        walk = node2vec_walk(adj_list, 0, 10, p=2, q=0.5)
        print(f"  Walk {i + 1}: {walk}")
    
    # Visit frequency (approximating stationary distribution)
    walks = generate_walks(adj_list, 10, 20)
    visit_count = defaultdict(int)
    for walk in walks:
        for v in walk:
            visit_count[v] += 1
    
    total = sum(visit_count.values())
    print(f"\nVisit frequencies (approx. stationary distribution):")
    for v in sorted(visit_count.keys()):
        print(f"  Vertex {v}: {visit_count[v] / total:.3f}")
    
    return random_walk, node2vec_walk


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all examples."""
    example_bfs()
    example_dfs()
    example_dijkstra()
    example_bellman_ford()
    example_floyd_warshall()
    example_kruskal()
    example_prim()
    example_topological_sort()
    example_scc()
    example_pagerank()
    example_graph_coloring()
    example_random_walks()
    
    print("\n" + "=" * 60)
    print("All Graph Algorithm Examples Completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
