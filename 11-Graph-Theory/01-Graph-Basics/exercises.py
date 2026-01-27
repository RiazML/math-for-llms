"""
Graph Theory Basics - Exercises
===============================
Practice problems for graph fundamentals.
"""

import numpy as np
from collections import defaultdict, deque


class Exercise1:
    """
    Exercise 1: Graph Implementation
    ================================
    
    Implement a complete Graph class with:
    1. add_vertex, add_edge, remove_vertex, remove_edge
    2. neighbors, degree, has_edge
    3. Support for both directed and undirected graphs
    """
    
    class Graph:
        def __init__(self, directed=False):
            self.directed = directed
            self.adj = defaultdict(set)
        
        def add_vertex(self, v):
            """Add a vertex to the graph."""
            # YOUR CODE HERE
            if v not in self.adj:
                self.adj[v] = set()
        
        def add_edge(self, u, v):
            """Add an edge (and vertices if needed)."""
            # YOUR CODE HERE
            self.add_vertex(u)
            self.add_vertex(v)
            self.adj[u].add(v)
            if not self.directed:
                self.adj[v].add(u)
        
        def remove_vertex(self, v):
            """Remove vertex and all incident edges."""
            # YOUR CODE HERE
            if v in self.adj:
                # Remove v from all adjacency lists
                for u in self.adj:
                    self.adj[u].discard(v)
                # Remove v's adjacency list
                del self.adj[v]
        
        def remove_edge(self, u, v):
            """Remove edge between u and v."""
            # YOUR CODE HERE
            self.adj[u].discard(v)
            if not self.directed:
                self.adj[v].discard(u)
        
        def has_edge(self, u, v):
            """Check if edge exists."""
            return v in self.adj.get(u, set())
        
        def neighbors(self, v):
            """Return neighbors of vertex v."""
            return self.adj.get(v, set())
        
        def degree(self, v):
            """Return degree of vertex v."""
            return len(self.adj.get(v, set()))
        
        def vertices(self):
            """Return all vertices."""
            return set(self.adj.keys())
        
        def edges(self):
            """Return all edges."""
            seen = set()
            result = []
            for u in self.adj:
                for v in self.adj[u]:
                    if self.directed or (v, u) not in seen:
                        result.append((u, v))
                        seen.add((u, v))
            return result
    
    @staticmethod
    def verify():
        """Test the Graph implementation."""
        print("Exercise 1: Graph Implementation")
        print("-" * 40)
        
        g = Exercise1.Graph(directed=False)
        
        # Add edges
        for u, v in [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]:
            g.add_edge(u, v)
        
        print(f"Vertices: {g.vertices()}")
        print(f"Edges: {g.edges()}")
        print(f"Degree of 1: {g.degree(1)}")
        print(f"Has edge (1, 2): {g.has_edge(1, 2)}")
        print(f"Has edge (1, 4): {g.has_edge(1, 4)}")
        print(f"Has edge (2, 4): {g.has_edge(2, 4)}")
        
        # Remove edge
        g.remove_edge(1, 3)
        print(f"\nAfter removing (1, 3):")
        print(f"Has edge (1, 3): {g.has_edge(1, 3)}")
        
        # Remove vertex
        g.remove_vertex(4)
        print(f"\nAfter removing vertex 4:")
        print(f"Vertices: {g.vertices()}")
        print(f"Edges: {g.edges()}")


class Exercise2:
    """
    Exercise 2: Adjacency Matrix Operations
    =======================================
    
    Implement functions for adjacency matrix operations:
    1. Convert between adjacency list and matrix
    2. Compute number of paths of length k
    3. Check if graph is connected using matrix powers
    """
    
    @staticmethod
    def list_to_matrix(adj_list, vertices):
        """Convert adjacency list to adjacency matrix."""
        # YOUR CODE HERE
        n = len(vertices)
        vertex_idx = {v: i for i, v in enumerate(vertices)}
        
        A = np.zeros((n, n), dtype=int)
        
        for u in adj_list:
            for v in adj_list[u]:
                i, j = vertex_idx[u], vertex_idx[v]
                A[i, j] = 1
        
        return A
    
    @staticmethod
    def matrix_to_list(A, vertices):
        """Convert adjacency matrix to adjacency list."""
        # YOUR CODE HERE
        n = len(vertices)
        adj_list = defaultdict(list)
        
        for i in range(n):
            for j in range(n):
                if A[i, j] == 1:
                    adj_list[vertices[i]].append(vertices[j])
        
        return dict(adj_list)
    
    @staticmethod
    def count_paths(A, k):
        """Count paths of length k between all pairs (A^k)."""
        # YOUR CODE HERE
        result = np.eye(len(A), dtype=int)
        for _ in range(k):
            result = result @ A
        return result
    
    @staticmethod
    def is_connected_matrix(A):
        """Check if graph is connected using matrix powers."""
        # YOUR CODE HERE
        n = len(A)
        # Compute (I + A)^(n-1) - if all entries positive, connected
        power_sum = np.eye(n, dtype=int)
        current = np.eye(n, dtype=int)
        
        for _ in range(n - 1):
            current = current @ A
            power_sum += current
        
        # Check if all pairs are reachable
        return np.all(power_sum > 0)
    
    @staticmethod
    def verify():
        """Test the adjacency matrix operations."""
        print("\nExercise 2: Adjacency Matrix Operations")
        print("-" * 40)
        
        adj_list = {
            'A': ['B', 'C'],
            'B': ['A', 'C', 'D'],
            'C': ['A', 'B'],
            'D': ['B']
        }
        vertices = ['A', 'B', 'C', 'D']
        
        A = Exercise2.list_to_matrix(adj_list, vertices)
        print("Adjacency matrix:")
        print(A)
        
        # Paths of length 2
        A2 = Exercise2.count_paths(A, 2)
        print(f"\nPaths of length 2 (A²):")
        print(A2)
        print(f"Paths from A to D of length 2: {A2[0, 3]}")
        
        # Check connectivity
        connected = Exercise2.is_connected_matrix(A)
        print(f"\nGraph is connected: {connected}")
        
        # Convert back
        adj_back = Exercise2.matrix_to_list(A, vertices)
        print(f"\nConverted back to list: {adj_back}")


class Exercise3:
    """
    Exercise 3: Graph Traversal Variants
    ====================================
    
    Implement:
    1. BFS with parent tracking
    2. DFS with discovery/finish times
    3. Iterative DFS (non-recursive)
    """
    
    @staticmethod
    def bfs_with_parents(graph, start):
        """
        BFS returning distances and parent pointers.
        
        Returns: (distances, parents) where parents[v] = predecessor on shortest path
        """
        # YOUR CODE HERE
        distances = {start: 0}
        parents = {start: None}
        queue = deque([start])
        
        while queue:
            v = queue.popleft()
            for neighbor in graph.get(v, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[v] + 1
                    parents[neighbor] = v
                    queue.append(neighbor)
        
        return distances, parents
    
    @staticmethod
    def dfs_with_times(graph, start):
        """
        DFS with discovery and finish times.
        
        Returns: (discovery, finish, parent) dictionaries
        """
        # YOUR CODE HERE
        discovery = {}
        finish = {}
        parent = {start: None}
        time = [0]
        
        def dfs_visit(v):
            time[0] += 1
            discovery[v] = time[0]
            
            for neighbor in graph.get(v, []):
                if neighbor not in discovery:
                    parent[neighbor] = v
                    dfs_visit(neighbor)
            
            time[0] += 1
            finish[v] = time[0]
        
        dfs_visit(start)
        return discovery, finish, parent
    
    @staticmethod
    def iterative_dfs(graph, start):
        """
        Iterative DFS using explicit stack.
        
        Returns list of vertices in DFS order.
        """
        # YOUR CODE HERE
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                result.append(v)
                
                # Add neighbors in reverse order for consistent ordering
                for neighbor in reversed(list(graph.get(v, []))):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    @staticmethod
    def verify():
        """Test traversal implementations."""
        print("\nExercise 3: Graph Traversal Variants")
        print("-" * 40)
        
        graph = {
            'A': ['B', 'C'],
            'B': ['A', 'D', 'E'],
            'C': ['A', 'F'],
            'D': ['B'],
            'E': ['B', 'F'],
            'F': ['C', 'E']
        }
        
        # BFS
        distances, parents = Exercise3.bfs_with_parents(graph, 'A')
        print(f"BFS from A:")
        print(f"  Distances: {distances}")
        print(f"  Parents: {parents}")
        
        # Reconstruct path A -> F
        path = []
        v = 'F'
        while v is not None:
            path.append(v)
            v = parents.get(v)
        print(f"  Path A → F: {path[::-1]}")
        
        # DFS with times
        discovery, finish, parent = Exercise3.dfs_with_times(graph, 'A')
        print(f"\nDFS from A:")
        print(f"  Discovery times: {discovery}")
        print(f"  Finish times: {finish}")
        
        # Iterative DFS
        dfs_order = Exercise3.iterative_dfs(graph, 'A')
        print(f"\nIterative DFS order: {dfs_order}")


class Exercise4:
    """
    Exercise 4: Cycle Detection
    ===========================
    
    Implement cycle detection for:
    1. Undirected graphs
    2. Directed graphs
    """
    
    @staticmethod
    def has_cycle_undirected(graph):
        """
        Detect cycle in undirected graph using DFS.
        
        Returns (has_cycle, cycle_vertices) if cycle found.
        """
        # YOUR CODE HERE
        visited = set()
        parent = {}
        
        def dfs(v, p):
            visited.add(v)
            parent[v] = p
            
            for neighbor in graph.get(v, []):
                if neighbor not in visited:
                    result = dfs(neighbor, v)
                    if result:
                        return result
                elif neighbor != p:
                    # Found cycle - reconstruct it
                    cycle = [neighbor]
                    current = v
                    while current != neighbor:
                        cycle.append(current)
                        current = parent[current]
                    cycle.append(neighbor)
                    return (True, cycle)
            
            return None
        
        for start in graph:
            if start not in visited:
                result = dfs(start, None)
                if result:
                    return result
        
        return (False, [])
    
    @staticmethod
    def has_cycle_directed(graph):
        """
        Detect cycle in directed graph using DFS colors.
        
        White = unvisited, Gray = in progress, Black = finished
        """
        # YOUR CODE HERE
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {v: WHITE for v in graph}
        
        def dfs(v):
            color[v] = GRAY
            
            for neighbor in graph.get(v, []):
                if neighbor not in color:
                    color[neighbor] = WHITE
                
                if color[neighbor] == GRAY:
                    return True  # Back edge = cycle
                elif color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            
            color[v] = BLACK
            return False
        
        for v in graph:
            if color.get(v, WHITE) == WHITE:
                if dfs(v):
                    return True
        
        return False
    
    @staticmethod
    def verify():
        """Test cycle detection."""
        print("\nExercise 4: Cycle Detection")
        print("-" * 40)
        
        # Undirected with cycle
        graph_cycle = {
            'A': ['B', 'D'],
            'B': ['A', 'C'],
            'C': ['B', 'D'],
            'D': ['A', 'C']
        }
        
        has_cycle, cycle = Exercise4.has_cycle_undirected(graph_cycle)
        print(f"Undirected graph (A-B-C-D-A):")
        print(f"  Has cycle: {has_cycle}")
        if cycle:
            print(f"  Cycle: {cycle}")
        
        # Tree (no cycle)
        tree = {
            'A': ['B', 'C'],
            'B': ['A', 'D'],
            'C': ['A'],
            'D': ['B']
        }
        
        has_cycle, _ = Exercise4.has_cycle_undirected(tree)
        print(f"\nTree graph:")
        print(f"  Has cycle: {has_cycle}")
        
        # Directed with cycle
        digraph_cycle = {
            'A': ['B'],
            'B': ['C'],
            'C': ['A']  # Back edge
        }
        
        has_cycle = Exercise4.has_cycle_directed(digraph_cycle)
        print(f"\nDirected graph (A→B→C→A):")
        print(f"  Has cycle: {has_cycle}")
        
        # DAG (no cycle)
        dag = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D'],
            'D': []
        }
        
        has_cycle = Exercise4.has_cycle_directed(dag)
        print(f"\nDAG:")
        print(f"  Has cycle: {has_cycle}")


class Exercise5:
    """
    Exercise 5: Shortest Paths
    ==========================
    
    Implement:
    1. BFS for unweighted shortest paths
    2. Dijkstra for weighted shortest paths
    3. Path reconstruction
    """
    
    @staticmethod
    def bfs_shortest_path(graph, start, end):
        """
        Find shortest path in unweighted graph.
        
        Returns (distance, path) or (inf, []) if no path.
        """
        # YOUR CODE HERE
        if start == end:
            return 0, [start]
        
        distances = {start: 0}
        parents = {start: None}
        queue = deque([start])
        
        while queue:
            v = queue.popleft()
            
            for neighbor in graph.get(v, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[v] + 1
                    parents[neighbor] = v
                    queue.append(neighbor)
                    
                    if neighbor == end:
                        # Reconstruct path
                        path = []
                        current = end
                        while current is not None:
                            path.append(current)
                            current = parents[current]
                        return distances[end], path[::-1]
        
        return float('inf'), []
    
    @staticmethod
    def dijkstra(graph, start, end):
        """
        Dijkstra's algorithm for weighted shortest path.
        
        graph: {vertex: [(neighbor, weight), ...]}
        Returns (distance, path).
        """
        # YOUR CODE HERE
        import heapq
        
        distances = {v: float('inf') for v in graph}
        distances[start] = 0
        parents = {start: None}
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == end:
                break
            
            for v, weight in graph.get(u, []):
                if v not in visited:
                    new_dist = distances[u] + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        parents[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        if distances[end] == float('inf'):
            return float('inf'), []
        
        # Reconstruct path
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = parents.get(current)
        
        return distances[end], path[::-1]
    
    @staticmethod
    def verify():
        """Test shortest path algorithms."""
        print("\nExercise 5: Shortest Paths")
        print("-" * 40)
        
        # Unweighted graph
        unweighted = {
            'A': ['B', 'C'],
            'B': ['A', 'D', 'E'],
            'C': ['A', 'F'],
            'D': ['B'],
            'E': ['B', 'F'],
            'F': ['C', 'E']
        }
        
        dist, path = Exercise5.bfs_shortest_path(unweighted, 'A', 'F')
        print(f"Unweighted: A → F")
        print(f"  Distance: {dist}")
        print(f"  Path: {' → '.join(path)}")
        
        # Weighted graph
        weighted = {
            'A': [('B', 4), ('C', 2)],
            'B': [('A', 4), ('C', 1), ('D', 5)],
            'C': [('A', 2), ('B', 1), ('D', 8)],
            'D': [('B', 5), ('C', 8)]
        }
        
        dist, path = Exercise5.dijkstra(weighted, 'A', 'D')
        print(f"\nWeighted: A → D")
        print(f"  Distance: {dist}")
        print(f"  Path: {' → '.join(path)}")


class Exercise6:
    """
    Exercise 6: Graph Properties
    ============================
    
    Compute various graph properties:
    1. Diameter and eccentricity
    2. Clustering coefficient
    3. Degree distribution
    """
    
    @staticmethod
    def eccentricity(graph, v):
        """Compute eccentricity of vertex v."""
        # YOUR CODE HERE
        distances = {v: 0}
        queue = deque([v])
        
        while queue:
            u = queue.popleft()
            for neighbor in graph.get(u, []):
                if neighbor not in distances:
                    distances[neighbor] = distances[u] + 1
                    queue.append(neighbor)
        
        if len(distances) < len(graph):
            return float('inf')  # Not connected
        
        return max(distances.values())
    
    @staticmethod
    def diameter(graph):
        """Compute diameter of graph."""
        # YOUR CODE HERE
        max_ecc = 0
        for v in graph:
            ecc = Exercise6.eccentricity(graph, v)
            max_ecc = max(max_ecc, ecc)
        return max_ecc
    
    @staticmethod
    def local_clustering_coefficient(graph, v):
        """Compute local clustering coefficient of vertex v."""
        # YOUR CODE HERE
        neighbors = list(graph.get(v, []))
        k = len(neighbors)
        
        if k < 2:
            return 0.0
        
        edges = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in graph.get(neighbors[i], []):
                    edges += 1
        
        return 2 * edges / (k * (k - 1))
    
    @staticmethod
    def degree_distribution(graph):
        """Compute degree distribution."""
        # YOUR CODE HERE
        degrees = [len(neighbors) for neighbors in graph.values()]
        
        distribution = defaultdict(int)
        for d in degrees:
            distribution[d] += 1
        
        # Normalize
        n = len(degrees)
        return {k: v / n for k, v in sorted(distribution.items())}
    
    @staticmethod
    def verify():
        """Test graph properties."""
        print("\nExercise 6: Graph Properties")
        print("-" * 40)
        
        graph = {
            'A': ['B', 'C', 'D'],
            'B': ['A', 'C'],
            'C': ['A', 'B', 'D'],
            'D': ['A', 'C', 'E'],
            'E': ['D']
        }
        
        print(f"Graph eccentricities:")
        for v in sorted(graph.keys()):
            ecc = Exercise6.eccentricity(graph, v)
            print(f"  ε({v}) = {ecc}")
        
        diam = Exercise6.diameter(graph)
        print(f"\nDiameter: {diam}")
        
        print(f"\nClustering coefficients:")
        for v in sorted(graph.keys()):
            cc = Exercise6.local_clustering_coefficient(graph, v)
            print(f"  C({v}) = {cc:.3f}")
        
        dist = Exercise6.degree_distribution(graph)
        print(f"\nDegree distribution: {dict(dist)}")


class Exercise7:
    """
    Exercise 7: Centrality Measures
    ===============================
    
    Implement centrality measures:
    1. Degree centrality
    2. Closeness centrality
    3. PageRank (simplified)
    """
    
    @staticmethod
    def degree_centrality(graph):
        """Compute normalized degree centrality."""
        # YOUR CODE HERE
        n = len(graph)
        return {v: len(neighbors) / (n - 1) for v, neighbors in graph.items()}
    
    @staticmethod
    def closeness_centrality(graph):
        """Compute closeness centrality."""
        # YOUR CODE HERE
        n = len(graph)
        centrality = {}
        
        for start in graph:
            distances = {start: 0}
            queue = deque([start])
            
            while queue:
                v = queue.popleft()
                for neighbor in graph.get(v, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[v] + 1
                        queue.append(neighbor)
            
            total_dist = sum(distances.values())
            centrality[start] = (len(distances) - 1) / total_dist if total_dist > 0 else 0
        
        return centrality
    
    @staticmethod
    def pagerank(graph, damping=0.85, iterations=100):
        """
        Compute PageRank scores.
        
        PR(v) = (1-d)/n + d * Σ PR(u)/deg(u) for all u→v
        """
        # YOUR CODE HERE
        n = len(graph)
        pr = {v: 1/n for v in graph}
        
        # Create incoming edges
        incoming = defaultdict(list)
        for u in graph:
            for v in graph[u]:
                incoming[v].append(u)
        
        for _ in range(iterations):
            new_pr = {}
            for v in graph:
                rank_sum = sum(pr[u] / len(graph[u]) for u in incoming.get(v, []) if len(graph[u]) > 0)
                new_pr[v] = (1 - damping) / n + damping * rank_sum
            pr = new_pr
        
        return pr
    
    @staticmethod
    def verify():
        """Test centrality measures."""
        print("\nExercise 7: Centrality Measures")
        print("-" * 40)
        
        # Web-like directed graph
        graph = {
            'A': ['B', 'C'],
            'B': ['C'],
            'C': ['A'],
            'D': ['C']
        }
        
        print("Directed graph for PageRank:")
        for v, neighbors in sorted(graph.items()):
            print(f"  {v} → {neighbors}")
        
        pr = Exercise7.pagerank(graph)
        print(f"\nPageRank scores:")
        for v in sorted(pr, key=pr.get, reverse=True):
            print(f"  {v}: {pr[v]:.4f}")
        
        # Undirected for other centralities
        undirected = {
            'A': ['B', 'C'],
            'B': ['A', 'C', 'D'],
            'C': ['A', 'B', 'D'],
            'D': ['B', 'C']
        }
        
        dc = Exercise7.degree_centrality(undirected)
        cc = Exercise7.closeness_centrality(undirected)
        
        print(f"\nUndirected graph centralities:")
        print(f"{'Vertex':<10} {'Degree':<12} {'Closeness'}")
        print("-" * 35)
        for v in sorted(undirected.keys()):
            print(f"{v:<10} {dc[v]:<12.3f} {cc[v]:.3f}")


class Exercise8:
    """
    Exercise 8: Bipartite Graph Operations
    ======================================
    
    Implement:
    1. Check if graph is bipartite
    2. Find maximum matching (greedy)
    3. Convert to bipartite adjacency matrix
    """
    
    @staticmethod
    def is_bipartite(graph):
        """
        Check if graph is bipartite.
        
        Returns (is_bipartite, partition) where partition is (set1, set2).
        """
        # YOUR CODE HERE
        color = {}
        
        for start in graph:
            if start in color:
                continue
            
            queue = deque([start])
            color[start] = 0
            
            while queue:
                v = queue.popleft()
                for neighbor in graph.get(v, []):
                    if neighbor not in color:
                        color[neighbor] = 1 - color[v]
                        queue.append(neighbor)
                    elif color[neighbor] == color[v]:
                        return False, (set(), set())
        
        set0 = {v for v, c in color.items() if c == 0}
        set1 = {v for v, c in color.items() if c == 1}
        return True, (set0, set1)
    
    @staticmethod
    def greedy_matching(graph, left_vertices, right_vertices):
        """
        Find a maximal matching using greedy algorithm.
        
        Returns set of matched edges.
        """
        # YOUR CODE HERE
        matching = set()
        matched_left = set()
        matched_right = set()
        
        for u in left_vertices:
            for v in graph.get(u, []):
                if v in right_vertices and v not in matched_right:
                    matching.add((u, v))
                    matched_left.add(u)
                    matched_right.add(v)
                    break
        
        return matching
    
    @staticmethod
    def verify():
        """Test bipartite operations."""
        print("\nExercise 8: Bipartite Graphs")
        print("-" * 40)
        
        # Bipartite graph
        bipartite = {
            'U1': ['I1', 'I2'],
            'U2': ['I2', 'I3'],
            'U3': ['I1'],
            'I1': ['U1', 'U3'],
            'I2': ['U1', 'U2'],
            'I3': ['U2']
        }
        
        is_bip, partition = Exercise8.is_bipartite(bipartite)
        print(f"User-Item graph is bipartite: {is_bip}")
        if is_bip:
            print(f"  Set 1: {partition[0]}")
            print(f"  Set 2: {partition[1]}")
        
        # Find matching
        left = {'U1', 'U2', 'U3'}
        right = {'I1', 'I2', 'I3'}
        matching = Exercise8.greedy_matching(bipartite, left, right)
        print(f"\nGreedy matching: {matching}")
        
        # Non-bipartite (triangle)
        triangle = {
            'A': ['B', 'C'],
            'B': ['A', 'C'],
            'C': ['A', 'B']
        }
        
        is_bip, _ = Exercise8.is_bipartite(triangle)
        print(f"\nTriangle is bipartite: {is_bip}")


class Exercise9:
    """
    Exercise 9: Topological Sort and DAG
    ====================================
    
    Implement:
    1. Check if directed graph is a DAG
    2. Topological sort (Kahn's algorithm)
    3. Longest path in DAG
    """
    
    @staticmethod
    def is_dag(graph):
        """Check if directed graph is acyclic."""
        # YOUR CODE HERE
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {v: WHITE for v in graph}
        
        def dfs(v):
            color[v] = GRAY
            for neighbor in graph.get(v, []):
                if neighbor not in color:
                    color[neighbor] = WHITE
                if color.get(neighbor, WHITE) == GRAY:
                    return False
                if color.get(neighbor, WHITE) == WHITE:
                    if not dfs(neighbor):
                        return False
            color[v] = BLACK
            return True
        
        for v in graph:
            if color.get(v, WHITE) == WHITE:
                if not dfs(v):
                    return False
        return True
    
    @staticmethod
    def topological_sort(graph):
        """
        Topological sort using Kahn's algorithm.
        
        Returns sorted list or None if cycle exists.
        """
        # YOUR CODE HERE
        in_degree = {v: 0 for v in graph}
        for v in graph:
            for neighbor in graph.get(v, []):
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        queue = deque([v for v in in_degree if in_degree[v] == 0])
        result = []
        
        while queue:
            v = queue.popleft()
            result.append(v)
            
            for neighbor in graph.get(v, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(in_degree) else None
    
    @staticmethod
    def longest_path_dag(graph, start):
        """
        Find longest path from start in DAG.
        
        Returns (length, path).
        """
        # YOUR CODE HERE
        # Topological order first
        order = Exercise9.topological_sort(graph)
        if order is None:
            return -1, []
        
        dist = {v: float('-inf') for v in graph}
        dist[start] = 0
        parent = {start: None}
        
        for v in order:
            if dist[v] != float('-inf'):
                for neighbor in graph.get(v, []):
                    if dist[v] + 1 > dist[neighbor]:
                        dist[neighbor] = dist[v] + 1
                        parent[neighbor] = v
        
        # Find vertex with maximum distance
        max_v = max(dist, key=dist.get)
        max_dist = dist[max_v]
        
        if max_dist == float('-inf'):
            return 0, [start]
        
        # Reconstruct path
        path = []
        current = max_v
        while current is not None:
            path.append(current)
            current = parent.get(current)
        
        return max_dist, path[::-1]
    
    @staticmethod
    def verify():
        """Test DAG operations."""
        print("\nExercise 9: DAG Operations")
        print("-" * 40)
        
        dag = {
            'A': ['B', 'C'],
            'B': ['D'],
            'C': ['D', 'E'],
            'D': ['F'],
            'E': ['F'],
            'F': []
        }
        
        print(f"Is DAG: {Exercise9.is_dag(dag)}")
        
        topo = Exercise9.topological_sort(dag)
        print(f"Topological order: {topo}")
        
        length, path = Exercise9.longest_path_dag(dag, 'A')
        print(f"Longest path from A: length={length}, path={path}")


class Exercise10:
    """
    Exercise 10: Graph Algorithms for ML
    ====================================
    
    Implement algorithms used in ML:
    1. k-hop neighbors (for GNN)
    2. Random walk (for node2vec-like)
    3. Subgraph extraction
    """
    
    @staticmethod
    def k_hop_neighbors(graph, v, k):
        """
        Find all vertices within k hops of v.
        
        Returns dict {vertex: distance} for vertices within k hops.
        """
        # YOUR CODE HERE
        distances = {v: 0}
        queue = deque([v])
        
        while queue:
            u = queue.popleft()
            if distances[u] < k:
                for neighbor in graph.get(u, []):
                    if neighbor not in distances:
                        distances[neighbor] = distances[u] + 1
                        queue.append(neighbor)
        
        return distances
    
    @staticmethod
    def random_walk(graph, start, length, seed=None):
        """
        Perform random walk starting from start.
        
        Returns list of visited vertices.
        """
        # YOUR CODE HERE
        if seed is not None:
            np.random.seed(seed)
        
        walk = [start]
        current = start
        
        for _ in range(length):
            neighbors = list(graph.get(current, []))
            if not neighbors:
                break
            current = neighbors[np.random.randint(len(neighbors))]
            walk.append(current)
        
        return walk
    
    @staticmethod
    def extract_subgraph(graph, vertices):
        """
        Extract induced subgraph containing given vertices.
        
        Returns new graph dict.
        """
        # YOUR CODE HERE
        vertex_set = set(vertices)
        subgraph = {}
        
        for v in vertex_set:
            if v in graph:
                neighbors = [n for n in graph[v] if n in vertex_set]
                subgraph[v] = neighbors
        
        return subgraph
    
    @staticmethod
    def verify():
        """Test ML-related graph algorithms."""
        print("\nExercise 10: Graph Algorithms for ML")
        print("-" * 40)
        
        graph = {
            'A': ['B', 'C'],
            'B': ['A', 'D', 'E'],
            'C': ['A', 'F'],
            'D': ['B', 'G'],
            'E': ['B'],
            'F': ['C'],
            'G': ['D']
        }
        
        # k-hop neighbors
        neighbors_2 = Exercise10.k_hop_neighbors(graph, 'A', 2)
        print(f"2-hop neighbors of A: {neighbors_2}")
        
        # Random walk
        walk = Exercise10.random_walk(graph, 'A', 5, seed=42)
        print(f"Random walk from A (length 5): {walk}")
        
        # Subgraph extraction
        subgraph = Exercise10.extract_subgraph(graph, ['A', 'B', 'C', 'D'])
        print(f"Subgraph on {{'A','B','C','D'}}: {subgraph}")


def run_all_exercises():
    """Run all exercises."""
    Exercise1.verify()
    Exercise2.verify()
    Exercise3.verify()
    Exercise4.verify()
    Exercise5.verify()
    Exercise6.verify()
    Exercise7.verify()
    Exercise8.verify()
    Exercise9.verify()
    Exercise10.verify()


if __name__ == "__main__":
    run_all_exercises()
