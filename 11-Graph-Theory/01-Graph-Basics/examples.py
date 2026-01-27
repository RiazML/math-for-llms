"""
Graph Theory Basics - Examples
==============================
Practical implementations of graph concepts using NetworkX.
"""

import numpy as np
from collections import defaultdict, deque


def example_1_graph_creation():
    """
    Example 1: Creating Graphs
    ==========================
    Basic graph construction without external libraries.
    """
    print("=" * 60)
    print("Example 1: Graph Creation")
    print("=" * 60)
    
    # Adjacency list representation
    class Graph:
        def __init__(self, directed=False):
            self.adj = defaultdict(set)
            self.directed = directed
        
        def add_edge(self, u, v, weight=None):
            self.adj[u].add(v)
            if not self.directed:
                self.adj[v].add(u)
        
        def neighbors(self, v):
            return self.adj[v]
        
        def vertices(self):
            return set(self.adj.keys())
        
        def edges(self):
            seen = set()
            edges = []
            for u in self.adj:
                for v in self.adj[u]:
                    if self.directed or (v, u) not in seen:
                        edges.append((u, v))
                        seen.add((u, v))
            return edges
        
        def degree(self, v):
            return len(self.adj[v])
    
    # Create undirected graph
    g = Graph(directed=False)
    edges = [(1, 2), (1, 4), (2, 3), (3, 4), (2, 4)]
    for u, v in edges:
        g.add_edge(u, v)
    
    print("Undirected Graph:")
    print(f"  Vertices: {g.vertices()}")
    print(f"  Edges: {g.edges()}")
    
    print(f"\nDegrees:")
    for v in sorted(g.vertices()):
        print(f"  deg({v}) = {g.degree(v)}")
    
    # Verify handshaking lemma
    total_degree = sum(g.degree(v) for v in g.vertices())
    print(f"\nSum of degrees: {total_degree}")
    print(f"2 × |E|: {2 * len(g.edges())}")
    print(f"Handshaking lemma verified: {total_degree == 2 * len(g.edges())}")


def example_2_directed_graphs():
    """
    Example 2: Directed Graphs
    ==========================
    In-degree and out-degree computation.
    """
    print("\n" + "=" * 60)
    print("Example 2: Directed Graphs")
    print("=" * 60)
    
    class DiGraph:
        def __init__(self):
            self.adj_out = defaultdict(set)  # u -> v
            self.adj_in = defaultdict(set)   # v <- u
        
        def add_edge(self, u, v):
            self.adj_out[u].add(v)
            self.adj_in[v].add(u)
            # Ensure all nodes exist
            if u not in self.adj_in:
                self.adj_in[u] = set()
            if v not in self.adj_out:
                self.adj_out[v] = set()
        
        def out_degree(self, v):
            return len(self.adj_out[v])
        
        def in_degree(self, v):
            return len(self.adj_in[v])
        
        def vertices(self):
            return set(self.adj_out.keys()) | set(self.adj_in.keys())
    
    # Create directed graph
    g = DiGraph()
    edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'B')]
    for u, v in edges:
        g.add_edge(u, v)
    
    print("Directed Graph: A→B, A→C, B→C, C→D, D→B")
    print(f"\n{'Vertex':<10} {'In-degree':<12} {'Out-degree'}")
    print("-" * 35)
    
    for v in sorted(g.vertices()):
        print(f"{v:<10} {g.in_degree(v):<12} {g.out_degree(v)}")
    
    # Find sources and sinks
    sources = [v for v in g.vertices() if g.in_degree(v) == 0]
    sinks = [v for v in g.vertices() if g.out_degree(v) == 0]
    
    print(f"\nSources (in-degree 0): {sources}")
    print(f"Sinks (out-degree 0): {sinks}")


def example_3_adjacency_matrix():
    """
    Example 3: Adjacency Matrix Representation
    ==========================================
    Matrix form useful for linear algebra operations.
    """
    print("\n" + "=" * 60)
    print("Example 3: Adjacency Matrix")
    print("=" * 60)
    
    # Graph as adjacency matrix
    # Vertices: 0, 1, 2, 3
    # Edges: 0-1, 0-2, 1-2, 2-3
    
    n = 4
    A = np.zeros((n, n), dtype=int)
    
    edges = [(0, 1), (0, 2), (1, 2), (2, 3)]
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # Undirected
    
    print("Adjacency Matrix:")
    print(A)
    
    print(f"\nDegree from row sum: {A.sum(axis=1)}")
    print(f"Number of edges: {A.sum() // 2}")
    
    # A² gives number of paths of length 2
    A2 = A @ A
    print(f"\nA² (paths of length 2):")
    print(A2)
    print(f"\nPaths of length 2 from vertex 0 to vertex 3: {A2[0, 3]}")
    
    # Trace of A³ / 6 = number of triangles
    A3 = A @ A @ A
    num_triangles = np.trace(A3) // 6
    print(f"Number of triangles: {num_triangles}")


def example_4_bfs_and_dfs():
    """
    Example 4: Graph Traversal
    ==========================
    BFS and DFS implementations.
    """
    print("\n" + "=" * 60)
    print("Example 4: BFS and DFS Traversal")
    print("=" * 60)
    
    # Graph as adjacency list
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    def bfs(graph, start):
        """Breadth-first search."""
        visited = set()
        queue = deque([start])
        visited.add(start)
        order = []
        distances = {start: 0}
        
        while queue:
            v = queue.popleft()
            order.append(v)
            
            for neighbor in graph[v]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    distances[neighbor] = distances[v] + 1
        
        return order, distances
    
    def dfs(graph, start):
        """Depth-first search (iterative)."""
        visited = set()
        stack = [start]
        order = []
        
        while stack:
            v = stack.pop()
            if v not in visited:
                visited.add(v)
                order.append(v)
                
                # Add neighbors in reverse for consistent ordering
                for neighbor in reversed(graph[v]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return order
    
    print("Graph: A-B, A-C, B-D, B-E, C-F, E-F")
    
    bfs_order, distances = bfs(graph, 'A')
    dfs_order = dfs(graph, 'A')
    
    print(f"\nBFS from A: {bfs_order}")
    print(f"Distances from A: {distances}")
    
    print(f"\nDFS from A: {dfs_order}")


def example_5_connected_components():
    """
    Example 5: Finding Connected Components
    =======================================
    Partition graph into maximal connected subgraphs.
    """
    print("\n" + "=" * 60)
    print("Example 5: Connected Components")
    print("=" * 60)
    
    def find_components(graph):
        """Find all connected components using BFS."""
        visited = set()
        components = []
        
        for start in graph:
            if start not in visited:
                # BFS from this vertex
                component = []
                queue = deque([start])
                visited.add(start)
                
                while queue:
                    v = queue.popleft()
                    component.append(v)
                    
                    for neighbor in graph[v]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    # Graph with multiple components
    graph = {
        'A': ['B'],
        'B': ['A', 'C'],
        'C': ['B'],
        'D': ['E'],
        'E': ['D'],
        'F': []  # Isolated vertex
    }
    
    components = find_components(graph)
    
    print(f"Graph vertices: {list(graph.keys())}")
    print(f"Number of components: {len(components)}")
    
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {comp}")


def example_6_shortest_paths():
    """
    Example 6: Shortest Path Algorithms
    ===================================
    Dijkstra and BFS for unweighted/weighted graphs.
    """
    print("\n" + "=" * 60)
    print("Example 6: Shortest Paths")
    print("=" * 60)
    
    import heapq
    
    def dijkstra(graph, start):
        """
        Dijkstra's algorithm for weighted graphs.
        graph: dict of {vertex: [(neighbor, weight), ...]}
        """
        distances = {v: float('inf') for v in graph}
        distances[start] = 0
        predecessors = {start: None}
        
        pq = [(0, start)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            for v, weight in graph[u]:
                if v not in visited:
                    new_dist = distances[u] + weight
                    if new_dist < distances[v]:
                        distances[v] = new_dist
                        predecessors[v] = u
                        heapq.heappush(pq, (new_dist, v))
        
        return distances, predecessors
    
    def reconstruct_path(predecessors, target):
        """Reconstruct path from start to target."""
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        return path[::-1]
    
    # Weighted graph
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('A', 4), ('C', 1), ('D', 5)],
        'C': [('A', 2), ('B', 1), ('D', 8), ('E', 10)],
        'D': [('B', 5), ('C', 8), ('E', 2)],
        'E': [('C', 10), ('D', 2)]
    }
    
    distances, predecessors = dijkstra(graph, 'A')
    
    print("Weighted Graph:")
    print("  A--4--B, A--2--C, B--1--C, B--5--D, C--8--D, C--10--E, D--2--E")
    
    print(f"\nShortest distances from A:")
    for v in sorted(distances):
        print(f"  d(A, {v}) = {distances[v]}")
    
    print(f"\nShortest path A → E:")
    path = reconstruct_path(predecessors, 'E')
    print(f"  {' → '.join(path)}")


def example_7_graph_properties():
    """
    Example 7: Computing Graph Properties
    =====================================
    Density, diameter, and clustering.
    """
    print("\n" + "=" * 60)
    print("Example 7: Graph Properties")
    print("=" * 60)
    
    def compute_all_distances(graph):
        """Compute all-pairs shortest paths using BFS."""
        distances = {}
        
        for start in graph:
            dist = {start: 0}
            queue = deque([start])
            
            while queue:
                v = queue.popleft()
                for neighbor in graph[v]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[v] + 1
                        queue.append(neighbor)
            
            distances[start] = dist
        
        return distances
    
    def graph_diameter(graph):
        """Compute diameter (longest shortest path)."""
        distances = compute_all_distances(graph)
        
        max_dist = 0
        for u in distances:
            for v in distances[u]:
                if distances[u][v] != float('inf'):
                    max_dist = max(max_dist, distances[u][v])
        
        return max_dist
    
    def clustering_coefficient(graph, v):
        """Local clustering coefficient of vertex v."""
        neighbors = list(graph[v])
        k = len(neighbors)
        
        if k < 2:
            return 0.0
        
        # Count edges between neighbors
        edges = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in graph[neighbors[i]]:
                    edges += 1
        
        max_edges = k * (k - 1) / 2
        return edges / max_edges
    
    # Example graph
    graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'C'],
        'C': ['A', 'B', 'D'],
        'D': ['A', 'C', 'E'],
        'E': ['D']
    }
    
    n = len(graph)
    m = sum(len(neighbors) for neighbors in graph.values()) // 2
    
    print(f"Graph: {n} vertices, {m} edges")
    
    # Density
    density = 2 * m / (n * (n - 1))
    print(f"\nDensity: {density:.3f}")
    
    # Diameter
    diam = graph_diameter(graph)
    print(f"Diameter: {diam}")
    
    # Clustering coefficients
    print(f"\nLocal clustering coefficients:")
    for v in sorted(graph.keys()):
        cc = clustering_coefficient(graph, v)
        print(f"  C({v}) = {cc:.3f}")
    
    # Average clustering
    avg_cc = sum(clustering_coefficient(graph, v) for v in graph) / n
    print(f"\nAverage clustering: {avg_cc:.3f}")


def example_8_centrality_measures():
    """
    Example 8: Centrality Measures
    ==============================
    Identify important nodes in a network.
    """
    print("\n" + "=" * 60)
    print("Example 8: Centrality Measures")
    print("=" * 60)
    
    def degree_centrality(graph):
        """Degree centrality: normalized degree."""
        n = len(graph)
        return {v: len(neighbors) / (n - 1) for v, neighbors in graph.items()}
    
    def closeness_centrality(graph):
        """Closeness centrality: inverse average distance."""
        n = len(graph)
        centrality = {}
        
        for start in graph:
            # BFS for distances
            dist = {start: 0}
            queue = deque([start])
            
            while queue:
                v = queue.popleft()
                for neighbor in graph[v]:
                    if neighbor not in dist:
                        dist[neighbor] = dist[v] + 1
                        queue.append(neighbor)
            
            total_dist = sum(dist.values())
            centrality[start] = (n - 1) / total_dist if total_dist > 0 else 0
        
        return centrality
    
    def betweenness_centrality(graph):
        """Betweenness centrality: fraction of shortest paths through node."""
        n = len(graph)
        betweenness = {v: 0.0 for v in graph}
        
        for s in graph:
            # BFS from s
            pred = {v: [] for v in graph}
            dist = {s: 0}
            sigma = {v: 0 for v in graph}
            sigma[s] = 1
            
            queue = deque([s])
            stack = []
            
            while queue:
                v = queue.popleft()
                stack.append(v)
                
                for w in graph[v]:
                    # First time visiting w
                    if w not in dist:
                        dist[w] = dist[v] + 1
                        queue.append(w)
                    
                    # Shortest path to w via v
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            
            # Accumulate
            delta = {v: 0.0 for v in graph}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += sigma[v] / sigma[w] * (1 + delta[w])
                if w != s:
                    betweenness[w] += delta[w]
        
        # Normalize
        scale = 2 / ((n - 1) * (n - 2))
        return {v: b * scale for v, b in betweenness.items()}
    
    # Star graph (central node should have high centrality)
    star_graph = {
        'center': ['A', 'B', 'C', 'D'],
        'A': ['center'],
        'B': ['center'],
        'C': ['center'],
        'D': ['center']
    }
    
    print("Star Graph (center connected to A, B, C, D):")
    
    dc = degree_centrality(star_graph)
    cc = closeness_centrality(star_graph)
    bc = betweenness_centrality(star_graph)
    
    print(f"\n{'Vertex':<10} {'Degree':<12} {'Closeness':<12} {'Betweenness'}")
    print("-" * 50)
    
    for v in sorted(star_graph.keys()):
        print(f"{v:<10} {dc[v]:<12.3f} {cc[v]:<12.3f} {bc[v]:<12.3f}")


def example_9_bipartite_graphs():
    """
    Example 9: Bipartite Graphs
    ===========================
    Two-colorable graphs and matching.
    """
    print("\n" + "=" * 60)
    print("Example 9: Bipartite Graphs")
    print("=" * 60)
    
    def is_bipartite(graph):
        """Check if graph is bipartite using 2-coloring."""
        color = {}
        
        for start in graph:
            if start in color:
                continue
            
            queue = deque([start])
            color[start] = 0
            
            while queue:
                v = queue.popleft()
                
                for neighbor in graph[v]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[v]
                        queue.append(neighbor)
                    elif color[neighbor] == color[v]:
                        return False, {}
        
        return True, color
    
    # Bipartite graph (users and items)
    bipartite_graph = {
        'U1': ['I1', 'I2'],
        'U2': ['I2', 'I3'],
        'U3': ['I1', 'I3'],
        'I1': ['U1', 'U3'],
        'I2': ['U1', 'U2'],
        'I3': ['U2', 'U3']
    }
    
    is_bip, coloring = is_bipartite(bipartite_graph)
    
    print("User-Item Graph:")
    print("  U1-I1, U1-I2, U2-I2, U2-I3, U3-I1, U3-I3")
    print(f"\nIs bipartite: {is_bip}")
    
    if is_bip:
        set_0 = [v for v, c in coloring.items() if c == 0]
        set_1 = [v for v, c in coloring.items() if c == 1]
        print(f"Partition 1 (Users): {sorted(set_0)}")
        print(f"Partition 2 (Items): {sorted(set_1)}")
    
    # Non-bipartite graph (odd cycle)
    non_bipartite = {
        'A': ['B', 'C'],
        'B': ['A', 'C'],
        'C': ['A', 'B']
    }
    
    is_bip2, _ = is_bipartite(non_bipartite)
    print(f"\nTriangle (A-B-C-A) is bipartite: {is_bip2}")


def example_10_dag_and_topological_sort():
    """
    Example 10: DAG and Topological Sort
    ====================================
    Order vertices respecting dependencies.
    """
    print("\n" + "=" * 60)
    print("Example 10: DAG and Topological Sort")
    print("=" * 60)
    
    def topological_sort(graph):
        """
        Kahn's algorithm for topological sort.
        Returns sorted list or None if cycle exists.
        """
        # Compute in-degrees
        in_degree = {v: 0 for v in graph}
        for v in graph:
            for neighbor in graph[v]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
        
        # Find vertices with no incoming edges
        queue = deque([v for v in in_degree if in_degree[v] == 0])
        result = []
        
        while queue:
            v = queue.popleft()
            result.append(v)
            
            for neighbor in graph[v]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) == len(graph):
            return result
        else:
            return None  # Cycle exists
    
    # Task dependency DAG
    tasks = {
        'install_deps': ['compile'],
        'compile': ['test', 'lint'],
        'test': ['deploy'],
        'lint': ['deploy'],
        'deploy': []
    }
    
    print("Task Dependencies:")
    for task, deps in tasks.items():
        if deps:
            print(f"  {task} → {deps}")
        else:
            print(f"  {task} (final)")
    
    order = topological_sort(tasks)
    
    if order:
        print(f"\nTopological order: {order}")
        print("(Each task can run after its predecessors)")
    else:
        print("\nCycle detected - not a DAG!")


def example_11_graph_coloring():
    """
    Example 11: Graph Coloring
    ==========================
    Assign colors so no adjacent vertices share a color.
    """
    print("\n" + "=" * 60)
    print("Example 11: Graph Coloring")
    print("=" * 60)
    
    def greedy_coloring(graph):
        """Greedy graph coloring algorithm."""
        color = {}
        
        for v in graph:
            # Find colors used by neighbors
            neighbor_colors = {color[n] for n in graph[v] if n in color}
            
            # Find smallest available color
            c = 0
            while c in neighbor_colors:
                c += 1
            
            color[v] = c
        
        return color
    
    def chromatic_number_upper_bound(graph):
        """Upper bound on chromatic number."""
        coloring = greedy_coloring(graph)
        return max(coloring.values()) + 1
    
    # Example graph (wheel graph)
    wheel = {
        'center': ['v1', 'v2', 'v3', 'v4', 'v5'],
        'v1': ['center', 'v2', 'v5'],
        'v2': ['center', 'v1', 'v3'],
        'v3': ['center', 'v2', 'v4'],
        'v4': ['center', 'v3', 'v5'],
        'v5': ['center', 'v4', 'v1']
    }
    
    coloring = greedy_coloring(wheel)
    
    print("Wheel Graph W5:")
    print("  (center connected to all, outer cycle v1-v2-v3-v4-v5-v1)")
    
    print(f"\nGreedy coloring:")
    for v in sorted(wheel.keys()):
        print(f"  {v}: color {coloring[v]}")
    
    num_colors = max(coloring.values()) + 1
    print(f"\nColors used: {num_colors}")
    print(f"(Chromatic number χ(W5) = 4 for odd wheel)")


def example_12_network_analysis():
    """
    Example 12: Social Network Analysis
    ====================================
    Analyzing a small social network.
    """
    print("\n" + "=" * 60)
    print("Example 12: Social Network Analysis")
    print("=" * 60)
    
    # Small social network
    network = {
        'Alice': ['Bob', 'Carol', 'Dave'],
        'Bob': ['Alice', 'Carol', 'Eve'],
        'Carol': ['Alice', 'Bob', 'Dave', 'Eve'],
        'Dave': ['Alice', 'Carol'],
        'Eve': ['Bob', 'Carol'],
        'Frank': ['Grace'],
        'Grace': ['Frank']
    }
    
    print("Social Network:")
    for person, friends in sorted(network.items()):
        print(f"  {person}: {friends}")
    
    # Basic statistics
    n = len(network)
    m = sum(len(friends) for friends in network.values()) // 2
    
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {m}")
    print(f"  Density: {2*m / (n*(n-1)):.3f}")
    
    # Degree distribution
    degrees = [len(friends) for friends in network.values()]
    print(f"\nDegree Statistics:")
    print(f"  Min: {min(degrees)}, Max: {max(degrees)}, Avg: {sum(degrees)/len(degrees):.2f}")
    
    # Find most connected person
    most_connected = max(network, key=lambda x: len(network[x]))
    print(f"\nMost connected: {most_connected} ({len(network[most_connected])} friends)")
    
    # Find isolated group
    print("\nNetwork has disconnected component: {Frank, Grace}")


def run_all_examples():
    """Run all examples."""
    example_1_graph_creation()
    example_2_directed_graphs()
    example_3_adjacency_matrix()
    example_4_bfs_and_dfs()
    example_5_connected_components()
    example_6_shortest_paths()
    example_7_graph_properties()
    example_8_centrality_measures()
    example_9_bipartite_graphs()
    example_10_dag_and_topological_sort()
    example_11_graph_coloring()
    example_12_network_analysis()


if __name__ == "__main__":
    run_all_examples()
