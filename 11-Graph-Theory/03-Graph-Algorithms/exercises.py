"""
Graph Algorithms - Exercises
============================

Hands-on exercises implementing classic graph algorithms
with applications to machine learning.
"""

import numpy as np
from collections import defaultdict, deque
import heapq
from typing import List, Tuple, Dict, Set, Optional, Any


class GraphAlgorithmsExercises:
    """Exercises for graph algorithm implementations."""
    
    # =========================================================================
    # Exercise 1: Multi-Source BFS
    # =========================================================================
    
    @staticmethod
    def exercise_1_multi_source_bfs():
        """
        Implement multi-source BFS.
        
        Given multiple source vertices, find the shortest distance
        from ANY source to each vertex. This is useful for:
        - Finding nearest facility
        - Label propagation in semi-supervised learning
        
        Also implement:
        - Level-wise node retrieval
        - Boundary detection (nodes at exactly distance k)
        """
        
        def multi_source_bfs(adj_list: Dict[int, List[int]], 
                            sources: List[int]) -> Dict[int, int]:
            """
            Find shortest distance from any source to each vertex.
            
            Args:
                adj_list: Adjacency list
                sources: List of source vertices
                
            Returns:
                distances: {vertex: min distance to any source}
            """
            # TODO: Implement
            # Hint: Initialize queue with all sources at distance 0
            pass
        
        def get_boundary_nodes(adj_list: Dict[int, List[int]],
                              sources: List[int],
                              k: int) -> Set[int]:
            """
            Get all nodes at exactly distance k from sources.
            
            This is the "k-hop boundary" useful for GNN receptive fields.
            """
            # TODO: Implement
            pass
        
        def label_propagation_step(adj_list: Dict[int, List[int]],
                                   labels: Dict[int, int],
                                   unlabeled: Set[int]) -> Dict[int, int]:
            """
            One step of label propagation.
            
            Each unlabeled node adopts the most common label among neighbors.
            """
            # TODO: Implement
            pass
        
        # Test
        print("Exercise 1: Multi-Source BFS")
        print("-" * 40)
        
        adj_list = {
            0: [1, 2], 1: [0, 3], 2: [0, 3, 4],
            3: [1, 2, 5], 4: [2, 5], 5: [3, 4]
        }
        sources = [0, 5]
        
        # Uncomment to test:
        # distances = multi_source_bfs(adj_list, sources)
        # print(f"Distances from sources {sources}: {distances}")
        # boundary = get_boundary_nodes(adj_list, sources, k=2)
        # print(f"Nodes at distance 2: {boundary}")
        
        return multi_source_bfs, get_boundary_nodes
    
    # =========================================================================
    # Exercise 2: Cycle Detection
    # =========================================================================
    
    @staticmethod
    def exercise_2_cycle_detection():
        """
        Implement cycle detection for both directed and undirected graphs.
        
        Applications:
        - Detecting circular dependencies
        - Validating DAGs for computation graphs
        - Finding feedback loops
        """
        
        def has_cycle_undirected(adj_list: Dict[int, List[int]], 
                                 vertices: List[int]) -> bool:
            """
            Detect cycle in undirected graph using DFS.
            
            A cycle exists if we visit an already-visited node
            that is not the parent.
            """
            # TODO: Implement
            pass
        
        def has_cycle_directed(adj_list: Dict[int, List[int]], 
                              vertices: List[int]) -> bool:
            """
            Detect cycle in directed graph using DFS.
            
            Use "coloring": WHITE (unvisited), GRAY (in stack), BLACK (done).
            Cycle exists if we reach a GRAY node.
            """
            # TODO: Implement
            pass
        
        def find_cycle(adj_list: Dict[int, List[int]], 
                      vertices: List[int]) -> List[int]:
            """
            Find and return one cycle if it exists.
            
            Returns empty list if no cycle.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 2: Cycle Detection")
        print("-" * 40)
        
        # Undirected with cycle
        adj_undirected = {0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}
        
        # Directed DAG (no cycle)
        adj_dag = {0: [1, 2], 1: [3], 2: [3], 3: []}
        
        # Directed with cycle
        adj_cycle = {0: [1], 1: [2], 2: [0]}
        
        # Uncomment to test:
        # print(f"Undirected has cycle: {has_cycle_undirected(adj_undirected, list(adj_undirected.keys()))}")
        # print(f"Directed DAG has cycle: {has_cycle_directed(adj_dag, list(adj_dag.keys()))}")
        # print(f"Directed has cycle: {has_cycle_directed(adj_cycle, list(adj_cycle.keys()))}")
        
        return has_cycle_undirected, has_cycle_directed
    
    # =========================================================================
    # Exercise 3: A* Search Algorithm
    # =========================================================================
    
    @staticmethod
    def exercise_3_astar():
        """
        Implement A* search algorithm.
        
        A* = Dijkstra + heuristic guidance.
        f(n) = g(n) + h(n) where:
        - g(n) = actual cost from start
        - h(n) = estimated cost to goal (must be admissible)
        
        Applications:
        - Pathfinding in games/robotics
        - Route planning
        """
        
        def astar(adj_list: Dict[int, List[Tuple[int, float]]],
                  start: int,
                  goal: int,
                  heuristic: Dict[int, float]) -> Tuple[List[int], float]:
            """
            A* search algorithm.
            
            Args:
                adj_list: {vertex: [(neighbor, cost), ...]}
                start: Start vertex
                goal: Goal vertex
                heuristic: {vertex: estimated cost to goal}
                
            Returns:
                path: List of vertices from start to goal
                cost: Total path cost
            """
            # TODO: Implement
            # Use priority queue with f-value = g + h
            pass
        
        def euclidean_heuristic(positions: Dict[int, Tuple[float, float]],
                               goal: int) -> Dict[int, float]:
            """
            Compute Euclidean distance heuristic.
            
            For grid/spatial graphs.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 3: A* Search")
        print("-" * 40)
        
        # Graph with positions
        positions = {
            0: (0, 0), 1: (1, 1), 2: (2, 0),
            3: (3, 1), 4: (4, 0), 5: (5, 1)
        }
        
        adj_list = {
            0: [(1, 1.5), (2, 2.0)],
            1: [(0, 1.5), (2, 1.0), (3, 1.5)],
            2: [(0, 2.0), (1, 1.0), (4, 2.0)],
            3: [(1, 1.5), (4, 1.0), (5, 1.5)],
            4: [(2, 2.0), (3, 1.0), (5, 1.5)],
            5: [(3, 1.5), (4, 1.5)]
        }
        
        # Uncomment to test:
        # h = euclidean_heuristic(positions, goal=5)
        # path, cost = astar(adj_list, start=0, goal=5, heuristic=h)
        # print(f"Path: {path}, Cost: {cost}")
        
        return astar
    
    # =========================================================================
    # Exercise 4: All Paths Between Two Vertices
    # =========================================================================
    
    @staticmethod
    def exercise_4_all_paths():
        """
        Find all paths between two vertices.
        
        Applications:
        - Network reliability
        - Feature extraction (path-based features)
        - Understanding graph structure
        """
        
        def find_all_paths(adj_list: Dict[int, List[int]],
                          start: int,
                          end: int,
                          max_length: Optional[int] = None) -> List[List[int]]:
            """
            Find all simple paths from start to end.
            
            Args:
                adj_list: Adjacency list
                start: Start vertex
                end: End vertex
                max_length: Maximum path length (optional)
                
            Returns:
                List of all paths (each path is a list of vertices)
            """
            # TODO: Implement using DFS with backtracking
            pass
        
        def count_paths(adj_list: Dict[int, List[int]],
                       start: int,
                       end: int,
                       length: int) -> int:
            """
            Count paths of exactly given length (can revisit nodes).
            
            This can be computed using matrix power: A^length[start, end]
            """
            # TODO: Implement
            pass
        
        def find_shortest_paths(adj_list: Dict[int, List[int]],
                               start: int,
                               end: int) -> List[List[int]]:
            """Find all shortest paths between start and end."""
            # TODO: Implement using BFS
            pass
        
        # Test
        print("\nExercise 4: All Paths")
        print("-" * 40)
        
        adj_list = {
            0: [1, 2],
            1: [2, 3],
            2: [3],
            3: []
        }
        
        # Uncomment to test:
        # paths = find_all_paths(adj_list, 0, 3)
        # print(f"All paths from 0 to 3: {paths}")
        # shortest = find_shortest_paths(adj_list, 0, 3)
        # print(f"Shortest paths: {shortest}")
        
        return find_all_paths, find_shortest_paths
    
    # =========================================================================
    # Exercise 5: Articulation Points and Bridges
    # =========================================================================
    
    @staticmethod
    def exercise_5_articulation_bridges():
        """
        Find articulation points and bridges.
        
        Articulation point: Vertex whose removal disconnects the graph.
        Bridge: Edge whose removal disconnects the graph.
        
        Applications:
        - Network vulnerability analysis
        - Critical node/edge identification
        """
        
        def find_articulation_points(adj_list: Dict[int, List[int]],
                                    vertices: List[int]) -> List[int]:
            """
            Find all articulation points using Tarjan's algorithm.
            
            Use DFS with discovery times and low values.
            """
            # TODO: Implement
            pass
        
        def find_bridges(adj_list: Dict[int, List[int]],
                        vertices: List[int]) -> List[Tuple[int, int]]:
            """
            Find all bridges using Tarjan's algorithm.
            
            An edge (u, v) is a bridge if low[v] > disc[u].
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 5: Articulation Points and Bridges")
        print("-" * 40)
        
        adj_list = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1, 3],
            3: [2, 4],
            4: [3, 5, 6],
            5: [4, 6],
            6: [4, 5]
        }
        
        # Uncomment to test:
        # art_points = find_articulation_points(adj_list, list(adj_list.keys()))
        # bridges = find_bridges(adj_list, list(adj_list.keys()))
        # print(f"Articulation points: {art_points}")
        # print(f"Bridges: {bridges}")
        
        return find_articulation_points, find_bridges
    
    # =========================================================================
    # Exercise 6: Maximum Flow (Ford-Fulkerson)
    # =========================================================================
    
    @staticmethod
    def exercise_6_max_flow():
        """
        Implement maximum flow using Ford-Fulkerson method.
        
        Applications:
        - Bipartite matching
        - Network routing
        - Assignment problems
        """
        
        def ford_fulkerson(capacity: Dict[Tuple[int, int], float],
                          source: int,
                          sink: int,
                          vertices: List[int]) -> float:
            """
            Ford-Fulkerson maximum flow.
            
            Args:
                capacity: {(u, v): capacity} edge capacities
                source: Source vertex
                sink: Sink vertex
                vertices: All vertices
                
            Returns:
                Maximum flow value
            """
            # TODO: Implement
            # 1. While there exists augmenting path:
            #    a. Find path using BFS (Edmonds-Karp)
            #    b. Find bottleneck capacity
            #    c. Update residual graph
            pass
        
        def bipartite_matching(left: List[int],
                              right: List[int],
                              edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
            """
            Find maximum bipartite matching using max flow.
            
            Convert to flow network with source connected to left,
            sink connected to right.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 6: Maximum Flow")
        print("-" * 40)
        
        # Flow network
        capacity = {
            (0, 1): 10, (0, 2): 10,
            (1, 2): 2, (1, 3): 4, (1, 4): 8,
            (2, 4): 9,
            (3, 5): 10,
            (4, 3): 6, (4, 5): 10
        }
        
        # Uncomment to test:
        # max_flow = ford_fulkerson(capacity, source=0, sink=5, vertices=list(range(6)))
        # print(f"Maximum flow: {max_flow}")
        
        return ford_fulkerson, bipartite_matching
    
    # =========================================================================
    # Exercise 7: Personalized PageRank
    # =========================================================================
    
    @staticmethod
    def exercise_7_personalized_pagerank():
        """
        Implement Personalized PageRank (PPR).
        
        PPR computes importance relative to a seed node(s).
        Used in recommendation, local community detection.
        
        PPR(v) = α * e_s + (1-α) * M * PPR
        where e_s is personalization vector.
        """
        
        def personalized_pagerank(adj_list: Dict[int, List[int]],
                                  seed: int,
                                  alpha: float = 0.15,
                                  max_iter: int = 100) -> Dict[int, float]:
            """
            Personalized PageRank with single seed.
            
            Args:
                adj_list: Adjacency list
                seed: Seed vertex
                alpha: Teleport probability (restart probability)
                max_iter: Maximum iterations
                
            Returns:
                PPR scores for each vertex
            """
            # TODO: Implement power iteration with personalization
            pass
        
        def approximate_ppr(adj_list: Dict[int, List[int]],
                           seed: int,
                           alpha: float = 0.15,
                           epsilon: float = 1e-6) -> Dict[int, float]:
            """
            Approximate PPR using push-based algorithm.
            
            More efficient for large graphs - only visits 
            relevant nodes.
            """
            # TODO: Implement push-based PPR
            pass
        
        def ppr_recommendation(adj_list: Dict[int, List[int]],
                              user: int,
                              items: Set[int],
                              alpha: float = 0.15) -> List[Tuple[int, float]]:
            """
            Item recommendation using PPR.
            
            Return items ranked by PPR score from user.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 7: Personalized PageRank")
        print("-" * 40)
        
        adj_list = {
            0: [1, 2],
            1: [0, 2, 3],
            2: [0, 1, 4],
            3: [1, 4],
            4: [2, 3, 5],
            5: [4]
        }
        
        # Uncomment to test:
        # ppr = personalized_pagerank(adj_list, seed=0)
        # print(f"PPR from node 0: {ppr}")
        
        return personalized_pagerank, approximate_ppr
    
    # =========================================================================
    # Exercise 8: Community Detection (Louvain)
    # =========================================================================
    
    @staticmethod
    def exercise_8_louvain():
        """
        Implement Louvain algorithm for community detection.
        
        Greedy modularity optimization:
        1. Assign each node to its own community
        2. Move nodes to maximize modularity gain
        3. Aggregate communities into super-nodes
        4. Repeat until no improvement
        """
        
        def compute_modularity(adj_matrix: np.ndarray,
                              communities: Dict[int, int]) -> float:
            """
            Compute modularity Q.
            
            Q = (1/2m) * Σ_ij [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
            """
            # TODO: Implement
            pass
        
        def modularity_gain(adj_matrix: np.ndarray,
                           node: int,
                           community: int,
                           communities: Dict[int, int]) -> float:
            """
            Compute modularity gain from moving node to community.
            """
            # TODO: Implement
            pass
        
        def louvain(adj_matrix: np.ndarray) -> Dict[int, int]:
            """
            Louvain community detection.
            
            Returns:
                communities: {node: community_id}
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 8: Louvain Community Detection")
        print("-" * 40)
        
        # Graph with clear community structure
        adj_matrix = np.array([
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0]
        ])
        
        # Uncomment to test:
        # communities = louvain(adj_matrix)
        # print(f"Communities: {communities}")
        # Q = compute_modularity(adj_matrix, communities)
        # print(f"Modularity: {Q:.4f}")
        
        return compute_modularity, louvain
    
    # =========================================================================
    # Exercise 9: Graph Kernels
    # =========================================================================
    
    @staticmethod
    def exercise_9_graph_kernels():
        """
        Implement graph kernels for graph similarity.
        
        Graph kernels measure similarity between graphs
        for graph classification tasks.
        """
        
        def shortest_path_kernel(adj1: np.ndarray, 
                                adj2: np.ndarray) -> float:
            """
            Shortest path graph kernel.
            
            K(G1, G2) = Σ_s,t,s',t' k(d(s,t), d(s',t'))
            where d is shortest path distance.
            """
            # TODO: Implement
            # 1. Compute APSP for both graphs
            # 2. Compare path length distributions
            pass
        
        def weisfeiler_lehman_subtree_kernel(adj1: np.ndarray,
                                            adj2: np.ndarray,
                                            h: int = 3) -> float:
            """
            WL subtree kernel.
            
            Iteratively refines node labels based on neighborhood.
            """
            # TODO: Implement
            pass
        
        def random_walk_kernel(adj1: np.ndarray,
                              adj2: np.ndarray,
                              steps: int = 5) -> float:
            """
            Random walk kernel.
            
            K(G1, G2) = Σ_l λ^l * |walks of length l in G1 × G2|
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 9: Graph Kernels")
        print("-" * 40)
        
        # Two similar graphs
        adj1 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        adj2 = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
        
        # Uncomment to test:
        # k_sp = shortest_path_kernel(adj1, adj2)
        # print(f"Shortest path kernel: {k_sp}")
        
        return shortest_path_kernel, weisfeiler_lehman_subtree_kernel
    
    # =========================================================================
    # Exercise 10: Computation Graph for Automatic Differentiation
    # =========================================================================
    
    @staticmethod
    def exercise_10_computation_graph():
        """
        Build and process computation graphs for autodiff.
        
        Applications:
        - Forward/backward pass ordering
        - Memory optimization
        - Parallelization analysis
        """
        
        class ComputationNode:
            """Node in computation graph."""
            
            def __init__(self, name: str, op: str = 'input'):
                self.name = name
                self.op = op  # 'input', 'add', 'mul', 'relu', etc.
                self.inputs = []
                self.value = None
                self.grad = None
        
        def build_computation_graph(expression: str) -> Tuple[Dict, List]:
            """
            Build computation graph from expression.
            
            Example: "z = relu(x * w + b)"
            
            Returns:
                nodes: {name: ComputationNode}
                edges: [(input_name, output_name)]
            """
            # TODO: Implement simple expression parser
            pass
        
        def topological_order_forward(nodes: Dict,
                                     edges: List) -> List[str]:
            """Get forward pass order."""
            # TODO: Implement
            pass
        
        def topological_order_backward(nodes: Dict,
                                      edges: List) -> List[str]:
            """Get backward pass order (reverse of forward)."""
            # TODO: Implement
            pass
        
        def compute_gradients(nodes: Dict,
                            edges: List,
                            output: str) -> Dict[str, float]:
            """
            Compute gradients using reverse-mode autodiff.
            
            This is what PyTorch/TensorFlow do!
            """
            # TODO: Implement
            pass
        
        def find_recomputation_candidates(nodes: Dict,
                                         edges: List,
                                         memory_limit: int) -> List[str]:
            """
            Find nodes that can be recomputed to save memory.
            
            Used in gradient checkpointing.
            """
            # TODO: Implement
            pass
        
        # Test
        print("\nExercise 10: Computation Graph")
        print("-" * 40)
        
        # Simple neural network: y = relu(W @ x + b)
        nodes = {
            'x': ComputationNode('x', 'input'),
            'W': ComputationNode('W', 'input'),
            'b': ComputationNode('b', 'input'),
            'Wx': ComputationNode('Wx', 'matmul'),
            'Wx_b': ComputationNode('Wx_b', 'add'),
            'y': ComputationNode('y', 'relu')
        }
        edges = [
            ('x', 'Wx'), ('W', 'Wx'),
            ('Wx', 'Wx_b'), ('b', 'Wx_b'),
            ('Wx_b', 'y')
        ]
        
        # Uncomment to test:
        # forward_order = topological_order_forward(nodes, edges)
        # backward_order = topological_order_backward(nodes, edges)
        # print(f"Forward order: {forward_order}")
        # print(f"Backward order: {backward_order}")
        
        return build_computation_graph, compute_gradients


def verify_implementations():
    """Verify all exercise implementations."""
    print("=" * 60)
    print("Graph Algorithms - Exercise Verification")
    print("=" * 60)
    
    exercises = GraphAlgorithmsExercises()
    
    exercises.exercise_1_multi_source_bfs()
    exercises.exercise_2_cycle_detection()
    exercises.exercise_3_astar()
    exercises.exercise_4_all_paths()
    exercises.exercise_5_articulation_bridges()
    exercises.exercise_6_max_flow()
    exercises.exercise_7_personalized_pagerank()
    exercises.exercise_8_louvain()
    exercises.exercise_9_graph_kernels()
    exercises.exercise_10_computation_graph()
    
    print("\n" + "=" * 60)
    print("Complete the TODO sections in each exercise!")
    print("=" * 60)


# =============================================================================
# Solutions (Reference Implementation)
# =============================================================================

class Solutions:
    """Reference solutions for exercises."""
    
    @staticmethod
    def solution_1_multi_source_bfs():
        """Solution for Exercise 1."""
        
        def multi_source_bfs(adj_list: Dict[int, List[int]], 
                            sources: List[int]) -> Dict[int, int]:
            distances = {s: 0 for s in sources}
            queue = deque(sources)
            
            while queue:
                u = queue.popleft()
                for v in adj_list.get(u, []):
                    if v not in distances:
                        distances[v] = distances[u] + 1
                        queue.append(v)
            
            return distances
        
        return multi_source_bfs
    
    @staticmethod
    def solution_2_cycle_detection():
        """Solution for Exercise 2."""
        
        def has_cycle_undirected(adj_list: Dict[int, List[int]], 
                                 vertices: List[int]) -> bool:
            visited = set()
            
            def dfs(u, parent):
                visited.add(u)
                for v in adj_list.get(u, []):
                    if v not in visited:
                        if dfs(v, u):
                            return True
                    elif v != parent:
                        return True
                return False
            
            for v in vertices:
                if v not in visited:
                    if dfs(v, -1):
                        return True
            return False
        
        def has_cycle_directed(adj_list: Dict[int, List[int]], 
                              vertices: List[int]) -> bool:
            WHITE, GRAY, BLACK = 0, 1, 2
            color = {v: WHITE for v in vertices}
            
            def dfs(u):
                color[u] = GRAY
                for v in adj_list.get(u, []):
                    if color[v] == GRAY:
                        return True
                    if color[v] == WHITE and dfs(v):
                        return True
                color[u] = BLACK
                return False
            
            for v in vertices:
                if color[v] == WHITE:
                    if dfs(v):
                        return True
            return False
        
        return has_cycle_undirected, has_cycle_directed
    
    @staticmethod
    def solution_7_personalized_pagerank():
        """Solution for Exercise 7."""
        
        def personalized_pagerank(adj_list: Dict[int, List[int]],
                                  seed: int,
                                  alpha: float = 0.15,
                                  max_iter: int = 100) -> Dict[int, float]:
            vertices = list(adj_list.keys())
            n = len(vertices)
            v_to_idx = {v: i for i, v in enumerate(vertices)}
            
            # Initialize
            pr = np.zeros(n)
            pr[v_to_idx[seed]] = 1.0
            
            # Compute out-degrees
            out_degree = np.zeros(n)
            for i, v in enumerate(vertices):
                out_degree[i] = len(adj_list.get(v, []))
            out_degree[out_degree == 0] = 1  # Handle dangling
            
            # Power iteration
            for _ in range(max_iter):
                pr_new = np.zeros(n)
                pr_new[v_to_idx[seed]] = alpha  # Personalization
                
                for u_idx, u in enumerate(vertices):
                    for v in adj_list.get(u, []):
                        v_idx = v_to_idx[v]
                        pr_new[v_idx] += (1 - alpha) * pr[u_idx] / out_degree[u_idx]
                
                if np.abs(pr_new - pr).sum() < 1e-8:
                    break
                pr = pr_new
            
            return {v: pr[i] for i, v in enumerate(vertices)}
        
        return personalized_pagerank
    
    @staticmethod
    def solution_8_modularity():
        """Solution for modularity computation."""
        
        def compute_modularity(adj_matrix: np.ndarray,
                              communities: Dict[int, int]) -> float:
            n = len(adj_matrix)
            m = adj_matrix.sum() / 2
            k = adj_matrix.sum(axis=1)
            
            Q = 0
            for i in range(n):
                for j in range(n):
                    if communities[i] == communities[j]:
                        Q += adj_matrix[i, j] - k[i] * k[j] / (2 * m)
            
            return Q / (2 * m)
        
        return compute_modularity


if __name__ == "__main__":
    verify_implementations()
