# Graph Theory Basics

[← Previous: Numerical Methods](../../10-Numerical-Methods) | [Next: Graph Representations →](../02-Graph-Representations)

## Overview

Graphs are mathematical structures modeling pairwise relations between objects. They are fundamental to network analysis, social networks, knowledge graphs, and increasingly to deep learning through graph neural networks.

## Files in This Section

| File | Description |
|------|-------------|
| [examples.ipynb](examples.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Learning Objectives

- Understand graph terminology and properties
- Differentiate between graph types
- Analyze graph connectivity and structure
- Recognize graph-based ML applications

## 1. Basic Definitions

### What is a Graph?

A **graph** $G = (V, E)$ consists of:

- **Vertices** (nodes) $V = \{v_1, v_2, \ldots, v_n\}$
- **Edges** (links) $E \subseteq V \times V$

```
Simple Graph Example:

    1 ─────── 2
    │         │
    │         │
    4 ─────── 3

    V = {1, 2, 3, 4}
    E = {(1,2), (1,4), (2,3), (3,4)}
```

### Types of Graphs

**Undirected Graph:**

- Edges have no direction: $(u, v) = (v, u)$
- Social network friendships

**Directed Graph (Digraph):**

- Edges have direction: $(u, v) \neq (v, u)$
- Web links, citations

```
Directed Graph:

    A ────→ B
    │       ↑
    ↓       │
    C ←──── D
```

**Weighted Graph:**

- Edges have associated weights $w: E \to \mathbb{R}$
- Road networks with distances

**Bipartite Graph:**

- Vertices can be split into two disjoint sets
- Edges only between sets, not within
- User-item interactions

```
Bipartite Graph:

    Users      Items
      U1 ─────── I1
         ╲     ╱
          ╲   ╱
      U2 ───✕─── I2
          ╱   ╲
         ╱     ╲
      U3 ─────── I3
```

**Multigraph:**

- Multiple edges between same pair of vertices
- Self-loops allowed

## 2. Graph Properties

### Vertex Properties

**Degree** of vertex $v$:

- **Undirected:** $\deg(v) = |\{u : (u,v) \in E\}|$
- **Directed:** in-degree + out-degree

$$\text{In-degree: } \deg^-(v) = |\{u : (u,v) \in E\}|$$
$$\text{Out-degree: } \deg^+(v) = |\{(v,u) : (v,u) \in E\}|$$

**Handshaking Lemma:**
$$\sum_{v \in V} \deg(v) = 2|E|$$

### Graph Properties

**Order:** Number of vertices $|V|$

**Size:** Number of edges $|E|$

**Density:**
$$\text{density} = \frac{|E|}{|V|(|V|-1)/2} \text{ (undirected)}$$

- Dense: $|E| \approx O(|V|^2)$
- Sparse: $|E| \approx O(|V|)$

## 3. Paths and Connectivity

### Paths

**Walk:** Sequence of vertices where consecutive vertices are connected

**Path:** Walk with no repeated vertices

**Cycle:** Path that starts and ends at same vertex

**Path Length:** Number of edges in path

```
Path from A to D:

    A → B → C → D

    Length = 3
```

### Connectivity

**Connected Graph:** Path exists between every pair of vertices

**Connected Component:** Maximal connected subgraph

```
Graph with 2 Components:

    1 ─── 2       5 ─── 6
    │     │           │
    4 ─── 3       7 ─── 8

    Component 1: {1,2,3,4}
    Component 2: {5,6,7,8}
```

**Strongly Connected (Directed):** Path exists in both directions between every pair

**Weakly Connected (Directed):** Underlying undirected graph is connected

## 4. Special Graphs

### Complete Graph $K_n$

Every pair of vertices is connected:
$$|E| = \binom{n}{2} = \frac{n(n-1)}{2}$$

```
K₄:
    1 ─────── 2
    │ ╲     ╱ │
    │   ╲ ╱   │
    │   ╱ ╲   │
    │ ╱     ╲ │
    4 ─────── 3
```

### Tree

Connected graph with no cycles:
$$|E| = |V| - 1$$

```
Tree:
         1
       ╱ │ ╲
      2  3  4
     ╱ ╲    │
    5   6   7
```

### Directed Acyclic Graph (DAG)

Directed graph with no cycles. Used for:

- Computational graphs
- Bayesian networks
- Task dependencies

```
DAG:
    A ────→ B ────→ D
    │               ↑
    └──→ C ─────────┘
```

## 5. Graph Metrics

### Distance Metrics

**Shortest Path:** $d(u,v)$ = minimum path length from $u$ to $v$

**Eccentricity:** $\epsilon(v) = \max_{u \in V} d(u, v)$

**Diameter:** $\text{diam}(G) = \max_{v \in V} \epsilon(v)$

**Radius:** $\text{rad}(G) = \min_{v \in V} \epsilon(v)$

### Centrality Measures

**Degree Centrality:**
$$C_D(v) = \frac{\deg(v)}{n-1}$$

**Closeness Centrality:**
$$C_C(v) = \frac{n-1}{\sum_{u \neq v} d(u,v)}$$

**Betweenness Centrality:**
$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

where $\sigma_{st}$ = number of shortest paths from $s$ to $t$

```
Betweenness Example:

    A ─── B ─── C
          │
          D

    B has high betweenness (all paths through B)
```

### Clustering Coefficient

Local clustering (how connected are neighbors):
$$C(v) = \frac{2 \cdot |\text{edges between neighbors of } v|}{\deg(v)(\deg(v)-1)}$$

Global clustering:
$$C = \frac{3 \times \text{number of triangles}}{\text{number of connected triples}}$$

## 6. Subgraphs and Structures

### Subgraph

$H = (V_H, E_H)$ is subgraph of $G$ if $V_H \subseteq V$ and $E_H \subseteq E$

### Induced Subgraph

Subgraph containing all edges of $G$ between vertices in $V_H$

### Common Structures

**Clique:** Complete subgraph (everyone connected)

**Independent Set:** No edges between vertices

**Matching:** Set of edges with no shared vertices

```
Clique of size 3:      Independent Set:

    A ─── B               A   B
     ╲   ╱
      ╲ ╱                 C   D
       C
```

## 7. Graph Isomorphism

Graphs $G_1$ and $G_2$ are **isomorphic** if there exists a bijection $f: V_1 \to V_2$ preserving edges:
$$(u, v) \in E_1 \Leftrightarrow (f(u), f(v)) \in E_2$$

```
Isomorphic Graphs:

    1 ─── 2         A ─── D
    │     │    ≅    │     │
    4 ─── 3         B ─── C

    f: 1→A, 2→D, 3→C, 4→B
```

## 8. Applications in Machine Learning

### Knowledge Graphs

Entities as nodes, relations as edges:

```
    [Albert Einstein] ──born_in──→ [Germany]
           │
       invented
           │
           ↓
    [Theory of Relativity]
```

### Computational Graphs

Neural network operations:

```
    x ─┬─→ [*W] ─→ [+b] ─→ [σ] ─→ y
       │              ↑
       └─→ [concat] ──┘
```

### Social Networks

- Community detection
- Influence propagation
- Friend recommendation

### Molecular Graphs

Atoms as nodes, bonds as edges:

```
    H       H
     ╲     ╱
      C ═ C      (Ethene/Ethylene)
     ╱     ╲
    H       H
```

## Summary

| Concept    | Formula/Definition                      |
| ---------- | --------------------------------------- |
| Degree     | Number of edges incident to vertex      |
| Path       | Sequence of vertices without repetition |
| Connected  | Path exists between all pairs           |
| Complete   | All pairs connected                     |
| Tree       | Connected, no cycles, $\|E\| = \|V\|-1$ |
| Clustering | Triangle density around vertex          |
| Centrality | Importance measure for vertices         |

## Key Takeaways

1. **Graphs model relationships** between entities
2. **Directed vs undirected** matters for applications
3. **Connectivity** determines reachability
4. **Centrality measures** identify important nodes
5. **Clustering** reveals community structure
6. **Graph properties** inform algorithm design

## Practice Problems

1. For a graph with 6 vertices and 8 edges, compute density
2. Find all connected components in a given graph
3. Compute degree centrality for all nodes
4. Identify whether a given graph is bipartite
5. Find the diameter of a small network

## Exercises

1. **Graph Construction**: Given a list of edges, construct both directed and undirected graph representations and verify the handshaking lemma
2. **Path Finding**: Implement an algorithm to find all simple paths between two vertices in an undirected graph
3. **Bipartite Verification**: Write a function that determines if a graph is bipartite and returns the two vertex sets if it is
4. **Clustering Coefficient**: Calculate the local and global clustering coefficients for a social network dataset
5. **Graph Isomorphism**: Given two small graphs, determine if they are isomorphic by comparing their structural properties
