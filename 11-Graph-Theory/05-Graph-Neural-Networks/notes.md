[<- Back to Graph Theory](../README.md) | [Previous: Spectral Graph Theory <-](../04-Spectral-Graph-Theory/notes.md) | [Next: Random Graphs ->](../06-Random-Graphs/notes.md)

---

# Graph Neural Networks

> _"A graph neural network is a machine that reads a graph and learns by listening to its neighbors - one round of message passing at a time, until every node knows not just what it is, but where it stands in the web of relationships that defines it."_

## Overview

Graph Neural Networks (GNNs) are deep learning architectures designed for graph-structured data - the most general and pervasive data structure in science, engineering, and AI. Social networks, molecules, knowledge bases, protein interaction maps, citation graphs, program abstract syntax trees, and even the attention pattern of a transformer are all graphs. The central question GNNs answer is: **how do we learn representations of nodes, edges, and graphs that are sensitive to their structural context?**

The key insight, formalized by Gilmer et al. (2017) as the Message Passing Neural Network (MPNN) framework, is that useful graph representations can be built iteratively: in each round, every node collects information from its immediate neighbors, transforms that information with a learned function, and updates its own representation. After $k$ rounds, every node's representation reflects the structure of its $k$-hop neighborhood. This is the graph analogue of a convolutional layer in an image network, but operating on irregular, variable-size neighborhoods rather than fixed grids.

This section develops GNN theory from first principles. We begin with the formal setup (permutation invariance, task types), develop the MPNN framework as the unifying abstraction, and then study the major GNN families: Graph Convolutional Networks (Kipf & Welling, 2017), GraphSAGE (Hamilton et al., 2017), Graph Attention Networks (Velickovic et al., 2018), and Graph Isomorphism Networks (Xu et al., 2019). We then turn to two fundamental theoretical questions - **expressiveness** (what structural properties can a GNN detect?) and **depth pathologies** (what goes wrong as GNNs get deeper?) - and connect these to practical remedies. We close with Graph Transformers, training at scale, and a survey of modern applications including AlphaFold 2, knowledge graph reasoning, and LLM+GNN architectures.

Students who complete this section will be able to implement and reason about all major GNN architectures, understand the Weisfeiler-Leman expressiveness hierarchy, diagnose and fix over-smoothing and over-squashing, and read current GNN research papers independently.

## Prerequisites

- Graph definitions: $G = (V, E)$, vertex degrees, paths, connectivity - [11-01 Graph Basics](../01-Graph-Basics/notes.md)
- Adjacency matrix $A$, degree matrix $D$, Laplacian $L = D - A$ - [11-02 Graph Representations](../02-Graph-Representations/notes.md)
- BFS and $k$-hop neighborhoods - [11-03 Graph Algorithms](../03-Graph-Algorithms/notes.md)
- Spectral graph theory: normalized Laplacian, Chebyshev approximation, GCN spectral derivation, over-smoothing preview - [11-04 Spectral Graph Theory](../04-Spectral-Graph-Theory/notes.md)
- Eigenvalues, PSD matrices, the spectral theorem - [03-01 Eigenvalues and Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)
- Attention mechanisms and transformer architecture - [14 Math for Specific Models](../../14-Math-for-Specific-Models/README.md)
- Gradient descent and backpropagation - [08 Optimization](../../08-Optimization/README.md)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive derivations: GCN propagation, GAT attention visualization, WL color refinement, over-smoothing dynamics, graph pooling, neighbor sampling |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from MPNN implementation through WL expressiveness, over-smoothing diagnosis, and molecular property prediction |

## Learning Objectives

After completing this section, you will:

- Implement the MPNN framework and express GCN, GraphSAGE, GAT, and GIN as special cases
- Derive the GCN layer $H^{[l+1]} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{[l]}W^{[l]})$ from the spectral approximation and interpret it spatially
- Define permutation invariance and equivariance and explain why GNNs must satisfy these properties
- Implement neighbor sampling (GraphSAGE), attention-weighted aggregation (GAT), and multi-head graph attention (GATv2)
- State and prove the GNN expressiveness theorem (Xu et al. 2018): MPNNs are at most as powerful as 1-WL
- Implement the Weisfeiler-Leman color refinement algorithm and construct graphs that WL cannot distinguish
- Construct GIN and explain why sum aggregation is strictly more powerful than mean or max
- Quantify over-smoothing via Dirichlet energy decay and over-squashing via Jacobian analysis
- Apply DropEdge, PairNorm, residual connections, and graph rewiring to mitigate depth pathologies
- Describe the GPS framework and distinguish Laplacian PE, RWSE, and degree encoding for graph Transformers
- Explain how AlphaFold 2, PinSage, and Graph RAG (Microsoft 2024) use GNN mathematics in production

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Core Challenge: Learning on Non-Euclidean Data](#11-the-core-challenge-learning-on-non-euclidean-data)
  - [1.2 The Message-Passing Insight](#12-the-message-passing-insight)
  - [1.3 Two Perspectives: Spectral vs Spatial](#13-two-perspectives-spectral-vs-spatial)
  - [1.4 What Makes Graphs Different from Other Data](#14-what-makes-graphs-different-from-other-data)
  - [1.5 Historical Timeline (2013-2026)](#15-historical-timeline-20132026)
- [2. Formal Setup: Graphs as Data](#2-formal-setup-graphs-as-data)
  - [2.1 Graph with Node and Edge Features](#21-graph-with-node-and-edge-features)
  - [2.2 Learning Tasks on Graphs](#22-learning-tasks-on-graphs)
  - [2.3 Permutation Invariance and Equivariance](#23-permutation-invariance-and-equivariance)
  - [2.4 Representations: From Single Graph to Batched Graphs](#24-representations-from-single-graph-to-batched-graphs)
- [3. The MPNN Framework](#3-the-mpnn-framework)
  - [3.1 The Gilmer et al. (2017) Formulation](#31-the-gilmer-et-al-2017-formulation)
  - [3.2 Message Functions](#32-message-functions)
  - [3.3 Aggregation: Permutation-Invariant Pooling](#33-aggregation-permutation-invariant-pooling)
  - [3.4 Update Functions](#34-update-functions)
  - [3.5 Readout for Graph-Level Tasks](#35-readout-for-graph-level-tasks)
  - [3.6 MPNN Instances: Unifying GCN, GraphSAGE, GAT, GIN](#36-mpnn-instances-unifying-gcn-graphsage-gat-gin)
- [4. Graph Convolutional Networks (GCN)](#4-graph-convolutional-networks-gcn)
  - [4.1 Spectral Foundation Recap](#41-spectral-foundation-recap)
  - [4.2 The GCN Layer](#42-the-gcn-layer)
  - [4.3 Matrix View: Propagation Rule](#43-matrix-view-propagation-rule)
  - [4.4 Semi-Supervised Node Classification](#44-semi-supervised-node-classification)
  - [4.5 Strengths and Weaknesses of GCN](#45-strengths-and-weaknesses-of-gcn)
- [5. GraphSAGE: Inductive Learning on Large Graphs](#5-graphsage-inductive-learning-on-large-graphs)
  - [5.1 The Inductive Learning Problem](#51-the-inductive-learning-problem)
  - [5.2 GraphSAGE Framework](#52-graphsage-framework)
  - [5.3 Aggregation Strategies](#53-aggregation-strategies)
  - [5.4 Unsupervised Training and Link Prediction](#54-unsupervised-training-and-link-prediction)
  - [5.5 Scaling to Billion-Node Graphs](#55-scaling-to-billion-node-graphs)
- [6. Graph Attention Networks (GAT)](#6-graph-attention-networks-gat)
  - [6.1 The Attention Mechanism on Graphs](#61-the-attention-mechanism-on-graphs)
  - [6.2 GAT Layer: Attention Coefficients](#62-gat-layer-attention-coefficients)
  - [6.3 Multi-Head Attention for Graphs](#63-multi-head-attention-for-graphs)
  - [6.4 GATv2: Fixing the Static Attention Problem](#64-gatv2-fixing-the-static-attention-problem)
  - [6.5 Attention Sparsity and Interpretability](#65-attention-sparsity-and-interpretability)
- [7. Expressiveness and the Weisfeiler-Leman Test](#7-expressiveness-and-the-weisfeiler-leman-test)
  - [7.1 The Graph Isomorphism Problem](#71-the-graph-isomorphism-problem)
  - [7.2 1-WL and Color Refinement](#72-1-wl-and-color-refinement)
  - [7.3 The GNN Expressiveness Theorem](#73-the-gnn-expressiveness-theorem)
  - [7.4 GIN: The Most Expressive 1-WL GNN](#74-gin-the-most-expressive-1-wl-gnn)
  - [7.5 Beyond 1-WL: Higher-Order GNNs](#75-beyond-1-wl-higher-order-gnns)
  - [7.6 Structural Features and Distance Encoding](#76-structural-features-and-distance-encoding)
- [8. Over-Smoothing, Over-Squashing, and Depth](#8-over-smoothing-over-squashing-and-depth)
  - [8.1 Over-Smoothing: Formal Analysis](#81-over-smoothing-formal-analysis)
  - [8.2 Information-Theoretic View via Dirichlet Energy](#82-information-theoretic-view-via-dirichlet-energy)
  - [8.3 Over-Squashing: Bottleneck Nodes](#83-over-squashing-bottleneck-nodes)
  - [8.4 Remedies for Over-Smoothing](#84-remedies-for-over-smoothing)
  - [8.5 Graph Rewiring](#85-graph-rewiring)
  - [8.6 Depth vs Width Trade-off in GNNs](#86-depth-vs-width-trade-off-in-gnns)
- [9. Graph Pooling and Hierarchical GNNs](#9-graph-pooling-and-hierarchical-gnns)
  - [9.1 Why Pooling Is Non-Trivial on Graphs](#91-why-pooling-is-non-trivial-on-graphs)
  - [9.2 Global Pooling Methods](#92-global-pooling-methods)
  - [9.3 DiffPool: Differentiable Graph Pooling](#93-diffpool-differentiable-graph-pooling)
  - [9.4 MinCutPool and Spectral Pooling](#94-mincutpool-and-spectral-pooling)
  - [9.5 SAGPool: Self-Attention Graph Pooling](#95-sagpool-self-attention-graph-pooling)
- [10. Graph Transformers](#10-graph-transformers)
  - [10.1 Motivation: GNNs vs Transformers](#101-motivation-gnns-vs-transformers)
  - [10.2 Positional Encodings for Graphs](#102-positional-encodings-for-graphs)
  - [10.3 GPS Framework](#103-gps-framework)
  - [10.4 Graphormer](#104-graphormer)
  - [10.5 Graph Mamba and Sequence-Based Methods](#105-graph-mamba-and-sequence-based-methods)
- [11. Training and Scaling GNNs](#11-training-and-scaling-gnns)
  - [11.1 Full-Batch vs Mini-Batch Training](#111-full-batch-vs-mini-batch-training)
  - [11.2 Neighbor Sampling](#112-neighbor-sampling)
  - [11.3 Cluster-GCN](#113-cluster-gcn)
  - [11.4 GraphSAINT: Subgraph Sampling](#114-graphsaint-subgraph-sampling)
- [12. Applications in Machine Learning](#12-applications-in-machine-learning)
  - [12.1 Molecular Property Prediction and Drug Discovery](#121-molecular-property-prediction-and-drug-discovery)
  - [12.2 Knowledge Graph Reasoning](#122-knowledge-graph-reasoning)
  - [12.3 Recommendation Systems](#123-recommendation-systems)
  - [12.4 Code and Program Analysis](#124-code-and-program-analysis)
  - [12.5 LLM Integration: Graph RAG and Structure-Aware Language Models](#125-llm-integration-graph-rag-and-structure-aware-language-models)
  - [12.6 Financial Networks, Credit Risk and Default Contagion](#126-financial-networks-credit-risk-and-default-contagion)
- [13. Common Mistakes](#13-common-mistakes)
- [14. Exercises](#14-exercises)
- [15. Why This Matters for AI (2026 Perspective)](#15-why-this-matters-for-ai-2026-perspective)
- [16. Conceptual Bridge](#16-conceptual-bridge)

---

## 1. Intuition

### 1.1 The Core Challenge: Learning on Non-Euclidean Data

Standard deep learning architectures - convolutional networks, recurrent networks, transformers - were designed for data that lives on regular, well-ordered domains. An image is a grid: every pixel has exactly the same number of neighbors, arranged in the same spatial pattern. A sentence is a sequence: every token has a left neighbor and a right neighbor, and positions are canonically ordered. These regular structures make convolution and positional encodings natural.

Real-world data is rarely so obliging. A molecule has atoms connected by bonds in a pattern determined by chemistry, not a grid. A social network has users connected by friendships in patterns determined by behavior and geography. A knowledge base has concepts connected by relations in patterns determined by the world's semantic structure. A protein's function is determined by the 3D arrangement of amino acid residues, which a graph can represent but a grid cannot. These are **non-Euclidean** data structures: they have no natural notion of translation, no canonical coordinate system, no uniform neighborhood size.

The challenge GNNs solve is: **how do we define a learnable function on a graph that generalizes across different graphs, respects the graph's structure, and can be trained end-to-end by gradient descent?** This is harder than it sounds. The same function applied to two isomorphic graphs (graphs with the same structure but differently labeled vertices) should produce the same output. The function must handle graphs of arbitrary size and arbitrary vertex degrees. And it must be expressive enough to detect structural patterns (triangles, paths of length $k$, specific subgraph motifs) that determine the graph's properties.

**For AI:** In 2026, GNNs are production components in AlphaFold 2's structure module (atoms as nodes, spatial edges), PinSage (users and items as nodes, interactions as edges with 3 billion monthly active users), and Microsoft's Graph RAG system (entities and relations from documents as knowledge graphs for retrieval-augmented generation). Understanding GNNs is not optional for anyone working with structured, relational, or molecular data.

### 1.2 The Message-Passing Insight

The insight that makes GNNs possible is conceptually simple: **a node's representation should depend on the representations of its neighbors, and this dependency should be computed iteratively**.

Consider a social network node - a person. Their social identity depends on whom they know. But whom they know also depends on whom those people know, and so on. The first round of "listening to neighbors" gives each person a sense of their immediate social circle. The second round gives a sense of friends-of-friends. After $k$ rounds, each node's representation encodes the structure of its $k$-hop neighborhood.

This is **message passing**: in each layer, every node sends a message to its neighbors, every node aggregates the messages it receives, and every node updates its representation based on the aggregated messages. Formally, for node $v$ at layer $l$:

$$\mathbf{m}_v^{[l]} = \operatorname{AGGREGATE}^{[l]}\!\left(\left\{\mathbf{h}_u^{[l]} : u \in \mathcal{N}(v)\right\}\right)$$

$$\mathbf{h}_v^{[l+1]} = \operatorname{UPDATE}^{[l]}\!\left(\mathbf{h}_v^{[l]},\, \mathbf{m}_v^{[l]}\right)$$

where $\mathcal{N}(v)$ is the set of neighbors of $v$, $\mathbf{h}_v^{[l]} \in \mathbb{R}^d$ is node $v$'s representation at layer $l$, and $\mathbf{h}_v^{[0]} = \mathbf{x}_v$ is the initial node feature.

**The connection to BFS:** Breadth-first search from node $v$ explores exactly the nodes reachable in $k$ steps at step $k$. A $k$-layer GNN computes a representation that depends on exactly the nodes reachable in $k$ steps. A BFS is the non-learned, binary-valued version of message passing: it answers "which nodes are in the $k$-hop neighborhood?" whereas a GNN answers "what is the learned summary of the $k$-hop neighborhood?"

**For AI:** This message-passing structure is the graph analogue of the convolutional receptive field. A pixel in a CNN "sees" a $k \times k$ grid after $k$ convolutional layers. A node in a GNN "sees" its $k$-hop neighborhood after $k$ message-passing layers. The critical difference: the CNN's receptive field grows quadratically in $k$ (for a 2D grid), while the GNN's receptive field can grow exponentially in $k$ (for a well-connected graph), which creates both expressiveness and scalability challenges we will study in 8.

### 1.3 Two Perspectives: Spectral vs Spatial

Graph convolution was first defined in the spectral domain. The key idea from 11-04 is that the graph Laplacian $L = D - A$ admits an eigendecomposition $L = U \Lambda U^\top$, and the columns of $U$ form a graph Fourier basis. A graph signal $\mathbf{x} \in \mathbb{R}^n$ (one scalar per node) is filtered by applying a function to its Fourier coefficients:

$$\hat{\mathbf{x}} = U^\top \mathbf{x}, \qquad \text{filtered: } \mathbf{y} = U \cdot g(\Lambda) \cdot \hat{\mathbf{x}}$$

where $g(\Lambda)$ is a diagonal matrix of filter coefficients. Full spectral convolution requires computing the full eigendecomposition - $O(n^3)$ cost, infeasible for large graphs.

> **Recall from 11-04 7.5:** Kipf & Welling (2017) derived the GCN layer as a first-order Chebyshev polynomial approximation to spectral filtering, further simplified by setting $\lambda_{\max} \approx 2$ and adding self-loops. This gives the propagation rule $H^{[l+1]} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{[l]}W^{[l]})$ - see [11-04 7.5](../04-Spectral-Graph-Theory/notes.md#75-from-chebyshev-to-gcn) for the full derivation.

Once we arrive at the GCN layer, the spectral scaffolding can be discarded and the layer reinterpreted spatially: it is simply **normalized aggregation of neighbor features followed by a linear transformation and nonlinearity**. This spatial reinterpretation is what makes GNNs extensible: one can design new aggregation functions, learn attention weights, incorporate edge features, and handle directed graphs - none of which fit neatly into the spectral framework.

The two perspectives are complementary:
- **Spectral:** principled derivation, frequency interpretation, connection to graph signal processing; limited to fixed graphs, requires eigendecomposition or polynomial approximation
- **Spatial:** flexible, scalable, inductive, handles heterogeneous graphs; less theoretically constrained, harder to analyze convergence

Modern GNN research lives primarily in the spatial perspective, with spectral theory providing theoretical grounding and positional encodings (10).

### 1.4 What Makes Graphs Different from Other Data

Four properties of graph-structured data require fundamentally different architectural choices:

**1. Variable and irregular neighborhoods.** Node $v$ in a molecule may have 1 bond (hydrogen) or 4 bonds (carbon). Node $u$ in a social network may have 3 friends or 3,000. Standard convolution assumes a fixed-size, fixed-pattern receptive field. GNNs must aggregate over neighborhoods of arbitrary size - which forces a choice of aggregation function (sum, mean, max, attention) that is insensitive to neighborhood size.

**2. No canonical node ordering.** An image has a canonical pixel ordering (row-major). A sentence has a canonical word ordering (left-to-right). A graph has no such ordering. If we relabel the nodes of a graph, we get the same graph - just with different indices. This means a GNN must be **permutation equivariant** (the output node representations permute consistently with any relabeling of input nodes) and any graph-level prediction must be **permutation invariant** (the same under any relabeling). This rules out any architecture that depends on node indices.

**3. Structural information is in the topology.** Two molecules with identical atom composition but different bond structures (structural isomers) have very different chemical properties. The GNN must detect this topological difference from the adjacency structure alone. This is the expressiveness question of 7: which topological patterns can a GNN distinguish?

**4. Heterogeneous graphs.** Real graphs often have multiple types of nodes and edges (knowledge graphs: `Person --worksAt--> Company`, `Person --bornIn--> City`). Relational GNNs (R-GCN) and heterogeneous GNNs handle this by type-specific transformation matrices. This section focuses on homogeneous graphs; heterogeneous extensions are discussed in 12.2.

### 1.5 Historical Timeline (2013-2026)

```
GRAPH NEURAL NETWORK HISTORICAL TIMELINE
========================================================================

  2009  Scarselli et al. - Original GNN: fixed-point iteration on graphs
  2013  Bruna et al.    - Spectral graph convolution (full eigendecomp.)
  2015  Defferrard et al. - ChebNet: Chebyshev polynomial spectral filters
  2016  Li et al.       - Gated GNN: GRU-based update functions
  2017  Kipf & Welling  - GCN: simplified spectral GNN for node classification
  2017  Hamilton et al. - GraphSAGE: inductive learning with neighbor sampling
  2017  Gilmer et al.   - MPNN: unified message-passing framework
  2018  Velickovic et al. - GAT: attention-weighted neighborhood aggregation
  2019  Xu et al.       - GIN + WL expressiveness theorem
  2019  Ying et al.     - DiffPool: differentiable hierarchical pooling
  2019  Chiang et al.   - Cluster-GCN: scalable training by graph partitioning
  2020  Rong et al.     - DropEdge: data augmentation against over-smoothing
  2021  Ying et al.     - Graphormer: transformer for graphs (OGB-LSC winner)
  2021  Brody et al.    - GATv2: fixed dynamic attention for GAT
  2022  Rampasek et al. - GPS: general, powerful, scalable graph transformer
  2023  Chen et al.     - Graph Mamba: state-space models on graphs
  2024  Microsoft       - Graph RAG: GNNs for retrieval-augmented generation
  2024  -2026           - LLM+GNN: language models with graph-structured memory

========================================================================
```

---

## 2. Formal Setup: Graphs as Data

### 2.1 Graph with Node and Edge Features

A **featured graph** is a tuple $G = (V, E, X, E_{\text{feat}})$ where:

- $V = \{1, 2, \ldots, n\}$ - the vertex set, $n = |V|$ nodes
- $E \subseteq V \times V$ - the edge set, $m = |E|$ edges
- $X \in \mathbb{R}^{n \times d_v}$ - **node feature matrix**: row $X_{v,:} = \mathbf{x}_v \in \mathbb{R}^{d_v}$ is the feature vector of node $v$
- $E_{\text{feat}} \in \mathbb{R}^{m \times d_e}$ - **edge feature matrix**: row $e_{uv} \in \mathbb{R}^{d_e}$ is the feature of edge $(u,v)$

The adjacency structure is encoded either as a dense matrix $A \in \{0,1\}^{n \times n}$ or (for large sparse graphs) as an **edge list** - a pair of tensors $(\mathbf{s}, \mathbf{t}) \in \mathbb{Z}^m \times \mathbb{Z}^m$ where $\mathbf{s}_k$ is the source and $\mathbf{t}_k$ is the target of edge $k$.

**Examples:**

| Domain | Node features $\mathbf{x}_v$ | Edge features $e_{uv}$ | Task |
|--------|------------------------------|------------------------|------|
| Molecular graph | Atom type (one-hot), charge, hybridization | Bond type (single/double/aromatic), distance | Predict solubility |
| Social network | User demographics, activity history | Interaction frequency, timestamp | Link prediction |
| Knowledge graph | Entity type embedding | Relation type embedding | Triple completion |
| Citation network | Bag-of-words of paper abstract | None (unweighted) | Classify research area |
| Code AST | Token type, identifier embedding | Syntactic relation type | Bug detection |

**Non-examples of graph data:**
- An image stored as a grid: regular structure makes CNN superior; GNNs are overkill and less efficient
- A flat table of independent samples: no edges, no relational structure; standard ML applies
- A time series: sequential structure with canonical ordering; RNN/Transformer preferred unless interactions are truly non-sequential

### 2.2 Learning Tasks on Graphs

GNNs support three levels of prediction:

**Node-level tasks:** predict a label $y_v$ for each node $v$ (or a subset). The GNN produces a node embedding $\mathbf{h}_v \in \mathbb{R}^d$ and a final classifier $\hat{y}_v = f(\mathbf{h}_v)$.

- *Semi-supervised node classification:* most nodes are unlabeled; a small fraction have known labels. The GNN propagates information from labeled to unlabeled nodes through the graph structure. Classic example: Cora/CiteSeer citation networks (Kipf & Welling, 2017).
- *Node regression:* predict a continuous value per node. Example: predicting traffic speed at road intersections.

**Edge-level tasks:** predict a property of a pair of nodes $(u, v)$, whether or not the edge exists.

- *Link prediction:* does edge $(u, v)$ exist? Used in recommendation (predict user-item interaction) and knowledge graph completion (predict missing relation). The GNN produces $\mathbf{h}_u$ and $\mathbf{h}_v$; the prediction is $\hat{y}_{uv} = f(\mathbf{h}_u, \mathbf{h}_v)$ (e.g., inner product or MLP).
- *Edge classification:* predict the type of an existing edge. Example: classify protein-protein interaction type (physical, genetic, etc.).

**Graph-level tasks:** predict a single label for an entire graph $G$.

- *Graph classification:* classify molecular graphs as toxic/non-toxic; classify citation subgraphs by topic. Requires a **readout** function that aggregates all node embeddings into a single graph embedding $\mathbf{h}_G \in \mathbb{R}^d$.
- *Graph regression:* predict a scalar property of the graph (e.g., binding affinity, quantum energy). Most molecular property prediction benchmarks (OGB-molhiv, QM9) are graph regression tasks.

### 2.3 Permutation Invariance and Equivariance

**Definition (Permutation Equivariance).** Let $\Pi_n$ be the group of $n \times n$ permutation matrices. A function $f: \mathbb{R}^{n \times d} \times \{0,1\}^{n \times n} \to \mathbb{R}^{n \times d'}$ is **permutation equivariant** if for every permutation matrix $P \in \Pi_n$:

$$f(PX, PAP^\top) = P \cdot f(X, A)$$

That is, permuting the node ordering in the input permutes the output representations in the same way. **Node-level GNNs must be permutation equivariant**: if we relabel the nodes, the node embeddings should permute consistently.

**Definition (Permutation Invariance).** A function $g: \mathbb{R}^{n \times d} \times \{0,1\}^{n \times n} \to \mathbb{R}$ is **permutation invariant** if for every permutation $P$:

$$g(PX, PAP^\top) = g(X, A)$$

**Graph-level GNNs must be permutation invariant**: the prediction for a graph is the same regardless of how we number its vertices.

**Why this constrains architecture.** Consider a simple approach: concatenate all node features into a vector $[\mathbf{x}_1^\top, \mathbf{x}_2^\top, \ldots, \mathbf{x}_n^\top]^\top$ and pass it through an MLP. This is *not* permutation equivariant - shuffling the nodes changes the input vector and changes the output. Similarly, computing any function of the matrix $A$ that uses specific row/column indices (like the $(1,2)$ entry) is not permutation invariant.

The only permutation-equivariant functions that aggregate neighborhoods are those that apply a fixed function to the *set* of neighbor features - which is precisely what GNN aggregation functions do. Functions of sets must be insensitive to the order of elements; sum, mean, and max all satisfy this; concatenation does not.

**Formal example.** Let $\mathbf{h}_v^{[1]} = \sigma(W \cdot \frac{1}{|\mathcal{N}(v)|}\sum_{u \in \mathcal{N}(v)} \mathbf{x}_u + W_0 \mathbf{x}_v)$ (GCN-style). This is equivariant because $\sum_{u \in \mathcal{N}(v)}$ sums over a set - reordering the neighbors does not change the sum. But $[\mathbf{x}_{u_1}^\top, \mathbf{x}_{u_2}^\top, \ldots]^\top$ for an ordered list of neighbors is *not* equivariant.

### 2.4 Representations: From Single Graph to Batched Graphs

Training GNNs on datasets of multiple small graphs (graph classification, molecular property prediction) requires efficient batching. Unlike image batches where tensors have uniform shape, graphs have variable numbers of nodes and edges.

**The block-diagonal batch trick.** Given graphs $G_1, G_2, \ldots, G_B$ with $n_1, n_2, \ldots, n_B$ nodes, form a single disconnected "super-graph" $G_{\text{batch}}$ with $N = \sum_i n_i$ nodes:

$$A_{\text{batch}} = \begin{pmatrix} A_1 & & \\ & A_2 & \\ & & \ddots \end{pmatrix}, \qquad X_{\text{batch}} = \begin{pmatrix} X_1 \\ X_2 \\ \vdots \end{pmatrix}$$

The adjacency matrix is block-diagonal (no edges between different graphs). Since graphs in the batch are disconnected, message passing does not cross graph boundaries. A batch index vector $\mathbf{b} \in \mathbb{Z}^N$ tracks which graph each node belongs to, enabling graph-level readout by aggregating within each block.

**Virtual node trick.** Adding a "super node" connected to all real nodes in a graph is a common trick to enable long-range information propagation without increasing depth. The virtual node aggregates the entire graph's information in one hop, then broadcasts back. Used in MPNN (Gilmer et al., 2017) and Graphormer (Ying et al., 2021). It also serves as a form of global pooling: $\mathbf{h}_{\text{virtual}} = \operatorname{READOUT}(\{\mathbf{h}_v\})$.

---

## 3. The MPNN Framework

### 3.1 The Gilmer et al. (2017) Formulation

The Message Passing Neural Network (MPNN) framework, introduced by Gilmer et al. (2017) for quantum chemistry property prediction, provides the canonical mathematical abstraction for GNNs. It unifies GCN, GraphSAGE, GAT, GIN, and most other spatial GNN architectures as special cases.

**Algorithm: Message Passing Phase**

For $l = 0, 1, \ldots, L-1$ (layers):

$$\mathbf{m}_v^{[l+1]} = \sum_{u \in \mathcal{N}(v)} M^{[l]}\!\left(\mathbf{h}_v^{[l]}, \mathbf{h}_u^{[l]}, \mathbf{e}_{uv}\right)$$

$$\mathbf{h}_v^{[l+1]} = U^{[l]}\!\left(\mathbf{h}_v^{[l]}, \mathbf{m}_v^{[l+1]}\right)$$

**Readout Phase** (for graph-level tasks):

$$\hat{y}_G = R\!\left(\left\{\mathbf{h}_v^{[L]} : v \in V\right\}\right)$$

Here:
- $M^{[l]}: \mathbb{R}^{d} \times \mathbb{R}^{d} \times \mathbb{R}^{d_e} \to \mathbb{R}^{d'}$ - **message function** at layer $l$; takes the central node's features, a neighbor's features, and the edge features, returns a message vector
- $U^{[l]}: \mathbb{R}^d \times \mathbb{R}^{d'} \to \mathbb{R}^{d''}$ - **update function** at layer $l$; combines the node's current representation with the aggregated message
- $R: 2^{\mathbb{R}^d} \to \mathbb{R}^k$ - **readout function**; maps a set of node representations to a graph-level prediction

The aggregation in the message phase is written as a sum, but can be replaced by any permutation-invariant function (mean, max, attention). The sum formulation is canonical because it is the most expressive (see 7).

**Initialization:** $\mathbf{h}_v^{[0]} = \mathbf{x}_v$ - the initial representation is the input node feature.

**For AI:** The MPNN framework is so general that it also subsumes certain attention mechanisms. The transformer's self-attention layer can be viewed as a fully-connected MPNN (every token attends to every other token) with $M(q_i, k_j, v_j) = \operatorname{softmax}(q_i^\top k_j / \sqrt{d_k}) \cdot v_j$ as the message function.

### 3.2 Message Functions

The message function $M(\mathbf{h}_v, \mathbf{h}_u, \mathbf{e}_{uv})$ determines what information flows along each edge. Several common designs:

**Edge-independent messages.** Many simple GNNs (GCN, GIN) ignore edge features and use:
$$\mathbf{m}_{uv} = M(\mathbf{h}_u) = W \mathbf{h}_u$$
or
$$\mathbf{m}_{uv} = M(\mathbf{h}_u, \mathbf{h}_v) = W_1 \mathbf{h}_u + W_2 \mathbf{h}_v$$

**Edge-conditioned messages.** When edge features are available:
$$\mathbf{m}_{uv} = f_\theta(\mathbf{h}_u, \mathbf{e}_{uv})$$
where $f_\theta$ is an MLP. Used in MPNN for molecular graphs (Gilmer et al., 2017): edge features encode bond type, bond length, and stereochemistry.

**Pair messages.** In models like NNConv, the edge feature defines the transformation matrix:
$$\mathbf{m}_{uv} = \Phi(\mathbf{e}_{uv}) \cdot \mathbf{h}_u$$
where $\Phi: \mathbb{R}^{d_e} \to \mathbb{R}^{d' \times d}$ is an MLP that produces a weight matrix from the edge features. This is extremely expressive but quadratically expensive in $d$.

### 3.3 Aggregation: Permutation-Invariant Pooling

The aggregation function must be permutation invariant over the set $\{\mathbf{m}_{uv} : u \in \mathcal{N}(v)\}$. Three canonical choices and their properties:

**Sum aggregation:**
$$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}$$
Sensitive to neighborhood size - a node with 10 neighbors gets a larger aggregate than a node with 2, even if their neighborhood compositions are identical. This turns out to be a *feature*, not a bug: sum aggregation is the most expressive (7.4). Used in GIN (Xu et al., 2019).

**Mean aggregation:**
$$\mathbf{m}_v = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{uv}$$
Normalizes by neighborhood size. Good when neighborhood size is not informative and you want to compare average neighbor characteristics across nodes of different degrees. Used in GCN (Kipf & Welling, 2017) and GraphSAGE (mean variant).

**Max aggregation:**
$$(\mathbf{m}_v)_k = \max_{u \in \mathcal{N}(v)} (\mathbf{m}_{uv})_k \quad \text{for each dimension } k$$
Detects whether *any* neighbor has feature $k$ above a threshold. Good for detecting the presence of particular node types in the neighborhood. Used in GraphSAGE (max-pool variant).

**Expressiveness hierarchy.** Xu et al. (2019) proved: for injective graph classification, **sum $\succ$ mean = max** in expressive power. Mean cannot distinguish a graph with two nodes each of degree 1 from a graph with four nodes each of degree 2 if neighbor features are identical. Max cannot count copies of identical features. Sum can distinguish both.

**Attention aggregation.** Instead of fixed weights, learn attention coefficients:
$$\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} \alpha_{uv} \cdot \mathbf{m}_{uv}, \qquad \alpha_{uv} = \operatorname{softmax}_u(e_{uv})$$
where $e_{uv}$ is a learned attention score. This is the GAT model (6).

### 3.4 Update Functions

The update function $U(\mathbf{h}_v^{[l]}, \mathbf{m}_v^{[l+1]})$ combines the node's current representation with the aggregated neighborhood message.

**Linear update (GCN):**
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W^{[l]} \mathbf{m}_v^{[l+1]}\right)$$
The current node representation is absorbed into the message (via $\tilde{A} = A + I$ including the self-loop).

**Concatenation update (GraphSAGE):**
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W^{[l]} \left[\mathbf{h}_v^{[l]} \,\|\, \mathbf{m}_v^{[l+1]}\right]\right)$$
Concatenating the node's current representation with the aggregated neighbor message before transforming. Preserves the node's own information explicitly.

**GRU update (Gated GNN, Li et al. 2016):**
$$\mathbf{h}_v^{[l+1]} = \operatorname{GRU}\!\left(\mathbf{h}_v^{[l]}, \mathbf{m}_v^{[l+1]}\right)$$
The GRU's gating mechanism adaptively controls how much of the current representation to retain vs. replace with the new message. More expressive than linear update but more parameters.

**Residual update:**
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W^{[l]} \mathbf{m}_v^{[l+1]}\right) + \mathbf{h}_v^{[l]}$$
Skip connections (as in ResNet) preserve gradient flow in deep GNNs and mitigate over-smoothing. Used in GCNII (Chen et al., 2020) for training 64-layer GCNs successfully.

### 3.5 Readout for Graph-Level Tasks

For graph classification and regression, the node representations $\{\mathbf{h}_v^{[L]}\}$ must be aggregated into a single graph embedding $\mathbf{h}_G \in \mathbb{R}^d$.

**Global pooling:**
$$\mathbf{h}_G = \operatorname{READOUT}\!\left(\left\{\mathbf{h}_v^{[L]} : v \in V\right\}\right)$$
Simple choices: $\sum_v \mathbf{h}_v$, $\frac{1}{n}\sum_v \mathbf{h}_v$, $\max_v \mathbf{h}_v$ (element-wise). Sum is most expressive; mean normalizes for graph size.

**Hierarchical pooling.** Rather than a single pooling step at the end, hierarchical GNNs (DiffPool, SAGPool) alternate GNN layers with coarsening steps that reduce the number of nodes. This mirrors how image CNNs alternate convolution with spatial downsampling.

**Jumping Knowledge (Xu et al., 2018).** Rather than using only the final layer's representations, JK-networks concatenate representations from all layers:
$$\mathbf{h}_v^{\text{final}} = \operatorname{COMBINE}\!\left(\mathbf{h}_v^{[0]}, \mathbf{h}_v^{[1]}, \ldots, \mathbf{h}_v^{[L]}\right)$$
This allows the readout to leverage both local (shallow layers) and global (deep layers) structural information, and empirically mitigates over-smoothing.

**Set2Set (Vinyals et al., 2016).** A more powerful readout using an LSTM-based attention over the node set, producing an order-invariant graph embedding that depends on all node representations through multiple rounds of attention. More expressive than simple global pooling but $O(n)$ sequential steps.

### 3.6 MPNN Instances: Unifying GCN, GraphSAGE, GAT, GIN

The following table shows how each major GNN architecture is a special case of the MPNN framework:

```
MPNN UNIFICATION TABLE
========================================================================

  Model        Message M(h_v, h_u, e_uv)    Aggregation   Update U(h_v, m_v)
  ---------------------------------------------------------------------
  GCN          W*h_u                         Mean (norm.)  \sigma(W*m_v)
  GraphSAGE    W_1*h_u                       Mean/Max/Sum  \sigma(W*[h_v || m_v])
  GAT          \alpha_uv * W*h_u                 Attention     \sigma(W*m_v)
               (\alpha_uv learned from h_v, h_u)
  GIN          h_u                           Sum           MLP((1+\epsilon)*h_v + m_v)
  MPNN-Gilmer  f_(h_u, e_uv)               Sum           GRU(h_v, m_v)
  ---------------------------------------------------------------------
  Key insight: choice of M, aggregation, U determines expressiveness
  and scalability. Sum agg. + injective MLP = most expressive (7)

========================================================================
```

---

## 4. Graph Convolutional Networks (GCN)

### 4.1 Spectral Foundation Recap

> **Recall from 11-04 7.5:** The GCN layer is derived as a first-order Chebyshev polynomial approximation to spectral graph convolution. Starting from the spectral filter $g_\theta(\Lambda) = \theta_0 I + \theta_1(\Lambda - I)$ applied to the normalized Laplacian $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$ (with eigenvalues in $[0,2]$), constraining $\theta = \theta_0 = -\theta_1$ for a single parameter per layer, and applying the renormalization trick $\tilde{A} = A + I$, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$, yields the GCN propagation rule. Full derivation: [11-04 Spectral Graph Theory 7.5](../04-Spectral-Graph-Theory/notes.md#75-from-chebyshev-to-gcn).

The key takeaway is that the GCN layer is a principled spectral filter, not an ad hoc heuristic. The choice of $\tilde{A}$ (adding self-loops) and $\tilde{D}^{-1/2}(\cdot)\tilde{D}^{-1/2}$ (symmetric normalization) both arise from the spectral derivation and can now be given spatial interpretations.

### 4.2 The GCN Layer

The GCN propagation rule for all nodes simultaneously (matrix form):

$$H^{[l+1]} = \sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{[l]} W^{[l]}\right)$$

where:
- $\tilde{A} = A + I_n$ - adjacency matrix with added self-loops
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ - degree matrix of $\tilde{A}$
- $H^{[l]} \in \mathbb{R}^{n \times d_l}$ - node feature matrix at layer $l$; $H^{[0]} = X$
- $W^{[l]} \in \mathbb{R}^{d_l \times d_{l+1}}$ - trainable weight matrix at layer $l$
- $\sigma$ - nonlinear activation (ReLU for hidden layers, softmax for output in classification)

**Node-level form.** For a single node $v$:
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W^{[l]\top} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{h}_u^{[l]}\right)$$

where $\tilde{d}_v = \tilde{D}_{vv} = d_v + 1$ (degree including self-loop). Each neighbor $u$'s contribution is weighted by $\frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}}$ - the geometric mean of the (augmented) degrees. This normalization prevents high-degree nodes from dominating and ensures the propagation matrix has spectral radius \leq 1.

**The propagation matrix.** Define $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$. This is the normalized adjacency of $\tilde{A}$. Its eigenvalues lie in $[-1, 1]$ (since $\tilde{A}$ is symmetric, non-negative). A key property:

$$\hat{A}^k_{ij} = \sum_{\text{paths of length }k: i \to j} \prod_{e \in \text{path}} \frac{1}{\sqrt{\tilde{d}_{v_s} \tilde{d}_{v_t}}}$$

After $L$ GCN layers, node $v$'s representation is a learned function of the (weighted) $L$-hop neighborhood - exactly the receptive field intuition from 1.2.

### 4.3 Matrix View: Propagation Rule

The GCN can be understood as a **learned diffusion process**. Define $S = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ as the **diffusion operator** (also called the normalized propagation matrix). Then:

$$H^{[l+1]} = \sigma\left(S H^{[l]} W^{[l]}\right)$$

- **$S H^{[l]}$** - neighborhood aggregation: each node's features become a weighted average of its neighbors' features. This is a graph smoothing step - it reduces variation across connected nodes.
- **$(\cdot) W^{[l]}$** - feature transformation: a learnable linear map on the $d_l$-dimensional feature space.
- **$\sigma(\cdot)$** - nonlinearity: introduces expressive power.

Repeating this $L$ times gives $H^{[L]} = \sigma(S \cdot \sigma(S \cdots \sigma(S X W^{[0]}) \cdots W^{[L-2]}) W^{[L-1]})$.

**Connection to heat diffusion.** The continuous heat equation on a graph is $\frac{\partial \mathbf{h}}{\partial t} = -L\mathbf{h}$, with solution $\mathbf{h}(t) = e^{-tL}\mathbf{h}(0)$. The GCN step $S H$ approximates one discrete step of this diffusion. Deep GCNs approximate long-time diffusion, which - as we will see in 8 - causes all node representations to converge to the same value (over-smoothing).

### 4.4 Semi-Supervised Node Classification

The original application of GCN (Kipf & Welling, 2017) is **semi-supervised node classification**: given a graph with $n$ nodes, $n_L \ll n$ labeled and $n_U = n - n_L$ unlabeled, train the GCN to predict class labels for all nodes.

**Setup:**
- Two-layer GCN: $\hat{Y} = \operatorname{softmax}\!\left(S \cdot \operatorname{ReLU}(S X W^{[0]}) \cdot W^{[1]}\right)$
- Loss: cross-entropy over labeled nodes only: $\mathcal{L} = -\sum_{v \in V_L} \sum_c Y_{vc} \log \hat{Y}_{vc}$
- The GCN propagates label information from labeled to unlabeled nodes through the graph structure (similar to label propagation, but learned)

**Benchmark results (Cora citation network, 140 labeled / 2708 total):**

| Method | Accuracy |
|--------|----------|
| DeepWalk + SVM | 67.2% |
| Label Propagation | 68.0% |
| Planetoid (Yang et al. 2016) | 75.7% |
| GCN (Kipf & Welling, 2017) | 81.5% |
| GAT (Velickovic et al., 2018) | 83.0% |
| GCNII (Chen et al., 2020, 64 layers) | 85.5% |

The GCN's success on this benchmark established GNNs as the go-to method for graph-structured learning and spawned the entire field.

**Why it works.** The graph encodes a "homophily" assumption: connected nodes tend to have the same class. In citation networks, papers citing each other tend to be in the same research area. The GCN leverages this assumption by smoothing features across edges, effectively propagating class information from labeled nodes to their unlabeled neighbors.

### 4.5 Strengths and Weaknesses of GCN

**Strengths:**
- Simple to implement (two matrix multiplications per layer)
- Efficient: $O(m \cdot d)$ per layer for sparse graphs
- Well-understood theoretically: spectral derivation, convergence analysis
- Strong baselines for semi-supervised node classification
- Easily extended with residual connections, BatchNorm, dropout

**Weaknesses:**
- **Transductive:** the normalization $\hat{A}$ is computed for the entire graph at training time. Adding a new node requires recomputing $\hat{A}$ and retraining - the GCN cannot generalize to unseen nodes
- **Fixed symmetric aggregation:** every neighbor is weighted by the same geometric mean of degrees. High-degree hub nodes get down-weighted regardless of their relevance
- **No edge features:** the standard GCN layer has no mechanism to incorporate edge attributes
- **Expressiveness limited to 1-WL:** cannot distinguish non-isomorphic graphs that the WL test cannot distinguish (7)
- **Over-smoothing at depth:** after many layers, all node representations converge (8)

---

## 5. GraphSAGE: Inductive Learning on Large Graphs

### 5.1 The Inductive Learning Problem

GCN is **transductive**: it operates on a fixed graph seen during training. The normalized adjacency $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ encodes the training graph's structure. When new nodes arrive (a new user on a social platform, a new molecule to screen), the GCN cannot produce representations without recomputing the full graph and rerunning the entire network.

This is impractical for large, dynamic graphs. Pinterest has billions of users; new pins are added every second. Training a fresh GCN daily is infeasible.

**Inductive learning** solves this: learn an **aggregation function** that can be applied to the neighborhood of *any* node, including nodes unseen during training. If the function is parameterized correctly, it generalizes: apply it to the neighborhood of a new node and get a useful embedding without retraining.

GraphSAGE (Hamilton, Ying & Leskovec, 2017) - **SA**mple and aggre**G**at**E** - is the canonical inductive GNN framework.

### 5.2 GraphSAGE Framework

**Key idea:** instead of using the entire neighborhood of a node (which can be millions of neighbors for hub nodes), *sample* a fixed-size subset $\mathcal{S}_v^{[l]} \subseteq \mathcal{N}(v)$ at each layer. Then aggregate only over the sampled set. The parameters of the aggregation function are shared across all nodes and all graphs.

**Algorithm (2-layer GraphSAGE):**

For each node $v$ in the target set (e.g., a mini-batch of nodes):

1. Sample neighbors: $\mathcal{S}_v^{[1]} \sim \mathcal{U}(\mathcal{N}(v), S_1)$ - sample $S_1$ neighbors uniformly
2. For each sampled neighbor $u \in \mathcal{S}_v^{[1]}$, sample their neighbors: $\mathcal{S}_u^{[2]} \sim \mathcal{U}(\mathcal{N}(u), S_2)$
3. Compute depth-2 embeddings bottom-up:
   - $\mathbf{h}_u^{[1]} = \sigma\!\left(W^{[1]} \cdot \operatorname{AGGREGATE}\!\left(\left\{\mathbf{x}_w : w \in \mathcal{S}_u^{[2]}\right\}\right)\right)$ for all $u \in \mathcal{S}_v^{[1]}$
   - $\mathbf{h}_v^{[2]} = \sigma\!\left(W^{[2]} \cdot \left[\mathbf{h}_v^{[1]} \,\|\, \operatorname{AGGREGATE}\!\left(\left\{\mathbf{h}_u^{[1]} : u \in \mathcal{S}_v^{[1]}\right\}\right)\right]\right)$
4. Normalize: $\mathbf{h}_v^{[2]} \leftarrow \mathbf{h}_v^{[2]} / \lVert\mathbf{h}_v^{[2]}\rVert_2$

**Inductive inference:** for a new node $v'$ with known features and edges (to training-time nodes), apply the same aggregation functions with the learned $W^{[1]}, W^{[2]}$ - no retraining needed.

**Sampling depth.** A $L$-layer GraphSAGE with neighborhood sizes $(S_1, S_2, \ldots, S_L)$ touches at most $\prod_{l=1}^L S_l$ nodes per target node. With $(S_1, S_2) = (25, 10)$, at most 250 nodes are accessed per target node, regardless of graph size. This makes training on massive graphs feasible.

### 5.3 Aggregation Strategies

GraphSAGE defines three aggregation variants with different expressiveness-efficiency tradeoffs:

**Mean aggregator:**
$$\mathbf{m}_v^{[l]} = \frac{1}{|\mathcal{S}_v^{[l]}|} \sum_{u \in \mathcal{S}_v^{[l]}} \mathbf{h}_u^{[l-1]}$$
$$\mathbf{h}_v^{[l]} = \sigma\!\left(W^{[l]} \cdot \left[\mathbf{h}_v^{[l-1]} \,\|\, \mathbf{m}_v^{[l]}\right]\right)$$

Similar to GCN but with uniform weights (no degree normalization). Concatenation rather than addition preserves the distinction between self and neighborhood.

**LSTM aggregator:**
$$\mathbf{m}_v^{[l]} = \operatorname{LSTM}\!\left(\left[\mathbf{h}_{\pi(u)}^{[l-1]}\right]_{u \in \mathcal{S}_v^{[l]}}\right)$$

Apply an LSTM over a *random permutation* $\pi$ of the sampled neighbors. The LSTM can model complex interactions between neighbors (e.g., in a sequence-like order), but the random permutation breaks the theoretical permutation invariance. In practice, the LSTM often performs best on benchmark tasks due to its representational power.

**Max-pool aggregator:**
$$\mathbf{m}_v^{[l]} = \max_{u \in \mathcal{S}_v^{[l]}} \sigma\!\left(W_{\text{pool}} \mathbf{h}_u^{[l-1]} + \mathbf{b}\right)$$
Apply a learnable transformation to each neighbor's representation, then take the element-wise max. Detects the presence of specific feature patterns in the neighborhood. Often the best choice for tasks where the presence of a rare feature type is informative.

**Choosing an aggregator.** Empirically: LSTM aggregator performs best on benchmark accuracy but is not permutation invariant. Mean aggregator is permutation invariant and performs well on inductive tasks. Max-pool aggregator is best at detecting rare structural patterns. Practitioners typically tune on validation data.

### 5.4 Unsupervised Training and Link Prediction

GraphSAGE can be trained without labels using a link prediction objective:

$$\mathcal{L}(u, v) = -\log \sigma\!\left(\mathbf{h}_u^\top \mathbf{h}_v\right) - Q \cdot \mathbb{E}_{v_n \sim P_n(V)}\!\left[\log\!\left(1 - \sigma\!\left(\mathbf{h}_u^\top \mathbf{h}_{v_n}\right)\right)\right]$$

where $(u, v)$ is a positive pair (connected in the graph), $v_n$ is a negative sample (random non-neighbor), $Q$ is the number of negative samples, $P_n(V)$ is the negative sampling distribution (typically uniform), and $\sigma$ is the sigmoid function.

**Intuition:** nodes that co-occur in short random walks should have similar embeddings; nodes that are far apart should have dissimilar embeddings. This is the DeepWalk objective extended to learned GNN embeddings.

**For AI:** unsupervised GraphSAGE embeddings are used in PinSage (Pinterest's recommendation system) to generate embeddings for pins (images with descriptions) in a bipartite user-pin-board graph. These embeddings power Pinterest's "more like this" recommendations, serving 3+ billion monthly active users.

### 5.5 Scaling to Billion-Node Graphs

**PinSage (Ying et al., 2018)** adapted GraphSAGE to Pinterest's graph with 3 billion nodes and 18 billion edges - the largest graph neural network deployment at the time.

Key engineering innovations:
1. **Importance-based sampling:** instead of uniform random sampling, sample neighbors proportional to their random walk visit frequency from the target node. This gives more weight to structurally important neighbors and reduces variance.
2. **Producer-consumer minibatch pipeline:** precompute the node neighborhoods offline in Spark, stream them to GPU workers as training data. Decouples graph traversal (CPU-bound) from gradient computation (GPU-bound).
3. **Hard negative mining:** the standard negative sampling produces easy negatives (random pins). Hard negatives are pins that are visually similar but semantically different - they force the model to learn finer-grained distinctions.
4. **Curriculum learning:** train first with easy negatives, then add hard negatives progressively.

**Deployment results:** 150% improvement in head-to-head pin recommendation quality over previous collaborative filtering system, with recommendations updated daily across 3+ billion items.

---

## 6. Graph Attention Networks (GAT)

### 6.1 The Attention Mechanism on Graphs

GCN's aggregation weights $\frac{1}{\sqrt{\tilde{d}_u \tilde{d}_v}}$ are fixed functions of node degrees - they depend only on the graph structure, not on node features. Two neighbors with identical degree contribute identically regardless of their relevance to the central node's representation.

Graph Attention Networks (Velickovic et al., 2018) replace fixed weights with **learned, data-dependent attention coefficients**. For each edge $(u, v)$, the attention coefficient $\alpha_{uv}$ is computed as a function of both nodes' features, allowing the network to focus on the most relevant neighbors.

This mirrors the transformer's self-attention, but with attention constrained to the graph's edge structure: node $v$ only attends to its actual neighbors $\mathcal{N}(v)$, not to all nodes (which would be $O(n^2)$). The graph acts as a **structural prior** on the attention pattern.

### 6.2 GAT Layer: Attention Coefficients

**Step 1: Linear transformation.** Apply a shared weight matrix $W \in \mathbb{R}^{d' \times d}$ to transform all node features:
$$\mathbf{z}_v = W\mathbf{h}_v^{[l]} \quad \forall v \in V$$

**Step 2: Attention scores.** For each edge $(u, v)$ (including self-loops), compute:
$$e_{uv} = \operatorname{LeakyReLU}\!\left(\mathbf{a}^\top \left[\mathbf{z}_v \,\|\, \mathbf{z}_u\right]\right)$$

where $\mathbf{a} \in \mathbb{R}^{2d'}$ is a learnable attention vector, and $[\cdot \| \cdot]$ denotes concatenation. The raw attention score $e_{uv}$ measures the "relevance" of node $u$ to node $v$. LeakyReLU (with negative slope $\alpha = 0.2$) allows gradient flow for zero scores.

**Step 3: Normalize.** Apply softmax over the neighborhood to get normalized coefficients:
$$\alpha_{uv} = \frac{\exp(e_{uv})}{\sum_{k \in \mathcal{N}(v) \cup \{v\}} \exp(e_{kv})}$$

Note: $\alpha_{uv} \geq 0$ and $\sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{uv} = 1$.

**Step 4: Aggregation.** Compute the updated representation:
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \alpha_{uv} \cdot \mathbf{z}_u\right)$$

**MPNN view:** the message is $M(\mathbf{h}_v, \mathbf{h}_u) = \alpha_{uv} \cdot W\mathbf{h}_u$ and the aggregation is sum.

### 6.3 Multi-Head Attention for Graphs

Single-head attention can be unstable and may attend to a limited range of features. Multi-head attention stabilizes training and allows the model to jointly attend to different aspects of neighbors.

**Multi-head GAT layer.** Run $K$ independent attention mechanisms in parallel, each with its own parameters $(W^{(k)}, \mathbf{a}^{(k)})$:
$$\mathbf{h}_v^{[l+1]} = \|_{k=1}^K \sigma\!\left(\sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(k)} W^{(k)} \mathbf{h}_u^{[l]}\right)$$

where $\|$ denotes concatenation. The output dimension is $K \cdot d'$.

**For the final layer** (before a classification head), averaging is often preferred over concatenation:
$$\mathbf{h}_v^{[L]} = \sigma\!\left(\frac{1}{K} \sum_{k=1}^K \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(k)} W^{(k)} \mathbf{h}_u^{[L-1]}\right)$$

**Complexity.** A $K$-head GAT layer costs $O(K \cdot m \cdot d)$ for attention computation ($m$ edges, $d$ features) and $O(K \cdot n \cdot d \cdot d')$ for the linear transformations. For dense attention (fully-connected graph), this is $O(K \cdot n^2 \cdot d)$ - the same scaling as transformer attention.

### 6.4 GATv2: Fixing the Static Attention Problem

Brody, Alon & Yahav (2022) identified a fundamental limitation of the original GAT: its attention is **static** - the ranking of a neighbor's importance does not depend on the query node.

**The problem.** In GAT, the attention score is:
$$e_{uv} = \mathbf{a}^\top [\mathbf{z}_v \| \mathbf{z}_u] = \mathbf{a}_{\text{left}}^\top \mathbf{z}_v + \mathbf{a}_{\text{right}}^\top \mathbf{z}_u$$

Since this is a sum of two terms - one depending only on $v$, one depending only on $u$ - the ranking of neighbors $u_1$ vs $u_2$ by their score $e_{u_1 v}$ vs $e_{u_2 v}$ is independent of $\mathbf{z}_v$. The "most important neighbor" is the same regardless of which node $v$ is doing the asking. This is "static attention."

**Dynamic attention (GATv2).** Fix by swapping the order of the linear transformation and concatenation:
$$e_{uv} = \mathbf{a}^\top \operatorname{LeakyReLU}\!\left(W \cdot \left[\mathbf{h}_v \| \mathbf{h}_u\right]\right)$$

Now the LeakyReLU nonlinearity is applied *before* the dot product with $\mathbf{a}$, creating genuine interaction between $\mathbf{h}_v$ and $\mathbf{h}_u$. The ranking of $u_1$ vs $u_2$ can now depend on $v$ - the attention is dynamic.

**Formal claim (Brody et al., 2022):** GAT is a strictly less powerful attention function than GATv2. There exist graphs where GAT's attention collapses to uniform weighting while GATv2's attention correctly assigns non-uniform importance. GATv2 matches or exceeds GAT on all standard benchmarks.

### 6.5 Attention Sparsity and Interpretability

A commonly cited advantage of GAT over GCN is interpretability: the attention coefficients $\alpha_{uv}$ can be read as edge importance scores. Edges with high $\alpha_{uv}$ are "important" for node $v$'s representation; edges with near-zero $\alpha_{uv}$ are "ignored."

**Limitations of this interpretation.** Jain & Wallace (2019) and Wiegreffe & Pinter (2019) showed for NLP models that attention weights do not reliably indicate feature importance. Similar caveats apply to GAT:
1. Attention weights are **post-softmax** and sum to 1; a weight of $0.9$ out of 3 neighbors is very different from $0.9$ out of 300 neighbors.
2. The softmax creates competition between neighbors - a "dominant" neighbor may capture high attention simply by having a large dot product with $\mathbf{a}$, not because it is semantically important.
3. Different attention heads may attend to completely different structural patterns, and the meaning of each head is not determined a priori.

Despite these caveats, in practice, GAT attention patterns do reveal meaningful structural patterns for molecular and knowledge graph tasks, especially when validated against domain knowledge.

---

## 7. Expressiveness and the Weisfeiler-Leman Test

### 7.1 The Graph Isomorphism Problem

**Definition (Graph Isomorphism).** Two graphs $G = (V_G, E_G)$ and $H = (V_H, E_H)$ are **isomorphic** (written $G \cong H$) if there exists a bijection $\phi: V_G \to V_H$ such that $(u, v) \in E_G \iff (\phi(u), \phi(v)) \in E_H$.

Two isomorphic graphs are structurally identical - they differ only in how the vertices are labeled. For any function of graph structure (predicting molecular properties, classifying graph type), the function must produce the same output on isomorphic graphs. This is exactly the permutation invariance requirement from 2.3.

The graph isomorphism (GI) problem - determining whether two given graphs are isomorphic - is one of the few natural problems not known to be in P or NP-complete. For practical purposes (GNN expressiveness), we ask a weaker question: can a GNN **distinguish** two non-isomorphic graphs by assigning them different representations?

**Non-isomorphic graphs that look similar.** Consider:

```
Graph 1: Two disjoint triangles (C_3 \cup C_3)
         o-o   o-o
          \ /   \ /
           o     o

Graph 2: One 6-cycle (C_6)
         o-o-o
         |     |
         o-o-o
```

Both graphs have 6 nodes, each of degree 2 - same degree sequence. Are they isomorphic? No: $C_3 \cup C_3$ has two triangles; $C_6$ has no triangles. A GNN should assign them different representations. Can it?

### 7.2 1-WL and Color Refinement

The **Weisfeiler-Leman (WL) graph isomorphism test** (Weisfeiler & Leman, 1968) is an efficient algorithm for testing graph isomorphism. While not complete (there exist non-isomorphic pairs it cannot distinguish), it correctly identifies isomorphism for most practical graphs.

**1-WL Color Refinement Algorithm:**

Initialize: assign all nodes the same color $c_v^{(0)} = 1$ (or, for node-attributed graphs, $c_v^{(0)} = \operatorname{hash}(\mathbf{x}_v)$).

Iterate for $t = 0, 1, 2, \ldots$ until convergence:
$$c_v^{(t+1)} = \operatorname{HASH}\!\left(c_v^{(t)},\, \{\{c_u^{(t)} : u \in \mathcal{N}(v)\}\}\right)$$

where $\{\{\cdot\}\}$ denotes a **multiset** (duplicate elements are preserved) and HASH is an injective hash function on (color, multiset of colors) pairs.

**Decision:** If at any iteration, graphs $G$ and $H$ have different multisets of node colors, declare them non-isomorphic. If the algorithm stabilizes with the same color multisets, declare them (possibly) isomorphic.

**Convergence:** The algorithm stabilizes in at most $n$ iterations (when no color changes occur). This gives $O(n^2 \log n)$ time complexity.

**Example.** For $C_3 \cup C_3$ vs $C_6$ with no node attributes:
- $t=0$: all nodes color $c=1$ - same for both graphs
- $t=1$: $c_v^{(1)} = \operatorname{HASH}(1, \{1, 1\}) = $ same for all nodes (all have degree 2 with same-colored neighbors) - still same!
- Algorithm stabilizes: **1-WL cannot distinguish $C_3 \cup C_3$ from $C_6$** - both are "two disjoint 3-regular rings."

This is a fundamental limitation: 1-WL cannot detect triangles (3-cycles), so neither can any MPNN (Theorem below).

### 7.3 The GNN Expressiveness Theorem

**Theorem (Xu et al., 2019).** Let $\mathcal{A}$ be any MPNN with a fixed number of layers $L$ and countably many colors (feature values). Then:

1. If $\mathcal{A}$ assigns different representations to graphs $G \not\cong H$, then 1-WL also distinguishes $G$ and $H$ (in $\leq L$ iterations).
2. For any pair $(G, H)$ that 1-WL distinguishes, there exists an MPNN $\mathcal{A}$ that also distinguishes them.

**Corollary.** The discriminative power of any MPNN is bounded above by 1-WL. Conversely, the most expressive MPNN achieves exactly the discriminative power of 1-WL.

**Proof sketch (upper bound).** At each MPNN layer, node $v$'s representation is a function of:
$(c_v^{[l]}, \{\{c_u^{[l]} : u \in \mathcal{N}(v)\}\})$. This is precisely what 1-WL computes. If the aggregation + update function is injective (different inputs -> different outputs), the MPNN exactly simulates 1-WL. If it is not injective (e.g., mean aggregation), it may collapse distinct multisets to the same representation, making it strictly weaker than 1-WL.

**Implications:**
- MPNNs with mean or max aggregation are **strictly weaker than 1-WL** - they collapse some distinct multisets
- MPNNs with sum aggregation and injective MLP update are **exactly 1-WL** - GIN achieves this
- No MPNN (regardless of architecture) can distinguish graphs that 1-WL cannot distinguish (e.g., $C_3 \cup C_3$ vs $C_6$, or regular graphs of the same degree)
- Expressiveness limitations are fundamental, not a matter of training or capacity

### 7.4 GIN: The Most Expressive 1-WL GNN

**Graph Isomorphism Network (Xu et al., 2019).** To achieve 1-WL expressiveness, the aggregation function must be injective on multisets. Xu et al. prove:

**Lemma.** Any injective function on multisets over a countable universe can be expressed as:
$$\mathbf{h} = \varphi\!\left(\sum_{u \in \mathcal{S}} f(\mathbf{h}_u)\right)$$

for some functions $\varphi, f$. In other words, **sum aggregation with an MLP is a universal approximator for injective multiset functions**.

**GIN layer:**
$$\mathbf{h}_v^{[l+1]} = \operatorname{MLP}^{[l]}\!\left((1 + \varepsilon^{[l]}) \cdot \mathbf{h}_v^{[l]} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{[l]}\right)$$

where $\varepsilon^{[l]}$ is either learned or set to $0$. The $(1+\varepsilon)$ term allows the network to distinguish the central node from its neighbors (otherwise, a node with features $\mathbf{h}$ and a neighbor also with features $\mathbf{h}$ would be aggregated indistinguishably from the node having two neighbors with features $\mathbf{h}/2$ each under mean aggregation).

**Why mean and max fail:**

| Aggregation | Counterexample multisets | Behavior |
|-------------|--------------------------|----------|
| Mean | $\{1, 1\}$ vs $\{1, 1, 1\}$ | Both give mean = 1 - indistinguishable |
| Max | $\{1, 2\}$ vs $\{2\}$ | Both give max = 2 - indistinguishable |
| Sum | $\{1, 1\}$ vs $\{1, 1, 1\}$ | Sums 2 vs 3 - distinguishable |

**GIN for graph classification.** Use sum readout at each layer and combine across layers:
$$\mathbf{h}_G = \operatorname{CONCAT}\!\left(\operatorname{READOUT}^{[l]}\!\left(\left\{\mathbf{h}_v^{[l]}\right\}\right) : l = 0, 1, \ldots, L\right)$$

This jumping-knowledge-style readout captures structural patterns at multiple scales.

### 7.5 Beyond 1-WL: Higher-Order GNNs

1-WL's limitations (e.g., failing to count triangles, failing on regular graphs) motivate higher-order extensions.

**$k$-WL test.** Instead of coloring individual nodes, color $k$-tuples of nodes $(v_1, \ldots, v_k)$. The refinement rule considers the colors of all $k$-tuples that differ in exactly one position. $k$-WL is strictly more powerful than $(k-1)$-WL for all $k \geq 2$.

**$k$-GNN (Morris et al., 2019).** Implement $k$-WL as a GNN by passing messages between $k$-tuples. The cost is $O(n^k)$ - prohibitive for $k \geq 3$ on large graphs.

**NGNN and subgraph GNNs.** A more practical approach: instead of node-level MPNNs, run MPNNs on subgraphs induced by $k$-hop neighborhoods, ego-graphs, or sampled subgraphs. These can detect cycles, cliques, and other motifs that 1-WL misses. Examples: NGNN (Zhang & Li, 2021), OSAN (Zhao et al., 2022).

**Practical trade-off:**

| Method | Expressiveness | Complexity | Practical use |
|--------|----------------|------------|---------------|
| 1-WL MPNN (GIN) | 1-WL | $O(m \cdot d)$ | Standard; most applications |
| Subgraph GNNs | $> $ 1-WL | $O(n \cdot m \cdot d)$ | Medium graphs, chemistry |
| $k$-GNN ($k=3$) | 3-WL | $O(n^3 \cdot d)$ | Small graphs only |
| Random features | Universal | $O(m \cdot d)$ | Simple, effective in practice |

### 7.6 Structural Features and Distance Encoding

A practical alternative to higher-order GNNs: **augment node features with handcrafted structural descriptors** that help the GNN detect patterns it otherwise cannot.

**Random Walk Structural Encoding (RWSE).** For each node $v$, compute the landing probability of a random walk of length $k$ starting and ending at $v$:
$$p_v^{(k)} = [A^k]_{vv} / d_v$$

The vector $[p_v^{(1)}, p_v^{(2)}, \ldots, p_v^{(K)}]$ encodes the local loop structure (triangles, 4-cycles, etc.). Used in GPS (Rampasek et al., 2022) and Graphormer.

**Laplacian Positional Encoding (LapPE).** Use the first $k$ eigenvectors of the normalized Laplacian as node features: $\mathbf{x}_v \leftarrow [\mathbf{x}_v \| \mathbf{u}_1(v), \ldots, \mathbf{u}_k(v)]$. (Reviewed in [11-04 9.5](../04-Spectral-Graph-Theory/notes.md#95-spectral-positional-encodings-for-transformers) - see there for the sign invariance issue and fix.)

**Degree features.** Simply appending the node's degree $d_v$ as a feature allows the GNN to distinguish nodes of different degree even when 1-WL would assign them the same color (since 1-WL with uniform initial colors cannot use degree beyond what the refinement computes). More sophisticated: distance encoding (Li et al., 2020) appends the shortest-path distances from a node to a set of anchor nodes.

---

## 8. Over-Smoothing, Over-Squashing, and Depth

### 8.1 Over-Smoothing: Formal Analysis

A 2-layer GCN typically outperforms a 6-layer GCN on standard node classification benchmarks. This seems paradoxical: more layers should give access to more of the graph. The reason is **over-smoothing**: as depth increases, all node representations converge to the same vector, destroying the discriminative information needed for classification.

**Formal statement (Li et al., 2018).** Let $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ be the GCN propagation matrix (assume $\tilde{A} = A + I$ for simplicity). Then:

$$\hat{A}^k \to \boldsymbol{\pi} \mathbf{1}^\top \quad \text{as } k \to \infty$$

where $\boldsymbol{\pi} \in \mathbb{R}^n$ is the stationary distribution of the random walk on $\tilde{A}$: $\pi_v = \tilde{d}_v / (2m + n)$. That is, the $k$-step propagation from any starting point converges to the stationary distribution regardless of initial conditions.

**Consequence.** For a linear GCN (no nonlinearities), $H^{[L]} = \hat{A}^L X W^{(0)} W^{(1)} \cdots W^{(L-1)}$. As $L \to \infty$:
$$H^{[L]}_{v,:} \to \pi_v \cdot \mathbf{1}^\top X W \propto \text{const} \times (\text{column average of } XW)$$

All nodes collapse to a constant multiple of the graph-wide feature average. For connected graphs, $H^{[L]}$ converges to a rank-1 matrix - all node representations are proportional to $\boldsymbol{\pi}$.

With nonlinearities (ReLU), convergence is not exact, but the trend holds: after 6-8 layers, node representations on typical graphs (small-world, power-law degree distribution) are nearly identical.

**MADGap metric (Chen et al., 2020).** Mean Average Distance (MAD) measures pairwise distances between node representations. MADGap = MAD(between-class) - MAD(within-class). For a good classifier, MADGap should be large (between-class distances large, within-class distances small). Over-smoothing drives MAD -> 0, collapsing MADGap.

### 8.2 Information-Theoretic View via Dirichlet Energy

The **Dirichlet energy** of the node representation matrix $H \in \mathbb{R}^{n \times d}$ measures total variation across edges:

$$E(H) = \sum_{(i,j) \in E} \lVert\mathbf{h}_i - \mathbf{h}_j\rVert_2^2 = \operatorname{tr}(H^\top L H)$$

where $L = D - A$ is the unnormalized Laplacian.

**Properties:**
- $E(H) = 0 \iff \mathbf{h}_i = \mathbf{h}_j$ for all connected $i, j$ (on a connected graph: all nodes identical)
- $E(H)$ is large when adjacent nodes have different representations - high "frequency content"
- The GCN smoothing step reduces $E(H)$: $E(SH) \leq E(H)$ where $S = \hat{A}$ (since $\hat{A}$ is the low-pass graph filter)

**Over-smoothing = energy dissipation.** Each GCN layer reduces $E(H)$. After $L$ layers:
$$E(H^{[L]}) \leq \lambda_{\max}(S)^{2L} \cdot E(H^{[0]})$$

Since $\lambda_{\max}(S) \leq 1$ for symmetric normalized $\hat{A}$ with self-loops, the energy decays geometrically. The rate depends on $\lambda_2(L)$ (the Fiedler value): graphs with large spectral gap (expanders) over-smooth faster than graphs with small spectral gap (clusters).

**Practical implication.** For clustered graphs (molecular graphs, citation networks with strong community structure), over-smoothing is slow - one can use deeper GNNs. For expander-like graphs (dense social networks, random graphs), over-smoothing is fast - 2-4 layers is optimal.

### 8.3 Over-Squashing: Bottleneck Nodes

Over-squashing is a distinct (and more subtle) problem: information from distant nodes is **exponentially compressed** as it flows through bottleneck edges, preventing long-range interactions from influencing predictions.

**Jacobian analysis (Alon & Yahav, 2021).** Consider the Jacobian of node $v$'s representation at layer $k$ with respect to node $u$'s initial feature:
$$\frac{\partial \mathbf{h}_v^{[k]}}{\partial \mathbf{x}_u} = \prod_{l=0}^{k-1} (W^{[l]})^\top \cdot \frac{\partial \mathbf{h}_v^{[k]}}{\partial \mathbf{h}_v^{[k-1]}} \cdots \frac{\partial \mathbf{h}_{u'}^{[1]}}{\partial \mathbf{x}_u}$$

where the product runs along any path from $u$ to $v$ of length $k$. The norm of this Jacobian is bounded by:

$$\left\lVert\frac{\partial \mathbf{h}_v^{[k]}}{\partial \mathbf{x}_u}\right\rVert \leq C \cdot \left(\hat{A}^k\right)_{vu}$$

where $(\hat{A}^k)_{vu}$ is the $(v,u)$ entry of the $k$-th power of the propagation matrix. For nodes far apart in the graph, $(\hat{A}^k)_{vu}$ is exponentially small in the distance $d(u,v)$.

**The bottleneck.** If $u$ and $v$ are separated by a single edge $(s, t)$ of high betweenness centrality (a "bridge"), all information from $u$-side to $v$-side must pass through $(s,t)$. The aggregation at $s$ receives messages from all $|\mathcal{N}(s)|$ neighbors and compresses them into a single $d$-dimensional vector - losing information at rate proportional to $|\mathcal{N}(s)| / d$.

**Over-squashing vs over-smoothing.** These are dual problems:
- Over-smoothing: too much aggregation -> representations become too similar
- Over-squashing: aggregation is a bottleneck -> distant information cannot influence predictions
- Over-smoothing is a property of the graph and its spectrum; over-squashing is a property of specific bottleneck edges

### 8.4 Remedies for Over-Smoothing

**DropEdge (Rong et al., 2020).** Randomly remove edges during training: replace $A$ with a sparse mask $\tilde{A}_{\text{drop}}$ that drops each edge independently with probability $p$. This is analogous to dropout for neurons but applied to edges. It slows the convergence of over-smoothing (less propagation per layer) and acts as a data augmentation method. Improves performance at depths 4-8 on standard benchmarks.

**PairNorm (Zhao & Akoglu, 2020).** After each GCN layer, re-normalize the node representations to ensure a fixed total pairwise distance:
$$\mathbf{h}_v \leftarrow \mathbf{h}_v - \bar{\mathbf{h}}, \quad \bar{\mathbf{h}} = \frac{1}{n}\sum_v \mathbf{h}_v$$
$$\mathbf{h}_v \leftarrow s \cdot \frac{\mathbf{h}_v}{\left(\frac{1}{n}\sum_v \lVert\mathbf{h}_v\rVert^2\right)^{1/2}}$$

where $s$ is a scaling hyperparameter. PairNorm prevents the Dirichlet energy from decaying to zero by keeping pairwise distances bounded below. Empirically effective for deep GCNs (8-16 layers).

**Initial residual connection (GCNII, Chen et al., 2020).**
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(\left[(1-\alpha)\hat{A}\mathbf{h}_v^{[l]} + \alpha \mathbf{h}_v^{[0]}\right] \cdot \left[(1-\beta) I + \beta W^{[l]}\right]\right)$$

Two additions: (1) $\alpha \mathbf{h}_v^{[0]}$ - always mixing in the initial feature with weight $\alpha$ prevents full convergence; (2) the weight matrix mixes between identity ($\beta=0$, pure smoothing) and full linear transform ($\beta=1$). With $\alpha=0.1, \beta = \log(\lambda/l+1)$ (decreasing $\beta$ with depth), GCNII successfully trains 64-layer GCNs, achieving state-of-the-art on Cora at the time (85.5% accuracy vs 81.5% for 2-layer GCN).

**GroupNorm / NodeNorm.** Normalize within groups of nodes to prevent scale collapse. Less commonly used than the above.

### 8.5 Graph Rewiring

A more aggressive approach: **change the graph structure** to improve information flow before running the GNN.

**Diffusion-based rewiring (DIGL, Gasteiger et al., 2019).** Replace $A$ with the heat kernel approximation $\Theta = \exp(-t L)$ (truncated) or personalized PageRank matrix $\Theta = \alpha(I - (1-\alpha)\hat{A})^{-1}$. The diffusion matrix connects distant nodes with non-zero weights, enabling long-range information flow in 1 GNN layer. Edges with small weights are pruned, keeping the graph sparse.

**Expander Graph Propagation (EGP, Deac et al., 2022).** Augment the graph with the edges of a sparse expander graph (a $d$-regular graph with large spectral gap, e.g., a Ramanujan graph). Expander edges provide "shortcuts" that reduce the effective diameter of the graph, alleviating over-squashing without the $O(n^2)$ cost of full connectivity.

**FoSR: First-Order Spectral Rewiring (Karhadkar et al., 2023).** Add edges that most increase $\lambda_2(L)$ (the Fiedler value), directly attacking the spectral bottleneck. Greedy algorithm: at each step, add the edge $(u,v) \notin E$ that maximally increases $\lambda_2$ of the augmented graph. With $k$ added edges, the graph diameter decreases and over-squashing is reduced.

### 8.6 Depth vs Width Trade-off in GNNs

**Why shallow GNNs dominate in practice.** On most benchmark tasks (node classification on citation networks, graph classification on molecular benchmarks), 2-4 GNN layers achieve the best performance. The reasons:

1. **Homophily dominates:** in social and citation networks, the most informative neighbors are at distance 1-2. Beyond distance 3, nodes are often from different classes, and aggregating them hurts classification.
2. **Over-smoothing:** as analyzed above, deep GCNs lose discriminative information.
3. **Exponential neighborhood growth:** a node's $k$-hop neighborhood in a power-law graph has $\sim \bar{d}^k$ nodes. For $\bar{d}=10$ and $k=5$, that's 100,000 nodes - mostly irrelevant.

**When depth helps.** Long-range tasks where predictions depend on distant nodes:
- Predicting molecular properties that depend on the entire molecular graph (QM9 quantum chemistry dataset)
- Reasoning over knowledge graphs with long inference chains
- Planning and path-finding tasks on sparse graphs

For these tasks, graph Transformers (10) - which compute full pairwise attention in $O(n^2)$ - often outperform deep GNNs.

**Jumping Knowledge Networks (Xu et al., 2018).** A principled approach: each node selects the representation from the most appropriate depth:
$$\mathbf{h}_v^{\text{final}} = \operatorname{COMBINE}\!\left(\mathbf{h}_v^{[1]}, \mathbf{h}_v^{[2]}, \ldots, \mathbf{h}_v^{[L]}\right)$$

COMBINE can be concatenation, LSTM-attention, or max-pooling across layers. This allows different nodes to "listen" at different distances - nodes in dense clusters use shallow representations; nodes that are bridges benefit from deeper representations.

---

## 9. Graph Pooling and Hierarchical GNNs

### 9.1 Why Pooling Is Non-Trivial on Graphs

In image CNNs, pooling is straightforward: reduce a $2 \times 2$ spatial block to a single pixel by averaging. The operation is well-defined because images have a fixed grid structure.

On graphs, pooling (coarsening) must answer: which nodes get merged? How are their features combined? How is the edge structure of the coarsened graph defined? All of these must be permutation invariant - the coarsened graph cannot depend on how vertices are labeled.

**The challenges:**
1. **No canonical merging:** unlike pixels in a grid, there is no obvious spatial proximity to guide merging. Spectral methods (9.4) use the Fiedler vector; learned methods (9.3) learn soft assignments.
2. **Edge reconstruction:** after merging nodes $\{v_1, v_2\}$ into a super-node $s$, which edges does $s$ inherit? Typically all edges incident to $v_1$ or $v_2$ - but this may create multi-edges and self-loops.
3. **Information loss:** pooling irreversibly reduces the graph. Unlike deconvolution in image models, there is no standard graph "unpooling" that perfectly recovers the original structure.

### 9.2 Global Pooling Methods

For graph-level tasks, a single pooling step at the end suffices. The readout function $R(\{\mathbf{h}_v^{[L]}\})$ maps the set of node representations to a single graph embedding.

**Sum pooling:** $\mathbf{h}_G = \sum_{v \in V} \mathbf{h}_v^{[L]}$. Sensitive to graph size (larger graphs get larger embeddings). Most expressive for graph-level tasks (by the same argument as sum aggregation for node-level).

**Mean pooling:** $\mathbf{h}_G = \frac{1}{n}\sum_{v} \mathbf{h}_v^{[L]}$. Normalizes for graph size; best when comparing graphs of different sizes.

**Max pooling:** $(\mathbf{h}_G)_k = \max_v (\mathbf{h}_v^{[L]})_k$. Detects whether any node has feature $k$ above a threshold; good for detecting rare structural motifs.

**Attention pooling (gated global pooling):**
$$\mathbf{h}_G = \sum_{v \in V} \operatorname{softmax}\!\left(f_\text{gate}(\mathbf{h}_v^{[L]})\right)_v \cdot f_\text{feat}(\mathbf{h}_v^{[L]})$$

where $f_\text{gate}: \mathbb{R}^d \to \mathbb{R}$ scores each node, and $f_\text{feat}: \mathbb{R}^d \to \mathbb{R}^{d'}$ transforms the features. Allows the model to focus on the most informative nodes. Used in Graph U-Net, Graphormer.

**Set2Set (Vinyals et al., 2016).** An LSTM-based readout that runs $T$ steps of attention over the node set, accumulating a context vector:
$$\mathbf{q}_t = \operatorname{LSTM}(\mathbf{q}_{t-1}), \qquad e_{tv} = \mathbf{q}_t^\top \mathbf{h}_v, \qquad \alpha_{tv} = \operatorname{softmax}_v(e_{tv})$$
$$\mathbf{c}_t = \sum_v \alpha_{tv} \mathbf{h}_v, \qquad \mathbf{h}_G = \mathbf{c}_T$$
More expressive than simple pooling but with $O(nT)$ sequential steps.

### 9.3 DiffPool: Differentiable Graph Pooling

**DiffPool (Ying et al., 2019)** learns soft cluster assignments to hierarchically coarsen the graph. At each pooling level, run two GNNs in parallel:

$$S^{(l)} = \operatorname{softmax}\!\left(\operatorname{GNN}_{\text{pool}}^{(l)}\!\left(A^{(l)}, H^{(l)}\right)\right) \in \mathbb{R}^{n_l \times n_{l+1}}$$
$$Z^{(l)} = \operatorname{GNN}_{\text{embed}}^{(l)}\!\left(A^{(l)}, H^{(l)}\right) \in \mathbb{R}^{n_l \times d}$$

where $n_l$ is the number of nodes at level $l$, $n_{l+1} < n_l$ is the target number of clusters, $S^{(l)}$ is the soft assignment matrix (each node $v$ assigns fractionally to each of $n_{l+1}$ clusters), and $Z^{(l)}$ is the GNN embedding of current-level nodes.

**Coarsened graph:**
$$H^{(l+1)} = S^{(l)\top} Z^{(l)} \in \mathbb{R}^{n_{l+1} \times d}$$
$$A^{(l+1)} = S^{(l)\top} A^{(l)} S^{(l)} \in \mathbb{R}^{n_{l+1} \times n_{l+1}}$$

The coarsened adjacency $A^{(l+1)}$ is dense - every pair of clusters has a (soft) connection weighted by the total edge weight between them.

**Auxiliary losses.** DiffPool adds two regularization terms to the main task loss:
- **Link prediction loss:** $\mathcal{L}_{\text{LP}} = \lVert A^{(l)} - S^{(l)} S^{(l)\top} \rVert_F$ - encourages clusters to correspond to connected subgraphs
- **Entropy loss:** $\mathcal{L}_{\text{E}} = \frac{1}{n}\sum_v H(S^{(l)}_{v,:})$ - encourages crisp (non-uniform) cluster assignments

**Limitation.** DiffPool produces a dense $A^{(l+1)}$ at each level - $O(n_{l+1}^2)$ memory. For large graphs ($n > 10^4$), this is prohibitive.

### 9.4 MinCutPool and Spectral Pooling

**MinCutPool (Bianchi et al., 2020)** addresses DiffPool's density problem by formulating pooling as a spectral clustering problem with a differentiable objective.

**The mincut objective.** For a soft cluster assignment $S \in \mathbb{R}^{n \times k}$, the normalized minimum cut is:

$$\text{minCUT}(S) = \frac{\operatorname{tr}(S^\top L S)}{\operatorname{tr}(S^\top D S)}$$

Minimizing this over soft assignments $S$ is equivalent to spectral clustering - the optimal $S$ consists of the $k$ Fiedler eigenvectors of $L_{\text{rw}} = D^{-1}L$. MinCutPool uses a GNN to produce $S$ and trains end-to-end with the minCUT loss plus an orthogonality regularizer $\lVert S^\top S / \lVert S^\top S \rVert_F - I / \sqrt{k} \rVert_F$ to prevent degenerate cluster assignments.

**Advantage over DiffPool:** the objective is theoretically motivated by spectral graph theory; the sparse regularization keeps the assignment matrix well-conditioned.

### 9.5 SAGPool: Self-Attention Graph Pooling

**SAGPool (Lee et al., 2019)** takes a different approach: instead of soft cluster assignments, select the top-$k$ nodes by a learned importance score and discard the rest.

**Algorithm:**
1. Run one GNN layer to compute node scores: $\mathbf{z} = \operatorname{GNN}(A, H) \in \mathbb{R}^n$
2. Select top-$k$ nodes: $\text{idx} = \operatorname{topk}(\mathbf{z}, k)$
3. Gate the selected features: $H' = H_{\text{idx}} \odot \operatorname{sigmoid}(\mathbf{z}_{\text{idx}})$
4. Induce the subgraph: $A' = A_{\text{idx,idx}}$ (restrict $A$ to selected nodes)

**Advantages:** simple, interpretable (which nodes are selected?), maintains sparsity of adjacency. **Disadvantage:** hard top-$k$ selection is not differentiable (uses straight-through estimator or continuous relaxation during training).

---

## 10. Graph Transformers

### 10.1 Motivation: GNNs vs Transformers

The fundamental constraint of MPNNs is locality: a $k$-layer GNN can only aggregate information within a $k$-hop neighborhood. For long-range dependencies (e.g., two atoms on opposite ends of a large molecule that interact through space), a deep GNN would need $O(\text{diameter})$ layers - running into over-smoothing and over-squashing.

The transformer architecture - with full $O(n^2)$ pairwise attention - solves the locality problem: every node can directly attend to every other node in a single layer. But transformers treat inputs as sets of independent tokens; they have no built-in notion of graph structure.

**Graph Transformers** combine: (1) the **global receptive field** of transformers (full pairwise attention), with (2) **graph-structural inductive biases** (local message passing, positional encodings derived from the graph topology).

The trade-off: full attention costs $O(n^2 d)$ per layer - practical only for small-to-medium graphs ($n \leq 10^4$). For large graphs ($n > 10^6$), neighbor-sampled MPNNs remain the only scalable option.

### 10.2 Positional Encodings for Graphs

In transformers, positional encodings break the permutation symmetry of the attention mechanism: without them, the transformer cannot distinguish token position, and all orderings of the same tokens produce the same output. The same problem occurs in graph transformers: the attention mechanism is permutation invariant, so we need positional encodings that break symmetry in a **structure-aware** way.

> **Recall from 11-04 9.5:** Laplacian eigenvectors provide a natural graph Fourier basis, and the first $k$ eigenvectors of $L_{\text{sym}}$ give a $k$-dimensional coordinate for each node that reflects the graph's spectral structure. The sign invariance problem (eigenvectors are defined up to sign) is addressed by random sign flipping during training or by using absolute values. Full treatment: [11-04 9.5](../04-Spectral-Graph-Theory/notes.md#95-spectral-positional-encodings-for-transformers).

Building on this, three main PE strategies for graph transformers:

**Laplacian PE (LapPE).** For each node $v$, extract the $v$-th row of the matrix $[\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k]$ where $\mathbf{u}_i$ are the first $k$ eigenvectors of $L_{\text{sym}}$ (excluding the constant eigenvector). Append to node features:
$$\mathbf{x}_v \leftarrow \left[\mathbf{x}_v \,\|\, \mathbf{u}_1(v), \ldots, \mathbf{u}_k(v)\right]$$

Sign ambiguity fix: during training, randomly flip the sign of each eigenvector independently (this has no effect on the Laplacian spectrum but makes the model invariant to sign choice). Used in Dwivedi et al. (2020), GPS (2022).

**Random Walk SE (RWSE).** For each node $v$ and walk length $p$, compute $(A^p D^{-1})_{vv}$ - the probability of a length-$p$ random walk returning to $v$. Stack for $p = 1, \ldots, P$:
$$\text{RWSE}_v = \left[(AD^{-1})_{vv}, (A^2 D^{-2})_{vv}, \ldots, (A^P D^{-P})_{vv}\right]$$

RWSE is always positive and sign-free (no sign ambiguity). It encodes local loop structure: $(AD^{-1})_{vv} = 0$ iff $v$ has no self-loops; $(A^2 D^{-2})_{vv} > 0$ iff any neighbor of $v$ is also connected back to $v$ (triangles). Used in GPS (2022), GIN+RWSE.

**Degree and centrality encoding.** Simply append $d_v$ (node degree), normalized closeness centrality, or betweenness centrality as scalar node features. Used in Graphormer (2021), which also encodes shortest-path distances as edge features.

### 10.3 GPS Framework

**General, Powerful, Scalable (GPS) Graph Transformer** (Rampasek et al., 2022) is the most general graph transformer framework, winning multiple tracks of the 2022 Long Range Graph Benchmark (LRGB).

**GPS Layer:**
$$H^{[l+1]} = \operatorname{LayerNorm}\!\left(H^{[l]} + \operatorname{MPNN}^{[l]}\!\left(H^{[l]}, A\right) + \operatorname{Transformer}^{[l]}\!\left(H^{[l]}\right)\right)$$

Each GPS layer combines:
1. **Local MPNN:** any standard GNN (GCN, GINE, GAT) operating on the sparse graph edges - captures local structure efficiently
2. **Global Transformer:** full multi-head self-attention over all nodes - captures long-range dependencies
3. **LayerNorm + residual:** stabilizes training

**Modularity.** GPS is a framework, not a fixed architecture: the MPNN and Transformer components are interchangeable. In practice, GINE (GIN with edge features) + Transformer works well; one can also use GAT + Performer (linear attention approximation) for scalability.

**Positional encodings.** GPS accepts arbitrary node-level PEs (LapPE, RWSE, degree) as extra input features, processed by a dedicated PE encoder:
$$\mathbf{x}_v^{\text{in}} = W_{\text{feat}} \mathbf{x}_v + W_{\text{PE}} \text{PE}_v$$

**Performance.** On the LRGB Peptides-func benchmark (a graph where predictions require integrating global molecular structure), GPS achieves significantly higher accuracy than any pure MPNN, demonstrating the necessity of global attention for long-range tasks.

### 10.4 Graphormer

**Graphormer (Ying et al., 2021)** uses a standard transformer architecture (with full pairwise attention) augmented by three graph-structural encodings. It won the OGB-LSC 2021 quantum chemistry track.

**Central encoding.** Add a degree-dependent bias to the node features:
$$\mathbf{x}_v^{\text{in}} = \mathbf{x}_v + \mathbf{z}_{d_v^-} + \mathbf{z}_{d_v^+}$$

where $\mathbf{z}_{d^-}$ and $\mathbf{z}_{d^+}$ are learned embeddings for in- and out-degree. This allows the transformer to distinguish hub nodes from leaf nodes.

**Spatial encoding.** Add a bias to the attention score between nodes $u$ and $v$ based on their shortest-path distance $\phi(u,v)$:
$$e_{uv} = \frac{(\mathbf{h}_u W_Q)(\mathbf{h}_v W_K)^\top}{\sqrt{d_k}} + b_{\phi(u,v)}$$

where $b_\phi$ is a scalar learned per distance $\phi$. Nodes far apart get a learned attention bias; nodes nearby get a different bias. This encodes graph topology directly into the attention pattern.

**Edge encoding.** For each pair $(u,v)$, average the edge features along the shortest path:
$$c_{uv} = \frac{1}{\phi(u,v)} \sum_{e \in \text{sp}(u,v)} \mathbf{w}_e^\top \mathbf{a}$$

where $\mathbf{w}_e \in \mathbb{R}^{d_e}$ is the feature of path edge $e$ and $\mathbf{a} \in \mathbb{R}^{d_e}$ is a learned vector. Added to the attention score as an additional bias.

### 10.5 Graph Mamba and Sequence-Based Methods

The quadratic $O(n^2)$ cost of full attention makes graph transformers impractical for large graphs. Recent work (2023-2024) explores linear-complexity alternatives.

**Converting graphs to sequences.** Several methods serialize the graph into a sequence of tokens, then apply a sequence model (LSTM, Mamba SSM):
- **BFS/DFS ordering:** nodes in BFS/DFS traversal order; nearby nodes in the graph -> nearby in sequence. Breaks permutation invariance (different traversal orderings give different results).
- **Node-and-edge tokenization (TokenGT, Kim et al., 2022):** represent each node and each edge as a separate token; full transformer attention over all $n + m$ tokens. Permutation invariant if node/edge PEs are orthogonalized.

**Graph Mamba (Chen et al., 2023).** Apply Mamba (state-space model with selective scan) to graphs by: (1) ordering nodes by degree or BFS, (2) running the SSM along this ordering as a "linearized graph sequence." Achieves $O(n \log n)$ complexity while matching transformer performance on some benchmarks. The key challenge: the SSM must be made robust to different node orderings (since graphs have no canonical ordering).

**Status (2026).** Graph Mamba and similar linear-attention methods are active research areas. For small-to-medium graphs ($n \leq 10^4$), GPS-style full attention is standard. For large graphs, MPNN-based methods (GraphSAGE, Cluster-GCN) remain dominant in production.

---

## 11. Training and Scaling GNNs

### 11.1 Full-Batch vs Mini-Batch Training

**Full-batch training.** Compute the GNN on the entire graph in each step. Requires the entire graph and all node features to fit in GPU memory. Practical for graphs with $n \leq 10^5$ nodes. Exact gradient computation.

**Mini-batch training.** Sample a set of target nodes (a "mini-batch") and compute the GNN only on those nodes and their $k$-hop neighborhoods. Much more memory-efficient but requires careful sampling to produce unbiased gradient estimates.

The challenge of mini-batch training for GNNs is the **neighborhood explosion**: if each node has degree $\bar{d}$, a $k$-hop neighborhood has $\bar{d}^k$ nodes. For $\bar{d}=10$, $k=3$, this is 1000 nodes per target node - often larger than the desired mini-batch size.

Three approaches manage this explosion: neighbor sampling (11.2), graph partitioning (11.3), and subgraph sampling (11.4).

### 11.2 Neighbor Sampling

**GraphSAGE neighbor sampling (Hamilton et al., 2017).** For each target node at each layer, sample a fixed-size subset $S_l$ of neighbors:
- Layer 1 (nearest neighbors): sample $S_1 = 25$ neighbors
- Layer 2 (2-hop): sample $S_2 = 10$ neighbors of each sampled layer-1 neighbor

Total nodes per target: $S_1 \times S_2 = 250$. Constant cost per target node, regardless of graph degree.

**Variance reduction.** Random sampling introduces variance in the gradient estimate. This can be reduced by importance sampling (weight each sampled neighbor by $d_v / S_l$) or by using multiple samples. In practice, $S_l = 5$-$25$ is sufficient for most tasks.

**VR-GCN (Chen et al., 2018).** Control variates method: maintain historical node embeddings $\bar{\mathbf{h}}_v^{[l]}$ as running averages. Use the historical embedding as a control variate for unsampled neighbors:
$$\hat{\mathbf{m}}_v = \frac{n}{|S|}\sum_{u \in S} \left(\mathbf{h}_u - \bar{\mathbf{h}}_u\right) + \sum_{u \in \mathcal{N}(v)} \bar{\mathbf{h}}_u$$

This gives an unbiased estimate of the full-batch gradient with lower variance than simple neighbor sampling.

### 11.3 Cluster-GCN

**Cluster-GCN (Chiang et al., 2019)** takes a graph partitioning approach: partition the graph into $C$ balanced clusters $\mathcal{V}_1, \ldots, \mathcal{V}_C$ using METIS or other graph partitioning algorithms. A mini-batch consists of one or more clusters.

**Key insight.** Within a cluster, most edges are intra-cluster (partitioning algorithms minimize the cut). Running GCN on the induced subgraph of a cluster produces embeddings that see most relevant edges. Between-cluster edges (the cut) are ignored - this introduces approximation error, but the cut is small by design.

**Memory:** a cluster of size $\lvert\mathcal{V}_c\rvert = n/C$ requires $O(n/C)$ node features + the induced adjacency. For $C=100$, memory is $100\times$ smaller than full-batch training.

**Variance:** the gradient approximation error comes from ignoring cross-cluster edges. To reduce this, Cluster-GCN uses **multiple cluster sampling**: each mini-batch consists of $q$ randomly selected clusters $\bigcup_{i=1}^q \mathcal{V}_{c_i}$, retaining all edges within the union. This recovers some cross-cluster edges.

### 11.4 GraphSAINT: Subgraph Sampling

**GraphSAINT (Zeng et al., 2020)** frames mini-batch training as subgraph sampling: each mini-batch is a subgraph $G[S]$ induced by a sampled node set $S \subseteq V$.

**Three sampling strategies:**

1. **Node sampler:** sample $|S|$ nodes uniformly; include all edges between sampled nodes. Simple but misses important edges.
2. **Edge sampler:** sample $|S_E|$ edges uniformly; include all nodes incident to sampled edges. Better edge coverage.
3. **Random walk sampler:** start $r$ random walks from random nodes; include all visited nodes (and their induced edges). Produces connected subgraphs with richer local structure.

**Normalization.** Since different edges are sampled with different probabilities, the gradient is biased without correction. GraphSAINT computes the sampling probability $p(v)$ for each node and $p(u,v)$ for each edge analytically (or by estimation), then reweights the loss:
$$\mathcal{L}(G[S]) = \frac{1}{\lvert S \rvert} \sum_{v \in S} \frac{1}{n \cdot p(v)} \ell(\hat{y}_v, y_v)$$

This gives an unbiased gradient estimate and removes systematic bias from subgraph sampling.

**Scaling.** GraphSAINT successfully trains GNNs on Flickr (89K nodes, 900K edges), Reddit (232K nodes, 114M edges), and Yelp (716K nodes, 13M edges) - orders of magnitude larger than full-batch training can handle.

---

## 12. Applications in Machine Learning

### 12.1 Molecular Property Prediction and Drug Discovery

Molecular property prediction is the "killer application" of GNNs. A molecule is naturally a graph: atoms are nodes (with features: atomic number, hybridization, formal charge, chirality), and bonds are edges (with features: bond type, aromaticity, stereo configuration).

**MPNN for quantum chemistry (Gilmer et al., 2017).** Applied to the QM9 dataset (134K small organic molecules, 12 quantum chemical properties including dipole moment, polarizability, and HOMO-LUMO gap). The MPNN with edge-network message functions achieves state-of-the-art on 11 of 12 properties, demonstrating that learned graph representations outperform hand-engineered molecular descriptors.

**SchNet (Schutt et al., 2017).** A rotationally equivariant GNN for 3D molecular geometry. Nodes are atoms; edges connect atoms within a cutoff radius $r_c$ with edge features encoding pairwise distances. Uses continuous-filter convolutional layers: $\mathbf{m}_{uv} = \mathbf{h}_u \odot f_\text{filter}(\lVert\mathbf{r}_u - \mathbf{r}_v\rVert)$ where $f_\text{filter}$ is an MLP applied to interatomic distance. Achieves chemical accuracy on QM9 for most properties.

**AlphaFold 2 (Jumper et al., 2021).** The protein structure prediction breakthrough. Key GNN component: the **Evoformer**, which processes pairs of residues (edge features) and individual residues (node features) through an iterative process. The structure module then uses equivariant graph operations to predict 3D residue coordinates. AlphaFold 2 won the CASP14 protein structure prediction competition with $>90\%$ accuracy on most targets - a problem open for 50 years.

**Drug discovery pipeline.** Modern computational drug discovery uses GNNs at every stage:
- *Virtual screening:* predict binding affinity between drug candidates (small molecules) and protein targets; GNNs on molecular graphs reduce wet-lab screening costs by $\sim 100\times$
- *ADMET prediction:* predict absorption, distribution, metabolism, excretion, and toxicity properties from molecular structure
- *De novo design:* generative GNNs (junction tree VAE, graph diffusion models) design novel molecules with desired properties

### 12.2 Knowledge Graph Reasoning

A **knowledge graph (KG)** is a directed, heterogeneous graph $G = (V, E, R)$ where $V$ is a set of entities, $R$ is a set of relation types, and $E \subseteq V \times R \times V$ is a set of triples (subject, relation, object): e.g., (Einstein, bornIn, Germany), (Germany, locatedIn, Europe).

**Triple completion (link prediction).** Given an incomplete KG, predict missing triples: is (Einstein, worksWith, Bohr) a true triple? Key methods:

- **TransE (Bordes et al., 2013):** model relation $r$ as a translation in embedding space: $\mathbf{h}_s + \mathbf{r} \approx \mathbf{h}_o$ for true triples $(s,r,o)$. Simple and effective for 1-to-1 relations.
- **RotatE (Sun et al., 2019):** model relation as a rotation in complex space: $\mathbf{h}_s \circ \mathbf{r} = \mathbf{h}_o$ where $\circ$ is element-wise complex multiplication. Captures symmetry, antisymmetry, and composition patterns.

**R-GCN (Schlichtkrull et al., 2018).** GNN for relational (heterogeneous) graphs. Separate weight matrix $W_r$ for each relation type:
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W_0 \mathbf{h}_v^{[l]} + \sum_{r \in R} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{v,r}} W_r \mathbf{h}_u^{[l]}\right)$$

where $\mathcal{N}_r(v)$ are the neighbors of $v$ through relation $r$ and $c_{v,r}$ is a normalization constant. Used for entity classification and relation prediction in Freebase and AIFB knowledge graphs.

**For AI:** In 2024, knowledge graphs power retrieval-augmented generation systems. Microsoft's Graph RAG (Edge et al., 2024) builds a knowledge graph from documents, then uses GNNs (and graph traversal) to answer complex multi-hop questions that flat vector retrieval cannot handle.

### 12.3 Recommendation Systems

Recommendation is a bipartite graph problem: users $U$ and items $I$ form nodes; interactions (clicks, purchases, ratings) form edges. The task is link prediction: predict which unobserved $(u, i)$ pairs represent true preferences.

**LightGCN (He et al., 2020).** Simplifies GCN for collaborative filtering by removing the feature transformation and nonlinearity, keeping only the graph smoothing:
$$\mathbf{h}_u^{[l+1]} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{d_u d_i}} \mathbf{h}_i^{[l]}, \qquad \mathbf{h}_i^{[l+1]} = \sum_{u \in \mathcal{N}_i} \frac{1}{\sqrt{d_i d_u}} \mathbf{h}_u^{[l]}$$

Final embeddings: $\mathbf{h}_v = \sum_{l=0}^L \alpha_l \mathbf{h}_v^{[l]}$ (JK-style layer combination). Prediction: $\hat{y}_{ui} = \mathbf{h}_u^\top \mathbf{h}_i$.

LightGCN outperforms standard GCN for collaborative filtering, suggesting that the feature transformation is unnecessary (or harmful) when the only input feature is an ID embedding. The key contribution is the propagation of user-item signals through multi-hop paths.

**PinSage** (already discussed in 5.5) extends GraphSAGE to the bipartite user-pin graph with importance-based sampling and hard negative mining.

### 12.4 Code and Program Analysis

Source code can be represented as multiple graphs simultaneously:
- **AST (Abstract Syntax Tree):** hierarchical tree showing syntactic structure
- **CFG (Control Flow Graph):** nodes are basic blocks; edges show execution flow (if/else, loops)
- **DFG (Data Flow Graph):** edges connect variable definitions to their uses
- **Call Graph:** nodes are functions; edges connect callers to callees

**code2vec (Alon et al., 2019).** Represents code snippets as bags of AST paths; learns path embeddings and aggregates them for downstream tasks. While not a full GNN, it demonstrates the power of structural code representations.

**GNNs for bug detection.** Allamanis et al. (2018) build heterogeneous graphs from Python/C code with AST, data flow, and control flow edges; train a GNN to classify whether a variable name is misused. Achieves high precision on detecting certain classes of bugs.

**Program synthesis.** GNNs operating on execution traces (program state as graph) guide neural program search. DeepCoder (Balog et al., 2017) and more recent systems use graph representations of input-output examples to synthesize programs.

### 12.5 LLM Integration: Graph RAG and Structure-Aware Language Models

The 2024-2026 frontier is combining GNNs with large language models - exploiting the complementary strengths of relational structure (GNNs) and natural language understanding (LLMs).

**Graph RAG (Edge et al., 2024, Microsoft Research).** Standard RAG (retrieval-augmented generation) retrieves flat text chunks. Graph RAG builds a knowledge graph from documents and uses graph community detection (Leiden algorithm) to generate hierarchical summaries. At query time, relevant communities are retrieved and their summaries are used to ground the LLM response.

Key insight: complex questions requiring synthesis across many documents (e.g., "What are the main themes in this corpus?") are better answered by traversing a knowledge graph than by retrieving isolated text chunks. Graph RAG achieves $\sim 72$% win rate over naive RAG on these "global sensemaking" queries.

**GNN+LLM for molecular generation.** LLMs can generate SMILES strings (text representations of molecules), but they struggle with structural validity constraints (valence, aromaticity). GNN-based validity checkers or graph-structured decoders can enforce these constraints.

**Graph-structured memory for agents.** Recent AI agents (2025) maintain working memory as knowledge graphs updated with each observation. A GNN processes the graph state to inform action selection. This enables multi-hop reasoning ("I know A->B->C, so if I see C, I should infer A") that flat token sequences handle poorly.

### 12.6 Financial Networks, Credit Risk and Default Contagion

*This subsection is a GizzZmo-specific addition to the upstream curriculum (the root README links here); it applies the chapter's machinery - message passing, spectral intuition, and graph representation learning - to the network where those ideas were, historically, first forced into mathematical form: the interbank system.*

**Financial systems are graphs.** A bank, insurer, or fund is a node; a loan, repo, derivative, or payment obligation is a directed, weighted edge. Write $L_{ij} \ge 0$ for the nominal amount institution $i$ owes institution $j$, $\bar{p}_i = \sum_j L_{ij}$ for $i$'s total interbank obligations, $e_i \ge 0$ for $i$'s external assets (loans to the real economy, securities, reserves), and $c_i$ for $i$'s capital or equity buffer. The balance-sheet identity is

$$\text{assets} \;=\; e_i + \sum_j L_{ji} \qquad\text{versus}\qquad \text{liabilities} \;=\; \bar{p}_i + \text{(outside liabilities)},$$

so solvency at nominal values is the statement $e_i + \sum_j L_{ji} \ge \bar{p}_i + \text{outside liabilities}$. The systemic-risk question is never whether one node is solvent in isolation; it is whether the *payment system* - all nodes settling simultaneously - has a consistent outcome, and how much of a single node's failure is transmitted along edges before it is absorbed by capital. That is a fixed-point problem on a graph, which is why this material belongs in a graph-learning chapter.

**Clearing payments: Eisenberg and Noe (2001).** Eisenberg and Noe, "Systemic Risk in Financial Systems", *Management Science* 47 (2001), model settlement as follows. If every institution paid in full, $i$ would receive $\sum_j \Pi_{ji} \bar{p}_j$, where $\Pi_{ij} = L_{ij} / \bar{p}_i$ is $i$'s relative-liability matrix. Under limited liability and pro-rata payment to creditors, a **clearing payment vector** $p^*$ satisfies

$$p_i^* \;=\; \min\!\Big\{\bar{p}_i,\; e_i + \sum_j \Pi_{ji} p_j^*\Big\} \qquad \text{for every } i,$$

i.e. each institution pays everything it has - external assets plus whatever it actually received - up to what it owes. Eisenberg and Noe prove a clearing vector always exists and is essentially unique under mild regularity (e.g. every institution has positive external assets reachable in the liability graph), and that the **fictitious default algorithm** - start from $p = \bar{p}$, repeatedly replace $p$ by the right-hand side, letting newly insolvent institutions default in rounds - finds the greatest clearing vector in at most $n$ default rounds. Two features deserve emphasis for a machine-learning audience. First, the clearing map is a monotone fixed-point iteration on a graph: information (shortfalls) propagates along liability edges exactly like messages in an MPNN, except the aggregation is accounting, not learned. Second, the algorithm's rounds are *default cascades with dates*: the round at which a node first pays less than $\bar{p}_i$ is its model default time.

**A three-bank clearing, computed by hand.** Banks A, B, C each owe 40 to the next bank in a cycle (A owes B, B owes C, C owes A) and 60 to outside creditors, so $\bar{p}_i = 100$ for all three and the interbank share of every payment is $40/100 = 0.4$. External assets are $e = (90, 20, 70)$. Receipts in the cycle run backwards along the obligation cycle: A receives $0.4\,p_C$, B receives $0.4\,p_A$, C receives $0.4\,p_B$.

Fictitious default algorithm, starting from full payment:

| Round | $p_A$ | $p_B$ | $p_C$ | What happened |
|-------|-------|-------|-------|----------------|
| 0 (nominal) | 100 | 100 | 100 | assumed full payment |
| 1 | 100 | 60 | 100 | B has $20 + 0.4(100) = 60 < 100$: B defaults fundamentally |
| 2 | 100 | 60 | 94 | C now receives $0.4(60) = 24$, so C has $70 + 24 = 94 < 100$: contagion default |
| 3 | 100 | 60 | 94 | A has $90 + 0.4(94) = 127.6 \ge 100$: fixed point reached |

The clearing vector is $p^* = (100, 60, 94)$: recovery rates 100%, 60%, 94%. Read the economics off the arithmetic. B is insolvent even at full counterparty payment ($20 + 40 = 60 < 100$) - a *fundamental* default. C would be solvent at full payment ($70 + 40 = 110 \ge 100$) but is dragged into partial default purely by B's shortfall: its 16 units of lost receipts exceed its 10 units of headroom. A survives comfortably because its external assets (90) dwarf its interbank exposure. The actual payments are $A \to B$ 40, $B \to C$ 24, $C \to A$ 37.6, with the remainders (60, 36, 56.4) going to outside creditors pro-rata. Equity after clearing is $(27.6, 0, 0)$: defaulting banks' shareholders are wiped out, exactly as limited liability prescribes. Note what the network did: a loss that "should" have been B's alone (40 unpaid) was partially exported to C (6 unpaid) - and would have travelled further around a longer cycle, shrinking at each hop only where capital absorbed it. Rogers and Veraart, "Failure and Rescue in an Interbank Network" (2013), extend the model with bankruptcy costs and show the clearing outcome can then be non-unique and materially worse; Glasserman and Young, "How Likely Is Contagion in Financial Networks?", *Journal of Economic Literature* 54 (2016), survey how sensitive all such conclusions are to the assumed network.

**Cascade and threshold models.** A complementary tradition models default as a threshold process rather than a payment equilibrium. Gai and Kapadia, "Contagion in financial networks", *Proceedings of the Royal Society A* 466 (2010), place banks on a random graph with capital buffers and declare a bank failed when losses on defaulted counterparties exceed its capital; failures then iterate - the same bootstrap-percolation dynamics as the cascade demos elsewhere in this curriculum. Their central finding is a robust-yet-fragile pattern: higher connectivity first *reduces* contagion (losses are shared thinly) and then, past a point, *increases* it (channels multiply). Acemoglu, Ozdaglar and Tahbaz-Salehi, "Systemic Risk and Stability in Financial Networks", *American Economic Review* 105 (2015), make the phase transition precise: for small shocks denser networks are more stable (risk sharing dominates), while for sufficiently large shocks the same density becomes the contagion vector - there is no connectivity level that is safest at every shock size. Allen and Gale, "Financial Contagion", *Journal of Political Economy* 108 (2000), is the canonical early statement that complete networks absorb liquidity shocks better than incomplete ones, with the modern literature (above) supplying the qualifications. For graph learners, the transferable lesson is that "more edges" is neither robustness nor fragility in itself; the sign of the effect depends on shock size relative to buffers - a claim no purely topological statistic can settle.

**DebtRank: distress before default.** Binary default is a coarse signal; regulators also want to know who is *distressed*. Battiston, Puliga, Kaushik, Tasca and Caldarelli, "DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk", *Scientific Reports* 2 (2012), propagate a continuous distress variable $h_i \in [0, 1]$ over the exposure network. Let $W_{ij} = L_{ji} / c_i$ be the impact of institution $j$'s distress on $i$: the exposure of $i$ to $j$ (what $j$ owes $i$, $L_{ji}$) measured in units of $i$'s capital. Starting from an initially distressed set, distress updates as

$$h_i(t+1) \;=\; \min\!\left\{1,\; h_i(1) + \sum_j W_{ij}\, h_j(t)\right\},$$

(with the original algorithm counting each node's distress transmission once, via an inactive/active state refinement). A node's DebtRank weights its final distress by its share of total system equity, $\sum_i h_i c_i / \sum_i c_i$, producing a systemic-importance ranking that, in the FED interbank application of the paper, flags institutions that degree or size alone would miss. Observe the form: DebtRank *is* linear message passing with a saturating nonlinearity - one GCN layer family, with weights fixed by accounting identities rather than learned. The theory notebook's appended demo computes it side by side with an Eisenberg-Noe clearing on the same synthetic balance sheets, which is the right way to see both what the two formalisms share (propagation along exposure edges) and where they differ (equilibrium payments versus heuristic distress diffusion, with no uniqueness claim for the latter).

**What GNNs are actually used for here.** Three uses recur in the applied literature and in industry risk teams.

- *Exposure link prediction.* Supervisory data on bilateral exposures is confidential and incomplete; the observable network is a sampled subgraph. Link prediction (the machinery of Sections 12.2-12.3) reconstructs plausible missing exposures, and clearing/cascade models are then run over the *ensemble* of reconstructed networks rather than a single point estimate. The honest output is a distribution of systemic losses, not a number.
- *Node default classification.* Given balance-sheet features and the exposure graph, a GNN scores institutions by default propensity, letting neighbourhood distress inform the score - a direct node-classification task in the sense of Section 3, with DebtRank-style propagation as a strong hand-crafted baseline that any learned model should be required to beat.
- *Stress-test scenario generation.* Learned embeddings of institutions and markets are used to propose adverse-but-plausible shock vectors (which nodes lose external assets, jointly), which are then fed to the *mechanical* clearing model for losses. The division of labour matters: ML proposes scenarios and fills data gaps; the accounting fixed point, not the neural network, computes who pays whom. Keeping the clearing step mechanistic is what makes the pipeline auditable by a regulator.

**Honest limitations.** This is the part of the section to read twice.

- *The true network is largely unobserved.* Bilateral exposures are among the most confidential data in finance. Reconstructions from aggregate balance sheets (maximum-entropy and descendants) systematically *underestimate* contagion concentration, because they spread exposures that are in reality lumpy. A GNN trained on a reconstructed graph learns the reconstruction's artefacts along with the world.
- *Labels barely exist.* Systemic crises are, mercifully, a handful of observations per century per jurisdiction, and institution-level defaults are rare, censored, and politically mediated (bailouts erase the label "would have defaulted"). Node classification here is extreme few-shot learning with distribution shift built in; held-out accuracy on calm-period data certifies little about crisis behaviour.
- *Procyclicality and feedback.* Risk models that many institutions share change the network they measure: common signals induce common asset sales and herding, so edges and node features respond to the model's own output. This is Goodhart's law with a balance sheet, and it invalidates the fixed-graph assumption every method in this chapter makes.
- *Equilibrium versus dynamics.* Eisenberg-Noe clearing is a settlement-day abstraction: no fire sales, no liquidity hoarding, no margin spirals, no intraday timing. Extensions (Rogers-Veraart bankruptcy costs; fire-sale overlays) change quantitative answers materially. A clearing computation is a lower bound on losses with a clear conscience, not a forecast.
- *Validation is structurally hard.* There is no counterfactual crisis in the test set. Backtesting can check payment mechanics and calm-period default ranking; it cannot certify tail behaviour. Consequently the defensible deliverables are rankings, scenario comparisons, and sensitivity ranges - not point predictions of systemic loss.

**Audit checklist for financial-network models.** Before trusting any output of this pipeline, ask: (1) Is the exposure network observed, reconstructed, or synthesised - and are results reported over that uncertainty? (2) Are capital buffers measured at market or book values, and as of when? (3) Does the contagion step use a stated clearing/cascade formalism (Eisenberg-Noe, threshold, DebtRank) whose fixed point was verified, rather than an unexamined simulation loop? (4) Are "contagion defaults" (solvent at full payment) reported separately from fundamental defaults, as in the worked example? (5) Has a dense-network stress been run at *large* shock size, where the Acemoglu-Ozdaglar-Tahbaz-Salehi phase flip lives? (6) Would the conclusion survive the maximum-entropy reconstruction being wrong in the direction of lumpier exposures?

**Canonical references for this subsection.** Eisenberg & Noe (2001), *Management Science* 47:236-249. Allen & Gale (2000), *Journal of Political Economy* 108:1-33. Gai & Kapadia (2010), *Proceedings of the Royal Society A* 466:2401-2423. Battiston et al. (2012), *Scientific Reports* 2:541. Rogers & Veraart (2013), *Management Science* 59:882-898. Acemoglu, Ozdaglar & Tahbaz-Salehi (2015), *American Economic Review* 105:564-608. Glasserman & Young (2016), *Journal of Economic Literature* 54:779-831. No other sources are cited in this subsection, and none are needed to reproduce its arithmetic.
---

## 13. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Using mean aggregation for graph classification when graph size varies | Mean normalizes by node count - two graphs with identical local structure but different sizes get identical embeddings, despite being different | Use sum aggregation (which preserves size information) or pair with a size feature |
| 2 | Forgetting self-loops in GCN | Without self-loops ($\tilde{A} = A + I$), a node does not include its own features in aggregation - it only receives neighbors' information, losing its own signal | Always add $I$ to $A$ before normalization; this is the renormalization trick from Kipf & Welling |
| 3 | Adding too many GCN layers and blaming model capacity | Deep GCNs fail due to over-smoothing, not lack of capacity - adding parameters won't help | Use 2-4 layers; add residual connections (GCNII), DropEdge, or PairNorm if more depth is needed |
| 4 | Treating GAT attention weights as feature importance | Attention weights tell you which neighbors were weighted more, not which features were important - this is the "attention is not explanation" problem | Use gradient-based attribution (GradCAM, Integrated Gradients) for feature importance; treat attention as architectural choice, not explanation |
| 5 | Using GIN with mean aggregation | GIN's theoretical power (matching 1-WL) requires sum aggregation with an injective MLP. Mean aggregation in GIN is worse than GCN in expressiveness | Always use sum aggregation in GIN; verify that the MLP has sufficient depth (\geq2 layers) |
| 6 | Forgetting that 1-WL cannot detect triangles | Any MPNN is bounded by 1-WL; 1-WL cannot count triangles or cliques. If your task requires triangle detection (e.g., social network analysis), a standard GNN will fail | Add structural features (triangle count, RWSE) or use a higher-order GNN (NGNN, subgraph GNN) |
| 7 | Normalizing node features but not edge features | Unnormalized edge features with large variance can dominate the attention scores in GAT or the message in MPNN | Normalize edge features to zero mean, unit variance; use LayerNorm before message computation |
| 8 | Using transductive GCN for inductive tasks | GCN's normalized adjacency is computed on the training graph; new nodes at test time require recomputing the entire adjacency and rerunning the network | Use inductive methods (GraphSAGE, GAT) that learn aggregation functions, not fixed propagation matrices |
| 9 | Applying global pooling before sufficient local aggregation | With only 1 GNN layer before readout, each node only knows its immediate neighbors; graph-level representations lack structural context | Use 3-5 GNN layers before readout; consider hierarchical pooling (DiffPool) for hierarchically structured graphs |
| 10 | Ignoring the over-squashing problem for long-range tasks | If the task requires integrating information from nodes far apart, a shallow MPNN will fail due to over-squashing even without over-smoothing | Use graph rewiring (DIGL, EGP) or graph transformers (GPS, Graphormer) for long-range tasks; measure effective receptive field |
| 11 | Training GNN on the test graph for transductive semi-supervised learning | Using the test graph topology during training is allowed (transductive setting), but using test node labels is a data leakage error | Only mask the labels of test nodes; the full graph adjacency is legitimately used during both training and testing in the transductive setting |
| 12 | Confusing graph-level and node-level tasks in the readout | Using node embeddings directly for graph classification (without pooling) produces an embedding for each node, not for the graph - dimensions won't match | Always apply a readout function (sum/mean/attention pooling) after the final GNN layer for graph-level tasks |

---

## 14. Exercises

**Exercise 1 * - MPNN Implementation**

Implement a 2-layer MPNN on a small graph and compute node representations from scratch.

(a) Construct the adjacency matrix $A$ for the graph: nodes $\{0,1,2,3,4\}$, edges $\{(0,1),(1,2),(2,3),(3,4),(0,3)\}$.

(b) Initialize node features $X = I_5$ (identity matrix - each node has a one-hot representation).

(c) Implement one GCN layer: $H^{[1]} = \operatorname{ReLU}(\hat{A} X W^{[0]})$ where $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$, $\tilde{A} = A + I$, and $W^{[0]} \in \mathbb{R}^{5 \times 3}$ is a random matrix (fixed seed).

(d) Apply a second GCN layer to get $H^{[2]} \in \mathbb{R}^{5 \times 2}$.

(e) Verify that permuting the nodes (applying a permutation $P$ to rows of $X$ and both rows and columns of $A$) produces permuted outputs $P H^{[2]}$.

---

**Exercise 2 * - GCN Propagation Matrix**

Analyze the spectral properties of the GCN propagation matrix $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$.

(a) For a path graph $P_5$ (5 nodes in a line), compute $\hat{A}$ and find its eigenvalues.

(b) Show that all eigenvalues of $\hat{A}$ lie in $(-1, 1]$ by relating $\hat{A}$ to the normalized Laplacian $L_{\text{sym}} = I - \hat{A}$ and using the PSD property of $L_{\text{sym}}$.

(c) Compute $\hat{A}^k X$ for $k = 1, 2, 4, 8$ and visualize the row norms as a function of $k$. Observe over-smoothing: the rows converge.

(d) Show that the limiting matrix $\hat{A}^\infty X$ has all rows proportional to $\boldsymbol{\pi}$, the stationary distribution. Compute $\boldsymbol{\pi}$ for $P_5$.

---

**Exercise 3 * - Aggregation Expressiveness**

Demonstrate that sum aggregation is strictly more expressive than mean for multisets.

(a) Construct two multisets $\mathcal{M}_1 = \{1, 1\}$ and $\mathcal{M}_2 = \{1, 1, 1\}$ of node features (scalar, value 1). Compute sum($\mathcal{M}_1$), mean($\mathcal{M}_1$), sum($\mathcal{M}_2$), mean($\mathcal{M}_2$). Verify that sum distinguishes them but mean does not.

(b) Construct two multisets $\mathcal{M}_3 = \{1, 2\}$ and $\mathcal{M}_4 = \{2\}$. Show that max($\mathcal{M}_3$) = max($\mathcal{M}_4$) = 2, but sum($\mathcal{M}_3$) $\neq$ sum($\mathcal{M}_4$).

(c) For two graphs $G_1 = K_{1,3}$ (star with 3 leaves) and $G_2 = P_4$ (path of 4 nodes), show that a GCN with mean aggregation assigns identical representations to some nodes, but a GIN with sum aggregation does not.

(d) Implement the WL color refinement algorithm for $G_1$ and $G_2$ with uniform initial colors. How many iterations until stable? Does WL distinguish $G_1$ from $G_2$?

---

**Exercise 4 ** - Weisfeiler-Leman Test**

Implement the 1-WL algorithm and apply it to pairs of graphs.

(a) Implement `wl_isomorphism_test(G1, G2, max_iter=10)`: run 1-WL color refinement on both graphs simultaneously; return `"MAYBE_ISOMORPHIC"` if the final color histograms match, or `"NOT_ISOMORPHIC"` at the first iteration where they differ.

(b) Test on $G_1 = C_6$ (6-cycle) and $G_2 = K_{3,3}$ (complete bipartite, 6 nodes, 9 edges). What does WL conclude? Are they actually isomorphic?

(c) Construct two non-isomorphic 3-regular graphs on 6 nodes ($K_{3,3}$ and the prism graph $Y_3$). Does 1-WL distinguish them? (Hint: both are 3-regular, so the degree-based first iteration is identical.)

(d) Explain why any MPNN with mean aggregation would fail to distinguish these graphs, while GIN with sum aggregation would succeed (or also fail, if WL itself fails).

---

**Exercise 5 ** - GAT Attention Mechanism**

Implement a single GAT attention head from scratch.

(a) For a graph with 5 nodes and edge set $E = \{(0,1),(1,2),(2,3),(3,4),(4,0),(0,2)\}$, compute attention coefficients $\alpha_{uv}$ for all edges.

(b) Use $d=4$, $d'=3$, random $W \in \mathbb{R}^{3 \times 4}$ and $\mathbf{a} \in \mathbb{R}^6$ (fixed seed). Initialize node features $H \in \mathbb{R}^{5 \times 4}$ randomly.

(c) Compute the attention logits $e_{uv} = \mathbf{a}^\top [\mathbf{z}_v \| \mathbf{z}_u]$ with LeakyReLU (slope 0.2) for all edges, then apply softmax over each node's neighborhood.

(d) Show the difference between GAT and GATv2: compute GATv2 attention scores $e_{uv}^{\text{v2}} = \mathbf{a}^\top \operatorname{LeakyReLU}(W[\mathbf{h}_v \| \mathbf{h}_u])$ and verify that the neighbor ranking for node 0 can differ between GAT and GATv2.

(e) Compute the updated node representations $H' = \sigma(\text{attention-weighted aggregation of }H)$.

---

**Exercise 6 ** - Over-Smoothing Dynamics**

Quantify over-smoothing as a function of GCN depth.

(a) Construct a Cora-like graph: generate a stochastic block model with 200 nodes, 4 blocks, within-block edge probability $p=0.15$, between-block probability $q=0.01$.

(b) Initialize node features as class one-hot vectors ($H^{[0]} = $ class assignments as a $200 \times 4$ matrix).

(c) Apply the GCN propagation $\hat{A}$ (no weight matrix, no nonlinearity - pure smoothing) for $L = 1, 2, 4, 8, 16, 32$ steps. Compute the Dirichlet energy $E(H^{[L]}) = \operatorname{tr}(H^{[L]\top} L H^{[L]})$ at each depth.

(d) Plot $E(H^{[L]})$ vs $L$ on a log scale. Fit an exponential $E \approx C \lambda^L$ and estimate $\lambda$.

(e) Compare with the theoretical rate: $\lambda \approx \lambda_2(L_{\text{rw}})^2$ where $\lambda_2$ is the Fiedler value of the random-walk Laplacian. Compute $\lambda_2$ numerically and compare.

---

**Exercise 7 *** - GIN Implementation and Expressiveness**

Implement GIN and compare its expressiveness to GCN on graph-level classification.

(a) Implement a 3-layer GIN with sum aggregation and 2-layer MLP update: $\mathbf{h}_v^{[l+1]} = \operatorname{MLP}^{[l]}((1+\varepsilon)\mathbf{h}_v^{[l]} + \sum_{u \in \mathcal{N}(v)}\mathbf{h}_u^{[l]})$ with $\varepsilon = 0$.

(b) Construct two non-isomorphic graphs $G_1 = C_6$ and $G_2 = C_3 \cup C_3$ (as computed in 7.2, WL cannot distinguish them without initial node features). Initialize all nodes with the same feature vector $\mathbf{x} = \mathbf{1} \in \mathbb{R}^d$.

(c) Show that with uniform initial features, neither GIN nor GCN can distinguish $G_1$ and $G_2$ (both produce identical graph-level sum representations). Why?

(d) Now add degree as an initial node feature: $\mathbf{x}_v = [d_v] \in \mathbb{R}^1$. Rerun both GIN and GCN. Does GIN now distinguish the two graphs? Does GCN?

(e) Generate a dataset of 500 random graphs (mix of Erdos-Renyi and stochastic block models); train GIN and GCN for graph-level binary classification. Report test accuracy. Verify that GIN \geq GCN accuracy.

---

**Exercise 8 *** - Graph Transformer with Positional Encoding**

Implement a simplified graph transformer layer with Laplacian positional encoding.

(a) For a graph with $n=20$ nodes (Barabasi-Albert model with $m=2$), compute the normalized Laplacian $L_{\text{sym}}$ and extract the first $k=4$ non-trivial eigenvectors $U_k \in \mathbb{R}^{20 \times 4}$.

(b) Initialize node features $X \in \mathbb{R}^{20 \times 8}$ randomly and augment with LapPE: $X^{\text{aug}} = [X \| U_k] \in \mathbb{R}^{20 \times 12}$.

(c) Implement one layer of scaled dot-product self-attention over all 20 nodes (fully connected, ignoring graph edges): $\operatorname{Attention}(Q, K, V) = \operatorname{softmax}(QK^\top / \sqrt{d_k})V$.

(d) Implement one GCN layer on the same graph. Visualize the attention matrices of both: which nodes attend to which? Does the graph transformer's attention recover some of the graph structure (do nearby nodes attend to each other more)?

(e) Compare: run 3 iterations of graph transformer attention vs 3 layers of GCN on the same random features. Compute the Dirichlet energy after each: which decreases faster? What does this reveal about over-smoothing in graph transformers vs GNNs?

---

## 15. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|-----------|
| MPNN framework | The unifying abstraction behind AlphaFold 2 (protein structure), drug screening, and materials design GNNs. Every new spatial GNN architecture published in 2024-2026 is a special case |
| GCN propagation rule | Directly computable from sparse adjacency; used in LightGCN powering Netflix/Pinterest/TikTok recommendation at billion-node scale |
| GraphSAGE inductive learning | Enables daily-updated embeddings for dynamic graphs (social networks, e-commerce) without retraining the full model |
| GAT / GATv2 | Learned, sparse attention patterns over structured data; used in molecular GNNs for drug-target interaction prediction (AstraZeneca, Recursion Pharmaceuticals) |
| WL expressiveness theorem | Theoretical bound establishing what structure any MPNN can detect; determines when to add structural features or upgrade to subgraph GNNs; guides architecture search |
| GIN sum aggregation | The foundation for most molecular property prediction models; identifies that all mainstream GCN deployments are sub-optimal in expressiveness |
| Over-smoothing analysis | Explains why 2-layer GNNs outperform 8-layer GNNs in most production deployments; directly informs depth tuning decisions |
| Over-squashing and graph rewiring | Motivates graph preprocessing in biochemistry (adding virtual bonds between distant atoms) and knowledge graph reasoning (adding transitive edges) |
| Graph Transformers (GPS, Graphormer) | State-of-the-art on quantum chemistry benchmarks (QM9, OGB-LSC); powering the next generation of molecular AI models |
| LapPE and RWSE | Standard input features for all graph foundation models; analogous to positional encodings in LLMs - without them, graph models cannot distinguish relative node positions |
| Graph RAG | Microsoft's production system (Azure AI Search) for document understanding queries requiring multi-hop reasoning; deployed across Office 365 and Microsoft 365 Copilot |
| Hierarchical pooling (DiffPool, MinCutPool) | Used in drug candidate scoring for proteins and polymer graphs where hierarchical structure (residues -> domains -> whole protein) is essential |
| Neighbor sampling (GraphSAGE, Cluster-GCN) | The key to training GNNs on billion-node graphs; enables GPU training without the full graph fitting in memory |

---

## 16. Conceptual Bridge

### Looking Backward: What This Section Builds On

Graph Neural Networks are the synthesis of all earlier mathematics in this curriculum. From **Chapter 2 Linear Algebra**, we use matrix multiplication (the propagation rule $\hat{A}H$), eigenvectors (the spectral derivation of GCN), and norms (Dirichlet energy). From **Chapter 3 Advanced Linear Algebra**, we use eigendecomposition (Laplacian positional encodings), PSD theory (the Laplacian is PSD, proven in 11-04), and the spectral theorem (graph Fourier transform). From **Chapter 4 Calculus**, we use gradients and the chain rule for backpropagation through GNN layers. From **Chapter 8 Optimization**, we use SGD variants and the mini-batch training strategies of 11. From **Chapter 11 Graph Theory** itself, we have built on 11-04's spectral theory (the GCN derivation via Chebyshev polynomials, over-smoothing as diffusion, Laplacian eigenmaps as positional encodings) and 11-03's algorithms (BFS neighborhoods as the receptive field, max-flow as the analogy to Cheeger's inequality).

### Looking Forward: What This Section Enables

**Chapter 12 Functional Analysis** generalizes the spectral ideas of GNNs to infinite-dimensional Hilbert spaces. The graph Fourier transform is the finite-dimensional analogue of the Fourier transform on $L^2(\mathbb{R})$; the Laplacian eigenvectors are the finite-dimensional analogue of Fourier modes. Functional analysis provides the rigorous framework for this extension, enabling kernel methods on graphs and the theoretical analysis of continuous-limit GNNs (graphons).

**Chapter 14 Math for Specific Models** will revisit GNN architectures in the context of equivariant neural networks - networks that respect symmetries (rotation, reflection, permutation) by design. Geometric GNNs for 3D molecular data (DimeNet, SE(3)-Transformers, NequIP) extend the MPNN framework with $SO(3)$-equivariant message functions, enabling prediction of orientation-dependent molecular properties.

**Chapter 21 Statistical Learning Theory** will study GNN generalization bounds: how many training graphs are needed for a GNN to generalize? The WL expressiveness hierarchy (7) directly informs these bounds - more expressive GNNs require more data to generalize.

**Chapter 22 Causal Inference** uses directed acyclic graphs (DAGs) - a special class of directed graphs studied in 11-01. The graph algorithms of 11-03 (topological sort for causal ordering) and the GNN architectures of 11-05 (for learning on causal graphs) connect these chapters.

```
GRAPH NEURAL NETWORKS - POSITION IN THE CURRICULUM
========================================================================

  Ch.02-03: Linear Algebra, Eigenvalues, SVD
       down
  Ch.11-04: Spectral Graph Theory
       |    [Laplacian, GCN spectral derivation,
       |     over-smoothing preview, LapPE]
       down
  +==========================================+
  |  Ch.11-05: Graph Neural Networks         |  <- YOU ARE HERE
  |                                          |
  |  MPNN -> GCN -> GraphSAGE -> GAT -> GIN     |
  |  WL Expressiveness -> Over-smoothing      |
  |  Graph Transformers -> GPS/Graphormer     |
  |  Applications: AlphaFold 2, Graph RAG    |
  +==========================================+
       down                    down
  Ch.11-06:            Ch.12: Functional Analysis
  Random Graphs        [Graph limits (graphons),
  [Graph models        kernel methods on graphs,
   for GNN             infinite-dimensional
   benchmarks]         spectral theory]
       down
  Ch.14: Math for Specific Models
  [Equivariant GNNs, SE(3)-Transformers,
   3D molecular AI, geometric deep learning]
       down
  Ch.21: Statistical Learning Theory
  [GNN generalization bounds,
   WL-based complexity, PAC learning on graphs]

========================================================================
```

---

[<- Back to Graph Theory](../README.md) | [Previous: Spectral Graph Theory <-](../04-Spectral-Graph-Theory/notes.md) | [Next: Random Graphs ->](../06-Random-Graphs/notes.md)

---

## Appendix A: Mathematical Derivations

### A.1 Proof that the GCN Propagation Rule is Permutation Equivariant

**Claim.** The GCN layer $H' = \sigma(\hat{A}HW)$ is permutation equivariant: for any permutation matrix $P$,
$$\sigma\!\left(\widehat{PAP^\top} \cdot PH \cdot W\right) = P \cdot \sigma\!\left(\hat{A}HW\right)$$

**Proof.** Let $\tilde{A}' = PAP^\top + I = P(A+I)P^\top = P\tilde{A}P^\top$. The degree matrix of $\tilde{A}'$ is:
$$\tilde{D}'_{ii} = \sum_j (P\tilde{A}P^\top)_{ij} = \sum_j \sum_k P_{ik}\tilde{A}_{kl}(P^\top)_{lj} = \sum_k \tilde{A}_{ki} = \tilde{D}_{ii}$$

Wait - let me be precise. Since $P$ is a permutation, $(P\tilde{A}P^\top)_{ij} = \tilde{A}_{\pi^{-1}(i),\pi^{-1}(j)}$ where $\pi$ is the permutation. Thus:

$$\tilde{D}'_{ii} = \sum_j \tilde{A}_{\pi^{-1}(i),\pi^{-1}(j)} = \tilde{D}_{\pi^{-1}(i),\pi^{-1}(i)} = (P\tilde{D}P^\top)_{ii}$$

So $\tilde{D}' = P\tilde{D}P^\top$, and:
$$\hat{A}' = \tilde{D}'^{-1/2}\tilde{A}'\tilde{D}'^{-1/2} = (P\tilde{D}P^\top)^{-1/2}(P\tilde{A}P^\top)(P\tilde{D}P^\top)^{-1/2}$$
$$= P\tilde{D}^{-1/2}P^\top \cdot P\tilde{A}P^\top \cdot P\tilde{D}^{-1/2}P^\top = P(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2})P^\top = P\hat{A}P^\top$$

Therefore:
$$\sigma\!\left(\hat{A}'(PH)W\right) = \sigma\!\left(P\hat{A}P^\top P H W\right) = \sigma\!\left(P\hat{A}HW\right) = P\sigma\!\left(\hat{A}HW\right) \quad \blacksquare$$

The last step uses the fact that $\sigma$ is applied element-wise and $P$ permutes rows: $\sigma(PZ) = P\sigma(Z)$ for any matrix $Z$ and element-wise $\sigma$.

### A.2 Universal Approximation of Injective Multiset Functions

The theoretical foundation of GIN rests on the following characterization:

**Theorem (Xu et al. 2019, following Zaheer et al. 2017).** Let $\mathcal{X}$ be a countable set. A function $f: \mathcal{M}(\mathcal{X}) \to \mathbb{R}^d$ on the space of finite multisets over $\mathcal{X}$ is injective if and only if there exists $\varphi: \mathcal{X} \to \mathbb{R}^d$ and $g: \mathbb{R}^d \to \mathbb{R}^d$ such that:

$$f(\mathcal{M}) = g\!\left(\sum_{x \in \mathcal{M}} \varphi(x)\right)$$

**Proof sketch (sufficiency).** Assume $\mathcal{X}$ is countable: enumerate as $\mathcal{X} = \{a_1, a_2, \ldots\}$. A multiset $\mathcal{M}$ over $\mathcal{X}$ is characterized by its multiplicity function $m: \mathcal{X} \to \mathbb{N}$ where $m(a_k)$ is the number of times $a_k$ appears. Choose $\varphi(a_k) = e^{-k}$ (unique real value per element). Then:

$$\sum_{x \in \mathcal{M}} \varphi(x) = \sum_k m(a_k) \cdot e^{-k}$$

This is an injective mapping from the multiplicity function $m$ to $\mathbb{R}$ (by the uniqueness of representations in base $e$, this holds for bounded multiplicities). Then $g$ can be any function that recovers $f$ from this sum.

**Necessity.** If $f$ is injective and we use sum aggregation $\sum \varphi(x)$, then different multisets must map to different sums. The existence of $\varphi$ and $g$ satisfying this is guaranteed by injectivity and the separating power of sums over countable sets. $\blacksquare$

**Consequence for GIN.** With a sufficiently expressive $\varphi$ (a deep MLP, by universal approximation) and $g$ (another deep MLP), GIN can represent any injective function on multisets of node features. This gives GIN the maximum discriminative power achievable by any MPNN.

### A.3 Dirichlet Energy Decay Rate

**Theorem.** Let $S = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ be the GCN propagation matrix with $\tilde{A} = A + I$. For any node feature matrix $H$:
$$E(SH) \leq \lambda_{\max}(S)^2 \cdot E(H)$$

where $E(H) = \operatorname{tr}(H^\top L_{\tilde{A}} H)$ and $L_{\tilde{A}} = I - S$ is the normalized Laplacian of $\tilde{A}$.

**Proof.** Write the eigendecomposition $S = U\Lambda U^\top$ where $U$ is orthonormal and $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$ with $|\lambda_i| \leq 1$ (since $S$ is doubly stochastic after normalization). Then:

$$E(SH) = \operatorname{tr}((SH)^\top L(SH)) = \operatorname{tr}(H^\top S^\top L S H)$$

Since $L_{\tilde{A}} = I - S$:
$$S^\top L S = S(I-S)S = S^2 - S^3$$

The eigenvalues of $S^2 - S^3$ are $\lambda_i^2 - \lambda_i^3 = \lambda_i^2(1-\lambda_i)$. For $\lambda_i \in [0,1]$ (which holds since $\hat{A}$ is a non-negative symmetric matrix with row sums at most 1 after self-loop normalization), we have $\lambda_i^2(1-\lambda_i) \leq \lambda_i^2 \leq \lambda_{\max}^2$.

Therefore: $\operatorname{tr}(H^\top S^\top L S H) \leq \lambda_{\max}^2 \operatorname{tr}(H^\top L H) = \lambda_{\max}^2 E(H)$. $\blacksquare$

For the GCN with self-loops, $\lambda_{\max}(S) < 1$ (strictly), so $E(H^{[L]}) \leq \lambda_{\max}^{2L} E(H^{[0]}) \to 0$ exponentially fast.

### A.4 Jacobian Bound for Over-Squashing

**Theorem (Alon & Yahav, 2021).** For a GNN with $L$ layers and bounded weight matrices $\lVert W^{[l]} \rVert \leq \alpha$ and Lipschitz activation $\sigma'(x) \leq \beta$:

$$\left\lVert\frac{\partial \mathbf{h}_v^{[L]}}{\partial \mathbf{x}_u}\right\rVert \leq C \cdot (\alpha\beta)^L \cdot \left(\hat{A}^L\right)_{vu}$$

where $C$ is a constant depending on the architecture.

**Key observation.** The entry $(\hat{A}^L)_{vu}$ counts the weighted number of walks of length $L$ from $u$ to $v$. For nodes $u$ and $v$ separated by a bottleneck edge (edge $(s,t)$ with large removal betweenness centrality), all walks from $u$ to $v$ must pass through $(s,t)$. The bottleneck effect:

If node $s$ has degree $d_s$, then at most $1/d_s$ of the walks that reach $s$ proceed to $t$ (in the normalized walk). Thus $(\hat{A}^L)_{vu} \leq (1/d_s)^{\lceil L / d(u,v) \rceil}$, which decays exponentially in $L$ when $d_s$ is large.

**Practical consequence.** Nodes separated by a high-degree hub (common in scale-free social networks) have near-zero Jacobian - the GNN cannot propagate useful gradient signal between them regardless of depth.

### A.5 Graph Isomorphism Network: Full Algorithm

**Algorithm: GIN for Graph Classification**

Input: Graph $G = (V, E, X)$ with $n$ nodes
Output: Graph embedding $\mathbf{h}_G \in \mathbb{R}^d$

1. Initialize: $\mathbf{h}_v^{[0]} = \mathbf{x}_v$ for all $v \in V$

2. For $l = 1, \ldots, L$:
   - For each node $v$: $\mathbf{h}_v^{[l]} = \operatorname{MLP}^{[l]}\!\left((1 + \varepsilon^{[l]}) \mathbf{h}_v^{[l-1]} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{[l-1]}\right)$
   - Apply batch normalization to $\{\mathbf{h}_v^{[l]}\}_{v \in V}$

3. Readout: for each layer $l$, compute layer-specific graph embedding:
   $$\mathbf{h}_G^{[l]} = \sum_{v \in V} \mathbf{h}_v^{[l]}$$

4. Concatenate: $\mathbf{h}_G = \left[\mathbf{h}_G^{[0]} \,\|\, \mathbf{h}_G^{[1]} \,\|\, \cdots \,\|\, \mathbf{h}_G^{[L]}\right]$

5. Apply MLP classifier: $\hat{y} = \operatorname{MLP}_{\text{pred}}(\mathbf{h}_G)$

**Why concatenate all layers?** Each layer captures patterns at a different structural scale: layer 0 is node features; layer 1 is immediate neighborhood; layer $l$ is the $l$-hop subgraph structure. Concatenating all layers allows the classifier to use patterns at all scales simultaneously, matching the JK-Net (Jumping Knowledge Networks) readout.

**Choice of $\varepsilon$.** In practice, $\varepsilon = 0$ works well - the MLP is expressive enough to learn the correct weighting. Learned $\varepsilon$ adds one scalar parameter per layer and sometimes improves performance on sparse graphs where the self-feature has a very different scale from the aggregated neighborhood.

### A.6 GPS Layer: Formal Specification

The GPS (General, Powerful, Scalable) Graph Transformer layer (Rampasek et al., 2022) is specified as follows.

**Input:** Node embeddings $H^{[l]} \in \mathbb{R}^{n \times d}$, adjacency $A$, positional encodings $\text{PE} \in \mathbb{R}^{n \times p}$

**MPNN sub-layer.** For each node $v$:
$$\tilde{\mathbf{h}}_v^{[l]} = \mathbf{h}_v^{[l]} + \operatorname{MPNN}\!\left(\mathbf{h}_v^{[l]}, \left\{\mathbf{h}_u^{[l]} : u \in \mathcal{N}(v)\right\}, A_{:,v}\right)$$

The MPNN can be any of: GCN, GINE, GAT, GATv2. GINE (GIN with Edge features) is:
$$\operatorname{GINE}(v) = \operatorname{MLP}\!\left((1+\varepsilon)\mathbf{h}_v + \sum_{u \in \mathcal{N}(v)} \operatorname{ReLU}\!\left(\mathbf{h}_u + \mathbf{e}_{uv}\right)\right)$$

**Transformer sub-layer.** Standard multi-head self-attention over all $n$ nodes:
$$\hat{H}^{[l]} = \tilde{H}^{[l]} + \operatorname{MultiHeadAttention}\!\left(\tilde{H}^{[l]} W_Q,\; \tilde{H}^{[l]} W_K,\; \tilde{H}^{[l]} W_V\right)$$

**Feed-forward + LayerNorm:**
$$H^{[l+1]} = \operatorname{LayerNorm}\!\left(\hat{H}^{[l]} + \operatorname{FFN}\!\left(\hat{H}^{[l]}\right)\right)$$

where $\operatorname{FFN}(\mathbf{h}) = W_2 \operatorname{ReLU}(W_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2$ is a 2-layer MLP.

**Complexity:** $O(m \cdot d)$ for the MPNN sub-layer (sparse), $O(n^2 \cdot d)$ for the Transformer sub-layer (dense). Total: $O((m + n^2) \cdot d)$. For sparse graphs where $m \ll n^2$, the Transformer dominates.

**Practical approximation.** For large graphs ($n > 10^4$), replace the full Transformer with a linear attention approximation (Performer, Longformer-style local+global attention, or Nystromformer). This reduces Transformer cost to $O(n \cdot d)$, making GPS scalable to large graphs while retaining the long-range attention benefit.

---

## Appendix B: Implementation Notes

### B.1 GCN in Matrix Form: Step-by-Step

Given a graph with adjacency matrix $A \in \mathbb{R}^{n \times n}$, node features $X \in \mathbb{R}^{n \times d_{\text{in}}}$, and weight matrices $W^{[0]} \in \mathbb{R}^{d_{\text{in}} \times d_1}$, $W^{[1]} \in \mathbb{R}^{d_1 \times d_{\text{out}}}$:

```
Step 1: Compute A = A + I_n
Step 2: Compute D = diag(A 1) = diag of row sums
Step 3: Compute D^{-1/2}: take 1/sqrt of diagonal entries
Step 4: Compute A = D^{-1/2} A D^{-1/2}   (sparse: O(m) entries)
Step 5: H^1 = ReLU(A X W^0)    - shape: (n, d_1)
Step 6: H^2 = softmax(A H^1 W^1) - shape: (n, d_out)   [for classification]
Step 7: Loss = -sum_{v in V_L} sum_c Y_{vc} log H^2_{vc}
Step 8: Backpropagate through H^2, H^1, W^1, W^0
```

**Sparsity.** $A$ for real-world graphs is extremely sparse (average degree 10 out of $n=10^6$ nodes means $\sim 10^{-5}$ fill). All operations involving $\hat{A}$ should use sparse matrix formats (CSR, COO). The computation $\hat{A}H$ for sparse $\hat{A}$ and dense $H$ costs $O(m \cdot d)$, not $O(n^2 \cdot d)$.

### B.2 Edge List Representation and Scatter Operations

In practice, GNNs are implemented using an **edge list** representation rather than adjacency matrices. The graph is stored as two arrays:

```
edge_index: shape (2, m) - edge_index[0] = source nodes, edge_index[1] = target nodes
edge_attr:  shape (m, d_e) - edge features
x:          shape (n, d_v) - node features
```

The message-passing computation $\mathbf{m}_v = \sum_{u \in \mathcal{N}(v)} M(\mathbf{h}_u, \mathbf{e}_{uv})$ is implemented as:

```python
# Gather source node features for all edges
messages = M(x[edge_index[0]], edge_attr)  # shape: (m, d')

# Scatter messages to target nodes (sum aggregation)
agg = torch.zeros(n, d').scatter_add(0, 
    edge_index[1].unsqueeze(-1).expand(-1, d'),
    messages)  # shape: (n, d')
```

This `scatter_add` operation (implemented in PyTorch Geometric and DGL) is the fundamental primitive of GPU-accelerated GNN training. It is equivalent to a sparse matrix-vector product $\hat{A}H$ but more flexible (supports arbitrary message functions).

### B.3 Mini-Batch Neighbor Sampling Implementation

```
GraphSAGE Training Step:
========================
1. Sample batch of target nodes: B \subseteq V, |B| = B_size
2. For l = L, L-1, ..., 1:
   - For each node v in current frontier:
     - Sample S_l neighbors: N_l(v) \subseteq N(v), |N_l(v)| = S_l
   - Current frontier = B \cup 1-hop(B) \cup ... \cup l-hop(B) [sampled]
3. Extract induced subgraph of all frontier nodes
4. Run GNN on induced subgraph (full-batch over small subgraph)
5. Compute loss only on target nodes B
6. Backpropagate
```

Total nodes per training step: $|B| \times \prod_{l=1}^L S_l$. For $|B|=256$, $L=2$, $S_1=S_2=10$: $256 \times 100 = 25,600$ nodes per step.

### B.4 Practical GNN Hyperparameter Guide

| Hyperparameter | Typical Range | Notes |
|----------------|---------------|-------|
| Number of layers $L$ | 2-4 | Start with 2; increase only if task requires long-range |
| Hidden dimension $d$ | 64-512 | Match to dataset size; 256 is a strong default |
| Dropout rate | 0.0-0.5 | 0.3-0.5 on feature/edge dropout helps regularize |
| Aggregation | Sum (GIN), Mean (GCN) | Sum for classification; mean for regression |
| Activation | ReLU, PReLU | PReLU slightly better on sparse graphs |
| Batch normalization | After each layer | Essential for deep GNNs (4+ layers) |
| Learning rate | 0.001-0.01 | Adam optimizer; cosine decay schedule |
| Neighbor sample sizes | $(25, 10)$ or $(15, 10, 5)$ | Larger $S_1$ improves accuracy; diminishing returns |
| Weight decay | $10^{-5}$-$10^{-4}$ | L2 regularization on weight matrices |
| Number of attention heads $K$ | 4-8 | For GAT; each head uses $d/K$ features |


---

## Appendix C: Connections to Classical Machine Learning

### C.1 GNNs and Kernel Methods

The GNN's computation can be viewed through the lens of kernel methods. Define the **graph neural tangent kernel (GNTK)** as the limiting kernel of an infinitely-wide GNN at initialization:

$$\Theta_{\text{GNN}}(G_1, G_2) = \lim_{d \to \infty} \langle \nabla_\theta f_\theta(G_1), \nabla_\theta f_\theta(G_2) \rangle$$

This connects GNNs to the NTK theory of Jacot et al. (2018) and provides a framework for analyzing GNN training dynamics. For graph classification:
- **Wide GNN $\equiv$ kernel SVM:** at infinite width, a GNN with random feature initialization and gradient descent is equivalent to a kernel machine with the GNTK
- **GNTK expressiveness:** the GNTK of a GIN-like architecture is exactly as expressive as 1-WL (matching the finite-depth GNN result)

**Graph Weisfeiler-Leman kernel (Shervashidze et al., 2011).** Before GNNs, graph kernels were the primary ML method for graphs. The WL kernel computes a feature vector from the color histograms at each WL refinement step:

$$K_{\text{WL}}(G_1, G_2) = \sum_{l=0}^L \langle \phi_l(G_1), \phi_l(G_2) \rangle$$

where $\phi_l(G)$ is the histogram of colors at WL iteration $l$. The WL kernel is equivalent to a GIN with fixed (non-learned) MLP - it is the non-parametric version of GIN.

**For AI:** Understanding GNNs as kernel machines allows the use of kernel theory (generalization bounds, bandwidth selection, representer theorem) to analyze GNN behavior - important for the theoretical foundations in Chapter 12 (Functional Analysis) and Chapter 21 (Statistical Learning Theory).

### C.2 GNNs and Belief Propagation

Loopy belief propagation (LBP) on probabilistic graphical models (Markov random fields, factor graphs) has exactly the same computational structure as MPNN message passing.

**Belief propagation update:**
$$\mu_{u \to v}^{(t+1)}(x_v) \propto \sum_{x_u} \psi_{uv}(x_u, x_v) \cdot \phi_u(x_u) \cdot \prod_{w \in \mathcal{N}(u) \setminus v} \mu_{w \to u}^{(t)}(x_u)$$

**MPNN message update:**
$$\mathbf{m}_{u \to v}^{[l+1]} = M\!\left(\mathbf{h}_v^{[l]}, \mathbf{h}_u^{[l]}, \mathbf{e}_{uv}\right)$$

The analogy:
- BP messages $\mu_{u \to v}(x_v)$ <-> GNN messages $\mathbf{m}_{u \to v}$
- BP potential $\psi_{uv}$ <-> GNN message function $M$
- BP product of incoming messages <-> GNN sum/product aggregation

**Implication:** GNNs can be viewed as learned approximations to belief propagation. Where BP computes exact marginals on trees (and approximate marginals on loopy graphs), GNNs learn to compute useful representations that may or may not correspond to probabilistic marginals. For tasks that arise from probabilistic inference on graphs (decoding, error correction, satisfiability), architectures inspired by BP (like LDPC decoders implemented as GNNs) achieve state-of-the-art results.

### C.3 Label Propagation vs GCN

Label propagation (Zhu et al., 2003) is a classical semi-supervised learning method that also propagates information across graph edges. The update rule:

$$F^{(t+1)} = \alpha S F^{(t)} + (1-\alpha) Y$$

where $S = D^{-1/2}AD^{-1/2}$, $F \in \mathbb{R}^{n \times C}$ is the label matrix (soft predictions), $Y$ is the observed labels (zero for unlabeled nodes), and $\alpha \in (0,1)$ controls the balance between propagation and fitting observed labels.

At convergence: $F^* = (1-\alpha)(I - \alpha S)^{-1} Y$ - a closed-form solution.

**GCN vs label propagation:**
- Label propagation: linear, closed-form, uses only graph structure (no node features)
- GCN: nonlinear, gradient-based, uses both node features and graph structure
- For feature-rich graphs: GCN consistently outperforms LP
- For feature-poor graphs (only node IDs): LP is competitive and much cheaper
- **APPNP (Gasteiger et al., 2019):** decouples the GNN feature transformation from the propagation step, applying propagation steps as a post-processing operation similar to LP. Achieves better performance than GCN on many node classification benchmarks by using more propagation steps without over-smoothing (due to the $\alpha$ mixing coefficient).

### C.4 Random Walk Methods and GNNs

Before GNNs, random walk methods (DeepWalk, Node2Vec) were the standard for learning node embeddings.

**DeepWalk (Perozzi et al., 2014).** Sample random walks from each node; treat the walk as a "sentence" and apply Skip-Gram (Word2Vec) to learn node embeddings that predict context nodes in the walk.

**Node2Vec (Grover & Leskovec, 2016).** Biased random walks with parameters $p$ (return probability) and $q$ (in-out probability) to balance BFS-like (homophily) and DFS-like (structural equivalence) exploration.

**Connection to GNNs.** The Skip-Gram objective for DeepWalk is equivalent to factorizing the matrix:

$$\log\!\left(\frac{\text{vol}(G)}{T} \sum_{t=1}^T A_t \cdot D^{-1}\right)$$

where $A_t$ is the $t$-step transition matrix. This is related to the graph's random walk kernel.

GNNs can incorporate random walk information through RWSE (7.6) or through the unsupervised pretraining objective of GraphSAGE (5.4). Modern practice often combines: pretrain with random walk objectives (fast, scalable), then fine-tune with GNN layers (expressive, task-specific).

### C.5 Over-smoothing and Spectral Perspective

From the spectral perspective of 11-04, over-smoothing has a clean frequency interpretation. A graph signal $\mathbf{x}$ can be decomposed into its graph Fourier components $\hat{x}_k = \mathbf{u}_k^\top \mathbf{x}$ (where $\mathbf{u}_k$ is the $k$-th Laplacian eigenvector). The GCN propagation $S = I - L_{\text{sym}}$ acts as a low-pass filter:

$$\widehat{(Sx)}_k = (1 - \lambda_k) \hat{x}_k$$

where $\lambda_k$ are the eigenvalues of $L_{\text{sym}}$. After $L$ layers:
$$\widehat{(S^L x)}_k = (1-\lambda_k)^L \hat{x}_k$$

- **Lowest frequency** ($k=1$, $\lambda_1 = 0$): constant component, amplified by factor 1 - perfectly preserved
- **Low frequencies** ($\lambda_k \approx 0$): slowly varying signals, amplified by $(1-\lambda_k)^L \approx 1$ - nearly preserved
- **High frequencies** ($\lambda_k \approx 2$): rapidly oscillating signals, suppressed by $(1-\lambda_k)^L \approx (-1)^L \to 0$ - exponentially suppressed

The class label signal is typically a **low-frequency** signal (connected nodes tend to have the same class in homophilic graphs). But even the label signal is eventually suppressed by many propagation steps - after enough layers, even the zero-frequency constant component dominates and all nodes collapse.

**Optimal depth.** The optimal number of layers $L^*$ balances two effects:
1. *More layers:* larger receptive field, more node context, better long-range information
2. *More layers:* stronger smoothing, loss of high-frequency discriminative features

For a graph with Fiedler value $\lambda_2$, the optimal depth is approximately $L^* \approx 1/\lambda_2$ (the mixing time of the random walk). For clustered graphs with small $\lambda_2$ (e.g., $\lambda_2 = 0.01$): $L^* \approx 100$ layers could be used. For expander-like graphs with large $\lambda_2$ (e.g., $\lambda_2 = 0.5$): $L^* \approx 2$ layers is optimal.


---

## Appendix D: GNN Training Recipes and Engineering Tricks

### D.1 Data Preprocessing for Graph ML

**Node feature normalization.** Unlike image pixels (naturally in $[0,255]$), node features in graphs span wildly different scales. Atom features in molecular graphs: atomic number (1-118), formal charge ($-4$ to $+4$), number of hydrogens ($0$-$4$). Normalizing to zero mean, unit variance per feature dimension is essential.

**Edge feature preprocessing.** Bond lengths in molecules range from 1.0 A (triple bond) to 1.5 A (single bond). If used as raw edge features, their small absolute range means they have little influence relative to bond type one-hot features. Apply normalization or use learned embeddings.

**Graph-level normalization.** For graph regression tasks (predicting molecular energy, which scales with the number of atoms), normalize the target $y_G$ by dividing by $n$ (atoms count). Otherwise, larger molecules trivially have larger predicted values.

**Degree-based normalization.** When using sum aggregation, node representations can grow with degree. Apply degree normalization as part of preprocessing: scale node features by $d_v^{-1/2}$ before input, matching the GCN normalization.

### D.2 Positional Encodings: Practical Considerations

**LapPE sign ambiguity.** Eigenvectors are defined up to sign: if $\mathbf{u}$ is an eigenvector of $L$, so is $-\mathbf{u}$. Naive use of eigenvectors as PEs means two identical graphs with differently-signed PEs produce different GNN outputs - violating permutation invariance over graph orientation.

Fix (Lim et al., 2022): during training, randomly flip the sign of each eigenvector independently at each mini-batch. This forces the GNN to be invariant to sign choice. At inference, use a consistent sign (e.g., always choose the sign such that the first non-zero element is positive).

**RWSE efficiency.** Computing $[A^p D^{-1}]_{vv}$ for $p = 1, \ldots, P$ requires matrix powers - expensive for large graphs. Efficient computation: start a random walk from node $v$; estimate the return probability $p_v^{(p)}$ by averaging over $R$ random walkers:

$$\hat{p}_v^{(p)} = \frac{1}{R} \sum_{r=1}^R \mathbf{1}[\text{walk } r \text{ returns to } v \text{ at step } p]$$

With $R=200$ walkers per node, this gives accurate RWSE estimates in $O(n \cdot P \cdot R)$ time.

**Eigenvalue degeneracy.** If two eigenvalues of $L$ are equal ($\lambda_k = \lambda_{k+1}$), the corresponding eigenspace is two-dimensional and any orthonormal basis for it is valid. This means LapPE is non-unique for graphs with symmetry (e.g., complete graphs, regular graphs). Degenerate eigenvectors encode no structural information - they should either be excluded from PEs or handled with random rotation augmentation.

### D.3 Loss Functions for Graph Tasks

**Node classification.** Cross-entropy over labeled nodes:
$$\mathcal{L} = -\frac{1}{|V_L|}\sum_{v \in V_L} \sum_{c=1}^C Y_{vc} \log \hat{Y}_{vc}$$

For class-imbalanced graphs (e.g., rare fraud nodes), use focal loss or weighted cross-entropy.

**Link prediction.** Binary cross-entropy with negative sampling:
$$\mathcal{L} = -\sum_{(u,v) \in E} \log \sigma(\mathbf{h}_u^\top \mathbf{h}_v) - \sum_{(u,v) \notin E} \log(1 - \sigma(\mathbf{h}_u^\top \mathbf{h}_v))$$

Negative sampling ratio (how many non-edges per positive edge): 1:1 to 1:5 is standard. Hard negative mining (sampling non-edges that are structurally close to positive edges) improves embedding quality at the cost of more complex sampling.

**Graph regression.** Mean absolute error (MAE) for quantum chemistry (physically meaningful, outlier-robust) or mean squared error (MSE) for property prediction where large errors are particularly bad. Normalize targets per-dataset: $(y - \mu_y) / \sigma_y$ for training, inverse-transform predictions.

**Contrastive learning on graphs.** Self-supervised GNNs (GraphCL, SimGRACE) maximize agreement between augmented views of the same graph:
$$\mathcal{L}_{\text{NT-Xent}} = -\log \frac{\exp(\operatorname{sim}(\mathbf{h}_G, \mathbf{h}_G') / \tau)}{\sum_{G^- \neq G} \exp(\operatorname{sim}(\mathbf{h}_G, \mathbf{h}_{G^-}) / \tau)}$$

where $\operatorname{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert)$ is cosine similarity, $\tau$ is temperature, and $\mathbf{h}_G'$ is the embedding of an augmented view (e.g., edge dropout, node feature masking). Graph augmentations must preserve semantic meaning - dropping a bond in a molecule changes its identity, so augmentation must be domain-aware.

### D.4 Debugging GNNs in Practice

**Symptom: GNN performs worse than MLP on node features alone.** Cause: the graph structure is not informative for the task (e.g., low homophily), or the GNN is over-smoothing and destroying the discriminative node features. Fix: verify homophily ratio $h = |{(u,v) \in E : y_u = y_v}| / |E|$; if $h < 0.2$, graph structure hurts. Use heterophilic GNNs (FAGCN, H2GCN) that can handle graphs where connected nodes have different labels.

**Symptom: Training loss decreases but validation loss explodes.** Cause: overfitting, likely due to GNN memorizing the training graph topology rather than learning transferable patterns. Fix: add dropout on node features (not just edges), reduce model capacity, add weight decay. For transductive semi-supervised learning, ensure the train/val split is performed correctly.

**Symptom: All node representations collapse to the same vector.** Cause: over-smoothing (too many layers) or numerical instability (exploding/vanishing gradients through $\hat{A}$). Fix: check Dirichlet energy after each layer; use PairNorm or GCNII; add gradient clipping.

**Symptom: GAT attention weights are all uniform ($\alpha_{uv} \approx 1/|\mathcal{N}(v)|$).** Cause: the attention vector $\mathbf{a}$ has collapsed - the model learned that uniform attention is safest. Often occurs with high learning rates or without proper initialization. Fix: initialize $\mathbf{a}$ with Xavier initialization; use a smaller learning rate for attention parameters; check for gradient vanishing in LeakyReLU (use a larger negative slope, e.g., $0.2$).

**Symptom: Training is extremely slow.** Cause: running GNN on the full large graph every step (full-batch training). Fix: switch to mini-batch training with neighbor sampling (GraphSAGE), graph partitioning (Cluster-GCN), or subgraph sampling (GraphSAINT). Profile GPU utilization - if below 70%, the bottleneck is data loading, not computation.


---

## Appendix E: Benchmark Datasets and Evaluation

### E.1 Standard Node Classification Benchmarks

| Dataset | Nodes | Edges | Features | Classes | Homophily | Task |
|---------|-------|-------|----------|---------|-----------|------|
| Cora | 2,708 | 5,429 | 1,433 | 7 | 0.81 | Transductive node clf |
| CiteSeer | 3,327 | 4,732 | 3,703 | 6 | 0.74 | Transductive node clf |
| PubMed | 19,717 | 44,338 | 500 | 3 | 0.80 | Transductive node clf |
| Amazon-Photo | 7,650 | 119,081 | 745 | 8 | 0.83 | Transductive node clf |
| Chameleon | 2,277 | 36,101 | 2,325 | 5 | 0.23 | Heterophilic node clf |
| Squirrel | 5,201 | 217,073 | 2,089 | 5 | 0.22 | Heterophilic node clf |
| OGB-arxiv | 169,343 | 1,166,243 | 128 | 40 | 0.65 | Large-scale node clf |
| OGB-products | 2,449,029 | 61,859,140 | 100 | 47 | 0.81 | Large-scale node clf |

**Evaluation protocol.** Standard split for Cora/CiteSeer/PubMed: 20 nodes per class for training, 500 for validation, 1000 for test (following Kipf & Welling, 2017). OGB uses official dataset-specific splits with multiple random seeds for reliable comparison.

### E.2 Standard Graph Classification Benchmarks

| Dataset | Graphs | Avg. nodes | Avg. edges | Classes | Domain |
|---------|--------|------------|------------|---------|--------|
| MUTAG | 188 | 17.9 | 19.8 | 2 | Molecular (mutagenicity) |
| PROTEINS | 1,113 | 39.1 | 72.8 | 2 | Protein function |
| IMDB-B | 1,000 | 19.8 | 96.5 | 2 | Social (movie collaboration) |
| COLLAB | 5,000 | 74.5 | 2,457.8 | 3 | Social (research collaboration) |
| OGB-molhiv | 41,127 | 25.5 | 27.5 | 2 | Molecular (HIV inhibition) |
| OGB-molpcba | 437,929 | 26.0 | 28.1 | 128 | Molecular (bioassays) |
| QM9 | 130,831 | 18.0 | 18.6 | - | Molecular (12 quantum properties) |

**Evaluation.** Graph classification: 10-fold cross-validation accuracy for TU datasets; ROC-AUC for OGB. Molecular regression (QM9): mean absolute error (MAE) per property, target unit in Angstroms, eV, Debye (domain-dependent).

### E.3 Long-Range Graph Benchmarks (LRGB)

Introduced by Dwivedi et al. (2022) to specifically test long-range dependency learning - tasks where local GNNs fail but graph transformers succeed.

| Dataset | Nodes | Task | Long-range property |
|---------|-------|------|---------------------|
| Peptides-func | 150K graphs, avg 150 nodes | Graph multi-label clf | Functional class depends on full peptide structure |
| Peptides-struct | 150K graphs, avg 150 nodes | Graph regression | 3D structural properties (end-to-end distances) |
| PCQM-Contact | 529K graphs | Link prediction | Molecular contact maps (spatial proximity) |
| COCO-SP | 123K graphs, avg 476 nodes | Node segmentation | Pixel graphs; semantic context is long-range |

**GNN vs Graph Transformer gap.** On Peptides-func (AP metric):
- GCN: 0.5930
- GIN: 0.6621
- GPS (GINE + Transformer): 0.7470
- GPS + RWSE: 0.7538

The $\sim 0.09$ gap between GIN and GPS demonstrates the clear advantage of global attention for long-range tasks.

### E.4 Choosing the Right Architecture

```
ARCHITECTURE SELECTION GUIDE
========================================================================

  Task requires long-range dependencies?
  +-- YES -> Graph Transformer (GPS, Graphormer)
  |         or Graph rewiring + MPNN
  +-- NO  down

  Graph has millions of nodes?
  +-- YES -> GraphSAGE (inductive) or Cluster-GCN/GraphSAINT (large)
  +-- NO  down

  Need to classify entire graphs?
  +-- YES -> GIN + sum pooling (expressiveness matters)
  |         or DiffPool/SAGPool for hierarchical structure
  +-- NO  down

  Need interpretable edge importance?
  +-- YES -> GAT or GATv2
  +-- NO  down

  Simple strong baseline needed first?
  +-- -> 2-layer GCN (fast to implement, often competitive)

  Has rich edge features (bond types, relation types)?
  +-- -> MPNN/GINE with edge message functions

========================================================================
```


---

## Appendix F: Notation Summary for This Section

| Symbol | Definition | First Used |
|--------|-----------|-----------|
| $G = (V, E)$ | Undirected graph with vertex set $V$, edge set $E$ | 2.1 |
| $n = \lvert V \rvert$, $m = \lvert E \rvert$ | Number of nodes and edges | 2.1 |
| $X \in \mathbb{R}^{n \times d_v}$ | Node feature matrix; row $v$ is $\mathbf{x}_v$ | 2.1 |
| $\mathcal{N}(v)$ | Neighborhood of node $v$: $\{u : (u,v) \in E\}$ | 1.2 |
| $d_v = \lvert\mathcal{N}(v)\rvert$ | Degree of node $v$ | 2.1 |
| $\mathbf{h}_v^{[l]} \in \mathbb{R}^d$ | Representation of node $v$ at layer $l$ | 3.1 |
| $H^{[l]} \in \mathbb{R}^{n \times d}$ | All node representations at layer $l$ | 4.2 |
| $A \in \{0,1\}^{n \times n}$ | Adjacency matrix | 2.1 |
| $\tilde{A} = A + I_n$ | Adjacency with self-loops | 4.2 |
| $D \in \mathbb{R}^{n \times n}$ | Degree matrix: $D_{ii} = d_i$ | 4.2 |
| $\tilde{D}$ | Degree matrix of $\tilde{A}$ | 4.2 |
| $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ | GCN propagation matrix | 4.2 |
| $L = D - A$ | Unnormalized Laplacian | 8.2 |
| $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$ | Normalized (symmetric) Laplacian | 10.2 |
| $E(H) = \operatorname{tr}(H^\top L H)$ | Dirichlet energy of $H$ | 8.2 |
| $\mathbf{m}_v^{[l]} \in \mathbb{R}^{d'}$ | Aggregated message at node $v$, layer $l$ | 3.1 |
| $M^{[l]}(\cdot)$ | Message function at layer $l$ | 3.1 |
| $U^{[l]}(\cdot)$ | Update function at layer $l$ | 3.1 |
| $R(\cdot)$ | Readout function for graph-level tasks | 3.1 |
| $W^{[l]} \in \mathbb{R}^{d \times d'}$ | Weight matrix at layer $l$ | 4.2 |
| $\alpha_{uv} \in [0,1]$ | GAT attention coefficient from $u$ to $v$ | 6.2 |
| $c_v^{(t)}$ | WL color of node $v$ at iteration $t$ | 7.2 |
| $\varepsilon^{[l]}$ | GIN learnable mixing parameter | 7.4 |
| $\operatorname{MAD}(H)$ | Mean Average Distance of node representations | 8.1 |
| $S \in \mathbb{R}^{n \times k}$ | DiffPool soft cluster assignment matrix | 9.3 |
| $\text{PE}_v \in \mathbb{R}^p$ | Positional encoding of node $v$ | 10.2 |
| $U_k \in \mathbb{R}^{n \times k}$ | First $k$ eigenvectors of $L_{\text{sym}}$ | 10.2 |
| $\phi(u,v)$ | Shortest-path distance between $u$ and $v$ | 10.4 |
| $b_\phi$ | Graphormer spatial encoding bias at distance $\phi$ | 10.4 |
| $P \in \Pi_n$ | $n \times n$ permutation matrix | 2.3 |
| $\boldsymbol{\pi} \in \mathbb{R}^n$ | Stationary distribution of random walk on $\tilde{A}$ | 8.1 |
| $\Theta_{\text{GNN}}(G_1, G_2)$ | Graph neural tangent kernel | C.1 |
| $K_{\text{WL}}(G_1, G_2)$ | Weisfeiler-Leman graph kernel | C.1 |

All vectors are column vectors by default ($\mathbf{h}_v \in \mathbb{R}^d$ means $d \times 1$). Matrix norms: $\lVert W \rVert_F$ (Frobenius), $\lVert W \rVert_2$ (spectral). Notation follows [docs/NOTATION_GUIDE.md](../../docs/NOTATION_GUIDE.md) throughout.


---

## Appendix G: Further Reading

### Foundational Papers (Must-Read)

1. **Scarselli et al. (2009)** - "The Graph Neural Network Model." *IEEE Transactions on Neural Networks.* Original GNN formulation with fixed-point iteration. Historical foundation.

2. **Defferrard, Bresson & Vandergheynst (2016)** - "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." *NeurIPS 2016.* ChebNet: polynomial spectral filters, making spectral GNNs computationally feasible.

3. **Kipf & Welling (2017)** - "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR 2017.* GCN: the simplification that started the field. Short, clear, highly cited.

4. **Hamilton, Ying & Leskovec (2017)** - "Inductive Representation Learning on Large Graphs." *NeurIPS 2017.* GraphSAGE: inductive learning, neighbor sampling, unsupervised objectives.

5. **Gilmer et al. (2017)** - "Neural Message Passing for Quantum Chemistry." *ICML 2017.* MPNN framework: the unified abstraction. The formalism most GNN papers use today.

6. **Velickovic et al. (2018)** - "Graph Attention Networks." *ICLR 2018.* GAT: learned attention weights for graphs. Widely deployed in production.

7. **Xu et al. (2019)** - "How Powerful are Graph Neural Networks?" *ICLR 2019.* GIN + WL expressiveness theorem: the theoretical foundation of GNN expressiveness.

### Depth Pathologies

8. **Li et al. (2018)** - "Deeper Insights into Graph Convolutional Networks for Semi-Supervised Classification." *AAAI 2018.* First formal analysis of over-smoothing.

9. **Alon & Yahav (2021)** - "On the Bottleneck of Graph Neural Networks and its Practical Implications." *ICLR 2021.* Over-squashing: Jacobian analysis and graph rewiring motivation.

10. **Chen et al. (2020)** - "Simple and Deep Graph Convolutional Networks." *ICML 2020.* GCNII: initial residual + identity mapping for training 64-layer GCNs.

### Graph Transformers

11. **Ying et al. (2021)** - "Do Transformers Really Perform Bad for Graph Representation?" *NeurIPS 2021.* Graphormer: spatial, central, and edge encodings for graph transformers. OGB-LSC winner.

12. **Rampasek et al. (2022)** - "Recipe for a General, Powerful, Scalable Graph Transformer." *NeurIPS 2022.* GPS framework: MPNN + Transformer combination with arbitrary PEs.

13. **Dwivedi et al. (2022)** - "Long Range Graph Benchmark." *NeurIPS 2022 Datasets.* LRGB: benchmarks specifically designed to test long-range dependency learning.

### Expressiveness

14. **Morris et al. (2019)** - "Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks." *AAAI 2019.* $k$-GNN: implementing $k$-WL as a neural network.

15. **Brody, Alon & Yahav (2022)** - "How Attentive are Graph Attention Networks?" *ICLR 2022.* GATv2: fixing the static attention problem in GAT.

### Applications

16. **Jumper et al. (2021)** - "Highly Accurate Protein Structure Prediction with AlphaFold." *Nature.* AlphaFold 2: the Evoformer and structure module. GNNs at scale for biology.

17. **Ying et al. (2018)** - "Graph Convolutional Neural Networks for Web-Scale Recommender Systems." *KDD 2018.* PinSage: GraphSAGE at 3 billion nodes. Engineering lessons for production GNNs.

18. **Edge et al. (2024)** - "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Microsoft Research.* Graph RAG: GNNs + LLMs for document understanding.

### Textbooks and Surveys

19. **Hamilton (2020)** - *Graph Representation Learning.* Synthesis Lectures on AI. The definitive textbook on GNNs; covers theory and practice comprehensively.

20. **Wu et al. (2022)** - "A Comprehensive Study on Large-Scale Graph Training." *NeurIPS 2022.* Empirical comparison of full-batch, neighbor sampling, and subgraph sampling methods.


---

## Appendix H: Quick Reference - Key Formulas

### GCN
$$H^{[l+1]} = \sigma\!\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{[l]} W^{[l]}\right), \quad \tilde{A} = A + I, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$$

### GAT Attention Coefficient
$$\alpha_{uv} = \frac{\exp\!\left(\operatorname{LeakyReLU}\!\left(\mathbf{a}^\top [W\mathbf{h}_v \| W\mathbf{h}_u]\right)\right)}{\sum_{k \in \mathcal{N}(v)} \exp\!\left(\operatorname{LeakyReLU}\!\left(\mathbf{a}^\top [W\mathbf{h}_v \| W\mathbf{h}_k]\right)\right)}$$

### GATv2 (Dynamic Attention)
$$e_{uv} = \mathbf{a}^\top \operatorname{LeakyReLU}\!\left(W\left[\mathbf{h}_v \| \mathbf{h}_u\right]\right)$$

### GIN
$$\mathbf{h}_v^{[l+1]} = \operatorname{MLP}^{[l]}\!\left((1 + \varepsilon^{[l]})\mathbf{h}_v^{[l]} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{[l]}\right)$$

### GraphSAGE Update
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(W^{[l]}\left[\mathbf{h}_v^{[l]} \,\|\, \frac{1}{|\mathcal{S}_v|}\sum_{u \in \mathcal{S}_v} \mathbf{h}_u^{[l]}\right]\right)$$

### GCNII (Deep GCN with Residual)
$$\mathbf{h}_v^{[l+1]} = \sigma\!\left(\left[(1-\alpha)\hat{A}\mathbf{h}_v^{[l]} + \alpha \mathbf{h}_v^{[0]}\right]\!\left[(1-\beta)I + \beta W^{[l]}\right]\right)$$

### DiffPool Coarsening
$$H^{(l+1)} = S^{(l)\top} Z^{(l)}, \qquad A^{(l+1)} = S^{(l)\top} A^{(l)} S^{(l)}$$

### Dirichlet Energy
$$E(H) = \sum_{(i,j) \in E} \lVert\mathbf{h}_i - \mathbf{h}_j\rVert_2^2 = \operatorname{tr}(H^\top L H)$$

### GPS Layer
$$H^{[l+1]} = \operatorname{LayerNorm}\!\left(H^{[l]} + \operatorname{MPNN}^{[l]}(H^{[l]}, A) + \operatorname{Attn}^{[l]}(H^{[l]})\right)$$

### Graphormer Attention with Spatial Bias
$$e_{uv} = \frac{(H W_Q)(H W_K)^\top_{uv}}{\sqrt{d_k}} + b_{\phi(u,v)}$$

### 1-WL Color Refinement
$$c_v^{(t+1)} = \operatorname{HASH}\!\left(c_v^{(t)},\; \left\{\!\left\{c_u^{(t)} : u \in \mathcal{N}(v)\right\}\!\right\}\right)$$

### Unsupervised GraphSAGE Loss
$$\mathcal{L}(u,v) = -\log\sigma\!\left(\mathbf{h}_u^\top \mathbf{h}_v\right) - Q\,\mathbb{E}_{v_n}\!\left[\log\!\left(1 - \sigma\!\left(\mathbf{h}_u^\top \mathbf{h}_{v_n}\right)\right)\right]$$

