# Graph Neural Networks: Mathematical Foundations

[← Previous: Spectral Graph Theory](../04-Spectral-Graph-Theory) | [Next: Functional Analysis →](../../12-Functional-Analysis)

## Overview

Graph Neural Networks (GNNs) learn representations of graph-structured data by combining graph topology with node/edge features through neural network architectures.

## Files in This Section

| File | Description |
|------|-------------|
| [examples.ipynb](examples.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## 1. The GNN Framework

### General Message Passing Neural Network (MPNN)

**Message Passing Update**:
$$h_v^{(k+1)} = \text{UPDATE}^{(k)}\left( h_v^{(k)}, \text{AGG}^{(k)}\left( \{ m_{uv}^{(k)} : u \in \mathcal{N}(v) \} \right) \right)$$

Where:

- $h_v^{(k)}$: Node $v$'s representation at layer $k$
- $m_{uv}^{(k)} = \text{MESSAGE}^{(k)}(h_u^{(k)}, h_v^{(k)}, e_{uv})$: Message from $u$ to $v$
- $\mathcal{N}(v)$: Neighborhood of node $v$

### Components

| Component | Function                  | Example          |
| --------- | ------------------------- | ---------------- |
| MESSAGE   | Compute edge messages     | $m_{uv} = W h_u$ |
| AGGREGATE | Combine neighbor messages | Sum, Mean, Max   |
| UPDATE    | Update node state         | MLP, GRU, LSTM   |

## 2. Graph Convolutional Network (GCN)

### Spectral Derivation

Starting from spectral convolution:
$$g_\theta \star x = U g_\theta(\Lambda) U^T x$$

With Chebyshev approximation and simplification:
$$H^{(l+1)} = \sigma\left( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} \right)$$

Where:

- $\tilde{A} = A + I_N$ (self-loops)
- $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$
- $W^{(l)}$: Trainable weight matrix

### Matrix Form

$$\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

**Per-node update**:
$$h_v^{(l+1)} = \sigma\left( W^{(l)} \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{h_u^{(l)}}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(v)|}} \right)$$

### Spectral Interpretation

GCN acts as a **low-pass filter**:
$$\hat{A} = I - \tilde{L}_{sym}$$

Eigenvalues of $\hat{A}$ are $1 - \tilde{\lambda}_i$, concentrated near 1 for small $\tilde{\lambda}_i$.

## 3. GraphSAGE (Sample and Aggregate)

### Inductive Learning

Unlike GCN, GraphSAGE:

- Learns aggregation functions
- Supports inductive inference on new nodes
- Uses neighborhood sampling

### Update Rule

$$h_v^{(k)} = \sigma\left( W^{(k)} \cdot \text{CONCAT}\left( h_v^{(k-1)}, \text{AGG}^{(k)}\left( \{h_u^{(k-1)} : u \in \mathcal{N}(v)\} \right) \right) \right)$$

### Aggregator Options

1. **Mean Aggregator**:
   $$\text{AGG}_{\text{mean}} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u$$

2. **LSTM Aggregator** (with random permutation):
   $$\text{AGG}_{\text{LSTM}} = \text{LSTM}([h_u : u \in \text{shuffle}(\mathcal{N}(v))])$$

3. **Pool Aggregator**:
   $$\text{AGG}_{\text{pool}} = \max_{u \in \mathcal{N}(v)} \sigma(W_{\text{pool}} h_u + b)$$

### Neighborhood Sampling

Sample fixed-size neighborhoods at each layer:
$$\mathcal{N}_{\text{sample}}(v) = \text{Sample}(\mathcal{N}(v), S)$$

Reduces computational complexity for dense graphs.

## 4. Graph Attention Networks (GAT)

### Attention Mechanism

**Attention coefficients**:
$$e_{ij} = a^T [\,W h_i \| W h_j\,]$$

**Softmax normalization**:
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(\text{LeakyReLU}(e_{ij}))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(e_{ik}))}$$

**Node update**:
$$h_i' = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j \right)$$

### Multi-Head Attention

$$h_i' = \|_{k=1}^{K} \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$$

Final layer uses averaging:
$$h_i' = \sigma\left( \frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} W^{(k)} h_j \right)$$

### Key Properties

| Property                | Benefit                        |
| ----------------------- | ------------------------------ |
| Learnable weights       | Adaptive importance            |
| Node-specific           | Different attention per node   |
| Implicit normalization  | No degree normalization needed |
| Permutation equivariant | Order-independent              |

## 5. Message Passing Framework

### Generalized Form

$$h_v^{(t+1)} = U_t\left( h_v^{(t)}, \bigoplus_{u \in \mathcal{N}(v)} M_t(h_v^{(t)}, h_u^{(t)}, e_{uv}) \right)$$

Where:

- $U_t$: Update function (neural network)
- $M_t$: Message function
- $\bigoplus$: Aggregation (permutation-invariant)

### Common Aggregations

**Sum**: $\bigoplus = \sum$ - Captures degree information
**Mean**: $\bigoplus = \frac{1}{|\mathcal{N}|}\sum$ - Degree normalized
**Max**: $\bigoplus = \max$ - Captures salient features
**Attention**: $\bigoplus = \sum \alpha_i \cdot$ - Weighted combination

### Expressiveness vs Efficiency Trade-off

| Model | Time Complexity | Expressiveness |
| ----- | --------------- | -------------- | ----------- | ----------- | ----------- | ------ |
| GCN   | $O(             | E              | \cdot d)$   | Low (≤1-WL) |
| GAT   | $O(             | E              | \cdot d +   | V           | \cdot d^2)$ | Medium |
| MPNN  | $O(             | E              | \cdot d^2)$ | High        |

## 6. Weisfeiler-Lehman Test and GNN Expressiveness

### 1-WL Test

Iterative color refinement:
$$c^{(k+1)}(v) = \text{HASH}\left( c^{(k)}(v), \{\!\{ c^{(k)}(u) : u \in \mathcal{N}(v) \}\!\} \right)$$

Two graphs are **1-WL equivalent** if they have same color distribution at convergence.

### GNN ≤ 1-WL

**Theorem**: Message passing GNNs cannot distinguish graphs that 1-WL cannot distinguish.

**Proof sketch**: GNN aggregation is similar to WL color refinement.

### GNN = 1-WL (Graph Isomorphism Network)

**GIN** achieves maximal expressiveness among MPNNs:
$$h_v^{(k)} = \text{MLP}^{(k)}\left( (1 + \epsilon^{(k)}) h_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} h_u^{(k-1)} \right)$$

Where $\epsilon$ is learnable or fixed.

### Beyond 1-WL

Higher-order GNNs using:

- k-tuples instead of nodes
- Higher-order WL tests
- Examples: k-GNN, folklore GNN

## 7. Graph Pooling and Readout

### Global Readout

$$h_G = \text{READOUT}(\{h_v : v \in V\})$$

**Options**:

- Sum: $h_G = \sum_v h_v$
- Mean: $h_G = \frac{1}{|V|} \sum_v h_v$
- Max: $h_G = \max_v h_v$
- Set2Set (attention-based)

### Hierarchical Pooling

**DiffPool**:
$$S^{(l)} = \text{softmax}(\text{GNN}_{pool}(A^{(l)}, H^{(l)}))$$
$$H^{(l+1)} = S^{(l)T} \text{GNN}_{embed}(A^{(l)}, H^{(l)})$$
$$A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}$$

**Top-K Pooling**:
Select top-k nodes based on learned scores.

### Graph-level Tasks

| Task           | Readout        | Loss           |
| -------------- | -------------- | -------------- |
| Classification | Sum/Mean + MLP | Cross-entropy  |
| Regression     | Sum/Mean + MLP | MSE            |
| Generation     | Autoencoder    | Reconstruction |

## 8. Over-Smoothing Problem

### Definition

As layer depth increases, node representations converge:
$$\lim_{k \to \infty} h_v^{(k)} = h_{\text{constant}}$$

### Theoretical Analysis

**Dirichlet energy decay**:
$$E(H^{(k)}) = \text{tr}(H^{(k)T} L H^{(k)})$$

For GCN: $E(H^{(k+1)}) \leq E(H^{(k)})$

### Mitigation Strategies

1. **Residual connections**: $H^{(l+1)} = H^{(l)} + \text{GNN}(H^{(l)})$
2. **Dense connections**: $H^{(l+1)} = \text{CONCAT}(H^{(0)}, ..., H^{(l)})$
3. **Skip connections**: Jump knowledge
4. **Normalization**: PairNorm, NodeNorm
5. **DropEdge**: Random edge dropout

## 9. Heterogeneous Graph Neural Networks

### Heterogeneous Graphs

$$G = (V, E, \phi, \psi)$$

Where:

- $\phi: V \to \mathcal{T}_V$ maps nodes to types
- $\psi: E \to \mathcal{T}_E$ maps edges to types

### Relational GCN (R-GCN)

$$h_v^{(l+1)} = \sigma\left( W_0^{(l)} h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{|\mathcal{N}_r(v)|} W_r^{(l)} h_u^{(l)} \right)$$

### Heterogeneous Graph Transformer (HGT)

**Multi-head attention per relation**:
$$\text{Attention}(s, e, t) = \text{softmax}\left( \frac{K(s) W_{\phi(e)}^{ATT} Q(t)^T}{\sqrt{d}} \right)$$

Where $\phi(e)$ is the edge type.

## 10. Temporal Graph Networks

### Dynamic Graphs

Graphs evolving over time: $G(t) = (V(t), E(t))$

### Temporal Encoding

**Time encoding**:
$$\Phi(t) = [\cos(\omega_1 t), \sin(\omega_1 t), ..., \cos(\omega_d t), \sin(\omega_d t)]$$

### TGN Architecture

1. **Message function**: $m_i(t) = \text{msg}(s_i(t^-), s_j(t^-), \Delta t, e_{ij})$
2. **Aggregator**: $\bar{m}_i(t) = \text{agg}(\{m_i(t) : \text{events involving } i\})$
3. **Memory update**: $s_i(t) = \text{mem}(s_i(t^-), \bar{m}_i(t))$

## Key Mathematical Properties

### Permutation Equivariance

For permutation matrix $P$:
$$f(PAP^T, PX) = P f(A, X)$$

Essential for valid graph learning.

### Locality

$k$-layer GNN: node representation depends only on $k$-hop neighborhood.

### Invariance for Graph-level Tasks

$$g(PAP^T, PX) = g(A, X)$$

Achieved through permutation-invariant readout.

## Summary Table

| Model     | Key Innovation       | Best For                            |
| --------- | -------------------- | ----------------------------------- |
| GCN       | Spectral convolution | Semi-supervised node classification |
| GraphSAGE | Inductive learning   | Large-scale, new nodes              |
| GAT       | Attention mechanism  | Heterogeneous importance            |
| GIN       | WL-equivalent        | Graph classification                |
| R-GCN     | Relation-specific    | Knowledge graphs                    |
| TGN       | Temporal memory      | Dynamic graphs                      |

## Exercises

1. **GCN Implementation**: Implement a 2-layer GCN from scratch using only NumPy and verify against PyTorch Geometric
2. **Aggregation Comparison**: Compare sum, mean, and max aggregation on node classification tasks and analyze when each works best
3. **Attention Visualization**: Implement GAT and visualize the learned attention weights on a citation network
4. **Expressiveness Analysis**: Construct pairs of non-isomorphic graphs that 1-WL (and hence standard GNNs) cannot distinguish
5. **Message Passing Framework**: Design a custom message passing layer that incorporates both node and edge features

## References

- Kipf & Welling (2017): GCN
- Hamilton et al. (2017): GraphSAGE
- Veličković et al. (2018): GAT
- Xu et al. (2019): GIN
- Gilmer et al. (2017): MPNN Framework
