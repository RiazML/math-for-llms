# Stochastic Optimization

> **Navigation**: [← 04-Constrained-Optimization](../04-Constrained-Optimization/) | [Optimization](../) | [06-Optimization-Landscape →](../06-Optimization-Landscape/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Stochastic optimization deals with optimization problems involving randomness, either in the objective function or constraints. In machine learning, stochastic methods are essential because we work with finite samples from unknown distributions and need to scale to massive datasets.

## Prerequisites

- Gradient descent
- Probability theory (expectations, variance)
- Convex optimization basics

## Learning Objectives

1. Understand stochastic gradient descent (SGD) theory
2. Analyze convergence of stochastic methods
3. Apply variance reduction techniques
4. Use mini-batch strategies effectively

---

## 1. The Stochastic Optimization Problem

### 1.1 Expected Risk Minimization

$$\min_{\mathbf{w}} F(\mathbf{w}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[f(\mathbf{w}; \mathbf{x}, y)]$$

We want to minimize expected loss over the data distribution $\mathcal{D}$, but we only have samples.

### 1.2 Empirical Risk Minimization (ERM)

$$\min_{\mathbf{w}} \hat{F}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n} f(\mathbf{w}; \mathbf{x}_i, y_i)$$

| Concept            | Description                                        |
| ------------------ | -------------------------------------------------- |
| Population risk    | $F(\mathbf{w})$ - true expected loss               |
| Empirical risk     | $\hat{F}(\mathbf{w})$ - average over training data |
| Generalization gap | $F(\mathbf{w}) - \hat{F}(\mathbf{w})$              |

---

## 2. Stochastic Gradient Descent (SGD)

### 2.1 The SGD Update

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \nabla f(\mathbf{w}_t; \mathbf{x}_{i_t}, y_{i_t})$$

where $i_t$ is randomly sampled from $\{1, \ldots, n\}$.

### 2.2 Unbiased Gradient Estimate

$$\mathbb{E}_{i}[\nabla f(\mathbf{w}; \mathbf{x}_i, y_i)] = \nabla \hat{F}(\mathbf{w})$$

The stochastic gradient is an **unbiased estimator** of the true gradient.

### 2.3 Gradient Variance

$$\text{Var}[\nabla f(\mathbf{w}; \mathbf{x}_i, y_i)] = \mathbb{E}[\|\nabla f_i - \nabla \hat{F}\|^2]$$

```
Full Gradient vs Stochastic Gradient:

True gradient ∇F(w)
       ↑
       │    ╱ stochastic gradient (noisy)
       │   ╱
       │  ╱
       │ ╱
       ●────→

SGD follows noisy directions but correct on average
```

---

## 3. Convergence Analysis

### 3.1 Assumptions

| Assumption        | Mathematical Form                                                                                                             |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| L-smooth          | $\|\nabla f(\mathbf{u}) - \nabla f(\mathbf{v})\| \leq L\|\mathbf{u} - \mathbf{v}\|$                                           |
| Bounded variance  | $\mathbb{E}[\|\nabla f_i - \nabla F\|^2] \leq \sigma^2$                                                                       |
| μ-strongly convex | $f(\mathbf{v}) \geq f(\mathbf{u}) + \nabla f(\mathbf{u})^T(\mathbf{v}-\mathbf{u}) + \frac{\mu}{2}\|\mathbf{v}-\mathbf{u}\|^2$ |

### 3.2 Convergence Rates

**Convex, non-smooth:**
$$\mathbb{E}[F(\bar{\mathbf{w}}_T)] - F(\mathbf{w}^*) = O\left(\frac{1}{\sqrt{T}}\right)$$

**Strongly convex:**
$$\mathbb{E}[\|\mathbf{w}_T - \mathbf{w}^*\|^2] = O\left(\frac{1}{T}\right)$$

with decreasing step size $\eta_t = O(1/t)$.

### 3.3 The Variance Problem

| Method | Per-iteration cost | Convergence         |
| ------ | ------------------ | ------------------- |
| GD     | $O(n)$             | Linear: $O(\rho^T)$ |
| SGD    | $O(1)$             | Sublinear: $O(1/T)$ |

SGD cannot achieve linear convergence due to gradient variance!

---

## 4. Learning Rate Schedules

### 4.1 Constant Learning Rate

$$\eta_t = \eta$$

- Does not converge to optimum
- Oscillates in a neighborhood
- Size of neighborhood $\propto \eta \sigma^2$

### 4.2 Decreasing Learning Rate

**$O(1/t)$ decay:**
$$\eta_t = \frac{\eta_0}{1 + \alpha t}$$

**$O(1/\sqrt{t})$ decay:**
$$\eta_t = \frac{\eta_0}{\sqrt{t}}$$

### 4.3 Step Decay

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Common: $\gamma = 0.1$, $s = $ epochs until decay

### 4.4 Cosine Annealing

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

---

## 5. Mini-Batch SGD

### 5.1 Mini-Batch Gradient

$$\mathbf{g}_t = \frac{1}{|B_t|}\sum_{i \in B_t} \nabla f(\mathbf{w}_t; \mathbf{x}_i, y_i)$$

where $B_t$ is a random subset of size $b$.

### 5.2 Variance Reduction

$$\text{Var}[\mathbf{g}_t] = \frac{\sigma^2}{b}$$

Larger batch $\Rightarrow$ lower variance $\Rightarrow$ more stable updates.

### 5.3 Batch Size Trade-offs

| Small batch                            | Large batch            |
| -------------------------------------- | ---------------------- |
| More noise                             | Less noise             |
| May generalize better                  | May overfit            |
| More iterations/epoch                  | Fewer iterations/epoch |
| Better GPU utilization (up to a point) | Diminishing returns    |

```
Gradient variance vs batch size:

Var(g)
   │╲
   │ ╲
   │  ╲
   │   ╲___________
   │
   └────────────────── batch size
        ↑
    diminishing returns
```

---

## 6. Variance Reduction Methods

### 6.1 Why Variance Reduction?

Standard SGD: $O(1/T)$ convergence (sublinear)
Variance-reduced SGD: $O(\rho^T)$ convergence (linear)

### 6.2 SVRG (Stochastic Variance Reduced Gradient)

**Algorithm:**

```
Every m iterations:
    Compute full gradient: μ = ∇F(w̃)
    Store snapshot: w̃ = w

Each iteration:
    Sample i uniformly
    g = ∇f_i(w) - ∇f_i(w̃) + μ
    w = w - η g
```

**Key insight:** $\mathbb{E}[\mathbf{g}] = \nabla F(\mathbf{w})$ and $\text{Var}[\mathbf{g}] \to 0$ as $\mathbf{w} \to \mathbf{w}^*$

### 6.3 SAGA

$$\mathbf{g}_t = \nabla f_i(\mathbf{w}_t) - \nabla f_i(\mathbf{w}_{[i]}) + \frac{1}{n}\sum_j \nabla f_j(\mathbf{w}_{[j]})$$

where $\mathbf{w}_{[i]}$ is the last iterate where $i$ was sampled.

### 6.4 Comparison

| Method | Storage | Convergence | Full gradient?  |
| ------ | ------- | ----------- | --------------- |
| SGD    | $O(d)$  | $O(1/T)$    | No              |
| SVRG   | $O(d)$  | Linear      | Every $m$ steps |
| SAGA   | $O(nd)$ | Linear      | No              |
| SAG    | $O(nd)$ | Linear      | No              |

---

## 7. Importance Sampling

### 7.1 Non-Uniform Sampling

Instead of uniform sampling, sample $i$ with probability $p_i$:

$$\mathbf{g} = \frac{1}{n p_i} \nabla f_i(\mathbf{w})$$

Still unbiased: $\mathbb{E}[\mathbf{g}] = \sum_i p_i \cdot \frac{1}{n p_i} \nabla f_i = \frac{1}{n}\sum_i \nabla f_i$

### 7.2 Optimal Sampling

Optimal probabilities (minimize variance):

$$p_i^* \propto \|\nabla f_i(\mathbf{w})\|$$

**Approximation:** Use Lipschitz constants
$$p_i \propto L_i$$

### 7.3 Practical Implementation

- Estimate importance weights periodically
- Use gradient norms from recent evaluations
- Balance overhead vs variance reduction

---

## 8. Parallelism in SGD

### 8.1 Synchronous SGD

```
All workers:
    Compute gradients on different batches

Aggregate:
    g = average of all gradients

Update:
    w = w - η g
```

**Issues:** Slowest worker determines speed (stragglers)

### 8.2 Asynchronous SGD

```
Each worker independently:
    Read current w
    Compute gradient g
    Update: w = w - η g
```

**Issues:** Stale gradients, but faster wall-clock time

### 8.3 Local SGD (Federated Learning)

```
Each worker:
    Run K local SGD steps

Periodically:
    Average all workers' models
```

---

## 9. Noise and Generalization

### 9.1 Implicit Regularization

SGD noise may help generalization:

- Escapes sharp minima
- Favors flat minima (better generalization)

### 9.2 Gradient Noise Scale

$$\text{Noise scale} = \frac{\eta \sigma^2}{b}$$

where $b$ is batch size.

### 9.3 The Temperature Analogy

SGD behaves like simulated annealing:

- High learning rate / small batch = high temperature
- Low learning rate / large batch = low temperature

```
Loss landscape:

        Sharp          Flat
        minimum        minimum
Loss    ╱╲            ___
       ╱  ╲          ╱   ╲
      ╱    ╲        ╱     ╲
─────╱      ╲──────╱       ╲─────

SGD noise helps escape sharp minima
→ finds flatter minima
→ better generalization
```

---

## 10. Practical Considerations

### 10.1 Shuffling

- **With replacement:** True SGD, may repeat samples
- **Without replacement:** Epoch-based, each sample once per epoch
- Random reshuffling each epoch often better in practice

### 10.2 Data Augmentation

Effectively increases dataset size:
$$\tilde{F}(\mathbf{w}) = \mathbb{E}_{\mathbf{x}, \tau}[f(\mathbf{w}; \tau(\mathbf{x}), y)]$$

where $\tau$ is a random transformation.

### 10.3 Gradient Accumulation

For limited memory, accumulate gradients over micro-batches:

```
for micro_batch in split(batch):
    g += compute_gradient(micro_batch)
g = g / num_micro_batches
update(w, g)
```

---

## 11. Beyond SGD

### 11.1 SGD with Momentum

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f_{i_t}(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

Momentum averages gradients, reducing variance.

### 11.2 Adam and Adaptive Methods

Adaptive methods adjust learning rates per-parameter:

- Less sensitive to learning rate choice
- May converge to different solutions than SGD

### 11.3 When to Use What

| Scenario           | Recommendation        |
| ------------------ | --------------------- |
| Deep learning      | SGD+Momentum or Adam  |
| Convex, finite sum | SVRG, SAGA            |
| Very large scale   | Asynchronous SGD      |
| Limited memory     | Gradient accumulation |

---

## 12. Summary

| Concept             | Key Point                                 |
| ------------------- | ----------------------------------------- |
| SGD                 | Unbiased gradient estimate, $O(1)$ cost   |
| Convergence         | Sublinear without variance reduction      |
| Mini-batch          | Reduces variance by factor $1/b$          |
| SVRG/SAGA           | Linear convergence via variance reduction |
| Importance sampling | Non-uniform sampling reduces variance     |
| Noise               | May help generalization                   |

**Key insight:** The variance-bias trade-off in SGD is fundamental to understanding deep learning optimization and generalization.

---

## Exercises

1. **SGD Variance**: For a dataset with $n$ samples, show that the variance of the mini-batch gradient estimate is $\sigma^2/b$ where $b$ is batch size.

2. **Convergence Rate**: Prove that SGD with constant learning rate $\eta$ cannot converge to the exact optimum, but only to a neighborhood of size $O(\eta\sigma^2)$.

3. **SVRG Implementation**: Implement SVRG for ridge regression and compare convergence to vanilla SGD. Plot the optimization gap vs iterations.

4. **Learning Rate Schedules**: Compare $O(1/t)$ and $O(1/\sqrt{t})$ learning rate schedules on a strongly convex problem. Which converges faster?

5. **Importance Sampling**: For a dataset where 10% of samples have 10x larger gradients, derive the optimal sampling probabilities and expected variance reduction.

---

## References

1. Bottou, Curtis, Nocedal - "Optimization Methods for Large-Scale ML"
2. Johnson & Zhang - "Accelerating SGD using SVRG"
3. Defazio et al. - "SAGA: A Fast Incremental Gradient Method"
