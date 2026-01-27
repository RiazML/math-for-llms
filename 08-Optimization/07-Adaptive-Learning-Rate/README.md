# Adaptive Learning Rate Methods

## Introduction

Adaptive learning rate methods automatically adjust the step size for each parameter based on historical gradient information. These methods are crucial for training deep neural networks efficiently, eliminating much of the manual tuning required for vanilla SGD.

## Prerequisites

- Gradient descent basics
- Exponential moving averages
- Matrix operations

## Learning Objectives

1. Understand momentum-based methods
2. Master adaptive methods (AdaGrad, RMSProp, Adam)
3. Learn when to use which optimizer
4. Implement and tune these methods

---

## 1. Momentum

### 1.1 The Problem with Vanilla SGD

- Oscillates in steep directions
- Slow progress in shallow directions
- Sensitive to learning rate

### 1.2 Momentum Update

Accumulate velocity in gradient direction:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

```
Effect of Momentum:

Without momentum:          With momentum:
   ↓                         ↓
  ↙↘                        ↓
 ↗  ↙                       ↓
  ↘↗                        ↓
   ●                        ●

Oscillating path           Smooth path
```

### 1.3 Effective Learning Rate

Over time, momentum amplifies consistent gradients:

$$\mathbf{v}_t = \sum_{i=0}^{t} \beta^{t-i} \nabla f(\mathbf{w}_i)$$

For constant gradient $\mathbf{g}$: $\mathbf{v}_\infty = \frac{\mathbf{g}}{1 - \beta}$

---

## 2. Nesterov Accelerated Gradient (NAG)

### 2.1 Look-Ahead Gradient

Compute gradient at the "look-ahead" position:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \nabla f(\mathbf{w}_t - \eta \beta \mathbf{v}_{t-1})$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_t$$

### 2.2 Intuition

```
Momentum:    Compute gradient at current position
             ●→→→→→→ (momentum)
                 ↓ (gradient)

Nesterov:    First apply momentum, then compute gradient
             ●→→→→→→○ (look-ahead)
                    ↓ (gradient)

Nesterov "corrects" the momentum direction
```

### 2.3 Convergence

For convex functions with Lipschitz gradients:

- SGD: $O(1/\sqrt{T})$
- Momentum: $O(1/\sqrt{T})$ (same rate, better constant)
- Nesterov: $O(1/T^2)$ for deterministic, accelerated convergence

---

## 3. AdaGrad

### 3.1 Per-Parameter Learning Rates

Adapt learning rate for each parameter based on historical gradients:

$$\mathbf{G}_t = \mathbf{G}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \mathbf{g}_t$$

where $\odot$ is element-wise multiplication.

### 3.2 Key Properties

| Property          | Behavior                    |
| ----------------- | --------------------------- |
| Frequent features | Smaller learning rate       |
| Rare features     | Larger learning rate        |
| Automatic         | No per-parameter tuning     |
| Problem           | Learning rate → 0 over time |

### 3.3 Use Cases

- Sparse data (NLP, recommendations)
- Features with varying frequencies
- NOT for deep learning (learning rate decay too aggressive)

---

## 4. RMSProp

### 4.1 Exponential Moving Average

Fix AdaGrad's aggressive decay with exponential moving average:

$$\mathbf{v}_t = \rho \mathbf{v}_{t-1} + (1-\rho) \mathbf{g}_t \odot \mathbf{g}_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t$$

Typical: $\rho = 0.9$, $\epsilon = 10^{-8}$

### 4.2 Comparison with AdaGrad

```
AdaGrad G_t:     RMSProp v_t:
Accumulates      Exponential average
forever          (forgets old gradients)

G_t → ∞          v_t stays bounded
η/√G_t → 0       η/√v_t stays reasonable
```

### 4.3 Effect on Optimization

Divides by RMS of recent gradients:

- High variance direction → smaller step
- Low variance direction → larger step
- Adapts to local curvature

---

## 5. Adam (Adaptive Moment Estimation)

### 5.1 Combining Momentum and Adaptive LR

Adam combines:

1. **First moment** (momentum): $\mathbf{m}_t$
2. **Second moment** (RMSProp): $\mathbf{v}_t$

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1) \mathbf{g}_t$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2) \mathbf{g}_t^2$$

### 5.2 Bias Correction

Initial moments are biased toward zero. Correct:

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$
$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

### 5.3 Adam Update

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### 5.4 Adam Algorithm

```
Initialize m = 0, v = 0, t = 0

For each step:
    t = t + 1
    g = ∇f(w)                              # Gradient
    m = β₁ m + (1 - β₁) g                  # First moment
    v = β₂ v + (1 - β₂) g²                 # Second moment
    m̂ = m / (1 - β₁ᵗ)                      # Bias correction
    v̂ = v / (1 - β₂ᵗ)                      # Bias correction
    w = w - η m̂ / (√v̂ + ε)                # Update
```

---

## 6. Adam Variants

### 6.1 AdamW (Weight Decay)

Standard Adam couples weight decay with adaptive learning rate incorrectly. AdamW fixes this:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \mathbf{w}_t \right)$$

Decoupled weight decay is applied directly, not through gradient.

### 6.2 AMSGrad

Address potential non-convergence of Adam:

$$\hat{\mathbf{v}}_t = \max(\hat{\mathbf{v}}_{t-1}, \mathbf{v}_t)$$

Uses maximum of all past second moments.

### 6.3 RAdam (Rectified Adam)

Dynamically adjusts learning rate based on variance of adaptive learning rate:

$$\rho_t = \rho_\infty - \frac{2t\beta_2^t}{1 - \beta_2^t}$$

where $\rho_\infty = \frac{2}{1-\beta_2} - 1$

Uses adaptive LR only when variance is low enough.

### 6.4 AdaFactor

Memory-efficient: factors second moment into row and column statistics:

$$\mathbf{v}_t \approx \mathbf{r}_t \mathbf{c}_t^T$$

Reduces memory from $O(mn)$ to $O(m + n)$.

---

## 7. Comparison of Methods

### 7.1 Summary Table

| Method   | First Moment | Second Moment  | Bias Correction | Memory  |
| -------- | ------------ | -------------- | --------------- | ------- |
| SGD      | ✗            | ✗              | ✗               | $O(d)$  |
| Momentum | ✓            | ✗              | ✗               | $O(d)$  |
| AdaGrad  | ✗            | ✓ (cumulative) | ✗               | $O(d)$  |
| RMSProp  | ✗            | ✓ (EMA)        | ✗               | $O(d)$  |
| Adam     | ✓            | ✓ (EMA)        | ✓               | $O(2d)$ |

### 7.2 When to Use What

| Scenario               | Recommended Method       |
| ---------------------- | ------------------------ |
| Simple convex          | SGD                      |
| Deep learning (vision) | SGD + Momentum, or AdamW |
| NLP / Transformers     | Adam / AdamW             |
| Sparse gradients       | AdaGrad, Adam            |
| Limited tuning time    | Adam (robust defaults)   |
| Best generalization    | SGD + Momentum (often)   |

### 7.3 Learning Rate Guidelines

| Method         | Typical $\eta$              |
| -------------- | --------------------------- |
| SGD            | 0.01 - 0.1                  |
| SGD + Momentum | 0.01 - 0.1                  |
| Adam           | 0.001 (1e-3)                |
| AdamW          | 0.001 with $\lambda$ = 0.01 |

---

## 8. Learning Rate Schedules

### 8.1 Common Schedules

**Step decay:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**Exponential decay:**
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

**Cosine annealing:**
$$\eta_t = \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2} \left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Warmup:**
$$\eta_t = \eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} \quad \text{for } t < t_{\text{warmup}}$$

### 8.2 Warmup + Decay

```
Learning Rate Schedule:

η  ↑
   │     ╱───────────╲
   │    ╱             ╲
   │   ╱               ╲
   │  ╱                 ╲
   │ ╱                   ╲___
   └────────────────────────→ t
     warmup    decay
```

### 8.3 One-Cycle Policy

1. Increase LR from low to high
2. Decrease LR from high to very low
3. Can achieve faster convergence

---

## 9. Practical Considerations

### 9.1 Hyperparameter Sensitivity

| Parameter                 | Sensitivity | Notes              |
| ------------------------- | ----------- | ------------------ |
| $\eta$ (learning rate)    | HIGH        | Most important     |
| $\beta_1$ (momentum)      | LOW         | 0.9 usually fine   |
| $\beta_2$ (second moment) | LOW         | 0.999 usually fine |
| $\epsilon$                | VERY LOW    | 1e-8 default       |

### 9.2 Debugging Tips

1. **Loss explodes:** LR too high
2. **Loss plateaus:** LR too low, or stuck in local min
3. **Oscillating loss:** LR too high, or batch too small
4. **NaN loss:** Numerical issues, check gradients

### 9.3 Adam vs SGD Debate

**Adam advantages:**

- Fast initial progress
- Less sensitive to hyperparameters
- Good for transformers/NLP

**SGD advantages:**

- Often better generalization
- Simpler, fewer hyperparameters
- Preferred for vision (with proper tuning)

---

## 10. Beyond First-Order Methods

### 10.1 Second-Order Approximations

**AdaHessian:** Use diagonal Hessian approximation

**K-FAC:** Kronecker-factored curvature approximation

**Shampoo:** Structured preconditioning

### 10.2 Memory-Efficient Methods

For very large models:

- **8-bit Adam:** Quantized optimizer states
- **Gradient checkpointing:** Trade compute for memory
- **AdaFactor:** Factored second moments

---

## 11. Summary

| Concept     | Key Point                                           |
| ----------- | --------------------------------------------------- |
| Momentum    | Accumulates gradient direction; dampens oscillation |
| AdaGrad     | Per-parameter LR; good for sparse data              |
| RMSProp     | EMA of squared gradients; non-aggressive decay      |
| Adam        | Combines momentum + adaptive LR + bias correction   |
| AdamW       | Decoupled weight decay; often better than Adam      |
| LR schedule | Start high, decay over training                     |

**Key insight:** Adaptive methods provide good defaults, but SGD with proper tuning often achieves best final performance.

---

## References

1. Kingma & Ba - "Adam: A Method for Stochastic Optimization"
2. Loshchilov & Hutter - "Decoupled Weight Decay Regularization" (AdamW)
3. Ruder - "An Overview of Gradient Descent Optimization Algorithms"
4. Smith - "Cyclical Learning Rates for Training Neural Networks"
