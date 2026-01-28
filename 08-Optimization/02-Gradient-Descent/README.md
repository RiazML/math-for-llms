# Gradient Descent Methods

## Introduction

Gradient descent is the workhorse of machine learning optimization. Understanding its variants, convergence properties, and hyperparameter tuning is essential for training any ML model. This module covers vanilla gradient descent through advanced adaptive methods.

## Prerequisites

- Multivariable calculus (gradients)
- Convex optimization basics
- Linear algebra

## Learning Objectives

1. Understand gradient descent mechanics
2. Analyze convergence properties
3. Master adaptive learning rate methods
4. Apply momentum and acceleration
5. Choose appropriate optimizers for ML tasks

---

## 1. Vanilla Gradient Descent

### 1.1 The Update Rule

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$$

where:

- $\mathbf{w}_t$: parameters at step $t$
- $\eta$: learning rate (step size)
- $\nabla f(\mathbf{w}_t)$: gradient at current point

### 1.2 Intuition

```
Gradient Descent on Loss Surface:

f(w)
 │╲
 │ ╲  w₀ (start)
 │  ╲  ↓
 │   ╲ w₁
 │    ╲ ↓
 │     ╲w₂
 │      ╲↓
 │       ●  w* (minimum)
 └─────────── w

Move in direction of steepest descent (negative gradient)
```

### 1.3 Batch vs Mini-batch vs Stochastic

| Type           | Gradient        | Pros          | Cons                        |
| -------------- | --------------- | ------------- | --------------------------- |
| **Batch GD**   | Full dataset    | Stable, exact | Slow, memory-intensive      |
| **Mini-batch** | Subset (32-256) | Balanced      | Hyperparameter (batch size) |
| **SGD**        | Single sample   | Fast updates  | Noisy, high variance        |

$$\nabla f(\mathbf{w}) \approx \frac{1}{|B|} \sum_{i \in B} \nabla f_i(\mathbf{w})$$

---

## 2. Convergence Analysis

### 2.1 For Convex Functions

**Condition:** $f$ is convex and $L$-smooth ($\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$)

**Theorem:** With $\eta \leq 1/L$:

$$f(\mathbf{w}_T) - f(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|^2}{2\eta T}$$

**Rate:** $O(1/T)$ convergence

### 2.2 For Strongly Convex Functions

**Condition:** $f$ is $\mu$-strongly convex ($\nabla^2 f \succeq \mu I$)

**Theorem:** With $\eta = 1/L$:

$$\|\mathbf{w}_T - \mathbf{w}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T \|\mathbf{w}_0 - \mathbf{w}^*\|^2$$

**Rate:** Linear (exponential) convergence

### 2.3 Condition Number

$$\kappa = \frac{L}{\mu}$$

- **Well-conditioned:** $\kappa$ small → fast convergence
- **Ill-conditioned:** $\kappa$ large → slow convergence

```
Well-conditioned (κ ≈ 1):    Ill-conditioned (κ >> 1):

      ○                            ─────○─────
     ╱│╲                                │
    ╱ │ ╲                               │
   ╱  │  ╲                         ellipse
   circular                        (elongated)
```

---

## 3. Learning Rate Selection

### 3.1 Effects of Learning Rate

| $\eta$     | Effect                        |
| ---------- | ----------------------------- |
| Too small  | Very slow convergence         |
| Just right | Smooth convergence to optimum |
| Too large  | Oscillation, divergence       |

```
Too small:        Just right:       Too large:
●                 ●                 ●
 ╲                 ╲               ╱ ╲
  ╲                 ╲             ╱   ╲
   ╲                 ╲           ●     ●
    ╲                 ●               ╱ ╲
     ●              minimum          diverge
(many steps)      (efficient)
```

### 3.2 Learning Rate Schedules

**Step decay:**
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**Exponential decay:**
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

**Inverse time decay:**
$$\eta_t = \frac{\eta_0}{1 + \lambda t}$$

**Cosine annealing:**
$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_0 - \eta_{min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

---

## 4. Momentum

### 4.1 Classical Momentum

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla f(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

- $\gamma$: momentum coefficient (typically 0.9)
- $\mathbf{v}$: velocity (accumulated gradient)

### 4.2 Intuition

```
Without momentum:          With momentum:
●─●                        ●
  │                         ╲
  ●─●                        ╲
    │                         ╲
    ●─●                        ╲
      │                         ╲
      ●                          ●
(oscillates)                (accelerates)
```

Momentum:

- Accelerates in consistent directions
- Dampens oscillations
- Helps escape shallow local minima

### 4.3 Nesterov Accelerated Gradient (NAG)

"Look ahead" before computing gradient:

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla f(\mathbf{w}_t - \gamma \mathbf{v}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

Evaluates gradient at the "lookahead" position.

---

## 5. Adaptive Learning Rates

### 5.1 AdaGrad

Adapt learning rate per-parameter based on history:

$$g_t = \nabla f(\mathbf{w}_t)$$
$$G_t = G_{t-1} + g_t \odot g_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t$$

- Parameters with large gradients get smaller updates
- Good for sparse data
- **Problem:** Learning rate only decreases

### 5.2 RMSProp

Fix AdaGrad's diminishing learning rate:

$$E[g^2]_t = \rho E[g^2]_{t-1} + (1-\rho) g_t^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

- $\rho$: decay rate (typically 0.9)
- Exponential moving average of squared gradients

### 5.3 Adam (Adaptive Moment Estimation)

Combines momentum and RMSProp:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

**Bias correction:**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update:**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Default hyperparameters:**

- $\beta_1 = 0.9$ (momentum)
- $\beta_2 = 0.999$ (RMSProp)
- $\epsilon = 10^{-8}$

---

## 6. Adam Variants

### 6.1 AdamW (Weight Decay)

Decouple weight decay from adaptive learning rate:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \mathbf{w}_t\right)$$

Better regularization than L2 in Adam.

### 6.2 AMSGrad

Fix potential non-convergence of Adam:

$$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$$

Use maximum of all past $v_t$ values.

### 6.3 LAMB (Layer-wise Adaptive Moments)

Scale updates by layer norm ratio:

$$r = \frac{\|\mathbf{w}\|}{\|\text{Adam update}\|}$$

Good for large batch training.

---

## 7. Comparison of Optimizers

| Optimizer    | Adaptive | Momentum | Best For             |
| ------------ | -------- | -------- | -------------------- |
| SGD          | ✗        | ✗        | Simple, theory       |
| SGD+Momentum | ✗        | ✓        | CNNs, well-tuned     |
| AdaGrad      | ✓        | ✗        | Sparse data, NLP     |
| RMSProp      | ✓        | ✗        | RNNs, non-stationary |
| Adam         | ✓        | ✓        | Default choice       |
| AdamW        | ✓        | ✓        | Transformers         |

```
Optimizer Selection:

Start with Adam (η=0.001)
        │
        ▼
  Good results? ─Yes──▶ Done
        │
       No
        │
        ▼
  Try SGD+Momentum (tune η)
        │
        ▼
  Better? ─Yes──▶ Use SGD
        │
       No
        │
        ▼
  Try AdamW / adjust hyperparams
```

---

## 8. Practical Considerations

### 8.1 Gradient Clipping

Prevent exploding gradients:

**By value:**
$$g = \text{clip}(g, -\tau, \tau)$$

**By norm:**
$$g = \frac{g}{\max(1, \|g\|/\tau)}$$

### 8.2 Warmup

Start with small learning rate, gradually increase:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{warmup}}$$

Helps stabilize early training.

### 8.3 Batch Size Effects

| Batch Size    | Generalization | Convergence |
| ------------- | -------------- | ----------- |
| Small (32)    | Often better   | Noisy       |
| Large (1024+) | May be worse   | Stable      |

**Linear scaling rule:** If batch size ×k, then $\eta$ ×k

---

## 9. Convergence Guarantees Summary

| Setting         | Rate            | Conditions          |
| --------------- | --------------- | ------------------- |
| Convex, smooth  | $O(1/T)$        | $\eta \leq 1/L$     |
| Strongly convex | $O(\exp(-T))$   | $\eta = 1/L$        |
| Non-convex      | $O(1/\sqrt{T})$ | To stationary point |

### 9.1 SGD Convergence

With decreasing learning rate $\eta_t = O(1/t)$:

**Convex:** $\mathbb{E}[f(\bar{w}_T)] - f(w^*) = O(1/\sqrt{T})$

**Strongly convex:** $\mathbb{E}[\|w_T - w^*\|^2] = O(1/T)$

---

## 10. Summary

| Concept          | Key Point                        |
| ---------------- | -------------------------------- |
| Gradient descent | $w \leftarrow w - \eta \nabla f$ |
| Learning rate    | Critical hyperparameter          |
| Momentum         | Accelerates, dampens oscillation |
| Adaptive (Adam)  | Per-parameter learning rates     |
| Convergence      | Depends on convexity, smoothness |

**Practical advice:**

1. Start with Adam (lr=0.001)
2. Try SGD+momentum for vision tasks
3. Use learning rate scheduling
4. Clip gradients for RNNs
5. Tune batch size carefully

---

## References

1. Ruder - "An overview of gradient descent optimization algorithms"
2. Kingma & Ba - "Adam: A Method for Stochastic Optimization"
3. Goodfellow et al. - "Deep Learning" Chapter 8
