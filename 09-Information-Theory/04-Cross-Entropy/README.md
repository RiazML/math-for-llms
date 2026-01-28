# Cross-Entropy

## Introduction

Cross-entropy is one of the most important loss functions in machine learning, particularly for classification tasks. It measures the difference between two probability distributions and connects information theory to optimization. Understanding cross-entropy is essential for training neural networks and interpreting model outputs.

## Prerequisites

- Entropy fundamentals
- KL divergence
- Probability distributions
- Likelihood and maximum likelihood estimation

## Learning Objectives

1. Understand cross-entropy definition and properties
2. Derive cross-entropy loss for classification
3. Connect cross-entropy to KL divergence and MLE
4. Apply cross-entropy in various ML contexts

---

## 1. Definition

### 1.1 Mathematical Definition

Cross-entropy between distributions $P$ (true) and $Q$ (predicted):

$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

For continuous distributions:

$$H(P, Q) = -\int p(x) \log q(x) dx$$

### 1.2 Interpretation

Cross-entropy measures:

- Average number of bits needed to encode data from $P$ using code optimized for $Q$
- "Surprise" when using $Q$ to predict outcomes from $P$

```
True distribution P          Model Q
       ▄▄▄▄                    ▄▄▄▄▄▄▄▄
      ██████                  ██████████
     ████████                ████████████
    ──────────              ──────────────

Cross-entropy: How well does Q's code work for P?
Lower cross-entropy = Better model
```

---

## 2. Key Properties

### 2.1 Relationship to Entropy and KL Divergence

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

**Implications:**

- $H(P, Q) \geq H(P)$ (entropy is minimum cross-entropy)
- Minimizing cross-entropy = Minimizing KL divergence (when $P$ fixed)
- $H(P, Q) = H(P)$ iff $P = Q$

### 2.2 Non-Negativity

$$H(P, Q) \geq 0$$

with equality only for degenerate distributions.

### 2.3 Asymmetry

$$H(P, Q) \neq H(Q, P)$$

Cross-entropy is not symmetric in its arguments.

### 2.4 Bounds

$$H(P) \leq H(P, Q)$$

Cross-entropy is always at least as large as entropy.

---

## 3. Cross-Entropy Loss in Machine Learning

### 3.1 Binary Classification

For binary labels $y \in \{0, 1\}$ and predicted probability $\hat{y} = P(y=1|x)$:

$$\mathcal{L}_{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$

**Expanded form:**

- If $y = 1$: $\mathcal{L} = -\log(\hat{y})$
- If $y = 0$: $\mathcal{L} = -\log(1-\hat{y})$

```
Loss curves for binary cross-entropy:

When y = 1:                  When y = 0:
Loss                         Loss
│                           │
│\                          │        /
│ \                         │       /
│  \                        │      /
│   \____                   │_____/
└────────────► ŷ           └────────────► ŷ
0            1              0            1
```

### 3.2 Multi-Class Classification

For $K$ classes with one-hot label $\mathbf{y}$ and predictions $\hat{\mathbf{y}}$:

$$\mathcal{L}_{CE} = -\sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

Since one-hot encoding has only one $y_k = 1$:

$$\mathcal{L}_{CE} = -\log(\hat{y}_c)$$

where $c$ is the true class.

### 3.3 With Softmax

For logits $\mathbf{z}$ and softmax:

$$\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Combined softmax + cross-entropy:

$$\mathcal{L} = -z_c + \log\left(\sum_j e^{z_j}\right)$$

This is numerically more stable than computing softmax then log.

---

## 4. Connection to Maximum Likelihood

### 4.1 MLE Derivation

For dataset $\{(x_i, y_i)\}_{i=1}^{n}$, maximize likelihood:

$$\max_\theta \prod_{i=1}^{n} P_\theta(y_i | x_i)$$

Taking log and negating:

$$\min_\theta -\frac{1}{n}\sum_{i=1}^{n} \log P_\theta(y_i | x_i)$$

This is exactly cross-entropy loss!

### 4.2 Interpretation

- **Cross-entropy loss** = Negative log-likelihood
- **Minimizing cross-entropy** = Maximum likelihood estimation
- **Lower cross-entropy** = Higher likelihood of data under model

---

## 5. Cross-Entropy for Different Tasks

### 5.1 Binary Classification (Logistic Regression)

Model: $P(y=1|x) = \sigma(w^T x + b)$

Loss: $\mathcal{L} = -y\log(\sigma(z)) - (1-y)\log(1-\sigma(z))$

Using $1 - \sigma(z) = \sigma(-z)$:

$$\mathcal{L} = \log(1 + e^{-yz})$$

where $y \in \{-1, +1\}$.

### 5.2 Multi-Class (Softmax Regression)

Model: $P(y=k|x) = \text{softmax}(Wx + b)_k$

Loss: $\mathcal{L} = -\log(\text{softmax}(z)_{y})$

### 5.3 Sequence Models (Language Modeling)

For sequence $\mathbf{x} = (x_1, ..., x_T)$:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})$$

**Perplexity** (common metric):

$$\text{PPL} = \exp(\mathcal{L}/T) = \exp(H(P_{data}, P_{model}))$$

### 5.4 Multi-Label Classification

Each label is independent binary:

$$\mathcal{L} = -\sum_{k=1}^{K} [y_k \log(\hat{y}_k) + (1-y_k) \log(1-\hat{y}_k)]$$

### 5.5 Weighted Cross-Entropy

For class imbalance:

$$\mathcal{L} = -\sum_{k=1}^{K} w_k y_k \log(\hat{y}_k)$$

where $w_k$ weights classes (higher for minority).

---

## 6. Properties as a Loss Function

### 6.1 Gradient Analysis

For softmax output:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$

Simple gradient: (predicted - actual)

### 6.2 Convexity

Cross-entropy is:

- **Convex** in the predicted probabilities
- **Not globally convex** in neural network parameters

### 6.3 Comparison with Other Losses

| Loss          | Formula                                  | Use Case       |
| ------------- | ---------------------------------------- | -------------- |
| Cross-Entropy | $-\sum y \log \hat{y}$                   | Classification |
| MSE           | $(y - \hat{y})^2$                        | Regression     |
| Hinge         | $\max(0, 1-y\hat{y})$                    | SVM            |
| Focal         | $-\alpha(1-\hat{y})^\gamma \log \hat{y}$ | Imbalanced     |

### 6.4 Why Cross-Entropy for Classification?

1. **Matches probabilistic interpretation** (MLE)
2. **Strong gradients** even for confident wrong predictions
3. **Proper scoring rule** (optimal prediction is true distribution)
4. **Works well with softmax** (stable gradients)

---

## 7. Numerical Stability

### 7.1 Log-Sum-Exp Trick

For numerical stability with softmax:

$$\log\sum_j e^{z_j} = \max_j z_j + \log\sum_j e^{z_j - \max_j z_j}$$

### 7.2 Avoiding Log(0)

```python
# Unstable
loss = -y * np.log(y_hat)

# Stable
eps = 1e-7
loss = -y * np.log(y_hat + eps)

# Or use clip
loss = -y * np.log(np.clip(y_hat, eps, 1-eps))
```

### 7.3 Combined Softmax-CrossEntropy

Most frameworks provide stable implementations:

```python
# PyTorch
loss = F.cross_entropy(logits, labels)  # Combines softmax + CE

# TensorFlow
loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
```

---

## 8. Variants and Extensions

### 8.1 Label Smoothing

Instead of one-hot $y_k \in \{0, 1\}$:

$$y_k^{smooth} = (1 - \epsilon) y_k + \frac{\epsilon}{K}$$

**Benefits:**

- Prevents overconfident predictions
- Improves generalization
- Provides regularization

### 8.2 Focal Loss

For imbalanced classification:

$$\mathcal{L}_{FL} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where:

- $p_t$: predicted probability for true class
- $\gamma$: focusing parameter (typically 2)
- $\alpha_t$: class weighting

### 8.3 Knowledge Distillation Loss

Transfer knowledge from teacher $T$ to student $S$:

$$\mathcal{L}_{KD} = \alpha H(y, S) + (1-\alpha) \tau^2 H(T_\tau, S_\tau)$$

where $T_\tau, S_\tau$ are softened with temperature $\tau$.

### 8.4 Symmetric Cross-Entropy

For noisy labels:

$$\mathcal{L}_{SCE} = \alpha H(P, Q) + \beta H(Q, P)$$

More robust to label noise than standard CE.

---

## 9. Cross-Entropy in Specific Models

### 9.1 Neural Networks

Output layer for classification:

```
Input → Hidden Layers → Logits → Softmax → Probabilities
                              ↓
                         Cross-Entropy Loss ← One-hot Labels
```

### 9.2 Variational Autoencoders

Reconstruction term for discrete data:

$$\mathcal{L}_{recon} = -\mathbb{E}_{q(z|x)}[\log p_\theta(x|z)]$$

This is cross-entropy between data and reconstruction.

### 9.3 Language Models

Predict next token:

$$\mathcal{L} = -\log P(x_{t+1} | x_1, ..., x_t)$$

Trained on all positions, average cross-entropy.

### 9.4 GANs (Discriminator)

Binary cross-entropy for real/fake:

$$\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]$$

---

## 10. Practical Considerations

### 10.1 Interpretation of Loss Values

| Cross-Entropy | Interpretation               |
| ------------- | ---------------------------- |
| 0.0           | Perfect prediction           |
| 0.69          | Random guessing (2 classes)  |
| 1.0           | Confident wrong (1 class)    |
| 2.3           | Random guessing (10 classes) |

For $K$ classes, random baseline: $\log(K)$

### 10.2 Monitoring Training

```
Cross-Entropy Loss over epochs:

Loss │
 2.0 │*
     │ **
 1.0 │   ****
     │       *****
 0.5 │            *******
     │                    ***
 0.0 └─────────────────────────► Epoch
```

### 10.3 Relationship to Accuracy

- Lower CE doesn't always mean higher accuracy
- CE considers confidence, accuracy only considers correctness
- Well-calibrated models: CE predicts accuracy better

---

## 11. Summary

| Concept        | Formula/Description                            |
| -------------- | ---------------------------------------------- |
| Cross-Entropy  | $H(P,Q) = -\sum P(x) \log Q(x)$                |
| Decomposition  | $H(P,Q) = H(P) + D_{KL}(P \| Q)$               |
| Binary CE      | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$       |
| Multi-class CE | $-\sum_k y_k \log \hat{y}_k = -\log \hat{y}_c$ |
| MLE Connection | Minimizing CE = Maximizing likelihood          |
| Gradient       | $\nabla_{z_k} \mathcal{L} = \hat{y}_k - y_k$   |

**Key Insights:**

- Cross-entropy loss is the standard for classification
- Connected to MLE and KL divergence
- Strong gradients encourage learning from mistakes
- Use numerical stability tricks in implementation

---

## References

1. Bishop - "Pattern Recognition and Machine Learning"
2. Goodfellow et al. - "Deep Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
4. Lin et al. - "Focal Loss for Dense Object Detection"
5. Müller et al. - "When Does Label Smoothing Help?"
