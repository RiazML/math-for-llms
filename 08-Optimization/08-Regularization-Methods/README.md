# Regularization Methods

## Introduction

Regularization is a fundamental technique that bridges optimization and generalization. It prevents overfitting by adding constraints or penalties to the learning process, ensuring models generalize well to unseen data. Understanding regularization is essential for training robust machine learning models.

## Prerequisites

- Gradient descent and optimization basics
- Loss functions
- Bias-variance tradeoff
- Linear algebra (norms)

## Learning Objectives

1. Understand why regularization is necessary
2. Master L1, L2, and elastic net regularization
3. Apply dropout and other implicit regularization
4. Connect regularization to Bayesian priors

---

## 1. The Need for Regularization

### 1.1 Overfitting Problem

```
Training vs Generalization:

Error │
      │ ╲                        Overfitting
      │  ╲    ___________        zone
      │   ╲__/           ╲___
      │    Training error    ╲
      │                       ╲
      │   ___________________
      │  /    Test error      ╲
      │ /
      └──────────────────────────► Model Complexity
           Underfitting  │  Overfitting
                    Sweet spot
```

### 1.2 Bias-Variance Tradeoff

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

| Issue         | Cause             | Solution        |
| ------------- | ----------------- | --------------- |
| High Bias     | Model too simple  | More complexity |
| High Variance | Model too complex | Regularization  |

### 1.3 Regularization as Constraint

**Unconstrained optimization:**
$$\min_\theta \mathcal{L}(\theta)$$

**Regularized optimization:**
$$\min_\theta \mathcal{L}(\theta) + \lambda R(\theta)$$

where $R(\theta)$ is the regularization term and $\lambda$ controls strength.

---

## 2. L2 Regularization (Ridge / Weight Decay)

### 2.1 Definition

$$\mathcal{L}_{L2} = \mathcal{L}_{data} + \frac{\lambda}{2} \|\mathbf{w}\|_2^2 = \mathcal{L}_{data} + \frac{\lambda}{2} \sum_i w_i^2$$

### 2.2 Gradient Update

$$\nabla_w \mathcal{L}_{L2} = \nabla_w \mathcal{L}_{data} + \lambda \mathbf{w}$$

**Update rule:**
$$\mathbf{w} \leftarrow \mathbf{w} - \eta(\nabla_w \mathcal{L}_{data} + \lambda \mathbf{w})$$
$$= (1 - \eta\lambda)\mathbf{w} - \eta \nabla_w \mathcal{L}_{data}$$

Hence the name "weight decay" - weights shrink by factor $(1 - \eta\lambda)$.

### 2.3 Closed-Form Solution (Linear Regression)

Without regularization: $\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$

With L2: $\mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$

**Benefits:**

- Always invertible (even if $X^TX$ is singular)
- Numerically stable
- Shrinks coefficients toward zero

### 2.4 Geometric Interpretation

```
        w₂
         │      Contours of L(w)
         │    ╱╲
         │   ╱  ╲
         │  (    )  Unconstrained
         │   ╲  ╱   optimum
         │    ╲╱
    ─────┼─────●─────── w₁
         │    ╱│╲
         │   ╱ │ ╲  L2 constraint
         │  (  │  )  (circle)
         │   ╲ │ ╱
         │    ╲│╱
         │     ●
         │  Regularized
         │  optimum
```

L2 constraint region is a sphere: $\|\mathbf{w}\|_2^2 \leq c$

### 2.5 Bayesian Interpretation

L2 regularization = Gaussian prior on weights:

$$P(\mathbf{w}) = \mathcal{N}(0, \sigma^2 I)$$

MAP estimation with this prior gives L2-regularized solution.

$$\lambda = \frac{1}{2\sigma^2}$$

---

## 3. L1 Regularization (Lasso)

### 3.1 Definition

$$\mathcal{L}_{L1} = \mathcal{L}_{data} + \lambda \|\mathbf{w}\|_1 = \mathcal{L}_{data} + \lambda \sum_i |w_i|$$

### 3.2 Sparsity-Inducing Property

L1 regularization drives some weights exactly to zero!

```
L1 vs L2 solutions:

Feature    │  L2 weights  │  L1 weights
───────────┼──────────────┼─────────────
Feature 1  │    0.45      │    0.60
Feature 2  │    0.32      │    0.40
Feature 3  │    0.15      │    0.00  ← Sparse!
Feature 4  │    0.08      │    0.00  ← Sparse!
```

### 3.3 Geometric Interpretation

```
        w₂
         │
         │    ╱╲  Loss contours
         │   ╱  ╲
         │  (    )
         │   ╲  ╱
         │    ╲╱
    ─────┼────◇─────── w₁
         │   /│╲
         │  / │ ╲  L1 constraint
         │ /  │  ╲  (diamond)
         │◇───●───◇
         │ ╲  │  ╱
         │  ╲ │ ╱
         │   ╲│╱
         │    ◇
```

Diamond shape has corners on axes → solutions often at corners (sparse).

### 3.4 Subgradient

L1 norm is not differentiable at zero:

$$\frac{\partial}{\partial w_i} |w_i| = \begin{cases} +1 & w_i > 0 \\ -1 & w_i < 0 \\ [-1, +1] & w_i = 0 \end{cases}$$

### 3.5 Proximal Operator (Soft Thresholding)

$$\text{prox}_{\lambda\|\cdot\|_1}(w) = \text{sign}(w) \max(|w| - \lambda, 0)$$

```
Soft thresholding:

Output │           ╱
       │         ╱
       │       ╱
       │─────╱──────────► Input
       │   ╱    λ
       │ ╱
       │╱
```

### 3.6 Bayesian Interpretation

L1 regularization = Laplace prior:

$$P(w_i) = \frac{\lambda}{2} e^{-\lambda|w_i|}$$

---

## 4. Elastic Net

### 4.1 Definition

Combines L1 and L2:

$$\mathcal{L}_{EN} = \mathcal{L}_{data} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

Or with mixing parameter $\alpha \in [0, 1]$:

$$\mathcal{L}_{EN} = \mathcal{L}_{data} + \lambda[\alpha \|\mathbf{w}\|_1 + (1-\alpha) \|\mathbf{w}\|_2^2]$$

### 4.2 Properties

| $\alpha$ | Behavior        |
| -------- | --------------- |
| 0        | Pure L2 (Ridge) |
| 1        | Pure L1 (Lasso) |
| 0.5      | Balanced        |

**Benefits:**

- Sparsity from L1
- Stability from L2 (handles correlated features)
- Groups correlated features together

### 4.3 When to Use

- **L1 (Lasso):** Feature selection needed, features are independent
- **L2 (Ridge):** All features relevant, multicollinearity present
- **Elastic Net:** Some sparsity needed, features are correlated

---

## 5. Dropout

### 5.1 Mechanism

Randomly set activations to zero during training:

$$\tilde{h}_i = \begin{cases} 0 & \text{with probability } p \\ h_i / (1-p) & \text{with probability } 1-p \end{cases}$$

### 5.2 Training vs Inference

**Training:** Apply dropout (stochastic)

**Inference:** No dropout, but scale weights by $(1-p)$

Or use "inverted dropout" - scale during training by $1/(1-p)$.

### 5.3 Interpretation as Ensemble

Dropout trains an ensemble of $2^n$ sub-networks (for $n$ units).

```
Full Network:    With Dropout (example):
    ●────●           ●    ●
   /│╲  /│╲         /      ╲
  ● ● ●● ● ●   →   ●   ●    ●
   ╲│╱  ╲│╱             ╲  ╱
    ●────●               ●
```

### 5.4 Regularization Effect

Dropout approximately performs:

- L2 regularization
- Adaptive regularization based on gradient magnitude

### 5.5 Variants

| Variant             | Description                         |
| ------------------- | ----------------------------------- |
| Dropout             | Standard random dropout             |
| DropConnect         | Drop weights instead of activations |
| Spatial Dropout     | Drop entire feature maps (CNNs)     |
| Variational Dropout | Learned dropout rates               |

---

## 6. Early Stopping

### 6.1 Concept

Stop training when validation error increases:

```
Error │
      │╲
      │ ╲      Training
      │  ╲___________
      │   ╲ Validation
      │    ╲____
      │         ╲____
      │              ╲____
      └───────────────────│──► Epochs
                   Early Stop
```

### 6.2 Connection to L2 Regularization

For linear models with gradient descent:

$$\mathbf{w}_t \approx (I - (I - \eta X^TX)^t)(X^TX)^{-1}X^T\mathbf{y}$$

Early stopping at iteration $t$ is equivalent to L2 regularization with:

$$\lambda \approx \frac{1}{\eta t}$$

### 6.3 Implementation

```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        break  # Early stop
```

---

## 7. Batch Normalization as Regularization

### 7.1 Implicit Regularization

Batch normalization provides regularization through:

1. Mini-batch statistics add noise
2. Smooths the optimization landscape
3. Reduces internal covariate shift

### 7.2 Mechanism

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

The mini-batch $\mu_B, \sigma_B^2$ inject noise → regularization effect.

---

## 8. Data Augmentation

### 8.1 Concept

Increase effective training set size through transformations:

$$\mathcal{D}_{aug} = \{(T(x), y) : (x, y) \in \mathcal{D}, T \in \mathcal{T}\}$$

### 8.2 Common Augmentations

| Domain  | Augmentations                         |
| ------- | ------------------------------------- |
| Images  | Flip, rotate, crop, color jitter      |
| Text    | Synonym replacement, back-translation |
| Audio   | Time stretch, pitch shift, noise      |
| Tabular | Mixup, SMOTE                          |

### 8.3 Mixup Regularization

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

---

## 9. Weight Constraints

### 9.1 Max-Norm Constraint

Constrain weight vectors to have bounded norm:

$$\|\mathbf{w}\|_2 \leq c$$

After each update:
$$\mathbf{w} \leftarrow \mathbf{w} \cdot \min\left(1, \frac{c}{\|\mathbf{w}\|_2}\right)$$

### 9.2 Spectral Normalization

Normalize weight matrix by spectral norm:

$$W_{SN} = \frac{W}{\sigma(W)}$$

where $\sigma(W)$ is the largest singular value.

**Use:** Stabilizes GAN training.

---

## 10. Regularization in Deep Learning

### 10.1 Implicit Regularization

| Technique           | Regularization Effect |
| ------------------- | --------------------- |
| SGD                 | Prefers flat minima   |
| Large learning rate | Escapes sharp minima  |
| Small batch size    | Adds gradient noise   |
| Architecture        | Inductive bias        |

### 10.2 Explicit vs Implicit

**Explicit:**

- L1, L2 penalty
- Dropout
- Weight constraints

**Implicit:**

- SGD dynamics
- Batch normalization
- Data augmentation
- Early stopping

### 10.3 Combining Regularizations

Common combinations:

- L2 + Dropout
- Batch Norm + Data Augmentation
- Early Stopping + L2

**Warning:** Too much regularization → underfitting.

---

## 11. Choosing Regularization Strength

### 11.1 Cross-Validation

```python
# K-fold cross-validation for λ selection
for lambda in lambda_candidates:
    scores = []
    for fold in k_folds:
        model = train(lambda, train_fold)
        score = evaluate(model, val_fold)
        scores.append(score)
    cv_score[lambda] = mean(scores)

best_lambda = argmax(cv_score)
```

### 11.2 Regularization Path

Plot coefficients vs $\lambda$:

```
Coefficient │
            │╲___
            │    ╲___
            │        ╲___
            │ ____       ╲___
            │╱    ╲____      ╲
            │          ╲______╲___
            └─────────────────────────► log(λ)
            Strong ←────────→ Weak
```

### 11.3 Guidelines

| Scenario      | Recommendation        |
| ------------- | --------------------- |
| Small data    | Strong regularization |
| Large data    | Less regularization   |
| Complex model | More regularization   |
| Simple model  | Less regularization   |

---

## 12. Summary

| Method         | Penalty              | Effect          | Use Case            |
| -------------- | -------------------- | --------------- | ------------------- |
| L2 (Ridge)     | $\|\mathbf{w}\|_2^2$ | Shrinks weights | Multicollinearity   |
| L1 (Lasso)     | $\|\mathbf{w}\|_1$   | Sparse weights  | Feature selection   |
| Elastic Net    | L1 + L2              | Both            | Correlated features |
| Dropout        | Random masking       | Ensemble        | Deep networks       |
| Early Stopping | Iteration limit      | Implicit L2     | Any model           |

**Key Insight:** Regularization trades training performance for generalization, helping models perform well on unseen data.

---

## References

1. Goodfellow et al. - "Deep Learning" (Chapter 7)
2. Hastie et al. - "Elements of Statistical Learning"
3. Srivastava et al. - "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
4. Zhang et al. - "Understanding Deep Learning Requires Rethinking Generalization"
