# Loss Functions for Machine Learning

## Overview

Loss functions quantify the discrepancy between model predictions and true targets, serving as the objective to minimize during training. The choice of loss function fundamentally shapes what the model learns.

## Mathematical Framework

### General Definition

A **loss function** $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$ measures prediction error:

$$\ell(\hat{y}, y) \geq 0 \quad \text{with} \quad \ell(y, y) = 0$$

The **empirical risk** over dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$:

$$\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)$$

### Expected Risk

The **true risk** we aim to minimize:

$$R(\theta) = \mathbb{E}_{(x,y) \sim P}[\ell(f_\theta(x), y)]$$

Empirical risk minimization (ERM) approximates this with finite samples.

## Regression Losses

### Mean Squared Error (MSE) / L2 Loss

$$\ell_{\text{MSE}}(\hat{y}, y) = (\hat{y} - y)^2$$

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2$$

**Properties:**

- Convex, smooth, differentiable everywhere
- Gradient: $\nabla_{\hat{y}} \ell = 2(\hat{y} - y)$
- Optimal prediction: conditional mean $\mathbb{E}[y|x]$
- Sensitive to outliers (squared penalty)

### Mean Absolute Error (MAE) / L1 Loss

$$\ell_{\text{MAE}}(\hat{y}, y) = |\hat{y} - y|$$

**Properties:**

- Convex but not differentiable at $\hat{y} = y$
- Subgradient: $\partial \ell \in \{-1, 1\}$
- Optimal prediction: conditional median
- Robust to outliers (linear penalty)

### Huber Loss (Smooth L1)

$$
\ell_{\text{Huber}}(\hat{y}, y) = \begin{cases}
\frac{1}{2}(\hat{y} - y)^2 & |r| \leq \delta \\
\delta |r| - \frac{1}{2}\delta^2 & |r| > \delta
\end{cases}
$$

where $r = \hat{y} - y$.

**Properties:**

- Differentiable everywhere
- Quadratic for small errors (like MSE)
- Linear for large errors (like MAE)
- Balances efficiency and robustness

### Quantile Loss

For quantile $\tau \in (0, 1)$:

$$
\ell_\tau(\hat{y}, y) = \begin{cases}
\tau(y - \hat{y}) & y \geq \hat{y} \\
(1-\tau)(\hat{y} - y) & y < \hat{y}
\end{cases}
$$

Equivalently: $\ell_\tau = (y - \hat{y})(\tau - \mathbf{1}_{y < \hat{y}})$

**Application:** Predicting different quantiles of the distribution.

### Log-Cosh Loss

$$\ell_{\text{log-cosh}}(\hat{y}, y) = \log(\cosh(\hat{y} - y))$$

**Properties:**

- Smooth approximation of MAE
- $\approx (\hat{y}-y)^2/2$ for small errors
- $\approx |\hat{y}-y| - \log 2$ for large errors

## Classification Losses

### Binary Cross-Entropy (Log Loss)

For prediction $\hat{p} \in (0, 1)$ and label $y \in \{0, 1\}$:

$$\ell_{\text{BCE}}(\hat{p}, y) = -y\log(\hat{p}) - (1-y)\log(1-\hat{p})$$

**Properties:**

- Proper scoring rule (minimized at true probabilities)
- Convex in logits $z = \log(\hat{p}/(1-\hat{p}))$
- Gradient: $\nabla_z \ell = \hat{p} - y$ (with sigmoid)
- Maximum likelihood for Bernoulli

### Categorical Cross-Entropy

For $K$ classes with prediction $\hat{p} \in \Delta^{K-1}$ and one-hot $y$:

$$\ell_{\text{CE}}(\hat{p}, y) = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)$$

If $y$ is the true class index:

$$\ell_{\text{CE}}(\hat{p}, y) = -\log(\hat{p}_y)$$

**Properties:**

- Information-theoretic: $H(p, \hat{p}) = H(p) + D_{KL}(p||\hat{p})$
- Maximum likelihood for categorical distribution
- Used with softmax activation

### Hinge Loss (SVM)

$$\ell_{\text{hinge}}(\hat{y}, y) = \max(0, 1 - y \cdot \hat{y})$$

where $y \in \{-1, +1\}$ and $\hat{y}$ is the raw score.

**Properties:**

- Convex surrogate for 0-1 loss
- Sparse gradients (zero for correctly classified with margin)
- Used in Support Vector Machines

### Squared Hinge Loss

$$\ell_{\text{sq-hinge}}(\hat{y}, y) = \max(0, 1 - y \cdot \hat{y})^2$$

**Properties:**

- Differentiable everywhere
- Stronger penalty for violating margin

### Focal Loss

For handling class imbalance:

$$\ell_{\text{focal}}(\hat{p}, y) = -\alpha_y (1-\hat{p}_y)^\gamma \log(\hat{p}_y)$$

**Parameters:**

- $\gamma > 0$: focusing parameter (higher = more focus on hard examples)
- $\alpha$: class weights

**Properties:**

- Down-weights easy examples
- Addresses class imbalance in object detection

## Probabilistic Losses

### Negative Log-Likelihood (NLL)

For probabilistic model $p_\theta(y|x)$:

$$\ell_{\text{NLL}}(x, y; \theta) = -\log p_\theta(y|x)$$

**Special cases:**

- Gaussian with fixed variance → MSE
- Bernoulli → Binary cross-entropy
- Categorical → Cross-entropy

### Gaussian NLL (Heteroscedastic)

For predicted mean $\mu$ and variance $\sigma^2$:

$$\ell = \frac{1}{2}\log(2\pi\sigma^2) + \frac{(y-\mu)^2}{2\sigma^2}$$

**Application:** Uncertainty estimation in regression.

### KL Divergence Loss

$$D_{KL}(p||q) = \mathbb{E}_p\left[\log\frac{p(x)}{q(x)}\right]$$

**Properties:**

- Non-symmetric: $D_{KL}(p||q) \neq D_{KL}(q||p)$
- Zero iff $p = q$
- Used in VAEs for regularization

### Evidence Lower Bound (ELBO)

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x)||p(z))$$

**Components:**

- Reconstruction term: likelihood of data
- Regularization term: KL to prior

## Structured Prediction Losses

### CTC Loss (Connectionist Temporal Classification)

For sequence-to-sequence without alignment:

$$p(y|x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_t p(\pi_t | x)$$

where $\mathcal{B}$ is the alignment mapping.

**Application:** Speech recognition, OCR

### Sequence Cross-Entropy

$$\mathcal{L} = -\sum_{t=1}^T \log p(y_t | y_{<t}, x)$$

**Application:** Language modeling, translation

## Ranking Losses

### Pairwise Ranking Loss

$$\ell(x_i, x_j) = \max(0, m - s(x_i) + s(x_j))$$

where $x_i$ should rank higher than $x_j$.

### Triplet Loss

$$\ell(a, p, n) = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m)$$

**Components:**

- $a$: anchor
- $p$: positive (same class as anchor)
- $n$: negative (different class)
- $m$: margin

### Contrastive Loss

$$\ell(x_i, x_j, y) = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2$$

where $d = \|f(x_i) - f(x_j)\|$ and $y \in \{0, 1\}$ indicates similarity.

### InfoNCE Loss

$$\ell = -\log \frac{\exp(s(x, x^+)/\tau)}{\sum_{j} \exp(s(x, x_j)/\tau)}$$

**Application:** Self-supervised learning (SimCLR, CLIP)

## Regularization as Loss Terms

### Weight Decay / L2 Regularization

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\|\theta\|_2^2$$

**Effect:** Gaussian prior on parameters

### L1 Regularization (Lasso)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda\|\theta\|_1$$

**Effect:** Laplace prior, promotes sparsity

### Elastic Net

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1\|\theta\|_1 + \lambda_2\|\theta\|_2^2$$

## Loss Function Properties

### Convexity

A loss $\ell$ is **convex** if:

$$\ell(\alpha \hat{y}_1 + (1-\alpha)\hat{y}_2, y) \leq \alpha\ell(\hat{y}_1, y) + (1-\alpha)\ell(\hat{y}_2, y)$$

**Importance:** Guarantees global optimum with convex models.

### Calibration

A loss produces **calibrated** probabilities if:

$$\mathbb{P}(Y=1 | \hat{p}=p) = p$$

Cross-entropy is a proper scoring rule, encouraging calibration.

### Consistency

A loss is **Fisher consistent** for classification if minimizing it yields the Bayes optimal classifier.

### Lipschitz Continuity

Loss gradient bounded: $\|\nabla \ell(\hat{y}_1) - \nabla \ell(\hat{y}_2)\| \leq L\|\hat{y}_1 - \hat{y}_2\|$

**Importance:** Ensures stable gradient descent.

## Computational Considerations

### Numerical Stability

**Log-sum-exp trick:**
$$\log\sum_i e^{x_i} = m + \log\sum_i e^{x_i - m}$$
where $m = \max_i x_i$.

**Label smoothing:** Replace hard labels $y$ with:
$$y_{\text{smooth}} = (1-\epsilon)y + \epsilon/K$$

### Loss Reduction

- **Mean:** Average over samples
- **Sum:** Total loss (affects learning rate scaling)
- **None:** Per-sample losses for weighted training

### Multi-Task Learning

$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T w_t \mathcal{L}_t$$

**Challenges:**

- Task weight selection
- Gradient magnitude balancing
- Uncertainty weighting: $w_t = 1/(2\sigma_t^2)$

## Applications in Modern ML

| Application            | Common Losses             |
| ---------------------- | ------------------------- |
| Image Classification   | Cross-entropy, Focal      |
| Object Detection       | Focal, IoU, GIoU          |
| Regression             | MSE, Huber, Quantile      |
| Language Models        | Cross-entropy             |
| GANs                   | Adversarial, Wasserstein  |
| VAEs                   | ELBO, β-VAE               |
| Contrastive Learning   | InfoNCE, Triplet          |
| Reinforcement Learning | TD error, Policy gradient |

## Summary

Loss functions determine:

1. **What the model optimizes** (objective)
2. **Sensitivity to errors** (robustness)
3. **Probability interpretation** (calibration)
4. **Optimization landscape** (convexity)

Key considerations:

- Match loss to problem type (regression/classification/ranking)
- Consider outlier sensitivity
- Ensure numerical stability
- Balance multiple objectives in multi-task learning
