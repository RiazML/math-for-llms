# 🎯 Interview Preparation Guide

> Common mathematical interview questions for ML/AI positions with detailed solutions.

---

## Table of Contents

1. [Linear Algebra Questions](#linear-algebra-questions)
2. [Calculus Questions](#calculus-questions)
3. [Probability & Statistics Questions](#probability--statistics-questions)
4. [Optimization Questions](#optimization-questions)
5. [Information Theory Questions](#information-theory-questions)
6. [Applied ML Math Questions](#applied-ml-math-questions)
7. [Deep Learning Math Questions](#deep-learning-math-questions)
8. [Generative Models Math Questions](#generative-models-math-questions)
9. [Quick Review Checklist](#quick-review-checklist)
10. [Study Plan](#-study-plan)

---

## Linear Algebra Questions

### Q1: What is the difference between eigenvalue decomposition and SVD?

**Answer:**

| Aspect     | Eigendecomposition        | SVD                                          |
| ---------- | ------------------------- | -------------------------------------------- |
| Applies to | Square matrices only      | Any matrix (m×n)                             |
| Formula    | $A = P\Lambda P^{-1}$     | $A = U\Sigma V^T$                            |
| Components | Eigenvalues, eigenvectors | Singular values, left/right singular vectors |
| Requires   | Diagonalizable matrix     | Always exists                                |

**Key Insight:** For symmetric positive semi-definite matrices, singular values equal eigenvalues, and $U = V$.

**ML Application:**

- Eigendecomposition: PCA on covariance matrix
- SVD: Recommender systems, image compression, pseudoinverse

---

### Q2: Explain the geometric interpretation of eigenvalues and eigenvectors.

**Answer:**

Eigenvectors are directions that remain unchanged (except for scaling) when a linear transformation is applied. Eigenvalues are the scaling factors.

```
Before transformation:          After transformation (A):
        ↑ v                              ↑ λv
        │                                │ (stretched by λ)
        │                                │
   ─────┼─────→                    ─────┼─────→
        │                                │
```

**ML Application:**

- In PCA, eigenvectors point in directions of maximum variance
- Large eigenvalue = high variance in that direction
- We keep eigenvectors with largest eigenvalues

---

### Q3: What makes a matrix positive definite? Why does it matter?

**Answer:**

A symmetric matrix $A$ is positive definite if:

- All eigenvalues are positive, OR
- $\mathbf{x}^T A \mathbf{x} > 0$ for all non-zero $\mathbf{x}$

**Why it matters:**

1. **Covariance matrices** are positive semi-definite
2. **Hessian being PD** guarantees local minimum
3. **Optimization**: Convex quadratic functions have PD Hessians
4. **Numerical stability**: PD matrices are invertible

---

### Q4: Explain the null space and column space. How do they relate to linear regression?

**Answer:**

- **Column space (range)**: All possible outputs $A\mathbf{x}$
- **Null space (kernel)**: All inputs $\mathbf{x}$ where $A\mathbf{x} = \mathbf{0}$

**In Linear Regression:**

```
Finding ŷ = Xw that's closest to y

        y
        ↗
       /
      /
     / ŷ (projection onto column space of X)
    ●─────────────────→ Column space of X

The residual (y - ŷ) is perpendicular to column space
```

**Insight:** If null space is non-trivial, there are infinitely many solutions (need regularization).

---

### Q5: What is the rank of a matrix and why is it important?

**Answer:**

Rank = number of linearly independent rows (or columns)

| Rank Property      | Implication                         |
| ------------------ | ----------------------------------- |
| rank(A) = min(m,n) | Full rank, unique solution possible |
| rank(A) < min(m,n) | Rank deficient, singular            |
| rank(X^TX) < n     | Multicollinearity in regression     |

**ML Applications:**

- Low-rank matrix approximation (compression)
- Detecting redundant features
- Matrix completion (recommender systems)

---

## Calculus Questions

### Q6: Derive the gradient of the sigmoid function.

**Answer:**

Given: $\sigma(x) = \frac{1}{1 + e^{-x}}$

Using quotient rule:
$$\frac{d\sigma}{dx} = \frac{0 \cdot (1+e^{-x}) - 1 \cdot (-e^{-x})}{(1+e^{-x})^2} = \frac{e^{-x}}{(1+e^{-x})^2}$$

Simplify:
$$= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot \frac{1+e^{-x}-1}{1+e^{-x}} = \sigma(x)(1-\sigma(x))$$

**Key Result:** $\sigma'(x) = \sigma(x)(1-\sigma(x))$

**Why it matters:** This makes backpropagation through sigmoid efficient!

---

### Q7: Explain the chain rule and its role in backpropagation.

**Answer:**

**Chain Rule:** For $h(x) = f(g(x))$:
$$\frac{dh}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**In Neural Networks:**

```
x → [Layer 1] → z₁ → [Activation] → a₁ → [Layer 2] → z₂ → [Loss] → L

∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
```

**Key Insight:** Backprop is just repeated application of chain rule, computed efficiently by caching intermediate values.

---

### Q8: What is the Jacobian matrix? When do you need it?

**Answer:**

The Jacobian is the matrix of all first-order partial derivatives for a vector-valued function:

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

**When needed:**

1. **Backprop through layers** with multiple outputs
2. **Change of variables** in probability (normalizing flows)
3. **Sensitivity analysis**: How outputs change with inputs

---

### Q9: What is the Hessian and how is it used in optimization?

**Answer:**

The Hessian is the matrix of second-order partial derivatives:

$$
H = \nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2}
\end{bmatrix}
$$

**Uses:**
| Condition | Meaning |
|-----------|---------|
| $H$ positive definite | Local minimum |
| $H$ negative definite | Local maximum |
| $H$ indefinite | Saddle point |

**In Optimization:**

- Newton's method: $x_{n+1} = x_n - H^{-1} \nabla f$
- Approximate Hessian: Adam, BFGS

---

### Q10: Explain Taylor series and its application in optimization.

**Answer:**

Taylor expansion around point $a$:
$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

**In Optimization:**

First-order approximation (gradient descent):
$$f(\theta + \Delta\theta) \approx f(\theta) + \nabla f(\theta)^T \Delta\theta$$

Second-order approximation (Newton's method):
$$f(\theta + \Delta\theta) \approx f(\theta) + \nabla f^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta$$

---

## Probability & Statistics Questions

### Q11: Derive Bayes' theorem and explain its components.

**Answer:**

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| Component | Name       | Meaning                       |
| --------- | ---------- | ----------------------------- |
| $P(A\|B)$ | Posterior  | Updated belief after seeing B |
| $P(B\|A)$ | Likelihood | Probability of B given A      |
| $P(A)$    | Prior      | Initial belief about A        |
| $P(B)$    | Evidence   | Normalizing constant          |

**ML Example - Naive Bayes:**
$$P(\text{spam}|\text{words}) = \frac{P(\text{words}|\text{spam}) \cdot P(\text{spam})}{P(\text{words})}$$

---

### Q12: What is the difference between MLE and MAP?

**Answer:**

**Maximum Likelihood Estimation (MLE):**
$$\hat{\theta}_{MLE} = \arg\max_\theta P(D|\theta)$$

- Only considers data likelihood
- Can overfit

**Maximum A Posteriori (MAP):**
$$\hat{\theta}_{MAP} = \arg\max_\theta P(D|\theta) \cdot P(\theta)$$

- Includes prior belief $P(\theta)$
- Acts as regularization

**Key Connection:**

- Gaussian prior → L2 regularization
- Laplace prior → L1 regularization

---

### Q13: Explain the bias-variance tradeoff mathematically.

**Answer:**

For any estimator, the expected error can be decomposed:

$$E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2$$

| Component | Meaning                      | High when...                    |
| --------- | ---------------------------- | ------------------------------- |
| Bias²     | Systematic error             | Model too simple (underfitting) |
| Variance  | Sensitivity to training data | Model too complex (overfitting) |
| σ²        | Irreducible noise            | Inherent in data                |

```
Error
  │
  │╲                  ╱
  │ ╲   Total Error  ╱
  │  ╲    ┌────┐    ╱
  │   ╲   │    │   ╱
  │    ╲──│────│──╱
  │     Bias²  │
  │            │ Variance
  └────────────┴──────────→ Model Complexity
     Simple          Complex
```

---

### Q14: What is the Central Limit Theorem and why does it matter?

**Answer:**

**CLT:** The sum (or average) of many independent random variables tends toward a normal distribution, regardless of the original distribution.

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)$$

**Why it matters:**

1. **Confidence intervals**: We can use normal distribution
2. **Hypothesis testing**: t-tests, z-tests
3. **SGD convergence**: Gradient estimates are approximately normal
4. **Batch normalization**: Large batch → normal activations

---

### Q15: Explain covariance and correlation. What's the difference?

**Answer:**

**Covariance:** Measures how two variables change together
$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

- Unbounded
- Units depend on X and Y

**Correlation:** Normalized covariance
$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- Always between -1 and 1
- Unitless

| Value  | Meaning                              |
| ------ | ------------------------------------ |
| ρ = 1  | Perfect positive linear relationship |
| ρ = 0  | No linear relationship               |
| ρ = -1 | Perfect negative linear relationship |

---

## Optimization Questions

### Q16: Explain gradient descent and its variants.

**Answer:**

**Basic Gradient Descent:**
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

| Variant    | Update Rule             | Pros/Cons              |
| ---------- | ----------------------- | ---------------------- |
| Batch GD   | Full dataset gradient   | Stable but slow        |
| SGD        | Single sample gradient  | Fast but noisy         |
| Mini-batch | Batch of samples        | Best of both           |
| Momentum   | Accumulate velocity     | Faster convergence     |
| Adam       | Adaptive learning rates | Works well in practice |

---

### Q17: What makes a function convex? Why does convexity matter?

**Answer:**

**Convexity Definition:**
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

for all $x$, $y$ and $\lambda \in [0, 1]$.

**Visual:**

```
Convex:                Non-convex:
     ╱╲                     ╱╲
    ╱  ╲                   ╱  ╲
   ╱    ╲                 ╱    ╲
  ╱  ★   ╲               ╱  ★   ╲  ★
 ╱        ╲             ╱    ╲   ╱╲
                            local minima!
```

**Why it matters:**

- Convex functions have **global minimum = local minimum**
- Gradient descent guaranteed to converge
- Linear regression loss is convex
- Neural network loss is non-convex (hence harder to optimize)

---

### Q18: Explain the Adam optimizer mathematically.

**Answer:**

Adam combines momentum and RMSprop:

```python
# Initialize
m = 0  # First moment (momentum)
v = 0  # Second moment (RMSprop)

for t in range(1, num_iterations):
    g = compute_gradient(theta)

    # Update moments
    m = beta1 * m + (1 - beta1) * g        # Momentum
    v = beta2 * v + (1 - beta2) * g**2     # RMSprop

    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    # Update parameters
    theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
```

**Defaults:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

---

### Q19: What are the KKT conditions?

**Answer:**

For constrained optimization:
$$\min f(x) \text{ subject to } g_i(x) \leq 0, h_j(x) = 0$$

**KKT Conditions:**

1. **Stationarity:** $\nabla f + \sum_i \lambda_i \nabla g_i + \sum_j \nu_j \nabla h_j = 0$
2. **Primal feasibility:** $g_i(x) \leq 0$, $h_j(x) = 0$
3. **Dual feasibility:** $\lambda_i \geq 0$
4. **Complementary slackness:** $\lambda_i g_i(x) = 0$

**ML Application:** SVM optimization satisfies KKT conditions.

---

## Information Theory Questions

### Q20: What is entropy and how is it used in ML?

**Answer:**

**Entropy:** Measure of uncertainty/randomness
$$H(X) = -\sum_x P(x) \log P(x)$$

| Distribution          | Entropy |
| --------------------- | ------- |
| Certain (one outcome) | 0       |
| Uniform               | Maximum |
| More spread out       | Higher  |

**ML Uses:**

- Decision tree splitting (maximize information gain)
- Regularization in neural networks
- Measuring model confidence

---

### Q21: Explain cross-entropy loss. Why is it used for classification?

**Answer:**

**Cross-Entropy:**
$$H(p, q) = -\sum_x p(x) \log q(x)$$

For classification:

- $p$ = true distribution (one-hot)
- $q$ = predicted probabilities

$$L = -\sum_{i=1}^K y_i \log(\hat{y}_i)$$

**Why use it:**

1. **Gradients**: Better gradients than MSE for classification
2. **Probability interpretation**: Natural for probability outputs
3. **Convexity**: Convex for logistic regression

---

### Q22: What is KL divergence and how is it related to cross-entropy?

**Answer:**

**KL Divergence:** Measures how different distribution Q is from P
$$D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

**Relation to Cross-Entropy:**
$$D_{KL}(P || Q) = H(P, Q) - H(P)$$
$$\text{KL Divergence} = \text{Cross-Entropy} - \text{Entropy of P}$$

**Key Properties:**

- $D_{KL} \geq 0$ (Gibbs' inequality)
- $D_{KL} = 0$ iff $P = Q$
- **Not symmetric:** $D_{KL}(P||Q) \neq D_{KL}(Q||P)$

**ML Uses:**

- VAE loss function
- Knowledge distillation
- Policy gradient (KL constraint)

---

## Applied ML Math Questions

### Q23: Derive the gradient for logistic regression.

**Answer:**

**Model:** $\hat{y} = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$

**Loss (single sample):** $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$

**Gradient derivation:**
$$\frac{\partial L}{\partial \mathbf{w}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}$$

Where $z = \mathbf{w}^T\mathbf{x}$:

- $\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}$
- $\frac{\partial \hat{y}}{\partial z} = \hat{y}(1-\hat{y})$
- $\frac{\partial z}{\partial \mathbf{w}} = \mathbf{x}$

**Result:** $\nabla_\mathbf{w} L = (\hat{y} - y)\mathbf{x}$

---

### Q24: Explain the math behind batch normalization.

**Answer:**

**Forward Pass:**

1. Compute batch statistics:
   - $\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$
   - $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$

2. Normalize:
   - $\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

3. Scale and shift:
   - $y_i = \gamma \hat{x}_i + \beta$

**Why it works:**

- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
- Smoother loss landscape

---

### Q25: Derive the attention mechanism math in Transformers.

**Answer:**

**Scaled Dot-Product Attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step by step:**

1. **Compute similarities:** $QK^T$ gives attention scores
2. **Scale:** Divide by $\sqrt{d_k}$ to prevent softmax saturation
3. **Softmax:** Convert to probability distribution
4. **Weighted sum:** Multiply by V

**Multi-Head Attention:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

---

## Deep Learning Math Questions

### Q21: Explain the math behind the Transformer's attention mechanism.

**Answer:**

Scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why $\sqrt{d_k}$?** Without scaling, dot products grow with dimension $d_k$, pushing softmax into saturation (near 0 or 1 gradients). Scaling keeps variance ~1.

**Multi-head attention** allows attending to information from different representation subspaces:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

**Complexity:** $O(n^2 d)$ where $n$ is sequence length — the quadratic bottleneck motivating efficient attention.

---

### Q22: Derive the backpropagation equations for a simple 2-layer network.

**Answer:**

Network: $z_1 = W_1x + b_1$, $a_1 = \sigma(z_1)$, $z_2 = W_2a_1 + b_2$, $\hat{y} = \text{softmax}(z_2)$

Loss: $L = -\sum_k y_k \log \hat{y}_k$ (cross-entropy)

**Backward pass** (using chain rule):
1. $\delta_2 = \hat{y} - y$ (softmax + cross-entropy simplification)
2. $\frac{\partial L}{\partial W_2} = \delta_2 a_1^T$
3. $\frac{\partial L}{\partial b_2} = \delta_2$
4. $\delta_1 = (W_2^T \delta_2) \odot \sigma'(z_1)$ (element-wise)
5. $\frac{\partial L}{\partial W_1} = \delta_1 x^T$
6. $\frac{\partial L}{\partial b_1} = \delta_1$

**Key insight:** Each layer's gradient depends on the downstream gradient $(W^T \delta)$ modulated by the local derivative $\sigma'(z)$.

---

### Q23: What causes vanishing/exploding gradients and how are they addressed?

**Answer:**

For deep network with $L$ layers, gradient magnitude scales as:
$$\prod_{l=1}^L \|W_l\| \cdot |\sigma'(z_l)|$$

- **Vanishing:** $\|W_l\| \cdot |\sigma'| < 1$ repeatedly → gradient → 0
- **Exploding:** $\|W_l\| \cdot |\sigma'| > 1$ repeatedly → gradient → ∞

**Solutions:**
| Technique | How it helps |
|-----------|-------------|
| ReLU | $\sigma'(x) = 1$ for $x > 0$ (no shrinkage) |
| Residual connections | Gradient flows through skip: $\frac{\partial}{\partial x}(x + F(x)) = 1 + F'(x)$ |
| Batch normalization | Keeps activations well-conditioned |
| Xavier/He initialization | Sets $\text{Var}(W) = 1/n_{\text{in}}$ or $2/n_{\text{in}}$ |
| Gradient clipping | Caps gradient norm to threshold |

---

### Q24: Explain batch normalization mathematically. Why does it help?

**Answer:**

**Forward pass** (for mini-batch $\mathcal{B}$):
$$\mu_\mathcal{B} = \frac{1}{m}\sum_i x_i, \quad \sigma^2_\mathcal{B} = \frac{1}{m}\sum_i(x_i - \mu_\mathcal{B})^2$$
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \quad y_i = \gamma\hat{x}_i + \beta$$

**Why it helps:**
1. Reduces internal covariate shift (distribution of inputs to each layer stabilizes)
2. Allows higher learning rates (smoother loss landscape)
3. Acts as regularizer (batch statistics add noise)
4. Makes the loss landscape smoother: $\|\nabla L\|$ varies less

---

## Generative Models Math Questions

### Q25: Derive the ELBO for VAEs.

**Answer:**

Start with log-likelihood:
$$\log p(x) = \log \int p(x|z)p(z)dz$$

Introduce variational distribution $q(z|x)$:
$$\log p(x) = \underbrace{E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))}_{\text{ELBO}} + D_{KL}(q(z|x) \| p(z|x))$$

Since KL ≥ 0: $\log p(x) \geq \text{ELBO}$

**ELBO = Reconstruction - KL:**
- Reconstruction: How well can we decode $z$ back to $x$?
- KL: How close is the encoder to the prior?

---

### Q26: What is the GAN objective and what does the optimal discriminator look like?

**Answer:**

$$\min_G \max_D \; E_{x \sim p_{\text{data}}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Optimal discriminator** (for fixed $G$):
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$$

With $D^*$, the generator minimizes:
$$2 \cdot D_{JS}(p_{\text{data}} \| p_G) - \log 4$$

where $D_{JS}$ is the Jensen-Shannon divergence.

**Problem:** When $D$ is too good, $\log(1 - D(G(z)))$ saturates → vanishing gradients for $G$.

---

## Quick Review Checklist

### Before Your Interview, Make Sure You Can:

**Linear Algebra:**

- [ ] Multiply matrices and explain the dimensions
- [ ] Explain eigenvalue decomposition geometrically
- [ ] Describe SVD and its applications
- [ ] Define and identify positive definite matrices
- [ ] Explain rank and its implications

**Calculus:**

- [ ] Derive common function derivatives (sigmoid, softmax)
- [ ] Apply the chain rule for backpropagation
- [ ] Explain gradient, Jacobian, and Hessian
- [ ] Use Taylor series for approximations

**Probability:**

- [ ] Derive and apply Bayes' theorem
- [ ] Explain MLE vs MAP
- [ ] Describe common distributions
- [ ] Explain bias-variance tradeoff
- [ ] Define and compute expectation, variance, covariance

**Optimization:**

- [ ] Explain gradient descent variants
- [ ] Describe convexity and its importance
- [ ] Walk through Adam optimizer
- [ ] Explain regularization mathematically

**Information Theory:**

- [ ] Define and compute entropy
- [ ] Explain cross-entropy loss
- [ ] Describe KL divergence and its properties

**Applied:**

- [ ] Derive gradients for logistic regression
- [ ] Explain batch normalization math
- [ ] Describe attention mechanism math

---

## Tips for Math Interviews

1. **Start with intuition** before diving into formulas
2. **Draw pictures** - geometric intuition is powerful
3. **Connect to ML applications** - show you understand why it matters
4. **Be honest** if you don't know something
5. **Practice derivations** by hand
6. **Understand, don't memorize** - interviewers test understanding

---

## 📅 Study Plan

### Week 1: Foundations
- **Day 1-2:** Linear algebra (vectors, matrices, eigenvalues)
- **Day 3-4:** Calculus (derivatives, gradients, chain rule)
- **Day 5-6:** Probability (Bayes, distributions, expectation)
- **Day 7:** Review + practice problems

### Week 2: Core ML Math
- **Day 1-2:** Optimization (gradient descent, convexity, Adam)
- **Day 3-4:** Information theory (entropy, KL, cross-entropy)
- **Day 5-6:** Statistics (MLE, MAP, hypothesis testing)
- **Day 7:** Review + mock interview

### Week 3: Advanced Topics
- **Day 1-2:** Deep learning math (backprop, batch norm, attention)
- **Day 3-4:** Generative models (VAE ELBO, GAN theory, diffusion)
- **Day 5-6:** Graph theory + kernel methods
- **Day 7:** Full review + timed practice

### Daily Practice Routine
1. ⏰ **10 min:** Review 5 formulas from cheatsheet
2. 📝 **20 min:** Derive one key result by hand
3. 💻 **20 min:** Implement one concept in code
4. 🎯 **10 min:** Answer one interview question aloud

---

_Good luck with your interviews!_ 🍀
