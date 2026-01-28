# 🗺️ ML Math Map

> A comprehensive guide showing which mathematics is used where in Machine Learning.

---

## Table of Contents

1. [Overview Diagram](#overview-diagram)
2. [Linear Algebra in ML](#linear-algebra-in-ml)
3. [Calculus in ML](#calculus-in-ml)
4. [Probability & Statistics in ML](#probability--statistics-in-ml)
5. [Optimization in ML](#optimization-in-ml)
6. [Information Theory in ML](#information-theory-in-ml)
7. [By ML Model/Algorithm](#by-ml-modelalgorithm)
8. [By Deep Learning Component](#by-deep-learning-component)

---

## Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MATHEMATICS FOR MACHINE LEARNING                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│LINEAR ALGEBRA │           │   CALCULUS    │           │ PROBABILITY   │
├───────────────┤           ├───────────────┤           ├───────────────┤
│• Data repr.   │           │• Optimization │           │• Uncertainty  │
│• Transforms   │           │• Gradients    │           │• Inference    │
│• Projections  │           │• Backprop     │           │• Distributions│
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│ OPTIMIZATION  │           │ INFO THEORY   │           │  NUMERICAL    │
├───────────────┤           ├───────────────┤           ├───────────────┤
│• Training     │           │• Loss funcs   │           │• Stability    │
│• Convergence  │           │• Compression  │           │• Precision    │
│• Regularize   │           │• Information  │           │• Efficiency   │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## Linear Algebra in ML

### Core Concepts → ML Applications

| Linear Algebra Concept    | ML Application                  | Example                                       |
| ------------------------- | ------------------------------- | --------------------------------------------- |
| **Vectors**               | Feature representation          | Each data point as a vector                   |
| **Matrix multiplication** | Forward propagation             | $\mathbf{y} = W\mathbf{x} + \mathbf{b}$       |
| **Dot product**           | Similarity measurement          | Cosine similarity                             |
| **Matrix transpose**      | Gradient computation            | $\nabla_W = \mathbf{x}^T \delta$              |
| **Matrix inverse**        | Linear regression (closed form) | $\hat{\mathbf{w}} = (X^TX)^{-1}X^T\mathbf{y}$ |
| **Eigenvalues/vectors**   | PCA, spectral clustering        | Dimensionality reduction                      |
| **SVD**                   | Recommender systems             | Matrix factorization                          |
| **Orthogonality**         | Feature decorrelation           | Gram-Schmidt in NNs                           |
| **Determinant**           | Change of variables             | Normalizing flows                             |
| **Trace**                 | Regularization                  | $\text{tr}(W^TW)$                             |
| **Rank**                  | Model capacity                  | Low-rank approximation                        |
| **Positive definiteness** | Covariance matrices             | Gaussian distributions                        |

### Detailed Breakdown

#### Data Representation

```
Data Matrix X ∈ ℝ^(n×d)
─────────────────────────
         Features
         d columns
      ┌─────────────┐
    n │  x₁₁ ··· x₁d│  → Sample 1
rows  │   ⋮  ⋱   ⋮  │
      │  xn₁ ··· xnd│  → Sample n
      └─────────────┘

Used in: Every ML algorithm!
```

#### Neural Network Layer

```
Linear Layer: y = Wx + b
─────────────────────────
Input x ∈ ℝ^d       →  Output y ∈ ℝ^m
Weight W ∈ ℝ^(m×d)
Bias b ∈ ℝ^m

Matrix mult = linear transformation
```

#### Eigendecomposition Applications

| Application         | How Eigenvalues Are Used            |
| ------------------- | ----------------------------------- |
| PCA                 | Eigenvectors = principal components |
| PageRank            | Dominant eigenvector = page scores  |
| Spectral Clustering | Eigenvectors of Laplacian           |
| Markov Chains       | Stationary distribution             |
| Recurrent NNs       | Stability analysis                  |

---

## Calculus in ML

### Core Concepts → ML Applications

| Calculus Concept        | ML Application                | Example                                                                                      |
| ----------------------- | ----------------------------- | -------------------------------------------------------------------------------------------- |
| **Derivatives**         | Gradient computation          | $\frac{\partial L}{\partial w}$                                                              |
| **Chain rule**          | Backpropagation               | $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y}\frac{\partial y}{\partial w}$ |
| **Partial derivatives** | Multivariate optimization     | Updating each weight                                                                         |
| **Gradient**            | Direction of steepest descent | $\nabla L$                                                                                   |
| **Jacobian**            | Multi-output functions        | Neural network layers                                                                        |
| **Hessian**             | Second-order optimization     | Newton's method                                                                              |
| **Taylor series**       | Local approximations          | Optimization analysis                                                                        |
| **Integration**         | Probability densities         | Normalization constants                                                                      |

### Backpropagation Flow

```
Forward Pass:
─────────────
Input → [Linear] → [Activation] → [Linear] → [Activation] → Output → Loss
  x        z₁          a₁           z₂          a₂           ŷ        L

Backward Pass (Chain Rule):
───────────────────────────
∂L/∂W₂ = ∂L/∂ŷ · ∂ŷ/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂W₂

∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂a₂ · ∂a₂/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁
         └──────────────────────────────────────────────────────────┘
                          Chain Rule in Action!
```

### Gradient Visualization

```
Loss Surface
─────────────
     High Loss
         ╱╲
        ╱  ╲
       ╱    ╲ ← ∇L points uphill
      ╱  ·   ╲
     ╱   ↓    ╲
    ╱  current ╲
   ╱   position ╲
  ╱              ╲
 ╱      ★        ╲ ← minimum (goal)
─────────────────────
     Low Loss

Update: w ← w - η∇L  (move opposite to gradient)
```

---

## Probability & Statistics in ML

### Core Concepts → ML Applications

| Probability Concept           | ML Application             | Example                |
| ----------------------------- | -------------------------- | ---------------------- | --- |
| **Probability distributions** | Modeling uncertainty       | Output probabilities   |
| **Bayes' theorem**            | Bayesian inference         | Naive Bayes, posterior |
| **Conditional probability**   | Classification             | $P(y                   | x)$ |
| **Joint distributions**       | Generative models          | $P(x, y)$              |
| **Expectation**               | Loss functions             | $E[L]$                 |
| **Variance**                  | Uncertainty quantification | Prediction intervals   |
| **Covariance**                | Feature relationships      | Multivariate Gaussian  |
| **MLE**                       | Parameter estimation       | Training               |
| **MAP**                       | Regularization             | L2 as Gaussian prior   |
| **Sampling**                  | Monte Carlo methods        | Dropout, MCMC          |

### Probabilistic View of ML

```
Discriminative vs Generative
────────────────────────────

Discriminative: Model P(y|x) directly
┌─────────────────────────────────────┐
│  Given input x, what is output y?   │
│                                     │
│  Examples: Logistic Regression,     │
│            Neural Networks, SVM     │
└─────────────────────────────────────┘

Generative: Model P(x,y) = P(x|y)P(y)
┌─────────────────────────────────────┐
│  How is the data generated?         │
│                                     │
│  Examples: Naive Bayes, GMM,        │
│            VAE, GAN                 │
└─────────────────────────────────────┘
```

### Regularization as Prior

```
MLE:  θ̂ = argmax P(D|θ)
                ↓
      No regularization

MAP:  θ̂ = argmax P(D|θ)P(θ)
                       ↑
                    Prior!

Gaussian prior P(θ) ∝ exp(-λ||θ||²)  →  L2 Regularization
Laplace prior P(θ) ∝ exp(-λ||θ||₁)   →  L1 Regularization
```

---

## Optimization in ML

### Core Concepts → ML Applications

| Optimization Concept         | ML Application            | Example            |
| ---------------------------- | ------------------------- | ------------------ |
| **Gradient descent**         | Training all models       | Weight updates     |
| **SGD**                      | Large-scale training      | Batch processing   |
| **Momentum**                 | Faster convergence        | SGD + momentum     |
| **Adam**                     | Adaptive learning         | Most deep learning |
| **Convexity**                | Global optimum guarantee  | Linear regression  |
| **Lagrange multipliers**     | Constraints               | SVM dual problem   |
| **Learning rate scheduling** | Training stability        | Warmup, decay      |
| **Newton's method**          | Second-order optimization | Natural gradient   |

### Optimizer Comparison

```
Optimization Landscape Navigation
─────────────────────────────────

GD:        SGD:       Momentum:    Adam:
  ↓          ↓ ↘        ↓→→→→→      ↓→→
  ↓          ↓↙         ↓           ↓→→
  ↓        ↙↓           ↓           ↓→→
  ↓       ↓  ↘          ↓           ↓
  ★        ↘ ★         ★           ★

Smooth    Noisy      Accelerated  Adaptive
but slow  but        and smooth   per-param
          escapes                 learning
          local min              rate
```

### Learning Rate Effect

```
Learning Rate (η) Effects
─────────────────────────

η too small:          η just right:         η too large:
     ·                     ·                     ·
    ╱                     ╱                     ╱ ╲
   ╱                     ╱                     ╱   ╲
  ·                     ·                     ·     ·
 ╱                     ╱                           ╱
·····★               ★                      ·····

Slow                 Converges             Diverges
convergence          smoothly              or oscillates
```

---

## Information Theory in ML

### Core Concepts → ML Applications

| Information Theory Concept | ML Application          | Example              |
| -------------------------- | ----------------------- | -------------------- |
| **Entropy**                | Uncertainty measurement | Decision tree splits |
| **Cross-entropy**          | Classification loss     | Softmax + CE loss    |
| **KL divergence**          | Distribution comparison | VAE loss, KD         |
| **Mutual information**     | Feature selection       | InfoGAN              |
| **Information gain**       | Feature importance      | Random forests       |

### Loss Functions from Information Theory

```
Cross-Entropy Loss
──────────────────

For classification with K classes:

L = -∑ᵢ yᵢ log(ŷᵢ)

Where:
- yᵢ = true distribution (one-hot)
- ŷᵢ = predicted probabilities (softmax output)

Example (3 classes):
True:      [1, 0, 0]
Predicted: [0.7, 0.2, 0.1]
Loss = -[1·log(0.7) + 0·log(0.2) + 0·log(0.1)]
     = -log(0.7) ≈ 0.357
```

### KL Divergence in VAEs

```
VAE Loss = Reconstruction Loss + KL Divergence
─────────────────────────────────────────────

            ┌──────────────┐
Input x  →  │   Encoder    │  →  μ, σ  →  z ~ N(μ,σ²)
            └──────────────┘                   │
                                               ▼
            ┌──────────────┐              ┌────────┐
Output x̂ ←  │   Decoder    │  ←───────── │ Sample │
            └──────────────┘              └────────┘

L = E[log p(x|z)] - KL(q(z|x) || p(z))
    ─────────────   ─────────────────
    Reconstruction   Regularization
    (want high)      (want low)
```

---

## By ML Model/Algorithm

### Linear Regression

| Math Concept   | Where Used                                                      |
| -------------- | --------------------------------------------------------------- |
| Linear algebra | $\hat{y} = X\mathbf{w}$                                         |
| Matrix inverse | Normal equations: $\hat{\mathbf{w}} = (X^TX)^{-1}X^T\mathbf{y}$ |
| Calculus       | Gradient descent: $\nabla_w L = X^T(X\mathbf{w} - \mathbf{y})$  |
| Statistics     | MSE loss, R² score                                              |

### Logistic Regression

| Math Concept       | Where Used                       |
| ------------------ | -------------------------------- | --------------- |
| Linear algebra     | $z = \mathbf{w}^T\mathbf{x} + b$ |
| Calculus           | Sigmoid derivative, gradient     |
| Probability        | $P(y=1                           | x) = \sigma(z)$ |
| Information theory | Cross-entropy loss               |

### Support Vector Machine (SVM)

| Math Concept   | Where Used                                                         |
| -------------- | ------------------------------------------------------------------ |
| Linear algebra | Hyperplane: $\mathbf{w}^T\mathbf{x} + b = 0$                       |
| Calculus       | Gradient of hinge loss                                             |
| Optimization   | Lagrange multipliers, dual problem                                 |
| Kernel methods | $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T\phi(\mathbf{x}')$ |

### Principal Component Analysis (PCA)

| Math Concept       | Where Used                   |
| ------------------ | ---------------------------- |
| Linear algebra     | Covariance matrix            |
| Eigendecomposition | Finding principal components |
| Optimization       | Maximize variance            |
| Statistics         | Explained variance ratio     |

### Decision Trees / Random Forests

| Math Concept       | Where Used                      |
| ------------------ | ------------------------------- |
| Probability        | Class probabilities in leaves   |
| Information theory | Information gain, Gini impurity |
| Statistics         | Bootstrap sampling (RF)         |

### K-Means Clustering

| Math Concept   | Where Used                         |
| -------------- | ---------------------------------- |
| Linear algebra | Distance calculations              |
| Calculus       | Minimizing within-cluster variance |
| Optimization   | EM-like algorithm                  |

### Neural Networks

| Math Concept       | Where Used              |
| ------------------ | ----------------------- |
| Linear algebra     | Every layer computation |
| Calculus           | Backpropagation         |
| Probability        | Output layer, dropout   |
| Optimization       | SGD, Adam               |
| Information theory | Cross-entropy loss      |

### Transformers

| Math Concept   | Where Used                              |
| -------------- | --------------------------------------- |
| Linear algebra | Attention: $QK^T/\sqrt{d}$, projections |
| Calculus       | Backprop through attention              |
| Probability    | Softmax attention weights               |
| Optimization   | Adam, learning rate schedules           |

---

## By Deep Learning Component

### Activation Functions

| Function | Math Involved                                |
| -------- | -------------------------------------------- |
| Sigmoid  | $\sigma(x) = \frac{1}{1+e^{-x}}$, derivative |
| Tanh     | Hyperbolic functions                         |
| ReLU     | Piecewise functions                          |
| Softmax  | Exponentials, normalization                  |
| GELU     | Gaussian CDF                                 |

### Loss Functions

| Loss          | Math Involved         |
| ------------- | --------------------- |
| MSE           | Expectation, variance |
| Cross-Entropy | Information theory    |
| Hinge         | Optimization theory   |
| Triplet       | Distance metrics      |
| Contrastive   | Information theory    |

### Regularization

| Technique  | Math Involved               |
| ---------- | --------------------------- |
| L1         | Norms, sparsity             |
| L2         | Norms, Gaussian prior       |
| Dropout    | Probability, sampling       |
| Batch Norm | Statistics (mean, variance) |
| Layer Norm | Statistics                  |

### Attention Mechanism

```
Scaled Dot-Product Attention
────────────────────────────

Attention(Q, K, V) = softmax(QK^T / √dₖ) V

Math involved:
├── Matrix multiplication: QK^T
├── Scaling: √dₖ (for numerical stability)
├── Softmax: probability distribution
└── Weighted sum: attention × V

Multi-Head Attention:
├── Multiple parallel attention
├── Concatenation
└── Linear projection
```

---

## Quick Reference by Task

### Classification

- Linear algebra: Feature vectors, weight matrices
- Probability: Class probabilities, Bayes
- Information theory: Cross-entropy loss
- Optimization: Gradient descent

### Regression

- Linear algebra: Matrix equations
- Calculus: Gradients, optimization
- Statistics: MSE, R², residuals

### Clustering

- Linear algebra: Distance metrics
- Optimization: K-means objective
- Probability: GMM

### Dimensionality Reduction

- Linear algebra: SVD, eigendecomposition
- Statistics: Variance explained
- Optimization: Reconstruction error

### Generative Models

- Probability: Distributions, sampling
- Information theory: KL divergence
- Linear algebra: Transformations

### Reinforcement Learning

- Probability: Markov decision processes
- Optimization: Policy gradient
- Linear algebra: Value function approximation

---

_"The math you need depends on what you're building!"_ 🔧
