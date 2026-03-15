[← Back to Math for Specific Models](../README.md) | [Next: Neural Networks →](../02-Neural-Networks/notes.md)

---

# Linear Models: Mathematical Foundations

> _"The method of least squares is the automobile of modern statistical analysis: despite its limitations, occasional accidents, and incidental pollution, it and its relatives will remain in use as long as we need to get from here to there quickly."_
> — Stephen Stigler

## Overview

Linear models are simultaneously the simplest and most profound family of machine learning methods. Every neural network layer is a linear map. Every attention head computes a weighted linear combination of values. Every weight decay regulariser is a ridge penalty. Understanding linear models rigorously — their geometry, their statistical properties, their connections to Bayesian inference, and their role inside modern LLMs — is not a preliminary step on the way to "real" ML: it _is_ real ML.

This section develops the full mathematical theory from the ordinary least squares projection through Bayesian inference, regularisation geometry, linear classification, support vector machines, and the double descent phenomenon. The final sections connect directly to LoRA, neural tangent kernels, and the linear-algebraic structure of Transformer attention.

The treatment is rigorous throughout: every estimator is derived from first principles, every key theorem is proved, and every formula is connected to its computational implementation and its role in modern AI systems.

## Prerequisites

- Linear algebra: matrix multiplication, projections, eigendecomposition, SVD (Chapters 02–03)
- Calculus: partial derivatives, gradients, matrix calculus (Chapter 04)
- Probability: Gaussian distribution, conditional probability, Bayes' theorem (Chapter 06)
- Statistics: MLE, bias, variance, hypothesis testing (Chapter 07)
- Optimisation: gradient descent, convexity, KKT conditions (Chapter 08)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: OLS geometry, ridge shrinkage paths, Lasso coordinate descent, Bayesian posteriors, SVM margins, bias-variance curves, LoRA visualisation |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: OLS geometry proof, ridge Bayesian equivalence, Lasso soft-thresholding, Gaussian posterior derivation, logistic gradient, LDA classifier, SVM dual, bias-variance decomposition, LoRA rank analysis, NTK regression |

## Learning Objectives

After completing this section, you will:

- Derive the OLS estimator $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$ both algebraically (normal equations) and geometrically (projection onto column space)
- Prove the Gauss-Markov theorem: OLS is BLUE under the classical assumptions
- Express Ridge regression as SVD singular-value shrinkage and as a Gaussian MAP estimator
- Derive the Lasso coordinate descent update and the soft-thresholding operator from KKT conditions
- Compute and interpret the Bayesian linear regression predictive distribution, including epistemic and aleatoric uncertainty
- Derive logistic regression's gradient and Hessian, and prove global convexity of the cross-entropy loss
- Derive the SVM dual formulation from the primal via Lagrangian duality and KKT conditions
- State the bias-variance decomposition and explain the double descent phenomenon in overparameterised models
- Explain how LoRA, weight decay, probing classifiers, and Transformer attention are all instances of linear model mathematics
- Choose the correct model, regulariser, and optimiser for a given ML problem based on its statistical structure

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 Why Linear Models Still Matter](#11-why-linear-models-still-matter)
  - [1.2 The Geometric Core](#12-the-geometric-core)
  - [1.3 Linear Models Inside Neural Networks](#13-linear-models-inside-neural-networks)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. The Linear Regression Framework](#2-the-linear-regression-framework)
  - [2.1 Model Specification](#21-model-specification)
  - [2.2 Ordinary Least Squares Derivation](#22-ordinary-least-squares-derivation)
  - [2.3 The Hat Matrix and Projection Geometry](#23-the-hat-matrix-and-projection-geometry)
  - [2.4 SVD-Based Solution](#24-svd-based-solution)
  - [2.5 BLUE and the Gauss-Markov Theorem](#25-blue-and-the-gauss-markov-theorem)
- [3. Statistical Inference for OLS](#3-statistical-inference-for-ols)
  - [3.1 Distributional Properties of OLS](#31-distributional-properties-of-ols)
  - [3.2 Estimating the Noise Variance](#32-estimating-the-noise-variance)
  - [3.3 Hypothesis Testing](#33-hypothesis-testing)
  - [3.4 Confidence and Prediction Intervals](#34-confidence-and-prediction-intervals)
  - [3.5 R² and Goodness of Fit](#35-r-and-goodness-of-fit)
- [4. Ridge Regression (L2 Regularisation)](#4-ridge-regression-l2-regularisation)
  - [4.1 Objective and Closed-Form Solution](#41-objective-and-closed-form-solution)
  - [4.2 SVD View and Singular Value Shrinkage](#42-svd-view-and-singular-value-shrinkage)
  - [4.3 Bayesian Interpretation](#43-bayesian-interpretation)
  - [4.4 Ridge Path and Effective Degrees of Freedom](#44-ridge-path-and-effective-degrees-of-freedom)
  - [4.5 Tikhonov Regularisation and Weight Decay](#45-tikhonov-regularisation-and-weight-decay)
- [5. Lasso Regression (L1 Regularisation)](#5-lasso-regression-l1-regularisation)
  - [5.1 Objective and Geometric Sparsity](#51-objective-and-geometric-sparsity)
  - [5.2 KKT Conditions and Soft-Thresholding](#52-kkt-conditions-and-soft-thresholding)
  - [5.3 Coordinate Descent Algorithm](#53-coordinate-descent-algorithm)
  - [5.4 LARS Algorithm](#54-lars-algorithm)
  - [5.5 Lasso as Laplace Prior MAP](#55-lasso-as-laplace-prior-map)
- [6. Elastic Net and Structured Sparsity](#6-elastic-net-and-structured-sparsity)
  - [6.1 Elastic Net Objective](#61-elastic-net-objective)
  - [6.2 Group Lasso](#62-group-lasso)
  - [6.3 Nuclear Norm Regularisation](#63-nuclear-norm-regularisation)
- [7. Bayesian Linear Regression](#7-bayesian-linear-regression)
  - [7.1 The Prior-Posterior Framework](#71-the-prior-posterior-framework)
  - [7.2 Posterior Mean and Variance](#72-posterior-mean-and-variance)
  - [7.3 The Predictive Distribution](#73-the-predictive-distribution)
  - [7.4 Evidence and Marginal Likelihood](#74-evidence-and-marginal-likelihood)
  - [7.5 Connection to Gaussian Processes and NTK](#75-connection-to-gaussian-processes-and-ntk)
- [8. Linear Classification](#8-linear-classification)
  - [8.1 Logistic Regression](#81-logistic-regression)
  - [8.2 Gradient, Hessian, and Global Convexity](#82-gradient-hessian-and-global-convexity)
  - [8.3 Softmax Regression (Multiclass)](#83-softmax-regression-multiclass)
  - [8.4 Decision Boundaries](#84-decision-boundaries)
  - [8.5 Maximum Entropy Interpretation](#85-maximum-entropy-interpretation)
- [9. Discriminative vs. Generative Models](#9-discriminative-vs-generative-models)
  - [9.1 The Fundamental Modelling Choice](#91-the-fundamental-modelling-choice)
  - [9.2 Linear Discriminant Analysis](#92-linear-discriminant-analysis)
  - [9.3 Quadratic Discriminant Analysis](#93-quadratic-discriminant-analysis)
  - [9.4 Naive Bayes](#94-naive-bayes)
  - [9.5 When Generative Wins and When Discriminative Wins](#95-when-generative-wins-and-when-discriminative-wins)
- [10. Support Vector Machines](#10-support-vector-machines)
  - [10.1 Margin Maximisation](#101-margin-maximisation)
  - [10.2 Lagrangian Duality and the Dual Problem](#102-lagrangian-duality-and-the-dual-problem)
  - [10.3 Soft Margin SVM](#103-soft-margin-svm)
  - [10.4 The Kernel Trick](#104-the-kernel-trick)
  - [10.5 Structural Risk Minimisation](#105-structural-risk-minimisation)
- [11. Bias-Variance Tradeoff and Model Selection](#11-bias-variance-tradeoff-and-model-selection)
  - [11.1 Bias-Variance Decomposition](#111-bias-variance-decomposition)
  - [11.2 The Double Descent Phenomenon](#112-the-double-descent-phenomenon)
  - [11.3 Cross-Validation](#113-cross-validation)
  - [11.4 Information Criteria](#114-information-criteria)
  - [11.5 Regularisation Paths and Hyperparameter Tuning](#115-regularisation-paths-and-hyperparameter-tuning)
- [12. Optimisation Methods for Linear Models](#12-optimisation-methods-for-linear-models)
  - [12.1 Gradient Descent and Convergence Analysis](#121-gradient-descent-and-convergence-analysis)
  - [12.2 Newton's Method and IRLS](#122-newtons-method-and-irls)
  - [12.3 Stochastic Gradient Descent](#123-stochastic-gradient-descent)
  - [12.4 Coordinate Descent](#124-coordinate-descent)
  - [12.5 Second-Order Methods: L-BFGS](#125-second-order-methods-l-bfgs)
- [13. Deep Learning and LLM Connections](#13-deep-learning-and-llm-connections)
  - [13.1 Linear Layers as Generalised Linear Models](#131-linear-layers-as-generalised-linear-models)
  - [13.2 LoRA as Low-Rank Linear Regression](#132-lora-as-low-rank-linear-regression)
  - [13.3 Probing Classifiers](#133-probing-classifiers)
  - [13.4 Neural Tangent Kernel](#134-neural-tangent-kernel)
  - [13.5 Attention as Linear Mixing](#135-attention-as-linear-mixing)
- [14. Common Mistakes](#14-common-mistakes)
- [15. Exercises](#15-exercises)
- [16. Why This Matters for AI (2026)](#16-why-this-matters-for-ai-2026)
- [17. Conceptual Bridge](#17-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 Why Linear Models Still Matter

In 2026, the most capable AI systems are billion-parameter Transformers trained on trillions of tokens. Yet every major capability — in-context learning, instruction following, chain-of-thought reasoning — is probed, understood, and improved using **linear models**. This is not a coincidence.

Linear models have three properties that make them irreplaceable:

1. **Interpretability**: The weight vector $\boldsymbol{\beta} \in \mathbb{R}^d$ has a direct semantic interpretation — each component quantifies the effect of one feature on the output. No other model class offers this transparency at scale.

2. **Computational efficiency**: The OLS solution $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$ can be computed in $O(nd^2 + d^3)$ time. For $n = 10^6$ samples and $d = 10^3$ features, this is milliseconds on a modern CPU.

3. **Theoretical tractability**: Linear models admit exact bias-variance analysis, exact posterior distributions (Bayesian regression), exact generalisation bounds (VC theory), and exact characterisation of interpolating solutions (minimum-norm estimators). These theoretical insights transfer to neural networks via the neural tangent kernel.

**For AI:** GPT-style models are probed almost exclusively with linear classifiers (Alain & Bengio, 2017; Tenney et al., 2019). If a concept is linearly decodable from a layer's activations, that layer "represents" the concept. The entire mechanistic interpretability programme at Anthropic and DeepMind rests on this principle.

### 1.2 The Geometric Core

All of linear model theory reduces to two geometric operations in $\mathbb{R}^n$:

**Regression = Orthogonal Projection.** The target vector $\mathbf{y} \in \mathbb{R}^n$ lives in observation space. The predictions $\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}}$ live in the column space $\mathcal{C}(X) \subseteq \mathbb{R}^n$ — a $d$-dimensional subspace. OLS finds the point in $\mathcal{C}(X)$ closest to $\mathbf{y}$ in Euclidean distance:

$$\hat{\mathbf{y}} = \underset{\mathbf{v} \in \mathcal{C}(X)}{\arg\min} \lVert \mathbf{y} - \mathbf{v} \rVert_2$$

The residual $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is perpendicular to every column of $X$: $X^\top \mathbf{e} = \mathbf{0}$.

**Classification = Hyperplane Separation.** A linear classifier partitions $\mathbb{R}^d$ into half-spaces using a hyperplane $\{\mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0\}$. The weight vector $\mathbf{w}$ points in the direction of maximum class separation; the margin is the distance from the hyperplane to the nearest data point.

**Three inductive biases** that linear models embody:
- **Additivity**: the effect of each feature is independent and additive
- **Proportionality**: doubling a feature doubles its contribution to the output
- **Global linearity**: the same linear map applies everywhere in input space

All three fail for real data at some scale — which is why we need nonlinear models. But understanding when and why they fail requires first understanding the linear case rigorously.

### 1.3 Linear Models Inside Neural Networks

Every neural network is a composition of linear maps and nonlinear activations. Strip away the nonlinearities and every layer reduces to a linear model. This is not merely a theoretical curiosity — it is the key to understanding:

| Neural Network Component | Linear Model Equivalent |
|---|---|
| `nn.Linear(d, k)` | OLS with $k$ output variables |
| Weight decay ($\lambda \lVert W \rVert_F^2$) | Ridge regression with $\lambda$ |
| L1 weight regularisation | Lasso regression |
| LoRA weight update $\Delta W = BA$ | Rank-constrained linear regression |
| Attention output $AV$ | Linear mixing of value vectors |
| Linear probe on activations | Logistic regression on fixed features |
| Fine-tuning last layer | Transfer learning = linear model on frozen features |

**For AI (2026):** LoRA (Hu et al., 2022) and its successors (DoRA, LoRA+, Flora) are the dominant parameter-efficient fine-tuning methods for LLMs. Their mathematical foundation is exactly rank-constrained linear regression: $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d,k)$. The nuclear norm regularisation that controls rank in matrix completion is the convex relaxation of the Lasso applied to singular values.

### 1.4 Historical Timeline

```
HISTORY OF LINEAR MODELS
════════════════════════════════════════════════════════════════════════

  1805  Legendre publishes least squares (Nouvelles méthodes)
  1809  Gauss claims prior derivation; proves OLS = MLE under Gaussian noise
  1886  Galton introduces "regression to the mean" — coins the term
  1901  Pearson introduces principal components (linear dimensionality reduction)
  1936  Fisher introduces LDA — optimal linear classifier under Gaussians
  1943  Wald proves minimax optimality of linear estimators
  1951  Dantzig: linear programming dual — foundation of SVM duality
  1959  Ridge regression (Hoerl & Kennard 1970, independently earlier)
  1964  Tikhonov regularisation for ill-posed inverse problems
  1979  Golub & Van Loan: numerically stable SVD algorithms
  1986  IRLS for GLMs (Green 1984); coordinate descent for Lasso (Fu 1998)
  1992  Boser, Guyon, Vapnik: SVMs with kernel trick
  1994  Tibshirani: Lasso (Least Absolute Shrinkage and Selection Operator)
  1995  Vapnik: Statistical Learning Theory, VC dimension bounds
  1996  Efron et al.: LARS algorithm — piecewise linear Lasso path
  2001  Ng & Jordan: discriminative vs. generative classifiers
  2003  Zou & Hastie: Elastic Net
  2006  Donoho: compressed sensing — Lasso recovers sparse signals from few measurements
  2019  Jacot et al.: Neural Tangent Kernel — infinite-width NN = kernel regression
  2020  Belkin et al.: double descent — overfitting is benign in overparameterised models
  2022  Hu et al.: LoRA — rank-decomposed weight updates for LLM fine-tuning
  2024  Zhao et al.: GaLore — gradient low-rank projection for pretraining

════════════════════════════════════════════════════════════════════════
```

---

## 2. The Linear Regression Framework

### 2.1 Model Specification

The linear regression model assumes:

$$y^{(i)} = \boldsymbol{\beta}^\top \mathbf{x}^{(i)} + \epsilon^{(i)}, \quad i = 1, \ldots, n$$

where $\mathbf{x}^{(i)} \in \mathbb{R}^d$ is the feature vector for observation $i$, $\boldsymbol{\beta} \in \mathbb{R}^d$ is the unknown parameter vector, $y^{(i)} \in \mathbb{R}$ is the scalar response, and $\epsilon^{(i)}$ is the noise term.

**Vectorised form.** Stacking all $n$ observations:

$$\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\epsilon}$$

where $X \in \mathbb{R}^{n \times d}$ is the **design matrix** (row $i$ = $(\mathbf{x}^{(i)})^\top$), $\mathbf{y} \in \mathbb{R}^n$, $\boldsymbol{\epsilon} \in \mathbb{R}^n$.

**Anatomy of the design matrix:**

```
DESIGN MATRIX STRUCTURE
════════════════════════════════════════════════════════════════════════

         feature 1  feature 2  ...  feature d
         ─────────  ─────────       ─────────
  obs 1 │  x₁₁       x₁₂    ...    x₁ₐ     │   y₁
  obs 2 │  x₂₁       x₂₂    ...    x₂ₐ     │   y₂
   ...  │   ...        ...          ...     │   ...
  obs n │  xₙ₁       xₙ₂    ...    xₙₐ     │   yₙ

  • Column j of X = measurements of feature j across all observations
  • Row i of X    = all features of observation i (the feature vector x^(i)ᵀ)
  • Intercept: prepend a column of 1s if needed → β₀ is the intercept
  • Convention: β ∈ ℝᵈ absorbs intercept if X already includes the ones column

════════════════════════════════════════════════════════════════════════
```

**The five Gauss-Markov assumptions:**

| Assumption | Mathematical statement | Consequence if violated |
|---|---|---|
| Linearity | $\mathbb{E}[\mathbf{y} \mid X] = X\boldsymbol{\beta}$ | Bias — model systematically wrong |
| Strict exogeneity | $\mathbb{E}[\boldsymbol{\epsilon} \mid X] = \mathbf{0}$ | OLS is biased (endogeneity) |
| Full rank | $\operatorname{rank}(X) = d$ | $(X^\top X)^{-1}$ doesn't exist |
| Homoscedasticity | $\operatorname{Cov}(\boldsymbol{\epsilon} \mid X) = \sigma^2 I_n$ | OLS is unbiased but not efficient |
| No autocorrelation | $\operatorname{Cov}(\epsilon^{(i)}, \epsilon^{(j)}) = 0$ for $i \neq j$ | Standard errors wrong |

The assumptions of normality ($\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$) are needed for exact finite-sample inference but not for the BLUE property.

**Non-examples of linear regression** (things that look linear but aren't):
- $y = \beta_0 + \beta_1 x + \beta_2 x^2$ — this IS linear regression (in features $1, x, x^2$)
- $y = \beta_0 e^{\beta_1 x}$ — this is NOT (the model is nonlinear in $\beta_1$)
- $y = \beta_0 + \beta_1 / (x + \beta_2)$ — NOT linear in parameters

The key test: **linear in $\boldsymbol{\beta}$**, not necessarily in $\mathbf{x}$.

### 2.2 Ordinary Least Squares Derivation

**The OLS objective:** minimise the sum of squared residuals over $\boldsymbol{\beta} \in \mathbb{R}^d$:

$$\mathcal{L}(\boldsymbol{\beta}) = \lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 = (\mathbf{y} - X\boldsymbol{\beta})^\top (\mathbf{y} - X\boldsymbol{\beta})$$

**Algebraic derivation (calculus).** Expand and differentiate:

$$\mathcal{L}(\boldsymbol{\beta}) = \mathbf{y}^\top \mathbf{y} - 2\boldsymbol{\beta}^\top X^\top \mathbf{y} + \boldsymbol{\beta}^\top X^\top X \boldsymbol{\beta}$$

Take the gradient with respect to $\boldsymbol{\beta}$:

$$\nabla_{\boldsymbol{\beta}} \mathcal{L} = -2 X^\top \mathbf{y} + 2 X^\top X \boldsymbol{\beta}$$

Set to zero — the **normal equations**:

$$X^\top X \hat{\boldsymbol{\beta}} = X^\top \mathbf{y}$$

When $X^\top X$ is invertible (i.e., $\operatorname{rank}(X) = d$), the unique solution is:

$$\boxed{\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}}$$

The Hessian $\nabla^2_{\boldsymbol{\beta}} \mathcal{L} = 2X^\top X \succeq 0$, so $\mathcal{L}$ is convex — every local minimum is global. When $X$ has full column rank, $X^\top X \succ 0$, so the minimum is unique.

**Geometric derivation (projection).** The minimum of $\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2$ over $\boldsymbol{\beta}$ is achieved when $X\hat{\boldsymbol{\beta}}$ is the orthogonal projection of $\mathbf{y}$ onto $\mathcal{C}(X)$. This requires the residual to be orthogonal to every vector in $\mathcal{C}(X)$:

$$X^\top (\mathbf{y} - X\hat{\boldsymbol{\beta}}) = \mathbf{0} \quad \Longleftrightarrow \quad X^\top X \hat{\boldsymbol{\beta}} = X^\top \mathbf{y}$$

Exactly the normal equations. Both derivations are equivalent — the geometric one reveals _why_ the formula works.

**MLE interpretation.** If $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$, then the log-likelihood of the data is:

$$\log p(\mathbf{y} \mid X, \boldsymbol{\beta}, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2$$

Maximising over $\boldsymbol{\beta}$ is equivalent to minimising $\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2$, so $\hat{\boldsymbol{\beta}}_{\text{OLS}} = \hat{\boldsymbol{\beta}}_{\text{MLE}}$ under Gaussian noise.

### 2.3 The Hat Matrix and Projection Geometry

The **hat matrix** (projection matrix) $H \in \mathbb{R}^{n \times n}$ maps observations to fitted values:

$$H = X(X^\top X)^{-1} X^\top, \quad \hat{\mathbf{y}} = H\mathbf{y}$$

**Key properties** (all follow from $H$ being an orthogonal projector onto $\mathcal{C}(X)$):

| Property | Equation | Interpretation |
|---|---|---|
| Idempotent | $H^2 = H$ | Projecting twice = projecting once |
| Symmetric | $H^\top = H$ | Orthogonal projection |
| Rank | $\operatorname{rank}(H) = d$ | Projects onto a $d$-dim subspace |
| Eigenvalues | $\lambda_i \in \{0, 1\}$ | In or out of $\mathcal{C}(X)$ |
| Residual projector | $(I - H)^2 = I - H$ | Residuals form complementary subspace |
| Trace | $\operatorname{tr}(H) = d$ | Degrees of freedom used by model |

**Leverage scores.** The diagonal entries $h_{ii} = H_{ii} = \mathbf{x}^{(i)\top}(X^\top X)^{-1}\mathbf{x}^{(i)}$ measure how much influence observation $i$ has on its own fitted value. Properties:

- $0 \leq h_{ii} \leq 1$ and $\sum_i h_{ii} = d$
- $h_{ii}$ large ($\approx 1$): observation $i$ is a **high-leverage point** — it nearly determines the fit at $\mathbf{x}^{(i)}$
- Predicted value: $\hat{y}^{(i)} = h_{ii} y^{(i)} + \sum_{j \neq i} H_{ij} y^{(j)}$

**For AI:** Leverage scores appear in **influence functions** (Koh & Liang, 2017), which measure how removing one training example changes the model parameters. This is the mathematical foundation of training data attribution for LLMs — identifying which training examples caused a specific model output.

**LOOCV shortcut via leverage.** For linear models, the leave-one-out cross-validation error has a closed form:

$$\text{LOOCV} = \frac{1}{n} \sum_{i=1}^n \left(\frac{y^{(i)} - \hat{y}^{(i)}}{1 - h_{ii}}\right)^2$$

This computes leave-one-out error at the cost of a single OLS fit — an $O(nd^2)$ computation instead of $O(n^2 d^2)$.

### 2.4 SVD-Based Solution

The **singular value decomposition** of $X \in \mathbb{R}^{n \times d}$ (with $n \geq d$):

$$X = U \Sigma V^\top$$

where $U \in \mathbb{R}^{n \times d}$ has orthonormal columns (left singular vectors), $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_d)$ with $\sigma_1 \geq \cdots \geq \sigma_d \geq 0$, and $V \in \mathbb{R}^{d \times d}$ is orthogonal (right singular vectors).

Substituting into the OLS formula:

$$\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y} = (V\Sigma^2 V^\top)^{-1} V\Sigma U^\top \mathbf{y} = V \Sigma^{-1} U^\top \mathbf{y}$$

The predicted values:

$$\hat{\mathbf{y}} = X\hat{\boldsymbol{\beta}} = U\Sigma V^\top \cdot V\Sigma^{-1} U^\top \mathbf{y} = UU^\top \mathbf{y}$$

So $H = UU^\top$ — the hat matrix projects onto the column space of $U$ (the left singular vectors of $X$).

**Component-wise view.** The OLS estimate decomposes as:

$$\hat{\boldsymbol{\beta}} = \sum_{j=1}^{d} \frac{\mathbf{u}_j^\top \mathbf{y}}{\sigma_j} \mathbf{v}_j$$

Each term is the component of $\mathbf{y}$ along left singular vector $\mathbf{u}_j$, divided by the corresponding singular value $\sigma_j$, and expressed in right singular vector direction $\mathbf{v}_j$.

**Numerical implication:** When $\sigma_d \approx 0$ (near-multicollinearity), $\hat{\boldsymbol{\beta}}$ becomes unstable — tiny changes in $\mathbf{y}$ produce huge changes in $\hat{\boldsymbol{\beta}}$. The condition number $\kappa(X) = \sigma_1/\sigma_d$ quantifies this sensitivity. This is why ridge regression (adding $\lambda$ to all singular values) is numerically superior to direct inversion.

**Rank-deficient case.** If $\operatorname{rank}(X) = r < d$, $X^\top X$ is singular and OLS has infinitely many solutions. The **minimum-norm solution** (Moore-Penrose pseudoinverse):

$$\hat{\boldsymbol{\beta}}^\dagger = X^\dagger \mathbf{y} = V_r \Sigma_r^{-1} U_r^\top \mathbf{y}$$

where subscript $r$ denotes the first $r$ singular vectors/values. This solution minimises $\lVert \boldsymbol{\beta} \rVert_2$ among all minimisers of $\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2$. **For AI:** overparameterised neural networks in the interpolation regime converge to minimum-norm solutions — explaining benign overfitting (Bartlett et al., 2020).

### 2.5 BLUE and the Gauss-Markov Theorem

**Theorem (Gauss-Markov).** Under assumptions GM1–GM5 (Section 2.1), the OLS estimator $\hat{\boldsymbol{\beta}}$ is the **Best Linear Unbiased Estimator (BLUE)**: among all estimators that are (1) linear functions of $\mathbf{y}$ and (2) unbiased for $\boldsymbol{\beta}$, OLS has the smallest variance.

**Proof sketch.**

*Step 1: Unbiasedness.* $\mathbb{E}[\hat{\boldsymbol{\beta}}] = (X^\top X)^{-1} X^\top \mathbb{E}[\mathbf{y}] = (X^\top X)^{-1} X^\top X \boldsymbol{\beta} = \boldsymbol{\beta}$.

*Step 2: Variance.* $\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} X^\top \operatorname{Cov}(\mathbf{y}) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$.

*Step 3: Optimality.* Let $\tilde{\boldsymbol{\beta}} = C\mathbf{y}$ be any other linear unbiased estimator, so $CX = I$ (unbiasedness condition). Write $C = (X^\top X)^{-1} X^\top + D$ for some matrix $D$ with $DX = 0$. Then:

$$\operatorname{Cov}(\tilde{\boldsymbol{\beta}}) = \sigma^2 CC^\top = \sigma^2 (X^\top X)^{-1} + \sigma^2 DD^\top \succeq \sigma^2 (X^\top X)^{-1} = \operatorname{Cov}(\hat{\boldsymbol{\beta}})$$

since $DD^\top \succeq 0$. Equality holds iff $D = 0$, i.e., $\tilde{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}$. $\square$

**Limitations:** "Best" means minimum variance in the PSD order — not best in any norm. BLUE is only optimal within the linear unbiased class; biased estimators (like Ridge) can have lower mean squared error by trading bias for variance reduction.


---

## 3. Statistical Inference for OLS

### 3.1 Distributional Properties of OLS

Assuming $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_n)$, the OLS estimator is normally distributed:

$$\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y} \sim \mathcal{N}\!\left(\boldsymbol{\beta},\; \sigma^2 (X^\top X)^{-1}\right)$$

**Derivation:** $\hat{\boldsymbol{\beta}}$ is a linear transformation of $\mathbf{y} \sim \mathcal{N}(X\boldsymbol{\beta}, \sigma^2 I)$.

- Mean: $\mathbb{E}[\hat{\boldsymbol{\beta}}] = (X^\top X)^{-1} X^\top X\boldsymbol{\beta} = \boldsymbol{\beta}$ ✓
- Covariance: $\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} X^\top (\sigma^2 I) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}$ ✓

The marginal distribution of each coefficient $\hat{\beta}_j$:

$$\hat{\beta}_j \sim \mathcal{N}\!\left(\beta_j,\; \sigma^2 [(X^\top X)^{-1}]_{jj}\right)$$

**Variance intuition:** $[(X^\top X)^{-1}]_{jj}$ is large when feature $j$ is nearly collinear with other features (the corresponding singular value of $X$ is small). More data reduces variance as $n \to \infty$ since $[(X^\top X)^{-1}]_{jj} = O(1/n)$ for i.i.d. data.

**Residual distribution:**

$$\mathbf{e} = (I - H)\mathbf{y} \sim \mathcal{N}(\mathbf{0},\; \sigma^2 (I - H))$$

Note $(I-H)$ is singular (rank $n - d$), so the residual vector lies in a $(n-d)$-dimensional affine subspace. The residuals and coefficients are **independent** because $H$ and $(I-H)$ project onto orthogonal subspaces.

### 3.2 Estimating the Noise Variance

The natural estimator of $\sigma^2$ uses the residual sum of squares (RSS):

$$\hat{\sigma}^2 = \frac{\text{RSS}}{n - d} = \frac{\lVert \mathbf{e} \rVert_2^2}{n - d} = \frac{\mathbf{y}^\top (I - H) \mathbf{y}}{n - d}$$

**Unbiasedness proof:**

$$\mathbb{E}[\text{RSS}] = \mathbb{E}[\mathbf{y}^\top (I-H) \mathbf{y}] = \operatorname{tr}\!\left((I-H)\operatorname{Cov}(\mathbf{y})\right) + \mathbf{y}^{*\top}(I-H)\mathbf{y}^*$$

where $\mathbf{y}^* = X\boldsymbol{\beta}$ and $(I-H)X = 0$ (residuals are orthogonal to column space), so the second term vanishes. The first term: $\operatorname{tr}((I-H)\sigma^2 I) = \sigma^2 \operatorname{tr}(I-H) = \sigma^2(n - d)$. Therefore $\mathbb{E}[\hat{\sigma}^2] = \sigma^2$.

The denominator $n - d$ counts **degrees of freedom**: $n$ total observations minus $d$ parameters estimated. Dividing by $n$ instead (as in MLE) gives a biased estimator. Furthermore:

$$\frac{(n-d)\hat{\sigma}^2}{\sigma^2} \sim \chi^2_{n-d}$$

which is independent of $\hat{\boldsymbol{\beta}}$ (by the independence of $H$ and $I-H$ projections).

### 3.3 Hypothesis Testing

**Individual coefficient $t$-test.** To test $H_0: \beta_j = 0$ vs. $H_1: \beta_j \neq 0$:

$$t_j = \frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{[(X^\top X)^{-1}]_{jj}}} \sim t_{n-d} \quad \text{under } H_0$$

The denominator $\hat{\sigma}\sqrt{[(X^\top X)^{-1}]_{jj}}$ is the **standard error** of $\hat{\beta}_j$, denoted $\text{SE}(\hat{\beta}_j)$.

**Interpretation:** If $|t_j| > t_{\alpha/2, n-d}$ (the critical value for significance level $\alpha$), we reject $H_0$. The $p$-value is $P(|T| > |t_j|)$ for $T \sim t_{n-d}$.

**Joint $F$-test.** To test $H_0: R\boldsymbol{\beta} = \mathbf{c}$ (linear restrictions on $q$ parameters) vs. the unrestricted model:

$$F = \frac{(\text{RSS}_R - \text{RSS}_U)/q}{\text{RSS}_U/(n-d)} \sim F_{q, n-d} \quad \text{under } H_0$$

where RSS$_R$ is the residual sum of squares under the restricted model and RSS$_U$ under the unrestricted model. Special case: testing all coefficients jointly ($R = I_d, \mathbf{c} = \mathbf{0}$) gives the overall model $F$-statistic.

**Multiple testing warning.** With $d$ coefficients, testing each at $\alpha = 0.05$ gives expected $0.05d$ false rejections under $H_0$. Use Bonferroni correction ($\alpha/d$ per test) or control the false discovery rate (Benjamini-Hochberg) when testing many coefficients simultaneously.

**Heteroscedasticity-robust standard errors.** When the homoscedasticity assumption fails (i.e., $\operatorname{Var}(\epsilon^{(i)}) = \sigma_i^2$ varies across observations), OLS is still unbiased but the standard formula for $\operatorname{Cov}(\hat{\boldsymbol{\beta}})$ is wrong. The **HC3 sandwich estimator** (MacKinnon & White, 1985):

$$\widehat{\operatorname{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} \left(\sum_{i=1}^n \frac{e_i^2}{(1-h_{ii})^2} \mathbf{x}^{(i)} \mathbf{x}^{(i)\top}\right) (X^\top X)^{-1}$$

is consistent even under heteroscedasticity. This is the standard approach in econometrics and is increasingly used in ML when errors are not i.i.d.

### 3.4 Confidence and Prediction Intervals

**Confidence interval for $\beta_j$:**

$$\hat{\beta}_j \pm t_{\alpha/2, n-d} \cdot \text{SE}(\hat{\beta}_j)$$

**Confidence interval for $\mathbb{E}[y \mid \mathbf{x}_*] = \boldsymbol{\beta}^\top \mathbf{x}_*$:**

$$\hat{y}_* \pm t_{\alpha/2, n-d} \cdot \hat{\sigma} \sqrt{\mathbf{x}_*^\top (X^\top X)^{-1} \mathbf{x}_*}$$

The interval is narrowest at $\mathbf{x}_* = \bar{\mathbf{x}}$ (the centroid) and widens as $\mathbf{x}_*$ moves away from the training data.

**Prediction interval for a new observation $y_*$:** must also account for the new noise $\epsilon_*$:

$$\hat{y}_* \pm t_{\alpha/2, n-d} \cdot \hat{\sigma} \sqrt{1 + \mathbf{x}_*^\top (X^\top X)^{-1} \mathbf{x}_*}$$

The extra $+1$ inside the square root reflects aleatoric (irreducible) uncertainty. This parallels the Bayesian predictive variance $\mathbf{x}_*^\top \Sigma_n \mathbf{x}_* + \sigma^2$ (Section 7.3).

### 3.5 $R^2$ and Goodness of Fit

The **coefficient of determination** $R^2$ measures the fraction of variance explained by the model:

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_i (y^{(i)} - \hat{y}^{(i)})^2}{\sum_i (y^{(i)} - \bar{y})^2}$$

where TSS (total sum of squares) = variance of $\mathbf{y}$ up to a constant.

**Properties:**
- $R^2 \in [0, 1]$ for OLS with an intercept (residuals are always smaller than deviations from the mean)
- $R^2 = r^2$ in simple regression ($d=1$), where $r$ is the Pearson correlation
- Adding any feature (even random noise) increases $R^2$ — the model uses additional degrees of freedom

**Adjusted $R^2$** penalises for model size:

$$\bar{R}^2 = 1 - \frac{\text{RSS}/(n-d)}{\text{TSS}/(n-1)} = 1 - (1 - R^2)\frac{n-1}{n-d}$$

$\bar{R}^2$ can decrease when adding useless features. **For AI:** $R^2$ is widely used in linear probing experiments to measure how well a representation encodes a target variable (e.g., syntactic depth, semantic category).

---

## 4. Ridge Regression (L2 Regularisation)

### 4.1 Objective and Closed-Form Solution

The Ridge objective adds an $\ell^2$ penalty on the weights:

$$\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \lambda \lVert \boldsymbol{\beta} \rVert_2^2, \quad \lambda \geq 0$$

**Solution.** Setting the gradient to zero:

$$-2X^\top(\mathbf{y} - X\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta} = \mathbf{0} \implies (X^\top X + \lambda I)\hat{\boldsymbol{\beta}}_\lambda = X^\top \mathbf{y}$$

$$\boxed{\hat{\boldsymbol{\beta}}_\lambda = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}}$$

**Key properties:**
- $(X^\top X + \lambda I)$ is **always invertible** for $\lambda > 0$, even when $X$ is rank-deficient
- The solution exists and is unique for all $\lambda > 0$
- As $\lambda \to 0$: $\hat{\boldsymbol{\beta}}_\lambda \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$
- As $\lambda \to \infty$: $\hat{\boldsymbol{\beta}}_\lambda \to \mathbf{0}$ (all coefficients shrink to zero)

**Bias-variance tradeoff.**

$$\text{Bias}(\hat{\boldsymbol{\beta}}_\lambda) = -\lambda(X^\top X + \lambda I)^{-1}\boldsymbol{\beta}$$

$$\operatorname{Var}(\hat{\boldsymbol{\beta}}_\lambda) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1}$$

For any $\lambda > 0$, Ridge is **biased** (bias $\neq 0$) but has **lower variance** than OLS. There always exists $\lambda > 0$ such that the MSE of Ridge is lower than OLS (Hoerl & Kennard, 1970).

### 4.2 SVD View and Singular Value Shrinkage

Substituting $X = U\Sigma V^\top$:

$$\hat{\boldsymbol{\beta}}_\lambda = V(\Sigma^2 + \lambda I)^{-1} \Sigma U^\top \mathbf{y} = \sum_{j=1}^d \frac{\sigma_j}{\sigma_j^2 + \lambda} (\mathbf{u}_j^\top \mathbf{y}) \mathbf{v}_j$$

Compare with OLS: $\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^d \frac{1}{\sigma_j} (\mathbf{u}_j^\top \mathbf{y}) \mathbf{v}_j$.

Ridge applies a **shrinkage factor** $\sigma_j^2/(\sigma_j^2 + \lambda) < 1$ to each component. Components with small $\sigma_j$ (near-collinear directions) are shrunk most aggressively.

```
RIDGE SINGULAR VALUE SHRINKAGE
════════════════════════════════════════════════════════════════════════

  OLS coefficient along v_j:  (u_j^T y) / σ_j

  Ridge coefficient along v_j:  (u_j^T y) × σ_j / (σ_j² + λ)

  Shrinkage factor:  σ_j² / (σ_j² + λ)

        σ_j >> √λ  →  factor ≈ 1  (OLS, large singular values unaffected)
        σ_j  = √λ  →  factor = ½  (half-shrinkage at the "knee")
        σ_j << √λ  →  factor ≈ 0  (near-zero, small singular values killed)

  Effective degrees of freedom:  df(λ) = Σⱼ σⱼ²/(σⱼ² + λ)
    λ=0: df = d (full OLS)
    λ→∞: df → 0 (null model)

════════════════════════════════════════════════════════════════════════
```

**Effective degrees of freedom** $\operatorname{df}(\lambda) = \sum_j \sigma_j^2/(\sigma_j^2 + \lambda) = \operatorname{tr}(H_\lambda)$ where $H_\lambda = X(X^\top X + \lambda I)^{-1}X^\top$ is the Ridge hat matrix. This generalises $\operatorname{df} = d$ from OLS and allows $\lambda$ to be compared across datasets.

### 4.3 Bayesian Interpretation

Ridge regression is the MAP estimator under a Gaussian prior:

$$\boldsymbol{\beta} \sim \mathcal{N}\!\left(\mathbf{0}, \tau^2 I\right), \quad \mathbf{y} \mid X, \boldsymbol{\beta} \sim \mathcal{N}(X\boldsymbol{\beta}, \sigma^2 I)$$

The posterior mode (MAP estimate) is:

$$\hat{\boldsymbol{\beta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\beta}} \left[\log p(\mathbf{y} \mid X, \boldsymbol{\beta}) + \log p(\boldsymbol{\beta})\right]$$

$$= \arg\min_{\boldsymbol{\beta}} \left[\frac{1}{2\sigma^2}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \frac{1}{2\tau^2}\lVert \boldsymbol{\beta} \rVert_2^2\right]$$

$$= \arg\min_{\boldsymbol{\beta}} \left[\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \underbrace{\frac{\sigma^2}{\tau^2}}_{\lambda} \lVert \boldsymbol{\beta} \rVert_2^2\right]$$

Therefore $\lambda = \sigma^2/\tau^2$: the regularisation strength is the ratio of noise variance to prior variance. Strong prior ($\tau^2$ small) → large $\lambda$ → aggressive shrinkage.

**For AI:** Weight decay in neural network training is exactly Ridge regularisation. In PyTorch, `optimizer = Adam(params, weight_decay=λ)` adds $\lambda \lVert \boldsymbol{\theta} \rVert_2^2$ to the loss. The optimal $\lambda$ depends on the signal-to-noise ratio of the training task — a connection formalised by the Bayesian interpretation.

### 4.4 Ridge Path and Effective Degrees of Freedom

The **ridge path** traces how each coefficient $\hat{\beta}_{j,\lambda}$ changes as $\lambda$ varies from $0$ to $\infty$:

- All paths start at $\hat{\boldsymbol{\beta}}_{\text{OLS}}$ (at $\lambda = 0$)
- All paths end at $\mathbf{0}$ (as $\lambda \to \infty$)
- Paths are smooth monotonic functions of $\lambda$ (unlike Lasso, which has kinks)
- No coefficient ever crosses zero to change sign (unlike Lasso)

**Choosing $\lambda$:** Use $k$-fold cross-validation over a logarithmic grid of $\lambda$ values. The **one-standard-error rule** (Hastie et al., 2009): choose the largest $\lambda$ whose CV error is within one standard error of the minimum — produces a sparser/simpler model.

### 4.5 Tikhonov Regularisation and Weight Decay

The **generalised Tikhonov regulariser** replaces $\lambda I$ with a general positive semidefinite matrix $\lambda \Gamma^\top \Gamma$:

$$\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \lambda \lVert \Gamma \boldsymbol{\beta} \rVert_2^2$$

Solution: $\hat{\boldsymbol{\beta}} = (X^\top X + \lambda \Gamma^\top \Gamma)^{-1} X^\top \mathbf{y}$.

Special cases: $\Gamma = I$ → Ridge; $\Gamma = $ finite-difference matrix → smoothness penalty (spline regression); $\Gamma = $ graph Laplacian → graph-regularised regression.

**For AI:** Weight decay $\lambda \lVert \boldsymbol{\theta} \rVert_2^2$ is the universal default regulariser in LLM pretraining. Llama 2 uses weight decay 0.1; GPT-4 architecture details suggest similar values. The Bayesian interpretation says weight decay encodes a Gaussian prior $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, (1/\lambda)I)$ — a mild belief that weights should be small.


---

## 5. Lasso Regression (L1 Regularisation)

### 5.1 Objective and Geometric Sparsity

The Lasso (Tibshirani, 1994) replaces the squared $\ell^2$ penalty with an $\ell^1$ penalty:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \lambda \lVert \boldsymbol{\beta} \rVert_1$$

The key property of Lasso is **sparsity**: many coefficients in $\hat{\boldsymbol{\beta}}_\lambda$ are exactly zero.

**Geometric explanation.** The constrained form of Lasso is:

$$\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 \quad \text{subject to} \quad \lVert \boldsymbol{\beta} \rVert_1 \leq t$$

The $\ell^1$ ball $\{\boldsymbol{\beta} : \lVert \boldsymbol{\beta} \rVert_1 \leq t\}$ is a **polytope** (diamond in 2D) with corners at $\pm t \mathbf{e}_j$. The ellipsoidal contours of the least-squares objective typically contact the $\ell^1$ ball at a corner — where exactly one coordinate is nonzero. The $\ell^2$ ball is smooth and round; contours contact it on its surface, where all coordinates are generally nonzero.

```
GEOMETRY OF REGULARISATION CONSTRAINTS
════════════════════════════════════════════════════════════════════════

    β₂                       β₂
     │           OLS           │           OLS
     │            ◉            │            ◉
     │         ╱   ╲           │         ╱   ╲
     │      ╱    ╱◯╲ ╲         │      ╱    ╱   ╲ ╲
     │   ╱     ╱     ╲  ╲      │   ╱     ╱  ◯   ╲  ╲
  ───┼─╱──────╱── β₁ ─╲──╲─   ───┼─╱──────╱── β₁ ─╲──╲─
     │ ╲      ╲       ╱  ╱     │ ╲      ╲       ╱  ╱
     │  ╲     ╲     ╱  ╱      │  ╲      ╲     ╱  ╱
     │   ╲     ╲   ╱  ╱       │   ╲       ╲   ╱  ╱
     │    ╲     ╲ ╱  ╱        │    ╲        ╲ ╱  ╱
     │                         │
    L1 ball (diamond) →       L2 ball (circle) →
    contact at corner!         contact NOT at corner!
    → one β_j = 0              → both β_j ≠ 0

════════════════════════════════════════════════════════════════════════
```

### 5.2 KKT Conditions and Soft-Thresholding

Since the Lasso objective is convex but non-smooth at $\beta_j = 0$, we use **subgradient** optimality conditions. Let $\mathbf{r} = \mathbf{y} - X\hat{\boldsymbol{\beta}}$ denote the residual. The KKT stationarity condition for $\hat{\beta}_j$ is:

$$\frac{\partial}{\partial \beta_j}\left[\frac{1}{2n}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2\right] + \lambda \partial|\beta_j| = 0$$

$$-\frac{1}{n}(\mathbf{x}_j)^\top \mathbf{r} + \lambda s_j = 0$$

where $s_j$ is a subgradient of $|\beta_j|$: $s_j = \operatorname{sign}(\beta_j)$ if $\beta_j \neq 0$, and $s_j \in [-1, 1]$ if $\beta_j = 0$.

**Coordinate-wise Lasso solution.** Suppose all coordinates except $\beta_j$ are fixed. Define the partial residual $\mathbf{r}^{(-j)} = \mathbf{y} - X_{-j}\boldsymbol{\beta}_{-j}$ and the OLS estimate for coordinate $j$ ignoring others:

$$z_j = \frac{1}{n}(\mathbf{x}_j)^\top \mathbf{r}^{(-j)} = \hat{\beta}_j^{\text{OLS, partial}}$$

The Lasso solution for coordinate $j$ is the **soft-thresholding operator**:

$$\hat{\beta}_j = S_\lambda(z_j) = \operatorname{sign}(z_j)\max(|z_j| - \lambda, 0)$$

**Derivation:** Consider $\min_\beta \frac{1}{2}(\beta - z)^2 + \lambda|\beta|$.
- If $z > \lambda$: minimum at $\hat{\beta} = z - \lambda > 0$
- If $z < -\lambda$: minimum at $\hat{\beta} = z + \lambda < 0$
- If $|z| \leq \lambda$: minimum at $\hat{\beta} = 0$ (the penalty dominates)

This gives $S_\lambda(z) = \operatorname{sign}(z)\max(|z|-\lambda, 0)$, which shrinks toward zero and thresholds small values to exactly zero.

### 5.3 Coordinate Descent Algorithm

**Algorithm (Cyclic Coordinate Descent for Lasso):**

```
Input: X, y, λ, initialise β = 0
Repeat until convergence:
  For j = 1 to d:
    Compute partial residual: r^(-j) = y - X_{-j} β_{-j}
    Compute OLS for coord j:  z_j = (1/n) x_j^T r^(-j)
    Update: β_j ← S_λ(z_j)
Return β
```

**Efficient implementation.** Maintaining the full residual $\mathbf{r} = \mathbf{y} - X\boldsymbol{\beta}$ and updating it each step:

$$\mathbf{r} \leftarrow \mathbf{r} + \mathbf{x}_j \beta_j^{\text{old}}, \quad z_j = \frac{1}{n}\mathbf{x}_j^\top \mathbf{r}, \quad \beta_j^{\text{new}} = S_\lambda(z_j), \quad \mathbf{r} \leftarrow \mathbf{r} - \mathbf{x}_j \beta_j^{\text{new}}$$

Each coordinate update costs $O(n)$; one pass through all $d$ coordinates costs $O(nd)$.

**Convergence.** Cyclic coordinate descent converges to the global minimum of the Lasso objective (which is convex) at a linear rate $O(1/t)$ for $t$ iterations. The **active set** (nonzero coordinates) typically stabilises quickly, after which only the active coordinates need updating.

**For AI:** Coordinate descent for Lasso is the algorithmic template for **sparse attention patterns** in efficient Transformers. Reformers and Longformers use variants of this idea to identify which key-query pairs to attend to.

### 5.4 LARS Algorithm

The **Least Angle Regression (LARS)** algorithm (Efron et al., 2004) traces the entire Lasso regularisation path efficiently.

**Key insight:** as $\lambda$ decreases from $\infty$ to $0$, the Lasso coefficient path is **piecewise linear**. LARS exploits this to compute the entire path in $O(nd^2)$ time — the same cost as a single OLS fit.

**LARS procedure:**
1. Start with $\boldsymbol{\beta} = \mathbf{0}$ (all coefficients zero, $\lambda = \infty$)
2. Find the feature $j^*$ most correlated with the residual $\mathbf{r} = \mathbf{y} - X\boldsymbol{\beta}$
3. Move $\beta_{j^*}$ in the direction of $\operatorname{sign}(\mathbf{x}_{j^*}^\top \mathbf{r})$ until another feature has equal correlation
4. Move $\beta_{j^*}$ and $\beta_{j^{**}}$ jointly in the direction that keeps their correlations equal, until a third feature "ties"
5. Continue until all features are active or residuals are zero

LARS produces the entire piecewise linear path of the Lasso solution with at most $d$ steps.

**For AI:** The Lasso path reveals the **feature entry order** — which features are most predictive. This is directly useful in model interpretability: for a logistic probe on LLM activations, the Lasso path shows which activation dimensions are most informative about a linguistic property.

### 5.5 Lasso as Laplace Prior MAP

The Lasso MAP interpretation uses a **Laplace (double-exponential) prior**:

$$p(\beta_j) = \frac{\lambda}{2}\exp(-\lambda|\beta_j|)$$

The log-posterior:

$$\log p(\boldsymbol{\beta} \mid \mathbf{y}, X) \propto -\frac{1}{2\sigma^2}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 - \lambda \sum_j |\beta_j|$$

Maximising is equivalent to the Lasso objective. The Laplace prior has a **sharp spike at zero** (unlike the Gaussian), which encodes the belief that most coefficients are exactly zero — a formal statement of the sparsity assumption.

**Compressed sensing connection.** When $n < d$ (more features than observations), OLS is underdetermined. If the true $\boldsymbol{\beta}^*$ has at most $s \ll d$ nonzero entries, and $X$ satisfies the **Restricted Isometry Property (RIP)**:

$$(1-\delta)\lVert \boldsymbol{\beta} \rVert_2^2 \leq \lVert X\boldsymbol{\beta} \rVert_2^2 \leq (1+\delta)\lVert \boldsymbol{\beta} \rVert_2^2 \quad \forall s\text{-sparse } \boldsymbol{\beta}$$

then Lasso with $\lambda \propto \sigma\sqrt{\log d / n}$ recovers $\boldsymbol{\beta}^*$ exactly (with high probability) from only $n = O(s \log d)$ measurements (Candès & Tao, 2005). This is the foundation of **sparse coding** in dictionary learning and signal processing.

---

## 6. Elastic Net and Structured Sparsity

### 6.1 Elastic Net Objective

The **Elastic Net** (Zou & Hastie, 2003) combines Ridge and Lasso penalties:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \lambda_1 \lVert \boldsymbol{\beta} \rVert_1 + \lambda_2 \lVert \boldsymbol{\beta} \rVert_2^2$$

**Coordinate descent update.** For coordinate $j$, the Elastic Net update is:

$$\hat{\beta}_j = \frac{S_{\lambda_1}(z_j)}{1 + 2\lambda_2}$$

where $z_j = \frac{1}{n}\mathbf{x}_j^\top \mathbf{r}^{(-j)}$ as before. The Ridge term adds a division by $(1 + 2\lambda_2)$ beyond the Lasso soft-thresholding.

**Grouping effect.** When features are highly correlated, Lasso tends to select one and ignore others (arbitrary choice). Elastic Net includes correlated features together — the $\ell^2$ penalty encourages correlated predictors to have similar (non-zero) coefficients. This is the **grouping effect**.

**Comparison table:**

| Property | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Exact zeros | No | Yes | Yes |
| Correlated features | Includes all (shrinks equally) | Picks one arbitrarily | Groups them |
| Closed form | Yes | No (subgradient) | No (subgradient) |
| Always unique | Yes | Yes (when $X$ full rank) | Yes |
| High $d > n$ | Cannot select | Selects at most $n$ | Selects > $n$ (with grouping) |
| Bayesian prior | Gaussian | Laplace | Gaussian + Laplace |

### 6.2 Group Lasso

When features come in known groups $\mathcal{G}_1, \ldots, \mathcal{G}_G$ (e.g., dummy variables for a categorical feature), **Group Lasso** penalises the $\ell^2$ norm of each group:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + \lambda \sum_{g=1}^G \lVert \boldsymbol{\beta}_{\mathcal{G}_g} \rVert_2$$

The update for group $g$ is **block soft-thresholding**:

$$\hat{\boldsymbol{\beta}}_{\mathcal{G}_g} = \left(1 - \frac{\lambda}{\lVert z_g \rVert_2}\right)_+ z_g$$

where $z_g$ is the partial OLS estimate for group $g$. This either zeros out the entire group or scales it — enforcing group-level sparsity.

**For AI:** Group Lasso is the mathematical foundation of **structured pruning** in LLMs — pruning entire attention heads or MLP layers rather than individual weights. The $\ell^{2,1}$ norm (sum of $\ell^2$ norms of row groups) achieves row-wise sparsity in weight matrices.

### 6.3 Nuclear Norm Regularisation

For matrix-valued parameters $W \in \mathbb{R}^{m \times n}$, the **nuclear norm** is:

$$\lVert W \rVert_* = \sum_{i=1}^{\min(m,n)} \sigma_i(W)$$

(sum of singular values). This is the matrix analogue of the $\ell^1$ norm applied to singular values.

**Nuclear norm regularisation** encourages low-rank solutions:

$$\min_W \frac{1}{2}\lVert Y - XW \rVert_F^2 + \lambda \lVert W \rVert_*$$

The proximal operator (analogous to soft-thresholding) is **singular value soft-thresholding**: $\hat{W} = U S_\lambda(\Sigma) V^\top$ where $X = U\Sigma V^\top$ and $S_\lambda$ applies element-wise to singular values.

**For AI:** Nuclear norm regularisation is the convex relaxation of rank minimisation. LoRA constrains $\Delta W = BA$ with $\operatorname{rank} \leq r$; GaLore (Zhao et al., 2024) projects gradients onto low-rank subspaces. Both methods are heuristic approximations to the convex nuclear norm problem.

---

## 7. Bayesian Linear Regression

### 7.1 The Prior-Posterior Framework

Bayesian linear regression places a prior distribution over the unknown parameters $\boldsymbol{\beta}$ and updates it to a posterior after observing data $(\mathbf{y}, X)$.

**Conjugate Gaussian prior:**

$$\boldsymbol{\beta} \sim \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$$

**Likelihood** (assuming known $\sigma^2$):

$$p(\mathbf{y} \mid X, \boldsymbol{\beta}) = \mathcal{N}(X\boldsymbol{\beta}, \sigma^2 I_n)$$

**Posterior by Bayes' theorem:**

$$p(\boldsymbol{\beta} \mid \mathbf{y}, X) \propto p(\mathbf{y} \mid X, \boldsymbol{\beta}) \cdot p(\boldsymbol{\beta})$$

Since both factors are Gaussian (exponentials of quadratic forms in $\boldsymbol{\beta}$), the posterior is also Gaussian — the Gaussian family is **closed under conditioning on linear Gaussian observations** (self-conjugacy).

### 7.2 Posterior Mean and Variance

**Completing the square** in the exponent of $p(\boldsymbol{\beta} \mid \mathbf{y}, X)$:

$$-\frac{1}{2}\left[\frac{1}{\sigma^2}\lVert \mathbf{y} - X\boldsymbol{\beta} \rVert_2^2 + (\boldsymbol{\beta} - \boldsymbol{\mu}_0)^\top \Sigma_0^{-1}(\boldsymbol{\beta} - \boldsymbol{\mu}_0)\right]$$

The posterior precision (inverse covariance) is:

$$\Sigma_n^{-1} = \Sigma_0^{-1} + \frac{1}{\sigma^2} X^\top X$$

The posterior mean:

$$\boldsymbol{\mu}_n = \Sigma_n\!\left(\Sigma_0^{-1}\boldsymbol{\mu}_0 + \frac{1}{\sigma^2} X^\top \mathbf{y}\right)$$

Therefore:

$$\boldsymbol{\beta} \mid \mathbf{y}, X \sim \mathcal{N}(\boldsymbol{\mu}_n, \Sigma_n)$$

**Intuition.** The posterior mean $\boldsymbol{\mu}_n$ is a **weighted combination of the prior mean and the MLE**:
- Prior mean $\boldsymbol{\mu}_0$, weighted by prior precision $\Sigma_0^{-1}$
- MLE $\hat{\boldsymbol{\beta}}_{\text{OLS}}$, weighted by likelihood precision $\frac{1}{\sigma^2}X^\top X$

As $n \to \infty$: the likelihood overwhelms the prior, $\boldsymbol{\mu}_n \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$, $\Sigma_n \to \mathbf{0}$ — the posterior concentrates at the MLE.

**Special case (zero-mean isotropic prior):** $\boldsymbol{\mu}_0 = \mathbf{0}$, $\Sigma_0 = \tau^2 I$:

$$\boldsymbol{\mu}_n = \left(\frac{1}{\tau^2}I + \frac{1}{\sigma^2}X^\top X\right)^{-1} \frac{1}{\sigma^2}X^\top \mathbf{y} = \left(X^\top X + \frac{\sigma^2}{\tau^2}I\right)^{-1} X^\top \mathbf{y} = \hat{\boldsymbol{\beta}}_{\lambda=\sigma^2/\tau^2}$$

The posterior mean equals the Ridge estimate. Ridge regression is **Bayesian MAP estimation** with a Gaussian prior.

### 7.3 The Predictive Distribution

Given a new input $\mathbf{x}_* \in \mathbb{R}^d$, we want to predict $y_* = \boldsymbol{\beta}^\top \mathbf{x}_* + \epsilon_*$.

**Marginalising** over the posterior $\boldsymbol{\beta} \mid \mathbf{y}, X$:

$$p(y_* \mid \mathbf{x}_*, \mathbf{y}, X) = \int p(y_* \mid \mathbf{x}_*, \boldsymbol{\beta}) \, p(\boldsymbol{\beta} \mid \mathbf{y}, X) \, d\boldsymbol{\beta}$$

Since both distributions are Gaussian, the integral is analytically tractable:

$$y_* \mid \mathbf{x}_*, \mathbf{y}, X \sim \mathcal{N}(\mathbf{x}_*^\top \boldsymbol{\mu}_n, \; \underbrace{\mathbf{x}_*^\top \Sigma_n \mathbf{x}_*}_{\text{epistemic}} + \underbrace{\sigma^2}_{\text{aleatoric}})$$

**Two types of uncertainty:**
- **Epistemic (reducible) uncertainty:** $\mathbf{x}_*^\top \Sigma_n \mathbf{x}_*$ — uncertainty about $\boldsymbol{\beta}$, decreases with more data
- **Aleatoric (irreducible) uncertainty:** $\sigma^2$ — inherent noise, constant regardless of sample size

The predictive mean $\hat{y}_* = \mathbf{x}_*^\top \boldsymbol{\mu}_n$ matches the Ridge prediction. The predictive variance matches the frequentist prediction interval formula from Section 3.4.

**For AI:** Bayesian predictive distributions are the foundation of **uncertainty quantification** in ML. Conformal prediction (Angelopoulos & Bates, 2023) provides distribution-free prediction intervals that reduce to the Bayesian intervals for linear models. LLM calibration is the empirical question of whether predicted token probabilities match observed frequencies — a Bayesian question about predictive accuracy.

### 7.4 Evidence and Marginal Likelihood

The **marginal likelihood** (evidence) $p(\mathbf{y} \mid X)$ integrates out $\boldsymbol{\beta}$:

$$p(\mathbf{y} \mid X) = \int p(\mathbf{y} \mid X, \boldsymbol{\beta}) \, p(\boldsymbol{\beta}) \, d\boldsymbol{\beta}$$

For the Gaussian model, this is analytically tractable:

$$\log p(\mathbf{y} \mid X) = -\frac{1}{2}\mathbf{y}^\top \!\left(\frac{1}{\sigma^2}I + \frac{1}{\sigma^2 \tau^2}X^\top X\right)^{-1}\!\mathbf{y} - \frac{1}{2}\log\det(\cdot) - \frac{n}{2}\log(2\pi)$$

The evidence automatically penalises model complexity — it is the Bayesian version of AIC/BIC. A more complex model (larger $\tau^2$) can fit the data better (higher likelihood) but pays a larger determinant penalty. The evidence is maximised at the value of $\tau^2$ (or $\lambda$) that best balances fit and complexity — this is **Empirical Bayes** or **Type-II MLE**.

### 7.5 Connection to Gaussian Processes and NTK

**Kernel regression as Bayesian linear regression.** If the prior is placed on a feature map $\phi: \mathbb{R}^d \to \mathbb{R}^m$:

$$f(\mathbf{x}) = \boldsymbol{\beta}^\top \phi(\mathbf{x}), \quad \boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$$

the predictive distribution over functions $f$ is a **Gaussian Process** with kernel $k(\mathbf{x}, \mathbf{x}') = \tau^2 \phi(\mathbf{x})^\top \phi(\mathbf{x}')$. Conditioning on data gives kernel ridge regression:

$$\hat{f}(\mathbf{x}_*) = \mathbf{k}_*^\top (K + \sigma^2 I)^{-1} \mathbf{y}$$

where $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ and $\mathbf{k}_* = [k(\mathbf{x}^{(1)}, \mathbf{x}_*), \ldots, k(\mathbf{x}^{(n)}, \mathbf{x}_*)]^\top$.

**Neural Tangent Kernel.** For infinitely wide neural networks trained with gradient descent, the kernel is the **NTK** $\Theta(\mathbf{x}, \mathbf{x}')$. Training corresponds to kernel ridge regression with this kernel — linear model dynamics in function space. This explains why wide networks can be analysed theoretically using linear model tools (Jacot et al., 2018; Lee et al., 2019).


---

## 8. Linear Classification

### 8.1 Logistic Regression

Logistic regression models the probability that $y \in \{0, 1\}$ given $\mathbf{x}$:

$$p(y = 1 \mid \mathbf{x}; \mathbf{w}, b) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}$$

where $\sigma(z) = 1/(1 + e^{-z})$ is the **sigmoid (logistic)** function.

**Log-odds (logit) interpretation.** Taking the log-odds:

$$\log \frac{p(y=1 \mid \mathbf{x})}{p(y=0 \mid \mathbf{x})} = \mathbf{w}^\top \mathbf{x} + b$$

The log-odds are **linear** in $\mathbf{x}$ — this is the defining property of logistic regression. The decision boundary $\{x : \mathbf{w}^\top \mathbf{x} + b = 0\}$ is a hyperplane.

**Cross-entropy loss.** The negative log-likelihood of the Bernoulli distribution:

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{n}\sum_{i=1}^n \left[y^{(i)} \log p^{(i)} + (1 - y^{(i)}) \log(1 - p^{(i)})\right]$$

where $p^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)} + b)$. This can be written as:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b)$$

i.e., logistic regression maximises the likelihood of the observed labels.

**For AI:** The cross-entropy loss for logistic regression is exactly the same loss used to train LLMs: $\mathcal{L} = -\sum_t \log p(x_t \mid x_{<t})$ is multi-class cross-entropy where the "classes" are tokens. The only difference is the model for $p$ — a softmax over a Transformer's output rather than a sigmoid over a linear function.

### 8.2 Gradient, Hessian, and Global Convexity

**Gradient.** Computing $\nabla_{\mathbf{w}} \mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial w_j} = -\frac{1}{n}\sum_i \left[y^{(i)} \frac{1}{p^{(i)}} \frac{\partial p^{(i)}}{\partial w_j} + (1-y^{(i)}) \frac{-1}{1-p^{(i)}} \frac{\partial p^{(i)}}{\partial w_j}\right]$$

Using $\frac{\partial \sigma(z)}{\partial z} = \sigma(z)(1-\sigma(z))$ and $\frac{\partial z^{(i)}}{\partial w_j} = x_j^{(i)}$:

$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{n}\sum_i (p^{(i)} - y^{(i)}) x_j^{(i)}$$

In matrix form:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{n} X^\top (\mathbf{p} - \mathbf{y})$$

This is elegant: the gradient is the residual $\mathbf{p} - \mathbf{y}$ (predicted minus actual) dotted with features — exactly the same structure as the OLS gradient.

**Hessian:**

$$H = \frac{1}{n} X^\top \operatorname{diag}(\mathbf{p} \odot (1-\mathbf{p})) X$$

**Global convexity proof.** For any vector $\mathbf{v}$:

$$\mathbf{v}^\top H \mathbf{v} = \frac{1}{n}\mathbf{v}^\top X^\top D X \mathbf{v} = \frac{1}{n} \lVert D^{1/2} X \mathbf{v} \rVert_2^2 \geq 0$$

where $D = \operatorname{diag}(\mathbf{p} \odot (1-\mathbf{p}))$ with all diagonal entries in $(0, 1)$. Therefore $H \succeq 0$ — the cross-entropy loss is **convex** in $(\mathbf{w}, b)$, and every local minimum is global. When $X$ has full column rank, $H \succ 0$ and the minimum is unique.

**IRLS (Iteratively Reweighted Least Squares).** Newton's method applied to logistic regression:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - H^{-1} \nabla_\mathbf{w} \mathcal{L} = (X^\top D X)^{-1} X^\top D \mathbf{z}$$

where $\mathbf{z} = X\mathbf{w}^{(t)} + D^{-1}(\mathbf{y} - \mathbf{p}^{(t)})$ is the **adjusted response**. Each Newton step is a weighted least squares problem — the "weights" are $D$ and they change each iteration. IRLS converges quadratically near the optimum.

### 8.3 Softmax Regression (Multiclass)

For $K$-class classification, the model is:

$$p(y = k \mid \mathbf{x}; W) = \frac{\exp(\mathbf{w}_k^\top \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{w}_j^\top \mathbf{x})} = \operatorname{softmax}(W\mathbf{x})_k$$

where $W \in \mathbb{R}^{K \times d}$ collects all class weight vectors as rows.

**Multi-class cross-entropy loss:**

$$\mathcal{L}(W) = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y^{(i)}_k \log p(y=k \mid \mathbf{x}^{(i)}; W)$$

where $\mathbf{y}^{(i)} \in \{0,1\}^K$ is the one-hot label vector.

**Gradient (matrix form):**

$$\nabla_W \mathcal{L} = \frac{1}{n}(P - Y)^\top X$$

where $P \in \mathbb{R}^{n \times K}$ has rows $\operatorname{softmax}(W\mathbf{x}^{(i)})^\top$ and $Y \in \mathbb{R}^{n \times K}$ has rows $\mathbf{y}^{(i)\top}$.

**Redundancy and identifiability.** The softmax is invariant to adding a constant to all logits: $\operatorname{softmax}(W\mathbf{x} + c\mathbf{1}) = \operatorname{softmax}(W\mathbf{x})$. Therefore $W$ is not identifiable — we can only identify differences between class weights. Fix this by setting $\mathbf{w}_K = \mathbf{0}$ (reducing to $K-1$ weight vectors).

**For AI:** The final layer of every language model is a softmax regression over the vocabulary ($K \approx 32{,}000$ to $256{,}000$ tokens). The weight matrix $W \in \mathbb{R}^{K \times d_{\text{model}}}$ is the **unembedding matrix**. In many LLMs, the unembedding matrix is tied to (or initialised from) the embedding matrix — weight tying reduces parameters and improves calibration.

### 8.4 Decision Boundaries

The decision boundary of a linear classifier $\hat{y} = \arg\max_k \mathbf{w}_k^\top \mathbf{x}$ between classes $k$ and $l$ is:

$$\{\mathbf{x} : \mathbf{w}_k^\top \mathbf{x} = \mathbf{w}_l^\top \mathbf{x}\} = \{\mathbf{x} : (\mathbf{w}_k - \mathbf{w}_l)^\top \mathbf{x} = 0\}$$

A hyperplane perpendicular to the difference vector $\mathbf{w}_k - \mathbf{w}_l$. The collection of all pairwise hyperplanes creates a Voronoi partition of $\mathbb{R}^d$.

**Voronoi polytopes interpretation.** The classification regions are convex polytopes (intersections of half-spaces). Linear classifiers can only shatter $d+1$ points in general position in $\mathbb{R}^d$ — this is the VC dimension of linear classifiers.

### 8.5 Maximum Entropy Interpretation

Logistic regression is the **maximum entropy classifier**: among all distributions $p(y \mid \mathbf{x})$ that match the empirical feature-label covariance $\frac{1}{n}\sum_i \mathbf{x}^{(i)} \mathbb{1}[y^{(i)}=k]$, logistic regression gives the distribution with maximum entropy.

**Derivation (binary case).** Maximise $H(p) = -\sum_i p_i \log p_i$ subject to linear constraints on feature expectations:

$$\max_p H(p) \quad \text{s.t.} \quad \mathbb{E}_p[x_j y] = \hat{\mu}_j, \quad p(y=0) + p(y=1) = 1$$

Via Lagrangian duality, the optimal $p$ has the form $p(y=1 \mid \mathbf{x}) = \sigma(\boldsymbol{\lambda}^\top \mathbf{x})$ — exactly logistic regression with Lagrange multipliers as weights. This provides a principled justification for logistic regression independent of the likelihood derivation.

---

## 9. Discriminative vs. Generative Models

### 9.1 The Fundamental Modelling Choice

Every supervised learning method implicitly or explicitly models one of:

- **Discriminative models**: $p(y \mid \mathbf{x}; \boldsymbol{\theta})$ — directly model the conditional distribution of labels given features
- **Generative models**: $p(\mathbf{x}, y; \boldsymbol{\theta}) = p(\mathbf{x} \mid y; \boldsymbol{\theta}) p(y; \boldsymbol{\theta})$ — model the joint distribution of features and labels

**Bayes' theorem connects them:** $p(y \mid \mathbf{x}) = p(\mathbf{x} \mid y) p(y) / p(\mathbf{x})$.

Generative classifiers use Bayes' theorem to turn the class-conditional density $p(\mathbf{x} \mid y)$ into a posterior $p(y \mid \mathbf{x})$.

**Ng & Jordan (2001) comparison:**

| Property | Discriminative (Logistic) | Generative (Naive Bayes) |
|---|---|---|
| Asymptotic accuracy | Higher (more expressive) | Lower (stronger assumptions) |
| Sample efficiency | Slow (needs more data) | Fast (makes assumptions) |
| Missing features | Hard (retrain) | Easy (marginalise over missing) |
| Outlier sensitivity | More robust | Less robust |
| Joint modelling | No | Yes (can generate $\mathbf{x}$) |
| Causal structure | Ignored | Partly modelled |

Key insight (Ng & Jordan): generative models reach their asymptotic error faster (with fewer samples) but have higher asymptotic error. Discriminative models are better with large $n$; generative models are better with small $n$.

### 9.2 Linear Discriminant Analysis

**Generative model:** assume class-conditional Gaussians with **shared covariance**:

$$p(\mathbf{x} \mid y = k) = \mathcal{N}(\boldsymbol{\mu}_k, \Sigma), \quad k = 1, \ldots, K$$

**MLE estimates:** $\hat{\boldsymbol{\mu}}_k = \frac{1}{n_k}\sum_{i: y^{(i)}=k} \mathbf{x}^{(i)}$ and $\hat{\Sigma} = \frac{1}{n-K}\sum_k \sum_{i: y^{(i)}=k} (\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}^{(i)} - \hat{\boldsymbol{\mu}}_k)^\top$ (pooled within-class covariance).

**Decision rule via Bayes:** assign $\hat{y} = \arg\max_k \log p(\mathbf{x} \mid y=k) + \log \pi_k$:

$$\delta_k(\mathbf{x}) = \mathbf{x}^\top \hat{\Sigma}^{-1} \hat{\boldsymbol{\mu}}_k - \frac{1}{2}\hat{\boldsymbol{\mu}}_k^\top \hat{\Sigma}^{-1} \hat{\boldsymbol{\mu}}_k + \log \hat{\pi}_k$$

This is **linear in $\mathbf{x}$** — LDA produces a linear decision boundary even though it starts from a generative model.

**Fisher's criterion.** LDA also maximises the **Fisher discriminant**: the ratio of between-class variance to within-class variance of the projected data:

$$\max_{\mathbf{w}} \frac{\mathbf{w}^\top S_B \mathbf{w}}{\mathbf{w}^\top S_W \mathbf{w}}$$

where $S_B = \sum_k n_k (\hat{\boldsymbol{\mu}}_k - \bar{\boldsymbol{\mu}})(\hat{\boldsymbol{\mu}}_k - \bar{\boldsymbol{\mu}})^\top$ (between-class scatter) and $S_W = \sum_k \sum_i (x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^\top$ (within-class scatter). The solution is the generalised eigenvectors of $(S_W^{-1} S_B)$ — a $K-1$ dimensional linear subspace.

**For AI:** LDA is used for **class activation analysis** in NLP — projecting token embeddings onto the LDA direction separating sentiment classes, for example. It is also the mathematical foundation of **probing with multiple classes**.

### 9.3 Quadratic Discriminant Analysis

**Generative model:** class-conditional Gaussians with **class-specific covariances**:

$$p(\mathbf{x} \mid y = k) = \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$$

The log-posterior:

$$\log p(y=k \mid \mathbf{x}) \propto -\frac{1}{2}\log\det(\Sigma_k) - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_k)^\top \Sigma_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k) + \log\pi_k$$

This is **quadratic in $\mathbf{x}$** — QDA has quadratic decision boundaries. More parameters required: $O(d^2)$ per class vs. $O(d^2)$ shared in LDA.

**LDA vs. QDA tradeoff:** LDA has fewer parameters ($O(Kd + d^2)$ vs. $O(Kd^2)$ for QDA) and is more robust when $n$ is small relative to $d$. QDA is more flexible but needs $n \gg Kd^2$ to estimate accurately.

### 9.4 Naive Bayes

**Assumption:** conditional independence of features given class:

$$p(\mathbf{x} \mid y = k) = \prod_{j=1}^d p(x_j \mid y = k)$$

This is almost certainly wrong (features are correlated) but often works well in practice due to:
1. Small number of parameters: $O(Kd)$ vs. $O(Kd^2)$ for LDA
2. Robustness to model misspecification at low $n$

**Decision rule:**

$$\hat{y} = \arg\max_k \left[\log \pi_k + \sum_{j=1}^d \log p(x_j \mid y=k)\right]$$

For Gaussian features, Naive Bayes has linear boundaries (like LDA). For Bernoulli features (binary text classification), it is equivalent to a linear classifier with **additive smoothing** (Laplace correction).

**For AI:** The original GPT-3 few-shot "classifier" is a naive Bayes model on tokens — it estimates $p(\text{label} \mid \text{context})$ by $p(\text{context} \mid \text{label}) p(\text{label}) / p(\text{context})$ using the LLM's density estimates. This generative interpretation of in-context learning was formalised by Xie et al. (2022).

### 9.5 When Generative Wins and When Discriminative Wins

The Ng & Jordan (2001) result can be stated precisely: for $d$-dimensional Gaussian LDA vs. logistic regression, LDA reaches its (higher) asymptotic error at $O(d)$ samples, while logistic regression needs $O(d^2)$ samples to reach its (lower) asymptotic error. Therefore:

- **Small $n/d$ ratio**: generative models (LDA, Naive Bayes) win
- **Large $n/d$ ratio**: discriminative models (logistic regression) win

In modern AI, $n \gg d$ for LLM pretraining (trillions of tokens, tens of thousands of features), which is why discriminative training (cross-entropy loss) dominates. For few-shot fine-tuning or domain adaptation with small $n$, Bayesian/generative methods become competitive.


---

## 10. Support Vector Machines

### 10.1 Margin Maximisation

An SVM finds the **maximum-margin hyperplane** separating two classes $y \in \{-1, +1\}$.

**Functional margin** of a hyperplane $(\mathbf{w}, b)$ for example $(\mathbf{x}^{(i)}, y^{(i)})$:

$$\hat{\gamma}^{(i)} = y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)$$

Positive if classified correctly. **Geometric margin** (distance from point to hyperplane):

$$\gamma^{(i)} = \frac{\hat{\gamma}^{(i)}}{\lVert \mathbf{w} \rVert_2} = \frac{y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)}{\lVert \mathbf{w} \rVert_2}$$

**Hard-margin SVM primal.** Maximise the minimum geometric margin (the margin $\rho$):

$$\max_{\mathbf{w}, b} \rho \quad \text{s.t.} \quad \frac{y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)}{\lVert \mathbf{w} \rVert_2} \geq \rho, \quad i = 1, \ldots, n$$

By rescaling $\mathbf{w}$ so that $\rho \lVert \mathbf{w} \rVert_2 = 1$ (the functional margin is 1 for support vectors), this becomes:

$$\min_{\mathbf{w}, b} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1, \quad i = 1, \ldots, n$$

The margin is $2/\lVert \mathbf{w} \rVert_2$ (distance between the two supporting hyperplanes $\mathbf{w}^\top \mathbf{x} + b = \pm 1$).

### 10.2 Lagrangian Duality and the Dual Problem

**Lagrangian:**

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 - \sum_{i=1}^n \alpha_i \left[y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) - 1\right]$$

**Stationarity conditions** (set partial derivatives to zero):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0 \implies \mathbf{w} = \sum_{i=1}^n \alpha_i y^{(i)} \mathbf{x}^{(i)}$$

$$\frac{\partial \mathcal{L}}{\partial b} = 0 \implies \sum_{i=1}^n \alpha_i y^{(i)} = 0$$

**Dual objective** (substitute back, simplify):

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} (\mathbf{x}^{(i)})^\top \mathbf{x}^{(j)}$$

$$\text{subject to} \quad \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y^{(i)} = 0$$

This is a **quadratic program** with $n$ variables. Strong duality holds (Slater's condition satisfied).

**KKT conditions** at optimality: for each $i$, either $\alpha_i = 0$ (example is not a support vector) or $y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) = 1$ (example is on the margin — a support vector). The optimal $\mathbf{w}$ is a sparse combination of training points:

$$\hat{\mathbf{w}} = \sum_{i: \alpha_i > 0} \alpha_i y^{(i)} \mathbf{x}^{(i)}$$

Only the support vectors (a small fraction of training data) determine the decision boundary.

### 10.3 Soft Margin SVM

For non-separable data, introduce **slack variables** $\xi_i \geq 0$ allowing violations:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C\sum_{i=1}^n \xi_i$$

$$\text{s.t.} \quad y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Hinge loss equivalence.** Note $\xi_i = \max(0, 1 - y^{(i)} f^{(i)})$ where $f^{(i)} = \mathbf{w}^\top \mathbf{x}^{(i)} + b$. So:

$$\min_{\mathbf{w}, b} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C\sum_{i=1}^n \max\!\left(0, 1 - y^{(i)}(\mathbf{w}^\top \mathbf{x}^{(i)} + b)\right)$$

The soft-margin SVM minimises **Ridge penalty + hinge loss** — a regularised loss minimisation problem. The dual is:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y^{(i)}y^{(j)}(\mathbf{x}^{(i)})^\top\mathbf{x}^{(j)}, \quad 0 \leq \alpha_i \leq C$$

The box constraint $\alpha_i \leq C$ bounds the influence of each training point.

### 10.4 The Kernel Trick

**Key observation:** the dual objective and prediction function $f(\mathbf{x}) = \sum_i \alpha_i y^{(i)} k(\mathbf{x}^{(i)}, \mathbf{x}) + b$ only depend on **pairwise inner products** $\langle \mathbf{x}^{(i)}, \mathbf{x}^{(j)} \rangle$. Replacing these with a kernel $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \langle \phi(\mathbf{x}^{(i)}), \phi(\mathbf{x}^{(j)}) \rangle$ implicitly operates in the feature space defined by $\phi$.

**Mercer's theorem.** A symmetric function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a valid kernel (i.e., there exists a feature map $\phi$ such that $k(\mathbf{x}, \mathbf{z}) = \langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle$) if and only if the kernel matrix $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ is **positive semidefinite** for all datasets.

**Standard kernels:**

| Kernel | Formula | Feature space dimension |
|---|---|---|
| Linear | $k(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top \mathbf{z}$ | $d$ |
| Polynomial | $k(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top \mathbf{z} + c)^p$ | $\binom{d+p}{p}$ |
| RBF/Gaussian | $k(\mathbf{x},\mathbf{z}) = \exp(-\lVert\mathbf{x}-\mathbf{z}\rVert^2 / 2\ell^2)$ | $\infty$ |
| Laplace | $k(\mathbf{x},\mathbf{z}) = \exp(-\lVert\mathbf{x}-\mathbf{z}\rVert / \ell)$ | $\infty$ |

**For AI:** Kernel methods are the mathematical foundation of **attention mechanisms**. Self-attention computes $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}(QK^\top/\sqrt{d_k})V$, where $QK^\top/\sqrt{d_k}$ is a matrix of kernel evaluations between queries and keys. Cosine-similarity attention uses $k(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k} / (\lVert\mathbf{q}\rVert \lVert\mathbf{k}\rVert)$ — a normalised linear kernel. Random Feature Attention (Peng et al., 2021) approximates the softmax kernel with random features to achieve linear-time attention.

### 10.5 Structural Risk Minimisation

SVMs minimise **structural risk**: the sum of empirical risk (training error) and a complexity term:

$$R[f] \leq R_{\text{emp}}[f] + \sqrt{\frac{h(\log(2n/h) + 1) - \log(\delta/4)}{n}}$$

where $h$ is the VC dimension of the hypothesis class and $\delta$ is the failure probability.

For SVMs with margin $\rho$: $h \leq \lfloor \min(\lVert\mathbf{w}\rVert^2 R^2 / \rho^2, d) \rfloor + 1$ where $R = \max_i \lVert\mathbf{x}^{(i)}\rVert$. Maximising the margin $\rho$ (equivalently, minimising $\lVert\mathbf{w}\rVert$) minimises $h$ and tightens the bound. This provides a **theoretical justification** for the SVM objective.

---

## 11. Bias-Variance Tradeoff and Model Selection

### 11.1 Bias-Variance Decomposition

For squared loss, the expected prediction error at a test point $\mathbf{x}$ can be decomposed:

$$\mathbb{E}\!\left[(y - \hat{f}(\mathbf{x}))^2\right] = \underbrace{\left[\mathbb{E}[\hat{f}(\mathbf{x})] - f^*(\mathbf{x})\right]^2}_{\text{Bias}^2} + \underbrace{\operatorname{Var}[\hat{f}(\mathbf{x})]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}$$

where $f^*(\mathbf{x}) = \mathbb{E}[y \mid \mathbf{x}]$ is the true conditional mean and expectation is over training sets.

**For linear models with Ridge ($\lambda$):**

$$\text{Bias}^2(\lambda) = \lambda^2 \boldsymbol{\beta}^{*\top}(X^\top X + \lambda I)^{-2}\boldsymbol{\beta}^* \cdot \lVert\mathbf{x}\rVert^2$$

$$\text{Variance}(\lambda) = \sigma^2 \mathbf{x}^\top (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} \mathbf{x}$$

Both are monotone functions of $\lambda$: bias increases, variance decreases. The optimal $\lambda^*$ minimises their sum.

**For AI:** Every regularisation decision in neural network training is a bias-variance tradeoff. Dropout increases bias (averaging over masked networks) but decreases variance (prevents co-adaptation of features). Early stopping is regularisation: stopping before convergence biases the model but reduces variance. The $\lambda$ parameter in Ridge, the mask rate in Dropout, and the number of training steps are all points on the same bias-variance frontier.

### 11.2 The Double Descent Phenomenon

Classical statistical learning theory predicts a U-shaped test error curve:

1. Underfitting regime ($d < n$): error decreases as model complexity increases
2. Overfitting regime ($d > n$): error increases (too many parameters, memorises noise)

**Double descent** (Belkin et al., 2020; Hastie et al., 2022) reveals a second descent:

3. Overparameterised regime ($d \gg n$): after crossing the **interpolation threshold** ($d = n$), error decreases again — even without explicit regularisation

```
DOUBLE DESCENT TEST ERROR CURVE
════════════════════════════════════════════════════════════════════════

   Test      ▲
   error     │
             │     classical
             │     regime        interpolation   modern regime
             │                   threshold
          ↑  │    ╭──────╮           │
  Classical  │   ╱       ╲          │╲            ╲
  regime     │  ╱         ╲         │  ╲            ╲___________
             │ ╱           ╲________│   ╲
             │                      │
             └──────────────────────┼──────────────────────────────▶
                                   d=n                        Model size

   • Left of threshold: traditional bias-variance tradeoff
   • At threshold: worst case (model fits all data but barely)
   • Right of threshold: "benign overfitting" — minimum-norm interpolator
     generalises better as d/n increases further

════════════════════════════════════════════════════════════════════════
```

**Mathematical explanation (linear case).** For $d > n$, the minimum-norm OLS solution $\hat{\boldsymbol{\beta}}^\dagger = X^\dagger \mathbf{y}$ has test MSE:

$$\mathbb{E}[\text{MSE}] = \sigma^2 \frac{n}{d - n - 1} + \sigma^2 \frac{d - n}{n} \cdot \frac{\lVert\boldsymbol{\beta}^*\rVert^2 - \lVert\boldsymbol{\beta}^*\rVert_{\mathcal{C}(X)}^2}{\lVert\boldsymbol{\beta}^*\rVert^2}$$

As $d/n \to \infty$ (extreme overparameterisation), both terms vanish — the interpolator generalises perfectly in the limit. **For AI:** LLMs operate in the extreme overparameterisation regime ($d \sim 10^9$, $n \sim 10^{13}$ tokens). The double descent theory explains why these models generalise despite memorising training data.

### 11.3 Cross-Validation

**$k$-fold CV.** Split data into $k$ folds. For each fold $j$: train on the other $k-1$ folds, evaluate on fold $j$. Average the $k$ evaluation scores:

$$\text{CV}_k = \frac{1}{k}\sum_{j=1}^k \text{Error}(\text{model trained on } \mathcal{D} \setminus \mathcal{D}_j, \text{ evaluated on } \mathcal{D}_j)$$

**LOOCV (leave-one-out):** $k = n$. For linear models, the LOOCV error is:

$$\text{LOOCV} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y^{(i)} - \hat{y}^{(i)}}{1 - h_{ii}}\right)^2$$

This requires only a single model fit — the leverage scores $h_{ii}$ encode how much each point controls its own prediction.

**Generalised CV (GCV).** Approximation for large datasets:

$$\text{GCV}(\lambda) = \frac{\text{RSS}/(n - \operatorname{df}(\lambda))^2}{1}$$

where $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda)$ is the effective degrees of freedom of the Ridge smoother.

### 11.4 Information Criteria

**Akaike Information Criterion (AIC).** Asymptotically equivalent to LOOCV for large $n$:

$$\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2k$$

where $\ell$ is the maximised log-likelihood and $k$ is the number of parameters.

**Bayesian Information Criterion (BIC).** Consistent model selection (selects the true model as $n \to \infty$):

$$\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}}) + k\log n$$

**Derivation of AIC.** AIC approximates the expected KL divergence from the true distribution $p^*$ to the fitted model $\hat{p}$:

$$\mathbb{E}_{\mathcal{D}}\!\left[-2 \log p(\mathbf{y}_{\text{new}} \mid \hat{\boldsymbol{\theta}}(\mathcal{D}))\right] \approx -2\ell(\hat{\boldsymbol{\theta}}) + 2k$$

AIC corrects for the optimism bias in training likelihood.

### 11.5 Regularisation Paths and Hyperparameter Tuning

For Ridge and Lasso, computing the entire regularisation path (solution for all $\lambda$) costs the same as a single fit:

- **Ridge path:** $\hat{\boldsymbol{\beta}}_\lambda = V(\Sigma^2 + \lambda I)^{-1}\Sigma U^\top \mathbf{y}$ — compute SVD once, evaluate for all $\lambda$ in $O(d)$ per value
- **Lasso path (LARS):** $O(nd^2)$ for the entire path using the piecewise-linear structure

**Practical hyperparameter selection:**
1. Define a logarithmic grid: $\lambda \in \{\lambda_{\max}/100, \lambda_{\max}/10, \ldots, \lambda_{\max}\}$ where $\lambda_{\max} = \lVert X^\top \mathbf{y} \rVert_\infty / n$ (largest $\lambda$ giving all-zero Lasso solution)
2. For each $\lambda$: compute $k$-fold CV error
3. Select $\lambda^* = \arg\min_\lambda \text{CV}(\lambda)$ or apply the one-SE rule


---

## 12. Optimisation Methods for Linear Models

### 12.1 Gradient Descent and Convergence Analysis

For a convex, $L$-smooth objective (Lipschitz gradient: $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq L\lVert \mathbf{x} - \mathbf{y} \rVert$):

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - \eta \nabla \mathcal{L}(\boldsymbol{\beta}^{(t)})$$

with step size $\eta \leq 1/L$ converges at rate:

$$\mathcal{L}(\boldsymbol{\beta}^{(t)}) - \mathcal{L}(\boldsymbol{\beta}^*) \leq \frac{\lVert \boldsymbol{\beta}^{(0)} - \boldsymbol{\beta}^* \rVert_2^2}{2\eta t}$$

i.e., $O(1/t)$ convergence. For $\mu$-strongly convex objectives (OLS, Ridge), linear convergence holds:

$$\lVert \boldsymbol{\beta}^{(t)} - \boldsymbol{\beta}^* \rVert_2 \leq \left(1 - \frac{\mu}{L}\right)^t \lVert \boldsymbol{\beta}^{(0)} - \boldsymbol{\beta}^* \rVert_2$$

For OLS: $L = \lambda_{\max}(X^\top X)/n$, $\mu = \lambda_{\min}(X^\top X)/n$ (for Ridge: $\mu = \lambda_{\min}(X^\top X)/n + \lambda$). The condition number $\kappa = L/\mu = \lambda_{\max}/\lambda_{\min}$ determines convergence speed.

**For Ridge:** $\mu_{\text{Ridge}} = \lambda_{\min}(X^\top X)/n + \lambda > \lambda > 0$ even when $X^\top X$ is singular. This is why Ridge is numerically preferable to OLS — the optimisation problem is better conditioned.

**For AI:** The condition number of the loss landscape determines training speed for neural networks. Batch normalisation, layer normalisation, and careful weight initialisation all aim to reduce the effective condition number, enabling larger learning rates and faster convergence.

### 12.2 Newton's Method and IRLS

For a twice-differentiable convex objective, Newton's method uses the Hessian to rescale the gradient:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - [H(\boldsymbol{\beta}^{(t)})]^{-1} \nabla \mathcal{L}(\boldsymbol{\beta}^{(t)})$$

**Quadratic convergence:** in a neighbourhood of $\boldsymbol{\beta}^*$:

$$\lVert \boldsymbol{\beta}^{(t+1)} - \boldsymbol{\beta}^* \rVert \leq c \lVert \boldsymbol{\beta}^{(t)} - \boldsymbol{\beta}^* \rVert^2$$

for some constant $c$. Quadratic convergence means the number of correct digits doubles each iteration — Newton's method needs $O(\log\log(1/\epsilon))$ iterations vs. $O(1/\epsilon)$ for gradient descent.

**IRLS for logistic regression.** As derived in Section 8.2, each Newton step is:

$$\boldsymbol{\beta}^{(t+1)} = (X^\top D^{(t)} X)^{-1} X^\top D^{(t)} \mathbf{z}^{(t)}$$

where $D^{(t)} = \operatorname{diag}(p^{(t)}_i(1-p^{(t)}_i))$ and $\mathbf{z}^{(t)} = X\boldsymbol{\beta}^{(t)} + (D^{(t)})^{-1}(\mathbf{y} - \mathbf{p}^{(t)})$.

This is a Weighted Least Squares problem with weights $D^{(t)}$ and response $\mathbf{z}^{(t)}$. IRLS reduces logistic regression to a sequence of weighted OLS problems — exploiting fast linear algebra solvers.

### 12.3 Stochastic Gradient Descent

**Mini-batch SGD.** Estimate the gradient on a random batch $\mathcal{B} \subseteq [n]$ of size $B$:

$$\tilde{\nabla} \mathcal{L}(\boldsymbol{\beta}) = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i(\boldsymbol{\beta})$$

For OLS: $\tilde{\nabla} = -\frac{2}{B} X_\mathcal{B}^\top (\mathbf{y}_\mathcal{B} - X_\mathcal{B}\boldsymbol{\beta})$. This is an **unbiased** estimator of the full gradient.

**SGD convergence.** For convex Lipschitz objectives with gradient variance $G^2 = \mathbb{E}[\lVert \tilde{\nabla} - \nabla \rVert^2]$:

$$\mathbb{E}[\mathcal{L}(\boldsymbol{\beta}^{(t)}) - \mathcal{L}(\boldsymbol{\beta}^*)] \leq \frac{R^2 + G^2 \sum_{t=1}^T \eta_t^2}{2\sum_{t=1}^T \eta_t}$$

For step size $\eta_t = c/\sqrt{t}$: convergence rate $O(1/\sqrt{T})$ — slower than GD but each step costs $O(Bd)$ instead of $O(nd)$.

**Variance-variance tradeoff.** Large batch $B \to n$: gradient variance → 0, convergence → GD rate. Small batch $B = 1$: maximum variance, cheapest per step, slowest convergence. In practice, $B = 32$–$4096$ balances hardware utilisation and gradient noise.

**For AI:** LLM pretraining uses mini-batch SGD with variants (AdamW). The **gradient noise** introduced by mini-batching acts as an implicit regulariser — models trained with small batches generalise better (Keskar et al., 2017). This is the stochastic version of the bias-variance tradeoff.

### 12.4 Coordinate Descent

**Cyclic coordinate descent** updates one coordinate $j$ at a time, keeping all others fixed. For a separable regulariser $R(\boldsymbol{\beta}) = \sum_j r(\beta_j)$:

$$\beta_j^{\text{new}} = \arg\min_{\beta_j} \mathcal{L}(\boldsymbol{\beta}) = \arg\min_{\beta_j} \left[\frac{1}{n}f(\boldsymbol{\beta}) + r(\beta_j)\right]$$

For Lasso ($r(\beta_j) = \lambda|\beta_j|$): the coordinate update has the closed-form soft-thresholding solution (Section 5.2). For Ridge ($r(\beta_j) = \lambda\beta_j^2$): the update is $\beta_j = z_j / (1 + 2\lambda)$.

**Convergence.** Cyclic coordinate descent converges for convex objectives with Lipschitz partial derivatives. Randomised coordinate descent (choosing $j$ uniformly at random) converges at $O(d/t)$ rate for smooth objectives.

**ADMM (Alternating Direction Method of Multipliers).** For group penalties and equality-constrained problems, ADMM splits the problem into simpler subproblems solved alternately. Used for Group Lasso, nuclear norm regularisation, and distributed optimisation.

### 12.5 Second-Order Methods: L-BFGS

**L-BFGS (Limited-memory BFGS)** approximates the inverse Hessian using the last $m = 5$–$20$ gradient vectors:

$$\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)} - H_t^{-1} \nabla \mathcal{L}(\boldsymbol{\beta}^{(t)})$$

where $H_t^{-1}$ is the L-BFGS quasi-Newton approximation. Memory: $O(md)$ vs. $O(d^2)$ for full BFGS.

**When to use which method:**

| Method | Cost/step | Rate | Best for |
|---|---|---|---|
| GD | $O(nd)$ | $O(1/t)$ or linear | Small $n$, smooth loss |
| SGD | $O(Bd)$ | $O(1/\sqrt{t})$ | Large $n$, batch parallelism |
| Newton/IRLS | $O(nd^2 + d^3)$ | Quadratic | Small $d$, logistic regression |
| Coord. descent | $O(n)$/coord | Linear | Lasso, separable penalty |
| L-BFGS | $O(mnd)$ | Superlinear | Medium $n$, smooth differentiable |

---

## 13. Deep Learning and LLM Connections

### 13.1 Linear Layers as Generalised Linear Models

A neural network layer $\mathbf{h} = W\mathbf{x} + \mathbf{b}$ with $W \in \mathbb{R}^{k \times d}$, $\mathbf{b} \in \mathbb{R}^k$ computes $k$ simultaneous linear regressions — one per output unit. With MSE loss, training this layer is equivalent to $k$ independent OLS problems. With weight decay $\lambda \lVert W \rVert_F^2$, it becomes $k$ independent Ridge problems.

**Frozen features + linear head.** Fine-tuning only the final linear layer of a pretrained model is exactly logistic regression on fixed features:

$$\hat{y} = \operatorname{softmax}(W \phi_\theta(\mathbf{x}))$$

where $\phi_\theta$ is the frozen encoder. This is the **linear probing** paradigm in representation learning evaluation (Chen et al., 2020; He et al., 2022). The quality of $\phi_\theta$ is measured by how well a linear head can classify from its representations.

### 13.2 LoRA as Low-Rank Linear Regression

**LoRA (Low-Rank Adaptation, Hu et al., 2022)** fine-tunes a pretrained weight $W_0 \in \mathbb{R}^{d \times k}$ by adding a low-rank update:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$. During training, $W_0$ is frozen; only $A$ and $B$ are updated.

**Mathematical connection:** the LoRA objective minimises:

$$\min_{A, B} \mathcal{L}(W_0 + BA) \approx \min_{A, B} \frac{1}{2}\lVert Y - X(W_0 + BA) \rVert_F^2$$

$$= \min_{A, B} \frac{1}{2}\lVert (Y - XW_0) - XBA \rVert_F^2 = \min_{\Delta W} \frac{1}{2}\lVert R - X\Delta W \rVert_F^2 \quad \text{s.t. } \operatorname{rank}(\Delta W) \leq r$$

where $R = Y - XW_0$ is the residual after the pretrained model. **LoRA is low-rank OLS on the residuals of the pretrained model.** The rank constraint is a non-convex substitute for nuclear norm regularisation.

**DoRA (Liu et al., 2024)** further decomposes $W = m \cdot (W_0 + BA)/\lVert W_0 + BA \rVert_{\text{col}}$ where $m$ is a learned magnitude vector — adding a direction/magnitude decomposition to the LoRA update.

**Parameter count:** full fine-tuning needs $dk$ parameters; LoRA needs $r(d+k)$. For $d = k = 4096$, $r = 8$: full $\approx 16.8$M, LoRA $\approx 65.5$K — a $256\times$ reduction.

### 13.3 Probing Classifiers

A **linear probe** trains a logistic regression on frozen representations to test if a concept is linearly encoded:

$$p(c = 1 \mid \mathbf{h}) = \sigma(\mathbf{w}^\top \mathbf{h} + b)$$

where $\mathbf{h} = \phi_\theta(\mathbf{x}) \in \mathbb{R}^d$ is a layer's representation of input $\mathbf{x}$ and $c$ is the probed concept (e.g., syntactic role, sentiment, entity type).

**Interpretation:** if probe accuracy is high, $\mathbf{h}$ contains linearly accessible information about $c$. Low accuracy means the representation does not encode $c$ in a linearly decodable way.

**Causal probing** (Meng et al., 2022) goes further: identify not just that a concept is encoded, but which neurons carry the information and whether patching those neurons causally affects model outputs. This is the core of **mechanistic interpretability** at Anthropic and DeepMind.

**Mutual information probing (ROLA).** More recently, probing is framed as estimating the mutual information $I(\mathbf{h}; c)$, avoiding the confound that a high-accuracy probe could reflect the probe's own learning capacity rather than the representation.

### 13.4 Neural Tangent Kernel

For an infinitely wide neural network with fixed random parameters at initialisation $\boldsymbol{\theta}_0$, the model $f_{\boldsymbol{\theta}}(\mathbf{x})$ at time $t$ of gradient flow training evolves as:

$$\frac{d f_{\boldsymbol{\theta}_t}(\mathbf{x})}{dt} = -\eta \Theta(\mathbf{x}, \cdot)(f_{\boldsymbol{\theta}_t}(\cdot) - y(\cdot))$$

where $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_{\boldsymbol{\theta}} f(\mathbf{x})^\top \nabla_{\boldsymbol{\theta}} f(\mathbf{x}')$ is the **Neural Tangent Kernel** (NTK).

For infinite-width networks, $\Theta$ is approximately **constant** during training (parameters barely move — "lazy training" regime). The solution at time $t$ is:

$$f_{\boldsymbol{\theta}_t}(\mathbf{x}) = \mathbf{k}_*^\top (I - e^{-\eta \Theta t})\Theta^{-1}\mathbf{y}$$

At convergence ($t \to \infty$): $f_{\boldsymbol{\theta}_\infty} = \mathbf{k}_*^\top \Theta^{-1} \mathbf{y}$ — **kernel regression with the NTK**. Finite-width networks deviate from this (feature learning), but the NTK provides an exact characterisation of linearised dynamics.

### 13.5 Attention as Linear Mixing

The output of a single attention head is:

$$\operatorname{Attention}(Q, K, V) = A V, \quad A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)$$

The output at position $t$ is:

$$\mathbf{o}_t = \sum_{s=1}^T A_{ts} \mathbf{v}_s = \sum_{s=1}^T A_{ts} W_V \mathbf{x}_s$$

This is a **weighted linear combination** of value vectors — a linear model with attention-computed weights. The value projection $W_V$ is a standard linear layer (Section 13.1). The output projection $W_O$ post-processes the head outputs with another linear layer.

**Low-rank structure.** The value-output composition $W_V W_O^\top \in \mathbb{R}^{d \times d}$ has rank at most $d_k \ll d$ (the head dimension). Multi-head attention effectively applies $h$ independent low-rank linear projections — the mathematical structure of Group Lasso and LoRA.

**For AI:** The "OV circuit" (Elhage et al., 2021) describes a single attention head by its $W_O W_V$ matrix — which tokens it copies and which it promotes. Heads with near-diagonal $W_O W_V$ are "copy heads"; heads with off-diagonal structure are "induction heads" or "translation heads". All of this analysis uses the linear algebra of matrix factorisation.

---

## 14. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using OLS when $n < d$ | $(X^\top X)^{-1}$ doesn't exist; infinitely many solutions | Use Ridge, Lasso, or pseudoinverse; always check $\operatorname{rank}(X)$ |
| 2 | Forgetting to standardise features before regularisation | Ridge/Lasso penalise all $\beta_j$ equally, so features on different scales are penalised differently | Always centre and scale features to unit variance before regularising |
| 3 | Interpreting $R^2$ as model quality | Adding useless features always increases $R^2$; high $R^2$ on training data means nothing about generalisation | Use adjusted $R^2$, AIC, BIC, or cross-validation error |
| 4 | Using Ridge when sparsity is desired | Ridge shrinks but never zeros out coefficients | Use Lasso or Elastic Net for feature selection |
| 5 | Forgetting the intercept in OLS | Predictions are biased; residuals don't sum to zero | Always include an intercept unless there's a domain-specific reason not to |
| 6 | Treating logistic regression as a regression model | Despite the name, logistic regression is a classification model — the output is a probability, not a numerical prediction | Use the output as $P(y=1|\mathbf{x})$; threshold at 0.5 for predictions |
| 7 | Ignoring multicollinearity | Highly correlated features make $(X^\top X)^{-1}$ ill-conditioned; standard errors explode | Check condition number; use Ridge or VIF analysis |
| 8 | Selecting $\lambda$ on the training set | Optimising $\lambda$ on training data is not penalised — the model memorises noise via $\lambda$ | Always select $\lambda$ via cross-validation on held-out data |
| 9 | Conflating MAP estimate with posterior mean (Bayesian) | MAP is a point estimate; posterior mean minimises MSE; they're equal only for symmetric posteriors | Be explicit: Ridge = MAP with Gaussian prior, not the full posterior |
| 10 | Confusing kernel SVM with Gaussian process regression | SVM uses kernels for decision boundaries (discriminative); GP regression uses kernels for probabilistic predictions (generative/Bayesian) | Both use kernel matrices, but objectives and outputs differ completely |

---

## 15. Exercises

**Exercise 1** ★☆☆ **OLS Geometry**

(a) Show that the residual vector $\mathbf{e} = \mathbf{y} - X\hat{\boldsymbol{\beta}}$ is orthogonal to every column of $X$. Start from the normal equations.

(b) Prove that the hat matrix $H = X(X^\top X)^{-1}X^\top$ is idempotent ($H^2 = H$) and symmetric.

(c) Compute the hat matrix for $X = [\mathbf{1}_n, \mathbf{x}]$ (intercept + one feature) and express the leverage $h_{ii}$ in terms of $x_i$ and $\bar{x}$.

(d) **For AI:** A linear probe is trained on embeddings $\Phi \in \mathbb{R}^{n \times d}$ for a binary label $\mathbf{y}$. Express the fitted values $\hat{\mathbf{y}}$ in terms of $\Phi$ and identify which training examples are "high leverage."

---

**Exercise 2** ★☆☆ **Ridge as MAP Estimation**

(a) Derive the Ridge solution $\hat{\boldsymbol{\beta}}_\lambda = (X^\top X + \lambda I)^{-1}X^\top \mathbf{y}$ starting from MAP estimation with prior $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ and likelihood $\mathbf{y} \mid X,\boldsymbol{\beta} \sim \mathcal{N}(X\boldsymbol{\beta}, \sigma^2 I)$.

(b) Express $\lambda$ in terms of $\sigma^2$ and $\tau^2$. What happens to $\hat{\boldsymbol{\beta}}_\lambda$ as $\tau^2 \to \infty$?

(c) Using SVD $X = U\Sigma V^\top$, show that Ridge shrinks the OLS estimate along each singular direction $\mathbf{v}_j$ by the factor $\sigma_j^2/(\sigma_j^2 + \lambda)$.

(d) Compute the effective degrees of freedom $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda)$ and verify that $\operatorname{df}(0) = d$ and $\operatorname{df}(\infty) = 0$.

---

**Exercise 3** ★★☆ **Lasso Soft-Thresholding**

(a) Derive the soft-thresholding update $\hat{\beta}_j = S_\lambda(z_j) = \operatorname{sign}(z_j)\max(|z_j|-\lambda, 0)$ from the subgradient optimality condition for the Lasso coordinate descent step.

(b) Implement cyclic coordinate descent for Lasso in Python (no external libraries). Verify convergence on a synthetic dataset with $n=100$, $d=10$, true $\boldsymbol{\beta}^*$ sparse.

(c) Show that Lasso MAP estimation corresponds to a Laplace prior $p(\beta_j) \propto e^{-\lambda|\beta_j|}$.

(d) **For AI:** Sparse autoencoders (SAEs) in mechanistic interpretability train $\hat{\mathbf{x}} = W_d \operatorname{ReLU}(W_e \mathbf{x} + \mathbf{b}_e) + \mathbf{b}_d$ with $\ell^1$ penalty on activations. Explain how the ReLU + $\ell^1$ penalty achieves sparsity similarly to the Lasso.

---

**Exercise 4** ★★☆ **Bayesian Predictive Distribution**

(a) Derive the posterior parameters $(\boldsymbol{\mu}_n, \Sigma_n)$ for Bayesian linear regression with prior $\boldsymbol{\beta} \sim \mathcal{N}(\boldsymbol{\mu}_0, \Sigma_0)$.

(b) Show that the predictive distribution $p(y_* \mid \mathbf{x}_*, \mathcal{D}) = \mathcal{N}(\mathbf{x}_*^\top \boldsymbol{\mu}_n, \mathbf{x}_*^\top \Sigma_n \mathbf{x}_* + \sigma^2)$.

(c) Implement Bayesian linear regression and plot posterior predictive intervals for $n = 5, 20, 100$ training points. Observe how epistemic uncertainty decreases with $n$.

(d) Show that the frequentist prediction interval from Section 3.4 matches the Bayesian predictive variance under an uninformative prior $\Sigma_0^{-1} \to 0$.

---

**Exercise 5** ★★☆ **Logistic Regression and Convexity**

(a) Derive the gradient $\nabla_\mathbf{w} \mathcal{L} = \frac{1}{n}X^\top(\mathbf{p} - \mathbf{y})$ and Hessian $H = \frac{1}{n}X^\top D X$ where $D = \operatorname{diag}(p_i(1-p_i))$ for the cross-entropy loss.

(b) Prove that the cross-entropy loss is globally convex in $\mathbf{w}$.

(c) Implement one step of IRLS (Newton-Raphson) for logistic regression and verify against numerical differentiation.

(d) **For AI:** Show that the gradient of the language model cross-entropy loss $-\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t})$ has the same form as the logistic regression gradient when the model is a linear softmax (unembedding only).

---

**Exercise 6** ★★☆ **LDA vs. Logistic Regression**

(a) Derive the LDA decision rule $\delta_k(\mathbf{x}) = \mathbf{x}^\top \hat{\Sigma}^{-1}\hat{\boldsymbol{\mu}}_k - \frac{1}{2}\hat{\boldsymbol{\mu}}_k^\top \hat{\Sigma}^{-1}\hat{\boldsymbol{\mu}}_k + \log\hat{\pi}_k$ from the Gaussian generative model.

(b) Show that for binary classification, LDA's decision boundary is the same as logistic regression with a specific parameterisation. What are the effective logistic regression weights $\mathbf{w}_{\text{LDA}}$?

(c) Generate a dataset where features are correlated and $n$ is small. Compare LDA and logistic regression accuracy. Explain the result using the Ng & Jordan (2001) theory.

(d) Implement LDA dimensionality reduction (Fisher's LDA) and project 2D data onto the discriminant direction. Verify it maximises the Fisher criterion.

---

**Exercise 7** ★★★ **SVM Dual Derivation**

(a) Starting from the hard-margin SVM primal, derive the Lagrangian, compute stationarity conditions, and substitute to obtain the dual maximisation problem.

(b) Show that the optimal weight vector $\hat{\mathbf{w}}$ is a sparse linear combination of training points (the support vectors).

(c) Implement a simple linear SVM using the dual (solving the QP via `scipy.optimize.minimize`) and verify that the margin equals $2/\lVert \hat{\mathbf{w}} \rVert$.

(d) **For AI:** Show that the attention mechanism $\operatorname{softmax}(QK^\top/\sqrt{d_k})$ can be interpreted as a kernel similarity function. Which kernel does the softmax correspond to? How does this relate to kernel SVM?

---

**Exercise 8** ★★★ **Bias-Variance Decomposition**

(a) Derive the bias-variance decomposition $\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$ for any estimator $\hat{f}$.

(b) For Ridge regression with regularisation $\lambda$, derive explicit formulas for Bias$^2(\lambda)$ and Variance$(\lambda)$ as functions of the singular values of $X$.

(c) Simulate the bias-variance tradeoff: vary $\lambda$ from $10^{-3}$ to $10^3$ and plot bias$^2$, variance, and MSE. Verify the U-shape of MSE.

(d) **For AI:** In the overparameterised regime ($d > n$), the minimum-norm interpolator has zero training error. Derive an expression for its bias and variance and show what happens as $d/n \to \infty$.

---

**Exercise 9** ★★★ **LoRA Analysis**

(a) Show that LoRA fine-tuning $W = W_0 + BA$ with $\operatorname{rank}(BA) \leq r$ is equivalent to minimum-rank regression on the residuals $R = Y - XW_0$.

(b) Compute the SVD of $\Delta W^* = BA$ that minimises $\lVert R - X\Delta W \rVert_F^2$ subject to $\operatorname{rank}(\Delta W) = r$ (using the Eckart-Young theorem). How does this compare to nuclear norm regularisation?

(c) For a transformer with $d_{\text{model}} = 4096$, compare parameter counts for: full fine-tuning, LoRA with $r = 4$, LoRA with $r = 64$, and prefix tuning with 10 tokens.

(d) Implement LoRA fine-tuning for a small linear model. Verify that the LoRA update captures the dominant singular directions of the residual.

---

**Exercise 10** ★★★ **Neural Tangent Kernel Regression**

(a) For a 2-layer linear network $f_\theta(\mathbf{x}) = W_2 W_1 \mathbf{x}$ with parameters $\theta = (W_1, W_2)$, compute the NTK $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x})^\top \nabla_\theta f(\mathbf{x}')$.

(b) Show that gradient flow training on MSE loss converges to $f_{\theta_\infty}(\mathbf{x}) = \mathbf{k}_*^\top \Theta^{-1} \mathbf{y}$ — kernel regression with the NTK.

(c) Implement gradient flow for the linear network and compare to the kernel regression prediction numerically.

(d) **For AI:** Explain why wide neural networks in the NTK regime are disadvantageous for feature learning. What property of practical LLMs (finite width, feature learning) allows them to outperform their NTK counterparts?


---

## 16. Why This Matters for AI (2026)

| Concept | AI Application | Specific Example |
|---|---|---|
| OLS / normal equations | Linear probing evaluation | Measuring representation quality in BERT, LLaMA (Tenney et al., 2019) |
| Ridge regression / weight decay | LLM pretraining regularisation | AdamW with weight decay 0.1 in LLaMA 2, GPT-4 |
| Lasso / $\ell^1$ sparsity | Sparse autoencoders (SAEs) | Anthropic SAE interpretability research (Templeton et al., 2024) |
| Nuclear norm / low rank | LoRA, DoRA, GaLore | PEFT methods for fine-tuning 70B+ parameter models |
| Bayesian linear regression | Uncertainty quantification | Conformal prediction for LLM outputs |
| Logistic regression | Linear probing for properties | Syntactic / semantic concept detection in transformer layers |
| Softmax regression | Language model head | Every token prediction in GPT-4, Claude, Gemini |
| LDA / Fisher discriminant | Representation geometry analysis | Linear separability of concepts in embedding space |
| SVM / kernel methods | Attention as kernel computation | Performers, Random Feature Attention (Peng et al., 2021) |
| Bias-variance decomposition | Model complexity / regularisation | Understanding why LLMs generalise despite memorisation |
| Double descent | Overparameterised LLMs | 70B+ models interpolate training but generalise — benign overfitting |
| LOOCV / leverage scores | Training data attribution | Influence functions for LLM output attribution (Koh & Liang, 2017) |
| KKT conditions | Sparse attention patterns | Reformer, Longformer locality constraints |
| NTK regime | Linearised LLM analysis | Lazy training at wide width, feature learning at practical width |
| Coordinate descent | Token-level optimisation | Inference-time compute scheduling in chain-of-thought |

---

## 17. Conceptual Bridge

### Where We Come From

This section builds on the entire mathematics curriculum: **linear algebra** (Chapter 02–03) provides the geometric language — column spaces, projections, eigendecomposition, SVD — that underlies every formula in OLS. **Calculus** (Chapter 04–05) enables the derivation of optimal solutions via gradient conditions. **Probability and statistics** (Chapters 06–07) supply the frequentist inference framework (hypothesis tests, confidence intervals) and the Bayesian framework (prior-posterior, evidence). **Optimisation** (Chapter 08) provides the algorithms that make all these models computable at scale. **Information theory** (Chapter 09) connects cross-entropy loss to maximum likelihood and minimum entropy principles.

Linear models synthesise all of these threads into a unified framework: the OLS formula $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1}X^\top\mathbf{y}$ is simultaneously a geometric projection (linear algebra), a gradient condition (calculus), a maximum likelihood estimator (statistics), a convex minimisation solution (optimisation), and an entropy-maximising classifier (information theory).

### Where We Go Next

From here, the curriculum extends in two directions:

**Towards neural networks (Section 02):** neural networks replace the single linear map with a composition of linear maps and nonlinear activations. The core ideas carry forward exactly: the objective is still a loss function minimised by gradient descent; regularisation is still Ridge/Lasso applied to weight matrices; the output layer is still a linear model (softmax regression). The mathematical novelty is backpropagation — efficient gradient computation through the composition.

**Towards advanced models (Sections 03–14):** each subsequent section takes one linear model idea and makes it richer. Attention mechanisms (Section 03) generalise weighted linear combinations of values; generative models (Section 07) extend Bayesian regression to latent variable models; reinforcement learning (Section 06) replaces supervised labels with delayed rewards but keeps the core machinery of linear function approximation.

### Curriculum Position

```
LINEAR MODELS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  FOUNDATIONS                    THIS SECTION              NEXT
  ─────────────────────────────────────────────────────────────────

  02 Linear Algebra    ──────▶   01 Linear Models   ──────▶  02 NNs
  03 Adv. Linear Alg.            (14-Math-for-               (Section
  04 Calculus                     Specific-Models)            14-02)
  06 Probability         ──────▶ • OLS + projection
  07 Statistics                  • Ridge / Lasso / BLR
  08 Optimisation        ──────▶ • Logistic / Softmax
  09 Information Theory          • SVM + kernels
  13 ML-Specific Math            • Bias-variance
                                 • LoRA / NTK / attention

  ════════════════════════════════════════════════════════════════════

  Every formula in linear models is active in modern LLMs:
   OLS → linear probes  |  Ridge → weight decay  |  Lasso → SAEs
   Logistic → CE loss   |  SVM → attention       |  Bayes → UQ

════════════════════════════════════════════════════════════════════════
```

The linear model framework is not a historical artifact — it is the analytical engine that powers mechanistic interpretability, parameter-efficient fine-tuning, uncertainty quantification, and the theoretical understanding of why LLMs generalise despite their enormous size. Mastering it rigorously is the prerequisite for understanding every subsequent section in this curriculum.

---

## References

1. Tibshirani, R. (1994). "Regression Shrinkage and Selection via the Lasso." *JRSS-B*, 58(1), 267–288.
2. Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55–67.
3. Zou, H. & Hastie, T. (2003). "Regularization and Variable Selection via the Elastic Net." *JRSS-B*, 67(2), 301–320.
4. Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). "Least Angle Regression." *Annals of Statistics*, 32(2), 407–499.
5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
6. Ng, A. Y. & Jordan, M. I. (2001). "On Discriminative vs. Generative Classifiers." *NeurIPS* 14.
7. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
8. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
9. Jacot, A., Gabriel, F., & Hongler, C. (2018). "Neural Tangent Kernel." *NeurIPS* 31.
10. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). "Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-Off." *PNAS*, 116(32).
11. Hu, E., Shen, Y., Wallis, P., et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*.
12. Koh, P. W. & Liang, P. (2017). "Understanding Black-box Predictions via Influence Functions." *ICML*.
13. Tenney, I., Das, D., & Pavlick, E. (2019). "BERT Rediscovers the Classical NLP Pipeline." *ACL*.
14. Zhao, R., Li, G., Zhang, Y., et al. (2024). "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection." *ICML*.
15. Elhage, N., Nanda, N., Olsson, C., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Transformer Circuits Thread*, Anthropic.
