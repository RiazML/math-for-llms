# Linear Models: Mathematical Foundations

[← Previous: ML-Specific Math](../../13-ML-Specific-Math) | [Next: Neural Networks →](../02-Neural-Networks)

## Overview

Linear models form the foundation of machine learning, providing interpretable, efficient, and theoretically well-understood methods for regression and classification. Despite their simplicity, they remain essential building blocks and baselines.

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Why This Matters for Machine Learning

Linear models are far more than a stepping stone to deep learning — they are the mathematical lens through which we understand what every neural network layer actually computes. The ordinary least squares (OLS) solution reveals that regression is fundamentally a geometric projection: the model finds the point in the column space of the feature matrix closest to the target vector. This geometric intuition carries directly into understanding embeddings, attention outputs, and residual connections in modern architectures.

Regularization techniques like Ridge and Lasso are not just engineering tricks; they have deep Bayesian interpretations. Ridge regression is equivalent to placing a Gaussian prior on the weights, while Lasso corresponds to a Laplace prior that encourages sparsity. Recognizing this duality connects frequentist optimization objectives to probabilistic inference, a bridge that recurs throughout machine learning — from weight decay in neural networks to sparsity-inducing priors in compressed sensing.

The bias-variance tradeoff, most clearly analyzed in linear models, establishes the fundamental tension in all of machine learning: a model must be complex enough to capture signal but simple enough to avoid fitting noise. Every choice of regularization strength, network depth, or ensemble size is an implicit negotiation along this tradeoff curve.

## Chapter Roadmap

- **Linear Regression**: OLS derivation, closed-form solutions, and the geometric interpretation of projections
- **Regularized Models**: Ridge (L2), Lasso (L1), and Elastic Net — shrinkage, sparsity, and the Bayesian connection
- **Bayesian Linear Regression**: Prior-posterior framework, predictive distributions, and uncertainty quantification
- **Linear Classification**: Logistic regression, softmax, cross-entropy, and decision boundaries
- **Discriminative vs Generative**: LDA, QDA, and the modeling choice between $p(y|x)$ and $p(x,y)$
- **Support Vector Machines**: Margin maximization, the kernel trick, and duality
- **Model Selection**: Bias-variance decomposition, information criteria, and cross-validation
- **Statistical Properties**: Hypothesis testing, confidence intervals, and OLS assumptions
- **Optimization Methods**: Gradient descent, coordinate descent, and Newton's method
- **Deep Learning Connections**: Linear layers, probing classifiers, and the neural tangent kernel

## 1. Linear Regression

### Ordinary Least Squares (OLS)

**Model**: $y = X\beta + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$

**Objective**:
$$\min_\beta \|y - X\beta\|_2^2$$

**Closed-form solution**:
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

**Properties**:

- BLUE (Best Linear Unbiased Estimator) under Gauss-Markov conditions
- Maximum likelihood under Gaussian noise
- Variance: $\text{Var}(\hat{\beta}) = \sigma^2 (X^T X)^{-1}$

### Geometric Interpretation

- Solution projects $y$ onto column space of $X$
- Residuals orthogonal to column space: $X^T(y - X\hat{\beta}) = 0$
- Hat matrix: $H = X(X^T X)^{-1}X^T$, $\hat{y} = Hy$

> 💡 **Insight:** The OLS solution is a *projection*. Imagine $y$ as a vector in $n$-dimensional space and the columns of $X$ as spanning a lower-dimensional subspace. The predicted $\hat{y} = Hy$ is the shadow of $y$ cast onto that subspace, and the residuals point straight "up" — perpendicular to every feature. This geometric picture explains why adding correlated features barely changes the fit (they don't expand the subspace much) and why orthogonal features are ideal.

## 2. Regularized Linear Models

### Ridge Regression (L2)

**Objective**:
$$\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2$$

**Solution**:
$$\hat{\beta}_\text{ridge} = (X^T X + \lambda I)^{-1} X^T y$$

**Properties**:

- Shrinks coefficients toward zero
- Always invertible (handles multicollinearity)
- Equivalent to Bayesian regression with Gaussian prior
- Bias-variance trade-off controlled by $\lambda$

### Lasso (L1)

**Objective**:
$$\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$

**Properties**:

- Induces sparsity (automatic feature selection)
- No closed-form solution (requires optimization)
- KKT conditions: $X^T(y - X\hat{\beta}) = n\lambda \cdot \text{sign}(\hat{\beta})$

### Elastic Net

**Objective**:
$$\min_\beta \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2$$

**Benefits**: Combines sparsity of Lasso with grouping effect of Ridge.

> 💡 **Insight:** Regularization *is* a Bayesian prior in disguise. Ridge regression ($\lambda\|\beta\|_2^2$) is algebraically identical to maximum a posteriori estimation with a Gaussian prior $\beta \sim \mathcal{N}(0, \sigma^2/\lambda \cdot I)$, while Lasso corresponds to a Laplace prior. This means every time you tune a regularization hyperparameter, you are implicitly choosing how strongly you believe the weights should be small — a belief expressed in the language of probability distributions.

## 3. Bayesian Linear Regression

### Prior-Posterior Framework

**Prior**: $\beta \sim \mathcal{N}(\mu_0, \Sigma_0)$

**Likelihood**: $y | X, \beta \sim \mathcal{N}(X\beta, \sigma^2 I)$

**Posterior**:
$$\beta | y \sim \mathcal{N}(\mu_n, \Sigma_n)$$

where:
$$\Sigma_n = \left(\Sigma_0^{-1} + \frac{1}{\sigma^2}X^T X\right)^{-1}$$
$$\mu_n = \Sigma_n \left(\Sigma_0^{-1}\mu_0 + \frac{1}{\sigma^2}X^T y\right)$$

### Predictive Distribution

$$p(y_* | x_*, y, X) = \mathcal{N}(x_*^T \mu_n, x_*^T \Sigma_n x_* + \sigma^2)$$

Captures both epistemic (model) and aleatoric (noise) uncertainty.

## 4. Linear Classification

### Logistic Regression

**Model**: $p(y=1|x) = \sigma(w^T x + b)$ where $\sigma(z) = \frac{1}{1+e^{-z}}$

**Log-likelihood**:
$$\ell(w,b) = \sum_{i=1}^n \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

**Cross-entropy loss**:
$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]$$

### Gradient and Hessian

**Gradient**:
$$\nabla_w \mathcal{L} = \frac{1}{n} X^T(p - y)$$

**Hessian**:
$$H = \frac{1}{n} X^T \text{diag}(p \odot (1-p)) X$$

Convex optimization → Newton-Raphson or L-BFGS.

### Softmax Regression (Multiclass)

**Model**: $p(y=k|x) = \frac{\exp(w_k^T x)}{\sum_{j=1}^K \exp(w_j^T x)}$

**Cross-entropy loss**:
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log p_{ik}$$

## 5. Discriminative vs Generative

### Linear Discriminant Analysis (LDA)

**Assumption**: $p(x|y=k) = \mathcal{N}(\mu_k, \Sigma)$ (shared covariance)

**Classification rule**:
$$\hat{y} = \arg\max_k \left( x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log \pi_k \right)$$

**Decision boundary**: Linear in $x$

### Quadratic Discriminant Analysis (QDA)

**Assumption**: $p(x|y=k) = \mathcal{N}(\mu_k, \Sigma_k)$ (class-specific covariance)

**Decision boundary**: Quadratic in $x$

## 6. Support Vector Machines

### Hard Margin SVM

**Primal**:
$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t. } y_i(w^T x_i + b) \geq 1$$

**Dual**:
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^T x_j$$
$$\text{s.t. } \alpha_i \geq 0, \sum_i \alpha_i y_i = 0$$

### Soft Margin SVM

**Primal**:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$
$$\text{s.t. } y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Hinge loss equivalent**:
$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_i \max(0, 1 - y_i(w^T x_i + b))$$

### Kernel SVM

Replace $x_i^T x_j$ with $k(x_i, x_j)$:

**Common kernels**:

- Linear: $k(x,z) = x^T z$
- Polynomial: $k(x,z) = (x^T z + c)^d$
- RBF: $k(x,z) = \exp(-\gamma \|x-z\|^2)$

## 7. Model Selection and Diagnostics

### Bias-Variance Decomposition

$$\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

> 💡 **Insight:** The bias-variance decomposition reveals that prediction error has an irreducible floor (noise), and the remaining error is split between a model being systematically wrong (bias) and being overly sensitive to the training sample (variance). Increasing $\lambda$ in regularized models shifts the balance: more regularization raises bias but lowers variance. The sweet spot — minimum total error — is the entire motivation for techniques like cross-validation.

### Information Criteria

**AIC**: $\text{AIC} = -2\ell + 2k$

**BIC**: $\text{BIC} = -2\ell + k\log n$

### Cross-Validation

**k-fold CV**:
$$\text{CV}_k = \frac{1}{k} \sum_{i=1}^k \text{Error}(\text{fold}_i)$$

**Leave-one-out** (special case):
$$\text{LOOCV} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_{-i})^2$$

For linear regression: LOOCV = $\frac{1}{n}\sum_i \left(\frac{y_i - \hat{y}_i}{1-h_{ii}}\right)^2$

## 8. Statistical Properties

### Hypothesis Testing

**t-test for coefficients**:
$$t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p-1}$$

**F-test for model comparison**:
$$F = \frac{(\text{RSS}_0 - \text{RSS}_1)/(p_1 - p_0)}{\text{RSS}_1/(n-p_1-1)}$$

### Confidence Intervals

$$\hat{\beta}_j \pm t_{\alpha/2, n-p-1} \cdot \text{SE}(\hat{\beta}_j)$$

### Assumptions (OLS)

1. **Linearity**: $\mathbb{E}[y|X] = X\beta$
2. **Independence**: Observations independent
3. **Homoscedasticity**: $\text{Var}(\epsilon_i) = \sigma^2$
4. **Normality**: $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (for inference)
5. **No multicollinearity**: $X^T X$ full rank

## 9. Optimization Methods

### Gradient Descent

$$\beta^{(t+1)} = \beta^{(t)} - \eta \nabla \mathcal{L}(\beta^{(t)})$$

### Coordinate Descent (for Lasso)

Update one coordinate at a time:
$$\beta_j = S_\lambda\left(\frac{1}{n}\sum_i x_{ij}(y_i - \hat{y}_i^{(-j)})\right)$$

where $S_\lambda(z) = \text{sign}(z)\max(|z|-\lambda, 0)$ is soft thresholding.

### Newton's Method (for Logistic)

$$\beta^{(t+1)} = \beta^{(t)} - H^{-1} \nabla \mathcal{L}$$

## 10. Applications in Deep Learning

### Linear Layers

Neural network linear layer: $h = Wx + b$

- Foundation of feed-forward networks
- Attention mechanism: linear projections Q, K, V

### Feature Extraction

- Use pre-trained features + linear classifier
- Probing classifiers for representation analysis

### Linearized Neural Networks

- Neural Tangent Kernel regime
- Infinite-width limits behave like kernel regression

## ML Connections

| Linear Concept      | Deep Learning Application                 |
| ------------------- | ----------------------------------------- |
| OLS                 | Training final layer with frozen features |
| Ridge               | Weight decay regularization               |
| Logistic            | Binary cross-entropy loss                 |
| Softmax             | Multi-class classification                |
| SVM margins         | Max-margin losses in embeddings           |
| Bayesian regression | Uncertainty quantification                |
| Feature selection   | Pruning, sparsity                         |

## Key Equations Summary

| Model    | Solution / Objective                                   |
| -------- | ------------------------------------------------------ | -------------- | --------------- |
| OLS      | $\hat{\beta} = (X^TX)^{-1}X^Ty$                        |
| Ridge    | $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$            |
| Lasso    | $\min \frac{1}{2n}\|y-X\beta\|^2 + \lambda\|\beta\|_1$ |
| Logistic | $p = \sigma(w^Tx + b)$, cross-entropy loss             |
| SVM      | $\min \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$              |
| Bayesian | $p(\beta                                               | y) \propto p(y | \beta)p(\beta)$ |

## Key Takeaways

- The OLS solution $\hat{\beta} = (X^TX)^{-1}X^Ty$ is a geometric projection of the target vector onto the column space of the design matrix — residuals are always orthogonal to the features.
- Ridge regression adds $\lambda I$ to $X^TX$, guaranteeing invertibility and shrinking coefficients; it is mathematically equivalent to a Gaussian prior on the weights.
- Lasso uses an L1 penalty that drives coefficients exactly to zero, performing automatic feature selection — useful when true models are sparse.
- The bias-variance tradeoff governs all model selection: more complexity reduces bias but increases variance, and the regularization strength $\lambda$ controls this balance.
- Logistic regression applies the sigmoid to a linear model, yielding convex cross-entropy optimization — the Hessian is always positive semi-definite.
- SVMs maximize the margin between classes; through the kernel trick, they implicitly map data to high-dimensional spaces where linear separation is possible.
- Bayesian linear regression provides full posterior distributions over predictions, naturally quantifying uncertainty with no additional machinery.

## Exercises

1. **OLS Geometry**: Show that the OLS residual vector $e = y - X\hat{\beta}$ is orthogonal to every column of $X$. Start from the normal equations and interpret the result geometrically as a projection.

2. **Ridge as Bayesian MAP**: Derive the Ridge regression solution $\hat{\beta}_{\text{ridge}} = (X^TX + \lambda I)^{-1}X^Ty$ starting from the Bayesian posterior with a Gaussian prior $\beta \sim \mathcal{N}(0, \tau^2 I)$ and Gaussian likelihood. Express $\lambda$ in terms of $\sigma^2$ and $\tau^2$.

3. **Bias-Variance for Ridge**: For a fixed design matrix $X$ and true model $y = X\beta^* + \epsilon$, derive expressions for the bias and variance of the Ridge estimator as functions of $\lambda$. Show that bias increases and variance decreases with $\lambda$.

4. **Lasso Soft Thresholding**: Prove that the coordinate descent update for a single Lasso coefficient is the soft-thresholding operator $S_\lambda(z) = \text{sign}(z)\max(|z| - \lambda, 0)$. Hint: take the subgradient of the Lasso objective with respect to $\beta_j$.

5. **Kernel SVM Duality**: Starting from the soft-margin SVM primal, derive the dual formulation using Lagrange multipliers. Show that the kernel trick allows replacing all inner products $x_i^Tx_j$ with $k(x_i, x_j)$ without ever computing the feature map explicitly.

## References

1. Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
4. Boyd & Vandenberghe - "Convex Optimization"
