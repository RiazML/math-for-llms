# Linear Models: Mathematical Foundations

## Overview

Linear models form the foundation of machine learning, providing interpretable, efficient, and theoretically well-understood methods for regression and classification. Despite their simplicity, they remain essential building blocks and baselines.

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

## References

1. Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
4. Boyd & Vandenberghe - "Convex Optimization"
