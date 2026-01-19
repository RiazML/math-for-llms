"""
Linear Models - Examples
========================

Comprehensive examples of linear models: regression, classification, and regularization.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict
from scipy import linalg


# =============================================================================
# Example 1: Ordinary Least Squares Regression
# =============================================================================

def example_ols_regression():
    """
    Ordinary Least Squares with closed-form solution.
    Shows geometric interpretation and statistical properties.
    """
    print("=" * 70)
    print("Example 1: Ordinary Least Squares Regression")
    print("=" * 70)
    
    class OLSRegression:
        """OLS Linear Regression with statistical diagnostics."""
        
        def __init__(self, fit_intercept: bool = True):
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
            self.sigma2_ = None
            self.cov_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'OLSRegression':
            n, p = X.shape
            
            if self.fit_intercept:
                X_aug = np.column_stack([np.ones(n), X])
            else:
                X_aug = X
            
            # Normal equations: (X'X)^{-1} X'y
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y
            
            # Solve using Cholesky for numerical stability
            L = np.linalg.cholesky(XtX)
            beta = linalg.cho_solve((L, True), Xty)
            
            if self.fit_intercept:
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
            else:
                self.intercept_ = 0.0
                self.coef_ = beta
            
            # Residuals and variance estimate
            residuals = y - X_aug @ beta
            df = n - len(beta)
            self.sigma2_ = np.sum(residuals**2) / df
            
            # Covariance of coefficients
            self.cov_ = self.sigma2_ * np.linalg.inv(XtX)
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
        
        def standard_errors(self) -> np.ndarray:
            """Standard errors of coefficients."""
            return np.sqrt(np.diag(self.cov_)[1:] if self.fit_intercept else np.diag(self.cov_))
        
        def t_statistics(self) -> np.ndarray:
            """t-statistics for coefficient significance."""
            return self.coef_ / self.standard_errors()
        
        def r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
            """Coefficient of determination."""
            y_pred = self.predict(X)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - ss_res / ss_tot
    
    # Generate data
    np.random.seed(42)
    n = 100
    p = 3
    
    X = np.random.randn(n, p)
    true_coef = np.array([2.0, -1.5, 0.5])
    true_intercept = 1.0
    noise_std = 0.5
    
    y = X @ true_coef + true_intercept + noise_std * np.random.randn(n)
    
    # Fit model
    model = OLSRegression()
    model.fit(X, y)
    
    print(f"\nTrue coefficients: {true_coef}")
    print(f"Estimated coefficients: {model.coef_.round(4)}")
    print(f"True intercept: {true_intercept}")
    print(f"Estimated intercept: {model.intercept_:.4f}")
    
    print(f"\nEstimated noise variance: {model.sigma2_:.4f} (true: {noise_std**2})")
    print(f"R²: {model.r_squared(X, y):.4f}")
    
    print(f"\nStatistical inference:")
    print(f"  Standard errors: {model.standard_errors().round(4)}")
    print(f"  t-statistics: {model.t_statistics().round(4)}")


# =============================================================================
# Example 2: Ridge Regression
# =============================================================================

def example_ridge_regression():
    """
    Ridge regression with L2 regularization.
    Shows regularization path and bias-variance trade-off.
    """
    print("\n" + "=" * 70)
    print("Example 2: Ridge Regression")
    print("=" * 70)
    
    class RidgeRegression:
        """Ridge Regression with closed-form solution."""
        
        def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
            self.alpha = alpha
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegression':
            n, p = X.shape
            
            if self.fit_intercept:
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)
                X_centered = X - X_mean
                y_centered = y - y_mean
            else:
                X_centered = X
                y_centered = y
            
            # Ridge solution: (X'X + αI)^{-1} X'y
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            
            self.coef_ = np.linalg.solve(XtX + self.alpha * np.eye(p), Xty)
            
            if self.fit_intercept:
                self.intercept_ = y_mean - X_mean @ self.coef_
            else:
                self.intercept_ = 0.0
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Generate data with multicollinearity
    np.random.seed(42)
    n = 50
    p = 10
    
    X = np.random.randn(n, p)
    # Add correlated features
    X[:, 5:] = X[:, :5] + 0.1 * np.random.randn(n, 5)
    
    true_coef = np.random.randn(p)
    y = X @ true_coef + 0.5 * np.random.randn(n)
    
    # Regularization path
    alphas = np.logspace(-3, 3, 50)
    coef_path = np.zeros((len(alphas), p))
    
    for i, alpha in enumerate(alphas):
        model = RidgeRegression(alpha=alpha)
        model.fit(X, y)
        coef_path[i] = model.coef_
    
    print(f"\nRegularization path analysis:")
    print(f"  Number of features: {p}")
    print(f"  Log α range: [{np.log10(alphas[0]):.1f}, {np.log10(alphas[-1]):.1f}]")
    
    # Compare specific values
    print(f"\nCoefficient norms at different α:")
    for alpha in [0.001, 0.1, 1.0, 10.0, 100.0]:
        model = RidgeRegression(alpha=alpha)
        model.fit(X, y)
        print(f"  α={alpha:6.3f}: ||β||₂ = {np.linalg.norm(model.coef_):.4f}")
    
    # Cross-validation for optimal alpha
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_errors = np.zeros(len(alphas))
    
    for i, alpha in enumerate(alphas):
        fold_errors = []
        for train_idx, val_idx in kf.split(X):
            model = RidgeRegression(alpha=alpha)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            fold_errors.append(np.mean((y[val_idx] - pred)**2))
        cv_errors[i] = np.mean(fold_errors)
    
    best_idx = np.argmin(cv_errors)
    print(f"\nCross-validation:")
    print(f"  Best α: {alphas[best_idx]:.4f}")
    print(f"  Best CV MSE: {cv_errors[best_idx]:.4f}")


# =============================================================================
# Example 3: Lasso Regression (Coordinate Descent)
# =============================================================================

def example_lasso_regression():
    """
    Lasso regression with L1 regularization using coordinate descent.
    Shows sparsity-inducing property.
    """
    print("\n" + "=" * 70)
    print("Example 3: Lasso Regression")
    print("=" * 70)
    
    class LassoRegression:
        """Lasso Regression via coordinate descent."""
        
        def __init__(self, alpha: float = 1.0, max_iter: int = 1000,
                     tol: float = 1e-4, fit_intercept: bool = True):
            self.alpha = alpha
            self.max_iter = max_iter
            self.tol = tol
            self.fit_intercept = fit_intercept
            self.coef_ = None
            self.intercept_ = None
        
        def _soft_threshold(self, z: float, threshold: float) -> float:
            """Soft thresholding operator."""
            if z > threshold:
                return z - threshold
            elif z < -threshold:
                return z + threshold
            return 0.0
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegression':
            n, p = X.shape
            
            if self.fit_intercept:
                X_mean = np.mean(X, axis=0)
                y_mean = np.mean(y)
                X = X - X_mean
                y = y - y_mean
            
            # Initialize
            self.coef_ = np.zeros(p)
            
            # Precompute X'X diagonal
            X_col_norms_sq = np.sum(X**2, axis=0)
            
            for iteration in range(self.max_iter):
                coef_old = self.coef_.copy()
                
                # Coordinate descent
                for j in range(p):
                    # Compute residual excluding feature j
                    r_j = y - X @ self.coef_ + X[:, j] * self.coef_[j]
                    
                    # Correlation with residual
                    rho_j = X[:, j] @ r_j
                    
                    # Soft thresholding update
                    if X_col_norms_sq[j] > 0:
                        self.coef_[j] = self._soft_threshold(
                            rho_j / X_col_norms_sq[j],
                            n * self.alpha / X_col_norms_sq[j]
                        )
                
                # Check convergence
                if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                    break
            
            if self.fit_intercept:
                self.intercept_ = y_mean - X_mean @ self.coef_
            else:
                self.intercept_ = 0.0
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Generate sparse data
    np.random.seed(42)
    n = 100
    p = 20
    
    X = np.random.randn(n, p)
    
    # True model is sparse: only 5 non-zero coefficients
    true_coef = np.zeros(p)
    true_coef[[0, 3, 7, 12, 18]] = [3.0, -2.0, 1.5, -1.0, 2.5]
    
    y = X @ true_coef + 0.5 * np.random.randn(n)
    
    print(f"\nTrue sparse coefficients (5 non-zero out of {p}):")
    print(f"  Non-zero indices: {np.where(true_coef != 0)[0]}")
    print(f"  Non-zero values: {true_coef[true_coef != 0]}")
    
    # Fit Lasso at different regularization strengths
    print(f"\nLasso regularization path:")
    for alpha in [0.001, 0.01, 0.1, 0.5, 1.0]:
        model = LassoRegression(alpha=alpha)
        model.fit(X, y)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-6)
        mse = np.mean((X @ model.coef_ + model.intercept_ - y)**2)
        print(f"  α={alpha:.3f}: {n_nonzero:2d} non-zero coefs, MSE={mse:.4f}")
    
    # Best model
    model = LassoRegression(alpha=0.05)
    model.fit(X, y)
    
    print(f"\nOptimal model (α=0.05):")
    nonzero_idx = np.where(np.abs(model.coef_) > 1e-6)[0]
    print(f"  Selected features: {nonzero_idx}")
    print(f"  Estimated coefs: {model.coef_[nonzero_idx].round(3)}")


# =============================================================================
# Example 4: Logistic Regression
# =============================================================================

def example_logistic_regression():
    """
    Logistic regression for binary classification.
    Gradient descent and Newton's method implementations.
    """
    print("\n" + "=" * 70)
    print("Example 4: Logistic Regression")
    print("=" * 70)
    
    class LogisticRegression:
        """Logistic Regression with Newton's method."""
        
        def __init__(self, max_iter: int = 100, tol: float = 1e-6,
                     l2_reg: float = 0.0):
            self.max_iter = max_iter
            self.tol = tol
            self.l2_reg = l2_reg
            self.coef_ = None
            self.intercept_ = None
            self.loss_history_ = []
        
        def _sigmoid(self, z: np.ndarray) -> np.ndarray:
            # Numerically stable sigmoid
            return np.where(z >= 0,
                            1 / (1 + np.exp(-z)),
                            np.exp(z) / (1 + np.exp(z)))
        
        def _loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
            z = X @ w
            loss = -np.mean(y * z - np.logaddexp(0, z))
            if self.l2_reg > 0:
                loss += 0.5 * self.l2_reg * np.sum(w[1:]**2)
            return loss
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
            n, p = X.shape
            X_aug = np.column_stack([np.ones(n), X])
            
            # Initialize
            w = np.zeros(p + 1)
            
            for iteration in range(self.max_iter):
                # Predictions
                p_hat = self._sigmoid(X_aug @ w)
                
                # Gradient
                grad = X_aug.T @ (p_hat - y) / n
                if self.l2_reg > 0:
                    grad[1:] += self.l2_reg * w[1:]
                
                # Hessian
                S = np.diag(p_hat * (1 - p_hat))
                H = X_aug.T @ S @ X_aug / n
                if self.l2_reg > 0:
                    H[1:, 1:] += self.l2_reg * np.eye(p)
                
                # Newton update
                try:
                    delta = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    delta = grad  # Fall back to gradient descent
                
                w_new = w - delta
                
                # Record loss
                loss = self._loss(X, y, w_new)
                self.loss_history_.append(loss)
                
                # Check convergence
                if np.max(np.abs(w_new - w)) < self.tol:
                    w = w_new
                    break
                
                w = w_new
            
            self.intercept_ = w[0]
            self.coef_ = w[1:]
            
            return self
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            z = X @ self.coef_ + self.intercept_
            p = self._sigmoid(z)
            return np.column_stack([1 - p, p])
        
        def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
    
    # Generate data
    np.random.seed(42)
    n = 200
    
    # Two classes
    X0 = np.random.randn(n // 2, 2) + np.array([-1, -1])
    X1 = np.random.randn(n // 2, 2) + np.array([1, 1])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    # Shuffle
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    
    # Fit
    model = LogisticRegression(l2_reg=0.01)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"\nCoefficients: {model.coef_.round(4)}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Training accuracy: {accuracy:.2%}")
    print(f"Converged in {len(model.loss_history_)} iterations")
    
    # Decision boundary equation
    print(f"\nDecision boundary: {model.coef_[0]:.2f}x₁ + {model.coef_[1]:.2f}x₂ + {model.intercept_:.2f} = 0")


# =============================================================================
# Example 5: Softmax Regression (Multiclass)
# =============================================================================

def example_softmax_regression():
    """
    Softmax regression for multiclass classification.
    """
    print("\n" + "=" * 70)
    print("Example 5: Softmax Regression")
    print("=" * 70)
    
    class SoftmaxRegression:
        """Multiclass logistic regression."""
        
        def __init__(self, n_classes: int, learning_rate: float = 0.1,
                     max_iter: int = 1000, l2_reg: float = 0.01):
            self.n_classes = n_classes
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.l2_reg = l2_reg
            self.W = None
            self.b = None
        
        def _softmax(self, z: np.ndarray) -> np.ndarray:
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftmaxRegression':
            n, p = X.shape
            
            # One-hot encode
            Y = np.eye(self.n_classes)[y]
            
            # Initialize
            self.W = np.random.randn(p, self.n_classes) * 0.01
            self.b = np.zeros(self.n_classes)
            
            for iteration in range(self.max_iter):
                # Forward
                z = X @ self.W + self.b
                probs = self._softmax(z)
                
                # Gradient
                grad_z = (probs - Y) / n
                grad_W = X.T @ grad_z + self.l2_reg * self.W
                grad_b = np.sum(grad_z, axis=0)
                
                # Update
                self.W -= self.learning_rate * grad_W
                self.b -= self.learning_rate * grad_b
            
            return self
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            z = X @ self.W + self.b
            return self._softmax(z)
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.argmax(self.predict_proba(X), axis=1)
    
    # Generate multiclass data
    np.random.seed(42)
    n_per_class = 100
    n_classes = 4
    
    X_list = []
    y_list = []
    centers = np.array([[0, 0], [0, 3], [3, 0], [3, 3]])
    
    for k in range(n_classes):
        X_k = np.random.randn(n_per_class, 2) * 0.5 + centers[k]
        X_list.append(X_k)
        y_list.extend([k] * n_per_class)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # Fit
    model = SoftmaxRegression(n_classes=n_classes, learning_rate=0.5)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"\nNumber of classes: {n_classes}")
    print(f"Training accuracy: {accuracy:.2%}")
    print(f"\nWeight matrix shape: {model.W.shape}")
    print(f"Per-class accuracy:")
    for k in range(n_classes):
        mask = y == k
        acc_k = np.mean(y_pred[mask] == y[mask])
        print(f"  Class {k}: {acc_k:.2%}")


# =============================================================================
# Example 6: Bayesian Linear Regression
# =============================================================================

def example_bayesian_linear_regression():
    """
    Bayesian linear regression with uncertainty quantification.
    """
    print("\n" + "=" * 70)
    print("Example 6: Bayesian Linear Regression")
    print("=" * 70)
    
    class BayesianLinearRegression:
        """Bayesian regression with conjugate prior."""
        
        def __init__(self, prior_mean: Optional[np.ndarray] = None,
                     prior_cov: Optional[np.ndarray] = None,
                     noise_var: float = 1.0):
            self.prior_mean = prior_mean
            self.prior_cov = prior_cov
            self.noise_var = noise_var
            self.posterior_mean = None
            self.posterior_cov = None
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
            n, p = X.shape
            
            # Default prior: N(0, I)
            if self.prior_mean is None:
                self.prior_mean = np.zeros(p)
            if self.prior_cov is None:
                self.prior_cov = np.eye(p)
            
            # Posterior computation
            prior_prec = np.linalg.inv(self.prior_cov)
            
            self.posterior_cov = np.linalg.inv(
                prior_prec + X.T @ X / self.noise_var
            )
            
            self.posterior_mean = self.posterior_cov @ (
                prior_prec @ self.prior_mean + X.T @ y / self.noise_var
            )
            
            return self
        
        def predict(self, X: np.ndarray, return_std: bool = False):
            """Predictive distribution."""
            mean = X @ self.posterior_mean
            
            if return_std:
                # Predictive variance: epistemic + aleatoric
                var = np.array([x @ self.posterior_cov @ x for x in X]) + self.noise_var
                return mean, np.sqrt(var)
            
            return mean
        
        def sample_posterior(self, n_samples: int = 100) -> np.ndarray:
            """Draw samples from posterior."""
            return np.random.multivariate_normal(
                self.posterior_mean, self.posterior_cov, n_samples
            )
    
    # Generate data
    np.random.seed(42)
    n = 20
    
    X = np.sort(np.random.uniform(-3, 3, n))[:, np.newaxis]
    true_coef = np.array([2.0])
    noise_std = 0.5
    y = X.flatten() * true_coef[0] + noise_std * np.random.randn(n)
    
    # Fit
    model = BayesianLinearRegression(noise_var=noise_std**2)
    model.fit(X, y)
    
    print(f"\nTrue coefficient: {true_coef[0]}")
    print(f"Posterior mean: {model.posterior_mean[0]:.4f}")
    print(f"Posterior std: {np.sqrt(model.posterior_cov[0, 0]):.4f}")
    
    # Predictions with uncertainty
    X_test = np.linspace(-4, 4, 50)[:, np.newaxis]
    y_pred, y_std = model.predict(X_test, return_std=True)
    
    print(f"\nPredictive uncertainty at different x:")
    for x in [-3, 0, 3]:
        y_p, y_s = model.predict(np.array([[x]]), return_std=True)
        print(f"  x={x:2d}: pred={y_p[0]:.2f} ± {2*y_s[0]:.2f}")
    
    # Posterior samples
    beta_samples = model.sample_posterior(1000)
    print(f"\nPosterior 95% credible interval: [{np.percentile(beta_samples, 2.5):.3f}, {np.percentile(beta_samples, 97.5):.3f}]")


# =============================================================================
# Example 7: Linear Discriminant Analysis
# =============================================================================

def example_lda():
    """
    Linear Discriminant Analysis for classification.
    """
    print("\n" + "=" * 70)
    print("Example 7: Linear Discriminant Analysis")
    print("=" * 70)
    
    class LDA:
        """Linear Discriminant Analysis classifier."""
        
        def __init__(self):
            self.class_means_ = None
            self.shared_cov_ = None
            self.priors_ = None
            self.classes_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'LDA':
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            n, p = X.shape
            
            # Class priors
            self.priors_ = np.array([np.mean(y == k) for k in self.classes_])
            
            # Class means
            self.class_means_ = np.array([
                np.mean(X[y == k], axis=0) for k in self.classes_
            ])
            
            # Shared (pooled) covariance
            self.shared_cov_ = np.zeros((p, p))
            for k in self.classes_:
                X_k = X[y == k]
                self.shared_cov_ += (X_k - self.class_means_[k]).T @ (X_k - self.class_means_[k])
            self.shared_cov_ /= (n - n_classes)
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            scores = self._discriminant_scores(X)
            return self.classes_[np.argmax(scores, axis=1)]
        
        def _discriminant_scores(self, X: np.ndarray) -> np.ndarray:
            """Compute linear discriminant scores."""
            cov_inv = np.linalg.inv(self.shared_cov_)
            scores = np.zeros((len(X), len(self.classes_)))
            
            for k, (mu_k, pi_k) in enumerate(zip(self.class_means_, self.priors_)):
                scores[:, k] = (X @ cov_inv @ mu_k 
                               - 0.5 * mu_k @ cov_inv @ mu_k 
                               + np.log(pi_k))
            
            return scores
    
    # Generate data
    np.random.seed(42)
    n_per_class = 100
    
    # Two classes with different means, same covariance
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    mu0 = np.array([-1, -1])
    mu1 = np.array([1, 1])
    
    X0 = np.random.multivariate_normal(mu0, cov, n_per_class)
    X1 = np.random.multivariate_normal(mu1, cov, n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    # Fit
    model = LDA()
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"\nTrue class means:")
    print(f"  Class 0: {mu0}")
    print(f"  Class 1: {mu1}")
    
    print(f"\nEstimated class means:")
    print(f"  Class 0: {model.class_means_[0].round(4)}")
    print(f"  Class 1: {model.class_means_[1].round(4)}")
    
    print(f"\nEstimated shared covariance:")
    print(model.shared_cov_.round(4))
    
    print(f"\nTraining accuracy: {accuracy:.2%}")


# =============================================================================
# Example 8: Support Vector Machine
# =============================================================================

def example_svm():
    """
    Support Vector Machine with soft margin.
    Solved via gradient descent on hinge loss.
    """
    print("\n" + "=" * 70)
    print("Example 8: Support Vector Machine")
    print("=" * 70)
    
    class LinearSVM:
        """Linear SVM with hinge loss."""
        
        def __init__(self, C: float = 1.0, learning_rate: float = 0.01,
                     max_iter: int = 1000):
            self.C = C
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.w = None
            self.b = None
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVM':
            n, p = X.shape
            
            # Convert labels to {-1, +1}
            y_svm = 2 * y - 1
            
            # Initialize
            self.w = np.zeros(p)
            self.b = 0.0
            
            for iteration in range(self.max_iter):
                # Margin
                margins = y_svm * (X @ self.w + self.b)
                
                # Gradient
                hinge_active = margins < 1
                
                grad_w = self.w - self.C * np.sum(
                    y_svm[hinge_active, np.newaxis] * X[hinge_active], axis=0
                )
                grad_b = -self.C * np.sum(y_svm[hinge_active])
                
                # Update
                self.w -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            scores = X @ self.w + self.b
            return (scores >= 0).astype(int)
        
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            return X @ self.w + self.b
    
    # Generate data
    np.random.seed(42)
    n = 200
    
    X0 = np.random.randn(n // 2, 2) + np.array([-1.5, 0])
    X1 = np.random.randn(n // 2, 2) + np.array([1.5, 0])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    # Shuffle
    idx = np.random.permutation(n)
    X, y = X[idx], y[idx]
    
    # Fit
    model = LinearSVM(C=1.0, learning_rate=0.01, max_iter=2000)
    model.fit(X, y)
    
    # Evaluate
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"\nWeight vector: {model.w.round(4)}")
    print(f"Bias: {model.b:.4f}")
    print(f"Training accuracy: {accuracy:.2%}")
    
    # Support vectors (margin violations)
    y_svm = 2 * y - 1
    margins = y_svm * model.decision_function(X)
    n_sv = np.sum(margins <= 1 + 1e-6)
    print(f"Support vectors: {n_sv} / {n}")
    
    # Margin width
    margin_width = 2 / np.linalg.norm(model.w)
    print(f"Margin width: {margin_width:.4f}")


# =============================================================================
# Example 9: Elastic Net
# =============================================================================

def example_elastic_net():
    """
    Elastic Net combining L1 and L2 regularization.
    """
    print("\n" + "=" * 70)
    print("Example 9: Elastic Net")
    print("=" * 70)
    
    class ElasticNet:
        """Elastic Net via coordinate descent."""
        
        def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                     max_iter: int = 1000, tol: float = 1e-4):
            self.alpha = alpha
            self.l1_ratio = l1_ratio
            self.max_iter = max_iter
            self.tol = tol
            self.coef_ = None
            self.intercept_ = None
        
        def _soft_threshold(self, z: float, threshold: float) -> float:
            if z > threshold:
                return z - threshold
            elif z < -threshold:
                return z + threshold
            return 0.0
        
        def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
            n, p = X.shape
            
            # Center
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
            
            self.coef_ = np.zeros(p)
            l1_weight = self.alpha * self.l1_ratio
            l2_weight = self.alpha * (1 - self.l1_ratio)
            
            X_col_norms_sq = np.sum(X**2, axis=0)
            
            for iteration in range(self.max_iter):
                coef_old = self.coef_.copy()
                
                for j in range(p):
                    r_j = y - X @ self.coef_ + X[:, j] * self.coef_[j]
                    rho_j = X[:, j] @ r_j
                    
                    denominator = X_col_norms_sq[j] + n * l2_weight
                    if denominator > 0:
                        self.coef_[j] = self._soft_threshold(
                            rho_j / denominator,
                            n * l1_weight / denominator
                        )
                
                if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                    break
            
            self.intercept_ = y_mean - X_mean @ self.coef_
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Generate data with groups of correlated features
    np.random.seed(42)
    n = 100
    p = 20
    
    # Create correlated feature groups
    X = np.random.randn(n, p)
    for i in range(0, p, 4):
        base = np.random.randn(n)
        for j in range(4):
            if i + j < p:
                X[:, i + j] = base + 0.1 * np.random.randn(n)
    
    # Sparse true model
    true_coef = np.zeros(p)
    true_coef[[0, 4, 8, 12, 16]] = [2.0, -1.5, 1.0, -0.5, 1.5]
    y = X @ true_coef + 0.3 * np.random.randn(n)
    
    print(f"\nComparing Lasso vs Elastic Net:")
    print(f"True non-zero: {np.where(true_coef != 0)[0]}")
    
    # Lasso (l1_ratio=1)
    lasso = ElasticNet(alpha=0.1, l1_ratio=1.0)
    lasso.fit(X, y)
    lasso_nonzero = np.where(np.abs(lasso.coef_) > 1e-4)[0]
    
    print(f"\nLasso (L1 only):")
    print(f"  Non-zero features: {lasso_nonzero}")
    
    # Elastic Net
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    enet.fit(X, y)
    enet_nonzero = np.where(np.abs(enet.coef_) > 1e-4)[0]
    
    print(f"\nElastic Net (L1 + L2):")
    print(f"  Non-zero features: {enet_nonzero}")
    print(f"  Note: Elastic Net tends to select correlated features together")


# =============================================================================
# Example 10: Cross-Validation and Model Selection
# =============================================================================

def example_model_selection():
    """
    Model selection using cross-validation.
    """
    print("\n" + "=" * 70)
    print("Example 10: Cross-Validation and Model Selection")
    print("=" * 70)
    
    def k_fold_cv(X: np.ndarray, y: np.ndarray, model_class,
                  params: dict, n_folds: int = 5) -> Tuple[float, float]:
        """K-fold cross-validation."""
        n = len(y)
        fold_size = n // n_folds
        indices = np.random.permutation(n)
        
        fold_errors = []
        
        for k in range(n_folds):
            # Split
            val_idx = indices[k * fold_size:(k + 1) * fold_size]
            train_idx = np.concatenate([indices[:k * fold_size], indices[(k + 1) * fold_size:]])
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Fit and evaluate
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            mse = np.mean((y_val - y_pred)**2)
            fold_errors.append(mse)
        
        return np.mean(fold_errors), np.std(fold_errors)
    
    # Simple Ridge for demo
    class SimpleRidge:
        def __init__(self, alpha=1.0):
            self.alpha = alpha
            self.coef_ = None
        
        def fit(self, X, y):
            n, p = X.shape
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
            self.coef_ = np.linalg.solve(
                X_c.T @ X_c + self.alpha * np.eye(p),
                X_c.T @ y_c
            )
            self.intercept_ = y_mean - X_mean @ self.coef_
            return self
        
        def predict(self, X):
            return X @ self.coef_ + self.intercept_
    
    # Generate data
    np.random.seed(42)
    n = 100
    p = 10
    
    X = np.random.randn(n, p)
    true_coef = np.random.randn(p)
    y = X @ true_coef + 0.5 * np.random.randn(n)
    
    # Grid search over alpha
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print(f"\nCross-validation results (5-fold):")
    print(f"{'Alpha':>10} {'CV MSE':>12} {'Std':>10}")
    print("-" * 35)
    
    results = []
    for alpha in alphas:
        cv_mse, cv_std = k_fold_cv(X, y, SimpleRidge, {'alpha': alpha})
        results.append((alpha, cv_mse, cv_std))
        print(f"{alpha:>10.3f} {cv_mse:>12.4f} {cv_std:>10.4f}")
    
    # Find best
    best = min(results, key=lambda x: x[1])
    print(f"\nBest α = {best[0]} with CV MSE = {best[1]:.4f}")
    
    # Information criteria
    print(f"\nInformation Criteria (full data):")
    for alpha in alphas:
        model = SimpleRidge(alpha=alpha)
        model.fit(X, y)
        residuals = y - model.predict(X)
        rss = np.sum(residuals**2)
        sigma2 = rss / n
        
        # Effective df for ridge
        eigenvalues = np.linalg.eigvalsh(X.T @ X)
        df_eff = np.sum(eigenvalues / (eigenvalues + alpha))
        
        # AIC and BIC
        log_lik = -n/2 * np.log(2*np.pi*sigma2) - rss/(2*sigma2)
        aic = -2*log_lik + 2*df_eff
        bic = -2*log_lik + df_eff*np.log(n)
        
        print(f"  α={alpha:6.3f}: AIC={aic:.2f}, BIC={bic:.2f}, df_eff={df_eff:.2f}")


def main():
    """Run all examples."""
    print("LINEAR MODELS - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    example_ols_regression()
    example_ridge_regression()
    example_lasso_regression()
    example_logistic_regression()
    example_softmax_regression()
    example_bayesian_linear_regression()
    example_lda()
    example_svm()
    example_elastic_net()
    example_model_selection()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
