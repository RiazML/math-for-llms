"""
Linear Models - Exercises
=========================

Practice exercises for linear models: regression, classification, and regularization.
"""

import numpy as np
from typing import Tuple, Optional, List, Callable, Dict
from scipy import stats


# =============================================================================
# Exercise 1: OLS from Scratch
# =============================================================================

def exercise_ols():
    """
    Exercise: Implement OLS regression with full diagnostics.
    
    Tasks:
    1. Closed-form solution
    2. Standard errors and t-statistics
    3. R² and adjusted R²
    """
    print("=" * 70)
    print("Exercise 1: OLS Regression")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class OLS:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
            self.se_ = None
            self.t_stats_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit OLS and compute diagnostics."""
            pass
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            pass
        
        def r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
            pass
        
        def adjusted_r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
            pass


def solution_ols():
    """Reference solution for OLS."""
    print("\n--- Solution ---\n")
    
    class OLS:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
            self.se_ = None
            self.t_stats_ = None
            self._sigma2 = None
            self._cov = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            n, p = X.shape
            X_aug = np.column_stack([np.ones(n), X])
            
            # Normal equations
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y
            beta = np.linalg.solve(XtX, Xty)
            
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
            
            # Residual variance
            residuals = y - X_aug @ beta
            df = n - p - 1
            self._sigma2 = np.sum(residuals**2) / df
            
            # Covariance and standard errors
            self._cov = self._sigma2 * np.linalg.inv(XtX)
            self.se_ = np.sqrt(np.diag(self._cov)[1:])
            self.t_stats_ = self.coef_ / self.se_
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
        
        def r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
            y_pred = self.predict(X)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            return 1 - ss_res / ss_tot
        
        def adjusted_r_squared(self, X: np.ndarray, y: np.ndarray) -> float:
            n, p = X.shape
            r2 = self.r_squared(X, y)
            return 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Test
    np.random.seed(42)
    n, p = 100, 3
    X = np.random.randn(n, p)
    true_coef = np.array([2.0, -1.0, 0.5])
    y = X @ true_coef + 1.0 + 0.5 * np.random.randn(n)
    
    model = OLS()
    model.fit(X, y)
    
    print(f"True coefficients: {true_coef}")
    print(f"Estimated coefficients: {model.coef_.round(4)}")
    print(f"Standard errors: {model.se_.round(4)}")
    print(f"t-statistics: {model.t_stats_.round(4)}")
    print(f"R²: {model.r_squared(X, y):.4f}")
    print(f"Adjusted R²: {model.adjusted_r_squared(X, y):.4f}")


# =============================================================================
# Exercise 2: Ridge Regression with Cross-Validation
# =============================================================================

def exercise_ridge_cv():
    """
    Exercise: Implement Ridge with automatic alpha selection.
    
    Tasks:
    1. Ridge regression
    2. K-fold cross-validation
    3. Optimal alpha selection
    """
    print("\n" + "=" * 70)
    print("Exercise 2: Ridge Regression with CV")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class RidgeCV:
        def __init__(self, alphas: np.ndarray = None, n_folds: int = 5):
            self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 20)
            self.n_folds = n_folds
            self.best_alpha_ = None
            self.coef_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit Ridge with CV to find best alpha."""
            pass


def solution_ridge_cv():
    """Reference solution for Ridge CV."""
    print("\n--- Solution ---\n")
    
    class RidgeCV:
        def __init__(self, alphas: np.ndarray = None, n_folds: int = 5):
            self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 20)
            self.n_folds = n_folds
            self.best_alpha_ = None
            self.cv_scores_ = None
            self.coef_ = None
            self.intercept_ = None
        
        def _ridge_fit(self, X, y, alpha):
            n, p = X.shape
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
            
            coef = np.linalg.solve(X_c.T @ X_c + alpha * np.eye(p), X_c.T @ y_c)
            intercept = y_mean - X_mean @ coef
            return coef, intercept
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            n = len(y)
            fold_size = n // self.n_folds
            indices = np.random.permutation(n)
            
            self.cv_scores_ = np.zeros(len(self.alphas))
            
            for i, alpha in enumerate(self.alphas):
                fold_mses = []
                
                for k in range(self.n_folds):
                    val_idx = indices[k * fold_size:(k + 1) * fold_size]
                    train_idx = np.concatenate([
                        indices[:k * fold_size],
                        indices[(k + 1) * fold_size:]
                    ])
                    
                    coef, intercept = self._ridge_fit(X[train_idx], y[train_idx], alpha)
                    y_pred = X[val_idx] @ coef + intercept
                    fold_mses.append(np.mean((y[val_idx] - y_pred)**2))
                
                self.cv_scores_[i] = np.mean(fold_mses)
            
            self.best_alpha_ = self.alphas[np.argmin(self.cv_scores_)]
            self.coef_, self.intercept_ = self._ridge_fit(X, y, self.best_alpha_)
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Test
    np.random.seed(42)
    n, p = 100, 20
    
    X = np.random.randn(n, p)
    # Add multicollinearity
    X[:, 10:] = X[:, :10] + 0.1 * np.random.randn(n, 10)
    
    true_coef = np.random.randn(p)
    y = X @ true_coef + 0.5 * np.random.randn(n)
    
    model = RidgeCV()
    model.fit(X, y)
    
    print(f"Best alpha: {model.best_alpha_:.4f}")
    print(f"Best CV MSE: {np.min(model.cv_scores_):.4f}")
    
    # Compare with OLS
    ols_coef = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"\n||β_ridge||₂: {np.linalg.norm(model.coef_):.4f}")
    print(f"||β_ols||₂: {np.linalg.norm(ols_coef):.4f}")


# =============================================================================
# Exercise 3: Lasso with Coordinate Descent
# =============================================================================

def exercise_lasso():
    """
    Exercise: Implement Lasso for sparse regression.
    
    Tasks:
    1. Soft thresholding operator
    2. Coordinate descent algorithm
    3. Feature selection
    """
    print("\n" + "=" * 70)
    print("Exercise 3: Lasso Regression")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def soft_threshold(z: float, threshold: float) -> float:
        """Soft thresholding operator."""
        pass
    
    class Lasso:
        def __init__(self, alpha: float = 1.0, max_iter: int = 1000):
            self.alpha = alpha
            self.max_iter = max_iter
            self.coef_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit via coordinate descent."""
            pass


def solution_lasso():
    """Reference solution for Lasso."""
    print("\n--- Solution ---\n")
    
    def soft_threshold(z: float, threshold: float) -> float:
        if z > threshold:
            return z - threshold
        elif z < -threshold:
            return z + threshold
        return 0.0
    
    class Lasso:
        def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
            self.alpha = alpha
            self.max_iter = max_iter
            self.tol = tol
            self.coef_ = None
            self.intercept_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            n, p = X.shape
            
            # Center
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_c = X - X_mean
            y_c = y - y_mean
            
            self.coef_ = np.zeros(p)
            col_norms_sq = np.sum(X_c**2, axis=0)
            
            for _ in range(self.max_iter):
                coef_old = self.coef_.copy()
                
                for j in range(p):
                    r_j = y_c - X_c @ self.coef_ + X_c[:, j] * self.coef_[j]
                    rho_j = X_c[:, j] @ r_j
                    
                    if col_norms_sq[j] > 0:
                        self.coef_[j] = soft_threshold(
                            rho_j / col_norms_sq[j],
                            n * self.alpha / col_norms_sq[j]
                        )
                
                if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                    break
            
            self.intercept_ = y_mean - X_mean @ self.coef_
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Test with sparse ground truth
    np.random.seed(42)
    n, p = 100, 30
    
    X = np.random.randn(n, p)
    true_coef = np.zeros(p)
    true_coef[[0, 5, 10, 15, 20]] = [3.0, -2.0, 1.5, -1.0, 2.0]
    y = X @ true_coef + 0.3 * np.random.randn(n)
    
    print(f"True non-zero indices: {np.where(true_coef != 0)[0]}")
    
    for alpha in [0.01, 0.05, 0.1, 0.5]:
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        nonzero = np.where(np.abs(model.coef_) > 1e-4)[0]
        print(f"α={alpha:.2f}: {len(nonzero)} non-zero, indices={nonzero}")


# =============================================================================
# Exercise 4: Logistic Regression
# =============================================================================

def exercise_logistic():
    """
    Exercise: Implement logistic regression with gradient descent.
    
    Tasks:
    1. Sigmoid function (numerically stable)
    2. Cross-entropy loss
    3. Gradient descent optimization
    """
    print("\n" + "=" * 70)
    print("Exercise 4: Logistic Regression")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        pass
    
    class LogisticRegression:
        def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000):
            self.lr = learning_rate
            self.max_iter = max_iter
            self.coef_ = None
            self.intercept_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            pass
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            pass
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            pass


def solution_logistic():
    """Reference solution for logistic regression."""
    print("\n--- Solution ---\n")
    
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return np.where(z >= 0, 
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))
    
    class LogisticRegression:
        def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000,
                     l2_reg: float = 0.01):
            self.lr = learning_rate
            self.max_iter = max_iter
            self.l2_reg = l2_reg
            self.coef_ = None
            self.intercept_ = None
            self.loss_history_ = []
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            n, p = X.shape
            
            self.coef_ = np.zeros(p)
            self.intercept_ = 0.0
            
            for _ in range(self.max_iter):
                z = X @ self.coef_ + self.intercept_
                p_hat = sigmoid(z)
                
                # Gradient
                grad_coef = X.T @ (p_hat - y) / n + self.l2_reg * self.coef_
                grad_intercept = np.mean(p_hat - y)
                
                # Update
                self.coef_ -= self.lr * grad_coef
                self.intercept_ -= self.lr * grad_intercept
                
                # Loss
                loss = -np.mean(y * np.log(p_hat + 1e-10) + 
                               (1 - y) * np.log(1 - p_hat + 1e-10))
                self.loss_history_.append(loss)
            
            return self
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            z = X @ self.coef_ + self.intercept_
            p = sigmoid(z)
            return np.column_stack([1 - p, p])
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    # Test
    np.random.seed(42)
    n = 200
    
    X0 = np.random.randn(n // 2, 2) + np.array([-1, -1])
    X1 = np.random.randn(n // 2, 2) + np.array([1, 1])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    model = LogisticRegression(learning_rate=0.5, max_iter=500)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"Coefficients: {model.coef_.round(4)}")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Final loss: {model.loss_history_[-1]:.4f}")


# =============================================================================
# Exercise 5: Bayesian Linear Regression
# =============================================================================

def exercise_bayesian_regression():
    """
    Exercise: Implement Bayesian linear regression.
    
    Tasks:
    1. Posterior distribution
    2. Predictive distribution
    3. Uncertainty quantification
    """
    print("\n" + "=" * 70)
    print("Exercise 5: Bayesian Linear Regression")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class BayesianRegression:
        def __init__(self, noise_var: float = 1.0, prior_var: float = 10.0):
            self.noise_var = noise_var
            self.prior_var = prior_var
            self.posterior_mean = None
            self.posterior_cov = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            pass
        
        def predict(self, X: np.ndarray, return_std: bool = False):
            pass


def solution_bayesian_regression():
    """Reference solution for Bayesian regression."""
    print("\n--- Solution ---\n")
    
    class BayesianRegression:
        def __init__(self, noise_var: float = 1.0, prior_var: float = 10.0):
            self.noise_var = noise_var
            self.prior_var = prior_var
            self.posterior_mean = None
            self.posterior_cov = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            n, p = X.shape
            
            # Prior: N(0, prior_var * I)
            prior_prec = np.eye(p) / self.prior_var
            
            # Posterior
            self.posterior_cov = np.linalg.inv(
                prior_prec + X.T @ X / self.noise_var
            )
            self.posterior_mean = self.posterior_cov @ (X.T @ y / self.noise_var)
            
            return self
        
        def predict(self, X: np.ndarray, return_std: bool = False):
            mean = X @ self.posterior_mean
            
            if return_std:
                var = np.array([x @ self.posterior_cov @ x for x in X]) + self.noise_var
                return mean, np.sqrt(var)
            
            return mean
        
        def sample_coefficients(self, n_samples: int = 100) -> np.ndarray:
            return np.random.multivariate_normal(
                self.posterior_mean, self.posterior_cov, n_samples
            )
    
    # Test
    np.random.seed(42)
    n = 30
    
    X = np.sort(np.random.uniform(-3, 3, n))[:, np.newaxis]
    true_coef = 2.0
    noise_std = 0.5
    y = X.flatten() * true_coef + noise_std * np.random.randn(n)
    
    model = BayesianRegression(noise_var=noise_std**2, prior_var=10.0)
    model.fit(X, y)
    
    print(f"True coefficient: {true_coef}")
    print(f"Posterior mean: {model.posterior_mean[0]:.4f}")
    print(f"Posterior std: {np.sqrt(model.posterior_cov[0, 0]):.4f}")
    
    # Predictions with uncertainty
    X_test = np.array([[-3], [0], [3]])
    y_pred, y_std = model.predict(X_test, return_std=True)
    
    print(f"\nPredictions with 95% intervals:")
    for i, x in enumerate(X_test.flatten()):
        print(f"  x={x:+.0f}: {y_pred[i]:.2f} ± {1.96*y_std[i]:.2f}")


# =============================================================================
# Exercise 6: Linear Discriminant Analysis
# =============================================================================

def exercise_lda():
    """
    Exercise: Implement LDA classifier.
    
    Tasks:
    1. Class means and shared covariance
    2. Discriminant scores
    3. Classification
    """
    print("\n" + "=" * 70)
    print("Exercise 6: Linear Discriminant Analysis")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class LDA:
        def __init__(self):
            self.means_ = None
            self.cov_ = None
            self.priors_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            pass
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            pass


def solution_lda():
    """Reference solution for LDA."""
    print("\n--- Solution ---\n")
    
    class LDA:
        def __init__(self):
            self.means_ = None
            self.cov_ = None
            self.priors_ = None
            self.classes_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            n, p = X.shape
            
            # Priors
            self.priors_ = np.array([np.mean(y == k) for k in self.classes_])
            
            # Means
            self.means_ = np.array([np.mean(X[y == k], axis=0) 
                                    for k in self.classes_])
            
            # Pooled covariance
            self.cov_ = np.zeros((p, p))
            for k in self.classes_:
                X_k = X[y == k]
                self.cov_ += (X_k - self.means_[k]).T @ (X_k - self.means_[k])
            self.cov_ /= (n - n_classes)
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            cov_inv = np.linalg.inv(self.cov_)
            scores = np.zeros((len(X), len(self.classes_)))
            
            for k, (mu, pi) in enumerate(zip(self.means_, self.priors_)):
                scores[:, k] = (X @ cov_inv @ mu - 0.5 * mu @ cov_inv @ mu 
                               + np.log(pi))
            
            return self.classes_[np.argmax(scores, axis=1)]
    
    # Test
    np.random.seed(42)
    n_per_class = 100
    
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    X0 = np.random.multivariate_normal([-1, -1], cov, n_per_class)
    X1 = np.random.multivariate_normal([1, 1], cov, n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    
    model = LDA()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"Class means:\n{model.means_.round(4)}")
    print(f"\nShared covariance:\n{model.cov_.round(4)}")
    print(f"\nAccuracy: {accuracy:.2%}")


# =============================================================================
# Exercise 7: Support Vector Machine (Hinge Loss)
# =============================================================================

def exercise_svm():
    """
    Exercise: Implement linear SVM with hinge loss.
    
    Tasks:
    1. Hinge loss function
    2. Subgradient descent
    3. Margin computation
    """
    print("\n" + "=" * 70)
    print("Exercise 7: Support Vector Machine")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class LinearSVM:
        def __init__(self, C: float = 1.0, learning_rate: float = 0.01):
            self.C = C
            self.lr = learning_rate
            self.w = None
            self.b = None
        
        def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 1000):
            """Fit using subgradient descent."""
            pass
        
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            pass
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            pass


def solution_svm():
    """Reference solution for SVM."""
    print("\n--- Solution ---\n")
    
    class LinearSVM:
        def __init__(self, C: float = 1.0, learning_rate: float = 0.01):
            self.C = C
            self.lr = learning_rate
            self.w = None
            self.b = None
        
        def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 1000):
            n, p = X.shape
            y_svm = 2 * y - 1  # Convert to {-1, +1}
            
            self.w = np.zeros(p)
            self.b = 0.0
            
            for _ in range(max_iter):
                margins = y_svm * (X @ self.w + self.b)
                hinge_active = margins < 1
                
                # Subgradient
                grad_w = self.w - self.C * np.sum(
                    y_svm[hinge_active, np.newaxis] * X[hinge_active], axis=0
                )
                grad_b = -self.C * np.sum(y_svm[hinge_active])
                
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b
            
            return self
        
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            return X @ self.w + self.b
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.decision_function(X) >= 0).astype(int)
    
    # Test
    np.random.seed(42)
    n = 200
    
    X0 = np.random.randn(n // 2, 2) + np.array([-1.5, 0])
    X1 = np.random.randn(n // 2, 2) + np.array([1.5, 0])
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    model = LinearSVM(C=1.0, learning_rate=0.01)
    model.fit(X, y, max_iter=2000)
    
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)
    
    print(f"Weights: {model.w.round(4)}")
    print(f"Bias: {model.b:.4f}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Margin
    y_svm = 2 * y - 1
    margins = y_svm * model.decision_function(X)
    print(f"Min margin: {np.min(margins):.4f}")
    print(f"Support vectors: {np.sum(margins <= 1 + 1e-4)}")


# =============================================================================
# Exercise 8: Polynomial Features and Basis Expansion
# =============================================================================

def exercise_polynomial():
    """
    Exercise: Implement polynomial feature expansion.
    
    Tasks:
    1. Polynomial feature transformation
    2. Fit and evaluate
    3. Overfitting demonstration
    """
    print("\n" + "=" * 70)
    print("Exercise 8: Polynomial Features")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
        """Create polynomial features up to given degree."""
        pass


def solution_polynomial():
    """Reference solution for polynomial features."""
    print("\n--- Solution ---\n")
    
    def polynomial_features(X: np.ndarray, degree: int, include_bias: bool = False) -> np.ndarray:
        """Create polynomial features."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n = len(X)
        features = []
        
        if include_bias:
            features.append(np.ones(n))
        
        for d in range(1, degree + 1):
            features.append(X[:, 0] ** d)
        
        return np.column_stack(features)
    
    class PolynomialRegression:
        def __init__(self, degree: int):
            self.degree = degree
            self.coef_ = None
        
        def fit(self, X, y):
            X_poly = polynomial_features(X, self.degree)
            self.coef_ = np.linalg.lstsq(X_poly, y, rcond=None)[0]
            return self
        
        def predict(self, X):
            X_poly = polynomial_features(X, self.degree)
            return X_poly @ self.coef_
    
    # Test
    np.random.seed(42)
    n = 30
    
    X = np.sort(np.random.uniform(-3, 3, n))
    y_true = np.sin(X)
    y = y_true + 0.3 * np.random.randn(n)
    
    print("Comparing polynomial degrees:")
    print(f"{'Degree':>8} {'Train MSE':>12} {'Complexity':>12}")
    print("-" * 35)
    
    for degree in [1, 3, 5, 10, 15]:
        model = PolynomialRegression(degree)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred)**2)
        print(f"{degree:>8} {mse:>12.4f} {degree:>12}")
    
    print("\n(Higher degree → lower train error but risk of overfitting)")


# =============================================================================
# Exercise 9: Weighted Least Squares
# =============================================================================

def exercise_wls():
    """
    Exercise: Implement weighted least squares.
    
    Tasks:
    1. WLS solution
    2. Handling heteroscedasticity
    3. Comparison with OLS
    """
    print("\n" + "=" * 70)
    print("Exercise 9: Weighted Least Squares")
    print("=" * 70)
    
    # YOUR CODE HERE
    
    class WLS:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
            """Weighted least squares: min Σ w_i (y_i - x_i'β)²"""
            pass


def solution_wls():
    """Reference solution for WLS."""
    print("\n--- Solution ---\n")
    
    class WLS:
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None
        
        def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
            n, p = X.shape
            X_aug = np.column_stack([np.ones(n), X])
            
            W = np.diag(weights)
            
            # (X'WX)^{-1} X'Wy
            XtWX = X_aug.T @ W @ X_aug
            XtWy = X_aug.T @ W @ y
            
            beta = np.linalg.solve(XtWX, XtWy)
            
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            return X @ self.coef_ + self.intercept_
    
    # Test with heteroscedastic data
    np.random.seed(42)
    n = 100
    
    X = np.random.uniform(0, 10, n)[:, np.newaxis]
    true_coef = 2.0
    
    # Variance increases with X
    noise_std = 0.5 + 0.3 * X.flatten()
    y = X.flatten() * true_coef + 1.0 + noise_std * np.random.randn(n)
    
    # OLS (ignores heteroscedasticity)
    ols_coef = np.linalg.lstsq(
        np.column_stack([np.ones(n), X]), y, rcond=None
    )[0]
    
    # WLS (weights = 1/variance)
    weights = 1 / noise_std**2
    wls = WLS()
    wls.fit(X, y, weights)
    
    print(f"True coefficient: {true_coef}")
    print(f"OLS coefficient: {ols_coef[1]:.4f}")
    print(f"WLS coefficient: {wls.coef_[0]:.4f}")
    print(f"\n(WLS is more efficient when variance structure is known)")


# =============================================================================
# Exercise 10: Complete Linear Model Pipeline
# =============================================================================

def exercise_pipeline():
    """
    Exercise: Build a complete linear modeling pipeline.
    
    Tasks:
    1. Data preprocessing
    2. Model selection via CV
    3. Final evaluation
    """
    print("\n" + "=" * 70)
    print("Exercise 10: Complete Pipeline")
    print("=" * 70)
    
    # YOUR CODE HERE
    pass


def solution_pipeline():
    """Reference solution for pipeline."""
    print("\n--- Solution ---\n")
    
    class LinearModelPipeline:
        def __init__(self):
            self.scaler_mean_ = None
            self.scaler_std_ = None
            self.best_model_ = None
            self.best_params_ = None
        
        def _standardize(self, X, fit=False):
            if fit:
                self.scaler_mean_ = np.mean(X, axis=0)
                self.scaler_std_ = np.std(X, axis=0) + 1e-8
            return (X - self.scaler_mean_) / self.scaler_std_
        
        def _ridge_fit(self, X, y, alpha):
            coef = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
            return coef
        
        def fit(self, X_train, y_train, X_val, y_val):
            # Standardize
            X_train_s = self._standardize(X_train, fit=True)
            X_val_s = self._standardize(X_val)
            
            # Center y
            y_mean = np.mean(y_train)
            y_train_c = y_train - y_mean
            
            # Grid search
            alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
            best_val_mse = np.inf
            
            for alpha in alphas:
                coef = self._ridge_fit(X_train_s, y_train_c, alpha)
                y_val_pred = X_val_s @ coef + y_mean
                val_mse = np.mean((y_val - y_val_pred)**2)
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    self.best_params_ = {'alpha': alpha}
                    self.best_model_ = {'coef': coef, 'intercept': y_mean}
            
            return self
        
        def predict(self, X):
            X_s = self._standardize(X)
            return X_s @ self.best_model_['coef'] + self.best_model_['intercept']
    
    # Test
    np.random.seed(42)
    n = 200
    p = 10
    
    X = np.random.randn(n, p)
    true_coef = np.random.randn(p)
    y = X @ true_coef + 0.5 * np.random.randn(n)
    
    # Split
    n_train = 120
    n_val = 40
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    pipeline = LinearModelPipeline()
    pipeline.fit(X_train, y_train, X_val, y_val)
    
    y_test_pred = pipeline.predict(X_test)
    test_mse = np.mean((y_test - y_test_pred)**2)
    
    print(f"Best alpha: {pipeline.best_params_['alpha']}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test R²: {1 - test_mse / np.var(y_test):.4f}")


def main():
    """Run all exercises with solutions."""
    print("LINEAR MODELS - EXERCISES")
    print("=" * 70)
    
    exercise_ols()
    solution_ols()
    
    exercise_ridge_cv()
    solution_ridge_cv()
    
    exercise_lasso()
    solution_lasso()
    
    exercise_logistic()
    solution_logistic()
    
    exercise_bayesian_regression()
    solution_bayesian_regression()
    
    exercise_lda()
    solution_lda()
    
    exercise_svm()
    solution_svm()
    
    exercise_polynomial()
    solution_polynomial()
    
    exercise_wls()
    solution_wls()
    
    exercise_pipeline()
    solution_pipeline()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
