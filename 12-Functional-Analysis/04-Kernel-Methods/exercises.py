"""
Kernel Methods - Exercises
==========================

Practice problems for kernel method concepts
with solutions and implementations.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# Exercise 1: Custom Kernel Implementation
# =============================================================================

def exercise_1_custom_kernel():
    """
    Exercise: Implement and verify custom kernel functions.
    
    Tasks:
    1. Implement the rational quadratic kernel
    2. Implement the spectral mixture kernel
    3. Verify positive definiteness
    4. Show that the sum/product of kernels is a kernel
    """
    print("Exercise 1: Custom Kernel Implementation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    def rational_quadratic(x: np.ndarray, y: np.ndarray,
                          alpha: float = 1.0, length_scale: float = 1.0) -> float:
        """
        Rational quadratic kernel:
        K(x, y) = (1 + ||x-y||²/(2αl²))^(-α)
        
        Equivalent to infinite mixture of RBFs with different length scales.
        """
        r_sq = np.linalg.norm(x - y)**2
        return (1 + r_sq / (2 * alpha * length_scale**2)) ** (-alpha)
    
    def spectral_mixture_kernel(x: np.ndarray, y: np.ndarray,
                               weights: np.ndarray, means: np.ndarray,
                               variances: np.ndarray) -> float:
        """
        Spectral mixture kernel:
        K(x, y) = Σ w_q exp(-2π²(x-y)²v_q) cos(2π(x-y)μ_q)
        """
        r = x - y
        r_sq = np.sum(r**2)
        
        result = 0
        for w, mu, v in zip(weights, means, variances):
            result += w * np.exp(-2 * np.pi**2 * r_sq * v) * np.cos(2 * np.pi * np.dot(r, mu))
        
        return result
    
    def verify_psd(kernel: Callable, X: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Verify kernel produces PSD matrix."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j])
        
        eigenvalues = np.linalg.eigvalsh(K)
        return np.all(eigenvalues >= -1e-10), eigenvalues
    
    def kernel_sum(k1: Callable, k2: Callable) -> Callable:
        """Sum of two kernels is a kernel."""
        return lambda x, y: k1(x, y) + k2(x, y)
    
    def kernel_product(k1: Callable, k2: Callable) -> Callable:
        """Product of two kernels is a kernel."""
        return lambda x, y: k1(x, y) * k2(x, y)
    
    # Test
    np.random.seed(42)
    X = np.random.randn(20, 3)
    
    # Verify rational quadratic
    print("\n1. Rational Quadratic Kernel:")
    for alpha in [0.5, 1.0, 2.0]:
        kernel = lambda x, y, a=alpha: rational_quadratic(x, y, a)
        is_psd, eigs = verify_psd(kernel, X)
        print(f"   α={alpha}: PSD={is_psd}, min_eig={eigs.min():.6f}")
    
    # Verify spectral mixture
    print("\n2. Spectral Mixture Kernel:")
    weights = np.array([0.5, 0.5])
    means = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    variances = np.array([0.1, 0.2])
    
    sm_kernel = lambda x, y: spectral_mixture_kernel(x, y, weights, means, variances)
    is_psd, eigs = verify_psd(sm_kernel, X)
    print(f"   PSD={is_psd}, min_eig={eigs.min():.6f}")
    
    # Kernel operations
    print("\n3. Kernel Operations:")
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2)
    poly = lambda x, y: (np.dot(x, y) + 1)**2
    
    is_psd_sum, _ = verify_psd(kernel_sum(rbf, poly), X)
    is_psd_prod, _ = verify_psd(kernel_product(rbf, poly), X)
    
    print(f"   RBF + Polynomial is PSD: {is_psd_sum}")
    print(f"   RBF × Polynomial is PSD: {is_psd_prod}")
    
    return rational_quadratic, spectral_mixture_kernel


# =============================================================================
# Exercise 2: Kernel Matrix Properties
# =============================================================================

def exercise_2_kernel_matrix():
    """
    Exercise: Analyze kernel matrix properties.
    
    Tasks:
    1. Compute kernel matrix eigenspectrum
    2. Analyze effective rank
    3. Examine condition number for different parameters
    4. Study the effect of regularization
    """
    print("\nExercise 2: Kernel Matrix Properties")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    def analyze_kernel_matrix(K: np.ndarray) -> Dict:
        """Analyze properties of a kernel matrix."""
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        
        # Effective rank (number of significant eigenvalues)
        total = np.sum(eigenvalues)
        cumsum = np.cumsum(eigenvalues) / (total + 1e-10)
        effective_rank_90 = np.searchsorted(cumsum, 0.9) + 1
        effective_rank_99 = np.searchsorted(cumsum, 0.99) + 1
        
        # Condition number
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        condition = pos_eigs[0] / pos_eigs[-1] if len(pos_eigs) > 1 else 1
        
        # Rank
        rank = np.sum(eigenvalues > 1e-10)
        
        return {
            'eigenvalues': eigenvalues,
            'rank': rank,
            'effective_rank_90': effective_rank_90,
            'effective_rank_99': effective_rank_99,
            'condition': condition,
            'trace': np.trace(K),
            'frobenius': np.linalg.norm(K, 'fro')
        }
    
    def gram_matrix(X: np.ndarray, kernel: Callable) -> np.ndarray:
        """Compute Gram matrix."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j])
        return K
    
    # Test data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 5)
    
    print(f"\nData: {n} points in 5D")
    
    # Study RBF with different bandwidths
    print("\n1. RBF Kernel - Effect of bandwidth:")
    print(f"{'sigma':<8} {'Rank':<8} {'Eff90':<8} {'Eff99':<8} {'Condition':<12}")
    print("-" * 50)
    
    for sigma in [0.1, 0.5, 1.0, 2.0, 5.0]:
        rbf = lambda x, y, s=sigma: np.exp(-np.linalg.norm(x - y)**2 / (2 * s**2))
        K = gram_matrix(X, rbf)
        props = analyze_kernel_matrix(K)
        
        print(f"{sigma:<8.1f} {props['rank']:<8d} {props['effective_rank_90']:<8d} "
              f"{props['effective_rank_99']:<8d} {props['condition']:<12.2e}")
    
    # Effect of regularization
    print("\n2. Effect of Regularization (RBF σ=1):")
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2)
    K = gram_matrix(X, rbf)
    
    print(f"{'λ':<10} {'Condition(K+λI)':<20}")
    print("-" * 30)
    
    for lam in [0, 1e-6, 1e-4, 1e-2, 1, 10]:
        K_reg = K + lam * np.eye(n)
        eigs = np.linalg.eigvalsh(K_reg)
        cond = eigs.max() / eigs.min()
        print(f"{lam:<10.0e} {cond:<20.2e}")
    
    return analyze_kernel_matrix


# =============================================================================
# Exercise 3: Kernel SVM Implementation
# =============================================================================

def exercise_3_kernel_svm():
    """
    Exercise: Implement kernel SVM with different kernels.
    
    Tasks:
    1. Implement SVM dual optimization
    2. Compare kernels on XOR problem
    3. Analyze support vector distribution
    4. Study margin and decision boundary
    """
    print("\nExercise 3: Kernel SVM Implementation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class KernelSVM:
        """Kernel SVM with SMO optimization."""
        
        def __init__(self, kernel: Callable, C: float = 1.0):
            self.kernel = kernel
            self.C = C
        
        def fit(self, X: np.ndarray, y: np.ndarray, 
               max_iter: int = 1000, tol: float = 1e-5):
            """Fit SVM using simplified SMO."""
            n = len(y)
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            self.alpha = np.zeros(n)
            self.b = 0
            
            for _ in range(max_iter):
                alpha_changed = 0
                
                for i in range(n):
                    E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                    
                    if ((y[i] * E_i < -tol and self.alpha[i] < self.C) or
                        (y[i] * E_i > tol and self.alpha[i] > 0)):
                        
                        j = np.random.choice([k for k in range(n) if k != i])
                        E_j = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]
                        
                        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                        
                        if y[i] != y[j]:
                            L, H = max(0, self.alpha[j] - self.alpha[i]), min(self.C, self.C + self.alpha[j] - self.alpha[i])
                        else:
                            L, H = max(0, self.alpha[i] + self.alpha[j] - self.C), min(self.C, self.alpha[i] + self.alpha[j])
                        
                        if L >= H:
                            continue
                        
                        eta = 2 * K[i, j] - K[i, i] - K[j, j]
                        if eta >= 0:
                            continue
                        
                        self.alpha[j] = np.clip(self.alpha[j] - y[j] * (E_i - E_j) / eta, L, H)
                        
                        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                            continue
                        
                        self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                        
                        b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                        b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                        
                        self.b = b1 if 0 < self.alpha[i] < self.C else (b2 if 0 < self.alpha[j] < self.C else (b1 + b2) / 2)
                        
                        alpha_changed += 1
                
                if alpha_changed == 0:
                    break
            
            self.X_train = X
            self.y_train = y
            self.support_vectors = np.where(self.alpha > 1e-5)[0]
            
            return self
        
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            """Compute decision values."""
            return np.array([
                sum(self.alpha[i] * self.y_train[i] * self.kernel(self.X_train[i], x)
                    for i in self.support_vectors) + self.b
                for x in X
            ])
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return np.sign(self.decision_function(X))
        
        def margin(self) -> float:
            """Compute geometric margin."""
            sv_decision = self.decision_function(self.X_train[self.support_vectors])
            return 2 / np.linalg.norm(sv_decision)
    
    # XOR problem
    np.random.seed(42)
    X = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=float)
    X = X + 0.3 * np.random.randn(4, 2)
    y = np.array([1, 1, -1, -1])
    
    # Add more points
    for _ in range(10):
        X = np.vstack([
            X,
            [1 + 0.3*np.random.randn(), 1 + 0.3*np.random.randn()],
            [-1 + 0.3*np.random.randn(), -1 + 0.3*np.random.randn()],
            [-1 + 0.3*np.random.randn(), 1 + 0.3*np.random.randn()],
            [1 + 0.3*np.random.randn(), -1 + 0.3*np.random.randn()]
        ])
        y = np.append(y, [1, 1, -1, -1])
    
    print(f"\nXOR-like data: {len(X)} points")
    
    # Compare kernels
    kernels = {
        'Linear': lambda x, y: np.dot(x, y),
        'Polynomial (d=2)': lambda x, y: (np.dot(x, y) + 1)**2,
        'RBF (σ=0.5)': lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 0.5),
        'RBF (σ=1)': lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2),
    }
    
    print(f"\n{'Kernel':<20} {'Accuracy':<12} {'#SV':<8} {'SV%':<8}")
    print("-" * 50)
    
    for name, kernel in kernels.items():
        svm = KernelSVM(kernel, C=10)
        svm.fit(X, y)
        
        accuracy = np.mean(svm.predict(X) == y)
        n_sv = len(svm.support_vectors)
        sv_pct = 100 * n_sv / len(X)
        
        print(f"{name:<20} {accuracy:<12.4f} {n_sv:<8d} {sv_pct:<8.1f}%")
    
    return KernelSVM


# =============================================================================
# Exercise 4: Kernel Ridge Regression with CV
# =============================================================================

def exercise_4_kernel_ridge_cv():
    """
    Exercise: Kernel ridge regression with cross-validation.
    
    Tasks:
    1. Implement leave-one-out CV efficiently
    2. Grid search over kernel and regularization parameters
    3. Compare different kernels
    """
    print("\nExercise 4: Kernel Ridge Regression with CV")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class KernelRidgeCV:
        """Kernel Ridge Regression with efficient LOO-CV."""
        
        def __init__(self, kernel_fn: Callable):
            self.kernel_fn = kernel_fn
        
        def loo_cv_error(self, X: np.ndarray, y: np.ndarray, 
                        lambda_: float) -> float:
            """
            Efficient LOO-CV using: LOO_error = (α_i / G_{ii})²
            where G = (K + λI)^{-1}
            """
            n = len(X)
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel_fn(X[i], X[j])
            
            # Solve
            K_reg = K + lambda_ * np.eye(n)
            G = np.linalg.inv(K_reg)
            alpha = G @ y
            
            # LOO predictions
            loo_errors = (alpha / np.diag(G))**2
            return np.mean(loo_errors)
        
        def fit(self, X: np.ndarray, y: np.ndarray, 
               lambdas: np.ndarray) -> float:
            """Fit with best regularization."""
            best_lambda = None
            best_error = float('inf')
            
            for lam in lambdas:
                error = self.loo_cv_error(X, y, lam)
                if error < best_error:
                    best_error = error
                    best_lambda = lam
            
            self.lambda_ = best_lambda
            self.X_train = X
            
            # Fit with best lambda
            n = len(X)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel_fn(X[i], X[j])
            
            self.alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)
            
            return best_lambda, best_error
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict."""
            return np.array([
                sum(self.alpha[i] * self.kernel_fn(x, self.X_train[i])
                    for i in range(len(self.X_train)))
                for x in X
            ])
    
    # Generate data
    np.random.seed(42)
    n = 50
    X = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
    y = np.sin(X.flatten()) + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = np.sin(X_test.flatten())
    
    print(f"\nData: sin(x) + noise, {n} points")
    
    lambdas = np.logspace(-6, 2, 20)
    
    # Compare kernels with different bandwidths
    print(f"\n{'Kernel':<20} {'Best λ':<12} {'LOO-CV':<12} {'Test MSE':<12}")
    print("-" * 56)
    
    for sigma in [0.3, 0.5, 1.0, 2.0]:
        kernel = lambda x, y, s=sigma: np.exp(-np.linalg.norm(x - y)**2 / (2 * s**2))
        model = KernelRidgeCV(kernel)
        best_lam, best_err = model.fit(X, y, lambdas)
        
        y_pred = model.predict(X_test)
        test_mse = np.mean((y_pred - y_true)**2)
        
        print(f"RBF (σ={sigma}){'':<11} {best_lam:<12.2e} {best_err:<12.6f} {test_mse:<12.6f}")
    
    return KernelRidgeCV


# =============================================================================
# Exercise 5: Kernel PCA on Real Data
# =============================================================================

def exercise_5_kernel_pca():
    """
    Exercise: Apply kernel PCA to complex data.
    
    Tasks:
    1. Implement kernel PCA with centering
    2. Project new data points
    3. Compare linear PCA vs kernel PCA
    4. Reconstruct from kernel PCA (approximately)
    """
    print("\nExercise 5: Kernel PCA on Real Data")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class KernelPCAComplete:
        """Complete Kernel PCA with out-of-sample projection."""
        
        def __init__(self, kernel: Callable, n_components: int = 2):
            self.kernel = kernel
            self.n_components = n_components
        
        def fit(self, X: np.ndarray):
            """Fit kernel PCA."""
            self.X_train = X
            n = len(X)
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            self.K_train = K
            
            # Center kernel
            one_n = np.ones((n, n)) / n
            K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
            self.K_centered = K_centered
            
            # Store centering terms
            self.K_col_mean = K.mean(axis=0)
            self.K_mean = K.mean()
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
            idx = np.argsort(eigenvalues)[::-1]
            
            self.eigenvalues = eigenvalues[idx][:self.n_components]
            self.eigenvectors = eigenvectors[:, idx][:, :self.n_components]
            
            return self
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform data."""
            n = len(X)
            n_train = len(self.X_train)
            
            # Kernel with training data
            K = np.zeros((n, n_train))
            for i in range(n):
                for j in range(n_train):
                    K[i, j] = self.kernel(X[i], self.X_train[j])
            
            # Center
            K_centered = K - K.mean(axis=1, keepdims=True) - self.K_col_mean + self.K_mean
            
            # Project
            return K_centered @ self.eigenvectors / np.sqrt(np.maximum(self.eigenvalues, 1e-10))
        
        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            """Fit and transform."""
            self.fit(X)
            return self.K_centered @ self.eigenvectors / np.sqrt(np.maximum(self.eigenvalues, 1e-10))
        
        def explained_variance_ratio(self) -> np.ndarray:
            """Variance explained by each component."""
            total = np.sum(np.maximum(np.linalg.eigvalsh(self.K_centered), 0))
            return self.eigenvalues / (total + 1e-10)
    
    # Generate Swiss roll-like data
    np.random.seed(42)
    n = 200
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(n))
    height = 10 * np.random.rand(n)
    
    X = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ])
    
    print(f"\nSwiss roll data: {n} points in 3D")
    
    # Linear PCA
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_linear = X_centered @ Vt[:2].T
    
    linear_var = S[:2]**2 / np.sum(S**2)
    print(f"\nLinear PCA - Variance explained: {linear_var.sum():.4f}")
    
    # Kernel PCA
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 50)
    kpca = KernelPCAComplete(rbf, n_components=2)
    X_kpca = kpca.fit_transform(X)
    
    kernel_var = kpca.explained_variance_ratio()
    print(f"Kernel PCA - Variance explained: {kernel_var.sum():.4f}")
    
    # Quality measure: preservation of local structure
    def local_preservation(X_orig, X_embed, k=10):
        """Measure preservation of k-nearest neighbors."""
        from scipy.spatial.distance import cdist
        
        D_orig = cdist(X_orig, X_orig)
        D_embed = cdist(X_embed, X_embed)
        
        preservation = 0
        for i in range(len(X_orig)):
            nn_orig = set(np.argsort(D_orig[i])[1:k+1])
            nn_embed = set(np.argsort(D_embed[i])[1:k+1])
            preservation += len(nn_orig & nn_embed) / k
        
        return preservation / len(X_orig)
    
    linear_pres = local_preservation(X, X_linear)
    kernel_pres = local_preservation(X, X_kpca)
    
    print(f"\nLocal structure preservation (k=10):")
    print(f"  Linear PCA: {linear_pres:.4f}")
    print(f"  Kernel PCA: {kernel_pres:.4f}")
    
    return KernelPCAComplete


# =============================================================================
# Exercise 6: Gaussian Process with Hyperparameter Optimization
# =============================================================================

def exercise_6_gp_optimization():
    """
    Exercise: Gaussian Process with hyperparameter learning.
    
    Tasks:
    1. Implement GP with different kernels
    2. Optimize hyperparameters via marginal likelihood
    3. Compute predictive uncertainty
    4. Active learning with GP
    """
    print("\nExercise 6: Gaussian Process Optimization")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class GaussianProcessOptimized:
        """GP with hyperparameter optimization."""
        
        def __init__(self, noise_var: float = 0.01):
            self.noise_var = noise_var
            self.length_scale = 1.0
            self.signal_var = 1.0
        
        def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
            """RBF kernel with parameters."""
            return self.signal_var * np.exp(-np.linalg.norm(x - y)**2 / (2 * self.length_scale**2))
        
        def log_marginal_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
            """Compute log p(y | X, θ)."""
            n = len(X)
            
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            K += self.noise_var * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                
                data_fit = -0.5 * y @ alpha
                complexity = -np.sum(np.log(np.diag(L)))
                const = -0.5 * n * np.log(2 * np.pi)
                
                return data_fit + complexity + const
            except:
                return -np.inf
        
        def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                    length_scales: np.ndarray,
                                    signal_vars: np.ndarray):
            """Grid search over hyperparameters."""
            best_lml = -np.inf
            best_params = None
            
            for ls in length_scales:
                for sv in signal_vars:
                    self.length_scale = ls
                    self.signal_var = sv
                    
                    lml = self.log_marginal_likelihood(X, y)
                    if lml > best_lml:
                        best_lml = lml
                        best_params = (ls, sv)
            
            self.length_scale, self.signal_var = best_params
            return best_params, best_lml
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit GP."""
            self.X_train = X
            self.y_train = y
            n = len(X)
            
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            K += self.noise_var * np.eye(n)
            
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
            
            return self
        
        def predict(self, X: np.ndarray, return_std: bool = False):
            """Predict with uncertainty."""
            n_train = len(self.X_train)
            n_test = len(X)
            
            K_star = np.zeros((n_test, n_train))
            for i in range(n_test):
                for j in range(n_train):
                    K_star[i, j] = self.kernel(X[i], self.X_train[j])
            
            mean = K_star @ self.alpha
            
            if not return_std:
                return mean
            
            v = np.linalg.solve(self.L, K_star.T)
            K_star_star = np.array([self.kernel(X[i], X[i]) for i in range(n_test)])
            var = K_star_star - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(var, 0))
            
            return mean, std
        
        def acquisition_ucb(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
            """Upper confidence bound for active learning."""
            mean, std = self.predict(X, return_std=True)
            return mean + beta * std
    
    # Generate data
    np.random.seed(42)
    n = 15
    X_train = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    
    print(f"\nTraining data: {n} points")
    
    # Optimize hyperparameters
    gp = GaussianProcessOptimized(noise_var=0.01)
    
    length_scales = np.logspace(-1, 1, 10)
    signal_vars = np.logspace(-1, 1, 10)
    
    best_params, best_lml = gp.optimize_hyperparameters(X_train, y_train, length_scales, signal_vars)
    
    print(f"\nOptimized hyperparameters:")
    print(f"  Length scale: {best_params[0]:.4f}")
    print(f"  Signal variance: {best_params[1]:.4f}")
    print(f"  Log marginal likelihood: {best_lml:.4f}")
    
    # Fit and predict
    gp.fit(X_train, y_train)
    mean, std = gp.predict(X_test, return_std=True)
    
    # Active learning: find next point
    ucb = gp.acquisition_ucb(X_test)
    next_point = X_test[np.argmax(ucb)]
    
    print(f"\nActive learning: next point to sample = {next_point[0]:.4f}")
    print(f"  (Point with highest uncertainty in unexplored region)")
    
    return GaussianProcessOptimized


# =============================================================================
# Exercise 7: Random Fourier Features for Large-Scale Learning
# =============================================================================

def exercise_7_rff_large_scale():
    """
    Exercise: Large-scale kernel approximation.
    
    Tasks:
    1. Implement RFF for different kernels
    2. Combine with linear regression
    3. Compare computation time
    4. Analyze approximation quality vs D
    """
    print("\nExercise 7: Random Fourier Features - Large Scale")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class RFFRegression:
        """Regression with Random Fourier Features."""
        
        def __init__(self, d: int, D: int, sigma: float = 1.0, 
                    lambda_: float = 0.01):
            self.d = d
            self.D = D
            self.sigma = sigma
            self.lambda_ = lambda_
            
            # Sample random features for RBF kernel
            self.omega = np.random.randn(D, d) / sigma
            self.b = np.random.uniform(0, 2 * np.pi, D)
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Compute random features."""
            projection = X @ self.omega.T + self.b
            return np.sqrt(2 / self.D) * np.cos(projection)
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit linear model on features."""
            Z = self.transform(X)
            
            # Ridge regression: w = (Z^T Z + λI)^{-1} Z^T y
            self.w = np.linalg.solve(
                Z.T @ Z + self.lambda_ * np.eye(self.D),
                Z.T @ y
            )
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict."""
            Z = self.transform(X)
            return Z @ self.w
    
    def exact_kernel_ridge(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, sigma: float, 
                          lambda_: float) -> np.ndarray:
        """Exact kernel ridge regression."""
        n = len(X_train)
        
        # Kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-np.linalg.norm(X_train[i] - X_train[j])**2 / (2 * sigma**2))
        
        # Solve
        alpha = np.linalg.solve(K + lambda_ * np.eye(n), y_train)
        
        # Predict
        predictions = []
        for x in X_test:
            k = np.array([np.exp(-np.linalg.norm(x - xi)**2 / (2 * sigma**2)) for xi in X_train])
            predictions.append(k @ alpha)
        
        return np.array(predictions)
    
    import time
    
    # Generate data
    np.random.seed(42)
    d = 10
    n_train = 1000
    n_test = 200
    
    X_train = np.random.randn(n_train, d)
    y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]**2 + 0.1 * np.random.randn(n_train)
    
    X_test = np.random.randn(n_test, d)
    y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]**2
    
    print(f"\nData: {n_train} train, {n_test} test, {d} dimensions")
    
    sigma = 1.0
    lambda_ = 0.01
    
    # Exact (on subset due to O(n³))
    subset = 200
    start = time.time()
    y_pred_exact = exact_kernel_ridge(X_train[:subset], y_train[:subset], X_test, sigma, lambda_)
    exact_time = time.time() - start
    
    print(f"\n{'Method':<15} {'D/n':<10} {'Time (s)':<12} {'MSE':<12}")
    print("-" * 50)
    
    mse_exact = np.mean((y_pred_exact - y_test)**2)
    print(f"{'Exact (n=200)':<15} {'200':<10} {exact_time:<12.4f} {mse_exact:<12.6f}")
    
    # RFF with different D
    for D in [50, 100, 200, 500, 1000]:
        np.random.seed(42)  # Reproducible
        
        start = time.time()
        model = RFFRegression(d, D, sigma, lambda_)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rff_time = time.time() - start
        
        mse = np.mean((y_pred - y_test)**2)
        print(f"{'RFF':<15} {D:<10} {rff_time:<12.4f} {mse:<12.6f}")
    
    return RFFRegression


# =============================================================================
# Exercise 8: Nyström with Different Sampling Strategies
# =============================================================================

def exercise_8_nystrom_sampling():
    """
    Exercise: Compare Nyström sampling strategies.
    
    Tasks:
    1. Implement random sampling
    2. Implement k-means sampling
    3. Implement leverage score sampling
    4. Compare approximation quality
    """
    print("\nExercise 8: Nyström Sampling Strategies")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class NystromAdvanced:
        """Nyström approximation with different sampling strategies."""
        
        def __init__(self, kernel: Callable, n_landmarks: int, 
                    strategy: str = 'random'):
            self.kernel = kernel
            self.n_landmarks = n_landmarks
            self.strategy = strategy
        
        def _random_sampling(self, X: np.ndarray) -> np.ndarray:
            """Random uniform sampling."""
            idx = np.random.choice(len(X), self.n_landmarks, replace=False)
            return idx
        
        def _kmeans_sampling(self, X: np.ndarray) -> np.ndarray:
            """K-means++ style sampling."""
            n = len(X)
            idx = [np.random.randint(n)]
            
            for _ in range(self.n_landmarks - 1):
                # Distance to nearest selected point
                distances = np.array([
                    min(np.linalg.norm(X[i] - X[j]) for j in idx)
                    for i in range(n)
                ])
                
                # Probability proportional to squared distance
                probs = distances**2
                probs[idx] = 0  # Don't reselect
                probs /= probs.sum() + 1e-10
                
                new_idx = np.random.choice(n, p=probs)
                idx.append(new_idx)
            
            return np.array(idx)
        
        def _leverage_sampling(self, X: np.ndarray) -> np.ndarray:
            """Leverage score sampling."""
            n = len(X)
            
            # Compute full kernel matrix (expensive)
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            # Leverage scores from top eigenvectors
            k = min(self.n_landmarks, n)
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            idx = np.argsort(eigenvalues)[::-1][:k]
            V = eigenvectors[:, idx]
            
            # Leverage scores
            leverage = np.sum(V**2, axis=1)
            probs = leverage / leverage.sum()
            
            return np.random.choice(n, self.n_landmarks, replace=False, p=probs)
        
        def fit(self, X: np.ndarray):
            """Fit Nyström approximation."""
            # Select landmarks
            if self.strategy == 'random':
                self.landmark_idx = self._random_sampling(X)
            elif self.strategy == 'kmeans':
                self.landmark_idx = self._kmeans_sampling(X)
            elif self.strategy == 'leverage':
                self.landmark_idx = self._leverage_sampling(X)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self.landmarks = X[self.landmark_idx]
            
            # K_mm
            m = self.n_landmarks
            K_mm = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    K_mm[i, j] = self.kernel(self.landmarks[i], self.landmarks[j])
            
            # Pseudo-inverse
            eigvals, eigvecs = np.linalg.eigh(K_mm)
            eigvals = np.maximum(eigvals, 1e-10)
            self.K_mm_pinv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
            self.K_mm_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            return self
        
        def approximate_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
            """Compute approximate kernel matrix."""
            n = len(X)
            m = self.n_landmarks
            
            K_nm = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    K_nm[i, j] = self.kernel(X[i], self.landmarks[j])
            
            return K_nm @ self.K_mm_pinv @ K_nm.T
    
    # Test
    np.random.seed(42)
    n = 200
    d = 5
    X = np.random.randn(n, d)
    
    # RBF kernel
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2)
    
    # Exact kernel matrix
    K_exact = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_exact[i, j] = rbf(X[i], X[j])
    
    print(f"\nData: {n} points, {d} dimensions")
    
    print(f"\n{'Strategy':<12} {'m':<6} {'Rel Error':<12} {'Trace Error':<12}")
    print("-" * 50)
    
    for m in [10, 20, 50]:
        for strategy in ['random', 'kmeans', 'leverage']:
            np.random.seed(42)
            
            nystrom = NystromAdvanced(rbf, m, strategy)
            nystrom.fit(X)
            
            K_approx = nystrom.approximate_kernel_matrix(X)
            
            rel_error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
            trace_error = abs(np.trace(K_exact) - np.trace(K_approx)) / np.trace(K_exact)
            
            print(f"{strategy:<12} {m:<6} {rel_error:<12.4f} {trace_error:<12.4f}")
    
    return NystromAdvanced


# =============================================================================
# Exercise 9: Kernel Methods for Density Estimation
# =============================================================================

def exercise_9_kernel_density():
    """
    Exercise: Kernel density estimation.
    
    Tasks:
    1. Implement Parzen window density estimator
    2. Bandwidth selection via cross-validation
    3. Compare with Gaussian mixture model
    """
    print("\nExercise 9: Kernel Density Estimation")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class KernelDensityEstimator:
        """Kernel density estimation with bandwidth selection."""
        
        def __init__(self, bandwidth: float = 1.0):
            self.bandwidth = bandwidth
        
        def fit(self, X: np.ndarray):
            """Store training data."""
            self.X_train = X
            self.n, self.d = X.shape
            return self
        
        def _kernel(self, x: np.ndarray) -> float:
            """Multivariate Gaussian kernel."""
            return np.exp(-np.sum(x**2) / 2) / np.sqrt(2 * np.pi)**self.d
        
        def evaluate(self, X: np.ndarray) -> np.ndarray:
            """Evaluate density at points."""
            densities = []
            
            for x in X:
                density = 0
                for xi in self.X_train:
                    density += self._kernel((x - xi) / self.bandwidth)
                density /= (self.n * self.bandwidth**self.d)
                densities.append(density)
            
            return np.array(densities)
        
        def log_likelihood(self, X: np.ndarray) -> float:
            """Compute log likelihood."""
            densities = self.evaluate(X)
            return np.sum(np.log(densities + 1e-10))
        
        def loo_cv_likelihood(self) -> float:
            """Leave-one-out cross-validation likelihood."""
            log_lik = 0
            
            for i in range(self.n):
                # Density at x_i using all other points
                density = 0
                for j in range(self.n):
                    if i != j:
                        density += self._kernel(
                            (self.X_train[i] - self.X_train[j]) / self.bandwidth
                        )
                density /= ((self.n - 1) * self.bandwidth**self.d)
                log_lik += np.log(density + 1e-10)
            
            return log_lik
        
        @staticmethod
        def select_bandwidth(X: np.ndarray, 
                           bandwidths: np.ndarray) -> Tuple[float, float]:
            """Select bandwidth via LOO-CV."""
            best_h = None
            best_lik = -np.inf
            
            for h in bandwidths:
                kde = KernelDensityEstimator(h)
                kde.fit(X)
                lik = kde.loo_cv_likelihood()
                
                if lik > best_lik:
                    best_lik = lik
                    best_h = h
            
            return best_h, best_lik
    
    # Generate bimodal data
    np.random.seed(42)
    n = 200
    
    X1 = np.random.randn(n//2, 2) + np.array([-2, 0])
    X2 = np.random.randn(n//2, 2) + np.array([2, 0])
    X = np.vstack([X1, X2])
    
    print(f"\nBimodal data: {n} points in 2D")
    
    # Bandwidth selection
    bandwidths = np.linspace(0.3, 2.0, 20)
    best_h, best_lik = KernelDensityEstimator.select_bandwidth(X, bandwidths)
    
    print(f"\nOptimal bandwidth (LOO-CV): h = {best_h:.4f}")
    
    # Compare different bandwidths
    print(f"\n{'Bandwidth':<12} {'LOO-CV LL':<15}")
    print("-" * 30)
    
    for h in [0.3, 0.5, best_h, 1.0, 2.0]:
        kde = KernelDensityEstimator(h)
        kde.fit(X)
        lik = kde.loo_cv_likelihood()
        print(f"{h:<12.2f} {lik:<15.4f}")
    
    # Evaluate at grid
    kde_optimal = KernelDensityEstimator(best_h)
    kde_optimal.fit(X)
    
    # Test points
    X_test = np.array([[-2, 0], [0, 0], [2, 0]])
    densities = kde_optimal.evaluate(X_test)
    
    print(f"\nDensity at key points:")
    for x, d in zip(X_test, densities):
        print(f"  {x} -> {d:.6f}")
    
    return KernelDensityEstimator


# =============================================================================
# Exercise 10: Complete Kernel Methods Pipeline
# =============================================================================

def exercise_10_complete_pipeline():
    """
    Exercise: Build complete kernel learning pipeline.
    
    Tasks:
    1. Data preprocessing
    2. Kernel selection and combination
    3. Model fitting with hyperparameter tuning
    4. Evaluation and comparison
    """
    print("\nExercise 10: Complete Kernel Methods Pipeline")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Solution
    # -------------------------------------------------------------------------
    
    class KernelMethodsPipeline:
        """Complete pipeline for kernel methods."""
        
        def __init__(self):
            self.scaler_mean = None
            self.scaler_std = None
            self.best_model = None
        
        def preprocess(self, X_train: np.ndarray, X_test: np.ndarray = None):
            """Standardize data."""
            self.scaler_mean = X_train.mean(axis=0)
            self.scaler_std = X_train.std(axis=0) + 1e-10
            
            X_train_scaled = (X_train - self.scaler_mean) / self.scaler_std
            
            if X_test is not None:
                X_test_scaled = (X_test - self.scaler_mean) / self.scaler_std
                return X_train_scaled, X_test_scaled
            
            return X_train_scaled
        
        def create_kernel(self, name: str, **params) -> Callable:
            """Create kernel function."""
            if name == 'rbf':
                sigma = params.get('sigma', 1.0)
                return lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
            elif name == 'polynomial':
                degree = params.get('degree', 2)
                c = params.get('c', 1.0)
                return lambda x, y: (np.dot(x, y) + c)**degree
            elif name == 'laplacian':
                sigma = params.get('sigma', 1.0)
                return lambda x, y: np.exp(-np.linalg.norm(x - y, 1) / sigma)
            else:
                raise ValueError(f"Unknown kernel: {name}")
        
        def kernel_ridge_fit(self, X: np.ndarray, y: np.ndarray,
                           kernel: Callable, lambda_: float):
            """Fit kernel ridge regression."""
            n = len(X)
            
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = kernel(X[i], X[j])
            
            alpha = np.linalg.solve(K + lambda_ * np.eye(n), y)
            
            return alpha, K
        
        def kernel_ridge_predict(self, X_train: np.ndarray, X_test: np.ndarray,
                                alpha: np.ndarray, kernel: Callable):
            """Predict with kernel ridge."""
            predictions = []
            for x in X_test:
                k = np.array([kernel(x, xi) for xi in X_train])
                predictions.append(k @ alpha)
            return np.array(predictions)
        
        def cross_validate(self, X: np.ndarray, y: np.ndarray,
                          kernel: Callable, lambda_: float, k: int = 5) -> float:
            """K-fold cross-validation."""
            n = len(X)
            indices = np.random.permutation(n)
            fold_size = n // k
            
            errors = []
            
            for i in range(k):
                val_idx = indices[i*fold_size:(i+1)*fold_size]
                train_idx = np.concatenate([indices[:i*fold_size], indices[(i+1)*fold_size:]])
                
                X_train_fold, y_train_fold = X[train_idx], y[train_idx]
                X_val_fold, y_val_fold = X[val_idx], y[val_idx]
                
                alpha, _ = self.kernel_ridge_fit(X_train_fold, y_train_fold, kernel, lambda_)
                y_pred = self.kernel_ridge_predict(X_train_fold, X_val_fold, alpha, kernel)
                
                mse = np.mean((y_pred - y_val_fold)**2)
                errors.append(mse)
            
            return np.mean(errors)
        
        def grid_search(self, X: np.ndarray, y: np.ndarray,
                       kernel_configs: List[Dict], 
                       lambdas: np.ndarray) -> Dict:
            """Grid search over kernels and regularization."""
            best_config = None
            best_error = float('inf')
            
            results = []
            
            for config in kernel_configs:
                kernel = self.create_kernel(**config)
                
                for lam in lambdas:
                    cv_error = self.cross_validate(X, y, kernel, lam)
                    
                    results.append({
                        'config': config,
                        'lambda': lam,
                        'cv_error': cv_error
                    })
                    
                    if cv_error < best_error:
                        best_error = cv_error
                        best_config = {'config': config, 'lambda': lam}
            
            return best_config, results
        
        def fit(self, X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray = None, y_test: np.ndarray = None):
            """Complete pipeline: preprocess, tune, fit, evaluate."""
            
            # Preprocess
            if X_test is not None:
                X_train_scaled, X_test_scaled = self.preprocess(X_train, X_test)
            else:
                X_train_scaled = self.preprocess(X_train)
            
            # Define search space
            kernel_configs = [
                {'name': 'rbf', 'sigma': 0.5},
                {'name': 'rbf', 'sigma': 1.0},
                {'name': 'rbf', 'sigma': 2.0},
                {'name': 'polynomial', 'degree': 2},
                {'name': 'polynomial', 'degree': 3},
                {'name': 'laplacian', 'sigma': 1.0},
            ]
            lambdas = np.logspace(-4, 1, 10)
            
            # Grid search
            best_config, results = self.grid_search(X_train_scaled, y_train, 
                                                    kernel_configs, lambdas)
            
            # Fit best model
            best_kernel = self.create_kernel(**best_config['config'])
            alpha, K = self.kernel_ridge_fit(X_train_scaled, y_train, 
                                            best_kernel, best_config['lambda'])
            
            self.X_train_scaled = X_train_scaled
            self.alpha = alpha
            self.best_kernel = best_kernel
            self.best_config = best_config
            
            # Evaluate
            result = {
                'best_config': best_config,
                'train_mse': None,
                'test_mse': None
            }
            
            y_train_pred = self.kernel_ridge_predict(X_train_scaled, X_train_scaled,
                                                     alpha, best_kernel)
            result['train_mse'] = np.mean((y_train_pred - y_train)**2)
            
            if X_test is not None and y_test is not None:
                y_test_pred = self.kernel_ridge_predict(X_train_scaled, X_test_scaled,
                                                        alpha, best_kernel)
                result['test_mse'] = np.mean((y_test_pred - y_test)**2)
            
            return result
    
    # Generate data
    np.random.seed(42)
    n_train, n_test = 100, 50
    d = 5
    
    X_train = np.random.randn(n_train, d)
    y_train = (np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1]**2 + 
              0.3 * X_train[:, 2] * X_train[:, 3] + 0.1 * np.random.randn(n_train))
    
    X_test = np.random.randn(n_test, d)
    y_test = (np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1]**2 + 
             0.3 * X_test[:, 2] * X_test[:, 3])
    
    print(f"\nData: {n_train} train, {n_test} test, {d} features")
    
    # Run pipeline
    pipeline = KernelMethodsPipeline()
    result = pipeline.fit(X_train, y_train, X_test, y_test)
    
    print(f"\nResults:")
    print(f"  Best kernel: {result['best_config']['config']['name']}")
    print(f"  Parameters: {result['best_config']['config']}")
    print(f"  Best λ: {result['best_config']['lambda']:.2e}")
    print(f"  Train MSE: {result['train_mse']:.6f}")
    print(f"  Test MSE: {result['test_mse']:.6f}")
    
    return KernelMethodsPipeline


def run_all_exercises():
    """Run all kernel method exercises."""
    print("=" * 70)
    print("KERNEL METHODS - EXERCISES")
    print("=" * 70)
    
    exercise_1_custom_kernel()
    exercise_2_kernel_matrix()
    exercise_3_kernel_svm()
    exercise_4_kernel_ridge_cv()
    exercise_5_kernel_pca()
    exercise_6_gp_optimization()
    exercise_7_rff_large_scale()
    exercise_8_nystrom_sampling()
    exercise_9_kernel_density()
    exercise_10_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All kernel method exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_exercises()
