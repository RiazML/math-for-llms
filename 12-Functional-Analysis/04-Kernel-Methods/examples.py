"""
Kernel Methods - Examples
=========================

Implementations demonstrating kernel method concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# Example 1: Kernel Functions
# =============================================================================

def example_1_kernels():
    """
    Implement and compare common kernel functions.
    """
    print("Example 1: Kernel Functions")
    print("=" * 60)
    
    def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
        """K(x, y) = x^T y"""
        return np.dot(x, y)
    
    def polynomial_kernel(x: np.ndarray, y: np.ndarray, 
                         degree: int = 2, c: float = 1.0) -> float:
        """K(x, y) = (x^T y + c)^d"""
        return (np.dot(x, y) + c) ** degree
    
    def rbf_kernel(x: np.ndarray, y: np.ndarray, 
                  sigma: float = 1.0) -> float:
        """K(x, y) = exp(-||x - y||² / 2σ²)"""
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
    
    def laplacian_kernel(x: np.ndarray, y: np.ndarray,
                        sigma: float = 1.0) -> float:
        """K(x, y) = exp(-||x - y||₁ / σ)"""
        return np.exp(-np.linalg.norm(x - y, 1) / sigma)
    
    def matern_kernel(x: np.ndarray, y: np.ndarray,
                     nu: float = 1.5, length_scale: float = 1.0) -> float:
        """
        Matérn kernel.
        ν = 1/2: Laplacian
        ν = 3/2: Once differentiable
        ν = 5/2: Twice differentiable
        """
        from scipy.special import gamma, kv
        
        d = np.linalg.norm(x - y)
        if d < 1e-10:
            return 1.0
        
        # Special cases
        if nu == 0.5:
            return np.exp(-d / length_scale)
        elif nu == 1.5:
            arg = np.sqrt(3) * d / length_scale
            return (1 + arg) * np.exp(-arg)
        elif nu == 2.5:
            arg = np.sqrt(5) * d / length_scale
            return (1 + arg + arg**2 / 3) * np.exp(-arg)
        else:
            # General case
            arg = np.sqrt(2 * nu) * d / length_scale
            return (2**(1-nu) / gamma(nu)) * (arg**nu) * kv(nu, arg)
    
    def gram_matrix(X: np.ndarray, kernel: Callable) -> np.ndarray:
        """Compute Gram matrix K_ij = K(x_i, x_j)."""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j])
        return K
    
    def verify_psd(K: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Check positive semi-definiteness."""
        eigenvalues = np.linalg.eigvalsh(K)
        return np.all(eigenvalues >= -1e-10), eigenvalues
    
    # Test data
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    
    print("Test points:")
    print(X)
    
    kernels = {
        'Linear': linear_kernel,
        'Polynomial (d=2)': lambda x, y: polynomial_kernel(x, y, 2),
        'RBF (σ=1)': lambda x, y: rbf_kernel(x, y, 1),
        'Laplacian (σ=1)': lambda x, y: laplacian_kernel(x, y, 1),
        'Matérn (ν=1.5)': lambda x, y: matern_kernel(x, y, 1.5)
    }
    
    print("\nKernel matrices:")
    for name, kernel in kernels.items():
        K = gram_matrix(X, kernel)
        is_psd, eigs = verify_psd(K)
        
        print(f"\n{name}:")
        print(K.round(4))
        print(f"PSD: {is_psd}, eigenvalues: {eigs.round(4)}")
    
    return gram_matrix, rbf_kernel


# =============================================================================
# Example 2: Kernel Trick Demonstration
# =============================================================================

def example_2_kernel_trick():
    """
    Demonstrate the kernel trick: same result, different computation.
    """
    print("\nExample 2: Kernel Trick Demonstration")
    print("=" * 60)
    
    def explicit_polynomial_features(x: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Compute polynomial features explicitly.
        For d=2: [x1², √2 x1x2, x2², √2 x1, √2 x2, 1]
        """
        if degree != 2:
            raise NotImplementedError("Only degree 2 implemented")
        
        x1, x2 = x[0], x[1]
        return np.array([
            x1**2,
            np.sqrt(2) * x1 * x2,
            x2**2,
            np.sqrt(2) * x1,
            np.sqrt(2) * x2,
            1
        ])
    
    def polynomial_kernel_direct(x: np.ndarray, y: np.ndarray) -> float:
        """K(x, y) = (x^T y + 1)^2"""
        return (np.dot(x, y) + 1) ** 2
    
    def polynomial_kernel_via_features(x: np.ndarray, y: np.ndarray) -> float:
        """K(x, y) = φ(x)^T φ(y)"""
        phi_x = explicit_polynomial_features(x)
        phi_y = explicit_polynomial_features(y)
        return np.dot(phi_x, phi_y)
    
    # Test
    x = np.array([1.0, 2.0])
    y = np.array([3.0, 4.0])
    
    K_direct = polynomial_kernel_direct(x, y)
    K_features = polynomial_kernel_via_features(x, y)
    
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"\nPolynomial kernel (direct): K(x,y) = (x^Ty + 1)² = {K_direct}")
    print(f"Via explicit features: φ(x)^Tφ(y) = {K_features}")
    print(f"Match: {np.isclose(K_direct, K_features)}")
    
    print(f"\nFeature dimensions:")
    print(f"  Original: {len(x)}")
    print(f"  Polynomial features: {len(explicit_polynomial_features(x))}")
    
    # RBF has infinite-dimensional features
    print(f"\nRBF kernel: infinite-dimensional feature space!")
    print("  But K(x,y) computes in O(d) time")
    
    return polynomial_kernel_direct


# =============================================================================
# Example 3: Support Vector Machine
# =============================================================================

def example_3_svm():
    """
    Implement SVM with kernel using SMO-like optimization.
    """
    print("\nExample 3: Support Vector Machine")
    print("=" * 60)
    
    class KernelSVM:
        """Simple kernel SVM implementation."""
        
        def __init__(self, kernel: Callable, C: float = 1.0):
            self.kernel = kernel
            self.C = C
            self.alpha = None
            self.b = None
            self.X_train = None
            self.y_train = None
            self.support_vectors = None
        
        def fit(self, X: np.ndarray, y: np.ndarray, 
               max_iter: int = 1000, tol: float = 1e-5):
            """
            Fit SVM using simplified SMO algorithm.
            
            Dual: max Σα_i - (1/2)Σα_i α_j y_i y_j K(x_i, x_j)
            s.t. 0 <= α_i <= C, Σα_i y_i = 0
            """
            n = len(y)
            self.X_train = X
            self.y_train = y
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            # Initialize
            self.alpha = np.zeros(n)
            self.b = 0
            
            for iteration in range(max_iter):
                alpha_changed = 0
                
                for i in range(n):
                    # Error for i
                    E_i = self._predict_raw(X[i], K, i) - y[i]
                    
                    # Check KKT conditions
                    if ((y[i] * E_i < -tol and self.alpha[i] < self.C) or
                        (y[i] * E_i > tol and self.alpha[i] > 0)):
                        
                        # Select j != i randomly
                        j = np.random.choice([k for k in range(n) if k != i])
                        E_j = self._predict_raw(X[j], K, j) - y[j]
                        
                        # Save old alphas
                        alpha_i_old = self.alpha[i]
                        alpha_j_old = self.alpha[j]
                        
                        # Compute bounds
                        if y[i] != y[j]:
                            L = max(0, self.alpha[j] - self.alpha[i])
                            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                        else:
                            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                            H = min(self.C, self.alpha[i] + self.alpha[j])
                        
                        if L >= H:
                            continue
                        
                        # Compute eta
                        eta = 2 * K[i, j] - K[i, i] - K[j, j]
                        if eta >= 0:
                            continue
                        
                        # Update alpha_j
                        self.alpha[j] -= y[j] * (E_i - E_j) / eta
                        self.alpha[j] = np.clip(self.alpha[j], L, H)
                        
                        if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                            continue
                        
                        # Update alpha_i
                        self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                        
                        # Update b
                        b1 = (self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i]
                              - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j])
                        b2 = (self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j]
                              - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j])
                        
                        if 0 < self.alpha[i] < self.C:
                            self.b = b1
                        elif 0 < self.alpha[j] < self.C:
                            self.b = b2
                        else:
                            self.b = (b1 + b2) / 2
                        
                        alpha_changed += 1
                
                if alpha_changed == 0:
                    break
            
            # Identify support vectors
            self.support_vectors = np.where(self.alpha > 1e-5)[0]
            
            return self
        
        def _predict_raw(self, x: np.ndarray, K: np.ndarray = None, 
                        idx: int = None) -> float:
            """Predict raw value (before sign)."""
            if K is not None and idx is not None:
                return np.sum(self.alpha * self.y_train * K[:, idx]) + self.b
            else:
                result = self.b
                for i in range(len(self.X_train)):
                    if self.alpha[i] > 1e-5:
                        result += self.alpha[i] * self.y_train[i] * self.kernel(self.X_train[i], x)
                return result
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict class labels."""
            return np.array([np.sign(self._predict_raw(x)) for x in X])
        
        def decision_function(self, X: np.ndarray) -> np.ndarray:
            """Return decision values."""
            return np.array([self._predict_raw(x) for x in X])
    
    # Generate linearly separable data
    np.random.seed(42)
    n = 50
    
    X_pos = np.random.randn(n//2, 2) + np.array([2, 2])
    X_neg = np.random.randn(n//2, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*(n//2) + [-1]*(n//2))
    
    print(f"Training data: {n} points, 2 classes")
    
    # Train with RBF kernel
    rbf = lambda x, x_: np.exp(-np.linalg.norm(x - x_)**2 / 2)
    svm = KernelSVM(rbf, C=1.0)
    svm.fit(X, y)
    
    accuracy = np.mean(svm.predict(X) == y)
    
    print(f"\nRBF SVM:")
    print(f"  Support vectors: {len(svm.support_vectors)} / {n}")
    print(f"  Training accuracy: {accuracy:.4f}")
    
    return KernelSVM


# =============================================================================
# Example 4: Kernel Ridge Regression
# =============================================================================

def example_4_kernel_ridge():
    """
    Implement kernel ridge regression.
    """
    print("\nExample 4: Kernel Ridge Regression")
    print("=" * 60)
    
    class KernelRidge:
        """Kernel Ridge Regression."""
        
        def __init__(self, kernel: Callable, lambda_: float = 0.1):
            self.kernel = kernel
            self.lambda_ = lambda_
            self.X_train = None
            self.alpha = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """
            Fit using closed-form solution:
            α = (K + λI)^{-1} y
            """
            self.X_train = X
            n = len(X)
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            # Solve
            self.alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)
            self.K_train = K
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict using kernel expansion."""
            predictions = []
            for x in X:
                k = np.array([self.kernel(x, xi) for xi in self.X_train])
                predictions.append(k @ self.alpha)
            return np.array(predictions)
        
        def rkhs_norm_squared(self) -> float:
            """||f||_H² = α^T K α"""
            return self.alpha @ self.K_train @ self.alpha
    
    # Generate nonlinear data
    np.random.seed(42)
    n = 50
    X_train = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = np.sin(X_test.flatten())
    
    print(f"Fitting sin(x) with {n} noisy samples")
    
    # Compare kernels
    kernels = {
        'RBF (σ=0.5)': lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 0.5),
        'RBF (σ=2)': lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 8),
        'Polynomial (d=5)': lambda x, y: (np.dot(x, y) + 1)**5,
    }
    
    for name, kernel in kernels.items():
        model = KernelRidge(kernel, lambda_=0.01)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = np.mean((y_pred - y_true)**2)
        
        print(f"\n{name}:")
        print(f"  Test MSE: {mse:.6f}")
        print(f"  ||f||_H²: {model.rkhs_norm_squared():.4f}")
    
    return KernelRidge


# =============================================================================
# Example 5: Kernel PCA
# =============================================================================

def example_5_kernel_pca():
    """
    Implement kernel PCA for nonlinear dimensionality reduction.
    """
    print("\nExample 5: Kernel PCA")
    print("=" * 60)
    
    class KernelPCA:
        """Kernel Principal Component Analysis."""
        
        def __init__(self, kernel: Callable, n_components: int = 2):
            self.kernel = kernel
            self.n_components = n_components
        
        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            """Fit and transform data."""
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
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
            
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select components
            self.eigenvalues = eigenvalues[:self.n_components]
            self.eigenvectors = eigenvectors[:, :self.n_components]
            
            # Project
            return K_centered @ self.eigenvectors / np.sqrt(self.eigenvalues + 1e-10)
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Transform new data."""
            n = len(X)
            n_train = len(self.X_train)
            
            # Kernel with training data
            K = np.zeros((n, n_train))
            for i in range(n):
                for j in range(n_train):
                    K[i, j] = self.kernel(X[i], self.X_train[j])
            
            # Center
            one_n = np.ones((n, n_train)) / n_train
            one_train = np.ones((n_train, n_train)) / n_train
            K_centered = K - one_n @ self.K_train - K @ one_train + one_n @ self.K_train @ one_train
            
            return K_centered @ self.eigenvectors / np.sqrt(self.eigenvalues + 1e-10)
    
    # Generate nonlinear data (concentric circles)
    np.random.seed(42)
    n = 200
    
    # Inner circle
    theta1 = np.random.uniform(0, 2*np.pi, n//2)
    X1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
    y1 = np.zeros(n//2)
    
    # Outer circle
    theta2 = np.random.uniform(0, 2*np.pi, n//2)
    X2 = 3 * np.column_stack([np.cos(theta2), np.sin(theta2)])
    y2 = np.ones(n//2)
    
    X = np.vstack([X1, X2]) + 0.1 * np.random.randn(n, 2)
    y = np.concatenate([y1, y2])
    
    print(f"Data: {n} points, 2 concentric circles")
    
    # Linear PCA
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_linear = X_centered @ Vt.T
    
    print("\nLinear PCA: cannot separate circles (they're already 2D)")
    
    # Kernel PCA with RBF
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2)
    kpca = KernelPCA(rbf, n_components=2)
    X_kpca = kpca.fit_transform(X)
    
    print(f"\nKernel PCA (RBF):")
    print(f"  Eigenvalues: {kpca.eigenvalues}")
    
    # Check separability
    class0_mean = X_kpca[y == 0].mean(axis=0)
    class1_mean = X_kpca[y == 1].mean(axis=0)
    separation = np.linalg.norm(class0_mean - class1_mean)
    print(f"  Class separation: {separation:.4f}")
    
    return KernelPCA


# =============================================================================
# Example 6: Gaussian Process Regression
# =============================================================================

def example_6_gaussian_process():
    """
    Implement Gaussian Process regression.
    """
    print("\nExample 6: Gaussian Process Regression")
    print("=" * 60)
    
    class GaussianProcess:
        """Gaussian Process with RBF kernel."""
        
        def __init__(self, kernel: Callable, noise_var: float = 0.01):
            self.kernel = kernel
            self.noise_var = noise_var
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Store training data and compute K inverse."""
            self.X_train = X
            self.y_train = y
            n = len(X)
            
            # Kernel matrix + noise
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            K += self.noise_var * np.eye(n)
            
            # Store for prediction
            self.K = K
            self.L = np.linalg.cholesky(K)
            self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y))
            
            return self
        
        def predict(self, X: np.ndarray, return_std: bool = False):
            """Predict mean and optionally std."""
            n_train = len(self.X_train)
            n_test = len(X)
            
            # K_* (test vs train)
            K_star = np.zeros((n_test, n_train))
            for i in range(n_test):
                for j in range(n_train):
                    K_star[i, j] = self.kernel(X[i], self.X_train[j])
            
            # Mean: K_* α
            mean = K_star @ self.alpha
            
            if not return_std:
                return mean
            
            # K_** (test vs test)
            K_star_star = np.zeros((n_test, n_test))
            for i in range(n_test):
                for j in range(n_test):
                    K_star_star[i, j] = self.kernel(X[i], X[j])
            
            # Variance: K_** - K_* K^{-1} K_*^T
            v = np.linalg.solve(self.L, K_star.T)
            var = K_star_star - v.T @ v
            std = np.sqrt(np.diag(var) + 1e-10)
            
            return mean, std
        
        def log_marginal_likelihood(self) -> float:
            """Compute log marginal likelihood for hyperparameter tuning."""
            n = len(self.y_train)
            
            # -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)
            data_fit = -0.5 * self.y_train @ self.alpha
            complexity = -np.sum(np.log(np.diag(self.L)))
            const = -0.5 * n * np.log(2 * np.pi)
            
            return data_fit + complexity + const
    
    # Generate data
    np.random.seed(42)
    n = 20
    X_train = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
    y_train = np.sin(X_train.flatten()) + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    
    print(f"Training GP on {n} points")
    
    # Different kernel bandwidths
    for sigma in [0.5, 1.0, 2.0]:
        kernel = lambda x, y, s=sigma: np.exp(-np.linalg.norm(x - y)**2 / (2 * s**2))
        
        gp = GaussianProcess(kernel, noise_var=0.01)
        gp.fit(X_train, y_train)
        
        mean, std = gp.predict(X_test, return_std=True)
        lml = gp.log_marginal_likelihood()
        
        print(f"\nσ = {sigma}:")
        print(f"  Log marginal likelihood: {lml:.4f}")
        print(f"  Mean uncertainty: {np.mean(std):.4f}")
    
    return GaussianProcess


# =============================================================================
# Example 7: Random Fourier Features
# =============================================================================

def example_7_random_fourier():
    """
    Approximate RBF kernel with random Fourier features.
    """
    print("\nExample 7: Random Fourier Features")
    print("=" * 60)
    
    def rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
        """Exact RBF kernel."""
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
    
    class RandomFourierFeatures:
        """
        Approximate RBF kernel using random Fourier features.
        
        K(x, y) ≈ z(x)^T z(y)
        z(x) = √(2/D) [cos(ω₁^T x + b₁), ..., cos(ω_D^T x + b_D)]
        """
        
        def __init__(self, d: int, D: int, sigma: float = 1.0):
            """
            Args:
                d: Input dimension
                D: Number of random features
                sigma: RBF bandwidth
            """
            self.d = d
            self.D = D
            self.sigma = sigma
            
            # Sample from spectral density (normal for RBF)
            self.omega = np.random.randn(D, d) / sigma
            self.b = np.random.uniform(0, 2 * np.pi, D)
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Compute random features."""
            projection = X @ self.omega.T + self.b
            return np.sqrt(2 / self.D) * np.cos(projection)
        
        def approximate_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
            """Approximate kernel via inner product."""
            z_x = self.transform(x.reshape(1, -1))[0]
            z_y = self.transform(y.reshape(1, -1))[0]
            return np.dot(z_x, z_y)
    
    # Compare approximation quality
    np.random.seed(42)
    d = 5  # Input dimension
    sigma = 1.0
    
    # Test points
    n = 50
    X = np.random.randn(n, d)
    
    # Exact kernel matrix
    K_exact = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_exact[i, j] = rbf_kernel(X[i], X[j], sigma)
    
    print(f"Input dimension: {d}")
    print(f"Test points: {n}")
    
    print("\nApproximation error (Frobenius norm):")
    for D in [10, 50, 100, 500, 1000]:
        rff = RandomFourierFeatures(d, D, sigma)
        
        # Approximate kernel matrix
        Z = rff.transform(X)
        K_approx = Z @ Z.T
        
        rel_error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        
        print(f"  D = {D:4d}: relative error = {rel_error:.4f}")
    
    return RandomFourierFeatures


# =============================================================================
# Example 8: Nyström Approximation
# =============================================================================

def example_8_nystrom():
    """
    Implement Nyström kernel approximation.
    """
    print("\nExample 8: Nyström Approximation")
    print("=" * 60)
    
    class NystromApproximation:
        """
        Nyström low-rank kernel approximation.
        
        K ≈ K_nm K_mm^{-1} K_mn
        """
        
        def __init__(self, kernel: Callable, n_landmarks: int):
            self.kernel = kernel
            self.n_landmarks = n_landmarks
        
        def fit(self, X: np.ndarray):
            """Select landmarks and compute approximation."""
            n = len(X)
            
            # Random landmark selection
            np.random.seed(42)
            self.landmark_idx = np.random.choice(n, self.n_landmarks, replace=False)
            self.landmarks = X[self.landmark_idx]
            
            # K_mm (landmark kernel matrix)
            K_mm = np.zeros((self.n_landmarks, self.n_landmarks))
            for i in range(self.n_landmarks):
                for j in range(self.n_landmarks):
                    K_mm[i, j] = self.kernel(self.landmarks[i], self.landmarks[j])
            
            # Eigendecomposition for numerical stability
            eigvals, eigvecs = np.linalg.eigh(K_mm)
            eigvals = np.maximum(eigvals, 1e-10)
            
            # K_mm^{-1/2}
            self.K_mm_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            return self
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            """Compute approximate features."""
            n = len(X)
            
            # K_nm (data to landmarks)
            K_nm = np.zeros((n, self.n_landmarks))
            for i in range(n):
                for j in range(self.n_landmarks):
                    K_nm[i, j] = self.kernel(X[i], self.landmarks[j])
            
            # Approximate features: K_nm K_mm^{-1/2}
            return K_nm @ self.K_mm_sqrt_inv
        
        def approximate_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
            """Compute approximate kernel matrix."""
            Z = self.transform(X)
            return Z @ Z.T
    
    # Test
    np.random.seed(42)
    n = 100
    d = 5
    X = np.random.randn(n, d)
    
    # RBF kernel
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2)
    
    # Exact kernel matrix
    K_exact = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_exact[i, j] = rbf(X[i], X[j])
    
    print(f"Data: {n} points, {d} dimensions")
    
    print("\nNyström approximation errors:")
    for m in [5, 10, 20, 50]:
        nystrom = NystromApproximation(rbf, m)
        nystrom.fit(X)
        
        K_approx = nystrom.approximate_kernel_matrix(X)
        rel_error = np.linalg.norm(K_exact - K_approx) / np.linalg.norm(K_exact)
        
        print(f"  m = {m:2d} landmarks: relative error = {rel_error:.4f}")
    
    return NystromApproximation


# =============================================================================
# Example 9: Multiple Kernel Learning
# =============================================================================

def example_9_mkl():
    """
    Simple multiple kernel learning by combining kernels.
    """
    print("\nExample 9: Multiple Kernel Learning")
    print("=" * 60)
    
    class SimpleMultipleKernelLearning:
        """
        Learn optimal kernel combination:
        K = Σ μ_m K_m, where μ_m >= 0, Σμ_m = 1
        """
        
        def __init__(self, kernels: List[Callable], lambda_: float = 0.1):
            self.kernels = kernels
            self.lambda_ = lambda_
            self.n_kernels = len(kernels)
        
        def fit(self, X: np.ndarray, y: np.ndarray, 
               max_iter: int = 100, lr: float = 0.1):
            """Fit with gradient descent on kernel weights."""
            n = len(X)
            
            # Precompute individual kernel matrices
            Ks = []
            for kernel in self.kernels:
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        K[i, j] = kernel(X[i], X[j])
                Ks.append(K)
            self.Ks = Ks
            
            # Initialize weights uniformly
            self.mu = np.ones(self.n_kernels) / self.n_kernels
            
            for iteration in range(max_iter):
                # Combined kernel
                K_combined = sum(mu * K for mu, K in zip(self.mu, Ks))
                
                # Solve kernel ridge regression
                alpha = np.linalg.solve(K_combined + self.lambda_ * np.eye(n), y)
                
                # Gradient for each weight
                gradients = np.zeros(self.n_kernels)
                for m in range(self.n_kernels):
                    # ∂L/∂μ_m ≈ -α^T K_m α (simplified)
                    gradients[m] = -alpha @ Ks[m] @ alpha
                
                # Gradient step
                self.mu -= lr * gradients
                
                # Project onto simplex
                self.mu = np.maximum(self.mu, 0)
                self.mu /= self.mu.sum()
            
            # Final solution
            K_combined = sum(mu * K for mu, K in zip(self.mu, Ks))
            self.alpha = np.linalg.solve(K_combined + self.lambda_ * np.eye(n), y)
            self.X_train = X
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict using learned kernel combination."""
            predictions = []
            
            for x in X:
                pred = 0
                for m, kernel in enumerate(self.kernels):
                    k = np.array([kernel(x, xi) for xi in self.X_train])
                    pred += self.mu[m] * (k @ self.alpha)
                predictions.append(pred)
            
            return np.array(predictions)
    
    # Generate data
    np.random.seed(42)
    n = 50
    X = np.sort(np.random.uniform(-3, 3, n)).reshape(-1, 1)
    y = np.sin(X.flatten()) + 0.5 * X.flatten()**2 + 0.1 * np.random.randn(n)
    
    print(f"Data: sin(x) + 0.5x² with noise")
    
    # Define kernels
    kernels = [
        lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 0.5),  # RBF narrow
        lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 2),    # RBF wide
        lambda x, y: (np.dot(x, y) + 1)**2,                    # Polynomial
        lambda x, y: np.dot(x, y),                              # Linear
    ]
    
    mkl = SimpleMultipleKernelLearning(kernels, lambda_=0.1)
    mkl.fit(X, y)
    
    print(f"\nLearned kernel weights:")
    names = ['RBF narrow', 'RBF wide', 'Polynomial', 'Linear']
    for name, mu in zip(names, mkl.mu):
        print(f"  {name}: {mu:.4f}")
    
    # Compare with individual kernels
    print(f"\nIndividual kernel MSEs:")
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_true = np.sin(X_test.flatten()) + 0.5 * X_test.flatten()**2
    
    for name, kernel in zip(names, kernels):
        # Kernel ridge with single kernel
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j])
        
        alpha = np.linalg.solve(K + 0.1 * np.eye(n), y)
        
        y_pred = []
        for x in X_test:
            k = np.array([kernel(x, xi) for xi in X])
            y_pred.append(k @ alpha)
        
        mse = np.mean((np.array(y_pred) - y_true)**2)
        print(f"  {name}: {mse:.6f}")
    
    # MKL result
    y_pred_mkl = mkl.predict(X_test)
    mse_mkl = np.mean((y_pred_mkl - y_true)**2)
    print(f"\n  MKL combined: {mse_mkl:.6f}")
    
    return SimpleMultipleKernelLearning


# =============================================================================
# Example 10: Neural Tangent Kernel
# =============================================================================

def example_10_ntk():
    """
    Demonstrate Neural Tangent Kernel connection.
    """
    print("\nExample 10: Neural Tangent Kernel")
    print("=" * 60)
    
    def ntk_kernel_1layer(x: np.ndarray, y: np.ndarray, 
                         sigma_w: float = 1.0) -> float:
        """
        Analytical NTK for single hidden layer ReLU network.
        """
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        
        if norm_x < 1e-10 or norm_y < 1e-10:
            return 0
        
        cos_theta = np.clip(np.dot(x, y) / (norm_x * norm_y), -1, 1)
        theta = np.arccos(cos_theta)
        
        # NTK formula for ReLU
        K_ntk = (sigma_w**2 / (2 * np.pi)) * norm_x * norm_y * (
            np.sin(theta) + (np.pi - theta) * cos_theta + (np.pi - theta)
        )
        
        return K_ntk
    
    def empirical_ntk(model_fn: Callable, X: np.ndarray, 
                     theta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute empirical NTK: K_ij = ∇_θ f(x_i)^T ∇_θ f(x_j)
        """
        n = len(X)
        p = len(theta)
        
        # Compute gradients via finite differences
        grads = np.zeros((n, p))
        
        for i in range(n):
            grad_i = np.zeros(p)
            f0 = model_fn(X[i], theta)
            
            for j in range(p):
                theta_plus = theta.copy()
                theta_plus[j] += epsilon
                f_plus = model_fn(X[i], theta_plus)
                grad_i[j] = (f_plus - f0) / epsilon
            
            grads[i] = grad_i
        
        return grads @ grads.T
    
    def simple_network(x: np.ndarray, theta: np.ndarray) -> float:
        """Single hidden layer network for NTK demo."""
        d = len(x)
        h = 50  # hidden units
        
        # Parse parameters
        W1 = theta[:h*d].reshape(h, d)
        b1 = theta[h*d:h*d+h]
        W2 = theta[h*d+h:h*d+2*h]
        b2 = theta[-1]
        
        # Forward
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)  # ReLU
        return W2 @ a1 + b2
    
    # Test
    np.random.seed(42)
    d = 3
    h = 50
    n_params = h * d + h + h + 1
    
    # Initialize parameters
    theta = np.random.randn(n_params) / np.sqrt(d)
    
    # Test points
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize
    
    print(f"Network: 1 hidden layer, {h} units, ReLU")
    print(f"Parameters: {n_params}")
    
    # Empirical NTK
    K_empirical = empirical_ntk(simple_network, X, theta)
    
    print(f"\nEmpirical NTK (via gradients):")
    print(K_empirical.round(4))
    
    # Analytical NTK
    K_analytical = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            K_analytical[i, j] = ntk_kernel_1layer(X[i], X[j])
    
    print(f"\nAnalytical NTK (infinite width):")
    print(K_analytical.round(4))
    
    # Check PSD
    print(f"\nEmpirical NTK PSD: {np.all(np.linalg.eigvalsh(K_empirical) >= -1e-10)}")
    print(f"Analytical NTK PSD: {np.all(np.linalg.eigvalsh(K_analytical) >= -1e-10)}")
    
    return ntk_kernel_1layer


def run_all_examples():
    """Run all kernel method examples."""
    print("=" * 70)
    print("KERNEL METHODS - EXAMPLES")
    print("=" * 70)
    
    example_1_kernels()
    example_2_kernel_trick()
    example_3_svm()
    example_4_kernel_ridge()
    example_5_kernel_pca()
    example_6_gaussian_process()
    example_7_random_fourier()
    example_8_nystrom()
    example_9_mkl()
    example_10_ntk()
    
    print("\n" + "=" * 70)
    print("All kernel method examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
