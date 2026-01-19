"""
Hilbert Spaces - Exercises
==========================

Hands-on exercises for Hilbert space concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# Exercise 1: Inner Product Verification
# =============================================================================

class Exercise1:
    """
    Verify and work with different inner products.
    """
    
    @staticmethod
    def problem():
        print("Exercise 1: Inner Product Verification")
        print("-" * 50)
        print("""
        Tasks:
        1. Implement a weighted inner product class
        2. Verify all axioms (symmetry, linearity, positive definiteness)
        3. Check Cauchy-Schwarz inequality
        4. Verify parallelogram law
        5. Show that l^p norms with p ≠ 2 don't satisfy parallelogram law
        """)
    
    @staticmethod
    def solution():
        class InnerProductSpace:
            """Generic inner product space."""
            
            def __init__(self, W: np.ndarray = None):
                """Initialize with optional weight matrix."""
                self.W = W  # For weighted inner product
            
            def inner(self, x: np.ndarray, y: np.ndarray) -> float:
                """Compute inner product."""
                if self.W is None:
                    return np.dot(x, y)
                return x @ self.W @ y
            
            def norm(self, x: np.ndarray) -> float:
                """Induced norm."""
                return np.sqrt(self.inner(x, x))
            
            def verify_axioms(self, x: np.ndarray, y: np.ndarray,
                            z: np.ndarray, a: float, b: float) -> Dict:
                """Verify all inner product axioms."""
                results = {}
                
                # A1: Symmetry
                results['Symmetry'] = np.isclose(
                    self.inner(x, y), self.inner(y, x)
                )
                
                # A2: Linearity
                lhs = self.inner(a*x + b*y, z)
                rhs = a * self.inner(x, z) + b * self.inner(y, z)
                results['Linearity'] = np.isclose(lhs, rhs)
                
                # A3: Positive definiteness
                results['Positive'] = self.inner(x, x) >= 0
                results['Zero iff zero'] = np.isclose(
                    self.inner(np.zeros_like(x), np.zeros_like(x)), 0
                )
                
                return results
            
            def cauchy_schwarz(self, x: np.ndarray, y: np.ndarray) -> Tuple[bool, float, float]:
                """Check |<x,y>| <= ||x|| ||y||."""
                lhs = abs(self.inner(x, y))
                rhs = self.norm(x) * self.norm(y)
                return lhs <= rhs + 1e-10, lhs, rhs
            
            def parallelogram_law(self, x: np.ndarray, 
                                 y: np.ndarray) -> Tuple[bool, float, float]:
                """Check ||x+y||² + ||x-y||² = 2(||x||² + ||y||²)."""
                lhs = self.norm(x + y)**2 + self.norm(x - y)**2
                rhs = 2 * (self.norm(x)**2 + self.norm(y)**2)
                return np.isclose(lhs, rhs), lhs, rhs
        
        def check_parallelogram_for_p_norm(x: np.ndarray, y: np.ndarray, 
                                          p: float) -> Tuple[bool, float, float]:
            """Check if p-norm satisfies parallelogram law."""
            norm_p = lambda v: np.linalg.norm(v, p)
            lhs = norm_p(x + y)**2 + norm_p(x - y)**2
            rhs = 2 * (norm_p(x)**2 + norm_p(y)**2)
            return np.isclose(lhs, rhs), lhs, rhs
        
        print("\nSolution:")
        
        # Test vectors
        x = np.array([1.0, 2.0, -1.0])
        y = np.array([3.0, -1.0, 2.0])
        z = np.array([0.5, 1.0, 1.0])
        a, b = 2.0, -0.5
        
        # Standard inner product
        print("Standard inner product:")
        ip = InnerProductSpace()
        
        print(f"  Axiom verification:")
        for axiom, satisfied in ip.verify_axioms(x, y, z, a, b).items():
            print(f"    {axiom}: {'✓' if satisfied else '✗'}")
        
        cs_ok, cs_lhs, cs_rhs = ip.cauchy_schwarz(x, y)
        print(f"\n  Cauchy-Schwarz: |<x,y>| = {cs_lhs:.4f} <= {cs_rhs:.4f}: {'✓' if cs_ok else '✗'}")
        
        pl_ok, pl_lhs, pl_rhs = ip.parallelogram_law(x, y)
        print(f"  Parallelogram: {pl_lhs:.4f} = {pl_rhs:.4f}: {'✓' if pl_ok else '✗'}")
        
        # Weighted inner product
        print("\nWeighted inner product (W = diag(1, 2, 3)):")
        W = np.diag([1, 2, 3])
        ip_w = InnerProductSpace(W)
        
        for axiom, satisfied in ip_w.verify_axioms(x, y, z, a, b).items():
            print(f"  {axiom}: {'✓' if satisfied else '✗'}")
        
        # p-norms and parallelogram law
        print("\nParallelogram law for p-norms:")
        for p in [1, 2, 3, np.inf]:
            ok, lhs, rhs = check_parallelogram_for_p_norm(x, y, p)
            print(f"  p = {p}: {lhs:.4f} vs {rhs:.4f} -> {'✓' if ok else '✗ (not from inner product)'}")
        
        return InnerProductSpace


# =============================================================================
# Exercise 2: Orthogonal Projection Implementation
# =============================================================================

class Exercise2:
    """
    Implement orthogonal projection onto various subspaces.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 2: Orthogonal Projection Implementation")
        print("-" * 50)
        print("""
        Tasks:
        1. Project onto a line through origin
        2. Project onto a plane through origin
        3. Project onto column space of a matrix
        4. Verify P² = P and P^T = P
        5. Show x - Px ⊥ range(P)
        """)
    
    @staticmethod
    def solution():
        def project_onto_line(x: np.ndarray, u: np.ndarray) -> np.ndarray:
            """Project x onto line spanned by u."""
            u = u / np.linalg.norm(u)  # Normalize
            return np.dot(x, u) * u
        
        def project_onto_plane(x: np.ndarray, n: np.ndarray) -> np.ndarray:
            """Project x onto plane with normal n."""
            n = n / np.linalg.norm(n)
            return x - np.dot(x, n) * n
        
        def project_onto_column_space(x: np.ndarray, A: np.ndarray) -> np.ndarray:
            """Project x onto col(A) using QR."""
            Q, R = np.linalg.qr(A)
            rank = np.sum(np.abs(np.diag(R)) > 1e-10)
            Q = Q[:, :rank]
            return Q @ (Q.T @ x)
        
        def projection_matrix_qr(A: np.ndarray) -> np.ndarray:
            """Compute P = Q Q^T for col(A)."""
            Q, R = np.linalg.qr(A)
            rank = np.sum(np.abs(np.diag(R)) > 1e-10)
            Q = Q[:, :rank]
            return Q @ Q.T
        
        def verify_projection_properties(P: np.ndarray, x: np.ndarray,
                                        A: np.ndarray) -> Dict:
            """Verify all projection properties."""
            results = {}
            
            # P² = P
            results['Idempotent (P²=P)'] = np.allclose(P @ P, P)
            
            # P^T = P
            results['Self-adjoint (P^T=P)'] = np.allclose(P, P.T)
            
            # x - Px ⊥ col(A)
            residual = x - P @ x
            results['Residual ⊥ col(A)'] = np.allclose(A.T @ residual, 0)
            
            # ||P|| <= 1
            results['||P|| <= 1'] = np.linalg.norm(P, 2) <= 1 + 1e-10
            
            return results
        
        print("\nSolution:")
        
        # Test: project onto line
        x = np.array([3.0, 4.0, 5.0])
        u = np.array([1.0, 0.0, 0.0])
        
        proj_line = project_onto_line(x, u)
        print(f"Project x = {x} onto x-axis:")
        print(f"  Projection: {proj_line}")
        
        # Test: project onto plane
        n = np.array([0.0, 0.0, 1.0])  # xy-plane
        proj_plane = project_onto_plane(x, n)
        print(f"\nProject x onto xy-plane:")
        print(f"  Projection: {proj_plane}")
        
        # Test: project onto column space
        A = np.array([[1, 0],
                      [1, 1],
                      [0, 1]], dtype=float)
        
        proj_col = project_onto_column_space(x, A)
        P = projection_matrix_qr(A)
        
        print(f"\nProject x onto col(A):")
        print(f"  Projection: {proj_col}")
        print(f"  Residual: {x - proj_col}")
        
        print(f"\n  Properties:")
        for prop, satisfied in verify_projection_properties(P, x, A).items():
            print(f"    {prop}: {'✓' if satisfied else '✗'}")
        
        return project_onto_column_space


# =============================================================================
# Exercise 3: Gram-Schmidt with Numerical Stability
# =============================================================================

class Exercise3:
    """
    Implement robust Gram-Schmidt orthonormalization.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 3: Gram-Schmidt with Numerical Stability")
        print("-" * 50)
        print("""
        Tasks:
        1. Implement classical Gram-Schmidt
        2. Implement modified Gram-Schmidt
        3. Implement twice-iterated Gram-Schmidt
        4. Compare numerical stability on ill-conditioned inputs
        5. Handle rank-deficient cases
        """)
    
    @staticmethod
    def solution():
        def classical_gram_schmidt(V: np.ndarray, tol: float = 1e-10) -> np.ndarray:
            """Classical Gram-Schmidt (less stable)."""
            n, m = V.shape
            Q = np.zeros((n, m))
            
            for k in range(m):
                u = V[:, k].copy()
                
                for j in range(k):
                    u -= np.dot(V[:, k], Q[:, j]) * Q[:, j]
                
                norm = np.linalg.norm(u)
                if norm > tol:
                    Q[:, k] = u / norm
            
            return Q
        
        def modified_gram_schmidt(V: np.ndarray, tol: float = 1e-10) -> np.ndarray:
            """Modified Gram-Schmidt (more stable)."""
            n, m = V.shape
            Q = V.astype(float).copy()
            
            for k in range(m):
                norm = np.linalg.norm(Q[:, k])
                if norm < tol:
                    Q[:, k] = 0
                    continue
                    
                Q[:, k] /= norm
                
                for j in range(k + 1, m):
                    Q[:, j] -= np.dot(Q[:, j], Q[:, k]) * Q[:, k]
            
            return Q
        
        def twice_iterated_gs(V: np.ndarray, tol: float = 1e-10) -> np.ndarray:
            """Twice-iterated Gram-Schmidt for maximum stability."""
            Q = modified_gram_schmidt(V, tol)
            Q = modified_gram_schmidt(Q, tol)
            return Q
        
        def orthonormality_error(Q: np.ndarray) -> float:
            """Measure deviation from orthonormality."""
            m = Q.shape[1]
            I = np.eye(m)
            return np.linalg.norm(Q.T @ Q - I)
        
        print("\nSolution:")
        
        # Well-conditioned case
        V_good = np.array([[1, 1, 0],
                          [1, 0, 1],
                          [0, 1, 1],
                          [1, 1, 1]], dtype=float)
        
        print("Well-conditioned input:")
        for name, gs in [('Classical', classical_gram_schmidt),
                        ('Modified', modified_gram_schmidt),
                        ('Twice-iterated', twice_iterated_gs)]:
            Q = gs(V_good)
            error = orthonormality_error(Q)
            print(f"  {name}: error = {error:.2e}")
        
        # Ill-conditioned case
        epsilon = 1e-8
        V_ill = np.array([[1, 1, 1 + epsilon],
                         [1, 1 + epsilon, 1],
                         [1 + epsilon, 1, 1]], dtype=float)
        
        print(f"\nIll-conditioned input (ε = {epsilon}):")
        for name, gs in [('Classical', classical_gram_schmidt),
                        ('Modified', modified_gram_schmidt),
                        ('Twice-iterated', twice_iterated_gs)]:
            Q = gs(V_ill)
            error = orthonormality_error(Q)
            print(f"  {name}: error = {error:.2e}")
        
        # Rank-deficient case
        V_rank = np.array([[1, 2, 3],
                          [2, 4, 6],
                          [1, 0, 1]], dtype=float)  # col 2 = 2*col 1
        
        print(f"\nRank-deficient input:")
        Q = modified_gram_schmidt(V_rank)
        rank = np.sum(np.linalg.norm(Q, axis=0) > 1e-10)
        print(f"  Detected rank: {rank}")
        
        return modified_gram_schmidt


# =============================================================================
# Exercise 4: Fourier Series Approximation
# =============================================================================

class Exercise4:
    """
    Implement Fourier series approximation.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 4: Fourier Series Approximation")
        print("-" * 50)
        print("""
        Tasks:
        1. Compute Fourier coefficients for a given function
        2. Reconstruct function from coefficients
        3. Verify Parseval's identity: ||f||² = Σ|c_n|²
        4. Demonstrate Bessel's inequality
        5. Analyze convergence rate for different functions
        """)
    
    @staticmethod
    def solution():
        def fourier_coefficients(f: Callable, n_terms: int,
                                n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
            """
            Compute real Fourier coefficients.
            
            Returns (a, b) where f ≈ a_0/2 + Σ(a_n cos(nx) + b_n sin(nx))
            """
            x = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            dx = 2*np.pi / n_points
            f_vals = f(x)
            
            a = np.zeros(n_terms + 1)
            b = np.zeros(n_terms + 1)
            
            # a_n = (1/π) ∫ f(x) cos(nx) dx
            # b_n = (1/π) ∫ f(x) sin(nx) dx
            
            for n in range(n_terms + 1):
                a[n] = np.sum(f_vals * np.cos(n * x)) * dx / np.pi
                b[n] = np.sum(f_vals * np.sin(n * x)) * dx / np.pi
            
            # a_0 needs factor of 1/2 in reconstruction
            return a, b
        
        def reconstruct(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
            """Reconstruct function from Fourier coefficients."""
            result = a[0] / 2
            
            for n in range(1, len(a)):
                result = result + a[n] * np.cos(n * x) + b[n] * np.sin(n * x)
            
            return result
        
        def parseval_identity(f: Callable, a: np.ndarray, b: np.ndarray,
                            n_points: int = 1000) -> Dict:
            """Verify Parseval's identity."""
            x = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            dx = 2*np.pi / n_points
            f_vals = f(x)
            
            # ||f||² = (1/π) ∫ |f(x)|² dx
            norm_sq_integral = np.sum(f_vals**2) * dx / np.pi
            
            # Σ|c_n|² = a_0²/2 + Σ(a_n² + b_n²)
            norm_sq_coeffs = a[0]**2 / 2 + np.sum(a[1:]**2 + b[1:]**2)
            
            return {
                '||f||² (integral)': norm_sq_integral,
                'Σ(a²+b²) (Parseval)': norm_sq_coeffs,
                'Relative error': abs(norm_sq_integral - norm_sq_coeffs) / (norm_sq_integral + 1e-10)
            }
        
        def bessel_inequality(f: Callable, n_terms: int) -> float:
            """Verify Bessel's inequality: Σ|c_n|² <= ||f||²."""
            a, b = fourier_coefficients(f, n_terms)
            
            x = np.linspace(0, 2*np.pi, 1000, endpoint=False)
            dx = 2*np.pi / 1000
            f_vals = f(x)
            
            norm_sq = np.sum(f_vals**2) * dx / np.pi
            coeff_sq = a[0]**2 / 2 + np.sum(a[1:]**2 + b[1:]**2)
            
            return coeff_sq <= norm_sq + 1e-10
        
        print("\nSolution:")
        
        # Test functions
        def sawtooth(x):
            return x - np.pi
        
        def square_wave(x):
            return np.sign(np.sin(x))
        
        def smooth_func(x):
            return np.sin(x) + 0.5 * np.cos(2*x)
        
        functions = [
            ('Sawtooth', sawtooth),
            ('Square wave', square_wave),
            ('Smooth', smooth_func)
        ]
        
        x_test = np.linspace(0, 2*np.pi, 200)
        
        for name, f in functions:
            print(f"\n{name} function:")
            
            for n_terms in [5, 20, 50]:
                a, b = fourier_coefficients(f, n_terms)
                approx = reconstruct(a, b, x_test)
                rmse = np.sqrt(np.mean((f(x_test) - approx)**2))
                
                print(f"  {n_terms} terms: RMSE = {rmse:.6f}")
            
            # Parseval
            a, b = fourier_coefficients(f, 50)
            parseval = parseval_identity(f, a, b)
            print(f"  Parseval relative error: {parseval['Relative error']:.2e}")
            
            # Bessel
            print(f"  Bessel inequality satisfied: {bessel_inequality(f, 50)}")
        
        return fourier_coefficients, reconstruct


# =============================================================================
# Exercise 5: RKHS and Kernel Methods
# =============================================================================

class Exercise5:
    """
    Work with Reproducing Kernel Hilbert Spaces.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 5: RKHS and Kernel Methods")
        print("-" * 50)
        print("""
        Tasks:
        1. Implement common kernels (RBF, polynomial, Laplacian)
        2. Verify positive semi-definiteness
        3. Demonstrate reproducing property
        4. Compute RKHS norm of functions
        5. Show kernel = inner product of feature maps
        """)
    
    @staticmethod
    def solution():
        class RKHS:
            """RKHS with various kernels."""
            
            @staticmethod
            def rbf_kernel(x, y, sigma=1.0):
                return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
            
            @staticmethod
            def polynomial_kernel(x, y, degree=2, c=1):
                return (np.dot(x, y) + c) ** degree
            
            @staticmethod
            def laplacian_kernel(x, y, sigma=1.0):
                return np.exp(-np.linalg.norm(x - y, 1) / sigma)
            
            @staticmethod
            def linear_kernel(x, y):
                return np.dot(x, y)
            
            @staticmethod
            def gram_matrix(X, kernel):
                """Compute Gram matrix K_ij = K(x_i, x_j)."""
                n = len(X)
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        K[i, j] = kernel(X[i], X[j])
                return K
            
            @staticmethod
            def is_psd(K):
                """Check positive semi-definiteness."""
                eigenvalues = np.linalg.eigvalsh(K)
                return np.all(eigenvalues >= -1e-10), eigenvalues
            
            @staticmethod
            def rkhs_function(alpha, X, kernel):
                """
                Return function f(x) = Σ α_i K(x_i, x).
                """
                def f(x):
                    return sum(a * kernel(xi, x) for a, xi in zip(alpha, X))
                return f
            
            @staticmethod
            def rkhs_norm_squared(alpha, X, kernel):
                """
                Compute ||f||_H² = α^T K α for f = Σ α_i K(·, x_i).
                """
                K = RKHS.gram_matrix(X, kernel)
                return alpha @ K @ alpha
            
            @staticmethod
            def reproducing_property(f, X, kernel, alpha):
                """
                Verify f(x_i) = <f, K(·, x_i)>_H = Σ_j α_j K(x_j, x_i).
                """
                K = RKHS.gram_matrix(X, kernel)
                f_values_direct = np.array([f(x) for x in X])
                f_values_kernel = K @ alpha
                return np.allclose(f_values_direct, f_values_kernel)
        
        print("\nSolution:")
        
        # Sample data
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=float)
        
        print("Data points:")
        print(X)
        
        # Test each kernel
        kernels = {
            'RBF (σ=1)': lambda x, y: RKHS.rbf_kernel(x, y, 1),
            'Polynomial (d=2)': lambda x, y: RKHS.polynomial_kernel(x, y, 2),
            'Laplacian (σ=1)': lambda x, y: RKHS.laplacian_kernel(x, y, 1),
            'Linear': RKHS.linear_kernel
        }
        
        for name, kernel in kernels.items():
            print(f"\n{name}:")
            
            K = RKHS.gram_matrix(X, kernel)
            is_psd, eigs = RKHS.is_psd(K)
            
            print(f"  PSD: {is_psd}")
            print(f"  Eigenvalues: {eigs}")
            
            # Test reproducing property
            alpha = np.array([1, -1, 0.5, -0.5, 0.2])
            f = RKHS.rkhs_function(alpha, X, kernel)
            repro = RKHS.reproducing_property(f, X, kernel, alpha)
            print(f"  Reproducing property: {repro}")
            
            # RKHS norm
            norm_sq = RKHS.rkhs_norm_squared(alpha, X, kernel)
            print(f"  ||f||_H² = {norm_sq:.4f}")
        
        return RKHS


# =============================================================================
# Exercise 6: Kernel Ridge Regression
# =============================================================================

class Exercise6:
    """
    Implement kernel ridge regression with cross-validation.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 6: Kernel Ridge Regression")
        print("-" * 50)
        print("""
        Tasks:
        1. Implement kernel ridge regression
        2. Add leave-one-out cross-validation
        3. Compare different kernels
        4. Analyze regularization effect on RKHS norm
        5. Implement dual representation
        """)
    
    @staticmethod
    def solution():
        class KernelRidgeCV:
            """Kernel ridge regression with CV."""
            
            def __init__(self, kernel, lambda_: float = 1.0):
                self.kernel = kernel
                self.lambda_ = lambda_
                self.X_train = None
                self.alpha = None
                self.K = None
            
            def fit(self, X, y):
                self.X_train = X
                n = len(X)
                
                self.K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        self.K[i, j] = self.kernel(X[i], X[j])
                
                self.alpha = np.linalg.solve(
                    self.K + self.lambda_ * np.eye(n), y
                )
                return self
            
            def predict(self, X):
                predictions = []
                for x in X:
                    k = np.array([self.kernel(x, xi) for xi in self.X_train])
                    predictions.append(k @ self.alpha)
                return np.array(predictions)
            
            def loo_cv_score(self, X, y):
                """
                Leave-one-out CV using closed form:
                LOO error = (y_i - f_{-i}(x_i)) = (y_i - f(x_i)) / (1 - H_ii)
                """
                self.fit(X, y)
                n = len(y)
                
                H = self.K @ np.linalg.inv(self.K + self.lambda_ * np.eye(n))
                y_pred = self.K @ self.alpha
                
                loo_errors = (y - y_pred) / (1 - np.diag(H) + 1e-10)
                return np.mean(loo_errors**2)
            
            def rkhs_norm_squared(self):
                return self.alpha @ self.K @ self.alpha
            
            @staticmethod
            def select_lambda(X, y, kernel, lambdas):
                """Select best lambda via LOO-CV."""
                scores = []
                for lam in lambdas:
                    model = KernelRidgeCV(kernel, lam)
                    score = model.loo_cv_score(X, y)
                    scores.append(score)
                
                best_idx = np.argmin(scores)
                return lambdas[best_idx], scores
        
        print("\nSolution:")
        
        # Generate nonlinear data
        np.random.seed(42)
        n = 30
        X_train = np.sort(np.random.uniform(-2, 2, n))
        y_train = np.sin(2 * X_train) + 0.2 * np.random.randn(n)
        
        X_test = np.linspace(-2, 2, 100)
        y_true = np.sin(2 * X_test)
        
        # Define kernels
        rbf = lambda x, y: np.exp(-(x - y)**2 / 0.5)
        poly = lambda x, y: (x * y + 1)**3
        
        print("Comparing kernels:")
        
        for name, kernel in [('RBF', rbf), ('Polynomial', poly)]:
            # Find best lambda
            lambdas = np.logspace(-4, 2, 20)
            best_lam, scores = KernelRidgeCV.select_lambda(
                X_train.reshape(-1, 1), y_train, kernel, lambdas
            )
            
            # Fit with best lambda
            model = KernelRidgeCV(kernel, best_lam)
            model.fit(X_train.reshape(-1, 1), y_train)
            
            y_pred = model.predict(X_test.reshape(-1, 1))
            mse = np.mean((y_pred - y_true)**2)
            
            print(f"\n  {name}:")
            print(f"    Best λ: {best_lam:.4f}")
            print(f"    Test MSE: {mse:.6f}")
            print(f"    ||f||_H²: {model.rkhs_norm_squared():.4f}")
        
        return KernelRidgeCV


# =============================================================================
# Exercise 7: Kernel PCA
# =============================================================================

class Exercise7:
    """
    Implement kernel PCA.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 7: Kernel PCA")
        print("-" * 50)
        print("""
        Tasks:
        1. Center kernel matrix
        2. Perform eigendecomposition
        3. Project data to principal components
        4. Project new points
        5. Compare with linear PCA
        """)
    
    @staticmethod
    def solution():
        class KernelPCA:
            """Kernel PCA implementation."""
            
            def __init__(self, kernel, n_components=2):
                self.kernel = kernel
                self.n_components = n_components
                self.X_train = None
                self.alpha = None
                self.lambdas = None
                self.K_train = None
                self.one_n = None
            
            def fit(self, X):
                self.X_train = X
                n = len(X)
                
                # Compute kernel matrix
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        K[i, j] = self.kernel(X[i], X[j])
                self.K_train = K
                
                # Center kernel: K_c = K - 1_n K - K 1_n + 1_n K 1_n
                self.one_n = np.ones((n, n)) / n
                K_centered = K - self.one_n @ K - K @ self.one_n + self.one_n @ K @ self.one_n
                
                # Eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
                
                # Sort descending
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Select top components
                self.lambdas = eigenvalues[:self.n_components]
                self.alpha = eigenvectors[:, :self.n_components]
                
                # Normalize
                for k in range(self.n_components):
                    if self.lambdas[k] > 1e-10:
                        self.alpha[:, k] /= np.sqrt(self.lambdas[k])
                
                return self
            
            def transform(self, X):
                """Project to principal components."""
                n_train = len(self.X_train)
                n_test = len(X)
                
                # Compute kernel with training data
                K = np.zeros((n_test, n_train))
                for i in range(n_test):
                    for j in range(n_train):
                        K[i, j] = self.kernel(X[i], self.X_train[j])
                
                # Center kernel
                one_n = np.ones((n_test, n_train)) / n_train
                K_centered = K - one_n @ self.K_train - K @ self.one_n + one_n @ self.K_train @ self.one_n
                
                return K_centered @ self.alpha
            
            def fit_transform(self, X):
                self.fit(X)
                K_centered = self.K_train - self.one_n @ self.K_train - self.K_train @ self.one_n + self.one_n @ self.K_train @ self.one_n
                return K_centered @ self.alpha
            
            def explained_variance_ratio(self):
                """Variance explained by each component."""
                total = np.sum(self.lambdas[self.lambdas > 0])
                return self.lambdas / total if total > 0 else self.lambdas
        
        print("\nSolution:")
        
        # Generate circular data
        np.random.seed(42)
        n = 100
        
        theta = np.random.uniform(0, 2*np.pi, n)
        r = 1 + 0.2 * np.random.randn(n)
        X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        
        print(f"Data: {n} points on noisy circle")
        
        # Linear PCA
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_linear_pca = X_centered @ Vt.T
        
        # Kernel PCA with RBF
        rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / 0.5)
        kpca = KernelPCA(rbf, n_components=2)
        X_kernel_pca = kpca.fit_transform(X)
        
        print(f"\nLinear PCA:")
        print(f"  Variance ratio: {np.var(X_linear_pca, axis=0) / np.var(X_linear_pca).sum()}")
        
        print(f"\nKernel PCA (RBF):")
        print(f"  Eigenvalues: {kpca.lambdas}")
        print(f"  Explained variance ratio: {kpca.explained_variance_ratio()}")
        
        return KernelPCA


# =============================================================================
# Exercise 8: Spectral Theorem Application
# =============================================================================

class Exercise8:
    """
    Apply spectral theorem to compact operators.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 8: Spectral Theorem Application")
        print("-" * 50)
        print("""
        Tasks:
        1. Approximate kernel operator eigendecomposition
        2. Verify Mercer decomposition: K(x,y) = Σ λ_n φ_n(x) φ_n(y)
        3. Implement Nyström approximation
        4. Analyze eigenvalue decay rates
        5. Apply to kernel approximation
        """)
    
    @staticmethod
    def solution():
        def kernel_eigendecomposition(X, kernel, n_components=None):
            """Compute eigendecomposition of kernel matrix."""
            n = len(X)
            if n_components is None:
                n_components = n
            
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = kernel(X[i], X[j])
            
            eigenvalues, eigenvectors = np.linalg.eigh(K)
            
            # Sort descending
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx][:n_components]
            eigenvectors = eigenvectors[:, idx][:, :n_components]
            
            return eigenvalues / n, eigenvectors
        
        def mercer_reconstruction(X, eigenvalues, eigenvectors, x, y, kernel):
            """
            Verify K(x, y) ≈ Σ λ_k φ_k(x) φ_k(y)
            """
            n = len(X)
            
            # Compute φ_k(x) = (1/√λ_k) Σ_i α_{ki} K(x_i, x)
            def phi_k(z, k):
                k_z = np.array([kernel(xi, z) for xi in X])
                return k_z @ eigenvectors[:, k] / np.sqrt(n * eigenvalues[k] + 1e-10)
            
            # Reconstruct
            reconstruction = 0
            for k in range(len(eigenvalues)):
                if eigenvalues[k] > 1e-10:
                    reconstruction += eigenvalues[k] * phi_k(x, k) * phi_k(y, k)
            
            actual = kernel(x, y)
            return reconstruction, actual
        
        def nystrom_approximation(X, kernel, n_landmarks):
            """
            Nyström approximation for kernel matrix.
            """
            n = len(X)
            np.random.seed(42)
            
            # Select landmarks
            landmark_idx = np.random.choice(n, n_landmarks, replace=False)
            landmarks = X[landmark_idx]
            
            # K_mm (landmark-landmark)
            K_mm = np.zeros((n_landmarks, n_landmarks))
            for i in range(n_landmarks):
                for j in range(n_landmarks):
                    K_mm[i, j] = kernel(landmarks[i], landmarks[j])
            
            # K_nm (data-landmark)
            K_nm = np.zeros((n, n_landmarks))
            for i in range(n):
                for j in range(n_landmarks):
                    K_nm[i, j] = kernel(X[i], landmarks[j])
            
            # Approximate: K ≈ K_nm K_mm^{-1} K_nm^T
            K_mm_inv = np.linalg.pinv(K_mm)
            K_approx = K_nm @ K_mm_inv @ K_nm.T
            
            return K_approx
        
        print("\nSolution:")
        
        # Generate data
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        
        # RBF kernel
        sigma = 1.0
        rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
        
        # True kernel matrix
        K_true = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K_true[i, j] = rbf(X[i], X[j])
        
        # Eigendecomposition
        eigenvalues, eigenvectors = kernel_eigendecomposition(X, rbf)
        
        print(f"Eigenvalue decay (top 10): {eigenvalues[:10]}")
        print(f"Sum of eigenvalues: {np.sum(eigenvalues):.4f}")
        print(f"Trace(K)/n: {np.trace(K_true)/n:.4f}")
        
        # Mercer verification
        x, y = X[0], X[5]
        recon, actual = mercer_reconstruction(X, eigenvalues, eigenvectors, x, y, rbf)
        print(f"\nMercer reconstruction:")
        print(f"  K(x, y) actual: {actual:.6f}")
        print(f"  K(x, y) reconstructed: {recon:.6f}")
        
        # Nyström approximation
        print(f"\nNyström approximation errors:")
        for m in [5, 10, 20]:
            K_approx = nystrom_approximation(X, rbf, m)
            rel_error = np.linalg.norm(K_true - K_approx) / np.linalg.norm(K_true)
            print(f"  {m} landmarks: relative error = {rel_error:.4f}")
        
        return kernel_eigendecomposition, nystrom_approximation


# =============================================================================
# Exercise 9: Gradient in Hilbert Space
# =============================================================================

class Exercise9:
    """
    Compute gradients in Hilbert space.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 9: Gradient in Hilbert Space")
        print("-" * 50)
        print("""
        Tasks:
        1. Define functional on RKHS
        2. Compute gradient in Hilbert space
        3. Implement gradient descent in function space
        4. Apply to kernel regression
        5. Analyze convergence
        """)
    
    @staticmethod
    def solution():
        class FunctionalGradientDescent:
            """Gradient descent in RKHS."""
            
            def __init__(self, kernel, X, y, lambda_=0.01):
                self.kernel = kernel
                self.X = X  # Training points
                self.y = y  # Labels
                self.lambda_ = lambda_
                self.n = len(X)
                
                # Kernel matrix
                self.K = np.zeros((self.n, self.n))
                for i in range(self.n):
                    for j in range(self.n):
                        self.K[i, j] = kernel(X[i], X[j])
            
            def loss(self, alpha):
                """
                L(f) = (1/2n) Σ(f(x_i) - y_i)² + (λ/2)||f||_H²
                
                For f = Σ α_j K(·, x_j):
                L(α) = (1/2n) ||Kα - y||² + (λ/2) α^T K α
                """
                pred = self.K @ alpha
                data_loss = np.mean((pred - self.y)**2) / 2
                reg_loss = self.lambda_ * alpha @ self.K @ alpha / 2
                return data_loss + reg_loss
            
            def gradient(self, alpha):
                """
                ∇L(f) represented as:
                ∇_α L = (1/n) K(Kα - y) + λKα
                """
                pred = self.K @ alpha
                residual = pred - self.y
                return self.K @ residual / self.n + self.lambda_ * self.K @ alpha
            
            def gradient_descent(self, lr=0.1, max_iter=1000, tol=1e-6):
                """Run gradient descent."""
                alpha = np.zeros(self.n)
                history = {'loss': [], 'alpha': []}
                
                for i in range(max_iter):
                    grad = self.gradient(alpha)
                    alpha_new = alpha - lr * grad
                    
                    loss = self.loss(alpha)
                    history['loss'].append(loss)
                    history['alpha'].append(alpha.copy())
                    
                    if np.linalg.norm(alpha_new - alpha) < tol:
                        break
                    
                    alpha = alpha_new
                
                return alpha, history
            
            def predict(self, X_test, alpha):
                """Make predictions."""
                predictions = []
                for x in X_test:
                    k = np.array([self.kernel(x, xi) for xi in self.X])
                    predictions.append(k @ alpha)
                return np.array(predictions)
        
        print("\nSolution:")
        
        # Generate data
        np.random.seed(42)
        n = 30
        X_train = np.sort(np.random.uniform(-2, 2, n))
        y_train = np.sin(X_train) + 0.1 * np.random.randn(n)
        
        # Kernel
        rbf = lambda x, y: np.exp(-(x - y)**2 / 0.5)
        
        # Gradient descent
        fgd = FunctionalGradientDescent(rbf, X_train.reshape(-1, 1), y_train, lambda_=0.01)
        
        alpha_gd, history = fgd.gradient_descent(lr=0.1, max_iter=500)
        
        print(f"Gradient descent in RKHS:")
        print(f"  Initial loss: {history['loss'][0]:.6f}")
        print(f"  Final loss: {history['loss'][-1]:.6f}")
        print(f"  Iterations: {len(history['loss'])}")
        
        # Compare with closed form
        K = fgd.K
        alpha_closed = np.linalg.solve(K + fgd.lambda_ * n * np.eye(n), y_train)
        loss_closed = fgd.loss(alpha_closed)
        
        print(f"\n  Closed form loss: {loss_closed:.6f}")
        print(f"  α difference: {np.linalg.norm(alpha_gd - alpha_closed):.6f}")
        
        # Convergence analysis
        print(f"\n  Loss history (selected):")
        for i in [0, 10, 50, 100, -1]:
            if i < len(history['loss']):
                print(f"    Iter {i}: {history['loss'][i]:.6f}")
        
        return FunctionalGradientDescent


# =============================================================================
# Exercise 10: Complete Kernel Method Pipeline
# =============================================================================

class Exercise10:
    """
    Build complete kernel method pipeline.
    """
    
    @staticmethod
    def problem():
        print("\nExercise 10: Complete Kernel Method Pipeline")
        print("-" * 50)
        print("""
        Tasks:
        1. Implement multiple kernel options
        2. Add kernel hyperparameter tuning
        3. Use cross-validation for model selection
        4. Compare kernel and linear methods
        5. Analyze computational complexity
        """)
    
    @staticmethod
    def solution():
        class KernelMethodsPipeline:
            """Complete kernel methods pipeline."""
            
            def __init__(self):
                self.kernels = {}
                self.best_kernel = None
                self.best_params = None
                self.model = None
            
            def add_kernel(self, name, kernel_func):
                """Add kernel to library."""
                self.kernels[name] = kernel_func
            
            def cross_validate(self, X, y, kernel_name, params, n_folds=5):
                """K-fold cross-validation."""
                n = len(y)
                fold_size = n // n_folds
                
                # Create folds
                indices = np.random.permutation(n)
                scores = []
                
                for fold in range(n_folds):
                    val_idx = indices[fold*fold_size:(fold+1)*fold_size]
                    train_idx = np.concatenate([
                        indices[:fold*fold_size],
                        indices[(fold+1)*fold_size:]
                    ])
                    
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Build kernel
                    kernel = self.kernels[kernel_name](**params)
                    
                    # Train
                    K = np.zeros((len(X_train), len(X_train)))
                    for i in range(len(X_train)):
                        for j in range(len(X_train)):
                            K[i, j] = kernel(X_train[i], X_train[j])
                    
                    lambda_ = params.get('lambda_', 0.01)
                    alpha = np.linalg.solve(K + lambda_ * np.eye(len(X_train)), y_train)
                    
                    # Predict
                    y_pred = []
                    for x in X_val:
                        k = np.array([kernel(x, xi) for xi in X_train])
                        y_pred.append(k @ alpha)
                    
                    mse = np.mean((np.array(y_pred) - y_val)**2)
                    scores.append(mse)
                
                return np.mean(scores), np.std(scores)
            
            def grid_search(self, X, y, kernel_name, param_grid, n_folds=5):
                """Grid search over hyperparameters."""
                from itertools import product
                
                keys = list(param_grid.keys())
                values = list(param_grid.values())
                
                best_score = np.inf
                best_params = None
                
                for combo in product(*values):
                    params = dict(zip(keys, combo))
                    score, _ = self.cross_validate(X, y, kernel_name, params, n_folds)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                
                return best_params, best_score
            
            def fit(self, X, y, kernel_name, params):
                """Fit final model."""
                kernel = self.kernels[kernel_name](**params)
                
                n = len(X)
                K = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        K[i, j] = kernel(X[i], X[j])
                
                lambda_ = params.get('lambda_', 0.01)
                alpha = np.linalg.solve(K + lambda_ * np.eye(n), y)
                
                self.model = {
                    'kernel': kernel,
                    'X_train': X,
                    'alpha': alpha
                }
                return self
            
            def predict(self, X):
                """Make predictions."""
                kernel = self.model['kernel']
                X_train = self.model['X_train']
                alpha = self.model['alpha']
                
                predictions = []
                for x in X:
                    k = np.array([kernel(x, xi) for xi in X_train])
                    predictions.append(k @ alpha)
                
                return np.array(predictions)
        
        print("\nSolution:")
        
        # Create pipeline
        pipeline = KernelMethodsPipeline()
        
        # Add kernels
        pipeline.add_kernel('rbf', 
            lambda sigma=1.0, lambda_=0.01: lambda x, y: np.exp(-np.linalg.norm(x-y)**2/(2*sigma**2))
        )
        pipeline.add_kernel('polynomial',
            lambda degree=2, c=1.0, lambda_=0.01: lambda x, y: (np.dot(x, y) + c)**degree
        )
        
        # Generate data
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(n)
        
        print(f"Data: {n} samples, 2D input")
        
        # Grid search for RBF
        param_grid = {
            'sigma': [0.5, 1.0, 2.0],
            'lambda_': [0.001, 0.01, 0.1]
        }
        
        best_params, best_score = pipeline.grid_search(X, y, 'rbf', param_grid)
        
        print(f"\nRBF kernel:")
        print(f"  Best params: {best_params}")
        print(f"  CV score (MSE): {best_score:.6f}")
        
        # Grid search for polynomial
        param_grid_poly = {
            'degree': [2, 3, 4],
            'c': [0.5, 1.0],
            'lambda_': [0.01, 0.1]
        }
        
        best_params_poly, best_score_poly = pipeline.grid_search(X, y, 'polynomial', param_grid_poly)
        
        print(f"\nPolynomial kernel:")
        print(f"  Best params: {best_params_poly}")
        print(f"  CV score (MSE): {best_score_poly:.6f}")
        
        # Compare with linear
        print(f"\nLinear regression (baseline):")
        X_bias = np.column_stack([np.ones(n), X])
        beta = np.linalg.lstsq(X_bias, y, rcond=None)[0]
        y_pred_linear = X_bias @ beta
        linear_mse = np.mean((y_pred_linear - y)**2)
        print(f"  Training MSE: {linear_mse:.6f}")
        
        return KernelMethodsPipeline


def run_all_exercises():
    """Run all exercises."""
    print("=" * 70)
    print("HILBERT SPACES - EXERCISES")
    print("=" * 70)
    
    Exercise1.problem()
    Exercise1.solution()
    
    Exercise2.problem()
    Exercise2.solution()
    
    Exercise3.problem()
    Exercise3.solution()
    
    Exercise4.problem()
    Exercise4.solution()
    
    Exercise5.problem()
    Exercise5.solution()
    
    Exercise6.problem()
    Exercise6.solution()
    
    Exercise7.problem()
    Exercise7.solution()
    
    Exercise8.problem()
    Exercise8.solution()
    
    Exercise9.problem()
    Exercise9.solution()
    
    Exercise10.problem()
    Exercise10.solution()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_exercises()
