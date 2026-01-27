"""
Hilbert Spaces - Examples
=========================

Implementations demonstrating Hilbert space concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional


# =============================================================================
# Example 1: Inner Products and Induced Norms
# =============================================================================

def example_1_inner_products():
    """
    Demonstrate different inner products and their properties.
    """
    print("Example 1: Inner Products and Induced Norms")
    print("=" * 60)
    
    def standard_inner_product(x: np.ndarray, y: np.ndarray) -> float:
        """Standard Euclidean inner product: <x, y> = x^T y"""
        return np.dot(x, y)
    
    def weighted_inner_product(x: np.ndarray, y: np.ndarray, 
                              W: np.ndarray) -> float:
        """Weighted inner product: <x, y>_W = x^T W y"""
        return x @ W @ y
    
    def verify_inner_product_axioms(inner_prod: Callable, 
                                   x: np.ndarray, y: np.ndarray, 
                                   z: np.ndarray, a: float, b: float) -> Dict:
        """Verify inner product axioms."""
        results = {}
        
        # Symmetry: <x, y> = <y, x>
        results['Symmetry'] = np.isclose(inner_prod(x, y), inner_prod(y, x))
        
        # Linearity: <ax + by, z> = a<x, z> + b<y, z>
        lhs = inner_prod(a * x + b * y, z)
        rhs = a * inner_prod(x, z) + b * inner_prod(y, z)
        results['Linearity'] = np.isclose(lhs, rhs)
        
        # Positive definiteness
        results['Positive'] = inner_prod(x, x) >= 0
        results['Zero iff zero'] = (inner_prod(np.zeros_like(x), np.zeros_like(x)) == 0)
        
        return results
    
    def induced_norm(x: np.ndarray, inner_prod: Callable) -> float:
        """Norm induced by inner product: ||x|| = sqrt(<x, x>)"""
        return np.sqrt(inner_prod(x, x))
    
    def verify_cauchy_schwarz(x: np.ndarray, y: np.ndarray,
                             inner_prod: Callable) -> bool:
        """Verify |<x, y>| <= ||x|| ||y||"""
        lhs = abs(inner_prod(x, y))
        rhs = induced_norm(x, inner_prod) * induced_norm(y, inner_prod)
        return lhs <= rhs + 1e-10
    
    def verify_parallelogram_law(x: np.ndarray, y: np.ndarray,
                                inner_prod: Callable) -> bool:
        """Verify ||x+y||² + ||x-y||² = 2(||x||² + ||y||²)"""
        norm = lambda v: induced_norm(v, inner_prod)
        lhs = norm(x + y)**2 + norm(x - y)**2
        rhs = 2 * (norm(x)**2 + norm(y)**2)
        return np.isclose(lhs, rhs)
    
    # Test vectors
    x = np.array([1, 2, 3])
    y = np.array([4, -1, 2])
    z = np.array([1, 1, 1])
    a, b = 2.0, -1.5
    
    print("Standard inner product:")
    inner = standard_inner_product
    print(f"  <x, y> = {inner(x, y)}")
    print(f"  ||x|| = {induced_norm(x, inner):.4f}")
    
    print("\n  Axiom verification:")
    for axiom, satisfied in verify_inner_product_axioms(inner, x, y, z, a, b).items():
        print(f"    {axiom}: {'✓' if satisfied else '✗'}")
    
    print(f"\n  Cauchy-Schwarz: {verify_cauchy_schwarz(x, y, inner)}")
    print(f"  Parallelogram law: {verify_parallelogram_law(x, y, inner)}")
    
    # Weighted inner product
    print("\nWeighted inner product (W = diag(1, 2, 3)):")
    W = np.diag([1, 2, 3])
    inner_W = lambda x, y: weighted_inner_product(x, y, W)
    
    print(f"  <x, y>_W = {inner_W(x, y)}")
    print(f"  ||x||_W = {induced_norm(x, inner_W):.4f}")
    
    return standard_inner_product, weighted_inner_product


# =============================================================================
# Example 2: Orthogonality and Orthogonal Complements
# =============================================================================

def example_2_orthogonality():
    """
    Demonstrate orthogonality and orthogonal complements.
    """
    print("\nExample 2: Orthogonality and Orthogonal Complements")
    print("=" * 60)
    
    def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
        """Check if x ⊥ y."""
        return np.isclose(np.dot(x, y), 0)
    
    def orthogonal_complement_basis(A: np.ndarray) -> np.ndarray:
        """
        Find orthonormal basis for column space complement.
        
        For column space of A, the complement is null space of A^T.
        """
        # SVD approach
        U, s, Vh = np.linalg.svd(A)
        rank = np.sum(s > 1e-10)
        
        # Columns of U beyond rank span the complement
        return U[:, rank:]
    
    def verify_direct_sum(M_basis: np.ndarray, 
                         M_perp_basis: np.ndarray, n: int) -> bool:
        """Verify H = M ⊕ M^⊥"""
        # Combined should span R^n
        combined = np.hstack([M_basis, M_perp_basis])
        rank = np.linalg.matrix_rank(combined)
        
        # Check orthogonality between subspaces
        orthogonal = np.allclose(M_basis.T @ M_perp_basis, 0)
        
        return rank == n and orthogonal
    
    def pythagorean_theorem(x: np.ndarray, y: np.ndarray) -> bool:
        """Verify ||x + y||² = ||x||² + ||y||² when x ⊥ y"""
        if not is_orthogonal(x, y):
            return None
        
        lhs = np.linalg.norm(x + y)**2
        rhs = np.linalg.norm(x)**2 + np.linalg.norm(y)**2
        return np.isclose(lhs, rhs)
    
    # Example: Subspace and its complement
    A = np.array([[1, 0],
                  [1, 1],
                  [0, 1],
                  [0, 0]], dtype=float)
    
    print(f"Column space of A (4×2):")
    print(A)
    
    # Get orthonormal basis for column space
    Q, R = np.linalg.qr(A)
    M_basis = Q[:, :np.linalg.matrix_rank(A)]
    
    # Get complement
    M_perp_basis = orthogonal_complement_basis(A)
    
    print(f"\nOrthonormal basis for col(A):")
    print(M_basis)
    
    print(f"\nOrthonormal basis for col(A)^⊥:")
    print(M_perp_basis)
    
    print(f"\nDirect sum verified: {verify_direct_sum(M_basis, M_perp_basis, 4)}")
    
    # Pythagorean theorem
    x = M_basis[:, 0]
    y = M_perp_basis[:, 0]
    print(f"\nPythagorean theorem for orthogonal vectors: {pythagorean_theorem(x, y)}")
    
    return is_orthogonal, orthogonal_complement_basis


# =============================================================================
# Example 3: Gram-Schmidt Orthonormalization
# =============================================================================

def example_3_gram_schmidt():
    """
    Implement and demonstrate Gram-Schmidt process.
    """
    print("\nExample 3: Gram-Schmidt Orthonormalization")
    print("=" * 60)
    
    def gram_schmidt_classical(V: np.ndarray) -> np.ndarray:
        """
        Classical Gram-Schmidt orthonormalization.
        
        u_k = v_k - Σ_{j<k} <v_k, e_j> e_j
        e_k = u_k / ||u_k||
        """
        n, m = V.shape
        Q = np.zeros((n, m))
        
        for k in range(m):
            u = V[:, k].copy()
            
            for j in range(k):
                u -= np.dot(V[:, k], Q[:, j]) * Q[:, j]
            
            norm = np.linalg.norm(u)
            if norm > 1e-10:
                Q[:, k] = u / norm
        
        return Q
    
    def gram_schmidt_modified(V: np.ndarray) -> np.ndarray:
        """
        Modified Gram-Schmidt (more numerically stable).
        
        Subtract projections from current vector, not original.
        """
        n, m = V.shape
        Q = V.copy().astype(float)
        
        for k in range(m):
            # Normalize
            norm = np.linalg.norm(Q[:, k])
            if norm < 1e-10:
                continue
            Q[:, k] /= norm
            
            # Orthogonalize remaining columns
            for j in range(k + 1, m):
                Q[:, j] -= np.dot(Q[:, j], Q[:, k]) * Q[:, k]
        
        return Q
    
    def verify_orthonormal(Q: np.ndarray) -> Tuple[bool, float]:
        """Check if Q is orthonormal."""
        I = Q.T @ Q
        error = np.linalg.norm(I - np.eye(Q.shape[1]))
        return error < 1e-10, error
    
    # Test with linearly independent vectors
    V = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1],
                  [1, 1, 1]], dtype=float)
    
    print("Input vectors (4×3):")
    print(V)
    
    Q_classical = gram_schmidt_classical(V)
    Q_modified = gram_schmidt_modified(V)
    
    print("\nClassical Gram-Schmidt result:")
    print(Q_classical)
    is_ortho, error = verify_orthonormal(Q_classical)
    print(f"Orthonormal: {is_ortho}, error = {error:.2e}")
    
    print("\nModified Gram-Schmidt result:")
    print(Q_modified)
    is_ortho, error = verify_orthonormal(Q_modified)
    print(f"Orthonormal: {is_ortho}, error = {error:.2e}")
    
    # Test numerical stability with ill-conditioned input
    print("\nNumerical stability test (nearly dependent vectors):")
    V_ill = np.array([[1, 1, 1 + 1e-10],
                      [1, 1, 1 + 2e-10],
                      [1, 1, 1 + 3e-10]], dtype=float)
    
    Q_c = gram_schmidt_classical(V_ill)
    Q_m = gram_schmidt_modified(V_ill)
    
    _, error_c = verify_orthonormal(Q_c[:, :2])
    _, error_m = verify_orthonormal(Q_m[:, :2])
    
    print(f"  Classical error: {error_c:.2e}")
    print(f"  Modified error: {error_m:.2e}")
    
    return gram_schmidt_modified


# =============================================================================
# Example 4: Orthogonal Projection
# =============================================================================

def example_4_projection():
    """
    Implement orthogonal projection onto subspaces.
    """
    print("\nExample 4: Orthogonal Projection")
    print("=" * 60)
    
    def projection_onto_span(x: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Project x onto column space of A.
        
        P_A x = A(A^T A)^{-1} A^T x
        """
        # Use QR for numerical stability
        Q, R = np.linalg.qr(A)
        return Q @ (Q.T @ x)
    
    def projection_matrix(A: np.ndarray) -> np.ndarray:
        """
        Compute projection matrix P = A(A^T A)^{-1} A^T
        """
        Q, R = np.linalg.qr(A)
        return Q @ Q.T
    
    def verify_projection_properties(P: np.ndarray) -> Dict:
        """Verify P is an orthogonal projection."""
        results = {}
        
        # P² = P (idempotent)
        results['Idempotent'] = np.allclose(P @ P, P)
        
        # P^T = P (self-adjoint)
        results['Self-adjoint'] = np.allclose(P, P.T)
        
        # ||P|| <= 1
        results['Norm <= 1'] = np.linalg.norm(P, 2) <= 1 + 1e-10
        
        return results
    
    def closest_point_in_subspace(x: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Find closest point to x in col(A)."""
        return projection_onto_span(x, A)
    
    # Example: Project onto a 2D subspace in R³
    A = np.array([[1, 0],
                  [1, 1],
                  [0, 1]], dtype=float)
    
    x = np.array([1, 2, 3], dtype=float)
    
    print(f"Projecting x = {x}")
    print(f"onto column space of A:")
    print(A)
    
    proj_x = projection_onto_span(x, A)
    P = projection_matrix(A)
    
    print(f"\nProjection: P_A(x) = {proj_x}")
    print(f"Residual: x - P_A(x) = {x - proj_x}")
    print(f"||residual|| = {np.linalg.norm(x - proj_x):.4f}")
    
    # Verify residual is orthogonal to column space
    residual = x - proj_x
    print(f"\nResidual orthogonal to col(A): {np.allclose(A.T @ residual, 0)}")
    
    print("\nProjection matrix properties:")
    for prop, satisfied in verify_projection_properties(P).items():
        print(f"  {prop}: {'✓' if satisfied else '✗'}")
    
    # Complementary projection
    print(f"\nComplementary projection:")
    P_perp = np.eye(3) - P
    print(f"  P⊥ x = {P_perp @ x}")
    print(f"  P x + P⊥ x = {P @ x + P_perp @ x}")
    
    return projection_onto_span, projection_matrix


# =============================================================================
# Example 5: Least Squares as Projection
# =============================================================================

def example_5_least_squares():
    """
    Solve least squares via projection theorem.
    """
    print("\nExample 5: Least Squares as Projection")
    print("=" * 60)
    
    def least_squares_projection(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve min ||Ax - b||² using projection.
        
        Solution: x = (A^T A)^{-1} A^T b
        """
        return np.linalg.solve(A.T @ A, A.T @ b)
    
    def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve least squares via QR decomposition.
        
        Ax = b → QRx = b → Rx = Q^T b
        """
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ b)
    
    def least_squares_svd(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve least squares via SVD (most stable).
        
        A = UΣV^T → x = VΣ^{-1}U^T b
        """
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        # Pseudo-inverse approach
        return Vt.T @ (np.diag(1/s) @ (U.T @ b))
    
    def geometric_interpretation(A: np.ndarray, b: np.ndarray,
                                x_hat: np.ndarray) -> Dict:
        """Show geometric meaning of least squares."""
        b_hat = A @ x_hat  # Projection of b onto col(A)
        residual = b - b_hat
        
        return {
            'b_hat (projection)': b_hat,
            'residual': residual,
            'residual_norm': np.linalg.norm(residual),
            'orthogonal_to_colA': np.allclose(A.T @ residual, 0)
        }
    
    # Linear regression example
    np.random.seed(42)
    n, p = 50, 3
    
    # Design matrix with bias term
    X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
    
    # True parameters
    beta_true = np.array([1, 2, -1])
    
    # Noisy observations
    y = X @ beta_true + 0.5 * np.random.randn(n)
    
    print(f"Linear regression: n={n}, p={p}")
    print(f"True parameters: {beta_true}")
    
    # Solve via different methods
    beta_proj = least_squares_projection(X, y)
    beta_qr = least_squares_qr(X, y)
    beta_svd = least_squares_svd(X, y)
    
    print(f"\nEstimated parameters:")
    print(f"  Projection: {beta_proj}")
    print(f"  QR: {beta_qr}")
    print(f"  SVD: {beta_svd}")
    
    print(f"\nGeometric interpretation:")
    geom = geometric_interpretation(X, y, beta_proj)
    print(f"  ||residual|| = {geom['residual_norm']:.4f}")
    print(f"  Residual ⊥ col(X): {geom['orthogonal_to_colA']}")
    
    return least_squares_qr


# =============================================================================
# Example 6: Fourier Series
# =============================================================================

def example_6_fourier():
    """
    Fourier series as orthogonal expansion in L²[0, 2π].
    """
    print("\nExample 6: Fourier Series")
    print("=" * 60)
    
    def fourier_basis_real(n_terms: int, x: np.ndarray) -> np.ndarray:
        """
        Real Fourier basis: 1/√(2π), cos(nx)/√π, sin(nx)/√π
        """
        basis = [np.ones_like(x) / np.sqrt(2 * np.pi)]
        
        for n in range(1, n_terms + 1):
            basis.append(np.cos(n * x) / np.sqrt(np.pi))
            basis.append(np.sin(n * x) / np.sqrt(np.pi))
        
        return np.array(basis)
    
    def fourier_coefficients(f: Callable, n_terms: int,
                            n_points: int = 1000) -> np.ndarray:
        """
        Compute Fourier coefficients via numerical integration.
        
        c_n = <f, e_n> = ∫ f(x) e_n(x) dx
        """
        x = np.linspace(0, 2 * np.pi, n_points)
        dx = 2 * np.pi / n_points
        
        f_vals = f(x)
        basis = fourier_basis_real(n_terms, x)
        
        # Inner products (numerical integration)
        coeffs = np.array([np.sum(f_vals * e) * dx for e in basis])
        
        return coeffs
    
    def fourier_approximation(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Reconstruct function from Fourier coefficients."""
        n_terms = (len(coeffs) - 1) // 2
        basis = fourier_basis_real(n_terms, x)
        
        return sum(c * e for c, e in zip(coeffs, basis))
    
    def parseval_identity(f: Callable, coeffs: np.ndarray,
                         n_points: int = 1000) -> Dict:
        """
        Verify Parseval's identity: ||f||² = Σ|c_n|²
        """
        x = np.linspace(0, 2 * np.pi, n_points)
        dx = 2 * np.pi / n_points
        
        # ||f||² via integration
        f_vals = f(x)
        norm_sq_integral = np.sum(f_vals**2) * dx
        
        # Σ|c_n|²
        norm_sq_coeffs = np.sum(coeffs**2)
        
        return {
            '||f||² (integral)': norm_sq_integral,
            'Σ|c_n|² (Parseval)': norm_sq_coeffs,
            'Relative error': abs(norm_sq_integral - norm_sq_coeffs) / norm_sq_integral
        }
    
    # Example: Approximate sawtooth wave
    def sawtooth(x):
        return x - np.pi
    
    print("Approximating sawtooth wave f(x) = x - π on [0, 2π]")
    
    x = np.linspace(0, 2 * np.pi, 200)
    
    for n_terms in [1, 3, 10, 30]:
        coeffs = fourier_coefficients(sawtooth, n_terms)
        approx = fourier_approximation(coeffs, x)
        
        error = np.sqrt(np.mean((sawtooth(x) - approx)**2))
        print(f"\n  {n_terms} terms: RMSE = {error:.4f}")
        
        if n_terms == 10:
            parseval = parseval_identity(sawtooth, coeffs)
            print(f"  Parseval verification:")
            for key, val in parseval.items():
                print(f"    {key}: {val:.6f}")
    
    return fourier_coefficients, fourier_approximation


# =============================================================================
# Example 7: Reproducing Kernel Hilbert Space (RKHS)
# =============================================================================

def example_7_rkhs():
    """
    Demonstrate RKHS concepts with common kernels.
    """
    print("\nExample 7: Reproducing Kernel Hilbert Space")
    print("=" * 60)
    
    def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
        """Linear kernel: K(x, y) = x^T y"""
        return np.dot(x, y)
    
    def polynomial_kernel(x: np.ndarray, y: np.ndarray, 
                         degree: int = 2, c: float = 1) -> float:
        """Polynomial kernel: K(x, y) = (x^T y + c)^d"""
        return (np.dot(x, y) + c) ** degree
    
    def rbf_kernel(x: np.ndarray, y: np.ndarray, 
                  sigma: float = 1.0) -> float:
        """RBF/Gaussian kernel: K(x, y) = exp(-||x-y||²/2σ²)"""
        return np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
    
    def laplacian_kernel(x: np.ndarray, y: np.ndarray,
                        sigma: float = 1.0) -> float:
        """Laplacian kernel: K(x, y) = exp(-||x-y||₁/σ)"""
        return np.exp(-np.linalg.norm(x - y, 1) / sigma)
    
    def kernel_matrix(X: np.ndarray, kernel: Callable) -> np.ndarray:
        """Compute kernel (Gram) matrix K_ij = K(x_i, x_j)"""
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j])
        return K
    
    def verify_positive_semidefinite(K: np.ndarray) -> bool:
        """Check if kernel matrix is PSD."""
        eigenvalues = np.linalg.eigvalsh(K)
        return np.all(eigenvalues >= -1e-10)
    
    def reproducing_property_demo(X: np.ndarray, kernel: Callable,
                                 alpha: np.ndarray) -> Dict:
        """
        Demonstrate reproducing property.
        
        For f(x) = Σ α_i K(x_i, x), we have f(x_j) = Σ α_i K(x_i, x_j)
        """
        K = kernel_matrix(X, kernel)
        
        # f(x_j) via kernel expansion
        f_values = K @ alpha
        
        return {
            'function_values': f_values,
            'kernel_matrix_shape': K.shape,
            'is_psd': verify_positive_semidefinite(K)
        }
    
    # Example data
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    alpha = np.array([1, -1, 1, -1], dtype=float)
    
    print("Data points:")
    print(X)
    
    print("\nKernel matrices (verify PSD):")
    
    kernels = {
        'Linear': lambda x, y: linear_kernel(x, y),
        'Poly(d=2)': lambda x, y: polynomial_kernel(x, y, 2),
        'RBF(σ=1)': lambda x, y: rbf_kernel(x, y, 1),
        'Laplacian(σ=1)': lambda x, y: laplacian_kernel(x, y, 1)
    }
    
    for name, kernel in kernels.items():
        K = kernel_matrix(X, kernel)
        is_psd = verify_positive_semidefinite(K)
        print(f"\n  {name}:")
        print(f"    K =\n{K}")
        print(f"    PSD: {is_psd}")
    
    return kernel_matrix, rbf_kernel


# =============================================================================
# Example 8: Kernel Ridge Regression
# =============================================================================

def example_8_kernel_ridge():
    """
    Implement kernel ridge regression.
    """
    print("\nExample 8: Kernel Ridge Regression")
    print("=" * 60)
    
    class KernelRidgeRegression:
        """
        Kernel ridge regression.
        
        min Σ(y_i - f(x_i))² + λ||f||_H²
        
        Solution: f*(x) = K(x, X)(K + λI)^{-1} y
        """
        
        def __init__(self, kernel: Callable, lambda_: float = 1.0):
            self.kernel = kernel
            self.lambda_ = lambda_
            self.X_train = None
            self.alpha = None
        
        def fit(self, X: np.ndarray, y: np.ndarray):
            """Fit the model."""
            self.X_train = X
            n = len(X)
            
            # Kernel matrix
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])
            
            # α = (K + λI)^{-1} y
            self.alpha = np.linalg.solve(K + self.lambda_ * np.eye(n), y)
            
            return self
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict for new points."""
            predictions = []
            
            for x in X:
                k_x = np.array([self.kernel(x, x_i) for x_i in self.X_train])
                predictions.append(k_x @ self.alpha)
            
            return np.array(predictions)
        
        def rkhs_norm_squared(self) -> float:
            """||f||_H² = α^T K α"""
            K = np.zeros((len(self.X_train), len(self.X_train)))
            for i in range(len(self.X_train)):
                for j in range(len(self.X_train)):
                    K[i, j] = self.kernel(self.X_train[i], self.X_train[j])
            
            return self.alpha @ K @ self.alpha
    
    # Generate nonlinear data
    np.random.seed(42)
    n = 50
    X_train = np.sort(np.random.uniform(-3, 3, n))
    y_train = np.sin(X_train) + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(-3, 3, 100)
    y_true = np.sin(X_test)
    
    print("Fitting sin(x) with different kernels:")
    
    # RBF kernel
    rbf = lambda x, y: np.exp(-(x - y)**2 / 2)
    model_rbf = KernelRidgeRegression(rbf, lambda_=0.1)
    model_rbf.fit(X_train.reshape(-1, 1), y_train)
    y_pred_rbf = model_rbf.predict(X_test.reshape(-1, 1))
    
    print(f"\n  RBF kernel:")
    print(f"    MSE: {np.mean((y_pred_rbf - y_true)**2):.6f}")
    print(f"    ||f||_H²: {model_rbf.rkhs_norm_squared():.4f}")
    
    # Polynomial kernel
    poly = lambda x, y: (x * y + 1)**3
    model_poly = KernelRidgeRegression(poly, lambda_=0.1)
    model_poly.fit(X_train.reshape(-1, 1), y_train)
    y_pred_poly = model_poly.predict(X_test.reshape(-1, 1))
    
    print(f"\n  Polynomial kernel (d=3):")
    print(f"    MSE: {np.mean((y_pred_poly - y_true)**2):.6f}")
    print(f"    ||f||_H²: {model_poly.rkhs_norm_squared():.4f}")
    
    return KernelRidgeRegression


# =============================================================================
# Example 9: Mercer's Theorem and Kernel Eigendecomposition
# =============================================================================

def example_9_mercer():
    """
    Demonstrate Mercer's theorem and kernel PCA.
    """
    print("\nExample 9: Mercer's Theorem and Kernel Eigendecomposition")
    print("=" * 60)
    
    def kernel_eigendecomposition(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition of kernel matrix.
        
        K = Σ λ_i φ_i φ_i^T
        """
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def kernel_pca(K: np.ndarray, n_components: int) -> np.ndarray:
        """
        Kernel PCA.
        
        Project data to principal components in feature space.
        """
        # Center kernel matrix
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        # Eigendecomposition
        eigenvalues, eigenvectors = kernel_eigendecomposition(K_centered)
        
        # Project (α_k = eigenvector / sqrt(eigenvalue))
        alpha = eigenvectors[:, :n_components] / np.sqrt(eigenvalues[:n_components] + 1e-10)
        
        return K_centered @ alpha
    
    def nystrom_approximation(X: np.ndarray, kernel: Callable,
                             n_landmarks: int) -> Callable:
        """
        Nyström approximation of kernel.
        
        K ≈ K_nm K_mm^{-1} K_mn
        """
        n = len(X)
        
        # Select landmarks randomly
        np.random.seed(42)
        landmark_idx = np.random.choice(n, n_landmarks, replace=False)
        landmarks = X[landmark_idx]
        
        # Compute K_mm
        K_mm = np.zeros((n_landmarks, n_landmarks))
        for i in range(n_landmarks):
            for j in range(n_landmarks):
                K_mm[i, j] = kernel(landmarks[i], landmarks[j])
        
        # Eigendecomposition of K_mm
        eigs, vecs = np.linalg.eigh(K_mm)
        eigs = np.maximum(eigs, 1e-10)
        K_mm_inv_sqrt = vecs @ np.diag(1 / np.sqrt(eigs)) @ vecs.T
        
        def approx_feature_map(x):
            k_xm = np.array([kernel(x, m) for m in landmarks])
            return K_mm_inv_sqrt @ k_xm
        
        return approx_feature_map
    
    # Example: Kernel PCA on circular data
    np.random.seed(42)
    n = 100
    
    # Circular data
    theta = np.random.uniform(0, 2*np.pi, n)
    r = 1 + 0.5 * np.random.randn(n)
    X = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    print("Kernel PCA on circular data:")
    
    # RBF kernel matrix
    sigma = 0.5
    rbf = lambda x, y: np.exp(-np.linalg.norm(x - y)**2 / (2 * sigma**2))
    
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = rbf(X[i], X[j])
    
    eigenvalues, _ = kernel_eigendecomposition(K)
    
    print(f"  Top eigenvalues: {eigenvalues[:5]}")
    print(f"  Eigenvalue sum: {sum(eigenvalues):.4f}")
    print(f"  Trace(K): {np.trace(K):.4f}")
    
    # Project to 2 components
    X_projected = kernel_pca(K, 2)
    print(f"\n  Projected data shape: {X_projected.shape}")
    
    # Nyström approximation
    print(f"\n  Nyström approximation (20 landmarks):")
    feature_map = nystrom_approximation(X, rbf, 20)
    
    # Verify approximation
    K_approx = np.zeros((n, n))
    features = [feature_map(x) for x in X]
    for i in range(n):
        for j in range(n):
            K_approx[i, j] = np.dot(features[i], features[j])
    
    approx_error = np.linalg.norm(K - K_approx) / np.linalg.norm(K)
    print(f"    Relative error: {approx_error:.4f}")
    
    return kernel_pca, nystrom_approximation


# =============================================================================
# Example 10: Neural Tangent Kernel
# =============================================================================

def example_10_ntk():
    """
    Demonstrate Neural Tangent Kernel for infinite-width networks.
    """
    print("\nExample 10: Neural Tangent Kernel")
    print("=" * 60)
    
    def relu(x):
        return np.maximum(0, x)
    
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    def ntk_single_layer(x1: np.ndarray, x2: np.ndarray,
                        sigma_w: float = 1.0,
                        sigma_b: float = 0.0) -> float:
        """
        NTK for single hidden layer ReLU network.
        
        K_NTK(x, x') = σ_b² + σ_w² <x, x'> [1 + analytical term]
        
        Simplified version using empirical computation.
        """
        # For single layer: K_NTK ∝ K_NNGP + derivative term
        
        # NNGP kernel (covariance of pre-activation)
        norm1 = np.linalg.norm(x1)
        norm2 = np.linalg.norm(x2)
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return sigma_b ** 2
        
        cos_theta = np.clip(np.dot(x1, x2) / (norm1 * norm2), -1, 1)
        theta = np.arccos(cos_theta)
        
        # ReLU NNGP kernel formula
        K_nngp = (sigma_w**2 / (2 * np.pi)) * norm1 * norm2 * (
            np.sin(theta) + (np.pi - theta) * np.cos(theta)
        ) + sigma_b**2
        
        # NTK for ReLU: includes derivative term
        K_ntk = K_nngp + (sigma_w**2 / (2 * np.pi)) * norm1 * norm2 * (np.pi - theta)
        
        return K_ntk
    
    def empirical_ntk(f: Callable, x1: np.ndarray, x2: np.ndarray,
                     theta: np.ndarray, epsilon: float = 1e-5) -> float:
        """
        Compute empirical NTK via finite differences.
        
        K_NTK(x, x') = ∇_θ f(x)^T ∇_θ f(x')
        """
        def grad_f(x):
            grad = np.zeros(len(theta))
            f0 = f(x, theta)
            
            for i in range(len(theta)):
                theta_plus = theta.copy()
                theta_plus[i] += epsilon
                f_plus = f(x, theta_plus)
                grad[i] = (f_plus - f0) / epsilon
            
            return grad
        
        grad1 = grad_f(x1)
        grad2 = grad_f(x2)
        
        return np.dot(grad1, grad2)
    
    def simple_network(x, theta):
        """Simple 2-layer network for NTK demo."""
        d = len(x)
        h = len(theta) // (d + 1) - 1
        
        # Parse parameters
        idx = 0
        W1 = theta[idx:idx + h*d].reshape(h, d)
        idx += h * d
        b1 = theta[idx:idx + h]
        idx += h
        W2 = theta[idx:idx + h]
        
        # Forward
        z1 = W1 @ x + b1
        a1 = relu(z1)
        return np.dot(W2, a1)
    
    # Example
    print("NTK for ReLU network:")
    
    d = 3  # input dimension
    x1 = np.array([1.0, 0.5, 0.2])
    x2 = np.array([0.8, 0.3, 0.5])
    
    # Analytical NTK
    K_analytical = ntk_single_layer(x1, x2)
    print(f"  Analytical NTK(x1, x2) = {K_analytical:.4f}")
    
    # Empirical NTK with random initialization
    np.random.seed(42)
    h = 100  # hidden units
    theta = np.random.randn(h * d + h + h) / np.sqrt(d)  # W1, b1, W2
    
    K_empirical = empirical_ntk(simple_network, x1, x2, theta)
    print(f"  Empirical NTK(x1, x2) = {K_empirical:.4f}")
    
    # NTK matrix for training
    print(f"\n  NTK matrix (4 points):")
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    
    K_ntk = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            K_ntk[i, j] = ntk_single_layer(X[i], X[j])
    
    print(K_ntk)
    print(f"  Is PSD: {np.all(np.linalg.eigvalsh(K_ntk) >= -1e-10)}")
    
    return ntk_single_layer


def run_all_examples():
    """Run all Hilbert space examples."""
    print("=" * 70)
    print("HILBERT SPACES - EXAMPLES")
    print("=" * 70)
    
    example_1_inner_products()
    example_2_orthogonality()
    example_3_gram_schmidt()
    example_4_projection()
    example_5_least_squares()
    example_6_fourier()
    example_7_rkhs()
    example_8_kernel_ridge()
    example_9_mercer()
    example_10_ntk()
    
    print("\n" + "=" * 70)
    print("All Hilbert space examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
