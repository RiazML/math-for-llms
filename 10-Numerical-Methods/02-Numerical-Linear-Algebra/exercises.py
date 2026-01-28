"""
Numerical Linear Algebra - Exercises
====================================
Practice problems for matrix computations.
"""

import numpy as np
from scipy import linalg as la


def exercise_1_lu_decomposition():
    """
    EXERCISE 1: Implement LU Decomposition
    ======================================
    
    Implement LU decomposition without pivoting.
    
    Tasks:
    a) Implement lu_decomposition(A) returning L, U
    b) Handle the case where pivot is zero
    c) Verify with a test matrix
    """
    print("=" * 60)
    print("EXERCISE 1: LU Decomposition")
    print("=" * 60)
    
    # YOUR CODE HERE
    def lu_decomposition(A):
        """
        Decompose A = LU without pivoting.
        
        Returns:
            L: Lower triangular with ones on diagonal
            U: Upper triangular
        """
        # TODO: Implement Gaussian elimination
        # For k = 0 to n-2:
        #   For i = k+1 to n-1:
        #     L[i,k] = A[i,k] / A[k,k]
        #     For j = k to n-1:
        #       A[i,j] -= L[i,k] * A[k,j]
        pass
    
    # Test matrix
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)


def exercise_1_solution():
    """Solution for Exercise 1."""
    print("=" * 60)
    print("SOLUTION 1: LU Decomposition")
    print("=" * 60)
    
    def lu_decomposition(A):
        A = A.copy().astype(float)
        n = A.shape[0]
        L = np.eye(n)
        
        for k in range(n - 1):
            if abs(A[k, k]) < 1e-12:
                raise ValueError("Zero pivot encountered")
            
            for i in range(k + 1, n):
                L[i, k] = A[i, k] / A[k, k]
                for j in range(k, n):
                    A[i, j] -= L[i, k] * A[k, j]
        
        U = A
        return L, U
    
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    L, U = lu_decomposition(A)
    
    print(f"\nL:\n{L}")
    print(f"\nU:\n{U}")
    print(f"\nL @ U:\n{L @ U}")
    print(f"\nReconstruction error: {np.linalg.norm(A - L @ U):.2e}")


def exercise_2_cholesky():
    """
    EXERCISE 2: Implement Cholesky Decomposition
    ============================================
    
    For symmetric positive definite matrices.
    
    Tasks:
    a) Implement cholesky(A) returning L such that A = LL^T
    b) Check if matrix is positive definite
    c) Compare performance with LU
    """
    print("\n" + "=" * 60)
    print("EXERCISE 2: Cholesky Decomposition")
    print("=" * 60)
    
    # YOUR CODE HERE
    def cholesky(A):
        """
        Decompose A = LL^T.
        
        L[i,i] = sqrt(A[i,i] - sum(L[i,k]² for k < i))
        L[j,i] = (A[j,i] - sum(L[j,k]*L[i,k] for k < i)) / L[i,i]
        """
        # TODO: Implement
        pass


def exercise_2_solution():
    """Solution for Exercise 2."""
    print("\n" + "=" * 60)
    print("SOLUTION 2: Cholesky Decomposition")
    print("=" * 60)
    
    def cholesky(A):
        A = A.copy().astype(float)
        n = A.shape[0]
        L = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    val = A[i, i] - np.sum(L[i, :j] ** 2)
                    if val <= 0:
                        raise ValueError("Matrix not positive definite")
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
        
        return L
    
    # Create SPD matrix
    np.random.seed(42)
    X = np.random.randn(4, 4)
    A = X @ X.T + 0.1 * np.eye(4)
    
    print("SPD matrix A:")
    print(A.round(3))
    
    L = cholesky(A)
    
    print(f"\nL:\n{L.round(3)}")
    print(f"\nL @ L.T:\n{(L @ L.T).round(3)}")
    print(f"\nReconstruction error: {np.linalg.norm(A - L @ L.T):.2e}")


def exercise_3_qr_householder():
    """
    EXERCISE 3: QR using Householder Reflections
    =============================================
    
    Implement QR decomposition using Householder reflections.
    
    Householder reflection: H = I - 2vv^T / (v^T v)
    
    Tasks:
    a) Implement householder_vector(x) to zero out below first element
    b) Implement qr_householder(A)
    c) Compare orthogonality with Gram-Schmidt
    """
    print("\n" + "=" * 60)
    print("EXERCISE 3: Householder QR")
    print("=" * 60)
    
    # YOUR CODE HERE
    def householder_vector(x):
        """
        Compute Householder vector v such that Hx = ||x|| * e_1.
        
        v = x + sign(x_0) * ||x|| * e_1
        """
        # TODO: Implement
        pass


def exercise_3_solution():
    """Solution for Exercise 3."""
    print("\n" + "=" * 60)
    print("SOLUTION 3: Householder QR")
    print("=" * 60)
    
    def householder_vector(x):
        """Compute Householder vector."""
        x = np.asarray(x, dtype=float)
        n = len(x)
        norm_x = np.linalg.norm(x)
        
        if norm_x == 0:
            return np.zeros(n), 0
        
        # v = x + sign(x_0) * ||x|| * e_1
        v = x.copy()
        v[0] += np.sign(x[0]) * norm_x
        
        beta = 2 / (v @ v) if v @ v > 0 else 0
        return v, beta
    
    def qr_householder(A):
        """QR decomposition using Householder reflections."""
        A = A.copy().astype(float)
        m, n = A.shape
        Q = np.eye(m)
        
        for j in range(min(m-1, n)):
            # Get column below diagonal
            x = A[j:, j]
            
            # Compute Householder vector
            v, beta = householder_vector(x)
            
            if beta == 0:
                continue
            
            # Apply H = I - beta * v @ v.T to remaining matrix
            A[j:, j:] -= beta * np.outer(v, v @ A[j:, j:])
            
            # Accumulate Q
            Q[:, j:] -= beta * np.outer(Q[:, j:] @ v, v)
        
        R = A
        return Q, R
    
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    Q, R = qr_householder(A)
    
    print(f"\nQ:\n{Q.round(4)}")
    print(f"\nR:\n{R.round(4)}")
    print(f"\nQ @ R:\n{(Q @ R).round(4)}")
    print(f"\nQ^T @ Q (should be I):\n{(Q.T @ Q).round(10)}")
    print(f"\nOrthogonality error: {np.linalg.norm(Q.T @ Q - np.eye(3)):.2e}")


def exercise_4_power_iteration():
    """
    EXERCISE 4: Power Iteration with Deflation
    ==========================================
    
    Find multiple eigenvalues using deflation.
    
    Tasks:
    a) Implement power_iteration for largest eigenvalue
    b) Use deflation to find second largest
    c) Implement inverse iteration for smallest
    """
    print("\n" + "=" * 60)
    print("EXERCISE 4: Power Iteration")
    print("=" * 60)
    
    # YOUR CODE HERE
    def power_iteration(A, tol=1e-10, max_iter=1000):
        """Find largest eigenvalue and eigenvector."""
        # TODO: Implement
        pass
    
    def deflate(A, eigenvalue, eigenvector):
        """
        Remove contribution of eigenvalue/eigenvector from A.
        A' = A - λ * v * v^T
        """
        # TODO: Implement
        pass


def exercise_4_solution():
    """Solution for Exercise 4."""
    print("\n" + "=" * 60)
    print("SOLUTION 4: Power Iteration")
    print("=" * 60)
    
    def power_iteration(A, tol=1e-10, max_iter=1000):
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            w = A @ v
            v_new = w / np.linalg.norm(w)
            eigenvalue = v_new @ A @ v_new
            
            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new
        
        return eigenvalue, v_new
    
    def inverse_iteration(A, tol=1e-10, max_iter=1000):
        """Find smallest eigenvalue using inverse iteration."""
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(max_iter):
            w = np.linalg.solve(A, v)
            v_new = w / np.linalg.norm(w)
            eigenvalue = v_new @ A @ v_new
            
            if np.linalg.norm(v_new - v) < tol:
                break
            v = v_new
        
        return eigenvalue, v_new
    
    def deflate(A, eigenvalue, eigenvector):
        v = eigenvector.reshape(-1, 1)
        return A - eigenvalue * v @ v.T
    
    # Symmetric matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 1],
        [1, 1, 2]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    # True eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"\nTrue eigenvalues: {np.sort(eigenvalues)[::-1]}")
    
    # Power iteration
    lambda1, v1 = power_iteration(A)
    print(f"\nPower iteration (largest): λ = {lambda1:.6f}")
    
    # Deflation
    A_deflated = deflate(A, lambda1, v1)
    lambda2, v2 = power_iteration(A_deflated)
    print(f"After deflation (2nd): λ = {lambda2:.6f}")
    
    # Inverse iteration (smallest)
    lambda_min, v_min = inverse_iteration(A)
    print(f"Inverse iteration (smallest): λ = {lambda_min:.6f}")


def exercise_5_conjugate_gradient():
    """
    EXERCISE 5: Conjugate Gradient Method
    =====================================
    
    Implement CG for Ax = b where A is SPD.
    
    Tasks:
    a) Implement conjugate_gradient(A, b, tol)
    b) Track and plot convergence
    c) Add preconditioning
    """
    print("\n" + "=" * 60)
    print("EXERCISE 5: Conjugate Gradient")
    print("=" * 60)
    
    # YOUR CODE HERE
    def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
        """
        Solve Ax = b using conjugate gradient.
        
        Algorithm:
        r₀ = b - Ax₀
        p₀ = r₀
        For k = 0, 1, 2, ...
            αₖ = (rₖ·rₖ)/(pₖ·Apₖ)
            xₖ₊₁ = xₖ + αₖpₖ
            rₖ₊₁ = rₖ - αₖApₖ
            βₖ = (rₖ₊₁·rₖ₊₁)/(rₖ·rₖ)
            pₖ₊₁ = rₖ₊₁ + βₖpₖ
        """
        # TODO: Implement
        pass


def exercise_5_solution():
    """Solution for Exercise 5."""
    print("\n" + "=" * 60)
    print("SOLUTION 5: Conjugate Gradient")
    print("=" * 60)
    
    def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if max_iter is None:
            max_iter = n
        
        r = b - A @ x
        p = r.copy()
        rs_old = r @ r
        residuals = [np.sqrt(rs_old)]
        
        for i in range(max_iter):
            Ap = A @ p
            alpha = rs_old / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r @ r
            residuals.append(np.sqrt(rs_new))
            
            if np.sqrt(rs_new) < tol:
                break
            
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        
        return x, residuals
    
    # Create SPD matrix
    np.random.seed(42)
    n = 20
    X = np.random.randn(n, n)
    A = X @ X.T + np.eye(n)
    b = np.random.randn(n)
    
    print(f"System size: {n}×{n}")
    print(f"Condition number: {np.linalg.cond(A):.2f}")
    
    x_cg, residuals = conjugate_gradient(A, b)
    x_true = np.linalg.solve(A, b)
    
    print(f"\nConverged in {len(residuals)-1} iterations")
    print(f"Final residual: {residuals[-1]:.2e}")
    print(f"Solution error: {np.linalg.norm(x_cg - x_true):.2e}")


def exercise_6_svd_compression():
    """
    EXERCISE 6: Image Compression with SVD
    ======================================
    
    Use truncated SVD for compression.
    
    Tasks:
    a) Create a synthetic "image" matrix
    b) Compress with different ranks
    c) Calculate compression ratio and error
    """
    print("\n" + "=" * 60)
    print("EXERCISE 6: SVD Compression")
    print("=" * 60)
    
    # YOUR CODE HERE
    def compress_svd(A, k):
        """Compress matrix A to rank k using SVD."""
        # TODO: Implement
        pass
    
    def compression_ratio(m, n, k):
        """Calculate compression ratio for rank-k approximation."""
        # Original: m * n
        # Compressed: k * (m + n + 1) for U_k, V_k, s_k
        # TODO: Implement
        pass


def exercise_6_solution():
    """Solution for Exercise 6."""
    print("\n" + "=" * 60)
    print("SOLUTION 6: SVD Compression")
    print("=" * 60)
    
    def compress_svd(A, k):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    def compression_ratio(m, n, k):
        original = m * n
        compressed = k * (m + n + 1)
        return compressed / original
    
    # Create image-like matrix with structure
    np.random.seed(42)
    m, n = 100, 80
    
    # Low-frequency content + noise
    x = np.linspace(0, 4*np.pi, m)
    y = np.linspace(0, 4*np.pi, n)
    X, Y = np.meshgrid(y, x)
    A = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X) + 0.1 * np.random.randn(m, n)
    
    print(f"Image size: {m}×{n}")
    
    # SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    print(f"\nSingular value decay:")
    print(f"  σ₁ = {s[0]:.2f}, σ₁₀ = {s[9]:.2f}, σ₂₀ = {s[19]:.2f}")
    
    print(f"\n{'Rank k':<10} {'RMSE':<12} {'Compression':<15} {'Storage':<10}")
    print("-" * 50)
    
    for k in [1, 5, 10, 20, 40]:
        A_k = compress_svd(A, k)
        rmse = np.sqrt(np.mean((A - A_k) ** 2))
        ratio = compression_ratio(m, n, k)
        
        print(f"{k:<10} {rmse:<12.4f} {ratio:<15.1%} {k*(m+n+1):<10}")


def exercise_7_least_squares():
    """
    EXERCISE 7: Least Squares Multiple Ways
    =======================================
    
    Solve min ||Ax - b||² using different methods.
    
    Tasks:
    a) Normal equations: A^TAx = A^Tb
    b) QR decomposition
    c) SVD pseudoinverse
    d) Compare numerical stability
    """
    print("\n" + "=" * 60)
    print("EXERCISE 7: Least Squares Methods")
    print("=" * 60)
    
    # YOUR CODE HERE
    def least_squares_normal(A, b):
        """Solve via normal equations."""
        # TODO: x = (A^T A)^{-1} A^T b
        pass
    
    def least_squares_qr(A, b):
        """Solve via QR decomposition."""
        # TODO: Rx = Q^T b
        pass
    
    def least_squares_svd(A, b):
        """Solve via SVD pseudoinverse."""
        # TODO: x = V Σ^{-1} U^T b
        pass


def exercise_7_solution():
    """Solution for Exercise 7."""
    print("\n" + "=" * 60)
    print("SOLUTION 7: Least Squares Methods")
    print("=" * 60)
    
    def least_squares_normal(A, b):
        return np.linalg.solve(A.T @ A, A.T @ b)
    
    def least_squares_qr(A, b):
        Q, R = np.linalg.qr(A)
        return la.solve_triangular(R, Q.T @ b)
    
    def least_squares_svd(A, b, tol=1e-10):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        s_inv = np.where(s > tol, 1/s, 0)
        return Vt.T @ (s_inv * (U.T @ b))
    
    # Well-conditioned problem
    np.random.seed(42)
    m, n = 100, 10
    A = np.random.randn(m, n)
    x_true = np.random.randn(n)
    b = A @ x_true + 0.1 * np.random.randn(m)
    
    print("Well-conditioned problem:")
    print(f"  Condition number: {np.linalg.cond(A):.2f}")
    
    x_normal = least_squares_normal(A, b)
    x_qr = least_squares_qr(A, b)
    x_svd = least_squares_svd(A, b)
    
    print(f"\nResiduals ||Ax - b||:")
    print(f"  Normal equations: {np.linalg.norm(A @ x_normal - b):.6f}")
    print(f"  QR:               {np.linalg.norm(A @ x_qr - b):.6f}")
    print(f"  SVD:              {np.linalg.norm(A @ x_svd - b):.6f}")
    
    # Ill-conditioned problem
    A_bad = np.vander(np.linspace(0, 1, m), n)  # Vandermonde - ill-conditioned
    b_bad = np.sin(np.linspace(0, np.pi, m))
    
    print(f"\nIll-conditioned problem:")
    print(f"  Condition number: {np.linalg.cond(A_bad):.2e}")
    
    x_normal = least_squares_normal(A_bad, b_bad)
    x_qr = least_squares_qr(A_bad, b_bad)
    x_svd = least_squares_svd(A_bad, b_bad)
    
    print(f"\nResiduals:")
    print(f"  Normal equations: {np.linalg.norm(A_bad @ x_normal - b_bad):.6f}")
    print(f"  QR:               {np.linalg.norm(A_bad @ x_qr - b_bad):.6f}")
    print(f"  SVD:              {np.linalg.norm(A_bad @ x_svd - b_bad):.6f}")


def exercise_8_sparse_systems():
    """
    EXERCISE 8: Sparse System Solving
    =================================
    
    Compare methods for sparse systems.
    
    Tasks:
    a) Create sparse tridiagonal matrix
    b) Solve with direct method
    c) Solve with iterative method
    d) Compare performance
    """
    print("\n" + "=" * 60)
    print("EXERCISE 8: Sparse Systems")
    print("=" * 60)
    
    # YOUR CODE HERE
    def create_tridiagonal(n, diag, off_diag):
        """Create sparse tridiagonal matrix."""
        # TODO: Create CSR matrix
        pass
    
    def jacobi_iteration(A, b, x0=None, tol=1e-10, max_iter=1000):
        """Jacobi iterative method."""
        # x_{k+1} = D^{-1}(b - (L+U)x_k)
        # TODO: Implement
        pass


def exercise_8_solution():
    """Solution for Exercise 8."""
    print("\n" + "=" * 60)
    print("SOLUTION 8: Sparse Systems")
    print("=" * 60)
    
    from scipy.sparse import diags, csr_matrix
    from scipy.sparse.linalg import spsolve, cg
    import time
    
    def create_tridiagonal(n, diag, off_diag):
        return diags([off_diag, diag, off_diag], [-1, 0, 1], 
                     shape=(n, n), format='csr')
    
    def jacobi_iteration(A, b, x0=None, tol=1e-10, max_iter=1000):
        A = csr_matrix(A)
        n = len(b)
        x = np.zeros(n) if x0 is None else x0.copy()
        
        D_inv = 1.0 / A.diagonal()
        
        for i in range(max_iter):
            r = b - A @ x
            if np.linalg.norm(r) < tol:
                return x, i+1
            x = x + D_inv * r
        
        return x, max_iter
    
    # Create sparse system
    n = 1000
    A = create_tridiagonal(n, 4.0, -1.0)
    b = np.ones(n)
    
    print(f"System size: {n}×{n}")
    print(f"Non-zeros: {A.nnz} ({100*A.nnz/n**2:.2f}%)")
    
    # Direct solve
    start = time.time()
    x_direct = spsolve(A, b)
    time_direct = time.time() - start
    
    # CG
    start = time.time()
    x_cg, info = cg(A, b, tol=1e-10)
    time_cg = time.time() - start
    
    # Jacobi
    start = time.time()
    x_jacobi, iters = jacobi_iteration(A, b, tol=1e-6)
    time_jacobi = time.time() - start
    
    print(f"\nSolve times:")
    print(f"  Direct (spsolve): {time_direct*1000:.2f} ms")
    print(f"  Conjugate Gradient: {time_cg*1000:.2f} ms")
    print(f"  Jacobi ({iters} iters): {time_jacobi*1000:.2f} ms")


def exercise_9_matrix_exponential():
    """
    EXERCISE 9: Matrix Exponential
    ==============================
    
    Compute e^A using different methods.
    
    Tasks:
    a) Taylor series: e^A = I + A + A²/2! + ...
    b) Eigendecomposition: e^A = V e^Λ V^{-1}
    c) Scaling and squaring
    """
    print("\n" + "=" * 60)
    print("EXERCISE 9: Matrix Exponential")
    print("=" * 60)
    
    # YOUR CODE HERE
    def matrix_exp_taylor(A, terms=20):
        """Compute e^A using Taylor series."""
        # TODO: Implement
        pass
    
    def matrix_exp_eigen(A):
        """Compute e^A using eigendecomposition."""
        # e^A = V diag(e^λ) V^{-1}
        # TODO: Implement
        pass


def exercise_9_solution():
    """Solution for Exercise 9."""
    print("\n" + "=" * 60)
    print("SOLUTION 9: Matrix Exponential")
    print("=" * 60)
    
    def matrix_exp_taylor(A, terms=20):
        n = A.shape[0]
        result = np.eye(n)
        term = np.eye(n)
        
        for k in range(1, terms):
            term = term @ A / k
            result = result + term
        
        return result
    
    def matrix_exp_eigen(A):
        eigenvalues, V = np.linalg.eig(A)
        exp_diag = np.diag(np.exp(eigenvalues))
        return V @ exp_diag @ np.linalg.inv(V)
    
    # Test matrix
    A = np.array([
        [0, 1],
        [-1, 0]
    ], dtype=float)  # Rotation generator
    
    print("Matrix A (rotation generator):")
    print(A)
    
    exp_A_taylor = matrix_exp_taylor(A, 30)
    exp_A_eigen = matrix_exp_eigen(A)
    exp_A_scipy = la.expm(A)
    
    print(f"\nTaylor series (30 terms):\n{exp_A_taylor.round(6)}")
    print(f"\nEigendecomposition:\n{np.real(exp_A_eigen).round(6)}")
    print(f"\nScipy expm:\n{exp_A_scipy.round(6)}")
    
    # For A = [[0,1],[-1,0]], e^A is a rotation matrix
    print(f"\nExpected (rotation by 1 radian):")
    print(f"  [[cos(1), sin(1)], [-sin(1), cos(1)]]")
    print(f"  [[{np.cos(1):.6f}, {np.sin(1):.6f}], [{-np.sin(1):.6f}, {np.cos(1):.6f}]]")


def exercise_10_schur_decomposition():
    """
    EXERCISE 10: Schur Decomposition
    ================================
    
    Factor A = QTQ* where T is upper triangular.
    
    Tasks:
    a) Use scipy.linalg.schur
    b) Find eigenvalues from diagonal of T
    c) Convert to real Schur form
    """
    print("\n" + "=" * 60)
    print("EXERCISE 10: Schur Decomposition")
    print("=" * 60)
    
    # Example using scipy
    A = np.array([
        [1, 2, 3],
        [0, 4, 5],
        [0, 0, 6]
    ], dtype=float)
    
    print("Upper triangular A:")
    print(A)
    
    # Schur decomposition
    T, Q = la.schur(A)
    
    print(f"\nSchur form T:\n{T.round(4)}")
    print(f"\nOrthogonal Q:\n{Q.round(4)}")
    print(f"\nQ @ T @ Q.T:\n{(Q @ T @ Q.T).round(4)}")
    
    # Eigenvalues are on diagonal
    print(f"\nEigenvalues from Schur: {np.diag(T)}")
    print(f"Eigenvalues (numpy): {np.linalg.eigvals(A)}")


def run_all_exercises():
    """Run all exercise solutions."""
    exercise_1_solution()
    exercise_2_solution()
    exercise_3_solution()
    exercise_4_solution()
    exercise_5_solution()
    exercise_6_solution()
    exercise_7_solution()
    exercise_8_solution()
    exercise_9_solution()
    exercise_10_schur_decomposition()


if __name__ == "__main__":
    run_all_exercises()
