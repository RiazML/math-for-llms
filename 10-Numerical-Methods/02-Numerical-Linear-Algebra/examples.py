"""
Numerical Linear Algebra - Examples
===================================
Practical implementations of core algorithms.
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
import time


def example_1_lu_decomposition():
    """
    Example 1: LU Decomposition
    ===========================
    Factor A = PLU for solving linear systems.
    """
    print("=" * 60)
    print("Example 1: LU Decomposition")
    print("=" * 60)
    
    # Create matrix
    A = np.array([
        [2, 1, 1],
        [4, 3, 3],
        [8, 7, 9]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    # Scipy LU with pivoting
    P, L, U = la.lu(A)
    
    print(f"\nPermutation matrix P:\n{P}")
    print(f"\nLower triangular L:\n{L}")
    print(f"\nUpper triangular U:\n{U}")
    
    # Verify
    reconstructed = P @ L @ U
    print(f"\nP @ L @ U:\n{reconstructed}")
    print(f"\nReconstruction error: {np.linalg.norm(A - reconstructed):.2e}")
    
    # Solve linear system
    b = np.array([4, 10, 24])
    
    # Method 1: Direct solve
    x_direct = np.linalg.solve(A, b)
    
    # Method 2: Using LU factors
    # Solve Ly = Pb (forward substitution)
    # Solve Ux = y (backward substitution)
    lu, piv = la.lu_factor(A)
    x_lu = la.lu_solve((lu, piv), b)
    
    print(f"\nSolving Ax = b where b = {b}")
    print(f"Direct solve: x = {x_direct}")
    print(f"LU solve:     x = {x_lu}")
    
    # Verify
    print(f"Residual ||Ax - b|| = {np.linalg.norm(A @ x_lu - b):.2e}")


def example_2_cholesky_decomposition():
    """
    Example 2: Cholesky Decomposition
    =================================
    For symmetric positive definite matrices.
    """
    print("\n" + "=" * 60)
    print("Example 2: Cholesky Decomposition")
    print("=" * 60)
    
    # Create symmetric positive definite matrix
    np.random.seed(42)
    X = np.random.randn(4, 4)
    A = X @ X.T + 0.1 * np.eye(4)  # Guaranteed SPD
    
    print("Symmetric positive definite matrix A:")
    print(A.round(3))
    
    # Check properties
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"\nEigenvalues: {eigenvalues.round(3)}")
    print(f"All positive? {np.all(eigenvalues > 0)}")
    
    # Cholesky decomposition
    L = np.linalg.cholesky(A)
    
    print(f"\nLower triangular L:\n{L.round(3)}")
    
    # Verify A = LL^T
    reconstructed = L @ L.T
    print(f"\nL @ L.T:\n{reconstructed.round(3)}")
    print(f"\nReconstruction error: {np.linalg.norm(A - reconstructed):.2e}")
    
    # Solving systems with Cholesky
    b = np.random.randn(4)
    
    # Solve Ax = b using Cholesky:
    # LL^T x = b
    # Step 1: Solve Ly = b
    y = la.solve_triangular(L, b, lower=True)
    # Step 2: Solve L^T x = y  
    x = la.solve_triangular(L.T, y, lower=False)
    
    print(f"\nSolving Ax = b:")
    print(f"Solution x = {x.round(4)}")
    print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")
    
    # Compare efficiency
    n = 500
    X = np.random.randn(n, n)
    A_large = X @ X.T + 0.1 * np.eye(n)
    b_large = np.random.randn(n)
    
    start = time.time()
    _ = np.linalg.solve(A_large, b_large)
    time_general = time.time() - start
    
    start = time.time()
    L_large = np.linalg.cholesky(A_large)
    y = la.solve_triangular(L_large, b_large, lower=True)
    _ = la.solve_triangular(L_large.T, y, lower=False)
    time_cholesky = time.time() - start
    
    print(f"\n{n}×{n} system solve time:")
    print(f"  General:  {time_general*1000:.2f} ms")
    print(f"  Cholesky: {time_cholesky*1000:.2f} ms")


def example_3_qr_decomposition():
    """
    Example 3: QR Decomposition
    ===========================
    For least squares and orthogonalization.
    """
    print("\n" + "=" * 60)
    print("Example 3: QR Decomposition")
    print("=" * 60)
    
    # Rectangular matrix (overdetermined system)
    A = np.array([
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4]
    ], dtype=float)
    
    print("Matrix A (4×2):")
    print(A)
    
    # Full QR
    Q, R = np.linalg.qr(A)
    
    print(f"\nOrthogonal Q (4×4):\n{Q.round(3)}")
    print(f"\nUpper triangular R (4×2):\n{R.round(3)}")
    
    # Verify orthogonality
    print(f"\nQ^T @ Q (should be I):\n{(Q.T @ Q).round(10)}")
    
    # Verify factorization
    print(f"\nReconstruction error: {np.linalg.norm(A - Q @ R):.2e}")
    
    # Economy/Thin QR
    Q_thin, R_thin = np.linalg.qr(A, mode='reduced')
    
    print(f"\nThin Q (4×2):\n{Q_thin.round(3)}")
    print(f"\nThin R (2×2):\n{R_thin.round(3)}")
    
    # Least squares using QR
    b = np.array([1, 2, 3, 4.1])
    
    # Solve min ||Ax - b||² using QR:
    # A = QR, so Ax ≈ b → Rx = Q^T b
    x_qr = la.solve_triangular(R_thin, Q_thin.T @ b)
    
    # Compare with direct least squares
    x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    print(f"\nLeast squares: min ||Ax - b||²")
    print(f"QR solution:     x = {x_qr.round(4)}")
    print(f"lstsq solution:  x = {x_lstsq.round(4)}")
    print(f"Residual ||Ax - b||: {np.linalg.norm(A @ x_qr - b):.4f}")


def example_4_gram_schmidt():
    """
    Example 4: Gram-Schmidt Orthogonalization
    =========================================
    Classical vs Modified Gram-Schmidt.
    """
    print("\n" + "=" * 60)
    print("Example 4: Gram-Schmidt Orthogonalization")
    print("=" * 60)
    
    def classical_gram_schmidt(A):
        """Classical Gram-Schmidt (less stable)."""
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            v = A[:, j].copy()
            for i in range(j):
                R[i, j] = Q[:, i] @ A[:, j]  # Project onto q_i
                v = v - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            Q[:, j] = v / R[j, j]
        
        return Q, R
    
    def modified_gram_schmidt(A):
        """Modified Gram-Schmidt (more stable)."""
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        V = A.copy()
        
        for j in range(n):
            for i in range(j):
                R[i, j] = Q[:, i] @ V[:, j]  # Use current v_j
                V[:, j] = V[:, j] - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(V[:, j])
            Q[:, j] = V[:, j] / R[j, j]
        
        return Q, R
    
    # Well-conditioned matrix
    A_good = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=float)
    
    print("Well-conditioned matrix:")
    Q_c, R_c = classical_gram_schmidt(A_good)
    Q_m, R_m = modified_gram_schmidt(A_good)
    
    ortho_error_c = np.linalg.norm(Q_c.T @ Q_c - np.eye(3))
    ortho_error_m = np.linalg.norm(Q_m.T @ Q_m - np.eye(3))
    
    print(f"  Classical GS orthogonality error: {ortho_error_c:.2e}")
    print(f"  Modified GS orthogonality error:  {ortho_error_m:.2e}")
    
    # Ill-conditioned matrix (nearly dependent columns)
    eps = 1e-8
    A_bad = np.array([
        [1, 1, 1],
        [eps, 0, 0],
        [0, eps, 0],
        [0, 0, eps]
    ], dtype=float)
    
    print(f"\nIll-conditioned matrix (eps={eps}):")
    print(f"  Condition number: {np.linalg.cond(A_bad):.2e}")
    
    Q_c, _ = classical_gram_schmidt(A_bad)
    Q_m, _ = modified_gram_schmidt(A_bad)
    
    ortho_error_c = np.linalg.norm(Q_c.T @ Q_c - np.eye(3))
    ortho_error_m = np.linalg.norm(Q_m.T @ Q_m - np.eye(3))
    
    print(f"  Classical GS orthogonality error: {ortho_error_c:.2e}")
    print(f"  Modified GS orthogonality error:  {ortho_error_m:.2e}")
    print("  → Modified GS is much more stable!")


def example_5_svd():
    """
    Example 5: Singular Value Decomposition
    =======================================
    The most important decomposition in ML.
    """
    print("\n" + "=" * 60)
    print("Example 5: Singular Value Decomposition")
    print("=" * 60)
    
    # Create matrix
    A = np.array([
        [1, 0, 0, 0, 2],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0]
    ], dtype=float)
    
    print("Matrix A (4×5):")
    print(A)
    
    # Full SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    print(f"\nU (4×4):\n{U.round(3)}")
    print(f"\nSingular values: {s.round(3)}")
    print(f"\nV^T (5×5):\n{Vt.round(3)}")
    
    # Construct Sigma matrix
    Sigma = np.zeros((4, 5))
    np.fill_diagonal(Sigma, s)
    
    # Verify
    reconstructed = U @ Sigma @ Vt
    print(f"\nReconstruction error: {np.linalg.norm(A - reconstructed):.2e}")
    
    # Economy SVD
    U_thin, s_thin, Vt_thin = np.linalg.svd(A, full_matrices=False)
    print(f"\nEconomy SVD shapes:")
    print(f"  U: {U_thin.shape}, s: {s_thin.shape}, V^T: {Vt_thin.shape}")
    
    # Matrix properties from SVD
    print(f"\nMatrix properties from SVD:")
    print(f"  Rank: {np.sum(s > 1e-10)}")
    print(f"  Spectral norm ||A||₂: {s[0]:.3f}")
    print(f"  Frobenius norm ||A||_F: {np.sqrt(np.sum(s**2)):.3f}")
    print(f"  Frobenius (direct): {np.linalg.norm(A, 'fro'):.3f}")


def example_6_truncated_svd():
    """
    Example 6: Low-Rank Approximation with SVD
    ==========================================
    Best rank-k approximation (Eckart-Young).
    """
    print("\n" + "=" * 60)
    print("Example 6: Low-Rank Approximation")
    print("=" * 60)
    
    # Create matrix with structure
    np.random.seed(42)
    m, n = 100, 50
    
    # Low-rank + noise
    rank_true = 5
    A_low_rank = np.random.randn(m, rank_true) @ np.random.randn(rank_true, n)
    noise = 0.1 * np.random.randn(m, n)
    A = A_low_rank + noise
    
    print(f"Original matrix: {m}×{n}")
    print(f"True rank: {rank_true}, with added noise")
    
    # Full SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    print(f"\nSingular values (first 10):")
    print(f"  {s[:10].round(2)}")
    
    # Truncated approximations
    print(f"\nApproximation errors (Frobenius norm):")
    print(f"{'Rank k':<10} {'Error':<15} {'Relative Error':<15} {'Storage Ratio':<15}")
    print("-" * 55)
    
    original_norm = np.linalg.norm(A, 'fro')
    original_storage = m * n
    
    for k in [1, 2, 5, 10, 20]:
        # Truncated SVD
        A_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        error = np.linalg.norm(A - A_k, 'fro')
        rel_error = error / original_norm
        storage = k * (m + n + 1)  # U_k, V_k, s_k
        storage_ratio = storage / original_storage
        
        print(f"{k:<10} {error:<15.4f} {rel_error:<15.4f} {storage_ratio:<15.2%}")
    
    # Optimal k choice: look for "elbow" in singular values
    print(f"\nEnergy captured by rank-k:")
    total_energy = np.sum(s**2)
    cumulative_energy = np.cumsum(s**2) / total_energy
    
    for k in [1, 2, 5, 10, 20]:
        print(f"  Rank {k}: {cumulative_energy[k-1]:.1%}")


def example_7_pseudoinverse():
    """
    Example 7: Moore-Penrose Pseudoinverse
    ======================================
    Using SVD to compute A⁺.
    """
    print("\n" + "=" * 60)
    print("Example 7: Pseudoinverse via SVD")
    print("=" * 60)
    
    def pseudoinverse_svd(A, tol=1e-10):
        """Compute pseudoinverse using SVD."""
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        
        # Invert non-zero singular values
        s_inv = np.where(s > tol, 1/s, 0)
        
        # A⁺ = V @ Σ⁺ @ U^T
        return Vt.T @ np.diag(s_inv) @ U.T
    
    # Overdetermined system (more equations than unknowns)
    A = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ], dtype=float)
    
    b = np.array([1, 2, 3, 4.5])
    
    print("Overdetermined system (4×2):")
    print(f"A:\n{A}")
    
    A_pinv_custom = pseudoinverse_svd(A)
    A_pinv_numpy = np.linalg.pinv(A)
    
    print(f"\nPseudoinverse A⁺ (custom):\n{A_pinv_custom.round(4)}")
    print(f"\nPseudoinverse A⁺ (NumPy):\n{A_pinv_numpy.round(4)}")
    
    # Solve least squares
    x_pinv = A_pinv_custom @ b
    x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    print(f"\nLeast squares solution:")
    print(f"  Pseudoinverse: x = {x_pinv.round(4)}")
    print(f"  lstsq:         x = {x_lstsq.round(4)}")
    
    # Underdetermined system (more unknowns than equations)
    A2 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ], dtype=float)
    
    b2 = np.array([1, 2])
    
    print(f"\nUnderdetermined system (2×4):")
    
    A2_pinv = np.linalg.pinv(A2)
    x_min_norm = A2_pinv @ b2
    
    print(f"Minimum norm solution: x = {x_min_norm.round(4)}")
    print(f"||x|| = {np.linalg.norm(x_min_norm):.4f}")
    print(f"Ax - b = {(A2 @ x_min_norm - b2).round(10)}")


def example_8_power_iteration():
    """
    Example 8: Power Iteration for Eigenvalues
    ==========================================
    Find dominant eigenvalue and eigenvector.
    """
    print("\n" + "=" * 60)
    print("Example 8: Power Iteration")
    print("=" * 60)
    
    def power_iteration(A, max_iter=100, tol=1e-10):
        """Find largest eigenvalue and eigenvector."""
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        eigenvalue_history = []
        
        for i in range(max_iter):
            w = A @ v
            v_new = w / np.linalg.norm(w)
            eigenvalue = v_new @ A @ v_new
            
            eigenvalue_history.append(eigenvalue)
            
            if np.linalg.norm(v_new - v) < tol:
                print(f"Converged in {i+1} iterations")
                break
            
            v = v_new
        
        return eigenvalue, v, eigenvalue_history
    
    # Symmetric matrix
    A = np.array([
        [4, 1, 1],
        [1, 3, 1],
        [1, 1, 2]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    # True eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    print(f"\nTrue eigenvalues: {eigenvalues}")
    print(f"Dominant eigenvalue: {eigenvalues[-1]:.6f}")
    print(f"Dominant eigenvector: {eigenvectors[:, -1]}")
    
    # Power iteration
    print("\nPower iteration:")
    lambda1, v1, history = power_iteration(A)
    
    print(f"Found eigenvalue: {lambda1:.6f}")
    print(f"Found eigenvector: {v1}")
    
    # Convergence rate
    ratio = abs(eigenvalues[-2] / eigenvalues[-1])
    print(f"\nConvergence rate |λ₂/λ₁| = {ratio:.4f}")
    
    # Show convergence
    print(f"\nConvergence history (first 10):")
    for i, ev in enumerate(history[:10]):
        print(f"  Iteration {i+1}: λ = {ev:.8f}")


def example_9_conjugate_gradient():
    """
    Example 9: Conjugate Gradient Method
    ====================================
    Iterative solver for symmetric positive definite systems.
    """
    print("\n" + "=" * 60)
    print("Example 9: Conjugate Gradient")
    print("=" * 60)
    
    def conjugate_gradient(A, b, x0=None, max_iter=None, tol=1e-10):
        """Conjugate gradient for Ax = b, A symmetric positive definite."""
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        if max_iter is None:
            max_iter = n
        
        r = b - A @ x
        p = r.copy()
        rsold = r @ r
        
        residual_history = [np.sqrt(rsold)]
        
        for i in range(max_iter):
            Ap = A @ p
            alpha = rsold / (p @ Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r @ r
            
            residual_history.append(np.sqrt(rsnew))
            
            if np.sqrt(rsnew) < tol:
                print(f"Converged in {i+1} iterations")
                break
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        return x, residual_history
    
    # Create symmetric positive definite matrix
    np.random.seed(42)
    n = 10
    X = np.random.randn(n, n)
    A = X @ X.T + 0.1 * np.eye(n)
    
    b = np.random.randn(n)
    
    print(f"System size: {n}×{n}")
    print(f"Condition number: {np.linalg.cond(A):.2f}")
    
    # Solve with CG
    x_cg, residuals = conjugate_gradient(A, b)
    
    # Compare with direct solve
    x_direct = np.linalg.solve(A, b)
    
    print(f"\nSolution error ||x_cg - x_direct||: {np.linalg.norm(x_cg - x_direct):.2e}")
    print(f"Final residual ||Ax - b||: {np.linalg.norm(A @ x_cg - b):.2e}")
    
    # Residual convergence
    print(f"\nResidual convergence:")
    for i in [0, 1, 2, 5, len(residuals)-1]:
        if i < len(residuals):
            print(f"  Iteration {i}: ||r|| = {residuals[i]:.2e}")


def example_10_sparse_matrices():
    """
    Example 10: Sparse Matrix Operations
    ====================================
    Efficient storage and computation.
    """
    print("\n" + "=" * 60)
    print("Example 10: Sparse Matrices")
    print("=" * 60)
    
    # Create sparse matrix
    n = 1000
    density = 0.01  # 1% non-zero
    
    # Random sparse matrix
    np.random.seed(42)
    nnz = int(n * n * density)
    rows = np.random.randint(0, n, nnz)
    cols = np.random.randint(0, n, nnz)
    data = np.random.randn(nnz)
    
    # CSR format
    A_sparse = csr_matrix((data, (rows, cols)), shape=(n, n))
    A_dense = A_sparse.toarray()
    
    print(f"Matrix size: {n}×{n}")
    print(f"Density: {density:.1%}")
    print(f"Non-zeros: {A_sparse.nnz}")
    
    # Memory comparison
    dense_memory = A_dense.nbytes
    sparse_memory = (A_sparse.data.nbytes + A_sparse.indices.nbytes + 
                     A_sparse.indptr.nbytes)
    
    print(f"\nMemory usage:")
    print(f"  Dense:  {dense_memory / 1e6:.2f} MB")
    print(f"  Sparse: {sparse_memory / 1e6:.2f} MB")
    print(f"  Ratio:  {sparse_memory / dense_memory:.1%}")
    
    # Matrix-vector multiplication speed
    x = np.random.randn(n)
    
    start = time.time()
    for _ in range(100):
        y_dense = A_dense @ x
    time_dense = (time.time() - start) / 100
    
    start = time.time()
    for _ in range(100):
        y_sparse = A_sparse @ x
    time_sparse = (time.time() - start) / 100
    
    print(f"\nMatrix-vector multiply (average of 100):")
    print(f"  Dense:  {time_dense*1000:.3f} ms")
    print(f"  Sparse: {time_sparse*1000:.3f} ms")
    print(f"  Speedup: {time_dense/time_sparse:.1f}x")
    
    # Different sparse formats
    print(f"\nSparse formats:")
    
    A_csc = csc_matrix(A_sparse)
    A_coo = A_sparse.tocoo()
    
    print(f"  CSR: Efficient row slicing, row-wise operations")
    print(f"  CSC: Efficient column slicing, column-wise operations")
    print(f"  COO: Efficient construction, conversion")


def example_11_condition_number():
    """
    Example 11: Condition Number Analysis
    =====================================
    Understanding numerical sensitivity.
    """
    print("\n" + "=" * 60)
    print("Example 11: Condition Number")
    print("=" * 60)
    
    def create_hilbert(n):
        """Create Hilbert matrix (very ill-conditioned)."""
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                H[i, j] = 1 / (i + j + 1)
        return H
    
    # Hilbert matrices
    print("Hilbert matrices (notoriously ill-conditioned):")
    print(f"{'Size':<10} {'Condition Number':<20} {'log₁₀(κ)':<15}")
    print("-" * 45)
    
    for n in [3, 5, 7, 10, 12]:
        H = create_hilbert(n)
        kappa = np.linalg.cond(H)
        print(f"{n:<10} {kappa:<20.2e} {np.log10(kappa):<15.1f}")
    
    # Effect on solving linear systems
    print("\nEffect on linear system solution (n=10):")
    
    n = 10
    H = create_hilbert(n)
    x_true = np.ones(n)
    b = H @ x_true
    
    # Solve
    x_computed = np.linalg.solve(H, b)
    
    # Error
    forward_error = np.linalg.norm(x_computed - x_true) / np.linalg.norm(x_true)
    residual = np.linalg.norm(H @ x_computed - b) / np.linalg.norm(b)
    
    print(f"  True solution: all ones")
    print(f"  Condition number: {np.linalg.cond(H):.2e}")
    print(f"  Relative forward error: {forward_error:.2e}")
    print(f"  Relative residual: {residual:.2e}")
    print(f"  Expected digit loss: ~{np.log10(np.linalg.cond(H)):.0f}")
    
    # Regularization helps
    print("\nWith Tikhonov regularization (λ=1e-10):")
    lambda_reg = 1e-10
    H_reg = H.T @ H + lambda_reg * np.eye(n)
    b_reg = H.T @ b
    x_reg = np.linalg.solve(H_reg, b_reg)
    
    print(f"  New condition number: {np.linalg.cond(H_reg):.2e}")
    print(f"  Relative error: {np.linalg.norm(x_reg - x_true) / np.linalg.norm(x_true):.2e}")


def example_12_pca_via_svd():
    """
    Example 12: PCA using SVD
    =========================
    Principal Component Analysis implementation.
    """
    print("\n" + "=" * 60)
    print("Example 12: PCA via SVD")
    print("=" * 60)
    
    # Generate data with structure
    np.random.seed(42)
    n_samples = 200
    
    # 2D data embedded in 5D
    t = np.linspace(0, 4*np.pi, n_samples)
    X_latent = np.column_stack([np.sin(t), np.cos(t)])
    
    # Random projection to 5D + noise
    W = np.random.randn(2, 5)
    X = X_latent @ W + 0.1 * np.random.randn(n_samples, 5)
    
    print(f"Data shape: {X.shape}")
    
    # Center data
    X_centered = X - X.mean(axis=0)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Singular values relate to variance explained
    variance_explained = s**2 / np.sum(s**2)
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"\nVariance explained by each component:")
    for i in range(5):
        print(f"  PC{i+1}: {variance_explained[i]:.1%} (cumulative: {cumulative_variance[i]:.1%})")
    
    # Principal components are rows of V^T
    print(f"\nFirst 2 principal components capture {cumulative_variance[1]:.1%} of variance")
    
    # Project to 2D
    X_pca = X_centered @ Vt[:2].T
    
    print(f"\nReduced data shape: {X_pca.shape}")
    
    # Reconstruction error
    X_reconstructed = X_pca @ Vt[:2] + X.mean(axis=0)
    recon_error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
    
    print(f"Reconstruction error (k=2): {recon_error:.4f}")


def run_all_examples():
    """Run all examples."""
    example_1_lu_decomposition()
    example_2_cholesky_decomposition()
    example_3_qr_decomposition()
    example_4_gram_schmidt()
    example_5_svd()
    example_6_truncated_svd()
    example_7_pseudoinverse()
    example_8_power_iteration()
    example_9_conjugate_gradient()
    example_10_sparse_matrices()
    example_11_condition_number()
    example_12_pca_via_svd()


if __name__ == "__main__":
    run_all_examples()
