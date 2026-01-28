"""
Matrix Decompositions - Examples
================================
Practical demonstrations of LU, QR, and Cholesky decompositions.
"""

import numpy as np
from numpy.linalg import qr, cholesky, inv, det, solve, lstsq
from scipy.linalg import lu, lu_factor, lu_solve


def example_lu_decomposition():
    """Demonstrate LU decomposition."""
    print("=" * 60)
    print("EXAMPLE 1: LU Decomposition")
    print("=" * 60)
    
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    
    print(f"A = \n{A}")
    
    # Manual LU (without pivoting)
    print("\n--- Manual LU (no pivoting) ---")
    n = 3
    L = np.eye(n)
    U = A.copy()
    
    for j in range(n - 1):
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, :] = U[i, :] - L[i, j] * U[j, :]
            print(f"Eliminating A[{i},{j}]: multiplier = {L[i,j]:.4f}")
    
    print(f"\nL = \n{np.round(L, 4)}")
    print(f"\nU = \n{np.round(U, 4)}")
    print(f"\nVerification: L @ U = \n{np.round(L @ U, 4)}")
    
    # SciPy LU with pivoting
    print("\n--- SciPy LU (with pivoting) ---")
    P, L_scipy, U_scipy = lu(A)
    print(f"P = \n{P}")
    print(f"L = \n{np.round(L_scipy, 4)}")
    print(f"U = \n{np.round(U_scipy, 4)}")
    print(f"P @ L @ U = \n{np.round(P @ L_scipy @ U_scipy, 4)}")


def example_solve_with_lu():
    """Solve linear system using LU decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Solving Ax = b with LU")
    print("=" * 60)
    
    A = np.array([[2, 1],
                  [4, 5]], dtype=float)
    b = np.array([3, 17], dtype=float)
    
    print(f"A = \n{A}")
    print(f"b = {b}")
    
    # Factor once
    lu_factored, piv = lu_factor(A)
    print(f"\nLU factored (stored in one matrix):\n{np.round(lu_factored, 4)}")
    
    # Solve
    x = lu_solve((lu_factored, piv), b)
    print(f"\nSolution x = {x}")
    print(f"Verification: A @ x = {A @ x}")
    
    # Multiple right-hand sides
    print("\n--- Multiple right-hand sides ---")
    B = np.array([[3, 1, 2],
                  [17, 5, 8]], dtype=float)
    
    X = lu_solve((lu_factored, piv), B)
    print(f"Solutions for 3 different b vectors:\n{X}")


def example_qr_gram_schmidt():
    """Demonstrate QR via Gram-Schmidt."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: QR via Gram-Schmidt")
    print("=" * 60)
    
    A = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], dtype=float)
    
    print(f"A = \n{A}")
    
    # Manual Gram-Schmidt
    print("\n--- Gram-Schmidt Process ---")
    n = A.shape[1]
    Q = np.zeros_like(A)
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j].copy()
        print(f"\nProcessing column {j+1}: a_{j+1} = {A[:, j]}")
        
        # Subtract projections
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
            print(f"  Projection onto q_{i+1}: {R[i,j]:.4f}")
        
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
        print(f"  Norm: {R[j,j]:.4f}")
        print(f"  q_{j+1} = {np.round(Q[:, j], 4)}")
    
    print(f"\nQ = \n{np.round(Q, 4)}")
    print(f"\nR = \n{np.round(R, 4)}")
    print(f"\nQ^T @ Q = \n{np.round(Q.T @ Q, 4)} (should be I)")
    print(f"\nQ @ R = \n{np.round(Q @ R, 4)}")


def example_qr_numpy():
    """Demonstrate QR using NumPy."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: QR Decomposition (NumPy)")
    print("=" * 60)
    
    A = np.array([[1, 2],
                  [3, 4],
                  [5, 6]], dtype=float)
    
    print(f"A (3×2) = \n{A}")
    
    # Reduced QR
    Q, R = qr(A, mode='reduced')
    print(f"\n--- Reduced QR ---")
    print(f"Q (3×2) = \n{np.round(Q, 4)}")
    print(f"R (2×2) = \n{np.round(R, 4)}")
    print(f"Q^T @ Q = \n{np.round(Q.T @ Q, 4)}")
    
    # Full QR
    Q_full, R_full = qr(A, mode='complete')
    print(f"\n--- Full QR ---")
    print(f"Q (3×3) = \n{np.round(Q_full, 4)}")
    print(f"R (3×2) = \n{np.round(R_full, 4)}")


def example_least_squares_qr():
    """Solve least squares using QR decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Least Squares via QR")
    print("=" * 60)
    
    # Overdetermined system: 4 points, fit line y = mx + c
    X = np.array([[1, 0],
                  [1, 1],
                  [1, 2],
                  [1, 3]], dtype=float)  # [1, x] columns
    y = np.array([1, 2.5, 3.5, 5], dtype=float)
    
    print("Fitting line y = c + mx to points:")
    print("x: [0, 1, 2, 3]")
    print(f"y: {y}")
    
    print(f"\nDesign matrix X = \n{X}")
    
    # QR approach
    Q, R = qr(X, mode='reduced')
    print(f"\nQ = \n{np.round(Q, 4)}")
    print(f"\nR = \n{np.round(R, 4)}")
    
    # Solve R @ coef = Q^T @ y
    coef = solve(R, Q.T @ y)
    print(f"\nCoefficients [c, m] = {np.round(coef, 4)}")
    print(f"Best fit line: y = {coef[0]:.4f} + {coef[1]:.4f}x")
    
    # Residual
    residual = np.linalg.norm(X @ coef - y)
    print(f"\nResidual norm: {residual:.4f}")


def example_cholesky():
    """Demonstrate Cholesky decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Cholesky Decomposition")
    print("=" * 60)
    
    A = np.array([[4, 2, 2],
                  [2, 10, 7],
                  [2, 7, 21]], dtype=float)
    
    print(f"A (symmetric positive definite) = \n{A}")
    
    # Manual Cholesky
    print("\n--- Manual Cholesky ---")
    n = 3
    L = np.zeros((n, n))
    
    for j in range(n):
        # Diagonal
        sum_sq = sum(L[j, k]**2 for k in range(j))
        L[j, j] = np.sqrt(A[j, j] - sum_sq)
        print(f"L[{j},{j}] = √({A[j,j]} - {sum_sq:.4f}) = {L[j,j]:.4f}")
        
        # Below diagonal
        for i in range(j + 1, n):
            sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
            L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    print(f"\nL = \n{np.round(L, 4)}")
    print(f"\nL @ L^T = \n{np.round(L @ L.T, 4)}")
    
    # NumPy Cholesky
    print("\n--- NumPy Cholesky ---")
    L_np = cholesky(A)
    print(f"L = \n{np.round(L_np, 4)}")


def example_solve_with_cholesky():
    """Solve Ax = b using Cholesky."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Solving with Cholesky")
    print("=" * 60)
    
    A = np.array([[4, 2],
                  [2, 5]], dtype=float)
    b = np.array([8, 11], dtype=float)
    
    print(f"A = \n{A}")
    print(f"b = {b}")
    
    # Cholesky factor
    L = cholesky(A)
    print(f"\nL = \n{L}")
    
    # Forward substitution: Ly = b
    y = np.zeros(2)
    y[0] = b[0] / L[0, 0]
    y[1] = (b[1] - L[1, 0] * y[0]) / L[1, 1]
    print(f"\nForward solve Ly = b: y = {y}")
    
    # Back substitution: L^T x = y
    LT = L.T
    x = np.zeros(2)
    x[1] = y[1] / LT[1, 1]
    x[0] = (y[0] - LT[0, 1] * x[1]) / LT[0, 0]
    print(f"Back solve L^T x = y: x = {x}")
    
    print(f"\nVerification: A @ x = {A @ x}")


def example_gaussian_sampling():
    """Sample from multivariate Gaussian using Cholesky."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Gaussian Sampling with Cholesky")
    print("=" * 60)
    
    # Define Gaussian N(mu, Sigma)
    mu = np.array([1, 2])
    Sigma = np.array([[2, 1],
                      [1, 3]])
    
    print(f"Mean: μ = {mu}")
    print(f"Covariance: Σ = \n{Sigma}")
    
    # Cholesky factor
    L = cholesky(Sigma)
    print(f"\nCholesky L = \n{np.round(L, 4)}")
    
    # Sample
    np.random.seed(42)
    n_samples = 5
    
    print("\n--- Sampling Process ---")
    print("For each sample: x = μ + L @ z, where z ~ N(0, I)")
    
    samples = []
    for i in range(n_samples):
        z = np.random.randn(2)
        x = mu + L @ z
        samples.append(x)
        print(f"Sample {i+1}: z = {np.round(z, 3)} → x = {np.round(x, 3)}")
    
    # Verify statistics
    samples = np.array(samples)
    print(f"\nSample mean: {np.round(samples.mean(axis=0), 3)}")
    print(f"True mean: {mu}")


def example_determinant_from_decomposition():
    """Compute determinant from decompositions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Determinant from Decompositions")
    print("=" * 60)
    
    A = np.array([[3, 1, 2],
                  [1, 4, 1],
                  [2, 1, 5]], dtype=float)
    
    print(f"A = \n{A}")
    print(f"\nDirect: det(A) = {det(A):.4f}")
    
    # From LU
    P, L, U = lu(A)
    det_lu = np.prod(np.diag(U)) * det(P)
    print(f"\nFrom LU: det = det(P) × ∏ u_ii = {det_lu:.4f}")
    
    # From Cholesky (A is SPD)
    try:
        L_chol = cholesky(A)
        det_chol = np.prod(np.diag(L_chol))**2
        print(f"From Cholesky: det = (∏ l_ii)² = {det_chol:.4f}")
    except:
        print("Cholesky failed (A not PD)")


def example_condition_number():
    """Effect of condition number on solutions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Condition Number and Stability")
    print("=" * 60)
    
    # Well-conditioned system
    A_good = np.array([[1, 0],
                       [0, 1]], dtype=float)
    
    # Ill-conditioned system
    A_bad = np.array([[1, 1],
                      [1, 1.0001]], dtype=float)
    
    for name, A in [("Well-conditioned", A_good), ("Ill-conditioned", A_bad)]:
        print(f"\n{name} system:")
        print(f"A = \n{A}")
        
        cond = np.linalg.cond(A)
        print(f"Condition number κ(A) = {cond:.2f}")
        
        b = np.array([2, 2], dtype=float)
        x = solve(A, b)
        print(f"Solution for b = {b}: x = {np.round(x, 4)}")
        
        # Perturb b slightly
        b_perturbed = b + np.array([0.0001, 0])
        x_perturbed = solve(A, b_perturbed)
        print(f"Solution for b + [0.0001, 0]: x = {np.round(x_perturbed, 4)}")
        print(f"Change in x: {np.linalg.norm(x_perturbed - x):.4f}")


def example_comparison():
    """Compare decomposition methods for same problem."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Comparing Methods for Ax = b")
    print("=" * 60)
    
    np.random.seed(42)
    n = 100
    
    # Create SPD matrix
    B = np.random.randn(n, n)
    A = B.T @ B + 0.1 * np.eye(n)  # Ensure PD
    b = np.random.randn(n)
    
    print(f"System size: {n} × {n}")
    
    import time
    
    # Direct solve
    t0 = time.time()
    x_direct = solve(A, b)
    t_direct = time.time() - t0
    
    # LU
    t0 = time.time()
    lu_piv = lu_factor(A)
    x_lu = lu_solve(lu_piv, b)
    t_lu = time.time() - t0
    
    # Cholesky
    t0 = time.time()
    L = cholesky(A)
    y = solve(L, b)
    x_chol = solve(L.T, y)
    t_chol = time.time() - t0
    
    # QR (for overdetermined, but works for square too)
    t0 = time.time()
    Q, R = qr(A)
    x_qr = solve(R, Q.T @ b)
    t_qr = time.time() - t0
    
    print("\nMethod      | Time (ms) | Residual norm")
    print("-" * 45)
    print(f"Direct      | {t_direct*1000:8.3f}  | {np.linalg.norm(A @ x_direct - b):.2e}")
    print(f"LU          | {t_lu*1000:8.3f}  | {np.linalg.norm(A @ x_lu - b):.2e}")
    print(f"Cholesky    | {t_chol*1000:8.3f}  | {np.linalg.norm(A @ x_chol - b):.2e}")
    print(f"QR          | {t_qr*1000:8.3f}  | {np.linalg.norm(A @ x_qr - b):.2e}")
    
    print("\nCholesky is typically fastest for SPD matrices!")


if __name__ == "__main__":
    example_lu_decomposition()
    example_solve_with_lu()
    example_qr_gram_schmidt()
    example_qr_numpy()
    example_least_squares_qr()
    example_cholesky()
    example_solve_with_cholesky()
    example_gaussian_sampling()
    example_determinant_from_decomposition()
    example_condition_number()
    example_comparison()
