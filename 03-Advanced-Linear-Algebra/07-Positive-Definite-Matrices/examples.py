"""
Positive Definite Matrices - Examples
=====================================
Practical demonstrations of positive definite matrix concepts.
"""

import numpy as np
from numpy.linalg import eigvalsh, cholesky, inv, det


def example_quadratic_form():
    """Demonstrate positive definiteness via quadratic form."""
    print("=" * 60)
    print("EXAMPLE 1: Quadratic Form Test")
    print("=" * 60)
    
    # Positive definite matrix
    A = np.array([[2, 1],
                  [1, 2]])
    
    print(f"A = \n{A}")
    print("\nTesting x^T A x for various x:")
    
    test_vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([1, -1]),
        np.array([2, 3])
    ]
    
    for x in test_vectors:
        quadratic = x.T @ A @ x
        print(f"  x = {x}: x^T A x = {quadratic}")
    
    print("\nAll values > 0 → A is positive definite!")
    
    # Compare with indefinite matrix
    print("\n" + "-" * 40)
    B = np.array([[1, 2],
                  [2, 1]])
    
    print(f"\nB = \n{B}")
    print("\nTesting x^T B x:")
    
    for x in test_vectors:
        quadratic = x.T @ B @ x
        sign = ">" if quadratic > 0 else ("<" if quadratic < 0 else "=")
        print(f"  x = {x}: x^T B x = {quadratic} {sign} 0")
    
    print("\nSome positive, some negative → B is indefinite!")


def example_eigenvalue_test():
    """Demonstrate eigenvalue test for positive definiteness."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Eigenvalue Test")
    print("=" * 60)
    
    matrices = {
        'A (PD)': np.array([[3, 1], [1, 2]]),
        'B (PSD)': np.array([[1, 1], [1, 1]]),
        'C (indefinite)': np.array([[1, 2], [2, 1]])
    }
    
    for name, M in matrices.items():
        eigenvalues = eigvalsh(M)  # Use eigvalsh for symmetric matrices
        print(f"\n{name}:")
        print(f"  Matrix:\n  {M}")
        print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
        
        if all(eigenvalues > 1e-10):
            print(f"  Classification: Positive Definite (all λ > 0)")
        elif all(eigenvalues >= -1e-10):
            print(f"  Classification: Positive Semi-Definite (all λ ≥ 0)")
        else:
            print(f"  Classification: Indefinite (mixed signs)")


def example_cholesky():
    """Demonstrate Cholesky decomposition."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Cholesky Decomposition")
    print("=" * 60)
    
    A = np.array([[4, 2, 2],
                  [2, 5, 1],
                  [2, 1, 6]], dtype=float)
    
    print(f"A = \n{A}")
    print(f"\nEigenvalues: {np.round(eigvalsh(A), 4)}")
    print("All positive → A is PD → Cholesky exists")
    
    # Manual Cholesky
    print("\n--- Manual Cholesky ---")
    n = 3
    L = np.zeros((n, n))
    
    for j in range(n):
        # Diagonal element
        sum_sq = sum(L[j, k]**2 for k in range(j))
        L[j, j] = np.sqrt(A[j, j] - sum_sq)
        print(f"L[{j},{j}] = √(A[{j},{j}] - Σ L[{j},k]²) = √({A[j,j]} - {sum_sq:.4f}) = {L[j,j]:.4f}")
        
        # Below diagonal
        for i in range(j + 1, n):
            sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
            L[i, j] = (A[i, j] - sum_prod) / L[j, j]
    
    print(f"\nL (lower triangular) = \n{np.round(L, 4)}")
    
    # Verify
    print(f"\nVerification: L @ L^T = \n{np.round(L @ L.T, 4)}")
    print(f"Matches A? {np.allclose(L @ L.T, A)}")
    
    # NumPy Cholesky
    print("\n--- NumPy Cholesky ---")
    L_np = cholesky(A)
    print(f"L = \n{np.round(L_np, 4)}")


def example_solve_with_cholesky():
    """Demonstrate solving linear systems with Cholesky."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Solving Ax = b with Cholesky")
    print("=" * 60)
    
    A = np.array([[4, 2],
                  [2, 5]], dtype=float)
    b = np.array([8, 11], dtype=float)
    
    print(f"A = \n{A}")
    print(f"b = {b}")
    
    # Step 1: Cholesky decomposition
    L = cholesky(A)
    print(f"\nStep 1: A = L L^T")
    print(f"L = \n{np.round(L, 4)}")
    
    # Step 2: Solve Ly = b (forward substitution)
    y = np.zeros(2)
    y[0] = b[0] / L[0, 0]
    y[1] = (b[1] - L[1, 0] * y[0]) / L[1, 1]
    print(f"\nStep 2: Solve Ly = b (forward)")
    print(f"y = {np.round(y, 4)}")
    
    # Step 3: Solve L^T x = y (back substitution)
    LT = L.T
    x = np.zeros(2)
    x[1] = y[1] / LT[1, 1]
    x[0] = (y[0] - LT[0, 1] * x[1]) / LT[0, 0]
    print(f"\nStep 3: Solve L^T x = y (backward)")
    print(f"x = {np.round(x, 4)}")
    
    # Verify
    print(f"\nVerification: A @ x = {np.round(A @ x, 4)} = b ✓")


def example_covariance_matrix():
    """Demonstrate covariance matrices are PSD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Covariance Matrices are PSD")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate data
    n = 1000
    X = np.random.randn(n, 3)
    # Add some correlation
    X[:, 1] = X[:, 0] * 0.8 + X[:, 1] * 0.2
    X[:, 2] = X[:, 0] * 0.5 + X[:, 2] * 0.5
    
    # Compute covariance
    X_centered = X - X.mean(axis=0)
    Sigma = X_centered.T @ X_centered / (n - 1)
    
    print(f"Covariance matrix Σ:\n{np.round(Sigma, 4)}")
    
    eigenvalues = eigvalsh(Sigma)
    print(f"\nEigenvalues: {np.round(eigenvalues, 4)}")
    print(f"All ≥ 0? {all(eigenvalues >= -1e-10)}")
    print("→ Σ is positive semi-definite!")
    
    # Why is it PSD?
    print("\n--- Why covariance is PSD ---")
    a = np.array([1, 2, 1])
    var_linear_combo = a.T @ Sigma @ a
    print(f"For any a = {a}:")
    print(f"a^T Σ a = Var(a^T X) = {var_linear_combo:.4f}")
    print("Variance is always ≥ 0!")


def example_kernel_matrix():
    """Demonstrate kernel matrices are PSD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Kernel Matrices are PSD")
    print("=" * 60)
    
    # Data points
    X = np.array([[0], [1], [2], [3]], dtype=float)
    print(f"Data points: {X.flatten()}")
    
    # RBF kernel
    gamma = 0.5
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.exp(-gamma * (X[i] - X[j])**2)
    
    print(f"\nRBF Kernel matrix (γ = {gamma}):")
    print(np.round(K, 4))
    
    eigenvalues = eigvalsh(K)
    print(f"\nEigenvalues: {np.round(eigenvalues, 4)}")
    print(f"All ≥ 0? {all(eigenvalues >= -1e-10)}")
    print("→ Kernel matrix is positive semi-definite!")
    
    # Linear kernel
    K_linear = X @ X.T
    print(f"\nLinear kernel K = X @ X^T:")
    print(K_linear)
    print(f"Eigenvalues: {np.round(eigvalsh(K_linear), 4)}")


def example_convexity():
    """Demonstrate connection to convexity."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Positive Definite Hessian → Convexity")
    print("=" * 60)
    
    print("Quadratic function: f(x) = 0.5 x^T A x - b^T x")
    print("Gradient: ∇f = Ax - b")
    print("Hessian: ∇²f = A")
    
    # Convex case
    A_convex = np.array([[2, 1], [1, 2]])
    b = np.array([1, 1])
    
    print(f"\nCase 1: A = \n{A_convex}")
    print(f"Eigenvalues: {eigvalsh(A_convex)}")
    print("All positive → f is strictly convex")
    print("Unique minimum at x* = A⁻¹b = ", np.round(inv(A_convex) @ b, 4))
    
    # Non-convex case
    A_nonconvex = np.array([[1, 2], [2, 1]])
    
    print(f"\nCase 2: A = \n{A_nonconvex}")
    print(f"Eigenvalues: {eigvalsh(A_nonconvex)}")
    print("Mixed signs → f has a saddle point (not convex)")


def example_regularization():
    """Demonstrate regularization making matrices PD."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Regularization Makes PD")
    print("=" * 60)
    
    # Singular matrix (not PD)
    X = np.array([[1, 2],
                  [2, 4],
                  [3, 6]])  # Columns are linearly dependent
    
    XTX = X.T @ X
    print(f"X = \n{X}")
    print(f"\nX^T X = \n{XTX}")
    print(f"Eigenvalues: {np.round(eigvalsh(XTX), 4)}")
    print("One eigenvalue ≈ 0 → Singular!")
    
    # Add regularization
    print("\n--- Adding regularization λI ---")
    
    for lam in [0.1, 1.0, 10.0]:
        XTX_reg = XTX + lam * np.eye(2)
        eigenvalues = eigvalsh(XTX_reg)
        print(f"\nλ = {lam}:")
        print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
        print(f"  All > 0? {all(eigenvalues > 0)} → Now invertible!")


def example_nearest_pd():
    """Demonstrate finding nearest PD matrix."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Finding Nearest PD Matrix")
    print("=" * 60)
    
    # Matrix that's not quite PD (due to numerical issues)
    A = np.array([[1.0, 0.9, 0.8],
                  [0.9, 1.0, 0.9],
                  [0.8, 0.9, 1.0]])
    
    # Perturb to make slightly non-PD
    A[0, 2] = 0.85
    A[2, 0] = 0.85
    
    print(f"A = \n{A}")
    eigenvalues = eigvalsh(A)
    print(f"Eigenvalues: {np.round(eigenvalues, 6)}")
    
    if any(eigenvalues < 0):
        print("Some eigenvalues negative → Not PD!")
        
        # Method 1: Set negative eigenvalues to small positive
        eigenvalues_fixed = np.maximum(eigenvalues, 1e-6)
        eigenvectors = np.linalg.eigh(A)[1]
        A_pd = eigenvectors @ np.diag(eigenvalues_fixed) @ eigenvectors.T
        
        print(f"\nNearest PD (eigenvalue projection):")
        print(f"Fixed eigenvalues: {np.round(eigenvalues_fixed, 6)}")
        print(f"A_pd = \n{np.round(A_pd, 4)}")
        
        # Verify
        try:
            cholesky(A_pd)
            print("Cholesky successful → A_pd is PD!")
        except:
            print("Still not PD")
    else:
        print("A is already PD!")


def example_mahalanobis():
    """Demonstrate Mahalanobis distance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Mahalanobis Distance")
    print("=" * 60)
    
    # Covariance matrix (must be PD for Mahalanobis)
    Sigma = np.array([[2, 1],
                      [1, 2]])
    
    mu = np.array([0, 0])
    
    print(f"Σ = \n{Sigma}")
    print(f"μ = {mu}")
    
    Sigma_inv = inv(Sigma)
    print(f"\nΣ⁻¹ = \n{np.round(Sigma_inv, 4)}")
    
    def mahalanobis(x, mu, Sigma_inv):
        diff = x - mu
        return np.sqrt(diff.T @ Sigma_inv @ diff)
    
    def euclidean(x, mu):
        diff = x - mu
        return np.sqrt(diff.T @ diff)
    
    # Compare distances
    points = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([2, 0])
    ]
    
    print("\nCompare Euclidean vs Mahalanobis distances:")
    for x in points:
        d_eucl = euclidean(x, mu)
        d_maha = mahalanobis(x, mu, Sigma_inv)
        print(f"  x = {x}: Euclidean = {d_eucl:.3f}, Mahalanobis = {d_maha:.3f}")
    
    print("\nMahalanobis accounts for correlation structure!")


def example_pd_tests():
    """Demonstrate multiple tests for positive definiteness."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Multiple PD Tests")
    print("=" * 60)
    
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    
    print(f"A = \n{A}")
    
    # Test 1: Symmetry
    print("\nTest 1: Symmetry")
    print(f"  A = A^T? {np.allclose(A, A.T)}")
    
    # Test 2: Eigenvalues
    print("\nTest 2: Eigenvalues")
    eigenvalues = eigvalsh(A)
    print(f"  Eigenvalues: {np.round(eigenvalues, 4)}")
    print(f"  All > 0? {all(eigenvalues > 0)}")
    
    # Test 3: Leading principal minors
    print("\nTest 3: Leading Principal Minors (Sylvester)")
    print(f"  a₁₁ = {A[0,0]} > 0? {A[0,0] > 0}")
    minor2 = det(A[:2, :2])
    print(f"  det(A[:2,:2]) = {minor2:.4f} > 0? {minor2 > 0}")
    minor3 = det(A)
    print(f"  det(A) = {minor3:.4f} > 0? {minor3 > 0}")
    
    # Test 4: Cholesky
    print("\nTest 4: Cholesky Decomposition")
    try:
        L = cholesky(A)
        print(f"  Cholesky exists!")
        print(f"  L = \n{np.round(L, 4)}")
    except:
        print("  Cholesky failed → Not PD")
    
    print("\n✓ A is positive definite!")


if __name__ == "__main__":
    example_quadratic_form()
    example_eigenvalue_test()
    example_cholesky()
    example_solve_with_cholesky()
    example_covariance_matrix()
    example_kernel_matrix()
    example_convexity()
    example_regularization()
    example_nearest_pd()
    example_mahalanobis()
    example_pd_tests()
