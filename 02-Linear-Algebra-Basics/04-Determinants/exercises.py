"""
Determinants - Exercises
========================
Practice problems for determinant computation and applications.
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# EXERCISE 1: 2×2 Determinant (Easy)
# =============================================================================

def det_2x2(A: np.ndarray) -> float:
    """
    Compute determinant of a 2×2 matrix WITHOUT using np.linalg.det.
    
    Formula: det([[a, b], [c, d]]) = ad - bc
    
    Parameters
    ----------
    A : np.ndarray
        2×2 matrix
    
    Returns
    -------
    float
        The determinant
    
    Example
    -------
    >>> det_2x2(np.array([[3, 2], [1, 4]]))
    10.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: 3×3 Determinant (Easy)
# =============================================================================

def det_3x3(A: np.ndarray) -> float:
    """
    Compute determinant of a 3×3 matrix using cofactor expansion.
    
    Expand along first row:
    det(A) = a11*M11 - a12*M12 + a13*M13
    where Mij is the minor (determinant of 2×2 submatrix)
    
    Parameters
    ----------
    A : np.ndarray
        3×3 matrix
    
    Returns
    -------
    float
        The determinant
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Cofactor Matrix (Medium)
# =============================================================================

def cofactor_matrix(A: np.ndarray) -> np.ndarray:
    """
    Compute the cofactor matrix of a square matrix.
    
    C_ij = (-1)^(i+j) * M_ij
    where M_ij is the minor (determinant of matrix with row i, col j removed)
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix (n×n)
    
    Returns
    -------
    np.ndarray
        Cofactor matrix (n×n)
    
    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> cofactor_matrix(A)
    array([[ 4, -3],
           [-2,  1]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Adjugate Matrix (Medium)
# =============================================================================

def adjugate(A: np.ndarray) -> np.ndarray:
    """
    Compute the adjugate (adjoint) matrix.
    
    adj(A) = C^T where C is the cofactor matrix
    
    The adjugate satisfies: A @ adj(A) = det(A) * I
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    
    Returns
    -------
    np.ndarray
        Adjugate matrix
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: Inverse via Adjugate (Medium)
# =============================================================================

def inverse_via_adjugate(A: np.ndarray) -> np.ndarray:
    """
    Compute matrix inverse using the formula: A^(-1) = adj(A) / det(A)
    
    Parameters
    ----------
    A : np.ndarray
        Invertible square matrix
    
    Returns
    -------
    np.ndarray
        Inverse matrix
    
    Raises
    ------
    ValueError
        If matrix is singular
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Check Invertibility (Easy)
# =============================================================================

def is_invertible(A: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a matrix is invertible by examining its determinant.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    tol : float
        Tolerance for considering determinant as zero
    
    Returns
    -------
    bool
        True if invertible (det ≠ 0)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Parallelogram Area (Easy)
# =============================================================================

def parallelogram_area(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the area of a parallelogram formed by two 2D vectors.
    
    Area = |det([u | v])| = |u₁v₂ - u₂v₁|
    
    Parameters
    ----------
    u, v : np.ndarray
        2D vectors
    
    Returns
    -------
    float
        Area of parallelogram
    
    Example
    -------
    >>> parallelogram_area(np.array([2, 0]), np.array([0, 3]))
    6.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Parallelepiped Volume (Easy)
# =============================================================================

def parallelepiped_volume(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the volume of a parallelepiped formed by three 3D vectors.
    
    Volume = |det([u | v | w])|
    
    Parameters
    ----------
    u, v, w : np.ndarray
        3D vectors
    
    Returns
    -------
    float
        Volume of parallelepiped
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: Cramer's Rule (Medium)
# =============================================================================

def solve_cramers(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Cramer's rule.
    
    x_i = det(A_i) / det(A)
    where A_i is A with column i replaced by b
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (n×n)
    b : np.ndarray
        Right-hand side (n,)
    
    Returns
    -------
    np.ndarray
        Solution vector
    
    Raises
    ------
    ValueError
        If det(A) = 0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 10: Determinant via Row Reduction (Medium)
# =============================================================================

def det_row_reduction(A: np.ndarray) -> float:
    """
    Compute determinant using row reduction to upper triangular form.
    
    Algorithm:
    1. Make copy of A
    2. Use row operations to get upper triangular form
    3. Track sign changes from row swaps
    4. det = (±1) × product of diagonal elements
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    
    Returns
    -------
    float
        Determinant
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 11: Generalized Variance (Easy)
# =============================================================================

def generalized_variance(X: np.ndarray) -> float:
    """
    Compute the generalized variance (determinant of covariance matrix).
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples × n_features)
    
    Returns
    -------
    float
        Determinant of covariance matrix
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 12: Log-Determinant (Medium)
# =============================================================================

def log_det_cholesky(A: np.ndarray) -> float:
    """
    Compute log-determinant of a positive definite matrix using Cholesky.
    
    log(det(A)) = 2 × Σ log(L_ii) where A = L @ L.T
    
    Parameters
    ----------
    A : np.ndarray
        Positive definite matrix
    
    Returns
    -------
    float
        log(det(A))
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 13: Jacobian Determinant (Medium)
# =============================================================================

def numerical_jacobian_det(f, x: np.ndarray, eps: float = 1e-5) -> float:
    """
    Compute the Jacobian determinant of a vector function numerically.
    
    Parameters
    ----------
    f : callable
        Function f: R^n -> R^n
    x : np.ndarray
        Point at which to evaluate
    eps : float
        Step size for numerical differentiation
    
    Returns
    -------
    float
        |det(J)| where J is the Jacobian matrix
    
    Example
    -------
    >>> # Polar to Cartesian
    >>> def f(rtheta):
    ...     r, theta = rtheta
    ...     return np.array([r * np.cos(theta), r * np.sin(theta)])
    >>> numerical_jacobian_det(f, np.array([2.0, np.pi/4]))
    2.0  # Jacobian det = r
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 14: Eigenvalues Product (Easy)
# =============================================================================

def verify_eigenvalue_product(A: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Verify that det(A) equals the product of eigenvalues.
    
    det(A) = λ₁ × λ₂ × ... × λₙ
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    tol : float
        Tolerance for comparison
    
    Returns
    -------
    bool
        True if property holds within tolerance
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    # Test 1: det_2x2
    print("Test 1: det_2x2")
    try:
        assert np.isclose(det_2x2(np.array([[3, 2], [1, 4]])), 10)
        assert np.isclose(det_2x2(np.array([[1, 2], [3, 4]])), -2)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 2: det_3x3
    print("Test 2: det_3x3")
    try:
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        assert np.isclose(det_3x3(A), np.linalg.det(A))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 3: cofactor_matrix
    print("Test 3: cofactor_matrix")
    try:
        A = np.array([[1, 2], [3, 4]])
        C = cofactor_matrix(A)
        expected = np.array([[4, -3], [-2, 1]])
        assert np.allclose(C, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 4: adjugate
    print("Test 4: adjugate")
    try:
        A = np.array([[1, 2], [3, 4]])
        adj_A = adjugate(A)
        # A @ adj(A) should equal det(A) * I
        det_A = np.linalg.det(A)
        assert np.allclose(A @ adj_A, det_A * np.eye(2))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 5: inverse_via_adjugate
    print("Test 5: inverse_via_adjugate")
    try:
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A_inv = inverse_via_adjugate(A)
        assert np.allclose(A @ A_inv, np.eye(2))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 6: is_invertible
    print("Test 6: is_invertible")
    try:
        assert is_invertible(np.eye(3)) == True
        assert is_invertible(np.array([[1, 2], [2, 4]])) == False
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 7: parallelogram_area
    print("Test 7: parallelogram_area")
    try:
        assert np.isclose(parallelogram_area(np.array([2, 0]), np.array([0, 3])), 6)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 8: parallelepiped_volume
    print("Test 8: parallelepiped_volume")
    try:
        u = np.array([1, 0, 0])
        v = np.array([0, 2, 0])
        w = np.array([0, 0, 3])
        assert np.isclose(parallelepiped_volume(u, v, w), 6)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 9: solve_cramers
    print("Test 9: solve_cramers")
    try:
        A = np.array([[2., 1.], [1., 3.]])
        b = np.array([5., 6.])
        x = solve_cramers(A, b)
        assert np.allclose(A @ x, b)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 10: det_row_reduction
    print("Test 10: det_row_reduction")
    try:
        A = np.array([[2., 1., 3.], [4., 2., 1.], [1., 5., 2.]])
        det_manual = det_row_reduction(A)
        det_numpy = np.linalg.det(A)
        assert np.isclose(det_manual, det_numpy, rtol=1e-5)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 11: generalized_variance
    print("Test 11: generalized_variance")
    try:
        np.random.seed(42)
        X = np.random.randn(100, 2)
        gv = generalized_variance(X)
        expected = np.linalg.det(np.cov(X.T))
        assert np.isclose(gv, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 12: log_det_cholesky
    print("Test 12: log_det_cholesky")
    try:
        A = np.array([[4., 2.], [2., 3.]])  # Positive definite
        logdet = log_det_cholesky(A)
        expected = np.log(np.linalg.det(A))
        assert np.isclose(logdet, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 13: verify_eigenvalue_product
    print("Test 13: verify_eigenvalue_product")
    try:
        A = np.random.randn(3, 3)
        assert verify_eigenvalue_product(A) == True
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    print("\nTests complete!")


# =============================================================================
# SOLUTIONS
# =============================================================================

def det_2x2_solution(A: np.ndarray) -> float:
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def det_3x3_solution(A: np.ndarray) -> float:
    # Cofactor expansion along first row
    det = 0
    for j in range(3):
        # Minor: delete row 0 and column j
        minor = np.delete(np.delete(A, 0, axis=0), j, axis=1)
        cofactor = ((-1) ** j) * det_2x2_solution(minor)
        det += A[0, j] * cofactor
    return det


def cofactor_matrix_solution(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    C = np.zeros_like(A, dtype=float)
    
    for i in range(n):
        for j in range(n):
            # Minor: delete row i and column j
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            minor_det = np.linalg.det(minor)
            C[i, j] = ((-1) ** (i + j)) * minor_det
    
    return C


def adjugate_solution(A: np.ndarray) -> np.ndarray:
    return cofactor_matrix_solution(A).T


def inverse_via_adjugate_solution(A: np.ndarray) -> np.ndarray:
    det_A = np.linalg.det(A)
    if np.abs(det_A) < 1e-10:
        raise ValueError("Matrix is singular")
    return adjugate_solution(A) / det_A


def is_invertible_solution(A: np.ndarray, tol: float = 1e-10) -> bool:
    return np.abs(np.linalg.det(A)) > tol


def parallelogram_area_solution(u: np.ndarray, v: np.ndarray) -> float:
    return np.abs(u[0] * v[1] - u[1] * v[0])


def parallelepiped_volume_solution(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    matrix = np.column_stack([u, v, w])
    return np.abs(np.linalg.det(matrix))


def solve_cramers_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    det_A = np.linalg.det(A)
    if np.abs(det_A) < 1e-10:
        raise ValueError("det(A) = 0, system has no unique solution")
    
    n = A.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        x[i] = np.linalg.det(A_i) / det_A
    
    return x


def det_row_reduction_solution(A: np.ndarray) -> float:
    n = A.shape[0]
    U = A.astype(float).copy()
    sign = 1
    
    for k in range(n - 1):
        # Find pivot
        max_idx = k + np.argmax(np.abs(U[k:, k]))
        if np.abs(U[max_idx, k]) < 1e-10:
            return 0  # Singular
        
        # Swap if needed
        if max_idx != k:
            U[[k, max_idx]] = U[[max_idx, k]]
            sign *= -1
        
        # Eliminate below pivot
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            U[i, k:] -= factor * U[k, k:]
    
    # Product of diagonal
    return sign * np.prod(np.diag(U))


def generalized_variance_solution(X: np.ndarray) -> float:
    cov = np.cov(X.T)
    return np.linalg.det(cov)


def log_det_cholesky_solution(A: np.ndarray) -> float:
    L = np.linalg.cholesky(A)
    return 2 * np.sum(np.log(np.diag(L)))


def numerical_jacobian_det_solution(f, x: np.ndarray, eps: float = 1e-5) -> float:
    n = len(x)
    J = np.zeros((n, n))
    
    f0 = f(x)
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        f_plus = f(x_plus)
        J[:, j] = (f_plus - f0) / eps
    
    return np.abs(np.linalg.det(J))


def verify_eigenvalue_product_solution(A: np.ndarray, tol: float = 1e-6) -> bool:
    det_A = np.linalg.det(A)
    eigenvalues = np.linalg.eigvals(A)
    product = np.prod(eigenvalues)
    # Compare real parts (eigenvalues might be complex)
    return np.isclose(det_A, np.real(product), rtol=tol)


# Uncomment to use solutions
# det_2x2 = det_2x2_solution
# det_3x3 = det_3x3_solution
# cofactor_matrix = cofactor_matrix_solution
# adjugate = adjugate_solution
# inverse_via_adjugate = inverse_via_adjugate_solution
# is_invertible = is_invertible_solution
# parallelogram_area = parallelogram_area_solution
# parallelepiped_volume = parallelepiped_volume_solution
# solve_cramers = solve_cramers_solution
# det_row_reduction = det_row_reduction_solution
# generalized_variance = generalized_variance_solution
# log_det_cholesky = log_det_cholesky_solution
# numerical_jacobian_det = numerical_jacobian_det_solution
# verify_eigenvalue_product = verify_eigenvalue_product_solution


if __name__ == "__main__":
    run_tests()
