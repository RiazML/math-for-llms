"""
Systems of Linear Equations - Exercises
=======================================
Practice problems for solving linear systems.
"""

import numpy as np
from typing import Tuple, Optional
from scipy import linalg


# =============================================================================
# EXERCISE 1: Back Substitution (Easy)
# =============================================================================

def back_substitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ux = b where U is upper triangular.
    
    Algorithm:
    - Start from last row: x_n = b_n / U_nn
    - Work backwards: x_i = (b_i - Σ U_ij * x_j) / U_ii
    
    Parameters
    ----------
    U : np.ndarray
        Upper triangular matrix (n×n)
    b : np.ndarray
        Right-hand side vector (n,)
    
    Returns
    -------
    np.ndarray
        Solution vector x
    
    Example
    -------
    >>> U = np.array([[2, 1], [0, 3]])
    >>> b = np.array([5, 6])
    >>> back_substitution(U, b)
    array([1., 2.])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Forward Substitution (Easy)
# =============================================================================

def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Lx = b where L is lower triangular.
    
    Algorithm:
    - Start from first row: x_1 = b_1 / L_11
    - Work forwards: x_i = (b_i - Σ L_ij * x_j) / L_ii
    
    Parameters
    ----------
    L : np.ndarray
        Lower triangular matrix (n×n)
    b : np.ndarray
        Right-hand side vector (n,)
    
    Returns
    -------
    np.ndarray
        Solution vector x
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Gaussian Elimination (Medium)
# =============================================================================

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Steps:
    1. Form augmented matrix [A|b]
    2. Forward elimination with pivoting
    3. Back substitution
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix (n×n)
    b : np.ndarray
        Right-hand side vector (n,)
    
    Returns
    -------
    np.ndarray
        Solution vector x
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Determine Solution Type (Easy)
# =============================================================================

def classify_system(A: np.ndarray, b: np.ndarray) -> str:
    """
    Classify a linear system as having unique, infinite, or no solution.
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix
    b : np.ndarray
        Right-hand side vector
    
    Returns
    -------
    str
        One of: "unique", "infinite", "none"
    
    Example
    -------
    >>> classify_system(np.array([[1, 0], [0, 1]]), np.array([1, 1]))
    'unique'
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: LU Decomposition (Medium)
# =============================================================================

def lu_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute LU decomposition of matrix A (without pivoting).
    
    A = LU where L is lower triangular (with 1s on diagonal)
    and U is upper triangular.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix (n×n)
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (L, U) matrices
    
    Example
    -------
    >>> A = np.array([[2, 1], [4, 3]], dtype=float)
    >>> L, U = lu_decomposition(A)
    >>> np.allclose(L @ U, A)
    True
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Solve Using LU (Medium)
# =============================================================================

def solve_lu(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve Ax = b given the LU decomposition of A.
    
    Steps:
    1. Solve Ly = b (forward substitution)
    2. Solve Ux = y (back substitution)
    
    Parameters
    ----------
    L, U : np.ndarray
        LU factors from decomposition
    b : np.ndarray
        Right-hand side vector
    
    Returns
    -------
    np.ndarray
        Solution vector x
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Normal Equations (Easy)
# =============================================================================

def solve_normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve the normal equations for linear regression.
    
    Minimizes ||Xw - y||² by solving (XᵀX)w = Xᵀy
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n×d)
    y : np.ndarray
        Target vector (n,)
    
    Returns
    -------
    np.ndarray
        Weight vector w (d,)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Ridge Regression (Easy)
# =============================================================================

def solve_ridge(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Solve ridge regression: minimize ||Xw - y||² + λ||w||²
    
    Solution: w = (XᵀX + λI)⁻¹Xᵀy
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n×d)
    y : np.ndarray
        Target vector (n,)
    lambda_reg : float
        Regularization parameter
    
    Returns
    -------
    np.ndarray
        Weight vector w (d,)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: Iterative Refinement (Medium)
# =============================================================================

def iterative_refinement(A: np.ndarray, b: np.ndarray, x0: np.ndarray, 
                         n_iter: int = 5) -> np.ndarray:
    """
    Improve an approximate solution using iterative refinement.
    
    Algorithm:
    1. Compute residual: r = b - Ax
    2. Solve: Az = r
    3. Update: x = x + z
    4. Repeat
    
    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix
    b : np.ndarray
        Right-hand side vector
    x0 : np.ndarray
        Initial approximate solution
    n_iter : int
        Number of refinement iterations
    
    Returns
    -------
    np.ndarray
        Refined solution
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 10: Block System (Hard)
# =============================================================================

def solve_block_system(A11: np.ndarray, A12: np.ndarray, 
                       A21: np.ndarray, A22: np.ndarray,
                       b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve a 2×2 block system using the Schur complement method.
    
    [A11 A12] [x1]   [b1]
    [A21 A22] [x2] = [b2]
    
    Method:
    1. Schur complement: S = A22 - A21 A11⁻¹ A12
    2. Solve: S x2 = b2 - A21 A11⁻¹ b1
    3. Solve: A11 x1 = b1 - A12 x2
    
    Parameters
    ----------
    A11, A12, A21, A22 : np.ndarray
        Block matrices
    b1, b2 : np.ndarray
        Block right-hand sides
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (x1, x2) solution blocks
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 11: Least Squares with Regularization (Medium)
# =============================================================================

def polynomial_fit_regularized(x: np.ndarray, y: np.ndarray, 
                                degree: int, lambda_reg: float) -> np.ndarray:
    """
    Fit a polynomial with ridge regularization.
    
    Parameters
    ----------
    x : np.ndarray
        Input values
    y : np.ndarray
        Output values
    degree : int
        Polynomial degree
    lambda_reg : float
        Regularization parameter
    
    Returns
    -------
    np.ndarray
        Polynomial coefficients [c0, c1, ..., c_degree]
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 12: Weighted Least Squares (Medium)
# =============================================================================

def weighted_least_squares(X: np.ndarray, y: np.ndarray, 
                           weights: np.ndarray) -> np.ndarray:
    """
    Solve weighted least squares: minimize Σ w_i (y_i - x_i'β)²
    
    Solution: β = (XᵀWX)⁻¹XᵀWy where W = diag(weights)
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n×d)
    y : np.ndarray
        Target vector (n,)
    weights : np.ndarray
        Weight for each sample (n,)
    
    Returns
    -------
    np.ndarray
        Weight vector β (d,)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    # Test 1: back_substitution
    print("Test 1: back_substitution")
    try:
        U = np.array([[2., 1.], [0., 3.]])
        b = np.array([5., 6.])
        x = back_substitution(U, b)
        assert np.allclose(U @ x, b)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 2: forward_substitution
    print("Test 2: forward_substitution")
    try:
        L = np.array([[2., 0.], [1., 3.]])
        b = np.array([4., 5.])
        x = forward_substitution(L, b)
        assert np.allclose(L @ x, b)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 3: gaussian_elimination
    print("Test 3: gaussian_elimination")
    try:
        A = np.array([[1., 2., 1.], [2., -1., 3.], [3., 1., -1.]])
        b = np.array([9., 8., 3.])
        x = gaussian_elimination(A, b)
        assert np.allclose(A @ x, b)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 4: classify_system
    print("Test 4: classify_system")
    try:
        assert classify_system(np.eye(2), np.ones(2)) == "unique"
        assert classify_system(np.array([[1, 1], [1, 1]]), np.array([2, 3])) == "none"
        assert classify_system(np.array([[1, 1], [2, 2]]), np.array([2, 4])) == "infinite"
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 5: lu_decomposition
    print("Test 5: lu_decomposition")
    try:
        A = np.array([[2., 1.], [4., 3.]])
        L, U = lu_decomposition(A)
        assert np.allclose(L @ U, A)
        assert np.allclose(np.diag(L), 1)  # L has 1s on diagonal
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 6: solve_lu
    print("Test 6: solve_lu")
    try:
        A = np.array([[2., 1.], [4., 3.]])
        L, U = lu_decomposition(A)
        b = np.array([3., 7.])
        x = solve_lu(L, U, b)
        assert np.allclose(A @ x, b)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 7: solve_normal_equations
    print("Test 7: solve_normal_equations")
    try:
        np.random.seed(42)
        X = np.random.randn(10, 3)
        true_w = np.array([1, 2, 3])
        y = X @ true_w
        w = solve_normal_equations(X, y)
        assert np.allclose(w, true_w)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 8: solve_ridge
    print("Test 8: solve_ridge")
    try:
        X = np.array([[1., 1.], [1., 2.], [1., 3.]])
        y = np.array([1., 2., 3.])
        w = solve_ridge(X, y, 0.1)
        assert w.shape == (2,)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 9: iterative_refinement
    print("Test 9: iterative_refinement")
    try:
        A = np.array([[1., 2.], [3., 4.]])
        b = np.array([5., 6.])
        x0 = np.array([0., 0.])
        x = iterative_refinement(A, b, x0)
        assert np.allclose(A @ x, b, atol=1e-10)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 10: polynomial_fit_regularized
    print("Test 10: polynomial_fit_regularized")
    try:
        x = np.array([0., 1., 2., 3., 4.])
        y = np.array([1., 3., 5., 7., 9.])  # y = 1 + 2x
        coeffs = polynomial_fit_regularized(x, y, 1, 0.0)
        # Should be close to [1, 2]
        assert len(coeffs) == 2
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 11: weighted_least_squares
    print("Test 11: weighted_least_squares")
    try:
        X = np.array([[1., 1.], [1., 2.], [1., 3.]])
        y = np.array([1., 2., 3.])
        w = np.array([1., 1., 1.])
        beta = weighted_least_squares(X, y, w)
        assert beta.shape == (2,)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    print("\nTests complete!")


# =============================================================================
# SOLUTIONS
# =============================================================================

def back_substitution_solution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x


def forward_substitution_solution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = L.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    
    return x


def gaussian_elimination_solution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    aug = np.column_stack([A.astype(float), b.astype(float)])
    
    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_idx = k + np.argmax(np.abs(aug[k:, k]))
        aug[[k, max_idx]] = aug[[max_idx, k]]
        
        # Eliminate
        for i in range(k + 1, n):
            factor = aug[i, k] / aug[k, k]
            aug[i, k:] -= factor * aug[k, k:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = aug[i, -1]
        for j in range(i + 1, n):
            x[i] -= aug[i, j] * x[j]
        x[i] /= aug[i, i]
    
    return x


def classify_system_solution(A: np.ndarray, b: np.ndarray) -> str:
    aug = np.column_stack([A, b])
    rank_A = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(aug)
    n = A.shape[1]  # Number of unknowns
    
    if rank_A < rank_aug:
        return "none"
    elif rank_A == n:
        return "unique"
    else:
        return "infinite"


def lu_decomposition_solution(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U


def solve_lu_solution(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = forward_substitution_solution(L, b)
    x = back_substitution_solution(U, y)
    return x


def solve_normal_equations_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def solve_ridge_solution(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    n, d = X.shape
    XtX = X.T @ X + lambda_reg * np.eye(d)
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def iterative_refinement_solution(A: np.ndarray, b: np.ndarray, x0: np.ndarray,
                                  n_iter: int = 5) -> np.ndarray:
    x = x0.copy()
    for _ in range(n_iter):
        r = b - A @ x
        z = np.linalg.solve(A, r)
        x = x + z
    return x


def solve_block_system_solution(A11, A12, A21, A22, b1, b2):
    A11_inv_A12 = np.linalg.solve(A11, A12)
    A11_inv_b1 = np.linalg.solve(A11, b1)
    
    # Schur complement
    S = A22 - A21 @ A11_inv_A12
    
    # Solve for x2
    x2 = np.linalg.solve(S, b2 - A21 @ A11_inv_b1)
    
    # Solve for x1
    x1 = np.linalg.solve(A11, b1 - A12 @ x2)
    
    return x1, x2


def polynomial_fit_regularized_solution(x, y, degree, lambda_reg):
    # Build design matrix
    X = np.column_stack([x ** i for i in range(degree + 1)])
    return solve_ridge_solution(X, y, lambda_reg)


def weighted_least_squares_solution(X, y, weights):
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    return np.linalg.solve(XtWX, XtWy)


# Uncomment to use solutions
# back_substitution = back_substitution_solution
# forward_substitution = forward_substitution_solution
# gaussian_elimination = gaussian_elimination_solution
# classify_system = classify_system_solution
# lu_decomposition = lu_decomposition_solution
# solve_lu = solve_lu_solution
# solve_normal_equations = solve_normal_equations_solution
# solve_ridge = solve_ridge_solution
# iterative_refinement = iterative_refinement_solution
# solve_block_system = solve_block_system_solution
# polynomial_fit_regularized = polynomial_fit_regularized_solution
# weighted_least_squares = weighted_least_squares_solution


if __name__ == "__main__":
    run_tests()
