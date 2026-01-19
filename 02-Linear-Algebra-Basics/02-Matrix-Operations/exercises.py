"""
Matrix Operations - Exercises
=============================
Practice problems for matrix operations.
"""

import numpy as np
from typing import Tuple, List


# =============================================================================
# EXERCISE 1: Manual Matrix Multiplication (Easy)
# =============================================================================

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Implement matrix multiplication WITHOUT using @ or np.matmul.
    
    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n)
    B : np.ndarray
        Matrix of shape (n, p)
    
    Returns
    -------
    np.ndarray
        Product matrix of shape (m, p)
    
    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> matrix_multiply(A, B)
    array([[19, 22],
           [43, 50]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Matrix Power (Easy)
# =============================================================================

def matrix_power(A: np.ndarray, n: int) -> np.ndarray:
    """
    Compute A^n (matrix raised to power n) for square matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    n : int
        Non-negative integer power
    
    Returns
    -------
    np.ndarray
        A^n
    
    Example
    -------
    >>> A = np.array([[1, 1], [0, 1]])
    >>> matrix_power(A, 3)
    array([[1, 3],
           [0, 1]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Symmetric Part (Easy)
# =============================================================================

def symmetric_part(A: np.ndarray) -> np.ndarray:
    """
    Extract the symmetric part of a square matrix.
    
    The symmetric part is: (A + A^T) / 2
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    
    Returns
    -------
    np.ndarray
        Symmetric part
    
    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> symmetric_part(A)
    array([[1. , 2.5],
           [2.5, 4. ]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Anti-Symmetric Part (Easy)
# =============================================================================

def antisymmetric_part(A: np.ndarray) -> np.ndarray:
    """
    Extract the anti-symmetric (skew-symmetric) part of a square matrix.
    
    The anti-symmetric part is: (A - A^T) / 2
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    
    Returns
    -------
    np.ndarray
        Anti-symmetric part
    
    Example
    -------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> antisymmetric_part(A)
    array([[ 0. , -0.5],
           [ 0.5,  0. ]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: Block Matrix Multiplication (Medium)
# =============================================================================

def block_multiply(A11: np.ndarray, A12: np.ndarray, A21: np.ndarray, A22: np.ndarray,
                   B11: np.ndarray, B12: np.ndarray, B21: np.ndarray, B22: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Multiply two 2×2 block matrices.
    
    [A11 A12]   [B11 B12]   [C11 C12]
    [A21 A22] × [B21 B22] = [C21 C22]
    
    where C11 = A11×B11 + A12×B21, etc.
    
    Parameters
    ----------
    A11, A12, A21, A22 : np.ndarray
        Blocks of matrix A
    B11, B12, B21, B22 : np.ndarray
        Blocks of matrix B
    
    Returns
    -------
    Tuple of C11, C12, C21, C22
    
    Example
    -------
    >>> A11 = np.array([[1, 2], [3, 4]])
    >>> # ... (all blocks)
    >>> C11, C12, C21, C22 = block_multiply(...)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Commutator (Medium)
# =============================================================================

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute the commutator [A, B] = AB - BA.
    
    If [A, B] = 0, then A and B commute.
    
    Parameters
    ----------
    A, B : np.ndarray
        Square matrices of the same size
    
    Returns
    -------
    np.ndarray
        The commutator
    
    Example
    -------
    >>> A = np.array([[1, 0], [0, 1]])  # Identity
    >>> B = np.array([[1, 2], [3, 4]])
    >>> commutator(A, B)  # Should be zero
    array([[0, 0],
           [0, 0]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Outer Product (Easy)
# =============================================================================

def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute the outer product of two vectors: u ⊗ v = uv^T.
    
    Parameters
    ----------
    u : np.ndarray
        Column vector (m,)
    v : np.ndarray
        Column vector (n,)
    
    Returns
    -------
    np.ndarray
        Matrix of shape (m, n)
    
    Example
    -------
    >>> outer_product(np.array([1, 2]), np.array([3, 4, 5]))
    array([[ 3,  4,  5],
           [ 6,  8, 10]])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Frobenius Norm (Easy)
# =============================================================================

def frobenius_norm(A: np.ndarray) -> float:
    """
    Compute the Frobenius norm of a matrix.
    
    ||A||_F = sqrt(sum of all squared elements) = sqrt(trace(A^T A))
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix
    
    Returns
    -------
    float
        Frobenius norm
    
    Example
    -------
    >>> frobenius_norm(np.array([[1, 2], [3, 4]]))
    5.477225575051661  # sqrt(1 + 4 + 9 + 16)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: Hadamard Identity (Medium)
# =============================================================================

def verify_hadamard_identity(A: np.ndarray, B: np.ndarray) -> bool:
    """
    Verify the Hadamard product identity:
    trace(A^T ⊙ B) = sum of element-wise product = trace(A^T @ B) if one is diagonal
    
    More generally, verify: sum(A ⊙ B) = trace(A^T @ B)
    
    Parameters
    ----------
    A, B : np.ndarray
        Matrices of the same shape
    
    Returns
    -------
    bool
        True if identity holds (within tolerance)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 10: Neural Network Forward Pass (Medium)
# =============================================================================

def forward_pass(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    """
    Implement forward pass through a neural network with ReLU activations.
    
    For each layer: H = ReLU(H @ W + b)
    Final layer has no activation.
    
    Parameters
    ----------
    X : np.ndarray
        Input data (batch_size, input_dim)
    weights : List[np.ndarray]
        List of weight matrices
    biases : List[np.ndarray]
        List of bias vectors
    
    Returns
    -------
    np.ndarray
        Network output
    
    Example
    -------
    >>> X = np.random.randn(32, 784)  # 32 samples, 784 features
    >>> W1 = np.random.randn(784, 256)
    >>> W2 = np.random.randn(256, 10)
    >>> b1 = np.zeros(256)
    >>> b2 = np.zeros(10)
    >>> output = forward_pass(X, [W1, W2], [b1, b2])
    >>> output.shape
    (32, 10)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 11: Batch Normalization (Medium)
# =============================================================================

def batch_normalize(X: np.ndarray, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply batch normalization to input.
    
    For each feature: X_norm = (X - mean) / sqrt(var + epsilon)
    
    Parameters
    ----------
    X : np.ndarray
        Input (batch_size, features)
    epsilon : float
        Small constant for numerical stability
    
    Returns
    -------
    Tuple of (normalized_X, mean, variance)
    
    Example
    -------
    >>> X = np.random.randn(100, 10)
    >>> X_norm, mean, var = batch_normalize(X)
    >>> np.allclose(X_norm.mean(axis=0), 0)  # Mean ≈ 0
    True
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 12: Matrix Exponential Approximation (Hard)
# =============================================================================

def matrix_exp_taylor(A: np.ndarray, n_terms: int = 20) -> np.ndarray:
    """
    Approximate matrix exponential using Taylor series.
    
    exp(A) = I + A + A²/2! + A³/3! + ...
    
    Parameters
    ----------
    A : np.ndarray
        Square matrix
    n_terms : int
        Number of Taylor series terms
    
    Returns
    -------
    np.ndarray
        Approximation of exp(A)
    
    Example
    -------
    >>> A = np.array([[0, 1], [-1, 0]])  # Rotation generator
    >>> exp_A = matrix_exp_taylor(A)
    >>> # exp_A should be a rotation matrix
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    # Test 1: matrix_multiply
    print("Test 1: matrix_multiply")
    try:
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        result = matrix_multiply(A, B)
        expected = A @ B
        assert np.allclose(result, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 2: matrix_power
    print("Test 2: matrix_power")
    try:
        A = np.array([[1, 1], [0, 1]])
        result = matrix_power(A, 3)
        expected = np.linalg.matrix_power(A, 3)
        assert np.allclose(result, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 3: symmetric_part
    print("Test 3: symmetric_part")
    try:
        A = np.array([[1, 2], [3, 4]])
        S = symmetric_part(A)
        assert np.allclose(S, S.T)  # Is symmetric
        assert np.allclose(S, (A + A.T) / 2)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 4: antisymmetric_part
    print("Test 4: antisymmetric_part")
    try:
        A = np.array([[1, 2], [3, 4]])
        K = antisymmetric_part(A)
        assert np.allclose(K, -K.T)  # Is anti-symmetric
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 5: commutator
    print("Test 5: commutator")
    try:
        I = np.eye(2)
        B = np.array([[1, 2], [3, 4]])
        assert np.allclose(commutator(I, B), np.zeros((2, 2)))
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 6: outer_product
    print("Test 6: outer_product")
    try:
        u = np.array([1, 2])
        v = np.array([3, 4, 5])
        result = outer_product(u, v)
        expected = np.outer(u, v)
        assert np.allclose(result, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 7: frobenius_norm
    print("Test 7: frobenius_norm")
    try:
        A = np.array([[1, 2], [3, 4]])
        result = frobenius_norm(A)
        expected = np.linalg.norm(A, 'fro')
        assert np.isclose(result, expected)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 8: forward_pass
    print("Test 8: forward_pass")
    try:
        np.random.seed(42)
        X = np.random.randn(4, 3)
        W1 = np.random.randn(3, 2)
        b1 = np.zeros(2)
        output = forward_pass(X, [W1], [b1])
        assert output.shape == (4, 2)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 9: batch_normalize
    print("Test 9: batch_normalize")
    try:
        X = np.random.randn(100, 10) * 5 + 3  # Non-zero mean, non-unit variance
        X_norm, _, _ = batch_normalize(X)
        assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_norm.std(axis=0), 1, atol=0.1)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 10: matrix_exp_taylor
    print("Test 10: matrix_exp_taylor")
    try:
        A = np.array([[1, 0], [0, 1]]) * 0.1  # Small matrix
        result = matrix_exp_taylor(A)
        expected = np.exp(0.1) * np.eye(2)  # For diagonal, exp is element-wise
        assert np.allclose(result, expected, atol=1e-5)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    print("\nTests complete!")


# =============================================================================
# SOLUTIONS
# =============================================================================

def matrix_multiply_solution(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "Incompatible dimensions"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matrix_power_solution(A: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.eye(A.shape[0])
    
    result = A.copy()
    for _ in range(n - 1):
        result = result @ A
    return result


def symmetric_part_solution(A: np.ndarray) -> np.ndarray:
    return (A + A.T) / 2


def antisymmetric_part_solution(A: np.ndarray) -> np.ndarray:
    return (A - A.T) / 2


def block_multiply_solution(A11, A12, A21, A22, B11, B12, B21, B22):
    C11 = A11 @ B11 + A12 @ B21
    C12 = A11 @ B12 + A12 @ B22
    C21 = A21 @ B11 + A22 @ B21
    C22 = A21 @ B12 + A22 @ B22
    return C11, C12, C21, C22


def commutator_solution(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def outer_product_solution(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u.reshape(-1, 1) @ v.reshape(1, -1)


def frobenius_norm_solution(A: np.ndarray) -> float:
    return np.sqrt(np.sum(A ** 2))


def verify_hadamard_identity_solution(A: np.ndarray, B: np.ndarray) -> bool:
    lhs = np.sum(A * B)
    rhs = np.trace(A.T @ B)
    return np.isclose(lhs, rhs)


def forward_pass_solution(X: np.ndarray, weights: List[np.ndarray], biases: List[np.ndarray]) -> np.ndarray:
    H = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        H = H @ W + b
        if i < len(weights) - 1:  # ReLU for all but last layer
            H = np.maximum(0, H)
    return H


def batch_normalize_solution(X: np.ndarray, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return X_norm, mean, var


def matrix_exp_taylor_solution(A: np.ndarray, n_terms: int = 20) -> np.ndarray:
    n = A.shape[0]
    result = np.eye(n)
    A_power = np.eye(n)
    factorial = 1.0
    
    for k in range(1, n_terms):
        A_power = A_power @ A
        factorial *= k
        result = result + A_power / factorial
    
    return result


# Uncomment to use solutions:
# matrix_multiply = matrix_multiply_solution
# matrix_power = matrix_power_solution
# symmetric_part = symmetric_part_solution
# antisymmetric_part = antisymmetric_part_solution
# block_multiply = block_multiply_solution
# commutator = commutator_solution
# outer_product = outer_product_solution
# frobenius_norm = frobenius_norm_solution
# verify_hadamard_identity = verify_hadamard_identity_solution
# forward_pass = forward_pass_solution
# batch_normalize = batch_normalize_solution
# matrix_exp_taylor = matrix_exp_taylor_solution


if __name__ == "__main__":
    run_tests()
