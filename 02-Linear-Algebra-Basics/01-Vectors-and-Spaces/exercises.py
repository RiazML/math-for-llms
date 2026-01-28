"""
Vectors and Spaces - Exercises
==============================
Practice problems to solidify understanding of vectors and vector spaces.

Instructions:
1. Implement each function
2. Run tests to verify
3. Solutions at bottom (try first!)
"""

import numpy as np
from typing import List, Tuple, Optional


# =============================================================================
# EXERCISE 1: Dot Product Implementation (Easy)
# =============================================================================

def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the dot product of two vectors WITHOUT using np.dot.
    
    Parameters
    ----------
    u, v : np.ndarray
        Input vectors of the same dimension
    
    Returns
    -------
    float
        The dot product u · v
    
    Example
    -------
    >>> dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    32
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 2: Vector Norm (Easy)
# =============================================================================

def compute_norm(v: np.ndarray, p: int = 2) -> float:
    """
    Compute the Lp norm of a vector WITHOUT using np.linalg.norm.
    
    Lp norm: ||v||_p = (Σ|v_i|^p)^(1/p)
    
    Parameters
    ----------
    v : np.ndarray
        Input vector
    p : int
        The order of the norm (1, 2, or np.inf for max norm)
    
    Returns
    -------
    float
        The Lp norm
    
    Example
    -------
    >>> compute_norm(np.array([3, 4]), 2)
    5.0
    >>> compute_norm(np.array([3, -4]), 1)
    7.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 3: Angle Between Vectors (Easy)
# =============================================================================

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute the angle (in radians) between two vectors.
    
    Use the formula: cos(θ) = (u · v) / (||u|| ||v||)
    
    Parameters
    ----------
    u, v : np.ndarray
        Input vectors
    
    Returns
    -------
    float
        Angle in radians
    
    Example
    -------
    >>> angle_between(np.array([1, 0]), np.array([0, 1]))
    1.5707963...  # π/2
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 4: Cosine Similarity (Easy)
# =============================================================================

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    cosine_similarity = (u · v) / (||u|| ||v||)
    
    Parameters
    ----------
    u, v : np.ndarray
        Input vectors
    
    Returns
    -------
    float
        Cosine similarity in range [-1, 1]
    
    Example
    -------
    >>> cosine_similarity(np.array([1, 2]), np.array([2, 4]))
    1.0  # Vectors point in same direction
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 5: Project Vector (Medium)
# =============================================================================

def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Project vector u onto vector v.
    
    proj_v(u) = ((u · v) / (v · v)) * v
    
    Parameters
    ----------
    u : np.ndarray
        Vector to project
    v : np.ndarray
        Vector to project onto (non-zero)
    
    Returns
    -------
    np.ndarray
        The projection of u onto v
    
    Example
    -------
    >>> project_vector(np.array([3, 4]), np.array([1, 0]))
    array([3., 0.])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 6: Check Linear Independence (Medium)
# =============================================================================

def are_linearly_independent(vectors: List[np.ndarray], tolerance: float = 1e-10) -> bool:
    """
    Check if a list of vectors is linearly independent.
    
    Hint: Use matrix rank. Vectors are independent if rank equals number of vectors.
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors (all same dimension)
    tolerance : float
        Numerical tolerance for rank calculation
    
    Returns
    -------
    bool
        True if vectors are linearly independent
    
    Example
    -------
    >>> are_linearly_independent([np.array([1, 0]), np.array([0, 1])])
    True
    >>> are_linearly_independent([np.array([1, 2]), np.array([2, 4])])
    False
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 7: Gram-Schmidt Orthogonalization (Medium)
# =============================================================================

def gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Apply Gram-Schmidt process to orthonormalize a set of vectors.
    
    Algorithm:
    1. u_1 = v_1 / ||v_1||
    2. For each subsequent vector v_i:
       - Subtract projections onto all previous u_j
       - Normalize the result
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of linearly independent vectors
    
    Returns
    -------
    List[np.ndarray]
        List of orthonormal vectors
    
    Example
    -------
    >>> result = gram_schmidt([np.array([1, 1]), np.array([1, 0])])
    >>> np.allclose(np.dot(result[0], result[1]), 0)  # Orthogonal
    True
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 8: Distance Matrix (Medium)
# =============================================================================

def compute_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distance matrix for a set of vectors.
    
    Parameters
    ----------
    X : np.ndarray
        Matrix of shape (n, d) where n is number of vectors, d is dimension
    metric : str
        'euclidean' or 'cosine'
    
    Returns
    -------
    np.ndarray
        Distance matrix of shape (n, n)
    
    Example
    -------
    >>> X = np.array([[0, 0], [1, 0], [0, 1]])
    >>> D = compute_distance_matrix(X)
    >>> D[0, 1]  # Distance from [0,0] to [1,0]
    1.0
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 9: Find K Nearest Neighbors (Medium)
# =============================================================================

def find_k_nearest(query: np.ndarray, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the k nearest neighbors to a query vector.
    
    Parameters
    ----------
    query : np.ndarray
        Query vector of shape (d,)
    data : np.ndarray
        Data matrix of shape (n, d)
    k : int
        Number of neighbors to find
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (indices of k nearest neighbors, their distances)
    
    Example
    -------
    >>> data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> indices, distances = find_k_nearest(np.array([0.5, 0.5]), data, 2)
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# EXERCISE 10: Linear Combination (Easy)
# =============================================================================

def linear_combination(vectors: List[np.ndarray], coefficients: List[float]) -> np.ndarray:
    """
    Compute a linear combination of vectors.
    
    result = c_1 * v_1 + c_2 * v_2 + ... + c_n * v_n
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of vectors (same dimension)
    coefficients : List[float]
        List of scalar coefficients
    
    Returns
    -------
    np.ndarray
        The linear combination
    
    Example
    -------
    >>> linear_combination([np.array([1, 0]), np.array([0, 1])], [3, 4])
    array([3, 4])
    """
    # YOUR CODE HERE
    pass


# =============================================================================
# TESTS
# =============================================================================

def run_tests():
    """Run all tests."""
    print("Running tests...\n")
    
    # Test 1: dot_product
    print("Test 1: dot_product")
    try:
        assert dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])) == 32
        assert dot_product(np.array([1, 0]), np.array([0, 1])) == 0
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 2: compute_norm
    print("Test 2: compute_norm")
    try:
        assert np.isclose(compute_norm(np.array([3, 4]), 2), 5.0)
        assert np.isclose(compute_norm(np.array([3, -4]), 1), 7.0)
        assert np.isclose(compute_norm(np.array([3, -4, 1]), np.inf), 4.0)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 3: angle_between
    print("Test 3: angle_between")
    try:
        assert np.isclose(angle_between(np.array([1, 0]), np.array([0, 1])), np.pi/2)
        assert np.isclose(angle_between(np.array([1, 0]), np.array([1, 0])), 0)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 4: cosine_similarity
    print("Test 4: cosine_similarity")
    try:
        assert np.isclose(cosine_similarity(np.array([1, 2]), np.array([2, 4])), 1.0)
        assert np.isclose(cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 5: project_vector
    print("Test 5: project_vector")
    try:
        proj = project_vector(np.array([3, 4]), np.array([1, 0]))
        assert np.allclose(proj, [3, 0])
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 6: are_linearly_independent
    print("Test 6: are_linearly_independent")
    try:
        assert are_linearly_independent([np.array([1, 0]), np.array([0, 1])]) == True
        assert are_linearly_independent([np.array([1, 2]), np.array([2, 4])]) == False
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 7: gram_schmidt
    print("Test 7: gram_schmidt")
    try:
        result = gram_schmidt([np.array([1., 1.]), np.array([1., 0.])])
        # Check orthogonality
        assert np.isclose(np.dot(result[0], result[1]), 0, atol=1e-10)
        # Check normalization
        assert np.isclose(np.linalg.norm(result[0]), 1.0)
        assert np.isclose(np.linalg.norm(result[1]), 1.0)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 8: compute_distance_matrix
    print("Test 8: compute_distance_matrix")
    try:
        X = np.array([[0, 0], [1, 0], [0, 1]])
        D = compute_distance_matrix(X)
        assert D.shape == (3, 3)
        assert np.isclose(D[0, 1], 1.0)
        assert np.isclose(D[0, 0], 0.0)
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 9: find_k_nearest
    print("Test 9: find_k_nearest")
    try:
        data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        indices, distances = find_k_nearest(np.array([0.1, 0.1]), data, 2)
        assert 0 in indices  # [0,0] should be nearest
        assert len(indices) == 2
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    # Test 10: linear_combination
    print("Test 10: linear_combination")
    try:
        result = linear_combination([np.array([1, 0]), np.array([0, 1])], [3, 4])
        assert np.allclose(result, [3, 4])
        print("  ✓ Passed")
    except (AssertionError, TypeError):
        print("  ✗ Failed")
    
    print("\nTests complete!")


# =============================================================================
# SOLUTIONS
# =============================================================================
"""
Scroll down for solutions...
"""

def dot_product_solution(u: np.ndarray, v: np.ndarray) -> float:
    return np.sum(u * v)


def compute_norm_solution(v: np.ndarray, p: int = 2) -> float:
    if p == np.inf:
        return np.max(np.abs(v))
    return np.power(np.sum(np.abs(v) ** p), 1/p)


def angle_between_solution(u: np.ndarray, v: np.ndarray) -> float:
    cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # Clip to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def cosine_similarity_solution(u: np.ndarray, v: np.ndarray) -> float:
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def project_vector_solution(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    scalar = np.dot(u, v) / np.dot(v, v)
    return scalar * v


def are_linearly_independent_solution(vectors: List[np.ndarray], tolerance: float = 1e-10) -> bool:
    if len(vectors) == 0:
        return True
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix, tol=tolerance)
    return rank == len(vectors)


def gram_schmidt_solution(vectors: List[np.ndarray]) -> List[np.ndarray]:
    orthonormal = []
    
    for v in vectors:
        # Subtract projections onto all previous vectors
        u = v.copy().astype(float)
        for prev in orthonormal:
            u = u - np.dot(v, prev) * prev
        
        # Normalize
        norm = np.linalg.norm(u)
        if norm > 1e-10:  # Avoid division by zero
            orthonormal.append(u / norm)
    
    return orthonormal


def compute_distance_matrix_solution(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    n = X.shape[0]
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if metric == 'euclidean':
                D[i, j] = np.linalg.norm(X[i] - X[j])
            elif metric == 'cosine':
                cos_sim = np.dot(X[i], X[j]) / (np.linalg.norm(X[i]) * np.linalg.norm(X[j]))
                D[i, j] = 1 - cos_sim  # Cosine distance
    
    return D


def find_k_nearest_solution(query: np.ndarray, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(data - query, axis=1)
    indices = np.argsort(distances)[:k]
    return indices, distances[indices]


def linear_combination_solution(vectors: List[np.ndarray], coefficients: List[float]) -> np.ndarray:
    result = np.zeros_like(vectors[0], dtype=float)
    for v, c in zip(vectors, coefficients):
        result = result + c * v
    return result


# Uncomment to use solutions:
# dot_product = dot_product_solution
# compute_norm = compute_norm_solution
# angle_between = angle_between_solution
# cosine_similarity = cosine_similarity_solution
# project_vector = project_vector_solution
# are_linearly_independent = are_linearly_independent_solution
# gram_schmidt = gram_schmidt_solution
# compute_distance_matrix = compute_distance_matrix_solution
# find_k_nearest = find_k_nearest_solution
# linear_combination = linear_combination_solution


if __name__ == "__main__":
    run_tests()
