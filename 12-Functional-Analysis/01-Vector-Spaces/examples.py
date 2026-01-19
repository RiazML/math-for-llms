"""
Vector Spaces - Examples
========================

Implementations demonstrating vector space concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


# =============================================================================
# Example 1: Vector Space Axiom Verification
# =============================================================================

def example_1_axiom_verification():
    """
    Verify vector space axioms for different spaces.
    """
    print("Example 1: Vector Space Axiom Verification")
    print("=" * 60)
    
    def verify_axioms_Rn(n: int = 3):
        """Verify axioms for R^n."""
        np.random.seed(42)
        
        u = np.random.randn(n)
        v = np.random.randn(n)
        w = np.random.randn(n)
        a, b = 2.5, -1.3
        
        results = {}
        
        # Commutativity
        results['A1: Commutativity'] = np.allclose(u + v, v + u)
        
        # Associativity
        results['A2: Associativity'] = np.allclose((u + v) + w, u + (v + w))
        
        # Zero vector
        zero = np.zeros(n)
        results['A3: Zero vector'] = np.allclose(v + zero, v)
        
        # Additive inverse
        results['A4: Additive inverse'] = np.allclose(v + (-v), zero)
        
        # Scalar multiplication compatibility
        results['M1: Compatibility'] = np.allclose(a * (b * v), (a * b) * v)
        
        # Identity
        results['M2: Identity'] = np.allclose(1 * v, v)
        
        # Distributivity over vector addition
        results['D1: Distributivity'] = np.allclose(a * (u + v), a * u + a * v)
        
        # Distributivity over scalar addition
        results['D2: Distributivity'] = np.allclose((a + b) * v, a * v + b * v)
        
        return results
    
    # Verify for R^3
    print("Verifying axioms for R^3:")
    results = verify_axioms_Rn(3)
    for axiom, satisfied in results.items():
        print(f"  {axiom}: {'✓' if satisfied else '✗'}")
    
    # Verify for matrix space
    print("\nVerifying axioms for R^{2x3}:")
    
    np.random.seed(42)
    A = np.random.randn(2, 3)
    B = np.random.randn(2, 3)
    C = np.random.randn(2, 3)
    a, b = 2.0, -0.5
    
    print(f"  Commutativity: {np.allclose(A + B, B + A)}")
    print(f"  Zero matrix: {np.allclose(A + np.zeros((2, 3)), A)}")
    print(f"  Scalar compatibility: {np.allclose(a * (b * A), (a * b) * A)}")
    
    return verify_axioms_Rn


# =============================================================================
# Example 2: Subspace Operations
# =============================================================================

def example_2_subspaces():
    """
    Working with fundamental subspaces of a matrix.
    """
    print("\nExample 2: Fundamental Subspaces")
    print("=" * 60)
    
    class MatrixSubspaces:
        """Compute and analyze matrix subspaces."""
        
        def __init__(self, A: np.ndarray):
            self.A = A
            self.m, self.n = A.shape
            
            # SVD for robust computation
            self.U, self.s, self.Vt = np.linalg.svd(A, full_matrices=True)
            
            # Numerical rank
            self.tol = max(self.m, self.n) * np.finfo(float).eps * self.s[0]
            self.rank = np.sum(self.s > self.tol)
        
        def column_space(self) -> np.ndarray:
            """Return orthonormal basis for Col(A)."""
            return self.U[:, :self.rank]
        
        def null_space(self) -> np.ndarray:
            """Return orthonormal basis for Null(A)."""
            return self.Vt[self.rank:, :].T
        
        def row_space(self) -> np.ndarray:
            """Return orthonormal basis for Row(A)."""
            return self.Vt[:self.rank, :].T
        
        def left_null_space(self) -> np.ndarray:
            """Return orthonormal basis for Null(A^T)."""
            return self.U[:, self.rank:]
        
        def verify_fundamental_theorem(self) -> Dict[str, bool]:
            """Verify the fundamental theorem of linear algebra."""
            col = self.column_space()
            null = self.null_space()
            row = self.row_space()
            left_null = self.left_null_space()
            
            results = {}
            
            # Col(A) ⊥ Null(A^T)
            if col.size > 0 and left_null.size > 0:
                results['Col(A) ⊥ Null(A^T)'] = np.allclose(col.T @ left_null, 0)
            else:
                results['Col(A) ⊥ Null(A^T)'] = True
            
            # Row(A) ⊥ Null(A)
            if row.size > 0 and null.size > 0:
                results['Row(A) ⊥ Null(A)'] = np.allclose(row.T @ null, 0)
            else:
                results['Row(A) ⊥ Null(A)'] = True
            
            # Dimensions
            results['dim(Col) + dim(LeftNull) = m'] = (
                col.shape[1] + left_null.shape[1] == self.m
            )
            results['dim(Row) + dim(Null) = n'] = (
                row.shape[1] + null.shape[1] == self.n
            )
            
            return results
    
    # Test matrix
    A = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],
        [1, 1, 1, 1]
    ], dtype=float)
    
    subspaces = MatrixSubspaces(A)
    
    print(f"Matrix shape: {A.shape}")
    print(f"Rank: {subspaces.rank}")
    print(f"\nDimensions:")
    print(f"  Column space: {subspaces.column_space().shape[1]}")
    print(f"  Null space: {subspaces.null_space().shape[1]}")
    print(f"  Row space: {subspaces.row_space().shape[1]}")
    print(f"  Left null space: {subspaces.left_null_space().shape[1]}")
    
    print("\nFundamental theorem verification:")
    for prop, satisfied in subspaces.verify_fundamental_theorem().items():
        print(f"  {prop}: {'✓' if satisfied else '✗'}")
    
    return MatrixSubspaces


# =============================================================================
# Example 3: Linear Independence and Basis
# =============================================================================

def example_3_linear_independence():
    """
    Testing linear independence and finding bases.
    """
    print("\nExample 3: Linear Independence and Basis")
    print("=" * 60)
    
    def is_linearly_independent(vectors: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if column vectors are linearly independent.
        
        Uses SVD to check rank.
        """
        if vectors.ndim == 1:
            return np.linalg.norm(vectors) > tol
        
        _, s, _ = np.linalg.svd(vectors)
        return np.all(s > tol)
    
    def find_basis(vectors: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Find a maximal linearly independent subset.
        
        Returns basis and indices of selected vectors.
        """
        n_vectors = vectors.shape[1]
        selected = []
        
        for i in range(n_vectors):
            test_vectors = vectors[:, selected + [i]]
            if is_linearly_independent(test_vectors):
                selected.append(i)
        
        return vectors[:, selected], selected
    
    def extend_to_basis(vectors: np.ndarray, space_dim: int) -> np.ndarray:
        """
        Extend linearly independent vectors to a full basis.
        """
        n = vectors.shape[0]
        current = vectors.copy()
        
        # Standard basis vectors
        standard = np.eye(n)
        
        for i in range(n):
            if current.shape[1] >= space_dim:
                break
            
            # Try adding standard basis vector
            test = np.column_stack([current, standard[:, i]])
            if is_linearly_independent(test):
                current = test
        
        return current
    
    # Test vectors
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 1, 0])  # Dependent on v1, v2
    v4 = np.array([0, 0, 1])
    
    vectors = np.column_stack([v1, v2, v3, v4])
    
    print("Test vectors (as columns):")
    print(vectors)
    
    print(f"\nAll 4 vectors independent: {is_linearly_independent(vectors)}")
    print(f"v1, v2, v4 independent: {is_linearly_independent(np.column_stack([v1, v2, v4]))}")
    
    # Find basis
    basis, indices = find_basis(vectors)
    print(f"\nBasis indices: {indices}")
    print(f"Basis vectors:\n{basis}")
    
    # Extend partial set to full basis
    partial = np.column_stack([v1])
    full_basis = extend_to_basis(partial, 3)
    print(f"\nExtended basis from v1:\n{full_basis}")
    
    return is_linearly_independent, find_basis


# =============================================================================
# Example 4: Gram-Schmidt Orthogonalization
# =============================================================================

def example_4_gram_schmidt():
    """
    Gram-Schmidt process for orthonormalization.
    """
    print("\nExample 4: Gram-Schmidt Process")
    print("=" * 60)
    
    def gram_schmidt(V: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Classical Gram-Schmidt orthogonalization.
        
        Args:
            V: Matrix with column vectors to orthogonalize
            normalize: Whether to normalize (orthonormal vs orthogonal)
        
        Returns:
            Matrix with orthogonal(ized) column vectors
        """
        n, k = V.shape
        Q = np.zeros((n, k))
        
        for j in range(k):
            # Start with original vector
            q = V[:, j].copy()
            
            # Subtract projections onto previous orthogonal vectors
            for i in range(j):
                q -= np.dot(Q[:, i], V[:, j]) * Q[:, i]
            
            # Normalize if requested
            norm = np.linalg.norm(q)
            if norm > 1e-10:
                if normalize:
                    Q[:, j] = q / norm
                else:
                    Q[:, j] = q
        
        return Q
    
    def modified_gram_schmidt(V: np.ndarray) -> np.ndarray:
        """
        Modified Gram-Schmidt (more numerically stable).
        
        Projects against already-computed orthonormal vectors.
        """
        n, k = V.shape
        Q = V.copy().astype(float)
        
        for j in range(k):
            # Normalize current column
            norm = np.linalg.norm(Q[:, j])
            if norm > 1e-10:
                Q[:, j] /= norm
            
            # Remove component from all subsequent columns
            for i in range(j + 1, k):
                Q[:, i] -= np.dot(Q[:, j], Q[:, i]) * Q[:, j]
        
        return Q
    
    def qr_decomposition(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """QR decomposition using Gram-Schmidt."""
        Q = modified_gram_schmidt(A)
        R = Q.T @ A
        return Q, R
    
    # Test
    np.random.seed(42)
    V = np.random.randn(4, 3)
    
    print("Original vectors (columns):")
    print(V)
    
    # Classical Gram-Schmidt
    Q_classical = gram_schmidt(V)
    print("\nClassical Gram-Schmidt result:")
    print(Q_classical)
    
    # Verify orthonormality
    print("\nQ^T Q (should be identity):")
    print(np.round(Q_classical.T @ Q_classical, 10))
    
    # QR decomposition
    Q, R = qr_decomposition(V)
    print("\nQR decomposition:")
    print(f"Q @ R ≈ V: {np.allclose(Q @ R, V)}")
    
    return gram_schmidt, modified_gram_schmidt


# =============================================================================
# Example 5: Orthogonal Projections
# =============================================================================

def example_5_projections():
    """
    Orthogonal projections onto subspaces.
    """
    print("\nExample 5: Orthogonal Projections")
    print("=" * 60)
    
    def project_onto_subspace(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
        """
        Project vector onto subspace spanned by orthonormal basis.
        
        Args:
            v: Vector to project
            basis: Matrix with orthonormal columns
        
        Returns:
            Projection of v
        """
        return basis @ (basis.T @ v)
    
    def projection_matrix_orthonormal(basis: np.ndarray) -> np.ndarray:
        """Projection matrix for orthonormal basis."""
        return basis @ basis.T
    
    def projection_matrix_general(A: np.ndarray) -> np.ndarray:
        """Projection matrix onto Col(A)."""
        return A @ np.linalg.pinv(A.T @ A) @ A.T
    
    def verify_projection_properties(P: np.ndarray) -> Dict[str, bool]:
        """Verify projection matrix properties."""
        n = P.shape[0]
        return {
            'Symmetric (P = P^T)': np.allclose(P, P.T),
            'Idempotent (P² = P)': np.allclose(P @ P, P),
            'Eigenvalues 0 or 1': np.allclose(
                np.sort(np.linalg.eigvalsh(P)), 
                np.sort(np.concatenate([np.zeros(n - np.linalg.matrix_rank(P)),
                                       np.ones(np.linalg.matrix_rank(P))]))
            )
        }
    
    # Create subspace (2D plane in R^3)
    basis = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ], dtype=float)
    basis, _ = np.linalg.qr(basis)  # Orthonormalize
    
    # Vector to project
    v = np.array([1, 2, 3])
    
    # Project
    proj = project_onto_subspace(v, basis)
    print(f"Original vector: {v}")
    print(f"Projection onto xy-plane: {proj}")
    print(f"Orthogonal component: {v - proj}")
    
    # Verify orthogonality
    print(f"proj ⊥ (v - proj): {np.allclose(np.dot(proj, v - proj), 0)}")
    
    # Projection matrix
    P = projection_matrix_orthonormal(basis)
    print("\nProjection matrix:")
    print(P)
    
    # Verify properties
    print("\nProjection matrix properties:")
    for prop, satisfied in verify_projection_properties(P).items():
        print(f"  {prop}: {'✓' if satisfied else '✗'}")
    
    return project_onto_subspace, projection_matrix_general


# =============================================================================
# Example 6: Least Squares via Projection
# =============================================================================

def example_6_least_squares():
    """
    Least squares as projection onto column space.
    """
    print("\nExample 6: Least Squares via Projection")
    print("=" * 60)
    
    def least_squares_normal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve least squares using normal equations.
        
        A^T A x = A^T b
        """
        return np.linalg.solve(A.T @ A, A.T @ b)
    
    def least_squares_projection(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, 
                                                                         np.ndarray]:
        """
        Solve least squares and return projection.
        
        Returns:
            x: Solution
            proj: Projection of b onto Col(A)
        """
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        proj = A @ x
        return x, proj
    
    def least_squares_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve least squares using QR decomposition.
        
        More stable than normal equations.
        """
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ b)
    
    # Generate data: y = 2x + 1 + noise
    np.random.seed(42)
    n_points = 20
    x = np.linspace(0, 5, n_points)
    y_true = 2 * x + 1
    y = y_true + 0.5 * np.random.randn(n_points)
    
    # Design matrix (linear regression)
    A = np.column_stack([np.ones(n_points), x])
    
    # Solve
    coeffs = least_squares_normal(A, y)
    print(f"True coefficients: [1.0, 2.0]")
    print(f"Estimated coefficients: {coeffs}")
    
    # Projection interpretation
    x_sol, y_proj = least_squares_projection(A, y)
    residual = y - y_proj
    
    print(f"\nResidual norm: {np.linalg.norm(residual):.4f}")
    print(f"Residual ⊥ Col(A): {np.allclose(A.T @ residual, 0)}")
    
    # Compare methods
    x_qr = least_squares_qr(A, y)
    print(f"\nQR solution matches: {np.allclose(coeffs, x_qr)}")
    
    return least_squares_normal, least_squares_qr


# =============================================================================
# Example 7: Change of Basis
# =============================================================================

def example_7_change_of_basis():
    """
    Coordinate transformations between bases.
    """
    print("\nExample 7: Change of Basis")
    print("=" * 60)
    
    def change_of_basis_matrix(from_basis: np.ndarray, 
                               to_basis: np.ndarray) -> np.ndarray:
        """
        Compute change of basis matrix.
        
        P converts coordinates from 'from_basis' to 'to_basis'.
        """
        return np.linalg.solve(to_basis, from_basis)
    
    def transform_coordinates(v_coords: np.ndarray,
                            from_basis: np.ndarray,
                            to_basis: np.ndarray) -> np.ndarray:
        """
        Transform coordinates from one basis to another.
        """
        P = change_of_basis_matrix(from_basis, to_basis)
        return P @ v_coords
    
    def transform_linear_map(A: np.ndarray,
                           old_basis: np.ndarray,
                           new_basis: np.ndarray) -> np.ndarray:
        """
        Transform matrix representation under change of basis.
        
        A' = P^{-1} A P
        """
        P = change_of_basis_matrix(old_basis, new_basis)
        return np.linalg.inv(P) @ A @ P
    
    # Standard basis
    std_basis = np.eye(3)
    
    # Alternative basis (rotated)
    theta = np.pi / 4
    alt_basis = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Vector in standard coordinates
    v_std = np.array([1, 0, 0])
    
    # Convert to alternative basis coordinates
    v_alt = transform_coordinates(v_std, std_basis, alt_basis)
    
    print(f"Vector in standard basis: {v_std}")
    print(f"Same vector in rotated basis: {np.round(v_alt, 4)}")
    
    # Verify: both represent same vector
    v_reconstructed = alt_basis @ v_alt
    print(f"Reconstructed: {np.round(v_reconstructed, 4)}")
    print(f"Match: {np.allclose(v_std, v_reconstructed)}")
    
    # Matrix transformation
    A = np.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 1]
    ], dtype=float)
    
    A_new = transform_linear_map(A, std_basis, alt_basis)
    print(f"\nMatrix A in standard basis:\n{A}")
    print(f"\nMatrix A' in rotated basis:\n{np.round(A_new, 4)}")
    
    return change_of_basis_matrix, transform_linear_map


# =============================================================================
# Example 8: Eigenspace Decomposition
# =============================================================================

def example_8_eigenspaces():
    """
    Eigenspace analysis and spectral decomposition.
    """
    print("\nExample 8: Eigenspace Decomposition")
    print("=" * 60)
    
    def compute_eigenspaces(A: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Compute eigenspaces for each distinct eigenvalue.
        
        Returns dict: eigenvalue -> orthonormal basis for eigenspace
        """
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # Group by eigenvalue (with tolerance for floating point)
        eigenspaces = {}
        tol = 1e-10
        
        for val, vec in zip(eigenvalues, eigenvectors.T):
            # Find matching eigenvalue
            found = False
            for existing_val in eigenspaces:
                if abs(val - existing_val) < tol:
                    eigenspaces[existing_val].append(vec)
                    found = True
                    break
            
            if not found:
                eigenspaces[float(np.real(val))] = [vec]
        
        # Orthonormalize each eigenspace
        for val in eigenspaces:
            vectors = np.column_stack(eigenspaces[val])
            Q, _ = np.linalg.qr(np.real(vectors))
            eigenspaces[val] = Q
        
        return eigenspaces
    
    def spectral_decomposition(A: np.ndarray) -> Tuple[List[float], 
                                                       List[np.ndarray]]:
        """
        Compute A = Σ λ_i P_i where P_i projects onto eigenspace.
        """
        eigenspaces = compute_eigenspaces(A)
        
        eigenvalues = []
        projections = []
        
        for val, basis in eigenspaces.items():
            eigenvalues.append(val)
            P = basis @ basis.T  # Projection matrix
            projections.append(P)
        
        return eigenvalues, projections
    
    def verify_spectral_decomposition(A: np.ndarray,
                                     eigenvalues: List[float],
                                     projections: List[np.ndarray]) -> bool:
        """Verify A = Σ λ_i P_i."""
        A_reconstructed = sum(lam * P for lam, P in zip(eigenvalues, projections))
        return np.allclose(A, A_reconstructed)
    
    # Symmetric matrix (guaranteed real eigenvalues)
    A = np.array([
        [2, 1, 0],
        [1, 2, 1],
        [0, 1, 2]
    ], dtype=float)
    
    print("Matrix A:")
    print(A)
    
    eigenspaces = compute_eigenspaces(A)
    print("\nEigenspaces:")
    for val, basis in eigenspaces.items():
        print(f"  λ = {val:.4f}: dim = {basis.shape[1]}")
    
    # Spectral decomposition
    eigenvalues, projections = spectral_decomposition(A)
    
    print("\nSpectral decomposition: A = Σ λᵢPᵢ")
    for val, P in zip(eigenvalues, projections):
        print(f"  λ = {val:.4f}, rank(P) = {np.linalg.matrix_rank(P)}")
    
    # Verify
    valid = verify_spectral_decomposition(A, eigenvalues, projections)
    print(f"\nDecomposition valid: {valid}")
    
    return compute_eigenspaces, spectral_decomposition


# =============================================================================
# Example 9: PCA as Subspace Problem
# =============================================================================

def example_9_pca_subspace():
    """
    PCA as finding optimal projection subspace.
    """
    print("\nExample 9: PCA as Subspace Problem")
    print("=" * 60)
    
    def pca_subspace(X: np.ndarray, k: int) -> Tuple[np.ndarray, 
                                                      np.ndarray, 
                                                      np.ndarray]:
        """
        Find k-dimensional subspace that maximizes variance.
        
        Args:
            X: Data matrix (n_samples, n_features), centered
            k: Target dimension
        
        Returns:
            basis: Orthonormal basis for principal subspace
            projected: Data projected onto subspace
            explained_variance: Variance explained by each component
        """
        # Center data
        X_centered = X - X.mean(axis=0)
        
        # Covariance matrix
        cov = X_centered.T @ X_centered / (len(X) - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top k
        basis = eigenvectors[:, :k]
        
        # Project data
        projected = X_centered @ basis
        
        # Variance explained
        total_var = eigenvalues.sum()
        explained = eigenvalues[:k] / total_var
        
        return basis, projected, explained
    
    def reconstruction_error(X: np.ndarray, basis: np.ndarray) -> float:
        """
        Compute reconstruction error from projection.
        
        ||X - X P P^T||_F / ||X||_F
        """
        X_centered = X - X.mean(axis=0)
        X_proj = X_centered @ basis @ basis.T
        
        return np.linalg.norm(X_centered - X_proj) / np.linalg.norm(X_centered)
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    
    # Data with clear principal directions
    X = np.random.randn(n_samples, 5)
    # Make first two components dominant
    X[:, 0] *= 3
    X[:, 1] *= 2
    
    print(f"Data shape: {X.shape}")
    
    for k in [1, 2, 3]:
        basis, projected, explained = pca_subspace(X, k)
        error = reconstruction_error(X, basis)
        
        print(f"\nk = {k}:")
        print(f"  Variance explained: {explained.sum():.2%}")
        print(f"  Reconstruction error: {error:.2%}")
    
    return pca_subspace


# =============================================================================
# Example 10: Function Spaces
# =============================================================================

def example_10_function_spaces():
    """
    Demonstrate vector space concepts in function spaces.
    """
    print("\nExample 10: Function Spaces")
    print("=" * 60)
    
    class FunctionSpace:
        """Discrete approximation of continuous function space."""
        
        def __init__(self, x_grid: np.ndarray):
            self.x = x_grid
            self.n = len(x_grid)
            self.dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1
        
        def inner_product(self, f: np.ndarray, g: np.ndarray) -> float:
            """L² inner product: ∫ f(x)g(x) dx."""
            return np.trapz(f * g, self.x)
        
        def norm(self, f: np.ndarray) -> float:
            """L² norm: ||f||₂."""
            return np.sqrt(self.inner_product(f, f))
        
        def project_onto_basis(self, f: np.ndarray, 
                              basis: List[np.ndarray]) -> np.ndarray:
            """Project f onto subspace spanned by basis functions."""
            coeffs = [self.inner_product(f, b) / self.inner_product(b, b) 
                     for b in basis]
            
            projection = np.zeros_like(f)
            for c, b in zip(coeffs, basis):
                projection += c * b
            
            return projection
        
        def orthogonalize_basis(self, 
                               basis: List[np.ndarray]) -> List[np.ndarray]:
            """Gram-Schmidt for function basis."""
            orthogonal = []
            
            for f in basis:
                g = f.copy()
                for h in orthogonal:
                    g -= (self.inner_product(g, h) / 
                          self.inner_product(h, h)) * h
                
                if self.norm(g) > 1e-10:
                    orthogonal.append(g / self.norm(g))
            
            return orthogonal
    
    # Create function space on [0, 2π]
    x = np.linspace(0, 2 * np.pi, 1000)
    space = FunctionSpace(x)
    
    # Fourier basis functions
    def fourier_basis(k: int) -> List[np.ndarray]:
        """First k Fourier basis functions."""
        basis = [np.ones_like(x)]  # Constant
        for n in range(1, k):
            basis.append(np.cos(n * x))
            basis.append(np.sin(n * x))
        return basis
    
    # Target function
    f = np.sin(x) + 0.5 * np.cos(2 * x) + 0.3 * np.sin(3 * x)
    
    print("Approximating f(x) = sin(x) + 0.5cos(2x) + 0.3sin(3x)")
    print("\nFourier approximation errors:")
    
    for k in [2, 4, 6]:
        basis = fourier_basis(k)
        f_approx = space.project_onto_basis(f, basis)
        error = space.norm(f - f_approx) / space.norm(f)
        print(f"  k = {k} terms: {error:.4%}")
    
    # Verify orthogonality of Fourier basis
    basis = fourier_basis(3)
    ortho_basis = space.orthogonalize_basis(basis)
    
    print(f"\nOriginal basis size: {len(basis)}")
    print(f"Orthogonal basis size: {len(ortho_basis)}")
    
    # Check orthonormality
    if len(ortho_basis) >= 2:
        ip = space.inner_product(ortho_basis[0], ortho_basis[1])
        print(f"Inner product of first two orthonormal: {ip:.6f}")
    
    return FunctionSpace


def run_all_examples():
    """Run all vector space examples."""
    print("=" * 70)
    print("VECTOR SPACES - EXAMPLES")
    print("=" * 70)
    
    example_1_axiom_verification()
    example_2_subspaces()
    example_3_linear_independence()
    example_4_gram_schmidt()
    example_5_projections()
    example_6_least_squares()
    example_7_change_of_basis()
    example_8_eigenspaces()
    example_9_pca_subspace()
    example_10_function_spaces()
    
    print("\n" + "=" * 70)
    print("All vector space examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
