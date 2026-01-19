"""
Vector Spaces - Exercises
=========================

Hands-on exercises for mastering vector space concepts
with applications to machine learning.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


class VectorSpaceExercises:
    """Exercises for vector spaces."""
    
    # =========================================================================
    # Exercise 1: Subspace Verification
    # =========================================================================
    
    @staticmethod
    def exercise_1_subspace_verification():
        """
        Verify whether given sets form subspaces.
        
        A set W is a subspace if:
        1. 0 ∈ W
        2. u, v ∈ W → u + v ∈ W (closed under addition)
        3. v ∈ W, c ∈ F → cv ∈ W (closed under scalar multiplication)
        """
        
        def is_subspace_Rn(check_membership: Callable[[np.ndarray], bool],
                          n: int,
                          num_tests: int = 100) -> Dict[str, bool]:
            """
            Test if a set defined by check_membership is a subspace of R^n.
            
            Args:
                check_membership: Function returning True if vector is in set
                n: Dimension
                num_tests: Number of random tests
            
            Returns:
                Dict with test results
            
            TODO: Implement
            """
            # 1. Check zero vector
            # 2. Test closure under addition
            # 3. Test closure under scalar multiplication
            pass
        
        def verify_column_space_is_subspace(A: np.ndarray) -> bool:
            """
            Verify Col(A) satisfies subspace axioms.
            
            TODO: Implement
            """
            pass
        
        def verify_solution_set_subspace(A: np.ndarray, b: np.ndarray) -> bool:
            """
            Check if solution set {x : Ax = b} is a subspace.
            
            Note: Only when b = 0!
            
            TODO: Implement
            """
            pass
        
        # Test cases
        print("Exercise 1: Subspace Verification")
        print("-" * 40)
        
        # Set 1: Vectors with first component = 0
        def set1(v):
            return abs(v[0]) < 1e-10
        
        # Set 2: Vectors with non-negative components
        def set2(v):
            return np.all(v >= 0)
        
        # Set 3: Vectors with components summing to 1
        def set3(v):
            return abs(sum(v) - 1) < 1e-10
        
        # Uncomment to test:
        # result1 = is_subspace_Rn(set1, n=3)
        # print(f"  Set 1 (first=0): {result1}")
        
        return is_subspace_Rn
    
    # =========================================================================
    # Exercise 2: Finding Bases
    # =========================================================================
    
    @staticmethod
    def exercise_2_finding_bases():
        """
        Find bases for various subspaces.
        """
        
        def find_null_space_basis(A: np.ndarray) -> np.ndarray:
            """
            Find orthonormal basis for Null(A).
            
            TODO: Use SVD approach
            """
            pass
        
        def find_column_space_basis(A: np.ndarray) -> np.ndarray:
            """
            Find orthonormal basis for Col(A).
            
            TODO: Use SVD or QR
            """
            pass
        
        def find_intersection_basis(U: np.ndarray, W: np.ndarray) -> np.ndarray:
            """
            Find basis for intersection of two subspaces.
            
            U, W are given as matrices whose columns span each subspace.
            
            TODO: Implement using the fact that
                  U ∩ W = {v : v = Ux = Wy for some x, y}
            """
            pass
        
        def find_sum_basis(U: np.ndarray, W: np.ndarray) -> np.ndarray:
            """
            Find basis for U + W (sum of subspaces).
            
            U + W = {u + w : u ∈ U, w ∈ W}
            
            TODO: Implement
            """
            pass
        
        def verify_dimension_formula(U: np.ndarray, W: np.ndarray) -> bool:
            """
            Verify: dim(U + W) = dim(U) + dim(W) - dim(U ∩ W)
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 2: Finding Bases")
        print("-" * 40)
        
        A = np.array([
            [1, 2, 3, 0],
            [0, 1, 1, 1],
            [1, 3, 4, 1]
        ], dtype=float)
        
        # Uncomment to test:
        # null_basis = find_null_space_basis(A)
        # print(f"  Null space dimension: {null_basis.shape[1]}")
        
        return find_null_space_basis, find_intersection_basis
    
    # =========================================================================
    # Exercise 3: Orthogonalization Methods
    # =========================================================================
    
    @staticmethod
    def exercise_3_orthogonalization():
        """
        Implement and compare orthogonalization methods.
        """
        
        def classical_gram_schmidt(V: np.ndarray) -> np.ndarray:
            """
            Classical Gram-Schmidt.
            
            TODO: Implement
            """
            pass
        
        def modified_gram_schmidt(V: np.ndarray) -> np.ndarray:
            """
            Modified Gram-Schmidt (more stable).
            
            TODO: Implement
            """
            pass
        
        def householder_qr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """
            QR decomposition using Householder reflections.
            
            More stable than Gram-Schmidt.
            
            TODO: Implement
            
            Hint: Householder reflection:
                  H = I - 2vv^T / (v^T v)
                  Choose v to zero out below-diagonal elements.
            """
            pass
        
        def compare_stability(n: int = 50, condition: float = 1e8):
            """
            Compare numerical stability of different methods.
            
            Create ill-conditioned matrix and compare orthogonality loss.
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 3: Orthogonalization Methods")
        print("-" * 40)
        
        np.random.seed(42)
        V = np.random.randn(5, 4)
        
        # Uncomment to test:
        # Q_classical = classical_gram_schmidt(V)
        # Q_modified = modified_gram_schmidt(V)
        # print(f"  Classical orthogonality error: {np.linalg.norm(Q_classical.T @ Q_classical - np.eye(4))}")
        # print(f"  Modified orthogonality error: {np.linalg.norm(Q_modified.T @ Q_modified - np.eye(4))}")
        
        return classical_gram_schmidt, modified_gram_schmidt, householder_qr
    
    # =========================================================================
    # Exercise 4: Projection Operations
    # =========================================================================
    
    @staticmethod
    def exercise_4_projections():
        """
        Implement and analyze projection operations.
        """
        
        def orthogonal_projection(v: np.ndarray, 
                                 W_basis: np.ndarray) -> np.ndarray:
            """
            Project v onto subspace spanned by columns of W_basis.
            
            TODO: Handle both orthonormal and non-orthonormal bases.
            """
            pass
        
        def oblique_projection(v: np.ndarray,
                              range_space: np.ndarray,
                              null_space: np.ndarray) -> np.ndarray:
            """
            Oblique projection onto range_space along null_space.
            
            The projection P satisfies:
            - Im(P) = range_space
            - Null(P) = null_space
            
            TODO: Implement
            """
            pass
        
        def projection_error(v: np.ndarray, 
                           W_basis: np.ndarray) -> float:
            """
            Compute ||v - proj_W(v)||.
            
            TODO: Implement
            """
            pass
        
        def best_rank_k_approx(A: np.ndarray, k: int) -> np.ndarray:
            """
            Find best rank-k approximation of matrix A.
            
            This is a projection problem in matrix space!
            
            TODO: Implement using SVD
            """
            pass
        
        # Test
        print("\nExercise 4: Projection Operations")
        print("-" * 40)
        
        # Subspace: xy-plane
        W_basis = np.array([
            [1, 0],
            [0, 1],
            [0, 0]
        ], dtype=float)
        
        v = np.array([1, 2, 3])
        
        # Uncomment to test:
        # proj = orthogonal_projection(v, W_basis)
        # print(f"  v = {v}")
        # print(f"  proj = {proj}")
        # print(f"  error = {projection_error(v, W_basis)}")
        
        return orthogonal_projection, best_rank_k_approx
    
    # =========================================================================
    # Exercise 5: Linear Transformations
    # =========================================================================
    
    @staticmethod
    def exercise_5_linear_transformations():
        """
        Analyze and construct linear transformations.
        """
        
        def verify_linearity(T: Callable, input_dim: int, 
                            num_tests: int = 100) -> bool:
            """
            Test if function T is linear.
            
            TODO: Check T(u+v) = T(u) + T(v) and T(cv) = cT(v)
            """
            pass
        
        def find_matrix_representation(T: Callable, 
                                      input_dim: int,
                                      output_dim: int) -> np.ndarray:
            """
            Find matrix A such that T(x) = Ax.
            
            TODO: Apply T to standard basis vectors.
            """
            pass
        
        def compose_transformations(A: np.ndarray, 
                                   B: np.ndarray) -> np.ndarray:
            """
            Find matrix for composition T_A ∘ T_B.
            
            (T_A ∘ T_B)(x) = T_A(T_B(x)) = A(Bx) = (AB)x
            
            TODO: Implement and verify
            """
            pass
        
        def inverse_transformation(A: np.ndarray) -> Optional[np.ndarray]:
            """
            Find inverse transformation if it exists.
            
            TODO: Check invertibility and compute inverse
            """
            pass
        
        # Test
        print("\nExercise 5: Linear Transformations")
        print("-" * 40)
        
        # Rotation by 45 degrees in R²
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # Scaling
        S = np.diag([2, 3])
        
        # Uncomment to test:
        # composition = compose_transformations(R, S)
        # print(f"  Rotation then scale matrix:\n{composition}")
        
        return find_matrix_representation
    
    # =========================================================================
    # Exercise 6: Eigenspaces and Diagonalization
    # =========================================================================
    
    @staticmethod
    def exercise_6_eigenspaces():
        """
        Work with eigenspaces and diagonalization.
        """
        
        def compute_eigenspace(A: np.ndarray, 
                              eigenvalue: float) -> np.ndarray:
            """
            Compute orthonormal basis for eigenspace E_λ = Null(A - λI).
            
            TODO: Implement
            """
            pass
        
        def is_diagonalizable(A: np.ndarray) -> bool:
            """
            Check if A is diagonalizable.
            
            A is diagonalizable iff algebraic multiplicity = geometric 
            multiplicity for all eigenvalues.
            
            TODO: Implement
            """
            pass
        
        def diagonalize(A: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
            """
            Find P, D such that A = PDP^{-1}.
            
            Returns None if not diagonalizable.
            
            TODO: Implement
            """
            pass
        
        def matrix_power_via_diagonalization(A: np.ndarray, 
                                            k: int) -> np.ndarray:
            """
            Compute A^k using diagonalization.
            
            A^k = P D^k P^{-1}
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 6: Eigenspaces and Diagonalization")
        print("-" * 40)
        
        A = np.array([
            [2, 1],
            [0, 2]
        ], dtype=float)  # Not diagonalizable
        
        B = np.array([
            [3, 1],
            [0, 2]
        ], dtype=float)  # Diagonalizable
        
        # Uncomment to test:
        # print(f"  A diagonalizable: {is_diagonalizable(A)}")
        # print(f"  B diagonalizable: {is_diagonalizable(B)}")
        
        return is_diagonalizable, diagonalize
    
    # =========================================================================
    # Exercise 7: Inner Products and Norms
    # =========================================================================
    
    @staticmethod
    def exercise_7_inner_products():
        """
        Work with general inner products and induced norms.
        """
        
        def verify_inner_product_axioms(inner_prod: Callable,
                                       dim: int) -> Dict[str, bool]:
            """
            Verify axioms of inner product.
            
            TODO: Check positivity, symmetry, linearity
            """
            pass
        
        def mahalanobis_inner_product(x: np.ndarray, y: np.ndarray,
                                     M: np.ndarray) -> float:
            """
            Weighted inner product: <x, y>_M = x^T M y
            
            M must be positive definite.
            
            TODO: Implement
            """
            pass
        
        def induced_norm(x: np.ndarray, M: np.ndarray) -> float:
            """
            Norm induced by weighted inner product.
            
            TODO: Implement
            """
            pass
        
        def angle_in_inner_product(x: np.ndarray, y: np.ndarray,
                                  M: np.ndarray) -> float:
            """
            Angle between vectors using weighted inner product.
            
            cos(θ) = <x, y> / (||x|| ||y||)
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 7: Inner Products and Norms")
        print("-" * 40)
        
        # Positive definite matrix for weighted inner product
        L = np.array([[2, 1], [0, 1]])
        M = L.T @ L  # M = L^T L is positive definite
        
        x = np.array([1, 0])
        y = np.array([1, 1])
        
        # Uncomment to test:
        # print(f"  Mahalanobis inner product: {mahalanobis_inner_product(x, y, M)}")
        # print(f"  Standard inner product: {np.dot(x, y)}")
        
        return mahalanobis_inner_product, induced_norm
    
    # =========================================================================
    # Exercise 8: Dual Spaces
    # =========================================================================
    
    @staticmethod
    def exercise_8_dual_spaces():
        """
        Explore dual spaces and linear functionals.
        """
        
        def evaluate_dual_vector(dual: np.ndarray, 
                                vector: np.ndarray) -> float:
            """
            Apply dual vector (linear functional) to vector.
            
            In finite dimensions: <dual, vector> = dual^T vector
            
            TODO: Implement
            """
            pass
        
        def find_dual_basis(basis: np.ndarray) -> np.ndarray:
            """
            Find dual basis {e^1, ..., e^n} such that e^i(e_j) = δ_ij.
            
            TODO: Implement
            """
            pass
        
        def represent_functional(f_values: np.ndarray, 
                               basis: np.ndarray) -> np.ndarray:
            """
            Find dual vector representing functional.
            
            Given f(e_i) = f_values[i], find dual representation.
            
            TODO: Implement
            """
            pass
        
        def bidual_isomorphism(v: np.ndarray) -> Callable:
            """
            Map vector to its canonical image in bidual V**.
            
            v → (φ → φ(v))
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 8: Dual Spaces")
        print("-" * 40)
        
        # Non-standard basis
        basis = np.array([
            [1, 1],
            [0, 1]
        ], dtype=float)
        
        # Uncomment to test:
        # dual_basis = find_dual_basis(basis)
        # Check: dual_basis[:, i] @ basis[:, j] = delta_ij
        
        return find_dual_basis
    
    # =========================================================================
    # Exercise 9: Quotient Spaces
    # =========================================================================
    
    @staticmethod
    def exercise_9_quotient_spaces():
        """
        Work with quotient spaces V/W.
        """
        
        def coset_representative(v: np.ndarray, 
                                W_basis: np.ndarray) -> np.ndarray:
            """
            Find canonical representative for coset v + W.
            
            Choose the element orthogonal to W.
            
            TODO: Implement
            """
            pass
        
        def quotient_space_basis(V_basis: np.ndarray,
                                W_basis: np.ndarray) -> np.ndarray:
            """
            Find basis for quotient space V/W.
            
            dim(V/W) = dim(V) - dim(W)
            
            TODO: Implement
            """
            pass
        
        def quotient_norm(v: np.ndarray, W_basis: np.ndarray) -> float:
            """
            Compute quotient norm: ||v + W|| = inf_{w ∈ W} ||v - w||
            
            This equals ||proj_{W⊥}(v)||.
            
            TODO: Implement
            """
            pass
        
        # Test
        print("\nExercise 9: Quotient Spaces")
        print("-" * 40)
        
        # V = R³, W = span{(1,0,0)}
        W_basis = np.array([[1], [0], [0]], dtype=float)
        v = np.array([1, 2, 3])
        
        # Uncomment to test:
        # rep = coset_representative(v, W_basis)
        # print(f"  v = {v}")
        # print(f"  Canonical representative: {rep}")
        
        return coset_representative, quotient_space_basis
    
    # =========================================================================
    # Exercise 10: Vector Spaces in ML
    # =========================================================================
    
    @staticmethod
    def exercise_10_ml_applications():
        """
        Apply vector space concepts to ML problems.
        """
        
        def kernel_feature_space(X: np.ndarray, 
                                kernel: str = 'polynomial',
                                degree: int = 2) -> np.ndarray:
            """
            Explicit feature map to kernel space.
            
            For polynomial kernel: φ(x) includes all monomials up to degree.
            
            TODO: Implement polynomial feature expansion
            """
            pass
        
        def embedding_subspace_analysis(embeddings: np.ndarray,
                                       labels: np.ndarray) -> Dict:
            """
            Analyze subspace structure of embeddings by class.
            
            TODO: 
            1. Compute mean per class
            2. Find principal subspace per class
            3. Measure overlap between class subspaces
            """
            pass
        
        def attention_as_projection(Q: np.ndarray, 
                                   K: np.ndarray,
                                   V: np.ndarray) -> np.ndarray:
            """
            Interpret attention as soft projection.
            
            Attention(Q, K, V) projects values based on query-key similarity.
            
            TODO: Implement scaled dot-product attention
            """
            pass
        
        def neural_network_subspaces(weights: List[np.ndarray]) -> Dict:
            """
            Analyze subspace properties of neural network layers.
            
            TODO:
            1. Compute rank of each layer
            2. Find null space (redundant parameters)
            3. Analyze effective dimension
            """
            pass
        
        # Test
        print("\nExercise 10: ML Applications")
        print("-" * 40)
        
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        
        # Uncomment to test:
        # X_expanded = kernel_feature_space(X, 'polynomial', degree=2)
        # print(f"  Original dim: {X.shape[1]}")
        # print(f"  Expanded dim: {X_expanded.shape[1]}")
        
        return attention_as_projection


def verify_implementations():
    """Run all exercise stubs."""
    print("=" * 60)
    print("Vector Spaces - Exercise Verification")
    print("=" * 60)
    
    exercises = VectorSpaceExercises()
    
    exercises.exercise_1_subspace_verification()
    exercises.exercise_2_finding_bases()
    exercises.exercise_3_orthogonalization()
    exercises.exercise_4_projections()
    exercises.exercise_5_linear_transformations()
    exercises.exercise_6_eigenspaces()
    exercises.exercise_7_inner_products()
    exercises.exercise_8_dual_spaces()
    exercises.exercise_9_quotient_spaces()
    exercises.exercise_10_ml_applications()
    
    print("\n" + "=" * 60)
    print("Complete the TODO sections in each exercise!")
    print("=" * 60)


# =============================================================================
# Solutions (Reference Implementation)
# =============================================================================

class Solutions:
    """Reference solutions for exercises."""
    
    @staticmethod
    def solution_1_subspace():
        """Solution for Exercise 1."""
        
        def is_subspace_Rn(check_membership, n, num_tests=100):
            np.random.seed(42)
            results = {}
            
            # Check zero vector
            results['zero_in_set'] = check_membership(np.zeros(n))
            
            # Check closure under addition
            addition_closed = True
            for _ in range(num_tests):
                u = np.random.randn(n)
                v = np.random.randn(n)
                if check_membership(u) and check_membership(v):
                    if not check_membership(u + v):
                        addition_closed = False
                        break
            results['closed_addition'] = addition_closed
            
            # Check closure under scalar multiplication
            scalar_closed = True
            for _ in range(num_tests):
                v = np.random.randn(n)
                c = np.random.randn()
                if check_membership(v):
                    if not check_membership(c * v):
                        scalar_closed = False
                        break
            results['closed_scalar'] = scalar_closed
            
            results['is_subspace'] = all(results.values())
            return results
        
        return is_subspace_Rn
    
    @staticmethod
    def solution_2_null_space():
        """Solution for Exercise 2."""
        
        def find_null_space_basis(A):
            U, s, Vt = np.linalg.svd(A, full_matrices=True)
            tol = max(A.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 0
            rank = np.sum(s > tol)
            return Vt[rank:, :].T
        
        return find_null_space_basis
    
    @staticmethod
    def solution_4_projection():
        """Solution for Exercise 4."""
        
        def orthogonal_projection(v, W_basis):
            # Orthonormalize basis first
            Q, _ = np.linalg.qr(W_basis)
            return Q @ (Q.T @ v)
        
        def best_rank_k_approx(A, k):
            U, s, Vt = np.linalg.svd(A, full_matrices=False)
            return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        
        return orthogonal_projection, best_rank_k_approx
    
    @staticmethod
    def solution_6_diagonalize():
        """Solution for Exercise 6."""
        
        def is_diagonalizable(A):
            eigenvalues, eigenvectors = np.linalg.eig(A)
            n = A.shape[0]
            
            # Check if we have n linearly independent eigenvectors
            _, s, _ = np.linalg.svd(eigenvectors)
            return np.sum(s > 1e-10) == n
        
        def diagonalize(A):
            if not is_diagonalizable(A):
                return None
            
            eigenvalues, eigenvectors = np.linalg.eig(A)
            P = eigenvectors
            D = np.diag(eigenvalues)
            
            return P, D
        
        return is_diagonalizable, diagonalize
    
    @staticmethod
    def solution_10_attention():
        """Solution for Exercise 10."""
        
        def attention_as_projection(Q, K, V):
            d_k = K.shape[-1]
            scores = Q @ K.T / np.sqrt(d_k)
            
            # Softmax
            exp_scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
            
            return attention_weights @ V
        
        return attention_as_projection


if __name__ == "__main__":
    verify_implementations()
