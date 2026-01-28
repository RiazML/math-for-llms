"""
Matrix Rank - Exercises
=======================
Practice problems for matrix rank concepts.
"""

import numpy as np
from scipy import linalg


class RankExercises:
    """Exercises for matrix rank concepts."""
    
    # ==================== BASIC EXERCISES ====================
    
    def exercise_1_compute_rank(self):
        """
        Exercise 1: Compute Rank
        
        Find the rank of each matrix:
        a) A = [[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]
        
        b) B = [[1, 0, 0],
                [0, 2, 0],
                [0, 0, 3]]
        
        c) C = [[1, 2],
                [2, 4],
                [3, 6]]
        
        d) D = [[1, 2, 1],
                [0, 1, 2],
                [1, 3, 3]]
        """
        # Your solutions here
        rank_a = None
        rank_b = None
        rank_c = None
        rank_d = None
        
        return rank_a, rank_b, rank_c, rank_d
    
    def solution_1(self):
        """Solution to Exercise 1."""
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        B = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [0, 0, 3]])
        
        C = np.array([[1, 2],
                      [2, 4],
                      [3, 6]])
        
        D = np.array([[1, 2, 1],
                      [0, 1, 2],
                      [1, 3, 3]])
        
        rank_a = np.linalg.matrix_rank(A)  # 2 (row3 = 2*row2 - row1)
        rank_b = np.linalg.matrix_rank(B)  # 3 (diagonal, full rank)
        rank_c = np.linalg.matrix_rank(C)  # 1 (all rows proportional)
        rank_d = np.linalg.matrix_rank(D)  # 2 (row3 = row1 + row2)
        
        print("Exercise 1 Solutions:")
        print(f"a) rank(A) = {rank_a}")
        print("   Row 3 = 2×Row 2 - Row 1")
        print(f"b) rank(B) = {rank_b}")
        print("   Diagonal matrix with non-zero diagonal")
        print(f"c) rank(C) = {rank_c}")
        print("   All rows are multiples of [1, 2]")
        print(f"d) rank(D) = {rank_d}")
        print("   Row 3 = Row 1 + Row 2")
        
        return rank_a, rank_b, rank_c, rank_d
    
    def exercise_2_rank_conditions(self):
        """
        Exercise 2: Rank Conditions
        
        For each matrix, determine if:
        - It has full row rank
        - It has full column rank
        - It is invertible (if square)
        
        a) A = [[1, 2, 3],
                [4, 5, 6]]
        
        b) B = [[1, 2],
                [3, 4],
                [5, 6]]
        
        c) C = [[1, 2],
                [3, 6]]
        """
        pass  # Analyze each matrix
    
    def solution_2(self):
        """Solution to Exercise 2."""
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        
        B = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        
        C = np.array([[1, 2],
                      [3, 6]])
        
        print("Exercise 2 Solutions:")
        
        # Matrix A (2×3)
        rank_A = np.linalg.matrix_rank(A)
        print(f"\na) A (2×3), rank = {rank_A}")
        print(f"   Full row rank? {rank_A == A.shape[0]} (rank = {rank_A} = {A.shape[0]} rows)")
        print(f"   Full column rank? {rank_A == A.shape[1]} (rank = {rank_A} ≠ {A.shape[1]} cols)")
        print("   Not square, so not invertible")
        
        # Matrix B (3×2)
        rank_B = np.linalg.matrix_rank(B)
        print(f"\nb) B (3×2), rank = {rank_B}")
        print(f"   Full row rank? {rank_B == B.shape[0]} (rank = {rank_B} ≠ {B.shape[0]} rows)")
        print(f"   Full column rank? {rank_B == B.shape[1]} (rank = {rank_B} = {B.shape[1]} cols)")
        print("   Not square, so not invertible")
        
        # Matrix C (2×2)
        rank_C = np.linalg.matrix_rank(C)
        det_C = np.linalg.det(C)
        print(f"\nc) C (2×2), rank = {rank_C}")
        print(f"   Full row rank? {rank_C == C.shape[0]}")
        print(f"   Full column rank? {rank_C == C.shape[1]}")
        print(f"   det(C) = {det_C}")
        print(f"   Invertible? {rank_C == min(C.shape) and C.shape[0] == C.shape[1]}")
    
    # ==================== INTERMEDIATE EXERCISES ====================
    
    def exercise_3_rank_nullity(self):
        """
        Exercise 3: Rank-Nullity Theorem
        
        For matrix A = [[1, 2, 0, 1],
                        [2, 4, 1, 3],
                        [3, 6, 1, 4]]:
        
        a) Compute rank(A)
        b) Compute nullity(A)
        c) Verify: rank + nullity = n (number of columns)
        d) Find a basis for the null space
        """
        A = np.array([[1, 2, 0, 1],
                      [2, 4, 1, 3],
                      [3, 6, 1, 4]])
        
        # Your solution
        rank = None
        nullity = None
        null_basis = None
        
        return rank, nullity, null_basis
    
    def solution_3(self):
        """Solution to Exercise 3."""
        A = np.array([[1, 2, 0, 1],
                      [2, 4, 1, 3],
                      [3, 6, 1, 4]])
        
        n = A.shape[1]
        rank = np.linalg.matrix_rank(A)
        nullity = n - rank
        null_basis = linalg.null_space(A)
        
        print("Exercise 3 Solution:")
        print(f"Matrix A ({A.shape[0]}×{A.shape[1]}):\n{A}")
        print(f"\na) rank(A) = {rank}")
        print(f"b) nullity(A) = n - rank = {n} - {rank} = {nullity}")
        print(f"c) Verification: {rank} + {nullity} = {rank + nullity} = {n} ✓")
        print(f"\nd) Null space basis (columns):\n{null_basis}")
        
        # Verify null space
        print("\nVerification that Ax = 0 for null space vectors:")
        for i in range(null_basis.shape[1]):
            v = null_basis[:, i]
            print(f"   A @ null_vec_{i+1} = {A @ v}")
        
        return rank, nullity, null_basis
    
    def exercise_4_find_missing_value(self):
        """
        Exercise 4: Find Missing Value
        
        For what value of k does the matrix have rank 2?
        
        A = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, k]]
        
        Hint: Think about linear dependence of rows.
        """
        # Your solution
        k = None
        return k
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("Exercise 4 Solution:")
        print("For rank to be 2, rows must be linearly dependent.")
        print("\nObservation: Row 3 = 2×Row 2 - Row 1")
        print("2×[4, 5, 6] - [1, 2, 3] = [7, 8, 9]")
        print("\nSo k = 9 makes the rank = 2")
        
        # Verify
        for k in [8, 9, 10]:
            A = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, k]])
            print(f"\nk = {k}: rank = {np.linalg.matrix_rank(A)}")
        
        return 9
    
    def exercise_5_rank_preserving(self):
        """
        Exercise 5: Rank-Preserving Operations
        
        Given A = [[1, 2, 3],
                   [0, 1, 2],
                   [1, 0, 1]]:
        
        Verify that these operations preserve rank:
        a) Scaling a row by non-zero constant
        b) Adding multiple of one row to another
        c) Transposing the matrix
        d) Multiplying by an invertible matrix
        """
        A = np.array([[1, 2, 3],
                      [0, 1, 2],
                      [1, 0, 1]], dtype=float)
        
        # Your verifications
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        A = np.array([[1, 2, 3],
                      [0, 1, 2],
                      [1, 0, 1]], dtype=float)
        
        original_rank = np.linalg.matrix_rank(A)
        
        print("Exercise 5 Solution:")
        print(f"Original matrix A:\n{A}")
        print(f"rank(A) = {original_rank}")
        
        # a) Scale row
        B = A.copy()
        B[0] *= 5
        print(f"\na) After scaling row 1 by 5:")
        print(f"   rank = {np.linalg.matrix_rank(B)} ✓")
        
        # b) Add multiple of row
        C = A.copy()
        C[2] = C[2] + 3 * C[0]
        print(f"\nb) After adding 3×Row1 to Row3:")
        print(f"   rank = {np.linalg.matrix_rank(C)} ✓")
        
        # c) Transpose
        print(f"\nc) rank(Aᵀ) = {np.linalg.matrix_rank(A.T)} ✓")
        
        # d) Multiply by invertible matrix
        P = np.array([[1, 0, 0],
                      [0, 2, 0],
                      [1, 0, 1]], dtype=float)  # Invertible
        print(f"\nd) P (invertible):\n{P}")
        print(f"   det(P) = {np.linalg.det(P)}")
        print(f"   rank(PA) = {np.linalg.matrix_rank(P @ A)} ✓")
        print(f"   rank(AP) = {np.linalg.matrix_rank(A @ P)} ✓")
    
    # ==================== ADVANCED EXERCISES ====================
    
    def exercise_6_rank_product_bound(self):
        """
        Exercise 6: Rank of Matrix Products
        
        For A (m×n) and B (n×p), prove/verify:
        rank(AB) ≤ min(rank(A), rank(B))
        
        Create matrices where:
        a) rank(AB) = min(rank(A), rank(B))
        b) rank(AB) < min(rank(A), rank(B))
        """
        # Your examples
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solution:")
        
        # a) Equality case
        A = np.array([[1, 0],
                      [0, 1],
                      [0, 0]])  # 3×2, rank 2
        
        B = np.array([[1, 0, 0],
                      [0, 1, 0]])  # 2×3, rank 2
        
        AB = A @ B
        
        print("a) Equality case:")
        print(f"   A ({A.shape}), rank = {np.linalg.matrix_rank(A)}")
        print(f"   B ({B.shape}), rank = {np.linalg.matrix_rank(B)}")
        print(f"   AB ({AB.shape}), rank = {np.linalg.matrix_rank(AB)}")
        print(f"   min(rank(A), rank(B)) = {min(np.linalg.matrix_rank(A), np.linalg.matrix_rank(B))}")
        print("   Equality achieved! ✓")
        
        # b) Strict inequality
        A2 = np.array([[1, 0],
                       [0, 1]])  # 2×2, rank 2
        
        B2 = np.array([[1, 1],
                       [-1, -1]])  # 2×2, rank 1 (but we need rank 2)
        
        # Let's use matrices that cancel
        C = np.array([[1, 0],
                      [0, 0]])  # rank 1
        
        D = np.array([[0, 1],
                      [0, 1]])  # rank 1
        
        CD = C @ D
        
        print("\nb) Strict inequality case:")
        print(f"   C:\n{C}")
        print(f"   rank(C) = {np.linalg.matrix_rank(C)}")
        print(f"   D:\n{D}")
        print(f"   rank(D) = {np.linalg.matrix_rank(D)}")
        print(f"   CD:\n{CD}")
        print(f"   rank(CD) = {np.linalg.matrix_rank(CD)}")
        print(f"   {np.linalg.matrix_rank(CD)} < min(1, 1) = 1? {np.linalg.matrix_rank(CD) < 1}")
        
        # Better example
        E = np.array([[1, 2],
                      [2, 4]])  # rank 1
        F = np.array([[-2, 0],
                      [1, 0]])  # rank 1
        
        EF = E @ F
        print(f"\n   Better example:")
        print(f"   E rank = {np.linalg.matrix_rank(E)}, F rank = {np.linalg.matrix_rank(F)}")
        print(f"   EF = \n{EF}")
        print(f"   rank(EF) = {np.linalg.matrix_rank(EF)}")
    
    def exercise_7_effective_rank(self):
        """
        Exercise 7: Effective (Numerical) Rank
        
        Create a matrix that is:
        a) Theoretically rank-deficient
        b) Add small noise to make it numerically full-rank
        c) Compare computed ranks with different tolerances
        
        Discuss implications for ML (condition number, regularization).
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("Exercise 7 Solution:")
        
        # Theoretically rank-deficient
        A = np.array([[1, 2, 3],
                      [2, 4, 6],
                      [3, 6, 9]], dtype=float)
        
        print(f"Original matrix A:\n{A}")
        print(f"Theoretical rank: 1 (all rows proportional to [1, 2, 3])")
        
        # SVD
        _, S, _ = np.linalg.svd(A)
        print(f"\nSingular values: {S}")
        print(f"NumPy rank: {np.linalg.matrix_rank(A)}")
        
        # Add noise
        np.random.seed(42)
        noise = 1e-10 * np.random.randn(3, 3)
        A_noisy = A + noise
        
        _, S_noisy, _ = np.linalg.svd(A_noisy)
        print(f"\nAfter adding 1e-10 noise:")
        print(f"Singular values: {S_noisy}")
        
        # Different tolerances
        for tol in [None, 1e-8, 1e-12, 1e-15]:
            if tol is None:
                rank = np.linalg.matrix_rank(A_noisy)
                print(f"Default tolerance: rank = {rank}")
            else:
                rank = np.linalg.matrix_rank(A_noisy, tol=tol)
                print(f"Tolerance {tol}: rank = {rank}")
        
        print("\nML Implications:")
        print("- Small singular values cause numerical instability")
        print("- Condition number = σ_max/σ_min indicates sensitivity")
        print("- Regularization (e.g., ridge regression) adds λI, preventing rank issues")
        cond = S_noisy[0] / S_noisy[-1]
        print(f"- Condition number of noisy A: {cond:.2e}")
    
    def exercise_8_pca_rank(self):
        """
        Exercise 8: PCA and Effective Rank
        
        Given data matrix X (100 samples × 10 features):
        - Some features are linear combinations of others
        
        a) Compute the rank of the covariance matrix
        b) Determine how many principal components are needed
        c) Relate this to the effective dimensionality
        """
        np.random.seed(42)
        
        # Create data with redundancy
        n_samples = 100
        x1 = np.random.randn(n_samples)
        x2 = np.random.randn(n_samples)
        x3 = np.random.randn(n_samples)
        
        # Redundant features
        X = np.column_stack([
            x1, x2, x3,
            x1 + x2,          # Redundant
            2 * x1 - x3,      # Redundant
            x2 + x3,          # Redundant
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples)
        ])
        
        # Your analysis
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        np.random.seed(42)
        
        n_samples = 100
        x1 = np.random.randn(n_samples)
        x2 = np.random.randn(n_samples)
        x3 = np.random.randn(n_samples)
        
        X = np.column_stack([
            x1, x2, x3,
            x1 + x2,
            2 * x1 - x3,
            x2 + x3,
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples),
            np.random.randn(n_samples)
        ])
        
        print("Exercise 8 Solution:")
        print(f"Data matrix X: {X.shape}")
        print(f"rank(X) = {np.linalg.matrix_rank(X)}")
        
        # Covariance matrix
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / (n_samples - 1)
        
        print(f"\nCovariance matrix: {cov.shape}")
        print(f"rank(Cov) = {np.linalg.matrix_rank(cov)}")
        
        # Eigenvalues (principal component variances)
        eigenvalues = np.linalg.eigvalsh(cov)[::-1]
        print(f"\nEigenvalues (sorted):\n{np.round(eigenvalues, 4)}")
        
        # Explained variance
        total_var = eigenvalues.sum()
        explained = eigenvalues / total_var
        cumulative = np.cumsum(explained)
        
        print("\nExplained variance ratio:")
        for i, (exp, cum) in enumerate(zip(explained, cumulative)):
            print(f"  PC{i+1}: {exp:.4f} (cumulative: {cum:.4f})")
        
        # Effective dimensionality
        n_components_95 = np.searchsorted(cumulative, 0.95) + 1
        print(f"\nComponents for 95% variance: {n_components_95}")
        print(f"Effective dimensionality ≈ {n_components_95}")
        print("\nNote: ~7 components despite 10 features (3 are redundant)")
    
    def exercise_9_system_solvability(self):
        """
        Exercise 9: System Solvability via Rank
        
        Classify each system as:
        - Unique solution
        - Infinitely many solutions
        - No solution
        
        a) [[1, 2], [3, 4]] x = [5, 11]
        b) [[1, 2], [2, 4]] x = [3, 6]
        c) [[1, 2], [2, 4]] x = [3, 7]
        d) [[1, 2, 3], [4, 5, 6]] x = [6, 15]
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("Exercise 9 Solution:")
        print("Rule: Compare rank(A) vs rank([A|b]) vs n (columns)")
        
        cases = [
            (np.array([[1, 2], [3, 4]]), np.array([5, 11]), "a"),
            (np.array([[1, 2], [2, 4]]), np.array([3, 6]), "b"),
            (np.array([[1, 2], [2, 4]]), np.array([3, 7]), "c"),
            (np.array([[1, 2, 3], [4, 5, 6]]), np.array([6, 15]), "d"),
        ]
        
        for A, b, label in cases:
            Ab = np.column_stack([A, b])
            rank_A = np.linalg.matrix_rank(A)
            rank_Ab = np.linalg.matrix_rank(Ab)
            n = A.shape[1]
            
            print(f"\n{label}) A = {A.tolist()}, b = {b.tolist()}")
            print(f"   rank(A) = {rank_A}, rank([A|b]) = {rank_Ab}, n = {n}")
            
            if rank_A != rank_Ab:
                print("   Result: NO SOLUTION (inconsistent)")
            elif rank_A == n:
                print("   Result: UNIQUE SOLUTION")
                x = np.linalg.lstsq(A, b, rcond=None)[0]
                print(f"   x = {x}")
            else:
                print(f"   Result: INFINITELY MANY SOLUTIONS")
                print(f"   Free variables: {n - rank_A}")
    
    def exercise_10_matrix_compression(self):
        """
        Exercise 10: Matrix Compression via Low-Rank Approximation
        
        Given a 100×80 matrix (e.g., grayscale image):
        a) Compute the SVD
        b) Approximate with rank-k for k = 5, 10, 20, 40
        c) Calculate compression ratio and error for each k
        d) Determine the minimum k for <5% relative error
        """
        np.random.seed(42)
        
        # Simulate structured matrix (not random)
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 80)
        X, Y = np.meshgrid(y, x)
        
        # Create pattern with low effective rank
        A = np.sin(2 * np.pi * X) * np.cos(3 * np.pi * Y) + \
            0.5 * np.sin(4 * np.pi * X) + \
            0.1 * np.random.randn(100, 80)
        
        # Your analysis
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        np.random.seed(42)
        
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 80)
        X, Y = np.meshgrid(y, x)
        
        A = np.sin(2 * np.pi * X) * np.cos(3 * np.pi * Y) + \
            0.5 * np.sin(4 * np.pi * X) + \
            0.1 * np.random.randn(100, 80)
        
        print("Exercise 10 Solution:")
        print(f"Original matrix: {A.shape}")
        print(f"Total elements: {A.size}")
        print(f"rank(A) = {np.linalg.matrix_rank(A)}")
        
        # SVD
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        print(f"\nTop 10 singular values: {np.round(S[:10], 3)}")
        
        orig_norm = np.linalg.norm(A, 'fro')
        
        print("\nLow-rank approximations:")
        print("-" * 60)
        
        min_k = None
        for k in [5, 10, 20, 40, 80]:
            # Rank-k approximation
            A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            
            # Error
            error = np.linalg.norm(A - A_k, 'fro')
            rel_error = error / orig_norm
            
            # Compression
            original_storage = A.shape[0] * A.shape[1]
            compressed_storage = k * (A.shape[0] + A.shape[1] + 1)
            compression_ratio = original_storage / compressed_storage
            
            print(f"k={k:2d}: Rel. error = {rel_error:.4f} ({rel_error*100:.1f}%), "
                  f"Compression = {compression_ratio:.1f}x "
                  f"({compressed_storage} elements)")
            
            if min_k is None and rel_error < 0.05:
                min_k = k
        
        print("-" * 60)
        print(f"Minimum k for <5% error: {min_k}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = RankExercises()
    
    print("MATRIX RANK EXERCISES")
    print("=" * 70)
    
    exercises.solution_1()
    print("\n" + "=" * 70)
    
    exercises.solution_2()
    print("\n" + "=" * 70)
    
    exercises.solution_3()
    print("\n" + "=" * 70)
    
    exercises.solution_4()
    print("\n" + "=" * 70)
    
    exercises.solution_5()
    print("\n" + "=" * 70)
    
    exercises.solution_6()
    print("\n" + "=" * 70)
    
    exercises.solution_7()
    print("\n" + "=" * 70)
    
    exercises.solution_8()
    print("\n" + "=" * 70)
    
    exercises.solution_9()
    print("\n" + "=" * 70)
    
    exercises.solution_10()


if __name__ == "__main__":
    run_all_exercises()
