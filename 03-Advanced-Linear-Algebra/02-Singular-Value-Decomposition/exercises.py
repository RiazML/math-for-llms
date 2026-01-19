"""
Singular Value Decomposition (SVD) - Exercises
===============================================
Practice problems for SVD concepts.
"""

import numpy as np
from numpy.linalg import svd, norm, matrix_rank, pinv, eig


class SVDExercises:
    """Exercises for Singular Value Decomposition."""
    
    def exercise_1_basic_svd(self):
        """
        Exercise 1: Basic SVD Computation
        
        Compute the SVD of A = [[3, 0], [0, 2], [0, 0]]
        
        a) What are the singular values?
        b) What are the left singular vectors (U)?
        c) What are the right singular vectors (V)?
        d) Verify A = UΣVᵀ
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Basic SVD")
        
        A = np.array([[3, 0],
                      [0, 2],
                      [0, 0]])
        
        print(f"A = \n{A}")
        
        print("\na) Since A is already 'diagonal', SVD is simple:")
        print("   σ₁ = 3, σ₂ = 2")
        
        U, s, Vt = svd(A)
        print(f"\nNumPy SVD:")
        print(f"   Singular values: {s}")
        print(f"\nb) U (left singular vectors):\n{np.round(U, 4)}")
        print(f"\nc) V (right singular vectors):\n{np.round(Vt.T, 4)}")
        
        print("\nd) Verification:")
        S = np.zeros_like(A, dtype=float)
        S[:2, :2] = np.diag(s)
        reconstructed = U @ S @ Vt
        print(f"   UΣVᵀ = \n{np.round(reconstructed, 4)}")
        print(f"   Matches A: {np.allclose(A, reconstructed)}")
    
    def exercise_2_singular_values_from_eigenvalues(self):
        """
        Exercise 2: Singular Values from Eigenvalues
        
        For A = [[1, 2], [0, 2]]:
        a) Compute AᵀA
        b) Find eigenvalues of AᵀA
        c) Compute singular values as √(eigenvalues)
        d) Verify with NumPy's SVD
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Singular Values from Eigenvalues")
        
        A = np.array([[1, 2],
                      [0, 2]])
        
        print(f"A = \n{A}")
        
        print("\na) Computing AᵀA:")
        AtA = A.T @ A
        print(f"   AᵀA = \n{AtA}")
        
        print("\nb) Eigenvalues of AᵀA:")
        eigenvalues, eigenvectors = eig(AtA)
        eigenvalues = np.sort(eigenvalues.real)[::-1]  # Sort descending
        print(f"   λ = {eigenvalues}")
        
        print("\nc) Singular values = √λ:")
        singular_values = np.sqrt(eigenvalues)
        print(f"   σ = {singular_values}")
        
        print("\nd) Verification with NumPy SVD:")
        U, s, Vt = svd(A)
        print(f"   NumPy singular values: {s}")
        print(f"   Match: {np.allclose(sorted(singular_values, reverse=True), s)}")
    
    def exercise_3_low_rank_approximation(self):
        """
        Exercise 3: Low-Rank Approximation
        
        For A = [[4, 0], [3, -5]]:
        a) Compute the SVD
        b) Find the best rank-1 approximation A₁
        c) Compute the approximation error ||A - A₁||_F
        d) Verify the error equals σ₂
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Low-Rank Approximation")
        
        A = np.array([[4, 0],
                      [3, -5]])
        
        print(f"A = \n{A}")
        
        print("\na) SVD of A:")
        U, s, Vt = svd(A)
        print(f"   σ₁ = {s[0]:.4f}, σ₂ = {s[1]:.4f}")
        print(f"   U = \n{np.round(U, 4)}")
        print(f"   Vᵀ = \n{np.round(Vt, 4)}")
        
        print("\nb) Rank-1 approximation A₁ = σ₁u₁v₁ᵀ:")
        A1 = s[0] * np.outer(U[:, 0], Vt[0, :])
        print(f"   A₁ = \n{np.round(A1, 4)}")
        
        print("\nc) Approximation error:")
        error_F = norm(A - A1, 'fro')
        print(f"   ||A - A₁||_F = {error_F:.4f}")
        
        print("\nd) Verification:")
        print(f"   σ₂ = {s[1]:.4f}")
        print(f"   ||A - A₁||_F = σ₂: {np.isclose(error_F, s[1])}")
        print("   (For rank-1 approx, ||A-A₁||₂ = σ₂, but ||A-A₁||_F = √(σ₂²) = σ₂)")
    
    def exercise_4_matrix_norms(self):
        """
        Exercise 4: Matrix Norms
        
        For A = [[3, 4], [0, 5]]:
        a) Compute singular values
        b) Find spectral norm ||A||₂
        c) Find Frobenius norm ||A||_F using singular values
        d) Find nuclear norm ||A||_*
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Matrix Norms")
        
        A = np.array([[3, 4],
                      [0, 5]])
        
        print(f"A = \n{A}")
        
        print("\na) Singular values:")
        U, s, Vt = svd(A)
        print(f"   σ = {s}")
        
        print("\nb) Spectral norm ||A||₂ = σ₁:")
        spectral = s[0]
        print(f"   ||A||₂ = {spectral:.4f}")
        print(f"   NumPy: {norm(A, 2):.4f}")
        
        print("\nc) Frobenius norm ||A||_F = √(Σσᵢ²):")
        frobenius_svd = np.sqrt(np.sum(s**2))
        print(f"   ||A||_F = √({s[0]**2:.4f} + {s[1]**2:.4f}) = {frobenius_svd:.4f}")
        print(f"   NumPy: {norm(A, 'fro'):.4f}")
        
        print("\nd) Nuclear norm ||A||_* = Σσᵢ:")
        nuclear = np.sum(s)
        print(f"   ||A||_* = {s[0]:.4f} + {s[1]:.4f} = {nuclear:.4f}")
        print(f"   NumPy: {np.linalg.norm(A, 'nuc'):.4f}")
    
    def exercise_5_pseudoinverse(self):
        """
        Exercise 5: Pseudoinverse
        
        For A = [[1, 0], [0, 1], [0, 0]]:
        a) Compute the SVD
        b) Compute the pseudoinverse A⁺ using SVD
        c) Verify A⁺AA⁺ = A⁺
        d) Solve the least squares problem Ax ≈ [1, 2, 3]ᵀ
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Pseudoinverse")
        
        A = np.array([[1, 0],
                      [0, 1],
                      [0, 0]])
        
        print(f"A = \n{A}")
        
        print("\na) SVD:")
        U, s, Vt = svd(A)
        print(f"   σ = {s}")
        
        print("\nb) Pseudoinverse A⁺ = VΣ⁺Uᵀ:")
        # Σ⁺ inverts non-zero singular values
        S_pinv = np.zeros((2, 3))
        for i in range(len(s)):
            if s[i] > 1e-10:
                S_pinv[i, i] = 1 / s[i]
        
        A_pinv = Vt.T @ S_pinv @ U.T
        print(f"   A⁺ = \n{np.round(A_pinv, 4)}")
        print(f"   NumPy pinv: \n{np.round(pinv(A), 4)}")
        
        print("\nc) Verification A⁺AA⁺ = A⁺:")
        result = A_pinv @ A @ A_pinv
        print(f"   A⁺AA⁺ = \n{np.round(result, 4)}")
        print(f"   Equals A⁺: {np.allclose(result, A_pinv)}")
        
        print("\nd) Least squares Ax ≈ b = [1, 2, 3]ᵀ:")
        b = np.array([1, 2, 3])
        x = A_pinv @ b
        print(f"   x = A⁺b = {x}")
        print(f"   Ax = {A @ x}")
        print(f"   Residual ||Ax - b|| = {norm(A @ x - b):.4f}")
        print("   (Best possible since b has component orthogonal to col(A))")
    
    def exercise_6_condition_number(self):
        """
        Exercise 6: Condition Number
        
        Compare the condition numbers of:
        A = [[1, 0], [0, 1]]  (identity)
        B = [[1, 0], [0, 0.01]]  (nearly singular)
        C = [[1, 1], [1, 1.01]]  (nearly rank-deficient)
        
        For each:
        a) Compute singular values
        b) Compute condition number κ = σ₁/σ_r
        c) Interpret the results
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Condition Number")
        
        matrices = {
            'A (identity)': np.array([[1, 0], [0, 1]]),
            'B (scaled)': np.array([[1, 0], [0, 0.01]]),
            'C (nearly singular)': np.array([[1, 1], [1, 1.01]])
        }
        
        for name, M in matrices.items():
            print(f"\n{name}:")
            print(f"Matrix:\n{M}")
            
            U, s, Vt = svd(M)
            print(f"\na) Singular values: {s}")
            
            cond = s[0] / s[-1]
            print(f"b) Condition number κ = σ₁/σ_r = {cond:.4f}")
            
            print("c) Interpretation:")
            if cond < 10:
                print("   Well-conditioned - stable computations")
            elif cond < 1000:
                print("   Moderately conditioned - some precision loss possible")
            else:
                print("   Ill-conditioned - significant numerical instability")
    
    def exercise_7_variance_explained(self):
        """
        Exercise 7: Variance Explained
        
        Given a centered data matrix X with SVD X = UΣVᵀ:
        
        Singular values: σ = [10, 5, 2, 1]
        
        a) Compute total variance (sum of σᵢ²)
        b) Compute variance explained by each component
        c) How many components to capture 90% variance?
        d) How many for 99% variance?
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Variance Explained")
        
        s = np.array([10, 5, 2, 1])
        
        print(f"Singular values: {s}")
        
        print("\na) Total variance = Σσᵢ²:")
        variance_per_component = s**2
        total_variance = np.sum(variance_per_component)
        print(f"   σ² = {variance_per_component}")
        print(f"   Total = {total_variance}")
        
        print("\nb) Variance explained by each component:")
        pct_variance = 100 * variance_per_component / total_variance
        for i, (var, pct) in enumerate(zip(variance_per_component, pct_variance)):
            print(f"   PC{i+1}: {var} ({pct:.1f}%)")
        
        print("\nc) Cumulative variance:")
        cumsum = np.cumsum(pct_variance)
        for i, cum in enumerate(cumsum):
            print(f"   First {i+1} components: {cum:.1f}%")
        
        print("\n   Components for 90% variance:")
        k_90 = np.searchsorted(cumsum, 90) + 1
        print(f"   k = {k_90}")
        
        print("\nd) Components for 99% variance:")
        k_99 = np.searchsorted(cumsum, 99) + 1
        print(f"   k = {k_99}")
    
    def exercise_8_truncated_svd(self):
        """
        Exercise 8: Truncated SVD for Dimensionality Reduction
        
        Given data matrix X (100 samples × 10 features):
        Simulate data and:
        a) Compute full SVD
        b) Compute rank-3 approximation
        c) Compute compression ratio
        d) Compute reconstruction error
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Truncated SVD")
        
        # Generate structured data (inherently low-rank with noise)
        np.random.seed(42)
        n, d = 100, 10
        
        # Low-rank structure (rank 3) plus noise
        U_true = np.random.randn(n, 3)
        V_true = np.random.randn(3, d)
        X = U_true @ V_true + 0.1 * np.random.randn(n, d)
        
        print(f"Data X: {X.shape}")
        print(f"Original rank: {matrix_rank(X)}")
        
        print("\na) Full SVD:")
        U, s, Vt = svd(X, full_matrices=False)
        print(f"   Singular values: {np.round(s, 4)}")
        
        print("\nb) Rank-3 approximation:")
        k = 3
        X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        print(f"   X_3 shape: {X_k.shape}")
        
        print("\nc) Compression ratio:")
        original_storage = n * d
        compressed_storage = k * (n + d + 1)
        compression = 100 * (1 - compressed_storage / original_storage)
        print(f"   Original: {original_storage} values")
        print(f"   Compressed: {compressed_storage} values (U_k, Σ_k, V_k)")
        print(f"   Compression: {compression:.1f}%")
        
        print("\nd) Reconstruction error:")
        error = norm(X - X_k, 'fro') / norm(X, 'fro')
        print(f"   Relative error ||X - X_k||_F / ||X||_F = {error:.4f}")
        print(f"   Variance captured: {100*(1-error**2):.1f}%")
    
    def exercise_9_image_approximation(self):
        """
        Exercise 9: Image Approximation
        
        Create a 64×64 "image" and approximate it using SVD.
        
        a) Create image using sin/cos patterns
        b) Compute SVD and plot singular value decay
        c) Approximate with k = 5, 10, 20 components
        d) Compare storage requirements
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Image Approximation")
        
        # Create structured "image"
        m, n = 64, 64
        x = np.linspace(0, 2*np.pi, m)
        y = np.linspace(0, 2*np.pi, n)
        X, Y = np.meshgrid(x, y)
        
        image = np.sin(X) + np.cos(Y) + 0.5*np.sin(2*X)*np.cos(2*Y)
        
        print(f"a) Image size: {m}×{n}")
        print(f"   Full storage: {m*n} values")
        
        print("\nb) SVD and singular value decay:")
        U, s, Vt = svd(image)
        print(f"   First 10 singular values: {np.round(s[:10], 4)}")
        
        # Cumulative energy
        energy = np.cumsum(s**2) / np.sum(s**2)
        print(f"   Energy captured by first 5 components: {100*energy[4]:.1f}%")
        print(f"   Energy captured by first 10 components: {100*energy[9]:.1f}%")
        
        print("\nc) Approximations:")
        for k in [5, 10, 20]:
            image_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            error = norm(image - image_k, 'fro') / norm(image, 'fro')
            storage = k * (m + n + 1)
            compression = 100 * (1 - storage / (m * n))
            
            print(f"\n   k = {k}:")
            print(f"   Storage: {storage} values ({compression:.1f}% compression)")
            print(f"   Relative error: {error:.4f}")
            print(f"   PSNR-like: {-20*np.log10(error):.1f} dB")
    
    def exercise_10_latent_semantic_analysis(self):
        """
        Exercise 10: Latent Semantic Analysis (LSA)
        
        Given a term-document matrix:
        Terms: [cat, dog, fish, car, bike, train]
        Documents: [D1, D2, D3, D4]
        
        A = [[2, 1, 0, 0],   # cat
             [1, 2, 0, 0],   # dog
             [0, 1, 0, 0],   # fish
             [0, 0, 2, 1],   # car
             [0, 0, 1, 2],   # bike
             [0, 0, 1, 0]]   # train
        
        Use SVD to find:
        a) Latent topics (via truncated SVD with k=2)
        b) Document similarity in latent space
        c) Interpret the topics
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Latent Semantic Analysis")
        
        terms = ['cat', 'dog', 'fish', 'car', 'bike', 'train']
        docs = ['D1', 'D2', 'D3', 'D4']
        
        A = np.array([[2, 1, 0, 0],
                      [1, 2, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 2, 1],
                      [0, 0, 1, 2],
                      [0, 0, 1, 0]], dtype=float)
        
        print("Term-Document Matrix:")
        print(f"         {docs}")
        for i, term in enumerate(terms):
            print(f"{term:6s}: {A[i]}")
        
        print("\na) Truncated SVD with k=2:")
        U, s, Vt = svd(A)
        k = 2
        
        print(f"   Singular values: {np.round(s, 4)}")
        print(f"   Top {k}: {np.round(s[:k], 4)}")
        
        # Latent representation of documents
        doc_latent = np.diag(s[:k]) @ Vt[:k, :]
        
        print("\nb) Documents in latent space:")
        for i, doc in enumerate(docs):
            print(f"   {doc}: [{doc_latent[0, i]:.3f}, {doc_latent[1, i]:.3f}]")
        
        # Document similarity (cosine in latent space)
        print("\n   Document similarity (cosine):")
        from numpy.linalg import norm as np_norm
        for i in range(len(docs)):
            for j in range(i+1, len(docs)):
                v1 = doc_latent[:, i]
                v2 = doc_latent[:, j]
                sim = np.dot(v1, v2) / (np_norm(v1) * np_norm(v2))
                print(f"   {docs[i]}-{docs[j]}: {sim:.3f}")
        
        print("\nc) Topic interpretation:")
        term_topics = U[:, :k]
        print("   Term loadings on latent topics:")
        print(f"         Topic1  Topic2")
        for i, term in enumerate(terms):
            print(f"   {term:6s}: {term_topics[i, 0]:6.3f}  {term_topics[i, 1]:6.3f}")
        
        print("\n   Interpretation:")
        print("   Topic 1: Animals (cat, dog, fish have high values)")
        print("   Topic 2: Vehicles (car, bike, train have high values)")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = SVDExercises()
    
    print("SINGULAR VALUE DECOMPOSITION EXERCISES")
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
