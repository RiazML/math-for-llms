"""
Eigenvalues and Eigenvectors - Exercises
========================================
Practice problems for eigenvalue concepts.
"""

import numpy as np
from numpy.linalg import eig, det, inv, matrix_power


class EigenExercises:
    """Exercises for eigenvalues and eigenvectors."""
    
    def exercise_1_compute_eigenvalues(self):
        """
        Exercise 1: Compute Eigenvalues
        
        Find eigenvalues of the following matrices by hand,
        then verify with NumPy:
        
        a) A = [[3, 1], [0, 2]]
        b) B = [[0, -1], [1, 0]]
        c) C = [[5, 4], [1, 2]]
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Compute Eigenvalues")
        
        print("\na) A = [[3, 1], [0, 2]]")
        print("   Triangular matrix → eigenvalues on diagonal!")
        print("   λ₁ = 3, λ₂ = 2")
        A = np.array([[3, 1], [0, 2]])
        vals, _ = eig(A)
        print(f"   NumPy: {vals}")
        
        print("\nb) B = [[0, -1], [1, 0]] (rotation by 90°)")
        print("   det(B - λI) = λ² + 1 = 0")
        print("   λ = ±i (complex!)")
        B = np.array([[0, -1], [1, 0]])
        vals, _ = eig(B)
        print(f"   NumPy: {vals}")
        
        print("\nc) C = [[5, 4], [1, 2]]")
        print("   det(C - λI) = (5-λ)(2-λ) - 4 = λ² - 7λ + 6 = 0")
        print("   (λ - 6)(λ - 1) = 0")
        print("   λ₁ = 6, λ₂ = 1")
        C = np.array([[5, 4], [1, 2]])
        vals, _ = eig(C)
        print(f"   NumPy: {vals}")
    
    def exercise_2_compute_eigenvectors(self):
        """
        Exercise 2: Compute Eigenvectors
        
        For A = [[4, 2], [1, 3]]:
        a) Find eigenvalues
        b) Find eigenvectors for each eigenvalue
        c) Verify Av = λv for each pair
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Compute Eigenvectors")
        
        A = np.array([[4, 2], [1, 3]])
        print(f"A = \n{A}")
        
        print("\na) Finding eigenvalues:")
        print("   det(A - λI) = (4-λ)(3-λ) - 2 = λ² - 7λ + 10 = 0")
        print("   (λ - 5)(λ - 2) = 0")
        print("   λ₁ = 5, λ₂ = 2")
        
        print("\nb) Finding eigenvectors:")
        
        print("\n   For λ₁ = 5:")
        print("   (A - 5I)v = 0")
        print("   [[-1, 2], [1, -2]]v = 0")
        print("   -v₁ + 2v₂ = 0 → v₁ = 2v₂")
        print("   v₁ = [2, 1]ᵀ (or any scalar multiple)")
        
        print("\n   For λ₂ = 2:")
        print("   (A - 2I)v = 0")
        print("   [[2, 2], [1, 1]]v = 0")
        print("   2v₁ + 2v₂ = 0 → v₁ = -v₂")
        print("   v₂ = [1, -1]ᵀ (or any scalar multiple)")
        
        print("\nc) Verification:")
        eigenvalues, eigenvectors = eig(A)
        for i in range(2):
            lam = eigenvalues[i]
            v = eigenvectors[:, i]
            Av = A @ v
            lam_v = lam * v
            print(f"\n   λ = {lam.real:.1f}:")
            print(f"   v = {v.real}")
            print(f"   Av = {Av.real}")
            print(f"   λv = {lam_v.real}")
            print(f"   Av = λv: {np.allclose(Av, lam_v)}")
    
    def exercise_3_trace_determinant(self):
        """
        Exercise 3: Verify Trace and Determinant Relations
        
        For A = [[1, 2, 3], [0, 4, 5], [0, 0, 6]]:
        a) Find eigenvalues (hint: triangular matrix)
        b) Verify tr(A) = sum of eigenvalues
        c) Verify det(A) = product of eigenvalues
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Trace and Determinant")
        
        A = np.array([[1, 2, 3],
                      [0, 4, 5],
                      [0, 0, 6]])
        
        print(f"A (upper triangular):\n{A}")
        
        print("\na) Eigenvalues of triangular matrix = diagonal entries")
        print("   λ₁ = 1, λ₂ = 4, λ₃ = 6")
        
        eigenvalues = np.array([1, 4, 6])
        
        print("\nb) Trace verification:")
        trace_A = np.trace(A)
        sum_lambda = np.sum(eigenvalues)
        print(f"   tr(A) = 1 + 4 + 6 = {trace_A}")
        print(f"   Σλᵢ = {sum_lambda}")
        print(f"   Equal: {trace_A == sum_lambda}")
        
        print("\nc) Determinant verification:")
        det_A = det(A)
        prod_lambda = np.prod(eigenvalues)
        print(f"   det(A) = 1 × 4 × 6 = {det_A:.0f}")
        print(f"   Πλᵢ = {prod_lambda}")
        print(f"   Equal: {det_A == prod_lambda}")
    
    def exercise_4_diagonalization(self):
        """
        Exercise 4: Diagonalization
        
        For A = [[3, 1], [0, 2]]:
        a) Find P and D such that A = PDP⁻¹
        b) Verify the decomposition
        c) Use diagonalization to compute A⁴
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Diagonalization")
        
        A = np.array([[3, 1], [0, 2]])
        print(f"A = \n{A}")
        
        print("\na) Finding P and D:")
        eigenvalues, eigenvectors = eig(A)
        D = np.diag(eigenvalues)
        P = eigenvectors
        
        print(f"   Eigenvalues: λ₁ = {eigenvalues[0]:.1f}, λ₂ = {eigenvalues[1]:.1f}")
        print(f"\n   D = diag(λ₁, λ₂) = \n{D.real}")
        print(f"\n   P (eigenvectors as columns) = \n{P.real}")
        
        print("\nb) Verification A = PDP⁻¹:")
        P_inv = inv(P)
        reconstructed = P @ D @ P_inv
        print(f"   PDP⁻¹ = \n{reconstructed.real}")
        print(f"   Equal to A: {np.allclose(A, reconstructed)}")
        
        print("\nc) Computing A⁴ via diagonalization:")
        D4 = np.diag(eigenvalues ** 4)
        A4_diag = P @ D4 @ P_inv
        A4_direct = matrix_power(A, 4)
        
        print(f"   D⁴ = diag(3⁴, 2⁴) = diag(81, 16)")
        print(f"   A⁴ = PD⁴P⁻¹ = \n{A4_diag.real}")
        print(f"   Direct A⁴ = \n{A4_direct}")
        print(f"   Match: {np.allclose(A4_diag, A4_direct)}")
    
    def exercise_5_symmetric_matrix(self):
        """
        Exercise 5: Symmetric Matrix Properties
        
        For A = [[2, 1], [1, 2]]:
        a) Verify eigenvalues are real
        b) Find eigenvectors and verify orthogonality
        c) Write the spectral decomposition A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Symmetric Matrix")
        
        A = np.array([[2, 1], [1, 2]])
        print(f"Symmetric A = \n{A}")
        
        eigenvalues, eigenvectors = eig(A)
        
        print("\na) Eigenvalues are real:")
        print(f"   λ₁ = {eigenvalues[0].real}")
        print(f"   λ₂ = {eigenvalues[1].real}")
        print(f"   Both real: {all(np.isreal(eigenvalues))}")
        
        print("\nb) Eigenvector orthogonality:")
        v1 = eigenvectors[:, 0].real
        v2 = eigenvectors[:, 1].real
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        
        print(f"   v₁ = {v1}")
        print(f"   v₂ = {v2}")
        print(f"   v₁·v₂ = {np.dot(v1, v2):.6f}")
        print(f"   Orthogonal: {np.isclose(np.dot(v1, v2), 0)}")
        
        print("\nc) Spectral decomposition:")
        lam1, lam2 = eigenvalues.real
        term1 = lam1 * np.outer(v1, v1)
        term2 = lam2 * np.outer(v2, v2)
        
        print(f"\n   λ₁v₁v₁ᵀ = {lam1:.1f} × \n{np.outer(v1, v1)}")
        print(f"   = \n{term1}")
        
        print(f"\n   λ₂v₂v₂ᵀ = {lam2:.1f} × \n{np.outer(v2, v2)}")
        print(f"   = \n{term2}")
        
        print(f"\n   A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ = \n{term1 + term2}")
        print(f"   Matches: {np.allclose(A, term1 + term2)}")
    
    def exercise_6_eigenvalue_transformations(self):
        """
        Exercise 6: Eigenvalue Transformations
        
        Given A = [[4, 0], [1, 3]] with eigenvalues λ₁=4, λ₂=3:
        
        Without computing directly, find eigenvalues of:
        a) A²
        b) A⁻¹
        c) A + 2I
        d) 5A
        Then verify with NumPy.
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Eigenvalue Transformations")
        
        A = np.array([[4, 0], [1, 3]])
        print(f"A = \n{A}")
        print("Given: eigenvalues λ₁=4, λ₂=3")
        
        print("\na) A²: eigenvalues are λ²")
        print("   Expected: 16, 9")
        vals, _ = eig(A @ A)
        print(f"   NumPy: {sorted(vals.real, reverse=True)}")
        
        print("\nb) A⁻¹: eigenvalues are 1/λ")
        print("   Expected: 1/4=0.25, 1/3≈0.333")
        vals, _ = eig(inv(A))
        print(f"   NumPy: {sorted(vals.real, reverse=True)}")
        
        print("\nc) A + 2I: eigenvalues are λ+2")
        print("   Expected: 6, 5")
        vals, _ = eig(A + 2*np.eye(2))
        print(f"   NumPy: {sorted(vals.real, reverse=True)}")
        
        print("\nd) 5A: eigenvalues are 5λ")
        print("   Expected: 20, 15")
        vals, _ = eig(5*A)
        print(f"   NumPy: {sorted(vals.real, reverse=True)}")
    
    def exercise_7_power_method(self):
        """
        Exercise 7: Power Method Implementation
        
        Implement the power method to find the dominant eigenvalue
        and eigenvector of A = [[2, 1], [1, 2]].
        
        Show convergence over 10 iterations.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Power Method")
        
        A = np.array([[2, 1], [1, 2]])
        print(f"A = \n{A}")
        
        # True values
        true_vals, true_vecs = eig(A)
        dom_idx = np.argmax(np.abs(true_vals))
        true_lambda = true_vals[dom_idx].real
        true_vec = true_vecs[:, dom_idx].real
        true_vec = true_vec / np.linalg.norm(true_vec)
        
        print(f"\nTrue dominant eigenvalue: {true_lambda}")
        print(f"True eigenvector: {true_vec}")
        
        # Power method
        np.random.seed(0)
        v = np.array([1.0, 0.0])  # Starting vector
        
        print(f"\nPower Method (starting with v = {v}):")
        print(f"{'Iter':>4} {'λ estimate':>12} {'v':>20} {'Error':>12}")
        
        for k in range(10):
            w = A @ v
            lambda_est = np.dot(w, v) / np.dot(v, v)
            v = w / np.linalg.norm(w)
            error = abs(lambda_est - true_lambda)
            print(f"{k+1:4d} {lambda_est:12.8f} [{v[0]:8.5f}, {v[1]:8.5f}] {error:12.8f}")
        
        print(f"\nFinal estimate: λ ≈ {lambda_est:.6f}, v ≈ {v}")
    
    def exercise_8_markov_chain(self):
        """
        Exercise 8: Markov Chain Stationary Distribution
        
        A mouse can be in room A or room B.
        - From A: 70% stay in A, 30% go to B
        - From B: 40% go to A, 60% stay in B
        
        a) Write the transition matrix P
        b) Find the stationary distribution π where πP = π
        c) Verify π sums to 1 and πP = π
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Markov Chain")
        
        print("\na) Transition matrix P:")
        P = np.array([[0.7, 0.3],   # From A
                      [0.4, 0.6]])  # From B
        print(f"   P = \n{P}")
        print("   P[i,j] = P(go to j | currently in i)")
        
        print("\nb) Finding stationary distribution:")
        print("   πP = π means π is left eigenvector of P with λ=1")
        print("   Equivalently, Pᵀπᵀ = πᵀ, so π is eigenvector of Pᵀ with λ=1")
        
        eigenvalues, eigenvectors = eig(P.T)
        print(f"\n   Eigenvalues of Pᵀ: {eigenvalues}")
        
        # Find λ = 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = eigenvectors[:, idx].real
        pi = pi / np.sum(pi)  # Normalize to probability
        
        print(f"\n   Stationary distribution π = {pi}")
        print(f"   π(A) = {pi[0]:.4f}")
        print(f"   π(B) = {pi[1]:.4f}")
        
        print("\nc) Verification:")
        print(f"   Sum of π: {np.sum(pi):.4f}")
        pi_P = pi @ P
        print(f"   πP = {pi_P}")
        print(f"   π = πP: {np.allclose(pi, pi_P)}")
        
        # Alternative: solve linear system
        print("\n   Alternative (solving πP = π directly):")
        print("   π₁(0.7) + π₂(0.4) = π₁  →  -0.3π₁ + 0.4π₂ = 0")
        print("   With π₁ + π₂ = 1:")
        print("   0.4π₂ = 0.3π₁ = 0.3(1-π₂)  →  0.7π₂ = 0.3  →  π₂ = 3/7")
        print(f"   π = [4/7, 3/7] = [{4/7:.4f}, {3/7:.4f}]")
    
    def exercise_9_positive_definite(self):
        """
        Exercise 9: Positive Definite Check
        
        Determine if the following matrices are positive definite:
        a) A = [[4, 2], [2, 1]]
        b) B = [[3, 1], [1, 3]]
        c) C = [[1, 2], [2, 1]]
        
        Use the eigenvalue criterion: all eigenvalues > 0.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Positive Definite Check")
        
        matrices = {
            'A': np.array([[4, 2], [2, 1]]),
            'B': np.array([[3, 1], [1, 3]]),
            'C': np.array([[1, 2], [2, 1]])
        }
        
        for name, M in matrices.items():
            print(f"\n{name}) Matrix:\n{M}")
            eigenvalues, _ = eig(M)
            eigenvalues = sorted(eigenvalues.real)
            print(f"   Eigenvalues: {eigenvalues}")
            is_pd = all(lam > 0 for lam in eigenvalues)
            print(f"   All positive: {is_pd}")
            print(f"   Positive definite: {is_pd}")
            
            if not is_pd:
                # Find x such that x^T A x <= 0
                _, vecs = eig(M)
                idx = np.argmin(eigenvalues)
                x = vecs[:, idx].real
                quad = x.T @ M @ x
                print(f"   Counterexample: x = {x.real}")
                print(f"   xᵀMx = {quad.real:.4f}")
    
    def exercise_10_spectral_norm(self):
        """
        Exercise 10: Spectral Norm and Condition Number
        
        For A = [[3, 1], [0, 2]]:
        a) Find singular values (sqrt of eigenvalues of AᵀA)
        b) Compute spectral norm ||A||₂ = max singular value
        c) Compute condition number κ(A) = σ_max / σ_min
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Spectral Norm")
        
        A = np.array([[3, 1], [0, 2]])
        print(f"A = \n{A}")
        
        print("\na) Finding singular values:")
        print("   Singular values = sqrt(eigenvalues of AᵀA)")
        
        AtA = A.T @ A
        print(f"\n   AᵀA = \n{AtA}")
        
        eigenvalues, _ = eig(AtA)
        eigenvalues = sorted(eigenvalues.real, reverse=True)
        print(f"   Eigenvalues of AᵀA: {eigenvalues}")
        
        singular_values = np.sqrt(eigenvalues)
        print(f"   Singular values: σ₁ = {singular_values[0]:.4f}, σ₂ = {singular_values[1]:.4f}")
        
        # Verify with SVD
        _, s, _ = np.linalg.svd(A)
        print(f"   NumPy SVD: {s}")
        
        print("\nb) Spectral norm:")
        spectral_norm = singular_values[0]
        print(f"   ||A||₂ = σ_max = {spectral_norm:.4f}")
        print(f"   NumPy: {np.linalg.norm(A, 2):.4f}")
        
        print("\nc) Condition number:")
        condition = singular_values[0] / singular_values[1]
        print(f"   κ(A) = σ_max / σ_min = {condition:.4f}")
        print(f"   NumPy: {np.linalg.cond(A):.4f}")
        print("\n   Interpretation: κ(A) measures sensitivity to perturbations")
        print("   κ(A) ≈ 1: well-conditioned, κ(A) >> 1: ill-conditioned")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = EigenExercises()
    
    print("EIGENVALUES AND EIGENVECTORS EXERCISES")
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
