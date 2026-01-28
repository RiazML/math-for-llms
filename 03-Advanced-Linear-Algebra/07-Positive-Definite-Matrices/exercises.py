"""
Positive Definite Matrices - Exercises
======================================
Practice problems for positive definite matrix concepts.
"""

import numpy as np
from numpy.linalg import eigvalsh, cholesky, inv, det


class PositiveDefiniteExercises:
    """Exercises for positive definite matrices."""
    
    def exercise_1_pd_test(self):
        """
        Exercise 1: Test Positive Definiteness
        
        Determine if each matrix is PD, PSD, or indefinite:
        a) A = [[2, 1], [1, 2]]
        b) B = [[1, 2], [2, 1]]
        c) C = [[1, 1], [1, 1]]
        d) D = [[4, 2], [2, 1]]
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Test Positive Definiteness")
        
        matrices = {
            'A': np.array([[2, 1], [1, 2]]),
            'B': np.array([[1, 2], [2, 1]]),
            'C': np.array([[1, 1], [1, 1]]),
            'D': np.array([[4, 2], [2, 1]])
        }
        
        for name, M in matrices.items():
            print(f"\n{name}) {name} = \n{M}")
            eigenvalues = eigvalsh(M)
            print(f"   Eigenvalues: {np.round(eigenvalues, 4)}")
            
            if all(eigenvalues > 1e-10):
                print(f"   → Positive Definite (all λ > 0)")
            elif all(eigenvalues >= -1e-10):
                print(f"   → Positive Semi-Definite (all λ ≥ 0)")
            else:
                print(f"   → Indefinite (mixed signs)")
    
    def exercise_2_sylvester(self):
        """
        Exercise 2: Sylvester's Criterion
        
        Use leading principal minors to test if PD:
        A = [[4, 2, 1],
             [2, 5, 2],
             [1, 2, 6]]
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Sylvester's Criterion")
        
        A = np.array([[4, 2, 1],
                      [2, 5, 2],
                      [1, 2, 6]])
        
        print(f"A = \n{A}")
        
        print("\nLeading principal minors:")
        
        # 1x1 minor
        m1 = A[0, 0]
        print(f"  M₁ = a₁₁ = {m1} > 0? {m1 > 0} ✓")
        
        # 2x2 minor
        m2 = det(A[:2, :2])
        print(f"  M₂ = det(A[:2,:2]) = 4×5 - 2×2 = {m2} > 0? {m2 > 0} ✓")
        
        # 3x3 minor (full determinant)
        m3 = det(A)
        print(f"  M₃ = det(A) = {m3:.4f} > 0? {m3 > 0} ✓")
        
        print("\nAll minors > 0 → A is positive definite!")
        
        # Verify with eigenvalues
        print(f"\nVerification: eigenvalues = {np.round(eigvalsh(A), 4)}")
    
    def exercise_3_cholesky(self):
        """
        Exercise 3: Cholesky Decomposition
        
        Find the Cholesky decomposition of:
        A = [[9, 6], [6, 8]]
        
        Verify: A = L L^T
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Cholesky Decomposition")
        
        A = np.array([[9, 6],
                      [6, 8]], dtype=float)
        
        print(f"A = \n{A}")
        
        # Manual computation
        print("\n--- Manual Cholesky ---")
        L = np.zeros((2, 2))
        
        # L[0,0]
        L[0, 0] = np.sqrt(A[0, 0])
        print(f"L[0,0] = √A[0,0] = √9 = {L[0,0]}")
        
        # L[1,0]
        L[1, 0] = A[1, 0] / L[0, 0]
        print(f"L[1,0] = A[1,0]/L[0,0] = 6/3 = {L[1,0]}")
        
        # L[1,1]
        L[1, 1] = np.sqrt(A[1, 1] - L[1, 0]**2)
        print(f"L[1,1] = √(A[1,1] - L[1,0]²) = √(8 - 4) = {L[1,1]}")
        
        print(f"\nL = \n{L}")
        
        print(f"\nVerification: L @ L^T = \n{L @ L.T}")
        print(f"Equals A? {np.allclose(L @ L.T, A)}")
    
    def exercise_4_pd_inverse(self):
        """
        Exercise 4: Inverse of PD Matrix
        
        Show that if A is PD, then A⁻¹ is also PD.
        
        Test with A = [[2, 1], [1, 2]]
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Inverse of PD Matrix")
        
        A = np.array([[2, 1],
                      [1, 2]])
        
        print(f"A = \n{A}")
        print(f"Eigenvalues of A: {eigvalsh(A)}")
        print("A is PD ✓")
        
        A_inv = inv(A)
        print(f"\nA⁻¹ = \n{np.round(A_inv, 4)}")
        print(f"Eigenvalues of A⁻¹: {np.round(eigvalsh(A_inv), 4)}")
        print("A⁻¹ is also PD ✓")
        
        print("\n--- Proof ---")
        print("If A has eigenvalues λᵢ > 0,")
        print("then A⁻¹ has eigenvalues 1/λᵢ > 0.")
        print("Therefore A⁻¹ is also PD.")
    
    def exercise_5_pd_sum(self):
        """
        Exercise 5: Sum of PD Matrices
        
        Show that if A and B are PD, then A + B is also PD.
        
        Test with:
        A = [[2, 0], [0, 3]]
        B = [[1, 0.5], [0.5, 1]]
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Sum of PD Matrices")
        
        A = np.array([[2, 0], [0, 3]])
        B = np.array([[1, 0.5], [0.5, 1]])
        
        print(f"A = \n{A}")
        print(f"Eigenvalues of A: {eigvalsh(A)} > 0 ✓")
        
        print(f"\nB = \n{B}")
        print(f"Eigenvalues of B: {eigvalsh(B)} > 0 ✓")
        
        C = A + B
        print(f"\nC = A + B = \n{C}")
        print(f"Eigenvalues of C: {np.round(eigvalsh(C), 4)} > 0 ✓")
        
        print("\n--- Proof ---")
        print("For any x ≠ 0:")
        print("x^T(A+B)x = x^T A x + x^T B x")
        print("         > 0 + 0 = 0")
        print("Therefore A + B is PD.")
    
    def exercise_6_ata_psd(self):
        """
        Exercise 6: A^T A is Always PSD
        
        Prove that A^T A is always positive semi-definite
        for any matrix A.
        
        Test with A = [[1, 2], [3, 4], [5, 6]]
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: A^T A is Always PSD")
        
        A = np.array([[1, 2],
                      [3, 4],
                      [5, 6]])
        
        print(f"A = \n{A}")
        
        ATA = A.T @ A
        print(f"\nA^T A = \n{ATA}")
        print(f"Eigenvalues: {np.round(eigvalsh(ATA), 4)}")
        print("All ≥ 0 → A^T A is PSD ✓")
        
        print("\n--- Proof ---")
        print("For any x:")
        print("x^T (A^T A) x = (Ax)^T (Ax) = ||Ax||² ≥ 0")
        print("Therefore A^T A is always PSD.")
        
        print("\nWhen is A^T A strictly PD?")
        print("When A has full column rank (Ax = 0 only if x = 0)")
    
    def exercise_7_regularization(self):
        """
        Exercise 7: Regularization Makes PD
        
        For A = [[1, 1], [1, 1]] (singular):
        Find the smallest λ > 0 such that A + λI is PD.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Regularization Makes PD")
        
        A = np.array([[1, 1], [1, 1]], dtype=float)
        
        print(f"A = \n{A}")
        eigenvalues = eigvalsh(A)
        print(f"Eigenvalues: {eigenvalues}")
        print("Minimum eigenvalue: 0 → A is PSD (not PD)")
        
        print("\n--- Analysis ---")
        print("A + λI has eigenvalues: λ₁ + λ, λ₂ + λ")
        print(f"= {eigenvalues[0]} + λ, {eigenvalues[1]} + λ")
        print(f"= λ, {eigenvalues[1]} + λ")
        
        print("\nFor A + λI to be PD, we need:")
        print("λ > 0 (so smallest eigenvalue > 0)")
        
        print("\nAny λ > 0 works!")
        
        for lam in [0.01, 0.1, 1.0]:
            A_reg = A + lam * np.eye(2)
            eigs = eigvalsh(A_reg)
            print(f"\nλ = {lam}: eigenvalues = {np.round(eigs, 4)}")
    
    def exercise_8_2x2_criterion(self):
        """
        Exercise 8: 2×2 PD Criterion
        
        For A = [[a, b], [b, c]], prove A is PD iff:
        a > 0 AND det(A) = ac - b² > 0
        
        Apply to A = [[3, k], [k, 3]]. Find range of k.
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: 2×2 PD Criterion")
        
        print("For A = [[a, b], [b, c]]:")
        print("Eigenvalues: λ = (a+c ± √((a-c)² + 4b²)) / 2")
        print("\nA is PD iff both eigenvalues > 0")
        print("This is equivalent to:")
        print("  1) trace(A) = a + c > 0")
        print("  2) det(A) = ac - b² > 0")
        print("Which simplifies to: a > 0 AND ac - b² > 0")
        
        print("\n--- Application to A = [[3, k], [k, 3]] ---")
        print("Here a = 3 > 0 ✓")
        print("Need: det(A) = 9 - k² > 0")
        print("      k² < 9")
        print("      -3 < k < 3")
        
        print("\nRange: k ∈ (-3, 3)")
        
        # Verify
        print("\nVerification:")
        for k in [-4, -2, 0, 2, 4]:
            A = np.array([[3, k], [k, 3]], dtype=float)
            eigs = eigvalsh(A)
            status = "PD" if all(eigs > 0) else "Not PD"
            print(f"  k = {k:2d}: eigenvalues = {np.round(eigs, 2)}, {status}")
    
    def exercise_9_covariance_estimation(self):
        """
        Exercise 9: Ensuring Valid Covariance
        
        An estimated covariance matrix might not be PD due to noise.
        
        Given: Σ_est = [[1.0, 0.9, 0.8],
                        [0.9, 1.0, 0.85],
                        [0.8, 0.85, 1.0]]
        
        Make it valid (PD) by fixing negative eigenvalues.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Ensuring Valid Covariance")
        
        Sigma_est = np.array([[1.0, 0.9, 0.8],
                              [0.9, 1.0, 0.85],
                              [0.8, 0.85, 1.0]])
        
        print(f"Σ_est = \n{Sigma_est}")
        
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_est)
        print(f"\nEigenvalues: {np.round(eigenvalues, 6)}")
        
        if any(eigenvalues < 0):
            print("Negative eigenvalue found → Not PD!")
            
            # Fix by projecting to PD cone
            eigenvalues_fixed = np.maximum(eigenvalues, 1e-6)
            Sigma_fixed = eigenvectors @ np.diag(eigenvalues_fixed) @ eigenvectors.T
            
            print(f"\nFixed eigenvalues: {np.round(eigenvalues_fixed, 6)}")
            print(f"\nΣ_fixed = \n{np.round(Sigma_fixed, 4)}")
            
            # Verify
            try:
                cholesky(Sigma_fixed)
                print("\nCholesky succeeds → Σ_fixed is PD ✓")
            except:
                print("\nStill not PD!")
        else:
            print("Already PD!")
    
    def exercise_10_quadratic_optimization(self):
        """
        Exercise 10: Quadratic Optimization
        
        For f(x) = 0.5 x^T A x - b^T x where:
        A = [[4, 2], [2, 2]]
        b = [4, 2]
        
        a) Verify A is PD
        b) Find the unique minimum x*
        c) Compute f(x*)
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Quadratic Optimization")
        
        A = np.array([[4, 2],
                      [2, 2]], dtype=float)
        b = np.array([4, 2], dtype=float)
        
        print(f"A = \n{A}")
        print(f"b = {b}")
        
        print("\na) Verify A is PD:")
        eigenvalues = eigvalsh(A)
        print(f"   Eigenvalues: {np.round(eigenvalues, 4)}")
        print(f"   All > 0? {all(eigenvalues > 0)} ✓")
        print("   A is PD → f has unique minimum")
        
        print("\nb) Find minimum:")
        print("   Set ∇f = Ax - b = 0")
        print("   x* = A⁻¹ b")
        x_star = inv(A) @ b
        print(f"   x* = {x_star}")
        
        print("\nc) Compute f(x*):")
        f_star = 0.5 * x_star.T @ A @ x_star - b.T @ x_star
        print(f"   f(x*) = 0.5 x*^T A x* - b^T x*")
        print(f"        = {f_star}")
        
        # Alternative formula
        f_star_alt = -0.5 * b.T @ inv(A) @ b
        print(f"   Also: f(x*) = -0.5 b^T A⁻¹ b = {f_star_alt}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = PositiveDefiniteExercises()
    
    print("POSITIVE DEFINITE MATRICES EXERCISES")
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
