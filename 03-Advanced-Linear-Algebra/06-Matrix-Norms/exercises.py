"""
Matrix Norms and Condition Numbers - Exercises
==============================================
Practice problems for matrix norm concepts.
"""

import numpy as np
from numpy.linalg import norm, cond, svd, inv


class MatrixNormExercises:
    """Exercises for matrix norms and condition numbers."""
    
    def exercise_1_vector_norms(self):
        """
        Exercise 1: Compute Vector Norms
        
        For x = [3, -4, 0, 5], compute:
        a) L¹ norm
        b) L² norm
        c) L∞ norm
        d) L³ norm
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Compute Vector Norms")
        
        x = np.array([3, -4, 0, 5])
        print(f"x = {x}")
        
        print(f"\na) L¹ norm: ||x||₁ = Σ|xᵢ|")
        l1 = np.sum(np.abs(x))
        print(f"   = |3| + |-4| + |0| + |5| = {l1}")
        
        print(f"\nb) L² norm: ||x||₂ = √(Σxᵢ²)")
        l2 = np.sqrt(np.sum(x**2))
        print(f"   = √(9 + 16 + 0 + 25) = √50 = {l2:.4f}")
        
        print(f"\nc) L∞ norm: ||x||_∞ = max|xᵢ|")
        linf = np.max(np.abs(x))
        print(f"   = max(3, 4, 0, 5) = {linf}")
        
        print(f"\nd) L³ norm: ||x||₃ = (Σ|xᵢ|³)^(1/3)")
        l3 = np.sum(np.abs(x)**3)**(1/3)
        print(f"   = (27 + 64 + 0 + 125)^(1/3) = 216^(1/3) = {l3:.4f}")
    
    def exercise_2_matrix_norms(self):
        """
        Exercise 2: Compute Matrix Norms
        
        For A = [[1, 2], [3, 4]], compute:
        a) ||A||₁ (max column sum)
        b) ||A||_∞ (max row sum)
        c) ||A||_F (Frobenius)
        d) ||A||₂ (spectral)
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Compute Matrix Norms")
        
        A = np.array([[1, 2],
                      [3, 4]])
        print(f"A = \n{A}")
        
        print(f"\na) ||A||₁ (max column sum):")
        print(f"   Column 1: |1| + |3| = 4")
        print(f"   Column 2: |2| + |4| = 6")
        print(f"   ||A||₁ = max(4, 6) = {norm(A, 1)}")
        
        print(f"\nb) ||A||_∞ (max row sum):")
        print(f"   Row 1: |1| + |2| = 3")
        print(f"   Row 2: |3| + |4| = 7")
        print(f"   ||A||_∞ = max(3, 7) = {norm(A, np.inf)}")
        
        print(f"\nc) ||A||_F (Frobenius):")
        print(f"   = √(1² + 2² + 3² + 4²)")
        print(f"   = √(1 + 4 + 9 + 16) = √30 = {norm(A, 'fro'):.4f}")
        
        print(f"\nd) ||A||₂ (spectral/2-norm):")
        _, S, _ = svd(A)
        print(f"   Singular values: {np.round(S, 4)}")
        print(f"   ||A||₂ = σ_max = {norm(A, 2):.4f}")
    
    def exercise_3_condition_number(self):
        """
        Exercise 3: Condition Number
        
        Compute the condition number for:
        a) A = [[2, 0], [0, 2]]
        b) B = [[1, 0], [0, 0.001]]
        c) C = [[1, 1], [1, 1.0001]]
        
        Which is most ill-conditioned?
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Condition Number")
        
        A = np.array([[2, 0], [0, 2]])
        B = np.array([[1, 0], [0, 0.001]])
        C = np.array([[1, 1], [1, 1.0001]])
        
        for name, M in [('A', A), ('B', B), ('C', C)]:
            print(f"\n{name}) {name} = \n{M}")
            U, S, Vt = svd(M)
            print(f"   Singular values: {S}")
            print(f"   κ({name}) = σ_max/σ_min = {S[0]:.4f}/{S[1]:.6f} = {cond(M):.2f}")
        
        print("\nMost ill-conditioned: C (κ ≈ 40000)")
        print("Reason: Nearly singular (rows almost identical)")
    
    def exercise_4_frobenius_trace(self):
        """
        Exercise 4: Frobenius Norm and Trace
        
        Show that ||A||_F = √(tr(A^T A)) for:
        A = [[1, 2, 3], [4, 5, 6]]
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Frobenius Norm and Trace")
        
        A = np.array([[1, 2, 3],
                      [4, 5, 6]])
        
        print(f"A = \n{A}")
        
        # Method 1: Direct computation
        fro_direct = np.sqrt(np.sum(A**2))
        print(f"\nMethod 1: ||A||_F = √(Σaᵢⱼ²)")
        print(f"   = √(1 + 4 + 9 + 16 + 25 + 36)")
        print(f"   = √91 = {fro_direct:.4f}")
        
        # Method 2: Using trace
        ATA = A.T @ A
        print(f"\nMethod 2: ||A||_F = √(tr(A^T A))")
        print(f"A^T A = \n{ATA}")
        trace_ATA = np.trace(ATA)
        print(f"tr(A^T A) = {trace_ATA}")
        fro_trace = np.sqrt(trace_ATA)
        print(f"√(tr(A^T A)) = {fro_trace:.4f}")
        
        print(f"\nBoth methods give: {fro_direct:.4f} ✓")
    
    def exercise_5_singular_value_norms(self):
        """
        Exercise 5: Norms from Singular Values
        
        For A with singular values σ = [5, 3, 1], find:
        a) ||A||₂ (spectral)
        b) ||A||_F (Frobenius)
        c) ||A||_* (nuclear)
        d) rank(A)
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Norms from Singular Values")
        
        S = np.array([5, 3, 1])
        print(f"Singular values: σ = {S}")
        
        print(f"\na) ||A||₂ = σ_max = {S[0]}")
        
        print(f"\nb) ||A||_F = √(Σσᵢ²)")
        print(f"   = √({S[0]**2} + {S[1]**2} + {S[2]**2})")
        print(f"   = √{np.sum(S**2)} = {np.sqrt(np.sum(S**2)):.4f}")
        
        print(f"\nc) ||A||_* = Σσᵢ = {S[0]} + {S[1]} + {S[2]} = {np.sum(S)}")
        
        print(f"\nd) rank(A) = number of non-zero singular values = {len(S[S > 0])}")
    
    def exercise_6_submultiplicativity(self):
        """
        Exercise 6: Verify Submultiplicativity
        
        For A = [[1, 2], [0, 1]] and B = [[2, 1], [1, 0]]:
        Verify ||AB||_F ≤ ||A||_F ||B||_F
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Verify Submultiplicativity")
        
        A = np.array([[1, 2], [0, 1]])
        B = np.array([[2, 1], [1, 0]])
        
        print(f"A = \n{A}")
        print(f"B = \n{B}")
        
        AB = A @ B
        print(f"\nAB = \n{AB}")
        
        norm_AB = norm(AB, 'fro')
        norm_A = norm(A, 'fro')
        norm_B = norm(B, 'fro')
        
        print(f"\n||AB||_F = {norm_AB:.4f}")
        print(f"||A||_F = {norm_A:.4f}")
        print(f"||B||_F = {norm_B:.4f}")
        print(f"||A||_F × ||B||_F = {norm_A * norm_B:.4f}")
        
        print(f"\n||AB||_F ≤ ||A||_F ||B||_F")
        print(f"{norm_AB:.4f} ≤ {norm_A * norm_B:.4f} ✓")
    
    def exercise_7_regularization(self):
        """
        Exercise 7: Regularization Effect
        
        For X^T X = [[1, 0.9], [0.9, 1]]:
        a) Compute κ(X^T X)
        b) Compute κ(X^T X + λI) for λ = 0.1
        c) How much does regularization improve conditioning?
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Regularization Effect")
        
        XTX = np.array([[1, 0.9],
                        [0.9, 1]])
        
        print(f"X^T X = \n{XTX}")
        
        _, S, _ = svd(XTX)
        print(f"\na) Singular values: {S}")
        print(f"   κ(X^T X) = {cond(XTX):.2f}")
        
        lam = 0.1
        XTX_reg = XTX + lam * np.eye(2)
        print(f"\nb) X^T X + λI (λ = {lam}):")
        print(f"   = \n{XTX_reg}")
        _, S_reg, _ = svd(XTX_reg)
        print(f"   Singular values: {np.round(S_reg, 4)}")
        print(f"   κ(X^T X + λI) = {cond(XTX_reg):.2f}")
        
        print(f"\nc) Improvement:")
        print(f"   Original: κ = {cond(XTX):.2f}")
        print(f"   Regularized: κ = {cond(XTX_reg):.2f}")
        print(f"   Factor: {cond(XTX) / cond(XTX_reg):.1f}x better")
    
    def exercise_8_error_bound(self):
        """
        Exercise 8: Error Bound
        
        For Ax = b with κ(A) = 100:
        If b has 0.1% error, what's the maximum possible error in x?
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Error Bound")
        
        kappa = 100
        b_error = 0.001  # 0.1%
        
        print(f"κ(A) = {kappa}")
        print(f"Relative error in b: {b_error * 100}%")
        
        print(f"\nError bound formula:")
        print(f"||Δx||/||x|| ≤ κ(A) × ||Δb||/||b||")
        
        max_x_error = kappa * b_error
        print(f"\nMaximum relative error in x:")
        print(f"= {kappa} × {b_error}")
        print(f"= {max_x_error} = {max_x_error * 100}%")
        
        print(f"\nInterpretation:")
        print(f"A 0.1% error in b can become up to 10% error in x!")
    
    def exercise_9_spectral_normalization(self):
        """
        Exercise 9: Spectral Normalization
        
        For W = [[3, 1], [2, 2]]:
        a) Compute ||W||₂
        b) Compute W_normalized = W / ||W||₂
        c) Verify ||W_normalized||₂ = 1
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Spectral Normalization")
        
        W = np.array([[3, 1],
                      [2, 2]])
        
        print(f"W = \n{W}")
        
        _, S, _ = svd(W)
        spec_norm = S[0]
        print(f"\na) ||W||₂ = σ_max = {spec_norm:.4f}")
        
        W_norm = W / spec_norm
        print(f"\nb) W_normalized = W / ||W||₂:")
        print(f"   = \n{np.round(W_norm, 4)}")
        
        _, S_norm, _ = svd(W_norm)
        print(f"\nc) ||W_normalized||₂ = {S_norm[0]:.4f}")
        print(f"   This equals 1! ✓")
        
        print(f"\nApplication in GANs:")
        print("Spectral normalization constrains the Lipschitz constant")
        print("of the discriminator to stabilize training.")
    
    def exercise_10_low_rank_approx(self):
        """
        Exercise 10: Low-Rank Approximation Error
        
        For A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]:
        a) Compute SVD
        b) Find rank-1 approximation
        c) Compute ||A - A₁||_F (error in Frobenius norm)
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Low-Rank Approximation")
        
        A = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=float)
        
        print(f"A = \n{A}")
        
        U, S, Vt = svd(A)
        print(f"\na) SVD:")
        print(f"   Singular values: {np.round(S, 4)}")
        print(f"   Note: σ₃ ≈ 0 (matrix is rank 2)")
        
        # Rank-1 approximation
        A1 = S[0] * np.outer(U[:, 0], Vt[0, :])
        print(f"\nb) Rank-1 approximation A₁ = σ₁ u₁ v₁^T:")
        print(f"   A₁ = \n{np.round(A1, 4)}")
        
        # Error
        error = norm(A - A1, 'fro')
        print(f"\nc) ||A - A₁||_F = {error:.4f}")
        print(f"   By Eckart-Young: this equals √(σ₂² + σ₃²)")
        print(f"   = √({S[1]**2:.4f} + {S[2]**2:.4f})")
        print(f"   = √{S[1]**2 + S[2]**2:.4f} = {np.sqrt(S[1]**2 + S[2]**2):.4f} ✓")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = MatrixNormExercises()
    
    print("MATRIX NORMS AND CONDITION NUMBERS EXERCISES")
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
