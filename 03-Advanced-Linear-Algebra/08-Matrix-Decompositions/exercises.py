"""
Matrix Decompositions - Exercises
=================================
Practice problems for LU, QR, and Cholesky decompositions.
"""

import numpy as np
from numpy.linalg import qr, cholesky, solve, det, inv
from scipy.linalg import lu, lu_factor, lu_solve


class MatrixDecompositionExercises:
    """Exercises for matrix decompositions."""
    
    def exercise_1_lu_manual(self):
        """
        Exercise 1: Manual LU Decomposition
        
        Find the LU decomposition (without pivoting) of:
        A = [[2, 1],
             [6, 4]]
        
        Verify: A = L @ U
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Manual LU Decomposition")
        
        A = np.array([[2, 1],
                      [6, 4]], dtype=float)
        
        print(f"A = \n{A}")
        
        print("\n--- Gaussian Elimination ---")
        print("To eliminate A[1,0] = 6:")
        print("Row2 = Row2 - (6/2)*Row1 = Row2 - 3*Row1")
        
        L = np.array([[1, 0],
                      [3, 1]], dtype=float)  # Multiplier is 6/2 = 3
        
        U = np.array([[2, 1],
                      [0, 1]], dtype=float)  # [6,4] - 3*[2,1] = [0,1]
        
        print(f"\nL = \n{L}")
        print(f"U = \n{U}")
        
        print(f"\nVerification: L @ U = \n{L @ U}")
        print(f"Equals A? {np.allclose(L @ U, A)}")
    
    def exercise_2_solve_lu(self):
        """
        Exercise 2: Solve System with LU
        
        Use LU decomposition to solve:
        [[3, 1],     [x]   [5]
         [6, 4]] ×  [y] = [14]
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Solve with LU")
        
        A = np.array([[3, 1],
                      [6, 4]], dtype=float)
        b = np.array([5, 14], dtype=float)
        
        print(f"A = \n{A}")
        print(f"b = {b}")
        
        # Step 1: LU decomposition
        print("\n--- Step 1: LU Decomposition ---")
        # Multiplier: 6/3 = 2
        L = np.array([[1, 0],
                      [2, 1]], dtype=float)
        # Row2 - 2*Row1: [6,4] - 2*[3,1] = [0,2]
        U = np.array([[3, 1],
                      [0, 2]], dtype=float)
        
        print(f"L = \n{L}")
        print(f"U = \n{U}")
        
        # Step 2: Forward solve Ly = b
        print("\n--- Step 2: Forward Solve Ly = b ---")
        print("y[0] = b[0] = 5")
        print("y[1] = b[1] - L[1,0]*y[0] = 14 - 2*5 = 4")
        y = np.array([5, 4], dtype=float)
        print(f"y = {y}")
        
        # Step 3: Back solve Ux = y
        print("\n--- Step 3: Back Solve Ux = y ---")
        print("x[1] = y[1]/U[1,1] = 4/2 = 2")
        print("x[0] = (y[0] - U[0,1]*x[1])/U[0,0] = (5 - 1*2)/3 = 1")
        x = np.array([1, 2], dtype=float)
        print(f"x = {x}")
        
        print(f"\nVerification: A @ x = {A @ x}")
    
    def exercise_3_qr_gram_schmidt(self):
        """
        Exercise 3: QR via Gram-Schmidt
        
        Find QR decomposition of:
        A = [[1, 1],
             [1, 0],
             [0, 1]]
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: QR via Gram-Schmidt")
        
        A = np.array([[1, 1],
                      [1, 0],
                      [0, 1]], dtype=float)
        
        print(f"A = \n{A}")
        
        print("\n--- Gram-Schmidt ---")
        
        # Column 1
        a1 = A[:, 0]
        print(f"a₁ = {a1}")
        r11 = np.linalg.norm(a1)
        q1 = a1 / r11
        print(f"||a₁|| = r₁₁ = {r11:.4f}")
        print(f"q₁ = a₁/r₁₁ = {np.round(q1, 4)}")
        
        # Column 2
        a2 = A[:, 1]
        print(f"\na₂ = {a2}")
        r12 = q1 @ a2
        print(f"r₁₂ = q₁ᵀa₂ = {r12:.4f}")
        v2 = a2 - r12 * q1
        print(f"v₂ = a₂ - r₁₂q₁ = {np.round(v2, 4)}")
        r22 = np.linalg.norm(v2)
        q2 = v2 / r22
        print(f"r₂₂ = ||v₂|| = {r22:.4f}")
        print(f"q₂ = v₂/r₂₂ = {np.round(q2, 4)}")
        
        Q = np.column_stack([q1, q2])
        R = np.array([[r11, r12], [0, r22]])
        
        print(f"\nQ = \n{np.round(Q, 4)}")
        print(f"R = \n{np.round(R, 4)}")
        print(f"\nQᵀQ = \n{np.round(Q.T @ Q, 4)}")
        print(f"\nQ @ R = \n{np.round(Q @ R, 4)}")
    
    def exercise_4_least_squares_qr(self):
        """
        Exercise 4: Least Squares via QR
        
        Use QR to find the least squares solution to:
        [[1],       [1]
         [1], x ≈  [2]
         [1]]      [3]
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Least Squares via QR")
        
        A = np.array([[1], [1], [1]], dtype=float)
        b = np.array([1, 2, 3], dtype=float)
        
        print(f"A = {A.flatten()}")
        print(f"b = {b}")
        
        print("\n--- QR Decomposition ---")
        Q, R = qr(A, mode='reduced')
        print(f"Q = {np.round(Q.flatten(), 4)} (3×1)")
        print(f"R = [[{R[0,0]:.4f}]] (1×1)")
        
        print("\n--- Solve Rx = Qᵀb ---")
        Qb = Q.T @ b
        print(f"Qᵀb = {Qb}")
        x = Qb / R[0, 0]
        print(f"x = Qᵀb / R = {x}")
        
        print(f"\nSolution: x = {x[0]:.4f}")
        print("(This is the mean of b: (1+2+3)/3 = 2)")
        
        print(f"\nResidual: ||Ax - b|| = {np.linalg.norm(A @ x - b):.4f}")
    
    def exercise_5_cholesky_manual(self):
        """
        Exercise 5: Manual Cholesky
        
        Find the Cholesky decomposition of:
        A = [[4, 6],
             [6, 13]]
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Manual Cholesky")
        
        A = np.array([[4, 6],
                      [6, 13]], dtype=float)
        
        print(f"A = \n{A}")
        
        print("\n--- Cholesky Formula ---")
        print("L = [[l₁₁, 0  ],")
        print("     [l₂₁, l₂₂]]")
        
        print("\nl₁₁ = √a₁₁ = √4 = 2")
        print("l₂₁ = a₂₁/l₁₁ = 6/2 = 3")
        print("l₂₂ = √(a₂₂ - l₂₁²) = √(13 - 9) = √4 = 2")
        
        L = np.array([[2, 0],
                      [3, 2]], dtype=float)
        
        print(f"\nL = \n{L}")
        print(f"\nVerification: L @ Lᵀ = \n{L @ L.T}")
        
        # Verify with numpy
        L_np = cholesky(A)
        print(f"\nNumPy Cholesky: \n{L_np}")
    
    def exercise_6_gaussian_sampling(self):
        """
        Exercise 6: Gaussian Sampling
        
        Use Cholesky to sample from N(μ, Σ) where:
        μ = [0, 0]
        Σ = [[1, 0.5],
             [0.5, 1]]
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Gaussian Sampling")
        
        mu = np.array([0, 0])
        Sigma = np.array([[1, 0.5],
                         [0.5, 1]])
        
        print(f"μ = {mu}")
        print(f"Σ = \n{Sigma}")
        
        # Cholesky
        L = cholesky(Sigma)
        print(f"\nL = cholesky(Σ) = \n{np.round(L, 4)}")
        
        # Sample
        print("\n--- Sampling Algorithm ---")
        print("1. Sample z ~ N(0, I)")
        print("2. Compute x = μ + L @ z")
        
        np.random.seed(0)
        print("\nSamples:")
        for i in range(5):
            z = np.random.randn(2)
            x = mu + L @ z
            print(f"  z = {np.round(z, 3)} → x = {np.round(x, 3)}")
    
    def exercise_7_determinant(self):
        """
        Exercise 7: Determinant from Decomposition
        
        Compute det(A) using Cholesky for:
        A = [[9, 3],
             [3, 5]]
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Determinant from Cholesky")
        
        A = np.array([[9, 3],
                      [3, 5]], dtype=float)
        
        print(f"A = \n{A}")
        
        # Direct
        det_direct = det(A)
        print(f"\nDirect: det(A) = 9×5 - 3×3 = {det_direct}")
        
        # From Cholesky
        L = cholesky(A)
        print(f"\nCholesky L = \n{np.round(L, 4)}")
        print(f"Diagonal of L: {np.round(np.diag(L), 4)}")
        
        det_chol = np.prod(np.diag(L))**2
        print(f"\ndet(A) = (∏ l_ii)² = ({np.diag(L)[0]:.4f} × {np.diag(L)[1]:.4f})²")
        print(f"       = {np.prod(np.diag(L)):.4f}² = {det_chol:.4f}")
    
    def exercise_8_ldlt(self):
        """
        Exercise 8: LDLᵀ Decomposition
        
        Find A = LDLᵀ for:
        A = [[4, 2],
             [2, 5]]
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: LDLᵀ Decomposition")
        
        A = np.array([[4, 2],
                      [2, 5]], dtype=float)
        
        print(f"A = \n{A}")
        
        print("\n--- LDLᵀ where L is unit lower triangular ---")
        
        # d₁ = a₁₁
        d1 = A[0, 0]
        print(f"d₁ = a₁₁ = {d1}")
        
        # l₂₁ = a₂₁/d₁
        l21 = A[1, 0] / d1
        print(f"l₂₁ = a₂₁/d₁ = 2/4 = {l21}")
        
        # d₂ = a₂₂ - l₂₁²·d₁
        d2 = A[1, 1] - l21**2 * d1
        print(f"d₂ = a₂₂ - l₂₁²·d₁ = 5 - 0.25×4 = {d2}")
        
        L = np.array([[1, 0],
                      [l21, 1]])
        D = np.diag([d1, d2])
        
        print(f"\nL = \n{L}")
        print(f"D = \n{D}")
        
        print(f"\nVerification: L @ D @ Lᵀ = \n{L @ D @ L.T}")
        
        print("\nRelation to Cholesky: A = (L√D)(L√D)ᵀ")
        L_chol = L @ np.diag(np.sqrt([d1, d2]))
        print(f"L_chol = \n{np.round(L_chol, 4)}")
    
    def exercise_9_condition_number(self):
        """
        Exercise 9: Condition Number Impact
        
        Compare solutions for Ax = b with:
        A1 = [[1, 0], [0, 1]]    (well-conditioned)
        A2 = [[1, 1], [1, 1+1e-10]]  (ill-conditioned)
        b = [1, 1]
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Condition Number Impact")
        
        A1 = np.array([[1, 0], [0, 1]], dtype=float)
        A2 = np.array([[1, 1], [1, 1 + 1e-10]], dtype=float)
        b = np.array([1, 1], dtype=float)
        
        print("--- Well-conditioned System ---")
        print(f"A₁ = \n{A1}")
        print(f"κ(A₁) = {np.linalg.cond(A1):.1f}")
        x1 = solve(A1, b)
        print(f"Solution: x = {x1}")
        
        print("\n--- Ill-conditioned System ---")
        print(f"A₂ = \n{A2}")
        print(f"κ(A₂) = {np.linalg.cond(A2):.2e}")
        x2 = solve(A2, b)
        print(f"Solution: x = {x2}")
        
        print("\n--- Effect of Small Perturbation ---")
        b_perturbed = b + np.array([1e-10, 0])
        
        x1_pert = solve(A1, b_perturbed)
        x2_pert = solve(A2, b_perturbed)
        
        print(f"Well-conditioned: Δx = {np.linalg.norm(x1_pert - x1):.2e}")
        print(f"Ill-conditioned:  Δx = {np.linalg.norm(x2_pert - x2):.2e}")
        print("\nIll-conditioned systems amplify errors!")
    
    def exercise_10_ridge_regression(self):
        """
        Exercise 10: Ridge Regression via Cholesky
        
        Solve (XᵀX + λI)w = Xᵀy using Cholesky.
        
        X = [[1, 1],
             [1, 2],
             [1, 3]]
        y = [1, 2, 2]
        λ = 0.1
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Ridge Regression via Cholesky")
        
        X = np.array([[1, 1],
                      [1, 2],
                      [1, 3]], dtype=float)
        y = np.array([1, 2, 2], dtype=float)
        lam = 0.1
        
        print(f"X = \n{X}")
        print(f"y = {y}")
        print(f"λ = {lam}")
        
        # Normal equations with regularization
        XTX = X.T @ X
        XTy = X.T @ y
        
        print(f"\nXᵀX = \n{XTX}")
        print(f"Xᵀy = {XTy}")
        
        A = XTX + lam * np.eye(2)
        print(f"\nXᵀX + λI = \n{A}")
        
        # Verify PD
        from numpy.linalg import eigvalsh
        eigs = eigvalsh(A)
        print(f"Eigenvalues: {np.round(eigs, 4)} (all > 0 ✓)")
        
        # Cholesky solve
        L = cholesky(A)
        print(f"\nCholesky L = \n{np.round(L, 4)}")
        
        # Forward: Lz = Xᵀy
        z = solve(L, XTy)
        # Backward: Lᵀw = z
        w = solve(L.T, z)
        
        print(f"\nSolution w = {np.round(w, 4)}")
        print(f"Predictions: Xw = {np.round(X @ w, 4)}")
        print(f"True y:      {y}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = MatrixDecompositionExercises()
    
    print("MATRIX DECOMPOSITIONS EXERCISES")
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
