"""
Linear Transformations - Exercises
==================================
Practice problems for linear transformation concepts.
"""

import numpy as np
from numpy.linalg import det, inv, eig, norm, matrix_rank


class LinearTransformationExercises:
    """Exercises for linear transformations."""
    
    def exercise_1_verify_linearity(self):
        """
        Exercise 1: Verify Linearity
        
        Determine which of the following are linear transformations:
        a) T(x, y) = (x + y, x - y)
        b) T(x, y) = (x + 1, y)
        c) T(x, y) = (xy, y)
        d) T(x, y) = (2x, 3y)
        
        For each, check T(u + v) = T(u) + T(v) and T(cv) = cT(v).
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Verify Linearity")
        
        print("\na) T(x, y) = (x + y, x - y)")
        print("   Matrix form: T(v) = [[1, 1], [1, -1]] @ v")
        print("   ✓ Linear (can be written as matrix multiplication)")
        
        print("\nb) T(x, y) = (x + 1, y)")
        print("   Check T(0, 0) = (0 + 1, 0) = (1, 0) ≠ (0, 0)")
        print("   ✗ Not linear (T(0) ≠ 0)")
        
        print("\nc) T(x, y) = (xy, y)")
        print("   Check T(2, 1) = (2, 1)")
        print("   Check 2T(1, 1) = 2(1, 1) = (2, 2)")
        print("   But T(2·(1, 1)) = T(2, 2) = (4, 2) ≠ (2, 2)")
        print("   ✗ Not linear (T(cv) ≠ cT(v))")
        
        print("\nd) T(x, y) = (2x, 3y)")
        print("   Matrix form: T(v) = [[2, 0], [0, 3]] @ v")
        print("   ✓ Linear (diagonal scaling matrix)")
        
        # Numerical verification for d)
        A = np.array([[2, 0], [0, 3]])
        u = np.array([1, 2])
        v = np.array([3, 4])
        c = 5
        
        print("\n   Numerical verification for (d):")
        print(f"   T(u + v) = {A @ (u + v)}")
        print(f"   T(u) + T(v) = {A @ u + A @ v}")
        print(f"   T({c}v) = {A @ (c * v)}")
        print(f"   {c}T(v) = {c * (A @ v)}")
    
    def exercise_2_find_matrix(self):
        """
        Exercise 2: Find the Matrix Representation
        
        Find the matrix A such that T(x) = Ax for:
        a) T rotates by 30° counterclockwise
        b) T reflects across the line y = x
        c) T projects onto the line y = 2x
        d) T scales by 2 in x and 0.5 in y
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Find Matrix Representation")
        
        print("\na) Rotation by 30°:")
        theta = np.pi / 6  # 30 degrees
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        print(f"   R = \n{np.round(R, 4)}")
        
        print("\nb) Reflection across y = x:")
        M = np.array([[0, 1],
                      [1, 0]])
        print(f"   M = \n{M}")
        print("   (Swaps x and y coordinates)")
        
        print("\nc) Projection onto y = 2x:")
        # Direction vector: (1, 2), normalized: (1/√5, 2/√5)
        # Projection matrix: P = vvᵀ/||v||²
        v = np.array([1, 2])
        P = np.outer(v, v) / np.dot(v, v)
        print(f"   P = vvᵀ/||v||² where v = {v}")
        print(f"   P = \n{P}")
        
        # Verify
        test = np.array([5, 0])
        projected = P @ test
        print(f"   Check: P @ {test} = {projected}")
        print(f"   Is on y=2x: y/x = {projected[1]/projected[0]:.1f}")
        
        print("\nd) Scale by 2 in x, 0.5 in y:")
        S = np.array([[2, 0],
                      [0, 0.5]])
        print(f"   S = \n{S}")
    
    def exercise_3_composition(self):
        """
        Exercise 3: Composition of Transformations
        
        Let R be rotation by 90° and S be reflection across x-axis.
        a) Find the matrix for R
        b) Find the matrix for S
        c) Compute RS (first S, then R)
        d) Compute SR (first R, then S)
        e) Are RS and SR equal? What is each geometrically?
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Composition")
        
        print("\na) Rotation by 90°:")
        theta = np.pi / 2
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        R = np.round(R, 10)  # Clean up near-zeros
        print(f"   R = \n{R}")
        
        print("\nb) Reflection across x-axis:")
        S = np.array([[1, 0],
                      [0, -1]])
        print(f"   S = \n{S}")
        
        print("\nc) RS (first S, then R):")
        RS = R @ S
        print(f"   RS = \n{RS}")
        
        print("\nd) SR (first R, then S):")
        SR = S @ R
        print(f"   SR = \n{SR}")
        
        print("\ne) Comparison:")
        print(f"   RS = SR? {np.allclose(RS, SR)}")
        print("\n   Geometric interpretation:")
        print("   RS: Reflect across x-axis, then rotate 90°")
        print("       → Equivalent to reflection across y = x")
        print("   SR: Rotate 90°, then reflect across x-axis")
        print("       → Equivalent to reflection across y = -x")
        
        # Verify
        v = np.array([1, 0])
        print(f"\n   Check with v = {v}:")
        print(f"   RS @ v = {RS @ v}")
        print(f"   SR @ v = {SR @ v}")
    
    def exercise_4_kernel_image(self):
        """
        Exercise 4: Kernel and Image
        
        For A = [[1, 2, 1], [2, 4, 2]]:
        a) Find the rank of A
        b) Find the nullity (dimension of kernel)
        c) Find a basis for the kernel
        d) Find a basis for the image
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Kernel and Image")
        
        A = np.array([[1, 2, 1],
                      [2, 4, 2]])
        print(f"A = \n{A}")
        
        print("\na) Rank of A:")
        rank = matrix_rank(A)
        print(f"   rank(A) = {rank}")
        print("   (Rows are linearly dependent: row2 = 2×row1)")
        
        print("\nb) Nullity:")
        nullity = A.shape[1] - rank
        print(f"   nullity = #columns - rank = {A.shape[1]} - {rank} = {nullity}")
        
        print("\nc) Basis for kernel (null space):")
        print("   Solve Ax = 0:")
        print("   x₁ + 2x₂ + x₃ = 0")
        print("   Let x₂ = s, x₃ = t (free variables)")
        print("   x₁ = -2s - t")
        print("   ")
        print("   Solution: x = s[-2, 1, 0]ᵀ + t[-1, 0, 1]ᵀ")
        print("   Basis: {[-2, 1, 0], [-1, 0, 1]}")
        
        # Verify
        v1 = np.array([-2, 1, 0])
        v2 = np.array([-1, 0, 1])
        print(f"\n   Verification:")
        print(f"   A @ {v1} = {A @ v1}")
        print(f"   A @ {v2} = {A @ v2}")
        
        print("\nd) Basis for image (column space):")
        print("   Image = span of columns of A")
        print("   Column 1: [1, 2]ᵀ")
        print("   Column 2: [2, 4]ᵀ = 2 × Column 1 (dependent)")
        print("   Column 3: [1, 2]ᵀ = Column 1 (dependent)")
        print("   Basis: {[1, 2]ᵀ}")
    
    def exercise_5_invertibility(self):
        """
        Exercise 5: Invertibility
        
        For each matrix, determine if invertible and if so, find inverse:
        a) A = [[1, 2], [3, 4]]
        b) B = [[1, 2], [2, 4]]
        c) C = [[cos θ, -sin θ], [sin θ, cos θ]] for θ = π/3
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Invertibility")
        
        print("\na) A = [[1, 2], [3, 4]]")
        A = np.array([[1, 2], [3, 4]])
        det_A = det(A)
        print(f"   det(A) = 1×4 - 2×3 = {det_A}")
        print(f"   Invertible: {det_A != 0}")
        if det_A != 0:
            A_inv = inv(A)
            print(f"   A⁻¹ = \n{A_inv}")
            print(f"   Verification A @ A⁻¹ = \n{np.round(A @ A_inv, 4)}")
        
        print("\nb) B = [[1, 2], [2, 4]]")
        B = np.array([[1, 2], [2, 4]])
        det_B = det(B)
        print(f"   det(B) = 1×4 - 2×2 = {det_B}")
        print(f"   Invertible: {det_B != 0}")
        print("   Reason: Row 2 = 2 × Row 1 (linearly dependent)")
        
        print("\nc) C (rotation by π/3):")
        theta = np.pi / 3
        C = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        print(f"   C = \n{np.round(C, 4)}")
        det_C = det(C)
        print(f"   det(C) = cos²θ + sin²θ = {det_C:.4f}")
        print(f"   Invertible: {abs(det_C - 1) < 1e-10}")
        
        C_inv = inv(C)
        print(f"   C⁻¹ = rotation by -π/3 = \n{np.round(C_inv, 4)}")
        print(f"   Note: C⁻¹ = Cᵀ (orthogonal matrix)")
        print(f"   C @ Cᵀ = \n{np.round(C @ C.T, 4)}")
    
    def exercise_6_eigenanalysis(self):
        """
        Exercise 6: Eigenanalysis of Transformations
        
        For A = [[2, 1], [0, 3]]:
        a) Find eigenvalues
        b) Find eigenvectors
        c) Interpret geometrically: what directions are only scaled?
        d) Diagonalize A = PDP⁻¹
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Eigenanalysis")
        
        A = np.array([[2, 1], [0, 3]])
        print(f"A = \n{A}")
        
        print("\na) Eigenvalues:")
        print("   Triangular matrix → eigenvalues on diagonal")
        eigenvalues, eigenvectors = eig(A)
        print(f"   λ₁ = {eigenvalues[0]}, λ₂ = {eigenvalues[1]}")
        
        print("\nb) Eigenvectors:")
        for i in range(2):
            print(f"   For λ = {eigenvalues[i]}:")
            print(f"   (A - λI)v = 0")
            v = eigenvectors[:, i]
            print(f"   v = {np.round(v, 4)}")
        
        print("\nc) Geometric interpretation:")
        print(f"   Direction {np.round(eigenvectors[:, 0], 2)} scaled by {eigenvalues[0]}")
        print(f"   Direction {np.round(eigenvectors[:, 1], 2)} scaled by {eigenvalues[1]}")
        
        print("\nd) Diagonalization A = PDP⁻¹:")
        P = eigenvectors
        D = np.diag(eigenvalues)
        P_inv = inv(P)
        
        print(f"   P = \n{np.round(P, 4)}")
        print(f"   D = \n{np.round(D, 4)}")
        print(f"   P⁻¹ = \n{np.round(P_inv, 4)}")
        
        # Verify
        reconstructed = P @ D @ P_inv
        print(f"   Verification PDP⁻¹ = \n{np.round(reconstructed.real, 4)}")
    
    def exercise_7_transformation_types(self):
        """
        Exercise 7: Classify Transformations
        
        For each matrix, determine if it is:
        - Orthogonal (QᵀQ = I)
        - Symmetric (A = Aᵀ)
        - Positive definite (xᵀAx > 0)
        
        a) A = [[1, 0], [0, 1]]
        b) B = [[0, -1], [1, 0]]
        c) C = [[2, 1], [1, 2]]
        d) D = [[1, 2], [0, 1]]
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: Classify Transformations")
        
        matrices = {
            'A': np.array([[1, 0], [0, 1]]),
            'B': np.array([[0, -1], [1, 0]]),
            'C': np.array([[2, 1], [1, 2]]),
            'D': np.array([[1, 2], [0, 1]])
        }
        
        for name, M in matrices.items():
            print(f"\n{name}) Matrix:\n{M}")
            
            # Check orthogonal
            is_orth = np.allclose(M.T @ M, np.eye(2))
            print(f"   Orthogonal (MᵀM = I): {is_orth}")
            
            # Check symmetric
            is_sym = np.allclose(M, M.T)
            print(f"   Symmetric (M = Mᵀ): {is_sym}")
            
            # Check positive definite (if symmetric)
            if is_sym:
                eigenvalues = np.linalg.eigvalsh(M)
                is_pd = all(eigenvalues > 0)
                print(f"   Positive definite (all λ > 0): {is_pd}")
                print(f"      Eigenvalues: {eigenvalues}")
            else:
                print("   Positive definite: N/A (not symmetric)")
    
    def exercise_8_affine_transformation(self):
        """
        Exercise 8: Affine Transformations
        
        Using homogeneous coordinates:
        a) Write the matrix for translation by (3, -2)
        b) Write the matrix for scaling by 2 centered at (1, 1)
           (Hint: translate to origin, scale, translate back)
        c) Compose these transformations
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Affine Transformations")
        
        print("\na) Translation by (3, -2):")
        T = np.array([[1, 0, 3],
                      [0, 1, -2],
                      [0, 0, 1]])
        print(f"   T = \n{T}")
        
        # Verify
        p = np.array([1, 2, 1])  # Point (1, 2) in homogeneous coords
        print(f"   T @ (1, 2, 1)ᵀ = {T @ p} → point ({(T @ p)[0]}, {(T @ p)[1]})")
        
        print("\nb) Scaling by 2 centered at (1, 1):")
        print("   Steps: Translate to origin, scale, translate back")
        
        # Translate (1, 1) to origin
        T1 = np.array([[1, 0, -1],
                       [0, 1, -1],
                       [0, 0, 1]])
        
        # Scale by 2
        S = np.array([[2, 0, 0],
                      [0, 2, 0],
                      [0, 0, 1]])
        
        # Translate back
        T2 = np.array([[1, 0, 1],
                       [0, 1, 1],
                       [0, 0, 1]])
        
        # Combined: T2 @ S @ T1
        M = T2 @ S @ T1
        print(f"   T₂ @ S @ T₁ = \n{M}")
        
        # Verify center is fixed
        center = np.array([1, 1, 1])
        print(f"   M @ center = {M @ center} (fixed point!)")
        
        # Other point
        other = np.array([2, 3, 1])
        result = M @ other
        print(f"   M @ (2, 3) = ({result[0]}, {result[1]})")
        print(f"   Distance from center: {np.sqrt(1+4):.2f} → {np.sqrt((result[0]-1)**2 + (result[1]-1)**2):.2f}")
        
        print("\nc) Compose translation then centered scaling:")
        composed = M @ T
        print(f"   M @ T = \n{composed}")
    
    def exercise_9_neural_layer(self):
        """
        Exercise 9: Neural Network Layer Analysis
        
        For a neural network layer with W = [[1, -1], [2, 1], [-1, 2]]:
        a) What are the input and output dimensions?
        b) What is the rank of W?
        c) Find the kernel of W (what inputs give zero output?)
        d) Apply W to input x = [3, 1]ᵀ
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Neural Network Layer")
        
        W = np.array([[1, -1],
                      [2, 1],
                      [-1, 2]])
        print(f"W = \n{W}")
        
        print("\na) Dimensions:")
        print(f"   Input dimension: {W.shape[1]}")
        print(f"   Output dimension: {W.shape[0]}")
        
        print("\nb) Rank of W:")
        rank = matrix_rank(W)
        print(f"   rank(W) = {rank}")
        
        print("\nc) Kernel of W:")
        print(f"   nullity = #columns - rank = {W.shape[1]} - {rank} = {W.shape[1] - rank}")
        if W.shape[1] - rank == 0:
            print("   Kernel = {0} (only zero vector)")
            print("   W is injective (one-to-one)")
        
        print("\nd) Apply to x = [3, 1]ᵀ:")
        x = np.array([3, 1])
        y = W @ x
        print(f"   W @ x = {y}")
    
    def exercise_10_transformation_visualization(self):
        """
        Exercise 10: Describe Transformations Geometrically
        
        For each matrix, describe what it does to a unit square:
        a) [[2, 0], [0, 2]]
        b) [[1, 0.5], [0, 1]]
        c) [[0, 1], [-1, 0]]
        d) [[1, 0], [0, 0]]
        
        Calculate how the area changes using determinants.
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Geometric Description")
        
        matrices = {
            'a) [[2, 0], [0, 2]]': np.array([[2, 0], [0, 2]]),
            'b) [[1, 0.5], [0, 1]]': np.array([[1, 0.5], [0, 1]]),
            'c) [[0, 1], [-1, 0]]': np.array([[0, 1], [-1, 0]]),
            'd) [[1, 0], [0, 0]]': np.array([[1, 0], [0, 0]])
        }
        
        for name, A in matrices.items():
            print(f"\n{name}")
            print(f"Matrix:\n{A}")
            
            d = det(A)
            print(f"det(A) = {d}")
            print(f"Area factor: |det(A)| = {abs(d)}")
            
            # Describe
            if np.allclose(A, 2*np.eye(2)):
                print("Description: Uniform scaling by 2")
            elif np.allclose(A, np.array([[1, 0.5], [0, 1]])):
                print("Description: Horizontal shear (top shifted right)")
            elif np.allclose(A, np.array([[0, 1], [-1, 0]])):
                print("Description: Rotation by -90° (clockwise)")
            elif np.allclose(A, np.array([[1, 0], [0, 0]])):
                print("Description: Projection onto x-axis")
                print("(Square collapses to a line segment, area = 0)")
            
            # Effect on corners
            corners = np.array([[0, 1, 1, 0],
                               [0, 0, 1, 1]])
            transformed = A @ corners
            print(f"Corners (0,0), (1,0), (1,1), (0,1) map to:")
            for i in range(4):
                print(f"   ({corners[0,i]}, {corners[1,i]}) → ({transformed[0,i]:.1f}, {transformed[1,i]:.1f})")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = LinearTransformationExercises()
    
    print("LINEAR TRANSFORMATIONS EXERCISES")
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
