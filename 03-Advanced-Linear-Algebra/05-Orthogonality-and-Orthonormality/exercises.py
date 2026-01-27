"""
Orthogonality and Orthonormality - Exercises
============================================
Practice problems for orthogonality concepts.
"""

import numpy as np
from numpy.linalg import norm, inv, qr, det


class OrthogonalityExercises:
    """Exercises for orthogonality and orthonormality."""
    
    def exercise_1_check_orthogonality(self):
        """
        Exercise 1: Check Orthogonality
        
        Determine which pairs of vectors are orthogonal:
        a) u = [1, 2, 3], v = [3, 0, -1]
        b) u = [1, 1, 1], v = [1, -1, 0]
        c) u = [2, -1, 3], v = [1, 5, -1]
        d) u = [1, 0, 1], v = [0, 1, 0]
        """
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solution: Check Orthogonality")
        
        pairs = [
            ('a', np.array([1, 2, 3]), np.array([3, 0, -1])),
            ('b', np.array([1, 1, 1]), np.array([1, -1, 0])),
            ('c', np.array([2, -1, 3]), np.array([1, 5, -1])),
            ('d', np.array([1, 0, 1]), np.array([0, 1, 0]))
        ]
        
        for label, u, v in pairs:
            dot = np.dot(u, v)
            print(f"\n{label}) u = {u}, v = {v}")
            print(f"   u · v = {dot}")
            print(f"   Orthogonal? {dot == 0}")
    
    def exercise_2_verify_orthonormal(self):
        """
        Exercise 2: Verify Orthonormal Set
        
        Verify that the following vectors form an orthonormal set:
        u1 = [1/√2, 1/√2, 0]
        u2 = [-1/√2, 1/√2, 0]
        u3 = [0, 0, 1]
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("\nExercise 2 Solution: Verify Orthonormal Set")
        
        u1 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
        u2 = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
        u3 = np.array([0, 0, 1])
        
        print(f"u1 = {np.round(u1, 4)}")
        print(f"u2 = {np.round(u2, 4)}")
        print(f"u3 = {u3}")
        
        print("\nOrthogonality (all pairs have zero dot product):")
        print(f"u1 · u2 = {np.round(np.dot(u1, u2), 10)}")
        print(f"u1 · u3 = {np.round(np.dot(u1, u3), 10)}")
        print(f"u2 · u3 = {np.round(np.dot(u2, u3), 10)}")
        
        print("\nNormalization (all have unit length):")
        print(f"||u1|| = {norm(u1):.4f}")
        print(f"||u2|| = {norm(u2):.4f}")
        print(f"||u3|| = {norm(u3):.4f}")
        
        print("\n✓ Orthonormal set verified!")
    
    def exercise_3_gram_schmidt(self):
        """
        Exercise 3: Gram-Schmidt Process
        
        Apply Gram-Schmidt to orthonormalize:
        v1 = [1, 0, 1]
        v2 = [1, 1, 0]
        v3 = [0, 1, 1]
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("\nExercise 3 Solution: Gram-Schmidt Process")
        
        v1 = np.array([1, 0, 1], dtype=float)
        v2 = np.array([1, 1, 0], dtype=float)
        v3 = np.array([0, 1, 1], dtype=float)
        
        print(f"Original vectors:")
        print(f"v1 = {v1}")
        print(f"v2 = {v2}")
        print(f"v3 = {v3}")
        
        # Step 1: u1 = v1 / ||v1||
        u1 = v1 / norm(v1)
        print(f"\nStep 1: u1 = v1 / ||v1||")
        print(f"||v1|| = √2")
        print(f"u1 = {np.round(u1, 4)}")
        
        # Step 2: Orthogonalize v2, then normalize
        proj_u1_v2 = np.dot(v2, u1) * u1
        w2 = v2 - proj_u1_v2
        u2 = w2 / norm(w2)
        print(f"\nStep 2: u2")
        print(f"proj_u1(v2) = (v2 · u1)u1 = {np.round(np.dot(v2, u1), 4)} × u1 = {np.round(proj_u1_v2, 4)}")
        print(f"w2 = v2 - proj = {np.round(w2, 4)}")
        print(f"u2 = w2 / ||w2|| = {np.round(u2, 4)}")
        
        # Step 3: Orthogonalize v3, then normalize
        proj_u1_v3 = np.dot(v3, u1) * u1
        proj_u2_v3 = np.dot(v3, u2) * u2
        w3 = v3 - proj_u1_v3 - proj_u2_v3
        u3 = w3 / norm(w3)
        print(f"\nStep 3: u3")
        print(f"proj_u1(v3) = {np.round(proj_u1_v3, 4)}")
        print(f"proj_u2(v3) = {np.round(proj_u2_v3, 4)}")
        print(f"w3 = v3 - proj_u1 - proj_u2 = {np.round(w3, 4)}")
        print(f"u3 = w3 / ||w3|| = {np.round(u3, 4)}")
        
        # Verify
        print("\nVerification:")
        print(f"u1 · u2 = {np.round(np.dot(u1, u2), 10)}")
        print(f"u1 · u3 = {np.round(np.dot(u1, u3), 10)}")
        print(f"u2 · u3 = {np.round(np.dot(u2, u3), 10)}")
    
    def exercise_4_orthogonal_matrix(self):
        """
        Exercise 4: Orthogonal Matrix
        
        Verify that the following matrix is orthogonal:
        Q = [[cos θ, -sin θ],
             [sin θ,  cos θ]]
        for θ = π/3
        
        Also verify: det(Q) = 1 and Q⁻¹ = Qᵀ
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("\nExercise 4 Solution: Orthogonal Matrix")
        
        theta = np.pi / 3
        Q = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        print(f"θ = π/3 = 60°")
        print(f"Q = \n{np.round(Q, 4)}")
        
        print("\nCheck Q^T Q = I:")
        QTQ = Q.T @ Q
        print(f"Q^T Q = \n{np.round(QTQ, 4)}")
        print(f"Is identity? {np.allclose(QTQ, np.eye(2))}")
        
        print("\nCheck det(Q) = ±1:")
        print(f"det(Q) = {det(Q):.4f}")
        print("det(Q) = 1 means rotation (preserves orientation)")
        
        print("\nCheck Q⁻¹ = Q^T:")
        Q_inv = inv(Q)
        print(f"Q⁻¹ = \n{np.round(Q_inv, 4)}")
        print(f"Q^T = \n{np.round(Q.T, 4)}")
        print(f"Q⁻¹ = Q^T? {np.allclose(Q_inv, Q.T)}")
    
    def exercise_5_projection_vector(self):
        """
        Exercise 5: Projection onto a Vector
        
        Find the projection of v = [5, 3, 2] onto u = [1, 2, 2].
        Also find the orthogonal component.
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("\nExercise 5 Solution: Projection onto a Vector")
        
        v = np.array([5, 3, 2])
        u = np.array([1, 2, 2])
        
        print(f"v = {v}")
        print(f"u = {u}")
        
        # Projection
        proj = (np.dot(v, u) / np.dot(u, u)) * u
        print(f"\nproj_u(v) = (v · u / ||u||²) × u")
        print(f"v · u = {np.dot(v, u)}")
        print(f"||u||² = {np.dot(u, u)}")
        print(f"proj = ({np.dot(v, u)}/{np.dot(u, u)}) × {u}")
        print(f"proj = {proj}")
        
        # Orthogonal component
        perp = v - proj
        print(f"\nOrthogonal component:")
        print(f"v_⊥ = v - proj = {perp}")
        
        # Verify
        print(f"\nVerification:")
        print(f"proj + v_⊥ = {proj + perp} = v ✓")
        print(f"proj · v_⊥ = {np.dot(proj, perp):.10f} ≈ 0 ✓")
    
    def exercise_6_projection_matrix(self):
        """
        Exercise 6: Projection Matrix
        
        Find the projection matrix that projects onto the line
        spanned by u = [1, 2].
        
        Use P = uu^T / (u^T u)
        
        Verify: P² = P and P^T = P
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("\nExercise 6 Solution: Projection Matrix")
        
        u = np.array([1, 2])
        print(f"u = {u}")
        
        # Projection matrix
        P = np.outer(u, u) / np.dot(u, u)
        print(f"\nP = uu^T / (u^T u)")
        print(f"u^T u = {np.dot(u, u)}")
        print(f"uu^T = \n{np.outer(u, u)}")
        print(f"\nP = \n{P}")
        
        print("\nVerification:")
        print(f"P² = \n{P @ P}")
        print(f"P² = P? {np.allclose(P @ P, P)}")
        
        print(f"\nP^T = \n{P.T}")
        print(f"P^T = P? {np.allclose(P.T, P)}")
        
        # Test
        v = np.array([3, 1])
        print(f"\nTest: Project v = {v}")
        print(f"Pv = {P @ v}")
        print(f"This is on the line y = 2x: y/x = {(P @ v)[1]/(P @ v)[0]:.1f}")
    
    def exercise_7_qr_decomposition(self):
        """
        Exercise 7: QR Decomposition
        
        Compute the QR decomposition of:
        A = [[1, 2],
             [0, 1],
             [1, 0]]
        
        Verify that Q has orthonormal columns and R is upper triangular.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("\nExercise 7 Solution: QR Decomposition")
        
        A = np.array([[1, 2],
                      [0, 1],
                      [1, 0]], dtype=float)
        
        print(f"A = \n{A}")
        
        # Manual QR via Gram-Schmidt
        print("\n--- Manual Gram-Schmidt ---")
        a1, a2 = A[:, 0], A[:, 1]
        
        # q1
        q1 = a1 / norm(a1)
        print(f"q1 = a1/||a1|| = {np.round(q1, 4)}")
        
        # q2
        proj = np.dot(a2, q1) * q1
        w2 = a2 - proj
        q2 = w2 / norm(w2)
        print(f"proj_q1(a2) = {np.round(proj, 4)}")
        print(f"w2 = a2 - proj = {np.round(w2, 4)}")
        print(f"q2 = w2/||w2|| = {np.round(q2, 4)}")
        
        Q = np.column_stack([q1, q2])
        R = Q.T @ A
        
        print(f"\nQ = \n{np.round(Q, 4)}")
        print(f"\nR = Q^T A = \n{np.round(R, 4)}")
        
        print("\nVerification:")
        print(f"Q has orthonormal columns: Q^T Q = \n{np.round(Q.T @ Q, 4)}")
        print(f"Q @ R = \n{np.round(Q @ R, 4)}")
        print(f"Q @ R = A? {np.allclose(Q @ R, A)}")
    
    def exercise_8_least_squares(self):
        """
        Exercise 8: Least Squares via QR
        
        Find the least squares solution to Ax = b:
        A = [[1, 1],
             [1, 2],
             [1, 3]]
        b = [2, 3, 5]
        
        Use QR decomposition: x = R⁻¹Q^Tb
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("\nExercise 8 Solution: Least Squares via QR")
        
        A = np.array([[1, 1],
                      [1, 2],
                      [1, 3]], dtype=float)
        b = np.array([2, 3, 5], dtype=float)
        
        print(f"A = \n{A}")
        print(f"b = {b}")
        
        # QR decomposition
        Q, R = qr(A)
        Q = Q[:, :2]  # Thin Q
        R = R[:2, :]  # Upper triangular part
        
        print(f"\nQ = \n{np.round(Q, 4)}")
        print(f"\nR = \n{np.round(R, 4)}")
        
        # Solve Rx = Q^T b
        QTb = Q.T @ b
        print(f"\nQ^T b = {np.round(QTb, 4)}")
        
        x = np.linalg.solve(R, QTb)
        print(f"\nSolve Rx = Q^T b:")
        print(f"x = {np.round(x, 4)}")
        
        # Interpret: y = x[0] + x[1]*t (linear fit)
        print(f"\nLinear fit: y = {x[0]:.4f} + {x[1]:.4f}t")
        
        # Residual
        residual = b - A @ x
        print(f"\nResidual: b - Ax = {np.round(residual, 4)}")
        print(f"||residual|| = {norm(residual):.4f}")
    
    def exercise_9_orthogonal_complement(self):
        """
        Exercise 9: Orthogonal Complement
        
        Let W be the plane spanned by v1 = [1, 0, 1] and v2 = [0, 1, 1].
        Find a basis for W^⊥ (orthogonal complement).
        
        Hint: Find vectors perpendicular to both v1 and v2.
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("\nExercise 9 Solution: Orthogonal Complement")
        
        v1 = np.array([1, 0, 1])
        v2 = np.array([0, 1, 1])
        
        print(f"W spanned by:")
        print(f"v1 = {v1}")
        print(f"v2 = {v2}")
        
        print("\nW^⊥ = vectors x where x · v1 = 0 and x · v2 = 0")
        print("This gives the system:")
        print("  x₁ + x₃ = 0")
        print("  x₂ + x₃ = 0")
        print("\nSolution: x₁ = -x₃, x₂ = -x₃")
        print("Let x₃ = 1: x = [-1, -1, 1]")
        
        w_perp = np.array([-1, -1, 1])
        print(f"\nBasis for W^⊥: {w_perp}")
        
        # Verify
        print("\nVerification:")
        print(f"w_⊥ · v1 = {np.dot(w_perp, v1)}")
        print(f"w_⊥ · v2 = {np.dot(w_perp, v2)}")
        
        # Dimension check
        print(f"\ndim(W) + dim(W^⊥) = 2 + 1 = 3 = dim(ℝ³) ✓")
    
    def exercise_10_signal_preservation(self):
        """
        Exercise 10: Signal Preservation
        
        Show that orthogonal matrices preserve signal energy.
        
        Given x = [3, 4] and rotation matrix R by 60°:
        a) Compute ||x||²
        b) Compute ||Rx||²
        c) Compare and explain
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("\nExercise 10 Solution: Signal Preservation")
        
        x = np.array([3, 4])
        theta = np.pi / 3  # 60 degrees
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        
        print(f"x = {x}")
        print(f"R (rotation by 60°) = \n{np.round(R, 4)}")
        
        print(f"\na) ||x||² = {x[0]}² + {x[1]}² = {norm(x)**2}")
        
        Rx = R @ x
        print(f"\nb) Rx = {np.round(Rx, 4)}")
        print(f"   ||Rx||² = {norm(Rx)**2:.4f}")
        
        print(f"\nc) ||x||² = ||Rx||² = {norm(x)**2}")
        print("\nExplanation:")
        print("Orthogonal matrices preserve the L2 norm (energy).")
        print("Proof: ||Rx||² = (Rx)^T(Rx) = x^T R^T R x = x^T x = ||x||²")
        print("Since R^T R = I for orthogonal matrices.")
        
        print("\nThis is why orthogonal weight init helps in neural networks:")
        print("- Prevents signal explosion or vanishing")
        print("- Gradients are also preserved during backprop")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = OrthogonalityExercises()
    
    print("ORTHOGONALITY AND ORTHONORMALITY EXERCISES")
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
