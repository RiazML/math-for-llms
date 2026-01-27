"""
Vector Spaces and Subspaces - Exercises
=======================================
Practice problems for vector space concepts.
"""

import numpy as np
from scipy import linalg


class VectorSpaceExercises:
    """Exercises for vector spaces and subspaces."""
    
    # ==================== BASIC EXERCISES ====================
    
    def exercise_1_subspace_test(self):
        """
        Exercise 1: Subspace Test
        
        Determine if each set is a subspace of the given vector space.
        Justify your answer.
        
        a) W = {(x, y, z) ∈ ℝ³ : x + 2y - z = 0}
        b) W = {(x, y) ∈ ℝ² : xy = 0}
        c) W = {(x, y, z) ∈ ℝ³ : x ≥ 0}
        d) W = {(x, y, z) ∈ ℝ³ : x = y = z}
        e) W = {A ∈ M₂ₓ₂ : A is symmetric}
        f) W = {A ∈ M₂ₓ₂ : det(A) = 0}
        """
        # Your analysis here
        pass
    
    def solution_1(self):
        """Solution to Exercise 1."""
        print("Exercise 1 Solutions:")
        
        print("\na) W = {(x,y,z) : x + 2y - z = 0}")
        print("   SUBSPACE ✓")
        print("   - (0,0,0): 0 + 0 - 0 = 0 ✓")
        print("   - If u, v ∈ W: (u₁+v₁) + 2(u₂+v₂) - (u₃+v₃)")
        print("                = (u₁+2u₂-u₃) + (v₁+2v₂-v₃) = 0 + 0 = 0 ✓")
        print("   - If v ∈ W, c ∈ ℝ: cx₁ + 2cy₂ - cz₃ = c(x+2y-z) = 0 ✓")
        
        print("\nb) W = {(x,y) : xy = 0}")
        print("   NOT A SUBSPACE ✗")
        print("   - (1,0) ∈ W and (0,1) ∈ W")
        print("   - But (1,0) + (0,1) = (1,1), and 1×1 = 1 ≠ 0")
        print("   - Not closed under addition")
        
        print("\nc) W = {(x,y,z) : x ≥ 0}")
        print("   NOT A SUBSPACE ✗")
        print("   - (1,0,0) ∈ W, but (-1)(1,0,0) = (-1,0,0)")
        print("   - -1 < 0, so not closed under scalar mult")
        
        print("\nd) W = {(x,y,z) : x = y = z}")
        print("   SUBSPACE ✓ (line through origin)")
        print("   - (0,0,0) ∈ W ✓")
        print("   - (a,a,a) + (b,b,b) = (a+b, a+b, a+b) ∈ W ✓")
        print("   - c(a,a,a) = (ca, ca, ca) ∈ W ✓")
        
        print("\ne) W = {A ∈ M₂ₓ₂ : A = Aᵀ}")
        print("   SUBSPACE ✓")
        print("   - Zero matrix is symmetric ✓")
        print("   - (A+B)ᵀ = Aᵀ + Bᵀ = A + B ✓")
        print("   - (cA)ᵀ = cAᵀ = cA ✓")
        
        print("\nf) W = {A ∈ M₂ₓ₂ : det(A) = 0}")
        print("   NOT A SUBSPACE ✗")
        A = np.array([[1, 0], [0, 0]])
        B = np.array([[0, 0], [0, 1]])
        print(f"   A = [[1,0],[0,0]], det(A) = {np.linalg.det(A):.0f}")
        print(f"   B = [[0,0],[0,1]], det(B) = {np.linalg.det(B):.0f}")
        print(f"   A + B = [[1,0],[0,1]], det(A+B) = {np.linalg.det(A+B):.0f} ≠ 0")
    
    def exercise_2_span(self):
        """
        Exercise 2: Span
        
        a) Does v = (1, 2, 3) lie in span{(1,0,1), (0,1,1)}?
        b) Does w = (1, 1, 1) lie in span{(1,0,1), (0,1,1)}?
        c) Find span{(1,2), (2,4), (3,6)} geometrically.
        d) What is span{0}?
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        print("Exercise 2 Solutions:")
        
        # a) v = (1,2,3) in span{(1,0,1), (0,1,1)}?
        print("\na) Is (1,2,3) in span{(1,0,1), (0,1,1)}?")
        v1 = np.array([1, 0, 1])
        v2 = np.array([0, 1, 1])
        v = np.array([1, 2, 3])
        
        # Solve [v1 v2][c1 c2]^T = v
        A = np.column_stack([v1, v2])
        print(f"   Solve: c₁{v1} + c₂{v2} = {v}")
        
        # Check if system is consistent
        Ab = np.column_stack([A, v])
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(Ab)
        print(f"   rank([v1,v2]) = {rank_A}, rank([v1,v2|v]) = {rank_Ab}")
        print(f"   Consistent? {rank_A == rank_Ab}")
        
        if rank_A == rank_Ab:
            c = np.linalg.lstsq(A, v, rcond=None)[0]
            print(f"   Solution: c₁ = {c[0]:.2f}, c₂ = {c[1]:.2f}")
            print(f"   Verify: {c[0]*v1 + c[1]*v2}")
            print("   YES, v is in the span ✓")
        
        # b) w = (1,1,1)
        print("\nb) Is (1,1,1) in span{(1,0,1), (0,1,1)}?")
        w = np.array([1, 1, 1])
        Ab = np.column_stack([A, w])
        rank_Ab = np.linalg.matrix_rank(Ab)
        print(f"   rank([v1,v2]) = {rank_A}, rank([v1,v2|w]) = {rank_Ab}")
        print(f"   Consistent? {rank_A == rank_Ab}")
        print("   NO, w is not in the span ✗")
        
        # c)
        print("\nc) span{(1,2), (2,4), (3,6)}")
        print("   All vectors are multiples of (1,2)")
        print("   span = line through origin with direction (1,2)")
        
        # d)
        print("\nd) span{0} = {0}")
        print("   c·0 = 0 for all scalars c")
    
    def exercise_3_linear_independence(self):
        """
        Exercise 3: Linear Independence
        
        Determine if each set is linearly independent:
        
        a) {(1,2,3), (4,5,6), (7,8,9)}
        b) {(1,0,0), (1,1,0), (1,1,1)}
        c) {(1,2), (2,1)}
        d) {1, x, x², x³} in P₃
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("Exercise 3 Solutions:")
        
        # a)
        print("\na) {(1,2,3), (4,5,6), (7,8,9)}")
        A = np.array([[1,2,3], [4,5,6], [7,8,9]]).T
        print(f"   rank = {np.linalg.matrix_rank(A)}")
        print("   DEPENDENT (rank < 3)")
        print("   Note: (7,8,9) = 2(4,5,6) - (1,2,3)")
        
        # b)
        print("\nb) {(1,0,0), (1,1,0), (1,1,1)}")
        B = np.array([[1,0,0], [1,1,0], [1,1,1]]).T
        print(f"   rank = {np.linalg.matrix_rank(B)}")
        print(f"   det = {np.linalg.det(B):.0f}")
        print("   INDEPENDENT ✓")
        
        # c)
        print("\nc) {(1,2), (2,1)}")
        C = np.array([[1,2], [2,1]]).T
        print(f"   rank = {np.linalg.matrix_rank(C)}")
        print(f"   det = {np.linalg.det(C):.0f}")
        print("   INDEPENDENT ✓")
        
        # d)
        print("\nd) {1, x, x², x³} in P₃")
        print("   These form the standard basis for P₃")
        print("   INDEPENDENT ✓ (no polynomial is a linear combo of others)")
    
    def exercise_4_basis(self):
        """
        Exercise 4: Find a Basis
        
        Find a basis and dimension for each:
        
        a) W = {(x,y,z) ∈ ℝ³ : x - y + z = 0}
        b) W = span{(1,1,0), (0,1,1), (1,2,1)}
        c) Column space of A = [[1,2,3],[2,4,5],[3,6,8]]
        d) Null space of A from part (c)
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        print("Exercise 4 Solutions:")
        
        # a)
        print("\na) W = {(x,y,z) : x - y + z = 0}")
        print("   Solve: x = y - z")
        print("   General solution: (y-z, y, z) = y(1,1,0) + z(-1,0,1)")
        print("   Basis: {(1,1,0), (-1,0,1)}")
        print("   Dimension: 2")
        
        # Verify
        b1, b2 = np.array([1,1,0]), np.array([-1,0,1])
        print(f"   Check: {b1[0]} - {b1[1]} + {b1[2]} = {b1[0]-b1[1]+b1[2]}")
        print(f"   Check: {b2[0]} - {b2[1]} + {b2[2]} = {b2[0]-b2[1]+b2[2]}")
        
        # b)
        print("\nb) W = span{(1,1,0), (0,1,1), (1,2,1)}")
        V = np.array([[1,1,0], [0,1,1], [1,2,1]]).T
        print(f"   Vectors as columns:\n{V}")
        print(f"   rank = {np.linalg.matrix_rank(V)}")
        print("   Note: (1,2,1) = (1,1,0) + (0,1,1)")
        print("   Basis: {(1,1,0), (0,1,1)}")
        print("   Dimension: 2")
        
        # c)
        print("\nc) Column space of A")
        A = np.array([[1,2,3],[2,4,5],[3,6,8]], dtype=float)
        print(f"   A =\n{A}")
        rank = np.linalg.matrix_rank(A)
        print(f"   rank(A) = {rank}")
        print("   Basis: columns 1 and 3 (pivot columns)")
        print(f"   {A[:,0]} and {A[:,2]}")
        print(f"   Dimension: {rank}")
        
        # d)
        print("\nd) Null space of A")
        null = linalg.null_space(A)
        print(f"   Null space basis:\n{null}")
        print(f"   Dimension: {null.shape[1]}")
        print(f"   Verify rank + nullity = n: {rank} + {null.shape[1]} = {A.shape[1]} ✓")
    
    # ==================== INTERMEDIATE EXERCISES ====================
    
    def exercise_5_fundamental_subspaces(self):
        """
        Exercise 5: Four Fundamental Subspaces
        
        For A = [[1, 2, 1, 0],
                 [0, 0, 1, 1],
                 [1, 2, 2, 1]]:
        
        a) Find bases for C(A), C(Aᵀ), N(A), N(Aᵀ)
        b) Verify dimensions add up correctly
        c) Verify orthogonality: C(Aᵀ) ⊥ N(A) and C(A) ⊥ N(Aᵀ)
        """
        A = np.array([[1, 2, 1, 0],
                      [0, 0, 1, 1],
                      [1, 2, 2, 1]])
        
        # Your solution
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        A = np.array([[1, 2, 1, 0],
                      [0, 0, 1, 1],
                      [1, 2, 2, 1]], dtype=float)
        
        m, n = A.shape
        r = np.linalg.matrix_rank(A)
        
        print("Exercise 5 Solution:")
        print(f"A ({m}×{n}):\n{A}")
        print(f"rank = {r}")
        
        # a) Find bases
        print("\na) Bases for fundamental subspaces:")
        
        # Column space
        print(f"\n   C(A): subspace of ℝ^{m}, dim = {r}")
        print(f"   Basis: columns 1 and 3 of A")
        print(f"   {A[:,0]}, {A[:,2]}")
        
        # Row space
        print(f"\n   C(Aᵀ): subspace of ℝ^{n}, dim = {r}")
        _, _, Vh = np.linalg.svd(A)
        for i in range(r):
            print(f"   {np.round(Vh[i], 4)}")
        
        # Null space
        null_A = linalg.null_space(A)
        print(f"\n   N(A): subspace of ℝ^{n}, dim = {n-r}")
        for i in range(null_A.shape[1]):
            print(f"   {np.round(null_A[:,i], 4)}")
        
        # Left null space
        null_At = linalg.null_space(A.T)
        print(f"\n   N(Aᵀ): subspace of ℝ^{m}, dim = {m-r}")
        for i in range(null_At.shape[1]):
            print(f"   {np.round(null_At[:,i], 4)}")
        
        # b) Verify dimensions
        print("\nb) Dimension verification:")
        print(f"   dim(C(A)) + dim(N(Aᵀ)) = {r} + {m-r} = {m} = m ✓")
        print(f"   dim(C(Aᵀ)) + dim(N(A)) = {r} + {n-r} = {n} = n ✓")
        
        # c) Orthogonality
        print("\nc) Orthogonality verification:")
        
        # Row space ⊥ Null space
        for i, row in enumerate(A):
            for j in range(null_A.shape[1]):
                dot = np.dot(row, null_A[:,j])
                print(f"   Row {i+1} · NullVec {j+1} = {dot:.6f}")
        
        # Column space ⊥ Left null space
        if null_At.shape[1] > 0:
            for i in range(A.shape[1]):
                for j in range(null_At.shape[1]):
                    dot = np.dot(A[:,i], null_At[:,j])
                    print(f"   Col {i+1} · LeftNullVec {j+1} = {dot:.6f}")
    
    def exercise_6_change_of_basis(self):
        """
        Exercise 6: Change of Basis
        
        Let B = {(1,1), (1,-1)} be a basis for ℝ².
        
        a) Find the coordinates of (3, 5) in basis B
        b) Find the coordinates of (2, -4) in basis B
        c) If [v]_B = (1, 2), what is v in standard coordinates?
        d) Find the change of basis matrix from B to standard
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solution:")
        
        B = np.array([[1, 1],
                      [1, -1]])
        
        print(f"Basis B: b1 = (1,1), b2 = (1,-1)")
        print(f"B matrix (columns are basis vectors):\n{B}")
        
        B_inv = np.linalg.inv(B)
        
        # a)
        v1 = np.array([3, 5])
        coords_v1 = B_inv @ v1
        print(f"\na) Coordinates of (3,5) in B:")
        print(f"   [v]_B = B⁻¹ @ v = {coords_v1}")
        print(f"   Verify: {coords_v1[0]}*(1,1) + {coords_v1[1]}*(1,-1)")
        print(f"         = ({coords_v1[0] + coords_v1[1]}, {coords_v1[0] - coords_v1[1]})")
        
        # b)
        v2 = np.array([2, -4])
        coords_v2 = B_inv @ v2
        print(f"\nb) Coordinates of (2,-4) in B:")
        print(f"   [v]_B = {coords_v2}")
        
        # c)
        coords_v3 = np.array([1, 2])
        v3 = B @ coords_v3
        print(f"\nc) If [v]_B = (1,2), then v = B @ [v]_B")
        print(f"   v = {v3}")
        
        # d)
        print(f"\nd) Change of basis matrix from B to standard:")
        print(f"   P_{B→E} = B =\n{B}")
        print(f"   Change of basis matrix from standard to B:")
        print(f"   P_{E→B} = B⁻¹ =\n{np.round(B_inv, 4)}")
    
    # ==================== ADVANCED EXERCISES ====================
    
    def exercise_7_direct_sum(self):
        """
        Exercise 7: Direct Sum
        
        Two subspaces U and W form a direct sum (V = U ⊕ W) if:
        - Every v ∈ V can be written uniquely as v = u + w
        - Equivalently: U ∩ W = {0} and dim(U) + dim(W) = dim(V)
        
        In ℝ³:
        a) Let U = span{(1,0,0), (0,1,0)} and W = span{(0,0,1)}.
           Show ℝ³ = U ⊕ W
        
        b) Let U = span{(1,1,0)} and W = span{(1,0,1), (0,1,1)}.
           Is ℝ³ = U ⊕ W?
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("Exercise 7 Solution:")
        
        # a)
        print("\na) U = span{(1,0,0), (0,1,0)}, W = span{(0,0,1)}")
        U = np.array([[1,0,0], [0,1,0]]).T
        W = np.array([[0,0,1]]).T
        
        print(f"   dim(U) = {np.linalg.matrix_rank(U)}")
        print(f"   dim(W) = {np.linalg.matrix_rank(W)}")
        
        # Check intersection
        UW = np.hstack([U, W])
        print(f"   dim(U) + dim(W) = 2 + 1 = 3 = dim(ℝ³) ✓")
        
        # U ∩ W = {0}?
        # v in both means v = c1(1,0,0) + c2(0,1,0) = d(0,0,1)
        # Only solution: c1 = c2 = d = 0
        print("   U ∩ W = {0}? Yes (xy-plane and z-axis meet only at origin)")
        print("   Therefore ℝ³ = U ⊕ W ✓")
        
        # b)
        print("\nb) U = span{(1,1,0)}, W = span{(1,0,1), (0,1,1)}")
        U2 = np.array([[1,1,0]]).T
        W2 = np.array([[1,0,1], [0,1,1]]).T
        
        print(f"   dim(U) = {np.linalg.matrix_rank(U2)}")
        print(f"   dim(W) = {np.linalg.matrix_rank(W2)}")
        print(f"   dim(U) + dim(W) = 1 + 2 = 3 = dim(ℝ³)")
        
        # Check if they span ℝ³
        UW2 = np.hstack([U2, W2])
        rank_combined = np.linalg.matrix_rank(UW2)
        print(f"   rank([U W]) = {rank_combined}")
        print(f"   Span ℝ³? {rank_combined == 3}")
        
        if rank_combined == 3:
            print("   Since rank = 3 and dim(U)+dim(W) = 3, we have U ∩ W = {0}")
            print("   Therefore ℝ³ = U ⊕ W ✓")
        else:
            print("   NOT a direct sum")
    
    def exercise_8_dimension_formula(self):
        """
        Exercise 8: Dimension Formula
        
        For subspaces U and W of V:
        dim(U + W) = dim(U) + dim(W) - dim(U ∩ W)
        
        Let U = span{(1,1,0,0), (0,0,1,1)}
        Let W = span{(1,0,1,0), (0,1,0,1)}
        
        a) Find dim(U), dim(W)
        b) Find U ∩ W and its dimension
        c) Find U + W and its dimension
        d) Verify the dimension formula
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("Exercise 8 Solution:")
        
        U = np.array([[1,1,0,0], [0,0,1,1]]).T
        W = np.array([[1,0,1,0], [0,1,0,1]]).T
        
        print("U = span{(1,1,0,0), (0,0,1,1)}")
        print("W = span{(1,0,1,0), (0,1,0,1)}")
        
        # a)
        dim_U = np.linalg.matrix_rank(U)
        dim_W = np.linalg.matrix_rank(W)
        print(f"\na) dim(U) = {dim_U}, dim(W) = {dim_W}")
        
        # b) U ∩ W
        # v ∈ U ∩ W means v = a(1,1,0,0) + b(0,0,1,1) = c(1,0,1,0) + d(0,1,0,1)
        # This gives: a = c, a = d, b = c, b = d
        # So a = b = c = d, and v = a(1,1,1,1)
        print("\nb) U ∩ W:")
        print("   Solving: a(1,1,0,0) + b(0,0,1,1) = c(1,0,1,0) + d(0,1,0,1)")
        print("   Leads to: a = c = d, b = c")
        print("   So a = b and v = a(1,1,1,1)")
        print("   U ∩ W = span{(1,1,1,1)}, dim = 1")
        
        # c) U + W
        UW = np.hstack([U, W])
        dim_sum = np.linalg.matrix_rank(UW)
        print(f"\nc) U + W = span of all vectors in U and W")
        print(f"   rank([U W]) = {dim_sum}")
        print(f"   dim(U + W) = {dim_sum}")
        
        # d) Verify
        dim_intersection = 1  # Calculated above
        print(f"\nd) Dimension formula verification:")
        print(f"   dim(U + W) = dim(U) + dim(W) - dim(U ∩ W)")
        print(f"   {dim_sum} = {dim_U} + {dim_W} - {dim_intersection}")
        print(f"   {dim_sum} = {dim_U + dim_W - dim_intersection} ✓")
    
    def exercise_9_quotient_space(self):
        """
        Exercise 9: Cosets and Quotient Spaces (Conceptual)
        
        For a subspace W of V, the quotient space V/W consists of cosets:
        v + W = {v + w : w ∈ W}
        
        dim(V/W) = dim(V) - dim(W)
        
        Let V = ℝ³ and W = {(x,y,z) : x + y + z = 0}
        
        a) What is dim(W) and dim(V/W)?
        b) Describe the cosets geometrically
        c) Show that (1,0,0) + W and (2,0,0) + W are different cosets
        d) Show that (1,0,0) + W and (0,1,0) + W are the same coset
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        print("Exercise 9 Solution:")
        print("V = ℝ³, W = {(x,y,z) : x + y + z = 0}")
        
        print("\na) Dimensions:")
        print("   W is a plane through origin (2D)")
        print("   dim(W) = 2")
        print("   dim(V/W) = dim(V) - dim(W) = 3 - 2 = 1")
        
        print("\nb) Cosets geometrically:")
        print("   Each coset v + W is a plane parallel to W")
        print("   W itself is the coset 0 + W")
        print("   Planes x + y + z = c for different c values")
        
        print("\nc) Are (1,0,0)+W and (2,0,0)+W different?")
        v1 = np.array([1, 0, 0])
        v2 = np.array([2, 0, 0])
        diff = v1 - v2
        in_W = np.isclose(diff[0] + diff[1] + diff[2], 0)
        print(f"   (1,0,0) - (2,0,0) = {diff}")
        print(f"   {diff} in W? {in_W}")
        print("   (-1,0,0): -1 + 0 + 0 = -1 ≠ 0, so NOT in W")
        print("   Therefore they are DIFFERENT cosets")
        
        print("\nd) Are (1,0,0)+W and (0,1,0)+W the same?")
        u1 = np.array([1, 0, 0])
        u2 = np.array([0, 1, 0])
        diff = u1 - u2
        in_W = np.isclose(diff[0] + diff[1] + diff[2], 0)
        print(f"   (1,0,0) - (0,1,0) = {diff}")
        print(f"   {diff} in W? {in_W}")
        print("   (1,-1,0): 1 + (-1) + 0 = 0, so YES in W")
        print("   Therefore they are the SAME coset")
    
    def exercise_10_isomorphism(self):
        """
        Exercise 10: Isomorphic Vector Spaces
        
        Two vector spaces are isomorphic if they have the same dimension.
        
        Show that:
        a) ℝ³ is isomorphic to P₂ (polynomials of degree ≤ 2)
        b) M₂ₓ₂ is isomorphic to ℝ⁴
        c) Define an explicit isomorphism for each
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("Exercise 10 Solution:")
        
        print("\na) ℝ³ ≅ P₂")
        print("   dim(ℝ³) = 3")
        print("   dim(P₂) = 3 (basis: {1, x, x²})")
        print("   Same dimension → isomorphic")
        print("\n   Explicit isomorphism T: ℝ³ → P₂:")
        print("   T(a, b, c) = a + bx + cx²")
        print("   Example: T(1, 2, 3) = 1 + 2x + 3x²")
        
        print("\nb) M₂ₓ₂ ≅ ℝ⁴")
        print("   dim(M₂ₓ₂) = 2×2 = 4")
        print("   dim(ℝ⁴) = 4")
        print("   Same dimension → isomorphic")
        print("\n   Explicit isomorphism T: M₂ₓ₂ → ℝ⁴:")
        print("   T([[a,b],[c,d]]) = (a, b, c, d)")
        print("   Example: T([[1,2],[3,4]]) = (1, 2, 3, 4)")
        
        # Verify linearity
        print("\n   Verifying linearity:")
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        c = 2
        
        T = lambda M: M.flatten()
        
        print(f"   T(A) = {T(A)}")
        print(f"   T(B) = {T(B)}")
        print(f"   T(A+B) = {T(A+B)}")
        print(f"   T(A)+T(B) = {T(A)+T(B)}")
        print(f"   T(cA) = {T(c*A)}")
        print(f"   cT(A) = {c*T(A)}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = VectorSpaceExercises()
    
    print("VECTOR SPACES AND SUBSPACES EXERCISES")
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
