"""
Sets and Logic - Exercises
==========================
Practice problems for set theory and logic.
"""

import numpy as np
from itertools import product


class SetsLogicExercises:
    """Exercises for sets and logic concepts."""
    
    # ==================== SET EXERCISES ====================
    
    def exercise_1_basic_operations(self):
        """
        Exercise 1: Basic Set Operations
        
        Given A = {1, 2, 3, 4, 5} and B = {4, 5, 6, 7, 8}, find:
        a) A ∪ B
        b) A ∩ B
        c) A - B
        d) B - A
        e) A △ B (symmetric difference)
        """
        A = {1, 2, 3, 4, 5}
        B = {4, 5, 6, 7, 8}
        
        # Your solutions
        union = None
        intersection = None
        diff_AB = None
        diff_BA = None
        sym_diff = None
        
        return union, intersection, diff_AB, diff_BA, sym_diff
    
    def solution_1(self):
        """Solution to Exercise 1."""
        A = {1, 2, 3, 4, 5}
        B = {4, 5, 6, 7, 8}
        
        print("Exercise 1 Solution:")
        print(f"A = {A}")
        print(f"B = {B}")
        
        print(f"\na) A ∪ B = {A | B}")
        print(f"b) A ∩ B = {A & B}")
        print(f"c) A - B = {A - B}")
        print(f"d) B - A = {B - A}")
        print(f"e) A △ B = {A ^ B}")
        
        return A | B, A & B, A - B, B - A, A ^ B
    
    def exercise_2_subset_relations(self):
        """
        Exercise 2: Subset Relations
        
        Let A = {1, 2}, B = {1, 2, 3}, C = {1, 2, 3}, D = {2, 3, 4}
        
        Determine True/False:
        a) A ⊆ B
        b) B ⊆ C
        c) B ⊂ C
        d) A ⊆ D
        e) ∅ ⊆ A
        f) A ⊆ A
        """
        pass
    
    def solution_2(self):
        """Solution to Exercise 2."""
        A = {1, 2}
        B = {1, 2, 3}
        C = {1, 2, 3}
        D = {2, 3, 4}
        
        print("Exercise 2 Solution:")
        print(f"A = {A}, B = {B}, C = {C}, D = {D}")
        
        print(f"\na) A ⊆ B: {A <= B} (A is contained in B)")
        print(f"b) B ⊆ C: {B <= C} (B equals C)")
        print(f"c) B ⊂ C: {B < C} (B must be strictly smaller)")
        print(f"d) A ⊆ D: {A <= D} ({A & D} ≠ A)")
        print(f"e) ∅ ⊆ A: {set() <= A} (empty set is subset of all)")
        print(f"f) A ⊆ A: {A <= A} (every set is subset of itself)")
    
    def exercise_3_cardinality(self):
        """
        Exercise 3: Cardinality Problems
        
        a) If |A| = 5 and |B| = 7 and |A ∩ B| = 3, find |A ∪ B|
        b) If |A| = 10 and |B| = 8 and |A ∪ B| = 15, find |A ∩ B|
        c) Find |𝒫({1, 2, 3, 4})|
        d) Find |{1, 2} × {a, b, c}|
        """
        pass
    
    def solution_3(self):
        """Solution to Exercise 3."""
        print("Exercise 3 Solution:")
        
        # a)
        A_size, B_size, intersection = 5, 7, 3
        union = A_size + B_size - intersection
        print(f"a) |A ∪ B| = |A| + |B| - |A ∩ B| = {A_size} + {B_size} - {intersection} = {union}")
        
        # b)
        A_size, B_size, union = 10, 8, 15
        intersection = A_size + B_size - union
        print(f"b) |A ∩ B| = |A| + |B| - |A ∪ B| = {A_size} + {B_size} - {union} = {intersection}")
        
        # c)
        n = 4
        power_set_size = 2 ** n
        print(f"c) |𝒫({{1,2,3,4}})| = 2^4 = {power_set_size}")
        
        # d)
        A = {1, 2}
        B = {'a', 'b', 'c'}
        cartesian_size = len(A) * len(B)
        print(f"d) |{{1,2}} × {{a,b,c}}| = 2 × 3 = {cartesian_size}")
    
    def exercise_4_demorgan(self):
        """
        Exercise 4: De Morgan's Laws
        
        Let U = {1,2,3,4,5,6,7,8,9,10}, A = {1,2,3,4,5}, B = {4,5,6,7}
        
        Verify:
        a) (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
        b) (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
        """
        pass
    
    def solution_4(self):
        """Solution to Exercise 4."""
        U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        A = {1, 2, 3, 4, 5}
        B = {4, 5, 6, 7}
        
        A_c = U - A
        B_c = U - B
        
        print("Exercise 4 Solution:")
        print(f"U = {U}")
        print(f"A = {A}, Aᶜ = {A_c}")
        print(f"B = {B}, Bᶜ = {B_c}")
        
        # a)
        lhs_a = U - (A | B)
        rhs_a = A_c & B_c
        print(f"\na) (A ∪ B)ᶜ = {lhs_a}")
        print(f"   Aᶜ ∩ Bᶜ = {rhs_a}")
        print(f"   Equal? {lhs_a == rhs_a} ✓")
        
        # b)
        lhs_b = U - (A & B)
        rhs_b = A_c | B_c
        print(f"\nb) (A ∩ B)ᶜ = {lhs_b}")
        print(f"   Aᶜ ∪ Bᶜ = {rhs_b}")
        print(f"   Equal? {lhs_b == rhs_b} ✓")
    
    # ==================== LOGIC EXERCISES ====================
    
    def exercise_5_truth_table(self):
        """
        Exercise 5: Construct Truth Table
        
        Construct the truth table for:
        (p → q) ∧ (q → r) → (p → r)
        
        (This is the hypothetical syllogism)
        """
        pass
    
    def solution_5(self):
        """Solution to Exercise 5."""
        print("Exercise 5 Solution:")
        print("(p → q) ∧ (q → r) → (p → r)")
        print("-" * 60)
        print(" p | q | r | p→q | q→r | (p→q)∧(q→r) | p→r | Result")
        print("-" * 60)
        
        impl = lambda a, b: (not a) or b
        
        for p in [True, False]:
            for q in [True, False]:
                for r in [True, False]:
                    p_impl_q = impl(p, q)
                    q_impl_r = impl(q, r)
                    premise = p_impl_q and q_impl_r
                    p_impl_r = impl(p, r)
                    result = impl(premise, p_impl_r)
                    
                    p_s, q_s, r_s = ('T' if p else 'F'), ('T' if q else 'F'), ('T' if r else 'F')
                    print(f" {p_s} | {q_s} | {r_s} |  {('T' if p_impl_q else 'F')}  |  {('T' if q_impl_r else 'F')}  |      {('T' if premise else 'F')}      |  {('T' if p_impl_r else 'F')}  |   {('T' if result else 'F')}")
        
        print("-" * 60)
        print("All T in Result column → TAUTOLOGY ✓")
    
    def exercise_6_logical_equivalence(self):
        """
        Exercise 6: Prove Logical Equivalence
        
        Show using truth tables that:
        a) p → q ≡ ¬p ∨ q
        b) ¬(p → q) ≡ p ∧ ¬q
        c) p ↔ q ≡ (p → q) ∧ (q → p)
        """
        pass
    
    def solution_6(self):
        """Solution to Exercise 6."""
        print("Exercise 6 Solution:")
        
        impl = lambda a, b: (not a) or b
        
        # a)
        print("\na) p → q ≡ ¬p ∨ q")
        print(" p | q | p→q | ¬p∨q")
        print("-" * 25)
        for p in [True, False]:
            for q in [True, False]:
                lhs = impl(p, q)
                rhs = (not p) or q
                print(f" {('T' if p else 'F')} | {('T' if q else 'F')} |  {('T' if lhs else 'F')}  |  {('T' if rhs else 'F')}")
        print("Same columns → Equivalent ✓")
        
        # b)
        print("\nb) ¬(p → q) ≡ p ∧ ¬q")
        print(" p | q | ¬(p→q) | p∧¬q")
        print("-" * 28)
        for p in [True, False]:
            for q in [True, False]:
                lhs = not impl(p, q)
                rhs = p and (not q)
                print(f" {('T' if p else 'F')} | {('T' if q else 'F')} |   {('T' if lhs else 'F')}   |  {('T' if rhs else 'F')}")
        print("Same columns → Equivalent ✓")
        
        # c)
        print("\nc) p ↔ q ≡ (p → q) ∧ (q → p)")
        print(" p | q | p↔q | (p→q)∧(q→p)")
        print("-" * 30)
        for p in [True, False]:
            for q in [True, False]:
                lhs = p == q
                rhs = impl(p, q) and impl(q, p)
                print(f" {('T' if p else 'F')} | {('T' if q else 'F')} |  {('T' if lhs else 'F')}  |      {('T' if rhs else 'F')}")
        print("Same columns → Equivalent ✓")
    
    def exercise_7_negate_statements(self):
        """
        Exercise 7: Negate Statements
        
        Write the negation of each statement:
        a) All students passed the exam.
        b) There exists a prime number greater than 100.
        c) For every ε > 0, there exists δ > 0 such that |f(x) - L| < ε.
        d) If it rains, then the ground is wet.
        """
        pass
    
    def solution_7(self):
        """Solution to Exercise 7."""
        print("Exercise 7 Solution:")
        
        print("\na) 'All students passed the exam'")
        print("   Original: ∀x (Student(x) → Passed(x))")
        print("   Negation: ∃x (Student(x) ∧ ¬Passed(x))")
        print("   English: 'There exists a student who did not pass.'")
        
        print("\nb) 'There exists a prime number greater than 100'")
        print("   Original: ∃x (Prime(x) ∧ x > 100)")
        print("   Negation: ∀x (Prime(x) → x ≤ 100)")
        print("   English: 'All prime numbers are at most 100.'")
        
        print("\nc) Limit definition: ∀ε>0 ∃δ>0 (|x-a|<δ → |f(x)-L|<ε)")
        print("   Negation: ∃ε>0 ∀δ>0 ∃x (|x-a|<δ ∧ |f(x)-L|≥ε)")
        print("   English: 'There exists ε>0 such that for all δ>0,")
        print("            there exists x with |x-a|<δ but |f(x)-L|≥ε.'")
        
        print("\nd) 'If it rains, then the ground is wet'")
        print("   Original: p → q")
        print("   Negation: ¬(p → q) ≡ p ∧ ¬q")
        print("   English: 'It rains and the ground is not wet.'")
    
    def exercise_8_quantifier_order(self):
        """
        Exercise 8: Quantifier Order
        
        Determine if each statement is True or False for x, y ∈ ℝ:
        a) ∀x ∃y (x + y = 0)
        b) ∃y ∀x (x + y = 0)
        c) ∀x ∀y (x + y = y + x)
        d) ∃x ∃y (x + y = 5)
        e) ∀x ∃y (xy = 1)
        """
        pass
    
    def solution_8(self):
        """Solution to Exercise 8."""
        print("Exercise 8 Solution:")
        
        print("\na) ∀x ∃y (x + y = 0)")
        print("   TRUE: For any x, choose y = -x")
        
        print("\nb) ∃y ∀x (x + y = 0)")
        print("   FALSE: No single y works for all x")
        print("   (Would need y = -x for all x, but y is fixed)")
        
        print("\nc) ∀x ∀y (x + y = y + x)")
        print("   TRUE: Addition is commutative")
        
        print("\nd) ∃x ∃y (x + y = 5)")
        print("   TRUE: x = 2, y = 3 works")
        
        print("\ne) ∀x ∃y (xy = 1)")
        print("   FALSE: When x = 0, no y satisfies 0·y = 1")
    
    # ==================== APPLICATION EXERCISES ====================
    
    def exercise_9_classification_sets(self):
        """
        Exercise 9: Classification with Sets
        
        Given predictions and actual labels:
        Actual Positive: {1, 3, 5, 7, 9, 11, 13}
        Actual Negative: {2, 4, 6, 8, 10, 12, 14}
        Predicted Positive: {1, 2, 3, 4, 5, 6, 7}
        
        Calculate:
        a) TP, FP, FN, TN (as sets and their cardinalities)
        b) Precision
        c) Recall
        d) Accuracy
        """
        pass
    
    def solution_9(self):
        """Solution to Exercise 9."""
        actual_pos = {1, 3, 5, 7, 9, 11, 13}
        actual_neg = {2, 4, 6, 8, 10, 12, 14}
        pred_pos = {1, 2, 3, 4, 5, 6, 7}
        pred_neg = {8, 9, 10, 11, 12, 13, 14}
        
        print("Exercise 9 Solution:")
        print(f"Actual Positive: {actual_pos}")
        print(f"Actual Negative: {actual_neg}")
        print(f"Predicted Positive: {pred_pos}")
        print(f"Predicted Negative: {pred_neg}")
        
        # a)
        TP = pred_pos & actual_pos
        FP = pred_pos & actual_neg
        FN = pred_neg & actual_pos
        TN = pred_neg & actual_neg
        
        print(f"\na) TP = Pred+ ∩ Act+ = {TP}, |TP| = {len(TP)}")
        print(f"   FP = Pred+ ∩ Act- = {FP}, |FP| = {len(FP)}")
        print(f"   FN = Pred- ∩ Act+ = {FN}, |FN| = {len(FN)}")
        print(f"   TN = Pred- ∩ Act- = {TN}, |TN| = {len(TN)}")
        
        # b)
        precision = len(TP) / len(pred_pos)
        print(f"\nb) Precision = |TP|/|Pred+| = {len(TP)}/{len(pred_pos)} = {precision:.3f}")
        
        # c)
        recall = len(TP) / len(actual_pos)
        print(f"c) Recall = |TP|/|Act+| = {len(TP)}/{len(actual_pos)} = {recall:.3f}")
        
        # d)
        total = len(actual_pos) + len(actual_neg)
        accuracy = (len(TP) + len(TN)) / total
        print(f"d) Accuracy = (|TP|+|TN|)/Total = ({len(TP)}+{len(TN)})/{total} = {accuracy:.3f}")
    
    def exercise_10_database_queries(self):
        """
        Exercise 10: Database Queries
        
        Express each SQL-like query using set operations:
        
        Table Users: {(1,'Alice',25), (2,'Bob',30), (3,'Carol',25), (4,'Dave',35)}
        Table Orders: {(1,100), (1,200), (2,150), (3,300)}
        
        a) Users who have placed orders
        b) Users who have NOT placed orders
        c) Users aged 25 OR who have orders > 200
        """
        pass
    
    def solution_10(self):
        """Solution to Exercise 10."""
        print("Exercise 10 Solution:")
        
        # Represent as sets
        users = {
            1: ('Alice', 25),
            2: ('Bob', 30),
            3: ('Carol', 25),
            4: ('Dave', 35)
        }
        orders = [(1, 100), (1, 200), (2, 150), (3, 300)]
        
        all_user_ids = set(users.keys())
        users_with_orders = set(order[0] for order in orders)
        
        print("Users:", users)
        print("Orders:", orders)
        print(f"\nAll user IDs: {all_user_ids}")
        print(f"Users with orders: {users_with_orders}")
        
        # a)
        print(f"\na) Users who have placed orders:")
        print(f"   All_Users ∩ Order_Users = {all_user_ids & users_with_orders}")
        result_a = {uid: users[uid] for uid in (all_user_ids & users_with_orders)}
        print(f"   Result: {result_a}")
        
        # b)
        print(f"\nb) Users who have NOT placed orders:")
        print(f"   All_Users - Order_Users = {all_user_ids - users_with_orders}")
        result_b = {uid: users[uid] for uid in (all_user_ids - users_with_orders)}
        print(f"   Result: {result_b}")
        
        # c)
        users_age_25 = {uid for uid, (name, age) in users.items() if age == 25}
        users_orders_200 = {order[0] for order in orders if order[1] > 200}
        
        print(f"\nc) Users aged 25 OR with orders > 200:")
        print(f"   Age_25: {users_age_25}")
        print(f"   Orders>200: {users_orders_200}")
        print(f"   Age_25 ∪ Orders>200 = {users_age_25 | users_orders_200}")
        result_c = {uid: users[uid] for uid in (users_age_25 | users_orders_200)}
        print(f"   Result: {result_c}")


def run_all_exercises():
    """Run all exercises with solutions."""
    exercises = SetsLogicExercises()
    
    print("SETS AND LOGIC EXERCISES")
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
