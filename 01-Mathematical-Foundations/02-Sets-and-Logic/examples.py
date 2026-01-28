"""
Sets and Logic - Examples
=========================
Practical demonstrations of set theory and logic.
"""

import numpy as np
from itertools import product, combinations


def example_basic_sets():
    """Basic set operations in Python."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Set Operations")
    print("=" * 60)
    
    A = {1, 2, 3, 4, 5}
    B = {4, 5, 6, 7, 8}
    
    print(f"A = {A}")
    print(f"B = {B}")
    
    # Union
    print(f"\nA ∪ B = {A | B}")
    print(f"       = {A.union(B)}")
    
    # Intersection
    print(f"\nA ∩ B = {A & B}")
    print(f"       = {A.intersection(B)}")
    
    # Difference
    print(f"\nA - B = {A - B}")
    print(f"       = {A.difference(B)}")
    
    print(f"\nB - A = {B - A}")
    
    # Symmetric difference
    print(f"\nA △ B = {A ^ B}")
    print(f"       = {A.symmetric_difference(B)}")


def example_subset_superset():
    """Demonstrate subset relationships."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Subset and Superset")
    print("=" * 60)
    
    A = {1, 2, 3}
    B = {1, 2, 3, 4, 5}
    C = {1, 2, 3}
    
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"C = {C}")
    
    print(f"\nA ⊆ B? {A.issubset(B)}")
    print(f"A ⊂ B (proper)? {A < B}")
    
    print(f"\nB ⊇ A? {B.issuperset(A)}")
    print(f"B ⊃ A (proper)? {B > A}")
    
    print(f"\nA ⊆ C? {A.issubset(C)}")
    print(f"A ⊂ C (proper)? {A < C}")
    print(f"A = C? {A == C}")
    
    # Empty set is subset of all sets
    print(f"\n∅ ⊆ A? {set().issubset(A)}")


def example_cardinality_formulas():
    """Verify cardinality formulas."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Cardinality Formulas")
    print("=" * 60)
    
    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6, 7}
    
    print(f"A = {A}, |A| = {len(A)}")
    print(f"B = {B}, |B| = {len(B)}")
    
    # Inclusion-exclusion
    union = A | B
    intersection = A & B
    
    print(f"\n|A ∪ B| = {len(union)}")
    print(f"|A| + |B| - |A ∩ B| = {len(A)} + {len(B)} - {len(intersection)} = {len(A) + len(B) - len(intersection)}")
    print("Inclusion-exclusion verified ✓")
    
    # Cartesian product
    cartesian = list(product(A, B))
    print(f"\n|A × B| = {len(cartesian)}")
    print(f"|A| × |B| = {len(A)} × {len(B)} = {len(A) * len(B)}")
    print("First few pairs:", cartesian[:5])


def example_power_set():
    """Generate and analyze power set."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Power Set")
    print("=" * 60)
    
    A = {1, 2, 3}
    print(f"A = {A}")
    
    # Generate power set
    def power_set(s):
        s_list = list(s)
        result = [set()]
        for elem in s_list:
            result = result + [subset | {elem} for subset in result]
        return result
    
    P_A = power_set(A)
    print(f"\n𝒫(A) = {P_A}")
    print(f"|𝒫(A)| = {len(P_A)}")
    print(f"2^|A| = 2^{len(A)} = {2**len(A)}")
    
    # Organize by size
    print("\nOrganized by cardinality:")
    for k in range(len(A) + 1):
        subsets_k = [s for s in P_A if len(s) == k]
        print(f"  Size {k}: {subsets_k} (count: {len(subsets_k)})")


def example_demorgan_sets():
    """Verify De Morgan's Laws for sets."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: De Morgan's Laws (Sets)")
    print("=" * 60)
    
    # Universe
    U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}
    
    print(f"U = {U}")
    print(f"A = {A}")
    print(f"B = {B}")
    
    # Complements
    A_comp = U - A
    B_comp = U - B
    
    print(f"\nAᶜ = {A_comp}")
    print(f"Bᶜ = {B_comp}")
    
    # De Morgan's Law 1: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
    lhs1 = U - (A | B)
    rhs1 = A_comp & B_comp
    
    print(f"\n(A ∪ B)ᶜ = {lhs1}")
    print(f"Aᶜ ∩ Bᶜ = {rhs1}")
    print(f"Equal? {lhs1 == rhs1} ✓")
    
    # De Morgan's Law 2: (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
    lhs2 = U - (A & B)
    rhs2 = A_comp | B_comp
    
    print(f"\n(A ∩ B)ᶜ = {lhs2}")
    print(f"Aᶜ ∪ Bᶜ = {rhs2}")
    print(f"Equal? {lhs2 == rhs2} ✓")


def example_truth_tables():
    """Generate truth tables for logical expressions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Truth Tables")
    print("=" * 60)
    
    print("Basic Connectives:")
    print("-" * 40)
    print("  p  |  q  | p∧q | p∨q | p→q | p↔q")
    print("-" * 40)
    
    for p in [True, False]:
        for q in [True, False]:
            p_and_q = p and q
            p_or_q = p or q
            p_implies_q = (not p) or q  # p → q ≡ ¬p ∨ q
            p_iff_q = p == q
            
            p_str = "T" if p else "F"
            q_str = "T" if q else "F"
            
            print(f"  {p_str}  |  {q_str}  |  {('T' if p_and_q else 'F')}  |  {('T' if p_or_q else 'F')}  |  {('T' if p_implies_q else 'F')}  |  {('T' if p_iff_q else 'F')}")
    
    # Verify logical equivalence
    print("\n--- Verifying p → q ≡ ¬p ∨ q ---")
    print("  p  |  q  | p→q | ¬p∨q")
    print("-" * 30)
    
    for p in [True, False]:
        for q in [True, False]:
            impl = (not p) or q
            disj = (not p) or q
            p_str = "T" if p else "F"
            q_str = "T" if q else "F"
            print(f"  {p_str}  |  {q_str}  |  {('T' if impl else 'F')}  |  {('T' if disj else 'F')}")
    
    print("Same column → logically equivalent ✓")


def example_demorgan_logic():
    """Verify De Morgan's Laws for logic."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: De Morgan's Laws (Logic)")
    print("=" * 60)
    
    print("¬(p ∧ q) ≡ ¬p ∨ ¬q")
    print("-" * 45)
    print("  p  |  q  | ¬(p∧q) | ¬p∨¬q")
    print("-" * 45)
    
    for p in [True, False]:
        for q in [True, False]:
            lhs = not (p and q)
            rhs = (not p) or (not q)
            p_str = "T" if p else "F"
            q_str = "T" if q else "F"
            print(f"  {p_str}  |  {q_str}  |   {('T' if lhs else 'F')}    |   {('T' if rhs else 'F')}")
    
    print("\n¬(p ∨ q) ≡ ¬p ∧ ¬q")
    print("-" * 45)
    print("  p  |  q  | ¬(p∨q) | ¬p∧¬q")
    print("-" * 45)
    
    for p in [True, False]:
        for q in [True, False]:
            lhs = not (p or q)
            rhs = (not p) and (not q)
            p_str = "T" if p else "F"
            q_str = "T" if q else "F"
            print(f"  {p_str}  |  {q_str}  |   {('T' if lhs else 'F')}    |   {('T' if rhs else 'F')}")


def example_contrapositive():
    """Demonstrate contrapositive equivalence."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Contrapositive")
    print("=" * 60)
    
    print("Statement: p → q")
    print("Converse: q → p")
    print("Inverse: ¬p → ¬q")
    print("Contrapositive: ¬q → ¬p")
    
    print("\n" + "-" * 55)
    print("  p  |  q  | p→q | q→p | ¬p→¬q | ¬q→¬p")
    print("-" * 55)
    
    for p in [True, False]:
        for q in [True, False]:
            impl = lambda a, b: (not a) or b
            
            original = impl(p, q)
            converse = impl(q, p)
            inverse = impl(not p, not q)
            contrapositive = impl(not q, not p)
            
            p_str = "T" if p else "F"
            q_str = "T" if q else "F"
            
            print(f"  {p_str}  |  {q_str}  |  {('T' if original else 'F')}  |  {('T' if converse else 'F')}  |   {('T' if inverse else 'F')}   |   {('T' if contrapositive else 'F')}")
    
    print("\nNote: p→q has same column as ¬q→¬p (contrapositive)")
    print("      q→p has same column as ¬p→¬q (converse ≡ inverse)")


def example_quantifiers():
    """Demonstrate quantified statements."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Quantifiers")
    print("=" * 60)
    
    # Domain
    domain = list(range(1, 11))
    print(f"Domain: {domain}")
    
    # Universal quantifier
    def P(x):
        return x > 0
    
    def Q(x):
        return x % 2 == 0
    
    def R(x):
        return x < 20
    
    print(f"\nP(x): x > 0")
    print(f"Q(x): x is even")
    print(f"R(x): x < 20")
    
    # ∀x P(x)
    all_P = all(P(x) for x in domain)
    print(f"\n∀x P(x): {all_P} (all elements > 0)")
    
    # ∃x Q(x)
    exists_Q = any(Q(x) for x in domain)
    print(f"∃x Q(x): {exists_Q} (some element is even)")
    
    # ∀x R(x)
    all_R = all(R(x) for x in domain)
    print(f"∀x R(x): {all_R} (all elements < 20)")
    
    # Negation examples
    print("\n--- Negation of Quantifiers ---")
    
    # ¬(∀x P(x)) ≡ ∃x ¬P(x)
    not_all_P = not all(P(x) for x in domain)
    exists_not_P = any(not P(x) for x in domain)
    print(f"¬(∀x P(x)) = {not_all_P}")
    print(f"∃x ¬P(x) = {exists_not_P}")
    
    # Try with Q
    not_all_Q = not all(Q(x) for x in domain)
    exists_not_Q = any(not Q(x) for x in domain)
    print(f"\n¬(∀x Q(x)) = {not_all_Q}")
    print(f"∃x ¬Q(x) = {exists_not_Q}")
    print(f"Counterexample: x = {[x for x in domain if not Q(x)][0]}")


def example_numpy_boolean():
    """Boolean operations with NumPy arrays."""
    print("\n" + "=" * 60)
    print("EXAMPLE 10: NumPy Boolean Operations")
    print("=" * 60)
    
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Array: {arr}")
    
    # Conditions
    cond_A = arr > 3
    cond_B = arr < 8
    
    print(f"\nA: arr > 3  → {cond_A}")
    print(f"B: arr < 8  → {cond_B}")
    
    # Logical operations
    print(f"\nA AND B (3 < x < 8): {cond_A & cond_B}")
    print(f"Elements: {arr[cond_A & cond_B]}")
    
    print(f"\nA OR B (x > 3 or x < 8): {cond_A | cond_B}")
    print(f"Elements: {arr[cond_A | cond_B]}")
    
    print(f"\nNOT A (x ≤ 3): {~cond_A}")
    print(f"Elements: {arr[~cond_A]}")
    
    # XOR (exclusive or)
    print(f"\nA XOR B: {cond_A ^ cond_B}")
    print(f"Elements: {arr[cond_A ^ cond_B]}")


def example_classification_metrics():
    """Classification metrics using set theory."""
    print("\n" + "=" * 60)
    print("EXAMPLE 11: Classification Metrics as Sets")
    print("=" * 60)
    
    # Simulated predictions
    np.random.seed(42)
    n = 100
    
    # Actual and predicted labels
    actual = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    predicted = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    
    # Define sets (indices)
    actual_positive = set(np.where(actual == 1)[0])
    actual_negative = set(np.where(actual == 0)[0])
    pred_positive = set(np.where(predicted == 1)[0])
    pred_negative = set(np.where(predicted == 0)[0])
    
    print(f"|Actual Positive| = {len(actual_positive)}")
    print(f"|Actual Negative| = {len(actual_negative)}")
    print(f"|Predicted Positive| = {len(pred_positive)}")
    print(f"|Predicted Negative| = {len(pred_negative)}")
    
    # Confusion matrix via set operations
    TP = pred_positive & actual_positive
    FP = pred_positive & actual_negative
    FN = pred_negative & actual_positive
    TN = pred_negative & actual_negative
    
    print(f"\nConfusion Matrix (set cardinalities):")
    print(f"  TP (Pred+ ∩ Act+) = {len(TP)}")
    print(f"  FP (Pred+ ∩ Act-) = {len(FP)}")
    print(f"  FN (Pred- ∩ Act+) = {len(FN)}")
    print(f"  TN (Pred- ∩ Act-) = {len(TN)}")
    
    # Metrics
    precision = len(TP) / len(pred_positive) if pred_positive else 0
    recall = len(TP) / len(actual_positive) if actual_positive else 0
    accuracy = (len(TP) + len(TN)) / n
    
    print(f"\nMetrics:")
    print(f"  Precision = |TP|/|Pred+| = {len(TP)}/{len(pred_positive)} = {precision:.3f}")
    print(f"  Recall = |TP|/|Act+| = {len(TP)}/{len(actual_positive)} = {recall:.3f}")
    print(f"  Accuracy = |TP∪TN|/|All| = {len(TP)+len(TN)}/{n} = {accuracy:.3f}")


def example_feature_sets():
    """Feature selection using set operations."""
    print("\n" + "=" * 60)
    print("EXAMPLE 12: Feature Selection with Sets")
    print("=" * 60)
    
    # Feature sets from different sources
    features_A = {'age', 'income', 'education', 'occupation', 'zipcode'}
    features_B = {'age', 'income', 'credit_score', 'debt', 'employment'}
    features_C = {'age', 'income', 'marital_status', 'children'}
    
    print("Dataset A features:", features_A)
    print("Dataset B features:", features_B)
    print("Dataset C features:", features_C)
    
    # Common features (can use for all datasets)
    common = features_A & features_B & features_C
    print(f"\nCommon features (A ∩ B ∩ C): {common}")
    
    # All available features
    all_features = features_A | features_B | features_C
    print(f"All features (A ∪ B ∪ C): {all_features}")
    
    # Unique to each
    unique_A = features_A - (features_B | features_C)
    unique_B = features_B - (features_A | features_C)
    unique_C = features_C - (features_A | features_B)
    
    print(f"\nUnique to A: {unique_A}")
    print(f"Unique to B: {unique_B}")
    print(f"Unique to C: {unique_C}")
    
    # Features in exactly two datasets
    in_AB_only = (features_A & features_B) - features_C
    in_AC_only = (features_A & features_C) - features_B
    in_BC_only = (features_B & features_C) - features_A
    
    print(f"\nIn A and B only: {in_AB_only}")
    print(f"In A and C only: {in_AC_only}")
    print(f"In B and C only: {in_BC_only}")


if __name__ == "__main__":
    example_basic_sets()
    example_subset_superset()
    example_cardinality_formulas()
    example_power_set()
    example_demorgan_sets()
    example_truth_tables()
    example_demorgan_logic()
    example_contrapositive()
    example_quantifiers()
    example_numpy_boolean()
    example_classification_metrics()
    example_feature_sets()
