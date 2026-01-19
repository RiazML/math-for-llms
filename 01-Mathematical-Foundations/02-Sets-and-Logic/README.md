# Sets and Logic

## Introduction

Set theory and mathematical logic form the foundational language of mathematics and computer science. Understanding sets is essential for probability theory, while logical reasoning underpins algorithm design and formal proofs in ML theory.

## Prerequisites

- Basic algebra
- Familiarity with mathematical notation

## Learning Objectives

1. Master set notation and operations
2. Understand logical connectives and quantifiers
3. Apply set theory to probability and data structures
4. Use logical reasoning in proofs and algorithms

---

## 1. Set Basics

### Definition

A **set** is an unordered collection of distinct objects called **elements** or **members**.

### Notation

$$x \in A \quad \text{(x is an element of A)}$$
$$x \notin A \quad \text{(x is not an element of A)}$$

### Set Representations

```
Roster (List) Notation:
A = {1, 2, 3, 4, 5}
B = {a, b, c}
C = {red, green, blue}

Set-Builder Notation:
A = {x : x is a positive integer less than 6}
A = {x ∈ ℤ : 1 ≤ x ≤ 5}
B = {x² : x ∈ {1,2,3}} = {1, 4, 9}
```

### Special Sets

| Symbol  | Name             | Definition                       |
| ------- | ---------------- | -------------------------------- |
| ∅ or {} | Empty set        | Set with no elements             |
| ℕ       | Natural numbers  | {1, 2, 3, ...} or {0, 1, 2, ...} |
| ℤ       | Integers         | {..., -2, -1, 0, 1, 2, ...}      |
| ℚ       | Rational numbers | {p/q : p, q ∈ ℤ, q ≠ 0}          |
| ℝ       | Real numbers     | All points on number line        |
| ℂ       | Complex numbers  | {a + bi : a, b ∈ ℝ}              |

### Cardinality

The **cardinality** |A| is the number of elements in set A.

$$|{1, 2, 3}| = 3$$
$$|\emptyset| = 0$$
$$|\mathbb{N}| = \aleph_0 \text{ (countably infinite)}$$

---

## 2. Set Relationships

### Subset and Superset

$$A \subseteq B \iff \forall x (x \in A \Rightarrow x \in B)$$

```
A ⊆ B : A is a subset of B (A may equal B)
A ⊂ B : A is a proper subset (A ≠ B)
A ⊇ B : A is a superset of B
A ⊃ B : A is a proper superset

Example:
{1, 2} ⊂ {1, 2, 3}
{1, 2, 3} ⊆ {1, 2, 3}
∅ ⊆ A (for any set A)
```

### Set Equality

$$A = B \iff (A \subseteq B) \land (B \subseteq A)$$

### Venn Diagrams

```
Union (A ∪ B):                    Intersection (A ∩ B):
┌─────────────────────┐          ┌─────────────────────┐
│    ┌─────┬─────┐    │          │    ┌─────┬─────┐    │
│    │█████│█████│    │          │    │     │█████│    │
│    │█████│█████│    │          │    │     │█████│    │
│    │█████│█████│    │          │    │     │█████│    │
│    └─────┴─────┘    │          │    └─────┴─────┘    │
│       A     B       │          │       A     B       │
└─────────────────────┘          └─────────────────────┘

Difference (A - B):              Complement (Aᶜ):
┌─────────────────────┐          ┌█████████████████████┐
│    ┌─────┬─────┐    │          │████┌─────┐█████████│
│    │█████│     │    │          │████│     │█████████│
│    │█████│     │    │          │████│     │█████████│
│    │█████│     │    │          │████│     │█████████│
│    └─────┴─────┘    │          │████└─────┘█████████│
│       A     B       │          │       A    Universe │
└─────────────────────┘          └─────────────────────┘
```

---

## 3. Set Operations

### Basic Operations

| Operation            | Notation       | Definition            |
| -------------------- | -------------- | --------------------- |
| Union                | A ∪ B          | {x : x ∈ A or x ∈ B}  |
| Intersection         | A ∩ B          | {x : x ∈ A and x ∈ B} |
| Difference           | A - B or A \ B | {x : x ∈ A and x ∉ B} |
| Complement           | Aᶜ or A'       | {x ∈ U : x ∉ A}       |
| Symmetric Difference | A △ B          | (A - B) ∪ (B - A)     |

### Properties of Set Operations

```
Commutative:
A ∪ B = B ∪ A
A ∩ B = B ∩ A

Associative:
(A ∪ B) ∪ C = A ∪ (B ∪ C)
(A ∩ B) ∩ C = A ∩ (B ∩ C)

Distributive:
A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)

De Morgan's Laws:
(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ

Identity:
A ∪ ∅ = A
A ∩ U = A

Complement:
A ∪ Aᶜ = U
A ∩ Aᶜ = ∅
```

### Cartesian Product

$$A \times B = \{(a, b) : a \in A, b \in B\}$$

```
Example:
A = {1, 2}, B = {a, b}
A × B = {(1,a), (1,b), (2,a), (2,b)}

|A × B| = |A| · |B|
```

### Power Set

$$\mathcal{P}(A) = \{S : S \subseteq A\}$$

```
Example:
A = {1, 2, 3}
𝒫(A) = {∅, {1}, {2}, {3}, {1,2}, {1,3}, {2,3}, {1,2,3}}

|𝒫(A)| = 2^|A|
```

---

## 4. Propositional Logic

### Propositions

A **proposition** is a statement that is either true (T) or false (F).

```
Propositions:                    Not Propositions:
"2 + 2 = 4" (True)              "What time is it?"
"Paris is in Germany" (False)    "x + 1 = 5" (depends on x)
"π > 3" (True)                   "Close the door."
```

### Logical Connectives

| Name          | Symbol     | Meaning            |
| ------------- | ---------- | ------------------ |
| Negation      | ¬p, ~p, p' | NOT p              |
| Conjunction   | p ∧ q      | p AND q            |
| Disjunction   | p ∨ q      | p OR q             |
| Implication   | p → q      | IF p THEN q        |
| Biconditional | p ↔ q      | p IF AND ONLY IF q |

### Truth Tables

```
Negation (¬):        Conjunction (∧):      Disjunction (∨):
┌───┬────┐          ┌───┬───┬───────┐     ┌───┬───┬───────┐
│ p │ ¬p │          │ p │ q │ p ∧ q │     │ p │ q │ p ∨ q │
├───┼────┤          ├───┼───┼───────┤     ├───┼───┼───────┤
│ T │ F  │          │ T │ T │   T   │     │ T │ T │   T   │
│ F │ T  │          │ T │ F │   F   │     │ T │ F │   T   │
└───┴────┘          │ F │ T │   F   │     │ F │ T │   T   │
                    │ F │ F │   F   │     │ F │ F │   F   │
                    └───┴───┴───────┘     └───┴───┴───────┘

Implication (→):               Biconditional (↔):
┌───┬───┬───────┐             ┌───┬───┬───────┐
│ p │ q │ p → q │             │ p │ q │ p ↔ q │
├───┼───┼───────┤             ├───┼───┼───────┤
│ T │ T │   T   │             │ T │ T │   T   │
│ T │ F │   F   │             │ T │ F │   F   │
│ F │ T │   T   │             │ F │ T │   F   │
│ F │ F │   T   │             │ F │ F │   T   │
└───┴───┴───────┘             └───┴───┴───────┘
```

### Implication Terminology

For p → q:

- **p** is the hypothesis/antecedent
- **q** is the conclusion/consequent
- **Converse**: q → p
- **Inverse**: ¬p → ¬q
- **Contrapositive**: ¬q → ¬p (logically equivalent to p → q)

---

## 5. Logical Equivalences

### Important Equivalences

```
Double Negation:
¬(¬p) ≡ p

De Morgan's Laws:
¬(p ∧ q) ≡ ¬p ∨ ¬q
¬(p ∨ q) ≡ ¬p ∧ ¬q

Implication:
p → q ≡ ¬p ∨ q
p → q ≡ ¬q → ¬p (contrapositive)

Biconditional:
p ↔ q ≡ (p → q) ∧ (q → p)

Distributive:
p ∧ (q ∨ r) ≡ (p ∧ q) ∨ (p ∧ r)
p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r)
```

### Tautology and Contradiction

- **Tautology**: Always true (e.g., p ∨ ¬p)
- **Contradiction**: Always false (e.g., p ∧ ¬p)
- **Contingency**: Sometimes true, sometimes false

---

## 6. Quantifiers

### Universal Quantifier (∀)

$$\forall x \, P(x) \quad \text{means "for all x, P(x) is true"}$$

### Existential Quantifier (∃)

$$\exists x \, P(x) \quad \text{means "there exists an x such that P(x) is true"}$$

### Negating Quantifiers

$$\neg(\forall x \, P(x)) \equiv \exists x \, \neg P(x)$$
$$\neg(\exists x \, P(x)) \equiv \forall x \, \neg P(x)$$

```
Example:
Statement: "All birds can fly"
∀x (Bird(x) → CanFly(x))

Negation: "There exists a bird that cannot fly"
∃x (Bird(x) ∧ ¬CanFly(x))
```

### Nested Quantifiers

$$\forall x \, \exists y \, P(x, y) \neq \exists y \, \forall x \, P(x, y)$$

```
∀x ∃y (x + y = 0): For every x, there exists a y such that x + y = 0
                   TRUE (y = -x works for any x)

∃y ∀x (x + y = 0): There exists a y such that for all x, x + y = 0
                   FALSE (no single y works for all x)
```

---

## 7. Applications in ML/AI

### 1. Probability Theory

Sets form the foundation of probability:

- **Sample space** Ω: Set of all possible outcomes
- **Event**: Subset of sample space
- **P(A ∪ B) = P(A) + P(B) - P(A ∩ B)**

```
Coin flip: Ω = {H, T}
Two coins: Ω = {HH, HT, TH, TT}
Event "at least one head": A = {HH, HT, TH}
```

### 2. Database Queries (SQL)

```sql
-- Set operations in SQL
SELECT * FROM A UNION SELECT * FROM B      -- A ∪ B
SELECT * FROM A INTERSECT SELECT * FROM B  -- A ∩ B
SELECT * FROM A EXCEPT SELECT * FROM B     -- A - B

-- Logical conditions
SELECT * FROM users WHERE age > 18 AND country = 'US'  -- ∧
SELECT * FROM users WHERE age < 18 OR age > 65         -- ∨
```

### 3. Boolean Indexing

```python
# NumPy/Pandas use logical operations
import numpy as np
arr = np.array([1, 2, 3, 4, 5])

# Logical AND
mask = (arr > 2) & (arr < 5)  # [False, False, True, True, False]

# Logical OR
mask = (arr < 2) | (arr > 4)  # [True, False, False, False, True]

# Logical NOT
mask = ~(arr == 3)  # [True, True, False, True, True]
```

### 4. Feature Selection

```
Features as sets:
All features: U = {f1, f2, f3, ..., fn}
Selected features: S ⊆ U
Dropped features: U - S

Feature intersection:
Common features across datasets:
Features_A ∩ Features_B
```

### 5. Classification Metrics

```
True Positives:  TP = Predicted_Positive ∩ Actual_Positive
False Positives: FP = Predicted_Positive ∩ Actual_Negative
False Negatives: FN = Predicted_Negative ∩ Actual_Positive
True Negatives:  TN = Predicted_Negative ∩ Actual_Negative

Precision = |TP| / |Predicted_Positive|
Recall = |TP| / |Actual_Positive|
```

### 6. Logical Rules in Expert Systems

```
Rule-based systems use logical implications:
IF fever AND cough THEN possible_flu
p ∧ q → r

Chaining rules:
(p → q) ∧ (q → r) ⊢ (p → r)
```

---

## 8. Summary

### Set Operations Table

| Operation         | Notation | Result                   |
| ----------------- | -------- | ------------------------ |
| Union             | A ∪ B    | Elements in A or B       |
| Intersection      | A ∩ B    | Elements in both A and B |
| Difference        | A - B    | Elements in A but not B  |
| Complement        | Aᶜ       | Elements not in A        |
| Cartesian Product | A × B    | All ordered pairs        |
| Power Set         | 𝒫(A)     | All subsets of A         |

### Logic Summary

| Connective | Symbol | True When         |
| ---------- | ------ | ----------------- |
| AND        | ∧      | Both true         |
| OR         | ∨      | At least one true |
| NOT        | ¬      | Operand is false  |
| IF-THEN    | →      | Not (T → F)       |
| IFF        | ↔      | Both same         |

### Key Formulas

$$|A \cup B| = |A| + |B| - |A \cap B|$$
$$|A \times B| = |A| \cdot |B|$$
$$|\mathcal{P}(A)| = 2^{|A|}$$

---

## Exercises

1. Given A = {1,2,3,4} and B = {3,4,5,6}, find A∪B, A∩B, A-B, A△B
2. Prove De Morgan's Law: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
3. Construct truth table for (p → q) ∧ (q → r) → (p → r)
4. Negate: "For all ε > 0, there exists δ > 0 such that |f(x) - L| < ε"
5. Express classification metrics using set notation

---

## References

1. Rosen, K. - "Discrete Mathematics and Its Applications"
2. Halmos, P. - "Naive Set Theory"
3. MIT 6.042J - Mathematics for Computer Science
