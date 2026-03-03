# Sets and Logic

[← Number Systems](../01-Number-Systems/notes.md) | [Next: Functions and Mappings →](../03-Functions-and-Mappings/notes.md)

## Introduction

Set theory and mathematical logic form the foundational language of mathematics and computer science. Understanding sets is essential for probability theory, while logical reasoning underpins algorithm design and formal proofs in ML theory.

In AI/ML practice, these ideas show up constantly:

- **Dataset slices and splits**: train/val/test as subsets; filtering examples is set intersection/difference.
- **Feature engineering**: selected features are a subset of all features; dropped features are a set difference.
- **Model evaluation**: confusion-matrix terms (TP/FP/FN/TN) are intersections of predicted/actual index sets.
- **Neural network masking**: attention/padding masks are boolean logic (AND/OR/NOT) over index sets.
- **Information retrieval & recommenders**: similarity and overlap (e.g., Jaccard) are set operations.

## Prerequisites

- Basic algebra
- Familiarity with mathematical notation

## Learning Objectives

1. Master set notation and operations
2. Understand logical connectives and quantifiers
3. Apply set theory to probability and data structures
4. Use logical reasoning in proofs and algorithms

---

## Table of Contents

1. [Set Basics](#1-set-basics)
2. [Set Relationships](#2-set-relationships)
3. [Set Operations](#3-set-operations)
4. [Propositional Logic](#4-propositional-logic)
5. [Logical Equivalences](#5-logical-equivalences)
6. [Quantifiers](#6-quantifiers)
7. [Applications in ML/AI](#7-applications-in-mlai)
8. [Common Pitfalls](#8-common-pitfalls)
9. [Interview Questions](#9-interview-questions)
10. [Summary](#10-summary)
11. [Further Reading](#11-further-reading)

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

### Multisets (Bags)

A **multiset** (or bag) is like a set, but elements **can repeat**. This is fundamental to NLP and counting-based ML.

$$\text{Set: } \{a, b, b, c\} = \{a, b, c\} \quad \text{(duplicates ignored)}$$
$$\text{Multiset: } \{\!\{a, b, b, c\}\!\} \quad \text{(b appears twice)}$$

```
MULTISETS IN ML
═══════════════════════════════════════════════════════════════════════

SET vs MULTISET:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Set:      {"the", "cat", "sat"} — just unique words               │
│  Multiset: {"the":2, "cat":1, "sat":1, "on":1, "the":→already 2}  │
│                                                                     │
│  "The cat sat on the mat"                                           │
│  As set:      {the, cat, sat, on, mat} — loses count info          │
│  As multiset: {the:2, cat:1, sat:1, on:1, mat:1} — preserves counts│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

ML APPLICATIONS:
  • Bag-of-Words (BoW):  document = multiset of words
  • TF-IDF:  term frequency = count in multiset
  • Histograms:  pixel/feature intensity multisets
  • Pooling:      mean/sum pooling = operations on multisets
```

#### Code Example

```python
from collections import Counter

# Bag-of-Words = multiset representation
doc = "the cat sat on the mat"
bow = Counter(doc.split())  # Counter({'the': 2, 'cat': 1, 'sat': 1, ...})

# Multiset operations
doc2 = "the dog sat on the log"
bow2 = Counter(doc2.split())

# Intersection (minimum counts) — common words with min frequency
common = bow & bow2  # Counter({'the': 2, 'sat': 1, 'on': 1})

# Union (maximum counts)
combined = bow | bow2  # Counter({'the': 2, 'cat': 1, 'dog': 1, ...})

# Sum (add counts) — concatenating documents
total = bow + bow2  # Counter({'the': 4, 'sat': 2, 'on': 2, ...})
```

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

### Indicator Functions (𝟙_A)

The **indicator function** (or characteristic function) of a set A maps every element to 0 or 1:

$$\mathbb{1}_A(x) = \begin{cases} 1 & \text{if } x \in A \\ 0 & \text{if } x \notin A \end{cases}$$

This simple concept appears **everywhere** in ML:

```
INDICATOR FUNCTIONS IN ML
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────┬──────────────────────────────────────────┐
│ ML Concept               │ Indicator Function Form                  │
├──────────────────────────┼──────────────────────────────────────────┤
│ One-hot encoding         │ 𝟙_{class=k}(x) for each class k         │
│ Binary cross-entropy     │ -[y·log(p) + (1-y)·log(1-p)]            │
│                          │  where y = 𝟙_{positive}(x)              │
│ Attention mask           │ 𝟙_{valid_position}(i)                   │
│ Dropout                  │ 𝟙_{keep}(neuron) with P(keep) = p       │
│ ReLU activation          │ f(x) = x · 𝟙_{x>0}(x)                  │
│ Loss masking             │ loss_i · 𝟙_{not_padding}(i)             │
│ Precision/Recall         │ TP = Σ 𝟙_{predicted=1}(i)·𝟙_{actual=1}(i)│
└──────────────────────────┴──────────────────────────────────────────┘
```

#### Code Example

```python
import numpy as np

# One-hot encoding IS an indicator function
def one_hot(label, num_classes):
    """𝟙_{class=k}(label) for each class k."""
    return np.array([1 if k == label else 0 for k in range(num_classes)])

print(one_hot(2, 5))  # [0, 0, 1, 0, 0]

# ReLU IS x · 𝟙_{x>0}(x)
x = np.array([-2, -1, 0, 1, 2])
relu = x * (x > 0)  # x * 𝟙_{x>0}
print(f"ReLU({x}) = {relu}")  # [0, 0, 0, 1, 2]

# Cross-entropy uses indicator function
y_true = np.array([1, 0, 1, 0])  # 𝟙_{positive}
y_pred = np.array([0.9, 0.1, 0.8, 0.3])
loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
print(f"Binary CE loss: {loss.mean():.4f}")
```

### Indexed Families and Partitions

In ML we often work with **families of sets** indexed by labels, layers, or batches.

$$\bigcup_{i=1}^{n} A_i = A_1 \cup A_2 \cup \cdots \cup A_n$$
$$\bigcap_{i=1}^{n} A_i = A_1 \cap A_2 \cap \cdots \cap A_n$$

**Partition**: A collection of non-empty, disjoint sets whose union is the whole set.

$$\{A_1, A_2, ..., A_k\} \text{ is a partition of } S \iff \begin{cases} A_i \neq \emptyset & \text{for all } i \\ A_i \cap A_j = \emptyset & \text{for } i \neq j \\ \bigcup_{i=1}^{k} A_i = S \end{cases}$$

```
PARTITIONS IN ML
═══════════════════════════════════════════════════════════════════════

Classification IS partitioning:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Dataset D with 3 classes:                                          │
│                                                                     │
│  C₀ = {samples with label 0}  ─┐                                  │
│  C₁ = {samples with label 1}  ─┼─ C₀ ∪ C₁ ∪ C₂ = D (covers all)  │
│  C₂ = {samples with label 2}  ─┘  Cᵢ ∩ Cⱼ = ∅ (no overlap)       │
│                                                                     │
│  Train/Val/Test split IS a partition of the dataset!                │
│  K-fold cross-validation = K partitions rotated                     │
│                                                                     │
│  Decision tree leaves partition the feature space                   │
│  K-means clusters partition the data points                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Code Example

```python
import numpy as np

# Verify a partition (useful for data pipeline validation)
def is_valid_partition(subsets, full_set):
    """Check that subsets form a valid partition."""
    # 1. Non-empty
    if any(len(s) == 0 for s in subsets):
        return False, "Contains empty subset"
    # 2. Pairwise disjoint
    for i in range(len(subsets)):
        for j in range(i + 1, len(subsets)):
            if subsets[i] & subsets[j]:
                return False, f"Subsets {i} and {j} overlap"
    # 3. Union = full set
    union = set().union(*subsets)
    if union != full_set:
        return False, f"Union missing {full_set - union}"
    return True, "Valid partition"

# Train/val/test split
all_indices = set(range(100))
train = set(range(0, 70))
val = set(range(70, 85))
test = set(range(85, 100))

valid, msg = is_valid_partition([train, val, test], all_indices)
print(f"Valid split: {valid} — {msg}")  # Valid partition
```

### Relations and Equivalence Classes

A **relation** R on set A is a subset of A × A. Relations formalize "connectedness" which maps directly to ML concepts.

```
RELATIONS IN ML
═══════════════════════════════════════════════════════════════════════

TYPES OF RELATIONS:
┌──────────────────┬─────────────────────┬───────────────────────────┐
│ Property         │ Definition          │ ML Example                │
├──────────────────┼─────────────────────┼───────────────────────────┤
│ Reflexive        │ ∀x: xRx             │ Every point is similar    │
│                  │                     │ to itself                 │
│ Symmetric        │ xRy → yRx           │ Similarity metrics        │
│                  │                     │ (cosine, Jaccard)         │
│ Transitive       │ xRy ∧ yRz → xRz    │ If A~B and B~C then A~C   │
│ Antisymmetric    │ xRy ∧ yRx → x=y    │ Topological sort          │
└──────────────────┴─────────────────────┴───────────────────────────┘

EQUIVALENCE RELATION (reflexive + symmetric + transitive):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  An equivalence relation PARTITIONS a set into equivalence classes  │
│                                                                     │
│  CLUSTERING IS AN EQUIVALENCE RELATION:                             │
│  • Reflexive: every point is in its own cluster ✓                   │
│  • Symmetric: if x is in same cluster as y, then y with x ✓        │
│  • Transitive: if x~y and y~z, all in same cluster ✓               │
│                                                                     │
│  Each cluster = one equivalence class                               │
│  All clusters together = partition of the dataset                   │
│                                                                     │
│  PARTIAL ORDER (reflexive + antisymmetric + transitive):             │
│  • Computation graphs (DAGs) — topological ordering                 │
│  • Layer dependencies in neural networks                            │
│  • Feature importance ranking                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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

## 7. Sigma Algebras (σ-algebra) — Foundation for Probability

A σ-algebra makes probability **mathematically rigorous**. Without it, you can't properly define P(A) for continuous distributions.

### Intuition

When we toss a coin, the sample space Ω = {H, T} is finite, and we can assign probabilities to every subset. But for continuous distributions (like Gaussian), we can't assign a probability to **every** subset of ℝ — some subsets are too "weird" to measure.

A σ-algebra F tells us **which subsets are measurable** (which events we can ask about).

### Definition

A **σ-algebra** F on a set Ω is a collection of subsets of Ω such that:

$$1. \quad \Omega \in \mathcal{F} \quad \text{(the whole space is measurable)}$$
$$2. \quad A \in \mathcal{F} \Rightarrow A^c \in \mathcal{F} \quad \text{(closed under complement)}$$
$$3. \quad A_1, A_2, ... \in \mathcal{F} \Rightarrow \bigcup_{i=1}^{\infty} A_i \in \mathcal{F} \quad \text{(closed under countable union)}$$

The triple (Ω, F, P) is called a **probability space**.

```
σ-ALGEBRA — WHY ML ENGINEERS SHOULD KNOW THIS
═══════════════════════════════════════════════════════════════════════

SIMPLE CASE (finite — you already use this):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Coin flip: Ω = {H, T}                                             │
│  σ-algebra: F = {∅, {H}, {T}, {H,T}} = 𝒫(Ω)                       │
│  All subsets are measurable — no problems here.                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

CONTINUOUS CASE (where it matters):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Gaussian distribution: Ω = ℝ (all real numbers)                    │
│  σ-algebra: Borel sets B(ℝ) — generated by all open intervals      │
│                                                                     │
│  Can ask: P(X ∈ [0, 1]) = ?  ✓ (interval is Borel)                 │
│  Can ask: P(X > 3)      = ?  ✓ (half-line is Borel)                │
│  Can ask: P(X ∈ weird non-measurable set) = ?  ✗ (undefined!)       │
│                                                                     │
│  WHY THIS MATTERS FOR ML:                                           │
│  • PDF f(x) is defined via P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx              │
│  • This integral only makes sense for measurable sets [a,b]         │
│  • Random variables are measurable functions: X: Ω → ℝ              │
│  • KL divergence, cross-entropy rely on proper measure theory       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

WHEN YOU ENCOUNTER σ-ALGEBRAS:
  • Reading ML theory papers (PAC learning, convergence proofs)
  • Conditional expectations: E[X | F] conditions on a σ-algebra
  • Martingales in online learning / bandit algorithms
  • Measure-theoretic probability in Bayesian inference
```

> **Practical takeaway**: You rarely construct σ-algebras in code, but understanding them lets you read ML theory papers and know why `P(X ∈ A)` isn't defined for arbitrary sets A.

---

## 8. Applications in ML/AI

These are common places where set operations and logic directly translate into ML code.

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

### 3. Boolean Indexing (Pandas/NumPy)

```python
# NumPy/Pandas use logical operations
import numpy as np
import pandas as pd

# NumPy boolean indexing
arr = np.array([1, 2, 3, 4, 5])

# Logical AND
mask = (arr > 2) & (arr < 5)  # [False, False, True, True, False]

# Logical OR
mask = (arr < 2) | (arr > 4)  # [True, False, False, False, True]

# Logical NOT
mask = ~(arr == 3)  # [True, True, False, True, True]

# De Morgan's Law in Pandas
df = pd.DataFrame({'age': [25, 17, 35, 15], 'income': [50000, 0, 75000, 0]})

# Select adults OR high earners
mask1 = (df['age'] >= 18) | (df['income'] > 60000)

# Equivalent using De Morgan: NOT (NOT adult AND NOT high earner)
mask2 = ~((df['age'] < 18) & (df['income'] <= 60000))

# mask1 and mask2 are identical! (De Morgan's Law)
assert (mask1 == mask2).all()
```

### 4. Feature Selection (sklearn)

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Feature sets
all_features = set(feature_names)           # Universe U
selected = SelectKBest(f_classif, k=2).fit(X, y)
selected_mask = selected.get_support()
selected_features = set(np.array(feature_names)[selected_mask])  # Subset S

dropped_features = all_features - selected_features  # Set difference U - S

print(f"All features:      {all_features}")
print(f"Selected features: {selected_features}")  # {'petal_length', 'petal_width'}
print(f"Dropped features:  {dropped_features}")   # {'sepal_length', 'sepal_width'}

# Feature unions and intersections across datasets
features_dataset1 = {'age', 'income', 'location', 'education'}
features_dataset2 = {'age', 'income', 'credit_score', 'employment'}

common_features = features_dataset1 & features_dataset2  # {'age', 'income'}
all_features = features_dataset1 | features_dataset2     # Combined for join
```

### 5. Transformer Attention Masking (PyTorch)

Logic operations are essential in transformer attention:

```python
import torch
import torch.nn.functional as F

# Causal (autoregressive) attention mask
# Lower triangular = can only attend to previous positions
seq_len = 4
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

# Padding mask (ignore PAD tokens)
# Sequence: ["Hello", "World", PAD, PAD]
padding_mask = torch.tensor([True, True, False, False])  # True = valid

# Combined mask using logical AND
# Can attend if: (not future) AND (not padding)
combined_mask = causal_mask.bool() & padding_mask.unsqueeze(0)

# In attention: masked positions get -inf before softmax
attention_scores = torch.randn(seq_len, seq_len)
attention_scores = attention_scores.masked_fill(~combined_mask, float('-inf'))
attention_weights = F.softmax(attention_scores, dim=-1)

# De Morgan in masking:
# ~(future | padding) = ~future & ~padding
```

### 6. Classification Metrics

```python
import numpy as np
from sklearn.metrics import confusion_matrix

# Predictions and ground truth as SETS of indices
y_true = np.array([1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

# Convert to sets of positive indices
actual_positive = set(np.where(y_true == 1)[0])    # {0, 1, 3, 6}
predicted_positive = set(np.where(y_pred == 1)[0]) # {0, 3, 5, 6}
actual_negative = set(np.where(y_true == 0)[0])    # {2, 4, 5, 7}
predicted_negative = set(np.where(y_pred == 0)[0]) # {1, 2, 4, 7}

# Confusion matrix as SET OPERATIONS
TP = predicted_positive & actual_positive   # {0, 3, 6}
FP = predicted_positive & actual_negative   # {5}
FN = predicted_negative & actual_positive   # {1}
TN = predicted_negative & actual_negative   # {2, 4, 7}

# Metrics
precision = len(TP) / len(predicted_positive)  # 3/4 = 0.75
recall = len(TP) / len(actual_positive)        # 3/4 = 0.75
f1 = 2 * precision * recall / (precision + recall)

print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
```

### 7. Logical Rules in Expert Systems

```
Rule-based systems use logical implications:
IF fever AND cough THEN possible_flu
p ∧ q → r

Chaining rules:
(p → q) ∧ (q → r) ⊢ (p → r)
```

### 8. NLP and Text Processing

Sets are fundamental in Natural Language Processing:

```python
# Bag-of-Words treats documents as sets of words
from sklearn.feature_extraction.text import CountVectorizer

doc1 = "machine learning is amazing"
doc2 = "deep learning is powerful"

vectorizer = CountVectorizer()
X = vectorizer.fit_transform([doc1, doc2])

# Vocabulary is a SET of unique words
vocab = set(vectorizer.get_feature_names_out())
# {'amazing', 'deep', 'is', 'learning', 'machine', 'powerful'}

# Common words (intersection) - useful for similarity
words1 = set(doc1.split())
words2 = set(doc2.split())
common = words1 & words2  # {'learning', 'is'}

# Unique words (symmetric difference) - useful for diversity
unique = words1 ^ words2  # {'machine', 'amazing', 'deep', 'powerful'}
```

**ML Applications:**

- **TF-IDF**: Computes over vocabulary sets
- **Stop word removal**: Set difference (words - stopwords)
- **Tokenization**: Creates word sets from text

### 9. Recommender Systems

Set operations power collaborative filtering:

```python
import numpy as np

# Users as sets of items they liked
user_A = {'movie1', 'movie2', 'movie3', 'movie4'}
user_B = {'movie2', 'movie3', 'movie5', 'movie6'}

# Jaccard Similarity using set operations
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)  # A ∩ B
    union = len(set1 | set2)          # A ∪ B
    return intersection / union

similarity = jaccard_similarity(user_A, user_B)
# = |{movie2, movie3}| / |{movie1-6}| = 2/6 = 0.33

# Items to recommend to user_A (items B liked but A hasn't seen)
recommendations = user_B - user_A  # {'movie5', 'movie6'}
```

**ML Applications:**

- **Collaborative filtering**: User-item set overlaps
- **Content-based filtering**: Feature set similarity
- **Association rules**: Itemset mining (Apriori algorithm)

### 10. Neural Networks and Logic Gates

Neural networks evolved from logical foundations:

```
PERCEPTRON AS LOGIC GATE
═══════════════════════════════════════════════════════════════════════

AND Gate (w1=1, w2=1, threshold=1.5):
┌─────┬─────┬───────────────┬────────┐
│ x1  │ x2  │ x1+x2 > 1.5?  │ Output │
├─────┼─────┼───────────────┼────────┤
│  0  │  0  │   0 > 1.5? N  │   0    │
│  0  │  1  │   1 > 1.5? N  │   0    │
│  1  │  0  │   1 > 1.5? N  │   0    │
│  1  │  1  │   2 > 1.5? Y  │   1    │
└─────┴─────┴───────────────┴────────┘

OR Gate (w1=1, w2=1, threshold=0.5):
┌─────┬─────┬───────────────┬────────┐
│ x1  │ x2  │ x1+x2 > 0.5?  │ Output │
├─────┼─────┼───────────────┼────────┤
│  0  │  0  │   0 > 0.5? N  │   0    │
│  0  │  1  │   1 > 0.5? Y  │   1    │
│  1  │  0  │   1 > 0.5? Y  │   1    │
│  1  │  1  │   2 > 0.5? Y  │   1    │
└─────┴─────┴───────────────┴────────┘

XOR Problem (NOT linearly separable - led to multi-layer networks!):
┌─────┬─────┬────────┐
│ x1  │ x2  │ XOR    │   The XOR problem motivated
├─────┼─────┼────────┤   hidden layers and modern
│  0  │  0  │   0    │   deep learning!
│  0  │  1  │   1    │
│  1  │  0  │   1    │   XOR = (x1 ∧ ¬x2) ∨ (¬x1 ∧ x2)
│  1  │  1  │   0    │
└─────┴─────┴────────┘
```

### 11. Knowledge Graphs & Semantic Web

Predicate logic powers knowledge representation:

```
KNOWLEDGE GRAPHS USE FIRST-ORDER LOGIC
═══════════════════════════════════════════════════════════════════════

RDF Triples as Logical Predicates:
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│  Triple: (Albert_Einstein, born_in, Germany)                       │
│  Logic:  born_in(Albert_Einstein, Germany)                         │
│                                                                    │
│  Triple: (Germany, located_in, Europe)                             │
│  Logic:  located_in(Germany, Europe)                               │
│                                                                    │
│  Inference Rule:                                                   │
│  ∀x,y,z: (born_in(x,y) ∧ located_in(y,z)) → born_in_region(x,z)   │
│                                                                    │
│  Result: born_in_region(Albert_Einstein, Europe) ✓                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**ML Applications:**

- **Graph Neural Networks**: Node set neighborhoods
- **Knowledge Graph Embeddings**: TransE, RotatE
- **Question Answering**: SPARQL queries use set operations
- **Neo4j / GraphQL**: Cypher uses logical predicates

### 12. Fuzzy Sets and Soft Membership

Classical sets have **hard membership**: x ∈ A or x ∉ A. **Fuzzy sets** allow **degrees of membership** μ_A(x) ∈ [0, 1].

This directly maps to how neural networks work — outputs are probabilities, not binary decisions.

```
FUZZY SETS — THE MATH BEHIND SOFT PREDICTIONS
═══════════════════════════════════════════════════════════════════════

CLASSICAL SET vs FUZZY SET:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Classical: Is this image a cat?                                    │
│    μ_{cat}(x) ∈ {0, 1}        → Yes or No                         │
│                                                                     │
│  Fuzzy: How much is this image a cat?                               │
│    μ_{cat}(x) ∈ [0, 1]        → 0.85 (very likely cat)            │
│    μ_{dog}(x) ∈ [0, 1]        → 0.10 (unlikely dog)               │
│    μ_{bird}(x) ∈ [0, 1]       → 0.05 (very unlikely bird)         │
│                                                                     │
│  Softmax output IS a fuzzy membership function!                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

FUZZY SET OPERATIONS:
┌──────────────────┬────────────────────────┬─────────────────────────┐
│ Operation        │ Fuzzy Version          │ ML Connection           │
├──────────────────┼────────────────────────┼─────────────────────────┤
│ Membership       │ μ_A(x) ∈ [0, 1]       │ Sigmoid/softmax output  │
│ Union            │ max(μ_A, μ_B)          │ OR-like combination     │
│ Intersection     │ min(μ_A, μ_B)          │ AND-like combination    │
│ Complement       │ 1 - μ_A               │ 1 - probability         │
│ Crisp (hard)     │ μ_A ∈ {0, 1}          │ argmax (hard decision)  │
└──────────────────┴────────────────────────┴─────────────────────────┘

WHERE FUZZY SETS APPEAR IN ML:
  • Softmax outputs: fuzzy membership across classes
  • Attention weights: fuzzy selection of which tokens to attend to
  • Label smoothing: convert hard labels [0,1] to fuzzy [0.05, 0.95]
  • Sigmoid: fuzzy membership for binary classification
  • Soft masks: differentiable alternatives to hard masking
```

#### Code Example

```python
import numpy as np

# Softmax IS a fuzzy membership function
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

logits = np.array([2.0, 1.0, 0.1])
fuzzy_membership = softmax(logits)
print(f"Fuzzy membership (softmax): {fuzzy_membership}")
# [0.659, 0.242, 0.099] — degrees of membership, sum to 1

# Label smoothing: hard labels → fuzzy labels
def label_smoothing(hard_label, num_classes, epsilon=0.1):
    """Convert hard one-hot to fuzzy distribution."""
    soft = np.full(num_classes, epsilon / num_classes)
    soft[hard_label] = 1.0 - epsilon + epsilon / num_classes
    return soft

hard = np.array([0, 0, 1, 0, 0])  # Crisp set: class 2
soft = label_smoothing(2, 5, epsilon=0.1)  # Fuzzy set
print(f"Hard label: {hard}")
print(f"Soft label: {soft.round(3)}")
# [0.02, 0.02, 0.92, 0.02, 0.02] — fuzzy membership!

# Attention weights are fuzzy set membership over tokens
attention = softmax(np.array([3.0, 0.5, 0.1, 2.0]))
print(f"\nAttention weights: {attention.round(3)}")
print("Token 0 membership: {:.1%} (highly attended)".format(attention[0]))
print("Token 2 membership: {:.1%} (barely attended)".format(attention[2]))
```

### 13. AI/ML Domain Quick Reference

| Set/Logic Concept  | AI/ML Domain        | Specific Application |
| ------------------ | ------------------- | -------------------- |
| Set membership (∈) | NLP                 | Vocabulary lookup    |
| Intersection (∩)   | Recommenders        | Common items/users   |
| Union (∪)          | Feature Engineering | Combine feature sets |
| Difference (−)     | Recommenders        | Items to recommend   |
| Jaccard Similarity | Clustering          | Document similarity  |
| Boolean AND (∧)    | Neural Networks     | Perceptron AND gate  |
| Boolean OR (∨)     | Neural Networks     | Perceptron OR gate   |
| Implication (→)    | Expert Systems      | IF-THEN rules        |
| Predicate Logic    | Knowledge Graphs    | RDF triples, SPARQL  |
| Quantifiers (∀,∃)  | Formal Verification | Neural net proofs    |
| Indicator (𝟙_A)    | Loss Functions      | One-hot, masking     |
| Multiset           | NLP                 | Bag-of-Words, TF-IDF |
| Partition          | Classification      | Class labels, splits |
| Equivalence class  | Clustering          | Cluster membership   |
| σ-algebra          | Probability Theory  | Measurable events    |
| Fuzzy membership   | Neural Networks     | Softmax, attention   |

---

## 9. Common Pitfalls

### Pitfall 1: Confusing ⊂ and ⊆

```
COMMON MISTAKE:
┌─────────────────────────────────────────────────────────────────────┐
│  ⊆ means "is a subset of" (may be equal)                            │
│  ⊂ means "is a proper subset of" (must be strictly smaller)         │
│                                                                     │
│  {1,2} ⊆ {1,2}  ✓ TRUE  (equal sets are subsets)                    │
│  {1,2} ⊂ {1,2}  ✗ FALSE (equal sets are NOT proper subsets)         │
│                                                                     │
│  {1,2} ⊆ {1,2,3}  ✓ TRUE                                            │
│  {1,2} ⊂ {1,2,3}  ✓ TRUE                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Pitfall 2: Implication (→) Truth Table

```
COMMON MISTAKE: Misunderstanding when p → q is TRUE

p → q is FALSE *only* when p is TRUE and q is FALSE!

┌───┬───┬───────┬────────────────────────────────────────────────────┐
│ p │ q │ p → q │ Explanation                                        │
├───┼───┼───────┼────────────────────────────────────────────────────┤
│ T │ T │   T   │ Promise kept                                       │
│ T │ F │   F   │ Promise broken (only false case!)                  │
│ F │ T │   T   │ No promise made, so not broken (vacuously true)    │
│ F │ F │   T   │ No promise made, so not broken (vacuously true)    │
└───┴───┴───────┴────────────────────────────────────────────────────┘

Example: "If it rains, I will bring an umbrella"
- Rains + umbrella = promise kept (T)
- Rains + no umbrella = promise broken (F)
- No rain + umbrella = I chose to, no promise broken (T)
- No rain + no umbrella = no promise made about this case (T)
```

### Pitfall 3: Order of Quantifiers Matters

```
COMMON MISTAKE: Thinking ∀x∃y and ∃y∀x are the same

∀x ∃y (x + y = 0)  ≠  ∃y ∀x (x + y = 0)
    ↓                      ↓
"For each x,            "There is ONE y
 there's a y             that works for
 (depends on x)"         ALL x"

∀x ∃y (x + y = 0): TRUE  (y = -x works for each x)
∃y ∀x (x + y = 0): FALSE (no single y works for all x)
```

### Pitfall 4: Empty Set Subtleties

```
∅ ⊆ A    TRUE (for any set A, including ∅ itself)
∅ ∈ A    Usually FALSE (unless A explicitly contains ∅)

Example:
A = {1, 2, 3}
∅ ⊆ A  ✓ TRUE (empty set is subset of everything)
∅ ∈ A  ✗ FALSE (empty set is not an element of A)

B = {1, 2, ∅}
∅ ⊆ B  ✓ TRUE
∅ ∈ B  ✓ TRUE (B explicitly contains ∅ as an element)
```

### Pitfall 5: Set vs Element

```python
# In Python, be careful with set membership

A = {1, 2, 3}
B = {{1, 2}, 3}  # B contains a SET as an element!

1 in A        # True (1 is element of A)
{1, 2} in A   # False ({1,2} is not an element of A)
{1, 2} in B   # True ({1,2} IS an element of B)

# Common bug in ML:
features = {'age', 'income'}
if 'age' in features:     # Correct
    pass
if {'age'} in features:   # Wrong! {'age'} is a set, not a string
    pass
```

---

## 10. Interview Questions

### Basic Questions

1. **Q: What is the difference between ∪ and ∩?**

   A: Union (∪) includes elements in _either_ set (OR), while intersection (∩) includes only elements in _both_ sets (AND).
   Example: {1,2} ∪ {2,3} = {1,2,3}, but {1,2} ∩ {2,3} = {2}.

2. **Q: Explain De Morgan's Laws for sets.**

   A: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ and (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ. The complement of a union is the intersection of complements, and vice versa. This is used in query optimization and boolean algebra.

3. **Q: What is the power set? What is its cardinality?**

   A: The power set 𝒫(A) is the set of all subsets of A, including ∅ and A itself. If |A| = n, then |𝒫(A)| = 2ⁿ. For A = {1,2}, 𝒫(A) = {∅, {1}, {2}, {1,2}}.

4. **Q: When is p → q false?**

   A: Only when p is TRUE and q is FALSE. This is the only case where a promise/implication is broken. If p is false, the implication is vacuously true.

### Advanced Questions

5. **Q: How are sets used in database operations?**

   A: SQL operations correspond to set operations: UNION (∪), INTERSECT (∩), EXCEPT (set difference). WHERE clauses use logical operations. Understanding set theory helps optimize queries.

6. **Q: Explain the connection between sets and probability.**

   A: In probability, the sample space Ω is a set of all outcomes. Events are subsets of Ω. P(A ∪ B) = P(A) + P(B) - P(A ∩ B) comes directly from the inclusion-exclusion principle for set cardinality.

7. **Q: What are confusion matrix metrics in set notation?**

   A:
   - TP = Predicted_Positive ∩ Actual_Positive
   - FP = Predicted_Positive ∩ Actual_Negative
   - Precision = |TP| / |Predicted_Positive|
   - Recall = |TP| / |Actual_Positive|

8. **Q: How would you negate "All ML models overfit"?**

   A: Original: ∀x (Model(x) → Overfits(x))
   Negation: ∃x (Model(x) ∧ ¬Overfits(x))
   In words: "There exists an ML model that does not overfit."

---

## 11. Summary

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

### Set Theory to Logic Mapping

| Set Operation | Logical Operation |
| ------------- | ----------------- |
| A ∪ B         | p ∨ q             |
| A ∩ B         | p ∧ q             |
| Aᶜ            | ¬p                |
| A ⊆ B         | p → q             |
| A = B         | p ↔ q             |

---

## Exercises

1. Given A = {1,2,3,4} and B = {3,4,5,6}, find A∪B, A∩B, A-B, A△B
2. Prove De Morgan's Law: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
3. Construct truth table for (p → q) ∧ (q → r) → (p → r)
4. Negate: "For all ε > 0, there exists δ > 0 such that |f(x) - L| < ε"
5. Express classification metrics using set notation
6. **NEW**: Prove that ∅ ⊆ A for any set A
7. **NEW**: Show that p → q ≡ ¬p ∨ q using a truth table

---

## 12. Further Reading

### Courses

- **Stanford CS103** (sets, logic, proofs): https://web.stanford.edu/class/cs103/
- **MIT 6.042J** (math for CS): https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/
- **Math for Machine Learning (book + course site)**: https://mml-book.github.io/

### Books

- **How to Prove It** (Velleman)
- **Mathematics for Machine Learning** (Deisenroth et al., free): https://mml-book.github.io/
- **Deep Learning** (Goodfellow et al., free): https://www.deeplearningbook.org/
- **Probability and Measure** (Billingsley) — for σ-algebras and measure theory

### Papers

- 📄 [Fuzzy Sets (Zadeh, 1965)](<https://doi.org/10.1016/S0019-9958(65)90241-X>) — the original paper
- 📄 [Rethinking Softmax: Label Smoothing](https://arxiv.org/abs/1906.02629) — fuzzy labels in practice

### Tools

- Truth tables: https://web.stanford.edu/class/cs103/tools/truth-table-tool/
- Venn diagrams: https://www.geogebra.org/m/vQ4cWXE6
- Proof checker: https://proofs.openlogicproject.org/

---

## Companion Notebooks

| Notebook                           | Description                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive examples: set operations, De Morgan's laws, confusion matrices, Jaccard similarity |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions                                                               |

---

## What's Next?

After mastering sets and logic, proceed to:
→ [Functions and Mappings](../03-Functions-and-Mappings/notes.md) — Mathematical functions essential for ML

---

_Last updated: February 2026_
