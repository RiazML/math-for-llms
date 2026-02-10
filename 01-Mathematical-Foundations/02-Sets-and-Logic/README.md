# Sets and Logic

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

### 12. AI/ML Domain Quick Reference

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

---

## 8. Common Pitfalls

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

## 9. Interview Questions

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

## 10. Summary

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

## 11. Further Reading

### Courses

- **Stanford CS103** (sets, logic, proofs): https://web.stanford.edu/class/cs103/
- **MIT 6.042J** (math for CS): https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/
- **Math for Machine Learning (book + course site)**: https://mml-book.github.io/

### Books

- **How to Prove It** (Velleman)
- **Mathematics for Machine Learning** (Deisenroth et al., free): https://mml-book.github.io/
- **Deep Learning** (Goodfellow et al., free): https://www.deeplearningbook.org/

### Tools

- Truth tables: https://web.stanford.edu/class/cs103/tools/truth-table-tool/
- Venn diagrams: https://www.geogebra.org/m/vQ4cWXE6
- Proof checker: https://proofs.openlogicproject.org/

---

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [examples.ipynb](examples.ipynb) | Interactive examples: set operations, De Morgan's laws, confusion matrices, Jaccard similarity |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

---

## What's Next?

After mastering sets and logic, proceed to:
→ [Functions and Mappings](../03-Functions-and-Mappings/README.md) - Mathematical functions essential for ML

---

_Last updated: February 2026_
