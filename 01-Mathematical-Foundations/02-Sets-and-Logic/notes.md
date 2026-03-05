# Sets and Logic

[← Number Systems](../01-Number-Systems/notes.md) | [Next: Functions and Mappings →](../03-Functions-and-Mappings/notes.md)

## Table of Contents

1. [Intuition](#1-intuition)
2. [Naive Set Theory](#2-naive-set-theory)
3. [Set Operations](#3-set-operations)
4. [Relations](#4-relations)
5. [Propositional Logic](#5-propositional-logic)
6. [Predicate Logic (First-Order Logic)](#6-predicate-logic-first-order-logic)
7. [Proof Techniques](#7-proof-techniques)
8. [Axiomatic Set Theory (ZFC)](#8-axiomatic-set-theory-zfc)
9. [Logic in Computation and AI](#9-logic-in-computation-and-ai)
10. [Cardinality and Infinite Sets](#10-cardinality-and-infinite-sets)
11. [Logic and Set Theory in Machine Learning](#11-logic-and-set-theory-in-machine-learning)
12. [Advanced Topics](#12-advanced-topics)
13. [Common Mistakes](#13-common-mistakes)
14. [Exercises](#14-exercises)
15. [Why This Matters for AI (2026 Perspective)](#15-why-this-matters-for-ai-2026-perspective)

---

## 1. Intuition

### 1.1 What Are Sets and Logic?

Set theory is the mathematical language for talking about **collections of objects**; logic is the mathematical language for talking about **truth and valid reasoning**. Together they form the foundation of all mathematics: every other mathematical structure — numbers, functions, probability, linear algebra — is built on sets and governed by logic.

A **set** answers the question: _what objects belong to this category?_

**Logic** answers the question: _given what we know, what can we validly conclude?_

These are not abstract curiosities. When you tokenise a sentence, the vocabulary $V$ is a set. When you compute attention, the mask selects a subset of positions. When you train a model, the loss function maps from a set of parameters to real numbers. When you evaluate whether a model's output is "correct", you're applying logical predicates. Sets and logic are not prerequisites you study once and forget — they are the operational language of every equation in this curriculum.

**For AI and LLMs specifically:**

- Sets formalise vocabularies, token sequences, and model outputs
- Logic underpins reasoning, constraint satisfaction, knowledge representation, and formal verification of AI systems
- Every probability distribution is defined over a set with a logical structure (sigma-algebra)
- Neural network verification expresses safety properties as logical formulas

### 1.2 Why Sets and Logic Matter for AI

| AI Concept                      | Set/Logic Foundation                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Tokenisation**                | Vocabulary $V$ is a set; the set of all token sequences $V^*$ is a formal language                       |
| **Probability**                 | A probability space is built on sets: events are subsets of sample space $\Omega$                        |
| **Neural network architecture** | Each layer is a function between sets (domain $\mathbb{R}^{d_{in}}$ and codomain $\mathbb{R}^{d_{out}}$) |
| **Attention masks**             | A mask selects a subset of positions; causal mask is a specific set relation $\{(i,j) \mid i \geq j\}$   |
| **Structured generation**       | Constrained decoding enforces logical constraints on output tokens at each step                          |
| **Knowledge graphs**            | Entities are sets; relations are functions between sets; reasoning is predicate logic                    |
| **Formal verification**         | Proving properties of AI systems requires predicate logic and SAT/SMT solvers                            |
| **Training data**               | Deduplication is a set operation; a dataset is a multiset of examples                                    |
| **Retrieval (RAG)**             | A query retrieves a subset of a corpus satisfying a relevance predicate                                  |
| **Type checking**               | Tensor shape inference is type theory; preventing dimension mismatches                                   |

### 1.3 The Hierarchy of Mathematical Foundations

Every mathematical structure used in AI rests on a hierarchy, and sets and logic form the very bottom:

```
Layer 7: Machine Learning     (SGD, backprop, transformers)
Layer 6: Probability          (distributions, Bayes, entropy)
Layer 5: Analysis             (continuity, limits, derivatives)
Layer 4: Linear Algebra       (vectors, matrices, eigenvalues)
Layer 3: Algebra              (groups, rings, fields)
Layer 2: Arithmetic           (natural numbers, integers, reals)
Layer 1: Set Theory (ZFC)     (what objects exist? what collections?)
Layer 0: Logic                (what is valid reasoning? what follows from what?)
```

Each layer is defined in terms of the layers below. Natural numbers are constructed from sets (von Neumann ordinals). Real numbers are equivalence classes of Cauchy sequences of rationals. Vector spaces are sets with operations satisfying axioms stated in predicate logic. Probability measures are functions from sets to $[0,1]$. The entire edifice stands on sets and logic.

This is not historical accident — it is the result of 150 years of foundational work from Cantor, Frege, Russell, Hilbert, Gödel, and Turing to make mathematics rigorous and unambiguous.

### 1.4 Formal vs Informal Mathematics

**Informal mathematics** is what most research papers use: proofs in natural language, intuitions supported by diagrams, appeals to "clearly" and "obviously". This is adequate for communication between mathematicians who share common training and conventions.

**Formal mathematics** is where every step is justified by an explicit rule, and proofs are machine-checkable. Nothing is taken for granted:

| Aspect             | Informal                    | Formal                                      |
| ------------------ | --------------------------- | ------------------------------------------- |
| Language           | Natural language + notation | Formal logic (FOL or type theory)           |
| Proof verification | Human reviewer              | Machine (proof assistant)                   |
| Ambiguity          | Tolerated                   | Forbidden                                   |
| Tools              | Paper, LaTeX                | Lean 4, Coq, Isabelle, Agda                 |
| Scale              | Research papers             | Verified libraries (Mathlib ~150K theorems) |

**Proof assistants** encode logic and set theory precisely and verify proofs automatically. They are no longer academic curiosities:

- **Lean 4** (Microsoft Research): Mathlib library; used by AlphaProof (Google DeepMind, 2024) to win IMO gold medals
- **Coq** (INRIA): CompCert verified C compiler; Four Colour Theorem formalised
- **Isabelle** (Cambridge/Munich): seL4 verified operating system kernel
- **AI and formal mathematics**: LLMs increasingly used to assist with formal proofs — Lean Copilot, Hypertree Proof Search, AlphaProof. Understanding sets and logic is prerequisite for understanding what LLMs are attempting when doing mathematical reasoning

### 1.5 Historical Timeline

The development of sets and logic spans 2,400 years. Here are the critical milestones:

| Year     | Person                  | Contribution                                                                     |
| -------- | ----------------------- | -------------------------------------------------------------------------------- |
| ~350 BCE | **Aristotle**           | Syllogistic logic — first formal system of deductive reasoning                   |
| 1679     | **Leibniz**             | Calculus ratiocinator — dream of universal logical calculus                      |
| 1847     | **Boole**               | _The Mathematical Analysis of Logic_ — Boolean algebra; logic as algebra         |
| 1874     | **Cantor**              | Set theory — infinite sets, cardinality, continuum hypothesis                    |
| 1879     | **Frege**               | Begriffsschrift — first complete formal logic system; predicate calculus         |
| 1901     | **Russell**             | Russell's Paradox — naive set theory is inconsistent                             |
| 1908     | **Zermelo**             | Axiomatic set theory — foundation for modern mathematics                         |
| 1910–13  | **Russell & Whitehead** | _Principia Mathematica_ — attempt to reduce all mathematics to logic             |
| 1922–30s | **Fraenkel, Skolem**    | ZFC axioms completed — standard foundation of modern mathematics                 |
| 1929     | **Gödel**               | Completeness theorem — every valid FOL formula has a proof                       |
| 1931     | **Gödel**               | Incompleteness theorems — no consistent system can prove all true arithmetic     |
| 1936     | **Church & Turing**     | Lambda calculus and Turing machines — computability theory                       |
| 1936     | **Tarski**              | Formal semantics — truth defined precisely for formal languages                  |
| 1965     | **Zadeh**               | Fuzzy logic — graded truth values in $[0,1]$                                     |
| 1972     | **Martin-Löf**          | Dependent type theory — constructive foundations connecting logic to programming |
| 2005–10  | —                       | SAT solvers industrialised — automated reasoning becomes practical               |
| 2024     | **Google DeepMind**     | AlphaProof — neural theorem prover achieves IMO gold medal level                 |

### 1.6 The Two Pillars

Mathematics rests on two pillars, and understanding their relationship is essential:

```
                        MATHEMATICS
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   LOGIC                                  SET THEORY
   (What is valid reasoning?)             (What objects exist?)
        │                                       │
   Propositional Logic                    Naive Set Theory
        ↓                                       ↓
   Predicate Logic (FOL)                  Axiomatic (ZFC)
        ↓                                       ↓
   Modal / Temporal Logic                 Type Theory
        │                                       │
        └───────────────────┬───────────────────┘
                            │
              Both together: the language in which
              all of AI's mathematical foundations
                        are written
```

**Logic** provides the rules of valid inference — what follows from what. It tells you that if "all transformers have attention" and "GPT is a transformer", then "GPT has attention" — and it tells you exactly _why_ this inference is valid (universal instantiation plus modus ponens).

**Set theory** provides the objects — the collections, the spaces, the structures. It tells you what $\mathbb{R}^d$ is (a Cartesian product of $d$ copies of $\mathbb{R}$), what a function is (a special kind of relation, which is a subset of a Cartesian product), and what a probability space is (a triple of sets and a measure).

Neither is sufficient alone. Logic without sets has nothing to reason about. Sets without logic have no way to draw conclusions. Together they form the complete foundation.

---

## 2. Naive Set Theory

### 2.1 What Is a Set?

A **set** is an unordered collection of distinct objects called **elements** (or **members**). This is the simplest and most fundamental concept in all of mathematics.

**Membership notation:**

- $a \in A$ means "$a$ is an element of $A$" (read: "$a$ is in $A$")
- $b \notin A$ means "$b$ is not an element of $A$" (read: "$b$ is not in $A$")

**Example:**
$$A = \{1, 2, 3\}$$

Then $1 \in A$, $2 \in A$, $3 \in A$, but $4 \notin A$, $0 \notin A$.

**The fundamental principle — Axiom of Extensionality:** A set is entirely determined by its members. Two sets are equal if and only if they have exactly the same elements:

$$A = B \iff \forall x\,(x \in A \iff x \in B)$$

This means:

- **Order does not matter:** $\{1, 2, 3\} = \{3, 1, 2\} = \{2, 3, 1\}$
- **Repetition does not matter:** $\{1, 1, 2\} = \{1, 2\}$ (sets have distinct elements)

> **Important distinction:** Sets have no notion of order or multiplicity. If you need order, use a sequence (tuple). If you need multiplicity, use a multiset (see §2.6). In ML, training batches are typically ordered sequences, not sets — but the mathematical specification of "training data" is a multiset (same example can appear multiple times).

### 2.2 Describing Sets

There are three standard ways to describe which elements belong to a set:

**Roster notation (listing):** Enumerate elements explicitly inside braces.

| Example                  | Description                                     |
| ------------------------ | ----------------------------------------------- |
| $\{1, 2, 3, 4, 5\}$      | Finite set with 5 elements                      |
| $\{a, b, c\}$            | Set of three letters                            |
| $\{2, 4, 6, 8, \ldots\}$ | Infinite set with clear pattern (use carefully) |

**Set-builder notation (predicate):** Describe the property that members satisfy.

$$A = \{x \mid P(x)\} \quad \text{or} \quad A = \{x : P(x)\}$$

Read: "the set of all $x$ such that $P(x)$ holds."

| Set-builder                                          | Meaning                         |
| ---------------------------------------------------- | ------------------------------- |
| $\{x \in \mathbb{Z} \mid x \bmod 2 = 0\}$            | Even integers                   |
| $\{x \in \mathbb{R} \mid x > 0\}$                    | Positive real numbers           |
| $\{t \in \Sigma^* \mid t \text{ is a valid token}\}$ | Vocabulary as a set             |
| $\{(i,j) \in [n]^2 \mid i \geq j\}$                  | Causal attention mask positions |

**Parametric description:** Define elements by explicit construction.

| Parametric                               | Meaning                          |
| ---------------------------------------- | -------------------------------- |
| $\{2k \mid k \in \mathbb{Z}\}$           | Even integers                    |
| $\{1/n \mid n \in \mathbb{N},\, n > 0\}$ | Reciprocals of positive integers |
| $\{W_q x \mid x \in \mathbb{R}^d\}$      | Range of a linear map $W_q$      |

### 2.3 Special Sets

The following sets are used so frequently that they have special symbols:

| Symbol             | Name            | Elements                                             |
| ------------------ | --------------- | ---------------------------------------------------- |
| $\emptyset = \{\}$ | Empty set       | No elements                                          |
| $\mathbb{N}$       | Natural numbers | $\{0, 1, 2, 3, \ldots\}$ (sometimes starting from 1) |
| $\mathbb{Z}$       | Integers        | $\{\ldots, -2, -1, 0, 1, 2, \ldots\}$                |
| $\mathbb{Q}$       | Rationals       | $\{p/q \mid p, q \in \mathbb{Z},\, q \neq 0\}$       |
| $\mathbb{R}$       | Real numbers    | All points on the number line                        |
| $\mathbb{C}$       | Complex numbers | $\{a + bi \mid a, b \in \mathbb{R},\, i^2 = -1\}$    |
| $U$                | Universal set   | All objects under consideration (context-dependent)  |

**The containment chain:**

$$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$$

Each step adds new elements: $\mathbb{Z}$ adds negatives, $\mathbb{Q}$ adds fractions, $\mathbb{R}$ adds irrationals ($\sqrt{2}$, $\pi$, $e$), $\mathbb{C}$ adds imaginary numbers.

**Critical distinctions with the empty set:**

| Expression                     | Elements                                      | Cardinality |
| ------------------------------ | --------------------------------------------- | ----------- |
| $\emptyset = \{\}$             | None                                          | 0           |
| $\{\emptyset\} = \{\{\}\}$     | One element: $\emptyset$                      | 1           |
| $\{\emptyset, \{\emptyset\}\}$ | Two elements: $\emptyset$ and $\{\emptyset\}$ | 2           |

> **Warning:** $\emptyset \neq \{\emptyset\}$. The empty set has no elements. The set containing the empty set has one element (which happens to be the empty set). This distinction is not pedantic — it is exactly how natural numbers are constructed from sets in ZFC (see §8.3).

### 2.4 Subset and Superset

**Subset:** $A$ is a subset of $B$ (written $A \subseteq B$) if every element of $A$ is also an element of $B$:

$$A \subseteq B \iff \forall x\,(x \in A \implies x \in B)$$

**Proper subset:** $A$ is a proper subset of $B$ (written $A \subsetneq B$ or $A \subset B$) if $A \subseteq B$ and $A \neq B$ — that is, $B$ has at least one element not in $A$.

**Superset:** $A \supseteq B$ means $B \subseteq A$ (read: "$A$ is a superset of $B$").

**Properties of the subset relation:**

| Property     | Statement                                                  | Meaning                                   |
| ------------ | ---------------------------------------------------------- | ----------------------------------------- |
| Reflexivity  | $A \subseteq A$                                            | Every set is a subset of itself           |
| Antisymmetry | $A \subseteq B$ and $B \subseteq A \implies A = B$         | Standard proof technique for set equality |
| Transitivity | $A \subseteq B$ and $B \subseteq C \implies A \subseteq C$ | Subset chains compose                     |
| Empty set    | $\emptyset \subseteq A$ for every set $A$                  | Vacuously true                            |

The antisymmetry property gives us the **standard method for proving two sets are equal**: show $A \subseteq B$ (every element of $A$ is in $B$) and $B \subseteq A$ (every element of $B$ is in $A$). This "double containment" proof is used constantly in mathematics.

> **Vacuous truth explanation:** Why is $\emptyset \subseteq A$ true? The definition says: for every $x$, if $x \in \emptyset$ then $x \in A$. Since nothing is in $\emptyset$, the "if" clause is never satisfied, so the implication is vacuously true. There is no counterexample — we would need an $x$ that is in $\emptyset$ but not in $A$, and no such $x$ exists. See §5.2 for more on vacuous truth and implication.

### 2.5 Russell's Paradox

**Naive comprehension** (the principle that doomed early set theory): For any property $P$, there exists a set $\{x \mid P(x)\}$. This seems natural — "the set of all things with property $P$" — but it leads to disaster.

**Russell's construction (1901):** Define

$$R = \{x \mid x \notin x\}$$

The set of all sets that don't contain themselves. Now ask: is $R \in R$?

**Case 1: Assume $R \in R$.**
Then $R$ satisfies its own membership condition, which says $R \notin R$. Contradiction.

**Case 2: Assume $R \notin R$.**
Then $R$ satisfies the condition $x \notin x$, so by the definition of $R$, we have $R \in R$. Contradiction.

Either way we get a contradiction. **Naive set theory is inconsistent.**

**The resolution:** We cannot form "the set of all sets" or unrestricted collections. The axiom of **Separation** (in ZFC) restricts comprehension: you can only form $\{x \in A \mid P(x)\}$ for an already-existing set $A$. You cannot form a set "from scratch" using just a property — you must separate from an existing set. This blocks Russell's paradox because there is no universal set of all sets from which to separate.

This is not merely a historical curiosity. Russell's paradox is the reason mathematics has axioms for set theory at all, and understanding it is prerequisite for understanding ZFC (§8), type theory (§9.4), and the limits of formal systems (§12.1).

### 2.6 Multisets (Bags)

Standard sets require distinct elements: $\{1, 1, 2\} = \{1, 2\}$. But in practice, we often need to track multiplicity — how many times each element appears.

A **multiset** (or **bag**) generalises sets by assigning each element a **multiplicity** (count). Formally, a multiset over a base set $A$ is a function $m: A \to \mathbb{N}_0$ where $m(a)$ gives the number of times $a$ appears.

**Notation:** $\{1, 1, 2, 3, 3, 3\}$ as a multiset has $m(1) = 2$, $m(2) = 1$, $m(3) = 3$.

**Multiset operations:**

| Operation          | Definition                               | Example                                    |
| ------------------ | ---------------------------------------- | ------------------------------------------ |
| Union (max)        | $(A \uplus B)(x) = \max(m_A(x), m_B(x))$ | $\{1,1,2\} \uplus \{1,2,2\} = \{1,1,2,2\}$ |
| Sum (add)          | $(A + B)(x) = m_A(x) + m_B(x)$           | $\{1,1,2\} + \{1,2,2\} = \{1,1,1,2,2,2\}$  |
| Intersection (min) | $(A \cap B)(x) = \min(m_A(x), m_B(x))$   | $\{1,1,2\} \cap \{1,2,2\} = \{1,2\}$       |

**AI applications of multisets:**

- **Training data:** A dataset is a multiset of examples — the same document may appear multiple times (data repetition, which affects training dynamics)
- **Token counting:** Term frequency in TF-IDF and BM25 is a multiset operation — how many times each word appears matters
- **Gradient accumulation:** Sum of gradient multiset over a batch: each micro-batch contributes a gradient, and we sum (multiset addition)
- **Batch statistics:** BatchNorm computes mean and variance of a multiset of activations per channel

---

## 3. Set Operations

Set operations are the algebraic toolkit for combining, comparing, and transforming sets. Every one of these operations appears directly in modern AI systems: attention masks use complement and intersection, retrieval uses union and difference, and probability theory uses all of them.

### 3.1 Union

$$A \cup B = \{x \mid x \in A \text{ or } x \in B\}$$

The union contains every element that belongs to **at least one** of the two sets. The "or" is inclusive — elements in both sets are included (once, since sets don't repeat).

**Visual:** Think of taking both sets and pouring them together. Any element from either set goes in.

**Properties of union:**

| Property      | Statement                               | Why                                    |
| ------------- | --------------------------------------- | -------------------------------------- |
| Commutativity | $A \cup B = B \cup A$                   | "or" is commutative                    |
| Associativity | $(A \cup B) \cup C = A \cup (B \cup C)$ | Can chain without parentheses          |
| Identity      | $A \cup \emptyset = A$                  | Adding nothing changes nothing         |
| Idempotence   | $A \cup A = A$                          | Duplicates collapse in sets            |
| Domination    | $A \cup U = U$                          | Everything plus anything is everything |

**Generalised union:** For a collection of sets $\{A_i\}_{i \in I}$:

$$\bigcup_{i \in I} A_i = \{x \mid x \in A_i \text{ for some } i \in I\}$$

**AI examples:**

- **Vocabulary merging:** When combining tokenisers from different models, $V_{\text{merged}} = V_1 \cup V_2$
- **Search results:** "Documents matching query A OR query B" = $\text{Results}(A) \cup \text{Results}(B)$
- **Multi-dataset training:** Training set = $D_1 \cup D_2 \cup \ldots \cup D_k$

### 3.2 Intersection

$$A \cap B = \{x \mid x \in A \text{ and } x \in B\}$$

The intersection contains only elements that belong to **both** sets simultaneously.

**Properties of intersection:**

| Property      | Statement                               | Why                                     |
| ------------- | --------------------------------------- | --------------------------------------- |
| Commutativity | $A \cap B = B \cap A$                   | "and" is commutative                    |
| Associativity | $(A \cap B) \cap C = A \cap (B \cap C)$ | Can chain without parentheses           |
| Identity      | $A \cap U = A$                          | Intersecting with everything keeps all  |
| Idempotence   | $A \cap A = A$                          | Set intersected with itself is itself   |
| Annihilation  | $A \cap \emptyset = \emptyset$          | Intersecting with nothing gives nothing |

**Disjoint sets:** Two sets $A$ and $B$ are **disjoint** if $A \cap B = \emptyset$ — they share no elements.

**Generalised intersection:** For a collection $\{A_i\}_{i \in I}$:

$$\bigcap_{i \in I} A_i = \{x \mid x \in A_i \text{ for all } i \in I\}$$

**AI examples:**

- **Common tokens:** Tokens shared by two vocabularies: $V_1 \cap V_2$
- **Data deduplication:** Documents appearing in both train and test: $D_{\text{train}} \cap D_{\text{test}}$ (should be $\emptyset$ for valid evaluation!)
- **Attention intersection:** Tokens attended by both head A and head B

**Distributive laws (union and intersection interact):**

$$A \cap (B \cup C) = (A \cap B) \cup (A \cap C)$$
$$A \cup (B \cap C) = (A \cup B) \cap (A \cup C)$$

These mirror the distributive laws of multiplication over addition ($a \cdot (b + c) = ab + ac$), which is no coincidence — Boolean algebra (§5.7) makes these analogies precise.

### 3.3 Set Difference

$$A \setminus B = A - B = \{x \mid x \in A \text{ and } x \notin B\}$$

Elements of $A$ that are **not** in $B$. Think of "removing" $B$'s elements from $A$.

**Properties of set difference:**

| Property                             | Statement                                         |
| ------------------------------------ | ------------------------------------------------- |
| Not commutative                      | $A \setminus B \neq B \setminus A$ in general     |
| $A \setminus \emptyset = A$          | Removing nothing changes nothing                  |
| $A \setminus A = \emptyset$          | Removing everything leaves nothing                |
| $A \setminus B \subseteq A$          | Can only reduce the original set                  |
| $(A \setminus B) \cap B = \emptyset$ | What remains shares nothing with what was removed |

**Relationship to other operations:**

$$A \setminus B = A \cap \bar{B}$$

Difference is intersection with complement (this is often the most useful way to think about it).

**AI examples:**

- **Vocabulary difference:** Tokens in GPT-4's vocabulary but not LLaMA's: $V_{\text{GPT4}} \setminus V_{\text{LLaMA}}$
- **Data filtering:** Training data after removing test examples: $D_{\text{train}} \setminus D_{\text{test}}$
- **Feature selection:** Selected features = all features minus dropped features

### 3.4 Symmetric Difference

$$A \triangle B = (A \setminus B) \cup (B \setminus A) = (A \cup B) \setminus (A \cap B)$$

Elements in **exactly one** of the two sets — not in both.

**Properties of symmetric difference:**

| Property      | Statement                                                   | Why                         |
| ------------- | ----------------------------------------------------------- | --------------------------- |
| Commutativity | $A \triangle B = B \triangle A$                             | Definition is symmetric     |
| Associativity | $(A \triangle B) \triangle C = A \triangle (B \triangle C)$ | Can chain                   |
| Identity      | $A \triangle \emptyset = A$                                 | XOR with nothing = original |
| Self-inverse  | $A \triangle A = \emptyset$                                 | XOR with self cancels       |
| Criterion     | $A \triangle B = \emptyset \iff A = B$                      | Zero difference means equal |

> **Algebraic structure:** The symmetric difference operation makes the power set $\mathcal{P}(U)$ into an abelian group with $\emptyset$ as identity. Combined with intersection as multiplication, $(\mathcal{P}(U), \triangle, \cap)$ forms a Boolean ring — connecting set theory to abstract algebra.

**AI examples:**

- **Dataset drift detection:** Tokens that changed between vocabulary v1 and v2: $V_1 \triangle V_2$
- **Model comparison:** Features used by model A XOR model B
- **Edit distance:** Symmetric difference of character sets is a crude string similarity measure

### 3.5 Complement

$$\bar{A} = A' = U \setminus A = \{x \in U \mid x \notin A\}$$

Everything in the universal set $U$ that is **not** in $A$. The complement depends on what $U$ is (must specify context).

**Properties of complement:**

| Property                     | Statement                    |
| ---------------------------- | ---------------------------- |
| Double complement            | $\overline{\bar{A}} = A$     |
| Complement of $\emptyset$    | $\bar{\emptyset} = U$        |
| Complement of $U$            | $\bar{U} = \emptyset$        |
| Union with complement        | $A \cup \bar{A} = U$         |
| Intersection with complement | $A \cap \bar{A} = \emptyset$ |

**AI example:** An attention mask $M$ selects positions to attend to. The complement $\bar{M}$ gives the positions that are masked out (set to $-\infty$ before softmax). In causal masking, the complement is the "future" positions that must not be attended to.

### 3.6 De Morgan's Laws for Sets

These are the most important algebraic identities for sets. They connect union and intersection through complementation:

**First law — complement of union = intersection of complements:**

$$\overline{A \cup B} = \bar{A} \cap \bar{B}$$

"Not in A-or-B" is the same as "not in A **and** not in B."

**Second law — complement of intersection = union of complements:**

$$\overline{A \cap B} = \bar{A} \cup \bar{B}$$

"Not in A-and-B" is the same as "not in A **or** not in B."

**Proof of the first law** (by double containment):

**($\subseteq$)**: Let $x \in \overline{A \cup B}$. Then $x \notin A \cup B$, which means $x \notin A$ and $x \notin B$ (if $x$ were in either, it would be in the union). Hence $x \in \bar{A}$ and $x \in \bar{B}$, so $x \in \bar{A} \cap \bar{B}$.

**($\supseteq$)**: Let $x \in \bar{A} \cap \bar{B}$. Then $x \notin A$ and $x \notin B$. Hence $x$ is not in $A$ or $B$, so $x \notin A \cup B$, which means $x \in \overline{A \cup B}$.

Both containments hold, so $\overline{A \cup B} = \bar{A} \cap \bar{B}$. $\square$

**Generalised De Morgan's Laws:**

$$\overline{\bigcup_i A_i} = \bigcap_i \bar{A}_i \qquad \qquad \overline{\bigcap_i A_i} = \bigcup_i \bar{A}_i$$

**AI examples:**

- **Attention mask logic:** "Not attending to position A or position B" = "not attending to A **and** not attending to B"
- **Retrieval negation:** "Documents not matching (query1 OR query2)" = "documents not matching query1 AND not matching query2"
- **Filter composition:** Negating a disjunctive filter becomes conjunctive

### 3.7 Cartesian Product

$$A \times B = \{(a, b) \mid a \in A,\, b \in B\}$$

The set of all **ordered pairs** where the first element comes from $A$ and the second from $B$. Unlike sets, order matters in pairs: $(a, b) \neq (b, a)$ unless $a = b$.

**Cardinality:** $|A \times B| = |A| \times |B|$

**Not commutative:** $A \times B \neq B \times A$ (different ordered pairs, unless $A = B$).

**Higher products:**

- $A \times B \times C = \{(a, b, c) \mid a \in A,\, b \in B,\, c \in C\}$ — ordered triples
- $A^n = \underbrace{A \times A \times \cdots \times A}_{n \text{ times}}$ — $n$-tuples of elements from $A$

**AI examples — Cartesian products are everywhere:**

| Expression                                                                   | Meaning                                      |
| ---------------------------------------------------------------------------- | -------------------------------------------- |
| $\mathbb{R}^d = \underbrace{\mathbb{R} \times \cdots \times \mathbb{R}}_{d}$ | $d$-dimensional embedding space              |
| $V^n$                                                                        | All possible token sequences of length $n$   |
| $\mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_k}$                 | Domain of attention: (queries, keys)         |
| $[n] \times [n]$                                                             | All position pairs for attention matrix      |
| $\Theta = \mathbb{R}^p$                                                      | Parameter space of model with $p$ parameters |

**Formal definition of ordered pair (Kuratowski):** $(a, b) = \{\{a\}, \{a, b\}\}$. This reduces ordered pairs to pure sets — the pair $(a, b)$ is a set that encodes the order by treating $a$ as the "singleton" element.

### 3.8 Power Set

$$\mathcal{P}(A) = \{B \mid B \subseteq A\}$$

The set of **all subsets** of $A$, including $\emptyset$ and $A$ itself.

**Cardinality:** $|\mathcal{P}(A)| = 2^{|A|}$ for finite $A$.

The reason: for each element of $A$, you independently choose "include" or "exclude" — this gives $2$ choices for each of the $|A|$ elements, hence $2^{|A|}$ subsets total.

**Examples:**

| Set $A$       | $\mathcal{P}(A)$                                  | $\|\mathcal{P}(A)\|$ |
| ------------- | ------------------------------------------------- | -------------------- |
| $\emptyset$   | $\{\emptyset\}$                                   | $2^0 = 1$            |
| $\{1\}$       | $\{\emptyset, \{1\}\}$                            | $2^1 = 2$            |
| $\{1, 2\}$    | $\{\emptyset, \{1\}, \{2\}, \{1,2\}\}$            | $2^2 = 4$            |
| $\{1, 2, 3\}$ | 8 subsets (including $\emptyset$ and $\{1,2,3\}$) | $2^3 = 8$            |

**Cantor's theorem (critical):** The power set is always **strictly larger** than the original set, even for infinite sets:

$$|A| < |\mathcal{P}(A)| \quad \text{for all sets } A$$

This means there are always more subsets than elements — a fact that generates the entire hierarchy of infinite cardinalities (see §10.4).

**AI implications:**

- If vocabulary $|V| = 32{,}000$, then $|\mathcal{P}(V)| = 2^{32{,}000}$ — the number of possible token subsets is astronomically larger than the vocabulary itself
- Structured generation constrains output to a subset of $V$ at each step — it selects one element of $\mathcal{P}(V)$
- The exponential blowup of power sets is why exhaustive search over subsets is intractable, driving the need for heuristic and approximate algorithms throughout AI

### 3.9 Operations Summary Table

For quick reference, here is the complete table of set operations with their logical counterparts:

| Set Operation        | Notation         | Logic Equivalent        | Python                                 |
| -------------------- | ---------------- | ----------------------- | -------------------------------------- |
| Union                | $A \cup B$       | $\vee$ (OR)             | `A \| B` or `A.union(B)`               |
| Intersection         | $A \cap B$       | $\wedge$ (AND)          | `A & B` or `A.intersection(B)`         |
| Difference           | $A \setminus B$  | $\wedge \neg$ (AND NOT) | `A - B` or `A.difference(B)`           |
| Symmetric difference | $A \triangle B$  | $\oplus$ (XOR)          | `A ^ B` or `A.symmetric_difference(B)` |
| Complement           | $\bar{A}$        | $\neg$ (NOT)            | `U - A`                                |
| Cartesian product    | $A \times B$     | —                       | `itertools.product(A, B)`              |
| Power set            | $\mathcal{P}(A)$ | —                       | `itertools.combinations` for each size |
| Subset test          | $A \subseteq B$  | $\implies$              | `A <= B` or `A.issubset(B)`            |

---

## 4. Relations

Relations generalise the concept of "connection" between objects. Functions, orderings, equivalences, and even neural network computations are all special kinds of relations. Understanding relations precisely is what separates informal intuition from rigorous mathematics.

### 4.1 Binary Relations

A **binary relation** $R$ from set $A$ to set $B$ is a subset of the Cartesian product:

$$R \subseteq A \times B$$

If $(a, b) \in R$, we write $aRb$ or $R(a, b)$ and read "$a$ is related to $b$ by $R$."

A **relation on $A$** is a relation from $A$ to itself: $R \subseteq A \times A$.

**Examples:**

| Relation               | Sets                           | Definition                                                                        |
| ---------------------- | ------------------------------ | --------------------------------------------------------------------------------- |
| $\leq$ on $\mathbb{R}$ | $\mathbb{R} \times \mathbb{R}$ | $(x, y) \in \leq$ iff $x$ is less than or equal to $y$                            |
| Parent-of              | Person $\times$ Person         | $(p, c)$ iff $p$ is parent of $c$                                                 |
| Token adjacency        | $V \times V$                   | $(t_1, t_2) \in R$ iff $t_1$ can precede $t_2$ in valid text                      |
| Attention mask         | $[n] \times [n]$               | $(i, j) \in \text{mask}$ iff query at position $i$ attends to key at position $j$ |
| Similarity             | $V \times V$                   | $(w_1, w_2) \in R$ iff $\cos(e_{w_1}, e_{w_2}) > \tau$                            |

A relation can be represented as:

- A set of pairs (the mathematical definition)
- A matrix (Boolean matrix $M$ where $M_{ij} = 1$ iff $iRj$; this is the attention mask representation)
- A directed graph (nodes = elements; edge from $a$ to $b$ iff $aRb$)

### 4.2 Properties of Relations

Let $R$ be a relation on set $A$ (i.e., $R \subseteq A \times A$). There are six fundamental properties a relation may or may not have:

**Reflexivity:** Every element is related to itself.

$$\forall a \in A:\, (a, a) \in R$$

- ✓ $\leq$ is reflexive ($x \leq x$ for all $x$)
- ✗ $<$ is not reflexive ($x \not< x$)
- AI: identity attention (token attends to itself) is a reflexive relation

**Irreflexivity:** No element is related to itself.

$$\forall a \in A:\, (a, a) \notin R$$

- ✓ $<$ is irreflexive
- ✓ "parent-of" is irreflexive (no one is their own parent)
- A relation can be neither reflexive nor irreflexive (some self-loops, not all)

**Symmetry:** If $a$ is related to $b$, then $b$ is related to $a$.

$$\forall a, b \in A:\, (a, b) \in R \implies (b, a) \in R$$

- ✓ "is sibling of" is symmetric
- ✓ "has same length as" is symmetric
- ✗ $\leq$ is not symmetric ($3 \leq 5$ but $5 \not\leq 3$)
- AI: cosine similarity is symmetric; attention is NOT symmetric (query→key direction)

**Antisymmetry:** If $a$ is related to $b$ AND $b$ is related to $a$, then $a = b$.

$$\forall a, b \in A:\, (a, b) \in R \wedge (b, a) \in R \implies a = b$$

- ✓ $\leq$ is antisymmetric ($x \leq y$ and $y \leq x$ implies $x = y$)
- ✓ $\subseteq$ on sets is antisymmetric
- Note: antisymmetry ≠ "not symmetric". A relation can be both symmetric and antisymmetric (e.g., equality)

**Asymmetry:** If $a$ is related to $b$, then $b$ is NOT related to $a$.

$$\forall a, b \in A:\, (a, b) \in R \implies (b, a) \notin R$$

- ✓ $<$ is asymmetric
- ✓ "parent-of" is asymmetric
- Asymmetry implies irreflexivity (if $R$ were reflexive, $(a, a) \in R$ would require $(a, a) \notin R$)

**Transitivity:** If $a$ is related to $b$ and $b$ is related to $c$, then $a$ is related to $c$.

$$\forall a, b, c \in A:\, (a, b) \in R \wedge (b, c) \in R \implies (a, c) \in R$$

- ✓ $\leq$ is transitive
- ✓ "ancestor-of" is transitive
- ✗ "friend-of" is not transitive (my friend's friend need not be my friend)
- AI: causal mask transitivity — if position $i$ attends to $j$ and $j$ attends to $k$, should $i$ attend to $k$? In a causal mask, yes (by transitivity of $\geq$)

### 4.3 Equivalence Relations

A relation $R$ on $A$ is an **equivalence relation** if it is:

1. **Reflexive:** $aRa$ for all $a$
2. **Symmetric:** $aRb \implies bRa$
3. **Transitive:** $aRb \wedge bRc \implies aRc$

Equivalence relations capture the idea of "sameness" or "interchangeability" — elements related by an equivalence relation are considered equivalent for the purpose at hand.

**Examples of equivalence relations:**

| Relation               | Domain              | Equivalence classes                      |
| ---------------------- | ------------------- | ---------------------------------------- |
| Equality $=$           | Any set             | Each class is a singleton $\{a\}$        |
| Same remainder mod $n$ | $\mathbb{Z}$        | $n$ classes: $[0], [1], \ldots, [n-1]$   |
| Same length            | Strings             | Strings of equal length grouped together |
| Same BPE encoding      | Character sequences | Sequences mapping to same token(s)       |
| Same prediction        | Inputs              | Inputs producing identical model output  |

**Equivalence class** of $a$:

$$[a] = \{b \in A \mid aRb\}$$

All elements equivalent to $a$ form the equivalence class $[a]$.

**Partition theorem:** An equivalence relation on $A$ **partitions** $A$ into disjoint equivalence classes:

- Every element belongs to **exactly one** equivalence class
- The union of all classes equals $A$: $\bigcup_a [a] = A$
- Distinct classes are disjoint: $[a] \neq [b] \implies [a] \cap [b] = \emptyset$

**Quotient set:** $A / R = \{[a] \mid a \in A\}$ — the set of all equivalence classes.

> **AI example — tokenisation as equivalence:** BPE tokenisation defines an equivalence relation on character sequences. Two character sequences are "equivalent" if they merge to the same token. The equivalence classes are precisely the tokens. The vocabulary $V$ is the quotient set $\Sigma^* / \sim_{\text{BPE}}$ (restricted to the merge rules).

### 4.4 Partial Orders

A relation $R$ on $A$ is a **partial order** if it is:

1. **Reflexive:** $aRa$ for all $a$
2. **Antisymmetric:** $aRb \wedge bRa \implies a = b$
3. **Transitive:** $aRb \wedge bRc \implies aRc$

A partial order is written $\leq$ or $\preceq$ conventionally. The pair $(A, \leq)$ is called a **partially ordered set** (poset).

**"Partial" means:** Not all pairs need be comparable. Some elements may be **incomparable** — neither $a \leq b$ nor $b \leq a$ holds.

**Total order (linear order):** A partial order where additionally

$$\forall a, b \in A:\, aRb \text{ or } bRa$$

All pairs are comparable. Total orders arrange elements in a single line.

**Examples:**

| Relation                        | Domain                              | Type          | Incomparable?            |
| ------------------------------- | ----------------------------------- | ------------- | ------------------------ |
| $\leq$ on $\mathbb{R}$          | Real numbers                        | Total order   | No                       |
| Lexicographic on strings        | $\Sigma^*$                          | Total order   | No                       |
| $\subseteq$ on $\mathcal{P}(A)$ | Power set                           | Partial order | Yes: $\{1\}$ and $\{2\}$ |
| Divisibility on $\mathbb{N}$    | $a \mid b$ iff $a$ divides $b$      | Partial order | Yes: $2$ and $3$         |
| Prefix relation                 | $s \leq t$ iff $s$ is prefix of $t$ | Partial order | Yes: "ab" and "ba"       |

**Hasse diagram:** A graphical representation of a finite poset. Draw $a$ above $b$ with an edge if $a > b$ and there is no element between them (no element $c$ with $a > c > b$). This gives a compact picture of the partial order structure.

**AI connection — dependency parsing:** In a syntactic parse tree, token $i$ dominates token $j$ if $j$ is a descendant of $i$. This dominance relation is a partial order on sentence tokens. Similarly, in DAG-structured computation graphs, the execution order is a partial order on operations.

### 4.5 Functions as Relations

A **function** $f: A \to B$ is a special case of a relation — a relation $f \subseteq A \times B$ satisfying the condition that every element of $A$ appears in **exactly one** pair:

$$\forall a \in A:\, \exists!\, b \in B:\, (a, b) \in f$$

We write $f(a) = b$ to mean $(a, b) \in f$.

**Terminology:**

| Term          | Symbol                                  | Meaning                        |
| ------------- | --------------------------------------- | ------------------------------ |
| Domain        | $\text{dom}(f) = A$                     | Set of inputs                  |
| Codomain      | $\text{cod}(f) = B$                     | Set of allowed outputs         |
| Range (image) | $\text{ran}(f) = \{f(a) \mid a \in A\}$ | Actual outputs ($\subseteq B$) |

**Types of functions:**

| Type                       | Definition                           | Meaning                             |
| -------------------------- | ------------------------------------ | ----------------------------------- |
| **Injective** (one-to-one) | $f(a_1) = f(a_2) \implies a_1 = a_2$ | Distinct inputs → distinct outputs  |
| **Surjective** (onto)      | $\text{ran}(f) = B$                  | Every output is hit                 |
| **Bijective**              | Injective and surjective             | Perfect pairing between $A$ and $B$ |

**AI examples — functions are the core abstraction:**

| Function  | Type Signature                                                                                                      | Properties                                     |
| --------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Tokeniser | $\Sigma^* \to V^*$                                                                                                  | Not injective (multiple strings → same tokens) |
| Embedding | $V \to \mathbb{R}^d$                                                                                                | Injective (each token gets unique vector)      |
| LM head   | $\mathbb{R}^d \to \mathbb{R}^{\vert V \vert}$                                                                       | Linear map (matrix multiplication)             |
| Attention | $\mathbb{R}^{n \times d} \times \mathbb{R}^{n \times d} \times \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ | Highly non-linear                              |
| Softmax   | $\mathbb{R}^k \to \Delta^{k-1}$                                                                                     | Surjective onto probability simplex            |
| argmax    | $\mathbb{R}^k \to [k]$                                                                                              | Not injective (ties); not differentiable       |

### 4.6 Composition and Inverse

**Composition:** Given $f: A \to B$ and $g: B \to C$, the **composition** $g \circ f: A \to C$ is defined by:

$$(g \circ f)(a) = g(f(a))$$

Read right to left: "apply $f$ first, then $g$."

**Properties of composition:**

- **Associativity:** $h \circ (g \circ f) = (h \circ g) \circ f$
- **Identity:** $f \circ \text{id}_A = f$ and $\text{id}_B \circ f = f$, where $\text{id}_A(a) = a$
- **Not commutative:** $g \circ f \neq f \circ g$ in general

**Inverse:** If $f: A \to B$ is **bijective**, there exists a unique $f^{-1}: B \to A$ satisfying:

$$f^{-1}(f(a)) = a \quad \text{and} \quad f(f^{-1}(b)) = b$$

Only bijections have inverses. Injective-but-not-surjective functions have **left inverses**; surjective-but-not-injective functions have **right inverses**.

> **AI example — neural networks as function composition:** A deep neural network is fundamentally a composition of functions:
>
> $$\hat{y} = f_L \circ f_{L-1} \circ \cdots \circ f_2 \circ f_1(x)$$
>
> Each $f_i$ is one layer (linear map + activation). Deep learning is deeply compositional — the power comes from composing many simple functions. The chain rule of calculus (backpropagation) exploits this compositional structure: $\frac{\partial}{\partial x}(g \circ f) = \frac{\partial g}{\partial f} \cdot \frac{\partial f}{\partial x}$.

---

## 5. Propositional Logic

Propositional logic is the simplest complete system of logical reasoning. It studies how the truth of compound statements depends on the truth of their components, without looking inside those components. It is the foundation for Boolean circuits, attention mask logic, SAT solvers, and the systematic reasoning that predicate logic (§6) and proof techniques (§7) build upon.

### 5.1 Propositions

A **proposition** is a declarative statement that is either **true** ($T$, 1) or **false** ($F$, 0). Never both, never neither.

**Examples of propositions:**

- "The sky is blue" — true proposition
- "$2 + 2 = 5$" — false proposition
- "LLaMA-3 has 8B parameters" — true proposition
- "Every even integer greater than 2 is the sum of two primes" — proposition (truth value unknown; Goldbach's conjecture)

**Not propositions:**

- "What is your name?" — question, no truth value
- "Close the door" — command, no truth value
- "$x > 3$" — depends on $x$; becomes a proposition only when $x$ is specified (this is a **predicate**, see §6)

**Propositional variables:** Letters $p, q, r, s, \ldots$ stand for unspecified propositions. Each has a **truth value** — either $T$ or $F$.

### 5.2 Logical Connectives

Connectives combine propositions into compound statements. There are six standard connectives:

#### Negation (NOT): $\neg p$

"Not $p$." Flips the truth value.

| $p$ | $\neg p$ |
| --- | -------- |
| $T$ | $F$      |
| $F$ | $T$      |

AI: Bitwise NOT in attention masks; complement of a token set.

#### Conjunction (AND): $p \wedge q$

"$p$ and $q$." True **only when both** are true.

| $p$ | $q$ | $p \wedge q$ |
| --- | --- | ------------ |
| $T$ | $T$ | $T$          |
| $T$ | $F$ | $F$          |
| $F$ | $T$ | $F$          |
| $F$ | $F$ | $F$          |

AI: Combining attention masks (token must be unmasked AND within window); intersecting constraints in structured generation.

#### Disjunction (OR): $p \vee q$

"$p$ or $q$." True when **at least one** is true (inclusive or).

| $p$ | $q$ | $p \vee q$ |
| --- | --- | ---------- |
| $T$ | $T$ | $T$        |
| $T$ | $F$ | $T$        |
| $F$ | $T$ | $T$        |
| $F$ | $F$ | $F$        |

AI: Union of token sets in structured generation; "match either pattern" in retrieval.

#### Exclusive Or (XOR): $p \oplus q$

"$p$ xor $q$." True when **exactly one** is true.

| $p$ | $q$ | $p \oplus q$ |
| --- | --- | ------------ |
| $T$ | $T$ | $F$          |
| $T$ | $F$ | $T$          |
| $F$ | $T$ | $T$          |
| $F$ | $F$ | $F$          |

AI: Symmetric difference of sets; error detection (XOR of prediction and ground truth bits); XOR is the simplest function not linearly separable — a single perceptron cannot learn it (Minsky & Papert 1969).

#### Implication (IF...THEN): $p \to q$

"If $p$ then $q$." Also read "$p$ implies $q$." False **only when $p$ is true and $q$ is false** — a true hypothesis cannot lead to a false conclusion.

| $p$ | $q$ | $p \to q$ |
| --- | --- | --------- |
| $T$ | $T$ | $T$       |
| $T$ | $F$ | $F$       |
| $F$ | $T$ | $T$       |
| $F$ | $F$ | $T$       |

**Vacuous truth:** When $p$ is false, $p \to q$ is **always true** regardless of $q$. This is often counterintuitive but essential:

- "If pigs fly, then the moon is green" is **true** (vacuously)
- The reason: $p \to q$ promises "whenever $p$ holds, $q$ holds." If $p$ never holds, the promise is never violated, so it's true by default.

**Vacuous truth in mathematics and AI:**

- "$\forall x \in \emptyset$, property $P(x)$ holds" — true, because there's nothing to check
- "$\emptyset \subseteq A$" — true, because every element of $\emptyset$ (there are none) is in $A$
- "If the model reaches 100% accuracy, then it has generalised" — vacuously true if the model never reaches 100%

#### Biconditional (IF AND ONLY IF): $p \leftrightarrow q$

"$p$ if and only if $q$" (abbreviated "iff"). True when $p$ and $q$ have the **same truth value**.

| $p$ | $q$ | $p \leftrightarrow q$ |
| --- | --- | --------------------- |
| $T$ | $T$ | $T$                   |
| $T$ | $F$ | $F$                   |
| $F$ | $T$ | $F$                   |
| $F$ | $F$ | $T$                   |

Equivalently: $p \leftrightarrow q \equiv (p \to q) \wedge (q \to p)$.

AI: Set equality ($A = B \iff A \subseteq B \wedge B \subseteq A$); defining exactly when a condition holds.

### 5.3 Operator Precedence

When multiple connectives appear without parentheses, the standard precedence (highest first) is:

$$\neg \quad > \quad \wedge \quad > \quad \vee \quad > \quad \to \quad > \quad \leftrightarrow$$

**Examples:**

- $\neg p \wedge q$ means $(\neg p) \wedge q$, **not** $\neg(p \wedge q)$
- $p \vee q \to r$ means $(p \vee q) \to r$, **not** $p \vee (q \to r)$
- $p \wedge q \vee r$ means $(p \wedge q) \vee r$, **not** $p \wedge (q \vee r)$

> **Best practice:** Always use parentheses when there's any ambiguity. Clarity is more important than brevity.

### 5.4 Tautologies, Contradictions, and Contingencies

**Tautology:** A formula that is true for **all possible** truth value assignments to its variables. Logically necessary — cannot possibly be false.

| Tautology                                   | Name                    |
| ------------------------------------------- | ----------------------- |
| $p \vee \neg p$                             | Law of excluded middle  |
| $p \to p$                                   | Self-implication        |
| $(p \wedge q) \to p$                        | Simplification          |
| $(p \to q) \leftrightarrow (\neg p \vee q)$ | Implication equivalence |

**Contradiction (Unsatisfiable):** A formula that is false for **all** truth value assignments.

| Contradiction                      | Why                           |
| ---------------------------------- | ----------------------------- |
| $p \wedge \neg p$                  | Cannot be both true and false |
| $(p \to q) \wedge p \wedge \neg q$ | Modus ponens violated         |

**Contingency:** A formula that is true for some assignments and false for others.

- $p \wedge q$ — true when both true, false otherwise
- $p \to q$ — false only when $T \to F$

**Satisfiability:** A formula is **satisfiable** if at least one truth assignment makes it true. Tautologies and contingencies are satisfiable; contradictions are not.

**The SAT problem:** Given a propositional formula $\phi$, is $\phi$ satisfiable? This is the canonical **NP-complete** problem (Cook–Levin theorem, 1971). Despite worst-case intractability, modern SAT solvers handle millions of variables — they are critical for AI constraint solving, formal verification, and automated planning.

### 5.5 Logical Equivalence

Two formulas $\phi$ and $\psi$ are **logically equivalent** (written $\phi \equiv \psi$) if they have the same truth value for every possible truth assignment. Equivalently: $\phi \leftrightarrow \psi$ is a tautology.

Here are the fundamental equivalences — the "algebra rules" of logic:

**Double Negation:**
$$\neg\neg p \equiv p$$

**De Morgan's Laws:**
$$\neg(p \wedge q) \equiv \neg p \vee \neg q$$
$$\neg(p \vee q) \equiv \neg p \wedge \neg q$$

These are the most used laws in practice. Notice they mirror the set-theoretic De Morgan's Laws (§3.6) exactly — this is the Boolean algebra correspondence (§5.7).

**Implication Equivalences:**
$$p \to q \equiv \neg p \vee q$$
$$p \to q \equiv \neg q \to \neg p \quad \text{(contrapositive)}$$
$$\neg(p \to q) \equiv p \wedge \neg q$$

The contrapositive equivalence is used constantly in proofs (see §7.3). It says "$p$ implies $q$" is the same as "not $q$ implies not $p$." Distinguish from the **converse** ($q \to p$), which is NOT equivalent to $p \to q$.

**Distributive Laws:**
$$p \wedge (q \vee r) \equiv (p \wedge q) \vee (p \wedge r)$$
$$p \vee (q \wedge r) \equiv (p \vee q) \wedge (p \vee r)$$

**Absorption Laws:**
$$p \wedge (p \vee q) \equiv p$$
$$p \vee (p \wedge q) \equiv p$$

**Idempotent Laws:**
$$p \wedge p \equiv p$$
$$p \vee p \equiv p$$

**Identity Laws:**
$$p \wedge T \equiv p$$
$$p \vee F \equiv p$$

**Domination Laws:**
$$p \wedge F \equiv F$$
$$p \vee T \equiv T$$

**Complement Laws:**
$$p \wedge \neg p \equiv F$$
$$p \vee \neg p \equiv T$$

### 5.6 Normal Forms

**Literal:** A variable or its negation ($p$ or $\neg p$).

**Clause:** A disjunction of literals ($p \vee \neg q \vee r$).

**Term:** A conjunction of literals ($p \wedge \neg q \wedge r$).

#### Conjunctive Normal Form (CNF)

A formula in CNF is a **conjunction of clauses** — an AND of ORs:

$$(p \vee q \vee \neg r) \wedge (\neg p \vee s) \wedge (r \vee \neg s)$$

Every propositional formula can be converted to an equivalent CNF. SAT solvers **require** CNF input — the DPLL and CDCL algorithms operate exclusively on CNF.

#### Disjunctive Normal Form (DNF)

A formula in DNF is a **disjunction of terms** — an OR of ANDs:

$$(p \wedge q \wedge \neg r) \vee (\neg p \wedge s) \vee (r \wedge \neg s)$$

Every formula can be converted to DNF. A DNF is satisfiable iff **any single term** is satisfiable (no complementary literals in a term), which is easy to check — but the conversion to DNF may be exponentially large.

#### Conversion to CNF (Algorithm)

Given an arbitrary formula $\phi$:

1. **Eliminate biconditionals:** $p \leftrightarrow q \longrightarrow (p \to q) \wedge (q \to p)$
2. **Eliminate implications:** $p \to q \longrightarrow \neg p \vee q$
3. **Push negations inward (De Morgan's + double negation):**
   - $\neg(p \wedge q) \longrightarrow \neg p \vee \neg q$
   - $\neg(p \vee q) \longrightarrow \neg p \wedge \neg q$
   - $\neg\neg p \longrightarrow p$
4. **Distribute $\vee$ over $\wedge$:** $p \vee (q \wedge r) \longrightarrow (p \vee q) \wedge (p \vee r)$

**Example:** Convert $p \to (q \wedge r)$ to CNF:

- Step 2: $\neg p \vee (q \wedge r)$
- Step 4: $(\neg p \vee q) \wedge (\neg p \vee r)$ — this is CNF ✓

### 5.7 Boolean Algebra

The laws of propositional logic form an algebraic structure called a **Boolean algebra**: $(B, \wedge, \vee, \neg, 0, 1)$ satisfying all the equivalences listed in §5.5.

**The fundamental connection — Stone Representation Theorem:** Every Boolean algebra is isomorphic to an algebra of sets:

| Logic                   | Sets                  | Python        |
| ----------------------- | --------------------- | ------------- |
| $\wedge$ (AND)          | $\cap$ (intersection) | `&`           |
| $\vee$ (OR)             | $\cup$ (union)        | `\|`          |
| $\neg$ (NOT)            | complement            | `~`           |
| $F$ (false)             | $\emptyset$           | `set()`       |
| $T$ (true)              | $U$                   | universal set |
| $\to$ (implies)         | $\subseteq$           | `<=`          |
| $\leftrightarrow$ (iff) | $=$                   | `==`          |

This is why De Morgan's laws look the same for sets and for logic — they **are** the same, in the precise algebraic sense. Boolean algebra unifies propositional logic, set operations, and circuit design under a single framework.

**AI connections:**

- **Attention masks as Boolean matrices:** An attention mask is a Boolean matrix $M \in \{0, 1\}^{n \times n}$ where $M_{ij} = 1$ iff query $i$ attends to key $j$. Combining masks (causal AND sliding window) is Boolean AND.
- **Structured generation FSMs:** The transition function of a finite state machine operates on Boolean states. Valid token sets at each state are computed via Boolean operations.
- **Circuit complexity of neural networks:** A ReLU network computes a piecewise linear function; each linear region corresponds to a Boolean pattern of active/inactive neurons — the network's behaviour is described by a Boolean function over activation patterns.

---

## 6. Predicate Logic (First-Order Logic)

Propositional logic treats statements as atomic units — it cannot look inside them. Predicate logic (FOL) opens them up, exposing the **objects**, **properties**, and **quantifiers** that make mathematical reasoning powerful. Every formal specification in AI — from type systems to knowledge bases to verification conditions — ultimately rests on predicate logic.

### 6.1 Predicates and Quantifiers — Motivation

The statement "every prime greater than 2 is odd" cannot be expressed in propositional logic. We need:

1. A way to talk about **individual objects** (numbers, tokens, vectors)
2. A way to express **properties** of objects ("is prime," "is odd")
3. A way to say "**for all**" or "**there exists**"

Predicate logic provides all three.

### 6.2 Syntax of FOL

**Terms:** The objects we talk about.

- **Constants:** Specific objects ($0$, $1$, $\pi$, `[PAD]`, `[CLS]`)
- **Variables:** Placeholders for objects ($x$, $y$, $z$)
- **Function symbols:** Operations that produce objects ($f(x)$, $x + y$, $\text{embed}(t)$)

**Predicates (relation symbols):** Properties of or relationships between objects that evaluate to true or false.

- **Unary:** $\text{Prime}(x)$, $\text{IsToken}(t)$, $\text{Positive}(x)$
- **Binary:** $x < y$, $\text{SameCluster}(a, b)$, $\text{Attends}(i, j)$
- **$n$-ary:** $\text{Between}(x, y, z)$ meaning $y$ is between $x$ and $z$

**Quantifiers:** The two pillars of FOL.

#### Universal Quantifier: $\forall$ ("for all")

$$\forall x\, P(x)$$

"For every $x$, $P(x)$ holds." True when $P(a)$ is true for **every** object $a$ in the domain.

**Examples:**

- $\forall x \in \mathbb{R},\, x^2 \geq 0$ — "every real number squared is non-negative"
- $\forall t \in V,\, \text{embed}(t) \in \mathbb{R}^d$ — "every token maps to a $d$-dimensional vector"
- $\forall i, j \in [n],\, i > j \implies M_{ij} = 0$ — causal mask definition

#### Existential Quantifier: $\exists$ ("there exists")

$$\exists x\, P(x)$$

"There exists an $x$ such that $P(x)$ holds." True when $P(a)$ is true for **at least one** object $a$ in the domain.

**Examples:**

- $\exists x \in \mathbb{R},\, x^2 = 2$ — "$\sqrt{2}$ exists (in $\mathbb{R}$)"
- $\exists t \in V,\, P(\text{next} = t \mid \text{context}) > 0.5$ — "there's a token with probability above 0.5"
- $\exists \theta,\, \mathcal{L}(\theta) = 0$ — "a global minimum exists" (not always true!)

#### Unique Existential: $\exists!$ ("there exists exactly one")

$$\exists! x\, P(x) \equiv \exists x\, [P(x) \wedge \forall y\, (P(y) \to y = x)]$$

"There is **exactly one** $x$ satisfying $P(x)$." This is a defined symbol — it's syntactic sugar for the expression on the right.

### 6.3 Free and Bound Variables

A variable $x$ is **bound** in a formula if it appears within the scope of a quantifier $\forall x$ or $\exists x$. Otherwise, it is **free**.

- In $\forall x\, (x > y)$: $x$ is bound, $y$ is free
- In $\exists x\, P(x, y) \wedge Q(z)$: $x$ is bound, $y$ and $z$ are free
- In $\forall x\, \exists y\, (x + y = 0)$: both $x$ and $y$ are bound

A **sentence** (closed formula) has **no free variables** — it has a definite truth value. A formula with free variables is a **predicate** — it becomes a sentence only when specific values are substituted for the free variables or when they are quantified.

> **AI connection:** In a loss function $\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$, the index $i$ is bound (by the sum, a disguised $\forall$), while $\theta$ is free — this is why we optimise **over** $\theta$.

### 6.4 Negating Quantifiers

Negation interacts with quantifiers via two fundamental rules:

$$\neg \forall x\, P(x) \equiv \exists x\, \neg P(x)$$

"Not everything satisfies $P$" $\equiv$ "Something fails to satisfy $P$."

$$\neg \exists x\, P(x) \equiv \forall x\, \neg P(x)$$

"Nothing satisfies $P$" $\equiv$ "Everything fails to satisfy $P$."

These are the quantifier analogues of De Morgan's laws. The pattern generalises:

- Negation swaps $\forall$ and $\exists$
- Negation swaps $\wedge$ and $\vee$
- Negation swaps $\cap$ and $\cup$

> **AI example:** "Not all tokens are attended to" $\equiv$ "There exists a token that is not attended to":
> $$\neg \forall j\, \text{Attends}(i, j) \equiv \exists j\, \neg\text{Attends}(i, j)$$

### 6.5 Nested Quantifiers

Quantifiers can be nested. The **order matters** for mixed quantifiers:

$$\forall x\, \exists y\, P(x, y) \quad \neq \quad \exists y\, \forall x\, P(x, y)$$

The first says: "for every $x$, there is (possibly different) $y$ such that $P(x, y)$."
The second says: "there is a single $y$ that works for all $x$."

**Comparison with examples:**

| Statement                                                    | Formal                                                                                  | True?          |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------- | -------------- |
| "For every real, there's a larger one"                       | $\forall x \in \mathbb{R}\, \exists y \in \mathbb{R}\, (y > x)$                         | True           |
| "There's a real larger than all reals"                       | $\exists y \in \mathbb{R}\, \forall x \in \mathbb{R}\, (y > x)$                         | False          |
| "Every continuous function on $[a,b]$ attains its max"       | $\forall f \in C[a,b]\, \exists x^* \in [a,b]\, \forall x \in [a,b]\, f(x^*) \geq f(x)$ | True (EVT)     |
| "For every $\varepsilon > 0$, there exists $\delta > 0$ ..." | $\forall \varepsilon\, \exists \delta\, \forall x\, \ldots$                             | Continuity def |

**Order rule:** Swapping $\forall$ and $\exists$ is valid only if one direction of implication holds:

$$\exists y\, \forall x\, P(x, y) \implies \forall x\, \exists y\, P(x, y)$$

The converse is NOT generally true. A single universal $y$ is a strictly stronger claim than an individual $y(x)$ for each $x$.

> **AI example — convergence of SGD:**
>
> - $\forall \varepsilon > 0\, \exists T\, \forall t > T\, |\mathcal{L}(\theta_t) - \mathcal{L}^*| < \varepsilon$ means "the loss converges to $\mathcal{L}^*$"
> - Here $T$ depends on $\varepsilon$ — this is the $\forall\exists$ pattern

### 6.6 Translating Between English and FOL

Translating natural language to FOL is a critical skill for formal verification, knowledge representation, and understanding what mathematical theorems actually say.

**Systematic approach:**

1. Identify the **domain** (what are the objects?)
2. Identify the **predicates** (what properties/relations?)
3. Identify the **quantifier structure** ($\forall$, $\exists$, their order)
4. Write the formal expression
5. Check by reading it back in English

**Practice translations:**

| English                                              | FOL                                                                                          |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| "All vectors in $S$ have unit norm"                  | $\forall v \in S,\, \|v\| = 1$                                                               |
| "Some weight is negative"                            | $\exists w \in W,\, w < 0$                                                                   |
| "No gradient is zero"                                | $\forall g \in G,\, g \neq 0$ or $\neg\exists g \in G,\, g = 0$                              |
| "If all gradients vanish, we're at a critical point" | $(\forall i,\, \nabla_i f = 0) \to \text{CriticalPoint}(\theta)$                             |
| "For every layer, there exists a neuron that fires"  | $\forall \ell\, \exists n\, \text{Fires}(\ell, n)$                                           |
| "There's a learning rate that works for all tasks"   | $\exists \eta\, \forall \text{task}\, \text{Converges}(\eta, \text{task})$ (probably false!) |

### 6.7 FOL in AI Contexts

**Knowledge graphs** store facts as FOL ground atoms: $\text{LocatedIn}(\text{Paris}, \text{France})$, $\text{InstanceOf}(\text{GPT-4}, \text{LLM})$. Rules are FOL implications: $\forall x, y, z\, [\text{LocatedIn}(x, y) \wedge \text{LocatedIn}(y, z) \to \text{LocatedIn}(x, z)]$.

**Description logics** (OWL, used in ontologies) are decidable fragments of FOL. They balance expressiveness with computational tractability — reasoning in full FOL is undecidable.

**Program specification:** Hoare logic uses FOL to specify pre/post-conditions: $\{P\}\, \text{program}\, \{Q\}$ means "if $P$ holds before, $Q$ holds after."

**LLM reasoning:** When an LLM performs chain-of-thought reasoning, it is (approximately) performing FOL inference — applying modus ponens, instantiating universals, existential witnesses. The fidelity of this approximation is an active research question.

---

## 7. Proof Techniques

Proofs are the technology of certainty. Unlike empirical verification (running more test cases) or statistical evidence (p-values), a proof establishes truth with **absolute logical necessity**. Every step follows from axioms and previously proven results via rules of inference. Even if you never write formal proofs in your AI work, understanding proof techniques is essential for:

- Reading and understanding theoretical papers
- Verifying that your mathematical reasoning is correct
- Understanding why certain algorithms work (and when they don't)
- Debug mathematical arguments in AI (convergence proofs, generalization bounds)

### 7.1 What is a Proof?

A **proof** of a statement $P$ is a finite sequence of logical steps, each of which is:

1. An **axiom** or previously proven theorem (assumed true)
2. A **hypothesis** (assumed true for purposes of the proof)
3. A step derived from previous steps by a **rule of inference** (modus ponens, universal instantiation, etc.)

The final step is the statement $P$ itself.

**Standard of proof:** Every step must be **logically necessary**, not just "reasonable" or "highly probable." A single gap in the chain — no matter how small — invalidates the proof.

### 7.2 Direct Proof

**Structure:** To prove $P \to Q$, assume $P$ is true and derive $Q$ through a chain of logical steps.

**Template:**

> **Theorem:** If $P$, then $Q$.
>
> **Proof.** Assume $P$. [Chain of implications.] Therefore $Q$. $\square$

**Example 1 — Sum of two even integers is even:**

> **Theorem.** If $m$ and $n$ are even integers, then $m + n$ is even.
>
> **Proof.** Assume $m$ and $n$ are even. By definition, $m = 2a$ and $n = 2b$ for some integers $a$ and $b$. Then:
> $$m + n = 2a + 2b = 2(a + b)$$
> Since $a + b$ is an integer, $m + n = 2(a + b)$ is even. $\square$

**Example 2 — If $\|v\| = 1$ and $\|w\| = 1$, then $|v \cdot w| \leq 1$:**

> **Theorem.** For unit vectors $v, w \in \mathbb{R}^n$, $|v \cdot w| \leq 1$.
>
> **Proof.** By the Cauchy-Schwarz inequality, $|v \cdot w| \leq \|v\| \cdot \|w\| = 1 \cdot 1 = 1$. $\square$

### 7.3 Proof by Contrapositive

**Structure:** To prove $P \to Q$, prove the logically equivalent **contrapositive** $\neg Q \to \neg P$. Often easier — start with the conclusion being false and show the hypothesis must also be false.

**Template:**

> **Theorem:** If $P$, then $Q$.
>
> **Proof.** We prove the contrapositive: if $\neg Q$, then $\neg P$. Assume $\neg Q$. [Derivation.] Therefore $\neg P$. $\square$

**Example — If $n^2$ is even, then $n$ is even:**

> **Theorem.** For integer $n$, if $n^2$ is even, then $n$ is even.
>
> **Proof.** We prove the contrapositive: if $n$ is odd, then $n^2$ is odd.
> Assume $n$ is odd. Then $n = 2k + 1$ for some integer $k$.
> $$n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$$
> Since $2k^2 + 2k$ is an integer, $n^2$ is odd. $\square$

> **AI analogy:** Contrapositive proofs often appear in impossibility results. "If a model generalises, it must have finite VC dimension" is equivalent to "if VC dimension is infinite, the model does not generalise."

### 7.4 Proof by Contradiction

**Structure:** To prove $P$, assume $\neg P$ and derive a **contradiction** (a statement of the form $Q \wedge \neg Q$). Since the assumption of $\neg P$ leads to an impossibility, $P$ must be true.

**Template:**

> **Theorem:** $P$.
>
> **Proof.** Assume, for the sake of contradiction, that $\neg P$. [Derivation.] This gives $Q \wedge \neg Q$, a contradiction. Therefore $P$ must hold. $\square$

**Example 1 — $\sqrt{2}$ is irrational:**

> **Theorem.** $\sqrt{2}$ is irrational.
>
> **Proof.** Assume, for contradiction, that $\sqrt{2}$ is rational. Then $\sqrt{2} = \frac{a}{b}$ where $a, b$ are integers with $\gcd(a, b) = 1$ (reduced form).
>
> Squaring: $2 = \frac{a^2}{b^2}$, so $a^2 = 2b^2$.
>
> Since $a^2$ is even, $a$ is even (by the contrapositive result above). Write $a = 2c$.
>
> Then $(2c)^2 = 2b^2 \implies 4c^2 = 2b^2 \implies b^2 = 2c^2$.
>
> So $b^2$ is even, hence $b$ is even. But then both $a$ and $b$ are even, contradicting $\gcd(a, b) = 1$. $\square$

**Example 2 — There are infinitely many primes (Euclid):**

> **Theorem.** There are infinitely many prime numbers.
>
> **Proof.** Assume, for contradiction, that there are only finitely many primes: $p_1, p_2, \ldots, p_n$. Consider $N = p_1 p_2 \cdots p_n + 1$.
>
> $N > 1$, so $N$ has a prime factor $p$. But $N$ is not divisible by any $p_i$ (since $N \div p_i$ leaves remainder 1). So $p \neq p_i$ for all $i$ — a prime not on our list. Contradiction. $\square$

### 7.5 Proof by Cases (Exhaustion)

**Structure:** To prove $P$, partition the possibilities into cases that cover all situations, and prove $P$ in each case.

**Template:**

> **Theorem:** $P$.
>
> **Proof.** There are $k$ cases to consider.
>
> **Case 1:** [Assumption.] [Proof of $P$.]
>
> **Case 2:** [Assumption.] [Proof of $P$.]
>
> $\vdots$
>
> Since the cases are exhaustive, $P$ holds. $\square$

**Example — $|x| \geq 0$ for all $x \in \mathbb{R}$:**

> **Proof.**
>
> **Case 1:** $x \geq 0$. Then $|x| = x \geq 0$.
>
> **Case 2:** $x < 0$. Then $|x| = -x > 0 \geq 0$.
>
> Both cases give $|x| \geq 0$. $\square$

### 7.6 Mathematical Induction

**Structure:** To prove that a property $P(n)$ holds for all natural numbers $n \geq n_0$:

1. **Base case:** Prove $P(n_0)$.
2. **Inductive step:** Prove that for any $k \geq n_0$, if $P(k)$ is true (the **inductive hypothesis**), then $P(k+1)$ is true.

The principle of induction then guarantees $P(n)$ for all $n \geq n_0$: the base case starts the chain, and the inductive step propagates it.

**Why it works:** Induction is an axiom of the natural numbers (Peano's fifth axiom / ZFC axiom of infinity). It formalises "domino reasoning" — knock over the first domino and ensure each domino knocks over the next.

**Example — Sum formula:**

> **Theorem.** For all $n \geq 1$: $\sum_{i=1}^n i = \frac{n(n+1)}{2}$.
>
> **Proof.**
>
> **Base case ($n = 1$):** $\sum_{i=1}^1 i = 1 = \frac{1 \cdot 2}{2}$. ✓
>
> **Inductive step:** Assume $\sum_{i=1}^k i = \frac{k(k+1)}{2}$ for some $k \geq 1$. Then:
> $$\sum_{i=1}^{k+1} i = \left(\sum_{i=1}^k i\right) + (k+1) = \frac{k(k+1)}{2} + (k+1) = \frac{k(k+1) + 2(k+1)}{2} = \frac{(k+1)(k+2)}{2}$$
> This is the formula with $n = k+1$. $\square$

**Strong induction:** Instead of assuming only $P(k)$, assume $P(n_0), P(n_0 + 1), \ldots, P(k)$ (all predecessors). Sometimes needed when $P(k+1)$ depends on cases earlier than $P(k)$.

> **AI connection — recursive proof and RNNs:** Induction proofs about sequence models (RNNs, autoregressive LLMs) often use induction on sequence length. The base case handles $n = 1$ (single token), and the inductive step shows that if the property holds for sequences of length $k$, it holds for length $k + 1$ (adding one token).

### 7.7 Existence and Uniqueness Proofs

**Existence proof:** Show that an object with certain properties exists. Two approaches:

1. **Constructive:** Explicitly construct the object and verify it has the claimed properties.
   - "There exists $x$ with $x^2 = 4$. Take $x = 2$. Then $2^2 = 4$. ✓"
   - Constructive proofs are preferred in CS because they yield algorithms.

2. **Non-constructive:** Prove existence indirectly (e.g., by contradiction), without exhibiting the object.
   - "There exist irrational numbers $a, b$ such that $a^b$ is rational."
   - **Proof.** Consider $\sqrt{2}^{\sqrt{2}}$. If it's rational, take $a = b = \sqrt{2}$. If it's irrational, take $a = \sqrt{2}^{\sqrt{2}}$ and $b = \sqrt{2}$; then $a^b = (\sqrt{2}^{\sqrt{2}})^{\sqrt{2}} = \sqrt{2}^2 = 2$. Either way, such $a, b$ exist. $\square$
   - We proved existence but don't know _which_ case holds.

**Uniqueness proof:** Show that at most one object with certain properties exists. Standard technique: assume two objects satisfy the conditions and prove they must be equal.

**Existence + Uniqueness:** Combine both. For "$\exists! x\, P(x)$":

1. **Existence:** Show at least one $x$ satisfies $P$.
2. **Uniqueness:** Suppose $P(x_1)$ and $P(x_2)$; show $x_1 = x_2$.

> **AI example — convergence to unique minimum:** "A strictly convex loss function has a unique global minimum."
>
> - **Existence:** Follow from the extreme value theorem (for bounded below, coercive functions).
> - **Uniqueness:** Assume $\theta_1 \neq \theta_2$ both minimise $\mathcal{L}$. Then by strict convexity, $\mathcal{L}\left(\frac{\theta_1 + \theta_2}{2}\right) < \frac{\mathcal{L}(\theta_1) + \mathcal{L}(\theta_2)}{2} = \mathcal{L}^*$, contradicting the assumption that $\mathcal{L}^*$ is the minimum.

### 7.8 Proof by Construction

A **constructive proof** proves existence by explicitly building the desired object. This is the most practically useful proof technique because it yields an **algorithm**.

**Example — For any finite set $A$ with $|A| = n \geq 2$, there exists a partition of $A$ into two non-empty subsets.**

> **Proof.** Since $|A| \geq 2$, there exist distinct $a_1, a_2 \in A$. Define $B_1 = \{a_1\}$ and $B_2 = A \setminus \{a_1\}$. Then $B_1 \neq \emptyset$ (contains $a_1$), $B_2 \neq \emptyset$ (contains $a_2$), $B_1 \cap B_2 = \emptyset$, and $B_1 \cup B_2 = A$. $\square$

> **AI example — Universal approximation:** The universal approximation theorem (Cybenko 1989, Hornik 1991) proves that for any continuous function $f$ on a compact domain and any $\varepsilon > 0$, there exists a neural network $N$ with $\|N - f\|_\infty < \varepsilon$. The original proof is non-constructive (uses the Hahn-Banach theorem). Later constructive proofs actually build the approximating network layer by layer — these are far more useful for practitioners.

### 7.9 Summary of Proof Techniques

| Technique                  | To prove                     | Strategy                            | When to use                                                          |
| -------------------------- | ---------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| Direct                     | $P \to Q$                    | Assume $P$, derive $Q$              | When the path from $P$ to $Q$ is clear                               |
| Contrapositive             | $P \to Q$                    | Prove $\neg Q \to \neg P$           | When $\neg Q$ gives more structure to work with                      |
| Contradiction              | $P$                          | Assume $\neg P$, derive paradox     | When direct approach is hard; especially for "no such object" claims |
| Cases                      | $P$                          | Split into exhaustive cases         | When the domain naturally partitions                                 |
| Induction                  | $\forall n \geq n_0,\, P(n)$ | Base + step                         | Statements about all natural numbers or recursive structures         |
| Constructive existence     | $\exists x\, P(x)$           | Build $x$ explicitly                | When you need an algorithm or concrete example                       |
| Non-constructive existence | $\exists x\, P(x)$           | Show non-existence is impossible    | When construction is unknown or unnecessary                          |
| Uniqueness                 | $\exists! x\, P(x)$          | Existence + assume two, prove equal | When you need "the" rather than "a"                                  |

---

## 8. Axiomatic Set Theory (ZFC)

Naive set theory (§2) is intuitive but broken — Russell's Paradox (§2.5) shows that unrestricted set formation leads to contradictions. **ZFC** (Zermelo–Fraenkel axioms with the Axiom of Choice) is the standard rigorous foundation for all of modern mathematics. Every mathematical object — numbers, functions, spaces, probability measures, neural networks — can be built from ZFC sets.

### 8.1 Why Axioms?

Axioms serve two purposes:

1. **Prevent paradoxes:** By restricting which sets exist, ZFC avoids Russell's Paradox and similar contradictions.
2. **Ensure consistency:** All mathematical reasoning ultimately reduces to statements provable from these axioms.

The axioms tell you exactly **which sets you are allowed to construct** and **what operations you may perform** on them. Nothing more, nothing less.

### 8.2 The ZFC Axioms

There are 9 axioms (some formulations give 8 or 10; the exact count depends on grouping). Here they are, with intuition and notation.

#### Axiom 1: Extensionality

$$\forall A\, \forall B\, [\forall x\, (x \in A \leftrightarrow x \in B) \to A = B]$$

Two sets are equal **if and only if** they have the same elements. A set is determined entirely by its membership — there is no "hidden structure."

> **Consequence:** $\{1, 2, 3\} = \{3, 1, 2\} = \{1, 1, 2, 3\}$. Order and repetition are irrelevant.

#### Axiom 2: Empty Set

$$\exists \emptyset\, \forall x\, (x \notin \emptyset)$$

There exists a set with no elements. By Extensionality, this set is unique — there is only one empty set.

#### Axiom 3: Pairing

$$\forall a\, \forall b\, \exists C\, \forall x\, (x \in C \leftrightarrow x = a \vee x = b)$$

For any two objects $a$ and $b$, the set $\{a, b\}$ exists. This lets us form pairs, which underpins ordered pairs and Cartesian products.

#### Axiom 4: Union

$$\forall \mathcal{F}\, \exists U\, \forall x\, (x \in U \leftrightarrow \exists A \in \mathcal{F},\, x \in A)$$

For any collection of sets $\mathcal{F}$, there exists a set $U = \bigcup \mathcal{F}$ containing exactly the elements that belong to at least one set in $\mathcal{F}$.

#### Axiom 5: Power Set

$$\forall A\, \exists P\, \forall B\, (B \in P \leftrightarrow B \subseteq A)$$

For any set $A$, the power set $\mathcal{P}(A)$ exists. This is the set of **all subsets** of $A$, and $|\mathcal{P}(A)| = 2^{|A|}$ for finite $A$.

#### Axiom 6: Separation (Specification / Comprehension)

$$\forall A\, \exists B\, \forall x\, (x \in B \leftrightarrow x \in A \wedge \phi(x))$$

For any set $A$ and any property $\phi$, the set $\{x \in A \mid \phi(x)\}$ exists. Crucially, you can only **separate out** elements from an **existing set** — you cannot form $\{x \mid \phi(x)\}$ without a bounding set $A$.

> **This is what blocks Russell's Paradox.** The "set of all sets that don't contain themselves" would require $\{x \mid x \notin x\}$ — but without a bounding set $A$, Separation doesn't let you form it.

#### Axiom 7: Infinity

$$\exists I\, [\emptyset \in I \wedge \forall x\, (x \in I \to x \cup \{x\} \in I)]$$

There exists an infinite set — specifically, a set containing $\emptyset$, $\{\emptyset\}$, $\{\emptyset, \{\emptyset\}\}$, and so on forever. This axiom guarantees the existence of the natural numbers (and prevents mathematics from being stuck in a finite world).

#### Axiom 8: Replacement

$$\forall A\, \forall F\, [\text{($F$ is a function)} \to \exists B\, B = \{F(x) \mid x \in A\}]$$

If $F$ is a definable function and $A$ is a set, then the image $\{F(x) \mid x \in A\}$ is a set. This is like Separation but more powerful — it lets you apply a function to every element and collect the results.

> **AI analogy:** Replacement is `map`. Given a set (dataset) and a function (model), the outputs form a set (predictions).

#### Axiom 9: Foundation (Regularity)

$$\forall A\, [A \neq \emptyset \to \exists x \in A\, (x \cap A = \emptyset)]$$

Every non-empty set $A$ has an element disjoint from $A$. This prevents circular membership chains ($a \in b \in a$) and self-containing sets ($a \in a$). Sets are "well-founded" — they are built from the bottom up (from $\emptyset$).

### 8.3 The Axiom of Choice (AC)

$$\forall \mathcal{F}\, [\emptyset \notin \mathcal{F} \to \exists f: \mathcal{F} \to \bigcup \mathcal{F}\, \forall A \in \mathcal{F},\, f(A) \in A]$$

For any collection of non-empty sets, there exists a **choice function** $f$ that selects exactly one element from each set. This seems obvious but has deep consequences.

**Why it's controversial:**

- It's non-constructive: it asserts the existence of a choice function without building one
- It implies paradoxical results: Banach-Tarski paradox (decompose a sphere into 5 pieces and reassemble into two spheres of the same size)
- It implies well-ordering theorem: every set can be well-ordered (including $\mathbb{R}$, despite having no known explicit well-ordering)

**Why it's accepted:**

- Without AC, many theorems fail (every vector space has a basis; Tychonoff's theorem; Hahn-Banach theorem)
- ZFC (with AC) has not been shown inconsistent
- AC is independent of ZF: both ZF+AC and ZF+¬AC are consistent (if ZF is consistent), by Gödel (1938) and Cohen (1963)

**AI applications of AC:** Most machine learning theory implicitly uses AC:

- "There exists a minimiser of the loss function" — may require AC for non-compact spaces
- "Every vector space has a basis" — needs AC for infinite-dimensional spaces (function spaces in kernel methods)
- In practice, everything is finite-dimensional, so AC is not needed for actual computations

### 8.4 Constructing Numbers from Sets

One of the most remarkable achievements of axiomatic set theory is building all number systems from pure sets:

**Natural numbers (von Neumann construction):**

$$0 = \emptyset, \quad 1 = \{\emptyset\}, \quad 2 = \{\emptyset, \{\emptyset\}\}, \quad 3 = \{\emptyset, \{\emptyset\}, \{\emptyset, \{\emptyset\}\}\}, \quad \ldots$$

The rule: $n + 1 = n \cup \{n\}$. Each natural number is the set of all smaller natural numbers: $n = \{0, 1, \ldots, n-1\}$, so $|n| = n$.

**Integers:** $\mathbb{Z}$ is constructed as equivalence classes of pairs of natural numbers: $[(a, b)]$ represents $a - b$.

**Rationals:** $\mathbb{Q}$ is constructed as equivalence classes of pairs of integers: $[(p, q)]$ represents $p/q$ where $q \neq 0$.

**Reals:** $\mathbb{R}$ is constructed via **Dedekind cuts** or **Cauchy sequences** of rationals.

> **The takeaway:** Every mathematical object — including the real numbers that neural network weights live in — is built from the empty set using ZFC axioms. This is why set theory is called the "foundation of mathematics."

### 8.5 Ordinals and Cardinals (Brief)

**Ordinal numbers** extend the natural numbers to describe the "length" of well-orderings:

$$0, 1, 2, \ldots, \omega, \omega + 1, \omega + 2, \ldots, \omega \cdot 2, \ldots, \omega^2, \ldots, \omega^\omega, \ldots, \varepsilon_0, \ldots$$

Here $\omega$ is the first infinite ordinal (the "ordinal type" of $\mathbb{N}$). Ordinals are used to measure the "height" of sets in the cumulative hierarchy.

**Cardinal numbers** measure the "size" of sets. For finite sets, the cardinal and ordinal notions coincide ($|S| = n$ means $S$ has $n$ elements). For infinite sets, cardinality is defined via bijections — two sets have the same cardinality iff a bijection exists between them. More on this in §10.

---

## 9. Logic in Computation and AI

This section bridges the pure theory of §5-8 to the concrete computational world. The key idea: logic is not just a tool for proofs — it is the foundation of computation itself.

### 9.1 Boolean Circuits

Every digital computer is built from **logic gates** — physical implementations of the logical connectives:

| Gate | Logic              | Symbol | Output                        |
| ---- | ------------------ | ------ | ----------------------------- |
| AND  | $p \wedge q$       | `&`    | 1 iff both inputs are 1       |
| OR   | $p \vee q$         | `\|`   | 1 iff at least one input is 1 |
| NOT  | $\neg p$           | `~`    | Flips the bit                 |
| NAND | $\neg(p \wedge q)$ | —      | 0 iff both inputs are 1       |
| XOR  | $p \oplus q$       | `^`    | 1 iff inputs differ           |

**Functional completeness:** $\{AND, OR, NOT\}$ can compute **any** Boolean function. Even more remarkably, $\{NAND\}$ alone is functionally complete — every Boolean function can be built from NAND gates.

**From gates to GPUs:** GPU cores perform massively parallel Boolean and arithmetic operations. The floating-point units, comparison circuits, and memory addressing logic in a GPU are all Boolean circuits. When you run `torch.matmul(A, B)`, underneath it is billions of transistors implementing Boolean functions.

### 9.2 SAT Solving

The **Boolean satisfiability problem** (SAT): given a propositional formula in CNF, is there a truth assignment that makes it true?

**DPLL Algorithm** (Davis–Putnam–Logemann–Loveland, 1962):

1. **Unit propagation:** If a clause has one unset literal, it must be true
2. **Pure literal elimination:** If a variable appears with only one polarity, set it to satisfy those clauses
3. **Branching:** Pick an unset variable, try both truth values, recurse
4. **Backtracking:** If a contradiction is found, undo the last choice

**CDCL** (Conflict-Driven Clause Learning): Modern SAT solvers extend DPLL with **learned clauses** — when a conflict occurs, they analyse why and add a clause preventing the same conflict pattern. This is a form of learning from mistakes.

**SAT in AI:**

- **Constraint satisfaction:** Many combinatorial AI problems reduce to SAT
- **Planning:** Classical AI planning (SATPLAN) encodes action sequences as SAT instances
- **Formal verification:** Proving neural network properties (robustness, fairness) reduces to SAT/SMT
- **Hardware design:** Chip verification uses SAT solvers extensively
- **Structured generation:** Ensuring LLM output satisfies a grammar can be framed as constraint satisfaction

### 9.3 Resolution and Automated Reasoning

**Resolution** is a single inference rule that is **complete** for refuting CNF formulas:

$$\frac{(A \vee p) \quad (\neg p \vee B)}{A \vee B}$$

If one clause contains $p$ and another contains $\neg p$, their **resolvent** combines the remaining literals.

**Completeness (Robinson, 1965):** A set of clauses is unsatisfiable if and only if the empty clause ($\bot$) can be derived by repeated resolution.

This is the backbone of **Prolog** and logic programming — computation as proof search.

### 9.4 Type Theory and Programming

Type systems are deductive systems closely related to logic. The **Curry-Howard correspondence** establishes a deep isomorphism:

| Logic                      | Type Theory                         | Programming                  |
| -------------------------- | ----------------------------------- | ---------------------------- |
| Proposition                | Type                                | Specification                |
| Proof                      | Term (program)                      | Implementation               |
| $P \to Q$ (implication)    | Function type $P \to Q$             | Function from $P$ to $Q$     |
| $P \wedge Q$ (conjunction) | Product type $P \times Q$           | Pair/tuple                   |
| $P \vee Q$ (disjunction)   | Sum type $P + Q$                    | Tagged union / either        |
| $\forall x\, P(x)$         | Dependent function $\Pi_{x:A} P(x)$ | Generic/polymorphic function |
| $\exists x\, P(x)$         | Dependent pair $\Sigma_{x:A} P(x)$  | Record with witness          |
| $\bot$ (false)             | Empty type                          | Unreachable code             |
| Proof of $P$               | Value of type $P$                   | Running program              |

**Consequence:** A valid type-checked program **is** a proof of its type signature. Type errors are logical contradictions.

> **AI connection — type-safe tensors:** Projects like `jaxtyping` and `beartype` enforce tensor shape constraints at the type level: `Float[Array, "batch seq d_model"]`. A shape mismatch is caught as a type error — i.e., a logical contradiction between what the function promises and what it receives.

### 9.5 Modal Logic (Brief)

Modal logic extends propositional logic with **necessity** ($\square$) and **possibility** ($\diamond$):

- $\square P$: "$P$ is necessarily true" (true in all possible worlds)
- $\diamond P$: "$P$ is possibly true" (true in some possible world)

**In AI:**

- **Epistemic logic:** $K_a P$ means "agent $a$ knows $P$." Used in multi-agent systems.
- **Temporal logic:** $\square P$ means "$P$ always holds" (safety); $\diamond P$ means "$P$ eventually holds" (liveness). Used in model checking and formal verification.
- **Deontic logic:** What an AI system "should" do — the logic of obligations and permissions, relevant to AI alignment.

### 9.6 Description Logics and Knowledge Representation

**Description logics** (DLs) are decidable fragments of FOL designed for knowledge representation:

- **Concepts** (unary predicates): `Person`, `Model`, `LargeLanguageModel`
- **Roles** (binary predicates): `trainedBy`, `hasParameter`, `outputs`
- **Constructors:** $C \sqcap D$ (intersection), $C \sqcup D$ (union), $\neg C$ (complement), $\forall R.C$ (all $R$-successors are in $C$), $\exists R.C$ (some $R$-successor is in $C$)

DLs underpin the **Web Ontology Language (OWL)**, used in knowledge graphs, biomedical ontologies, and semantic web. The reasoning complexity depends on which constructors are included — a careful balance between expressiveness and decidability.

### 9.7 LLM Reasoning as Approximate Logic

When an LLM performs chain-of-thought (CoT) reasoning, it is (approximately) performing logical inference:

| Logical step                                           | LLM behaviour                                  | Fidelity              |
| ------------------------------------------------------ | ---------------------------------------------- | --------------------- |
| Modus ponens ($P, P \to Q \vdash Q$)                   | "Since $P$ and $P$ implies $Q$, therefore $Q$" | High for simple cases |
| Universal instantiation ($\forall x P(x) \vdash P(a)$) | "Since all X are Y, this X is Y"               | Moderate              |
| Contradiction detection                                | "But this contradicts..."                      | Fragile               |
| Multi-step chains                                      | Chaining 5+ inference steps                    | Degrades rapidly      |

**Open research questions:**

- Can LLMs perform **reliable** deductive reasoning, or only pattern-matched approximation?
- Can we formally verify LLM reasoning chains against logical rules?
- Can symbolic logic modules be integrated with neural networks (neuro-symbolic AI)?
- How does the "chain-of-thought" prompt format affect logical correctness?

Current evidence suggests LLMs are surprisingly good at local logical steps but struggle with long chains and quantifier reasoning (especially nested $\forall\exists$ patterns). This is precisely the gap that formal methods aim to fill.

---

## 10. Cardinality and Infinite Sets

Cardinality answers the question: **how big is a set?** For finite sets the answer is obvious — just count. For infinite sets, the answer is one of the most profound discoveries in mathematics: **not all infinities are the same size**.

### 10.1 Finite Cardinality

If $A$ is finite, $|A| = n$ means there exists a bijection $f: A \to \{1, 2, \ldots, n\}$. Counting is bijection.

**Key counting facts for AI:**

| Formula                                                                       | Name                | AI Example                          |
| ----------------------------------------------------------------------------- | ------------------- | ----------------------------------- |
| $\vert A \cup B \vert = \vert A \vert + \vert B \vert - \vert A \cap B \vert$ | Inclusion-exclusion | Deduplicating token sets            |
| $\vert A \times B \vert = \vert A \vert \cdot \vert B \vert$                  | Product rule        | Size of joint state space           |
| $\vert \mathcal{P}(A) \vert = 2^{\vert A \vert}$                              | Power set           | Number of feature subsets           |
| $\vert A^B \vert = \vert A \vert^{\vert B \vert}$                             | Exponential         | Number of functions from $B$ to $A$ |
| $n! = n \cdot (n-1) \cdots 1$                                                 | Factorial           | Permutation count                   |
| $\binom{n}{k} = \frac{n!}{k!(n-k)!}$                                          | Binomial            | Ways to choose $k$ items from $n$   |

### 10.2 Comparing Infinite Sets — Bijections

**Definition:** Two sets $A$ and $B$ have the same cardinality ($|A| = |B|$) if and only if there exists a bijection $f: A \to B$.

This is the only sensible definition for infinite sets — we cannot "count" them, but we can pair their elements one-to-one.

**Example — $|\mathbb{N}| = |\mathbb{Z}|$:**

This seems wrong — $\mathbb{Z}$ "has twice as many elements." But pair them:

$$0 \mapsto 0, \quad 1 \mapsto 1, \quad 2 \mapsto -1, \quad 3 \mapsto 2, \quad 4 \mapsto -2, \quad 5 \mapsto 3, \quad \ldots$$

Formally: $f(n) = \begin{cases} n/2 & \text{if $n$ is even} \\ -(n+1)/2 & \text{if $n$ is odd} \end{cases}$

This is a bijection, so $|\mathbb{N}| = |\mathbb{Z}|$. Infinity doesn't work like finite numbers.

### 10.3 Countable Sets

A set $A$ is **countably infinite** if $|A| = |\mathbb{N}|$ — i.e., there exists a bijection $A \to \mathbb{N}$. A set is **countable** if it is finite or countably infinite.

Equivalently, $A$ is countable iff its elements can be listed as a sequence $a_1, a_2, a_3, \ldots$ (possibly terminating).

**Countable sets include:**

| Set                                              | Why countable                                                              |
| ------------------------------------------------ | -------------------------------------------------------------------------- |
| $\mathbb{N}$                                     | By definition                                                              |
| $\mathbb{Z}$                                     | Bijection shown above                                                      |
| $\mathbb{Q}$                                     | Cantor's zig-zag (diagonalisation of $\frac{p}{q}$ table)                  |
| $\mathbb{N} \times \mathbb{N}$                   | Cantor pairing function: $(m, n) \mapsto \frac{(m+n)(m+n+1)}{2} + m$       |
| All finite strings over finite alphabet $\Sigma$ | $\Sigma^* = \bigcup_{n=0}^\infty \Sigma^n$, countable union of finite sets |
| All programs (source code)                       | Programs are finite strings                                                |
| All rational polynomials                         | Countable coefficients, finite degree                                      |
| Algebraic numbers                                | Roots of polynomials with integer coefficients                             |

**Key theorem:** A countable union of countable sets is countable:

$$A_1, A_2, A_3, \ldots \text{ countable} \implies \bigcup_{i=1}^\infty A_i \text{ countable}$$

> **AI consequence:** The set of all possible tokenised prompts (finite strings over a finite vocabulary $V$) is **countable**. There are $\aleph_0$ possible inputs to an LLM.

### 10.4 Uncountable Sets — Cantor's Diagonal Argument

**Theorem (Cantor, 1891).** $\mathbb{R}$ is uncountable — there is **no** bijection $\mathbb{N} \to \mathbb{R}$.

**Proof (diagonal argument).** We show that even the interval $(0, 1)$ is uncountable. Assume, for contradiction, that $(0, 1)$ is countable. Then we can list all real numbers in $(0, 1)$:

$$r_1 = 0.d_{11}d_{12}d_{13}d_{14}\ldots$$
$$r_2 = 0.d_{21}d_{22}d_{23}d_{24}\ldots$$
$$r_3 = 0.d_{31}d_{32}d_{33}d_{34}\ldots$$
$$\vdots$$

Construct $r^* = 0.d_1^* d_2^* d_3^* \ldots$ where $d_n^*$ is any digit $\neq d_{nn}$ (and $\neq 0, 9$ to avoid alternative decimal representations).

Then $r^* \in (0, 1)$ but $r^* \neq r_n$ for every $n$ (they differ at the $n$-th digit). This contradicts the assumption that **all** reals in $(0, 1)$ were listed. $\square$

The symbol $|\mathbb{N}| = \aleph_0$ ("aleph-null") denotes the first infinite cardinal. The set $\mathbb{R}$ has cardinality $|\mathbb{R}| = 2^{\aleph_0} = \mathfrak{c}$ (the cardinality of the **continuum**), which is strictly larger.

### 10.5 Cantor's Theorem (Generalised)

**Theorem.** For any set $A$: $|A| < |\mathcal{P}(A)|$.

No set is as large as its power set. This holds for **every** set, finite or infinite.

**Proof sketch.** The map $a \mapsto \{a\}$ injects $A$ into $\mathcal{P}(A)$, so $|A| \leq |\mathcal{P}(A)|$. To show strict inequality, suppose $f: A \to \mathcal{P}(A)$ is a bijection. Let $D = \{a \in A \mid a \notin f(a)\}$. Then $D \in \mathcal{P}(A)$, so $D = f(d)$ for some $d$ (by surjectivity). Is $d \in D$?

- If $d \in D$, then $d \notin f(d) = D$ — contradiction.
- If $d \notin D$, then $d \in f(d) = D$ — contradiction.

Either way, contradiction. So no such bijection exists. $\square$

**Consequence — the infinite tower:**

$$|\mathbb{N}| < |\mathcal{P}(\mathbb{N})| < |\mathcal{P}(\mathcal{P}(\mathbb{N}))| < \cdots$$

There is no largest infinity. The hierarchy of infinities is itself infinite (and larger than any infinity in the hierarchy).

### 10.6 The Continuum Hypothesis

**Claim (Cantor, 1878):** There is no set $S$ with $|\mathbb{N}| < |S| < |\mathbb{R}|$.

In other words, there is no "intermediate" infinity between the countable and the continuum.

**Resolution (Gödel 1940, Cohen 1963):** The Continuum Hypothesis is **independent** of ZFC — it can neither be proved nor disproved from the standard axioms. Both ZFC + CH and ZFC + ¬CH are consistent (if ZFC is).

This is one of the most surprising results in mathematics: there are meaningful mathematical questions that our axioms simply cannot answer.

### 10.7 Cardinality and AI

| Object                                        | Cardinality                                  | Consequence                                        |
| --------------------------------------------- | -------------------------------------------- | -------------------------------------------------- |
| Finite vocabulary $V$                         | $\vert V \vert$ (finite, e.g., 32,000)       | Discrete, tractable                                |
| All prompts $V^*$                             | $\aleph_0$ (countable)                       | Enumerable in principle                            |
| All real-valued weight vectors $\mathbb{R}^d$ | $\mathfrak{c}$                               | Uncountable — can't enumerate models               |
| All functions $\mathbb{R}^d \to \mathbb{R}$   | $\mathfrak{c}^{\mathfrak{c}} > \mathfrak{c}$ | Vastly larger than the set of computable functions |
| All computable functions                      | $\aleph_0$                                   | Programs are countable strings                     |

**The gap:** There are uncountably many functions but only countably many programs (and neural network architectures, at any fixed precision). Most functions are **not computable** — and most mathematical objects are **not representable** by any model. This is a fundamental limitation.

---

## 11. Logic and Set Theory in Machine Learning

This section shows how the abstract tools of §2–10 appear concretely in modern ML.

### 11.1 Formal Languages and Automata

**Formal language theory** is set theory applied to strings:

- An **alphabet** $\Sigma$ is a finite set of symbols
- A **string** over $\Sigma$ is a finite sequence of symbols from $\Sigma$
- A **language** $L$ is a set of strings: $L \subseteq \Sigma^*$

**Connection to AI:**

| Concept               | ML Instantiation                                          |
| --------------------- | --------------------------------------------------------- |
| Alphabet $\Sigma$     | Vocabulary $V$ (set of tokens)                            |
| String                | Token sequence (prompt or completion)                     |
| Language $L$          | Set of valid outputs                                      |
| Grammar               | Rules defining valid outputs (JSON schema, Python syntax) |
| Automaton (DFA/NFA)   | Constrained decoding state machine                        |
| Regular language      | Languages recognisable by finite automata                 |
| Context-free language | Languages parseable by pushdown automata                  |

**Structured generation** constrains an LLM to produce only strings from a target language $L$. At each decoding step, the valid next tokens form a set $V_{\text{valid}} \subseteq V$, computed by the automaton/parser. Tokens outside $V_{\text{valid}}$ are masked (logits set to $-\infty$). This is Boolean logic applied to decoding.

### 11.2 Sets in Probability

The entire framework of probability theory is built on set theory (Kolmogorov axioms):

| Probability concept                    | Set-theoretic object                         |
| -------------------------------------- | -------------------------------------------- |
| Sample space $\Omega$                  | Set of all possible outcomes                 |
| Event $A$                              | Subset of $\Omega$ ($A \subseteq \Omega$)    |
| $P(A \cup B)$                          | Probability of $A$ or $B$                    |
| $P(A \cap B)$                          | Probability of $A$ and $B$                   |
| $P(A^c)$                               | Probability of not $A$; equals $1 - P(A)$    |
| $\sigma$-algebra $\mathcal{F}$         | Collection of measurable subsets of $\Omega$ |
| Independence: $P(A \cap B) = P(A)P(B)$ | Multiplicativity under intersection          |

**Measure theory** (the rigorous foundation of probability) relies heavily on ZFC: $\sigma$-algebras are sets of sets, measures are set functions, and the Axiom of Choice is needed for results like the Vitali set (a non-measurable set).

### 11.3 Sets in Linear Algebra

| Linear algebra concept | Set structure                                                             |
| ---------------------- | ------------------------------------------------------------------------- |
| Vector space $V$       | Set with addition and scalar multiplication                               |
| Subspace $W \leq V$    | Subset that is itself a vector space                                      |
| Span                   | $\text{span}(S) = \{$ all linear combinations of $S\}$ — a set            |
| Basis                  | Linearly independent spanning set                                         |
| Eigenspace             | $E_\lambda = \{v \mid Av = \lambda v\}$ — defined by set-builder notation |
| Column space           | $\text{Col}(A) = \{Ax \mid x \in \mathbb{R}^n\}$ — image of $A$ (a set)   |
| Null space             | $\text{Null}(A) = \{x \mid Ax = 0\}$ — kernel of $A$ (a set)              |

### 11.4 Logic in Training

**Loss functions encode logical conditions:**

| Condition                   | Logic                                        | Loss                                                           |
| --------------------------- | -------------------------------------------- | -------------------------------------------------------------- |
| Output $= $ target          | Equality                                     | MSE: $\|f_\theta(x) - y\|^2$                                   |
| Output matches distribution | $P_\theta \approx P_{\text{data}}$           | KL divergence, cross-entropy                                   |
| Margin condition            | $y_i (\mathbf{w} \cdot \mathbf{x}_i) \geq 1$ | Hinge loss: $\max(0, 1 - y_i (\mathbf{w} \cdot \mathbf{x}_i))$ |
| Constraint satisfaction     | $g(\theta) \leq 0$                           | Lagrangian: $\mathcal{L} + \lambda g(\theta)$                  |

**Training as logic**: gradient descent searches for $\theta$ satisfying $\nabla \mathcal{L}(\theta) = 0$ — this is an existential claim: $\exists \theta^*\, \nabla \mathcal{L}(\theta^*) = 0$.

**Early stopping as logic**: "Stop when the validation loss has not decreased for $k$ epochs" is a quantified condition: $\forall t \in [T, T+k],\, \mathcal{L}_{\text{val}}^{(t)} \geq \mathcal{L}_{\text{val}}^{(T)}$.

### 11.5 Sets in Model Architecture

**Transformer attention as set operations:**

The key insight of attention is that it operates on a **set** of key-value pairs (modulo positional encoding). Self-attention is permutation-equivariant: it treats the input as a set, not a sequence.

- **Query:** "What am I looking for?"
- **Key set:** $K = \{k_1, k_2, \ldots, k_n\}$ — the set of keys to match against
- **Value set:** $V = \{v_1, v_2, \ldots, v_n\}$ — the set of values to aggregate
- **Attention mask:** $M \subseteq [n] \times [n]$ — the set of allowed (query, key) pairs

Combining masks:

- **Causal AND local:** $M_{\text{causal}} \cap M_{\text{local}}$ — intersection (Boolean AND)
- **Multi-head union:** $\bigcup_h M_h$ — each head sees a different subset of positions
- **Padding mask:** $M_{\text{pad}} = \{(i, j) \mid j \text{ is not a padding token}\}$

### 11.6 Formal Verification of ML Systems

Logic provides tools to **prove** properties of ML systems rather than merely test them:

| Property     | Formal statement                                                                | Method                                   |
| ------------ | ------------------------------------------------------------------------------- | ---------------------------------------- |
| Robustness   | $\forall x',\, \|x' - x\| < \varepsilon \to \text{class}(x') = \text{class}(x)$ | SMT solving, MILP                        |
| Fairness     | $P(\hat{y} \mid A=0) = P(\hat{y} \mid A=1)$                                     | Statistical testing + formal constraints |
| Safety       | $\forall \text{input},\, \text{output} \in \text{SafeSet}$                      | Reachability analysis                    |
| Monotonicity | $x_i \leq x_i' \to f(x) \leq f(x')$                                             | Lattice-based verification               |

**Certified defences** use logical proofs (often via abstract interpretation or interval arithmetic) to guarantee that no adversarial perturbation within an $\varepsilon$-ball can change the classification. This is the gold standard for ML safety — moving from empirical robustness to **provable** robustness.

---

## 12. Advanced Topics

### 12.1 Gödel's Incompleteness Theorems

The most profound results in mathematical logic, with deep implications for AI.

**First Incompleteness Theorem (Gödel, 1931):** Any consistent formal system $F$ capable of expressing basic arithmetic contains a statement $G$ such that:

- $G$ is true (in the standard model of arithmetic)
- $G$ is not provable in $F$
- $\neg G$ is not provable in $F$

In other words, **there are true mathematical statements that cannot be proved.** No axiom system — ZFC, or any extension — can be both consistent and complete for arithmetic.

**Gödel's self-referential trick:** Gödel constructs $G$ to say, essentially, "I am not provable in system $F$." If $G$ were provable, it would be false (since it says it's unprovable), contradicting consistency. If $\neg G$ were provable, $G$ would be provable (since $G$ says it's not), contradiction. So neither $G$ nor $\neg G$ is provable.

**Second Incompleteness Theorem:** No consistent system $F$ capable of expressing arithmetic can prove its own consistency.

$$F \text{ consistent} \implies F \not\vdash \text{Con}(F)$$

**AI implications:**

- No AI system (which is a formal system) can verify all true mathematical statements — there will always be truths beyond its reach
- The claim "this AI is guaranteed safe" cannot be proved within the AI's own framework (a version of the second theorem)
- Gödel's theorems do NOT mean mathematics is broken — they mean it is **inexhaustible**

### 12.2 The Halting Problem

**Theorem (Turing, 1936).** There is no algorithm that can determine, for every program $P$ and input $I$, whether $P$ halts on $I$.

**Proof (by contradiction).** Suppose $H(P, I)$ is a program that returns "halts" or "loops" for any $P, I$. Define:

$$D(P) = \begin{cases} \text{loop forever} & \text{if } H(P, P) = \text{"halts"} \\ \text{halt} & \text{if } H(P, P) = \text{"loops"} \end{cases}$$

What does $D(D)$ do?

- If $H(D, D) = $ "halts" → $D(D)$ loops forever. But $H$ said it halts. Contradiction.
- If $H(D, D) = $ "loops" → $D(D)$ halts. But $H$ said it loops. Contradiction.

So $H$ cannot exist. $\square$

**Connections to AI:**

- This is the computation analogue of Russell's Paradox — self-reference plus negation creates paradox
- You cannot write a "universal bug detector" that catches all infinite loops
- Theorem provers and code verifiers must be incomplete (they may time out or return "unknown")
- LLM code generation cannot guarantee termination of generated code

### 12.3 Zorn's Lemma

**Zorn's Lemma** (equivalent to the Axiom of Choice and the Well-Ordering Theorem):

> If every chain in a partially ordered set $(P, \leq)$ has an upper bound in $P$, then $P$ has a maximal element.

A **chain** is a totally ordered subset; a **maximal element** $m$ is one with no element strictly above it ($\nexists x \in P,\, x > m$).

**Applications in mathematics:**

- Every vector space has a basis (even infinite-dimensional ones)
- Every ideal in a ring is contained in a maximal ideal
- Every filter can be extended to an ultrafilter

**AI relevance:** Zorn's Lemma is used in the theoretical foundations of kernel methods (Hilbert space bases) and functional analysis (Hahn-Banach theorem), which underlies support vector machines and reproducing kernel Hilbert spaces.

### 12.4 Fuzzy Logic

Classical logic is crisp: $T$ or $F$. **Fuzzy logic** (Zadeh, 1965) allows truth values in $[0, 1]$:

$$\mu_A(x) \in [0, 1]$$

where $\mu_A(x)$ is the **degree of membership** of $x$ in fuzzy set $A$.

**Fuzzy operations:**
| Classical | Fuzzy |
|---|---|
| $A \cap B$ | $\mu_{A \cap B}(x) = \min(\mu_A(x), \mu_B(x))$ |
| $A \cup B$ | $\mu_{A \cup B}(x) = \max(\mu_A(x), \mu_B(x))$ |
| $\neg A$ | $\mu_{\neg A}(x) = 1 - \mu_A(x)$ |

**AI connections:**

- Softmax outputs are like fuzzy membership values — a token has degree-of-membership in each class
- Attention weights $\alpha_{ij} \in [0, 1]$ are fuzzy: not binary "attend or not" but "how much to attend"
- Fuzzy control systems (early AI) used fuzzy rules: "IF temperature IS high THEN fan IS fast"
- Modern neural networks have largely replaced fuzzy systems, but the conceptual framework persists in the soft/differentiable operations that make gradient descent possible

### 12.5 Second-Order Logic (Brief)

**First-order logic** quantifies over **objects**: $\forall x, \exists y, \ldots$

**Second-order logic** quantifies over **properties** (sets, predicates, functions): $\forall P, \exists f, \ldots$

**Example — Completeness of $\mathbb{R}$:**

- First-order: CANNOT express "every bounded set has a least upper bound" (needs quantification over sets)
- Second-order: $\forall S \subseteq \mathbb{R}\, [S \neq \emptyset \wedge \exists M\, \forall x \in S\, (x \leq M) \to \exists \sup S]$

**Trade-off:**

- First-order logic: semi-decidable (complete proof systems exist), but less expressive
- Second-order logic: more expressive, but no complete proof system exists (incompleteness is "worse")

In AI, first-order logic is preferred because it has better computational properties. Description logics, Datalog, and Prolog all work within FOL or its fragments.

---

## 13. Common Mistakes and Misconceptions

| #   | Mistake                                                       | Why it's wrong                                                                                  | Correct version                                                                                    |
| --- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 1   | Confusing $\in$ and $\subseteq$                               | "$3 \in \{1,2,3\}$" (membership) vs "$\{3\} \subseteq \{1,2,3\}$" (subset)                      | $a \in A$ = element belongs; $B \subseteq A$ = every element of $B$ is in $A$                      |
| 2   | $\{1,2\} \in \{1,2,3\}$                                       | The set $\{1,2\}$ is not an element of $\{1,2,3\}$; its elements are $1$, $2$, $3$              | $\{1,2\} \subseteq \{1,2,3\}$ ✓ but $\{1,2\} \in \{1,2,3\}$ ✗                                      |
| 3   | $\emptyset = \{\emptyset\}$                                   | $\emptyset$ has 0 elements; $\{\emptyset\}$ has 1 element (namely $\emptyset$)                  | $\emptyset \neq \{\emptyset\}$; $\vert \emptyset \vert = 0$, $\vert \{\emptyset\} \vert = 1$       |
| 4   | "Universal set exists"                                        | In ZFC, there is no set of all sets (leads to Russell's Paradox)                                | Work relative to a fixed domain $U$ (a set large enough for your purpose)                          |
| 5   | Confusing $\to$ and $\leftrightarrow$                         | $p \to q$ does NOT mean $q \to p$ (converse)                                                    | Implication is one-directional; biconditional is both                                              |
| 6   | "Vacuous truth is a bug"                                      | $F \to Q$ is always $T$ — this is a feature, not a bug                                          | Vacuous truth prevents edge-case contradictions; it's why $\emptyset \subseteq A$ for all $A$      |
| 7   | Converse = contrapositive                                     | $q \to p$ (converse) $\neq$ $\neg q \to \neg p$ (contrapositive)                                | Contrapositive $\equiv$ original; converse is NOT equivalent                                       |
| 8   | Swapping $\forall$ and $\exists$                              | $\forall x \exists y P(x,y) \neq \exists y \forall x P(x,y)$                                    | Quantifier order matters; only $\exists y \forall x \Rightarrow \forall x \exists y$ (not reverse) |
| 9   | "Countable = small"                                           | $\mathbb{Q}$ is countable but dense in $\mathbb{R}$; countable sets can be structurally complex | Countable means "no bigger than $\mathbb{N}$" — still infinite                                     |
| 10  | Induction hypothesis forgotten                                | Claiming $P(k+1)$ without using $P(k)$                                                          | The inductive step MUST use the hypothesis $P(k)$ (or $P(1), \ldots, P(k)$ for strong induction)   |
| 11  | Using induction without base case                             | Proving only the inductive step                                                                 | Without base case, the chain has no starting point; the "proof" proves nothing                     |
| 12  | Confusing "not proven" with "false"                           | Gödel: some true statements are unprovable                                                      | Unprovability $\neq$ falsity; independence results are about axiom limitations                     |
| 13  | $A - B = B - A$                                               | Set difference is NOT symmetric (unlike symmetric difference)                                   | $\{1,2,3\} - \{2,3,4\} = \{1\}$; $\{2,3,4\} - \{1,2,3\} = \{4\}$                                   |
| 14  | $\vert \mathcal{P}(A) \vert = \vert A \vert$ for infinite $A$ | Cantor's theorem: $\vert A \vert < \vert \mathcal{P}(A) \vert$ always, even for infinite sets   | There is no bijection between $A$ and $\mathcal{P}(A)$ — ever                                      |
| 15  | Treating attention as symmetric                               | $\text{Attends}(i,j)$ does NOT imply $\text{Attends}(j,i)$; attention is a directed relation    | Attention is asymmetric: query→key direction matters                                               |

---

## 14. Exercises

### Exercise 1: Set Operations (Warm-up)

Given $A = \{1, 2, 3, 4, 5\}$, $B = \{3, 4, 5, 6, 7\}$, $C = \{5, 6, 7, 8, 9\}$, compute:

1. $A \cup B$
2. $A \cap B \cap C$
3. $A - B$
4. $A \triangle B$ (symmetric difference)
5. $(A \cup B) \cap C$
6. $A \cap (B \cup C)$
7. Verify De Morgan's law: $(A \cup B)^c = A^c \cap B^c$ (with $U = \{1,\ldots,9\}$)
8. $\mathcal{P}(\{1,2,3\})$ — list all elements

### Exercise 2: Relations

Let $A = \{1, 2, 3, 4\}$ and $R = \{(1,1), (1,2), (2,1), (2,2), (3,3), (4,4), (3,4), (4,3)\}$.

1. Is $R$ reflexive? Symmetric? Transitive?
2. Is $R$ an equivalence relation? If yes, find the equivalence classes.
3. Draw the relation as a directed graph.
4. Write $R$ as a Boolean matrix.

### Exercise 3: Partial Orders

Consider the divisibility relation $|$ on the set $S = \{1, 2, 3, 4, 6, 12\}$ ($a \mid b$ iff $a$ divides $b$).

1. Verify that $|$ is a partial order on $S$ (check reflexivity, antisymmetry, transitivity).
2. Draw the Hasse diagram.
3. Find all maximal elements, minimal elements, greatest element, least element.
4. Is $(S, |)$ a total order? Why or why not?
5. Find all chains of maximum length.

### Exercise 4: Propositional Logic

1. Construct truth tables for:
   - $(p \to q) \wedge (q \to r) \to (p \to r)$ — is this a tautology?
   - $(p \wedge q) \to p$ — simplification law
   - $\neg(p \to q) \leftrightarrow (p \wedge \neg q)$
2. Prove or disprove: $(p \to q) \equiv (\neg q \to \neg p)$ (using truth tables).
3. Convert to CNF: $p \to (q \vee r)$.
4. Convert to DNF: $\neg(p \wedge q) \vee r$.

### Exercise 5: Predicate Logic

Translate to FOL (define your predicates):

1. "Every transformer has at least one attention head."
2. "No model with zero parameters can learn."
3. "There exists a learning rate that makes every batch converge."
4. "For every $\varepsilon > 0$, there exists $N$ such that for all $n > N$, $|a_n - L| < \varepsilon$."
5. Negate statement (3) formally and interpret in English.

### Exercise 6: Proof Practice

Prove each of the following using the specified technique:

1. **Direct proof:** If $n$ is odd, then $n^2$ is odd.
2. **Contrapositive:** If $n^2$ is divisible by 3, then $n$ is divisible by 3.
3. **Contradiction:** $\log_2 3$ is irrational.
4. **Induction:** For all $n \geq 1$: $\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$.
5. **Cases:** For all integers $n$: $n^3 - n$ is divisible by 6.

### Exercise 7: Cardinality

1. Prove that $|(0, 1)| = |\mathbb{R}|$ by constructing an explicit bijection.
2. Prove that $|\mathbb{N} \times \mathbb{N}| = |\mathbb{N}|$ using the Cantor pairing function.
3. Explain why the set of all Python programs is countable.
4. Explain why the set of all functions $\mathbb{N} \to \{0, 1\}$ is uncountable. (Hint: relate to $\mathcal{P}(\mathbb{N})$.)
5. A vocabulary has $|V| = 32{,}000$ tokens. How many distinct prompts of length exactly $n$ exist? Of length at most $n$?

### Exercise 8: AI Applications

1. **Attention mask logic:** A sliding window attention mask allows query $i$ to attend to key $j$ iff $|i - j| \leq w$. A causal mask allows $i$ to attend to $j$ iff $j \leq i$. Write the combined mask as a set: $M = \{(i, j) \mid \ldots\}$. Is $M$ an equivalence relation? A partial order?

2. **Tokeniser as equivalence:** Define an equivalence relation on $\{a, b, c\}^*$ where two strings are equivalent iff they map to the same token sequence under the BPE merges: $ab \to X$. List the equivalence classes (up to length 3).

3. **Loss convergence as FOL:** Write the formal statement "SGD converges to a critical point" using quantifiers. Then write its negation and interpret what it means for training to fail.

4. **Function composition:** A model has: embedding $E: V \to \mathbb{R}^{768}$, encoder $T: \mathbb{R}^{n \times 768} \to \mathbb{R}^{n \times 768}$ (12 layers), and head $H: \mathbb{R}^{768} \to \mathbb{R}^{|V|}$. Write the full model as a composition. Is the composition associative? Does the order matter?

---

## 15. Why This Matters: Conceptual Bridge

### 15.1 Summary of Connections

| Topic in This Chapter     | Where It Appears in AI/ML                                    | Chapter Reference |
| ------------------------- | ------------------------------------------------------------ | ----------------- |
| Sets, subsets, operations | Data partitioning (train/val/test), vocabularies, token sets | §2–3              |
| Relations                 | Attention masks, knowledge graphs, dependency parsing        | §4                |
| Equivalence relations     | Tokenisation, clustering, quotient spaces                    | §4.3              |
| Partial orders            | Computation graphs, dependency ordering, beam search         | §4.4              |
| Functions                 | Neural network layers, loss functions, activations           | §4.5              |
| Composition               | Deep networks as function chains, backpropagation            | §4.6              |
| Propositional logic       | Boolean masks, circuit design, SAT solving                   | §5                |
| Truth tables, CNF         | Constraint satisfaction, structured generation               | §5.2, §5.6        |
| Predicate logic (FOL)     | Formal specifications, knowledge representation              | §6                |
| Quantifiers               | Convergence definitions, generalization bounds               | §6.2–6.5          |
| Proof techniques          | Understanding theoretical papers, convergence proofs         | §7                |
| Induction                 | Proofs about recursive algorithms and sequence models        | §7.6              |
| ZFC axioms                | Why mathematics is consistent; foundational security         | §8                |
| Axiom of Choice           | Existence of bases, minima in non-compact spaces             | §8.3              |
| Cardinality               | Expressiveness limits, countability of programs              | §10               |
| Diagonal argument         | Impossibility results, undecidability                        | §10.4, §12.2      |
| Gödel's theorems          | Limits of formal verification and AI reasoning               | §12.1             |
| Formal verification       | Provable robustness, safety guarantees                       | §11.6             |

### 15.2 What to Study Next

Now that you have the foundations of sets and logic, the natural next steps are:

1. **Functions and Mappings (Chapter 03):** Deepens the treatment of functions from §4.5 — injectivity, surjectivity, inverse functions, and their role in defining neural network layers precisely.

2. **Linear Algebra (Chapters 02-03):** Sets and logic provide the language for defining vector spaces, subspaces, and linear maps rigorously. Every "for all" and "there exists" in a linear algebra textbook uses the predicate logic from §6.

3. **Calculus (Chapters 04-05):** The $\varepsilon$-$\delta$ definition of continuity ($\forall \varepsilon > 0, \exists \delta > 0, \forall x, |x - a| < \delta \to |f(x) - f(a)| < \varepsilon$) is a nested quantifier statement from §6.5. Optimisation theory builds on this.

4. **Probability Theory (Chapters 06-07):** Built entirely on measure theory, which is set theory on steroids. Events are sets, probability is a set function, and conditional probability uses set intersection.

5. **Information Theory (Chapter 09):** Cross-entropy and KL divergence are defined using sets (probability distributions), logarithms (built from real numbers, which are built from sets), and summations (which are disguised quantifiers).

### 15.3 The Big Picture

```
     ┌─────────────────────────────────────────────────────┐
     │                  THIS CHAPTER                        │
     │                                                      │
     │   Sets ──── Logic ──── Proofs ──── Axioms            │
     │    │          │          │            │               │
     │    ▼          ▼          ▼            ▼               │
     │  Data      Masks     Theory     Foundations          │
     │ structures  & rules   papers     of math             │
     │    │          │          │            │               │
     │    └──────────┴──────────┴────────────┘               │
     │                    │                                  │
     │                    ▼                                  │
     │          ┌─────────────────┐                         │
     │          │    EVERYTHING    │                         │
     │          │   IN MATH/CS    │                         │
     │          │   BUILDS HERE   │                         │
     │          └─────────────────┘                         │
     └─────────────────────────────────────────────────────┘
```

Sets and logic are the **lingua franca** of mathematics. Every mathematical definition, theorem, and proof uses the language developed in this chapter. When you encounter a statement like:

> "For every $\varepsilon > 0$, there exists a neural network $N$ with at most $n(\varepsilon)$ neurons such that $\|N - f\|_\infty < \varepsilon$"

You now know:

- **$\forall \varepsilon > 0$** — universal quantifier (§6.2)
- **$\exists N$** — existential quantifier (§6.2)
- **$\|N - f\|_\infty < \varepsilon$** — a predicate (§6.1)
- The whole statement is a nested $\forall\exists$ claim (§6.5)
- It can be proved by construction (§7.8) or contradiction (§7.4)
- The set of all such networks has a specific cardinality (§10)

**You can read mathematics now.** The rest of this course builds on this foundation.
