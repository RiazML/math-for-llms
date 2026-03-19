[← Previous: 05-Einstein-Summation-and-Index-Notation](../05-Einstein-Summation-and-Index-Notation/notes.md) | [Home](../../README.md)

---

# Proof Techniques

> _"A proof is a finite sequence of logical steps that establishes a truth beyond any possible doubt. Unlike experiments, which can always be overturned by a new observation, a mathematical proof is eternal — valid for all cases, for all time."_

## Overview

A proof technique is a general strategy for establishing that a mathematical statement is true — not probably true, not true in every case we checked, but **necessarily** true as a consequence of axioms, definitions, and previously established results. Proof techniques are the algorithms of mathematical reasoning: given a goal, which technique do you apply? Given a structure, which argument do you use?

For AI practitioners, proof techniques are not optional academic exercises. Every convergence theorem for gradient descent, every generalisation bound in PAC learning, every correctness argument for attention mechanisms, every hardness result in complexity theory — all rest on specific proof strategies. Without fluency in these strategies, theoretical machine learning papers are inaccessible, and the foundations of why models work (or fail) remain opaque.

This chapter covers the complete landscape of proof techniques, from elementary direct proofs through probabilistic existence arguments and analytic convergence proofs, with emphasis on the patterns that recur throughout ML theory.

## Prerequisites

- Propositional and predicate logic (from Sets and Logic module)
- Basic set operations (union, intersection, complement)
- Familiarity with functions, injectivity, surjectivity
- Elementary number theory concepts (even, odd, prime, divisibility)
- Basic probability (for Sections 10, 13)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations: proof verification, induction chains, probabilistic arguments |
| [exercises.ipynb](exercises.ipynb) | Practice problems: direct proofs, contradiction, induction, counting, ML proof patterns |

## Learning Objectives

After completing this section, you will:

- Identify and apply the correct proof technique for any mathematical statement
- Write rigorous direct proofs, proofs by contradiction, and proofs by contrapositive
- Master mathematical induction (ordinary, strong, and structural)
- Understand and apply the probabilistic method for existence proofs
- Use counting arguments, pigeonhole principle, and double counting
- Write ε-δ proofs for limits, continuity, and convergence
- Recognise and apply standard ML proof patterns: union bound, concentration inequalities, PAC learning, reduction proofs
- Identify common proof errors and evaluate the validity of AI-generated proofs
- Read and understand proofs in theoretical ML papers

---

## Table of Contents

- [Proof Techniques](#proof-techniques)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Are Proof Techniques?](#11-what-are-proof-techniques)
    - [1.2 Why Proof Techniques Matter for AI](#12-why-proof-techniques-matter-for-ai)
    - [1.3 The Landscape of Proof Techniques](#13-the-landscape-of-proof-techniques)
    - [1.4 The Structure of a Mathematical Statement](#14-the-structure-of-a-mathematical-statement)
    - [1.5 Reading and Writing Proofs](#15-reading-and-writing-proofs)
    - [1.6 Historical Timeline](#16-historical-timeline)
  - [2. Direct Proof](#2-direct-proof)
    - [2.1 The Strategy](#21-the-strategy)
    - [2.2 Template](#22-template)
    - [2.3 Worked Example — Even Times Even Is Even](#23-worked-example--even-times-even-is-even)
    - [2.4 Worked Example — If n² Is Even Then n Is Even](#24-worked-example--if-n²-is-even-then-n-is-even)
    - [2.5 Worked Example — Sum of Continuous Functions](#25-worked-example--sum-of-continuous-functions)
    - [2.6 Direct Proof in AI Contexts](#26-direct-proof-in-ai-contexts)
  - [3. Proof by Construction](#3-proof-by-construction)
    - [3.1 The Strategy](#31-the-strategy)
    - [3.2 Template](#32-template)
    - [3.3 Worked Example — Prime Between n and 2n](#33-worked-example--prime-between-n-and-2n)
    - [3.4 Worked Example — Constructing a Bijection](#34-worked-example--constructing-a-bijection)
    - [3.5 Worked Example — Neural Network Approximation](#35-worked-example--neural-network-approximation)
    - [3.6 Constructive vs Non-Constructive](#36-constructive-vs-non-constructive)
  - [4. Proof by Contrapositive](#4-proof-by-contrapositive)
    - [4.1 The Strategy](#41-the-strategy)
    - [4.2 Template](#42-template)
    - [4.3 Worked Example — If n² Is Even Then n Is Even](#43-worked-example--if-n²-is-even-then-n-is-even)
    - [4.4 Worked Example — Irrational Product](#44-worked-example--irrational-product)
    - [4.5 Contrapositive in AI Contexts](#45-contrapositive-in-ai-contexts)
    - [4.6 Recognising When to Use Contrapositive](#46-recognising-when-to-use-contrapositive)
  - [5. Proof by Contradiction](#5-proof-by-contradiction)
    - [5.1 The Strategy](#51-the-strategy)
    - [5.2 Template](#52-template)
    - [5.3 Worked Example — √2 Is Irrational](#53-worked-example--2-is-irrational)
    - [5.4 Worked Example — Infinitely Many Primes](#54-worked-example--infinitely-many-primes)
    - [5.5 Worked Example — No Largest Real Number](#55-worked-example--no-largest-real-number)
    - [5.6 Non-Constructive Existence via Contradiction](#56-non-constructive-existence-via-contradiction)
    - [5.7 Contradiction vs Contrapositive](#57-contradiction-vs-contrapositive)
    - [5.8 Contradiction in AI Contexts](#58-contradiction-in-ai-contexts)
  - [6. Proof by Cases](#6-proof-by-cases)
    - [6.1 The Strategy](#61-the-strategy)
    - [6.2 Template](#62-template)
    - [6.3 Worked Example — |xy| = |x||y|](#63-worked-example--xy--xy)
    - [6.4 Worked Example — ReLU Gradient](#64-worked-example--relu-gradient)
    - [6.5 Worked Example — Parity Argument](#65-worked-example--parity-argument)
    - [6.6 Proof by Exhaustion](#66-proof-by-exhaustion)
    - [6.7 Case Analysis in AI](#67-case-analysis-in-ai)
  - [7. Mathematical Induction](#7-mathematical-induction)
    - [7.1 The Strategy](#71-the-strategy)
    - [7.2 Why Induction Works](#72-why-induction-works)
    - [7.3 Template](#73-template)
    - [7.4 Worked Example — Sum Formula](#74-worked-example--sum-formula)
    - [7.5 Worked Example — Geometric Series](#75-worked-example--geometric-series)
    - [7.6 Worked Example — Power of 2 Bound](#76-worked-example--power-of-2-bound)
    - [7.7 Common Mistakes in Induction](#77-common-mistakes-in-induction)
  - [8. Strong Induction](#8-strong-induction)
    - [8.1 The Strategy](#81-the-strategy)
    - [8.2 Template](#82-template)
    - [8.3 Worked Example — Prime Factorisation](#83-worked-example--prime-factorisation)
    - [8.4 Worked Example — Fibonacci Bound](#84-worked-example--fibonacci-bound)
    - [8.5 Strong Induction in AI Analysis](#85-strong-induction-in-ai-analysis)
  - [9. Structural Induction](#9-structural-induction)
    - [9.1 The Strategy](#91-the-strategy)
    - [9.2 Common Structures](#92-common-structures)
    - [9.3 Worked Example — Tree Node Count](#93-worked-example--tree-node-count)
    - [9.4 Worked Example — Formula Length](#94-worked-example--formula-length)
    - [9.5 Structural Induction in AI](#95-structural-induction-in-ai)
  - [10. The Probabilistic Method](#10-the-probabilistic-method)
    - [10.1 The Strategy](#101-the-strategy)
    - [10.2 Template](#102-template)
    - [10.3 Worked Example — Tournament](#103-worked-example--tournament)
    - [10.4 Worked Example — Bipartite Subgraph](#104-worked-example--bipartite-subgraph)
    - [10.5 Lovász Local Lemma](#105-lovász-local-lemma)
    - [10.6 Expectation Argument](#106-expectation-argument)
    - [10.7 Probabilistic Method in AI](#107-probabilistic-method-in-ai)
  - [11. Counting Arguments](#11-counting-arguments)
    - [11.1 Double Counting](#111-double-counting)
    - [11.2 Worked Example — Sum of Degrees](#112-worked-example--sum-of-degrees)
    - [11.3 Worked Example — Vandermonde Identity](#113-worked-example--vandermonde-identity)
    - [11.4 Bijection Arguments](#114-bijection-arguments)
    - [11.5 Pigeonhole Principle](#115-pigeonhole-principle)
    - [11.6 Inclusion-Exclusion](#116-inclusion-exclusion)
  - [12. Epsilon-Delta and Analytic Arguments](#12-epsilon-delta-and-analytic-arguments)
    - [12.1 The ε-δ Framework](#121-the-ε-δ-framework)
    - [12.2 Template for Continuity Proof](#122-template-for-continuity-proof)
    - [12.3 Worked Example — Continuity of x²](#123-worked-example--continuity-of-x²)
    - [12.4 Sequence Convergence](#124-sequence-convergence)
    - [12.5 Gradient Descent Convergence](#125-gradient-descent-convergence)
    - [12.6 Compactness Arguments](#126-compactness-arguments)
    - [12.7 Fixed Point Theorems](#127-fixed-point-theorems)
  - [13. Proof Patterns in ML Theory](#13-proof-patterns-in-ml-theory)
    - [13.1 Union Bound](#131-union-bound)
    - [13.2 Concentration Inequalities](#132-concentration-inequalities)
    - [13.3 PAC Learning Proof Structure](#133-pac-learning-proof-structure)
    - [13.4 Rademacher Complexity](#134-rademacher-complexity)
    - [13.5 Information-Theoretic Proofs](#135-information-theoretic-proofs)
    - [13.6 Interchange of Limit and Sum](#136-interchange-of-limit-and-sum)
    - [13.7 Reduction Proofs](#137-reduction-proofs)
  - [14. Common Mistakes in Proofs](#14-common-mistakes-in-proofs)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI](#16-why-this-matters-for-ai)
  - [17. Conceptual Bridge](#17-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Are Proof Techniques?

A proof technique is a general strategy for establishing that a mathematical statement is true beyond any possible doubt. Unlike empirical evidence (which can always be overturned by a new counterexample) or intuition (which is frequently wrong), a mathematical proof is an **absolute guarantee** — valid for all cases, for all time.

Proof techniques are the **algorithms of mathematical reasoning**: given a goal, which technique do you apply? Given a structure, which argument do you use? Mastering proof techniques means mastering the ability to move from "I believe this is true" to "I can **prove** this is true and explain exactly why."

For AI: proofs establish correctness of algorithms, validity of bounds, convergence of optimisers, and soundness of theoretical guarantees. Without proof techniques, theoretical ML is inaccessible.

### 1.2 Why Proof Techniques Matter for AI

| AI Domain | Proof Technique Required | Example |
|-----------|-------------------------|---------|
| **Convergence proofs** | Telescoping, induction, ε-δ | Proving gradient descent converges to a stationary point |
| **Generalisation bounds** | Probabilistic method, union bound | PAC learning, VC dimension, Rademacher complexity |
| **Algorithm correctness** | Direct proof, induction | Proving BPE terminates; proving softmax is well-defined |
| **Complexity theory** | Reduction proofs, contradiction | Showing neural network verification is NP-hard |
| **LLM reasoning evaluation** | All techniques | Evaluating whether an LLM's "proof" is logically valid |
| **Formal verification** | All techniques | Lean, Coq, Isabelle verify AI system properties |

### 1.3 The Landscape of Proof Techniques

```
╔══════════════════════════════════════════════════════════════╗
║                    PROOF TECHNIQUES                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Direct Methods                                              ║
║  ├── Direct proof (assume P; derive Q)                       ║
║  ├── Proof by construction (exhibit the object)              ║
║  └── Proof by exhaustion (check all cases)                   ║
║                                                              ║
║  Indirect Methods                                            ║
║  ├── Contrapositive (prove ¬Q → ¬P instead of P → Q)        ║
║  ├── Contradiction (assume ¬P; derive ⊥)                    ║
║  └── Proof by cases (partition; prove each)                  ║
║                                                              ║
║  Inductive Methods                                           ║
║  ├── Mathematical induction (base + step)                    ║
║  ├── Strong induction (use all previous cases)               ║
║  └── Structural induction (induct on recursive structure)    ║
║                                                              ║
║  Probabilistic Methods                                       ║
║  ├── Probabilistic method (show Pr[property] > 0)            ║
║  ├── Expectation argument (E[X] implies existence)           ║
║  └── Lovász Local Lemma (avoid rare bad events)              ║
║                                                              ║
║  Algebraic Methods                                           ║
║  ├── Counting arguments (double counting; bijection)         ║
║  ├── Pigeonhole principle (n+1 items in n bins)              ║
║  └── Generating functions (encode combinatorics)             ║
║                                                              ║
║  Analytic Methods                                            ║
║  ├── ε-δ arguments (limits; continuity; convergence)         ║
║  ├── Compactness arguments (Heine-Borel; sequential)         ║
║  └── Fixed point arguments (Banach; Brouwer; Kakutani)       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

### 1.4 The Structure of a Mathematical Statement

Every theorem has the form: **"If [hypotheses], then [conclusion]."**

Formally: $H_1 \land H_2 \land \dots \land H_n \to C$

- **Hypotheses**: conditions assumed to hold; the "given" part
- **Conclusion**: what must be shown to follow; the "prove" part

A proof is a finite sequence of logically valid steps from hypotheses to conclusion. Each step follows from axioms, definitions, previously proved results, or hypotheses.

**Key skill**: identifying what you are allowed to **assume** and what you must **derive**.

```
╔═══════════════════════════════════════════════════╗
║          ANATOMY OF A THEOREM                     ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  "If  n is even,  then  n² is even"              ║
║       ─────────        ──────────                 ║
║       hypothesis       conclusion                 ║
║       (assume this)    (derive this)              ║
║                                                   ║
║  Proof = chain of valid steps:                    ║
║                                                   ║
║  Assume n is even                                 ║
║    → n = 2k for some integer k      (definition)  ║
║    → n² = (2k)² = 4k²              (algebra)     ║
║    → n² = 2(2k²)                   (factor)      ║
║    → n² is even                     (definition)  ║
║                                                   ║
║  Each arrow: justified by a named rule.           ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
```

### 1.5 Reading and Writing Proofs

**Reading a proof**: identify hypotheses → identify conclusion → trace each step → check each logical inference → verify completeness.

**Writing a proof**: state what you are proving → state your strategy → execute strategy → end with explicit conclusion.

Common proof structure markers:

| Marker | Purpose |
|--------|---------|
| "Assume…" / "Suppose…" | Introduce hypothesis or assumption |
| "Let…" | Introduce a variable or object |
| "Since…" / "Because…" / "By…" | Justify a step |
| "Therefore…" / "Thus…" / "Hence…" | Conclude a step |
| "We have shown…" / "This completes the proof" | Explicit closure |
| "QED" / "□" / "∎" | End of proof |
| "Case 1: … Case 2: …" | Proof by cases |
| "Base case: … Inductive step: …" | Induction |

### 1.6 Historical Timeline

| Date | Milestone | Significance |
|------|-----------|-------------|
| ~300 BCE | Euclid's *Elements* | First systematic proofs; 467 propositions from 5 axioms |
| ~250 BCE | Archimedes | Method of exhaustion; proto-integration |
| 9th–13th c. | Arab mathematicians | Algebraic proofs; al-Khwarizmi |
| 1821 | Cauchy | Rigorous ε-δ definitions of limits |
| 1860s | Weierstrass | Formal epsilon-delta proofs |
| 1874 | Cantor | Diagonal argument; ℝ is uncountable |
| 1900 | Hilbert | Programme to formalise all mathematics |
| 1901 | Russell | Paradox via self-reference |
| 1931 | Gödel | Incompleteness; limits of formal systems |
| 1976 | Appel & Haken | Four-colour theorem; computer-assisted proof |
| 1995 | Wiles | Fermat's Last Theorem; 129-page proof |
| 2000s– | Coq, Lean, Isabelle | Machine-verified proofs |
| 2024 | AlphaProof (DeepMind) | LLM + Lean; silver medal at IMO 2024 |

---

## 2. Direct Proof

### 2.1 The Strategy

To prove $P \to Q$:

1. **Assume** $P$ is true
2. Through a sequence of logical steps — each justified by definitions, axioms, or previously proved results — **derive** $Q$
3. **Conclude** $Q$ follows from $P$

This is the most natural proof technique. Try it first for any implication. It works best when the hypothesis gives you something concrete to work with and the conclusion has a clear definition to aim for.

### 2.2 Template

```
╔══════════════════════════════════════════════╗
║          DIRECT PROOF TEMPLATE               ║
╠══════════════════════════════════════════════╣
║                                              ║
║  Proof:                                      ║
║    Assume [P].                               ║
║    [Step 1: apply definition/theorem]        ║
║    [Step 2: algebraic or logical step]       ║
║    ...                                       ║
║    [Final step: arrive at Q]                 ║
║    Therefore [Q].  □                         ║
║                                              ║
╚══════════════════════════════════════════════╝
```

### 2.3 Worked Example — Even Times Even Is Even

**Theorem.** If $m$ and $n$ are both even integers, then $mn$ is even.

**Proof.**

Assume $m$ and $n$ are both even.

By definition of even: $\exists j \in \mathbb{Z}: m = 2j$ and $\exists k \in \mathbb{Z}: n = 2k$.

Then:

$$mn = (2j)(2k) = 4jk = 2(2jk)$$

Since $2jk \in \mathbb{Z}$, $mn$ is of the form $2 \times (\text{integer})$.

Therefore $mn$ is even. $\square$

**Dissection:** Every step names its justification — definition of even, integer closure under multiplication, definition of even again. No hand-waving.

### 2.4 Worked Example — If n² Is Even Then n Is Even (Direct Proof Fails)

**Theorem.** If $n^2$ is even, then $n$ is even.

**Attempted direct proof:** Assume $n^2$ is even. Then $n^2 = 2k$ for some $k$. So $n = \sqrt{2k}$... stuck. The square root of an even number is not obviously an integer, let alone obviously even.

**Lesson:** When direct proof stalls, switch technique. This theorem is proved cleanly by contrapositive (Section 4).

### 2.5 Worked Example — Sum of Continuous Functions Is Continuous

**Theorem.** If $f$ and $g$ are continuous at $a$, then $f + g$ is continuous at $a$.

**Proof.**

Assume $f$ and $g$ are continuous at $a$. Let $\varepsilon > 0$.

We need to find $\delta > 0$ such that $|x - a| < \delta$ implies $|(f+g)(x) - (f+g)(a)| < \varepsilon$.

By continuity of $f$: $\exists \delta_1 > 0$: $|x - a| < \delta_1 \implies |f(x) - f(a)| < \varepsilon/2$.

By continuity of $g$: $\exists \delta_2 > 0$: $|x - a| < \delta_2 \implies |g(x) - g(a)| < \varepsilon/2$.

Let $\delta = \min(\delta_1, \delta_2)$.

Then $|x - a| < \delta$ implies:

$$|(f+g)(x) - (f+g)(a)| = |f(x) + g(x) - f(a) - g(a)|$$
$$\leq |f(x) - f(a)| + |g(x) - g(a)| < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon$$

Therefore $f + g$ is continuous at $a$. $\square$

**Structure:** Uses both hypotheses; constructs $\delta$ from $\delta_1$ and $\delta_2$; the $\varepsilon/2$ trick ensures the sum stays below $\varepsilon$ after applying the triangle inequality.

### 2.6 Direct Proof in AI Contexts

**Theorem.** The output of a self-attention head lies in the convex hull of the value vectors.

**Proof.**

Let $\alpha_{ij}$ be the attention weights and $V_j$ the value vectors.

- $\alpha_{ij} \geq 0$ for all $j$ (softmax outputs are non-negative)
- $\sum_j \alpha_{ij} = 1$ (softmax outputs sum to 1)
- $O_i = \sum_j \alpha_{ij} V_j$ (output is a weighted sum)

By definition of convex hull: a point is in $\text{conv}(\{V_j\})$ if and only if it is a convex combination (non-negative weights summing to 1) of the $V_j$.

Therefore $O_i \in \text{conv}(\{V_j\})$. $\square$

**Theorem.** Cross-entropy loss is convex in logits.

**Proof.**

$L(z) = -z_y + \log \sum_v \exp(z_v)$

- $-z_y$ is linear in $z$, hence convex
- $\log \sum_v \exp(z_v)$ is the log-sum-exp function, which is convex (standard result: composition of affine functions with $\log \circ \text{sum} \circ \exp$)
- Sum of convex functions is convex

Therefore $L$ is convex. $\square$

---

## 3. Proof by Construction

### 3.1 The Strategy

To prove $\exists x\, P(x)$: **exhibit** a specific $x$ and **verify** that $P(x)$ holds.

A constructive proof provides an algorithm — the proof itself tells you **how to find** the object. This stands in contrast to non-constructive existence proofs (Section 5), which prove something exists without showing you how to find it.

Constructive proofs are strictly more informative: they tell you not just *that* something exists, but *what* it is.

### 3.2 Template

```
╔══════════════════════════════════════════════╗
║        CONSTRUCTIVE PROOF TEMPLATE           ║
╠══════════════════════════════════════════════╣
║                                              ║
║  Proof:                                      ║
║    [Define or describe object x explicitly]  ║
║    We claim x satisfies [P].                 ║
║    [Verify each required property of x]      ║
║    Therefore ∃x satisfying P.  □             ║
║                                              ║
╚══════════════════════════════════════════════╝
```

### 3.3 Worked Example — Prime Between n and 2n

**Theorem** (Bertrand's Postulate). For every $n \geq 1$, there exists a prime $p$ with $n < p \leq 2n$.

This is proved by careful counting (Chebyshev's and later Erdős's proof): one bounds the central binomial coefficient $\binom{2n}{n}$ from above and below, showing that a prime in $(n, 2n]$ must exist. The key is that the proof is constructive in principle — it establishes existence by showing the alternative (no prime in the range) leads to a bound violation.

### 3.4 Worked Example — Constructing a Bijection

**Theorem.** $|\mathbb{Z}| = |\mathbb{N}|$ (the integers and natural numbers have the same cardinality).

**Proof (by construction).** Exhibit an explicit bijection $f: \mathbb{N} \to \mathbb{Z}$:

$$f(n) = \begin{cases} 0 & \text{if } n = 0 \\ k & \text{if } n = 2k - 1 \text{ for } k \geq 1 \\ -k & \text{if } n = 2k \text{ for } k \geq 1 \end{cases}$$

This maps: $0 \mapsto 0,\; 1 \mapsto 1,\; 2 \mapsto -1,\; 3 \mapsto 2,\; 4 \mapsto -2,\; 5 \mapsto 3,\; 6 \mapsto -3, \dots$

**Verify injective:** Different natural numbers map to different integers (each integer appears exactly once in the sequence above).

**Verify surjective:** Every integer $z \in \mathbb{Z}$ is hit:
- $z = 0$: $f(0) = 0$ ✓
- $z > 0$: $f(2z - 1) = z$ ✓
- $z < 0$: $f(2|z|) = z$ ✓

Therefore $f$ is a bijection; $|\mathbb{Z}| = |\mathbb{N}|$. $\square$

### 3.5 Worked Example — Neural Network Approximation

**Theorem.** For the target function $f(x) = \mathbb{1}[x > 0.5]$, there exists a 1-hidden-layer ReLU network approximating $f$ to within $\varepsilon = 0.01$ on $[0, 1] \setminus (0.49, 0.51)$.

**Proof (by construction).**

Define the network:

$$h(x) = \frac{\text{ReLU}(Mx - M/2)}{M}$$

for a parameter $M > 0$. This is a single-neuron ReLU network with:
- $W_1 = [M]$, $b_1 = [-M/2]$, $W_2 = [1/M]$, $b_2 = 0$

**Behaviour:**
- For $x \leq 0.5$: $Mx - M/2 \leq 0$, so $\text{ReLU}(\cdot) = 0$, so $h(x) = 0 = f(x)$ ✓
- For $x > 0.5$: $h(x) = (Mx - M/2)/M = x - 0.5$
  - At $x = 0.51$: $h(0.51) = 0.01$; $f(0.51) = 1$; error $= 0.99$ (transition zone)
  - At $x = 0.5 + 1/M$: $h = 1/M$; for $M = 100$: $h(0.51) = 0.01$

For $M = 100$ and $x \geq 0.51$: $h(x) = x - 0.5 \geq 0.01$. We need $h(x) \approx 1$, so this simple construction works on $[0, 0.5] \cup \{1\}$ but needs adjustment for the full interval. A better construction uses two ReLU units to create a step:

$$g(x) = \text{ReLU}(M(x - 0.5)) - \text{ReLU}(M(x - 0.5) - 1)$$

This clips to $[0, 1]$ and approximates the step function with transition width $1/M$. For $M = 100$, the approximation error is $< 0.01$ outside $(0.49, 0.51)$. $\square$

### 3.6 Constructive vs Non-Constructive — A Key Distinction

| | Constructive | Non-Constructive |
|---|---|---|
| **What it gives** | The object itself | Proof that it exists |
| **Algorithm?** | Yes — the proof is the algorithm | No — existence without method |
| **Informativeness** | Very high | Moderate |
| **Typical technique** | Direct construction | Contradiction, probabilistic method |
| **AI relevance** | Algorithms, architectures | Performance bounds, impossibility results |

**Example — Normal Numbers:**
- Constructive: "Let $x = 0.123456789101112\dots$" (Champernowne's constant). This is provably normal in base 10 by construction.
- Non-constructive: "A normal number must exist because the set of non-normal numbers has Lebesgue measure zero." Proves existence without exhibiting the number.

**AI relevance:** Constructive proofs correspond to implementable algorithms. Non-constructive proofs establish performance bounds without telling you how to achieve them.

---

## 4. Proof by Contrapositive

### 4.1 The Strategy

**Logical equivalence:** $(P \to Q) \equiv (\lnot Q \to \lnot P)$

To prove $P \to Q$, you may equivalently prove $\lnot Q \to \lnot P$. This is the **contrapositive**.

When to use: when the negation of $Q$ gives more useful information than $P$ itself. Often, $\lnot Q$ hands you a concrete object to work with, while $P$ is too abstract.

**Critical distinction from contradiction:** Contrapositive proves a specific alternate implication $\lnot Q \to \lnot P$. Contradiction assumes $\lnot P$ and derives *any* false statement $\bot$. They are different techniques (see Section 5.7).

### 4.2 Template

```
╔══════════════════════════════════════════════════════╗
║         CONTRAPOSITIVE PROOF TEMPLATE                ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by contrapositive):                          ║
║    We prove the contrapositive: assume ¬Q.           ║
║    [Derive ¬P through logical steps]                 ║
║    Therefore ¬P.                                     ║
║    By contrapositive equivalence, P → Q.  □          ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### 4.3 Worked Example — If n² Is Even Then n Is Even

**Theorem.** For $n \in \mathbb{Z}$, if $n^2$ is even then $n$ is even.

**Contrapositive:** If $n$ is odd, then $n^2$ is odd.

**Proof (by contrapositive).**

Assume $n$ is odd. Then $\exists k \in \mathbb{Z}: n = 2k + 1$.

$$n^2 = (2k+1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1$$

Since $2k^2 + 2k \in \mathbb{Z}$, $n^2$ is of the form $2m + 1$; therefore $n^2$ is odd.

By contrapositive equivalence: if $n^2$ is even, then $n$ is even. $\square$

**Compare with Section 2.4:** The direct proof got stuck at $n = \sqrt{2k}$. The contrapositive transforms "even → even" (hard direction) into "odd → odd" (easy direction with concrete algebraic manipulation).

### 4.4 Worked Example — Irrational Product

**Theorem.** If $xy$ is rational and $x$ is irrational, then $y = 0$.

**Contrapositive:** If $y \neq 0$ and $x$ is irrational, then $xy$ is irrational.

**Proof (by contrapositive).**

Assume $y \neq 0$ and $x$ is irrational. Suppose for contradiction that $xy$ is rational: $xy = p/q$ with $q \neq 0$. Then $x = p/(qy)$. If $y$ is rational, $y = r/s$, then $x = ps/(qr)$, which is rational — contradicting $x$ irrational. So either $y$ is irrational (and we are in the case where $xy$ could still be irrational) or we get the contradiction.

Actually, this theorem is more naturally proved by contradiction — showing that technique selection requires judgment. The contrapositive reformulation doesn't simplify the argument. **Not every theorem is best proved by contrapositive.**

### 4.5 Contrapositive in AI Contexts

**Theorem.** If a neural network $f_\theta$ is not Lipschitz continuous, then it is not robust to adversarial examples.

**Contrapositive:** If $f_\theta$ is robust to adversarial examples (all perturbations $\|\delta\| < r$ change output by less than $\varepsilon$), then $f_\theta$ is Lipschitz with constant $L = \varepsilon / r$.

This direction is often easier to prove: robustness directly gives the Lipschitz bound.

**Theorem.** If $\nabla L(\theta) \neq 0$, then $\theta$ is not a local minimum.

**Contrapositive:** If $\theta$ is a local minimum, then $\nabla L(\theta) = 0$.

This is the standard necessary condition for optimality, and the contrapositive direction is the one typically proved: at a local minimum, any directional derivative must be non-negative, which forces $\nabla L = 0$.

### 4.6 Recognising When to Use Contrapositive

| Sign | Explanation |
|------|-------------|
| Conclusion $Q$ involves a **negation** | "x is not…", "cannot be…", "there is no…" |
| Negating $Q$ gives **concrete structure** | $\lnot Q$ hands you an object to work with |
| Direct proof **gets stuck** | Reformulating as $\lnot Q \to \lnot P$ opens new paths |
| Hypothesis $P$ has **existential** content | $\lnot P$ may be simpler than $P$ |

**Test:** Write out $\lnot P$ and $\lnot Q$. If $\lnot Q \to \lnot P$ seems more natural to argue, use contrapositive.

---

## 5. Proof by Contradiction

### 5.1 The Strategy

To prove $P$: assume $\lnot P$; derive a **contradiction** — a statement known to be false, or both a statement $C$ and its negation $\lnot C$ simultaneously.

Valid by the **law of excluded middle**: $P \lor \lnot P$. If $\lnot P$ leads to a contradiction $\bot$, then $P$ must hold.

When to use: when $P$ has the form "X does not exist" or "X is impossible"; when no positive construction is available; when the assumption $\lnot P$ has powerful consequences to exploit.

### 5.2 Template

```
╔══════════════════════════════════════════════════════╗
║         CONTRADICTION PROOF TEMPLATE                 ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by contradiction):                           ║
║    Assume for contradiction that ¬P.                 ║
║    [Derive consequences of ¬P]                       ║
║    [Arrive at statement C and ¬C simultaneously]     ║
║    This is a contradiction.                          ║
║    Therefore P must hold.  □                         ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### 5.3 Worked Example — √2 Is Irrational

**Theorem.** $\sqrt{2} \notin \mathbb{Q}$.

**Proof (by contradiction).**

Assume for contradiction that $\sqrt{2} \in \mathbb{Q}$.

Then $\sqrt{2} = p/q$ for some $p, q \in \mathbb{Z}$, $q \neq 0$, with $\gcd(p, q) = 1$ (fraction in lowest terms).

Squaring: $2 = p^2/q^2$, so $p^2 = 2q^2$.

$p^2$ is even. By the result of Section 4.3: $p$ is even. So $p = 2k$ for some $k \in \mathbb{Z}$.

Substituting: $(2k)^2 = 2q^2 \implies 4k^2 = 2q^2 \implies q^2 = 2k^2$.

$q^2$ is even. By the same result: $q$ is even.

But both $p$ and $q$ even contradicts $\gcd(p, q) = 1$.

**Contradiction.** Therefore $\sqrt{2} \notin \mathbb{Q}$. $\square$

**Note how the proof builds on Section 4.3:** The irrationality of $\sqrt{2}$ uses the lemma "$n^2$ even $\implies$ $n$ even" as a sub-result. This is typical — complex proofs assemble simpler proven facts.

### 5.4 Worked Example — Infinitely Many Primes (Euclid)

**Theorem.** There are infinitely many primes.

**Proof (by contradiction).**

Assume for contradiction there are only finitely many primes: $p_1, p_2, \dots, p_n$.

Construct $N = p_1 \times p_2 \times \dots \times p_n + 1$.

$N > 1$. Every integer greater than 1 has a prime factor. Let $p$ be a prime factor of $N$.

$p$ must be one of $p_1, \dots, p_n$ (by assumption, these are all the primes).

But $p \mid N$ and $p \mid p_1 p_2 \cdots p_n$, so $p \mid (N - p_1 p_2 \cdots p_n) = 1$.

No prime divides 1. **Contradiction.**

Therefore there are infinitely many primes. $\square$

### 5.5 Worked Example — No Largest Real Number

**Theorem.** There is no largest real number.

**Proof (by contradiction).**

Assume for contradiction $\exists M \in \mathbb{R}$: $M \geq x$ for all $x \in \mathbb{R}$.

Consider $M + 1 \in \mathbb{R}$.

$M + 1 > M$, contradicting $M$ being the largest.

**Contradiction.** Therefore no largest real number exists. $\square$

### 5.6 Non-Constructive Existence via Contradiction

One can prove "$\exists x\, P(x)$" by contradiction: assume $\forall x\, \lnot P(x)$; derive a contradiction. This proves existence **without constructing** the object.

**Example:** The Intermediate Value Theorem proves $\exists c \in (a, b): f(c) = y$ without constructing $c$.

**AI relevance:** Existence of optimal parameters $\theta^*$ minimising loss can be proved via compactness (if loss is continuous on a compact set) — but the proof gives no algorithm for finding $\theta^*$. This is why optimisation theory is a separate discipline from existence theory.

### 5.7 Contradiction vs Contrapositive — Clarifying the Difference

These two techniques are frequently confused. They are **not** the same:

| | Contrapositive | Contradiction |
|---|---|---|
| **Goal** | Prove $P \to Q$ | Prove $P$ (or $P \to Q$) |
| **Assume** | $\lnot Q$ | $\lnot P$ (or $P \land \lnot Q$) |
| **Derive** | $\lnot P$ (specifically) | Any contradiction $\bot$ |
| **Conclusion** | $P \to Q$ (by logical equivalence) | $P$ (since $\lnot P$ led to $\bot$) |
| **When best** | $\lnot Q \to \lnot P$ is natural | $\lnot P$ has powerful consequences |

Contrapositive is more structured (you know exactly what to derive: $\lnot P$). Contradiction is more flexible (any absurdity suffices), but can make proofs harder to follow.

### 5.8 Contradiction in AI Contexts

**Theorem.** Cross-entropy loss $L(z, y) = -z_y + \log \sum_v \exp(z_v)$ has no finite lower bound.

**Proof (by contradiction).**

Assume $\exists M$: $L(z, y) \geq M$ for all $z$.

Set $z_y = T$ and all other $z_v = 0$. Then:

$$L = -T + \log(\exp(T) + |V| - 1)$$

For large $T$: $\log(\exp(T) + |V| - 1) \approx T + \log(1 + (|V|-1)\exp(-T)) \to T$.

So $L \to -T + T + 0 = 0$ from above? Let's be more careful. Set all $z_v = 0$ for $v \neq y$ and vary $z_y$:

$$L = -z_y + \log(e^{z_y} + C) \quad \text{where } C = |V| - 1$$

As $z_y \to +\infty$: $L \to 0^+$ (approaches but never reaches 0).

So the infimum is 0, but it is not achieved. For any $M > 0$, choosing $T$ large enough makes $L < M$. The infimum 0 is not a minimum.

**More precisely:** The loss approaches 0 but never equals it. There is no $z$ achieving $L = 0$. The infimum is not attained. $\square$

**Theorem.** A single ReLU hidden unit cannot represent the XOR function.

**Proof (by contradiction).**

Assume a network with one ReLU hidden unit computes XOR. The hidden unit computes $h(x) = \text{ReLU}(w^T x + b)$, which is a piecewise linear function with at most 2 linear pieces (separated by the hyperplane $w^T x + b = 0$). The output is $f(x) = v \cdot h(x) + c$, also piecewise linear with at most 2 pieces.

XOR on $\{(0,0), (0,1), (1,0), (1,1)\}$ requires: $f(0,0) = 0$, $f(0,1) = 1$, $f(1,0) = 1$, $f(1,1) = 0$. But two linear pieces cannot separate these four points correctly — any half-plane places at least one positive and one negative example on the same side. **Contradiction.** $\square$

---

## 6. Proof by Cases (Exhaustion)

### 6.1 The Strategy

To prove $P$, partition the universe into exhaustive, mutually exclusive cases $C_1, C_2, \dots, C_n$ such that $C_1 \lor C_2 \lor \dots \lor C_n$ covers all possibilities. Prove $P$ holds in **each** case separately.

Valid because: $(C_1 \to P) \land (C_2 \to P) \land \dots \land (C_n \to P)$ together with $(C_1 \lor C_2 \lor \dots \lor C_n)$ yield $P$.

When to use: when no single argument covers all cases, but the problem has a natural partition (parity, sign, relative order, etc.).

### 6.2 Template

```
╔══════════════════════════════════════════════════════╗
║          PROOF BY CASES TEMPLATE                     ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by cases):                                   ║
║    Let x be arbitrary. We consider all cases.        ║
║                                                      ║
║    Case 1: [C₁ holds]                                ║
║      [Prove P under assumption C₁]                   ║
║                                                      ║
║    Case 2: [C₂ holds]                                ║
║      [Prove P under assumption C₂]                   ║
║                                                      ║
║    ⋮                                                 ║
║    Since C₁ ∨ C₂ ∨ … ∨ Cₙ is exhaustive, P.  □     ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### 6.3 Worked Example — Triangle Inequality for Absolute Value

**Theorem.** For all $a, b \in \mathbb{R}$: $|a + b| \leq |a| + |b|$.

**Proof (by cases).**

**Case 1:** $a + b \geq 0$.

$|a + b| = a + b \leq |a| + |b|$ since $a \leq |a|$ and $b \leq |b|$.

**Case 2:** $a + b < 0$.

$|a + b| = -(a + b) = (-a) + (-b) \leq |a| + |b|$ since $-a \leq |a|$ and $-b \leq |b|$.

Both cases yield $|a + b| \leq |a| + |b|$. $\square$

### 6.4 Worked Example — n² mod 4

**Theorem.** For all $n \in \mathbb{Z}$, $n^2 \bmod 4 \in \{0, 1\}$.

**Proof (by cases on $n \bmod 4$).**

**Case 1:** $n \equiv 0 \pmod{4}$. Then $n = 4k$, $n^2 = 16k^2 \equiv 0 \pmod{4}$.

**Case 2:** $n \equiv 1 \pmod{4}$. Then $n = 4k+1$, $n^2 = 16k^2 + 8k + 1 \equiv 1 \pmod{4}$.

**Case 3:** $n \equiv 2 \pmod{4}$. Then $n = 4k+2$, $n^2 = 16k^2 + 16k + 4 \equiv 0 \pmod{4}$.

**Case 4:** $n \equiv 3 \pmod{4}$. Then $n = 4k+3$, $n^2 = 16k^2 + 24k + 9 \equiv 1 \pmod{4}$.

All cases give $n^2 \bmod 4 \in \{0, 1\}$. $\square$

### 6.5 Cases in AI — Activation Function Analysis

**Theorem.** The ReLU function $\sigma(x) = \max(0, x)$ is convex.

**Proof (by cases).**

For convexity: $\sigma(\lambda x + (1-\lambda)y) \leq \lambda \sigma(x) + (1-\lambda) \sigma(y)$ for $\lambda \in [0,1]$.

Let $z = \lambda x + (1-\lambda) y$.

**Case 1:** $z \leq 0$.

$\sigma(z) = 0 \leq \lambda \sigma(x) + (1-\lambda)\sigma(y)$ since RHS is $\geq 0$.

**Case 2:** $z > 0$.

$\sigma(z) = z = \lambda x + (1-\lambda)y \leq \lambda \max(0,x) + (1-\lambda)\max(0,y) = \lambda \sigma(x) + (1-\lambda) \sigma(y)$

(using $x \leq \max(0,x)$ and $y \leq \max(0,y)$).

Both cases confirm convexity. $\square$

### 6.6 Without Loss of Generality (WLOG)

When cases are symmetric, you can reduce the number by stating "without loss of generality" (WLOG).

**Example:** Proving a property about $\max(a,b)$: WLOG assume $a \geq b$; then $\max(a,b) = a$. The case $b > a$ is symmetric with roles swapped.

**Caution:** WLOG is only valid when the cases are truly symmetric under relabelling. A common mistake is claiming WLOG when the problem is not symmetric.

---

## 7. Mathematical Induction

### 7.1 The Strategy

To prove $P(n)$ holds for all $n \geq n_0$ (typically $n_0 = 0$ or $1$):

1. **Base case:** Prove $P(n_0)$.
2. **Inductive step:** Prove $\forall k \geq n_0: P(k) \implies P(k+1)$.

Then $P(n)$ holds for all $n \geq n_0$ by the **well-ordering principle** (or equivalently, the **axiom of induction** for $\mathbb{N}$).

Induction is the foundational proof technique for discrete structures and is heavily used in analysing algorithms, data structures, and recursive models.

### 7.2 Template

```
╔══════════════════════════════════════════════════════╗
║         INDUCTION PROOF TEMPLATE                     ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by induction on n):                          ║
║                                                      ║
║    Base case (n = n₀):                               ║
║      [Verify P(n₀) directly]                         ║
║                                                      ║
║    Inductive step:                                   ║
║      Assume P(k) for some k ≥ n₀.  (IH)             ║
║      [Prove P(k+1) using the inductive              ║
║       hypothesis P(k)]                               ║
║                                                      ║
║    By induction, P(n) for all n ≥ n₀.  □            ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

**Important:** In the inductive step, you **assume** $P(k)$ (the "inductive hypothesis," IH) and prove $P(k+1)$. This is not circular — you are proving a conditional statement.

### 7.3 Worked Example — Sum Formula

**Theorem.** $\displaystyle\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ for all $n \geq 1$.

**Proof (by induction on $n$).**

**Base case ($n = 1$):**

LHS: $\sum_{i=1}^{1} i = 1$. RHS: $\frac{1 \cdot 2}{2} = 1$. ✓

**Inductive step:**

Assume $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$ for some $k \geq 1$. (IH)

Then:

$$\sum_{i=1}^{k+1} i = \left(\sum_{i=1}^{k} i\right) + (k+1) = \frac{k(k+1)}{2} + (k+1) \quad \text{(by IH)}$$

$$= \frac{k(k+1) + 2(k+1)}{2} = \frac{(k+1)(k+2)}{2}$$

This is $\frac{(k+1)((k+1)+1)}{2}$, which is $P(k+1)$. ✓

By induction, $P(n)$ holds for all $n \geq 1$. $\square$

### 7.4 Worked Example — Power Set Cardinality

**Theorem.** A set with $n$ elements has $2^n$ subsets.

**Proof (by induction on $n$).**

**Base case ($n = 0$):**

$\varnothing$ has one subset: $\varnothing$ itself. $2^0 = 1$. ✓

**Inductive step:**

Assume a set with $k$ elements has $2^k$ subsets. (IH)

Let $S$ be a set with $k + 1$ elements. Pick $x \in S$ and let $T = S \setminus \{x\}$, so $|T| = k$.

Every subset of $S$ either contains $x$ or does not:
- Subsets not containing $x$: exactly the subsets of $T$. There are $2^k$ of these (by IH).
- Subsets containing $x$: formed by taking a subset of $T$ and adding $x$. There are $2^k$ of these.

Total: $2^k + 2^k = 2 \cdot 2^k = 2^{k+1}$. ✓

By induction, $|S| = n \implies |\mathcal{P}(S)| = 2^n$ for all $n \geq 0$. $\square$

### 7.5 Induction for Neural Network Depth

**Theorem.** A ReLU network with $d$ hidden layers can represent a function with at most $2^d$ linear regions (given one hidden unit per layer).

**Proof (by induction on $d$).**

**Base case ($d = 1$):**

One ReLU hidden unit: $h(x) = \max(0, wx + b)$. This produces at most 2 linear pieces (one where input $\leq -b/w$ and one where $> -b/w$). $2^1 = 2$. ✓

**Inductive step:**

Assume a network with $k$ layers (one unit each) has $\leq 2^k$ linear regions. (IH)

Adding layer $k+1$ with one ReLU: the new ReLU unit can at most double the number of regions by "folding" each existing piece. The fold happens at the ReLU's breakpoint, splitting at most all existing regions in two.

Maximum new regions: $2 \cdot 2^k = 2^{k+1}$. ✓

By induction, the bound is $2^d$ for $d$ layers. $\square$

**Remark:** With $w$ units per layer, the bound becomes $O(w^d)$, explaining the exponential expressiveness of depth — a key insight behind deep learning.

### 7.6 Common Induction Mistakes

| Mistake | Example | Why It Fails |
|---------|---------|--------------|
| **Forgetting the base case** | "Assume $P(k)$, prove $P(k+1)$" — but never check $P(1)$ | The chain has no starting anchor |
| **Wrong base case** | Prove $P(0)$ but claim result for $n \geq 1$ | May miss that $P(1)$ needs separate attention |
| **Assuming what you prove** | "Assume $P(k+1)$" instead of proving it | Circular reasoning |
| **Not using the IH** | Proving $P(k+1)$ from scratch, never invoking $P(k)$ | Valid but not induction (just direct proof) |
| **Wrong inductive "direction"** | Assume $P(k)$, prove $P(k-1)$ | Proves for all $n \leq n_0$, not $n \geq n_0$ |

---

## 8. Strong Induction

### 8.1 The Strategy

Strong induction (also called **complete induction** or **course-of-values induction**) strengthens the inductive hypothesis: instead of assuming only $P(k)$, you assume $P(n_0) \land P(n_0 + 1) \land \dots \land P(k)$ — that is, $P(j)$ for **all** $n_0 \leq j \leq k$ — and then prove $P(k+1)$.

**Logical equivalence:** Strong induction is equivalent to ordinary induction on the statement "for all $j \leq n$, $P(j)$." It is not a stronger proof system, just a more convenient packaging.

When to use: when $P(k+1)$ depends on $P(k-1)$, $P(k-3)$, or some earlier case — not just the immediately preceding one.

### 8.2 Template

```
╔══════════════════════════════════════════════════════╗
║         STRONG INDUCTION TEMPLATE                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by strong induction on n):                   ║
║                                                      ║
║    Base case(s) (n = n₀, …, n₀+r):                  ║
║      [Verify P(n₀), …, P(n₀+r) directly]            ║
║                                                      ║
║    Inductive step:                                   ║
║      Assume P(j) for all n₀ ≤ j ≤ k.  (SIH)        ║
║      [Prove P(k+1) using any/all of                 ║
║       P(n₀), P(n₀+1), …, P(k)]                     ║
║                                                      ║
║    By strong induction, P(n) for all n ≥ n₀.  □     ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

**Note:** Strong induction often requires **multiple base cases** because the inductive step may reach back several positions.

### 8.3 Worked Example — Fundamental Theorem of Arithmetic

**Theorem.** Every integer $n \geq 2$ can be written as a product of primes.

**Proof (by strong induction on $n$).**

**Base case ($n = 2$):**

$2$ is prime, so it is (trivially) a product of primes. ✓

**Inductive step:**

Assume every integer $j$ with $2 \leq j \leq k$ can be written as a product of primes. (SIH)

Consider $k + 1$. Two cases:

**Case 1:** $k+1$ is prime. Then it is a product of one prime. ✓

**Case 2:** $k+1$ is composite. Then $k+1 = a \cdot b$ for some $2 \leq a, b \leq k$.

By the SIH, $a = p_1 \cdots p_r$ and $b = q_1 \cdots q_s$ are products of primes.

Therefore $k+1 = p_1 \cdots p_r \cdot q_1 \cdots q_s$, a product of primes. ✓

By strong induction, every $n \geq 2$ is a product of primes. $\square$

**Why standard induction fails:** $P(k+1)$ for composite $k+1$ requires $P(a)$ and $P(b)$ where $a, b < k+1$, but not necessarily $P(k)$ specifically.

### 8.4 Worked Example — Binary Representation

**Theorem.** Every positive integer has a binary representation: $n = \sum_{i} a_i 2^i$ with $a_i \in \{0, 1\}$.

**Proof (by strong induction on $n$).**

**Base case ($n = 1$):** $1 = 2^0$, so $a_0 = 1$. ✓

**Inductive step:**

Assume all $1 \leq j \leq k$ have binary representations. (SIH)

Consider $k + 1$.

**Case 1:** $k+1$ is even. Then $k+1 = 2m$ where $1 \leq m \leq k$. By SIH, $m = \sum_i a_i 2^i$. So $k+1 = \sum_i a_i 2^{i+1}$, a valid binary representation (shift all digits left one position). ✓

**Case 2:** $k+1$ is odd. Then $k+1 = 2m + 1$ where $1 \leq m \leq k$ (for $k \geq 1$). By SIH, $m = \sum_i a_i 2^i$. So $k+1 = 1 + \sum_i a_i 2^{i+1}$, setting $a_0 = 1$. ✓

By strong induction, every positive integer has a binary representation. $\square$

**AI relevance:** Binary representations are fundamental to digital computation. This proof legitimises the binary encoding that underlies all neural network implementations.

### 8.5 Strong Induction in AI — Recursive Network Correctness

**Theorem.** A tree-structured recursive neural network correctly computes the representation of any tree of depth $d \geq 0$ if (a) it correctly handles leaves and (b) whenever it correctly handles subtrees, its composition function correctly handles their parent.

**Proof (by strong induction on $d$).**

**Base case ($d = 0$):** Leaf nodes. By assumption (a), the network correctly computes their representations. ✓

**Inductive step:** Assume the network correctly represents all trees of depth $\leq k$. (SIH)

A tree of depth $k+1$ has a root whose children are roots of subtrees of depth $\leq k$. By SIH, these subtrees are correctly represented. By assumption (b), the composition function correctly computes the parent's representation from its children's representations. ✓

By strong induction, the network correctly handles all trees. $\square$

---

## 9. Structural Induction

### 9.1 The Strategy

Structural induction generalises mathematical induction from natural numbers to **inductively defined structures**: lists, trees, formulas, programs, and other recursive data types.

The idea: an inductively defined set has **base constructors** (creating atomic elements) and **recursive constructors** (building larger structures from smaller ones). Structural induction mirrors this:

1. **Base case:** Prove $P$ for each base constructor.
2. **Inductive step:** For each recursive constructor, assume $P$ for all sub-structures (structural inductive hypothesis, SIH); prove $P$ for the constructed structure.

### 9.2 Template

```
╔══════════════════════════════════════════════════════╗
║       STRUCTURAL INDUCTION TEMPLATE                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by structural induction):                    ║
║                                                      ║
║    Base case: [For each base constructor c₀]         ║
║      [Verify P(c₀)]                                 ║
║                                                      ║
║    Inductive step: [For each recursive               ║
║                     constructor C(x₁, …, xₘ)]       ║
║      Assume P(x₁), …, P(xₘ).  (SIH)                ║
║      [Prove P(C(x₁, …, xₘ))]                       ║
║                                                      ║
║    By structural induction, P holds for all          ║
║    elements of the inductively defined set.  □       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### 9.3 Induction on Lists

**Definition (Lists).** The set of lists over a type $A$ is inductively defined:
- **Base:** $[]$ (the empty list) is a list.
- **Recursive:** If $x \in A$ and $\ell$ is a list, then $x :: \ell$ (cons) is a list.

**Theorem.** $\text{length}(\text{append}(\ell_1, \ell_2)) = \text{length}(\ell_1) + \text{length}(\ell_2)$.

where $\text{append}([], \ell_2) = \ell_2$ and $\text{append}(x :: \ell_1', \ell_2) = x :: \text{append}(\ell_1', \ell_2)$.

**Proof (by structural induction on $\ell_1$).**

**Base case ($\ell_1 = []$):**

$\text{length}(\text{append}([], \ell_2)) = \text{length}(\ell_2) = 0 + \text{length}(\ell_2) = \text{length}([]) + \text{length}(\ell_2)$. ✓

**Inductive step ($\ell_1 = x :: \ell_1'$):**

Assume $\text{length}(\text{append}(\ell_1', \ell_2)) = \text{length}(\ell_1') + \text{length}(\ell_2)$. (SIH)

$$\text{length}(\text{append}(x :: \ell_1', \ell_2)) = \text{length}(x :: \text{append}(\ell_1', \ell_2))$$
$$= 1 + \text{length}(\text{append}(\ell_1', \ell_2)) = 1 + \text{length}(\ell_1') + \text{length}(\ell_2) \quad \text{(by SIH)}$$
$$= \text{length}(x :: \ell_1') + \text{length}(\ell_2)$$

✓ By structural induction, the result holds for all lists $\ell_1$. $\square$

### 9.4 Induction on Binary Trees

**Definition (Binary trees).**
- **Base:** $\text{Leaf}$ is a binary tree.
- **Recursive:** If $L$ and $R$ are binary trees, then $\text{Node}(L, R)$ is a binary tree.

**Theorem.** A binary tree with $n$ internal nodes has $n + 1$ leaves.

**Proof (by structural induction).**

**Base case ($\text{Leaf}$):**

0 internal nodes, 1 leaf. $0 + 1 = 1$. ✓

**Inductive step ($\text{Node}(L, R)$):**

Assume $L$ has $n_L$ internal nodes and $n_L + 1$ leaves (SIH). Similarly $R$ has $n_R$ internal nodes and $n_R + 1$ leaves (SIH).

$\text{Node}(L, R)$ has $n_L + n_R + 1$ internal nodes (the $+1$ is the new root).

Leaves: $(n_L + 1) + (n_R + 1) = n_L + n_R + 2 = (n_L + n_R + 1) + 1$. ✓

By structural induction, a tree with $n$ internal nodes has $n + 1$ leaves. $\square$

**AI relevance:** Decision trees, parse trees in NLP, and hierarchical attention structures are all binary/n-ary trees. Properties proved by structural induction apply directly.

### 9.5 Induction on Computation Graphs

**Definition (Computation graph for automatic differentiation).**
- **Base:** An input variable $x_i$ is a computation node.
- **Recursive:** If $v_1, \dots, v_k$ are computation nodes and $f$ is a differentiable function, then $v = f(v_1, \dots, v_k)$ is a computation node.

**Theorem (Correctness of backpropagation).** For any output node $v$ and any input $x_i$, the backpropagation algorithm correctly computes $\partial v / \partial x_i$.

**Proof sketch (by structural induction on computation nodes).**

**Base case:** $v = x_i$. Then $\partial x_i / \partial x_i = 1$ and $\partial x_i / \partial x_j = 0$ for $j \neq i$. Backpropagation initialises these correctly. ✓

**Inductive step:** $v = f(v_1, \dots, v_k)$.

Assume backprop correctly computes $\partial v_j / \partial x_i$ for all $j$ and all $i$. (SIH)

By the chain rule:

$$\frac{\partial v}{\partial x_i} = \sum_{j=1}^{k} \frac{\partial f}{\partial v_j} \cdot \frac{\partial v_j}{\partial x_i}$$

Backpropagation computes this sum: the local gradients $\partial f / \partial v_j$ are computed by the node, and $\partial v_j / \partial x_i$ are correctly computed by SIH. Their product and sum are correctly accumulated. ✓

By structural induction, backpropagation is correct for all computation graphs. $\square$

### 9.6 When to Use Which Induction Variant

| Variant | Domain | Inductive Hypothesis | Use When |
|---------|--------|---------------------|----------|
| **Standard** | $\mathbb{N}$ | $P(k)$ | $P(k+1)$ depends only on $P(k)$ |
| **Strong** | $\mathbb{N}$ | $P(n_0), \dots, P(k)$ | $P(k+1)$ depends on earlier cases |
| **Structural** | Inductive types | $P$ for all sub-structures | Domain is recursively defined |
| **Transfinite** | Ordinals | $P(\alpha)$ for all $\alpha < \beta$ | Infinite/uncountable well-ordered sets |

---

## 10. The Probabilistic Method

### 10.1 The Strategy

To prove that an object with property $P$ **exists**, define a probability distribution over a space of candidate objects. Show that the probability of $P$ is strictly positive: $\Pr[P] > 0$. Then an object with property $P$ must exist (otherwise the probability would be 0).

This technique, pioneered by **Paul Erdős**, is profoundly non-constructive: it proves existence without exhibiting a specific example. Yet it is one of the most powerful tools in combinatorics, computer science, and — increasingly — machine learning theory.

### 10.2 Template

```
╔══════════════════════════════════════════════════════╗
║       PROBABILISTIC METHOD TEMPLATE                  ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Proof (by the probabilistic method):                ║
║                                                      ║
║    Define a probability space Ω over candidates.     ║
║    Let X be a random candidate drawn from Ω.         ║
║    Compute E[cost(X)] or Pr[P(X)].                   ║
║                                                      ║
║    Show that E[cost(X)] < threshold                  ║
║      (or Pr[P(X)] > 0).                              ║
║                                                      ║
║    Therefore ∃ x ∈ Ω with cost(x) < threshold       ║
║      (or satisfying P).  □                           ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

**Key principle:** If a random variable has expected value $\mu$, then $\exists$ an outcome $\leq \mu$ (and $\exists$ an outcome $\geq \mu$). This is the **first moment method**.

### 10.3 Worked Example — Ramsey Lower Bound

**Theorem (Erdős, 1947).** $R(k, k) > 2^{k/2}$ (the diagonal Ramsey number exceeds $2^{k/2}$).

**Proof (by the probabilistic method).**

Consider the complete graph $K_n$ on $n$ vertices. Colour each edge red or blue independently with probability $1/2$.

For any set $S$ of $k$ vertices: $\Pr[\text{all } \binom{k}{2} \text{ edges in } S \text{ are red}] = 2^{-\binom{k}{2}}$.

By union bound over all $\binom{n}{k}$ choices of $S$:

$$\Pr[\exists \text{ monochromatic } K_k] \leq 2 \binom{n}{k} 2^{-\binom{k}{2}}$$

(factor 2 for red or blue).

Using $\binom{n}{k} \leq n^k / k!$:

$$\Pr[\exists \text{ monochromatic } K_k] \leq 2 \cdot \frac{n^k}{k!} \cdot 2^{-k(k-1)/2}$$

For $n = \lfloor 2^{k/2} \rfloor$: $n^k \approx 2^{k^2/2}$ and $2^{-k(k-1)/2} = 2^{-k^2/2 + k/2}$. So:

$$\Pr \leq \frac{2 \cdot 2^{k/2}}{k!} < 1 \quad \text{for } k \geq 3$$

Since probability $< 1$, there exists a 2-colouring with no monochromatic $K_k$. Therefore $R(k,k) > 2^{k/2}$. $\square$

### 10.4 Random Hyperplane Rounding (AI Application)

**Theorem (Goemans–Williamson, 1995).** The random hyperplane rounding algorithm achieves a 0.878-approximation for MAX-CUT.

**Proof sketch (probabilistic method).**

Solve the SDP relaxation: assign unit vectors $v_i \in \mathbb{R}^n$ to vertices. For each edge $(i,j)$, the relaxation value is $\frac{1}{2}(1 - v_i \cdot v_j)$.

**Rounding:** Choose a random hyperplane through the origin (uniformly random normal $r$). Assign vertex $i$ to side $+$ if $r \cdot v_i \geq 0$, else side $-$.

Edge $(i,j)$ is cut iff $\text{sign}(r \cdot v_i) \neq \text{sign}(r \cdot v_j)$.

$$\Pr[(i,j) \text{ is cut}] = \frac{\arccos(v_i \cdot v_j)}{\pi}$$

The key inequality:

$$\frac{\arccos(v_i \cdot v_j)}{\pi} \geq 0.878 \cdot \frac{1 - v_i \cdot v_j}{2}$$

Therefore: $\mathbb{E}[\text{cut value}] \geq 0.878 \cdot \text{SDP value} \geq 0.878 \cdot \text{OPT}$.

By linearity of expectation, there **exists** a hyperplane achieving $\geq 0.878 \cdot \text{OPT}$. $\square$

**AI connection:** Approximation algorithms via random rounding are conceptually related to random projection methods (Johnson–Lindenstrauss lemma, locality-sensitive hashing) used in approximate nearest neighbor search for embedding retrieval.

### 10.5 The Probabilistic Method in Deep Learning Theory

**Theorem (Random initialisation has non-zero gradients — informal).** For a randomly initialised deep network with ReLU activations, with probability 1 the gradient $\nabla_\theta L$ is non-zero at initialisation.

**Proof sketch.**

The set of parameter values where $\nabla L = 0$ exactly is a measure-zero subset of $\mathbb{R}^p$ (it is the zero set of a non-trivial analytic function of the parameters, assuming the loss is analytic). A continuous distribution over parameters assigns probability zero to any measure-zero set. Therefore, with probability 1, the initialisation lies outside this set. $\square$

This is why random initialisation "works": it almost surely places us at a point where gradient descent can make progress.

---

## 11. Counting and Combinatorial Arguments

### 11.1 The Strategy

Prove identities or existence results by **counting the same set in two different ways** (double counting), or by establishing **bijections** between sets, or by applying **combinatorial inequalities** (pigeonhole principle, inclusion-exclusion).

### 11.2 Double Counting

**Principle:** If you count the elements of a set $S$ in two different ways and get expressions $A$ and $B$, then $A = B$.

**Theorem (Handshaking Lemma).** In any graph $G = (V, E)$: $\sum_{v \in V} \deg(v) = 2|E|$.

**Proof (by double counting).**

Count the set $S = \{(v, e) : v \in V, e \in E, v \text{ is an endpoint of } e\}$.

Method 1: For each vertex $v$, it contributes $\deg(v)$ pairs. Total: $\sum_v \deg(v)$.

Method 2: For each edge $e = \{u, w\}$, it contributes exactly 2 pairs: $(u, e)$ and $(w, e)$. Total: $2|E|$.

Same set, two counts: $\sum_v \deg(v) = 2|E|$. $\square$

**AI relevance:** Graph neural networks (GNNs) aggregate messages along edges. The handshaking lemma ensures that the total number of message-passing operations equals $2|E|$, which determines the computational cost of one GNN layer.

### 11.3 The Pigeonhole Principle

**Principle:** If $n$ items are placed into $m$ containers and $n > m$, then some container holds more than one item.

**Generalised:** If $n$ items are placed into $m$ containers, some container holds $\geq \lceil n/m \rceil$ items.

**Theorem.** In any set of 13 real numbers, at least two have a difference whose fractional part $< 1/12$.

**Proof (by pigeonhole).**

Partition $[0, 1)$ into 12 intervals: $[0, 1/12), [1/12, 2/12), \dots, [11/12, 1)$.

Given 13 numbers, consider their fractional parts. By pigeonhole (13 numbers, 12 intervals), at least two fractional parts lie in the same interval. Their difference has fractional part $< 1/12$. $\square$

### 11.4 Pigeonhole in AI — Collision Arguments

**Theorem.** Any hash function $h: \{0,1\}^n \to \{0,1\}^m$ with $m < n$ has collisions.

**Proof (by pigeonhole).**

$|\text{domain}| = 2^n > 2^m = |\text{range}|$. By pigeonhole, $\exists x \neq y: h(x) = h(y)$. $\square$

**Corollary for embeddings:** An embedding $f: \mathcal{X} \to \mathbb{R}^d$ where $|\mathcal{X}| > c$ for any finite $c$ cannot be injective if restricted to a finite-precision representation with $< \log_2 |\mathcal{X}|$ bits per dimension.

**Theorem.** Any fixed-length tokeniser mapping variable-length strings to fixed-size vocabularies must have collisions (multiple strings mapping to the same token sequence of bounded length).

### 11.5 Inclusion-Exclusion

**Principle.**

$$\left|\bigcup_{i=1}^n A_i\right| = \sum_i |A_i| - \sum_{i<j} |A_i \cap A_j| + \sum_{i<j<k} |A_i \cap A_j \cap A_k| - \cdots + (-1)^{n+1} |A_1 \cap \cdots \cap A_n|$$

**AI application — Estimating vocabulary coverage:**

Given $n$ documents with vocabularies $V_1, \dots, V_n$:

$$|V_1 \cup \cdots \cup V_n| = \sum |V_i| - \sum |V_i \cap V_j| + \cdots$$

In practice, the higher-order terms are estimated by sampling. This is used in corpus analysis for NLP.

### 11.6 Bijective Proofs

**Principle:** To prove $|A| = |B|$, exhibit an explicit bijection $f: A \to B$.

**Theorem.** The number of subsets of $\{1, \dots, n\}$ of even size equals the number of odd size (for $n \geq 1$).

**Proof (by bijection).**

Fix element 1. Define $f(S) = S \triangle \{1\}$ (symmetric difference: add 1 if absent, remove if present).

$f$ maps even-sized subsets to odd-sized subsets and vice versa. $f$ is its own inverse ($f \circ f = \text{id}$), so it is a bijection. $\square$

Therefore: $\sum_{k \text{ even}} \binom{n}{k} = \sum_{k \text{ odd}} \binom{n}{k} = 2^{n-1}$.

---

## 12. Epsilon-Delta and Analytic Proofs

### 12.1 The ε-δ Framework

In analysis, many proofs require showing that a quantity can be made **arbitrarily small**. The standard framework:

$$\forall \varepsilon > 0,\ \exists \delta > 0: \text{[condition on } \delta\text{]} \implies \text{[conclusion within } \varepsilon\text{]}$$

The key challenge is **constructing** $\delta$ as a function of $\varepsilon$ (and sometimes other parameters).

### 12.2 Convergence of Gradient Descent (Fixed Step Size)

**Theorem.** Let $f: \mathbb{R}^d \to \mathbb{R}$ be convex and $L$-smooth ($\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$ for all $x, y$). Gradient descent with step size $\eta = 1/L$ satisfies:

$$f(x_T) - f(x^*) \leq \frac{L \|x_0 - x^*\|^2}{2T}$$

**Proof.**

By $L$-smoothness, for any $x, y$:

$$f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|^2$$

Set $y = x_{t+1} = x_t - \eta \nabla f(x_t)$ and $x = x_t$:

$$f(x_{t+1}) \leq f(x_t) + \nabla f(x_t)^T (-\eta \nabla f(x_t)) + \frac{L}{2} \eta^2 \|\nabla f(x_t)\|^2$$

$$= f(x_t) - \eta \|\nabla f(x_t)\|^2 + \frac{L \eta^2}{2} \|\nabla f(x_t)\|^2$$

$$= f(x_t) - \eta\left(1 - \frac{L\eta}{2}\right) \|\nabla f(x_t)\|^2$$

With $\eta = 1/L$: $f(x_{t+1}) \leq f(x_t) - \frac{1}{2L} \|\nabla f(x_t)\|^2$.

By convexity: $f(x^*) \geq f(x_t) + \nabla f(x_t)^T (x^* - x_t)$, so:

$$\|\nabla f(x_t)\|^2 \geq \frac{(\nabla f(x_t)^T (x_t - x^*))^2}{\|x_t - x^*\|^2}$$

Summing the per-step descent and using $f(x_t) - f(x^*) \leq \nabla f(x_t)^T (x_t - x^*)$ (convexity):

$$f(x_T) - f(x^*) \leq \frac{L \|x_0 - x^*\|^2}{2T}$$

This is an $O(1/T)$ convergence rate. $\square$

**Reading this proof:** Note the structure — establish a per-step improvement bound, then telescope the sum. This pattern recurs throughout optimisation theory.

### 12.3 PAC-Learning Bound (Finite Hypothesis Classes)

**Theorem.** Let $\mathcal{H}$ be a finite hypothesis class. For any distribution $\mathcal{D}$, any target $h^* \in \mathcal{H}$, and any $\varepsilon, \delta > 0$: if $m \geq \frac{1}{\varepsilon} \ln \frac{|\mathcal{H}|}{\delta}$, then with probability $\geq 1 - \delta$, any $h \in \mathcal{H}$ consistent with $m$ i.i.d. samples has error $\leq \varepsilon$.

**Proof.**

Fix a "bad" hypothesis $h$ with $\text{err}(h) > \varepsilon$. The probability it is consistent with one sample:

$$\Pr[h(x) = h^*(x)] \leq 1 - \varepsilon$$

Over $m$ i.i.d. samples:

$$\Pr[h \text{ consistent with all } m \text{ samples}] \leq (1 - \varepsilon)^m \leq e^{-\varepsilon m}$$

Union bound over all bad hypotheses ($\leq |\mathcal{H}|$ of them):

$$\Pr[\exists \text{ bad consistent } h] \leq |\mathcal{H}| \cdot e^{-\varepsilon m}$$

Set this $\leq \delta$: $|\mathcal{H}| e^{-\varepsilon m} \leq \delta \iff m \geq \frac{1}{\varepsilon} \ln \frac{|\mathcal{H}|}{\delta}$. $\square$

**Significance:** This is the foundational bound in computational learning theory. It connects sample complexity to hypothesis class size logarithmically — explaining why large model families need more data.

### 12.4 Uniform Convergence via ε-Nets

**Theorem (ε-net argument).** For a hypothesis class with VC-dimension $d$, $m = O\left(\frac{d + \ln(1/\delta)}{\varepsilon^2}\right)$ samples suffice for uniform convergence of empirical risk to true risk within $\varepsilon$.

**Proof idea.**

The ε-net approach discretises the infinite hypothesis class: find a finite ε-net $\mathcal{N}_\varepsilon$ (of size determined by the VC dimension) such that every $h \in \mathcal{H}$ is close to some $h' \in \mathcal{N}_\varepsilon$. Apply the finite class bound to $\mathcal{N}_\varepsilon$.

The key technical tool is the **symmetrisation argument** (Vapnik–Chervonenkis): replace the true risk with an empirical risk on a "ghost sample," then bound the probability of large deviations using Rademacher complexity or growth function bounds.

This is the theoretical foundation of **generalisation bounds** in machine learning.

---

## 13. Proof Patterns in Machine Learning

### 13.1 Overview

ML proofs recurrently use a small number of **structural patterns**. Recognising these patterns helps you read and construct proofs more efficiently.

```
╔══════════════════════════════════════════════════════╗
║       ML PROOF PATTERN LANDSCAPE                     ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Bound-Based    ──── Union bound + per-element       ║
║                       probability control             ║
║                                                      ║
║  Descent-Based  ──── Per-step improvement            ║
║                       + telescoping sum               ║
║                                                      ║
║  Convexity      ──── Jensen's inequality /            ║
║                       definition verification         ║
║                                                      ║
║  Symmetry       ──── Rademacher / permutation         ║
║                       arguments                       ║
║                                                      ║
║  Information    ──── Data processing inequality /     ║
║                       mutual information bounds       ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

### 13.2 The Union Bound Pattern

**Structure:** To bound $\Pr[\text{bad event}]$, decompose into sub-events and apply:

$$\Pr\left[\bigcup_i A_i\right] \leq \sum_i \Pr[A_i]$$

**Used in:** PAC-learning, uniform convergence, multi-task bounds, reward model error analysis.

**Pro tip:** If the union bound is too loose (sum exceeds 1), try:
- Chernoff/Hoeffding bounds for tighter individual terms
- ε-net arguments to reduce the number of terms
- Symmetrisation to avoid direct union bounding

### 13.3 The Descent Lemma Pattern

**Structure:**

1. Show $f(x_{t+1}) \leq f(x_t) - c \cdot g(x_t)$ for some positive quantity $g$.
2. Telescope: $\sum_{t=0}^{T-1} g(x_t) \leq \frac{f(x_0) - f(x_T)}{c} \leq \frac{f(x_0) - f^*}{c}$.
3. Conclude: $\min_t g(x_t) \leq \frac{f(x_0) - f^*}{cT}$.

**Used in:** Gradient descent convergence, SGD analysis, Adam convergence, policy gradient convergence.

### 13.4 The Convexity Verification Pattern

**Structure:** To prove $f$ is convex, verify **one** of:
1. **Definition:** $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$
2. **First-order:** $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ for all $x, y$
3. **Second-order:** $\nabla^2 f(x) \succeq 0$ for all $x$ (Hessian is PSD)

**Example — Cross-entropy is convex in logits:**

$$L(z) = -z_y + \log \sum_v e^{z_v}$$

$$\frac{\partial^2 L}{\partial z_i \partial z_j} = \text{diag}(p) - pp^T \quad \text{where } p_i = \frac{e^{z_i}}{\sum_v e^{z_v}}$$

This is $\text{diag}(p) - pp^T$, which is the covariance matrix of a categorical distribution — always PSD. $\square$

### 13.5 The Symmetrisation Argument

**Structure:** To bound $\sup_{h \in \mathcal{H}} |\hat{R}(h) - R(h)|$ (empirical vs true risk):

1. Introduce a "ghost" sample $S'$ of equal size.
2. Replace $R(h)$ with $\hat{R}_{S'}(h)$ (close with high probability by concentration).
3. $\sup_h |\hat{R}_S(h) - \hat{R}_{S'}(h)|$ is now a function of two finite samples.
4. Introduce Rademacher variables $\sigma_i \in \{-1, +1\}$: swapping $S$ and $S'$ entries is equivalent.
5. Bound the Rademacher complexity $\mathcal{R}_m(\mathcal{H})$.

This converts an infinite supremum into a computable quantity and is the basis of **Rademacher complexity** generalisation bounds.

### 13.6 Proof Strategy Selection Guide

| I want to show… | Technique | Key tool |
|---|---|---|
| Algorithm converges | Descent lemma | Per-step bound + telescope |
| Model generalises | Union bound / Rademacher | PAC / VC / Rademacher |
| Loss is well-behaved | Convexity verification | Hessian PSD check |
| Architecture has capacity | Structural induction | Induction on depth/width |
| Existence of good model | Probabilistic method | Random construction |
| Lower bound on complexity | Contradiction / counting | Information-theoretic |
| Two quantities are equal | Bijection / double counting | Combinatorial identity |
| Gradient computation is correct | Structural induction | Induction on computation graph |

---

## 14. Common Mistakes and Pitfalls

### 14.1 Affirming the Consequent

**Fallacy:** From $P \to Q$ and $Q$, conclude $P$.

**Example:** "If a model overfits, training loss is low. Training loss is low. Therefore the model overfits." — **Invalid.** Low training loss could be from a well-generalising model.

**Correct reasoning:** $P \to Q$ and $Q$ tells you nothing about $P$. Only $P \to Q$ and $\lnot Q$ lets you conclude $\lnot P$ (modus tollens).

### 14.2 Denying the Antecedent

**Fallacy:** From $P \to Q$ and $\lnot P$, conclude $\lnot Q$.

**Example:** "If learning rate is too high, training diverges. Learning rate is not too high. Therefore training does not diverge." — **Invalid.** Training can diverge for other reasons (exploding gradients, numerical issues).

### 14.3 Confusing Necessary and Sufficient Conditions

| | Necessary | Sufficient |
|---|---|---|
| **Form** | $Q \to P$ | $P \to Q$ |
| **Meaning** | $P$ is required for $Q$ | $P$ guarantees $Q$ |
| **Negation** | $\lnot P \to \lnot Q$ | $\lnot Q \to \lnot P$ |

**Example:** "Gradient = 0 is **necessary** for a minimum, but **not sufficient**." (Saddle points also have zero gradient.)

### 14.4 Proof by Example (Invalid)

Verifying $P(n)$ for specific values of $n$ does not prove $\forall n\, P(n)$.

**Famous counterexample:** $n^2 + n + 41$ is prime for $n = 0, 1, 2, \dots, 39$ — but $40^2 + 40 + 41 = 41^2$ is not prime.

**In ML:** "The model works on the test set" does not prove "the model works on all inputs." This is precisely why generalisation theory exists.

### 14.5 Circular Reasoning

Using the conclusion as a premise (possibly through a chain of implications that loops back).

**Example:** "Neural networks are universal approximators because they can approximate any function because they are universal approximators."

**Detection:** Trace the chain of reasoning. If any step assumes (even implicitly) what you are trying to prove, the argument is circular.

### 14.6 Misusing Induction

**"All horses are the same colour" fallacy:**

"Claim: Any group of $n$ horses are the same colour.

Base: $n = 1$. One horse has one colour. ✓

Inductive step: Assume any $k$ horses are the same colour. Given $k+1$ horses $\{h_1, \dots, h_{k+1}\}$:
- $\{h_1, \dots, h_k\}$: same colour by IH.
- $\{h_2, \dots, h_{k+1}\}$: same colour by IH.
- These overlap in $\{h_2, \dots, h_k\}$, so all $k+1$ are the same colour."

**The bug:** When $k = 1$, the overlap $\{h_2, \dots, h_k\} = \varnothing$ — there is no overlap! The inductive step fails at $k = 1$.

**Lesson:** Always check that the inductive step works at the **boundary** ($k = n_0$).

### 14.7 Quantifier Errors

| Written | Intended | Problem |
|---------|----------|---------|
| $\forall \varepsilon > 0\ \exists N:$ ... | For each $\varepsilon$, find $N$ | Correct (convergence) |
| $\exists N\ \forall \varepsilon > 0:$ ... | One $N$ works for all $\varepsilon$ | Wrong (much stronger) |

**In ML context:**

- "$\forall \varepsilon, \exists m$: with $m$ samples, error $< \varepsilon$" — correct PAC statement.
- "$\exists m, \forall \varepsilon$: with $m$ samples, error $< \varepsilon$" — would mean finite samples give zero error, which is false.

The **order of quantifiers matters**. Swapping $\forall$ and $\exists$ changes the meaning entirely.

---

## 15. Practice Exercises

### 15.1 Direct Proof Exercises

**Exercise 1.** Prove: the sum of two odd integers is even.

**Exercise 2.** Prove: if $a | b$ and $b | c$, then $a | c$ (divisibility is transitive).

**Exercise 3.** Prove: $\text{ReLU}(x) + \text{ReLU}(-x) = |x|$.

### 15.2 Contrapositive Exercises

**Exercise 4.** Prove by contrapositive: if $n^3$ is odd, then $n$ is odd.

**Exercise 5.** Prove by contrapositive: if $f$ is not Lipschitz, then $f'$ is unbounded (for differentiable $f$ on $\mathbb{R}$).

### 15.3 Contradiction Exercises

**Exercise 6.** Prove by contradiction: $\sqrt{3}$ is irrational.

**Exercise 7.** Prove by contradiction: there is no smallest positive rational number.

**Exercise 8.** Prove: $\log_2 3$ is irrational.

### 15.4 Induction Exercises

**Exercise 9.** Prove by induction: $\sum_{i=0}^{n} 2^i = 2^{n+1} - 1$.

**Exercise 10.** Prove by induction: a binary tree of height $h$ has at most $2^{h+1} - 1$ nodes.

**Exercise 11.** Prove by strong induction: every integer $n \geq 2$ can be expressed as a sum of distinct powers of 2 (binary representation with no repeats).

### 15.5 Cases and Construction Exercises

**Exercise 12.** Prove by cases: $\max(a, b) = \frac{a + b + |a - b|}{2}$.

**Exercise 13.** Construct a bijection between $\{0, 1\}^n$ (binary strings of length $n$) and $\mathcal{P}(\{1, \dots, n\})$ (subsets of $\{1, \dots, n\}$).

### 15.6 Counting and Probabilistic Exercises

**Exercise 14.** Use the pigeonhole principle to show: in any group of 5 integers, some pair has the same remainder mod 4.

**Exercise 15.** Use the handshaking lemma to prove: in any graph, the number of vertices with odd degree is even.

### 15.7 ML-Specific Exercises

**Exercise 16.** Prove that softmax output lies in the probability simplex: $\sum_i \text{softmax}(z)_i = 1$ and $\text{softmax}(z)_i > 0$ for all $i$.

**Exercise 17.** Prove by induction on network depth: the composition of $d$ Lipschitz functions with constants $L_1, \dots, L_d$ is Lipschitz with constant $\prod_{i=1}^d L_i$.

**Exercise 18.** Prove by contradiction: a continuous function on a closed bounded interval $[a, b]$ attains its maximum (use sequences and the Bolzano–Weierstrass theorem).

### 15.8 Solution Sketches

**Exercise 1 solution sketch:**

Let $a = 2k + 1$, $b = 2m + 1$. Then $a + b = 2k + 2m + 2 = 2(k + m + 1)$. Since $k + m + 1 \in \mathbb{Z}$, the sum is even. $\square$

**Exercise 6 solution sketch:**

Assume $\sqrt{3} = p/q$ with $\gcd(p,q) = 1$. Then $3q^2 = p^2$, so $3 | p^2$, hence $3 | p$ (since 3 is prime). Write $p = 3k$: $3q^2 = 9k^2 \implies q^2 = 3k^2$, so $3 | q$. But $3 | p$ and $3 | q$ contradicts $\gcd(p,q) = 1$. $\square$

**Exercise 9 solution sketch:**

Base ($n = 0$): $2^0 = 1 = 2^1 - 1$. ✓

IH: Assume $\sum_{i=0}^{k} 2^i = 2^{k+1} - 1$.

$\sum_{i=0}^{k+1} 2^i = (2^{k+1} - 1) + 2^{k+1} = 2 \cdot 2^{k+1} - 1 = 2^{k+2} - 1$. ✓ $\square$

**Exercise 16 solution sketch:**

Positivity: $e^{z_i} > 0$ for all $z_i \in \mathbb{R}$, and $\sum_v e^{z_v} > 0$, so $\text{softmax}(z)_i = e^{z_i} / \sum_v e^{z_v} > 0$.

Sum to 1: $\sum_i \text{softmax}(z)_i = \sum_i e^{z_i} / \sum_v e^{z_v} = (\sum_i e^{z_i}) / (\sum_v e^{z_v}) = 1$. $\square$

---

## 16. Why It Matters for AI

### 16.1 Proofs as Guarantees

Machine learning operates in a landscape of empirical results. Proofs provide the **theoretical guardrails**:

| What proofs guarantee | Example |
|---|---|
| **Correctness** | Backpropagation correctly computes gradients (structural induction, §9.5) |
| **Convergence** | Gradient descent reaches optimum at rate $O(1/T)$ (analytic proof, §12.2) |
| **Generalisation** | PAC bounds on sample complexity (union bound + counting, §12.3) |
| **Expressiveness** | Universal approximation — networks can represent any continuous function |
| **Impossibility** | No Free Lunch — no single algorithm dominates on all problems |
| **Robustness** | Lipschitz bounds ↔ adversarial robustness (contrapositive, §4.5) |

Without proofs, we are running experiments in the dark. With proofs, we know **why** things work (or fail) and **when** guarantees hold.

### 16.2 Proofs and Algorithms

There is a deep connection between proofs and algorithms (the **Curry–Howard correspondence** in its broadest sense):

| Proof concept | Algorithm concept |
|---|---|
| Constructive existence proof | Algorithm that finds the object |
| Inductive proof | Recursive algorithm |
| Case analysis | Branching / conditional logic |
| Contradiction | Exception handling / invariant violation |
| Bound derivation | Complexity analysis |

When you prove a convergence rate of $O(1/T)$, you are simultaneously describing the algorithm (gradient descent) and its guarantee. When you prove existence by construction, you are writing a program.

### 16.3 Proof Literacy for ML Practitioners

You don't need to prove new theorems to benefit from proof literacy. Knowing how to **read** proofs lets you:

1. **Understand assumptions:** Every theorem has hypotheses. Knowing them tells you when the guarantee applies to your model and when it doesn't.

2. **Spot gaps in arguments:** "This architecture is better because it has more parameters" — is that a proof? What's the implicit theorem? What are the hidden assumptions?

3. **Debug theoretical claims:** Papers sometimes contain proof errors. Proof literacy lets you verify claims before building on them.

4. **Transfer results:** If you understand *why* a theorem holds (not just *that* it holds), you can adapt it to new settings.

### 16.4 Modern Frontiers — AI for Proofs

The field is coming full circle: AI systems are now being used to **discover and verify proofs**:

| System | Achievement |
|---|---|
| **AlphaProof** (DeepMind, 2024) | Solved IMO-level problems via reinforcement learning + formal verification |
| **Lean / Mathlib** | Formal proof library with growing ML-generated contributions |
| **LLEMMA** (2023) | Language model fine-tuned for mathematical reasoning |
| **Autoformalization** | Translating informal proofs to formal (verifiable) proofs via LLMs |
| **ATP systems** | Automated theorem provers (Vampire, E, Z3) integrated with ML heuristics |

The ability to formalise and verify proofs is becoming a key AI capability, and understanding proof techniques is essential for building and evaluating these systems.

---

## 17. Conceptual Bridge

### 17.1 From Proof Strategy to ML Reasoning

This chapter sits underneath the rest of the curriculum in a quiet but important way. Later topics will look like they are about vectors, derivatives, optimization, probability, or model architecture, but the moment a theorem appears, the same proof patterns return:

- direct proof for algebraic identities and gradient derivations
- contradiction for impossibility results and lower bounds
- induction for sequence length, layer depth, and recursive structure
- construction for algorithms, witnesses, and counterexamples

Proof techniques are therefore not separate from applied ML mathematics. They are the reasoning templates that make later chapters readable.

### 17.2 What Comes Next

As you move into linear algebra, calculus, probability, and optimization, try to read every result with three questions in mind: what is assumed, what is being concluded, and why this proof strategy fits the shape of the claim. That habit turns proof literacy into practical engineering judgment, because it helps you distinguish empirical patterns from actual guarantees.

---

## References

1. **Velleman, D.J.** *How to Prove It: A Structured Approach.* Cambridge University Press, 3rd ed., 2019.
2. **Hammack, R.** *Book of Proof.* Virginia Commonwealth University, 3rd ed., 2018. (Free online)
3. **Alon, N. & Spencer, J.H.** *The Probabilistic Method.* Wiley, 4th ed., 2016.
4. **Shalev-Shwartz, S. & Ben-David, S.** *Understanding Machine Learning: From Theory to Algorithms.* Cambridge University Press, 2014.
5. **Boyd, S. & Vandenberghe, L.** *Convex Optimization.* Cambridge University Press, 2004.
6. **Goemans, M.X. & Williamson, D.P.** "Improved approximation algorithms for maximum cut…" *JACM* 42(6), 1995.
7. **Erdős, P.** "Some remarks on the theory of graphs." *Bulletin of the AMS* 53, 1947.
8. **Nesterov, Y.** *Introductory Lectures on Convex Optimization.* Springer, 2004.
9. **Trinh, T. et al.** "Solving olympiad geometry without human demonstrations." *Nature* 625, 2024.

---

[← Previous: Einstein Summation and Index Notation](../05-Einstein-Summation-and-Index-Notation/notes.md) | [Home](../../README.md) | [Next: Vectors and Spaces →](../../02-Linear-Algebra-Basics/01-Vectors-and-Spaces/notes.md)
