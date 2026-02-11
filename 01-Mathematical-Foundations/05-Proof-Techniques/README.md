# Proof Techniques

## Introduction

Mathematical proofs are the foundation of rigorous reasoning in ML theory. Understanding proof techniques helps you verify algorithm correctness, understand convergence guarantees, and read research papers. This module covers the essential proof methods used in ML literature.

## Prerequisites

- Basic logic and set theory
- Functions and their properties
- Mathematical notation

## Learning Objectives

1. Understand and apply direct proofs
2. Use proof by contradiction
3. Master mathematical induction
4. Apply proof by contrapositive
5. Recognize when to use each technique

---

## 1. Direct Proof

### Concept

A **direct proof** shows that if P is true, then Q must be true, by a chain of logical steps.

```
Direct Proof Structure:

P (Hypothesis) ─────▶ Step 1 ─────▶ Step 2 ─────▶ ... ─────▶ Q (Conclusion)

Each step follows logically from previous statements.
```

### Template

1. Assume the hypothesis P is true
2. Use definitions, known theorems, and logical steps
3. Arrive at the conclusion Q

### Example 1: Sum of Two Even Numbers

**Theorem**: The sum of two even numbers is even.

**Proof**:

1. Let a and b be even numbers (hypothesis)
2. By definition of even, a = 2m and b = 2n for some integers m, n
3. Then a + b = 2m + 2n = 2(m + n)
4. Since (m + n) is an integer, a + b = 2k where k = m + n
5. By definition, a + b is even ∎

### Example 2: Product of Negative Numbers

**Theorem**: The product of two negative numbers is positive.

**Proof**:

1. Let a < 0 and b < 0
2. Then a = -|a| and b = -|b| where |a|, |b| > 0
3. a · b = (-|a|)(-|b|) = |a| · |b|
4. Since |a| > 0 and |b| > 0, their product |a| · |b| > 0
5. Therefore a · b > 0 ∎

---

## 2. Proof by Contradiction

### Concept

Assume the statement is false, then derive a contradiction. This proves the original statement must be true.

```
Proof by Contradiction:

Assume ¬Q (Q is false)
    ↓
Logical steps
    ↓
Arrive at contradiction (P ∧ ¬P)
    ↓
Therefore Q must be true
```

### Template

1. Assume the negation of what you want to prove
2. Use logical reasoning
3. Derive a contradiction (something impossible)
4. Conclude the original statement is true

### Example 1: √2 is Irrational

**Theorem**: √2 is irrational.

**Proof**:

1. Assume √2 is rational (negation)
2. Then √2 = p/q where p, q are integers with no common factors
3. So 2 = p²/q², which means p² = 2q²
4. Therefore p² is even, so p is even
5. Let p = 2k, then 4k² = 2q², so q² = 2k²
6. Therefore q² is even, so q is even
7. But both p and q being even contradicts having no common factors
8. **Contradiction!** Therefore √2 is irrational ∎

### Example 2: Infinitely Many Primes

**Theorem**: There are infinitely many prime numbers.

**Proof** (Euclid):

1. Assume there are finitely many primes: p₁, p₂, ..., pₙ
2. Consider N = p₁ · p₂ · ... · pₙ + 1
3. N is not divisible by any pᵢ (remainder is always 1)
4. Either N is prime, or N has a prime factor not in our list
5. Either way, there exists a prime not in {p₁, ..., pₙ}
6. **Contradiction!** Therefore infinitely many primes exist ∎

---

## 3. Proof by Contrapositive

### Concept

To prove "P ⟹ Q", prove the equivalent "¬Q ⟹ ¬P".

```
Logical Equivalence:

P ⟹ Q  ≡  ¬Q ⟹ ¬P

"If it's raining, the ground is wet"
≡
"If the ground is not wet, it's not raining"
```

### When to Use

Use contrapositive when:

- The conclusion Q is easier to negate
- Working forward from ¬Q is clearer
- Direct proof seems stuck

### Example 1: Squares and Evenness

**Theorem**: If n² is odd, then n is odd.

**Contrapositive**: If n is even, then n² is even.

**Proof of Contrapositive**:

1. Assume n is even
2. Then n = 2k for some integer k
3. n² = (2k)² = 4k² = 2(2k²)
4. Since 2k² is an integer, n² is even ∎

### Example 2: Divisibility

**Theorem**: If n² is divisible by 3, then n is divisible by 3.

**Contrapositive**: If n is not divisible by 3, then n² is not divisible by 3.

**Proof of Contrapositive**:

1. Assume n is not divisible by 3
2. Then n = 3k + r where r ∈ {1, 2}
3. If r = 1: n² = 9k² + 6k + 1 = 3(3k² + 2k) + 1 (remainder 1)
4. If r = 2: n² = 9k² + 12k + 4 = 3(3k² + 4k + 1) + 1 (remainder 1)
5. In both cases, n² has remainder 1 when divided by 3
6. Therefore n² is not divisible by 3 ∎

---

## 4. Mathematical Induction

### Concept

Prove a statement P(n) holds for all natural numbers n ≥ n₀.

```
Mathematical Induction:

┌────────────────┐     ┌────────────────┐
│  Base Case     │     │  Inductive     │
│  P(n₀) is true │ ──▶ │  Step: P(k) ⟹  │
└────────────────┘     │  P(k+1)        │
                       └────────────────┘
                              │
                              ▼
                       P(n) true for all n ≥ n₀

Like dominoes: knock first one, each knocks the next
```

### Template

1. **Base case**: Prove P(n₀) is true
2. **Inductive hypothesis**: Assume P(k) is true for some k ≥ n₀
3. **Inductive step**: Prove P(k+1) using the hypothesis
4. **Conclusion**: By induction, P(n) holds for all n ≥ n₀

### Example 1: Sum Formula

**Theorem**: For all n ≥ 1, $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$

**Proof**:

**Base case** (n = 1):

- LHS: $\sum_{i=1}^{1} i = 1$
- RHS: $\frac{1(2)}{2} = 1$
- ✓ Equal

**Inductive hypothesis**: Assume $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$ for some k ≥ 1

**Inductive step** (show for k+1):
$$\sum_{i=1}^{k+1} i = \sum_{i=1}^{k} i + (k+1)$$
$$= \frac{k(k+1)}{2} + (k+1)$$ (by inductive hypothesis)
$$= \frac{k(k+1) + 2(k+1)}{2}$$
$$= \frac{(k+1)(k+2)}{2}$$

This is exactly the formula with n = k+1 ∎

### Example 2: Inequality

**Theorem**: For all n ≥ 4, 2ⁿ > n²

**Proof**:

**Base case** (n = 4):

- LHS: 2⁴ = 16
- RHS: 4² = 16
- Actually 16 = 16, let's use n = 5: 2⁵ = 32 > 25 = 5² ✓

**Inductive hypothesis**: Assume 2ᵏ > k² for some k ≥ 5

**Inductive step**:
$$2^{k+1} = 2 \cdot 2^k > 2k^2$$ (by hypothesis)

Need to show 2k² > (k+1)²:
$$2k^2 - (k+1)^2 = 2k^2 - k^2 - 2k - 1 = k^2 - 2k - 1$$

For k ≥ 5: k² - 2k - 1 = k(k-2) - 1 ≥ 5(3) - 1 = 14 > 0

Therefore 2^{k+1} > 2k² > (k+1)² ∎

---

## 5. Strong Induction

### Concept

Instead of assuming only P(k), assume P(n₀), P(n₀+1), ..., P(k) all hold.

```
Strong Induction:

Assume P(n₀), P(n₀+1), ..., P(k) all true
                ↓
         Prove P(k+1)
```

### When to Use

Use strong induction when P(k+1) depends on multiple previous cases.

### Example: Every Integer > 1 Has a Prime Factor

**Theorem**: Every integer n > 1 has a prime factor.

**Proof** (Strong Induction):

**Base case** (n = 2): 2 is prime, so it has itself as a prime factor ✓

**Strong inductive hypothesis**: Assume every integer from 2 to k has a prime factor.

**Inductive step** (show for k+1):

- Case 1: k+1 is prime → k+1 is its own prime factor ✓
- Case 2: k+1 is composite → k+1 = a·b where 1 < a, b < k+1
  - By strong hypothesis, a has a prime factor p
  - Since p divides a and a divides k+1, p divides k+1 ✓

Therefore every n > 1 has a prime factor ∎

---

## 6. Proof by Cases

### Concept

Divide into exhaustive cases and prove each separately.

```
Proof by Cases:

Statement to prove: P

Case 1: If condition A → P holds
Case 2: If condition B → P holds
...
Since A, B, ... cover all possibilities, P is true.
```

### Example: |xy| = |x||y|

**Theorem**: For all real x, y: |xy| = |x| · |y|

**Proof** by cases:

**Case 1**: x ≥ 0 and y ≥ 0

- |xy| = xy = |x| · |y| ✓

**Case 2**: x ≥ 0 and y < 0

- |xy| = |−xy| = −xy = x · (−y) = |x| · |y| ✓

**Case 3**: x < 0 and y ≥ 0

- |xy| = |−xy| = −xy = (−x) · y = |x| · |y| ✓

**Case 4**: x < 0 and y < 0

- |xy| = xy = (−x)(−y) = |x| · |y| ✓

All cases covered, therefore |xy| = |x| · |y| ∎

---

## 7. Existence and Uniqueness Proofs

### Existence (∃)

Show at least one object with the property exists.

**Methods**:

1. **Constructive**: Build an explicit example
2. **Non-constructive**: Show existence without construction (often via contradiction)

### Uniqueness

Show at most one object with the property exists.

**Method**: Assume two objects x and y both satisfy the property, prove x = y.

### Example: Division Algorithm

**Theorem**: For integers a and d > 0, there exist **unique** q and r such that:
$$a = qd + r, \quad 0 \leq r < d$$

**Existence**: (Constructive)

- q = ⌊a/d⌋ (floor division)
- r = a - qd
- Verify: 0 ≤ r < d ✓

**Uniqueness**:

1. Suppose a = q₁d + r₁ = q₂d + r₂ with 0 ≤ r₁, r₂ < d
2. Then (q₁ - q₂)d = r₂ - r₁
3. Since |r₂ - r₁| < d and d|(r₂ - r₁), we must have r₁ = r₂
4. Therefore q₁ = q₂ ∎

---

## 8. Applications in ML/AI

### 1. Convergence Proofs

**Theorem**: Gradient descent converges for convex functions with bounded gradients.

Uses: Direct proof, induction on iterations, inequalities.

### 2. Correctness of Algorithms

**Theorem**: Quicksort correctly sorts any input array.

Uses: Strong induction on array size.

### 3. VC Dimension Bounds

**Theorem**: The VC dimension of linear classifiers in ℝⁿ is n+1.

Uses: Existence (constructing shattered set), proof by contradiction (can't shatter n+2).

### 4. PAC Learning Bounds

**Theorem**: With probability ≥ 1-δ, the generalization error is bounded.

Uses: Probability inequalities, union bounds.

### 5. Neural Network Expressivity

**Theorem**: ReLU networks can approximate any continuous function.

Uses: Constructive existence proof.

---

## 9. Summary

### Proof Technique Selection Guide

| Situation                            | Technique                                 |
| ------------------------------------ | ----------------------------------------- |
| Show P implies Q directly            | Direct proof                              |
| Q is hard to prove directly          | Contrapositive (prove ¬Q ⟹ ¬P)            |
| Statement seems impossible otherwise | Contradiction                             |
| Statement about all n ∈ ℕ            | Induction                                 |
| P(k+1) needs multiple previous cases | Strong induction                          |
| Natural division into scenarios      | Proof by cases                            |
| "There exists"                       | Existence (construct or non-constructive) |
| "There is exactly one"               | Existence + Uniqueness                    |

### Quick Reference

```
Direct:        P ⟹ Q (chain of implications)
Contrapositive: ¬Q ⟹ ¬P (equivalent to P ⟹ Q)
Contradiction: Assume ¬S, derive P ∧ ¬P
Induction:     P(n₀) ∧ [P(k) ⟹ P(k+1)] ⟹ ∀n≥n₀: P(n)
Strong Ind:    P(n₀...k) ⟹ P(k+1)
Cases:         (C₁ ⟹ P) ∧ (C₂ ⟹ P) ∧ (C₁ ∨ C₂) ⟹ P
```

### Common Mistakes

1. **Induction**: Forgetting base case
2. **Contradiction**: Not clearly stating the assumption
3. **Contrapositive**: Confusing with converse (Q ⟹ P)
4. **Cases**: Not covering all possibilities
5. **Existence**: Claiming uniqueness without proof

---

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of all proof techniques with Python verification |
| [exercises.ipynb](exercises.ipynb) | Practice problems with step-by-step solutions |

---

## Practice Problems

1. Prove: The sum of an odd and even number is odd (direct proof)
2. Prove: If n² is even, then n is even (contrapositive)
3. Prove: There is no smallest positive rational number (contradiction)
4. Prove: $\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$ (induction)
5. Prove: Every integer can be written as a product of primes (strong induction)

---

## References

1. Velleman - "How to Prove It"
2. Hammack - "Book of Proof"
3. Sipser - "Introduction to the Theory of Computation"
4. Shalev-Shwartz & Ben-David - "Understanding Machine Learning"
