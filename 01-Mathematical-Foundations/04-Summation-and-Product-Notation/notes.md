[← Back to Mathematical Foundations](../README.md) | [Next: Einstein Summation and Index Notation →](../05-Einstein-Summation-and-Index-Notation/notes.md)

---

# Summation and Product Notation

> _"Summation and product notation are the syntax of quantitative AI reasoning — the grammar in which every loss function, every gradient, every attention score, and every probability distribution is expressed. Fluency with Σ and Π is the difference between reading ML papers as fluid prose and decoding them symbol by symbol."_

## Overview

Summation notation (Σ) and product notation (Π) are compact symbolic languages for expressing repeated addition and multiplication over collections of terms. They are not merely abbreviations — they are precise mathematical objects with well-defined rules for manipulation, interchange, and transformation.

For AI and LLMs specifically, virtually every formula you will encounter is written in Σ or Π notation: loss functions, gradients, attention scores, probability distributions, normalisation constants, perplexity, entropy, KL divergence, matrix multiplication. Without mastery of these notations, every formula requires line-by-line decoding; with it, mathematical expressions read like natural language.

This chapter covers summation and product notation from first principles through to the specific patterns that appear in modern transformer-based AI systems. Every concept is grounded in how it appears in real machine learning computation.

## Prerequisites

- Basic arithmetic operations (addition, multiplication, exponentiation)
- Familiarity with sets and index notation
- Functions and mappings (previous chapter)
- Logarithms and exponentials
- Familiarity with Python/NumPy (for companion notebook)

## Companion Notebooks

| Notebook                     | Description                                                                                    |
| ---------------------------- | ---------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb) | Interactive code: summation formulas, product identities, convergence demos, einsum operations |

## Learning Objectives

After completing this section, you will:

- Read and write summation (Σ) and product (Π) notation fluently in any ML context
- Apply all algebraic properties: linearity, splitting, reindexing, interchange, telescoping
- Derive closed-form results for arithmetic, geometric, and power series
- Convert between products and sums using logarithms (the foundation of log-likelihood)
- Understand Einstein summation convention and implement tensor operations via `torch.einsum`
- Manipulate double sums, change summation order, and handle triangular index regions
- Determine convergence of infinite series using ratio, root, comparison, and integral tests
- Decompose every major AI formula (softmax, cross-entropy, attention, perplexity, BM25) into its summation/product structure
- Identify and avoid the top 10 summation/product mistakes that cause bugs in ML code

---

## Table of Contents

- [Summation and Product Notation](#summation-and-product-notation)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Summation and Product Notation?](#11-what-is-summation-and-product-notation)
    - [1.2 Why This Notation Exists](#12-why-this-notation-exists)
    - [1.3 Where These Appear in AI](#13-where-these-appear-in-ai)
    - [1.4 The Index as a Variable](#14-the-index-as-a-variable)
    - [1.5 Historical Timeline](#15-historical-timeline)
    - [1.6 The Relationship to Integration](#16-the-relationship-to-integration)
  - [2. Summation Notation — Formal Definitions](#2-summation-notation--formal-definitions)
    - [2.1 Basic Definition](#21-basic-definition)
    - [2.2 Formal Definition via Recursion](#22-formal-definition-via-recursion)
    - [2.3 Index Set Notation](#23-index-set-notation)
    - [2.4 Multiple Indices](#24-multiple-indices)
    - [2.5 Sigma Notation Conventions](#25-sigma-notation-conventions)
  - [3. Product Notation — Formal Definitions](#3-product-notation--formal-definitions)
    - [3.1 Basic Definition](#31-basic-definition)
    - [3.2 Why Empty Product = 1 and Empty Sum = 0](#32-why-empty-product--1-and-empty-sum--0)
    - [3.3 Factorial as Product Notation](#33-factorial-as-product-notation)
    - [3.4 Product Notation in Probability](#34-product-notation-in-probability)
  - [4. Algebraic Properties of Summation](#4-algebraic-properties-of-summation)
    - [4.1 Linearity of Summation](#41-linearity-of-summation)
    - [4.2 Constant Summand](#42-constant-summand)
    - [4.3 Splitting and Combining Sums](#43-splitting-and-combining-sums)
    - [4.4 Index Shifting (Reindexing)](#44-index-shifting-reindexing)
    - [4.5 Telescoping Sums](#45-telescoping-sums)
    - [4.6 Interchange of Summation Order](#46-interchange-of-summation-order)
  - [5. Algebraic Properties of Products](#5-algebraic-properties-of-products)
    - [5.1 Product of Products (Splitting)](#51-product-of-products-splitting)
    - [5.2 Product of a Constant](#52-product-of-a-constant)
    - [5.3 Product of a Product (Nested Products)](#53-product-of-a-product-nested-products)
    - [5.4 Distributing Product over Multiplication](#54-distributing-product-over-multiplication)
    - [5.5 Logarithm Converts Product to Sum](#55-logarithm-converts-product-to-sum)
  - [6. Important Summation Formulas](#6-important-summation-formulas)
    - [6.1 Arithmetic Series](#61-arithmetic-series)
    - [6.2 Sum of Squares](#62-sum-of-squares)
    - [6.3 Sum of Cubes](#63-sum-of-cubes)
    - [6.4 Geometric Series (Finite)](#64-geometric-series-finite)
    - [6.5 Geometric Series (Infinite)](#65-geometric-series-infinite)
    - [6.6 Exponential and Taylor Series](#66-exponential-and-taylor-series)
    - [6.7 Harmonic Series and Harmonic Numbers](#67-harmonic-series-and-harmonic-numbers)
  - [7. Summation Bounds and Estimation](#7-summation-bounds-and-estimation)
    - [7.1 Comparison Test for Series](#71-comparison-test-for-series)
    - [7.2 Integral Test](#72-integral-test)
    - [7.3 Bounding Partial Sums](#73-bounding-partial-sums)
    - [7.4 Stirling's Approximation](#74-stirlings-approximation)
    - [7.5 Asymptotic Notation for Sums](#75-asymptotic-notation-for-sums)
  - [8. Special Summation Patterns in AI](#8-special-summation-patterns-in-ai)
    - [8.1 Softmax and the Normalisation Sum](#81-softmax-and-the-normalisation-sum)
    - [8.2 Cross-Entropy Loss as Double Sum](#82-cross-entropy-loss-as-double-sum)
    - [8.3 Attention as Weighted Sum](#83-attention-as-weighted-sum)
    - [8.4 Matrix Multiplication as Sum](#84-matrix-multiplication-as-sum)
    - [8.5 Gradient as Sum Over Batch](#85-gradient-as-sum-over-batch)
    - [8.6 Perplexity as Geometric Mean](#86-perplexity-as-geometric-mean)
    - [8.7 BM25 as Weighted Sum](#87-bm25-as-weighted-sum)
  - [9. Einstein Summation Convention](#9-einstein-summation-convention)
    - [9.1 Motivation](#91-motivation)
    - [9.2 Examples of Einstein Notation](#92-examples-of-einstein-notation)
    - [9.3 Rules for Einstein Notation](#93-rules-for-einstein-notation)
    - [9.4 Einstein Notation in PyTorch: einsum](#94-einstein-notation-in-pytorch-einsum)
    - [9.5 Tensor Contraction](#95-tensor-contraction)
  - [10. Double Sums and Index Interaction](#10-double-sums-and-index-interaction)
    - [10.1 Diagonal Sums](#101-diagonal-sums)
    - [10.2 Changing Order of Summation in Double Sums](#102-changing-order-of-summation-in-double-sums)
    - [10.3 Cauchy Product (Product of Two Series)](#103-cauchy-product-product-of-two-series)
    - [10.4 Vandermonde's Identity](#104-vandermondes-identity)
    - [10.5 Summation by Parts (Abel's Summation)](#105-summation-by-parts-abels-summation)
  - [11. Convergence of Infinite Series](#11-convergence-of-infinite-series)
    - [11.1 Partial Sums and Convergence](#111-partial-sums-and-convergence)
    - [11.2 Absolute Convergence](#112-absolute-convergence)
    - [11.3 Key Convergence Tests](#113-key-convergence-tests)
    - [11.4 Power Series](#114-power-series)
    - [11.5 Practical Convergence in AI Training](#115-practical-convergence-in-ai-training)
  - [12. Summation in Probability and Statistics](#12-summation-in-probability-and-statistics)
    - [12.1 Expectation as Weighted Sum](#121-expectation-as-weighted-sum)
    - [12.2 Variance as Sum](#122-variance-as-sum)
    - [12.3 Entropy as Sum](#123-entropy-as-sum)
    - [12.4 Law of Total Expectation](#124-law-of-total-expectation)
    - [12.5 Inclusion-Exclusion Principle](#125-inclusion-exclusion-principle)
  - [13. Notation Variants and Conventions](#13-notation-variants-and-conventions)
    - [13.1 Compact Notations for AI](#131-compact-notations-for-ai)
    - [13.2 Subscript and Superscript Conventions in ML](#132-subscript-and-superscript-conventions-in-ml)
    - [13.3 Dot Notation and Abbreviations](#133-dot-notation-and-abbreviations)
    - [13.4 Sigma Notation for Neural Network Layers](#134-sigma-notation-for-neural-network-layers)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters for AI (2026 Perspective)](#16-why-this-matters-for-ai-2026-perspective)
  - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is Summation and Product Notation?

Summation notation (Σ) and product notation (Π) are compact symbolic languages for expressing repeated addition and multiplication over collections of terms. They are the mathematical equivalent of a `for` loop — but with algebraic structure that allows formal manipulation.

Without them, writing the loss over a million training examples requires a million terms. With them, one line:

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(y_i \mid x_i)$$

This single expression encodes: "for each of the $N$ training examples, compute the log-probability of the correct label $y_i$ given input $x_i$ under model parameters $\theta$, negate, and average."

They are not merely abbreviations — they are precise mathematical objects with well-defined rules for manipulation, interchange, and transformation. You can:

- **Split** a sum into parts: $\sum_{i=1}^{100} f(i) = \sum_{i=1}^{50} f(i) + \sum_{i=51}^{100} f(i)$
- **Factor out** constants: $\sum_{i=1}^{n} c \cdot f(i) = c \cdot \sum_{i=1}^{n} f(i)$
- **Interchange** order: $\sum_i \sum_j f(i,j) = \sum_j \sum_i f(i,j)$ (under certain conditions)
- **Convert** products to sums: $\log \prod_i f(i) = \sum_i \log f(i)$
- **Telescope**: $\sum_{i=1}^{n}(f(i+1) - f(i)) = f(n+1) - f(1)$

For AI, fluency with Σ and Π is non-negotiable. Virtually every formula in machine learning is written in summation or product notation — loss functions, gradients, attention scores, probability distributions, normalisation constants. The researcher who reads these fluently thinks in terms of structure; the one who doesn't decodes symbol by symbol and loses the forest for the trees.

```
THE NOTATION AS A COMPRESSION TOOL
═══════════════════════════════════════════════════════════════════════

Without notation (N = 5):
L = -(1/5)(log P(y₁|x₁) + log P(y₂|x₂) + log P(y₃|x₃) + log P(y₄|x₄) + log P(y₅|x₅))

With notation (N = anything):
L = -(1/N) ∑ᵢ₌₁ᴺ log P(yᵢ|xᵢ)

The notation doesn't just save space — it reveals structure:
  • The 1/N shows it's an average
  • The log shows we're in log-probability space
  • The ∑ shows we're aggregating over examples
  • The index i shows which variable changes
```

### 1.2 Why This Notation Exists

The development of summation notation was driven by several fundamental needs:

**Human working memory is finite.** Writing out 10,000 individual terms is impossible and obscures structure. A mathematician needs to reason about _patterns_ in sums, not individual terms. Compact notation makes patterns visible.

**Notation reveals structure.** Consider $\sum_{i=1}^{n} i^2$. This immediately communicates "sum of squares from 1 to $n$" — the pattern is evident in a way that $1 + 4 + 9 + 16 + \ldots$ only suggests. The reader can immediately ask: does this have a closed form? Is it $O(n^3)$? How does it relate to $\int_0^n x^2 \, dx$?

**Enables algebraic manipulation.** Once expressed in Σ/Π notation, sums can be split, combined, reindexed, bounded, and transformed using precise algebraic rules. This is how mathematicians _prove things_ about sums, not by expanding and hand-checking.

**Generalises naturally.** The same notation scales from finite sums to infinite series to integrals:

$$\sum_{i=1}^{n} f(i) \quad \longrightarrow \quad \sum_{i=1}^{\infty} f(i) \quad \longrightarrow \quad \int_1^{\infty} f(x) \, dx$$

Same conceptual framework, deeper mathematics.

**Historical necessity.** As mathematics moved from specific numerical examples to general theorems about arbitrary $n$, notation had to scale. Gauss couldn't write "$1 + 2 + 3 + \ldots + 100 = 5050$" for every value — he needed $\sum_{i=1}^{n} i = n(n+1)/2$.

### 1.3 Where These Appear in AI

Every major formula in machine learning is a sum or product (or both). Here is a comprehensive inventory:

**Loss functions (sums):**

$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{v \in V} \mathbb{1}[y_i = v] \log P(v \mid x_i)$$

This is the categorical cross-entropy loss: a double sum over training examples ($i$) and vocabulary tokens ($v$).

**Softmax normalisation (sum in denominator):**

$$P(v \mid x) = \frac{\exp(z_v)}{\sum_{v' \in V} \exp(z_{v'})}$$

The denominator is a sum over the entire vocabulary — the partition function that ensures probabilities sum to 1.

**Attention mechanism (weighted sum):**

$$\text{Attn}(Q, K, V)_i = \sum_{j=1}^{n} \alpha_{ij} V_j, \qquad \alpha_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{\ell=1}^{n} \exp(q_i \cdot k_\ell / \sqrt{d_k})}$$

Each attention output is a weighted sum of value vectors, where the weights $\alpha_{ij}$ themselves involve summation in their softmax normalisation.

**Dot product (sum of elementwise products):**

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

The fundamental operation in attention: a sum of $d_k$ products.

**Matrix multiplication (sum over shared dimension):**

$$(AB)_{ij} = \sum_{k=1}^{d} A_{ik} B_{kj}$$

Every linear layer, every projection, every attention computation involves matrix multiplication — which is itself a collection of sums.

**Perplexity (product converted to sum via log):**

$$\text{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(t_i \mid t_1, \ldots, t_{i-1})\right)$$

The standard evaluation metric for language models: an exponentiated average of log-probabilities.

**Sequence likelihood (product):**

$$P(\text{sequence}) = \prod_{i=1}^{n} P(t_i \mid t_1, \ldots, t_{i-1})$$

The chain rule of probability applied to a sequence — a product of conditional probabilities.

**Joint probability of independent events (product):**

$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = \prod_{i=1}^{n} P(A_i)$$

**BM25 retrieval score (weighted sum):**

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1\!\left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

The standard sparse retrieval function in RAG systems: a sum over query terms, each contributing an IDF-weighted term frequency score.

**BPE merge cost (sum over pairs):**

$$\text{Cost} = \sum_{(a,b)} \text{count}(a,b) \cdot \text{merge\_cost}(a,b)$$

### 1.4 The Index as a Variable

The index variable (the $i$ in $\sum_i$) is a **bound variable** — it has no meaning outside the sum, just as the variable of integration has no meaning outside the integral.

$$\sum_{i=1}^{n} i^2 = \sum_{j=1}^{n} j^2 = \sum_{k=1}^{n} k^2 = \sum_{\text{🐱}=1}^{n} \text{🐱}^2$$

All four expressions are identical. The name of the index does not matter. This is called **alpha-equivalence**: renaming bound variables preserves meaning. It is the same principle in:

- Summation: $\sum_{i=1}^{n} f(i) = \sum_{j=1}^{n} f(j)$
- Integration: $\int_0^1 x \, dx = \int_0^1 t \, dt$
- Programming: `sum(f(i) for i in range(n))` = `sum(f(j) for j in range(n))`
- Logic: $\forall x: P(x)$ is the same as $\forall y: P(y)$

**Critical for AI.** When multiple sums interact, index names carry essential information about _which_ sum they belong to:

$$\sum_i \alpha_{ij} \neq \sum_j \alpha_{ij}$$

The left side sums over the first index (rows), producing a result that still depends on $j$. The right side sums over the second index (columns), producing a result that depends on $i$. In attention: $\sum_j \alpha_{ij} = 1$ (attention weights for query $i$ sum to 1), but $\sum_i \alpha_{ij}$ gives the total attention received by key position $j$ — a completely different quantity.

### 1.5 Historical Timeline

```
HISTORICAL DEVELOPMENT OF SUMMATION NOTATION
═══════════════════════════════════════════════════════════════════════

~1800 BCE  Egypt/Babylon    Arithmetic series computed via specific examples;
                            no general notation; "add these five numbers"

~250 BCE   Archimedes       Summation of geometric series; area under parabola
                            via method of exhaustion; still no compact notation

1593       Viète            Introduced systematic symbolic algebra; first use
                            of letters for unknowns; paved way for notation

1675       Leibniz          Introduced ∫ as elongated S for "summa" (sum);
                            integration conceived as continuous summation

18th c.    Euler            Introduced Σ (capital sigma) for discrete summation;
                            standardised the notation we use today; wrote
                            ∑ₙ₌₁^∞ 1/n² = π²/6 (Basel problem, 1735)

1777–1855  Gauss            Master of summation manipulation; derived closed
                            forms for many classical sums; the "prince of
                            mathematicians" famously summed 1+2+...+100 as a child

1821       Cauchy           Rigorous theory of series convergence; defined when
                            infinite sums are valid; ε-δ framework

1854       Riemann          Riemann sum → integral; summation as the formal
                            foundation of integration theory

1916       Einstein         Einstein summation convention: repeated indices
                            imply summation; revolutionised tensor notation

20th c.    Standard         Σ and Π become universal mathematical notation;
                            every textbook, every paper

2016+      Modern ML        Every NeurIPS/ICML/ICLR paper assumes fluency;
                            torch.einsum brings Einstein notation to code
```

### 1.6 The Relationship to Integration

The connection between summation and integration is not merely an analogy — it is a deep mathematical fact. The Riemann integral is _defined_ as a limit of sums:

$$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$

where $\Delta x = (b-a)/n$ and $x_i^*$ is a sample point in the $i$-th subinterval.

```
DISCRETE ←→ CONTINUOUS CORRESPONDENCE
═══════════════════════════════════════════════════════════════════════

SUMMATION                          INTEGRATION
──────────                         ───────────
∑ᵢ₌₁ⁿ f(i)               →       ∫₁ⁿ f(x) dx

Index i ∈ {1, 2, ..., n}   →      x ∈ [1, n]  (continuous variable)
Step size = 1               →      Infinitesimal dx
∑ → ∫ as step → 0                 (Riemann sum definition)

∏ᵢ₌₁ⁿ f(i)               →       exp(∫₁ⁿ ln f(x) dx)
Discrete product            →      Continuous product (via log)

∑ᵢ f(i)·P(i)              →       ∫ f(x)·p(x) dx
Discrete expectation        →      Continuous expectation

−∑ᵢ pᵢ log pᵢ             →       −∫ p(x) log p(x) dx
Discrete entropy            →      Differential entropy
```

This bridge matters for AI because many derivations convert between sums and integrals:

- The **evidence lower bound (ELBO)** involves expectations that are sums (discrete latent variables) or integrals (continuous latent variables)
- **Monte Carlo estimation** approximates integrals by sums: $\int f(x) p(x) dx \approx \frac{1}{K} \sum_{k=1}^{K} f(x_k)$
- **Riemann sum approximations** appear in numerical integration throughout scientific ML

---

## 2. Summation Notation — Formal Definitions

### 2.1 Basic Definition

The summation of $f(i)$ as $i$ ranges from $m$ to $n$ is defined as:

$$\sum_{i=m}^{n} f(i) = f(m) + f(m+1) + f(m+2) + \ldots + f(n)$$

The components of this notation are:

| Component | Name               | Role                                             |
| --------- | ------------------ | ------------------------------------------------ |
| $\Sigma$  | Summation operator | Capital Greek sigma; indicates repeated addition |
| $i$       | Index variable     | Dummy/bound variable; any name works             |
| $m$       | Lower limit        | First value of the index (inclusive)             |
| $n$       | Upper limit        | Last value of the index (inclusive)              |
| $f(i)$    | Summand            | Expression evaluated at each index value         |

**Number of terms:** $n - m + 1$ (when $n \geq m$).

**Empty sum convention:** If $n < m$, the sum has no terms. By convention, the value of an empty sum is **0** — the additive identity:

$$\sum_{i=5}^{3} f(i) = 0 \qquad (\text{empty sum; } 3 < 5)$$

This convention is not arbitrary; it is the _only_ value consistent with the algebraic properties of summation (see §3.2).

**Concrete examples:**

$$\sum_{i=1}^{4} i^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30$$

$$\sum_{i=0}^{3} 2^i = 2^0 + 2^1 + 2^2 + 2^3 = 1 + 2 + 4 + 8 = 15$$

$$\sum_{i=1}^{n} 1 = \underbrace{1 + 1 + \ldots + 1}_{n \text{ times}} = n$$

### 2.2 Formal Definition via Recursion

The summation notation can be defined with complete mathematical precision using recursion:

**Base case (empty sum):**

$$\sum_{i=m}^{m-1} f(i) = 0$$

**Recursive step:**

$$\sum_{i=m}^{n} f(i) = \left(\sum_{i=m}^{n-1} f(i)\right) + f(n) \qquad (n \geq m)$$

This definition is equivalent to the "expand and add" interpretation but is fully rigorous. It also makes clear why the empty sum equals 0: from the recursive step with $n = m$:

$$\sum_{i=m}^{m} f(i) = \left(\sum_{i=m}^{m-1} f(i)\right) + f(m) = 0 + f(m) = f(m) \quad \checkmark$$

The recursion unfolds exactly like a `for` loop:

```python
def sigma(f, m, n):
    """Recursive definition of ∑ᵢ₌ₘⁿ f(i)"""
    if n < m:
        return 0          # Empty sum = additive identity
    return sigma(f, m, n-1) + f(n)  # Recursive step
```

### 2.3 Index Set Notation

The basic $\sum_{i=m}^{n}$ notation assumes the index ranges over consecutive integers $\{m, m+1, \ldots, n\}$. But sums often range over arbitrary sets. The **index set notation** generalises:

$$\sum_{i \in S} f(i) = \text{sum of } f(i) \text{ for all } i \in S$$

This is the most general form. The limits notation is a special case where $S = \{m, m+1, \ldots, n\}$.

**AI-relevant examples:**

$$\sum_{v \in V} \exp(z_v) \qquad \text{(sum over all tokens } v \text{ in vocabulary } V \text{)}$$

$$\sum_{\substack{(i,j) \\ i < j}} a_{ij} \qquad \text{(sum over all pairs with } i \text{ less than } j \text{; upper triangular)}$$

$$\sum_{p \text{ prime}, \, p \leq N} \frac{1}{p} \qquad \text{(sum over primes up to } N \text{)}$$

$$\sum_{l=1}^{L} \sum_{h=1}^{H} \text{Attn}_{l,h} \qquad \text{(sum over all layers and attention heads)}$$

**Set-builder notation with conditions:**

$$\sum_{\substack{i=1 \\ i \text{ odd}}}^{n} f(i) = f(1) + f(3) + f(5) + \ldots$$

This sums only over odd values of $i$. The condition acts as a filter on the index set.

### 2.4 Multiple Indices

A **double sum** sums over two independent indices:

$$\sum_{i=1}^{m} \sum_{j=1}^{n} f(i,j) = \sum_{i=1}^{m} \left(\sum_{j=1}^{n} f(i,j)\right)$$

The process: for each fixed $i$, compute the inner sum over $j$; then sum those results over $i$.

**Total number of terms:** $m \times n$.

**Concrete example:**

$$\sum_{i=1}^{2} \sum_{j=1}^{3} ij = \sum_{i=1}^{2} (i \cdot 1 + i \cdot 2 + i \cdot 3) = \sum_{i=1}^{2} 6i = 6 + 12 = 18$$

**Triple sums** and higher are defined analogously by nesting more $\Sigma$ symbols. In AI, you frequently see:

$$\sum_{b=1}^{B} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_{b,i,j} \qquad \text{(sum over batch, query position, key position)}$$

**AI example — matrix multiplication as double sum:**

Computing all entries of $C = AB$ where $A \in \mathbb{R}^{m \times d}$, $B \in \mathbb{R}^{d \times n}$:

$$C_{ij} = \sum_{k=1}^{d} A_{ik} B_{kj}$$

To compute the entire matrix $C$, we sum over all $i, j, k$:

$$\sum_{i=1}^{m} \sum_{j=1}^{n} \sum_{k=1}^{d} A_{ik} B_{kj}$$

This triple sum involves $m \times n \times d$ individual multiply-add operations.

### 2.5 Sigma Notation Conventions

In practice, several shorthand conventions are widely used:

**Implicit limits.** When the index set is clear from context:

$$\sum_i f(i) \qquad \text{(limits understood from context)}$$

**Vector sum.** For a vector $\mathbf{x} = (x_1, x_2, \ldots, x_n)$:

$$\sum_i x_i = x_1 + x_2 + \ldots + x_n$$

**All-ones vector.** The sum of all components can be written as a dot product with the all-ones vector $\mathbf{1} = (1, 1, \ldots, 1)^\top$:

$$\mathbf{1}^\top \mathbf{x} = \sum_i x_i$$

This is heavily used in ML: `torch.sum(x)` is the same as `ones @ x`.

**Indicator function notation.** The indicator (or Iverson bracket) $\mathbb{1}[\text{condition}]$ equals 1 when the condition is true and 0 otherwise:

$$\sum_i \mathbb{1}[\text{condition}(i)] = \text{number of } i \text{ satisfying the condition}$$

For example, $\sum_{i=1}^{n} \mathbb{1}[x_i > 0]$ counts the number of positive elements in a vector.

**Vector/matrix sums.** When summing vectors or matrices, the sum is computed componentwise:

$$\sum_{i=1}^{n} \mathbf{v}_i = \mathbf{v}_1 + \mathbf{v}_2 + \ldots + \mathbf{v}_n \qquad \text{(each } \mathbf{v}_i \text{ is a vector)}$$

This appears in attention: $\text{Attn}_i = \sum_j \alpha_{ij} \mathbf{V}_j$ is a weighted sum of vectors.

---

## 3. Product Notation — Formal Definitions

### 3.1 Basic Definition

The product of $f(i)$ as $i$ ranges from $m$ to $n$ is defined as:

$$\prod_{i=m}^{n} f(i) = f(m) \cdot f(m+1) \cdot f(m+2) \cdots f(n)$$

The components are identical to summation notation but with $\Pi$ (capital Greek pi) indicating repeated multiplication instead of addition.

| Component | Name             | Role                                                |
| --------- | ---------------- | --------------------------------------------------- |
| $\Pi$     | Product operator | Capital Greek pi; indicates repeated multiplication |
| $i$       | Index variable   | Dummy/bound variable; any name works                |
| $m$       | Lower limit      | First value of the index (inclusive)                |
| $n$       | Upper limit      | Last value of the index (inclusive)                 |
| $f(i)$    | Factor           | Expression evaluated at each index value            |

**Number of factors:** $n - m + 1$ (same counting as summation).

**Empty product convention:** If $n < m$, the product has no factors. By convention, the value of an empty product is **1** — the multiplicative identity:

$$\prod_{i=5}^{3} f(i) = 1 \qquad (\text{empty product; } 3 < 5)$$

**Recursive definition:**

$$\prod_{i=m}^{m-1} f(i) = 1 \qquad (\text{base case: empty product})$$

$$\prod_{i=m}^{n} f(i) = \left(\prod_{i=m}^{n-1} f(i)\right) \cdot f(n) \qquad (n \geq m)$$

**Concrete examples:**

$$\prod_{i=1}^{4} i = 1 \cdot 2 \cdot 3 \cdot 4 = 24 = 4!$$

$$\prod_{i=0}^{3} 2 = 2^4 = 16$$

$$\prod_{i=1}^{n} \frac{1}{P(t_i | \text{context})} = \frac{1}{P(t_1|\text{ctx})} \cdot \frac{1}{P(t_2|\text{ctx})} \cdots \frac{1}{P(t_n|\text{ctx})} \qquad \text{(inverse probability product)}$$

### 3.2 Why Empty Product = 1 and Empty Sum = 0

These conventions are not arbitrary — they are the **only values** consistent with the algebraic rules:

**Additive identity argument.** The identity for addition is 0: $a + 0 = a$. An empty sum contributes no terms, so it must equal the additive identity:

$$\sum_{\emptyset} = 0$$

**Consistency check:** The recursive definition gives $\sum_{i=1}^{1} f(i) = \sum_{i=1}^{0} f(i) + f(1) = 0 + f(1) = f(1)$. If the empty sum were anything other than 0, this would fail. ✓

**Multiplicative identity argument.** The identity for multiplication is 1: $a \cdot 1 = a$. An empty product contributes no factors, so it must equal the multiplicative identity:

$$\prod_{\emptyset} = 1$$

**Consistency check:** The recursive definition gives $\prod_{i=1}^{1} f(i) = \prod_{i=1}^{0} f(i) \cdot f(1) = 1 \cdot f(1) = f(1)$. ✓

**Factorial consistency.** The factorial $n! = \prod_{i=1}^{n} i$ must satisfy $0! = 1$:

$$0! = \prod_{i=1}^{0} i = 1 \qquad (\text{empty product convention})$$

This is also consistent with $\Gamma(1) = 1$, the combinatorial identity $\binom{n}{0} = \frac{n!}{0! \cdot n!} = 1$, and the Taylor series $e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$ (which needs $0! = 1$ for the $n = 0$ term to be $x^0/0! = 1$).

```
IDENTITY ELEMENTS — THE PATTERN
═══════════════════════════════════════════════════════════════════════

Operation      Identity     Empty Result     Why
─────────      ────────     ────────────     ───
Addition       0            ∑∅ = 0           Adding nothing = 0
Multiplication 1            ∏∅ = 1           Multiplying nothing = 1
Union          ∅            ⋃∅ = ∅           Unioning nothing = empty set
Intersection   Universal    ⋂∅ = U           Intersecting nothing = everything
Logical AND    True         ∧∅ = True        No conditions to fail
Logical OR     False        ∨∅ = False       No conditions to satisfy
```

### 3.3 Factorial as Product Notation

The factorial of $n$ is defined as:

$$n! = \prod_{i=1}^{n} i = 1 \cdot 2 \cdot 3 \cdots n$$

with the convention $0! = 1$ (empty product).

**Growth rate.** Factorial grows faster than any polynomial or exponential base:

| $n$ | $n!$        | $n^n$          | $2^n$     | $n^2$ |
| --- | ----------- | -------------- | --------- | ----- |
| 1   | 1           | 1              | 2         | 1     |
| 5   | 120         | 3,125          | 32        | 25    |
| 10  | 3,628,800   | 10,000,000,000 | 1,024     | 100   |
| 20  | 2.43 × 10¹⁸ | 1.05 × 10²⁶    | 1,048,576 | 400   |

For large $n$, Stirling's approximation gives:

$$n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$$

**AI relevance:**

- **Permutations:** The number of permutations of $n$ items is $n!$. This creates combinatorial explosion in search spaces. A vocabulary of just 30 tokens has $30! \approx 2.65 \times 10^{32}$ possible orderings.
- **Permutation symmetry in neural networks:** A hidden layer with $n$ neurons has $n!$ equivalent parameter configurations obtained by permuting the neurons. This symmetry means the loss landscape has at least $n!$ equivalent minima.
- **Combinatorics:** Binomial coefficients $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ appear in counting arguments throughout ML theory.

### 3.4 Product Notation in Probability

Product notation is the natural language of probability, where independent events multiply:

**Joint probability of independent events:**

$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = \prod_{i=1}^{n} P(A_i) \qquad \text{(if } A_1, \ldots, A_n \text{ are independent)}$$

**Likelihood of a sequence (chain rule of probability):**

$$P(t_1, t_2, \ldots, t_n) = \prod_{i=1}^{n} P(t_i \mid t_1, \ldots, t_{i-1})$$

This is the fundamental equation of autoregressive language models. A transformer computes each factor $P(t_i \mid t_1, \ldots, t_{i-1})$ using causal attention, and multiplies them all together to get the sequence probability.

**Log-likelihood converts product to sum:**

$$\log P(t_1, \ldots, t_n) = \log \prod_{i=1}^{n} P(t_i \mid t_1, \ldots, t_{i-1}) = \sum_{i=1}^{n} \log P(t_i \mid t_1, \ldots, t_{i-1})$$

This conversion from $\Pi$ to $\Sigma$ via logarithm is one of the most important transformations in all of machine learning. It is used because:

1. **Numerical stability:** Products of many small probabilities underflow to zero ($0.01^{100} = 10^{-200}$, which is below FP64 minimum). Taking logarithms converts to a sum of manageable-sized numbers ($100 \times \log(0.01) = -460.5$).

2. **Computational convenience:** Sums are easier to differentiate than products. The gradient of $\sum_i \log P_i$ is $\sum_i \frac{\nabla P_i}{P_i}$, cleanly decomposing into per-example terms.

3. **Statistical theory:** The negative log-likelihood is the cross-entropy loss, connecting maximum likelihood estimation to information theory.

```
THE PRODUCT-TO-SUM CONVERSION VIA LOGARITHM
═══════════════════════════════════════════════════════════════════════

                    log
    ∏ᵢ P(tᵢ|ctx) ────────→ ∑ᵢ log P(tᵢ|ctx)

    Product of           Sum of
    probabilities        log-probabilities

    Underflows for       Numerically stable
    long sequences       for any length

    Hard to              Easy to
    differentiate        differentiate

    ← ─ ─ ─ ─ exp ─ ─ ─ ─ ←

    exp(∑ᵢ log Pᵢ) = ∏ᵢ Pᵢ     (inverse transformation)
```

---

## 4. Algebraic Properties of Summation

These properties are the tools for manipulating sums — the algebraic rules that allow you to simplify, decompose, and transform summation expressions.

### 4.1 Linearity of Summation

The most important property. Summation is a **linear operator**:

**Scalar multiple:**

$$\sum_{i=1}^{n} c \cdot f(i) = c \cdot \sum_{i=1}^{n} f(i) \qquad \text{(}c \text{ constant w.r.t. } i\text{)}$$

**Sum of sums:**

$$\sum_{i=1}^{n} \bigl(f(i) + g(i)\bigr) = \sum_{i=1}^{n} f(i) + \sum_{i=1}^{n} g(i)$$

**Combined (linearity):**

$$\sum_{i=1}^{n} \bigl(\alpha f(i) + \beta g(i)\bigr) = \alpha \sum_{i=1}^{n} f(i) + \beta \sum_{i=1}^{n} g(i)$$

**Proof sketch:** Expand both sides by the definition of summation (write out all terms); use commutativity and associativity of addition to rearrange; observe both sides are equal term by term.

**Why this matters for AI:** Linearity allows decomposition of complex sums into simpler parts. The most critical application:

$$\nabla_\theta \sum_{i=1}^{N} L(\theta, x_i) = \sum_{i=1}^{N} \nabla_\theta L(\theta, x_i)$$

The gradient of a sum equals the sum of gradients. This is why mini-batch gradient descent works: You can compute per-example gradients independently and then sum them. It is also why gradient accumulation works: you can split a large batch into smaller sub-batches, compute gradients for each, and sum them — the result is identical to computing the gradient of the full batch.

**Warning:** The constant $c$ must **not** depend on the summation index $i$. If $c$ depends on $i$, it cannot be factored out:

$$\sum_{i=1}^{n} c(i) \cdot f(i) \neq c(\cdot) \cdot \sum_{i=1}^{n} f(i) \qquad \text{(WRONG if } c \text{ depends on } i\text{)}$$

### 4.2 Constant Summand

$$\sum_{i=1}^{n} c = n \cdot c$$

The sum of $n$ copies of a constant $c$ equals $n$ times $c$. This follows from linearity: $\sum_{i=1}^{n} c = c \cdot \sum_{i=1}^{n} 1 = c \cdot n$.

**Special case:** $\sum_{i=1}^{n} 1 = n$ — a useful identity for counting.

**AI applications:**

- **Normalisation check:** $\sum_{v=1}^{|V|} P(v) = 1$ — all probabilities must sum to 1. If $P$ is uniform over $|V|$ tokens, then $P(v) = 1/|V|$ and $\sum_v P(v) = |V| \cdot (1/|V|) = 1$. ✓
- **Batch average:** $\sum_{i=1}^{N} (1/N) = 1$ — confirms that $(1/N) \sum_i f(i)$ is a proper average.
- **Attention conservation:** $\sum_j \alpha_{ij} = 1$ for softmax attention weights — each query's weights sum to 1.

### 4.3 Splitting and Combining Sums

**Splitting at index $k$** (where $m \leq k < n$):

$$\sum_{i=m}^{n} f(i) = \sum_{i=m}^{k} f(i) + \sum_{i=k+1}^{n} f(i)$$

Both parts are contiguous and their union covers the full range.

**Merging adjacent ranges:**

$$\sum_{i=m}^{k} f(i) + \sum_{i=k+1}^{n} f(i) = \sum_{i=m}^{n} f(i)$$

**Non-adjacent ranges:** For disjoint sets:

$$\sum_{i \in S} f(i) + \sum_{i \in T} f(i) = \sum_{i \in S \cup T} f(i) \qquad \text{(only if } S \cap T = \emptyset\text{)}$$

If $S \cap T \neq \emptyset$, elements in the intersection are counted twice.

**AI applications:**

- **Mini-batch splitting:** Split gradient computation across mini-batches: $\sum_{i=1}^{N} \nabla l_i = \sum_{i=1}^{|B_1|} \nabla l_i + \sum_{i=|B_1|+1}^{|B_1|+|B_2|} \nabla l_i + \ldots$
- **Distributed training:** Each GPU computes a partial sum; the all-reduce operation merges them.
- **Gradient accumulation:** Accumulate gradients over $K$ micro-batches before applying an update: $\nabla L_{\text{effective}} = \sum_{b=1}^{K} \nabla L_{B_b}$.

### 4.4 Index Shifting (Reindexing)

**Shift by constant $k$:**

$$\sum_{i=m}^{n} f(i) = \sum_{j=m+k}^{n+k} f(j - k) \qquad \text{where } j = i + k$$

Both limits shift by $k$, and the summand substitutes $i = j - k$.

**Rename index (alpha-equivalence):**

$$\sum_{i=m}^{n} f(i) = \sum_{j=m}^{n} f(j) \qquad \text{(different name, same sum)}$$

**Reverse index:**

$$\sum_{i=m}^{n} f(i) = \sum_{i=m}^{n} f(m + n - i) \qquad \text{(substitute } i \to m + n - i\text{)}$$

When $i$ goes from $m$ to $n$, the value $m + n - i$ goes from $n$ down to $m$. The same terms are summed in reverse order.

**Applications of reindexing:**

- **Gauss's trick:** Reverse and add to derive $\sum_{i=1}^{n} i = n(n+1)/2$:
  - $S = 1 + 2 + \ldots + n$
  - $S = n + (n-1) + \ldots + 1$ (reversed)
  - $2S = (n+1) + (n+1) + \ldots + (n+1) = n(n+1)$
  - $S = n(n+1)/2$
- **AI: relative positional encodings:** Reindex from absolute positions to relative: $\sum_j f(i, j) = \sum_{\delta} f(i, i - \delta)$ where $\delta = i - j$.
- **Causal attention:** Reindex to count from current position backward.

### 4.5 Telescoping Sums

A **telescoping sum** is one where most terms cancel in pairs, leaving only the first and last:

$$\sum_{i=1}^{n} \bigl(f(i+1) - f(i)\bigr) = f(n+1) - f(1)$$

**Proof by expansion:**

$$\bigl(f(2) - f(1)\bigr) + \bigl(f(3) - f(2)\bigr) + \bigl(f(4) - f(3)\bigr) + \ldots + \bigl(f(n+1) - f(n)\bigr)$$

Every intermediate term appears once with $+$ and once with $-$; they cancel:

$$= -f(1) + \cancel{f(2)} - \cancel{f(2)} + \cancel{f(3)} - \cancel{f(3)} + \ldots + \cancel{f(n)} - \cancel{f(n)} + f(n+1) = f(n+1) - f(1)$$

**Generalised form:**

$$\sum_{i=1}^{n} \bigl(g(i) - g(i-1)\bigr) = g(n) - g(0)$$

**Example — partial fractions:** Show that $\sum_{i=1}^{n} \frac{1}{i(i+1)} = \frac{n}{n+1}$:

$$\frac{1}{i(i+1)} = \frac{1}{i} - \frac{1}{i+1}$$

$$\sum_{i=1}^{n} \left(\frac{1}{i} - \frac{1}{i+1}\right) = \frac{1}{1} - \frac{1}{n+1} = \frac{n}{n+1}$$

**AI applications:**

- **Residual networks:** The total change in a ResNet's representation across $T$ layers: $\Delta h = h_T - h_0 = \sum_{t=1}^{T} (h_t - h_{t-1}) = \sum_{t=1}^{T} \delta h_t$. The total transformation telescopes: it equals the sum of all residual updates.
- **Gradient descent accumulation:** Total parameter change: $\theta_T - \theta_0 = \sum_{t=1}^{T} (\theta_t - \theta_{t-1}) = -\sum_{t=1}^{T} \eta_t \nabla L(\theta_{t-1})$.
- **Proving closed-form formulas:** Many summation identities are proved by finding a function $g$ such that $f(i) = g(i) - g(i-1)$.

### 4.6 Interchange of Summation Order

For a **double sum over a rectangular region**, the order of summation can always be interchanged:

$$\sum_{i=1}^{m} \sum_{j=1}^{n} f(i,j) = \sum_{j=1}^{n} \sum_{i=1}^{m} f(i,j)$$

**Why this works:** The double sum ranges over all pairs $(i, j)$ with $1 \leq i \leq m$ and $1 \leq j \leq n$. The left side sums row by row (fix $i$, sweep $j$); the right side sums column by column (fix $j$, sweep $i$). Both cover the same set of $m \times n$ pairs, and addition is commutative and associative.

```
SUMMING A MATRIX: ROW-FIRST vs COLUMN-FIRST
═══════════════════════════════════════════════════════════════════════

    j=1  j=2  j=3          Row-first (∑ᵢ ∑ⱼ):
   ┌────┬────┬────┐         Row 1: a₁₁ + a₁₂ + a₁₃ = R₁
i=1│ a₁₁│ a₁₂│ a₁₃│        Row 2: a₂₁ + a₂₂ + a₂₃ = R₂
   ├────┼────┼────┤         Total: R₁ + R₂
i=2│ a₂₁│ a₂₂│ a₂₃│
   └────┴────┴────┘        Column-first (∑ⱼ ∑ᵢ):
                             Col 1: a₁₁ + a₂₁ = C₁
                             Col 2: a₁₂ + a₂₂ = C₂
                             Col 3: a₁₃ + a₂₃ = C₃
                             Total: C₁ + C₂ + C₃

Both give the same total: all 6 elements summed.
```

**For finite sums:** interchange is **always** safe. No conditions needed.

**For infinite sums:** interchange requires **absolute convergence**: $\sum_i \sum_j |f(i,j)| < \infty$. Without this, rearrangement can change the value (Riemann rearrangement theorem).

**AI applications:**

- **Linearity of expectation:** $\mathbb{E}\left[\sum_i X_i\right] = \sum_i \mathbb{E}[X_i]$. This is interchange of summation and expectation (which is itself a sum/integral).
- **Multi-head attention:** $\sum_h \sum_i \equiv \sum_i \sum_h$ — compute attention for all heads first then sum, or compute per-position then sum over heads; same result.
- **Batch-feature decomposition:** When computing gradients, you can sum over the batch dimension first or the feature dimension first; the order doesn't matter for finite sums.

---

## 5. Algebraic Properties of Products

### 5.1 Product of Products (Splitting)

Just as sums can be split at any index, products can be split at any index:

$$\prod_{i=m}^{n} f(i) = \prod_{i=m}^{k} f(i) \cdot \prod_{i=k+1}^{n} f(i) \qquad (m \leq k < n)$$

**Proof:** The full product $f(m) \cdot f(m+1) \cdots f(k) \cdot f(k+1) \cdots f(n)$ can be grouped into two sub-products at any break point $k$.

**AI application:** Factoring a sequence probability:

$$P(t_1, \ldots, t_n) = \underbrace{P(t_1, \ldots, t_k)}_{\text{prefix probability}} \cdot \underbrace{P(t_{k+1}, \ldots, t_n \mid t_1, \ldots, t_k)}_{\text{continuation probability}}$$

This splitting enables prefix-based caching in autoregressive generation: compute the prefix probability once, then extend it token by token.

### 5.2 Product of a Constant

$$\prod_{i=1}^{n} c = c^n$$

Multiplying $n$ copies of constant $c$ equals $c$ to the $n$-th power.

**Special cases:**

- $\prod_{i=1}^{n} 2 = 2^n$ — binary growth
- $\prod_{i=1}^{n} (1/2) = (1/2)^n = 2^{-n}$ — exponential decay
- $\prod_{i=1}^{n} (-1) = (-1)^n$ — alternating sign

**AI application:** If a model assigns uniform probability $1/|V|$ to each token in a sequence of length $n$:

$$P(\text{sequence}) = \prod_{i=1}^{n} \frac{1}{|V|} = \frac{1}{|V|^n}$$

With vocabulary size $|V| = 50{,}000$ and $n = 100$ tokens, the probability is $(1/50{,}000)^{100} \approx 10^{-470}$ — far below any floating-point representation, which is exactly why we use log-probabilities.

### 5.3 Product of a Product (Nested Products)

Nested products over independent index sets can be interchanged, just like double sums:

$$\prod_{i=1}^{m} \prod_{j=1}^{n} f(i,j) = \prod_{j=1}^{n} \prod_{i=1}^{m} f(i,j)$$

Both sides multiply together the same $m \times n$ factors; only the order differs. Since multiplication is commutative and associative, the result is the same.

### 5.4 Distributing Product over Multiplication

Product distributes over **pointwise multiplication** of terms:

$$\prod_{i=1}^{n} \bigl(f(i) \cdot g(i)\bigr) = \left(\prod_{i=1}^{n} f(i)\right) \cdot \left(\prod_{i=1}^{n} g(i)\right)$$

**Proof:** Expand: $(f(1)g(1))(f(2)g(2))\cdots(f(n)g(n))$. By commutativity of multiplication, rearrange to $(f(1)f(2)\cdots f(n))(g(1)g(2)\cdots g(n))$.

**Product does NOT distribute over addition:**

$$\prod_{i=1}^{n} \bigl(f(i) + g(i)\bigr) \neq \prod_{i=1}^{n} f(i) + \prod_{i=1}^{n} g(i)$$

**Counterexample:** $(a + b)(c + d) = ac + ad + bc + bd \neq ac + bd$.

This is a common source of errors. While multiplying the product distributes cleanly, adding inside a product creates cross-terms — the result expands into $2^n$ terms in general.

### 5.5 Logarithm Converts Product to Sum

This is arguably the most important property of product notation for all of machine learning:

$$\log \prod_{i=1}^{n} f(i) = \sum_{i=1}^{n} \log f(i)$$

**Proof:** By the fundamental logarithm identity $\log(ab) = \log(a) + \log(b)$, applied repeatedly:

$$\log(f(1) \cdot f(2) \cdots f(n)) = \log f(1) + \log f(2) + \ldots + \log f(n) = \sum_{i=1}^{n} \log f(i)$$

**Inverse direction:**

$$\prod_{i=1}^{n} f(i) = \exp\!\left(\sum_{i=1}^{n} \log f(i)\right)$$

**Why this matters immensely for AI:**

1. **Numerical stability.** Products of many small probabilities underflow to zero in finite-precision arithmetic. If $P(t_i | \text{ctx}) = 0.01$ for all $i$ in a sequence of length 100, then $\prod P_i = 10^{-200}$, which is below the minimum of FP32 ($\approx 10^{-38}$). But $\sum \log P_i = -460.5$, perfectly representable.

2. **Gradient computation.** Differentiating sums is straightforward: $\frac{\partial}{\partial \theta} \sum_i \log P_i = \sum_i \frac{\partial \log P_i}{\partial \theta}$. Differentiating products requires the product rule applied $n-1$ times.

3. **Decomposition.** Log-likelihood decomposes into per-example terms: each $\log P(t_i | \text{ctx})$ can be computed, stored, and analysed independently.

**AI examples of product-to-sum conversion:**

$$\underbrace{\log P(t_1, \ldots, t_n)}_{\text{log-likelihood}} = \underbrace{\sum_{i=1}^{n} \log P(t_i \mid \text{context})}_{\text{sum of log-probs}}$$

$$\underbrace{-\frac{1}{N} \sum_{i=1}^{N} \log P(t_i \mid \text{ctx})}_{\text{cross-entropy loss}} = \underbrace{-\log \prod_{i=1}^{N} P(t_i \mid \text{ctx})^{1/N}}_{\text{negative log geometric mean}}$$

$$\underbrace{\text{PPL}}_{\text{perplexity}} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \log P\right) = \left(\prod_{i=1}^{N} \frac{1}{P(t_i \mid \text{ctx})}\right)^{1/N}$$

---

## 6. Important Summation Formulas

These closed-form results are essential tools. Knowing them lets you immediately evaluate sums that arise in complexity analysis, error bounds, and algorithmic design.

### 6.1 Arithmetic Series

$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

**Gauss's derivation (pairing argument):** Write the sum forward and backward:

$$S = 1 + 2 + 3 + \ldots + n$$
$$S = n + (n-1) + (n-2) + \ldots + 1$$

Adding these two equations column by column:

$$2S = (1+n) + (2+n-1) + (3+n-2) + \ldots + (n+1) = n \times (n+1)$$

Therefore $S = n(n+1)/2$.

**General arithmetic series** (first term $a$, common difference $d$, $n$ terms):

$$\sum_{i=0}^{n-1} (a + id) = na + \frac{n(n-1)}{2} d = \frac{n}{2}\bigl(2a + (n-1)d\bigr) = \frac{n(\text{first} + \text{last})}{2}$$

**Sum of first $n$ odd numbers:**

$$\sum_{i=1}^{n} (2i - 1) = n^2$$

Proof: $\sum_{i=1}^{n} (2i - 1) = 2 \sum_{i=1}^{n} i - \sum_{i=1}^{n} 1 = 2 \cdot \frac{n(n+1)}{2} - n = n^2 + n - n = n^2$.

**AI relevance:** The number of pairwise interactions in full self-attention over $n$ tokens is $\sum_{i=1}^{n} n = n^2$ (each of $n$ queries attends to $n$ keys), giving $O(n^2)$ attention complexity. More precisely, causal attention has $\sum_{i=1}^{n} i = n(n+1)/2$ interactions — still $O(n^2)$ but roughly half.

### 6.2 Sum of Squares

$$\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$$

**Proof by telescoping.** Use the identity $i^3 - (i-1)^3 = 3i^2 - 3i + 1$:

$$\sum_{i=1}^{n} \bigl(i^3 - (i-1)^3\bigr) = n^3 \qquad \text{(telescoping)}$$

$$\sum_{i=1}^{n} (3i^2 - 3i + 1) = n^3$$

$$3\sum i^2 - 3\sum i + n = n^3$$

$$3\sum i^2 = n^3 + 3 \cdot \frac{n(n+1)}{2} - n = n^3 + \frac{3n^2 + 3n}{2} - n$$

$$\sum i^2 = \frac{n(n+1)(2n+1)}{6}$$

**AI relevance:** Squared norms $\|\mathbf{x}\|^2 = \sum_i x_i^2$ appear everywhere — L2 regularisation, variance calculations, gradient norms. While $\sum_i i^2$ specifically appears in complexity analysis, the sum-of-squares pattern is ubiquitous.

### 6.3 Sum of Cubes

$$\sum_{i=1}^{n} i^3 = \left(\frac{n(n+1)}{2}\right)^2 = \left(\sum_{i=1}^{n} i\right)^2$$

A remarkable identity: **the sum of cubes equals the square of the sum of the first $n$ integers**.

| $n$ | $\sum i^3$            | $(\sum i)^2$ | Equal? |
| --- | --------------------- | ------------ | ------ |
| 1   | 1                     | 1² = 1       | ✓      |
| 2   | 1 + 8 = 9             | 3² = 9       | ✓      |
| 3   | 1 + 8 + 27 = 36       | 6² = 36      | ✓      |
| 4   | 1 + 8 + 27 + 64 = 100 | 10² = 100    | ✓      |

**Proof by induction:** Base case $n = 1$: $1^3 = 1 = (1 \cdot 2/2)^2 = 1$. ✓  
Inductive step: assume $\sum_{i=1}^{k} i^3 = (k(k+1)/2)^2$. Then:

$$\sum_{i=1}^{k+1} i^3 = \frac{k^2(k+1)^2}{4} + (k+1)^3 = (k+1)^2 \left(\frac{k^2}{4} + k + 1\right) = (k+1)^2 \cdot \frac{(k+2)^2}{4}$$

which equals $((k+1)(k+2)/2)^2$. ✓

### 6.4 Geometric Series (Finite)

$$\sum_{i=0}^{n-1} r^i = \frac{1 - r^n}{1 - r} \qquad (r \neq 1)$$

**Proof:** Let $S = 1 + r + r^2 + \ldots + r^{n-1}$. Then:

$$rS = r + r^2 + \ldots + r^n$$
$$S - rS = 1 - r^n$$
$$S(1 - r) = 1 - r^n$$
$$S = \frac{1 - r^n}{1 - r}$$

**Special case $r = 1$:** $\sum_{i=0}^{n-1} 1 = n$.

**Shifted version:** $\sum_{i=m}^{n} r^i = r^m \cdot \frac{1 - r^{n-m+1}}{1 - r}$.

**Alternative form (ratio > 1):** $\sum_{i=0}^{n-1} r^i = \frac{r^n - 1}{r - 1}$ for $r > 1$.

**AI relevance:**

- **ALiBi (Attention with Linear Biases):** Uses geometric-like decay in attention: positions further away receive exponentially decreasing attention bias.
- **Exponential moving averages:** In Adam optimizer, $\hat{m}_t = (1 - \beta_1) \sum_{i=0}^{t-1} \beta_1^i g_{t-i}$ — the weights form a geometric series summing to $\frac{1 - \beta_1^t}{1 - \beta_1}$.
- **Discounted rewards in RL:** $\sum_{t=0}^{T} \gamma^t r_t$ where $\gamma < 1$ is the discount factor.

### 6.5 Geometric Series (Infinite)

$$\sum_{i=0}^{\infty} r^i = \frac{1}{1 - r} \qquad (|r| < 1)$$

This is the limit of the finite geometric series as $n \to \infty$: since $|r| < 1$, we have $r^n \to 0$, so $\frac{1 - r^n}{1 - r} \to \frac{1}{1 - r}$.

For $|r| \geq 1$, the series **diverges** (terms do not shrink to zero).

**Derivatives of geometric series:**

$$\sum_{i=1}^{\infty} i r^{i-1} = \frac{1}{(1 - r)^2} \qquad (|r| < 1)$$

$$\sum_{i=0}^{\infty} i r^i = \frac{r}{(1 - r)^2} \qquad (|r| < 1)$$

These are obtained by differentiating $\sum r^i = \frac{1}{1 - r}$ with respect to $r$ term by term (valid within the radius of convergence).

**AI relevance:**

- **Expected geometric distribution:** If a random variable $X \sim \text{Geom}(p)$, then $\mathbb{E}[X] = \sum_{k=1}^{\infty} k(1-p)^{k-1} p = 1/p$, which uses the derivative of the geometric series.
- **Discounted return in RL:** $\sum_{t=0}^{\infty} \gamma^t = \frac{1}{1 - \gamma}$ gives the maximum possible return under discount $\gamma$.

### 6.6 Exponential and Taylor Series

Many functions used in AI are defined by their Taylor (power) series — which are infinite sums:

**Exponential function:**

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots$$

Converges for all $x \in \mathbb{R}$. The exponential function is the unique function satisfying $f'(x) = f(x)$ with $f(0) = 1$.

**Sine and cosine:**

$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \ldots$$

$$\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \ldots$$

**Natural logarithm:**

$$\ln(1 + x) = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \ldots \qquad (|x| \leq 1, \; x \neq -1)$$

**Binomial series:**

$$(1 + x)^\alpha = \sum_{n=0}^{\infty} \binom{\alpha}{n} x^n \qquad (|x| < 1)$$

where $\binom{\alpha}{n} = \frac{\alpha(\alpha-1)\cdots(\alpha - n + 1)}{n!}$ (generalised binomial coefficient).

**AI relevance:**

- **Softmax** uses $\exp(z)$; its Taylor series explains why $\exp(z) \approx 1 + z$ for small $z$ — helpful for linearised attention approximations (e.g., Performers).
- **Log-sum-exp trick** uses $\ln()$; understanding its series expansion helps with numerical stability analysis.
- **Positional encodings** in transformers use $\sin()$ and $\cos()$; their series representations explain periodicity and frequency properties.
- **GELU activation:** $\text{GELU}(x) = x \Phi(x)$ where $\Phi$ involves the error function, which has a Taylor series expansion.

### 6.7 Harmonic Series and Harmonic Numbers

The **harmonic series** is the sum of reciprocals:

$$\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \ldots = \infty \qquad \text{(diverges!)}$$

Despite the terms approaching zero, the sum grows without bound. This is a classic example showing that $a_n \to 0$ is necessary but not sufficient for convergence.

The $n$-th **harmonic number** is the partial sum:

$$H_n = \sum_{i=1}^{n} \frac{1}{i} \approx \ln(n) + \gamma$$

where $\gamma \approx 0.5772$ is the **Euler-Mascheroni constant**.

**The p-series** generalises the harmonic series:

$$\sum_{n=1}^{\infty} \frac{1}{n^p} \qquad \begin{cases} \text{converges} & \text{if } p > 1 \\ \text{diverges} & \text{if } p \leq 1 \end{cases}$$

The special case $p = 2$ gives Euler's famous result: $\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6} \approx 1.6449$.

The Riemann zeta function $\zeta(p) = \sum_{n=1}^{\infty} \frac{1}{n^p}$ unifies all $p$-series.

**AI relevance:**

- **Zipf's law:** Word frequency in natural language follows $\text{freq}(r) \propto 1/r$ where $r$ is the rank. The total frequency is proportional to $H_{|V|}$, a harmonic number.
- **BPE merge counts** follow approximately Zipfian distributions; analysis involves harmonic numbers.
- **Complexity analysis:** $\sum_{i=1}^{n} 1/i = O(\log n)$; this appears in analyses of algorithm running times and expected values of random processes.
- **Coupon collector problem:** Expected number of samples to see all $n$ items is $n H_n \approx n \ln n$ — relevant for coverage analysis in data sampling.

---

## 7. Summation Bounds and Estimation

In both theoretical analysis and practical AI engineering, we rarely need the **exact** value of a sum. Instead, we need **tight bounds** — upper and lower estimates that tell us how a sum grows. This section equips you with the main bounding techniques.

### 7.1 Comparison Test

If $0 \leq a_i \leq b_i$ for all $i$, then:

$$
\sum_{i} a_i \leq \sum_{i} b_i
$$

**Why it matters:** This is the workhorse for bounding sums whose exact evaluation is intractable.

**Worked example — bounding softmax tail:**

Suppose attention logits satisfy $z_i \leq z_{\max} - \delta_i$ where $\delta_i \geq c \cdot i$ for tokens sorted by relevance. Then:

$$
\sum_{i=k}^{n} e^{z_i} \leq \sum_{i=k}^{n} e^{z_{\max} - ci} = e^{z_{\max}} \sum_{i=k}^{n} e^{-ci} \leq e^{z_{\max}} \cdot \frac{e^{-ck}}{1 - e^{-c}}
$$

This shows that the tail of the softmax distribution decays exponentially — justifying **sparse attention** approximations that ignore low-scoring tokens.

**General pattern:**

```
To bound ∑ f(i):
  1. Find a simpler g(i) with f(i) ≤ g(i) for all i     (upper bound)
  2. Find a simpler h(i) with h(i) ≤ f(i) for all i     (lower bound)
  3. Evaluate ∑ g(i) and ∑ h(i) using known formulas
```

### 7.2 Integral Test

For a positive, monotonically decreasing function $f$:

$$
\int_{1}^{n+1} f(x)\,dx \leq \sum_{i=1}^{n} f(i) \leq f(1) + \int_{1}^{n} f(x)\,dx
$$

```
  f(x) │
       │█
       │█░█
       │█░█░█
       │█░█░█░░░
       │█░█░█░░░░░░
       └──┴──┴──┴──┴──── i
        1  2  3  4  5

  █ = f(i) rectangles (left sum = upper bound)
  The curve f(x) passes through the tops of the bars
  ░ = gap between rectangle and curve
  Integral = area under curve (between the bounds)
```

**Derivation:** Since $f$ is decreasing, on the interval $[i, i+1]$:

$$
f(i+1) \leq f(x) \leq f(i) \quad \text{for } x \in [i, i+1]
$$

Integrating both sides over $[i, i+1]$:

$$
f(i+1) \leq \int_{i}^{i+1} f(x)\,dx \leq f(i)
$$

Summing from $i = 1$ to $n-1$ for the right inequality, and from $i = 1$ to $n$ for the left, yields the result.

**AI example — harmonic sum bounds:**

With $f(x) = 1/x$:

$$
\ln(n+1) \leq H_n \leq 1 + \ln n
$$

This gives us $H_n = \ln n + \gamma + O(1/n)$ where $\gamma \approx 0.5772$ is the Euler–Mascheroni constant.

**Application:** Bounding the expected number of unique tokens seen after $m$ samples from a vocabulary of size $V$:

$$
E[\text{unique tokens}] = V \left(1 - \left(1 - \frac{1}{V}\right)^m\right) \approx V(1 - e^{-m/V})
$$

The integral test helps bound partial sums that arise in the coverage analysis.

### 7.3 Cauchy–Schwarz and Triangle Inequality for Sums

**Triangle inequality for sums:**

$$
\left| \sum_{i=1}^{n} a_i \right| \leq \sum_{i=1}^{n} |a_i|
$$

This bounds the magnitude of a signed sum by the sum of absolute values. It explains why **gradient norms** after summing over a batch can be bounded:

$$
\left\| \sum_{i=1}^{B} \nabla_\theta \mathcal{L}_i \right\| \leq \sum_{i=1}^{B} \left\| \nabla_\theta \mathcal{L}_i \right\|
$$

**Cauchy–Schwarz inequality for sums:**

$$
\left( \sum_{i=1}^{n} a_i b_i \right)^2 \leq \left( \sum_{i=1}^{n} a_i^2 \right) \left( \sum_{i=1}^{n} b_i^2 \right)
$$

Or in vector notation: $(a \cdot b)^2 \leq \|a\|^2 \|b\|^2$, which means $|\cos \theta| \leq 1$.

**Proof sketch:**

Consider the quadratic $Q(t) = \sum_{i=1}^{n} (a_i + t b_i)^2 \geq 0$ for all $t \in \mathbb{R}$.

Expanding: $Q(t) = \left(\sum b_i^2\right) t^2 + 2\left(\sum a_i b_i\right) t + \left(\sum a_i^2\right) \geq 0$.

Since this quadratic is non-negative for all $t$, its discriminant must be $\leq 0$:

$$
4\left(\sum a_i b_i\right)^2 - 4\left(\sum a_i^2\right)\left(\sum b_i^2\right) \leq 0
$$

which gives the result.

**AI applications:**

| Application                | Bound Used               | Purpose                            |
| -------------------------- | ------------------------ | ---------------------------------- |
| Cosine similarity          | $\|\cos \theta\| \leq 1$ | Validates similarity metric range  |
| Gradient clipping analysis | Triangle inequality      | Bounds effect of clip on sum       |
| Attention weight analysis  | Cauchy–Schwarz           | Bounds QK dot product magnitude    |
| Variance decomposition     | Cauchy–Schwarz           | Jensen-like bounds on expectations |

### 7.4 Stirling's Approximation

Stirling's approximation connects the product (factorial) to continuous functions:

$$
n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n
$$

Taking logarithms:

$$
\ln(n!) = \sum_{k=1}^{n} \ln k \approx n \ln n - n + \frac{1}{2} \ln(2\pi n)
$$

**Derivation via integral test:**

$$
\sum_{k=1}^{n} \ln k \approx \int_{1}^{n} \ln x \, dx = [x \ln x - x]_{1}^{n} = n \ln n - n + 1
$$

The $\sqrt{2\pi n}$ correction comes from more careful analysis (Laplace's method on the Gamma function integral).

**Accuracy table:**

| $n$ | $n!$                   | Stirling               | Relative Error |
| --- | ---------------------- | ---------------------- | -------------- |
| 1   | 1                      | 0.922                  | 7.8%           |
| 5   | 120                    | 118.02                 | 1.6%           |
| 10  | 3628800                | 3598695                | 0.83%          |
| 50  | $3.04 \times 10^{64}$  | $3.04 \times 10^{64}$  | 0.17%          |
| 100 | $9.33 \times 10^{157}$ | $9.32 \times 10^{157}$ | 0.083%         |

The approximation improves rapidly: relative error $\sim 1/(12n)$.

**AI relevance:**

- **Combinatorial arguments:** The number of permutations of $n$ tokens is $n!$; Stirling tells us $\log_2(n!) \approx n \log_2 n$ bits to represent a permutation — relevant for **sorting networks** used in differentiable sorting.
- **Log-likelihood of multinomial:** $\ln \binom{n}{k_1, \ldots, k_V} = \ln n! - \sum_{v} \ln k_v!$; Stirling converts this to entropy: $\approx n H(p)$ where $p_v = k_v/n$.
- **Capacity arguments:** Channel capacity and model capacity bounds often use $\ln \binom{n}{k} \approx n H(k/n)$.

### 7.5 Asymptotic Notation for Sums

When exact sums are intractable, we characterise their **growth rate**:

| Notation       | Meaning                         | Example Sum                      | Growth            |
| -------------- | ------------------------------- | -------------------------------- | ----------------- |
| $O(g(n))$      | Upper bounded by $c \cdot g(n)$ | $\sum_{i=1}^{n} i = O(n^2)$      | At most           |
| $\Omega(g(n))$ | Lower bounded by $c \cdot g(n)$ | $\sum_{i=1}^{n} i = \Omega(n^2)$ | At least          |
| $\Theta(g(n))$ | Tightly bounded                 | $\sum_{i=1}^{n} i = \Theta(n^2)$ | Exactly this rate |
| $o(g(n))$      | Dominated by $g(n)$             | $\sum_{i=1}^{n} 1/i = o(n)$      | Strictly less     |
| $\sim$         | Asymptotically equal            | $H_n \sim \ln n$                 | Ratio → 1         |

**Key asymptotic results for common sums:**

$$
\sum_{i=1}^{n} i^k = \Theta(n^{k+1}) \quad \text{(polynomial sums)}
$$

$$
\sum_{i=0}^{n} r^i = \begin{cases} \Theta(r^n) & \text{if } r > 1 \\ \Theta(n) & \text{if } r = 1 \\ \Theta(1) & \text{if } 0 < r < 1 \end{cases} \quad \text{(geometric sums)}
$$

$$
\sum_{i=1}^{n} \frac{1}{i^p} = \begin{cases} \Theta(\ln n) & \text{if } p = 1 \\ \Theta(1) & \text{if } p > 1 \\ \Theta(n^{1-p}) & \text{if } 0 < p < 1 \end{cases} \quad \text{(harmonic-type sums)}
$$

**AI complexity examples:**

| Operation                        | Sum Form                 | Asymptotic            | Practical Impact                        |
| -------------------------------- | ------------------------ | --------------------- | --------------------------------------- |
| Self-attention (full)            | $\sum_{i=1}^{n} n = n^2$ | $\Theta(n^2)$         | Quadratic in sequence length            |
| Self-attention (sparse, top-$k$) | $\sum_{i=1}^{n} k$       | $\Theta(nk)$          | Linear if $k = O(1)$                    |
| Batch gradient                   | $\sum_{i=1}^{B} O(p)$    | $\Theta(Bp)$          | Linear in batch × params                |
| Transformer layer                | $O(n^2 d + n d^2)$       | depends on $n$ vs $d$ | $n < d$: $O(nd^2)$; $n > d$: $O(n^2 d)$ |
| Vocabulary projection            | $\sum_{v=1}^{V} d$       | $\Theta(Vd)$          | Bottleneck for large vocab              |

---

## 8. Special Summation Patterns in AI

This section catalogues the specific summation patterns that appear repeatedly across AI and machine learning. Recognising these patterns lets you read papers fluently, debug implementations, and reason about computational costs.

### 8.1 Softmax and the Partition Function

The softmax function normalises a vector of logits $z \in \mathbb{R}^K$ into a probability distribution:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

The denominator $Z = \sum_{j=1}^{K} e^{z_j}$ is the **partition function** — a term borrowed from statistical mechanics.

**Anatomy of softmax:**

```
  Input logits:     z = [2.0,  1.0,  0.5,  -1.0]
                         │      │      │      │
  Exponentiate:    eᶻ = [7.39, 2.72, 1.65,  0.37]
                         │      │      │      │
  Partition sum:   Z  = 7.39 + 2.72 + 1.65 + 0.37 = 12.13
                         │      │      │      │
  Normalise:       p  = [0.61, 0.22, 0.14,  0.03]
                         │      │      │      │
  Verify:         ∑p  = 0.61 + 0.22 + 0.14 + 0.03 = 1.00  ✓
```

**Numerical stability — the log-sum-exp trick:**

Direct computation overflows for large logits. The standard trick:

$$
\log Z = \log \sum_{j} e^{z_j} = m + \log \sum_{j} e^{z_j - m} \quad \text{where } m = \max_j z_j
$$

This works because $\sum e^{z_j} = e^m \sum e^{z_j - m}$, and now the largest exponent is $e^0 = 1$.

**Gradient of log-partition:**

A beautiful identity links the partition function to expectations:

$$
\frac{\partial \log Z}{\partial z_i} = \frac{e^{z_i}}{Z} = \text{softmax}(z)_i = p_i
$$

Higher derivatives give **cumulants** of the distribution — the second derivative gives the variance.

**Where partition functions appear in AI:**

| Context                        | Partition Function                           | Size                     |
| ------------------------------ | -------------------------------------------- | ------------------------ |
| Classification (softmax)       | $\sum_{c=1}^{C} e^{z_c}$                     | num_classes              |
| Language modelling             | $\sum_{v=1}^{V} e^{z_v}$                     | vocab_size (~50K–100K)   |
| Contrastive learning (InfoNCE) | $\sum_{j=1}^{N} e^{\text{sim}(q, k_j)/\tau}$ | num_negatives            |
| Energy-based models            | $\int e^{-E(x)} dx$                          | Intractable (continuous) |
| CRF layer (NER)                | $\sum_{\text{all paths}} e^{s(\text{path})}$ | Exponential (use DP)     |

### 8.2 Cross-Entropy Loss as Summation

Cross-entropy loss is a double summation — over the batch and over classes:

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}
$$

For one-hot targets ($y_{ic} = \mathbb{1}[c = c_i^*]$), the inner sum collapses:

$$
\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \log \hat{y}_{i, c_i^*}
$$

**Expansion for a single sample:**

$$
-\log \hat{y}_c = -\log \frac{e^{z_c}}{\sum_j e^{z_j}} = -z_c + \log \sum_j e^{z_j}
$$

So the loss is: (negative logit of correct class) + (log partition function).

**Gradient with respect to logit $z_k$:**

$$
\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k = p_k - \mathbb{1}[k = c^*]
$$

This is the remarkably clean result that makes softmax + cross-entropy so popular: the gradient is simply **(predicted probability − target probability)** for each class.

**Label smoothing variant:**

Instead of one-hot $y$, use smoothed targets: $y_c^{\text{smooth}} = (1 - \epsilon) \cdot \mathbb{1}[c = c^*] + \epsilon / C$.

$$
\mathcal{L}_{\text{smooth}} = -\sum_{c=1}^{C} y_c^{\text{smooth}} \log \hat{y}_c = (1-\epsilon)\mathcal{L}_{\text{CE}} + \epsilon \cdot \frac{1}{C} H_{\text{uniform}}(\hat{y})
$$

The extra term $\frac{\epsilon}{C} \sum_c (-\log \hat{y}_c)$ encourages the model to keep non-target probabilities from collapsing to zero — acting as a **regulariser on confidence**.

### 8.3 Attention as Weighted Sum

The core attention operation is a **weighted sum** of value vectors:

$$
\text{Attention}(Q, K, V)_i = \sum_{j=1}^{n} \alpha_{ij} V_j
$$

where attention weights $\alpha_{ij}$ come from softmax over scaled dot-product scores:

$$
\alpha_{ij} = \frac{\exp(Q_i \cdot K_j / \sqrt{d_k})}{\sum_{\ell=1}^{n} \exp(Q_i \cdot K_\ell / \sqrt{d_k})}
$$

**The full computation graph:**

```
  Q_i ──┐
        ├── Q_i · K_j / √d_k ──→ softmax over j ──→ α_ij
  K_j ──┘                                              │
                                                        ├── ∑_j α_ij · V_j ──→ output_i
  V_j ─────────────────────────────────────────────────┘
```

**Three sums in attention:**

1. **Dot product:** $Q_i \cdot K_j = \sum_{d=1}^{d_k} Q_{i,d} \cdot K_{j,d}$ — sum over the head dimension
2. **Softmax normaliser:** $Z_i = \sum_{j=1}^{n} \exp(Q_i \cdot K_j / \sqrt{d_k})$ — sum over keys
3. **Weighted value:** $o_i = \sum_{j=1}^{n} \alpha_{ij} V_j$ — sum over values

**Multi-head attention adds a fourth sum:**

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O = \sum_{h=1}^{H} \text{head}_h W_h^O
$$

(The concatenation + linear projection is equivalent to a sum of projected heads.)

**Computational cost (counting multiplications):**

| Step                                           | Operation                        | Cost                      |
| ---------------------------------------------- | -------------------------------- | ------------------------- |
| QK^T                                           | $n \times d_k \times n$ per head | $O(n^2 d_k)$              |
| Softmax                                        | Per row of $n$ values            | $O(n^2)$                  |
| Attention × V                                  | $n \times n \times d_v$ per head | $O(n^2 d_v)$              |
| All heads                                      | Multiply by $h$                  | $O(h \cdot n^2 d_k)$      |
| Total (since $h \cdot d_k = d_{\text{model}}$) |                                  | $O(n^2 d_{\text{model}})$ |

### 8.4 Matrix Multiplication as Summation

Every matrix multiplication is a summation in disguise:

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}
$$

This single formula underlies virtually all neural network computation.

**Layer types expressed as matrix multiply:**

| Layer                     | Formula                            | Dimensions                                                             |
| ------------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| Dense/Linear              | $Y = XW + b$                       | $(B \times d_{\text{in}}) \cdot (d_{\text{in}} \times d_{\text{out}})$ |
| Multi-head QKV projection | $Q = XW^Q$                         | $(n \times d) \cdot (d \times d_k h)$                                  |
| Embedding lookup          | $E = \text{one\_hot}(x) \cdot W_E$ | $(B \times V) \cdot (V \times d)$                                      |
| Vocabulary projection     | $Z = H W_E^T$                      | $(B \times d) \cdot (d \times V)$                                      |
| Convolution (im2col)      | $Y = \text{unfold}(X) \cdot W$     | Reshaped to GEMM                                                       |

**FLOPs formula:** Multiplying an $(m \times p)$ matrix by a $(p \times n)$ matrix requires:

- $m \times p \times n$ multiplications
- $m \times (p-1) \times n$ additions
- Total: $\approx 2mpn$ FLOPs

For a transformer with $L$ layers, the total FLOPs per token is approximately $12 L d^2$ (accounting for QKV projections, attention output, and FFN).

### 8.5 Gradient as Sum Over Batch

The empirical risk (training loss) is an average over the batch:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \ell(f_\theta(x_i), y_i)
$$

Its gradient is therefore also an average:

$$
\nabla_\theta \mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \ell(f_\theta(x_i), y_i)
$$

**Key insight:** The gradient is linear in the summation — each sample contributes independently. This enables:

1. **Data parallelism:** Compute per-sample gradients on different GPUs, then sum (all-reduce).
2. **Gradient accumulation:** Sum gradients over micro-batches before updating weights.
3. **Per-sample gradient clipping** (DP-SGD): Clip $\|\nabla_\theta \ell_i\|$ before averaging — required for differential privacy.

**Gradient accumulation pattern:**

```
  Micro-batch 1:  g₁ = ∇L(θ; B₁)     ─┐
  Micro-batch 2:  g₂ = ∇L(θ; B₂)     ─┤
  Micro-batch 3:  g₃ = ∇L(θ; B₃)     ─┼──→  g = (g₁ + g₂ + g₃ + g₄) / 4
  Micro-batch 4:  g₄ = ∇L(θ; B₄)     ─┘
                                              θ ← θ - η · g
```

Effective batch size = $\sum_{k=1}^{A} |B_k|$ where $A$ is the number of accumulation steps.

**Variance of stochastic gradient:**

$$
\text{Var}\left[\frac{1}{B} \sum_{i=1}^{B} g_i\right] = \frac{1}{B^2} \sum_{i=1}^{B} \text{Var}[g_i] = \frac{\sigma^2}{B}
$$

(assuming i.i.d. samples). Doubling the batch size halves the gradient variance — but has **diminishing returns** on convergence speed (linear speedup only up to a critical batch size).

### 8.6 Perplexity as Geometric Mean

Perplexity is the standard evaluation metric for language models:

$$
\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p(w_t \mid w_{<t})\right)
$$

**Decomposing the formula:**

The inner sum $-\frac{1}{T} \sum_{t=1}^{T} \log p(w_t | w_{<t})$ is the **average cross-entropy** per token.

Exponentiating converts it to a **geometric mean** of inverse probabilities:

$$
\text{PPL} = \left(\prod_{t=1}^{T} \frac{1}{p(w_t \mid w_{<t})}\right)^{1/T}
$$

**Intuition:** If PPL = 50, the model is "as confused as if it were choosing uniformly among 50 tokens at each step."

**Worked example:**

```
  Sentence: "The cat sat on the mat"
  Token probabilities: [0.1, 0.05, 0.2, 0.3, 0.1, 0.15]

  Log probs: [-2.303, -2.996, -1.609, -1.204, -2.303, -1.897]
  Sum of log probs: -12.312
  Average: -12.312 / 6 = -2.052
  PPL = exp(2.052) = 7.78

  Interpretation: Model is ~8-way confused on average
  (a perfect model would have PPL = 1)
```

**Properties arising from the summation structure:**

- Perplexity is **multiplicative** over independent segments: if we split text into parts A and B, $\text{PPL}(AB) = \text{PPL}(A)^{|A|/T} \cdot \text{PPL}(B)^{|B|/T}$ (weighted geometric mean).
- **Outlier sensitivity:** A single very-low-probability token (e.g., $p = 10^{-6}$) drastically inflates PPL. This is why models are sometimes evaluated with rare-token filtering.
- **Bits per character (BPC):** $\text{BPC} = \frac{1}{T} \sum_{t} (-\log_2 p(w_t | w_{<t})) = \log_2(\text{PPL})$.

### 8.7 BM25 as Weighted Sum

BM25, the backbone of traditional information retrieval, is a weighted sum over query terms:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot |d| / \text{avgdl})}
$$

**Components as summation:**

- **Outer sum $\sum_{t \in q}$:** Iterates over unique query terms — typically 3–10 terms.
- **IDF (Inverse Document Frequency):** $\text{IDF}(t) = \log \frac{N - n_t + 0.5}{n_t + 0.5}$ where $N$ = total documents, $n_t$ = documents containing term $t$.
- **TF saturation:** The fraction $\frac{f(t,d)(k_1+1)}{f(t,d) + k_1(\ldots)}$ saturates — repeated occurrences have diminishing returns.

**Why BM25 is a pure summation pattern:**

```
  Score = 0
  For each query term t:
      Score += IDF(t) × TF_saturation(t, d)
             ────────   ──────────────────
             weight     adjusted count

  This is a weighted sum: ∑ wₜ · sₜ
```

**Modern hybrid retrieval** combines BM25 with dense retrieval:

$$
\text{score}(q, d) = \alpha \cdot \text{BM25}(q, d) + (1 - \alpha) \cdot \text{sim}_{\text{dense}}(q, d)
$$

Both components are summations. The dense similarity is a dot product $\sum_i q_i d_i$, and BM25 is a weighted sum over terms. The hybrid score is itself a weighted sum — sums all the way down.

---

## 9. Einstein Summation Convention

The Einstein summation convention, introduced by Albert Einstein in his 1916 paper on general relativity, eliminates the explicit $\Sigma$ symbol entirely. Any index that appears **exactly twice** in a single term is implicitly summed over. This convention is now the standard notation in tensor calculus, physics, and — thanks to `torch.einsum` and `np.einsum` — a powerful tool in modern deep learning code.

### 9.1 Motivation: When Sigma Becomes Noise

Consider the matrix multiplication formula:

**Explicit sigma notation:**

$$
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}
$$

**Einstein convention:**

$$
C_{ij} = A_{ik} B_{kj}
$$

The index $k$ appears twice (once in $A$, once in $B$), so it is **summed over** automatically. The indices $i$ and $j$ each appear once, so they are **free indices** that label the output.

**Why drop the sigma?**

In expressions with many summations, the $\Sigma$ symbols obscure the structure. Compare the Ricci curvature tensor transformation:

With sigmas: $R'_{\mu\nu} = \sum_{\alpha} \sum_{\beta} \frac{\partial x^\alpha}{\partial x'^\mu} \frac{\partial x^\beta}{\partial x'^\nu} R_{\alpha\beta}$

Einstein: $R'_{\mu\nu} = \frac{\partial x^\alpha}{\partial x'^\mu} \frac{\partial x^\beta}{\partial x'^\nu} R_{\alpha\beta}$

The structure is identical — the repeated indices $\alpha, \beta$ tell us what's being summed.

### 9.2 The Rules

**Rule 1 — Repeated index = summation (dummy index):**

An index that appears exactly twice in a monomial (a single product term) is summed over all its values.

$$
a_i b_i \equiv \sum_{i=1}^{n} a_i b_i \qquad \text{(dot product)}
$$

**Rule 2 — Free index = unsummed:**

An index that appears exactly once on each side of an equation labels the components. The equation holds for every value of the free index.

$$
c_i = A_{ij} b_j \quad \Leftrightarrow \quad c_i = \sum_j A_{ij} b_j \quad \text{for each } i
$$

**Rule 3 — No index appears three or more times:**

If you see an index three times, the expression is either wrong or needs explicit sigmas.

**Rule 4 — Free indices must match on both sides:**

In $c_i = A_{ij} b_j$, the left side has free index $i$, and the right side also has free index $i$ (after summing out $j$). This is consistent.

**Index classification summary:**

```
  Expression:  C_ij = A_ik B_kj

  Index analysis:
  ┌───────┬────────────┬─────────┬──────────┐
  │ Index │ Appears in │ Count   │ Type     │
  ├───────┼────────────┼─────────┼──────────┤
  │  i    │ C, A       │ 1 each  │ Free     │
  │  j    │ C, B       │ 1 each  │ Free     │
  │  k    │ A, B       │ 2 total │ Dummy    │
  │       │            │ (summed)│          │
  └───────┴────────────┴─────────┴──────────┘
```

### 9.3 Common Operations in Einstein Notation

| Operation        | Standard                       | Einstein          | `einsum` string  |
| ---------------- | ------------------------------ | ----------------- | ---------------- |
| Dot product      | $\sum_i a_i b_i$               | $a_i b_i$         | `'i,i->'`        |
| Matrix-vector    | $\sum_j A_{ij} b_j$            | $A_{ij} b_j$      | `'ij,j->i'`      |
| Matrix multiply  | $\sum_k A_{ik} B_{kj}$         | $A_{ik} B_{kj}$   | `'ik,kj->ij'`    |
| Outer product    | $a_i b_j$ (no sum)             | $a_i b_j$         | `'i,j->ij'`      |
| Trace            | $\sum_i A_{ii}$                | $A_{ii}$          | `'ii->'`         |
| Transpose        | $B_{ji} = A_{ij}$              | $B_{ji} = A_{ij}$ | `'ij->ji'`       |
| Batch matmul     | $\sum_k A_{bik} B_{bkj}$       | $A_{bik} B_{bkj}$ | `'bik,bkj->bij'` |
| Bilinear form    | $\sum_i \sum_j a_i M_{ij} b_j$ | $a_i M_{ij} b_j$  | `'i,ij,j->'`     |
| Hadamard product | $C_{ij} = A_{ij} B_{ij}$       | $A_{ij} B_{ij}$   | `'ij,ij->ij'`    |
| Frobenius norm²  | $\sum_i \sum_j A_{ij}^2$       | $A_{ij} A_{ij}$   | `'ij,ij->'`      |

### 9.4 `torch.einsum` and `np.einsum`

The einsum function in PyTorch and NumPy translates Einstein notation strings directly into tensor operations:

```python
import torch

# Attention scores: Q @ K^T / sqrt(d_k)
# Q: (batch, heads, seq_q, d_k)
# K: (batch, heads, seq_k, d_k)
scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) / (d_k ** 0.5)

# Attention output: scores @ V
# V: (batch, heads, seq_k, d_v)
output = torch.einsum('bhqk,bhkd->bhqd', attention_weights, V)
```

**Reading an einsum string:**

```
  'bhqd,bhkd->bhqk'
   ││││ ││││  ││││
   ││││ ││││  ││└─ k: output has seq_k dimension (free)
   ││││ ││││  │└── q: output has seq_q dimension (free)
   ││││ ││││  └─── h: output has heads dimension (free)
   ││││ │││└────── b: output has batch dimension (free)
   ││││ │││
   ││││ ││└─ d: appears in both inputs → SUMMED (contraction)
   ││││ ││
   │└┘│ └┘── remaining indices determine shapes
   │  │
   Input 1    Input 2
```

**Performance note:** Modern `einsum` implementations use **optimal contraction paths**. The `opt_einsum` library finds the best order to contract indices, sometimes giving 10× speedups for complex multi-tensor operations:

```python
# Without optimisation: might be O(n^5)
# With optimisation: breaks into O(n^3) matmuls
import opt_einsum
path, _ = opt_einsum.contract_path('ijk,jkl,klm->im', A, B, C)
result = opt_einsum.contract('ijk,jkl,klm->im', A, B, C, optimize=path)
```

### 9.5 Tensor Contraction

Einstein summation generalises to **tensor contraction** — the fundamental operation in multilinear algebra.

**Definition:** Given tensors $A$ with indices ${i_1, \ldots, i_p}$ and $B$ with indices ${j_1, \ldots, j_q}$, a contraction identifies a pair of indices (one from each tensor) and sums over them:

$$
C_{(\text{remaining indices})} = \sum_{k} A_{\ldots k \ldots} B_{\ldots k \ldots}
$$

**Contraction reduces rank:** If $A$ has rank $p$ and $B$ has rank $q$, and we contract $r$ pairs of indices, the result has rank $p + q - 2r$.

```
  Rank arithmetic:
  ┌─────────────────────────────────────────────────────┐
  │  Operation     │ Input Ranks │ Contractions │ Output │
  ├─────────────────────────────────────────────────────┤
  │  Dot product   │  1 + 1      │  1           │  0     │
  │  Mat-vec       │  2 + 1      │  1           │  1     │
  │  Mat-mat       │  2 + 2      │  1           │  2     │
  │  Batch matmul  │  3 + 3      │  1           │  3*    │
  │  Trace         │  2          │  1 (self)    │  0     │
  │  Double contraction│ 2 + 2   │  2           │  0     │
  └─────────────────────────────────────────────────────┘
  * Batch indices preserved, not contracted
```

**AI application — attention as tensor contraction:**

Multi-head attention involves a sequence of contractions:

$$
S_{bhij} = Q_{bhid} K_{bhjd} \quad \text{(contract } d\text{)}
$$

$$
O_{bhid} = \alpha_{bhij} V_{bhjd} \quad \text{(contract } j\text{)}
$$

$$
\text{out}_{bid} = O_{bhid} W^O_{hdd'} \quad \text{(contract } h, d\text{)}
$$

Each contraction is one line of `einsum` — the entire multi-head attention is three contractions.

---

## 10. Double Sums and Index Interaction

When two or more indices appear in the same summation, their interaction creates rich mathematical structures. This section explores double (and higher) sums with attention to the patterns that arise in AI.

### 10.1 Row-Major vs Column-Major Intuition

A double sum $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ sums all entries of an $m \times n$ array. The question is: **in what order?**

```
  Row-major (outer=i, inner=j):     Column-major (outer=j, inner=i):

  ┌──────────────────┐              ┌──────────────────┐
  │ →  →  →  →  →  → │  row 1      │ ↓  ↓  ↓  ↓  ↓  ↓ │
  │ →  →  →  →  →  → │  row 2      │ ↓  ↓  ↓  ↓  ↓  ↓ │
  │ →  →  →  →  →  → │  row 3      │ ↓  ↓  ↓  ↓  ↓  ↓ │
  └──────────────────┘              └──────────────────┘

  Process each row left-to-right,    Process each column top-to-bottom,
  then move to next row.             then move to next column.
```

For **finite sums**, both orders give the same result (Fubini's theorem for sums). But the **computational pattern** differs:

- **Row-major** accesses contiguous memory in C/PyTorch (row-contiguous tensors).
- **Column-major** accesses contiguous memory in Fortran/MATLAB.

In AI, choosing the right traversal order affects **cache performance** and **GPU coalescing**.

### 10.2 Diagonal Sums

Summing along diagonals of a matrix isolates elements where indices satisfy $j - i = k$ (constant offset):

**Main diagonal ($i = j$):** The trace.

$$
\text{tr}(A) = \sum_{i=1}^{n} a_{ii}
$$

**Off-diagonals:** $\sum_{i} a_{i, i+k}$ sums the $k$-th super-diagonal (for $k > 0$) or sub-diagonal (for $k < 0$).

**AI connection — relative position in attention:**

In relative positional encodings (e.g., ALiBi, RoPE), the attention bias depends on $|i - j|$:

$$
\text{score}_{ij} = Q_i \cdot K_j + b(|i - j|)
$$

The bias matrix $B$ where $B_{ij} = b(|i - j|)$ is a **Toeplitz matrix** — constant along each diagonal. Summing along diagonals gives the total contribution of each relative distance.

### 10.3 Upper and Lower Triangular Sums

Many AI computations restrict summation to triangular regions:

$$
\sum_{i=1}^{n} \sum_{j=i}^{n} a_{ij} \quad \text{(upper triangle: } j \geq i\text{)}
$$

$$
\sum_{i=1}^{n} \sum_{j=1}^{i} a_{ij} \quad \text{(lower triangle: } j \leq i\text{)}
$$

```
  Upper triangular:         Lower triangular (causal):
  ┌─ ─ ─ ─ ─ ─ ─┐          ┌─ ─ ─ ─ ─ ─ ─┐
  │ ■ ■ ■ ■ ■ ■ │          │ ■ · · · · · │
  │ · ■ ■ ■ ■ ■ │          │ ■ ■ · · · · │
  │ · · ■ ■ ■ ■ │          │ ■ ■ ■ · · · │
  │ · · · ■ ■ ■ │          │ ■ ■ ■ ■ · · │
  │ · · · · ■ ■ │          │ ■ ■ ■ ■ ■ · │
  │ · · · · · ■ │          │ ■ ■ ■ ■ ■ ■ │
  └─ ─ ─ ─ ─ ─ ─┘          └─ ─ ─ ─ ─ ─ ─┘

  ■ = included in sum       Used in causal (autoregressive)
  · = excluded               attention masks
```

**Causal attention** uses lower-triangular masking:

$$
\alpha_{ij} = \begin{cases} \frac{\exp(s_{ij})}{\sum_{k \leq i} \exp(s_{ik})} & \text{if } j \leq i \\ 0 & \text{if } j > i \end{cases}
$$

**Count of elements:**

- Upper or lower triangle: $\sum_{i=1}^{n} (n - i + 1) = \frac{n(n+1)}{2}$ elements
- Strict upper/lower (no diagonal): $\frac{n(n-1)}{2}$ elements
- This halving is why causal attention uses roughly half the FLOPs of bidirectional attention.

### 10.4 Changing Order of Summation

**Finite case (always valid):**

$$
\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{m} a_{ij}
$$

**Triangular region — changing bounds:**

$$
\sum_{i=1}^{n} \sum_{j=i}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{j} a_{ij}
$$

**Proof by picture:**

```
  Original:                    Swapped:
  (fix i, vary j≥i)           (fix j, vary i≤j)

  j →                         j →
  1 2 3 4                     1 2 3 4
  ┌─────────                  ┌─────────
i=1│ ■ ■ ■ ■              i=1│ ■ ■ ■ ■
i=2│   ■ ■ ■              i=2│   ■ ■ ■
i=3│     ■ ■              i=3│     ■ ■
i=4│       ■              i=4│       ■

  Same set of (i,j) pairs — just visited in different order!
```

**General rule for dependent limits:**

$$
\sum_{i=a}^{b} \sum_{j=f(i)}^{g(i)} = \sum_{j=\min f}^{\max g} \sum_{i \in S(j)}
$$

where $S(j) = \{i \in [a,b] : f(i) \leq j \leq g(i)\}$.

**AI application — expectation of loss over data and weights:**

In Bayesian deep learning, we often need to interchange an expectation over parameters and a sum over data:

$$
E_\theta\left[\sum_{i=1}^{n} \ell(x_i; \theta)\right] = \sum_{i=1}^{n} E_\theta[\ell(x_i; \theta)]
$$

This interchange (linearity of expectation) lets us compute per-example expected losses separately.

### 10.5 Cauchy Product of Series

The Cauchy product defines multiplication of two series:

$$
\left(\sum_{i=0}^{\infty} a_i\right) \left(\sum_{j=0}^{\infty} b_j\right) = \sum_{n=0}^{\infty} c_n \quad \text{where } c_n = \sum_{k=0}^{n} a_k b_{n-k}
$$

The inner sum $c_n = \sum_{k=0}^{n} a_k b_{n-k}$ is a **convolution** — the same operation that defines conv layers in CNNs.

**Connection to convolution:**

```
  1D discrete convolution:

  a = [a₀, a₁, a₂, a₃]     (filter/kernel)
  b = [b₀, b₁, b₂, b₃, b₄] (input signal)

  (a * b)_n = ∑_k a_k · b_{n-k}

  This is exactly the Cauchy product — the coefficient of x^n
  in the polynomial product A(x)·B(x)
```

**Mertens' theorem:** If $\sum a_i$ converges absolutely and $\sum b_j$ converges, then the Cauchy product converges to the product of the sums. This guarantees that polynomial multiplication "works" for convergent power series.

**AI applications:**

- **1D convolution** in CNNs/WaveNet is a finite Cauchy product.
- **Polynomial multiplication** via FFT: the Cauchy product of two length-$n$ sequences can be computed in $O(n \log n)$ using FFT, rather than $O(n^2)$ directly.
- **Generating functions** in combinatorics: if $A(x)$ and $B(x)$ are generating functions, their product $A(x) B(x)$ has coefficients given by the Cauchy product — used in counting arguments for data structures.

### 10.6 Vandermonde's Identity and Convolution

Vandermonde's identity is a combinatorial convolution:

$$
\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k} \binom{n}{r-k}
$$

**Proof idea:** Count the number of ways to choose $r$ items from a group of $m + n$ people (divided into group A of $m$ and group B of $n$). Choose $k$ from A and $r - k$ from B, then sum over all valid $k$.

**Connection to generating functions:**

$(1+x)^m (1+x)^n = (1+x)^{m+n}$. Comparing coefficients of $x^r$ on both sides gives Vandermonde's identity — the coefficient on the left is the Cauchy product of binomial coefficients.

**AI relevance:**

- **Feature interaction counting:** If model A has $m$ features and model B has $n$ features, the number of combined feature subsets of size $r$ follows Vandermonde's identity.
- **Mixture models:** When combining categorical distributions, the convolution of their PMFs follows the same pattern.

### 10.7 Abel's Summation by Parts

Abel's summation is the discrete analogue of integration by parts:

$$
\sum_{k=1}^{n} a_k b_k = A_n b_n - \sum_{k=1}^{n-1} A_k (b_{k+1} - b_k)
$$

where $A_k = \sum_{i=1}^{k} a_i$ is the partial sum.

**Compare with integration by parts:**

| Continuous                                      | Discrete                                       |
| ----------------------------------------------- | ---------------------------------------------- |
| $\int u \, dv = uv - \int v \, du$              | $\sum a_k b_k = A_n b_n - \sum A_k \Delta b_k$ |
| $u \leftrightarrow A_k$ (antiderivative of $a$) | $A_k = \sum_{i=1}^k a_i$                       |
| $dv \leftrightarrow b_k$                        | $\Delta b_k = b_{k+1} - b_k$                   |

**When to use Abel's summation:**

The technique is useful when:

1. The partial sums $A_k$ are bounded (even if the $a_k$ oscillate).
2. The sequence $b_k$ is monotonically decreasing to 0.

Under these conditions, Abel's summation shows the original sum converges (**Dirichlet's test**).

**AI application — convergence of weighted gradient sums:**

In exponentially weighted moving averages (used in Adam, EMA models):

$$
\bar{g}_t = \sum_{k=1}^{t} \beta^{t-k} g_k = \sum_{k=1}^{t} \beta^{t-k} g_k
$$

Abel's summation helps analyse the properties of this weighted sum, particularly showing that the contribution of early gradients decays geometrically and the sum converges even if individual gradients $g_k$ oscillate.

---

## 11. Convergence of Infinite Series

Infinite sums — series — appear throughout AI: Taylor expansions for activation functions, geometric series for discounted rewards, power series for matrix functions. Understanding **when** an infinite sum converges (has a finite value) is essential for both theoretical proofs and numerical stability.

### 11.1 Partial Sums and the Definition of Convergence

An infinite series $\sum_{n=1}^{\infty} a_n$ is defined as the limit of its partial sums:

$$
S = \sum_{n=1}^{\infty} a_n = \lim_{N \to \infty} S_N \quad \text{where } S_N = \sum_{n=1}^{N} a_n
$$

The series **converges** if this limit exists and is finite; otherwise it **diverges**.

**Three types of behavior:**

```
  Converges (e.g., ∑ 1/2ⁿ):        Diverges to ∞ (e.g., ∑ 1/n):

  S_N │        ────────── S=2       S_N │              ╱
      │      ╱                          │            ╱
      │    ╱                            │          ╱
      │  ╱                              │        ╱
      │╱                                │      ╱
      └──────────────── N               │    ╱
                                        │  ╱
  Partial sums approach a limit         │╱
                                        └──────────────── N
                                        Partial sums grow without bound

  Oscillates/Diverges (e.g., ∑ (-1)ⁿ):

  S_N │    ·       ·       ·
      │  1 ──┐  1 ──┐  1 ──
      │      │      │
      │  0 ──┘  0 ──┘  0 ──
      └──────────────── N
  Partial sums bounce between 0 and 1
```

**Necessary condition (divergence test):**

If $\sum a_n$ converges, then $\lim_{n \to \infty} a_n = 0$.

**Contrapositive:** If $\lim a_n \neq 0$ or doesn't exist, the series diverges.

> **Warning:** The converse is false! $a_n \to 0$ does NOT guarantee convergence. The harmonic series $\sum 1/n$ has $a_n = 1/n \to 0$ but diverges.

### 11.2 Absolute vs Conditional Convergence

**Absolute convergence:** $\sum |a_n|$ converges.

**Conditional convergence:** $\sum a_n$ converges but $\sum |a_n|$ diverges.

**Key theorem:** Absolute convergence implies convergence. (The converse is false.)

**Example:** The alternating harmonic series $\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n} = \ln 2$ converges conditionally — it converges, but $\sum 1/n$ diverges.

**Why absolute convergence matters in AI:**

- Absolutely convergent series can be **rearranged** in any order without changing the sum. Conditionally convergent series cannot (Riemann's rearrangement theorem — you can rearrange a conditionally convergent series to converge to any value!).
- **Parallel computation** of sums (across GPUs) implicitly rearranges terms. If the series is only conditionally convergent, different summation orders give different results — a source of **numerical non-determinism**.
- Most sums in AI (loss over data, gradient components) involve non-negative terms, so convergence is always absolute.

### 11.3 Convergence Tests Summary

| Test                   | Statement                                                                              | Best For                 |
| ---------------------- | -------------------------------------------------------------------------------------- | ------------------------ | ------------------------------------------------------ | ------------------------ |
| **Divergence**         | If $a_n \not\to 0$, diverges                                                           | Quick rejection          |
| **Comparison**         | If $0 \leq a_n \leq b_n$ and $\sum b_n$ converges, so does $\sum a_n$                  | Bounding by known series |
| **Limit comparison**   | If $\lim a_n/b_n = L > 0$, both converge or both diverge                               | Same rate detection      |
| **Ratio**              | If $\lim                                                                               | a\_{n+1}/a_n             | = r$: converges if $r < 1$, diverges if $r > 1$        | Factorials, exponentials |
| **Root**               | If $\lim                                                                               | a_n                      | ^{1/n} = r$: converges if $r < 1$, diverges if $r > 1$ | $n$-th powers            |
| **Integral**           | $\sum f(n)$ and $\int f(x) dx$ converge/diverge together (for positive decreasing $f$) | Continuous analogues     |
| **Alternating series** | If $                                                                                   | a_n                      | $ decreases to 0, $\sum (-1)^n a_n$ converges          | Sign-alternating terms   |
| **$p$-series**         | $\sum 1/n^p$ converges iff $p > 1$                                                     | Polynomial decay         |

**Ratio test in detail (most commonly used):**

$$
L = \lim_{n \to \infty} \left| \frac{a_{n+1}}{a_n} \right|
$$

- $L < 1$: Absolutely convergent. The terms decay at least geometrically.
- $L > 1$: Divergent. The terms grow.
- $L = 1$: **Inconclusive**. Need another test.

**AI example — convergence of Taylor series for $e^x$:**

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}, \quad \frac{a_{n+1}}{a_n} = \frac{|x|}{n+1} \to 0 \text{ as } n \to \infty
$$

Since $L = 0 < 1$, the series converges for **all** $x$. This is why $e^x$ (and hence softmax) is well-defined everywhere.

### 11.4 Power Series and Radius of Convergence

A power series $\sum_{n=0}^{\infty} c_n (x - a)^n$ converges in some interval centered at $a$:

$$
R = \frac{1}{\limsup_{n \to \infty} |c_n|^{1/n}} \quad \text{(radius of convergence)}
$$

The series converges absolutely for $|x - a| < R$ and diverges for $|x - a| > R$.

**Key power series in AI:**

| Function            | Power Series                                          | Radius       |
| ------------------- | ----------------------------------------------------- | ------------ |
| $e^x$               | $\sum \frac{x^n}{n!}$                                 | $R = \infty$ |
| $\ln(1+x)$          | $\sum \frac{(-1)^{n+1} x^n}{n}$                       | $R = 1$      |
| $\frac{1}{1-x}$     | $\sum x^n$                                            | $R = 1$      |
| $\tanh(x)$          | $x - \frac{x^3}{3} + \frac{2x^5}{15} - \cdots$        | $R = \pi/2$  |
| GELU $(x \Phi(x))$  | Complicated                                           | $R = \infty$ |
| $\text{sigmoid}(x)$ | $\frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + \cdots$ | $R = \pi$    |

**Why radius of convergence matters:**

- The Taylor approximation of $\tanh$ is only valid for $|x| < \pi/2$ — outside this range, the polynomial approximation diverges wildly. This matters for **hardware implementations** that approximate activation functions with polynomials.
- The $\ln(1+x)$ series only converges for $|x| \leq 1$ — so `log1p(x)` is numerically implemented differently for large $x$.

### 11.5 Convergence of Training — An Analogy

While not an infinite series in the mathematical sense, training a neural network produces a sequence of losses $\{\mathcal{L}_t\}_{t=0}^{\infty}$ that we hope converges.

**Connections to series convergence:**

| Series Concept          | Training Analogue                  |
| ----------------------- | ---------------------------------- |
| Partial sum $S_N$       | Cumulative loss or running average |
| $a_n \to 0$ necessary   | Loss improvements must shrink      |
| Absolute convergence    | All parameter dimensions converge  |
| Conditional convergence | Loss oscillates but trends down    |
| Radius of convergence   | Learning rate stability region     |
| Geometric decay $r^n$   | Exponential learning rate schedule |

**Learning rate as a convergence factor:**

For SGD with learning rate $\eta$ on a quadratic loss surface with Hessian eigenvalue $\lambda$:

$$
\text{error}_t \propto (1 - \eta \lambda)^t
$$

This is a geometric series ratio. Convergence requires $|1 - \eta \lambda| < 1$, i.e., $\eta < 2/\lambda$.

For the maximum eigenvalue $\lambda_{\max}$ of the Hessian, the critical learning rate is:

$$
\eta_{\text{crit}} = \frac{2}{\lambda_{\max}}
$$

Beyond this, training diverges — exactly like a geometric series with $|r| > 1$.

---

## 12. Summation in Probability and Statistics

Summation is the **language of probability theory** for discrete random variables. Every key concept — expectation, variance, entropy — is defined as a sum.

### 12.1 Expectation as Weighted Sum

The expected value of a discrete random variable $X$ with PMF $p(x)$:

$$
E[X] = \sum_{x \in \mathcal{X}} x \cdot p(x)
$$

More generally, for any function $g$:

$$
E[g(X)] = \sum_{x \in \mathcal{X}} g(x) \cdot p(x)
$$

**This is a weighted sum** where the weights are probabilities (summing to 1):

```
  Expectation = weighted average

  Values:      x₁     x₂     x₃     x₄
  Probabilities: p₁     p₂     p₃     p₄
                  │       │       │       │
                  ▼       ▼       ▼       ▼
  E[X] = ────── x₁p₁ + x₂p₂ + x₃p₃ + x₄p₄ ──────

  Constraint: p₁ + p₂ + p₃ + p₄ = 1
```

**Key properties (all following from linearity of summation):**

| Property       | Formula                            | Summation Origin                        |
| -------------- | ---------------------------------- | --------------------------------------- |
| Linearity      | $E[aX + bY] = aE[X] + bE[Y]$       | $\sum (ax + by)p = a\sum xp + b\sum yp$ |
| Constant       | $E[c] = c$                         | $\sum c \cdot p(x) = c \sum p(x) = c$   |
| Non-negativity | $X \geq 0 \Rightarrow E[X] \geq 0$ | Sum of non-negative terms               |
| Indicator      | $E[\mathbb{1}_A] = P(A)$           | $\sum_{x \in A} p(x) = P(A)$            |

**AI example — expected loss:**

The training objective $E_{(x,y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]$ becomes, for a finite dataset:

$$
\hat{E}[\ell] = \frac{1}{n} \sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)
$$

This is the empirical mean — a sum with equal weights $1/n$ — that approximates the true expectation.

### 12.2 Variance and Higher Moments

**Variance** measures spread around the mean:

$$
\text{Var}(X) = E[(X - \mu)^2] = \sum_x (x - \mu)^2 p(x)
$$

**Computational formula** (more numerically stable in some cases):

$$
\text{Var}(X) = E[X^2] - (E[X])^2 = \left(\sum_x x^2 p(x)\right) - \left(\sum_x x p(x)\right)^2
$$

**Proof:**

$$
\text{Var}(X) = E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2
$$

**Variance of a sum (independent variables):**

$$
\text{Var}\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \text{Var}(X_i) + 2 \sum_{i<j} \text{Cov}(X_i, X_j)
$$

For independent variables, covariances vanish: $\text{Var}(\sum X_i) = \sum \text{Var}(X_i)$.

**AI application — gradient variance:**

The variance of the stochastic gradient with batch size $B$:

$$
\text{Var}[\hat{g}] = \text{Var}\left[\frac{1}{B} \sum_{i=1}^{B} g_i\right] = \frac{1}{B^2} \sum_{i=1}^{B} \text{Var}[g_i] = \frac{\sigma^2}{B}
$$

This $1/B$ scaling is why **larger batches reduce gradient noise** but with diminishing returns on convergence speed.

### 12.3 Entropy as Expected Surprise

Shannon entropy measures the average "surprise" or "information content" of a distribution:

$$
H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = E[-\log p(X)]
$$

**Decomposition of cross-entropy:**

$$
H(p, q) = -\sum_x p(x) \log q(x) = H(p) + D_{\text{KL}}(p \| q)
$$

where $D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \geq 0$.

**AI significance:**

- Training with cross-entropy loss **minimises** $H(p, q)$ where $p$ is the true distribution and $q$ is the model. Since $H(p)$ is fixed, this is equivalent to minimising $D_{\text{KL}}(p \| q)$.
- Maximum entropy occurs at the uniform distribution: $H_{\max} = \log |\mathcal{X}|$.
- **Temperature scaling** in softmax: $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$. As $T \to \infty$, entropy increases toward $\log K$; as $T \to 0$, entropy decreases toward 0 (argmax).

**Entropy of common distributions:**

| Distribution   | PMF             | Entropy                                |
| -------------- | --------------- | -------------------------------------- |
| Bernoulli($p$) | $p, (1-p)$      | $-p\log p - (1-p)\log(1-p)$            |
| Uniform($K$)   | $1/K$ each      | $\log K$                               |
| Geometric($p$) | $(1-p)^{n-1} p$ | $\frac{-(1-p)\log(1-p) - p \log p}{p}$ |

### 12.4 Law of Total Expectation and Total Variance

**Law of total expectation (tower property):**

$$
E[X] = E[E[X \mid Y]] = \sum_y E[X \mid Y = y] P(Y = y)
$$

This is a **double summation** — the inner sum computes $E[X \mid Y = y]$ for each $y$, and the outer sum averages over $y$.

**Proof:**

$$
\sum_y E[X \mid Y=y] P(Y=y) = \sum_y \left(\sum_x x \, P(X=x \mid Y=y)\right) P(Y=y)
$$

$$
= \sum_y \sum_x x \, P(X=x, Y=y) = \sum_x x \sum_y P(X=x, Y=y) = \sum_x x \, P(X=x) = E[X]
$$

**AI application — mixture models:**

For a Gaussian mixture model $p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \sigma_k^2)$:

$$
E[X] = \sum_{k=1}^{K} \pi_k \mu_k
$$

**Law of total variance:**

$$
\text{Var}(X) = E[\text{Var}(X \mid Y)] + \text{Var}(E[X \mid Y])
$$

$= \text{(average within-group variance)} + \text{(between-group variance)}$

This decomposition appears in **ANOVA**, **random effects models**, and the **bias-variance decomposition** of ML models.

### 12.5 Inclusion-Exclusion Principle

For the union of events:

$$
P\left(\bigcup_{i=1}^{n} A_i\right) = \sum_{k=1}^{n} (-1)^{k+1} \sum_{|S|=k} P\left(\bigcap_{i \in S} A_i\right)
$$

Expanded for $n = 3$:

$$
P(A_1 \cup A_2 \cup A_3) = P(A_1) + P(A_2) + P(A_3) - P(A_1 \cap A_2) - P(A_1 \cap A_3) - P(A_2 \cap A_3) + P(A_1 \cap A_2 \cap A_3)
$$

```
  Venn diagram (n=3):

         A₁
        ╱  ╲
       ╱ +1 ╲
      ╱  ╱─╲  ╲
     │  │-1 │  │
  A₂─── │+1│ ───A₃
      ╲  ╲─╱  ╱
       ╲ -1 ╱
        ╲  ╱

  Each region counted: singles (+), pairs (-), triple (+)
  Alternating signs prevent over-counting
```

**The sum has $2^n - 1$ terms** — exponential in the number of sets.

**AI applications:**

- **Deduplication analysis:** Probability that at least one near-duplicate exists in a dataset.
- **Hash collision bounds:** $P(\text{any collision among } n \text{ items in } m \text{ buckets})$.
- **Union bound (Boole's inequality):** The simpler upper bound $P(\bigcup A_i) \leq \sum P(A_i)$ is often sufficient and avoids the exponential blowup.
- **Coverage testing:** Probability that a test suite covers at least one of several code paths.

**Bonferroni inequalities** truncate the inclusion-exclusion sum:

- Truncating after odd terms gives an **upper bound**.
- Truncating after even terms gives a **lower bound**.

This provides a practical way to get bounds without computing all $2^n$ terms.

---

## 13. Notation Variants and Conventions Across Fields

Different communities use different summation conventions. Recognising these variants prevents confusion when reading papers from different fields.

### 13.1 Computer Science / Programming

| Convention                    | Meaning                                           | Notes                                |
| ----------------------------- | ------------------------------------------------- | ------------------------------------ |
| 0-indexed: $\sum_{i=0}^{n-1}$ | Sum over $n$ elements                             | Default in Python, C, most languages |
| `sum(list)`                   | $\sum_{i=0}^{\text{len}-1} \text{list}[i]$        | No explicit indices                  |
| `reduce(+, arr)`              | Left fold: $(\cdots((a_0 + a_1) + a_2) + \cdots)$ | Associativity matters for floats     |
| Array slice: `a[i:j]`         | Elements $a_i, a_{i+1}, \ldots, a_{j-1}$          | Exclusive upper bound                |

### 13.2 Physics

| Convention          | Meaning                           | Notes                               |
| ------------------- | --------------------------------- | ----------------------------------- |
| Einstein summation  | Repeated indices summed           | No $\Sigma$ written                 |
| Upper/lower indices | $a^i b_i = \sum_i a^i b_i$        | Contravariant/covariant distinction |
| Metric contraction  | $a_i = g_{ij} a^j$                | Lowering with metric tensor         |
| $\partial_\mu$      | $\frac{\partial}{\partial x^\mu}$ | 4-vector gradient                   |

### 13.3 Statistics

| Convention                      | Meaning                             | Notes                    |
| ------------------------------- | ----------------------------------- | ------------------------ |
| $\bar{x} = \frac{1}{n}\sum x_i$ | Sample mean                         | Indices often omitted    |
| $\sum_i$ with no bounds         | Sum over all observations           | Context determines range |
| $E_p[f(X)]$                     | $\sum_x f(x) p(x)$                  | Expectation notation     |
| $\sum_{i<j}$                    | $\sum_{i=1}^{n-1} \sum_{j=i+1}^{n}$ | All unordered pairs      |

### 13.4 LaTeX / Typesetting

```latex
% Inline: limits on side          Display: limits above/below
$\sum_{i=1}^{n} a_i$              $$\sum_{i=1}^{n} a_i$$

% Product
$\prod_{j=1}^{m} b_j$            $$\prod_{j=1}^{m} b_j$$

% Multiple indices
$\sum_{\substack{i=1\\i \neq k}}^{n}$    % Exclusion

% Sum over a set
$\sum_{x \in S}$                  % Index set notation
```

### 13.5 Notation Translation Table

| Your Field Says  | Mathematicians Write             | Einstein Writes | Python Says                |
| ---------------- | -------------------------------- | --------------- | -------------------------- |
| Dot product      | $\sum_{i=1}^{n} a_i b_i$         | $a_i b_i$       | `a @ b` or `np.dot(a,b)`   |
| Matrix multiply  | $\sum_k A_{ik} B_{kj}$           | $A_{ik} B_{kj}$ | `A @ B`                    |
| Weighted average | $\sum_i w_i x_i$                 | $w_i x_i$       | `np.average(x, weights=w)` |
| Trace            | $\sum_i A_{ii}$                  | $A_{ii}$        | `np.trace(A)`              |
| Batch mean       | $\frac{1}{B}\sum_{i=1}^{B} x_i$  | —               | `x.mean(dim=0)`            |
| Softmax          | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | —               | `F.softmax(z, dim=-1)`     |

---

## 14. Common Mistakes and Pitfalls

Summation notation is deceptively simple, but subtle errors can cause bugs in proofs, papers, and code. This section catalogues the most frequent mistakes.

### 14.1 The Top 10 Summation Mistakes

| #   | Mistake                                 | Wrong                                              | Correct                                       | AI Consequence                |
| --- | --------------------------------------- | -------------------------------------------------- | --------------------------------------------- | ----------------------------- |
| 1   | **Index scope leak**                    | Using $i$ outside $\sum_i$                         | $i$ is a bound (dummy) variable               | Undefined variable bugs       |
| 2   | **Off-by-one bounds**                   | $\sum_{i=0}^{n}$ vs $\sum_{i=1}^{n}$               | Count elements: $n+1$ vs $n$                  | Dimension mismatch errors     |
| 3   | **Distributing $\Sigma$ over products** | $\sum a_i b_i = (\sum a_i)(\sum b_i)$              | $\sum a_i b_i \neq (\sum a_i)(\sum b_i)$      | Wrong loss gradients          |
| 4   | **Distributing $\Pi$ over sums**        | $\prod(a_i + b_i) = \prod a_i + \prod b_i$         | Product distributes over factors, not addends | Wrong likelihood              |
| 5   | **Swapping $\sum$ and non-linear $f$**  | $f(\sum a_i) = \sum f(a_i)$                        | Only true for linear $f$                      | Jensen's inequality violation |
| 6   | **Forgetting empty cases**              | Assuming $n \geq 1$                                | Empty sum = 0, empty product = 1              | Edge case crashes             |
| 7   | **Wrong $\Sigma$ ↔ $\Pi$ conversion**   | $\log \sum = \sum \log$                            | $\log \prod = \sum \log$, NOT $\log \sum$     | log-sum-exp vs sum-of-logs    |
| 8   | **Index collision**                     | $\sum_i a_i \sum_i b_i$                            | $\sum_i a_i \sum_j b_j$                       | Ambiguous scope               |
| 9   | **Telescoping failure**                 | Cancelling wrong terms                             | Write out the first few terms explicitly      | Wrong simplification          |
| 10  | **Infinite sum truncation**             | Treating $\sum_{n=0}^{N}$ as $\sum_{n=0}^{\infty}$ | Bound the tail: $\sum_{n>N}$                  | Approximation error           |

### 14.2 Detailed Examples

**Mistake 3 in detail — sum of products ≠ product of sums:**

$$
\sum_{i=1}^{2} a_i b_i = a_1 b_1 + a_2 b_2
$$

$$
\left(\sum_{i=1}^{2} a_i\right)\left(\sum_{i=1}^{2} b_i\right) = a_1 b_1 + a_1 b_2 + a_2 b_1 + a_2 b_2
$$

The product of sums has **cross terms** ($a_1 b_2, a_2 b_1$). The correct relationship:

$$
\left(\sum_i a_i\right)\left(\sum_j b_j\right) = \sum_i \sum_j a_i b_j
$$

This is a **double sum** (outer product), not a single sum (dot product).

**Mistake 5 in detail — Jensen's inequality:**

For convex $f$: $f\left(\sum_i p_i x_i\right) \leq \sum_i p_i f(x_i)$ (where $\sum p_i = 1$, $p_i \geq 0$).

For concave $f$: the inequality reverses.

Since $\log$ is concave:

$$
\log\left(\sum_i p_i x_i\right) \geq \sum_i p_i \log(x_i)
$$

This is the basis of the **ELBO** (Evidence Lower Bound) in variational inference:

$$
\log p(x) = \log \sum_z p(x,z) = \log \sum_z q(z) \frac{p(x,z)}{q(z)} \geq \sum_z q(z) \log \frac{p(x,z)}{q(z)} = \text{ELBO}
$$

**Mistake 7 in detail — log of sum vs sum of logs:**

```
  ╔══════════════════════════════════════════════════╗
  ║  CRITICAL DISTINCTION:                           ║
  ║                                                  ║
  ║  log(∑ aᵢ)  ≠  ∑ log(aᵢ)     ← WRONG to swap   ║
  ║                                                  ║
  ║  log(∏ aᵢ)  =  ∑ log(aᵢ)     ← CORRECT          ║
  ║                                                  ║
  ║  Logs convert PRODUCTS to SUMS, not SUMS to SUMS ║
  ╠══════════════════════════════════════════════════╣
  ║  NLL loss = -log p(data)                         ║
  ║          = -log ∏ p(xᵢ)     (for i.i.d. data)   ║
  ║          = -∑ log p(xᵢ)     ← the log-sum trick ║
  ║                                                  ║
  ║  BUT:  -log ∑ p(xᵢ)  is DIFFERENT — this is     ║
  ║        the log of a mixture, not a product.      ║
  ╚══════════════════════════════════════════════════╝
```

### 14.3 Numerical Pitfalls

**Catastrophic cancellation:**

When summing numbers with alternating signs or vastly different magnitudes, floating-point errors accumulate:

$$
\sum_{i=1}^{n} a_i \quad \text{where some } a_i \gg 0 \text{ and some } a_i \ll 0
$$

**Solutions:**

- **Kahan summation:** Tracks the running compensation term; reduces error from $O(n \epsilon)$ to $O(\epsilon)$.
- **Pairwise summation:** Recursively splits the array and sums pairs; reduces error to $O(\epsilon \log n)$. (This is what NumPy uses.)
- **Sort-then-sum:** Sum smallest-magnitude terms first.

**Overflow/underflow in products:**

$$
\prod_{i=1}^{n} p_i \to 0 \quad \text{(underflow for large } n \text{)}
$$

**Solution:** Always use log-space: $\sum \log p_i$ instead of $\log \prod p_i$.

---

## 15. Exercises and Practice Problems

### Exercise Set 1: Basic Manipulation

**E1.1** Expand and compute: $\sum_{k=1}^{5} (2k - 1)$

**E1.2** Write in sigma notation: $3 + 6 + 9 + 12 + \cdots + 3n$

**E1.3** Expand and compute: $\prod_{j=2}^{5} \frac{j}{j-1}$

**E1.4** Prove: $\sum_{i=1}^{n} (a_i - a_{i-1}) = a_n - a_0$ (telescoping)

### Exercise Set 2: Algebraic Properties

**E2.1** Show that $\sum_{i=1}^{n} (x_i - \bar{x}) = 0$ where $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$.

**E2.2** Prove: $\sum_{i=1}^{n} (x_i - \bar{x})^2 = \sum_{i=1}^{n} x_i^2 - n\bar{x}^2$.

**E2.3** Verify Gauss's trick: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ for $n = 100$.

**E2.4** Show that $\prod_{i=1}^{n} e^{a_i} = e^{\sum_{i=1}^{n} a_i}$.

### Exercise Set 3: Summation Formulas

**E3.1** Derive $\sum_{i=0}^{n-1} r^i = \frac{1 - r^n}{1 - r}$ by multiplying both sides by $(1-r)$.

**E3.2** Compute $\sum_{k=0}^{\infty} \frac{1}{3^k}$.

**E3.3** Find a closed form for $\sum_{k=1}^{n} k \cdot 2^k$ using the derivative trick on the geometric series.

**E3.4** Compute $\sum_{k=1}^{n} \frac{1}{k(k+1)}$ using partial fractions and telescoping.

### Exercise Set 4: Double Sums

**E4.1** Compute $\sum_{i=1}^{3} \sum_{j=1}^{3} ij$ two ways (row-first and column-first).

**E4.2** Change the order of summation: $\sum_{i=1}^{n} \sum_{j=i}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{j} a_{ij}$.

**E4.3** Show that $\sum_{i=1}^{n} \sum_{j=1}^{n} a_i b_j = \left(\sum_{i=1}^{n} a_i\right)\left(\sum_{j=1}^{n} b_j\right)$.

### Exercise Set 5: Einstein Notation

**E5.1** Convert to standard sigma notation: $T_{ij} = A_{ik} B_{kj} + C_{ij}$.

**E5.2** Write the `einsum` string for computing the trace of a matrix product: $\text{tr}(AB) = A_{ij} B_{ji}$.

**E5.3** What does `'bhqd,bhkd->bhqk'` compute? Identify free and dummy indices.

### Exercise Set 6: AI Applications

**E6.1** Show that softmax probabilities sum to 1: $\sum_{i=1}^{K} \frac{e^{z_i}}{\sum_j e^{z_j}} = 1$.

**E6.2** Derive the gradient $\frac{\partial}{\partial z_k}\left(-\log \frac{e^{z_c}}{\sum_j e^{z_j}}\right) = \text{softmax}(z)_k - \mathbb{1}[k=c]$.

**E6.3** Compute the perplexity of a model that assigns probability 0.1 to each of 5 tokens in a sequence.

**E6.4** Show that the gradient of the mean loss $\frac{1}{B}\sum_i \ell_i$ equals the mean of the gradients $\frac{1}{B}\sum_i \nabla \ell_i$.

### Exercise Set 7: Convergence

**E7.1** Apply the ratio test to determine convergence of $\sum_{n=1}^{\infty} \frac{n^2}{2^n}$.

**E7.2** Determine whether $\sum_{n=1}^{\infty} \frac{1}{n^{3/2}}$ converges.

**E7.3** Find the radius of convergence of $\sum_{n=0}^{\infty} \frac{x^n}{n^2 + 1}$.

### Exercise Set 8: Probability

**E8.1** For a fair die, compute $E[X]$ and $\text{Var}(X)$ from the definitions using summation.

**E8.2** Verify that $H(\text{Bernoulli}(1/2)) = \log 2 \approx 0.693$ nats (= 1 bit).

**E8.3** Use the law of total expectation to find $E[X]$ where $X \mid N=n \sim \text{Uniform}\{1, \ldots, n\}$ and $N \sim \text{Geometric}(1/2)$.

---

## 16. Why This Matters: The Summation Lens on AI

Summation and product notation are not just mathematical conveniences — they are the **computational atoms** from which all of AI is built.

### The Impact Map

| AI Concept         | Core Summation Pattern                       | Without Understanding It         |
| ------------------ | -------------------------------------------- | -------------------------------- |
| Forward pass       | Matrix multiply $\sum_k W_{ik} x_k$          | Cannot read model architectures  |
| Backpropagation    | Chain rule + sum over paths                  | Cannot derive or debug gradients |
| Loss functions     | Cross-entropy $-\sum y \log \hat{y}$         | Cannot design new objectives     |
| Optimisers         | $\theta_t = \theta_{t-1} - \eta \sum$        | Cannot tune learning rates       |
| Attention          | Weighted sum $\sum \alpha_j V_j$             | Cannot understand transformers   |
| Evaluation         | Perplexity = $\exp(-\frac{1}{T}\sum \log p)$ | Cannot interpret model quality   |
| Probability        | $E[X] = \sum x \, p(x)$                      | Cannot reason about uncertainty  |
| Regularisation     | $\lambda \sum \theta_i^2$ (L2)               | Cannot balance fit vs complexity |
| Batch processing   | $\frac{1}{B} \sum \ell_i$                    | Cannot scale training            |
| Information theory | $H = -\sum p \log p$                         | Cannot measure information       |

### The Three Skills This Section Builds

```
  ┌─────────────────────────────────────────────────────┐
  │           Fluency with Summation Notation            │
  │                                                     │
  │  1. READ:  Parse ∑ and ∏ expressions in papers      │
  │            Identify index ranges and dummy vars      │
  │            Recognise standard patterns               │
  │                                                     │
  │  2. MANIPULATE:  Apply algebraic properties          │
  │                  Change summation order               │
  │                  Use telescoping and cancellation     │
  │                  Convert between ∑, ∏, and log       │
  │                                                     │
  │  3. CONNECT:  Map formulas to code                   │
  │               Map code to formulas                   │
  │               Estimate computational cost            │
  │               Debug via mathematical reasoning       │
  └─────────────────────────────────────────────────────┘
```

---

## Conceptual Bridge: From Summation to Einstein Summation and Index Notation

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                      CONCEPTUAL BRIDGE                          │
  │                                                                  │
  │  This Section (04)              Next Section (05)                │
  │  Summation & Products           Einstein Summation & Tensors     │
  │                                                                  │
  │  ∑ᵢ aᵢbᵢ          ──────→     aᵢbᵢ (implicit sum)             │
  │  ∑ₖ AᵢₖBₖⱼ        ──────→     AᵢₖBₖⱼ (matrix multiply)       │
  │  Scalar indices     ──────→     Tensor indices (rank ≥ 2)       │
  │  Finite sums        ──────→     Tensor contraction              │
  │  Index manipulation ──────→     Index gymnastics                 │
  │  torch.sum()        ──────→     torch.einsum()                  │
  │                                                                  │
  │  Key transition:                                                 │
  │  "Explicit sigma" → "Implicit contraction" → "Tensor networks"  │
  │                                                                  │
  │  You now have:                                                   │
  │  ✓ Fluency with ∑ and ∏ notation                                │
  │  ✓ Algebraic manipulation tools                                  │
  │  ✓ Closed-form formulas for standard sums                       │
  │  ✓ Convergence analysis for infinite series                     │
  │  ✓ Recognition of AI summation patterns                         │
  │                                                                  │
  │  Next you'll learn:                                              │
  │  → Einstein convention as shorthand for repeated-index sums      │
  │  → Tensor rank, contraction, and index notation                  │
  │  → Covariant vs contravariant indices                            │
  │  → Advanced einsum patterns for multi-dimensional operations     │
  │  → Tensor networks and their connection to neural architectures  │
  └──────────────────────────────────────────────────────────────────┘
```

---

## References and Further Reading

1. **Knuth, D. E.** (1997). _The Art of Computer Programming, Vol. 1: Fundamental Algorithms_, 3rd ed. Addison-Wesley. — Definitive treatment of summation techniques (Chapter 1.2.3).

2. **Graham, R. L., Knuth, D. E., & Patashnik, O.** (1994). _Concrete Mathematics_, 2nd ed. Addison-Wesley. — The best resource for summation methods, with hundreds of worked examples.

3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). _Deep Learning_. MIT Press. — See Chapter 2 for notation conventions and Chapter 3 for probability/information theory.

4. **Vaswani, A., et al.** (2017). "Attention Is All You Need." _NeurIPS_. — The transformer paper that made $\sum \alpha_{ij} V_j$ the central operation in modern AI.

5. **Einstein, A.** (1916). "Die Grundlage der allgemeinen Relativitätstheorie." _Annalen der Physik_. — Introduction of the summation convention.

6. **Apostol, T. M.** (1974). _Mathematical Analysis_, 2nd ed. Addison-Wesley. — Rigorous treatment of series convergence.

7. **Robertson, S. E., & Zaragoza, H.** (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." _Foundations and Trends in Information Retrieval_. — BM25 derivation and analysis.

---

_Navigation: [← 03 Functions and Mappings](../03-Functions-and-Mappings/notes.md) | [↑ Mathematical Foundations](../README.md) | [05 Einstein Summation and Index Notation →](../05-Einstein-Summation-and-Index-Notation/notes.md)_
