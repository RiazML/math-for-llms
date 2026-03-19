# Functions and Mappings

[← Sets and Logic](../02-Sets-and-Logic/notes.md) | [Next: Summation and Product Notation →](../04-Summation-and-Product-Notation/notes.md)

> _"A neural network is a composition of functions. Understanding functions — what they preserve, what they destroy, how they compose, where they break — is understanding neural networks at the deepest possible level."_

## Overview

A function is the most fundamental concept in mathematics after sets. It is a rule that assigns to each input exactly one output. Every mathematical operation you have ever performed — adding, multiplying, differentiating, integrating — is a function. Every layer in a neural network is a function. The entire model from tokens in to probabilities out is a single (very complex) function. Training is optimisation over a space of functions. Inference is function evaluation.

This chapter develops function theory from first principles through to the frontier concepts needed to understand modern LLMs and AI systems at a mathematical level. We cover the formal definition, types (injective, surjective, bijective), composition, inverses, special classes (linear, affine, activation functions, convex, Lipschitz), higher-order functions and functionals, continuity, differentiability, measure-theoretic foundations, the systematic role of functions in machine learning, and the category-theoretic perspective that unifies it all.

Every concept is connected to how it appears in real neural network computation, with emphasis on transformer architectures, attention mechanisms, and the training pipeline.

## Prerequisites

- Set theory fundamentals (Chapter 02: Sets and Logic)
- Basic algebra and coordinate geometry
- Familiarity with vectors and matrices (helpful but not required yet)
- Python/NumPy basics for code examples

## Companion Notebooks

| Notebook                           | Description                                                                                              |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive code: function visualisation, composition chains, activation functions, Lipschitz demos       |
| [exercises.ipynb](exercises.ipynb) | Practice problems: injectivity proofs, composition calculations, inverse construction, approximation      |

## Learning Objectives

After completing this section, you will:

- Understand functions formally as subsets of Cartesian products satisfying totality and uniqueness
- Distinguish domain, codomain, range, image, and preimage — and know why each matters for AI
- Classify functions as injective, surjective, bijective, monotone, or periodic
- Compose functions and understand associativity, non-commutativity, and the identity function
- Construct inverses (left, right, two-sided, pseudo-inverse) and connect invertibility to information preservation
- Master special function classes: linear, affine, multilinear, polynomial, activation functions, convex, Lipschitz
- Work with higher-order functions, functionals, operators, currying, and function spaces
- Understand continuity (pointwise, uniform) and its role in gradient-based optimisation
- Apply derivatives, Jacobians, and the chain rule to function composition (backpropagation)
- Connect functions to measure theory: measurable functions, Lebesgue integration, probability distributions
- See every component of a neural network as a specific type of function with specific mathematical properties
- Appreciate the category-theoretic perspective: morphisms, functors, natural transformations

---

## Table of Contents

1. [Intuition](#1-intuition)
   - [1.1 What Is a Function?](#11-what-is-a-function)
   - [1.2 Why Functions Are Central to AI](#12-why-functions-are-central-to-ai)
   - [1.3 Functions as Transformations](#13-functions-as-transformations)
   - [1.4 The Language of Functions in Mathematics](#14-the-language-of-functions-in-mathematics)
   - [1.5 Historical Timeline](#15-historical-timeline)
   - [1.6 Pipeline Position in AI](#16-pipeline-position-in-ai)
2. [Formal Definitions](#2-formal-definitions)
   - [2.1 Function as a Set of Ordered Pairs](#21-function-as-a-set-of-ordered-pairs)
   - [2.2 Domain, Codomain, and Range](#22-domain-codomain-and-range)
   - [2.3 Image and Preimage](#23-image-and-preimage)
   - [2.4 Equality of Functions](#24-equality-of-functions)
   - [2.5 Partial Functions](#25-partial-functions)
3. [Types of Functions](#3-types-of-functions)
   - [3.1 Injective Functions (One-to-One)](#31-injective-functions-one-to-one)
   - [3.2 Surjective Functions (Onto)](#32-surjective-functions-onto)
   - [3.3 Bijective Functions (One-to-One and Onto)](#33-bijective-functions-one-to-one-and-onto)
   - [3.4 Summary Table](#34-summary-table)
   - [3.5 Monotone Functions](#35-monotone-functions)
   - [3.6 Periodic Functions](#36-periodic-functions)
4. [Function Composition](#4-function-composition)
   - [4.1 Definition and Notation](#41-definition-and-notation)
   - [4.2 Associativity of Composition](#42-associativity-of-composition)
   - [4.3 The Identity Function](#43-the-identity-function)
   - [4.4 Composition and Function Properties](#44-composition-and-function-properties)
   - [4.5 Neural Networks as Function Composition](#45-neural-networks-as-function-composition)
   - [4.6 Non-Commutativity of Composition](#46-non-commutativity-of-composition)
   - [4.7 Iterated Composition and Fixed Points](#47-iterated-composition-and-fixed-points)
5. [Inverse Functions](#5-inverse-functions)
   - [5.1 Left and Right Inverses](#51-left-and-right-inverses)
   - [5.2 The Inverse Function](#52-the-inverse-function)
   - [5.3 Inverse of a Composition](#53-inverse-of-a-composition)
   - [5.4 Pseudo-Inverse (Moore-Penrose)](#54-pseudo-inverse-moore-penrose)
   - [5.5 Invertibility and Information Preservation](#55-invertibility-and-information-preservation)
6. [Special Classes of Functions](#6-special-classes-of-functions)
   - [6.1 Linear Functions](#61-linear-functions)
   - [6.2 Affine Functions](#62-affine-functions)
   - [6.3 Multilinear Functions](#63-multilinear-functions)
   - [6.4 Polynomial Functions](#64-polynomial-functions)
   - [6.5 Activation Functions](#65-activation-functions)
   - [6.6 Convex Functions](#66-convex-functions)
   - [6.7 Lipschitz Functions](#67-lipschitz-functions)
7. [Higher-Order Functions and Functionals](#7-higher-order-functions-and-functionals)
   - [7.1 Functions as Arguments](#71-functions-as-arguments)
   - [7.2 Functionals (Functions of Functions)](#72-functionals-functions-of-functions)
   - [7.3 Operators (Functions on Function Spaces)](#73-operators-functions-on-function-spaces)
   - [7.4 Calculus of Variations](#74-calculus-of-variations)
   - [7.5 Transformations in AI Frameworks](#75-transformations-in-ai-frameworks)
8. [Continuity and Limits](#8-continuity-and-limits)
   - [8.1 The Epsilon-Delta Definition](#81-the-epsilon-delta-definition)
   - [8.2 Types of Continuity](#82-types-of-continuity)
   - [8.3 Continuity in Higher Dimensions](#83-continuity-in-higher-dimensions)
   - [8.4 Limits and Convergence](#84-limits-and-convergence)
   - [8.5 Discontinuities and Their Consequences](#85-discontinuities-and-their-consequences)
9. [Differentiable Functions and Derivatives](#9-differentiable-functions-and-derivatives)
   - [9.1 Derivative as a Linear Approximation](#91-derivative-as-a-linear-approximation)
   - [9.2 Smoothness Classes](#92-smoothness-classes)
   - [9.3 The Chain Rule and Backpropagation](#93-the-chain-rule-and-backpropagation)
   - [9.4 Gradient Problems in Deep Networks](#94-gradient-problems-in-deep-networks)
   - [9.5 Automatic Differentiation](#95-automatic-differentiation)
10. [Measure-Theoretic View of Functions](#10-measure-theoretic-view-of-functions)
    - [10.1 Why Measure Theory?](#101-why-measure-theory)
    - [10.2 Measurable Functions](#102-measurable-functions)
    - [10.3 Random Variables as Measurable Functions](#103-random-variables-as-measurable-functions)
    - [10.4 Lebesgue Integration](#104-lebesgue-integration)
    - [10.5 Almost Everywhere and Null Sets](#105-almost-everywhere-and-null-sets)
11. [Functions in Machine Learning](#11-functions-in-machine-learning)
    - [11.1 Universal Approximation Theorems](#111-universal-approximation-theorems)
    - [11.2 Loss Functions as Function Compositions](#112-loss-functions-as-function-compositions)
    - [11.3 Hypothesis Classes and Function Spaces](#113-hypothesis-classes-and-function-spaces)
    - [11.4 Attention as a Function](#114-attention-as-a-function)
    - [11.5 Tokenisation and Embedding as Functions](#115-tokenisation-and-embedding-as-functions)
    - [11.6 The Transformer Block as a Function](#116-the-transformer-block-as-a-function)
12. [Category Theory Perspective](#12-category-theory-perspective)
    - [12.1 Categories and Morphisms](#121-categories-and-morphisms)
    - [12.2 Functors: Maps Between Categories](#122-functors-maps-between-categories)
    - [12.3 Natural Transformations](#123-natural-transformations)
    - [12.4 Isomorphisms and Equivalences](#124-isomorphisms-and-equivalences)
    - [12.5 The Yoneda Lemma and Representations](#125-the-yoneda-lemma-and-representations)
13. [Common Mistakes and Misconceptions](#13-common-mistakes-and-misconceptions)
14. [Exercises and Practice Problems](#14-exercises-and-practice-problems)
15. [Why This Matters for AI/LLMs](#15-why-this-matters-for-aillms)
    - [15.1 Everything Is a Function](#151-everything-is-a-function)
    - [15.2 The Function Properties That Matter Most](#152-the-function-properties-that-matter-most)
    - [15.3 Scale and the Function Perspective](#153-scale-and-the-function-perspective)
16. [Conceptual Bridge](#16-conceptual-bridge)
    - [16.1 Looking Ahead](#161-looking-ahead)
    - [16.2 The Meta-Insight](#162-the-meta-insight)

---

## 1. Intuition

### 1.1 What Is a Function?

A function is a rule that assigns to each input exactly one output. No ambiguity, no missing outputs, no contradictions. If you give a function the same input twice, you get the same output both times. This is the single most important mathematical concept you will ever encounter.

The word "function" comes from the Latin _functio_, meaning "performance" or "execution". A function _performs_ a transformation: it takes something and produces something else, reliably and deterministically.

**The machine metaphor.** Think of a function as a machine with an input slot and an output slot. You drop an object into the input slot; the machine does something to it; exactly one object comes out of the output slot. The same input always produces the same output. The machine never refuses an input (within its declared domain), and it never produces two outputs for a single input.

```
              ┌──────────────┐
  input  ───▶│   f(x) = 2x  │───▶  output
  x = 3      │              │      f(3) = 6
              └──────────────┘

  Same input, same output — always:
  f(3) = 6, f(3) = 6, f(3) = 6, ...
```

**Formal view.** A function $f: A \to B$ is a subset of the Cartesian product $A \times B$ such that every element of $A$ appears as the first element of exactly one ordered pair. If $(a, b_1) \in f$ and $(a, b_2) \in f$, then $b_1 = b_2$. This definition reduces the concept of a function to the concept of a set — and since set theory is the foundation of all mathematics, this means functions are built on the most basic level of the mathematical hierarchy.

**The word "mapping".** The word "mapping" is a synonym for "function" that emphasises the geometric or structural idea. When we say "$f$ maps $A$ to $B$", we are emphasising that $f$ takes each point in one space and places it in another space. In linear algebra, we often say "linear map" rather than "linear function". In topology, we say "continuous map". The word "map" carries the same formal definition as "function".

**For AI and LLMs.** Every layer of a neural network is a function. Every activation function is a function. Every loss function is a function. The entire model — from raw text input to probability distribution over vocabulary — is a single (complicated) function. Training is optimisation: finding the best function within a parameterised family. Inference is function evaluation: computing the output for a given input. Understanding functions deeply is not a prerequisite for AI — it _is_ understanding AI.

### 1.2 Why Functions Are Central to AI

Every component of a modern AI system is a function. Here is a systematic mapping:

| AI Component | Mathematical Function | Type Signature |
|---|---|---|
| Neural network layer | $f_l$: vector space → vector space | $f_l: \mathbb{R}^{d} \to \mathbb{R}^{d'}$ |
| Embedding layer | Maps discrete tokens to continuous vectors | $E: V \to \mathbb{R}^d$ |
| Attention mechanism | Maps query, key, value triples to output | $\text{Attn}: \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_k} \times \mathbb{R}^{n \times d_v} \to \mathbb{R}^{n \times d_v}$ |
| Loss function | Maps parameters to scalar loss | $L: \Theta \to \mathbb{R}$ |
| Tokeniser | Maps strings to token sequences | $T: \Sigma^* \to V^*$ |
| Probability head | Maps logits to probability distribution | $\sigma: \mathbb{R}^{|V|} \to \Delta^{|V|-1}$ |
| Activation function | Elementwise nonlinearity | $\sigma: \mathbb{R} \to \mathbb{R}$ |
| Optimiser step | Maps current parameters to updated parameters | $\text{step}: \Theta \to \Theta$ |
| Learning rate scheduler | Maps step count to learning rate | $\eta: \mathbb{N} \to \mathbb{R}^+$ |
| Positional encoding | Maps position index to vector | $\text{PE}: \mathbb{N} \to \mathbb{R}^d$ |

Everything that happens inside an LLM — from the moment text enters as bytes to the moment a probability distribution over the next token is produced — is a composition of functions. Understanding the mathematical properties of these functions (continuity, differentiability, Lipschitz constants, injectivity, equivariance) directly determines what the model can learn, how stable training is, and what the model can represent.

### 1.3 Functions as Transformations

The geometric viewpoint reveals what functions _do_ to space. A function $f: \mathbb{R}^n \to \mathbb{R}^m$ takes every point in $n$-dimensional space and moves it to a point in $m$-dimensional space. Collections of points — lines, surfaces, regions — are transformed into new collections.

**Linear functions** preserve structure. Under a linear function, straight lines map to straight lines (or to points). The origin maps to the origin. Parallelism is preserved. Grid lines remain evenly spaced. Linear functions can rotate, reflect, stretch, compress, and project, but they cannot bend or fold.

```
Linear transformation of a grid:

  Before (identity):          After (rotation + stretch):
  ┌───┬───┬───┐               ╱   ╱   ╱   ╱
  │   │   │   │              ╱   ╱   ╱   ╱
  ├───┼───┼───┤             ╱   ╱   ╱   ╱
  │   │   │   │            ╱   ╱   ╱   ╱
  ├───┼───┼───┤           ╱   ╱   ╱   ╱
  │   │   │   │
  └───┴───┴───┘
  Grid lines remain straight; origin stays fixed.
```

**Nonlinear functions** bend, fold, and warp space. Under a nonlinear function, straight lines can become curves, parallel lines can converge or diverge, and the structure of space is fundamentally altered. This is exactly what enables neural networks to learn complex decision boundaries — if all functions were linear, a neural network would be no more powerful than a single matrix multiplication.

**Invertible functions (bijections)** transform space without destroying any information. Every point in the output space came from exactly one point in the input space, so you can always "undo" the transformation and recover the original. This is the principle behind normalising flows: a chain of bijective transformations that transforms a simple distribution (like Gaussian) into a complex one (like the data distribution), with the guarantee that the transformation can be reversed.

**Non-invertible functions (many-to-one)** destroy information. Multiple input points map to the same output point. Once the function is applied, you cannot tell which input was used. ReLU is a simple example: both $x = -3$ and $x = -5$ map to $\text{ReLU}(x) = 0$, so if you only see the output $0$, you cannot recover the input. This information destruction is actually useful — it's a form of compression, throwing away irrelevant details and keeping what matters.

**Deep networks** compose many simple functions. Each individual function may be simple (an affine map followed by an elementwise nonlinearity), but the composition of many such functions creates a global transformation that is highly nonlinear and enormously expressive. This is the fundamental insight of deep learning: complexity emerges from the composition of simplicity.

### 1.4 The Language of Functions in Mathematics

Functions appear everywhere in mathematics, but under different names depending on the context. Each name emphasises a different aspect of the same underlying concept:

| Name | Emphasis | Example |
|---|---|---|
| **Function** | General concept; rule from inputs to outputs | $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$ |
| **Map / Mapping** | Geometric/structural view; moving points between spaces | Linear map $T: V \to W$ |
| **Morphism** | Structure-preserving; respects algebraic properties | Group homomorphism $\phi: G \to H$ |
| **Operator** | Function acting on functions (or function spaces) | Differentiation $D$, integration $\int$ |
| **Functional** | Function whose input is a function, output is a scalar | Definite integral $I[f] = \int_a^b f(x)\,dx$ |
| **Transformation** | Emphasis on changing/deforming objects | Linear transformation, Fourier transform |
| **Kernel** | Function of two arguments used in integral operators or ML | RBF kernel $K(x, y) = \exp(-\|x-y\|^2/2\sigma^2)$ |
| **Distribution** | Generalised function (Schwartz); or probability function | Dirac delta $\delta(x)$; Gaussian PDF |

The key insight is that **all of these are instances of the same concept**. A linear map is a function that happens to satisfy linearity. An operator is a function whose domain is a function space. A functional is an operator that happens to return scalars. Understanding that these are all functions — with the same formal definition, the same composition rules, the same invertibility questions — unifies vast swaths of mathematics under a single framework.

### 1.5 Historical Timeline

The concept of a function evolved over centuries, from implicit geometric constructions to the fully abstract modern definition:

| Date | Contributor | Development |
|---|---|---|
| ~300 BCE | Euclid | Geometric constructions implicitly define functions (compass-and-straightedge); proportionality as functional relationship |
| 1694 | Leibniz | First explicit use of the word "function" (_functio_) in a mathematical context; described quantities depending on a curve |
| 1748 | Euler | Function as analytic expression; introduced the $f(x)$ notation; classified functions as algebraic or transcendental |
| 1837 | Dirichlet | Modern definition: function as _arbitrary_ rule assigning outputs to inputs, not necessarily given by a formula. The Dirichlet function $D(x) = 1$ if $x \in \mathbb{Q}$, $0$ otherwise, has no formula but is a valid function |
| 1870s | Cantor | Function as set of ordered pairs; foundation in set theory; functions between infinite sets |
| 1879 | Frege | Functions in formal logic; predicate as function mapping objects to truth values |
| 1888 | Dedekind | Morphism concept; structure-preserving maps between number systems |
| 1900s | Hilbert | Operator theory; functions on infinite-dimensional spaces; functional analysis foundations |
| 1920s | Banach | Normed function spaces (Banach spaces); fixed-point theorem; foundation of functional analysis |
| 1936 | Church | Lambda calculus: functions as fundamental computational objects; every computation is function evaluation |
| 1945 | Eilenberg & Mac Lane | Category theory: morphisms (functions) as primary objects; composition as fundamental operation |
| 1960s–1980s | Various | Functional programming languages (ML, Haskell, Lisp); functions as first-class values in computation |
| 1986 | Rumelhart, Hinton, Williams | Backpropagation popularised: neural network as differentiable function composition; chain rule as algorithm |
| 1989 | Cybenko, Hornik | Universal Approximation Theorem: neural networks can approximate any continuous function |
| 2017 | Vaswani et al. | Transformer architecture: attention as specific function class; self-attention as permutation-equivariant function |
| 2020–2026 | Scaling era | Functions at unprecedented scale: GPT-4, LLaMA, Gemini — trillions of parameters defining single functions from text to text |

The trend is clear: the concept of a function has become increasingly abstract and increasingly central. From Euler's "formula in $x$" to Dirichlet's "arbitrary rule" to Church's "computational object" to modern AI's "parameterised differentiable map" — each step expanded what counts as a function and what we can do with functions.

### 1.6 Pipeline Position in AI

Every stage of the LLM pipeline is a function. Here is the complete chain:

```
Input Data (text, tokens, images)
    ↓ [Tokeniser function T: Σ* → V*]
Token Sequences (discrete integer IDs)
    ↓ [Embedding function E: Vⁿ → ℝⁿˣᵈ]
Embedding Matrices (continuous vectors per token)
    ↓ [Positional encoding PE: ℝⁿˣᵈ → ℝⁿˣᵈ]
Position-Aware Embeddings
    ↓ [Transformer layer 1: f₁: ℝⁿˣᵈ → ℝⁿˣᵈ]
    ↓ [Transformer layer 2: f₂: ℝⁿˣᵈ → ℝⁿˣᵈ]
    ↓ [...]
    ↓ [Transformer layer L: fₗ: ℝⁿˣᵈ → ℝⁿˣᵈ]
Contextual Representations (hidden states)
    ↓ [Layer Norm: LN: ℝⁿˣᵈ → ℝⁿˣᵈ]
Normalised Representations
    ↓ [LM head (linear projection): W: ℝᵈ → ℝ|V|]
Logits (unnormalised scores per vocabulary token)
    ↓ [Softmax σ: ℝ|V| → Δ|V|⁻¹]
Probability Distribution (simplex)
    ↓ [Sampling function: sample: Δ|V|⁻¹ → V]
Output Token (single discrete token)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions everywhere. The entire pipeline is one
   composed function: sample ∘ σ ∘ W ∘ LN ∘ fₗ ∘ ... ∘ f₁ ∘ PE ∘ E ∘ T
```

Each arrow in this pipeline is a function with specific mathematical properties:
- **T** (tokeniser): discrete, non-differentiable, deterministic
- **E** (embedding): linear (lookup is matrix multiply with one-hot), differentiable
- **PE** (positional encoding): additive, fixed or learned, differentiable
- **f₁ through fₗ** (transformer layers): nonlinear, differentiable, permutation-equivariant, residual
- **LN** (layer norm): nonlinear, differentiable, scale-invariant
- **W** (LM head): linear, differentiable
- **σ** (softmax): nonlinear, differentiable, maps to probability simplex
- **sample**: stochastic (not a deterministic function in the classical sense; involves randomness)

Training optimises the parameters of the differentiable functions (E, f₁–fₗ, W) to minimise a loss function L over the training data. The loss function itself is a function $L: \Theta \to \mathbb{R}$ mapping the entire parameter vector to a single scalar.

---


## 2. Formal Definitions

### 2.1 Function as a Set of Ordered Pairs

**Definition.** A **function** from set $A$ to set $B$, written $f: A \to B$, is a subset $f \subseteq A \times B$ satisfying two conditions:

1. **Totality** (every input has an output): For every $a \in A$, there exists some $b \in B$ such that $(a, b) \in f$.
2. **Uniqueness** (single-valued): If $(a, b_1) \in f$ and $(a, b_2) \in f$, then $b_1 = b_2$.

Totality says the function is defined everywhere on its domain. Uniqueness says the function does not assign two different outputs to the same input. Together they guarantee: for each $a \in A$, there is **exactly one** $b \in B$ such that $(a, b) \in f$. We write this unique $b$ as $f(a)$.

**Example.** Consider $A = \{1, 2, 3\}$ and $B = \{a, b, c, d\}$. Then:

- $f = \{(1, a), (2, c), (3, c)\}$ is a valid function (total, single-valued). Note: two inputs map to the same output ($c$) — this is allowed.
- $g = \{(1, a), (3, b)\}$ is NOT a function (not total: input 2 has no output).
- $h = \{(1, a), (1, b), (2, c), (3, d)\}$ is NOT a function (not single-valued: input 1 maps to both $a$ and $b$).

**Relation vs function.** A **relation** from $A$ to $B$ is any subset of $A \times B$, without the totality or uniqueness requirements. Every function is a relation, but not every relation is a function. A relation that satisfies uniqueness but not totality is called a **partial function**.

```
   Relations ⊇ Partial Functions ⊇ Functions ⊇ Bijections
     (any subset     (unique but       (unique and    (unique, total,
      of A × B)      not total)         total)        one-to-one, onto)
```

**For AI.** When we write a neural network layer $f_l(x) = \sigma(Wx + b)$, we are defining a function from $\mathbb{R}^{d_{\text{in}}}$ to $\mathbb{R}^{d_{\text{out}}}$. Totality is guaranteed: for any input vector $x$, the matrix multiplication $Wx + b$ is defined, and $\sigma$ is defined on all of $\mathbb{R}$. Uniqueness is guaranteed: matrix multiplication is deterministic. During training with dropout, the computation is stochastic — but we model this as sampling a different function at each step, not as a single multi-valued function.

### 2.2 Domain, Codomain, and Range

Three fundamental sets are associated with every function $f: A \to B$:

| Concept | Symbol | Definition | Example: $f(x) = x^2$, $f: \mathbb{R} \to \mathbb{R}$ |
|---|---|---|---|
| **Domain** | $\text{dom}(f) = A$ | The set of all valid inputs | $\mathbb{R}$ (all real numbers) |
| **Codomain** | $\text{cod}(f) = B$ | The "target" set where outputs live | $\mathbb{R}$ (all real numbers) |
| **Range** (Image) | $\text{range}(f) = f(A)$ | The set of outputs actually produced | $[0, \infty)$ (non-negative reals) |

The distinction between codomain and range is subtle but crucial:

- The **codomain** is part of the function's definition — it is the set where outputs are declared to live.
- The **range** is a property of the function — it is the set of outputs that are actually achieved.
- Always: $\text{range}(f) \subseteq \text{cod}(f)$.
- A function is **surjective** (onto) if and only if $\text{range}(f) = \text{cod}(f)$.

**Why the distinction matters for AI.** Consider a neural network's output layer:

- Domain: $\mathbb{R}^d$ (any hidden representation is a valid input).
- Codomain: $\mathbb{R}^{|V|}$ (the output space has dimension equal to vocabulary size).
- Range: typically a strict subset of $\mathbb{R}^{|V|}$ — not every logit vector will actually be produced.

When we apply softmax, the codomain becomes the probability simplex $\Delta^{|V|-1} = \{p \in \mathbb{R}^{|V|} : p_i \geq 0, \sum p_i = 1\}$. The range is a strict subset: softmax can never output exactly 0 or exactly 1 for any component, because $e^z > 0$ for all finite $z$.

```
Domain (ℝᵈ)                    Codomain (ℝ|V|)
┌─────────────┐               ┌─────────────────┐
│             │               │                 │
│  All hidden │───f(x)───────▶│  All possible   │
│  states     │               │  logit vectors  │
│             │               │                 │
│             │               │ ┌─────────────┐ │
│             │               │ │ Range:      │ │
│             │               │ │ Actually    │ │
│             │               │ │ produced    │ │
│             │               │ │ logits      │ │
│             │               │ └─────────────┘ │
└─────────────┘               └─────────────────┘
```

### 2.3 Image and Preimage

**Image.** For a function $f: A \to B$ and a subset $S \subseteq A$, the **image** of $S$ under $f$ is:

$$f(S) = \{f(a) : a \in S\} = \{b \in B : \exists a \in S, \, f(a) = b\}$$

**Preimage (Inverse Image).** For a subset $T \subseteq B$, the **preimage** of $T$ under $f$ is:

$$f^{-1}(T) = \{a \in A : f(a) \in T\}$$

**Critical note:** The preimage notation $f^{-1}(T)$ does NOT require $f$ to have an inverse function. Even for non-invertible functions, the preimage of a set is always well-defined.

**Properties of images and preimages:**

| Property | Image | Preimage |
|---|---|---|
| Union | $f(A_1 \cup A_2) = f(A_1) \cup f(A_2)$ ✓ | $f^{-1}(B_1 \cup B_2) = f^{-1}(B_1) \cup f^{-1}(B_2)$ ✓ |
| Intersection | $f(A_1 \cap A_2) \subseteq f(A_1) \cap f(A_2)$ (⊆ only!) | $f^{-1}(B_1 \cap B_2) = f^{-1}(B_1) \cap f^{-1}(B_2)$ ✓ |
| Complement | No simple rule | $f^{-1}(B \setminus T) = A \setminus f^{-1}(T)$ ✓ |
| Subset | $S_1 \subseteq S_2 \implies f(S_1) \subseteq f(S_2)$ | $T_1 \subseteq T_2 \implies f^{-1}(T_1) \subseteq f^{-1}(T_2)$ |

**The key asymmetry.** Preimage preserves ALL set operations (unions, intersections, complements). Image only preserves unions. Image preserves intersections **if and only if** $f$ is injective.

**Proof that image does not preserve intersection in general.**

Let $f(x) = x^2$, $A_1 = \{-1, 2\}$, $A_2 = \{1, 3\}$.
- $A_1 \cap A_2 = \emptyset$, so $f(A_1 \cap A_2) = \emptyset$.
- $f(A_1) = \{1, 4\}$, $f(A_2) = \{1, 9\}$, so $f(A_1) \cap f(A_2) = \{1\}$.
- $\emptyset \neq \{1\}$, so $f(A_1 \cap A_2) \subsetneq f(A_1) \cap f(A_2)$. ∎

**For AI.** Preimages appear naturally in classification. The preimage $f^{-1}(\{\text{class } k\})$ is the **decision region** for class $k$ — the set of all inputs that the classifier assigns to class $k$. Understanding the geometry of these preimages (connected? convex? simply connected?) is central to understanding what the classifier has learned.

### 2.4 Equality of Functions

**Definition.** Two functions $f: A \to B$ and $g: A \to B$ are **equal**, written $f = g$, if and only if:

1. They have the same domain: $\text{dom}(f) = \text{dom}(g)$
2. They have the same codomain: $\text{cod}(f) = \text{cod}(g)$
3. They agree on all inputs: $f(a) = g(a)$ for all $a \in A$

This is the **principle of extensionality**: two functions are equal if and only if they produce the same output for every input. How they compute their output is irrelevant — only the input-output mapping matters.

**Example.** The functions:
- $f(x) = (x+1)^2 - 2x - 1$
- $g(x) = x^2$

are **equal** as functions $\mathbb{R} \to \mathbb{R}$, because $(x+1)^2 - 2x - 1 = x^2 + 2x + 1 - 2x - 1 = x^2$ for all $x$.

**For AI.** Two neural networks with completely different weights can compute the same function (e.g., by permuting neurons — see Section 11.4). Conversely, two architectures that "look" the same but have different weights compute different functions. Function equality is about the input-output mapping, not the implementation.

### 2.5 Partial Functions

A **partial function** $f: A \rightharpoonup B$ satisfies uniqueness but not necessarily totality. There may be elements of $A$ for which $f$ is undefined.

**The domain of definition:** $\text{dom}(f) = \{a \in A : f(a) \text{ is defined}\} \subseteq A$.

**Examples in mathematics:**
- **Division**: $f(x) = 1/x$ is partial on $\mathbb{R}$ — undefined at $x = 0$
- **Logarithm**: $\log(x)$ is partial on $\mathbb{R}$ — undefined for $x \leq 0$
- **Square root**: $\sqrt{x}$ is partial on $\mathbb{R}$ — undefined for $x < 0$

**Examples in AI:**
- **Masked attention**: In causal (autoregressive) attention, token $i$ cannot attend to token $j > i$. The attention weight function is "undefined" for future positions — set to $-\infty$ before softmax, giving weight 0.
- **Sparse attention**: Only a subset of positions are attended to; the attention function is partial.
- **Variable-length sequences**: A function defined on sequences up to length $n$ is partial on the space of all sequences.

**Making partial functions total.** Two standard approaches:
1. **Restrict the domain**: change $A$ to only elements where $f$ is defined. $\sqrt{x}$ becomes total on $[0, \infty)$.
2. **Extend the codomain**: add a special "undefined" value $\bot$ to $B$ and set $f(a) = \bot$ for undefined inputs.

In neural network implementations, the extension approach is standard: masked attention sets logits to $-\infty$ (a stand-in for "undefined"), and after softmax, these become 0, contributing nothing to the weighted average.

---


## 3. Types of Functions

### 3.1 Injective Functions (One-to-One)

**Definition.** A function $f: A \to B$ is **injective** (or **one-to-one**) if different inputs always produce different outputs:

$$\forall a_1, a_2 \in A: \quad f(a_1) = f(a_2) \implies a_1 = a_2$$

Equivalently (contrapositive): $a_1 \neq a_2 \implies f(a_1) \neq f(a_2)$.

**Intuition.** An injective function preserves distinctness. No two inputs "collide" at the same output. If you see the output, you can determine which input produced it.

```
Injective:                    NOT injective:
  A        B                    A        B
  1 ────▶ a                    1 ────▶ a
  2 ────▶ b                    2 ────┐
  3 ────▶ c                    3 ────┘▶ b
           d (not hit)                  c (not hit)
  
Each output comes from         Two inputs map to
at most one input              the same output (b)
```

**Proof technique for injectivity.** Assume $f(a_1) = f(a_2)$ and derive $a_1 = a_2$.

**Example 1.** Prove that $f(x) = 2x + 3$ is injective.

*Proof.* Suppose $f(a_1) = f(a_2)$. Then $2a_1 + 3 = 2a_2 + 3$, so $2a_1 = 2a_2$, so $a_1 = a_2$. ∎

**Example 2.** Prove that $f(x) = e^x$ is injective.

*Proof.* Suppose $e^{a_1} = e^{a_2}$. Taking $\ln$ of both sides: $a_1 = a_2$. ∎

**Example 3.** Prove that $f(x) = x^2$ is NOT injective on $\mathbb{R}$.

*Proof.* Counterexample: $f(1) = 1 = f(-1)$ but $1 \neq -1$. ∎

**Proof technique to disprove injectivity**: find two distinct inputs with the same output (a single counterexample suffices).

**For AI.** Injectivity means **no information is lost**. If a layer is injective, the original input can (in principle) be recovered from the output. Non-injective layers destroy information.

| Component | Injective? | Why |
|---|---|---|
| Token embedding $E$ | Yes | Each token maps to a unique vector |
| ReLU activation | No | All negative values map to 0 |
| Sigmoid activation | Yes | Strictly monotone increasing |
| Softmax ($\mathbb{R}^n \to \Delta^{n-1}$) | No | Adding constant to all inputs gives same output |
| Max pooling | No | Different inputs with same max give same output |
| Average pooling | No | Different inputs with same mean give same output |
| Invertible ResNet layers | Yes | Designed to be bijective |

### 3.2 Surjective Functions (Onto)

**Definition.** A function $f: A \to B$ is **surjective** (or **onto**) if every element of the codomain is achieved:

$$\forall b \in B, \; \exists a \in A: \quad f(a) = b$$

Equivalently: $\text{range}(f) = B$. Every element of $B$ is "hit" by at least one input.

```
Surjective:                   NOT surjective:
  A        B                    A        B
  1 ────▶ a                    1 ────▶ a
  2 ───┐                       2 ────▶ b
  3 ───┘▶ b                    3 ────▶ c
  4 ────▶ c                             d ← never reached
  
Every output is hit            Element d in codomain
by at least one input          is never produced
```

**Proof technique for surjectivity.** Take an arbitrary $b \in B$ and explicitly find an $a \in A$ with $f(a) = b$.

**Example 1.** Prove that $f: \mathbb{R} \to \mathbb{R}$, $f(x) = 2x + 3$ is surjective.

*Proof.* Let $b \in \mathbb{R}$. We need $a$ with $f(a) = b$: solve $2a + 3 = b$ to get $a = (b-3)/2$. Since $b \in \mathbb{R}$, we have $a \in \mathbb{R}$, and $f(a) = 2 \cdot \frac{b-3}{2} + 3 = b$. ∎

**Example 2.** Prove that $f(x) = x^2$ is NOT surjective as $f: \mathbb{R} \to \mathbb{R}$.

*Proof.* Take $b = -1$. For any $a \in \mathbb{R}$, $f(a) = a^2 \geq 0 > -1$, so no $a$ maps to $-1$. ∎

**Note on codomain choice.** The same formula $f(x) = x^2$ gives:
- $f: \mathbb{R} \to \mathbb{R}$ — NOT surjective (negative numbers not in range)
- $f: \mathbb{R} \to [0, \infty)$ — surjective (every non-negative number is a perfect square of something)

Changing the codomain changes whether the function is surjective.

**For AI — surjectivity means the output space is fully utilised:**

| Component | Surjective? | Why |
|---|---|---|
| Softmax onto open simplex | Yes | Any strictly positive distribution achievable |
| ReLU: $\mathbb{R} \to \mathbb{R}$ | No | Negative outputs never produced |
| ReLU: $\mathbb{R} \to [0, \infty)$ | Yes | Every non-negative value achievable |
| Tanh: $\mathbb{R} \to (-1, 1)$ | Yes | Approaches but never reaches $\pm 1$ |
| Linear $Wx$: $\mathbb{R}^n \to \mathbb{R}^m$ | Only if $\text{rank}(W) = m$ | Range is column space of $W$ |

### 3.3 Bijective Functions (One-to-One and Onto)

**Definition.** A function $f: A \to B$ is **bijective** if it is both injective AND surjective. Every element of $B$ is the image of **exactly one** element of $A$.

```
Bijective (perfect pairing):
  A        B
  1 ────▶ a
  2 ────▶ b
  3 ────▶ c
  
  Every element of A maps to a unique element of B.
  Every element of B is hit exactly once.
  No information created, no information destroyed.
```

**Key property.** A function has a two-sided inverse $f^{-1}: B \to A$ if and only if it is bijective.

**Bijections and cardinality.** A bijection $f: A \to B$ proves that $|A| = |B|$ (same cardinality). This is the formal definition of "same size" for sets, including infinite sets. For example, $f(n) = 2n$ is a bijection $\mathbb{N} \to \{0, 2, 4, 6, \ldots\}$, proving that the natural numbers and the even numbers have the same cardinality — a result that seems paradoxical until you see the bijection.

**For AI — where bijections are critical:**

| Application | How Bijection Is Used |
|---|---|
| **Normalising flows** | Chain of bijections transforms simple distribution to complex one. Change of variables: $p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)|$ |
| **Reversible architectures** | Invertible ResNets (iRevNet, Glow) save memory by recomputing activations during backprop |
| **Permutation matrices** | Reordering tokens is a bijection on indices; attention is permutation-equivariant |
| **Tokeniser / Detokeniser** | Bijection between token IDs and token strings (within vocabulary) |
| **Encryption** | Encryption and decryption are inverse bijections; information-preserving |

### 3.4 Summary Table

| Property | Condition | Injective? | Surjective? | Invertible? | Example |
|---|---|---|---|---|---|
| **Injective only** | $f(a_1) = f(a_2) \Rightarrow a_1 = a_2$ | ✓ | ✗ | Left inverse | $f(x) = e^x$, $\mathbb{R} \to \mathbb{R}$ |
| **Surjective only** | $\text{range}(f) = B$ | ✗ | ✓ | Right inverse | $f(x) = x^2$, $\mathbb{R} \to [0,\infty)$ |
| **Bijective** | Both | ✓ | ✓ | Two-sided | $f(x) = 2x + 1$, $\mathbb{R} \to \mathbb{R}$ |
| **Neither** | Neither | ✗ | ✗ | None | $f(x) = \sin(x)$, $\mathbb{R} \to [-2, 2]$ |

### 3.5 Monotone Functions

**Definition.** A function $f: \mathbb{R} \to \mathbb{R}$ is:

- **Strictly increasing**: $x_1 < x_2 \implies f(x_1) < f(x_2)$
- **Non-decreasing** (weakly increasing): $x_1 < x_2 \implies f(x_1) \leq f(x_2)$
- **Strictly decreasing**: $x_1 < x_2 \implies f(x_1) > f(x_2)$
- **Non-increasing** (weakly decreasing): $x_1 < x_2 \implies f(x_1) \geq f(x_2)$

A function is **monotone** if it is either non-decreasing or non-increasing.

**Key theorem.** Every strictly monotone function is injective.

*Proof.* Suppose $f$ is strictly increasing and $f(a_1) = f(a_2)$. If $a_1 < a_2$, then $f(a_1) < f(a_2)$ — contradiction. If $a_1 > a_2$, then $f(a_1) > f(a_2)$ — contradiction. So $a_1 = a_2$. ∎

**Monotonicity in AI:**

| Function | Monotone? | Type | Consequence |
|---|---|---|---|
| Sigmoid $\sigma(x)$ | Strictly increasing | On all $\mathbb{R}$ | Injective; preserves input order |
| Tanh | Strictly increasing | On all $\mathbb{R}$ | Injective; preserves order |
| ReLU | Non-decreasing (not strict) | Flat for $x < 0$ | Not injective; loses sign info |
| GELU | NOT monotone | Dip near $x \approx -0.17$ | Not order-preserving |
| Leaky ReLU ($\alpha > 0$) | Strictly increasing | On all $\mathbb{R}$ | Injective; all info preserved |
| SiLU/Swish $x\sigma(x)$ | NOT monotone | Dip near $x \approx -1.28$ | Similar to GELU |
| Cross-entropy $L(p)$ | Strictly decreasing in $p$ | Correct class prob | Higher confidence → lower loss |

**Why monotonicity matters for optimisation.** If the loss function is monotone in some parameter direction, that direction has no local minima — the loss either always decreases or always increases. Non-monotone loss landscapes create local minima and saddle points that can trap gradient descent.

### 3.6 Periodic Functions

**Definition.** A function $f: \mathbb{R} \to \mathbb{R}$ is **periodic** with period $T > 0$ if:

$$f(x + T) = f(x) \quad \text{for all } x \in \mathbb{R}$$

The smallest such $T$ is the **fundamental period**.

**Classical examples:**
- $\sin(x)$ and $\cos(x)$: period $2\pi$
- $\tan(x)$: period $\pi$
- $e^{ix}$: period $2\pi$ (Euler's formula)

**Key property.** Periodic functions on $\mathbb{R}$ are **never injective** (since $f(x) = f(x + T)$ with $x \neq x + T$). However, they are injective when restricted to one period $[0, T)$.

**Periodic functions in modern AI:**

**1. Sinusoidal Positional Encoding (Vaswani et al. 2017):**

$$\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \quad \text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)$$

Each dimension is a periodic function of position with a different frequency. Low dimensions have short periods (capture local position); high dimensions have long periods (capture global position). The periods range from $2\pi$ to $2\pi \cdot 10000$.

**Why sinusoidal?** For any fixed offset $k$, $\text{PE}(\text{pos} + k)$ can be written as a linear function of $\text{PE}(\text{pos})$:

$$\begin{pmatrix} \sin(\omega(\text{pos}+k)) \\ \cos(\omega(\text{pos}+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} \sin(\omega \cdot \text{pos}) \\ \cos(\omega \cdot \text{pos}) \end{pmatrix}$$

This means relative position information is accessible via a linear transformation — the model can learn to attend to "3 positions back" using a simple linear projection.

**2. Rotary Position Embedding (RoPE, Su et al. 2021):**

RoPE rotates the query and key vectors by position-dependent angles:

$$f_{\text{RoPE}}(x_m, m) = R_{\Theta, m} \cdot x_m$$

where $R_{\Theta, m}$ is a block-diagonal rotation matrix with angles $\theta_i \cdot m$. Each block rotates by angle $\theta_i \cdot m$ — a periodic function of position with period $2\pi/\theta_i$.

**Key property of RoPE:** The dot product $\langle f_{\text{RoPE}}(q, m), f_{\text{RoPE}}(k, n) \rangle$ depends only on $q$, $k$, and the relative position $m - n$ (not the absolute positions). This gives the model a natural notion of relative distance.

**3. Fourier features in NeRF (Neural Radiance Fields):**

$$\gamma(p) = (\sin(2^0 \pi p), \cos(2^0 \pi p), \sin(2^1 \pi p), \cos(2^1 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p))$$

Periodic functions at multiple frequencies encode position information at multiple scales, enabling the network to represent high-frequency details.

---


## 4. Function Composition

### 4.1 Definition and Notation

**Definition.** Given functions $f: A \to B$ and $g: B \to C$, the **composition** of $g$ with $f$ is the function:

$$(g \circ f): A \to C, \quad (g \circ f)(x) = g(f(x))$$

Read: "g composed with f" or "g after f". The function $f$ is applied first, then $g$. Note the order: in $g \circ f$, the rightmost function ($f$) is applied first.

**Requirement.** The composition $g \circ f$ is only defined when $\text{range}(f) \subseteq \text{dom}(g)$, i.e., the outputs of $f$ must be valid inputs for $g$.

```
     f            g
A ──────▶ B ──────▶ C

     g ∘ f
A ────────────────▶ C

(g ∘ f)(x) = g(f(x))
```

**For AI.** This requirement means the output dimension of layer $l$ must match the input dimension of layer $l+1$. Dimension mismatches between layers are among the most common neural network implementation bugs.

### 4.2 Associativity of Composition

**Theorem.** Function composition is associative: for $f: A \to B$, $g: B \to C$, $h: C \to D$:

$$(h \circ g) \circ f = h \circ (g \circ f)$$

*Proof.* For any $x \in A$:

$$((h \circ g) \circ f)(x) = (h \circ g)(f(x)) = h(g(f(x)))$$

$$(h \circ (g \circ f))(x) = h((g \circ f)(x)) = h(g(f(x)))$$

Both sides equal $h(g(f(x)))$. Since this holds for all $x \in A$, the functions are equal by extensionality. ∎

**Consequence.** We can write $h \circ g \circ f$ without ambiguity — no parentheses needed.

**For AI.** Associativity means we can group layers however we want. A 96-layer transformer can be viewed as:
- 96 individual layers
- Two groups of 48 layers
- Three groups of 32 layers

The computed function is the same regardless of grouping. This is why we can meaningfully talk about "the first 10 layers" or "the last 3 layers" as coherent sub-computations. It also allows pipeline parallelism: split a model across GPUs at any layer boundary, and the overall function is unchanged.

### 4.3 The Identity Function

**Definition.** The **identity function** on set $A$ is:

$$\text{id}_A: A \to A, \quad \text{id}_A(x) = x$$

**Properties:**
- $f \circ \text{id}_A = f$ for any $f: A \to B$ (right identity)
- $\text{id}_B \circ f = f$ for any $f: A \to B$ (left identity)

*Proof of right identity.* For any $x \in A$: $(f \circ \text{id}_A)(x) = f(\text{id}_A(x)) = f(x)$. ∎

**For AI — Residual Connections.** The identity function appears in **residual connections** (He et al. 2016):

$$\text{ResBlock}(x) = x + F(x) = \text{id}(x) + F(x)$$

The output is the input plus a learned residual. If $F(x) \approx 0$, the block approximates the identity. This design allows the network to learn modifications to the identity, rather than learning the entire transformation from scratch.

**Why this helps gradient flow:** The Jacobian of a residual block is:

$$\frac{\partial}{\partial x}\big(x + F(x)\big) = I + \frac{\partial F}{\partial x}$$

The identity term $I$ ensures gradients always flow through, even if $\frac{\partial F}{\partial x}$ is small. Without residual connections, gradients must pass through every layer multiplicatively — if any layer's Jacobian has small singular values, gradients vanish. With residual connections, there is always a "skip path" carrying the identity gradient.

**Pre-LN Transformer block (GPT-2, LLaMA, etc.):**

$$\text{Block}(x) = x + \text{FFN}(\text{LN}(x + \text{Attn}(\text{LN}(x))))$$

Two residual connections per block: one around attention, one around FFN. Each is $\text{id} + \text{learnable function}$.

### 4.4 Composition and Function Properties

**Theorem.** Composition preserves injectivity, surjectivity, and bijectivity:

| If $f$ is... | and $g$ is... | then $g \circ f$ is... | Proof sketch |
|---|---|---|---|
| Injective | Injective | **Injective** | $g(f(a_1)) = g(f(a_2)) \xRightarrow{g \text{ inj}} f(a_1) = f(a_2) \xRightarrow{f \text{ inj}} a_1 = a_2$ |
| Surjective | Surjective | **Surjective** | For $c \in C$: $g$ surj $\Rightarrow \exists b$ s.t. $g(b)=c$; $f$ surj $\Rightarrow \exists a$ s.t. $f(a)=b$ |
| Bijective | Bijective | **Bijective** | Combine both rows above |

*Full proof of injectivity preservation.* Suppose $f$ and $g$ are both injective. Let $(g \circ f)(a_1) = (g \circ f)(a_2)$, i.e., $g(f(a_1)) = g(f(a_2))$. Since $g$ is injective, $f(a_1) = f(a_2)$. Since $f$ is injective, $a_1 = a_2$. ∎

**Converse results (partial converses):**
- If $g \circ f$ is injective, then $f$ must be injective (but $g$ need not be).
- If $g \circ f$ is surjective, then $g$ must be surjective (but $f$ need not be).

*Proof that $g \circ f$ injective implies $f$ injective.* Suppose $f(a_1) = f(a_2)$. Then $g(f(a_1)) = g(f(a_2))$, i.e., $(g \circ f)(a_1) = (g \circ f)(a_2)$. Since $g \circ f$ is injective, $a_1 = a_2$. ∎

**For AI.** If every layer is injective, the entire network is injective (no information loss). If any single layer is non-injective, the composition is non-injective. Since ReLU is non-injective (maps all negatives to 0), standard ReLU networks are non-injective — they destroy information about the sign of pre-activations.

### 4.5 Neural Networks as Function Composition

A deep neural network is exactly a composition of simpler functions. For a transformer with $L$ layers:

$$f_{\text{transformer}} = \text{LM\_head} \circ \text{LN} \circ f_L \circ f_{L-1} \circ \cdots \circ f_1 \circ \text{PE} \circ \text{Embed}$$

where each transformer layer $f_l$ is itself a composition:

$$f_l(x) = x + \text{FFN}\big(\text{LN}_2(x + \text{Attn}(\text{LN}_1(x)))\big)$$

expanding further:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

$$\text{Attn}(X) = \text{softmax}\!\left(\frac{XW_Q(XW_K)^T}{\sqrt{d_k}}\right)XW_V \cdot W_O$$

The total function is a composition of $O(L)$ sub-functions, each parameterised by learnable weights. All are differentiable (except sampling at inference), enabling end-to-end gradient-based training via the chain rule.

**Counting the depth.** A single transformer layer involves roughly:
- 2 layer norms (each: subtract mean, divide by std, scale, shift)
- 4 linear projections (Q, K, V, O)
- 1 softmax
- 2 more linear projections (FFN up and down)
- 1 activation function
- 2 residual additions

That is approximately 12 elementary functions per transformer layer. A 96-layer model composes $96 \times 12 \approx 1{,}152$ elementary functions. The fact that gradient-based optimisation works over such deep compositions is remarkable and depends critically on residual connections and careful normalisation.

### 4.6 Non-Commutativity of Composition

**Composition is NOT commutative.** In general, $f \circ g \neq g \circ f$.

**Example.** Let $f(x) = x^2$ and $g(x) = x + 1$:

- $(f \circ g)(x) = f(g(x)) = f(x+1) = (x+1)^2 = x^2 + 2x + 1$
- $(g \circ f)(x) = g(f(x)) = g(x^2) = x^2 + 1$

These are different: $(f \circ g)(2) = 9$ but $(g \circ f)(2) = 5$.

**When does composition commute?** Two functions $f$ and $g$ commute (i.e., $f \circ g = g \circ f$) in special cases:
- $f = \text{id}$ or $g = \text{id}$ (trivially)
- $f$ and $g$ are both powers of the same function: $f^{(m)} \circ f^{(n)} = f^{(n)} \circ f^{(m)} = f^{(m+n)}$
- $f$ and $g$ are both linear: if $f(x) = Ax$ and $g(x) = Bx$, then $f \circ g = g \circ f$ iff $AB = BA$ (matrices commute)
- Diagonal matrices always commute with each other

**For AI.** The order of operations matters enormously:

| Ordering | Effect |
|---|---|
| Pre-LN: $\text{LN} \to \text{Attn}$ | More stable training; used in GPT-2, LLaMA, most modern LLMs |
| Post-LN: $\text{Attn} \to \text{LN}$ | Original Transformer; training can be unstable without warmup |
| BatchNorm before activation | Standard in CNNs |
| BatchNorm after activation | Different normalisation statistics; rarely used |
| Dropout before softmax | Drops attention logits; called "attention dropout" |
| Dropout after softmax | Drops attention weights; different regularisation effect |

The shift from Post-LN to Pre-LN significantly improved training stability for deep transformers — same components, different composition order. This is a direct consequence of non-commutativity.

### 4.7 Iterated Composition and Fixed Points

**Definition.** The $n$-fold composition of $f$ with itself:

$$f^{(n)} = \underbrace{f \circ f \circ \cdots \circ f}_{n \text{ times}}$$

with $f^{(0)} = \text{id}$.

**Fixed point.** A point $x^*$ is a **fixed point** of $f$ if $f(x^*) = x^*$. Fixed points are invariant under iteration: $f^{(n)}(x^*) = x^*$ for all $n$.

**Banach Fixed Point Theorem (Contraction Mapping Theorem).** Let $(X, d)$ be a complete metric space and $f: X \to X$ a **contraction**, meaning there exists $L \in [0, 1)$ such that:

$$d(f(x), f(y)) \leq L \cdot d(x, y) \quad \text{for all } x, y \in X$$

Then:
1. $f$ has a **unique** fixed point $x^* \in X$
2. For any starting point $x_0 \in X$, the sequence $x_{n+1} = f(x_n)$ converges to $x^*$
3. Rate of convergence: $d(x_n, x^*) \leq \frac{L^n}{1-L} \cdot d(x_0, x_1)$

*Proof sketch.* The sequence $x_0, f(x_0), f(f(x_0)), \ldots$ is Cauchy because $d(x_{n+1}, x_n) \leq L \cdot d(x_n, x_{n-1}) \leq L^n \cdot d(x_1, x_0) \to 0$. Since $X$ is complete, it converges to some $x^*$. By continuity of $f$: $f(x^*) = f(\lim x_n) = \lim f(x_n) = \lim x_{n+1} = x^*$. Uniqueness: if $f(x^*) = x^*$ and $f(y^*) = y^*$, then $d(x^*, y^*) = d(f(x^*), f(y^*)) \leq L \cdot d(x^*, y^*)$, so $(1-L) \cdot d(x^*, y^*) \leq 0$, giving $d(x^*, y^*) = 0$. ∎

**For AI: Deep Equilibrium Models (DEQ, Bai et al. 2019).** Instead of stacking $L$ copies of a layer (using $O(L)$ memory for activations), DEQ finds the fixed point of a single layer:

$$z^* = f_\theta(z^*, x) \quad \text{where } z^* = \lim_{n \to \infty} f_\theta^{(n)}(z_0, x)$$

This is equivalent to an infinitely deep weight-tied network that has converged.

| Property | Standard Deep Network | DEQ |
|---|---|---|
| Memory | $O(L)$ — store all activations | $O(1)$ — store only $z^*$ |
| Computation | $L$ forward passes | Iterate until convergence |
| Backprop | Through all $L$ layers | Implicit differentiation at fixed point |
| Convergence guarantee | N/A | Guaranteed if $f_\theta$ is a contraction |

The Banach theorem guarantees convergence if the layer is a contraction — which can be enforced via spectral normalisation (constraining $\|W\|_{\text{op}} < 1$).

**Example: iterating cosine.**
$f(x) = \cos(x)$ on $[0, 1]$ is a contraction (since $|f'(x)| = |\sin(x)| \leq \sin(1) \approx 0.84 < 1$).

Starting from $x_0 = 0.5$:
- $x_1 = \cos(0.5) \approx 0.8776$
- $x_2 = \cos(0.8776) \approx 0.6390$
- $x_3 = \cos(0.6390) \approx 0.8027$
- Converges to the Dottie number: $x^* \approx 0.7391$ where $\cos(x^*) = x^*$.

---


## 5. Inverse Functions

### 5.1 Left and Right Inverses

Not every function has a full (two-sided) inverse. The general picture involves left and right inverses:

**Definition.** For $f: A \to B$:

- A **left inverse** of $f$ is a function $g: B \to A$ such that $g \circ f = \text{id}_A$ (i.e., $g(f(a)) = a$ for all $a$)
- A **right inverse** of $f$ is a function $h: B \to A$ such that $f \circ h = \text{id}_B$ (i.e., $f(h(b)) = b$ for all $b$)
- A **two-sided inverse** is a function that is both a left and right inverse

**Key theorems connecting inverses to injectivity/surjectivity:**

| Statement | Condition |
|---|---|
| $f$ has a left inverse | $\iff$ $f$ is injective |
| $f$ has a right inverse | $\iff$ $f$ is surjective (requires Axiom of Choice) |
| $f$ has a two-sided inverse | $\iff$ $f$ is bijective |
| If both left and right inverses exist | They are equal and unique |

**Proof that injective implies left inverse exists.** Suppose $f: A \to B$ is injective with $A \neq \emptyset$. Fix some $a_0 \in A$. Define $g: B \to A$ by:

$$g(b) = \begin{cases} a & \text{if } b = f(a) \text{ for some (unique by injectivity) } a \in A \\ a_0 & \text{if } b \notin f(A) \end{cases}$$

Then for any $a \in A$: $g(f(a)) = a$ (by the first case), so $g \circ f = \text{id}_A$. ∎

**Proof that left and right inverse, if both exist, are equal.** Suppose $g$ is a left inverse ($g \circ f = \text{id}_A$) and $h$ is a right inverse ($f \circ h = \text{id}_B$). Then:

$$g = g \circ \text{id}_B = g \circ (f \circ h) = (g \circ f) \circ h = \text{id}_A \circ h = h$$

So $g = h$. ∎

**For AI.** The distinction between left and right inverses appears in:
- **Encoder-decoder architectures**: The encoder $f$ maps high-dimensional input to a low-dimensional bottleneck. The decoder $g$ maps back. We want $g \circ f \approx \text{id}$ (reconstruct the original), making $g$ an approximate left inverse. But $f \circ g \neq \text{id}$ (not every bottleneck vector corresponds to a real input).
- **Projection layers**: Reducing dimension from $d$ to $d' < d$ has a left inverse (pseudo-inverse) but not a right inverse.

### 5.2 The Inverse Function

**Definition.** If $f: A \to B$ is bijective, the **inverse function** $f^{-1}: B \to A$ is defined by:

$$f^{-1}(b) = a \quad \iff \quad f(a) = b$$

**Properties of inverse functions:**

| Property | Statement |
|---|---|
| Composition with original | $f^{-1} \circ f = \text{id}_A$ and $f \circ f^{-1} = \text{id}_B$ |
| Inverse of inverse | $(f^{-1})^{-1} = f$ |
| Bijectivity | $f^{-1}$ is itself bijective |
| Graph reflection | The graph of $f^{-1}$ is the graph of $f$ reflected across $y = x$ |
| Continuity | If $f$ is continuous and bijective (on appropriate spaces), $f^{-1}$ is continuous |

**Common inverse pairs used in AI:**

| Function $f(x)$ | Inverse $f^{-1}(y)$ | Domain of $f$ | AI Use |
|---|---|---|---|
| $e^x$ | $\ln(y)$ | $\mathbb{R} \to (0,\infty)$ | Log-probabilities; $\log \circ \text{softmax}$ (LogSoftmax) |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\text{logit}(y) = \ln\!\left(\frac{y}{1-y}\right)$ | $\mathbb{R} \to (0,1)$ | Logit-space operations; probability calibration |
| $Ax$ ($A$ invertible) | $A^{-1}y$ | $\mathbb{R}^n \to \mathbb{R}^n$ | Solving linear systems; whitening transforms |
| $ax + b$ ($a \neq 0$) | $(y - b)/a$ | $\mathbb{R} \to \mathbb{R}$ | Undoing normalisation; de-standardisation |
| $\tanh(x)$ | $\text{arctanh}(y) = \frac{1}{2}\ln\!\left(\frac{1+y}{1-y}\right)$ | $\mathbb{R} \to (-1,1)$ | Inverse activation in normalising flows |

### 5.3 Inverse of a Composition

**Theorem (Socks-and-Shoes Lemma).** If $f: A \to B$ and $g: B \to C$ are both bijective, then:

$$(g \circ f)^{-1} = f^{-1} \circ g^{-1}$$

The inverse of a composition **reverses the order**.

*Proof.* We verify that $f^{-1} \circ g^{-1}$ is the inverse of $g \circ f$:

$$(f^{-1} \circ g^{-1}) \circ (g \circ f) = f^{-1} \circ (g^{-1} \circ g) \circ f = f^{-1} \circ \text{id}_B \circ f = f^{-1} \circ f = \text{id}_A$$

Similarly: $(g \circ f) \circ (f^{-1} \circ g^{-1}) = g \circ (f \circ f^{-1}) \circ g^{-1} = g \circ \text{id}_B \circ g^{-1} = g \circ g^{-1} = \text{id}_C$. ∎

**Name origin.** You put on socks first, then shoes. To undo: you remove shoes first, then socks. The inverse reverses the order.

**Generalisation.** For $n$ bijections:

$$(f_n \circ f_{n-1} \circ \cdots \circ f_1)^{-1} = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_n^{-1}$$

**For AI: Normalising Flows.** A normalising flow is a composition of bijections:

$$f = f_K \circ f_{K-1} \circ \cdots \circ f_1$$

By the socks-and-shoes lemma, the inverse is:

$$f^{-1} = f_1^{-1} \circ f_2^{-1} \circ \cdots \circ f_K^{-1}$$

**Sampling** (generation): start with noise $z \sim \mathcal{N}(0, I)$ and apply $f$ (forward pass through all layers in order $f_1, f_2, \ldots, f_K$).

**Likelihood computation** (training): given data $x$, apply $f^{-1}$ (inverse pass through layers in reverse order $f_K^{-1}, \ldots, f_1^{-1}$), and use the change-of-variables formula:

$$\log p(x) = \log p(z) + \sum_{k=1}^{K} \log\!\left|\det \frac{\partial f_k^{-1}}{\partial z_k}\right|$$

The order matters: forward pass and inverse pass traverse the layers in opposite directions.

### 5.4 Pseudo-Inverse (Moore-Penrose)

When $f$ is not bijective (e.g., a non-square matrix, or a square but singular matrix), no true inverse exists. The **Moore-Penrose pseudo-inverse** provides the "best substitute".

**Definition.** For a matrix $A \in \mathbb{R}^{m \times n}$, the pseudo-inverse $A^+ \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying the four **Penrose conditions**:

1. $A A^+ A = A$ (acts like an inverse when it can)
2. $A^+ A A^+ = A^+$ (acts like an inverse when it can)
3. $(A A^+)^T = A A^+$ (symmetric projection onto column space)
4. $(A^+ A)^T = A^+ A$ (symmetric projection onto row space)

**Via SVD.** If $A = U \Sigma V^T$ (singular value decomposition), then:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ is obtained by taking the reciprocal of each non-zero singular value and transposing. If $\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_r, 0, \ldots, 0)$, then $\Sigma^+ = \text{diag}(1/\sigma_1, \ldots, 1/\sigma_r, 0, \ldots, 0)^T$.

**Geometric interpretation.** $x^* = A^+ b$ gives the **minimum-norm least-squares solution** to $Ax = b$:

- If $Ax = b$ has a unique solution: $A^+ b$ is that solution (same as $A^{-1}b$).
- If $Ax = b$ has multiple solutions (underdetermined): $A^+ b$ is the solution with smallest $\|x\|$.
- If $Ax = b$ has no exact solution (overdetermined): $A^+ b$ minimises $\|Ax - b\|^2$, and among all minimisers, is the one with smallest $\|x\|$.

**Special cases:**

| Matrix Type | Pseudo-Inverse |
|---|---|
| Square invertible ($m = n$, full rank) | $A^+ = A^{-1}$ |
| Tall full-rank ($m > n$, $\text{rank} = n$) — overdetermined | $A^+ = (A^T A)^{-1} A^T$ (left inverse) |
| Wide full-rank ($m < n$, $\text{rank} = m$) — underdetermined | $A^+ = A^T (A A^T)^{-1}$ (right inverse) |
| Rank-deficient | Use SVD formula |

**For AI.** The pseudo-inverse appears in:

- **Linear regression**: closed-form solution $\hat{\theta} = (X^TX)^{-1}X^Ty = X^+ y$
- **LoRA (Low-Rank Adaptation)**: weight update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ with $r \ll d$; the effective inverse involves pseudo-inverses of low-rank matrices
- **Weight initialisation analysis**: understanding the effective rank and conditioning of weight matrices
- **Gradient computation**: when computing gradients through rank-deficient Jacobians

### 5.5 Invertibility and Information Preservation

**Theorem.** A function is bijective (invertible) if and only if it preserves all information — i.e., the input can be recovered from the output.

**Linear case.** For a linear map $f(x) = Ax$ with $A \in \mathbb{R}^{m \times n}$:

| Property | Condition | Meaning |
|---|---|---|
| Injective | $\ker(A) = \{0\}$; $\text{rank}(A) = n$; columns linearly independent | Nothing in the kernel; no information destroyed |
| Surjective | $\text{range}(A) = \mathbb{R}^m$; $\text{rank}(A) = m$; rows span $\mathbb{R}^m$ | Full range; all outputs reachable |
| Bijective | $m = n$ and $\text{rank}(A) = n$ ($\det(A) \neq 0$) | Square and full rank |

**The dimension argument.** By the rank-nullity theorem ($\text{rank}(A) + \text{nullity}(A) = n$):

- If $n > m$ (more inputs than outputs): $\text{rank}(A) \leq m < n$, so $\text{nullity}(A) > 0$, so $f$ is NOT injective. The layer necessarily compresses.
- If $n < m$ (fewer inputs than outputs): $\text{rank}(A) \leq n < m$, so $f$ is NOT surjective. Not all outputs are reachable.
- Only when $n = m$ can $f$ be bijective (and even then, only if $\det(A) \neq 0$).

**Consequences for neural network architecture:**

```
                    Information Flow in a Network
                    
     d=768         d=768         d=768         d=768
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │ Layer 1│───▶│ Layer 2│───▶│ Layer 3│───▶│ Layer 4│
  └────────┘    └────────┘    └────────┘    └────────┘
  Same dimension throughout → bijection is POSSIBLE
  (Transformer: d_model stays constant through all layers)
  
     d=784        d=256          d=64          d=10
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │Compress│───▶│Compress│───▶│Compress│───▶│ Output │
  └────────┘    └────────┘    └────────┘    └────────┘
  Decreasing dimension → NOT injective → information lost
  (Classifier: must compress to number of classes)
```

**Autoencoders** exploit this: the encoder maps high dimension to low (non-injective, compresses, loses information), and the decoder maps low to high (non-surjective, reconstructs). The bottleneck forces the network to learn a compact representation preserving the most important information.

**Invertible neural networks** (iRevNet, Glow, NICE) maintain $d = d'$ throughout and use carefully designed bijective layers (coupling layers, additive/affine transforms), enabling exact likelihood computation and perfect reconstruction.

---


## 6. Special Classes of Functions

### 6.1 Linear Functions

**Definition.** A function $f: V \to W$ between vector spaces is **linear** if it preserves vector addition and scalar multiplication:

1. **Additivity**: $f(u + v) = f(u) + f(v)$ for all $u, v \in V$
2. **Homogeneity**: $f(\alpha v) = \alpha f(v)$ for all $\alpha \in \mathbb{R}$, $v \in V$

These two conditions can be combined into one: $f(\alpha u + \beta v) = \alpha f(u) + \beta f(v)$ (preservation of linear combinations).

**Key consequences:**
- $f(0) = 0$ (the zero vector must map to the zero vector)
- $f$ is completely determined by its action on a basis: if $\{e_1, \ldots, e_n\}$ is a basis for $V$, then $f(v) = f(\sum \alpha_i e_i) = \sum \alpha_i f(e_i)$
- Every linear function $f: \mathbb{R}^n \to \mathbb{R}^m$ can be represented as matrix multiplication: $f(x) = Ax$ where $A \in \mathbb{R}^{m \times n}$

**The kernel (null space).** $\ker(f) = \{v \in V : f(v) = 0\}$ — the set of vectors that $f$ maps to zero.

**Rank-Nullity Theorem.** For $f: V \to W$ with $V$ finite-dimensional:

$$\dim(V) = \text{rank}(f) + \text{nullity}(f) = \dim(\text{range}(f)) + \dim(\ker(f))$$

This is a conservation law: dimension is neither created nor destroyed, just redistributed between the range and the kernel.

**For AI.** Linear layers $f(x) = Wx$ (without bias) are the backbone of neural networks:
- The attention Q, K, V projections are linear maps
- The FFN up/down projections are linear maps
- The LM head (hidden state → logits) is a linear map
- LoRA approximates weight updates with low-rank linear maps: $\Delta W = BA$ where $\text{rank}(BA) = r \ll d$

Without nonlinear activation functions, composing any number of linear layers gives a single linear layer: $W_L \cdots W_2 W_1 = W_{\text{total}}$. This is why activations are essential — they break the linearity.

### 6.2 Affine Functions

**Definition.** A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **affine** if:

$$f(x) = Ax + b$$

where $A \in \mathbb{R}^{m \times n}$ is a matrix and $b \in \mathbb{R}^m$ is a bias vector.

**Affine = linear + translation.** An affine function is a linear function followed by a translation (shift by $b$). It preserves affine combinations: $f(\alpha x + (1-\alpha) y) = \alpha f(x) + (1-\alpha) f(y)$ (preserves weighted averages where weights sum to 1).

**Key difference from linear:**
- Linear: $f(0) = 0$ always. Lines through the origin map to lines through the origin.
- Affine: $f(0) = b \neq 0$ in general. Lines map to lines, but the origin can shift.

**Composition of affine functions is affine:**

$$f_2(f_1(x)) = A_2(A_1 x + b_1) + b_2 = (A_2 A_1)x + (A_2 b_1 + b_2)$$

Result is affine with matrix $A_2 A_1$ and bias $A_2 b_1 + b_2$.

**Critical consequence.** Composing any number of affine functions (without nonlinear activations) gives a single affine function. A 100-layer network with only affine layers is equivalent to a single affine layer. This is why nonlinear activation functions are essential — they prevent the entire network from collapsing into a single affine transformation.

**For AI.** Nearly every "linear layer" in a neural network is actually affine: $f(x) = Wx + b$. The bias term $b$ allows the function to shift the output away from the origin, which is necessary for representing functions that do not pass through zero. Some architectures omit the bias (e.g., some attention projections in LLaMA), but most include it.

### 6.3 Multilinear Functions

**Definition.** A function $f: V_1 \times V_2 \times \cdots \times V_k \to W$ is **multilinear** ($k$-linear) if it is linear in each argument separately, with the others held fixed:

$$f(v_1, \ldots, \alpha v_i + \beta v_i', \ldots, v_k) = \alpha f(v_1, \ldots, v_i, \ldots, v_k) + \beta f(v_1, \ldots, v_i', \ldots, v_k)$$

The most important case is **bilinear** ($k = 2$):

$$f: V \times W \to U, \quad f(\alpha v_1 + \beta v_2, w) = \alpha f(v_1, w) + \beta f(v_2, w)$$
$$f(v, \alpha w_1 + \beta w_2) = \alpha f(v, w_1) + \beta f(v, w_2)$$

**Key property.** A bilinear function is linear in each argument but NOT linear overall:

$$f(\alpha v, \alpha w) = \alpha^2 f(v, w) \neq \alpha f(v, w) \quad \text{(unless } \alpha = 0 \text{ or } 1\text{)}$$

**Common bilinear operations in AI:**

| Operation | Bilinear Map | Variables |
|---|---|---|
| **Dot product** | $f(u, v) = u^T v$ | Linear in $u$, linear in $v$ |
| **Matrix-vector product** | $f(A, x) = Ax$ | Linear in $A$, linear in $x$ |
| **Attention scores** | $f(Q, K) = QK^T$ | Linear in $Q$, linear in $K$ |
| **Outer product** | $f(u, v) = uv^T$ | Linear in $u$, linear in $v$ |
| **Bilinear form** | $f(x, y) = x^T A y$ | Linear in $x$, linear in $y$ |

**For AI: Attention is bilinear (before softmax).** The raw attention score between query $q$ and key $k$ is:

$$\text{score}(q, k) = \frac{q^T k}{\sqrt{d_k}}$$

This is bilinear: linear in $q$ (doubling $q$ doubles the score) and linear in $k$ (doubling $k$ doubles the score). The softmax that follows makes the full attention mechanism nonlinear.

### 6.4 Polynomial Functions

**Definition.** A polynomial function $f: \mathbb{R} \to \mathbb{R}$ has the form:

$$f(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0 = \sum_{k=0}^{n} a_k x^k$$

The **degree** of the polynomial is $n$ (the highest power with non-zero coefficient).

**Weierstrass Approximation Theorem.** Every continuous function on a closed interval $[a, b]$ can be uniformly approximated by polynomials. That is, for any continuous $f: [a,b] \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a polynomial $p$ with $|f(x) - p(x)| < \varepsilon$ for all $x \in [a,b]$.

This is a classical universal approximation result — but it requires the degree (number of terms) to grow, and in general, the required degree grows quickly for complex functions.

**Multivariate polynomials.** A polynomial in $n$ variables:

$$f(x_1, \ldots, x_n) = \sum_{\alpha} c_\alpha x_1^{\alpha_1} x_2^{\alpha_2} \cdots x_n^{\alpha_n}$$

where $\alpha = (\alpha_1, \ldots, \alpha_n)$ is a multi-index.

**Connection to neural networks (depth separation).** Telgarsky (2016) showed that there exist functions expressible by depth-$k$ ReLU networks with polynomial width that require **exponential** width in depth-$(k-1)$ networks. This is analogous to the fact that high-degree polynomials can represent functions that low-degree polynomials cannot.

**Why neural networks don't use polynomial activations.** If the activation function $\sigma(x) = x^k$ were polynomial, then composing affine layers with polynomial activations gives a polynomial of degree $k^L$ (where $L$ is depth). While high-degree polynomials are universal approximators, they have catastrophic extrapolation behaviour: $x^{100}$ explodes for $|x| > 1$ and vanishes for $|x| < 1$. ReLU networks avoid this by producing piecewise linear functions, which extrapolate more gracefully.

### 6.5 Activation Functions

Activation functions are the nonlinear components that give neural networks their expressive power. Without them, any deep network collapses to a single affine transformation.

**Detailed analysis of key activation functions:**

**Sigmoid:**

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

| Property | Value |
|---|---|
| Range | $(0, 1)$ |
| Derivative | $\sigma'(x) = \sigma(x)(1 - \sigma(x))$ |
| Max derivative | $1/4$ at $x = 0$ |
| Monotone | Strictly increasing |
| Injective | Yes |
| Lipschitz constant | $1/4$ |
| Bounded | Yes |
| Symmetric | About $(0, 0.5)$: $\sigma(-x) = 1 - \sigma(x)$ |
| Saturation | Gradients vanish for $|x| \gg 0$: $\sigma'(x) \to 0$ |

**Problem.** Max derivative is $1/4$. After $L$ layers, gradient is at most $(1/4)^L$. For $L = 20$: $(1/4)^{20} \approx 10^{-12}$. This is the **vanishing gradient problem** — the reason sigmoid was abandoned as a hidden-layer activation in deep networks.

**Tanh:**

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

| Property | Value |
|---|---|
| Range | $(-1, 1)$ |
| Derivative | $\tanh'(x) = 1 - \tanh^2(x)$ |
| Max derivative | $1$ at $x = 0$ |
| Monotone | Strictly increasing |
| Zero-centred | Yes: $\tanh(0) = 0$ and $\tanh(-x) = -\tanh(x)$ |
| Saturation | Still saturates for large $|x|$, but max derivative is $1$ (better than sigmoid's $1/4$) |

**ReLU (Rectified Linear Unit):**

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$$

| Property | Value |
|---|---|
| Range | $[0, \infty)$ |
| Derivative | $1$ for $x > 0$; $0$ for $x < 0$; undefined at $x = 0$ |
| Monotone | Non-decreasing (not strict) |
| Injective | No (all negatives map to 0) |
| Lipschitz constant | $1$ |
| Bounded | No (unbounded above) |
| Computation | Extremely cheap: just a comparison |
| Dying ReLU | If pre-activation is always negative, gradient is always 0 — neuron is "dead" |
| Sparsity | Roughly 50% of neurons output 0, creating sparse representations |

**Why ReLU dominates.** Despite losing information (not injective) and not being differentiable at 0, ReLU works extremely well because:
1. Gradient is exactly 1 for positive inputs — no vanishing gradient
2. Creates piecewise linear functions — efficient to compute and optimise
3. Induces sparsity — only active neurons contribute
4. Extremely cheap to compute

**GELU (Gaussian Error Linear Units, Hendrycks & Gimpel 2016):**

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi$ is the standard normal CDF. Approximation: $\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{2/\pi}(x + 0.044715 x^3)\right]\right)$.

| Property | Value |
|---|---|
| Range | $[\approx -0.17, \infty)$ |
| Monotone | No (slight dip near $x \approx -0.17$) |
| Smooth | Yes ($C^\infty$ — infinitely differentiable) |
| Used in | GPT-2, BERT, most transformer models before SwiGLU era |

**SwiGLU (Shazeer 2020, used in LLaMA, PaLM, Gemini):**

$$\text{SwiGLU}(x, W_1, W_2, W_3) = \text{SiLU}(xW_1) \odot (xW_2)$$

where $\text{SiLU}(x) = x \cdot \sigma(x)$ (Swish activation) and $\odot$ is elementwise multiplication.

| Property | Value |
|---|---|
| Gated | Yes — one path gates the other via sigmoid |
| Parameters | Requires 3 weight matrices instead of 2 (50% more parameters per FFN) |
| Used in | LLaMA 1/2/3, PaLM, Gemini, Mistral, most 2023-2026 LLMs |
| Why preferred | Empirically better than GELU for large-scale language models |

**Mish (Misra 2019):**

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

| Property | Value |
|---|---|
| Smooth | Yes ($C^\infty$) |
| Non-monotone | Slight negative dip (similar to GELU) |
| Self-regularising | Bounded below, unbounded above |
| Used in | YOLOv4, some vision models |

**Comprehensive comparison table:**

| Activation | Formula | Range | Smooth? | Monotone? | Injective? | Lipschitz $L$ | Vanishing grad? |
|---|---|---|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | $(0,1)$ | Yes | Yes | Yes | $1/4$ | Yes (severe) |
| Tanh | $\tanh(x)$ | $(-1,1)$ | Yes | Yes | Yes | $1$ | Yes (moderate) |
| ReLU | $\max(0,x)$ | $[0,\infty)$ | No ($x=0$) | Yes (weak) | No | $1$ | No (but dying neurons) |
| Leaky ReLU | $\max(\alpha x, x)$ | $\mathbb{R}$ | No ($x=0$) | Yes | Yes | $1$ | No |
| GELU | $x\Phi(x)$ | $\approx[-0.17,\infty)$ | Yes | No | No | $\approx 1.1$ | No |
| SiLU/Swish | $x\sigma(x)$ | $\approx[-0.28,\infty)$ | Yes | No | No | $\approx 1.1$ | No |
| Mish | $x\tanh(\text{sp}(x))$ | $\approx[-0.31,\infty)$ | Yes | No | No | $\approx 1.0$ | No |
| Softplus | $\ln(1+e^x)$ | $(0,\infty)$ | Yes | Yes | Yes | $1$ | No |

### 6.6 Convex Functions

**Definition.** A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $x, y \in \text{dom}(f)$ and $\lambda \in [0, 1]$:

$$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$$

Geometrically: the line segment connecting any two points on the graph lies above (or on) the graph. The function "curves upward".

```
         f(x)
          │    ╱
          │   ╱  Line segment (always above curve)
          │  ·╱·
          │ ╱ · 
          │╱  ·  ← f curves below the line
          ╱   ·    (this is convexity)
         ╱────·───────── x
```

**Three equivalent characterisations (for differentiable $f$):**

1. **Definition**: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$
2. **First-order condition**: $f(y) \geq f(x) + \nabla f(x)^T(y - x)$ (function lies above every tangent plane)
3. **Second-order condition**: $\nabla^2 f(x) \succeq 0$ (Hessian is positive semi-definite everywhere)

**Strict convexity**: replace $\leq$ with $<$ (for $x \neq y$ and $\lambda \in (0,1)$). Strict convexity implies a unique global minimum.

**Strong convexity**: $f(y) \geq f(x) + \nabla f(x)^T(y - x) + \frac{\mu}{2}\|y - x\|^2$ for some $\mu > 0$. Strong convexity implies:
- Unique global minimum
- Gradient descent converges at rate $O(e^{-\mu t})$ (exponential)
- The Hessian satisfies $\nabla^2 f(x) \succeq \mu I$

**Why convexity matters for optimisation:**

| Property | Convex $f$ | Non-convex $f$ |
|---|---|---|
| Local minima | Every local minimum is global | Can have many local minima |
| Gradient descent | Guaranteed to converge to global min | May converge to local min or saddle |
| Uniqueness | Strict convex → unique minimum | Multiple minima possible |
| Rate of convergence | $O(1/t)$ or $O(e^{-\mu t})$ | No guarantees |

**For AI.** The loss landscape of neural networks is **non-convex** in the parameters $\theta$. However:
- Cross-entropy loss is **convex in the logits** $z$ (for fixed true labels): $L(z) = -z_y + \log\sum e^{z_v}$ is convex because log-sum-exp is convex
- The $L_2$ regularisation term $\frac{\lambda}{2}\|\theta\|^2$ is **strongly convex** in $\theta$
- Adding regularisation improves the conditioning of the loss landscape

Despite non-convexity, SGD finds good solutions in practice. This is one of the central mysteries of deep learning theory.

### 6.7 Lipschitz Functions

**Definition.** A function $f: X \to Y$ (between metric spaces) is **Lipschitz continuous** with constant $L \geq 0$ if:

$$d_Y(f(x_1), f(x_2)) \leq L \cdot d_X(x_1, x_2) \quad \text{for all } x_1, x_2 \in X$$

In $\mathbb{R}^n$ with Euclidean norm: $\|f(x_1) - f(x_2)\| \leq L \|x_1 - x_2\|$.

**Intuition.** A Lipschitz function cannot change its output faster than $L$ times the change in input. The constant $L$ is a bound on the "slope" or "sensitivity" of the function.

**The smallest valid $L$ is the Lipschitz constant:**

$$\text{Lip}(f) = \sup_{x_1 \neq x_2} \frac{\|f(x_1) - f(x_2)\|}{\|x_1 - x_2\|}$$

For differentiable functions: $\text{Lip}(f) = \sup_x \|Jf(x)\|_{\text{op}}$ (supremum of the operator norm of the Jacobian).

**Hierarchy:**

$$\text{Contraction} \subsetneq \text{Lipschitz} \subsetneq \text{Uniformly Continuous} \subsetneq \text{Continuous}$$

(Contraction: $L < 1$. Lipschitz: $L < \infty$.)

**Lipschitz constants for common AI functions:**

| Function | Lipschitz Constant | Notes |
|---|---|---|
| ReLU | $1$ | $|\text{ReLU}(x) - \text{ReLU}(y)| \leq |x - y|$ |
| Sigmoid | $1/4$ | Max derivative at $x = 0$ |
| Tanh | $1$ | Max derivative at $x = 0$ |
| Softmax | $1$ | In $\ell^1$ norm |
| Linear $Wx$ | $\|W\|_{\text{op}} = \sigma_{\max}(W)$ | Largest singular value |
| Layer Norm | $\leq \sqrt{d}$ (approximately) | Depends on the input distribution |
| Affine $Wx + b$ | $\|W\|_{\text{op}}$ | Bias does not affect Lipschitz constant |

**Lipschitz constant of a composition.** If $f$ has Lipschitz constant $L_f$ and $g$ has Lipschitz constant $L_g$, then $g \circ f$ has Lipschitz constant at most $L_g \cdot L_f$:

$$\|g(f(x)) - g(f(y))\| \leq L_g \|f(x) - f(y)\| \leq L_g L_f \|x - y\|$$

**For a deep network with $L$ layers, each with Lipschitz constant $L_l$:**

$$\text{Lip}(f_{\text{network}}) \leq \prod_{l=1}^{L} L_l$$

If each layer has Lipschitz constant $> 1$, the bound grows exponentially with depth — a small input perturbation could be amplified exponentially. This is exactly the phenomenon behind **adversarial examples**: a tiny, imperceptible change to the input causes a large change in the output.

**Spectral normalisation (Miyato et al. 2018).** To control the Lipschitz constant of each layer, divide the weight matrix by its largest singular value:

$$\tilde{W} = \frac{W}{\sigma_{\max}(W)}$$

This ensures $\|\tilde{W}\|_{\text{op}} = 1$, so each layer has Lipschitz constant $\leq 1$ (when composed with a 1-Lipschitz activation like ReLU), and the overall network has Lipschitz constant $\leq 1^L = 1$.

**Applications of Lipschitz constraints in AI:**
- **Wasserstein GANs (WGAN)**: The discriminator must be 1-Lipschitz; enforced via spectral normalisation or gradient penalty
- **Certified robustness**: A network with Lipschitz constant $L$ is guaranteed to be robust to perturbations of size $\varepsilon/L$ — no adversarial example can flip the prediction within this radius
- **Training stability**: Constraining layer Lipschitz constants prevents gradient explosion
- **ODE-based models**: Neural ODEs require Lipschitz dynamics for well-posedness (existence and uniqueness of solutions)

---


## 7. Higher-Order Functions and Functionals

### 7.1 Functions as Arguments

**Definition.** A **higher-order function** is a function that takes one or more functions as arguments, returns a function as output, or both.

In mathematical notation, if $\mathcal{F}$ is a set of functions, a higher-order function has the form:

$$H: \mathcal{F} \to \mathcal{G} \quad \text{or} \quad H: \mathcal{F} \times X \to Y$$

**Formal framework.** Let $\text{Fun}(X, Y) = Y^X = \{f : X \to Y\}$ denote the set of all functions from $X$ to $Y$. A higher-order function is simply a function whose domain or codomain includes such function spaces.

**Examples in mathematics:**
- **Derivative operator**: $D: C^1(\mathbb{R}) \to C^0(\mathbb{R})$, $D(f) = f'$. Takes a differentiable function, returns its derivative.
- **Integration operator**: $I_a^b: C([a,b]) \to \mathbb{R}$, $I_a^b(f) = \int_a^b f(x)\,dx$. Takes a continuous function, returns a number.
- **Composition operator**: $C_g: \text{Fun}(X, Y) \to \text{Fun}(X, Z)$, $C_g(f) = g \circ f$. Takes a function, returns a function.

**Examples in programming (Python/JAX):**

```
# map applies a function to each element
map(f, [x1, x2, x3]) → [f(x1), f(x2), f(x3)]

# JAX grad: takes a function, returns its gradient function
grad_f = jax.grad(f)  # grad: (R^n → R) → (R^n → R^n)

# JAX vmap: takes a function, returns its batched version  
batched_f = jax.vmap(f)  # vmap: (R^n → R^m) → (R^{B×n} → R^{B×m})
```

### 7.2 Functionals (Functions of Functions)

**Definition.** A **functional** is a function from a function space to the real numbers:

$$J: \mathcal{F} \to \mathbb{R}$$

where $\mathcal{F}$ is a set of functions. A functional maps an entire function to a single number.

**Key examples:**

| Functional | Definition | What It Measures |
|---|---|---|
| **Definite integral** | $J(f) = \int_a^b f(x)\,dx$ | Area under the curve |
| **$L^p$ norm** | $J(f) = \left(\int |f(x)|^p\,dx\right)^{1/p}$ | Size of the function |
| **Evaluation** | $J(f) = f(x_0)$ | Value at a specific point |
| **Supremum norm** | $J(f) = \sup_x |f(x)|$ | Maximum absolute value |
| **Entropy** | $H(p) = -\int p(x) \log p(x)\,dx$ | Uncertainty/information |
| **KL divergence** | $D_{KL}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)}\,dx$ | Distance between distributions |

**Loss functions as functionals.** In machine learning, the loss function is a functional on the space of models:

$$\mathcal{L}: \mathcal{F} \to \mathbb{R}, \quad \mathcal{L}(f) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f(x), y)]$$

This maps each model $f$ to a single number measuring how well $f$ fits the data. Training is the process of minimising this functional over the space of representable functions.

### 7.3 Operators (Functions on Function Spaces)

**Definition.** An **operator** is a function from one function space to another:

$$T: \mathcal{F} \to \mathcal{G}$$

Unlike functionals (which output numbers), operators output functions.

**Linear operators.** An operator $T$ is linear if $T(\alpha f + \beta g) = \alpha T(f) + \beta T(g)$. These are the infinite-dimensional analogue of matrices.

**Key examples of operators:**

| Operator | Definition | Type |
|---|---|---|
| **Derivative** | $D(f) = f'$ | Linear |
| **Integration** | $I(f)(x) = \int_0^x f(t)\,dt$ | Linear |
| **Fourier transform** | $\hat{f}(\omega) = \int f(t) e^{-i\omega t}\,dt$ | Linear |
| **Convolution** | $(T_g f)(x) = \int g(x-t)f(t)\,dt$ | Linear (for fixed $g$) |
| **Composition** | $C_g(f) = g \circ f$ | Generally nonlinear |
| **Softmax** | $\text{softmax}(f)_i = \frac{e^{f_i}}{\sum e^{f_j}}$ | Nonlinear |

**For AI: Neural networks as operators.** Recent work treats neural networks as operators between function spaces:

- **Neural Operators (FNO, DeepONet)**: Learn mappings between function spaces directly, e.g., mapping initial conditions to PDE solutions
- **Attention as an operator**: Self-attention can be viewed as a nonlinear operator on a sequence of vectors: $\text{Attn}: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$, where $n$ varies (the operator handles variable-length inputs)

### 7.4 Calculus of Variations

**Problem.** Find the function $f$ that minimises (or maximises) a functional $J[f]$.

This is optimisation over function spaces rather than over finite-dimensional parameter vectors. The classical framework is the **calculus of variations**.

**Euler-Lagrange equation.** Consider the functional:

$$J[f] = \int_a^b \mathcal{L}(x, f(x), f'(x))\,dx$$

where $\mathcal{L}$ is the **Lagrangian**. The function $f^*$ that minimises $J$ satisfies:

$$\frac{\partial \mathcal{L}}{\partial f} - \frac{d}{dx}\frac{\partial \mathcal{L}}{\partial f'} = 0$$

This is the **Euler-Lagrange equation** — an ODE whose solutions are the critical points of the functional.

**Example: shortest path.** The length of a curve $y = f(x)$ from $(a, f(a))$ to $(b, f(b))$ is:

$$J[f] = \int_a^b \sqrt{1 + (f'(x))^2}\,dx$$

The Euler-Lagrange equation gives $f''(x) = 0$, meaning the shortest path is a straight line $f(x) = cx + d$. (In Euclidean geometry.)

**Functional derivative.** The analogue of a gradient for functionals:

$$\frac{\delta J}{\delta f}(x) = \lim_{\varepsilon \to 0} \frac{J[f + \varepsilon \delta_x] - J[f]}{\varepsilon}$$

This measures how $J$ changes when $f$ is perturbed at the point $x$.

**Connection to machine learning.** Training a neural network by gradient descent on parameters $\theta$ is a finite-dimensional approximation to functional optimisation:

```
Functional view:     min_{f ∈ F}     L(f)        ← optimise over all functions
                            ↑
                     Parameterise f_θ
                            ↓
Parameter view:      min_{θ ∈ R^p}   L(f_θ)      ← optimise over parameters
```

The "lottery ticket hypothesis" and neural architecture search can be seen as trying to find the right parameterisation of the function space.

### 7.5 Transformations in AI Frameworks

Modern deep learning frameworks extensively use higher-order functions. Here is a conceptual map:

**JAX's transformation system:**

| Transform | Type Signature | Description |
|---|---|---|
| `grad` | $(f: \mathbb{R}^n \to \mathbb{R}) \to (g: \mathbb{R}^n \to \mathbb{R}^n)$ | Returns gradient function |
| `vmap` | $(f: X \to Y) \to (g: X^B \to Y^B)$ | Vectorises (auto-batches) |
| `jit` | $(f: X \to Y) \to (g: X \to Y)$ | Compiles for speed |
| `pmap` | $(f: X \to Y) \to (g: X^D \to Y^D)$ | Parallelises across devices |

**The power of composability:** These transforms compose:

$$\text{jit}(\text{vmap}(\text{grad}(f)))$$

This gives a compiled, batched gradient function — three layers of higher-order functions applied in sequence. Each transform takes a function and returns a new, enhanced function.

**Autodiff as a higher-order function.** Automatic differentiation (the engine behind backpropagation) is fundamentally a higher-order function:

$$\text{AD}: (f: \mathbb{R}^n \to \mathbb{R}^m) \to (\nabla f: \mathbb{R}^n \to \mathbb{R}^{m \times n})$$

It takes any differentiable function and returns its Jacobian (or gradient, for scalar outputs). The reverse-mode variant computes gradients in $O(\text{cost}(f))$ time regardless of the number of parameters — this is why training networks with billions of parameters is feasible.

---


## 8. Continuity and Limits

This chapter only needs the core idea of continuity: functions can preserve nearby structure instead of tearing inputs apart with abrupt jumps. The full story belongs to [Limits and Continuity](../../04-Calculus-Fundamentals/01-Limits-and-Continuity/notes.md), where epsilon-delta reasoning, theorem statements, and examples are treated systematically.

### 8.1 The Epsilon-Delta Definition

A function is **continuous at $a$** when sufficiently small input changes force small output changes:

$$\forall \varepsilon > 0,\; \exists \delta > 0:\; |x-a| < \delta \implies |f(x)-f(a)| < \varepsilon$$

You do not need to master epsilon-delta proofs yet. What matters in this chapter is the interpretation: continuity is a structural property of a function, not a plotting aesthetic. It tells us whether a function respects local neighborhoods.

### 8.2 Types of Continuity

Ordinary continuity is pointwise: the allowed $\delta$ may depend on the location $a$. **Uniform continuity** is stronger: one choice of $\delta$ works across the whole set. **Lipschitz continuity** is stronger still and gives an explicit bound on how much outputs can move relative to inputs.

For AI, this hierarchy matters because robustness, sensitivity, and optimization stability are all function-shape questions. When we say a layer is "well behaved," we usually mean something in this family.

### 8.3 Continuity in Higher Dimensions

For $f: \mathbb{R}^n \to \mathbb{R}^m$, continuity is stated with norms:

$$\forall \varepsilon > 0,\; \exists \delta > 0:\; \|x-a\| < \delta \implies \|f(x)-f(a)\| < \varepsilon$$

This is the version that matters for neural networks, where inputs, hidden states, and parameters all live in high-dimensional spaces. A small step in parameter space should not cause an uncontrolled jump in loss unless the model includes genuinely discontinuous operations.

### 8.4 Limits and Convergence

Limits describe what a function approaches near a point, even if the function value at that point is missing or different. Continuity can be summarized as "the function value agrees with the limiting value."

This perspective is especially useful for activation functions and optimization trajectories: sigmoid and tanh saturate toward limiting values, while ReLU becomes asymptotically linear on the positive side. We will return to these ideas in [Limits and Continuity](../../04-Calculus-Fundamentals/01-Limits-and-Continuity/notes.md).

### 8.5 Discontinuities and Their Consequences

Not every useful AI function is smooth or even continuous. `argmax`, hard thresholding, rounding, and hard attention all introduce abrupt changes. That is why so much modern ML relies on soft relaxations such as softmax, sigmoid gates, or temperature-controlled approximations.

The important bridge for this chapter is simple: continuity is one of the first major properties that distinguishes function classes in a way that directly affects trainability and robustness.

---


## 9. Differentiable Functions and Derivatives

Differentiability is the next major refinement: not only does the function avoid jumps, it admits a meaningful local linear approximation. The detailed machinery lives in [Derivatives and Differentiation](../../04-Calculus-Fundamentals/02-Derivatives-and-Differentiation/notes.md), [Partial Derivatives and Gradients](../../05-Multivariate-Calculus/01-Partial-Derivatives-and-Gradients/notes.md), [Chain Rule and Backpropagation](../../05-Multivariate-Calculus/03-Chain-Rule-and-Backpropagation/notes.md), and [Automatic Differentiation](../../05-Multivariate-Calculus/05-Automatic-Differentiation/notes.md).

### 9.1 Derivative as a Linear Approximation

For a scalar function, the derivative at $a$ is the limit

$$f'(a) = \lim_{h \to 0}\frac{f(a+h)-f(a)}{h}$$

when that limit exists. In higher dimensions, the same idea becomes the gradient or Jacobian: the best linear map that approximates the function near a point. This "best local linear model" viewpoint is the right mental model for optimization.

### 9.2 Smoothness Classes

Analysts organize functions by regularity: continuous functions, continuously differentiable functions, twice differentiable functions, and so on. In ML terms, this is the difference between kinked activations such as ReLU and smoother choices such as GELU or softplus.

You do not need the full taxonomy yet. The important lesson is that stronger regularity assumptions buy stronger theorems, but many practical models deliberately work at the edge of those assumptions because piecewise-linear functions are computationally convenient.

### 9.3 The Chain Rule and Backpropagation

Differentiation becomes operationally important because neural networks are compositions:

$$f = f_L \circ \cdots \circ f_2 \circ f_1$$

The **chain rule** tells us how derivatives of compositions combine, and backpropagation is the computational form of that rule. This is the main reason the function-composition viewpoint earlier in the chapter matters so much.

### 9.4 Gradient Problems in Deep Networks

Composing many layers can make gradients shrink, blow up, or become numerically unstable. Vanishing and exploding gradients are not separate mysteries; they are consequences of how many Jacobian-like factors are multiplied together through depth.

Residual connections, normalization, initialization schemes, and gradient clipping are all function-shaping interventions designed to keep this derivative information usable.

### 9.5 Automatic Differentiation

Modern frameworks do not symbolically differentiate full formulas by hand. They build a computation graph and apply local derivative rules mechanically. That is **automatic differentiation**, and it is what turns abstract calculus into a practical training system.

For this section, the main bridge is conceptual: once functions are composed from simple pieces, their derivatives can also be assembled systematically. That idea powers nearly all large-scale deep learning.

---


## 10. Measure-Theoretic View of Functions

Functions eventually need to interact with probability, integration, and "almost everywhere" reasoning. This chapter only needs a preview so the vocabulary will feel familiar later. The closest developed companions in the current curriculum are [Introduction to Probability and Random Variables](../../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md), [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md), and [Integration](../../04-Calculus-Fundamentals/03-Integration/notes.md); the dedicated measure-theory chapter sequence is still mostly scaffolded.

### 10.1 Why Measure Theory?

Probability theory needs a rigorous way to say which subsets count as events, which functions are valid random variables, and when an expectation is well defined. Measure theory supplies that language.

From the function viewpoint, the key upgrade is that we stop asking only "what output does this function produce?" and start asking "how does this function interact with a measure on its domain?"

### 10.2 Measurable Functions

A function between measurable spaces is **measurable** when preimages of measurable sets remain measurable. This is the measure-theoretic analogue of continuity preserving open-set structure.

That condition is what lets us push probabilities through a function and talk about the distribution of outputs in a mathematically controlled way.

### 10.3 Random Variables as Measurable Functions

A random variable is not fundamentally a mysterious object that "has a distribution." It is a measurable function

$$X : (\Omega, \mathcal{F}, P) \to (\mathbb{R}, \mathcal{B}(\mathbb{R}))$$

from outcomes to values. The distribution is the pushforward of $P$ through $X$. This perspective becomes especially helpful when we later treat generative models as maps that transform one distribution into another.

### 10.4 Lebesgue Integration

Expectation is integration with respect to a probability measure. Lebesgue integration is the framework that makes this precise for very general classes of functions, far beyond the simple Riemann-integrable cases you first see in basic calculus.

For ML, this matters because loss expectations, likelihoods, densities, and interchange-of-limit arguments all rest on integration theory, even when the notation looks innocently familiar.

### 10.5 Almost Everywhere and Null Sets

Many theorems in analysis and ML care only about what happens outside sets of measure zero. A property that holds **almost everywhere** may fail on a negligible exceptional set and still be enough for integration, optimization, and probability arguments.

This is why ReLU's non-differentiability at a single point is usually harmless in practice: the bad set is tiny in the measure-theoretic sense that later theory makes precise.

---


## 11. Functions in Machine Learning

### 11.1 Universal Approximation Theorems

**The fundamental question.** Can neural networks represent any function? The **universal approximation theorem** says yes — with sufficient width or depth.

**Width version (Cybenko 1989, Hornik 1991).** For any continuous function $f: [0,1]^n \to \mathbb{R}$ and any $\varepsilon > 0$, there exists a single-hidden-layer network:

$$g(x) = \sum_{i=1}^{N} \alpha_i \sigma(w_i^T x + b_i) + \beta$$

with $N$ sufficiently large (but finite), such that $\sup_{x \in [0,1]^n} |f(x) - g(x)| < \varepsilon$.

**Requirements on $\sigma$:**
- Original (Cybenko): $\sigma$ must be sigmoidal (monotone, $\sigma(x) \to 1$ as $x \to \infty$, $\sigma(x) \to 0$ as $x \to -\infty$)
- Extended (Hornik): Any continuous, non-polynomial $\sigma$ works
- ReLU: Specific constructions show that ReLU networks are universal approximators

**Depth version (Lu et al. 2017).** ReLU networks with width $n + 1$ (where $n$ is input dimension) but arbitrary depth are universal approximators. Width $\leq n$ is NOT sufficient — there exist continuous functions that cannot be approximated by arbitrarily deep but narrow ReLU networks.

**What the theorem does NOT say:**
- It does NOT guarantee that gradient descent can find the approximation
- It does NOT bound the number of parameters needed (could be astronomically large)
- It does NOT say anything about generalisation (fitting training data $\neq$ generalising)
- It does NOT say the approximation is efficient

**Depth separation results.** There exist functions that:
- Can be computed by depth-$k$ ReLU networks with polynomial width
- Require **exponential** width to approximate with depth-$(k-1)$ networks (Telgarsky 2016, Eldan & Shamir 2016)

This justifies deeper architectures: depth provides exponential efficiency gains in representation.

### 11.2 Loss Functions as Function Compositions

The training loss decomposes as a composition of functions:

```
Input x → Model f_θ(x) → Prediction ŷ → Loss ℓ(ŷ, y) → Scalar L
```

$$\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i)$$

**Common loss functions and their properties:**

| Loss | Formula | Domain → Range | Convex in pred? | Bounded? |
|---|---|---|---|---|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $\mathbb{R}^n \to [0, \infty)$ | Yes (strongly) | No |
| **Cross-entropy** | $-\sum y_i \log \hat{y}_i$ | $\Delta^k \to [0, \infty)$ | Yes | No ($\to \infty$ as $\hat{y}_i \to 0$) |
| **Hinge** | $\max(0, 1 - y\hat{y})$ | $\mathbb{R} \to [0, \infty)$ | Yes | No |
| **Huber** | MSE if $|e| < \delta$, MAE otherwise | $\mathbb{R} \to [0, \infty)$ | Yes | No |
| **KL divergence** | $\sum p_i \log(p_i/q_i)$ | $\Delta^k \times \Delta^k \to [0,\infty)$ | Yes (in $q$) | No |
| **Focal** | $-(1-\hat{y})^\gamma \log \hat{y}$ | $(0,1) \to [0, \infty)$ | Depends on $\gamma$ | No |

**Important subtlety: convexity in predictions vs. parameters.** Cross-entropy is convex in the logits $z$ (input to softmax), but the overall loss $\mathcal{L}(\theta)$ is **non-convex** in the parameters $\theta$ because $f_\theta$ is a non-convex function of $\theta$.

### 11.3 Hypothesis Classes and Function Spaces

**Definition.** A **hypothesis class** $\mathcal{H}$ is the set of functions that a learning algorithm can select from:

$$\mathcal{H} = \{f_\theta : \theta \in \Theta\}$$

The choice of architecture determines $\mathcal{H}$:

| Architecture | Hypothesis Class | Properties |
|---|---|---|
| **Linear regression** | $\{x \mapsto w^T x + b : w \in \mathbb{R}^d, b \in \mathbb{R}\}$ | Finite-dim, convex, small |
| **Polynomial degree $k$** | $\{x \mapsto \sum a_\alpha x^\alpha : |\alpha| \leq k\}$ | Finite-dim, convex, larger |
| **1-hidden-layer MLP** | $\{x \mapsto \sum \alpha_i \sigma(w_i^T x + b_i)\}$ | Finite-dim, non-convex |
| **Deep network** | Very complex function class | Finite-dim, highly non-convex |
| **Kernel methods** | $\{x \mapsto \sum \alpha_i K(x, x_i)\}$ | Can be infinite-dim, convex |
| **Transformer** | See below | Finite-dim, non-convex |

**The hypothesis class of a transformer.** A transformer with $L$ layers, $H$ heads, hidden dimension $d$, and vocabulary $V$ defines:

$$\mathcal{H}_{\text{transformer}} \subset \{f: V^* \to \Delta^{|V|}\}$$

This is a subset of all functions from variable-length token sequences to distributions over the vocabulary. The subset is determined by the architecture and parameter count.

**Bias-variance tradeoff through function space lens:**
- **Small $\mathcal{H}$** (few parameters): High bias (can't fit complex functions), low variance (stable across datasets)
- **Large $\mathcal{H}$** (many parameters): Low bias (can fit anything), but potentially high variance
- **Modern deep learning paradox**: Over-parameterised networks ($|\theta| \gg n$) have low bias AND low variance due to implicit regularisation by SGD

### 11.4 Attention as a Function

**Self-attention** defines a function $\text{Attn}: \mathbb{R}^{n \times d} \to \mathbb{R}^{n \times d}$ where $n$ is sequence length and $d$ is hidden dimension:

$$\text{Attn}(X) = \text{softmax}\left(\frac{XW_Q (XW_K)^T}{\sqrt{d_k}}\right) XW_V$$

**Decomposing the function:**

1. **Linear projections**: $Q = XW_Q$, $K = XW_K$, $V = XW_V$ (affine functions)
2. **Attention scores**: $S = QK^T / \sqrt{d_k}$ (bilinear function of $Q$ and $K$)
3. **Softmax**: $A = \text{softmax}(S)$ (nonlinear, applied row-wise)
4. **Weighted sum**: $O = AV$ (bilinear function of $A$ and $V$)

**Properties of attention as a function:**

| Property | Attention |
|---|---|
| **Linear?** | No (softmax makes it nonlinear) |
| **Injective?** | Generally yes (for distinct inputs) |
| **Equivariant?** | Yes — permuting input rows permutes output rows |
| **Lipschitz?** | Yes, but constant depends on sequence length |
| **Differentiable?** | Yes ($C^\infty$) |
| **Input size** | Variable ($n$ can change) |

**Attention is not a fixed function.** Unlike a typical neural network layer with fixed computational graph, attention's computation depends on the input through the attention pattern $A$. This makes it a **data-dependent** function: the effective "weight matrix" $A \cdot W_V$ changes with every input.

### 11.5 Tokenisation and Embedding as Functions

**Tokenisation** is a function from strings to token sequences:

$$\text{Tokenise}: \Sigma^* \to V^*$$

where $\Sigma$ is the character alphabet and $V$ is the token vocabulary. For BPE/SentencePiece tokenisation, this function is:
- **Deterministic** (same input always gives same output)
- **Not injective** in general (some tokenisers normalise text, losing information)
- **Not surjective** (not all token sequences correspond to valid text)

**Embedding** is a learned function from discrete tokens to continuous vectors:

$$\text{Embed}: V \to \mathbb{R}^d, \quad \text{Embed}(t) = E[t, :]$$

where $E \in \mathbb{R}^{|V| \times d}$ is the embedding matrix. This is:
- **Injective** (distinct tokens get distinct vectors — assuming no hash collisions)
- **Not continuous** (the domain $V$ is discrete, so continuity doesn't apply)
- A **lookup table**, not a "computation" — but mathematically it's a function

**Complete pipeline as function composition:**

```
String ──→ Tokens ──→ Embeddings ──→ + Position ──→ Transformer ──→ Logits ──→ Probs
       Tokenise    Embed          PosEnc         f_θ           LM Head    Softmax
```

$$P(\text{next token} | \text{context}) = \text{softmax}(W_{\text{head}} \cdot f_\theta(\text{Embed}(\text{Tokenise}(\text{text})) + \text{PE}))$$

### 11.6 The Transformer Block as a Function

A single Pre-LN transformer block (as used in GPT-2+, LLaMA, etc.) defines a residual function:

$$\text{Block}(X) = X + \text{FFN}(\text{LN}_2(X + \text{Attn}(\text{LN}_1(X))))$$

Breaking this down:

| Step | Function | Type |
|---|---|---|
| 1. Layer Norm | $\text{LN}_1(X)$ | Nonlinear (normalisation) |
| 2. Self-Attention | $\text{Attn}(\cdot)$ | Nonlinear (softmax), data-dependent |
| 3. Residual Add | $X + \text{Attn}(\text{LN}_1(X))$ | Linear in $X$ and in attention output |
| 4. Layer Norm | $\text{LN}_2(\cdot)$ | Nonlinear |
| 5. FFN | $W_2 \cdot \sigma(W_1 \cdot (\cdot))$ or SwiGLU variant | Nonlinear (activation function) |
| 6. Residual Add | $(\cdot) + \text{FFN}(\cdot)$ | Linear |

**The full transformer** stacks $L$ such blocks:

$$f_\theta = \text{LN}_{\text{final}} \circ \text{Block}_L \circ \text{Block}_{L-1} \circ \cdots \circ \text{Block}_1$$

**Parameter count and function expressiveness.** For a transformer with hidden dimension $d$, $L$ layers, $H$ heads, FFN dimension $d_{\text{ff}} = 4d$, and vocabulary $|V|$:

$$|\theta| \approx L(12d^2) + |V| \cdot d \approx 12Ld^2$$

For GPT-3 (175B): $L = 96$, $d = 12288$, giving $\approx 12 \times 96 \times 12288^2 \approx 174B$ parameters. Each parameter is a coordinate in the hypothesis space $\Theta = \mathbb{R}^{174B}$.

---


## 12. Category Theory Perspective

### 12.1 Categories and Morphisms

**Definition.** A **category** $\mathcal{C}$ consists of:
1. A collection of **objects** $\text{Ob}(\mathcal{C})$
2. For each pair of objects $A, B$, a set of **morphisms** $\text{Hom}(A, B)$ (arrows from $A$ to $B$)
3. A **composition** operation: if $f: A \to B$ and $g: B \to C$, then $g \circ f: A \to C$
4. An **identity** morphism $\text{id}_A: A \to A$ for each object

Subject to:
- **Associativity**: $h \circ (g \circ f) = (h \circ g) \circ f$
- **Identity**: $f \circ \text{id}_A = f$ and $\text{id}_B \circ f = f$

**This should look familiar.** The axioms of a category are precisely the axioms of function composition from Section 4!

**Key categories:**

| Category | Objects | Morphisms |
|---|---|---|
| **Set** | Sets | Functions |
| **Vect** | Vector spaces | Linear maps |
| **Top** | Topological spaces | Continuous maps |
| **Meas** | Measurable spaces | Measurable functions |
| **Grp** | Groups | Group homomorphisms |

**For AI.** A neural network layer is a morphism in the category of **parameterised differentiable functions**:
- Objects: $\mathbb{R}^n$ (Euclidean spaces of various dimensions)
- Morphisms: differentiable functions $f_\theta: \mathbb{R}^n \to \mathbb{R}^m$
- Composition: layered network construction

### 12.2 Functors: Maps Between Categories

**Definition.** A **functor** $F: \mathcal{C} \to \mathcal{D}$ is a "structure-preserving map" between categories:
1. $F$ maps objects: $A \mapsto F(A)$
2. $F$ maps morphisms: $(f: A \to B) \mapsto (F(f): F(A) \to F(B))$
3. $F$ preserves composition: $F(g \circ f) = F(g) \circ F(f)$
4. $F$ preserves identities: $F(\text{id}_A) = \text{id}_{F(A)}$

**Intuition.** A functor is a "homomorphism of categories" — it maps the entire structure (objects + arrows + composition) from one category to another.

**Examples relevant to AI:**

| Functor | From | To | Maps |
|---|---|---|---|
| **Forgetful** | Vect → Set | Vector spaces | Forget the linear structure |
| **Free** | Set → Vect | Sets | Map to free vector space |
| **Probability** | Meas → Set | Measurable spaces | Map to set of probability measures |
| **Dual space** | Vect → Vect$^{\text{op}}$ | Vector spaces | $V \mapsto V^*$ (contravariant) |

**For AI: JAX transforms as functors.** The `vmap` transform in JAX can be viewed as a functor:
- It maps functions $f: \mathbb{R}^n \to \mathbb{R}^m$ to batched functions $\text{vmap}(f): \mathbb{R}^{B \times n} \to \mathbb{R}^{B \times m}$
- It preserves composition: $\text{vmap}(g \circ f) = \text{vmap}(g) \circ \text{vmap}(f)$
- It preserves identity: $\text{vmap}(\text{id}) = \text{id}$

Similarly, `grad` is a (contravariant-like) functor from scalar-valued functions to vector-valued functions.

### 12.3 Natural Transformations

**Definition.** A **natural transformation** $\eta: F \Rightarrow G$ between functors $F, G: \mathcal{C} \to \mathcal{D}$ assigns to each object $A$ a morphism $\eta_A: F(A) \to G(A)$ such that the following diagram commutes for every morphism $f: A \to B$:

```
F(A) ──── η_A ────→ G(A)
  │                    │
F(f)│                    │G(f)
  │                    │
  ▼                    ▼
F(B) ──── η_B ────→ G(B)
```

Commutative: $G(f) \circ \eta_A = \eta_B \circ F(f)$.

**Intuition.** A natural transformation is a "systematic" way of converting one construction into another, one that works uniformly across all objects and is compatible with all morphisms.

**For AI.** Natural transformations appear in:
- **Activation functions**: The ReLU activation defines a natural transformation from the identity functor to itself (in a suitable category of parameterised functions). It transforms each layer's output in a way that commutes with the layer's parameterisation.
- **Normalisation layers**: Layer norm transforms representations in a way that is "natural" — it doesn't depend on the specific basis chosen for the hidden dimension.

### 12.4 Isomorphisms and Equivalences

**Definition.** A morphism $f: A \to B$ is an **isomorphism** if there exists $g: B \to A$ with $g \circ f = \text{id}_A$ and $f \circ g = \text{id}_B$.

In different categories, isomorphisms have different names:

| Category | Isomorphism Called | Meaning |
|---|---|---|
| **Set** | Bijection | One-to-one correspondence |
| **Vect** | Linear isomorphism | Invertible linear map |
| **Top** | Homeomorphism | Continuous bijection with continuous inverse |
| **Smooth manifolds** | Diffeomorphism | Smooth bijection with smooth inverse |
| **Groups** | Group isomorphism | Structure-preserving bijection |

**For AI: Diffeomorphisms in normalising flows.** A normalising flow is a sequence of diffeomorphisms $f_1, f_2, \ldots, f_K$ in the category of smooth manifolds. Each $f_k$ is:
- Smooth (differentiable)
- Bijective (invertible)
- Has a smooth inverse

The composition $f = f_K \circ \cdots \circ f_1$ is also a diffeomorphism, and the change-of-variables formula allows exact likelihood computation:

$$\log p(x) = \log p_z(f^{-1}(x)) + \sum_{k=1}^{K} \log \left|\det J_{f_k^{-1}}\right|$$

### 12.5 The Yoneda Lemma and Representations

**Yoneda Lemma (informal).** An object is completely determined by its relationships (morphisms) to all other objects. More precisely:

$$\text{Nat}(\text{Hom}(A, -), F) \cong F(A)$$

Natural transformations from the representable functor $\text{Hom}(A, -)$ to any functor $F$ are in bijection with elements of $F(A)$.

**Consequence.** Two objects are isomorphic if and only if they have the same "relationship pattern" to all other objects: $A \cong B$ if and only if $\text{Hom}(A, C) \cong \text{Hom}(B, C)$ for all $C$.

**For AI (conceptual analogy).** The Yoneda perspective suggests that a representation (embedding) is good if it captures all the relationships of the represented object:
- A word embedding $v_w \in \mathbb{R}^d$ is a "compressed" version of the word's relationships to all other words
- Two words are "the same" (isomorphic) if they have the same relationship to all contexts
- This is precisely the distributional hypothesis: "a word is characterised by the company it keeps" (Firth 1957)
- Word2Vec, GloVe, and contextual embeddings all operationalise this principle: embeddings derive from co-occurrence patterns (relationships to other words)

The Yoneda perspective also connects to **representation learning** more broadly: a good representation of an object captures all the relevant "morphisms" (transformations, relationships) between it and everything else.

---


## 13. Common Mistakes and Misconceptions

### 13.1 Confusing Function with Formula

**Mistake:** Treating "$f(x) = x^2$" as the function, rather than understanding that the function is the mapping $f: \mathbb{R} \to \mathbb{R}$ defined by $x \mapsto x^2$.

**Why it matters:** The same formula can define different functions depending on domain and codomain:
- $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$ — NOT injective
- $g: [0, \infty) \to \mathbb{R}$, $g(x) = x^2$ — injective
- $h: \mathbb{R} \to [0, \infty)$, $h(x) = x^2$ — surjective
- $k: [0, \infty) \to [0, \infty)$, $k(x) = x^2$ — bijective

These are four different functions with the same formula.

### 13.2 Confusing Codomain and Range

**Mistake:** Assuming the codomain equals the range.

The codomain is part of the function's definition (what it can in principle output). The range is what it actually outputs. For $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$:
- Codomain: $\mathbb{R}$ (all reals)
- Range: $[0, \infty)$ (non-negative reals only)
- The function is NOT surjective because range $\neq$ codomain

In AI: a classifier $f: \mathbb{R}^d \to \mathbb{R}^{10}$ has codomain $\mathbb{R}^{10}$, but after softmax the actual outputs are in $\Delta^{10}$ (the probability simplex) — a strict subset.

### 13.3 Assuming Composition is Commutative

**Mistake:** Assuming $f \circ g = g \circ f$.

Composition is almost never commutative. In neural networks:
- Pre-LN: $x + \text{Attn}(\text{LN}(x))$ — normalise then attend
- Post-LN: $\text{LN}(x + \text{Attn}(x))$ — attend then normalise
- These produce different results and have different training dynamics

### 13.4 Forgetting That Linear Means f(0) = 0

**Mistake:** Calling $f(x) = Wx + b$ a "linear function".

It is **affine**, not linear (unless $b = 0$). A truly linear function satisfies $f(0) = 0$, $f(x + y) = f(x) + f(y)$, and $f(\alpha x) = \alpha f(x)$. The term "linear layer" in deep learning is a misnomer — PyTorch's `nn.Linear` is actually an affine layer.

### 13.5 Assuming Differentiable Implies Smooth

**Mistake:** Thinking once-differentiable ($C^1$) means infinitely differentiable ($C^\infty$).

ReLU is not even $C^1$ (not differentiable at $x = 0$). The function $f(x) = x|x|$ is $C^1$ but not $C^2$. For optimisation, $C^1$ is sufficient for gradient descent, but $C^2$ is needed for second-order methods.

### 13.6 Ignoring Domain Restrictions

**Mistake:** Applying $\log(x)$ without checking that $x > 0$, or computing $1/x$ at $x = 0$.

In AI, this appears as:
- `log(0)` in cross-entropy loss when prediction is exactly 0 — fix: use `log(p + eps)` or `log_softmax`
- Division by zero in normalisation when variance is 0 — fix: use $\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \varepsilon}$
- `sqrt(0)` gradient is infinite — fix: use `sqrt(x + eps)`

### 13.7 Confusing Bijectivity with Invertibility

**Mistake:** Trying to "invert" a non-bijective function.

The pseudo-inverse $A^+$ is NOT a true inverse — $AA^+A = A$ but $A^+AA^+ = A^+$ does not give $AA^+ = I$ or $A^+A = I$ in general. In AI:
- You cannot perfectly reconstruct an input from a lower-dimensional embedding (dimensionality reduction is not injective)
- An autoencoder with bottleneck $d_{\text{latent}} < d_{\text{input}}$ cannot be bijective
- Pooling layers (max pool, average pool) are not injective — information is lost

---

## 14. Exercises and Practice Problems

### 14.1 Foundational Exercises

**Exercise 1 (Injectivity).** Prove that the function $f: \mathbb{R}^n \to \mathbb{R}^n$ defined by $f(x) = Ax$ is injective if and only if $\det(A) \neq 0$.

*Hint:* Injective means $\ker(A) = \{0\}$. Use the fact that $\det(A) \neq 0 \iff \ker(A) = \{0\}$.

**Exercise 2 (Composition).** Let $f: \mathbb{R}^m \to \mathbb{R}^k$ and $g: \mathbb{R}^n \to \mathbb{R}^m$ be linear functions represented by matrices $A$ and $B$ respectively. Show that $f \circ g$ is represented by the matrix $AB$.

**Exercise 3 (Surjectivity of softmax).** Show that $\text{softmax}: \mathbb{R}^n \to \Delta^n$ is surjective onto the interior of the simplex $\text{int}(\Delta^n) = \{p \in \mathbb{R}^n : p_i > 0, \sum p_i = 1\}$ but NOT surjective onto the boundary.

*Hint:* For interior points, use $z_i = \log(p_i)$. For the boundary, note that softmax outputs are always strictly positive.

**Exercise 4 (Inverse of sigmoid).** Derive the inverse of the sigmoid function $\sigma(x) = 1/(1 + e^{-x})$.

*Solution sketch:* $y = 1/(1 + e^{-x}) \implies 1 + e^{-x} = 1/y \implies e^{-x} = 1/y - 1 = (1-y)/y \implies x = \log(y/(1-y))$. This is the logit function: $\text{logit}(y) = \log(y/(1-y))$.

### 14.2 Intermediate Exercises

**Exercise 5 (Lipschitz constant of composition).** A network has 3 layers: ReLU activation (Lipschitz constant 1), a weight matrix $W_1$ with $\sigma_{\max}(W_1) = 2$, and another weight matrix $W_2$ with $\sigma_{\max}(W_2) = 3$. What is the upper bound on the network's Lipschitz constant?

*Answer:* $L \leq 1 \times 2 \times 3 = 6$.

**Exercise 6 (Residual connection Jacobian).** For a residual block $f(x) = x + g(x)$ where $g(x) = W_2 \cdot \text{ReLU}(W_1 x)$:

(a) Write the Jacobian $J_f(x)$ in terms of $J_g(x)$.
(b) Show that if $\|J_g\|_{\text{op}} < 1$, then $J_f$ is invertible.
(c) What does this imply about the information flow through the residual block?

**Exercise 7 (Convexity).** Prove that the cross-entropy loss $L(z) = -z_y + \log\sum_{j} e^{z_j}$ is convex in the logits $z$.

*Hint:* Compute the Hessian and show it is positive semi-definite. The Hessian of log-sum-exp is $\text{diag}(p) - pp^T$ where $p = \text{softmax}(z)$.

### 14.3 Advanced Exercises

**Exercise 8 (Universal approximation).** Construct a ReLU network with one hidden layer that exactly represents the function $f(x) = |x|$ for $x \in \mathbb{R}$.

*Solution:* $|x| = \text{ReLU}(x) + \text{ReLU}(-x)$. This is a network with 2 hidden units, weights $w_1 = 1, w_2 = -1$, biases $b_1 = b_2 = 0$, output weights $\alpha_1 = \alpha_2 = 1$.

**Exercise 9 (Measure theory).** Let $X \sim \mathcal{N}(0, 1)$ and $Y = X^2$. Find the pushforward measure $P_Y$ (the distribution of $Y$).

*Hint:* $Y$ follows a chi-squared distribution with 1 degree of freedom. Use the change-of-variables formula for non-injective functions by splitting the domain into $(-\infty, 0)$ and $(0, \infty)$.

**Exercise 10 (Category theory).** Verify that neural network layers with composition form a category:
(a) Define the objects and morphisms.
(b) Verify associativity of composition.
(c) Define the identity morphism and verify the identity laws.
(d) Is this category a subcategory of **Set**? Of **Vect**? Why or why not?

---

## 15. Why This Matters for AI/LLMs

### 15.1 Everything Is a Function

The central thesis of this chapter: **every component of a modern AI system is a function**, and the properties of those functions (injectivity, surjectivity, linearity, continuity, differentiability, Lipschitz constants) directly determine the system's capabilities and limitations.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE AI SYSTEM AS A FUNCTION                  │
│                                                                 │
│  String ──→ Tokens ──→ Embeddings ──→ Hidden ──→ Logits ──→ P  │
│         f₁         f₂            f₃         f₄         f₅     │
│                                                                 │
│  f₁: Tokenise (deterministic, not injective in general)        │
│  f₂: Embed     (injective lookup, discrete → continuous)       │
│  f₃: Transform (nonlinear, differentiable, high-dimensional)   │
│  f₄: LM Head   (linear projection, dimension reduction)        │
│  f₅: Softmax   (continuous, surjective onto int(Δ), bounded)  │
│                                                                 │
│  Full system: f = f₅ ∘ f₄ ∘ f₃ ∘ f₂ ∘ f₁                     │
│  Training: find θ* = argmin_θ E[L(f_θ(x), y)]                 │
│  Inference: sample from f_θ*(context)                           │
└─────────────────────────────────────────────────────────────────┘
```

### 15.2 The Function Properties That Matter Most

| Property | Why It Matters | Where It Appears |
|---|---|---|
| **Differentiability** | Enables gradient-based training | Everywhere (backpropagation) |
| **Lipschitz continuity** | Controls sensitivity and robustness | All layers (adversarial robustness) |
| **Injectivity** | Preserves information | Invertible networks, normalising flows |
| **Composition** | Builds complex from simple | Deep networks = composition of layers |
| **Linearity** | Efficient, well-understood | Attention projections, FFN |
| **Convexity** | Guarantees global optimum | Loss (in logits), regularisation |
| **Bilinearity** | Captures interactions | Attention scores, dot products |
| **Measurability** | Enables probabilistic reasoning | Random variables, generative models |

### 15.3 Scale and the Function Perspective

The remarkable discovery of the scaling laws era (2020-present): the function class $\mathcal{H}_\theta$ indexed by a few hyperparameters (depth $L$, width $d$, data $D$, compute $C$) follows predictable relationships:

$$\mathcal{L}(C) \approx \left(\frac{C_0}{C}\right)^{\alpha_C}, \quad \mathcal{L}(D) \approx \left(\frac{D_0}{D}\right)^{\alpha_D}, \quad \mathcal{L}(N) \approx \left(\frac{N_0}{N}\right)^{\alpha_N}$$

where $\mathcal{L}$ is the loss, $C$ is compute, $D$ is data, and $N$ is parameter count. These are power laws — the loss is a specific functional form of the "size" of the function class.

**From a function theory perspective:**
- More parameters → larger hypothesis class → can approximate more complex functions
- More data → better estimation of the target function → lower approximation error
- More compute → more SGD steps → better search through the hypothesis class
- The power-law form suggests deep structural regularities in the relationship between function class capacity and approximation quality

## 16. Conceptual Bridge

### 16.1 Looking Ahead

The mathematical tools from this chapter appear throughout the rest of this curriculum:

| Topic | Function Concepts Used |
|---|---|
| **Linear Algebra** | Linear maps, matrix representations of functions |
| **Calculus** | Derivatives (local linear approximations of functions) |
| **Optimisation** | Minimising loss functionals over parameterised function spaces |
| **Probability** | Random variables as measurable functions, distributions as pushforwards |
| **Information Theory** | Entropy and mutual information as functionals on distributions |
| **Backpropagation** | Chain rule for compositions of differentiable functions |

### 16.2 The Meta-Insight

Mathematics is, in a deep sense, the study of functions and their properties. Linear algebra studies linear functions. Calculus studies differentiable functions. Topology studies continuous functions. Measure theory studies measurable functions. Category theory studies all structure-preserving functions (morphisms) at once.

The bridge forward is this: later chapters will often look like they are about matrices, gradients, probability distributions, or optimization algorithms, but underneath they are still about maps between spaces, compositions of transformations, and constraints on what those transformations can preserve. Understanding functions — truly understanding them — is understanding the language in which all of AI is written.

---

## References

1. Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
2. Hornik, K. (1991). "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251-257.
3. Hendrycks, D. & Gimpel, K. (2016). "Gaussian Error Linear Units (GELUs)." *arXiv:1606.08415*.
4. Shazeer, N. (2020). "GLU Variants Improve Transformer." *arXiv:2002.05202*.
5. Miyato, T. et al. (2018). "Spectral Normalization for Generative Adversarial Networks." *ICLR 2018*.
6. Telgarsky, M. (2016). "Benefits of depth in neural networks." *COLT 2016*.
7. He, K. et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR 2016*.
8. Kingma, D.P. & Dhariwal, P. (2018). "Glow: Generative Flow with Invertible 1×1 Convolutions." *NeurIPS 2018*.
9. Jang, E. et al. (2017). "Categorical Reparameterization with Gumbel-Softmax." *ICLR 2017*.
10. Lu, Z. et al. (2017). "The Expressive Power of Neural Networks: A View from the Width." *NeurIPS 2017*.
11. Penrose, R. (1955). "A generalized inverse for matrices." *Mathematical Proceedings of the Cambridge Philosophical Society*, 51(3), 406-413.
12. Mac Lane, S. (1998). *Categories for the Working Mathematician*. Springer.
13. Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." *arXiv:2001.08361*.
14. Misra, D. (2019). "Mish: A Self Regularized Non-Monotonic Activation Function." *arXiv:1908.08681*.
15. Bai, S. et al. (2019). "Deep Equilibrium Models." *NeurIPS 2019*.

---

*Next: [04-Summation-and-Product-Notation](../04-Summation-and-Product-Notation/notes.md)*
