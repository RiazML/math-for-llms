[← Back to Probability Theory](../README.md) | [Next: Common Distributions →](../02-Common-Distributions/notes.md)

---

# Introduction to Probability and Random Variables

> _"Probability theory is nothing but common sense reduced to calculation."_
> — Pierre-Simon Laplace, 1814

## Overview

Probability theory is the mathematical language for reasoning under uncertainty — and uncertainty is everywhere in machine learning. Every training example is a noisy sample from an unknown distribution; every weight initialisation is a random draw; every language model output is a distribution over tokens. To reason rigorously about any of these, we need the formal apparatus built in this section.

We begin at the foundations: the Kolmogorov axioms that define what a probability is, and the algebraic rules (complement, union, conditional probability, Bayes' theorem) that follow from them. We then introduce the central object of probabilistic modelling — the *random variable* — and its two modes: discrete (characterised by a probability mass function) and continuous (characterised by a probability density function). We close with the expectation and variance of a random variable, which distil a full distribution into its most useful summary statistics.

**Scope:** This section is the canonical home for probability axioms, conditional probability, independence, random variable definitions, CDF/PDF/PMF, and the Bernoulli/Uniform/Gaussian distributions at an introductory level. The full catalogue of named distributions (Binomial, Poisson, Beta, Dirichlet, Categorical, Student-t) is covered in §02. Joint distributions and Bayes' theorem for random variables are developed in §03. The full treatment of expectation, covariance, LOTUS, and moment generating functions is in §04.

## Prerequisites

- Basic set theory: union ($A \cup B$), intersection ($A \cap B$), complement ($A^c$), subset ($A \subseteq B$)
- Single-variable calculus: integration, differentiation — [§03-Integration](../../04-Calculus-Fundamentals/03-Integration/notes.md)
- Series and summation notation — [§04-Series-and-Sequences](../../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: Kolmogorov axioms, Bayes' theorem, CDF/PDF/PMF, Bernoulli, Gaussian intro, change of variables, cross-entropy |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: axiom proofs through probabilistic language model analysis |

## Learning Objectives

After completing this section, you will:

- State the three Kolmogorov axioms and derive the complement, union, and monotonicity rules from them
- Compute conditional probabilities and apply the chain rule and law of total probability
- Apply Bayes' theorem to update beliefs from evidence, naming prior, likelihood, and posterior
- Distinguish independence from conditional independence and give an example where they differ
- Define a random variable formally as a measurable function from a sample space to $\mathbb{R}$
- Write the CDF $F(x) = P(X \leq x)$ for a discrete and a continuous random variable, and list its four properties
- Compute probabilities from a PMF (discrete case) and a PDF (continuous case)
- Describe the Bernoulli and Uniform distributions completely: PMF/PDF, mean, variance, and ML uses
- Introduce the Gaussian distribution and state the 68–95–99.7 rule
- Compute $\mathbb{E}[X]$ and $\operatorname{Var}(X)$ for simple distributions and apply linearity of expectation
- Apply the change-of-variables formula for monotone transformations of a continuous random variable
- Express cross-entropy loss as negative log-likelihood and explain what it minimises

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is Probability? Three Interpretations](#11-what-is-probability-three-interpretations)
  - [1.2 Why Probability Is the Language of AI](#12-why-probability-is-the-language-of-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
- [2. Formal Definitions — Probability Spaces](#2-formal-definitions--probability-spaces)
  - [2.1 Sample Spaces and Events](#21-sample-spaces-and-events)
  - [2.2 The Three Kolmogorov Axioms](#22-the-three-kolmogorov-axioms)
  - [2.3 Immediate Consequences of the Axioms](#23-immediate-consequences-of-the-axioms)
  - [2.4 Probability Measures — Examples and Non-Examples](#24-probability-measures--examples-and-non-examples)
- [3. Computing Probabilities — The Core Rules](#3-computing-probabilities--the-core-rules)
  - [3.1 Complement and Monotonicity](#31-complement-and-monotonicity)
  - [3.2 Inclusion-Exclusion Principle](#32-inclusion-exclusion-principle)
  - [3.3 Conditional Probability](#33-conditional-probability)
  - [3.4 Chain Rule of Probability](#34-chain-rule-of-probability)
  - [3.5 Law of Total Probability](#35-law-of-total-probability)
  - [3.6 Bayes' Theorem](#36-bayes-theorem)
- [4. Independence](#4-independence)
  - [4.1 Unconditional Independence](#41-unconditional-independence)
  - [4.2 Conditional Independence](#42-conditional-independence)
  - [4.3 Pairwise vs Mutual Independence](#43-pairwise-vs-mutual-independence)
  - [4.4 Why Independence Matters for AI](#44-why-independence-matters-for-ai)
- [5. Random Variables — Formal Foundation](#5-random-variables--formal-foundation)
  - [5.1 Definition: Random Variable as a Measurable Function](#51-definition-random-variable-as-a-measurable-function)
  - [5.2 Discrete vs Continuous — Taxonomy](#52-discrete-vs-continuous--taxonomy)
  - [5.3 The Cumulative Distribution Function](#53-the-cumulative-distribution-function)
  - [5.4 CDF Properties and the Fundamental Theorem](#54-cdf-properties-and-the-fundamental-theorem)
- [6. Discrete Random Variables](#6-discrete-random-variables)
  - [6.1 Probability Mass Function](#61-probability-mass-function)
  - [6.2 Bernoulli Distribution — The Canonical Example](#62-bernoulli-distribution--the-canonical-example)
  - [6.3 Geometric Distribution](#63-geometric-distribution)
  - [6.4 Preview: Binomial, Categorical, Poisson](#64-preview-binomial-categorical-poisson)
- [7. Continuous Random Variables](#7-continuous-random-variables)
  - [7.1 Probability Density Function](#71-probability-density-function)
  - [7.2 From CDF to PDF: the Derivative Relationship](#72-from-cdf-to-pdf-the-derivative-relationship)
  - [7.3 Uniform Distribution](#73-uniform-distribution)
  - [7.4 Gaussian Distribution — Introduction](#74-gaussian-distribution--introduction)
  - [7.5 Preview: Exponential, Beta, Gamma](#75-preview-exponential-beta-gamma)
- [8. Expectation and Variance — Foundations](#8-expectation-and-variance--foundations)
  - [8.1 Expected Value: Definition and Linearity](#81-expected-value-definition-and-linearity)
  - [8.2 Variance and Standard Deviation](#82-variance-and-standard-deviation)
  - [8.3 Key Properties and Common Pitfalls](#83-key-properties-and-common-pitfalls)
  - [8.4 Preview: Covariance, LOTUS, MGF](#84-preview-covariance-lotus-mgf)
- [9. Transformations of Random Variables](#9-transformations-of-random-variables)
  - [9.1 Functions of a Random Variable](#91-functions-of-a-random-variable)
  - [9.2 Change-of-Variables Formula](#92-change-of-variables-formula)
  - [9.3 Standardisation and the Z-Score](#93-standardisation-and-the-z-score)
- [10. Applications in Machine Learning](#10-applications-in-machine-learning)
  - [10.1 Cross-Entropy Loss as Negative Log-Likelihood](#101-cross-entropy-loss-as-negative-log-likelihood)
  - [10.2 Bayesian Inference: Prior, Likelihood, Posterior](#102-bayesian-inference-prior-likelihood-posterior)
  - [10.3 Probabilistic Language Models](#103-probabilistic-language-models)
  - [10.4 Sources of Randomness in Neural Network Training](#104-sources-of-randomness-in-neural-network-training)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---
## 1. Intuition

### 1.1 What Is Probability? Three Interpretations

Probability is a number in $[0, 1]$ assigned to an event, measuring how likely that event is to occur. But what does "likely" mean? There are three major interpretations, and each underpins a different school of thought in statistics and machine learning.

**1. Classical (equally likely outcomes).** If a sample space $\Omega$ has $N$ equally likely outcomes and event $A$ contains $k$ of them, then $P(A) = k/N$. This is the interpretation of textbook dice and card problems. It breaks down when outcomes are not equally likely — flipping a biased coin, for instance.

**2. Frequentist (long-run relative frequency).** $P(A)$ is the limiting proportion of times $A$ occurs in an infinite sequence of identical, independent experiments: $P(A) = \lim_{n \to \infty} \frac{\text{count of } A \text{ in } n \text{ trials}}{n}$. This is the foundation of classical statistics (hypothesis testing, confidence intervals). Its weakness: it cannot assign probabilities to one-off events ("the probability that GPT-5 passes the bar exam").

**3. Bayesian (degree of belief).** $P(A)$ represents a rational agent's degree of belief that $A$ is true, updated as new evidence arrives. This interpretation allows probabilities for singular events and underpins Bayesian inference, Bayesian neural networks, and probabilistic generative models. Crucially, different agents can assign different prior probabilities to the same event — and Bayes' theorem tells them how to update rationally.

```
THREE INTERPRETATIONS OF PROBABILITY
════════════════════════════════════════════════════════════════════════

  Classical           Frequentist              Bayesian
  ─────────────────   ────────────────────     ──────────────────────
  P(A) = k/N          P(A) = lim freq(A)       P(A) = degree of belief
  Equally likely       Long-run proportion      Updated by evidence
  outcomes             Objective               Subjective / rational
  Dice, cards          Hypothesis tests         Generative models
                       Confidence intervals     Bayesian NNs, VAEs

  All three satisfy the same axioms (§2.2).
  They differ only in WHAT those axioms are applied to.

════════════════════════════════════════════════════════════════════════
```

**For AI:** Modern ML systems implicitly blend all three. A softmax output layer uses the classical interpretation (outputs sum to 1). Training loop analysis uses frequentist reasoning (expected loss over the data distribution). Bayesian neural networks, RLHF reward models, and variational autoencoders use Bayesian reasoning explicitly.

### 1.2 Why Probability Is the Language of AI

Every component of a modern AI system is probabilistic:

- **Data:** Training sets are finite samples from an unknown data-generating distribution $p_{\text{data}}(\mathbf{x}, y)$. The goal of supervised learning is to find $f_\theta$ that estimates $p_{\text{data}}(y \mid \mathbf{x})$.
- **Model outputs:** A language model defines a probability distribution $P(x_t \mid x_{<t})$ over the next token. Generation is sampling from this distribution. Temperature scaling, top-$k$ and top-$p$ sampling are all operations on probability distributions.
- **Loss functions:** Cross-entropy is the negative log-likelihood of the data under the model's distribution. Minimising cross-entropy is equivalent to maximising the probability the model assigns to the training data.
- **Regularisation:** Dropout applies Bernoulli random variables to activations. Weight decay corresponds to a Gaussian prior in a Bayesian interpretation (MAP estimation).
- **Uncertainty quantification:** Calibration of a classifier means its predicted probabilities match empirical frequencies. A well-calibrated model saying "70% confident" should be right 70% of the time.
- **Generative models:** Diffusion models define a forward Markov chain (adding Gaussian noise) and learn to reverse it. VAEs learn $p_\theta(\mathbf{x} \mid \mathbf{z})$ and $q_\phi(\mathbf{z} \mid \mathbf{x})$. Normalising flows learn invertible transformations of simple distributions. All are explicitly probabilistic.

### 1.3 Historical Timeline

| Year | Person | Contribution |
|------|--------|-------------|
| 1654 | Pascal & Fermat | Correspondence on gambling problems — foundations of combinatorial probability |
| 1713 | Jacob Bernoulli | *Ars Conjectandi* — law of large numbers, Bernoulli distribution |
| 1763 | Thomas Bayes (posthumous) | Essay on inverse probability — what we now call Bayes' theorem |
| 1812 | Pierre-Simon Laplace | *Théorie analytique des probabilités* — systematic probability theory, Laplace approximation |
| 1837 | Siméon Poisson | Poisson distribution — rare events in large samples |
| 1900 | Karl Pearson | Chi-squared distribution, correlation coefficient |
| 1933 | Andrei Kolmogorov | Rigorous axiomatic foundation — the three axioms that unify all interpretations |
| 1950s | Shannon | Information theory — entropy connects probability to communication |
| 1980s | Judea Pearl | Bayesian networks — graphical models for probabilistic reasoning |
| 1990s | Gelman, Rubin et al. | MCMC methods — practical Bayesian inference for complex models |
| 2013 | Kingma & Welling | Variational Autoencoders — learned latent distributions via reparameterisation |
| 2020 | Ho et al. | Denoising Diffusion Probabilistic Models — probabilistic generative modelling at scale |

---

## 2. Formal Definitions — Probability Spaces

### 2.1 Sample Spaces and Events

**Definition 2.1 (Sample space).** The *sample space* $\Omega$ is the set of all possible outcomes of a random experiment. Each element $\omega \in \Omega$ is called an *elementary outcome* or *sample point*.

**Examples:**
- Tossing a fair coin: $\Omega = \{\text{H}, \text{T}\}$
- Rolling a six-sided die: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Measuring a person's height: $\Omega = (0, \infty) \subset \mathbb{R}$
- Choosing a word from a vocabulary: $\Omega = \{\text{the}, \text{cat}, \text{sat}, \ldots\}$ (finite vocabulary)
- One training step of a neural network: $\Omega = $ all possible mini-batch draws

**Definition 2.2 (Event).** An *event* $A$ is a subset of $\Omega$, i.e., $A \subseteq \Omega$. We say event $A$ *occurs* if the observed outcome $\omega \in A$.

**Event algebra.** Events combine using set operations:
- **Complement:** $A^c = \Omega \setminus A$ — "$A$ does not occur"
- **Union:** $A \cup B$ — "$A$ or $B$ (or both) occur"
- **Intersection:** $A \cap B$ — "both $A$ and $B$ occur"
- **Difference:** $A \setminus B = A \cap B^c$ — "$A$ occurs but $B$ does not"

**The $\sigma$-algebra (brief note).** For continuous sample spaces (e.g., $\Omega = \mathbb{R}$), we cannot assign probabilities to *every* subset — some are too pathological (Vitali sets). The solution is to restrict to a *$\sigma$-algebra* $\mathcal{F}$: a collection of subsets of $\Omega$ that is closed under complement and countable unions. The standard choice for $\Omega = \mathbb{R}$ is the *Borel $\sigma$-algebra* $\mathcal{B}(\mathbb{R})$, generated by all open intervals. For this course, you can think of "events" as "any reasonable subset of $\Omega$" — the measure-theoretic subtlety is noted but not required.

### 2.2 The Three Kolmogorov Axioms

In 1933, Andrei Kolmogorov placed probability theory on a rigorous axiomatic foundation, resolving a century of informal debate about what probability "really is."

**Definition 2.3 (Probability measure).** A *probability measure* is a function $P : \mathcal{F} \to [0, 1]$ satisfying:

**Axiom 1 (Non-negativity):**
$$P(A) \geq 0 \quad \text{for all events } A$$

**Axiom 2 (Normalisation):**
$$P(\Omega) = 1$$

**Axiom 3 (Countable additivity / $\sigma$-additivity):**
If $A_1, A_2, A_3, \ldots$ are *pairwise disjoint* events (i.e., $A_i \cap A_j = \emptyset$ for $i \neq j$), then:
$$P\!\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

The triple $(\Omega, \mathcal{F}, P)$ is called a *probability space*.

**Why these three axioms?** They are minimal: they are logically independent (none follows from the other two), they rule out degenerate assignments (e.g., $P(A) = -0.5$ or $P(\Omega) = 2$), and they imply everything else in probability theory via logical deduction.

### 2.3 Immediate Consequences of the Axioms

From just three axioms, we can derive all the basic rules of probability:

**Theorem 2.1 (Probability of the empty set):** $P(\emptyset) = 0$.

*Proof.* $\Omega$ and $\emptyset$ are disjoint with $\Omega \cup \emptyset = \Omega$. By Axiom 3: $P(\Omega) + P(\emptyset) = P(\Omega) = 1$. So $P(\emptyset) = 0$. $\square$

**Theorem 2.2 (Finite additivity):** For disjoint $A_1, \ldots, A_n$:
$$P(A_1 \cup \cdots \cup A_n) = P(A_1) + \cdots + P(A_n)$$

*Proof.* Set $A_{n+1} = A_{n+2} = \cdots = \emptyset$, apply Axiom 3, then Theorem 2.1. $\square$

**Theorem 2.3 (Complement rule):** $P(A^c) = 1 - P(A)$.

*Proof.* $A$ and $A^c$ are disjoint with $A \cup A^c = \Omega$. By Axiom 3: $P(A) + P(A^c) = P(\Omega) = 1$. $\square$

**Theorem 2.4 (Monotonicity):** If $A \subseteq B$, then $P(A) \leq P(B)$.

*Proof.* Write $B = A \cup (B \setminus A)$ where $A$ and $B \setminus A$ are disjoint. By Axiom 3: $P(B) = P(A) + P(B \setminus A) \geq P(A)$ (since $P(B \setminus A) \geq 0$ by Axiom 1). $\square$

**Theorem 2.5 (Bounds):** $0 \leq P(A) \leq 1$ for all $A$.

*Proof.* $\emptyset \subseteq A \subseteq \Omega$, so by monotonicity: $0 = P(\emptyset) \leq P(A) \leq P(\Omega) = 1$. $\square$

### 2.4 Probability Measures — Examples and Non-Examples

**Examples of valid probability measures:**

| Sample space $\Omega$ | Probability measure $P$ |
|---|---|
| $\{1,2,3,4,5,6\}$ | $P(\{k\}) = 1/6$ for each $k$ (fair die) |
| $\{1,2,3,4,5,6\}$ | $P(\{k\}) = k/21$ for each $k$ (loaded die, $\sum k/21 = 1$) |
| $[0,1]$ | $P([a,b]) = b-a$ for $0 \leq a \leq b \leq 1$ (uniform) |
| $\mathbb{R}$ | $P(A) = \int_A \frac{1}{\sqrt{2\pi}}e^{-x^2/2}dx$ (standard Gaussian) |
| $\{0, 1\}$ | $P(\{1\}) = p$, $P(\{0\}) = 1-p$ for $p \in [0,1]$ (Bernoulli) |

**Non-examples (violate at least one axiom):**

| Assignment | Axiom violated |
|---|---|
| $P(\{H\}) = 0.6$, $P(\{T\}) = 0.6$ | Axiom 2: total = 1.2 ≠ 1 |
| $P(\{H\}) = -0.1$, $P(\{T\}) = 1.1$ | Axiom 1: negative probability |
| $P(\{H\}) = 0.5$, $P(\{T\}) = 0.5$, $P(\{H,T\}) = 0.8$ | Axiom 3: $P(\{H\} \cup \{T\}) \neq P(\{H\}) + P(\{T\})$ |

---

## 3. Computing Probabilities — The Core Rules

### 3.1 Complement and Monotonicity

The complement rule is one of the most practically useful consequences of the axioms.

**Complement rule:** $P(A^c) = 1 - P(A)$

This is particularly useful when $P(A^c)$ is easier to compute than $P(A)$ directly. This technique — "compute the complement" — is ubiquitous in probability:

**Example.** What is the probability that at least one of 10 coin flips is heads (for a fair coin)?

Direct approach: sum $P(\text{exactly } k \text{ heads})$ for $k = 1, \ldots, 10$ — tedious.

Complement approach: $P(\text{at least one head}) = 1 - P(\text{no heads}) = 1 - (1/2)^{10} = 1023/1024 \approx 0.999$.

**For AI:** The complement rule underlies the "at least one" union bound used in statistical learning theory. If we want $P(\text{at least one of } n \text{ bad events occurs}) \leq \delta$, we equivalently want $P(\text{all events are good}) \geq 1 - \delta$.

### 3.2 Inclusion-Exclusion Principle

**Theorem 3.1 (Inclusion-exclusion, two events):**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

*Proof.* Decompose $A \cup B = A \cup (B \setminus A)$ into disjoint parts. Then $B = (A \cap B) \cup (B \setminus A)$ into disjoint parts. By Axiom 3: $P(A \cup B) = P(A) + P(B \setminus A)$ and $P(B) = P(A \cap B) + P(B \setminus A)$. Subtracting: $P(B \setminus A) = P(B) - P(A \cap B)$. Substituting: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. $\square$

The subtraction corrects for double-counting $A \cap B$.

**Generalisation to $n$ events:**
$$P\!\left(\bigcup_{i=1}^n A_i\right) = \sum_i P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \cdots$$

**Union bound (Boole's inequality):** A simpler but looser bound:
$$P\!\left(\bigcup_{i=1}^n A_i\right) \leq \sum_{i=1}^n P(A_i)$$

**For AI:** The union bound is the workhorse of statistical learning theory. To prove that a learned classifier generalises to unseen data with high probability, one typically applies the union bound over all possible hypotheses, then uses concentration inequalities (§05) to bound each term.

### 3.3 Conditional Probability

When we observe that event $B$ has occurred, this changes our state of knowledge. The probability of $A$ given that $B$ occurred is:

**Definition 3.1 (Conditional probability):**
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Intuition:** Conditioning on $B$ restricts the sample space from $\Omega$ to $B$. Within this restricted space, we renormalise by dividing by $P(B)$ to ensure the conditional probabilities sum to 1.

```
CONDITIONAL PROBABILITY — GEOMETRIC INTUITION
════════════════════════════════════════════════════════════════════════

  Before conditioning:         After conditioning on B:
  ┌─────────────────────┐      ┌─────────────────────┐
  │          Ω           │      │         B            │
  │   ┌────────────┐    │      │   ┌──────────────┐  │
  │   │  A   │A∩B │ B  │      │   │  A∩B         │  │
  │   │      │    │    │      │   │ (renormalised)│  │
  │   └────────────┘    │      │   └──────────────┘  │
  └─────────────────────┘      └─────────────────────┘

  P(A) = area(A) / area(Ω)     P(A|B) = area(A∩B) / area(B)

════════════════════════════════════════════════════════════════════════
```

**Key properties of conditional probability:**
- $P(\cdot \mid B)$ is itself a valid probability measure on $(\Omega, \mathcal{F})$: it satisfies all three Kolmogorov axioms
- $P(B \mid B) = 1$ — given $B$ occurred, $B$ certainly occurred
- $P(A^c \mid B) = 1 - P(A \mid B)$ — the complement rule holds conditionally

**Non-example:** $P(A \mid B) \neq P(B \mid A)$ in general. This asymmetry is the source of the classic prosecutor's fallacy: $P(\text{DNA match} \mid \text{innocent})$ is very small, but $P(\text{innocent} \mid \text{DNA match})$ could be substantial in a large population.

### 3.4 Chain Rule of Probability

Rearranging the definition of conditional probability gives:
$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

This extends to any number of events:

**Theorem 3.2 (Chain rule / product rule):**
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1, A_2) \cdots P(A_n \mid A_1, \ldots, A_{n-1})$$

**For AI:** The chain rule of probability is the mathematical foundation of autoregressive language models. A language model with vocabulary $\mathcal{V}$ and context window $T$ defines:

$$P(x_1, x_2, \ldots, x_T) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_T \mid x_1, \ldots, x_{T-1})$$

Every GPT-family model is directly computing these conditional probabilities. The model is trained to minimise the average negative log of each factor — i.e., the cross-entropy loss.

### 3.5 Law of Total Probability

**Definition 3.2 (Partition).** Events $B_1, B_2, \ldots, B_n$ form a *partition* of $\Omega$ if they are pairwise disjoint ($B_i \cap B_j = \emptyset$ for $i \neq j$) and exhaustive ($\bigcup_i B_i = \Omega$).

**Theorem 3.3 (Law of total probability).** If $B_1, \ldots, B_n$ partition $\Omega$ and $P(B_i) > 0$ for all $i$:
$$P(A) = \sum_{i=1}^n P(A \mid B_i) \cdot P(B_i)$$

*Proof.* Since $B_1, \ldots, B_n$ partition $\Omega$: $A = \bigcup_i (A \cap B_i)$, a disjoint union. By Axiom 3: $P(A) = \sum_i P(A \cap B_i) = \sum_i P(A \mid B_i) P(B_i)$. $\square$

**Example.** A language model generates a sentence that is either grammatical ($B_1$, probability 0.7) or ungrammatical ($B_2$, probability 0.3). Given grammatical, the probability a human accepts it is 0.9; given ungrammatical, 0.2. What is the overall acceptance probability?

$$P(\text{accepted}) = P(\text{acc} \mid B_1) P(B_1) + P(\text{acc} \mid B_2) P(B_2) = 0.9 \times 0.7 + 0.2 \times 0.3 = 0.63 + 0.06 = 0.69$$

### 3.6 Bayes' Theorem

Bayes' theorem inverts conditional probability: from $P(A \mid B)$ we can compute $P(B \mid A)$.

**Theorem 3.4 (Bayes' theorem):**
$$P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A)}$$

Applying the law of total probability to the denominator (with a partition $B, B^c$):
$$P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A \mid B) P(B) + P(A \mid B^c) P(B^c)}$$

**Bayesian inference terminology:**

| Term | Symbol | Meaning |
|------|--------|---------|
| Prior | $P(B)$ | Belief about $B$ before seeing $A$ |
| Likelihood | $P(A \mid B)$ | How probable is $A$ if $B$ is true? |
| Evidence | $P(A)$ | Total probability of observation $A$ |
| Posterior | $P(B \mid A)$ | Updated belief about $B$ after seeing $A$ |

$$\underbrace{P(B \mid A)}_{\text{posterior}} = \frac{\overbrace{P(A \mid B)}^{\text{likelihood}} \cdot \overbrace{P(B)}^{\text{prior}}}{\underbrace{P(A)}_{\text{evidence}}}$$

**Classic example — Spam filter.** Email is spam with prior $P(\text{spam}) = 0.3$. The word "lottery" appears in 80% of spam emails but only 5% of legitimate ones. Given an email contains "lottery", what is $P(\text{spam} \mid \text{lottery})$?

$$P(\text{spam} \mid \text{lottery}) = \frac{0.80 \times 0.30}{0.80 \times 0.30 + 0.05 \times 0.70} = \frac{0.24}{0.24 + 0.035} = \frac{0.24}{0.275} \approx 0.873$$

**For AI:** Bayes' theorem is the core of:
- **Bayesian neural networks:** parameters $\theta$ have a prior $P(\theta)$; training updates to the posterior $P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta) P(\theta)$
- **RLHF reward modelling:** given human preference data, update beliefs about which completions are better
- **Naive Bayes classifiers:** classify documents by computing $P(\text{class} \mid \text{words}) \propto P(\text{words} \mid \text{class}) P(\text{class})$

---

## 4. Independence

### 4.1 Unconditional Independence

Intuitively, events $A$ and $B$ are independent if knowing one occurred gives no information about whether the other occurred.

**Definition 4.1 (Independence).** Events $A$ and $B$ are *independent*, written $A \perp B$, if:
$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently (when $P(B) > 0$): $P(A \mid B) = P(A)$ — conditioning on $B$ does not change the probability of $A$.

**Why this definition?** The conditional probability $P(A \mid B) = P(A \cap B)/P(B)$ equals $P(A)$ if and only if $P(A \cap B) = P(A)P(B)$. The multiplicative form is preferred as it is symmetric and well-defined even when $P(A) = 0$ or $P(B) = 0$.

**Examples of independent events:**
- Two fair coin flips: $P(\text{H on flip 1} \cap \text{H on flip 2}) = 1/4 = 1/2 \times 1/2$ ✓
- Drawing with replacement: second draw is independent of first
- Two separate neural network weight initialisations drawn i.i.d.

**Non-examples (dependent events):**
- Drawing without replacement: if first card is an ace, second ace is less likely
- Correlated features in a dataset
- Sequential tokens in a language model: $P(x_t \mid x_{t-1}) \neq P(x_t)$

**For $n$ events (mutual independence):** Events $A_1, \ldots, A_n$ are mutually independent if for every subset $S \subseteq \{1, \ldots, n\}$:
$$P\!\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)$$

This is strictly stronger than pairwise independence (see §4.3).

### 4.2 Conditional Independence

Conditional independence is one of the most important concepts in probabilistic modelling and graphical models.

**Definition 4.2 (Conditional independence).** Events $A$ and $B$ are *conditionally independent given* $C$, written $A \perp B \mid C$, if:
$$P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)$$

Equivalently (when $P(B \cap C) > 0$): $P(A \mid B, C) = P(A \mid C)$ — once we know $C$, learning $B$ gives no additional information about $A$.

**The classic example — explaining away.** Let $A$ = "the grass is wet", $B$ = "the sprinkler is on", $C$ = "it rained". Rain and sprinkler are independent causes of wet grass. But:

- Without knowing whether it rained: $A$ and $B$ are marginally dependent (if the grass is wet, the sprinkler being on is more likely)
- Given that it rained: $A \perp B \mid C$ — knowing the sprinkler status adds nothing once we know it rained

**Caution:** Independence and conditional independence are logically independent:
- $A \perp B$ does NOT imply $A \perp B \mid C$ (conditioning can introduce dependence)
- $A \perp B \mid C$ does NOT imply $A \perp B$ (marginalising out $C$ can introduce dependence)

**For AI:** Conditional independence is the foundation of:
- **Naive Bayes:** assumes features are conditionally independent given the class label
- **Hidden Markov Models:** $x_t \perp x_{t'} \mid z_t$ — observations are independent given the hidden state
- **Bayesian networks:** the factorisation $P(X_1, \ldots, X_n) = \prod_i P(X_i \mid \text{parents}(X_i))$ encodes a set of conditional independence assumptions
- **Attention masks:** attention weights in transformers implement conditional independence (causal masking = future tokens are conditionally independent of past given present)

### 4.3 Pairwise vs Mutual Independence

**Pairwise independence** means $P(A_i \cap A_j) = P(A_i)P(A_j)$ for all pairs $(i,j)$.

**Mutual independence** means the joint probability factorises for every subset (Definition 4.1 for $n$ events).

**Mutual independence implies pairwise independence but not vice versa.** The following counterexample (Bernstein, 1928) shows that pairwise independence can hold without mutual independence:

Roll two fair dice. Define:
- $A$ = "first die shows even" — $P(A) = 1/2$
- $B$ = "second die shows even" — $P(B) = 1/2$
- $C$ = "sum of dice is even" — $P(C) = 1/2$

Check pairwise: $P(A \cap B) = 1/4 = P(A)P(B)$ ✓, $P(A \cap C) = 1/4$ ✓, $P(B \cap C) = 1/4$ ✓.

But $P(A \cap B \cap C) = P(\text{both even, sum even}) = P(\text{both even}) = 1/4 \neq P(A)P(B)P(C) = 1/8$.

So $A, B, C$ are pairwise independent but not mutually independent.

**For AI:** The i.i.d. (independent and identically distributed) assumption in ML training is a mutual independence assumption: $P((\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(n)}, y^{(n)})) = \prod_{i=1}^n P(\mathbf{x}^{(i)}, y^{(i)})$. This assumption underlies the derivation of cross-entropy loss as a maximum likelihood estimator.

### 4.4 Why Independence Matters for AI

Independence assumptions make otherwise intractable problems tractable.

**Without independence:** The joint distribution of $n$ binary random variables requires $2^n - 1$ parameters. For $n = 100$ features, this is $2^{100} - 1 \approx 10^{30}$ — utterly infeasible.

**With conditional independence (Naive Bayes):** Given class $C$, features are independent: $P(\mathbf{x} \mid C) = \prod_{j=1}^n P(x_j \mid C)$. Now only $n$ parameters per class — linear in $n$.

**Independence in optimisation:** Mini-batch gradient descent assumes that samples in a batch are independent draws from the training distribution. This independence makes the batch gradient an unbiased estimator of the full gradient.

**Independence and parallelism:** Independently distributed data can be processed in parallel without communication — the foundation of distributed ML training (data parallelism).


---

## 5. Random Variables — Formal Foundation

### 5.1 Definition: Random Variable as a Measurable Function

The outcomes in a sample space $\Omega$ can be anything — text strings, images, dice faces. To do mathematics with them, we need to convert them to numbers. A random variable does exactly this.

**Definition 5.1 (Random variable).** A *random variable* $X$ is a (measurable) function from the sample space $\Omega$ to the real line $\mathbb{R}$:
$$X : \Omega \to \mathbb{R}, \quad \omega \mapsto X(\omega)$$

The *measurability* requirement ensures that $\{\omega \in \Omega : X(\omega) \leq x\}$ is a valid event (belongs to $\mathcal{F}$) for every $x \in \mathbb{R}$, so that we can assign it a probability. For all practical purposes, every function you would naturally write down is measurable.

**Examples:**
- Die roll: $\Omega = \{1,\ldots,6\}$, $X(\omega) = \omega$ (the outcome itself)
- Coin flip: $\Omega = \{H, T\}$, $X(H) = 1$, $X(T) = 0$ (Bernoulli representation)
- Sentence length: $\Omega = \{\text{all English sentences}\}$, $X(\omega) = |\omega|$ (number of words)
- Model loss: $\Omega = \{\text{all possible mini-batches}\}$, $X(\omega) = \mathcal{L}(\theta; \omega)$

**Non-examples:**
- The sample space $\Omega$ itself is not a random variable (it is the domain, not a function)
- A deterministic constant $c$ can be viewed as a degenerate random variable $X(\omega) = c$ for all $\omega$, but calling it "random" is misleading

**Notation:** Random variables are typically denoted by uppercase letters ($X$, $Y$, $Z$); their realised values (specific numbers) by lowercase ($x$, $y$, $z$). So "$X = x$" means "the random variable $X$ takes the value $x$."

The event $\{X \leq x\} = \{\omega \in \Omega : X(\omega) \leq x\}$ is well-defined, and we write $P(X \leq x)$ as shorthand for $P(\{X \leq x\})$.

### 5.2 Discrete vs Continuous — Taxonomy

Random variables split into two main types based on the range of $X$.

```
TAXONOMY OF RANDOM VARIABLES
════════════════════════════════════════════════════════════════════════

  Random Variable X: Ω → ℝ
  │
  ├── Discrete: X takes countably many values {x₁, x₂, x₃, ...}
  │   │
  │   ├── Characterised by PMF: p(x) = P(X = x)
  │   ├── CDF: F(x) = Σ p(xᵢ) for all xᵢ ≤ x   (staircase)
  │   └── Examples: Bernoulli, Geometric, Binomial, Poisson, Categorical
  │
  └── Continuous: X takes values in an interval (uncountably many)
      │
      ├── Characterised by PDF: f(x) such that P(a≤X≤b) = ∫ₐᵇ f(x)dx
      ├── CDF: F(x) = ∫₋∞ˣ f(t)dt   (smooth, differentiable)
      └── Examples: Uniform, Gaussian, Exponential, Beta, Gamma

  Note: "Mixed" random variables exist (CDF is a mix of jumps and
  smooth parts) but are rare in practice.

════════════════════════════════════════════════════════════════════════
```

**Key distinction — probability at a point:**
- Discrete: $P(X = x)$ can be positive (it is the PMF value)
- Continuous: $P(X = x) = 0$ for every specific $x$ (the probability of hitting any exact value is zero). Probabilities come from integrating over intervals.

**For AI:** The distinction is crucial for loss functions. Cross-entropy loss for classification uses a discrete distribution (Categorical) over a finite vocabulary or class set. For regression, the loss is often the negative log-likelihood of a Gaussian (continuous distribution). Diffusion models use a mixture: discrete time steps with continuous noise.

### 5.3 The Cumulative Distribution Function

The CDF provides a unified description of both discrete and continuous random variables.

**Definition 5.2 (Cumulative distribution function).** The *CDF* of a random variable $X$ is:
$$F_X(x) = P(X \leq x), \quad x \in \mathbb{R}$$

**Examples:**
- Fair die: $F(x) = 0$ for $x < 1$; $F(x) = k/6$ for $k \leq x < k+1$, $k = 1, \ldots, 6$; $F(x) = 1$ for $x \geq 6$. A staircase with jumps of $1/6$ at each integer.
- Standard normal: $F(x) = \Phi(x) = \frac{1}{\sqrt{2\pi}}\int_{-\infty}^x e^{-t^2/2}\,dt$. A smooth S-shaped curve.
- Bernoulli($p$): $F(x) = 0$ for $x < 0$; $F(x) = 1-p$ for $0 \leq x < 1$; $F(x) = 1$ for $x \geq 1$.

Computing interval probabilities from the CDF:
$$P(a < X \leq b) = F(b) - F(a)$$
$$P(X > x) = 1 - F(x)$$

### 5.4 CDF Properties and the Fundamental Theorem

**Theorem 5.1 (CDF properties).** Any CDF satisfies:

1. **Monotone non-decreasing:** $x_1 \leq x_2 \implies F(x_1) \leq F(x_2)$
2. **Right-continuous:** $\lim_{y \downarrow x} F(y) = F(x)$ for all $x$
3. **Limits:** $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to +\infty} F(x) = 1$
4. **Jump characterisation:** $P(X = x) = F(x) - F(x^-)$ where $F(x^-) = \lim_{y \uparrow x} F(y)$. For continuous $X$, this is 0 everywhere; for discrete $X$, it equals the PMF at jump points.

*These four properties completely characterise CDFs* — any function satisfying them is the CDF of some random variable. This is both the universality and the tractability of the CDF.

**Fundamental theorem (continuous case):** For continuous $X$ with CDF $F$ and PDF $f$:
$$f(x) = \frac{d}{dx} F(x), \quad F(x) = \int_{-\infty}^x f(t)\, dt$$

The PDF is the derivative of the CDF; the CDF is the integral of the PDF.

---

## 6. Discrete Random Variables

### 6.1 Probability Mass Function

**Definition 6.1 (PMF).** For a discrete random variable $X$ with countable range $\mathcal{X} = \{x_1, x_2, \ldots\}$, the *probability mass function* is:
$$p_X(x) = P(X = x), \quad x \in \mathcal{X}$$

**Properties:**
1. $p_X(x) \geq 0$ for all $x$ (Axiom 1)
2. $\sum_{x \in \mathcal{X}} p_X(x) = 1$ (Axiom 2)
3. $p_X(x) = 0$ for $x \notin \mathcal{X}$

**CDF from PMF:** $F_X(x) = \sum_{x_i \leq x} p_X(x_i)$ — sum all PMF values at or below $x$.

**Non-examples of valid PMFs:**
- $p(0) = 0.6, p(1) = 0.6$: sum = 1.2 > 1
- $p(0) = -0.2, p(1) = 1.2$: negative value
- $p(k) = 1/k$ for $k = 1, 2, \ldots$: $\sum 1/k = \infty$ (harmonic series diverges)

**For AI:** Every classifier's output layer implicitly defines a PMF over classes. The softmax function $\text{softmax}(\mathbf{z})_k = e^{z_k}/\sum_j e^{z_j}$ produces a vector of non-negative values summing to 1 — a valid PMF over the $K$ classes. Training minimises the cross-entropy between this PMF and the one-hot PMF of the true label.

### 6.2 Bernoulli Distribution — The Canonical Example

The Bernoulli distribution is the simplest non-trivial probability distribution: a single binary trial.

**Definition 6.2 (Bernoulli distribution).** $X \sim \text{Bernoulli}(p)$ for $p \in [0,1]$ if:
$$P(X = 1) = p, \quad P(X = 0) = 1 - p$$

or equivalently: $p_X(x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$.

**Properties:**
- **Mean:** $\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1-p) = p$
- **Variance:** $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = p - p^2 = p(1-p)$
- **Maximum variance** at $p = 1/2$: $\operatorname{Var}(X) = 1/4$
- **CDF:** $F(x) = 0$ for $x < 0$; $F(x) = 1-p$ for $0 \leq x < 1$; $F(x) = 1$ for $x \geq 1$

**ML applications of the Bernoulli distribution:**
- **Binary classification:** $Y \sim \text{Bernoulli}(\sigma(\mathbf{w}^\top \mathbf{x}))$ where $\sigma$ is sigmoid
- **Dropout:** each neuron is active with probability $1 - p_{\text{drop}}$; the mask is $\text{Bernoulli}(1-p_{\text{drop}})^d$
- **Coin-flip initialisation tests:** checking whether random initialisation is breaking symmetry
- **Binary attention masks:** a position is masked with $\text{Bernoulli}(p_{\text{mask}})$ in masked language modelling (BERT, RoBERTa)
- **Bernoulli reward signals:** in RL, some environments give binary rewards ($0$ or $1$)

**Bernoulli and entropy.** The *binary entropy* of a Bernoulli($p$) variable is:
$$H(p) = -p \log p - (1-p) \log(1-p)$$

This is maximised at $p = 1/2$ (maximum uncertainty) and equals 0 at $p \in \{0, 1\}$ (certainty). Binary cross-entropy loss is exactly this entropy when the true label is $p \in \{0,1\}$ and the model predicts $\hat{p}$: $\mathcal{L} = -y \log \hat{p} - (1-y)\log(1-\hat{p})$.

### 6.3 Geometric Distribution

The Geometric distribution models waiting times: how many Bernoulli trials until the first success?

**Definition 6.3 (Geometric distribution).** $X \sim \text{Geometric}(p)$ for $p \in (0,1]$ if:
$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

(Here $X$ is the trial number of the first success; an alternative convention has $X$ = number of failures before first success, giving PMF $(1-p)^k p$ for $k = 0, 1, 2, \ldots$)

**Properties:**
- **Mean:** $\mathbb{E}[X] = 1/p$
- **Variance:** $\operatorname{Var}(X) = (1-p)/p^2$
- **Memoryless property:** $P(X > m+n \mid X > m) = P(X > n)$ — the past gives no information about the remaining wait time. The Geometric is the only discrete memoryless distribution.

**Geometric series check (normalisation):**
$$\sum_{k=1}^\infty (1-p)^{k-1}p = p \sum_{k=0}^\infty (1-p)^k = p \cdot \frac{1}{1-(1-p)} = p \cdot \frac{1}{p} = 1 \checkmark$$

**ML applications:**
- **Sentence length models:** the length of a generated sentence can be modelled geometrically (each word has probability $p$ of being the last)
- **Number of gradient steps to convergence:** in simplified convergence analyses, the number of steps until $\|\nabla\mathcal{L}\| < \varepsilon$ can have an approximately geometric distribution under certain conditions
- **Retry logic:** in RL exploration, the number of random actions before discovering a reward follows a geometric-like distribution in uniform random environments

### 6.4 Preview: Binomial, Categorical, Poisson

The following distributions have their full treatment in §02-Common-Distributions. They are introduced here briefly for orientation:

> **Preview: Binomial Distribution**
> $\text{Binomial}(n, p)$ counts the number of successes in $n$ independent Bernoulli($p$) trials: $P(X = k) = \binom{n}{k}p^k(1-p)^{n-k}$ for $k = 0, 1, \ldots, n$. Mean $np$, variance $np(1-p)$. As $n \to \infty$ with $np = \lambda$ fixed, the Binomial converges to Poisson($\lambda$).
>
> → _Full treatment: [Common Distributions §2](../02-Common-Distributions/notes.md)_

> **Preview: Categorical Distribution**
> $\text{Categorical}(\mathbf{p})$ with $\mathbf{p} \in \Delta^{K-1}$ (the probability simplex) models selection of one of $K$ categories: $P(X = k) = p_k$. This is the distribution underlying softmax classifiers and language model next-token prediction.
>
> → _Full treatment: [Common Distributions §4](../02-Common-Distributions/notes.md)_

> **Preview: Poisson Distribution**
> $\text{Poisson}(\lambda)$ models the number of events in a fixed interval when events occur independently at constant rate $\lambda$: $P(X = k) = e^{-\lambda}\lambda^k/k!$. Mean = Variance = $\lambda$. Used in modelling word frequencies (Zipf/Poisson approximation), network packet arrivals, and mutation rates.
>
> → _Full treatment: [Common Distributions §3](../02-Common-Distributions/notes.md)_

---

## 7. Continuous Random Variables

### 7.1 Probability Density Function

For continuous random variables, individual points have probability zero. Probability is distributed over intervals via the PDF.

**Definition 7.1 (Probability density function).** A non-negative function $f_X : \mathbb{R} \to [0, \infty)$ is a *PDF* for $X$ if:
1. $f_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f_X(x)\, dx = 1$
3. For any $a \leq b$: $P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$

**Critical distinction:** The PDF $f_X(x)$ is NOT a probability. It is a probability *density* — probability per unit length. In particular, $f_X(x)$ can exceed 1 (though the integral must equal 1).

**Example:** $f_X(x) = 2x$ for $x \in [0,1]$ and $0$ elsewhere. This is a valid PDF: non-negative, integrates to $\int_0^1 2x\,dx = [x^2]_0^1 = 1$. Note $f_X(0.9) = 1.8 > 1$, but this is fine — it is a density.

**Probability at a point:** $P(X = x) = \int_x^x f(t)\,dt = 0$. This is why conditioning on a continuous event requires care and is handled via conditional densities (§03).

**Non-examples of valid PDFs:**
- $f(x) = x$ on $[0,1]$: $\int_0^1 x\,dx = 1/2 \neq 1$ (not normalised)
- $f(x) = -e^{-x}$ for $x \geq 0$: negative values
- $f(x) = 1$ for $x \in [0,2]$: $\int_0^2 1\,dx = 2 \neq 1$

### 7.2 From CDF to PDF: the Derivative Relationship

For a continuous random variable with PDF $f_X$ and CDF $F_X$:

$$F_X(x) = \int_{-\infty}^x f_X(t)\, dt \quad \Longleftrightarrow \quad f_X(x) = \frac{d}{dx} F_X(x)$$

This is the Fundamental Theorem of Calculus applied to probability. The CDF accumulates density; the PDF is the rate of accumulation.

**Using the CDF to compute probabilities:**
$$P(a \leq X \leq b) = F_X(b) - F_X(a)$$
$$P(X > x) = 1 - F_X(x)$$
$$P(X \leq x) = F_X(x)$$

Since continuous $X$ has $P(X = a) = 0$: the endpoints don't matter, $P(a < X \leq b) = P(a \leq X \leq b) = P(a < X < b)$.

### 7.3 Uniform Distribution

The Uniform distribution is the maximum-entropy distribution over a bounded interval: all values are equally likely.

**Definition 7.2 (Uniform distribution).** $X \sim \text{Uniform}(a, b)$ for $a < b$ if:
$$f_X(x) = \begin{cases} \frac{1}{b-a} & a \leq x \leq b \\ 0 & \text{otherwise} \end{cases}$$

**CDF:** $F_X(x) = 0$ for $x < a$; $F_X(x) = (x-a)/(b-a)$ for $a \leq x \leq b$; $F_X(x) = 1$ for $x > b$.

**Properties:**
- **Mean:** $\mathbb{E}[X] = (a+b)/2$ (the midpoint)
- **Variance:** $\operatorname{Var}(X) = (b-a)^2/12$
- **Symmetry:** $f$ is symmetric about $(a+b)/2$

**Derivation of variance:**
$$\mathbb{E}[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\, dx = \frac{1}{b-a} \cdot \frac{b^3 - a^3}{3} = \frac{a^2 + ab + b^2}{3}$$
$$\operatorname{Var}(X) = \frac{a^2+ab+b^2}{3} - \left(\frac{a+b}{2}\right)^2 = \frac{(b-a)^2}{12}$$

**ML applications of Uniform:**
- **Weight initialisation (Xavier/Glorot):** weights drawn from $\text{Uniform}(-1/\sqrt{n}, 1/\sqrt{n})$ to maintain activation variance across layers
- **Batch selection:** selecting which training examples to include in a mini-batch (sampling uniformly without replacement)
- **Data augmentation:** random cropping positions, rotation angles, colour jitter magnitudes drawn from uniform distributions
- **Hyperparameter search:** random search over hyperparameter ranges samples uniformly from the search space

### 7.4 Gaussian Distribution — Introduction

The Gaussian (Normal) distribution is the most important continuous distribution in all of probability and statistics, due to the Central Limit Theorem (§06-Stochastic-Processes) and its mathematical tractability.

**Definition 7.3 (Gaussian distribution).** $X \sim \mathcal{N}(\mu, \sigma^2)$ with mean $\mu \in \mathbb{R}$ and variance $\sigma^2 > 0$ if:
$$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

The **standard normal** $Z \sim \mathcal{N}(0,1)$ has $\mu = 0$, $\sigma^2 = 1$:
$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

Its CDF is $\Phi(z) = \int_{-\infty}^z \phi(t)\, dt$, which has no closed form but is tabulated and implemented in all numerical libraries.

**Properties:**
- **Mean:** $\mathbb{E}[X] = \mu$
- **Variance:** $\operatorname{Var}(X) = \sigma^2$
- **Symmetry:** $f(\mu - x) = f(\mu + x)$ — symmetric about $\mu$
- **68–95–99.7 rule:**
  - $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68.27\%$
  - $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95.45\%$
  - $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.73\%$
- **Standardisation:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $Z = (X-\mu)/\sigma \sim \mathcal{N}(0,1)$
- **Affine stability:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$

**Why the Gaussian is ubiquitous:**
1. **Central Limit Theorem (preview):** The sum of many i.i.d. random variables with finite variance converges to a Gaussian — regardless of the original distribution. See §06.
2. **Maximum entropy:** Among all continuous distributions with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian maximises entropy (is the "least informative" given these constraints)
3. **Conjugacy:** The Gaussian is conjugate to itself under Bayesian updating with Gaussian likelihood — the posterior is also Gaussian

**ML applications of the Gaussian:**
- **Weight initialisation (He/Kaiming):** weights drawn from $\mathcal{N}(0, 2/n_{\text{in}})$ for ReLU networks
- **Noise models:** residuals $y - f_\theta(\mathbf{x}) \sim \mathcal{N}(0, \sigma^2)$ gives mean squared error loss
- **VAE latent space:** $z \sim \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$ with Gaussian prior $\mathcal{N}(0, I)$
- **Diffusion models forward process:** $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$, $\varepsilon \sim \mathcal{N}(0, I)$
- **Score functions:** the score $\nabla_x \log p(x)$ for a Gaussian is linear: $\nabla_x \log \mathcal{N}(x;\mu,\sigma^2) = -(x-\mu)/\sigma^2$

> **Preview: Exponential, Beta, Gamma Distributions**
> The Exponential distribution models waiting times between Poisson events; the Beta models probabilities on $[0,1]$ (conjugate prior for Bernoulli); the Gamma generalises both. Full treatment: [Common Distributions §5, §6, §7](../02-Common-Distributions/notes.md).


---

## 8. Expectation and Variance — Foundations

### 8.1 Expected Value: Definition and Linearity

The expected value (or mean) of a random variable is its probability-weighted average — the "centre of mass" of its distribution.

**Definition 8.1 (Expected value — discrete).** For a discrete $X$ with PMF $p_X$:
$$\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \cdot p_X(x)$$

(provided the sum converges absolutely: $\sum |x| p_X(x) < \infty$).

**Definition 8.2 (Expected value — continuous).** For a continuous $X$ with PDF $f_X$:
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f_X(x)\, dx$$

(provided the integral converges absolutely).

**Examples:**
- Bernoulli$(p)$: $\mathbb{E}[X] = 1 \cdot p + 0 \cdot (1-p) = p$
- Uniform$(0,1)$: $\mathbb{E}[X] = \int_0^1 x \cdot 1\, dx = 1/2$
- $\mathcal{N}(\mu, \sigma^2)$: $\mathbb{E}[X] = \mu$ (by symmetry and definition)
- Fair die: $\mathbb{E}[X] = (1+2+3+4+5+6)/6 = 3.5$

**Non-example (expectation undefined):** The Cauchy distribution with PDF $f(x) = 1/(\pi(1+x^2))$ has $\int_{-\infty}^\infty |x|/(π(1+x^2))\,dx = \infty$, so $\mathbb{E}[X]$ does not exist. This is why heavy-tailed distributions require care.

**Theorem 8.1 (Linearity of expectation).** For any random variables $X$, $Y$ and constants $a$, $b$:
$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

This holds whether or not $X$ and $Y$ are independent — linearity of expectation is unconditional.

*Proof (discrete):* $\mathbb{E}[aX + bY] = \sum_{x,y}(ax+by)P(X=x, Y=y) = a\sum_x x \sum_y P(X=x,Y=y) + b\sum_y y \sum_x P(X=x,Y=y) = a\mathbb{E}[X] + b\mathbb{E}[Y]$. $\square$

**For AI:** Linearity of expectation is why mini-batch gradient descent works. The gradient of the average loss over a batch equals the average of individual gradients:
$$\nabla_\theta \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\!\left[\mathcal{L}(\mathbf{x};\theta)\right] = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\!\left[\nabla_\theta \mathcal{L}(\mathbf{x};\theta)\right]$$

This follows from linearity of both expectation and differentiation.

**Expected value of a function:** For $g : \mathbb{R} \to \mathbb{R}$:
$$\mathbb{E}[g(X)] = \sum_x g(x) p_X(x) \quad (\text{discrete}), \qquad \mathbb{E}[g(X)] = \int g(x) f_X(x)\, dx \quad (\text{continuous})$$

This is the *Law of the Unconscious Statistician (LOTUS)* — you can compute $\mathbb{E}[g(X)]$ directly from the distribution of $X$ without finding the distribution of $g(X)$ first. (Full treatment in §04.)

**Warning:** $\mathbb{E}[g(X)] \neq g(\mathbb{E}[X])$ in general. For example, $\mathbb{E}[X^2] \neq (\mathbb{E}[X])^2$ (the gap is the variance). Jensen's inequality gives the direction: if $g$ is convex, $g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]$.

### 8.2 Variance and Standard Deviation

Variance measures spread — how far a random variable typically deviates from its mean.

**Definition 8.3 (Variance).** The *variance* of $X$ is:
$$\operatorname{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right]$$

**Computational formula:**
$$\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

*Proof:* $\mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu \mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - \mu^2$. $\square$

**Definition 8.4 (Standard deviation).** $\sigma_X = \sqrt{\operatorname{Var}(X)}$. This has the same units as $X$ and $\mathbb{E}[X]$.

**Examples:**
- Bernoulli$(p)$: $\operatorname{Var}(X) = p - p^2 = p(1-p)$. Maximum at $p=1/2$: $\operatorname{Var}(X) = 1/4$.
- Uniform$(a,b)$: $\operatorname{Var}(X) = (b-a)^2/12$
- $\mathcal{N}(\mu,\sigma^2)$: $\operatorname{Var}(X) = \sigma^2$ (by definition)
- Constant $c$: $\operatorname{Var}(c) = 0$ (no randomness)

**Properties of variance:**
1. $\operatorname{Var}(X) \geq 0$, with equality iff $X$ is a constant a.s.
2. $\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)$ (shifting by $b$ doesn't change spread; scaling by $a$ scales variance by $a^2$)
3. $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)$
4. If $X \perp Y$: $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$

**For AI:** Variance appears throughout ML:
- **Bias-variance tradeoff:** test error = bias² + variance + irreducible noise. High-variance models overfit.
- **Gradient variance:** the variance of the stochastic gradient estimator determines training stability. Variance reduction techniques (control variates, importance sampling) reduce this.
- **Batch normalisation:** standardises activations to have mean 0 and variance 1 within each batch, stabilising training.
- **Uncertainty in predictions:** the variance of a Gaussian output head models aleatoric uncertainty.

### 8.3 Key Properties and Common Pitfalls

**Expectation is linear; variance is not.** For independent $X, Y$:
- $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (independence gives factorisation)
- $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ (independence cancels covariance)
- But in general: $\operatorname{Var}(X+Y) \neq \operatorname{Var}(X) + \operatorname{Var}(Y)$

**Jensen's inequality:** For convex $g$:
$$g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]$$

Examples: $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$ (taking $g(x)=x^2$); $e^{\mathbb{E}[X]} \leq \mathbb{E}[e^X]$ (taking $g(x)=e^x$). Jensen's inequality is used to prove Markov's inequality, the EM lower bound (ELBO), and the KL divergence non-negativity.

**Expectation and independence:** $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ holds if $X \perp Y$, but the converse is false: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (uncorrelated) does not imply independence.

### 8.4 Preview: Covariance, LOTUS, MGF

> **Preview: Covariance and Correlation**
> The covariance $\operatorname{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ measures linear dependence between two random variables. The covariance matrix $\Sigma$ of a random vector extends this to multiple dimensions. These are the core tools for analysing multivariate distributions.
>
> → _Full treatment: [Expectation and Moments §3](../04-Expectation-and-Moments/notes.md)_

> **Preview: Law of the Unconscious Statistician (LOTUS)**
> LOTUS allows computing $\mathbb{E}[g(X)]$ using the distribution of $X$ directly: no change-of-variables required. It is the foundational tool for deriving moments and the expected loss under a distribution.
>
> → _Full treatment: [Expectation and Moments §2](../04-Expectation-and-Moments/notes.md)_

> **Preview: Moment Generating Functions**
> The MGF $M_X(t) = \mathbb{E}[e^{tX}]$ encodes all moments of $X$: $\mathbb{E}[X^n] = M_X^{(n)}(0)$. MGFs are the key tool for proving the Central Limit Theorem and computing tail bounds.
>
> → _Full treatment: [Expectation and Moments §5](../04-Expectation-and-Moments/notes.md)_

---

## 9. Transformations of Random Variables

### 9.1 Functions of a Random Variable

Given a random variable $X$ and a function $g : \mathbb{R} \to \mathbb{R}$, the composition $Y = g(X)$ is a new random variable. Computing the distribution of $Y$ from the distribution of $X$ is a central skill.

**Discrete case.** If $X$ is discrete with PMF $p_X$ and $g$ is any function, then $Y = g(X)$ is discrete with:
$$p_Y(y) = P(Y = y) = P(g(X) = y) = \sum_{x : g(x) = y} p_X(x)$$

**Example.** $X \sim \text{Uniform}\{1,2,3,4,5,6\}$ (fair die), $Y = X^2 \bmod 6$. Then $p_Y(1) = P(X^2 \equiv 1 \bmod 6) = P(X \in \{1,5\}) = 1/3$, etc.

**Continuous case — CDF method.** For continuous $X$ and $Y = g(X)$, compute the CDF of $Y$:
$$F_Y(y) = P(Y \leq y) = P(g(X) \leq y) = P(X \in \{x : g(x) \leq y\})$$

then differentiate to get the PDF: $f_Y(y) = F_Y'(y)$.

### 9.2 Change-of-Variables Formula

For monotone transformations, the CDF method yields a clean formula.

**Theorem 9.1 (Change of variables — monotone, 1-D).** Let $X$ have PDF $f_X$ and CDF $F_X$. Let $g$ be a differentiable, strictly monotone function on the support of $X$, with inverse $h = g^{-1}$. Then $Y = g(X)$ has PDF:
$$f_Y(y) = f_X(h(y)) \cdot \left|\frac{dh}{dy}\right| = f_X(h(y)) \cdot \left|\frac{1}{g'(h(y))}\right|$$

*Derivation (increasing $g$):*
$$F_Y(y) = P(Y \leq y) = P(g(X) \leq y) = P(X \leq h(y)) = F_X(h(y))$$
$$f_Y(y) = \frac{d}{dy}F_Y(y) = f_X(h(y)) \cdot h'(y) = f_X(h(y)) \cdot \frac{1}{g'(h(y))}$$

For decreasing $g$, $F_Y(y) = 1 - F_X(h(y))$, and differentiating gives a negative sign, cancelled by the absolute value in the formula.

**Example.** $X \sim \mathcal{N}(0,1)$, $Y = e^X$ (log-normal distribution). Here $g(x) = e^x$, $h(y) = \ln y$, $h'(y) = 1/y$. So:
$$f_Y(y) = \phi(\ln y) \cdot \frac{1}{y} = \frac{1}{\sqrt{2\pi}}\, e^{-(\ln y)^2/2} \cdot \frac{1}{y}, \quad y > 0$$

**Example.** $X \sim \mathcal{N}(\mu, \sigma^2)$, $Y = (X - \mu)/\sigma$ (standardisation). Here $g(x) = (x-\mu)/\sigma$, $h(y) = \mu + \sigma y$, $h'(y) = \sigma$:
$$f_Y(y) = f_X(\mu + \sigma y) \cdot \sigma = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-y^2/2} \cdot \sigma = \frac{1}{\sqrt{2\pi}} e^{-y^2/2} = \phi(y)$$

So $Y \sim \mathcal{N}(0,1)$. ✓

**For AI:** The change-of-variables formula is the mathematical foundation of *normalising flows* — generative models that learn a bijection $g : \mathbb{R}^d \to \mathbb{R}^d$ transforming a simple base distribution (e.g., $\mathcal{N}(0,I)$) into a complex target distribution. The log-density of the transformed variable is:
$$\log p_Y(y) = \log p_X(g^{-1}(y)) + \log |\det J_{g^{-1}}(y)|$$
where $J_{g^{-1}}$ is the Jacobian of the inverse transformation.

### 9.3 Standardisation and the Z-Score

**Definition 9.1 (Z-score / standardisation).** For $X$ with mean $\mu$ and standard deviation $\sigma > 0$:
$$Z = \frac{X - \mu}{\sigma}$$

Then $\mathbb{E}[Z] = 0$ and $\operatorname{Var}(Z) = 1$ regardless of the distribution of $X$.

**Proof:** $\mathbb{E}[Z] = \mathbb{E}[(X-\mu)/\sigma] = (\mathbb{E}[X] - \mu)/\sigma = 0$. $\operatorname{Var}(Z) = \operatorname{Var}(X)/\sigma^2 = \sigma^2/\sigma^2 = 1$. $\square$

**Applications in ML:**
- **Feature normalisation (preprocessing):** standardise each feature so that the gradient has comparable scale across dimensions, improving optimiser convergence
- **Batch normalisation (BN):** standardises activations within each mini-batch, then applies learned scale and shift parameters $(\gamma, \beta)$: $\hat{x} = (x - \hat{\mu})/\hat{\sigma}$, output $= \gamma\hat{x} + \beta$
- **Layer normalisation (LN):** same idea but normalising across features rather than the batch dimension — standard in transformer models
- **RMS Normalisation (RMSNorm):** a simplified variant using only the RMS (root mean square), dropping the mean subtraction — used in LLaMA, Gemma, and Mistral models: $\hat{x}_i = x_i / \text{RMS}(\mathbf{x})$

---

## 10. Applications in Machine Learning

### 10.1 Cross-Entropy Loss as Negative Log-Likelihood

**Setup.** Suppose we observe data $\mathcal{D} = \{(x^{(1)}, y^{(1)}), \ldots, (x^{(n)}, y^{(n)})\}$ drawn i.i.d. from the true distribution $p_{\text{data}}(y \mid x)$. A model $f_\theta$ with parameters $\theta$ defines a predicted distribution $p_\theta(y \mid x)$.

**Maximum likelihood estimation (MLE).** We want to choose $\theta$ to maximise the probability assigned to the observed data:
$$\theta^* = \arg\max_\theta \prod_{i=1}^n p_\theta(y^{(i)} \mid x^{(i)}) \quad [\text{likelihood}]$$

Taking the logarithm (monotone transformation, same argmax):
$$\theta^* = \arg\max_\theta \sum_{i=1}^n \log p_\theta(y^{(i)} \mid x^{(i)}) = \arg\min_\theta \underbrace{-\frac{1}{n}\sum_{i=1}^n \log p_\theta(y^{(i)} \mid x^{(i)})}_{\text{negative log-likelihood loss}}$$

**For classification.** With $K$ classes, $p_\theta(y \mid x) = \text{Categorical}(\text{softmax}(f_\theta(x)))_y$:
$$-\log p_\theta(y \mid x) = -\log \text{softmax}(f_\theta(x))_y = -\log \frac{e^{z_y}}{\sum_j e^{z_j}} = \log\sum_j e^{z_j} - z_y$$

where $z = f_\theta(x)$ are the logits. For a one-hot label $\mathbf{y}$, this is exactly the *cross-entropy loss*:
$$\mathcal{L}(\theta; x, y) = -\sum_{k=1}^K y_k \log p_\theta(k \mid x) = H(p_{\text{data}}, p_\theta)$$

**For regression.** If $p_\theta(y \mid x) = \mathcal{N}(f_\theta(x), \sigma^2)$:
$$-\log p_\theta(y \mid x) = \frac{1}{2\sigma^2}(y - f_\theta(x))^2 + \text{const}$$

Minimising negative log-likelihood under a Gaussian noise model is equivalent to minimising mean squared error.

**For language modelling.** Each token has distribution $p_\theta(x_t \mid x_{<t}) = \text{Categorical}(\text{softmax}(z_t))$. The NLL loss over a sequence of $T$ tokens is:
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})$$

This is the standard pretraining loss for GPT-family models. The perplexity is $\exp(\mathcal{L})$ — a lower perplexity means the model assigns higher probability to the actual next token.

### 10.2 Bayesian Inference: Prior, Likelihood, Posterior

Bayesian inference applies Bayes' theorem to update beliefs about parameters $\theta$ after observing data $\mathcal{D}$:

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} \propto \underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}$$

The normalising constant $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta$ is the *evidence* or *marginal likelihood*.

**MAP estimation.** The Maximum A Posteriori (MAP) estimate is the mode of the posterior:
$$\theta^*_{\text{MAP}} = \arg\max_\theta \log p(\theta \mid \mathcal{D}) = \arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]$$

With a Gaussian prior $p(\theta) = \mathcal{N}(0, \lambda^{-1}I)$, the $\log p(\theta)$ term becomes $-\lambda\|\theta\|^2/2$ — i.e., L2 regularisation (weight decay). MAP estimation with a Gaussian prior is exactly regularised MLE.

**For AI:** Bayesian thinking is increasingly central to modern AI:
- **Model uncertainty (epistemic):** the posterior over models captures uncertainty from limited data — Bayesian deep learning approximates this posterior
- **RLHF reward model:** human preferences give a likelihood $P(\text{human prefers } y_1 \text{ over } y_2 \mid r_1, r_2)$; updating the reward model is a posterior update
- **In-context learning:** some interpretations frame few-shot ICL as implicit Bayesian updating over hypotheses about the task

### 10.3 Probabilistic Language Models

A language model defines a probability distribution over sequences of tokens. Given vocabulary $\mathcal{V}$, a sequence $\mathbf{x} = (x_1, \ldots, x_T)$ is assigned probability:

$$P(\mathbf{x}) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_T \mid x_1, \ldots, x_{T-1}) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

by the chain rule of probability (§3.4). Each conditional $P(x_t \mid x_{<t})$ is a Categorical distribution over $|\mathcal{V}|$ tokens, produced by the transformer's softmax output.

**Sampling strategies as distribution operations:**
- **Greedy decoding:** always take $\arg\max P(x_t \mid x_{<t})$ — equivalent to a degenerate point mass
- **Temperature scaling:** replace logits $z$ with $z/\tau$; $\tau \to 0$ approaches greedy, $\tau \to \infty$ approaches uniform
- **Top-$k$ sampling:** restrict to the $k$ most probable tokens, renormalise — truncates the distribution
- **Top-$p$ (nucleus) sampling:** restrict to the smallest set of tokens whose cumulative probability $\geq p$, renormalise — adapts the cutoff to the entropy of the distribution

**Bits-per-character (BPC) and perplexity:** The quality of a language model is measured by how much probability mass it assigns to held-out text. Perplexity = $\exp(-\frac{1}{T}\sum_t \log P(x_t \mid x_{<t}))$. A perplexity of $k$ means the model is "as uncertain as a uniform distribution over $k$ choices" at each step.

### 10.4 Sources of Randomness in Neural Network Training

Training a neural network involves multiple distinct sources of randomness, each with a different probability distribution:

| Source | Distribution | When it occurs |
|---|---|---|
| Weight initialisation | $\mathcal{N}(0, \sigma^2)$ or Uniform | Once, at start |
| Mini-batch selection | Uniform (without replacement) | Every step |
| Dropout masks | $\text{Bernoulli}(1-p_{\text{drop}})^d$ | Every forward pass |
| Data augmentation | Uniform (flips), Uniform (crop positions), Gaussian (colour jitter) | Every sample |
| Label smoothing | Convex combination of Categorical and Uniform | During loss computation |
| Noise injection (DP-SGD) | $\mathcal{N}(0, \sigma^2 C^2)$ (clipped Gaussian) | Every gradient update |
| Sampling from generative models | Model-defined (Categorical, Gaussian, etc.) | Inference |

Each source of randomness introduces variance into the training process. Understanding their distributions is essential for:
- Reproducibility (fixing random seeds)
- Differential privacy analysis (noise calibration)
- Variance reduction in gradient estimation
- Debugging training instabilities (distinguishing random vs systematic failures)


---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Confusing $P(A \mid B)$ with $P(B \mid A)$ | These are completely different quantities (prosecutor's fallacy, base rate neglect) | Always identify which is the "given" and which is the "unknown" before applying Bayes' theorem |
| 2 | Treating $f_X(x)$ as a probability | The PDF is a density — it can exceed 1, and individual point probabilities are always 0 for continuous $X$ | Probabilities come from integrating the PDF: $P(a \leq X \leq b) = \int_a^b f(x)dx$ |
| 3 | Assuming independence from $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ | Uncorrelated does not imply independent; independence is strictly stronger | Use the definition: $X \perp Y \iff P(X \leq x, Y \leq y) = P(X \leq x)P(Y \leq y)$ for all $x,y$ |
| 4 | Applying Axiom 3 to non-disjoint events | $P(A \cup B) = P(A) + P(B)$ holds only when $A \cap B = \emptyset$ | Use inclusion-exclusion: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| 5 | Forgetting to check normalisation when constructing a PMF/PDF | An un-normalised function cannot be a PMF or PDF | Always verify $\sum_x p(x) = 1$ (discrete) or $\int f(x)dx = 1$ (continuous) |
| 6 | Using $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$ without checking independence | Variance is additive only for uncorrelated (not just any) random variables | Add the covariance term: $\operatorname{Var}(X+Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) + 2\operatorname{Cov}(X,Y)$ |
| 7 | Confusing $\mathbb{E}[g(X)]$ with $g(\mathbb{E}[X])$ | Jensen's inequality shows these differ whenever $g$ is nonlinear | Compute $\mathbb{E}[g(X)]$ using LOTUS: $\mathbb{E}[g(X)] = \sum_x g(x)p(x)$ or $\int g(x)f(x)dx$ |
| 8 | Confusing pairwise independence with mutual independence | Events can be pairwise independent but not mutually independent (Bernstein example) | For $n > 2$ events, verify independence for all $2^n - n - 1$ subsets, or use a structural argument |
| 9 | Assuming $P(A \cap B) = 0$ when $A$ and $B$ are independent | Independence means $P(A \cap B) = P(A)P(B)$; this is zero only when $P(A) = 0$ or $P(B) = 0$ | Independence ≠ mutual exclusivity. Disjoint events ($A \cap B = \emptyset$) with positive probability are necessarily dependent |
| 10 | Applying the change-of-variables formula without the absolute value of the Jacobian | For decreasing transformations, the Jacobian $dh/dy$ is negative; omitting the absolute value gives a negative PDF | Always write $f_Y(y) = f_X(h(y)) \cdot |h'(y)|$ |
| 11 | Concluding $X$ and $Y$ are independent from a correlation of zero in non-Gaussian data | Zero correlation $\Leftrightarrow$ uncorrelated, which does NOT imply independence for non-Gaussian distributions | Independence implies zero correlation, but not vice versa. Check independence directly or use copulas |
| 12 | Treating conditional probability as symmetric: $P(A \mid B) = P(B \mid A)$ | $P(A \mid B) = P(B \mid A)$ only if $P(A) = P(B)$ (from Bayes' theorem with equal priors) | Apply Bayes' theorem to convert between the two: $P(B \mid A) = P(A \mid B) P(B) / P(A)$ |

---

## 12. Exercises

**Exercise 1 ★ — Axiom Derivations**
Using only the three Kolmogorov axioms, prove each of the following:
(a) $P(\emptyset) = 0$
(b) $P(A^c) = 1 - P(A)$
(c) $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (inclusion-exclusion for two events)
(d) If $A \subseteq B$ then $P(A) \leq P(B)$
(e) $0 \leq P(A) \leq 1$ for all events $A$

*Each proof should cite exactly which axiom(s) and which previously proved results it uses.*

**Exercise 2 ★ — Bayes' Theorem Application**
A medical test for a disease has sensitivity 95% (true positive rate) and specificity 98% (true negative rate). The disease affects 0.5% of the population.
(a) Define events $D$ (has disease) and $T$ (tests positive). State all given probabilities.
(b) Compute $P(T)$ using the law of total probability.
(c) Compute $P(D \mid T)$ using Bayes' theorem.
(d) Explain why the result might surprise a clinician unfamiliar with Bayes' theorem.
(e) How would the answer change if the disease prevalence were 10% instead of 0.5%? Compute and interpret.

**Exercise 3 ★ — PMF Construction and Verification**
A biased die has $P(X = k) = c \cdot k$ for $k \in \{1, 2, 3, 4, 5, 6\}$.
(a) Find the normalisation constant $c$.
(b) Verify this is a valid PMF.
(c) Compute $\mathbb{E}[X]$.
(d) Compute $\operatorname{Var}(X)$.
(e) What is $P(X \geq 4)$? Compare to the fair die.

**Exercise 4 ★★ — CDF Analysis**
Let $X$ have CDF $F(x) = 1 - e^{-\lambda x}$ for $x \geq 0$ and $F(x) = 0$ for $x < 0$ (Exponential distribution).
(a) Show this is a valid CDF (check all four properties from Theorem 5.1).
(b) Find the PDF $f(x)$ by differentiating the CDF.
(c) Compute $P(X > t)$ for any $t > 0$.
(d) Show the memoryless property: $P(X > s+t \mid X > s) = P(X > t)$ for all $s, t > 0$.
(e) For $\lambda = 1$: compute $P(1 \leq X \leq 2)$ exactly and numerically.

**Exercise 5 ★★ — Independence and Conditional Independence**
Roll two fair dice, letting $X$ be the result of die 1 and $Y$ the result of die 2. Define $Z = X + Y$ (the sum).
(a) Verify that $X$ and $Y$ are independent.
(b) Show that $X$ and $Z$ are not independent (hint: compute $P(X=1, Z=2)$ and compare to $P(X=1)P(Z=2)$).
(c) Now condition on the event $\{Z = 7\}$. Show that $X$ and $Y$ are conditionally independent given $Z = 7$ fails — or more precisely, show $P(X=1, Y=6 \mid Z=7) = P(X=1 \mid Z=7) P(Y=6 \mid Z=7)$ and check whether this holds.
(d) Explain in words why knowing $Z$ induces dependence between $X$ and $Y$.

**Exercise 6 ★★ — Change of Variables**
Let $X \sim \text{Uniform}(0, 1)$.
(a) Find the PDF of $Y = -\ln X$. What distribution is this?
(b) Find the PDF of $W = X^2$.
(c) Find the CDF and PDF of $V = \sqrt{X}$.
(d) If $U \sim \text{Uniform}(0,1)$, show that $F_X^{-1}(U)$ has the same distribution as $X$ for any CDF $F_X$ (the inverse CDF / quantile transform method).
(e) How is part (d) used in sampling from arbitrary distributions in ML?

**Exercise 7 ★★★ — Cross-Entropy Loss Derivation**
Consider a $K$-class classification problem with model $p_\theta(y \mid \mathbf{x}) = \text{softmax}(f_\theta(\mathbf{x}))_y$.
(a) Write the negative log-likelihood for a single example $(\mathbf{x}, y^*)$.
(b) Show this equals the cross-entropy $H(p_{\text{true}}, p_\theta)$ where $p_{\text{true}}$ is the one-hot distribution.
(c) Show the gradient of the NLL w.r.t. logits $\mathbf{z} = f_\theta(\mathbf{x})$ is $\text{softmax}(\mathbf{z}) - \mathbf{e}_{y^*}$ (softmax output minus one-hot target).
(d) Explain why label smoothing replaces the one-hot $p_{\text{true}}$ with $(1-\varepsilon)\mathbf{e}_{y^*} + \varepsilon/K \cdot \mathbf{1}$, and how this changes the loss.
(e) For language modelling with vocabulary size $|\mathcal{V}| = 32000$: if the model assigns probability 0.8 to the correct next token, what is the NLL? The perplexity?

**Exercise 8 ★★★ — Probabilistic Analysis of Dropout**
During training, dropout independently sets each of $d$ activations to 0 with probability $p_{\text{drop}}$, and scales the remaining by $1/(1-p_{\text{drop}})$.
(a) Let $M_i \sim \text{Bernoulli}(1-p_{\text{drop}})$ be the mask for activation $i$. What is $\mathbb{E}[M_i]$?
(b) Let $h_i$ be the pre-dropout activation. Define $\tilde{h}_i = M_i \cdot h_i / (1-p_{\text{drop}})$. Show $\mathbb{E}[\tilde{h}_i] = h_i$ (inverted dropout preserves the expected activation).
(c) Compute $\operatorname{Var}(\tilde{h}_i)$ as a function of $h_i$, $p_{\text{drop}}$.
(d) The output of a layer is $s = \sum_{i=1}^d \tilde{h}_i$. Assuming $h_i$ and $M_i$ are independent across neurons, what is $\operatorname{Var}(s)$?
(e) Explain why high $p_{\text{drop}}$ increases gradient variance and can destabilise training. At what $p_{\text{drop}}$ is variance maximised?

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | Impact on AI/LLMs |
|---|---|
| Kolmogorov axioms | The foundation of every probabilistic model. MLE, MAP, VAEs, diffusion models, Bayesian NNs — all rest on these three axioms. Knowing the axioms lets you verify whether a proposed probability model is well-defined. |
| Conditional probability | The core operation in Bayesian inference and autoregressive language modelling. Every $P(x_t \mid x_{<t})$ in a language model is a conditional probability. Causal attention implements conditional independence structure. |
| Bayes' theorem | Underlies RLHF (updating reward model beliefs from human feedback), Bayesian hyperparameter search, and the theoretical analysis of in-context learning as implicit Bayesian inference. |
| Independence | The i.i.d. assumption justifies mini-batch gradient descent. Conditional independence defines the graphical structure of Bayesian networks and the Markov property of diffusion models. |
| PMF / Categorical distribution | Every token prediction in a language model is a Categorical PMF. Softmax produces a valid PMF over $|\mathcal{V}|$ tokens. Sampling strategies (top-$k$, top-$p$, temperature) are operations on this PMF. |
| Gaussian distribution | Used in weight initialisation (He, Glorot), VAE latent spaces, diffusion forward processes, Gaussian process priors, and noise injection for differential privacy. The CLT (§06) explains its ubiquity. |
| Expected value | The training objective is an expected loss $\mathbb{E}_{\mathbf{x},y}[\mathcal{L}(\theta)]$ over the data distribution. Linearity of expectation justifies mini-batch gradient estimation. |
| Variance | Gradient variance determines training stability. BN and LN are variance-control mechanisms. Dropout variance analysis guides the choice of dropout rate. |
| Change of variables | Foundation of normalising flows (Glow, RealNVP, neural spline flows). The change-of-variables formula computes the density of a transformed distribution. |
| Negative log-likelihood | The universal training objective. Cross-entropy loss (classification), MSE (regression under Gaussian noise), ELBO (VAEs) all derive from minimising NLL. |

---

## Conceptual Bridge

This section establishes the probability-theoretic language that all subsequent sections build upon. We began from the Kolmogorov axioms — three simple rules that define what probabilities must be — and derived the core computational rules (complement, union, conditioning, total probability, Bayes) that allow us to reason under uncertainty. The random variable formalism converts outcomes to numbers, enabling the full machinery of analysis.

**Looking back:** The section draws on set theory (union, intersection, complement — assumed), on calculus for the continuous case (integration for CDFs and PDFs, differentiation for the fundamental theorem), and on the concept of a function (random variables are functions). The proof techniques (direct calculation from axioms, algebraic manipulation) mirror those in linear algebra and calculus sections of this course.

**Looking forward — within this chapter:**
- §02-Common-Distributions takes the distribution vocabulary started here (Bernoulli, Uniform, Gaussian) and completes it: Binomial, Poisson, Exponential, Beta, Dirichlet, Categorical, Student-t, plus the exponential family.
- §03-Joint-Distributions extends the single-variable machinery to multiple random variables: joint PDFs, marginalisation, and Bayes' theorem for random variables rather than events.
- §04-Expectation-and-Moments develops expectation much more deeply: LOTUS, covariance matrices, MGFs, characteristic functions, and the full moment theory.
- §05-Concentration-Inequalities proves how expectations and variances control tail probabilities (Markov, Chebyshev, Hoeffding) — the mathematical tools for generalisation guarantees.
- §06-Stochastic-Processes introduces the Central Limit Theorem (explaining Gaussian ubiquity) and Gaussian processes (infinite-dimensional generalisations).
- §07-Markov-Chains uses conditional independence (§4.2) and Bayes' theorem (§3.6) to define memoryless sequential processes and MCMC sampling.

**Looking forward — across chapters:**
- Chapter 7 (Statistics) builds estimation theory (MLE, MAP, confidence intervals) directly on the probability foundation here.
- Chapter 8 (Optimisation) uses the Gaussian noise model (§7.4) and NLL loss (§10.1) to motivate SGD and Adam.
- Chapter 9 (Information Theory) defines entropy as $-\mathbb{E}[\log p(X)]$ and KL divergence in terms of two probability measures — direct extensions of §8.1 and §10.1.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §04 Calculus Fundamentals
  §05 Multivariate Calculus
  │
  └── §06 Probability Theory
      ├── 01-Introduction-and-Random-Variables   ◄══ YOU ARE HERE
      │     (probability spaces, RVs, CDF/PDF/PMF, E[X], Var[X])
      ├── 02-Common-Distributions
      │     (Binomial, Poisson, Gaussian, Beta, Dirichlet, ...)
      ├── 03-Joint-Distributions
      │     (joint PDF, marginalisation, Bayes for RVs)
      ├── 04-Expectation-and-Moments
      │     (LOTUS, covariance, MGF, higher moments)
      ├── 05-Concentration-Inequalities
      │     (Markov, Chebyshev, Hoeffding, PAC bounds)
      ├── 06-Stochastic-Processes
      │     (LLN, CLT, Gaussian processes)
      └── 07-Markov-Chains
            (MCMC, detailed balance, stationary distributions)

════════════════════════════════════════════════════════════════════════
```

[← Back to Probability Theory](../README.md) | [Next: Common Distributions →](../02-Common-Distributions/notes.md)

---

## Appendix A: The Gaussian Normalisation Integral

The fact that $\int_{-\infty}^{\infty} e^{-x^2/2} dx = \sqrt{2\pi}$ is not obvious. Here is the classical proof using a two-dimensional trick.

**Claim:** $I = \int_{-\infty}^{\infty} e^{-x^2/2} dx = \sqrt{2\pi}$.

**Proof (Gaussian integral via polar coordinates):**
$$I^2 = \left(\int_{-\infty}^{\infty} e^{-x^2/2} dx\right)\left(\int_{-\infty}^{\infty} e^{-y^2/2} dy\right) = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\, dx\, dy$$

Convert to polar coordinates $(r, \theta)$ with $x = r\cos\theta$, $y = r\sin\theta$, $dx\, dy = r\, dr\, d\theta$:
$$I^2 = \int_0^{2\pi}\int_0^{\infty} e^{-r^2/2} r\, dr\, d\theta = 2\pi \int_0^{\infty} r e^{-r^2/2}\, dr$$

Let $u = r^2/2$, $du = r\,dr$:
$$I^2 = 2\pi \int_0^{\infty} e^{-u}\, du = 2\pi \cdot 1 = 2\pi$$

Therefore $I = \sqrt{2\pi}$ (since $I > 0$). $\square$

This means the normalisation constant $1/\sqrt{2\pi\sigma^2}$ in the Gaussian PDF is exactly what is needed to make the integral equal 1. The proof that the Gaussian with any $\mu, \sigma^2$ integrates to 1 follows by the substitution $z = (x-\mu)/\sigma$.

---

## Appendix B: Discrete Probability — Worked Examples in Full

### B.1 The Birthday Problem

**Question:** In a room of $n$ people, what is the probability that at least two share a birthday (assuming 365 days, uniform distribution, no leap years)?

This is a classic application of the complement rule.

**Let $A$ = "at least two people share a birthday"** and $A^c$ = "all birthdays are distinct."

$$P(A^c) = \frac{365}{365} \cdot \frac{364}{365} \cdot \frac{363}{365} \cdots \frac{365-n+1}{365} = \frac{365!/(365-n)!}{365^n}$$

The probability of at least one shared birthday:
$$P(A) = 1 - P(A^c) = 1 - \prod_{k=0}^{n-1}\frac{365-k}{365}$$

| $n$ | $P(\text{at least one match})$ |
|---|---|
| 10 | 11.7% |
| 23 | 50.7% — **crossover point** |
| 50 | 97.0% |
| 70 | 99.9% |

**For AI:** The birthday paradox is closely related to hash collision probability and the birthday attack in cryptography. In machine learning, it appears in:
- **Retrieval collision:** The probability that two embeddings in a high-dimensional space are closer than a threshold depends on how many embeddings exist — birthday-paradox type reasoning
- **Token collision in tokenisation:** In BPE tokenisation, character pairs collide and merge; the frequency of collisions follows birthday-paradox dynamics
- **Data deduplication:** Estimating the probability of exact or near-duplicate examples in large training datasets

### B.2 Geometric Series and PMF Normalisation

Many PMFs involve geometric series. Recall:
$$\sum_{k=0}^{\infty} r^k = \frac{1}{1-r}, \quad |r| < 1$$

**Application to Geometric PMF:** $P(X = k) = (1-p)^{k-1}p$ for $k = 1, 2, 3, \ldots$

$$\sum_{k=1}^{\infty}(1-p)^{k-1}p = p\sum_{j=0}^{\infty}(1-p)^j = p \cdot \frac{1}{1-(1-p)} = \frac{p}{p} = 1 \checkmark$$

**Computing the mean of Geometric via differentiation of generating function:**

$$\mathbb{E}[X] = \sum_{k=1}^{\infty}k(1-p)^{k-1}p = p\sum_{k=1}^{\infty}k(1-p)^{k-1} = p \cdot \frac{d}{dq}\sum_{k=0}^{\infty}q^k\bigg|_{q=1-p}$$
$$= p \cdot \frac{d}{dq}\frac{1}{1-q}\bigg|_{q=1-p} = p \cdot \frac{1}{(1-q)^2}\bigg|_{q=1-p} = p \cdot \frac{1}{p^2} = \frac{1}{p}$$

The expected wait time is $1/p$ — intuitively, if each trial has probability $p$ of success, on average you need $1/p$ trials.

---

## Appendix C: Continuous Distributions — Computing Moments Directly

### C.1 Gaussian Mean and Variance from First Principles

**Mean of $\mathcal{N}(\mu, \sigma^2)$:** By symmetry of the Gaussian PDF about $\mu$:
$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}\, dx$$

Substitute $z = (x-\mu)/\sigma$ so $x = \mu + \sigma z$ and $dx = \sigma\, dz$:
$$= \int_{-\infty}^{\infty} (\mu + \sigma z) \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2}\, dz = \mu \underbrace{\int_{-\infty}^{\infty}\phi(z)\,dz}_{=1} + \sigma\underbrace{\int_{-\infty}^{\infty} z\phi(z)\, dz}_{=0\ (\text{odd integrand})} = \mu$$

**Variance of $\mathcal{N}(\mu, \sigma^2)$:** With the same substitution, $\operatorname{Var}(X) = \mathbb{E}[(X-\mu)^2] = \sigma^2 \mathbb{E}[Z^2]$ where $Z \sim \mathcal{N}(0,1)$.

$$\mathbb{E}[Z^2] = \int_{-\infty}^{\infty} z^2 \cdot \frac{1}{\sqrt{2\pi}} e^{-z^2/2}\, dz$$

Integrate by parts: $u = z$, $dv = z e^{-z^2/2}\, dz$, so $v = -e^{-z^2/2}$:
$$= \frac{1}{\sqrt{2\pi}}\left[-ze^{-z^2/2}\right]_{-\infty}^{\infty} + \frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-z^2/2}\, dz = 0 + 1 = 1$$

Therefore $\operatorname{Var}(X) = \sigma^2 \cdot 1 = \sigma^2$. $\square$

### C.2 Uniform Distribution Variance — Direct Computation

For $X \sim \text{Uniform}(a,b)$:

$$\mathbb{E}[X^2] = \int_a^b x^2 \cdot \frac{1}{b-a}\, dx = \frac{1}{b-a}\cdot\frac{x^3}{3}\bigg|_a^b = \frac{b^3-a^3}{3(b-a)} = \frac{a^2+ab+b^2}{3}$$

using the factorisation $b^3 - a^3 = (b-a)(b^2+ab+a^2)$.

$$\operatorname{Var}(X) = \frac{a^2+ab+b^2}{3} - \frac{(a+b)^2}{4} = \frac{4(a^2+ab+b^2) - 3(a+b)^2}{12} = \frac{(b-a)^2}{12}$$

---

## Appendix D: Key Inequalities and Their Probability Theory Roots

Several important inequalities follow directly from the basic probability theory developed in this section:

### D.1 Markov's Inequality (Preview)

For any non-negative random variable $X$ and $t > 0$:
$$P(X \geq t) \leq \frac{\mathbb{E}[X]}{t}$$

*Proof sketch:* $\mathbb{E}[X] = \mathbb{E}[X \cdot \mathbf{1}_{X \geq t}] + \mathbb{E}[X \cdot \mathbf{1}_{X < t}] \geq \mathbb{E}[X \cdot \mathbf{1}_{X \geq t}] \geq t \cdot P(X \geq t)$.

This connects the expected value (§8.1) to tail probabilities. Full treatment with applications to PAC bounds: [§05-Concentration-Inequalities](../05-Concentration-Inequalities/notes.md).

### D.2 Jensen's Inequality

For a convex function $g$ and random variable $X$:
$$g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]$$

Examples:
- $g(x) = x^2$ (convex): $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$ — the computational variance formula
- $g(x) = e^x$ (convex): $e^{\mathbb{E}[X]} \leq \mathbb{E}[e^X]$ — used in exponential tail bounds
- $g(x) = -\log x$ (convex for $x > 0$): $-\log \mathbb{E}[X] \leq \mathbb{E}[-\log X]$ — used in ELBO derivation for VAEs

### D.3 Cauchy-Schwarz for Expectations

$$|\mathbb{E}[XY]|^2 \leq \mathbb{E}[X^2] \cdot \mathbb{E}[Y^2]$$

This bounds the correlation between random variables and is used to prove the variance of sums and covariance properties in §04.

---

## Appendix E: Probability in High Dimensions — A Preview

All of the distributions introduced in this section are one-dimensional ($X : \Omega \to \mathbb{R}$). In machine learning, we almost always work with high-dimensional data — image pixels, word embeddings, weight tensors. High-dimensional probability has surprising and sometimes counterintuitive properties:

### E.1 The Curse of Dimensionality for Probability

In $d$ dimensions, a unit hypercube $[0,1]^d$ has volume 1. But an inscribed hypersphere of radius 0.5 has volume:

$$V_d = \frac{\pi^{d/2}}{\Gamma(d/2+1)} \cdot 0.5^d$$

For $d = 2$: $V_2 = \pi/4 \approx 0.785$. For $d = 10$: $V_{10} \approx 0.0025$. For $d = 100$: essentially 0.

Almost all the volume of a high-dimensional cube is in its corners, not near its centre. A uniform distribution over the hypercube concentrates most of its mass far from the centre — which means nearest-neighbour methods based on Euclidean distance break down.

### E.2 Gaussian Concentration in High Dimensions

For $\mathbf{z} \sim \mathcal{N}(0, I_d)$ (standard Gaussian in $d$ dimensions):
$$\mathbb{E}[\|\mathbf{z}\|^2] = d, \quad \operatorname{Var}(\|\mathbf{z}\|^2) = 2d$$

By the law of large numbers: $\|\mathbf{z}\|^2/d \to 1$ as $d \to \infty$ (concentrates on the sphere of radius $\sqrt{d}$). This means:
- High-dimensional Gaussian samples are approximately uniformly distributed on the $\sqrt{d}$-sphere
- The dot product $\mathbf{z}_1 \cdot \mathbf{z}_2 \approx 0$ for two independent samples (random vectors are nearly orthogonal)

**For AI:** This explains why random embedding matrices in transformers initialised with $\mathcal{N}(0, 1/d)$ produce approximately orthogonal row vectors in high dimensions — a useful inductive bias for attention.

These high-dimensional considerations are fully developed in §03-Joint-Distributions (multivariate Gaussian) and §05-Concentration-Inequalities (concentration of measure).


---

## Appendix F: Information Theory Preview — Entropy and KL Divergence

This appendix previews two information-theoretic quantities that appear constantly in ML loss functions. The full derivations using the expectation operator belong in §04 (Expectation and Moments) and Chapter 9 (Information Theory).

### F.1 Entropy: Measuring Uncertainty

The **Shannon entropy** of a discrete random variable $X$ with PMF $p$ is:
$$H(X) = -\sum_{x} p(x) \log p(x)$$

Entropy measures the average "surprise" or uncertainty in $X$. Higher entropy = more spread-out distribution = more uncertainty.

**Key values:**
- Bernoulli($p$): $H = -p\log p - (1-p)\log(1-p)$. Maximum $H = \log 2 \approx 0.693$ nats at $p = 0.5$ (fair coin). Zero when $p = 0$ or $p = 1$ (certain outcome).
- Uniform on $n$ values: $H = \log n$ — this is the **maximum entropy** distribution for a discrete variable with $n$ outcomes.
- Deterministic: $H = 0$ — no uncertainty.

For continuous random variables, the **differential entropy** is:
$$h(X) = -\int p(x) \log p(x) \, dx$$

The Gaussian $\mathcal{N}(\mu, \sigma^2)$ has differential entropy $h = \frac{1}{2}\log(2\pi e \sigma^2)$, which is the **maximum entropy** distribution for a continuous variable with fixed variance.

**For AI:** The cross-entropy loss for classification is $H(y, \hat{p}) = -\sum_c y_c \log \hat{p}_c$. When $y$ is a one-hot label, this equals $-\log \hat{p}_{y}$ — the negative log probability the model assigned to the correct class.

### F.2 KL Divergence: Measuring Distribution Distance

The **Kullback-Leibler divergence** from distribution $q$ to distribution $p$ is:
$$D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\left[\log \frac{p(X)}{q(X)}\right]$$

Properties (proved using Jensen's inequality in Appendix D):
1. $D_{\mathrm{KL}}(p \| q) \geq 0$ (non-negativity; follows from $-\log$ being convex)
2. $D_{\mathrm{KL}}(p \| q) = 0 \iff p = q$ (equals zero only when identical)
3. $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$ (asymmetric — not a metric)

**Decomposition:** Cross-entropy $= $ entropy $+$ KL divergence:
$$H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$$

Since $H(p)$ is fixed when $p$ is the true label distribution, minimising cross-entropy loss is equivalent to minimising $D_{\mathrm{KL}}(p \| q)$ — pushing the model's distribution $q$ toward the true distribution $p$.

**For AI:** VAE training minimises the ELBO, which includes a $D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$ regularisation term. RLHF preference modelling computes KL penalties between the fine-tuned and reference policy distributions.

> → _Full treatment: [Chapter 9 — Information Theory](../../09-Information-Theory/README.md)_

### F.3 Negative Log-Likelihood Connects Entropy and Probability

For a dataset $\{x_1, \ldots, x_n\}$ drawn i.i.d. from $p$, the **negative log-likelihood** under model $q_\theta$ is:
$$-\log L(\theta) = -\sum_{i=1}^n \log q_\theta(x_i)$$

By the law of large numbers (§06 preview), as $n \to \infty$:
$$-\frac{1}{n}\log L(\theta) \to \mathbb{E}_p[-\log q_\theta(X)] = H(p, q_\theta)$$

Minimising NLL is therefore equivalent to minimising cross-entropy, which is equivalent to minimising $D_{\mathrm{KL}}(p \| q_\theta)$. **Maximum likelihood estimation is KL minimisation.**

---

## Appendix G: The Exponential Family — A Preview

Almost every distribution in §02 belongs to a unified family called the **exponential family**. Understanding this family's structure explains why Gaussian, Bernoulli, Poisson, and many other distributions share the same algorithmic properties in ML.

### G.1 Exponential Family Form

A distribution belongs to the exponential family if its PDF/PMF can be written as:
$$p(x; \eta) = h(x) \exp\left(\eta^\top T(x) - A(\eta)\right)$$

where:
- $\eta \in \mathbb{R}^k$ — **natural parameters** (the parameterisation used in optimisation)
- $T(x) \in \mathbb{R}^k$ — **sufficient statistics** (all information about $\eta$ in the data is captured by $T(x)$)
- $A(\eta) = \log \int h(x)\exp(\eta^\top T(x)) \, dx$ — **log-partition function** (ensures normalisation)
- $h(x)$ — **base measure** (does not depend on $\eta$)

### G.2 Standard Distributions as Exponential Family Members

| Distribution | Natural param $\eta$ | Sufficient stat $T(x)$ | Log-partition $A(\eta)$ |
|---|---|---|---|
| Bernoulli($p$) | $\log\frac{p}{1-p}$ | $x$ | $\log(1 + e^\eta)$ |
| Gaussian($\mu,\sigma^2$, fixed $\sigma^2$) | $\mu/\sigma^2$ | $x$ | $\eta^2\sigma^2/2$ |
| Poisson($\lambda$) | $\log \lambda$ | $x$ | $e^\eta$ |
| Exponential($\lambda$) | $-\lambda$ | $x$ | $-\log(-\eta)$ |

The Bernoulli natural parameter $\eta = \log(p/(1-p))$ is the **log-odds** or **logit**. The softmax function is the multivariate inverse: $\sigma(\eta) = e^\eta/(1+e^\eta)$ recovers $p$ from $\eta$. This is why logistic regression outputs are Bernoulli exponential family probabilities.

### G.3 Why the Log-Partition Function Matters

The log-partition function $A(\eta)$ is **convex** and its derivatives give the cumulants:
$$\nabla_\eta A(\eta) = \mathbb{E}_{p_\eta}[T(X)] \quad \text{(mean of sufficient statistics)}$$
$$\nabla^2_\eta A(\eta) = \operatorname{Cov}_{p_\eta}[T(X)] \quad \text{(covariance of sufficient statistics)}$$

This means computing moments of exponential family distributions reduces to differentiating $A(\eta)$ — a powerful computational shortcut.

**For AI:** Softmax attention uses the log-partition function. The logsumexp operation $\log \sum_j e^{a_j}$ is the log-partition function of a categorical distribution over query-key matches. The numerical stability trick $\log \sum_j e^{a_j - \max a} + \max a$ exploits properties of $A(\eta)$.

> → _Full treatment: [§02-Common-Distributions](../02-Common-Distributions/notes.md)_

---

## Appendix H: Extended Worked Examples

### H.1 The Monty Hall Problem — Conditional Probability in Action

**Setup:** A car is behind one of three doors. You choose Door 1. The host (who knows where the car is) opens Door 3, revealing a goat. Should you switch to Door 2?

**Solution using conditional probability:**

Let $C_i$ = event car is behind Door $i$, $O_3$ = event host opens Door 3.

Prior: $P(C_1) = P(C_2) = P(C_3) = 1/3$.

Host strategy: if car is at Door 1, host opens Door 2 or 3 with equal probability $1/2$; if car is at Door 2, host must open Door 3 (probability 1); if car is at Door 3, host must open Door 2 (probability 0 for Door 3).

$$P(O_3 | C_1) = 1/2, \quad P(O_3 | C_2) = 1, \quad P(O_3 | C_3) = 0$$

By Bayes' theorem:
$$P(C_1 | O_3) = \frac{P(O_3|C_1)P(C_1)}{P(O_3)} = \frac{(1/2)(1/3)}{1/6 + 1/3 + 0} = \frac{1/6}{1/2} = \frac{1}{3}$$
$$P(C_2 | O_3) = \frac{P(O_3|C_2)P(C_2)}{P(O_3)} = \frac{(1)(1/3)}{1/2} = \frac{2}{3}$$

**Conclusion:** Switching wins with probability $2/3$. Staying wins with probability $1/3$.

**Intuition:** The host's action is not random — they always reveal a goat. This action transfers probability mass from Door 3 to Door 2 (given your initial choice was Door 1). The host's knowledge changes the posterior.

**For AI:** This illustrates why naive probabilistic reasoning fails when the observation process is not independent of the hypothesis. In model evaluation, the selection of which test examples to report affects the posterior over model quality — a key issue in benchmark gaming.

### H.2 The Prosecutor's Fallacy — $P(A|B) \neq P(B|A)$

A DNA test for a rare genetic marker has false positive rate $0.1\%$ and false negative rate $0\%$. The marker is present in $0.01\%$ of the population. A defendant tests positive. A prosecutor claims: "The probability of a false positive is only $0.1\%$, so there is a $99.9\%$ chance the defendant is guilty."

**What's wrong:** The prosecutor is claiming $P(\text{innocent} | \text{positive}) = 0.001$, but they are calculating $P(\text{positive} | \text{innocent}) = 0.001$.

By Bayes' theorem:
- $P(\text{guilty}) = 0.0001$ (base rate)
- $P(\text{positive} | \text{guilty}) = 1$ (no false negatives)
- $P(\text{positive} | \text{innocent}) = 0.001$ (false positive rate)

$$P(\text{guilty} | \text{positive}) = \frac{1 \cdot 0.0001}{1 \cdot 0.0001 + 0.001 \cdot 0.9999} \approx \frac{0.0001}{0.001} = 9.1\%$$

The actual probability of guilt given the positive test is only about $9\%$, not $99.9\%$.

**For AI:** This fallacy appears in anomaly detection. A model with $99.9\%$ accuracy on a test set may have only $50\%$ precision on a rare-class detection task if the base rate of the rare class is $0.1\%$.

### H.3 Coupon Collector Problem — Expected Waiting Times

There are $n$ distinct coupons. Each cereal box contains one coupon uniformly at random. How many boxes must you buy (in expectation) to collect all $n$ coupons?

Let $T_k$ = number of boxes needed to get the $k$-th new coupon (given you already have $k-1$ distinct coupons). At that point, the probability of drawing a new coupon is $(n-k+1)/n$, so $T_k \sim \text{Geometric}\left(\frac{n-k+1}{n}\right)$.

$$\mathbb{E}[T_k] = \frac{n}{n-k+1}$$

Total boxes: $T = T_1 + T_2 + \cdots + T_n$, so:
$$\mathbb{E}[T] = \sum_{k=1}^{n} \frac{n}{n-k+1} = n \sum_{j=1}^{n} \frac{1}{j} = n H_n$$

where $H_n = 1 + 1/2 + 1/3 + \cdots + 1/n \approx \ln n + 0.577$ (harmonic number).

For $n = 365$ (days of the year): $\mathbb{E}[T] = 365 \cdot H_{365} \approx 365 \cdot 6.46 \approx 2365$ attempts needed to see all birthdays.

**For AI:** The coupon collector framework models how many gradient steps are needed for a stochastic optimizer to see every training example at least once. For $n$ training examples, about $n \ln n$ steps are needed in expectation — which is why practitioners set epoch counts based on logarithmic growth.

---

## Appendix I: Characteristic Functions and Moment Generating Functions — Preview

### I.1 The Moment Generating Function (MGF)

The **moment generating function** of a random variable $X$ is:
$$M_X(t) = \mathbb{E}[e^{tX}] = \int e^{tx} p(x) \, dx \quad \text{(when it exists)}$$

Why "moment generating"? Differentiating and evaluating at $t=0$:
$$M_X^{(k)}(0) = \mathbb{E}[X^k]$$

The $k$-th derivative of the MGF at zero is the $k$-th raw moment of $X$.

**For Gaussian** $X \sim \mathcal{N}(\mu, \sigma^2)$:
$$M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

This elegant form is why the Gaussian is easy to work with analytically: all moments can be extracted by differentiating a simple exponential.

**Key property (moment generating functions and sums):** If $X \perp Y$:
$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$$

This product rule for MGFs is the probabilistic analogue of the convolution theorem, and it is the key tool for proving that sums of independent Gaussians are Gaussian and that sums of independent Poissons are Poisson.

### I.2 The Characteristic Function

The **characteristic function** of $X$ is $\varphi_X(t) = \mathbb{E}[e^{itX}]$ (Fourier transform of the density). Unlike the MGF, the characteristic function always exists and uniquely determines the distribution. It is the tool used in the proof of the Central Limit Theorem.

> → _Full treatment: [§04-Expectation-and-Moments](../04-Expectation-and-Moments/notes.md)_


---

## Appendix J: Simulation and Monte Carlo Methods — A Preview

A recurring theme throughout this chapter is that computing exact probabilities analytically is often hard or impossible. **Monte Carlo simulation** is the practical alternative: approximate $\mathbb{E}[g(X)]$ by averaging over many samples.

### J.1 The Monte Carlo Principle

If $X_1, X_2, \ldots, X_n$ are i.i.d. samples from $p(x)$, then by the Law of Large Numbers (§06):
$$\frac{1}{n}\sum_{i=1}^n g(X_i) \xrightarrow{n \to \infty} \mathbb{E}_p[g(X)]$$

The Monte Carlo estimator $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n g(X_i)$ is **unbiased** ($\mathbb{E}[\hat{\mu}_n] = \mathbb{E}[g(X)]$) and has variance $\operatorname{Var}(g(X))/n$, so the standard error is $\sigma/\sqrt{n}$ regardless of dimension.

**Why this matters:** Unlike numerical integration (which requires exponentially many grid points in high dimensions), Monte Carlo has convergence rate $O(1/\sqrt{n})$ independent of dimension. This makes it the tool of choice for high-dimensional integrals — exactly the setting of modern ML.

### J.2 Estimating Probabilities

$P(A) = \mathbb{E}[\mathbf{1}_A(X)]$, so we can estimate any probability:
$$\hat{P}(A) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[X_i \in A]$$

**Example:** Estimate $P(Z > 1.96)$ for $Z \sim \mathcal{N}(0,1)$ by sampling:
1. Draw $Z_1, \ldots, Z_n \sim \mathcal{N}(0,1)$
2. Count the fraction where $Z_i > 1.96$
3. Answer converges to $0.025$ (the true tail probability)

### J.3 Sampling from Basic Distributions

**Inverse CDF method (Universality of the Uniform):** If $U \sim \text{Uniform}(0,1)$ and $F$ is a CDF, then $F^{-1}(U)$ has distribution $F$. This is because:
$$P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)$$

**Consequences:**
- Sample Exponential($\lambda$): $X = -\frac{1}{\lambda}\log(U)$ where $U \sim \text{Uniform}(0,1)$
- Sample Geometric($p$): $X = \lceil \log(U) / \log(1-p) \rceil$
- Sample Bernoulli($p$): $X = \mathbf{1}[U \leq p]$

The Box-Muller transform generates Gaussian samples from uniform:
$$Z_1 = \sqrt{-2\log U_1}\cos(2\pi U_2), \quad Z_2 = \sqrt{-2\log U_1}\sin(2\pi U_2)$$

where $U_1, U_2 \sim \text{Uniform}(0,1)$ i.i.d., and $Z_1, Z_2 \sim \mathcal{N}(0,1)$ i.i.d.

### J.4 Rejection Sampling

When the inverse CDF has no closed form, use **rejection sampling**: to sample from $p(x)$, find a proposal $q(x)$ and constant $M$ such that $p(x) \leq M q(x)$ everywhere.

**Algorithm:**
1. Sample $Y \sim q$
2. Accept $Y$ with probability $p(Y)/(Mq(Y))$; otherwise reject and repeat

The accepted samples have distribution $p$. The acceptance rate is $1/M$, so choose $M$ as small as possible.

**For AI:** Rejection sampling underlies early language model decoding strategies. Modern methods like nucleus sampling (top-$p$) and temperature sampling directly manipulate the token probability distribution — understanding these requires the CDF and quantile framework from §7.

### J.5 Monte Carlo Integration of Normalisation Constants

Many probabilistic models have densities of the form $p(x) = \tilde{p}(x)/Z$ where computing the normalisation constant $Z = \int \tilde{p}(x) \, dx$ is intractable. Examples:
- Bayesian posterior: $Z = \int \text{likelihood}(x) \cdot \text{prior}(x) \, dx$
- Boltzmann distribution: $Z = \sum_x e^{-E(x)/T}$ over exponentially many states
- Normalising flow: $Z = 1$ by construction but requires a change-of-variables Jacobian

Monte Carlo methods (particularly MCMC, developed in §07) are the primary tools for inference in these models.

> → _Full treatment: [§07-Markov-Chains](../07-Markov-Chains/notes.md) (MCMC); [§06-Stochastic-Processes](../06-Stochastic-Processes/notes.md) (LLN and CLT)_


---

## Appendix K: Probability Distributions in PyTorch and NumPy

Modern ML frameworks have built-in probability distribution objects. Understanding the mathematical definitions in this section makes framework APIs immediately transparent.

### K.1 NumPy / SciPy Distributions

```python
import numpy as np
from scipy import stats

# Bernoulli(p=0.3)
rv = stats.bernoulli(p=0.3)
rv.pmf(1)          # P(X=1) = 0.3
rv.cdf(0)          # P(X≤0) = 0.7
rv.rvs(size=100)   # 100 random samples

# Gaussian N(mu=2, sigma=1.5)
rv = stats.norm(loc=2, scale=1.5)
rv.pdf(2.0)        # f(2.0) = 1/(1.5*sqrt(2π)) ≈ 0.266
rv.cdf(2.0)        # Φ(0) = 0.5
rv.ppf(0.975)      # inverse CDF (quantile): z such that P(X≤z)=0.975

# Uniform(a=0, b=1)
rv = stats.uniform(loc=0, scale=1)
```

The `loc` and `scale` parameters implement the location-scale transform: $X = \text{loc} + \text{scale} \cdot Z$ where $Z$ is the standard form.

### K.2 PyTorch `torch.distributions`

```python
import torch
from torch.distributions import Bernoulli, Normal, Uniform, Categorical

# Bernoulli — for binary classification output
dist = Bernoulli(probs=torch.tensor(0.3))
dist.log_prob(torch.tensor(1.0))  # log P(X=1) = log(0.3) ≈ -1.204
dist.sample()                      # sample {0,1}
dist.entropy()                     # H(X) ≈ 0.611 nats

# Normal — for regression, VAE latent, etc.
dist = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))
dist.log_prob(torch.tensor(1.96))  # log f(1.96) ≈ -2.84
dist.cdf(torch.tensor(1.96))       # ≈ 0.975

# Categorical — for language model outputs
logits = torch.tensor([2.0, 1.0, 0.5])  # unnormalised log probabilities
dist = Categorical(logits=logits)
dist.probs                              # softmax probabilities
dist.log_prob(torch.tensor(0))          # log P(X=0)
dist.entropy()                          # entropy of the distribution
```

### K.3 Reparameterisation — The Cornerstone of VAE Training

The **reparameterisation trick** lets us differentiate through a sampling operation:

Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ (which has no gradient w.r.t. $\mu, \sigma$), write:
$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

Now $\partial z / \partial \mu = 1$ and $\partial z / \partial \sigma = \epsilon$ — gradients flow through.

```python
# Without reparameterisation (WRONG for training)
z = Normal(mu, sigma).sample()  # no gradient

# With reparameterisation (CORRECT)
eps = torch.randn_like(sigma)   # ε ~ N(0,1)
z = mu + sigma * eps             # z ~ N(mu, sigma^2), gradients flow
```

This technique requires understanding that $\mathcal{N}(\mu, \sigma^2)$ and $\mu + \sigma \mathcal{N}(0,1)$ have the same distribution — a direct consequence of the location-scale property developed in §7.3.

**For AI:** VAE encoders output $(\mu, \log\sigma^2)$ and use this trick to sample the latent code $z$ while maintaining a differentiable path back to the encoder parameters. The same idea extends to normalising flows and diffusion model sampling.

---

## Appendix L: Measure Theory — The Foundations Beneath the Formalism

This section develops probability from elementary axioms without requiring measure theory. But for completeness, this appendix sketches where the measure-theoretic foundations live and when you need them.

### L.1 Why Elementary Probability Is Insufficient

The elementary theory (as developed in §2) works when the sample space $\Omega$ is discrete or when we can write explicit PDF formulas. It breaks for:

1. **Mixed distributions**: a random variable that is sometimes discrete (e.g., exactly zero) and sometimes continuous (e.g., positive real) — arises in zero-inflated models and ReLU activations.

2. **Limits**: taking limits of sequences of random variables requires knowing what "$X_n \to X$" means precisely — there are multiple notions of convergence ($L^2$, almost sure, in probability, in distribution) that are not cleanly expressible without measure theory.

3. **Conditional expectation on events of probability zero**: $\mathbb{E}[Y | X = x]$ when $P(X = x) = 0$ (any continuous $X$) requires the Radon-Nikodym theorem.

4. **Change-of-measure / importance sampling**: $\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[\frac{p(X)}{q(X)}f(X)\right]$ when $p$ and $q$ are continuous requires the Radon-Nikodym derivative.

### L.2 The Measure-Theoretic Setup

A **probability space** is a triple $(\Omega, \mathcal{F}, P)$ where:
- $\Omega$ — sample space (any set)
- $\mathcal{F}$ — **sigma-algebra**: a collection of subsets of $\Omega$ closed under complement and countable union (the measurable events)
- $P: \mathcal{F} \to [0,1]$ — probability measure satisfying Kolmogorov's axioms

A **random variable** is a **measurable function** $X: \Omega \to \mathbb{R}$, meaning $\{X \leq x\} \in \mathcal{F}$ for all $x$ (so that $P(X \leq x)$ is well-defined).

The **distribution** of $X$ is the induced measure $\mu_X(B) = P(X^{-1}(B))$ for Borel sets $B \subseteq \mathbb{R}$.

The **Lebesgue integral** replaces the Riemann integral and is defined for any measurable function — this is what makes $\mathbb{E}[X] = \int X \, dP$ well-defined even for random variables with no PDF.

### L.3 The Lebesgue-Stieltjes Integral Unifies PDF and PMF

For any random variable with CDF $F$, the expectation can be written as a Stieltjes integral:
$$\mathbb{E}[g(X)] = \int g(x) \, dF(x)$$

This notation works for both discrete and continuous cases:
- Continuous: $dF(x) = f(x)\,dx$ (use Riemann integral)
- Discrete: $dF(x) = \sum_k p_k \,\delta(x - x_k)$ (use a sum)
- Mixed: combination of both

**For AI:** Importance sampling in policy gradient methods (PPO, TRPO) uses the Radon-Nikodym derivative implicitly when bounding the ratio $\pi_\theta(a|s)/\pi_{\theta_{\text{old}}}(a|s)$ — this is the change-of-measure formula at work.

> The full measure-theoretic treatment is beyond the scope of this curriculum. The reader interested in rigorous foundations should consult Billingsley's *Probability and Measure* or Durrett's *Probability: Theory and Examples*.


---

## Appendix M: Classic Probability Paradoxes and Their Resolutions

### M.1 Bertrand's Paradox — Why "Uniform at Random" Needs Specification

**Problem:** A chord is drawn "at random" in a unit circle. What is the probability the chord is longer than the side of the inscribed equilateral triangle (length $\sqrt{3}$)?

The answer depends on how you sample "at random":

**Method 1 — Random endpoints:** Choose two points uniformly on the circle's circumference. Probability = $1/3$.

**Method 2 — Random midpoint:** Choose a point uniformly inside the circle as the chord's midpoint. Probability = $1/4$.

**Method 3 — Random radius and distance:** Choose a radius uniformly; choose the midpoint's distance from centre uniformly on $[0,1]$. Probability = $1/2$.

All three methods are geometrically natural, yet give different answers.

**Resolution:** The phrase "uniform at random" is not well-defined without specifying the **sample space** and the **probability measure** on it. The three methods define different probability spaces $(\Omega, \mathcal{F}, P)$ — they are asking different questions.

**Lesson for ML:** When we say "sample a random minibatch" or "sample a random initialisation," we must specify the distribution explicitly. The implicit assumption (uniform distribution) can lead to different algorithmic properties depending on whether we sample with or without replacement, sample tokens or sequences, etc.

### M.2 The Two-Envelope Paradox

Two envelopes each contain money; one has twice the amount of the other. You pick one, see $X$ inside, and are offered the chance to switch. You reason: the other envelope contains either $X/2$ (probability $1/2$) or $2X$ (probability $1/2$). Expected value of switching: $\frac{1}{2}(X/2) + \frac{1}{2}(2X) = \frac{5X}{4} > X$. So you should always switch — but the same reasoning applies after switching back!

**Resolution:** The calculation is flawed because the event "the other envelope has $2X$" and "the other envelope has $X/2$" cannot both be assigned probability $1/2$ simultaneously for a fixed $X$. The reasoning treats $X$ as fixed and random simultaneously. A rigorous Bayesian analysis shows no advantage to switching when the prior over the smaller amount is proper (integrable).

**Lesson for ML:** The paradox arises from mixing conditional and unconditional reasoning. In ML, this appears in off-policy evaluation (evaluating policy $\pi_\text{new}$ using data from $\pi_\text{old}$) — naively computing expected reward leads to analogous double-counting errors.

### M.3 The St. Petersburg Paradox — When Expected Value Is Infinite

A game: flip a fair coin repeatedly. If the first head appears on flip $k$, you win $2^k$ dollars. Expected value:
$$\mathbb{E}[\text{winnings}] = \sum_{k=1}^\infty 2^k \cdot \frac{1}{2^k} = \sum_{k=1}^\infty 1 = \infty$$

Yet no rational person would pay more than ~$25 to play.

**Resolution:** Human decision-making is based on **utility**, not raw monetary value. If utility is $u(w) = \log(w)$ (a common assumption in economics), the expected utility is finite. This gives **expected utility theory** — the rational framework for decisions under uncertainty.

**For AI:** Reinforcement learning explicitly maximises expected return (reward), which can lead to high-variance, unstable policies if reward distributions have heavy tails. Techniques like reward normalisation, reward clipping (PPO), and distributional RL (C51, QR-DQN) are direct engineering responses to the St. Petersburg problem.

---

## Appendix N: Summary of Key Inequalities and Bounds

This appendix collects all the inequalities introduced in this section and previewed from later sections, as a reference card.

### Probability Bounds

| Inequality | Statement | When to Use |
|---|---|---|
| **Boole (union bound)** | $P\!\left(\bigcup_i A_i\right) \leq \sum_i P(A_i)$ | When events may overlap; gives upper bound |
| **Inclusion-exclusion** | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Exact for two events |
| **Bonferroni** | $P\!\left(\bigcap_i A_i^c\right) \geq 1 - \sum_i P(A_i)$ | Lower bound on "all good" probability |
| **Fréchet** | $\max(0, P(A)+P(B)-1) \leq P(A \cap B) \leq \min(P(A),P(B))$ | Bounds on intersection without independence |

### Moment-Based Bounds (from §05)

| Inequality | Statement | Uses |
|---|---|---|
| **Markov** | $P(X \geq t) \leq \mathbb{E}[X]/t$ for $X \geq 0$ | Only needs $\mathbb{E}[X]$ |
| **Chebyshev** | $P(\|X-\mu\| \geq t) \leq \sigma^2/t^2$ | Needs $\mathbb{E}[X]$ and $\operatorname{Var}(X)$ |
| **Jensen** | $g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]$ for convex $g$ | Log-sum-exp, ELBO, entropy |
| **Cauchy-Schwarz** | $|\mathbb{E}[XY]|^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2]$ | Bounds correlation |

### Concentration Bounds (from §05)

| Inequality | Statement | When to Use |
|---|---|---|
| **Hoeffding** | $P(\bar{X}-\mu \geq t) \leq e^{-2n^2t^2/\sum(b_i-a_i)^2}$ for bounded $X_i \in [a_i,b_i]$ | i.i.d. bounded variables |
| **Chernoff** | $P(X \geq (1+\delta)\mu) \leq e^{-\mu\delta^2/3}$ for Poisson binomial | Sums of independent Bernoullis |
| **McDiarmid** | If $f$ changes by at most $c_i$ in coordinate $i$: $P(f - \mathbb{E}[f] \geq t) \leq e^{-2t^2/\sum c_i^2}$ | General functions of independent variables |

**For AI:** Hoeffding's inequality directly gives PAC learning generalisation bounds: with $n$ i.i.d. training examples and loss values in $[0,1]$, the gap between empirical and expected loss satisfies $P(L_{\text{test}} - L_{\text{train}} \geq \varepsilon) \leq 2e^{-2n\varepsilon^2}$ for any fixed hypothesis.


---

## Appendix O: Sigma-Algebras and the Filtration Framework

This appendix provides a slightly more formal treatment of sigma-algebras for readers who want mathematical precision, without requiring full measure theory.

### O.1 Why Sigma-Algebras Appear

The Kolmogorov axioms in §2 assume we know which subsets of $\Omega$ count as "events." For finite or countable $\Omega$, every subset is an event. For continuous $\Omega = \mathbb{R}$, we cannot assign probabilities to all subsets (Vitali sets are non-measurable), so we restrict to a collection $\mathcal{F}$ closed under the operations we need.

### O.2 The Borel Sigma-Algebra

The **Borel sigma-algebra** $\mathcal{B}(\mathbb{R})$ is the smallest sigma-algebra containing all open intervals $(a, b)$. It includes:
- All open and closed sets
- All countable unions and intersections of intervals
- All CDFs are measurable with respect to $\mathcal{B}(\mathbb{R})$

For practical purposes, every set you will encounter in ML is Borel-measurable.

### O.3 Information and Filtrations

In sequential problems (reinforcement learning, time series, stochastic processes), we need to track how much information is available at each time step. A **filtration** is an increasing sequence of sigma-algebras:
$$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \mathcal{F}_2 \subseteq \cdots$$

where $\mathcal{F}_t$ represents "all information available by time $t$."

A random variable $X_t$ is **$\mathcal{F}_t$-measurable** if its value is determined by the information in $\mathcal{F}_t$ — that is, you can compute $X_t$ from the history up to time $t$.

**Example:** In a sequential decision problem:
- $\mathcal{F}_0 = \{\emptyset, \Omega\}$ — no information (only trivial events)
- $\mathcal{F}_1 = \sigma(X_1)$ — sigma-algebra generated by the first observation
- $\mathcal{F}_t = \sigma(X_1, \ldots, X_t)$ — sigma-algebra generated by observations up to time $t$

The **Markov property** (§07) states: $X_{t+1}$ depends on $\mathcal{F}_t$ only through $X_t$ (not the full history).

**For AI:** The attention mechanism's causal mask in GPT-style models enforces a filtration: token $t$ can only attend to tokens at positions $\leq t$. The mask ensures $\mathcal{F}_t$-measurability of each token prediction, making the autoregressive language model a valid probabilistic model of sequences.

### O.4 The Generated Sigma-Algebra

For any random variable $X$, the **generated sigma-algebra** is:
$$\sigma(X) = \{X^{-1}(B) : B \in \mathcal{B}(\mathbb{R})\} = \{\{\omega : X(\omega) \in B\} : B \in \mathcal{B}(\mathbb{R})\}$$

This is the smallest sigma-algebra that makes $X$ measurable — it contains exactly the information encoded in $X$.

For a discrete random variable: $\sigma(X) = \sigma(\{X = x_1\}, \{X = x_2\}, \ldots)$ — the partition of $\Omega$ induced by the values of $X$.

**Key insight:** Conditional expectation $\mathbb{E}[Y | X]$ is really $\mathbb{E}[Y | \sigma(X)]$ — the expectation of $Y$ given all the information encoded in $X$. This formulation (using sigma-algebras) is what makes conditional expectation rigorously defined even when $P(X = x) = 0$.

---

## Appendix P: Probability Generating Functions

### P.1 Definition

For a non-negative integer-valued random variable $X$ (taking values $0, 1, 2, \ldots$), the **probability generating function (PGF)** is:
$$G_X(z) = \mathbb{E}[z^X] = \sum_{k=0}^\infty p_k z^k$$

where $p_k = P(X = k)$.

The PGF converges for $|z| \leq 1$ and stores all probabilities as coefficients of a power series.

### P.2 Recovering Probabilities and Moments

$$p_k = \frac{G_X^{(k)}(0)}{k!} \quad \text{($k$-th derivative at zero)}$$

$$\mathbb{E}[X] = G_X'(1), \quad \mathbb{E}[X(X-1)] = G_X''(1), \quad \operatorname{Var}(X) = G_X''(1) + G_X'(1) - [G_X'(1)]^2$$

### P.3 PGFs of Common Distributions

| Distribution | PGF $G(z)$ |
|---|---|
| Bernoulli($p$) | $1-p+pz$ |
| Binomial($n,p$) | $(1-p+pz)^n$ |
| Poisson($\lambda$) | $e^{\lambda(z-1)}$ |
| Geometric($p$) | $pz/(1-(1-p)z)$ |

The Binomial PGF $(1-p+pz)^n$ is the $n$-fold product of Bernoulli PGFs — directly encoding that a Binomial is a sum of $n$ i.i.d. Bernoullis (by the product property for independent variables).

**Product property:** If $X \perp Y$: $G_{X+Y}(z) = G_X(z) \cdot G_Y(z)$.

This is the PGF analogue of the convolution theorem, and it gives a clean proof that the sum of independent Poisson($\lambda_1$) and Poisson($\lambda_2$) is Poisson($\lambda_1 + \lambda_2$):
$$G_{\text{Pois}(\lambda_1)}(z) \cdot G_{\text{Pois}(\lambda_2)}(z) = e^{\lambda_1(z-1)} \cdot e^{\lambda_2(z-1)} = e^{(\lambda_1+\lambda_2)(z-1)} = G_{\text{Pois}(\lambda_1+\lambda_2)}(z)$$

**For AI:** The token generation process in language models can be viewed as repeatedly sampling from a categorical PGF. The PGF of the output token count under various sampling strategies (greedy, top-$k$, nucleus) encodes the statistical properties of generated text length distributions.


---

## Appendix Q: Historical Development of Probability Theory

Understanding how probability theory evolved helps locate the axioms, distributions, and theorems in their proper intellectual context.

### Q.1 Pre-Axiomatic Period (1654–1900)

**Pascal and Fermat (1654):** The famous exchange of letters about the "Problem of Points" (how to divide stakes in an unfinished game) is often cited as the birth of probability theory. They developed the idea of expected value and laid groundwork for combinatorics.

**Bernoulli (1713):** Jakob Bernoulli's *Ars Conjectandi* stated the first version of the Law of Large Numbers: relative frequencies of outcomes converge to true probabilities. This transformed probability from gambling mathematics to a science of inference.

**Bayes (1763):** Thomas Bayes' posthumous *Essay* gave the first treatment of inverse probability — inferring parameters from observations. Richard Price edited and published it; Laplace independently developed the same ideas more completely.

**Laplace (1812):** *Théorie analytique des probabilités* synthesised the field, introducing characteristic functions, the Central Limit Theorem, the method of moments, and the formal definition of probability as a fraction of equally likely cases.

**Gauss (1809, 1823):** Derived the normal distribution as the error distribution minimising the sum of squared residuals. Proved that the sample mean is the best linear unbiased estimator — establishing connections between probability and statistics.

**Chebyshev, Markov, Lyapunov (1867–1901):** Proved increasingly general versions of the CLT. Markov introduced his eponymous chains; Lyapunov's proof of the CLT via characteristic functions is essentially the modern proof.

### Q.2 Axiomatic Period (1900–1950)

**Borel (1909):** Proved the strong law of large numbers for the Bernoulli case, introducing Borel sets and measure-theoretic methods.

**Kolmogorov (1933):** *Grundbegriffe der Wahrscheinlichkeitsrechnung* placed probability theory on rigorous measure-theoretic foundations. The three axioms in §2 are Kolmogorov's. This work resolved foundational debates and enabled the rigorous development of stochastic processes.

**Wiener (1923):** Constructed Brownian motion rigorously as a probability measure on function spaces — the first infinite-dimensional probability space.

**Doob (1940s):** Developed martingale theory, providing rigorous tools for studying random processes with conditional structure.

### Q.3 Modern Period (1950–present)

**Shannon (1948):** *A Mathematical Theory of Communication* introduced entropy, mutual information, and channel capacity — connecting probability to communication. Cross-entropy loss in modern ML is a direct descendant.

**Robbins-Monro (1951):** Stochastic approximation — the theoretical foundation for stochastic gradient descent. Proved that gradient estimates from mini-batches converge to the true gradient.

**Dempster-Laird-Rubin (1977):** The EM algorithm — an expectation-maximisation procedure for MLE with latent variables. Built on conditional expectation and Jensen's inequality.

**Pearl (1988):** Bayesian networks and causal inference — graphical models encoding conditional independence structure. Direct descendant of conditional probability in §4 and §03 of this chapter.

**Goodfellow et al. (2014):** Generative Adversarial Networks — training a generator to match a target distribution by playing a minimax game. The divergence between generated and true distributions is what is being minimised (implicitly or explicitly).

**Sohl-Dickstein et al. (2015), Ho et al. (2020):** Diffusion models — add Gaussian noise progressively (forward process), then learn to denoise (reverse process). Grounded in stochastic processes and Gaussian distributions.

### Q.4 The Connections to Modern LLMs

Every component of a modern language model has a probabilistic interpretation rooted in the theory developed in this section:

| LLM Component | Probability Foundation | Section |
|---|---|---|
| Token prediction | Categorical distribution over vocabulary | §6 (discrete RV) |
| Temperature scaling | Scale parameter of Gumbel distribution | §7 (continuous RV) |
| Dropout | Bernoulli mask on activations | §6 (Bernoulli) |
| Weight initialisation | Normal distribution, variance scaling | §7 (Gaussian) |
| KV cache attention | Conditional probability over past keys | §4 (conditioning) |
| RLHF reward model | Bradley-Terry preference model | §3 (conditional probability) |
| Beam search | Greedy argmax over joint probability | §6 (joint events) |
| Perplexity metric | Exponential of cross-entropy | §8 (expectation) |
| Calibration | Agreement between predicted probs and frequencies | §2 (frequentist) |
| Chain-of-thought | Latent variable model over reasoning traces | §4 (conditional independence) |


---

## Appendix R: Notation Reference for Chapter 6

This reference card collects all notation used in this chapter. Consistent notation prevents ambiguity when working across sections.

### R.1 Sets and Events

| Symbol | Meaning |
|---|---|
| $\Omega$ | Sample space |
| $\omega \in \Omega$ | Elementary outcome |
| $A, B, C$ | Events (subsets of $\Omega$) |
| $A^c$ | Complement of $A$ |
| $A \cap B$ | Intersection (both $A$ and $B$) |
| $A \cup B$ | Union (either $A$ or $B$) |
| $A \setminus B$ | Set difference ($A$ but not $B$) |
| $A \triangle B$ | Symmetric difference |
| $\mathbf{1}_A$ or $\mathbf{1}[A]$ | Indicator function of event $A$ |
| $\mathcal{F}$ | Sigma-algebra of events |

### R.2 Probability

| Symbol | Meaning |
|---|---|
| $P(A)$ | Probability of event $A$ |
| $P(A \| B)$ | Conditional probability of $A$ given $B$ |
| $P(A, B)$ | Joint probability $P(A \cap B)$ |
| $A \perp B$ | Events $A$ and $B$ are independent |

### R.3 Random Variables

| Symbol | Meaning |
|---|---|
| $X, Y, Z$ | Random variables |
| $x, y, z$ | Values (realisations) of random variables |
| $F_X(x)$ | CDF of $X$: $P(X \leq x)$ |
| $f_X(x)$ | PDF of $X$ (continuous) |
| $p_X(x)$ | PMF of $X$ (discrete) |
| $\text{supp}(X)$ | Support: $\{x : p_X(x) > 0\}$ or $\{x : f_X(x) > 0\}$ |
| $X \sim p$ | $X$ has distribution $p$ |
| $X \perp Y$ | $X$ and $Y$ are independent random variables |
| $X \perp Y \| Z$ | $X$ and $Y$ are conditionally independent given $Z$ |

### R.4 Named Distributions

| Notation | Distribution |
|---|---|
| $X \sim \text{Bernoulli}(p)$ | Bernoulli with success probability $p$ |
| $X \sim \text{Binomial}(n, p)$ | Binomial: $n$ trials, success probability $p$ |
| $X \sim \text{Geometric}(p)$ | Geometric: trials until first success |
| $X \sim \text{Poisson}(\lambda)$ | Poisson with rate $\lambda$ |
| $X \sim \text{Uniform}(a, b)$ | Uniform on interval $[a,b]$ |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$ and variance $\sigma^2$ |
| $X \sim \text{Exponential}(\lambda)$ | Exponential with rate $\lambda$ |
| $Z \sim \mathcal{N}(0,1)$ | Standard normal |
| $\Phi(z)$ | CDF of the standard normal |

### R.5 Expectation and Moments

| Symbol | Meaning |
|---|---|
| $\mathbb{E}[X]$ | Expected value of $X$ |
| $\mathbb{E}[X \| Y]$ | Conditional expectation of $X$ given $Y$ |
| $\mu_X$ | Mean of $X$ ($= \mathbb{E}[X]$) |
| $\sigma^2_X$ or $\text{Var}(X)$ | Variance of $X$ |
| $\sigma_X$ | Standard deviation of $X$ |
| $\text{Cov}(X,Y)$ | Covariance of $X$ and $Y$ |
| $\rho_{XY}$ | Pearson correlation |
| $M_X(t)$ | Moment generating function |
| $G_X(z)$ | Probability generating function |
| $\varphi_X(t)$ | Characteristic function |

### R.6 Information Theory

| Symbol | Meaning |
|---|---|
| $H(X)$ | Shannon entropy of $X$ |
| $H(X, Y)$ | Joint entropy |
| $H(X \| Y)$ | Conditional entropy |
| $I(X; Y)$ | Mutual information |
| $D_{\mathrm{KL}}(p \| q)$ | KL divergence from $q$ to $p$ |
| $H(p, q)$ | Cross-entropy of $q$ under $p$ |

---

*End of Appendices. Return to [Table of Contents](#table-of-contents).*

