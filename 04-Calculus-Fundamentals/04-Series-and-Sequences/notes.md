[← Back to Calculus Fundamentals](../README.md) | [Next Chapter: Multivariate Calculus →](../../05-Multivariate-Calculus/README.md)

---

# Series and Sequences

> _"The series $1 + \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \cdots = 2$ may seem paradoxical — an infinite number of terms summing to a finite value. But this is exactly the miracle that makes calculus possible."_
> — Augustin-Louis Cauchy, on the foundations of analysis

## Overview

Sequences and series are the mathematics of infinity made tractable. A sequence is an ordered list without end; a series is the sum of such a list. The central question — *does an infinite process converge to something finite?* — is one of the deepest and most practically consequential questions in mathematics.

For machine learning, this is not abstract. Every gradient descent trajectory is a sequence of parameter vectors. Every optimizer update rule is derived via Taylor series truncation. Every attention mechanism approximation exploits the Taylor expansion of the exponential function. Every positional encoding scheme (RoPE, ALiBi, sinusoidal) draws on Fourier series theory. The Laplace approximation — used in Bayesian deep learning — is a quadratic Taylor expansion of the log-posterior. Understanding why these approximations work, when they fail, and how accurate they are requires the theory in this section.

This section builds from sequences through convergence tests through power series to Taylor series, culminating in concrete ML applications with rigorous error analysis.

## Prerequisites

- **Limits and continuity** — [§04/01](../01-Limits-and-Continuity/notes.md): ε-δ definition, limit laws, continuity
- **Derivatives** — [§04/02](../02-Derivatives-and-Differentiation/notes.md): differentiation rules, higher-order derivatives
- **Integration** — [§04/03](../03-Integration/notes.md): definite integrals, improper integrals, p-integral test

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive: convergence visualization, partial sums, Taylor approximation error, ML applications |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from geometric series to linear attention approximations |

## Learning Objectives

After completing this section, you will be able to:

1. **Define** a sequence formally and determine convergence using the ε-N definition
2. **State and prove** the Monotone Convergence Theorem for sequences
3. **Determine convergence** of infinite series using all standard tests
4. **Distinguish** absolute from conditional convergence and explain the rearrangement theorem
5. **Compute** the radius and interval of convergence of a power series
6. **Differentiate and integrate** power series term-by-term within the radius of convergence
7. **Derive** Taylor series for $e^x$, $\sin x$, $\cos x$, $\ln(1+x)$, and $(1+x)^\alpha$
8. **Bound** approximation error using the Lagrange remainder
9. **Apply** series theory to ML: softmax temperature, linear attention, GELU, Adam, Laplace
10. **Recognise** Fourier series as orthogonal function expansion and connect to positional encodings

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Sequences and Series Represent](#11-what-sequences-and-series-represent)
  - [1.2 Historical Context](#12-historical-context)
  - [1.3 Why Series Are Central to AI](#13-why-series-are-central-to-ai)
- [2. Sequences](#2-sequences)
  - [2.1 Formal Definition](#21-formal-definition)
  - [2.2 Convergence: ε-N Definition](#22-convergence-ε-n-definition)
  - [2.3 Bounded and Monotone Sequences](#23-bounded-and-monotone-sequences)
  - [2.4 Subsequences and Bolzano-Weierstrass](#24-subsequences-and-bolzano-weierstrass)
  - [2.5 Important Sequences in ML](#25-important-sequences-in-ml)
- [3. Infinite Series](#3-infinite-series)
  - [3.1 Partial Sums and Convergence](#31-partial-sums-and-convergence)
  - [3.2 Geometric Series](#32-geometric-series)
  - [3.3 The Harmonic Series Diverges](#33-the-harmonic-series-diverges)
  - [3.4 p-Series](#34-p-series)
  - [3.5 Telescoping Series](#35-telescoping-series)
- [4. Convergence Tests](#4-convergence-tests)
  - [4.1 Divergence Test](#41-divergence-test)
  - [4.2 Integral Test](#42-integral-test)
  - [4.3 Comparison and Limit Comparison Tests](#43-comparison-and-limit-comparison-tests)
  - [4.4 Ratio Test](#44-ratio-test)
  - [4.5 Root Test](#45-root-test)
  - [4.6 Alternating Series Test](#46-alternating-series-test)
  - [4.7 Absolute vs. Conditional Convergence](#47-absolute-vs-conditional-convergence)
- [5. Power Series](#5-power-series)
  - [5.1 Definition and Convergence Interval](#51-definition-and-convergence-interval)
  - [5.2 Radius of Convergence](#52-radius-of-convergence)
  - [5.3 Behavior at Endpoints](#53-behavior-at-endpoints)
  - [5.4 Term-by-Term Differentiation and Integration](#54-term-by-term-differentiation-and-integration)
  - [5.5 Uniqueness of Power Series Representations](#55-uniqueness-of-power-series-representations)
- [6. Taylor and Maclaurin Series](#6-taylor-and-maclaurin-series)
  - [6.1 Deriving the Taylor Series Formula](#61-deriving-the-taylor-series-formula)
  - [6.2 Standard Maclaurin Series](#62-standard-maclaurin-series)
  - [6.3 Lagrange Remainder and Error Bounds](#63-lagrange-remainder-and-error-bounds)
  - [6.4 Analytic Functions and Convergence Domain](#64-analytic-functions-and-convergence-domain)
  - [6.5 Algebraic Manipulation of Taylor Series](#65-algebraic-manipulation-of-taylor-series)
- [7. Key Series in Machine Learning](#7-key-series-in-machine-learning)
  - [7.1 Softmax and Temperature Scaling](#71-softmax-and-temperature-scaling)
  - [7.2 GELU via Hermite Polynomials](#72-gelu-via-hermite-polynomials)
  - [7.3 Optimizer Update Rules as Taylor Approximations](#73-optimizer-update-rules-as-taylor-approximations)
  - [7.4 Linear Attention via Taylor Expansion of exp](#74-linear-attention-via-taylor-expansion-of-exp)
  - [7.5 Log-Sum-Exp and Numerical Stability](#75-log-sum-exp-and-numerical-stability)
  - [7.6 Laplace Approximation](#76-laplace-approximation)
- [8. Fourier Series (Preview)](#8-fourier-series-preview)
  - [8.1 Periodic Functions as Trigonometric Series](#81-periodic-functions-as-trigonometric-series)
  - [8.2 Orthogonality and Fourier Coefficients](#82-orthogonality-and-fourier-coefficients)
  - [8.3 ML Relevance: Positional Encodings and RoPE](#83-ml-relevance-positional-encodings-and-rope)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)
- [Appendix A: Extended Convergence Proofs](#appendix-a-extended-convergence-proofs)
- [Appendix B: Uniform Convergence](#appendix-b-uniform-convergence)
- [Appendix C: Complex Power Series](#appendix-c-complex-power-series)
- [Appendix D: Generating Functions](#appendix-d-generating-functions)
- [Appendix E: Multivariable Taylor (Preview)](#appendix-e-multivariable-taylor-preview)
- [Appendix F: Fourier Series Extended](#appendix-f-fourier-series-extended)
- [Appendix G: ML Derivations in Full](#appendix-g-ml-derivations-in-full)
- [Appendix H: Numerical Summation Methods](#appendix-h-numerical-summation-methods)
- [Appendix I: Glossary and Reference Tables](#appendix-i-glossary-and-reference-tables)

---

## 1. Intuition

### 1.1 What Sequences and Series Represent

A **sequence** is a function from the natural numbers to the real numbers: an infinite, ordered list $a_1, a_2, a_3, \ldots$ The key question is always: *where does this list go?* Does it home in on a specific value, grow without bound, or oscillate forever?

A **series** adds up all the terms: $\sum_{n=1}^\infty a_n = a_1 + a_2 + a_3 + \cdots$ This is more subtle. An infinite sum might converge to a finite value — or it might grow forever. The sum $1 + \frac{1}{2} + \frac{1}{4} + \cdots = 2$ converges. The sum $1 + \frac{1}{2} + \frac{1}{3} + \cdots$ — the harmonic series — diverges to infinity, even though each term goes to zero. This counterintuitive behavior is why convergence tests exist.

```
SEQUENCES vs. SERIES — VISUAL INTUITION
════════════════════════════════════════════════════════════════════════

  SEQUENCE {aₙ}: where do the terms go?
  ─────────────────────────────────────
  aₙ = 1/n:   1, 1/2, 1/3, 1/4, ...  →  0      (converges)
  aₙ = n:     1,   2,   3,   4, ...  →  ∞      (diverges)
  aₙ = (-1)ⁿ: -1,  1, -1,  1, ...   →  ?      (oscillates, diverges)

  SERIES Σaₙ: where does the cumulative sum go?
  ─────────────────────────────────────────────
  Σ (1/2)ⁿ:  1/2 + 1/4 + 1/8 + ...  →  1      (converges, geometric)
  Σ 1/n:     1 + 1/2 + 1/3 + ...    →  ∞      (diverges! harmonic)
  Σ 1/n²:    1 + 1/4 + 1/9 + ...    → π²/6    (converges, Basel problem)

  Key insight: aₙ → 0 is NECESSARY but NOT SUFFICIENT for Σaₙ to converge.

════════════════════════════════════════════════════════════════════════
```

The fundamental distinction between sequences and series is that sequences ask about the terms themselves; series ask about the cumulative totals. A sequence can converge to zero while its series diverges — the harmonic series is the canonical example.

### 1.2 Historical Context

The history of series is a history of infinity being tamed:

| Era | Mathematician | Achievement |
|-----|--------------|-------------|
| ~250 BCE | Archimedes | Geometric series sum $\sum r^n$ for quadrature of parabola |
| 1350 | Nicole Oresme | First proof that the harmonic series diverges |
| 1600s | Newton | Binomial series $(1+x)^\alpha$; power series for $\sin$, $\cos$, $\ln$ |
| 1735 | Euler | Basel problem: $\sum 1/n^2 = \pi^2/6$; $e^{i\pi} + 1 = 0$ |
| 1748 | Euler | *Introductio*: systematic treatment of infinite series |
| 1821 | Cauchy | Rigorous convergence theory; ratio test; uniform convergence concept |
| 1826 | Abel | Abel's theorem on power series at boundary |
| 1837 | Dirichlet | Fourier series convergence conditions; conditional vs. absolute convergence |
| 1854 | Riemann | Rearrangement theorem: conditionally convergent series can be rearranged to any sum |
| 1900 | Taylor | Taylor series already known since Gregory (1668), but named after Brook Taylor (1715) |

The key conceptual revolution was Cauchy's insistence on rigor. Before him, mathematicians manipulated series freely, often getting wrong answers. The definition of convergence as a limit of partial sums made the subject precise and allowed the rich theory we use today.

### 1.3 Why Series Are Central to AI

Series appear throughout modern machine learning — often hidden in plain sight:

| ML Context | Series Connection |
|-----------|-------------------|
| **Softmax function** | $\text{softmax}(\mathbf{x}/T)_i = e^{x_i/T}/\sum_j e^{x_j/T}$ — the denominator is a sum over all classes; as $T \to 0$, only the max survives (argmax); as $T \to \infty$, uniform distribution (Taylor expansion of $e^x \approx 1+x$ dominates) |
| **Linear attention** | Approximates $\exp(qk^\top) \approx 1 + qk^\top$ (first-order Taylor) to reduce complexity from $O(n^2)$ to $O(n)$ |
| **GELU activation** | Defined via error function, approximated by Taylor-Hermite expansion |
| **Adam optimizer** | Bias correction $\hat{m}_t = m_t/(1-\beta^t)$ uses geometric series $\sum \beta^k$; step size derivation uses Taylor expansion |
| **Positional encodings** | Sinusoidal (original Transformer), RoPE — both are Fourier series evaluated at discrete positions |
| **Laplace approximation** | Quadratic Taylor expansion of $\log p(\theta|\mathcal{D})$ around the MAP estimate $\hat\theta$ |
| **Neural ODEs** | Euler method is first-order Taylor; higher-order Runge-Kutta methods use more Taylor terms |
| **Residual networks** | $\text{ResNet}(x) = x + F(x) \approx e^F(x)$ — the skip connections are the partial sums of a Taylor series for the exponential of the network function |


---

## 2. Sequences

### 2.1 Formal Definition

**Definition**: A **sequence** is a function $a: \mathbb{N} \to \mathbb{R}$. We write the sequence as $(a_n)_{n=1}^\infty$ or $\{a_n\}$, where $a_n = a(n)$ is the $n$-th term.

This formal view — sequence as function — is the right way to think about it. Everything we know about functions (domain, range, composition) applies to sequences.

**Examples**:
- $a_n = \frac{1}{n}$: terms $1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \ldots$ — decreasing, converges to $0$
- $a_n = (-1)^n$: terms $-1, 1, -1, 1, \ldots$ — oscillates, does not converge
- $a_n = \left(1 + \frac{1}{n}\right)^n$: terms $2, 2.25, 2.370\ldots, 2.441\ldots$ — converges to $e$
- $a_n = \frac{n!}{n^n}$: terms decrease rapidly to $0$ (Stirling: $n! \approx \sqrt{2\pi n}(n/e)^n$)
- $a_n = r^n$ for $|r| < 1$: geometric decay, converges to $0$

**Non-examples** (not standard sequences):
- $a_n = \frac{1}{n - 5}$ for $n = 5$: undefined at $n=5$; need to restrict domain to $n > 5$
- $a_n = \sin(n)$: well-defined for all $n \in \mathbb{N}$, but its limit behavior is subtle (it doesn't converge)

### 2.2 Convergence: ε-N Definition

**Definition**: A sequence $(a_n)$ **converges** to the limit $L \in \mathbb{R}$ if for every $\epsilon > 0$ there exists $N \in \mathbb{N}$ such that for all $n > N$:
$$|a_n - L| < \epsilon$$

We write $\lim_{n\to\infty} a_n = L$ or $a_n \to L$. If no such $L$ exists, the sequence **diverges**.

**Reading the definition**: For any desired accuracy $\epsilon$ (no matter how small), there is a threshold $N$ beyond which every term is within $\epsilon$ of $L$. The threshold $N$ may depend on $\epsilon$ — tighter accuracy requires waiting longer.

**Worked example**: Prove $\lim_{n\to\infty} \frac{1}{n} = 0$.

*Proof*: Given $\epsilon > 0$, choose $N = \lceil 1/\epsilon \rceil$ (the smallest integer $\ge 1/\epsilon$). Then for all $n > N$:
$$\left|\frac{1}{n} - 0\right| = \frac{1}{n} < \frac{1}{N} \le \epsilon$$
Therefore $\frac{1}{n} \to 0$. $\square$

**Uniqueness**: If $(a_n)$ converges, its limit is unique. (Proof: if $a_n \to L$ and $a_n \to M$, then for any $\epsilon > 0$, $|L - M| \le |L - a_n| + |a_n - M| < \epsilon + \epsilon = 2\epsilon$ for large $n$; since $\epsilon$ is arbitrary, $|L-M| = 0$.)

**Limit laws for sequences**: If $a_n \to L$ and $b_n \to M$, then:
- $a_n + b_n \to L + M$
- $a_n b_n \to LM$
- $a_n/b_n \to L/M$ provided $M \ne 0$ and $b_n \ne 0$ eventually
- $f(a_n) \to f(L)$ if $f$ is continuous at $L$ (sequential characterization of continuity)

**Squeeze theorem for sequences**: If $a_n \le b_n \le c_n$ and $a_n \to L$ and $c_n \to L$, then $b_n \to L$.

### 2.3 Bounded and Monotone Sequences

**Definition**: A sequence $(a_n)$ is:
- **Bounded above** if there exists $M$ such that $a_n \le M$ for all $n$
- **Bounded below** if there exists $m$ such that $a_n \ge m$ for all $n$
- **Bounded** if both: $|a_n| \le B$ for some $B$ and all $n$
- **Monotone increasing** if $a_{n+1} \ge a_n$ for all $n$ (strictly increasing if $>$)
- **Monotone decreasing** if $a_{n+1} \le a_n$ for all $n$

**Theorem (Monotone Convergence Theorem — MCT for sequences)**: Every bounded monotone sequence converges.

*More precisely*: If $(a_n)$ is monotone increasing and bounded above, then $a_n \to \sup\{a_n : n \in \mathbb{N}\}$.

*Proof sketch*: Let $L = \sup\{a_n\}$, which exists by the completeness of $\mathbb{R}$ (every non-empty set bounded above has a supremum). Given $\epsilon > 0$, by definition of supremum, there exists $N$ with $a_N > L - \epsilon$. Since the sequence is increasing, $a_n \ge a_N > L - \epsilon$ for all $n \ge N$. Also $a_n \le L < L + \epsilon$. So $|a_n - L| < \epsilon$ for all $n \ge N$. $\square$

**Key consequence**: To prove convergence without knowing the limit, it suffices to show the sequence is monotone and bounded. This is widely used in analysis to establish existence of limits.

**Example**: $a_n = (1 + 1/n)^n$ is increasing and bounded above by $3$ (provable by AM-GM). Therefore it converges. Its limit is defined to be $e \approx 2.71828$.

**For ML**: Gradient descent sequences $(θ_t)$ are not monotone in parameter space, but the **loss sequence** $(L(θ_t))$ is often bounded below (losses are non-negative) and may be monotone decreasing. The MCT guarantees loss convergence for many convex problems.

### 2.4 Subsequences and Bolzano-Weierstrass

**Definition**: A **subsequence** of $(a_n)$ is a sequence of the form $(a_{n_k})_{k=1}^\infty$ where $n_1 < n_2 < n_3 < \cdots$ is a strictly increasing sequence of indices.

**Theorem**: If $a_n \to L$, then every subsequence also converges to $L$.

*Converse (useful)**: If a sequence has two subsequences converging to different limits, the original sequence diverges. This proves $(-1)^n$ diverges: the subsequences $(-1)^{2k} = 1$ and $(-1)^{2k+1} = -1$ converge to $1$ and $-1$ respectively.

**Theorem (Bolzano-Weierstrass)**: Every bounded sequence has a convergent subsequence.

*Proof idea (bisection)*: Let $(a_n)$ be bounded in $[m, M]$. Bisect the interval into $[m, \frac{m+M}{2}]$ and $[\frac{m+M}{2}, M]$. At least one half contains infinitely many terms — pick that half and pick any term in it as $a_{n_1}$. Repeat: bisect the chosen half, pick the half with infinitely many terms, pick $a_{n_2}$ from it (with $n_2 > n_1$). The resulting intervals have lengths $\frac{M-m}{2^k} \to 0$, so the chosen terms form a Cauchy sequence and hence converge. $\square$

**For ML**: Bolzano-Weierstrass underlies the proof that compact sets in $\mathbb{R}^n$ always contain convergent subsequences — which is why SGD on compact parameter spaces always has convergent subsequences (though full convergence requires additional conditions).

### 2.5 Important Sequences in ML

The following sequences appear constantly in machine learning. Understanding their convergence is practically important:

```
ML SEQUENCES — CONVERGENCE ANALYSIS
════════════════════════════════════════════════════════════════════════

  Learning rate schedules:
  ─────────────────────────────────────────────────────────────────
  Step decay:    αₜ = α₀ · γ^⌊t/s⌋         geometric, → 0
  Cosine:        αₜ = α₀/2 · (1 + cos(πt/T)) bounded, periodic
  Warmup:        αₜ = t·α₀/T for t ≤ T      linear ramp, then decay
  Polynomial:    αₜ = α₀/(1 + βt)^p         → 0 if p > 0

  Exponential moving average (EMA / Adam momentum):
  ─────────────────────────────────────────────────────────────────
  m_t = β·m_{t-1} + (1-β)·g_t
      = (1-β) Σₖ₌₀^{t-1} β^k · g_{t-k}     geometric series weights!
  As t → ∞: weights sum to 1 (geometric series identity)
  Bias-corrected: m̂_t = m_t / (1 - βᵗ)    corrects for initial zero

  Convergence of gradient norms in Adam:
  ─────────────────────────────────────────────────────────────────
  Required for convergence: Σ αₜ = ∞ (enough total movement)
  And:                      Σ αₜ² < ∞ (steps shrink fast enough)
  This is the Robbins-Monro condition from stochastic approximation.

════════════════════════════════════════════════════════════════════════
```

**Exponential moving average as geometric series**: The Adam first moment $m_t = \beta m_{t-1} + (1-\beta)g_t$ unfolds to:
$$m_t = (1-\beta)\sum_{k=0}^{t-1} \beta^k g_{t-k} + \beta^t m_0$$

For $\beta = 0.9$ and $m_0 = 0$: the weights $(1-\beta)\beta^k$ form a geometric series summing to $1 - \beta^t$. The bias correction $\hat{m}_t = m_t/(1-\beta^t)$ normalizes this to 1, giving a proper weighted average of past gradients. As $t \to \infty$, $\beta^t \to 0$ and the correction becomes negligible.


---

## 3. Infinite Series

### 3.1 Partial Sums and Convergence

**Definition**: Given a sequence $(a_n)_{n=1}^\infty$, the $N$-th **partial sum** is:
$$S_N = \sum_{n=1}^N a_n = a_1 + a_2 + \cdots + a_N$$

The **infinite series** $\sum_{n=1}^\infty a_n$ **converges** to $S$ if the sequence of partial sums $(S_N)$ converges to $S$:
$$\sum_{n=1}^\infty a_n = S \iff \lim_{N\to\infty} S_N = S$$

If $(S_N)$ diverges, the series **diverges**.

**Properties** (inherited from sequence limit laws):
- **Linearity**: $\sum (ca_n + db_n) = c\sum a_n + d\sum b_n$ (if both converge)
- **Tail behavior**: $\sum_{n=1}^\infty a_n$ converges iff $\sum_{n=k}^\infty a_n$ converges for any fixed $k$ (convergence is about the tail, not finitely many initial terms)
- **Term rearrangement**: For absolutely convergent series, any rearrangement converges to the same sum. For conditionally convergent series, rearrangement can change or destroy convergence (Riemann's theorem).

### 3.2 Geometric Series

The **geometric series** is the most important series in mathematics:
$$\sum_{n=0}^\infty r^n = 1 + r + r^2 + r^3 + \cdots$$

**Theorem**: The geometric series converges if and only if $|r| < 1$, and:
$$\sum_{n=0}^\infty r^n = \frac{1}{1-r}$$

**Proof**: The $N$-th partial sum satisfies $S_N = 1 + r + \cdots + r^N$. Multiply by $r$: $rS_N = r + r^2 + \cdots + r^{N+1}$. Subtract: $S_N - rS_N = 1 - r^{N+1}$, so $S_N = \frac{1-r^{N+1}}{1-r}$. As $N \to \infty$: if $|r| < 1$ then $r^{N+1} \to 0$, so $S_N \to \frac{1}{1-r}$. If $|r| \ge 1$, $r^{N+1}$ does not go to zero, so $S_N$ diverges. $\square$

**Shifted and scaled versions**:
$$\sum_{n=0}^\infty ar^n = \frac{a}{1-r}, \qquad \sum_{n=k}^\infty r^n = \frac{r^k}{1-r}, \qquad \sum_{n=1}^\infty nr^{n-1} = \frac{1}{(1-r)^2}$$

The last identity comes from differentiating $\sum r^n = \frac{1}{1-r}$ term-by-term.

**ML applications of geometric series**:

1. **Discounted future rewards (RL)**: $G_t = \sum_{k=0}^\infty \gamma^k r_{t+k} = r_t + \gamma G_{t+1}$. The total return is a geometric series with ratio $\gamma \in [0,1)$; it converges to $r_{\text{max}}/(1-\gamma)$ when rewards are bounded.

2. **Adam bias correction**: $m_t = (1-\beta)\sum_{k=0}^{t-1}\beta^k g_{t-k}$. Sum of weights: $(1-\beta)\sum_{k=0}^{t-1}\beta^k = 1 - \beta^t$. Correction factor: $1/(1-\beta^t)$.

3. **Infinite-horizon value function**: $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s]$, which is the Bellman equation's foundation.

4. **Attention denominator as geometric series**: For scalar queries and keys with $q = k = c$ (constant), $\text{softmax}(c)_i = e^c/\sum_j e^c = 1/n$ — the denominator is a finite geometric-like sum.

### 3.3 The Harmonic Series Diverges

The **harmonic series** $\sum_{n=1}^\infty \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \frac{1}{4} + \cdots$ diverges despite $\frac{1}{n} \to 0$.

**Proof (Oresme's grouping argument, ~1350)**: Group terms in blocks of doubling size:
$$1 + \frac{1}{2} + \underbrace{\frac{1}{3} + \frac{1}{4}}_{\ge 2 \cdot \frac{1}{4} = \frac{1}{2}} + \underbrace{\frac{1}{5} + \frac{1}{6} + \frac{1}{7} + \frac{1}{8}}_{\ge 4 \cdot \frac{1}{8} = \frac{1}{2}} + \underbrace{\frac{1}{9} + \cdots + \frac{1}{16}}_{\ge 8 \cdot \frac{1}{16} = \frac{1}{2}} + \cdots$$

Each block contributes at least $\frac{1}{2}$. Adding infinitely many blocks of $\frac{1}{2}$ gives infinity. $\square$

**Why this matters**: The harmonic series illustrates that $a_n \to 0$ alone does not guarantee convergence. Every convergence test addresses this problem: it must establish convergence faster than harmonic decay.

**The harmonic series in ML**: The divergence of $\sum 1/n$ is why the learning rate schedule $\alpha_t = \alpha_0/t$ satisfies the Robbins-Monro condition $\sum \alpha_t = \infty$ — it decreases slowly enough to ensure the optimizer explores sufficiently. The companion condition $\sum \alpha_t^2 < \infty$ holds since $\sum 1/t^2 < \infty$ (p-series with $p = 2 > 1$).

### 3.4 p-Series

The **p-series** is $\sum_{n=1}^\infty \frac{1}{n^p}$ for $p > 0$.

**Theorem**: The p-series converges if and only if $p > 1$.

| $p$ | Series | Sum | Converges? |
|-----|--------|-----|-----------|
| $p = 1$ | Harmonic | $\infty$ | No |
| $p = 2$ | $\sum 1/n^2$ | $\pi^2/6$ (Basel) | Yes |
| $p = 3$ | $\sum 1/n^3$ | $\approx 1.202$ (Apéry) | Yes |
| $p = 1/2$ | $\sum 1/\sqrt{n}$ | $\infty$ | No |
| $p = 0.99$ | $\sum 1/n^{0.99}$ | $\infty$ | No (barely!) |

*Proof*: By the integral test (proven in §4.2): $\sum_{n=1}^\infty \frac{1}{n^p}$ converges iff $\int_1^\infty \frac{1}{x^p}\,dx$ converges. This integral equals $\frac{x^{1-p}}{1-p}\Big|_1^\infty$ for $p \ne 1$. For $p > 1$: $x^{1-p} \to 0$ as $x \to \infty$, so the integral $= \frac{1}{p-1}$ (converges). For $p < 1$: $x^{1-p} \to \infty$, so the integral diverges. For $p = 1$: $\int_1^\infty \frac{1}{x}\,dx = \ln x\big|_1^\infty = \infty$. $\square$

### 3.5 Telescoping Series

A **telescoping series** has a form that allows consecutive cancellation:
$$\sum_{n=1}^N (b_n - b_{n+1}) = b_1 - b_{N+1} \to b_1 - \lim_{n\to\infty} b_n$$

**Example**: $\sum_{n=1}^\infty \frac{1}{n(n+1)} = \sum_{n=1}^\infty \left(\frac{1}{n} - \frac{1}{n+1}\right) = 1 - 0 = 1$

*Proof via partial fractions*: $\frac{1}{n(n+1)} = \frac{1}{n} - \frac{1}{n+1}$. Then $S_N = (1 - \frac{1}{2}) + (\frac{1}{2} - \frac{1}{3}) + \cdots + (\frac{1}{N} - \frac{1}{N+1}) = 1 - \frac{1}{N+1} \to 1$.

**Another example**: $\sum_{n=1}^\infty \frac{1}{\sqrt{n+1} + \sqrt{n}} = \sum_{n=1}^\infty (\sqrt{n+1} - \sqrt{n})$ diverges (telescopes to $\sqrt{N+1} - 1 \to \infty$).


---

## 4. Convergence Tests

The fundamental challenge: given a series $\sum a_n$, does it converge? The convergence tests below address different structural properties of $(a_n)$.

```
CONVERGENCE TEST SELECTION GUIDE
════════════════════════════════════════════════════════════════════════

  aₙ → 0? → NO → DIVERGES (divergence test)

  Is aₙ = f(n) with f decreasing, positive?
    → Use INTEGRAL TEST: compare to ∫f(x)dx

  Is aₙ comparable to a known series bₙ?
    → aₙ ≤ Cbₙ and Σbₙ converges → CONVERGES
    → aₙ ≥ Cbₙ and Σbₙ diverges  → DIVERGES (comparison)
    → lim aₙ/bₙ = L ∈ (0,∞) → same behavior (limit comparison)

  Does aₙ involve factorials or exponentials?
    → Try RATIO TEST: L = lim |aₙ₊₁/aₙ|
       L < 1 → converges; L > 1 → diverges; L = 1 → inconclusive

  Does aₙ involve nᵗʰ powers?
    → Try ROOT TEST: L = lim |aₙ|^{1/n}
       L < 1 → converges; L > 1 → diverges; L = 1 → inconclusive

  Does the series alternate signs?
    → Try ALTERNATING SERIES TEST: aₙ decreasing, → 0 → converges

════════════════════════════════════════════════════════════════════════
```

### 4.1 Divergence Test

**Theorem (Divergence Test)**: If $\sum_{n=1}^\infty a_n$ converges, then $a_n \to 0$.

*Equivalently (contrapositive)*: If $a_n \not\to 0$, then $\sum a_n$ diverges.

*Proof*: If $\sum a_n = S$, then $a_n = S_n - S_{n-1} \to S - S = 0$. $\square$

**Important**: The converse is FALSE. $a_n \to 0$ does NOT imply $\sum a_n$ converges. The harmonic series is the canonical counterexample.

**Examples**:
- $\sum \frac{n}{n+1}$: since $\frac{n}{n+1} \to 1 \ne 0$, diverges immediately.
- $\sum (-1)^n$: since $(-1)^n$ does not converge to $0$, diverges.
- $\sum \frac{1}{n}$: $\frac{1}{n} \to 0$, so the test is inconclusive. Must use another test.

### 4.2 Integral Test

**Theorem (Integral Test)**: Suppose $f: [1, \infty) \to \mathbb{R}$ is continuous, positive, and decreasing, with $a_n = f(n)$. Then:
$$\sum_{n=1}^\infty a_n \text{ converges} \iff \int_1^\infty f(x)\,dx \text{ converges}$$

*Proof sketch*: Since $f$ is decreasing, $f(n) \ge f(x) \ge f(n+1)$ for $x \in [n, n+1]$, so:
$$f(n) = \int_n^{n+1} f(n)\,dx \ge \int_n^{n+1} f(x)\,dx \ge \int_n^{n+1} f(n+1)\,dx = f(n+1)$$

Summing: $\sum_{n=1}^N f(n) \ge \int_1^{N+1} f(x)\,dx \ge \sum_{n=2}^{N+1} f(n)$. As $N \to \infty$, the partial sums and the integral bound each other, so both converge or both diverge. $\square$

**Applications**:
- p-series: $\int_1^\infty x^{-p}\,dx$ converges iff $p > 1$ → p-series converges iff $p > 1$
- $\sum \frac{1}{n\ln n}$: $\int_2^\infty \frac{1}{x\ln x}\,dx = \ln(\ln x)\big|_2^\infty = \infty$ → diverges
- $\sum \frac{1}{n(\ln n)^2}$: $\int_2^\infty \frac{1}{x(\ln x)^2}\,dx = \frac{-1}{\ln x}\big|_2^\infty = \frac{1}{\ln 2}$ → converges

**Error bound from integral test**: If the series converges,
$$\int_{N+1}^\infty f(x)\,dx \le \sum_{n=N+1}^\infty a_n \le \int_N^\infty f(x)\,dx$$

This gives a computable bound on the tail error after $N$ terms.

### 4.3 Comparison and Limit Comparison Tests

**Theorem (Direct Comparison)**: Suppose $0 \le a_n \le b_n$ for all sufficiently large $n$.
- If $\sum b_n$ converges, then $\sum a_n$ converges.
- If $\sum a_n$ diverges, then $\sum b_n$ diverges.

*Proof*: For the first part: partial sums of $\sum a_n$ are bounded above by the (convergent) sum of $\sum b_n$, and they're increasing, so they converge by MCT. $\square$

**Theorem (Limit Comparison)**: Suppose $a_n, b_n > 0$ and $\lim_{n\to\infty} \frac{a_n}{b_n} = L$ with $0 < L < \infty$. Then $\sum a_n$ and $\sum b_n$ either both converge or both diverge.

*Why*: For large $n$, $a_n \approx Lb_n$, so $a_n$ and $b_n$ have the same order of magnitude.

**Examples**:
- $\sum \frac{1}{n^2 + 1}$: compare to $\sum \frac{1}{n^2}$ (p-series, converges). Limit comparison: $\frac{1/(n^2+1)}{1/n^2} = \frac{n^2}{n^2+1} \to 1 \in (0,\infty)$. Conclusion: converges.
- $\sum \frac{3n^2 - 1}{n^4 + 2n}$: dominant term is $\frac{3}{n^2}$. Limit comparison with $1/n^2$: ratio $\to 3$. Converges.

### 4.4 Ratio Test

**Theorem (Ratio Test, d'Alembert)**: Let $L = \lim_{n\to\infty} \left|\frac{a_{n+1}}{a_n}\right|$.
- If $L < 1$: $\sum a_n$ **converges absolutely**
- If $L > 1$ (or $L = \infty$): $\sum a_n$ **diverges**
- If $L = 1$: **inconclusive**

*Proof (convergence case)*: If $L < 1$, pick $r$ with $L < r < 1$. Then $|a_{n+1}| \le r|a_n|$ for all large $n$. By induction, $|a_n| \le C r^n$ for some constant $C$. So $\sum |a_n| \le C \sum r^n < \infty$ (geometric series). $\square$

**Best for**: Factorials ($n!$), exponentials ($c^n$), or products involving $n$.

**Examples**:
- $\sum \frac{n!}{n^n}$: ratio $= \frac{(n+1)!/(n+1)^{n+1}}{n!/n^n} = \frac{(n+1)n^n}{(n+1)^{n+1}} = \frac{n^n}{(n+1)^n} = \left(\frac{n}{n+1}\right)^n \to e^{-1} < 1$ → converges
- $\sum \frac{2^n}{n!}$: ratio $= \frac{2^{n+1}/(n+1)!}{2^n/n!} = \frac{2}{n+1} \to 0 < 1$ → converges

### 4.5 Root Test

**Theorem (Root Test, Cauchy)**: Let $L = \lim_{n\to\infty} |a_n|^{1/n}$.
- If $L < 1$: $\sum a_n$ **converges absolutely**
- If $L > 1$: **diverges**
- If $L = 1$: **inconclusive**

*Proof*: Same structure as ratio test — if $L < 1$, eventually $|a_n|^{1/n} \le r < 1$, so $|a_n| \le r^n$, a convergent geometric series. $\square$

**Best for**: Terms of the form $f(n)^n$.

**Example**: $\sum \left(\frac{n}{2n+1}\right)^n$: root test gives $L = \lim \frac{n}{2n+1} = \frac{1}{2} < 1$ → converges.

**Relationship to ratio test**: If the ratio test limit exists, the root test limit also exists and equals the same value. The root test is stronger (applicable when ratio test fails), but harder to compute.

### 4.6 Alternating Series Test

**Definition**: An **alternating series** has the form $\sum (-1)^{n+1} b_n = b_1 - b_2 + b_3 - b_4 + \cdots$ where $b_n > 0$.

**Theorem (Alternating Series Test / Leibniz criterion)**: If:
1. $b_n > 0$ for all $n$
2. $b_{n+1} \le b_n$ (decreasing)
3. $b_n \to 0$

Then $\sum (-1)^{n+1} b_n$ converges.

*Proof idea*: Even and odd partial sums are monotone and bounded, converging to a common limit. $\square$

**Error bound**: The error in approximating the sum by $S_N$ satisfies $|S - S_N| \le b_{N+1}$.

**Key example**: The alternating harmonic series $\sum_{n=1}^\infty \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots = \ln 2$.

This is **conditionally convergent** — it converges but $\sum |1/n| = \infty$ (harmonic series diverges).

### 4.7 Absolute vs. Conditional Convergence

**Definition**: A series $\sum a_n$ is:
- **Absolutely convergent** if $\sum |a_n|$ converges
- **Conditionally convergent** if $\sum a_n$ converges but $\sum |a_n|$ diverges

**Theorem**: Absolute convergence implies convergence.

*Proof*: $0 \le a_n + |a_n| \le 2|a_n|$, so $\sum (a_n + |a_n|)$ converges by comparison. Then $\sum a_n = \sum(a_n + |a_n|) - \sum|a_n|$ converges as a difference of convergent series. $\square$

**Riemann Rearrangement Theorem**: If $\sum a_n$ is conditionally convergent, then for any $L \in \mathbb{R}$ (or $\pm\infty$), there exists a rearrangement of the terms that converges to $L$.

This shocking result means: the value of a conditionally convergent series depends on the order of summation! This is not a curiosity — it has direct implications for:
- **Stochastic gradient descent**: the order in which mini-batches are processed affects the path (though not the limit for convergent algorithms)
- **Floating-point summation**: the order of floating-point additions affects the accumulated rounding error

**For ML**: Practically, all series in ML are absolutely convergent (finite sums, or convergent geometric/p-series). But the theory of conditional convergence explains why naive summation of large arrays can lose precision.


---

## 5. Power Series

### 5.1 Definition and Convergence Interval

**Definition**: A **power series centered at $a$** is a series of the form:
$$\sum_{n=0}^\infty c_n (x-a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

where $c_n$ are the **coefficients** and $a$ is the **center**. When $a = 0$ we get a Maclaurin (or standard) power series $\sum c_n x^n$.

A power series is fundamentally different from the series we've studied: it is a **function of $x$**, not just a number. For each value of $x$, we get a numerical series that may or may not converge.

**Abel's Theorem on convergence structure**: For any power series $\sum c_n(x-a)^n$, exactly one of three things holds:
1. The series converges only at $x = a$
2. The series converges for all $x \in \mathbb{R}$
3. There exists $R > 0$ (the **radius of convergence**) such that the series converges absolutely for $|x - a| < R$ and diverges for $|x - a| > R$

The interval $(a-R, a+R)$ is the **interval of convergence** (open, possibly including one or both endpoints).

### 5.2 Radius of Convergence

**Theorem (Radius of Convergence via Ratio Test)**: If $\lim_{n\to\infty} \left|\frac{c_{n+1}}{c_n}\right| = L$, then $R = \frac{1}{L}$.

*Derivation*: Apply the ratio test to $\sum c_n(x-a)^n$:
$$\left|\frac{c_{n+1}(x-a)^{n+1}}{c_n(x-a)^n}\right| = |x-a| \cdot \left|\frac{c_{n+1}}{c_n}\right| \to |x-a| \cdot L$$

Ratio test converges when $|x-a| \cdot L < 1$, i.e., $|x-a| < 1/L = R$. $\square$

**Hadamard formula (root test version)**:
$$R = \frac{1}{\limsup_{n\to\infty} |c_n|^{1/n}}$$

This formula always works, even when the ratio test limit doesn't exist.

**Key cases**:
- $R = 0$: converges only at center (e.g., $\sum n! x^n$)
- $R = \infty$: converges for all $x$ (e.g., $\sum x^n/n!$, the exponential series)
- $R = 1$: converges in $(-1, 1)$ or possibly $[-1, 1]$ (e.g., $\sum x^n$, geometric series)

**Examples**:

$\sum_{n=0}^\infty \frac{x^n}{n!}$: ratio $= \frac{|x|^{n+1}/(n+1)!}{|x|^n/n!} = \frac{|x|}{n+1} \to 0$ for all $x$. So $R = \infty$.

$\sum_{n=0}^\infty n! \cdot x^n$: ratio $= (n+1)|x| \to \infty$ unless $x = 0$. So $R = 0$.

$\sum_{n=1}^\infty \frac{x^n}{n}$: ratio $\to |x|$, so $R = 1$. Interval: $(-1, 1)$ — check endpoints separately.

### 5.3 Behavior at Endpoints

The ratio test is inconclusive at $|x - a| = R$ (endpoints). Endpoints must be checked individually using series convergence tests.

**Example**: $\sum_{n=1}^\infty \frac{x^n}{n}$ has $R = 1$.
- At $x = 1$: $\sum \frac{1}{n}$ — harmonic series, **diverges**
- At $x = -1$: $\sum \frac{(-1)^n}{n}$ — alternating harmonic series, **converges** (to $-\ln 2$)

So the interval of convergence is $[-1, 1)$.

**Example**: $\sum_{n=1}^\infty \frac{x^n}{n^2}$ has $R = 1$.
- At $x = 1$: $\sum \frac{1}{n^2}$ — p-series with $p=2$, **converges**
- At $x = -1$: $\sum \frac{(-1)^n}{n^2}$ — absolutely convergent (compare to $\sum 1/n^2$), **converges**

Interval of convergence: $[-1, 1]$ (closed).

### 5.4 Term-by-Term Differentiation and Integration

**Theorem**: If $f(x) = \sum_{n=0}^\infty c_n(x-a)^n$ has radius of convergence $R > 0$, then:

1. **Term-by-term differentiation**: $f'(x) = \sum_{n=1}^\infty nc_n(x-a)^{n-1}$, with radius of convergence $R$

2. **Term-by-term integration**: $\int_a^x f(t)\,dt = \sum_{n=0}^\infty \frac{c_n}{n+1}(x-a)^{n+1}$, with radius of convergence $R$

Both operations preserve the radius of convergence (though endpoint behavior may change).

*Why this works*: Inside the interval of convergence, power series converge **uniformly** (see Appendix B), which allows interchange of limit and derivative/integral. This is the key technical condition.

**Application — deriving new series from known ones**:

From $\frac{1}{1-x} = \sum_{n=0}^\infty x^n$ (geometric series, $|x| < 1$):

*Differentiate*: $\frac{1}{(1-x)^2} = \sum_{n=1}^\infty nx^{n-1} = \sum_{n=0}^\infty (n+1)x^n$

*Integrate from 0 to x*: $-\ln(1-x) = \sum_{n=1}^\infty \frac{x^n}{n}$, so $\ln(1-x) = -\sum_{n=1}^\infty \frac{x^n}{n}$

*Substitute $x \to -x$*: $\ln(1+x) = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$

This is the Taylor series for $\ln(1+x)$, derived purely from the geometric series plus integration.

### 5.5 Uniqueness of Power Series Representations

**Theorem**: If $f(x) = \sum_{n=0}^\infty c_n(x-a)^n$ for all $x$ in some interval around $a$, then the coefficients are uniquely determined:
$$c_n = \frac{f^{(n)}(a)}{n!}$$

*Proof*: Set $x = a$: $f(a) = c_0$. Differentiate: $f'(a) = c_1$. Differentiate again: $f''(a) = 2c_2$, so $c_2 = f''(a)/2$. Inductively: $c_n = f^{(n)}(a)/n!$. $\square$

**Consequence**: The Taylor series of $f$ at $a$ is the *only* power series centered at $a$ that can represent $f$. If you find any power series for $f$, it must be the Taylor series.


---

## 6. Taylor and Maclaurin Series

### 6.1 Deriving the Taylor Series Formula

The question: given a smooth function $f$, can we represent it as a power series near a point $a$? If so, the coefficients must be $c_n = f^{(n)}(a)/n!$ (by §5.5). This leads to:

**Definition (Taylor Series)**: The **Taylor series** of $f$ at $x = a$ is:
$$f(x) \stackrel{?}{=} \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

The question mark is crucial: even if $f$ has derivatives of all orders, the Taylor series may not converge to $f$. The conditions for equality are given in §6.4.

**Intuition via polynomial fitting**: The $n$-th Taylor polynomial $T_n(x)$ is the unique polynomial of degree $\le n$ that matches $f$ and its first $n$ derivatives at $x = a$:
$$T_n(a) = f(a), \quad T_n'(a) = f'(a), \quad \ldots, \quad T_n^{(n)}(a) = f^{(n)}(a)$$

As $n \to \infty$, we hope $T_n(x) \to f(x)$ — this is the convergence question.

```
TAYLOR POLYNOMIAL APPROXIMATION
════════════════════════════════════════════════════════════════════════

  f(x) = sin(x) at a = 0:

  T₁(x) = x                        (tangent line)
  T₃(x) = x - x³/6                 (cubic approximation)
  T₅(x) = x - x³/6 + x⁵/120       (quintic approximation)
  T₇(x) = x - x³/6 + x⁵/120 - x⁷/5040

  Error |sin(x) - T₅(x)| at x = 0.1: ≈ 2.6 × 10⁻¹¹  (tiny!)
  Error |sin(x) - T₅(x)| at x = 2.0: ≈ 4.6 × 10⁻³   (small but visible)
  Error |sin(x) - T₅(x)| at x = π:   ≈ 1.9 × 10⁻¹   (large near π!)

  The polynomial is most accurate near x = 0 (the expansion point).

════════════════════════════════════════════════════════════════════════
```

### 6.2 Standard Maclaurin Series

The most important Maclaurin series (Taylor series at $a = 0$):

**Exponential**: $e^x = \sum_{n=0}^\infty \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$, $R = \infty$

*Derivation*: $f^{(n)}(x) = e^x$ for all $n$, so $f^{(n)}(0) = 1$, giving $c_n = 1/n!$.

**Sine**: $\sin x = \sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{6} + \frac{x^5}{120} - \cdots$, $R = \infty$

**Cosine**: $\cos x = \sum_{n=0}^\infty \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2} + \frac{x^4}{24} - \cdots$, $R = \infty$

*Note*: $\sin$ has only odd powers; $\cos$ has only even powers — reflecting their symmetry properties.

**Natural logarithm**: $\ln(1+x) = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$, $R = 1$

**Geometric series**: $\frac{1}{1-x} = \sum_{n=0}^\infty x^n = 1 + x + x^2 + \cdots$, $R = 1$

**Binomial series**: $(1+x)^\alpha = \sum_{n=0}^\infty \binom{\alpha}{n} x^n$, $R = 1$ (for any $\alpha \in \mathbb{R}$)

where $\binom{\alpha}{n} = \frac{\alpha(\alpha-1)\cdots(\alpha-n+1)}{n!}$ are generalized binomial coefficients.

**Arctangent**: $\arctan x = \sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{2n+1} = x - \frac{x^3}{3} + \frac{x^5}{5} - \cdots$, $R = 1$

*Interesting consequence*: At $x=1$: $\pi/4 = 1 - 1/3 + 1/5 - \cdots$ (Leibniz formula for $\pi$ — converges very slowly).

### 6.3 Lagrange Remainder and Error Bounds

**Theorem (Taylor's Theorem with Lagrange Remainder)**: If $f$ has $n+1$ continuous derivatives on an interval containing $a$ and $x$, then:
$$f(x) = T_n(x) + R_n(x)$$

where the **Lagrange remainder** is:
$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$$

for some $c$ strictly between $a$ and $x$.

**Error bound**: If $|f^{(n+1)}(t)| \le M$ for all $t$ between $a$ and $x$, then:
$$|R_n(x)| \le \frac{M}{(n+1)!}|x-a|^{n+1}$$

*Proof sketch*: Apply the Cauchy Mean Value Theorem or integrate the $(n+1)$-th derivative $n+1$ times. $\square$

**Worked example**: Bound the error in approximating $e^{0.1}$ by $T_3(x) = 1 + x + x^2/2 + x^3/6$ at $x = 0.1$, $a = 0$.

$f^{(4)}(x) = e^x$. On $[0, 0.1]$: $|e^x| \le e^{0.1} < 1.11$, so $M = 1.11$.
$$|R_3(0.1)| \le \frac{1.11}{4!}(0.1)^4 = \frac{1.11 \times 0.0001}{24} \approx 4.6 \times 10^{-6}$$

True value: $e^{0.1} \approx 1.10517$; $T_3(0.1) = 1 + 0.1 + 0.005 + 0.000167 = 1.105167$. Error $\approx 3.3 \times 10^{-7}$ (even smaller than our bound, as expected).

**For ML**: The Lagrange remainder quantifies how many Taylor terms are needed for a desired approximation accuracy. This determines, e.g., how many terms of the linear attention approximation are needed for a given accuracy trade-off.

### 6.4 Analytic Functions and Convergence Domain

A function is **analytic** at $a$ if its Taylor series at $a$ converges to $f(x)$ in some neighborhood of $a$. All functions we encounter in ML are analytic wherever they are smooth.

**However**: A function can be infinitely differentiable (smooth) without being analytic. The classic example:
$$f(x) = \begin{cases} e^{-1/x^2} & x \ne 0 \\ 0 & x = 0 \end{cases}$$

This function has $f^{(n)}(0) = 0$ for all $n$, so its Taylor series is the zero series — which does NOT converge to $f$ (except at $x = 0$).

**For ML functions**: All standard activation functions and loss functions are analytic on their domains. The Taylor series of $\sigma(x) = 1/(1+e^{-x})$, $\text{ReLU}(x)$, and $\text{GELU}(x)$ converge to the function on appropriate intervals.

**Domain of convergence**: For $\ln(1+x)$, the Taylor series has $R = 1$ — it converges in $(-1, 1]$ but not at $x = -1$ or for $|x| > 1$. Computing $\ln(1 + x)$ for large $x$ via truncated Taylor series will give wrong results. This is why implementations use the formula $\ln(1 + e^x) \approx x + e^{-x}$ for large $x$ (log-sum-exp trick).

### 6.5 Algebraic Manipulation of Taylor Series

Taylor series can be combined algebraically to produce new series without recomputing from scratch.

**Substitution**: Replace $x$ with a multiple, power, or function:
- $e^{-x^2} = \sum \frac{(-x^2)^n}{n!} = \sum \frac{(-1)^n x^{2n}}{n!}$ (substitute $x \to -x^2$ in $e^x$)
- $\cos(x^2) = \sum \frac{(-1)^n x^{4n}}{(2n)!}$ (substitute $x \to x^2$ in $\cos x$)

**Multiplication**: Cauchy product of two absolutely convergent series:
$$\left(\sum_{n=0}^\infty a_n x^n\right)\left(\sum_{n=0}^\infty b_n x^n\right) = \sum_{n=0}^\infty \left(\sum_{k=0}^n a_k b_{n-k}\right) x^n$$

**Composition**: $\sin(e^x - 1) = \sin\left(x + x^2/2 + \cdots\right) = x + x^2/2 + \cdots - \frac{(x+\cdots)^3}{6} + \cdots$

Collect terms by power of $x$.

**Example — softmax Taylor expansion**: Near $T = \infty$ (high temperature), $e^{x_i/T} \approx 1 + x_i/T + \frac{1}{2}(x_i/T)^2 + \cdots$. To first order:
$$\text{softmax}(x/T)_i \approx \frac{1 + x_i/T}{\sum_j (1 + x_j/T)} = \frac{1 + x_i/T}{n + \bar{x}/T}$$

As $T \to \infty$, this $\to 1/n$ (uniform). The correction term $x_i/T$ shows that even at high temperature, the softmax is not exactly uniform — it favors larger $x_i$ by $O(1/T)$.


---

## 7. Key Series in Machine Learning

### 7.1 Softmax and Temperature Scaling

The softmax function with temperature $T > 0$:
$$\text{softmax}(\mathbf{x}/T)_i = \frac{e^{x_i/T}}{\sum_{j=1}^n e^{x_j/T}}$$

**Low temperature ($T \to 0$)**: Let $x_{\max} = \max_j x_j$. Factor out $e^{x_{\max}/T}$:
$$\text{softmax}(x/T)_i = \frac{e^{(x_i - x_{\max})/T}}{\sum_j e^{(x_j - x_{\max})/T}}$$

As $T \to 0$: terms with $x_j < x_{\max}$ have $e^{(x_j - x_{\max})/T} \to 0$; the $i^*$ term (the max) stays at 1. So $\text{softmax}(x/T) \to \mathbf{e}_{i^*}$ (one-hot on argmax). This is the temperature $\to 0$ limit used in **hard attention** and **Gumbel-Softmax**.

**High temperature ($T \to \infty$)**: Taylor expand $e^{x_i/T} = 1 + x_i/T + O(1/T^2)$:
$$\text{softmax}(x/T)_i \approx \frac{1 + x_i/T}{n + \bar{x}/T} \to \frac{1}{n}$$

The distribution becomes uniform. This is used in **knowledge distillation** (Hinton 2015): training the student on soft targets with high temperature captures the "dark knowledge" in the teacher's probability distribution.

**Series representation of the denominator**: For the standard softmax, $Z = \sum_j e^{x_j}$ is the partition function — a finite sum but in theory for continuous distributions it becomes a genuine integral or series.

### 7.2 GELU via Hermite Polynomials

The **GELU** (Gaussian Error Linear Unit) activation is:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

where $\Phi$ is the standard Gaussian CDF. This has no closed-form Taylor series at $x=0$ in elementary functions, but the approximation:
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right]\right)$$

uses a polynomial approximation inside $\tanh$. This is the approximation used in practice (GPT-2, BERT).

**Taylor series of erf**: Since $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}\,dt$, substitute $e^{-t^2} = \sum \frac{(-1)^n t^{2n}}{n!}$ and integrate term-by-term:
$$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{n!(2n+1)}$$

Near $x = 0$: $\text{erf}(x) \approx \frac{2x}{\sqrt\pi} - \frac{2x^3}{3\sqrt\pi} + \cdots$

So: $\text{GELU}(x) \approx x \cdot \frac{1}{2}\left(1 + \frac{2x}{\sqrt\pi}\right) = \frac{x}{2} + \frac{x^2}{\sqrt\pi}$ near $x = 0$.

**Connection to Hermite polynomials**: The derivatives of GELU at 0 involve Hermite polynomials $H_n$ — the orthogonal polynomials with respect to the Gaussian weight $e^{-x^2/2}$. This connection explains why GELU interacts well with Gaussian-distributed inputs.

### 7.3 Optimizer Update Rules as Taylor Approximations

**Newton's method** (second-order optimization) applies a quadratic Taylor approximation to the loss:
$$\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + \nabla\mathcal{L}^\top \delta + \frac{1}{2}\delta^\top H \delta$$

Minimizing over $\delta$: $\delta^* = -H^{-1}\nabla\mathcal{L}$ (Newton step). This is exact for quadratic losses.

**SGD** discards the Hessian ($H \approx I/\alpha$): $\delta = -\alpha \nabla\mathcal{L}$ — first-order Taylor only.

**Adam's bias correction via geometric series**:

The Adam first moment is $m_t = (1-\beta_1)\sum_{k=0}^{t-1} \beta_1^k g_{t-k}$. Its expectation:
$$\mathbb{E}[m_t] = (1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k \mathbb{E}[g_{t-k}] = (1-\beta_1^t)\mathbb{E}[g]$$

using the geometric series partial sum $\sum_{k=0}^{t-1}\beta_1^k = (1-\beta_1^t)/(1-\beta_1)$. The bias correction $\hat{m}_t = m_t/(1-\beta_1^t)$ restores $\mathbb{E}[\hat{m}_t] = \mathbb{E}[g]$.

**Gradient clipping via Taylor**: For very large gradients, the loss is approximated by its first-order Taylor, and clipping bounds the step size to stay within a trust region where the linear approximation is valid.

### 7.4 Linear Attention via Taylor Expansion of exp

Standard self-attention has complexity $O(n^2 d)$ because it computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

The key operation is $\exp(q_i^\top k_j / \sqrt{d})$ for all $i, j$ pairs.

**Linear attention (Katharopoulos et al., 2020)** replaces $\exp(\cdot)$ with its first-order Taylor approximation:
$$\exp(q^\top k/\sqrt{d}) \approx 1 + q^\top k/\sqrt{d} = \phi(q)^\top \phi(k)$$

where $\phi(x) = [1, x/\sqrt{d}]$ (feature map). Then:
$$\text{LinearAttn}(q_i) = \frac{\sum_j \phi(q_i)^\top\phi(k_j) v_j}{\sum_j \phi(q_i)^\top\phi(k_j)} = \frac{\phi(q_i)^\top \left(\sum_j \phi(k_j)v_j^\top\right)}{\phi(q_i)^\top \left(\sum_j \phi(k_j)\right)}$$

The sums $\sum_j \phi(k_j)v_j^\top$ and $\sum_j \phi(k_j)$ can be computed once in $O(nd)$, reducing complexity to $O(n)$.

**Error of first-order approximation**: By Taylor's theorem, $|e^x - (1+x)| \le \frac{x^2}{2}e^{|x|}$. For $x = q^\top k/\sqrt{d}$ small (as when $q, k$ have small norm), the error is $O(1/d)$ — small in high dimension.

**Higher-order approximations** use $e^x \approx 1 + x + x^2/2$, giving **Performer-style** random feature approximations via the polynomial kernel $1 + q^\top k + (q^\top k)^2/2$.

### 7.5 Log-Sum-Exp and Numerical Stability

The **LogSumExp** operation appears in softmax computation:
$$\text{LSE}(x_1, \ldots, x_n) = \ln\left(\sum_{i=1}^n e^{x_i}\right)$$

**Numerical problem**: For large $x_i$ (e.g., $x_i = 1000$), $e^{x_i}$ overflows float32 ($\max \approx 3.4 \times 10^{38}$).

**Stable computation**: Let $m = \max_i x_i$. Then:
$$\text{LSE}(x) = m + \ln\left(\sum_i e^{x_i - m}\right)$$

Each $e^{x_i - m} \le 1$, so no overflow.

**Taylor analysis**: Why does this work? Write $x_i = m + \delta_i$ where $\delta_i \le 0$:
$$\ln\sum e^{x_i} = m + \ln\left(1 + \sum_{i: x_i < m} e^{\delta_i}\right)$$

For the dominant term ($\delta_{i^*} = 0$, the max): contribution is $e^0 = 1$. Other terms are exponentially small: $e^{\delta_i} \ll 1$. The Taylor expansion $\ln(1 + \epsilon) \approx \epsilon - \epsilon^2/2 + \cdots$ shows the correction is small when all other $x_i$ are much smaller than $m$.

### 7.6 Laplace Approximation

**Setup**: Given a posterior $p(\theta|\mathcal{D}) \propto p(\mathcal{D}|\theta)p(\theta)$, we want a Gaussian approximation around the MAP estimate $\hat\theta$.

**Method**: Taylor-expand $\log p(\theta|\mathcal{D})$ around $\hat\theta$ (where gradient is zero):
$$\log p(\theta|\mathcal{D}) \approx \log p(\hat\theta|\mathcal{D}) - \frac{1}{2}(\theta - \hat\theta)^\top H (\theta - \hat\theta)$$

where $H = -\nabla^2 \log p(\hat\theta|\mathcal{D})$ (negative Hessian at MAP). The second-order Taylor expansion gives:
$$p(\theta|\mathcal{D}) \approx \mathcal{N}(\hat\theta,\, H^{-1})$$

This is the **Laplace approximation** — a Gaussian centered at MAP with covariance equal to the inverse of the negative log-posterior Hessian.

**Where Taylor truncation is valid**: The approximation is accurate when the log-posterior is nearly quadratic near $\hat\theta$, i.e., when the third-order remainder $O(|\theta - \hat\theta|^3)$ is small. This holds when the posterior is concentrated (many data points, by Bernstein-von Mises theorem).

**Applications in ML (2026)**:
- **Bayesian neural networks**: Laplace approximation used in Laplace-Redux (Immer et al., 2021), PostHoc Laplace (Daxberger et al., 2021)
- **Uncertainty quantification**: covariance $H^{-1}$ estimates predictive uncertainty
- **Last-layer Laplace**: apply Laplace approximation only to the last linear layer (computationally tractable for large models)


---

## 8. Fourier Series (Preview)

> **Scope note**: The full theory of Fourier series — convergence theorems, Parseval's identity, Fourier transform, and their ML applications — belongs in a dedicated treatment. Here we give the essential concepts needed to understand positional encodings and frequency decomposition in transformers.

### 8.1 Periodic Functions as Trigonometric Series

A **Fourier series** represents a periodic function $f$ with period $2\pi$ as an infinite sum of sines and cosines:
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \left(a_n \cos(nx) + b_n \sin(nx)\right)$$

Unlike Taylor series (which expand around a single point using polynomials), Fourier series use **global basis functions** — sine and cosine waves of different frequencies — to represent the function over its entire period.

**The key difference from Taylor series**:

| Feature | Taylor Series | Fourier Series |
|---------|--------------|----------------|
| Basis functions | $1, x, x^2, x^3, \ldots$ (powers) | $1, \cos x, \sin x, \cos 2x, \ldots$ (sinusoids) |
| Expansion point | Local (around $a$) | Global (over the full period) |
| Best for | Smooth, analytic functions | Periodic functions, possibly discontinuous |
| Convergence | In a neighborhood of $a$ | In $L^2$ norm over the period |
| Derivatives | Easy (power rule) | Easy (differentiation rotates sin/cos) |

**Key property**: The derivative of a Fourier series is obtained by differentiating term-by-term:
$$f'(x) = \sum_{n=1}^\infty n(-a_n \sin(nx) + b_n \cos(nx))$$

Differentiation multiplies each frequency-$n$ component by $n$ — high-frequency components dominate the derivative.

### 8.2 Orthogonality and Fourier Coefficients

The key property that makes Fourier series work is **orthogonality**: different basis functions are orthogonal with respect to the inner product $\langle f, g\rangle = \frac{1}{\pi}\int_{-\pi}^\pi f(x)g(x)\,dx$:

$$\frac{1}{\pi}\int_{-\pi}^\pi \cos(mx)\cos(nx)\,dx = \begin{cases} 0 & m \ne n \\ 1 & m = n \ne 0 \\ 2 & m = n = 0 \end{cases}$$

$$\frac{1}{\pi}\int_{-\pi}^\pi \sin(mx)\sin(nx)\,dx = \begin{cases} 0 & m \ne n \\ 1 & m = n \end{cases}$$

$$\frac{1}{\pi}\int_{-\pi}^\pi \cos(mx)\sin(nx)\,dx = 0 \quad \text{for all } m, n$$

These orthogonality relations are provable using the integration formulas for products of sines and cosines (from §03-Integration).

**Deriving the Fourier coefficients**: Multiply both sides of the Fourier series by $\cos(nx)$ and integrate over $[-\pi, \pi]$. All terms vanish by orthogonality except the $a_n$ term:
$$a_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\cos(nx)\,dx, \qquad b_n = \frac{1}{\pi}\int_{-\pi}^\pi f(x)\sin(nx)\,dx$$

These are exactly the **projection coefficients** of $f$ onto the orthogonal basis $\{\cos(nx), \sin(nx)\}$ — the Fourier series is the best $L^2$ approximation of $f$ in the sinusoidal basis.

**Complex form**: Using Euler's formula $e^{inx} = \cos(nx) + i\sin(nx)$:
$$f(x) = \sum_{n=-\infty}^\infty \hat{f}(n)\,e^{inx}, \qquad \hat{f}(n) = \frac{1}{2\pi}\int_{-\pi}^\pi f(x)e^{-inx}\,dx$$

The complex coefficients $\hat{f}(n)$ encode both amplitude and phase of the $n$-th frequency component.

### 8.3 ML Relevance: Positional Encodings and RoPE

**Original Transformer positional encodings** (Vaswani et al., 2017) use sinusoidal functions evaluated at position $p$ and dimension $d$:
$$PE_{p,2k} = \sin\left(\frac{p}{10000^{2k/d}}\right), \qquad PE_{p,2k+1} = \cos\left(\frac{p}{10000^{2k/d}}\right)$$

This is a Fourier-like encoding: position $p$ is represented by a superposition of sinusoids at $d/2$ different frequencies $\omega_k = 1/10000^{2k/d}$.

**Why sinusoidal?** Any relative shift in position corresponds to a linear transformation of the encoding vector — provable because the Fourier basis diagonalizes the shift operator. This allows the attention mechanism to learn relative positions.

**Rotary Position Embeddings (RoPE)** (Su et al., 2021, used in LLaMA, GPT-NeoX) rotate the query and key vectors in the complex plane by angle $m\theta_k$ at position $m$:
$$q_m = q \cdot e^{im\theta}, \qquad k_n = k \cdot e^{in\theta}$$

The dot product $q_m^\top k_n = \text{Re}(q e^{im\theta} \cdot (k e^{in\theta})^*) = q^\top k \cos((m-n)\theta)$ depends only on $m-n$ (relative position). This is the Fourier multiplication theorem: convolution in position space ↔ pointwise multiplication in frequency space.

> _Full Fourier theory: Fourier transform, Parseval's theorem, Gibbs phenomenon, and non-periodic functions are treated in the Probability chapter (characteristic functions) and an advanced analysis appendix._


---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Concluding convergence from $a_n \to 0$ | Necessary but not sufficient: harmonic series has $1/n \to 0$ but diverges | Use a proper convergence test (ratio, comparison, integral) |
| 2 | Using ratio test when ratio $= 1$ | L = 1 is the inconclusive case: works for p-series, where the ratio always → 1 | Switch to integral test or comparison test |
| 3 | Forgetting to check endpoints separately | Ratio/root tests are strict inequalities: they say nothing at $|x-a| = R$ | Always test each endpoint using other convergence tests |
| 4 | Rearranging conditionally convergent series | Riemann's theorem: rearrangement can change the sum to any value | Only rearrange absolutely convergent series freely |
| 5 | Applying a Taylor series outside its radius of convergence | The series may diverge or converge to the wrong value | Check $|x - a| < R$ before using the series approximation |
| 6 | Confusing the Taylor series existing with converging to $f$ | A smooth function may have a Taylor series that doesn't equal $f$ (e.g., $e^{-1/x^2}$) | Check analyticity; use Lagrange remainder to bound error |
| 7 | Differentiating a power series and changing the radius of convergence | Radius is preserved under differentiation/integration — endpoints can change | Re-check endpoints only; radius stays the same |
| 8 | Treating $\sum (-1)^n/n$ as the negative of $\sum 1/n$ | The alternating series converges to $-\ln 2$; the non-alternating diverges entirely | Absolute and conditional convergence are fundamentally different |
| 9 | Applying the integral test to non-monotone functions | The test requires $f$ to be eventually decreasing and positive | Verify monotonicity; if $f$ is not monotone, use comparison instead |
| 10 | Assuming $\sum a_n$ and $\sum b_n$ both diverge implies $\sum (a_n + b_n)$ diverges | Both can diverge while their sum converges: $a_n = 1/n$, $b_n = -1/n$ | Convergence of sums is NOT additive for divergent series |
| 11 | Using $e^x \approx 1 + x$ for $x$ not small | First-order Taylor has error $O(x^2)$: for $x = 1$, error is $\approx e - 2 \approx 0.72$ | Check $|x| \ll 1$; use more terms or the exact function if $x$ is large |
| 12 | Confusing $R$ (radius of convergence) with the interval of convergence | $R$ is the radius; the interval is $(a-R, a+R)$, and endpoints need separate analysis | Always state both $R$ and the interval (with endpoints explicitly checked) |

---

## 10. Exercises

**Exercise 1** ★ — Geometric Series
Compute the sum of each geometric series or state that it diverges:
(a) $\sum_{n=0}^\infty \left(\frac{2}{3}\right)^n$
(b) $\sum_{n=2}^\infty \left(\frac{3}{2}\right)^n$
(c) $\sum_{n=0}^\infty \frac{(-1)^n}{4^n}$
(d) In reinforcement learning, the discounted return is $G_0 = \sum_{t=0}^\infty \gamma^t r$ where $r = 1$ (constant reward) and $\gamma = 0.95$. Compute $G_0$.

**Exercise 2** ★ — Convergence Tests
Determine whether each series converges or diverges. State which test you use:
(a) $\sum_{n=1}^\infty \frac{1}{\sqrt{n^3}}$
(b) $\sum_{n=1}^\infty \frac{n}{e^n}$
(c) $\sum_{n=1}^\infty \frac{(-1)^n}{\sqrt{n}}$
(d) $\sum_{n=1}^\infty \frac{n!}{2^n}$
(e) $\sum_{n=2}^\infty \frac{1}{n \ln n}$

**Exercise 3** ★ — Radius of Convergence
Find the radius of convergence $R$ and the interval of convergence (including endpoint checks) for each power series:
(a) $\sum_{n=0}^\infty \frac{x^n}{3^n}$
(b) $\sum_{n=1}^\infty \frac{(-1)^n x^n}{n}$
(c) $\sum_{n=0}^\infty n! \cdot x^n$
(d) $\sum_{n=0}^\infty \frac{(x-2)^n}{n^2 + 1}$

**Exercise 4** ★★ — Taylor Series Derivation
(a) Derive the Maclaurin series for $f(x) = e^{-x}$ from scratch (compute coefficients via $c_n = f^{(n)}(0)/n!$).
(b) Use the series for $e^x$ to derive the series for $e^{x^2}$.
(c) Using integration of the series for $e^{-t^2}$ from 0 to $x$, express $\int_0^x e^{-t^2}\,dt$ as a power series.
(d) How many terms are needed to approximate $\int_0^{0.5} e^{-t^2}\,dt$ to within $10^{-8}$?

**Exercise 5** ★★ — Lagrange Remainder and Error Bounds
(a) Find the 4th-order Taylor polynomial for $f(x) = \ln(1+x)$ at $a = 0$.
(b) Use the Lagrange remainder to bound $|f(0.2) - T_4(0.2)|$.
(c) Compute the actual error and verify your bound holds.
(d) How many terms of the Taylor series for $\ln(1+x)$ are needed to compute $\ln(1.1)$ to 10 decimal places?

**Exercise 6** ★★ — GELU Taylor Analysis
(a) Show that $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{n!(2n+1)}$ by integrating $e^{-t^2}$ term-by-term.
(b) Write out the first 4 terms of the GELU series: $\text{GELU}(x) = x \cdot \frac{1}{2}(1 + \text{erf}(x/\sqrt{2}))$.
(c) Compare the GELU Taylor approximation to $T_3(x) = x$ (ReLU-like) and $T_5(x)$ for $x \in [-2, 2]$.
(d) At what $|x|$ does the 5th-order approximation exceed 1% relative error?

**Exercise 7** ★★★ — Linear Attention Approximation
(a) Show that $e^{q^\top k} \approx 1 + q^\top k$ to first order, and bound the error $|e^{q^\top k} - (1 + q^\top k)|$ in terms of $|q^\top k|$.
(b) If queries and keys are unit vectors in $\mathbb{R}^d$ (so $|q^\top k| \le 1$), bound the maximum first-order approximation error.
(c) Write the standard attention formula and show how the linear approximation reduces complexity from $O(n^2 d)$ to $O(n d^2)$.
(d) Would using the second-order approximation $e^x \approx 1 + x + x^2/2$ improve accuracy? What is the trade-off?

**Exercise 8** ★★★ — Geometric Series in Adam
(a) Show that the Adam first moment $m_t = (1-\beta)\sum_{k=0}^{t-1}\beta^k g_{t-k}$ (for $m_0 = 0$) satisfies the recurrence $m_t = \beta m_{t-1} + (1-\beta) g_t$.
(b) Compute $\mathbb{E}[m_t]$ assuming all gradients have the same mean $\mu$ (i.e., $\mathbb{E}[g_k] = \mu$). Show it equals $(1-\beta^t)\mu$.
(c) Why is the bias correction $\hat{m}_t = m_t/(1-\beta^t)$ needed in early training ($t$ small)?
(d) For $\beta = 0.9$, at what step $t$ is the bias correction less than 1% (i.e., $\beta^t < 0.01$)?

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Application | Where in Modern Systems |
|---------|------------------|------------------------|
| **Geometric series** | Discounted return in RL; Adam bias correction | PPO, SAC, A3C; all Adam-family optimizers |
| **Convergence tests** | Proving SGD convergence; Robbins-Monro conditions | Convergence proofs in NeurIPS/ICML papers |
| **Taylor series (1st order)** | Linear attention; Performer approximation | Linformer, Performer, FNet |
| **Taylor series (2nd order)** | Newton-CG optimizer; K-FAC; Gauss-Newton | Natural gradient methods, second-order optimizers |
| **Lagrange remainder** | Quantifying approximation error in linearized attention | Guarantees in approximation papers |
| **Power series / radius of convergence** | Convergence radius of optimizer iterates; spectral radius | Convergence theory for RNNs, ResNets |
| **Maclaurin series for $e^x$** | Deriving the exponential map on Lie groups (rotation matrices in equivariant networks) | SE(3)-equivariant networks, FrameDiff |
| **Log-sum-exp stability** | Numerically stable softmax in all transformer implementations | PyTorch, JAX — used in every attention block |
| **Fourier series / sinusoidal PE** | Position encoding in original Transformer; period-length encoding | Vaswani et al. 2017; long-context models |
| **RoPE (Rotary PE)** | Relative position encoding in LLaMA, GPT-NeoX, Falcon | Most open-source LLMs ≥ 2023 |
| **Laplace approximation** | Bayesian uncertainty quantification for LLMs | Laplace-Redux; PostHoc Laplace for safety |
| **Alternating series (MCMC)** | Controlling numerical error in annealed importance sampling | Bayesian inference; diffusion model likelihoods |
| **$e^{i\theta} = \cos\theta + i\sin\theta$** | Complex rotations in RoPE; frequency analysis of loss landscapes | Mechanistic interpretability of sinusoidal features |
| **Binomial series $(1+x)^\alpha$** | Legendre polynomials; polynomial feature maps in kernels | Polynomial attention, spherical harmonics |

---

## Conceptual Bridge

**Where you came from**: This section is the culmination of single-variable calculus. Limits (§04/01) are the foundation — sequence convergence IS a limit. Derivatives (§04/02) provide the Taylor coefficients $f^{(n)}(a)/n!$. Integration (§04/03) allows deriving new power series from old ones via term-by-term integration. Every idea in this section rests on the three preceding sections.

**What this unlocks forward**: Series are the bridge from single-variable to infinite-dimensional mathematics:

- **→ §05 Multivariate Calculus**: The multivariable Taylor theorem uses the same structure: $f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x}) + \nabla f^\top \mathbf{h} + \frac{1}{2}\mathbf{h}^\top H \mathbf{h} + \cdots$ where $H$ is the Hessian matrix. Every gradient descent convergence proof uses the second-order Taylor expansion.

- **→ §06 Probability and Statistics**: Generating functions $G(x) = \sum p_n x^n$ encode probability distributions as power series. Characteristic functions $\phi(t) = \mathbb{E}[e^{itX}]$ are Fourier transforms of probability densities. The cumulant generating function $K(t) = \log\mathbb{E}[e^{tX}]$ has Taylor coefficients equal to the cumulants (mean, variance, skewness, kurtosis).

- **→ Functional Analysis** (advanced): Functions in $L^2[a,b]$ can be expanded in any orthonormal basis (Legendre polynomials, Fourier series, wavelets). This is the infinite-dimensional generalization of expressing a vector in an orthonormal basis. Neural networks implicitly learn such basis decompositions.

```
POSITION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §04/01 Limits ──→ ε-N convergence used in §04/04 sequence theory
  §04/02 Derivatives ──→ Taylor coefficients f⁽ⁿ⁾(a)/n! in §04/04
  §04/03 Integration ──→ term-by-term integration; integral test
         ↓
  §04/04 SERIES AND SEQUENCES  ← YOU ARE HERE
         │
         ├──→ §05 Multivariable Taylor theorem
         │         (gradient descent convergence)
         │
         ├──→ §06 Probability: generating functions,
         │         characteristic functions, CLT proof
         │
         └──→ Advanced: Fourier analysis, functional
                   analysis, spectral methods in ML

════════════════════════════════════════════════════════════════════════
```

The single most important takeaway: **Taylor series and the geometric series are not just abstract tools — they are the mathematical engine behind every major optimization algorithm and attention approximation in modern AI.** Every time a researcher writes $e^x \approx 1 + x$, they are using the first-order Maclaurin approximation with a remainder bounded by the Lagrange error formula.


---

## Appendix A: Extended Convergence Proofs

### A.1 Proof of the Integral Test

**Theorem**: Let $f: [1, \infty) \to [0,\infty)$ be continuous and decreasing, with $a_n = f(n)$. Then $\sum_{n=1}^\infty a_n$ converges iff $\int_1^\infty f(x)\,dx < \infty$.

**Proof**: Since $f$ is decreasing, for $x \in [n, n+1]$:
$$f(n) \ge f(x) \ge f(n+1)$$

Integrating over $[n, n+1]$:
$$f(n) \ge \int_n^{n+1} f(x)\,dx \ge f(n+1)$$

Sum from $n = 1$ to $N$:
$$\sum_{n=1}^N f(n) \ge \int_1^{N+1} f(x)\,dx \ge \sum_{n=2}^{N+1} f(n) = \sum_{n=1}^N f(n+1)$$

**If $\int_1^\infty f\,dx < \infty$**: The partial sums $S_N = \sum_{n=1}^N a_n$ are bounded above by $a_1 + \int_1^\infty f\,dx < \infty$. Since $S_N$ is increasing (terms are positive) and bounded above, $S_N$ converges by MCT.

**If $\sum a_n < \infty$**: Then $\int_1^{N+1} f\,dx \le \sum_{n=1}^N f(n) \to \sum_{n=1}^\infty a_n < \infty$. Since $\int_1^{N+1} f\,dx$ is increasing and bounded, it converges. $\square$

### A.2 Proof of the Ratio Test

**Theorem**: If $L = \lim |a_{n+1}/a_n| < 1$, then $\sum a_n$ converges absolutely.

**Proof**: Choose $r$ with $L < r < 1$. Since $|a_{n+1}/a_n| \to L$, there exists $N$ such that $|a_{n+1}/a_n| \le r$ for all $n \ge N$. Then for $n \ge N$:
$$|a_{n+1}| \le r|a_n| \le r^2|a_{N-1}| \le \cdots \le r^{n-N+1}|a_N|$$

So $|a_{n}| \le |a_N| \cdot r^{-N} \cdot r^n = C r^n$ for $C = |a_N| r^{-N}$. Therefore:
$$\sum_{n=N}^\infty |a_n| \le C \sum_{n=N}^\infty r^n = \frac{Cr^N}{1-r} < \infty$$

Since the tail converges, so does the full series. $\square$

**If $L > 1$**: Then $|a_{n+1}/a_n| > 1$ eventually, so $|a_n| \to \infty$, violating the necessary condition $a_n \to 0$. Series diverges. $\square$

### A.3 Proof of the Alternating Series Test (Leibniz)

**Theorem**: If $b_n \ge 0$, $b_{n+1} \le b_n$, and $b_n \to 0$, then $\sum (-1)^{n+1}b_n$ converges.

**Proof**: Consider even and odd partial sums:
$$S_{2m} = (b_1 - b_2) + (b_3 - b_4) + \cdots + (b_{2m-1} - b_{2m})$$

Each parenthesis $b_{2k-1} - b_{2k} \ge 0$ (since $b_n$ is decreasing), so $S_{2m}$ is increasing.

Also: $S_{2m} = b_1 - (b_2 - b_3) - (b_4 - b_5) - \cdots - b_{2m} \le b_1$. So $S_{2m}$ is bounded above.

By MCT, $S_{2m} \to L$ for some $L \le b_1$.

Now $S_{2m+1} = S_{2m} + b_{2m+1} \to L + 0 = L$ (since $b_n \to 0$).

Both subsequences of partial sums converge to $L$, so $S_N \to L$. $\square$

**Error bound proof**: Note $S_{2m} \le L \le S_{2m+1}$ (odd sums overshoot from above). So:
$$|L - S_N| \le |S_{N+1} - S_N| = b_{N+1} \qquad \square$$


---

## Appendix B: Uniform Convergence

### B.1 Pointwise vs. Uniform Convergence

**Definition (Pointwise convergence)**: A sequence of functions $f_n: D \to \mathbb{R}$ converges **pointwise** to $f$ if for each fixed $x \in D$:
$$\lim_{n\to\infty} f_n(x) = f(x)$$

**Definition (Uniform convergence)**: $f_n$ converges **uniformly** to $f$ on $D$ if:
$$\lim_{n\to\infty} \sup_{x \in D} |f_n(x) - f(x)| = 0$$

Uniform convergence is stronger: the same $N$ works for ALL $x$ simultaneously.

**The critical distinction**:

```
POINTWISE vs. UNIFORM CONVERGENCE
════════════════════════════════════════════════════════════════════════

  fₙ(x) = xⁿ on [0,1]:
  ─────────────────────────────────────────────────────────────────
  At x = 0.5: (0.5)ⁿ → 0  (converges)
  At x = 1.0: 1ⁿ = 1 → 1  (converges)
  Pointwise limit: f(x) = 0 for x ∈ [0,1), f(1) = 1

  But: sup_{x∈[0,1]} |xⁿ - f(x)| = sup_{x∈[0,1)} xⁿ = 1 for all n
  → NOT uniformly convergent on [0,1]

  On [0, 0.9]: sup xⁿ = 0.9ⁿ → 0 → uniformly convergent here.

════════════════════════════════════════════════════════════════════════
```

### B.2 Why Uniform Convergence Enables Term-by-Term Operations

**Theorem**: If $f_n \to f$ uniformly on $[a,b]$ and each $f_n$ is continuous:
1. $f$ is continuous
2. $\lim_{n\to\infty} \int_a^b f_n(x)\,dx = \int_a^b f(x)\,dx$ (limit and integral interchange)

**Theorem**: If $f_n \to f$ pointwise and $f_n' \to g$ uniformly, then $f' = g$ (limit and derivative interchange).

These theorems are exactly what justify **term-by-term differentiation and integration of power series** inside the radius of convergence: the partial sums $\sum_{k=0}^n c_k(x-a)^k$ converge uniformly on any closed interval $[a-r, a+r]$ with $r < R$.

### B.3 Failure Without Uniform Convergence

Without uniform convergence, swapping limits and integrals leads to errors:

**Example**: $f_n(x) = n x e^{-nx}$ on $[0,1]$.
- Pointwise: $f_n(x) \to 0$ for all $x \ge 0$ (compute via L'Hôpital or squeeze)
- But: $\int_0^1 f_n(x)\,dx = 1 - e^{-n} \to 1 \ne \int_0^1 0\,dx = 0$

The limit and integral don't commute! Convergence is NOT uniform ($\sup_x f_n(x) = 1/e$ for all $n$).

**For ML**: This is why convergence arguments for neural network training must be careful about the order of limits. The Dominated Convergence Theorem (from integration theory, §03/Appendix O) provides the right condition under which limits and integrals commute.


---

## Appendix C: Complex Power Series and Euler's Formula

### C.1 Extending Power Series to Complex Numbers

Power series work equally well for complex $z \in \mathbb{C}$. The exponential series $e^z = \sum z^n/n!$ converges for all $z \in \mathbb{C}$ (same ratio test argument).

**Euler's formula**: Setting $z = i\theta$ (purely imaginary):
$$e^{i\theta} = \sum_{n=0}^\infty \frac{(i\theta)^n}{n!} = \sum_{n=0}^\infty \frac{i^n \theta^n}{n!}$$

Using $i^0=1$, $i^1=i$, $i^2=-1$, $i^3=-i$, $i^4=1$, $\ldots$ (period 4):
$$e^{i\theta} = \left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right) + i\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right) = \cos\theta + i\sin\theta$$

**Euler's identity** ($\theta = \pi$): $e^{i\pi} = \cos\pi + i\sin\pi = -1 + 0i = -1$, giving $e^{i\pi} + 1 = 0$.

This remarkable identity connects the five most fundamental constants of mathematics: $e$, $i$, $\pi$, $1$, $0$.

### C.2 Rotations as Complex Exponentials

Multiplying by $e^{i\theta}$ rotates a complex number by angle $\theta$:
$$e^{i\theta} \cdot re^{i\phi} = r e^{i(\theta+\phi)}$$

This is the mathematical foundation of **RoPE (Rotary Position Embeddings)**:

In RoPE, each attention head dimension pair $(q_{2k}, q_{2k+1})$ is treated as a complex number $q_{2k} + i q_{2k+1}$. Multiplying by $e^{im\theta_k}$ (rotation by position $m$ times frequency $\theta_k$) encodes position:
$$\tilde{q}_m = q \cdot e^{im\theta} \implies \tilde{q}_m^\top \tilde{k}_n = \text{Re}[q\bar{k} e^{i(m-n)\theta}]$$

The inner product depends only on the relative position $m-n$ — the rotation encodes position via complex multiplication, making it equivariant to sequence shifts.

### C.3 Hyperbolic Functions

The hyperbolic functions emerge from the real and imaginary parts of $e^x$:
$$\cosh x = \frac{e^x + e^{-x}}{2} = \sum_{n=0}^\infty \frac{x^{2n}}{(2n)!}, \qquad \sinh x = \frac{e^x - e^{-x}}{2} = \sum_{n=0}^\infty \frac{x^{2n+1}}{(2n+1)!}$$

The **tanh** activation function: $\tanh(x) = \sinh x/\cosh x$. Its Taylor series at 0:
$$\tanh x = x - \frac{x^3}{3} + \frac{2x^5}{15} - \cdots$$

This shows tanh is approximately linear ($\approx x$) for small $x$ and saturates to $\pm 1$ for large $x$.


---

## Appendix D: Generating Functions

### D.1 Ordinary Generating Functions

A **generating function** packages a sequence $(a_0, a_1, a_2, \ldots)$ as a formal power series:
$$G(x) = \sum_{n=0}^\infty a_n x^n$$

The sequence $(a_n)$ can be recovered as $a_n = G^{(n)}(0)/n!$ (or via coefficient extraction). The power of generating functions is that **algebraic operations on $G(x)$ correspond to combinatorial operations on the sequence**.

**Key examples**:

| Sequence | Generating function | Notes |
|----------|--------------------|----|
| $1, 1, 1, \ldots$ | $\frac{1}{1-x}$ | Geometric series |
| $\binom{n}{k}$ (fixed $k$) | $\frac{x^k}{(1-x)^{k+1}}$ | Binomial coefficients |
| Fibonacci: $0,1,1,2,3,5,\ldots$ | $\frac{x}{1-x-x^2}$ | Closed form via partial fractions |
| $n^2$ | $\frac{x(1+x)}{(1-x)^3}$ | Derived by differentiating geometric series |

**Operations**:
- Shift: if $G(x) = \sum a_n x^n$, then $xG(x) = \sum a_n x^{n+1}$ (sequence shifted right by 1)
- Differentiation: $G'(x) = \sum na_n x^{n-1}$ (sequence $(na_n)$)
- Multiplication: $G(x)H(x) = \sum_n \left(\sum_{k=0}^n a_k b_{n-k}\right)x^n$ (convolution)

### D.2 Moment Generating Functions (MGF) in ML

The **moment generating function** of a random variable $X$:
$$M_X(t) = \mathbb{E}[e^{tX}] = \sum_{n=0}^\infty \frac{\mathbb{E}[X^n]}{n!} t^n$$

The Taylor coefficients encode the moments: $M_X^{(n)}(0) = \mathbb{E}[X^n]$.

**For Gaussian $X \sim \mathcal{N}(\mu, \sigma^2)$**: $M_X(t) = e^{\mu t + \sigma^2 t^2/2}$. This gives:
- $M'(0) = \mu$ (mean)
- $M''(0) = \mu^2 + \sigma^2$ (second moment)
- Variance: $M''(0) - (M'(0))^2 = \sigma^2$ ✓

**Cumulant generating function**: $K_X(t) = \log M_X(t) = \mu t + \frac{\sigma^2}{2}t^2$ for Gaussian. Taylor coefficients of $K_X$ are the **cumulants**: $\kappa_1 = \mu$ (mean), $\kappa_2 = \sigma^2$ (variance), $\kappa_3$ (skewness), $\kappa_4$ (excess kurtosis). For Gaussian, $\kappa_n = 0$ for $n \ge 3$ — a Gaussian is completely characterized by its first two cumulants. This is why many normality tests check higher cumulants.


---

## Appendix E: Multivariable Taylor (Preview)

> **Forward reference**: Full treatment in [§05 Multivariate Calculus](../../05-Multivariate-Calculus/README.md).

The Taylor series extends naturally to functions $f: \mathbb{R}^n \to \mathbb{R}$. Expanding around $\mathbf{a}$:

$$f(\mathbf{x}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^\top(\mathbf{x}-\mathbf{a}) + \frac{1}{2}(\mathbf{x}-\mathbf{a})^\top H(\mathbf{a})(\mathbf{x}-\mathbf{a}) + O(\|\mathbf{x}-\mathbf{a}\|^3)$$

where $\nabla f(\mathbf{a})$ is the gradient vector and $H(\mathbf{a})$ is the Hessian matrix.

**Reading the terms**:
- **0th order**: $f(\mathbf{a})$ — the function value at the expansion point
- **1st order**: $\nabla f^\top \delta$ — the linear (gradient) term; this is the basis of SGD
- **2nd order**: $\frac{1}{2}\delta^\top H\delta$ — the curvature term; this is why quadratic forms appear in optimization

**Gradient descent as first-order Taylor**: Setting $\mathbf{x} = \mathbf{a} - \alpha\nabla f(\mathbf{a})$ in the 1st-order expansion:
$$f(\mathbf{a} - \alpha\nabla f) \approx f(\mathbf{a}) - \alpha\|\nabla f(\mathbf{a})\|^2$$

The loss decreases by $\alpha\|\nabla f\|^2$ per step — valid when $\alpha$ is small enough that the linear approximation holds. The "right" $\alpha$ is determined by the second-order term.

**Newton's method as second-order Taylor**: Minimize the 2nd-order Taylor model exactly:
$$\delta^* = -H^{-1}\nabla f, \qquad f(\mathbf{a} + \delta^*) \approx f(\mathbf{a}) - \frac{1}{2}\nabla f^\top H^{-1}\nabla f$$

For a quadratic $f$, Newton's method is exact in one step. This is why L-BFGS and other quasi-Newton methods (which approximate $H^{-1}$) converge much faster than SGD on smooth losses.


---

## Appendix F: Fourier Series Extended

### F.1 Parseval's Identity

For a function $f$ with Fourier series $f(x) = a_0/2 + \sum(a_n\cos nx + b_n\sin nx)$:
$$\frac{1}{\pi}\int_{-\pi}^\pi |f(x)|^2\,dx = \frac{a_0^2}{2} + \sum_{n=1}^\infty (a_n^2 + b_n^2)$$

**Interpretation**: The $L^2$ norm squared of $f$ equals the sum of squared Fourier coefficients. This is an infinite-dimensional **Pythagorean theorem** — the orthonormal basis $\{1/\sqrt{2}, \cos nx, \sin nx\}$ satisfies the same relationship as a standard orthonormal basis in $\mathbb{R}^n$.

**For ML**: In the theory of neural networks as function approximators, the Fourier perspective shows that learning high-frequency functions requires exponentially more training data (the "spectral bias" of neural networks — they learn low frequencies first).

### F.2 Gibbs Phenomenon

For a discontinuous function (like a square wave), the Fourier partial sum $S_N(x)$ overshoots near the discontinuity by approximately $8.9\% = \frac{1}{\pi}\int_0^\pi \frac{\sin t}{t}\,dt - 1$ of the jump, regardless of $N$.

This **Gibbs phenomenon** does not disappear as $N \to \infty$ — it is an intrinsic property of Fourier series near discontinuities. The partial sum does converge to the function's average at the discontinuity.

**For ML**: Neural networks trained on discontinuous target functions (e.g., step functions, indicator functions) exhibit analogous ringing artifacts near edges. Regularization and smooth activations mitigate this.

### F.3 Convergence in $L^2$ vs. Pointwise

**Theorem (Riesz-Fischer)**: Every square-integrable function $f \in L^2[-\pi,\pi]$ equals the limit of its Fourier partial sums in $L^2$ norm:
$$\left\|f - S_N\right\|_{L^2} = \left(\int_{-\pi}^\pi |f(x) - S_N(x)|^2\,dx\right)^{1/2} \to 0$$

However, pointwise convergence is more subtle — Carleson's theorem (1966) states that the Fourier series of any $L^2$ function converges pointwise almost everywhere, but this is a very deep result.

**The $L^2$ convergence** is sufficient for ML applications — it means the Fourier approximation is the best $L^2$ approximation of the function in the sinusoidal basis, analogous to how the truncated SVD is the best low-rank approximation (Eckart-Young theorem).


---

## Appendix G: ML Derivations in Full

### G.1 Full Derivation: Adam Bias Correction

The Adam update rule:
$$m_0 = 0, \qquad m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

Unrolling the recurrence:
$$m_t = (1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k g_{t-k} + \beta_1^t m_0 = (1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k g_{t-k}$$

(since $m_0 = 0$). Taking expectation with $\mathbb{E}[g_k] = g$ (true gradient, stationary assumption):
$$\mathbb{E}[m_t] = (1-\beta_1)\left(\sum_{k=0}^{t-1}\beta_1^k\right)g = (1-\beta_1) \cdot \frac{1-\beta_1^t}{1-\beta_1} \cdot g = (1-\beta_1^t) g$$

The geometric series sum $\sum_{k=0}^{t-1}\beta_1^k = \frac{1-\beta_1^t}{1-\beta_1}$ is the key step.

**Bias correction**: $\hat{m}_t = m_t/(1-\beta_1^t)$ ensures $\mathbb{E}[\hat{m}_t] = g$. For $\beta_1 = 0.9$:
- $t=1$: $1-\beta_1^t = 0.1$, correction factor = 10 (large!)
- $t=10$: $1-0.9^{10} \approx 0.651$, correction $\approx 1.54$
- $t=100$: $1-0.9^{100} \approx 1.0$, correction negligible

Similarly for the second moment $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ with correction $\hat{v}_t = v_t/(1-\beta_2^t)$.

### G.2 Full Derivation: Linear Attention Complexity

Standard attention for $n$ tokens and $d$-dimensional heads:
$$A = \text{softmax}(QK^\top/\sqrt{d}) \in \mathbb{R}^{n\times n}, \quad \text{Output} = AV$$

Cost: $QK^\top$ is $O(n^2 d)$; applying $A$ to $V$ is $O(n^2 d)$. Total: $O(n^2 d)$.

Linear attention with $\phi(x) = [1, x]$ (first-order Taylor):
$$\text{LinearAttn}(q_i) = \frac{\sum_j \phi(q_i)^\top\phi(k_j)v_j}{\sum_j \phi(q_i)^\top\phi(k_j)}$$

The key insight: define $S = \sum_j \phi(k_j)v_j^\top \in \mathbb{R}^{2d \times d}$ and $z = \sum_j \phi(k_j) \in \mathbb{R}^{2d}$. These are computed ONCE in $O(nd^2)$. Then for each query:
$$\text{output}_i = \frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}$$

Cost per query: $O(d^2)$. Total: $O(nd^2)$. For fixed $d$, this is $O(n)$ vs. $O(n^2)$.

**Trade-off**: The first-order Taylor approximation introduces error $O((q^\top k)^2/2)$. For unit vectors in $\mathbb{R}^d$: $|q^\top k| \le 1$, error $\le 0.5$. In practice, the error is managed by choosing $d$ large and using random feature approximations (Performer).

### G.3 Full Derivation: Softmax Temperature and Entropy

For logits $\mathbf{x} \in \mathbb{R}^n$ and temperature $T > 0$, let $p_i = e^{x_i/T}/\sum_j e^{x_j/T}$.

**Entropy** $H = -\sum_i p_i \log p_i$:

*As $T \to 0$*: $p \to \mathbf{e}_{i^*}$ (one-hot), $H \to 0$ (minimum entropy).

*As $T \to \infty$*: $p \to (1/n, \ldots, 1/n)$, $H \to \log n$ (maximum entropy). Proof via Taylor:
$$p_i \approx \frac{1 + x_i/T}{n + \bar{x}/T} \approx \frac{1}{n}\left(1 + \frac{x_i - \bar{x}}{T}\right)$$

$$H \approx -\sum_i \frac{1}{n}\left(1 + \frac{x_i-\bar{x}}{T}\right)\log\left[\frac{1}{n}\left(1 + \frac{x_i-\bar{x}}{T}\right)\right]$$

$$\approx \log n - \frac{1}{n}\sum_i \frac{x_i-\bar{x}}{T} \cdot \frac{1}{1/(n)} + O(1/T^2) \to \log n$$

So entropy increases with temperature, controlled by the $O(1/T)$ correction term.


---

## Appendix H: Numerical Summation Methods

### H.1 Euler-Maclaurin Formula

The Euler-Maclaurin formula connects sums to integrals with correction terms:
$$\sum_{n=a}^b f(n) = \int_a^b f(x)\,dx + \frac{f(a)+f(b)}{2} + \sum_{k=1}^p \frac{B_{2k}}{(2k)!}\left[f^{(2k-1)}(b) - f^{(2k-1)}(a)\right] + R_p$$

where $B_{2k}$ are Bernoulli numbers ($B_2 = 1/6$, $B_4 = -1/30$, $B_6 = 1/42$, $\ldots$).

**Application**: Summing $\sum_{n=1}^N 1/n^2$ (Basel-like): the formula gives the exact asymptotic expansion. This is how the exact value $\pi^2/6$ was computed.

**For ML**: Euler-Maclaurin underlies the **Euler method for ODEs** (Neural ODEs), and is the bridge between discrete (sequence) and continuous (integral) formulations of the same problem.

### H.2 Kahan Summation Algorithm

Naive summation of floating-point numbers accumulates $O(n\epsilon)$ rounding error where $\epsilon$ is machine epsilon (~$10^{-7}$ for float32). **Kahan summation** reduces this to $O(\epsilon)$:

```python
def kahan_sum(arr):
    total = 0.0
    c = 0.0  # compensation
    for x in arr:
        y = x - c            # corrected term
        t = total + y        # new sum (may lose low bits)
        c = (t - total) - y  # lost low bits
        total = t
    return total
```

This is why `torch.sum()` and `numpy.sum()` use pairwise summation (not Kahan but better than naive) for numerical stability.

### H.3 Richardson Extrapolation

If $S(h)$ approximates $S$ with error $S(h) = S + c_1 h^p + c_2 h^{p+1} + \cdots$, then combining two approximations eliminates the leading error:
$$S \approx \frac{2^p S(h/2) - S(h)}{2^p - 1}$$

**Example**: Trapezoidal rule has error $O(h^2)$. Richardson extrapolation on two trapezoidal estimates gives Simpson's rule ($O(h^4)$). Repeated extrapolation gives **Romberg integration** with $O(h^{2k})$ accuracy.

For series, Richardson extrapolation can accelerate convergence of slowly converging sequences — a technique used in numerical analysis for computing $\pi$, $e$, and other constants.


---

## Appendix I: Glossary and Reference Tables

### I.1 Key Definitions

| Term | Definition |
|------|-----------|
| **Sequence** | Function $a: \mathbb{N} \to \mathbb{R}$; an ordered list $(a_1, a_2, \ldots)$ |
| **Converges (sequence)** | $\lim_{n\to\infty} a_n = L$ exists and is finite |
| **Bounded sequence** | $\exists B: |a_n| \le B$ for all $n$ |
| **Monotone sequence** | Either always increasing ($a_{n+1} \ge a_n$) or always decreasing |
| **Infinite series** | $\sum_{n=1}^\infty a_n = \lim_{N\to\infty} \sum_{n=1}^N a_n$ |
| **Absolutely convergent** | $\sum |a_n| < \infty$ |
| **Conditionally convergent** | $\sum a_n$ converges but $\sum |a_n| = \infty$ |
| **Power series** | $\sum c_n (x-a)^n$ — a series whose terms are polynomial monomials |
| **Radius of convergence** | Largest $R$ such that $\sum c_n x^n$ converges absolutely for $|x| < R$ |
| **Taylor series** | $\sum \frac{f^{(n)}(a)}{n!}(x-a)^n$ — power series coefficients from derivatives |
| **Maclaurin series** | Taylor series at $a = 0$ |
| **Analytic function** | Function that equals its Taylor series in a neighborhood of each point |
| **Uniform convergence** | $\sup_x |f_n(x) - f(x)| \to 0$ — convergence rate independent of $x$ |
| **Fourier series** | Expansion of a periodic function in the sinusoidal basis |
| **Lagrange remainder** | $R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$ — exact error in Taylor approximation |

### I.2 Standard Series Table

| Series | Sum | Domain |
|--------|-----|--------|
| $\sum_{n=0}^\infty x^n$ | $\frac{1}{1-x}$ | $|x| < 1$ |
| $\sum_{n=0}^\infty (-1)^n x^n$ | $\frac{1}{1+x}$ | $|x| < 1$ |
| $\sum_{n=0}^\infty \frac{x^n}{n!}$ | $e^x$ | All $x$ |
| $\sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{(2n+1)!}$ | $\sin x$ | All $x$ |
| $\sum_{n=0}^\infty \frac{(-1)^n x^{2n}}{(2n)!}$ | $\cos x$ | All $x$ |
| $\sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n}$ | $\ln(1+x)$ | $-1 < x \le 1$ |
| $\sum_{n=0}^\infty \binom{\alpha}{n}x^n$ | $(1+x)^\alpha$ | $|x| < 1$ |
| $\sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{2n+1}$ | $\arctan x$ | $|x| \le 1$ |
| $\sum_{n=0}^\infty \frac{x^{2n}}{(2n)!}$ | $\cosh x$ | All $x$ |
| $\sum_{n=0}^\infty \frac{x^{2n+1}}{(2n+1)!}$ | $\sinh x$ | All $x$ |
| $\sum_{n=1}^\infty \frac{1}{n^2}$ | $\frac{\pi^2}{6}$ | (Basel problem) |
| $\sum_{n=0}^\infty \frac{(-1)^n}{2n+1}$ | $\frac{\pi}{4}$ | (Leibniz) |

### I.3 Convergence Test Decision Table

| Test | Best For | Condition for Convergence | Condition for Divergence | Inconclusive |
|------|---------|--------------------------|-------------------------|--------------|
| Divergence | Any | — | $a_n \not\to 0$ | $a_n \to 0$ |
| Geometric | $\sum r^n$ | $|r| < 1$ | $|r| \ge 1$ | — |
| p-series | $\sum 1/n^p$ | $p > 1$ | $p \le 1$ | — |
| Integral | Decreasing positive $f(n)$ | $\int f\,dx < \infty$ | $\int f\,dx = \infty$ | $f$ not monotone |
| Comparison | $0 \le a_n \le b_n$ | $\sum b_n$ converges | $\sum a_n$ diverges | — |
| Limit Comparison | $a_n, b_n > 0$ | $L \in (0,\infty)$ → same | same | $L = 0$ or $\infty$ |
| Ratio | Factorials, exponentials | $L < 1$ | $L > 1$ | $L = 1$ |
| Root | $n$-th powers $(f(n))^n$ | $L < 1$ | $L > 1$ | $L = 1$ |
| Alternating | $\sum(-1)^n b_n$ | $b_n \ge 0$ decreasing, $\to 0$ | — | $b_n$ not decreasing |

### I.4 Self-Assessment Checklist

After completing this section:

**Sequences**
- [ ] Can give ε-N proof that $1/n \to 0$
- [ ] Can state and use Monotone Convergence Theorem
- [ ] Can explain Bolzano-Weierstrass and its ML relevance

**Series and Convergence Tests**
- [ ] Can prove harmonic series diverges (Oresme)
- [ ] Can apply all 7 convergence tests and know when each applies
- [ ] Can explain absolute vs. conditional convergence and Riemann rearrangement

**Power Series**
- [ ] Can find radius of convergence via ratio test
- [ ] Can differentiate and integrate power series term-by-term
- [ ] Can derive $\ln(1+x)$ series from geometric series via integration

**Taylor Series**
- [ ] Can derive Taylor series for $e^x$, $\sin x$, $\cos x$, $\ln(1+x)$
- [ ] Can bound Taylor approximation error via Lagrange remainder
- [ ] Can manipulate series algebraically (substitution, multiplication, composition)

**ML Applications**
- [ ] Can derive Adam bias correction using geometric series
- [ ] Can explain linear attention as first-order Taylor approximation
- [ ] Can explain softmax temperature behavior at $T \to 0$ and $T \to \infty$
- [ ] Can describe Laplace approximation as quadratic Taylor expansion


---

## Appendix J: Worked Solutions to Selected Exercises

### J.1 Exercise 1 — Geometric Series

**(a)** $\sum_{n=0}^\infty (2/3)^n$: geometric series with $a=1$, $r = 2/3$. Since $|r| = 2/3 < 1$:
$$\sum_{n=0}^\infty (2/3)^n = \frac{1}{1-2/3} = \frac{1}{1/3} = 3$$

**(b)** $\sum_{n=2}^\infty (3/2)^n$: ratio $r = 3/2 > 1$. Diverges.

**(c)** $\sum_{n=0}^\infty (-1)^n/4^n = \sum_{n=0}^\infty (-1/4)^n$: geometric with $r = -1/4$, $|r| = 1/4 < 1$:
$$\frac{1}{1-(-1/4)} = \frac{1}{5/4} = \frac{4}{5}$$

**(d)** $G_0 = \sum_{t=0}^\infty 0.95^t \cdot 1 = \frac{1}{1-0.95} = \frac{1}{0.05} = 20$.

### J.2 Exercise 3 — Radius of Convergence

**(a)** $\sum (x/3)^n$: geometric series with ratio $x/3$. Converges when $|x/3| < 1$, i.e., $|x| < 3$. So $R = 3$.
- At $x = 3$: $\sum 1^n = \sum 1$ diverges.
- At $x = -3$: $\sum (-1)^n$ diverges (oscillates).
- Interval of convergence: $(-3, 3)$.

**(b)** $\sum (-1)^n x^n / n$: ratio test: $|a_{n+1}/a_n| = \frac{n|x|}{n+1} \to |x|$. So $R = 1$.
- At $x = 1$: $\sum (-1)^n/n$ — alternating harmonic, converges to $-\ln 2$.
- At $x = -1$: $\sum (-1)^n(-1)^n/n = \sum 1/n$ — harmonic, diverges.
- Interval of convergence: $(-1, 1]$.

**(c)** $\sum n! x^n$: ratio $= (n+1)|x| \to \infty$ for any $x \ne 0$. So $R = 0$; converges only at $x = 0$.

**(d)** $\sum (x-2)^n/(n^2+1)$: ratio $\to |x-2|$. So $R = 1$, centered at $a = 2$.
- At $x = 3$ ($x-2 = 1$): $\sum 1/(n^2+1)$ converges (compare to p-series $\sum 1/n^2$).
- At $x = 1$ ($x-2 = -1$): $\sum (-1)^n/(n^2+1)$ converges absolutely.
- Interval of convergence: $[1, 3]$.

### J.3 Exercise 5 — Lagrange Remainder

**(a)** $\ln(1+x)$: derivatives at $a=0$:
- $f(x) = \ln(1+x)$, $f(0) = 0$
- $f'(x) = 1/(1+x)$, $f'(0) = 1$
- $f''(x) = -1/(1+x)^2$, $f''(0) = -1$
- $f'''(x) = 2/(1+x)^3$, $f'''(0) = 2$
- $f^{(4)}(x) = -6/(1+x)^4$, $f^{(4)}(0) = -6$

$$T_4(x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4}$$

**(b)** At $x = 0.2$: $f^{(5)}(x) = 24/(1+x)^5$. On $[0, 0.2]$: $|f^{(5)}| \le 24/(1+0)^5 = 24$. So:
$$|R_4(0.2)| \le \frac{24}{5!}(0.2)^5 = \frac{24}{120} \cdot 0.00032 = 0.2 \times 0.00032 = 6.4 \times 10^{-5}$$

**(c)** True: $\ln(1.2) \approx 0.182322$. $T_4(0.2) = 0.2 - 0.02 + 0.002667 - 0.0004 = 0.182267$. Error $\approx 5.5 \times 10^{-5}$ ✓ (less than our bound).

**(d)** Need $|R_n(0.1)| \le \frac{1}{n+1}(0.1)^{n+1} \le 10^{-10}$. For $n = 8$: $\frac{1}{9}(0.1)^9 \approx 1.1 \times 10^{-10}$. So **9 terms** suffice.


---

## Appendix K: Historical Deep Dives

### K.1 The Basel Problem (Euler, 1735)

The Basel problem asks: what is $\sum_{n=1}^\infty 1/n^2$?

This series was known to converge (p-series with $p=2$), but its exact value was unknown until Euler's stunning proof in 1735.

**Euler's argument** (not fully rigorous by modern standards, but correct):

The function $\sin(\pi x)/(\pi x)$ has zeros at $x = \pm 1, \pm 2, \pm 3, \ldots$ So it factors as:
$$\frac{\sin(\pi x)}{\pi x} = \prod_{n=1}^\infty \left(1 - \frac{x^2}{n^2}\right)$$

Expanding the left side via the Taylor series for $\sin$:
$$\frac{\sin(\pi x)}{\pi x} = 1 - \frac{\pi^2 x^2}{6} + \frac{\pi^4 x^4}{120} - \cdots$$

Expanding the right side (infinite product), the coefficient of $x^2$ is:
$$-\sum_{n=1}^\infty \frac{1}{n^2}$$

Equating: $-\sum 1/n^2 = -\pi^2/6$, so $\sum_{n=1}^\infty \frac{1}{n^2} = \frac{\pi^2}{6}$.

This approach — identifying a function by its zeros and comparing Taylor series — is now a standard technique in complex analysis. Euler's original argument was made rigorous by Weierstrass's factorization theorem (1876).

### K.2 Ramanujan's Enigmatic Formulas

Srinivasa Ramanujan (1887–1920) discovered series identities that seemed impossible:
$$\frac{1}{\pi} = \frac{2\sqrt{2}}{9801}\sum_{n=0}^\infty \frac{(4n)!}{(n!)^4} \cdot \frac{26390n + 1103}{396^{4n}}$$

This series was discovered empirically and later proven using the theory of elliptic integrals and modular forms. It converges incredibly fast — each term adds about 8 decimal digits to $\pi$.

**Modern use**: The **Chudnovsky algorithm** (1988), based on a Ramanujan-like formula, is used for computing $\pi$ to trillions of digits and is implemented in the `mpmath` Python library.

### K.3 Taylor vs. Fourier: Two Ways to Decompose a Function

The Taylor and Fourier series represent two fundamentally different decomposition strategies:

| | Taylor | Fourier |
|--|--------|---------|
| **Basis** | Powers $\{1, x, x^2, \ldots\}$ | Sinusoids $\{\sin nx, \cos nx\}$ |
| **Locality** | Local: describes $f$ near $a$ | Global: describes $f$ over $[-\pi, \pi]$ |
| **Best for** | Analytic functions, approximation | Periodic functions, signal processing |
| **Convergence** | In disk $|x-a| < R$ | In $L^2$ norm |
| **Derivatives** | Change polynomial degrees | Multiply by $n$ (frequency) |
| **ML use** | Optimizer derivation, GELU, attention | Positional encoding, spectral analysis |

Both are special cases of **harmonic analysis** on different groups:
- Taylor series: the group $(\mathbb{R}, +)$ near 0 (polynomial approximation)
- Fourier series: the circle group $\mathbb{T} = \mathbb{R}/2\pi\mathbb{Z}$ (periodic structure)
- Fourier transform (on $\mathbb{R}$): the full translation group


---

## Appendix L: Advanced Convergence Topics

### L.1 Cauchy Sequences and Completeness

**Definition**: A sequence $(a_n)$ is a **Cauchy sequence** if for every $\epsilon > 0$, there exists $N$ such that for all $m, n > N$: $|a_m - a_n| < \epsilon$.

Intuitively: the terms get arbitrarily close to each other as $n \to \infty$.

**Theorem (Completeness of $\mathbb{R}$)**: Every Cauchy sequence in $\mathbb{R}$ converges. (Equivalently: $\mathbb{R}$ is a **complete metric space**.)

This is the formal statement of the fact that $\mathbb{R}$ has "no gaps". It is what distinguishes $\mathbb{R}$ from $\mathbb{Q}$: the sequence $3, 3.1, 3.14, 3.141, \ldots$ (rational approximations to $\pi$) is Cauchy in $\mathbb{Q}$ but does not converge in $\mathbb{Q}$ (since $\pi \notin \mathbb{Q}$). It does converge in $\mathbb{R}$.

**For ML**: Gradient descent sequences are sometimes analyzed as Cauchy sequences — if parameter updates become arbitrarily small, the sequence is Cauchy in the complete space $\mathbb{R}^d$ and therefore converges (to some local minimum or saddle point).

### L.2 Absolute Convergence and Rearrangements

**Theorem (Absolute convergence → any rearrangement converges to same sum)**: If $\sum a_n$ converges absolutely to $S$, then any rearrangement $\sum a_{\sigma(n)}$ also converges to $S$.

*Proof sketch*: For any $\epsilon > 0$, choose $N$ so that $\sum_{n>N}|a_n| < \epsilon/2$. Any rearrangement that keeps the first $N$ terms (in possibly different order) agrees with the original sum to within $\epsilon$. $\square$

**Theorem (Riemann Rearrangement)**: If $\sum a_n$ converges conditionally, then for any $L \in [-\infty, \infty]$, there exists a rearrangement converging to $L$.

*Proof idea*: Separate positive and negative parts: $p_n = \max(a_n, 0)$ and $q_n = \max(-a_n, 0)$. Conditional convergence implies $\sum p_n = \sum q_n = \infty$ (both diverge). To get sum $L$: greedily add positive terms until partial sum exceeds $L$, then negative terms until below $L$, and repeat. Since $p_n, q_n \to 0$, the oscillations decrease and the series converges to $L$. $\square$

**Concrete example**: The alternating harmonic series $1 - 1/2 + 1/3 - 1/4 + \cdots = \ln 2$.

A rearrangement (two positives, one negative): $1 + 1/3 - 1/2 + 1/5 + 1/7 - 1/4 + \cdots = \frac{3}{2}\ln 2$.

This is not a curiosity — it shows that the value of a conditionally convergent series is an artifact of the order of summation.

### L.3 Power Series at the Boundary: Abel's Theorem

**Theorem (Abel)**: If $\sum_{n=0}^\infty c_n R^n$ converges (where $R$ is the radius of convergence), then:
$$\lim_{x \to R^-} \sum_{n=0}^\infty c_n x^n = \sum_{n=0}^\infty c_n R^n$$

In other words: if the power series converges at an endpoint, the function is continuous there from the inside.

**Example**: $\ln(1+x) = \sum (-1)^{n+1}x^n/n$ for $|x| < 1$. This series converges at $x=1$ (alternating harmonic). By Abel's theorem:
$$\ln 2 = \lim_{x\to 1^-} \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n} = \sum_{n=1}^\infty \frac{(-1)^{n+1}}{n} = 1 - \frac{1}{2} + \frac{1}{3} - \frac{1}{4} + \cdots$$

This justifies the formula $\ln 2 = 1 - 1/2 + 1/3 - 1/4 + \cdots$ from the power series perspective.

### L.4 Summability Methods

Sometimes a divergent series can be assigned a meaningful sum via a **summability method**:

**Cesàro summability**: $\lim_{N\to\infty} \frac{1}{N}\sum_{n=1}^N S_n = L$. The series $1 - 1 + 1 - 1 + \cdots$ has partial sums $1, 0, 1, 0, \ldots$ with average $\to 1/2$. So it is Cesàro summable to $1/2$.

**Abel summability**: $\lim_{x\to 1^-} \sum a_n x^n = L$. For $\sum (-1)^n$: $\sum (-x)^n = 1/(1+x) \to 1/2$ as $x \to 1^-$. Abel sum $= 1/2$.

**For ML**: The value $1 - 1 + 1 - 1 + \cdots = 1/2$ appears in zeta function regularization, used in string theory and occasionally in neural network theory. The regularized sum $\sum_{n=1}^\infty n = -1/12$ (via analytic continuation of the Riemann zeta function) appears in Casimir effect calculations and dimensional regularization in physics.


---

## Appendix M: Worked Examples — Taylor Series Applications

### M.1 Computing $e$ to High Precision

The series $e = \sum_{n=0}^\infty 1/n!$ converges extremely fast (ratio test: ratio $= 1/(n+1) \to 0$).

Error after $N$ terms: $|e - T_N| = \sum_{n=N+1}^\infty 1/n! \le \frac{1}{(N+1)!}\cdot\frac{1}{1 - 1/(N+2)} \approx \frac{1}{(N+1)!}$

For $N = 10$: error $\le 1/11! \approx 2.5 \times 10^{-8}$. For $N = 20$: error $\le 1/21! \approx 10^{-19}$.

This explains why float64 (15 significant digits) is enough: just $N=17$ terms suffice.

### M.2 Taylor Series for Numerical Integration

Integrals like $\int_0^1 e^{-x^2}\,dx$ have no elementary antiderivative, but the series gives:
$$\int_0^1 e^{-x^2}\,dx = \int_0^1 \sum_{n=0}^\infty \frac{(-1)^n x^{2n}}{n!}\,dx = \sum_{n=0}^\infty \frac{(-1)^n}{n!(2n+1)}$$

This alternating series has error bounded by the first omitted term. For 10 significant digits:
$$\left|\frac{1}{n!(2n+1)}\right| < 10^{-10} \implies n \ge 9$$

So 10 terms give: $1 - 1/3 + 1/10 - 1/42 + 1/216 - 1/1320 + 1/9360 - 1/75600 + 1/685440 - 1/6894720 \approx 0.7468241328$

True value: $\frac{\sqrt\pi}{2}\text{erf}(1) \approx 0.7468241329$. ✓

### M.3 L'Hôpital's Rule via Taylor Series

Limits of the form $0/0$ are often cleaner via Taylor series:

$$\lim_{x\to 0}\frac{\sin x - x}{x^3} = \lim_{x\to 0}\frac{(x - x^3/6 + x^5/120 - \cdots) - x}{x^3} = \lim_{x\to 0}\frac{-x^3/6 + O(x^5)}{x^3} = -\frac{1}{6}$$

$$\lim_{x\to 0}\frac{e^x - 1 - x}{x^2} = \lim_{x\to 0}\frac{x^2/2 + O(x^3)}{x^2} = \frac{1}{2}$$

$$\lim_{x\to 0}\frac{1-\cos x}{x^2} = \lim_{x\to 0}\frac{x^2/2 - x^4/24 + \cdots}{x^2} = \frac{1}{2}$$

The Taylor series approach reveals the **rate of vanishing** explicitly — it shows not just that the limit exists but also the leading behavior, which L'Hôpital's rule alone does not.

**For ML**: This technique is used to analyze the behavior of loss functions near saddle points and to derive learning rate bounds from second-order Taylor expansions.

### M.4 Stability of Softmax Computed via log-sum-exp

Standard softmax: `softmax(x)[i] = exp(x[i]) / sum(exp(x))`.

For $x = [1000, 1001, 1002]$ in float32: `exp(1002)` overflows. Result: `nan`.

Log-sum-exp stable version: subtract $m = \max(x) = 1002$:
$$\text{LSE}(x) = 1002 + \ln(e^{-2} + e^{-1} + e^0) = 1002 + \ln(e^{-2} + e^{-1} + 1)$$

Numerically: $e^{-2} \approx 0.135$, $e^{-1} \approx 0.368$, $e^0 = 1$. Sum $\approx 1.503$. $\ln(1.503) \approx 0.408$.

$\text{LSE}(x) \approx 1002.408$. Softmax values: $e^{x_i - \text{LSE}(x)} = e^{-0.408}, e^{0.592-1.0}, e^{0-0.408}$.

The Taylor connection: $\text{LSE}(x) = m + \ln(1 + \sum_{i \ne i^*} e^{x_i - m}) \approx m + \sum_{i \ne i^*} e^{x_i - m}$ (first-order Taylor of $\ln(1+\epsilon)$ for small $\epsilon$) — valid when the max logit dominates by a large margin.


---

## Appendix N: Connections to Advanced Topics

### N.1 $\zeta$ Function and the Riemann Hypothesis

The **Riemann zeta function** extends the p-series to complex $s$:
$$\zeta(s) = \sum_{n=1}^\infty \frac{1}{n^s}, \quad \text{Re}(s) > 1$$

This converges for $\text{Re}(s) > 1$ (by the real p-series result). Riemann's 1859 paper showed $\zeta(s)$ can be analytically continued to all of $\mathbb{C} \setminus \{1\}$ via functional equation.

**Special values**:
- $\zeta(2) = \pi^2/6$ (Basel problem)
- $\zeta(4) = \pi^4/90$
- $\zeta(2n) = (-1)^{n+1}\frac{(2\pi)^{2n}B_{2n}}{2(2n)!}$ (Bernoulli numbers)
- $\zeta(-1) = -1/12$ (regularized value; related to $\sum n = -1/12$ by analytic continuation)

The **Riemann Hypothesis**: all non-trivial zeros of $\zeta(s)$ have $\text{Re}(s) = 1/2$. Unsolved since 1859, it is one of the Millennium Prize Problems (worth $1M).

**Connections to ML**: Spectral methods in deep learning analysis (neural tangent kernel theory) involve the spectrum of infinite-width limits, which connect to $L$-function theory. The random matrix theory of neural network weight spectra has connections to the distribution of Riemann zeta zeros.

### N.2 Spectral Bias of Neural Networks

**Theorem (Rahaman et al., 2019)**: Neural networks with smooth activations learn low-frequency components of the target function before high-frequency components.

**Fourier analysis explanation**: If the target function has Fourier decomposition $f(x) = \sum_k c_k e^{ikx}$, a neural network with gradient flow $\partial_t f_{NN} = -\nabla_{NN}\mathcal{L}$ learns the $k$-th frequency component at rate proportional to the $k$-th Fourier coefficient of the NTK (Neural Tangent Kernel).

For standard activations, the NTK decays with frequency $k$, so low frequencies $k$ are learned first. This **spectral bias** (also called "frequency principle") has practical implications:
- High-frequency functions (sharp edges, textures) require more training
- Random Fourier features can overcome this for specific tasks
- This is why position encodings must explicitly inject high-frequency sinusoids (the transformer couldn't learn high-frequency positional patterns from data alone)

### N.3 Nonlinear Series: Fixed-Point Theorems

Many iterative algorithms in ML produce sequences converging to a fixed point. The key tool is:

**Banach Fixed-Point Theorem (Contraction Mapping)**: If $T: X \to X$ is a contraction on a complete metric space (i.e., $d(Tx, Ty) \le q \cdot d(x,y)$ for $q < 1$), then $T$ has a unique fixed point $x^* = Tx^*$, and the iteration $x_{n+1} = Tx_n$ converges to $x^*$ with geometric rate $q^n$.

**Application — Bellman equations**: The Bellman backup operator $T$ in reinforcement learning satisfies $\|TV - TW\|_\infty \le \gamma\|V - W\|_\infty$ with contraction rate $\gamma < 1$. By Banach's theorem, the value iteration $V_{n+1} = TV_n$ converges to the unique optimal value function $V^*$.

The convergence rate $\gamma^n$ is itself a geometric series — each policy evaluation step reduces the error by factor $\gamma$.

### N.4 Matrix Exponential and Lie Groups

For a square matrix $A$, the **matrix exponential** is defined by the same power series:
$$e^A = \sum_{n=0}^\infty \frac{A^n}{n!} = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots$$

This series always converges (operator norm satisfies $\|A^n\| \le \|A\|^n$, so the series is bounded by the scalar exponential $e^{\|A\|}$).

**Applications in ML**:
- **Equivariant networks**: The group of rotations $SO(3)$ has a Lie algebra $\mathfrak{so}(3)$ (skew-symmetric matrices). The exponential map $\exp: \mathfrak{so}(3) \to SO(3)$ parameterizes rotations via power series. Used in SE(3)-equivariant networks (FrameDiff, AlphaFold2 structure module).
- **Continuous normalizing flows**: The flow $z(t) = e^{tA}z(0)$ for a learned $A$; the trace $\text{tr}(A) = \sum_i \lambda_i$ controls the log-determinant.
- **Mamba / state space models**: The discrete-time recurrence $(A_k, B_k)$ is derived from continuous matrices via matrix exponential with zero-order hold.


---

## Appendix O: Final Reference and Checklist

### O.1 Essential Formulas at a Glance

**Geometric series**:
$$\sum_{n=0}^\infty r^n = \frac{1}{1-r} \quad (|r|<1), \qquad \sum_{n=0}^{N-1} r^n = \frac{1-r^N}{1-r}$$

**p-series criterion**: $\sum 1/n^p$ converges $\iff p > 1$

**Radius of convergence**: $R = \lim_{n\to\infty} \left|\frac{c_n}{c_{n+1}}\right|$ (ratio method); $R = 1/\limsup |c_n|^{1/n}$ (root method)

**Taylor series**: $f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n$

**Lagrange error bound**: $|R_n(x)| \le \frac{M}{(n+1)!}|x-a|^{n+1}$ where $M = \max|f^{(n+1)}|$

**Euler's formula**: $e^{i\theta} = \cos\theta + i\sin\theta$

**Key Maclaurin series**:
- $e^x = \sum x^n/n!$ (all $x$)
- $\sin x = \sum (-1)^n x^{2n+1}/(2n+1)!$ (all $x$)
- $\cos x = \sum (-1)^n x^{2n}/(2n)!$ (all $x$)
- $\ln(1+x) = \sum (-1)^{n+1}x^n/n$ ($-1 < x \le 1$)
- $1/(1-x) = \sum x^n$ ($|x| < 1$)

**Adam bias correction (geometric series)**:
$$\mathbb{E}[m_t] = (1-\beta^t)g, \quad \hat{m}_t = \frac{m_t}{1-\beta^t}$$

**Linear attention (Taylor approximation)**:
$$e^{q^\top k} \approx 1 + q^\top k, \quad \text{error} \le \frac{(q^\top k)^2}{2} e^{|q^\top k|}$$

**Laplace approximation**:
$$p(\theta|\mathcal{D}) \approx \mathcal{N}\!\left(\hat\theta, \left[-\nabla^2\log p(\hat\theta|\mathcal{D})\right]^{-1}\right)$$

### O.2 Final Self-Assessment

After completing this entire section, you should be able to:

**Prove from scratch**:
- [ ] Harmonic series diverges (Oresme grouping)
- [ ] Ratio test (convergence case)
- [ ] Alternating series test (Leibniz)
- [ ] Taylor's theorem with Lagrange remainder
- [ ] Adam bias correction derivation using geometric series

**Compute fluently**:
- [ ] Sums of geometric and telescoping series
- [ ] Radius and interval of convergence for power series
- [ ] Taylor/Maclaurin series for standard functions
- [ ] Approximation errors using Lagrange remainder
- [ ] New series via substitution, differentiation, integration

**Explain to a colleague**:
- [ ] Why conditional convergence allows Riemann rearrangement
- [ ] How linear attention reduces complexity using Taylor series
- [ ] Why softmax becomes uniform at high temperature
- [ ] What the Laplace approximation approximates and when it's valid
- [ ] How RoPE uses complex exponentials for relative position encoding


---

## Appendix P: Connections to Adjacent Sections

### P.1 ← §04/01 Limits and Continuity

The entire theory of sequences is the theory of limits restricted to integer inputs. Every result about sequence convergence is a special case of the limit definition from §01:

- $\lim_{n\to\infty} a_n = L$ is the ε-δ limit with $x \to \infty$ and $\delta$ replaced by $N$
- The Squeeze Theorem for sequences is the Squeeze Theorem from §01 restricted to integers
- Continuous functions preserve sequence limits: $a_n \to L \implies f(a_n) \to f(L)$ — this is the sequential characterization of continuity from §01

**Formal link**: A function $f: \mathbb{R} \to \mathbb{R}$ is continuous at $a$ if and only if for every sequence $x_n \to a$, we have $f(x_n) \to f(a)$. This makes sequences a fundamental tool for studying continuous functions.

### P.2 ← §04/02 Derivatives and Differentiation

Taylor series are the direct children of differentiation:

- The $n$-th Taylor coefficient is $f^{(n)}(a)/n!$, requiring $n$ derivatives at $a$
- The Lagrange remainder involves the $(n+1)$-th derivative
- L'Hôpital's rule (for $0/0$ limits) is superseded by Taylor series analysis, which gives the limit value AND the rate of approach
- The derivative of a power series is a power series (via term-by-term differentiation) — justified by uniform convergence

### P.3 ← §04/03 Integration

Integration provides two key tools for series theory:

1. **Integral test**: $\sum a_n$ converges iff $\int f\,dx$ converges — directly using the improper integrals from §03
2. **Term-by-term integration**: $\int \sum c_n x^n\,dx = \sum c_n x^{n+1}/(n+1)$ — used to derive $\ln(1+x)$ from $1/(1+x)$ and $\arctan x$ from $1/(1+x^2)$

The connection is deep: both integration and summation are instances of the same "linear functional on functions" idea from functional analysis.

### P.4 → §05 Multivariate Calculus

The multivariable Taylor theorem (the most important formula in optimization theory) is the direct extension of the univariate Taylor series:
$$f(\mathbf{x} + \mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{h} + \frac{1}{2}\mathbf{h}^\top H(\mathbf{x})\mathbf{h} + O(\|\mathbf{h}\|^3)$$

Every gradient descent convergence proof, every Newton method derivation, and every trust region method is an application of this formula. The radius of convergence concept extends to a "trust region" around the expansion point.

→ *Full treatment: [§05 Multivariate Calculus](../../05-Multivariate-Calculus/README.md)*

### P.5 → §06 Probability and Statistics

Generating functions, characteristic functions, and moment generating functions are all power series / Fourier transforms in disguise:

- **MGF**: $M_X(t) = \mathbb{E}[e^{tX}] = \sum_n \mathbb{E}[X^n]t^n/n!$
- **PGF**: $G_X(z) = \mathbb{E}[z^X] = \sum_n p_n z^n$ (discrete distributions)
- **Characteristic function**: $\phi_X(t) = \mathbb{E}[e^{itX}]$ (Fourier transform of density)

The Central Limit Theorem proof via characteristic functions uses Taylor expansion of $\log\phi_X(t)$ and the fact that the Gaussian characteristic function is $e^{-t^2/2}$.

→ *Full treatment: [§06 Probability and Statistics](../../06-Probability-and-Statistics/README.md)*


---

## Appendix Q: Additional Exercises with Full Solutions

### Q.1 Power Series Manipulation

**Problem**: Starting from $\frac{1}{1-x} = \sum_{n=0}^\infty x^n$, derive:
(a) $\frac{1}{(1-x)^2} = \sum_{n=1}^\infty n x^{n-1}$
(b) $\frac{x}{(1-x)^2} = \sum_{n=1}^\infty n x^n$
(c) $\ln(2) = 1 - 1/2 + 1/3 - 1/4 + \cdots$ (alternating harmonic series)

**Solutions**:

**(a)** Differentiate $\sum x^n = 1/(1-x)$ term-by-term (valid for $|x| < 1$):
$$\frac{d}{dx}\sum_{n=0}^\infty x^n = \sum_{n=1}^\infty nx^{n-1} = \frac{d}{dx}\frac{1}{1-x} = \frac{1}{(1-x)^2}$$

**(b)** Multiply (a) by $x$: $\sum_{n=1}^\infty nx^n = \frac{x}{(1-x)^2}$.

**(c)** From $\ln(1+x) = \sum_{n=1}^\infty \frac{(-1)^{n+1}x^n}{n}$ (derived by integrating $1/(1+x)$). At $x=1$: the series $\sum (-1)^{n+1}/n$ converges by the alternating series test ($1/n$ is decreasing to 0). By Abel's theorem (continuity at boundary), $\ln(1+1) = \ln 2 = \sum_{n=1}^\infty (-1)^{n+1}/n = 1 - 1/2 + 1/3 - \cdots$.

### Q.2 Convergence Test Practice

Determine convergence of $\sum_{n=1}^\infty \frac{(-3)^n}{n^2 \cdot 2^n}$.

**Solution**: Rewrite as $\sum (-3/2)^n / n^2$. The ratio $|-3/2| = 3/2 > 1$, so we first try the absolute value: $\sum (3/2)^n/n^2$.

Ratio test: $\frac{(3/2)^{n+1}/(n+1)^2}{(3/2)^n/n^2} = \frac{3}{2} \cdot \frac{n^2}{(n+1)^2} \to \frac{3}{2} > 1$.

So $\sum (3/2)^n/n^2$ diverges. Since the series diverges absolutely, check conditional convergence: alternating series test needs $b_n = (3/2)^n/n^2 \to 0$? No — $(3/2)^n \to \infty$. So $b_n \to \infty$, violating the necessary condition. The series **diverges**.

### Q.3 Full Taylor Expansion of $\arctan(x)$

**Derive**: $\arctan x = \sum_{n=0}^\infty \frac{(-1)^n x^{2n+1}}{2n+1}$ for $|x| \le 1$.

**Solution**: Start from $\frac{1}{1+t^2} = \sum_{n=0}^\infty (-1)^n t^{2n}$ (geometric series with $r = -t^2$, valid for $|t| < 1$).

Integrate from 0 to $x$:
$$\arctan x = \int_0^x \frac{dt}{1+t^2} = \int_0^x \sum_{n=0}^\infty (-1)^n t^{2n}\,dt = \sum_{n=0}^\infty (-1)^n \frac{x^{2n+1}}{2n+1}$$

Valid for $|x| < 1$. At $x = 1$: $\sum (-1)^n/(2n+1)$ converges by alternating series test. At $x = -1$: also converges. By Abel's theorem, the formula holds on $[-1, 1]$.

**Leibniz formula for $\pi$**: Setting $x = 1$:
$$\frac{\pi}{4} = 1 - \frac{1}{3} + \frac{1}{5} - \frac{1}{7} + \cdots$$

This converges, but VERY slowly (error after $N$ terms $\approx 1/(2N+1)$; need $N \approx 5 \times 10^9$ for 10 digits). In practice, $\pi$ is computed via faster-converging formulas (Machin's: $\pi/4 = 4\arctan(1/5) - \arctan(1/239)$).


---

## Appendix R: Series in Numerical Computing

### R.1 Float Arithmetic and Series Truncation

Every floating-point number is represented with finite precision. When computing a power series numerically, two errors compete:

1. **Truncation error**: stopping at $N$ terms instead of $\infty$. Bounded by Lagrange remainder.
2. **Rounding error**: each term $c_n x^n$ is rounded to nearest float. Accumulates with $N$ terms.

The optimal $N$ balances these — adding more terms reduces truncation error but eventually increases rounding error. For `float64` with $\epsilon_{\text{mach}} \approx 10^{-16}$:

**Optimal $N$ for $e^x$ near $x=0$**: truncation error $\approx |x|^{N+1}/(N+1)!$; rounding error $\approx N\epsilon_{\text{mach}} e^{|x|}$. Set equal: $N! \approx |x|^N / (N\epsilon_{\text{mach}})$. For $|x| = 1$: optimal $N \approx 17$ (matches the 17 significant digits of float64).

**For large $x$**: The terms $x^n/n!$ grow before shrinking (the series oscillates in magnitude). Catastrophic cancellation: many large positive and negative terms cancel. Solution: use `exp(x) = exp(x/2)^2` (argument reduction) to bring $x$ near 0 before applying the series.

### R.2 Compensated Summation (Kahan Algorithm)

When summing many floating-point numbers, naive summation accumulates $O(n\varepsilon)$ error. The **Kahan compensated sum** tracks the lost bits:

```python
def kahan_sum(values):
    s = 0.0
    c = 0.0          # running compensation
    for v in values:
        y = v - c    # compensate the next term
        t = s + y    # new sum (low bits of y may be lost)
        c = (t - s) - y  # recover what was lost
        s = t
    return s
```

This reduces error to $O(\varepsilon)$ regardless of $n$. Libraries like NumPy use **pairwise summation** (divide-and-conquer), achieving $O(\varepsilon \log n)$ error — a good balance.

**For ML**: In mixed-precision training (float16 weights, float32 accumulation), the summation algorithm in the optimizer matters. Kahan summation in the gradient accumulation step significantly reduces precision loss.

### R.3 Euler's Transform for Alternating Series Acceleration

The alternating harmonic series $\sum (-1)^{n+1}/n = \ln 2$ converges slowly. **Euler's transform** accelerates convergence:

$$\sum_{n=0}^\infty (-1)^n a_n = \sum_{n=0}^\infty \frac{\Delta^n a_0}{2^{n+1}}$$

where $\Delta^n a_0$ are finite differences: $\Delta^0 a_k = a_k$, $\Delta^{n+1} a_k = \Delta^n a_{k+1} - \Delta^n a_k$.

For $a_n = 1/(n+1)$ (alternating harmonic):
- $\Delta^0 a_0 = 1$, $\Delta^1 a_0 = 1/2 - 1 = -1/2$, $\Delta^2 a_0 = -1/3 + 1/2 - 1/2 = 1/3 - 1 = ...$

The transformed series converges much faster. This is used in high-precision computation of $\ln 2$, $\pi$, and other constants.

---

## Appendix S: Topic Map and Reading Guide

### S.1 Minimum Viable Reading

For an ML practitioner who needs only the most practically relevant material:

1. **§1.3**: Why series matter for AI (overview table)
2. **§3.2**: Geometric series (Adam bias correction, discounted rewards)
3. **§4.4**: Ratio test (to understand radius of convergence)
4. **§5.2**: Radius of convergence (when to trust a Taylor approximation)
5. **§6.1–6.3**: Taylor series derivation and Lagrange error bounds
6. **§7.1–7.6**: All ML applications in full
7. **§8.3**: RoPE and positional encodings

### S.2 Theoretical Deep Dive

For a reader interested in rigorous analysis:

1. **§2.2**: ε-N definition of sequence convergence (in full)
2. **§2.3**: Monotone Convergence Theorem (proof)
3. **§4**: All convergence tests (with proofs in Appendix A)
4. **§5**: Power series theory, radius of convergence, uniform convergence (App. B)
5. **§6.4**: When does the Taylor series converge to $f$? (Analytic functions)
6. **Appendix L**: Cauchy sequences, Riemann rearrangement, Abel's theorem
7. **Appendix N**: Zeta function, spectral bias, Banach fixed-point theorem

### S.3 Prerequisites by Section

| Section | Requires |
|---------|---------|
| §2 Sequences | §04/01 Limits (ε-δ definition) |
| §3 Infinite Series | §2 (convergence of sequences) |
| §4 Convergence Tests | §3.1 (partial sums), §04/03 Integration (integral test) |
| §5 Power Series | §4 (convergence tests), especially ratio test |
| §6 Taylor Series | §04/02 Derivatives (all orders), §5 (power series) |
| §7 ML Applications | §6 (Taylor series), §04/03 (integration) |
| §8 Fourier Preview | §04/03 (integration, orthogonality integrals) |
| Appendix B | §5 (uniform convergence motivation) |
| Appendix C | Complex numbers (basic) |
| Appendix D | §6 (generating functions), §04/03 probability |
| Appendix N | §6 (Taylor), §04/01 (limits), some complex analysis |

