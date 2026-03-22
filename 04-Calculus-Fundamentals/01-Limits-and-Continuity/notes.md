[← Back to Calculus Fundamentals](../README.md) | [Next: Derivatives and Differentiation →](../02-Derivatives-and-Differentiation/notes.md)

---

# Limits and Continuity

> _"The concept of the limit is the cornerstone on which the whole of mathematical analysis ultimately rests."_
> — Aleksandr Khinchin, _Mathematical Foundations of Information Theory_ (1957)

## Overview

Limits formalize the intuition of "approaching" — what value does a function tend toward as its input gets arbitrarily close to some point? This deceptively simple question required two centuries to answer rigorously, from Newton and Leibniz's intuitive infinitesimals to Cauchy's sequential definition to Weierstrass's definitive ε-δ framework. The result is the foundation upon which all of calculus — and by extension, all of continuous optimization — is built.

Continuity is the qualitative property that emerges from limits behaving well: a function is continuous at a point if its limit there equals its actual value. Continuous functions are the "nice" functions of analysis — they preserve neighborhoods, satisfy the Intermediate Value Theorem, and attain extrema on compact sets. In machine learning, continuity is not just a mathematical nicety; it is a design criterion for activation functions, loss landscapes, and optimization trajectories.

For AI practitioners, limits appear in at least five critical contexts: (1) the definition of the derivative as a limit of difference quotients, which underlies all backpropagation; (2) softmax temperature annealing, where $T \to 0$ recovers argmax and $T \to \infty$ recovers uniform; (3) the vanishing gradient problem, where $\sigma(x) \to 0$ as $x \to \pm\infty$ kills gradient signal; (4) numerical stability, where naive limit computations suffer catastrophic cancellation; and (5) learning rate schedules, where the Robbins-Monro conditions $\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$ guarantee convergence.

## Prerequisites

- **Functions**: domain, codomain, composition, inverse — [§01-Mathematical-Foundations](../../01-Mathematical-Foundations/)
- **Algebra**: polynomial factoring, rationalization, conjugate multiplication
- **Exponential and logarithmic functions**: $e^x$, $\ln x$, basic identities
- **Trigonometry**: $\sin$, $\cos$, $\tan$ and their basic properties
- **Real number system**: $\mathbb{R}$, absolute value $\lvert x \rvert$, the Archimedean property

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive examples: ε-δ visualization, limit laws, IVT, discontinuity types, ML applications |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from basic limit computation to gradient-as-limit and cross-entropy |

## Learning Objectives

After completing this section, you will be able to:

1. **State** the ε-δ definition of a limit and verify it for elementary functions
2. **Compute** limits using algebraic manipulation, L'Hôpital's Rule, and the Squeeze Theorem
3. **Distinguish** one-sided from two-sided limits and determine when each exists
4. **Identify** and classify discontinuities as removable, jump, or essential
5. **State and apply** the Intermediate Value Theorem and Extreme Value Theorem
6. **Prove** the Squeeze Theorem and use it to evaluate $\lim_{x \to 0} \frac{\sin x}{x}$
7. **Recognize** indeterminate forms and choose the appropriate resolution technique
8. **Implement** numerically stable limit computations using `expm1`, `log1p`, and `log_softmax`
9. **Analyze** softmax temperature, sigmoid saturation, and ReLU continuity via limits
10. **Connect** the limit definition to the derivative as preparation for §02

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Approaching Without Arriving](#11-approaching-without-arriving)
  - [1.2 Historical Necessity: From Newton to Weierstrass](#12-historical-necessity-from-newton-to-weierstrass)
  - [1.3 Why Limits Matter for AI](#13-why-limits-matter-for-ai)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The ε-δ Definition](#21-the-ε-δ-definition)
  - [2.2 Limit Laws](#22-limit-laws)
  - [2.3 One-Sided Limits](#23-one-sided-limits)
  - [2.4 Limits at Infinity](#24-limits-at-infinity)
  - [2.5 Infinite Limits and Vertical Asymptotes](#25-infinite-limits-and-vertical-asymptotes)
- [3. Fundamental Limits](#3-fundamental-limits)
  - [3.1 The Sine Limit: sin(x)/x → 1](#31-the-sine-limit-sinxx--1)
  - [3.2 Exponential and Logarithmic Limits](#32-exponential-and-logarithmic-limits)
  - [3.3 Euler's Number as a Limit](#33-eulers-number-as-a-limit)
  - [3.4 Logarithmic Limits and xln(x)](#34-logarithmic-limits-and-xlnx)
- [4. Computation Techniques](#4-computation-techniques)
  - [4.1 Algebraic Techniques](#41-algebraic-techniques)
  - [4.2 L'Hôpital's Rule](#42-lhôpitals-rule)
  - [4.3 The Squeeze Theorem](#43-the-squeeze-theorem)
  - [4.4 Substitution and Composition](#44-substitution-and-composition)
- [5. Continuity](#5-continuity)
  - [5.1 The Three-Part Definition](#51-the-three-part-definition)
  - [5.2 Types of Discontinuity](#52-types-of-discontinuity)
  - [5.3 Continuity on Intervals](#53-continuity-on-intervals)
  - [5.4 Operations Preserving Continuity](#54-operations-preserving-continuity)
- [6. Key Theorems](#6-key-theorems)
  - [6.1 Intermediate Value Theorem](#61-intermediate-value-theorem)
  - [6.2 Extreme Value Theorem](#62-extreme-value-theorem)
  - [6.3 Uniform Continuity](#63-uniform-continuity)
- [7. Asymptotic Behavior](#7-asymptotic-behavior)
  - [7.1 Horizontal and Vertical Asymptotes](#71-horizontal-and-vertical-asymptotes)
  - [7.2 Polynomial vs. Exponential Growth](#72-polynomial-vs-exponential-growth)
  - [7.3 Big-O and Little-o as Limit Statements](#73-big-o-and-little-o-as-limit-statements)
- [8. Numerical Stability Near Limits](#8-numerical-stability-near-limits)
  - [8.1 Catastrophic Cancellation](#81-catastrophic-cancellation)
  - [8.2 Numerically Stable Alternatives](#82-numerically-stable-alternatives)
  - [8.3 Machine Epsilon and Floating-Point Limits](#83-machine-epsilon-and-floating-point-limits)
- [9. Machine Learning Applications](#9-machine-learning-applications)
  - [9.1 Softmax Temperature Limits](#91-softmax-temperature-limits)
  - [9.2 Sigmoid Saturation and Vanishing Gradients](#92-sigmoid-saturation-and-vanishing-gradients)
  - [9.3 ReLU and GELU: Continuity at the Origin](#93-relu-and-gelu-continuity-at-the-origin)
  - [9.4 Learning Rate Decay: Robbins-Monro Conditions](#94-learning-rate-decay-robbins-monro-conditions)
  - [9.5 Gradient as a Limit](#95-gradient-as-a-limit)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 Approaching Without Arriving

Consider the function $f(x) = \frac{x^2 - 4}{x - 2}$. At $x = 2$, this function is undefined: both numerator and denominator are zero, and division by zero is not permitted. Yet if you evaluate $f$ at values close to $2$ — say $x = 1.9, 1.99, 1.999$ and $x = 2.1, 2.01, 2.001$ — you observe that $f(x)$ gets arbitrarily close to $4$. The function is not defined at $x = 2$, but its behavior near $x = 2$ is perfectly predictable.

This is the key insight: a **limit** describes the behavior of $f(x)$ as $x$ approaches $a$, regardless of what $f(a)$ is (or whether $f(a)$ exists at all). We write:

$$\lim_{x \to a} f(x) = L$$

and read it as: "the limit of $f(x)$ as $x$ approaches $a$ equals $L$."

For the example above:

$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = \lim_{x \to 2} \frac{(x-2)(x+2)}{x-2} = \lim_{x \to 2} (x + 2) = 4$$

The cancellation of $(x - 2)$ is valid because we are considering $x$ close to (but not equal to) $2$.

```
LIMITS: THE CORE PICTURE
════════════════════════════════════════════════════════════════════════

         f(x)
           │
         5 ┤
           │
         4 ┤- - - - - - - - - ○ ← limit value (hole at x=2)
           │              ╱
         3 ┤           ╱
           │        ╱
         2 ┤     ╱
           │  ╱
         1 ┤╱
           │
           └─────┬────┬───┬──┬──┬───┬────┬──── x
                 1   1.5  1.9 2  2.1  2.5   3

  f(x) = (x²-4)/(x-2)   Undefined at x=2, but the limit is 4.
  As x → 2 from either side, f(x) → 4.

════════════════════════════════════════════════════════════════════════
```

The informal statement is: $\lim_{x \to a} f(x) = L$ means "we can make $f(x)$ as close to $L$ as we like by taking $x$ sufficiently close to $a$."

The crucial distinction: a limit is about **neighborhood behavior**, not about what happens at the point itself. This distinction is what makes calculus work: the derivative of $f$ at $a$ is defined as the limit of difference quotients, even though the difference quotient $\frac{f(a+h) - f(a)}{h}$ is never evaluated at $h = 0$.

### 1.2 Historical Necessity: From Newton to Weierstrass

The limit concept was not discovered all at once — it was forced on mathematicians by the need to make calculus rigorous.

**Newton (1666) and Leibniz (1675)** independently invented calculus using "infinitesimals" — quantities smaller than any real number but not zero. Their methods worked in practice but were logically incoherent: you cannot simultaneously treat $h$ as nonzero (to divide) and as zero (to drop $h^2$ terms). Bishop Berkeley famously mocked infinitesimals as "ghosts of departed quantities."

**Cauchy (1821)** introduced the sequential approach: $\lim_{x \to a} f(x) = L$ if for every sequence $x_n \to a$ (with $x_n \neq a$), the sequence $f(x_n) \to L$. This was much cleaner, but still relied on intuitive notions of "approaching."

**Weierstrass (1861)** gave the definitive formulation, now universal in analysis: the ε-δ definition (Section 2.1). This eliminated all appeals to intuition and motion — limits became purely a statement about inequalities between real numbers.

**Chronological summary:**

| Year | Mathematician | Contribution |
|------|--------------|--------------|
| 1666/1675 | Newton / Leibniz | Calculus via infinitesimals (intuitive, not rigorous) |
| 1797 | Lagrange | Attempted algebraic foundation via power series |
| 1821 | Cauchy | Sequential definition; "sum formula" for continuity |
| 1861 | Weierstrass | ε-δ definition; made analysis fully rigorous |
| 1960s | Robinson | Nonstandard analysis: infinitesimals made rigorous via model theory |

The modern formulation we use today is Weierstrass's. Interestingly, nonstandard analysis (Abraham Robinson, 1966) later vindicated Newton's intuition by making infinitesimals rigorous through model theory — but the ε-δ approach remains standard in mathematical practice.

### 1.3 Why Limits Matter for AI

Limits are not just historical curiosities — they are load-bearing mathematics in modern AI systems.

**Gradient computation**: Every gradient in a neural network is a limit. The gradient of loss $\mathcal{L}$ with respect to weight $w$ is:
$$\frac{\partial \mathcal{L}}{\partial w} = \lim_{h \to 0} \frac{\mathcal{L}(w + h) - \mathcal{L}(w)}{h}$$
Automatic differentiation implements this limit algebraically rather than numerically, but the underlying mathematics is limit-theoretic.

**Softmax temperature**: The softmax function with temperature $T$ is:
$$\text{softmax}_T(\mathbf{z})_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$
Taking $\lim_{T \to 0^+}$ recovers the argmax (one-hot); $\lim_{T \to \infty}$ recovers the uniform distribution. Understanding these limits is essential for temperature-scaled inference in LLMs (Radford et al., 2019; Ouyang et al., 2022).

**Vanishing gradients**: The sigmoid $\sigma(x) = 1/(1 + e^{-x})$ satisfies $\lim_{x \to \pm\infty} \sigma(x) \in \{0, 1\}$, and $\lim_{x \to \pm\infty} \sigma'(x) = 0$. Deep networks with sigmoid activations suffer gradient vanishing precisely because of this limit behavior (Hochreiter & Schmidhuber, 1997).

**Numerical stability**: Computing $(e^x - 1)/x$ naively near $x = 0$ loses up to 16 digits of precision due to catastrophic cancellation. The function `numpy.expm1` computes $e^x - 1$ directly to avoid this — a practical implementation of numerically stable limits.

**Convergence guarantees**: Stochastic gradient descent converges (in expectation) if the learning rate sequence $\{\alpha_t\}$ satisfies the Robbins-Monro conditions $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$ (Robbins & Monro, 1951). These are conditions on the limiting behavior of a series — pure limit theory applied to optimization.

---

## 2. Formal Definitions

### 2.1 The ε-δ Definition

**Definition (Limit).** Let $f$ be a function defined on some open interval containing $a$, except possibly at $a$ itself. We say

$$\lim_{x \to a} f(x) = L$$

if for every $\varepsilon > 0$, there exists $\delta > 0$ such that

$$0 < \lvert x - a \rvert < \delta \implies \lvert f(x) - L \rvert < \varepsilon$$

**Unpacking the definition:**
- "$\varepsilon > 0$" — your challenger specifies how close to $L$ you must get
- "there exists $\delta > 0$" — you respond with a proximity to $a$
- "$0 < \lvert x - a \rvert < \delta$" — $x$ is within $\delta$ of $a$, but $x \neq a$
- "$\lvert f(x) - L \rvert < \varepsilon$" — then $f(x)$ is within $\varepsilon$ of $L$

The definition is a **game**: for every precision challenge $\varepsilon$ your opponent poses, you must produce a $\delta$ that wins. If you can always win, the limit is $L$.

```
ε-δ VISUALIZATION
════════════════════════════════════════════════════════════════════════

  f(x)
    │
 L+ε┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← ε-strip (top)
    │         f(x) must stay inside │
    L┤  - - - - - - - ○ - - - - - -│- - - -   ← limit value L
    │                 │            │
 L-ε┤━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← ε-strip (bottom)
    │          a-δ    a    a+δ
    └──────────┬──────┬────┬────────────────→ x
               │←  2δ  →│
           δ-window: x must stay inside this

  For x in (a-δ, a+δ) \ {a}, f(x) must lie in (L-ε, L+ε).

════════════════════════════════════════════════════════════════════════
```

**Example (proving a limit with ε-δ):** Prove $\lim_{x \to 3} (2x - 1) = 5$.

We need: given $\varepsilon > 0$, find $\delta > 0$ such that $0 < \lvert x - 3 \rvert < \delta \implies \lvert (2x-1) - 5 \rvert < \varepsilon$.

Note: $\lvert (2x-1) - 5 \rvert = \lvert 2x - 6 \rvert = 2\lvert x - 3 \rvert$.

So if $\lvert x - 3 \rvert < \delta$, then $\lvert (2x-1) - 5 \rvert = 2\lvert x - 3 \rvert < 2\delta$.

Choose $\delta = \varepsilon/2$. Then $2\delta = \varepsilon$, so $\lvert (2x-1) - 5 \rvert < \varepsilon$. $\square$

**Example (limit does not exist):** $f(x) = \sin(1/x)$ as $x \to 0$. For any $\delta > 0$, the interval $(0, \delta)$ contains points where $\sin(1/x) = 1$ and points where $\sin(1/x) = -1$. No single $L$ can satisfy the ε-δ condition with $\varepsilon = 1/2$. The limit does not exist.

**Non-examples (limit fails to exist):**
- $f(x) = \lvert x \rvert / x$ as $x \to 0$: left limit is $-1$, right limit is $+1$ — unequal one-sided limits
- $f(x) = 1/x^2$ as $x \to 0$: $f(x) \to +\infty$ — no finite $L$
- $f(x) = \sin(1/x)$ as $x \to 0$: oscillates between $-1$ and $1$ without settling

### 2.2 Limit Laws

Limits interact well with algebraic operations. If $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$, then:

| Law | Statement |
|-----|-----------|
| Sum | $\lim_{x \to a} [f(x) + g(x)] = L + M$ |
| Difference | $\lim_{x \to a} [f(x) - g(x)] = L - M$ |
| Constant Multiple | $\lim_{x \to a} [c \cdot f(x)] = c \cdot L$ |
| Product | $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$ |
| Quotient | $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}$, provided $M \neq 0$ |
| Power | $\lim_{x \to a} [f(x)]^n = L^n$ for integer $n$ |
| Root | $\lim_{x \to a} \sqrt[n]{f(x)} = \sqrt[n]{L}$ (for even $n$, require $L \geq 0$) |
| Composition | If $g$ is continuous at $L$: $\lim_{x \to a} g(f(x)) = g(L)$ |

**Proof sketch (Sum Law):** Given $\varepsilon > 0$. Since $L_f$ and $L_g$ are limits, choose $\delta_1, \delta_2$ such that $\lvert f(x) - L \rvert < \varepsilon/2$ and $\lvert g(x) - M \rvert < \varepsilon/2$ within their respective $\delta$-windows. Set $\delta = \min(\delta_1, \delta_2)$. Then:
$$\lvert f(x) + g(x) - (L+M) \rvert \leq \lvert f(x) - L \rvert + \lvert g(x) - M \rvert < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon \qquad \square$$

**Direct substitution:** If $f$ is a polynomial, rational function (with nonzero denominator), or composed of continuous functions, then $\lim_{x \to a} f(x) = f(a)$. This is valid whenever $f$ is continuous at $a$ — essentially the definition of continuity (Section 5).

### 2.3 One-Sided Limits

For functions that behave differently approaching from the left versus the right, we introduce one-sided limits.

**Definition (Right-hand limit).** $\lim_{x \to a^+} f(x) = L$ if for every $\varepsilon > 0$ there exists $\delta > 0$ such that $a < x < a + \delta \implies \lvert f(x) - L \rvert < \varepsilon$.

**Definition (Left-hand limit).** $\lim_{x \to a^-} f(x) = L$ if for every $\varepsilon > 0$ there exists $\delta > 0$ such that $a - \delta < x < a \implies \lvert f(x) - L \rvert < \varepsilon$.

**Key theorem:** $\lim_{x \to a} f(x) = L$ if and only if both one-sided limits exist and are equal:
$$\lim_{x \to a^-} f(x) = L = \lim_{x \to a^+} f(x)$$

**Examples:**

*Sign function* $\text{sgn}(x) = x / \lvert x \rvert$ for $x \neq 0$:
$$\lim_{x \to 0^-} \text{sgn}(x) = -1, \quad \lim_{x \to 0^+} \text{sgn}(x) = +1$$
Since $-1 \neq 1$, the two-sided limit $\lim_{x \to 0} \text{sgn}(x)$ does not exist.

*Floor function* $\lfloor x \rfloor$: at any integer $n$,
$$\lim_{x \to n^-} \lfloor x \rfloor = n - 1, \quad \lim_{x \to n^+} \lfloor x \rfloor = n$$
Jump discontinuity at every integer.

*Absolute value* $\lvert x \rvert$:
$$\lim_{x \to 0^-} \lvert x \rvert = 0 = \lim_{x \to 0^+} \lvert x \rvert$$
Two-sided limit exists and equals $0 = \lvert 0 \rvert$, so $\lvert x \rvert$ is continuous at $0$.

**For AI:** ReLU$(x) = \max(0, x)$ has well-defined one-sided limits everywhere. At $x = 0$:
$$\lim_{x \to 0^-} \text{ReLU}(x) = 0, \quad \lim_{x \to 0^+} \text{ReLU}(x) = 0$$
Both equal $\text{ReLU}(0) = 0$, so ReLU is continuous at the origin. However, its derivative (Section 9.3) has a jump discontinuity there.

### 2.4 Limits at Infinity

**Definition.** $\lim_{x \to +\infty} f(x) = L$ means: for every $\varepsilon > 0$, there exists $M > 0$ such that $x > M \implies \lvert f(x) - L \rvert < \varepsilon$.

Similarly for $\lim_{x \to -\infty} f(x) = L$ with $x < -M$.

**Standard examples:**

$$\lim_{x \to \infty} \frac{1}{x} = 0, \quad \lim_{x \to \infty} \frac{1}{x^2} = 0, \quad \lim_{x \to \infty} e^{-x} = 0$$

$$\lim_{x \to \infty} \arctan(x) = \frac{\pi}{2}, \quad \lim_{x \to -\infty} \arctan(x) = -\frac{\pi}{2}$$

**Rational function limits at infinity:** For $p(x)/q(x)$ with $\deg(p) = m$, $\deg(q) = n$:
- If $m < n$: limit is $0$
- If $m = n$: limit is ratio of leading coefficients
- If $m > n$: limit is $\pm\infty$

**Example:**
$$\lim_{x \to \infty} \frac{3x^2 + x - 1}{2x^2 - 5} = \lim_{x \to \infty} \frac{3 + 1/x - 1/x^2}{2 - 5/x^2} = \frac{3 + 0 - 0}{2 - 0} = \frac{3}{2}$$

**For AI:** Sigmoid saturation at infinity:
$$\lim_{x \to +\infty} \sigma(x) = 1, \quad \lim_{x \to -\infty} \sigma(x) = 0$$
This is why sigmoid-based networks suffer from vanishing gradients: for large $\lvert x \rvert$, the sigmoid is essentially constant, so its derivative is essentially zero.

### 2.5 Infinite Limits and Vertical Asymptotes

**Definition.** $\lim_{x \to a} f(x) = +\infty$ means: for every $M > 0$, there exists $\delta > 0$ such that $0 < \lvert x - a \rvert < \delta \implies f(x) > M$.

Note: $+\infty$ is not a real number — saying the limit is $+\infty$ means the function grows without bound, not that it converges.

**Vertical asymptote:** $x = a$ is a vertical asymptote of $f$ if any one-sided limit as $x \to a$ is $\pm\infty$.

**Examples:**
$$\lim_{x \to 0^+} \frac{1}{x} = +\infty, \quad \lim_{x \to 0^-} \frac{1}{x} = -\infty$$
$$\lim_{x \to 0} \frac{1}{x^2} = +\infty \quad \text{(both sides)}$$
$$\lim_{x \to (\pi/2)^-} \tan(x) = +\infty, \quad \lim_{x \to (\pi/2)^+} \tan(x) = -\infty$$

**For AI — log barrier:** Interior point methods use log barriers $-\sum_i \log(s_i)$ where $s_i > 0$. As $s_i \to 0^+$, the barrier $\to +\infty$, keeping the optimization in the feasible interior. This is a vertical asymptote engineered as a constraint.

---

## 3. Fundamental Limits

These limits appear repeatedly in analysis, ML, and numerical methods. Every practitioner should know them cold.

### 3.1 The Sine Limit: sin(x)/x → 1

$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

**Geometric proof:** Consider a unit circle. For $0 < x < \pi/2$, the areas satisfy:
$$\text{Area}(\triangle OAP) \leq \text{Area(sector }OAP) \leq \text{Area}(\triangle OAT)$$

In terms of $x$ (the angle):
$$\frac{\sin x}{2} \leq \frac{x}{2} \leq \frac{\tan x}{2}$$

Dividing by $\sin x / 2 > 0$:
$$1 \leq \frac{x}{\sin x} \leq \frac{1}{\cos x}$$

Taking reciprocals (reversing inequalities):
$$\cos x \leq \frac{\sin x}{x} \leq 1$$

Since $\lim_{x \to 0} \cos x = 1$, by the Squeeze Theorem: $\lim_{x \to 0} \frac{\sin x}{x} = 1$. $\square$

**Consequence:**
$$\lim_{x \to 0} \frac{1 - \cos x}{x} = 0, \quad \lim_{x \to 0} \frac{1 - \cos x}{x^2} = \frac{1}{2}$$

The first follows from $\frac{1-\cos x}{x} = \frac{1-\cos x}{x} \cdot \frac{1+\cos x}{1+\cos x} = \frac{\sin^2 x}{x(1+\cos x)} \to \frac{0}{2} = 0$.

**For AI — attention scores:** The sinusoidal positional encoding in transformers (Vaswani et al., 2017) uses $\sin(k/10000^{2i/d})$. The smooth interpolation behavior of sine (controlled by this limit) ensures smooth positional representations.

### 3.2 Exponential and Logarithmic Limits

**The fundamental exponential limit:**
$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

This is the definition of the derivative of $e^x$ at $x = 0$, and it says the exponential grows at rate $1$ near the origin.

**Proof (via series):** $e^x = 1 + x + x^2/2! + \cdots$, so $\frac{e^x - 1}{x} = 1 + x/2! + x^2/3! + \cdots \to 1$.

**Consequence:**
$$\lim_{x \to 0} \frac{a^x - 1}{x} = \ln a \quad \text{for any } a > 0$$

**Logarithmic limits:**
$$\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1, \quad \lim_{x \to \infty} \frac{\ln x}{x^p} = 0 \text{ for any } p > 0$$

The second says logarithm grows slower than any positive power — critical for understanding why $O(\log n)$ algorithms are so efficient.

**For AI — cross-entropy:** The cross-entropy loss $\mathcal{L} = -\sum_i y_i \log p_i$ uses $\lim_{p \to 0^+} p \log p = 0$ (Section 3.4) to handle the convention $0 \log 0 = 0$.

### 3.3 Euler's Number as a Limit

$$e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = \lim_{x \to 0} (1 + x)^{1/x}$$

Both forms define the same constant $e \approx 2.71828\ldots$.

**Motivation:** If interest on principal $P$ is compounded $n$ times per year at annual rate $r$, after one year the amount is $P(1 + r/n)^n$. As $n \to \infty$ (continuous compounding), this approaches $Pe^r$.

**General form:**
$$\lim_{n \to \infty} \left(1 + \frac{r}{n}\right)^n = e^r, \quad \lim_{x \to 0} (1 + rx)^{1/x} = e^r$$

**Proof sketch:** Let $f(n) = (1 + 1/n)^n$. Taking logarithm: $\ln f(n) = n \ln(1 + 1/n)$. Setting $h = 1/n$: $\ln f = \frac{\ln(1+h)}{h} \to 1$ as $h \to 0$. So $f \to e^1 = e$. $\square$

**For AI — learning rate warm-up:** The cosine schedule and exponential decay are engineering analogs of $(1 + 1/n)^n$ — discrete compounding that approaches a continuous exponential in the limit of fine schedules.

### 3.4 Logarithmic Limits and xln(x)

$$\lim_{x \to 0^+} x \ln x = 0$$

This is a $0 \cdot (-\infty)$ indeterminate form. Rewrite:
$$\lim_{x \to 0^+} x \ln x = \lim_{x \to 0^+} \frac{\ln x}{1/x}$$

This is $-\infty/+\infty$, so L'Hôpital applies:
$$= \lim_{x \to 0^+} \frac{1/x}{-1/x^2} = \lim_{x \to 0^+} (-x) = 0$$

**For AI — entropy:** The Shannon entropy $H(p) = -\sum_i p_i \log p_i$ uses the convention $0 \log 0 = 0$, justified by $\lim_{p \to 0^+} p \log p = 0$. This makes entropy a continuous function of the probability distribution even at the boundary of the simplex.

---

## 4. Computation Techniques

### 4.1 Algebraic Techniques

When direct substitution gives $0/0$ or other indeterminate forms, algebraic manipulation often resolves the indeterminacy.

**Factoring:**
$$\lim_{x \to 2} \frac{x^3 - 8}{x - 2} = \lim_{x \to 2} \frac{(x-2)(x^2+2x+4)}{x-2} = \lim_{x \to 2} (x^2 + 2x + 4) = 4 + 4 + 4 = 12$$

**Rationalization (conjugate multiplication):**
$$\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x} = \lim_{x \to 0} \frac{(\sqrt{x+4}-2)(\sqrt{x+4}+2)}{x(\sqrt{x+4}+2)} = \lim_{x \to 0} \frac{x}{x(\sqrt{x+4}+2)} = \frac{1}{4}$$

**Common factor in rational functions:**
$$\lim_{x \to -3} \frac{x^2 + 5x + 6}{x^2 - x - 12} = \lim_{x \to -3} \frac{(x+2)(x+3)}{(x+3)(x-4)} = \lim_{x \to -3} \frac{x+2}{x-4} = \frac{-1}{-7} = \frac{1}{7}$$

**Multiplying by $1/x^n$** (for rational functions at infinity): already shown in Section 2.4.

### 4.2 L'Hôpital's Rule

**Theorem (L'Hôpital's Rule).** Suppose $\lim_{x \to a} f(x) = 0$ and $\lim_{x \to a} g(x) = 0$ (the $0/0$ form), or both limits are $\pm\infty$ (the $\infty/\infty$ form). If $f$ and $g$ are differentiable near $a$ (but not necessarily at $a$), $g'(x) \neq 0$ near $a$, and $\lim_{x \to a} f'(x)/g'(x)$ exists (or is $\pm\infty$), then:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

The same holds for one-sided limits and for $a = \pm\infty$.

**Critical:** L'Hôpital applies only to $0/0$ or $\infty/\infty$. Never apply it to other forms. Always verify the indeterminate form first.

**Examples:**

$\lim_{x \to 0} \frac{\sin x}{x}$: This is $0/0$. L'Hôpital: $\frac{\cos x}{1} \to 1$. ✓

$\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$: This is $0/0$. L'Hôpital: $\frac{e^x - 1}{2x}$, still $0/0$. Apply again: $\frac{e^x}{2} \to \frac{1}{2}$.

$\lim_{x \to \infty} x e^{-x}$: Rewrite as $\frac{x}{e^x}$ ($\infty/\infty$). L'Hôpital: $\frac{1}{e^x} \to 0$.

**Other indeterminate forms** — convert before applying L'Hôpital:

| Form | Conversion |
|------|-----------|
| $0 \cdot \infty$ | Write as $\frac{f}{1/g}$ or $\frac{g}{1/f}$ (0/0 or ∞/∞) |
| $\infty - \infty$ | Factor or find common denominator |
| $1^\infty$, $0^0$, $\infty^0$ | Take $\ln$: $\lim \ln(f^g) = \lim g \ln f$ |

**Example ($1^\infty$):** $\lim_{x \to 0} (1+x)^{1/x}$.
Let $y = (1+x)^{1/x}$, so $\ln y = \frac{\ln(1+x)}{x}$. This is $0/0$; L'Hôpital gives $\frac{1/(1+x)}{1} \to 1$. So $y \to e^1 = e$. ✓

**Warning:** L'Hôpital can fail to terminate or produce incorrect answers if misapplied. Always check that the form is truly indeterminate, and stop as soon as the limit becomes evaluable by direct substitution.

### 4.3 The Squeeze Theorem

**Theorem (Squeeze / Sandwich Theorem).** Suppose $h(x) \leq f(x) \leq g(x)$ for all $x$ near $a$ (but not necessarily at $a$), and:
$$\lim_{x \to a} h(x) = \lim_{x \to a} g(x) = L$$
Then $\lim_{x \to a} f(x) = L$.

**Proof:** Given $\varepsilon > 0$. Choose $\delta_1, \delta_2 > 0$ so that:
- $0 < \lvert x - a \rvert < \delta_1 \implies \lvert h(x) - L \rvert < \varepsilon$, i.e., $L - \varepsilon < h(x) < L + \varepsilon$
- $0 < \lvert x - a \rvert < \delta_2 \implies \lvert g(x) - L \rvert < \varepsilon$, i.e., $L - \varepsilon < g(x) < L + \varepsilon$

Set $\delta = \min(\delta_1, \delta_2)$. For $0 < \lvert x - a \rvert < \delta$:
$$L - \varepsilon < h(x) \leq f(x) \leq g(x) < L + \varepsilon$$
So $\lvert f(x) - L \rvert < \varepsilon$. $\square$

**Example 1:** $\lim_{x \to 0} x^2 \sin(1/x)$.
Since $-1 \leq \sin(1/x) \leq 1$: $-x^2 \leq x^2 \sin(1/x) \leq x^2$.
Both $-x^2 \to 0$ and $x^2 \to 0$, so $\lim_{x \to 0} x^2 \sin(1/x) = 0$.

**Example 2:** $\lim_{x \to 0} \frac{\sin x}{x} = 1$ (proved geometrically in Section 3.1 using squeeze).

**Example 3:** $\lim_{n \to \infty} \sqrt[n]{n}$.
We know $1 \leq \sqrt[n]{n} \leq 1 + \sqrt{2/n}$ (can be proved via AM-GM). Since $1 + \sqrt{2/n} \to 1$, the squeeze gives $\sqrt[n]{n} \to 1$.

**For AI:** The squeeze theorem is the theoretical justification for proving that "soft" approximations converge to "hard" decisions. For example, if $L_\text{smooth} \leq L_\text{target} \leq L_\text{upper}$ and both bounds converge, then $L_\text{target}$ converges — used in proving convergence of temperature-annealed sampling.

### 4.4 Substitution and Composition

**Theorem.** If $g$ is continuous at $L$ and $\lim_{x \to a} f(x) = L$, then:
$$\lim_{x \to a} g(f(x)) = g\!\left(\lim_{x \to a} f(x)\right) = g(L)$$

This allows moving limits inside continuous functions.

**Examples:**
$$\lim_{x \to 0} e^{\sin x} = e^{\lim_{x \to 0} \sin x} = e^0 = 1$$
$$\lim_{x \to \pi} \cos(x + \pi/2) = \cos\!\left(\lim_{x \to \pi}(x + \pi/2)\right) = \cos(3\pi/2) = 0$$
$$\lim_{x \to 0^+} \ln(\sin x / x) = \ln\!\left(\lim_{x \to 0^+} \frac{\sin x}{x}\right) = \ln(1) = 0$$

---

## 5. Continuity

### 5.1 The Three-Part Definition

**Definition (Continuity at a Point).** A function $f$ is **continuous at $a$** if:
1. $f(a)$ is defined (the function exists at $a$)
2. $\lim_{x \to a} f(x)$ exists (the limit exists at $a$)
3. $\lim_{x \to a} f(x) = f(a)$ (limit equals the function value)

All three conditions must hold. Failure of any one gives a discontinuity.

**Equivalent ε-δ formulation:** $f$ is continuous at $a$ if for every $\varepsilon > 0$ there exists $\delta > 0$ such that $\lvert x - a \rvert < \delta \implies \lvert f(x) - f(a) \rvert < \varepsilon$. (Note: unlike the limit definition, $x = a$ is now permitted.)

**Examples:**
- $f(x) = x^2$: continuous at every $a \in \mathbb{R}$ (polynomial)
- $f(x) = \sin x$: continuous at every $a \in \mathbb{R}$
- $f(x) = 1/x$: continuous at every $a \neq 0$; not defined (hence not continuous) at $0$
- $f(x) = \lfloor x \rfloor$: continuous at every non-integer; discontinuous at every integer

**Non-examples (failing each condition):**
- Condition 1 fails: $f(x) = (x^2-4)/(x-2)$ at $x = 2$ (undefined)
- Condition 2 fails: $f(x) = \sin(1/x)$ at $x = 0$ (limit doesn't exist)
- Condition 3 fails: $f(x) = x^2$ for $x \neq 1$, $f(1) = 5$ (limit is $1$ but $f(1) = 5$)

### 5.2 Types of Discontinuity

```
DISCONTINUITY TAXONOMY
════════════════════════════════════════════════════════════════════════

  REMOVABLE                  JUMP                   ESSENTIAL (INFINITE)
  (Hole in graph)            (Step)                 (Blows up)

     •                       │                       │  ╱
    ╱ ╲         ○            │    ●                  │╱
   ╱   ╲       ╱            ○    │                   │
  ╱     ╲─────╱              │                        │

  lim exists,            lim doesn't exist         lim = ±∞
  ≠ f(a) or              (one-sided limits         (vertical asymptote)
  f(a) undefined         exist but differ)

  Fix: redefine f(a)     Cannot fix: jump          Cannot fix: infinity
  Example: (x²-4)/(x-2)  Example: sgn(x)           Example: 1/x at 0
  at x=2                 at x=0

════════════════════════════════════════════════════════════════════════
```

**Removable discontinuity:** $\lim_{x \to a} f(x)$ exists but either $f(a)$ is undefined or $f(a) \neq \lim_{x\to a}f(x)$. Fixable by redefining $f(a) = \lim_{x \to a} f(x)$.

**Jump discontinuity:** Both one-sided limits exist but $\lim_{x \to a^-} f(x) \neq \lim_{x \to a^+} f(x)$. The function "jumps" at $a$.

**Essential (infinite) discontinuity:** At least one one-sided limit is $\pm\infty$. Also called an **infinite discontinuity**.

**Oscillatory discontinuity** (a subtype of essential): $\sin(1/x)$ at $x = 0$ — neither one-sided limit exists (even as $\pm\infty$).

**For AI:**
- ReLU has no discontinuities (continuous everywhere), but its derivative has a jump at $0$
- Hard attention (argmax) is a jump discontinuity at ties — made differentiable via softmax temperature annealing
- Quantized activations (INT8) introduce jump discontinuities — handled with straight-through estimator in training

### 5.3 Continuity on Intervals

**Definition.** $f$ is **continuous on the open interval $(a,b)$** if $f$ is continuous at every $c \in (a,b)$.

$f$ is **continuous on the closed interval $[a,b]$** if:
- $f$ is continuous on $(a,b)$
- $\lim_{x \to a^+} f(x) = f(a)$ (right-continuous at $a$)
- $\lim_{x \to b^-} f(x) = f(b)$ (left-continuous at $b$)

**Standard continuous functions:** Polynomials, rational functions (on their domains), $\sin x$, $\cos x$, $e^x$, $\ln x$ (on $(0,\infty)$), $\sqrt{x}$ (on $[0,\infty)$), and any composition thereof.

**For AI:** Neural networks with continuous activation functions (sigmoid, tanh, GELU, SiLU) define continuous functions $\mathbb{R}^n \to \mathbb{R}^m$. Networks with ReLU are piecewise linear and continuous. Continuity of the network function is necessary (though not sufficient) for gradient-based optimization to work reliably.

### 5.4 Operations Preserving Continuity

If $f$ and $g$ are continuous at $a$, then so are:
- $f + g$, $f - g$, $f \cdot g$, $c \cdot f$ (for any constant $c$)
- $f / g$ provided $g(a) \neq 0$
- $f \circ g$ provided $f$ is continuous at $g(a)$

**Consequence:** Every function built by composing, adding, multiplying elementary continuous functions is continuous on its natural domain. This covers virtually all standard functions in analysis.

---

## 6. Key Theorems

### 6.1 Intermediate Value Theorem

**Theorem (IVT).** Let $f$ be continuous on the closed interval $[a, b]$. If $k$ is any value strictly between $f(a)$ and $f(b)$, then there exists $c \in (a, b)$ such that $f(c) = k$.

Informally: a continuous function cannot skip values — if it goes from $f(a)$ to $f(b)$, it must pass through every intermediate value.

```
INTERMEDIATE VALUE THEOREM
════════════════════════════════════════════════════════════════════════

  f(x)
    │
 f(b)┤           ●  ← f(b)
    │          ╱
    k┤ - - - ╱ ← f(c) = k (guaranteed to exist)
    │       ╱
 f(a)┤  ●  ╱  ← f(a)
    │    c
    └────┬───────┬──────→ x
         a               b

  If f is continuous on [a,b], then for any k between f(a) and f(b),
  there exists c ∈ (a,b) with f(c) = k.

════════════════════════════════════════════════════════════════════════
```

**Proof sketch (requires completeness of $\mathbb{R}$):** Let $S = \{x \in [a,b] : f(x) < k\}$. If $f(a) < k < f(b)$, then $S$ is nonempty (contains $a$) and bounded above (by $b$). Let $c = \sup S$. By continuity of $f$ at $c$, one can show $f(c) = k$. $\square$

**Applications:**

1. **Root finding (bisection method):** If $f(a) < 0 < f(b)$ and $f$ is continuous, then $f$ has a root in $(a, b)$. Bisect: evaluate $f((a+b)/2)$ and recurse into the half-interval where sign changes. Converges in $O(\log_2(1/\varepsilon))$ steps.

2. **Fixed-point existence:** If $f: [0,1] \to [0,1]$ is continuous, then $f$ has a fixed point $f(c) = c$. (Apply IVT to $g(x) = f(x) - x$: $g(0) = f(0) \geq 0$ and $g(1) = f(1) - 1 \leq 0$.)

3. **Equilibrium existence:** Used in proving existence of Nash equilibria (via Kakutani fixed-point theorem), stopping times for Brownian motion, and zero crossings of residuals in iterative solvers.

**For AI:** The bisection method is the simplest root-finding algorithm. Modern AI uses line search (Wolfe conditions) and trust-region methods — all of which rely on IVT guarantees to find step sizes satisfying sufficient decrease conditions.

### 6.2 Extreme Value Theorem

**Theorem (EVT).** Let $f$ be continuous on the closed interval $[a, b]$. Then $f$ attains its maximum and minimum on $[a, b]$: there exist $c, d \in [a, b]$ such that $f(d) \leq f(x) \leq f(c)$ for all $x \in [a, b]$.

**Why compactness matters:** The EVT requires $[a,b]$ to be closed and bounded (compact). Counterexamples on non-compact domains:
- $f(x) = x$ on $(0, 1)$: no maximum (approaches $1$ but never attains it)
- $f(x) = 1/x$ on $(0, 1]$: no maximum (blows up as $x \to 0^+$)

**For AI:** Loss functions trained over bounded parameter spaces (after gradient clipping or weight constraints) attain their minimum. This theoretical guarantee underpins the well-posedness of constrained optimization problems in ML.

### 6.3 Uniform Continuity

**Definition.** $f$ is **uniformly continuous** on $S$ if for every $\varepsilon > 0$ there exists $\delta > 0$ such that for all $x, y \in S$: $\lvert x - y \rvert < \delta \implies \lvert f(x) - f(y) \rvert < \varepsilon$.

**Key distinction:** In pointwise continuity, $\delta$ can depend on both $\varepsilon$ and the point $a$. In uniform continuity, $\delta$ depends only on $\varepsilon$ — the same $\delta$ works everywhere.

**Theorem (Heine-Cantor).** If $f$ is continuous on a closed bounded interval $[a,b]$, then $f$ is uniformly continuous on $[a,b]$.

**Examples:**
- $f(x) = x^2$ is uniformly continuous on $[0, 1]$ but not on $\mathbb{R}$ (the derivative grows without bound)
- $f(x) = \sin x$ is uniformly continuous on $\mathbb{R}$ (bounded derivative)
- $f(x) = 1/x$ is not uniformly continuous on $(0, 1)$ (derivative blows up near $0$)

**For AI:** Lipschitz continuity (a stronger condition: $\lvert f(x) - f(y) \rvert \leq L \lvert x - y \rvert$) implies uniform continuity. Lipschitz constraints appear in spectral normalization (Miyato et al., 2018) for GANs and in Wasserstein GAN training — all are quantitative versions of uniform continuity.

---

## 7. Asymptotic Behavior

### 7.1 Horizontal and Vertical Asymptotes

**Horizontal asymptote:** $y = L$ is a horizontal asymptote if $\lim_{x \to +\infty} f(x) = L$ or $\lim_{x \to -\infty} f(x) = L$.

**Vertical asymptote:** $x = a$ is a vertical asymptote if any one-sided limit as $x \to a$ is $\pm\infty$.

**Oblique (slant) asymptote:** $y = mx + b$ is an oblique asymptote if $\lim_{x \to \pm\infty} [f(x) - (mx+b)] = 0$. Occurs when $\deg(\text{numerator}) = \deg(\text{denominator}) + 1$.

**Example:** $f(x) = \frac{x^2 + 1}{x - 1}$. Polynomial division: $f(x) = x + 1 + \frac{2}{x-1}$. As $x \to \pm\infty$: $f(x) - (x+1) = \frac{2}{x-1} \to 0$. So $y = x + 1$ is an oblique asymptote and $x = 1$ is a vertical asymptote.

### 7.2 Polynomial vs. Exponential Growth

**Growth hierarchy (as $x \to +\infty$):**
$$\ln x \ll x^p \ll e^x \ll x^x$$

Formally:
$$\lim_{x \to \infty} \frac{\ln x}{x^p} = 0 \quad (p > 0), \qquad \lim_{x \to \infty} \frac{x^n}{e^x} = 0 \quad (n > 0)$$

Every polynomial is dominated by every exponential; every logarithm is dominated by every positive power.

**Proof of $x^n/e^x \to 0$:** Apply L'Hôpital $n$ times: $\frac{x^n}{e^x} \to \frac{n!}{e^x} \to 0$.

**For AI — complexity:** Algorithm complexity classes: $O(\log n)$ (logarithm) $\ll O(n^k)$ (polynomial) $\ll O(e^n)$ (exponential). This hierarchy, grounded in limit theory, governs which algorithms scale to LLM-scale data (e.g., transformers are $O(n^2 d)$ for sequence length $n$ and dimension $d$).

### 7.3 Big-O and Little-o as Limit Statements

**Definition.** $f(x) = O(g(x))$ as $x \to a$ means there exist $C, \delta > 0$ such that $\lvert f(x) \rvert \leq C \lvert g(x) \rvert$ for $\lvert x - a \rvert < \delta$.

**Definition.** $f(x) = o(g(x))$ as $x \to a$ means $\lim_{x \to a} \frac{f(x)}{g(x)} = 0$.

In words: $O$ means "bounded by a multiple"; $o$ means "negligible compared to."

**Examples:**
- $\sin x = O(x)$ as $x \to 0$ (since $\lvert \sin x \rvert \leq \lvert x \rvert$)
- $x^2 = o(x)$ as $x \to 0$ (since $x^2/x = x \to 0$)
- $e^x - 1 = x + O(x^2)$ as $x \to 0$ (Taylor expansion)

**For AI:** The Adam optimizer has per-parameter learning rate $\alpha / (\sqrt{\hat{v}} + \varepsilon)$. The $\varepsilon$ term prevents division by zero when $\hat{v} \approx 0$. In the limit $\hat{v} \to 0$, the effective learning rate is $O(\alpha/\varepsilon)$; for large $\hat{v}$, it is $O(\alpha/\sqrt{\hat{v}})$. This piecewise asymptotic analysis (via big-O) explains Adam's behavior in both sparse and dense gradient regimes.

---

## 8. Numerical Stability Near Limits

### 8.1 Catastrophic Cancellation

When computing $(e^x - 1)/x$ for $x$ close to $0$, the numerator involves subtracting two nearly equal quantities. In IEEE 754 double precision (64-bit float, $u \approx 1.1 \times 10^{-16}$), numbers are stored to $\sim 16$ significant digits. If $e^x = 1 + x + x^2/2 + \ldots$ and $x = 10^{-8}$, then:
- $e^x \approx 1.0000000100000000050\ldots$
- $1.0$ in floating point: exactly $1.0$
- $e^x - 1$ in floating point: only the digits beyond the $1$ are meaningful, losing $\sim 8$ digits

The error in the subtraction propagates, giving up to 8 digits of lost precision. The limit value (which is $1$) is computed incorrectly.

```
CATASTROPHIC CANCELLATION EXAMPLE
════════════════════════════════════════════════════════════════════════

  x = 1e-8 (IEEE 754 double)

  True value: (e^x - 1)/x  ≈  1.000000005000000017...

  Naive computation:
    e^x   = 1.00000001000000002   (15-16 sig digits, ok)
    e^x-1 = 0.00000001000000002   (only ~9 sig digits now!)
    /x    ≈  1.000000002          (lost ~8 digits of precision)

  Stable computation (expm1):
    expm1(x) = 1.0000000050000000...   (full precision maintained)
    /x       = 1.0000000050000000      (correct!)

════════════════════════════════════════════════════════════════════════
```

### 8.2 Numerically Stable Alternatives

**`numpy.expm1(x)` = $e^x - 1$:** Computed using a special algorithm that avoids cancellation for small $x$.

**`numpy.log1p(x)` = $\ln(1 + x)$:** Similarly avoids cancellation when $x \approx 0$.

**`scipy.special.log_softmax(x)`:** Avoids overflow in $e^{x_i}$ by computing $x_i - \log\sum_j e^{x_j}$ via the log-sum-exp trick:
$$\log\sum_j e^{x_j} = m + \log\sum_j e^{x_j - m}, \quad m = \max_j x_j$$

All $e^{x_j - m} \leq 1$, preventing overflow.

**Stable sigmoid:** $\sigma(x) = 1/(1+e^{-x})$ can overflow $e^{-x}$ for large negative $x$. Stable version:
$$\sigma(x) = \begin{cases} 1/(1+e^{-x}) & x \geq 0 \\ e^x/(1+e^x) & x < 0 \end{cases}$$

**For AI — loss computation:** Cross-entropy loss using `torch.nn.CrossEntropyLoss` internally uses `log_softmax` for numerical stability. Implementing naive `log(softmax(x))` instead causes overflow/underflow for large logits.

### 8.3 Machine Epsilon and Floating-Point Limits

**Machine epsilon** $u = 2^{-52} \approx 2.22 \times 10^{-16}$ (double precision): the smallest $\varepsilon$ such that $\text{fl}(1 + \varepsilon) > 1$ in IEEE 754.

In floating-point arithmetic, the limit $\lim_{h \to 0}$ cannot be taken exactly — there is a practical lower bound on $h$ below which finite difference approximations deteriorate:

- For first-order finite differences: optimal $h \approx \sqrt{u} \approx 10^{-8}$
- For second-order: optimal $h \approx u^{1/3} \approx 10^{-5}$

This is why gradient checking in ML uses $h = 10^{-5}$ or $10^{-7}$ rather than $10^{-15}$.

---

## 9. Machine Learning Applications

### 9.1 Softmax Temperature Limits

The **softmax with temperature $T > 0$** is:
$$\text{softmax}_T(\mathbf{z})_i = \frac{e^{z_i / T}}{\sum_{j=1}^n e^{z_j / T}}$$

**Limit as $T \to 0^+$ (cold / sharp):**
$$\lim_{T \to 0^+} \text{softmax}_T(\mathbf{z})_i = \begin{cases} 1 & \text{if } i = \arg\max_j z_j \\ 0 & \text{otherwise} \end{cases}$$
(assuming unique argmax). The distribution collapses to a point mass on the highest-scoring class. This is why $T \to 0$ is called the "deterministic" or "greedy" limit.

**Limit as $T \to \infty$ (hot / uniform):**
$$\lim_{T \to \infty} \text{softmax}_T(\mathbf{z})_i = \frac{1}{n} \quad \text{for all } i$$
All logits become irrelevant; the distribution becomes uniform.

**Proof (cold limit):** WLOG $z_1 > z_j$ for $j > 1$. Factor out $e^{z_1/T}$:
$$\text{softmax}_T(\mathbf{z})_1 = \frac{1}{1 + \sum_{j>1} e^{(z_j - z_1)/T}}$$
As $T \to 0^+$: $(z_j - z_1)/T \to -\infty$ (since $z_j - z_1 < 0$), so $e^{(z_j-z_1)/T} \to 0$. Thus the denominator $\to 1$.

**For AI:** In LLM inference (GPT, LLaMA, etc.), temperature controls creativity vs. determinism. Temperature $T = 1$ is trained distribution; $T < 1$ sharpens (more predictable); $T > 1$ flattens (more diverse/random). The limits above show $T \to 0$ recovers beam search top-1, and $T \to \infty$ recovers pure random sampling.

### 9.2 Sigmoid Saturation and Vanishing Gradients

The sigmoid function $\sigma: \mathbb{R} \to (0, 1)$:
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Limits at infinity:**
$$\lim_{x \to +\infty} \sigma(x) = 1, \quad \lim_{x \to -\infty} \sigma(x) = 0$$

**Derivative:**
$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Limits of derivative:**
$$\lim_{x \to \pm\infty} \sigma'(x) = 0$$

The derivative at saturation is zero, not just small — the function is completely flat in the limit.

**Vanishing gradient mechanism:** In a deep network with sigmoid activations, the gradient of loss with respect to layer $l$'s weights involves products of $\sigma'(x^{[l]})$ across all layers $l+1, \ldots, L$. If each factor is $\leq 0.25$ (the max of $\sigma(1-\sigma)$), then for $L - l = 20$ layers: $(0.25)^{20} \approx 10^{-12}$. The gradient effectively vanishes — Hochreiter & Schmidhuber (1997) identified this as the key failure mode of early RNNs.

**Fix:** ReLU activations have $\lim_{x \to +\infty} \text{ReLU}'(x) = 1$ (no saturation for positive inputs). GELU and SiLU have smoother saturation profiles. Layer normalization and residual connections also mitigate the issue at the architecture level.

### 9.3 ReLU and GELU: Continuity at the Origin

**ReLU:** $\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$

Continuity check at $x = 0$:
$$\lim_{x \to 0^-} \text{ReLU}(x) = 0, \quad \lim_{x \to 0^+} \text{ReLU}(x) = 0, \quad \text{ReLU}(0) = 0$$
All equal: ReLU is **continuous** everywhere.

Derivative at $x = 0$: $\lim_{h \to 0^+} \frac{\text{ReLU}(h) - 0}{h} = 1$ but $\lim_{h \to 0^-} \frac{0 - 0}{h} = 0$. One-sided derivatives differ: ReLU is **not differentiable** at $0$. This is a removable issue in practice — the subgradient $\{0\}$ or $\{1\}$ (or $1/2$) is used.

**GELU:** $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF.

$$\lim_{x \to 0} \text{GELU}(x) = 0 \cdot \Phi(0) = 0 \cdot \frac{1}{2} = 0 = \text{GELU}(0)$$
$$\text{GELU}'(x) = \Phi(x) + x\phi(x)$$
$$\lim_{x \to 0} \text{GELU}'(x) = \frac{1}{2} + 0 = \frac{1}{2}$$

GELU is smooth ($C^\infty$) at the origin — no kink, unlike ReLU. This is why GELU-based networks (BERT, GPT-2 onward) train more smoothly in practice.

**GELU vs. ReLU continuity comparison:**

| Property | ReLU | GELU |
|---|---|---|
| Continuous | Yes (everywhere) | Yes (everywhere) |
| Differentiable at 0 | No (jump in derivative) | Yes ($C^\infty$) |
| Saturates for $x \to -\infty$ | Gradient = 0 (dying ReLU) | Gradient $\to 0$ (soft) |
| Saturates for $x \to +\infty$ | Gradient = 1 (no saturation) | Gradient $\to 1$ (no saturation) |

### 9.4 Learning Rate Decay: Robbins-Monro Conditions

The stochastic gradient descent update $\theta_{t+1} = \theta_t - \alpha_t \nabla_\theta \mathcal{L}_t(\theta_t)$ converges (under convexity and bounded variance assumptions) if and only if the learning rate schedule $\{\alpha_t\}$ satisfies:

$$\sum_{t=1}^\infty \alpha_t = \infty \qquad \text{and} \qquad \sum_{t=1}^\infty \alpha_t^2 < \infty$$

(Robbins & Monro, 1951)

**Interpretation:**
- $\sum \alpha_t = \infty$: the total step size is infinite — SGD can travel arbitrarily far and cannot get stuck permanently
- $\sum \alpha_t^2 < \infty$: the noise contribution (proportional to $\alpha_t^2$) is summable — gradient noise eventually becomes negligible

**Standard schedules:**

| Schedule | $\alpha_t$ | $\sum \alpha_t$ | $\sum \alpha_t^2$ | Satisfies RM? |
|---|---|---|---|---|
| Constant | $\alpha$ | $\infty$ | $\infty$ | No (second fails) |
| $1/t$ | $\alpha / t$ | $\infty$ (harmonic) | $\pi^2/6 < \infty$ | Yes |
| $1/\sqrt{t}$ | $\alpha / \sqrt{t}$ | $\infty$ | $\infty$ | No (second fails) |
| Exponential | $\alpha \cdot c^t$ | $\alpha/(1-c) < \infty$ | Finite | No (first fails) |

Only $1/t$ satisfies both — which is why it's the canonical theoretical schedule, even though practitioners prefer others (cosine, warmup) for practical reasons.

### 9.5 Gradient as a Limit

The derivative (Section 9, forward reference: [§02-Derivatives](../02-Derivatives-and-Differentiation/notes.md)) is defined as a limit:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

This limit must exist (be finite, with equal one-sided limits) for $f$ to be differentiable at $a$.

**For AI — automatic differentiation:** PyTorch's `autograd` and JAX's `grad` do not compute this limit numerically. Instead, they apply the **chain rule** symbolically/computationally. But the mathematical object they compute is precisely this limit — evaluated at each intermediate node in the computation graph.

**Numerical gradient checking:** To verify a gradient implementation, compute:
$$\frac{f(\theta + h \mathbf{e}_i) - f(\theta - h \mathbf{e}_i)}{2h} \approx \frac{\partial f}{\partial \theta_i}$$
with $h = 10^{-5}$. The centered finite difference has error $O(h^2)$ vs. $O(h)$ for one-sided — both are limit approximations, with the centered form being more accurate.

> **Preview: Derivatives**
> The limit $f'(a) = \lim_{h\to 0}[f(a+h)-f(a)]/h$ defines the derivative — the instantaneous rate of change. All differentiation rules (product, chain, etc.) and backpropagation follow from this limit.
>
> → _Full treatment: [Derivatives and Differentiation](../02-Derivatives-and-Differentiation/notes.md)_

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Evaluating $\lim_{x \to a} f(x)$ by computing $f(a)$ when $f$ is discontinuous at $a$ | The limit is about neighborhood behavior, not the point value; $f(a)$ may not exist or may differ | Always check whether $f$ is continuous at $a$ first; if not, compute the limit separately |
| 2 | Concluding the limit doesn't exist because $f(a)$ is undefined | Limits are independent of $f(a)$; a removable discontinuity has a well-defined limit | Compute the limit algebraically; undefined $f(a)$ does not imply no limit |
| 3 | Applying L'Hôpital's Rule to non-indeterminate forms | L'Hôpital is only valid for $0/0$ or $\pm\infty/\pm\infty$; other forms give wrong answers | Check: substitute and verify the form is indeterminate before applying L'Hôpital |
| 4 | Applying L'Hôpital repeatedly without checking if the form is still indeterminate | After one application, direct substitution may be possible; repeated L'Hôpital may cycle | After each application, try direct substitution before applying again |
| 5 | Confusing one-sided and two-sided limits | The two-sided limit requires both one-sided limits to exist and be equal | Compute $\lim^-$ and $\lim^+$ separately; two-sided limit exists iff they agree |
| 6 | Asserting $\lim_{x\to\infty}f(x) = \infty$ means the limit exists | $\pm\infty$ are not real numbers; saying the limit is $\infty$ means divergence, not existence | Distinguish between "limit exists (finite)" and "limit is $\pm\infty$" |
| 7 | Assuming all three parts of continuity hold because $f$ looks smooth | Continuity requires all three conditions; a piecewise-defined function may fail condition 3 | Verify all three: $f(a)$ defined, limit exists, limit equals $f(a)$ |
| 8 | Computing $(e^x - 1)/x$ naively near $x = 0$ in code | Catastrophic cancellation destroys 8+ digits of precision for small $x$ | Use `numpy.expm1(x) / x` for numerically stable computation |
| 9 | Applying the Squeeze Theorem without verifying the inequalities hold near $a$ | The bounds must hold in a punctured neighborhood of $a$, not just at one point | Carefully verify $h(x) \leq f(x) \leq g(x)$ near $a$; sketch or prove each bound |
| 10 | Assuming sigmoid saturation means gradient is small but nonzero | $\sigma'(x) \to 0$ exactly as $x \to \pm\infty$, not just approximately | For deep networks, gradient multiplied across many saturated layers goes to zero exponentially fast |

---

## 11. Exercises

Eight graded exercises with worked solutions in [exercises.ipynb](exercises.ipynb).

| # | Difficulty | Topic | Parts |
|---|-----------|-------|-------|
| 1 | ★ | Basic limit computation: factoring, trig, conjugates | (a)-(c) |
| 2 | ★ | One-sided limits and existence | (a)-(b) |
| 3 | ★★ | L'Hôpital's Rule: three indeterminate forms | (a)-(c) |
| 4 | ★★ | Continuity analysis: classify discontinuities | (a)-(c) |
| 5 | ★★ | Squeeze Theorem: prove $x^2\sin(1/x)\to 0$ and verify IVT | (a)-(b) |
| 6 | ★★ | ε-δ proof: verify $\lim_{x\to 2}(3x-1)=5$ and find explicit $\delta(\varepsilon)$ | (a)-(c) |
| 7 | ★★★ | Gradient as a limit: numerical vs. analytic finite differences | (a)-(c) |
| 8 | ★★★ | Cross-entropy limit: $\lim_{p\to 0^+} p\log p$ and entropy continuity | (a)-(c) |

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI / LLM Application | Specific Example |
|---|---|---|
| ε-δ limits | Foundation of automatic differentiation | PyTorch `autograd`, JAX `grad` implement the limit $f'(a) = \lim_{h\to 0}(f(a+h)-f(a))/h$ algebraically |
| One-sided limits | Derivative of ReLU at origin | Subgradient at $x=0$ chosen from $[0,1]$; training stable because ReLU is continuous |
| Squeeze Theorem | Convergence proofs for SGD | Bounding optimization error between upper and lower envelopes of the loss landscape |
| Continuity | Activation function design | GELU / SiLU chosen over hard ReLU for $C^\infty$ smoothness; improves gradient flow |
| IVT | Loss surface root finding | Line search algorithms (backtracking, Wolfe conditions) rely on IVT to guarantee sufficient decrease |
| EVT | Well-posedness of optimization | Constrained optimization over compact parameter sets attains minimum by EVT |
| Limits at infinity | Vanishing/exploding gradients | $\sigma'(x)\to 0$ as $x\to\pm\infty$: mathematical root of LSTM/ResNet motivation |
| $\lim_{T\to 0}\text{softmax}_T$ | Temperature-scaled LLM decoding | Temperature in GPT-4, LLaMA-3 inference controls creativity vs. accuracy |
| $\lim x\ln x = 0$ | Shannon entropy convention | $0\log 0 = 0$ makes cross-entropy continuous; used in every classification loss |
| Catastrophic cancellation | Numerical stability in training | `torch.nn.CrossEntropyLoss` uses `log_softmax` to avoid overflow — a direct fix for limit instability |
| Robbins-Monro conditions | SGD learning rate theory | Cosine decay + warmup approximates RM conditions while being practically efficient |
| Big-O at infinity | Transformer complexity | Attention is $O(n^2 d)$; linear attention approximations are $O(nd)$ — limit theory quantifies the gap |

---

## Conceptual Bridge

**Looking backward:** Limits formalize the intuition of "approaching" that appears throughout earlier sections. The real number system (§01-Mathematical-Foundations) provides the completeness axiom — every nonempty bounded set has a supremum — which is what makes limits well-defined. Without $\mathbb{R}$ being complete, Cauchy sequences might not converge, and the entire limit framework would collapse. Linear algebra (§02-03) introduced matrix norms $\lVert A \rVert$; limit-based analysis of norm sequences $\lVert A^k \rVert$ underlies the study of matrix powers and iterative methods.

**Looking forward:** Limits are the gateway to all of calculus. The derivative (§02-Derivatives-and-Differentiation) is defined as $\lim_{h\to 0}[f(a+h)-f(a)]/h$ — every differentiation rule follows from properties of limits. The integral (§03-Integration) is a limit of Riemann sums. Series (§04-Series-and-Sequences) are limits of partial sums; their convergence is governed by limit tests (ratio, root, comparison). In multivariate calculus (§05), limits extend to functions of several variables, and continuity becomes the prerequisite for partial derivatives and the multivariable chain rule — the mathematical core of backpropagation.

Beyond calculus, functional analysis (§12) studies limits in infinite-dimensional spaces (Banach and Hilbert spaces), and the operator norm $\lVert T \rVert$ is defined as $\sup_{\lVert x \rVert \leq 1} \lVert Tx \rVert$ — a supremum, hence a limit. Measure theory (§24) defines integration via limits of simple functions. The entire edifice of continuous mathematics rests on the foundation laid here.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01-Mathematical-Foundations        §02-Linear-Algebra-Basics
  (Number systems, functions)    ──►  (Vectors, matrices, norms)
           │                                      │
           │                                      │
           ▼                                      ▼
  ┌─────────────────────────────────────────────────────────────┐
  │          §04-01: LIMITS AND CONTINUITY (YOU ARE HERE)       │
  │                                                             │
  │  ε-δ limits · Limit laws · One-sided limits                 │
  │  L'Hôpital · Squeeze Theorem · Fundamental limits           │
  │  Continuity · IVT · EVT · Asymptotic behavior               │
  │  Numerical stability · ML applications                      │
  └─────────────────────────────────────────────────────────────┘
           │                              │
           │                              │
           ▼                              ▼
  §04-02: Derivatives            §04-04: Series & Sequences
  (Derivative as limit,          (Partial sums as limits,
   chain rule, backprop)          Taylor series, convergence)
           │
           │
           ▼
  §05: Multivariate Calculus
  (Partial derivatives, gradient,
   chain rule in ℝⁿ, Jacobian)
           │
           ▼
  §08: Optimization
  (Gradient descent, Newton's method,
   convergence guarantees)

════════════════════════════════════════════════════════════════════════
```

---

[← Back to Calculus Fundamentals](../README.md) | [Next: Derivatives and Differentiation →](../02-Derivatives-and-Differentiation/notes.md)

---

## Appendix A: Extended Proofs and Derivations

### A.1 Proving Limit Laws from ε-δ

**Theorem (Product Law).** If $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$, then $\lim_{x \to a} f(x)g(x) = LM$.

**Proof.** The key trick is to write:
$$f(x)g(x) - LM = f(x)g(x) - Lg(x) + Lg(x) - LM = g(x)[f(x) - L] + L[g(x) - M]$$

Given $\varepsilon > 0$. Since $g(x) \to M$, $g$ is bounded near $a$: choose $\delta_0$ so that $\lvert g(x) - M \rvert < 1$ implies $\lvert g(x) \rvert \leq \lvert M \rvert + 1$ for $\lvert x - a \rvert < \delta_0$.

Set $B = \lvert M \rvert + 1$. Choose:
- $\delta_1$: $\lvert x - a \rvert < \delta_1 \implies \lvert f(x) - L \rvert < \frac{\varepsilon}{2B}$
- $\delta_2$: $\lvert x - a \rvert < \delta_2 \implies \lvert g(x) - M \rvert < \frac{\varepsilon}{2(\lvert L \rvert + 1)}$

Set $\delta = \min(\delta_0, \delta_1, \delta_2)$. Then for $0 < \lvert x-a \rvert < \delta$:
$$\lvert f(x)g(x) - LM \rvert \leq \lvert g(x) \rvert \lvert f(x) - L \rvert + \lvert L \rvert \lvert g(x) - M \rvert < B \cdot \frac{\varepsilon}{2B} + \lvert L \rvert \cdot \frac{\varepsilon}{2(\lvert L \rvert+1)} < \frac{\varepsilon}{2} + \frac{\varepsilon}{2} = \varepsilon \quad \square$$

**Theorem (Composition Law).** If $\lim_{x \to a} f(x) = L$ and $g$ is continuous at $L$, then $\lim_{x \to a} g(f(x)) = g(L)$.

**Proof.** Given $\varepsilon > 0$. Since $g$ is continuous at $L$: $\exists \eta > 0$ such that $\lvert u - L \rvert < \eta \implies \lvert g(u) - g(L) \rvert < \varepsilon$.

Since $\lim_{x \to a} f(x) = L$: $\exists \delta > 0$ such that $0 < \lvert x - a \rvert < \delta \implies \lvert f(x) - L \rvert < \eta$.

Combining: $0 < \lvert x - a \rvert < \delta \implies \lvert f(x) - L \rvert < \eta \implies \lvert g(f(x)) - g(L) \rvert < \varepsilon$. $\square$

### A.2 The Cauchy Criterion for Limits

An equivalent characterization that avoids specifying the limit value:

**Theorem (Cauchy Criterion).** $\lim_{x \to a} f(x)$ exists (as a finite number) if and only if: for every $\varepsilon > 0$ there exists $\delta > 0$ such that for all $x, y$ with $0 < \lvert x-a \rvert < \delta$ and $0 < \lvert y-a \rvert < \delta$:
$$\lvert f(x) - f(y) \rvert < \varepsilon$$

This is useful when you suspect a limit exists but don't know its value.

### A.3 Sequential Characterization of Limits

**Theorem (Heine's Theorem).** $\lim_{x \to a} f(x) = L$ if and only if for every sequence $(x_n)$ with $x_n \to a$ and $x_n \neq a$ for all $n$, we have $f(x_n) \to L$.

**Uses of sequential characterization:**
1. **Proving limits exist:** Find sequences that suggest the limit, then verify
2. **Proving limits don't exist:** Find two sequences $x_n \to a$ and $y_n \to a$ such that $f(x_n) \to L_1 \neq L_2 \leftarrow f(y_n)$

**Example (limit doesn't exist via sequences):** $\lim_{x \to 0} \sin(1/x)$.
- Take $x_n = \frac{1}{2\pi n}$: then $\sin(1/x_n) = \sin(2\pi n) = 0 \to 0$
- Take $y_n = \frac{1}{\pi/2 + 2\pi n}$: then $\sin(1/y_n) = \sin(\pi/2 + 2\pi n) = 1 \to 1$

Since $0 \neq 1$, the limit does not exist. $\square$

### A.4 Proof of the Intermediate Value Theorem

**Full Proof.** Let $f: [a,b] \to \mathbb{R}$ be continuous, $f(a) < k < f(b)$. We show $\exists c \in (a,b)$ with $f(c) = k$.

Define $S = \{x \in [a,b] : f(x) < k\}$. Then:
- $S \neq \emptyset$: $a \in S$ since $f(a) < k$
- $S$ is bounded above: by $b$

Let $c = \sup S$. We claim $f(c) = k$.

**Case $f(c) < k$:** By continuity, $\exists \delta_1 > 0$ such that $\lvert x - c \rvert < \delta_1 \implies f(x) < k$. So $(c, c + \delta_1) \cap [a,b] \subseteq S$, meaning $c$ is not an upper bound for $S$ — contradiction.

**Case $f(c) > k$:** By continuity, $\exists \delta_2 > 0$ such that $\lvert x - c \rvert < \delta_2 \implies f(x) > k$. So $(c - \delta_2, c) \cap [a,b]$ contains no elements of $S$, meaning $c - \delta_2/2$ is a smaller upper bound — contradiction.

Therefore $f(c) = k$. Note $c \neq a$ (since $f(a) < k$) and $c \neq b$ (since $f(b) > k$), so $c \in (a,b)$. $\square$

**Key insight:** The proof uses the completeness of $\mathbb{R}$ (every nonempty bounded set has a supremum) — IVT fails for functions $f: \mathbb{Q} \to \mathbb{Q}$ on rationals because $\mathbb{Q}$ is not complete.

### A.5 Limit Superior and Limit Inferior

For a deeper understanding of oscillating limits, we introduce:

**Definition.** The **limit superior** is:
$$\limsup_{x \to a} f(x) = \lim_{\delta \to 0^+} \sup_{0 < \lvert x-a \rvert < \delta} f(x)$$

The **limit inferior** is:
$$\liminf_{x \to a} f(x) = \lim_{\delta \to 0^+} \inf_{0 < \lvert x-a \rvert < \delta} f(x)$$

**Theorem.** $\lim_{x \to a} f(x)$ exists and equals $L$ if and only if:
$$\limsup_{x \to a} f(x) = \liminf_{x \to a} f(x) = L$$

**Example:** For $f(x) = \sin(1/x)$:
- $\limsup_{x \to 0} f(x) = 1$ (achieved along $x_n = 1/(π/2 + 2\pi n)$)
- $\liminf_{x \to 0} f(x) = -1$ (achieved along $y_n = 1/(-\pi/2 + 2\pi n)$ for large $n$)

Since $1 \neq -1$, the limit does not exist — confirming our earlier result.

**For AI:** Limsup and liminf appear in the analysis of optimization algorithms. The limsup of gradient norms determines whether an algorithm has "bounded oscillation" — a prerequisite for convergence. In AdaGrad and Adam, the effective learning rate limsup is controlled by the accumulated second moment.

---

## Appendix B: Worked Examples with Full Solutions

### B.1 Twelve Limit Computations

Work through each, identifying the technique before computing.

**1.** $\displaystyle\lim_{x \to 0} \frac{\tan x}{x}$

Form: $0/0$. Since $\tan x = \sin x / \cos x$:
$$\frac{\tan x}{x} = \frac{\sin x}{x} \cdot \frac{1}{\cos x} \to 1 \cdot \frac{1}{1} = 1$$

**2.** $\displaystyle\lim_{x \to \infty} \left(\sqrt{x^2 + x} - x\right)$

Form: $\infty - \infty$. Rationalize:
$$\sqrt{x^2+x} - x = \frac{(x^2+x) - x^2}{\sqrt{x^2+x}+x} = \frac{x}{\sqrt{x^2+x}+x} = \frac{1}{\sqrt{1+1/x}+1} \to \frac{1}{2}$$

**3.** $\displaystyle\lim_{x \to 0^+} x^x$

Form: $0^0$. Let $y = x^x = e^{x\ln x}$. We showed $x\ln x \to 0$, so $y \to e^0 = 1$.

**4.** $\displaystyle\lim_{x \to \pi/2} \frac{\cos x}{x - \pi/2}$

Substitute $u = x - \pi/2$ (so $\cos x = \cos(u + \pi/2) = -\sin u$):
$$\lim_{u \to 0} \frac{-\sin u}{u} = -1$$

**5.** $\displaystyle\lim_{n \to \infty} n^{1/n}$

Let $y = n^{1/n} = e^{(\ln n)/n}$. Since $(\ln n)/n \to 0$ (log grows slower than linear):
$$y \to e^0 = 1$$

**6.** $\displaystyle\lim_{x \to 0} \frac{e^{3x} - e^{2x}}{x}$

Factor: $\frac{e^{2x}(e^x - 1)}{x} = e^{2x} \cdot \frac{e^x - 1}{x} \to 1 \cdot 1 = 1$.

Wait: check again. $e^{3x} - e^{2x} = e^{2x}(e^x - 1)$.
$$\lim_{x \to 0} \frac{e^{2x}(e^x - 1)}{x} = e^0 \cdot 1 = 1$$

**7.** $\displaystyle\lim_{x \to 0} \frac{\ln(1 + \sin x)}{x}$

Since $\sin x \approx x$ near $0$ and $\ln(1+u)/u \to 1$ as $u \to 0$:
$$\frac{\ln(1+\sin x)}{x} = \frac{\ln(1+\sin x)}{\sin x} \cdot \frac{\sin x}{x} \to 1 \cdot 1 = 1$$

**8.** $\displaystyle\lim_{x \to 0} \frac{1 - \cos x}{x^2}$

Multiply by conjugate:
$$\frac{1-\cos x}{x^2} = \frac{(1-\cos x)(1+\cos x)}{x^2(1+\cos x)} = \frac{\sin^2 x}{x^2(1+\cos x)} = \left(\frac{\sin x}{x}\right)^2 \cdot \frac{1}{1+\cos x} \to 1 \cdot \frac{1}{2} = \frac{1}{2}$$

**9.** $\displaystyle\lim_{x \to 0^+} \frac{\sin x}{\sqrt{x}}$

$$\frac{\sin x}{\sqrt{x}} = \sqrt{x} \cdot \frac{\sin x}{x} \to 0 \cdot 1 = 0$$

**10.** $\displaystyle\lim_{x \to \infty} \frac{\ln x}{x}$

L'Hôpital ($\infty/\infty$): $\frac{1/x}{1} = \frac{1}{x} \to 0$.

**11.** $\displaystyle\lim_{x \to 1} \frac{x^n - 1}{x - 1}$

Factor: $x^n - 1 = (x-1)(x^{n-1} + x^{n-2} + \cdots + 1)$. Cancel $(x-1)$: limit is $n$.

Alternatively: this is the derivative of $x^n$ at $x = 1$, which is $n \cdot 1^{n-1} = n$.

**12.** $\displaystyle\lim_{x \to 0} \frac{\arcsin x}{x}$

Let $y = \arcsin x$, so $x = \sin y$ and $y \to 0$ as $x \to 0$:
$$\frac{\arcsin x}{x} = \frac{y}{\sin y} = \left(\frac{\sin y}{y}\right)^{-1} \to 1^{-1} = 1$$

### B.2 Continuity Verification Walkthrough

**Problem:** Determine where $f(x) = \frac{x^2 - 1}{\lvert x - 1 \rvert}$ is continuous and classify any discontinuities.

**Solution:**

For $x \neq 1$:
$$f(x) = \frac{(x-1)(x+1)}{\lvert x-1 \rvert}$$

Case $x > 1$: $\lvert x-1 \rvert = x-1$, so $f(x) = x + 1$. Continuous for $x > 1$.

Case $x < 1$: $\lvert x-1 \rvert = -(x-1) = 1-x$, so $f(x) = -(x+1)$. Continuous for $x < 1$.

At $x = 1$: $f(1)$ is undefined (division by zero). Check one-sided limits:
$$\lim_{x \to 1^+} f(x) = 1 + 1 = 2, \quad \lim_{x \to 1^-} f(x) = -(1+1) = -2$$

One-sided limits differ: $2 \neq -2$. This is a **jump discontinuity** at $x = 1$ with jump of $4$.

The discontinuity is not removable (both limits are finite but unequal).

### B.3 ε-δ Proof for a Nonlinear Limit

**Claim:** $\lim_{x \to 2} x^2 = 4$.

**Proof:** Given $\varepsilon > 0$. We need $\lvert x^2 - 4 \rvert < \varepsilon$ when $\lvert x - 2 \rvert < \delta$.

Factor: $\lvert x^2 - 4 \rvert = \lvert x-2 \rvert \cdot \lvert x+2 \rvert$.

**Control $\lvert x+2 \rvert$:** Assume $\lvert x - 2 \rvert < 1$, so $1 < x < 3$, giving $3 < x + 2 < 5$, so $\lvert x+2 \rvert < 5$.

Then: $\lvert x^2 - 4 \rvert = \lvert x-2 \rvert \cdot \lvert x+2 \rvert < 5\lvert x-2 \rvert$.

Choose $\delta = \min\!\left(1, \frac{\varepsilon}{5}\right)$. Then:
$$\lvert x-2 \rvert < \delta \implies \lvert x^2 - 4 \rvert < 5 \cdot \frac{\varepsilon}{5} = \varepsilon \quad \square$$

**Pattern:** For polynomial limits, always: (1) factor out $\lvert x-a \rvert$, (2) bound the remaining factor using $\lvert x-a \rvert < 1$, (3) choose $\delta = \min(1, \varepsilon/\text{bound})$.

---

## Appendix C: Connections to Advanced Mathematics

### C.1 Topological Perspective

In topology, continuity is defined without reference to metrics. A function $f: X \to Y$ between topological spaces is continuous if for every open set $V \subseteq Y$, the preimage $f^{-1}(V) = \{x \in X : f(x) \in V\}$ is open in $X$.

**For $\mathbb{R}$:** Open sets are unions of open intervals. The topological definition is equivalent to the ε-δ definition because ε-balls are the open sets of $\mathbb{R}$.

**Consequence:** The image of a compact set under a continuous function is compact (the continuous image of a compact set is compact). For $[a,b]$, this gives: $f([a,b])$ is closed and bounded, i.e., $f$ attains its maximum and minimum — the Extreme Value Theorem.

### C.2 Uniform Continuity and Lipschitz Maps in ML

A function $f: \mathbb{R}^n \to \mathbb{R}^m$ is **$L$-Lipschitz** if:
$$\lVert f(\mathbf{x}) - f(\mathbf{y}) \rVert \leq L \lVert \mathbf{x} - \mathbf{y} \rVert \quad \forall \mathbf{x}, \mathbf{y}$$

Lipschitz continuity implies uniform continuity (take $\delta = \varepsilon/L$).

**In ML:**
- **Spectral normalization** (Miyato et al., 2018): constrains each weight matrix to have spectral norm $\leq 1$, making each layer 1-Lipschitz, making the whole network $O(1)$-Lipschitz
- **Wasserstein GAN** (Arjovsky et al., 2017): the discriminator must be 1-Lipschitz (enforced via gradient penalty or spectral norm), which makes the Wasserstein distance well-defined
- **Gradient clipping**: bounding $\lVert \nabla \mathcal{L} \rVert \leq C$ ensures the loss is $C$-Lipschitz in parameters — controls training instability

### C.3 Limits in Metric Spaces

The ε-δ definition generalizes directly to metric spaces: replace $\lvert x - a \rvert$ with $d(x, a)$ for any metric $d$. This covers:
- **$\ell^p$ spaces:** $\lVert \mathbf{x} - \mathbf{a} \rVert_p < \delta$ for vector limits
- **Function spaces:** convergence of sequences of functions (pointwise, uniform, $L^2$)
- **Matrix sequences:** convergence in Frobenius norm $\lVert A_n - A \rVert_F < \varepsilon$

**Convergence of matrix series:** The matrix exponential $e^A = \sum_{k=0}^\infty A^k / k!$ is defined as a limit of partial sums in the matrix operator norm — a direct generalization of the scalar exponential limit.

### C.4 Dirichlet and Thomae Functions: Pathological Examples

**Dirichlet function:**
$$D(x) = \begin{cases} 1 & x \in \mathbb{Q} \\ 0 & x \notin \mathbb{Q} \end{cases}$$

This function has no limit at any point (rationals and irrationals are dense in $\mathbb{R}$, so any neighborhood of $a$ contains both). It is continuous nowhere.

**Thomae's function (popcorn function):**
$$T(x) = \begin{cases} 1/q & x = p/q \text{ in lowest terms}, x \in \mathbb{Q} \\ 0 & x \notin \mathbb{Q} \end{cases}$$

This function satisfies $\lim_{x \to a} T(x) = 0$ for every $a$ (irrationals accumulate near every point, and the rational values $1/q$ become small for large $q$). So $T$ is continuous at every irrational and discontinuous at every rational — a function continuous exactly on the irrationals.

These pathological examples show that the ε-δ framework is necessary: intuition from smooth curves does not predict the behavior of all functions.

---

## Appendix D: Numerical Methods Grounded in Limits

### D.1 Bisection Method

The bisection method exploits the IVT to find roots of $f(x) = 0$.

**Algorithm:**
1. Start with $[a_0, b_0]$ where $f(a_0) \cdot f(b_0) < 0$
2. At step $n$: let $m_n = (a_n + b_n)/2$
   - If $f(a_n) \cdot f(m_n) < 0$: set $[a_{n+1}, b_{n+1}] = [a_n, m_n]$
   - Else: set $[a_{n+1}, b_{n+1}] = [m_n, b_n]$
3. By IVT, $f$ has a root in every $[a_n, b_n]$
4. $\lim_{n \to \infty} a_n = \lim_{n \to \infty} b_n = c$ where $f(c) = 0$

**Convergence:** After $n$ steps, the interval has length $(b_0 - a_0)/2^n$. To achieve accuracy $\varepsilon$: need $n \geq \log_2((b_0-a_0)/\varepsilon)$ steps. This is $O(\log 1/\varepsilon)$ — linear in the number of bits of precision.

**Connection to limits:** Bisection computes $\lim_{n \to \infty} m_n$ numerically. The sequence $(m_n)$ is Cauchy (differences go to zero), so it converges in $\mathbb{R}$ by completeness. The limit is the root — IVT guarantees existence, completeness guarantees the numerical process converges.

### D.2 Newton's Method as a Limit Process

Newton's method computes:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

This converges (quadratically near a simple root) to $c$ where $f(c) = 0$.

The update formula comes from the linear approximation: $f(x) \approx f(x_n) + f'(x_n)(x - x_n)$, set to zero: $x = x_n - f(x_n)/f'(x_n)$. This linear approximation is itself a limit statement — the tangent line is $\lim_{h \to 0}$ of secant lines.

**For AI:** Quasi-Newton methods (BFGS, L-BFGS) approximate $f'(x_n)^{-1}$ (the Hessian inverse) via finite differences of gradients. At the heart of each update is a finite difference approximation of the second derivative — a discrete limit.

### D.3 Gradient Checking Implementation

To verify that an implemented gradient $\hat{g}_i$ matches the true gradient $\partial f / \partial \theta_i$, use the centered finite difference:

$$g_i^{\text{FD}} = \frac{f(\theta + h \mathbf{e}_i) - f(\theta - h \mathbf{e}_i)}{2h}$$

The relative error check:
$$\frac{\lVert \hat{g} - g^{\text{FD}} \rVert}{\lVert \hat{g} \rVert + \lVert g^{\text{FD}} \rVert} < 10^{-5}$$

is a heuristic validation that the implementation of the analytic gradient is correct.

**Why centered?** The centered difference approximates $f'$ with error $O(h^2)$ (from Taylor: $f(x+h) - f(x-h) = 2hf'(x) + (h^3/3)f'''(x) + \ldots$), while the one-sided approximation has error $O(h)$. The limit is the same but the rate of approach is faster for centered differences.

---

## Appendix E: Limits in Probability and Statistics

### E.1 Law of Large Numbers

The weak law of large numbers states: if $X_1, X_2, \ldots$ are i.i.d. with mean $\mu$, then for every $\varepsilon > 0$:
$$\lim_{n \to \infty} P\!\left(\left\lvert \frac{1}{n}\sum_{i=1}^n X_i - \mu \right\rvert > \varepsilon\right) = 0$$

This is a limit statement about a sequence of probabilities — a "convergence in probability." The expected loss over the training distribution is $\mu = \mathbb{E}[\mathcal{L}]$; the empirical loss $\hat{\mathcal{L}}_n = \frac{1}{n}\sum_i \mathcal{L}(\mathbf{x}_i)$ converges to it in the limit of infinite data.

### E.2 Central Limit Theorem as a Limit

The CLT states: for i.i.d. $X_i$ with mean $\mu$ and variance $\sigma^2$:
$$\sqrt{n}\left(\frac{\bar{X}_n - \mu}{\sigma}\right) \xrightarrow{d} \mathcal{N}(0, 1)$$

This "convergence in distribution" is a limit of CDFs: for every continuity point $t$ of the standard normal CDF $\Phi$:
$$\lim_{n \to \infty} P\!\left(\sqrt{n}\left(\frac{\bar{X}_n - \mu}{\sigma}\right) \leq t\right) = \Phi(t)$$

**For AI:** Mini-batch gradient estimates converge to the full-batch gradient in distribution as batch size grows — the CLT justifies treating mini-batch gradients as Gaussian perturbations of the true gradient, underpinning stochastic optimization theory (Mandt, Hoffman, Blei, 2017).

### E.3 KL Divergence and Limit Continuity

The KL divergence $D_{\text{KL}}(P \| Q) = \sum_i p_i \log(p_i / q_i)$ requires the convention $0 \log 0 = 0$ (using $\lim_{p \to 0^+} p \log p = 0$) and is undefined when $q_i = 0$ but $p_i > 0$.

**Continuity:** With the convention $0 \log 0 = 0$, $D_{\text{KL}}$ is lower-semicontinuous: for any sequence $P_n \to P$, $\liminf_{n} D_{\text{KL}}(P_n \| Q) \geq D_{\text{KL}}(P \| Q)$. This is a limit property that ensures KL minimization is well-posed.

**For AI:** RLHF training (Ouyang et al., 2022) includes a KL penalty $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ to prevent the model from drifting too far from the reference policy. The continuity of KL guarantees that small policy changes produce small KL penalties — essential for stable RLHF training.

---

## Appendix F: Glossary of Limit-Related Terms

| Term | Definition |
|------|-----------|
| **Limit** | $\lim_{x\to a}f(x) = L$: $f(x)$ can be made arbitrarily close to $L$ by taking $x$ sufficiently close to (but not equal to) $a$ |
| **ε-δ definition** | Formal definition: $\forall\varepsilon>0,\exists\delta>0: 0<\lvert x-a\rvert<\delta\implies\lvert f(x)-L\rvert<\varepsilon$ |
| **One-sided limit** | $\lim_{x\to a^+}$ (right) or $\lim_{x\to a^-}$ (left) — approaches $a$ from one side only |
| **Limit at infinity** | $\lim_{x\to\infty}f(x)=L$: $f(x)\to L$ as $x$ grows without bound |
| **Infinite limit** | $\lim_{x\to a}f(x)=\pm\infty$: $f(x)$ grows without bound as $x\to a$; not a real limit |
| **Continuity** | $f$ is continuous at $a$ if $f(a)$ exists, $\lim_{x\to a}f(x)$ exists, and they are equal |
| **Removable discontinuity** | $\lim_{x\to a}f(x)$ exists but $f(a)$ is undefined or $\neq$ limit |
| **Jump discontinuity** | One-sided limits exist but differ: $\lim_{x\to a^-}f(x)\neq\lim_{x\to a^+}f(x)$ |
| **Essential discontinuity** | At least one one-sided limit is $\pm\infty$ or doesn't exist |
| **Indeterminate form** | Forms like $0/0$, $\infty/\infty$, $0\cdot\infty$, $1^\infty$ that require further analysis |
| **L'Hôpital's Rule** | For $0/0$ or $\pm\infty/\pm\infty$: $\lim f/g = \lim f'/g'$ |
| **Squeeze Theorem** | $h\leq f\leq g$ and $h,g\to L$ implies $f\to L$ |
| **IVT** | Continuous $f$ on $[a,b]$ attains every value between $f(a)$ and $f(b)$ |
| **EVT** | Continuous $f$ on $[a,b]$ attains its maximum and minimum |
| **Uniform continuity** | $\delta$ works uniformly for all points (not point-dependent) |
| **Lipschitz continuity** | $\lvert f(x)-f(y)\rvert\leq L\lvert x-y\rvert$ — quantitative uniform continuity |
| **Machine epsilon** | Smallest $\varepsilon$ with $\text{fl}(1+\varepsilon)>1$ in floating point ($\approx 10^{-16}$ for float64) |
| **Catastrophic cancellation** | Loss of significant digits when subtracting nearly equal floating-point numbers |
| **limsup / liminf** | Largest/smallest accumulation point of $f(x)$ as $x\to a$ |
| **Big-O** | $f=O(g)$: $\lvert f\rvert\leq C\lvert g\rvert$ near $a$ for some $C>0$ |
| **Little-o** | $f=o(g)$: $f/g\to 0$ as $x\to a$; $f$ is negligible compared to $g$ |

---

## Appendix G: Further Reading and References

### Textbooks

1. **Stewart, J.** (2015). *Calculus: Early Transcendentals* (8th ed.). Cengage. — Standard undergraduate reference; clear motivation and worked examples.

2. **Spivak, M.** (2006). *Calculus* (4th ed.). Publish or Perish. — Rigorous treatment; complete proofs of all major theorems including IVT and EVT.

3. **Rudin, W.** (1976). *Principles of Mathematical Analysis* (3rd ed.). McGraw-Hill. — The graduate-level standard; ε-δ proofs throughout; metric space generalization.

4. **Apostol, T. M.** (1974). *Mathematical Analysis* (2nd ed.). Addison-Wesley. — Thorough treatment of limits, continuity, and the Riemann integral.

### Machine Learning Connections

5. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. Ch. 4 (Numerical computation), Ch. 6 (Activation functions). — Numerical stability, sigmoid/ReLU analysis.

6. **Hochreiter, S., & Schmidhuber, J.** (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780. — Identifies vanishing gradient via limit behavior of sigmoid.

7. **Robbins, H., & Monro, S.** (1951). A stochastic approximation method. *Annals of Mathematical Statistics*, 22(3), 400–407. — Original convergence conditions for SGD.

8. **Vaswani, A., et al.** (2017). Attention is All You Need. *NeurIPS*. — Softmax temperature in attention.

9. **Miyato, T., et al.** (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR*. — Lipschitz continuity in deep learning.

10. **Mandt, S., Hoffman, M. D., & Blei, D. M.** (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. *JMLR*. — CLT applied to SGD noise.

### Online Resources

11. **3Blue1Brown — "Essence of Calculus"** (YouTube). Visual intuition for limits and derivatives.

12. **Paul's Online Math Notes** (tutorial.math.lamar.edu). Comprehensive worked examples at undergraduate level.

13. **MIT OpenCourseWare 18.01** (Single Variable Calculus). Full lecture notes and problem sets.

---

## Appendix H: Detailed ML Worked Examples

### H.1 Softmax Numerical Stability: Full Derivation

The **naive softmax** computation:
```python
def softmax_naive(z):
    return np.exp(z) / np.sum(np.exp(z))
```
fails for large $\lvert z_i \rvert$ due to overflow ($e^{1000} = \infty$ in float64) or underflow ($e^{-1000} = 0$).

**The log-sum-exp trick** exploits the limit-preserving shift:
$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}} = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \quad m = \max_j z_j$$

This is valid because $e^{z_i - m} / \sum_j e^{z_j - m}$ equals $e^{z_i}/\sum_j e^{z_j}$ (both numerator and denominator are multiplied by $e^{-m}$). After shifting, all exponents are $\leq 0$, so no overflow. The maximum term contributes $e^0 = 1$, preventing underflow of the sum.

**Derivation of stability bound:** Let $a_i = z_i - m \leq 0$. Then $e^{a_i} \in (0, 1]$ for all $i$. The sum $\sum_j e^{a_j} \in [1, n]$ (at least $e^{a_{\text{argmax}}} = 1$, at most $n$ terms each $\leq 1$). No overflow or underflow.

**Log-softmax** (needed for cross-entropy):
$$\log\text{softmax}(\mathbf{z})_i = z_i - m - \log\sum_j e^{z_j - m}$$

This is the numerically stable form used in `torch.nn.CrossEntropyLoss`.

**Limit interpretation:** The log-sum-exp function $\text{LSE}(\mathbf{z}) = \log\sum_j e^{z_j}$ is a smooth approximation to the max:
$$\lim_{\beta \to \infty} \frac{1}{\beta} \log\sum_j e^{\beta z_j} = \max_j z_j$$

This is another limit — the "softmax" smoothly approaches the argmax as temperature $T = 1/\beta \to 0$.

### H.2 Vanishing Gradient: Quantitative Analysis via Limits

Consider an $L$-layer network with sigmoid activations. The gradient of loss $\mathcal{L}$ with respect to weights in layer $l$ is:
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \prod_{k=l+1}^{L} \frac{\partial \mathbf{a}^{[k]}}{\partial \mathbf{a}^{[k-1]}}$$

Each factor $\frac{\partial \mathbf{a}^{[k]}}{\partial \mathbf{a}^{[k-1]}} = \text{diag}(\sigma'(\mathbf{z}^{[k]})) \cdot W^{[k]}$ involves $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 1/4$.

If we model each factor as a scalar $\leq 1/4$:
$$\left\lVert \frac{\partial \mathcal{L}}{\partial W^{[l]}} \right\rVert \leq C \cdot \left(\frac{1}{4}\right)^{L-l}$$

As $L - l \to \infty$ (very deep network, gradient flows from the last layer to layer $l$):
$$\lim_{L \to \infty} \left(\frac{1}{4}\right)^{L-l} = 0$$

The gradient vanishes exponentially in the depth. For $L - l = 10$: $(1/4)^{10} \approx 10^{-6}$. For $L - l = 20$: $(1/4)^{20} \approx 10^{-12}$.

**Contrast with ReLU:** $\text{ReLU}'(x) = \mathbf{1}[x > 0] \in \{0, 1\}$. For positive pre-activations, each factor is $1$ (no attenuation). The limit is $1^{L-l} = 1$ — gradient passes unchanged. This is the key advantage of ReLU for deep networks (He et al., 2015).

**ResNet correction:** Residual connections add a skip path: $\mathbf{a}^{[k]} = \mathcal{F}(\mathbf{a}^{[k-1]}) + \mathbf{a}^{[k-1]}$. The gradient becomes:
$$\frac{\partial \mathbf{a}^{[k]}}{\partial \mathbf{a}^{[k-1]}} = \frac{\partial \mathcal{F}}{\partial \mathbf{a}^{[k-1]}} + I$$

The identity term $I$ prevents the product from decaying to zero — the residual "highway" carries gradients regardless of the activation. Mathematically: even if $\lVert \partial\mathcal{F}/\partial\mathbf{a}\rVert \to 0$ (saturated activations), the full Jacobian stays bounded away from zero.

### H.3 Learning Rate Schedules: Limit Analysis

**Cosine annealing:**
$$\alpha_t = \alpha_\text{min} + \frac{1}{2}(\alpha_\text{max} - \alpha_\text{min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

As $t \to T$: $\cos(\pi t/T) \to \cos(\pi) = -1$, so $\alpha_t \to \alpha_\text{min}$.

The schedule is continuous (cosine is continuous) and smooth, avoiding the discontinuous jumps of step-decay schedules. The limit $\alpha_T = \alpha_\text{min} > 0$ means it does not satisfy the Robbins-Monro conditions — in practice this is acceptable because modern large-batch training typically runs for a fixed number of steps, not until convergence.

**Warmup + linear decay:**
$$\alpha_t = \begin{cases} \alpha_\text{max} \cdot t/T_\text{warm} & t \leq T_\text{warm} \\ \alpha_\text{max} \cdot (T - t)/(T - T_\text{warm}) & t > T_\text{warm} \end{cases}$$

At $t = T_\text{warm}$, both formulas give $\alpha_\text{max}$: continuity is maintained. The limit $\lim_{t \to T} \alpha_t = 0$ satisfies $\sum \alpha_t = \infty$ (roughly $T^2/(2(T-T_\text{warm}))$) but $\sum \alpha_t^2$ may be finite or not depending on the schedule length.

**GPT-3 / LLaMA learning rate schedule:** Uses cosine decay with warmup, which empirically outperforms the theoretically optimal $1/t$ schedule for large-batch transformer training. The theory is underdeveloped; the practice is empirically justified.

### H.4 Adam Optimizer: Limit Behavior

The Adam update (Kingma & Ba, 2014):
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \quad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \alpha \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)$$

**Limit as $t \to \infty$:** The bias correction $1/(1-\beta_1^t) \to 1$ and $1/(1-\beta_2^t) \to 1$ (since $\beta_1^t \to 0$). So for large $t$, Adam behaves like the bias-corrected version without correction.

**Limit for sparse gradients:** If $g_t = 0$ for many steps, then $v_t \to 0$ (exponential decay). The effective learning rate $\alpha/(\sqrt{v_t} + \varepsilon) \to \alpha/\varepsilon$ — Adam gives a large step for a parameter that hasn't received gradients recently. This is the "adaptive" feature: rarely-updated parameters get large steps when they finally do receive gradients. This is controlled by the limit behavior of the second moment estimator.

---

## Appendix I: Self-Assessment Questions

### Conceptual Questions

1. Explain in your own words why the ε-δ definition requires $0 < \lvert x - a \rvert$ (strict inequality, excluding $x = a$) but the continuity definition does not exclude $x = a$.

2. Give an example of a function $f$ such that:
   - $f$ is defined at $a$
   - $\lim_{x \to a} f(x)$ exists
   - But $f$ is not continuous at $a$
   Explain which of the three continuity conditions fails.

3. Why does the Squeeze Theorem require the inequality $h(x) \leq f(x) \leq g(x)$ to hold in a neighborhood of $a$, not just at $a$?

4. L'Hôpital's Rule is often stated as: "just differentiate numerator and denominator separately." What is wrong with this description, and what are the actual conditions for the rule to apply?

5. Why does the IVT require the function to be continuous on a closed interval $[a,b]$ and not just on the open interval $(a,b)$? Give a counterexample showing what can go wrong on an open interval.

6. Explain why $\lim_{x \to \infty} f(x) = L$ (finite) and $\lim_{x \to \infty} f(x) = \infty$ represent fundamentally different situations, even though both involve $x \to \infty$.

7. In what sense is the derivative $f'(a)$ "just a limit"? What property of the limit (one-sided vs. two-sided) determines whether $f$ is differentiable at $a$?

### Computational Practice

8. Compute $\lim_{x \to 0} \frac{e^{2x} - 2e^x + 1}{x^2}$ using three different methods: (a) L'Hôpital twice, (b) Taylor expansion, (c) substitution.

9. Find all discontinuities of $f(x) = \frac{\sin(\pi x)}{x^2 - 1}$ and classify each.

10. Prove from the ε-δ definition that $\lim_{x \to 3} \sqrt{x} = \sqrt{3}$. (Hint: rationalize $\lvert \sqrt{x} - \sqrt{3} \rvert$.)

11. Use the Squeeze Theorem to show $\lim_{x \to 0^+} x \lfloor 1/x \rfloor = 1$. (Hint: $1/x - 1 < \lfloor 1/x \rfloor \leq 1/x$.)

12. Determine whether $f(x) = \sum_{n=0}^\infty x^n / n!$ (the Taylor series for $e^x$) is continuous on all of $\mathbb{R}$. (Hint: uniform convergence on compact sets.)

### ML Application Questions

13. Write Python code to compute $\lim_{T \to 0^+} \text{softmax}_T([3, 1, 2])$ numerically for $T = 1, 0.1, 0.01, 0.001, 0.0001$. What do you observe? At what temperature does floating-point underflow cause problems with naive implementation?

14. The learning rate $\alpha_t = C/t^\gamma$ satisfies the Robbins-Monro conditions for which values of $\gamma$? Prove your answer by computing $\sum \alpha_t$ and $\sum \alpha_t^2$.

15. Implement the stable sigmoid $\sigma_{\text{stable}}(x)$ that avoids overflow for both large positive and large negative $x$. Verify that it matches `scipy.special.expit(x)` to full precision for $x \in \{-1000, -100, -1, 0, 1, 100, 1000\}$.

---

## Appendix J: Connection Map — Limits Throughout the Curriculum

This section appears at the foundation of calculus, but limits permeate the entire curriculum.

```
WHERE LIMITS APPEAR ACROSS THE CURRICULUM
════════════════════════════════════════════════════════════════════════

§04-01  LIMITS (here) ──────────────────────────────────────────────►
  │                                                         All of Analysis
  │
  ├──► §04-02 Derivatives     f'(a) = lim_{h→0} [f(a+h)-f(a)]/h
  │
  ├──► §04-03 Integration     ∫ₐᵇ f = lim_{n→∞} Σ f(xᵢ*)Δxᵢ
  │
  ├──► §04-04 Series          Σaₙ = lim_{N→∞} Sₙ  (partial sums)
  │
  ├──► §05 Multivariate       ∂f/∂xᵢ = lim_{h→0} [f(x+heᵢ)-f(x)]/h
  │
  ├──► §06 Probability        P(A) = lim_{n→∞} #{outcomes in A}/n (freq.)
  │
  ├──► §08 Optimization       Convergence: lim_{t→∞} ||∇L(θₜ)|| = 0
  │
  ├──► §10 Numerical Methods  Finite differences, iterative convergence
  │
  ├──► §12 Functional Analysis Operator limits, Banach/Hilbert spaces
  │
  └──► §24 Measure Theory     Integration as limit of simple functions

════════════════════════════════════════════════════════════════════════
```

Every subsequent section in this curriculum builds on the limit concept introduced here. The ε-δ definition is the seed; the rest of mathematics is the tree.

---

## Appendix K: Proofs of Fundamental Limits

### K.1 Proof that lim(n→∞)(1 + 1/n)^n = e

We show the sequence $a_n = (1 + 1/n)^n$ is increasing and bounded, hence convergent, and define $e$ as its limit.

**Step 1: $a_n$ is increasing.**

By AM-GM: for positive numbers $x_1, \ldots, x_{n+1}$:
$$\frac{x_1 + \cdots + x_{n+1}}{n+1} \geq (x_1 \cdots x_{n+1})^{1/(n+1)}$$

Apply with $n$ copies of $(1 + 1/n)$ and one copy of $1$:
$$\frac{n(1+1/n) + 1}{n+1} \geq \left[(1+1/n)^n \cdot 1\right]^{1/(n+1)}$$
$$\frac{n+2}{n+1} = 1 + \frac{1}{n+1} \geq a_n^{1/(n+1)}$$

Raising both sides to the $(n+1)$-th power: $a_{n+1} \geq a_n$. $\square$

**Step 2: $a_n$ is bounded above by $3$.**

By the binomial theorem:
$$a_n = \left(1+\frac{1}{n}\right)^n = \sum_{k=0}^n \binom{n}{k}\frac{1}{n^k} = \sum_{k=0}^n \frac{n(n-1)\cdots(n-k+1)}{k! \cdot n^k}$$

Each term $\frac{n(n-1)\cdots(n-k+1)}{k! \cdot n^k} = \frac{1}{k!}\cdot\frac{n}{n}\cdot\frac{n-1}{n}\cdots < \frac{1}{k!}$.

So $a_n < \sum_{k=0}^n \frac{1}{k!} \leq \sum_{k=0}^\infty \frac{1}{k!} = e \leq \sum_{k=0}^\infty \frac{1}{2^k} = 2 \cdot \frac{1}{1-1/2}$... 

More carefully: $\sum_{k=0}^\infty 1/k! \leq 1 + 1 + 1/2 + 1/4 + \ldots = 1 + 2 = 3$.

By the monotone convergence theorem, the increasing bounded sequence $a_n$ converges. We define $e = \lim_{n\to\infty}(1+1/n)^n \approx 2.718\ldots$ $\square$

### K.2 Proof that lim(x→0) sin(x)/x = 1

We give the geometric proof in detail.

**Setup:** Consider the unit circle centered at the origin. Let $O = (0,0)$, $A = (1,0)$, $P = (\cos x, \sin x)$ for $0 < x < \pi/2$, and $T = (1, \tan x)$ (the tangent line at $A$ meets the line $OP$ at $T$).

**Area inequalities:**
$$\text{Area}(\triangle OAP) \leq \text{Area(sector }OAP) \leq \text{Area}(\triangle OAT)$$

Computing each:
- $\text{Area}(\triangle OAP) = \frac{1}{2} \cdot 1 \cdot \sin x = \frac{\sin x}{2}$ (base $OA = 1$, height $= \sin x$)
- $\text{Area(sector)} = \frac{x}{2\pi} \cdot \pi(1^2) = \frac{x}{2}$ (fraction $x/2\pi$ of unit circle area $\pi$)
- $\text{Area}(\triangle OAT) = \frac{1}{2} \cdot 1 \cdot \tan x = \frac{\tan x}{2}$

So: $\frac{\sin x}{2} \leq \frac{x}{2} \leq \frac{\tan x}{2}$

Multiply by $\frac{2}{\sin x} > 0$:
$$1 \leq \frac{x}{\sin x} \leq \frac{1}{\cos x}$$

Take reciprocals (reverse inequalities):
$$\cos x \leq \frac{\sin x}{x} \leq 1$$

Since $\lim_{x\to 0^+} \cos x = 1$ and $\lim_{x\to 0^+} 1 = 1$, by Squeeze: $\lim_{x\to 0^+} \frac{\sin x}{x} = 1$.

For $x \to 0^-$: use $\sin(-x)/(-x) = \sin(x)/x \to 1$ (same by symmetry). $\square$

### K.3 Proof of L'Hôpital's Rule (0/0 case)

**Theorem.** Let $f, g$ be differentiable on $(a - r, a + r) \setminus \{a\}$ for some $r > 0$. Suppose $\lim_{x\to a} f(x) = \lim_{x\to a} g(x) = 0$, $g'(x) \neq 0$ near $a$, and $\lim_{x\to a} f'(x)/g'(x) = L$. Then $\lim_{x\to a} f(x)/g(x) = L$.

**Proof (using Cauchy's MVT):** Extend $f$ and $g$ by $f(a) = g(a) = 0$ (so they are continuous at $a$).

**Cauchy's Mean Value Theorem:** For any $x \neq a$ (say $x > a$), there exists $c_x \in (a, x)$ such that:
$$\frac{f(x) - f(a)}{g(x) - g(a)} = \frac{f'(c_x)}{g'(c_x)}$$

Since $f(a) = g(a) = 0$: $\frac{f(x)}{g(x)} = \frac{f'(c_x)}{g'(c_x)}$.

As $x \to a^+$: $c_x \in (a, x) \to a$, so $c_x \to a^+$.

Therefore: $\lim_{x\to a^+} \frac{f(x)}{g(x)} = \lim_{x\to a^+} \frac{f'(c_x)}{g'(c_x)} = \lim_{c\to a^+} \frac{f'(c)}{g'(c)} = L$.

Similarly from the left. $\square$

---

## Appendix L: Extended ML Applications — Advanced Topics

### L.1 Limits in Attention Mechanisms

The scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The scaling by $\sqrt{d_k}$ prevents the inner products $QK^\top$ from growing large in magnitude. Without scaling, as $d_k \to \infty$, the logits $QK^\top$ grow like $O(\sqrt{d_k})$ (random initialization variance), pushing softmax into the saturation regime:

$$\lim_{d_k \to \infty} \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \to \text{(depends on structure)}$$

With scaling: $QK^\top / \sqrt{d_k}$ remains $O(1)$ as $d_k \to \infty$ (assuming unit-variance queries and keys), keeping softmax in its linear regime where gradients are large.

**Formal statement:** If $q, k \in \mathbb{R}^{d_k}$ with independent $\mathcal{N}(0,1)$ components:
$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = d_k$$
So $q \cdot k / \sqrt{d_k}$ has variance $1$ regardless of $d_k$ — this is the limit $\lim_{d_k\to\infty} \text{Var}(q\cdot k/\sqrt{d_k}) = 1$.

### L.2 Token Probability Limits in Language Modeling

An autoregressive language model assigns probability:
$$P(w_t \mid w_1, \ldots, w_{t-1}) = \text{softmax}(W \mathbf{h}_{t-1})_{w_t}$$

**Limit as context grows:** What happens to $P(w_t \mid w_1, \ldots, w_{t-1})$ as $t \to \infty$? Under standard ergodicity assumptions on the language distribution, the model's uncertainty should approach the entropy of the distribution:
$$\lim_{t \to \infty} H(w_t \mid w_1, \ldots, w_{t-1}) = H_\infty$$

where $H_\infty$ is the entropy rate of the language. This limit (which exists for stationary ergodic processes) is the asymptotic uncertainty per token — the information-theoretic lower bound on perplexity.

### L.3 Neural Tangent Kernel and Infinite Width Limits

As the width $n \to \infty$ of a neural network, the network function converges (in distribution) to a Gaussian process with a specific kernel called the **Neural Tangent Kernel** (Jacot et al., 2018):

$$\lim_{n \to \infty} f_\theta(\mathbf{x}) \sim \mathcal{GP}(0, K_{\text{NTK}})$$

Moreover, in this limit, training with gradient descent is equivalent to kernel regression with $K_{\text{NTK}}$:

$$\lim_{n \to \infty} \theta_t = \theta_0 - \eta \int_0^t \nabla_\theta f_{\theta_s} \nabla_\theta \mathcal{L}_s \, ds$$

This **infinite-width limit** is an active research area connecting neural network training to the well-understood theory of kernel methods and Gaussian processes — all via limit theory applied to neural networks as a function of width.

### L.4 Grokking as a Limit Phenomenon

**Grokking** (Power et al., 2022): the phenomenon where a neural network first overfits (near-zero training loss, high test loss), then after continued training, suddenly generalizes (both losses drop). 

From a limit perspective: the training loss is minimized early (the limit of the optimization trajectory is a global minimum), but the test loss requires the model to find a different basin that generalizes. The sudden transition can be understood as crossing a threshold where the solution's "complexity" (measured by norms or effective rank) crosses a critical value.

The limit $\lim_{t\to\infty} \mathcal{L}_{\text{test}}(\theta_t)$ may be much smaller than $\lim_{t\to T_1} \mathcal{L}_{\text{test}}(\theta_t)$ for an intermediate time $T_1$ — the limit of the process depends on running it long enough. This is a reminder that limits of optimization trajectories are not always achieved quickly.

---

## Appendix M: Quick Reference Card

```
LIMITS AND CONTINUITY — QUICK REFERENCE
════════════════════════════════════════════════════════════════════════

DEFINITION            lim_{x→a} f(x) = L
                      ⟺ ∀ε>0, ∃δ>0: 0<|x-a|<δ ⟹ |f(x)-L|<ε

ONE-SIDED             lim_{x→a⁺} and lim_{x→a⁻}
                      Two-sided limit ⟺ both one-sided limits agree

LIMIT LAWS            lim(f±g) = lim f ± lim g
                      lim(fg) = (lim f)(lim g)
                      lim(f/g) = (lim f)/(lim g) if lim g ≠ 0

KEY LIMITS            lim_{x→0} sin(x)/x = 1
                      lim_{x→0} (eˣ-1)/x = 1
                      lim_{n→∞} (1+1/n)ⁿ = e
                      lim_{x→0⁺} x·ln(x) = 0

TECHNIQUES            • Direct substitution (if continuous)
                      • Factoring / cancellation
                      • Rationalization (conjugate)
                      • L'Hôpital (0/0 or ∞/∞ only!)
                      • Squeeze Theorem

CONTINUITY            f continuous at a ⟺
                      (1) f(a) defined
                      (2) lim_{x→a} f(x) exists
                      (3) lim_{x→a} f(x) = f(a)

DISCONTINUITIES       Removable: limit exists, ≠ f(a) or f(a) undefined
                      Jump: one-sided limits exist but differ
                      Essential: at least one one-sided limit = ±∞

IVT                   f continuous on [a,b], k between f(a) and f(b)
                      ⟹ ∃c∈(a,b): f(c) = k

EVT                   f continuous on [a,b] ⟹ f attains max and min

SQUEEZE               h≤f≤g near a, lim h = lim g = L ⟹ lim f = L

STABILITY             (eˣ-1)/x near x=0: use numpy.expm1(x)/x
                      log(1+x)/x near x=0: use numpy.log1p(x)/x
                      log(softmax(z)): use log_softmax (LSE trick)

ML CONNECTIONS        softmax_T → argmax as T→0, uniform as T→∞
                      σ(x) → 0/1 as x→±∞ (vanishing gradient)
                      gradient = lim_{h→0}[f(x+h)-f(x)]/h
                      Robbins-Monro: Σαₜ=∞ AND Σαₜ²<∞

════════════════════════════════════════════════════════════════════════
```

---

## Appendix N: Problem Set — Additional Exercises

### N.1 Computation Problems (★)

**N1.** Compute the following limits without L'Hôpital's Rule:

(a) $\displaystyle\lim_{x \to 4} \frac{\sqrt{x} - 2}{x - 4}$

(b) $\displaystyle\lim_{h \to 0} \frac{(2+h)^3 - 8}{h}$

(c) $\displaystyle\lim_{x \to 0} \frac{\sin(3x)}{5x}$

(d) $\displaystyle\lim_{x \to \infty} \frac{4x^3 - 2x + 1}{7x^3 + x^2 - 3}$

*Solutions:* (a) $1/4$ (rationalize), (b) $12$ (derivative of $x^3$ at $x=2$), (c) $3/5$ (fundamental limit), (d) $4/7$ (leading coefficients).

**N2.** Classify the discontinuities of $f(x) = \frac{x^2 - 3x + 2}{x^2 - 1}$ on $\mathbb{R}$.

Factor: numerator $= (x-1)(x-2)$, denominator $= (x-1)(x+1)$.

At $x = 1$: $f(x) = \frac{x-2}{x+1}$ after cancellation; $\lim_{x\to 1} f(x) = -1/2$. Removable discontinuity (redefine $f(1) = -1/2$).

At $x = -1$: denominator $\to 0$, numerator $\to (-2)(-3) = 6 \neq 0$; $f(x) \to \pm\infty$. Vertical asymptote / essential discontinuity.

**N3.** For which values of $c$ is $g(x) = \begin{cases} cx^2 + 2x & x < 2 \\ x^3 - cx & x \geq 2 \end{cases}$ continuous at $x = 2$?

Require: $c(4) + 4 = 8 - 2c$, so $4c + 4 = 8 - 2c$, giving $6c = 4$, $c = 2/3$.

### N.2 Theory Problems (★★)

**N4.** Prove using the ε-δ definition that $\lim_{x \to 0} x \sin(1/x) = 0$.

*Proof:* For any $\varepsilon > 0$, choose $\delta = \varepsilon$. Then for $0 < \lvert x \rvert < \delta$:
$$\lvert x \sin(1/x) - 0 \rvert = \lvert x \rvert \cdot \lvert \sin(1/x) \rvert \leq \lvert x \rvert \cdot 1 < \delta = \varepsilon \quad \square$$

**N5.** Show that the converse of the Extreme Value Theorem is false: give an example of $f$ attaining its maximum and minimum on $(0,1)$ without $f$ being continuous.

*Example:* $f(x) = 0$ for $x \in (0,1)$ and $f(x)$ undefined elsewhere. Then $f$ attains max and min (both equal $0$) but is "vacuously" not defined as a function on $[0,1]$ for the purposes of continuity. A cleaner example: $f: [0,1] \to \mathbb{R}$, $f(0) = f(1) = 1$, $f(x) = 0$ for $x \in (0,1)$ — attains max ($1$) and min ($0$) but is discontinuous at $0$ and $1$.

**N6.** (Banach Fixed-Point Theorem, simplified.) Let $f: [0,1] \to [0,1]$ be continuous. Show $f$ has at least one fixed point.

Define $g(x) = f(x) - x$. Then $g(0) = f(0) - 0 = f(0) \geq 0$ and $g(1) = f(1) - 1 \leq 0$. By IVT applied to $g$ (continuous) on $[0,1]$: $\exists c \in [0,1]$ with $g(c) = 0$, i.e., $f(c) = c$. $\square$

### N.3 ML Application Problems (★★★)

**N7.** (Numerical stability.) Implement both naive and stable computations of $\log(1 + e^x)$ (the softplus function). For $x = 100, 50, 10, 0, -10, -50$, compare results and relative errors.

*Analysis:* For large $x > 0$: $\log(1+e^x) \approx x$. Naive $e^x$ overflows for $x > 709$ (float64). Stable version: $\log(1+e^x) = x + \log(1 + e^{-x})$ for $x > 0$ (the $e^{-x}$ term is small and doesn't overflow). For $x < 0$: use naive form since $e^x < 1$.

**N8.** (Learning rate theory.) Consider the schedule $\alpha_t = \alpha_0 / (1 + \beta t)$ for constants $\alpha_0, \beta > 0$.

(a) Show this satisfies the first Robbins-Monro condition: $\sum_{t=1}^\infty \alpha_t = \infty$.

(b) Show this satisfies the second condition: $\sum_{t=1}^\infty \alpha_t^2 < \infty$.

(c) Compare to $\alpha_t = \alpha_0 / \sqrt{t}$: does this satisfy both conditions?

*Solutions:* (a) $\alpha_t \sim \alpha_0/(\beta t)$; $\sum 1/t = \infty$ (harmonic series). (b) $\alpha_t^2 \sim \alpha_0^2/(\beta^2 t^2)$; $\sum 1/t^2 = \pi^2/6 < \infty$. (c) $1/\sqrt{t}$: $\sum 1/\sqrt{t} = \infty$ (first holds), $\sum 1/t = \infty$ (second fails).

---

## Appendix O: Typesetting Reference for LaTeX

When writing mathematical content on limits in LaTeX (following the notation guide):

```latex
% Limit notation
\lim_{x \to a} f(x) = L           % standard limit
\lim_{x \to a^+} f(x)             % right-hand limit  
\lim_{x \to a^-} f(x)             % left-hand limit
\lim_{x \to +\infty} f(x)         % limit at +infinity

% ε-δ definition
\forall \varepsilon > 0, \; \exists \delta > 0 : \;
  0 < \lvert x - a \rvert < \delta \implies \lvert f(x) - L \rvert < \varepsilon

% Fundamental limits
\lim_{x \to 0} \frac{\sin x}{x} = 1
\lim_{x \to 0} \frac{e^x - 1}{x} = 1
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e

% Continuity
f \text{ continuous at } a \iff
  \lim_{x \to a} f(x) = f(a)

% Big-O notation
f(x) = O(g(x)) \text{ as } x \to a
f(x) = o(g(x)) \text{ as } x \to a
```

**Common errors in notation:**
- Use `\lvert \cdot \rvert` not `|\cdot|` for absolute value in LaTeX
- Use `\varepsilon` not `\epsilon` (matches standard analysis texts)
- Use `\to` not `:` for limit variable (i.e., $x \to a$, not $x:a$)
- Use `\infty` not `inf` or `Inf`
- Subscripts of limit use `_` correctly: `\lim_{x \to 0}` not `\lim{x \to 0}`

---

## Appendix P: Historical Problems and Their Solutions

### P.1 Zeno's Paradoxes: The Ancient Limit Problem

Zeno of Elea (~450 BCE) posed paradoxes about motion that are precisely limit problems in disguise.

**Achilles and the Tortoise:** Achilles (speed $v$) chases a tortoise (speed $u < v$) with head start $d$. Zeno argued: first Achilles must reach the tortoise's initial position (time $d/v$); by then the tortoise has moved $d \cdot u/v$ further; then Achilles must cover that gap (time $du/v^2$); and so on. The paradox claims this infinite sequence of tasks cannot be completed.

**Resolution via limits:** The total time is:
$$T = \frac{d}{v} + \frac{du}{v^2} + \frac{du^2}{v^3} + \cdots = \frac{d}{v} \cdot \sum_{n=0}^\infty \left(\frac{u}{v}\right)^n = \frac{d}{v} \cdot \frac{1}{1 - u/v} = \frac{d}{v-u}$$

This is the correct finite time — the geometric series converges. The limit $\lim_{N\to\infty} \sum_{n=0}^N (d/v)(u/v)^n = d/(v-u)$ is finite and gives the time Achilles catches the tortoise. Zeno's error was assuming an infinite series must have an infinite sum — a mistake corrected by the theory of limits.

**Lesson:** Infinite processes can have finite limits. The notion $\lim_{N\to\infty}$ is precisely the mathematical resolution to Zeno's paradox.

### P.2 Berkeley's Objection: "Ghosts of Departed Quantities"

Bishop George Berkeley (1734), in *The Analyst*, critiqued Newton's infinitesimals:

> "And what are these Fluxions? The Velocities of evanescent Increments. And what are these same evanescent Increments? They are neither finite Quantities, nor Quantities infinitely small, nor yet nothing. May we not call them the Ghosts of departed Quantities?"

Berkeley's critique was valid: Newton used $h \neq 0$ to divide, then set $h = 0$ to drop higher-order terms. This is logically inconsistent.

**The resolution:** Weierstrass's ε-δ definition never actually "sets $h = 0$" — instead, it asks: for every $\varepsilon > 0$, can we find $\delta > 0$ (with $h \neq 0$) such that the difference quotient is within $\varepsilon$ of $f'(a)$? This makes no appeal to $h = 0$ — only to inequalities between real numbers. Berkeley's ghost is exorcised by phrasing everything as "$h$ approaches but never reaches $0$."

### P.3 Dirichlet's Proof of the Convergence of Fourier Series

Dirichlet (1829) gave the first rigorous proof that the Fourier series of a "reasonable" function converges to the function at points of continuity. The key ingredient was precise limit analysis: showing that the Dirichlet kernel $D_N(x) = \frac{\sin((N+1/2)x)}{\sin(x/2)}$ satisfies $\lim_{N\to\infty} \int D_N(x) f(a - x) dx = f(a)$ at continuity points.

This required careful ε-δ arguments for limits of integrals — exactly the kind of rigorous limit theory that Weierstrass would later systematize.

**For AI:** Fourier analysis (§20) is the mathematical foundation of signal processing. The convergence of Fourier series is a limit statement — the same limit theory studied here extends to function spaces and ensures that truncated Fourier representations converge to the original signal at points of continuity.

### P.4 Cauchy's Error and Its Correction

Cauchy (1821) stated (incorrectly): "The limit of a sum of continuous functions is continuous." This is false in general — the pointwise limit of continuous functions need not be continuous.

**Counterexample:** $f_n(x) = x^n$ on $[0,1]$. Each $f_n$ is continuous. The pointwise limit:
$$f(x) = \lim_{n\to\infty} x^n = \begin{cases} 0 & 0 \leq x < 1 \\ 1 & x = 1 \end{cases}$$
is discontinuous at $x = 1$. Cauchy's error!

**Correction:** If the convergence is *uniform* (i.e., $\sup_x \lvert f_n(x) - f(x) \rvert \to 0$), then the limit is continuous. This stronger notion — **uniform convergence** — is the correct condition, discovered by Stokes and Seidel (1847).

**For AI:** Neural network functions $f_\theta$ trained on finite data may not converge uniformly to the true function — only pointwise (or in some norm). The distinction between pointwise and uniform convergence explains generalization gaps: the network may correctly learn the training points but fail elsewhere.

---

## Appendix Q: Summary of All Proof Techniques

| Technique | When to Use | Key Idea |
|---|---|---|
| **Direct substitution** | $f$ continuous at $a$ | $\lim_{x\to a}f(x) = f(a)$ |
| **Factoring** | $0/0$ form with polynomial | Cancel common factor $(x-a)$ |
| **Rationalization** | Square roots in numerator or denominator | Multiply by conjugate |
| **ε-δ construction** | Proving a limit rigorously | Bound $\|f(x)-L\|$ in terms of $\|x-a\|$; choose $\delta = f(\varepsilon)$ |
| **Squeeze Theorem** | $f$ is bounded between two functions with known limit | Find $h \leq f \leq g$ with $h,g \to L$ |
| **L'Hôpital's Rule** | $0/0$ or $\infty/\infty$ form | Replace $f/g$ by $f'/g'$ |
| **Cauchy's MVT** | Proving L'Hôpital, comparing rates | $f(x)-f(a) / g(x)-g(a) = f'(c)/g'(c)$ for some $c$ |
| **Series expansion** | Polynomial-like behavior near $0$ | Expand $e^x, \sin x, \ln(1+x)$ in Taylor series |
| **Substitution** | Simplify the variable | Replace $x$ by $u = \varphi(x)$ |
| **Sequential argument** | Proving limit DNE | Find two sequences approaching $a$ with different limits |
| **Limsup/liminf** | Oscillating functions | Compute $\limsup$ and $\liminf$; limit exists iff they agree |

---

## Appendix R: Connections to Linear Algebra

### R.1 Limits of Matrix Sequences

The theory of limits extends to matrices via matrix norms. For a sequence of matrices $\{A_k\}_{k=1}^\infty$:

**Definition.** $\lim_{k\to\infty} A_k = A$ (in norm) means $\lim_{k\to\infty} \lVert A_k - A \rVert = 0$ for any matrix norm $\lVert \cdot \rVert$.

**Matrix exponential as a limit:**
$$e^A = \lim_{N\to\infty} \sum_{k=0}^N \frac{A^k}{k!} = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \cdots$$

This series converges absolutely (in any norm) for all matrices $A \in \mathbb{R}^{n\times n}$.

**Power iteration:** The sequence $\mathbf{v}_{k+1} = A\mathbf{v}_k / \lVert A\mathbf{v}_k \rVert$ converges (under mild conditions) to the eigenvector corresponding to the largest eigenvalue:
$$\lim_{k\to\infty} \mathbf{v}_k = \mathbf{u}_1 \quad \text{(dominant eigenvector)}$$

This is a limit of a vector sequence — continuity of the norm and eigenvalue structure determine convergence.

### R.2 Spectral Radius and Stability

**Spectral radius:** $\rho(A) = \max_i \lvert\lambda_i\rvert$ (largest magnitude eigenvalue).

**Theorem (Gelfand's formula):**
$$\rho(A) = \lim_{k\to\infty} \lVert A^k \rVert^{1/k}$$

This limit always exists and equals the spectral radius — a remarkable result connecting limits of norms to eigenvalue structure.

**Stability of linear systems:** The iteration $\mathbf{x}_{k+1} = A\mathbf{x}_k$ converges to $\mathbf{0}$ for any initial $\mathbf{x}_0$ if and only if $\rho(A) < 1$:
$$\rho(A) < 1 \iff \lim_{k\to\infty} A^k = 0$$

**For AI — gradient descent as a linear system:** Near a minimum $\theta^*$, gradient descent $\theta_{k+1} = \theta_k - \alpha \nabla^2\mathcal{L}(\theta^*)(\theta_k - \theta^*)$ is a linear iteration with matrix $I - \alpha H$ (where $H$ is the Hessian). Convergence requires $\rho(I - \alpha H) < 1$, i.e., $\lvert 1 - \alpha\lambda_i \rvert < 1$ for all eigenvalues $\lambda_i > 0$ of $H$. This gives the condition $0 < \alpha < 2/\lambda_{\max}$ — a limit-theory result on convergence of gradient descent.

### R.3 Condition Number and Sensitivity Analysis

The condition number $\kappa(A) = \lVert A \rVert \lVert A^{-1} \rVert$ measures the sensitivity of $Ax = b$ to perturbations in $b$:

$$\frac{\lVert \delta x \rVert}{\lVert x \rVert} \leq \kappa(A) \frac{\lVert \delta b \rVert}{\lVert b \rVert}$$

In the limit $\kappa(A) \to \infty$ (ill-conditioned), a tiny relative perturbation $\delta b$ causes an arbitrarily large relative change in the solution $x$. This is the matrix-level analog of the numerical instability (catastrophic cancellation) discussed in Section 8 — both arise from near-singularity, and both are limit phenomena.

---

## Appendix S: Python and NumPy Reference for Limit Computations

```python
import numpy as np

# === Numerically stable computations near limits ===

# (e^x - 1)/x near x=0: use expm1
def f_stable(x):
    """Compute (e^x - 1)/x stably for small x."""
    return np.expm1(x) / x  # NOT (np.exp(x) - 1) / x

# log(1 + x)/x near x=0: use log1p
def g_stable(x):
    """Compute log(1+x)/x stably for small x."""
    return np.log1p(x) / x  # NOT np.log(1+x) / x

# Stable softmax
def softmax_stable(z):
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum()

# Stable log-softmax (for cross-entropy)
def log_softmax_stable(z):
    m = np.max(z)
    return z - m - np.log(np.sum(np.exp(z - m)))

# Stable sigmoid
def sigmoid_stable(x):
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))

# === Numerical limit approximation ===

def numerical_limit(f, a, h_values=None):
    """Numerically approximate lim_{x->a} f(x) via centered differences."""
    if h_values is None:
        h_values = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    print(f"{'h':>12} | {'f(a+h)':>15} | {'f(a-h)':>15} | {'avg':>15}")
    print("-" * 65)
    for h in h_values:
        fph = f(a + h)
        fmh = f(a - h)
        avg = (fph + fmh) / 2
        print(f"{h:>12.2e} | {fph:>15.10f} | {fmh:>15.10f} | {avg:>15.10f}")

# Example: lim_{x->0} sin(x)/x = 1
numerical_limit(lambda x: np.sin(x)/x if x != 0 else 1.0, a=1e-14)

# === Gradient checking ===

def grad_check(f, theta, i, h=1e-5):
    """Centered finite difference for d/d(theta_i) f(theta)."""
    theta_plus = theta.copy(); theta_plus[i] += h
    theta_minus = theta.copy(); theta_minus[i] -= h
    return (f(theta_plus) - f(theta_minus)) / (2 * h)
```

**Key takeaway:** The choice of `h` in `grad_check` uses $h = 10^{-5}$ — the optimal value from Section 8.3 (centered finite difference has error $O(h^2)$, minimized around $h \approx u^{1/3}$ where $u = 2^{-52}$).

---

*End of Appendices. See [theory.ipynb](theory.ipynb) for interactive examples and [exercises.ipynb](exercises.ipynb) for graded problems.*

---

## Appendix T: Summary of Key Results

| Result | Statement | Where Proved |
|--------|-----------|--------------|
| ε-δ limit definition | $\forall\varepsilon>0,\exists\delta>0: 0<|x-a|<\delta\Rightarrow|f(x)-L|<\varepsilon$ | §2.1 |
| Limit laws | Sum, product, quotient of limits | §2.2 |
| Squeeze Theorem | $h\leq f\leq g$, $h,g\to L$ $\Rightarrow$ $f\to L$ | §4.3, App. K.2 |
| $\lim_{x\to 0}\sin(x)/x=1$ | Geometric area argument | §3.1, App. K.2 |
| $\lim_{x\to 0}(e^x-1)/x=1$ | Taylor series / definition of $e^x$ derivative | §3.2 |
| $\lim(1+1/n)^n = e$ | Monotone convergence + binomial theorem | §3.3, App. K.1 |
| $\lim_{x\to 0^+}x\ln x=0$ | L'Hôpital on $\ln x/(1/x)$ | §3.4 |
| L'Hôpital's Rule | $\lim f/g = \lim f'/g'$ for $0/0$ or $\infty/\infty$ | §4.2, App. K.3 |
| Continuity definition | All three conditions: existence, limit, equality | §5.1 |
| Intermediate Value Theorem | Continuous $f$ on $[a,b]$ hits every intermediate value | §6.1, App. A.4 |
| Extreme Value Theorem | Continuous $f$ on $[a,b]$ attains max and min | §6.2 |
| Heine-Cantor Theorem | Continuous on $[a,b]$ $\Rightarrow$ uniformly continuous | §6.3 |
| Gelfand's formula | $\rho(A)=\lim_k\|A^k\|^{1/k}$ | App. R.2 |
| Robbins-Monro conditions | SGD converges iff $\sum\alpha_t=\infty$ and $\sum\alpha_t^2<\infty$ | §9.4 |
| Softmax temperature limits | $T\to 0$: argmax; $T\to\infty$: uniform | §9.1 |
| Vanishing gradient rate | $(1/4)^{L-l}$ for $L-l$-deep sigmoid networks | App. H.2 |

---

[← Back to Calculus Fundamentals](../README.md) | [Next: Derivatives and Differentiation →](../02-Derivatives-and-Differentiation/notes.md)

<!-- Section word count target: 2000+ lines. Sections covered: 12 main + 20 appendices.
     Core content: ε-δ, limit laws, one-sided, infinity, fundamental limits, L'Hôpital,
     Squeeze, continuity, IVT, EVT, uniform continuity, asymptotics, numerical stability,
     ML applications (softmax, sigmoid, ReLU, GELU, Robbins-Monro, gradient-as-limit).
     Appendices: proofs, worked examples, advanced math, numerical methods, probability,
     glossary, references, ML extensions, Python reference, linear algebra connections. -->
