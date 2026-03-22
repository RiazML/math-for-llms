[← Back to Calculus Fundamentals](../README.md) | [Next: Series and Sequences →](../04-Series-and-Sequences/notes.md)

---

# Integration

> _"Integration is not just finding areas — it is the mathematics of accumulation, and accumulation is the mathematics of learning."_

## Overview

Integration is the second pillar of calculus, dual to differentiation. Where the derivative asks "how fast is this changing right now?", the integral asks "how much has accumulated over this interval?" The Fundamental Theorem of Calculus (FTC) reveals these two operations as inverses of each other — one of the most profound connections in mathematics.

Every core operation in machine learning involves integration. The expected loss $\mathbb{E}[\mathcal{L}]$ is an integral over a data distribution. KL divergence and entropy — the foundations of information-theoretic learning — are improper integrals of probability densities. Stochastic gradient descent is a Monte Carlo estimator of a gradient integral. Normalizing flows use the change-of-variables formula from integration theory. Understanding integration at the level of definitions and proofs, not just formulas, separates practitioners who can innovate from those who can only apply recipes.

This section builds the complete single-variable integration theory: Riemann sums, the FTC, all standard techniques (substitution, parts, partial fractions), improper integrals, numerical methods (trapezoid, Simpson's, Monte Carlo), and integration in probability theory. The treatment is rigorous but always anchored to concrete ML applications.

## Prerequisites

- **Limits and continuity** — [§01-Limits-and-Continuity](../01-Limits-and-Continuity/notes.md): limit definition, squeeze theorem, continuity on $[a,b]$
- **Derivatives and differentiation** — [§02-Derivatives-and-Differentiation](../02-Derivatives-and-Differentiation/notes.md): chain rule, product rule, all elementary derivatives
- **Algebra**: polynomial long division, factoring, partial fraction setup
- **Trigonometry**: $\sin$, $\cos$, $\tan$ identities; inverse trig derivatives

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Riemann sums, FTC, all techniques, numerical methods, probability integration |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from basic antiderivatives to Monte Carlo and KL divergence |

## Learning Objectives

After completing this section, you will:

- Define the definite integral via Riemann sums and compute it as a limit
- State and prove both parts of the Fundamental Theorem of Calculus
- Apply u-substitution, integration by parts, and partial fractions to evaluate integrals
- Classify and evaluate improper integrals using limit definitions and convergence tests
- Implement trapezoid, Simpson's, and Monte Carlo numerical integration
- Express expectation, variance, KL divergence, and entropy as integrals
- Explain why stochastic gradient descent is a Monte Carlo estimator of the gradient integral
- Verify numerical integration accuracy and analyze error bounds

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Accumulation as the Core Idea](#11-accumulation-as-the-core-idea)
  - [1.2 Historical Motivation](#12-historical-motivation)
  - [1.3 Why Integration Is Central to AI](#13-why-integration-is-central-to-ai)
- [2. The Definite Integral — Riemann's Definition](#2-the-definite-integral--riemanns-definition)
  - [2.1 Partitions and Riemann Sums](#21-partitions-and-riemann-sums)
  - [2.2 The Limit Definition](#22-the-limit-definition)
  - [2.3 Geometric Interpretation: Signed Area](#23-geometric-interpretation-signed-area)
  - [2.4 Properties of the Definite Integral](#24-properties-of-the-definite-integral)
  - [2.5 Integrability Conditions](#25-integrability-conditions)
- [3. Antiderivatives and the Indefinite Integral](#3-antiderivatives-and-the-indefinite-integral)
  - [3.1 Definition and the +C Convention](#31-definition-and-the-c-convention)
  - [3.2 Basic Antiderivative Table](#32-basic-antiderivative-table)
  - [3.3 Linearity](#33-linearity)
  - [3.4 Initial Value Problems](#34-initial-value-problems)
- [4. The Fundamental Theorem of Calculus](#4-the-fundamental-theorem-of-calculus)
  - [4.1 FTC Part 1](#41-ftc-part-1)
  - [4.2 FTC Part 2](#42-ftc-part-2)
  - [4.3 The Bridge](#43-the-bridge)
  - [4.4 Worked Examples](#44-worked-examples)
  - [4.5 FTC and Automatic Differentiation](#45-ftc-and-automatic-differentiation)
- [5. Integration by Substitution](#5-integration-by-substitution)
  - [5.1 The Reverse Chain Rule](#51-the-reverse-chain-rule)
  - [5.2 Definite Integrals: Changing Limits](#52-definite-integrals-changing-limits)
  - [5.3 Worked Examples](#53-worked-examples)
  - [5.4 For AI: Change of Variables in Flows](#54-for-ai-change-of-variables-in-flows)
- [6. Integration by Parts](#6-integration-by-parts)
  - [6.1 The Reverse Product Rule](#61-the-reverse-product-rule)
  - [6.2 Choosing u and dv — LIATE](#62-choosing-u-and-dv--liate)
  - [6.3 Reduction Formulas and Tabular Method](#63-reduction-formulas-and-tabular-method)
  - [6.4 Worked Examples](#64-worked-examples)
  - [6.5 For AI: REINFORCE Gradient Estimator](#65-for-ai-reinforce-gradient-estimator)
- [7. Partial Fractions](#7-partial-fractions)
  - [7.1 Decomposing Rational Functions](#71-decomposing-rational-functions)
  - [7.2 Cases and Worked Examples](#72-cases-and-worked-examples)
- [8. Improper Integrals](#8-improper-integrals)
  - [8.1 Type I: Infinite Limits](#81-type-i-infinite-limits)
  - [8.2 Type II: Unbounded Integrands](#82-type-ii-unbounded-integrands)
  - [8.3 Convergence Tests](#83-convergence-tests)
  - [8.4 The Gaussian Integral](#84-the-gaussian-integral)
  - [8.5 For AI: Entropy, KL Divergence, Expected Loss](#85-for-ai-entropy-kl-divergence-expected-loss)
- [9. Numerical Integration](#9-numerical-integration)
  - [9.1 Trapezoid Rule](#91-trapezoid-rule)
  - [9.2 Simpson's Rule](#92-simpsons-rule)
  - [9.3 Gaussian Quadrature](#93-gaussian-quadrature)
  - [9.4 Monte Carlo Integration](#94-monte-carlo-integration)
  - [9.5 For AI: SGD as Monte Carlo Expectation](#95-for-ai-sgd-as-monte-carlo-expectation)
- [10. Integration in Probability](#10-integration-in-probability)
  - [10.1 Probability Density Functions](#101-probability-density-functions)
  - [10.2 CDF and FTC](#102-cdf-and-ftc)
  - [10.3 Expectation as a Weighted Integral](#103-expectation-as-a-weighted-integral)
  - [10.4 Variance and Second Moments](#104-variance-and-second-moments)
  - [10.5 KL Divergence](#105-kl-divergence)
  - [10.6 Entropy](#106-entropy)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [14. Conceptual Bridge](#14-conceptual-bridge)

---
## 1. Intuition

### 1.1 Accumulation as the Core Idea

The derivative measures instantaneous rate of change. The integral measures total accumulation. These are two sides of the same coin — and the FTC is the theorem that makes this precise.

**Physical intuition.** If $v(t)$ is velocity at time $t$, then $\int_a^b v(t)\,dt$ is the total distance traveled from time $a$ to $b$. Each infinitesimal time slice $dt$ contributes a tiny distance $v(t)\,dt$; the integral sums them all.

**Geometric intuition.** $\int_a^b f(x)\,dx$ is the signed area between the curve $y = f(x)$ and the $x$-axis over $[a,b]$. Area above the axis is positive; area below is negative.

**Probabilistic intuition.** If $p(x)$ is a probability density, $\int_a^b p(x)\,dx$ is the probability that a random draw falls in $[a,b]$. Integration is the language of probability.

### 1.2 Historical Motivation

| Year | Development |
|------|-------------|
| ~250 BCE | Archimedes uses method of exhaustion to compute areas of parabolas |
| 1670s | Newton and Leibniz independently develop calculus; Leibniz introduces $\int$ notation |
| 1823 | Cauchy gives rigorous definition of the definite integral |
| 1854 | Riemann formalizes the integral via sums — the definition used today |
| 1875 | Darboux introduces upper/lower sums, simplifying Riemann's theory |
| 1902 | Lebesgue generalizes the integral to a much broader class of functions |

The $\int$ symbol is an elongated "S" for _summa_ (Latin: sum) — Leibniz's notation for the limit of infinitely many infinitesimal summands.

### 1.3 Why Integration Is Central to AI

**Expected loss.** The training objective of every probabilistic model is:

$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{x},y)\sim p_{\text{data}}}[\ell(f_\theta(\mathbf{x}), y)] = \int \ell(f_\theta(\mathbf{x}), y)\,p(\mathbf{x},y)\,d\mathbf{x}\,dy$$

This is an integral. Stochastic gradient descent estimates it via a Monte Carlo sum over a mini-batch.

**KL divergence.** The loss function used in variational autoencoders (VAEs), diffusion models, and language model alignment (RLHF):

$$\text{KL}(q\|p) = \int q(x)\ln\frac{q(x)}{p(x)}\,dx$$

**Entropy.** The Shannon entropy of a continuous distribution:

$$H(p) = -\int p(x)\ln p(x)\,dx$$

This integral appears in cross-entropy loss, mutual information, and the information-theoretic analysis of generalization.

**Normalizing flows.** Change-of-variables: if $\mathbf{z} = f(\mathbf{x})$ is a bijection, the density transforms as:

$$p_X(\mathbf{x}) = p_Z(f(\mathbf{x}))\left|\det\frac{\partial f}{\partial \mathbf{x}}\right|$$

The absolute Jacobian determinant is the multidimensional version of the substitution rule from this section.

**Gaussian integral.** The normalization constant of every Gaussian distribution relies on $\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi}$ — an improper integral computed in §8.4.

> **For AI:** Every backpropagation pass computes a stochastic estimate of the gradient of an integral (the expected loss) — integration and differentiation are inseparable in machine learning.


---

## 2. The Definite Integral — Riemann's Definition

### 2.1 Partitions and Riemann Sums

A **partition** of $[a,b]$ is a finite set of points $a = x_0 < x_1 < \cdots < x_n = b$. The **mesh** (or norm) is $\|\mathcal{P}\| = \max_k(x_k - x_{k-1})$.

For each subinterval $[x_{k-1}, x_k]$ of width $\Delta x_k = x_k - x_{k-1}$, choose a **sample point** $x_k^* \in [x_{k-1}, x_k]$. The **Riemann sum** is:

$$S(\mathcal{P}, f) = \sum_{k=1}^n f(x_k^*)\,\Delta x_k$$

**Three standard choices of $x_k^*$:**
- **Left endpoint:** $x_k^* = x_{k-1}$ → left Riemann sum $L_n$
- **Right endpoint:** $x_k^* = x_k$ → right Riemann sum $R_n$
- **Midpoint:** $x_k^* = (x_{k-1}+x_k)/2$ → midpoint Riemann sum $M_n$

**Example.** $f(x) = x^2$ on $[0,1]$, uniform partition with $n$ intervals, right endpoints:

$$R_n = \sum_{k=1}^n \left(\frac{k}{n}\right)^2 \cdot \frac{1}{n} = \frac{1}{n^3}\sum_{k=1}^n k^2 = \frac{1}{n^3}\cdot\frac{n(n+1)(2n+1)}{6} \xrightarrow{n\to\infty} \frac{1}{3}$$

### 2.2 The Limit Definition

The **definite integral** of $f$ over $[a,b]$ is:

$$\int_a^b f(x)\,dx = \lim_{\|\mathcal{P}\|\to 0} S(\mathcal{P}, f)$$

provided this limit exists and is the same for every choice of sample points. When this limit exists, $f$ is **Riemann integrable** on $[a,b]$.

For uniform partitions ($\Delta x = (b-a)/n$, right endpoints):

$$\int_a^b f(x)\,dx = \lim_{n\to\infty}\sum_{k=1}^n f\!\left(a + k\cdot\frac{b-a}{n}\right)\cdot\frac{b-a}{n}$$

### 2.3 Geometric Interpretation: Signed Area

The integral computes **signed area**: regions where $f(x) > 0$ contribute positively; regions where $f(x) < 0$ contribute negatively.

$$\int_0^{2\pi}\sin x\,dx = 0 \quad \text{(positive and negative areas cancel)}$$

$$\int_0^{2\pi}|\sin x|\,dx = 4 \quad \text{(total unsigned area)}$$

### 2.4 Properties of the Definite Integral

For integrable $f, g$ on $[a,b]$:

| Property | Formula |
|----------|---------|
| Linearity | $\int_a^b [cf(x)+g(x)]\,dx = c\int_a^b f(x)\,dx + \int_a^b g(x)\,dx$ |
| Additivity | $\int_a^b f\,dx = \int_a^c f\,dx + \int_c^b f\,dx$ for any $c \in [a,b]$ |
| Monotonicity | $f \leq g \Rightarrow \int_a^b f\,dx \leq \int_a^b g\,dx$ |
| Reverse limits | $\int_b^a f\,dx = -\int_a^b f\,dx$ |
| Zero-width | $\int_a^a f\,dx = 0$ |
| Bound | $\left|\int_a^b f\,dx\right| \leq \int_a^b |f|\,dx \leq M(b-a)$ if $|f| \leq M$ |

**Mean Value Theorem for Integrals.** If $f$ is continuous on $[a,b]$, there exists $c \in (a,b)$ with:

$$f(c) = \frac{1}{b-a}\int_a^b f(x)\,dx$$

The integral equals the function value at some interior point times the interval length.

### 2.5 Integrability Conditions

**Sufficient condition 1:** If $f$ is continuous on $[a,b]$, then $f$ is Riemann integrable. (This covers all smooth functions and all activation functions in ML.)

**Sufficient condition 2:** If $f$ is bounded and monotone on $[a,b]$, then $f$ is Riemann integrable.

**Sufficient condition 3:** If $f$ is bounded on $[a,b]$ and has only finitely many discontinuities, then $f$ is Riemann integrable. (This covers ReLU, which is discontinuous in derivative at 0 but not in value.)


---

## 3. Antiderivatives and the Indefinite Integral

### 3.1 Definition and the +C Convention

A function $F$ is an **antiderivative** of $f$ on an interval $I$ if $F'(x) = f(x)$ for all $x \in I$.

**Theorem (Uniqueness up to constant).** If $F$ and $G$ are both antiderivatives of $f$ on $I$, then $F(x) - G(x) = C$ for some constant $C$.

**Proof.** Let $H = F - G$. Then $H'(x) = F'(x) - G'(x) = f(x) - f(x) = 0$ for all $x \in I$. By the MVT corollary, $H' \equiv 0$ implies $H$ is constant. $\square$

The **indefinite integral** encodes the entire family of antiderivatives:

$$\int f(x)\,dx = F(x) + C$$

where $C$ is an arbitrary constant. The $+C$ is not optional — it represents a genuinely different function for each value of $C$.

### 3.2 Basic Antiderivative Table

| $f(x)$ | $\int f(x)\,dx$ | Condition |
|--------|----------------|-----------|
| $x^n$ | $\dfrac{x^{n+1}}{n+1} + C$ | $n \neq -1$ |
| $x^{-1} = 1/x$ | $\ln|x| + C$ | $x \neq 0$ |
| $e^x$ | $e^x + C$ | |
| $a^x$ | $\dfrac{a^x}{\ln a} + C$ | $a > 0, a\neq 1$ |
| $\sin x$ | $-\cos x + C$ | |
| $\cos x$ | $\sin x + C$ | |
| $\sec^2 x$ | $\tan x + C$ | |
| $1/\sqrt{1-x^2}$ | $\arcsin x + C$ | $|x| < 1$ |
| $1/(1+x^2)$ | $\arctan x + C$ | |
| $\sinh x$ | $\cosh x + C$ | |
| $\cosh x$ | $\sinh x + C$ | |

**Verification:** Every row can be checked by differentiating the right side.

### 3.3 Linearity

$$\int [cf(x) + g(x)]\,dx = c\int f(x)\,dx + \int g(x)\,dx$$

This follows immediately from the linearity of differentiation.

**Example.** $\int(3x^2 - 5\cos x + e^x)\,dx = x^3 - 5\sin x + e^x + C$.

### 3.4 Initial Value Problems

An **initial value problem** (IVP) specifies $f'(x)$ and an initial condition $f(x_0) = y_0$, and asks for $f(x)$.

**Procedure:**
1. Find the general antiderivative: $f(x) = \int f'(x)\,dx = F(x) + C$
2. Apply the initial condition: $y_0 = F(x_0) + C \Rightarrow C = y_0 - F(x_0)$

**Example.** $f'(x) = 3x^2 - 2x$, $f(1) = 5$.

$f(x) = x^3 - x^2 + C$. At $x=1$: $1 - 1 + C = 5 \Rightarrow C = 5$.

So $f(x) = x^3 - x^2 + 5$.

**For AI.** Solving $\dot{\theta} = -\nabla\mathcal{L}(\theta)$ as an ODE gives the continuous-time analogue of gradient descent. The solution is an integral: $\theta(T) = \theta(0) - \int_0^T \nabla\mathcal{L}(\theta(t))\,dt$. Neural ODEs (Chen et al., 2018) make this explicit by parameterizing the dynamics as a neural network and solving the integral numerically via an ODE solver.


---

## 4. The Fundamental Theorem of Calculus

The FTC is the central theorem of calculus — it reveals that differentiation and integration are inverse operations and provides a practical method for evaluating definite integrals.

### 4.1 FTC Part 1

**Theorem (FTC Part 1).** Let $f$ be continuous on $[a,b]$ and define:

$$G(x) = \int_a^x f(t)\,dt, \quad x \in [a,b]$$

Then $G$ is differentiable on $(a,b)$ and $G'(x) = f(x)$.

**Proof.** For $h > 0$:

$$\frac{G(x+h) - G(x)}{h} = \frac{1}{h}\int_x^{x+h} f(t)\,dt$$

By the MVT for integrals, there exists $c_h \in (x, x+h)$ with $\frac{1}{h}\int_x^{x+h} f(t)\,dt = f(c_h)$.

As $h \to 0^+$: $c_h \to x$, so by continuity of $f$: $f(c_h) \to f(x)$. A symmetric argument handles $h \to 0^-$. Therefore $G'(x) = f(x)$. $\square$

**Interpretation.** The area-accumulation function $G(x) = \int_a^x f(t)\,dt$ has derivative $f(x)$ — the rate of growth of accumulated area at $x$ equals the function value $f(x)$. This is obvious geometrically: adding a thin strip of height $f(x)$ and width $h$ gives area $\approx f(x) \cdot h$.

**Generalization (Leibniz rule):**

$$\frac{d}{dx}\int_{g(x)}^{h(x)} f(t)\,dt = f(h(x))\cdot h'(x) - f(g(x))\cdot g'(x)$$

### 4.2 FTC Part 2

**Theorem (FTC Part 2).** If $f$ is continuous on $[a,b]$ and $F$ is any antiderivative of $f$, then:

$$\int_a^b f(x)\,dx = F(b) - F(a) \equiv \Big[F(x)\Big]_a^b$$

**Proof.** Let $G(x) = \int_a^x f(t)\,dt$. By Part 1, $G'(x) = f(x) = F'(x)$. So $G(x) - F(x) = C$ (constant). At $x = a$: $G(a) - F(a) = 0 - F(a)$, giving $C = -F(a)$. At $x = b$: $G(b) = \int_a^b f(t)\,dt = F(b) - F(a)$. $\square$

### 4.3 The Bridge

**Why FTC Part 2 is powerful.** Before the FTC, computing $\int_a^b f(x)\,dx$ required constructing Riemann sums and taking limits — laborious for any non-trivial function. The FTC reduces this to: find any antiderivative $F$, evaluate at $b$ and $a$, subtract.

**Example.** $\int_0^1 x^2\,dx = \Big[\frac{x^3}{3}\Big]_0^1 = \frac{1}{3} - 0 = \frac{1}{3}$. (Recall: directly computing via Riemann sums required summing $\sum k^2$.)

### 4.4 Worked Examples

**Example 1.** $\int_1^e \frac{1}{x}\,dx = [\ln x]_1^e = \ln e - \ln 1 = 1 - 0 = 1$.

**Example 2.** $\int_0^\pi \sin x\,dx = [-\cos x]_0^\pi = (-\cos\pi) - (-\cos 0) = 1 + 1 = 2$.

**Example 3.** $\int_{-1}^{1} e^x\,dx = [e^x]_{-1}^1 = e - e^{-1} = e - 1/e$.

**Example 4 (net displacement vs distance).** A particle has velocity $v(t) = t^2 - 4$. On $[0,3]$:

$$\text{Net displacement} = \int_0^3 (t^2-4)\,dt = \left[\frac{t^3}{3} - 4t\right]_0^3 = 9-12 = -3$$

$$\text{Distance} = \int_0^3 |t^2-4|\,dt = \int_0^2(4-t^2)\,dt + \int_2^3(t^2-4)\,dt = \frac{16}{3} + \frac{7}{3} = \frac{23}{3}$$

### 4.5 FTC and Automatic Differentiation

FTC Part 1 has a direct counterpart in modern ML: **the adjoint method** for training neural ODEs. The derivative of a loss $\mathcal{L}$ with respect to the initial state $\mathbf{z}(0)$, where $\mathbf{z}(T) = \mathbf{z}(0) + \int_0^T f(\mathbf{z}(t),t;\theta)\,dt$, is computed via an integral that runs backward in time. This is FTC Part 1 applied to the adjoint state — the computational core of the `torchdiffeq` library.


---

## 5. Integration by Substitution

### 5.1 The Reverse Chain Rule

**Theorem.** If $u = g(x)$ is differentiable and $f$ is continuous on the range of $g$, then:

$$\int f(g(x))\,g'(x)\,dx = \int f(u)\,du \quad \text{(evaluated at } u = g(x)\text{)}$$

**Proof.** Let $F$ be an antiderivative of $f$, so $F' = f$. By the chain rule:

$$\frac{d}{dx}[F(g(x))] = F'(g(x))\cdot g'(x) = f(g(x))\cdot g'(x)$$

Therefore $F(g(x))$ is an antiderivative of $f(g(x))\cdot g'(x)$. $\square$

**Procedure:**
1. Identify a function $u = g(x)$ whose derivative $g'(x)$ appears (or nearly appears) in the integrand.
2. Compute $du = g'(x)\,dx$.
3. Substitute: replace $g(x)$ with $u$ and $g'(x)\,dx$ with $du$.
4. Integrate in $u$.
5. Back-substitute: replace $u$ with $g(x)$.

### 5.2 Definite Integrals: Changing Limits

For $\int_a^b f(g(x))g'(x)\,dx$ with $u = g(x)$: change limits to $u(a)$ and $u(b)$:

$$\int_a^b f(g(x))g'(x)\,dx = \int_{g(a)}^{g(b)} f(u)\,du$$

This avoids back-substitution at the end.

### 5.3 Worked Examples

**Example 1.** $\int \sin(x^2)\cdot 2x\,dx$. Let $u = x^2$, $du = 2x\,dx$:
$$= \int \sin u\,du = -\cos u + C = -\cos(x^2) + C$$

**Example 2.** $\int_0^1 e^{-x^2}\cdot(-2x)\,dx$. Let $u = -x^2$, limits: $u(0)=0$, $u(1)=-1$:
$$= \int_0^{-1} e^u\,du = [e^u]_0^{-1} = e^{-1} - 1$$

**Example 3.** $\int \frac{x}{x^2+1}\,dx$. Let $u = x^2+1$, $du = 2x\,dx$:
$$= \frac{1}{2}\int\frac{du}{u} = \frac{1}{2}\ln|u| + C = \frac{1}{2}\ln(x^2+1) + C$$

**Example 4.** $\int \tan x\,dx = \int \frac{\sin x}{\cos x}\,dx$. Let $u = \cos x$, $du = -\sin x\,dx$:
$$= -\int\frac{du}{u} = -\ln|\cos x| + C = \ln|\sec x| + C$$

**Example 5 (softmax normalization).** For $\mathbf{z} \in \mathbb{R}^K$:

$$Z = \sum_{k=1}^K e^{z_k} = e^{z_{\max}}\sum_{k=1}^K e^{z_k - z_{\max}}$$

The subtraction of $z_{\max}$ is a discrete analog of substitution $u = z - z_{\max}$, making all terms $\leq 1$ for numerical stability.

### 5.4 For AI: Change of Variables in Normalizing Flows

A normalizing flow defines a bijection $\mathbf{x} = f(\mathbf{z})$ where $\mathbf{z} \sim p_Z$. The change-of-variables formula:

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x}))\left|\det J_{f^{-1}}(\mathbf{x})\right|$$

is the multivariate version of $u$-substitution: $\int_a^b f(g(x))g'(x)\,dx = \int_{g(a)}^{g(b)}f(u)\,du$, where $|g'(x)|$ becomes the absolute Jacobian determinant. Real NVP, Glow, and FFJORD all implement variants of this transformation to learn complex densities from simple base distributions.


---

## 6. Integration by Parts

### 6.1 The Reverse Product Rule

**Theorem.** If $u(x)$ and $v(x)$ are differentiable, then:

$$\int u\,dv = uv - \int v\,du$$

**Proof.** The product rule gives $(uv)' = u'v + uv'$. Integrating both sides:

$$uv = \int u'v\,dx + \int uv'\,dx$$

Rearranging: $\int uv'\,dx = uv - \int u'v\,dx$. Writing $dv = v'dx$ and $du = u'dx$ gives the formula. $\square$

**For definite integrals:**

$$\int_a^b u\,dv = \Big[uv\Big]_a^b - \int_a^b v\,du$$

### 6.2 Choosing u and dv — LIATE

The acronym **LIATE** gives a priority order for choosing $u$ (choose the type that comes first):
- **L**ogarithms: $\ln x$, $\log_a x$
- **I**nverse trig: $\arctan x$, $\arcsin x$
- **A**lgebraic: polynomials $x^n$, $\sqrt{x}$
- **T**rigonometric: $\sin x$, $\cos x$
- **E**xponential: $e^x$, $a^x$

The complementary factor becomes $dv$ and must be something we can integrate.

### 6.3 Reduction Formulas and Tabular Method

For integrals like $\int x^n e^x\,dx$, repeated integration by parts produces a **reduction formula**:

$$\int x^n e^x\,dx = x^n e^x - n\int x^{n-1}e^x\,dx$$

The **tabular method** (also called "tic-tac-toe" or "successive differentiation") organizes repeated parts efficiently:

| Sign | Differentiate ($u$) | Integrate ($dv$) |
|------|--------------------|--------------------|
| $+$ | $x^3$ | $e^x$ |
| $-$ | $3x^2$ | $e^x$ |
| $+$ | $6x$ | $e^x$ |
| $-$ | $6$ | $e^x$ |
| $+$ | $0$ | $e^x$ |

Result: $\int x^3 e^x\,dx = e^x(x^3 - 3x^2 + 6x - 6) + C$.

### 6.4 Worked Examples

**Example 1.** $\int x e^x\,dx$. Let $u = x$, $dv = e^x\,dx$:

$$= xe^x - \int e^x\,dx = xe^x - e^x + C = e^x(x-1) + C$$

**Example 2.** $\int \ln x\,dx$. Let $u = \ln x$, $dv = dx$:

$$= x\ln x - \int x\cdot\frac{1}{x}\,dx = x\ln x - x + C = x(\ln x - 1) + C$$

**Example 3.** $\int x^2\sin x\,dx$. Apply tabular method (alternating signs):

$$= -x^2\cos x + 2x\sin x + 2\cos x + C$$

**Example 4 (cyclic).** $\int e^x\sin x\,dx$. Let $I = \int e^x\sin x\,dx$. Integrate by parts twice:

$$I = e^x\sin x - e^x\cos x - I \implies 2I = e^x(\sin x - \cos x) \implies I = \frac{e^x(\sin x - \cos x)}{2} + C$$

### 6.5 For AI: REINFORCE Gradient Estimator

The REINFORCE algorithm (Williams, 1992) estimates $\nabla_\theta \mathbb{E}_{\tau\sim p_\theta}[R(\tau)]$ where $\tau$ is a trajectory and $R$ is return. Integration by parts (in the form of the log-derivative trick) gives:

$$\nabla_\theta\mathbb{E}_{x\sim p_\theta}[f(x)] = \mathbb{E}_{x\sim p_\theta}[f(x)\nabla_\theta\log p_\theta(x)]$$

This identity — differentiating through an expectation — is the continuous-distribution version of integration by parts, and is the core of policy gradient methods in reinforcement learning (PPO, GRPO, RLHF fine-tuning).


---

## 7. Partial Fractions

### 7.1 Decomposing Rational Functions

Partial fraction decomposition writes a rational function $P(x)/Q(x)$ (where $\deg P < \deg Q$) as a sum of simpler fractions. This converts integrals of rational functions into sums of basic integrals.

**Setup:**
1. Factor $Q(x)$ completely over $\mathbb{R}$.
2. Write the partial fraction decomposition.
3. Solve for unknown constants (cover-up method or comparing coefficients).
4. Integrate each term.

### 7.2 Cases and Worked Examples

**Case 1: Distinct linear factors.** $Q(x) = (x-a_1)(x-a_2)\cdots(x-a_n)$:

$$\frac{P(x)}{Q(x)} = \frac{A_1}{x-a_1} + \frac{A_2}{x-a_2} + \cdots + \frac{A_n}{x-a_n}$$

**Example.** $\int\frac{1}{x^2-1}\,dx = \int\frac{1}{(x-1)(x+1)}\,dx$.

Decompose: $\frac{1}{(x-1)(x+1)} = \frac{A}{x-1}+\frac{B}{x+1}$.

Multiply both sides by $(x-1)(x+1)$: $1 = A(x+1) + B(x-1)$.

At $x=1$: $1 = 2A \Rightarrow A = 1/2$. At $x=-1$: $1 = -2B \Rightarrow B = -1/2$.

$$\int\frac{1}{x^2-1}\,dx = \frac{1}{2}\ln|x-1| - \frac{1}{2}\ln|x+1| + C = \frac{1}{2}\ln\left|\frac{x-1}{x+1}\right| + C$$

**Case 2: Repeated linear factors.** For $(x-a)^m$:

$$\frac{A_1}{x-a} + \frac{A_2}{(x-a)^2} + \cdots + \frac{A_m}{(x-a)^m}$$

**Example.** $\int\frac{x}{(x-1)^2}\,dx$.

$\frac{x}{(x-1)^2} = \frac{A}{x-1} + \frac{B}{(x-1)^2}$. Multiply: $x = A(x-1) + B$.

At $x=1$: $B=1$. Compare $x$-coefficients: $1 = A$, so $A=1$.

$$\int\frac{x}{(x-1)^2}\,dx = \ln|x-1| - \frac{1}{x-1} + C$$

**Case 3: Irreducible quadratic factors.** For $x^2 + bx + c$ (no real roots):

$$\frac{Ax+B}{x^2+bx+c}$$

**Example.** $\int\frac{1}{x(x^2+1)}\,dx$.

$\frac{1}{x(x^2+1)} = \frac{A}{x} + \frac{Bx+C}{x^2+1}$.

Multiply: $1 = A(x^2+1) + (Bx+C)x$. At $x=0$: $A=1$. Comparing $x^2$: $0=A+B \Rightarrow B=-1$. Comparing $x^1$: $0=C$.

$$\int\frac{1}{x(x^2+1)}\,dx = \ln|x| - \frac{1}{2}\ln(x^2+1) + C = \frac{1}{2}\ln\frac{x^2}{x^2+1} + C$$

**For AI.** Partial fractions appear in z-transform analysis (discrete-time signal processing), computing closed-form solutions to linear recurrences (relevant to LSTM and S4 state space models), and in Laplace transform computations for control-theoretic analysis of learning dynamics.


---

## 8. Improper Integrals

### 8.1 Type I: Infinite Limits

**Definition.** For $f$ continuous on $[a,\infty)$:

$$\int_a^\infty f(x)\,dx = \lim_{b\to\infty}\int_a^b f(x)\,dx$$

If the limit exists and is finite, the integral **converges**; otherwise it **diverges**.

Similarly $\int_{-\infty}^b f(x)\,dx = \lim_{a\to-\infty}\int_a^b f(x)\,dx$ and $\int_{-\infty}^\infty f\,dx = \int_{-\infty}^c f\,dx + \int_c^\infty f\,dx$ for any $c$.

**Example 1 (exponential decay).**

$$\int_0^\infty e^{-x}\,dx = \lim_{b\to\infty}[-e^{-x}]_0^b = \lim_{b\to\infty}(1-e^{-b}) = 1$$

**Example 2 ($p$-integral).**

$$\int_1^\infty \frac{1}{x^p}\,dx = \begin{cases} \dfrac{1}{p-1} & p > 1 \\ \infty & p \leq 1 \end{cases}$$

**Proof.** For $p \neq 1$: $\int_1^b x^{-p}\,dx = \frac{b^{1-p}-1}{1-p}$. As $b\to\infty$: converges iff $1-p < 0$ iff $p > 1$. $\square$

### 8.2 Type II: Unbounded Integrands

**Definition.** If $f$ has a vertical asymptote at $x = a$:

$$\int_a^b f(x)\,dx = \lim_{\varepsilon\to 0^+}\int_{a+\varepsilon}^b f(x)\,dx$$

**Example.** $\int_0^1 \frac{1}{\sqrt{x}}\,dx = \lim_{\varepsilon\to 0^+}[2\sqrt{x}]_\varepsilon^1 = 2 - 0 = 2$. Converges.

$\int_0^1 \frac{1}{x}\,dx = \lim_{\varepsilon\to 0^+}[\ln x]_\varepsilon^1 = 0 - (-\infty) = \infty$. Diverges.

### 8.3 Convergence Tests

**Comparison test.** If $0 \leq f(x) \leq g(x)$ for $x \geq a$:
- $\int_a^\infty g\,dx$ converges $\Rightarrow$ $\int_a^\infty f\,dx$ converges
- $\int_a^\infty f\,dx$ diverges $\Rightarrow$ $\int_a^\infty g\,dx$ diverges

**Limit comparison test.** If $f, g > 0$ and $\lim_{x\to\infty} f(x)/g(x) = L \in (0,\infty)$, then $\int_a^\infty f$ and $\int_a^\infty g$ both converge or both diverge.

**Absolute convergence.** If $\int_a^\infty |f(x)|\,dx < \infty$, then $\int_a^\infty f(x)\,dx$ converges.

### 8.4 The Gaussian Integral

The most important improper integral in probability and machine learning:

$$\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi}$$

**Proof (polar coordinates trick).** Let $I = \int_{-\infty}^\infty e^{-x^2}\,dx$. Then:

$$I^2 = \int_{-\infty}^\infty e^{-x^2}\,dx\cdot\int_{-\infty}^\infty e^{-y^2}\,dy = \int_{-\infty}^\infty\int_{-\infty}^\infty e^{-(x^2+y^2)}\,dx\,dy$$

Convert to polar ($x = r\cos\theta$, $y = r\sin\theta$, $dx\,dy = r\,dr\,d\theta$):

$$I^2 = \int_0^{2\pi}\int_0^\infty e^{-r^2}r\,dr\,d\theta = 2\pi\int_0^\infty re^{-r^2}\,dr = 2\pi\cdot\frac{1}{2} = \pi$$

Therefore $I = \sqrt{\pi}$. $\square$

**Consequence.** The standard Gaussian $\mathcal{N}(0,1)$ normalizes:

$$\int_{-\infty}^\infty \frac{1}{\sqrt{2\pi}}e^{-x^2/2}\,dx = 1$$

(substitution $u = x/\sqrt{2}$ converts to the Gaussian integral).

### 8.5 For AI: Entropy, KL Divergence, and Expected Loss

**Entropy of a Gaussian.** For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$H(X) = -\int_{-\infty}^\infty p(x)\ln p(x)\,dx = \frac{1}{2}\ln(2\pi e\sigma^2)$$

This improper integral converges because $p(x)\ln p(x) \to 0$ faster than any polynomial as $x\to\pm\infty$.

**KL divergence between Gaussians.** For $p = \mathcal{N}(\mu_1,\sigma_1^2)$, $q = \mathcal{N}(\mu_2,\sigma_2^2)$:

$$\text{KL}(p\|q) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

This closed form follows from evaluating $\int_{-\infty}^\infty p(x)[\ln p(x) - \ln q(x)]\,dx$ using the Gaussian integral. It appears as the regularization term in the VAE ELBO objective.

**Expected cross-entropy loss.** The population risk:

$$R(\theta) = -\int p(\mathbf{x},y)\log q_\theta(y|\mathbf{x})\,d\mathbf{x}\,dy$$

is an improper integral over the joint distribution. SGD computes an unbiased Monte Carlo estimate using a mini-batch.


---

## 9. Numerical Integration

When an antiderivative cannot be expressed in closed form (e.g., $\int e^{-x^2}\,dx$, $\int \sin(x^2)\,dx$), numerical methods approximate the definite integral.

### 9.1 Trapezoid Rule

Approximate the integrand on each subinterval by a straight line (trapezoid).

For uniform step $h = (b-a)/n$ and nodes $x_k = a + kh$:

$$\int_a^b f(x)\,dx \approx T_n = \frac{h}{2}\left[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)\right]$$

**Error bound.** If $|f''(x)| \leq M$ on $[a,b]$:

$$|E_T| = \left|\int_a^b f\,dx - T_n\right| \leq \frac{M(b-a)^3}{12n^2} = O(h^2)$$

The trapezoid rule is second-order accurate — halving $h$ reduces error by a factor of 4.

### 9.2 Simpson's Rule

Approximate the integrand on each pair of subintervals by a quadratic (parabola). Requires $n$ even:

$$S_n = \frac{h}{3}\left[f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + 4f(x_{n-1}) + f(x_n)\right]$$

Pattern: 1, 4, 2, 4, 2, ..., 4, 1 with coefficients summing to $2n/3 \cdot 3 = 2n$.

**Error bound.** If $|f^{(4)}(x)| \leq M$:

$$|E_S| \leq \frac{M(b-a)^5}{180n^4} = O(h^4)$$

Simpson's rule is fourth-order — halving $h$ reduces error by a factor of 16.

**Example.** Estimate $\int_0^1 e^x\,dx$ (true value: $e-1 \approx 1.71828$) with $n = 4$:

$h = 0.25$, $x_k \in \{0, 0.25, 0.5, 0.75, 1\}$.

$S_4 = \frac{0.25}{3}[e^0 + 4e^{0.25} + 2e^{0.5} + 4e^{0.75} + e^1] \approx 1.71828$ (accurate to 7 decimal places).

### 9.3 Gaussian Quadrature

Instead of equally spaced nodes, choose $n$ optimal nodes $\{x_k\}$ and weights $\{w_k\}$ to exactly integrate all polynomials of degree $\leq 2n-1$:

$$\int_{-1}^1 f(x)\,dx \approx \sum_{k=1}^n w_k f(x_k)$$

The nodes are roots of the Legendre polynomial $P_n(x)$. Gauss-Legendre quadrature is the most accurate quadrature rule for smooth functions — $n$ nodes achieve $O(h^{2n})$ error.

### 9.4 Monte Carlo Integration

**Idea.** For $\int_a^b f(x)\,dx$: draw $n$ i.i.d. samples $X_1,\ldots,X_n \sim \text{Uniform}(a,b)$ and estimate:

$$\hat{I}_n = \frac{b-a}{n}\sum_{k=1}^n f(X_k)$$

By the Law of Large Numbers: $\hat{I}_n \xrightarrow{a.s.} \int_a^b f(x)\,dx$.

**Error.** By the CLT:

$$\sqrt{n}(\hat{I}_n - I) \xrightarrow{d} \mathcal{N}(0, (b-a)^2\text{Var}[f(X)])$$

Standard error: $\text{SE} = \frac{(b-a)\,\text{Std}[f(X)]}{\sqrt{n}} = O(1/\sqrt{n})$.

**Key property.** The $O(1/\sqrt{n})$ convergence rate is dimension-independent. Trapezoid and Simpson's rules suffer from the **curse of dimensionality** ($O(n^{-2/d})$ for $d$-dimensional integrals), but Monte Carlo's rate stays $O(1/\sqrt{n})$ regardless of dimension. This is why high-dimensional integration in ML (expectation over data distributions, latent variables, trajectories) always uses Monte Carlo.

**Variance reduction.** Importance sampling: draw $X_k \sim q(x)$ instead of uniform, estimate:

$$\hat{I} = \frac{1}{n}\sum_{k=1}^n \frac{f(X_k)}{q(X_k)/((b-a)^{-1})} = \frac{1}{n}\sum_{k=1}^n \frac{f(X_k)(b-a)q_{\text{uniform}}}{q(X_k)}$$

Choosing $q \propto |f|$ minimizes variance — the basis of importance-weighted autoencoders (IWAE).

### 9.5 For AI: SGD as Monte Carlo Expectation

The true gradient update is:

$$\nabla_\theta \mathcal{L}(\theta) = \int \nabla_\theta \ell(f_\theta(\mathbf{x}), y)\,p(\mathbf{x},y)\,d\mathbf{x}\,dy$$

SGD with mini-batch $\mathcal{B}$ of size $B$ estimates this as:

$$\widehat{\nabla}_\theta\mathcal{L}(\theta) = \frac{1}{B}\sum_{i\in\mathcal{B}} \nabla_\theta\ell(f_\theta(\mathbf{x}_i), y_i)$$

This is a Monte Carlo estimate of the gradient integral. The mini-batch is drawn i.i.d. from the data distribution (uniformly at random from the training set), so it is an unbiased estimator with variance $O(1/B)$. Larger batch sizes reduce variance but increase compute per step.


---

## 10. Integration in Probability

> **Forward reference to §06-Probability Theory.** The full treatment of random variables, distributions, expectation, and probabilistic reasoning is in [§06-Probability-Theory](../../06-Probability-Theory/README.md). Here we cover the integration mechanics — how to compute expectations, variances, KL divergence, and entropy as definite or improper integrals.

### 10.1 Probability Density Functions

A **probability density function** (PDF) $p: \mathbb{R} \to [0,\infty)$ satisfies:

$$\int_{-\infty}^\infty p(x)\,dx = 1, \qquad p(x) \geq 0 \text{ for all } x$$

The probability that $X$ falls in $[a,b]$ is $\Pr(a \leq X \leq b) = \int_a^b p(x)\,dx$.

**Common PDFs and their normalization integrals:**

| Distribution | PDF | Normalization relies on |
|-------------|-----|------------------------|
| Uniform on $[a,b]$ | $\frac{1}{b-a}$ | Elementary |
| Gaussian $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}$ | Gaussian integral (§8.4) |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$, $x\geq 0$ | $\int_0^\infty \lambda e^{-\lambda x}\,dx = 1$ |
| Laplace($\mu, b$) | $\frac{1}{2b}e^{-|x-\mu|/b}$ | Symmetric exponential |

### 10.2 CDF and FTC

The **cumulative distribution function** (CDF) is:

$$F(x) = \Pr(X \leq x) = \int_{-\infty}^x p(t)\,dt$$

By FTC Part 1: $F'(x) = p(x)$ — the PDF is the derivative of the CDF. This connects probability theory directly to the FTC: the CDF is the "area accumulation function" of the PDF, and differentiating it recovers the density.

**Properties of the CDF:**
- $F(-\infty) = 0$, $F(+\infty) = 1$
- $F$ is non-decreasing: $x_1 < x_2 \Rightarrow F(x_1) \leq F(x_2)$
- $F$ is right-continuous
- $\Pr(a < X \leq b) = F(b) - F(a)$ (FTC Part 2)

### 10.3 Expectation as a Weighted Integral

$$\mathbb{E}[X] = \int_{-\infty}^\infty x\,p(x)\,dx$$

$$\mathbb{E}[g(X)] = \int_{-\infty}^\infty g(x)\,p(x)\,dx \quad \text{(Law of the Unconscious Statistician)}$$

**Gaussian expectation.** For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$\mathbb{E}[X] = \int_{-\infty}^\infty x\cdot\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}\,dx = \mu$$

**Proof.** Substitute $u = (x-\mu)/\sigma$: $\mathbb{E}[X] = \int_{-\infty}^\infty (\sigma u + \mu)\frac{1}{\sqrt{2\pi}}e^{-u^2/2}\,du$. The $\sigma u$ term integrates to 0 (odd function); the $\mu$ term gives $\mu \cdot 1$. $\square$

### 10.4 Variance and Second Moments

$$\text{Var}(X) = \mathbb{E}[(X-\mu)^2] = \int_{-\infty}^\infty (x-\mu)^2\,p(x)\,dx = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

**Gaussian variance.** For $X \sim \mathcal{N}(\mu,\sigma^2)$:

$$\text{Var}(X) = \int_{-\infty}^\infty (x-\mu)^2 \frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/(2\sigma^2)}\,dx = \sigma^2$$

Proven via substitution $u=(x-\mu)/\sigma$ and the identity $\int_{-\infty}^\infty u^2 e^{-u^2/2}\,du = \sqrt{2\pi}$ (integration by parts with $u \cdot ue^{-u^2/2}$).

### 10.5 KL Divergence

The **Kullback-Leibler divergence** from $q$ to $p$:

$$\text{KL}(p\|q) = \int_{-\infty}^\infty p(x)\ln\frac{p(x)}{q(x)}\,dx$$

**Properties:**
- $\text{KL}(p\|q) \geq 0$ (Gibbs' inequality — proven via Jensen's inequality and the concavity of $\ln$)
- $\text{KL}(p\|q) = 0$ iff $p = q$ a.e.
- Not symmetric: $\text{KL}(p\|q) \neq \text{KL}(q\|p)$ in general

**Forward vs. reverse KL:**
- $\text{KL}(p\|q)$ (forward/exclusive): forces $q$ to cover all modes of $p$. Used in maximum likelihood estimation.
- $\text{KL}(q\|p)$ (reverse/inclusive): forces $q$ to fit one mode of $p$ well. Used in variational inference (ELBO = $-\text{KL}(q\|p) + \text{const}$).

**Gibbs' inequality proof.** By concavity of $\ln$: $\ln t \leq t - 1$ for all $t > 0$. Apply with $t = q(x)/p(x)$:

$$-\text{KL}(p\|q) = \int p(x)\ln\frac{q(x)}{p(x)}\,dx \leq \int p(x)\left(\frac{q(x)}{p(x)}-1\right)\,dx = \int q(x)\,dx - \int p(x)\,dx = 1-1=0$$

### 10.6 Entropy

The **differential entropy** of a continuous random variable $X$ with PDF $p$:

$$H(X) = -\int_{-\infty}^\infty p(x)\ln p(x)\,dx = -\mathbb{E}[\ln p(X)]$$

**Gaussian entropy.** For $X \sim \mathcal{N}(\mu,\sigma^2)$:

$$H(X) = \frac{1}{2}\ln(2\pi e\sigma^2) = \frac{1}{2}[1 + \ln(2\pi\sigma^2)]$$

**Maximum entropy principle.** Among all distributions with mean $\mu$ and variance $\sigma^2$, the Gaussian maximizes entropy. This makes the Gaussian the natural distribution for uncertainty — used throughout Bayesian deep learning (Gaussian priors, Gaussian posteriors in VAEs).

**Connection to cross-entropy loss.** For a model $q_\theta$ trained on data from $p$:

$$\mathbb{E}_{x\sim p}[-\ln q_\theta(x)] = H(p) + \text{KL}(p\|q_\theta)$$

Minimizing cross-entropy loss minimizes $\text{KL}(p\|q_\theta)$ (since $H(p)$ is constant w.r.t. $\theta$). This is why cross-entropy training is equivalent to maximum likelihood estimation.


---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Forgetting $+C$ in indefinite integrals | Every antiderivative family has a free constant; omitting it loses solutions to IVPs | Always write $+C$ and determine it from initial conditions |
| 2 | $\int f(x)g(x)\,dx = \int f\,dx \cdot \int g\,dx$ | Integration does NOT distribute over products | Use substitution or integration by parts |
| 3 | $\int_a^b f\,dx = F(b) - F(a)$ without checking $F' = f$ | If $F$ is wrong, the evaluation is wrong | Always verify the antiderivative by differentiating |
| 4 | Not changing limits in definite $u$-substitution | If you substitute $u = g(x)$, the limits must become $g(a)$ and $g(b)$ | Either change limits or back-substitute |
| 5 | $\int \frac{1}{x^2}\,dx = \ln(x^2)+C$ | Incorrect — $\int x^{-2}\,dx = -x^{-1}+C$; $\ln$ antiderivative only applies to $1/x$ | Power rule: $\int x^n\,dx = x^{n+1}/(n+1)+C$ for $n\neq -1$ |
| 6 | $\int_0^1 \frac{1}{x}\,dx = [\ln x]_0^1 = 0$ | This is an improper integral — $\ln(0) = -\infty$; the integral diverges | Always check for discontinuities before applying FTC Part 2 |
| 7 | $\int_{-1}^{1}\frac{1}{x}\,dx = 0$ by symmetry | The integrand is odd, but the integral diverges — symmetry argument fails for divergent integrals | Confirm convergence before using symmetry |
| 8 | Wrong LIATE choice causes circular integration | Choosing $u =$ exponential in $\int xe^x\,dx$ → no simplification | Let $u$ be LIATE-first: $u=x$, $dv=e^x\,dx$ |
| 9 | $\int_a^\infty f\,dx$ treated as finite without checking | Infinite integration limits require explicit convergence check | Write as $\lim_{b\to\infty}\int_a^b$ and evaluate the limit |
| 10 | Monte Carlo $O(1/\sqrt{n})$ confused with deterministic $O(h^2)$ | Monte Carlo error is stochastic, in expectation/variance; not a uniform bound | Report $\pm 1.96\,\text{SE}$ confidence intervals for MC estimates |
| 11 | $\text{KL}(p\|q) = \text{KL}(q\|p)$ | KL divergence is not symmetric | Know the difference: forward KL is mode-covering, reverse KL is mode-seeking |
| 12 | $\int u\,dv = uv + \int v\,du$ (sign error in parts) | The formula has a minus sign: $\int u\,dv = uv - \int v\,du$ | Derive from product rule to remember the minus sign |

---

## 12. Exercises

**Exercise 1 ★ — Riemann Sums.** For $f(x) = x^2$ on $[0,2]$ with $n = 8$ equal subintervals:
(a) Compute the left Riemann sum $L_8$.
(b) Compute the right Riemann sum $R_8$.
(c) Compute the exact value $\int_0^2 x^2\,dx$ via FTC and verify $L_8 \leq$ exact $\leq R_8$.

**Exercise 2 ★ — FTC and Antiderivatives.** Evaluate:
(a) $\int_1^e \frac{(\ln x)^2}{x}\,dx$ (b) $\int_0^{\pi/2}\sin^3 x\cos x\,dx$ (c) $\int_0^{\ln 2}e^x\sqrt{1+e^x}\,dx$

**Exercise 3 ★ — Integration by Parts.** Compute:
(a) $\int x^2 e^{-x}\,dx$ (b) $\int \ln(x^2+1)\,dx$ (c) $\int e^x\cos x\,dx$

**Exercise 4 ★★ — Improper Integrals.** Determine convergence and evaluate if finite:
(a) $\int_1^\infty \frac{1}{x^{3/2}}\,dx$ (b) $\int_0^1 \frac{\ln x}{\sqrt{x}}\,dx$ (c) $\int_{-\infty}^\infty xe^{-x^2}\,dx$

**Exercise 5 ★★ — Numerical Integration.** For $f(x) = e^{-x^2}$ on $[0,2]$:
(a) Compute $T_n$ and $S_n$ for $n \in \{4, 8, 16\}$.
(b) Compare to `scipy.integrate.quad` result.
(c) Plot the absolute error vs. $n$ for both methods on a log-log scale. Measure the observed convergence rates.

**Exercise 6 ★★ — Monte Carlo Integration.** Estimate $\int_0^1 \sin(\pi x^2)\,dx$:
(a) Implement basic Monte Carlo with $n \in \{100, 1000, 10000, 100000\}$.
(b) Plot the estimate and $\pm 2\,\text{SE}$ confidence band vs. $n$.
(c) Verify the $O(1/\sqrt{n})$ convergence by plotting $n \cdot \text{Var}[\hat{I}_n]$ vs. $n$.

**Exercise 7 ★★★ — KL Divergence.** Let $p = \mathcal{N}(0,1)$ and $q = \mathcal{N}(\mu, \sigma^2)$:
(a) Derive the closed-form $\text{KL}(p\|q)$ by evaluating the integral.
(b) Implement numerical KL via Monte Carlo with $10^5$ samples. Verify against the closed form.
(c) Plot $\text{KL}(p\|q)$ as a function of $\mu \in [-3,3]$ (fixed $\sigma=1$) and as a function of $\sigma \in [0.1, 3]$ (fixed $\mu=0$). Observe the asymmetry.

**Exercise 8 ★★★ — ELBO and Variational Inference.** The ELBO (Evidence Lower BOund) is:

$$\text{ELBO}(q) = \mathbb{E}_{z\sim q}[\ln p(x,z)] - \mathbb{E}_{z\sim q}[\ln q(z)] = \mathbb{E}_{z\sim q}[\ln p(x|z)] - \text{KL}(q\|p_z)$$

(a) Show that $\ln p(x) = \text{ELBO}(q) + \text{KL}(q\|p(\cdot|x))$ using the definition of conditional probability and KL divergence.
(b) Explain why maximizing the ELBO is equivalent to maximizing a lower bound on $\ln p(x)$.
(c) For $p_z = \mathcal{N}(0,1)$ and $q = \mathcal{N}(\mu,\sigma^2)$, compute $\text{KL}(q\|p_z)$ and implement the reparameterization trick: $z = \mu + \sigma\varepsilon$, $\varepsilon\sim\mathcal{N}(0,1)$.


---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---------|-------------|
| Riemann sums | Conceptual foundation of all numerical expectation estimates; discrete sum over a dataset approximates the continuous integral over the data-generating distribution |
| FTC Part 2 | Evaluating KL divergences, entropies, and moments in closed form; normalizing flow log-likelihoods |
| FTC Part 1 | Adjoint method for neural ODEs; sensitivity analysis of dynamical systems; continuous-time RL |
| u-Substitution | Change-of-variables formula in normalizing flows (Real NVP, Glow, FFJORD); reparameterization trick in VAEs |
| Integration by parts | Log-derivative (REINFORCE) trick for policy gradient methods (PPO, GRPO, DPO in RLHF) |
| Improper integrals | Convergence of expected losses over infinite data; entropy and KL divergence for continuous distributions |
| Gaussian integral | Normalization of all Gaussian distributions; closed-form KL divergences between Gaussians (VAE regularizer) |
| Trapezoid/Simpson | Numerical integration in scientific ML; ODE solver stepping (Euler, RK4 in neural ODEs) |
| Monte Carlo | SGD and mini-batch training; IWAE (importance-weighted); MCMC in Bayesian neural networks |
| PDF/CDF via FTC | Score matching (diffusion model training); CDF inversion sampling; cumulative reward functions |
| Expectation as integral | Every loss function; ELBO; reward expectation in RL; attention weights as expectation over keys |
| KL divergence | VAE regularizer; DPO alignment loss; diffusion model score function; maximum entropy RL |
| Entropy | Information bottleneck principle; attention entropy regularization; exploration in RL |
| Cross-entropy ↔ KL | Cross-entropy training = MLE = minimizing KL from true distribution to model |

---

## 14. Conceptual Bridge

**Looking back.** This section built on two pillars from previous sections:

1. **Limits** ([§01](../01-Limits-and-Continuity/notes.md)) — the Riemann integral is defined as a limit of sums; the FTC proof uses the MVT for integrals; convergence of improper integrals is a limit.
2. **Derivatives** ([§02](../02-Derivatives-and-Differentiation/notes.md)) — antiderivatives are "reverse derivatives"; FTC Part 1 says the area-accumulation function has derivative equal to the integrand; u-substitution reverses the chain rule; integration by parts reverses the product rule.

Every integration technique is the reverse of a differentiation technique. The FTC is the theorem that makes this reversal exact.

**Looking forward.**

- **[§04-Series-and-Sequences](../04-Series-and-Sequences/notes.md)** — Taylor series are derived using higher-order derivatives; the Taylor remainder formula involves a definite integral of the $(n+1)$-th derivative. Power series can be integrated term-by-term.
- **[§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md)** — double and triple integrals extend the Riemann definition to higher dimensions; Fubini's theorem allows iterated integration; the substitution rule becomes the Jacobian change-of-variables formula.
- **[§06-Probability Theory](../../06-Probability-Theory/README.md)** — all continuous probability is integration: distributions, expectations, variances, moment-generating functions, characteristic functions, conditional distributions.
- **[§08-Optimization](../../08-Optimization/README.md)** — population risk is an integral; SGD is a Monte Carlo gradient estimator; natural gradient uses the Fisher information matrix, which is defined via an integral.

**Position in the curriculum:**

```
CHAPTER 4 — CALCULUS FUNDAMENTALS
════════════════════════════════════════════════════════════════════════

  §01 Limits and Continuity ──────────────────────────────────────────
         │  (limit foundations, ε-δ, continuity)
         ▼
  §02 Derivatives and Differentiation ───────────────────────────────
         │  (chain rule, product rule, activation derivatives)
         ▼
  §03 Integration  ◄─── YOU ARE HERE ─────────────────────────────────
         │  (FTC links §02 ↔ §03; substitution reverses chain rule)
         ▼
  §04 Series and Sequences ───────────────────────────────────────────
         │  (Taylor series uses §02 derivatives + §03 integration)
         ▼
  §05 Multivariate Calculus ──────────────────────────────────────────
         (double integrals, Fubini, Jacobians — extends §03)
         ▼
  §06 Probability Theory ─────────────────────────────────────────────
         (all continuous probability IS integration from §03)

════════════════════════════════════════════════════════════════════════
```

---

[← Back to Calculus Fundamentals](../README.md) | [Next: Series and Sequences →](../04-Series-and-Sequences/notes.md)

---

## Appendix A: Extended Substitution Examples and Patterns

### A.1 Recognising the Pattern

The hardest part of $u$-substitution is identifying the right $u$. The pattern to look for: one function is the derivative of another part of the integrand.

| Integrand pattern | Choose $u$ | Because |
|------------------|-----------|---------|
| $f(x^n)\cdot x^{n-1}$ | $u = x^n$ | $du = nx^{n-1}\,dx$ |
| $f(e^x)\cdot e^x$ | $u = e^x$ | $du = e^x\,dx$ |
| $f(\ln x)\cdot \frac{1}{x}$ | $u = \ln x$ | $du = \frac{1}{x}\,dx$ |
| $f(\sin x)\cdot \cos x$ | $u = \sin x$ | $du = \cos x\,dx$ |
| $f(\cos x)\cdot \sin x$ | $u = \cos x$ | $du = -\sin x\,dx$ |
| $f(\sqrt{x})\cdot \frac{1}{\sqrt{x}}$ | $u = \sqrt{x}$ | $du = \frac{1}{2\sqrt{x}}\,dx$ |

### A.2 Rationalizing Substitutions

For integrals involving $\sqrt{ax+b}$: let $u = \sqrt{ax+b}$ so $u^2 = ax+b$ and $2u\,du = a\,dx$.

**Example.** $\int x\sqrt{2x+1}\,dx$. Let $u = \sqrt{2x+1}$, $u^2 = 2x+1$, $x = (u^2-1)/2$, $dx = u\,du$:

$$\int \frac{u^2-1}{2}\cdot u\cdot u\,du = \frac{1}{2}\int(u^4-u^2)\,du = \frac{u^5}{10} - \frac{u^3}{6} + C = \frac{(2x+1)^{5/2}}{10} - \frac{(2x+1)^{3/2}}{6} + C$$

### A.3 Trigonometric Substitutions

For integrands involving $\sqrt{a^2-x^2}$, $\sqrt{a^2+x^2}$, or $\sqrt{x^2-a^2}$:

| Form | Substitution | Identity used |
|------|-------------|---------------|
| $\sqrt{a^2-x^2}$ | $x = a\sin\theta$ | $1-\sin^2\theta = \cos^2\theta$ |
| $\sqrt{a^2+x^2}$ | $x = a\tan\theta$ | $1+\tan^2\theta = \sec^2\theta$ |
| $\sqrt{x^2-a^2}$ | $x = a\sec\theta$ | $\sec^2\theta-1 = \tan^2\theta$ |

**Example.** $\int\frac{1}{\sqrt{4-x^2}}\,dx$. Let $x = 2\sin\theta$, $dx = 2\cos\theta\,d\theta$:

$$\int\frac{2\cos\theta\,d\theta}{\sqrt{4-4\sin^2\theta}} = \int\frac{2\cos\theta}{2\cos\theta}\,d\theta = \theta + C = \arcsin\frac{x}{2} + C$$

**For AI.** Trigonometric substitutions appear when integrating radial functions in high-dimensional probability (e.g., volumes of spherical shells used in $d$-dimensional Gaussian integrals).

### A.4 The Softmax Integral — Partition Function

The softmax denominator (partition function) is a sum $Z = \sum_k e^{z_k}$, the discrete analogue of the integral $Z = \int e^{f(x)}\,dx$ that appears in energy-based models. In the continuous case, computing $Z$ is intractable in general — this is the core computational challenge of energy-based models. Variational autoencoders and diffusion models avoid computing $Z$ directly by working with ratios or lower bounds.


---

## Appendix B: Integration by Parts — Extended Examples and Theory

### B.1 The Cyclic Trick

When integration by parts produces $I = (\text{something}) - cI$ for a constant $c \neq -1$, solve for $I$:

$$I + cI = \text{something} \implies I = \frac{\text{something}}{1+c}$$

This works for $\int e^{ax}\cos(bx)\,dx$ and $\int e^{ax}\sin(bx)\,dx$:

$$\int e^{ax}\cos(bx)\,dx = \frac{e^{ax}(a\cos(bx) + b\sin(bx))}{a^2+b^2} + C$$

$$\int e^{ax}\sin(bx)\,dx = \frac{e^{ax}(a\sin(bx) - b\cos(bx))}{a^2+b^2} + C$$

**Verification:** Differentiate the right side to confirm.

### B.2 Integration by Parts for Definite Integrals

**Example.** Show $\int_0^\infty xe^{-x}\,dx = 1$.

Let $u=x$, $dv=e^{-x}\,dx$:

$$\int_0^\infty xe^{-x}\,dx = \Big[-xe^{-x}\Big]_0^\infty + \int_0^\infty e^{-x}\,dx$$

The boundary term: $\lim_{x\to\infty} xe^{-x} = 0$ (L'Hôpital: $x/e^x \to 0$) and at $x=0$: $0$.

$$= 0 + [-e^{-x}]_0^\infty = 0 - (-1) = 1$$

### B.3 The Gamma Function

The **Gamma function** generalizes factorials to real arguments:

$$\Gamma(s) = \int_0^\infty x^{s-1}e^{-x}\,dx, \quad s > 0$$

**Key properties** (proven by integration by parts):
- $\Gamma(s+1) = s\,\Gamma(s)$ (reduction formula via parts)
- $\Gamma(n) = (n-1)!$ for positive integers $n$
- $\Gamma(1/2) = \sqrt{\pi}$ (from the Gaussian integral)

**Proof of reduction.** $\Gamma(s+1) = \int_0^\infty x^s e^{-x}\,dx$. Let $u = x^s$, $dv = e^{-x}\,dx$:

$$= [-x^s e^{-x}]_0^\infty + s\int_0^\infty x^{s-1}e^{-x}\,dx = 0 + s\,\Gamma(s) \quad \square$$

**For AI.** The Gamma function appears in the normalizing constants of many probability distributions used in Bayesian ML: the Gamma distribution (prior for precision in Gaussian models), the Beta distribution (prior for probabilities in Dirichlet-multinomial models), the Student-t distribution (robust regression).

### B.4 Wallis's Formula

From repeated integration by parts on $\int_0^{\pi/2}\sin^n x\,dx$:

$$\frac{\pi}{2} = \frac{2\cdot 2\cdot 4\cdot 4\cdot 6\cdot 6\cdots}{1\cdot 3\cdot 3\cdot 5\cdot 5\cdot 7\cdots} = \prod_{n=1}^\infty\frac{4n^2}{4n^2-1}$$

This is one of the earliest infinite product formulas for $\pi$ — an unexpected connection between integration and $\pi$.


---

## Appendix C: Improper Integrals — Convergence Tests in Detail

### C.1 The p-Test Summary

$$\int_1^\infty \frac{1}{x^p}\,dx \begin{cases} = \dfrac{1}{p-1} & p > 1 \\ = \infty & p \leq 1 \end{cases} \qquad \int_0^1 \frac{1}{x^p}\,dx \begin{cases} = \dfrac{1}{1-p} & p < 1 \\ = \infty & p \geq 1 \end{cases}$$

The boundary $p = 1$ always diverges ($\int 1/x\,dx = \ln x$, which diverges at both limits).

### C.2 Comparison Test — Worked Examples

**Example 1.** Does $\int_1^\infty \frac{1}{x^2+\sqrt{x}}\,dx$ converge?

For $x \geq 1$: $x^2 + \sqrt{x} \geq x^2$, so $\frac{1}{x^2+\sqrt{x}} \leq \frac{1}{x^2}$.

Since $\int_1^\infty x^{-2}\,dx = 1$ converges, by comparison the original converges. $\square$

**Example 2.** Does $\int_1^\infty \frac{\ln x}{x}\,dx$ converge?

For $x \geq e$: $\ln x \geq 1$, so $\frac{\ln x}{x} \geq \frac{1}{x}$.

Since $\int_e^\infty x^{-1}\,dx$ diverges, by comparison the original diverges. $\square$

### C.3 Absolute vs Conditional Convergence

$\int_a^\infty f(x)\,dx$ is **absolutely convergent** if $\int_a^\infty |f(x)|\,dx < \infty$.

$\int_a^\infty f(x)\,dx$ is **conditionally convergent** if it converges but not absolutely.

**Example.** $\int_0^\infty \frac{\sin x}{x}\,dx = \frac{\pi}{2}$ (Dirichlet integral) — converges conditionally but NOT absolutely ($\int_0^\infty |\sin x|/x\,dx = \infty$).

**For ML.** Absolutely convergent integrals behave nicely: they can be split, reordered, and approximated by truncated versions. The expected loss $\mathbb{E}[\mathcal{L}]$ is absolutely convergent (non-negative integrand) — this is why expectation estimates via Monte Carlo are reliable.

### C.4 Laplace Transform Preview

The **Laplace transform** is an improper integral parametrized by $s$:

$$\mathcal{L}\{f\}(s) = \int_0^\infty f(t)e^{-st}\,dt$$

It converts differential equations to algebraic equations (used in control theory). For neural networks, the Laplace transform of the loss trajectory $\mathcal{L}\{t \mapsto \mathcal{L}(\theta_t)\}$ is related to the training dynamics in Laplace domain — an emerging tool in the theoretical analysis of gradient descent.


---

## Appendix D: Numerical Integration — Error Analysis and Advanced Methods

### D.1 Trapezoid Rule — Derivation from Scratch

On each subinterval $[x_{k-1}, x_k]$, the trapezoid rule approximates $f$ by the linear interpolant:

$$f(x) \approx f(x_{k-1}) + \frac{f(x_k)-f(x_{k-1})}{h}(x-x_{k-1})$$

Integrating from $x_{k-1}$ to $x_k$:

$$\int_{x_{k-1}}^{x_k} f\,dx \approx f(x_{k-1})\cdot h + \frac{f(x_k)-f(x_{k-1})}{h}\cdot\frac{h^2}{2} = \frac{h}{2}[f(x_{k-1})+f(x_k)]$$

Summing $n$ subintervals with telescoping:

$$T_n = \frac{h}{2}[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)]$$

**Error derivation.** By Taylor expansion on each subinterval:

$$\int_{x_{k-1}}^{x_k} f\,dx = \frac{h}{2}[f(x_{k-1})+f(x_k)] - \frac{h^3}{12}f''(\xi_k)$$

Summing: $E_T = -\frac{h^2(b-a)}{12}\bar{f}''$ where $\bar{f}''$ is some average of $f''$ on $[a,b]$ (MVT). So $|E_T| \leq \frac{M_2(b-a)^3}{12n^2}$ where $M_2 = \max|f''|$.

### D.2 Simpson's Rule — Parabolic Approximation

On each pair of subintervals $[x_{2k-2}, x_{2k}]$, use the unique parabola through three points:

$$S_{2k} = \frac{h}{3}[f(x_{2k-2}) + 4f(x_{2k-1}) + f(x_{2k})]$$

The coefficient pattern comes from integrating the Lagrange interpolating polynomial. The error:

$$E_S = -\frac{h^4(b-a)}{180}\bar{f}^{(4)} \implies |E_S| \leq \frac{M_4(b-a)^5}{180n^4}$$

### D.3 Richardson Extrapolation

If $T_n$ has error $E_T = c_2/n^2 + c_4/n^4 + \cdots$, then combining $T_n$ and $T_{2n}$:

$$\frac{4T_{2n} - T_n}{3}$$

eliminates the $O(1/n^2)$ term, giving a method with error $O(1/n^4)$ — equal to Simpson's! This idea, applied recursively, gives **Romberg integration** with error $O(h^{2k})$ for any $k$.

### D.4 Quasi-Monte Carlo

Standard Monte Carlo has error $O(1/\sqrt{n})$ regardless of dimension. **Quasi-Monte Carlo** (QMC) uses **low-discrepancy sequences** (Halton, Sobol) instead of random points:

- Random points clump and leave gaps
- Low-discrepancy sequences fill space more uniformly

QMC achieves $O((\log n)^d / n)$ error for smooth integrands in $d$ dimensions — much better than $O(1/\sqrt{n})$ when $d$ is small. Used in financial derivatives pricing and high-dimensional integration in Bayesian neural networks.

### D.5 Adaptive Integration

`scipy.integrate.quad` uses **Gaussian-Kronrod quadrature** with adaptive refinement: if the error estimate on a subinterval exceeds tolerance, subdivide and integrate each piece separately. This automatically focuses computational effort on regions where $f$ varies rapidly — crucial for integrands with sharp peaks (e.g., probability densities in high-dimensional tails).


---

## Appendix E: Probability Integration — Extended Topics

### E.1 Moment-Generating Functions

The **moment-generating function** (MGF) of $X$ is:

$$M_X(t) = \mathbb{E}[e^{tX}] = \int_{-\infty}^\infty e^{tx}p(x)\,dx$$

provided this integral converges for $t$ in a neighborhood of 0.

**Why it generates moments:** Differentiate $k$ times at $t=0$:

$$M_X^{(k)}(0) = \mathbb{E}[X^k e^{0\cdot X}] = \mathbb{E}[X^k]$$

So $\mathbb{E}[X] = M'(0)$, $\mathbb{E}[X^2] = M''(0)$, $\text{Var}(X) = M''(0) - [M'(0)]^2$.

**Gaussian MGF.** For $X \sim \mathcal{N}(\mu,\sigma^2)$:

$$M_X(t) = e^{\mu t + \sigma^2 t^2/2}$$

Proven by completing the square in the exponent under the integral and using the Gaussian integral.

### E.2 Characteristic Functions and Fourier Transforms

The **characteristic function** $\phi_X(t) = \mathbb{E}[e^{itX}] = \int e^{itx}p(x)\,dx$ is the Fourier transform of the PDF. Unlike the MGF, the characteristic function always exists (since $|e^{itx}| = 1$).

The **inverse Fourier transform** recovers the PDF:

$$p(x) = \frac{1}{2\pi}\int_{-\infty}^\infty e^{-itx}\phi_X(t)\,dt$$

**For AI.** Fourier transforms appear in:
- **Random Fourier features** (Rahimi & Recht, 2007): approximate shift-invariant kernels via $\hat{k}(x-y) = \mathbb{E}[e^{i\omega(x-y)}]$ where $\omega \sim p(\omega)$
- **Frequency domain attention**: Fourier attention mechanisms (FNet) replace self-attention with 2D FFT
- **Spectral normalization**: weight matrix spectral norm computed via power iteration of $\|W\|_2 = $ largest singular value

### E.3 Conditional Expectations as Integrals

The conditional expectation $\mathbb{E}[Y|X=x]$ is defined via the conditional density $p(y|x)$:

$$\mathbb{E}[Y|X=x] = \int y\,p(y|x)\,dy$$

**Law of total expectation:** $\mathbb{E}[Y] = \mathbb{E}_X[\mathbb{E}[Y|X]] = \int \mathbb{E}[Y|X=x]\,p(x)\,dx$

**For AI.** Diffusion model training minimizes $\mathbb{E}_t\mathbb{E}_{x_0}\mathbb{E}_{x_t|x_0}[\|\epsilon_\theta(x_t,t) - \epsilon\|^2]$ — a triple nested expectation. Each $\mathbb{E}$ is an integral; Monte Carlo (sampling) handles them all.

### E.4 Integration and Maximum Likelihood

Maximum likelihood estimation (MLE) maximizes $\prod_{i=1}^n p(x_i;\theta)$ — equivalently, maximizes the log-likelihood $\sum_i \log p(x_i;\theta)$.

This sum is a Monte Carlo estimate of the integral:

$$\frac{1}{n}\sum_{i=1}^n \log p(x_i;\theta) \xrightarrow{n\to\infty} \int \log p(x;\theta)\,p_{\text{true}}(x)\,dx = -H(p_{\text{true}}) - \text{KL}(p_{\text{true}}\|p_\theta)$$

Maximizing MLE is equivalent to minimizing $\text{KL}(p_{\text{true}}\|p_\theta)$ — the forward KL divergence from the true data distribution to the model. This is a fundamental connection between MLE, integration, and information theory.

### E.5 Score Matching

The **score function** of a distribution $p$ is $s(x) = \nabla_x \log p(x)$. Score matching (Hyvärinen, 2005) estimates $p$ without computing its normalization constant $Z = \int e^{f(x)}\,dx$:

$$J(\theta) = \mathbb{E}_{x\sim p}\left[\text{tr}(\nabla_x s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right]$$

Integration by parts shows this equals $\mathbb{E}_{x\sim p}[\|s_\theta(x) - s(x)\|^2]$ plus a constant — so minimizing $J(\theta)$ fits the model score to the true score without integrating $p$. This is the training objective of **denoising diffusion probabilistic models (DDPMs)**.


---

## Appendix F: Key Proofs and Derivations

### F.1 Proof: Linearity of the Definite Integral

**Theorem.** $\int_a^b [f(x)+g(x)]\,dx = \int_a^b f(x)\,dx + \int_a^b g(x)\,dx$.

**Proof.** By definition:

$$\int_a^b [f+g]\,dx = \lim_{n\to\infty}\sum_{k=1}^n [f(x_k^*)+g(x_k^*)]\Delta x_k = \lim_{n\to\infty}\left[\sum_{k=1}^n f(x_k^*)\Delta x_k + \sum_{k=1}^n g(x_k^*)\Delta x_k\right]$$

Since both limits exist separately: $= \int_a^b f\,dx + \int_a^b g\,dx$. $\square$

### F.2 Proof: Substitution Rule for Definite Integrals

**Theorem.** If $u = g(x)$, $g$ differentiable, $f$ continuous:

$$\int_a^b f(g(x))g'(x)\,dx = \int_{g(a)}^{g(b)} f(u)\,du$$

**Proof.** Let $F$ be an antiderivative of $f$. By the chain rule, $\frac{d}{dx}[F(g(x))] = f(g(x))g'(x)$. By FTC Part 2:

$$\int_a^b f(g(x))g'(x)\,dx = [F(g(x))]_a^b = F(g(b)) - F(g(a)) = \int_{g(a)}^{g(b)} f(u)\,du \quad \square$$

### F.3 Proof: Integration by Parts for Definite Integrals

**Theorem.** $\int_a^b u\,v'\,dx = [uv]_a^b - \int_a^b u'v\,dx$.

**Proof.** Product rule: $(uv)' = u'v + uv'$. Integrate: $\int_a^b(uv)'\,dx = \int_a^b u'v\,dx + \int_a^b uv'\,dx$.

FTC: $[uv]_a^b = \int_a^b u'v\,dx + \int_a^b uv'\,dx$. Rearrange. $\square$

### F.4 Proof: $\int_1^\infty x^{-p}\,dx$ Converges iff $p > 1$

For $p \neq 1$:

$$\int_1^b x^{-p}\,dx = \left[\frac{x^{1-p}}{1-p}\right]_1^b = \frac{b^{1-p}-1}{1-p}$$

As $b \to \infty$: $b^{1-p} \to 0$ if $1-p < 0$ (i.e., $p > 1$), giving $\int = 1/(p-1)$.

If $p < 1$: $b^{1-p} \to \infty$, so integral diverges.

For $p = 1$: $\int_1^b x^{-1}\,dx = \ln b \to \infty$. $\square$

### F.5 Proof: Gibbs' Inequality — $\text{KL}(p\|q) \geq 0$

**Theorem.** For probability densities $p$ and $q$: $\int p(x)\ln\frac{p(x)}{q(x)}\,dx \geq 0$.

**Proof.** Since $\ln$ is concave, $\ln t \leq t - 1$ for all $t > 0$ (equality at $t=1$). Apply to $t = q(x)/p(x)$ where $p(x) > 0$:

$$\ln\frac{q(x)}{p(x)} \leq \frac{q(x)}{p(x)} - 1$$

Multiply by $p(x) > 0$ and integrate:

$$\int p(x)\ln\frac{q(x)}{p(x)}\,dx \leq \int [q(x)-p(x)]\,dx = 1 - 1 = 0$$

Therefore $-\text{KL}(p\|q) \leq 0 \Rightarrow \text{KL}(p\|q) \geq 0$. Equality holds iff $q = p$ a.e. $\square$


---

## Appendix G: FTC — Applications in Machine Learning

### G.1 Neural ODEs and the Adjoint Method

A **neural ODE** (Chen et al., 2018) defines the hidden state dynamics as:

$$\frac{d\mathbf{h}(t)}{dt} = f(\mathbf{h}(t), t; \theta)$$

The hidden state at time $T$ is:

$$\mathbf{h}(T) = \mathbf{h}(0) + \int_0^T f(\mathbf{h}(t),t;\theta)\,dt$$

This is FTC Part 2 in reverse: given the "derivative" $f$, integrate to get the accumulated change.

Training requires $\frac{\partial \mathcal{L}}{\partial \theta}$. The **adjoint method** computes this via a backward ODE:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top \frac{\partial f}{\partial \mathbf{h}}$$

where $\mathbf{a}(t) = \partial\mathcal{L}/\partial\mathbf{h}(t)$ is the adjoint state. FTC Part 1 guarantees that integrating this backward ODE recovers the gradient exactly — with $O(1)$ memory (no storing intermediate states).

### G.2 Attention as Expectation

Softmax attention computes:

$$\text{Attn}(q, K, V) = \sum_k \alpha_k v_k, \qquad \alpha_k = \frac{e^{q\cdot k_j/\sqrt{d}}}{\sum_j e^{q\cdot k_j/\sqrt{d}}}$$

This is a **discrete expectation** $\mathbb{E}_{\alpha}[V]$ where $\alpha$ is the softmax distribution over keys. In the continuous limit (as the number of keys grows and positions become dense), this becomes an integral:

$$\text{Attn}(q) = \int v(s)\,\frac{e^{q(s)\cdot k(s)/\sqrt{d}}}{\int e^{q(s)\cdot k(s')/\sqrt{d}}\,ds'}\,ds$$

This connection motivates **kernel attention** approximations (Performer, Random Feature Attention) that use random features to approximate the exponential kernel via Monte Carlo integration of the Gaussian integral: $e^{q\cdot k} = \int e^{q\cdot\omega}\cdot e^{k\cdot\omega}\,p(\omega)\,d\omega$.

### G.3 Diffusion Models — Score Matching via Integration by Parts

The denoising score matching objective (Vincent, 2011; Song & Ermon, 2019):

$$\mathbb{E}_{t,x_0,\epsilon}\left[\lambda(t)\|\epsilon_\theta(x_t,t) - \epsilon\|^2\right]$$

where $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$, $\epsilon \sim \mathcal{N}(0,I)$.

The equivalence to score matching follows from integration by parts in function space. Specifically, $\nabla_{x_t}\log p(x_t) = -\epsilon/\sqrt{1-\bar{\alpha}_t}$, so the network learns the score of the noisy distribution — an integral relationship between the score function and the data density.

### G.4 Variational Autoencoder ELBO

The VAE training objective is the ELBO:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{z\sim q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)\|p(z))$$

Each term is an integral:

- $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)] = \int q_\phi(z|x)\log p_\theta(x|z)\,dz$ — estimated via Monte Carlo (reparameterization trick)
- $\text{KL}(q_\phi\|p) = \int q_\phi(z|x)\log\frac{q_\phi(z|x)}{p(z)}\,dz$ — computed in closed form for Gaussian $q_\phi$ and $p$

The reparameterization trick ($z = \mu_\phi(x) + \sigma_\phi(x)\odot\varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$) is a change of variables (substitution rule) that makes the Monte Carlo estimator differentiable w.r.t. $\phi$.


---

## Appendix H: Notation Reference and Quick-Reference Tables

### H.1 Integration Notation Comparison

| Notation | Meaning | Context |
|----------|---------|---------|
| $\int_a^b f(x)\,dx$ | Definite integral from $a$ to $b$ | Riemann, most contexts |
| $\int f(x)\,dx$ | Indefinite integral (antiderivative) | General antiderivative |
| $[F(x)]_a^b$ | $F(b) - F(a)$ | FTC shorthand |
| $\int f\,d\mu$ | Lebesgue integral w.r.t. measure $\mu$ | Measure theory, probability |
| $\mathbb{E}_{x\sim p}[f(x)]$ | $\int f(x)p(x)\,dx$ | Probabilistic expectation |
| $\hat{I}_n = \frac{1}{n}\sum f(x_i)$ | Monte Carlo estimate | Stochastic approximation |

### H.2 Standard Antiderivatives — Extended Table

| $f(x)$ | $\int f(x)\,dx$ | Notes |
|--------|----------------|-------|
| $x^n$, $n\neq-1$ | $x^{n+1}/(n+1)+C$ | Power rule |
| $1/x$ | $\ln|x|+C$ | $x \neq 0$ |
| $e^x$ | $e^x+C$ | |
| $e^{ax}$ | $e^{ax}/a+C$ | |
| $a^x$ | $a^x/\ln a+C$ | |
| $\sin x$ | $-\cos x+C$ | |
| $\cos x$ | $\sin x+C$ | |
| $\tan x$ | $\ln|\sec x|+C$ | via substitution |
| $\sec x$ | $\ln|\sec x+\tan x|+C$ | |
| $\sec^2 x$ | $\tan x+C$ | |
| $1/\sqrt{1-x^2}$ | $\arcsin x+C$ | |
| $1/(1+x^2)$ | $\arctan x+C$ | |
| $1/(a^2+x^2)$ | $\frac{1}{a}\arctan(x/a)+C$ | |
| $\sinh x$ | $\cosh x+C$ | |
| $\cosh x$ | $\sinh x+C$ | |
| $x\ln x - x$ | $\int\ln x\,dx$ | via parts |

### H.3 Numerical Methods Comparison

| Method | Formula | Error | Evaluations |
|--------|---------|-------|------------|
| Left Riemann | $h\sum_{k=0}^{n-1}f(x_k)$ | $O(h)$ | $n$ |
| Trapezoid | $h[f_0/2 + f_1+\cdots+f_{n-1}+f_n/2]$ | $O(h^2)$ | $n+1$ |
| Simpson's | $h/3[f_0+4f_1+2f_2+\cdots+4f_{n-1}+f_n]$ | $O(h^4)$ | $n+1$ (n even) |
| Gauss-Legendre ($n$ pts) | $\sum w_k f(x_k)$ | $O(h^{2n})$ | $n$ |
| Monte Carlo | $(b-a)\frac{1}{n}\sum f(X_k)$ | $O(1/\sqrt{n})$ stochastic | $n$ |

### H.4 Key Integration Formulas for AI

$$\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi} \qquad \int_{-\infty}^\infty e^{-ax^2}\,dx = \sqrt{\pi/a}$$

$$\int_{-\infty}^\infty xe^{-ax^2}\,dx = 0 \qquad \int_{-\infty}^\infty x^2 e^{-ax^2}\,dx = \frac{\sqrt{\pi}}{2a^{3/2}}$$

$$\text{KL}(\mathcal{N}(\mu_1,\sigma_1^2)\|\mathcal{N}(\mu_2,\sigma_2^2)) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

$$H(\mathcal{N}(\mu,\sigma^2)) = \frac{1}{2}\ln(2\pi e\sigma^2) \qquad H(\text{Uniform}(a,b)) = \ln(b-a)$$

$$\mathbb{E}[X] = \int_0^\infty \Pr(X > t)\,dt \quad \text{(for } X \geq 0\text{)}$$


---

## Appendix I: Worked Solutions — Section 12 Exercises

### I.1 Exercise 1 — Riemann Sums

$f(x) = x^2$ on $[0,2]$, $n=8$, $h = 0.25$, nodes $x_k = kh$:

$$L_8 = h\sum_{k=0}^7 f(kh) = 0.25[0^2+0.25^2+0.5^2+0.75^2+1^2+1.25^2+1.5^2+1.75^2]$$
$$= 0.25[0+0.0625+0.25+0.5625+1+1.5625+2.25+3.0625] = 0.25\times 8.75 = 2.1875$$

$$R_8 = h\sum_{k=1}^8 f(kh) = 0.25[0.25^2+0.5^2+\cdots+2^2] = 0.25\times 11.75 = 2.9375$$

Exact: $\int_0^2 x^2\,dx = [x^3/3]_0^2 = 8/3 \approx 2.6\overline{6}$.

Verify: $L_8 = 2.1875 \leq 8/3 \leq 2.9375 = R_8$. ✓

### I.2 Exercise 2a — $\int_1^e \frac{(\ln x)^2}{x}\,dx$

Let $u = \ln x$, $du = dx/x$. Limits: $u(1) = 0$, $u(e) = 1$.

$$= \int_0^1 u^2\,du = [u^3/3]_0^1 = \frac{1}{3}$$

### I.2b — $\int_0^{\pi/2}\sin^3 x\cos x\,dx$

Let $u = \sin x$, $du = \cos x\,dx$. Limits: 0 to 1.

$$= \int_0^1 u^3\,du = [u^4/4]_0^1 = \frac{1}{4}$$

### I.2c — $\int_0^{\ln 2}e^x\sqrt{1+e^x}\,dx$

Let $u = 1+e^x$, $du = e^x\,dx$. Limits: $u(0) = 2$, $u(\ln 2) = 3$.

$$= \int_2^3 \sqrt{u}\,du = [2u^{3/2}/3]_2^3 = \frac{2}{3}(3\sqrt{3}-2\sqrt{2})$$

### I.3 Exercise 3a — $\int x^2 e^{-x}\,dx$

Tabular method (differentiate $x^2$, integrate $e^{-x}$):

| Sign | $D$ | $I$ |
|------|-----|-----|
| $+$ | $x^2$ | $e^{-x}$ |
| $-$ | $2x$ | $-e^{-x}$ |
| $+$ | $2$ | $e^{-x}$ |
| $-$ | $0$ | $-e^{-x}$ |

$$\int x^2 e^{-x}\,dx = -x^2 e^{-x} - 2xe^{-x} - 2e^{-x} + C = -e^{-x}(x^2+2x+2) + C$$

### I.3b — $\int \ln(x^2+1)\,dx$

Let $u = \ln(x^2+1)$, $dv = dx$:

$$= x\ln(x^2+1) - \int x\cdot\frac{2x}{x^2+1}\,dx = x\ln(x^2+1) - 2\int\frac{x^2}{x^2+1}\,dx$$

Since $\frac{x^2}{x^2+1} = 1 - \frac{1}{x^2+1}$:

$$= x\ln(x^2+1) - 2x + 2\arctan x + C$$

### I.4 Exercise 4a — $\int_1^\infty x^{-3/2}\,dx$

$$= \lim_{b\to\infty}[-2x^{-1/2}]_1^b = \lim_{b\to\infty}\left(-\frac{2}{\sqrt{b}}+2\right) = 2$$

### I.4c — $\int_{-\infty}^\infty xe^{-x^2}\,dx$

The integrand $f(x) = xe^{-x^2}$ is an odd function ($f(-x) = -f(x)$). Since $\int_0^\infty xe^{-x^2}\,dx = [-e^{-x^2}/2]_0^\infty = 1/2 < \infty$, the integral converges absolutely and:

$$\int_{-\infty}^\infty xe^{-x^2}\,dx = 0 \quad \text{(by symmetry)}$$


---

## Appendix J: Glossary

| Term | Definition |
|------|-----------|
| **Antiderivative** | $F$ such that $F'(x) = f(x)$; the indefinite integral $F(x) + C$ |
| **Riemann sum** | $\sum_{k=1}^n f(x_k^*)\Delta x_k$ — finite approximation to the integral |
| **Definite integral** | $\int_a^b f\,dx = \lim_{\|\mathcal{P}\|\to 0}S(\mathcal{P},f)$ |
| **Indefinite integral** | $\int f\,dx = F(x) + C$ — the family of all antiderivatives |
| **FTC Part 1** | $\frac{d}{dx}\int_a^x f(t)\,dt = f(x)$ |
| **FTC Part 2** | $\int_a^b f(x)\,dx = F(b) - F(a)$ for antiderivative $F$ |
| **Improper integral** | Integral with infinite limits or unbounded integrand; defined via limits |
| **Convergent integral** | Improper integral whose limit exists and is finite |
| **u-Substitution** | $\int f(g(x))g'(x)\,dx = \int f(u)\,du$ — reversal of chain rule |
| **Integration by parts** | $\int u\,dv = uv - \int v\,du$ — reversal of product rule |
| **Partial fractions** | Decompose $P/Q$ into simpler rational terms before integrating |
| **PDF** | $p(x) \geq 0$ with $\int p\,dx = 1$ — probability density function |
| **CDF** | $F(x) = \int_{-\infty}^x p(t)\,dt$ — cumulative distribution function |
| **Expectation** | $\mathbb{E}[X] = \int x\,p(x)\,dx$ — weighted average |
| **KL divergence** | $\text{KL}(p\|q) = \int p\ln(p/q)\,dx \geq 0$ |
| **Entropy** | $H(p) = -\int p\ln p\,dx$ — information content |
| **Trapezoid rule** | Numerical integration with $O(h^2)$ error |
| **Simpson's rule** | Numerical integration with $O(h^4)$ error (parabolic approximation) |
| **Monte Carlo** | Stochastic integration via random sampling; $O(1/\sqrt{n})$ error |
| **ELBO** | Evidence lower bound: $\mathcal{L}_{\text{ELBO}} = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q\|p_z)$ |
| **Score function** | $\nabla_x\log p(x)$ — gradient of log-density; used in diffusion models |
| **Reparameterization trick** | $z = \mu + \sigma\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$ — makes MC estimator differentiable |

---

## Appendix K: Connections to Adjacent Sections

### K.1 What §04-Series-and-Sequences Needs from This Section

- **Integration term-by-term**: $\int\sum_{n=0}^\infty a_n x^n\,dx = \sum_{n=0}^\infty \frac{a_n x^{n+1}}{n+1}$ — requires uniform convergence
- **Taylor remainder as integral**: $R_n(x) = \frac{1}{n!}\int_a^x (x-t)^n f^{(n+1)}(t)\,dt$
- **Integral test for series**: $\sum_{n=1}^\infty f(n)$ converges iff $\int_1^\infty f(x)\,dx$ converges (for decreasing $f \geq 0$)

### K.2 What §05-Multivariate Calculus Needs from This Section

- **Fubini's theorem**: $\int\int f(x,y)\,dx\,dy = \int\left(\int f(x,y)\,dx\right)\,dy$ — iterated integration reduces a 2D integral to two 1D integrals
- **Change of variables**: $\int_R f(\mathbf{x})\,d\mathbf{x} = \int_S f(\mathbf{g}(\mathbf{u}))|\det J_\mathbf{g}(\mathbf{u})|\,d\mathbf{u}$ — the Jacobian generalizes $|g'(x)|$ from substitution
- **Line integrals**: $\int_C f\,ds$ along a curve — generalization of $\int_a^b f(x)\,dx$

### K.3 What §06-Probability Theory Needs from This Section

All of continuous probability theory is integration. The sections needs:
- PDF normalization: $\int p\,dx = 1$
- CDF definition and FTC connection
- Expectation: $\mathbb{E}[g(X)] = \int g(x)p(x)\,dx$
- Moment computations: Gaussian moments via Gaussian integral and substitution
- KL divergence and entropy as improper integrals over $\mathbb{R}$


---

## Appendix L: Additional Worked Examples — FTC and Techniques

### L.1 Leibniz Rule Examples

**Example 1.** $\frac{d}{dx}\int_x^{x^2} \sin(t^2)\,dt$.

Upper limit $h(x) = x^2$, lower limit $g(x) = x$. Leibniz rule:

$$= \sin((x^2)^2)\cdot 2x - \sin(x^2)\cdot 1 = 2x\sin(x^4) - \sin(x^2)$$

**Example 2.** $F(x) = \int_0^x \frac{t^2-1}{t^4+1}\,dt$. Find $F'(x)$.

By FTC Part 1: $F'(x) = \frac{x^2-1}{x^4+1}$.

Critical points of $F$: $F'(x) = 0 \Rightarrow x^2 = 1 \Rightarrow x = \pm 1$.

$F'$ changes sign: $+$ for $|x|<1$, $-$ for $|x|>1$. So $F$ has a local max at $x=1$ and local min at $x=-1$.

**Example 3 (FTC + u-sub).** $\frac{d}{dx}\int_1^{e^x}\ln t\,dt$.

Let $u = e^x$: $\frac{d}{dx}\int_1^{e^x}\ln t\,dt = \ln(e^x)\cdot e^x = x\cdot e^x$.

### L.2 Trigonometric Integrals

**Strategy for $\int\sin^m x\cos^n x\,dx$:**

- If $m$ odd: save one $\sin x$, convert rest to $\cos x$ via $\sin^2 = 1-\cos^2$, substitute $u = \cos x$.
- If $n$ odd: save one $\cos x$, convert, substitute $u = \sin x$.
- If both even: use double-angle formulas $\sin^2 x = (1-\cos 2x)/2$, $\cos^2 x = (1+\cos 2x)/2$.

**Example.** $\int\sin^3 x\cos^4 x\,dx$. Odd power of $\sin$:

$\int\sin^2 x\cos^4 x\cdot\sin x\,dx = \int(1-\cos^2 x)\cos^4 x\cdot\sin x\,dx$

Let $u = \cos x$: $= -\int(1-u^2)u^4\,du = -\int(u^4-u^6)\,du = -\frac{u^5}{5}+\frac{u^7}{7}+C = -\frac{\cos^5 x}{5}+\frac{\cos^7 x}{7}+C$.

### L.3 Integrals of Rational Functions — Full Pipeline

**Example.** $\int\frac{x^3-4x+1}{x^2-x-2}\,dx$.

**Step 1.** Long division (numerator degree > denominator degree):

$x^3-4x+1 = (x^2-x-2)\cdot(x+1) + (-x+3)$

**Step 2.** Partial fractions of remainder:

$\frac{-x+3}{x^2-x-2} = \frac{-x+3}{(x-2)(x+1)} = \frac{A}{x-2}+\frac{B}{x+1}$

At $x=2$: $1/(3) = A/3 \Rightarrow A = 1/3$. Wait — $-2+3=1$ and at $x=2$: $A = 1/3$.
At $x=-1$: $1+3 = B(-3) \Rightarrow B = -4/3$.

**Step 3.** Integrate:

$$\int\frac{x^3-4x+1}{x^2-x-2}\,dx = \int\left(x+1+\frac{1/3}{x-2}-\frac{4/3}{x+1}\right)\,dx$$
$$= \frac{x^2}{2}+x+\frac{1}{3}\ln|x-2|-\frac{4}{3}\ln|x+1|+C$$

### L.4 The Dirichlet Integral

$$\int_0^\infty \frac{\sin x}{x}\,dx = \frac{\pi}{2}$$

This is a conditionally convergent improper integral (not absolutely convergent). One proof uses the Laplace transform: define $F(s) = \int_0^\infty e^{-sx}\frac{\sin x}{x}\,dx$ and differentiate w.r.t. $s$:

$$F'(s) = -\int_0^\infty e^{-sx}\sin x\,dx = -\frac{1}{1+s^2}$$

Integrating: $F(s) = -\arctan s + C$. As $s\to\infty$: $F(s)\to 0$, so $C = \pi/2$. At $s=0$: $F(0) = \pi/2$.


---

## Appendix M: Monte Carlo Methods — Extended Analysis

### M.1 Variance of the Monte Carlo Estimator

Let $X_1,\ldots,X_n \overset{i.i.d.}{\sim} \text{Uniform}(a,b)$. The estimator $\hat{I}_n = \frac{b-a}{n}\sum_{k=1}^n f(X_k)$.

$$\mathbb{E}[\hat{I}_n] = (b-a)\mathbb{E}[f(X)] = (b-a)\cdot\frac{1}{b-a}\int_a^b f(x)\,dx = \int_a^b f(x)\,dx \quad \text{(unbiased)}$$

$$\text{Var}[\hat{I}_n] = \frac{(b-a)^2}{n}\text{Var}[f(X)] = \frac{(b-a)^2}{n}\left[\frac{1}{b-a}\int_a^b f(x)^2\,dx - \left(\frac{\int_a^b f(x)\,dx}{b-a}\right)^2\right]$$

Standard error: $\text{SE}(\hat{I}_n) = \sqrt{\text{Var}[\hat{I}_n]} = O(1/\sqrt{n})$.

### M.2 Importance Sampling

To estimate $I = \int f(x)\,dx$, sample $X_k \sim q(x)$ (a proposal distribution) and use:

$$\hat{I}^{\text{IS}}_n = \frac{1}{n}\sum_{k=1}^n \frac{f(X_k)}{q(X_k)}$$

Since $\int f(x)\,dx = \int \frac{f(x)}{q(x)}\cdot q(x)\,dx = \mathbb{E}_q\left[\frac{f(X)}{q(X)}\right]$, this is also unbiased.

**Optimal $q$.** $\text{Var}[\hat{I}^{\text{IS}}]$ is minimized when $q(x) \propto |f(x)|$. With this choice, $\text{Var} = 0$ if $f \geq 0$ everywhere (one-sample exact!).

**In practice.** Choose $q$ to concentrate samples where $|f(x)|$ is large — focusing effort on the important region. Used in:
- **IWAE** (Importance Weighted Autoencoders) — tighter ELBO via importance sampling
- **Particle filters** — sequential importance sampling for state estimation
- **MCMC** — Metropolis-Hastings acceptance-rejection is importance sampling on steroids

### M.3 Central Limit Theorem for Monte Carlo

$$\frac{\hat{I}_n - I}{\text{SE}(\hat{I}_n)} \xrightarrow{d} \mathcal{N}(0,1)$$

This gives a 95% confidence interval: $\hat{I}_n \pm 1.96\cdot\text{SE}(\hat{I}_n)$.

The $\text{SE}$ is estimated from the sample standard deviation:

$$\widehat{\text{SE}} = \frac{(b-a)\hat{\sigma}_f}{\sqrt{n}}, \qquad \hat{\sigma}_f^2 = \frac{1}{n-1}\sum_{k=1}^n\left(f(X_k) - \hat{\mu}_f\right)^2$$

### M.4 Quasi-Monte Carlo — Discrepancy

The error of quasi-Monte Carlo is bounded by the **Koksma-Hlawka inequality**:

$$|\hat{I}_n - I| \leq D_n^*(\mathbf{x})\cdot V(f)$$

where $D_n^*$ is the **star discrepancy** of the point set and $V(f)$ is the total variation of $f$. Low-discrepancy sequences (Halton, Sobol) achieve $D_n^* = O((\log n)^d/n)$, giving error $O((\log n)^d/n)$ vs. $O(1/\sqrt{n})$ for random.

For $d=1$: QMC is always better than Monte Carlo (for smooth integrands). For large $d$, the $(\log n)^d$ factor can dominate.


---

## Appendix N: The Fundamental Theorem — Historical and Conceptual Depth

### N.1 Why the FTC Is Deep

Before the FTC, two problems seemed completely unrelated:
1. **The tangent problem**: find the slope of a curve at a point → derivative
2. **The area problem**: find the area under a curve → integral

Newton and Leibniz discovered they are inverse operations. This is not obvious. There is no reason, a priori, to expect that summing infinitesimally thin rectangles (integration) should be related to measuring instantaneous slope (differentiation).

The FTC says: **accumulation is the reverse of rate**. If you know how fast something is accumulating at every instant, you can find the total accumulation — just by finding an antiderivative. This is one of the most non-obvious and deep theorems in all of mathematics.

### N.2 What Makes the FTC Work

The key ingredients:
1. **Continuity of $f$**: ensures $f$ can be approximated uniformly by step functions (Riemann integrability)
2. **MVT for integrals**: $\frac{1}{h}\int_x^{x+h}f(t)\,dt = f(c_h)$ for some $c_h \in (x, x+h)$
3. **Continuity of $f$ at $x$**: ensures $f(c_h) \to f(x)$ as $h \to 0$

If $f$ has discontinuities, Part 1 may fail: $G'(x) = f(x)$ only holds at points of continuity of $f$. The Lebesgue integral extends the FTC to a much broader class of functions.

### N.3 The FTC in Multiple Dimensions

The FTC generalizes to higher dimensions in several forms:

| 1D FTC | Higher-dimensional version |
|--------|--------------------------|
| $\int_a^b f'(x)\,dx = f(b)-f(a)$ | **Green's theorem**: $\oint_C \mathbf{F}\cdot d\mathbf{r} = \iint_D \text{curl}\,\mathbf{F}\,dA$ |
| | **Stokes' theorem**: $\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r} = \iint_S (\nabla\times\mathbf{F})\cdot d\mathbf{S}$ |
| | **Divergence theorem**: $\oiiint_{\partial V}\mathbf{F}\cdot d\mathbf{S} = \iiint_V \nabla\cdot\mathbf{F}\,dV$ |

All are special cases of **Stokes' theorem on manifolds**: $\int_M d\omega = \int_{\partial M}\omega$.

For ML, the divergence theorem underlies the **divergence of a vector field** used in:
- Flow matching (training vector fields for generative models)
- Continuous normalizing flows with trace of the Jacobian
- Optimal transport with the continuity equation

### N.4 Antiderivatives That Cannot Be Expressed in Closed Form

Some elementary functions have no elementary antiderivative. Famous examples:

| Integrand | "Antiderivative" | Notes |
|----------|----------------|-------|
| $e^{-x^2}$ | $\frac{\sqrt{\pi}}{2}\text{erf}(x)$ | Error function — not elementary |
| $\frac{\sin x}{x}$ | $\text{Si}(x)$ (sine integral) | Not elementary |
| $\frac{e^x}{x}$ | $\text{Ei}(x)$ (exponential integral) | Not elementary |
| $\frac{1}{\ln x}$ | $\text{Li}(x)$ (logarithmic integral) | Counts primes! Not elementary |
| $\sqrt{1-k^2\sin^2 x}$ | Elliptic integral | Not elementary |

These "special functions" appear constantly in ML:
- `scipy.special.erf`, `scipy.special.erfinv` — in GELU, quantile functions, CDFs
- `scipy.special.gammaln` — in Dirichlet, Beta, Gamma distributions
- `scipy.special.bessel` — in von Mises distributions (circular statistics in positional encoding)


---

## Appendix O: Lebesgue vs. Riemann Integration

### O.1 Why a Different Theory?

The Riemann integral is sufficient for continuous functions and most smooth ML applications. But the **Lebesgue integral** is the foundation of modern probability theory and is essential for rigorous statements about expected values, convergence theorems, and measure theory.

**The key difference**: Riemann integration partitions the *domain* (x-axis); Lebesgue integration partitions the *range* (y-axis).

```
RIEMANN vs. LEBESGUE INTEGRATION
═══════════════════════════════════════════════════════════════════════════

  Riemann: slice domain into [x_i, x_{i+1}], sum f(x_i)*Δx
  ┌─────────────────────────────────────────────┐
  │   f(x)                                      │
  │    │█████                                   │
  │    │██████████                              │
  │    │███████████████                         │
  │    └──────────────────── x                  │
  │      "partition the x-axis"                 │
  └─────────────────────────────────────────────┘

  Lebesgue: slice range into [y_i, y_{i+1}], multiply by measure
            of preimage {x : f(x) ∈ [y_i, y_{i+1}]}
  ┌─────────────────────────────────────────────┐
  │   y                                         │
  │    │       ← how much x gives f(x) ≈ 4?    │
  │  4 ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌         │
  │  3 ├╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌      │
  │    └──────────────────── x                  │
  │      "partition the y-axis"                 │
  └─────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

### O.2 Key Theorem (Lebesgue vs. Riemann)

**Theorem (Lebesgue, 1901)**: A bounded function on $[a, b]$ is Riemann integrable if and only if it is continuous **almost everywhere** (i.e., the set of discontinuities has Lebesgue measure zero).

For ML, this means:
- ReLU is Riemann integrable (discontinuous only at 0, a set of measure zero)
- Indicator functions $\mathbf{1}[x > 0]$ are Riemann integrable for the same reason
- Pathological functions like the Dirichlet function ($\mathbf{1}_{\mathbb{Q}}$) are NOT Riemann integrable but ARE Lebesgue integrable

### O.3 Convergence Theorems (The Real Advantage)

The Lebesgue integral enables three fundamental convergence theorems that Riemann lacks:

**Monotone Convergence Theorem (MCT)**: If $0 \le f_1 \le f_2 \le \cdots$ and $f_n \to f$ pointwise, then:
$$\lim_{n\to\infty} \int f_n \, d\mu = \int \lim_{n\to\infty} f_n \, d\mu = \int f \, d\mu$$

**Dominated Convergence Theorem (DCT)**: If $f_n \to f$ pointwise and $|f_n| \le g$ for an integrable $g$, then:
$$\lim_{n\to\infty} \int f_n \, d\mu = \int f \, d\mu$$

**Fatou's Lemma**: $\int \liminf_{n\to\infty} f_n \, d\mu \le \liminf_{n\to\infty} \int f_n \, d\mu$

**Why ML cares**: The DCT justifies **differentiating under the integral sign** — the key step in computing gradients of expected values:
$$\frac{\partial}{\partial\theta} \mathbb{E}_{p_\theta}[f(x)] = \frac{\partial}{\partial\theta} \int f(x)\,p_\theta(x)\,dx = \int f(x)\,\frac{\partial p_\theta}{\partial\theta}\,dx$$

This step is legal when $f \cdot |\partial p_\theta/\partial\theta|$ is dominated by an integrable function — which is why REINFORCE and the reparameterization trick both have regularity conditions.

### O.4 Probability as Measure Theory

Modern probability uses the Lebesgue framework directly:

| Measure theory | Probability |
|---------------|-------------|
| Measure space $(\Omega, \mathcal{F}, \mu)$ | Probability space $(\Omega, \mathcal{F}, P)$ |
| Measurable function $f: \Omega \to \mathbb{R}$ | Random variable $X: \Omega \to \mathbb{R}$ |
| $\int f \, d\mu$ | $\mathbb{E}[X]$ |
| $\mu(A) = \int_A 1 \, d\mu$ | $P(A)$ |
| Radon-Nikodym derivative $dP/dQ$ | Likelihood ratio |

The **Radon-Nikodym theorem** — which says that if $P \ll Q$ (P is absolutely continuous w.r.t. Q), there exists a measurable function $\frac{dP}{dQ}$ such that $P(A) = \int_A \frac{dP}{dQ} \, dQ$ — is the rigorous foundation for:
- KL divergence: $D_{KL}(P \| Q) = \int \log\frac{dP}{dQ} \, dP$
- Change of variables in normalizing flows
- Importance sampling weights

For day-to-day ML, Riemann integration suffices. But when reading probability theory papers or understanding why certain gradient estimators require regularity conditions, the Lebesgue framework is the right language.


---

## Appendix P: Quick Reference Card

### P.1 Standard Antiderivatives

| $f(x)$ | $\int f(x)\,dx$ | Notes |
|--------|----------------|-------|
| $x^n$ | $\frac{x^{n+1}}{n+1} + C$ | $n \ne -1$ |
| $x^{-1}$ | $\ln|x| + C$ | |
| $e^x$ | $e^x + C$ | |
| $a^x$ | $\frac{a^x}{\ln a} + C$ | $a > 0, a \ne 1$ |
| $\ln x$ | $x\ln x - x + C$ | by parts |
| $\sin x$ | $-\cos x + C$ | |
| $\cos x$ | $\sin x + C$ | |
| $\tan x$ | $-\ln|\cos x| + C$ | |
| $\sec^2 x$ | $\tan x + C$ | |
| $\frac{1}{1+x^2}$ | $\arctan x + C$ | |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ | |
| $\sinh x$ | $\cosh x + C$ | |
| $\cosh x$ | $\sinh x + C$ | |
| $\frac{1}{\sqrt{2\pi}}\,e^{-x^2/2}$ | $\Phi(x) + C$ | CDF of $N(0,1)$ |

### P.2 Key Definite Integrals

$$\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi} \qquad \int_0^\infty x^{n-1}e^{-x}\,dx = \Gamma(n) \qquad \int_0^1 x^{a-1}(1-x)^{b-1}\,dx = B(a,b)$$

$$\int_0^\infty \frac{\sin x}{x}\,dx = \frac{\pi}{2} \qquad \int_{-\pi}^{\pi}\sin(mx)\cos(nx)\,dx = 0 \quad \text{(orthogonality)}$$

### P.3 Integration Strategies Flowchart

```
WHICH TECHNIQUE?
════════════════════════════════════════════════════════════════════════

  Is there a composite function f(g(x))?
    YES → try u-substitution: u = g(x)

  Is the integrand a product of two "different" types?
    YES → try integration by parts (LIATE order)

  Is the integrand a rational function P(x)/Q(x)?
    YES → try partial fractions (after polynomial division if deg P ≥ deg Q)

  Does the integrand contain √(a²−x²), √(a²+x²), or √(x²−a²)?
    YES → try trig substitution (x = a sinθ, a tanθ, a secθ)

  Does the integrand contain e^x times polynomial or trig?
    YES → try tabular integration by parts

  Nothing works? → check integral tables / computer algebra system

════════════════════════════════════════════════════════════════════════
```

### P.4 Numerical Integration Comparison

| Method | Error | Formula | When to use |
|--------|-------|---------|-------------|
| Midpoint | $O(h^2)$ | $h\sum f(x_i + h/2)$ | Smooth functions, simple implementation |
| Trapezoid | $O(h^2)$ | $h[\frac{f(a)+f(b)}{2} + \sum f(x_i)]$ | Periodic functions (exponentially fast) |
| Simpson's | $O(h^4)$ | $\frac{h}{3}[f_0 + 4f_1 + 2f_2 + 4f_3 + \cdots + f_n]$ | Smooth functions, better accuracy |
| Gauss-Legendre | $O(h^{2n})$ | Optimal nodes & weights | High accuracy, smooth integrands |
| Monte Carlo | $O(n^{-1/2})$ | $\frac{1}{n}\sum f(x_i)$, $x_i \sim U[a,b]$ | High dimensions, discontinuous $f$ |
| Quasi-MC | $O((\log n)^d / n)$ | Low-discrepancy sequences | High dimensions with structure |

### P.5 Key ML Formulas Involving Integration

$$\mathbb{E}_{x\sim p}[f(x)] = \int f(x)\,p(x)\,dx \approx \frac{1}{n}\sum_{i=1}^n f(x_i) \quad \text{(Monte Carlo ≡ mini-batch)}$$

$$D_{KL}(p\|q) = \int p(x)\ln\frac{p(x)}{q(x)}\,dx \ge 0 \quad \text{(Gibbs inequality)}$$

$$\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\ln p_\theta(x|z)] - D_{KL}(q_\phi(z|x)\|p(z)) \quad \text{(ELBO)}$$

$$\nabla_\theta \mathbb{E}_{p_\theta}[f(x)] = \mathbb{E}_{p_\theta}[f(x)\,\nabla_\theta\ln p_\theta(x)] \quad \text{(REINFORCE / score gradient)}$$

$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t) \implies \mathbf{z}(T) = \mathbf{z}(0) + \int_0^T f_\theta(\mathbf{z}(t),t)\,dt \quad \text{(Neural ODE)}$$


---

## Appendix Q: Exercises — Worked Solutions (Continued)

### Q.1 Exercise 5 — Numerical Integration Full Walkthrough

**Problem**: Compare trapezoid vs. Simpson's for $\int_0^1 e^{-x^2}\,dx$.

**True value**: Using the error function, $\int_0^1 e^{-x^2}\,dx = \frac{\sqrt\pi}{2}\,\text{erf}(1) \approx 0.746824$.

**Trapezoid with $n=4$ ($h = 0.25$)**:

$x$-values: $0, 0.25, 0.5, 0.75, 1.0$

$f$-values: $1,\ e^{-0.0625} \approx 0.93941,\ e^{-0.25} \approx 0.77880,\ e^{-0.5625} \approx 0.56978,\ e^{-1} \approx 0.36788$

$$T_4 = 0.25\left[\frac{1 + 0.36788}{2} + 0.93941 + 0.77880 + 0.56978\right] = 0.25[0.68394 + 2.28799] = 0.74298$$

Error: $|0.74298 - 0.74682| = 0.00384$ — $O(h^2)$ as expected.

**Simpson's with $n=4$** (need even $n$):

$$S_4 = \frac{0.25}{3}[f_0 + 4f_1 + 2f_2 + 4f_3 + f_4]$$
$$= \frac{0.25}{3}[1 + 4(0.93941) + 2(0.77880) + 4(0.56978) + 0.36788]$$
$$= \frac{0.25}{3}[1 + 3.75764 + 1.55760 + 2.27912 + 0.36788] = \frac{0.25}{3}(8.96224) = 0.74685$$

Error: $|0.74685 - 0.74682| = 0.00003$ — vastly smaller. Simpson's is $O(h^4)$.

### Q.2 Exercise 6 — Expected Value and KL Divergence

**Problem**: Compute $D_{KL}(p \| q)$ where $p = N(\mu, 1)$ and $q = N(0, 1)$.

**Solution**: 

$$D_{KL}(p\|q) = \int_{-\infty}^\infty p(x)\ln\frac{p(x)}{q(x)}\,dx$$

With $p(x) = \frac{1}{\sqrt{2\pi}}e^{-(x-\mu)^2/2}$ and $q(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$:

$$\ln\frac{p(x)}{q(x)} = -\frac{(x-\mu)^2}{2} + \frac{x^2}{2} = \frac{x^2 - (x-\mu)^2}{2} = \frac{2\mu x - \mu^2}{2} = \mu x - \frac{\mu^2}{2}$$

$$D_{KL}(p\|q) = \mathbb{E}_p\!\left[\mu x - \frac{\mu^2}{2}\right] = \mu\,\mathbb{E}_p[x] - \frac{\mu^2}{2} = \mu^2 - \frac{\mu^2}{2} = \frac{\mu^2}{2}$$

**Interpretation**: The KL divergence from $N(0,1)$ to $N(\mu,1)$ is exactly $\mu^2/2$. This is the term in the VAE ELBO that penalizes the encoder mean from drifting far from zero.

### Q.3 Exercise 7 — Gradient of Expected Loss

**Problem**: Compute $\nabla_\theta \mathbb{E}_{x \sim p_\theta}[\ell(x)]$ via the log-derivative trick.

**Solution**: Using the score function / REINFORCE estimator:

$$\nabla_\theta \mathbb{E}_{p_\theta}[\ell(x)] = \nabla_\theta \int \ell(x)\,p_\theta(x)\,dx$$

Assuming we can differentiate under the integral (DCT applies):

$$= \int \ell(x)\,\nabla_\theta p_\theta(x)\,dx = \int \ell(x)\,p_\theta(x)\,\frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}\,dx = \mathbb{E}_{p_\theta}[\ell(x)\,\nabla_\theta\ln p_\theta(x)]$$

This is the REINFORCE gradient estimator. It requires only samples from $p_\theta$ and the ability to evaluate $\ln p_\theta$ — no reparameterization needed. The cost is high variance, motivating control variates (baselines) to reduce $\text{Var}[\ell(x)\nabla_\theta\ln p_\theta(x)]$.


---

## Appendix R: Historical Timeline Extended

### R.1 Chronological Development

| Era | Contributor | Achievement |
|-----|-------------|-------------|
| ~250 BCE | Archimedes | Method of exhaustion for area of parabolic segment; $\pi$ bounds |
| ~1640s | Cavalieri | Cavalieri's principle; "method of indivisibles" |
| 1665–1666 | Newton | Inverse tangent problem; "method of fluxions"; FTC discovered privately |
| 1675–1684 | Leibniz | Independent discovery; modern $\int$ notation; publication of FTC (1684) |
| 1696 | L'Hôpital | First calculus textbook (based on Bernoulli's lectures) |
| 1734 | Bishop Berkeley | *The Analyst* — criticism of infinitesimals as "ghosts of departed quantities" |
| 1748 | Euler | *Introductio in Analysin Infinitorum* — systematic treatment of functions |
| 1821–1823 | Cauchy | Rigorous definition of limit; definite integral via Riemann-style sums |
| 1854 | Riemann | Rigorous Riemann integral; characterization of integrable functions |
| 1875 | Darboux | Upper/lower sums; cleaner formulation of Riemann integral |
| 1894–1902 | Lebesgue | Measure theory; Lebesgue integral; MCT and DCT |
| 1900s | Hilbert | $L^2$ spaces; integration as inner product; functional analysis |
| 1920s–1940s | Kolmogorov | Probability as measure theory; rigorous foundation for ML |
| 1940s–1950s | Monte Carlo | Ulam, von Neumann, Metropolis — stochastic integration for physics |
| 1986 | Rumelhart et al. | Backpropagation as chain rule for integrals (Jacobians) |
| 2018 | Chen et al. | Neural ODEs — continuous-depth networks via ODE integration |
| 2020s | Diffusion models | Score matching, DDPM, flow matching — integration at the core of generation |

### R.2 The Newton-Leibniz Priority Dispute

The FTC was discovered independently by Newton (1666, unpublished) and Leibniz (1675–1684, published). The Royal Society's official investigation (1712) wrongly accused Leibniz of plagiarism, damaging his reputation and creating a rift between British and Continental mathematicians. British mathematicians, loyal to Newton's notation, fell behind Continental Europe for ~100 years. The lesson: notation matters — Leibniz's $\int$ and $d/dx$ notation won out and is the notation used universally today.

### R.3 Why Newton Called Integration "Quadrature"

Newton's original term for integration was "quadrature" — from the Latin for "making a square." The original problem was: given a curve, find a square with the same area. This geometric framing persisted for centuries. Leibniz's more algebraic approach (anti-differentiation) is what we use today, but the term "quadrature" survives in "Gaussian quadrature" — numerical integration using optimal node placement.

---

## Appendix S: Advanced Topics Preview

This section previews topics that build directly on integration and appear in subsequent sections:

### S.1 → §04 Sequences and Series

**Taylor series** represents a function as an infinite sum:
$$f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n$$

The **remainder term** in Taylor's theorem is an integral:
$$R_n(x) = \frac{1}{n!}\int_a^x (x-t)^n f^{(n+1)}(t)\,dt$$

Integration and series interact via **term-by-term integration** — valid when a series converges uniformly:
$$\int \sum_{n=0}^\infty a_n x^n \, dx = \sum_{n=0}^\infty \frac{a_n x^{n+1}}{n+1}$$

→ *Full treatment: [Sequences and Series](../04-Sequences-and-Series/notes.md)*

### S.2 → §05 Multivariable Calculus

Double and triple integrals extend the 1D theory:
$$\iint_D f(x,y)\,dA = \int_a^b \int_{g(x)}^{h(x)} f(x,y)\,dy\,dx \quad \text{(Fubini's theorem)}$$

The **change of variables formula** with Jacobian $|J|$:
$$\iint_D f(x,y)\,dA = \iint_{D'} f(x(u,v), y(u,v))\,|J|\,du\,dv$$

→ *Full treatment: [Multivariable Calculus](../05-Multivariable-Calculus/notes.md)*

### S.3 → §06 Probability and Statistics

Continuous random variables live entirely in the integration framework. The CDF is an integral of the PDF; expectation is an integral; the central limit theorem involves convergence of distribution functions; characteristic functions are Fourier transforms — all integration.

→ *Full treatment: [Probability and Statistics](../06-Probability-and-Statistics/notes.md)*


---

## Appendix T: Self-Assessment Checklist

After completing this section, you should be able to:

**Conceptual Understanding**
- [ ] Explain integration as accumulated change (area, probability mass, expected value)
- [ ] State both parts of the FTC and explain why they are profound
- [ ] Distinguish when a function is/is not Riemann integrable
- [ ] Explain why $\int_a^\infty \frac{1}{x^p}\,dx$ converges iff $p > 1$

**Computational Skills**
- [ ] Compute integrals using power rule, u-substitution, and integration by parts
- [ ] Decompose rational functions into partial fractions
- [ ] Evaluate improper integrals or determine their convergence
- [ ] Implement the trapezoid rule, Simpson's rule, and basic Monte Carlo integration

**Probability Connections**
- [ ] Verify that a given function is a valid PDF (non-negative, integrates to 1)
- [ ] Compute $\mathbb{E}[X]$, $\text{Var}[X]$ as integrals for common distributions
- [ ] Compute KL divergence between two Gaussian distributions
- [ ] Derive the cross-entropy loss as negative log-likelihood

**ML Applications**
- [ ] Explain Monte Carlo as an approximation to an integral
- [ ] State the REINFORCE gradient estimator and its derivation via log-derivative trick
- [ ] Explain the VAE ELBO as a sum of two integrals (reconstruction + KL)
- [ ] Describe the neural ODE forward pass as numerical ODE integration
- [ ] Sketch how score matching connects to integration in diffusion models

