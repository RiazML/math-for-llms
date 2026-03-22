[← Back to Calculus Fundamentals](../README.md) | [Next: Integration →](../03-Integration/notes.md)

---

# Derivatives and Differentiation

> _"The derivative is the heart of calculus. Everything else — integrals, series, differential equations — is built around it."_
> — Michael Spivak, _Calculus_ (1967)

## Overview

The derivative formalizes the intuition of instantaneous rate of change. Where a limit asks "what value does $f(x)$ approach?", a derivative asks "how fast is $f$ changing right now?" This single idea — the slope of the tangent line, the instantaneous velocity, the marginal cost — is the load-bearing concept of all continuous optimization.

In machine learning, every gradient computation is a derivative. The backpropagation algorithm is the chain rule applied systematically to a computation graph. The vanishing gradient problem is a consequence of activation function derivatives approaching zero. The Adam optimizer's adaptive learning rates are functions of accumulated squared derivatives. Understanding derivatives at the mathematical level — not just as "the thing PyTorch computes" — is essential for debugging, designing, and innovating in modern AI.

This section develops single-variable derivatives from the limit definition through all standard rules, applies them to activation functions and computation graphs, and connects numerical differentiation to practical gradient checking.

## Prerequisites

- **Limits and continuity** — [§04/01-Limits-and-Continuity](../01-Limits-and-Continuity/notes.md): ε-δ definition, limit laws, continuity
- **Functions**: domain, range, composition, inverse
- **Algebra**: polynomial manipulation, exponential and logarithmic identities
- **Trigonometry**: $\sin x$, $\cos x$ and their basic identities

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive: derivative definition, rules, activation functions, backprop, numerical differentiation |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from power rule to backpropagation and gradient checking |

## Learning Objectives

After completing this section, you will be able to:

1. **Define** the derivative as a limit and verify it for elementary functions from first principles
2. **Apply** the power, product, quotient, and chain rules fluently
3. **Differentiate** composite, implicit, and inverse functions
4. **Compute** derivatives of all standard activation functions and explain their ML significance
5. **State and prove** the Mean Value Theorem and apply it to optimization arguments
6. **Classify** critical points using first and second derivative tests
7. **Implement** one-sided and centered finite difference approximations with correct error order
8. **Perform** gradient checking to validate autodiff implementations
9. **Trace** backpropagation through a scalar computation graph using the chain rule
10. **Connect** derivative concepts to vanishing gradients, dying ReLU, and learning rate behavior

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What a Derivative Measures](#11-what-a-derivative-measures)
  - [1.2 Historical Motivation](#12-historical-motivation)
  - [1.3 Why Derivatives Are Central to AI](#13-why-derivatives-are-central-to-ai)
- [2. Formal Definition](#2-formal-definition)
  - [2.1 The Derivative as a Limit](#21-the-derivative-as-a-limit)
  - [2.2 Differentiability vs. Continuity](#22-differentiability-vs-continuity)
  - [2.3 Notational Systems](#23-notational-systems)
  - [2.4 One-Sided Derivatives](#24-one-sided-derivatives)
- [3. Basic Differentiation Rules](#3-basic-differentiation-rules)
  - [3.1 Constant and Power Rules](#31-constant-and-power-rules)
  - [3.2 Sum and Constant-Multiple Rules](#32-sum-and-constant-multiple-rules)
  - [3.3 Product Rule](#33-product-rule)
  - [3.4 Quotient Rule](#34-quotient-rule)
  - [3.5 Elementary Function Derivatives](#35-elementary-function-derivatives)
- [4. The Chain Rule](#4-the-chain-rule)
  - [4.1 Statement and Proof](#41-statement-and-proof)
  - [4.2 Composite Function Examples](#42-composite-function-examples)
  - [4.3 Chain Rule as Backpropagation](#43-chain-rule-as-backpropagation)
  - [4.4 Forward Reference: Multivariable Chain Rule](#44-forward-reference-multivariable-chain-rule)
- [5. Implicit Differentiation and Related Rates](#5-implicit-differentiation-and-related-rates)
  - [5.1 Implicit Differentiation](#51-implicit-differentiation)
  - [5.2 Inverse Function Derivative](#52-inverse-function-derivative)
  - [5.3 Related Rates](#53-related-rates)
- [6. Higher-Order Derivatives](#6-higher-order-derivatives)
  - [6.1 Second and n-th Derivatives](#61-second-and-n-th-derivatives)
  - [6.2 Concavity and Inflection Points](#62-concavity-and-inflection-points)
  - [6.3 Linear Approximation](#63-linear-approximation)
- [7. Applications: Extrema and the Mean Value Theorem](#7-applications-extrema-and-the-mean-value-theorem)
  - [7.1 Critical Points and First Derivative Test](#71-critical-points-and-first-derivative-test)
  - [7.2 Second Derivative Test](#72-second-derivative-test)
  - [7.3 Rolle's Theorem and the Mean Value Theorem](#73-rolles-theorem-and-the-mean-value-theorem)
  - [7.4 Convexity and Global Minima](#74-convexity-and-global-minima)
- [8. Activation Function Derivatives](#8-activation-function-derivatives)
  - [8.1 Sigmoid](#81-sigmoid)
  - [8.2 Tanh](#82-tanh)
  - [8.3 ReLU and Dying ReLU](#83-relu-and-dying-relu)
  - [8.4 GELU and SiLU](#84-gelu-and-silu)
  - [8.5 Vanishing and Exploding Gradients](#85-vanishing-and-exploding-gradients)
  - [8.6 Softmax Preview](#86-softmax-preview)
- [9. Numerical Differentiation](#9-numerical-differentiation)
  - [9.1 Finite Differences](#91-finite-differences)
  - [9.2 Optimal Step Size](#92-optimal-step-size)
  - [9.3 Gradient Checking](#93-gradient-checking)
  - [9.4 Automatic vs. Numerical Differentiation](#94-automatic-vs-numerical-differentiation)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 What a Derivative Measures

Consider a car's position $s(t)$ at time $t$. The **average velocity** over the interval $[t, t+h]$ is:
$$\bar{v} = \frac{s(t+h) - s(t)}{h}$$

As $h \to 0$, this average approaches the **instantaneous velocity** at time $t$. That limit — if it exists — is the derivative $s'(t)$.

Geometrically, the average velocity is the slope of a **secant line** through $(t, s(t))$ and $(t+h, s(t+h))$. As $h \to 0$, the secant approaches the **tangent line** at $(t, s(t))$.

```
DERIVATIVE: FROM SECANT TO TANGENT
════════════════════════════════════════════════════════════════════════

  f(x)
    │                         ● (x+h, f(x+h))
    │                       ╱ |
    │                     ╱   |
    │           ●───────╱─────┘   secant slope = [f(x+h)-f(x)]/h
    │         ╱ (x, f(x))
    │       ╱
    │     ╱ ← tangent line at x (slope = f'(x))
    └────────────────────────────────────→  x
              x       x+h

  As h → 0: secant → tangent, average rate → instantaneous rate.

════════════════════════════════════════════════════════════════════════
```

The derivative measures **sensitivity**: how much does output change per unit change in input? If $f'(a) = 5$, a small nudge $\delta$ in input produces a change of approximately $5\delta$ in output.

### 1.2 Historical Motivation

The derivative was invented twice — independently and almost simultaneously — by two of history's greatest mathematicians, triggering a bitter priority dispute that divided European mathematics for a generation.

**Isaac Newton (1665–1666)** developed "the method of fluxions" during the Great Plague, when Cambridge was closed and Newton worked alone at Woolsthorpe Manor. He called the derivative a "fluxion" — the rate of flow of a "fluent" quantity. He used the notation $\dot{x}$ (dot notation still used in physics). Newton did not publish for decades.

**Gottfried Wilhelm Leibniz (1675)** independently developed calculus in Paris, motivated by geometric problems. He introduced the notation $dy/dx$ and $\int$ (still universal today). He published first, in 1684.

The Royal Society's 1713 judgment in favor of Newton — almost certainly biased — poisoned British-Continental mathematical relations for over a century, isolating British mathematics from advances made by the Bernoullis, Euler, and Lagrange.

| Year | Event |
|------|-------|
| 1665 | Newton develops fluxions (unpublished) |
| 1675 | Leibniz develops calculus independently |
| 1684 | Leibniz publishes first |
| 1687 | Newton publishes Principia (uses fluxions implicitly) |
| 1713 | Royal Society (controlled by Newton) rules for Newton |
| 1821 | Cauchy formalizes with limits (modern foundation) |
| 1861 | Weierstrass gives ε-δ definition |

**Lagrange (1797)** introduced the prime notation $f'(x)$ and attempted an algebraic foundation via power series. **Cauchy (1821)** finally gave the limit-based definition we use today.

### 1.3 Why Derivatives Are Central to AI

Every parameter update in every gradient-based learning algorithm is computed via derivatives.

**Backpropagation** (Rumelhart, Hinton, Williams, 1986) is the chain rule applied to a computation graph. Given a loss $\mathcal{L}$ and parameters $\theta_1, \ldots, \theta_n$, the update $\theta_i \leftarrow \theta_i - \alpha \partial\mathcal{L}/\partial\theta_i$ requires computing all partial derivatives efficiently — which backprop does in $O(\text{forward pass})$ time.

**Vanishing gradients** (Hochreiter, 1991; Hochreiter & Schmidhuber, 1997): sigmoid activations have $\sigma'(x) \leq 1/4$. In a 20-layer network, the chain rule multiplies 20 such factors: $(1/4)^{20} \approx 10^{-12}$. Early layers receive effectively zero gradient — they cannot learn. The fix (ReLU, residual connections, layer norm) is entirely about derivative behavior.

**Adam optimizer** (Kingma & Ba, 2014): the adaptive learning rate $\alpha/(\sqrt{\hat{v}_t} + \varepsilon)$ where $\hat{v}_t$ is the exponential moving average of squared gradients $(\partial\mathcal{L}/\partial\theta)^2$. Every component is derivative-driven.

**Neural architecture search, meta-learning, differentiable programming**: modern AI increasingly treats the architecture itself as differentiable — requiring derivatives through discrete operations via straight-through estimators, Gumbel-softmax, and continuous relaxations.

---

## 2. Formal Definition

### 2.1 The Derivative as a Limit

> **Recall:** The limit $\lim_{x \to a} f(x)$ was defined rigorously in [§01-Limits-and-Continuity](../01-Limits-and-Continuity/notes.md#2-formal-definitions). The derivative is a specific instance of that definition.

**Definition (Derivative).** The **derivative** of $f$ at $a$, written $f'(a)$, is:
$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$
provided this limit exists (is finite). If it exists, $f$ is **differentiable at $a$**.

The expression $\frac{f(a+h)-f(a)}{h}$ is called a **difference quotient**. It is the slope of the secant line through $(a, f(a))$ and $(a+h, f(a+h))$.

**Example: $f(x) = x^2$.**
$$f'(a) = \lim_{h \to 0} \frac{(a+h)^2 - a^2}{h} = \lim_{h \to 0} \frac{2ah + h^2}{h} = \lim_{h \to 0}(2a + h) = 2a$$

So $(x^2)' = 2x$. The derivative of $x^2$ at $a = 3$ is $f'(3) = 6$.

**Example: $f(x) = \sqrt{x}$ at $a > 0$.**
$$f'(a) = \lim_{h \to 0} \frac{\sqrt{a+h} - \sqrt{a}}{h} \cdot \frac{\sqrt{a+h}+\sqrt{a}}{\sqrt{a+h}+\sqrt{a}} = \lim_{h \to 0} \frac{h}{h(\sqrt{a+h}+\sqrt{a})} = \frac{1}{2\sqrt{a}}$$

**Example: $f(x) = e^x$.**
$$f'(a) = \lim_{h \to 0} \frac{e^{a+h} - e^a}{h} = e^a \lim_{h \to 0} \frac{e^h - 1}{h} = e^a \cdot 1 = e^a$$

using the fundamental limit $\lim_{h\to 0}(e^h - 1)/h = 1$ from [§01 §3.2](../01-Limits-and-Continuity/notes.md#32-exponential-and-logarithmic-limits). So $(e^x)' = e^x$ — the exponential is its own derivative.

**The derivative as a function.** If $f$ is differentiable at every $x$ in its domain, the rule $x \mapsto f'(x)$ defines a new function $f': \mathbb{R} \to \mathbb{R}$ called the **derivative function**.

### 2.2 Differentiability vs. Continuity

**Theorem.** If $f$ is differentiable at $a$, then $f$ is continuous at $a$.

**Proof.** We show $\lim_{h\to 0} f(a+h) = f(a)$:
$$f(a+h) - f(a) = \frac{f(a+h)-f(a)}{h} \cdot h \to f'(a) \cdot 0 = 0 \qquad (h \to 0)$$
So $\lim_{h\to 0} f(a+h) = f(a)$. $\square$

**The converse is false**: continuity does not imply differentiability.

**Counterexamples:**
- $f(x) = \lvert x \rvert$: continuous everywhere, but at $x=0$: left derivative $= -1$, right derivative $= +1$ — not equal, so not differentiable
- $f(x) = x^{1/3}$: continuous at $0$, but $\lvert h^{1/3} \rvert / \lvert h \rvert = \lvert h \rvert^{-2/3} \to \infty$ — vertical tangent, not differentiable
- Weierstrass function (1872): continuous everywhere, differentiable nowhere — the first published example showing the two concepts are genuinely distinct

```
DIFFERENTIABILITY VS. CONTINUITY
════════════════════════════════════════════════════════════════════════

  All differentiable functions          Continuous functions
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │   ┌─────────────────────┐                                   │
  │   │  Differentiable      │   |x|  x^{1/3}  Weierstrass fn  │
  │   │  (smooth, no kinks)  │                                   │
  │   └─────────────────────┘                                   │
  └─────────────────────────────────────────────────────────────┘

  Differentiable ⟹ Continuous    (proven above)
  Continuous ⟹ Differentiable    FALSE (counterexamples above)

════════════════════════════════════════════════════════════════════════
```

**For AI:** ReLU$(x) = \max(0,x)$ is continuous everywhere but not differentiable at $x=0$ (left derivative $0$, right derivative $1$). In practice, we use a **subgradient** — any value in $[0,1]$ at $x=0$, typically $0$ or $1/2$. Training converges despite this because $P(\text{activation} = 0) = 0$ for continuously distributed pre-activations.

### 2.3 Notational Systems

Four equivalent notations coexist; knowing all is essential for reading across fields:

| Notation | Symbol | Usage |
|----------|--------|-------|
| Lagrange (prime) | $f'(x)$, $f''(x)$, $f^{(n)}(x)$ | Most common in pure math |
| Leibniz (fraction) | $\dfrac{dy}{dx}$, $\dfrac{d^2y}{dx^2}$ | Chain rule, implicit diff, physics |
| Newton (dot) | $\dot{x}$, $\ddot{x}$ | Physics, ODEs, time derivatives |
| Euler (operator) | $Df$, $D^2 f$ | Functional analysis, operator theory |

**Leibniz notation** is the most powerful for computation because it makes the chain rule look like fraction cancellation:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Higher order:** $f''(x) = \frac{d^2y}{dx^2} = \frac{d}{dx}\!\left(\frac{dy}{dx}\right)$. Note: $\left(\frac{dy}{dx}\right)^2 \neq \frac{d^2y}{dx^2}$ — a common error.

### 2.4 One-Sided Derivatives

**Definition.** The **right-hand derivative** is $f'_+(a) = \lim_{h \to 0^+}\frac{f(a+h)-f(a)}{h}$. The **left-hand derivative** is $f'_-(a) = \lim_{h\to 0^-}\frac{f(a+h)-f(a)}{h}$.

$f$ is differentiable at $a$ if and only if both one-sided derivatives exist and are equal: $f'_-(a) = f'_+(a)$.

**For AI — ReLU subgradient:** At $x = 0$: $\text{ReLU}'_-(0) = 0$ and $\text{ReLU}'_+(0) = 1$. Since $0 \neq 1$, ReLU is not differentiable at $0$. Frameworks like PyTorch use the convention $\text{ReLU}'(0) = 0$ (the left derivative), which works in practice.

---

## 3. Basic Differentiation Rules

These rules allow computing derivatives algebraically without evaluating limits each time. All are proved from the limit definition.

### 3.1 Constant and Power Rules

**Constant Rule:** $(c)' = 0$ for any constant $c$.

*Proof:* $\lim_{h\to 0}(c - c)/h = \lim_{h\to 0} 0 = 0$. $\square$

**Power Rule:** $(x^n)' = nx^{n-1}$ for any real $n$.

*Proof (integer $n \geq 1$):* Expand $(x+h)^n$ by the binomial theorem:
$$(x+h)^n = x^n + nx^{n-1}h + \binom{n}{2}x^{n-2}h^2 + \cdots$$
Subtract $x^n$, divide by $h$, let $h \to 0$: only the $nx^{n-1}$ term survives. $\square$

The power rule holds for all real $n$ (including fractions and negatives):
$$\frac{d}{dx}x^{1/2} = \frac{1}{2}x^{-1/2}, \quad \frac{d}{dx}x^{-1} = -x^{-2}, \quad \frac{d}{dx}x^\pi = \pi x^{\pi-1}$$

### 3.2 Sum and Constant-Multiple Rules

**Sum Rule:** $(f + g)' = f' + g'$

**Constant-Multiple Rule:** $(cf)' = cf'$

Together: $(af + bg)' = af' + bg'$ — differentiation is **linear**.

**Consequence:** Differentiating polynomials term by term:
$$\frac{d}{dx}(3x^4 - 2x^2 + 7x - 1) = 12x^3 - 4x + 7$$

### 3.3 Product Rule

**Theorem.** $(fg)' = f'g + fg'$

**Proof:**
$$\frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$
Add and subtract $f(x+h)g(x)$:
$$= \frac{f(x+h)-f(x)}{h} \cdot g(x+h) + f(x) \cdot \frac{g(x+h)-g(x)}{h}$$
As $h \to 0$: first term $\to f'(x)g(x)$, second $\to f(x)g'(x)$. $\square$

**Memory aid:** "derivative of first times second, plus first times derivative of second."

**Examples:**
$$\frac{d}{dx}[x^2 e^x] = 2xe^x + x^2 e^x = xe^x(2+x)$$
$$\frac{d}{dx}[x\sin x] = \sin x + x\cos x$$

**Generalized product (Leibniz rule):**
$$(f_1 f_2 \cdots f_n)' = \sum_{k=1}^n f_1 \cdots f_{k-1} f_k' f_{k+1} \cdots f_n$$

### 3.4 Quotient Rule

**Theorem.** $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$, provided $g(x) \neq 0$.

**Proof:** Write $f = (f/g) \cdot g$ and apply the product rule to both sides. $\square$

**Memory aid:** "lo d-hi minus hi d-lo, over lo squared" (where hi $= f$, lo $= g$).

**Example:**
$$\frac{d}{dx}\frac{\sin x}{x} = \frac{x\cos x - \sin x}{x^2}$$

### 3.5 Elementary Function Derivatives

**Core table — memorize these:**

| Function | Derivative | Notes |
|----------|-----------|-------|
| $e^x$ | $e^x$ | Only function equal to its own derivative |
| $a^x$ | $a^x \ln a$ | For any base $a > 0$ |
| $\ln x$ | $1/x$ | For $x > 0$ |
| $\log_a x$ | $1/(x \ln a)$ | Change of base |
| $\sin x$ | $\cos x$ | |
| $\cos x$ | $-\sin x$ | Note the negative |
| $\tan x$ | $\sec^2 x$ | |
| $\arcsin x$ | $1/\sqrt{1-x^2}$ | Via inverse function rule |
| $\arctan x$ | $1/(1+x^2)$ | Via inverse function rule |

**Proof: $(\ln x)' = 1/x$.**
$$\lim_{h\to 0} \frac{\ln(x+h) - \ln x}{h} = \lim_{h\to 0} \frac{1}{h}\ln\!\left(1 + \frac{h}{x}\right) = \frac{1}{x}\lim_{u\to 0}\frac{\ln(1+u)}{u} = \frac{1}{x} \cdot 1 = \frac{1}{x}$$
using $\lim_{u\to 0}\ln(1+u)/u = 1$ (from [§01 §3.2](../01-Limits-and-Continuity/notes.md)). $\square$

---

## 4. The Chain Rule

### 4.1 Statement and Proof

The chain rule handles composites — functions of functions. If $y = f(g(x))$, then:

$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x) \qquad \text{(Leibniz: } \frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}\text{)}$$

**Proof (informal but rigorous enough for most purposes).** Let $u = g(x)$ and $\Delta u = g(x+h) - g(x)$. Then:

$$\frac{\Delta y}{\Delta x} = \frac{\Delta y}{\Delta u} \cdot \frac{\Delta u}{\Delta x}$$

Taking $h \to 0$: if $g'(x)$ exists and $f'(u)$ exists at $u = g(x)$, the limit yields the chain rule. (The formal proof handles the case $\Delta u = 0$ separately via an auxiliary function.)

### 4.2 Worked Examples

**Example 1.** $y = \sin(x^2)$. Let $u = x^2$, $y = \sin u$.
$$y' = \cos(x^2) \cdot 2x$$

**Example 2.** $y = e^{-x^2/2}$ (Gaussian kernel).
$$y' = e^{-x^2/2} \cdot (-x) = -x\,e^{-x^2/2}$$

**Example 3 (triple composition).** $y = \ln(\sin(e^x))$.
$$y' = \frac{1}{\sin(e^x)} \cdot \cos(e^x) \cdot e^x = \frac{e^x \cos(e^x)}{\sin(e^x)}$$

The pattern: differentiate outer, keep inner untouched, multiply by derivative of inner — repeat inward.

### 4.3 Chain Rule and Backpropagation

Every neural network is a deeply composed function:

$$\mathcal{L} = \ell(f_n(f_{n-1}(\cdots f_1(\mathbf{x})\cdots)))$$

Backpropagation **is** the chain rule applied to a computation graph. The gradient of the loss with respect to any parameter is a product of local derivatives along the path from that parameter to the output. The chain rule guarantees this product equals the true gradient — the theoretical justification for gradient-based learning.

> **Forward reference.** The multivariable chain rule (Jacobians, backpropagation for vector-valued functions) is the canonical topic of [§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md). Here we develop the single-variable foundation only.


---

## 5. Implicit Differentiation and Related Rates

### 5.1 Implicit Differentiation

Not every curve is a function $y = f(x)$. The unit circle $x^2 + y^2 = 1$ defines $y$ implicitly. To find $dy/dx$, differentiate both sides with respect to $x$, treating $y$ as a function of $x$:

$$\frac{d}{dx}[x^2 + y^2] = \frac{d}{dx}[1] \implies 2x + 2y\frac{dy}{dx} = 0 \implies \frac{dy}{dx} = -\frac{x}{y}$$

**General procedure:**
1. Differentiate both sides w.r.t. $x$; apply chain rule wherever $y$ appears (treat $y$ as $y(x)$).
2. Collect all $dy/dx$ terms on one side.
3. Solve algebraically for $dy/dx$.

**Example.** Derive $(\arctan x)' = 1/(1+x^2)$.

Let $y = \arctan x$, so $\tan y = x$. Differentiate implicitly:
$$\sec^2 y \cdot \frac{dy}{dx} = 1 \implies \frac{dy}{dx} = \cos^2 y = \frac{1}{1 + \tan^2 y} = \frac{1}{1+x^2}. \quad \square$$

**For AI:** Implicit differentiation underlies the derivation of softmax Jacobians and the inverse function theorem applied to normalizing flows — generative models that learn invertible mappings.

### 5.2 Related Rates

Related rates problems ask: if two quantities are linked by an equation, how fast does one change relative to the other? Differentiate the linking equation w.r.t. time $t$.

**Example.** The cross-entropy loss $\mathcal{L} = -\log p$ and the logit $z$ are linked by the sigmoid: $p = \sigma(z) = 1/(1+e^{-z})$. Rate of change of loss w.r.t. logit:

$$\frac{d\mathcal{L}}{dz} = \frac{d\mathcal{L}}{dp} \cdot \frac{dp}{dz} = \frac{-1}{p} \cdot p(1-p) = -(1-p) = p - 1$$

This is the gradient passed to the logit layer during backpropagation — related rates in disguise.

### 5.3 Linear Approximation

The tangent line at $x = a$ approximates $f$ locally:

$$f(x) \approx f(a) + f'(a)(x-a)$$

This **linearization** is the first-order Taylor approximation. It underlies gradient descent: the loss surface near $\theta$ looks like a hyperplane, and gradient descent steps in the direction of steepest descent on that plane.

> **Forward reference.** The full Taylor series (all higher-order terms, remainder, convergence) is the canonical topic of [§04-Series-and-Sequences](../04-Series-and-Sequences/notes.md).


---

## 6. Higher-Order Derivatives and Concavity

### 6.1 Definition and Notation

The second derivative is the derivative of the derivative:

$$f''(x) = \frac{d}{dx}[f'(x)] = \frac{d^2f}{dx^2}$$

Higher-order derivatives: $f^{(n)}(x) = \frac{d^n f}{dx^n}$.

**Geometric meaning:**
- $f'(x) > 0$: $f$ is increasing at $x$
- $f'(x) < 0$: $f$ is decreasing at $x$
- $f''(x) > 0$: $f'$ is increasing → $f$ is **concave up** (bowl shape ∪)
- $f''(x) < 0$: $f'$ is decreasing → $f$ is **concave down** (dome shape ∩)

### 6.2 Inflection Points

A point $x_0$ is an **inflection point** if $f''$ changes sign at $x_0$. The concavity flips — from ∪ to ∩ or vice versa. Necessary condition: $f''(x_0) = 0$ (but not sufficient — check sign change).

**Example.** $f(x) = x^3$: $f''(x) = 6x$. Changes sign at $x = 0$: inflection point.
**Non-example.** $f(x) = x^4$: $f''(x) = 12x^2 \geq 0$, equals zero at $x=0$ but no sign change — not an inflection point.

### 6.3 Significance for Optimization

The second derivative determines the local curvature of the loss surface:
- **Positive curvature** ($f'' > 0$): minimum is possible here; gradient descent converges toward it.
- **Negative curvature** ($f'' < 0$): maximum; gradient descent moves away from saddle regions.
- **Zero curvature**: saddle point candidate — common in high-dimensional loss landscapes.

The **Hessian** (matrix of second partial derivatives, covered in [§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md)) generalizes $f''$ to multi-parameter models. Positive definite Hessians certify local minima.

> **Recall:** Positive definite matrices were defined in [§07-Positive-Definite-Matrices](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md) — a matrix $H$ is PD iff $\mathbf{v}^\top H \mathbf{v} > 0$ for all nonzero $\mathbf{v}$.

### 6.4 Taylor Coefficients Foreshadowed

The $n$-th derivative at a point gives the $n$-th Taylor coefficient:

$$a_n = \frac{f^{(n)}(a)}{n!}$$

So $f''$ controls the quadratic correction to the linear approximation — exactly the second-order term Adam exploits by tracking gradient variance as a proxy for curvature.

> **Forward reference.** The full derivation of Taylor series using higher-order derivatives is in [§04-Series-and-Sequences](../04-Series-and-Sequences/notes.md).


---

## 7. Extrema, Critical Points, and the Mean Value Theorem

### 7.1 Critical Points and the First Derivative Test

A **critical point** of $f$ is any $x_0$ where $f'(x_0) = 0$ or $f'(x_0)$ does not exist.

**First Derivative Test.** If $f'$ changes sign at $x_0$:
- $- \to +$: local minimum
- $+ \to -$: local maximum
- No sign change: saddle point / inflection

**Example.** $f(x) = x^3 - 3x$. Critical points: $f'(x) = 3x^2 - 3 = 0 \Rightarrow x = \pm 1$.
- $x = -1$: $f'$ changes $+ \to -$ → local max, $f(-1) = 2$
- $x = 1$: $f'$ changes $- \to +$ → local min, $f(1) = -2$

### 7.2 Second Derivative Test

If $f'(x_0) = 0$:
- $f''(x_0) > 0$: local **minimum**
- $f''(x_0) < 0$: local **maximum**
- $f''(x_0) = 0$: inconclusive — higher-order test needed

**For AI:** In a scalar model, the second derivative test tells us whether a parameter has converged to a local minimum or is at a saddle. In neural networks this extends to the Hessian eigenspectrum — if all eigenvalues are positive, we've found a local minimum.

### 7.3 Global Extrema on Closed Intervals

> **Recall:** The Extreme Value Theorem (EVT) from [§01-Limits-and-Continuity](../01-Limits-and-Continuity/notes.md) guarantees that a continuous function on $[a,b]$ attains its maximum and minimum.

**Algorithm for global extrema on $[a,b]$:**
1. Find all critical points in $(a,b)$ where $f'(x) = 0$ or $f'$ undefined.
2. Evaluate $f$ at critical points and endpoints $a$, $b$.
3. The largest value is the global max; the smallest is the global min.

### 7.4 The Mean Value Theorem

**Theorem (MVT).** If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, there exists $c \in (a,b)$ such that:

$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

Geometrically: there is a point where the tangent slope equals the average (secant) slope.

**Proof.** Apply Rolle's Theorem (MVT with $f(a) = f(b)$) to $g(x) = f(x) - \frac{f(b)-f(a)}{b-a}(x-a)$.

**For optimization:** The MVT bounds how much a function can change along a gradient descent path. If $|f'| \leq L$ everywhere, then $|f(x) - f(y)| \leq L|x-y|$ — a Lipschitz bound used to set safe learning rates.

> **Forward reference.** Critical point analysis extends to multivariable functions via gradient = 0 and Hessian definiteness in [§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md) and [§08-Optimization](../../08-Optimization/README.md).


---

## 8. Activation Function Derivatives

> **Recall:** Continuity of activation functions (whether ReLU, GELU, sigmoid are continuous at the origin) was analyzed in [§01-Limits-and-Continuity §10](../01-Limits-and-Continuity/notes.md). Here we derive their **derivatives** — essential for backpropagation.

### 8.1 Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

**Derivative.** Using the quotient rule with $u = 1$, $v = 1+e^{-x}$:

$$\sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(x)(1-\sigma(x))$$

This self-referential form means the gradient can be computed from the forward-pass output — no second exponential needed. **Peak value:** $\sigma'(0) = 1/4$.

**Vanishing gradient.** As $|x| \to \infty$: $\sigma(x) \to 0$ or $1$, so $\sigma'(x) \to 0$. Products of many near-zero gradients → exponential decay through layers.

### 8.2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Derivative.** Using $\tanh(x) = 2\sigma(2x) - 1$ or direct quotient rule:

$$\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$$

**Peak value:** $\tanh'(0) = 1$ — double that of sigmoid. Still vanishes for large $|x|$.

### 8.3 ReLU and Variants

$$\text{ReLU}(x) = \max(0, x), \qquad \text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \end{cases}$$

Undefined at $x = 0$; in practice subgradient $0$ is used. **No vanishing gradient for active units** ($x > 0$) — the key advantage over sigmoid/tanh.

**Leaky ReLU:** $\text{LReLU}(x) = \max(\alpha x, x)$ with $\alpha \approx 0.01$, derivative $\alpha$ for $x < 0$.

### 8.4 GELU

The Gaussian Error Linear Unit:

$$\text{GELU}(x) = x \cdot \Phi(x), \qquad \Phi(x) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Derivative** (product rule):

$$\text{GELU}'(x) = \Phi(x) + x \cdot \phi(x), \qquad \phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$$

where $\phi$ is the standard normal PDF. GELU is smooth everywhere, approximating ReLU for large positive $x$ and gating negative values by their Gaussian probability. Used in GPT-2, BERT, and most modern transformers.

### 8.5 Derivative Summary Table

| Activation | Formula | Derivative | Range of $f'$ | Used In |
|-----------|---------|-----------|--------------|---------|
| Sigmoid $\sigma$ | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ | $(0, 1/4]$ | Logistic regression, old RNNs |
| Tanh | $(e^x-e^{-x})/(e^x+e^{-x})$ | $1-\tanh^2$ | $(0,1]$ | LSTMs, NLP |
| ReLU | $\max(0,x)$ | $\mathbf{1}[x>0]$ | $\{0,1\}$ | CNNs, ResNets |
| GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ | $\approx(−0.17,1]$ | Transformers |
| Softplus | $\ln(1+e^x)$ | $\sigma(x)$ | $(0,1)$ | Smooth ReLU |


---

## 9. Numerical Differentiation

> **Recall:** §01 previewed the gradient as a limit in [§01 §11](../01-Limits-and-Continuity/notes.md). Here we develop the full numerical treatment — the canonical home for this topic.

### 9.1 Finite Difference Approximations

**Forward difference (first order, $O(h)$ error):**

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

Taylor expansion: $f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \cdots$, so the error is $\frac{h}{2}f''(x) + O(h^2)$.

**Centered difference (second order, $O(h^2)$ error):**

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

Expand both: odd-order error terms cancel, leaving error $\frac{h^2}{6}f'''(x) + O(h^4)$. Centered differences are strongly preferred.

**Second derivative via finite differences:**

$$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$

### 9.2 Optimal Step Size

Two competing errors:
- **Truncation error**: $O(h^2)$ — decreases as $h \to 0$
- **Round-off error**: floating-point arithmetic contributes $\varepsilon_{\text{mach}}/h$ — increases as $h \to 0$

Balancing these (minimize total error $h^2 + \varepsilon_{\text{mach}}/h$) gives optimal:

$$h^* \approx \varepsilon_{\text{mach}}^{1/3} \approx 10^{-5.3} \quad (\text{for float64, } \varepsilon_{\text{mach}} \approx 10^{-16})$$

In practice: $h \approx 10^{-5}$ for float64 centered differences.

### 9.3 Gradient Checking

**Gradient checking** verifies an analytical gradient implementation by comparing it to a numerical estimate:

```python
def grad_check(f, x, analytic_grad, h=1e-5):
    numerical_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy(); x_plus[i] += h
        x_minus = x.copy(); x_minus[i] -= h
        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    rel_error = np.linalg.norm(analytic_grad - numerical_grad) / \
                (np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad) + 1e-8)
    return rel_error  # < 1e-5 typically acceptable
```

Libraries like PyTorch provide `torch.autograd.gradcheck()` implementing exactly this. Every custom backward pass should be verified.

### 9.4 Automatic Differentiation

**Numerical differentiation** approximates derivatives; **automatic differentiation** (AD) computes them exactly (up to floating-point) by applying the chain rule to elementary operations in forward or reverse mode.

- **Forward mode AD**: accumulates derivative alongside function evaluation — efficient for few inputs, many outputs
- **Reverse mode AD** (backpropagation): accumulates gradient from output backward — efficient for many inputs, one output (the loss)

All modern deep learning frameworks (PyTorch, JAX, TensorFlow) implement reverse-mode AD. Numerical differentiation is used for checking, not training.


---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | $(fg)' = f'g'$ | Product rule is $(fg)' = f'g + fg'$ | Always use product rule |
| 2 | $(f/g)' = f'/g'$ | Quotient rule is $(f'g - fg')/g^2$ | Derive from quotient rule |
| 3 | Forgetting chain rule: $(e^{x^2})' = e^{x^2}$ | Outer function derivative × inner derivative needed | $(e^{x^2})' = 2x\,e^{x^2}$ |
| 4 | $(\ln x)' = 1/\ln x$ | Confused with change-of-base; correct is $1/x$ | Memorize: $(\ln x)' = 1/x$ |
| 5 | $(\sin^2 x)' = 2\sin x$ | Missing chain rule: multiply by $(\sin x)' = \cos x$ | $(\sin^2 x)' = 2\sin x\cos x = \sin 2x$ |
| 6 | Differentiable implies continuous — false direction | True direction: differentiable → continuous (not converse) | ReLU is continuous but not differentiable at 0 |
| 7 | $f''(x_0) = 0$ means saddle/inflection | Not sufficient — need sign change of $f''$ | Check $f''$ sign on both sides |
| 8 | Optimal $h$ for finite differences is $h = 10^{-16}$ | Catastrophic cancellation — round-off dominates | Use $h \approx 10^{-5}$ for float64 |
| 9 | Applying L'Hôpital when limit is not $0/0$ or $\infty/\infty$ | L'Hôpital only applies to indeterminate forms | Check form first |
| 10 | Chain rule direction error: $(f \circ g)' = f' \circ g'$ | Must evaluate $f'$ at $g(x)$: $(f\circ g)'(x) = f'(g(x))g'(x)$ | Track composition carefully |
| 11 | Assuming $\text{ReLU}'(0) = 1$ or $0.5$ | ReLU is not differentiable at 0; subgradient $0$ is standard | Use subgradient convention |
| 12 | Gradient of $\|Ax - b\|^2$ as $2A(Ax-b)$ | Matrix derivative rules needed; correct form is $2A^\top(Ax-b)$ | See [§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md) |

---

## 11. Exercises

**Exercise 1 ★ — Basic rules.** Differentiate: (a) $f(x) = 3x^4 - 2x^2 + 7$, (b) $g(x) = \sqrt{x}\,e^x$, (c) $h(x) = \frac{x^2+1}{x-1}$.

**Exercise 2 ★ — Chain rule.** Differentiate: (a) $\sin(x^3)$, (b) $\ln(\cos x)$, (c) $(1+x^2)^{10}$, (d) $e^{\sin x}$.

**Exercise 3 ★ — Implicit differentiation.** Given $x^3 + y^3 = 6xy$ (folium of Descartes), find $dy/dx$.

**Exercise 4 ★★ — Activation derivatives.** (a) Derive $\sigma'(x) = \sigma(x)(1-\sigma(x))$ from scratch. (b) Verify numerically using centered differences at $x \in \{-2, 0, 2\}$ with $h = 10^{-5}$.

**Exercise 5 ★★ — Critical points.** Find and classify all critical points of $f(x) = x^4 - 4x^3 + 6x^2$. Identify global extrema on $[-1, 4]$.

**Exercise 6 ★★ — MVT application.** Prove that $|\sin a - \sin b| \leq |a - b|$ for all $a, b \in \mathbb{R}$ using the MVT.

**Exercise 7 ★★★ — Gradient checking.** Implement centered finite differences to approximate the derivative of $f(x) = x\ln(x + e^{-x})$ at $x = 1$. Compare to the analytic derivative. Experiment with $h \in \{10^{-1}, 10^{-5}, 10^{-10}, 10^{-15}\}$ and explain the behavior.

**Exercise 8 ★★★ — Backpropagation.** A 2-layer scalar network: $z_1 = w_1 x$, $a_1 = \sigma(z_1)$, $z_2 = w_2 a_1$, $\hat{y} = z_2$, $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$. Derive $\partial\mathcal{L}/\partial w_1$ and $\partial\mathcal{L}/\partial w_2$ using the chain rule. Verify numerically.


---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---------|-------------|
| Derivative definition | Formal basis of gradient — every optimizer (SGD, Adam, Adagrad) descends along $-f'(\theta)$ |
| Chain rule | Backpropagation: gradient of loss through composed layers; foundational for all neural network training |
| Product/quotient rules | Attention score derivatives; gating mechanisms in LSTMs and GRUs |
| Activation derivatives | $\sigma'$, ReLU', GELU' determine gradient magnitude at each neuron; control vanishing/exploding gradients |
| Second derivative / concavity | Hessian approximations in Adam (diagonal), K-FAC (block), Shampoo (full); Newton's method in optimization |
| Critical points | Saddle points dominate high-dimensional loss landscapes; flat minima → better generalization |
| MVT and Lipschitz bounds | Learning rate theory: $\eta \leq 1/L$ ensures descent for $L$-smooth losses |
| Implicit differentiation | Normalizing flows, invertible networks, implicit models (DEQ, neural ODEs) |
| Numerical differentiation | Gradient checking in unit tests; finite-difference Hessian in small-scale studies |
| Automatic differentiation | PyTorch `autograd`, JAX `grad`, TensorFlow `GradientTape` — all implement reverse-mode AD |
| Linear approximation | Gradient descent step as linear model of the loss surface near current parameters |
| GELU derivative | Used in GPT-2, BERT, T5, and effectively every transformer post-2018 |

---

## Conceptual Bridge

**Looking back.** This section built on the definition of a limit ([§01](../01-Limits-and-Continuity/notes.md)) — the derivative is a limit of a difference quotient. The continuity of activation functions established in §01 is a prerequisite for their differentiability here (differentiable implies continuous, but not conversely).

**Looking forward.** The chain rule derived here for scalar functions becomes the multivariable chain rule in [§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md) — the chain rule expressed through Jacobians is the mathematical heart of backpropagation for vector-valued functions. The linear approximation introduced in §5.3 becomes the full Taylor series in [§04-Series-and-Sequences](../04-Series-and-Sequences/notes.md), which justifies optimizer update rules. Critical point analysis extends to gradient = 0 conditions in [§08-Optimization](../../08-Optimization/README.md).

**Position in the curriculum:**

```
CHAPTER 4 — CALCULUS FUNDAMENTALS
════════════════════════════════════════════════════════════════════════

  §01 Limits and Continuity ──────────────────────────────────────────
         │  (limit definition, L'Hôpital, continuity of activations)
         ▼
  §02 Derivatives and Differentiation  ◄─── YOU ARE HERE ────────────
         │  (rules, chain rule, activation f', numerical diff)
         ▼
  §03 Integration ────────────────────────────────────────────────────
         │  (FTC links derivatives ↔ integrals)
         ▼
  §04 Series and Sequences ───────────────────────────────────────────
         │  (Taylor series uses all derivatives from §02)
         ▼
  §05 Multivariate Calculus ──────────────────────────────────────────
         (multivariable chain rule, Jacobians, full backprop)

════════════════════════════════════════════════════════════════════════
```

---

[← Back to Calculus Fundamentals](../README.md) | [Next: Integration →](../03-Integration/notes.md)

---

## Appendix A: Extended Worked Examples — Basic Rules

### A.1 Power Rule — Negative and Fractional Exponents

The power rule $\frac{d}{dx}x^n = nx^{n-1}$ applies to all real $n$:

**Negative integer:** $\frac{d}{dx}x^{-3} = -3x^{-4} = -3/x^4$.

**Fractional:** $\frac{d}{dx}x^{2/3} = \frac{2}{3}x^{-1/3}$. Note: not defined at $x = 0$.

**Irrational:** $\frac{d}{dx}x^\pi = \pi x^{\pi-1}$.

**Proof for $n \in \mathbb{Z}^+$ (inductive, using product rule).** Base case $n=1$: $(x)' = 1$ from definition. Inductive step: assume $(x^k)' = kx^{k-1}$. Then $(x^{k+1})' = (x \cdot x^k)' = 1 \cdot x^k + x \cdot kx^{k-1} = (k+1)x^k$. $\square$

For general real $n$: write $x^n = e^{n\ln x}$ and apply chain rule: $\frac{d}{dx}e^{n\ln x} = e^{n\ln x} \cdot \frac{n}{x} = x^n \cdot \frac{n}{x} = nx^{n-1}$.

### A.2 Product Rule — Triple Product

For three functions $(uvw)' = u'vw + uv'w + uvw'$.

**Proof.** Write $uvw = (uv)w$ and apply product rule twice:
$(uv)'w + (uv)w' = (u'v + uv')w + uvw' = u'vw + uv'w + uvw'$.

**Example.** $(x^2 \sin x \ln x)' = 2x\sin x\ln x + x^2\cos x\ln x + x^2\sin x \cdot \frac{1}{x}$.

### A.3 Quotient Rule — Deriving from Product Rule

$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$

**Proof.** Let $h = f/g$, so $f = gh$. By product rule: $f' = g'h + gh'$. Solving: $h' = (f' - g'h)/g = (f' - g'f/g)/g = (f'g - fg')/g^2$. $\square$

**Mnemonic** ("lo d-hi minus hi d-lo over lo-lo"): $\frac{d}{dx}\frac{\text{hi}}{\text{lo}} = \frac{\text{lo}\cdot d(\text{hi}) - \text{hi}\cdot d(\text{lo})}{\text{lo}^2}$.

### A.4 Logarithmic Differentiation

When $f(x)$ is a product/quotient of many terms, take $\ln$ first:

$$y = \frac{x^3\sqrt{x+1}}{(x^2+2)^4} \implies \ln y = 3\ln x + \tfrac{1}{2}\ln(x+1) - 4\ln(x^2+2)$$

Differentiate implicitly:
$$\frac{y'}{y} = \frac{3}{x} + \frac{1}{2(x+1)} - \frac{8x}{x^2+2}$$

Multiply by $y$. Logarithmic differentiation is also the only elementary route to $y = x^x$:
$$\ln y = x\ln x \implies \frac{y'}{y} = \ln x + 1 \implies y' = x^x(\ln x + 1)$$


---

## Appendix B: Extended Chain Rule Examples and Patterns

### B.1 Identifying the Composition Structure

Every application of the chain rule begins with identifying:
- **Outer function** $f$: what is applied last
- **Inner function** $g$: what is the argument of the outer function

| Expression | Outer $f(u)$ | Inner $u = g(x)$ | Derivative |
|-----------|-------------|-----------------|-----------|
| $e^{-x^2}$ | $e^u$ | $-x^2$ | $-2x\,e^{-x^2}$ |
| $\sin^3 x$ | $u^3$ | $\sin x$ | $3\sin^2 x \cos x$ |
| $\ln(x^2+1)$ | $\ln u$ | $x^2+1$ | $2x/(x^2+1)$ |
| $\sqrt{\tan x}$ | $\sqrt{u}$ | $\tan x$ | $\sec^2 x / (2\sqrt{\tan x})$ |
| $(1+e^x)^{-1}$ | $u^{-1}$ | $1+e^x$ | $-e^x/(1+e^x)^2$ |

Note: $(1+e^x)^{-1} = \sigma(-x)$ — the sigmoid of $-x$.

### B.2 Nested Chain Rule — Four Levels Deep

$y = \sin(\ln(\sqrt{x^2+1}))$. Work from outside in:

Let $u_1 = \ln(\sqrt{x^2+1})$, $u_2 = \sqrt{x^2+1}$, $u_3 = x^2+1$.

$$y' = \cos(u_1) \cdot \frac{1}{u_2} \cdot \frac{1}{2\sqrt{u_3}} \cdot 2x = \cos(\ln\sqrt{x^2+1}) \cdot \frac{x}{(x^2+1)}$$

### B.3 Scalar Backpropagation Step-by-Step

Consider the output node $\mathcal{L} = -\log\sigma(z)$ where $z = \mathbf{w}^\top\mathbf{x}$ (binary cross-entropy, scalar model).

**Chain rule cascade:**
$$\frac{d\mathcal{L}}{dz} = \frac{d\mathcal{L}}{d\sigma}\cdot\frac{d\sigma}{dz} = -\frac{1}{\sigma(z)}\cdot\sigma(z)(1-\sigma(z)) = -(1-\sigma(z)) = \sigma(z) - 1$$

**Gradient w.r.t. weight $w_j$:**
$$\frac{d\mathcal{L}}{dw_j} = \frac{d\mathcal{L}}{dz}\cdot\frac{dz}{dw_j} = (\sigma(z)-1)\cdot x_j$$

This is the gradient used in logistic regression's SGD update. The chain rule reduces the gradient to a simple product of the prediction error $(\sigma(z)-1)$ and the input feature $x_j$.

### B.4 The Derivative as a Linear Map

Abstractly, the derivative $f'(a)$ is the **best linear approximation** to $f$ near $a$. More precisely, $f'(a)$ is the unique number such that:

$$f(a + h) = f(a) + f'(a)\,h + o(h) \quad \text{as } h \to 0$$

where $o(h)$ denotes terms that shrink faster than $h$. This abstract definition extends directly to vector functions (Fréchet derivative) — in that setting, $f'(a)$ becomes a linear map (matrix), which is the Jacobian.


---

## Appendix C: Deep Dive — Differentiability and Regularity

### C.1 The Formal ε-δ Definition of Differentiability

$f$ is **differentiable at $a$** if there exists a number $L$ such that:

$$\lim_{h \to 0} \frac{f(a+h) - f(a)}{h} = L$$

i.e., for all $\varepsilon > 0$, there exists $\delta > 0$ such that $0 < |h| < \delta$ implies:
$$\left|\frac{f(a+h)-f(a)}{h} - L\right| < \varepsilon$$

We write $f'(a) = L$.

### C.2 Non-Differentiable Points — A Taxonomy

**Type 1 — Corner:** $f(x) = |x|$. Left derivative: $-1$. Right derivative: $+1$. Limit does not exist.

**Type 2 — Cusp:** $f(x) = x^{2/3}$. At $x = 0$: $f'(x) = \frac{2}{3}x^{-1/3} \to \pm\infty$. The tangent line becomes vertical — the derivative limit is $\pm\infty$, not finite.

**Type 3 — Vertical tangent:** Same as cusp; the function is continuous but the difference quotient diverges.

**Type 4 — Oscillatory discontinuity of derivative:** $f(x) = x^2 \sin(1/x)$ for $x \neq 0$, $f(0) = 0$. The derivative exists at 0 ($f'(0) = 0$) but $f'(x)$ is not continuous at 0 — $f$ is differentiable but not $C^1$.

**For AI — ReLU.** ReLU has a corner at $x = 0$: left derivative is $0$, right derivative is $1$. In practice, the subgradient $\partial \text{ReLU}(0) = [0,1]$ (any value in the interval) is used; most frameworks return $0$.

### C.3 Smoothness Classes

| Class | Condition | Examples |
|-------|-----------|---------|
| $C^0$ | Continuous | ReLU, $|x|$ |
| $C^1$ | Derivative exists and is continuous | Softplus, GELU |
| $C^2$ | Second derivative exists and is continuous | $e^x$, $\sin x$, polynomials |
| $C^\infty$ | All derivatives exist | $e^x$, $\sin x$, $\cos x$ |
| $C^\omega$ | Real analytic (equals its Taylor series) | $e^x$, $\sin x$ |

**For AI:** Smooth activation functions ($C^\infty$) are required for second-order optimizers (Newton, L-BFGS) and for second-order sensitivity analysis. Rectified activations ($C^0$ only) work for SGD/Adam but complicate curvature estimation.

### C.4 Differentiability vs. Continuity — Summary

```
Differentiable at x₀
         │
         ▼ (implication)
Continuous at x₀
         │
         ✗ (no implication)
         ▼
Differentiable at x₀
```

- **Differentiable → Continuous**: Proven via $\lim_{h\to 0}[f(a+h)-f(a)] = \lim_{h\to 0}\frac{f(a+h)-f(a)}{h}\cdot h = f'(a)\cdot 0 = 0$.
- **Continuous ↛ Differentiable**: Counterexample: $|x|$ is continuous everywhere but not differentiable at 0.
- **Differentiable ↛ Continuously Differentiable**: Counterexample: $x^2\sin(1/x)$ above.


---

## Appendix D: Advanced Topics — Inverse Function Theorem and Derivatives of Inverses

### D.1 Inverse Function Theorem

If $f$ is differentiable at $a$ and $f'(a) \neq 0$, then $f^{-1}$ is differentiable at $b = f(a)$ and:

$$(f^{-1})'(b) = \frac{1}{f'(f^{-1}(b))} = \frac{1}{f'(a)}$$

**Intuition.** If $f$ multiplies tangent slopes by $f'(a)$, then $f^{-1}$ divides by $f'(a)$. In Leibniz notation: if $y = f(x)$, then $dx/dy = 1/(dy/dx)$.

**Geometric meaning.** The graph of $f^{-1}$ is the reflection of the graph of $f$ over the line $y = x$. The slope of the tangent to $f$ at $(a, f(a))$ is $f'(a)$; the slope of the reflected tangent to $f^{-1}$ at $(f(a), a)$ is $1/f'(a)$.

### D.2 Derivatives of Inverse Trigonometric Functions

Using the inverse function theorem and implicit differentiation:

**$\arcsin$:** Let $y = \arcsin x$, so $\sin y = x$. Differentiate: $\cos y \cdot y' = 1$. Since $\cos y = \sqrt{1-\sin^2 y} = \sqrt{1-x^2}$:

$$(\arcsin x)' = \frac{1}{\sqrt{1-x^2}}, \quad x \in (-1,1)$$

**$\arccos$:** Similarly, $(\arccos x)' = -1/\sqrt{1-x^2}$.

**$\arctan$:** Let $y = \arctan x$, so $\tan y = x$. Differentiate: $\sec^2 y \cdot y' = 1$. Since $\sec^2 y = 1 + \tan^2 y = 1 + x^2$:

$$(\arctan x)' = \frac{1}{1+x^2}$$

**For AI.** The softplus function $\text{softplus}(x) = \ln(1+e^x)$ is the antiderivative of sigmoid: $\text{softplus}'(x) = \sigma(x)$. This connects the inverse-function perspective to activation function design.

### D.3 Generalized Power Rule via Logarithmic Differentiation

For $y = [f(x)]^{g(x)}$ (variable base, variable exponent):

$$\ln y = g(x)\ln f(x) \implies \frac{y'}{y} = g'(x)\ln f(x) + g(x)\frac{f'(x)}{f(x)}$$

$$y' = [f(x)]^{g(x)}\left[g'(x)\ln f(x) + \frac{g(x)f'(x)}{f(x)}\right]$$

**Special cases:**
- $g(x) = n$ (constant): gives power rule $y' = nf^{n-1}f'$
- $f(x) = e$ (base $e$): gives exponential derivative $y' = e^{g(x)}g'(x)$


---

## Appendix E: Optimization Theory Foundations

### E.1 First and Second Derivative Tests — Full Treatment

**First Derivative Test (complete statement):**

Let $f$ be continuous on $(a,b)$ and $x_0 \in (a,b)$ a critical point.

- If $f'(x) > 0$ for $x \in (a,x_0)$ and $f'(x) < 0$ for $x \in (x_0, b)$: **local maximum** at $x_0$.
- If $f'(x) < 0$ for $x \in (a,x_0)$ and $f'(x) > 0$ for $x \in (x_0, b)$: **local minimum** at $x_0$.
- If $f'$ does not change sign: **neither** (saddle or inflection).

**Second Derivative Test:**

If $f'(x_0) = 0$:
- $f''(x_0) > 0$ → local min (concave up)
- $f''(x_0) < 0$ → local max (concave down)
- $f''(x_0) = 0$ → inconclusive; examine higher-order derivatives or use first derivative test

**Inconclusive case — higher-order test.** If the first $k-1$ derivatives vanish at $x_0$ and $f^{(k)}(x_0) \neq 0$:
- $k$ odd: neither max nor min (inflection-type)
- $k$ even, $f^{(k)}(x_0) > 0$: local min
- $k$ even, $f^{(k)}(x_0) < 0$: local max

**Example.** $f(x) = x^4$ at $x_0 = 0$: $f'(0) = f''(0) = f'''(0) = 0$, $f^{(4)}(0) = 24 > 0$, $k = 4$ even → local (and global) min.

### E.2 Rolle's Theorem

**Theorem.** If $f$ is continuous on $[a,b]$, differentiable on $(a,b)$, and $f(a) = f(b)$, then there exists $c \in (a,b)$ with $f'(c) = 0$.

**Proof.** By EVT, $f$ attains its maximum and minimum on $[a,b]$. If both are attained at endpoints, then $f$ is constant and $f' \equiv 0$. Otherwise, an interior maximum or minimum $c$ exists; at an interior extremum, $f'(c) = 0$ (since $f'(c) \neq 0$ would imply $f$ is locally monotone, contradicting the extremum). $\square$

Rolle's Theorem is the key lemma for the MVT.

### E.3 Gradient Descent — Single Variable View

The gradient descent update for minimizing $f(\theta)$:

$$\theta_{t+1} = \theta_t - \eta f'(\theta_t)$$

**Convergence (quadratic $f$).** Let $f(\theta) = \frac{1}{2}L\theta^2$ (1D version of $L$-smooth quadratic). Then:

$$f(\theta_{t+1}) = \frac{1}{2}L(\theta_t - \eta L\theta_t)^2 = (1-\eta L)^2 f(\theta_t)$$

Converges if $|1 - \eta L| < 1$, i.e., $0 < \eta < 2/L$. Optimal $\eta = 1/L$ gives one-step convergence: $f(\theta_1) = 0$.

**For general smooth $f$ (Taylor + MVT argument):**

If $|f''| \leq L$ everywhere (Lipschitz gradient), then:
$$f(\theta - \eta f'(\theta)) \leq f(\theta) - \eta\left(1 - \frac{\eta L}{2}\right)|f'(\theta)|^2$$

The right side decreases as long as $\eta < 2/L$ and $f'(\theta) \neq 0$ — guaranteeing descent at each step.


---

## Appendix F: Numerical Differentiation — Advanced Analysis

### F.1 Error Analysis in Detail

**Forward difference error.** Taylor expand $f(x+h)$:

$$f(x+h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + \frac{h^3}{6}f'''(x) + \cdots$$

Solve for $f'(x)$:

$$f'(x) = \frac{f(x+h) - f(x)}{h} - \frac{h}{2}f''(x) - \frac{h^2}{6}f'''(x) - \cdots$$

The leading error term is $-\frac{h}{2}f''(x)$, so the **truncation error is $O(h)$**.

**Centered difference error.** Also expand $f(x-h)$:

$$f(x-h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) - \frac{h^3}{6}f'''(x) + \cdots$$

Subtract: $f(x+h) - f(x-h) = 2hf'(x) + \frac{h^3}{3}f'''(x) + \cdots$

$$f'(x) = \frac{f(x+h) - f(x-h)}{2h} - \frac{h^2}{6}f'''(x) - \cdots$$

The leading error is $-\frac{h^2}{6}f'''(x)$, so the **truncation error is $O(h^2)$** — one order better.

**Round-off error.** In floating-point arithmetic, $f(x+h)$ is computed with absolute error $\sim \varepsilon_{\text{mach}}|f(x)|$. The round-off contribution to the centered difference is:

$$\text{round-off error} \approx \frac{2\varepsilon_{\text{mach}}|f(x)|}{2h} = \frac{\varepsilon_{\text{mach}}|f(x)|}{h}$$

**Total error:** $E(h) \approx \frac{h^2}{6}|f'''(x)| + \frac{\varepsilon_{\text{mach}}|f(x)|}{h}$

Minimize over $h$: $dE/dh = \frac{h}{3}|f'''| - \frac{\varepsilon_{\text{mach}}|f|}{h^2} = 0$ gives:

$$h^* = \left(\frac{3\varepsilon_{\text{mach}}|f|}{|f'''|}\right)^{1/3} \approx \varepsilon_{\text{mach}}^{1/3}$$

For float64 ($\varepsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$): $h^* \approx 6 \times 10^{-6}$.

### F.2 Richardson Extrapolation

Given two centered-difference approximations at step sizes $h$ and $h/2$, eliminate the leading $O(h^2)$ error term:

$$f'(x) \approx \frac{4 \cdot D(h/2) - D(h)}{3}, \quad D(h) = \frac{f(x+h)-f(x-h)}{2h}$$

This Richardson-extrapolated formula has error $O(h^4)$ — a massive improvement. Recursive application yields **Romberg differentiation** with arbitrarily high accuracy.

### F.3 Complex-Step Derivative

An alternative that avoids catastrophic cancellation: compute $f$ at $x + ih$ (complex argument):

$$f'(x) \approx \frac{\text{Im}[f(x + ih)]}{h}$$

For analytic $f$, this has only round-off error $O(\varepsilon_{\text{mach}})$ with no subtraction cancellation, and allows very small $h \approx 10^{-20}$ without accuracy loss. Used in aerospace and scientific computing.


---

## Appendix G: Activation Functions — Extended Analysis

### G.1 Vanishing Gradient Problem — Quantitative Analysis

In a network with $L$ layers using sigmoid activations, the gradient at the first layer involves a product of $L$ sigmoid derivatives:

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial a_L} \cdot \prod_{l=2}^{L} \sigma'(z_l)$$

Since $\sigma'(z) \leq 1/4$ for all $z$, the product is bounded:

$$\left|\prod_{l=2}^{L}\sigma'(z_l)\right| \leq \left(\frac{1}{4}\right)^{L-1}$$

For a 10-layer network: $\leq (1/4)^9 \approx 4 \times 10^{-6}$ — the gradient is effectively zero. This is the **vanishing gradient problem** that made training deep sigmoid networks impractical before ReLU.

**ReLU advantage.** For active ReLU units ($z > 0$): $\text{ReLU}'(z) = 1$. The product of $L$ active ReLU derivatives is exactly 1 — no decay. This is why deep residual networks with ReLU can be trained with hundreds of layers.

### G.2 Exploding Gradients

The opposite pathology: if weights are initialized too large, the product $\prod_l f'_l(z_l)$ can exceed 1 exponentially in $L$. Gradients explode, leading to NaN loss.

**Mitigations:** gradient clipping ($\|g\| \leq c$), weight initialization (Xavier/He), batch normalization, residual connections.

### G.3 Softmax Derivative

The softmax function maps $\mathbf{z} \in \mathbb{R}^K$ to a probability vector:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} = p_i$$

**Partial derivatives:**

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1-p_i) & i = j \\ -p_ip_j & i \neq j \end{cases}$$

The **Jacobian** of softmax is $J_{ij} = p_i(\delta_{ij} - p_j)$, or in matrix form: $J = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.

**For backprop.** Combined with cross-entropy loss $\mathcal{L} = -\sum_i y_i\log p_i$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = p_j - y_j$$

The gradient is simply the difference between the predicted probability and the true label — remarkably clean due to the cancellation between softmax and cross-entropy.

### G.4 GELU vs ReLU — Gradient Behavior

| Property | ReLU | GELU |
|----------|------|------|
| Differentiable | No (at 0) | Yes ($C^\infty$) |
| Dead neurons | Yes (gradient = 0 for $x < 0$ always) | No (small gradient for $x < 0$) |
| Gradient at $x=0$ | $0$ (subgradient) | $0.5$ (smooth) |
| Computation | Trivial ($\max$) | Requires erf evaluation |
| Used in | ResNets, older models | GPT, BERT, modern transformers |

The GELU's smooth gradient in the negative regime allows small updates even for negative pre-activations, which may contribute to the empirical finding that GELU outperforms ReLU on large-scale language tasks.


---

## Appendix H: Historical Context and Mathematical Development

### H.1 The Calculus War

The derivative was developed independently in the 1660s-1680s by Newton and Leibniz — one of the most famous priority disputes in mathematics.

| Year | Newton | Leibniz |
|------|--------|---------|
| 1666 | Develops "method of fluxions" (flux = rate of change) | — |
| 1675 | — | Introduces $dy/dx$ notation |
| 1684 | — | Publishes first calculus paper |
| 1687 | Publishes *Principia* (uses fluents, not derivatives explicitly) | — |
| 1693 | — | Publishes $d/dx$ systematically |
| 1704 | Newton publishes "Of the Quadrature of Curves" | — |
| 1711 | Royal Society initiates plagiarism inquiry | — |

Modern consensus: independent discovery. Newton's notation ($\dot{x}$ for time derivatives) persists in physics; Leibniz's $dy/dx$ dominates mathematics and engineering.

### H.2 Rigorous Foundations — 19th Century

The limit-based definition of the derivative was not rigorously formulated until the 1820s:

- **Cauchy (1821):** Formalized limits and gave the first rigorous definition of the derivative as a limit of a difference quotient.
- **Weierstrass (1872):** Constructed a continuous-everywhere, differentiable-nowhere function — proving that "most" continuous functions are not differentiable.
- **Riemann (1854):** Formalized integration; the FTC was proved rigorously.

### H.3 Connection to Modern AI

| Year | Development | Calculus Foundation |
|------|-------------|-------------------|
| 1986 | Rumelhart et al. popularize backpropagation | Chain rule on computation graphs |
| 1989 | LeCun's convolutional backprop | Chain rule through spatial filters |
| 2012 | AlexNet — deep ReLU networks | Avoids vanishing gradient (§8.1) |
| 2014 | Adam optimizer | Adaptive learning rates, second-moment gradient estimate |
| 2017 | Transformer attention | Softmax Jacobian (§G.3) in attention scores |
| 2018 | BERT (GELU activation) | Smooth gradient, $C^\infty$ activations |
| 2022+ | JAX's functional AD | Efficient forward/reverse mode composition |
| 2023+ | LoRA / DoRA (PEFT) | Low-rank gradient updates — rank of Jacobians |
| 2024+ | Mechanistic interpretability | Gradient attribution to individual features |


---

## Appendix I: Applications in Machine Learning — Deep Dives

### I.1 Gradient Descent and Its Variants

All first-order optimization algorithms are based on the derivative. The basic update:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

In the scalar case ($\theta \in \mathbb{R}$): $\theta_{t+1} = \theta_t - \eta \mathcal{L}'(\theta_t)$.

**Momentum (Polyak 1964):**

$$v_{t+1} = \beta v_t + \mathcal{L}'(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}$$

Accumulates a running average of gradients, dampening oscillations in high-curvature directions.

**Adam (Kingma & Ba, 2015):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\mathcal{L}'(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)[\mathcal{L}'(\theta_t)]^2$$
$$\hat{m}_t = m_t/(1-\beta_1^t), \quad \hat{v}_t = v_t/(1-\beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta\,\hat{m}_t/(\sqrt{\hat{v}_t}+\varepsilon)$$

$m_t$ is a first moment (mean) estimate; $v_t$ is a second moment (uncentered variance) estimate. Dividing by $\sqrt{v_t}$ adapts the step size per parameter: large gradient → small effective step; small gradient → larger effective step. This diagonal approximation to the inverse curvature is why Adam often outperforms vanilla SGD.

### I.2 Automatic Differentiation — Under the Hood

**Forward mode.** Alongside each function value $f(x)$, propagate the derivative $f'(x)$ using dual numbers $(a, b)$ where arithmetic follows $(a_1, b_1) \cdot (a_2, b_2) = (a_1 a_2, a_1 b_2 + a_2 b_1)$ (product rule built in).

**Reverse mode.** In the forward pass, record a computation graph (tape). In the backward pass, traverse the graph in reverse, accumulating partial derivatives using the chain rule at each node.

**Cost comparison:**
- Forward mode: $O(n)$ passes for $n$ inputs → $O(n)$ for scalar output, $O(nm)$ for $m$ outputs
- Reverse mode: one backward pass → $O(1)$ for scalar output regardless of $n$

For neural network training ($n$ = millions of parameters, scalar loss $\mathcal{L}$): reverse mode is $O(1)$ passes vs. $O(n)$ for forward mode — the reason backpropagation is computationally feasible.

### I.3 Derivative-Based Interpretability

**Saliency maps.** For an image classification model $f$, the gradient $\partial f(x)/\partial x_i$ measures how sensitive the output is to pixel $i$ — used as a first-order feature importance score.

**Input × Gradient.** $x_i \cdot \partial f/\partial x_i$ — a refinement that accounts for the magnitude of the feature, not just the sensitivity.

**Integrated Gradients (Sundararajan et al. 2017).** Computes the path integral of the gradient from a baseline to the input:

$$\text{IG}_i(x) = (x_i - x_i^0)\int_0^1 \frac{\partial f(x^0 + t(x-x^0))}{\partial x_i}\,dt$$

Satisfies sensitivity and completeness axioms — considered one of the most principled gradient-based attribution methods.


---

## Appendix J: Extended Exercises and Practice Problems

### J.1 Warm-Up Drill — Compute Derivatives

Differentiate each function (answers at end of section):

1. $f(x) = x^5 - 3x^3 + 2x - 7$
2. $f(x) = \frac{x^2 - 1}{x^2 + 1}$
3. $f(x) = (2x+1)^{10}$
4. $f(x) = x^2 e^{-x}$
5. $f(x) = \ln(\sin x)$
6. $f(x) = \arctan(e^x)$
7. $f(x) = x^{\sin x}$
8. $f(x) = \frac{\sqrt{x+1}}{x^2+3}$

**Answers:**
1. $5x^4 - 9x^2 + 2$
2. $\frac{4x}{(x^2+1)^2}$
3. $20(2x+1)^9$
4. $xe^{-x}(2-x)$
5. $\cot x$
6. $\frac{e^x}{1+e^{2x}}$
7. $x^{\sin x}\left(\cos x\ln x + \frac{\sin x}{x}\right)$
8. $\frac{(x^2+3)/(2\sqrt{x+1}) - 2x\sqrt{x+1}}{(x^2+3)^2}$

### J.2 Conceptual Questions

1. **True or False.** If $f'(c) = 0$, then $f$ has a local extremum at $c$. (False — consider $f(x) = x^3$ at $c=0$.)

2. **Proof.** Show that if $f$ and $g$ are differentiable, $(f \circ g)'(x) = f'(g(x))g'(x)$ implies that the derivative of the identity $f \circ f^{-1}(x) = x$ gives $(f^{-1})'(f(x)) = 1/f'(x)$.

3. **Explain.** Why does the gradient descent step $\eta = 2/L$ (with $L = \max|f''|$) lead to oscillatory behavior for a quadratic, while $\eta = 1/L$ gives smooth convergence?

4. **Derivation.** Starting from $\sigma(x) = 1/(1+e^{-x})$, derive the update rule for the gradient of binary cross-entropy loss $\mathcal{L} = -[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$ with respect to $z$.

   **Solution.** $\partial\mathcal{L}/\partial z = \sigma(z) - y$.

5. **Analysis.** Centered finite differences have $O(h^2)$ error, but why do software implementations (like `torch.autograd.gradcheck`) still use $h = 10^{-3}$ or even $h = 10^{-2}$ rather than the optimal $h^* \approx 10^{-5}$?

   **Answer.** With 32-bit floating point ($\varepsilon_{\text{mach}} \approx 10^{-7}$), the optimal step is $h^* \approx 10^{-7/3} \approx 10^{-2.3}$. The larger step avoids round-off without sacrificing much truncation accuracy in float32.

### J.3 Computational Exercises

**Exercise J.1.** Implement the forward, backward, and centered finite difference formulas for $f'(x)$. Plot the absolute error vs. $h$ for $f(x) = \sin(x)$ at $x = 1$, using $h \in [10^{-16}, 10^{-1}]$ on a log scale. Observe the optimal $h$ region.

**Exercise J.2.** Implement a simple scalar backpropagation for the function $f(x) = \sigma(\ln(x^2 + 1))$:
- Forward pass: compute $u_1 = x^2+1$, $u_2 = \ln u_1$, $y = \sigma(u_2)$
- Backward pass: compute $\partial y/\partial x$ analytically via chain rule
- Verify with centered finite differences

**Exercise J.3.** For the sigmoid function, plot: (a) $\sigma(x)$, (b) $\sigma'(x) = \sigma(1-\sigma)$, (c) $\sigma''(x)$. Identify: the inflection point of $\sigma$, the maximum of $\sigma'$, and the zeros of $\sigma''$.


---

## Appendix K: Notation Reference and Quick-Reference Tables

### K.1 Derivative Notation — Comparison

| Notation | Author | Reads as | Common usage |
|----------|--------|---------|-------------|
| $f'(x)$ | Lagrange | "$f$ prime of $x$" | Pure mathematics |
| $\frac{df}{dx}$ | Leibniz | "$df$ by $dx$" | Engineering, physics |
| $\dot{f}$ | Newton | "$f$ dot" | Physics (time derivatives) |
| $Df(x)$ | Euler/operator | "D of $f$" | Differential equations |
| $\partial f/\partial x$ | — | "partial $f$ partial $x$" | Multivariable calculus |
| $\nabla_x f$ | Hamilton | "nabla $f$" or "gradient" | Machine learning, optimization |

In this curriculum: $f'(x)$ for scalar single-variable, $\nabla_\theta \mathcal{L}$ for multi-parameter ML context.

### K.2 Differentiation Rules Quick Reference

| Rule | Formula | Notes |
|------|---------|-------|
| Constant | $(c)' = 0$ | |
| Identity | $(x)' = 1$ | |
| Power | $(x^n)' = nx^{n-1}$ | All real $n$ |
| Constant multiple | $(cf)' = cf'$ | |
| Sum | $(f+g)' = f' + g'$ | |
| Product | $(fg)' = f'g + fg'$ | |
| Quotient | $(f/g)' = (f'g-fg')/g^2$ | |
| Chain | $(f\circ g)'(x) = f'(g(x))g'(x)$ | |
| Exponential | $(e^x)' = e^x$ | |
| Natural log | $(\ln x)' = 1/x$ | $x > 0$ |
| Sine | $(\sin x)' = \cos x$ | |
| Cosine | $(\cos x)' = -\sin x$ | |
| Arctangent | $(\arctan x)' = 1/(1+x^2)$ | |
| Inverse | $(f^{-1})'(y) = 1/f'(f^{-1}(y))$ | |

### K.3 Second Derivative Test Quick Reference

| Condition | Classification |
|-----------|---------------|
| $f'(x_0)=0$, $f''(x_0)>0$ | Local minimum |
| $f'(x_0)=0$, $f''(x_0)<0$ | Local maximum |
| $f'(x_0)=0$, $f''(x_0)=0$ | Inconclusive |
| $f''(x_0)>0$ (any $x_0$) | Concave up |
| $f''(x_0)<0$ (any $x_0$) | Concave down |
| $f''$ changes sign at $x_0$ | Inflection point |

### K.4 Finite Difference Error Summary

| Method | Formula | Error order | Optimal $h$ (float64) |
|--------|---------|------------|----------------------|
| Forward | $[f(x+h)-f(x)]/h$ | $O(h)$ | $\approx 10^{-8}$ |
| Backward | $[f(x)-f(x-h)]/h$ | $O(h)$ | $\approx 10^{-8}$ |
| Centered | $[f(x+h)-f(x-h)]/(2h)$ | $O(h^2)$ | $\approx 10^{-5.3}$ |
| Second derivative | $[f(x+h)-2f(x)+f(x-h)]/h^2$ | $O(h^2)$ | $\approx 10^{-5}$ |
| Richardson | Combines two centered | $O(h^4)$ | larger $h$ acceptable |


---

## Appendix L: Implicit Differentiation — Extended Examples

### L.1 Folium of Descartes

The curve $x^3 + y^3 = 3axy$ (with $a = 1$) is the **folium of Descartes**, defined implicitly. Differentiating:

$$3x^2 + 3y^2\frac{dy}{dx} = 3y + 3x\frac{dy}{dx}$$

$$\frac{dy}{dx}(3y^2 - 3x) = 3y - 3x^2 \implies \frac{dy}{dx} = \frac{y - x^2}{y^2 - x}$$

Horizontal tangents: $dy/dx = 0 \Rightarrow y = x^2$. Substituting into $x^3 + y^3 = 3xy$:
$x^3 + x^6 = 3x^3 \Rightarrow x^6 - 2x^3 = 0 \Rightarrow x = 0$ or $x = 2^{1/3}$.

### L.2 Ellipse and Normal Lines

The ellipse $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$. Differentiating:

$$\frac{2x}{a^2} + \frac{2y}{b^2}\frac{dy}{dx} = 0 \implies \frac{dy}{dx} = -\frac{b^2 x}{a^2 y}$$

The **normal line** (perpendicular to tangent) at $(x_0, y_0)$ has slope $\frac{a^2 y_0}{b^2 x_0}$.

### L.3 Implicit Differentiation for Normalizing Flows

In normalizing flows, a bijective function $\mathbf{z} = f(\mathbf{x})$ is used. The change-of-variables formula for densities requires the determinant of the Jacobian:

$$p_X(\mathbf{x}) = p_Z(f(\mathbf{x})) \left|\det\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right|$$

When $f$ is defined implicitly via $g(\mathbf{x}, \mathbf{z}) = 0$ (e.g., in continuous normalizing flows via an ODE), the **implicit function theorem** guarantees the Jacobian exists and computes it as:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = -\left(\frac{\partial g}{\partial \mathbf{z}}\right)^{-1}\frac{\partial g}{\partial \mathbf{x}}$$

This is implicit differentiation generalized to the multivariable setting — a key tool in modern generative models (FFJORD, Neural ODEs).

---

## Appendix M: Mean Value Theorem — Consequences and Applications

### M.1 Monotonicity via Derivatives

**Theorem.** If $f'(x) > 0$ for all $x \in (a,b)$, then $f$ is strictly increasing on $[a,b]$.

**Proof.** Let $x_1 < x_2$ in $[a,b]$. By MVT, there exists $c \in (x_1, x_2)$ with:

$$f(x_2) - f(x_1) = f'(c)(x_2 - x_1) > 0$$

since $f'(c) > 0$ and $x_2 - x_1 > 0$. Thus $f(x_2) > f(x_1)$. $\square$

**Corollary.** $f' \equiv 0$ on $(a,b)$ iff $f$ is constant on $[a,b]$.

### M.2 Error Bounds via MVT

**Theorem (MVT error bound).** If $|f'(x)| \leq M$ for all $x \in [a,b]$, then:

$$|f(x) - f(y)| \leq M|x-y| \quad \text{for all } x, y \in [a,b]$$

This is the **Lipschitz condition** with constant $M$.

**Application — gradient descent learning rate.** If $|f''| \leq L$ (second derivative bounded), then $|f'(x) - f'(y)| \leq L|x-y|$ — the gradient is $L$-Lipschitz. The descent lemma states:

$$f\!\left(\theta - \frac{1}{L}f'(\theta)\right) \leq f(\theta) - \frac{1}{2L}|f'(\theta)|^2$$

ensuring a sufficient decrease of $\frac{1}{2L}|f'(\theta)|^2$ per step.

### M.3 Cauchy Mean Value Theorem

**Theorem.** If $f, g$ are continuous on $[a,b]$, differentiable on $(a,b)$, and $g'(x) \neq 0$, then:

$$\frac{f(b) - f(a)}{g(b) - g(a)} = \frac{f'(c)}{g'(c)}$$

for some $c \in (a,b)$.

**Application.** L'Hôpital's Rule (§01) follows from the Cauchy MVT: for $f(a) = g(a) = 0$,

$$\frac{f(x)}{g(x)} = \frac{f(x)-f(a)}{g(x)-g(a)} = \frac{f'(c)}{g'(c)} \xrightarrow{x \to a} \frac{f'(a)}{g'(a)}$$

provided the last limit exists.


---

## Appendix N: Parametric and Polar Curves

### N.1 Derivatives of Parametric Curves

A parametric curve $(x(t), y(t))$ defines $y$ as an implicit function of $x$ via the parameter $t$. The derivative:

$$\frac{dy}{dx} = \frac{dy/dt}{dx/dt} = \frac{y'(t)}{x'(t)}$$

provided $x'(t) \neq 0$.

**Second derivative:**

$$\frac{d^2y}{dx^2} = \frac{d}{dx}\left(\frac{dy}{dx}\right) = \frac{\frac{d}{dt}\left(\frac{y'(t)}{x'(t)}\right)}{x'(t)}$$

**Example.** Cycloid: $x = t - \sin t$, $y = 1 - \cos t$.

$dx/dt = 1-\cos t$, $dy/dt = \sin t$.

$$\frac{dy}{dx} = \frac{\sin t}{1-\cos t}$$

At $t = \pi/2$: $dy/dx = \frac{1}{1} = 1$ — 45° angle.
At $t \to 0$: $dy/dx \to \sin t/(1-\cos t) \approx t/(t^2/2) = 2/t \to \infty$ — vertical tangent.

### N.2 Applications to Attention Curves

In transformer attention, the attention weight for query-key pair $(q,k)$ is:

$$\alpha = \frac{e^{q \cdot k / \sqrt{d}}}{\sum_{j} e^{q \cdot k_j / \sqrt{d}}}$$

The **derivative of attention weight with respect to the scale factor** $\beta = 1/\sqrt{d}$:

$$\frac{d\alpha_i}{d\beta} = \alpha_i\left(q\cdot k_i - \sum_j \alpha_j\, q\cdot k_j\right) = \alpha_i\left(q\cdot k_i - \mathbb{E}_\alpha[q\cdot k]\right)$$

This is the derivative of a softmax composition — it shows that attention sharpens (moves weight to higher-scored keys) as $\beta$ increases, consistent with the "temperature" interpretation from §01.

---

## Appendix O: Numerically Stable Derivative Computations

### O.1 Sigmoid: Numerically Stable Computation

Naive $\sigma(x) = 1/(1+e^{-x})$ overflows for large negative $x$ ($e^{|x|}$ can be > `float64` max $\approx 10^{308}$ for $|x| > 710$).

**Stable implementation:**

```python
def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1 / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))
```

Both branches are equivalent; the first is used for $x \geq 0$ (where $e^{-x}$ is safe), the second for $x < 0$ (where $e^x$ is safe).

**Derivative — stable form:**

```python
def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)
```

No additional overflow risk once $\sigma(x)$ is stably computed.

### O.2 Log-Sigmoid: Avoiding Catastrophic Cancellation

For the log-sigmoid $\log\sigma(x) = -\log(1+e^{-x})$:
- For large positive $x$: $\log(1+e^{-x}) \approx e^{-x}$ — no cancellation.
- For large negative $x$: $\log(1+e^{-x}) \approx -x$ — use this directly.

**Stable:**
```python
def log_sigmoid(x):
    return -np.logaddexp(0, -x)  # = -log(e^0 + e^{-x})
```

`np.logaddexp(a, b)` computes $\log(e^a + e^b)$ stably by extracting the maximum:
$$\log(e^a + e^b) = \max(a,b) + \log(1 + e^{-(|a-b|)})$$

### O.3 Softmax: Subtract Maximum

Naive softmax overflows for large logits. Standard fix:

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}$$

Subtracting the maximum shifts all logits to $\leq 0$, making $e^{z_i - z_{\max}} \in (0,1]$. The softmax value is unchanged since the normalization cancels the factor $e^{-z_{\max}}$.

These numerical stability techniques are directly connected to the limits $\lim_{x\to\infty}(1+e^{-x})^{-1} = 1$ and $\lim_{x\to-\infty}e^x = 0$ from [§01](../01-Limits-and-Continuity/notes.md).


---

## Appendix P: Derivatives in Probability and Statistics

### P.1 Score Function and Fisher Information

In statistics, the **score function** is the derivative of the log-likelihood with respect to the parameter $\theta$:

$$s(\theta; x) = \frac{\partial}{\partial\theta}\log p(x;\theta)$$

**Properties:**
- $\mathbb{E}_\theta[s(\theta;X)] = 0$ (score has zero mean)
- $\text{Var}_\theta[s(\theta;X)] = \mathcal{I}(\theta)$ (Fisher information)

**Fisher information:**

$$\mathcal{I}(\theta) = \mathbb{E}\left[\left(\frac{\partial\log p(x;\theta)}{\partial\theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2\log p(x;\theta)}{\partial\theta^2}\right]$$

The second equality (Fisher's identity) relates the variance of the first derivative to the expected value of the second derivative — a deep connection between information geometry and calculus.

### P.2 KL Divergence Gradient

The KL divergence $\text{KL}(q\|p) = \int q(x)\log\frac{q(x)}{p(x)}dx$ appears throughout ML. Its derivative with respect to a parameter $\theta$ of $p(\cdot;\theta)$:

$$\frac{\partial}{\partial\theta}\text{KL}(q\|p_\theta) = -\int q(x)\frac{\partial}{\partial\theta}\log p(x;\theta)\,dx = -\mathbb{E}_q\left[\frac{\partial}{\partial\theta}\log p(x;\theta)\right]$$

This is the negative expected score under $q$. In variational inference (ELBO optimization), this derivative structure underlies the "reparameterization trick" for training VAEs.

### P.3 Maximum Likelihood — First Order Conditions

For i.i.d. data $\{x_1,\ldots,x_n\}$, MLE maximizes:

$$\hat{\theta} = \arg\max_\theta \sum_{i=1}^n \log p(x_i;\theta)$$

The first-order condition $\frac{\partial}{\partial\theta}\sum_i\log p(x_i;\theta) = 0$ defines the MLE. For the Gaussian with unknown mean: $\sum_i(x_i - \mu) = 0 \Rightarrow \hat{\mu} = \bar{x}$.

For neural networks, MLE is equivalent to minimizing cross-entropy loss — the derivative-based optimization (SGD, Adam) maximizes the likelihood of the training data.

---

## Appendix Q: Special Topics — Subgradients and Non-Smooth Analysis

### Q.1 Subgradients

For a convex function $f$ that may not be differentiable, a **subgradient** at $x_0$ is any $g \in \mathbb{R}$ satisfying:

$$f(x) \geq f(x_0) + g(x - x_0) \quad \text{for all } x$$

The **subdifferential** $\partial f(x_0)$ is the set of all subgradients.

**Examples:**
- $f(x) = |x|$: $\partial f(0) = [-1, 1]$; $\partial f(x) = \{\text{sign}(x)\}$ for $x \neq 0$.
- $f(x) = \text{ReLU}(x) = \max(0,x)$: $\partial f(0) = [0, 1]$; $\partial f(x) = \{0\}$ for $x<0$; $\partial f(x) = \{1\}$ for $x>0$.

### Q.2 Subgradient Descent

For non-smooth convex $f$, the subgradient descent update uses any $g_t \in \partial f(\theta_t)$:

$$\theta_{t+1} = \theta_t - \eta_t g_t$$

Unlike gradient descent for smooth functions, subgradient descent does NOT necessarily decrease $f$ at each step. Convergence requires diminishing step sizes: $\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$ (Robbins-Monro conditions from §01).

### Q.3 Proximal Operators

For non-smooth regularization (L1 sparsity), instead of subgradient descent, the **proximal gradient** method is often used. The proximal operator of $h$ at $v$ with step size $\eta$:

$$\text{prox}_{\eta h}(v) = \arg\min_x\left[h(x) + \frac{1}{2\eta}\|x-v\|^2\right]$$

For $h(x) = \lambda|x|$: $\text{prox}_{\eta h}(v) = \text{sign}(v)\max(|v|-\eta\lambda, 0)$ — the **soft-thresholding** operator. This is the exact solution to L1 penalized regression (LASSO).


---

## Appendix R: Key Proofs in Full

### R.1 Proof: Product Rule

**Theorem.** If $f$ and $g$ are differentiable at $x$, then $(fg)'(x) = f'(x)g(x) + f(x)g'(x)$.

**Proof.**

$$\lim_{h\to 0}\frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

Add and subtract $f(x+h)g(x)$:

$$= \lim_{h\to 0}\frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$

$$= \lim_{h\to 0}\left[f(x+h)\cdot\frac{g(x+h)-g(x)}{h} + g(x)\cdot\frac{f(x+h)-f(x)}{h}\right]$$

$$= \lim_{h\to 0}f(x+h) \cdot g'(x) + g(x) \cdot f'(x)$$

Since $f$ is differentiable at $x$, it is continuous there, so $\lim_{h\to 0}f(x+h) = f(x)$:

$$= f(x)g'(x) + g(x)f'(x). \quad \square$$

### R.2 Proof: Chain Rule (via auxiliary function)

**Theorem.** If $g$ is differentiable at $x$ and $f$ is differentiable at $g(x)$, then $(f\circ g)'(x) = f'(g(x))g'(x)$.

**Proof.** Define the auxiliary function:

$$\phi(k) = \begin{cases} \frac{f(g(x)+k) - f(g(x))}{k} & k \neq 0 \\ f'(g(x)) & k = 0 \end{cases}$$

By differentiability of $f$ at $g(x)$: $\lim_{k\to 0}\phi(k) = f'(g(x))$, so $\phi$ is continuous at $k=0$.

Let $k = g(x+h) - g(x)$. Then for $h \neq 0$:

$$\frac{f(g(x+h)) - f(g(x))}{h} = \phi(k) \cdot \frac{g(x+h)-g(x)}{h}$$

(This holds even when $k = 0$ for some values of $h$, since both sides are 0 in that case.)

Taking $h \to 0$: $k \to 0$ (by continuity of $g$), so $\phi(k) \to f'(g(x))$, and $(g(x+h)-g(x))/h \to g'(x)$:

$$(f\circ g)'(x) = f'(g(x)) \cdot g'(x). \quad \square$$

### R.3 Proof: Differentiable Implies Continuous

**Theorem.** If $f$ is differentiable at $a$, then $f$ is continuous at $a$.

**Proof.**

$$\lim_{x\to a}[f(x) - f(a)] = \lim_{x\to a}\frac{f(x)-f(a)}{x-a}\cdot(x-a) = f'(a)\cdot 0 = 0$$

Therefore $\lim_{x\to a}f(x) = f(a)$, which is the definition of continuity at $a$. $\square$

### R.4 Proof: $(\sin x)' = \cos x$

**Proof.** Using the sine addition formula and two fundamental limits from §01:

$$\frac{\sin(x+h)-\sin x}{h} = \frac{\sin x\cos h + \cos x\sin h - \sin x}{h}$$

$$= \sin x\cdot\frac{\cos h - 1}{h} + \cos x\cdot\frac{\sin h}{h}$$

As $h\to 0$: $(\cos h - 1)/h \to 0$ (even function, $\sin h/h \to 1$ implies $(\cos h-1)/h \to 0$ by squeeze), and $\sin h/h \to 1$:

$$(\sin x)' = \sin x\cdot 0 + \cos x\cdot 1 = \cos x. \quad \square$$


---

## Appendix S: Worked Examples — Chain Rule in Neural Networks

### S.1 Single-Layer Sigmoid Network — Full Backprop

**Setup.** Input $x$, single weight $w$, bias $b$, sigmoid activation, MSE loss:

$$z = wx + b, \quad a = \sigma(z), \quad \hat{y} = a, \quad \mathcal{L} = \tfrac{1}{2}(a - y)^2$$

**Forward pass (left to right):**

| Variable | Value |
|---------|-------|
| $z$ | $wx + b$ |
| $a$ | $\sigma(z)$ |
| $\mathcal{L}$ | $\frac{1}{2}(a-y)^2$ |

**Backward pass (chain rule, right to left):**

$$\frac{\partial\mathcal{L}}{\partial a} = a - y$$

$$\frac{\partial\mathcal{L}}{\partial z} = \frac{\partial\mathcal{L}}{\partial a}\cdot\frac{\partial a}{\partial z} = (a-y)\cdot\sigma'(z) = (a-y)\cdot a(1-a)$$

$$\frac{\partial\mathcal{L}}{\partial w} = \frac{\partial\mathcal{L}}{\partial z}\cdot\frac{\partial z}{\partial w} = (a-y)\cdot a(1-a)\cdot x$$

$$\frac{\partial\mathcal{L}}{\partial b} = \frac{\partial\mathcal{L}}{\partial z}\cdot\frac{\partial z}{\partial b} = (a-y)\cdot a(1-a)\cdot 1$$

The structure: error signal $(a-y)$ is modulated by the local gradient $\sigma'(z) = a(1-a)$ before being distributed to the weight $w$ (scaled by input $x$) and bias $b$.

### S.2 Two-Layer ReLU Network

Input $x$, hidden layer $h_1 = \text{ReLU}(w_1 x + b_1)$, output $\hat{y} = w_2 h_1 + b_2$:

$$\mathcal{L} = \tfrac{1}{2}(\hat{y}-y)^2$$

**Gradients:**

$$\frac{\partial\mathcal{L}}{\partial w_2} = (\hat{y}-y)\cdot h_1$$

$$\frac{\partial\mathcal{L}}{\partial b_2} = (\hat{y}-y)$$

$$\delta_1 = \frac{\partial\mathcal{L}}{\partial(w_1 x+b_1)} = (\hat{y}-y)\cdot w_2\cdot\mathbf{1}[w_1 x+b_1 > 0]$$

$$\frac{\partial\mathcal{L}}{\partial w_1} = \delta_1\cdot x, \qquad \frac{\partial\mathcal{L}}{\partial b_1} = \delta_1$$

The indicator $\mathbf{1}[\cdot]$ is the ReLU subgradient: if the pre-activation was negative, the gradient is blocked (dead neuron).

### S.3 Gradient Flow Visualization

```
BACKPROPAGATION GRADIENT FLOW
════════════════════════════════════════════════════════════════════════

  INPUT                                              LOSS
    x ──→ [z = wx+b] ──→ [a = σ(z)] ──→ [L = ½(a-y)²]
              │                  │                 │
              │      Forward ─── ► ─── Forward     │
              │                                    │
              │      ◄ ─── Backward ─── Backward   │
              │                  │                 │
           ∂L/∂w = (a-y)·a(1-a)·x              ∂L/∂a = a-y

  Each node multiplies its local gradient into the flowing signal.
  This IS the chain rule: ∂L/∂w = (∂L/∂a)(∂a/∂z)(∂z/∂w)

════════════════════════════════════════════════════════════════════════
```

### S.4 Gradient Tape — PyTorch Analogy

In PyTorch, the computation graph is built dynamically during the forward pass. `.backward()` traverses it in reverse, applying the chain rule at each node:

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
z = w * x + 1.0          # z = 7.0
a = torch.sigmoid(z)      # a ≈ 0.999
L = 0.5 * (a - 0.5) ** 2  # L

L.backward()              # Reverse-mode AD: computes all ∂L/∂w, ∂L/∂x
print(w.grad)             # ∂L/∂w = (a-0.5)*a*(1-a)*x
print(x.grad)             # ∂L/∂x = (a-0.5)*a*(1-a)*w
```

This is the chain rule automated — the "tape" records $(z, a, L)$ in the forward pass, then `.backward()` applies our formulas in reverse.


---

## Appendix T: Summary and Connections

### T.1 Core Theorem Summary

| Theorem | Statement | Key Condition | Consequence |
|---------|-----------|--------------|------------|
| Differentiable → Continuous | $f'(a)$ exists → $f$ continuous at $a$ | Differentiability | Justifies assuming continuity in proofs |
| Power Rule | $(x^n)' = nx^{n-1}$ | $n \in \mathbb{R}$ | Foundation of all polynomial differentiation |
| Product Rule | $(fg)' = f'g+fg'$ | $f,g$ differentiable | Essential for activation analysis |
| Chain Rule | $(f\circ g)'=f'(g)\cdot g'$ | Both differentiable | Backbone of backpropagation |
| Mean Value Theorem | $\exists c: f'(c) = (f(b)-f(a))/(b-a)$ | Continuous $[a,b]$, diff $(a,b)$ | Lipschitz bounds, descent lemmas |
| First Derivative Test | Sign change of $f'$ classifies extrema | $f'(x_0)=0$ | Optimization — finding local minima |
| Second Derivative Test | $f''$ sign at critical point | $f'(x_0)=0$, $f''$ exists | Confirms min/max; generalizes to PD Hessian |
| Inverse Function Theorem | $(f^{-1})'=1/f'(f^{-1})$ | $f'(a)\neq 0$ | Derives inverse trig derivatives; flows |

### T.2 Dependency Graph — All Concepts in This Section

```
DERIVATIVE DEPENDENCY MAP
════════════════════════════════════════════════════════════════════════

  Limit (§01) ─────────────→ Derivative Definition
                                      │
             ┌────────────────────────┼────────────────┐
             ▼                        ▼                 ▼
       Constant/Power           Product Rule        Chain Rule
       Sum/Difference           Quotient Rule           │
             │                        │                 │
             └────────────────────────┼─────────────────┘
                                      ▼
                               Differentiation Rules
                                      │
          ┌───────────────────────────┼──────────────────┐
          ▼                           ▼                   ▼
   Higher Derivatives         Implicit Diff.      Activation f'
   Concavity/Inflection        Related Rates     σ', ReLU', GELU'
          │                                              │
          ▼                                              ▼
   Critical Points /                            Vanishing Gradients
   Extrema / MVT                                Backpropagation
          │
          ▼
   Gradient Descent Theory → §08-Optimization

════════════════════════════════════════════════════════════════════════
```

### T.3 What to Review if You're Stuck

| Struggling with | Review |
|----------------|--------|
| Chain rule applications | §4.1–4.2, Appendix B |
| Activation function derivatives | §8.1–8.5, Appendix G |
| Critical point classification | §7.1–7.3 |
| Numerical differentiation step size | §9.1–9.2, Appendix F |
| Backpropagation derivation | §4.3, Appendix S |
| Implicit differentiation | §5.1, Appendix L |
| Why ReLU avoids vanishing gradients | Appendix G.1 |
| Softmax gradient computation | Appendix G.3 |
| Adam optimizer intuition | Appendix I.1 |
| Gradient checking in code | §9.3 |


---

## Appendix U: Practice Problems — Solutions

### U.1 Full Solutions: Section 11 Exercises

**Exercise 1 — Basic rules.**

(a) $f(x) = 3x^4 - 2x^2 + 7$:
$$f'(x) = 12x^3 - 4x$$

(b) $g(x) = \sqrt{x}\,e^x = x^{1/2}e^x$. Product rule:
$$g'(x) = \tfrac{1}{2}x^{-1/2}e^x + x^{1/2}e^x = e^x\!\left(\frac{1}{2\sqrt{x}} + \sqrt{x}\right) = e^x\cdot\frac{1+2x}{2\sqrt{x}}$$

(c) $h(x) = (x^2+1)/(x-1)$. Quotient rule ($f=x^2+1$, $g=x-1$):
$$h'(x) = \frac{2x(x-1) - (x^2+1)(1)}{(x-1)^2} = \frac{2x^2-2x-x^2-1}{(x-1)^2} = \frac{x^2-2x-1}{(x-1)^2}$$

**Exercise 2 — Chain rule.**

(a) $(\sin(x^3))' = \cos(x^3)\cdot 3x^2$

(b) $(\ln(\cos x))' = \frac{-\sin x}{\cos x} = -\tan x$

(c) $((1+x^2)^{10})' = 10(1+x^2)^9\cdot 2x = 20x(1+x^2)^9$

(d) $(e^{\sin x})' = e^{\sin x}\cos x$

**Exercise 3 — Implicit differentiation: $x^3 + y^3 = 6xy$.**

Differentiate: $3x^2 + 3y^2 y' = 6y + 6x y'$

Collect $y'$: $y'(3y^2 - 6x) = 6y - 3x^2$

$$y' = \frac{6y - 3x^2}{3y^2 - 6x} = \frac{2y - x^2}{y^2 - 2x}$$

**Exercise 4 — Sigmoid derivative.**

(a) Derivation:
$$\sigma'(x) = \frac{d}{dx}(1+e^{-x})^{-1} = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2}$$

Factor: $= \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)\cdot(1-\sigma(x))$. $\square$

**Exercise 6 — MVT proof: $|\sin a - \sin b| \leq |a-b|$.**

The sine function is differentiable with $|(\sin x)' | = |\cos x| \leq 1$ for all $x$. By the MVT, for any $a \neq b$ there exists $c$ between $a$ and $b$ with:

$$\frac{\sin a - \sin b}{a - b} = \cos c$$

Therefore:
$$|\sin a - \sin b| = |\cos c||a-b| \leq 1\cdot|a-b| = |a-b|. \quad \square$$

(The inequality also holds for $a = b$: $0 \leq 0$.)

**Exercise 8 — Backpropagation chain rule:**

Forward: $z_1 = w_1 x$, $a_1 = \sigma(z_1)$, $z_2 = w_2 a_1$, $\mathcal{L} = \frac{1}{2}(z_2-y)^2$.

$$\frac{\partial\mathcal{L}}{\partial z_2} = z_2 - y = w_2 a_1 - y$$

$$\frac{\partial\mathcal{L}}{\partial w_2} = (z_2-y)\cdot a_1$$

$$\frac{\partial\mathcal{L}}{\partial a_1} = (z_2-y)\cdot w_2$$

$$\frac{\partial\mathcal{L}}{\partial z_1} = (z_2-y)\cdot w_2\cdot\sigma'(z_1) = (z_2-y)\cdot w_2\cdot a_1(1-a_1)$$

$$\frac{\partial\mathcal{L}}{\partial w_1} = (z_2-y)\cdot w_2\cdot a_1(1-a_1)\cdot x$$

This reproduces the four-factor chain: loss sensitivity × downstream weight × local gate × input.


---

## Appendix V: Connections to Later Chapters

### V.1 What Integration Needs from This Section

The next section ([§03-Integration](../03-Integration/notes.md)) uses:
- **Antiderivatives**: integration reverses differentiation. Every differentiation formula gives an integration formula: $(\sin x)' = \cos x \Rightarrow \int\cos x\,dx = \sin x + C$.
- **Fundamental Theorem of Calculus (Part 2)**: if $F' = f$, then $\int_a^b f(x)\,dx = F(b) - F(a)$. This requires knowing derivatives of common functions.
- **Substitution rule**: the reverse chain rule. If $u = g(x)$, $\int f(g(x))g'(x)\,dx = \int f(u)\,du$.

Every derivative formula in §3.5 has a corresponding integral formula. Master both sides.

### V.2 What Series Needs from This Section

[§04-Series-and-Sequences](../04-Series-and-Sequences/notes.md) uses:
- **Higher-order derivatives**: Taylor coefficients $a_n = f^{(n)}(a)/n!$ require computing many derivatives.
- **Linear approximation**: the first-order Taylor expansion $f(x) \approx f(a) + f'(a)(x-a)$ is a direct extension of §5.3.
- **Error bounds**: the Taylor remainder involves $f^{(n+1)}$ at some intermediate point (MVT applied to integrals).

### V.3 What Multivariate Calculus Needs from This Section

[§05-Multivariate Calculus](../../05-Multivariate-Calculus/README.md) extends every concept here:

| §02 Concept | §05 Extension |
|-------------|--------------|
| Derivative $f'(x)$ | Gradient $\nabla f(\mathbf{x})$ |
| Chain rule $df/dx = (df/du)(du/dx)$ | Chain rule with Jacobians: $\mathbf{J}_{f\circ g} = \mathbf{J}_f \mathbf{J}_g$ |
| Second derivative $f''(x)$ | Hessian $\mathbf{H} = \nabla^2 f$ |
| Inflection point | Saddle point in $\mathbb{R}^n$ |
| Linear approximation $f(x)\approx f(a)+f'(a)(x-a)$ | $f(\mathbf{x})\approx f(\mathbf{a})+\nabla f(\mathbf{a})^\top(\mathbf{x}-\mathbf{a})$ |
| $f'(a) = 0$ at extremum | $\nabla f(\mathbf{a}) = \mathbf{0}$ at extremum |

The scalar theory developed here is the essential foundation. Every formula in §05 will look familiar with vector notation substituted.

### V.4 What Optimization Needs from This Section

[§08-Optimization](../../08-Optimization/README.md) uses directly:
- Gradient descent theory (§7.3 MVT, descent lemma)
- Critical point classification (saddles dominate in high dimensions)
- Chain rule for backpropagation
- Adam optimizer — second moment as curvature estimate
- Gradient checking (§9.3) for debugging

---

## Appendix W: Glossary

| Term | Definition |
|------|-----------|
| **Derivative** | $f'(a) = \lim_{h\to 0}[f(a+h)-f(a)]/h$ — instantaneous rate of change |
| **Differentiable** | The derivative exists and is finite at the point |
| **Smooth** ($C^\infty$) | All derivatives of all orders exist and are continuous |
| **Critical point** | $x_0$ where $f'(x_0) = 0$ or $f'(x_0)$ undefined |
| **Local minimum** | $f(x_0) \leq f(x)$ for all $x$ near $x_0$ |
| **Concave up** | $f''(x) > 0$ — bowl shape, gradient is increasing |
| **Inflection point** | Where $f''$ changes sign |
| **Subgradient** | Generalized derivative for non-smooth convex functions |
| **Lipschitz constant** | $L$ such that $|f(x)-f(y)|\leq L|x-y|$ — controls max gradient magnitude |
| **Forward difference** | $(f(x+h)-f(x))/h$ — $O(h)$ approximation to $f'(x)$ |
| **Centered difference** | $(f(x+h)-f(x-h))/(2h)$ — $O(h^2)$ approximation to $f'(x)$ |
| **Automatic differentiation** | Exact derivative computation by applying chain rule to elementary ops |
| **Backpropagation** | Reverse-mode AD on neural network computation graph |
| **Vanishing gradient** | Product of many small derivatives → exponential decay |
| **Score function** | $\nabla_\theta \log p(x;\theta)$ — derivative of log-likelihood |


---

## Appendix X: Additional Examples — Differentiating Through Compositions

### X.1 Softplus as Smooth ReLU

The softplus function $f(x) = \ln(1+e^x)$ is a smooth approximation to ReLU.

**Derivative:**
$$f'(x) = \frac{e^x}{1+e^x} = \frac{1}{1+e^{-x}} = \sigma(x)$$

Softplus's derivative is the sigmoid! This means:
- Softplus is always increasing ($\sigma(x) > 0$ for all $x$)
- Its second derivative: $f''(x) = \sigma'(x) = \sigma(x)(1-\sigma(x)) > 0$ — always concave up
- Softplus approaches ReLU as temperature $T \to 0$: $T\ln(1+e^{x/T}) \to \max(0,x)$

### X.2 SiLU / Swish

Swish: $\text{swish}(x) = x\sigma(x)$.

**Derivative (product rule):**
$$\text{swish}'(x) = \sigma(x) + x\sigma'(x) = \sigma(x) + x\sigma(x)(1-\sigma(x)) = \sigma(x)(1 + x(1-\sigma(x)))$$

At $x = 0$: $\text{swish}'(0) = \sigma(0)(1+0) = 1/2$.

Swish is non-monotone: has a local minimum at $x \approx -1.28$ (where $\text{swish}'(x) = 0$). This property, combined with self-gating, is hypothesized to improve gradient flow in deep networks.

### X.3 Mish

Mish: $\text{mish}(x) = x\tanh(\text{softplus}(x)) = x\tanh(\ln(1+e^x))$.

**Derivative (product rule + chain rule):**
$$\text{mish}'(x) = \tanh(\text{softplus}(x)) + x \cdot \text{sech}^2(\text{softplus}(x)) \cdot \sigma(x)$$

Mish is smooth, non-monotone (like Swish), and has shown improvements over ReLU in some image classification benchmarks. Its derivative involves three activations composed — a showcase for the chain rule.

### X.4 Gradient of Cross-Entropy Loss — Full Derivation

For $K$-class classification with softmax output $\mathbf{p} = \text{softmax}(\mathbf{z})$ and one-hot label $\mathbf{y}$:

$$\mathcal{L} = -\sum_{k=1}^K y_k \log p_k$$

$$\frac{\partial\mathcal{L}}{\partial z_j} = -\sum_k y_k \frac{\partial\log p_k}{\partial z_j} = -\sum_k y_k \frac{1}{p_k}\frac{\partial p_k}{\partial z_j}$$

Using the softmax Jacobian from §G.3:

$$= -\sum_k y_k \frac{1}{p_k}p_k(\delta_{kj} - p_j) = -\sum_k y_k(\delta_{kj} - p_j)$$

$$= -y_j + p_j\sum_k y_k = p_j - y_j$$

(since $\sum_k y_k = 1$ for a one-hot vector). This elegant result — gradient = predicted probability minus true probability — is why the combined softmax + cross-entropy layer is computationally convenient and numerically stable.


---

## Appendix Y: Connecting to the Broader Curriculum

### Y.1 Prerequisites Review

Before using the material in this section, ensure you are solid on:

- **Limit definition** ([§01 §2](../01-Limits-and-Continuity/notes.md)): $\lim_{h\to 0}(...)$ is the backbone of the derivative definition
- **Fundamental limits** ([§01 §3](../01-Limits-and-Continuity/notes.md)): $\lim_{h\to 0}\sin(h)/h = 1$ is used to derive $(\sin x)' = \cos x$
- **Continuity** ([§01 §6](../01-Limits-and-Continuity/notes.md)): differentiability implies continuity; you need continuity to apply EVT and FTC
- **Algebra**: polynomial factoring, rational expression manipulation — needed for quotient rule applications
- **Exponential/log identities**: $e^{a+b} = e^a e^b$, $\ln(ab) = \ln a + \ln b$ — used in logarithmic differentiation

### Y.2 Key Formulas from This Section (Memorize These)

$$\boxed{(x^n)' = nx^{n-1}} \qquad \boxed{(e^x)' = e^x} \qquad \boxed{(\ln x)' = \frac{1}{x}}$$

$$\boxed{(fg)' = f'g + fg'} \qquad \boxed{\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}} \qquad \boxed{(f\circ g)' = f'(g)\cdot g'}$$

$$\boxed{\sigma'(x) = \sigma(x)(1-\sigma(x))} \qquad \boxed{\text{ReLU}'(x) = \mathbf{1}[x>0]}$$

$$\boxed{f'(x) \approx \frac{f(x+h)-f(x-h)}{2h} + O(h^2)}$$

These formulas appear throughout the rest of the curriculum, in optimization, probability, and all applied sections.


---

_This section is part of the [Math for LLMs](../../README.md) curriculum. For issues or contributions, see [CONTRIBUTING.md](../../CONTRIBUTING.md)._
