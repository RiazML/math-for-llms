[← Previous Chapter: Advanced Linear Algebra](../03-Advanced-Linear-Algebra/README.md) | [Next Chapter: Multivariate Calculus →](../05-Multivariate-Calculus/README.md)

---

# Chapter 4 — Calculus Fundamentals

> _"Calculus is the science of measuring and harnessing change — and every gradient descent step, every backpropagation pass, every convergence proof in machine learning is calculus in action."_

## Overview

This chapter builds the single-variable calculus foundation that underpins all continuous mathematics in machine learning. The progression moves from the rigorous notion of a limit (the definition of "approaching"), through the derivative as the instantaneous rate of change, through integration as accumulation, and finally through series as infinite-precision approximation.

Every concept introduced here has a direct ML counterpart: limits underlie gradient definitions and softmax temperature; derivatives are backpropagation; integration is expectation over probability densities; Taylor series are the theoretical foundation of optimizer update rules and attention approximations. Understanding the mathematical substance — not just the mechanics — separates practitioners who can debug and innovate from those who can only apply recipes.

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|-----------|----------------|-----------------|
| 01 | [Limits and Continuity](01-Limits-and-Continuity/notes.md) | Rigorous foundation of approaching, convergence, and function regularity | ε-δ definition, limit laws, Squeeze Theorem, L'Hôpital's Rule, one-sided limits, IVT, EVT, continuity, discontinuity types, numerical stability near limits |
| 02 | [Derivatives and Differentiation](02-Derivatives-and-Differentiation/notes.md) | Instantaneous rate of change; differentiation rules; activation function analysis | Derivative definition, power/product/quotient/chain rules, implicit differentiation, higher-order derivatives, activation function derivatives, critical points, numerical differentiation |
| 03 | [Integration](03-Integration/notes.md) | Accumulation and area; the Fundamental Theorem; probabilistic expectations | Riemann integral, antiderivatives, FTC parts I & II, substitution, integration by parts, improper integrals, numerical integration, probability density integration |
| 04 | [Series and Sequences](04-Series-and-Sequences/notes.md) | Infinite sums, convergence, and function approximation via Taylor/power series | Sequences, convergence tests, power series, radius of convergence, Taylor series, Maclaurin series, error bounds |

---

## Reading Order and Dependencies

```
01-Limits-and-Continuity        (ε-δ, squeeze, IVT — rigorous foundation)
        ↓
02-Derivatives-and-Differentiation  (derivative as a limit; all rules; activation analysis)
        ↓
03-Integration                  (antiderivatives; FTC connects integrals to derivatives)
        ↓
04-Series-and-Sequences         (Taylor series uses derivatives; convergence uses limits)
        ↓
05-Multivariate-Calculus        (next chapter — extends §02 chain rule, §03 Fubini, §04 Taylor)
```

---

## What Belongs Where — Canonical Homes

This table is the authoritative scoping guide. **If a topic has a canonical home, every other section must give at most a 1–2 paragraph preview with a forward/backward reference — never a full treatment.**

| Topic | Canonical Home | Preview Only In |
|-------|---------------|-----------------|
| ε-δ definition of limits | §01 | — |
| Limit laws (sum, product, quotient, composition) | §01 | — |
| Squeeze Theorem (proof + applications) | §01 | — |
| L'Hôpital's Rule | §01 | — |
| One-sided limits | §01 | — |
| Limits at infinity and infinite limits | §01 | — |
| Fundamental limits: $\sin(x)/x$, $(e^x-1)/x$, $(1+1/n)^n$ | §01 | §02 (used in derivatives) |
| Continuity: definition, three-part condition | §01 | §02 (differentiability implies continuity) |
| Types of discontinuity: removable, jump, essential | §01 | — |
| Intermediate Value Theorem | §01 | §03 (root existence in FTC proof) |
| Extreme Value Theorem | §01 | §02 (applied to critical points) |
| Uniform continuity, Lipschitz continuity | §01 | — |
| Numerical stability near limits (`expm1`, `log1p`) | §01 | §02 (numerically stable gradients) |
| Derivative definition (limit of difference quotient) | §02 | §01 (brief forward preview only) |
| Power rule, product rule, quotient rule | §02 | — |
| Chain rule (single-variable) | §02 | §05 (multivariable extension) |
| Implicit differentiation | §02 | — |
| Higher-order derivatives, concavity | §02 | §04 (used in Taylor remainder) |
| Activation function derivatives: sigmoid, tanh, ReLU, GELU | §02 | §01 (continuity of activations only) |
| Critical points, local extrema (first/second derivative test) | §02 | §08-Optimization (multivariable extension) |
| Numerical differentiation / finite differences | §02 | §01 (gradient-as-limit preview only) |
| Related rates, linear approximation | §02 | — |
| Antiderivatives and indefinite integrals | §03 | — |
| Riemann sums and definite integral definition | §03 | — |
| Fundamental Theorem of Calculus (both parts) | §03 | §02 (brief forward reference) |
| Integration by substitution (u-substitution) | §03 | — |
| Integration by parts | §03 | — |
| Partial fractions | §03 | — |
| Improper integrals | §03 | — |
| Numerical integration: trapezoid, Simpson's, Monte Carlo | §03 | — |
| Integration in probability: PDF, CDF, expectation | §03 | §06-Probability (uses these results) |
| Sequences: definition, convergence, Cauchy criterion | §04 | §01 (sequential characterisation of limits) |
| Infinite series, partial sums | §04 | — |
| Convergence tests: ratio, root, comparison, integral | §04 | — |
| Power series, radius of convergence | §04 | — |
| Taylor series (full derivation, remainder, convergence) | §04 | §02 (first-order/linear approximation only) |
| Maclaurin series for $e^x$, $\sin x$, $\cos x$, $\ln(1+x)$ | §04 | §01 (used to evaluate limits) |
| Taylor series in ML: optimizer analysis, attention approximations | §04 | — |

---

## Overlap Danger Zones

These topics are the most commonly duplicated across sections. **Read this before writing any content.**

### 1. Derivative ↔ Limit
- **§01** may preview the derivative as a limit of a difference quotient (1–2 paragraphs, forward ref to §02). It must NOT include differentiation rules, worked differentiation examples, or activation function derivatives.
- **§02** owns the full treatment of the derivative: definition, rules, applications.

### 2. Taylor Expansion ↔ Derivative / Limit
- **§02** may use first-order Taylor approximation $f(x) \approx f(a) + f'(a)(x-a)$ (linear approximation) because it requires only derivatives. It must NOT prove Taylor's theorem or discuss convergence.
- **§01** may use the Taylor expansion of $e^x$ informally to evaluate limits (e.g., $(e^x-1)/x$). It must NOT derive the series.
- **§04** owns the full treatment: derivation of Taylor series, remainder theorem, convergence radius, ML applications.

### 3. Chain Rule ↔ Backpropagation
- **§02** derives the single-variable chain rule and applies it to simple function compositions and backpropagation through scalar computations.
- **§05-Multivariate Calculus** owns the multivariable chain rule, Jacobians, and the full backpropagation algorithm for vector-valued functions.
- **§02** should forward-reference §05 for the general case.

### 4. Integration ↔ Probability
- **§03** covers integration of probability density functions and the definition of expectation $\mathbb{E}[X] = \int x\, p(x)\, dx$ as a worked example of improper integration.
- **§06-Probability Theory** owns the full treatment of random variables, distributions, and probabilistic reasoning.
- **§03** should forward-reference §06 for the probabilistic interpretation; §06 should backward-reference §03 for the integration mechanics.

### 5. Continuity of Activation Functions
- **§01** covers continuity of ReLU, GELU, sigmoid as examples of continuity and discontinuity types (at the origin — is the function continuous? is its derivative continuous?).
- **§02** covers the derivatives of those same activation functions (sigmoid', ReLU', GELU').
- Do NOT duplicate: §01 discusses continuity only; §02 discusses differentiability and the derivative formula only.

### 6. Numerical Differentiation
- **§01** previews the gradient as a limit (1–2 paragraphs, as motivation for why limits matter for AI).
- **§02** owns numerical differentiation: one-sided vs centered finite differences, error order $O(h)$ vs $O(h^2)$, optimal step size, gradient checking implementation.

---

## Forward and Backward Reference Format

When a topic belongs to another section, use this exact format rather than covering it in depth:

**Forward reference** (full treatment is later):
```markdown
> **Preview: Taylor Series**
> Every smooth function can be represented as an infinite polynomial:
> $f(x) = \sum_{n=0}^\infty \frac{f^{(n)}(a)}{n!}(x-a)^n$.
> The first-order truncation $f(x) \approx f(a) + f'(a)(x-a)$ is the linear approximation
> used throughout optimisation.
>
> → _Full treatment: [Series and Sequences](../04-Series-and-Sequences/notes.md)_
```

**Backward reference** (builds on earlier section):
```markdown
> **Recall:** The derivative $f'(a) = \lim_{h\to 0}[f(a+h)-f(a)]/h$ was defined in
> [Limits and Continuity](../01-Limits-and-Continuity/notes.md#95-gradient-as-a-limit).
> We now develop rules for computing it efficiently.
```

---

## Key Cross-Chapter Dependencies

**From Chapter 3 — Advanced Linear Algebra:**
- Matrix derivatives (Jacobians, Hessians) appear in [§05-Multivariate Calculus](../05-Multivariate-Calculus/README.md) — the present chapter provides the scalar foundation
- Positive definite Hessians (from [§07-Positive-Definite-Matrices](../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)) appear in §02 second-derivative tests and §08-Optimization

**Into Chapter 5 — Multivariate Calculus:**
- §01 (limits) → multivariable limits and partial derivative definition
- §02 (chain rule) → multivariable chain rule, Jacobians, backpropagation
- §03 (integration) → double/triple integrals, Fubini's theorem
- §04 (Taylor series) → multivariate Taylor expansion, Hessian as second-order term

**Into Chapter 8 — Optimization:**
- §02 (critical points) → gradient descent, Newton's method
- §02 (chain rule) → backpropagation
- §04 (Taylor series) → momentum, Adam, trust-region analysis

**Into Chapter 6 — Probability Theory:**
- §03 (integration) → continuous probability, PDF/CDF, expectation, variance

---

## ML Concept Map

| ML Concept | Calculus Foundation | Section |
|-----------|--------------------|----|
| Backpropagation | Chain rule applied to computation graph | §02 |
| Gradient descent update | Derivative of loss w.r.t. parameters | §02 |
| Softmax temperature ($T \to 0$) | Limits at zero | §01 |
| Vanishing gradient | $\sigma'(x) \to 0$ as $x \to \pm\infty$ | §01, §02 |
| Expected loss $\mathbb{E}[\mathcal{L}]$ | Integration over data distribution | §03 |
| KL divergence, entropy | Improper integrals of $p \log p$ | §03 |
| Adam optimizer (second moment) | Power series expansion of update rule | §04 |
| Attention kernel approximation | Taylor expansion of $e^{qk^\top}$ | §04 |
| Learning rate convergence (Robbins-Monro) | Series convergence conditions | §01, §04 |
| Numerical gradient checking | Finite difference approximation of limit | §02 |
| Monte Carlo expectation | Numerical integration via sampling | §03 |
| Log-sum-exp stability | Limits near singularities | §01 |

---

## Prerequisites

Before starting this chapter, ensure you are comfortable with:

- **Real number system**: $\mathbb{R}$, absolute value, inequalities — [§01-Mathematical-Foundations](../01-Mathematical-Foundations/README.md)
- **Functions**: domain, codomain, composition, inverse — [§01-Mathematical-Foundations](../01-Mathematical-Foundations/README.md)
- **Algebra**: polynomial factoring, rational expressions, exponential/logarithmic identities
- **Trigonometry**: $\sin$, $\cos$, $\tan$ and their basic identities
- **Matrix operations**: matrix multiply, transpose — [§02-Linear-Algebra-Basics](../02-Linear-Algebra-Basics/README.md) (for later ML applications)

---

[← Previous Chapter: Advanced Linear Algebra](../03-Advanced-Linear-Algebra/README.md) | [Next Chapter: Multivariate Calculus →](../05-Multivariate-Calculus/README.md)
