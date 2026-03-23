[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Stochastic Processes →](../06-Stochastic-Processes/notes.md)

---

# Concentration Inequalities

> _"The tendency of random quantities to concentrate near their expectations is one of the most powerful and pervasive phenomena in all of mathematics."_
> — Stéphane Boucheron, Gábor Lugosi & Pascal Massart, *Concentration Inequalities* (2013)

## Overview

A random variable $X$ has an expectation $\mathbb{E}[X]$, but individual realisations scatter around it. **Concentration inequalities** answer the quantitative question: how far from $\mathbb{E}[X]$ can $X$ stray, and with what probability? These are not mere abstract bounds — they are the mathematical engine behind almost every theoretical guarantee in machine learning.

Every time a practitioner asks "how many training examples do I need to achieve 95% accuracy with 99% confidence?", the answer is a concentration inequality. Every time a theorist proves that a learning algorithm *generalises* from training data to unseen data, the proof uses a concentration inequality. The PAC learning framework, VC theory, Rademacher complexity, and modern neural tangent kernel theory all rest on the same probabilistic foundation built in this section.

The central insight is a hierarchy: weaker assumptions yield looser bounds, stronger assumptions yield tighter ones. Markov's inequality needs only non-negativity and a finite mean. Chebyshev's needs a finite variance. Hoeffding's needs bounded support. Chernoff's needs a moment generating function. The practitioner's art is matching the right bound to the right problem.

## Prerequisites

- [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md) — $\mathbb{E}[X]$, $\operatorname{Var}(X)$, MGF $M_X(t) = \mathbb{E}[e^{tX}]$, Jensen's inequality, indicator random variables
- [§03 Joint Distributions](../03-Joint-Distributions/notes.md) — independence, conditional distributions
- [§02 Common Distributions](../02-Common-Distributions/notes.md) — Gaussian, Bernoulli, Binomial
- [§01 Introduction and Random Variables](../01-Introduction-and-Random-Variables/notes.md) — probability axioms, union bound preview

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive derivations: Markov, Chebyshev, Hoeffding, Chernoff, McDiarmid, PAC bounds, Rademacher complexity |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from tail bound verification to VC dimension and Rademacher estimation |

## Learning Objectives

After completing this section, you will be able to:

1. State and prove Markov's inequality and identify when it is tight
2. Derive Chebyshev's inequality from Markov and apply the $k$-sigma rule quantitatively
3. Define sub-Gaussian random variables and prove the sub-Gaussian tail bound
4. State and prove Hoeffding's inequality for bounded independent random variables
5. Apply the Chernoff method to derive tail bounds via MGF optimisation
6. State Bernstein's inequality and explain when it improves on Hoeffding
7. Apply McDiarmid's inequality to functions of independent random variables
8. Use the union bound with covering arguments to control events over hypothesis classes
9. Derive the PAC generalisation bound for finite hypothesis classes
10. Define VC dimension and state the generalisation bound with Sauer-Shelah lemma
11. Define Rademacher complexity and interpret the Rademacher generalisation bound
12. Compute required sample sizes for given $(\varepsilon, \delta)$-PAC guarantees

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is Concentration?](#11-what-is-concentration)
  - [1.2 The Hierarchy of Bounds](#12-the-hierarchy-of-bounds)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Why Concentration Matters for ML](#14-why-concentration-matters-for-ml)
- [2. Markov's and Chebyshev's Inequalities](#2-markovs-and-chebyshevs-inequalities)
  - [2.1 Markov's Inequality](#21-markovs-inequality)
  - [2.2 Chebyshev's Inequality](#22-chebyshevs-inequality)
  - [2.3 One-Sided Chebyshev (Cantelli)](#23-one-sided-chebyshev-cantelli)
  - [2.4 Limitations of Moment-Based Bounds](#24-limitations-of-moment-based-bounds)
- [3. Sub-Gaussian Random Variables](#3-sub-gaussian-random-variables)
  - [3.1 Definition and MGF Condition](#31-definition-and-mgf-condition)
  - [3.2 Examples of Sub-Gaussian Variables](#32-examples-of-sub-gaussian-variables)
  - [3.3 The Sub-Gaussian Tail Bound](#33-the-sub-gaussian-tail-bound)
  - [3.4 Closure Under Sums](#34-closure-under-sums)
- [4. Hoeffding's Inequality](#4-hoeffdings-inequality)
  - [4.1 Hoeffding's Lemma](#41-hoeffdings-lemma)
  - [4.2 Hoeffding's Inequality](#42-hoeffdings-inequality)
  - [4.3 Two-Sided Hoeffding and Sample Complexity](#43-two-sided-hoeffding-and-sample-complexity)
  - [4.4 Required Sample Size](#44-required-sample-size)
- [5. Chernoff Bounds](#5-chernoff-bounds)
  - [5.1 The Chernoff Method](#51-the-chernoff-method)
  - [5.2 Chernoff for Bernoulli Sums](#52-chernoff-for-bernoulli-sums)
  - [5.3 Multiplicative Chernoff Form](#53-multiplicative-chernoff-form)
  - [5.4 Lower Tail Chernoff](#54-lower-tail-chernoff)
- [6. Bernstein's Inequality](#6-bernsteins-inequality)
  - [6.1 Sub-Exponential Random Variables](#61-sub-exponential-random-variables)
  - [6.2 Bernstein's Inequality](#62-bernsteins-inequality)
  - [6.3 Bernstein vs Hoeffding](#63-bernstein-vs-hoeffding)
  - [6.4 Application to Gradient Noise](#64-application-to-gradient-noise)
- [7. McDiarmid's Inequality](#7-mcdiarmids-inequality)
  - [7.1 Bounded Differences Condition](#71-bounded-differences-condition)
  - [7.2 McDiarmid's Inequality](#72-mcdiarmids-inequality)
  - [7.3 Relationship to Hoeffding](#73-relationship-to-hoeffding)
  - [7.4 Applications to ML Stability](#74-applications-to-ml-stability)
- [8. The Union Bound and Covering Arguments](#8-the-union-bound-and-covering-arguments)
  - [8.1 Union Bound](#81-union-bound)
  - [8.2 Multiple Hypothesis Testing](#82-multiple-hypothesis-testing)
  - [8.3 Covering Numbers and ε-Nets](#83-covering-numbers-and-ε-nets)
  - [8.4 The Net Argument](#84-the-net-argument)
- [9. PAC Learning and Generalisation Bounds](#9-pac-learning-and-generalisation-bounds)
  - [9.1 The PAC Framework](#91-the-pac-framework)
  - [9.2 Finite Hypothesis Class](#92-finite-hypothesis-class)
  - [9.3 Uniform Convergence](#93-uniform-convergence)
  - [9.4 VC Dimension](#94-vc-dimension)
  - [9.5 Generalisation Bound with VC Dimension](#95-generalisation-bound-with-vc-dimension)
- [10. Rademacher Complexity](#10-rademacher-complexity)
  - [10.1 Definition](#101-definition)
  - [10.2 Rademacher Generalisation Bound](#102-rademacher-generalisation-bound)
  - [10.3 Rademacher of Linear Classifiers](#103-rademacher-of-linear-classifiers)
  - [10.4 Rademacher vs VC](#104-rademacher-vs-vc)
- [11. ML Applications](#11-ml-applications)
  - [11.1 SGD and Gradient Concentration](#111-sgd-and-gradient-concentration)
  - [11.2 Confidence Intervals via Hoeffding](#112-confidence-intervals-via-hoeffding)
  - [11.3 Random Features and Kernel Approximation](#113-random-features-and-kernel-approximation)
  - [11.4 Concentration in High Dimensions](#114-concentration-in-high-dimensions)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)
- [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
- [15. Conceptual Bridge](#15-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is Concentration?

A random variable $X$ has a mean $\mu = \mathbb{E}[X]$. But knowing the mean alone tells us nothing about how a single draw is distributed around it. Concentration inequalities fill this gap by bounding the probability that $X$ deviates far from $\mu$.

The fundamental question is: **given what we know about $X$ (its mean, variance, boundedness, or MGF), what is the tightest bound we can place on $P(|X - \mu| \geq t)$?**

This matters deeply in practice. In machine learning, we observe the *empirical* mean of a loss function on training data and want to know how well it approximates the *true* (population) mean. If $X_1, \ldots, X_n$ are i.i.d. losses with mean $\mu$ and we observe $\bar{X}_n = \frac{1}{n}\sum X_i$, concentration inequalities tell us how close $\bar{X}_n$ is to $\mu$ with high probability. This is the foundation of statistical learning theory.

**The sample mean concentrates around the true mean.** As $n$ grows, the deviation $|\bar{X}_n - \mu|$ becomes increasingly small with high probability. This is the content of the Law of Large Numbers — but concentration inequalities give *quantitative, finite-$n$* bounds, not just asymptotic guarantees.

**For AI:** Every evaluation metric in ML (accuracy, F1, BLEU, ROUGE, perplexity) is estimated from a finite test set. Concentration inequalities tell us how many test examples we need to trust the estimate. HELM benchmarks for LLMs, for instance, implicitly rely on Hoeffding-type bounds for the confidence intervals around model comparisons.

### 1.2 The Hierarchy of Bounds

Concentration inequalities form a clear hierarchy. Moving down requires stronger assumptions but yields exponentially tighter bounds:

```
CONCENTRATION INEQUALITY HIERARCHY
════════════════════════════════════════════════════════════════════════

  Assumption          Inequality         Tail bound form
  ─────────────────────────────────────────────────────────────────
  E[X] < ∞, X ≥ 0   Markov             P(X ≥ t) ≤ μ/t
  (polynomial decay)

  Var(X) < ∞         Chebyshev          P(|X-μ| ≥ t) ≤ σ²/t²
  (polynomial decay)

  E[e^{sX}] < ∞      Chernoff method    P(X ≥ t) ≤ min_s e^{-st}M(s)
  (general expo)

  X ∈ [a,b] a.s.     Hoeffding          P(X̄-μ ≥ t) ≤ exp(-2nt²/(b-a)²)
  (sub-Gaussian)      (exponential decay)

  Var known + bounded Bernstein          exp(-nt²/(2σ²+2ct/3))
  (variance-aware)    (tighter for small t)

  f(X₁,...,Xₙ)       McDiarmid          exp(-2t²/Σcᵢ²)
  bounded differences (functions of iid)

════════════════════════════════════════════════════════════════════════
```

The key insight: **moment-based bounds** (Markov, Chebyshev) give polynomial tails $O(1/t^k)$, while **MGF-based bounds** give exponential tails $O(e^{-ct^2})$. Exponential tails are vastly superior — a Gaussian has $P(|X| \geq 3\sigma) \approx 0.003$, while Chebyshev gives only $1/9 \approx 0.11$.

### 1.3 Historical Timeline

| Year | Person | Contribution |
|------|--------|-------------|
| 1867 | Chebyshev | Proved the inequality bearing his name; used it to prove the weak LLN |
| 1899 | Markov (student of Chebyshev) | Simpler proof via indicator argument; Markov's inequality |
| 1952 | Herman Chernoff | MGF-based exponential tail bounds for sums of independent variables |
| 1963 | Wassily Hoeffding | Tight bounds for bounded variables; modern form used everywhere |
| 1971 | Vapnik & Chervonenkis | VC dimension theory; combinatorial approach to generalisation |
| 1984 | Leslie Valiant | PAC learning framework — theoretical model for machine learning |
| 1989 | Colin McDiarmid | Bounded differences inequality for general functions of independent variables |
| 1995 | Bartlett & Mendelson | Rademacher complexity — data-dependent alternative to VC theory |
| 2013 | Boucheron, Lugosi & Massart | Comprehensive monograph unifying modern concentration theory |

### 1.4 Why Concentration Matters for ML

**Generalisation:** A neural network achieves 99% training accuracy. Does it generalise? Concentration inequalities bound $|R(h) - \hat{R}(h)|$ — the gap between true and empirical risk. They tell us when we can trust training metrics.

**SGD analysis:** Stochastic gradient descent uses a mini-batch estimate $\hat{g}$ of the true gradient $g$. Concentration inequalities bound $\|\hat{g} - g\|_2$, determining the noise level and required step sizes for convergence.

**Confidence intervals:** When evaluating an LLM on 1000 benchmark questions with accuracy 73.2%, concentration bounds tell us the confidence interval around this estimate.

**Random features:** The Rahimi-Recht random features method approximates kernel matrices using random projections. How many random features are needed? The answer is a Hoeffding bound.

**Dropout:** Dropout randomly zeroes activations. The kept activations are bounded, so McDiarmid's inequality bounds the output variance.

---

## 2. Markov's and Chebyshev's Inequalities

### 2.1 Markov's Inequality

**Theorem (Markov's Inequality).** Let $X \geq 0$ be a non-negative random variable with $\mathbb{E}[X] < \infty$. For any $t > 0$:
$$P(X \geq t) \leq \frac{\mathbb{E}[X]}{t}$$

**Proof.** This follows immediately from the indicator $\mathbf{1}[X \geq t] \leq X/t$ for $X \geq 0$, $t > 0$:
$$P(X \geq t) = \mathbb{E}[\mathbf{1}[X \geq t]] \leq \mathbb{E}\!\left[\frac{X}{t}\right] = \frac{\mathbb{E}[X]}{t}$$

This proof is a master class in the indicator trick: multiply by 1, then bound the indicator.

**Tightness.** Markov's bound is tight. Consider $X = t$ with probability $\mu/t$ and $X = 0$ otherwise. Then $\mathbb{E}[X] = \mu$ and $P(X \geq t) = \mu/t$ — exactly matching the bound.

**Extensions.** Markov applies to any non-negative function of $X$: for any non-decreasing $\phi \geq 0$,
$$P(X \geq t) = P(\phi(X) \geq \phi(t)) \leq \frac{\mathbb{E}[\phi(X)]}{\phi(t)}$$

Setting $\phi(x) = x^k$ gives the *$k$-th moment bound*: $P(X \geq t) \leq \mathbb{E}[X^k]/t^k$. Setting $\phi(x) = e^{sx}$ gives the Chernoff method.

**For AI:** Markov's inequality underlies gradient clipping analysis. If $\mathbb{E}[\|g\|_2] \leq G$, then $P(\|g\|_2 \geq cG) \leq 1/c$ — so gradient norms exceed $10G$ at most 10% of steps.

### 2.2 Chebyshev's Inequality

**Theorem (Chebyshev's Inequality).** For any random variable $X$ with $\mathbb{E}[X] = \mu$ and $\operatorname{Var}(X) = \sigma^2 < \infty$, for any $t > 0$:
$$P(|X - \mu| \geq t) \leq \frac{\sigma^2}{t^2}$$

Equivalently, setting $t = k\sigma$: $P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$.

**Proof.** Apply Markov to the non-negative variable $Y = (X - \mu)^2$:
$$P(|X - \mu| \geq t) = P((X-\mu)^2 \geq t^2) \leq \frac{\mathbb{E}[(X-\mu)^2]}{t^2} = \frac{\sigma^2}{t^2}$$

**The $k$-sigma rule.** Chebyshev guarantees: at least $1 - 1/k^2$ of probability mass lies within $k\sigma$ of the mean. For any distribution with finite variance:

| $k$ | Chebyshev bound | Gaussian actual |
|-----|----------------|----------------|
| 2 | $\geq 75\%$ within $2\sigma$ | $\approx 95.4\%$ |
| 3 | $\geq 88.9\%$ within $3\sigma$ | $\approx 99.7\%$ |
| 4 | $\geq 93.75\%$ within $4\sigma$ | $\approx 99.994\%$ |
| 5 | $\geq 96\%$ within $5\sigma$ | $\approx 99.99994\%$ |

**For AI:** Chebyshev justifies batch normalisation. If activations have $\operatorname{Var}(X) = \sigma^2$, then at most $1/k^2$ of activations lie beyond $k\sigma$ — this quantifies the scale-normalisation benefit.

> **Recall:** The variance $\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$ and its properties were developed fully in [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md). We use it here as a plug-in parameter.

### 2.3 One-Sided Chebyshev (Cantelli's Inequality)

Chebyshev bounds both tails symmetrically. Cantelli's inequality gives a tighter one-sided bound:

**Theorem (Cantelli).** For any $t > 0$:
$$P(X - \mu \geq t) \leq \frac{\sigma^2}{\sigma^2 + t^2}$$

**Proof.** For any $s > 0$, by Markov applied to $(X - \mu + s)^2$:
$$P(X - \mu \geq t) = P(X - \mu + s \geq t + s) \leq \frac{\mathbb{E}[(X-\mu+s)^2]}{(t+s)^2} = \frac{\sigma^2 + s^2}{(t+s)^2}$$
Optimise over $s$: $\partial/\partial s = 0$ gives $s = \sigma^2/t$, yielding $\sigma^2/(\sigma^2 + t^2)$.

For large $t$: Chebyshev gives $\sigma^2/t^2$ per tail (so $2\sigma^2/t^2$ two-sided), while Cantelli gives $\sigma^2/(\sigma^2 + t^2) \approx \sigma^2/t^2$ one-sided — roughly the same. But for small $t$, Cantelli avoids the factor of 2.

### 2.4 Limitations of Moment-Based Bounds

Markov and Chebyshev have polynomial tails. The Gaussian has $P(X \geq 3\sigma) \approx 0.0013$, but Chebyshev gives only $1/9 \approx 0.111$ — an 85× overestimate. For $t = 10\sigma$, Chebyshev gives $0.01$, while the true Gaussian probability is $\approx 10^{-23}$.

**Why the gap?** Chebyshev uses only the first two moments. A distribution could concentrate all its mass at $\mu \pm t$ and match any $(\mu, \sigma^2)$ pair — that's the worst case for Chebyshev. Real distributions with bounded support or thin tails concentrate far more.

**When Chebyshev is the right tool:**
- Distribution-free bounds (don't know the family)
- Heavy-tailed distributions (Pareto, $t$-distribution with low df)
- Quick estimates without assuming sub-Gaussianity

---

## 3. Sub-Gaussian Random Variables

### 3.1 Definition and MGF Condition

Sub-Gaussian random variables are those whose tails decay at least as fast as a Gaussian. The key condition is on the MGF.

**Definition.** A mean-zero random variable $X$ is **$\sigma^2$-sub-Gaussian** if for all $t \in \mathbb{R}$:
$$\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2}$$

The parameter $\sigma^2$ is the **sub-Gaussian parameter** (also called the *proxy variance*). Note: $\sigma^2$ need not equal $\operatorname{Var}(X)$, though $\operatorname{Var}(X) \leq \sigma^2$ always holds (by differentiation at $t=0$).

**Why this condition?** Recall from [§04](../04-Expectation-and-Moments/notes.md) that the MGF of $\mathcal{N}(0, \sigma^2)$ is exactly $e^{\sigma^2 t^2/2}$. The sub-Gaussian condition says the MGF of $X$ is dominated by that of a Gaussian with variance $\sigma^2$. This immediately implies Gaussian-like tail bounds.

**Equivalent characterisations** (for mean-zero $X$):
1. MGF condition: $\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2}$ for all $t$
2. Tail condition: $P(|X| \geq t) \leq 2e^{-t^2/(2\sigma^2)}$ for all $t \geq 0$
3. Moment condition: $\mathbb{E}[X^{2k}] \leq (2k-1)!! \cdot \sigma^{2k}$ for all $k \geq 1$

### 3.2 Examples of Sub-Gaussian Variables

**Gaussian.** $X \sim \mathcal{N}(0, \sigma^2)$ is $\sigma^2$-sub-Gaussian. The MGF equals $e^{\sigma^2 t^2/2}$ exactly — Gaussian is the *equality case*.

**Bounded random variable.** If $X \in [a, b]$ almost surely with $\mathbb{E}[X] = \mu$, then $X - \mu$ is $\frac{(b-a)^2}{4}$-sub-Gaussian. This is **Hoeffding's lemma** — proved in §4.

**Rademacher variable.** $\varepsilon \in \{-1, +1\}$ with equal probability. Then $\mathbb{E}[e^{t\varepsilon}] = \cosh(t) \leq e^{t^2/2}$, so $\varepsilon$ is 1-sub-Gaussian. Rademacher variables appear in Rademacher complexity (§10).

**Bernoulli centered.** $X = \operatorname{Bern}(p) - p \in \{-p, 1-p\}$. Since $X \in [-p, 1-p]$, it is $\frac{1}{4}$-sub-Gaussian by Hoeffding's lemma.

**Non-example.** Heavy-tailed distributions are *not* sub-Gaussian. A $t$-distribution with $\nu$ degrees of freedom has $\mathbb{E}[e^{tX}] = \infty$ for all $t \neq 0$ when $\nu \leq 2$. A Pareto distribution is not sub-Gaussian regardless of parameters.

### 3.3 The Sub-Gaussian Tail Bound

**Theorem.** If $X$ is $\sigma^2$-sub-Gaussian with $\mathbb{E}[X] = 0$, then for all $t \geq 0$:
$$P(X \geq t) \leq e^{-t^2/(2\sigma^2)}$$

**Proof (Chernoff method).** For any $s > 0$:
$$P(X \geq t) = P(e^{sX} \geq e^{st}) \leq \frac{\mathbb{E}[e^{sX}]}{e^{st}} \leq \frac{e^{\sigma^2 s^2/2}}{e^{st}} = e^{\sigma^2 s^2/2 - st}$$

Optimise over $s$: $\partial/\partial s(\sigma^2 s^2/2 - st) = 0 \Rightarrow s^* = t/\sigma^2$. Substituting:
$$P(X \geq t) \leq e^{\sigma^2(t/\sigma^2)^2/2 - t \cdot t/\sigma^2} = e^{t^2/(2\sigma^2) - t^2/\sigma^2} = e^{-t^2/(2\sigma^2)}$$

This proof pattern — Markov + MGF bound + optimise — is the **Chernoff method** and will recur throughout this section.

### 3.4 Closure Under Sums

**Theorem.** If $X_1, \ldots, X_n$ are independent with $X_i$ being $\sigma_i^2$-sub-Gaussian and $\mathbb{E}[X_i] = 0$, then $S = \sum_i X_i$ is $(\sum_i \sigma_i^2)$-sub-Gaussian.

**Proof.** By independence:
$$\mathbb{E}[e^{tS}] = \prod_{i=1}^n \mathbb{E}[e^{tX_i}] \leq \prod_{i=1}^n e^{\sigma_i^2 t^2/2} = e^{(\sum_i \sigma_i^2)t^2/2}$$

**Corollary.** For $n$ i.i.d. $\sigma^2$-sub-Gaussian variables, the sample mean $\bar{X}_n = S/n$ satisfies $P(\bar{X}_n \geq t) \leq e^{-nt^2/(2\sigma^2)}$.

**For AI:** Mini-batch gradient estimation. If each sample's contribution to the gradient is sub-Gaussian with parameter $\sigma^2$, then the mini-batch gradient of size $m$ is $\sigma^2/m$-sub-Gaussian — the noise decreases as $1/m$.

---

## 4. Hoeffding's Inequality

### 4.1 Hoeffding's Lemma

Hoeffding's lemma is the core technical ingredient: it establishes that any bounded, mean-zero random variable is sub-Gaussian.

**Lemma (Hoeffding, 1963).** Let $X$ be a random variable with $\mathbb{E}[X] = 0$ and $X \in [a, b]$ almost surely. Then for all $t \in \mathbb{R}$:
$$\mathbb{E}[e^{tX}] \leq \exp\!\left(\frac{t^2(b-a)^2}{8}\right)$$

In other words, $X$ is $\frac{(b-a)^2}{4}$-sub-Gaussian.

**Proof sketch.** Since $[a, b]$ is a bounded interval and $e^{tx}$ is convex in $x$, we can bound $e^{tx}$ by the chord from $(a, e^{ta})$ to $(b, e^{tb})$:
$$e^{tx} \leq \frac{b-x}{b-a} e^{ta} + \frac{x-a}{b-a} e^{tb}$$

Taking expectations (using $\mathbb{E}[X] = 0$, so $\mathbb{E}[(b-X)/(b-a)] = b/(b-a)$ and $\mathbb{E}[(X-a)/(b-a)] = -a/(b-a)$):
$$\mathbb{E}[e^{tX}] \leq \frac{b}{b-a} e^{ta} - \frac{a}{b-a} e^{tb} = e^{g(u)}$$

where $u = t(b-a)$ and $g(u) = -pu + \log(1 - p + pe^u)$ with $p = -a/(b-a)$. Expanding $g$ via Taylor series around $u=0$ and bounding $g''(u) \leq 1/4$ (achieved at $p = 1/2$) gives $g(u) \leq u^2/8 = t^2(b-a)^2/8$.

**Why $(b-a)^2/8$?** The factor $1/8$ comes from the worst case $p = 1/2$ (centred in the interval). For a $\{0,1\}$ Bernoulli minus its mean, $b - a = 1$ and the sub-Gaussian parameter is $1/4$.

### 4.2 Hoeffding's Inequality

**Theorem (Hoeffding, 1963).** Let $X_1, \ldots, X_n$ be independent random variables with $\mathbb{E}[X_i] = \mu_i$ and $X_i \in [a_i, b_i]$ almost surely. Then for all $t > 0$:
$$P\!\left(\sum_{i=1}^n (X_i - \mu_i) \geq t\right) \leq \exp\!\left(-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$$

**Proof.** Let $Y_i = X_i - \mu_i$ (mean-zero, $Y_i \in [a_i - \mu_i, b_i - \mu_i]$). By Hoeffding's lemma, each $Y_i$ is $\frac{(b_i - a_i)^2}{4}$-sub-Gaussian. By independence and closure under sums (§3.4), $\sum Y_i$ is $\frac{\sum(b_i-a_i)^2}{4}$-sub-Gaussian. Applying the sub-Gaussian tail bound:
$$P\!\left(\sum Y_i \geq t\right) \leq \exp\!\left(-\frac{t^2}{2 \cdot \frac{\sum(b_i-a_i)^2}{4}}\right) = \exp\!\left(-\frac{2t^2}{\sum(b_i-a_i)^2}\right)$$

### 4.3 Two-Sided Hoeffding and Sample Complexity

For i.i.d. $X_i \in [a, b]$ with mean $\mu$, the two-sided version follows by symmetry (apply one-sided to $X$ and $-X$):
$$P(|\bar{X}_n - \mu| \geq t) \leq 2\exp\!\left(-\frac{2nt^2}{(b-a)^2}\right)$$

**Reading the bound:** The probability of a large deviation decays *exponentially* in $n$ and $t^2$. Doubling the sample size squares the exponential factor. Halving the allowed deviation quadruples the required $n$.

### 4.4 Required Sample Size

One of the most practically useful results: how large must $n$ be to guarantee $|\bar{X}_n - \mu| \leq \varepsilon$ with probability at least $1 - \delta$?

Setting $2\exp(-2nt^2/(b-a)^2) \leq \delta$ and solving for $n$:
$$n \geq \frac{(b-a)^2 \log(2/\delta)}{2\varepsilon^2}$$

**Example.** Coin flips ($X_i \in \{0,1\}$, $b - a = 1$): to achieve $\varepsilon = 0.01$ accuracy with $\delta = 0.05$ confidence:
$$n \geq \frac{\log(40)}{0.0002} \approx 18{,}444$$

So roughly 18,500 coin flips are needed to estimate a probability to within $\pm 0.01$ with 95% confidence.

**For AI:** Benchmark evaluation. To measure LLM accuracy to within $\pm 1\%$ at 95% confidence on a binary task ($b-a=1$), Hoeffding requires $n \geq 19{,}122$ examples. This explains why serious LLM evaluations (MMLU, HumanEval, BIG-bench) use thousands of questions.

---

## 5. Chernoff Bounds

### 5.1 The Chernoff Method

The Chernoff method is a general technique for deriving exponential tail bounds using the MGF. It is more powerful than Hoeffding when the MGF has a known closed form.

**The Chernoff method.** For any random variable $X$ and $t > 0$:
$$P(X \geq a) \leq \inf_{s > 0} e^{-sa} \cdot \mathbb{E}[e^{sX}] = \inf_{s > 0} e^{-sa} M_X(s)$$

**Proof.** For any $s > 0$: $P(X \geq a) = P(e^{sX} \geq e^{sa}) \leq e^{-sa} \mathbb{E}[e^{sX}]$ by Markov. Optimise over $s$.

The infimum over $s$ gives the tightest bound. The optimal $s^*$ satisfies $M_X'(s^*)/M_X(s^*) = a$, i.e., the mean of $X$ under the *tilted distribution* equals $a$.

> **Recall:** $M_X(t) = \mathbb{E}[e^{tX}]$ was developed in [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md#5-moment-generating-functions). The key property used here is $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$ for independent $X, Y$.

### 5.2 Chernoff for Bernoulli Sums

Let $X_1, \ldots, X_n \sim \operatorname{Bern}(p)$ i.i.d. and $S = \sum_{i=1}^n X_i \sim \operatorname{Bin}(n, p)$ with mean $\mu = np$.

The MGF of $S$ is $M_S(t) = (1 - p + pe^t)^n$.

**Upper Chernoff bound.** For $\delta > 0$:
$$P(S \geq (1+\delta)\mu) \leq \left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu$$

**Derivation.** Apply the Chernoff method with $a = (1+\delta)\mu$:
$$P(S \geq (1+\delta)\mu) \leq \inf_{s>0} e^{-s(1+\delta)\mu} M_S(s)$$

Substituting $M_S(s) = (1-p+pe^s)^n \leq e^{\mu(e^s-1)}$ (using $1 + x \leq e^x$):
$$\leq \inf_{s>0} \exp(\mu(e^s - 1) - s(1+\delta)\mu) = \exp(\mu(e^s - 1 - s(1+\delta)))$$

Optimising: $s^* = \log(1+\delta)$, giving the stated bound.

### 5.3 Multiplicative Chernoff Form

For $\delta \in (0, 1)$, a simpler but slightly looser form:
$$P(S \geq (1+\delta)\mu) \leq e^{-\mu\delta^2/3}$$

This follows from the inequality $\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu \leq e^{-\mu\delta^2/3}$.

**Comparison with Hoeffding.** Hoeffding (applied to $\operatorname{Bin}(n,p)$ with $b-a=1$) gives:
$$P(S \geq (1+\delta)\mu) = P(\bar{X} \geq p(1+\delta)) \leq \exp(-2n(p\delta)^2)$$

Chernoff gives $\exp(-np^2\delta^2/3) = \exp(-\mu^2\delta^2/(3n))$. For small $p$ (rare events), Chernoff's dependence on $\mu = np$ rather than $n$ makes it far tighter.

### 5.4 Lower Tail Chernoff

For the lower tail, for $\delta \in (0, 1)$:
$$P(S \leq (1-\delta)\mu) \leq e^{-\mu\delta^2/2}$$

Note the factor of $1/2$ vs $1/3$ in the exponent — the lower tail is slightly tighter.

**Application: balls into bins.** When $n$ items are assigned to $m$ bins uniformly at random, the expected load per bin is $\mu = n/m$. Chernoff bounds control the maximum load — crucial for hashing and load-balancing proofs in distributed systems.

---

## 6. Bernstein's Inequality

### 6.1 Sub-Exponential Random Variables

Some random variables have heavier tails than sub-Gaussian but still lighter than arbitrary — these are sub-exponential.

**Definition.** A mean-zero random variable $X$ is **sub-exponential** with parameters $(\nu^2, b)$ if:
$$\mathbb{E}[e^{tX}] \leq e^{\nu^2 t^2/2} \quad \text{for all } |t| \leq 1/b$$

Sub-exponential tails decay as $e^{-t/b}$ (exponential), slower than Gaussian $e^{-t^2/(2\sigma^2)}$.

**Examples:**
- Exponential$(1)$: sub-exponential with $(\nu^2, b) = (1, 1)$
- $\chi^2_k$: sub-exponential (sum of squared Gaussians)
- Any bounded variable is also sub-exponential

### 6.2 Bernstein's Inequality

**Theorem (Bernstein, 1924/1946).** Let $X_1, \ldots, X_n$ be independent mean-zero variables with $|X_i| \leq c$ and $\sum_{i=1}^n \mathbb{E}[X_i^2] = \nu^2$. Then for all $t > 0$:
$$P\!\left(\sum_{i=1}^n X_i \geq t\right) \leq \exp\!\left(-\frac{t^2/2}{\nu^2 + ct/3}\right)$$

**Intuition.** This interpolates between two regimes:
- **Small $t$ ($t \ll \nu^2/c$):** Denominator $\approx \nu^2$, gives $e^{-t^2/(2\nu^2)}$ — sub-Gaussian with variance $\nu^2$
- **Large $t$ ($t \gg \nu^2/c$):** Denominator $\approx ct/3$, gives $e^{-3t/(2c)}$ — exponential tail

### 6.3 Bernstein vs Hoeffding

For $n$ i.i.d. variables with $|X_i| \leq c$ and sample variance $s^2 = \frac{1}{n}\sum \mathbb{E}[X_i^2]$:

| Bound | Formula | Better when |
|-------|---------|------------|
| Hoeffding | $\exp(-2nt^2/(2c)^2) = \exp(-nt^2/(2c^2))$ | No variance info |
| Bernstein | $\exp(-nt^2/2/(ns^2 + ct/3))$ | $s^2 \ll c^2$ (small variance) |

**When Bernstein wins.** If $s^2 \ll c^2$ (data has small variance despite large range), Bernstein gives $\exp(-nt^2/(2s^2))$ for small $t$, while Hoeffding gives $\exp(-nt^2/(2c^2))$. Since $s^2 \ll c^2$, Bernstein is exponentially tighter.

**Example.** Suppose $X_i \in [-1, 1]$ but $\operatorname{Var}(X_i) = 0.01$. Hoeffding uses range $(b-a)^2 = 4$, while Bernstein uses variance $0.01$ — a 400× improvement in the exponent for small $t$.

### 6.4 Application to Gradient Noise

In SGD, the stochastic gradient $\hat{g}_t = \frac{1}{m}\sum_{i \in B_t} \nabla \ell(w_t; x_i)$ estimates the true gradient $g_t = \nabla \mathcal{L}(w_t)$.

If each $\nabla\ell(w; x)$ is bounded: $\|\nabla\ell(w; x)\|_2 \leq G$, and the sample gradient variance is $\sigma_g^2$, then Bernstein bounds:
$$P\!\left(\|\hat{g}_t - g_t\|_2 \geq t\right) \leq d \cdot \exp\!\left(-\frac{mt^2/2}{d\sigma_g^2 + Gt/3}\right)$$

where the factor $d$ comes from a union bound over coordinates. This bound shows why variance reduction methods (SVRG, SAGA) that reduce $\sigma_g^2$ improve convergence more dramatically than reducing $G$ alone.

---

## 7. McDiarmid's Inequality

### 7.1 Bounded Differences Condition

McDiarmid's inequality generalises Hoeffding from sums to general functions.

**Definition (Bounded Differences).** A function $f: \mathcal{X}^n \to \mathbb{R}$ satisfies the **bounded differences condition** with constants $c_1, \ldots, c_n > 0$ if for all $i \in [n]$ and all $x_1, \ldots, x_n, x_i' \in \mathcal{X}$:
$$|f(x_1, \ldots, x_i, \ldots, x_n) - f(x_1, \ldots, x_i', \ldots, x_n)| \leq c_i$$

Intuitively: changing the $i$-th coordinate by any amount changes $f$ by at most $c_i$.

### 7.2 McDiarmid's Inequality

**Theorem (McDiarmid, 1989).** Let $X_1, \ldots, X_n$ be independent random variables and let $f$ satisfy the bounded differences condition with constants $c_1, \ldots, c_n$. Then for all $t > 0$:
$$P(f(X_1, \ldots, X_n) - \mathbb{E}[f] \geq t) \leq \exp\!\left(-\frac{2t^2}{\sum_{i=1}^n c_i^2}\right)$$

**Proof sketch (martingale method).** Define the Doob martingale $Z_k = \mathbb{E}[f \mid X_1, \ldots, X_k]$ with $Z_0 = \mathbb{E}[f]$ and $Z_n = f$. The differences $D_k = Z_k - Z_{k-1}$ satisfy $|D_k| \leq c_k$ by the bounded differences assumption. Apply Azuma's inequality (concentration for bounded martingale differences) to get the result.

### 7.3 Relationship to Hoeffding

Hoeffding's inequality is a special case of McDiarmid's. For $f(X_1, \ldots, X_n) = \frac{1}{n}\sum_{i=1}^n X_i$ with $X_i \in [a_i, b_i]$, changing $X_i$ changes $f$ by at most $(b_i - a_i)/n$, so $c_i = (b_i - a_i)/n$. Then:
$$\sum_i c_i^2 = \frac{\sum_i (b_i - a_i)^2}{n^2}$$

Substituting into McDiarmid recovers exactly Hoeffding's bound.

### 7.4 Applications to ML Stability

**Training loss with missing data.** Let $\mathcal{D} = \{z_1, \ldots, z_n\}$ be a training set and $\hat{R}(\mathcal{D}) = \frac{1}{n}\sum_{i=1}^n \ell(h, z_i)$ the empirical risk with $\ell \in [0, 1]$. Swapping one training example changes $\hat{R}$ by at most $1/n$. McDiarmid gives:
$$P(|\hat{R} - \mathbb{E}[\hat{R}]| \geq t) \leq 2\exp(-2nt^2)$$

**Dropout.** With dropout probability $p$, each neuron's activation $a_i$ is scaled by $\mathbf{1}[\text{kept}] / (1-p)$. The network output $f(z_1, \ldots, z_d)$ changes by at most $c_i = |w_i| \cdot a_i^{\max} / (1-p)$ when the $i$-th neuron is toggled. McDiarmid bounds the output variance from dropout noise.

**Data augmentation.** If augmenting a single training example can change the training loss by at most $c$, McDiarmid bounds the sensitivity of the trained model to augmentation choices.

---

## 8. The Union Bound and Covering Arguments

### 8.1 Union Bound

**Lemma (Boole's / Union Bound).** For any events $A_1, \ldots, A_k$ (not necessarily independent):
$$P\!\left(\bigcup_{i=1}^k A_i\right) \leq \sum_{i=1}^k P(A_i)$$

**Proof.** By induction: $P(A \cup B) = P(A) + P(B) - P(A \cap B) \leq P(A) + P(B)$.

The union bound is exact when events are disjoint and pessimistic otherwise. Despite its looseness, it is indispensable because it requires no assumption about dependence between events.

**For AI:** The union bound appears in virtually every multi-class or multi-hypothesis analysis. PAC theory for a finite hypothesis class $\mathcal{H}$ asks for the probability that *any* $h \in \mathcal{H}$ is bad — union bound over all $|\mathcal{H}|$ hypotheses.

### 8.2 Multiple Hypothesis Testing

Consider $m$ statistical tests each with individual significance level $\alpha$. If we want the probability that *at least one* false positive occurs to be at most $\delta$, the Bonferroni correction sets each test at level $\delta/m$.

**In ML:** When training $m$ models and reporting the best, the union bound says the best result can be a false positive with probability up to $m\alpha$. This is the multiple comparison problem underlying claims like "our method beats 10 baselines on 5 datasets."

**Balancing union bound and concentration.** For $k$ bad events $A_i$ each with $P(A_i) \leq \varepsilon$:
$$P(\exists i: A_i) \leq k\varepsilon$$

To make this $\leq \delta$: need each $P(A_i) \leq \delta/k$. In PAC theory, this costs an extra $\log k$ in the required sample size.

### 8.3 Covering Numbers and ε-Nets

The union bound can only be applied to finite collections of events. For continuous hypothesis spaces (e.g., all linear classifiers, all neural networks), we need to discretise first.

**Definition.** An **$\varepsilon$-net** of a set $\mathcal{H}$ under metric $d$ is a finite set $\mathcal{N}_\varepsilon \subseteq \mathcal{H}$ such that for every $h \in \mathcal{H}$, there exists $\hat{h} \in \mathcal{N}_\varepsilon$ with $d(h, \hat{h}) \leq \varepsilon$.

The **covering number** $\mathcal{N}(\mathcal{H}, d, \varepsilon)$ is the size of the smallest $\varepsilon$-net.

**Covering number of a ball.** The $\ell_2$-ball $\{\mathbf{w}: \|\mathbf{w}\|_2 \leq R\}$ in $\mathbb{R}^d$ has covering number:
$$\mathcal{N}(\mathcal{B}_R, \ell_2, \varepsilon) \leq \left(\frac{2R}{\varepsilon} + 1\right)^d \leq \left(\frac{3R}{\varepsilon}\right)^d$$

This grows polynomially in $1/\varepsilon$ but exponentially in $d$ — the **curse of dimensionality** for covering.

### 8.4 The Net Argument

The standard approach for continuous hypothesis classes:

1. Build an $\varepsilon$-net $\mathcal{N}_\varepsilon$ with $|\mathcal{N}_\varepsilon| = \mathcal{N}(\mathcal{H}, d, \varepsilon)$
2. Apply concentration + union bound over the net: control $\sup_{h \in \mathcal{N}_\varepsilon} |\hat{R}(h) - R(h)|$
3. Use Lipschitz continuity to extend from net to all of $\mathcal{H}$

The final bound pays $\log \mathcal{N}(\mathcal{H}, d, \varepsilon)$ extra compared to a single hypothesis, plus the approximation error $\varepsilon$ from the net.

**Johnson-Lindenstrauss Lemma.** As an application of covering, if $m = O(\log(n)/\varepsilon^2)$ random projections are used, $n$ points in $\mathbb{R}^d$ embed into $\mathbb{R}^m$ with $(1\pm\varepsilon)$ distortion of all pairwise distances, with high probability. This is proved using the union bound over all $\binom{n}{2}$ pairs.

---

## 9. PAC Learning and Generalisation Bounds

### 9.1 The PAC Framework

The **Probably Approximately Correct (PAC)** framework, introduced by Valiant (1984), provides a formal model for machine learning under uncertainty.

**Setup:** A learner observes $n$ i.i.d. examples $\{(x^{(i)}, y^{(i)})\}$ from an unknown distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$. The learner selects a hypothesis $h$ from a hypothesis class $\mathcal{H}$.

**True risk:** $R(h) = P_{(x,y) \sim \mathcal{D}}(h(x) \neq y) = \mathbb{E}[\mathbf{1}[h(x) \neq y]]$

**Empirical risk:** $\hat{R}(h) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[h(x^{(i)}) \neq y^{(i)}]$

**PAC guarantee:** A learning algorithm $\mathcal{A}$ is **PAC-learning** $\mathcal{H}$ if for any $\varepsilon, \delta > 0$ and any distribution $\mathcal{D}$, given $n \geq n_0(\varepsilon, \delta)$ examples, with probability at least $1-\delta$:
$$R(\mathcal{A}(\mathcal{D})) \leq \min_{h \in \mathcal{H}} R(h) + \varepsilon$$

The **sample complexity** $n_0(\varepsilon, \delta)$ is the central quantity: how many examples are needed?

### 9.2 Finite Hypothesis Class

**Theorem.** Let $\mathcal{H}$ be a finite hypothesis class with $|\mathcal{H}| < \infty$. For any ERM algorithm (minimising $\hat{R}(h)$) and any $\varepsilon, \delta > 0$, if:
$$n \geq \frac{\log(|\mathcal{H}|/\delta)}{2\varepsilon^2}$$
then with probability at least $1-\delta$, the selected hypothesis $\hat{h}$ satisfies $R(\hat{h}) \leq \min_{h^*} R(h^*) + \varepsilon$ (in the realisable case with $\hat{R}(h^*) = 0$).

**Proof.** Consider the event "bad $h$": $B_h = \{R(h) > \varepsilon \text{ but } \hat{R}(h) = 0\}$.

By Hoeffding: $P(B_h) \leq P(\hat{R}(h) = 0 \mid R(h) > \varepsilon) \leq (1-\varepsilon)^n \leq e^{-n\varepsilon}$.

By union bound: $P(\exists h \in \mathcal{H}: B_h) \leq |\mathcal{H}| \cdot e^{-n\varepsilon}$.

Setting this $\leq \delta$ and solving: $n \geq \frac{\log(|\mathcal{H}|/\delta)}{\varepsilon}$. The factor $\varepsilon^2$ version covers the agnostic (non-realisable) case.

### 9.3 Uniform Convergence

**Theorem (Uniform Convergence).** For finite $\mathcal{H}$ and losses in $[0, 1]$, for $n \geq \frac{\log(2|\mathcal{H}|/\delta)}{2\varepsilon^2}$:
$$P\!\left(\sup_{h \in \mathcal{H}} |\hat{R}(h) - R(h)| > \varepsilon\right) \leq \delta$$

**Proof.** Fix any $h \in \mathcal{H}$. By Hoeffding: $P(|\hat{R}(h) - R(h)| > \varepsilon) \leq 2e^{-2n\varepsilon^2}$.

By union bound over all $|\mathcal{H}|$ hypotheses: $P(\sup_h |\hat{R}(h) - R(h)| > \varepsilon) \leq 2|\mathcal{H}| e^{-2n\varepsilon^2} \leq \delta$.

**Consequence:** If uniform convergence holds, then for ERM $\hat{h} = \arg\min_h \hat{R}(h)$:
$$R(\hat{h}) \leq \hat{R}(\hat{h}) + \varepsilon \leq \hat{R}(h^*) + \varepsilon \leq R(h^*) + 2\varepsilon$$

The ERM generalises to within $2\varepsilon$ of the best hypothesis.

### 9.4 VC Dimension

For infinite hypothesis classes, we need a different complexity measure.

**Definition.** A hypothesis class $\mathcal{H}$ **shatters** a set $C = \{x_1, \ldots, x_k\}$ if for every labelling $y \in \{0,1\}^k$, there exists $h \in \mathcal{H}$ with $h(x_i) = y_i$ for all $i$.

**VC dimension** $d_{VC}(\mathcal{H})$ is the size of the largest set shattered by $\mathcal{H}$.

**Examples:**
- Threshold classifiers on $\mathbb{R}$: $d_{VC} = 1$ (can shatter any single point but not two)
- Linear classifiers (halfspaces) in $\mathbb{R}^d$: $d_{VC} = d + 1$
- Degree-$k$ polynomial classifiers in $\mathbb{R}^d$: $d_{VC} = \binom{d+k}{k}$
- Circles in $\mathbb{R}^2$: $d_{VC} = 3$
- Neural networks: grows with parameter count, approximately linearly

**Sauer-Shelah Lemma.** The growth function $\Pi_\mathcal{H}(n) = \max_{x_1,\ldots,x_n} |\{(h(x_1),\ldots,h(x_n)): h \in \mathcal{H}\}|$ satisfies:
$$\Pi_\mathcal{H}(n) \leq \sum_{i=0}^{d} \binom{n}{i} \leq \left(\frac{en}{d}\right)^d$$

for $n \geq d = d_{VC}(\mathcal{H})$. This says: an $n$-point set can be labelled in at most $(en/d)^d$ ways by $\mathcal{H}$, even if $|\mathcal{H}| = \infty$.

### 9.5 Generalisation Bound with VC Dimension

**Theorem (VC Generalisation Bound).** For any $\delta > 0$, with probability at least $1-\delta$ over $n$ i.i.d. training examples, every $h \in \mathcal{H}$ satisfies:
$$R(h) \leq \hat{R}(h) + \sqrt{\frac{d(\log(2n/d) + 1) + \log(4/\delta)}{2n}}$$

where $d = d_{VC}(\mathcal{H})$.

**Sample complexity.** Setting the right-hand side excess to $\varepsilon$:
$$n = O\!\left(\frac{d \log(1/(\varepsilon\delta)) + \log(1/\delta)}{\varepsilon^2}\right) = O\!\left(\frac{d + \log(1/\delta)}{\varepsilon^2}\right)$$

The $\log(|\mathcal{H}|)$ from the finite case is replaced by $d_{VC}$. For a $d$-dimensional linear classifier ($d_{VC} = d+1$), need $n = O(d/\varepsilon^2)$ — linear in dimension, not exponential.

**For AI (2026):** Modern transformers have billions of parameters, implying enormous VC dimension. Yet they generalise. This "paradox" is partly resolved by:
1. **Implicit regularisation:** gradient descent finds minimum-norm solutions
2. **Overparameterisation:** interpolation threshold, double descent (covered in §04)
3. **PAC-Bayes:** data-dependent bounds that are tighter for specific algorithms

---

## 10. Rademacher Complexity

### 10.1 Definition

Rademacher complexity measures the richness of a function class on specific data, making it data-dependent and often tighter than VC bounds.

**Definition.** Given a sample $S = (x^{(1)}, \ldots, x^{(n)})$ and a function class $\mathcal{F}$, the **empirical Rademacher complexity** is:
$$\hat{\mathfrak{R}}_S(\mathcal{F}) = \mathbb{E}_{\boldsymbol{\sigma}}\!\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(x^{(i)})\right]$$

where $\boldsymbol{\sigma} = (\sigma_1, \ldots, \sigma_n)$ with $\sigma_i \sim \operatorname{Uniform}(\{-1, +1\})$ i.i.d. (Rademacher variables).

The **Rademacher complexity** is $\mathfrak{R}_n(\mathcal{F}) = \mathbb{E}_S[\hat{\mathfrak{R}}_S(\mathcal{F})]$.

**Interpretation.** $\hat{\mathfrak{R}}_S(\mathcal{F})$ measures how well $\mathcal{F}$ can fit random $\pm 1$ labels on the training points. If the class is rich enough to fit any labelling, $\hat{\mathfrak{R}}_S \approx 1$. If it can only fit structured labels, $\hat{\mathfrak{R}}_S \approx 0$.

### 10.2 Rademacher Generalisation Bound

**Theorem.** For any function class $\mathcal{F}$ with values in $[0, 1]$, with probability at least $1-\delta$:
$$R(f) \leq \hat{R}(f) + 2\mathfrak{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}$$

for all $f \in \mathcal{F}$ simultaneously.

**Comparison to VC.** The VC bound has $\sqrt{d/n}$; the Rademacher bound has $\mathfrak{R}_n(\mathcal{F})$. For specific data distributions where the function class is "effectively smaller," Rademacher can be significantly tighter.

### 10.3 Rademacher of Linear Classifiers

For linear classifiers $\mathcal{F} = \{\mathbf{x} \mapsto \langle \mathbf{w}, \mathbf{x} \rangle : \|\mathbf{w}\|_2 \leq B\}$ on data with $\|\mathbf{x}^{(i)}\|_2 \leq C$:

$$\hat{\mathfrak{R}}_S(\mathcal{F}) \leq \frac{BC}{\sqrt{n}} \cdot \frac{\|\hat{\Sigma}\|_F}{B} = \frac{C\sqrt{\operatorname{tr}(\hat{\Sigma})}}{B\sqrt{n}}$$

More precisely, using Cauchy-Schwarz:
$$\hat{\mathfrak{R}}_S(\mathcal{F}) \leq \frac{B}{n}\mathbb{E}_\sigma\!\left[\left\|\sum_{i=1}^n \sigma_i \mathbf{x}^{(i)}\right\|_2\right] \leq \frac{B}{\sqrt{n}} \cdot \frac{\sqrt{\sum_i \|\mathbf{x}^{(i)}\|_2^2}}{n^{1/2}} \leq \frac{BC}{\sqrt{n}}$$

This is $O(1/\sqrt{n})$, independent of the ambient dimension $d$! This explains why linear classifiers generalise well even in high dimensions, as long as the norm $\|\mathbf{w}\|_2$ is controlled.

### 10.4 Rademacher vs VC

| Property | VC Dimension | Rademacher Complexity |
|----------|-------------|----------------------|
| Distribution | Distribution-free | Data-dependent |
| Calculation | Combinatorial | Expectation over labels |
| Estimate | Hard (often NP-hard) | MC approximation possible |
| Tightness | Worst-case | Often tighter in practice |
| Handles regression | No (classification) | Yes (arbitrary losses) |
| Modern NNs | Vacuous (too large $d$) | Can be finite for sparse/low-rank models |

**For AI:** Rademacher complexity gives finite (non-vacuous) generalisation bounds for attention heads with bounded query/key norm, explaining why transformers with weight decay generalise despite huge parameter counts.


---

## 11. ML Applications

### 11.1 SGD and Gradient Concentration

In stochastic gradient descent, at each step $t$ we compute a mini-batch gradient:
$$\hat{g}_t = \frac{1}{m}\sum_{i \in B_t} \nabla_\theta \ell(\theta_t; x^{(i)})$$

where $B_t$ is a random mini-batch of size $m$. The true gradient is $g_t = \nabla_\theta \mathcal{L}(\theta_t) = \mathbb{E}[\nabla_\theta \ell]$.

**Concentration of gradient norm.** Assuming $\|\nabla\ell(\theta; x)\|_2 \leq G$ almost surely, by Hoeffding applied coordinate-wise and union bound:
$$P(\|\hat{g}_t - g_t\|_2 \geq \varepsilon) \leq 2d \cdot \exp\!\left(-\frac{m\varepsilon^2}{2G^2}\right)$$

**Implication for learning rate.** For SGD to make progress, we need the gradient noise $\|\hat{g}_t - g_t\|_2$ to be smaller than the signal. Concentration tells us: with probability $1-\delta$, the noise is bounded by $G\sqrt{2\log(2d/\delta)/m}$. This justifies the learning rate scaling $\eta \propto 1/\sqrt{t}$ in theory.

**Variance reduction.** Methods like SVRG (Stochastic Variance-Reduced Gradient) periodically compute the full gradient, reducing $G$ to $G/\sqrt{n}$. Bernstein's inequality quantifies the benefit: convergence can be linear (not just $1/\sqrt{t}$) when variance is controlled.

### 11.2 Confidence Intervals via Hoeffding

Suppose we evaluate an LLM on a benchmark: out of $n$ binary questions (correct/incorrect), we observe accuracy $\hat{p} = k/n$.

By Hoeffding's inequality (with $X_i \in \{0,1\}$, $b-a = 1$), with probability at least $1-\delta$:
$$|\hat{p} - p| \leq \sqrt{\frac{\log(2/\delta)}{2n}}$$

This gives the **Hoeffding confidence interval**: $p \in \left[\hat{p} - \sqrt{\frac{\log(2/\delta)}{2n}}, \hat{p} + \sqrt{\frac{\log(2/\delta)}{2n}}\right]$.

**Numerical example.** For $n = 1000$ questions and $\delta = 0.05$:
$$\text{CI half-width} = \sqrt{\frac{\log(40)}{2000}} \approx \sqrt{\frac{3.69}{2000}} \approx 0.043$$

So a measured accuracy of 73.2% on 1000 questions gives 95% CI $[69.0\%, 77.4\%]$ — a $\pm 4.2\%$ interval. This explains why claiming "Model A (73.2%) outperforms Model B (72.8%)" on 1000 questions has no statistical support.

### 11.3 Random Features and Kernel Approximation

The Rahimi-Recht random features method approximates the RBF kernel $k(x, x') = e^{-\|x-x'\|^2/(2\sigma^2)}$ as:
$$k(x, x') \approx \frac{1}{D}\sum_{j=1}^D \phi_j(x)\phi_j(x'), \quad \phi_j(x) = \sqrt{2}\cos(\omega_j^\top x + b_j)$$

where $\omega_j \sim \mathcal{N}(0, I/\sigma^2)$ and $b_j \sim \mathcal{U}[0, 2\pi]$.

**How many features?** The approximation error is bounded by Hoeffding:
$$P\!\left(\left|\frac{1}{D}\sum_{j=1}^D z_j - k(x,x')\right| \geq \varepsilon\right) \leq 2e^{-D\varepsilon^2/2}$$

where $z_j = \phi_j(x)\phi_j(x') \in [-1, 1]$. For $\varepsilon = 0.01$ and $\delta = 0.01$: $D \geq 2\log(2/0.01)/0.01^2 \approx 92{,}000$.

In practice, $D = 1000$--$10000$ suffices with $\varepsilon \approx 0.1$, suggesting the bound is conservative. This is common: worst-case bounds are loose, actual concentrations are tighter.

### 11.4 Concentration in High Dimensions

High-dimensional geometry exhibits surprising concentration phenomena. For $\mathbf{x} \sim \mathcal{N}(0, I_d)$:

**Chi-squared concentration.** $\|\mathbf{x}\|_2^2 \sim \chi^2_d$ with mean $d$ and variance $2d$. Bernstein gives:
$$P\!\left(\left|\frac{\|\mathbf{x}\|_2^2}{d} - 1\right| \geq t\right) \leq 2\exp\!\left(-d \cdot \min\!\left(\frac{t^2}{4}, \frac{t}{4}\right)\right)$$

So $\|\mathbf{x}\|_2 / \sqrt{d} \to 1$ in probability — **all Gaussian vectors in high dimensions have nearly the same norm $\sqrt{d}$.**

**Concentration of inner products.** For independent $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(0, I_d/d)$ (unit-variance):
$$P(|\langle \mathbf{x}, \mathbf{y} \rangle| \geq t) \leq 2e^{-dt^2/2}$$

Nearly-orthogonal random vectors: with $d = 512$ (typical embedding dimension), random vectors have $|\langle \mathbf{x}, \mathbf{y}\rangle| < 0.1$ with very high probability.

**For AI (attention).** In transformers with $d_k = 64$ (key dimension), random query-key inner products concentrate around 0. This justifies the $1/\sqrt{d_k}$ scaling in attention: without it, softmax inputs would have variance $d_k$, causing near-zero gradients through the softmax saturation.

> **Preview: Law of Large Numbers and CLT**
>
> The concentration results above provide finite-sample bounds. The LLN says $\bar{X}_n \to \mu$ as $n \to \infty$; the CLT says the deviation $\sqrt{n}(\bar{X}_n - \mu)/\sigma$ converges to $\mathcal{N}(0,1)$. These are the asymptotic counterparts of Hoeffding's inequality.
>
> → *Full treatment: [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md)*

---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Applying Markov to a signed random variable | Markov requires $X \geq 0$; for signed $X$, $P(X \geq t)$ can exceed $\mathbb{E}[X]/t$ | Apply Markov to $|X|$ or $(X - \mu)^2$ |
| 2 | Using Chebyshev for exponential tails | Chebyshev gives $1/k^2$ decay, vastly looser than sub-Gaussian $e^{-k^2/2}$ | If data is bounded or Gaussian, use Hoeffding/Chernoff |
| 3 | Forgetting the factor of 2 in two-sided Hoeffding | One-sided: $e^{-2nt^2/(b-a)^2}$; two-sided has factor 2 in front | Always write "two-sided" explicitly and include the factor |
| 4 | Confusing sub-Gaussian parameter with variance | Sub-Gaussian parameter $\sigma^2$ satisfies $\operatorname{Var}(X) \leq \sigma^2$ but is not equal | $\operatorname{Var}(X) = M''(0)$; sub-Gaussian parameter can be larger |
| 5 | Applying union bound over infinite classes directly | Union bound only works for countable (finite/countably infinite) collections | Use $\varepsilon$-net covering argument first, then union bound over the net |
| 6 | Computing VC dimension by counting parameters | VC dimension is not the number of parameters; depends on the function class structure | For linear classifiers in $\mathbb{R}^d$: $d_{VC} = d+1$, not number of weights |
| 7 | Ignoring the $\log(1/\delta)$ term in sample complexity | Setting $\delta = 0.01$ adds $\log(100) \approx 4.6$ — small, but not negligible | Always compute full bound including $\log(1/\delta)$ |
| 8 | Claiming Chernoff applies to dependent variables | Chernoff requires independence (product of MGFs) | For dependent variables, use McDiarmid or martingale concentration |
| 9 | Using Hoeffding when variance is known to be small | Bernstein would give an exponentially tighter bound for small-variance data | Check if $\operatorname{Var}(X_i) \ll (b-a)^2/4$; if so, use Bernstein |
| 10 | Interpreting VC generalisation bounds as practically tight | VC bounds are often vacuous for deep networks; describe worst-case behaviour | Use PAC-Bayes or Rademacher bounds, which can be data-dependent and tighter |

---

## 13. Exercises

**Exercise 1 ★** (Markov and Chebyshev)
For $X \sim \operatorname{Exp}(1)$ and $Y \sim \mathcal{N}(0,1)$:
(a) Compute $P(X \geq 3)$ exactly and via Markov's inequality. What is the ratio?
(b) Compute $P(|Y| \geq 3)$ exactly and via Chebyshev. What is the ratio?
(c) For which $k$ does Chebyshev give $P(|Y| \geq k) \leq 0.05$? Compare to the exact Gaussian answer.
(d) Verify both bounds numerically with $N = 10^6$ samples.

**Exercise 2 ★** (Hoeffding Sample Complexity)
A spam filter classifies emails. On $n$ test emails, it achieves accuracy $\hat{p}$.
(a) Using Hoeffding, find the smallest $n$ such that with 99% confidence, $|\hat{p} - p| \leq 0.02$.
(b) How does the required $n$ change if we only require $|\hat{p} - p| \leq 0.05$?
(c) If $n = 500$ and $\hat{p} = 0.92$, give the 95% Hoeffding confidence interval for $p$.
(d) Compare the Hoeffding CI to the Normal-approximation CI $\hat{p} \pm 1.96\sqrt{\hat{p}(1-\hat{p})/n}$.

**Exercise 3 ★** (Chernoff Bounds for Binomial)
Let $S \sim \operatorname{Bin}(n, p)$ with $n = 1000$, $p = 0.1$ (so $\mu = 100$).
(a) Compute $P(S \geq 130)$ exactly and via the multiplicative Chernoff bound ($\delta = 0.3$).
(b) Compute $P(S \geq 130)$ via Hoeffding. Compare to Chernoff — which is tighter?
(c) Compute $P(S \leq 70)$ via the lower-tail Chernoff bound and compare to the exact value.
(d) Plot all three bounds (Hoeffding, upper Chernoff, lower Chernoff) as a function of the threshold.

**Exercise 4 ★★** (Bernstein: Gradient Noise in SGD)
Assume each sample gradient $g_i \in [-G, G]^d$ with $\mathbb{E}[g_i] = g$ and per-coordinate variance $\sigma_k^2$.
(a) Apply Bernstein (with union bound) to bound $P(\|\hat{g} - g\|_\infty \geq \varepsilon)$ for mini-batch size $m$.
(b) Compare to the Hoeffding-based bound. Under what condition does Bernstein win?
(c) For $G = 1$, $d = 100$, $\sigma_k^2 = 0.01$, find the $m$ needed for $P(\|\hat{g} - g\|_\infty \geq 0.1) \leq 0.05$ via both bounds.
(d) Verify numerically: generate synthetic gradients and measure empirical exceedance probability.

**Exercise 5 ★★** (McDiarmid on Empirical Risk)
Let $\hat{R}(h, \mathcal{D}) = \frac{1}{n}\sum_{i=1}^n \ell(h, z_i)$ with $\ell \in [0, 1]$ and $z_i$ i.i.d.
(a) Show that swapping one training example changes $\hat{R}$ by at most $1/n$.
(b) Apply McDiarmid's inequality to bound $P(|\hat{R} - \mathbb{E}[\hat{R}]| \geq t)$.
(c) Compare this to the direct Hoeffding application. Are they the same? Why?
(d) What if $\ell \in [0, L]$ instead of $[0, 1]$? How does the bound scale?

**Exercise 6 ★★** (PAC Bounds for Finite Class)
Consider $\mathcal{H} = \{h_1, \ldots, h_{1000}\}$ with 1000 decision rules for binary classification.
(a) How many training examples are needed for uniform convergence with $\varepsilon = 0.05$, $\delta = 0.05$?
(b) ERM achieves training error 0. Using the PAC bound, what is the worst-case true error with 95% confidence?
(c) If we use $n = 5000$ examples and observe $\hat{R}(\hat{h}) = 0.03$, bound $R(\hat{h})$ with 99% confidence.
(d) How does the bound change if $|\mathcal{H}| = 10^6$ (e.g., a lookup table over binary features)?

**Exercise 7 ★★★** (VC Dimension and Generalisation)
Consider linear classifiers in $\mathbb{R}^d$: $\mathcal{H} = \{\operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b): \mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}\}$.
(a) Show by construction that $d+1$ points can be shattered (exhibit a configuration).
(b) Show that no $d+2$ points in general position can be shattered.
(c) Write the VC generalisation bound for $d = 100$, $n = 10000$, $\delta = 0.05$.
(d) Compare to the Rademacher bound for the same class with $\|\mathbf{w}\|_2 \leq 1$ and $\|\mathbf{x}\|_2 \leq 1$.

**Exercise 8 ★★★** (Rademacher Complexity: Monte Carlo Estimation)
Given a dataset $\{x^{(1)}, \ldots, x^{(n)}\} \subset \mathbb{R}^d$ and linear function class $\mathcal{F} = \{\mathbf{x} \mapsto \mathbf{w}^\top \mathbf{x}: \|\mathbf{w}\|_2 \leq 1\}$:
(a) Show that $\hat{\mathfrak{R}}_S(\mathcal{F}) = \frac{1}{n}\mathbb{E}_\sigma\!\left[\left\|\sum_i \sigma_i x^{(i)}\right\|_2\right]$.
(b) Implement a Monte Carlo estimator using $T = 1000$ random sign vectors $\sigma$.
(c) On the MNIST training set (or a synthetic dataset), compute $\hat{\mathfrak{R}}_S$ and the resulting generalisation bound.
(d) How does $\hat{\mathfrak{R}}_S$ change as you vary $n$? Verify the $O(1/\sqrt{n})$ scaling.

---

## 14. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Connection | Current Importance |
|---------|-----------------|-------------------|
| Hoeffding confidence intervals | LLM benchmark evaluation: how many test questions for reliable rankings? | Critical — used implicitly in all leaderboard comparisons (MMLU, HELM, LMSYS) |
| PAC-Bayes bounds | Tightest known generalisation bounds for overparameterised models; connects to flat minima and sharpness-aware minimisation (SAM) | Active research frontier, 2023-2026 |
| VC dimension | Classical foundation; vacuous for LLMs with $10^9$+ parameters | Conceptually important; practically replaced by norm-based bounds |
| Rademacher complexity | Data-dependent bounds; finite for norm-constrained transformers | Used in theoretical analysis of attention, LoRA rank bounds |
| McDiarmid stability | Algorithmic stability theory: output doesn't change much when one training point changes | Foundation of differential privacy; relevant to RLHF data sensitivity |
| Concentration in high dims | $\sqrt{d_k}$ scaling in attention; JL lemma for random projections in retrieval | Directly explains transformer architectural choices (FlashAttention, MLA) |
| Chernoff for random graphs | Theoretical analysis of dropout, random initialisation, random feature networks | Relevant to lottery ticket hypothesis, neural architecture search |
| Union bound + covering | Uniform convergence for function classes; foundation of all finite-sample learning theory | Every theoretical ML paper uses this framework |
| Sub-Gaussian gradients | Convergence theory for Adam, SGD; required sample complexity for fine-tuning | Used in LoRA theoretical analysis and RLHF convergence guarantees |
| Bernstein vs Hoeffding | Variance-aware bounds; relevant when gradient variance is small (near optima) | Foundation of variance reduction methods (SVRG, SAG, SAGA) |

---

## 15. Conceptual Bridge

**Looking back.** This section builds on §04 Expectation and Moments, which established $\mathbb{E}[X]$, $\operatorname{Var}(X)$, and the MGF $M_X(t) = \mathbb{E}[e^{tX}]$. The MGF is the critical bridge: Hoeffding's lemma uses convexity of $e^{tx}$; the Chernoff method optimises over $M_X(s)$; Bernstein's inequality exploits the variance $\mathbb{E}[X^2]$. Jensen's inequality (from §04) underlies both Hoeffding's lemma proof and the information-theoretic interpretation of Chernoff bounds.

**Looking forward.** The Law of Large Numbers says $\bar{X}_n \to \mu$ almost surely. This is the *qualitative* counterpart of Hoeffding's quantitative bound: Hoeffding gives the *rate* at which $\bar{X}_n$ concentrates. The Central Limit Theorem (CLT) says the *shape* of the distribution of $\bar{X}_n$ converges to Gaussian — explaining why sub-Gaussian bounds are the right model for sums of bounded variables. Both are developed in [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md).

**The PAC-statistics connection.** PAC learning theory is, at its core, applied concentration theory. The "probably" in PAC = concentration inequality. The "approximately correct" = $\varepsilon$ tolerance. The sample complexity $n = O((d + \log(1/\delta))/\varepsilon^2)$ is exactly what concentration inequalities give. Chapter 7 (Statistics) will build on this: confidence intervals are Hoeffding bounds, hypothesis tests are Chernoff bounds, and MLE theory uses Bernstein to prove asymptotic normality.

```
CONCENTRATION INEQUALITIES IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  §04 Expectation and Moments
  ├─ E[X], Var(X), MGF M_X(t)  ────────────────────────── input
  ├─ Jensen's inequality         ────────────────────────── used in proofs
  └─ Markov/Chebyshev preview    ────────────────────────── previewed there

         ↓

  §05 Concentration Inequalities   ◄─── YOU ARE HERE
  ├─ Markov / Chebyshev            (moment-based, polynomial tails)
  ├─ Sub-Gaussian / Hoeffding      (bounded, exponential tails)
  ├─ Chernoff / Bernstein          (MGF-based, variance-aware)
  ├─ McDiarmid                     (functions of independent variables)
  ├─ Union bound + covering        (infinite hypothesis classes)
  ├─ PAC learning / VC dimension   (generalisation theory)
  └─ Rademacher complexity         (data-dependent bounds)

         ↓                                      ↓

  §06 Stochastic Processes         Chapter 7: Statistics
  ├─ Weak LLN (via Chebyshev)      ├─ Confidence intervals
  ├─ Strong LLN                    ├─ Hypothesis testing
  └─ CLT (limit of Hoeffding)      └─ MLE consistency (Bernstein)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix A: Summary of Key Bounds

| Inequality | Conditions | Bound | Rate |
|-----------|-----------|-------|------|
| Markov | $X \geq 0$, $\mathbb{E}[X] = \mu$ | $P(X \geq t) \leq \mu/t$ | $O(1/t)$ |
| Chebyshev | $\operatorname{Var}(X) = \sigma^2$ | $P(\lvert X-\mu\rvert \geq t) \leq \sigma^2/t^2$ | $O(1/t^2)$ |
| Cantelli | $\operatorname{Var}(X) = \sigma^2$ | $P(X-\mu \geq t) \leq \sigma^2/(\sigma^2+t^2)$ | $O(1/t^2)$ |
| Sub-Gaussian | $X$ $\sigma^2$-sub-G | $P(X \geq t) \leq e^{-t^2/(2\sigma^2)}$ | $O(e^{-t^2})$ |
| Hoeffding | $X_i \in [a_i,b_i]$ i.i.d. | $P(\bar{X}-\mu \geq t) \leq \exp(-2nt^2/\sum(b_i-a_i)^2)$ | $O(e^{-nt^2})$ |
| Chernoff (mult.) | $\operatorname{Bin}(n,p)$, $\delta \in (0,1)$ | $P(S \geq (1+\delta)\mu) \leq e^{-\mu\delta^2/3}$ | $O(e^{-\mu})$ |
| Bernstein | $\lvert X_i\rvert \leq c$, variance $\nu^2$ | $P(\bar{X} \geq t) \leq \exp(-nt^2/2/(\nu^2+ct/3))$ | $O(e^{-nt^2/\nu^2})$ |
| McDiarmid | $f$ with $c_i$-bounded diff. | $P(f - \mathbb{E}[f] \geq t) \leq e^{-2t^2/\sum c_i^2}$ | $O(e^{-t^2})$ |

## Appendix B: Sample Complexity Table

Given: i.i.d. data in $[0,1]$, want $P(|\bar{X}_n - \mu| \leq \varepsilon) \geq 1-\delta$.

| $\varepsilon$ | $\delta$ | Hoeffding $n$ | Normal approx $n$ |
|--------|------|------------|-----------------|
| 0.10 | 0.10 | 133 | 68 |
| 0.05 | 0.10 | 530 | 271 |
| 0.05 | 0.05 | 600 | 384 |
| 0.01 | 0.05 | 14,979 | 9,604 |
| 0.01 | 0.01 | 19,933 | 16,587 |

Hoeffding is distribution-free (works for worst case); Normal approximation assumes CLT holds.

---

[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Stochastic Processes →](../06-Stochastic-Processes/notes.md)

## Appendix C: Detailed Proofs and Derivations

### C.1 Proof of Hoeffding's Lemma in Full

We prove: if $\mathbb{E}[X] = 0$ and $X \in [a, b]$ a.s., then $\mathbb{E}[e^{tX}] \leq e^{t^2(b-a)^2/8}$.

**Step 1: Convexity bound.** Since $e^{tx}$ is convex in $x$, for $x \in [a, b]$:
$$e^{tx} \leq \frac{b - x}{b - a} e^{ta} + \frac{x - a}{b - a} e^{tb}$$

**Step 2: Take expectations.** Let $p = \frac{-a}{b-a} \in [0,1]$ (since $a \leq 0 \leq b$ after centering):
$$\mathbb{E}[e^{tX}] \leq \frac{b - \mathbb{E}[X]}{b-a} e^{ta} + \frac{\mathbb{E}[X] - a}{b-a} e^{tb} = \frac{b}{b-a} e^{ta} + \frac{-a}{b-a} e^{tb}$$
$$= (1-p) e^{ta} + p e^{tb} = e^{ta}\left[(1-p) + p e^{t(b-a)}\right]$$

**Step 3: Exponential bound.** Let $h = t(b-a)$ and $\phi(h) = \log[(1-p) + pe^h] - ph$. Then:
$$\mathbb{E}[e^{tX}] \leq e^{ta + ph} \cdot e^{\phi(h)} = e^{ta + p \cdot t(b-a)} \cdot e^{\phi(h)}$$

Note: $ta + p \cdot t(b-a) = ta + t(-a/(b-a))(b-a) = ta - ta = 0$ (since $p = -a/(b-a)$). So:
$$\mathbb{E}[e^{tX}] \leq e^{\phi(h)}$$

**Step 4: Bound $\phi(h)$.** We have $\phi(0) = 0$ and $\phi'(0) = 0$. By Taylor's theorem:
$$\phi(h) = \phi(0) + h\phi'(0) + \frac{h^2}{2}\phi''(\xi)$$

for some $\xi \in [0, h]$. Computing: $\phi''(h) = \frac{pe^h(1-p)}{((1-p)+pe^h)^2} \leq \frac{1}{4}$

(since $xy \leq (x+y)^2/4$ for $x, y \geq 0$, applied to $x = (1-p)$, $y = pe^h$).

Therefore: $\phi(h) \leq \frac{h^2}{8} = \frac{t^2(b-a)^2}{8}$. $\blacksquare$

### C.2 Proof of McDiarmid's Inequality via Azuma's Inequality

**Azuma's Inequality.** If $(Z_0, Z_1, \ldots, Z_n)$ is a martingale with $|Z_k - Z_{k-1}| \leq c_k$ a.s., then:
$$P(Z_n - Z_0 \geq t) \leq \exp\!\left(-\frac{2t^2}{\sum_{k=1}^n c_k^2}\right)$$

**Doob martingale construction.** Given independent $X_1, \ldots, X_n$ and $f$ with bounded differences $c_i$, define:
$$Z_k = \mathbb{E}[f(X_1, \ldots, X_n) \mid X_1, \ldots, X_k]$$

Then:
- $Z_0 = \mathbb{E}[f]$ (constant, before observing anything)
- $Z_n = f(X_1, \ldots, X_n)$ (the function value itself)
- $(Z_k)$ is a martingale by the tower property

**Bounding differences.** For each $k$:
$$Z_k - Z_{k-1} = \mathbb{E}[f \mid X_1,\ldots,X_k] - \mathbb{E}[f \mid X_1,\ldots,X_{k-1}]$$

Since changing $X_k$ changes $f$ by at most $c_k$ and $Z_{k-1}$ doesn't depend on $X_k$, we have $|Z_k - Z_{k-1}| \leq c_k$.

Applying Azuma to this Doob martingale gives McDiarmid's inequality. $\blacksquare$

### C.3 Proof of the Finite-Class PAC Bound (Agnostic Case)

**Setup.** $\mathcal{H}$ finite with $|\mathcal{H}| = M$. Losses $\ell_h^{(i)} = \ell(h, z^{(i)}) \in [0,1]$. True risk $R(h) = \mathbb{E}[\ell_h]$, empirical risk $\hat{R}(h) = \frac{1}{n}\sum_i \ell_h^{(i)}$.

**Goal.** Show $P(\sup_h |R(h) - \hat{R}(h)| > \varepsilon) \leq \delta$ when $n \geq \frac{\log(2M/\delta)}{2\varepsilon^2}$.

**Step 1.** Fix $h$. By Hoeffding: $P(|R(h) - \hat{R}(h)| > \varepsilon) \leq 2e^{-2n\varepsilon^2}$.

**Step 2.** Union bound over all $M$ hypotheses:
$$P(\exists h: |R(h) - \hat{R}(h)| > \varepsilon) \leq \sum_{h \in \mathcal{H}} 2e^{-2n\varepsilon^2} = 2M e^{-2n\varepsilon^2}$$

**Step 3.** Set $2Me^{-2n\varepsilon^2} \leq \delta$:
$$n \geq \frac{\log(2M/\delta)}{2\varepsilon^2}$$

**ERM generalisation.** Under uniform convergence, the ERM $\hat{h} = \arg\min_h \hat{R}(h)$ satisfies:
$$R(\hat{h}) \leq \hat{R}(\hat{h}) + \varepsilon \leq \hat{R}(h^*) + \varepsilon \leq R(h^*) + 2\varepsilon$$

where $h^* = \arg\min_h R(h)$ is the Bayes-optimal hypothesis in $\mathcal{H}$. $\blacksquare$

---

## Appendix D: VC Theory in Depth

### D.1 The Growth Function

**Definition.** The **growth function** $\Pi_\mathcal{H}(n)$ counts the maximum number of distinct labellings achievable by $\mathcal{H}$ on any $n$ points:
$$\Pi_\mathcal{H}(n) = \max_{x^{(1)},\ldots,x^{(n)} \in \mathcal{X}} \left|\{(h(x^{(1)}),\ldots,h(x^{(n)})): h \in \mathcal{H}\}\right|$$

If $\mathcal{H}$ can shatter $n$ points, $\Pi_\mathcal{H}(n) = 2^n$. The VC dimension $d = d_{VC}$ is the largest $n$ for which $\Pi_\mathcal{H}(n) = 2^n$.

### D.2 Sauer-Shelah Lemma

**Lemma.** For $d_{VC}(\mathcal{H}) = d$ and $n \geq d$:
$$\Pi_\mathcal{H}(n) \leq \sum_{k=0}^d \binom{n}{k}$$

**Proof idea.** By induction on $n$ and $d$. Split: either $x^{(n)}$ makes no difference to the labelling count, or it doubles some labellings. Careful counting bounds the total.

**Consequence.** For $n \geq d$: $\Pi_\mathcal{H}(n) \leq (en/d)^d$. This transitions from exponential $2^n$ (when class is rich) to polynomial $(en/d)^d$ (once $n$ exceeds VC dimension).

### D.3 VC Dimension Examples

| Hypothesis Class | VC Dimension |
|-----------------|-------------|
| Threshold on $\mathbb{R}$ | 1 |
| Intervals on $\mathbb{R}$ | 2 |
| Halfspaces in $\mathbb{R}^d$ | $d+1$ |
| Axis-aligned rectangles in $\mathbb{R}^2$ | 4 |
| Convex polygons in $\mathbb{R}^2$ | $\infty$ |
| Degree-$k$ polynomials in $\mathbb{R}$ | $k+1$ |
| Neural nets with $W$ weights | $O(W \log W)$ |
| Kernel classifiers (universal) | $\infty$ |

**Why VC = $d+1$ for halfspaces.** In $\mathbb{R}^d$, any $d+1$ points in general position (no $d+1$ are on a hyperplane) can be shattered by a halfspace. But for $d+2$ points, by Radon's theorem, they can be split into two groups whose convex hulls intersect — no halfspace separates them. Hence $d_{VC} = d+1$.

### D.4 Fundamental Theorem of Statistical Learning

**Theorem.** For binary classification, the following are equivalent:
1. $\mathcal{H}$ is PAC-learnable
2. ERM is a successful learner for $\mathcal{H}$
3. $\mathcal{H}$ has finite VC dimension
4. $\mathcal{H}$ has the uniform convergence property

This theorem, proved through Sauer-Shelah and the symmetrisation technique (doubling the sample and introducing Rademacher variables), is the bedrock of statistical learning theory.

---

## Appendix E: Advanced Topics in Concentration

### E.1 Talagrand's Inequality

Talagrand's inequality (1995) strengthens McDiarmid for "self-bounding" functions — those where the effect of each coordinate is bounded by the function value itself.

**Self-bounding functions.** $f: \mathcal{X}^n \to \mathbb{R}_{\geq 0}$ is self-bounding if there exist $f_i: \mathcal{X}^{n-1} \to \mathbb{R}$ such that:
$$0 \leq f(\mathbf{x}) - f_i(\mathbf{x}^{-i}) \leq 1 \quad \text{and} \quad \sum_{i=1}^n (f(\mathbf{x}) - f_i(\mathbf{x}^{-i})) \leq f(\mathbf{x})$$

**Talagrand's bound.** For self-bounding $f$:
$$P(f \geq \mathbb{E}[f] + t) \leq e^{-t^2/(2(\mathbb{E}[f]+t/3))}$$

This is a *variance-adaptive* McDiarmid — analogous to Bernstein vs Hoeffding.

**Applications:** Number of ones in a sum of Bernoullis, size of random matchings, empirical risk when the model fits the data.

### E.2 Concentration for Heavy-Tailed Distributions

When distributions are heavy-tailed, sub-Gaussian bounds don't apply. Modern techniques include:

**Catoni's estimator.** For estimating the mean of a distribution with finite variance $\sigma^2$ (but potentially infinite MGF), Catoni's estimator achieves:
$$P(|\hat{\mu}_\psi - \mu| \geq t) \leq 2\exp\!\left(-\frac{nt^2}{2\sigma^2 + 2t\sigma}\right) \approx e^{-nt^2/(2\sigma^2)}$$

using a truncated exponential influence function. This gives Bernstein-like rates without boundedness.

**Median of means.** Partition $n$ samples into $k$ groups of $n/k$. Compute group means and take the median. The median is sub-Gaussian with parameter $\sigma^2 k/n$ — achieving sub-Gaussian rates for distributions with only two moments.

**For AI:** Training data for LLMs often has heavy-tailed length distributions (documents follow power laws). Heavy-tail-robust mean estimation is relevant for unbiased training.

### E.3 Matrix Concentration Inequalities

Scalar concentration extends to matrices. For many AI applications, we care about concentration of random matrices.

**Matrix Hoeffding.** For independent random PSD matrices $X_1, \ldots, X_n \in \mathbb{R}^{d \times d}$ with $0 \preceq X_i \preceq cI$ and $\mathbb{E}[\sum X_i] = M$, for $t > 0$:
$$P\!\left(\lambda_{\max}\!\left(\sum_i X_i - M\right) \geq t\right) \leq d \cdot e^{-t^2/(2nc^2)}$$

The extra factor $d$ (matrix dimension) accounts for the eigenvalue structure.

**Application: random features.** When approximating a kernel matrix $K_{ij} = k(x^{(i)}, x^{(j)})$ by random features, the approximation error in spectral norm concentrates with matrix Hoeffding, justifying the use of $D = O(d^2/\varepsilon^2)$ random features for $\varepsilon$-accurate Gram matrix approximation.

**Matrix Bernstein.** For independent zero-mean random matrices $X_i$ with $\|X_i\|_2 \leq R$ and $\sigma^2 = \|\sum \mathbb{E}[X_i^2]\|_2$:
$$P\!\left(\left\|\sum_i X_i\right\|_2 \geq t\right) \leq 2d \cdot \exp\!\left(-\frac{t^2/2}{\sigma^2 + Rt/3}\right)$$

Used in randomised linear algebra (randomised SVD, Nyström approximation).

### E.4 PAC-Bayes Theory

PAC-Bayes bounds combine the PAC framework with Bayesian prior knowledge.

**Setup.** Given a prior $P$ over $\mathcal{H}$ (before seeing data) and posterior $Q$ (after seeing data):

**PAC-Bayes Theorem (McAllester, 1999).** For any prior $P$ and $\delta > 0$, with probability $\geq 1-\delta$:
$$\mathbb{E}_{h \sim Q}[R(h)] \leq \mathbb{E}_{h \sim Q}[\hat{R}(h)] + \sqrt{\frac{D_{\mathrm{KL}}(Q \| P) + \log(1/\delta)}{2(n-1)}}$$

**Why this is powerful:** The KL term $D_{\mathrm{KL}}(Q\|P)$ replaces $\log|\mathcal{H}|$ (which can be infinite). For neural networks, if the posterior concentrates near the prior (small KL), the bound is tight. Flat minima (low sharpness) correspond to posteriors that don't drift far from the prior.

**Connection to SAM.** Sharpness-Aware Minimisation (SAM, 2021) minimises a PAC-Bayes-inspired upper bound on generalisation error by seeking flat minima — directly operationalising PAC-Bayes in practice.

---

## Appendix F: Worked Examples

### F.1 Coin Flip Estimation

**Problem.** We flip a biased coin $n$ times and observe $\hat{p}$ heads. How many flips to estimate $p$ within $\pm 0.05$ with 99% confidence?

**Hoeffding solution.** With $X_i \in \{0,1\}$, $b-a=1$:
$$n \geq \frac{\log(2/0.01)}{2(0.05)^2} = \frac{\log 200}{0.005} = \frac{5.298}{0.005} = 1060$$

**Normal approximation.** $n \geq (2.576/0.05)^2 \cdot 0.25 = 664$. Less conservative because it uses the Gaussian CLT approximation (valid for large $n$).

**Interpretation.** About 1000-1100 flips are needed for reliable probability estimation. This is why A/B tests often require thousands of users: smaller samples give confidence intervals too wide to detect meaningful differences.

### F.2 Multi-Armed Bandit via Chernoff

**Problem.** A recommendation system has $K = 100$ arms (content types). Each arm $a$ has unknown click-through rate $p_a$. We want to identify the best arm within $\varepsilon = 0.01$ with $\delta = 0.01$.

**Naive approach.** Pull each arm $n_0$ times. By Hoeffding + union bound:
$$P(\exists a: |\hat{p}_a - p_a| > \varepsilon) \leq 2K e^{-2n_0\varepsilon^2}$$

Set $\leq \delta$: $n_0 \geq \frac{\log(2K/\delta)}{2\varepsilon^2} = \frac{\log(20000)}{0.0002} \approx 48{,}200$. Total pulls: $4{,}820{,}000$.

**Adaptive strategy (UCB).** Upper Confidence Bound algorithms concentrate exploration on promising arms, achieving total pulls $O(K\log n/\varepsilon^2)$ — much smaller in practice.

### F.3 Neural Network Generalisation: PAC-Bayes in Practice

**Setup.** 2-layer ReLU network, 100 hidden units, trained on 50,000 MNIST examples. Training error 0.3%, test error 1.2%. Standard VC bound: vacuous (parameter count $\gg n$).

**PAC-Bayes approach.** Use Gaussian prior $P = \mathcal{N}(0, \sigma_0^2 I)$ and posterior $Q = \mathcal{N}(\hat{\theta}, \sigma^2 I)$ (trained weights $\hat{\theta}$). Compute:
$$D_{\mathrm{KL}}(Q\|P) = \frac{\|\hat{\theta}\|_2^2}{2\sigma_0^2} + d\log(\sigma_0/\sigma) + d\sigma^2/(2\sigma_0^2) - d/2$$

With careful tuning of $\sigma$, this can give non-vacuous bounds (< 50% error guarantee) even for deep networks — unlike VC theory.

### F.4 Random Projection (Johnson-Lindenstrauss)

**Problem.** We have $n = 10000$ points in $\mathbb{R}^{1000}$. We want to project to $\mathbb{R}^m$ preserving all pairwise distances within factor $(1 \pm \varepsilon)$ for $\varepsilon = 0.1$.

**JL Lemma.** For $m = O(\log n / \varepsilon^2)$ random projections (from $\mathcal{N}(0, I/m)$):
$$m \geq \frac{4 + 2\log(n^2)}{\varepsilon^2/2 - \varepsilon^3/3} \approx \frac{8\log n}{\varepsilon^2} = \frac{8 \cdot \ln(10000)}{0.01} \approx 7400$$

So 7400 dimensions suffice — far less than 1000 (already doing 26% compression), and compression from 10000 would be massive.

**Proof sketch.** A random Gaussian matrix $R \in \mathbb{R}^{m \times d}$ satisfies: for any fixed pair of points $x, y$:
$$P\!\left(\left|\|Rx - Ry\|_2^2/m - \|x-y\|_2^2\right| > \varepsilon\|x-y\|_2^2\right) \leq 2e^{-m\varepsilon^2/8}$$

by chi-squared concentration. Union bound over $\binom{n}{2}$ pairs gives the JL lemma.

**For AI:** Locality-sensitive hashing, approximate nearest-neighbour search (used in RAG retrieval), and low-rank adaptation (LoRA) all rely on Johnson-Lindenstrauss-type arguments to justify dimensionality reduction.

---

## Appendix G: Computing Concentration Bounds in Practice

### G.1 Choosing the Right Bound

```
DECISION TREE: WHICH BOUND TO USE
════════════════════════════════════════════════════════════════════════

  Do you know E[X]? ─── No ──► Can't bound tail
       │
       Yes
       │
  Is X ≥ 0? ─── Yes ──► Markov: P(X ≥ t) ≤ E[X]/t
       │
       No
       │
  Do you know Var(X)? ─── Yes ──► Chebyshev: P(|X-μ| ≥ t) ≤ σ²/t²
       │
       No
       │
  Is X bounded in [a,b]? ─── Yes ──► Hoeffding (exponential bound)
       │                             or Bernstein if Var(X) known
       │
       No
       │
  Is MGF finite? ─── Yes ──► Chernoff method (optimise over s)
       │
       No
       │
  Distribution-free needed ──► Student-t CI or bootstrap

════════════════════════════════════════════════════════════════════════
```

### G.2 Python Recipe for Hoeffding CIs

```python
import numpy as np
from scipy import stats

def hoeffding_ci(x, delta=0.05, lo=0.0, hi=1.0):
    """Hoeffding confidence interval for bounded data in [lo, hi]."""
    n = len(x)
    x_bar = np.mean(x)
    radius = (hi - lo) * np.sqrt(np.log(2 / delta) / (2 * n))
    return x_bar - radius, x_bar + radius

def hoeffding_sample_size(epsilon, delta, lo=0.0, hi=1.0):
    """Required n for epsilon-accuracy at 1-delta confidence."""
    return int(np.ceil((hi - lo)**2 * np.log(2 / delta) / (2 * epsilon**2)))
```

### G.3 Interpreting Generalisation Gaps

When a model achieves training error 1% and test error 5%:
- Generalisation gap = 4%
- Is this within bounds? With $n = 10000$, $\delta = 0.05$: Hoeffding gives $\sqrt{\log(40)/20000} \approx 1.4\%$ per hypothesis
- For $|\mathcal{H}| = 10^6$ hypotheses: add $\sqrt{\log(10^6)/20000} \approx 2.6\%$
- Total bound: $\approx 4\%$ — exactly matching the observed gap!

In practice, neural networks occupy a tiny corner of their hypothesis class (due to inductive bias of SGD), so effective complexity is much smaller than the VC dimension suggests.

---

## Appendix H: Self-Assessment Checklist

Before moving to §06 (Stochastic Processes), verify you can:

- [ ] **State Markov's inequality** from memory and prove it with the indicator trick
- [ ] **Derive Chebyshev from Markov** by applying Markov to $(X-\mu)^2$
- [ ] **Prove Hoeffding's lemma** using convexity of $e^{tx}$ and Taylor expansion
- [ ] **Apply the Chernoff method**: Markov + MGF bound + optimise over $s$
- [ ] **State McDiarmid's inequality** and identify $c_i$ for a given function
- [ ] **Compute sample complexity** for given $(\varepsilon, \delta)$ using Hoeffding
- [ ] **Define VC dimension** and compute it for halfspaces ($= d+1$ in $\mathbb{R}^d$)
- [ ] **Write the PAC generalisation bound** with $|\mathcal{H}|$ or $d_{VC}$
- [ ] **Define Rademacher complexity** and explain its interpretation
- [ ] **Identify when Bernstein beats Hoeffding** (small variance case)
- [ ] **Give a non-example** for sub-Gaussian (heavy-tailed distribution)
- [ ] **Explain why VC bounds are vacuous for LLMs** and what PAC-Bayes offers

---

## Appendix I: Connections to Information Theory

### I.1 Kullback-Leibler Divergence and Exponential Families

The Chernoff method has a deep information-theoretic interpretation. The optimal exponent in the Chernoff bound equals the **KL divergence** under the tilted distribution.

For $X_1, \ldots, X_n$ i.i.d. with distribution $P$, and target $\bar{x} = a > \mathbb{E}[X]$:
$$\lim_{n \to \infty} \frac{1}{n}\log P(\bar{X}_n \geq a) = -D_{\mathrm{KL}}(P_a \| P)$$

where $P_a$ is the *exponential tilt* of $P$ that puts its mean at $a$: $P_a(x) \propto e^{s^* x} P(x)$ with $\mathbb{E}_{P_a}[X] = a$.

This is **Cramér's theorem** — the foundation of large deviations theory. The probability of a rare event decays exponentially at rate given by the KL divergence to the closest distribution that makes the event typical.

**For AI:** This connection explains why cross-entropy loss (KL divergence from data to model) is the right loss for language models: minimising cross-entropy = maximising the probability of the training data = minimising the Chernoff exponent for generalisation error.

### I.2 Entropy and Sauer-Shelah

The growth function $\Pi_\mathcal{H}(n)$ counts the number of distinct behaviours of $\mathcal{H}$ on $n$ points. The log growth rate:
$$h(\mathcal{H}) = \lim_{n \to \infty} \frac{\log_2 \Pi_\mathcal{H}(n)}{n}$$

is the **combinatorial entropy** of $\mathcal{H}$. Sauer-Shelah shows:
- If $d_{VC} < \infty$: $h(\mathcal{H}) = 0$ (subexponential growth, polynomial in $n$)
- If $d_{VC} = \infty$: $h(\mathcal{H}) = 1$ (full exponential growth $2^n$)

This binary distinction — polynomial vs exponential growth — is precisely what separates PAC-learnable from non-learnable hypothesis classes.

---

## Appendix J: Connections to Optimisation

### J.1 Convergence of SGD via Hoeffding

The standard convergence analysis of SGD for convex functions uses:

1. **Descent lemma:** $\mathcal{L}(\theta_{t+1}) \leq \mathcal{L}(\theta_t) - \eta g_t^\top(\theta_t - \theta^*) + \frac{\eta^2 L}{2}\|g_t\|_2^2$

2. **Gradient concentration:** $\|\hat{g}_t - g_t\|_2 \leq \varepsilon$ with high probability (Hoeffding)

3. **Telescoping:** Sum over $T$ steps and apply Hoeffding union bound over all $T$ steps (adding $\log T$ to the sample complexity)

The result: after $T = O(G^2/(\varepsilon^2\eta^2))$ steps, SGD achieves $\mathcal{L}(\theta_T) - \mathcal{L}(\theta^*) \leq \varepsilon$ with high probability.

### J.2 Generalisation and Algorithmic Stability

**Definition (Uniform Stability).** Algorithm $\mathcal{A}$ is $\beta$-uniformly stable if for any two datasets $S$, $S'$ differing in one example:
$$\sup_z |\ell(\mathcal{A}(S), z) - \ell(\mathcal{A}(S'), z)| \leq \beta$$

**Theorem (Bousquet-Elisseeff, 2002).** If $\mathcal{A}$ is $\beta$-stable with $\beta \leq c/n$ for some constant $c$, and losses are bounded in $[0, 1]$, then with probability $\geq 1-\delta$:
$$R(\mathcal{A}(S)) \leq \hat{R}(\mathcal{A}(S)) + 2\beta + (4n\beta + 1)\sqrt{\frac{\log(1/\delta)}{2n}}$$

**SGD is stable.** For $L$-smooth convex losses, SGD with step size $\eta$ for $T$ steps is $2\eta TL/n$-uniformly stable. Setting $\eta = O(1/\sqrt{T})$ gives stability $O(\sqrt{T}/n)$ — small when $T \ll n^2$.

This explains why early stopping helps: more SGD steps = less stable = worse generalisation.


---

## Appendix K: Concentration Phenomena in Transformer Architectures

### K.1 Attention Score Concentration

In scaled dot-product attention, query-key inner products $Q_i K_j^\top / \sqrt{d_k}$ determine the attention weights. For randomly initialised $Q, K \sim \mathcal{N}(0, 1/d_k)$, the inner products satisfy:
$$P(|Q_i K_j^\top| \geq t) \leq 2e^{-d_k t^2/2}$$

Without the $\sqrt{d_k}$ scaling, the softmax input $Q_i K_j^\top$ has standard deviation $O(\sqrt{d_k})$, pushing softmax into its saturating regime (near-uniform or near-one-hot). The scaling ensures standard deviation $O(1)$, keeping gradients flowing.

**Formal justification.** With $d_k$-dimensional queries and keys, $Q_i K_j^\top = \sum_{l=1}^{d_k} Q_{il} K_{jl}$ where each summand has variance $1/d_k^2$. By sub-Gaussian closure, the sum is $1/d_k$-sub-Gaussian, so standard deviation $\approx 1/\sqrt{d_k}$. After scaling: $Q_i K_j^\top / \sqrt{d_k}$ has standard deviation $\approx 1$, optimal for softmax.

### K.2 Concentration in Residual Networks

Deep residual networks $x_{l+1} = x_l + f_l(x_l)$ accumulate residual perturbations. If each residual block $f_l$ has bounded output $\|f_l(x)\|_2 \leq c$, then by Azuma's inequality applied to the martingale $x_l$:
$$P(\|x_L - x_0\|_2 \geq t) \leq \exp\!\left(-\frac{t^2}{2Lc^2}\right)$$

This says: even with $L = 100$ layers, if each residual block adds a small perturbation ($c \ll 1$), the total drift is bounded. **LayerNorm and weight decay enforce exactly this:** they keep $\|f_l(x)\|_2$ small, ensuring the representation doesn't drift exponentially with depth.

### K.3 LoRA and Low-Rank Concentration

LoRA (Low-Rank Adaptation) approximates weight updates as $\Delta W = AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$ with small rank $r$. The resulting hypothesis class has reduced complexity:

**Rademacher complexity of LoRA.** For the function class $\{x \mapsto W_0 x + AB x : \|A\|_F \leq s_A, \|B\|_F \leq s_B\}$:
$$\mathfrak{R}_n(\mathcal{F}_{\text{LoRA}}) \leq \frac{s_A s_B \sqrt{r} \cdot C}{\sqrt{n}}$$

where $C = \max_i \|x^{(i)}\|_2$. The key: Rademacher complexity scales as $\sqrt{r}$, not $\sqrt{dk}$ (full fine-tuning). With $r \ll \min(d,k)$, LoRA has provably smaller generalisation gap than full fine-tuning, explaining why it doesn't overfit despite being used on small datasets.

---

## Appendix L: Practical Guidelines for ML Practitioners

### L.1 When to Report Confidence Intervals

**Always report** when:
- Sample size $n < 10000$ (Hoeffding CI is meaningful)
- Comparing two systems (the CI overlap tells you if the difference is significant)
- Benchmarking LLMs on standard evaluations

**Minimum information to report:** point estimate, sample size, CI method (Hoeffding vs Normal approximation), confidence level.

### L.2 Common Benchmarking Mistakes

| Mistake | Statistical Problem | Fix |
|---------|-------------------|-----|
| "Model A (73.2%) > Model B (72.8%) on 1000 questions" | 95% CI is ±4.2% — difference insignificant | Need $n \geq 19000$ for ±1% CI |
| "Best of 10 model variants is significant" | Multiple comparison: $10 \times 0.05 = 0.5$ false positive rate | Bonferroni: each test at $0.05/10 = 0.005$ |
| "Averaging over 5 datasets gives reliable comparison" | Each dataset is a separate test | Report CI for each; meta-analysis requires care |
| "Our method wins on 8/10 tasks" | Sign test: need $\geq 9/10$ for $p < 0.05$ | Use paired t-test or Wilcoxon signed-rank |
| "1000-epoch training curve shows improvement" | Training curve is not test performance | Report test error at convergence with CI |

### L.3 Required Sample Sizes (Reference Table)

For binary outcomes (accuracy), using Hoeffding with 95% confidence ($\delta = 0.05$):

| Desired precision ($\varepsilon$) | Required $n$ |
|----------------------------------|-------------|
| ±10% (0.10) | 185 |
| ±5% (0.05) | 738 |
| ±3% (0.03) | 2,056 |
| ±2% (0.02) | 4,626 |
| ±1% (0.01) | 18,445 |
| ±0.5% (0.005) | 73,779 |

*Note: These are conservative (distribution-free). Normal approximation gives roughly half as many, valid when $n\hat{p}(1-\hat{p}) \geq 5$.*

### L.4 Interpreting Generalisation Bounds for Deep Learning

Current theoretical bounds are often **loose** for deep networks. Practitioners should interpret them as:

1. **Order of magnitude guidance:** A bound of $O(\sqrt{d/n})$ correctly predicts that more data helps linearly, but may overestimate the absolute gap by 10-100×
2. **Relative comparisons:** PAC-Bayes bounds correctly rank models by generalisation risk even when absolute values are imprecise
3. **Design principles:** Results like "norm-constrained models generalise better" translate directly to regularisation practices (weight decay, gradient clipping, dropout)
4. **Not absolute guarantees:** A theoretical bound of 40% error doesn't mean the model has 40% test error — it means we can't rule out 40% test error from the theory alone

The theory is most useful as a framework for thinking, not as a numerical oracle.

---

## Appendix M: References and Further Reading

### Primary Sources

1. **Boucheron, Lugosi & Massart** (2013). *Concentration Inequalities: A Nonasymptotic Theory of Independence.* Oxford University Press. — The definitive modern treatment.

2. **Hoeffding, W.** (1963). Probability Inequalities for Sums of Bounded Random Variables. *Journal of the American Statistical Association*, 58(301), 13–30.

3. **McDiarmid, C.** (1989). On the Method of Bounded Differences. *Surveys in Combinatorics*, 141, 148–188.

4. **Vapnik, V. & Chervonenkis, A.** (1971). On the Uniform Convergence of Relative Frequencies to Their Probabilities. *Theory of Probability and Its Applications*, 16(2), 264–280.

5. **Valiant, L.** (1984). A Theory of the Learnable. *Communications of the ACM*, 27(11), 1134–1142.

### Learning Theory Textbooks

6. **Shalev-Shwartz & Ben-David** (2014). *Understanding Machine Learning: From Theory to Algorithms.* Cambridge University Press. — Best introduction to PAC learning for ML practitioners.

7. **Mohri, Rostamizadeh & Talwalkar** (2018). *Foundations of Machine Learning* (2nd ed.). MIT Press. — Covers Rademacher complexity and generalisation theory in depth.

8. **Wainwright, M.** (2019). *High-Dimensional Statistics: A Non-Asymptotic Viewpoint.* Cambridge University Press. — Advanced treatment, covers matrix concentration and modern techniques.

### Deep Learning Theory

9. **Zhang, Bengio et al.** (2017). Understanding Deep Learning Requires Rethinking Generalization. *ICLR 2017.* — Showed memorisation capacity, challenging classical VC theory.

10. **Dziugaite & Roy** (2017). Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many Parameters. *UAI 2017.* — First non-vacuous PAC-Bayes bounds for deep nets.

11. **Belkin et al.** (2019). Reconciling Modern Machine-Learning Practice and the Classical Bias–Variance Trade-Off. *PNAS 2019.* — Double descent, overparameterisation.

12. **Foret et al.** (2021). Sharpness-Aware Minimization for Efficiently Improving Generalization. *ICLR 2021.* — SAM as PAC-Bayes operationalisation.


---

## Appendix N: Extended Worked Problems

### N.1 Bounding the Maximum of Random Variables

**Problem.** Let $X_1, \ldots, X_n \sim \mathcal{N}(0,1)$ i.i.d. What is $\mathbb{E}[\max_i X_i]$ approximately?

**Answer.** Each $X_i$ is 1-sub-Gaussian. By the union bound:
$$P(\max_i X_i \geq t) \leq n \cdot P(X_1 \geq t) \leq n e^{-t^2/2}$$

Setting $ne^{-t^2/2} = 1$: $t^* = \sqrt{2\log n}$. So $\max_i X_i = O(\sqrt{\log n})$ with high probability.

More precisely, $\mathbb{E}[\max_i X_i] \approx \sqrt{2\log n}(1 - O(1/\log n))$. For $n = 1000$: $\approx \sqrt{2 \cdot 6.9} \approx 3.7$, matching Monte Carlo.

**For AI:** The maximum attention logit in a head with $n$ tokens and $d_k$-dimensional keys satisfies $\max_j Q K_j^\top \approx \sqrt{2d_k \log n}$. The softmax of this concentrates on one or few tokens for large $d_k$ — explaining why attention becomes "sharp" (spiky) in deeper layers where $d_k$ is large.

### N.2 Symmetric Rounding

**Problem.** An algorithm rounds each of $n = 100$ numbers independently up or down by at most 0.5. What is the probability the total rounding error exceeds 10?

**Solution.** Each error $E_i \in [-0.5, 0.5]$ with $\mathbb{E}[E_i] = 0$. The sum $S = \sum E_i$ has $|E_i| \leq 0.5$, so by Hoeffding:
$$P(S \geq 10) \leq \exp\!\left(-\frac{2 \cdot 100}{\sum (1.0)^2}\right) \cdot e^{-t^2} = \exp\!\left(-\frac{2 \cdot 10^2}{100 \cdot 1^2}\right) = e^{-2} \approx 0.135$$

The two-sided version: $P(|S| \geq 10) \leq 2e^{-2} \approx 0.27$.

### N.3 PAC Bound for a Neural Network Classifier

**Problem.** A student trains a 3-layer neural network on $n = 5000$ examples and achieves 2% training error. Using the finite-class PAC bound with $|\mathcal{H}| = 2^{10^6}$ (very conservatively, assuming 1-bit per parameter for $10^6$ parameters), what does the bound say about test error?

**Solution.** The PAC bound (agnostic, finite class):
$$R(\hat{h}) \leq \hat{R}(\hat{h}) + \sqrt{\frac{\log(2|\mathcal{H}|/\delta)}{2n}}$$

With $\delta = 0.05$ and $|\mathcal{H}| = 2^{10^6}$:
$$\sqrt{\frac{\log(2 \cdot 2^{10^6}/0.05)}{10000}} = \sqrt{\frac{10^6 \ln 2 + \ln 40}{10000}} \approx \sqrt{\frac{693150}{10000}} \approx 8.3$$

Test error bound: $0.02 + 8.3 \approx 8.32$ — a completely vacuous bound (exceeds 100%). This illustrates why VC/parameter-counting bounds are useless for deep networks. PAC-Bayes or Rademacher complexity (using $\|\mathbf{w}\|_F$ rather than parameter count) gives non-vacuous results.

### N.4 Proving Generalisation for 1-Nearest Neighbour

**Problem.** For $k$-nearest neighbour classification with $k = 1$:
(a) Show that the leave-one-out error is an unbiased estimate of the true error.
(b) Use McDiarmid to bound the variance of the generalisation error around the leave-one-out estimate.

**Solution.**
(a) The leave-one-out (LOO) error $\hat{R}_{\text{LOO}} = \frac{1}{n}\sum_i \mathbf{1}[\hat{y}_{-i} \neq y^{(i)}]$ where $\hat{y}_{-i}$ is the prediction on $x^{(i)}$ using all other training points. By symmetry of the i.i.d. draw, $\mathbb{E}[\hat{R}_{\text{LOO}}] = R_{n-1}$ (the true error of 1-NN trained on $n-1$ examples).

(b) Swapping one training example changes $\hat{R}_{\text{LOO}}$ by at most $2/n$ (affects at most 2 LOO predictions). By McDiarmid:
$$P(|\hat{R}_{\text{LOO}} - \mathbb{E}[\hat{R}_{\text{LOO}}]| \geq t) \leq 2\exp\!\left(-\frac{2t^2}{\sum_i (2/n)^2}\right) = 2\exp\!\left(-\frac{nt^2}{2}\right)$$

### N.5 Confidence Region for Gradient Descent Convergence

**Problem.** SGD is run for $T$ iterations with step size $\eta = 1/\sqrt{T}$. Each gradient estimate is sub-Gaussian with parameter $\sigma^2$. With what probability does SGD achieve $\mathcal{L}(\theta_T) - \mathcal{L}^* \leq \varepsilon$?

**Solution.** Standard SGD analysis shows: $\mathbb{E}[\mathcal{L}(\theta_T) - \mathcal{L}^*] \leq \frac{R^2 + \sigma^2}{\sqrt{T}}$ where $R = \|\theta_0 - \theta^*\|_2$.

For the high-probability version: at each step $t$, $P(\|g_t - \hat{g}_t\|_2 \geq u) \leq 2e^{-mu^2/(2\sigma^2)}$ by sub-Gaussianity (batch size $m$). Taking union bound over all $T$ steps and using the Azuma-type argument for the algorithm trajectory:
$$P(\mathcal{L}(\theta_T) - \mathcal{L}^* \geq \varepsilon) \leq \delta$$

when $T = O\left(\frac{R^2 \log(1/\delta) + \sigma^2 \log(T/\delta)}{\varepsilon^2}\right)$.

---

## Appendix O: Visualisation Notes

### O.1 Plotting Concentration Inequalities

When visualising tail bounds, always show:
1. **The exact tail probability** (computed numerically) as the ground truth
2. **The bound** as a curve above the exact probability
3. **The ratio** (bound/exact) to show how loose the bound is

```
TAIL BOUND TIGHTNESS — Gaussian(0,1)
════════════════════════════════════════════════════════════════════════

  t   | P(X≥t) exact | Markov* | Chebyshev | Hoeffding | Bound/Exact
  ────────────────────────────────────────────────────────────────────
  1   | 0.159  | N/A    | 1.000    | 0.607     |  4x / 3.8x
  2   | 0.023  | N/A    | 0.250    | 0.135     | 11x / 5.9x
  3   | 0.0013 | N/A    | 0.111    | 0.011     | 85x / 8.5x
  4   | 3.2e-5 | N/A    | 0.0625   | 3.4e-4    | 1953x / 11x
  5   | 2.9e-7 | N/A    | 0.040    | 3.7e-7    | 138,000x / 1.3x

  *Markov requires X ≥ 0; not directly applicable to symmetric Gaussian

════════════════════════════════════════════════════════════════════════
```

At $t = 5\sigma$: Chebyshev overestimates by 138,000×! Hoeffding for Gaussian is essentially tight — Gaussian IS the sub-Gaussian equality case.

### O.2 Sample Complexity Curves

The sample complexity $n(\varepsilon, \delta) = \frac{\log(2/\delta)}{2\varepsilon^2}$ has a characteristic shape:
- $O(1/\varepsilon^2)$: doubling precision requires 4× more data
- $O(\log(1/\delta))$: going from 95% to 99.9% confidence adds only $\log(200/20)/\log(40/20) \approx 3.3 \times$ more data — logarithmically cheap

This $1/\varepsilon^2$ scaling is fundamental: it appears in:
- Hoeffding (statistical estimation)
- Monte Carlo (variance $\sigma^2$, error $\sigma/\sqrt{n}$)
- PAC learning (generalisation gap)
- Stochastic optimisation (SGD convergence rate)

All are manifestations of the same underlying concentration phenomenon.

---

## Appendix P: Glossary of Key Terms

| Term | Definition | Section |
|------|-----------|---------|
| **Concentration inequality** | A bound on $P(X - \mathbb{E}[X] \geq t)$ or $P(\|X - \mathbb{E}[X]\| \geq t)$ | §1 |
| **Sub-Gaussian random variable** | $X$ with $\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2}$ for all $t$ | §3.1 |
| **Sub-Gaussian parameter** | $\sigma^2$: the proxy variance in the sub-Gaussian definition | §3.1 |
| **Rademacher variable** | $\sigma \in \{-1, +1\}$ with equal probability | §3.2 |
| **Hoeffding's lemma** | Bounded RV $X \in [a,b]$ with zero mean $\Rightarrow$ $\mathbb{E}[e^{tX}] \leq e^{t^2(b-a)^2/8}$ | §4.1 |
| **Chernoff method** | Tail bound via $P(X \geq t) \leq e^{-st}\mathbb{E}[e^{sX}]$, optimised over $s$ | §5.1 |
| **Bounded differences** | $|f(\cdots x_i \cdots) - f(\cdots x_i' \cdots)| \leq c_i$ for all $i$ | §7.1 |
| **Union bound** | $P(\bigcup A_i) \leq \sum P(A_i)$ | §8.1 |
| **Covering number** | Smallest $\varepsilon$-net size for a set under a metric | §8.3 |
| **Shattering** | $\mathcal{H}$ shatters $C$ if it can realise all $2^{|C|}$ labellings | §9.4 |
| **VC dimension** | Size of largest shattered set | §9.4 |
| **Growth function** | $\Pi_\mathcal{H}(n)$: max labellings of $n$ points by $\mathcal{H}$ | App D.1 |
| **PAC learning** | Probably approximately correct: $R(\hat{h}) \leq \min_h R(h) + \varepsilon$ w.p. $\geq 1-\delta$ | §9.1 |
| **Uniform convergence** | $\sup_h |\hat{R}(h) - R(h)| \leq \varepsilon$ w.h.p. | §9.3 |
| **Rademacher complexity** | $\mathfrak{R}_n(\mathcal{F}) = \mathbb{E}_\sigma[\sup_f \frac{1}{n}\sum \sigma_i f(x^{(i)})]$ | §10.1 |
| **PAC-Bayes** | Generalisation bound using $D_{\mathrm{KL}}(\text{posterior}\|\text{prior})$ | App E.4 |
| **Algorithmic stability** | Output insensitive to swapping one training example | App J.2 |
| **Exponential tilt** | Twisted distribution $P_s(x) \propto e^{sx}P(x)$ used in Chernoff | App I.1 |


---

## Appendix Q: The Method of Types and Large Deviations

### Q.1 Types and Empirical Distributions

The **method of types** is an elegant combinatorial approach to concentration that complements the analytical MGF-based approach.

**Definition.** For a sequence $x^n = (x_1, \ldots, x_n) \in \mathcal{X}^n$, the **type** (empirical distribution) is:
$$P_{x^n}(a) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}[x_i = a], \quad a \in \mathcal{X}$$

The set of all sequences of type $Q$ is the **type class** $T_n(Q)$.

**Key counting result.** For a finite alphabet $|\mathcal{X}| = k$:
$$|T_n(Q)| = \binom{n}{nQ(x_1), \ldots, nQ(x_k)} \approx e^{nH(Q)}$$

where $H(Q) = -\sum_{x} Q(x)\log Q(x)$ is the entropy of $Q$. This means there are approximately $e^{nH(Q)}$ sequences with empirical distribution $Q$.

### Q.2 Connection to Chernoff Bounds

For i.i.d. $X_i \sim P$ and empirical distribution $P_{x^n}$:
$$P(P_{X^n} = Q) \leq e^{-n D_{\mathrm{KL}}(Q \| P)}$$

This is the **method of types** bound. Taking $Q$ to be the distribution concentrated at $\bar{x} = a$:
$$P(\bar{X}_n \geq a) \leq (n+1)^{|\mathcal{X}|} e^{-n D_{\mathrm{KL}}(Q^* \| P)}$$

where $Q^*$ is the type closest to $P$ with mean $\geq a$. For large $n$, the polynomial factor $(n+1)^{|\mathcal{X}|}$ is negligible, and this matches the Chernoff bound with the KL divergence interpretation.

### Q.3 Cramér's Theorem

**Theorem (Cramér, 1938).** For i.i.d. $X_1, \ldots, X_n$ with distribution $P$:
$$\lim_{n\to\infty} \frac{1}{n} \log P(\bar{X}_n \geq a) = -I(a)$$

where $I(a) = \sup_{t} \{ta - \log M_X(t)\}$ is the **Cramér-Legendre transform** (rate function), and $I(a) = D_{\mathrm{KL}}(P_a \| P)$ (KL divergence to the tilted distribution).

This is the fundamental result of **large deviations theory**: rare events occur at exponential rate determined by the KL distance to the "closest" distribution that makes the event typical.

**For AI:** Language model perplexity is essentially $\exp(D_{\mathrm{KL}}(\text{true dist}\|\text{model}))$. Low perplexity = model close to true distribution = rare events (unusual tokens) are not overly penalised. Cramér's theorem connects the generalisation gap to KL divergence, explaining why cross-entropy is the right loss.

---

## Appendix R: Generalisation in the Overparameterised Regime

### R.1 The Classical Puzzle

Classical theory (VC, Rademacher) says: more parameters = larger hypothesis class = worse generalisation. Yet in practice, overparameterised models (large $d \gg n$) often generalise better than their smaller counterparts.

The resolution involves multiple mechanisms:

**1. Implicit regularisation.** Gradient descent on overparameterised linear models converges to the minimum-$\ell_2$-norm solution. Among all interpolating solutions, this has the smallest Rademacher complexity.

**2. Double descent.** As model capacity increases past the interpolation threshold ($d \approx n$), test error first rises (overfitting) then falls (benign overfitting). The minimum-norm interpolating solution's generalisation error is:
$$\text{MSE} \approx \frac{\sigma^2 d}{n} \cdot \frac{1}{1 - n/d}$$
which decreases as $d/n \to \infty$ (overparameterised regime).

**3. Benign overfitting.** Bartlett et al. (2020) showed that for linear models with Gaussian data, interpolation (zero training error) can generalise well when most singular values of the data matrix are small — the model "memorises" noise in directions orthogonal to the signal.

### R.2 Neural Tangent Kernel (NTK) Theory

Wide neural networks in the "lazy training" (NTK) regime behave like kernel methods. The generalisation bound becomes:
$$R(\hat{h}) \leq \hat{R}(\hat{h}) + O\!\left(\sqrt{\frac{\hat{\mathbf{y}}^\top (K_{nn})^{-1} \hat{\mathbf{y}} + \log(1/\delta)}{n}}\right)$$

where $K_{nn}$ is the NTK kernel matrix. The key quantity $\hat{\mathbf{y}}^\top K_{nn}^{-1} \hat{\mathbf{y}}$ is the **effective complexity** — analogous to Rademacher complexity but computed from the kernel. For networks trained with weight decay, this remains bounded even as depth and width grow.

### R.3 PAC-Bayes for Overparameterised Models

The most successful approach for non-vacuous bounds on deep networks uses PAC-Bayes. The bound:
$$\mathbb{E}_{h\sim Q}[R(h)] \leq \mathbb{E}_{h\sim Q}[\hat{R}(h)] + \sqrt{\frac{D_{\mathrm{KL}}(Q\|P) + \log(1/\delta)}{2(n-1)}}$$

applies to a **stochastic** classifier $h \sim Q$. For weight-perturbed networks (Gaussian posterior with small variance $\sigma^2$):
$$D_{\mathrm{KL}}(Q\|P) \approx \frac{\|\hat{\theta} - \theta_0\|_2^2}{2\sigma_0^2}$$

This is small when the fine-tuned weights $\hat{\theta}$ don't stray far from the initialisation $\theta_0$. SAM (Sharpness-Aware Minimisation) explicitly seeks flat minima where the posterior concentrates near a broad region around $\hat{\theta}$, minimising this KL term.

---

## Appendix S: Concentration in Continuous-Time Settings

### S.1 Azuma's Inequality for Martingales

We've seen Azuma's inequality used to prove McDiarmid. Here's the full statement:

**Theorem (Azuma-Hoeffding).** Let $(Z_0, Z_1, \ldots, Z_n)$ be a martingale (or supermartingale) with $Z_0 = 0$ and $|Z_k - Z_{k-1}| \leq c_k$ a.s. Then:
$$P(Z_n \geq t) \leq \exp\!\left(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\right)$$

**Proof.** Apply the Chernoff method: $P(Z_n \geq t) \leq e^{-st} \mathbb{E}[e^{sZ_n}]$. For a martingale, $\mathbb{E}[e^{sZ_k} \mid Z_{k-1}] \leq e^{sZ_{k-1}} \cdot e^{s^2c_k^2/2}$ by Hoeffding's lemma (conditional on $Z_{k-1}$, the increment $Z_k - Z_{k-1} \in [-c_k, c_k]$ is bounded). Telescoping: $\mathbb{E}[e^{sZ_n}] \leq e^{s^2\sum c_k^2/2}$. Optimise: $s = t/\sum c_k^2$. $\blacksquare$

### S.2 Doob's Optional Stopping

**Theorem.** If $(M_n)$ is a martingale and $\tau$ is a bounded stopping time, then $\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$.

**Application to SGD.** Define $M_t = \|\theta_t - \theta^*\|_2^2 - t \cdot C\eta^2\sigma^2$ where $C$ is a constant. Under mild conditions, $M_t$ is a supermartingale. Azuma's inequality applied to the stopped process gives high-probability guarantees on convergence time.

This is the mathematical foundation of *high-probability* SGD convergence — stronger than the usual in-expectation guarantees.

---

## Appendix T: Summary Diagrams

### T.1 The Proof Technique Map

```
PROOF TECHNIQUES FOR CONCENTRATION INEQUALITIES
════════════════════════════════════════════════════════════════════════

  Core tool: Markov's inequality
  ───────────────────────────────────────────────────────────────────

  Apply Markov to (X-μ)²          ──► Chebyshev
  Apply Markov to e^{sX}          ──► Chernoff method
       ├── with Hoeffding's lemma  ──► Hoeffding's inequality
       ├── with Bernstein's lemma  ──► Bernstein's inequality
       └── with Binomial MGF       ──► Chernoff for Bin(n,p)

  Apply Azuma to Doob martingale  ──► McDiarmid's inequality
  Union bound over hypothesis class ──► PAC generalisation bound
  Union bound + covering number    ──► Infinite-class PAC
  Rademacher symmetrisation        ──► Rademacher complexity bound

  ───────────────────────────────────────────────────────────────────
  Common pattern: P(X ≥ t) ≤ E[e^{sX}] / e^{st}  (all MGF methods)
════════════════════════════════════════════════════════════════════════
```

### T.2 Bound Quality Comparison

```
TAIL PROBABILITY COMPARISON — X = mean of n=100 Bernoulli(0.5), t=0.1
════════════════════════════════════════════════════════════════════════

  Method           Bound          Notes
  ─────────────────────────────────────────────────────────────────
  Exact (CLT)     ~0.046         Normal approx; valid for large n
  Hoeffding        0.135          Tight for this case
  Chebyshev        0.250          3× looser
  Markov           0.500          5× looser (applied to |X-0.5|/0.5)

  Same setup, t=0.2:
  Exact            ~0.000046
  Hoeffding        0.018          400× looser
  Chebyshev        0.0625         1360× looser

════════════════════════════════════════════════════════════════════════
```


---

## Appendix U: Proofs of Sub-Gaussian Closure and Examples

### U.1 Cosh Inequality for Rademacher Variables

**Claim.** For $\varepsilon \in \{-1, +1\}$ uniform: $\mathbb{E}[e^{t\varepsilon}] = \cosh(t) \leq e^{t^2/2}$.

**Proof.** $\cosh(t) = \sum_{k=0}^\infty t^{2k}/(2k)! \leq \sum_{k=0}^\infty t^{2k}/(2^k k!) = \sum_{k=0}^\infty (t^2/2)^k/k! = e^{t^2/2}$, using $(2k)! \geq 2^k k!$.

**Interpretation.** Rademacher variables are the "most concentrated" bounded variables in $[-1,1]$: their MGF is exactly $\cosh(t)$, slightly below the sub-Gaussian bound $e^{t^2/2}$. This is also why Rademacher complexity multiplied by 2 appears in generalisation bounds — the factor accounts for the Rademacher supremum.

### U.2 Sub-Gaussian Parameter for Bounded Variables

**Claim.** If $X \in [a,b]$ with $\mathbb{E}[X] = 0$, then $X$ is $\frac{(b-a)^2}{4}$-sub-Gaussian.

The parameter $\frac{(b-a)^2}{4}$ is achieved when $X = \frac{b-a}{2}\varepsilon$ where $\varepsilon$ is Rademacher — a two-point distribution at the endpoints. This is the "hardest" distribution on $[a,b]$ for Hoeffding's lemma.

**Tightness example.** For $X \in \{-1, +1\}$ uniform: variance $= 1$, sub-Gaussian parameter $= \frac{(1-(-1))^2}{4} = 1$. They agree! For the Rademacher distribution, sub-Gaussian parameter = variance. For other distributions in $[-1,1]$, sub-Gaussian parameter $\geq$ variance.

### U.3 Sub-Gaussian Characterisation via Tail Probabilities

The following are equivalent for a mean-zero random variable $X$:
1. (MGF) $\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2}$ for all $t$
2. (Tail) $P(|X| \geq t) \leq 2e^{-t^2/(2\sigma^2)}$ for all $t \geq 0$  
3. (Moments) $\|X\|_{L^p} = (\mathbb{E}[|X|^p])^{1/p} \leq \sigma\sqrt{p}$ for all $p \geq 1$

The equivalence (up to constants) is non-trivial. The key step: $(2) \Rightarrow (1)$ uses $\mathbb{E}[e^{tX}] = 1 + \sum_{k=1}^\infty \frac{t^k \mathbb{E}[X^k]}{k!}$ and bounds $\mathbb{E}[X^k]$ from the tail condition.

---

## Appendix V: Exercises with Solutions

### V.1 Markov and Chebyshev Calculations

**Problem.** $X \sim \text{Poisson}(5)$. Compute $P(X \geq 12)$ using:
(a) Markov's inequality ($\mathbb{E}[X] = 5$)
(b) Chebyshev ($\operatorname{Var}(X) = 5$, so $t = 12 - 5 = 7$, $k = 7/\sqrt{5} \approx 3.13$)
(c) The exact value (Poisson CDF)

**Solution.**
(a) Markov: $P(X \geq 12) \leq 5/12 \approx 0.417$
(b) Chebyshev: $P(X \geq 12) \leq P(|X - 5| \geq 7) \leq 5/49 \approx 0.102$
(c) Exact: $P(X \geq 12) = 1 - \sum_{k=0}^{11} e^{-5}5^k/k! \approx 0.020$

Chebyshev is 5× loose; Markov is 21× loose. Both bounds hold but are very conservative.

### V.2 Hoeffding Bound Computation

**Problem.** We estimate the click-through rate $p$ of an ad by showing it to $n$ users and recording clicks $X_i \in \{0,1\}$. We observe $\hat{p} = 0.08$ from $n = 2000$ users. Give a 99% Hoeffding confidence interval.

**Solution.** With $\delta = 0.01$, $b-a=1$:
$$\varepsilon = \sqrt{\frac{\log(2/0.01)}{2 \cdot 2000}} = \sqrt{\frac{\log 200}{4000}} = \sqrt{\frac{5.298}{4000}} = \sqrt{0.001325} \approx 0.0364$$

99% CI: $[0.08 - 0.0364, 0.08 + 0.0364] = [0.0436, 0.1164]$.

Interpretation: we're 99% confident the true CTR is between 4.4% and 11.6% — a very wide interval! To tighten to ±1%: need $n \geq \frac{\log 200}{2(0.01)^2} \approx 26{,}490$ users.

### V.3 Chernoff vs Hoeffding

**Problem.** $S \sim \text{Bin}(500, 0.02)$, so $\mu = 10$. Bound $P(S \geq 20)$.

**Hoeffding:** $P(S \geq 20) = P(\bar{X} \geq 0.04) \leq \exp(-2 \cdot 500 \cdot (0.04-0.02)^2) = e^{-0.4} \approx 0.67$. Useless!

**Chernoff:** $\delta = (20-10)/10 = 1$. $P(S \geq 20) \leq e^{-10 \cdot 1^2/3} = e^{-3.33} \approx 0.036$.

**Exact:** $P(S \geq 20) \approx 0.0085$ (Poisson approximation: $e^{-10}\sum_{k=20}^\infty 10^k/k!$).

Chernoff is 4× loose; Hoeffding is essentially vacuous (bound > 0.5 is useless). For rare events with $p \ll 1$ and large $n$, Chernoff is the right tool.

### V.4 McDiarmid on Dropout

**Problem.** A network output $f(\mathbf{z}_1, \ldots, \mathbf{z}_L)$ depends on $L=10$ layer activations. Dropout independently zeroes each activation, changing the contribution of neuron $j$ in layer $l$ by at most $c_{lj} = |w_{lj}|\cdot a_{lj}^{\max} / (1-p)$. If $\sum_{l,j} c_{lj}^2 = 4$, bound $P(|f - \mathbb{E}[f]| \geq 0.5)$.

**Solution.** By McDiarmid: $P(|f - \mathbb{E}[f]| \geq 0.5) \leq 2\exp(-2 \cdot 0.25/4) = 2e^{-0.125} \approx 1.76$. The bound is vacuous! This means either $c_{lj}$ are too large, or $t$ is too small relative to the noise.

With larger networks and better weight norms ($\sum c_{lj}^2 = 0.1$): $2e^{-2 \cdot 0.25/0.1} = 2e^{-5} \approx 0.013$. Now meaningful: dropout output deviates by more than 0.5 less than 1.3% of the time.

---

## Appendix W: Notation Reference for This Section

| Symbol | Meaning |
|--------|---------|
| $P(A)$ | Probability of event $A$ |
| $\mu = \mathbb{E}[X]$ | Mean of $X$ |
| $\sigma^2 = \operatorname{Var}(X)$ | Variance of $X$ |
| $M_X(t) = \mathbb{E}[e^{tX}]$ | Moment generating function |
| $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ | Sample mean |
| $R(h)$ | True (population) risk of hypothesis $h$ |
| $\hat{R}(h)$ | Empirical risk of $h$ on training data |
| $\mathcal{H}$ | Hypothesis class |
| $d_{VC}$ | VC dimension of $\mathcal{H}$ |
| $\Pi_\mathcal{H}(n)$ | Growth function |
| $\mathfrak{R}_n(\mathcal{F})$ | Rademacher complexity |
| $\hat{\mathfrak{R}}_S(\mathcal{F})$ | Empirical Rademacher complexity |
| $\varepsilon$ | Accuracy parameter (how close to optimal) |
| $\delta$ | Confidence parameter (failure probability) |
| $c_i$ | Bounded differences constant for coordinate $i$ |
| $\varepsilon$-net | Covering set with spacing $\varepsilon$ |
| $\mathcal{N}(\mathcal{H}, d, \varepsilon)$ | Covering number: size of smallest $\varepsilon$-net |
| $D_{\mathrm{KL}}(P\|Q)$ | KL divergence from $Q$ to $P$ |
| $I(a) = \sup_t(ta - \log M(t))$ | Cramér rate function |
| $\sigma_i$ | Rademacher random variable $\in \{-1,+1\}$ |
| $B_\delta(h)$ | Ball of radius $\delta$ around hypothesis $h$ |


---

## Appendix X: More on Rademacher Complexity

### X.1 Computing Rademacher Complexity for Common Classes

**Halfspaces with margin.** For $\mathcal{H}_\gamma = \{\mathbf{x} \mapsto \operatorname{sign}(\mathbf{w}^\top \mathbf{x}): \|\mathbf{w}\|_2 \leq 1, \gamma\text{-margin}\}$ on data with $\|\mathbf{x}^{(i)}\|_2 \leq 1$:
$$\mathfrak{R}_n(\mathcal{H}_\gamma) \leq \frac{1}{\gamma\sqrt{n}}$$

The margin $\gamma$ effectively replaces dimension $d$! This is why SVMs with large margins generalise well in high dimensions — the effective complexity is $O(1/\gamma^2 n)$, not $O(d/n)$.

**Two-layer neural network.** For $\mathcal{F} = \{\mathbf{x} \mapsto \mathbf{v}^\top \sigma(W\mathbf{x}): \|W\|_F \leq B_W, \|\mathbf{v}\|_1 \leq B_v\}$:
$$\mathfrak{R}_n(\mathcal{F}) \leq \frac{B_v B_W C \sqrt{2\log(2d)}}{\sqrt{n}}$$

where $C = \max_i \|\mathbf{x}^{(i)}\|_2$. The $\log d$ factor comes from the union bound over $d$ output neurons; the $\sqrt{n}$ denominator shows $O(1/\sqrt{n})$ rate as expected.

### X.2 Rademacher Complexity and Mutual Information

A deep connection: the Rademacher complexity $\hat{\mathfrak{R}}_S(\mathcal{F})$ can be bounded via mutual information between the training labels and the class predictions. For bounded function classes:
$$\hat{\mathfrak{R}}_S(\mathcal{F}) \leq \sqrt{\frac{2I(Y^n; f(X^n))}{n}}$$

where $I(Y^n; f(X^n))$ is the mutual information between labels and class predictions. This connects concentration theory to information theory — a richer theory being actively developed (2020-2026) under "information-theoretic generalisation bounds."

### X.3 Monte Carlo Estimation of Rademacher Complexity

In practice, $\hat{\mathfrak{R}}_S(\mathcal{F})$ can be estimated by Monte Carlo:

```python
def estimate_rademacher(X, W_norm=1.0, T=1000):
    """Monte Carlo estimate of Rademacher complexity for ||w||_2 <= W_norm."""
    n, d = X.shape
    rademachers = []
    for _ in range(T):
        sigma = np.random.choice([-1, 1], size=n)
        # Optimal w: proportional to X^T sigma, normalised
        w_opt = X.T @ sigma
        w_norm = np.linalg.norm(w_opt)
        if w_norm > 0:
            w_opt = w_opt * W_norm / w_norm
        rademachers.append((sigma @ (X @ w_opt)) / n)
    return np.mean(rademachers)
```

For a dataset of $n=500$ MNIST digits (784-dimensional), $W_\text{norm}=1$: empirical $\hat{\mathfrak{R}}_S \approx 0.038$. Rademacher bound: test error $\leq$ train error + $2 \cdot 0.038 + \sqrt{\log(20)/1000} \approx$ train error + 0.076 + 0.055. For 2% train error: test error $\leq 13.1\%$ — loose but not vacuous!

---

## Appendix Y: Bounds in the Context of Modern ML Systems

### Y.1 LLM Benchmark Evaluation Standards

As of 2026, serious LLM evaluations use concentration-aware practices:

**HELM (Holistic Evaluation of Language Models):** Uses 1000+ examples per scenario, reporting bootstrap confidence intervals. Assumes bounded loss and applies Hoeffding-style reasoning.

**BIG-bench:** 200+ tasks, with tasks having 100-10000 examples. Standard deviations are reported, equivalent to reporting $\sigma/\sqrt{n}$ which matches the leading term in Hoeffding CI.

**LiveBench:** Uses human-pairwise comparisons (Elo-style). The Elo update rule has a concentration interpretation: the ELO gap needed for significance scales as $O(1/\sqrt{n_{\text{comparisons}}})$.

**Statistical significance in model comparisons.** With $n$ test examples and two models achieving accuracies $\hat{p}_1$ and $\hat{p}_2$:
- Paired comparison: use McNemar's test (exact)
- Unpaired: significance threshold $|\hat{p}_1 - \hat{p}_2| \geq 2\sqrt{\frac{\log(4/\delta)}{2n}}$ (Hoeffding)
- For $n = 1000$, $\delta = 0.05$: need difference $\geq 2 \times 0.042 = 8.4\%$ for significance

### Y.2 Fine-Tuning Sample Complexity via PAC-Bayes

When fine-tuning a pretrained LLM on task-specific data:

**Prior:** Pre-trained weights $\theta_0$ from language model pre-training on internet-scale data
**Posterior:** Fine-tuned weights $\hat{\theta}$ on $n$ task examples
**PAC-Bayes bound:** 
$$R(\hat{\theta}) \leq \hat{R}(\hat{\theta}) + \sqrt{\frac{\|\hat{\theta} - \theta_0\|_2^2/(2\sigma_0^2) + \log(1/\delta)}{2(n-1)}}$$

The $\|\hat{\theta} - \theta_0\|_2^2$ term is small when fine-tuning doesn't move far from pre-training — exactly what LoRA enforces by constraining $\Delta W = AB$. With rank $r \ll \min(d,k)$:
$$\|\hat{\theta} - \theta_0\|_2^2 = \|AB\|_F^2 \leq \|A\|_F^2 \|B\|_F^2 \leq r \cdot \text{(per-param budget)}$$

PAC-Bayes theoretically justifies why LoRA generalises better than full fine-tuning on small datasets.

### Y.3 Privacy-Preserving ML via Concentration

**Differential Privacy (DP)** and concentration inequalities are deeply connected. An algorithm is $(\varepsilon, \delta)$-DP if changing one training example changes the output distribution by at most $\varepsilon$ in TV distance, except with probability $\delta$.

**DP-SGD (Abadi et al., 2016):** Clips each gradient to norm $C$, adds Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$. The privacy guarantee follows from the Gaussian mechanism, and the utility (convergence) uses concentration inequalities on the noisy gradients.

**McDiarmid connection:** DP is essentially algorithmic stability with $c_i = C/n$ (each example changes the model by at most $C/n$ in some norm). McDiarmid's inequality gives the same $O(\exp(-nt^2/C^2))$ bound for both privacy and stability.


---

## Appendix Z: Complete Theorem Index

This appendix lists every named result in this section for quick reference.

### Moment-Based Inequalities

**Theorem 2.1 (Markov).** $X \geq 0$, $\mathbb{E}[X] = \mu$: $P(X \geq t) \leq \mu/t$.

**Theorem 2.2 (Chebyshev).** $\operatorname{Var}(X) = \sigma^2$: $P(|X-\mu| \geq t) \leq \sigma^2/t^2$.

**Theorem 2.3 (Cantelli).** One-sided: $P(X - \mu \geq t) \leq \sigma^2/(\sigma^2 + t^2)$.

### Sub-Gaussian Inequalities

**Definition 3.1 (Sub-Gaussian).** $X$ is $\sigma^2$-sub-G if $\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2}$ for all $t$.

**Theorem 3.2 (Sub-Gaussian tail).** $P(X \geq t) \leq e^{-t^2/(2\sigma^2)}$.

**Theorem 3.3 (Sub-Gaussian sum).** Independent $\sigma_i^2$-sub-G: sum is $\sum \sigma_i^2$-sub-G.

### Hoeffding-Type Inequalities

**Lemma 4.1 (Hoeffding's lemma).** $X \in [a,b]$, $\mathbb{E}[X]=0$: $\mathbb{E}[e^{tX}] \leq e^{t^2(b-a)^2/8}$.

**Theorem 4.2 (Hoeffding).** Independent $X_i \in [a_i,b_i]$: $P(\sum(X_i-\mu_i) \geq t) \leq e^{-2t^2/\sum(b_i-a_i)^2}$.

**Theorem 4.3 (Two-sided Hoeffding).** $P(|\bar{X}_n-\mu| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$.

### Chernoff Bounds

**Theorem 5.1 (Chernoff method).** $P(X \geq t) \leq \inf_{s>0} e^{-st} M_X(s)$.

**Theorem 5.2 (Chernoff upper tail).** $S \sim \operatorname{Bin}(n,p)$, $\delta > 0$: $P(S \geq (1+\delta)\mu) \leq (e^\delta/(1+\delta)^{1+\delta})^\mu$.

**Theorem 5.3 (Chernoff multiplicative).** $P(S \geq (1+\delta)\mu) \leq e^{-\mu\delta^2/3}$ for $\delta \in (0,1)$.

**Theorem 5.4 (Chernoff lower tail).** $P(S \leq (1-\delta)\mu) \leq e^{-\mu\delta^2/2}$.

### Bernstein and McDiarmid

**Theorem 6.2 (Bernstein).** $|X_i| \leq c$, variance $\nu^2$: $P(\sum X_i \geq t) \leq \exp(-t^2/2/(\nu^2+ct/3))$.

**Theorem 7.2 (McDiarmid).** $f$ with $c_i$-bounded differences: $P(f-\mathbb{E}[f] \geq t) \leq e^{-2t^2/\sum c_i^2}$.

**Theorem S.1 (Azuma).** Martingale $|Z_k-Z_{k-1}| \leq c_k$: $P(Z_n \geq t) \leq e^{-t^2/2\sum c_k^2}$.

### Statistical Learning Theory

**Lemma 8.1 (Union bound).** $P(\bigcup A_i) \leq \sum P(A_i)$.

**Lemma 8.3 (Covering number).** $\ell_2$-ball: $\mathcal{N}(\mathcal{B}_R, \ell_2, \varepsilon) \leq (3R/\varepsilon)^d$.

**Theorem 9.2 (Finite-class PAC).** $n \geq (\log(|\mathcal{H}|/\delta))/(2\varepsilon^2) \Rightarrow$ uniform convergence.

**Lemma 9.4 (Sauer-Shelah).** $\Pi_\mathcal{H}(n) \leq (en/d)^d$ for $n \geq d = d_{VC}$.

**Theorem 9.5 (VC bound).** $R(h) \leq \hat{R}(h) + \sqrt{(d\log(2n/d)+\log(4/\delta))/(2n)}$.

**Theorem 10.2 (Rademacher bound).** $R(f) \leq \hat{R}(f) + 2\mathfrak{R}_n(\mathcal{F}) + \sqrt{\log(1/\delta)/(2n)}$.

**Theorem E.4 (PAC-Bayes).** $\mathbb{E}_Q[R(h)] \leq \mathbb{E}_Q[\hat{R}(h)] + \sqrt{(D_{\mathrm{KL}}(Q\|P)+\log(1/\delta))/(2(n-1))}$.

**Theorem J.2 (Stability).** $\beta$-stable algo: $R \leq \hat{R} + 2\beta + (4n\beta+1)\sqrt{\log(1/\delta)/(2n)}$.

---

*This completes §05 Concentration Inequalities. The material here is used extensively in [Chapter 7: Statistics](../../07-Statistics/README.md) for confidence intervals and hypothesis tests, and throughout learning theory.*

[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Stochastic Processes →](../06-Stochastic-Processes/notes.md)

---

## Supplementary: Concentration in High Dimensions

### The Curse and Blessing of Dimensionality

High-dimensional geometry is counterintuitive. In $\mathbb{R}^d$ with large $d$:

- Almost all the volume of a ball lies near its surface
- Random unit vectors are nearly orthogonal: $|\langle u, v \rangle| \approx 1/\sqrt{d}$ w.h.p.
- The norm of a Gaussian vector concentrates: $\|\mathbf{x}\|_2 \approx \sqrt{d}$ for $\mathbf{x} \sim \mathcal{N}(0, I_d)$

These facts follow directly from concentration inequalities applied to sums of squares.

**Gaussian norm concentration.** Let $\mathbf{x} \sim \mathcal{N}(0, I_d)$. Then $\|\mathbf{x}\|_2^2 = \sum_{i=1}^d X_i^2$ where $X_i \stackrel{iid}{\sim} \mathcal{N}(0,1)$. By Bernstein's inequality (since $X_i^2 - 1$ is sub-exponential):
$$P\!\left(\left|\,\|\mathbf{x}\|_2 - \sqrt{d}\,\right| \geq t\right) \leq 2e^{-ct^2}$$
for a universal constant $c > 0$.

**Implication for dot products.** For two independent $\mathbf{x}, \mathbf{y} \sim \mathcal{N}(0, I_d/d)$:
$$P\!\left(\left|\langle \mathbf{x}, \mathbf{y}\rangle\right| \geq t/\sqrt{d}\right) \leq 2e^{-ct^2}$$

This is why scaled dot-product attention uses $1/\sqrt{d_k}$ — it keeps logits $O(1)$ despite $d_k$-dimensional embeddings.

### Geometry of Softmax

Let $\mathbf{q}, \mathbf{k}_1, \ldots, \mathbf{k}_n \in \mathbb{R}^{d_k}$ be the query and keys in a transformer. The attention weight on key $j$ is:
$$\alpha_j = \frac{e^{\mathbf{q}^\top \mathbf{k}_j / \sqrt{d_k}}}{\sum_{\ell} e^{\mathbf{q}^\top \mathbf{k}_\ell / \sqrt{d_k}}}$$

Without the $1/\sqrt{d_k}$ scaling, the logits $\mathbf{q}^\top \mathbf{k}_j$ have variance $d_k$ (for unit random vectors), pushing softmax into near-zero gradient regions — a form of implicit dimension dependence that concentration inequalities make precise.

### Covering Numbers in Transformer Weight Space

The number of parameters in a transformer layer is $\Theta = O(d^2)$. For LoRA with rank $r$, the effective dimensionality is $O(dr)$. By the $\varepsilon$-covering number bound:
$$\log \mathcal{N}(\mathcal{F}_{\text{LoRA}}, \varepsilon, \|\cdot\|) = O\!\left(dr \log(1/\varepsilon)\right)$$

This converts directly to a PAC generalisation bound showing that LoRA generalises with $O(\sqrt{dr \log n / n})$ excess risk — substantially better than $O(\sqrt{d^2 \log n / n})$ for full fine-tuning. Concentration inequalities thus provide the theoretical foundation for parameter-efficient fine-tuning.

### Summary: Why This Section Matters

Concentration inequalities are the engine behind every finite-sample guarantee in machine learning. They explain:

1. **Why more data helps**: bounds shrink as $1/\sqrt{n}$
2. **Why simpler models generalise**: bounds grow with $\sqrt{\log |\mathcal{H}|}$ or $\sqrt{d_{\text{VC}}}$
3. **Why high dimensions are manageable**: Gaussian concentration keeps norms stable
4. **Why LoRA works**: Rademacher complexity of low-rank updates is provably small
5. **Why LLM benchmarks need many examples**: CIs on accuracy require $n \propto 1/\varepsilon^2$

The progression Markov → Chebyshev → sub-Gaussian → Chernoff → McDiarmid is not merely a tour of tricks — it is a systematic strengthening of assumptions that yields exponentially tighter guarantees, mirroring the progress from "some data" to "structured data" in modern ML.


---

_End of §05 Concentration Inequalities_

> **Next:** [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md) — where concentration inequalities meet martingales in continuous time.

> **Back:** [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md) — the moment-based tools (MGF, Jensen) used throughout this section.

> **Chapter:** [§06 Probability Theory README](../README.md)
