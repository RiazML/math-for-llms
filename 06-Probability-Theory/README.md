[← Previous Chapter: Multivariate Calculus](../05-Multivariate-Calculus/README.md) | [Next Chapter: Statistics →](../07-Statistics/README.md)

---

# Chapter 6 — Probability Theory

> _"Probability is not about chance — it is about the precise language for reasoning under uncertainty. That language is the foundation of every machine learning model."_

## Overview

Probability theory is the mathematical framework for reasoning about uncertainty, and every component of modern machine learning is probabilistic at its core. Training data is a sample from an unknown distribution; model parameters are drawn randomly at initialisation; predictions are distributions over outcomes; generative models explicitly model data-generating processes; language model outputs are probability distributions over tokens.

This chapter builds the formal theory from sample spaces and probability axioms through random variables, common distributions, joint and conditional probability, expectation and moments, concentration inequalities, stochastic processes, and Markov chains. The goal is not abstract measure theory for its own sake — every concept is grounded in its ML applications, from cross-entropy loss to the central limit theorem underlying batch statistics to Markov chain Monte Carlo in Bayesian inference.

**The conceptual arc:** probability spaces (§01) → named distributions and their properties (§02) → joint and conditional probability, independence (§03) → expectation, variance, and moments (§04) → concentration inequalities and tail bounds (§05) → stochastic processes and martingales (§06) → Markov chains and steady-state theory (§07).

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|---|---|---|
| 01 | [Introduction and Random Variables](01-Introduction-and-Random-Variables/notes.md) | Probability spaces, axioms, discrete and continuous random variables, CDF/PDF/PMF | Probability axioms, sample space, events, random variables, CDF, PDF, PMF, Bernoulli, uniform; basic rules (addition, complement, conditional) |
| 02 | [Common Distributions](02-Common-Distributions/notes.md) | The named distributions that appear throughout ML; their parameters, moments, and relationships | Bernoulli, Binomial, Poisson, Gaussian, Exponential, Beta, Dirichlet, Categorical, Multinomial, Student-t, Gamma; moment generating functions; exponential family |
| 03 | [Joint Distributions](03-Joint-Distributions/notes.md) | Multivariate probability; conditional distributions; independence; Bayes' theorem | Joint PDF/PMF, marginalisation, conditional distributions, statistical independence, Bayes' theorem, chain rule of probability, multivariate Gaussian |
| 04 | [Expectation and Moments](04-Expectation-and-Moments/notes.md) | The expectation operator; variance, covariance, and higher moments; moment generating functions | Expected value, linearity of expectation, variance, standard deviation, covariance matrix, correlation, skewness, kurtosis, MGF, characteristic function, LOTUS |
| 05 | [Concentration Inequalities](05-Concentration-Inequalities/notes.md) | How probability mass concentrates around the mean; tail bounds used in statistical learning theory | Markov's inequality, Chebyshev's inequality, Hoeffding's inequality, Chernoff bounds, McDiarmid's inequality, PAC learning framework, generalisation bounds |
| 06 | [Stochastic Processes](06-Stochastic-Processes/notes.md) | Sequences of random variables indexed by time; convergence theorems; central limit theorem | Law of large numbers (weak and strong), central limit theorem, Gaussian processes, Brownian motion, stationary processes, ergodicity |
| 07 | [Markov Chains](07-Markov-Chains/notes.md) | Memoryless sequential processes; steady-state distributions; MCMC and its role in Bayesian ML | Markov property, transition matrices, steady-state and stationary distributions, detailed balance, MCMC (Metropolis-Hastings, Gibbs sampling), PageRank, hidden Markov models |

---

## Reading Order and Dependencies

```
01-Introduction-and-Random-Variables   (foundation: probability spaces, random variables)
        ↓
02-Common-Distributions                (vocabulary: named distributions and their properties)
        ↓
03-Joint-Distributions                 (multivariate: joint, conditional, Bayes' theorem)
        ↓
04-Expectation-and-Moments             (summary statistics: E[X], Var[X], covariance)
        ↓
05-Concentration-Inequalities          (bounds: how far can a random variable stray from its mean?)
        ↓
06-Stochastic-Processes                (sequences: LLN, CLT, Gaussian processes)
        ↓
07-Markov-Chains                       (structured processes: MCMC, Bayesian inference machinery)
        ↓
07-Statistics (Chapter 7)             (estimation: MLE, MAP, confidence intervals, hypothesis tests)
```

---

## What Belongs Where — Canonical Homes

This table is the authoritative scoping guide. **If a topic has a canonical home, every other section must give at most a 1–2 paragraph preview with a forward/backward reference — never a full treatment.**

| Topic | Canonical Home | Preview Only In |
|---|---|---|
| Probability axioms (Kolmogorov) | §01 | — |
| Sample space, events, set operations | §01 | — |
| CDF, PDF, PMF definitions | §01 | §02 (used for named distributions) |
| Conditional probability $P(A\|B)$ | §01 | §03 (extended to joint distributions) |
| Bernoulli and uniform (simple intro) | §01 | §02 (full properties) |
| Named discrete distributions (Binomial, Poisson, Geometric, Categorical) | §02 | §01 (Bernoulli only), §03 (marginals) |
| Named continuous distributions (Gaussian, Exponential, Beta, Dirichlet, Gamma, Student-t) | §02 | §04 (moments of Gaussian), §06 (Gaussian process) |
| Exponential family and natural parameters | §02 | §04 (sufficient statistics and moments) |
| Moment generating functions | §02 | §04 (moments from MGF) |
| Joint distributions, joint PDF/PMF | §03 | §04 (expectation under joint) |
| Marginalisation | §03 | §04 (iterated expectation) |
| Statistical independence $X \perp Y$ | §03 | §01 (preview), §04 (Var[X+Y]=Var[X]+Var[Y] if independent) |
| Bayes' theorem | §03 | §01 (introduced as formula), §07 (Bayesian update) |
| Conditional independence | §03 | §07 (Markov property as conditional independence) |
| Multivariate Gaussian | §03 | §06 (as Gaussian process marginals) |
| Expected value (definition and linearity) | §04 | §01 (informal preview) |
| Variance, covariance, correlation | §04 | §02 (variance of named distributions), §03 (covariance matrix) |
| LOTUS (law of the unconscious statistician) | §04 | — |
| Jensen's inequality | §04 | §05 (used to derive Markov's inequality) |
| Moment generating function applications | §04 | §02 (MGF definitions) |
| Markov's inequality | §05 | §04 (preview: E[X] bounds tails) |
| Chebyshev's inequality | §05 | §04 (preview: Var[X] bounds tails) |
| Hoeffding's, Chernoff bounds | §05 | — |
| PAC learning and generalisation bounds | §05 | — |
| Union bound | §05 | §01 (probability of union, preview) |
| Law of large numbers | §06 | §04 (preview: sample mean → E[X]) |
| Central limit theorem | §06 | §04 (preview: sum of iid → Gaussian) |
| Gaussian processes | §06 | §02 (Gaussian distribution) |
| Brownian motion | §06 | — |
| Markov property | §07 | §03 (conditional independence preview) |
| Transition matrices and steady state | §07 | — |
| Detailed balance (reversibility) | §07 | — |
| Metropolis-Hastings, Gibbs sampling | §07 | §03 (Bayes' theorem used) |
| PageRank | §07 | — |
| Hidden Markov models | §07 | §03 (conditional independence structure) |

---

## Overlap Danger Zones

### 1. Probability Axioms ↔ Common Distributions
- **§01** introduces probability through axioms and defines the simplest distributions (Bernoulli, uniform) as concrete examples of the formalism.
- **§02** gives the complete treatment of each named distribution — parameters, PDF/PMF, CDF, mean, variance, MGF, relationships. §02 must not re-derive the axioms; §01 must not provide the full moments and relationships of Binomial, Gaussian, etc.

### 2. Conditional Probability ↔ Joint Distributions ↔ Bayes
- **§01** defines conditional probability $P(A|B) = P(A \cap B)/P(B)$ and states Bayes' theorem as a formula.
- **§03** develops the full machinery: joint PDF/PMF, marginalisation, conditional distributions for random variables, chain rule of probability, and Bayes' theorem with prior/likelihood/posterior structure.
- **§07** applies Bayesian updates in the Markov chain setting. §07 must backward-reference §03 for Bayes' theorem.

### 3. Expectation ↔ Moments ↔ MGF
- **§04** is the exclusive home of the expectation operator, variance, covariance, higher moments, LOTUS, and moment generating functions as analytical tools.
- **§02** may state the mean and variance of each named distribution (as facts), but must not derive them from first principles — §04 owns those derivations.

### 4. Concentration Inequalities ↔ Law of Large Numbers
- **§05** proves Markov's and Chebyshev's inequalities and derives Hoeffding/Chernoff bounds, culminating in PAC-learning generalisation bounds.
- **§06** proves the weak and strong LLN using the tools from §05. §06 should backward-reference §05 rather than re-proving the concentration results it uses.

### 5. Markov Chains ↔ Conditional Independence
- **§03** defines conditional independence $X \perp Y | Z$.
- **§07** uses conditional independence as the definition of the Markov property: $X_{t+1} \perp X_{0:t-1} | X_t$. §07 must backward-reference §03 rather than re-define conditional independence.

### 6. Gaussian Distribution ↔ Gaussian Process ↔ Multivariate Gaussian
- **§02** defines the univariate Gaussian: PDF, CDF (via $\Phi$), moments, MGF, standard normal.
- **§03** defines the multivariate Gaussian: joint PDF, covariance matrix, marginals, conditionals, affine transformations.
- **§06** defines the Gaussian process as an infinite-dimensional extension: any finite collection of function values follows a multivariate Gaussian. §06 must backward-reference §02 and §03 rather than re-derive Gaussian properties.

---

## Forward and Backward Reference Format

**Forward reference** (full treatment is later):
```markdown
> **Preview: Bayes' Theorem for Distributions**
> For random variables $X$ and $Y$ with joint density $p(x,y)$, Bayes' theorem becomes
> $p(x|y) = p(y|x)p(x)/p(y)$, giving the posterior over $x$ given observation $y$.
>
> → _Full treatment: [Joint Distributions](../03-Joint-Distributions/notes.md)_
```

**Backward reference** (builds on earlier section):
```markdown
> **Recall:** The expected value $\mathbb{E}[X] = \int x\, p(x)\, dx$ was defined in
> [Expectation and Moments](../04-Expectation-and-Moments/notes.md). The covariance
> $\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$ defined there is the key
> parameter of the multivariate Gaussian developed here.
```

---

## Key Cross-Chapter Dependencies

**From Chapter 5 — Multivariate Calculus:**
- Integration in multiple variables (§01/§03 use multivariate integrals for marginalisation)
- Lagrange multipliers (§04 in Ch5) → maximum entropy distributions (§02 exponential family derivation)
- Gradients and log-derivatives → score functions, Fisher information matrix (§03, §04)

**From Chapter 4 — Calculus Fundamentals:**
- Integration (computing CDFs, normalisation constants, moments)
- Series (computing MGFs, Poisson approximation)

**Into Chapter 7 — Statistics:**
- §02 (distributions) → likelihood functions and MLE
- §03 (Bayes' theorem) → MAP estimation and Bayesian inference
- §04 (expectation, variance) → unbiased estimators, sample statistics
- §05 (concentration) → confidence intervals and hypothesis testing
- §06 (CLT) → asymptotic normality of estimators
- §07 (Markov chains) → MCMC-based posterior sampling

**Into Chapter 8 — Optimisation:**
- §02 (Gaussian, softmax) → cross-entropy loss derivation
- §03 (conditional distributions) → EM algorithm (expectation step)
- §04 (KL divergence, entropy) → variational inference

**Into Chapter 9 — Information Theory:**
- §02 (exponential family) → natural exponential family and sufficient statistics
- §04 (entropy = $-\mathbb{E}[\log p(X)]$) → Shannon entropy definition
- §05 (MGF) → Cramér's theorem, large deviations

---

## ML Concept Map

| ML Concept | Probability Theory Foundation | Section |
|---|---|---|
| Cross-entropy loss $-\sum y \log \hat{p}$ | Categorical distribution; negative log-likelihood | §02, §03 |
| Softmax output layer | Categorical distribution parameterisation | §02 |
| Dropout regularisation | Bernoulli random variables over activations | §01, §02 |
| Batch normalisation | Sample mean and variance as random variables; CLT | §04, §06 |
| Variational autoencoder (VAE) | Gaussian reparameterisation; KL divergence | §02, §03 |
| Bayesian neural networks | Prior × likelihood → posterior; Bayes' theorem | §03 |
| Gaussian processes for regression | Gaussian process prior; posterior conditioning | §06 |
| MCMC for posterior sampling | Markov chains; detailed balance; stationary distribution | §07 |
| Attention mechanism as soft lookup | Categorical distribution over keys | §02 |
| Language model outputs | Categorical distribution over vocabulary | §02 |
| Data augmentation | Distribution over augmented inputs | §01, §03 |
| Importance sampling | Change of measure; Radon-Nikodym derivative | §03, §06 |
| PAC generalisation bounds | Hoeffding's inequality; union bound | §05 |
| Double descent / bias-variance | Variance of estimators; bias-variance decomposition | §04 |
| Reinforcement learning (policy) | Markov decision process (MDP); Markov property | §07 |
| RLHF reward modelling | Bradley-Terry model; probability of preference | §02, §03 |
| Normalising flows | Change of variables formula for densities | §03 |
| Diffusion models | Gaussian noise schedule; score function | §02, §06 |

---

## Prerequisites

Before starting this chapter, ensure you are comfortable with:

- **Integration** (single and multi-variable): computing areas under curves, normalisation constants — [§03-Integration](../04-Calculus-Fundamentals/03-Integration/notes.md)
- **Series**: power series, convergence — [§04-Series-and-Sequences](../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)
- **Set theory basics**: union, intersection, complement — introduced in §01 of this chapter
- **Matrix algebra**: matrix–vector products, determinants, positive definite matrices — [Chapter 3](../03-Advanced-Linear-Algebra/README.md)
- **Lagrange multipliers** (for exponential family derivation): [§04-Optimality-Conditions](../05-Multivariate-Calculus/04-Optimality-Conditions/notes.md)

---

[← Previous Chapter: Multivariate Calculus](../05-Multivariate-Calculus/README.md) | [Next Chapter: Statistics →](../07-Statistics/README.md)
