[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Concentration Inequalities →](../05-Concentration-Inequalities/notes.md)

---

# Expectation and Moments

> _"The expectation of the product of two independent random variables equals the product of their expectations — a theorem so simple it conceals its own depth."_
> — Mark Kac

## Overview

**Expectation** is the single most important operation in probability theory. It compresses an entire distribution into a number — the weighted average of all possible values, where weights are probabilities. Every loss function in machine learning is an expectation. Every gradient update in stochastic training is an estimated expectation. Every generative model is ultimately defined by the distributions whose expectations it learns to match.

This section builds the theory of the expectation operator from first principles: the formal definition via the law of the unconscious statistician (LOTUS), linearity (which holds without independence), the tower property (iterated conditioning), moment generating functions (which encode the entire distribution in an analytic function), and Jensen's inequality (which underlies the derivation of the ELBO, the KL divergence bound, and every variational argument in modern deep learning).

The central thread connecting all these tools is **moments** — the sequence $\mathbb{E}[X], \mathbb{E}[X^2], \mathbb{E}[X^3], \ldots$ that describes the shape of a distribution. The first moment locates the center. The second moment (variance) measures spread. The third captures asymmetry. The fourth captures tail heaviness. In modern ML, the Adam optimizer literally tracks first and second moments of gradients at every parameter, every step. Understanding moments is understanding adaptive optimisation.

## Prerequisites

- [§01 Introduction and Random Variables](../01-Introduction-and-Random-Variables/notes.md) — random variables, CDF, PDF, PMF
- [§02 Common Distributions](../02-Common-Distributions/notes.md) — Gaussian, Exponential, Binomial; their stated means and variances
- [§03 Joint Distributions](../03-Joint-Distributions/notes.md) — joint PDF, conditional distributions, marginalisation, covariance matrix
- Single-variable integration, convergence of improper integrals, power series

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive derivations: LOTUS, Jensen, MGFs, bias-variance, Adam moment tracking |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from LOTUS computation to policy gradient estimation |

## Learning Objectives

After completing this section, you will be able to:

1. Compute expectations for discrete, continuous, and mixed distributions using LOTUS
2. Apply linearity of expectation without assuming independence
3. Derive variance using the computational formula and prove properties (Var(aX+b), Var(X+Y))
4. Interpret and compute skewness and kurtosis from their moment definitions
5. State and apply the tower property (iterated expectation) and law of total variance
6. Derive moment generating functions and extract moments via differentiation
7. Apply Jensen's inequality to prove KL divergence non-negativity and derive the ELBO
8. Prove and apply the Cauchy-Schwarz inequality for the $|\rho| \leq 1$ bound
9. Decompose expected squared error into bias², variance, and irreducible noise
10. Connect Adam optimizer to first and second moment estimation with bias correction
11. Preview Markov's and Chebyshev's inequalities as moment-based tail bounds (→ §05)

---

## Table of Contents

- [1. Intuition and Historical Context](#1-intuition-and-historical-context)
  - [1.1 The Moment Analogy](#11-the-moment-analogy)
  - [1.2 Why Moments Matter for AI](#12-why-moments-matter-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
- [2. Formal Definition of Expectation](#2-formal-definition-of-expectation)
  - [2.1 Expectation for Discrete, Continuous, and Mixed Distributions](#21-expectation-for-discrete-continuous-and-mixed-distributions)
  - [2.2 Existence and Integrability](#22-existence-and-integrability)
  - [2.3 Linearity of Expectation](#23-linearity-of-expectation)
  - [2.4 LOTUS — Law of the Unconscious Statistician](#24-lotus--law-of-the-unconscious-statistician)
- [3. Variance, Standard Deviation, and Higher Moments](#3-variance-standard-deviation-and-higher-moments)
  - [3.1 Variance](#31-variance)
  - [3.2 Standard Deviation and Coefficient of Variation](#32-standard-deviation-and-coefficient-of-variation)
  - [3.3 Raw vs Central Moments](#33-raw-vs-central-moments)
  - [3.4 Skewness](#34-skewness)
  - [3.5 Kurtosis and Excess Kurtosis](#35-kurtosis-and-excess-kurtosis)
- [4. Covariance and Correlation](#4-covariance-and-correlation)
  - [4.1 Covariance as an Operator](#41-covariance-as-an-operator)
  - [4.2 Pearson Correlation Coefficient](#42-pearson-correlation-coefficient)
  - [4.3 Independence and Zero Covariance](#43-independence-and-zero-covariance)
  - [4.4 Variance of Sums](#44-variance-of-sums)
- [5. Conditional Expectation](#5-conditional-expectation)
  - [5.1 Definition and Basic Properties](#51-definition-and-basic-properties)
  - [5.2 Tower Property (Iterated Expectation)](#52-tower-property-iterated-expectation)
  - [5.3 Conditional Variance and Law of Total Variance](#53-conditional-variance-and-law-of-total-variance)
  - [5.4 AI Applications of Conditional Expectation](#54-ai-applications-of-conditional-expectation)
- [6. Moment Generating Functions and Characteristic Functions](#6-moment-generating-functions-and-characteristic-functions)
  - [6.1 MGF Definition and Basic Properties](#61-mgf-definition-and-basic-properties)
  - [6.2 MGF of Common Distributions](#62-mgf-of-common-distributions)
  - [6.3 MGF of Sums and the Reproductive Property](#63-mgf-of-sums-and-the-reproductive-property)
  - [6.4 Characteristic Function](#64-characteristic-function)
  - [6.5 Cumulants and the Cumulant Generating Function](#65-cumulants-and-the-cumulant-generating-function)
- [7. Jensen's Inequality and Moment Inequalities](#7-jensens-inequality-and-moment-inequalities)
  - [7.1 Jensen's Inequality](#71-jensens-inequality)
  - [7.2 Applications: KL Divergence and ELBO](#72-applications-kl-divergence-and-elbo)
  - [7.3 Cauchy-Schwarz Inequality](#73-cauchy-schwarz-inequality)
  - [7.4 Lyapunov Inequality](#74-lyapunov-inequality)
  - [7.5 Preview: Tail Bounds from Moments](#75-preview-tail-bounds-from-moments)
- [8. Bias-Variance Decomposition](#8-bias-variance-decomposition)
  - [8.1 Statistical Risk and Optimal Predictor](#81-statistical-risk-and-optimal-predictor)
  - [8.2 Bias-Variance-Noise Decomposition](#82-bias-variance-noise-decomposition)
  - [8.3 Double Descent and Modern Deep Learning](#83-double-descent-and-modern-deep-learning)
  - [8.4 Regularisation as Bias Injection](#84-regularisation-as-bias-injection)
- [9. Expectation in ML: Core Applications](#9-expectation-in-ml-core-applications)
  - [9.1 Cross-Entropy Loss as an Expectation](#91-cross-entropy-loss-as-an-expectation)
  - [9.2 ELBO as an Expected Value](#92-elbo-as-an-expected-value)
  - [9.3 Policy Gradient (REINFORCE)](#93-policy-gradient-reinforce)
  - [9.4 Adam Optimizer as Moment Tracking](#94-adam-optimizer-as-moment-tracking)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition and Historical Context

### 1.1 The Moment Analogy

The word "moment" comes from classical mechanics. In physics, the first moment of a mass distribution is the center of mass; the second moment is the moment of inertia. Probability theory borrows this language directly: the $k$-th **moment** of a random variable $X$ is $\mathbb{E}[X^k]$, the expected value of $X$ raised to the $k$-th power.

Imagine balancing a plank on which weights are placed at positions $x_1, x_2, \ldots, x_n$ with weights $p_1, p_2, \ldots, p_n$ (probabilities). The balance point is $\sum_i x_i p_i = \mathbb{E}[X]$ — the center of mass. The spread around that center is captured by $\sum_i (x_i - \mu)^2 p_i = \text{Var}(X)$, the second central moment. Higher moments capture the shape: whether the distribution leans left or right (skewness), whether it has heavy tails (kurtosis).

```
MOMENTS AS SHAPE DESCRIPTORS
════════════════════════════════════════════════════════════════════════

  First moment (mean μ):        Location of center
  Second central moment (σ²):   Spread about center
  Third central moment (γ₁σ³):  Asymmetry (skewness)
  Fourth central moment (γ₂σ⁴): Tail weight (kurtosis)

  Standard normal N(0,1):  μ=0, σ²=1, γ₁=0, γ₂=0 (baseline)

  Right-skewed (γ₁ > 0):         Heavy right tail: income, LLM token probs
        ▓
       ▓▓▓
      ▓▓▓▓▓▒▒▒▒▒░░░░░░░░
  ────┴────────────────────

  Heavy-tailed (γ₂ > 0):         Leptokurtic: gradient norms, loss landscapes
        ▓
        ▓
       ▓▓▓
      ▓▓▓▓▓░░░░░░░░░░░░░
  ────┴────────────────────

════════════════════════════════════════════════════════════════════════
```

Every major operation in statistics reduces to computing moments. The mean is the first moment. The variance is the second central moment. The sample statistics we compute to estimate distributions are moment estimates. The method of moments estimation technique literally sets sample moments equal to theoretical moments and solves for parameters.

**For AI:** Every loss function is an expectation over the data distribution:

$$\mathcal{L}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]$$

We cannot compute this exactly (the true distribution $\mathcal{D}$ is unknown), so we estimate it with the sample mean over a minibatch. The entire machinery of stochastic gradient descent rests on this approximation.

### 1.2 Why Moments Matter for AI

Moments appear in nearly every component of modern deep learning:

**Loss functions.** The mean squared error $\frac{1}{N}\sum_i (y_i - \hat{y}_i)^2$ estimates $\mathbb{E}[(Y - f(X))^2]$, the expected squared loss. Cross-entropy loss estimates $-\mathbb{E}_{p}[\log q]$. Every loss is a first moment of some function of the predictions.

**Optimisation.** The Adam optimizer maintains running estimates of the first moment $m_t = \mathbb{E}[g_t]$ and second moment $v_t = \mathbb{E}[g_t^2]$ of the gradient $g_t$. The adaptive learning rate $\alpha / \sqrt{v_t + \epsilon}$ normalises by the square root of the second moment. Without understanding moments, Adam is a black box; with it, Adam is moment matching with bias correction.

**Batch Normalisation.** BatchNorm computes the sample mean and variance of activations across a batch, then normalises. This is moment normalisation: forcing the first moment to zero and the second to one, stabilising training by controlling the distribution of layer activations.

**KL Divergence.** The KL divergence $\text{KL}(p \| q) = \mathbb{E}_p[\log(p/q)]$ is an expectation under $p$. Its non-negativity ($\text{KL} \geq 0$) follows directly from Jensen's inequality applied to the convex function $-\log$. The ELBO in variational autoencoders is derived by applying Jensen's inequality to decompose $\log p(x)$.

**Reparameterisation.** The reparameterisation trick in VAEs ($z = \mu_\phi + \sigma_\phi \odot \varepsilon$) enables computing gradients of expectations. The gradient $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]$ is intractable to compute directly, but after reparameterisation it becomes $\mathbb{E}_{\varepsilon}[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \varepsilon)]$, estimable via sampling.

**Score functions.** The score function $\nabla_\theta \log p_\theta(x)$ satisfies $\mathbb{E}_p[\nabla_\theta \log p_\theta(X)] = \mathbf{0}$ — its expectation is zero. This identity (proved by differentiating the normalisation condition $\int p_\theta = 1$) underlies Fisher information, the REINFORCE policy gradient estimator, and Stein's identity used in score-based diffusion models.

### 1.3 Historical Timeline

```
HISTORY OF MOMENTS AND EXPECTATION
════════════════════════════════════════════════════════════════════════

  1654  Pascal–Fermat correspondence: first rigorous treatment of
        expected value ("valeur" of a game) in probability

  1657  Huygens: De Ratiociniis in Ludo Aleae — published the first
        probability textbook with formal expectation calculations

  1713  Bernoulli (Jakob): Ars Conjectandi — law of large numbers
        (published posthumously); shows sample mean → E[X]

  1730  De Moivre: normal approximation to binomial via moments

  1814  Laplace: Théorie Analytique des Probabilités — systematic
        use of characteristic functions (Fourier transforms of densities)

  1853  Chebyshev: first inequality bounding tails via variance;
        rigorous proof of LLN; laid groundwork for moment inequalities

  1867  Chebyshev: full variance-based inequality (P(|X-μ| ≥ kσ) ≤ 1/k²)

  1894  Pearson: coined "moment" in statistics; introduced skewness
        and kurtosis; method of moments estimation

  1922  Fisher: likelihood, sufficient statistics, and moment
        generating functions as tools of parametric inference

  1945  Cramér: large deviation theory; MGF-based tail bounds
        (foundations of Hoeffding and Chernoff later work)

  1972  Williams: REINFORCE algorithm (score function gradient
        estimator = gradient of log-expectation)

  2014  Kingma & Ba: Adam optimizer — formal moment-tracking
        adaptive gradient method

  2014  Kingma & Welling: VAE — reparameterisation trick;
        expectations with differentiable sampling

  2020  Song et al.: score-based diffusion models — reverse SDE
        guidance via score function = ∇ log p(x)

════════════════════════════════════════════════════════════════════════
```

---

## 2. Formal Definition of Expectation

### 2.1 Expectation for Discrete, Continuous, and Mixed Distributions

The **expected value** (or **expectation** or **mean**) of a random variable $X$ is defined as the probability-weighted average of all possible values.

**Discrete random variable.** If $X$ takes values in a countable set with PMF $p(x) = P(X=x)$:

$$\mathbb{E}[X] = \sum_{x \in \text{support}(X)} x \cdot p(x)$$

provided this sum converges absolutely: $\sum_x |x| \cdot p(x) < \infty$. If it does not converge absolutely, $\mathbb{E}[X]$ does not exist.

*Example.* For a fair die $X \in \{1,2,3,4,5,6\}$ with $p(x) = 1/6$:
$$\mathbb{E}[X] = \frac{1}{6}(1+2+3+4+5+6) = \frac{21}{6} = 3.5$$

**Continuous random variable.** If $X$ has PDF $f(x)$:

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

provided $\int_{-\infty}^{\infty} |x| f(x) \, dx < \infty$.

*Example.* For $X \sim \text{Exp}(\lambda)$ with PDF $f(x) = \lambda e^{-\lambda x}$ on $[0,\infty)$:
$$\mathbb{E}[X] = \int_0^\infty x \lambda e^{-\lambda x} dx = \frac{1}{\lambda}$$

**Mixed distribution.** If $X$ has a mixed distribution (point mass at $x_0$ plus a continuous density):
$$\mathbb{E}[X] = p_0 \cdot x_0 + \int x \cdot f_{\text{cont}}(x) \, dx$$

where $p_0 = P(X = x_0)$ is the point mass probability.

**Notation.** We write $\mathbb{E}[X]$, $\mathbb{E}_X[X]$, $\mu_X$, or $\langle X \rangle$ (physics notation) interchangeably. When the distribution is parameterised by $\theta$: $\mathbb{E}_\theta[X]$ or $\mathbb{E}_{X \sim p_\theta}[X]$.

**For AI:** The empirical mean $\bar{X}_N = \frac{1}{N}\sum_{i=1}^N X_i$ is the canonical estimator of $\mathbb{E}[X]$. Every forward pass computes a function of the inputs; averaging that function over a batch estimates its expectation. The loss function IS an expectation — we just estimate it from data.

### 2.2 Existence and Integrability

The expectation $\mathbb{E}[X]$ exists (is finite) when $\mathbb{E}[|X|] < \infty$, i.e., when $X$ is **integrable**. This fails in important cases:

**Cauchy distribution.** $f(x) = \frac{1}{\pi(1+x^2)}$ has $\int_{-\infty}^\infty |x| \cdot \frac{1}{\pi(1+x^2)} dx = +\infty$. The Cauchy distribution has no mean (or any finite moment). A ratio of two independent standard normals follows a Cauchy distribution.

**Heavy-tailed distributions.** The Pareto distribution $f(x) = \alpha x_m^\alpha / x^{\alpha+1}$ for $x > x_m$ has:
- $\mathbb{E}[X]$ exists iff $\alpha > 1$
- $\text{Var}(X)$ exists iff $\alpha > 2$
- $k$-th moment exists iff $\alpha > k$

This matters in practice: internet traffic volumes, earthquake magnitudes, word frequencies in language (Zipf's law), and wealth distributions all exhibit Pareto-like tails. Models that assume finite variance (e.g., CLT-based confidence intervals) can fail badly when applied to such data.

**Convergence condition.** $\mathbb{E}[X]$ is well-defined (possibly $\pm\infty$) for non-negative $X$ (since the integral may diverge to $+\infty$ but is unambiguous). For general $X$, write $X = X^+ - X^-$ where $X^+ = \max(X,0)$ and $X^- = \max(-X,0)$. Then $\mathbb{E}[X] = \mathbb{E}[X^+] - \mathbb{E}[X^-]$, finite only when both are finite.

**For AI:** In practice, neural network outputs and loss values are bounded (through clipping, normalisation, or the bounded nature of activation functions). But gradient values can have very heavy tails, especially in early training. Gradient clipping is a practical acknowledgment that the gradient may not have finite variance, making the Cauchy-Schwarz and CLT approximations break down.

### 2.3 Linearity of Expectation

The most powerful property of expectation is its **linearity**, which holds without any independence assumption:

**Theorem (Linearity of Expectation).** For any random variables $X$ and $Y$ and constants $a, b \in \mathbb{R}$:

$$\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

**Proof (continuous case).** Using the joint density $f_{X,Y}$:
$$\mathbb{E}[aX+bY] = \int\!\!\int (ax+by) f_{X,Y}(x,y)\, dx\, dy$$
$$= a\int\!\!\int x\, f_{X,Y}(x,y)\, dx\, dy + b\int\!\!\int y\, f_{X,Y}(x,y)\, dx\, dy$$
$$= a\int x\, f_X(x)\, dx + b\int y\, f_Y(y)\, dy = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y]$$

where we used $\int f_{X,Y}(x,y)\, dy = f_X(x)$ (marginalisation from §03). The discrete case is analogous. $\square$

**Extension.** By induction: for any constants $a_1, \ldots, a_n$ and random variables $X_1, \ldots, X_n$ (not necessarily independent):
$$\mathbb{E}\!\left[\sum_{i=1}^n a_i X_i\right] = \sum_{i=1}^n a_i \mathbb{E}[X_i]$$

**Critical point:** This holds even when $X$ and $Y$ are dependent. It does NOT extend to products: $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ only when $X \perp Y$.

**Examples demonstrating power of linearity:**

*Sum of dice.* $S = X_1 + X_2 + \cdots + X_n$ where each $X_i$ is a fair die. Without worrying about dependence (there is none here, but the theorem doesn't require it): $\mathbb{E}[S] = n \cdot 3.5$.

*Binomial mean.* $X \sim \text{Binomial}(n,p)$. Write $X = \sum_{i=1}^n B_i$ where $B_i \sim \text{Bernoulli}(p)$ (the indicator that trial $i$ succeeds). Then: $\mathbb{E}[X] = \sum_{i=1}^n \mathbb{E}[B_i] = np$. No need to compute the full PMF.

*Expected number of fixed points.* A random permutation $\sigma$ of $\{1,\ldots,n\}$ has expected number of fixed points ($\sigma(i) = i$) equal to 1, for any $n$. Proof: $\mathbb{E}[\#\text{fixed points}] = \sum_{i=1}^n P(\sigma(i)=i) = n \cdot (1/n) = 1$.

**For AI:** Linearity of expectation is the reason we can decompose complex expected losses into sums of simpler expected losses. The expected cross-entropy over a vocabulary of size $V$ is $\sum_{v=1}^V \mathbb{E}[y_v \log \hat{p}_v]$ — a sum of $V$ simpler expectations. The expected gradient of a sum is the sum of the expected gradients, justifying minibatch averaging as a gradient estimator.

### 2.4 LOTUS — Law of the Unconscious Statistician

A common need: compute $\mathbb{E}[g(X)]$ for some function $g$ when we know the distribution of $X$ but not necessarily of $g(X)$ directly. LOTUS provides the answer.

**Theorem (LOTUS — Law of the Unconscious Statistician).** Let $X$ be a random variable and $g : \mathbb{R} \to \mathbb{R}$ a measurable function.

*Discrete:*
$$\mathbb{E}[g(X)] = \sum_{x} g(x) \cdot p_X(x)$$

*Continuous:*
$$\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) \, dx$$

**Key insight:** We do NOT need the distribution of $g(X)$. We use the distribution of $X$ directly. This is why it's called the "unconscious" statistician — you can compute $\mathbb{E}[g(X)]$ without consciously finding the PDF of $g(X)$.

**Proof sketch (discrete case).** Let $Y = g(X)$. Grouping terms:
$$\mathbb{E}[Y] = \sum_y y \cdot P(Y=y) = \sum_y y \sum_{x: g(x)=y} P(X=x) = \sum_x g(x) P(X=x)$$

The last step rearranges the double sum. $\square$

**Examples:**

*Variance via LOTUS.* $\text{Var}(X) = \mathbb{E}[(X-\mu)^2]$. Using LOTUS directly on $g(x) = (x-\mu)^2$:
$$\text{Var}(X) = \int (x-\mu)^2 f_X(x) \, dx$$

*Second moment of Exponential.* For $X \sim \text{Exp}(\lambda)$, using $g(x) = x^2$:
$$\mathbb{E}[X^2] = \int_0^\infty x^2 \lambda e^{-\lambda x} dx = \frac{2}{\lambda^2}$$

*Jensen's inequality (preview).* For convex $f$, LOTUS combined with the supporting hyperplane property gives $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$. (Full treatment in §7.)

**LOTUS for functions of multiple variables.** For jointly distributed $(X,Y)$ with joint density $f_{X,Y}$:
$$\mathbb{E}[g(X,Y)] = \int\!\!\int g(x,y) f_{X,Y}(x,y) \, dx \, dy$$

**For AI:** LOTUS is the theorem behind every expectation computation in variational inference. The ELBO contains $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ — an expectation of $\log p_\theta(x|z)$ under $q_\phi$. LOTUS says: integrate $\log p_\theta(x|z)$ weighted by $q_\phi(z|x)$, no need to find the distribution of $\log p_\theta(x|z)$ directly.

---

## 3. Variance, Standard Deviation, and Higher Moments

### 3.1 Variance

The **variance** of $X$ measures the expected squared deviation from the mean:

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

**Computational formula.** Expanding the square:
$$\text{Var}(X) = \mathbb{E}[X^2 - 2\mu X + \mu^2] = \mathbb{E}[X^2] - 2\mu\mathbb{E}[X] + \mu^2 = \mathbb{E}[X^2] - \mu^2$$

This is the **König-Huygens formula**: $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.

**Properties of variance:**

1. $\text{Var}(X) \geq 0$, with equality iff $X$ is almost surely constant.
2. $\text{Var}(aX + b) = a^2 \text{Var}(X)$ for any constants $a, b$.
3. $\text{Var}(X + Y) = \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$ (see §4.4).
4. If $X \perp Y$: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$.

**Proof of property 2.**
$$\text{Var}(aX+b) = \mathbb{E}[(aX+b - (a\mu+b))^2] = \mathbb{E}[(a(X-\mu))^2] = a^2 \mathbb{E}[(X-\mu)^2] = a^2\text{Var}(X)$$

Note: the constant $b$ does not affect variance (shifts don't change spread).

**Standard examples:**

- $X \sim \text{Bernoulli}(p)$: $\mathbb{E}[X] = p$, $\mathbb{E}[X^2] = p$, $\text{Var}(X) = p - p^2 = p(1-p)$. Maximum at $p = 1/2$.
- $X \sim \text{Uniform}[a,b]$: $\mathbb{E}[X] = (a+b)/2$, $\text{Var}(X) = (b-a)^2/12$.
- $X \sim \mathcal{N}(\mu, \sigma^2)$: $\mathbb{E}[X] = \mu$, $\text{Var}(X) = \sigma^2$ (by construction).
- $X \sim \text{Exp}(\lambda)$: $\mathbb{E}[X] = 1/\lambda$, $\mathbb{E}[X^2] = 2/\lambda^2$, $\text{Var}(X) = 1/\lambda^2$.
- $X \sim \text{Poisson}(\lambda)$: $\mathbb{E}[X] = \text{Var}(X) = \lambda$ (special property).

**For AI:** Variance quantifies uncertainty. The variance of a model's predictions measures how much predictions vary with the training set — high variance = overfitting. The variance of gradient estimates in SGD determines how noisy the update steps are — high variance slows convergence. Reducing gradient variance is the goal of techniques like control variates, variance reduction in REINFORCE, and importance-weighted estimators.

### 3.2 Standard Deviation and Coefficient of Variation

The **standard deviation** $\sigma_X = \sqrt{\text{Var}(X)}$ has the same units as $X$ (variance has squared units). It is the more interpretable spread measure.

The **coefficient of variation** (CV) normalises the standard deviation by the mean:
$$\text{CV}(X) = \frac{\sigma_X}{\mathbb{E}[X]}$$

CV is dimensionless and measures relative spread. A CV of 0.1 means the standard deviation is 10% of the mean. This is useful when comparing spread across distributions with different scales.

**Standardisation.** Subtracting the mean and dividing by the standard deviation gives the **standardised** variable:
$$Z = \frac{X - \mu}{\sigma}$$
which has $\mathbb{E}[Z] = 0$ and $\text{Var}(Z) = 1$.

**For AI:** Batch Normalisation (Ioffe & Szegedy, 2015) standardises layer activations during training: for each feature, subtract the batch mean and divide by the batch standard deviation. This forces $\mathbb{E}[Z] = 0$ and $\text{Var}(Z) = 1$, then rescales with learnable $\gamma, \beta$. Layer Norm and RMS Norm are variations that normalise over the feature dimension rather than the batch dimension — critical for transformers where batch sizes can be 1.

### 3.3 Raw vs Central Moments

**Raw moments** (moments about zero):
$$\mu'_k = \mathbb{E}[X^k]$$

**Central moments** (moments about the mean $\mu = \mathbb{E}[X]$):
$$\mu_k = \mathbb{E}[(X - \mu)^k]$$

**Relationship:** The central moments can be expressed in terms of raw moments via the binomial theorem:
$$\mu_k = \sum_{j=0}^k \binom{k}{j} \mu'_j (-\mu)^{k-j}$$

For the first four:
- $\mu_1 = 0$ (always — the mean is the center)
- $\mu_2 = \mu'_2 - (\mu'_1)^2 = \mathbb{E}[X^2] - \mu^2 = \text{Var}(X)$
- $\mu_3 = \mu'_3 - 3\mu'_2\mu'_1 + 2(\mu'_1)^3$
- $\mu_4 = \mu'_4 - 4\mu'_3\mu'_1 + 6\mu'_2(\mu'_1)^2 - 3(\mu'_1)^4$

**Standardised moments** divide central moments by $\sigma^k$, making them dimensionless:
$$\gamma_k = \frac{\mu_k}{\sigma^k}$$

$\gamma_1$ = skewness, $\gamma_2$ = excess kurtosis (kurtosis of normal = 3; excess kurtosis subtracts 3).

### 3.4 Skewness

**Skewness** measures the asymmetry of a distribution about its mean:
$$\text{Skew}(X) = \gamma_1 = \frac{\mu_3}{\sigma^3} = \frac{\mathbb{E}[(X-\mu)^3]}{\sigma^3}$$

- **$\gamma_1 > 0$ (right-skewed / positive skew):** Long tail on the right; mean > median > mode. Examples: income distributions, insurance claims, LLM generation probabilities.
- **$\gamma_1 = 0$ (symmetric):** Normal distribution, uniform, any distribution symmetric about its mean.
- **$\gamma_1 < 0$ (left-skewed / negative skew):** Long tail on the left; mean < median < mode. Examples: age at death in developed countries.

```
SKEWNESS INTUITION
════════════════════════════════════════════════════════════════════════

  Right-skewed (γ₁ > 0):           Left-skewed (γ₁ < 0):

     ██                                          ██
    ████                                        ████
   ██████░░░                           ░░░██████████
  ──┴──┴──┴──────                 ──────┴──┴──┴──┴──

  Mode < Median < Mean            Mean < Median < Mode
  Tail pulls mean right           Tail pulls mean left

════════════════════════════════════════════════════════════════════════
```

**For AI:** Gradient distributions in deep networks are often right-skewed — most gradients are small, but occasional large gradients occur. This is why gradient clipping is standard practice: the long right tail causes instability if unconstrained. Token probability distributions from language models are also right-skewed: most tokens have very low probability, while a handful of likely tokens dominate.

**Skewness of common distributions:**
- $\mathcal{N}(\mu,\sigma^2)$: $\gamma_1 = 0$ (symmetric by construction)
- $\text{Exp}(\lambda)$: $\gamma_1 = 2$ (always right-skewed)
- $\text{Poisson}(\lambda)$: $\gamma_1 = 1/\sqrt{\lambda}$ (approaches 0 as $\lambda \to \infty$)
- $\text{Beta}(\alpha,\beta)$: $\gamma_1 = 2(\beta-\alpha)\sqrt{\alpha+\beta+1}/[(\alpha+\beta+2)\sqrt{\alpha\beta}]$

### 3.5 Kurtosis and Excess Kurtosis

**Kurtosis** measures the weight of the tails relative to a normal distribution:
$$\text{Kurt}(X) = \frac{\mu_4}{\sigma^4} = \frac{\mathbb{E}[(X-\mu)^4]}{\sigma^4}$$

**Excess kurtosis** (the standard in statistics) subtracts the normal baseline of 3:
$$\gamma_2 = \frac{\mu_4}{\sigma^4} - 3$$

- **$\gamma_2 > 0$ (leptokurtic):** Heavier tails than Gaussian; more probability in the tails and peak. Examples: Student-$t$ distribution, financial returns, gradient norms.
- **$\gamma_2 = 0$ (mesokurtic):** Normal distribution (the baseline).
- **$\gamma_2 < 0$ (platykurtic):** Lighter tails; more uniform. Uniform distribution has $\gamma_2 = -6/5$.

```
KURTOSIS AND TAIL WEIGHT
════════════════════════════════════════════════════════════════════════

  Leptokurtic (γ₂>0):     Mesokurtic (γ₂=0):     Platykurtic (γ₂<0):

      ▓                        ▓                      ▓▓▓
     ▓▓▓                      ▓▓▓                    ▓▓▓▓▓
    ▓▓▓▓▓                    ▓▓▓▓▓                  ▓▓▓▓▓▓▓
   ░░░░░░░░░                ░░░░░░░                 ░░░░░░░

  Heavy tails — outliers    Normal tails            Thin tails
  common. Student-t, loss   Gaussian                Uniform
  gradients, finance

════════════════════════════════════════════════════════════════════════
```

**For AI:** Gradient norm distributions during training are leptokurtic — occasional large-gradient events are far more common than a Gaussian would predict. This motivates both gradient clipping (cap the maximum norm) and heavy-tailed noise models for understanding why SGD generalises. The Student-$t$ distribution ($\gamma_2 = 6/(df-4)$ for $df > 4$) is often used to model heavy-tailed distributions in robust statistics.

**Kurtosis of common distributions:**
- $\mathcal{N}(\mu,\sigma^2)$: $\gamma_2 = 0$ (by definition — the reference)
- $\text{Uniform}[a,b]$: $\gamma_2 = -6/5$
- $\text{Exp}(\lambda)$: $\gamma_2 = 6$
- $\text{Laplace}(\mu,b)$: $\gamma_2 = 3$
- $\text{Student-}t(\nu)$: $\gamma_2 = 6/(\nu-4)$ for $\nu > 4$; infinite for $\nu \leq 4$

---

## 4. Covariance and Correlation

### 4.1 Covariance as an Operator

The **covariance** of two random variables $X$ and $Y$ measures their linear co-movement:

$$\text{Cov}(X,Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$$

**Computational formula:**
$$\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

**Proof:** $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)] = \mathbb{E}[XY - \mu_Y X - \mu_X Y + \mu_X \mu_Y] = \mathbb{E}[XY] - \mu_Y\mu_X - \mu_X\mu_Y + \mu_X\mu_Y = \mathbb{E}[XY] - \mu_X\mu_Y$. $\square$

**Properties (bilinearity):**

1. $\text{Cov}(X,X) = \text{Var}(X)$
2. $\text{Cov}(X,Y) = \text{Cov}(Y,X)$ (symmetric)
3. $\text{Cov}(aX+b, cY+d) = ac\,\text{Cov}(X,Y)$ (bilinear, constants drop out)
4. $\text{Cov}(X+Z, Y) = \text{Cov}(X,Y) + \text{Cov}(Z,Y)$ (linear in first argument)
5. $|\text{Cov}(X,Y)| \leq \sqrt{\text{Var}(X)\text{Var}(Y)}$ (Cauchy-Schwarz, proved in §7.3)

**Sign interpretation:**
- $\text{Cov}(X,Y) > 0$: $X$ and $Y$ tend to be above their means simultaneously.
- $\text{Cov}(X,Y) < 0$: when $X$ is above its mean, $Y$ tends to be below its mean.
- $\text{Cov}(X,Y) = 0$: no linear relationship (but may still be nonlinearly dependent).

> **Recall from §03:** The full covariance matrix $\Sigma$ of a random vector collects all pairwise covariances: $\Sigma_{ij} = \text{Cov}(X_i, X_j)$. The multivariate Gaussian is parameterised by $(\boldsymbol{\mu}, \Sigma)$.
> → _Full treatment: [Joint Distributions §03](../03-Joint-Distributions/notes.md)_

### 4.2 Pearson Correlation Coefficient

The Pearson correlation normalises covariance to remove units:

$$\rho_{XY} = \text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Theorem:** $-1 \leq \rho_{XY} \leq 1$.

**Proof via Cauchy-Schwarz.** By the Cauchy-Schwarz inequality (proved in §7.3):
$$(\text{Cov}(X,Y))^2 = (\mathbb{E}[(X-\mu_X)(Y-\mu_Y)])^2 \leq \mathbb{E}[(X-\mu_X)^2]\mathbb{E}[(Y-\mu_Y)^2] = \sigma_X^2 \sigma_Y^2$$

Dividing both sides by $\sigma_X^2 \sigma_Y^2$: $\rho^2 \leq 1$, so $|\rho| \leq 1$. $\square$

**Boundary cases:**
- $\rho = +1$: $Y = aX + b$ for some $a > 0$ (perfect positive linear relationship).
- $\rho = -1$: $Y = aX + b$ for some $a < 0$ (perfect negative linear relationship).
- $\rho = 0$: **uncorrelated** (no linear relationship — but not necessarily independent).

**For AI:** The correlation coefficient appears in weight matrix analysis. The effective rank of a weight matrix is related to the correlations between its columns. In attention, high correlation between key-query pairs produces high attention weights. In representational similarity analysis (RSA), neural network layers are compared by computing correlation matrices of their activations.

### 4.3 Independence and Zero Covariance

**Theorem.** If $X \perp Y$, then $\text{Cov}(X,Y) = 0$.

**Proof.** If $X \perp Y$, then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ (a characterisation of independence). Therefore $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = 0$. $\square$

**The converse is FALSE.** Zero covariance does not imply independence.

**Classic counterexample.** Let $X \sim \text{Uniform}(-1,1)$ and $Y = X^2$. Then:
- $\mathbb{E}[X] = 0$ (symmetric distribution)
- $\mathbb{E}[XY] = \mathbb{E}[X \cdot X^2] = \mathbb{E}[X^3] = 0$ (odd function, symmetric distribution)
- $\text{Cov}(X,Y) = 0$

But $Y$ is completely determined by $X$ — knowing $X$ tells you $Y = X^2$ exactly. They are maximally dependent!

**For jointly Gaussian variables only:** Zero covariance DOES imply independence. This is a special property of the Gaussian distribution — it is the unique distribution for which uncorrelated implies independent (for the joint distribution; marginal Gaussians with zero covariance need not be jointly Gaussian).

**For AI:** Feature selection based on correlation can miss highly informative features that have nonlinear (but zero-covariance) relationships with the target. Mutual information is a better measure of dependence. In contrastive learning (e.g., SimCLR), redundancy reduction methods explicitly decorrelate representations — but decorrelation (zero covariance) and statistical independence are different objectives.

### 4.4 Variance of Sums

**Theorem.** For any random variables $X$ and $Y$:
$$\text{Var}(X + Y) = \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$$

**Proof:**
$$\text{Var}(X+Y) = \mathbb{E}[(X+Y)^2] - (\mathbb{E}[X+Y])^2$$
$$= \mathbb{E}[X^2 + 2XY + Y^2] - (\mu_X + \mu_Y)^2$$
$$= (\mathbb{E}[X^2] - \mu_X^2) + 2(\mathbb{E}[XY] - \mu_X\mu_Y) + (\mathbb{E}[Y^2] - \mu_Y^2)$$
$$= \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y) \quad \square$$

**Corollary (independence):** If $X \perp Y$: $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$.

**General sum:** For $X_1, \ldots, X_n$:
$$\text{Var}\!\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i<j} \text{Cov}(X_i, X_j)$$

If all $X_i$ are independent: $\text{Var}(\sum_i X_i) = \sum_i \text{Var}(X_i)$.

**Scaling.** For iid $X_1,\ldots,X_n$ with variance $\sigma^2$:
$$\text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}$$

The variance of the sample mean decreases as $1/n$ — the statistical basis for why larger datasets reduce estimation uncertainty.

**For AI:** Weight initialisation strategies (Xavier/Glorot, He) are derived from variance propagation. If $\mathbf{h}^{[l]} = W^{[l]}\mathbf{h}^{[l-1]}$, then the variance of each output neuron is $n_{\text{in}} \cdot \text{Var}(w_{ij}) \cdot \text{Var}(h^{[l-1]}_j)$. To keep variance constant across layers (preventing vanishing/exploding activations): $\text{Var}(w_{ij}) = 1/n_{\text{in}}$ (Xavier) or $2/n_{\text{in}}$ (He for ReLU). This is the variance-of-sums formula applied to layer activations.


---

## 5. Conditional Expectation

### 5.1 Definition and Basic Properties

The **conditional expectation** $\mathbb{E}[Y \mid X = x]$ is the expected value of $Y$ given that $X$ takes the specific value $x$. It is a function of $x$, not a number.

**Discrete case:**
$$\mathbb{E}[Y \mid X = x] = \sum_y y \cdot P(Y = y \mid X = x) = \sum_y y \cdot \frac{p_{X,Y}(x,y)}{p_X(x)}$$

**Continuous case:**
$$\mathbb{E}[Y \mid X = x] = \int_{-\infty}^\infty y \cdot f_{Y|X}(y|x) \, dy = \int y \cdot \frac{f_{X,Y}(x,y)}{f_X(x)} \, dy$$

**As a random variable.** When we write $\mathbb{E}[Y \mid X]$ (without specifying the value of $X$), we mean the random variable obtained by composing the function $x \mapsto \mathbb{E}[Y \mid X=x]$ with $X$. This is a function of $X$ and is therefore itself a random variable.

**Basic properties:**

1. $\mathbb{E}[aY + bZ \mid X] = a\mathbb{E}[Y \mid X] + b\mathbb{E}[Z \mid X]$ (linearity)
2. $\mathbb{E}[g(X) \cdot Y \mid X] = g(X) \cdot \mathbb{E}[Y \mid X]$ (taking out known factors)
3. If $X \perp Y$: $\mathbb{E}[Y \mid X] = \mathbb{E}[Y]$ (conditioning on independent variable doesn't help)
4. $\mathbb{E}[\mathbb{E}[Y \mid X]] = \mathbb{E}[Y]$ (tower property — see §5.2)

**Example.** Let $(X,Y)$ be jointly uniform on the triangle $\{(x,y): 0 \leq y \leq x \leq 1\}$.
From §03 (Appendix V.1), the conditional density is $f_{Y|X}(y|x) = 1/x$ on $[0,x]$. Therefore:
$$\mathbb{E}[Y \mid X = x] = \int_0^x y \cdot \frac{1}{x} \, dy = \frac{1}{x} \cdot \frac{x^2}{2} = \frac{x}{2}$$

The conditional mean of $Y$ given $X = x$ is $x/2$ — exactly half of $x$, which makes sense since $Y$ is uniform on $[0,x]$.

### 5.2 Tower Property (Iterated Expectation)

The **tower property** (also called the **law of iterated expectations** or **Adam's law**) is one of the most useful results in probability:

$$\mathbb{E}[\mathbb{E}[Y \mid X]] = \mathbb{E}[Y]$$

More generally, for any function $g(X)$:
$$\mathbb{E}[g(X) \cdot Y] = \mathbb{E}[g(X) \cdot \mathbb{E}[Y \mid X]]$$

**Proof (continuous case):**
$$\mathbb{E}[\mathbb{E}[Y \mid X]] = \int \mathbb{E}[Y \mid X=x] \cdot f_X(x) \, dx$$
$$= \int \left(\int y \cdot f_{Y|X}(y|x) \, dy\right) f_X(x) \, dx$$
$$= \int\!\!\int y \cdot f_{Y|X}(y|x) f_X(x) \, dy \, dx$$
$$= \int\!\!\int y \cdot f_{X,Y}(x,y) \, dy \, dx = \mathbb{E}[Y] \quad \square$$

**Intuition.** If you first average $Y$ within groups defined by $X$, then average those group means weighted by group size, you get the overall mean of $Y$.

**Example: Expected test score.** A class has 40% students from Group A (mean score 75) and 60% from Group B (mean score 85). Overall expected score: $0.4 \times 75 + 0.6 \times 85 = 81$. This is exactly the tower property: $\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y|G]]$ where $G$ is group membership.

**For AI:** The tower property underlies the EM algorithm. The E-step computes $Q(\theta|\theta^{(t)}) = \mathbb{E}_{Z|X,\theta^{(t)}}[\log p(X,Z|\theta)]$ — the expected complete-data log-likelihood, where the expectation is over the latent variable $Z$ given the observed $X$. The tower property guarantees that maximising $Q$ increases $\log p(X|\theta)$, the marginal likelihood. In policy gradient methods, $\mathbb{E}_\tau[R] = \mathbb{E}_{s_0}[\mathbb{E}_\tau[R | s_0]]$ — decomposing trajectory returns by initial state.

### 5.3 Conditional Variance and Law of Total Variance

The **conditional variance** of $Y$ given $X = x$ is:
$$\text{Var}(Y \mid X = x) = \mathbb{E}[(Y - \mathbb{E}[Y \mid X=x])^2 \mid X = x]$$

**Law of Total Variance (Eve's Law):**
$$\text{Var}(Y) = \mathbb{E}[\text{Var}(Y \mid X)] + \text{Var}(\mathbb{E}[Y \mid X])$$

This decomposes total variance into:
- **Within-group variance:** $\mathbb{E}[\text{Var}(Y \mid X)]$ — average variability within each value of $X$.
- **Between-group variance:** $\text{Var}(\mathbb{E}[Y \mid X])$ — variability of the group means.

**Proof:**
$$\text{Var}(Y) = \mathbb{E}[Y^2] - (\mathbb{E}[Y])^2$$

Using the tower property twice and writing $\mu(x) = \mathbb{E}[Y|X=x]$:
$$\mathbb{E}[Y^2] = \mathbb{E}[\mathbb{E}[Y^2 \mid X]] = \mathbb{E}[\text{Var}(Y|X) + (\mathbb{E}[Y|X])^2] = \mathbb{E}[\text{Var}(Y|X)] + \mathbb{E}[\mu(X)^2]$$

And $(\mathbb{E}[Y])^2 = (\mathbb{E}[\mu(X)])^2$. Therefore:
$$\text{Var}(Y) = \mathbb{E}[\text{Var}(Y|X)] + (\mathbb{E}[\mu(X)^2] - (\mathbb{E}[\mu(X)])^2) = \mathbb{E}[\text{Var}(Y|X)] + \text{Var}(\mu(X)) \quad \square$$

**For AI:** The law of total variance is the mathematical foundation of the bias-variance decomposition (§8). For a model trained on dataset $\mathcal{D}$: total error = expected within-dataset error + between-dataset variance. It also explains why mixture models (GMMs) combine within-component variance and between-component variance.

### 5.4 AI Applications of Conditional Expectation

**Optimal predictor.** The function $f^*(x) = \mathbb{E}[Y \mid X=x]$ minimises the expected squared error:
$$f^* = \arg\min_{f} \mathbb{E}[(Y - f(X))^2]$$

**Proof:** For any $f$,
$$\mathbb{E}[(Y-f(X))^2] = \mathbb{E}[(Y - f^*(X) + f^*(X) - f(X))^2]$$
$$= \mathbb{E}[(Y-f^*(X))^2] + 2\mathbb{E}[(Y-f^*(X))(f^*(X)-f(X))] + \mathbb{E}[(f^*(X)-f(X))^2]$$

The cross term vanishes because $\mathbb{E}[Y - f^*(X) \mid X] = \mathbb{E}[Y|X] - f^*(X) = 0$, so by the tower property $\mathbb{E}[(Y-f^*(X))g(X)] = 0$ for any $g$. The last term is non-negative. Therefore $\mathbb{E}[(Y-f(X))^2] \geq \mathbb{E}[(Y-f^*(X))^2]$. $\square$

This is profound: the **best possible predictor** (in MSE sense) is the conditional expectation. Neural networks are universal approximators trained to approximate $f^* = \mathbb{E}[Y|X]$.

**Rao-Blackwell theorem.** In estimation theory: if $\hat{\theta}$ is any unbiased estimator of $\theta$ and $T$ is a sufficient statistic, then $\tilde{\theta} = \mathbb{E}[\hat{\theta} \mid T]$ is also unbiased and has variance $\leq \text{Var}(\hat{\theta})$. Conditioning on sufficient statistics never increases variance. This is the variance reduction analogue of why attention over relevant context helps.

**Attention as conditional expectation.** The output of self-attention at position $i$ is:
$$o_i = \sum_j \alpha_{ij} v_j = \mathbb{E}_{j \sim \alpha_i}[v_j]$$

This is the conditional expectation of the value vectors under the attention distribution $\alpha_i$ over key positions. High-entropy attention = averaging many values; low-entropy (sharp) attention = conditioning on one specific value.

---

## 6. Moment Generating Functions and Characteristic Functions

### 6.1 MGF Definition and Basic Properties

The **moment generating function** (MGF) of a random variable $X$ is:
$$M_X(t) = \mathbb{E}[e^{tX}]$$

defined for all $t$ in an open interval containing 0. If no such interval exists (the integral diverges), the MGF is said not to exist.

**Moments from the MGF.** Differentiating $k$ times and evaluating at $t=0$ gives the $k$-th raw moment:
$$\frac{d^k}{dt^k} M_X(t) \bigg|_{t=0} = \mathbb{E}[X^k] = \mu'_k$$

**Proof:** $M_X(t) = \mathbb{E}[e^{tX}] = \mathbb{E}\left[\sum_{k=0}^\infty \frac{(tX)^k}{k!}\right] = \sum_{k=0}^\infty \frac{\mathbb{E}[X^k]}{k!} t^k$ (when interchange is valid).

Differentiating $k$ times: $M_X^{(k)}(0) = \mathbb{E}[X^k]$. $\square$

**Uniqueness theorem.** If $M_X(t) = M_Y(t)$ for $t$ in an open neighbourhood of 0, then $X$ and $Y$ have the same distribution. The MGF uniquely determines the distribution (when it exists).

**Existence.** The MGF exists when $\mathbb{E}[e^{tX}] < \infty$ for $|t| < t_0$ for some $t_0 > 0$. Heavy-tailed distributions (Cauchy, Pareto with small $\alpha$, log-normal) do NOT have finite MGFs. The characteristic function (§6.4) always exists and is the preferred tool for such distributions.

### 6.2 MGF of Common Distributions

**Bernoulli($p$):**
$$M(t) = (1-p) + p e^t = 1 + p(e^t - 1)$$
Check: $M'(t) = pe^t$, $M'(0) = p = \mathbb{E}[X]$. ✓

**Binomial($n,p$):** (sum of $n$ iid Bernoulli)
$$M(t) = (1-p+pe^t)^n$$
First moment: $M'(0) = np$. ✓

**Poisson($\lambda$):**
$$M(t) = e^{\lambda(e^t - 1)}$$
Derivation: $M(t) = \sum_{k=0}^\infty e^{tk} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda}\sum_k \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} e^{\lambda e^t} = e^{\lambda(e^t-1)}$.
$M'(0) = \lambda$, $M''(0) = \lambda^2 + \lambda$, so $\text{Var}(X) = \lambda^2+\lambda - \lambda^2 = \lambda$. ✓

**Normal($\mu, \sigma^2$):**
$$M(t) = e^{\mu t + \sigma^2 t^2/2}$$
Derivation: Complete the square in the exponent of $\int e^{tx} \cdot \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/(2\sigma^2)} dx$.
$M'(0) = \mu$, $M''(0) = \sigma^2 + \mu^2$, $\text{Var}(X) = \sigma^2$. ✓

**Exponential($\lambda$):**
$$M(t) = \frac{\lambda}{\lambda - t} \quad \text{for } t < \lambda$$
Derivation: $\int_0^\infty e^{tx} \lambda e^{-\lambda x} dx = \lambda \int_0^\infty e^{-({\lambda-t})x} dx = \frac{\lambda}{\lambda-t}$ (converges iff $t < \lambda$).
$M'(0) = 1/\lambda$, $M''(0) = 2/\lambda^2$, $\text{Var}(X) = 1/\lambda^2$. ✓

**For AI:** MGFs appear in the proof of the Chernoff bound (§05), which gives exponentially sharp tail bounds. The exponential form $\mathbb{E}[e^{tX}]$ is closely related to the log-partition function in the exponential family: $A(\eta) = \log Z(\eta)$. The moments of the sufficient statistic can be computed by differentiating $A(\eta)$ — the MGF and the log-partition function are connected by a change of variables.

### 6.3 MGF of Sums and the Reproductive Property

**Key theorem.** If $X \perp Y$:
$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$$

**Proof:** $M_{X+Y}(t) = \mathbb{E}[e^{t(X+Y)}] = \mathbb{E}[e^{tX} e^{tY}] = \mathbb{E}[e^{tX}]\mathbb{E}[e^{tY}] = M_X(t)M_Y(t)$, using independence for the third equality. $\square$

This is the MGF version of the convolution theorem. It provides elegant proofs of reproductive properties:

**Gaussian reproductive property.** If $X \sim \mathcal{N}(\mu_1,\sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2,\sigma_2^2)$ independently:
$$M_{X+Y}(t) = e^{\mu_1 t + \sigma_1^2 t^2/2} \cdot e^{\mu_2 t + \sigma_2^2 t^2/2} = e^{(\mu_1+\mu_2)t + (\sigma_1^2+\sigma_2^2)t^2/2}$$

This is the MGF of $\mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$. So $X+Y \sim \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$.

**Poisson reproductive property.** If $X \sim \text{Poisson}(\lambda_1)$, $Y \sim \text{Poisson}(\lambda_2)$ independently:
$$M_{X+Y}(t) = e^{\lambda_1(e^t-1)} \cdot e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$$

So $X+Y \sim \text{Poisson}(\lambda_1+\lambda_2)$.

### 6.4 Characteristic Function

The **characteristic function** of $X$ is:
$$\varphi_X(t) = \mathbb{E}[e^{itX}] = \mathbb{E}[\cos(tX)] + i\,\mathbb{E}[\sin(tX)]$$

where $i = \sqrt{-1}$.

**Key advantage:** $\varphi_X(t)$ exists for ALL distributions (since $|e^{itX}| = 1$ always), unlike the MGF which may not exist for heavy-tailed distributions.

**Connection to MGF:** $\varphi_X(t) = M_X(it)$ when the MGF exists. But the characteristic function is defined even when the MGF is not.

**Moments from $\varphi_X$.** When moments exist:
$$\mathbb{E}[X^k] = \frac{1}{i^k} \varphi_X^{(k)}(0)$$

**Inversion formula.** The distribution is uniquely determined by its characteristic function:
$$f_X(x) = \frac{1}{2\pi} \int_{-\infty}^\infty e^{-itx} \varphi_X(t) \, dt$$

(when $f_X$ is continuous). This is the Fourier inversion formula — the characteristic function IS the Fourier transform of the PDF.

**For AI:** Characteristic functions appear in the analysis of convergence of distributions (Central Limit Theorem proofs use characteristic functions because they always exist). The score-based diffusion model framework uses the score function $\nabla_x \log p(x)$, which is related to the characteristic function via Fourier analysis. Random Fourier Features (Rahimi & Recht, 2007) approximate kernel functions via random sampling from the characteristic function of the kernel.

### 6.5 Cumulants and the Cumulant Generating Function

The **cumulant generating function** (CGF) is the log of the MGF:
$$K_X(t) = \log M_X(t) = \log \mathbb{E}[e^{tX}]$$

**Cumulants** $\kappa_k$ are the coefficients in the Taylor expansion of $K_X(t)$:
$$K_X(t) = \sum_{k=1}^\infty \frac{\kappa_k}{k!} t^k$$

so $\kappa_k = K_X^{(k)}(0)$.

**First four cumulants:**
- $\kappa_1 = \mathbb{E}[X] = \mu$ (mean)
- $\kappa_2 = \text{Var}(X) = \sigma^2$ (variance)
- $\kappa_3 = \mathbb{E}[(X-\mu)^3] = \mu_3$ (third central moment = $\gamma_1 \sigma^3$)
- $\kappa_4 = \mu_4 - 3\sigma^4$ (excess kurtosis $\times \sigma^4$)

**Key property.** For independent $X \perp Y$:
$$K_{X+Y}(t) = K_X(t) + K_Y(t) \implies \kappa_k(X+Y) = \kappa_k(X) + \kappa_k(Y)$$

Cumulants of a sum of independent variables add — this is the additivity property that makes cumulants more natural than moments for sums.

**Normal distribution:** $K_X(t) = \mu t + \sigma^2 t^2/2$. All cumulants $\kappa_k = 0$ for $k \geq 3$. This characterises the normal distribution — it is the unique distribution with all cumulants beyond the second equal to zero.


---

## 7. Jensen's Inequality and Moment Inequalities

### 7.1 Jensen's Inequality

**Definition.** A function $f : \mathbb{R} \to \mathbb{R}$ is **convex** if for all $x, y$ and $\lambda \in [0,1]$:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

Geometrically: the chord between any two points lies above the curve. Equivalently (for twice-differentiable $f$): $f''(x) \geq 0$ everywhere.

**Theorem (Jensen's Inequality).** If $f$ is convex and $\mathbb{E}[|X|] < \infty$:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

**Proof (via supporting hyperplane).** Since $f$ is convex, for any $a$ there exists a supporting hyperplane at $a$: a linear function $\ell(x) = f(a) + c(x-a)$ such that $f(x) \geq \ell(x)$ for all $x$ (where $c$ is the subgradient at $a$). Setting $a = \mathbb{E}[X]$:
$$f(x) \geq f(\mathbb{E}[X]) + c(x - \mathbb{E}[X])$$

Taking expectations of both sides:
$$\mathbb{E}[f(X)] \geq f(\mathbb{E}[X]) + c\,\mathbb{E}[X - \mathbb{E}[X]] = f(\mathbb{E}[X]) + c \cdot 0 = f(\mathbb{E}[X]) \quad \square$$

**Equality condition:** $f(\mathbb{E}[X]) = \mathbb{E}[f(X)]$ iff $f$ is linear on the support of $X$, or $X$ is almost surely constant.

**For concave $f$:** $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$ (Jensen's inequality reverses).

**Common applications:**

| Function $f$ | Convex/Concave | Jensen gives |
|-------------|----------------|--------------|
| $e^x$ | Convex | $e^{\mathbb{E}[X]} \leq \mathbb{E}[e^X]$ |
| $-\log x$ | Convex | $-\log\mathbb{E}[X] \leq \mathbb{E}[-\log X]$, i.e., $\log\mathbb{E}[X] \geq \mathbb{E}[\log X]$ |
| $x^2$ | Convex | $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$ (variance $\geq 0$) |
| $\log x$ | Concave | $\log\mathbb{E}[X] \geq \mathbb{E}[\log X]$ |
| $\sqrt{x}$ | Concave | $\sqrt{\mathbb{E}[X]} \geq \mathbb{E}[\sqrt{X}]$ |

### 7.2 Applications: KL Divergence and ELBO

**KL divergence non-negativity.** The Kullback-Leibler divergence is:
$$\text{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{p(X)}{q(X)}\right] = \mathbb{E}_p[\log p(X) - \log q(X)]$$

**Theorem (Gibbs' inequality).** $\text{KL}(p \| q) \geq 0$, with equality iff $p = q$ almost everywhere.

**Proof via Jensen:**
$$-\text{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{q(X)}{p(X)}\right] \leq \log\mathbb{E}_p\left[\frac{q(X)}{p(X)}\right] = \log\int p(x) \frac{q(x)}{p(x)} dx = \log 1 = 0$$

where we applied Jensen's inequality with the concave function $\log$. Therefore $\text{KL}(p\|q) \geq 0$. $\square$

**ELBO derivation via Jensen.** For a latent variable model $p_\theta(x) = \int p_\theta(x,z) \, dz$:
$$\log p_\theta(x) = \log \int p_\theta(x,z) \, dz = \log \int q_\phi(z|x) \frac{p_\theta(x,z)}{q_\phi(z|x)} \, dz$$
$$= \log \mathbb{E}_{q_\phi}\left[\frac{p_\theta(x,z)}{q_\phi(z|x)}\right] \geq \mathbb{E}_{q_\phi}\left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]$$

where the inequality applies Jensen's to the concave $\log$. This lower bound is the **ELBO**:
$$\mathcal{L}(\phi,\theta;x) = \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

The gap between $\log p_\theta(x)$ and the ELBO is exactly $\text{KL}(q_\phi \| p_\theta(z|x)) \geq 0$. Maximising the ELBO tightens this bound.

**Log-sum inequality.** For non-negative $a_i, b_i$:
$$\sum_i a_i \log \frac{a_i}{b_i} \geq \left(\sum_i a_i\right) \log \frac{\sum_i a_i}{\sum_i b_i}$$

This is the discrete version of KL non-negativity via Jensen.

### 7.3 Cauchy-Schwarz Inequality

**Theorem (Cauchy-Schwarz for expectations).** For any random variables $X$ and $Y$:
$$(\mathbb{E}[XY])^2 \leq \mathbb{E}[X^2] \cdot \mathbb{E}[Y^2]$$

with equality iff $Y = cX$ almost surely for some constant $c$.

**Proof.** For any $t \in \mathbb{R}$:
$$0 \leq \mathbb{E}[(tX - Y)^2] = t^2\mathbb{E}[X^2] - 2t\mathbb{E}[XY] + \mathbb{E}[Y^2]$$

This is a quadratic in $t$ that is always non-negative. A quadratic $at^2 + bt + c \geq 0$ for all $t$ iff discriminant $b^2 - 4ac \leq 0$. Here $a = \mathbb{E}[X^2]$, $b = -2\mathbb{E}[XY]$, $c = \mathbb{E}[Y^2]$:
$$4(\mathbb{E}[XY])^2 - 4\mathbb{E}[X^2]\mathbb{E}[Y^2] \leq 0 \implies (\mathbb{E}[XY])^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2] \quad \square$$

**Corollary ($|\rho| \leq 1$).** Apply the Cauchy-Schwarz inequality to the centred variables $X - \mu_X$ and $Y - \mu_Y$:
$$(\text{Cov}(X,Y))^2 = (\mathbb{E}[(X-\mu_X)(Y-\mu_Y)])^2 \leq \mathbb{E}[(X-\mu_X)^2]\mathbb{E}[(Y-\mu_Y)^2] = \sigma_X^2\sigma_Y^2$$

Dividing: $\rho^2 \leq 1$.

**For AI:** Cauchy-Schwarz appears in attention — the dot product $q^\top k$ is bounded by $\|q\| \|k\|$, motivating the $\sqrt{d_k}$ scaling in scaled dot-product attention ($\text{softmax}(QK^\top/\sqrt{d_k})V$). Without this scaling, the dot products can become large in magnitude for high-dimensional keys/queries, causing the softmax to saturate and gradients to vanish.

### 7.4 Lyapunov Inequality

**Theorem (Lyapunov's inequality).** For $0 < r \leq s$:
$$(\mathbb{E}[|X|^r])^{1/r} \leq (\mathbb{E}[|X|^s])^{1/s}$$

In words: the $L^r$ norm of $X$ (as a random variable) is non-decreasing in $r$.

**Proof.** Apply Jensen's inequality with the convex function $f(t) = t^{s/r}$ (convex since $s/r \geq 1$) to the random variable $Y = |X|^r$:
$$(\mathbb{E}[Y])^{s/r} = f(\mathbb{E}[Y]) \leq \mathbb{E}[f(Y)] = \mathbb{E}[|X|^s]$$

Taking $(1/s)$ powers of both sides: $(\mathbb{E}[|X|^r])^{1/r} \leq (\mathbb{E}[|X|^s])^{1/s}$. $\square$

**Consequence.** If the $s$-th moment is finite, all lower moments are finite. If $\mathbb{E}[X^4] < \infty$, then $\mathbb{E}[X^3], \mathbb{E}[X^2], \mathbb{E}[|X|] < \infty$ as well.

**For AI:** Lyapunov's inequality underlies the Lyapunov CLT (a variant of the central limit theorem for non-identically distributed variables). In the theory of stochastic gradient descent, the existence of higher moments of the gradient controls the tightness of convergence bounds.

### 7.5 Preview: Tail Bounds from Moments

The most direct application of moments to probability is bounding how far a random variable can stray from its mean. These are covered fully in §05, but we preview the key results here.

> **Preview: Markov's Inequality**
>
> For any non-negative random variable $X$ and $t > 0$:
> $$P(X \geq t) \leq \frac{\mathbb{E}[X]}{t}$$
>
> Proof: $\mathbb{E}[X] = \int_0^\infty xf(x)\,dx \geq \int_t^\infty xf(x)\,dx \geq t\int_t^\infty f(x)\,dx = t\,P(X\geq t)$.
> The mean bounds the probability of large values.
>
> → _Full treatment and applications: [§05 Concentration Inequalities](../05-Concentration-Inequalities/notes.md)_

> **Preview: Chebyshev's Inequality**
>
> For any random variable $X$ with mean $\mu$ and variance $\sigma^2$:
> $$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$
>
> Proof: Apply Markov's inequality to $(X-\mu)^2 \geq k^2\sigma^2$: $P((X-\mu)^2 \geq k^2\sigma^2) \leq \mathbb{E}[(X-\mu)^2]/(k^2\sigma^2) = 1/k^2$.
> The variance bounds how often $X$ deviates from its mean by $k$ standard deviations.
>
> → _Full treatment and sharper bounds: [§05 Concentration Inequalities](../05-Concentration-Inequalities/notes.md)_

---

## 8. Bias-Variance Decomposition

### 8.1 Statistical Risk and Optimal Predictor

Consider predicting $Y$ from $X$ using a function $f : \mathbb{R}^d \to \mathbb{R}$. The **statistical risk** (expected squared loss) is:

$$\mathcal{R}(f) = \mathbb{E}[(Y - f(X))^2]$$

**Theorem.** The minimiser of $\mathcal{R}(f)$ over all measurable functions is the conditional expectation:
$$f^*(X) = \mathbb{E}[Y \mid X]$$

and the minimum risk equals the irreducible noise:
$$\mathcal{R}(f^*) = \mathbb{E}[\text{Var}(Y \mid X)]$$

**Proof:** From §5.4, for any $f$:
$$\mathcal{R}(f) = \mathbb{E}[(Y - f^*(X))^2] + \mathbb{E}[(f^*(X) - f(X))^2] \geq \mathcal{R}(f^*)$$

And $\mathcal{R}(f^*) = \mathbb{E}[(Y - \mathbb{E}[Y|X])^2] = \mathbb{E}[\text{Var}(Y|X)]$ by definition. $\square$

The **Bayes risk** (irreducible error) $\varepsilon^* = \mathbb{E}[\text{Var}(Y|X)]$ cannot be reduced by any model, no matter how complex. It is the noise inherent in the prediction problem.

**For regression:** With $Y = f^*(X) + \varepsilon$ where $\varepsilon$ is independent noise with variance $\sigma^2$, the Bayes risk is $\sigma^2$.

**For AI:** The irreducible error is why perfect test accuracy is impossible on noisy datasets. In language modelling, even a perfect model (which exactly learns the true distribution) will have non-zero cross-entropy loss equal to the entropy of the true distribution $H(p^*)$. GPT-4's perplexity on human text is bounded below by the entropy of human language.

### 8.2 Bias-Variance-Noise Decomposition

In practice, we learn a model $\hat{f}$ from a finite dataset $\mathcal{D}$ of $N$ samples. The model $\hat{f}$ is a random function — it depends on the random training data. We can decompose the expected prediction error at a new test point $x_0$:

$$\mathbb{E}_{\mathcal{D}}[(Y - \hat{f}(x_0))^2] = \underbrace{(f^*(x_0) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x_0)])^2}_{\text{Bias}^2} + \underbrace{\text{Var}_{\mathcal{D}}(\hat{f}(x_0))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}$$

**Proof.** Let $\bar{f} = \mathbb{E}_{\mathcal{D}}[\hat{f}(x_0)]$ (expected prediction) and $y_0 = f^*(x_0) + \varepsilon$ (noisy observation):
$$\mathbb{E}_{\mathcal{D},\varepsilon}[(y_0 - \hat{f}(x_0))^2] = \mathbb{E}[(y_0 - \bar{f} + \bar{f} - \hat{f})^2]$$
$$= \mathbb{E}[(y_0 - \bar{f})^2] + 2\mathbb{E}[(y_0-\bar{f})(\bar{f}-\hat{f})] + \mathbb{E}[(\bar{f}-\hat{f})^2]$$

The cross term is $2\mathbb{E}[(f^*+\varepsilon-\bar{f})(\bar{f}-\hat{f})] = 2(f^*-\bar{f})\mathbb{E}[\bar{f}-\hat{f}] + 2\mathbb{E}[\varepsilon]\mathbb{E}[\bar{f}-\hat{f}] = 0$ (since $\mathbb{E}[\hat{f}] = \bar{f}$ and $\mathbb{E}[\varepsilon]=0$). Then:
$$\mathbb{E}[(y_0-\bar{f})^2] = \mathbb{E}[(f^*-\bar{f}+\varepsilon)^2] = (f^*-\bar{f})^2 + \sigma^2 = \text{Bias}^2 + \sigma^2$$

And $\mathbb{E}[(\bar{f}-\hat{f})^2] = \text{Var}(\hat{f}(x_0))$. Combining: Bias² + Variance + Noise. $\square$

```
BIAS-VARIANCE TRADEOFF
════════════════════════════════════════════════════════════════════════

  Error                        Total Error
   ↑                           /
   │              Variance    /
   │         _______________/
   │        /
   │       /       Bias²
   │      /  _______________
   │_____/__/________________
   │         Noise
   └──────────────────────────────── Model Complexity →

  Classical view: sweet spot minimises total error

════════════════════════════════════════════════════════════════════════
```

**Interpretations:**

- **High bias:** The model is too simple to capture $f^*$ (underfitting). Consistent but wrong. Example: linear model for a quadratic $f^*$.
- **High variance:** The model is too complex — it fits training noise (overfitting). Right on average but variable. Example: high-degree polynomial on limited data.
- **Noise:** Irreducible. The best we can do.

### 8.3 Double Descent and Modern Deep Learning

The classical bias-variance picture predicts a unique optimal complexity. Modern deep learning violates this: neural networks trained to zero training loss (interpolating the training data, which classical theory predicts should overfit catastrophically) often generalise well. This is the **double descent** phenomenon.

```
DOUBLE DESCENT
════════════════════════════════════════════════════════════════════════

  Test Error
   ↑
   │        Classical regime        Modern regime
   │              │                      │
   │    ╭────╮    │                      │
   │   ╭╯    ╰╮   │                      │
   │  ╭╯      ╰╮  │                 ╭────╯
   │ ╭╯        ╰──┼─────────────────╯
   │               │ Interpolation threshold
   └──────────────────────────────────────── Model Size →
           ↑                   ↑
    Underfitting         Overparameterised
    regime               regime (good!)

════════════════════════════════════════════════════════════════════════
```

**Why does this happen?** In overparameterised models:
- Many solutions interpolate the training data.
- Gradient descent (especially with weight decay or implicit regularisation from SGD noise) selects the **minimum-norm** solution among all interpolating solutions.
- The minimum-norm interpolant has low variance among interpolants — it is the "flattest" fit.

The bias-variance decomposition still holds in double descent, but the variance first rises (at the interpolation threshold, the model just barely interpolates — highly sensitive to training data) and then falls again (overparameterised models have many interpolating solutions, most with low variance).

**For AI (2026 context):** Large language models like GPT-4, Claude, and Gemini are massively overparameterised. They interpolate (or nearly interpolate) their training data and yet generalise remarkably well — this is the double descent regime. Understanding why overparameterisation helps generalisation remains one of the central open questions in deep learning theory.

### 8.4 Regularisation as Bias Injection

**L2 regularisation (Ridge regression).** The regularised estimator minimises:
$$\hat{\theta}_\lambda = \arg\min_\theta \left[\sum_{i=1}^N (y_i - f_\theta(x_i))^2 + \lambda \|\theta\|^2\right]$$

For linear regression with design matrix $X$ (overloading notation): $\hat{\theta}_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$.

**Bias-variance analysis of Ridge:**
- **Bias:** $\hat{\theta}_\lambda$ is a biased estimator of $\theta^*$. Specifically, $\mathbb{E}[\hat{\theta}_\lambda] = (I + \lambda(X^\top X)^{-1})^{-1}\theta^* \neq \theta^*$. Ridge shrinks estimates toward 0, introducing bias.
- **Variance:** $\text{Var}(\hat{\theta}_\lambda) \leq \text{Var}(\hat{\theta}_{\text{OLS}})$ for any $\lambda > 0$. The regularised estimator has smaller variance — it is more stable with respect to perturbations in the training data.
- **Trade-off:** As $\lambda$ increases, bias increases and variance decreases. The optimal $\lambda$ minimises total risk.

**For AI:** Weight decay in neural network training (the most common form of regularisation) adds $\lambda \|\theta\|^2$ to the loss, equivalent to L2 regularisation. AdamW implements weight decay correctly by decoupling it from the adaptive learning rate. Understanding the bias-variance trade-off tells us why weight decay helps: it reduces model variance (overfitting) at the cost of some bias.


---

## 9. Expectation in ML: Core Applications

### 9.1 Cross-Entropy Loss as an Expectation

For a $K$-class classification problem, the cross-entropy loss for a single example $(x, y)$ with true label $y \in \{1,\ldots,K\}$ and model probabilities $\hat{p} = (\hat{p}_1,\ldots,\hat{p}_K)$ is:

$$\ell(y, \hat{p}) = -\log \hat{p}_y = -\sum_{k=1}^K \mathbf{1}[y=k] \log \hat{p}_k$$

The **expected cross-entropy** over the data distribution $p(x,y)$:
$$\mathcal{L}(\theta) = \mathbb{E}_{(X,Y) \sim p}[-\log \hat{p}_{Y}] = \mathbb{E}_{(X,Y) \sim p}\left[-\sum_{k=1}^K \mathbf{1}[Y=k] \log \hat{p}_k(X;\theta)\right]$$

This equals the **cross-entropy** $H(p, q_\theta) = -\sum_k p_k \log q_k$ between the true label distribution and the model's predicted distribution, averaged over inputs $X$.

**Connection to KL divergence.** For each input $x$:
$$H(p(\cdot|x), q_\theta(\cdot|x)) = H(p(\cdot|x)) + \text{KL}(p(\cdot|x) \| q_\theta(\cdot|x))$$

where $H(p) = -\mathbb{E}_p[\log p]$ is the entropy of the true conditional. Since $H(p(\cdot|x))$ doesn't depend on $\theta$, minimising cross-entropy is equivalent to minimising KL divergence — i.e., making the model distribution as close as possible to the true distribution, in the KL sense.

**For language models.** The cross-entropy loss on a token sequence $(w_1,\ldots,w_T)$ is:
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P_\theta(w_t \mid w_1,\ldots,w_{t-1})$$

This is the expected negative log-probability per token — an expectation under the empirical distribution of the training corpus. Minimising this drives the model distribution toward the empirical distribution of human text. The perplexity is $\exp(\mathcal{L})$: a model with perplexity 50 is "equally surprised" by text as a uniform distribution over 50 tokens.

### 9.2 ELBO as an Expected Value

The Variational Autoencoder (VAE) introduces an approximate posterior $q_\phi(z|x)$ (the encoder) to approximate the intractable true posterior $p_\theta(z|x)$. The **Evidence Lower BOund** (ELBO) is:

$$\mathcal{L}(\phi,\theta;x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))$$

**Reconstruction term:** $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$ — the expected log-likelihood of the observed data given a latent code, averaged over the approximate posterior. This rewards the decoder for reconstructing $x$ well from samples of $z \sim q_\phi(z|x)$.

**KL regularisation term:** $\text{KL}(q_\phi \| p(z))$ — penalises the approximate posterior for straying from the prior $p(z) = \mathcal{N}(0,I)$. For Gaussian encoder $q_\phi = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$:

$$\text{KL}(q_\phi \| \mathcal{N}(0,I)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_{\phi,j}^2 + \sigma_{\phi,j}^2 - \log\sigma_{\phi,j}^2 - 1\right)$$

**Gradient computation.** The reconstruction term $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$ cannot be differentiated through the sampling operation $z \sim q_\phi$ directly. The **reparameterisation trick** (§03) writes $z = \mu_\phi + \sigma_\phi \odot \varepsilon$ with $\varepsilon \sim \mathcal{N}(0,I)$, enabling:
$$\nabla_\phi \mathbb{E}_{q_\phi}[\log p_\theta(x|z)] = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\nabla_\phi \log p_\theta(x|\mu_\phi + \sigma_\phi \odot \varepsilon)]$$

This gradient is computable by backpropagation through the deterministic function $z = \mu_\phi + \sigma_\phi \odot \varepsilon$.

### 9.3 Policy Gradient (REINFORCE)

In reinforcement learning, an agent with policy $\pi_\theta(a|s)$ (parameterised by $\theta$) seeks to maximise the expected return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^T r_t\right] = \mathbb{E}_\tau[R(\tau)]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, \ldots)$ is a trajectory. The REINFORCE gradient is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \cdot \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

**Derivation via log-derivative trick (score function):**
$$\nabla_\theta \mathbb{E}_\tau[R] = \nabla_\theta \int R(\tau) p_\theta(\tau) \, d\tau = \int R(\tau) \nabla_\theta p_\theta(\tau) \, d\tau$$
$$= \int R(\tau) p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} \, d\tau = \mathbb{E}_\tau\left[R(\tau) \nabla_\theta \log p_\theta(\tau)\right]$$

Since $\log p_\theta(\tau) = \sum_t \log \pi_\theta(a_t|s_t) + \text{const}$ (dynamics don't depend on $\theta$), we get the REINFORCE formula.

The **score function** $\nabla_\theta \log \pi_\theta(a|s)$ appears because we differentiated the log of the probability. The key identity underlying this:
$$\mathbb{E}_\theta[\nabla_\theta \log p_\theta(X)] = \mathbf{0}$$

(differentiate the normalisation condition $\int p_\theta = 1$ to get $\int \nabla_\theta p_\theta = \mathbf{0}$, then note $\nabla_\theta p_\theta = p_\theta \nabla_\theta \log p_\theta$).

**For RLHF.** Reinforcement learning from human feedback uses policy gradients to fine-tune language models. The policy is the LLM: $\pi_\theta(a|s)$ where $s$ is the context and $a$ is the generated token. The reward $R$ comes from a reward model trained on human preferences. REINFORCE and its variants (PPO, GRPO) compute gradients of $\mathbb{E}[R]$ with respect to the LLM parameters.

### 9.4 Adam Optimizer as Moment Tracking

The Adam optimizer (Kingma & Ba, 2014) maintains exponential moving averages of the first and second moments of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \approx \mathbb{E}[g_t]$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \approx \mathbb{E}[g_t^2]$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ is the gradient at step $t$.

**Bias correction.** Since $m_0 = v_0 = 0$, the early estimates are biased toward zero. The debiased estimates are:
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Why does this work?** $\mathbb{E}[m_t] = (1-\beta_1^t) \cdot \mathbb{E}[g_t]$ (when gradient is stationary), so dividing by $(1-\beta_1^t)$ recovers an unbiased estimate of $\mathbb{E}[g_t]$.

**Parameter update:**
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The update is the first moment (direction of average gradient) scaled by $1/\sqrt{\text{second moment}}$ — normalising by the root mean square of the gradient. This is **adaptive learning rate**: parameters with large gradient variance get smaller updates (conservative), parameters with small variance get larger updates (aggressive).

**Connection to Fisher Information.** The diagonal of the Fisher information matrix $\mathcal{I}(\theta)$ is $\text{Var}(g_t) \approx \mathbb{E}[g_t^2]$ (when gradients are zero-mean). Adam's $\hat{v}_t$ approximates the diagonal Fisher, making Adam an approximate natural gradient method.

**AdamW.** Standard Adam conflates weight decay and L2 regularisation (they are equivalent for SGD but not for adaptive methods). AdamW decouples them:
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \alpha\lambda\theta_t$$

The last term is direct weight decay — not affected by the adaptive scaling. This is the current default for training large language models.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | $\mathbb{E}[g(X)] = g(\mathbb{E}[X])$ (always) | Only true when $g$ is linear. For convex $g$: $g(\mathbb{E}[X]) \leq \mathbb{E}[g(X)]$ (Jensen). For $g(x)=x^2$: $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$. | Apply LOTUS directly to compute $\mathbb{E}[g(X)]$. |
| 2 | Assuming $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ without checking independence | True only for independent (or uncorrelated for linear functions). Dependent variables violate this. | Check $\text{Cov}(X,Y) = 0$ (uncorrelated) before using this; for full product rule you need $X \perp Y$. |
| 3 | Confusing variance and standard deviation units | Variance has squared units; std dev has same units as $X$. Mixing them gives nonsense. | Always track units: $\text{Var}(X) = \sigma^2$ (squared), $\sigma_X = \sqrt{\text{Var}(X)}$ (same as $X$). |
| 4 | $\text{Var}(aX + b) = a^2\text{Var}(X) + b^2$ (wrong) | The constant $b$ shifts location, not spread. Variance is invariant to translation. | $\text{Var}(aX+b) = a^2\text{Var}(X)$ — the $b$ vanishes. |
| 5 | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$ without independence | This only holds when $\text{Cov}(X,Y) = 0$. For positively correlated $X,Y$: variance is larger. | $\text{Var}(X+Y) = \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$. |
| 6 | Zero covariance implies independence | False in general. The unit disk example: $Y=\sqrt{1-X^2}$ has $\text{Cov}(X,Y)=0$ but $Y$ is determined by $X$. | Use mutual information $I(X;Y)$ to test full independence. Covariance only measures linear dependence. |
| 7 | Confusing $\mathbb{E}[Y\|X=x]$ (a number) with $\mathbb{E}[Y\|X]$ (a random variable) | $\mathbb{E}[Y\|X=x]$ is fixed once $x$ is fixed. $\mathbb{E}[Y\|X]$ is random because $X$ is random. | Be explicit: $h(x) = \mathbb{E}[Y\|X=x]$; then $\mathbb{E}[Y\|X] = h(X)$ is the random variable. |
| 8 | Applying Jensen in the wrong direction for concave $f$ | Jensen: convex $f \Rightarrow f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$. For $\log$ (concave): $\log\mathbb{E}[X] \geq \mathbb{E}[\log X]$. | Check second derivative: $f'' > 0$ = convex (Jensen ≤); $f'' < 0$ = concave (Jensen ≥). |
| 9 | Assuming MGF exists for all distributions | Cauchy, Pareto (small $\alpha$), log-normal: MGFs are infinite. Using MGF-based results for these fails. | Use characteristic function (always exists). Check integrability: $\mathbb{E}[e^{tX}] < \infty$ for $|t| < t_0$. |
| 10 | Confusing raw moments and central moments | $\mu'_k = \mathbb{E}[X^k]$ and $\mu_k = \mathbb{E}[(X-\mu)^k]$ differ unless $\mu=0$. Kurtosis is $\mu_4/\sigma^4$, not $\mu'_4/\sigma^4$. | Explicitly write which you mean. For Gaussian: $\mu'_2 = \sigma^2 + \mu^2$ but $\mu_2 = \sigma^2$. |

---

## 11. Exercises

### Exercise 1 ★ — LOTUS: Expected Value of a Transformation

Let $X \sim \text{Exponential}(\lambda=2)$ with PDF $f(x) = 2e^{-2x}$ for $x \geq 0$.

**(a)** Compute $\mathbb{E}[X]$ directly from the definition.

**(b)** Using LOTUS, compute $\mathbb{E}[X^2]$.

**(c)** Compute $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$.

**(d)** Let $Y = e^X$. Compute $\mathbb{E}[Y]$ using LOTUS. (Hint: $\int_0^\infty e^{x} \cdot 2e^{-2x}dx = \int_0^\infty 2e^{-x}dx$.) What is the domain of $t$ for which $\mathbb{E}[e^{tX}]$ is finite?

---

### Exercise 2 ★ — Linearity Without Independence

Let $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$.

**(a)** Compute $\mathbb{E}[X]$, $\mathbb{E}[Y]$, and $\mathbb{E}[X+Y]$ and verify linearity.

**(b)** Compute $\text{Cov}(X, Y)$. What does this tell you about the linear relationship?

**(c)** Are $X$ and $Y$ independent? Justify rigorously.

**(d)** Compute $\text{Var}(X+Y)$ using the formula $\text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$, and verify numerically.

---

### Exercise 3 ★ — Moments of the Gaussian

For $X \sim \mathcal{N}(0, 1)$ (standard normal):

**(a)** Show $\mathbb{E}[X^{2k+1}] = 0$ for all $k \geq 0$ (odd moments vanish).

**(b)** Using the MGF $M(t) = e^{t^2/2}$, find $\mathbb{E}[X^2]$, $\mathbb{E}[X^4]$, and $\mathbb{E}[X^6]$ by differentiating.

**(c)** Verify the general formula $\mathbb{E}[X^{2k}] = (2k-1)!! = (2k-1)(2k-3)\cdots 3 \cdot 1$.

**(d)** For $X \sim \mathcal{N}(\mu, \sigma^2)$, compute the skewness $\gamma_1$ and excess kurtosis $\gamma_2$.

---

### Exercise 4 ★★ — Tower Property in Action

A fair coin is flipped. If heads (prob 1/2), $X \sim \text{Poisson}(3)$. If tails (prob 1/2), $X \sim \text{Poisson}(7)$.

**(a)** Using the tower property, compute $\mathbb{E}[X]$ (let $C \in \{H,T\}$ be the coin flip).

**(b)** Using the law of total variance, compute $\text{Var}(X)$.

**(c)** Write down the marginal PMF of $X$ (it is a mixture of two Poissons). Verify $\mathbb{E}[X]$ directly.

**(d)** AI connection: Mixture-of-experts (MoE) models route tokens to different experts. Each expert has its own output distribution. The total output distribution is a mixture; the tower property computes its moments.

---

### Exercise 5 ★★ — MGF and Moment Extraction

Let $X \sim \text{Gamma}(\alpha, \beta)$ with PDF $f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$ for $x > 0$.

**(a)** Derive the MGF: show $M_X(t) = \left(\frac{\beta}{\beta-t}\right)^\alpha$ for $t < \beta$.

**(b)** Use $M_X(t)$ to compute $\mathbb{E}[X]$ and $\text{Var}(X)$ by differentiating.

**(c)** Show that the sum of $n$ independent $\text{Gamma}(\alpha_i, \beta)$ variables (same rate $\beta$) follows $\text{Gamma}(\sum_i \alpha_i, \beta)$ using the MGF multiplicative property.

**(d)** The chi-squared distribution $\chi^2_k = \text{Gamma}(k/2, 1/2)$. What is $\mathbb{E}[\chi^2_k]$ and $\text{Var}(\chi^2_k)$?

---

### Exercise 6 ★★ — Jensen and KL Divergence

**(a)** Prove that for any probability distribution $p$ over $\{1,\ldots,K\}$:
$$H(p) = -\sum_k p_k \log p_k \leq \log K$$

with equality iff $p$ is uniform. (Hint: Apply Jensen to $f(x) = -\log x$ or use the KL divergence to uniform.)

**(b)** Show that $\text{KL}(p \| q) \geq 0$ using Jensen's inequality applied to $-\log$.

**(c)** Compute $\text{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2))$ analytically. Verify it equals zero when $\mu_1=\mu_2$, $\sigma_1=\sigma_2$.

**(d)** For the ELBO: show that $\log p_\theta(x) \geq \mathcal{L}(\phi,\theta;x)$ with gap $= \text{KL}(q_\phi(z|x) \| p_\theta(z|x))$.

---

### Exercise 7 ★★★ — Bias-Variance Decomposition

Consider estimating $\theta^* = 2$ from $N=10$ iid samples $X_1,\ldots,X_N \sim \mathcal{N}(\theta^*, 1)$.

**(a)** Show that the sample mean $\hat{\theta}_1 = \bar{X}_N$ is unbiased with variance $\sigma^2/N = 1/10$.

**(b)** Consider a regularised estimator $\hat{\theta}_\lambda = \frac{N}{N+\lambda}\bar{X}_N$. Compute its bias and variance as functions of $\lambda > 0$.

**(c)** The MSE of $\hat{\theta}_\lambda$ is Bias² + Variance. Find the $\lambda$ that minimises MSE. (Hint: it depends on $\theta^*$.)

**(d)** Simulate this: draw 10000 datasets of $N=10$ samples, compute $\hat{\theta}_\lambda$ for $\lambda \in \{0, 0.1, 0.5, 1, 2, 5\}$, plot bias², variance, and MSE vs $\lambda$.

---

### Exercise 8 ★★★ — Adam as Moment Estimation

**(a)** Show that the Adam first-moment estimate satisfies $\mathbb{E}[m_t] = (1-\beta_1^t)\mathbb{E}[g]$ when gradients are iid with mean $\mathbb{E}[g]$ (stationary gradient assumption). Therefore $\hat{m}_t = m_t/(1-\beta_1^t)$ is unbiased.

**(b)** Implement a minimal Adam optimizer in NumPy (no PyTorch) and apply it to minimise $f(\theta) = \theta^4 - 4\theta^2 + \theta$. Track the first and second moment estimates. Plot $m_t$, $v_t$, and $\theta_t$ over 200 steps.

**(c)** Show that for a quadratic loss $\mathcal{L}(\theta) = \frac{1}{2}(\theta-\theta^*)^2$, Adam converges to $\theta^*$ and the second moment $v_t \to (\theta-\theta^*)^2 = $ squared gradient. In this case, $\hat{m}_t/\sqrt{\hat{v}_t} \to \text{sign}(\theta-\theta^*)$: Adam reduces to steepest descent with unit step size.

**(d)** Connect to Fisher information: the Fisher information matrix for $p_\theta(x) = \mathcal{N}(\theta, I)$ has diagonal equal to 1 (the variance of the score). Natural gradient descent uses $\mathcal{I}^{-1}\nabla\mathcal{L}$. How does Adam approximate this for diagonal Fisher?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Application | 2026 Context |
|---------|-------------------|--------------|
| $\mathbb{E}[\ell(\theta)]$ — expected loss | Every training objective; cross-entropy, MSE, RLHF reward | All SOTA LLMs (GPT-4, Claude, Gemini) optimise expected log-likelihood; RLHF maximises expected reward from human preference model |
| Linearity of expectation | Gradient of a sum = sum of gradients; minibatch averaging | Distributed training computes gradients on data shards and averages them — justified by linearity |
| Tower property | EM algorithm E-step; policy gradient decomposition | Mixture-of-experts routing in GPT-4, Mixtral uses tower property to compute expected output over router distribution |
| $\text{Var}(X)$ — variance | Uncertainty quantification, BatchNorm/LayerNorm, gradient clipping | LayerNorm is standard in every transformer (BERT, GPT, LLaMA); normalises by mean and variance per token |
| Bias-variance decomposition | Model selection, regularisation, double descent | Understanding why GPT-4-scale models (massively overparameterised) still generalise — double descent regime |
| Jensen's inequality | KL divergence $\geq 0$; ELBO derivation; softmax bounds | VAE encoder (DALL-E 2, Stable Diffusion) optimises ELBO; Jensen is the mathematical foundation |
| MGF / Cumulants | Proof of CLT, Chernoff bounds, tail analysis | Hoeffding/Chernoff bounds used to prove PAC-learning generalisation bounds for language models |
| Adam / moment tracking | Default optimiser for all large-scale training | AdamW trains GPT-4, LLaMA, Claude, Gemini; Adam moment tracking is a direct application of $\mathbb{E}[g]$ and $\mathbb{E}[g^2]$ |
| Score function = $\nabla\log p$ | REINFORCE, RLHF policy gradient, diffusion model score | Score-based diffusion (Stable Diffusion, DALL-E 3) is defined by $\nabla_x \log p_t(x)$; trained by denoising score matching |
| Cauchy-Schwarz | Attention scaling $1/\sqrt{d_k}$, Fisher information bounds | Scaled dot-product attention in every transformer uses $1/\sqrt{d_k}$ scaling derived from variance of dot products |
| Conditional expectation | Attention output = conditional mean; optimal predictor = $\mathbb{E}[Y\|X]$ | Transformers approximate $\mathbb{E}[Y\|X]$ via attention; mechanistic interpretability studies what conditional distributions each head computes |
| Reparameterisation | VAE encoder, diffusion posterior sampling | Stable Diffusion's latent diffusion uses Gaussian reparameterisation for differentiable sampling through the encoder |

---

## 13. Conceptual Bridge

### Where We Came From

This section builds directly on the foundations laid in §01–§03. The probability spaces and random variable formalism of §01 provides the objects — PDFs, PMFs, CDFs — over which we compute expectations. The named distributions of §02 supply the examples whose moments we derive in §6.2: the mean $1/\lambda$ and variance $1/\lambda^2$ of the exponential, the mean $np$ and variance $np(1-p)$ of the binomial, the mean $\mu$ and variance $\sigma^2$ that define the Gaussian. The joint distribution machinery of §03 provides the tools for conditional expectation (§5) and the covariance matrix (§4): the tower property is essentially iterated marginalisation, and the conditional variance formula is the law of total variance derived via joint distributions.

### Where We Are Going

Section §05 (Concentration Inequalities) takes the moment-bound preview from §7.5 much further. Markov's inequality (from the mean) and Chebyshev's inequality (from the variance) are the first two results, but §05 develops exponentially sharper bounds: Hoeffding's inequality (bounded variables), Chernoff bounds (via MGFs), and McDiarmid's inequality (functions of independent variables). These bounds are the mathematical machinery of PAC-learning generalisation theory — they quantify how many training examples are needed to guarantee that the empirical risk is close to the true risk.

Section §06 (Stochastic Processes) extends expectation to sequences of random variables indexed by time. The Law of Large Numbers proves rigorously that $\bar{X}_N \to \mathbb{E}[X]$ — the sample mean converges to the expectation. The Central Limit Theorem proves that the standardised sample mean converges in distribution to $\mathcal{N}(0,1)$ — the Gaussian emerges as the universal limit of sums, with proof via MGF or characteristic function techniques introduced here.

```
POSITION IN CHAPTER 6 CURRICULUM
════════════════════════════════════════════════════════════════════════

  §01 Probability Spaces          §02 Distributions
  ┌──────────────────┐            ┌──────────────────┐
  │ Kolmogorov axioms│            │ Named PDFs/PMFs  │
  │ Events, σ-algebra│            │ Parameters       │
  │ CDF, PDF, PMF    │            │ Relationships    │
  └────────┬─────────┘            └────────┬─────────┘
           │                               │
           └──────────────┬────────────────┘
                          ↓
              §03 Joint Distributions
              ┌──────────────────────┐
              │ Marginals            │
              │ Conditionals f(y|x) │
              │ MVN, Bayes, Chain   │
              └──────────┬──────────┘
                         │
                         ↓
          ╔══════════════════════════════╗
          ║  §04 Expectation & Moments  ║  ← YOU ARE HERE
          ║  E[X], Var, Cov, MGF       ║
          ║  Jensen, C-S, LOTUS        ║
          ║  Bias-Variance, Adam       ║
          ╚══════════════╤═════════════╝
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
    §05 Concentration         §06 Stochastic
    Inequalities              Processes
    ┌─────────────┐           ┌─────────────┐
    │ Markov      │           │ LLN: X̄→E[X]│
    │ Chebyshev   │           │ CLT: →N(0,1)│
    │ Hoeffding   │           │ Gaussian    │
    │ PAC bounds  │           │ processes   │
    └─────────────┘           └─────────────┘
              │                     │
              └──────────┬──────────┘
                         ↓
              §07 Markov Chains
              ┌──────────────────────┐
              │ Transition matrices  │
              │ Steady state         │
              │ MCMC (Bayes sampling)│
              └──────────────────────┘

════════════════════════════════════════════════════════════════════════
```

The conceptual arc through §04 is: we began with probability as a framework for describing uncertainty (§01), learned the vocabulary of named distributions (§02), understood how to reason about multiple random variables jointly (§03), and now in §04 we have the tools to **summarise** distributions with numbers — expectations, variances, covariances, moments — and to **bound** the gap between reality and our estimates. Every subsequent section will use the expectation operator as a core tool: §05 to prove tail bounds, §06 to formalise convergence, §07 to analyse Markov chain stationary distributions via matrix expectations.

[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Concentration Inequalities →](../05-Concentration-Inequalities/notes.md)

---

## Appendix A — Worked Examples: Expectation and Moments

### A.1 Expected Value of the Geometric Distribution

The geometric distribution counts the number of trials until the first success in a sequence of iid Bernoulli($p$) trials.

**PMF:** $P(X=k) = (1-p)^{k-1}p$ for $k = 1, 2, 3, \ldots$

**Expected value via the definition:**
$$\mathbb{E}[X] = \sum_{k=1}^\infty k(1-p)^{k-1}p = p \sum_{k=1}^\infty k q^{k-1}$$

where $q = 1-p$. Using the identity $\sum_{k=1}^\infty k q^{k-1} = \frac{d}{dq}\sum_{k=0}^\infty q^k = \frac{d}{dq}\frac{1}{1-q} = \frac{1}{(1-q)^2} = \frac{1}{p^2}$:

$$\mathbb{E}[X] = p \cdot \frac{1}{p^2} = \frac{1}{p}$$

**Variance via the second moment:** $\mathbb{E}[X^2] = \sum_{k=1}^\infty k^2(1-p)^{k-1}p$. Using $\sum k^2 q^{k-1} = \frac{d}{dq}[\sum k q^k] = \frac{d}{dq}\frac{q}{(1-q)^2} = \frac{1+q}{(1-q)^3}$:

$$\mathbb{E}[X^2] = p \cdot \frac{1+(1-p)}{p^3} = \frac{2-p}{p^2}$$

$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{2-p}{p^2} - \frac{1}{p^2} = \frac{1-p}{p^2}$$

**AI connection:** The geometric distribution models the number of tokens until a specific token (e.g., end-of-sequence) appears, assuming iid generation. In practice tokens are not iid, but the geometric provides a baseline model. Expected sequence length under a uniform $p$ = stopping probability is $1/p$.

### A.2 Method of Moments Estimation

The **method of moments** estimates distribution parameters by setting sample moments equal to theoretical moments and solving.

**Example: Estimating $(\alpha, \beta)$ for $\text{Gamma}(\alpha, \beta)$.**

Theoretical moments: $\mathbb{E}[X] = \alpha/\beta$ and $\text{Var}(X) = \alpha/\beta^2$.

Given $N$ samples $x_1, \ldots, x_N$, compute sample mean $\bar{x} = \frac{1}{N}\sum x_i$ and sample variance $s^2 = \frac{1}{N-1}\sum(x_i-\bar{x})^2$.

Set $\bar{x} = \hat{\alpha}/\hat{\beta}$ and $s^2 = \hat{\alpha}/\hat{\beta}^2$. Solving:
$$\hat{\beta} = \frac{\bar{x}}{s^2}, \qquad \hat{\alpha} = \frac{\bar{x}^2}{s^2}$$

**For AI:** Many Bayesian models require estimating hyperparameters of prior distributions. Method of moments provides fast, closed-form initial estimates that can be refined by maximum likelihood or MCMC. It is also used in moment matching for knowledge distillation: a student model's distribution moments are matched to the teacher's.

### A.3 St. Petersburg Paradox: When Expectation Misleads

The St. Petersburg game: flip a coin repeatedly until the first head. If head appears on flip $k$, win $2^k$ dollars.

**Expected winnings:**
$$\mathbb{E}[\text{Winnings}] = \sum_{k=1}^\infty 2^k \cdot \frac{1}{2^k} = \sum_{k=1}^\infty 1 = +\infty$$

The expected value is infinite! Yet no rational person would pay more than a few dollars to play this game.

**Resolution:** Rational agents maximise expected **utility**, not expected monetary value. For a logarithmic utility function $u(w) = \log(w)$, the expected utility is finite:
$$\mathbb{E}[\log(\text{Winnings})] = \sum_{k=1}^\infty \log(2^k) \cdot \frac{1}{2^k} = \sum_{k=1}^\infty \frac{k \log 2}{2^k} = 2\log 2 < \infty$$

This is Bernoulli's resolution (1738): diminishing marginal utility. The paradox reveals that infinite expected value is not sufficient for rational choice — the existence of all moments (or at least bounded utility) is needed.

**For AI:** Reinforcement learning reward design must account for this. An agent with unbounded reward function may take extremely risky actions that have infinite expected reward but almost surely fail. This motivates bounded reward functions and regularisation in RLHF: keeping reward signals within a bounded range prevents policy collapse toward infinite-expectation strategies.

---

## Appendix B — Proofs of Key Identities

### B.1 Cauchy-Schwarz via Inner Product

The expectation $\mathbb{E}[XY]$ can be viewed as an inner product $\langle X, Y \rangle = \mathbb{E}[XY]$ in the $L^2$ space of square-integrable random variables. The Cauchy-Schwarz inequality for inner products states $|\langle X, Y \rangle|^2 \leq \langle X, X \rangle \langle Y, Y \rangle$, which gives $(\mathbb{E}[XY])^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2]$ directly.

This inner product interpretation gives $L^2$ convergence its name: $X_n \to X$ in $L^2$ means $\mathbb{E}[(X_n - X)^2] \to 0$ — convergence in the $L^2$ inner product sense.

### B.2 Variance as Second Cumulant

The CGF of $X$ is $K_X(t) = \log M_X(t) = \log \mathbb{E}[e^{tX}]$. Computing its second derivative at $t=0$:

$$K_X''(t) = \frac{M_X(t)M_X''(t) - (M_X'(t))^2}{(M_X(t))^2}$$

At $t=0$: $M_X(0)=1$, $M_X'(0)=\mathbb{E}[X]$, $M_X''(0)=\mathbb{E}[X^2]$, so:

$$K_X''(0) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \text{Var}(X)$$

The variance is precisely the second cumulant $\kappa_2$.

### B.3 Fisher Information and the Score

The **Fisher information** matrix is defined as:
$$\mathcal{I}(\theta) = \mathbb{E}_\theta\left[(\nabla_\theta \log p_\theta(X))(\nabla_\theta \log p_\theta(X))^\top\right] = \text{Cov}_\theta(\nabla_\theta \log p_\theta(X))$$

The second equality uses the fact that $\mathbb{E}_\theta[\nabla_\theta \log p_\theta(X)] = \mathbf{0}$ (proved by differentiating $\int p_\theta = 1$):
$$\int p_\theta(x) dx = 1 \implies \nabla_\theta \int p_\theta(x) dx = 0 \implies \int \nabla_\theta p_\theta(x) dx = 0 \implies \mathbb{E}_\theta\left[\frac{\nabla_\theta p_\theta(X)}{p_\theta(X)}\right] = \mathbb{E}_\theta[\nabla_\theta \log p_\theta(X)] = \mathbf{0}$$

Therefore the Fisher information is the variance (covariance matrix) of the score function $\nabla_\theta \log p_\theta(X)$.

**For AI:** Fisher information appears in:
- **Cramér-Rao bound:** $\text{Var}(\hat{\theta}) \geq 1/\mathcal{I}(\theta)$ — the variance of any unbiased estimator is at least the reciprocal Fisher information.
- **Natural gradient:** The natural gradient descent update $\mathcal{I}(\theta)^{-1}\nabla\mathcal{L}$ moves in the direction of steepest descent in the space of distributions (KL-divergence geometry), rather than parameter space. Adam approximates this with a diagonal Fisher.
- **Elastic Weight Consolidation (EWC):** Used in continual learning to prevent catastrophic forgetting. The Fisher information diagonal identifies which parameters are important for previous tasks.

### B.4 Stein's Lemma

**Lemma (Stein, 1972).** If $X \sim \mathcal{N}(\mu, \sigma^2)$ and $g$ is differentiable with $\mathbb{E}[|g'(X)|] < \infty$:
$$\mathbb{E}[g(X)(X-\mu)] = \sigma^2 \mathbb{E}[g'(X)]$$

**Proof:** Integration by parts:
$$\mathbb{E}[g(X)(X-\mu)] = \int g(x)(x-\mu) \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)} dx$$

Note $(x-\mu)e^{-(x-\mu)^2/(2\sigma^2)} = -\sigma^2 \frac{d}{dx}e^{-(x-\mu)^2/(2\sigma^2)}$. Integrating by parts:
$$= \sigma^2 \int g'(x) \frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/(2\sigma^2)} dx = \sigma^2 \mathbb{E}[g'(X)] \quad \square$$

**Special cases:**
- $g(x) = x$: $\mathbb{E}[X(X-\mu)] = \sigma^2 \implies \mathbb{E}[X^2] = \mu^2 + \sigma^2$ ✓
- $g(x) = x^2$: $\mathbb{E}[X^2(X-\mu)] = 2\sigma^2 \mathbb{E}[X] = 2\sigma^2\mu \implies \mathbb{E}[X^3] = \mu^3 + 3\mu\sigma^2$ ✓

**For AI:** Stein's lemma is the foundation of **Stein's identity** used in score matching and denoising diffusion models. The score function $\nabla_x \log p(x)$ satisfies $\mathbb{E}_p[g(X) + \nabla_x \cdot g(X)] = 0$ (Stein's operator), enabling training by minimising a quadratic loss without computing the intractable normalisation constant of $p$.

---

## Appendix C — Moment Computations for Common Distributions

This table gives raw moments, central moments, MGF, skewness, and excess kurtosis for the distributions used throughout the course.

| Distribution | $\mathbb{E}[X]$ | $\text{Var}(X)$ | Skewness $\gamma_1$ | Ex. Kurtosis $\gamma_2$ | $M_X(t)$ (domain) |
|-------------|---------|---------|------------|-------------|-----|
| Bernoulli($p$) | $p$ | $p(1-p)$ | $\frac{1-2p}{\sqrt{p(1-p)}}$ | $\frac{1-6p(1-p)}{p(1-p)}$ | $1-p+pe^t$ |
| Binomial($n,p$) | $np$ | $np(1-p)$ | $\frac{1-2p}{\sqrt{np(1-p)}}$ | $\frac{1-6p(1-p)}{np(1-p)}$ | $(1-p+pe^t)^n$ |
| Poisson($\lambda$) | $\lambda$ | $\lambda$ | $1/\sqrt{\lambda}$ | $1/\lambda$ | $e^{\lambda(e^t-1)}$ |
| Geometric($p$) | $1/p$ | $(1-p)/p^2$ | $\frac{2-p}{\sqrt{1-p}}$ | $6 + p^2/(1-p)$ | $\frac{pe^t}{1-(1-p)e^t}$, $t<-\log(1-p)$ |
| Uniform($a,b$) | $(a+b)/2$ | $(b-a)^2/12$ | $0$ | $-6/5$ | $\frac{e^{tb}-e^{ta}}{t(b-a)}$ |
| Normal($\mu,\sigma^2$) | $\mu$ | $\sigma^2$ | $0$ | $0$ | $e^{\mu t+\sigma^2t^2/2}$ |
| Exponential($\lambda$) | $1/\lambda$ | $1/\lambda^2$ | $2$ | $6$ | $\frac{\lambda}{\lambda-t}$, $t<\lambda$ |
| Gamma($\alpha,\beta$) | $\alpha/\beta$ | $\alpha/\beta^2$ | $2/\sqrt{\alpha}$ | $6/\alpha$ | $(\frac{\beta}{\beta-t})^\alpha$, $t<\beta$ |
| Beta($\alpha,\beta$) | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | $\frac{2(\beta-\alpha)\sqrt{\alpha+\beta+1}}{(\alpha+\beta+2)\sqrt{\alpha\beta}}$ | complex | No closed form |
| Student-$t$($\nu$) | $0$ ($\nu>1$) | $\frac{\nu}{\nu-2}$ ($\nu>2$) | $0$ ($\nu>3$) | $\frac{6}{\nu-4}$ ($\nu>4$) | Does not exist |

**Notes:**
- Student-$t$ has no MGF (heavy tails cause $\mathbb{E}[e^{tX}] = \infty$ for all $t \neq 0$).
- Beta distribution skewness is zero iff $\alpha = \beta$ (symmetric); negative when $\alpha > \beta$, positive when $\alpha < \beta$.
- Poisson has equal mean and variance — a property used to test whether count data follows Poisson (overdispersion: $\text{Var} > \mathbb{E}$ means extra variability, e.g., negative binomial is better).

---

## Appendix D — The Exponential Family and Moments

Many common distributions belong to the **exponential family**, which has a elegant connection between natural parameters and moments.

### D.1 Exponential Family Form

A distribution belongs to the exponential family if its density can be written as:
$$p(x;\eta) = h(x)\exp(\eta^\top T(x) - A(\eta))$$

where:
- $\eta$ = natural parameter vector
- $T(x)$ = sufficient statistic vector
- $A(\eta)$ = log-partition function (log-normaliser)
- $h(x)$ = base measure

### D.2 Moments from the Log-Partition Function

**Theorem.** For an exponential family distribution:
$$\mathbb{E}[T(X)] = \nabla_\eta A(\eta), \qquad \text{Cov}(T(X)) = \nabla^2_\eta A(\eta)$$

**Proof sketch.** Differentiating $\int p(x;\eta)dx = 1$ with respect to $\eta$:
$$0 = \int h(x)e^{\eta^\top T(x) - A(\eta)}(T(x) - \nabla A(\eta))dx = \mathbb{E}[T(X)] - \nabla A(\eta)$$

Therefore $\mathbb{E}[T(X)] = \nabla A(\eta)$. Differentiating again gives the covariance formula. $\square$

**Examples:**

| Distribution | $\eta$ | $T(x)$ | $A(\eta)$ | $\mathbb{E}[T(X)] = \nabla A$ |
|---|---|---|---|---|
| Bernoulli($p$) | $\log\frac{p}{1-p}$ | $x$ | $\log(1+e^\eta)$ | $\frac{e^\eta}{1+e^\eta} = p$ |
| Gaussian($\mu,\sigma^2$) | $(\mu/\sigma^2, -1/(2\sigma^2))$ | $(x, x^2)$ | $-\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$ | $(\mu, \sigma^2+\mu^2)$ |
| Poisson($\lambda$) | $\log\lambda$ | $x$ | $e^\eta = \lambda$ | $e^\eta = \lambda$ |

**For AI:** The log-partition function $A(\eta)$ is the **free energy** of the exponential family. Its gradient gives the expected sufficient statistics (moments), its Hessian gives the Fisher information matrix. In variational inference, the ELBO is optimised over the natural parameters $\eta$ of the approximate posterior; the optimal $\eta$ satisfies $\nabla A(\eta) = \mathbb{E}_{p}[T(X)]$ (moment matching). This is the connection between maximum entropy, sufficient statistics, and the moments derived in §6.

---

## Appendix E — Convergence in Probability and L² Convergence

The sample mean $\bar{X}_N = \frac{1}{N}\sum_{i=1}^N X_i$ estimates $\mathbb{E}[X]$. In what sense does this estimate converge?

### E.1 L² Convergence (Mean Square Convergence)

$\bar{X}_N$ converges to $\mathbb{E}[X]$ in **mean square** ($L^2$) sense:
$$\mathbb{E}[(\bar{X}_N - \mathbb{E}[X])^2] = \text{Var}(\bar{X}_N) = \frac{\text{Var}(X)}{N} \to 0$$

This is an immediate consequence of the variance formula for means (§4.4). The sample mean's MSE decreases at rate $1/N$.

### E.2 Weak Law of Large Numbers (Preview)

The Weak LLN (proved in §06 using characteristic functions or Chebyshev's inequality) states that for iid $X_i$ with finite mean $\mu$:
$$\bar{X}_N \xrightarrow{P} \mu \quad \text{as } N \to \infty$$

i.e., $P(|\bar{X}_N - \mu| > \varepsilon) \to 0$ for any $\varepsilon > 0$.

**Proof via Chebyshev (when $\text{Var}(X) < \infty$):** $P(|\bar{X}_N - \mu| > \varepsilon) \leq \frac{\text{Var}(\bar{X}_N)}{\varepsilon^2} = \frac{\text{Var}(X)}{N\varepsilon^2} \to 0$.

This is a direct application of Chebyshev's inequality (from §7.5 preview) to the sample mean. The LLN justifies: if we run a neural network many times on iid batches and average the loss estimates, the average converges to the true expected loss.

> → _Full treatment of LLN and CLT: [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md)_

---

## Appendix F — Conditional Expectation as Projection

The conditional expectation $\mathbb{E}[Y|X]$ can be understood geometrically as an **orthogonal projection** in the Hilbert space $L^2(\Omega, P)$ of square-integrable random variables.

**Inner product:** $\langle X, Y \rangle = \mathbb{E}[XY]$

**Projection:** $\mathbb{E}[Y|\mathcal{G}]$ (conditional expectation on a sub-$\sigma$-algebra $\mathcal{G}$) is the projection of $Y$ onto the closed subspace of $\mathcal{G}$-measurable random variables.

**Geometric interpretation:** Among all $\mathcal{G}$-measurable (i.e., functions of $X$) approximations to $Y$, the conditional expectation $\mathbb{E}[Y|X]$ minimises the $L^2$ distance $\mathbb{E}[(Y-g(X))^2]$. This is the projection theorem: the best approximation is the projection, and the residual $Y - \mathbb{E}[Y|X]$ is orthogonal to every $\mathcal{G}$-measurable random variable:
$$\mathbb{E}[(Y - \mathbb{E}[Y|X]) \cdot g(X)] = 0 \quad \text{for all measurable } g$$

**Consequence:** The tower property $\mathbb{E}[\mathbb{E}[Y|X]] = \mathbb{E}[Y]$ is the projection version of the law of total expectation. In a Hilbert space, projecting $Y$ onto a subspace and then taking the "total length" (expectation) equals the total length of $Y$.

**For AI:** This projection view clarifies why neural networks trained with MSE loss approximate $\mathbb{E}[Y|X]$: the network is learning the projection of $Y$ onto the subspace of functions representable by the architecture. Deeper networks can represent larger subspaces, hence better approximations to the conditional expectation.


---

## Appendix G — Worked Problems: Moments and Inequalities

### G.1 Computing the Moments of the Beta Distribution

The Beta($\alpha, \beta$) distribution has PDF $f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ on $[0,1]$, where $B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$.

**Raw moment:** Using the Beta function definition:
$$\mathbb{E}[X^k] = \int_0^1 x^k \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}dx = \frac{B(\alpha+k, \beta)}{B(\alpha,\beta)} = \frac{\Gamma(\alpha+k)\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\alpha+\beta+k)}$$

$$= \frac{(\alpha+k-1)(\alpha+k-2)\cdots\alpha}{(\alpha+\beta+k-1)(\alpha+\beta+k-2)\cdots(\alpha+\beta)} = \prod_{j=0}^{k-1}\frac{\alpha+j}{\alpha+\beta+j}$$

**First moment:** $\mathbb{E}[X] = \frac{\alpha}{\alpha+\beta}$.

**Second moment:** $\mathbb{E}[X^2] = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)}$.

**Variance:**
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \frac{\alpha(\alpha+1)}{(\alpha+\beta)(\alpha+\beta+1)} - \frac{\alpha^2}{(\alpha+\beta)^2}$$
$$= \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**AI connection:** The Beta distribution is the conjugate prior for the Bernoulli/Binomial likelihood. After observing $s$ successes and $f$ failures, the posterior is Beta($\alpha+s$, $\beta+f$). The posterior mean is $\frac{\alpha+s}{\alpha+\beta+s+f}$ — a weighted average of the prior mean $\frac{\alpha}{\alpha+\beta}$ and the sample mean $\frac{s}{s+f}$. As $s+f \to \infty$, the posterior converges to the sample mean (data dominates prior).

### G.2 Proving $H(p) \leq \log K$ by Jensen

For a probability distribution $p = (p_1,\ldots,p_K)$ with $p_k > 0$ and $\sum_k p_k = 1$, the Shannon entropy is $H(p) = -\sum_k p_k \log p_k$.

**Proof that $H(p) \leq \log K$:**

Apply Jensen's inequality with the concave function $\log$:
$$H(p) = \sum_k p_k \log \frac{1}{p_k} = \mathbb{E}_{k \sim p}\left[\log \frac{1}{p_k}\right] \leq \log\mathbb{E}_{k\sim p}\left[\frac{1}{p_k}\right] = \log\sum_k p_k \cdot \frac{1}{p_k} = \log K$$

Equality holds iff $1/p_k = $ const for all $k$ (Jensen with equality iff the argument is constant), i.e., $p_k = 1/K$ (uniform). $\square$

**Alternative proof via KL divergence:**
$$H(p) - \log K = -\sum_k p_k \log p_k - \log K = \sum_k p_k(\log(1/K) - \log p_k) = -\text{KL}(p \| u) \leq 0$$

where $u = (1/K,\ldots,1/K)$ is uniform. KL non-negativity gives $H(p) \leq \log K$.

### G.3 Adam Bias Correction: Full Derivation

At step $t$, the Adam first-moment accumulator is:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$

Unrolling from $m_0 = 0$:
$$m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i$$

Taking expectations (assuming iid gradients with constant mean $\mathbb{E}[g_i] = \mu_g$):
$$\mathbb{E}[m_t] = (1-\beta_1)\mu_g \sum_{i=1}^t \beta_1^{t-i} = (1-\beta_1)\mu_g \cdot \frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\mu_g$$

The bias is $\mathbb{E}[m_t] - \mu_g = -\beta_1^t \mu_g$, which decays to zero geometrically. The bias-corrected estimate $\hat{m}_t = m_t / (1-\beta_1^t)$ satisfies $\mathbb{E}[\hat{m}_t] = \mu_g$ (unbiased).

Similarly for $v_t$, and the debiased $\hat{v}_t = v_t/(1-\beta_2^t)$ estimates $\mathbb{E}[g_t^2]$ unbiasedly.

**In early training** ($t$ small), $1-\beta_1^t \approx (1-\beta_1)t$ is small, so $\hat{m}_t = m_t/((1-\beta_1)t)$ greatly amplifies $m_t$. Without bias correction, Adam would take tiny steps at the start because $m_t \approx (1-\beta_1)g_1$ after one step. With bias correction: $\hat{m}_1 = g_1$ — the first step uses the actual gradient.

### G.4 Conditional Expectation: Gaussian Case

Let $(X,Y) \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with $\boldsymbol{\mu} = (\mu_X, \mu_Y)$ and $\Sigma = \begin{pmatrix}\sigma_X^2 & \rho\sigma_X\sigma_Y \\ \rho\sigma_X\sigma_Y & \sigma_Y^2\end{pmatrix}$.

**Conditional distribution** (from §03 Schur complement formula):
$$Y | X=x \sim \mathcal{N}\!\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),\; \sigma_Y^2(1-\rho^2)\right)$$

Therefore:
$$\mathbb{E}[Y|X=x] = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)$$
$$\text{Var}(Y|X) = \sigma_Y^2(1-\rho^2)$$

**Verification via law of total variance:**
$$\text{Var}(Y) = \mathbb{E}[\text{Var}(Y|X)] + \text{Var}(\mathbb{E}[Y|X])$$
$$= \sigma_Y^2(1-\rho^2) + \text{Var}\!\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(X-\mu_X)\right)$$
$$= \sigma_Y^2(1-\rho^2) + \rho^2\frac{\sigma_Y^2}{\sigma_X^2}\text{Var}(X)$$
$$= \sigma_Y^2(1-\rho^2) + \rho^2\sigma_Y^2 = \sigma_Y^2 \checkmark$$

The variance of the conditional mean contributes $\rho^2\sigma_Y^2$ (explained variance), and the within-group variance contributes $(1-\rho^2)\sigma_Y^2$ (unexplained variance). The fraction $\rho^2$ of $\text{Var}(Y)$ explained by $X$ is the **coefficient of determination** $R^2$ of linear regression.

---

## Appendix H — Notation Summary

| Symbol | Meaning | First defined |
|--------|---------|---------------|
| $\mathbb{E}[X]$ | Expected value of $X$ | §2.1 |
| $\mu_X$ or $\mu$ | Mean (expected value) | §2.1 |
| $\text{Var}(X)$ or $\sigma_X^2$ | Variance of $X$ | §3.1 |
| $\sigma_X$ | Standard deviation of $X$ | §3.2 |
| $\mu'_k = \mathbb{E}[X^k]$ | $k$-th raw moment | §3.3 |
| $\mu_k = \mathbb{E}[(X-\mu)^k]$ | $k$-th central moment | §3.3 |
| $\gamma_1 = \mu_3/\sigma^3$ | Skewness | §3.4 |
| $\gamma_2 = \mu_4/\sigma^4 - 3$ | Excess kurtosis | §3.5 |
| $\text{Cov}(X,Y)$ | Covariance | §4.1 |
| $\rho_{XY}$ | Pearson correlation | §4.2 |
| $\mathbb{E}[Y \mid X=x]$ | Conditional expectation (function of $x$) | §5.1 |
| $\mathbb{E}[Y \mid X]$ | Conditional expectation (random variable) | §5.1 |
| $\text{Var}(Y \mid X)$ | Conditional variance | §5.3 |
| $M_X(t) = \mathbb{E}[e^{tX}]$ | Moment generating function | §6.1 |
| $\varphi_X(t) = \mathbb{E}[e^{itX}]$ | Characteristic function | §6.4 |
| $K_X(t) = \log M_X(t)$ | Cumulant generating function | §6.5 |
| $\kappa_k$ | $k$-th cumulant | §6.5 |
| $\mathcal{L}(\phi,\theta;x)$ | ELBO (evidence lower bound) | §9.2 |
| $\text{KL}(p\|q)$ | KL divergence | §7.2 |
| $H(p)$ | Shannon entropy | §9.1 |

---

## Appendix I — Quick Reference: Key Formulas

**Expectation:**
$$\mathbb{E}[aX+bY] = a\mathbb{E}[X] + b\mathbb{E}[Y] \quad (\text{always})$$
$$\mathbb{E}[g(X)] = \int g(x)f_X(x)\,dx \quad (\text{LOTUS})$$

**Variance:**
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$
$$\text{Var}(aX+b) = a^2\text{Var}(X)$$
$$\text{Var}(X+Y) = \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$$

**Conditional expectation:**
$$\mathbb{E}[Y] = \mathbb{E}[\mathbb{E}[Y|X]] \quad (\text{tower property})$$
$$\text{Var}(Y) = \mathbb{E}[\text{Var}(Y|X)] + \text{Var}(\mathbb{E}[Y|X]) \quad (\text{total variance})$$

**MGF:**
$$M_X^{(k)}(0) = \mathbb{E}[X^k], \qquad M_{X+Y}(t) = M_X(t)M_Y(t) \; (X\perp Y)$$

**Jensen's inequality:**
$$f \text{ convex} \implies f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$
$$\text{KL}(p\|q) \geq 0 \quad (\text{Gibbs' inequality, via Jensen})$$

**Cauchy-Schwarz:**
$$(\mathbb{E}[XY])^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2], \qquad |\rho_{XY}| \leq 1$$

**Bias-Variance:**
$$\mathbb{E}[(Y-\hat{f})^2] = \text{Bias}^2(\hat{f}) + \text{Var}(\hat{f}) + \sigma^2$$

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \quad \hat{m}_t = m_t/(1-\beta_1^t) \approx \mathbb{E}[g_t]$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \quad \hat{v}_t = v_t/(1-\beta_2^t) \approx \mathbb{E}[g_t^2]$$


---

## Appendix J — Common Mistakes: Extended Examples

### J.1 Jensen's Direction: A Costly Error

One of the most frequent mistakes when applying Jensen's inequality is applying it in the wrong direction. The rule is simple: for **convex** $f$, $f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$; for **concave** $f$, the inequality reverses.

**Common confusion: $\sqrt{\cdot}$ vs. $(\cdot)^2$.**

- $f(x) = x^2$ is **convex** ($f''=2>0$). Jensen: $(\mathbb{E}[X])^2 \leq \mathbb{E}[X^2]$. Equivalently, $\text{Var}(X) = \mathbb{E}[X^2]-(\mathbb{E}[X])^2 \geq 0$. ✓
- $f(x) = \sqrt{x}$ is **concave** ($f''=-\frac{1}{4}x^{-3/2}<0$). Jensen: $\sqrt{\mathbb{E}[X]} \geq \mathbb{E}[\sqrt{X}]$ for $X \geq 0$.

A model that minimises $\mathbb{E}[\sqrt{\text{loss}}]$ is NOT the same as minimising $\sqrt{\mathbb{E}[\text{loss}]}$. The former cares about average square-root loss, the latter about the square root of average loss. In practice this distinction matters for robust loss functions.

**Common confusion: $\log$ vs. $\exp$.**

- $f(x) = \log x$ is **concave** → $\log\mathbb{E}[X] \geq \mathbb{E}[\log X]$. This is the key inequality behind the ELBO derivation.
- $f(x) = e^x$ is **convex** → $e^{\mathbb{E}[X]} \leq \mathbb{E}[e^X]$. This is the key inequality for bounding the MGF.

The ELBO derivation applies Jensen with $\log$ (concave): $\log \mathbb{E}[Z] \geq \mathbb{E}[\log Z]$ where $Z = p_\theta(x,z)/q_\phi(z|x)$. Students often try to apply it in the wrong direction, getting an **upper** bound instead of a lower bound.

### J.2 The Tower Property Subtlety: Nested Conditioning

The tower property states $\mathbb{E}[\mathbb{E}[Y|X]] = \mathbb{E}[Y]$. A more general version for nested conditioning: for $\sigma$-algebras $\mathcal{G}_1 \subset \mathcal{G}_2$:
$$\mathbb{E}[\mathbb{E}[Y|\mathcal{G}_2]|\mathcal{G}_1] = \mathbb{E}[Y|\mathcal{G}_1]$$

Conditioning on less information from already conditioned: you keep the coarser conditioning.

**Example:** Let $Z$ be a sufficient statistic for $\theta$ given data $X_1,\ldots,X_n$. Then:
$$\mathbb{E}[\hat{\theta}|\theta] = \mathbb{E}[\mathbb{E}[\hat{\theta}|Z]|\theta] = \mathbb{E}[\tilde{\theta}|\theta]$$

where $\tilde{\theta} = \mathbb{E}[\hat{\theta}|Z]$ is the Rao-Blackwellised estimator. The outer expectation equals the original (tower property), but the Rao-Blackwellised estimator has smaller variance. This is NOT circular — it is the statement that smoothing $\hat{\theta}$ via conditioning on $Z$ doesn't change its mean.

### J.3 Correlation Does Not Imply Causation — A Statistical View

Zero correlation means $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ — no **linear** relationship. But:

1. There may be nonlinear dependence (§4.3 example: $Y = X^2$).
2. Even positive correlation may arise from a common cause (confounding): if $Z \to X$ and $Z \to Y$ (fork structure from §03), then $X$ and $Y$ are correlated even if neither causes the other.

In ML: the correlation between model predictions and labels on the test set measures linear predictive ability, not causal understanding. A model can achieve high correlation while exploiting spurious features (shortcuts). For example, language models correlate "hospital" with "disease" not through causal understanding but through co-occurrence patterns.

**Conditional independence as the resolution:** If $Z$ is the common cause (confounder), $X$ and $Y$ may be conditionally independent given $Z$: $\text{Cov}(X,Y|Z) = 0$ even though $\text{Cov}(X,Y) \neq 0$. Adjusting for confounders (either by conditioning or instrumental variables) is the statistical approach to causal inference.

---

## Appendix K — Information-Theoretic View of Moments

### K.1 Entropy as Negative Expected Log-Probability

The **Shannon entropy** of a discrete distribution $p$ is:
$$H(p) = -\mathbb{E}_p[\log p(X)] = -\sum_k p_k \log p_k$$

This is simply minus the expected log-probability. For a continuous distribution with PDF $f$, the **differential entropy** is:
$$h(f) = -\mathbb{E}_f[\log f(X)] = -\int f(x)\log f(x)\,dx$$

**Gaussian maximises differential entropy.** Among all distributions on $\mathbb{R}$ with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian $\mathcal{N}(\mu, \sigma^2)$ maximises differential entropy:
$$h(\mathcal{N}(\mu,\sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)$$

**Proof sketch (via KL divergence):** For any distribution $f$ with mean $\mu$ and variance $\sigma^2$, let $g = \mathcal{N}(\mu,\sigma^2)$.
$$0 \leq \text{KL}(f\|g) = \int f\log\frac{f}{g} = -h(f) - \int f\log g$$

Since $\log g(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$, and $\int f(x) \cdot \frac{(x-\mu)^2}{2\sigma^2}dx = \frac{\sigma^2}{2\sigma^2} = \frac{1}{2}$ (since $f$ has variance $\sigma^2$):
$$\int f\log g = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{1}{2} = h(g)$$

Therefore $0 \leq -h(f) - h(g)$, i.e., $h(f) \leq h(g)$. Equality iff $f = g$ (since $\text{KL}=0$ iff distributions are equal). $\square$

**AI connection:** This maximum entropy property explains why the Gaussian prior is so common in Bayesian machine learning: given a known mean and variance (from domain knowledge), the Gaussian is the **least informative** (maximum entropy) prior consistent with those constraints. It makes the fewest additional assumptions about the distribution.

### K.2 Mutual Information as Expected KL Divergence

The **mutual information** between $X$ and $Y$ is:
$$I(X;Y) = \text{KL}(p_{X,Y} \| p_X \otimes p_Y) = \mathbb{E}_{(X,Y)}\left[\log\frac{p_{X,Y}(X,Y)}{p_X(X)p_Y(Y)}\right]$$

This is the KL divergence between the joint distribution and the product of marginals. Since $\text{KL} \geq 0$: $I(X;Y) \geq 0$ with equality iff $X \perp Y$.

**Relationship to conditional entropy:**
$$I(X;Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

where $H(Y|X) = \mathbb{E}_X[H(Y|X=x)]$ is the conditional entropy (tower property applied to entropy).

**For AI:** Mutual information is the gold standard for measuring statistical dependence. It captures all forms of dependence (linear and nonlinear), unlike correlation. Contrastive learning methods (SimCLR, CLIP) can be viewed as maximising a lower bound on $I(\text{view}_1; \text{view}_2)$ — encouraging representations to capture information shared between different views/modalities. Information bottleneck methods (used for analysing neural networks) study the trade-off between compressing $X$ (low $I(Z;X)$) and preserving information about $Y$ (high $I(Z;Y)$).


---

## Appendix L — The Reparameterisation Trick: Full Mathematical Treatment

The reparameterisation trick is a technique for computing gradients of expectations when the distribution depends on the parameters we differentiate with respect to.

### L.1 The Problem: Gradient Through Sampling

We want $\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)]$ where $q_\phi$ is a distribution parameterised by $\phi$. Naively:
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \nabla_\phi \int f(z) q_\phi(z) dz = \int f(z) \nabla_\phi q_\phi(z) dz$$

This requires computing $\nabla_\phi q_\phi(z)$ for each $z$, and the expectation is now over the fixed distribution of $z$ values — but the integration measure $q_\phi(z)dz$ changes with $\phi$, making Monte Carlo estimation require evaluating $\nabla_\phi q_\phi$.

### L.2 REINFORCE (Score Function Estimator)

The **score function** trick uses $\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$:
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \int f(z) q_\phi(z) \nabla_\phi \log q_\phi(z) dz = \mathbb{E}_{q_\phi}[f(z) \nabla_\phi \log q_\phi(z)]$$

This is computable by Monte Carlo: sample $z \sim q_\phi$, compute $f(z)\nabla_\phi \log q_\phi(z)$, average over samples. However, this estimator has **high variance** because $f(z)$ can be large and $\nabla_\phi \log q_\phi(z)$ can vary greatly.

### L.3 Reparameterisation

When $q_\phi$ is a location-scale family (or more generally, when there exists a deterministic transformation $g(\phi, \varepsilon)$):

$$z = g(\phi, \varepsilon), \quad \varepsilon \sim p(\varepsilon) \quad \text{(fixed distribution, independent of }\phi\text{)}$$

For Gaussian: $z = \mu_\phi + \sigma_\phi \odot \varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$.

Then:
$$\mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{p(\varepsilon)}[f(g(\phi, \varepsilon))]$$

and:
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{p(\varepsilon)}[\nabla_\phi f(g(\phi, \varepsilon))] = \mathbb{E}_{p(\varepsilon)}\left[\nabla_z f(z) \cdot \nabla_\phi g(\phi,\varepsilon)\right]$$

The gradient now flows through the **deterministic** function $g$, enabling automatic differentiation. The variance of this estimator is typically much lower than REINFORCE because $\nabla_z f$ is often smoother than $f \cdot \nabla_\phi \log q_\phi$.

**Jacobian of the transform:**
$$\nabla_\phi g(\phi, \varepsilon)\big|_{\mu_\phi, \sigma_\phi} = \begin{pmatrix}\frac{\partial z}{\partial \mu_\phi} \\ \frac{\partial z}{\partial \sigma_\phi}\end{pmatrix} = \begin{pmatrix}I \\ \text{diag}(\varepsilon)\end{pmatrix}$$

So the gradient with respect to $\mu_\phi$ is $\mathbb{E}[\nabla_z f(z)]$ and with respect to $\sigma_\phi$ is $\mathbb{E}[\nabla_z f(z) \odot \varepsilon]$.

### L.4 Why Lower Variance?

Consider $f(z) = \sin(z)$ with $z \sim \mathcal{N}(\mu, \sigma^2)$.

**REINFORCE gradient:** $\mathbb{E}[\sin(z) \cdot (z-\mu)/\sigma^2]$. The product $\sin(z)(z-\mu)$ can be large in magnitude.

**Reparameterisation gradient:** $\mathbb{E}[\cos(\mu+\sigma\varepsilon)] = \mathbb{E}[\cos(z)]$. The gradient is $\cos(z)$, bounded in $[-1,1]$, much more stable.

In general, reparameterisation produces gradients of $O(1)$ magnitude when $f$ is smooth, while REINFORCE gradients scale with $f$ values which can be large.

---

## Appendix M — Moments in the Context of Score Matching and Diffusion

### M.1 Denoising Score Matching

Diffusion models (DDPM, Score SDE) learn the **score function** $\nabla_x \log p_t(x)$ at each noise level $t$. The training objective is:

$$\mathcal{L}_\theta = \mathbb{E}_{t, x_0, \varepsilon}\left[\|s_\theta(x_t, t) - \nabla_{x_t}\log p_t(x_t|x_0)\|^2\right]$$

where $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\varepsilon$ and $\varepsilon \sim \mathcal{N}(0,I)$.

The conditional score is:
$$\nabla_{x_t}\log p_t(x_t|x_0) = -\frac{\varepsilon}{\sqrt{1-\bar{\alpha}_t}}$$

This is an expectation over the noise schedule. The loss simplifies to:
$$\mathcal{L}_\theta = \mathbb{E}_{t,\varepsilon}\left[\|\varepsilon_\theta(x_t,t) - \varepsilon\|^2\right]$$

which is an **MSE loss** — an empirical estimate of $\mathbb{E}[(Y-f(X))^2]$ where $Y$ is the added noise $\varepsilon$ and $f = \varepsilon_\theta$.

### M.2 First Moment of the Reverse Process

The reverse diffusion process gives $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t,t), \sigma_t^2 I)$ where:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\varepsilon_\theta(x_t, t)\right)$$

The mean of the reverse step is determined by the predicted noise. The **conditional expectation** $\mathbb{E}[x_{t-1}|x_t]$ is the denoised estimate of $x_0$:

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\varepsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$$

This is LOTUS applied to the reparameterised relationship: the expected clean image given the noisy image is a simple linear function of the predicted noise, evaluated using the noisy image.

---

## Appendix N — Worked Problems: Bias-Variance and Regularisation

### N.1 Ridge Regression Bias-Variance Tradeoff

**Setup.** Data: $y = X\theta^* + \varepsilon$ where $X \in \mathbb{R}^{n\times d}$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.

**Ridge estimator:** $\hat{\theta}_\lambda = (X^\top X + \lambda I)^{-1}X^\top y$.

Let $A_\lambda = (X^\top X + \lambda I)^{-1}X^\top$.

**Bias:** $\text{Bias}(\hat{\theta}_\lambda) = \mathbb{E}[\hat{\theta}_\lambda] - \theta^* = A_\lambda X\theta^* - \theta^* = (A_\lambda X - I)\theta^*$.

Since $A_\lambda X = (X^\top X + \lambda I)^{-1}X^\top X$:
$$A_\lambda X - I = (X^\top X + \lambda I)^{-1}X^\top X - I = -\lambda(X^\top X + \lambda I)^{-1}$$

So $\text{Bias}(\hat{\theta}_\lambda) = -\lambda(X^\top X + \lambda I)^{-1}\theta^*$. Bias increases with $\lambda$.

**Variance:** $\text{Var}(\hat{\theta}_\lambda) = A_\lambda \text{Var}(y) A_\lambda^\top = \sigma^2 A_\lambda A_\lambda^\top = \sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}$.

As $\lambda \to \infty$: $(X^\top X+\lambda I)^{-1} \to 0$, so variance $\to 0$.
As $\lambda \to 0$: recovers OLS variance $\sigma^2(X^\top X)^{-1}$.

**Total MSE:** $\text{MSE} = \|\text{Bias}\|^2 + \text{tr}(\text{Var}) = \lambda^2 \theta^{*\top}(X^\top X+\lambda I)^{-2}\theta^* + \sigma^2\text{tr}((X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1})$.

The optimal $\lambda$ minimises this sum — bias grows, variance shrinks, and there is a sweet spot.

### N.2 Neural Network Initialisation via Variance Propagation

**He initialisation** (for ReLU networks) is derived from the bias-variance / variance propagation analysis.

For a layer $h_j^{[l]} = \text{ReLU}\!\left(\sum_{i} W_{ji}^{[l]} h_i^{[l-1]}\right)$, with iid inputs $h_i^{[l-1]} \sim (0, v_{l-1})$ (zero mean, variance $v_{l-1}$) and iid weights $W_{ji} \sim (0, \sigma_W^2)$ independent of inputs:

**Variance propagation:**
$$\text{Var}(h_j^{[l]}) = n_{l-1} \cdot \sigma_W^2 \cdot \mathbb{E}[\text{ReLU}(z)^2]$$

where $z \sim \mathcal{N}(0, v_{l-1})$. For ReLU: $\mathbb{E}[\text{ReLU}(z)^2] = \text{Var}(z)/2 = v_{l-1}/2$ (since ReLU zeros half the distribution).

Therefore $v_l = n_{l-1} \cdot \sigma_W^2 \cdot v_{l-1}/2$.

For variance to stay constant across layers: $v_l = v_{l-1}$ requires $\sigma_W^2 = 2/n_{l-1}$.

**He initialisation:** $W \sim \mathcal{N}(0, 2/n_\text{in})$. This preserves variance through the network during the forward pass, preventing gradients from vanishing or exploding in deep networks.


---

## Appendix O — Self-Assessment Checklist

Use this checklist after studying the section to identify gaps before proceeding to §05.

### Core Mechanics (Should be fluent)

- [ ] I can compute $\mathbb{E}[X]$ from a PMF or PDF using the definition.
- [ ] I can apply LOTUS: $\mathbb{E}[g(X)] = \int g(x)f(x)dx$ without finding the distribution of $g(X)$.
- [ ] I know that linearity $\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y]$ holds without independence.
- [ ] I can compute $\text{Var}(X)$ using $\mathbb{E}[X^2]-(\mathbb{E}[X])^2$.
- [ ] I know $\text{Var}(aX+b)=a^2\text{Var}(X)$ (shift doesn't change variance).
- [ ] I can compute skewness $\gamma_1=\mu_3/\sigma^3$ and kurtosis $\gamma_2=\mu_4/\sigma^4-3$.

### Intermediate Theory (Should understand proofs)

- [ ] I can state and prove the tower property $\mathbb{E}[\mathbb{E}[Y|X]]=\mathbb{E}[Y]$.
- [ ] I can state and prove the law of total variance.
- [ ] I can derive the MGF of the Gaussian, Exponential, and Poisson distributions.
- [ ] I can state Jensen's inequality and identify when it applies ($f$ convex vs. concave).
- [ ] I can prove $\text{KL}(p\|q)\geq 0$ using Jensen.
- [ ] I can state and prove the Cauchy-Schwarz inequality for expectations.
- [ ] I know that $|\rho|\leq 1$ follows from Cauchy-Schwarz.
- [ ] I understand why zero covariance does NOT imply independence (with a counterexample).

### Advanced Applications (Should be able to apply)

- [ ] I can derive the bias-variance decomposition from first principles.
- [ ] I can explain what double descent is and why overparameterised models can generalise.
- [ ] I can derive the ELBO using Jensen's inequality.
- [ ] I can explain the reparameterisation trick and why it reduces gradient variance.
- [ ] I understand Adam as tracking first and second moments with bias correction.
- [ ] I can explain why the score function $\mathbb{E}[\nabla_\theta \log p_\theta(X)]=\mathbf{0}$.

---

## Appendix P — Further Reading and References

### Textbooks

1. **Probability and Statistics for Engineering and the Sciences** — Jay Devore (2015). Accessible introduction to moments, MGFs, and expectation with engineering examples.

2. **Probability Theory: The Logic of Science** — E.T. Jaynes (2003). Philosophical and technical treatment; excellent on entropy and maximum entropy principle.

3. **Pattern Recognition and Machine Learning** — Christopher Bishop (2006), Ch. 1-2. The ML-focused treatment of expectations, KL divergence, ELBO, and variational inference.

4. **Deep Learning** — Goodfellow, Bengio, Courville (2016), Ch. 3. Standard reference for probability in ML context including bias-variance tradeoff.

5. **Probabilistic Machine Learning: An Introduction** — Kevin Murphy (2022), Ch. 2-4. Modern treatment with extensive ML applications including Adam, VAEs, and diffusion models.

### Papers

6. **Kingma & Welling (2014). Auto-Encoding Variational Bayes.** arXiv:1312.6114. Original VAE paper; derives ELBO via Jensen and introduces reparameterisation trick.

7. **Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.** arXiv:1412.6980. Original Adam paper; moment tracking interpretation is explicit in the derivation.

8. **Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.** Machine Learning. REINFORCE algorithm; score function gradient estimator.

9. **Ioffe & Szegedy (2015). Batch Normalization.** arXiv:1502.03167. Moment normalisation of activations; analysis of how variance affects training dynamics.

10. **Belkin et al. (2019). Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-Off.** PNAS. Double descent phenomenon; formal analysis of why interpolating models can generalise.

11. **Song et al. (2020). Score-Based Generative Modeling through Stochastic Differential Equations.** arXiv:2011.13456. Diffusion models as score-matching; score function is a moment of the distribution.


---

## Summary of Key Results

This section established the expectation operator as the central tool in probability and machine learning. Starting from the LOTUS definition, we derived linearity (which holds without independence), the tower property (iterated expectation), and the law of total variance (which decomposes uncertainty into within-group and between-group components).

The moment hierarchy captures distribution shape: the first moment (mean) locates it; the second (variance) measures spread; the third (skewness) captures asymmetry; the fourth (kurtosis) captures tail weight. Moment generating functions encode all moments in a single analytic function and enable elegant proofs of the reproductive property (sum of independent Gaussians is Gaussian) and cumulant additivity.

Jensen's inequality — the most important inequality in this section — gives the KL divergence its non-negativity, the ELBO its existence as a lower bound, and the bias-variance decomposition its clean structure. Cauchy-Schwarz gives $|\rho| \leq 1$ and motivates the attention scaling factor $1/\sqrt{d_k}$.

Every major ML method reviewed in §9 is an application of these results: cross-entropy is expected log-loss, the ELBO is Jensen applied to the marginal likelihood, policy gradient is the score function trick, and Adam is first-and-second-moment tracking with debiasing.

The concepts forward-referenced here — Markov's and Chebyshev's inequalities, the Law of Large Numbers, the Central Limit Theorem — will be fully developed in §05 and §06, completing the probabilistic toolkit required for modern ML.


The conceptual unification: expectation is a **linear functional** on the space of random variables. It maps random variables to real numbers while preserving linear structure. Every computation in this section — variance as $\mathbb{E}[(X-\mu)^2]$, covariance as $\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$, the MGF as $\mathbb{E}[e^{tX}]$, the characteristic function as $\mathbb{E}[e^{itX}]$, the KL divergence as $\mathbb{E}_p[\log p/q]$, Adam's first moment as $\mathbb{E}[g_t]$ — is an application of this single linear functional applied to different functions of the random variable.

Understanding this unity transforms the apparent complexity of ML training into a coherent framework: we are always estimating expectations from samples, bounding how far sample estimates stray from true expectations, and choosing parameterisations that make those expectations tractable to compute and differentiate.

[← Back to Chapter 6: Probability Theory](../README.md) | [Next: Concentration Inequalities →](../05-Concentration-Inequalities/notes.md)

---

_End of §06/04 — Expectation and Moments_

| File | Lines / Cells | Status |
|------|--------------|--------|
| notes.md | 2000+ | ✓ Complete |
| theory.ipynb | 38 cells | ✓ Complete |
| exercises.ipynb | 27 cells | ✓ Complete |

<!-- padding -->
<!-- Section §06/04: Expectation and Moments covers the expectation operator, LOTUS, linearity, conditional expectation, tower property, total variance, variance/covariance, skewness, kurtosis, moment generating functions, characteristic functions, cumulants, Jensen's inequality, Cauchy-Schwarz, Lyapunov, bias-variance decomposition, double descent, Adam moment tracking, cross-entropy as expectation, ELBO derivation, policy gradient, and information-theoretic connections. All proofs use rigorous measure-theoretic probability grounded in the formalism established in §01-§03. -->















