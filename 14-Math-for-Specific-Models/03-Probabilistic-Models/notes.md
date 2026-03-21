[← Back to Math for Specific Models](../README.md) | [Next: RNN and LSTM Math →](../04-RNN-and-LSTM-Math/notes.md)

---

# Probabilistic Models

> _"Probability theory is nothing but common sense reduced to calculation."_
> — Pierre-Simon Laplace, 1812

## Overview

Probabilistic models are the mathematical language of uncertainty. Every dataset is finite and noisy, every measurement is corrupted, and every model is an approximation — probability theory provides the uniquely coherent framework for reasoning under these conditions. This section develops the full mathematical machinery of probabilistic machine learning, from the measure-theoretic foundations of Bayes' theorem through the modern generative models that power state-of-the-art image synthesis, protein folding, and language generation.

The section is structured in three arcs. The **foundational arc** (§1–§5) covers the exponential family, graphical models, latent variable models, and the EM algorithm — classical tools that remain essential for understanding every modern method. The **variational arc** (§6–§8) develops variational inference, the ELBO, VAEs, and MCMC — the engine of scalable approximate Bayesian inference. The **modern arc** (§9–§15) covers Gaussian processes, HMMs, Bayesian neural networks, normalizing flows, score matching, diffusion models, flow matching, and the striking 2021–2025 theory of in-context learning as Bayesian inference.

Every topic is connected to concrete AI systems: the ELBO reappears in VQ-VAE (DALL-E), score matching drives Stable Diffusion and Sora, flow matching powers Flux and Stable Diffusion 3, GPs underpin Bayesian hyperparameter optimisation, and in-context learning theory explains why GPT-4 can solve tasks from a few examples.

## Prerequisites

- Probability basics: expectation, variance, Gaussian distribution (§13-01 Probability and Statistics)
- Linear algebra: eigendecomposition, SVD, matrix calculus (§02 Linear Algebra)
- Neural networks: forward pass, backpropagation, activation functions (§14-02 Neural Networks)
- Calculus: gradients, chain rule, Jensen's inequality (§05 Calculus)
- Information theory: KL divergence, cross-entropy, entropy (§13-02 Information Theory)

## Companion Notebooks

| Notebook                           | Description                                                                                                                                                             |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demos: exponential family moments, EM for GMM, ELBO optimisation, VAE from scratch, MCMC chains, GP regression, HMM inference, diffusion forward/reverse    |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: conjugate updates, EM convergence, CAVI, reparameterisation, MH sampler, GP posterior, Viterbi, flow log-likelihood, score matching loss, ICL Bayes |

## Learning Objectives

After completing this section, you will:

- State Bayes' theorem and derive the posterior update for any conjugate family
- Write the exponential family canonical form and compute moments from the log-partition function
- Apply d-separation to read conditional independence from a Bayesian network
- Derive the EM lower bound using Jensen's inequality and state the E- and M-step updates
- Derive the ELBO and explain why mean-field CAVI minimises $D_{\mathrm{KL}}(q \| p)$
- Implement the reparameterisation trick and explain why it enables gradients through stochastic nodes
- Describe the Metropolis-Hastings acceptance ratio and explain why it preserves the target distribution
- Write the GP predictive mean and variance in closed form and optimise hyperparameters via marginal likelihood
- Implement the forward-backward algorithm for HMMs in $O(TK^2)$
- Connect denoising score matching to diffusion model training and state the reverse-time SDE
- Explain flow matching and how it differs from diffusion models
- Describe the Xie et al. (2021) framework for in-context learning as implicit Bayesian inference

---

## Table of Contents

- [1. Intuition and Historical Context](#1-intuition-and-historical-context)
  - [1.1 What Is a Probabilistic Model?](#11-what-is-a-probabilistic-model)
  - [1.2 Why Probability for AI?](#12-why-probability-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 The Bayesian–Frequentist Divide](#14-the-bayesianfrequentist-divide)
- [2. Formal Foundations](#2-formal-foundations)
  - [2.1 Probability Spaces and Random Variables](#21-probability-spaces-and-random-variables)
  - [2.2 Bayes' Theorem](#22-bayes-theorem)
  - [2.3 Exponential Family](#23-exponential-family)
  - [2.4 Conjugate Priors](#24-conjugate-priors)
- [3. Graphical Models](#3-graphical-models)
  - [3.1 Bayesian Networks](#31-bayesian-networks)
  - [3.2 D-Separation](#32-d-separation)
  - [3.3 Markov Random Fields](#33-markov-random-fields)
  - [3.4 Factor Graphs and Belief Propagation](#34-factor-graphs-and-belief-propagation)
- [4. Latent Variable Models](#4-latent-variable-models)
  - [4.1 Gaussian Mixture Models](#41-gaussian-mixture-models)
  - [4.2 Factor Analysis and Probabilistic PCA](#42-factor-analysis-and-probabilistic-pca)
  - [4.3 Why Latent Variables?](#43-why-latent-variables)
- [5. Expectation-Maximization Algorithm](#5-expectation-maximization-algorithm)
  - [5.1 The EM Framework](#51-the-em-framework)
  - [5.2 Convergence Proof](#52-convergence-proof)
  - [5.3 EM for Gaussian Mixture Models](#53-em-for-gaussian-mixture-models)
  - [5.4 Generalised EM and Hard-EM](#54-generalised-em-and-hard-em)
- [6. Variational Inference](#6-variational-inference)
  - [6.1 The ELBO](#61-the-elbo)
  - [6.2 Mean-Field Approximation and CAVI](#62-mean-field-approximation-and-cavi)
  - [6.3 Black-Box Variational Inference](#63-black-box-variational-inference)
  - [6.4 Importance-Weighted Bounds](#64-importance-weighted-bounds)
- [7. Variational Autoencoders](#7-variational-autoencoders)
  - [7.1 Architecture and Generative Model](#71-architecture-and-generative-model)
  - [7.2 Reparameterisation Trick](#72-reparameterisation-trick)
  - [7.3 VAE Training Dynamics](#73-vae-training-dynamics)
  - [7.4 Latent Space Geometry](#74-latent-space-geometry)
- [8. Markov Chain Monte Carlo](#8-markov-chain-monte-carlo)
  - [8.1 Markov Chains and Stationarity](#81-markov-chains-and-stationarity)
  - [8.2 Metropolis-Hastings and Gibbs Sampling](#82-metropolis-hastings-and-gibbs-sampling)
  - [8.3 Hamiltonian Monte Carlo](#83-hamiltonian-monte-carlo)
  - [8.4 Convergence Diagnostics](#84-convergence-diagnostics)
- [9. Gaussian Processes](#9-gaussian-processes)
  - [9.1 GP as Distribution over Functions](#91-gp-as-distribution-over-functions)
  - [9.2 GP Regression](#92-gp-regression)
  - [9.3 Kernel Design](#93-kernel-design)
  - [9.4 Scalable GPs](#94-scalable-gps)
- [10. Hidden Markov Models](#10-hidden-markov-models)
  - [10.1 HMM Structure](#101-hmm-structure)
  - [10.2 Forward-Backward Algorithm](#102-forward-backward-algorithm)
  - [10.3 Viterbi Algorithm](#103-viterbi-algorithm)
  - [10.4 Baum-Welch and EM for HMMs](#104-baum-welch-and-em-for-hmms)
- [11. Bayesian Neural Networks and Uncertainty](#11-bayesian-neural-networks-and-uncertainty)
  - [11.1 Weight Distributions](#111-weight-distributions)
  - [11.2 Laplace Approximation](#112-laplace-approximation)
  - [11.3 MC Dropout as Approximate Inference](#113-mc-dropout-as-approximate-inference)
  - [11.4 Aleatoric vs. Epistemic Uncertainty](#114-aleatoric-vs-epistemic-uncertainty)
- [12. Normalizing Flows](#12-normalizing-flows)
  - [12.1 Change of Variables Formula](#121-change-of-variables-formula)
  - [12.2 Flow Architectures](#122-flow-architectures)
  - [12.3 Continuous Normalizing Flows](#123-continuous-normalizing-flows)
  - [12.4 Training and Applications](#124-training-and-applications)
- [13. Score Matching and Diffusion Models](#13-score-matching-and-diffusion-models)
  - [13.1 Score Functions and Energy-Based Models](#131-score-functions-and-energy-based-models)
  - [13.2 Denoising Score Matching](#132-denoising-score-matching)
  - [13.3 Score-Based SDEs](#133-score-based-sdes)
  - [13.4 DDPM, DDIM, and Guidance](#134-ddpm-ddim-and-guidance)
- [14. Flow Matching](#14-flow-matching)
  - [14.1 Probability Paths and Vector Fields](#141-probability-paths-and-vector-fields)
  - [14.2 Flow Matching Objective](#142-flow-matching-objective)
  - [14.3 Rectified Flows](#143-rectified-flows)
  - [14.4 Flow Matching vs. Diffusion](#144-flow-matching-vs-diffusion)
- [15. In-Context Learning as Bayesian Inference](#15-in-context-learning-as-bayesian-inference)
  - [15.1 Xie et al. (2021) Framework](#151-xie-et-al-2021-framework)
  - [15.2 Transformer as Approximate Posterior](#152-transformer-as-approximate-posterior)
  - [15.3 Modern Extensions (2024–2025)](#153-modern-extensions-20242025)
  - [15.4 Implications for LLM Design](#154-implications-for-llm-design)
- [16. Common Mistakes](#16-common-mistakes)
- [17. Exercises](#17-exercises)
- [18. Why This Matters for AI (2026 Perspective)](#18-why-this-matters-for-ai-2026-perspective)
- [19. Conceptual Bridge](#19-conceptual-bridge)

---

## 1. Intuition and Historical Context

### 1.1 What Is a Probabilistic Model?

A probabilistic model is a mathematical object that assigns a probability distribution to every quantity of interest — data, parameters, latent structure, future observations. Unlike deterministic models that return a single answer, a probabilistic model returns a _distribution over_ answers, encoding not just a best guess but a calibrated measure of how confident that guess is.

**Three types of uncertainty** that probabilistic models handle:

- **Aleatoric uncertainty** (irreducible noise): randomness inherent in the data-generating process — measurement noise, sensor jitter, human label disagreement. No amount of additional data removes it.
- **Epistemic uncertainty** (model uncertainty): uncertainty due to limited data. Can be reduced by observing more. A trained Bayesian model knows what it doesn't know.
- **Distributional shift**: the test distribution differs from training. Calibrated models detect this as high uncertainty on out-of-distribution inputs.

**For AI**: a language model that outputs a single token is a deterministic policy; a language model that maintains a full distribution $p(w_t \mid w_{<t})$ is a probabilistic model. Sampling temperature, nucleus sampling, and uncertainty quantification all require this probabilistic view. Diffusion models, VAEs, and normalizing flows _are_ probabilistic generative models — they define a distribution over images, audio, or proteins.

**Non-examples** (deterministic models that lack the probabilistic structure):

- Standard SVMs without probability calibration
- Deterministic autoencoders without latent distributions
- Decision trees without probability estimates at leaves

### 1.2 Why Probability for AI?

**Calibration matters.** A model that predicts 70% confidence should be correct 70% of the time. Modern neural networks are notoriously overconfident — they output near-zero entropy even on inputs far from training data. Probabilistic models with proper priors are naturally calibrated because the prior acts as regularisation that prevents overconfident posteriors.

**Missing data is everywhere.** Clinical datasets have missing lab values; text has missing context; recommendation systems have sparse interaction matrices. Probabilistic models handle missing data principled: marginalise over what is unknown.

$$p(\mathbf{x}_\text{obs}) = \int p(\mathbf{x}_\text{obs}, \mathbf{x}_\text{miss})\, d\mathbf{x}_\text{miss}$$

**Bayesian learning is optimal.** By the Bernardo-Smith representation theorem, any coherent belief system satisfying the axioms of probability must update via Bayes' theorem. Frequentist alternatives violate coherence in pathological cases. In finite-sample regimes, Bayesian methods often outperform MLE because the prior captures inductive bias.

**For LLMs**: in-context learning (§15) is now understood as _implicit_ Bayesian inference — the transformer posterior-updates over latent hypotheses given in-context examples. Constitutional AI, RLHF preference modelling, and uncertainty-aware decoding all rest on probabilistic foundations.

### 1.3 Historical Timeline

```text
PROBABILISTIC MODELS — HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1763  Bayes' theorem (Thomas Bayes, posthumous)
  1812  Théorie analytique des probabilités (Laplace) — systematic Bayesian
  1877  Boltzmann distribution — statistical mechanics, energy-based models
  1933  Kolmogorov axioms — rigorous measure-theoretic foundations
  1950  EM precursor (Hartley) — likelihood maximisation with latent vars
  1953  Metropolis algorithm — Monte Carlo for statistical physics
  1957  Gibbs distribution (Besag) — MRFs for spatial statistics
  1972  Jensen's inequality in statistics (Jensen 1906, ML use 1970s)
  1977  EM algorithm formalised — Dempster, Laird & Rubin
  1982  Belief propagation — Pearl; BN notation (1985)
  1987  Hamiltonian Monte Carlo — Duane, Kennedy, Pendleton, Roweth
  1988  Probabilistic Graphical Models framework — Pearl
  1995  MCMC revolution — geostatistics, Bayesian hierarchical models
  1996  Factor graphs — Forney
  1997  Normalizing flows prototype — Rezende & Mohamed (2015 paper)
  2003  Latent Dirichlet Allocation — Blei, Ng, Jordan
  2006  Gaussian Processes for ML — Rasmussen & Williams book
  2013  Variational Autoencoders — Kingma & Welling (arXiv 1312.6114)
  2015  Normalizing flows — Rezende & Mohamed; NICE (Dinh et al.)
  2019  Score-based generative models — Song & Ermon
  2020  DDPM — Ho et al.; score SDEs — Song et al.
  2021  In-context learning as Bayes — Xie et al.; Flow Matching — Lipman
  2022  Diffusion + transformers (DiT — Peebles & Xie)
  2023  Consistency models; Mamba SSMs
  2024  Flow Matching in production (Stable Diffusion 3, Flux)
  2025  Prior-data fitted networks; Energy Matching; ICL Bayes ICML 2025

════════════════════════════════════════════════════════════════════════
```

### 1.4 The Bayesian–Frequentist Divide

The frequentist interprets probability as long-run frequency: $P(A) = \lim_{n\to\infty} n_A/n$. Parameters are fixed unknowns; only data is random. Maximum likelihood estimation finds $\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} p(\mathcal{D} \mid \boldsymbol{\theta})$.

The Bayesian treats parameters as random variables with a prior $p(\boldsymbol{\theta})$ encoding beliefs before data. After observing $\mathcal{D}$:

$$p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})}{p(\mathcal{D})}$$

**Key consequences:**

- Bayesian inference returns a _distribution_, not a point estimate — uncertainty is explicit
- Predictions integrate over parameter uncertainty: $p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid \mathbf{x}^*, \boldsymbol{\theta})\, p(\boldsymbol{\theta} \mid \mathcal{D})\, d\boldsymbol{\theta}$
- MAP estimation ($\arg\max p(\boldsymbol{\theta} \mid \mathcal{D})$) is a compromise: Bayesian objective, point estimate
- **Bernstein-von Mises theorem**: as $n \to \infty$, the posterior concentrates around the MLE (under regularity), so Bayesian and frequentist methods agree asymptotically. Differences matter most in small-$n$ regimes — exactly where AI practitioners care most.

**For AI**: batch normalisation statistics, learning rate schedules, and weight decay are all implicitly frequentist. Meanwhile, hyperparameter optimisation via Gaussian processes (Bayesian optimisation), VAEs, and diffusion models are explicitly Bayesian. Understanding both sides clarifies _why_ weight decay equals MAP with a Gaussian prior: $\mathcal{L}_\text{MAP} = -\log p(\mathcal{D} \mid \boldsymbol{\theta}) - \log p(\boldsymbol{\theta}) = \mathcal{L}_\text{MLE} + \lambda \lVert \boldsymbol{\theta} \rVert_2^2$.

---

## 2. Formal Foundations

### 2.1 Probability Spaces and Random Variables

**Definition (Probability Space).** A probability space is a triple $(\Omega, \mathcal{F}, P)$ where:

- $\Omega$ is the sample space (set of all outcomes)
- $\mathcal{F}$ is a $\sigma$-algebra over $\Omega$ (closed under complement and countable union)
- $P: \mathcal{F} \to [0,1]$ is a probability measure satisfying $P(\Omega)=1$ and countable additivity

**Definition (Random Variable).** A random variable $X: \Omega \to \mathcal{X}$ is a measurable function from the sample space to a measurable space $(\mathcal{X}, \mathcal{B})$. When $\mathcal{X} = \mathbb{R}$, the **distribution** of $X$ is characterised by its CDF $F_X(x) = P(X \leq x)$.

**Discrete** random variables: $P(X=x) = p(x)$, the probability mass function (PMF). $\sum_x p(x) = 1$.

**Continuous** random variables: characterised by a probability density function (PDF) $f_X(x)$ such that $P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$. Note $f_X(x) \geq 0$ but $f_X(x)$ can exceed 1 — it is a density, not a probability.

**Joint, marginal, conditional** distributions:
$$p(x,y) = p(y \mid x)\, p(x) = p(x \mid y)\, p(y)$$
$$p(x) = \int p(x,y)\, dy \quad \text{(marginalisation)}$$

**Independence**: $X \perp\!\!\!\perp Y \iff p(x,y) = p(x)\,p(y)$ for all $x, y$.

**Conditional independence**: $X \perp\!\!\!\perp Y \mid Z \iff p(x,y \mid z) = p(x \mid z)\,p(y \mid z)$.

**Standard distributions** (see notation guide §5.2):

| Distribution                              | PMF/PDF                                                         | Mean                                   | Variance   | Conjugate to                    |
| ----------------------------------------- | --------------------------------------------------------------- | -------------------------------------- | ---------- | ------------------------------- |
| $\operatorname{Bern}(p)$                  | $p^x(1-p)^{1-x}$                                                | $p$                                    | $p(1-p)$   | Beta prior                      |
| $\operatorname{Cat}(\mathbf{p})$          | $\prod_k p_k^{\mathbf{1}[x=k]}$                                 | $\mathbf{p}$                           | —          | Dirichlet prior                 |
| $\mathcal{N}(\mu,\sigma^2)$               | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$                                  | $\sigma^2$ | Normal prior (known $\sigma^2$) |
| $\operatorname{Poisson}(\lambda)$         | $\frac{\lambda^x e^{-\lambda}}{x!}$                             | $\lambda$                              | $\lambda$  | Gamma prior                     |
| $\operatorname{Dir}(\boldsymbol{\alpha})$ | $\frac{1}{B(\boldsymbol{\alpha})}\prod_k p_k^{\alpha_k-1}$      | $\frac{\boldsymbol{\alpha}}{\alpha_0}$ | —          | Conjugate to Categorical        |

### 2.2 Bayes' Theorem

**Theorem (Bayes, 1763).** For events $A, B$ with $P(B) > 0$:

$$P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}$$

In the statistical inference setting with parameters $\boldsymbol{\theta}$ and data $\mathcal{D}$:

$$\underbrace{p(\boldsymbol{\theta} \mid \mathcal{D})}_{\text{posterior}} = \frac{\underbrace{p(\mathcal{D} \mid \boldsymbol{\theta})}_{\text{likelihood}} \cdot \underbrace{p(\boldsymbol{\theta})}_{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

The **evidence** (also called marginal likelihood) is:
$$p(\mathcal{D}) = \int p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})\, d\boldsymbol{\theta}$$

This integral is typically intractable for complex models — motivating variational inference (§6) and MCMC (§8).

**Sequential updating.** Bayes' theorem is composable: after observing $\mathcal{D}_1$, the posterior $p(\boldsymbol{\theta} \mid \mathcal{D}_1)$ becomes the prior for $\mathcal{D}_2$:

$$p(\boldsymbol{\theta} \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \boldsymbol{\theta})\, p(\mathcal{D}_1 \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})$$

assuming conditional independence $\mathcal{D}_1 \perp\!\!\!\perp \mathcal{D}_2 \mid \boldsymbol{\theta}$ (i.i.d. data). This makes Bayesian learning naturally **online** — each batch updates the posterior, which becomes the prior for the next batch.

**For AI**: language model perplexity is proportional to $-\log p(\mathcal{D})$, the negative log evidence. Bayesian model comparison uses the evidence as a complexity-penalised measure of fit (the "Occam factor"). GPT perplexity on validation sets is computing exactly this quantity, averaged over tokens.

### 2.3 Exponential Family

The exponential family is the most important class of distributions in statistics. It unifies Gaussian, Bernoulli, Categorical, Poisson, Gamma, Beta, Dirichlet, and Wishart distributions under a single parameterisation that enables conjugate prior analysis, efficient sufficient statistics, and the information geometry of machine learning.

**Definition.** A distribution belongs to the exponential family if its density/mass function can be written as:

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x})\exp\!\left(\boldsymbol{\eta}^\top T(\mathbf{x}) - A(\boldsymbol{\eta})\right)$$

where:

- $\boldsymbol{\eta} \in \mathcal{H} \subseteq \mathbb{R}^k$ — **natural parameters** (also called canonical parameters)
- $T(\mathbf{x}): \mathcal{X} \to \mathbb{R}^k$ — **sufficient statistics** (captures all information about $\boldsymbol{\eta}$ in the data)
- $A(\boldsymbol{\eta}) = \log \int h(\mathbf{x}) \exp(\boldsymbol{\eta}^\top T(\mathbf{x}))\, d\mathbf{x}$ — **log-partition function** (normaliser)
- $h(\mathbf{x})$ — **base measure** (independent of $\boldsymbol{\eta}$)

**Key property (Moment generating).** The log-partition function generates cumulants:

$$\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \mathbb{E}_{p}[T(\mathbf{x})]$$
$$\nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \operatorname{Cov}_{p}[T(\mathbf{x})] \succeq 0$$

This means $A(\boldsymbol{\eta})$ is **convex** (its Hessian is a covariance matrix, hence PSD). The natural parameter space $\mathcal{H} = \{\boldsymbol{\eta} : A(\boldsymbol{\eta}) < \infty\}$ is a convex set.

**Example: Gaussian as exponential family.**

$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Rewrite: $-\frac{x^2}{2\sigma^2} + \frac{\mu}{\sigma^2} x - \frac{\mu^2}{2\sigma^2} - \frac{1}{2}\log(2\pi\sigma^2)$.

Identify: $\boldsymbol{\eta} = \begin{pmatrix}\mu/\sigma^2 \\ -1/(2\sigma^2)\end{pmatrix}$, $T(x) = \begin{pmatrix}x \\ x^2\end{pmatrix}$, $h(x) = \frac{1}{\sqrt{2\pi}}$, $A(\boldsymbol{\eta}) = -\frac{\eta_1^2}{4\eta_2} - \frac{1}{2}\log(-2\eta_2)$.

**Example: Categorical as exponential family.**

$$p(\mathbf{x} \mid \boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{x_k} = \exp\!\left(\sum_{k=1}^K x_k \log\pi_k\right)$$

Natural parameters: $\eta_k = \log\pi_k$ (up to normalisation). Sufficient statistic: $T(\mathbf{x}) = \mathbf{x}$ (one-hot vector). Log-partition: $A(\boldsymbol{\eta}) = \log\sum_k e^{\eta_k}$ — this is exactly the **log-sum-exp** function that appears in softmax!

**For AI**: the softmax output of a transformer is computing the mean parameter of a Categorical exponential family. Cross-entropy loss is negative log-likelihood of this family. The natural gradient (used in K-FAC optimisation) is the inverse Fisher information matrix — the inverse of $\nabla^2 A$, the covariance of sufficient statistics.

**Fisher information.** For exponential families, the Fisher information matrix equals the Hessian of $A$:

$$I(\boldsymbol{\eta}) = \operatorname{Cov}[T(\mathbf{x})] = \nabla^2 A(\boldsymbol{\eta})$$

This connects exponential family geometry to information geometry: the parameter space with metric $I$ is a Riemannian manifold. Natural gradient descent moves in this metric and is Fisher-invariant — unlike vanilla gradient descent, it is invariant to reparameterisation.

### 2.4 Conjugate Priors

**Definition.** A prior $p(\boldsymbol{\theta})$ is **conjugate** to a likelihood $p(\mathcal{D} \mid \boldsymbol{\theta})$ if the posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$ is in the same distributional family as the prior.

For exponential family likelihoods, conjugate priors always exist and take the form:

$$p(\boldsymbol{\eta} \mid \boldsymbol{\chi}, \nu) = h_c(\boldsymbol{\eta}) \exp\!\left(\boldsymbol{\eta}^\top \boldsymbol{\chi} - \nu A(\boldsymbol{\eta}) - \log Z_c(\boldsymbol{\chi}, \nu)\right)$$

After observing $n$ data points with sufficient statistics $\sum_{i=1}^n T(\mathbf{x}^{(i)})$, the posterior has:

$$\boldsymbol{\chi}_\text{post} = \boldsymbol{\chi} + \sum_{i=1}^n T(\mathbf{x}^{(i)}), \quad \nu_\text{post} = \nu + n$$

**Posterior updating = accumulating sufficient statistics.** This is why sufficient statistics are called "sufficient" — they capture everything the data says about $\boldsymbol{\eta}$.

**Standard conjugate pairs:**

| Likelihood                        | Prior                            | Posterior                                     |
| --------------------------------- | -------------------------------- | --------------------------------------------- |
| Bernoulli($p$)                    | Beta($\alpha, \beta$)            | Beta($\alpha + n_1$, $\beta + n_0$)           |
| Categorical($\boldsymbol{\pi}$)   | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \mathbf{n}$) |
| Gaussian($\mu$, known $\sigma^2$) | Gaussian($\mu_0, \sigma_0^2$)    | Gaussian (precision-weighted mean)            |
| Gaussian($\mu$, $\sigma^2$)       | Normal-Inverse-Gamma             | Normal-Inverse-Gamma                          |
| Poisson($\lambda$)                | Gamma($a$, $b$)                  | Gamma($a+\sum x_i$, $b+n$)                    |
| Multinomial($\boldsymbol{\pi}$)   | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha} + \mathbf{n}$) |

**Example: Gaussian posterior with known variance.**

Prior: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$. Likelihood: $x^{(i)} \mid \mu \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\mu, \sigma^2)$. After $n$ observations:

$$\mu \mid \mathcal{D} \sim \mathcal{N}\!\left(\frac{\frac{\mu_0}{\sigma_0^2} + \frac{\sum x^{(i)}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}},\; \left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}\right)$$

The posterior mean is a **precision-weighted average** of prior mean and sample mean. As $n \to \infty$, the posterior mean converges to $\bar{x}$ (frequentist MLE) and posterior variance converges to $\sigma^2/n$.

**For AI**: Bayesian optimisation for hyperparameter search uses GP priors conjugate to Gaussian likelihoods. Language model vocabulary distributions $p(w)$ are Dirichlet-Categorical models. Token smoothing (Laplace smoothing) is Bayesian estimation with a $\operatorname{Dir}(\mathbf{1})$ prior.

---

## 3. Graphical Models

### 3.1 Bayesian Networks

A **Bayesian network** (also directed graphical model) is a directed acyclic graph (DAG) $\mathcal{G} = (V, E)$ where each node $i \in V$ corresponds to a random variable $X_i$, and edges encode conditional independence structure.

**Factorisation theorem.** Any joint distribution that is Markov with respect to $\mathcal{G}$ factorises as:

$$p(\mathbf{x}) = \prod_{i=1}^d p(x_i \mid \mathbf{x}_{\mathrm{pa}(i)})$$

where $\mathrm{pa}(i)$ denotes the parents of node $i$ in $\mathcal{G}$.

**Example: Naive Bayes classifier.**

```text
       Y (class)
      / | \
    X₁  X₂  X₃  (features, conditionally independent given Y)
```

Factorisation: $p(Y, X_1, X_2, X_3) = p(Y)\, p(X_1 \mid Y)\, p(X_2 \mid Y)\, p(X_3 \mid Y)$.

This assumes features are conditionally independent given the class label — a strong but often effective assumption.

**Example: Hidden Markov Model as BN.**

$$p(Z_1, \ldots, Z_T, X_1, \ldots, X_T) = p(Z_1) \prod_{t=2}^T p(Z_t \mid Z_{t-1}) \prod_{t=1}^T p(X_t \mid Z_t)$$

Each latent state $Z_t$ has one parent ($Z_{t-1}$) and one observed child ($X_t$). The BN structure enforces the Markov property.

**Plate notation.** Repeated variables are denoted with a rectangle ("plate"). A plate with $N$ indicates $N$ i.i.d. repetitions. This compresses large BN diagrams for hierarchical models and latent variable models.

**For AI**: every generative model in deep learning has an implicit BN structure. VAEs: $Z \to X$ (latent causes observations). Diffusion models: $X_0 \leftarrow X_1 \leftarrow \cdots \leftarrow X_T$ (a chain BN). Transformers with a causal mask: $W_1 \to W_2 \to \cdots \to W_T$ (autoregressive BN). Understanding the DAG structure tells you the efficient factorisation for inference.

### 3.2 D-Separation

D-separation is a graphical criterion for reading conditional independence directly from a BN without computing probabilities. It was developed by Judea Pearl (1988) and is the foundation of causal inference.

**Three structural patterns:**

| Pattern      | Structure                      | $X \perp\!\!\!\perp Y \mid Z$?                |
| ------------ | ------------------------------ | --------------------------------------------- |
| **Chain**    | $X \to Z \to Y$                | Yes: conditioning on $Z$ blocks the path      |
| **Fork**     | $X \leftarrow Z \rightarrow Y$ | Yes: conditioning on common cause $Z$ blocks  |
| **Collider** | $X \rightarrow Z \leftarrow Y$ | No: conditioning on collider $Z$ _opens_ path |

**The Bayes Ball Algorithm.** To test whether $X \perp\!\!\!\perp Y \mid \mathbf{Z}$:

1. Mark all nodes in $\mathbf{Z}$ as observed
2. Send "balls" from $X$, following these rules:
   - **Chain/Fork**: ball passes through unobserved nodes, blocked at observed nodes
   - **Collider** ($\to Z \leftarrow$): ball is blocked at $Z$ unless $Z$ or a descendant of $Z$ is observed
3. If no ball reaches $Y$, then $X \perp\!\!\!\perp Y \mid \mathbf{Z}$

**Collider bias** (Berkson's paradox): conditioning on a collider creates spurious correlations between its parents. If $Z = X + Y$ (collider), knowing $Z$ makes $X$ and $Y$ negatively correlated — even if they are marginally independent. This appears in:

- **Selection bias**: conditioning on selected participants (collider) induces confounding
- **Transformer attention**: attending to a summary token can create spurious correlations between irrelevant tokens

**For AI**: the causal graph of a language model's training data contains many colliders (topics cause both words and sentence structures). Selecting texts that contain a keyword creates Berkson's bias in the training distribution. Understanding d-separation is essential for mechanistic interpretability — it tells you which circuits can and cannot influence each other given an intervention.

### 3.3 Markov Random Fields

A **Markov Random Field** (MRF) or undirected graphical model uses an undirected graph $\mathcal{G} = (V, E)$ where edges encode unconditional dependencies (no direction required).

**Gibbs distribution.** The joint distribution of an MRF factorises over maximal cliques $\mathcal{C}$ of $\mathcal{G}$:

$$p(\mathbf{x}) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(\mathbf{x}_c)$$

where $\psi_c(\mathbf{x}_c) \geq 0$ are **potential functions** (also called factors or compatibility functions), and $Z = \sum_\mathbf{x} \prod_c \psi_c(\mathbf{x}_c)$ is the **partition function**.

**Hammersley-Clifford Theorem.** $p > 0$ satisfies the Markov property for $\mathcal{G}$ if and only if it factorises as a Gibbs distribution over cliques of $\mathcal{G}$.

**Energy-based formulation.** Setting $\psi_c = \exp(-E_c(\mathbf{x}_c))$:

$$p(\mathbf{x}) = \frac{1}{Z} \exp\!\left(-\sum_{c \in \mathcal{C}} E_c(\mathbf{x}_c)\right) = \frac{e^{-E(\mathbf{x})}}{Z}$$

This is the **Boltzmann distribution**. The total energy $E(\mathbf{x}) = \sum_c E_c(\mathbf{x}_c)$ is a sum of local terms — tractable to evaluate even when $Z$ is intractable to compute.

**Ising model.** The canonical binary MRF: $x_i \in \{-1, +1\}$, grid graph:

$$p(\mathbf{x}) \propto \exp\!\left(J \sum_{(i,j) \in E} x_i x_j + h \sum_i x_i\right)$$

$J > 0$: ferromagnetic (neighbours tend to align). $h$: external field. The Ising model is used for image denoising, community detection, and as a prototype for energy-based models in DL.

**For AI**: Restricted Boltzmann Machines (RBMs), energy-based models (EBMs), and score-based models are all MRFs. The score function $\nabla_\mathbf{x} \log p(\mathbf{x}) = -\nabla_\mathbf{x} E(\mathbf{x})$ is the negative energy gradient — the fundamental object in diffusion models (§13).

### 3.4 Factor Graphs and Belief Propagation

A **factor graph** is a bipartite graph with two node types: variable nodes $x_i$ and factor nodes $f_j$. Each factor $f_j$ connects to a subset of variables $\partial f_j$ and represents a local function. The joint factorises as:

$$p(\mathbf{x}) = \frac{1}{Z} \prod_j f_j(\mathbf{x}_{\partial f_j})$$

Factor graphs unify BNs (each CPD becomes a factor) and MRFs (each clique potential becomes a factor) in one framework.

**Sum-product (belief propagation) algorithm.** Exact inference on tree-structured factor graphs via message passing:

- **Variable-to-factor message**: $\mu_{x \to f}(x) = \prod_{f' \in \partial x \setminus f} \mu_{f' \to x}(x)$
- **Factor-to-variable message**: $\mu_{f \to x}(x) = \sum_{\sim x} f(\mathbf{x}_{\partial f}) \prod_{x' \in \partial f \setminus x} \mu_{x' \to f}(x')$

After messages converge, the **belief** (marginal) at variable $x$ is:

$$b(x) \propto \prod_{f \in \partial x} \mu_{f \to x}(x)$$

**Complexity**: $O(N K^2)$ for a tree with $N$ variables and alphabet size $K$ — linear in $N$, not exponential.

**Max-sum algorithm.** Replace $\sum$ with $\max$ and $\times$ with $+$ (work in log domain): finds the MAP assignment (most probable explanation). This is the **Viterbi algorithm** for HMMs (§10.3) as a special case.

**For AI**: belief propagation is exact on trees; loopy BP (running the same algorithm on graphs with cycles) is approximate but empirically effective for LDPC codes, SAT solving, and image segmentation. Attention mechanisms in transformers can be interpreted as a differentiable, learned message-passing algorithm — each attention head passes messages between tokens along learned connectivity patterns.

---

## 4. Latent Variable Models

### 4.1 Gaussian Mixture Models

The Gaussian Mixture Model is the canonical latent variable model: observed data $\mathbf{x}$ is generated by first sampling a discrete cluster assignment $z \in \{1, \ldots, K\}$ (the latent variable), then sampling from the corresponding Gaussian component.

**Generative process:**
$$z \sim \operatorname{Cat}(\boldsymbol{\pi}), \qquad \mathbf{x} \mid z = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$$

**Marginal density** (marginalising out $z$):
$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k\, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k)$$

where $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \Sigma_k\}_{k=1}^K$ and $\sum_k \pi_k = 1$.

**Responsibility** (posterior over cluster assignment):
$$r_{ik} = p(z^{(i)} = k \mid \mathbf{x}^{(i)}, \boldsymbol{\theta}) = \frac{\pi_k\, \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_k, \Sigma_k)}{\sum_{j=1}^K \pi_j\, \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_j, \Sigma_j)}$$

The responsibilities are **soft cluster assignments**: $r_{ik} \in (0,1)$, $\sum_k r_{ik} = 1$.

**Connection to k-means**: k-means is the hard-assignment limit of GMM with isotropic components $\Sigma_k = \sigma^2 I$ as $\sigma^2 \to 0$. Responsibilities collapse to 0/1 indicators, and the M-step reduces to computing centroid means.

**For AI**: GMMs model the distribution of token embeddings in LLM hidden layers. Polysemanticity — a single neuron responding to multiple unrelated concepts — can be understood as the neuron's activation distribution being a GMM. Mixture of experts (MoE) layers in models like Mixtral and GPT-4 are discrete latent variable models: the router selects a subset of experts (latent $z$), then each expert applies a transformation.

### 4.2 Factor Analysis and Probabilistic PCA

**Factor Analysis (FA).** The linear generative model with a low-dimensional latent representation:

$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I_q)$$
$$\mathbf{x} = W \mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \Psi)$$

where $W \in \mathbb{R}^{d \times q}$ is the **loading matrix** ($q \ll d$), $\boldsymbol{\mu} \in \mathbb{R}^d$ is the mean, and $\Psi = \operatorname{diag}(\psi_1, \ldots, \psi_d)$ is diagonal (each dimension has its own noise variance).

**Marginal distribution**: $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, WW^\top + \Psi)$. The covariance has **low-rank plus diagonal** structure — FA assumes features share variance through the $q$ latent factors but have independent residual noise.

**Probabilistic PCA (PPCA).** Special case with isotropic noise $\Psi = \sigma^2 I_d$:

$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, WW^\top + \sigma^2 I)$$

As $\sigma^2 \to 0$, the ML estimate of $W$ spans the top-$q$ principal components. PPCA provides a principled probabilistic interpretation of PCA, with uncertainty quantification and a natural way to handle missing data.

**Posterior over latents** (both FA and PPCA):

$$p(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid M^{-1}W^\top \Psi^{-1}(\mathbf{x}-\boldsymbol{\mu}),\; M^{-1})$$

where $M = I + W^\top \Psi^{-1} W \in \mathbb{R}^{q \times q}$. This is the efficient $q \times q$ matrix to invert, not $d \times d$.

**For AI**: low-rank decomposition in LoRA is a variant of FA applied to weight matrices. A weight update $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ is exactly the FA loading structure. The latent dimension $r$ controls how many "directions" of adaptation are learned. In mechanistic interpretability, linear probing — fitting a linear model to residual stream activations — is equivalent to fitting FA to find the latent concept directions.

### 4.3 Why Latent Variables?

**Marginalisation adds expressiveness.** A single Gaussian has an ellipsoidal density. A mixture of $K$ Gaussians can approximate any density (universal approximator in the limit $K \to \infty$ — this is the Gaussian mixture density theorem). By marginalising out the discrete assignment $z$, a simple model (Gaussian) gains exponential expressiveness.

**Identifiability challenges.** Latent variable models often have non-identifiable parameters:

- GMMs: permuting cluster labels gives identical likelihoods ($K!$ symmetries)
- FA/PPCA: rotating $W$ by any orthogonal matrix $R$ gives identical marginal: $W' = WR$, since $WW^\top = WRR^\top W^\top$
- VAE decoder: the prior is $\mathcal{N}(\mathbf{0}, I)$ but any rotation of the latent space gives equal ELBO

Non-identifiability does not prevent learning useful representations, but it means **the latent coordinates have no canonical meaning** without additional constraints (sparsity, disentanglement objectives, etc.).

**Model evidence for model selection.** The marginal likelihood $p(\mathcal{D}) = \int p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})\, d\boldsymbol{\theta}$ automatically penalises complexity (Occam's razor): a model with too many parameters "spreads probability mass" over many configurations, reducing $p(\mathcal{D})$ even if it fits perfectly. This gives a principled way to choose the number of mixture components $K$ or latent dimensions $q$ without cross-validation.

---

## 5. Expectation-Maximization Algorithm

### 5.1 The EM Framework

The EM algorithm (Dempster, Laird & Rubin 1977) solves a fundamental problem: maximise the log-likelihood $\log p(\mathcal{D} \mid \boldsymbol{\theta})$ when the data has latent structure that makes direct optimisation intractable.

**The core identity.** For any distribution $q(Z)$ over latent variables $Z$:

$$\log p(\mathbf{x} \mid \boldsymbol{\theta}) = \underbrace{\mathbb{E}_{q}[\log p(\mathbf{x}, Z \mid \boldsymbol{\theta})] - \mathbb{E}_{q}[\log q(Z)]}_{\mathcal{L}(q, \boldsymbol{\theta}) \text{ (ELBO)}} + \underbrace{D_{\mathrm{KL}}(q \| p(Z \mid \mathbf{x}, \boldsymbol{\theta}))}_{\geq\, 0}$$

Since KL divergence is non-negative, $\mathcal{L}(q, \boldsymbol{\theta}) \leq \log p(\mathbf{x} \mid \boldsymbol{\theta})$ — a lower bound on the log-likelihood.

**E-step**: Fix $\boldsymbol{\theta}^{(t)}$, maximise $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$ over $q$. The maximum is achieved when:
$$q^*(Z) = p(Z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)})$$
At this $q^*$, the KL term is zero and $\mathcal{L} = \log p(\mathbf{x} \mid \boldsymbol{\theta}^{(t)})$. The bound is **tight**.

**M-step**: Fix $q^*(Z) = p(Z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)})$, maximise $\mathcal{L}(q^*, \boldsymbol{\theta})$ over $\boldsymbol{\theta}$:
$$\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{p(Z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)})}\!\left[\log p(\mathbf{x}, Z \mid \boldsymbol{\theta})\right]$$

This is maximising the **expected complete-data log-likelihood** $\mathbb{E}[Q(\boldsymbol{\theta})]$ — usually tractable because $\log p(\mathbf{x}, Z \mid \boldsymbol{\theta})$ decomposes nicely.

### 5.2 Convergence Proof

**Theorem.** EM monotonically increases the marginal log-likelihood: $\log p(\mathbf{x} \mid \boldsymbol{\theta}^{(t+1)}) \geq \log p(\mathbf{x} \mid \boldsymbol{\theta}^{(t)})$.

**Proof:**
$$\log p(\mathbf{x} \mid \boldsymbol{\theta}^{(t+1)}) \geq \mathcal{L}(q^*, \boldsymbol{\theta}^{(t+1)}) \quad \text{(lower bound property)}$$
$$\geq \mathcal{L}(q^*, \boldsymbol{\theta}^{(t)}) \quad \text{(M-step maximises over } \boldsymbol{\theta}\text{)}$$
$$= \log p(\mathbf{x} \mid \boldsymbol{\theta}^{(t)}) \quad \text{(E-step made bound tight)}$$

EM converges to a **local maximum** of the log-likelihood. It is not guaranteed to find the global maximum — initialisation matters greatly (multiple restarts are standard practice).

**Convergence rate**: locally linear (first-order), with rate determined by the fraction of "missing information": $r = I_\text{miss} / I_\text{obs}$. The closer $r$ is to 1 (most information is missing), the slower EM converges. Acceleration methods (SQUAREM, quasi-Newton EM) address this.

### 5.3 EM for Gaussian Mixture Models

**Complete-data log-likelihood** (treating $z^{(i)}$ as known):
$$\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) = \sum_{i=1}^n \sum_{k=1}^K z_{ik}\!\left[\log\pi_k + \log\mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_k, \Sigma_k)\right]$$

**E-step** — compute responsibilities:
$$r_{ik} = \mathbb{E}[z_{ik} \mid \mathbf{x}^{(i)}, \boldsymbol{\theta}^{(t)}] = \frac{\pi_k^{(t)}\, \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_k^{(t)}, \Sigma_k^{(t)})}{\sum_j \pi_j^{(t)}\, \mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_j^{(t)}, \Sigma_j^{(t)})}$$

Define effective cluster counts: $N_k = \sum_{i=1}^n r_{ik}$.

**M-step** — update parameters:
$$\boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_i r_{ik} \mathbf{x}^{(i)}}{N_k}$$
$$\Sigma_k^{(t+1)} = \frac{\sum_i r_{ik} (\mathbf{x}^{(i)} - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}^{(i)} - \boldsymbol{\mu}_k^{(t+1)})^\top}{N_k}$$
$$\pi_k^{(t+1)} = \frac{N_k}{n}$$

Each update is a **weighted mean/covariance** with weights given by the responsibilities. The mixture weights are just the average responsibility.

**Degenerate solutions**: a component collapses to a single data point ($\boldsymbol{\mu}_k = \mathbf{x}^{(i)}$, $\Sigma_k \to 0$), yielding $\log\mathcal{N} \to +\infty$. Regularisation (add $\epsilon I$ to $\Sigma_k$) or a prior on $\Sigma_k$ (MAP-EM) prevents this.

### 5.4 Generalised EM and Hard-EM

**Generalised EM (GEM)**: the M-step need not maximise exactly — any update that increases $\mathcal{L}$ is valid. This allows partial M-steps (e.g., one gradient step instead of solving the M-step exactly) and is the basis of online/stochastic EM.

**Hard-EM**: replace soft responsibilities $r_{ik} \in (0,1)$ with hard assignments $\hat{z}_{ik} = \mathbf{1}[k = \arg\max_j r_{ij}]$. This is the limit of EM with vanishing temperature. **k-means is hard EM for isotropic GMM** with $\Sigma_k = I$ for all $k$.

**For AI**: the VQVAE training algorithm is hard EM — the commitment loss forces the encoder output to commit to the nearest codebook vector (E-step), then codebook vectors are updated as cluster centroids (M-step). EM underlies Baum-Welch for HMMs (§10.4), the forward algorithm in CTC training, and variational EM (which replaces the exact E-step with VI when the posterior is intractable).

---

## 6. Variational Inference

### 6.1 The ELBO

Variational inference (VI) transforms intractable posterior computation $p(\mathbf{z} \mid \mathbf{x})$ into an optimisation problem: find the distribution $q(\mathbf{z})$ in a tractable family $\mathcal{Q}$ that is closest (in KL divergence) to the true posterior.

**Derivation of the ELBO.** Starting from the log-evidence:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z})\, d\mathbf{z} = \log \int q(\mathbf{z}) \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\, d\mathbf{z}$$

By Jensen's inequality (log is concave):

$$\geq \int q(\mathbf{z}) \log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\, d\mathbf{z} = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]$$

This is the **Evidence Lower Bound (ELBO)**:
$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{x}, \mathbf{z})] + H(q)$$

where $H(q) = -\mathbb{E}_q[\log q(\mathbf{z})]$ is the entropy of $q$.

**Alternative decomposition** (reconstruction minus KL):
$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{x} \mid \mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z}))$$

This form is essential for VAEs (§7): the first term encourages $q$ to explain the data, the second term keeps $q$ close to the prior.

**Gap between ELBO and log-evidence:**
$$\log p(\mathbf{x}) - \mathcal{L}(q) = D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z} \mid \mathbf{x})) \geq 0$$

Maximising the ELBO over $q$ is equivalent to minimising $D_{\mathrm{KL}}(q \| p(\cdot \mid \mathbf{x}))$. When the ELBO is tight ($q = p(\mathbf{z} \mid \mathbf{x})$), we recover exact Bayesian inference — this is the E-step.

**Forward vs. reverse KL**: VI minimises $D_{\mathrm{KL}}(q \| p)$ (reverse KL), which is **zero-forcing** — $q$ avoids putting mass where $p$ is zero, leading to underestimation of the posterior variance (mode-seeking). Minimising $D_{\mathrm{KL}}(p \| q)$ (forward KL, used in expectation propagation) is **mass-covering** — $q$ must cover all modes of $p$.

### 6.2 Mean-Field Approximation and CAVI

**Mean-field assumption**: restrict $q$ to fully factored distributions:
$$q(\mathbf{z}) = \prod_{j=1}^m q_j(z_j)$$

No conditional dependencies between latent variables in $q$ — each factor $q_j$ is a marginal distribution.

**Coordinate Ascent VI (CAVI).** Maximise the ELBO by iteratively updating each $q_j$ while holding others fixed:

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}\!\left[\log p(\mathbf{x}, \mathbf{z})\right] + \mathrm{const}$$

where $\mathbb{E}_{q_{-j}}$ averages over all latent variables except $z_j$.

**Key result**: the optimal $q_j^*$ is always in the **conjugate family** of the conditional $p(z_j \mid \mathbf{z}_{-j}, \mathbf{x})$. For exponential family models with conjugate priors, CAVI has **closed-form updates** — no gradient computation required.

**CAVI for GMM:**

- Update $q(z_i)$: $r_{ik} \propto \exp(\mathbb{E}[\log\pi_k] + \mathbb{E}[\log\mathcal{N}(\mathbf{x}^{(i)} \mid \boldsymbol{\mu}_k, \Sigma_k)])$
- Update $q(\boldsymbol{\mu}_k)$: Gaussian with mean $= \frac{N_k \bar{\mathbf{x}}_k}{\lambda_0^{-1} + N_k}$ (precision-weighted posterior)

CAVI is **guaranteed to converge** to a local optimum of the ELBO (ascending steps on a bounded function). It may converge to different optima depending on initialisation.

### 6.3 Black-Box Variational Inference

When the model is not conjugate, CAVI's closed-form updates do not apply. Black-box VI (BBVI) estimates the ELBO gradient using Monte Carlo.

**Score function estimator** (REINFORCE / log-derivative trick):

$$\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{q_\phi}[f(\mathbf{z}) \nabla_\phi \log q_\phi(\mathbf{z})]$$

Estimate with samples $\mathbf{z}^{(s)} \sim q_\phi$: $\nabla_\phi \mathcal{L} \approx \frac{1}{S}\sum_s f(\mathbf{z}^{(s)}) \nabla_\phi \log q_\phi(\mathbf{z}^{(s)})$.

**Problem**: this estimator has very high variance — impractical without variance reduction. Control variates (baselines) reduce variance: subtract a baseline $b$ (independent of $\mathbf{z}$) from $f$ without changing the mean.

**Reparameterisation estimator** (when $\mathbf{z} = g_\phi(\boldsymbol{\epsilon})$, $\boldsymbol{\epsilon} \sim p_0$):

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{p_0}\!\left[\nabla_\phi f(g_\phi(\boldsymbol{\epsilon}))\right] \approx \frac{1}{S}\sum_s \nabla_\phi f(g_\phi(\boldsymbol{\epsilon}^{(s)}))$$

**Much lower variance** than score function estimator — gradients flow directly through $g_\phi$. This is the key innovation in VAEs (§7.2).

### 6.4 Importance-Weighted Bounds

**IWAE (Importance-Weighted Autoencoder, Burda et al. 2015).** Draw $K$ samples from $q_\phi$:

$$\mathcal{L}_K = \mathbb{E}_{\mathbf{z}^{1:K} \sim q_\phi}\!\left[\log \frac{1}{K}\sum_{k=1}^K \frac{p(\mathbf{x}, \mathbf{z}^k)}{q_\phi(\mathbf{z}^k \mid \mathbf{x})}\right]$$

**Bound hierarchy**: $\log p(\mathbf{x}) \geq \mathcal{L}_{K+1} \geq \mathcal{L}_K \geq \mathcal{L}_1 = \mathcal{L}_\text{ELBO}$.

As $K \to \infty$, $\mathcal{L}_K \to \log p(\mathbf{x})$ (by law of large numbers for importance sampling). More samples gives a tighter bound.

**Tradeoff**: tighter bound does not always give better $q_\phi$ — the gradient of $\mathcal{L}_K$ can be smaller than $\nabla\mathcal{L}_1$ (signal dilution from multiple samples), leading to worse inference networks despite tighter bounds.

**For AI**: the IWAE bound is used to evaluate VAE quality. It connects to **self-normalised importance sampling** — a core technique in particle filters, sequential Monte Carlo, and RLHF reward modelling (where $q$ is the LLM policy and $p$ is the ideal target distribution).

---

## 7. Variational Autoencoders

### 7.1 Architecture and Generative Model

The VAE (Kingma & Welling 2013, ICLR 2014) is an amortised variational inference algorithm implemented as a neural network. It simultaneously learns:

1. A generative model $p_\theta(\mathbf{x} \mid \mathbf{z})$ (decoder)
2. An inference network $q_\phi(\mathbf{z} \mid \mathbf{x})$ (encoder / recognition network)

**Generative model** (prior + likelihood):
$$\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I), \qquad \mathbf{x} \mid \mathbf{z} \sim p_\theta(\mathbf{x} \mid \mathbf{z})$$

For continuous data: $p_\theta(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 I)$ where $\boldsymbol{\mu}_\theta$ is a neural network. For binary data: Bernoulli with logit $f_\theta(\mathbf{z})$.

**Inference model** (approximate posterior):
$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$$

The encoder network outputs $\boldsymbol{\mu}_\phi$ and $\log\boldsymbol{\sigma}^2_\phi$ (the latter parameterised to ensure positivity).

**Training objective** (ELBO per datapoint):
$$\mathcal{L}(\phi, \theta; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))}_{\text{regularisation}}$$

For Gaussian $q$ and Gaussian prior, the KL has a closed form:
$$D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0}, I)) = \frac{1}{2}\sum_j\!\left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

**Amortisation**: instead of optimising a separate $q$ per datapoint (as in standard VI), the VAE shares encoder parameters $\phi$ across all datapoints — the encoder _amortises_ the cost of variational inference. This makes training $O(n)$ instead of $O(n^2)$.

### 7.2 Reparameterisation Trick

**Problem**: the ELBO requires $\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]$. The expectation is over $q_\phi$ which depends on $\phi$ — standard Monte Carlo cannot backprop through the sampling operation.

**Reparameterisation**: express $\mathbf{z}$ as a deterministic function of $\phi$ and a base noise $\boldsymbol{\epsilon}$:
$$\mathbf{z} = g_\phi(\boldsymbol{\epsilon}, \mathbf{x}) = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

Now the expectation is over $\boldsymbol{\epsilon}$ (independent of $\phi$):
$$\mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},I)}[f(g_\phi(\boldsymbol{\epsilon}, \mathbf{x}))]$$

Gradients flow through $g_\phi$ via standard backpropagation. The Monte Carlo estimator:
$$\nabla_\phi \mathcal{L} \approx \frac{1}{S}\sum_{s=1}^S \nabla_\phi \log p_\theta(\mathbf{x} \mid g_\phi(\boldsymbol{\epsilon}^{(s)}, \mathbf{x}))$$

has **dramatically lower variance** than the score function estimator because the gradient of a smooth function of $\boldsymbol{\epsilon}$ concentrates much faster than the product of a function value and a score.

**Generalisation**: reparameterisation works for any distribution with a location-scale transform. For non-Gaussian distributions (Gamma, Dirichlet), the **implicit reparameterisation trick** (Figurnov et al. 2018) differentiates through the CDF $F^{-1}(\boldsymbol{\epsilon})$ where $\boldsymbol{\epsilon} \sim \mathcal{U}[0,1]$.

### 7.3 VAE Training Dynamics

**Posterior collapse.** A pathological mode where $q_\phi(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z})$ — the encoder ignores $\mathbf{x}$ and the decoder ignores $\mathbf{z}$. The model degenerates to a non-conditional decoder $p_\theta(\mathbf{x})$. Causes:

- Decoder too powerful (e.g., autoregressive decoder can model $p(\mathbf{x})$ without $\mathbf{z}$)
- KL term vanishes early in training ("KL collapse")

**Fixes:**

- **$\beta$-VAE** (Higgins et al. 2017): weight the KL term by $\beta > 1$, forcing the latent code to carry information
- **KL annealing**: start with $\beta=0$, linearly increase to $\beta=1$ during training
- **Free bits** (Kingma et al. 2016): allow each latent dimension a minimum KL budget $\lambda$ before penalising

**$\beta$-VAE and disentanglement.** Higher $\beta$ encourages more independent latent factors (lower mutual information between dimensions). This promotes **disentanglement**: each latent dimension encodes a single interpretable generative factor (pose, lighting, shape). There is a fundamental tradeoff: higher $\beta$ → more disentanglement but worse reconstruction.

**Hierarchical VAEs.** Models like NVAE (Vahdat & Kautz 2020) and VDVAE (Child 2021) stack latent variables at multiple scales:
$$p(\mathbf{z}_1, \ldots, \mathbf{z}_L, \mathbf{x}) = p(\mathbf{x} \mid \mathbf{z}_1) \prod_{l=1}^L p(\mathbf{z}_l \mid \mathbf{z}_{l+1:L})$$

Top-level latents capture global structure; lower-level latents capture fine-grained details. These achieve near-diffusion-quality generation while maintaining interpretable latents.

### 7.4 Latent Space Geometry

**Aggregated posterior mismatch.** The prior is $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$, but the aggregated posterior $q(\mathbf{z}) = \frac{1}{n}\sum_i q_\phi(\mathbf{z} \mid \mathbf{x}^{(i)})$ may not match it. Sampling $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ at test time may decode to unrealistic samples if the prior is not a good approximation of the aggregated posterior.

**For AI**: VQ-VAE (Van den Oord et al. 2017) replaces the continuous latent with a discrete codebook. The encoder maps to the nearest codebook vector (hard assignment), and the decoder reconstructs from that code. DALL-E 1 and 2 use VQ-VAE to tokenise images into discrete codes, then train a transformer over image codes (analogous to language modelling over text tokens).

---

## 8. Markov Chain Monte Carlo

### 8.1 Markov Chains and Stationarity

A **Markov chain** on state space $\mathcal{X}$ is a sequence of random variables $X_0, X_1, X_2, \ldots$ where each state depends only on the previous:
$$p(X_{t+1} = x' \mid X_0, \ldots, X_t) = T(x' \mid X_t)$$

The function $T(x' \mid x) \geq 0$ (with $\int T(x' \mid x)\, dx' = 1$) is the **transition kernel**.

**Stationary distribution** $\pi$ satisfies:
$$\pi(x') = \int \pi(x)\, T(x' \mid x)\, dx$$

The chain "mixes" to $\pi$ regardless of starting state (under ergodicity conditions).

**Detailed balance** (sufficient condition for stationarity):
$$\pi(x)\, T(x' \mid x) = \pi(x')\, T(x \mid x')$$

If a kernel satisfies detailed balance with respect to $\pi$, then $\pi$ is the stationary distribution.

**MCMC goal**: construct a Markov chain whose stationary distribution is the target $\pi = p(\mathbf{z} \mid \mathbf{x})$. Then run the chain long enough to collect approximately-i.i.d. samples from $p$.

**Mixing time**: the number of steps until the chain distribution $\lVert T^t(x_0, \cdot) - \pi \rVert_\text{TV} < \epsilon$. Poor mixing (slow convergence) is the central practical challenge in MCMC.

### 8.2 Metropolis-Hastings and Gibbs Sampling

**Metropolis-Hastings (MH) Algorithm** (Metropolis et al. 1953; Hastings 1970):

At state $\mathbf{x}$:

1. Propose $\mathbf{x}' \sim q(\mathbf{x}' \mid \mathbf{x})$ (proposal distribution)
2. Compute acceptance ratio: $\alpha = \min\!\left(1,\; \frac{p(\mathbf{x}')\, q(\mathbf{x} \mid \mathbf{x}')}{p(\mathbf{x})\, q(\mathbf{x}' \mid \mathbf{x})}\right)$
3. Accept with probability $\alpha$: $\mathbf{x}_{t+1} = \mathbf{x}'$; else $\mathbf{x}_{t+1} = \mathbf{x}$

**Key properties:**

- Only requires $p(\mathbf{x})$ up to normalisation — no need to compute $Z$
- Detailed balance is satisfied by construction: the accept/reject step corrects for the proposal asymmetry
- **Random-walk MH**: $q(\mathbf{x}' \mid \mathbf{x}) = \mathcal{N}(\mathbf{x}, \sigma^2 I)$. Step size $\sigma$ tradeoff: too small → high acceptance but slow exploration; too large → low acceptance, mostly stuck
- Optimal acceptance rate for RW-MH in $d$ dimensions: $\approx 23.4\%$ (Roberts, Gelman & Gilks 1997)

**Gibbs sampling**: special case of MH with acceptance ratio = 1. Sample each variable from its full conditional:
$$x_j^{(t+1)} \sim p(x_j \mid \mathbf{x}_{-j}^{(t)})$$

Requires tractable full conditionals — available for conjugate models and many graphical models. Gibbs is efficient when conditionals are cheap but mixing can be slow in high correlation.

### 8.3 Hamiltonian Monte Carlo

**Intuition**: augment the state $\mathbf{x}$ with auxiliary momentum $\mathbf{p} \sim \mathcal{N}(\mathbf{0}, M)$ and simulate Hamiltonian dynamics to make large, high-quality proposals.

**Hamiltonian function**:
$$\mathcal{H}(\mathbf{x}, \mathbf{p}) = U(\mathbf{x}) + K(\mathbf{p}) = -\log p(\mathbf{x}) + \frac{1}{2}\mathbf{p}^\top M^{-1}\mathbf{p}$$

$U(\mathbf{x})$ is the **potential energy** (negative log-density), $K(\mathbf{p})$ is **kinetic energy**.

**Hamiltonian dynamics** (continuous time):
$$\dot{\mathbf{x}} = \frac{\partial \mathcal{H}}{\partial \mathbf{p}} = M^{-1}\mathbf{p}, \qquad \dot{\mathbf{p}} = -\frac{\partial \mathcal{H}}{\partial \mathbf{x}} = \nabla_\mathbf{x} \log p(\mathbf{x})$$

These preserve the Hamiltonian ($d\mathcal{H}/dt = 0$) and volume (Liouville's theorem).

**Leapfrog integrator** (numerically stable, time-reversible, volume-preserving):
$$\tilde{\mathbf{p}}_{t+\epsilon/2} = \mathbf{p}_t + \frac{\epsilon}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_t)$$
$$\mathbf{x}_{t+\epsilon} = \mathbf{x}_t + \epsilon M^{-1} \tilde{\mathbf{p}}_{t+\epsilon/2}$$
$$\mathbf{p}_{t+\epsilon} = \tilde{\mathbf{p}}_{t+\epsilon/2} + \frac{\epsilon}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t+\epsilon})$$

After $L$ leapfrog steps, accept the proposal $(\mathbf{x}', \mathbf{p}')$ with MH ratio (corrects for numerical error):
$$\alpha = \min\!\left(1, \exp(-\mathcal{H}(\mathbf{x}', \mathbf{p}') + \mathcal{H}(\mathbf{x}, \mathbf{p}))\right)$$

**Advantages of HMC over RW-MH**:

- Proposals are far from current position (long trajectories)
- High acceptance rate (typically >80%)
- Scales well with dimension (acceptance rate degrades as $O(d^{-1/4})$ vs. $O(d^{-1})$ for RW-MH)

**NUTS (No-U-Turn Sampler)**: adaptive choice of trajectory length — doubles the leapfrog steps until the trajectory starts turning back. Eliminates the step-size tuning problem. NUTS is the default sampler in Stan and PyMC.

### 8.4 Convergence Diagnostics

**$\hat{R}$ (R-hat, Gelman-Rubin statistic)**: run $M$ chains in parallel. $\hat{R} \approx 1$ indicates between-chain variance ≈ within-chain variance — chains have mixed. $\hat{R} > 1.01$ indicates poor mixing.

**Effective sample size (ESS)**: accounts for autocorrelation between successive samples:
$$\text{ESS} = \frac{n}{1 + 2\sum_{k=1}^\infty \rho_k}$$

where $\rho_k$ is the autocorrelation at lag $k$. HMC typically achieves much higher ESS per gradient evaluation than RW-MH.

**For AI**: HMC is used in Bayesian deep learning (weight space sampling), Langevin dynamics (a continuous-time limit of MCMC) underlies score-based diffusion models (§13), and stochastic gradient Langevin dynamics (SGLD, Welling & Teh 2011) is a scalable MCMC method that uses mini-batches — bridging MCMC and deep learning optimisation.

---

## 9. Gaussian Processes

### 9.1 GP as Distribution over Functions

**Definition.** A Gaussian process is a collection of random variables, any finite subset of which has a joint Gaussian distribution. It defines a distribution over functions $f: \mathcal{X} \to \mathbb{R}$.

$$f(\cdot) \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$$

where:

- $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$ — mean function (often taken as zero for simplicity)
- $k(\mathbf{x}, \mathbf{x}') = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}') - m(\mathbf{x}'))]$ — **kernel** or covariance function

**Finite marginals**: for any finite set of inputs $\mathbf{X} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$, the function values $\mathbf{f} = [f(\mathbf{x}^{(1)}), \ldots, f(\mathbf{x}^{(n)})]^\top$ are jointly Gaussian:
$$\mathbf{f} \sim \mathcal{N}(\mathbf{m}, K)$$

where $m_i = m(\mathbf{x}^{(i)})$ and $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$.

**Kernel requirements**: $k$ must be a **positive semidefinite** function (also called Mercer kernel). Equivalently, for any finite input set, the Gram matrix $K$ must be PSD. This ensures that the covariance matrix of any finite marginal is valid.

**For AI**: Neural networks at infinite width converge to GPs (Neal 1996, Matthews et al. 2018). The NTK (Neural Tangent Kernel) describes the covariance of the function space explored by a wide neural network. GPs are the theoretical lens through which kernel methods and infinite-width networks are unified.

### 9.2 GP Regression

**Noisy observations**: $y^{(i)} = f(\mathbf{x}^{(i)}) + \epsilon^{(i)}$, $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma_n^2)$ (i.i.d. noise).

**Prior**: $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, K_{XX})$ (zero mean).

**Likelihood**: $\mathbf{y} \mid \mathbf{f} \sim \mathcal{N}(\mathbf{f}, \sigma_n^2 I)$, hence $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K_{XX} + \sigma_n^2 I)$.

**Posterior predictive distribution** at test points $X^*$:

$$\mathbf{f}^* \mid \mathbf{y}, X, X^* \sim \mathcal{N}(\bar{\mathbf{f}}^*, \operatorname{Cov}[\mathbf{f}^*])$$

$$\bar{\mathbf{f}}^* = K_{X^*X}(K_{XX} + \sigma_n^2 I)^{-1}\mathbf{y}$$

$$\operatorname{Cov}[\mathbf{f}^*] = K_{X^*X^*} - K_{X^*X}(K_{XX} + \sigma_n^2 I)^{-1}K_{XX^*}$$

This is closed-form Bayesian inference! No approximations needed. The predictive mean is a weighted sum of kernel evaluations with training data — a **kernel regression** with automatic weighting.

**Computational cost**: $O(n^3)$ for the Cholesky decomposition of $(K_{XX} + \sigma_n^2 I)$. This is the fundamental bottleneck — GPs are exact only for $n \lesssim 10{,}000$.

**Hyperparameter optimisation** via marginal likelihood (type-II ML / empirical Bayes):
$$\log p(\mathbf{y} \mid X, \boldsymbol{\psi}) = -\frac{1}{2}\mathbf{y}^\top (K_{\boldsymbol{\psi}} + \sigma_n^2 I)^{-1}\mathbf{y} - \frac{1}{2}\log\det(K_{\boldsymbol{\psi}} + \sigma_n^2 I) - \frac{n}{2}\log 2\pi$$

Maximise over kernel hyperparameters $\boldsymbol{\psi}$ using gradient descent. The first term rewards data fit; the second term (log-determinant) penalises complexity — automatic Occam's razor.

### 9.3 Kernel Design

The kernel encodes prior beliefs about function smoothness, periodicity, and scale.

**Common kernels:**

| Kernel     | Formula $k(\mathbf{x}, \mathbf{x}')$                                               | Smoothness          | Notes                                                   |
| ---------- | ---------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------- |
| RBF / SE   | $\exp(-\lVert \mathbf{x}-\mathbf{x}' \rVert^2 / 2\ell^2)$                          | $C^\infty$          | Infinitely differentiable; too smooth for many tasks    |
| Matérn 3/2 | $(1 + \frac{\sqrt{3}r}{\ell})\exp(-\frac{\sqrt{3}r}{\ell})$                        | $C^1$               | Once differentiable; preferred for physical processes   |
| Matérn 5/2 | $(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2})\exp(-\frac{\sqrt{5}r}{\ell})$ | $C^2$               | Twice differentiable; standard in Bayesian optimisation |
| Linear     | $\sigma_b^2 + \sigma_v^2 (\mathbf{x}-c)^\top(\mathbf{x}'-c)$                       | $C^\infty$ (linear) | Equivalent to Bayesian linear regression                |
| Periodic   | $\exp(-\frac{2\sin^2(\pi\lVert\mathbf{x}-\mathbf{x}'\rVert/p)}{\ell^2})$           | $C^\infty$          | Models periodic functions                               |

**Kernel composition rules:**

- Sum: $k_1 + k_2$ is a valid kernel (models additive structure)
- Product: $k_1 \cdot k_2$ is a valid kernel (models interaction effects)
- Convolution: $(k_1 \star k_2)(\mathbf{x}) = \int k_1(\mathbf{x}-\mathbf{t})\, k_2(\mathbf{t})\, d\mathbf{t}$ (diffused kernels)

**RKHS interpretation.** Every PSD kernel $k$ defines a Reproducing Kernel Hilbert Space $\mathcal{H}_k$. The GP prior is a distribution over $\mathcal{H}_k$. GP regression corresponds to finding the minimum-norm element of $\mathcal{H}_k$ that fits the data — exactly kernel ridge regression.

### 9.4 Scalable GPs

For $n \gtrsim 10{,}000$, exact GPs are computationally prohibitive. Sparse approximations introduce $m \ll n$ inducing points.

**Sparse GP (Titsias 2009, FITC).** Augment the model with inducing variables $\mathbf{u} = f(\mathbf{Z})$ at locations $\mathbf{Z} = \{\mathbf{z}^{(j)}\}_{j=1}^m$:

$$p(\mathbf{f} \mid \mathbf{u}) \approx \mathcal{N}(K_{XZ}K_{ZZ}^{-1}\mathbf{u},\; \Lambda)$$

where $\Lambda$ is a diagonal correction. Inference involves $m \times m$ matrices: $O(nm^2)$ cost.

Optimise $\mathbf{Z}$ jointly with kernel hyperparameters by maximising the ELBO of the sparse model.

**Deep kernels (Wilson et al. 2016)**: $k(\mathbf{x}, \mathbf{x}') = k_\text{base}(f_\theta(\mathbf{x}), f_\theta(\mathbf{x}'))$ where $f_\theta$ is a neural network. Combines GP uncertainty quantification with deep feature learning. Used in neural architecture search and Bayesian deep learning.

**For AI**: Bayesian optimisation (BO) — the standard approach for neural architecture search, hyperparameter tuning (used in AutoML, GPT training runs) — uses GPs to model the objective function. BO acquires training points by maximising an acquisition function (upper confidence bound, expected improvement) that balances exploration and exploitation. Google Brain and DeepMind use BO extensively in production for LLM hyperparameter sweeps.

---

## 10. Hidden Markov Models

### 10.1 HMM Structure

A Hidden Markov Model (HMM) models a sequence of observations $\mathbf{x}_{1:T} = (x_1, \ldots, x_T)$ generated by an underlying hidden Markov chain $\mathbf{z}_{1:T} = (z_1, \ldots, z_T)$ with states $z_t \in \{1, \ldots, K\}$.

**Generative model (three components):**

1. **Initial distribution**: $\boldsymbol{\pi}_0$ where $\pi_{0,k} = p(z_1 = k)$
2. **Transition matrix**: $A$ where $A_{jk} = p(z_{t+1} = k \mid z_t = j)$ (row-stochastic)
3. **Emission distribution**: $B_k(x) = p(x_t = x \mid z_t = k)$

**Joint distribution:**
$$p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) = p(z_1) \prod_{t=2}^T p(z_t \mid z_{t-1}) \prod_{t=1}^T p(x_t \mid z_t)$$

**Parameter set**: $\boldsymbol{\theta} = \{\boldsymbol{\pi}_0, A, B\}$.

**Three inference problems:**

1. **Filtering**: $p(z_t \mid x_{1:t})$ — online state estimation
2. **Smoothing**: $p(z_t \mid x_{1:T})$ — offline state estimation using full sequence
3. **Decoding**: $\arg\max_{z_{1:T}} p(z_{1:T} \mid x_{1:T})$ — most likely state sequence

### 10.2 Forward-Backward Algorithm

**Forward variable** (filtering distribution up to time $t$):
$$\alpha_t(k) = p(x_1, \ldots, x_t, z_t = k)$$

**Recursion** ($O(K^2)$ per step):
$$\alpha_1(k) = \pi_{0,k}\, B_k(x_1)$$
$$\alpha_t(k) = B_k(x_t) \sum_{j=1}^K \alpha_{t-1}(j)\, A_{jk}$$

**Marginal likelihood** (sum over states):
$$p(\mathbf{x}_{1:T}) = \sum_{k=1}^K \alpha_T(k)$$

**Backward variable** (probability of future observations given state):
$$\beta_t(k) = p(x_{t+1}, \ldots, x_T \mid z_t = k)$$

$$\beta_T(k) = 1$$
$$\beta_t(k) = \sum_{j=1}^K A_{kj}\, B_j(x_{t+1})\, \beta_{t+1}(j)$$

**Posterior (smoothed) marginals** (combine forward and backward):
$$\gamma_t(k) = p(z_t = k \mid \mathbf{x}_{1:T}) = \frac{\alpha_t(k)\, \beta_t(k)}{\sum_j \alpha_t(j)\, \beta_t(j)}$$

**Pairwise posteriors** (used in Baum-Welch):
$$\xi_t(j, k) = p(z_t = j, z_{t+1} = k \mid \mathbf{x}_{1:T}) \propto \alpha_t(j)\, A_{jk}\, B_k(x_{t+1})\, \beta_{t+1}(k)$$

**Total complexity**: $O(TK^2)$ — linear in sequence length $T$.

**For AI**: the forward-backward algorithm is exact inference in a BN with chain structure. CTC (Connectionist Temporal Classification) — used in speech recognition and OCR — is the forward algorithm for a special HMM where repeated labels are collapsed. Transformers with autoregressive decoding compute the forward variable at each step via the attention mechanism.

### 10.3 Viterbi Algorithm

The Viterbi algorithm finds the **MAP state sequence** — the single most probable hidden sequence:

$$\mathbf{z}^* = \arg\max_{z_{1:T}} p(\mathbf{z}_{1:T} \mid \mathbf{x}_{1:T})$$

**Viterbi variable** (best path to state $k$ at time $t$):
$$\delta_t(k) = \max_{z_{1:t-1}} p(z_1, \ldots, z_{t-1}, z_t = k, x_1, \ldots, x_t)$$

**Recursion:**
$$\delta_1(k) = \pi_{0,k}\, B_k(x_1)$$
$$\delta_t(k) = B_k(x_t) \max_{j} \left[\delta_{t-1}(j)\, A_{jk}\right]$$

Store the argmax for traceback:
$$\psi_t(k) = \arg\max_j \left[\delta_{t-1}(j)\, A_{jk}\right]$$

**Traceback** (backtrack from $z_T^* = \arg\max_k \delta_T(k)$):
$$z_t^* = \psi_{t+1}(z_{t+1}^*)$$

Work in **log-space** to avoid underflow: replace products with sums, and $\max$ instead of $\sum$.

**Key insight**: Viterbi is the max-product version of forward-backward; forward-backward is the sum-product version of Viterbi. The relationship:

| Algorithm | Operation      | Computes                                       |
| --------- | -------------- | ---------------------------------------------- |
| Forward   | $\sum$-product | $p(\mathbf{x}_{1:T})$, marginals $\gamma_t(k)$ |
| Viterbi   | $\max$-product | MAP sequence $\mathbf{z}^*$                    |

### 10.4 Baum-Welch and EM for HMMs

**Baum-Welch** is EM applied to HMMs, with closed-form E- and M-steps.

**E-step**: compute $\gamma_t(k)$ and $\xi_t(j,k)$ using forward-backward.

**M-step**: update parameters:
$$\hat{\pi}_{0,k} = \gamma_1(k)$$
$$\hat{A}_{jk} = \frac{\sum_{t=1}^{T-1} \xi_t(j,k)}{\sum_{t=1}^{T-1} \gamma_t(j)}$$
$$\hat{B}_k(v) = \frac{\sum_{t=1}^T \gamma_t(k)\, \mathbf{1}[x_t = v]}{\sum_{t=1}^T \gamma_t(k)}$$

These are **weighted frequency counts**: the transition frequency $A_{jk}$ is the expected number of $j \to k$ transitions divided by expected number of transitions from $j$.

**Convergence**: Baum-Welch is guaranteed to increase $\log p(\mathbf{x}_{1:T} \mid \boldsymbol{\theta})$ at each iteration. Local optima are common — multiple random initialisations are recommended.

**State space models** (generalisation): replace discrete states with continuous $\mathbf{z}_t \in \mathbb{R}^d$ and linear-Gaussian transitions/emissions:
$$\mathbf{z}_t = F \mathbf{z}_{t-1} + \boldsymbol{\epsilon}_t, \quad \mathbf{x}_t = H \mathbf{z}_t + \boldsymbol{\delta}_t$$

This is the **linear dynamical system** (LDS) / Kalman filter model. The Kalman filter computes the exact forward pass in $O(d^3)$ (no exponential state space to sum over). S4/Mamba SSMs are learned LDS models parameterised for efficient parallel inference.

---

## 11. Bayesian Neural Networks and Uncertainty

### 11.1 Weight Distributions

A Bayesian Neural Network (BNN) places a prior distribution over weights $W$ and maintains the full posterior $p(W \mid \mathcal{D})$ rather than a point estimate.

**Generative model:**
$$p(W) = \prod_{l,i,j} \mathcal{N}(W^{[l]}_{ij} \mid 0, \sigma_p^2) \qquad \text{(prior)}$$
$$p(\mathcal{D} \mid W) = \prod_{i=1}^n p(y^{(i)} \mid f_W(\mathbf{x}^{(i)})) \qquad \text{(likelihood)}$$

**Posterior**: $p(W \mid \mathcal{D}) \propto p(\mathcal{D} \mid W)\, p(W)$ — intractable for non-trivial architectures because the network is non-linear in $W$.

**Predictive distribution** (the Bayesian answer to "what does the model predict?"):
$$p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid f_W(\mathbf{x}^*))\, p(W \mid \mathcal{D})\, dW$$

This integral marginalises over all plausible weight configurations — it is the "Bayesian model average" over infinitely many neural networks.

**Approaches to approximate the posterior:**

1. **Laplace approximation** (§11.2): Gaussian centred at MAP, curvature from Hessian
2. **Variational BNN** (Graves 2011, Blundell et al. 2015): mean-field $q(W) = \prod_{ij} \mathcal{N}(w_{ij} \mid \mu_{ij}, \sigma_{ij}^2)$
3. **MC Dropout** (§11.3): dropout at test time ≈ approximate inference
4. **Deep Ensembles** (Lakshminarayanan et al. 2017): $M$ independently trained networks, simple and empirically strong

### 11.2 Laplace Approximation

**Idea**: approximate $p(W \mid \mathcal{D})$ with a Gaussian centred at the MAP estimate $W_\text{MAP} = \arg\max_W \log p(W \mid \mathcal{D})$.

**Taylor expand** $\log p(W \mid \mathcal{D})$ around $W_\text{MAP}$:
$$\log p(W \mid \mathcal{D}) \approx \log p(W_\text{MAP} \mid \mathcal{D}) - \frac{1}{2}(W - W_\text{MAP})^\top H (W - W_\text{MAP})$$

where $H = -\nabla^2 \log p(W \mid \mathcal{D})\big|_{W_\text{MAP}}$ is the **Hessian of the negative log posterior** (positive definite at a local maximum).

**Laplace approximation**: $p(W \mid \mathcal{D}) \approx \mathcal{N}(W_\text{MAP}, H^{-1})$.

For a network with $P$ parameters: $H \in \mathbb{R}^{P \times P}$, $P \sim 10^8$–$10^{12}$ for modern LLMs. Full Hessian storage is infeasible.

**Practical approximations:**

- **Diagonal**: keep only diagonal of $H$ (variances, no covariances)
- **Kronecker-factored** (K-FAC): approximate each layer's Hessian as Kronecker product
- **Last-layer Laplace** (Daxberger et al. 2021): fix all layers except the last, apply Laplace to final layer only — cheap and empirically competitive

**Bernstein-von Mises theorem**: as $n \to \infty$, the posterior contracts to $\mathcal{N}(W_\text{MLE}, I^{-1}(W_\text{MLE})/n)$ where $I$ is the Fisher information. The Laplace approximation becomes exact asymptotically — it is a valid approximation in large-$n$ regimes.

### 11.3 MC Dropout as Approximate Inference

**Gal & Ghahramani (2016)** showed that a neural network with dropout applied before every weight layer is equivalent to approximate variational inference in a deep Gaussian process.

**Interpretation**: each dropout mask $\boldsymbol{\zeta}$ (binary vector) specifies a subnetwork. Dropout training optimises a variational lower bound where $q(W) = \prod_j \sum_z \boldsymbol{\zeta}_j^z W_j$ — a mixture of two Gaussians (zero + non-zero) per weight.

**MC Dropout for uncertainty**: at test time, keep dropout on and run $T$ forward passes:
$$\hat{y}_t = f_{W_t}(\mathbf{x}^*), \quad W_t \sim q(W)$$

**Predictive mean**: $\bar{y} = \frac{1}{T}\sum_t \hat{y}_t$

**Predictive uncertainty** (empirical variance):
$$\operatorname{Var}[\hat{y}] = \frac{1}{T}\sum_t \hat{y}_t^2 - \bar{y}^2 + \hat{\sigma}^2$$

where $\hat{\sigma}^2$ is the aleatoric noise (modelled by heteroscedastic output).

**Calibration**: MC Dropout is not perfectly calibrated — its uncertainty estimates are better than deterministic networks but worse than properly trained ensembles or GP models. The connection to BNNs is approximate; strong priors are needed to make the dropout rate meaningful.

### 11.4 Aleatoric vs. Epistemic Uncertainty

**Aleatoric uncertainty** (data uncertainty): inherent randomness in the target — cannot be reduced by more data.

- **Homoscedastic**: constant $\sigma^2$ for all inputs
- **Heteroscedastic**: $\sigma^2(\mathbf{x})$ depends on input — model outputs $(\mu(\mathbf{x}), \sigma^2(\mathbf{x}))$
- Loss: $\mathcal{L} = \sum_i \frac{(y^{(i)} - \mu(\mathbf{x}^{(i)}))^2}{2\sigma^2(\mathbf{x}^{(i)})} + \frac{1}{2}\log\sigma^2(\mathbf{x}^{(i)})$ (negative log Gaussian likelihood)

**Epistemic uncertainty** (model uncertainty): due to insufficient data. Reduced by observing more data. Estimated by variance across ensemble members or BNN posterior samples.

**Decomposition of total uncertainty** (using predictive entropy):
$$H[y^* \mid \mathbf{x}^*, \mathcal{D}] = \underbrace{\mathbb{E}_{p(W \mid \mathcal{D})}[H[y^* \mid \mathbf{x}^*, W]]}_{\text{aleatoric}} + \underbrace{I(y^*; W \mid \mathbf{x}^*, \mathcal{D})}_{\text{epistemic (mutual information)}}$$

**For AI**: uncertainty estimation is critical for:

- **Safety**: LLMs should be more uncertain on out-of-distribution queries (medical/legal advice)
- **Active learning**: query the most uncertain inputs for human annotation
- **RLHF reward models**: reward model uncertainty informs how much to trust reward signal during PPO training
- **Conformal prediction**: use empirical uncertainty for provably calibrated prediction sets

---

## 12. Normalizing Flows

### 12.1 Change of Variables Formula

A normalizing flow transforms a simple base distribution $p_Z(\mathbf{z})$ (e.g., $\mathcal{N}(\mathbf{0}, I)$) into a complex distribution $p_X(\mathbf{x})$ via an invertible, differentiable transformation $f: \mathbb{R}^d \to \mathbb{R}^d$.

**Change of variables (exact density):**
$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \left\lvert \det J_{f^{-1}}(\mathbf{x}) \right\rvert = p_Z(\mathbf{z}) \left\lvert \det J_f(\mathbf{z}) \right\rvert^{-1}$$

where $J_f(\mathbf{z}) = \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} \in \mathbb{R}^{d \times d}$ is the Jacobian of $f$.

**Log-likelihood** (for training by maximum likelihood):
$$\log p_X(\mathbf{x}) = \log p_Z(f^{-1}(\mathbf{x})) + \log\left\lvert\det J_{f^{-1}}(\mathbf{x})\right\rvert$$

**Key challenge**: computing $\log\lvert\det J\rvert$ is $O(d^3)$ for a general $d \times d$ matrix — prohibitive for large $d$ (images, audio). Flow architectures are designed to make Jacobian computation cheap.

**Composing flows**: if $f = f_K \circ \cdots \circ f_1$, then:
$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \sum_{k=1}^K \log\left\lvert\det J_{f_k}(\mathbf{z}_{k-1})\right\rvert$$

### 12.2 Flow Architectures

**NICE (Non-linear Independent Components Estimation, Dinh et al. 2014):**
Split $\mathbf{x}$ into $(\mathbf{x}_1, \mathbf{x}_2)$. **Additive coupling layer:**
$$\mathbf{y}_1 = \mathbf{x}_1, \quad \mathbf{y}_2 = \mathbf{x}_2 + m_\theta(\mathbf{x}_1)$$

Inverse: $\mathbf{x}_2 = \mathbf{y}_2 - m_\theta(\mathbf{y}_1)$. Jacobian is **triangular** → $\det J = 1$. No log-det computation needed.

**RealNVP (Dinh et al. 2017):** Affine coupling:
$$\mathbf{y}_2 = \mathbf{x}_2 \odot \exp(s_\theta(\mathbf{x}_1)) + t_\theta(\mathbf{x}_1)$$

$\log\lvert\det J\rvert = \sum_j s_\theta(\mathbf{x}_1)_j$ — computable in $O(d)$.

**Masked Autoregressive Flow (MAF, Papamakarios et al. 2017):**
$$x_i = \mu_i(\mathbf{x}_{<i}) + \sigma_i(\mathbf{x}_{<i}) \cdot z_i$$

Each variable depends on all previous. Fast density evaluation ($O(d)$, parallel), slow sampling ($O(d)$, sequential). Used in density estimation and variational inference.

**Inverse Autoregressive Flow (IAF, Kingma et al. 2016):**
Reverse of MAF — fast sampling, slow density evaluation. Used in VAE decoders for flexible posteriors.

**Neural Spline Flows (Durkan et al. 2019):**
Replace affine coupling with monotone spline transformations — more expressive per layer while maintaining tractable Jacobians. State-of-the-art on tabular data.

### 12.3 Continuous Normalizing Flows

**Neural ODEs (Chen et al. 2018 — NeurIPS Best Paper):**
Instead of discrete flow layers, parameterise a continuous vector field:
$$\frac{d\mathbf{z}(t)}{dt} = f_\theta(\mathbf{z}(t), t)$$

The change in log-density evolves as:
$$\frac{d\log p(\mathbf{z}(t))}{dt} = -\operatorname{tr}\!\left(\frac{\partial f_\theta}{\partial \mathbf{z}}\right)$$

**Instantaneous change of variables:**
$$\log p(\mathbf{x}) = \log p(\mathbf{z}_0) - \int_0^1 \operatorname{tr}\!\left(\frac{\partial f_\theta(\mathbf{z}(t), t)}{\partial \mathbf{z}(t)}\right)\, dt$$

**Hutchinson's trace estimator**: $\operatorname{tr}(A) = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},I)}[\boldsymbol{\epsilon}^\top A \boldsymbol{\epsilon}]$ — reduces $O(d^2)$ Jacobian to $O(d)$ expected cost.

CNFs are more expressive than discrete flows (infinite "layers") but training requires ODE solver, making them slower than discrete counterparts.

### 12.4 Training and Applications

**Maximum likelihood training**: minimise $-\frac{1}{n}\sum_i \log p_X(\mathbf{x}^{(i)})$.

**Applications:**

- **Density estimation**: model complex data distributions (tabular, point cloud, molecular)
- **Variational inference**: use a flow as $q_\phi(\mathbf{z} \mid \mathbf{x})$ — richer posterior approximation than Gaussian
- **Dequantisation**: discrete images require continuous density; flows handle this via learned dequantisation
- **Posterior sampling**: in Bayesian networks, learn the posterior shape with a flow

**For AI**: normalising flows are used in:

- **Glow** (Kingma & Dhariwal 2018): flow-based image generation, real-time synthesis
- **WaveGlow**: audio synthesis
- **Neural posterior estimation**: likelihood-free inference for scientific simulators (used in physics, neuroscience)
- **Flow Matching** (§14): the modern successor to CNFs, training flows without density evaluation

---

## 13. Score Matching and Diffusion Models

### 13.1 Score Functions and Energy-Based Models

**Score function**: the gradient of the log-density with respect to the data:
$$\mathbf{s}(\mathbf{x}) = \nabla_\mathbf{x} \log p(\mathbf{x})$$

For an energy-based model $p(\mathbf{x}) = \frac{1}{Z} e^{-E_\theta(\mathbf{x})}$:
$$\mathbf{s}_\theta(\mathbf{x}) = -\nabla_\mathbf{x} E_\theta(\mathbf{x})$$

The partition function $Z$ cancels — score estimation does not require computing $Z$. This is the key advantage over maximum likelihood for energy-based models.

**Score matching objective** (Hyvärinen 2005): minimise the Fisher divergence between the true score $\mathbf{s}(\mathbf{x})$ and model score $\mathbf{s}_\theta(\mathbf{x})$:

$$\mathcal{J}_\text{SM}(\theta) = \frac{1}{2}\mathbb{E}_{p(\mathbf{x})}\!\left[\lVert \mathbf{s}_\theta(\mathbf{x}) - \nabla_\mathbf{x}\log p(\mathbf{x})\rVert^2\right]$$

Using integration by parts, this simplifies to:
$$\mathcal{J}_\text{SM}(\theta) = \mathbb{E}_{p(\mathbf{x})}\!\left[\operatorname{tr}(\nabla_\mathbf{x} \mathbf{s}_\theta(\mathbf{x})) + \frac{1}{2}\lVert \mathbf{s}_\theta(\mathbf{x})\rVert^2\right] + \mathrm{const}$$

Tractable when $\operatorname{tr}(\nabla_\mathbf{x} \mathbf{s}_\theta)$ is cheap (sliced score matching uses random projections to avoid the full trace).

**Sampling via Langevin dynamics**: given $\mathbf{s}_\theta(\mathbf{x})$, generate samples:
$$\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\epsilon}{2}\mathbf{s}_\theta(\mathbf{x}_t) + \sqrt{\epsilon}\boldsymbol{\xi}_t, \quad \boldsymbol{\xi}_t \sim \mathcal{N}(\mathbf{0}, I)$$

As $\epsilon \to 0$ and $t \to \infty$, this converges to samples from $p(\mathbf{x})$.

### 13.2 Denoising Score Matching

**Denoising Score Matching (DSM, Vincent 2011)**: add Gaussian noise to data, train a denoiser.

Corrupt data: $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\sigma}\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$.

**Key theorem** (Vincent 2011): the score of the smoothed distribution equals the conditional expectation of the clean data:
$$\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}) = \frac{\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] - \tilde{\mathbf{x}}}{\sigma^2}$$

**DSM objective**: train $\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma)$ to predict $(\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2 = -\boldsymbol{\epsilon}/\sigma$:

$$\mathcal{L}_\text{DSM} = \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}, \sigma}\!\left[\lVert \mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}, \sigma) + \boldsymbol{\epsilon}/\sigma\rVert^2\right]$$

Equivalently (DDPM parameterisation), predict the noise $\boldsymbol{\epsilon}$ from the noisy data.

**Multi-scale noise** (Song & Ermon 2019): train on multiple noise levels $\sigma_1 > \sigma_2 > \cdots > \sigma_L$:

- Large $\sigma$: high-noise samples cover all modes (score well-defined globally)
- Small $\sigma$: low-noise samples capture fine-grained structure

Anneal noise during sampling (start with large noise, gradually decrease) — **annealed Langevin dynamics**.

### 13.3 Score-Based SDEs

**Song et al. (2021) — ICLR 2021 Outstanding Paper.** Unify diffusion models and score-based models via Stochastic Differential Equations (SDEs).

**Forward SDE** (adds noise gradually): starting from data $\mathbf{x}_0 \sim p_\text{data}$:
$$d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$$

where $\mathbf{W}$ is Brownian motion. The marginal $p_t(\mathbf{x})$ transitions from $p_\text{data}$ to a simple prior (typically $\mathcal{N}(\mathbf{0}, I)$).

**Reverse SDE** (removes noise, generates data): starting from $p_T \approx \mathcal{N}(\mathbf{0}, I)$:
$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})\right]\, dt + g(t)\, d\bar{\mathbf{W}}$$

$\nabla_\mathbf{x} \log p_t(\mathbf{x})$ is the **score function** — estimated by a time-conditioned score network $\mathbf{s}_\theta(\mathbf{x}, t)$.

**Two canonical SDEs:**

- **VE-SDE** (Variance Exploding): $\mathbf{f}=0$, $g(t) = \sqrt{d[\sigma^2(t)]/dt}$. Variance grows without bound.
- **VP-SDE** (Variance Preserving): $\mathbf{f}(\mathbf{x},t) = -\frac{1}{2}\beta(t)\mathbf{x}$, $g(t)=\sqrt{\beta(t)}$. Variance stays bounded; subsumes DDPM.

**Probability flow ODE**: the reverse SDE has a deterministic counterpart (no noise):
$$d\mathbf{x} = \left[\mathbf{f}(\mathbf{x},t) - \frac{1}{2}g(t)^2 \nabla_\mathbf{x}\log p_t(\mathbf{x})\right]\, dt$$

This ODE has the same marginals as the SDE but allows exact likelihood computation (via continuous change-of-variables) and deterministic, faster sampling.

### 13.4 DDPM, DDIM, and Guidance

**DDPM (Ho et al. 2020 — NeurIPS 2020):** Discrete-time VP-SDE with linear noise schedule $\beta_1, \ldots, \beta_T$.

**Forward process** (Markov chain, closed-form marginals):
$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t \mid \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)I)$$

where $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$.

**Training objective** (simplified — predict noise):
$$\mathcal{L}_\text{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\!\left[\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon},\, t)\rVert^2\right]$$

This is denoising score matching in disguise — the network predicts the noise $\boldsymbol{\epsilon}$ added at each step.

**DDIM (Song et al. 2020):** Replace the stochastic reverse process with a deterministic ODE solver — reduces $T=1000$ steps to $\approx 50$ steps with similar quality. Quality–speed tradeoff controlled by $\eta \in [0,1]$ (0 = deterministic, 1 = DDPM).

**Classifier-free guidance (Ho & Salimans 2021):**
Train a single model that handles both unconditional and conditional generation:
$$\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, c) = (1+w)\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, c) - w\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \emptyset)$$

$w > 0$: guidance scale — pushes toward conditioned region. High $w$ → sharp images, low diversity; low $w$ → diverse samples, less sharp. Stable Diffusion uses $w \approx 7.5$.

**For AI**: diffusion models underpin Stable Diffusion, DALL-E 3, Midjourney, Sora (video diffusion), AlphaFold 3 protein structure prediction, and molecular drug design. The score network architecture (originally U-Net, now DiT — Diffusion Transformer) is trained on massive datasets and conditions on text embeddings from CLIP or T5.

---

## 14. Flow Matching

### 14.1 Probability Paths and Vector Fields

**Continuity equation.** A time-varying vector field $\mathbf{v}_t: \mathbb{R}^d \to \mathbb{R}^d$ defines a flow — a trajectory $\mathbf{x}(t)$ for each starting point via $d\mathbf{x}/dt = \mathbf{v}_t(\mathbf{x}(t))$. The marginal density $p_t$ evolves according to the **continuity equation**:

$$\frac{\partial p_t(\mathbf{x})}{\partial t} + \nabla_\mathbf{x} \cdot (p_t(\mathbf{x})\, \mathbf{v}_t(\mathbf{x})) = 0$$

We want to find $\mathbf{v}_t$ such that $p_0 = \mathcal{N}(\mathbf{0}, I)$ (noise) and $p_1 = p_\text{data}$ (data).

**Flow Matching idea** (Lipman et al. 2022): directly train a neural network $\mathbf{v}_\theta(\mathbf{x}, t)$ to match the vector field $\mathbf{v}_t$ generating a desired probability path.

### 14.2 Flow Matching Objective

**Problem**: the marginal vector field $\mathbf{v}_t(\mathbf{x})$ is intractable — computing it requires marginalising over all data points.

**Conditional Flow Matching (CFM)**: choose a **conditional probability path** $p_t(\mathbf{x} \mid \mathbf{x}_1)$ that interpolates from noise to each individual data point $\mathbf{x}_1$:

$$p_t(\mathbf{x} \mid \mathbf{x}_1) = \mathcal{N}(\mathbf{x} \mid t\mathbf{x}_1, (1-(1-\sigma_\text{min})t)^2 I)$$

The **conditional vector field** that generates this path is:
$$\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - (1-\sigma_\text{min})\mathbf{x}}{1 - (1-\sigma_\text{min})t}$$

**CFM loss** (regress on conditional vector field):
$$\mathcal{L}_\text{CFM}(\theta) = \mathbb{E}_{t, \mathbf{x}_1 \sim p_\text{data}, \mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)}\!\left[\lVert \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{u}_t(\mathbf{x}_t \mid \mathbf{x}_1)\rVert^2\right]$$

where $\mathbf{x}_t = t\mathbf{x}_1 + (1-(1-\sigma_\text{min})t)\mathbf{x}_0$ (linear interpolation).

**Key theorem** (Lipman et al. 2022): $\nabla_\theta \mathcal{L}_\text{FM} = \nabla_\theta \mathcal{L}_\text{CFM}$ — the conditional and marginal losses have identical gradients. This makes the conditional loss tractable to optimise.

**Advantages over diffusion:**

- Simpler training objective (no noise schedule engineering)
- Straighter trajectories → fewer NFE at inference
- Flexible path design (not restricted to Gaussian noise schedule)

### 14.3 Rectified Flows

**Rectified Flows** (Liu et al. 2022): the simplest possible path — straight lines between noise $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)$ and data $\mathbf{x}_1 \sim p_\text{data}$:

$$\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$$

Vector field: $\mathbf{v}_t(\mathbf{x}_t) = \mathbf{x}_1 - \mathbf{x}_0$ — constant along each trajectory.

**Reflow**: after training, generate a new dataset of (noise, data) pairs using ODE simulation. Train a second flow on these pairs — their paths are straighter (less trajectory crossing), requiring fewer ODE steps.

**Optimal transport**: the CFM objective with independent couplings between $\mathbf{x}_0$ and $\mathbf{x}_1$ is suboptimal. OT-based couplings (minimax OT matching) minimise trajectory length — resulting in the straightest possible paths and fewest NFE.

### 14.4 Flow Matching vs. Diffusion

| Property               | Diffusion (DDPM/SDE)                  | Flow Matching                     |
| ---------------------- | ------------------------------------- | --------------------------------- |
| Path                   | Stochastic, SDE                       | Deterministic ODE                 |
| Training objective     | Predict noise $\boldsymbol{\epsilon}$ | Predict vector field $\mathbf{v}$ |
| Inference steps        | 50–1000 (DDIM: 20–50)                 | 5–30 (ODE solver)                 |
| Likelihood computation | Via probability flow ODE              | Direct (change of variables)      |
| Theory                 | Well-established                      | Newer (2022–)                     |
| Production models      | SD 1.x/2.x, DALL-E 3                  | SD3, Flux, CogVideoX              |

**Mathematical equivalence**: under Gaussian conditional paths and matching parameterisations, flow matching and diffusion models with the probability flow ODE are equivalent. Flow matching can be viewed as a reparameterisation of the DDPM training objective — but the conceptual framing is cleaner and leads to better trajectory designs.

**For AI**: Stable Diffusion 3 (2024) and Flux (2024) use rectified flows / CFM. Sora and other video generation models use diffusion transformers with flow-based training. The trend in 2024–2025 is a clear shift from noise-prediction diffusion to vector-field flow matching due to training stability and inference efficiency.

---

## 15. In-Context Learning as Bayesian Inference

### 15.1 Xie et al. (2021) Framework

**Setup** (Xie et al., "Explanation of In-Context Learning as Implicit Bayesian Inference", NeurIPS 2022): assume the pretraining corpus is generated by a hierarchical model:

1. Sample a **latent concept** $C \sim p(C)$ (e.g., the "style" or "topic" of a document)
2. Sample document tokens $w_1, \ldots, w_T$ i.i.d. from $p(w \mid C)$

**In-context learning as posterior update**: given $n$ in-context examples $(w_1, \ldots, w_n)$:
$$p(C \mid w_1, \ldots, w_n) \propto p(C) \prod_{i=1}^n p(w_i \mid C)$$

The posterior over $C$ is updated by each example — Bayesian inference over which document type/task $C$ is occurring.

**One-step-ahead prediction** (what the LM learns to compute):
$$p(w_{n+1} \mid w_1, \ldots, w_n) = \sum_C p(w_{n+1} \mid C)\, p(C \mid w_1, \ldots, w_n)$$

This is a **Bayes-optimal predictor** — the LM minimises perplexity by implicitly computing the posterior $p(C \mid \text{context})$.

**Key insight**: in-context learning does not update weights. Instead, the context serves as evidence that updates a latent posterior over tasks/concepts, and the transformer computes the posterior-predictive by marginalising over this posterior. No gradient descent is needed.

### 15.2 Transformer as Approximate Posterior

**How does the transformer implement Bayesian inference?** Akyürek et al. (2022) and von Oswald et al. (2023) showed that transformer attention implements gradient descent steps on a ridge regression problem — connecting ICL to gradient-based meta-learning.

More directly: attention can compute weighted sums that approximate sufficient statistics. For a Gaussian likelihood $p(w \mid \mu_C) = \mathcal{N}(\mu_C, \sigma^2)$ with $p(\mu_C) = \mathcal{N}(\mu_0, \sigma_0^2)$:

$$p(\mu_C \mid w_1, \ldots, w_n) = \mathcal{N}\!\left(\frac{\frac{\mu_0}{\sigma_0^2} + \frac{\sum_i w_i}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}},\; \left(\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}\right)^{-1}\right)$$

The numerator $\sum_i w_i$ is exactly what attention aggregates. The transformer is learning to compute these sums with learned weights — a **differentiable implementation of Bayesian posterior updating**.

### 15.3 Modern Extensions (2024–2025)

**Prior-Data Fitted Networks (PFNs, Müller et al. 2022 → tabPFN 2024)**: train a transformer on synthetic Bayesian posteriors. During training:

1. Sample a prior $p(\boldsymbol{\theta})$ (e.g., random Bayesian network)
2. Sample dataset $\mathcal{D}_\text{train} \sim p(\mathbf{x}, y \mid \boldsymbol{\theta})$
3. Train the transformer to predict $p(y^* \mid \mathbf{x}^*, \mathcal{D}_\text{train})$ exactly

At test time, the PFN **is** the Bayesian posterior predictor for the in-context data — no fine-tuning required.

**tabPFN** achieves state-of-the-art on tabular classification by implementing exact Bayesian inference in-context, outperforming XGBoost on small datasets.

**Full Bayesian Inference in Context (Müller et al. ICML 2025)**: extends PFNs to infer full posterior distributions (not just predictions) — outputs a distribution over $\boldsymbol{\theta}$ given in-context data. Handles generalized linear models, latent factor models, and other statistical models without explicit parameter updates.

### 15.4 Implications for LLM Design

**Prompt as prior**: the system prompt encodes the prior $p(C)$. A strong, specific system prompt narrows the posterior over tasks — equivalent to an informative Bayesian prior.

**Context window = posterior conditioning**: more in-context examples → sharper posterior over $C$ → better task performance. The length of effective in-context conditioning is limited by the transformer's ability to maintain posterior accuracy over long contexts.

**Temperature as precision**: sampling temperature $\tau$ scales the logits. Low $\tau \to 0$: MAP decoding (argmax). High $\tau$: flat distribution. In the Bayesian framework, temperature controls the **sharpness** of the categorical likelihood — $\tau^{-1}$ is the precision of the Categorical distribution.

**RAG as Bayesian updating**: Retrieval-Augmented Generation retrieves relevant documents and adds them to the context. In the Bayesian view, each retrieved document is a new observation that updates $p(C \mid \text{context})$, sharpening the posterior toward the true task.

**For AI**: this theoretical framework explains:

- Why few-shot prompting works even without gradient updates
- Why prompts with examples are strictly better than prompts without (posterior sharpening)
- Why longer context improves performance monotonically (more evidence)
- Why instruction-following degrades with too many conflicting examples (posterior confusion)

---

## 16. Common Mistakes

| #   | Mistake                                                                                          | Why It's Wrong                                                                                                                         | Fix                                                                                                                  |
| --- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| 1   | Treating $p(x)$ as a probability when $X$ is continuous                                          | For continuous RVs, $p(x)\,dx$ is a probability; $p(x)$ can exceed 1                                                                   | Always integrate densities; reserve $P$ for probabilities                                                            |
| 2   | Confusing forward and reverse KL: minimising $D_{\mathrm{KL}}(p\|q)$ vs. $D_{\mathrm{KL}}(q\|p)$ | Reverse KL is mode-seeking (underestimates posterior width); forward KL is mass-covering                                               | VI minimises $D_{\mathrm{KL}}(q\|p)$; use forward KL when full coverage matters                                      |
| 3   | Posterior collapse in VAEs interpreted as convergence                                            | The ELBO can be high even when $q(\mathbf{z} \mid \mathbf{x}) \approx p(\mathbf{z})$ if the decoder ignores $\mathbf{z}$               | Monitor KL term and mutual information $I(\mathbf{x}; \mathbf{z})$ separately; use KL annealing                      |
| 4   | Using EM to find global maximum of likelihood                                                    | EM only guarantees local maximum (or saddle point)                                                                                     | Multiple random restarts; warm-start from k-means; check convergence of likelihood                                   |
| 5   | Treating MCMC samples as i.i.d.                                                                  | MCMC samples are autocorrelated; effective sample size $\ll$ number of iterations                                                      | Thin the chain; compute ESS; use NUTS for lower autocorrelation                                                      |
| 6   | Ignoring identifiability in latent variable models                                               | GMM has $K!$ equivalent solutions; FA has rotation ambiguity; conclusions about specific latents are meaningless                       | Add identifiability constraints; report invariant quantities                                                         |
| 7   | Conflating aleatoric and epistemic uncertainty                                                   | Aleatoric cannot be reduced; conflating them leads to wasted data collection                                                           | Decompose total entropy into aleatoric + epistemic components explicitly                                             |
| 8   | Using GP with wrong kernel                                                                       | SE kernel assumes infinite smoothness — a poor assumption for most physical processes                                                  | Use Matérn 5/2 by default; compare via marginal likelihood                                                           |
| 9   | Forgetting burn-in in MCMC                                                                       | Early samples are not from the stationary distribution                                                                                 | Discard first $T_\text{burn}$ samples; assess via $\hat{R}$ across multiple chains                                   |
| 10  | Confusing denoising score matching loss with diffusion model loss                                | DSM loss uses $(y - \mathbf{x})/\sigma^2$ as target; DDPM uses $\boldsymbol{\epsilon}$ as target — they differ by a $\sigma^2$ scaling | Track whether your model predicts score, noise, or $\mathbf{x}_0$ — they are related but require different inference |
| 11  | Running belief propagation on loopy graphs and expecting exact results                           | BP is only exact on trees; loopy BP is an approximation with no convergence guarantee                                                  | Use junction tree algorithm for exact inference; loopy BP as approximate method                                      |
| 12  | Comparing ELBO values across models with different architectures                                 | ELBO depends on the normalisation constant; different encoders make ELBOs incomparable                                                 | Use IWAE with $K=1000$ samples for fair comparison                                                                   |

---

## 17. Exercises

**Exercise 1 ★ — Conjugate Posterior**

Let $X_1, \ldots, X_n \sim \operatorname{Bern}(p)$ and $p \sim \operatorname{Beta}(\alpha, \beta)$.

(a) Derive the posterior $p(p \mid X_{1:n})$ by computing $p(X_{1:n} \mid p)\, p(p)$ up to normalisation.
(b) Show the posterior is $\operatorname{Beta}(\alpha + \sum X_i, \beta + n - \sum X_i)$.
(c) Compute the posterior predictive $p(X_{n+1} = 1 \mid X_{1:n})$ as a function of sufficient statistics.
(d) Verify that as $n \to \infty$, the posterior predictive converges to the MLE $\hat{p} = \bar{X}$.

**Exercise 2 ★ — Exponential Family Moments**

Consider the Poisson distribution with natural parameter $\eta = \log\lambda$.

(a) Write the Poisson in exponential family form $h(x)\exp(\eta T(x) - A(\eta))$.
(b) Compute $A(\eta)$ and verify $\nabla_\eta A(\eta) = \mathbb{E}[X] = \lambda$.
(c) Compute $\nabla^2_\eta A(\eta)$ and verify it equals $\operatorname{Var}[X] = \lambda$.
(d) Derive the conjugate prior for $\eta$ and compute the posterior after observing $x_1, \ldots, x_n$.

**Exercise 3 ★ — EM Lower Bound**

(a) Using Jensen's inequality, prove that $\log p(\mathbf{x}) \geq \mathcal{L}(q, \boldsymbol{\theta})$ for any $q$.
(b) Show the bound is tight when $q = p(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta})$.
(c) Implement EM for a 2-component 1D Gaussian mixture. Verify monotonic increase of the log-likelihood across 30 iterations.
(d) Plot the responsibility $r_{1k}$ for 100 data points as a function of EM iteration.

**Exercise 4 ★★ — CAVI for a Simple Model**

Consider a Gaussian model: $\mu \sim \mathcal{N}(0, \tau^{-1})$, $x_i \mid \mu \sim \mathcal{N}(\mu, \sigma^2)$.

(a) Derive the CAVI update for $q(\mu) = \mathcal{N}(\mu \mid m, s^2)$.
(b) Show the optimal $q^*(\mu)$ is Gaussian with $m = \frac{n\bar{x}/\sigma^2}{n/\sigma^2 + \tau}$ and $s^2 = (n/\sigma^2 + \tau)^{-1}$.
(c) Compare the variational posterior to the exact conjugate posterior — are they the same?
(d) Compute the ELBO as a function of $m$ and $s^2$ and verify the CAVI update maximises it.

**Exercise 5 ★★ — Reparameterisation Gradient**

Let $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi, \text{diag}(\boldsymbol{\sigma}^2_\phi))$ with $\boldsymbol{\mu}_\phi = W\mathbf{x} + \mathbf{b}$ and $\log\boldsymbol{\sigma}^2_\phi = V\mathbf{x} + \mathbf{c}$.

(a) Write the reparameterised sample $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$.
(b) Compute $\nabla_W \mathcal{L}$ using the reparameterised estimator with one sample.
(c) Compare empirically: estimate $\nabla_W \mathcal{L}$ with 1000 samples using (i) score function estimator and (ii) reparameterisation. Compute the variance of each estimator.
(d) Derive the closed-form KL $D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\sigma}^2) \| \mathcal{N}(\mathbf{0},I))$ and implement it without sampling.

**Exercise 6 ★★ — Metropolis-Hastings Sampler**

Implement MH to sample from a bimodal target $p(x) \propto \exp(-\frac{(x-2)^2}{2}) + 0.5\exp(-\frac{(x+2)^2}{2})$.

(a) Use Gaussian proposal $q(x' \mid x) = \mathcal{N}(x, \sigma^2)$ with $\sigma \in \{0.1, 1.0, 5.0\}$. Plot trace plots and compare mixing.
(b) Compute the acceptance rate for each $\sigma$. What value gives $\approx 23\%$ acceptance?
(c) Compute the effective sample size for each $\sigma$ using the autocorrelation function.
(d) Compare sample histograms to the true density for 10,000 samples after burn-in of 1,000.

**Exercise 7 ★★ — GP Regression**

(a) Implement GP regression with RBF kernel $k(x,x') = \exp(-\frac{(x-x')^2}{2\ell^2})$ from scratch using NumPy.
(b) Generate 20 noisy observations from $f(x) = \sin(x)$ with $\sigma_n = 0.2$, $x \in [-5, 5]$.
(c) Compute and plot the posterior mean and ±2 standard deviation credible bands.
(d) Optimise $\ell$ and $\sigma_n$ by maximising the log marginal likelihood. Report the optimal values.
(e) Compare the GP posterior with $\ell \in \{0.5, 2, 10\}$ — how does the length-scale affect the posterior?

**Exercise 8 ★★★ — HMM Forward-Backward**

Consider a 2-state HMM with $K=2$ states and Gaussian emissions.

(a) Implement the forward algorithm in log-space. Verify: $\log p(\mathbf{x}_{1:T}) = \text{logsumexp}(\alpha_T)$.
(b) Implement the backward algorithm. Compute $\gamma_t(k) = p(z_t=k \mid \mathbf{x}_{1:T})$ and verify $\sum_k \gamma_t(k)=1$.
(c) Implement Baum-Welch. Run 50 iterations on 1000 observations. Plot the log-likelihood curve.
(d) Generate data from a known HMM and verify that Baum-Welch recovers the true parameters (up to label permutation).

**Exercise 9 ★★★ — Normalizing Flow Log-Likelihood**

Implement a 2D RealNVP flow with 4 affine coupling layers on a 2D two-moons dataset.

(a) Implement the forward and inverse pass of one coupling layer.
(b) Compute $\log\lvert\det J\rvert$ analytically (it should be $\sum_j s_\theta(\mathbf{x}_1)_j$).
(c) Train by maximising the exact log-likelihood. Plot the learned density over a 2D grid.
(d) Sample 1000 points from the learned distribution and compare to the training data visually.
(e) Compare the log-likelihood of: (i) Gaussian MLE, (ii) GMM with 5 components, (iii) your flow.

**Exercise 10 ★★★ — Diffusion Score Matching**

(a) Implement denoising score matching for 1D Gaussian target $p(\mathbf{x}) = \mathcal{N}(0, 1)$ with noise level $\sigma=0.5$.
(b) Train a small MLP $\mathbf{s}_\theta(x, \sigma)$ to predict $(x_\text{clean} - \tilde{x})/\sigma^2$.
(c) Implement annealed Langevin dynamics with noise levels $\sigma \in \{1.0, 0.5, 0.2, 0.05\}$.
(d) Generate 1000 samples. Compare their distribution to $\mathcal{N}(0,1)$ with a KS test.
(e) Repeat on a bimodal target $0.5\mathcal{N}(-2, 0.5) + 0.5\mathcal{N}(2, 0.5)$. Does the multi-scale approach capture both modes?

---

## 18. Why This Matters for AI (2026 Perspective)

| Concept                      | AI/LLM Application                                                               | Impact                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Bayes' theorem               | Bayesian hyperparameter optimisation (Optuna, Vizier)                            | Finds optimal LLM training hyperparameters 10× more efficiently than random search |
| Exponential family / softmax | Every transformer output layer; cross-entropy training loss                      | The mathematical foundation of all language model training                         |
| Graphical models / BNs       | Causal modelling; mechanistic interpretability circuit diagrams                  | Understand information flow in neural networks; identify polysemantic circuits     |
| EM algorithm                 | VQ-VAE training; Baum-Welch for S4/Mamba; CTC training                           | Core algorithm in sequence models, discrete tokenisation, ASR                      |
| ELBO / VI                    | VAE-based image tokenisers (VQ-VAE, VQGAN) used by DALL-E, Stable Diffusion      | Compress images to discrete codes for transformer-based generation                 |
| Reparameterisation trick     | Any stochastic differentiable computation; diffusion model training              | Makes gradient-based learning through sampling possible                            |
| VAEs                         | DALL-E 1 image codebooks; VQ-VAE in VideoGPT, MAGVIT                             | Discrete image tokenisation enabling LLM-style generation over visual tokens       |
| MCMC / Langevin              | Score-based models; diffusion sampling; Bayesian neural net inference            | Foundation of all diffusion-based generative models                                |
| Gaussian Processes           | Bayesian optimisation of neural architecture search (NAS); hyperparameter tuning | GPT-4, Claude hyperparameter tuning uses GP-based BO internally                    |
| Hidden Markov Models         | CTC (speech recognition); structured state space models (S4, Mamba)              | Mamba's selective SSM is a learned non-linear HMM with $O(T)$ inference            |
| Bayesian neural networks     | MC Dropout for uncertainty; ensembles for RLHF reward model confidence           | Reward model uncertainty prevents reward hacking in PPO RLHF                       |
| Normalizing flows            | Glow (image synthesis); posterior inference; dequantisation                      | Enables exact likelihood computation in generative models                          |
| Diffusion models             | Stable Diffusion, DALL-E 3, Sora, AlphaFold 3                                    | State-of-the-art image/video/protein generation (2022–2026)                        |
| Flow Matching                | Stable Diffusion 3, Flux, CogVideoX                                              | Cleaner theory, faster inference, replacing diffusion in production 2024–2026      |
| ICL as Bayes                 | Explains GPT/Claude few-shot learning; PFNs for tabular ML                       | Theoretical grounding for prompt engineering; tabPFN outperforms XGBoost           |

---

## 19. Conceptual Bridge

This section sits at the intersection of classical statistics and modern deep learning. From the neural networks section (§14-02), you brought the tools of representation learning, backpropagation, and deep architectures. This section adds the **probabilistic lens** — the ability to reason about distributions, uncertainty, and latent structure rather than just point estimates.

**Looking backward**, the prerequisite material provides:

- Linear algebra (§02): Gaussian distributions are defined by their mean vectors and covariance matrices; matrix operations underlie GP regression and PPCA
- Calculus (§05): gradients of the ELBO and score functions are the computational core of VI and diffusion
- Probability theory (§13-01): all of this section builds on basic probability axioms, expectation, and conditional independence
- Information theory (§13-02): KL divergence is the central object in VI; entropy quantifies uncertainty; mutual information measures latent code quality

**Looking forward**, this section enables:

- Generative models (§14-07): VAEs, GANs, diffusion models, flow models all require this section's probabilistic machinery
- Transformers (§14-05): attention is a soft lookup — a probabilistic weighted average. The theoretical analysis of transformers via the exponential family and in-context Bayesian inference (§15) connects directly to transformer architecture design
- Reinforcement learning (§14-06): reward modelling uses probabilistic regression; policy gradients are closely related to the score function estimator (§6.3); Bayesian RL uses GPs and probabilistic world models

```text
PROBABILISTIC MODELS — CURRICULUM POSITION
════════════════════════════════════════════════════════════════════════

  PREREQUISITES                THIS SECTION           ENABLES
  ─────────────────────────    ────────────────────   ─────────────────
  Probability Foundations  ──► Bayesian Inference ──► Generative Models
  Linear Algebra           ──► Graphical Models   ──► Transformers (VI)
  Calculus / Optimisation  ──► Latent Variables   ──► RL (policy grad)
  Neural Networks          ──► EM / VI / VAE      ──► LLM Theory (ICL)
  Information Theory       ──► MCMC / GPs / HMMs  ──► Diffusion / Flow
                                    │
                           Score Matching
                                    │
                           Flow Matching (2022)
                                    │
                      ICL as Bayes (2021–2025)

════════════════════════════════════════════════════════════════════════
```

**The unifying thread**: probability is not a framework imposed on top of deep learning — it is the natural language in which deep learning's core operations are best understood. The softmax is the exponential family mean map. Cross-entropy is negative log-likelihood. Weight decay is a Gaussian prior. Dropout is approximate VI. Diffusion training is denoising score matching. Transformer ICL is Bayesian posterior updating. Every technique in this curriculum has a probabilistic interpretation, and understanding that interpretation deepens both the theory and the practice.

---

## References

1. Bayes, T. (1763). "An Essay towards Solving a Problem in the Doctrine of Chances." _Philosophical Transactions of the Royal Society of London_ 53: 370–418.
2. Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm." _Journal of the Royal Statistical Society B_ 39(1): 1–22.
3. Pearl, J. (1988). _Probabilistic Reasoning in Intelligent Systems_. Morgan Kaufmann.
4. Duane, S., Kennedy, A.D., Pendleton, B.J., & Roweth, D. (1987). "Hybrid Monte Carlo." _Physics Letters B_ 195(2): 216–222.
5. Kingma, D.P., & Welling, M. (2014). "Auto-Encoding Variational Bayes." _ICLR 2014_. arXiv:1312.6114.
6. Rezende, D.J., & Mohamed, S. (2015). "Variational Inference with Normalizing Flows." _ICML 2015_. arXiv:1505.05770.
7. Rasmussen, C.E., & Williams, C.K.I. (2006). _Gaussian Processes for Machine Learning_. MIT Press.
8. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). "Density Estimation using Real-valued Non-Volume Preserving Transformations." _ICLR 2017_. arXiv:1605.08803.
9. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). "Neural Ordinary Differential Equations." _NeurIPS 2018_ (Best Paper). arXiv:1806.07366.
10. Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." _NeurIPS 2020_. arXiv:2006.11239.
11. Song, Y., Sohl-Dickstein, J., Kingma, D.P., et al. (2021). "Score-Based Generative Modeling through SDEs." _ICLR 2021 Outstanding Paper_. arXiv:2011.13456.
12. Xie, S.M., Raghunathan, A., Liang, P., & Ma, T. (2022). "An Explanation of In-Context Learning as Implicit Bayesian Inference." _NeurIPS 2022_. arXiv:2111.02080.
13. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). "Flow Matching for Generative Modeling." arXiv:2210.02747.
14. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." _Neural Computation_ 23(7): 1661–1674.
15. Gal, Y., & Ghahramani, Z. (2016). "Dropout as a Bayesian Approximation." _ICML 2016_. arXiv:1506.02142.
16. Wainwright, M.J., & Jordan, M.I. (2008). _Graphical Models, Exponential Families, and Variational Inference_. Now Publishers.
17. Blei, D.M., Kucukelbir, A., & McAuliffe, J.D. (2017). "Variational Inference: A Review for Statisticians." _JASA_ 112(518): 859–877.
18. Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
19. Müller, S., Hollmann, N., Arango, S.P., et al. (2022). "Transformers Can Do Bayesian Inference." _ICLR 2022_. arXiv:2112.10510.
20. Liu, X., Gong, C., & Liu, Q. (2022). "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." arXiv:2209.03003.

---

## Appendix A: Key Derivations

### A.1 ELBO Derivation via Jensen's Inequality (Step by Step)

For any distribution $q(\mathbf{z})$:

$$\log p(\mathbf{x}) = \log \int p(\mathbf{x}, \mathbf{z})\, d\mathbf{z}$$

$$= \log \int q(\mathbf{z}) \cdot \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\, d\mathbf{z}$$

$$= \log \mathbb{E}_{q(\mathbf{z})}\!\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right]$$

$$\geq \mathbb{E}_{q(\mathbf{z})}\!\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right] \quad \text{(Jensen: } \log \text{ is concave)}$$

$$= \mathbb{E}_{q}\!\left[\log p(\mathbf{x}, \mathbf{z})\right] - \mathbb{E}_{q}\!\left[\log q(\mathbf{z})\right]$$

$$= \mathbb{E}_{q}\!\left[\log p(\mathbf{x} \mid \mathbf{z}) + \log p(\mathbf{z})\right] + H(q)$$

$$= \underbrace{\mathbb{E}_{q}\!\left[\log p(\mathbf{x} \mid \mathbf{z})\right]}_{\text{expected log-likelihood}} - \underbrace{D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z}))}_{\text{KL from prior}}$$

The gap is exactly $D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z} \mid \mathbf{x}))$, as can be verified by expanding the KL:

$$D_{\mathrm{KL}}(q \| p(\cdot \mid \mathbf{x})) = \mathbb{E}_q\!\left[\log\frac{q(\mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})}\right] = \log p(\mathbf{x}) - \mathcal{L}(q)$$

### A.2 Closed-Form KL Between Gaussians

$$D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1) \| \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2))$$
$$= \frac{1}{2}\left[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - d + \log\frac{\det\Sigma_2}{\det\Sigma_1}\right]$$

**VAE special case** ($\Sigma_2 = I$, $\boldsymbol{\mu}_2 = \mathbf{0}$):
$$D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(\mathbf{0}, I)) = \frac{1}{2}\sum_{j=1}^d \left(\sigma_j^2 + \mu_j^2 - \log\sigma_j^2 - 1\right)$$

### A.3 Baum-Welch E-Step Derivation

The complete-data log-likelihood is:
$$\log p(\mathbf{x}, \mathbf{z} \mid \boldsymbol{\theta}) = \log p(z_1) + \sum_{t=2}^T \log p(z_t \mid z_{t-1}) + \sum_{t=1}^T \log p(x_t \mid z_t)$$

Taking expectation under $p(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta}^{(t)})$:
$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)}) = \sum_k \gamma_1(k)\log\pi_{0,k} + \sum_{t=2}^T\sum_{j,k}\xi_t(j,k)\log A_{jk} + \sum_{t,k}\gamma_t(k)\log B_k(x_t)$$

The three terms in $Q$ decouple: optimising over $\boldsymbol{\pi}_0$, $A$, and $B$ separately is equivalent to solving independent Lagrange-constrained optimisation problems, yielding the closed-form M-step updates.

### A.4 HMC Acceptance Probability Derivation

The MH ratio for the $((\mathbf{x}, \mathbf{p}) \to (\mathbf{x}', \mathbf{p}'))$ proposal from $L$ leapfrog steps:

The augmented target is $\tilde{\pi}(\mathbf{x}, \mathbf{p}) \propto \exp(-\mathcal{H}(\mathbf{x}, \mathbf{p})) = p(\mathbf{x})\, \mathcal{N}(\mathbf{p} \mid \mathbf{0}, M)$.

The leapfrog proposal is time-reversible (proposing $(\mathbf{x}', -\mathbf{p}')$ from $(\mathbf{x}', \mathbf{p}')$ returns to $(\mathbf{x}, \mathbf{p})$) and volume-preserving (Jacobian = 1). Therefore the MH ratio simplifies to:

$$\alpha = \min\!\left(1, \frac{\tilde{\pi}(\mathbf{x}', \mathbf{p}')}{\tilde{\pi}(\mathbf{x}, \mathbf{p})}\right) = \min\!\left(1, e^{-\mathcal{H}(\mathbf{x}', \mathbf{p}') + \mathcal{H}(\mathbf{x}, \mathbf{p})}\right)$$

If the Hamiltonian were exactly conserved ($\mathcal{H}(\mathbf{x}', \mathbf{p}') = \mathcal{H}(\mathbf{x}, \mathbf{p})$), every proposal would be accepted. Numerical integration errors cause the Hamiltonian to drift — the MH step corrects for this.

### A.5 Score Matching Equivalence to DSM

**Theorem (Vincent 2011).** The explicit score matching objective:
$$\mathcal{J}_\text{ESM}(\theta) = \mathbb{E}_p\!\left[\operatorname{tr}(\nabla_\mathbf{x} \mathbf{s}_\theta(\mathbf{x})) + \frac{1}{2}\lVert \mathbf{s}_\theta(\mathbf{x})\rVert^2\right]$$

equals the denoising score matching objective (up to a constant):
$$\mathcal{J}_\text{DSM}(\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\!\left[\lVert \mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}}\log p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x})\rVert^2\right]$$

**Proof sketch**: expand $\mathcal{J}_\text{ESM}$ using integration by parts; the resulting expression involves $\mathbb{E}[\nabla_\mathbf{x}\log p(\mathbf{x})]$ which can be replaced by the denoising direction $(\mathbf{x} - \tilde{\mathbf{x}})/\sigma^2$.

---

## Appendix B: Information Geometry and Natural Gradients

**Statistical manifold**: the space of distributions $\{p(\mathbf{x} \mid \boldsymbol{\theta}) : \boldsymbol{\theta} \in \Theta\}$ forms a Riemannian manifold equipped with the **Fisher information metric**:

$$g_{ij}(\boldsymbol{\theta}) = I(\boldsymbol{\theta})_{ij} = \mathbb{E}_{p(\mathbf{x} \mid \boldsymbol{\theta})}\!\left[\frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j}\right]$$

**Natural gradient descent**: instead of the Euclidean update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla_{\boldsymbol{\theta}}\mathcal{L}$, move in the Fisher metric:

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta I(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}}\mathcal{L}$$

The natural gradient is **parameter-invariant**: if you reparameterise from $\boldsymbol{\theta}$ to $\boldsymbol{\phi} = h(\boldsymbol{\theta})$, the natural gradient gives the same update direction. Standard gradient descent changes with reparameterisation.

**K-FAC (Kronecker-Factored Approximate Curvature, Martens & Grosse 2015)**: approximate the Fisher information block-diagonally by layer, then Kronecker-factoring each block:
$$I^{[l]} \approx A^{[l]} \otimes G^{[l]}$$

where $A^{[l]} = \mathbb{E}[\mathbf{a}^{[l-1]}\mathbf{a}^{[l-1]\top}]$ (activation covariance) and $G^{[l]} = \mathbb{E}[\mathbf{g}^{[l]}\mathbf{g}^{[l]\top}]$ (gradient covariance). K-FAC is used in second-order optimisation for LLMs (e.g., in Shampoo).

**For exponential families**: the Fisher information equals the Hessian of $A(\boldsymbol{\eta})$:
$$I(\boldsymbol{\eta}) = \nabla^2 A(\boldsymbol{\eta}) = \operatorname{Cov}[T(\mathbf{x})]$$

Natural gradient in the exponential family: $\tilde{\nabla}\mathcal{L} = I^{-1}\nabla\mathcal{L} = (\nabla^2 A)^{-1}\nabla\mathcal{L}$.

---

## Appendix C: Modern Variants and Extensions

### C.1 Stein Variational Gradient Descent (SVGD)

**SVGD** (Liu & Wang 2016): approximate the posterior with a set of particles $\{\mathbf{x}^{(i)}\}_{i=1}^N$ updated by:

$$\mathbf{x}^{(i)} \leftarrow \mathbf{x}^{(i)} + \epsilon \phi^*(\mathbf{x}^{(i)})$$

$$\phi^*(\mathbf{x}) = \frac{1}{N}\sum_j\!\left[k(\mathbf{x}^{(j)}, \mathbf{x}) \nabla_{\mathbf{x}^{(j)}} \log p(\mathbf{x}^{(j)}) + \nabla_{\mathbf{x}^{(j)}} k(\mathbf{x}^{(j)}, \mathbf{x})\right]$$

The first term drives particles toward high-probability regions; the second term (gradient of kernel) repels particles from each other, preventing collapse to a single mode.

SVGD bridges variational inference (optimisation-based) and MCMC (sampling-based), achieving competitive uncertainty estimation with fewer particles than MCMC.

### C.2 Dirichlet Process and Bayesian Nonparametrics

A **Dirichlet Process** (Ferguson 1973) is a distribution over distributions. $G \sim \operatorname{DP}(\alpha, G_0)$ means:

- For any finite partition $(A_1, \ldots, A_k)$ of $\Omega$: $(G(A_1), \ldots, G(A_k)) \sim \operatorname{Dir}(\alpha G_0(A_1), \ldots, \alpha G_0(A_k))$

**Stick-breaking construction** (Sethuraman 1994):
$$G = \sum_{k=1}^\infty \pi_k \delta_{\theta_k}, \quad \theta_k \sim G_0$$
$$\pi_k = V_k \prod_{j<k}(1-V_j), \quad V_k \sim \operatorname{Beta}(1, \alpha)$$

The DP generates discrete distributions almost surely — clustering structure emerges automatically. The number of clusters grows as $O(\alpha \log n)$ with data size.

**Dirichlet Process Mixture Model (DPMM)**: infinite GMM where $K$ is inferred from data. Allows the number of components to grow as more data is observed — "nonparametric" refers to the infinite parameter count.

**For AI**: the DP is a prototype for how transformers might implement infinite-capacity memory — the attention mechanism soft-assigns each token to one of many "concept slots," analogous to a DPMM with learned cluster centers.

### C.3 Conditional Independence and Causal Models

**Pearl's do-calculus**: distinguish $P(Y \mid X=x)$ (observational) from $P(Y \mid \text{do}(X=x))$ (interventional). Observational conditioning on $X$ includes all shared common causes; interventional conditioning (do) sets $X$ and removes its causal parents.

$$P(Y \mid \text{do}(X=x)) = \sum_Z P(Y \mid X=x, Z)\, P(Z) \qquad \text{(adjustment formula)}$$

**Backdoor criterion**: $Z$ satisfies the backdoor criterion for $(X \to Y)$ if:

1. $Z$ blocks all backdoor paths from $X$ to $Y$ (paths with arrows into $X$)
2. $Z$ contains no descendants of $X$

When satisfied, $P(Y \mid \text{do}(X)) = \sum_Z P(Y \mid X, Z)\, P(Z)$.

**For AI**: causal models explain why spurious correlations fail under distribution shift — the model learned $P(Y \mid X)$ but the interventional $P(Y \mid \text{do}(X))$ is different. In LLM evaluation, in-context examples can create spurious correlations (collider bias) that inflate benchmark performance without reflecting true capability.

---

## Appendix D: Quick Reference Card

```text
PROBABILISTIC MODELS — QUICK REFERENCE
════════════════════════════════════════════════════════════════════════

  BAYES' THEOREM                    EXPONENTIAL FAMILY
  p(θ|D) = p(D|θ)p(θ) / p(D)       p(x|η) = h(x) exp(η·T(x) - A(η))
  posterior ∝ likelihood × prior    ∇A(η) = E[T(x)],  ∇²A = Cov[T(x)]

  ELBO                              MEAN-FIELD CAVI
  L(q) = E_q[log p(x,z)] + H(q)    log q*_j(z_j) = E_{-j}[log p(x,z)]
  log p(x) = L(q) + KL(q||p(·|x))

  VAE TRAINING OBJECTIVE            REPARAMETERISATION
  L(φ,θ;x) = E_q[log p(x|z)]       z = μ_φ(x) + σ_φ(x) ⊙ ε, ε~N(0,I)
             - KL(q_φ(z|x) || p(z))

  METROPOLIS-HASTINGS               GP REGRESSION
  α = min(1, p(x')q(x|x')           f*|y ~ N(K_*K⁻¹y, K_** - K_*K⁻¹K*ᵀ)
            / p(x)q(x'|x))          K = K_XX + σ²_n I

  DDPM FORWARD                      HMC LEAPFROG
  q(x_t|x_0) = N(√ᾱ_t x_0,         p̃ = p + ε/2 ∇log p(x)
               (1-ᾱ_t)I)            x' = x + ε M⁻¹ p̃
  ᾱ_t = ∏ₛ(1-βₛ)                   p' = p̃ + ε/2 ∇log p(x')

  SCORE MATCHING                    FLOW MATCHING
  s(x) = ∇_x log p(x)              dx/dt = v_θ(x,t)
  DSM: train s_θ(x̃) → (x-x̃)/σ²    L_CFM = E||v_θ(x_t,t) - u_t(x_t|x₁)||²

  HMM FORWARD                       HMM BACKWARD
  α_t(k) = B_k(x_t) Σ_j α_{t-1}(j) A_{jk}  β_T(k) = 1
  p(x_{1:T}) = Σ_k α_T(k)         β_t(k) = Σ_j A_{kj} B_j(x_{t+1}) β_{t+1}(j)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix E: Probabilistic Models Taxonomy

```text
PROBABILISTIC MODEL TAXONOMY
════════════════════════════════════════════════════════════════════════

  DIRECTED (Bayesian Networks)        UNDIRECTED (MRFs)
  ─────────────────────────────────   ──────────────────────────────
  Naive Bayes                         Ising model
  Hidden Markov Model (HMM)           Boltzmann Machine
  Latent Dirichlet Allocation         Conditional Random Field (CRF)
  VAE (encoder: z→x)                  MRF for image segmentation
  Diffusion (forward: x₀→x_T)
  Autoregressive (GPT: w_{<t}→w_t)

  LATENT VARIABLE MODELS              NONPARAMETRIC MODELS
  ─────────────────────────────────   ──────────────────────────────
  GMM: discrete z                     Gaussian Process (function prior)
  FA/PPCA: continuous z, linear W     Dirichlet Process (infinite clusters)
  VAE: continuous z, nonlinear W      Gaussian Process Classifier
  HMM: sequential discrete z          Indian Buffet Process (sparse features)
  LDA: hierarchical discrete z

  APPROXIMATE INFERENCE               EXACT INFERENCE
  ─────────────────────────────────   ──────────────────────────────
  Mean-field VI (CAVI)                Conjugate models (closed form)
  Black-box VI (BBVI)                 GP regression (O(n³))
  Laplace approximation               HMM forward-backward (O(TK²))
  MCMC (MH, Gibbs, HMC, NUTS)        Factor graph BP (tree-structured)
  MC Dropout                          Kalman filter (linear-Gaussian)
  Variational autoencoders            Variable elimination (small graphs)

  GENERATIVE MODELS (DENSITY ESTIMATION)
  ─────────────────────────────────────────────────────────────────
  Normalizing Flows    — exact likelihood, invertible transforms
  VAEs                 — approximate likelihood (ELBO), latent codes
  GANs                 — implicit density, adversarial training
  Diffusion / Score    — approximate density via denoising score
  Flow Matching        — ODE vector field, deterministic sampling
  Energy-Based Models  — unnormalised density, MCMC sampling

════════════════════════════════════════════════════════════════════════
```

---

## Appendix F: Connections Between Methods

### F.1 EM ↔ Variational Inference

EM is a special case of VI where the E-step uses the **exact** posterior $q^* = p(\mathbf{z} \mid \mathbf{x}, \boldsymbol{\theta})$. When the posterior is tractable, EM and VI coincide. When the posterior is intractable (e.g., deep generative models), VI replaces the exact E-step with a parametric approximation — yielding variational EM or the VAE training algorithm.

The ELBO is:

- **EM**: maximised in $\boldsymbol{\theta}$ (M-step) and exactly tightened in $q$ (E-step)
- **VAE**: maximised jointly in $\boldsymbol{\theta}$ (decoder weights) and $\phi$ (encoder weights)

Both share the same objective; they differ in how they handle the E-step.

### F.2 Diffusion ↔ Score Matching ↔ Flow Matching

These three frameworks are mathematically unified:

**Score matching** trains $\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_\mathbf{x}\log p(\mathbf{x})$ — the gradient of the log-density.

**Diffusion** (DDPM/SDE): trains $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}$ — the noise added at step $t$. The relationship:
$$\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

Score and noise prediction are equivalent (same model, different parameterisation).

**Flow Matching**: trains $\mathbf{v}_\theta(\mathbf{x}_t, t) \approx \mathbf{u}_t(\mathbf{x}_t \mid \mathbf{x}_1)$ — the velocity field. Under VP-SDE, the probability flow ODE has velocity:
$$\mathbf{v}_t = \mathbf{f}(\mathbf{x}_t, t) - \frac{1}{2}g(t)^2 \mathbf{s}_\theta(\mathbf{x}_t, t)$$

So all three are parameterisations of the same object: the vector field that transports noise to data.

### F.3 GPs ↔ Bayesian Linear Regression ↔ Kernel Methods

**Bayesian linear regression** with feature map $\boldsymbol{\phi}(\mathbf{x})$ and prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \sigma_p^2 I)$:

Predictive distribution: $f(\mathbf{x}^*) = \mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}^*) \sim \mathcal{N}(\bar{f}^*, v^*)$.

As the feature map grows, this converges to a **Gaussian Process** with kernel $k(\mathbf{x}, \mathbf{x}') = \sigma_p^2 \boldsymbol{\phi}(\mathbf{x})^\top\boldsymbol{\phi}(\mathbf{x}')$.

**Kernel methods** (SVMs, kernel ridge regression): use the same kernel $k$ to implicitly compute inner products in a RKHS. The GP provides the **Bayesian interpretation** of kernel methods: the kernel defines a prior over functions, and kernel ridge regression is MAP inference in that prior.

**Neural network ↔ GP** (Neal 1996): a single-hidden-layer network with $H$ hidden units and i.i.d. weight initialisation converges in distribution to a GP as $H \to \infty$. The kernel is determined by the activation function and architecture.

### F.4 HMMs ↔ State Space Models ↔ Transformers

| Model          | State                                      | Transition                                                 | Emission                                             |
| -------------- | ------------------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------- |
| HMM            | Discrete $z_t \in [K]$                     | $A_{jk} = p(z_t \mid z_{t-1})$                             | $B_k(x_t) = p(x_t \mid z_t)$                         |
| Kalman Filter  | Continuous $\mathbf{z}_t \in \mathbb{R}^d$ | $\mathbf{z}_t = F\mathbf{z}_{t-1} + \boldsymbol{\epsilon}$ | $\mathbf{x}_t = H\mathbf{z}_t + \boldsymbol{\delta}$ |
| SSM (S4/Mamba) | Continuous, learned $A$, $B$, $C$          | $\mathbf{h}_t = \bar{A}\mathbf{h}_{t-1} + \bar{B}x_t$      | $y_t = C\mathbf{h}_t$                                |
| Transformer    | Attention over all $t' \leq t$             | Softmax-weighted sum (global)                              | Linear output projection                             |

**Key insight**: HMMs, SSMs, and transformers are all models for sequential data with different inductive biases. HMMs assume short-range Markov dependence; SSMs assume linear dynamics (with selection mechanism in Mamba); transformers assume all-to-all soft attention (quadratic cost, captures long-range).

---

## Appendix G: Practical Implementation Notes

### G.1 Numerical Stability

**Log-sum-exp trick** (for normalisation):
$$\log\sum_k e^{a_k} = a_{\max} + \log\sum_k e^{a_k - a_{\max}}$$

Apply in all forward passes, HMM computation, and softmax to prevent underflow/overflow.

**MCMC in log-space**: always compute acceptance ratio $\log\alpha = \log p(\mathbf{x}') - \log p(\mathbf{x}) + \log q(\mathbf{x} \mid \mathbf{x}') - \log q(\mathbf{x}' \mid \mathbf{x})$. Accept if $\log\mathcal{U}(0,1) < \log\alpha$.

**Cholesky for GP covariance**: never invert $K$ directly. Use Cholesky $K = LL^\top$; solve $L\mathbf{v} = \mathbf{y}$ (forward substitution) and compute $\log\det K = 2\sum_i \log L_{ii}$.

### G.2 VAE Training Tips

1. **Initialise encoder variance output to 0** (so initial $\sigma = 1$, matching prior)
2. **Use KL annealing**: weight $\beta(t) = \min(1, t/T_\text{warmup})$ on KL term
3. **Free bits**: $\mathcal{L} = \mathbb{E}[\log p(\mathbf{x} \mid \mathbf{z})] - \sum_j \max(\lambda, D_{\mathrm{KL}}(q_j \| p_j))$ with $\lambda=0.5$ bits per dimension
4. **Monitor**: KL per dimension (should not be zero), reconstruction loss, and latent space PCA variance

### G.3 Diffusion Model Noise Schedules

| Schedule                 | $\beta_t$                                                                               | Suitable for                              |
| ------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------- |
| Linear                   | $\beta_t = \beta_1 + (\beta_T - \beta_1)\frac{t-1}{T-1}$                                | DDPM original; suboptimal                 |
| Cosine                   | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$         | Improved DDPM (Nichol & Dhariwal 2021)    |
| Sigmoid                  | $\beta_t = \beta(\operatorname{sigmoid}(-5 + 10t/T))$                                   | Better for high-resolution                |
| EDM (Karras et al. 2022) | Continuous $\sigma(t) = t$, $\mathbf{x}_t = \mathbf{x}_0 + \sigma\boldsymbol{\epsilon}$ | State-of-the-art; VE-SDE parameterisation |

For **flow matching** (§14), the noise schedule is replaced by a linear interpolation coefficient $\alpha_t = t$ — much simpler and avoids schedule engineering entirely.

---

## Appendix H: Connections to Neural Network Training

### H.1 Cross-Entropy as NLL of Categorical Exponential Family

The cross-entropy loss for a classifier with softmax output is:
$$\mathcal{L}_\text{CE} = -\frac{1}{n}\sum_{i=1}^n \log p_\theta(y^{(i)} \mid \mathbf{x}^{(i)}) = -\frac{1}{n}\sum_{i=1}^n \log\frac{e^{\mathbf{W}_{y^{(i)}}^\top f(\mathbf{x}^{(i)})}}{\sum_k e^{\mathbf{W}_k^\top f(\mathbf{x}^{(i)})}}$$

This is exactly the **negative log-likelihood of a Categorical distribution** in the exponential family. The logits $\mathbf{W}_k^\top f(\mathbf{x})$ are the natural parameters $\eta_k$; the log-sum-exp is the log-partition function $A(\boldsymbol{\eta})$; the one-hot labels are the sufficient statistics $T(y)$.

Maximum likelihood training of a softmax classifier is **fitting an exponential family** by minimising the KL divergence between the empirical distribution $\hat{p}$ and the model $p_\theta$:
$$D_{\mathrm{KL}}(\hat{p} \| p_\theta) = -H(\hat{p}) + \mathcal{L}_\text{CE}$$

Since $H(\hat{p})$ is constant, minimising CE ≡ minimising KL.

### H.2 Dropout Rate as Variational Parameter

In Gal & Ghahramani (2016)'s variational interpretation:

- **Dropout rate $p$** controls the mixture probability of the zero component
- **Learning rate** corresponds to the prior precision $\tau$
- **Weight decay** corresponds to $\tau p / (2n)$ (prior on non-zero weights)
- **Number of MC samples** at test time controls the approximation quality of the posterior predictive

The optimal dropout rate is **data-dependent** and should be tuned per layer to match the posterior uncertainty level. In practice, $p=0.1$ (10% dropout) is often too low for uncertainty estimation; $p=0.5$ provides better uncertainty quantification but worse test accuracy.

### H.3 L2 Regularisation as Gaussian MAP

**Weight decay** adds $\frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert^2$ to the loss. This is equivalent to MAP estimation with a Gaussian prior:

$$\mathcal{L}_\text{MAP} = -\log p(\mathcal{D} \mid \boldsymbol{\theta}) - \log p(\boldsymbol{\theta}) = \mathcal{L}_\text{MLE} + \frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert^2$$

with $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{0}, \frac{1}{\lambda}I)$.

**AdamW** (Loshchilov & Hutter 2019) implements **decoupled weight decay** — $\boldsymbol{\theta} \leftarrow (1 - \eta\lambda)\boldsymbol{\theta} - \eta\nabla\mathcal{L}$ — which correctly implements L2 regularisation for adaptive optimisers (unlike adding $\lambda\boldsymbol{\theta}$ to the gradient, which conflates regularisation with adaptive learning rates).

**L1 regularisation** ($\lambda\lVert\boldsymbol{\theta}\rVert_1$) corresponds to a Laplace prior: $p(\theta) \propto e^{-\lambda\lvert\theta\rvert}$. This induces sparsity because the Laplace prior has a sharp peak at zero — the MAP estimate is pushed exactly to zero for small parameters.

### H.4 Batch Normalisation as Approximate Posterior Normalisation

BatchNorm normalises activations to zero mean and unit variance within a mini-batch. This can be interpreted as:

1. **Whitening** the input distribution to each layer (approximate Gaussian normalisation)
2. **Removing internal covariate shift** — preventing the input distribution of each layer from changing during training
3. **Approximate posterior re-centering**: normalise activations so they behave like samples from a standard distribution before the next layer's transformation

The **trainable parameters** $\gamma$ and $\beta$ re-scale and re-shift after normalisation — they learn the optimal scale/location of the normalised distribution, which corresponds to the **posterior mean and variance** of the normalised activations.

---

_This completes the Probabilistic Models reference. For interactive exploration of all concepts, see [theory.ipynb](theory.ipynb). For practice problems, see [exercises.ipynb](exercises.ipynb)._

---

## Appendix I: Notation Summary for This Section

| Symbol                                            | Meaning                                       | First Use    |
| ------------------------------------------------- | --------------------------------------------- | ------------ |
| $p(\boldsymbol{\theta} \mid \mathcal{D})$         | Posterior over parameters                     | §2.2         |
| $\boldsymbol{\eta}$                               | Natural parameters (exponential family)       | §2.3         |
| $T(\mathbf{x})$                                   | Sufficient statistics                         | §2.3         |
| $A(\boldsymbol{\eta})$                            | Log-partition function                        | §2.3         |
| $\mathcal{L}(q, \boldsymbol{\theta})$             | Evidence Lower Bound (ELBO)                   | §6.1         |
| $q(\mathbf{z})$                                   | Variational distribution                      | §6.1         |
| $r_{ik}$                                          | Responsibility of component $k$ for point $i$ | §4.1         |
| $\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi$ | VAE encoder mean and std                      | §7.1         |
| $\mathcal{H}(\mathbf{x}, \mathbf{p})$             | Hamiltonian (HMC)                             | §8.3         |
| $\alpha_t(k), \beta_t(k)$                         | HMM forward/backward variables                | §10.2        |
| $\gamma_t(k), \xi_t(j,k)$                         | HMM state marginals                           | §10.2        |
| $k(\mathbf{x}, \mathbf{x}')$                      | Kernel (covariance) function                  | §9.1         |
| $\mathbf{s}_\theta(\mathbf{x}, t)$                | Score network                                 | §13.1        |
| $\mathbf{v}_\theta(\mathbf{x}, t)$                | Flow matching vector field                    | §14.1        |
| $\bar{\alpha}_t$                                  | DDPM cumulative noise schedule                | §13.4        |
| $C$                                               | Latent concept (ICL Bayesian framework)       | §15.1        |
| $G \sim \operatorname{DP}(\alpha, G_0)$           | Dirichlet process                             | App. C.2     |
| $I(\boldsymbol{\eta})$                            | Fisher information matrix                     | §2.3, App. B |

**Notation follows `docs/NOTATION_GUIDE.md` throughout. Distributions in calligraphic ($\mathcal{N}, \operatorname{Dir}$), random variables in uppercase italic ($X, Z$), vectors in bold lowercase ($\mathbf{x}, \boldsymbol{\mu}$), matrices in uppercase ($W, K, A$), KL divergence as $D_{\mathrm{KL}}(p \| q)$ with double-bar.**
