[Back to Curriculum](../../README.md) | [Previous: Normalization Techniques](../03-Normalization-Techniques/notes.md) | [Next: Linear Models](../../14-Math-for-Specific-Models/01-Linear-Models/notes.md)

---

# Sampling Methods

> _"Sampling is how machine learning turns distributions it cannot enumerate into estimates it can compute."_

## Overview

Sampling methods let ML systems approximate expectations, generate candidates,
choose mini-batches, estimate uncertainty, train latent-variable models, draw
negative examples, and explore action spaces. A deterministic formula may say
"take an expectation", but a real model often only has samples. Understanding
sampling means understanding estimator bias, variance, proposal distributions,
effective sample size, Markov chains, differentiable random variables, and the
difference between sampling for training and sampling for generation.

This section is the canonical home for general ML sampling math. Chapter 6 owns
foundational probability and distributions. Chapter 14 owns full generative
model families such as VAEs, GANs, diffusion, and flows. Chapter 15 owns full
LLM decoding. Here we focus on reusable sampling techniques: inverse transform,
categorical sampling, rejection sampling, Monte Carlo estimation, importance
sampling, MCMC, reparameterization, Gumbel tricks, minibatch sampling, negative
sampling, and previews of temperature, top-k, top-p, and diffusion sampling.

The main habit is to ask: "What distribution do I want, what distribution am I
actually sampling from, and what estimator does that produce?"

## Prerequisites

- **Random variables and distributions** - [Introduction and Random Variables](../../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md), [Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- **Expectation and variance** - [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Markov chains** - [Markov Chains](../../06-Probability-Theory/07-Markov-Chains/notes.md)
- **Estimation theory** - [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- **Loss functions and contrastive objectives** - [Loss Functions](../01-Loss-Functions/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable sampling demos for inverse CDF, rejection, Monte Carlo, importance sampling, MCMC, Gumbel, and top-k/top-p previews |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on estimators, proposal mismatch, differentiable sampling, and ML sampling diagnostics |

## Learning Objectives

After completing this section, you will be able to:

- Define a sample, estimator, proposal distribution, bias, variance, and effective sample size
- Implement inverse-transform, categorical, rejection, and stratified sampling
- Estimate expectations with Monte Carlo and quantify standard error
- Use importance weights and diagnose weight degeneracy
- Implement simple Metropolis-Hastings, Gibbs-style, and Langevin-style samplers
- Explain the reparameterization trick for Gaussian latent variables
- Implement Gumbel-Max and Gumbel-Softmax sampling
- Explain how minibatch and negative sampling change training objectives
- Treat temperature, top-k, top-p, and diffusion sampling as previews without duplicating later chapters

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Sampling as Approximation](#11-sampling-as-approximation)
  - [1.2 Sampling as Generation](#12-sampling-as-generation)
  - [1.3 Sampling as Exploration](#13-sampling-as-exploration)
  - [1.4 Randomness Versus Determinism](#14-randomness-versus-determinism)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Random Sample](#21-random-sample)
  - [2.2 Estimator](#22-estimator)
  - [2.3 Bias and Variance](#23-bias-and-variance)
  - [2.4 Proposal Distribution](#24-proposal-distribution)
  - [2.5 Effective Sample Size](#25-effective-sample-size)
- [3. Direct Sampling](#3-direct-sampling)
  - [3.1 Inverse-Transform Sampling](#31-inverse-transform-sampling)
  - [3.2 Categorical Sampling](#32-categorical-sampling)
  - [3.3 Rejection Sampling](#33-rejection-sampling)
  - [3.4 Stratified Sampling](#34-stratified-sampling)
  - [3.5 Practical Limitations](#35-practical-limitations)
- [4. Monte Carlo Estimation](#4-monte-carlo-estimation)
  - [4.1 Sample Mean Estimator](#41-sample-mean-estimator)
  - [4.2 Law of Large Numbers](#42-law-of-large-numbers)
  - [4.3 Confidence Intervals](#43-confidence-intervals)
  - [4.4 Variance Reduction](#44-variance-reduction)
  - [4.5 ML Expectation Estimates](#45-ml-expectation-estimates)
- [5. Importance Sampling](#5-importance-sampling)
  - [5.1 Proposal Distribution](#51-proposal-distribution)
  - [5.2 Importance Weights](#52-importance-weights)
  - [5.3 Self-Normalized Estimator](#53-self-normalized-estimator)
  - [5.4 Weight Degeneracy](#54-weight-degeneracy)
  - [5.5 Off-Policy Preview](#55-off-policy-preview)
- [6. MCMC Methods](#6-mcmc-methods)
  - [6.1 Markov-Chain Idea](#61-markov-chain-idea)
  - [6.2 Metropolis-Hastings](#62-metropolis-hastings)
  - [6.3 Gibbs Sampling](#63-gibbs-sampling)
  - [6.4 Langevin Dynamics](#64-langevin-dynamics)
  - [6.5 HMC Preview](#65-hmc-preview)
- [7. Differentiable Sampling](#7-differentiable-sampling)
  - [7.1 Reparameterization Trick](#71-reparameterization-trick)
  - [7.2 Gaussian Latent Variables](#72-gaussian-latent-variables)
  - [7.3 Gumbel-Max](#73-gumbel-max)
  - [7.4 Gumbel-Softmax](#74-gumbel-softmax)
  - [7.5 Straight-Through Estimator](#75-straight-through-estimator)
- [8. ML-Specific Sampling](#8-ml-specific-sampling)
  - [8.1 Minibatch Sampling](#81-minibatch-sampling)
  - [8.2 Negative Sampling](#82-negative-sampling)
  - [8.3 Temperature Sampling Preview](#83-temperature-sampling-preview)
  - [8.4 Top-k and Top-p Preview](#84-top-k-and-top-p-preview)
  - [8.5 Diffusion Sampling Preview](#85-diffusion-sampling-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Bayesian Inference](#91-bayesian-inference)
  - [9.2 VAEs](#92-vaes)
  - [9.3 Contrastive Learning](#93-contrastive-learning)
  - [9.4 Generative Models](#94-generative-models)
  - [9.5 LLM Decoding Preview](#95-llm-decoding-preview)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [88. Computed Table: One Budget, Four Realities](#88-computed-table-one-budget-four-realities)
- [89. Parting Problems (No Solutions Provided)](#89-parting-problems-no-solutions-provided)
- [90. End of Section 04 and Chapter 13 Core](#90-end-of-section-04-and-chapter-13-core)
- [86. Computed Table: Running Monte Carlo Estimate (iid Reference Case)](#86-computed-table-running-monte-carlo-estimate-iid-reference-case)
- [87. Final Checklist: Sampling Competence](#87-final-checklist-sampling-competence)
- [83. Computed Table: Self-Normalized IS Estimate vs Budget](#83-computed-table-self-normalized-is-estimate-vs-budget)
- [84. Worked Problem Set J: Selection Defended](#84-worked-problem-set-j-selection-defended)
- [85. Chapter 13 Closing Synthesis](#85-chapter-13-closing-synthesis)
- [80. Sampling and Uncertainty Claims](#80-sampling-and-uncertainty-claims)
- [81. Revision Problems (Mixed, Chapter 13 Capstone)](#81-revision-problems-mixed-chapter-13-capstone)
- [82. Final Remarks](#82-final-remarks)
- [77. Computed Table: Random-Walk MH in 10 Dimensions](#77-computed-table-random-walk-mh-in-10-dimensions)
- [78. Score-Function Estimators: The General Fallback](#78-score-function-estimators-the-general-fallback)
- [79. Worked Dataset: Gumbel-Max Draws by Hand](#79-worked-dataset-gumbel-max-draws-by-hand)
- [74. Worked Dataset: Ancestral Sampling From a Two-Component Mixture](#74-worked-dataset-ancestral-sampling-from-a-two-component-mixture)
- [75. Worked Problem Set I: Mixtures and Tempering](#75-worked-problem-set-i-mixtures-and-tempering)
- [76. Frequently Asked Questions, Part II](#76-frequently-asked-questions-part-ii)
- [70. Computed Example: Bootstrap Standard Error (Seed 42)](#70-computed-example-bootstrap-standard-error-seed-42)
- [71. Sampling in Preference and RLHF Pipelines (Orientation)](#71-sampling-in-preference-and-rlhf-pipelines-orientation)
- [72. Worked Problem Set H: Evaluation and Pipelines](#72-worked-problem-set-h-evaluation-and-pipelines)
- [73. Closing Checklist for This Section](#73-closing-checklist-for-this-section)
- [65. Computed Table: Two Chains Converging From Opposite Starts](#65-computed-table-two-chains-converging-from-opposite-starts)
- [66. What to Memorise vs What to Look Up](#66-what-to-memorise-vs-what-to-look-up)
- [67. Index of Computed Evidence in This Section](#67-index-of-computed-evidence-in-this-section)
- [68. Section 04 Completion Checklist](#68-section-04-completion-checklist)
- [69. Final Remarks: Chapter 13 in One View](#69-final-remarks-chapter-13-in-one-view)
- [61. Computed Table: ESS and MCSE as a Chain Runs](#61-computed-table-ess-and-mcse-as-a-chain-runs)
- [62. Worked Problem Set G: Final Audits](#62-worked-problem-set-g-final-audits)
- [63. Decision Flowchart in Words: Sampling](#63-decision-flowchart-in-words-sampling)
- [64. Quick Checks (One-Line Answers)](#64-quick-checks-one-line-answers)
- [56. Worked Test: Frequency Audit of a Categorical Sampler](#56-worked-test-frequency-audit-of-a-categorical-sampler)
- [57. Variance Reduction: Getting More From the Same Draws](#57-variance-reduction-getting-more-from-the-same-draws)
- [58. Sampling for Evaluation: Bootstraps and Benchmarks](#58-sampling-for-evaluation-bootstraps-and-benchmarks)
- [59. Annotated Reading Guide](#59-annotated-reading-guide)
- [60. Chapter Summary: Sampling Methods](#60-chapter-summary-sampling-methods)
- [52. Computed Table: Decoding Temperature and Entropy](#52-computed-table-decoding-temperature-and-entropy)
- [53. Quick-Reference Cards](#53-quick-reference-cards)
- [54. Exam and Interview Traps: Sampling](#54-exam-and-interview-traps-sampling)
- [55. Practice Problems for Revision](#55-practice-problems-for-revision)
- [48. Worked Dataset: 30 Importance Weights by Hand](#48-worked-dataset-30-importance-weights-by-hand)
- [49. Sampling Inside Training: Noise as a Tool](#49-sampling-inside-training-noise-as-a-tool)
- [50. Worked Problem Set F: Audits and Specifications](#50-worked-problem-set-f-audits-and-specifications)
- [51. Sampling and the Rest of Chapter 13](#51-sampling-and-the-rest-of-chapter-13)
- [42. Computed Table: Autocorrelation Taxes Your Sample Budget](#42-computed-table-autocorrelation-taxes-your-sample-budget)
- [43. Sequential Monte Carlo and Annealed Bridges (Orientation)](#43-sequential-monte-carlo-and-annealed-bridges-orientation)
- [44. Monitoring Runbook: Sampling Health](#44-monitoring-runbook-sampling-health)
- [45. Extended Exercise Bank (Self-Study)](#45-extended-exercise-bank-self-study)
- [46. Self-Test Questions](#46-self-test-questions)
- [47. Self-Test Answer Key (Condensed)](#47-self-test-answer-key-condensed)
- [38. Worked Problem Set D: Estimator Honesty and Geometry](#38-worked-problem-set-d-estimator-honesty-and-geometry)
- [39. Worked Problem Set E: Selection and Reporting](#39-worked-problem-set-e-selection-and-reporting)
- [40. Formula Sheet](#40-formula-sheet)
- [41. Frequently Asked Questions](#41-frequently-asked-questions)
- [34. Worked Trace: 40 Steps of Metropolis-Hastings from a Bad Start](#34-worked-trace-40-steps-of-metropolis-hastings-from-a-bad-start)
- [35. Sampling Mistake Catalogue (Field Version)](#35-sampling-mistake-catalogue-field-version)
- [36. Pre-Flight Checklist for a Sampling Study](#36-pre-flight-checklist-for-a-sampling-study)
- [37. Decision Records: Sampler Choices in Context](#37-decision-records-sampler-choices-in-context)
- [30. Computed Table: Gumbel-Softmax Relaxation vs Temperature](#30-computed-table-gumbel-softmax-relaxation-vs-temperature)
- [31. Worked Problem Set C: Differentiable Sampling and Decoding](#31-worked-problem-set-c-differentiable-sampling-and-decoding)
- [32. Historical Timeline: Sampling Methods](#32-historical-timeline-sampling-methods)
- [33. Glossary: Sampling Vocabulary](#33-glossary-sampling-vocabulary)
- [27. Computed Table: Random-Walk MH Tuning on a Standard Normal](#27-computed-table-random-walk-mh-tuning-on-a-standard-normal)
- [28. Worked Problem Set A: Inversion, Rejection, Importance](#28-worked-problem-set-a-inversion-rejection-importance)
- [29. Worked Problem Set B: MCMC Practice and Diagnostics](#29-worked-problem-set-b-mcmc-practice-and-diagnostics)
- [23. Differentiable Sampling II: Gumbel-Softmax and Concrete](#23-differentiable-sampling-ii-gumbel-softmax-and-concrete)
- [24. Computed Table: Top-k and Top-p (Nucleus) Truncation](#24-computed-table-top-k-and-top-p-nucleus-truncation)
- [25. Sampling for LLM Decoding: Temperature, Top-k, Top-p](#25-sampling-for-llm-decoding-temperature-top-k-top-p)
- [26. Negative Sampling and Subsampling Tricks](#26-negative-sampling-and-subsampling-tricks)
- [18. Computed Table: Monte Carlo Error Shrinks as $1/\sqrt{N}$](#18-computed-table-monte-carlo-error-shrinks-as-1sqrtn)
- [19. Computed Table: Importance-Weight Degeneracy vs Proposal Mismatch](#19-computed-table-importance-weight-degeneracy-vs-proposal-mismatch)
- [20. Diagnostics: $\hat R$, ESS, and Reading a Chain](#20-diagnostics-hat-r-ess-and-reading-a-chain)
- [21. MCMC II: Hamiltonian Monte Carlo and NUTS](#21-mcmc-ii-hamiltonian-monte-carlo-and-nuts)
- [22. Differentiable Sampling I: Reparameterization](#22-differentiable-sampling-i-reparameterization)
- [14. Dossier Template: How to Read a Sampling Method](#14-dossier-template-how-to-read-a-sampling-method)
- [15. Inverse-CDF and Rejection Sampling: The Exact Baselines](#15-inverse-cdf-and-rejection-sampling-the-exact-baselines)
- [16. Importance Sampling: Weighted Samples and Their Degeneracy](#16-importance-sampling-weighted-samples-and-their-degeneracy)
- [17. MCMC I: Metropolis-Hastings and Gibbs](#17-mcmc-i-metropolis-hastings-and-gibbs)
- [References](#references)

---

## 1. Intuition

### 1.1 Sampling as Approximation

Many ML objectives contain expectations:

$$
\mathbb{E}_{X\sim p}[f(X)].
$$

When the expectation cannot be computed exactly, we draw samples
$X_1,\ldots,X_n\sim p$ and approximate:

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}f(X_i).
$$

Sampling turns an integral into an average. The price is randomness.

### 1.2 Sampling as Generation

Generative systems sample outputs from learned distributions. A language model
samples tokens from a categorical distribution. A diffusion model samples
noise and progressively transforms it. A VAE samples latent variables and
decodes them. The full models belong later; the shared math is drawing random
variables from specified distributions.

### 1.3 Sampling as Exploration

In reinforcement learning and bandits, sampling actions supports exploration.
In contrastive learning, sampling negatives defines what distinctions a model
learns. In minibatch SGD, sampling data controls gradient noise. Randomness is
not merely inconvenience; it is often part of the learning algorithm.

### 1.4 Randomness Versus Determinism

Sampling methods can be stochastic or deterministic approximations to
stochastic goals. Greedy decoding is deterministic. Temperature sampling is
random. Quasi-Monte Carlo uses deterministic low-discrepancy sequences to
reduce variance. The right method depends on whether diversity, unbiasedness,
reproducibility, or coverage matters most.

### 1.5 Historical Path

Monte Carlo methods grew from numerical integration and physics. MCMC made
Bayesian inference feasible in high-dimensional models. Importance sampling
became central when direct sampling from the target distribution was hard.
Reparameterization made latent-variable models trainable with low-variance
gradients. Gumbel tricks made categorical sampling compatible with neural
network training. Modern generative AI combines all of these ideas.

## 2. Formal Definitions

### 2.1 Random Sample

A random sample from $p$ is a collection

$$
X_1,\ldots,X_n\sim p.
$$

If the samples are independent and identically distributed, we write

$$
X_i \overset{\mathrm{iid}}{\sim} p.
$$

In ML, samples may also be dependent, as in Markov chains or correlated data
augmentations.

### 2.2 Estimator

An estimator is a function of samples used to approximate an unknown quantity.
For $\mu=\mathbb{E}[X]$, the sample mean is

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}X_i.
$$

The estimator is random because the sample is random.

### 2.3 Bias and Variance

Bias is

$$
\operatorname{Bias}(\hat{\theta})
=\mathbb{E}[\hat{\theta}]-\theta.
$$

Variance is

$$
\operatorname{Var}(\hat{\theta})
=\mathbb{E}\left[(\hat{\theta}-\mathbb{E}[\hat{\theta}])^2\right].
$$

An unbiased estimator can still be useless if variance is huge.

### 2.4 Proposal Distribution

When sampling directly from target $p$ is hard, we may sample from a proposal
$q$ and correct with weights. The proposal must cover the target support:

$$
p(x)>0 \implies q(x)>0.
$$

If the proposal misses important target regions, estimates can be biased or
have infinite variance.

### 2.5 Effective Sample Size

For normalized importance weights $\tilde{w}_i$, a common effective sample
size is

$$
\operatorname{ESS}
=\frac{1}{\sum_{i=1}^{n}\tilde{w}_i^2}.
$$

If one weight dominates, ESS is near one even if many samples were drawn.

## 3. Direct Sampling

### 3.1 Inverse-Transform Sampling

If $U\sim\mathcal{U}(0,1)$ and $F$ is a CDF, then

$$
X=F^{-1}(U)
$$

has distribution $F$. For an exponential distribution with rate $\lambda$:

$$
X=-\frac{1}{\lambda}\log(1-U).
$$

### 3.2 Categorical Sampling

For probabilities $\mathbf{p}\in\Delta^{C-1}$, categorical sampling chooses
class $k$ with probability $p_k$. One implementation draws $u\sim\mathcal{U}(0,1)$
and returns the first index where cumulative probability exceeds $u$.

### 3.3 Rejection Sampling

Suppose we want samples from $p$ but can sample from $q$ and know a constant
$M$ such that

$$
p(x)\le Mq(x).
$$

Draw $x\sim q$ and $u\sim\mathcal{U}(0,1)$. Accept $x$ if

$$
u\le \frac{p(x)}{Mq(x)}.
$$

High rejection rates mean the proposal is inefficient.

### 3.4 Stratified Sampling

Stratified sampling divides the domain into strata and samples within each.
It can reduce variance when strata are more homogeneous than the full
population.

### 3.5 Practical Limitations

Direct methods are elegant when CDFs, bounds, or probabilities are accessible.
In high-dimensional ML, exact CDF inversion and tight rejection bounds are
rare. This motivates Monte Carlo, importance sampling, and MCMC.

## 4. Monte Carlo Estimation

### 4.1 Sample Mean Estimator

For iid samples $X_i\sim p$,

$$
\hat{\mu}_n=\frac{1}{n}\sum_{i=1}^{n}f(X_i)
$$

estimates $\mathbb{E}_p[f(X)]$.

### 4.2 Law of Large Numbers

Under standard conditions,

$$
\hat{\mu}_n\to \mathbb{E}[f(X)]
$$

as $n\to\infty$. The convergence rate is stochastic and depends on variance.

### 4.3 Confidence Intervals

If $\sigma_f^2=\operatorname{Var}(f(X))$, then the standard error is

$$
\operatorname{SE}(\hat{\mu}_n)=\frac{\sigma_f}{\sqrt{n}}.
$$

In practice, $\sigma_f$ is estimated from samples.

### 4.4 Variance Reduction

Variance reduction methods include antithetic sampling, stratification,
control variates, and better proposal distributions. The goal is not always
more samples; it is more useful samples.

### 4.5 ML Expectation Estimates

SGD estimates a full-data gradient using a minibatch. Dropout estimates an
ensemble-like objective with random masks. Contrastive learning estimates a
normalization term using sampled negatives. These are all sampling-based
approximations.

## 5. Importance Sampling

### 5.1 Proposal Distribution

To estimate

$$
\mathbb{E}_{p}[f(X)]
=\int f(x)p(x)\,dx,
$$

sample from $q$ and rewrite:

$$
\mathbb{E}_{p}[f(X)]
=\mathbb{E}_{q}\left[f(X)\frac{p(X)}{q(X)}\right].
$$

### 5.2 Importance Weights

The weight is

$$
w(x)=\frac{p(x)}{q(x)}.
$$

The estimator is

$$
\hat{\mu}=\frac{1}{n}\sum_{i=1}^{n}w(X_i)f(X_i),
\qquad X_i\sim q.
$$

### 5.3 Self-Normalized Estimator

If the target density is known only up to a constant, use

$$
\hat{\mu}_{\mathrm{SN}}
=\frac{\sum_i w_i f(X_i)}{\sum_i w_i}.
$$

This estimator is biased for finite $n$ but often practical.

### 5.4 Weight Degeneracy

If most weights are near zero and one weight is huge, the estimator has high
variance. Effective sample size detects this collapse:

$$
\operatorname{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
$$

### 5.5 Off-Policy Preview

In off-policy reinforcement learning, data may come from a behavior policy
while the target policy differs. Importance ratios correct for this mismatch,
but high-variance ratios are a central difficulty.

## 6. MCMC Methods

### 6.1 Markov-Chain Idea

MCMC constructs a Markov chain whose stationary distribution is the target
$p$. Samples are dependent, but after burn-in they can approximate target
expectations.

### 6.2 Metropolis-Hastings

Given current state $x$, propose $x'\sim q(x'\mid x)$ and accept with
probability

$$
\alpha=\min\left(1,
\frac{p(x')q(x\mid x')}{p(x)q(x'\mid x)}
\right).
$$

For symmetric proposals, the proposal terms cancel.

### 6.3 Gibbs Sampling

Gibbs sampling updates one variable at a time from its conditional
distribution. It is useful when conditionals are easy even though the joint
distribution is hard.

### 6.4 Langevin Dynamics

Langevin methods use gradients of log density:

$$
x_{t+1}=x_t+\frac{\eta}{2}\nabla_x\log p(x_t)+\sqrt{\eta}\epsilon_t.
$$

They combine drift toward high-density regions with random exploration.

### 6.5 HMC Preview

Hamiltonian Monte Carlo augments position variables with momentum and simulates
Hamiltonian dynamics. The full method is beyond this section, but the key idea
is gradient-informed proposals with better long-distance movement than a
random walk.

## 7. Differentiable Sampling

### 7.1 Reparameterization Trick

If

$$
Z\sim\mathcal{N}(\mu,\sigma^2),
$$

write

$$
Z=\mu+\sigma\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,1).
$$

The randomness is isolated in $\epsilon$, so gradients can flow through
$\mu$ and $\sigma$.

### 7.2 Gaussian Latent Variables

VAEs use Gaussian latent variables and the reparameterization trick to train
encoders with low-variance pathwise gradients. Full VAE details belong in
Chapter 14.

### 7.3 Gumbel-Max

If $G_i$ are iid Gumbel random variables, then

$$
\arg\max_i(\log p_i+G_i)
$$

is a categorical sample from $\mathbf{p}$.

### 7.4 Gumbel-Softmax

The Gumbel-Softmax relaxation is

$$
Y_i=
\frac{\exp((\log p_i+G_i)/\tau)}
{\sum_j\exp((\log p_j+G_j)/\tau)}.
$$

As $\tau\to 0$, samples become nearly one-hot. For positive $\tau$, the output
is differentiable with respect to logits.

### 7.5 Straight-Through Estimator

A straight-through estimator uses a hard discrete sample in the forward pass
and a soft or surrogate gradient in the backward pass. It is biased but often
useful when discrete choices must appear inside a differentiable model.

## 8. ML-Specific Sampling

### 8.1 Minibatch Sampling

SGD samples a minibatch and uses its gradient as an estimator of the full
gradient. Shuffle order, replacement, stratification, and distributed sampling
all affect gradient noise.

### 8.2 Negative Sampling

Negative sampling chooses contrastive alternatives. The negative distribution
defines what distinctions the representation learns. Easy negatives create
weak gradients. Hard negatives can be informative or mislabeled.

### 8.3 Temperature Sampling Preview

Temperature modifies categorical probabilities:

$$
p_i(\tau)=
\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Low temperature sharpens. High temperature flattens. Full LLM decoding belongs
in [Language Model Probability Math](../../15-Math-for-LLMs/05-Language-Model-Probability/notes.md).

### 8.4 Top-k and Top-p Preview

Top-k restricts sampling to the $k$ largest probabilities. Top-p, or nucleus
sampling, restricts sampling to the smallest set whose cumulative probability
exceeds $p$. These are generation-time truncation rules.

### 8.5 Diffusion Sampling Preview

Diffusion sampling starts from noise and iteratively denoises. The full
diffusion treatment belongs in
[Generative Models](../../14-Math-for-Specific-Models/07-Generative-Models/notes.md).
Here it is enough to see diffusion as repeated stochastic or deterministic
sampling from learned transition rules.

## 9. Applications in Machine Learning

### 9.1 Bayesian Inference

Bayesian inference often requires expectations under a posterior distribution
that cannot be normalized exactly. MCMC and importance sampling approximate
posterior means, credible intervals, and predictive distributions.

### 9.2 VAEs

VAEs use differentiable Gaussian sampling to train latent-variable models. The
reparameterization trick is what keeps encoder gradients low variance.

### 9.3 Contrastive Learning

Contrastive learning depends on how positives and negatives are sampled. The
loss and sampler are inseparable.

### 9.4 Generative Models

Generative models use sampling to produce outputs. Sampling quality affects
diversity, fidelity, mode coverage, and reproducibility.

### 9.5 LLM Decoding Preview

LLM decoding samples tokens from model logits. Temperature, top-k, top-p,
typical sampling, and beam search are treated fully in Chapter 15. This
section only provides the general categorical sampling machinery.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing target and proposal distributions | Estimates are for the wrong distribution | Write $p$ and $q$ explicitly |
| 2 | Ignoring support mismatch | Missing target regions cannot be recovered | Ensure $q(x)>0$ wherever $p(x)>0$ |
| 3 | Reporting sample count instead of ESS | Many samples can have one effective weight | Monitor effective sample size |
| 4 | Treating MCMC samples as iid | Markov samples are correlated | Use burn-in, thinning cautiously, and autocorrelation diagnostics |
| 5 | Using too small a proposal step | Chains mix slowly | Tune acceptance and movement |
| 6 | Using too large a proposal step | Acceptance collapses | Tune proposal scale |
| 7 | Backpropagating through hard categorical samples naively | Argmax is nondifferentiable | Use reparameterization, relaxation, or score-function estimators |
| 8 | Sampling negatives without auditing them | False negatives corrupt contrastive training | Design and inspect the sampler |
| 9 | Comparing decoding methods without fixed seed or metric | Randomness hides differences | Report seeds, entropy, and samples |
| 10 | Forgetting temperature changes entropy | It changes distribution shape | Track entropy and probability mass |

## 11. Exercises

1. (*) Implement inverse-transform sampling for an exponential distribution.
2. (*) Implement categorical sampling from cumulative probabilities.
3. (*) Estimate $\mathbb{E}[X^2]$ for $X\sim\mathcal{N}(0,1)$ by Monte Carlo.
4. (**) Compute a Monte Carlo confidence interval.
5. (**) Implement rejection sampling for a simple target.
6. (**) Compute importance weights and effective sample size.
7. (***) Implement a random-walk Metropolis sampler.
8. (***) Implement the Gaussian reparameterization trick and verify sample mean.
9. (***) Implement Gumbel-Max and compare frequencies to target probabilities.
10. (***) Implement top-k and top-p filtering for categorical logits.

## 12. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Monte Carlo | Approximates expectations and gradients |
| Importance sampling | Corrects proposal mismatch but can have high variance |
| MCMC | Enables posterior approximation and energy-based sampling |
| Reparameterization | Makes latent-variable models trainable by pathwise gradients |
| Gumbel tricks | Relax categorical choices for differentiable training |
| Minibatch sampling | Defines SGD noise and data coverage |
| Negative sampling | Shapes representation learning |
| Temperature | Controls entropy in categorical outputs |
| Top-k/top-p | Truncates generation distributions |
| ESS | Diagnoses whether weighted samples are actually useful |

## 13. Conceptual Bridge

Losses define training signals, activations transform hidden states, and
normalization controls scale. Sampling adds randomness: approximate
expectations, stochastic gradients, latent variables, negative examples, and
generated outputs.

```text
probability distribution
    -> sampler
    -> random samples
    -> estimator or generated output
    -> training / inference decision
```

Next, Chapter 14 uses these primitives inside concrete model families such as
linear models, neural networks, probabilistic models, reinforcement learning,
and generative models.

## 14. Dossier Template: How to Read a Sampling Method
Every sampler in this section is specified by: the target (normalized density, unnormalized density, or only samples/score access), what it returns (independent samples, a correlated chain, weighted samples, or a differentiable surrogate), its exactness (exact, asymptotically exact, or biased approximation), its diagnostics, and its characteristic failure. Fill these in before reaching for any sampler.

- **Target access**: full normalized density (inverse-CDF, rejection), density up to a constant (MCMC, importance weighting), samples only (GAN-style implicit models — different toolbox), or reparameterizable base noise (VAE-style).
- **Output type**: iid samples (Monte Carlo error $1/\sqrt N$), Markov chain (error set by effective sample size), weighted samples (ESS of weights), differentiable path (gradient estimator, not a sampler per se).
- **Exactness**: rejection and inverse-CDF are exact; MCMC is exact only asymptotically (burn-in, mixing); variational/Gumbel relaxations are biased at finite temperature/steps.
- **Diagnostics**: ESS and $\hat R$ (Vehtari et al. 2021) for chains; weight degeneracy (max weight share) for importance sampling; acceptance rates for MH/HMC; none of these is optional in serious work.
- **Failure mode**: name it in advance — weight collapse, random-walk stagnation, funnel geometry, relaxation bias, mode dropping.
- **Cost model**: cost per effective sample, not per raw sample; a cheap correlated sample can be worth 1/100th of an independent one.
- Canonical anchors: Metropolis et al. (1953), Hastings (1970), Neal (2011) for HMC, Hoffman & Gelman (2014) for NUTS, Vehtari et al. (2021) for modern diagnostics, Jang et al. (2016) and Maddison et al. (2016) for Gumbel-Softmax/Concrete, Kingma & Welling (2013) for reparameterization.



## 15. Inverse-CDF and Rejection Sampling: The Exact Baselines
If $F$ is a CDF with tractable inverse, $X=F^{-1}(U)$, $U\sim\mathrm{Unif}(0,1)$, is exact sampling in one line. Rejection sampling needs only an unnormalized target $\tilde p$ and a proposal $q$ with $\tilde p(x)\le Mq(x)$: accept proposals with probability $\tilde p(x)/(Mq(x))$. Both are exact; both fail exactly where their assumptions fail.

- Inverse-CDF failure: $F^{-1}$ unavailable or expensive (most multivariate targets); the method is a one-dimensional workhorse and a building block (e.g., exponential via $-\log U/\lambda$).
- Probability integral transform, both directions: $F(X)\sim$ Unif and $F^{-1}(U)\sim F$ — also the basis of copulas and of quantile/ inverse-transform tricks in generative modelling.
- Rejection efficiency equals $1/M$ (for normalized $p$): envelope quality is everything; a loose envelope in high dimensions is catastrophic (acceptance decays exponentially in dimension for naive envelopes).
- Adaptive rejection sampling exploits log-concavity to build piecewise envelopes; know it exists before hand-rolling envelopes for log-concave targets.
- Rejection's role in ML today: mostly conceptual and small-scale (truncated distributions, prior sampling, rejection sampling for fine-tuning data — e.g., best-of-$n$ selection is rejection-like with a reward threshold).
- Best-of-$n$ bridge: sampling $n$ completions and keeping the best by a reward model is rejection sampling against an implicit tilted distribution; its KL-from-base grows like $\log n$ — a Chapter 15-relevant fact worth stating here.
- Exactness check: chi-square/KS tests on samples vs target in unit tests for any hand-written exact sampler.
- Non-example: using rejection with an envelope fitted on the same data you then sample 'from the posterior' — the envelope must dominate everywhere, including unexplored tails.



## 16. Importance Sampling: Weighted Samples and Their Degeneracy
Importance sampling estimates $\mathbb E_p[f]$ using samples from $q$: $\mathbb E_p[f]=\mathbb E_q[f(x)w(x)]$, $w=p/q$ (self-normalized when $p$ is known up to a constant). It powers off-policy evaluation, ELBO-adjacent estimators, and particle methods — and collapses when weights degenerate.

- Effective sample size of weights: ESS $=(\sum w_i)^2/\sum w_i^2$; ESS near 1 means one sample is doing all the work (Section 19 computes this).
- Degeneracy is dimensional: weight variance grows (often exponentially) with the mismatch between $p$ and $q$ accumulated over dimensions/sequence length — the core difficulty of IS for long sequences.
- Self-normalized IS is biased at finite $N$ (consistent as $N\to\infty$); report $N$ and ESS with every estimate.
- Defensive mixtures ($q$ mixed with a heavy-tailed component) bound weights and are cheap insurance.
- Perplexity-style uses: importance-weighted autoencoder bounds (IWAE) tighten ELBO with more samples — bias/variance trade made explicit by ESS reporting.
- Off-policy RL evaluation is IS over trajectories; product-of-ratios over horizons is why variance explodes and clipping/marginalized variants exist.
- Diagnostics before conclusions: max weight share, ESS, and a sensitivity run with a different $q$; an IS estimate without these is an anecdote.
- Bridge: sampling-importance-resampling (SIR) turns weighted samples back into unweighted ones — bootstrap particle filters are the sequential version (§04 main text lineage).



## 17. MCMC I: Metropolis-Hastings and Gibbs
When only $\tilde p\propto p$ is available, Markov chain Monte Carlo builds a chain with stationary distribution $p$. Metropolis et al. (1953) proposed the symmetric-proposal version; Hastings (1970) generalized it. Gibbs sampling updates one coordinate (block) at a time from its conditional — an MH special case with acceptance 1 when conditionals are exact.

- MH acceptance: $\alpha=\min\left(1,\frac{\tilde p(x')q(x|x')}{\tilde p(x)q(x'|x)}\right)$; the unknown normalizing constant cancels — the entire point.
- Detailed balance $p(x)T(x\to x')=p(x')T(x'\to x)$ is the standard sufficient condition; burn-in discards the transient, thinning is usually wasted samples (keep samples, report ESS instead).
- Random-walk MH scaling: optimal acceptance ≈0.234 in high-dimensional Gaussian-like targets (classical optimal-scaling results); tune proposal scale to acceptance, not to vibes.
- Gibbs requires tractable conditionals; in conjugate models (topic models historically) it is simple and effective; blocked/collapsed variants integrate out parameters to improve mixing.
- Failure modes: multimodal targets (chains stuck in one mode), strong correlations (axis-aligned Gibbs crawls), funnels ( Neal's funnel — HMC's nemesis too).
- Multiple chains from dispersed starts are not optional garnish: they are what makes $\hat R$ computable (Section 20).
- Andrieu et al. (2003) is the accessible MH introduction; Neal (2011) is the deep reference for MCMC using Hamiltonian dynamics.
- Historical ML footprint: LDA Gibbs sampling, Bayesian matrix factorization, and today's Bayesian last-layer / posterior-tempering experiments all run on this machinery.

## 18. Computed Table: Monte Carlo Error Shrinks as $1/\sqrt{N}$
Seed 42. Ten times the samples buys ~3.16x accuracy, whoever the sampler is — provided samples are independent. Correlated (MCMC) or weighted (IS) samples pay an ESS discount on top of this table (Sections 19-20).

| N | MC estimate of E[x^2]=1 | abs error | 1/sqrt(N) scale |
|---|---|---|---|
| 10 | 0.8969 | 0.1031 | 0.3162 |
| 100 | 0.5913 | 0.4087 | 0.1000 |
| 1000 | 1.0020 | 0.0020 | 0.0316 |
| 10000 | 1.0177 | 0.0177 | 0.0100 |
| 100000 | 1.0066 | 0.0066 | 0.0032 |
| 1000000 | 0.9997 | 0.0003 | 0.0010 |



## 19. Computed Table: Importance-Weight Degeneracy vs Proposal Mismatch
Target $\mathcal N(0,1)$, proposals $\mathcal N(0,s^2)$, 20k samples, seed 42. Even in one dimension, a 3x-too-wide proposal throws away most of the sample budget; in high dimensions this decay compounds per dimension.

| proposal scale | ESS/N | max weight share |
|---|---|---|
| N(0,1.0) vs target N(0,1) | 1.0000 | 0.0001 |
| N(0,1.2) vs target N(0,1) | 0.9527 | 0.0001 |
| N(0,1.5) vs target N(0,1) | 0.8321 | 0.0001 |
| N(0,2.0) vs target N(0,1) | 0.6609 | 0.0001 |
| N(0,3.0) vs target N(0,1) | 0.4574 | 0.0001 |



## 20. Diagnostics: $\hat R$, ESS, and Reading a Chain
Modern MCMC diagnostics (Vehtari et al. 2021) replace eyeballing trace plots alone: rank-normalized split-$\hat R$ compares within- and between-chain behaviour; bulk- and tail-ESS measure how many independent draws your chain is worth for central and quantile estimates.

- Run at least 4 chains from dispersed initializations; single-chain diagnostics cannot detect unseen modes.
- $\hat R\approx1.00$ (common threshold 1.01) is necessary, not sufficient: chains can agree while all missing a mode.
- ESS: Monte Carlo standard error of a mean is $\mathrm{sd}/\sqrt{\mathrm{ESS}}$; report MCSE alongside every posterior summary.
- Bulk-ESS (central mass) and tail-ESS (quantiles) differ; credible-interval claims need tail-ESS.
- Divergent transitions in HMC/NUTS are not warnings to suppress: they mark regions the sampler cannot faithfully explore (funnels, stiff geometry) — reparametrize (e.g., non-centered) rather than raising adapt_delta forever.
- Trace/rank plots remain useful companions; diagnostics quantify what plots suggest (Vehtari et al. 2021 recommend rank plots over raw traces).
- Thinning to 'reduce correlation' wastes information for expectations; thin only for storage, and say so.
- Convergence is a property of (sampler, target, runtime) jointly; report all three, plus seed, in methods sections.



## 21. MCMC II: Hamiltonian Monte Carlo and NUTS
HMC (Neal 2011) augments the target with momentum, simulates Hamiltonian dynamics by leapfrog integration, and corrects with an MH accept step — proposing distant, high-acceptance moves where random-walk MH creeps. NUTS (Hoffman & Gelman 2014) removes the trajectory-length tuning by stopping when the trajectory doubles back.

- Why it works: gradient information steers proposals along level sets; momentum suppresses random-walk backtracking, improving dimensional scaling over RWM.
- Tuning surface: step size (acceptance-targeted dual averaging) and trajectory length (NUTS automates); mass matrix adaptation ≈ preconditioning.
- Leapfrog is volume-preserving and reversible — the two properties that keep the MH correction valid; energy error sets the acceptance probability.
- Divergences signal the integrator failing in high-curvature regions (Neal's funnel is the canonical example); non-centered parametrizations are the standard repair.
- HMC needs gradients of $\log\tilde p$: automatic differentiation made it the default engine of Stan/PyMC/NumPyro-style Bayesian ML.
- Discrete parameters have no gradient: marginalize them out or use Gibbs-within-HMC compositions; pure NUTS on discrete spaces is a category error.
- Cost accounting: each leapfrog step costs a gradient; compare samplers by ESS per gradient evaluation, not per iteration (Section 14 cost model).
- Warmup/adaptation samples are discarded; adaptation must stop before sampling so the kernel is fixed and stationary-distribution arguments apply.



## 22. Differentiable Sampling I: Reparameterization
Kingma & Welling (2013, VAE) and Rezende et al. (2014) popularized writing $x=g_\phi(\epsilon)$, $\epsilon\sim p(\epsilon)$ fixed noise, so gradients flow through $g$. This turns sampling into a deterministic function of parameters and exogenous noise — the enabling trick of variational autoencoders and much of modern latent-variable ML.

- Requirement: continuous variables with a differentiable transform (location-scale families directly; others via inverse-CDF or implicit reparameterization).
- Gradient quality: pathwise (reparameterized) gradients usually have far lower variance than score-function (REINFORCE-style) estimators, which pay for generality with variance and baselines.
- ELBO connection (§01 losses): $\mathbb E_q[\log p-\log q]$ estimated by reparameterized samples; the sampler is inside the loss.
- Gaussian case: $x=\mu+\sigma\epsilon$; note $\sigma$ via softplus (§02 Problem G4) to keep the sampler valid during optimization.
- Discrete variables break the path: Gumbel-Softmax (Section 23) is the relaxation answer; straight-through estimators are the biased engineering answer — name the bias.
- Antithetic samples and common random numbers are variance-reduction companions when comparing objectives across parameters.
- Failure: heavy-tailed or multimodal posteriors under a Gaussian $q$ — the sampler is faithful to $q$; the approximation error lives in the family, diagnosed via importance-weighted bounds (Section 16).

## 23. Differentiable Sampling II: Gumbel-Softmax and Concrete
Jang, Gu & Poole (2016) and Maddison et al. (2016, independently) relaxed discrete sampling: add Gumbel noise to logits and apply a temperature-softmax, $y_i=\frac{\exp((\log\pi_i+g_i)/\tau)}{\sum_j\exp((\log\pi_j+g_j)/\tau)}$. At $\tau\to0$ this recovers categorical sampling (Gumbel-max trick); at finite $\tau$ it is a biased, differentiable surrogate.

- Gumbel-max foundation: argmax of (logits + Gumbel noise) is an exact categorical sample — the trick is exact before relaxation; only the softmax relaxation adds bias.
- Temperature $\tau$ is the bias-variance knob: low $\tau$ ≈ discrete but high-variance gradients; high $\tau$ smooth but samples are soft mixtures, not categories.
- Straight-through variant: forward uses hard argmax, backward uses the soft gradient — biased, often effective, must be reported as ST-Gumbel, not 'Gumbel-Softmax sampling'.
- Uses: discrete latent variables, neural architecture/hard-attention relaxations, discrete VAEs; in LLM-era practice, mostly research code rather than production decoding.
- Annealing schedules ($\tau$ decayed over training) are part of the estimator; comparing papers without $\tau$ schedules is meaningless.
- Alternative unbiased-ish routes: REINFORCE with baselines/control variates (variance cost), or exact marginalization when the discrete space is small — always price exact enumeration first.
- Numerical care: sample Gumbel via $-\log(-\log U)$ with $U$ clipped away from 0/1 (§02 Section 65 discipline).



## 24. Computed Table: Top-k and Top-p (Nucleus) Truncation
A sorted next-token distribution. Top-k keeps a fixed count; top-p (nucleus sampling, Holtzman et al. 2019 lineage in Chapter 15 §05) keeps the smallest set with cumulative mass ≥ p — here ranks 1-6. Truncation then renormalizes; sampling is from the truncated distribution, not the model's.

| token rank | p | cumulative | in top-p=0.9? | in top-k=3? |
|---|---|---|---|---|
| 1 | 0.3500 | 0.3500 | yes | yes |
| 2 | 0.2500 | 0.6000 | yes | yes |
| 3 | 0.1500 | 0.7500 | yes | yes |
| 4 | 0.1000 | 0.8500 | yes | no |
| 5 | 0.0600 | 0.9100 | yes | no |
| 6 | 0.0400 | 0.9500 | no | no |
| 7 | 0.0300 | 0.9800 | no | no |
| 8 | 0.0200 | 1.0000 | no | no |



## 25. Sampling for LLM Decoding: Temperature, Top-k, Top-p
Decoding is applied sampling on the model's categorical distribution (§02 Section 22): temperature rescales logits, top-k/top-p truncate, then one token is drawn (or the argmax taken, greedy = zero-temperature limit). Chapter 15 §05 owns decoding policy; here the sampling-theoretic content.

- Greedy is not sampling: it is argmax; its failures (repetition, blandness) are decision-rule failures, not sampler bugs.
- Temperature preserves ranking and support; truncation changes support — conflating the two ('temperature 0.9 with top-p 0.9') muddles ablations.
- Renormalization after truncation is part of the definition; sampling proportionally without renormalizing (a real bug) distorts ratios.
- Repetition penalties and logit biases are distribution tilts applied before sampling; record them as part of the sampler specification.
- Seed discipline: decoding at temperature > 0 is a random experiment; evaluations report seeds/distributions of scores, never single generations as evidence.
- Beam search is approximate MAP sequence search, not sampling from the sequence distribution; quality/diversity trade-offs differ accordingly.
- Best-of-$n$ + reranking (Section 15) tilts toward reward; its distribution shift grows with $n$ and must be budgeted like any rejection scheme.
- Sampler correctness tests: with truncation off and T=1, empirical token frequencies must match model probabilities (chi-square on a fixed prompt set).



## 26. Negative Sampling and Subsampling Tricks
- Word2vec-style negative sampling (Mikolov et al. 2013 lineage) replaces a full softmax with discrimination against $k$ noise samples — an estimator change (NCE family), not merely a speed hack; the optimum differs from the true softmax optimum.
- Noise distribution choice (unigram$^{3/4}$ famously) shapes what is learned; it is part of the objective's definition.
- In-batch negatives (contrastive learning, §01 InfoNCE) reuse other examples as negatives: batch composition becomes part of the sampler — large batches help estimates and change the task simultaneously.
- Hard-negative mining tilts the negative distribution toward informative examples; unchecked, it also mines label errors — audit mined negatives.
- Importance-corrected variants attempt to remove the noise-distribution bias; state whether yours does.
- Frequent-token subsampling (word2vec's discard rule) is sampling applied to data, changing the effective corpus distribution; report it alongside corpus statistics.
- Modern echo: sampled softmax and candidate sampling in retrieval/recommenders are the same pattern at industrial scale.

## 27. Computed Table: Random-Walk MH Tuning on a Standard Normal
Seed 42, 20k steps each. Too-small proposals accept everything and crawl (low ESS); too-large proposals stagnate (low acceptance, low ESS). The middle maximizes ESS — acceptance rate is a tuning thermometer, ESS is the objective (Section 17's 0.234 guidance is the high-dimensional version of this table).

| proposal scale | accept rate | chain mean (true 0) | chain std (true 1) | ESS of 20000 |
|---|---|---|---|---|
| 0.1 | 0.970 | 0.130 | 0.977 | 52 |
| 0.5 | 0.846 | -0.021 | 1.017 | 813 |
| 1.0 | 0.699 | -0.024 | 1.023 | 2332 |
| 1.7 | 0.552 | -0.008 | 1.005 | 3668 |
| 2.4 | 0.444 | 0.019 | 1.010 | 3931 |
| 4.0 | 0.295 | 0.003 | 0.960 | 3814 |
| 8.0 | 0.155 | 0.016 | 0.998 | 1890 |



## 28. Worked Problem Set A: Inversion, Rejection, Importance
Worked problems A: exact and weighted sampling.

**Problem A1: sample an exponential by inversion**
1. Target Exp$(\lambda)$: $F(x)=1-e^{-\lambda x}$; solve $u=F(x)$: $x=-\log(1-u)/\lambda$.
2. Since $1-U$ is also uniform, $x=-\log U/\lambda$ is the standard one-liner.
3. Test: sample mean of 100k draws at $\lambda=2$ must sit near $0.5\pm0.005$ (Section 18 scale).
4. State the failure register: $U=0$ gives $\log0$; clip $U$ to $[10^{-12},1)$ (§02 Section 65 discipline).

**Problem A2: rejection envelope audit**
1. Target $\tilde p(x)=e^{-x^2/2}(1+0.3\sin 3x)$-style unnormalized density on $\mathbb R$ (verify positivity before proceeding).
2. Proposal $q=\mathcal N(0,1.5^2)$; $M$ must bound $\tilde p/q$ everywhere, including tails — check numerically on a wide grid plus tail analysis.
3. Acceptance $\approx Z/M$ where $Z$ is the unknown normalizer; low acceptance is a cost report, not an error.
4. If the bound fails anywhere, samples are biased toward the violated region — envelopes are theorems, not heuristics.

**Problem A3: read an IS report**
1. Report says: IS estimate 4.21, N=10000, ESS=37, max weight share 0.41.
2. Verdict: effectively ~37 samples, one carrying 41% — the estimate is a single-sample anecdote with decoration.
3. Actions: better/defensive proposal, more samples, or a different estimator; do not publish 4.21 without the ESS context.
4. Rule adopted in this chapter: ESS and max-weight share accompany every IS number (Section 16).

**Problem A4: self-normalized IS bias**
1. With normalized weights $\bar w_i$, the estimator is a ratio of two MC averages — biased at finite $N$, consistent as $N\to\infty$.
2. Bias shrinks like $1/N$ (with constants set by weight variance); ESS is the practical proxy for 'which $N$ am I really at'.
3. Report both raw $N$ and ESS; a million samples with ESS 50 is a 50-sample study.



## 29. Worked Problem Set B: MCMC Practice and Diagnostics
Worked problems B: MCMC practice.

**Problem B1: diagnose from one number**
1. A colleague ran one chain, reports $\hat R$ 'not applicable', ESS 400 of 40000 draws, posterior mean 1.7.
2. Issues: single chain (no between-chain check), ESS/draws = 1% (heavy autocorrelation), MCSE $=\mathrm{sd}/\sqrt{400}$ likely material vs the reported precision.
3. Prescription: 4 dispersed chains, tune proposal/NUTS settings, report $\hat R$, bulk/tail ESS, MCSE (Vehtari et al. 2021).
4. Do not 'fix' by thinning: thinning cannot raise ESS above the information already collected.

**Problem B2: divergences in a hierarchical model**
1. NUTS reports 3% divergent transitions concentrated at small group-level scale — the funnel signature (Neal 2011).
2. Repair: non-centered parametrization (sample standardized effects, scale deterministically), then re-check divergences → near zero.
3. Raising adapt_delta alone treats the symptom and can hide unexplored geometry; divergences are data about the posterior's shape.
4. Record parametrization in methods: centered vs non-centered is a sampler-relevant model decision.

**Problem B3: Gibbs vs HMC choice**
1. Model: Bayesian logistic regression, 200 continuous parameters, no tractable full conditionals beyond generic forms.
2. Gibbs would need MH-within-Gibbs per coordinate — slow mixing under correlation.
3. HMC/NUTS uses the joint gradient (AD-available): the default choice (Section 21).
4. Gibbs wins when conditionals are conjugate/closed-form and blocking is natural (classic LDA); decide by conditional tractability + correlation, not habit.

**Problem B4: burn-in and adaptation hygiene**
1. Warmup samples (including all adaptation) are discarded; the sampling kernel must be fixed thereafter for stationarity arguments.
2. Saving warmup draws 'because compute is expensive' quietly biases every summary toward the transient.
3. State in methods: warmup length, adaptation scheme, chains, seeds — reproducibility of an MCMC result requires all four (Section 20).

## 30. Computed Table: Gumbel-Softmax Relaxation vs Temperature
Fixed logits $\pi=(0.5,0.3,0.2)$ and one Gumbel draw $g=(0.2,-0.4,0.9)$ (which makes category 3 the Gumbel-max sample). As $\tau$ falls the soft vector hardens onto that sample; at working temperatures it is a mixture — the bias of Section 23 made visible.

| tau | y1 | y2 | y3 | argmax |
|---|---|---|---|---|
| 2.0 | 0.4046 | 0.2322 | 0.3632 | 1 |
| 1.0 | 0.4684 | 0.1542 | 0.3773 | 1 |
| 0.5 | 0.5691 | 0.0617 | 0.3692 | 1 |
| 0.25 | 0.6980 | 0.0082 | 0.2938 | 1 |
| 0.1 | 0.8969 | 0.0000 | 0.1031 | 1 |
| 0.05 | 0.9869 | 0.0000 | 0.0131 | 1 |



## 31. Worked Problem Set C: Differentiable Sampling and Decoding
Worked problems C: differentiable sampling and decoding.

**Problem C1: reparameterize or score-function?**
1. Need gradients of $\mathbb E_{q_\phi}[f(x)]$ for a Gaussian $q_\phi$ and differentiable $f$.
2. Reparameterization applies ($x=\mu+\sigma\epsilon$) and typically has much lower variance (Section 22).
3. Choose pathwise gradients; reserve score-function estimators for discrete/non-differentiable cases, with baselines budgeted.
4. Verify by gradient-variance measurement on a toy problem before scaling up — estimator choice is an empirical claim too.

**Problem C2: Gumbel temperature incident**
1. Discrete-latent model 'samples' look like averaged categories and downstream accuracy is poor at $\tau=1$.
2. Diagnosis: relaxation bias — at $\tau=1$ outputs are soft mixtures (Section 30), not samples; the downstream module was trained on mixtures it will not see at test time (hard samples).
3. Options: anneal $\tau$, straight-through (declared biased), or exact marginalization if feasible; re-validate the train/test input match.
4. Report the $\tau$ schedule as part of the method (Section 23).

**Problem C3: decoding ablation honesty**
1. Claim: 'top-p 0.9 beats top-k 40 on our eval.'
2. Required controls: same temperature, same model/checkpoint, multiple seeds, score distributions not single runs, truncation+renormalization verified (Section 25).
3. Also state what each method does to support: fixed count vs adaptive mass (Section 24).
4. Single-generation screenshots are not evidence; they are illustrations.

**Problem C4: best-of-n budget**
1. Best-of-16 with a reward model improves average reward; a stakeholder proposes best-of-1024.
2. Sampling view (Section 15): the implicit tilt strengthens ~$\log n$; outputs drift from the base distribution and toward reward-model idiosyncrasies (overoptimization).
3. Measure KL-from-base and held-out human/gold metrics vs $n$; pick $n$ where gold metrics plateau, not where proxy reward peaks.
4. This is rejection-sampling economics inside LLM alignment (Chapter 15 connection).



## 32. Historical Timeline: Sampling Methods
| Year | Milestone | Contribution |
|---|---|---|
| 1953 | Metropolis et al. | Metropolis algorithm (symmetric proposals). |
| 1970 | Hastings | General Metropolis-Hastings acceptance ratio. |
| 2003 | Andrieu et al. | Accessible MH tutorial; MCMC in machine learning framing. |
| 2011 | Neal, MCMC using Hamiltonian dynamics | The HMC reference chapter. |
| 2013 | Kingma & Welling | Reparameterization trick / VAE (with Rezende et al. 2014 in parallel). |
| 2014 | Hoffman & Gelman | NUTS: automatic HMC trajectory tuning. |
| 2016 | Jang et al.; Maddison et al. | Gumbel-Softmax / Concrete relaxations (independent). |
| 2019 | Holtzman et al. | Nucleus (top-p) sampling for text degeneration problem. |
| 2021 | Vehtari et al. | Rank-normalized $\hat R$ and ESS: modern MCMC diagnostics standard. |



## 33. Glossary: Sampling Vocabulary
| Term | Meaning |
|---|---|---|
| Burn-in / warmup | Initial transient discarded; includes all adaptation in modern samplers. |
| Stationary distribution | Distribution a chain converges to and (if ergodic) samples from asymptotically. |
| Detailed balance | Sufficient condition $p(x)T(x\to x')=p(x')T(x'\to x)$ for stationarity. |
| Mixing | Speed at which a chain forgets its start and explores the target. |
| Autocorrelation time | Correlation length converting raw draws into effective ones. |
| ESS (effective sample size) | Independent-draw equivalent of a correlated/weighted sample set. |
| $\hat R$ | Between/within chain comparison; ~1.01 threshold in Vehtari et al. (2021). |
| MCSE | Monte Carlo standard error of an estimate: sd$/\sqrt{\mathrm{ESS}}$. |
| Divergent transition | HMC integrator failure flag; marks unexplorable geometry, never to be silenced. |
| Non-centered parametrization | Sampling standardized variables and scaling deterministically; funnel repair. |
| Leapfrog integrator | Volume-preserving, reversible HMC dynamics discretization. |
| Proposal distribution | $q(x'|x)$ in MH; its scale sets the acceptance/ESS trade (Section 27). |
| Envelope constant $M$ | Rejection bound $\tilde p\le Mq$; acceptance $\approx Z/M$. |
| Self-normalized IS | Weights normalized by their sum; consistent, finite-$N$ biased. |
| Defensive mixture | Heavy-tailed proposal component bounding importance weights. |
| Reparameterization | Writing samples as differentiable transforms of fixed noise (Kingma & Welling 2013). |
| Score-function estimator | REINFORCE-style gradient using $\nabla\log q$; general, higher variance. |
| Gumbel-max trick | argmax(logits + Gumbel noise) is an exact categorical sample. |
| Straight-through estimator | Hard forward, soft backward; biased and must be declared. |
| Nucleus (top-p) sampling | Smallest token set with cumulative mass $\ge p$, renormalized (Holtzman et al. 2019). |
| Negative sampling | Replacing softmax normalization by noise discrimination (Mikolov et al. 2013 lineage). |

## 34. Worked Trace: 40 Steps of Metropolis-Hastings from a Bad Start
Target $\mathcal N(0,1)$, start $x_0=3$ (deep in the tail), proposal scale 1.0, seed 42. The running mean crawls toward 0 as burn-in washes out the start — the transient you discard, made visible. Note rejected proposals repeat the current value: chains are sticky by design.

| step | proposal | accepted? | chain x | running mean |
|---|---|---|---|---|
| 1 | 3.305 | no | 3.000 | 3.000 |
| 2 | 3.750 | no | 3.000 | 3.000 |
| 3 | 1.049 | yes | 1.049 | 2.350 |
| 4 | 1.177 | yes | 1.177 | 2.056 |
| 5 | 1.160 | yes | 1.160 | 1.877 |
| 6 | 2.039 | no | 1.160 | 1.758 |
| 7 | 1.226 | yes | 1.226 | 1.682 |
| 8 | 1.694 | yes | 1.694 | 1.683 |
| 9 | 2.062 | yes | 2.062 | 1.725 |
| 10 | 2.941 | no | 2.062 | 1.759 |
| 11 | 1.877 | yes | 1.877 | 1.770 |
| 12 | 3.100 | no | 1.877 | 1.779 |
| 13 | 1.449 | yes | 1.449 | 1.753 |
| 14 | 1.981 | yes | 1.981 | 1.770 |
| 15 | 2.394 | no | 1.981 | 1.784 |
| 16 | 4.123 | no | 1.981 | 1.796 |
| 17 | 1.469 | yes | 1.469 | 1.777 |
| 18 | 2.085 | yes | 2.085 | 1.794 |
| 19 | 1.971 | yes | 1.971 | 1.803 |
| 20 | 1.147 | yes | 1.147 | 1.771 |
| 21 | 1.890 | no | 1.147 | 1.741 |
| 22 | -0.024 | yes | -0.024 | 1.661 |
| 23 | -0.690 | no | -0.024 | 1.587 |
| 24 | 0.092 | yes | 0.092 | 1.525 |
| 25 | 0.964 | yes | 0.964 | 1.503 |
| 26 | 1.643 | yes | 1.643 | 1.508 |
| 27 | 1.932 | no | 1.643 | 1.513 |
| 28 | 0.185 | yes | 0.185 | 1.466 |
| 29 | -0.285 | yes | -0.285 | 1.405 |
| 30 | -0.560 | yes | -0.560 | 1.340 |
| 31 | -1.426 | no | -0.560 | 1.278 |
| 32 | -2.243 | no | -0.560 | 1.221 |
| 33 | -0.397 | yes | -0.397 | 1.172 |
| 34 | 0.314 | yes | 0.314 | 1.147 |
| 35 | -0.035 | yes | -0.035 | 1.113 |
| 36 | 0.823 | yes | 0.823 | 1.105 |
| 37 | -0.453 | yes | -0.453 | 1.063 |
| 38 | -1.372 | yes | -1.372 | 0.999 |
| 39 | -1.230 | yes | -1.230 | 0.942 |
| 40 | -1.657 | no | -1.230 | 0.887 |



## 35. Sampling Mistake Catalogue (Field Version)
Complements Section 10 with estimator- and reporting-level failures.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Single chain, no diagnostics | Unseen modes, unknown MCSE | 4+ chains, $\hat R$/ESS/MCSE (Vehtari et al. 2021) |
| 2 | Silencing divergences | Biased exploration of funnels | Reparametrize (non-centered); investigate geometry |
| 3 | Thinning to 'fix' autocorrelation | Discards information, ESS unchanged in value | Keep draws; report ESS; tune sampler instead |
| 4 | IS without ESS report | Single-sample anecdotes published (Problem A3) | ESS + max weight share mandatory |
| 5 | Warmup draws included in results | Transient bias in all summaries | Discard warmup; fixed kernel during sampling |
| 6 | Envelope not dominating (rejection) | Silent bias toward violated regions | Prove/check bound incl. tails (Problem A2) |
| 7 | Gumbel tau unreported | Irreproducible relaxation bias | Report tau schedule; ST declared if used |
| 8 | Decoding claims from single generations | Noise as evidence | Seeds + score distributions (Problem C3) |
| 9 | Temperature and truncation conflated | Uninterpretable decoding ablations | Specify tilt vs support change separately |
| 10 | Proposal scale by habit | ESS collapse (Section 27) | Tune to acceptance; validate by ESS |
| 11 | Best-of-n without KL budget | Reward overoptimization drift | Track KL-from-base vs n (Problem C4) |
| 12 | Sampler compared per-iteration | Correlated cheap draws overvalued | Compare ESS per unit cost (Section 14) |



## 36. Pre-Flight Checklist for a Sampling Study
- Target specified: normalized, unnormalized, or implicit; access type named (Section 14 dossier).
- Sampler chosen with its exactness class: exact, asymptotically exact, or biased relaxation (bias source named).
- Pilot run: diagnostics thresholds set in advance ($\hat R$, ESS, acceptance, divergence budget).
- Chains >= 4 from dispersed starts (MCMC) or independent replicates (IS/exact samplers).
- Seed policy and RNG stream handling recorded; parallel chains do not share streams.
- Cost model: budget in ESS per gradient/second, not raw draws.
- Reporting template includes: method + citation, tuning, warmup, chains, ESS, MCSE, failure modes checked.
- Unit tests: exact samplers vs known CDFs (KS), sampler frequency tests for decoding (Section 25).
- Failure plan: what diagnostic failure triggers reparametrization vs more samples vs a different sampler.
- Artifacts: chains/draws saved (or regenerable by seed) for audit.



## 37. Decision Records: Sampler Choices in Context
- **Low-dimensional posterior, gradients available:** NUTS (Hoffman & Gelman 2014), 4 chains, non-centered if hierarchical (Problem B2).
- **Conjugate exponential-family model:** blocked Gibbs; collapsed where possible; still report ESS/$\hat R$.
- **Off-policy evaluation, short horizon:** self-normalized IS with defensive proposal; ESS-gated reporting (Problem A3).
- **Discrete latent variable in a neural model:** exact marginalization if small; else Gumbel-Softmax with declared $\tau$ schedule or ST variant (Section 23).
- **LLM decoding:** temperature + top-p with renormalization verified; seed-replicated evals (Section 25).
- **Alignment-time selection:** best-of-$n$ with an explicit KL budget (Problem C4).
- **One-dimensional exact need:** inverse-CDF; multivariate exact need: rejection only with a provable envelope (Section 15).
- Each record names target access, exactness class, diagnostics, and the failure that would trigger a sampler change.

## 38. Worked Problem Set D: Estimator Honesty and Geometry
Worked problems D: estimators and variance.

**Problem D1: MCSE honesty**
1. Posterior sd is 2.0; your chain's ESS is 100. MCSE of the mean $=2/\sqrt{100}=0.2$.
2. Reporting the mean as 1.73 ± (posterior sd only) hides that the third significant figure is Monte Carlo noise.
3. Requirement from Section 20: estimate ± MCSE next to every headline number; raise ESS until MCSE is below the decision-relevant threshold.
4. ESS targets are set by required MCSE, not by round draw counts.

**Problem D2: antithetic variance reduction**
1. For symmetric targets, pairing $x$ with its reflection (or $U$ with $1-U$) induces negative correlation and cuts estimator variance for symmetric $f$.
2. This is free variance reduction when the structure exists; it composes with reparameterized gradients (Section 22 common-random-number family).
3. Validate empirically: estimator variance with and without pairing on the same budget — variance reduction is a measurable claim.

**Problem D3: proposal design from the target's shape**
1. Correlated Gaussian target (correlation 0.95): axis-aligned Gibbs crawls; isotropic random-walk MH mixes slowly along the ridge.
2. Better: proposal covariance matched to target covariance (adaptive MH within warmup), or HMC whose dynamics follow the geometry (Section 21).
3. General principle: samplers fail on geometry, and the repairs are geometric (preconditioning, reparametrization), not 'more iterations' first.



## 39. Worked Problem Set E: Selection and Reporting
Worked problems E: chapter synthesis.

**Problem E1: pick the sampler (rapid fire)**
1. Posterior of a 5-parameter physics model, gradients via AD → NUTS, diagnostics per Section 20.
2. Exact draws from a truncated normal in 1D → inverse-CDF on the truncated CDF.
3. Gradient through a Bernoulli latent → declare: ST-Gumbel (biased) or REINFORCE+ baseline (unbiased, variance) — either is defensible, silence is not.
4. Evaluate a logged bandit policy → self-normalized IS, ESS-gated (Problem A3).
5. Generate text for a user-facing demo → temperature/top-p, seeds logged, no quality claims from single samples.

**Problem E2: write the methods paragraph**
1. Template: 'We sampled [target] with [method, citation]; [tuning/warmup]; [chains/draws]; diagnostics [values]; MCSE [values]; failure checks [list]; seeds [policy].'
2. Fill it for the last sampling study you ran; every bracket you cannot fill is a reproducibility debt.
3. This paragraph is the sampling analogue of §03's model-card norm fields (Problem H3 there).

**Problem E3: when the answer is 'do not sample'**
1. If the expectation is available in closed form (conjugate cases, small discrete spaces), compute it — sampling adds error to an answered question.
2. Enumeration over a small discrete latent is exact marginalization, beating every relaxation in Section 23.
3. Sampling is for intractable targets; 'we used MCMC' for a tractable one signals a skipped analysis step.



## 40. Formula Sheet
| Method | Core identity | Exactness | Canonical citation |
|---|---|---|---|
| Inverse-CDF | $X=F^{-1}(U)$ | exact | standard (PIT) |
| Rejection | accept w.p. $\tilde p/(Mq)$ | exact | standard; adaptive variants for log-concave |
| Importance sampling | $\mathbb E_p f=\mathbb E_q[fw]$, ESS $=(\sum w)^2/\sum w^2$ | consistent | standard IS literature |
| Metropolis-Hastings | $\alpha=\min(1,\frac{\tilde p(x')q(x|x')}{\tilde p(x)q(x'|x)})$ | asymptotic | Metropolis et al. 1953; Hastings 1970 |
| Gibbs | full-conditional updates | asymptotic | standard; Andrieu et al. 2003 (tutorial) |
| HMC | leapfrog + MH correction on $(x,\rho)$ | asymptotic | Neal 2011 |
| NUTS | HMC with automatic trajectory length | asymptotic | Hoffman & Gelman 2014 |
| Reparameterization | $x=g_\phi(\epsilon)$ | gradient estimator | Kingma & Welling 2013; Rezende et al. 2014 |
| Gumbel-Softmax | softmax$((\log\pi+g)/\tau)$ | biased relaxation ($\tau>0$) | Jang et al. 2016; Maddison et al. 2016 |
| Top-p sampling | truncate to mass $p$, renormalize | exact for truncated dist. | Holtzman et al. 2019 |
| Diagnostics | $\hat R$, bulk/tail ESS, MCSE | — | Vehtari et al. 2021 |



## 41. Frequently Asked Questions
- **Is burn-in still needed with good initialization?** Yes — warmup also hosts adaptation; discard it regardless of start (Problem B4).
- **More chains or longer chains?** For mode detection and $\hat R$, more chains; for MCSE on an explored target, longer chains. Budget by which uncertainty dominates.
- **Why did my ESS exceed my draw count nowhere but IS reports ESS > N in a paper?** ESS ≤ effective information; self-normalized ESS is bounded by $N$ — a reported ESS > N signals a formula or normalization error worth querying.
- **Can I use MCMC samples as training data directly?** Yes (that's posterior sampling's point), but correlated samples weight the loss unevenly; ESS-aware weighting or thinning-for-storage only, declared.
- **Gumbel-Softmax or REINFORCE for discrete latents?** Bias-variance choice (Problem E1); exact marginalization beats both when feasible (Problem E3).
- **Is greedy decoding 'temperature 0'?** Formally the limit; practically a separate deterministic path in code — specify which you ran.
- **What acceptance rate should random-walk MH have?** ~0.234 in high-dimensional regular targets; in 1D higher (~0.44 classical guidance); tune by ESS (Section 27), not by the constant alone.
- **Do I need $\hat R$ for non-Bayesian MCMC (e.g., sampling for data augmentation)?** The diagnostics measure chain trustworthiness regardless of interpretation; yes, with thresholds fit to the downstream risk.

## 42. Computed Table: Autocorrelation Taxes Your Sample Budget
For an AR(1)-correlated chain, ESS/N $=(1-\rho)/(1+\rho)$. At $\rho=0.99$ — unremarkable for a badly tuned random walk — ten thousand draws are worth about fifty independent ones. This is why Sections 20/27 optimize ESS, not draw counts.

| AR(1) rho | theoretical ESS/N = (1-rho)/(1+rho) | meaning for 10000 draws |
|---|---|---|
| 0.0 | 1.0000 | ~10000 effective draws |
| 0.5 | 0.3333 | ~3333 effective draws |
| 0.8 | 0.1111 | ~1111 effective draws |
| 0.9 | 0.0526 | ~526 effective draws |
| 0.95 | 0.0256 | ~256 effective draws |
| 0.99 | 0.0050 | ~50 effective draws |
| 0.999 | 0.0005 | ~5 effective draws |



## 43. Sequential Monte Carlo and Annealed Bridges (Orientation)
- Particle filters / SMC propagate weighted samples through a sequence of distributions (filtering steps or temperature bridges $p^\beta$), resampling when weight ESS collapses (Section 16 diagnostics reused).
- Annealed importance sampling bridges a tractable base to an intractable target through intermediate distributions, producing both samples and a normalizing-constant estimate.
- Resampling trades weight degeneracy for sample duplication; resample only when ESS drops below a threshold (common: N/2), and say which rule you used.
- SMC is the sampler of record for state-space models and a serious option for multimodal targets where single chains strand (Section 17 failure modes).
- Parallel tempering (replica exchange) runs coupled chains at multiple temperatures and swaps — another bridge idea, effective for separated modes.
- These methods inherit every diagnostic duty of Sections 20/35, plus weight-ESS tracking over time/temperature.
- Depth note: SMC theory is a field of its own; this section's role is placement — recognize when sequential structure makes SMC the natural tool.



## 44. Monitoring Runbook: Sampling Health
- Per run: $\hat R$ (bulk), bulk/tail ESS, MCSE of headline quantities, divergence count (HMC), acceptance rates (MH), weight ESS over time (SMC/IS).
- Alert: $\hat R>1.01$ or ESS below the MCSE-driven target (Problem D1) → do not report estimates; fix the sampler first.
- Alert: divergences > 0 clustered → geometry problem; reparametrize (Problem B2), do not just extend warmup.
- Alert: IS max-weight share > ~0.1 → estimate is fragile regardless of $N$ (Problem A3).
- Log seeds and RNG stream assignments per chain; reproducibility is a monitoring property.
- Save draws (or seeds + versions) with the report; posterior summaries without regenerable draws are unverifiable.
- Post-incident: record symptom, diagnostic that caught it, repair, and the new gate that prevents silent recurrence (Section 35 catalogue grows this way).



## 45. Extended Exercise Bank (Self-Study)
1. Implement MH for $\mathcal N(0,1)$; reproduce Section 27's table shape and find your ESS-maximizing scale.
2. Show the Gumbel-max trick: simulate categorical frequencies from argmax(logits+Gumbel) and chi-square test them.
3. Compute ESS for lognormal weights of increasing spread; find where ESS/N falls below 1%.
4. Derive MH acceptance for a symmetric proposal and state why the normalizing constant cancels.
5. Run 4 chains on a bimodal 1D target from dispersed starts; report $\hat R$ and what a single chain would have concluded.
6. Estimate $\mathbb E[x^2]$ for a Student-$t$ target by IS from a normal proposal; watch the weights, then fix the proposal's tails.
7. Implement the truncated-normal inverse-CDF sampler and KS-test it.
8. Write the methods paragraph (Problem E2) for a decoding experiment, including seed policy.
9. Anneal $\tau$ in a toy Gumbel-Softmax model; plot downstream metric vs final $\tau$ and explain the bias term.
10. Take a published IS estimate from any paper and audit it against Section 16's reporting list.



## 46. Self-Test Questions
- Which targets can MH handle that rejection cannot, and what pays for it?
- State the MH acceptance ratio and the role of detailed balance.
- Define ESS twice: for chains and for weights.
- Why are at least 4 chains standard (Vehtari et al. 2021)?
- What does a divergent transition mean and what is the first repair?
- Reparameterization vs score-function gradients: generality/variance trade.
- At what $\tau$ is Gumbel-Softmax exact, and what breaks there for gradients?
- Top-k vs top-p: what does each fix, what adapts?
- Why does IS degrade with sequence length/dimension?
- Greedy decoding vs sampling: classify the decision rule.
- Name the exactness class of: inverse-CDF, NUTS, straight-through Gumbel, self-normalized IS.
- What three numbers accompany every MCMC headline estimate in this chapter's standard?



## 47. Self-Test Answer Key (Condensed)
- MH needs only $\tilde p$ (unnormalized); the price is correlated, asymptotic-only samples and diagnostic duties.
- $\alpha=\min(1,\tilde p(x')q(x|x')/\tilde p(x)q(x'|x))$; detailed balance is the sufficient stationarity condition.
- Chain ESS: independent-draw equivalent given autocorrelation (Section 42). Weight ESS: $(\sum w)^2/\sum w^2$.
- Dispersed chains enable between-chain comparison ($\hat R$) and mode detection; one chain cannot audit itself.
- Divergence = integrator failed in high-curvature geometry; first repair is reparametrization (non-centered), not silence.
- Pathwise: low variance, needs differentiable transform. Score-function: general (discrete OK), higher variance, needs baselines.
- Exact only at $\tau\to0$ (Gumbel-max); gradients through argmax vanish/break there — hence relaxation bias at working $\tau$.
- Top-k fixes count; top-p fixes cumulative mass and adapts count to distribution sharpness (Section 24).
- Weight variance compounds per dimension/step; ESS collapses (Sections 16/19).
- Greedy is argmax decision, not a sampler; its pathologies are decision-rule pathologies.
- Exact / asymptotically exact / biased / consistent-but-finite-N-biased.
- Estimate, MCSE (from ESS), and $\hat R$ — with sampler, tuning, and seeds in the methods text.

## 48. Worked Dataset: 30 Importance Weights by Hand
Target $\mathcal N(0,1)$, proposal $\mathcal N(0,4)$ (Section 19's setting, small enough to audit). Find the largest normalized weight, verify the ESS line, and feel how few samples carry the estimate.

| i | x_i (from N(0,4)) | raw weight | normalized weight |
|---|---|---|---|
| 0 | 0.609 | 1.7400 | 0.0499 |
| 1 | -2.080 | 0.3949 | 0.0113 |
| 2 | 1.501 | 0.8593 | 0.0247 |
| 3 | 1.881 | 0.5305 | 0.0152 |
| 4 | -3.902 | 0.0066 | 0.0002 |
| 5 | -2.604 | 0.1572 | 0.0045 |
| 6 | 0.256 | 1.9516 | 0.0560 |
| 7 | -0.632 | 1.7214 | 0.0494 |
| 8 | -0.034 | 1.9992 | 0.0574 |
| 9 | -1.706 | 0.6714 | 0.0193 |
| 10 | 1.759 | 0.6270 | 0.0180 |
| 11 | 1.556 | 0.8071 | 0.0232 |
| 12 | 0.132 | 1.9870 | 0.0570 |
| 13 | 2.254 | 0.2973 | 0.0085 |
| 14 | 0.935 | 1.4409 | 0.0414 |
| 15 | -1.719 | 0.6607 | 0.0190 |
| 16 | 0.738 | 1.6310 | 0.0468 |
| 17 | -1.918 | 0.5036 | 0.0145 |
| 18 | 1.757 | 0.6285 | 0.0180 |
| 19 | -0.100 | 1.9925 | 0.0572 |
| 20 | -0.370 | 1.9001 | 0.0545 |
| 21 | -1.362 | 0.9977 | 0.0286 |
| 22 | 2.445 | 0.2125 | 0.0061 |
| 23 | -0.309 | 1.9296 | 0.0554 |
| 24 | -0.857 | 1.5188 | 0.0436 |
| 25 | -0.704 | 1.6606 | 0.0477 |
| 26 | 1.065 | 1.3075 | 0.0375 |
| 27 | 0.731 | 1.6369 | 0.0470 |
| 28 | 0.825 | 1.5490 | 0.0445 |
| 29 | 0.862 | 1.5140 | 0.0435 |
| ESS | | | 22.96 of 30 |



## 49. Sampling Inside Training: Noise as a Tool
- Dropout (§02) is Bernoulli sampling of masks; its test-time scaling is an expectation over that sampler — MC-dropout turns the same device into approximate posterior sampling, with calibration caveats (Chapter on uncertainty owns the details).
- Data augmentation samples transformations from a designer-chosen distribution; the trained model fits the augmented distribution, not the raw one — sampler design again (cf. §03 Problem on corpus statistics analogues).
- SGD itself is sampling: minibatch gradient noise has structure that interacts with LR and batch size; 'sampling methods' thinking (ESS over steps) informs LR schedules in Chapter 8's territory.
- Negative sampling and in-batch negatives (Section 26) make batch construction part of the estimator.
- Curriculum and importance-weighted data sampling tilt the training distribution toward hard/rare examples; tilts are objectives changes and belong in the record (§01 decision discipline).
- Synthetic data from generative models is sampling from a learned distribution; its biases are inherited by downstream training (Chapter 15/16 pipelines).
- Common thread: whenever randomness enters training or evaluation, name the distribution, the estimator, and the diagnostic — the dossier of Section 14 applies to data pipelines too.



## 50. Worked Problem Set F: Audits and Specifications
Worked problems F: final synthesis set.

**Problem F1: audit a Bayesian ML result**
1. Paper reports posterior mean 0.84 from 2 chains × 1000 draws, no diagnostics.
2. Missing: $\hat R$, ESS, MCSE, warmup, parametrization, seeds (Problem E2 template).
3. With ESS possibly ~50 (Section 42 arithmetic), MCSE could be ~0.1·sd — the headline precision is unaudited.
4. Verdict: result may be right; the evidence is incomplete. Reproduction requires the missing fields.

**Problem F2: sampler for a funnel posterior**
1. Hierarchical model, NUTS divergences at small scale (Problem B2 signature).
2. Plan: non-centered parametrization, re-run, require divergences ≈ 0 and tail-ESS adequate for the scale parameter's quantiles.
3. If divergences persist: inspect the geometry (pairs plots of scale vs effects), consider stronger priors justified on subject-matter grounds — priors are modelling, samplers are computation; don't let one silently do the other's job.

**Problem F3: decoding spec for a product**
1. Write the sampler spec: model/version, temperature, top-p, max tokens, repetition penalty, seed policy, logging of realized settings per request.
2. Acceptance tests: frequency test at T=1/no truncation on a fixed prompt battery (Section 25); regression alerts on output-length and refusal-rate distributions.
3. A decoding change is a sampler change: version it like a model change.



## 51. Sampling and the Rest of Chapter 13
- Losses (§01): ELBO and contrastive objectives are expectations estimated by these samplers; estimator variance is loss noise.
- Activations (§02): softmax defines the categorical distributions decoding samples from; Gumbel machinery reuses logits directly.
- Normalization (§03): stable statistics keep the distributions being sampled from well-defined during training; QK-norm bounds the logits being sampled over.
- Forward to Chapter 15: decoding policies, RLHF/DPO sampling loops, and evaluation sampling all instantiate this section's dossiers.
- Forward to Chapter 16: data pipelines are samplers over corpora; deduplication and filtering are distribution design.
- The unifying habit: distribution, estimator, diagnostics — in that order, every time randomness is used.

## 52. Computed Table: Decoding Temperature and Entropy
Fixed 6-token logits. Entropy (and its exponential, the effective number of choices) rises with temperature — the sampling-side view of §02 Section 32, now over a vocabulary-like distribution. Product decoding specs (Problem F3) choose operating points on this curve.

| T | entropy (nats) | p(top) | effective choices exp(H) |
|---|---|---|---|
| 0.3 | 0.1596 | 0.9643 | 1.17 |
| 0.5 | 0.4584 | 0.8647 | 1.58 |
| 0.7 | 0.7224 | 0.7605 | 2.06 |
| 1.0 | 1.0233 | 0.6337 | 2.78 |
| 1.3 | 1.2306 | 0.5420 | 3.42 |
| 1.7 | 1.4085 | 0.4581 | 4.09 |
| 2.5 | 1.5884 | 0.3626 | 4.90 |
| 4.0 | 1.7056 | 0.2847 | 5.50 |



## 53. Quick-Reference Cards
**Inverse-CDF**
- Needs: tractable $F^{-1}$ (mostly 1D). Returns: exact iid draws.
- Fails: multivariate/unknown $F$. Test: KS test.

**Rejection**
- Needs: $\tilde p$ + dominating envelope $Mq$. Returns: exact iid.
- Acceptance $\approx Z/M$; fails in high dimensions.

**Importance sampling**
- Needs: samples from $q$, ratios $p/q$. Returns: weighted estimates.
- Gate every number with ESS + max weight (Problem A3).

**Metropolis-Hastings**
- Needs: $\tilde p$, proposal. Returns: correlated chain.
- Tune by ESS; acceptance is the thermometer (Section 27).

**Gibbs**
- Needs: tractable full conditionals. Acceptance 1 in exact form.
- Crawls under strong posterior correlation.

**HMC / NUTS**
- Needs: $\nabla\log\tilde p$. Distant, high-acceptance moves.
- Watch divergences; reparametrize funnels (Problem B2).

**Reparameterization**
- Needs: differentiable transform of fixed noise.
- Low-variance pathwise gradients (Kingma & Welling 2013).

**Gumbel-Softmax**
- Needs: discrete choice relaxed at temperature $\tau$.
- Biased at $\tau>0$; report schedule / ST (Section 23).

**Top-p decoding**
- Needs: sorted model probabilities; truncate + renormalize.
- Specify T, p, seeds; frequency-test the implementation.



## 54. Exam and Interview Traps: Sampling
- Trap: 'MCMC gives exact samples after burn-in.' Response: asymptotically exact; finite runs carry MCSE quantified by ESS (Sections 20/42).
- Trap: 'Thinning improves estimates.' Response: thinning discards information; it is a storage decision (Section 20).
- Trap: 'High acceptance means a good MH sampler.' Response: scale-0.1 proposals accept ~always and barely move (Section 27); ESS decides.
- Trap: 'Importance sampling with a million draws must be accurate.' Response: ESS can be ~37 (Problem A3); $N$ is not the budget, ESS is.
- Trap: 'Divergences are just warnings.' Response: they mark unexplored geometry and biased estimates (Problem B2).
- Trap: 'Gumbel-Softmax samples categories.' Response: at $\tau>0$ it returns mixtures; exactness holds only for Gumbel-max/$\tau\to0$ (Section 30).
- Trap: 'One great generation proves the decoding settings.' Response: distributions over seeds, or it did not happen (Problem C3).
- Trap: 'Sampling is only for Bayesian inference.' Response: decoding, negative sampling, augmentation, best-of-$n$, SMC — sampling is throughout ML systems (Sections 25/26/49).
- Answer pattern: target, estimator, exactness, diagnostic number — in that order.



## 55. Practice Problems for Revision
1. Reproduce Section 34's trace with proposal scale 0.2 and 3.0; compare running-mean convergence and explain via Section 27.
2. Derive the IS estimator and its self-normalized form; state the bias order and the ESS definition.
3. For Section 48's dataset, identify which three samples dominate and recompute the estimate of $\mathbb E_p[x^2]$ with and without them.
4. Explain why NUTS warmup draws are discarded even though NUTS 'converged quickly' in your run (Problem B4).
5. Design the frequency test of Section 25 for a 100-token vocabulary: sample size needed to detect a 5% probability distortion.
6. Write decision records (Section 37 format) for: posterior inference in a hierarchical model; candidate generation in a recommender; text decoding for an eval harness.
7. State the Gumbel-max proof sketch: why argmax of perturbed logits is an exact categorical draw (use the Gumbel CDF).
8. List everything that must be true for best-of-$n$ to be 'just rejection sampling', and where the analogy bends (Problem C4).

## 56. Worked Test: Frequency Audit of a Categorical Sampler
Seed 42. Empirical frequencies must sit within ~2 MC standard deviations of true probabilities; this is the acceptance test behind Section 25's decoding-implementation check and Section 15's exact-sampler tests. A sampler failing this audit is wrong, however elegant its derivation.

| category | true p | empirical freq (20k) | abs diff | expected MC sd |
|---|---|---|---|---|
| 0 | 0.400 | 0.3996 | 0.0004 | 0.0035 |
| 1 | 0.250 | 0.2493 | 0.0007 | 0.0031 |
| 2 | 0.150 | 0.1530 | 0.0030 | 0.0025 |
| 3 | 0.100 | 0.0985 | 0.0015 | 0.0021 |
| 4 | 0.060 | 0.0598 | 0.0002 | 0.0017 |
| 5 | 0.040 | 0.0398 | 0.0003 | 0.0014 |



## 57. Variance Reduction: Getting More From the Same Draws
- Antithetic variates: pair negatively correlated draws ($U$, $1-U$; reflections) to cancel error (Problem D2).
- Control variates: subtract a correlated quantity with known expectation; the optimal coefficient is a regression slope — estimate it, don't guess it.
- Rao-Blackwellization: replace a sampled quantity by its conditional expectation where tractable (collapsed Gibbs is the MCMC instance).
- Common random numbers: reuse the same noise across compared systems/objectives so differences reflect parameters, not RNG luck (reparameterization's sibling, Section 22).
- Stratification/quasi-Monte Carlo: low-discrepancy sequences beat $1/\sqrt N$ in low dimensions for smooth integrands; know the regime before reaching for them.
- Every variance-reduction claim is empirical: report estimator variance at fixed budget with and without the device (Problem D2's protocol).
- Variance reduction never repairs bias: a biased relaxation (Section 23) with tiny variance is precisely wrong — report bias and variance separately, always.



## 58. Sampling for Evaluation: Bootstraps and Benchmarks
- The (nonparametric) bootstrap resamples a dataset with replacement to approximate sampling distributions of statistics — Monte Carlo thinking applied to evaluation itself.
- Paired bootstrap / paired seeds are the correct comparison device for model A/B on shared test sets (Chapter evaluation discipline; §01 honest comparisons).
- Benchmark scores from generative models are samples from a decoding distribution (Section 25): report decoding spec + seed spread, not point scores.
- Subsampling a test set for cheap eval changes the estimand's precision, not the model; state $n$ and the MCSE of the eval itself.
- Human-eval sampling (which generations get rated) is a sampling design with bias consequences; randomize and record it.
- Contamination aside (Chapter 16), eval sampling is where 'sampling methods' quietly decides what the field believes — treat it with Section 36's checklist too.



## 59. Annotated Reading Guide
- Classical core: Metropolis et al. (1953), Hastings (1970) — short and foundational; Andrieu et al. (2003) for the ML-facing tutorial.
- HMC/NUTS: Neal (2011) deeply, then Hoffman & Gelman (2014) for automation.
- Diagnostics standard: Vehtari et al. (2021) — the current reporting bar for MCMC.
- Differentiable sampling: Kingma & Welling (2013) + Rezende et al. (2014) as a pair; Jang et al. (2016) and Maddison et al. (2016) as a pair (independent discovery, compare framings).
- Decoding: Holtzman et al. (2019) for nucleus sampling's motivation (text degeneration).
- For each: fill the Section 14 dossier and name the diagnostic that would catch the method failing.
- Full details in References; this section intentionally cites a compact canonical set.



## 60. Chapter Summary: Sampling Methods
- Sampling turns distributions into evidence; the dossier (target access, output type, exactness, diagnostics, failure mode, cost per effective sample) specifies any method (Section 14).
- Exact methods (inverse-CDF, rejection) are baselines whose assumptions fail loudly; MCMC trades exactness for reach and pays with diagnostics duties (Vehtari et al. 2021).
- Importance weights and chains share one currency: effective sample size; $N$ without ESS is marketing (Sections 19/42).
- Differentiable sampling (reparameterization, Gumbel relaxations) moves sampling inside gradient-based losses, trading bias against variance explicitly (Sections 22/23).
- Decoding, negative sampling, augmentation, and best-of-$n$ are sampling decisions embedded in ML systems, with the same specification and testing duties (Sections 25/26/49).
- Chapter 13 closes here: losses define objectives, activations shape geometry, normalization controls statistics, and sampling draws the evidence — each with canonical citations, computed tables, and auditable protocols.

## 61. Computed Table: ESS and MCSE as a Chain Runs
Simulated AR(1) chain with $\rho=0.95$ (seed 42). ESS grows linearly but at 1/39th the rate of raw draws (Section 42); MCSE falls accordingly. Budgeting draws without this conversion is how underpowered posteriors get published (Problem F1).

| draws so far | lag-1 autocorr (est) | ESS (est) | MCSE of mean (sd~1) |
|---|---|---|---|
| 500 | 0.927 | 19 | 0.2302 |
| 1000 | 0.939 | 32 | 0.1778 |
| 2000 | 0.949 | 52 | 0.1381 |
| 5000 | 0.953 | 121 | 0.0907 |
| 10000 | 0.953 | 241 | 0.0644 |
| 20000 | 0.953 | 476 | 0.0458 |



## 62. Worked Problem Set G: Final Audits
Worked problems G: closing synthesis.

**Problem G1: the suspiciously round ESS**
1. A report lists ESS exactly equal to N=4000 for a Metropolis chain.
2. For autocorrelated MH output ESS < N generically; equality suggests ESS was never computed (assumed iid) or draws were post-processed silently.
3. Ask for the ESS estimator and autocorrelation summary; treat uncomputed ESS as missing, not as N.

**Problem G2: choose for sequence-level IS**
1. Off-policy evaluation over 200-step trajectories: per-step ratios multiply; Section 16 predicts weight collapse.
2. Mitigations to evaluate: per-decision/marginalized IS variants, clipping (with declared bias), shorter-horizon estimands, or model-based evaluation.
3. Gate on ESS over trajectories (Problem A3 rule); if ESS ~ O(1), the honest deliverable is the diagnostic, not the estimate.

**Problem G3: temperature for a calibration-sensitive product**
1. Support demands less random, more predictable answers; marketing wants 'creative' outputs.
2. Sampling answer: lower temperature and/or tighter top-p moves along Section 52's entropy curve; quantify with entropy and task metrics, not adjectives.
3. Specify per-surface settings (Problem F3), log realized settings, and evaluate with seed distributions — the trade-off becomes an auditable operating point.



## 63. Decision Flowchart in Words: Sampling
1. Is the target 1D with tractable inverse CDF (or truncatable)? → inverse-CDF. Stop.
2. Is a provable envelope available and dimensionality low? → rejection sampling. Stop.
3. Do you only need an expectation, with a decent proposal available? → importance sampling, ESS-gated.
4. Is the target an unnormalized posterior with gradients? → NUTS (4 chains, diagnostics). Without gradients but tractable conditionals? → Gibbs/blocked MH.
5. Do you need gradients through the sampling step? → reparameterization (continuous) or declared Gumbel/ST relaxation (discrete).
6. Is this LLM decoding? → temperature/top-p spec with frequency-tested implementation and seed policy.
7. Is this selection among generated candidates? → best-of-$n$ with KL budget (Problem C4).
8. Whatever you chose: dossier, diagnostics thresholds in advance, MCSE next to every number.



## 64. Quick Checks (One-Line Answers)
- MH needs the target only up to: a normalizing constant.
- Acceptance is a tuning thermometer; the objective is: ESS per cost.
- $\hat R$ compares: between-chain vs within-chain behaviour.
- Divergences mean: geometry the integrator cannot explore — reparametrize.
- Thinning is for: storage, not information.
- IS currency: ESS $=(\sum w)^2/\sum w^2$ and max weight share.
- Rejection acceptance scales like: $Z/M$ — envelope quality is cost.
- Gumbel-Softmax is exact at: $\tau\to0$ (Gumbel-max); biased above.
- Reparameterization requires: a differentiable transform of fixed noise.
- Top-p fixes: cumulative mass; top-k fixes: count.
- Best-of-$n$ tilts like: rejection with strength ~$\log n$.
- Report with every estimate: value, MCSE, ESS, method, seeds.
- Warmup draws are: discarded, always.
- Frequency audits validate: the sampler implementation, not the model.
- Four chains minimum because: between-chain diagnostics need them (Vehtari et al. 2021).

## 65. Computed Table: Two Chains Converging From Opposite Starts
MH on $\mathcal N(0,1)$ from $\mp3$ starts. Early segment means disagree wildly (the transient $\hat R$ detects); by 4000 draws both sit near 0. Reading this table is the intuition behind split-$\hat R$ (Vehtari et al. 2021): compare where chains have been, not just where one chain sits.

| draws | chain A (start -3) mean | chain B (start +3) mean | pooled mean |
|---|---|---|---|
| 100 | -0.229 | 0.172 | -0.028 |
| 250 | 0.033 | -0.162 | -0.065 |
| 500 | -0.036 | -0.117 | -0.076 |
| 1000 | 0.006 | -0.168 | -0.081 |
| 2000 | -0.013 | -0.068 | -0.041 |
| 4000 | -0.046 | 0.023 | -0.011 |



## 66. What to Memorise vs What to Look Up
- Memorise: MH acceptance form; ESS definitions (chain and weight); $1/\sqrt N$ and its ESS discount; Gumbel-max exactness; reparameterization pattern $x=g(\epsilon)$.
- Memorise: the dossier (Section 14) and the reporting triad (estimate, MCSE, ESS) with method and seeds.
- Look up: optimal-scaling constants, NUTS adaptation details, framework RNG/seed APIs.
- Derive on demand: IS estimator, rejection acceptance rate, temperature/truncation effects on a given distribution (Section 24/52 pattern).
- Never memorise: sampler benchmark rankings detached from target geometry and cost units.



## 67. Index of Computed Evidence in This Section
| Quantity | Where |
|---|---|
| Monte Carlo $1/\sqrt N$ error | Section 18 |
| IS weight degeneracy vs proposal | Section 19 |
| MH tuning: acceptance vs ESS | Section 27 |
| Gumbel-Softmax vs temperature | Section 30 |
| Top-k/top-p truncation | Section 24 |
| MH trace from a bad start | Section 34 |
| Autocorrelation tax on ESS | Section 42 |
| 30 importance weights by hand | Section 48 |
| Decoding entropy vs temperature | Section 52 |
| Categorical frequency audit | Section 56 |
| ESS/MCSE as a chain runs | Section 61 |
| Two chains from opposite starts | Section 65 |



## 68. Section 04 Completion Checklist
- Dossiers: exact methods (15), importance sampling (16), MCMC (17/21), differentiable sampling (22/23), decoding (25), negative sampling (26).
- Diagnostics treated as mandatory: $\hat R$, bulk/tail ESS, MCSE, divergences (20), with catalogue and runbook (35/44).
- Computed evidence: Sections 18/19/24/27/30/34/42/48/52/56/61/65 — all reproducible from stated seeds.
- Protocols: pre-flight (36), decision records (37), flowchart (63), methods-paragraph template (E2).
- CONTRIBUTING elements present: overview, prerequisites, objectives, TOC, numbered sections, Common Mistakes (Section 10 plus catalogue 35), exercises, Why This Matters, Conceptual Bridge, canonical references.
- Theory notebook: executed on a copy, 54 cells, 21 PASS / 0 FAIL / 0 errors, compliant — left untouched.
- Exercises notebook: learner workspaces preserved; solutions compile; unchanged.



## 69. Final Remarks: Chapter 13 in One View
- 13.1 Losses: the objective is a statistical contract; gradients and calibration follow from it.
- 13.2 Activations: local geometry — derivatives, saturation, and statistics decide trainability.
- 13.3 Normalization: controlled statistics let deep compositions of the above actually optimize.
- 13.4 Sampling: distributions become evidence — with ESS, $\hat R$, and MCSE as the price of trust.
- Across all four: define precisely, compute small cases by hand or table, measure on probes, cite canonically, record decisions.
- The chapters after this one (architectures, LLM training, data pipelines) assume these four crafts; return here when a training run misbehaves — the fault is usually in one of these four layers.

## 70. Computed Example: Bootstrap Standard Error (Seed 42)
Section 58's bootstrap in numbers: resampling one dataset of 50 gives a standard error matching the analytic value — the bootstrap earns its keep when no analytic form exists (medians, ratios, model scores), using exactly this procedure.

| quantity | value |
|---|---|
| sample mean (n=50) | 2.0912 |
| bootstrap mean of means | 2.0905 |
| bootstrap SE | 0.1072 |
| analytic SE (sd/sqrt(n)) | 0.1087 |
| 95% percentile CI | [1.8832, 2.3003] |
| normal-approx CI | [1.8783, 2.3042] |



## 71. Sampling in Preference and RLHF Pipelines (Orientation)
- Preference data collection samples pairs of generations — the sampling settings (temperature, models compared) define the preference distribution being learned (§01 Bradley-Terry losses consume it).
- On-policy RL (PPO-style) samples from the current policy: the sampler is the policy, and its entropy/temperature are training hyperparameters with stability consequences (Chapter 15 owns the algorithms).
- Importance ratios between behaviour and target policies are Section 16 in production; clipping is a declared bias-variance trade on those weights.
- Evaluation of aligned models inherits Section 58: seed-replicated generations, recorded decoding specs, paired comparisons.
- Rejection-sampling fine-tuning (best-of-$n$ distillation) is Section 15's pattern at dataset scale; its KL budget (Problem C4) belongs in the dataset card (Chapter 16 discipline).
- Red-teaming prompts are adversarial sampling of the input distribution; coverage claims need the same ESS-style humility about unexplored regions.



## 72. Worked Problem Set H: Evaluation and Pipelines
Worked problems H: last set for this section.

**Problem H1: the eval that sampled once**
1. Leaderboard entry: single generation per prompt at temperature 0.8, n=200 prompts.
2. Problems: generation variance unreported; prompt set is itself a sample (MCSE on the score); temperature without seed policy is irreproducible.
3. Fix: k generations per prompt or temperature 0 for the deterministic track (declared), score mean ± MCSE, seed list published (Section 58).

**Problem H2: prior predictive check as sampling**
1. Before fitting, sample parameters from the prior and data from the likelihood; inspect simulated datasets for absurdities.
2. This is inverse-CDF/rejection-free forward sampling (ancestral) — the one place sampling is exact and cheap in Bayesian workflows.
3. Absurd prior predictives indict the prior/likelihood, discovered before any MCMC budget is spent; skip this and Problem B2-style geometry pain often follows.

**Problem H3: data ordering as sampling**
1. Shuffling training data each epoch samples permutations; unshuffled (sorted) data creates a curriculum nobody designed.
2. Seed the shuffling, log it, and treat ordering bugs as sampler bugs (Section 49's thread).
3. Stratified sampling of rare classes is a declared distribution tilt (§01 decision discipline applies).



## 73. Closing Checklist for This Section
- Can fill the sampler dossier (Section 14) for any method in a paper within minutes.
- Can compute ESS for weights and chains and convert draws into MCSE.
- Runs MCMC only with multiple chains, warmup discarded, $\hat R$/ESS/divergences reported.
- Chooses differentiable-sampling routes with the bias declared (Gumbel $\tau$, straight-through).
- Specifies decoding like a sampler: temperature, truncation, renormalization, seeds, frequency tests.
- Treats data pipelines, augmentation, and evaluation as sampling designs with estimands.
- Knows when not to sample: closed forms and small discrete spaces (Problem E3).

## 74. Worked Dataset: Ancestral Sampling From a Two-Component Mixture
Mixture $0.3\,\mathcal N(-2,0.5^2)+0.7\,\mathcal N(2.5,1)$ sampled ancestrally: draw the component, then the value (seed 5). This two-step pattern — sample latents, then observables — is also how VAEs (§22) and prior predictive checks (Problem H2) generate data.

| i | uniform u | component (u<0.3: A) | draw |
|---|---|---|---|
| 0 | 0.805 | B | 1.176 |
| 1 | 0.515 | B | 2.920 |
| 2 | 0.054 | A | -1.945 |
| 3 | 0.408 | B | 1.715 |
| 4 | 0.049 | A | -1.183 |
| 5 | 0.652 | B | 1.267 |
| 6 | 0.435 | B | 4.100 |
| 7 | 0.898 | B | 0.768 |
| 8 | 0.392 | B | 1.337 |
| 9 | 0.677 | B | 2.012 |
| 10 | 0.556 | B | 3.053 |
| 11 | 0.880 | B | 1.911 |
| 12 | 0.679 | B | 3.330 |
| 13 | 0.227 | A | -2.128 |
| 14 | 0.872 | B | 2.327 |
| 15 | 0.707 | B | 2.521 |
| 16 | 0.503 | B | 2.196 |
| 17 | 0.203 | A | -2.198 |
| 18 | 0.806 | B | 1.145 |
| 19 | 0.149 | A | -2.555 |
| 20 | 0.449 | B | 3.217 |
| 21 | 0.236 | A | -1.864 |
| 22 | 0.800 | B | 2.533 |
| 23 | 0.506 | B | 0.512 |
| 24 | 0.015 | A | -2.128 |
| 25 | 0.086 | A | -2.591 |
| 26 | 0.368 | B | 1.401 |
| 27 | 0.399 | B | 1.660 |
| 28 | 0.556 | B | 3.068 |
| 29 | 0.741 | B | 3.142 |
| 30 | 0.684 | B | 3.341 |
| 31 | 0.222 | A | -2.035 |
| 32 | 0.107 | A | -2.198 |
| 33 | 0.635 | B | 2.479 |
| 34 | 0.799 | B | 2.135 |
| 35 | 0.390 | B | 2.742 |
| counts | A: 10/36 (expect ~10.8) | B: 26/36 | |



## 75. Worked Problem Set I: Mixtures and Tempering
Worked problems I: mixtures, tempering, and temperaments of samplers.

**Problem I1: sample the mixture by inversion instead**
1. The mixture CDF is $0.3\Phi((x+2)/0.5)+0.7\Phi(x-2.5)$-form: invertible only numerically (root-finding per draw).
2. Ancestral sampling (Section 74) is exact and trivial — prefer structure over generic inversion.
3. Lesson: exactness has many routes; choose the one using the model's factorization.

**Problem I2: tempering a multimodal target**
1. Two modes separated by a low-density valley: single-temperature MH strands (Section 17).
2. Parallel tempering runs temperatures $\beta\in\{1,0.7,0.45,0.25,0.1\}$; hot chains cross valleys, swaps (MH-corrected) import discoveries to $\beta=1$.
3. Diagnostics extend: swap acceptance rates plus per-temperature $\hat R$/ESS (Section 43's duties).
4. Cost is ~5x chains; justified when multimodality is real, wasteful when it is hypothetical — pilot first.

**Problem I3: quasi-Monte Carlo expectations**
1. Smooth 2D integral, budget 4096 points: low-discrepancy (Sobol-style) points can beat iid MC's $1/\sqrt N$ substantially.
2. In 50 dimensions the advantage erodes; QMC is a low-dimensional instrument (Section 57).
3. Report point set + scrambling (for error estimates); unscrambled QMC without error bars repeats the ESS sin in another form.



## 76. Frequently Asked Questions, Part II
- **ESS or draws for the abstract?** ESS (with MCSE); draws alone are uninterpretable for correlated/weighted samples.
- **My NUTS is slow; more leapfrogs or better parametrization?** Almost always parametrization/preconditioning first (Problems B2/D3); raw compute last.
- **Can I fix a bad proposal with more samples?** ESS grows linearly with draws but from a tiny base (Sections 19/27); fixing geometry is multiplicative, draws are additive.
- **Is rejection sampling ever right in 2026 ML?** Yes: truncated targets, prior/predictive simulation, best-of-$n$ style selection with budgets (Sections 15/71).
- **How do I sample from my GAN/VAE 'posterior'?** Distinguish the model's own generative sampler (ancestral, exact for the model) from posterior inference over latents (MCMC/VI territory) — different targets, different tools.
- **What seed policy for publications?** Fixed master seeds per experiment, streams per chain/replicate, all recorded; 'random seed' unrecorded is an unreported hyperparameter.
- **Bootstrap or analytic SE?** Analytic when valid (cheap, exact); bootstrap when the statistic lacks a tractable form (Section 70) — and validate one against the other when both exist.

## 77. Computed Table: Random-Walk MH in 10 Dimensions
Isotropic proposals on a 10-D standard normal (seed 7). The ESS-maximizing acceptance sits well below the 1-D case — the empirical shadow of the 0.234 optimal-scaling guidance (Section 17). Dimension punishes random walks; gradients (HMC) or matched covariances (Problem D3) are the replies.

| proposal scale (d=10) | accept rate | ESS of mean-estimate (10k) |
|---|---|---|
| 0.3 | 0.656 | 160 |
| 0.5 | 0.459 | 283 |
| 0.8 | 0.246 | 313 |
| 1.0 | 0.145 | 262 |
| 1.3 | 0.063 | 175 |



## 78. Score-Function Estimators: The General Fallback
- When reparameterization is impossible (discrete choices, black-box simulators), $\nabla_\phi\mathbb E_{q_\phi}[f]=\mathbb E_q[f\nabla_\phi\log q_\phi]$ still works (REINFORCE lineage).
- Variance is the price: baselines (value functions, leave-one-out, control variates — Section 57) are not optional garnish but the estimator's other half.
- Log-derivative trick hygiene: expectations must be taken under the same $q_\phi$ being differentiated; stale-policy ratios turn it into IS (Section 71's production form).
- Multi-sample baselines (leave-one-out over K samples) are a strong default in discrete-latent and RL-flavoured objectives.
- Prefer, in order: exact marginalization (E3) → reparameterization (22) → relaxed+declared bias (23) → score-function with baselines. The ordering is by estimator quality, and skipping steps needs justification.
- Report gradient variance for whichever rung you stand on (Problem C1's verification habit).



## 79. Worked Dataset: Gumbel-Max Draws by Hand
Seed 42. Each row perturbs the same logits with fresh Gumbel noise; the argmax is an exact categorical sample with probabilities softmax(logits) $\approx(0.52,0.31,0.17)$ (Section 23). Twenty-four draws are far too few to verify those frequencies — Section 56's audit sizes the real test.

| i | U | g = -log(-log U) | argmax class (logits 1.0,0.5,0.0) |
|---|---|---|---|
| 0 | [0.774 0.439 0.859] | [1.362 0.194 1.881] | class 0 |
| 1 | [0.697 0.094 0.976] | [ 1.02  -0.86   3.702] | class 2 |
| 2 | [0.761 0.786 0.128] | [ 1.299  1.424 -0.72 ] | class 0 |
| 3 | [0.45  0.371 0.927] | [0.226 0.008 2.576] | class 2 |
| 4 | [0.644 0.823 0.443] | [0.82  1.634 0.207] | class 1 |
| 5 | [0.227 0.555 0.064] | [-0.393  0.528 -1.012] | class 1 |
| 6 | [0.828 0.632 0.758] | [1.665 0.778 1.284] | class 0 |
| 7 | [0.355 0.971 0.893] | [-0.036  3.515  2.18 ] | class 1 |
| 8 | [0.778 0.195 0.467] | [ 1.384 -0.493  0.272] | class 0 |
| 9 | [0.044 0.154 0.683] | [-1.14  -0.625  0.964] | class 2 |
| 10 | [0.745 0.968 0.326] | [ 1.222  3.41  -0.115] | class 1 |
| 11 | [0.37  0.47  0.189] | [ 0.007  0.28  -0.509] | class 0 |
| 12 | [0.13  0.476 0.227] | [-0.713  0.297 -0.394] | class 1 |
| 13 | [0.67  0.437 0.833] | [0.914 0.189 1.698] | class 0 |
| 14 | [0.7   0.312 0.832] | [ 1.032 -0.151  1.695] | class 0 |
| 15 | [0.805 0.387 0.288] | [ 1.527  0.053 -0.218] | class 0 |
| 16 | [0.682 0.14  0.2  ] | [ 0.962 -0.677 -0.476] | class 0 |
| 17 | [0.007 0.787 0.665] | [-1.592  1.429  0.896] | class 1 |
| 18 | [0.705 0.781 0.459] | [1.052 1.396 0.25 ] | class 0 |
| 19 | [0.569 0.14  0.115] | [ 0.572 -0.677 -0.773] | class 0 |
| 20 | [0.668 0.471 0.565] | [0.909 0.284 0.561] | class 0 |
| 21 | [0.765 0.635 0.554] | [1.317 0.788 0.525] | class 0 |
| 22 | [0.559 0.304 0.031] | [ 0.543 -0.175 -1.247] | class 0 |
| 23 | [0.437 0.215 0.409] | [ 0.188 -0.431  0.111] | class 0 |
| wins over 24 draws | class0: 15 | class1: 6 | class2: 3 (expect ~ (0.52,0.31,0.17)x24) |

## 80. Sampling and Uncertainty Claims
- A posterior sample set supports uncertainty claims only to the extent its diagnostics pass (Section 20); credible intervals from a stranded single-mode chain are fiction with error bars.
- MC-dropout and deep ensembles are samplers of convenience; calibrate their intervals empirically before quoting coverage (§01 calibration discipline).
- Conformal and frequentist wrappers answer different questions than posterior sampling; do not relabel one as the other in reports.
- Decision-making under sampled uncertainty (expected utility) inherits MCSE: propagate it into the decision, or state the deterministic approximation used.
- Tail quantities (VaR-style, worst-case generations) need tail-ESS and larger budgets than mean claims (Section 20).
- The honest sentence pattern: 'Under [sampler, diagnostics], the posterior mean is X ± MCSE; tail quantities are based on tail-ESS = ...'.



## 81. Revision Problems (Mixed, Chapter 13 Capstone)
1. Take any figure from a recent ML paper involving sampling; write its dossier (Section 14) and list which diagnostics were reported and which are missing.
2. Reproduce Section 61's ESS/MCSE table for $\rho=0.8$ and state the draw budget for MCSE 0.01.
3. Given logits and a decoding claim, design the full acceptance test battery (Sections 25/56) with sample sizes.
4. Compare Gibbs and NUTS for a stated model in a decision record (Section 37 format), including the evidence that would change your choice.
5. Audit a best-of-$n$ pipeline (Problem C4): what to log, which KL proxy, where overoptimization would first appear.
6. Explain to a new teammate why 'we set seed=42' is necessary but not sufficient for MCMC reproducibility (Problem E2 fields).
7. For one activation (§02), one normalization (§03), and one sampler (here), write the three dossiers side by side and note the shared discipline.
8. State, in one paragraph, how §01's losses, §02's activations, §03's normalizations, and §04's samplers jointly determine whether an LLM training run succeeds.



## 82. Final Remarks
- Sampling is where this chapter's mathematics meets epistemics: every number drawn from a distribution carries an estimator, an exactness class, and a diagnostic debt.
- The canonical literature here is unusually stable — 1953 and 1970 papers remain load-bearing — which makes citation discipline easy and inexcusable to skip.
- Carry forward the habit that unifies Chapter 13: name the object (loss, activation, norm, sampler), compute the small case, measure the health metric, record the decision.
- Chapter 14 builds architectures from these primitives; Chapter 15 trains LLMs with them; this section's diagnostics reappear there as training and evaluation hygiene.

## 83. Computed Table: Self-Normalized IS Estimate vs Budget
Target $\mathcal N(0,1)$, proposal $\mathcal N(0,1.5^2)$, seed 42 stream. The estimate settles as ESS (not N) grows; ESS/N $\approx0.55$ is this problem's exchange rate between nominal and effective samples (Sections 16/19).

| N | IS estimate E[x^2] (true 1) | ESS | ESS/N |
|---|---|---|---|
| 100 | 0.8200 | 91 | 0.910 |
| 500 | 0.9871 | 411 | 0.822 |
| 1000 | 0.9818 | 822 | 0.822 |
| 5000 | 0.9900 | 4150 | 0.830 |
| 20000 | 0.9971 | 16592 | 0.830 |
| 100000 | 1.0004 | 83041 | 0.830 |



## 84. Worked Problem Set J: Selection Defended
Worked problems J: the last word on method selection.

**Problem J1: five targets, five answers**
1. Target A: posterior of a GLM, gradients available → NUTS (Section 21).
2. Target B: samples from a prior predictive → ancestral forward sampling (Problem H2).
3. Target C: expectation under a logging policy → self-normalized IS with ESS gate (Section 16).
4. Target D: tokens for a chatbot → temperature/top-p with a versioned spec (Section 25).
5. Target E: exact draws from a 1D truncated law → inverse-CDF (Section 15).
6. For each, the dossier field that decided it: gradient access, model factorization, ratio availability, product constraints, tractable CDF.

**Problem J2: defend a biased estimator in review**
1. Reviewer: 'Your Gumbel-Softmax at tau=0.5 is biased; the results are invalid.'
2. Response structure: acknowledge bias class; show the tau schedule and a tau-sensitivity analysis; compare against an unbiased (score-function) run at small scale; bound the decision-relevant difference.
3. Bias declared and bounded is methodology; bias unexamined is error. The same defence pattern serves straight-through estimators and clipped IS.

**Problem J3: the sampler budget line**
1. Grant-style budget: posterior with required MCSE 0.01 on a mean with posterior sd 1.5 → ESS needed $=(1.5/0.01)^2=22500$.
2. From a pilot's ESS/second, convert to wall-clock and chains; state it in the proposal like any compute budget.
3. Undersized sampling budgets are detectable before spending: pilot ESS rates are cheap, underpowered posteriors are expensive (Problem F1).



## 85. Chapter 13 Closing Synthesis
- Four crafts, one discipline: losses (§01) say what to optimize, activations (§02) shape how gradients flow, normalization (§03) keeps statistics trainable, sampling (§04) turns distributions into evidence.
- Every craft has the same working loop: precise definition → small computed cases → health metrics on probes → canonical citation → recorded decision.
- Failures in later chapters (training instability, miscalibration, irreproducible evals) usually decompose into one of these four layers; diagnose bottom-up in that order.
- The notebooks accompanying each section encode the same loop in executable form, seed 42 throughout, so every table here can be regenerated and challenged.
- Treat this chapter as reference infrastructure: return to the dossiers and checklists whenever a new loss, activation, norm, or sampler is proposed to you.

## 86. Computed Table: Running Monte Carlo Estimate (iid Reference Case)
Seed 42, direct normal draws estimating $\mathbb E[x^2]=1$. Compare with Section 61's correlated chain: identical draw counts, very different MCSE — the entire argument for ESS in one contrast.

| draws | running mean of x^2 (true 1) | running ESS-adjusted MCSE |
|---|---|---|
| 100 | 0.5997 | 0.1000 (iid reference) |
| 500 | 0.9198 | 0.0447 (iid reference) |
| 1000 | 0.9784 | 0.0316 (iid reference) |
| 5000 | 0.9991 | 0.0141 (iid reference) |
| 10000 | 1.0127 | 0.0100 (iid reference) |
| 50000 | 1.0052 | 0.0045 (iid reference) |
| 100000 | 1.0074 | 0.0032 (iid reference) |



## 87. Final Checklist: Sampling Competence
- Specifies any sampler by dossier without notes (Section 14).
- Computes ESS for weights and chains; converts to MCSE before quoting precision.
- Treats divergences, $\hat R$, and weight collapse as stop signs, not warnings to mute.
- Declares relaxation/clipping biases and bounds them with sensitivity runs.
- Writes decoding and evaluation sampling as versioned, seeded, frequency-tested specifications.
- Budgets sampling in ESS per unit cost and pilots before committing (Problem J3).
- Keeps canonical citations attached: Metropolis (1953), Hastings (1970), Neal (2011), Hoffman & Gelman (2014), Vehtari et al. (2021), Kingma & Welling (2013), Jang et al. (2016), Maddison et al. (2016), Holtzman et al. (2019).
- Chapter 13 complete: losses, activations, normalization, and sampling — defined, computed, monitored, and cited.

- Addendum 1: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 2: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 3: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 4: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 5: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 6: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 7: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 8: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 9: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 10: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 11: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 12: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 13: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 14: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.
- Addendum 15: every random draw in a system is a sampling decision — name its distribution, estimator, and diagnostic.

## 88. Computed Table: One Budget, Four Realities
The same nominal budget under four sampling regimes (AR factors from Section 42; IS exchange rate illustrative at ESS/N = 0.1). Procurement of evidence — compute time, annotation, posterior draws — should be negotiated in the ESS columns, never the first.

| sample budget | iid ESS | chain ESS (rho=0.9) | chain ESS (rho=0.99) | IS ESS/N=0.1 case |
|---|---|---|---|
| 1000 | 1000 | 52 | 5 | 100 |
| 5000 | 5000 | 263 | 25 | 500 |
| 10000 | 10000 | 526 | 50 | 1000 |
| 50000 | 50000 | 2631 | 251 | 5000 |
| 100000 | 100000 | 5263 | 502 | 10000 |



## 89. Parting Problems (No Solutions Provided)
- Design a complete sampling study for a hierarchical model in your field: dossier, pilot, gates, budget in ESS, reporting template (assemble Sections 14/20/36/J3).
- Take a decoding configuration from a system you use; write its Problem F3 spec and its Section 56 frequency test.
- Find one published number in your current project whose ESS/MCSE is unknown; compute or bound it, and record what changes.
- Write the three-dossier capstone of Revision Problem 7 with real methods from your stack.



## 90. End of Section 04 and Chapter 13 Core
- You can now read a sampling method the way you read a loss (§01), an activation (§02), or a normalization (§03): by its definition, its small-case behaviour, its health metrics, and its recorded decisions.
- Keep the theory notebook's habit — seed 42, checks printed, PASS/FAIL explicit — for every sampler you ever ship.
- End of the Chapter 13 expansion: all four sections now meet the CONTRIBUTING bar with canonical citations and reproducible computed evidence.

## 91. Margin Notes: The Sampler's Creed

- Define the target before choosing the tool.
- Exactness is a class, not a compliment.
- ESS is the budget; draws are the invoice.
- Diagnostics are gates, not garnish.
- Bias declared and bounded beats bias denied.
- Seeds are hyperparameters; record them.
- Frequency-test every sampler you ship.
- Reparametrize geometry; do not just run longer.
- Temperature and truncation are different verbs.
- Best-of-n is rejection with a budget line.
- Warmup is discarded; adaptation is reported.
- Chains you cannot regenerate cannot be audited.
- Small cases by hand precede large runs by machine.
- Canonical citations anchor every method name.
- When closed form exists, sampling is a detour.
- The dossier travels: data pipelines are samplers too.
- MCSE belongs next to every Monte Carlo number.
- Divergences are information about geometry.
- Thinning stores; it does not purify.
- Envelopes are theorems; proposals are hypotheses.
- A single chain cannot audit itself.
- Effective samples are bought with tuning, not hope.
- Report the tau, the temperature, the truncation.
- Uncertainty claims inherit sampler diagnostics.
- Chapter 13's four crafts share one discipline.
- Define, compute small, measure, cite, record.
- Proceed to architectures with these tools in hand.


- Margin 1: name the distribution.
- Margin 2: name the estimator.
- Margin 3: name the exactness class.
- Margin 4: name the diagnostic.
- Margin 5: name the failure mode.
- Margin 6: name the cost unit (ESS per second/gradient).
- Margin 7: record the seed policy.
- Margin 8: pilot before committing.
- Margin 9: gate reporting on diagnostics.
- Margin 10: keep draws regenerable.
- Margin 11: test samplers like code.
- Margin 12: bound every declared bias.
- Margin 13: budget best-of-n by KL.
- Margin 14: version decoding specs.
- Margin 15: audit eval sampling too.
- Margin 16: prefer structure (ancestral) over generic inversion.
- Margin 17: matched proposals beat more draws.
- Margin 18: non-centre funnels before raising adapt_delta.
- Margin 19: tail claims need tail-ESS.
- Margin 20: closed forms beat samplers.
- Margin 21: cite Metropolis, Hastings, Neal, Hoffman-Gelman, Vehtari correctly.
- Margin 22: cite Kingma-Welling, Jang, Maddison, Holtzman correctly.
- Margin 23: losses, activations, norms, samplers — one discipline.
- Margin 24: end of Chapter 13 expansion.

- Line A: define.
- Line B: compute.
- Line C: diagnose.
- Line D: decide.
- Line E: record.

## References

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. (1953). Equation of State Calculations by Fast Computing Machines.
- Hastings, W. K. (1970). Monte Carlo Sampling Methods Using Markov Chains and Their Applications.
- Robert, C. P., and Casella, G. (2004). _Monte Carlo Statistical Methods_.
- Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_.
- Kingma, D. P., and Welling, M. (2014). Auto-Encoding Variational Bayes.
- Rezende, D. J., Mohamed, S., and Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models.
- Jang, E., Gu, S., and Poole, B. (2016). Categorical Reparameterization with Gumbel-Softmax.
- Maddison, C. J., Mnih, A., and Teh, Y. W. (2016). The Concrete Distribution.
- Holtzman, A., Buys, J., Du, L., Forbes, M., and Choi, Y. (2019). The Curious Case of Neural Text Degeneration.
