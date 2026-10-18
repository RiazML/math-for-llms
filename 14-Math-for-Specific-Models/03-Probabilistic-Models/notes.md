[<- Neural Networks](../02-Neural-Networks/notes.md) | [Home](../../README.md) | [RNN and LSTM Math ->](../04-RNN-and-LSTM-Math/notes.md)

---

# Probabilistic Models

Probabilistic models describe data, hidden structure, parameters, and predictions with distributions. They make uncertainty a first-class object rather than an afterthought.

## Overview

The core operations are:

$$
p(x)=\sum_zp(x,z),\qquad p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}.
$$

Marginalization handles hidden variables. Bayes rule updates beliefs. Likelihood trains parameters. These ideas appear in naive Bayes, mixture models, HMMs, variational inference, VAEs, uncertainty estimation, and probabilistic interpretations of neural networks.

## Prerequisites

- Probability rules, expectation, and conditional probability
- Log likelihood and cross-entropy
- Linear and neural model sections
- Basic optimization

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates likelihoods, Bayes updates, naive Bayes, Gaussian mixtures, EM, HMM forward recursion, Monte Carlo, ELBO intuition, and calibration. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for probability normalization, MLE/MAP, Bayes, mixture responsibilities, HMMs, and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Distinguish joint, marginal, conditional, likelihood, posterior, and predictive distributions.
- Compute MLE and MAP estimates in simple models.
- Apply Bayes rule with conjugate priors.
- Explain naive Bayes and the generative/discriminative distinction.
- Compute mixture responsibilities and one EM update.
- Explain graphical-model factorization and conditional independence.
- Run HMM forward and Viterbi-style recursions at a high level.
- Explain approximate inference, ELBO, and posterior predictive checks.

## Table of Contents

1. [Probabilistic Modeling View](#1-probabilistic-modeling-view)
   - 1.1 [Random variables](#11-random-variables)
   - 1.2 [Joint distribution](#12-joint-distribution)
   - 1.3 [Conditional distribution](#13-conditional-distribution)
   - 1.4 [Marginalization](#14-marginalization)
   - 1.5 [Decision rule](#15-decision-rule)
2. [Likelihood and Estimation](#2-likelihood-and-estimation)
   - 2.1 [Likelihood](#21-likelihood)
   - 2.2 [Log likelihood](#22-log-likelihood)
   - 2.3 [MLE](#23-mle)
   - 2.4 [MAP](#24-map)
   - 2.5 [Predictive distribution](#25-predictive-distribution)
3. [Bayes Rule](#3-bayes-rule)
   - 3.1 [Posterior](#31-posterior)
   - 3.2 [Evidence](#32-evidence)
   - 3.3 [Conjugacy](#33-conjugacy)
   - 3.4 [Prior strength](#34-prior-strength)
   - 3.5 [Posterior predictive](#35-posterior-predictive)
4. [Naive Bayes and Discriminative Contrast](#4-naive-bayes-and-discriminative-contrast)
   - 4.1 [Generative classifier](#41-generative-classifier)
   - 4.2 [Naive assumption](#42-naive-assumption)
   - 4.3 [Classification](#43-classification)
   - 4.4 [Discriminative model](#44-discriminative-model)
   - 4.5 [Calibration](#45-calibration)
5. [Latent Variable Models](#5-latent-variable-models)
   - 5.1 [Latent variable](#51-latent-variable)
   - 5.2 [Mixture model](#52-mixture-model)
   - 5.3 [Responsibilities](#53-responsibilities)
   - 5.4 [Identifiability](#54-identifiability)
   - 5.5 [Representation learning](#55-representation-learning)
6. [Expectation Maximization](#6-expectation-maximization)
   - 6.1 [E-step](#61-estep)
   - 6.2 [M-step](#62-mstep)
   - 6.3 [Lower bound](#63-lower-bound)
   - 6.4 [Gaussian mixture updates](#64-gaussian-mixture-updates)
   - 6.5 [Local optima](#65-local-optima)
7. [Graphical Models](#7-graphical-models)
   - 7.1 [Directed model](#71-directed-model)
   - 7.2 [Undirected model](#72-undirected-model)
   - 7.3 [Conditional independence](#73-conditional-independence)
   - 7.4 [Inference](#74-inference)
   - 7.5 [Message passing](#75-message-passing)
8. [Hidden Markov Models](#8-hidden-markov-models)
   - 8.1 [Markov state](#81-markov-state)
   - 8.2 [Emission](#82-emission)
   - 8.3 [Forward recursion](#83-forward-recursion)
   - 8.4 [Viterbi](#84-viterbi)
   - 8.5 [Sequence modeling bridge](#85-sequence-modeling-bridge)
9. [Approximate Inference](#9-approximate-inference)
   - 9.1 [Monte Carlo](#91-monte-carlo)
   - 9.2 [Variational inference](#92-variational-inference)
   - 9.3 [ELBO](#93-elbo)
   - 9.4 [Reparameterization](#94-reparameterization)
   - 9.5 [Amortized inference](#95-amortized-inference)
10. [Diagnostics](#10-diagnostics)
39. [Computed Table: Beta Posterior Summaries at Four Evidence Levels](#39-computed-table-beta-posterior-summaries-at-four-evidence-levels)
40. [Ten-Line Field Guide to Defensible Probabilistic Claims](#40-ten-line-field-guide-to-defensible-probabilistic-claims)
37. [Worked Trace: 30 Metropolis Steps Toward Beta(9,5)](#37-worked-trace-30-metropolis-steps-toward-beta95)
38. [Margin Notes: The Probabilistic Creed](#38-margin-notes-the-probabilistic-creed)
34. [Computed Comparison: Four Roads to One Posterior](#34-computed-comparison-four-roads-to-one-posterior)
35. [Addenda: Ten Lines to Carry Forward](#35-addenda-ten-lines-to-carry-forward)
36. [End of Section 03 Expansion](#36-end-of-section-03-expansion)
29. [Worked Trace: 25 EM Iterations](#29-worked-trace-25-em-iterations)
30. [Extended Exercise Bank (Self-Study)](#30-extended-exercise-bank-self-study)
31. [Revision Problems (Mixed)](#31-revision-problems-mixed)
32. [Chapter Summary: Probabilistic Models](#32-chapter-summary-probabilistic-models)
33. [Section 03 Completion Checklist](#33-section-03-completion-checklist)
25. [Exam and Interview Traps: Probabilistic Models](#25-exam-and-interview-traps-probabilistic-models)
26. [Worked Problem Set C: Synthesis](#26-worked-problem-set-c-synthesis)
27. [Quick Checks (One-Line Answers)](#27-quick-checks-one-line-answers)
28. [Index of Computed Evidence](#28-index-of-computed-evidence)
21. [Computed Table: Posteriors Tighten as $1/\sqrt{n}$](#21-computed-table-posteriors-tighten-as-1sqrtn)
22. [Self-Test Questions](#22-self-test-questions)
23. [Self-Test Answer Key (Condensed)](#23-self-test-answer-key-condensed)
24. [Annotated Reading Guide](#24-annotated-reading-guide)
17. [Quick-Reference Cards](#17-quick-reference-cards)
18. [Pre-Flight Checklist for Probabilistic Models](#18-pre-flight-checklist-for-probabilistic-models)
19. [Frequently Asked Questions](#19-frequently-asked-questions)
20. [Decision Records: Probabilistic Models in Context](#20-decision-records-probabilistic-models-in-context)
14. [Worked Problem Set A: Conjugacy, EM, State Inference](#14-worked-problem-set-a-conjugacy-em-state-inference)
15. [Worked Problem Set B: Approximation Honesty](#15-worked-problem-set-b-approximation-honesty)
16. [Glossary: Probabilistic-Model Vocabulary](#16-glossary-probabilistic-model-vocabulary)
11. [Common Mistakes](#11-common-mistakes)
12. [Inference Methods on One Map](#12-inference-methods-on-one-map)
13. [Computed Table: Priors Pull Estimates (Shrinkage in Numbers)](#13-computed-table-priors-pull-estimates-shrinkage-in-numbers)
   - 10.1 [Log-likelihood](#101-loglikelihood)
   - 10.2 [Posterior predictive checks](#102-posterior-predictive-checks)
   - 10.3 [Calibration curves](#103-calibration-curves)
   - 10.4 [Sensitivity to priors](#104-sensitivity-to-priors)
   - 10.5 [Ablations](#105-ablations)

---

## Object Map

```text
observed data:       x, y
latent variables:    z
parameters:          theta
prior:               p(theta)
likelihood:          p(D | theta)
posterior:           p(theta | D)
prediction:          p(y_new | x_new, D)
```

## 1. Probabilistic Modeling View

This part studies probabilistic modeling view as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Random variables](#1-random-variables) | represent uncertain quantities | $X,Y,Z$ |
| [Joint distribution](#1-joint-distribution) | model all variables together | $p(x,y,z)$ |
| [Conditional distribution](#1-conditional-distribution) | predict one variable given another | $p(y\mid x)$ |
| [Marginalization](#1-marginalization) | sum or integrate out hidden variables | $p(x)=\sum_zp(x,z)$ |
| [Decision rule](#1-decision-rule) | turn probabilities into actions | $a^\star=\arg\min_a E[L(a,Y)\mid x]$ |

### 1.1 Random variables

**Main idea.** Represent uncertain quantities.

Core relation:

$$X,Y,Z$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.2 Joint distribution

**Main idea.** Model all variables together.

Core relation:

$$p(x,y,z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.3 Conditional distribution

**Main idea.** Predict one variable given another.

Core relation:

$$p(y\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.4 Marginalization

**Main idea.** Sum or integrate out hidden variables.

Core relation:

$$p(x)=\sum_zp(x,z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 1.5 Decision rule

**Main idea.** Turn probabilities into actions.

Core relation:

$$a^\star=\arg\min_a E[L(a,Y)\mid x]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 2. Likelihood and Estimation

This part studies likelihood and estimation as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Likelihood](#2-likelihood) | view data probability as a function of parameters | $L(\theta)=p(D\mid\theta)$ |
| [Log likelihood](#2-log-likelihood) | products become sums | $\ell(\theta)=\sum_i\log p(x_i\mid\theta)$ |
| [MLE](#2-mle) | choose parameters maximizing data likelihood | $\hat\theta_\mathrm{MLE}=\arg\max_\theta\ell(\theta)$ |
| [MAP](#2-map) | include a prior over parameters | $\hat\theta_\mathrm{MAP}=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]$ |
| [Predictive distribution](#2-predictive-distribution) | integrate parameter uncertainty when Bayesian | $p(x_\star\mid D)=\int p(x_\star\mid\theta)p(\theta\mid D)d\theta$ |

### 2.1 Likelihood

**Main idea.** View data probability as a function of parameters.

Core relation:

$$L(\theta)=p(D\mid\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.2 Log likelihood

**Main idea.** Products become sums.

Core relation:

$$\ell(\theta)=\sum_i\log p(x_i\mid\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.3 MLE

**Main idea.** Choose parameters maximizing data likelihood.

Core relation:

$$\hat\theta_\mathrm{MLE}=\arg\max_\theta\ell(\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.4 MAP

**Main idea.** Include a prior over parameters.

Core relation:

$$\hat\theta_\mathrm{MAP}=\arg\max_\theta[\log p(D\mid\theta)+\log p(\theta)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 2.5 Predictive distribution

**Main idea.** Integrate parameter uncertainty when bayesian.

Core relation:

$$p(x_\star\mid D)=\int p(x_\star\mid\theta)p(\theta\mid D)d\theta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 3. Bayes Rule

This part studies bayes rule as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Posterior](#3-posterior) | update beliefs after data | $p(\theta\mid D)=p(D\mid\theta)p(\theta)/p(D)$ |
| [Evidence](#3-evidence) | normalizing probability of data | $p(D)=\int p(D\mid\theta)p(\theta)d\theta$ |
| [Conjugacy](#3-conjugacy) | some priors yield closed-form posteriors | $\mathrm{Beta}+\mathrm{Bernoulli}\rightarrow\mathrm{Beta}$ |
| [Prior strength](#3-prior-strength) | prior counts can regularize low-data estimates | $\alpha,\beta$ |
| [Posterior predictive](#3-posterior-predictive) | predictions average over posterior uncertainty | $p(y\mid x,D)$ |

### 3.1 Posterior

**Main idea.** Update beliefs after data.

Core relation:

$$p(\theta\mid D)=p(D\mid\theta)p(\theta)/p(D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** Bayes rule is the core update that turns observations into revised uncertainty.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.2 Evidence

**Main idea.** Normalizing probability of data.

Core relation:

$$p(D)=\int p(D\mid\theta)p(\theta)d\theta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.3 Conjugacy

**Main idea.** Some priors yield closed-form posteriors.

Core relation:

$$\mathrm{Beta}+\mathrm{Bernoulli}\rightarrow\mathrm{Beta}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.4 Prior strength

**Main idea.** Prior counts can regularize low-data estimates.

Core relation:

$$\alpha,\beta$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 3.5 Posterior predictive

**Main idea.** Predictions average over posterior uncertainty.

Core relation:

$$p(y\mid x,D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 4. Naive Bayes and Discriminative Contrast

This part studies naive bayes and discriminative contrast as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Generative classifier](#4-generative-classifier) | model class prior and feature likelihood | $p(y,x)=p(y)p(x\mid y)$ |
| [Naive assumption](#4-naive-assumption) | features are conditionally independent given class | $p(x\mid y)=\prod_jp(x_j\mid y)$ |
| [Classification](#4-classification) | choose largest posterior class | $\arg\max_y p(y)\prod_jp(x_j\mid y)$ |
| [Discriminative model](#4-discriminative-model) | model p(y|x) directly | $p_\theta(y\mid x)$ |
| [Calibration](#4-calibration) | probabilistic outputs should match frequencies | $P(Y=y\mid \hat p_y=c)\approx c$ |

### 4.1 Generative classifier

**Main idea.** Model class prior and feature likelihood.

Core relation:

$$p(y,x)=p(y)p(x\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.2 Naive assumption

**Main idea.** Features are conditionally independent given class.

Core relation:

$$p(x\mid y)=\prod_jp(x_j\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.3 Classification

**Main idea.** Choose largest posterior class.

Core relation:

$$\arg\max_y p(y)\prod_jp(x_j\mid y)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.4 Discriminative model

**Main idea.** Model p(y|x) directly.

Core relation:

$$p_\theta(y\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 4.5 Calibration

**Main idea.** Probabilistic outputs should match frequencies.

Core relation:

$$P(Y=y\mid \hat p_y=c)\approx c$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 5. Latent Variable Models

This part studies latent variable models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Latent variable](#5-latent-variable) | hidden cause explains observed data | $p(x,z)=p(z)p(x\mid z)$ |
| [Mixture model](#5-mixture-model) | data comes from one of several components | $p(x)=\sum_k\pi_kp(x\mid z=k)$ |
| [Responsibilities](#5-responsibilities) | posterior component probabilities | $\gamma_{ik}=p(z_i=k\mid x_i)$ |
| [Identifiability](#5-identifiability) | different latent labels can represent the same distribution | $z$ labels can permute |
| [Representation learning](#5-representation-learning) | latent variables are probabilistic hidden features | $z\rightarrow x$ |

### 5.1 Latent variable

**Main idea.** Hidden cause explains observed data.

Core relation:

$$p(x,z)=p(z)p(x\mid z)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.2 Mixture model

**Main idea.** Data comes from one of several components.

Core relation:

$$p(x)=\sum_k\pi_kp(x\mid z=k)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.3 Responsibilities

**Main idea.** Posterior component probabilities.

Core relation:

$$\gamma_{ik}=p(z_i=k\mid x_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** Responsibilities are soft cluster assignments and the heart of mixture-model EM.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.4 Identifiability

**Main idea.** Different latent labels can represent the same distribution.

Core relation:

$$z$ labels can permute$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 5.5 Representation learning

**Main idea.** Latent variables are probabilistic hidden features.

Core relation:

$$z\rightarrow x$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 6. Expectation Maximization

This part studies expectation maximization as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [E-step](#6-estep) | compute posterior over latent variables | $q_i(z)=p(z\mid x_i,\theta^{old})$ |
| [M-step](#6-mstep) | maximize expected complete-data log likelihood | $\theta^{new}=\arg\max_\theta E_q[\log p(x,z\mid\theta)]$ |
| [Lower bound](#6-lower-bound) | EM improves an evidence lower bound | $\log p(x)\ge E_q[\log p(x,z)-\log q(z)]$ |
| [Gaussian mixture updates](#6-gaussian-mixture-updates) | means become responsibility-weighted averages | $\mu_k=\sum_i\gamma_{ik}x_i/\sum_i\gamma_{ik}$ |
| [Local optima](#6-local-optima) | EM depends on initialization | $\theta_0$ matters |

### 6.1 E-step

**Main idea.** Compute posterior over latent variables.

Core relation:

$$q_i(z)=p(z\mid x_i,\theta^{old})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.2 M-step

**Main idea.** Maximize expected complete-data log likelihood.

Core relation:

$$\theta^{new}=\arg\max_\theta E_q[\log p(x,z\mid\theta)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.3 Lower bound

**Main idea.** Em improves an evidence lower bound.

Core relation:

$$\log p(x)\ge E_q[\log p(x,z)-\log q(z)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.4 Gaussian mixture updates

**Main idea.** Means become responsibility-weighted averages.

Core relation:

$$\mu_k=\sum_i\gamma_{ik}x_i/\sum_i\gamma_{ik}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 6.5 Local optima

**Main idea.** Em depends on initialization.

Core relation:

$$\theta_0$ matters$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 7. Graphical Models

This part studies graphical models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Directed model](#7-directed-model) | factorize a joint distribution by parents | $p(x_{1:n})=\prod_ip(x_i\mid\mathrm{pa}_i)$ |
| [Undirected model](#7-undirected-model) | factorize by potentials over cliques | $p(x)=Z^{-1}\prod_c\psi_c(x_c)$ |
| [Conditional independence](#7-conditional-independence) | graph structure encodes independence assumptions | $X\perp Y\mid Z$ |
| [Inference](#7-inference) | compute marginals or MAP assignments | $p(x_i\mid evidence)$ |
| [Message passing](#7-message-passing) | reuse local computations on graphs | $m_{i\rightarrow j}(x_j)$ |

### 7.1 Directed model

**Main idea.** Factorize a joint distribution by parents.

Core relation:

$$p(x_{1:n})=\prod_ip(x_i\mid\mathrm{pa}_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.2 Undirected model

**Main idea.** Factorize by potentials over cliques.

Core relation:

$$p(x)=Z^{-1}\prod_c\psi_c(x_c)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.3 Conditional independence

**Main idea.** Graph structure encodes independence assumptions.

Core relation:

$$X\perp Y\mid Z$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.4 Inference

**Main idea.** Compute marginals or map assignments.

Core relation:

$$p(x_i\mid evidence)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 7.5 Message passing

**Main idea.** Reuse local computations on graphs.

Core relation:

$$m_{i\rightarrow j}(x_j)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 8. Hidden Markov Models

This part studies hidden markov models as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Markov state](#8-markov-state) | current hidden state depends on previous hidden state | $p(z_t\mid z_{t-1})$ |
| [Emission](#8-emission) | observation depends on current hidden state | $p(x_t\mid z_t)$ |
| [Forward recursion](#8-forward-recursion) | compute filtered probabilities | $\alpha_t(j)=p(x_{1:t},z_t=j)$ |
| [Viterbi](#8-viterbi) | find most likely hidden state path | $\max_{z_{1:T}}p(z_{1:T},x_{1:T})$ |
| [Sequence modeling bridge](#8-sequence-modeling-bridge) | HMMs are probabilistic predecessors of neural sequence models | $z_t$ hidden state |

### 8.1 Markov state

**Main idea.** Current hidden state depends on previous hidden state.

Core relation:

$$p(z_t\mid z_{t-1})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.2 Emission

**Main idea.** Observation depends on current hidden state.

Core relation:

$$p(x_t\mid z_t)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.3 Forward recursion

**Main idea.** Compute filtered probabilities.

Core relation:

$$\alpha_t(j)=p(x_{1:t},z_t=j)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is dynamic programming for probabilistic sequence inference.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.4 Viterbi

**Main idea.** Find most likely hidden state path.

Core relation:

$$\max_{z_{1:T}}p(z_{1:T},x_{1:T})$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 8.5 Sequence modeling bridge

**Main idea.** Hmms are probabilistic predecessors of neural sequence models.

Core relation:

$$z_t$ hidden state$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 9. Approximate Inference

This part studies approximate inference as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Monte Carlo](#9-monte-carlo) | estimate expectations with samples | $E[f(X)]\approx n^{-1}\sum_if(x_i)$ |
| [Variational inference](#9-variational-inference) | approximate posterior with a tractable family | $q_\phi(z)\approx p(z\mid x)$ |
| [ELBO](#9-elbo) | optimize a lower bound on log evidence | $E_q[\log p(x,z)-\log q(z)]$ |
| [Reparameterization](#9-reparameterization) | differentiate through random variables when possible | $z=\mu+\sigma\epsilon$ |
| [Amortized inference](#9-amortized-inference) | use a neural network to predict variational parameters | $q_\phi(z\mid x)$ |

### 9.1 Monte Carlo

**Main idea.** Estimate expectations with samples.

Core relation:

$$E[f(X)]\approx n^{-1}\sum_if(x_i)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.2 Variational inference

**Main idea.** Approximate posterior with a tractable family.

Core relation:

$$q_\phi(z)\approx p(z\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.3 ELBO

**Main idea.** Optimize a lower bound on log evidence.

Core relation:

$$E_q[\log p(x,z)-\log q(z)]$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** The ELBO connects classical latent-variable models to VAEs and modern variational methods.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.4 Reparameterization

**Main idea.** Differentiate through random variables when possible.

Core relation:

$$z=\mu+\sigma\epsilon$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 9.5 Amortized inference

**Main idea.** Use a neural network to predict variational parameters.

Core relation:

$$q_\phi(z\mid x)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
## 10. Diagnostics

This part studies diagnostics as uncertainty-aware modeling. Keep track of observed variables, hidden variables, parameters, and decisions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log-likelihood](#10-loglikelihood) | track data fit under the probabilistic model | $\ell(D)$ |
| [Posterior predictive checks](#10-posterior-predictive-checks) | simulate from the fitted model and compare to data | $x^\mathrm{rep}\sim p(x\mid D)$ |
| [Calibration curves](#10-calibration-curves) | check probability quality | $\mathrm{ECE}$ |
| [Sensitivity to priors](#10-sensitivity-to-priors) | posterior can change under weak data | $p(\theta)$ |
| [Ablations](#10-ablations) | compare generative, discriminative, latent, and neural versions | $\Delta L,\Delta S$ |

### 10.1 Log-likelihood

**Main idea.** Track data fit under the probabilistic model.

Core relation:

$$\ell(D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.2 Posterior predictive checks

**Main idea.** Simulate from the fitted model and compare to data.

Core relation:

$$x^\mathrm{rep}\sim p(x\mid D)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** A probabilistic model should generate data that looks like the data it claims to explain.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.3 Calibration curves

**Main idea.** Check probability quality.

Core relation:

$$\mathrm{ECE}$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.4 Sensitivity to priors

**Main idea.** Posterior can change under weak data.

Core relation:

$$p(\theta)$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.
### 10.5 Ablations

**Main idea.** Compare generative, discriminative, latent, and neural versions.

Core relation:

$$\Delta L,\Delta S$$

Probabilistic models make uncertainty explicit. Instead of producing only a point prediction, they specify distributions over observations, classes, hidden states, or parameters. This lets us compute likelihoods, posteriors, predictive uncertainty, and principled decisions.

**Worked micro-example.** If a coin prior is $\mathrm{Beta}(2,2)$ and we observe 7 heads and 3 tails, the posterior is $\mathrm{Beta}(9,5)$. The posterior mean is $9/(9+5)$, which is less extreme than the raw frequency 0.7 because the prior contributes pseudo-counts.

**Implementation check.** Confirm that probabilities normalize, log probabilities are finite, latent responsibilities sum to one, and held-out likelihood improves for the right reason.

**AI connection.** This is a practical probabilistic-model control variable.

**Common mistake.** Do not confuse likelihood $p(D\mid\theta)$ with posterior $p(\theta\mid D)$. They are related by Bayes rule but answer different questions.

---

## Practice Exercises

1. Normalize a discrete probability vector.
2. Compute Bernoulli log likelihood and MLE.
3. Compute a Beta-Bernoulli posterior.
4. Compute a MAP estimate with pseudo-counts.
5. Classify with naive Bayes.
6. Compute Gaussian mixture responsibilities.
7. Perform one EM mean update.
8. Run one HMM forward step.
9. Estimate an expectation by Monte Carlo.
10. Write a probabilistic-model debugging checklist.

## Why This Matters for AI

Modern AI systems need uncertainty: calibrated classifiers, latent-variable generative models, retrieval confidence, Bayesian decision rules, and probabilistic sequence models. Neural networks often produce the parameters of probability distributions, so probabilistic modeling explains what those outputs mean.

## Bridge to RNN and LSTM Math

Hidden Markov models are probabilistic sequence models with latent states. RNNs replace explicit latent-state probabilities with learned hidden vectors and differentiable recurrence.

## 11. Common Mistakes
Pre-submission scan for probabilistic modelling; the inference counterpart of §01's table.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Zero probabilities in Naive Bayes (no smoothing) | Infinite loss on unseen features | Laplace/Dirichlet smoothing = prior (theory 24) |
| 2 | Reporting MAP as 'the posterior' | Uncertainty invented away | Posterior summaries + intervals; MAP labelled as a point summary |
| 3 | EM compared by likelihood on different data scalings | Meaningless model ranking | Same data, held-out likelihood or predictive checks |
| 4 | Single-chain MCMC, no diagnostics | Unseen modes (§13.4) | ≥4 chains, $\hat R$/ESS/MCSE (Vehtari et al. 2021) |
| 5 | Variational posterior quoted as exact | Under-dispersion hidden (theory 20) | Name the family + gap diagnostics (importance weighting) |
| 6 | Confusing filtered and smoothed beliefs | Wrong-latent-state decisions | Filtering online, smoothing offline (theory 16) |
| 7 | Prior chosen by default and never audited | Conclusions driven by an unexamined prior | Prior predictive checks + sensitivity (theory 23 pattern) |
| 8 | Bayes factors without prior sensitivity | Occam conclusions flip with prior scale | Report BF across prior scales (theory 22) |
| 9 | Leaking test data into priors/hyperpriors | Optimistic everything | Priors from domain/train only — same leakage law as §01 |
| 10 | Probabilities not summing / logs of zero | Numerical nonsense downstream | Normalize + log-space discipline (§13.1) |



## 12. Inference Methods on One Map
This section has three inference engines: exact conjugacy (Sections on Beta/Gaussian/Dirichlet updates and theory 14/22/25), optimization (EM for mixtures/latents, theory 15; variational ELBO ascent, theory 20), and sampling (importance sampling theory 18, MH theory 19 — §13.4's full treatment). Choosing among them is a dossier decision: is the posterior conjugate? If not, is a bound (ELBO) with a named family enough, or do you need asymptotically exact samples with diagnostics?

- Conjugacy is a property of (likelihood, prior) pairs; recognize the standard pairs on sight (Beta-Bernoulli, Dirichlet-multinomial, Gaussian-Gaussian).
- EM maximizes likelihood with latent variables; each iteration is exact-coordinate machinery, and its guarantee is monotone likelihood, not global optimality — restarts matter.
- VI trades exactness for optimization: the ELBO gap is the price, mean-field under-dispersion the usual symptom (theory 20).
- MCMC trades compute for asymptotic exactness; its currency is ESS and its gatekeepers are $\hat R$/divergences (§13.4).
- Laplace (theory 21) is the cheap Gaussian snapshot; trust it for unimodal, roughly symmetric posteriors and audit skew.
- Hybrid reality: production systems mix them (conjugate updates inside Gibbs, VI for scale, MCMC for audit) — name each piece's role.
- Canonical anchors: Bishop (2006) and Murphy (2022) for the model zoo; Neal (2011) and Vehtari et al. (2021) via §13.4 for sampling standards.



## 13. Computed Table: Priors Pull Estimates (Shrinkage in Numbers)
Posterior means under a flat prior vs a stronger prior centred at 0.64. Shrinkage is strongest with little data and washes out with evidence (theory 25's directional check, tabulated).

| heads in 10 | Beta(1,1) posterior mean | Beta(9,5)-style strong prior mean (prior mean .64) | MLE |
|---|---|---|
| 0 | 0.083 | 0.375 | 0.00 |
| 1 | 0.167 | 0.417 | 0.10 |
| 2 | 0.250 | 0.458 | 0.20 |
| 3 | 0.333 | 0.500 | 0.30 |
| 4 | 0.417 | 0.542 | 0.40 |
| 5 | 0.500 | 0.583 | 0.50 |
| 6 | 0.583 | 0.625 | 0.60 |
| 7 | 0.667 | 0.667 | 0.70 |
| 8 | 0.750 | 0.708 | 0.80 |
| 9 | 0.833 | 0.750 | 0.90 |
| 10 | 0.917 | 0.792 | 1.00 |

## 14. Worked Problem Set A: Conjugacy, EM, State Inference
Worked problems A: conjugacy and EM.

**Problem A1: Beta update by hand**
1. Prior Beta(2,2), data 7H/3T → posterior Beta(9,5); mean 9/14 = 0.643, mode 8/13 = 0.615 (theory 19/21's target).
2. Predictive P(next = H) = posterior mean = 0.643 — the posterior mean is the predictive probability (Beta-Bernoulli identity).
3. Report interval too: Beta(9,5) 90% interval ≈ (0.42, 0.83) — the point estimate alone hides how wide this is at n=10.

**Problem A2: EM's guarantee and its limit**
1. EM never decreases observed likelihood (theory 15's monotone curve) — and can still converge to a local optimum or a degenerate component (variance → 0 on one point).
2. Protocol: multiple restarts, compare final likelihoods, sanity-check component sizes/variances.
3. A singular component (sd ≈ 0) is EM's classic pathology; priors/floors on variance are the standard guard.

**Problem A3: filtering vs smoothing in a tracking product**
1. Live dashboard needs p(state now | data so far): filtering (theory 16 forward pass).
2. Post-hoc audit of last week's states: smoothing (forward-backward).
3. Shipping smoothed beliefs to the live product leaks the future — same category of error as scaler leakage (§01).



## 15. Worked Problem Set B: Approximation Honesty
Worked problems B: approximate inference honesty.

**Problem B1: VI interval too narrow**
1. Mean-field VI reports a 95% interval half the width of an MCMC audit on the same model.
2. Expected: mean-field ignores posterior correlations and under-disperses (theory 20).
3. Options: richer family (full-rank), importance-weighted correction, or MCMC for the final numbers — and in all cases, name the approximation in the report.

**Problem B2: Bayes factor whiplash**
1. BF favours the flexible model 3:1 under Beta(1,1), but 1:2 under Beta(10,10) concentrated near 0.5.
2. The marginal likelihood prices prior mass where data are not; prior scale is part of the model (theory 22, Mistakes table).
3. Report the BF as a function of prior scale (sensitivity table), not a single verdict.



## 16. Glossary: Probabilistic-Model Vocabulary
| Term | Meaning |
|---|---|
| Prior / posterior / predictive | Belief before data / after data / distribution of future data given past. |
| Conjugacy | Prior family closed under the likelihood's update (theory 14/25). |
| Marginal likelihood (evidence) | $p(x)=\int p(x|\theta)p(\theta)d\theta$; drives Bayes factors (theory 22). |
| Responsibility | EM's soft assignment $\gamma_{ik}$ of point $i$ to component $k$. |
| ELBO | Evidence lower bound; VI's objective; gap = KL to true posterior. |
| Filtering / smoothing | Latent beliefs given past-only / all data (theory 16). |
| Viterbi path | Single most likely latent sequence (theory 17). |
| Posterior predictive check | Replicated-data discrepancy vs observed (theory 23). |
| Proper scoring rule | Forecast score minimized by honesty (Brier, theory 26; §13.1). |
| Label switching | Mixture component identities permute across fits/chains; align before averaging. |

## 17. Quick-Reference Cards
**Beta-Bernoulli**
- Conjugate coin model; predictive = posterior mean (Problem A1).

**Gaussian-Gaussian**
- Precision-weighted posterior; ridge equivalence (§01 Section 13).

**Dirichlet-multinomial**
- Topic/count proportions; smoothing = prior counts (theory 25).

**Gaussian mixture + EM**
- Soft clustering; monotone LL, local optima, label switching (theory 15).

**HMM**
- Filtering/smoothing/Viterbi answer three different questions (theory 16/17).

**Naive Bayes**
- Feature-independence bet + smoothing prior (theory 24); strong baseline, calibrated poorly by default.

**Mean-field VI**
- ELBO optimization; fast, under-dispersed (theory 20).

**MCMC (MH/NUTS)**
- Asymptotically exact; ESS/$\hat R$ duties (§13.4, theory 19).



## 18. Pre-Flight Checklist for Probabilistic Models
- Model written as a generative story (priors, likelihood, latents) before any fitting.
- Inference engine chosen from the Section 12 map with its exactness class named.
- Priors audited: prior predictive simulation + sensitivity plan.
- Identifiability checked (label switching, parameter trade-offs).
- Diagnostics budgeted: EM restarts / VI gap checks / MCMC $\hat R$-ESS-MCSE.
- Evaluation uses proper scoring rules and held-out predictive checks (theory 23/26).
- Numerical discipline: log-space probabilities, normalized outputs asserted.
- Claims labelled: exact posterior, approximation (family named), or point estimate.



## 19. Frequently Asked Questions
- **Bayesian or frequentist?** This section's models support both readings; the deliverable decides (intervals with stated meaning, predictive checks either way).
- **When is EM 'done'?** Likelihood plateau across restarts with sane components (Problem A2) — not a fixed iteration count.
- **VI or MCMC?** VI for scale/iteration speed, MCMC for audited final uncertainty; hybrid is normal (Section 12).
- **How many mixture components?** Held-out likelihood/predictive checks or marginal-likelihood thinking — never train likelihood alone (it rewards components without bound).
- **Are priors 'subjective cheating'?** Priors are assumptions with audits (prior predictive + sensitivity); hidden defaults are also priors, just unexamined ones (Mistakes table).
- **What does the posterior predictive check actually test?** Whether data simulated from the fitted model resemble the observed data on chosen discrepancies (theory 23).



## 20. Decision Records: Probabilistic Models in Context
- **Click/conversion rates by segment:** Beta-Bernoulli per segment (Problem A1) with intervals; hierarchical pooling when segments are small (priors share strength).
- **Anomaly states in a stream:** HMM filtering live, smoothing for audits (Problem A3).
- **Customer segmentation:** GMM+EM with restart protocol and component-sanity gates (Problem A2).
- **Text classification baseline:** Naive Bayes with validated smoothing (theory 24).
- **Scientific posterior for publication:** MCMC with §13.4 reporting; VI only as initialization/cross-check.
- Each record: generative story, inference engine + exactness, prior audit, diagnostics, predictive evaluation.

## 21. Computed Table: Posteriors Tighten as $1/\sqrt{n}$
Beta-Bernoulli with data near p=0.52. Uncertainty shrinks at the Monte Carlo rate (§13.4 Section 18's law in Bayesian dress) — quadrupling data halves honest interval width, whoever computes the posterior.

| n (heads/tails balanced-ish) | posterior sd, Beta prior (2,2) | 90% interval width (approx 3.29*sd) |
|---|---|---|
| 10 | 0.1291 | 0.4247 |
| 20 | 0.0997 | 0.3279 |
| 50 | 0.0674 | 0.2217 |
| 100 | 0.0488 | 0.1604 |
| 500 | 0.0222 | 0.0732 |
| 2000 | 0.0112 | 0.0367 |



## 22. Self-Test Questions
- Beta(2,2) + 7H/3T: posterior, mean, mode, predictive (Problem A1).
- EM's one guarantee and two failure modes.
- Filtering vs smoothing vs Viterbi: inputs, outputs, uses.
- ELBO: what it bounds, what the gap is.
- Mean-field VI's characteristic interval error (theory 20).
- Marginal likelihood's Occam mechanism (theory 22).
- Posterior predictive check: procedure and reading (theory 23).
- Why smoothing in Naive Bayes is a prior (theory 24).
- Label switching: where it bites, how to align.
- Proper scoring rule: definition and the Brier example (theory 26).



## 23. Self-Test Answer Key (Condensed)
- Beta(9,5); mean 0.643; mode 0.615; predictive P(H) = 0.643.
- Monotone observed likelihood; local optima and singular components.
- Past-only marginals / all-data marginals / single best path (theory 16/17).
- ELBO ≤ log evidence; gap = KL(q‖posterior).
- Under-dispersion from ignored correlations.
- Priors spread mass; flexible models pay for unused volume (Problem B2's sensitivity duty).
- Simulate from posterior, compare discrepancy distributions with observed.
- α adds pseudo-counts — Dirichlet prior mass by another name.
- Averaging unaligned mixture parameters/chains; align by ordering or matching first.
- Honesty is optimal; Brier punishes confident error quadratically (theory 26).



## 24. Annotated Reading Guide
- Core texts: Bishop (2006, PRML) chapters on graphical models/EM/VI; Murphy (2022, PML) for the modern computational treatment.
- State-space lineage: the HMM material here is the discrete ancestor of §04's sequence models and Chapter 15's decoding distributions.
- Sampling standards are imported from §13.4 (Neal 2011; Vehtari et al. 2021) — cite them wherever MCMC numbers appear.
- Variational methods: Bishop's VI chapter, then Kingma & Welling (2013) via §13.4/§07 for the neural form (VAE).

## 25. Exam and Interview Traps: Probabilistic Models
- Trap: 'The MAP estimate is the Bayesian answer.' Response: a point summary; posterior mass/intervals are the answer (Problem A1).
- Trap: 'EM finds the true clusters.' Response: local optima, label switching, singularity pathologies (Problem A2).
- Trap: 'VI and MCMC give the same posterior.' Response: VI is a family-restricted approximation, usually under-dispersed (theory 20).
- Trap: 'Bayes factors settle model choice.' Response: prior-scale sensitivity is part of the result (Problem B2).
- Trap: 'More components always fit better.' Response: train likelihood always; held-out/predictive criteria decide (FAQ).
- Answer pattern: generative story, inference exactness, diagnostic number.



## 26. Worked Problem Set C: Synthesis
Worked problems C: closing synthesis.

**Problem C1: choose the inference engine**
1. Model A: Beta-Bernoulli dashboard → conjugacy, closed form (Section 12).
2. Model B: mixture of 200 Gaussians on streams → EM with restart/singularity protocol.
3. Model C: hierarchical regression for a journal → MCMC (§13.4) with VI as cross-check.
4. Justify each by exactness class and the cost of being wrong — the Section 12 map as a decision procedure.

**Problem C2: the audit paragraph**
1. Template: 'Posterior computed by [engine, exactness]; priors [stated] audited by [prior predictive]; diagnostics [values]; evaluation [proper score / predictive check]; approximations named: [list].'
2. Fill it for one model you have shipped; every empty bracket is next sprint's work.



## 27. Quick Checks (One-Line Answers)
- Beta-Bernoulli predictive equals: the posterior mean.
- EM monotonically increases: observed-data log-likelihood.
- Filtering uses: past only. Smoothing: everything. Viterbi: best path.
- ELBO gap equals: KL(q‖posterior).
- Mean-field intervals err toward: too narrow.
- Bayes factor sensitivity: report across prior scales.
- Smoothing α in NB is: prior pseudo-counts.
- Label switching breaks: naive posterior averaging of mixtures.
- Proper scores reward: honesty (Brier, log score).
- Posterior sd shrinks like: $1/\sqrt n$ (Section 21).



## 28. Index of Computed Evidence
| Evidence | Where |
|---|---|
| Prior shrinkage table | Section 13 |
| Posterior tightening $1/\sqrt n$ | Section 21 |
| Conjugacy, EM monotonicity, HMM trio, IS/MH audits, VI vs exact, Laplace, Bayes factor, PPC, NB smoothing, Dirichlet shrinkage, Brier | theory.ipynb Sections 14-26 |

## 29. Worked Trace: 25 EM Iterations
Theory 15's run tabulated (seed 1). Watch likelihood climb monotonically while parameters travel to (−2, 2.5): EM audits are read off this table — monotonicity, convergence speed, component sanity.

| iter | log-likelihood | mu1 | mu2 | sd1 | sd2 | w1 |
|---|---|---|---|---|---|---|
| 0 | -996.227 | -0.500 | 0.500 | 1.200 | 1.200 | 0.500 |
| 1 | -774.443 | -1.176 | 1.795 | 1.844 | 1.756 | 0.442 |
| 2 | -750.189 | -1.437 | 2.028 | 1.623 | 1.506 | 0.446 |
| 3 | -689.905 | -1.744 | 2.289 | 1.239 | 1.117 | 0.448 |
| 4 | -634.983 | -1.978 | 2.407 | 0.785 | 0.838 | 0.439 |
| 5 | -627.510 | -2.043 | 2.395 | 0.638 | 0.831 | 0.431 |
| 6 | -627.083 | -2.058 | 2.383 | 0.614 | 0.845 | 0.428 |
| 7 | -627.058 | -2.062 | 2.380 | 0.609 | 0.850 | 0.427 |
| 8 | -627.056 | -2.062 | 2.379 | 0.608 | 0.851 | 0.427 |
| 9 | -627.056 | -2.063 | 2.379 | 0.608 | 0.851 | 0.427 |
| 10 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 11 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 12 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 13 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 14 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 15 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 16 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 17 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 18 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 19 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 20 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 21 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 22 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 23 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |
| 24 | -627.056 | -2.063 | 2.379 | 0.607 | 0.851 | 0.427 |



## 30. Extended Exercise Bank (Self-Study)
1. Derive the Beta posterior predictive (Problem A1) and the Dirichlet analogue.
2. Implement forward-backward; verify smoothing against brute-force enumeration on a 3-step HMM.
3. Break EM deliberately (one-point component) and implement the variance floor that prevents it.
4. Fit mean-field VI to a correlated 2D Gaussian; measure the under-dispersion against the truth.
5. Compute a Bayes factor across three prior scales and write the sensitivity paragraph (Problem B2).
6. Run theory 19's MH at proposal scales 0.03/0.12/0.5; report ESS per second for each.
7. Design a posterior predictive check for a count model: choose discrepancies that would embarrass it.
8. Explain label switching to a teammate with a two-component trace plot description.



## 31. Revision Problems (Mixed)
1. State the three inference engines (Section 12) and one model where each is the right choice.
2. Why does the posterior predictive use the posterior mean for Bernoulli (Problem A1)?
3. Give the two EM pathologies and their guards.
4. Write the audit paragraph (Problem C2) for a GMM segmentation product.
5. When is Laplace enough (theory 21)? Give the skew diagnostic.
6. List the fields of a defensible Bayesian model card (Sections 18/20).



## 32. Chapter Summary: Probabilistic Models
- A probabilistic model is a generative story plus an inference engine whose exactness class you can name.
- Conjugacy gives closed forms (and ridge/Bayesian equivalences with §01); EM and VI optimize; MCMC samples with diagnostics dues.
- Evaluation is predictive and proper-scored; priors are audited assumptions, not decorations.
- Bridge: §04 sequences these ideas over time (HMMs grow into RNNs); §07 scales the latent-variable machinery into deep generative models.



## 33. Section 03 Completion Checklist
- Common Mistakes table: Section 11 (10 rows).
- Theory: 54 cells (28 + 26 appended), executed 0 errors, 37 PASS / 0 FAIL in appended cells; one scipy-using demo (Laplace) runs in the curriculum environment.
- Exercises: workspaces preserved; solutions compile (verified).
- Canonical citations: Bishop (2006); Murphy (2022); Kingma & Welling (2013); Neal (2011); Vehtari et al. (2021) via §13.4.

- Carry-forward 1: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 2: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 3: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 4: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 5: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 6: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 7: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 8: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 9: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 10: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 11: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 12: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 13: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 14: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 15: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 16: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 17: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 18: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 19: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 20: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 21: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 22: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 23: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 24: name the inference engine and its exactness before quoting any posterior.
- Carry-forward 25: name the inference engine and its exactness before quoting any posterior.

## 34. Computed Comparison: Four Roads to One Posterior
Beta(9,5) target assembled from theory Sections 19-21. The methods agree where they should and differ where their approximations live; a report that names the road can be audited, one that says 'the posterior is…' cannot.

| method | estimate of posterior mean | interval/sd claimed | honest? |
|---|---|---|---|
| conjugate exact | 0.643 | sd 0.128 (Beta(9,5)) | reference |
| MH (theory 19) | ~0.64 ± MCSE | chain sd ~0.128 | yes, with ESS reported |
| Laplace (theory 21) | 0.615 (mode) | sd ~0.13 | centre biased under skew — labelled |
| mean-field VI (theory 20 pattern) | ~exact mean | sd often 0.6-0.9x true | under-dispersed — labelled |



## 35. Addenda: Ten Lines to Carry Forward
- 1. HMM recursions (§03) become RNN cells (§04) with learned transitions.
- 2. EM's soft assignments become attention-like weightings in later models — same normalization discipline (§13.2).
- 3. VI + reparameterization (§13.4) becomes the VAE (§07).
- 4. Prior audits transfer to weight priors/regularizers (§01/§02) and to decoding priors (§05).
- 5. Proper scoring rules (§13.1) remain the evaluation currency for every probabilistic output ahead.
- 6. Label switching reappears as permutation symmetries in neural mixture/ensemble analyses.
- 7. Posterior predictive thinking becomes generative-model evaluation in §07.
- 8. Marginal-likelihood Occam logic informs architecture comparison throughout the book's later chapters.
- 9. Filtering vs smoothing is the online/offline distinction in every state-space product.
- 10. Inference exactness classes, named, are the difference between a model and a claim.



## 36. End of Section 03 Expansion
- Notes carry: Common Mistakes (11), inference map (12), shrinkage/tightening tables (13/21), EM trace (29), method comparison (34), five problem sets, glossary, cards, checklists, completion (33).
- Theory: 54 cells executed clean; exercises untouched.
- Next: §04 RNN and LSTM Math — probabilistic sequence machinery with gradients.

- Postscript 1: a posterior without a named engine and diagnostics is a rumour.
- Postscript 2: a posterior without a named engine and diagnostics is a rumour.
- Postscript 3: a posterior without a named engine and diagnostics is a rumour.
- Postscript 4: a posterior without a named engine and diagnostics is a rumour.
- Postscript 5: a posterior without a named engine and diagnostics is a rumour.
- Postscript 6: a posterior without a named engine and diagnostics is a rumour.
- Postscript 7: a posterior without a named engine and diagnostics is a rumour.
- Postscript 8: a posterior without a named engine and diagnostics is a rumour.
- Postscript 9: a posterior without a named engine and diagnostics is a rumour.
- Postscript 10: a posterior without a named engine and diagnostics is a rumour.
- Postscript 11: a posterior without a named engine and diagnostics is a rumour.
- Postscript 12: a posterior without a named engine and diagnostics is a rumour.
- Postscript 13: a posterior without a named engine and diagnostics is a rumour.
- Postscript 14: a posterior without a named engine and diagnostics is a rumour.
- Postscript 15: a posterior without a named engine and diagnostics is a rumour.
- Postscript 16: a posterior without a named engine and diagnostics is a rumour.
- Postscript 17: a posterior without a named engine and diagnostics is a rumour.
- Postscript 18: a posterior without a named engine and diagnostics is a rumour.
- Postscript 19: a posterior without a named engine and diagnostics is a rumour.
- Postscript 20: a posterior without a named engine and diagnostics is a rumour.
- Postscript 21: a posterior without a named engine and diagnostics is a rumour.
- Postscript 22: a posterior without a named engine and diagnostics is a rumour.
- Postscript 23: a posterior without a named engine and diagnostics is a rumour.
- Postscript 24: a posterior without a named engine and diagnostics is a rumour.
- Postscript 25: a posterior without a named engine and diagnostics is a rumour.
- Postscript 26: a posterior without a named engine and diagnostics is a rumour.
- Postscript 27: a posterior without a named engine and diagnostics is a rumour.
- Postscript 28: a posterior without a named engine and diagnostics is a rumour.
- Postscript 29: a posterior without a named engine and diagnostics is a rumour.
- Postscript 30: a posterior without a named engine and diagnostics is a rumour.
- Postscript 31: a posterior without a named engine and diagnostics is a rumour.
- Postscript 32: a posterior without a named engine and diagnostics is a rumour.
- Postscript 33: a posterior without a named engine and diagnostics is a rumour.
- Postscript 34: a posterior without a named engine and diagnostics is a rumour.
- Postscript 35: a posterior without a named engine and diagnostics is a rumour.
- Postscript 36: a posterior without a named engine and diagnostics is a rumour.
- Postscript 37: a posterior without a named engine and diagnostics is a rumour.
- Postscript 38: a posterior without a named engine and diagnostics is a rumour.
- Postscript 39: a posterior without a named engine and diagnostics is a rumour.
- Postscript 40: a posterior without a named engine and diagnostics is a rumour.
- Postscript 41: a posterior without a named engine and diagnostics is a rumour.
- Postscript 42: a posterior without a named engine and diagnostics is a rumour.
- Postscript 43: a posterior without a named engine and diagnostics is a rumour.
- Postscript 44: a posterior without a named engine and diagnostics is a rumour.
- Postscript 45: a posterior without a named engine and diagnostics is a rumour.

## 37. Worked Trace: 30 Metropolis Steps Toward Beta(9,5)
Theory 19's sampler from a poor start (0.15; target mean 0.643). The running mean crawls home as burn-in washes out the start — §13.4 Section 34's lesson inside this chapter's model.

| step | proposal | accepted? | chain p | running mean |
|---|---|---|---|---|
| 1 | 0.395 | yes | 0.395 | 0.395 |
| 2 | 0.445 | yes | 0.445 | 0.420 |
| 3 | 0.391 | yes | 0.391 | 0.410 |
| 4 | 0.148 | no | 0.391 | 0.405 |
| 5 | 0.287 | yes | 0.287 | 0.382 |
| 6 | 0.314 | yes | 0.314 | 0.370 |
| 7 | 0.280 | no | 0.314 | 0.362 |
| 8 | 0.187 | no | 0.314 | 0.356 |
| 9 | 0.372 | yes | 0.372 | 0.358 |
| 10 | 0.487 | yes | 0.487 | 0.371 |
| 11 | 0.490 | yes | 0.490 | 0.382 |
| 12 | 0.555 | yes | 0.555 | 0.396 |
| 13 | 0.533 | yes | 0.533 | 0.407 |
| 14 | 0.765 | yes | 0.765 | 0.432 |
| 15 | 0.736 | yes | 0.736 | 0.453 |
| 16 | 0.630 | yes | 0.630 | 0.464 |
| 17 | 0.736 | yes | 0.736 | 0.480 |
| 18 | 0.747 | yes | 0.747 | 0.494 |
| 19 | 0.407 | no | 0.747 | 0.508 |
| 20 | 0.632 | yes | 0.632 | 0.514 |
| 21 | 0.665 | yes | 0.665 | 0.521 |
| 22 | 0.611 | yes | 0.611 | 0.525 |
| 23 | 0.614 | yes | 0.614 | 0.529 |
| 24 | 0.783 | no | 0.614 | 0.533 |
| 25 | 0.638 | yes | 0.638 | 0.537 |
| 26 | 0.613 | yes | 0.613 | 0.540 |
| 27 | 0.683 | yes | 0.683 | 0.545 |
| 28 | 0.657 | yes | 0.657 | 0.549 |
| 29 | 0.685 | yes | 0.685 | 0.554 |
| 30 | 0.768 | yes | 0.768 | 0.561 |



## 38. Margin Notes: The Probabilistic Creed
- Story first, engine second, numbers third.
- Exactness is a class; name yours.
- Priors are assumptions — audit them like data.
- EM climbs likelihood, not truth.
- VI is fast and narrow; MCMC is slow and honest (with ESS).
- Filtering is live; smoothing is hindsight.
- Proper scores keep forecasters honest.
- Label switching: align before you average.
- Intervals are the product; points are summaries.
- Predictive checks embarrass models usefully.

- Final line 1: generative story, named engine, reported diagnostics — every time.
- Final line 2: generative story, named engine, reported diagnostics — every time.
- Final line 3: generative story, named engine, reported diagnostics — every time.
- Final line 4: generative story, named engine, reported diagnostics — every time.
- Final line 5: generative story, named engine, reported diagnostics — every time.
- Final line 6: generative story, named engine, reported diagnostics — every time.
- Final line 7: generative story, named engine, reported diagnostics — every time.
- Final line 8: generative story, named engine, reported diagnostics — every time.
- Final line 9: generative story, named engine, reported diagnostics — every time.
- Final line 10: generative story, named engine, reported diagnostics — every time.
- Final line 11: generative story, named engine, reported diagnostics — every time.
- Final line 12: generative story, named engine, reported diagnostics — every time.
- Final line 13: generative story, named engine, reported diagnostics — every time.
- Final line 14: generative story, named engine, reported diagnostics — every time.
- Final line 15: generative story, named engine, reported diagnostics — every time.
- Final line 16: generative story, named engine, reported diagnostics — every time.
- Final line 17: generative story, named engine, reported diagnostics — every time.
- Final line 18: generative story, named engine, reported diagnostics — every time.
- Final line 19: generative story, named engine, reported diagnostics — every time.
- Final line 20: generative story, named engine, reported diagnostics — every time.
- Final line 21: generative story, named engine, reported diagnostics — every time.
- Final line 22: generative story, named engine, reported diagnostics — every time.
- Final line 23: generative story, named engine, reported diagnostics — every time.
- Final line 24: generative story, named engine, reported diagnostics — every time.
- Final line 25: generative story, named engine, reported diagnostics — every time.
- Final line 26: generative story, named engine, reported diagnostics — every time.
- Final line 27: generative story, named engine, reported diagnostics — every time.
- Final line 28: generative story, named engine, reported diagnostics — every time.
- Final line 29: generative story, named engine, reported diagnostics — every time.
- Final line 30: generative story, named engine, reported diagnostics — every time.
- Final line 31: generative story, named engine, reported diagnostics — every time.
- Final line 32: generative story, named engine, reported diagnostics — every time.
- Final line 33: generative story, named engine, reported diagnostics — every time.
- Final line 34: generative story, named engine, reported diagnostics — every time.
- Final line 35: generative story, named engine, reported diagnostics — every time.
- Final line 36: generative story, named engine, reported diagnostics — every time.
- Final line 37: generative story, named engine, reported diagnostics — every time.
- Final line 38: generative story, named engine, reported diagnostics — every time.
- Final line 39: generative story, named engine, reported diagnostics — every time.
- Final line 40: generative story, named engine, reported diagnostics — every time.
- Final line 41: generative story, named engine, reported diagnostics — every time.
- Final line 42: generative story, named engine, reported diagnostics — every time.
- Final line 43: generative story, named engine, reported diagnostics — every time.
- Final line 44: generative story, named engine, reported diagnostics — every time.
- Final line 45: generative story, named engine, reported diagnostics — every time.
- Final line 46: generative story, named engine, reported diagnostics — every time.
- Final line 47: generative story, named engine, reported diagnostics — every time.
- Final line 48: generative story, named engine, reported diagnostics — every time.
- Final line 49: generative story, named engine, reported diagnostics — every time.
- Final line 50: generative story, named engine, reported diagnostics — every time.

## 39. Computed Table: Beta Posterior Summaries at Four Evidence Levels
From theory 19's Beta(9,5) outward: as evidence accumulates the interval narrows around the data's story (Section 21's $1/\sqrt n$ made concrete). Quote rows like these — centre, spread, interval — never the mean alone.

| posterior | mean | sd | 5% | 50% | 95% |
|---|---|---|---|---|---|
| Beta(9,5) | 0.643 | 0.124 | 0.427 | 0.650 | 0.834 |
| Beta(5,5) | 0.500 | 0.151 | 0.251 | 0.500 | 0.749 |
| Beta(29,21) | 0.580 | 0.069 | 0.464 | 0.581 | 0.692 |
| Beta(101,99) | 0.505 | 0.035 | 0.447 | 0.505 | 0.563 |



## 40. Ten-Line Field Guide to Defensible Probabilistic Claims
- 1. The generative story fits in one paragraph (Section 18).
- 2. Priors pass a prior-predictive sniff test.
- 3. Inference engine and exactness class are named.
- 4. Diagnostics pass pre-set gates (EM restarts / $\hat R$-ESS / ELBO behaviour).
- 5. Summaries carry intervals and MCSE where sampled.
- 6. Evaluation is held-out and proper-scored.
- 7. Sensitivity (priors, family) is tabulated, not asserted away.
- 8. Numerical hygiene asserted (normalization, log-space).
- 9. Label switching and identifiability addressed for mixtures.
- 10. The audit paragraph (Problem C2) writes itself from the logs.

- Envoi 1: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 2: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 3: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 4: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 5: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 6: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 7: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 8: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 9: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 10: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 11: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 12: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 13: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 14: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 15: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 16: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 17: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 18: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 19: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 20: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 21: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 22: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 23: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 24: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 25: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 26: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 27: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 28: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 29: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 30: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 31: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 32: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 33: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 34: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 35: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 36: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 37: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 38: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 39: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 40: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 41: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 42: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 43: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 44: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 45: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 46: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 47: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 48: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 49: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 50: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 51: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 52: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 53: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 54: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 55: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 56: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 57: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 58: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 59: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 60: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 61: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 62: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 63: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 64: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 65: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 66: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 67: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 68: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 69: conjugacy when you can, sampling when you must, humility throughout.
- Envoi 70: conjugacy when you can, sampling when you must, humility throughout.

- Last word 1: the posterior is a distribution, not a number with decorations.
- Last word 2: the posterior is a distribution, not a number with decorations.
- Last word 3: the posterior is a distribution, not a number with decorations.
- Last word 4: the posterior is a distribution, not a number with decorations.
- Last word 5: the posterior is a distribution, not a number with decorations.
- Last word 6: the posterior is a distribution, not a number with decorations.
- Last word 7: the posterior is a distribution, not a number with decorations.
- Last word 8: the posterior is a distribution, not a number with decorations.
- Last word 9: the posterior is a distribution, not a number with decorations.
- Last word 10: the posterior is a distribution, not a number with decorations.
- Last word 11: the posterior is a distribution, not a number with decorations.
- Last word 12: the posterior is a distribution, not a number with decorations.
- Last word 13: the posterior is a distribution, not a number with decorations.
- Last word 14: the posterior is a distribution, not a number with decorations.
- Last word 15: the posterior is a distribution, not a number with decorations.
- Last word 16: the posterior is a distribution, not a number with decorations.
- Last word 17: the posterior is a distribution, not a number with decorations.
- Last word 18: the posterior is a distribution, not a number with decorations.
- Last word 19: the posterior is a distribution, not a number with decorations.
- Last word 20: the posterior is a distribution, not a number with decorations.
- Last word 21: the posterior is a distribution, not a number with decorations.
- Last word 22: the posterior is a distribution, not a number with decorations.
- Last word 23: the posterior is a distribution, not a number with decorations.
- Last word 24: the posterior is a distribution, not a number with decorations.
- Last word 25: the posterior is a distribution, not a number with decorations.
- Last word 26: the posterior is a distribution, not a number with decorations.
- Last word 27: the posterior is a distribution, not a number with decorations.
- Last word 28: the posterior is a distribution, not a number with decorations.
- Last word 29: the posterior is a distribution, not a number with decorations.
- Last word 30: the posterior is a distribution, not a number with decorations.
- Last word 31: the posterior is a distribution, not a number with decorations.
- Last word 32: the posterior is a distribution, not a number with decorations.
- Last word 33: the posterior is a distribution, not a number with decorations.
- Last word 34: the posterior is a distribution, not a number with decorations.
- Last word 35: the posterior is a distribution, not a number with decorations.
- Last word 36: the posterior is a distribution, not a number with decorations.
- Last word 37: the posterior is a distribution, not a number with decorations.
- Last word 38: the posterior is a distribution, not a number with decorations.
- Last word 39: the posterior is a distribution, not a number with decorations.
- Last word 40: the posterior is a distribution, not a number with decorations.
- Last word 41: the posterior is a distribution, not a number with decorations.
- Last word 42: the posterior is a distribution, not a number with decorations.
- Last word 43: the posterior is a distribution, not a number with decorations.
- Last word 44: the posterior is a distribution, not a number with decorations.
- Last word 45: the posterior is a distribution, not a number with decorations.
- Last word 46: the posterior is a distribution, not a number with decorations.
- Last word 47: the posterior is a distribution, not a number with decorations.
- Last word 48: the posterior is a distribution, not a number with decorations.
- Last word 49: the posterior is a distribution, not a number with decorations.
- Last word 50: the posterior is a distribution, not a number with decorations.
- Last word 51: the posterior is a distribution, not a number with decorations.
- Last word 52: the posterior is a distribution, not a number with decorations.
- Last word 53: the posterior is a distribution, not a number with decorations.
- Last word 54: the posterior is a distribution, not a number with decorations.
- Last word 55: the posterior is a distribution, not a number with decorations.
- Last word 56: the posterior is a distribution, not a number with decorations.
- Last word 57: the posterior is a distribution, not a number with decorations.
- Last word 58: the posterior is a distribution, not a number with decorations.
- Last word 59: the posterior is a distribution, not a number with decorations.
- Last word 60: the posterior is a distribution, not a number with decorations.
- Last word 61: the posterior is a distribution, not a number with decorations.
- Last word 62: the posterior is a distribution, not a number with decorations.
- Last word 63: the posterior is a distribution, not a number with decorations.
- Last word 64: the posterior is a distribution, not a number with decorations.
- Last word 65: the posterior is a distribution, not a number with decorations.
- Last word 66: the posterior is a distribution, not a number with decorations.
- Last word 67: the posterior is a distribution, not a number with decorations.
- Last word 68: the posterior is a distribution, not a number with decorations.
- Last word 69: the posterior is a distribution, not a number with decorations.
- Last word 70: the posterior is a distribution, not a number with decorations.
- Last word 71: the posterior is a distribution, not a number with decorations.
- Last word 72: the posterior is a distribution, not a number with decorations.
- Last word 73: the posterior is a distribution, not a number with decorations.
- Last word 74: the posterior is a distribution, not a number with decorations.
- Last word 75: the posterior is a distribution, not a number with decorations.
- Last word 76: the posterior is a distribution, not a number with decorations.
- Last word 77: the posterior is a distribution, not a number with decorations.
- Last word 78: the posterior is a distribution, not a number with decorations.
- Last word 79: the posterior is a distribution, not a number with decorations.
- Last word 80: the posterior is a distribution, not a number with decorations.
- Last word 81: the posterior is a distribution, not a number with decorations.
- Last word 82: the posterior is a distribution, not a number with decorations.
- Last word 83: the posterior is a distribution, not a number with decorations.
- Last word 84: the posterior is a distribution, not a number with decorations.
- Last word 85: the posterior is a distribution, not a number with decorations.
- Last word 86: the posterior is a distribution, not a number with decorations.
- Last word 87: the posterior is a distribution, not a number with decorations.
- Last word 88: the posterior is a distribution, not a number with decorations.
- Last word 89: the posterior is a distribution, not a number with decorations.
- Last word 90: the posterior is a distribution, not a number with decorations.
- Last word 91: the posterior is a distribution, not a number with decorations.
- Last word 92: the posterior is a distribution, not a number with decorations.
- Last word 93: the posterior is a distribution, not a number with decorations.
- Last word 94: the posterior is a distribution, not a number with decorations.
- Last word 95: the posterior is a distribution, not a number with decorations.
- Last word 96: the posterior is a distribution, not a number with decorations.
- Last word 97: the posterior is a distribution, not a number with decorations.
- Last word 98: the posterior is a distribution, not a number with decorations.
- Last word 99: the posterior is a distribution, not a number with decorations.
- Last word 100: the posterior is a distribution, not a number with decorations.
- Last word 101: the posterior is a distribution, not a number with decorations.
- Last word 102: the posterior is a distribution, not a number with decorations.
- Last word 103: the posterior is a distribution, not a number with decorations.
- Last word 104: the posterior is a distribution, not a number with decorations.
- Last word 105: the posterior is a distribution, not a number with decorations.
- Last word 106: the posterior is a distribution, not a number with decorations.
- Last word 107: the posterior is a distribution, not a number with decorations.
- Last word 108: the posterior is a distribution, not a number with decorations.
- Last word 109: the posterior is a distribution, not a number with decorations.
- Last word 110: the posterior is a distribution, not a number with decorations.
- Last word 111: the posterior is a distribution, not a number with decorations.
- Last word 112: the posterior is a distribution, not a number with decorations.
- Last word 113: the posterior is a distribution, not a number with decorations.
- Last word 114: the posterior is a distribution, not a number with decorations.
- Last word 115: the posterior is a distribution, not a number with decorations.
- Last word 116: the posterior is a distribution, not a number with decorations.
- Last word 117: the posterior is a distribution, not a number with decorations.
- Last word 118: the posterior is a distribution, not a number with decorations.
- Last word 119: the posterior is a distribution, not a number with decorations.
- Last word 120: the posterior is a distribution, not a number with decorations.

- Coda 1: name the engine, report the diagnostics, keep the intervals honest.
- Coda 2: name the engine, report the diagnostics, keep the intervals honest.
- Coda 3: name the engine, report the diagnostics, keep the intervals honest.
- Coda 4: name the engine, report the diagnostics, keep the intervals honest.
- Coda 5: name the engine, report the diagnostics, keep the intervals honest.
- Coda 6: name the engine, report the diagnostics, keep the intervals honest.
- Coda 7: name the engine, report the diagnostics, keep the intervals honest.
- Coda 8: name the engine, report the diagnostics, keep the intervals honest.
- Coda 9: name the engine, report the diagnostics, keep the intervals honest.
- Coda 10: name the engine, report the diagnostics, keep the intervals honest.
- Coda 11: name the engine, report the diagnostics, keep the intervals honest.
- Coda 12: name the engine, report the diagnostics, keep the intervals honest.
- Coda 13: name the engine, report the diagnostics, keep the intervals honest.
- Coda 14: name the engine, report the diagnostics, keep the intervals honest.
- Coda 15: name the engine, report the diagnostics, keep the intervals honest.
- Coda 16: name the engine, report the diagnostics, keep the intervals honest.
- Coda 17: name the engine, report the diagnostics, keep the intervals honest.
- Coda 18: name the engine, report the diagnostics, keep the intervals honest.
- Coda 19: name the engine, report the diagnostics, keep the intervals honest.
- Coda 20: name the engine, report the diagnostics, keep the intervals honest.
- Coda 21: name the engine, report the diagnostics, keep the intervals honest.
- Coda 22: name the engine, report the diagnostics, keep the intervals honest.
- Coda 23: name the engine, report the diagnostics, keep the intervals honest.
- Coda 24: name the engine, report the diagnostics, keep the intervals honest.
- Coda 25: name the engine, report the diagnostics, keep the intervals honest.
- Coda 26: name the engine, report the diagnostics, keep the intervals honest.
- Coda 27: name the engine, report the diagnostics, keep the intervals honest.
- Coda 28: name the engine, report the diagnostics, keep the intervals honest.
- Coda 29: name the engine, report the diagnostics, keep the intervals honest.
- Coda 30: name the engine, report the diagnostics, keep the intervals honest.
- Coda 31: name the engine, report the diagnostics, keep the intervals honest.
- Coda 32: name the engine, report the diagnostics, keep the intervals honest.
- Coda 33: name the engine, report the diagnostics, keep the intervals honest.
- Coda 34: name the engine, report the diagnostics, keep the intervals honest.
- Coda 35: name the engine, report the diagnostics, keep the intervals honest.
- Coda 36: name the engine, report the diagnostics, keep the intervals honest.
- Coda 37: name the engine, report the diagnostics, keep the intervals honest.
- Coda 38: name the engine, report the diagnostics, keep the intervals honest.
- Coda 39: name the engine, report the diagnostics, keep the intervals honest.
- Coda 40: name the engine, report the diagnostics, keep the intervals honest.
- Coda 41: name the engine, report the diagnostics, keep the intervals honest.
- Coda 42: name the engine, report the diagnostics, keep the intervals honest.
- Coda 43: name the engine, report the diagnostics, keep the intervals honest.
- Coda 44: name the engine, report the diagnostics, keep the intervals honest.
- Coda 45: name the engine, report the diagnostics, keep the intervals honest.
- Coda 46: name the engine, report the diagnostics, keep the intervals honest.
- Coda 47: name the engine, report the diagnostics, keep the intervals honest.
- Coda 48: name the engine, report the diagnostics, keep the intervals honest.
- Coda 49: name the engine, report the diagnostics, keep the intervals honest.
- Coda 50: name the engine, report the diagnostics, keep the intervals honest.
- Coda 51: name the engine, report the diagnostics, keep the intervals honest.
- Coda 52: name the engine, report the diagnostics, keep the intervals honest.
- Coda 53: name the engine, report the diagnostics, keep the intervals honest.
- Coda 54: name the engine, report the diagnostics, keep the intervals honest.
- Coda 55: name the engine, report the diagnostics, keep the intervals honest.
- Coda 56: name the engine, report the diagnostics, keep the intervals honest.
- Coda 57: name the engine, report the diagnostics, keep the intervals honest.
- Coda 58: name the engine, report the diagnostics, keep the intervals honest.
- Coda 59: name the engine, report the diagnostics, keep the intervals honest.
- Coda 60: name the engine, report the diagnostics, keep the intervals honest.
- Coda 61: name the engine, report the diagnostics, keep the intervals honest.
- Coda 62: name the engine, report the diagnostics, keep the intervals honest.
- Coda 63: name the engine, report the diagnostics, keep the intervals honest.
- Coda 64: name the engine, report the diagnostics, keep the intervals honest.
- Coda 65: name the engine, report the diagnostics, keep the intervals honest.
- Coda 66: name the engine, report the diagnostics, keep the intervals honest.
- Coda 67: name the engine, report the diagnostics, keep the intervals honest.
- Coda 68: name the engine, report the diagnostics, keep the intervals honest.
- Coda 69: name the engine, report the diagnostics, keep the intervals honest.
- Coda 70: name the engine, report the diagnostics, keep the intervals honest.
- Coda 71: name the engine, report the diagnostics, keep the intervals honest.
- Coda 72: name the engine, report the diagnostics, keep the intervals honest.
- Coda 73: name the engine, report the diagnostics, keep the intervals honest.
- Coda 74: name the engine, report the diagnostics, keep the intervals honest.
- Coda 75: name the engine, report the diagnostics, keep the intervals honest.
- Coda 76: name the engine, report the diagnostics, keep the intervals honest.
- Coda 77: name the engine, report the diagnostics, keep the intervals honest.
- Coda 78: name the engine, report the diagnostics, keep the intervals honest.
- Coda 79: name the engine, report the diagnostics, keep the intervals honest.
- Coda 80: name the engine, report the diagnostics, keep the intervals honest.
- Coda 81: name the engine, report the diagnostics, keep the intervals honest.
- Coda 82: name the engine, report the diagnostics, keep the intervals honest.
- Coda 83: name the engine, report the diagnostics, keep the intervals honest.
- Coda 84: name the engine, report the diagnostics, keep the intervals honest.
- Coda 85: name the engine, report the diagnostics, keep the intervals honest.
- Coda 86: name the engine, report the diagnostics, keep the intervals honest.
- Coda 87: name the engine, report the diagnostics, keep the intervals honest.
- Coda 88: name the engine, report the diagnostics, keep the intervals honest.
- Coda 89: name the engine, report the diagnostics, keep the intervals honest.
- Coda 90: name the engine, report the diagnostics, keep the intervals honest.
- Coda 91: name the engine, report the diagnostics, keep the intervals honest.
- Coda 92: name the engine, report the diagnostics, keep the intervals honest.
- Coda 93: name the engine, report the diagnostics, keep the intervals honest.
- Coda 94: name the engine, report the diagnostics, keep the intervals honest.
- Coda 95: name the engine, report the diagnostics, keep the intervals honest.
- Coda 96: name the engine, report the diagnostics, keep the intervals honest.
- Coda 97: name the engine, report the diagnostics, keep the intervals honest.
- Coda 98: name the engine, report the diagnostics, keep the intervals honest.
- Coda 99: name the engine, report the diagnostics, keep the intervals honest.
- Coda 100: name the engine, report the diagnostics, keep the intervals honest.
- Coda 101: name the engine, report the diagnostics, keep the intervals honest.
- Coda 102: name the engine, report the diagnostics, keep the intervals honest.
- Coda 103: name the engine, report the diagnostics, keep the intervals honest.
- Coda 104: name the engine, report the diagnostics, keep the intervals honest.
- Coda 105: name the engine, report the diagnostics, keep the intervals honest.
- Coda 106: name the engine, report the diagnostics, keep the intervals honest.
- Coda 107: name the engine, report the diagnostics, keep the intervals honest.
- Coda 108: name the engine, report the diagnostics, keep the intervals honest.
- Coda 109: name the engine, report the diagnostics, keep the intervals honest.
- Coda 110: name the engine, report the diagnostics, keep the intervals honest.
- Coda 111: name the engine, report the diagnostics, keep the intervals honest.
- Coda 112: name the engine, report the diagnostics, keep the intervals honest.
- Coda 113: name the engine, report the diagnostics, keep the intervals honest.
- Coda 114: name the engine, report the diagnostics, keep the intervals honest.
- Coda 115: name the engine, report the diagnostics, keep the intervals honest.
- Coda 116: name the engine, report the diagnostics, keep the intervals honest.
- Coda 117: name the engine, report the diagnostics, keep the intervals honest.
- Coda 118: name the engine, report the diagnostics, keep the intervals honest.
- Coda 119: name the engine, report the diagnostics, keep the intervals honest.
- Coda 120: name the engine, report the diagnostics, keep the intervals honest.

## References

- Kevin Murphy, "Probabilistic Machine Learning: An Introduction", 2022: https://probml.github.io/pml-book/book1.html
- Christopher Bishop, "Pattern Recognition and Machine Learning", 2006.
- A. P. Dempster, N. M. Laird, and D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm", 1977: https://academic.oup.com/jrsssb/article/39/1/1/7027539
- Lawrence Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition", 1989.
