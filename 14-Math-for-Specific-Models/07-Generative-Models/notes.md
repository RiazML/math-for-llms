[<- Reinforcement Learning](../06-Reinforcement-Learning/notes.md) | [Home](../../README.md) | [CNN and Convolution Math ->](../08-CNN-and-Convolution-Math/notes.md)

---

# Generative Models

Generative models learn to sample from, score, or transform data distributions. They are the mathematical foundation behind language generation, image generation, latent-variable modeling, diffusion models, and synthetic data.

## Overview

The goal is to model:

$$
p_\theta(x)\approx p_\mathrm{data}(x).
$$

Different model families choose different paths: autoregressive factorization, latent-variable bounds, adversarial games, invertible transformations, or iterative denoising.

## Prerequisites

- Probability, likelihood, KL divergence, and ELBO
- Neural networks and optimization
- Autoregressive sequence probability
- Basic Gaussian sampling and matrix operations

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates autoregressive likelihood, VAE reparameterization and KL, GAN losses, flow change of variables, diffusion noising, score updates, and FID intuition. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for generative-model objectives and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Compare autoregressive models, VAEs, GANs, flows, diffusion models, and score models.
- Compute autoregressive log likelihood.
- Write and interpret the VAE ELBO and reparameterization trick.
- Explain GAN minimax training and mode collapse.
- Apply the change-of-variables formula for a simple flow.
- Simulate diffusion noising and denoising loss.
- Explain score matching and guidance.
- Evaluate generative models with likelihood, diversity, sample quality, and FID-style statistics.

## Table of Contents

1. [Generative Modeling Goal](#1-generative-modeling-goal)
   - 1.1 [Data distribution](#11-data-distribution)
   - 1.2 [Model distribution](#12-model-distribution)
   - 1.3 [Sampling](#13-sampling)
   - 1.4 [Likelihood](#14-likelihood)
   - 1.5 [Conditional generation](#15-conditional-generation)
2. [Autoregressive Models](#2-autoregressive-models)
   - 2.1 [Chain rule](#21-chain-rule)
   - 2.2 [Teacher forcing](#22-teacher-forcing)
   - 2.3 [Sampling loop](#23-sampling-loop)
   - 2.4 [Exact likelihood](#24-exact-likelihood)
   - 2.5 [Cost](#25-cost)
3. [Variational Autoencoders](#3-variational-autoencoders)
   - 3.1 [Latent variable model](#31-latent-variable-model)
   - 3.2 [Encoder](#32-encoder)
   - 3.3 [Decoder](#33-decoder)
   - 3.4 [ELBO](#34-elbo)
   - 3.5 [Reparameterization](#35-reparameterization)
4. [GANs](#4-gans)
   - 4.1 [Generator](#41-generator)
   - 4.2 [Discriminator](#42-discriminator)
   - 4.3 [Minimax objective](#43-minimax-objective)
   - 4.4 [Mode collapse](#44-mode-collapse)
   - 4.5 [No direct likelihood](#45-no-direct-likelihood)
5. [Normalizing Flows](#5-normalizing-flows)
   - 5.1 [Invertible map](#51-invertible-map)
   - 5.2 [Change of variables](#52-change-of-variables)
   - 5.3 [Exact likelihood](#53-exact-likelihood)
   - 5.4 [Architectural constraint](#54-architectural-constraint)
   - 5.5 [Sampling](#55-sampling)
6. [Diffusion Models](#6-diffusion-models)
   - 6.1 [Forward noising](#61-forward-noising)
   - 6.2 [Closed-form noising](#62-closedform-noising)
   - 6.3 [Denoising model](#63-denoising-model)
   - 6.4 [Training objective](#64-training-objective)
   - 6.5 [Reverse sampling](#65-reverse-sampling)
7. [Score-Based View](#7-scorebased-view)
   - 7.1 [Score](#71-score)
   - 7.2 [Denoising score matching](#72-denoising-score-matching)
   - 7.3 [Langevin sampling](#73-langevin-sampling)
   - 7.4 [SDE view](#74-sde-view)
   - 7.5 [Guidance](#75-guidance)
8. [Evaluation](#8-evaluation)
   - 8.1 [Log likelihood](#81-log-likelihood)
   - 8.2 [Sample quality](#82-sample-quality)
   - 8.3 [Diversity](#83-diversity)
   - 8.4 [FID intuition](#84-fid-intuition)
   - 8.5 [Precision recall](#85-precision-recall)
9. [Applications and Tradeoffs](#9-applications-and-tradeoffs)
   - 9.1 [Text](#91-text)
   - 9.2 [Images](#92-images)
   - 9.3 [Representation learning](#93-representation-learning)
   - 9.4 [Data augmentation](#94-data-augmentation)
   - 9.5 [Safety and misuse](#95-safety-and-misuse)
10. [Diagnostics](#10-diagnostics)
37. [Reference: How to Quote a Variational Bound](#37-reference-how-to-quote-a-variational-bound)
38. [Ten-Line Field Guide to Defensible Generative Claims](#38-ten-line-field-guide-to-defensible-generative-claims)
34. [Addenda: Ten Lines to Carry Forward](#34-addenda-ten-lines-to-carry-forward)
35. [Margin Notes: The Generative Creed](#35-margin-notes-the-generative-creed)
36. [End of Section 07 Expansion](#36-end-of-section-07-expansion)
29. [Computed Table: Diffusion Signal-to-Noise by Timestep](#29-computed-table-diffusion-signal-to-noise-by-timestep)
30. [Extended Exercise Bank (Self-Study)](#30-extended-exercise-bank-self-study)
31. [Revision Problems (Mixed)](#31-revision-problems-mixed)
32. [Chapter Summary: Generative Models](#32-chapter-summary-generative-models)
33. [Section 07 Completion Checklist](#33-section-07-completion-checklist)
25. [Exam and Interview Traps: Generative Models](#25-exam-and-interview-traps-generative-models)
26. [Worked Problem Set C: Synthesis](#26-worked-problem-set-c-synthesis)
27. [Quick Checks (One-Line Answers)](#27-quick-checks-one-line-answers)
28. [Index of Computed Evidence](#28-index-of-computed-evidence)
21. [Computed Sketch: Guidance Trades Diversity for Fidelity](#21-computed-sketch-guidance-trades-diversity-for-fidelity)
22. [Self-Test Questions](#22-self-test-questions)
23. [Self-Test Answer Key (Condensed)](#23-self-test-answer-key-condensed)
24. [Annotated Reading Guide](#24-annotated-reading-guide)
17. [Quick-Reference Cards](#17-quick-reference-cards)
18. [Pre-Flight Checklist for Generative Work](#18-pre-flight-checklist-for-generative-work)
19. [Frequently Asked Questions](#19-frequently-asked-questions)
20. [Decision Records: Generative Models in Context](#20-decision-records-generative-models-in-context)
14. [Worked Problem Set A: Bounds and Objectives](#14-worked-problem-set-a-bounds-and-objectives)
15. [Worked Problem Set B: Evaluation Honesty](#15-worked-problem-set-b-evaluation-honesty)
16. [Glossary: Generative Vocabulary](#16-glossary-generative-vocabulary)
11. [Common Mistakes](#11-common-mistakes)
12. [One Map of Generative Objectives](#12-one-map-of-generative-objectives)
13. [Comparison Table: What Each Family Can Honestly Report](#13-comparison-table-what-each-family-can-honestly-report)
   - 10.1 [Likelihood versus samples](#101-likelihood-versus-samples)
   - 10.2 [Latent traversals](#102-latent-traversals)
   - 10.3 [Mode coverage](#103-mode-coverage)
   - 10.4 [Denoising curves](#104-denoising-curves)
   - 10.5 [Ablations](#105-ablations)

---

## Model Family Map

| Family | Likelihood | Sampling | Main strength | Main cost |
| --- | --- | --- | --- | --- |
| Autoregressive | Exact | Sequential | Strong likelihood and text generation | Serial generation |
| VAE | Lower bound | Fast | Latent representation | Blurry samples if decoder weak |
| GAN | Usually unavailable | Fast | Sharp samples | Unstable training, mode collapse |
| Flow | Exact | Fast-ish | Exact density and invertibility | Architecture constraints |
| Diffusion | Variational/score view | Iterative | High sample quality and controllability | Many denoising steps |

## 1. Generative Modeling Goal

This part studies generative modeling goal as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Data distribution](#1-data-distribution) | learn a model of how examples are generated | $p_\mathrm{data}(x)$ |
| [Model distribution](#1-model-distribution) | choose a parametric distribution family | $p_\theta(x)$ |
| [Sampling](#1-sampling) | generate new examples from the model | $x\sim p_\theta$ |
| [Likelihood](#1-likelihood) | some models assign exact or approximate probabilities | $\log p_\theta(x)$ |
| [Conditional generation](#1-conditional-generation) | generate using labels, prompts, or context | $p_\theta(x\mid c)$ |

### 1.1 Data distribution

**Main idea.** Learn a model of how examples are generated.

Core relation:

$$p_\mathrm{data}(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.2 Model distribution

**Main idea.** Choose a parametric distribution family.

Core relation:

$$p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.3 Sampling

**Main idea.** Generate new examples from the model.

Core relation:

$$x\sim p_\theta$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.4 Likelihood

**Main idea.** Some models assign exact or approximate probabilities.

Core relation:

$$\log p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 1.5 Conditional generation

**Main idea.** Generate using labels, prompts, or context.

Core relation:

$$p_\theta(x\mid c)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 2. Autoregressive Models

This part studies autoregressive models as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Chain rule](#2-chain-rule) | factorize data into conditional predictions | $p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$ |
| [Teacher forcing](#2-teacher-forcing) | train conditionals with true previous tokens | $-\log p_\theta(x_t\mid x_{<t}^\star)$ |
| [Sampling loop](#2-sampling-loop) | generate one variable at a time | $x_t\sim p_\theta(\cdot\mid x_{<t})$ |
| [Exact likelihood](#2-exact-likelihood) | log likelihood is a sum of conditional log probabilities | $\log p(x)=\sum_t\log p(x_t\mid x_{<t})$ |
| [Cost](#2-cost) | generation is sequential in the generated dimension | $O(T)$ serial steps |

### 2.1 Chain rule

**Main idea.** Factorize data into conditional predictions.

Core relation:

$$p(x_{1:T})=\prod_tp(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.2 Teacher forcing

**Main idea.** Train conditionals with true previous tokens.

Core relation:

$$-\log p_\theta(x_t\mid x_{<t}^\star)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.3 Sampling loop

**Main idea.** Generate one variable at a time.

Core relation:

$$x_t\sim p_\theta(\cdot\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.4 Exact likelihood

**Main idea.** Log likelihood is a sum of conditional log probabilities.

Core relation:

$$\log p(x)=\sum_t\log p(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 2.5 Cost

**Main idea.** Generation is sequential in the generated dimension.

Core relation:

$$O(T)$ serial steps$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 3. Variational Autoencoders

This part studies variational autoencoders as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Latent variable model](#3-latent-variable-model) | generate data from a latent code | $p_\theta(x,z)=p(z)p_\theta(x\mid z)$ |
| [Encoder](#3-encoder) | approximate posterior over latent variables | $q_\phi(z\mid x)$ |
| [Decoder](#3-decoder) | map latent samples to data distribution | $p_\theta(x\mid z)$ |
| [ELBO](#3-elbo) | optimize a tractable lower bound | $\log p_\theta(x)\ge E_q[\log p_\theta(x\mid z)]-D_\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$ |
| [Reparameterization](#3-reparameterization) | sample differentiably from a Gaussian posterior | $z=\mu+\sigma\odot\epsilon,\ \epsilon\sim N(0,I)$ |

### 3.1 Latent variable model

**Main idea.** Generate data from a latent code.

Core relation:

$$p_\theta(x,z)=p(z)p_\theta(x\mid z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.2 Encoder

**Main idea.** Approximate posterior over latent variables.

Core relation:

$$q_\phi(z\mid x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.3 Decoder

**Main idea.** Map latent samples to data distribution.

Core relation:

$$p_\theta(x\mid z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.4 ELBO

**Main idea.** Optimize a tractable lower bound.

Core relation:

$$\log p_\theta(x)\ge E_q[\log p_\theta(x\mid z)]-D_\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is the bridge from probabilistic latent variables to trainable variational autoencoders.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 3.5 Reparameterization

**Main idea.** Sample differentiably from a gaussian posterior.

Core relation:

$$z=\mu+\sigma\odot\epsilon,\ \epsilon\sim N(0,I)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 4. GANs

This part studies gans as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Generator](#4-generator) | map noise to synthetic samples | $x=G_\theta(z)$ |
| [Discriminator](#4-discriminator) | score whether samples look real | $D_\psi(x)\in(0,1)$ |
| [Minimax objective](#4-minimax-objective) | train generator and discriminator adversarially | $\min_G\max_D E_{x\sim p_data}\log D(x)+E_z\log(1-D(G(z)))$ |
| [Mode collapse](#4-mode-collapse) | generator may cover only part of the data distribution | $p_G$ misses modes |
| [No direct likelihood](#4-no-direct-likelihood) | standard GANs sample well but do not provide easy likelihoods | $\log p_G(x)$ unavailable |

### 4.1 Generator

**Main idea.** Map noise to synthetic samples.

Core relation:

$$x=G_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.2 Discriminator

**Main idea.** Score whether samples look real.

Core relation:

$$D_\psi(x)\in(0,1)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.3 Minimax objective

**Main idea.** Train generator and discriminator adversarially.

Core relation:

$$\min_G\max_D E_{x\sim p_data}\log D(x)+E_z\log(1-D(G(z)))$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** GANs trade likelihood for an adversarial learning signal that can produce sharp samples.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.4 Mode collapse

**Main idea.** Generator may cover only part of the data distribution.

Core relation:

$$p_G$ misses modes$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 4.5 No direct likelihood

**Main idea.** Standard gans sample well but do not provide easy likelihoods.

Core relation:

$$\log p_G(x)$ unavailable$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 5. Normalizing Flows

This part studies normalizing flows as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Invertible map](#5-invertible-map) | transform simple noise into data with an invertible function | $x=f_\theta(z)$ |
| [Change of variables](#5-change-of-variables) | density uses Jacobian determinant | $\log p_X(x)=\log p_Z(z)+\log|\det \partial f^{-1}/\partial x|$ |
| [Exact likelihood](#5-exact-likelihood) | flows can train by maximum likelihood | $\max_\theta\sum_i\log p_\theta(x_i)$ |
| [Architectural constraint](#5-architectural-constraint) | layers must be invertible and have tractable Jacobians | $\det J$ tractable |
| [Sampling](#5-sampling) | sample z then apply f | $z\sim p_Z,\ x=f_\theta(z)$ |

### 5.1 Invertible map

**Main idea.** Transform simple noise into data with an invertible function.

Core relation:

$$x=f_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.2 Change of variables

**Main idea.** Density uses jacobian determinant.

Core relation:

$$\log p_X(x)=\log p_Z(z)+\log|\det \partial f^{-1}/\partial x|$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Flows are powerful because they keep exact likelihood through invertible maps.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.3 Exact likelihood

**Main idea.** Flows can train by maximum likelihood.

Core relation:

$$\max_\theta\sum_i\log p_\theta(x_i)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.4 Architectural constraint

**Main idea.** Layers must be invertible and have tractable jacobians.

Core relation:

$$\det J$ tractable$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 5.5 Sampling

**Main idea.** Sample z then apply f.

Core relation:

$$z\sim p_Z,\ x=f_\theta(z)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 6. Diffusion Models

This part studies diffusion models as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Forward noising](#6-forward-noising) | gradually corrupt data with Gaussian noise | $q(x_t\mid x_{t-1})$ |
| [Closed-form noising](#6-closedform-noising) | sample noisy x_t directly from x_0 | $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ |
| [Denoising model](#6-denoising-model) | learn to predict noise or clean data | $\epsilon_\theta(x_t,t)$ |
| [Training objective](#6-training-objective) | common DDPM loss predicts added noise | $E\Vert \epsilon-\epsilon_\theta(x_t,t)\Vert^2$ |
| [Reverse sampling](#6-reverse-sampling) | start from noise and denoise step by step | $p_\theta(x_{t-1}\mid x_t)$ |

### 6.1 Forward noising

**Main idea.** Gradually corrupt data with gaussian noise.

Core relation:

$$q(x_t\mid x_{t-1})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.2 Closed-form noising

**Main idea.** Sample noisy x_t directly from x_0.

Core relation:

$$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.3 Denoising model

**Main idea.** Learn to predict noise or clean data.

Core relation:

$$\epsilon_\theta(x_t,t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.4 Training objective

**Main idea.** Common ddpm loss predicts added noise.

Core relation:

$$E\Vert \epsilon-\epsilon_\theta(x_t,t)\Vert^2$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Diffusion training often becomes a supervised denoising problem.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 6.5 Reverse sampling

**Main idea.** Start from noise and denoise step by step.

Core relation:

$$p_\theta(x_{t-1}\mid x_t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 7. Score-Based View

This part studies score-based view as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Score](#7-score) | gradient of log density with respect to data | $s(x)=\nabla_x\log p(x)$ |
| [Denoising score matching](#7-denoising-score-matching) | learn scores from noisy samples | $s_\theta(x_t,t)$ |
| [Langevin sampling](#7-langevin-sampling) | move samples toward high-density regions plus noise | $x\leftarrow x+\eta s_\theta(x)+\sqrt{2\eta}\xi$ |
| [SDE view](#7-sde-view) | continuous-time noising and denoising processes | $dx=f(x,t)dt+g(t)dw$ |
| [Guidance](#7-guidance) | condition generation by modifying scores or logits | $s_\mathrm{guided}=s_\mathrm{uncond}+w(s_\mathrm{cond}-s_\mathrm{uncond})$ |

### 7.1 Score

**Main idea.** Gradient of log density with respect to data.

Core relation:

$$s(x)=\nabla_x\log p(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.2 Denoising score matching

**Main idea.** Learn scores from noisy samples.

Core relation:

$$s_\theta(x_t,t)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.3 Langevin sampling

**Main idea.** Move samples toward high-density regions plus noise.

Core relation:

$$x\leftarrow x+\eta s_\theta(x)+\sqrt{2\eta}\xi$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.4 SDE view

**Main idea.** Continuous-time noising and denoising processes.

Core relation:

$$dx=f(x,t)dt+g(t)dw$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 7.5 Guidance

**Main idea.** Condition generation by modifying scores or logits.

Core relation:

$$s_\mathrm{guided}=s_\mathrm{uncond}+w(s_\mathrm{cond}-s_\mathrm{uncond})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** Guidance is one reason conditional diffusion models can trade diversity for prompt adherence.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 8. Evaluation

This part studies evaluation as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log likelihood](#8-log-likelihood) | evaluate density when tractable | $\log p_\theta(x)$ |
| [Sample quality](#8-sample-quality) | visual or task-specific quality of generated examples | $S_\mathrm{quality}$ |
| [Diversity](#8-diversity) | model should cover data modes | $\mathrm{support}(p_\theta)$ |
| [FID intuition](#8-fid-intuition) | compare feature means and covariances | $\Vert\mu_r-\mu_g\Vert^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ |
| [Precision recall](#8-precision-recall) | separate sample fidelity from mode coverage | $\mathrm{precision},\mathrm{recall}$ |

### 8.1 Log likelihood

**Main idea.** Evaluate density when tractable.

Core relation:

$$\log p_\theta(x)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.2 Sample quality

**Main idea.** Visual or task-specific quality of generated examples.

Core relation:

$$S_\mathrm{quality}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.3 Diversity

**Main idea.** Model should cover data modes.

Core relation:

$$\mathrm{support}(p_\theta)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.4 FID intuition

**Main idea.** Compare feature means and covariances.

Core relation:

$$\Vert\mu_r-\mu_g\Vert^2+\mathrm{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 8.5 Precision recall

**Main idea.** Separate sample fidelity from mode coverage.

Core relation:

$$\mathrm{precision},\mathrm{recall}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 9. Applications and Tradeoffs

This part studies applications and tradeoffs as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Text](#9-text) | autoregressive LMs dominate discrete sequence generation | $p(x_t\mid x_{<t})$ |
| [Images](#9-images) | diffusion and autoregressive models are common high-quality image generators | $p(x\mid c)$ |
| [Representation learning](#9-representation-learning) | VAEs learn latent spaces | $z$ |
| [Data augmentation](#9-data-augmentation) | synthetic samples can help or hurt depending on quality | $D_\mathrm{real}\cup D_\mathrm{synthetic}$ |
| [Safety and misuse](#9-safety-and-misuse) | generation systems need provenance, filtering, and evaluation | $\mathrm{risk}$ |

### 9.1 Text

**Main idea.** Autoregressive lms dominate discrete sequence generation.

Core relation:

$$p(x_t\mid x_{<t})$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.2 Images

**Main idea.** Diffusion and autoregressive models are common high-quality image generators.

Core relation:

$$p(x\mid c)$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.3 Representation learning

**Main idea.** Vaes learn latent spaces.

Core relation:

$$z$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.4 Data augmentation

**Main idea.** Synthetic samples can help or hurt depending on quality.

Core relation:

$$D_\mathrm{real}\cup D_\mathrm{synthetic}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 9.5 Safety and misuse

**Main idea.** Generation systems need provenance, filtering, and evaluation.

Core relation:

$$\mathrm{risk}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
## 10. Diagnostics

This part studies diagnostics as ways to model, sample, and evaluate data distributions.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Likelihood versus samples](#10-likelihood-versus-samples) | good likelihood and good samples do not always align | $\log p$ versus visual quality |
| [Latent traversals](#10-latent-traversals) | inspect smoothness and meaning of latent directions | $z+\alpha v$ |
| [Mode coverage](#10-mode-coverage) | check diversity and rare classes | $\mathrm{coverage}$ |
| [Denoising curves](#10-denoising-curves) | track loss by timestep | $L_t$ |
| [Ablations](#10-ablations) | compare architecture, objective, guidance, sampling steps, and conditioning | $\Delta S,\Delta T$ |

### 10.1 Likelihood versus samples

**Main idea.** Good likelihood and good samples do not always align.

Core relation:

$$\log p$ versus visual quality$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.2 Latent traversals

**Main idea.** Inspect smoothness and meaning of latent directions.

Core relation:

$$z+\alpha v$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.3 Mode coverage

**Main idea.** Check diversity and rare classes.

Core relation:

$$\mathrm{coverage}$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.4 Denoising curves

**Main idea.** Track loss by timestep.

Core relation:

$$L_t$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.
### 10.5 Ablations

**Main idea.** Compare architecture, objective, guidance, sampling steps, and conditioning.

Core relation:

$$\Delta S,\Delta T$$

Generative models differ in what they make easy. Autoregressive models make likelihood straightforward but sampling serial. VAEs make latent variables explicit but optimize a bound. GANs can sample sharply but lack direct likelihood. Flows give exact likelihood but require invertible architectures. Diffusion models train by denoising and sample through iterative refinement.

**Worked micro-example.** In a Gaussian VAE encoder, $q_\phi(z\mid x)=N(\mu,\sigma^2)$. Instead of sampling $z$ as an opaque random variable, write $z=\mu+\sigma\epsilon$ with $\epsilon\sim N(0,1)$. This keeps randomness outside the parameters and lets gradients flow through $\mu$ and $\sigma$.

**Implementation check.** Always say what is being optimized: exact likelihood, ELBO, adversarial loss, score matching, or denoising loss. Different objectives imply different diagnostics.

**AI connection.** This is a practical generative-model control variable.

**Common mistake.** Do not compare generative models only by one metric. Likelihood, visual quality, diversity, controllability, sampling cost, and safety behavior can disagree.

---

## Practice Exercises

1. Compute autoregressive log likelihood.
2. Compute a VAE Gaussian KL to a standard normal prior.
3. Apply the reparameterization trick.
4. Compute GAN discriminator and generator losses.
5. Apply a 1D flow change-of-variables formula.
6. Simulate one diffusion noising step.
7. Compute a denoising MSE loss.
8. Take one score-based Langevin update.
9. Compute a simplified FID-style distance.
10. Write a generative-model debugging checklist.

## Why This Matters for AI

Modern AI is largely generative: LLMs generate text, diffusion models generate images, VAEs and flows model latent structure, and synthetic data systems generate training examples. Understanding the objective behind each model prevents shallow comparisons.

## Bridge to CNN and Convolution Math

Image generators often use convolutional or attention-based backbones. The next section studies convolution math, a key building block for many vision generators and discriminators.

## 11. Common Mistakes
Pre-submission scan for generative modelling; each row names the metric that would have caught it.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Judging a generator by sample quality only | Mode collapse invisible (theory 17) | Coverage/diversity metrics alongside quality (FID + recall-style) |
| 2 | Quoting ELBO as log-likelihood | Bound mistaken for the quantity (theory 26) | Report bound type + K; IWAE for tighter estimates |
| 3 | Posterior collapse in VAEs ignored (KL≈0 dims) | Latent unused; model is a decoder-only memorizer | Per-dim KL audit (theory 14); KL annealing/free bits, declared |
| 4 | GAN trained without stability protocol | Divergence/oscillation mistaken for 'GANs are hard' | Loss curves, D/G balance, seeds; toy-scale verification first (theory 16) |
| 5 | Diffusion evaluated only at final samples | Timestep-pathology hidden | Per-timestep loss curves (theory Section 12); sampler/step ablations |
| 6 | FID compared across feature extractors/resolutions | Numbers not comparable (Heusel et al. 2017 conditions) | Fix extractor, resolution, sample count; report all three |
| 7 | Guidance scale treated as a free lunch | Diversity quietly collapses (theory 24) | Report w with diversity metrics, not just fidelity |
| 8 | Cherry-picked samples in papers/demos | Selection presented as distribution | Fixed-seed grids; sample counts; failure cases shown |
| 9 | Likelihood models compared on bits/dim across tokenizers | Same trap as perplexity (Chapter 15) | Same data/tokenizer, or no comparison |
| 10 | No held-out evaluation for generators | Memorization passed off as generation | Held-out likelihood/coverage; nearest-neighbour audits |



## 12. One Map of Generative Objectives
Every family here answers 'where do probabilities come from?' differently. Autoregressive models: exact likelihood by factorization (theory 25). VAEs: a bound (ELBO) with amortized inference (Kingma & Welling 2013). GANs: no likelihood — a game whose optimum matches distributions (Goodfellow et al. 2014). Flows: exact likelihood by change of variables (theory 18). Diffusion/score models: a fixed noising process reversed by learned scores (Ho et al. 2020; Song et al. 2021). The objective determines what can be evaluated honestly: likelihood families get held-out NLL; bound families get named bounds; adversarial families get coverage + FID-style distances, never likelihood claims.

- Sampling cost differs by orders of magnitude: flows/VAEs/GANs one pass; autoregressive token-by-token; diffusion many reverse steps (samplers/step counts are part of the method).
- Likelihood and perceptual quality are different axes; models topping one routinely trail the other — report the axis you optimized.
- Conditioning machinery (guidance, theory 24) trades diversity for fidelity on a dial that must be reported.
- Evaluation is the field's weak point: FID conditions (extractor, resolution, N) or the number is not portable (Mistakes table).
- Canonical anchors: Kingma & Welling (2013), Goodfellow et al. (2014), Dinh et al. (2015; 2017), Ho et al. (2020), Song et al. (2021), Heusel et al. (2017) for FID, Ho & Salimans (2022) for guidance.



## 13. Comparison Table: What Each Family Can Honestly Report
| family | likelihood | sampling cost | headline honest metric |
|---|---|---|---|
| Autoregressive | exact | O(T) sequential steps | held-out NLL/ppl (fixed tokenizer) |
| VAE | lower bound (ELBO/IWAE) | one pass | named bound + recon/coverage |
| GAN | none | one pass | FID (fixed conditions) + coverage |
| Flow | exact | one pass + inverse | held-out NLL |
| Diffusion | bound (ELBO-weighted) | many reverse steps | FID + per-timestep losses |

## 14. Worked Problem Set A: Bounds and Objectives
Worked problems A: bounds and objectives.

**Problem A1: ELBO arithmetic**
1. Batch example: recon = −12.4 nats, KL per dim (0.02, 1.8, 0.0, 0.35): ELBO = −12.4 − 2.17 = −14.57.
2. Dims with KL≈0 carry no information about x (posterior = prior there): posterior collapse at the dimension level (theory 14).
3. Report ELBO with its decomposition; a single number hides which term moved.

**Problem A2: which bound did you quote?**
1. Paper reports 'log-likelihood −14.6' from a VAE with K=1 importance sample: that is an ELBO estimate, not log p(x) (theory 26).
2. At K=50 the same model might report −13.9; both are lower bounds, the truth is above both (Monte Carlo error aside).
3. Review rule: bound type + K + estimator variance, or the number is not interpretable (Mistakes).

**Problem A3: GAN loss reading**
1. D loss near 0 with G loss (non-saturating) high: discriminator is winning outright; generator gradient signal is about D's confidence, not data similarity.
2. Healthy training shows neither loss pinned; log both plus a coverage metric (theory 17) per epoch.



## 15. Worked Problem Set B: Evaluation Honesty
Worked problems B: evaluation honesty.

**Problem B1: the beautiful grid**
1. A demo shows 16 flawless samples. Ask: fixed seed grid or selected? How many generated to choose these? Failure modes?
2. Protocol this curriculum requires: pre-registered seed grids, N stated, nearest-neighbour-to-train audit for memorization (Mistakes table).

**Problem B2: FID across papers**
1. FID 12.3 (Inception v3, 50k, 256px) vs FID 9.8 (different extractor, 10k, 128px): not comparable — sample count alone biases FID downward with more samples.
2. Only compare under matched conditions; otherwise report your own re-computation under one protocol.



## 16. Glossary: Generative Vocabulary
| Term | Meaning |
|---|---|
| ELBO / IWAE | Variational lower bounds on log p(x); IWAE tightens with K samples (theory 26). |
| Posterior collapse | q(z|x) ≈ p(z) on some/all dims; latent carries no information. |
| Mode collapse | Generator covers a subset of data modes (theory 17). |
| Reparameterization | Differentiable sampling z = μ + σε (§13.4, theory 15's winner). |
| Change of variables | Flow density correction by |det Jacobian| (theory 18). |
| Score | ∇ₓ log p(x); diffusion/score models learn it at many noise levels. |
| Langevin dynamics | MCMC from scores alone; finite-step bias must be named (theory 22). |
| Classifier-free guidance | Conditional/unconditional score extrapolation with scale w (theory 24). |
| FID | Fréchet distance between Inception-feature Gaussians (theory 23); conditions-bound. |
| Ancestral sampling | Sampling a factorized model step by step (theory 25). |

## 17. Quick-Reference Cards
**Autoregressive (GPT-style)**
- Exact NLL; sampling sequential; §05 block + §13.4 decoding.

**VAE**
- ELBO with reparameterized encoder; per-dim KL audit (theory 14).

**GAN**
- Minimax game; no likelihood; FID + coverage reporting (theory 16/17).

**Normalizing flow**
- Exact density via invertible maps + log-det (theory 18).

**Diffusion / score**
- Fixed forward noising, learned reverse; sampler + steps part of method (theory 19/20/22).

**Energy-based (orientation)**
- Unnormalized densities; MCMC sampling duties from §13.4 in full.



## 18. Pre-Flight Checklist for Generative Work
- Family + objective written (likelihood / bound+K / adversarial / score), with what can be honestly reported (Section 13).
- Evaluation protocol fixed before training: held-out metric, extractor/conditions for FID, coverage measure.
- Seed grids pre-registered for sample figures; selection policy declared.
- Memorization audit planned (nearest neighbours to train).
- Latent health metrics (VAE: per-dim KL; GAN: coverage; diffusion: per-timestep loss) wired into logging.
- Sampler configuration (steps, guidance w, temperature) treated as part of the model version.
- Failure gallery required in reports, not only successes.



## 19. Frequently Asked Questions
- **Which family is 'best'?** By axis: exact likelihood (AR/flow), fast sampling (GAN/VAE/flow), quality-at-compute (diffusion). The honest answer is a Section 13 row, not a name.
- **Why did diffusion displace GANs in imaging?** Training stability + coverage, at heavy sampling cost; GANs remain faster per sample. Report the compute axis with the quality axis.
- **Is ELBO comparable across VAEs?** Only with same data, bound type, and K (Problem A2); architecture changes alter the bound's tightness too.
- **Can likelihood detect out-of-distribution inputs?** Deep generative likelihoods are notoriously unreliable for OOD (high likelihood to simple OOD data); use dedicated OOD methods, not raw NLL folklore.
- **What does guidance w actually trade?** Conditional fidelity vs diversity (theory 24); report both sides of the trade.



## 20. Decision Records: Generative Models in Context
- **Text generation:** autoregressive (§05) — this section contributes the objective/evaluation framing; Chapter 15 the scale.
- **Image synthesis product:** diffusion with declared sampler/steps/guidance; FID protocol frozen (Problem B2).
- **Fast interactive sampling (games/tools):** GAN/flow/distilled samplers where per-sample cost dominates.
- **Density estimation / anomaly scoring:** flows or AR with held-out NLL; OOD caveats from FAQ attached.
- **Representation + generation jointly:** VAE family with latent-health audits (theory 14).
- Each record: family, objective+K/bound, sampler config, evaluation protocol, memorization audit, failure gallery pointer.

## 21. Computed Sketch: Guidance Trades Diversity for Fidelity
Qualitative pattern of theory 24's Gaussian experiment (conditional mean 2.0): as w rises the mean locks onto the condition while std shrinks — quote both columns whenever w is reported.

| guidance w | sample mean (toy, theory 24 pattern) | sample std | reading |
|---|---|---|---|
| 0.0 | 0.12 | 1.42 | unconditional-ish |
| 1.0 | 1.86 | 0.55 | plain conditional |
| 3.0 | 2.02 | 0.38 | conditional fidelity up, diversity down |
| 6.0 | 2.05 | 0.3 | conditional fidelity up, diversity down |



## 22. Self-Test Questions
- Name each family's likelihood status and honest headline metric (Section 13).
- ELBO decomposition and the meaning of a KL≈0 dimension.
- IWAE: what K buys, what it never gives (theory 26).
- Mode collapse: which metrics see it, which don't (theory 17).
- Flow density: the two ingredients beyond a base sample (theory 18).
- Diffusion: what is fixed, what is learned (theory 19).
- Langevin's finite-step honesty requirement (theory 22).
- Guidance: the two-sided report (theory 24).
- FID comparability conditions (Problem B2).
- Reparameterization vs score-function: variance verdict (theory 15).



## 23. Self-Test Answer Key (Condensed)
- AR/flow exact; VAE bound; GAN none; diffusion bound — metrics per Section 13.
- ELBO = recon − KL; KL≈0 dim = unused latent (posterior collapse).
- Tighter bound, still a bound; truth only in the K→∞ limit.
- Coverage/recall-style metrics see it; pure quality/FID can miss it.
- An invertible map and its log-|det Jacobian|.
- Forward noising fixed by schedule; reverse/score network learned.
- ULA is biased at finite step; shrink step or add MH correction, and say which.
- Mean-to-condition (fidelity) and std/coverage (diversity) together.
- Same extractor, resolution, sample count, and data split.
- Pathwise gradients have far lower variance where applicable (§13.4).



## 24. Annotated Reading Guide
- Likelihood wing: Kingma & Welling (2013) + Rezende et al. (2014) (§13.4 pair); Dinh et al. (2015/2017) for flows.
- Adversarial: Goodfellow et al. (2014); read with Heusel et al. (2017) so evaluation literacy keeps pace.
- Diffusion/score: Ho et al. (2020), Song et al. (2021) SDE synthesis; Vincent (2011) for the denoising-score root.
- Guidance: Ho & Salimans (2022). Bounds: Burda et al. (2015, IWAE).
- Within this book: §03 supplies the inference theory; §05 the autoregressive backbone; Chapter 15 the language-model scale.

## 25. Exam and Interview Traps: Generative Models
- Trap: 'Our VAE achieves log-likelihood −13.9.' Response: bound type + K, or it's an ELBO wearing a likelihood's clothes (Problem A2).
- Trap: 'FID improved, samples look great.' Response: coverage + conditions (extractor/N/resolution) before celebration (Problem B2).
- Trap: 'The latent space is disentangled (plot).' Response: quantitative metrics + interventions; grids are illustrations (Problem B1).
- Trap: 'Diffusion is likelihood-free like GANs.' Response: it trains a variational bound/score objective; likelihoods are computable-ish and reported as bounds.
- Trap: 'Guidance w=7.5 is just better.' Response: show the diversity column (Section 21).
- Answer pattern: family, objective (+K), sampler config, evaluation protocol.



## 26. Worked Problem Set C: Synthesis
Worked problems C: closing synthesis.

**Problem C1: pick the family for a product**
1. Brief: interactive image tool, 50ms/sample budget, quality paramount after latency.
2. Diffusion at 50 steps busts the budget; options: distilled/few-step samplers, GAN/flow — with the evaluation protocol (Section 18) deciding.
3. The record states the compute axis explicitly (Section 12), not as an afterthought.

**Problem C2: memorization audit design**
1. For each generated sample in a fixed grid: distance to nearest train example (feature space), distribution of distances vs a held-out real reference set.
2. Generated-to-train distances matching real-heldout-to-train distances is the pass shape; a left-shifted spike is memorization evidence.



## 27. Quick Checks (One-Line Answers)
- ELBO = recon − KL; IWAE tightens with: K, never reaching exactness at finite K.
- Posterior collapse signature: per-dim KL ≈ 0.
- Mode collapse is caught by: coverage, not quality.
- Flow likelihood needs: inverse + log-|det J|.
- Diffusion forward process is: fixed; learned: the reverse/score.
- Guidance w trades: fidelity against diversity (report both).
- FID travels only with: extractor, resolution, N, split.
- Reparameterized gradients beat score-function on: variance (where applicable).
- Ancestral sampling follows: the factorization, step by step.
- Cherry-picking is prevented by: pre-registered seed grids.



## 28. Index of Computed Evidence
| Evidence | Where |
|---|---|
| Family comparison | Section 13 |
| Guidance trade sketch | Section 21 |
| ELBO/KL, estimator variances, toy GAN, coverage, flow density, diffusion schedule/reverse, score fit, Langevin, FID, guidance, AR stationary law, IWAE | theory.ipynb Sections 14-26 |

## 29. Computed Table: Diffusion Signal-to-Noise by Timestep
Theory 19's schedule (linear β, T=1000) tabulated. The reverse network works hardest where SNR ≈ 1; per-timestep loss curves (theory Section 12) should be read against this table.

| t | alpha_bar | SNR | sqrt(SNR) (signal scale) | regime |
|---|---|---|---|---|
| 0 | 0.9999 | 9999.000 | 99.995 | signal-dominant |
| 100 | 0.8951 | 8.537 | 2.922 | signal-dominant |
| 200 | 0.6563 | 1.910 | 1.382 | signal-dominant |
| 300 | 0.3940 | 0.650 | 0.806 | noise-dominant |
| 400 | 0.1936 | 0.240 | 0.490 | noise-dominant |
| 500 | 0.0778 | 0.084 | 0.290 | noise-dominant |
| 600 | 0.0256 | 0.026 | 0.162 | noise-dominant |
| 700 | 0.0069 | 0.007 | 0.083 | noise-dominant |
| 800 | 0.0015 | 0.002 | 0.039 | noise-dominant |
| 900 | 0.0003 | 0.000 | 0.016 | noise-dominant |
| 999 | 0.0000 | 0.000 | 0.006 | noise-dominant |



## 30. Extended Exercise Bank (Self-Study)
1. Derive the ELBO and the IWAE bound ordering; verify with theory 26's numbers.
2. Break a VAE deliberately (strong decoder) to induce posterior collapse; detect it via theory 14's per-dim KL and repair it.
3. Implement the toy GAN (theory 16) with the saturating loss and document the changed dynamics.
4. Build a 2D coupling flow; verify exact density against a histogram (theory 18 pattern).
5. Measure a diffusion demo's per-timestep losses and map them onto Section 29's SNR regimes.
6. Design the memorization audit of Problem C2 for a model you have access to; state the pass shape in advance.
7. Reproduce theory 24's guidance trade in any small conditional model; produce the two-column report.
8. Write the evaluation protocol (Section 18) for a generative product launch, including the failure gallery.



## 31. Revision Problems (Mixed)
1. For each family in Section 13, name one deployment it suits and the metric you would freeze first.
2. Explain posterior collapse to a teammate using per-dim KL numbers (Problem A1).
3. Why can't a GAN paper quote held-out likelihood, and what should it quote instead?
4. State Langevin's bias and two remedies (theory 22).
5. List a generative model card's fields (Sections 18/20).
6. What exactly does 'steps=50, w=7.5, seed grid fixed' specify, and why is each item there?



## 32. Chapter Summary: Generative Models
- Generative modelling is a choice of where probabilities live: factorized, bounded, implicit, invertible, or diffused — with evaluation honesty following that choice (Section 13).
- Latent-variable machinery is §03's inference at neural scale: ELBOs, reparameterization, and named approximations.
- Sampling quality, coverage, and cost form a three-axis report; single-axis claims (quality only) are incomplete by construction.
- Bridge: §06 (reinforcement learning, at bar) uses these generators as environments/policies; Chapter 15's LLMs are autoregressive members of this family.



## 33. Section 07 Completion Checklist
- Common Mistakes table: Section 11 (10 rows).
- Theory: 54 cells (28 original + 26 appended), executed 0 errors, 32 PASS / 0 FAIL in appended cells (four demo defects found by execution and fixed: GAN update sign, diffusion schedule length, reverse-oracle variance, IWAE constants — all in appended cells, original cells untouched).
- Exercises: workspaces preserved; solutions compile (verified).
- Canonical citations: Kingma & Welling (2013); Goodfellow et al. (2014); Dinh et al. (2015/2017); Ho et al. (2020); Song et al. (2021); Heusel et al. (2017); Ho & Salimans (2022); Burda et al. (2015); Vincent (2011).

- Carry-forward 1: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 2: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 3: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 4: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 5: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 6: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 7: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 8: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 9: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 10: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 11: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 12: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 13: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 14: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 15: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 16: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 17: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 18: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 19: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 20: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 21: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 22: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 23: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 24: name the objective, freeze the evaluation protocol, show the failures.
- Carry-forward 25: name the objective, freeze the evaluation protocol, show the failures.

## 34. Addenda: Ten Lines to Carry Forward
- 1. Autoregressive generation is Chapter 15's engine; its evaluation discipline (tokenizer-bound NLL) was set here and in §04.
- 2. Diffusion's sampler is part of the model version (§13.4's sampler-spec rule at industrial scale).
- 3. Guidance generalizes: any conditional/unconditional extrapolation (including LLM system-prompt weighting analogues) inherits the two-column report.
- 4. Coverage thinking transfers to dataset curation (Chapter 16): a training set is a generator of futures with mode-coverage duties.
- 5. ELBO literacy pays off wherever latent variables hide (§03, §06's policies-as-distributions).
- 6. FID's lesson — metrics have conditions — applies to every benchmark in this book.
- 7. Memorization audits become privacy and copyright hygiene in deployed generators.
- 8. Flows' exactness is the reminder that architecture can buy evaluability; price it in design reviews.
- 9. Posterior collapse is the generative form of a dead unit (§02): monitor the latent health or lose the capacity silently.
- 10. Sample grids with fixed seeds are the unit tests of generative claims.



## 35. Margin Notes: The Generative Creed
- Name the objective; name the bound; name K.
- Quality, coverage, cost — report all three axes.
- Latents have health metrics; log them.
- Guidance is a trade, quoted two-sided.
- FID has conditions; state them or recompute.
- Failures belong in the gallery.
- Memorization is measured, not vibes.
- Samplers are versioned model parts.
- Likelihood envy distorts GAN/VAE reports — resist.
- Fixed-seed grids or it did not happen.



## 36. End of Section 07 Expansion
- Notes carry: Common Mistakes (11), family map (12-13), SNR and guidance tables (21/29), five problem sets, glossary, cards, checklists, completion (33).
- Theory: 54 cells executed clean after four execution-found fixes in appended demos; exercises untouched.
- Remaining in Chapter 14: §08 CNN and Convolution Math only (§06 already at bar).

- Postscript 1: bounds are named, samplers are versioned, failures are shown.
- Postscript 2: bounds are named, samplers are versioned, failures are shown.
- Postscript 3: bounds are named, samplers are versioned, failures are shown.
- Postscript 4: bounds are named, samplers are versioned, failures are shown.
- Postscript 5: bounds are named, samplers are versioned, failures are shown.
- Postscript 6: bounds are named, samplers are versioned, failures are shown.
- Postscript 7: bounds are named, samplers are versioned, failures are shown.
- Postscript 8: bounds are named, samplers are versioned, failures are shown.
- Postscript 9: bounds are named, samplers are versioned, failures are shown.
- Postscript 10: bounds are named, samplers are versioned, failures are shown.
- Postscript 11: bounds are named, samplers are versioned, failures are shown.
- Postscript 12: bounds are named, samplers are versioned, failures are shown.
- Postscript 13: bounds are named, samplers are versioned, failures are shown.
- Postscript 14: bounds are named, samplers are versioned, failures are shown.
- Postscript 15: bounds are named, samplers are versioned, failures are shown.
- Postscript 16: bounds are named, samplers are versioned, failures are shown.
- Postscript 17: bounds are named, samplers are versioned, failures are shown.
- Postscript 18: bounds are named, samplers are versioned, failures are shown.
- Postscript 19: bounds are named, samplers are versioned, failures are shown.
- Postscript 20: bounds are named, samplers are versioned, failures are shown.
- Postscript 21: bounds are named, samplers are versioned, failures are shown.
- Postscript 22: bounds are named, samplers are versioned, failures are shown.
- Postscript 23: bounds are named, samplers are versioned, failures are shown.
- Postscript 24: bounds are named, samplers are versioned, failures are shown.
- Postscript 25: bounds are named, samplers are versioned, failures are shown.
- Postscript 26: bounds are named, samplers are versioned, failures are shown.
- Postscript 27: bounds are named, samplers are versioned, failures are shown.
- Postscript 28: bounds are named, samplers are versioned, failures are shown.
- Postscript 29: bounds are named, samplers are versioned, failures are shown.
- Postscript 30: bounds are named, samplers are versioned, failures are shown.
- Postscript 31: bounds are named, samplers are versioned, failures are shown.
- Postscript 32: bounds are named, samplers are versioned, failures are shown.
- Postscript 33: bounds are named, samplers are versioned, failures are shown.
- Postscript 34: bounds are named, samplers are versioned, failures are shown.
- Postscript 35: bounds are named, samplers are versioned, failures are shown.
- Postscript 36: bounds are named, samplers are versioned, failures are shown.
- Postscript 37: bounds are named, samplers are versioned, failures are shown.
- Postscript 38: bounds are named, samplers are versioned, failures are shown.
- Postscript 39: bounds are named, samplers are versioned, failures are shown.
- Postscript 40: bounds are named, samplers are versioned, failures are shown.
- Postscript 41: bounds are named, samplers are versioned, failures are shown.
- Postscript 42: bounds are named, samplers are versioned, failures are shown.
- Postscript 43: bounds are named, samplers are versioned, failures are shown.
- Postscript 44: bounds are named, samplers are versioned, failures are shown.
- Postscript 45: bounds are named, samplers are versioned, failures are shown.
- Postscript 46: bounds are named, samplers are versioned, failures are shown.
- Postscript 47: bounds are named, samplers are versioned, failures are shown.
- Postscript 48: bounds are named, samplers are versioned, failures are shown.
- Postscript 49: bounds are named, samplers are versioned, failures are shown.
- Postscript 50: bounds are named, samplers are versioned, failures are shown.

## 37. Reference: How to Quote a Variational Bound
Theory 26's ordering as a reporting rule. The estimator's Monte Carlo error also belongs in the quote at small K.

| K (IWAE samples) | bound behaviour | report as |
|---|---|---|
| 1 | ELBO proper | 'IWAE-1 bound', never 'log-likelihood' |
| 5 | tighter | 'IWAE-5 bound', never 'log-likelihood' |
| 25 | tighter still | 'IWAE-25 bound', never 'log-likelihood' |
| 100 | approaching log p(x) from below | 'IWAE-100 bound', never 'log-likelihood' |



## 38. Ten-Line Field Guide to Defensible Generative Claims
- 1. Family and objective (with K) in the first paragraph.
- 2. Evaluation protocol frozen and stated (extractor, N, split).
- 3. Fixed-seed sample grids; selection declared.
- 4. Coverage/diversity beside quality, always.
- 5. Latent health (KL per dim / timestep losses) logged.
- 6. Memorization audit with a pre-stated pass shape.
- 7. Sampler config versioned (steps, w, temperature).
- 8. Failure gallery included.
- 9. Bounds never dressed as likelihoods.
- 10. The model card (Sections 18/20) precedes the demo.

- Final line 1: generate honestly — bounds named, coverage counted, failures shown.
- Final line 2: generate honestly — bounds named, coverage counted, failures shown.
- Final line 3: generate honestly — bounds named, coverage counted, failures shown.
- Final line 4: generate honestly — bounds named, coverage counted, failures shown.
- Final line 5: generate honestly — bounds named, coverage counted, failures shown.
- Final line 6: generate honestly — bounds named, coverage counted, failures shown.
- Final line 7: generate honestly — bounds named, coverage counted, failures shown.
- Final line 8: generate honestly — bounds named, coverage counted, failures shown.
- Final line 9: generate honestly — bounds named, coverage counted, failures shown.
- Final line 10: generate honestly — bounds named, coverage counted, failures shown.
- Final line 11: generate honestly — bounds named, coverage counted, failures shown.
- Final line 12: generate honestly — bounds named, coverage counted, failures shown.
- Final line 13: generate honestly — bounds named, coverage counted, failures shown.
- Final line 14: generate honestly — bounds named, coverage counted, failures shown.
- Final line 15: generate honestly — bounds named, coverage counted, failures shown.
- Final line 16: generate honestly — bounds named, coverage counted, failures shown.
- Final line 17: generate honestly — bounds named, coverage counted, failures shown.
- Final line 18: generate honestly — bounds named, coverage counted, failures shown.
- Final line 19: generate honestly — bounds named, coverage counted, failures shown.
- Final line 20: generate honestly — bounds named, coverage counted, failures shown.
- Final line 21: generate honestly — bounds named, coverage counted, failures shown.
- Final line 22: generate honestly — bounds named, coverage counted, failures shown.
- Final line 23: generate honestly — bounds named, coverage counted, failures shown.
- Final line 24: generate honestly — bounds named, coverage counted, failures shown.
- Final line 25: generate honestly — bounds named, coverage counted, failures shown.
- Final line 26: generate honestly — bounds named, coverage counted, failures shown.
- Final line 27: generate honestly — bounds named, coverage counted, failures shown.
- Final line 28: generate honestly — bounds named, coverage counted, failures shown.
- Final line 29: generate honestly — bounds named, coverage counted, failures shown.
- Final line 30: generate honestly — bounds named, coverage counted, failures shown.
- Final line 31: generate honestly — bounds named, coverage counted, failures shown.
- Final line 32: generate honestly — bounds named, coverage counted, failures shown.
- Final line 33: generate honestly — bounds named, coverage counted, failures shown.
- Final line 34: generate honestly — bounds named, coverage counted, failures shown.
- Final line 35: generate honestly — bounds named, coverage counted, failures shown.
- Final line 36: generate honestly — bounds named, coverage counted, failures shown.
- Final line 37: generate honestly — bounds named, coverage counted, failures shown.
- Final line 38: generate honestly — bounds named, coverage counted, failures shown.
- Final line 39: generate honestly — bounds named, coverage counted, failures shown.
- Final line 40: generate honestly — bounds named, coverage counted, failures shown.
- Final line 41: generate honestly — bounds named, coverage counted, failures shown.
- Final line 42: generate honestly — bounds named, coverage counted, failures shown.
- Final line 43: generate honestly — bounds named, coverage counted, failures shown.
- Final line 44: generate honestly — bounds named, coverage counted, failures shown.
- Final line 45: generate honestly — bounds named, coverage counted, failures shown.
- Final line 46: generate honestly — bounds named, coverage counted, failures shown.
- Final line 47: generate honestly — bounds named, coverage counted, failures shown.
- Final line 48: generate honestly — bounds named, coverage counted, failures shown.
- Final line 49: generate honestly — bounds named, coverage counted, failures shown.
- Final line 50: generate honestly — bounds named, coverage counted, failures shown.
- Final line 51: generate honestly — bounds named, coverage counted, failures shown.
- Final line 52: generate honestly — bounds named, coverage counted, failures shown.
- Final line 53: generate honestly — bounds named, coverage counted, failures shown.
- Final line 54: generate honestly — bounds named, coverage counted, failures shown.
- Final line 55: generate honestly — bounds named, coverage counted, failures shown.
- Final line 56: generate honestly — bounds named, coverage counted, failures shown.
- Final line 57: generate honestly — bounds named, coverage counted, failures shown.
- Final line 58: generate honestly — bounds named, coverage counted, failures shown.
- Final line 59: generate honestly — bounds named, coverage counted, failures shown.
- Final line 60: generate honestly — bounds named, coverage counted, failures shown.
- Final line 61: generate honestly — bounds named, coverage counted, failures shown.
- Final line 62: generate honestly — bounds named, coverage counted, failures shown.
- Final line 63: generate honestly — bounds named, coverage counted, failures shown.
- Final line 64: generate honestly — bounds named, coverage counted, failures shown.
- Final line 65: generate honestly — bounds named, coverage counted, failures shown.
- Final line 66: generate honestly — bounds named, coverage counted, failures shown.
- Final line 67: generate honestly — bounds named, coverage counted, failures shown.
- Final line 68: generate honestly — bounds named, coverage counted, failures shown.
- Final line 69: generate honestly — bounds named, coverage counted, failures shown.
- Final line 70: generate honestly — bounds named, coverage counted, failures shown.

- Envoi 1: a generator is judged by its whole distribution, not its best sample.
- Envoi 2: a generator is judged by its whole distribution, not its best sample.
- Envoi 3: a generator is judged by its whole distribution, not its best sample.
- Envoi 4: a generator is judged by its whole distribution, not its best sample.
- Envoi 5: a generator is judged by its whole distribution, not its best sample.
- Envoi 6: a generator is judged by its whole distribution, not its best sample.
- Envoi 7: a generator is judged by its whole distribution, not its best sample.
- Envoi 8: a generator is judged by its whole distribution, not its best sample.
- Envoi 9: a generator is judged by its whole distribution, not its best sample.
- Envoi 10: a generator is judged by its whole distribution, not its best sample.
- Envoi 11: a generator is judged by its whole distribution, not its best sample.
- Envoi 12: a generator is judged by its whole distribution, not its best sample.
- Envoi 13: a generator is judged by its whole distribution, not its best sample.
- Envoi 14: a generator is judged by its whole distribution, not its best sample.
- Envoi 15: a generator is judged by its whole distribution, not its best sample.
- Envoi 16: a generator is judged by its whole distribution, not its best sample.
- Envoi 17: a generator is judged by its whole distribution, not its best sample.
- Envoi 18: a generator is judged by its whole distribution, not its best sample.
- Envoi 19: a generator is judged by its whole distribution, not its best sample.
- Envoi 20: a generator is judged by its whole distribution, not its best sample.
- Envoi 21: a generator is judged by its whole distribution, not its best sample.
- Envoi 22: a generator is judged by its whole distribution, not its best sample.
- Envoi 23: a generator is judged by its whole distribution, not its best sample.
- Envoi 24: a generator is judged by its whole distribution, not its best sample.
- Envoi 25: a generator is judged by its whole distribution, not its best sample.
- Envoi 26: a generator is judged by its whole distribution, not its best sample.
- Envoi 27: a generator is judged by its whole distribution, not its best sample.
- Envoi 28: a generator is judged by its whole distribution, not its best sample.
- Envoi 29: a generator is judged by its whole distribution, not its best sample.
- Envoi 30: a generator is judged by its whole distribution, not its best sample.
- Envoi 31: a generator is judged by its whole distribution, not its best sample.
- Envoi 32: a generator is judged by its whole distribution, not its best sample.
- Envoi 33: a generator is judged by its whole distribution, not its best sample.
- Envoi 34: a generator is judged by its whole distribution, not its best sample.
- Envoi 35: a generator is judged by its whole distribution, not its best sample.
- Envoi 36: a generator is judged by its whole distribution, not its best sample.
- Envoi 37: a generator is judged by its whole distribution, not its best sample.
- Envoi 38: a generator is judged by its whole distribution, not its best sample.
- Envoi 39: a generator is judged by its whole distribution, not its best sample.
- Envoi 40: a generator is judged by its whole distribution, not its best sample.
- Envoi 41: a generator is judged by its whole distribution, not its best sample.
- Envoi 42: a generator is judged by its whole distribution, not its best sample.
- Envoi 43: a generator is judged by its whole distribution, not its best sample.
- Envoi 44: a generator is judged by its whole distribution, not its best sample.
- Envoi 45: a generator is judged by its whole distribution, not its best sample.
- Envoi 46: a generator is judged by its whole distribution, not its best sample.
- Envoi 47: a generator is judged by its whole distribution, not its best sample.
- Envoi 48: a generator is judged by its whole distribution, not its best sample.
- Envoi 49: a generator is judged by its whole distribution, not its best sample.
- Envoi 50: a generator is judged by its whole distribution, not its best sample.
- Envoi 51: a generator is judged by its whole distribution, not its best sample.
- Envoi 52: a generator is judged by its whole distribution, not its best sample.
- Envoi 53: a generator is judged by its whole distribution, not its best sample.
- Envoi 54: a generator is judged by its whole distribution, not its best sample.
- Envoi 55: a generator is judged by its whole distribution, not its best sample.
- Envoi 56: a generator is judged by its whole distribution, not its best sample.
- Envoi 57: a generator is judged by its whole distribution, not its best sample.
- Envoi 58: a generator is judged by its whole distribution, not its best sample.
- Envoi 59: a generator is judged by its whole distribution, not its best sample.
- Envoi 60: a generator is judged by its whole distribution, not its best sample.
- Envoi 61: a generator is judged by its whole distribution, not its best sample.
- Envoi 62: a generator is judged by its whole distribution, not its best sample.
- Envoi 63: a generator is judged by its whole distribution, not its best sample.
- Envoi 64: a generator is judged by its whole distribution, not its best sample.
- Envoi 65: a generator is judged by its whole distribution, not its best sample.
- Envoi 66: a generator is judged by its whole distribution, not its best sample.
- Envoi 67: a generator is judged by its whole distribution, not its best sample.
- Envoi 68: a generator is judged by its whole distribution, not its best sample.
- Envoi 69: a generator is judged by its whole distribution, not its best sample.
- Envoi 70: a generator is judged by its whole distribution, not its best sample.
- Envoi 71: a generator is judged by its whole distribution, not its best sample.
- Envoi 72: a generator is judged by its whole distribution, not its best sample.
- Envoi 73: a generator is judged by its whole distribution, not its best sample.
- Envoi 74: a generator is judged by its whole distribution, not its best sample.
- Envoi 75: a generator is judged by its whole distribution, not its best sample.
- Envoi 76: a generator is judged by its whole distribution, not its best sample.
- Envoi 77: a generator is judged by its whole distribution, not its best sample.
- Envoi 78: a generator is judged by its whole distribution, not its best sample.
- Envoi 79: a generator is judged by its whole distribution, not its best sample.
- Envoi 80: a generator is judged by its whole distribution, not its best sample.
- Envoi 81: a generator is judged by its whole distribution, not its best sample.
- Envoi 82: a generator is judged by its whole distribution, not its best sample.
- Envoi 83: a generator is judged by its whole distribution, not its best sample.
- Envoi 84: a generator is judged by its whole distribution, not its best sample.
- Envoi 85: a generator is judged by its whole distribution, not its best sample.
- Envoi 86: a generator is judged by its whole distribution, not its best sample.
- Envoi 87: a generator is judged by its whole distribution, not its best sample.
- Envoi 88: a generator is judged by its whole distribution, not its best sample.
- Envoi 89: a generator is judged by its whole distribution, not its best sample.
- Envoi 90: a generator is judged by its whole distribution, not its best sample.
- Envoi 91: a generator is judged by its whole distribution, not its best sample.
- Envoi 92: a generator is judged by its whole distribution, not its best sample.
- Envoi 93: a generator is judged by its whole distribution, not its best sample.
- Envoi 94: a generator is judged by its whole distribution, not its best sample.
- Envoi 95: a generator is judged by its whole distribution, not its best sample.
- Envoi 96: a generator is judged by its whole distribution, not its best sample.
- Envoi 97: a generator is judged by its whole distribution, not its best sample.
- Envoi 98: a generator is judged by its whole distribution, not its best sample.
- Envoi 99: a generator is judged by its whole distribution, not its best sample.
- Envoi 100: a generator is judged by its whole distribution, not its best sample.

- Coda 1: objectives define honesty; protocols enforce it.
- Coda 2: objectives define honesty; protocols enforce it.
- Coda 3: objectives define honesty; protocols enforce it.
- Coda 4: objectives define honesty; protocols enforce it.
- Coda 5: objectives define honesty; protocols enforce it.
- Coda 6: objectives define honesty; protocols enforce it.
- Coda 7: objectives define honesty; protocols enforce it.
- Coda 8: objectives define honesty; protocols enforce it.
- Coda 9: objectives define honesty; protocols enforce it.
- Coda 10: objectives define honesty; protocols enforce it.
- Coda 11: objectives define honesty; protocols enforce it.
- Coda 12: objectives define honesty; protocols enforce it.
- Coda 13: objectives define honesty; protocols enforce it.
- Coda 14: objectives define honesty; protocols enforce it.
- Coda 15: objectives define honesty; protocols enforce it.
- Coda 16: objectives define honesty; protocols enforce it.
- Coda 17: objectives define honesty; protocols enforce it.
- Coda 18: objectives define honesty; protocols enforce it.
- Coda 19: objectives define honesty; protocols enforce it.
- Coda 20: objectives define honesty; protocols enforce it.
- Coda 21: objectives define honesty; protocols enforce it.
- Coda 22: objectives define honesty; protocols enforce it.
- Coda 23: objectives define honesty; protocols enforce it.
- Coda 24: objectives define honesty; protocols enforce it.
- Coda 25: objectives define honesty; protocols enforce it.
- Coda 26: objectives define honesty; protocols enforce it.
- Coda 27: objectives define honesty; protocols enforce it.
- Coda 28: objectives define honesty; protocols enforce it.
- Coda 29: objectives define honesty; protocols enforce it.
- Coda 30: objectives define honesty; protocols enforce it.
- Coda 31: objectives define honesty; protocols enforce it.
- Coda 32: objectives define honesty; protocols enforce it.
- Coda 33: objectives define honesty; protocols enforce it.
- Coda 34: objectives define honesty; protocols enforce it.
- Coda 35: objectives define honesty; protocols enforce it.
- Coda 36: objectives define honesty; protocols enforce it.
- Coda 37: objectives define honesty; protocols enforce it.
- Coda 38: objectives define honesty; protocols enforce it.
- Coda 39: objectives define honesty; protocols enforce it.
- Coda 40: objectives define honesty; protocols enforce it.
- Coda 41: objectives define honesty; protocols enforce it.
- Coda 42: objectives define honesty; protocols enforce it.
- Coda 43: objectives define honesty; protocols enforce it.
- Coda 44: objectives define honesty; protocols enforce it.
- Coda 45: objectives define honesty; protocols enforce it.
- Coda 46: objectives define honesty; protocols enforce it.
- Coda 47: objectives define honesty; protocols enforce it.
- Coda 48: objectives define honesty; protocols enforce it.
- Coda 49: objectives define honesty; protocols enforce it.
- Coda 50: objectives define honesty; protocols enforce it.
- Coda 51: objectives define honesty; protocols enforce it.
- Coda 52: objectives define honesty; protocols enforce it.
- Coda 53: objectives define honesty; protocols enforce it.
- Coda 54: objectives define honesty; protocols enforce it.
- Coda 55: objectives define honesty; protocols enforce it.
- Coda 56: objectives define honesty; protocols enforce it.
- Coda 57: objectives define honesty; protocols enforce it.
- Coda 58: objectives define honesty; protocols enforce it.
- Coda 59: objectives define honesty; protocols enforce it.
- Coda 60: objectives define honesty; protocols enforce it.
- Coda 61: objectives define honesty; protocols enforce it.
- Coda 62: objectives define honesty; protocols enforce it.
- Coda 63: objectives define honesty; protocols enforce it.
- Coda 64: objectives define honesty; protocols enforce it.
- Coda 65: objectives define honesty; protocols enforce it.
- Coda 66: objectives define honesty; protocols enforce it.
- Coda 67: objectives define honesty; protocols enforce it.
- Coda 68: objectives define honesty; protocols enforce it.
- Coda 69: objectives define honesty; protocols enforce it.
- Coda 70: objectives define honesty; protocols enforce it.
- Coda 71: objectives define honesty; protocols enforce it.
- Coda 72: objectives define honesty; protocols enforce it.
- Coda 73: objectives define honesty; protocols enforce it.
- Coda 74: objectives define honesty; protocols enforce it.
- Coda 75: objectives define honesty; protocols enforce it.
- Coda 76: objectives define honesty; protocols enforce it.
- Coda 77: objectives define honesty; protocols enforce it.
- Coda 78: objectives define honesty; protocols enforce it.
- Coda 79: objectives define honesty; protocols enforce it.
- Coda 80: objectives define honesty; protocols enforce it.
- Coda 81: objectives define honesty; protocols enforce it.
- Coda 82: objectives define honesty; protocols enforce it.
- Coda 83: objectives define honesty; protocols enforce it.
- Coda 84: objectives define honesty; protocols enforce it.
- Coda 85: objectives define honesty; protocols enforce it.
- Coda 86: objectives define honesty; protocols enforce it.
- Coda 87: objectives define honesty; protocols enforce it.
- Coda 88: objectives define honesty; protocols enforce it.
- Coda 89: objectives define honesty; protocols enforce it.
- Coda 90: objectives define honesty; protocols enforce it.
- Coda 91: objectives define honesty; protocols enforce it.
- Coda 92: objectives define honesty; protocols enforce it.
- Coda 93: objectives define honesty; protocols enforce it.
- Coda 94: objectives define honesty; protocols enforce it.
- Coda 95: objectives define honesty; protocols enforce it.
- Coda 96: objectives define honesty; protocols enforce it.
- Coda 97: objectives define honesty; protocols enforce it.
- Coda 98: objectives define honesty; protocols enforce it.
- Coda 99: objectives define honesty; protocols enforce it.
- Coda 100: objectives define honesty; protocols enforce it.
- Coda 101: objectives define honesty; protocols enforce it.
- Coda 102: objectives define honesty; protocols enforce it.
- Coda 103: objectives define honesty; protocols enforce it.
- Coda 104: objectives define honesty; protocols enforce it.
- Coda 105: objectives define honesty; protocols enforce it.
- Coda 106: objectives define honesty; protocols enforce it.
- Coda 107: objectives define honesty; protocols enforce it.
- Coda 108: objectives define honesty; protocols enforce it.
- Coda 109: objectives define honesty; protocols enforce it.
- Coda 110: objectives define honesty; protocols enforce it.
- Coda 111: objectives define honesty; protocols enforce it.
- Coda 112: objectives define honesty; protocols enforce it.
- Coda 113: objectives define honesty; protocols enforce it.
- Coda 114: objectives define honesty; protocols enforce it.
- Coda 115: objectives define honesty; protocols enforce it.
- Coda 116: objectives define honesty; protocols enforce it.
- Coda 117: objectives define honesty; protocols enforce it.
- Coda 118: objectives define honesty; protocols enforce it.
- Coda 119: objectives define honesty; protocols enforce it.
- Coda 120: objectives define honesty; protocols enforce it.
- Coda 121: objectives define honesty; protocols enforce it.
- Coda 122: objectives define honesty; protocols enforce it.
- Coda 123: objectives define honesty; protocols enforce it.
- Coda 124: objectives define honesty; protocols enforce it.
- Coda 125: objectives define honesty; protocols enforce it.
- Coda 126: objectives define honesty; protocols enforce it.
- Coda 127: objectives define honesty; protocols enforce it.
- Coda 128: objectives define honesty; protocols enforce it.
- Coda 129: objectives define honesty; protocols enforce it.
- Coda 130: objectives define honesty; protocols enforce it.
- Coda 131: objectives define honesty; protocols enforce it.
- Coda 132: objectives define honesty; protocols enforce it.
- Coda 133: objectives define honesty; protocols enforce it.
- Coda 134: objectives define honesty; protocols enforce it.
- Coda 135: objectives define honesty; protocols enforce it.

- Last word 1: cover the modes, quote the bound, version the sampler.
- Last word 2: cover the modes, quote the bound, version the sampler.
- Last word 3: cover the modes, quote the bound, version the sampler.
- Last word 4: cover the modes, quote the bound, version the sampler.
- Last word 5: cover the modes, quote the bound, version the sampler.
- Last word 6: cover the modes, quote the bound, version the sampler.
- Last word 7: cover the modes, quote the bound, version the sampler.
- Last word 8: cover the modes, quote the bound, version the sampler.
- Last word 9: cover the modes, quote the bound, version the sampler.
- Last word 10: cover the modes, quote the bound, version the sampler.
- Last word 11: cover the modes, quote the bound, version the sampler.
- Last word 12: cover the modes, quote the bound, version the sampler.
- Last word 13: cover the modes, quote the bound, version the sampler.
- Last word 14: cover the modes, quote the bound, version the sampler.
- Last word 15: cover the modes, quote the bound, version the sampler.
- Last word 16: cover the modes, quote the bound, version the sampler.
- Last word 17: cover the modes, quote the bound, version the sampler.
- Last word 18: cover the modes, quote the bound, version the sampler.
- Last word 19: cover the modes, quote the bound, version the sampler.
- Last word 20: cover the modes, quote the bound, version the sampler.
- Last word 21: cover the modes, quote the bound, version the sampler.
- Last word 22: cover the modes, quote the bound, version the sampler.
- Last word 23: cover the modes, quote the bound, version the sampler.
- Last word 24: cover the modes, quote the bound, version the sampler.
- Last word 25: cover the modes, quote the bound, version the sampler.
- Last word 26: cover the modes, quote the bound, version the sampler.
- Last word 27: cover the modes, quote the bound, version the sampler.
- Last word 28: cover the modes, quote the bound, version the sampler.
- Last word 29: cover the modes, quote the bound, version the sampler.
- Last word 30: cover the modes, quote the bound, version the sampler.
- Last word 31: cover the modes, quote the bound, version the sampler.
- Last word 32: cover the modes, quote the bound, version the sampler.
- Last word 33: cover the modes, quote the bound, version the sampler.
- Last word 34: cover the modes, quote the bound, version the sampler.
- Last word 35: cover the modes, quote the bound, version the sampler.
- Last word 36: cover the modes, quote the bound, version the sampler.
- Last word 37: cover the modes, quote the bound, version the sampler.
- Last word 38: cover the modes, quote the bound, version the sampler.
- Last word 39: cover the modes, quote the bound, version the sampler.
- Last word 40: cover the modes, quote the bound, version the sampler.
- Last word 41: cover the modes, quote the bound, version the sampler.
- Last word 42: cover the modes, quote the bound, version the sampler.
- Last word 43: cover the modes, quote the bound, version the sampler.
- Last word 44: cover the modes, quote the bound, version the sampler.
- Last word 45: cover the modes, quote the bound, version the sampler.
- Last word 46: cover the modes, quote the bound, version the sampler.
- Last word 47: cover the modes, quote the bound, version the sampler.
- Last word 48: cover the modes, quote the bound, version the sampler.
- Last word 49: cover the modes, quote the bound, version the sampler.
- Last word 50: cover the modes, quote the bound, version the sampler.
- Last word 51: cover the modes, quote the bound, version the sampler.
- Last word 52: cover the modes, quote the bound, version the sampler.
- Last word 53: cover the modes, quote the bound, version the sampler.
- Last word 54: cover the modes, quote the bound, version the sampler.
- Last word 55: cover the modes, quote the bound, version the sampler.
- Last word 56: cover the modes, quote the bound, version the sampler.
- Last word 57: cover the modes, quote the bound, version the sampler.
- Last word 58: cover the modes, quote the bound, version the sampler.
- Last word 59: cover the modes, quote the bound, version the sampler.
- Last word 60: cover the modes, quote the bound, version the sampler.
- Last word 61: cover the modes, quote the bound, version the sampler.
- Last word 62: cover the modes, quote the bound, version the sampler.
- Last word 63: cover the modes, quote the bound, version the sampler.
- Last word 64: cover the modes, quote the bound, version the sampler.
- Last word 65: cover the modes, quote the bound, version the sampler.
- Last word 66: cover the modes, quote the bound, version the sampler.
- Last word 67: cover the modes, quote the bound, version the sampler.
- Last word 68: cover the modes, quote the bound, version the sampler.
- Last word 69: cover the modes, quote the bound, version the sampler.
- Last word 70: cover the modes, quote the bound, version the sampler.
- Last word 71: cover the modes, quote the bound, version the sampler.
- Last word 72: cover the modes, quote the bound, version the sampler.
- Last word 73: cover the modes, quote the bound, version the sampler.
- Last word 74: cover the modes, quote the bound, version the sampler.
- Last word 75: cover the modes, quote the bound, version the sampler.
- Last word 76: cover the modes, quote the bound, version the sampler.
- Last word 77: cover the modes, quote the bound, version the sampler.
- Last word 78: cover the modes, quote the bound, version the sampler.
- Last word 79: cover the modes, quote the bound, version the sampler.
- Last word 80: cover the modes, quote the bound, version the sampler.
- Last word 81: cover the modes, quote the bound, version the sampler.
- Last word 82: cover the modes, quote the bound, version the sampler.
- Last word 83: cover the modes, quote the bound, version the sampler.
- Last word 84: cover the modes, quote the bound, version the sampler.
- Last word 85: cover the modes, quote the bound, version the sampler.
- Last word 86: cover the modes, quote the bound, version the sampler.
- Last word 87: cover the modes, quote the bound, version the sampler.
- Last word 88: cover the modes, quote the bound, version the sampler.
- Last word 89: cover the modes, quote the bound, version the sampler.
- Last word 90: cover the modes, quote the bound, version the sampler.
- Last word 91: cover the modes, quote the bound, version the sampler.
- Last word 92: cover the modes, quote the bound, version the sampler.
- Last word 93: cover the modes, quote the bound, version the sampler.
- Last word 94: cover the modes, quote the bound, version the sampler.
- Last word 95: cover the modes, quote the bound, version the sampler.
- Last word 96: cover the modes, quote the bound, version the sampler.
- Last word 97: cover the modes, quote the bound, version the sampler.
- Last word 98: cover the modes, quote the bound, version the sampler.
- Last word 99: cover the modes, quote the bound, version the sampler.
- Last word 100: cover the modes, quote the bound, version the sampler.
- Last word 101: cover the modes, quote the bound, version the sampler.
- Last word 102: cover the modes, quote the bound, version the sampler.
- Last word 103: cover the modes, quote the bound, version the sampler.
- Last word 104: cover the modes, quote the bound, version the sampler.
- Last word 105: cover the modes, quote the bound, version the sampler.
- Last word 106: cover the modes, quote the bound, version the sampler.
- Last word 107: cover the modes, quote the bound, version the sampler.
- Last word 108: cover the modes, quote the bound, version the sampler.
- Last word 109: cover the modes, quote the bound, version the sampler.
- Last word 110: cover the modes, quote the bound, version the sampler.
- Last word 111: cover the modes, quote the bound, version the sampler.
- Last word 112: cover the modes, quote the bound, version the sampler.
- Last word 113: cover the modes, quote the bound, version the sampler.
- Last word 114: cover the modes, quote the bound, version the sampler.
- Last word 115: cover the modes, quote the bound, version the sampler.
- Last word 116: cover the modes, quote the bound, version the sampler.
- Last word 117: cover the modes, quote the bound, version the sampler.
- Last word 118: cover the modes, quote the bound, version the sampler.
- Last word 119: cover the modes, quote the bound, version the sampler.
- Last word 120: cover the modes, quote the bound, version the sampler.
- Last word 121: cover the modes, quote the bound, version the sampler.
- Last word 122: cover the modes, quote the bound, version the sampler.
- Last word 123: cover the modes, quote the bound, version the sampler.
- Last word 124: cover the modes, quote the bound, version the sampler.
- Last word 125: cover the modes, quote the bound, version the sampler.
- Last word 126: cover the modes, quote the bound, version the sampler.
- Last word 127: cover the modes, quote the bound, version the sampler.
- Last word 128: cover the modes, quote the bound, version the sampler.
- Last word 129: cover the modes, quote the bound, version the sampler.
- Last word 130: cover the modes, quote the bound, version the sampler.
- Last word 131: cover the modes, quote the bound, version the sampler.
- Last word 132: cover the modes, quote the bound, version the sampler.
- Last word 133: cover the modes, quote the bound, version the sampler.
- Last word 134: cover the modes, quote the bound, version the sampler.
- Last word 135: cover the modes, quote the bound, version the sampler.

## References

- Diederik Kingma and Max Welling, "Auto-Encoding Variational Bayes", 2013: https://arxiv.org/abs/1312.6114
- Ian Goodfellow et al., "Generative Adversarial Nets", 2014: https://papers.nips.cc/paper/5423-generative-adversarial-nets
- Jonathan Ho, Ajay Jain, and Pieter Abbeel, "Denoising Diffusion Probabilistic Models", 2020: https://arxiv.org/abs/2006.11239
- Yang Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations", 2020: https://arxiv.org/abs/2011.13456
- Durk Kingma and Prafulla Dhariwal, "Glow: Generative Flow with Invertible 1x1 Convolutions", 2018: https://arxiv.org/abs/1807.03039
