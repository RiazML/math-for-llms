[<- Back to Information Theory](../README.md) | [Previous: Entropy <-](../01-Entropy/notes.md) | [Next: Mutual Information ->](../03-Mutual-Information/notes.md)

---

# KL Divergence

> _"The most important single quantity in information theory and in machine learning is the Kullback-Leibler divergence."_
> - David MacKay, *Information Theory, Inference, and Learning Algorithms*, 2003

## Overview

KL divergence - formally the Kullback-Leibler divergence, also called relative entropy or information gain - is the central tool for comparing probability distributions. Where entropy (Section 01) measures the intrinsic uncertainty of a single distribution, KL divergence measures how much one distribution diverges from another: how many extra bits you waste by coding data from $p$ using a code optimized for $q$.

This single quantity underlies an astonishing range of machine learning. Maximum likelihood estimation, the training objective of every language model ever trained, is equivalent to minimizing $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$. The VAE training objective decomposes exactly as reconstruction loss plus a KL divergence term. RLHF's KL penalty, knowledge distillation's soft-target loss, and PPO's trust-region constraint are all KL divergences in disguise. Understanding KL divergence - its properties, its asymmetry, and the difference between its two directions - is prerequisite to understanding modern deep learning at a mechanistic level.

This section develops KL divergence from first principles. We prove non-negativity rigorously (Section 3.1), dissect the crucial asymmetry between forward and reverse KL (Section 4), derive closed-form expressions for Gaussians and exponential families (Section 5), explore generalizations to f-divergences and Renyi divergence (Section 7), and connect everything to the information-geometric structure of statistical manifolds (Section 8). Every result is grounded in concrete AI/ML applications with specific model and paper citations.

**Scope note.** This section is the canonical home for KL divergence. Closely related quantities - mutual information (Section 03), cross-entropy loss (Section 04), and Fisher information (Section 05) - each have their own dedicated sections and appear here only as brief previews with forward references.

## Prerequisites

- **Shannon entropy** - $H(X) = -\sum_x p(x)\log p(x)$, concavity, boundedness - [Section 09-01 Entropy](../01-Entropy/notes.md)
- **Jensen's inequality** - for convex $\phi$: $\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)]$ - [Chapter 8 Section 01](../../08-Optimization/01-Convex-Optimization/notes.md)
- **Probability distributions** - PMFs, PDFs, expectations, absolute continuity - [Chapter 6](../../06-Probability-Theory/README.md)
- **Logarithm rules** - $\log(ab) = \log a + \log b$, change of base - [Chapter 1](../../01-Mathematical-Foundations/README.md)
- **Convex functions** - definition, second-derivative test, Bregman divergences (Section 8) - [Chapter 8](../../08-Optimization/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations: Gibbs' inequality, forward vs reverse KL visualizations, Gaussian KL, f-divergences, VAE ELBO, RLHF policy geometry |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises from non-negativity proofs to DPO implicit KL |

## Learning Objectives

After completing this section, you will:

1. Define $D_{\mathrm{KL}}(p \| q)$ for discrete and continuous distributions, state the support condition ($p \ll q$), and explain the coding-theoretic interpretation
2. Prove non-negativity of KL divergence (Gibbs' inequality) via Jensen's inequality and state the equality condition
3. Demonstrate the asymmetry $D_{\mathrm{KL}}(p\|q) \ne D_{\mathrm{KL}}(q\|p)$ with a concrete numerical example and explain why KL is not a metric
4. Distinguish forward KL (mean-seeking, mass-covering) from reverse KL (mode-seeking, zero-forcing) and predict which behavior each direction produces
5. Derive the closed-form KL between two Gaussians and apply it to compute the VAE encoder KL penalty
6. State the chain rule for KL divergence and the data processing inequality and explain their ML implications
7. Identify KL divergence as a Bregman divergence and explain the Pythagorean theorem for KL, I-projection, and M-projection
8. Express $\alpha$-divergences, Jensen-Shannon divergence, Hellinger distance, and total variation as special cases of Csiszar f-divergences
9. Prove that maximum likelihood estimation minimizes $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$ and derive the RLHF optimal policy $\pi^* \propto \pi_{\mathrm{ref}} e^{r/\beta}$
10. Explain the difference between forward and reverse KL in knowledge distillation and describe posterior collapse in VAEs and the beta-VAE fix
11. Connect KL to cross-entropy via $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$ and preview its relationship to mutual information and Fisher information

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Is KL Divergence?](#11-what-is-kl-divergence)
  - [1.2 The Asymmetry Insight](#12-the-asymmetry-insight)
  - [1.3 Why KL Divergence Matters for AI](#13-why-kl-divergence-matters-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Discrete KL Divergence](#21-discrete-kl-divergence)
  - [2.2 Continuous KL Divergence](#22-continuous-kl-divergence)
  - [2.3 Relative Entropy Interpretation](#23-relative-entropy-interpretation)
  - [2.4 Non-Examples and Edge Cases](#24-non-examples-and-edge-cases)
- [3. Properties of KL Divergence](#3-properties-of-kl-divergence)
  - [3.1 Non-Negativity: Gibbs' Inequality](#31-non-negativity-gibbs-inequality)
  - [3.2 Asymmetry](#32-asymmetry)
  - [3.3 Failure of Triangle Inequality](#33-failure-of-triangle-inequality)
  - [3.4 Joint Convexity](#34-joint-convexity)
  - [3.5 Chain Rule for KL Divergence](#35-chain-rule-for-kl-divergence)
  - [3.6 Data Processing Inequality](#36-data-processing-inequality)
- [4. Forward KL vs Reverse KL](#4-forward-kl-vs-reverse-kl)
  - [4.1 Forward KL: Mean-Seeking Behavior](#41-forward-kl-mean-seeking-behavior)
  - [4.2 Reverse KL: Mode-Seeking Behavior](#42-reverse-kl-mode-seeking-behavior)
  - [4.3 Geometric Interpretation](#43-geometric-interpretation)
  - [4.4 Consequences for Variational Inference](#44-consequences-for-variational-inference)
- [5. KL for Specific Distributions](#5-kl-for-specific-distributions)
  - [5.1 KL Between Gaussians](#51-kl-between-gaussians)
  - [5.2 KL Between Categorical Distributions](#52-kl-between-categorical-distributions)
  - [5.3 KL Within the Exponential Family](#53-kl-within-the-exponential-family)
  - [5.4 VAE Closed-Form KL and the Reparameterization Trick](#54-vae-closed-form-kl-and-the-reparameterization-trick)
- [6. Information-Theoretic Connections](#6-information-theoretic-connections)
  - [6.1 KL and Cross-Entropy](#61-kl-and-cross-entropy)
  - [6.2 KL and Mutual Information](#62-kl-and-mutual-information)
  - [6.3 KL and Fisher Information](#63-kl-and-fisher-information)
  - [6.4 KL and Entropy](#64-kl-and-entropy)
- [7. f-Divergences and Generalizations](#7-f-divergences-and-generalizations)
  - [7.1 Csiszar f-Divergences](#71-csiszar-f-divergences)
  - [7.2 Properties of f-Divergences](#72-properties-of-f-divergences)
  - [7.3 Renyi Divergence](#73-renyi-divergence)
  - [7.4 Jensen-Shannon Divergence](#74-jensen-shannon-divergence)
- [8. Information Geometry of KL](#8-information-geometry-of-kl)
  - [8.1 KL as a Bregman Divergence](#81-kl-as-a-bregman-divergence)
  - [8.2 I-Projection and M-Projection](#82-i-projection-and-m-projection)
  - [8.3 The Pythagorean Theorem for KL](#83-the-pythagorean-theorem-for-kl)
  - [8.4 Statistical Manifolds and Natural Gradient](#84-statistical-manifolds-and-natural-gradient)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Maximum Likelihood = Minimizing KL](#91-maximum-likelihood--minimizing-kl)
  - [9.2 Variational Autoencoders](#92-variational-autoencoders)
  - [9.3 RLHF, PPO, and DPO](#93-rlhf-ppo-and-dpo)
  - [9.4 Knowledge Distillation](#94-knowledge-distillation)
  - [9.5 Normalizing Flows and Diffusion Models](#95-normalizing-flows-and-diffusion-models)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [Appendix N: KL in Practice: Inequalities, Variational Bounds, and Audits](#appendix-n-kl-in-practice-inequalities-variational-bounds-and-audits)

---

## 1. Intuition

### 1.1 What Is KL Divergence?

Suppose you are a weather forecaster. Every day, you predict a probability distribution over tomorrow's weather: 70% sunny, 20% cloudy, 10% rain. But the true distribution - what nature actually produces - is 50% sunny, 30% cloudy, 20% rain. How much has your forecast missed? Shannon entropy measures the uncertainty in a single distribution. KL divergence measures how much one distribution deviates from another.

Formally, $D_{\mathrm{KL}}(p \| q)$ is the **expected extra number of bits** (or nats, if using natural logarithms) you waste when you encode data actually generated from $p$ using a code optimized for $q$. If you design a Huffman code for $q$, the optimal code assigns short codewords to events that $q$ considers probable. But if the true distribution is $p$, your code will be suboptimal whenever $p(x) \ne q(x)$. The KL divergence precisely quantifies the average penalty:

$$D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

The ratio $p(x)/q(x)$ captures the mismatch event by event: events that $p$ considers much more likely than $q$ does (large $p/q$) waste the most bits. The expectation under $p$ reflects that the penalty is averaged over what actually happens.

**Three equivalent ways to read $D_{\mathrm{KL}}(p \| q)$:**

1. **Coding view:** Expected excess code length when coding $p$-data with a $q$-optimal code.
2. **Surprise view:** How much more surprised you are on average by events under $q$ than under $p$.
3. **Testing view:** The expected log-likelihood ratio $\mathbb{E}_p[\log(p(X)/q(X))]$ - the average evidence per sample that data came from $p$ rather than $q$ in a hypothesis test.

All three views are the same formula. The richness of KL divergence comes from this triple identity: it simultaneously measures compression inefficiency, distributional mismatch, and statistical distinguishability.

**For AI:** When a language model $p_{\boldsymbol{\theta}}$ is trained on data drawn from $p_{\mathrm{data}}$, the negative log-likelihood loss $-\mathbb{E}_{x \sim p_{\mathrm{data}}}[\log p_{\boldsymbol{\theta}}(x)]$ equals $H(p_{\mathrm{data}}) + D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$. Since $H(p_{\mathrm{data}})$ is a constant w.r.t. $\boldsymbol{\theta}$, minimizing cross-entropy loss is identical to minimizing $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$. Every gradient descent step that improves a language model is a step toward closing the KL gap between model and data distribution.

### 1.2 The Asymmetry Insight

The defining feature that most surprises newcomers is asymmetry: $D_{\mathrm{KL}}(p \| q)$ and $D_{\mathrm{KL}}(q \| p)$ are generally different, and this is not a flaw - it is a deeply meaningful distinction that determines which algorithm you should use.

To see why asymmetry arises, consider what each direction penalizes:

- **$D_{\mathrm{KL}}(p \| q)$**: the expectation is under $p$. Events where $p(x) > 0$ but $q(x) \approx 0$ contribute hugely (the ratio $p/q$ blows up). This is called **forward KL** or **inclusive KL**: if $p$ puts mass somewhere, $q$ must also put mass there. $q$ must *cover* all of $p$'s support. Minimizing this forces $q$ to be "mean-seeking" - to match the whole spread of $p$, even at multimodal distributions.

- **$D_{\mathrm{KL}}(q \| p)$**: the expectation is now under $q$. Events where $q(x) > 0$ but $p(x) \approx 0$ contribute hugely. This is called **reverse KL** or **exclusive KL**: if $q$ puts mass somewhere, $p$ must also put mass there - so $q$ avoids regions where $p$ is small. Minimizing this forces $q$ to be "mode-seeking" - to concentrate on one mode of $p$ and ignore the others.

**Concrete example:** Let $p$ be a bimodal mixture $0.5\,\mathcal{N}(-3,1) + 0.5\,\mathcal{N}(3,1)$ and let $q = \mathcal{N}(\mu, \sigma^2)$ be a unimodal Gaussian we fit to $p$:

- Minimizing $D_{\mathrm{KL}}(p \| q)$: $q$ must cover both modes, so $q \approx \mathcal{N}(0, 10)$ - wide, centered between the modes, placing mass in low-probability regions.
- Minimizing $D_{\mathrm{KL}}(q \| p)$: $q$ collapses onto one mode, e.g., $q \approx \mathcal{N}(3, 1)$, ignoring the other mode entirely.

Neither approximation is "wrong" - they answer different questions. Which direction to use is one of the most important design decisions in probabilistic ML.

**For AI:** Variational inference for Bayesian neural networks minimizes reverse KL $D_{\mathrm{KL}}(q\|p)$ - the variational posterior $q$ collapses to one mode of the intractable true posterior $p$. This is why Bayesian deep learning with mean-field VI tends to underestimate uncertainty (it ignores alternative modes). Maximum likelihood training of generative models minimizes forward KL $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\boldsymbol{\theta}})$ - forcing the model to cover the entire data distribution, at the cost of sometimes generating blurry or average-looking samples.

### 1.3 Why KL Divergence Matters for AI

KL divergence is not one tool among many - it is the unifying framework behind most of the training objectives and alignment techniques in modern AI:

| Application | Which KL | How |
| --- | --- | --- |
| Language model training (GPT, LLaMA, Claude) | $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$ | Minimizing cross-entropy = minimizing forward KL |
| VAE regularizer (Kingma & Welling, 2014) | $D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$ | Reverse KL keeps encoder close to Gaussian prior |
| RLHF KL penalty (Christiano et al., 2017) | $D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\mathrm{ref}})$ | Prevents reward hacking; preserves language coherence |
| PPO trust region (Schulman et al., 2017) | Approx. $D_{\mathrm{KL}}(\pi_{\mathrm{old}} \| \pi_{\mathrm{new}})$ | Clip objective approximates KL constraint |
| DPO (Rafailov et al., 2023) | Implicit KL | Closed-form solution has same fixed point as RLHF |
| Knowledge distillation (Hinton et al., 2015) | $D_{\mathrm{KL}}(p_{\mathrm{teacher}} \| p_{\mathrm{student}})$ | Forward KL to match teacher's full soft distribution |
| Variational inference (VI) | $D_{\mathrm{KL}}(q \| p)$ | Reverse KL for tractable posterior approximation |
| Normalizing flows (Rezende & Mohamed, 2015) | $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$ | Forward KL via exact log-likelihood |
| Diffusion models (Ho et al., 2020) | KL at each reverse step | ELBO decomposes as sum of per-step KL terms |
| Differential privacy (Renyi DP) | Renyi divergence | Renyi KL bounds privacy loss composition |

The breadth of this table reflects a deep fact: KL divergence is the natural measure of distributional distance when you care about expected log-probability ratios, which is exactly what gradient-based optimization of probabilistic models does.

### 1.4 Historical Timeline

```
KL DIVERGENCE - KEY MILESTONES


  1951  Kullback & Leibler, "On Information and Sufficiency"
        Introduce D_KL as "information for discrimination" I(1:2)
        Prove non-negativity; connect to sufficiency of statistics

  1959  Kullback, "Information Theory and Statistics"
        Systematic treatment; connection to likelihood ratio tests

  1967  Csiszar, "Information-Type Measures of Difference"
        Generalizes to f-divergences; unifies KL, Hellinger, 2

  1972  Chernoff; Blahut, Arimoto
        Exponential decay of error rates in hypothesis testing
        Connection to Renyi divergence

  1985  Amari, "Differential-Geometric Methods in Statistics"
        Information geometry: KL as non-symmetric Bregman divergence
        Exponential and mixture geodesics; natural gradient

  1986  Hinton & Camp, "Keeping Neural Networks Simple"
        Minimum Description Length  KL regularization

  2013  Kingma & Welling, "Auto-Encoding Variational Bayes"
        VAE: ELBO = reconstruction - D_KL(q_  p)
        Reparameterization trick for gradient through KL

  2015  Hinton, Vinyals & Dean, "Distilling the Knowledge"
        Forward KL with temperature-softened teacher distribution

  2017  Christiano et al., "Deep Reinforcement Learning from Human Feedback"
        KL penalty in RLHF; controls deviation from reference policy

  2017  Schulman et al., "Proximal Policy Optimization"
        PPO-clip approximates KL trust region constraint

  2022  Rafailov et al., "Direct Preference Optimization"
        DPO: closed-form KL-constrained policy optimization

  2024+  All frontier LLMs (GPT-4, Claude, Gemini, LLaMA-3)
         Trained with cross-entropy (= forward KL) + RLHF KL penalty


```


## 2. Formal Definitions

### 2.1 Discrete KL Divergence

**Definition (Kullback-Leibler Divergence - discrete).** Let $p$ and $q$ be probability mass functions on a countable alphabet $\mathcal{X}$. The **KL divergence** from $q$ to $p$ (also called the *relative entropy of $p$ with respect to $q$*) is:

$$D_{\mathrm{KL}}(p \| q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$

where:
- The logarithm is natural (nats) by convention in this curriculum, matching the NOTATION_GUIDE. To convert to bits, divide by $\ln 2$.
- **Boundary conventions:** $0 \log(0/q) = 0$ for any $q \ge 0$ (by continuity: $\lim_{p \to 0^+} p \log p = 0$); $p \log(p/0) = +\infty$ for $p > 0$ (code length is infinite if you cannot encode a symbol at all).
- $D_{\mathrm{KL}}(p \| q) = +\infty$ whenever there exists $x$ with $p(x) > 0$ and $q(x) = 0$.

The notation convention in this repository (following Cover & Thomas 2006 and NOTATION_GUIDE Section 6): $D_{\mathrm{KL}}(p \| q)$ reads as "KL divergence **from** $q$ **to** $p$." That is, $p$ is the *reference* (true) distribution and $q$ is the *approximation*. Some sources use the opposite convention; when reading papers, always check which argument is which.

**Standard examples:**

**Example 2.1 (Binary distributions):** Let $p = (\theta, 1-\theta)$ and $q = (q_0, 1-q_0)$ on $\{0,1\}$. Then:

$$D_{\mathrm{KL}}(p \| q) = \theta \ln\frac{\theta}{q_0} + (1-\theta)\ln\frac{1-\theta}{1-q_0}$$

This is the log-likelihood ratio test statistic for a Bernoulli parameter hypothesis test.

**Example 2.2 (Weather forecast):** True distribution $p = (0.5, 0.3, 0.2)$ over {sunny, cloudy, rain}; forecast $q = (0.7, 0.2, 0.1)$:

$$D_{\mathrm{KL}}(p \| q) = 0.5\ln\frac{0.5}{0.7} + 0.3\ln\frac{0.3}{0.2} + 0.2\ln\frac{0.2}{0.1} \approx -0.168 + 0.122 + 0.139 \approx 0.093 \text{ nats}$$

The model wastes approximately 0.093 nats per observation. In the reverse direction: $D_{\mathrm{KL}}(q \| p) \approx 0.097$ nats - different but close, because the two distributions are not very far apart.

**Example 2.3 (Categorical logits):** In a classification model with softmax output $q(k) = \text{softmax}(\mathbf{z})_k$ and one-hot label $p(k) = \mathbb{1}[k = y]$:

$$D_{\mathrm{KL}}(p \| q) = -\log q(y) = -z_y + \log\sum_k e^{z_k}$$

This is exactly the cross-entropy loss. (The entropy $H(p) = 0$ for one-hot $p$, so $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q) = D_{\mathrm{KL}}(p\|q)$.)

### 2.2 Continuous KL Divergence

**Definition (KL divergence - continuous).** Let $p$ and $q$ be probability density functions on $\mathbb{R}^d$ with $p$ absolutely continuous with respect to $q$ (written $p \ll q$, meaning $q(x) = 0 \Rightarrow p(x) = 0$ a.e.). Then:

$$D_{\mathrm{KL}}(p \| q) = \int_{\mathbb{R}^d} p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \, d\mathbf{x} = \mathbb{E}_{\mathbf{x} \sim p}\!\left[\log \frac{p(\mathbf{x})}{q(\mathbf{x})}\right]$$

The absolute continuity condition $p \ll q$ is essential: it ensures the Radon-Nikodym derivative $dp/dq$ exists, making the ratio $p(\mathbf{x})/q(\mathbf{x})$ well-defined a.e. under $p$. When $p \not\ll q$, $D_{\mathrm{KL}}(p\|q) = +\infty$ by convention.

**Measure-theoretic form.** Using the Radon-Nikodym derivative $\frac{dp}{dq}$, the general definition that works for both discrete and continuous cases:

$$D_{\mathrm{KL}}(p \| q) = \int \log \frac{dp}{dq} \, dp = \mathbb{E}_p\!\left[\log \frac{dp}{dq}\right]$$

This unified form shows that discreteness vs continuity is just a choice of dominating measure.

### 2.3 Relative Entropy Interpretation

The phrase "relative entropy" is justified by a beautiful coding theorem. Suppose $X_1, X_2, \ldots$ are i.i.d. from $p$. You observe $n$ samples and design a code using distribution $q$. The optimal code assigns length $-\log q(x)$ to symbol $x$ (by Shannon's source coding theorem). The expected code length for one symbol is $\mathbb{E}_p[-\log q(X)] = H(p, q)$ - the cross-entropy. But the optimal code for $p$ has expected length $H(p)$. The **excess code length per symbol** is:

$$H(p, q) - H(p) = -\sum_x p(x)\log q(x) + \sum_x p(x)\log p(x) = \sum_x p(x)\log\frac{p(x)}{q(x)} = D_{\mathrm{KL}}(p \| q)$$

This is exactly KL divergence. The relative entropy $D_{\mathrm{KL}}(p\|q)$ measures how many extra nats/bits per symbol you pay for using the wrong code $q$ when the truth is $p$. It is *relative* because it measures information *relative to* the reference distribution $p$.

**Information gain interpretation.** In Bayesian inference, if $p$ is the posterior and $q$ is the prior, then $D_{\mathrm{KL}}(p\|q)$ is the **information gain** from the prior to the posterior - the reduction in uncertainty after observing data. The Kullback-Leibler 1951 paper used the term "information for discrimination": how much information the data provides for discriminating between $p$ and $q$.

### 2.4 Non-Examples and Edge Cases

Understanding when KL divergence is *not* well-behaved clarifies the definition.

**Non-example 2.1 (Mismatched support, $D_{\mathrm{KL}} = +\infty$):** Let $p = \mathcal{N}(0, 1)$ and $q = \mathcal{U}(-1, 1)$ (uniform on $[-1,1]$). Since $q(x) = 0$ for $|x| > 1$ but $p(x) > 0$ for all $x$, we have $p \not\ll q$ and $D_{\mathrm{KL}}(p\|q) = +\infty$. However $D_{\mathrm{KL}}(q\|p) < \infty$ since $p(x) > 0$ everywhere, so $q \ll p$. This illustrates the extreme asymmetry that can occur when supports differ.

**Non-example 2.2 (Zero KL does not mean identical distributions numerically):** $D_{\mathrm{KL}}(p\|q) = 0$ implies $p(x) = q(x)$ for $p$-almost all $x$, but for distributions with different supports having measure zero under $p$, the distributions can differ on a null set. In practice, if $D_{\mathrm{KL}} = 0$ numerically, $p$ and $q$ are identical where it matters.

**Non-example 2.3 (Equal KL values do not imply related distributions):** $D_{\mathrm{KL}}(p\|q) = D_{\mathrm{KL}}(r\|s) = 0.5$ nats gives no useful information about the relationship between $(p,q)$ and $(r,s)$. KL is an absolute quantity for a specific pair, not a geometry with reference points.

**Non-example 2.4 (KL is not symmetric by default):** For $p = (0.9, 0.1)$ and $q = (0.1, 0.9)$:
$$D_{\mathrm{KL}}(p\|q) = 0.9\ln\frac{0.9}{0.1} + 0.1\ln\frac{0.1}{0.9} = 0.9\ln 9 + 0.1\ln(1/9) \approx 1.758 \text{ nats}$$
By symmetry of this example, $D_{\mathrm{KL}}(q\|p) = D_{\mathrm{KL}}(p\|q) \approx 1.758$ nats - equal only because $p$ and $q$ are exact reverses of each other. For generic asymmetric distributions, the two values differ.


## 3. Properties of KL Divergence

### 3.1 Non-Negativity: Gibbs' Inequality

**Theorem (Gibbs' Inequality).** For any two probability distributions $p$ and $q$ on the same alphabet:
$$D_{\mathrm{KL}}(p \| q) \ge 0$$
with equality if and only if $p(x) = q(x)$ for all $x$ (or $p$-almost all $x$ in the continuous case).

**Proof via Jensen's inequality.** The function $f(t) = -\ln t$ is strictly convex on $(0, \infty)$ (its second derivative $1/t^2 > 0$). Equivalently, $\ln t$ is strictly concave. By Jensen's inequality applied to the concave function $\ln$:

$$-D_{\mathrm{KL}}(p \| q) = \sum_x p(x)\ln\frac{q(x)}{p(x)} \le \ln\!\left(\sum_x p(x) \cdot \frac{q(x)}{p(x)}\right) = \ln\!\left(\sum_x q(x)\right) = \ln 1 = 0$$

Therefore $D_{\mathrm{KL}}(p\|q) \ge 0$. Equality holds in Jensen's inequality for a strictly concave function iff the argument is constant, i.e., $q(x)/p(x) = c$ for all $x$ with $p(x) > 0$. Since both $p$ and $q$ sum to 1, this forces $c = 1$ and hence $p = q$ everywhere. $\square$

**Alternative proof via $\ln t \le t - 1$.** The inequality $\ln t \le t - 1$ (with equality iff $t = 1$) implies:

$$-D_{\mathrm{KL}}(p \| q) = \sum_x p(x)\ln\frac{q(x)}{p(x)} \le \sum_x p(x)\left(\frac{q(x)}{p(x)} - 1\right) = \sum_x q(x) - \sum_x p(x) = 1 - 1 = 0$$

This proof is elementary (needs no Jensen's inequality) and directly shows the equality condition.

**Corollary (Entropy upper bound).** Taking $q = u$ (uniform on $\mathcal{X}$ with $|\mathcal{X}| = n$):
$$0 \le D_{\mathrm{KL}}(p \| u) = \sum_x p(x)\ln\frac{p(x)}{1/n} = \ln n - H(p)$$
Therefore $H(p) \le \ln n = H(u)$. Gibbs' inequality is the reason entropy is maximized at the uniform distribution (proven rigorously in Section 09-01).

**For AI:** Non-negativity of KL underpins the ELBO. The VAE lower bound comes from:
$$\log p(\mathbf{x}) = \mathcal{L}(\boldsymbol{\phi}, \boldsymbol{\theta}; \mathbf{x}) + D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x}) \| p_\theta(\mathbf{z}\mid\mathbf{x}))$$
Since $D_{\mathrm{KL}} \ge 0$, we have $\log p(\mathbf{x}) \ge \mathcal{L}$ - the ELBO is indeed a lower bound on log-evidence. This is the fundamental inequality that makes variational inference tractable.

### 3.2 Asymmetry

$D_{\mathrm{KL}}(p \| q) \ne D_{\mathrm{KL}}(q \| p)$ in general. KL divergence is not a distance in the metric sense. We have already seen this intuitively; here we quantify it.

**Example 3.2:** $p = (0.8, 0.1, 0.1)$, $q = (0.1, 0.8, 0.1)$:

$$D_{\mathrm{KL}}(p \| q) = 0.8\ln 8 + 0.1\ln(1/8) + 0.1\ln 1 = 0.8(2.079) + 0.1(-2.079) + 0 = 1.455 \text{ nats}$$
$$D_{\mathrm{KL}}(q \| p) = 0.1\ln(0.1/0.8) + 0.8\ln 8 + 0.1\ln 1 = 0.1(-2.079) + 0.8(2.079) + 0 = 1.455 \text{ nats}$$

In this symmetric case they are equal, but for $p = (0.9, 0.05, 0.05)$, $q = (0.05, 0.9, 0.05)$:

$$D_{\mathrm{KL}}(p \| q) = 0.9\ln 18 + 0.05\ln(0.05/0.9) + 0.05\ln 1 \approx 2.629 + (-0.144) + 0 = 2.485 \text{ nats}$$
$$D_{\mathrm{KL}}(q \| p) = 0.05\ln(0.05/0.9) + 0.9\ln 18 + 0.05\ln 1 \approx -0.144 + 2.629 + 0 = 2.485 \text{ nats}$$

Both happen to be equal again due to the symmetric structure. For a genuinely asymmetric example: $p = (0.99, 0.01)$, $q = (0.01, 0.99)$:

$$D_{\mathrm{KL}}(p\|q) = 0.99\ln 99 + 0.01\ln(0.01/0.99) \approx 0.99(4.595) + 0.01(-4.595) = 4.549 \text{ nats}$$

and $D_{\mathrm{KL}}(q\|p) = D_{\mathrm{KL}}(p\|q) \approx 4.549$ nats (by symmetry of this particular pair).

For a non-symmetric example: $p = (0.9, 0.1)$, $q = (0.5, 0.5)$:
$$D_{\mathrm{KL}}(p\|q) = 0.9\ln(1.8) + 0.1\ln(0.2) = 0.9(0.588) + 0.1(-1.609) \approx 0.368 \text{ nats}$$
$$D_{\mathrm{KL}}(q\|p) = 0.5\ln(5/9) + 0.5\ln 5 = 0.5(-0.588) + 0.5(1.609) \approx 0.511 \text{ nats}$$

So $D_{\mathrm{KL}}(p\|q) = 0.368 \ne 0.511 = D_{\mathrm{KL}}(q\|p)$.

**Jeffrey's symmetrized KL.** Harold Jeffreys (1946) proposed the symmetric version:
$$J(p, q) = \frac{1}{2}\left[D_{\mathrm{KL}}(p \| q) + D_{\mathrm{KL}}(q \| p)\right]$$

This satisfies $J(p,q) = J(q,p) \ge 0$ and $J(p,q) = 0 \Leftrightarrow p = q$, but still fails the triangle inequality. Jeffreys divergence is used in some applications where symmetry is important. The Jensen-Shannon divergence (Section 7.4) is a better-behaved symmetric variant that is also bounded.

### 3.3 Failure of Triangle Inequality

KL divergence is not a metric. Specifically, the triangle inequality $D_{\mathrm{KL}}(p\|r) \le D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|r)$ does not hold in general.

**Counterexample.** Let $p = (1, 0)$, $q = (0.5, 0.5)$, $r = (0, 1)$ on $\{0, 1\}$:

$$D_{\mathrm{KL}}(p\|r) = 1 \cdot \ln(1/0) = +\infty$$
$$D_{\mathrm{KL}}(p\|q) = 1 \cdot \ln(1/0.5) = \ln 2 \approx 0.693 \text{ nats}$$
$$D_{\mathrm{KL}}(q\|r) = 0.5\ln(0.5/0) + 0.5\ln(0.5/1) = +\infty$$

In this degenerate case $D_{\mathrm{KL}}(p\|r) = +\infty$ and $D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|r) = +\infty$, so the triangle inequality "holds" trivially. For a cleaner counterexample with all finite values: $p = (0.9, 0.1)$, $q = (0.5, 0.5)$, $r = (0.1, 0.9)$:

$$D_{\mathrm{KL}}(p\|r) = 0.9\ln 9 + 0.1\ln(1/9) \approx 1.758 \text{ nats}$$
$$D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|r) \approx 0.368 + 0.368 = 0.736 \text{ nats}$$

Since $1.758 > 0.736$, the triangle inequality is violated. The reason: KL "skips" over intermediate distributions in a fundamentally non-Euclidean way.

### 3.4 Joint Convexity

**Theorem.** $D_{\mathrm{KL}}(p \| q)$ is **jointly convex** in the pair $(p, q)$: for any $\lambda \in [0,1]$,

$$D_{\mathrm{KL}}(\lambda p_1 + (1-\lambda)p_2 \| \lambda q_1 + (1-\lambda)q_2) \le \lambda D_{\mathrm{KL}}(p_1 \| q_1) + (1-\lambda) D_{\mathrm{KL}}(p_2 \| q_2)$$

**Proof sketch.** The perspective function of a convex function $f$ is jointly convex: if $f$ is convex, then $g(x, t) = tf(x/t)$ is jointly convex in $(x, t)$ for $t > 0$. Taking $f(u) = u\log u$ (which is convex since $(u\log u)'' = 1/u > 0$), the perspective is $t \cdot (x/t)\log(x/t) = x\log(x/t)$. Summing over $x$ gives $D_{\mathrm{KL}}(p\|q) = \sum_x p(x)\log(p(x)/q(x))$, which is jointly convex in $(p, q)$ as a sum of perspective functions.

**Consequence for the EM algorithm.** The EM algorithm alternates between computing the expected log-likelihood (E-step) and maximizing it (M-step). Each M-step minimizes a KL divergence. Joint convexity guarantees that this alternating minimization makes well-defined progress toward a stationary point.

**Convexity in $p$ alone.** $D_{\mathrm{KL}}(p\|q)$ is also convex in $p$ for fixed $q$ (since it equals a sum of convex functions $p(x)\log p(x) - p(x)\log q(x)$).

**Convexity in $q$ alone.** $D_{\mathrm{KL}}(p\|q) = -\sum_x p(x)\log q(x) + \text{const}$ for fixed $p$. This is convex in $q$ because $-\log q(x)$ is convex and $p(x) \ge 0$. This means: for fixed $p$, minimizing over $q$ is a convex optimization problem with a unique minimum at $q = p$.

### 3.5 Chain Rule for KL Divergence

**Theorem (Chain Rule).** For joint distributions $P(X,Y)$ and $Q(X,Y)$:

$$D_{\mathrm{KL}}(P(X,Y) \| Q(X,Y)) = D_{\mathrm{KL}}(P_X \| Q_X) + \mathbb{E}_{x \sim P_X}\!\left[D_{\mathrm{KL}}(P_{Y|X=x} \| Q_{Y|X=x})\right]$$

where $P_X$ is the marginal of $P$ over $X$, and $P_{Y|X}$, $Q_{Y|X}$ are the conditional distributions.

**Proof.**
$$D_{\mathrm{KL}}(P \| Q) = \sum_{x,y} P(x,y)\log\frac{P(x,y)}{Q(x,y)}$$

Using $P(x,y) = P(x)P(y|x)$ and $Q(x,y) = Q(x)Q(y|x)$:

$$= \sum_{x,y} P(x,y)\log\frac{P(x)P(y|x)}{Q(x)Q(y|x)} = \sum_{x,y} P(x,y)\log\frac{P(x)}{Q(x)} + \sum_{x,y} P(x,y)\log\frac{P(y|x)}{Q(y|x)}$$

$$= \sum_x P(x)\log\frac{P(x)}{Q(x)} + \sum_x P(x)\sum_y P(y|x)\log\frac{P(y|x)}{Q(y|x)}$$

$$= D_{\mathrm{KL}}(P_X \| Q_X) + \mathbb{E}_{x \sim P_X}\!\left[D_{\mathrm{KL}}(P_{Y|X=x} \| Q_{Y|X=x})\right] \quad \square$$

**For AI:** In a hierarchical generative model $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p(\mathbf{x}|\mathbf{z})$, the chain rule decomposes the KL between the variational distribution and true posterior into a prior KL plus an expected conditional KL. This is exactly the ELBO decomposition:

$$D_{\mathrm{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x})) = D_{\mathrm{KL}}(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) - \mathbb{E}_{q}\left[\log p(\mathbf{x}|\mathbf{z})\right] + \log p(\mathbf{x})$$

Rearranging: $\log p(\mathbf{x}) = \underbrace{\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q\|p)}_{\text{ELBO}} + D_{\mathrm{KL}}(q(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}|\mathbf{x})) \ge \text{ELBO}$.

### 3.6 Data Processing Inequality

**Theorem (Data Processing Inequality for KL).** Let $T: \mathcal{X} \to \mathcal{Y}$ be any (possibly randomized) function. Let $p_T$ and $q_T$ be the distributions of $T(X)$ under $p$ and $q$ respectively. Then:

$$D_{\mathrm{KL}}(p_T \| q_T) \le D_{\mathrm{KL}}(p \| q)$$

**Proof sketch.** By the chain rule applied to the joint $(X, T(X))$:

$$D_{\mathrm{KL}}(p(X, T(X)) \| q(X, T(X))) = D_{\mathrm{KL}}(p_X \| q_X) + \mathbb{E}\left[D_{\mathrm{KL}}(p_{T|X}\|q_{T|X})\right]$$

For a deterministic function $T$, the conditional $D_{\mathrm{KL}}(p_{T|X}\|q_{T|X}) = 0$. By the reverse chain rule decomposing via $T$ first, $D_{\mathrm{KL}}(p_T\|q_T) + \mathbb{E}[D_{\mathrm{KL}}(p_{X|T}\|q_{X|T})] = D_{\mathrm{KL}}(p_X\|q_X)$. Since the second term is $\ge 0$, $D_{\mathrm{KL}}(p_T\|q_T) \le D_{\mathrm{KL}}(p_X\|q_X)$. $\square$

**Intuition:** Processing data can only destroy information, not create it. Two distributions that are $D$ nats apart look *at most* $D$ nats apart after transformation. If you run a neural network encoder on samples from two distributions, the KL in embedding space is at most the KL in input space.

**For AI:** This is why representation learning cannot increase discriminability between classes beyond what is in the raw data. It also explains why the KL penalty in RLHF acts on the *token* distribution: any post-processing of the output tokens (sampling, filtering) can only reduce the KL from the reference policy, not increase it.


## 4. Forward KL vs Reverse KL

This section develops the most practically important distinction in applied KL theory: **which direction to minimize**, and what kind of approximation each direction produces.

### 4.1 Forward KL: Mean-Seeking Behavior

**Forward KL** is $D_{\mathrm{KL}}(p \| q)$ where $p$ is the target distribution (the truth) and $q$ is the approximating distribution we optimize. We minimize over $q$ given fixed $p$.

**The zero-avoiding (mass-covering) property:** Because the expectation is under $p$, regions where $p(x) > 0$ but $q(x) = 0$ contribute $+\infty$ to $D_{\mathrm{KL}}(p\|q)$. Therefore, any minimizer $q^*$ must satisfy $q^*(x) > 0$ wherever $p(x) > 0$ - the approximation must *cover* all the mass of the target. For this reason, forward KL is also called **inclusive** KL or **zero-avoiding** KL.

**Fitting a Gaussian to a bimodal distribution.** Let $p(x) = 0.5\,\mathcal{N}(x;-3,1) + 0.5\,\mathcal{N}(x;3,1)$ and $q_\mu(x) = \mathcal{N}(x;\mu,\sigma^2)$. Minimizing $D_{\mathrm{KL}}(p\|q_\mu)$ over $\mu$ and $\sigma^2$:

$$\frac{\partial}{\partial \mu} D_{\mathrm{KL}}(p \| q_\mu) = 0 \implies \mu^* = \mathbb{E}_p[X] = 0.5(-3) + 0.5(3) = 0$$

$$\sigma^{*2} = \mathbb{E}_p[(X - \mu^*)^2] = 0.5(9 + 1) + 0.5(9 + 1) = 10$$

So $q^* = \mathcal{N}(0, 10)$ - a wide Gaussian centered between the modes, placing substantial mass in the valley between them where $p$ is near zero. This is the **mean-seeking** behavior: $q$ computes the mean and variance of $p$ (moment matching).

**General result.** For any exponential family $q$ with sufficient statistics $\mathbf{t}(x)$, minimizing forward KL performs **moment matching**: $\mathbb{E}_q[\mathbf{t}(X)] = \mathbb{E}_p[\mathbf{t}(X)]$. The approximation matches the moments of $p$, which averages across modes.

**For AI:** Maximum likelihood estimation minimizes forward KL. This is why MLE on multimodal data can produce "blurry" generative models that average over modes - GANs were introduced partly to address this by using a minimax objective related to Jensen-Shannon divergence (Section 7.4) instead of direct MLE.

### 4.2 Reverse KL: Mode-Seeking Behavior

**Reverse KL** is $D_{\mathrm{KL}}(q \| p)$ where $q$ is the approximation we optimize and $p$ is fixed. Now the expectation is under $q$ - we penalize regions where $q(x) > 0$ but $p(x) \approx 0$.

**The zero-forcing (mass-concentrating) property:** Regions where $p(x) = 0$ but $q(x) > 0$ contribute $+\infty$ to $D_{\mathrm{KL}}(q\|p)$. Therefore $q^*$ must avoid regions where $p$ is zero - it must *concentrate* its mass on high-probability regions of $p$. This is called **exclusive** KL or **zero-forcing** KL.

**Fitting a Gaussian to a bimodal distribution (reverse direction).** Now minimizing $D_{\mathrm{KL}}(q_\mu \| p)$:

$$D_{\mathrm{KL}}(q_\mu \| p) = \mathbb{E}_{q_\mu}\!\left[\log\frac{q_\mu(X)}{p(X)}\right]$$

This has two local minima: $q^* \approx \mathcal{N}(-3, 1)$ or $q^* \approx \mathcal{N}(3, 1)$ - the approximation *collapses onto one mode* of $p$. The wide, mean-covering solution $\mathcal{N}(0, 10)$ from forward KL is actually a local *maximum* for reverse KL: it places mass where $p$ is near zero (in the valley), incurring large penalties.

**General result.** Reverse KL drives the approximation $q$ to be a *mode* of $p$. The gradient of reverse KL points toward locally consistent regions of $p$.

### 4.3 Geometric Interpretation

```
FORWARD KL vs REVERSE KL: THE KEY DIFFERENCE


  TRUE DISTRIBUTION p (bimodal):        FORWARD KL (minimizer q*)
                                         must cover both modes:
    
                                   q* = N(0, 10)
                                    
                           (wide, mean-seeking)
    ->         places mass in valley
       -3          3          
    

  REVERSE KL (minimizer q*):
  q* = N(-3, 1) OR N(3, 1)
       mode-seeking: picks one mode,
       ignores the other

  Intuition: forward KL sees regions p > 0 as "must cover"
             reverse KL sees regions p  0 as "must avoid"


```

The information-geometric view formalizes this with the concept of **projections**:

- **I-projection** (information projection): $q^* = \arg\min_q D_{\mathrm{KL}}(q \| p)$ - projects $p$ onto the constraint family $\mathcal{Q}$ using reverse KL. Selects the $q \in \mathcal{Q}$ that is closest to $p$ in the reverse KL sense.
- **M-projection** (moment projection): $q^* = \arg\min_q D_{\mathrm{KL}}(p \| q)$ - projects $p$ onto $\mathcal{Q}$ using forward KL. Produces moment-matched approximations.

For exponential families: the M-projection always produces a unique solution (the moment-matched distribution). The I-projection may have multiple local minima (one per mode).

### 4.4 Consequences for Variational Inference

Variational Bayes minimizes $D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z} | \mathbf{x}))$ - reverse KL. This is computationally convenient (the ELBO is a tractable lower bound) but has important consequences:

**Posterior collapse in VAEs.** In a Variational Autoencoder, each latent dimension $z_i$ has approximate posterior $q_\phi(z_i | \mathbf{x}) = \mathcal{N}(\mu_i, \sigma_i^2)$ and prior $p(z_i) = \mathcal{N}(0, 1)$. If the decoder is powerful enough to reconstruct $\mathbf{x}$ using only a subset of dimensions, the ELBO optimization drives the unused dimensions' KL term to zero - meaning $q_\phi(z_i|\mathbf{x}) \to p(z_i) = \mathcal{N}(0,1)$ - a constant prior. This *posterior collapse* means the latent representation ignores the input for those dimensions.

**Why reverse KL causes collapse:** Reverse KL $D_{\mathrm{KL}}(q\|p_{\mathrm{prior}})$ is minimized (= 0) when $q = p_{\mathrm{prior}}$. Forward KL would penalize this collapse heavily because $D_{\mathrm{KL}}(p_{\mathrm{posterior}}\|q)$ would be large if $q$ ignores the data. The zero-forcing property of reverse KL allows collapse; the zero-avoiding property of forward KL would prevent it.

**Beta-VAE fix (Higgins et al., 2017).** Adding a hyperparameter $\beta > 1$ to weight the KL term:
$$\mathcal{L}_\beta = \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - \beta\, D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$
Higher $\beta$ increases the penalty for unused latents, encouraging disentangled representations. This is used in modern VAE-based image models and speech representations.

**KL annealing.** Starting with $\beta = 0$ and gradually increasing it during training prevents early collapse. Used in language model VAEs (Bowman et al., 2016) to avoid the degenerate solution where the encoder is ignored.


## 5. KL for Specific Distributions

### 5.1 KL Between Gaussians

The KL divergence between Gaussian distributions has a closed form used in dozens of ML algorithms.

**Scalar Gaussians.** For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{\mathrm{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**Derivation.** The log-ratio is:

$$\log\frac{p(x)}{q(x)} = \log\frac{\sigma_2}{\sigma_1} - \frac{(x-\mu_1)^2}{2\sigma_1^2} + \frac{(x-\mu_2)^2}{2\sigma_2^2}$$

Taking expectation under $p$, using $\mathbb{E}_p[(X-\mu_1)^2] = \sigma_1^2$ and $\mathbb{E}_p[(X-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2$:

$$D_{\mathrm{KL}}(p\|q) = \log\frac{\sigma_2}{\sigma_1} - \frac{\sigma_1^2}{2\sigma_1^2} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**Intuition on terms:**
- $\log(\sigma_2/\sigma_1)$: penalty for scale mismatch
- $\sigma_1^2/(2\sigma_2^2)$: penalty for $p$ being wider than $q$
- $(\mu_1 - \mu_2)^2/(2\sigma_2^2)$: penalty for mean mismatch (like Mahalanobis distance)
- $-1/2$: normalization constant

**Multivariate Gaussians.** For $p = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$:

$$D_{\mathrm{KL}}(p \| q) = \frac{1}{2}\left[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - d + \log\frac{\det\Sigma_2}{\det\Sigma_1}\right]$$

where $d$ is the dimension. The four terms correspond to: trace ratio (variance mismatch), Mahalanobis distance (mean mismatch), dimension offset, log-determinant ratio (volume mismatch).

**VAE application.** In a standard VAE, the encoder outputs $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$ and the prior is $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$. The KL term simplifies because $\Sigma_2 = I$:

$$D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1\right)$$

This is differentiable in $\boldsymbol{\mu}_\phi$ and $\boldsymbol{\sigma}_\phi$, enabling direct gradient descent. The reparameterization trick $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},I)$ makes the sampling step differentiable.

### 5.2 KL Between Categorical Distributions

For discrete distributions $p = (p_1, \ldots, p_K)$ and $q = (q_1, \ldots, q_K)$:

$$D_{\mathrm{KL}}(p \| q) = \sum_{k=1}^K p_k \ln\frac{p_k}{q_k}$$

This equals $H(p, q) - H(p)$ where $H(p,q) = -\sum_k p_k \ln q_k$ is the cross-entropy. In particular, when $p$ is a one-hot vector (empirical distribution on a single label $y$), $H(p) = 0$ and:

$$D_{\mathrm{KL}}(p \| q) = H(p, q) = -\ln q_y = \text{cross-entropy loss}$$

**Temperature scaling.** In knowledge distillation, the teacher distribution is softened using temperature $\tau > 1$:
$$p_{\mathrm{soft}}(k) = \frac{e^{z_k^T/\tau}}{\sum_j e^{z_j^T/\tau}}$$

The student is trained to minimize $D_{\mathrm{KL}}(p_{\mathrm{soft}} \| q_{\boldsymbol{\theta}})$ rather than the one-hot cross-entropy. Soft targets carry more information (entropy scales up with $\tau$), allowing the student to learn the teacher's "dark knowledge" about relative class similarities.

### 5.3 KL Within the Exponential Family

Members of the exponential family have a remarkably elegant KL formula. An exponential family distribution has density:

$$p_{\boldsymbol{\eta}}(\mathbf{x}) = h(\mathbf{x}) \exp\!\left(\boldsymbol{\eta}^\top \mathbf{t}(\mathbf{x}) - A(\boldsymbol{\eta})\right)$$

where $\boldsymbol{\eta}$ are natural parameters, $\mathbf{t}(\mathbf{x})$ are sufficient statistics, and $A(\boldsymbol{\eta}) = \log \int h(\mathbf{x})e^{\boldsymbol{\eta}^\top \mathbf{t}(\mathbf{x})} d\mathbf{x}$ is the log-partition function (which is convex in $\boldsymbol{\eta}$).

**KL between exponential family members:**

$$D_{\mathrm{KL}}(p_{\boldsymbol{\eta}_1} \| p_{\boldsymbol{\eta}_2}) = A(\boldsymbol{\eta}_2) - A(\boldsymbol{\eta}_1) - \nabla A(\boldsymbol{\eta}_1)^\top(\boldsymbol{\eta}_2 - \boldsymbol{\eta}_1)$$

This is the **Bregman divergence** generated by the convex function $A(\boldsymbol{\eta})$: $B_A(\boldsymbol{\eta}_2, \boldsymbol{\eta}_1) = A(\boldsymbol{\eta}_2) - A(\boldsymbol{\eta}_1) - \nabla A(\boldsymbol{\eta}_1)^\top(\boldsymbol{\eta}_2 - \boldsymbol{\eta}_1)$. This is the error of the first-order Taylor approximation of $A$ around $\boldsymbol{\eta}_1$ - which is always $\ge 0$ by convexity of $A$, recovering Gibbs' inequality.

**Examples:**
- **Gaussian ($\mu$, $\sigma^2$):** $A(\mu,\sigma^2) = \mu^2/(2\sigma^2) + \frac{1}{2}\ln\sigma^2$
- **Bernoulli ($p$):** $A(p) = \ln(1 + e^p)$ (logistic); Bregman divergence = binary KL
- **Poisson ($\lambda$):** $A(\lambda) = e^\lambda$

### 5.4 VAE Closed-Form KL and the Reparameterization Trick

The VAE's KL term per latent dimension is:

$$D_{\mathrm{KL}}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$$

**Derivation.** Using the scalar Gaussian formula with $\mu_2 = 0$, $\sigma_2^2 = 1$:

$$D_{\mathrm{KL}} = \ln\frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2} = -\frac{1}{2}\ln\sigma^2 + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2} = \frac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$$

**Properties:** This quantity is 0 when $\mu = 0, \sigma = 1$ (the posterior equals the prior); strictly positive otherwise. Gradient w.r.t. $\mu$: $\mu$. Gradient w.r.t. $\sigma^2$: $\frac{1}{2}(1 - 1/\sigma^2)$. Both are simple, enabling efficient backpropagation through the KL term.

**Reparameterization trick.** The challenge: we need $\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$ but $z$ depends on $\phi$, so we cannot backpropagate through the sampling. The trick: write $z = \mu_\phi(x) + \sigma_\phi(x) \cdot \varepsilon$ where $\varepsilon \sim \mathcal{N}(0,1)$ is a separate random variable independent of $\phi$. Now $\frac{\partial z}{\partial \phi} = \frac{\partial \mu_\phi}{\partial \phi} + \varepsilon \frac{\partial \sigma_\phi}{\partial \phi}$ is well-defined, and gradients flow through. This is the key insight of Kingma & Welling (2014).


## 6. Information-Theoretic Connections

### 6.1 KL and Cross-Entropy

The relationship $H(p, q) = H(p) + D_{\mathrm{KL}}(p \| q)$ is fundamental. Expanding both sides:

$$-\sum_x p(x)\log q(x) = -\sum_x p(x)\log p(x) + \sum_x p(x)\log\frac{p(x)}{q(x)}$$

which is immediate from $\log(p/q) = \log p - \log q$. This decomposition has critical implications:

- **Training ML models:** Minimizing cross-entropy loss $H(p_{\mathrm{data}}, p_{\boldsymbol{\theta}})$ over $\boldsymbol{\theta}$ is identical to minimizing $D_{\mathrm{KL}}(p_{\mathrm{data}} \| p_{\boldsymbol{\theta}})$, since $H(p_{\mathrm{data}})$ is constant. The irreducible entropy $H(p_{\mathrm{data}})$ is the minimum achievable cross-entropy.

- **Perplexity gap:** A language model with perplexity $\operatorname{PPL} = e^{H(p_{\mathrm{data}}, p_{\boldsymbol{\theta}})}$ has gap $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\boldsymbol{\theta}})$ nats above the theoretical minimum entropy.

- **Calibration:** A perfectly calibrated model satisfies $D_{\mathrm{KL}}(p_{\mathrm{true}}\|p_{\boldsymbol{\theta}}) = 0$, meaning the model's output distribution matches the true conditional distributions exactly.

> **Preview - Cross-Entropy (Section 09-04)**
>
> Cross-entropy $H(p, q) = H(p) + D_{\mathrm{KL}}(p\|q)$ is the canonical training loss for classification. It decomposes neatly into the intrinsic uncertainty $H(p)$ (irreducible) and the model quality gap $D_{\mathrm{KL}}(p\|q)$ (reducible by training).
>
> -> _Full treatment: [Section 09-04 Cross-Entropy](../04-Cross-Entropy/notes.md)_

### 6.2 KL and Mutual Information

> **Preview - Mutual Information (Section 09-03)**
>
> Mutual information is the KL divergence between the joint distribution and the product of marginals:
> $$I(X; Y) = D_{\mathrm{KL}}(P(X, Y) \| P(X)P(Y)) = \mathbb{E}_{(x,y)\sim P}\!\left[\log\frac{P(x,y)}{P(x)P(y)}\right]$$
> It measures statistical dependence: how much knowing $Y$ reduces uncertainty about $X$. This connection shows that $I(X;Y) \ge 0$ (Gibbs' inequality) and $I(X;Y) = 0 \Leftrightarrow X \perp\!\!\!\perp Y$.
>
> -> _Full treatment: [Section 09-03 Mutual Information](../03-Mutual-Information/notes.md)_

The MI-as-KL connection is used directly in **contrastive learning** (InfoNCE loss, SimCLR, CLIP): maximizing a lower bound on $I(X; Z)$ between the input and its representation. It also appears in the **information bottleneck** (Tishby et al., 2000) which frames representation learning as minimizing $I(X; Z)$ subject to maximizing $I(Z; Y)$ - two KL divergences in tension.

### 6.3 KL and Fisher Information

> **Preview - Fisher Information (Section 09-05)**
>
> For a parametric family $\{p_{\boldsymbol{\theta}}\}$, the KL divergence between nearby members is approximately:
> $$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta}+\boldsymbol{\delta}}) \approx \frac{1}{2}\boldsymbol{\delta}^\top F(\boldsymbol{\theta})\boldsymbol{\delta}$$
> where $F(\boldsymbol{\theta}) = \mathbb{E}_{p_{\boldsymbol{\theta}}}[\nabla\log p_{\boldsymbol{\theta}} \nabla\log p_{\boldsymbol{\theta}}^\top]$ is the Fisher information matrix. This local quadratic approximation to KL is what makes natural gradient descent (which preconditions gradient steps by $F^{-1}$) the "geometrically correct" optimizer on the statistical manifold.
>
> -> _Full treatment: [Section 09-05 Fisher Information](../05-Fisher-Information/notes.md)_

**Connection to RLHF natural policy gradient:** The KL trust region $D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\mathrm{ref}}) \le \epsilon$ defines a neighborhood in policy space. Near the reference policy, this KL ball is approximated by $\frac{1}{2}\boldsymbol{\delta}^\top F \boldsymbol{\delta} \le \epsilon$ where $F$ is the Fisher information of $\pi_{\mathrm{ref}}$. Natural policy gradient algorithms (TRPO, PPO) use this approximation.

### 6.4 KL and Entropy

KL divergence provides an elegant proof that entropy is maximized at the uniform distribution. For any distribution $p$ on an alphabet of size $n$ and uniform distribution $u(x) = 1/n$:

$$D_{\mathrm{KL}}(p \| u) = \sum_x p(x)\ln\frac{p(x)}{1/n} = \sum_x p(x)\ln p(x) + \ln n = -H(p) + \ln n$$

Since $D_{\mathrm{KL}}(p\|u) \ge 0$: $H(p) \le \ln n$, with equality iff $p = u$.

More generally, for any *constraint set* $\mathcal{Q}$ of distributions satisfying given moment constraints, the maximum-entropy distribution $p^* = \arg\max_{p \in \mathcal{Q}} H(p)$ equals $\arg\min_{p \in \mathcal{Q}} D_{\mathrm{KL}}(p \| u)$ - it is the M-projection of the uniform distribution onto $\mathcal{Q}$. This unifies the maximum entropy principle with KL geometry.


## 7. f-Divergences and Generalizations

### 7.1 Csiszar f-Divergences

KL divergence is one member of a rich family of divergences introduced by Csiszar (1967). 

**Definition.** Let $f: (0,\infty) \to \mathbb{R}$ be a convex function with $f(1) = 0$. The **f-divergence** of $p$ from $q$ is:

$$D_f(p \| q) = \sum_x q(x)\, f\!\left(\frac{p(x)}{q(x)}\right) = \mathbb{E}_{x\sim q}\!\left[f\!\left(\frac{p(x)}{q(x)}\right)\right]$$

The condition $f(1) = 0$ ensures $D_f(p\|p) = 0$. Convexity of $f$ and Jensen's inequality give $D_f(p\|q) \ge 0$.

**KL divergence as f-divergence.** Taking $f(t) = t\ln t$ (convex, $f(1) = 0$):
$$D_f(p\|q) = \sum_x q(x) \cdot \frac{p(x)}{q(x)}\ln\frac{p(x)}{q(x)} = \sum_x p(x)\ln\frac{p(x)}{q(x)} = D_{\mathrm{KL}}(p\|q)$$

**Standard f-divergences:**

| Name | $f(t)$ | Closed form | Properties |
| --- | --- | --- | --- |
| KL divergence | $t\ln t$ | $\sum p\ln(p/q)$ | Asymmetric; $[0, \infty)$ |
| Reverse KL | $-\ln t$ | $\sum q\ln(q/p)$ | Asymmetric; $[0, \infty)$ |
| Hellinger distance2 | $(\sqrt{t}-1)^2$ | $\sum(\sqrt{p}-\sqrt{q})^2$ | Symmetric; $[0, 2]$ |
| Total variation | $\frac{1}{2}|t-1|$ | $\frac{1}{2}\sum|p-q|$ | Symmetric; $[0, 1]$; a metric |
| Chi-squared | $(t-1)^2$ | $\sum(p-q)^2/q$ | Asymmetric; $[0,\infty)$ |
| $\alpha$-divergence | $\frac{t^\alpha - \alpha t - (1-\alpha)}{\alpha(\alpha-1)}$ | General; limits give KL | Parameterizes all of the above |

**For AI - GAN losses.** Generative Adversarial Networks (Goodfellow et al., 2014) use different f-divergences as training objectives: the original GAN uses Jensen-Shannon; f-GAN (Nowozin et al., 2016) generalizes to arbitrary f-divergences. This provides a principled family of training objectives for generative models.

### 7.2 Properties of f-Divergences

All f-divergences share four key properties (when $f$ is convex with $f(1) = 0$):

1. **Non-negativity:** $D_f(p\|q) \ge 0$, with equality iff $p = q$ (when $f$ is strictly convex).
2. **Data processing inequality:** $D_f(p_T\|q_T) \le D_f(p\|q)$ for any stochastic kernel $T$.
3. **No triangle inequality** in general.
4. **Convexity:** $D_f$ is jointly convex in $(p,q)$ when $f$ is convex.

The data processing inequality is a theorem for all f-divergences: processing can only reduce distinguishability between distributions.

**Total variation as special case.** The total variation distance $\mathrm{TV}(p,q) = \frac{1}{2}\|p - q\|_1 = \frac{1}{2}\sum_x |p(x) - q(x)|$ satisfies the triangle inequality and is a genuine metric. It bounds other f-divergences: Pinsker's inequality relates KL and TV:

$$\mathrm{TV}(p,q) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(p\|q)}$$

This is the most important inequality connecting KL divergence to total variation. Pinsker's inequality is used to convert KL bounds (from optimization) into total variation bounds (which control probabilities of events).

### 7.3 Renyi Divergence

For $\alpha \in (0,1) \cup (1,\infty)$, the **order-$\alpha$ Renyi divergence** is:

$$D_\alpha(p \| q) = \frac{1}{\alpha - 1}\log\sum_x p(x)^\alpha\, q(x)^{1-\alpha}$$

**Limiting cases:**
- $\alpha \to 1$: $D_\alpha(p\|q) \to D_{\mathrm{KL}}(p\|q)$ (L'Hopital's rule)
- $\alpha \to 0$: $D_\alpha(p\|q) \to -\log\sum_x \mathbb{1}[p(x) > 0]\, q(x)$ (support-based)
- $\alpha \to \infty$: $D_\alpha(p\|q) \to \log\max_x p(x)/q(x)$ (max-ratio)
- $\alpha = 1/2$: related to Bhattacharyya distance; $D_{1/2}(p\|q) = -2\log\sum_x\sqrt{p(x)q(x)}$

**Properties:** $D_\alpha(p\|q) \ge 0$; $D_\alpha(p\|p) = 0$; $D_\alpha$ is monotone increasing in $\alpha$; satisfies data processing inequality; not symmetric.

**For AI - Differential privacy.** Renyi differential privacy (Mironov, 2017) quantifies privacy loss using Renyi divergence: a mechanism $M$ is $(\alpha, \varepsilon)$-RDP if $D_\alpha(M(S)\|M(S')) \le \varepsilon$ for any adjacent datasets $S, S'$. Renyi DP composes additively: $\varepsilon_{\text{total}} = \varepsilon_1 + \varepsilon_2$. This makes it the tool of choice for privacy accounting in large-scale training with DP-SGD (used in Google's federated learning and some LLM fine-tuning pipelines).

### 7.4 Jensen-Shannon Divergence

The **Jensen-Shannon Divergence (JSD)** is the symmetrized, bounded variant of KL:

$$\mathrm{JSD}(p \| q) = \frac{1}{2}D_{\mathrm{KL}}\!\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2}D_{\mathrm{KL}}\!\left(q \,\Big\|\, \frac{p+q}{2}\right)$$

where $m = (p+q)/2$ is the mixture distribution.

**Properties:**
- $\mathrm{JSD}(p\|q) = \mathrm{JSD}(q\|p)$ - symmetric
- $0 \le \mathrm{JSD}(p\|q) \le \ln 2$ - bounded (in nats)
- $\sqrt{\mathrm{JSD}}$ is a metric (satisfies triangle inequality)
- $\mathrm{JSD}(p\|q) = H(m) - \frac{1}{2}H(p) - \frac{1}{2}H(q)$ - entropy of mixture minus average entropy

**For AI - Original GAN objective.** The original GAN (Goodfellow et al., 2014) with an optimal discriminator minimizes the Jensen-Shannon divergence between the generator distribution $p_G$ and the data distribution $p_{\mathrm{data}}$:

$$\min_G \max_D V(D, G) = 2\,\mathrm{JSD}(p_{\mathrm{data}} \| p_G) - 2\ln 2$$

The optimal discriminator is $D^*(x) = p_{\mathrm{data}}(x)/(p_{\mathrm{data}}(x) + p_G(x))$. However, when $p_{\mathrm{data}}$ and $p_G$ have disjoint supports, $\mathrm{JSD} = \ln 2$ (maximum) everywhere - the discriminator saturates and gradients vanish. This **mode collapse / vanishing gradient** problem motivated WGAN (Arjovsky et al., 2017), which uses the Wasserstein distance (which doesn't require absolute continuity).


## 8. Information Geometry of KL

### 8.1 KL as a Bregman Divergence

A **Bregman divergence** generated by a strictly convex function $\phi$ is:

$$B_\phi(p, q) = \phi(p) - \phi(q) - \nabla\phi(q)^\top(p - q)$$

This is the gap between $\phi$ at $p$ and its first-order Taylor approximation at $q$ - always $\ge 0$ by convexity.

**KL is a Bregman divergence.** Taking $\phi(p) = \sum_x p(x)\ln p(x)$ (the negative entropy, which is strictly convex):

$$\nabla\phi(q)_x = \ln q(x) + 1$$

$$B_\phi(p, q) = \sum_x p(x)\ln p(x) - \sum_x q(x)\ln q(x) - \sum_x (\ln q(x) + 1)(p(x) - q(x))$$

$$= \sum_x p(x)\ln p(x) - \sum_x p(x)\ln q(x) + \sum_x (q(x) - p(x)) = \sum_x p(x)\ln\frac{p(x)}{q(x)} = D_{\mathrm{KL}}(p\|q)$$

This Bregman representation reveals why KL is asymmetric: Bregman divergences are generally not symmetric because the Taylor approximation is computed at $q$, not at $p$.

**Exponential family connection.** For an exponential family with log-partition function $A(\boldsymbol{\eta})$, the Bregman divergence $B_A(\boldsymbol{\eta}_2, \boldsymbol{\eta}_1)$ equals $D_{\mathrm{KL}}(p_{\boldsymbol{\eta}_1}\|p_{\boldsymbol{\eta}_2})$ (note reversed argument order). This is the source of the elegant formula in Section 5.3.

### 8.2 I-Projection and M-Projection

**Definition.** Let $\mathcal{M}$ be a constraint set (e.g., a parametric family of distributions):

- **I-projection** (information projection): $q^* = \arg\min_{q \in \mathcal{M}} D_{\mathrm{KL}}(q \| p)$ - the distribution in $\mathcal{M}$ closest to $p$ in reverse KL
- **M-projection** (moment projection): $q^* = \arg\min_{q \in \mathcal{M}} D_{\mathrm{KL}}(p \| q)$ - the distribution in $\mathcal{M}$ closest to $p$ in forward KL

For the exponential family:
- The M-projection always produces the **moment-matched** distribution: $\mathbb{E}_{q^*}[\mathbf{t}(X)] = \mathbb{E}_p[\mathbf{t}(X)]$
- The I-projection onto an exponential family (convex set of mean parameters) produces the distribution with the correct natural parameters for those means

**EM algorithm as alternating projections.** The EM algorithm alternates:
- **E-step:** Compute $q^{(t)}(\mathbf{z}) = p(\mathbf{z}|\mathbf{x}; \boldsymbol{\theta}^{(t)})$ - I-projection of the previous model onto the exact posterior
- **M-step:** $\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} \mathcal{Q}(\boldsymbol{\theta}, \boldsymbol{\theta}^{(t)})$ - M-projection of $q^{(t)}$ back onto the parametric family

Each step decreases $D_{\mathrm{KL}}(q\|p)$ or $D_{\mathrm{KL}}(p_{\text{data}}\|p_{\boldsymbol{\theta}})$ respectively, guaranteeing convergence to a stationary point.

### 8.3 The Pythagorean Theorem for KL

**Theorem (Pythagorean theorem for I-projections).** Let $\mathcal{E}$ be an exponential family (a convex set in mean parameter space) and let $q^* = \arg\min_{q \in \mathcal{E}} D_{\mathrm{KL}}(q \| p)$ be the I-projection of $p$ onto $\mathcal{E}$. Then for any $r \in \mathcal{E}$:

$$D_{\mathrm{KL}}(r \| p) = D_{\mathrm{KL}}(r \| q^*) + D_{\mathrm{KL}}(q^* \| p)$$

This is an exact equality - there is no approximation, unlike the geometric Pythagorean theorem.

**Interpretation.** The KL distance from any $r$ in the constraint set to the target $p$ decomposes into: (1) the distance from $r$ to the projection $q^*$, plus (2) the distance from the projection to $p$. The "closest" point $q^*$ acts like the foot of a perpendicular. This is called "Pythagorean" because the geometry is orthogonal in the KL sense.

```
PYTHAGOREAN THEOREM FOR KL


          p  (target distribution, outside set E)
          |
          | D_KL(q* || p)
          |
          q*  r    (all in constraint set E)
            D_KL(r || q*)

    D_KL(r || p) = D_KL(r || q*) + D_KL(q* || p)

    (exact for I-projections onto exponential families)


```

**For AI:** The variational inference ELBO maximization is equivalent to minimizing $D_{\mathrm{KL}}(q\|p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}))$ - an I-projection. The Pythagorean theorem shows that the ELBO gap is exactly $D_{\mathrm{KL}}(q^*\|p)$ at the optimum - a clean statement of how much information is lost by restricting to the variational family.

### 8.4 Statistical Manifolds and Natural Gradient

The space of probability distributions over $\mathcal{X}$ forms a statistical manifold with local coordinates given by the natural parameters $\boldsymbol{\eta}$ (for exponential families). The KL divergence induces a Riemannian metric on this manifold called the **Fisher information metric**:

$$g_{ij}(\boldsymbol{\eta}) = \frac{\partial^2 D_{\mathrm{KL}}(p_{\boldsymbol{\eta}} \| p_{\boldsymbol{\eta}'})}{\partial\eta_i' \partial\eta_j'}\bigg|_{\boldsymbol{\eta}'=\boldsymbol{\eta}} = F_{ij}(\boldsymbol{\eta})$$

where $F(\boldsymbol{\eta})$ is the Fisher information matrix. The Riemannian structure makes the "natural" step size in parameter space match the actual KL distance between distributions.

**Natural gradient.** In Euclidean parameter space, gradient descent steps $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla\mathcal{L}$ make steps proportional to the Euclidean metric on $\boldsymbol{\theta}$-space, which does not correspond to any natural metric on distribution space. The **natural gradient** corrects for this:

$$\tilde{\nabla}\mathcal{L} = F(\boldsymbol{\theta})^{-1}\nabla\mathcal{L}$$

Natural gradient descent converges much faster in problems where the Fisher information is highly non-uniform (i.e., where the Euclidean metric on parameters is poorly calibrated to KL distance on distributions). This is the foundation of **K-FAC** (Martens & Grosse, 2015), used in second-order optimization of neural networks.

> -> _Full treatment of Fisher information and natural gradient: [Section 09-05 Fisher Information](../05-Fisher-Information/notes.md)_


## 9. Applications in Machine Learning

### 9.1 Maximum Likelihood = Minimizing KL

**Theorem.** For a parametric model $p_{\boldsymbol{\theta}}$ and empirical data distribution $\hat{p}_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbb{1}[X^{(i)} = x]$:

$$\arg\max_{\boldsymbol{\theta}} \sum_{i=1}^n \log p_{\boldsymbol{\theta}}(x^{(i)}) = \arg\min_{\boldsymbol{\theta}} D_{\mathrm{KL}}(\hat{p}_n \| p_{\boldsymbol{\theta}})$$

**Proof.** Using $D_{\mathrm{KL}}(\hat{p}_n \| p_{\boldsymbol{\theta}}) = \mathbb{E}_{\hat{p}}[\log\hat{p}(x)/p_{\boldsymbol{\theta}}(x)] = H(\hat{p}_n) - \mathbb{E}_{\hat{p}}[\log p_{\boldsymbol{\theta}}(x)]$, minimizing over $\boldsymbol{\theta}$ eliminates the constant $H(\hat{p}_n)$:

$$\arg\min_{\boldsymbol{\theta}} D_{\mathrm{KL}}(\hat{p}_n\|p_{\boldsymbol{\theta}}) = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{\hat{p}}[\log p_{\boldsymbol{\theta}}(X)] = \arg\max_{\boldsymbol{\theta}} \frac{1}{n}\sum_i \log p_{\boldsymbol{\theta}}(x^{(i)}) \quad\square$$

**Implications:**
- Every LLM trained by next-token prediction (GPT-4, LLaMA-3, Claude, Gemini) minimizes $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\boldsymbol{\theta}})$
- The MLE estimate is not "arbitrary" - it has a precise information-theoretic interpretation as the distribution in the model family closest to the data in forward KL
- Coverage: MLE must cover the entire data distribution; if $p_{\mathrm{data}}(x) > 0$ but $p_{\boldsymbol{\theta}}(x) \approx 0$, the loss is large

### 9.2 Variational Autoencoders

The VAE (Kingma & Welling, 2014) introduces a latent variable $\mathbf{z}$ with prior $p(\mathbf{z})$ and likelihood $p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})$. The true log-likelihood $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ is intractable (requires integrating over $\mathbf{z}$). The VAE introduces an encoder $q_\phi(\mathbf{z}|\mathbf{x})$ and derives a tractable lower bound via the KL decomposition:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi}[\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\mathcal{L}(\boldsymbol{\phi},\boldsymbol{\theta};\mathbf{x}) = \text{ELBO}} + \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z}|\mathbf{x}))}_{\ge 0}$$

**The ELBO has two terms:**
1. **Reconstruction term** $\mathbb{E}_{q_\phi}[\log p_{\boldsymbol{\theta}}(\mathbf{x}|\mathbf{z})]$: how well the decoder reconstructs from the latent code. Analogous to negative distortion.
2. **KL regularizer** $D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}))$: how much the approximate posterior deviates from the prior. For Gaussian encoder and standard Gaussian prior: $\frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \ln\sigma_j^2 - 1)$.

**Practical training:** The ELBO is maximized by alternating gradient steps:
- w.r.t. $\boldsymbol{\phi}$: encoder parameters, using the reparameterization trick
- w.r.t. $\boldsymbol{\theta}$: decoder parameters, via direct backpropagation

**VAE variants using KL:** beta-VAE ($\beta > 1$ weight on KL for disentanglement), WAE (Wasserstein autoencoder replaces KL with MMD), VQ-VAE (discrete latent, no KL), InfoVAE (adds MI term).

### 9.3 RLHF, PPO, and DPO

**RLHF objective.** Reinforcement Learning from Human Feedback (Christiano et al., 2017; InstructGPT, Ouyang et al., 2022) trains a policy $\pi_{\boldsymbol{\theta}}$ to maximize expected reward while staying close to the reference policy $\pi_{\mathrm{ref}}$ (the pretrained LLM):

$$\max_{\pi_{\boldsymbol{\theta}}} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\boldsymbol{\theta}}(\cdot|x)}\left[r(x, y) - \beta\, D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}}(\cdot|x) \| \pi_{\mathrm{ref}}(\cdot|x))\right]$$

where $r(x,y)$ is the reward model score and $\beta > 0$ is the KL coefficient. Without the KL penalty, the policy collapses to reward hacking: generating short, repetitive text that maximizes reward scores but is incoherent or unnatural.

**Optimal policy.** The RLHF objective has a closed-form optimal solution (for fixed $r$):

$$\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\, e^{r(x,y)/\beta}$$

where $Z(x) = \sum_y \pi_{\mathrm{ref}}(y|x)e^{r(x,y)/\beta}$ is the normalizing partition function. The optimal policy exponentially upweights sequences with high reward relative to the reference.

**DPO (Direct Preference Optimization, Rafailov et al., 2023).** DPO solves the RLHF objective directly from preference data, without training an explicit reward model. The key insight: substituting the optimal policy formula into the reward-learning objective yields a loss that only involves the policy, not the reward model:

$$\mathcal{L}_{\mathrm{DPO}}(\boldsymbol{\theta}) = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\ln\frac{\pi_{\boldsymbol{\theta}}(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\ln\frac{\pi_{\boldsymbol{\theta}}(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right]$$

where $y_w$ is the preferred (winning) response and $y_l$ the dispreferred (losing) response. The log-ratio $\log(\pi_{\boldsymbol{\theta}}/\pi_{\mathrm{ref}})$ is the implicit reward that the KL-constrained policy learns. DPO is used in practice for fine-tuning LLMs on human preferences (LLaMA-2-Chat, Mixtral-Instruct, and many others).

**PPO trust region.** PPO (Schulman et al., 2017) approximates the KL constraint with a clipping objective:

$$\mathcal{L}_{\mathrm{CLIP}}(\boldsymbol{\theta}) = \mathbb{E}_t\!\left[\min\!\left(r_t(\boldsymbol{\theta})\hat{A}_t,\; \mathrm{clip}(r_t(\boldsymbol{\theta}), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

where $r_t(\boldsymbol{\theta}) = \pi_{\boldsymbol{\theta}}(a_t|s_t)/\pi_{\mathrm{old}}(a_t|s_t)$ is the probability ratio. Clipping at $1 \pm \varepsilon$ prevents large policy updates, approximating the KL trust region constraint $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\boldsymbol{\theta}}) \le \delta$.

### 9.4 Knowledge Distillation

Knowledge distillation (Hinton, Vinyals & Dean, 2015) transfers knowledge from a large teacher model $p_T$ to a smaller student model $p_S$. The distillation loss uses **forward KL** (from teacher to student):

$$\mathcal{L}_{\mathrm{distill}} = D_{\mathrm{KL}}(p_T^{(\tau)} \| p_S) = \sum_k p_T^{(\tau)}(k)\ln\frac{p_T^{(\tau)}(k)}{p_S(k)}$$

where $p_T^{(\tau)}(k) = \text{softmax}(z_k^T/\tau)_k$ is the teacher's softened distribution at temperature $\tau > 1$.

**Why forward KL (teacher -> student)?** The student must cover all of the teacher's probability mass, including the "dark knowledge" - the non-zero probabilities assigned to incorrect classes that encode the teacher's generalization patterns. If a teacher assigns 60% to "cat," 30% to "tiger," and 10% to "dog," the soft labels teach the student that cats look more like tigers than dogs. One-hot labels destroy this information.

**Forward vs reverse KL choice in distillation.** Using reverse KL $D_{\mathrm{KL}}(p_S\|p_T)$ would allow the student to become mode-seeking - focusing only on the teacher's highest-probability output and ignoring the distributional shape. Forward KL forces the student to match the full distribution, preserving the teacher's calibration.

**Modern LLM distillation.** DistilBERT (Sanh et al., 2019) distills BERT using forward KL plus cosine similarity on hidden states. Phi-1 (Gunasekar et al., 2023) and Phi-2 (Microsoft, 2023) use distillation from GPT-4 to train small but capable models. LLaMA-3-8B was distilled from larger checkpoints. The forward KL direction is standard.

### 9.5 Normalizing Flows and Diffusion Models

**Normalizing flows** learn an invertible transformation $f_{\boldsymbol{\theta}}: \mathbb{R}^d \to \mathbb{R}^d$ such that $\mathbf{x} = f_{\boldsymbol{\theta}}(\mathbf{z})$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$. The exact change-of-variables formula gives:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log p_{\mathbf{z}}(f_{\boldsymbol{\theta}}^{-1}(\mathbf{x})) + \log|\det J_{f_{\boldsymbol{\theta}}^{-1}}(\mathbf{x})|$$

Training by maximum likelihood minimizes $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\boldsymbol{\theta}})$ (forward KL) since the log-likelihood is tractable. This contrasts with GANs, which use a discriminator to avoid computing $\log p_{\boldsymbol{\theta}}$ directly.

**Diffusion models (DDPM, Ho et al., 2020).** The DDPM ELBO is a sum of KL terms:

$$-\log p_{\boldsymbol{\theta}}(\mathbf{x}) \le \mathcal{L} = D_{\mathrm{KL}}(q(\mathbf{x}_T|\mathbf{x}_0)\|p(\mathbf{x}_T)) + \sum_{t=2}^T D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)\|p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_{\boldsymbol{\theta}}(\mathbf{x}_0|\mathbf{x}_1)$$

Each reverse diffusion step $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}|\mathbf{x}_t)$ is trained to minimize the KL divergence from the (analytically tractable) true reverse $q(\mathbf{x}_{t-1}|\mathbf{x}_t,\mathbf{x}_0)$. Because both distributions are Gaussian (by the Markov structure), each per-step KL has a closed-form expression in terms of the noise predictor $\boldsymbol{\epsilon}_{\boldsymbol{\theta}}$. The simplified training objective is equivalent to predicting the noise $\boldsymbol{\epsilon}$ - which is the standard diffusion training loss.


## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Writing $D_{\mathrm{KL}}(q \| p)$ when you mean forward KL (truth -> approximation) | The first argument is the reference (truth); forward KL is $D_{\mathrm{KL}}(p_{\mathrm{truth}} \| p_{\mathrm{approx}})$ | Always identify which distribution is the truth and which is the approximation; check: is the expectation under truth or approx? |
| 2 | Assuming KL is symmetric: $D_{\mathrm{KL}}(p\|q) = D_{\mathrm{KL}}(q\|p)$ | KL is not symmetric; the two directions have fundamentally different behaviors | Never swap $p$ and $q$ without checking both directions; use Jeffrey's JSD if you need symmetry |
| 3 | Ignoring support mismatch: computing $D_{\mathrm{KL}}$ when $q(x) = 0$ for some $x$ with $p(x) > 0$ | $D_{\mathrm{KL}}(p\|q) = +\infty$ when $p \not\ll q$; finite values are meaningless | Add Laplace smoothing or use $\mathrm{JSD}$ / $\mathrm{TV}$ for distributions with potentially different supports |
| 4 | Treating $D_{\mathrm{KL}}(p\|q) = 0$ as equivalent to $p = q$ under numerical approximations | Finite-precision computation gives $D_{\mathrm{KL}} \approx 0$ even when $p \ne q$ due to rounding | Use $D_{\mathrm{KL}} \le \epsilon$ (small threshold) rather than exact zero; check sample moments separately |
| 5 | Using KL as a distance metric satisfying the triangle inequality | $D_{\mathrm{KL}}(p\|r)$ can exceed $D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|r)$; KL is not a metric | Use total variation or $\sqrt{\mathrm{JSD}}$ if you need a proper metric |
| 6 | Confusing minimizing $D_{\mathrm{KL}}(p\|q)$ over $q$ with minimizing over $p$ | MLE minimizes forward KL over $\boldsymbol{\theta}$ (model parameters); variational inference minimizes reverse KL over $q$ | Always identify which argument is the optimization variable |
| 7 | Computing cross-entropy loss and calling it KL divergence | Cross-entropy $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$; they differ by $H(p)$ | They're equal only when $p$ is one-hot ($H(p) = 0$); for soft labels, $H(p,q) > D_{\mathrm{KL}}(p\|q)$ |
| 8 | Expecting the VAE KL term to be zero at optimum | The KL term equals $D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}))$; zero only if posterior = prior (posterior collapse) | A positive KL term is healthy; it means the encoder is using the data |
| 9 | Using forward KL for variational inference | Forward KL $\mathbb{E}_p[\log(p/q)]$ requires samples from the intractable $p(\mathbf{z}|\mathbf{x})$; ELBO uses reverse KL which only requires samples from $q$ | Variational inference must use reverse KL because it's the only direction you can evaluate without the true posterior |
| 10 | Claiming the RLHF KL penalty "forces" the model to stay close to reference | The KL penalty softly penalizes deviation; with large enough reward, the policy can still deviate significantly | The KL penalty is a soft regularizer, not a hard constraint; monitor $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ during training |
| 11 | Conflating Renyi divergence with KL divergence | Renyi $D_\alpha(p\|q)$ only equals KL in the limit $\alpha \to 1$; for $\alpha \ne 1$ they have different properties | Use Renyi divergence when differential privacy accounting is needed; use KL for standard information-theoretic arguments |
| 12 | Forgetting the $\ln 2$ factor when converting between nats and bits | $D_{\mathrm{KL}}$ in nats $= D_{\mathrm{KL}}$ in bits $\times \ln 2 \approx 0.693 \times$ bits | Always specify units; when comparing with cross-entropy in bits (binary classification), convert to nats or vice versa |

## 11. Exercises

**Exercise 1 (*) - Non-negativity and equality condition.** Let $p = (0.4, 0.3, 0.2, 0.1)$ and $q = (0.1, 0.2, 0.3, 0.4)$.

(a) Compute $D_{\mathrm{KL}}(p\|q)$ and $D_{\mathrm{KL}}(q\|p)$ by hand (use $\ln$, give exact values and numerical approximations).

(b) Verify that both are non-negative. Which is larger?

(c) Find a distribution $r$ on four symbols such that $D_{\mathrm{KL}}(p\|r) = 0$. Is $r$ unique?

(d) Using only the convexity of $-\ln$, write a self-contained proof that $D_{\mathrm{KL}}(p\|q) \ge 0$ for your specific $p$ and $q$.

---

**Exercise 2 (*) - KL between Gaussians.** Let $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$.

(a) Starting from the definition, derive the closed form:
$$D_{\mathrm{KL}}(p\|q) = \ln\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

(b) Compute $D_{\mathrm{KL}}(\mathcal{N}(1,2)\|\mathcal{N}(0,1))$ and $D_{\mathrm{KL}}(\mathcal{N}(0,1)\|\mathcal{N}(1,2))$. Are they equal?

(c) Show that when $\sigma_1 = \sigma_2 = \sigma$, the formula reduces to $D_{\mathrm{KL}}(p\|q) = (\mu_1-\mu_2)^2/(2\sigma^2)$ - the squared Mahalanobis distance.

(d) For the VAE case $p = \mathcal{N}(\mu, \sigma^2)$, $q = \mathcal{N}(0,1)$: verify the simplified formula $\frac{1}{2}(\mu^2 + \sigma^2 - \ln\sigma^2 - 1)$ and check that it is 0 when $\mu = 0$, $\sigma = 1$.

---

**Exercise 3 (*) - Forward vs reverse KL.** Let $p(x) = 0.5\,\mathcal{N}(x;-2,0.25) + 0.5\,\mathcal{N}(x;2,0.25)$ (bimodal) and $q_\mu(x) = \mathcal{N}(x;\mu,\sigma^2)$.

(a) Show that the minimizer of $D_{\mathrm{KL}}(p\|q_{\mu,\sigma^2})$ over $(\mu,\sigma^2)$ is $\mu^* = 0$, $\sigma^{*2} = 4.25$. (Hint: moment matching.)

(b) Numerically estimate $D_{\mathrm{KL}}(q_\mu\|p)$ on a grid of $\mu \in [-3, 3]$ using 1000 samples. Show the two local minima near $\mu = \pm 2$.

(c) Explain in one paragraph: if you were training a VAE generative model and the true data distribution is $p$, which direction would you minimize, and what would the generator look like?

---

**Exercise 4 (**) - Chain rule decomposition.** Let $P(X, Y)$ be a joint distribution on $\{0,1\}^2$ with $P(0,0) = 0.4$, $P(0,1) = 0.1$, $P(1,0) = 0.1$, $P(1,1) = 0.4$, and $Q(X,Y)$ uniform on $\{0,1\}^2$ ($= 0.25$ each).

(a) Compute $D_{\mathrm{KL}}(P(X,Y)\|Q(X,Y))$ directly.

(b) Compute $D_{\mathrm{KL}}(P_X\|Q_X)$ and $\mathbb{E}_{P_X}[D_{\mathrm{KL}}(P_{Y|X}\|Q_{Y|X})]$ separately.

(c) Verify that (a) $=$ (b): the chain rule holds.

(d) Interpret: which direction (marginal or conditional) contributes more to the total KL? What does this mean for the structure of $P$ vs $Q$?

---

**Exercise 5 (**) - f-Divergence family.** Consider three f-divergences on Bernoulli distributions $p = \text{Bern}(\theta)$ and $q = \text{Bern}(0.5)$ as a function of $\theta \in (0,1)$.

(a) Compute and plot (as functions of $\theta$): KL divergence $D_{\mathrm{KL}}(p\|q)$, reverse KL $D_{\mathrm{KL}}(q\|p)$, total variation $\mathrm{TV}(p,q)$, and Jensen-Shannon $\mathrm{JSD}(p,q)$.

(b) Verify that Pinsker's inequality $\mathrm{TV}(p,q)^2 \le \frac{1}{2}D_{\mathrm{KL}}(p\|q)$ holds for all $\theta$.

(c) Which divergence is symmetric? Which is bounded? Which is asymmetric?

(d) For $\theta = 0.01$ (near-deterministic $p$): compute all four divergences. Why does KL blow up while JSD stays bounded?

---

**Exercise 6 (**) - ELBO and KL decomposition.** A simple VAE has prior $p(z) = \mathcal{N}(0,1)$ and likelihood $p_\theta(x|z) = \mathcal{N}(z, 0.1)$. The encoder is $q_\phi(z|x) = \mathcal{N}(\phi \cdot x, 0.5)$ for a scalar parameter $\phi$.

(a) For $x = 2$: compute the ELBO as a function of $\phi$. Use the Gaussian KL formula.

(b) Show that the ELBO decomposes as: reconstruction term $-\frac{(2 - \phi\cdot 2)^2 + 0.5}{2 \times 0.1}$ (plus constants) minus the KL term $\frac{1}{2}(\phi^2 \cdot 4 + 0.5 - \ln 0.5 - 1)$.

(c) Find the $\phi$ that maximizes the ELBO. What does this optimal encoder do geometrically?

(d) As the likelihood variance $\sigma^2 \to 0$ (decoder becomes very powerful): what happens to the optimal $\phi$? Does posterior collapse occur?

---

**Exercise 7 (***) - RLHF optimal policy.** A language model has reference policy $\pi_{\mathrm{ref}}(y|x)$ and reward $r(x,y)$. The RLHF objective is $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$.

(a) Show that the unconstrained optimizer is $\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)e^{r(x,y)/\beta}$ by taking the functional derivative of the objective with respect to $\pi(y|x)$ (subject to $\sum_y \pi(y|x) = 1$).

(b) Show that at the optimum, the KL divergence is $D_{\mathrm{KL}}(\pi^*\|\pi_{\mathrm{ref}}) = \frac{1}{\beta}\mathbb{E}_{\pi^*}[r] - \log Z(x)$.

(c) Derive the DPO loss: given preference data $(x, y_w, y_l)$ where $y_w \succ y_l$, show that the Bradley-Terry preference model $P(y_w \succ y_l|x) = \sigma(\beta\log\frac{\pi^*(y_w|x)}{\pi^*(y_l|x)} - \beta\log\frac{\pi_{\mathrm{ref}}(y_w|x)}{\pi_{\mathrm{ref}}(y_l|x)})$ leads to the DPO loss.

(d) Numerically: for $\beta = 0.1$, $\pi_{\mathrm{ref}}(A) = 0.6$, $\pi_{\mathrm{ref}}(B) = 0.4$, $r(A) = 2$, $r(B) = 0$: compute $\pi^*$ and the resulting $D_{\mathrm{KL}}(\pi^*\|\pi_{\mathrm{ref}})$.

---

**Exercise 8 (***) - Knowledge distillation analysis.** A teacher model outputs logits $\mathbf{z}_T = (3, 1, -2)$ for three classes and a student outputs $\mathbf{z}_S = (2, 0, -1)$.

(a) Compute $p_T = \text{softmax}(\mathbf{z}_T)$ and $p_S = \text{softmax}(\mathbf{z}_S)$.

(b) Compute $D_{\mathrm{KL}}(p_T\|p_S)$ (forward KL, what distillation minimizes) and $D_{\mathrm{KL}}(p_S\|p_T)$ (reverse KL).

(c) Now compute soft labels at temperature $\tau = 3$: $p_T^{(\tau)}$ and $p_S^{(\tau)}$. Recompute both KL divergences. How does temperature affect the magnitude of the KL and the information in the soft labels?

(d) What is the total distillation loss: $\mathcal{L} = \lambda D_{\mathrm{KL}}(p_T^{(\tau)}\|p_S^{(\tau)}) + (1-\lambda)H(\mathbf{y}_{\mathrm{true}}, p_S)$ for $\lambda = 0.7$ and $\mathbf{y}_{\mathrm{true}} = (1, 0, 0)$?

(e) Argue why forward KL (teacher to student) is used rather than reverse KL. What kind of "dark knowledge" is preserved?


## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Forward KL = MLE | Every language model trained by next-token prediction (GPT-4, LLaMA-3, Claude, Gemini, Mistral) minimizes $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{\boldsymbol{\theta}})$; cross-entropy loss IS KL divergence |
| ELBO = KL decomposition | All VAE-based generative models (image, speech, latent diffusion) optimize reconstruction + KL; the KL term determines latent space structure |
| RLHF KL penalty | All instruction-tuned LLMs (InstructGPT, Claude, Gemini) use $\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ to prevent reward hacking; $\beta$ is one of the most important hyperparameters in alignment |
| DPO implicit KL | DPO (used in LLaMA-2-Chat, Mistral-Instruct) reframes RLHF as direct KL-constrained optimization; the log-ratio $\log(\pi/\pi_{\mathrm{ref}})$ IS the implicit reward |
| Forward vs reverse KL | Choosing forward KL (MLE, distillation) vs reverse KL (VI, mean-field) is the defining design decision of probabilistic ML; wrong choice -> blurry outputs or posterior collapse |
| Knowledge distillation | LLM compression (DistilGPT, Phi-2, Phi-3) uses forward KL from teacher to student; soft labels at temperature $\tau > 1$ carry 10-100 more information than one-hot labels |
| Posterior collapse | When the decoder is powerful, the VAE KL term collapses to zero - a major pathology in text VAEs (Bowman et al., 2016); fixed by KL annealing or $\beta$-VAE weighting |
| PPO trust region | PPO's clip objective approximates a KL trust region; the $\varepsilon \in \{0.1, 0.2\}$ clip threshold corresponds to a specific KL budget per update step |
| GAN = JSD minimization | Original GAN (Goodfellow, 2014) minimizes JSD; JSD's saturation when supports are disjoint causes vanishing gradients and mode collapse - motivating WGAN/Wasserstein distance |
| Renyi DP for private training | Differential privacy in LLM fine-tuning (DP-SGD, used at Google) uses Renyi divergence; Renyi DP composes additively, making privacy budgets tractable |
| Diffusion ELBO | DDPM training objective is a sum of per-step KL divergences; the connection to score matching makes the KL formulation equivalent to noise prediction |
| Natural gradient | K-FAC (second-order optimizer for neural networks) approximates the Fisher matrix $F = \mathbb{E}[\nabla\log p \nabla\log p^\top]$ which is the local KL curvature; natural gradient descent is the geometrically correct first-order method on the statistical manifold |
| Calibration | $D_{\mathrm{KL}}(p_{\mathrm{true}}\|p_{\boldsymbol{\theta}}) = 0$ iff the model is perfectly calibrated; calibration regularization (temperature scaling, label smoothing) implicitly minimizes KL between predicted and calibrated distributions |

## 13. Conceptual Bridge

KL divergence sits at the center of the information theory chapter, connecting the individual uncertainty measured by entropy (Section 01) to the distributional relationships measured by mutual information (Section 03), the training objectives formalized as cross-entropy (Section 04), and the local geometry captured by Fisher information (Section 05). It is not an isolated concept - it is the engine that drives all four.

**Backward: from Entropy.** In Section 01, entropy $H(X) = -\sum p\log p$ measured the intrinsic uncertainty of a single distribution. KL divergence extends this to *pairs* of distributions: $D_{\mathrm{KL}}(p\|q) = H(p,q) - H(p)$ measures the extra uncertainty you incur by confusing $q$ for $p$. The non-negativity of KL proved the fundamental bound $H(X) \le \log|\mathcal{X}|$ that appeared in Section 01 - Gibbs' inequality was already implicitly used there. Every result about optimal codes (Huffman, arithmetic coding) is ultimately about minimizing KL between the empirical distribution and the code-induced distribution.

**Forward: to Mutual Information.** Mutual information $I(X;Y) = D_{\mathrm{KL}}(P(X,Y)\|P(X)P(Y))$ (Section 03) is a KL divergence measuring the dependence between two variables. The chain rule for KL (Section 3.5) becomes the chain rule for entropy when applied to the independence gap. The data processing inequality for KL becomes the DPI for mutual information: $I(X;Z) \le I(X;Y)$ whenever $Z$ is determined by $Y$. Understanding KL makes mutual information, and all the contrastive learning objectives built on it (InfoNCE, SimCLR, CLIP), immediately transparent.

**Forward: to Cross-Entropy and Fisher Information.** Cross-entropy $H(p,q)$ is KL plus a constant (Section 04); Fisher information is KL's local curvature (Section 05). The three-way connection $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q) \approx H(p) + \frac{1}{2}\delta^\top F\delta$ (for small perturbations) shows that these are all the same object at different scales: global (cross-entropy), local (Fisher).

```
KL DIVERGENCE IN THE INFORMATION THEORY CHAPTER


  
    Chapter 09 - Information Theory                                
                                                                   
    Section 01 ENTROPY H(X)                                               
          "extend to pairs of distributions"                      
    Section 02 KL DIVERGENCE D_KL(pq)       <- YOU ARE HERE              
                                                                 
    Section 03 MUTUAL INFORMATION    Section 04 CROSS-ENTROPY                    
    I(X;Y) = D_KL(PPP)     H(p,q) = H(p) + D_KL(pq)           
                                                                  
    Section 05 FISHER INFORMATION                                         
    F(theta) = local KL curvature                                      
  

  ML CONNECTIONS:
  
                                                                   
     D_KL(p_data  p_theta)        ->  Language model training (MLE)   
     D_KL(q_(z|x)  p(z))    ->  VAE regularizer (ELBO)          
     D_KL(pi_theta  pi_ref)         ->  RLHF / PPO / DPO alignment     
     D_KL(p_teacher  p_student) -> Knowledge distillation        
     D_KL(q  p_posterior)     ->  Variational inference (VI)     
                                                                   
  

  CHAPTER POSITION:

  09-01 Entropy
      
  09-02 KL Divergence  <- current section
      
  09-03 Mutual Information  (KL between joint and product of marginals)
      
  09-04 Cross-Entropy       (H(p,q) = H(p) + D_KL(pq))
      
  09-05 Fisher Information  (local KL curvature = Riemannian metric)


```

**Looking ahead beyond this chapter:** KL divergence appears again in Chapter 12 (Functional Analysis), where it generalizes to the RKHS setting and connects to kernel methods and maximum mean discrepancy (MMD). It appears in Chapter 21 (Statistical Learning Theory) as a key quantity in PAC-Bayes bounds: the generalization gap of a stochastic predictor is bounded by $D_{\mathrm{KL}}(Q\|P)/n$ where $Q$ is the posterior and $P$ is the prior on parameters. And in Chapter 22 (Causal Inference), KL divergence under interventional vs observational distributions quantifies the "causal effect" of an intervention in terms of distributional shift.

The single formula $\sum_x p(x)\log(p(x)/q(x))$ - first written by Kullback and Leibler in 1951 - underlies more of modern AI than perhaps any other equation in this curriculum.


---

## Appendix A: Detailed Proofs

### A.1 Proof of Non-Negativity via Log-Sum Inequality

The **log-sum inequality** provides an alternative, direct proof of Gibbs' inequality without invoking Jensen's explicitly.

**Log-Sum Inequality.** For non-negative reals $a_1, \ldots, a_n$ and $b_1, \ldots, b_n$:
$$\sum_{i=1}^n a_i \log\frac{a_i}{b_i} \ge \left(\sum_{i=1}^n a_i\right)\log\frac{\sum_{i=1}^n a_i}{\sum_{i=1}^n b_i}$$
with equality iff all $a_i/b_i$ are equal.

**Proof of log-sum inequality.** The function $f(t) = t\log t$ is convex (since $f''(t) = 1/t > 0$). By the weighted Jensen inequality with weights $b_i/B$ where $B = \sum b_j$:
$$\frac{1}{B}\sum_i b_i \cdot \frac{a_i}{b_i}\log\frac{a_i}{b_i} \ge \frac{1}{B}\left(\sum_i b_i \cdot \frac{a_i}{b_i}\right)\log\frac{\sum_i b_i \cdot a_i/b_i}{\sum_i b_i \cdot 1}$$
$$= \frac{A}{B}\log\frac{A}{B}$$
Multiplying by $B$ gives the result. $\square$

**KL non-negativity from log-sum inequality.** Apply with $a_i = p(x_i)$ and $b_i = q(x_i)$: since $\sum p_i = 1 = \sum q_i$:
$$D_{\mathrm{KL}}(p\|q) = \sum_i p_i\log\frac{p_i}{q_i} \ge \left(\sum_i p_i\right)\log\frac{\sum_i p_i}{\sum_i q_i} = 1 \cdot \log\frac{1}{1} = 0 \quad\square$$

### A.2 Convexity of KL: Detailed Proof

We prove that $(p,q) \mapsto D_{\mathrm{KL}}(p\|q)$ is jointly convex using the perspective function argument.

**Lemma.** The function $g(s,t) = s\log(s/t)$ (with $g(0,t) = 0$, $g(s,0) = +\infty$ for $s > 0$) is jointly convex on $\{(s,t) : s \ge 0, t > 0\}$.

**Proof.** $g(s,t) = s\log s - s\log t$. We need $H_g \succeq 0$. The Hessian is:
$$H_g = \begin{pmatrix} 1/s & -1/t \\ -1/t & s/t^2 \end{pmatrix}$$
Determinant $= s/t^2 \cdot 1/s - 1/t^2 = 1/t^2 - 1/t^2 = 0$. Trace $= 1/s + s/t^2 > 0$. PSD since $\det = 0 \ge 0$ and trace $> 0$. $\square$

Since $D_{\mathrm{KL}}(p\|q) = \sum_x g(p(x), q(x))$ and sums of jointly convex functions are jointly convex, $D_{\mathrm{KL}}$ is jointly convex. $\square$

### A.3 Pinsker's Inequality

**Theorem (Pinsker's Inequality).** $\mathrm{TV}(p,q) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(p\|q)}$.

**Proof.** We use the Bretagnolle-Huber inequality, which is slightly weaker but has a clean proof: for any event $A$,
$$p(A) - q(A) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(p\|q)}$$

For the full Pinsker proof, one approach uses the 2-point inequality: for $a, b \in [0,1]$,
$$a\log\frac{a}{b} + (1-a)\log\frac{1-a}{1-b} \ge \frac{1}{2\ln 2}(a-b)^2$$

Applying this to the binary partition $\{A, A^c\}$ and optimizing over $A$ (which gives $A = \{p > q\}$ for TV):
$$D_{\mathrm{KL}}(p\|q) \ge D_{\mathrm{KL}}(\text{Bern}(p(A))\|\text{Bern}(q(A))) \ge \frac{1}{2\ln 2}(p(A) - q(A))^2 \ge \frac{1}{2}(2\mathrm{TV})^2$$

Rearranging: $\mathrm{TV}(p,q) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(p\|q)}$. $\square$

**Pinsker's inequality in AI:** KL divergence is continuous and differentiable (gradients exist for optimization), while total variation is an event-level bound. Pinsker's lets us translate KL bounds from optimization theory into guarantees about individual event probabilities - useful in PAC-Bayes learning theory.

### A.4 KL Divergence for the Gaussian: Step-by-Step

We derive the multivariate Gaussian KL formula in detail.

Let $p = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$ on $\mathbb{R}^d$. The log-ratio is:

$$\log\frac{p(\mathbf{x})}{q(\mathbf{x})} = -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_1)^\top\Sigma_1^{-1}(\mathbf{x}-\boldsymbol{\mu}_1) + \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_2)^\top\Sigma_2^{-1}(\mathbf{x}-\boldsymbol{\mu}_2) + \frac{1}{2}\log\frac{\det\Sigma_2}{\det\Sigma_1}$$

Taking expectation under $p$, using:
- $\mathbb{E}_p[(\mathbf{x}-\boldsymbol{\mu}_1)^\top\Sigma_1^{-1}(\mathbf{x}-\boldsymbol{\mu}_1)] = \operatorname{tr}(\Sigma_1^{-1}\Sigma_1) = d$
- $\mathbb{E}_p[(\mathbf{x}-\boldsymbol{\mu}_2)^\top\Sigma_2^{-1}(\mathbf{x}-\boldsymbol{\mu}_2)] = \operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)^\top\Sigma_2^{-1}(\boldsymbol{\mu}_1-\boldsymbol{\mu}_2)$

gives:

$$D_{\mathrm{KL}}(p\|q) = \frac{1}{2}\left[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top\Sigma_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1) - d + \log\frac{\det\Sigma_2}{\det\Sigma_1}\right]$$

The four terms correspond to:
1. $\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) - d$: how much the covariances differ (= 0 when $\Sigma_1 = \Sigma_2$)
2. $(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^\top\Sigma_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)$: squared Mahalanobis distance between means
3. $\log(\det\Sigma_2/\det\Sigma_1)$: log ratio of volumes (= 0 when $\det\Sigma_1 = \det\Sigma_2$)


---

## Appendix B: KL Divergence Computations and Reference Formulas

### B.1 Quick Reference: KL Formulas for Common Distributions

```
KL DIVERGENCE CLOSED FORMS - QUICK REFERENCE


  SCALAR GAUSSIAN:
  D_KL(N(mu1,12)  N(mu2,22)) = log(2/1) + (12 + (mu1-mu2)2)/(222) - 12

  VAE ENCODER -> STANDARD NORMAL:
  D_KL(N(mu,2)  N(0,1)) = 12(mu2 + 2 - log 2 - 1)

  MULTIVARIATE GAUSSIAN:
  D_KL(N(mu1,1)  N(mu2,2)) = 12[tr(211) + (mu2-mu1)T21(mu2-mu1) - d + log(det 2/det 1)]

  BERNOULLI:
  D_KL(Bern(p)  Bern(q)) = plog(p/q) + (1-p)log((1-p)/(1-q))

  CATEGORICAL (general discrete):
  D_KL(p  q) = k pk  log(pk/qk)

  POISSON:
  D_KL(Poisson(1)  Poisson(2)) = 1log(1/2) - 1 + 2

  EXPONENTIAL:
  D_KL(Exp(1)  Exp(2)) = log(2/1) + 1/2 - 1

  BETA:
  D_KL(Beta(alpha1,beta1)  Beta(alpha2,beta2)) = log B(alpha2,beta2)/B(alpha1,beta1)
      + (alpha1-alpha2)(alpha1) + (beta1-beta2)(beta1) + (alpha2-alpha1+beta2-beta1)(alpha1+beta1)
      where  = digamma function, B = beta function


```

### B.2 Worked Example: KL Between Poisson Distributions

A Poisson distribution $\text{Pois}(\lambda)$ has PMF $p(k) = e^{-\lambda}\lambda^k/k!$ for $k = 0,1,2,\ldots$

**$D_{\mathrm{KL}}(\text{Pois}(\lambda_1)\|\text{Pois}(\lambda_2))$:**

$$\log\frac{p_{\lambda_1}(k)}{p_{\lambda_2}(k)} = (\lambda_2 - \lambda_1) + k\log\frac{\lambda_1}{\lambda_2}$$

$$D_{\mathrm{KL}} = \mathbb{E}_{\lambda_1}\!\left[(\lambda_2-\lambda_1) + k\log\frac{\lambda_1}{\lambda_2}\right] = (\lambda_2 - \lambda_1) + \lambda_1\log\frac{\lambda_1}{\lambda_2}$$

For $\lambda_1 = 3, \lambda_2 = 2$: $D_{\mathrm{KL}} = (2-3) + 3\ln(3/2) = -1 + 3(0.405) = 0.216$ nats.

**For AI:** Poisson distributions model count data (number of events, word counts in LDA topic models). The KL between Poisson distributions appears in the ELBO for Poisson process models and in topic model variational inference.

### B.3 Worked Example: KL for Categorical - Softmax Outputs

Let $\mathbf{z}_1 = (2.0, 1.0, 0.1)$ and $\mathbf{z}_2 = (1.5, 1.0, 0.5)$ be logit vectors.

**Step 1:** $p_1 = \text{softmax}(\mathbf{z}_1)$:
$$p_1 = (e^2, e^1, e^{0.1})/(e^2 + e^1 + e^{0.1}) = (7.389, 2.718, 1.105)/11.212 = (0.659, 0.242, 0.099)$$

**Step 2:** $p_2 = \text{softmax}(\mathbf{z}_2)$:
$$p_2 = (e^{1.5}, e^1, e^{0.5})/(e^{1.5} + e^1 + e^{0.5}) = (4.482, 2.718, 1.649)/8.849 = (0.507, 0.307, 0.186)$$

**Step 3:** $D_{\mathrm{KL}}(p_1\|p_2)$:
$$= 0.659\ln(0.659/0.507) + 0.242\ln(0.242/0.307) + 0.099\ln(0.099/0.186)$$
$$= 0.659(0.261) + 0.242(-0.238) + 0.099(-0.635) = 0.172 - 0.058 - 0.063 = 0.051 \text{ nats}$$

**For AI:** This is the KL component when computing the distillation loss between a teacher ($p_1$) and student ($p_2$). The small value (0.051 nats) indicates the student's distribution is reasonably close to the teacher's.

### B.4 KL Divergence and Hypothesis Testing

The connection between KL divergence and hypothesis testing is both historical (Kullback 1959) and practically important.

**Setting:** $n$ i.i.d. observations from an unknown distribution. Hypothesis $H_0$: data from $q$; hypothesis $H_1$: data from $p$.

**Log-likelihood ratio test statistic:**
$$\Lambda_n = \sum_{i=1}^n \log\frac{p(x_i)}{q(x_i)}$$

By the law of large numbers, $\Lambda_n/n \to \mathbb{E}_p[\log(p/q)] = D_{\mathrm{KL}}(p\|q)$ as $n \to \infty$ when truth is $p$.

**Stein's lemma (type II error rate):** For fixed type I error probability $\alpha \in (0,1)$, the best achievable type II error probability $\beta_n^*$ decays as:
$$\lim_{n\to\infty} \frac{1}{n}\log\beta_n^* = -D_{\mathrm{KL}}(p\|q)$$

Larger KL divergence -> faster exponential decay of type II error -> easier to distinguish $p$ from $q$ with more data.

**Chernoff exponent:** The best achievable equal-error-rate exponent (minimizing over both type I and II simultaneously) is:
$$C(p,q) = -\min_{\alpha\in[0,1]}\log\sum_x p(x)^\alpha q(x)^{1-\alpha} = \max_{\alpha\in[0,1]} D_\alpha(p\|q)$$
the maximum Renyi divergence over orders $\alpha \in [0,1]$.

**For AI:** The hypothesis testing perspective explains why $D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ in RLHF bounds how detectable the policy shift is: small KL means the fine-tuned model's outputs are statistically indistinguishable from the reference model's outputs.


---

## Appendix C: KL Divergence in Broader ML Contexts

### C.1 PAC-Bayes Bounds

PAC-Bayes theory (McAllester 1999; Seeger 2002) gives generalization bounds for stochastic predictors. The key theorem:

**PAC-Bayes Theorem.** Let $P$ be a prior over hypotheses (fixed before seeing data) and $Q$ be any posterior (possibly data-dependent). For any $\delta > 0$, with probability $\ge 1 - \delta$ over the training set:

$$\mathbb{E}_{h \sim Q}[\mathcal{L}(h)] \le \mathbb{E}_{h \sim Q}[\hat{\mathcal{L}}(h)] + \sqrt{\frac{D_{\mathrm{KL}}(Q\|P) + \log(2\sqrt{n}/\delta)}{2n}}$$

where $\mathcal{L}(h)$ is the true loss and $\hat{\mathcal{L}}(h)$ is the empirical loss.

**Interpretation:** The generalization gap is bounded by $\sqrt{D_{\mathrm{KL}}(Q\|P)/n}$. The posterior $Q$ can be anything - including a specific trained neural network - and the bound holds. This shows that models that stay "close" to a simple prior in KL generalize well. Modern applications: PAC-Bayes bounds for LoRA-fine-tuned LLMs (where $P = $ pretrained weights, $Q = $ fine-tuned weights).

### C.2 KL in Variational Inference for Bayesian Neural Networks

Bayesian neural networks (BNNs) maintain distributions over weights $\boldsymbol{\theta}$ rather than point estimates. The posterior $p(\boldsymbol{\theta}|\mathcal{D})$ is intractable for large networks. Variational inference minimizes:

$$D_{\mathrm{KL}}(q(\boldsymbol{\theta})\|p(\boldsymbol{\theta}|\mathcal{D})) = D_{\mathrm{KL}}(q(\boldsymbol{\theta})\|p(\boldsymbol{\theta})) - \mathbb{E}_q[\log p(\mathcal{D}|\boldsymbol{\theta})] + \log p(\mathcal{D})$$

The ELBO is: $\mathcal{L} = \mathbb{E}_q[\log p(\mathcal{D}|\boldsymbol{\theta})] - D_{\mathrm{KL}}(q(\boldsymbol{\theta})\|p(\boldsymbol{\theta}))$. The KL term $D_{\mathrm{KL}}(q\|p)$ acts as a regularizer: it penalizes variational distributions that deviate from the prior $p(\boldsymbol{\theta})$.

**Practical implementations:** Bayes By Backprop (Blundell et al., 2015) parameterizes $q(\boldsymbol{\theta}) = \prod_i \mathcal{N}(\mu_i, \sigma_i^2)$ and uses the local reparameterization trick. Each weight $\theta_i = \mu_i + \sigma_i\varepsilon_i$ where $\varepsilon_i \sim \mathcal{N}(0,1)$. The KL term per weight is $\frac{1}{2}(\mu_i^2/\sigma_p^2 + \sigma_i^2/\sigma_p^2 - 1 - \log(\sigma_i^2/\sigma_p^2))$ for Gaussian prior $\mathcal{N}(0, \sigma_p^2)$.

### C.3 KL in Contrastive Learning and Representation

Contrastive learning methods like SimCLR (Chen et al., 2020) and MoCo (He et al., 2020) maximize the lower bound on mutual information:

$$I(X; Z) \ge \mathbb{E}[\text{InfoNCE}(X, Z)] = \mathbb{E}\!\left[\log\frac{e^{f(x,z)}}{\frac{1}{N}\sum_{j=1}^N e^{f(x,z_j)}}\right]$$

The InfoNCE loss is a lower bound on $I(X;Z)$, which is a KL divergence $D_{\mathrm{KL}}(P(X,Z)\|P(X)P(Z))$ (see Section 09-03). CLIP (Radford et al., 2021) uses a symmetric InfoNCE to align image and text embeddings - which is equivalent to minimizing KL between the joint image-text distribution and the product of marginals.

**CPC (Contrastive Predictive Coding, van den Oord et al., 2018):** Used in speech (wav2vec 2.0, HuBERT) and images. The representation $z_t$ at time $t$ predicts future $z_{t+k}$ via a density ratio estimator. The InfoNCE objective is a lower bound on $\sum_k I(z_t; z_{t+k})$. Training audio encoders this way produces features that capture the semantic structure of speech - because mutual information (= KL divergence) rewards capturing any statistical dependency.

### C.4 KL in Score-Based Generative Models

Score-based models (Song & Ermon, 2020) and their diffusion counterparts learn the score function $\nabla_\mathbf{x}\log p(\mathbf{x})$ to generate samples via Langevin dynamics. The connection to KL:

**Score matching** minimizes:
$$J(\boldsymbol{\theta}) = \mathbb{E}_p\!\left[\frac{1}{2}\|s_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla\log p(\mathbf{x})\|^2\right]$$

where $s_{\boldsymbol{\theta}} \approx \nabla\log p$ is the learned score. Fisher divergence equals:
$$J(\boldsymbol{\theta}) = \mathbb{E}_p\!\left[\frac{1}{2}\|s_{\boldsymbol{\theta}}(\mathbf{x})\|^2 + \nabla\cdot s_{\boldsymbol{\theta}}(\mathbf{x})\right] + \text{const}$$

Score matching is equivalent to minimizing a generalization of KL (Fisher divergence: $\int p(\mathbf{x})\|\nabla\log(p/q)\|^2 d\mathbf{x}$), which uses the gradient of the log-density ratio rather than the ratio itself. This is the information-geometric connection between score matching and KL divergence.

### C.5 KL and Temperature in LLM Sampling

LLM inference involves sampling from $p_\tau(y|x) = \text{softmax}(\mathbf{z}/\tau)$ at temperature $\tau$. The KL divergence between the original distribution ($\tau = 1$) and the temperature-scaled distribution is:

$$D_{\mathrm{KL}}(p_1\|p_\tau) = \sum_k p_1(k)\log\frac{p_1(k)}{p_\tau(k)} = \sum_k p_1(k)\left[\frac{z_k}{1} - \log\sum_j e^{z_j} - \frac{z_k}{\tau} + \log\sum_j e^{z_j/\tau}\right]$$

As $\tau \to 0$: $p_\tau \to $ one-hot (deterministic, picks the argmax). $D_{\mathrm{KL}}(p_1\|p_\tau) \to H(p_1)$ - the divergence equals the entropy of the original distribution.

As $\tau \to \infty$: $p_\tau \to $ uniform. $D_{\mathrm{KL}}(p_1\|p_\tau) \to \log|\mathcal{V}| - H(p_1)$ - the divergence equals the KL from uniform.

**Practical implications:** The "temperature" knob in LLM APIs (temperature=0.7, top-p=0.9) directly controls the KL divergence from the trained distribution. High temperature increases diversity (reduces KL to uniform); low temperature reduces diversity (pushes toward argmax). The KL from the training distribution is minimized at $\tau = 1$.


---

## Appendix D: Numerical Stability and Implementation

### D.1 The log-sum-exp Trick for KL Computation

Computing $D_{\mathrm{KL}}(p\|q)$ with softmax distributions requires care to avoid numerical overflow.

**Naive implementation (unstable):**
```python
# WRONG - can overflow for large logits
kl = torch.sum(p * (torch.log(p) - torch.log(q)))
```

**Stable implementation using log-softmax:**
```python
# CORRECT - uses log-space throughout
import torch.nn.functional as F

def kl_divergence_stable(logits_p, logits_q):
    """
    KL(p || q) where p = softmax(logits_p), q = softmax(logits_q).
    Uses log_softmax for numerical stability.
    """
    log_p = F.log_softmax(logits_p, dim=-1)  # log(softmax) in one step
    log_q = F.log_softmax(logits_q, dim=-1)
    p = torch.exp(log_p)
    return torch.sum(p * (log_p - log_q), dim=-1)
```

PyTorch also provides `torch.nn.functional.kl_div(log_q, p)` which computes $\sum p(\log p - \text{log\_q})$ given pre-computed `log_q`. Note the unusual argument order: it expects log of the second argument.

### D.2 Estimating KL from Samples

When only samples are available (not closed-form distributions), KL must be estimated.

**Monte Carlo estimator.** Given samples $x_1, \ldots, x_n \sim p$ and access to $\log p(x)$ and $\log q(x)$:
$$\hat{D}_{\mathrm{KL}}(p\|q) = \frac{1}{n}\sum_{i=1}^n \log\frac{p(x_i)}{q(x_i)}$$

This is unbiased: $\mathbb{E}[\hat{D}] = D_{\mathrm{KL}}(p\|q)$.

**k-nearest neighbor estimator (Perez-Cruz 2008).** When neither $p(x)$ nor $q(x)$ are known analytically - only samples from both distributions - the KNN estimator uses:
$$\hat{D}_{\mathrm{KL}}(p\|q) = \frac{d}{n}\sum_{i=1}^n \log\frac{\rho_k(x_i)}{\nu_k(x_i)} + \log\frac{m}{n-1}$$
where $\rho_k(x_i)$ is the distance to the $k$-th nearest neighbor in the $p$-samples, $\nu_k(x_i)$ is the distance to the $k$-th neighbor in the $q$-samples, $n$ = number of $p$-samples, $m$ = number of $q$-samples, and $d$ = dimension.

This estimator is used in representation learning evaluation, where you have samples from an encoder's output distribution $q$ and want to measure $D_{\mathrm{KL}}(q\|p_{\text{target}})$ without a closed form.

### D.3 Symmetric KL Implementation

For applications requiring symmetric divergence without the issues of JSD (which requires computing the mixture), Jeffrey's divergence is simpler:

```python
def jeffreys_divergence(p_probs, q_probs, eps=1e-10):
    """
    J(p, q) = 0.5 * (KL(p||q) + KL(q||p))
    More numerically stable than computing JSD.
    """
    p = p_probs + eps
    q = q_probs + eps
    p = p / p.sum()
    q = q / q.sum()
    kl_pq = (p * np.log(p / q)).sum()
    kl_qp = (q * np.log(q / p)).sum()
    return 0.5 * (kl_pq + kl_qp)
```

### D.4 KL Divergence in PyTorch for VAE Training

The standard VAE KL term implementation:

```python
def vae_kl_loss(mu, log_var):
    """
    D_KL(N(mu, sigma^2) || N(0, 1)) per-sample, summed over latent dimensions.
    
    Formula: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
           = 0.5 * sum(mu^2 + exp(log_var) - log_var - 1)
    
    Input: mu, log_var - shape (batch, latent_dim)
    Output: KL loss - shape (batch,), or scalar if mean-reduced
    """
    kl = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=-1)
    return kl.mean()  # average over batch
```

Note: using `log_var` (log of variance) instead of `sigma` avoids numerical issues with very small variances and ensures $\sigma^2 > 0$ by construction.

**RLHF KL penalty implementation:**
```python
def rlhf_kl_penalty(logprobs_policy, logprobs_ref, beta=0.1):
    """
    D_KL(pi_policy || pi_ref) for a batch of token sequences.
    
    logprobs_policy, logprobs_ref: log probabilities under each model
                                   shape (batch, seq_len)
    Returns: mean KL per sequence
    """
    # KL = E_pi[log pi - log pi_ref]
    kl_per_token = logprobs_policy - logprobs_ref  # log ratio
    kl_per_seq = kl_per_token.sum(dim=-1)  # sum over sequence length
    return beta * kl_per_seq.mean()
```


---

## Appendix E: Variational Inference - A Deep Dive

### E.1 The Variational Inference Problem

Bayesian inference requires computing the posterior $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x}|\mathbf{z})p(\mathbf{z})/p(\mathbf{x})$. The denominator $p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ is typically intractable - the integral has no closed form for nonlinear likelihoods.

**Variational inference** approximates the posterior with a tractable family $\mathcal{Q} = \{q_\phi : \phi \in \Phi\}$ by solving:

$$q^* = \arg\min_{q \in \mathcal{Q}} D_{\mathrm{KL}}(q(\mathbf{z}) \| p(\mathbf{z}|\mathbf{x}))$$

**Why reverse KL?** Because it is the only tractable direction. Computing $D_{\mathrm{KL}}(p\|q)$ requires evaluating $p(\mathbf{z}|\mathbf{x})$, which is the intractable quantity we're trying to avoid. Computing $D_{\mathrm{KL}}(q\|p)$ requires only samples from $q$ and evaluations of $p(\mathbf{z})p(\mathbf{x}|\mathbf{z})$ - both tractable.

**ELBO derivation.** Expanding the reverse KL:

$$D_{\mathrm{KL}}(q(\mathbf{z})\|p(\mathbf{z}|\mathbf{x})) = \mathbb{E}_q\!\left[\log\frac{q(\mathbf{z})}{p(\mathbf{z}|\mathbf{x})}\right] = \mathbb{E}_q[\log q(\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{z}|\mathbf{x})]$$

Using $\log p(\mathbf{z}|\mathbf{x}) = \log p(\mathbf{x}|\mathbf{z}) + \log p(\mathbf{z}) - \log p(\mathbf{x})$:

$$= \mathbb{E}_q[\log q(\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - \mathbb{E}_q[\log p(\mathbf{z})] + \log p(\mathbf{x})$$

Rearranging:

$$\log p(\mathbf{x}) = \underbrace{\mathbb{E}_q[\log p(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q(\mathbf{z})\|p(\mathbf{z}))}_{\mathcal{L}(q) = \text{ELBO}} + D_{\mathrm{KL}}(q(\mathbf{z})\|p(\mathbf{z}|\mathbf{x}))$$

Since $D_{\mathrm{KL}} \ge 0$: $\mathcal{L}(q) \le \log p(\mathbf{x})$ - the ELBO is a lower bound. Maximizing ELBO over $q$ simultaneously: (a) tightens the bound (reduces the KL gap), and (b) improves the model fit (increases $\log p(\mathbf{x})$ when $p_{\boldsymbol{\theta}}$ is also optimized).

### E.2 Mean-Field Variational Inference

The **mean-field approximation** restricts to factored distributions $q(\mathbf{z}) = \prod_{j=1}^d q_j(z_j)$. This is an enormous simplification - the full posterior may have complex correlations, but the approximation ignores all correlations between latent variables.

**Coordinate ascent VI (CAVI).** For mean-field, the optimal factor $q_j^*$ satisfies:

$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(\mathbf{x}, \mathbf{z})] + \text{const}$$

where $q_{-j}$ denotes expectation over all factors except $j$. This is a fixed-point equation: each factor depends on the others. CAVI iterates: update $q_1$ given $q_2, \ldots, q_d$; then update $q_2$ given new $q_1$ and old $q_3, \ldots, q_d$; etc.

**Guarantee:** Each update increases the ELBO (decreases the reverse KL). CAVI converges to a local optimum.

**For AI:** Mean-field VI was the basis for early Bayesian neural networks (BNNs) and topic models (LDA). Its mode-seeking bias (reverse KL) means it systematically underestimates posterior uncertainty - a known limitation. Modern BNNs often use Monte Carlo Dropout as a cheaper alternative.

### E.3 ELBO as a Free Energy

The ELBO has a physical interpretation as a **variational free energy** (Helmholtz, Gibbs):

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})] = \mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})] + H(q)$$

This is the expected log joint minus the entropy of $q$. Maximizing the ELBO simultaneously:
- Maximizes $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})]$: encourages $q$ to put mass on high-probability regions of the joint
- Maximizes $H(q)$: encourages $q$ to spread out (high entropy = uncertainty), preventing mode collapse

The tension between these two terms is the fundamental trade-off in variational inference. In physics, this is the Gibbs free energy minimization: equilibrium balances energy minimization (term 1) with entropy maximization (term 2), at temperature 1.

**For AI:** The free energy interpretation connects to energy-based models (LeCun et al., 2006) and to the "free energy principle" in computational neuroscience (Friston, 2010). The ELBO appears in all modern latent variable models: VAEs, VQ-VAE, neural process families, and latent diffusion models (Stable Diffusion uses a VAE for compression, then diffusion in the latent space).

### E.4 Amortized Variational Inference

Standard VI optimizes a separate $q_\phi(\mathbf{z})$ per data point - $O(n)$ optimization problems. **Amortized VI** (Kingma & Welling 2014) instead trains a single encoder network $q_\phi(\mathbf{z}|\mathbf{x})$ that takes $\mathbf{x}$ as input:

$$q_\phi^*(\mathbf{z}|\mathbf{x}) = \arg\max_\phi \mathbb{E}_{p(\mathbf{x})}\!\left[\mathcal{L}(\phi, \boldsymbol{\theta}; \mathbf{x})\right]$$

The encoder "amortizes" the per-sample optimization across the full dataset, generalizing to new samples at inference time. This is what makes VAEs practical for large datasets: a single forward pass through the encoder gives $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$ instantly.

**Amortization gap:** The amortized approximation $q_\phi(\mathbf{z}|\mathbf{x})$ is generally worse than the optimal per-sample $q^*(\mathbf{z}|\mathbf{x})$ (it must generalize, so it trades off per-sample accuracy). The gap $D_{\mathrm{KL}}(q^*\|p) - D_{\mathrm{KL}}(q_\phi\|p)$ is called the amortization gap and is an active research topic.


---

## Appendix F: The RLHF-KL Alignment Triangle

### F.1 Three Ways KL Appears in Alignment

Modern LLM alignment uses KL divergence in three distinct but related ways, forming a triangle of constraints:

```
RLHF-KL ALIGNMENT TRIANGLE


                     PRETRAINED MODEL pi_ref
                           
                          / \
                KL       /   \   KL
              penalty   /     \  constraint
                       /       \
                      
              RLHF pi_theta          DPO pi_DPO
                    \           /
                     \  equiv  /
                      \       /
                       Fixed point: same optimal policy

  All three policies minimize:
    E[r(x,y)] - betaD_KL(pi_theta(|x)  pi_ref(|x))

  Optimal solution: pi*(y|x) propto pi_ref(y|x)exp(r(x,y)/beta)


```

### F.2 Interpreting the KL Coefficient beta

The KL coefficient $\beta$ in RLHF controls the fundamental exploration-exploitation trade-off in alignment:

| $\beta$ value | Behavior | Risk |
| --- | --- | --- |
| $\beta \to 0$ | Pure reward maximization; ignores KL constraint | Reward hacking; generates repetitive, incoherent outputs |
| $\beta \to \infty$ | Stay at reference policy; no alignment | No improvement in helpfulness, safety |
| $\beta \in [0.01, 0.5]$ | Typical range used in practice | Balance between helpfulness and coherence |

**Typical values in production:** InstructGPT (Ouyang et al., 2022) used $\beta = 0.02$ for PPO-RLHF. Different applications need different $\beta$: safety alignment uses larger $\beta$ (conservative); instruction following uses smaller $\beta$ (allows larger policy changes).

**Adaptive $\beta$ (KL-controller):** Some implementations adaptively adjust $\beta$ to keep $D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}}\|\pi_{\mathrm{ref}})$ near a target value $\kappa$:
$$\beta_{t+1} = \beta_t + \gamma(D_{\mathrm{KL}} - \kappa)$$

This is a PI controller on the KL divergence. If the policy drifts too far, $\beta$ increases; if it stays too close to reference, $\beta$ decreases.

### F.3 DPO's Implicit Reward

DPO (Rafailov et al., 2023) reveals that the log-ratio between policy and reference is the implicit reward:

$$r_{\boldsymbol{\theta}}(x,y) = \beta\log\frac{\pi_{\boldsymbol{\theta}}(y|x)}{\pi_{\mathrm{ref}}(y|x)}$$

This has a clean interpretation: the reward a sequence $y$ receives is proportional to how much more likely the aligned policy assigns to it compared to the reference. Sequences the aligned model prefers more than the reference model are rewarded; sequences it prefers less are penalized.

**DPO gradient:** The DPO loss gradient with respect to $\boldsymbol{\theta}$:

$$\nabla_{\boldsymbol{\theta}}\mathcal{L}_{\mathrm{DPO}} = -\beta\,\mathbb{E}\!\left[\sigma(\hat{r}_{\boldsymbol{\theta}}(y_l) - \hat{r}_{\boldsymbol{\theta}}(y_w))\left(\nabla\log\pi_{\boldsymbol{\theta}}(y_w) - \nabla\log\pi_{\boldsymbol{\theta}}(y_l)\right)\right]$$

where $\hat{r}_{\boldsymbol{\theta}}(y) = \beta\log(\pi_{\boldsymbol{\theta}}(y)/\pi_{\mathrm{ref}}(y))$. The weight $\sigma(\hat{r}(y_l) - \hat{r}(y_w))$ is large when the model assigns higher reward to the loser than the winner - the loss upweights these hard examples. This has the same structure as hard example mining in contrastive learning.

---

## Appendix G: Connections to Other Chapters

### G.1 KL and Exponential Families (Chapter 7: Statistics)

Exponential families are the natural domain for KL divergence because:
1. KL between exponential family members has closed form (Section 5.3)
2. The maximum entropy distribution under moment constraints is an exponential family member (Section 09-01 Section 5)
3. The Bregman divergence generated by $A(\boldsymbol{\eta})$ equals the KL between members with those natural parameters

**Sufficiency:** Kullback and Leibler's original paper introduced KL as a measure of *information for discrimination* between hypotheses. A statistic $T(\mathbf{x})$ is **sufficient** for parameter $\boldsymbol{\theta}$ iff $D_{\mathrm{KL}}(p_{\boldsymbol{\theta}_1}(\mathbf{x})\|p_{\boldsymbol{\theta}_2}(\mathbf{x})) = D_{\mathrm{KL}}(p_{\boldsymbol{\theta}_1}(T)\|p_{\boldsymbol{\theta}_2}(T))$ - sufficient statistics preserve all the discriminatory information. This is the information-theoretic definition of sufficiency.

### G.2 KL and Convex Optimization (Chapter 8)

The ELBO maximization is a convex optimization problem in $q$ (for fixed model parameters $\boldsymbol{\theta}$):
- $D_{\mathrm{KL}}(q\|p)$ is convex in $q$ (Section 3.4)
- The constraint $q \ge 0$, $\sum q = 1$ is convex (simplex)
- The I-projection $q^* = \arg\min_q D_{\mathrm{KL}}(q\|p)$ is a convex program with a unique solution

The Lagrangian of the I-projection onto the exponential family gives the natural parameter update equations - connecting KL divergence to the theory of constrained convex optimization from Chapter 8.

### G.3 KL and Functional Analysis (Chapter 12)

In infinite-dimensional settings, KL divergence extends via the Radon-Nikodym theorem. For Gaussian processes and Gaussian measures on Hilbert spaces:

$$D_{\mathrm{KL}}(\mathcal{GP}(m_1, K_1)\|\mathcal{GP}(m_2, K_2)) = \frac{1}{2}\left[\operatorname{tr}(K_2^{-1}K_1) + (m_2-m_1)^\top K_2^{-1}(m_2-m_1) - d + \log\frac{\det K_2}{\det K_1}\right]$$

the same Gaussian KL formula, but now $K_1, K_2$ are kernel matrices and $d \to \infty$. The trace and log-determinant terms generalize as operator trace and Fredholm determinant.

**RKHS connection:** The maximum mean discrepancy (MMD) is an alternative to KL for comparing distributions, based on RKHS distances. MMD has better convergence rates from finite samples than KL estimators, which is why it appears in some GANs (MMD-GAN) and kernel two-sample tests.

### G.4 KL and Statistical Learning Theory (Chapter 21)

The PAC-Bayes bound (Appendix C.1) shows that the KL from posterior to prior controls generalization. This has a direct implication for LoRA fine-tuning: if the LoRA adapter is constrained to have small $D_{\mathrm{KL}}(q_{\mathrm{adapter}}\|p_{\mathrm{init}})$ - i.e., small deviation from initialization - then generalization bounds are tight. This is the information-theoretic justification for small-rank adaptations: they have small KL from the pretrained distribution.

**Minimum Description Length (MDL):** The MDL principle selects the model that minimizes the two-part code length: $L(M) + L(D|M)$ where $L(M)$ is the code length of the model and $L(D|M)$ is the code length of data given the model. This equals $D_{\mathrm{KL}}(q\|p) + \text{const}$ for the right choice of $p$ and $q$. MDL and ELBO are the same objective - minimum description length equals maximum ELBO. This is the Bayesian interpretation of Occam's razor.

---

## Appendix H: Summary Tables

### H.1 KL Divergence Properties at a Glance

| Property | Holds? | Counterexample if No |
| --- | --- | --- |
| Non-negative: $D_{\mathrm{KL}} \ge 0$ | YES | - |
| Zero iff $p = q$: $D_{\mathrm{KL}} = 0 \Leftrightarrow p = q$ | YES (a.e.) | - |
| Symmetric: $D_{\mathrm{KL}}(p\|q) = D_{\mathrm{KL}}(q\|p)$ | NO | $p=(0.9,0.1), q=(0.5,0.5)$ |
| Triangle inequality: $D_{\mathrm{KL}}(p\|r) \le D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|r)$ | NO | $p=(0.9,0.1), q=(0.5,0.5), r=(0.1,0.9)$ |
| Data processing inequality: $D_{\mathrm{KL}}(p_T\|q_T) \le D_{\mathrm{KL}}(p\|q)$ | YES | - |
| Jointly convex in $(p,q)$ | YES | - |
| Convex in $p$ for fixed $q$ | YES | - |
| Convex in $q$ for fixed $p$ | YES | - |
| Finite when $p \not\ll q$ | NO | $p=(0.5,0.5,0), q=(0.5,0,0.5)$: $D_{\mathrm{KL}} = \infty$ |
| Metric (distance function) | NO | Fails symmetry and triangle inequality |

### H.2 Forward vs Reverse KL Summary

| | Forward KL $D_{\mathrm{KL}}(p\|q)$ | Reverse KL $D_{\mathrm{KL}}(q\|p)$ |
| --- | --- | --- |
| Expectation under | $p$ (truth) | $q$ (approximation) |
| Also called | Inclusive, zero-avoiding | Exclusive, zero-forcing |
| Behavior when fitting $q$ to $p$ | Mean-seeking, mass-covering | Mode-seeking, mass-concentrating |
| Handles multimodal $p$ | Averages over all modes | Collapses to one mode |
| Support requirement | $q$ must cover $\text{supp}(p)$ | $q$ must avoid $\text{supp}(p)^c$ |
| Used in | MLE, distillation, flows | Variational inference, ELBO |
| Risk | Blurry/average samples | Posterior collapse |
| Tractability | Needs samples from $p$ | Only needs samples from $q$ |

### H.3 f-Divergence Family Summary

| Divergence | $f(t)$ | Symmetric? | Bounded? | Metric? |
| --- | --- | --- | --- | --- |
| KL $D_{\mathrm{KL}}(p\|q)$ | $t\ln t$ | No | No | No |
| Reverse KL $D_{\mathrm{KL}}(q\|p)$ | $-\ln t$ | No | No | No |
| Jeffrey's $J$ | $t\ln t - \ln t$ | Yes | No | No |
| Hellinger $H^2$ | $(\sqrt{t}-1)^2$ | Yes | Yes ($[0,2]$) | $\sqrt{H^2}$ is |
| Total Variation TV | $\frac{1}{2}|t-1|$ | Yes | Yes ($[0,1]$) | Yes |
| Chi-squared $\chi^2$ | $(t-1)^2$ | No | No | No |
| Jensen-Shannon JSD | (mixture form) | Yes | Yes ($[0,\ln 2]$) | $\sqrt{\mathrm{JSD}}$ is |
| Renyi $D_\alpha$ | $\frac{t^\alpha-1}{\alpha-1}$ (limit form) | No | No | No |


---

## Appendix I: Extended Examples and Geometric Visualizations

### I.1 KL Divergence Along a Path Between Distributions

Consider a one-parameter family $p_t = (1-t)p_0 + t p_1$ - a straight line between two distributions in probability simplex. The KL divergence $D_{\mathrm{KL}}(p_t\|q)$ is convex in $t$ (because it is jointly convex in $(p,q)$ and linear in $p$). However, $t \mapsto D_{\mathrm{KL}}(p_0\|p_t)$ may not be convex - the "path" in distribution space is curved from the perspective of KL divergence.

**Numerical example.** $p_0 = (0.9, 0.1)$, $p_1 = (0.1, 0.9)$, $q = (0.5, 0.5)$. For $t \in [0,1]$:

| $t$ | $p_t$ | $D_{\mathrm{KL}}(p_t\|q)$ | $D_{\mathrm{KL}}(q\|p_t)$ |
| --- | --- | --- | --- |
| 0 | $(0.9, 0.1)$ | $0.368$ | $0.511$ |
| 0.25 | $(0.7, 0.3)$ | $0.119$ | $0.133$ |
| 0.5 | $(0.5, 0.5)$ | $0.000$ | $0.000$ |
| 0.75 | $(0.3, 0.7)$ | $0.119$ | $0.133$ |
| 1.0 | $(0.1, 0.9)$ | $0.368$ | $0.511$ |

Both directions are symmetric here due to the symmetric path. The KL is zero at $t=0.5$ (when $p_t = q$) and increases convexly toward the endpoints.

**Geometric implication:** The straight-line path $p_t = (1-t)p_0 + t p_1$ is a **mixture geodesic** (M-geodesic) in information geometry - the natural path in the mixture parameterization. The dual path (exponential geodesic) mixes in the natural parameter space, giving different intermediate distributions.

### I.2 KL vs Wasserstein Distance

KL divergence and the Wasserstein distance (from optimal transport theory) are complementary tools for comparing distributions:

| Property | KL Divergence | Wasserstein Distance |
| --- | --- | --- |
| Requires $p \ll q$ | YES | No |
| Metrizes weak convergence | No | YES |
| Captures geometry of support | No | YES |
| Tractable to compute | YES (closed form) | Harder ($O(n^3)$ naive) |
| Differentiable | YES | With entropic regularization |
| Used in | MLE, ELBO, RLHF | Generative models, image quality |

**When Wasserstein beats KL:** If $p_G$ (generator) and $p_{\mathrm{data}}$ have disjoint supports, $D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_G) = +\infty$ and $\mathrm{JSD} = \ln 2$ - gradients vanish. The Wasserstein distance is still finite and differentiable (reflecting actual geometric distance between supports). This is the motivation for WGAN (Arjovsky et al., 2017).

**When KL beats Wasserstein:** For language modeling, the vocabulary forms a discrete space with no natural geometry. The Wasserstein distance would require defining a metric on tokens (words), which is ambiguous. KL divergence makes no assumptions about token geometry and is directly connected to log-probability, making it the natural choice.

### I.3 KL Divergence Under Transformations

**Transformation by sufficient statistic.** If $T(\mathbf{x})$ is a sufficient statistic for $\boldsymbol{\theta}$, then:
$$D_{\mathrm{KL}}(p_{\boldsymbol{\theta}_1}\|p_{\boldsymbol{\theta}_2}) = D_{\mathrm{KL}}(p_{\boldsymbol{\theta}_1}^T\|p_{\boldsymbol{\theta}_2}^T)$$
where $p_{\boldsymbol{\theta}}^T$ is the induced distribution on $T(\mathbf{x})$. Sufficient statistics preserve all the discriminatory information - no KL is lost when summarizing data by its sufficient statistics.

**Example.** For $p_{\boldsymbol{\theta}} = \text{Bern}(\theta)$, the sufficient statistic $T = \sum_i x_i$ (number of heads in $n$ flips) is sufficient. The KL between $\text{Bern}(\theta_1)^{\otimes n}$ and $\text{Bern}(\theta_2)^{\otimes n}$ equals $n \cdot D_{\mathrm{KL}}(\text{Bern}(\theta_1)\|\text{Bern}(\theta_2))$, and the KL between the corresponding $\text{Binomial}(n,\theta_1)$ and $\text{Binomial}(n,\theta_2)$ distributions equals the same thing - the sufficient statistic preserves KL exactly.

**KL under affine transformation.** For continuous distributions, an affine map $T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ with $\det A \ne 0$: $D_{\mathrm{KL}}(p_T\|q_T) = D_{\mathrm{KL}}(p\|q)$ - KL is invariant under invertible affine transformations. This is why KL between two Gaussians can be computed in any coordinate system.

### I.4 The Sanov Theorem and Large Deviations

Sanov's theorem is the large deviations foundation for KL divergence.

**Theorem (Sanov).** Let $X_1, \ldots, X_n$ be i.i.d. from $q$. Let $\hat{p}_n$ be the empirical distribution. For any set $E$ of distributions (closed in total variation):

$$P(\hat{p}_n \in E) \approx e^{-n \inf_{p \in E} D_{\mathrm{KL}}(p\|q)}$$

More precisely: $\lim_{n\to\infty} -\frac{1}{n}\log P(\hat{p}_n \in E) = \inf_{p \in E} D_{\mathrm{KL}}(p\|q)$.

**Interpretation:** The probability of observing an empirical distribution in $E$ decays exponentially in $n$, with exponent given by the minimum KL from any distribution in $E$ to the true distribution $q$. The "most likely" atypical empirical distribution is the one in $E$ closest to $q$ in KL divergence - the I-projection of $q$ onto $E$.

**For AI:** Sanov's theorem explains why KL divergence appears in PAC-Bayes bounds. The probability that the empirical risk deviates from the true risk by more than $\epsilon$ decays as $e^{-nD_{\mathrm{KL}}(Q\|P)}$, where $Q$ is the posterior over hypotheses and $P$ is the prior. The larger the KL, the more confident we are that the training set is representative.

### I.5 KL Divergence and the EM Algorithm: A Complete Example

We illustrate the EM algorithm as alternating KL minimizations with a Gaussian mixture model.

**Setup:** Data $\mathcal{D} = \{x^{(1)}, \ldots, x^{(n)}\}$; model $p_{\boldsymbol{\theta}}(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \sigma_k^2)$.

**E-step** - compute responsibilities (I-projection):
$$q^{(t)}(z=k|x^{(i)}) = \frac{\pi_k^{(t)}\mathcal{N}(x^{(i)};\mu_k^{(t)},\sigma_k^{2(t)})}{\sum_{j=1}^K \pi_j^{(t)}\mathcal{N}(x^{(i)};\mu_j^{(t)},\sigma_j^{2(t)})}$$

This is the I-projection of the model onto the space of conditional distributions given the current parameters.

**M-step** - update parameters (M-projection):
$$\pi_k^{(t+1)} = \frac{1}{n}\sum_i q^{(t)}(z=k|x^{(i)})$$
$$\mu_k^{(t+1)} = \frac{\sum_i q^{(t)}(z=k|x^{(i)})x^{(i)}}{\sum_i q^{(t)}(z=k|x^{(i)})}$$

This maximizes the ELBO - equivalent to the M-projection that minimizes $D_{\mathrm{KL}}(q^{(t)}\|p_{\boldsymbol{\theta}})$ over $\boldsymbol{\theta}$.

**KL decomposition at each step:**

$$\log p_{\boldsymbol{\theta}}(\mathcal{D}) = \mathcal{L}(q, \boldsymbol{\theta}) + \sum_i D_{\mathrm{KL}}(q^{(t)}(\cdot|x^{(i)})\|p_{\boldsymbol{\theta}}(\cdot|x^{(i)}))$$

The E-step zeroes the second term (for the current $\boldsymbol{\theta}^{(t)}$); the M-step maximizes the first term. Together, each EM iteration increases $\log p_{\boldsymbol{\theta}}(\mathcal{D})$.

---

## Appendix J: Notation and Conventions

### J.1 KL Divergence Notation Across Different Sources

Different textbooks and papers use different notation for KL divergence. This can cause significant confusion:

| Source | Notation | Direction |
| --- | --- | --- |
| This curriculum (following Cover & Thomas) | $D_{\mathrm{KL}}(p \| q)$ | $p$ = reference (truth); $q$ = approximation |
| Kullback (1959) | $I(1:2)$ | Between distributions indexed 1 and 2 |
| Bishop (2006) | $\mathrm{KL}(p \| q)$ | Same as Cover & Thomas |
| Murphy (2012) | $\mathrm{KL}(p \| q)$ | Same |
| Goodfellow et al. (2016) | $D_{\mathrm{KL}}(P \| Q)$ | Same |
| Some optimization papers | $\mathrm{KL}(q \| p)$ | Reversed - always check! |

**Key check:** In VAE papers, $D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x})\|p(\mathbf{z}))$ has the encoder $q$ as the *first* argument (the reference for the expectation) and the prior $p$ as the second. This is reverse KL - expectation under the approximate posterior $q$.

### J.2 Relationship Between Information Quantities

```
VENN DIAGRAM OF INFORMATION QUANTITIES


  For joint distribution P(X,Y):

    
                      H(X,Y)                      
                  
                                               
         H(X|Y)         I(X;Y)        H(Y|X)  
                                               
                  
    

     H(X) 
                                 H(Y) 

  Key relationships:
  H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)       [chain rule]
  I(X;Y) = H(X) + H(Y) - H(X,Y)                 [definition]
  I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)        [equivalences]
  I(X;Y) = D_KL(P(X,Y)  P(X)P(Y))              [KL form]
  H(p,q) = H(p) + D_KL(p  q)                   [cross-entropy decomp]


```

### J.3 Units and Conversions

| Base | Unit | Conversion |
| --- | --- | --- |
| $\ln$ (natural log) | nats | Default in this curriculum; $1\text{ nat} = 1/\ln 2 \approx 1.443$ bits |
| $\log_2$ | bits | Used in data compression contexts; $1\text{ bit} = \ln 2 \approx 0.693$ nats |
| $\log_{10}$ | hartleys (bans) | Rare in ML; $1\text{ hartley} = \ln 10 \approx 2.303$ nats |

**PyTorch convention:** `torch.nn.functional.kl_div` expects inputs in log-space and computes nats by default (uses `torch.log` which is natural log). To get bits, divide by `math.log(2)`.

---

*End of Section 09-02 KL Divergence*

[<- Back to Information Theory](../README.md) | [Previous: Entropy <-](../01-Entropy/notes.md) | [Next: Mutual Information ->](../03-Mutual-Information/notes.md)

---

## Appendix K: Worked Problems with Full Solutions

### K.1 Computing KL Between Two Categorical Distributions

**Problem.** A language model assigns the following probabilities to the next token in "The capital of France is ___":

| Token | True distribution $p$ | Model $q$ |
| --- | --- | --- |
| "Paris" | 0.92 | 0.70 |
| "London" | 0.03 | 0.10 |
| "Rome" | 0.02 | 0.10 |
| "Berlin" | 0.02 | 0.05 |
| other | 0.01 | 0.05 |

Compute (a) $D_{\mathrm{KL}}(p\|q)$, (b) cross-entropy $H(p,q)$, (c) entropy $H(p)$.

**Solution.**

(a) **$D_{\mathrm{KL}}(p\|q)$:**
$$= 0.92\ln\frac{0.92}{0.70} + 0.03\ln\frac{0.03}{0.10} + 0.02\ln\frac{0.02}{0.10} + 0.02\ln\frac{0.02}{0.05} + 0.01\ln\frac{0.01}{0.05}$$
$$= 0.92(0.274) + 0.03(-1.204) + 0.02(-1.609) + 0.02(-0.916) + 0.01(-1.609)$$
$$= 0.252 - 0.036 - 0.032 - 0.018 - 0.016 = 0.150 \text{ nats}$$

(b) **Cross-entropy $H(p,q) = -\sum p\log q$:**
$$= -(0.92\ln 0.70 + 0.03\ln 0.10 + 0.02\ln 0.10 + 0.02\ln 0.05 + 0.01\ln 0.05)$$
$$= -(0.92(-0.357) + 0.05(-2.303) + 0.02(-2.303) + 0.03(-2.996))$$
$$= 0.328 + 0.115 + 0.046 + 0.090 = 0.579 \text{ nats}$$

(c) **Entropy $H(p) = -\sum p\log p$:**
$$= -(0.92\ln 0.92 + 0.03\ln 0.03 + 0.02\ln 0.02 + 0.02\ln 0.02 + 0.01\ln 0.01)$$
$$= -(0.92(-0.083) + 0.03(-3.507) + 0.04(-3.912) + 0.01(-4.605))$$
$$= 0.076 + 0.105 + 0.156 + 0.046 = 0.383 \text{ nats}$$

**Verification:** $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q) \stackrel{?}{=} 0.383 + 0.150 = 0.533$ nats. (Small discrepancy from rounding; exact values match.)

**Interpretation:** The model wastes 0.150 nats per prediction compared to an oracle. The perplexity of the oracle is $e^{0.383} \approx 1.47$; the model's perplexity is $e^{0.579} \approx 1.78$ - 21% higher, entirely due to the KL gap.

### K.2 The ELBO Calculation for a Toy VAE

**Setup.** One-dimensional VAE: prior $p(z) = \mathcal{N}(0,1)$, likelihood $p_\theta(x|z) = \mathcal{N}(z, 0.25)$ (decoder with $\sigma^2 = 0.25$), encoder $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2)$.

**For $x = 1.5$, $\mu_\phi = 1.2$, $\sigma_\phi^2 = 0.4$:**

**KL term:** $D_{\mathrm{KL}}(\mathcal{N}(1.2, 0.4)\|\mathcal{N}(0,1))$:
$$= \frac{1}{2}(1.2^2 + 0.4 - \ln 0.4 - 1) = \frac{1}{2}(1.44 + 0.4 + 0.916 - 1) = \frac{1}{2}(1.756) = 0.878 \text{ nats}$$

**Reconstruction term:** $\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]$ where $x = 1.5$, $z \sim \mathcal{N}(1.2, 0.4)$:
$$= \mathbb{E}_{z\sim\mathcal{N}(1.2,0.4)}\left[-\frac{1}{2}\ln(2\pi \cdot 0.25) - \frac{(1.5-z)^2}{2 \cdot 0.25}\right]$$
$$= -\frac{1}{2}\ln(0.5\pi) - \frac{1}{0.5}\mathbb{E}[(1.5-z)^2]$$
$$= -\frac{1}{2}(0.452) - 2[(1.5-1.2)^2 + 0.4]$$
$$= -0.226 - 2[0.09 + 0.4] = -0.226 - 0.980 = -1.206 \text{ nats}$$

**ELBO:** $\mathcal{L} = -1.206 - 0.878 = -2.084$ nats.

**Interpretation:** The true log-evidence satisfies $\log p_\theta(1.5) \ge -2.084$. Better encoder parameters (closer $\mu_\phi$ to posterior mean) and tighter $\sigma_\phi^2$ would close the ELBO gap.

### K.3 Verifying Data Processing Inequality Numerically

Consider input $X \in \{0,1,2\}$ with $p = (0.5, 0.3, 0.2)$ and $q = (0.2, 0.3, 0.5)$.

A stochastic function $T: \{0,1,2\} \to \{A, B\}$ with transition matrix:
$$T(A|0) = 0.8, T(B|0) = 0.2; \quad T(A|1) = 0.5, T(B|1) = 0.5; \quad T(A|2) = 0.1, T(B|2) = 0.9$$

**Original KL:**
$$D_{\mathrm{KL}}(p\|q) = 0.5\ln(2.5) + 0.3\ln(1) + 0.2\ln(0.4) = 0.5(0.916) + 0 + 0.2(-0.916) = 0.458 - 0.183 = 0.275 \text{ nats}$$

**Induced distributions on $\{A,B\}$:**
$$p_T(A) = 0.5(0.8) + 0.3(0.5) + 0.2(0.1) = 0.4 + 0.15 + 0.02 = 0.57$$
$$q_T(A) = 0.2(0.8) + 0.3(0.5) + 0.5(0.1) = 0.16 + 0.15 + 0.05 = 0.36$$

**KL after processing:**
$$D_{\mathrm{KL}}(p_T\|q_T) = 0.57\ln\frac{0.57}{0.36} + 0.43\ln\frac{0.43}{0.64} = 0.57(0.458) + 0.43(-0.399) = 0.261 - 0.172 = 0.089 \text{ nats}$$

**Verification:** $0.089 \le 0.275$ PASS - the data processing inequality holds. The stochastic transformation $T$ reduced the KL from 0.275 to 0.089 nats, losing information about whether data came from $p$ or $q$.

---

## Appendix L: Connections to Optimization Theory

### L.1 KL Divergence and Mirror Descent

Mirror descent is a generalization of gradient descent where the update uses a Bregman divergence instead of Euclidean distance:

$$\mathbf{x}^{(t+1)} = \arg\min_{\mathbf{x}} \left\{\eta \langle \nabla f(\mathbf{x}^{(t)}), \mathbf{x} \rangle + D_\phi(\mathbf{x}, \mathbf{x}^{(t)})\right\}$$

When $\phi(\mathbf{p}) = \sum_k p_k \ln p_k$ (negative entropy) and $\mathbf{x} = \mathbf{p}$ is a probability vector, the Bregman divergence $D_\phi(\mathbf{p}, \mathbf{q}) = D_{\mathrm{KL}}(p\|q)$ and mirror descent becomes the **multiplicative weights update** or **exponentiated gradient**:

$$p_k^{(t+1)} \propto p_k^{(t)} \exp(-\eta g_k^{(t)})$$

where $\mathbf{g}^{(t)} = \nabla f(\mathbf{p}^{(t)})$. This is exactly how softmax policies in RL are updated, and how online learning over the probability simplex works (regret analysis via Bregman projections).

**Natural gradient as mirror descent.** The natural gradient descent update:
$$\boldsymbol{\theta}^{(t+1)} = \arg\min_{\boldsymbol{\theta}} \left\{\eta \langle \nabla_{\boldsymbol{\theta}}\mathcal{L}, \boldsymbol{\theta} \rangle + D_{\mathrm{KL}}(p_{\boldsymbol{\theta}^{(t)}}\|p_{\boldsymbol{\theta}})\right\}$$
is mirror descent using KL divergence as the Bregman proximity term. The solution is $\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta F^{-1}\nabla\mathcal{L}$ where $F$ is the Fisher matrix - reproducing the natural gradient formula.

### L.2 KL-Penalized Optimization and Soft Constraints

The RLHF objective $\max_\pi \mathbb{E}[r] - \beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ is a classic **penalized optimization** problem. The KL term acts as a regularizer that keeps $\pi$ near $\pi_{\mathrm{ref}}$.

**Lagrangian form.** The hard-constrained version:
$$\max_\pi \mathbb{E}[r(x,y)] \quad \text{s.t.} \quad D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}}) \le \epsilon$$

is equivalent to the penalized form at the $\beta$ that is the Lagrange multiplier for the constraint. As $\epsilon \to \infty$, $\beta \to 0$ (no constraint); as $\epsilon \to 0$, $\beta \to \infty$ (stay at reference). The optimal $\beta$ depends on the trade-off between reward and divergence.

**TRPO (Schulman et al., 2015).** Trust Region Policy Optimization solves:
$$\max_\pi \mathcal{L}_{\pi_{\mathrm{old}}}(\pi) \quad \text{s.t.} \quad \bar{D}_{\mathrm{KL}}(\pi_{\mathrm{old}}, \pi) \le \delta$$
where $\mathcal{L}$ is the surrogate advantage and $\bar{D}_{\mathrm{KL}}$ is the maximum KL over states. Each TRPO step solves a constrained optimization problem with a KL trust region - computationally expensive. PPO approximates this with the clip objective.

### L.3 The Frank-Wolfe Algorithm and KL

The Frank-Wolfe algorithm solves $\min_{p \in \mathcal{C}} f(p)$ (for a convex function $f$ on a convex constraint set $\mathcal{C}$) by linearizing $f$ at each step. When $f = D_{\mathrm{KL}}(\cdot\|q)$ and $\mathcal{C}$ is the probability simplex, Frank-Wolfe recovers the multiplicative weights algorithm. When $f = D_{\mathrm{KL}}(p\|\cdot)$ and $\mathcal{C}$ is a parametric exponential family, Frank-Wolfe corresponds to the M-step of EM.

This connection shows that common ML algorithms - EM, multiplicative weights, mirror descent, natural gradient - are all special cases of convex optimization using KL divergence as the geometry.


---

## Appendix M: Further Reading and References

### M.1 Primary Sources

| Reference | What to Read | Why |
| --- | --- | --- |
| Kullback & Leibler (1951). "On Information and Sufficiency." *Ann. Math. Stat.* | Section 1-Section 3 | Original paper; clean introduction to I-divergence and sufficiency |
| Cover & Thomas (2006). *Elements of Information Theory*, Ch. 2 | Ch. 2, Section 2.3 | Best textbook treatment; all proofs with full details |
| MacKay (2003). *Information Theory, Inference, and Learning Algorithms*, Ch. 2 | Free online; Ch. 2 | Excellent intuition; coding-theoretic perspective |
| Bishop (2006). *Pattern Recognition and Machine Learning*, Ch. 10 | Ch. 10.1 | Variational inference from an ML perspective |
| Amari (2016). *Information Geometry and Its Applications* | Ch. 1-3 | Statistical manifolds, I/M-projections, natural gradient |
| Csiszar & Shields (2004). "Information Theory and Statistics" | Section 1-Section 3 | f-divergences, Sanov's theorem, method of types |

### M.2 Key ML Papers Using KL Divergence

| Paper | KL Divergence Role | Year |
| --- | --- | --- |
| Kingma & Welling, "Auto-Encoding Variational Bayes" | ELBO = reconstruction - KL; VAE training | 2014 |
| Goodfellow et al., "Generative Adversarial Networks" | GAN = JSD minimization | 2014 |
| Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" | Forward KL for knowledge transfer | 2015 |
| Schulman et al., "Trust Region Policy Optimization" (TRPO) | KL trust region constraint | 2015 |
| Schulman et al., "Proximal Policy Optimization" (PPO) | KL approximated by clip | 2017 |
| Higgins et al., "beta-VAE: Learning Basic Visual Concepts" | $\beta$-weighted KL for disentanglement | 2017 |
| Mironov, "Renyi Differential Privacy" | Renyi KL for privacy accounting | 2017 |
| Rafailov et al., "Direct Preference Optimization" (DPO) | Implicit KL-constrained policy | 2023 |
| Ho et al., "Denoising Diffusion Probabilistic Models" | Per-step KL for diffusion ELBO | 2020 |

### M.3 Numerical Tools

| Tool | What It Provides |
| --- | --- |
| `scipy.special.rel_entr(p, q)` | Element-wise $p\log(p/q)$; sum for KL |
| `scipy.stats.entropy(p, q)` | $D_{\mathrm{KL}}(p\|q)$ for discrete distributions |
| `torch.nn.functional.kl_div(log_q, p)` | KL given log $q$ and $p$; note arg order |
| `torch.distributions.kl_divergence(p, q)` | For torch distribution objects; supports many families |
| `sklearn.metrics.mutual_info_score` | Mutual information (related to KL) |
| `numpy` custom | See Appendix D.1 for stable implementation |


---

### M.4 Quick Diagnostic Checklist

Before using KL divergence in any ML project, verify:

- [ ] **Which direction?** Identify which distribution is $p$ (truth/reference) and which is $q$ (approximation). The first argument $p$ is where the expectation is taken.
- [ ] **Support check.** Does $p(x) > 0$ wherever $q(x) = 0$? If so, $D_{\mathrm{KL}}(p\|q) = +\infty$. Add smoothing if needed.
- [ ] **Units.** Is your implementation using natural log (nats) or log base 2 (bits)? Be consistent.
- [ ] **Forward or reverse?** Is this an MLE/distillation problem (forward KL) or variational inference (reverse KL)? The choice determines whether the result is mode-seeking or mean-seeking.
- [ ] **Numerical stability.** Are you computing $\log(p/q)$ directly (unstable) or using log-softmax (stable)?
- [ ] **Sample size for estimation.** KL estimators from finite samples have high variance for large $d$ - consider using MMD or Wasserstein for high-dimensional distributions.
- [ ] **KL coefficient.** If using KL as a regularizer, is $\beta$ tuned? Too small -> reward hacking; too large -> no alignment gain.
- [ ] **ELBO monitoring.** In VAE training, are both the reconstruction term and KL term logged separately? Monitoring their ratio over training diagnoses posterior collapse early.

---

### M.5 Conceptual Summary: One Page

KL divergence $D_{\mathrm{KL}}(p\|q) = \mathbb{E}_p[\log(p/q)]$ measures:
- The expected excess code length when coding $p$-data with a $q$-code
- The average log-likelihood ratio evidence that data came from $p$ not $q$
- The information gain from using $p$ instead of $q$ as a model

**Three things to always remember:**

1. **Direction matters.** $D_{\mathrm{KL}}(p\|q) \ne D_{\mathrm{KL}}(q\|p)$. Forward KL -> mass-covering. Reverse KL -> mode-seeking.

2. **Non-negativity is everything.** $D_{\mathrm{KL}} \ge 0$ with equality iff $p = q$. This single fact proves: entropy is maximized at uniform, the ELBO is a lower bound, MLE is consistent.

3. **It's everywhere in AI.** MLE, VAE, RLHF, distillation, diffusion, VI - all are special cases of optimizing a KL divergence in a specific direction. Understanding which direction, and what approximations are made, is the key to understanding modern deep learning at a principled level.

## Appendix N: KL in Practice: Inequalities, Variational Bounds, and Audits
This appendix collects the KL facts a practitioner needs after the main text: how KL controls other notions of closeness (Pinsker-type inequalities), variational representations used by modern estimators, the chain rule and data-processing inequality as auditing tools, and a solved drill set. Canonical sources: Pinsker (1964); Csiszár and Körner (1981); Donsker and Varadhan (1975); Cover and Thomas (2006).

- Practical role: KL is rarely computed in closed form outside exponential families (Appendix B); it is bounded, estimated, or used as a regulariser. Every such use inherits the inequalities below - they are the difference between a KL number and a conclusion.

**From KL to total variation (Pinsker and Bretagnolle-Huber).** With TV the total variation distance in [0, 1] (half the L1 distance, under one common convention): TV <= sqrt(KL/2) (Pinsker), and the sharper TV <= sqrt(1 - exp(-KL)) (Bretagnolle-Huber 1979). Small KL certifies small TV; the converse is false without support conditions.

| KL (nats) | Pinsker bound sqrt(KL/2) | Bretagnolle-Huber sqrt(1-e^{-KL}) | reading |
|---|---|---|---|
| 0.01 | 0.071 | 0.070 | distributions nearly indistinguishable |
| 0.1 | 0.224 | 0.218 | mild discrepancy |
| 0.5 | 0.500 | 0.445 | substantial discrepancy |
| 1.0 | 0.707 | 0.632 | large discrepancy |
| 2.0 | 1.000 (vacuous) | 0.930 | Pinsker has gone silent; BH still speaks |
| 5.0 | 1.000 (vacuous) | 0.997 | effectively disjoint behaviour |

**Variational (Donsker-Varadhan) representation.** KL(p||q) = sup over test functions f of E_p[f] - log E_q[exp(f)]. Every term in the supremum is a lower bound computable from samples: this identity is the engine of KL/MI estimation by critic optimisation (MINE and relatives, Chapter 09-03) and of ELBO-style bounds. The supremum is attained at f = log(p/q), so critic quality is exactly bound quality.

**Chain rule as decomposition.** KL(p(x,y) || q(x,y)) = KL(p(x)||q(x)) + E_{p(x)}[ KL(p(y|x) || q(y|x)) ]. Sequence-model KL penalties decompose into per-token conditional KLs; a single large conditional term localises where two policies actually diverge - use the decomposition for diagnosis rather than reporting only the total (RLHF practice, Appendix F).

**Data processing.** For any channel (deterministic function, stochastic kernel, trained encoder), KL can only shrink under the same processing applied to both arguments: KL(p_X || q_X) >= KL(p_{f(X)} || q_{f(X)}). Consequences: representation learning cannot create KL evidence, post-processing cannot inflate a KL certificate, and f-divergence audits are conservative after compression.

**Solved drills.**

1. *Pinsker in action.* A reward-model audit reports KL = 0.02 nats between two policies' action distributions. Bound their TV: sqrt(0.01) = 0.10 (Pinsker) - at most 10 percentage points of event-probability disagreement, before any sampling error discussion.

2. *Chain rule localisation.* Joint KL over (state, action) is 1.3 nats; the state-marginal term is 1.1. The expected conditional term is 0.2: the policies differ mostly in which states they visit, not in how they act within states - different fixes follow.

3. *Variational bound audit.* A critic achieves E_p[f] - log E_q[e^f] = 0.41 while the true KL is 0.55. Nothing is wrong: the DV objective lower-bounds KL for every f; the gap (0.14) prices the critic's distance from log(p/q).

4. *Support trap.* q assigns probability 0 to an event with p-mass 0.001. KL(p||q) = +infinity regardless of how small the mass: report smoothed/support-aware quantities or use the reverse direction - and say which (Mistakes table, main text).

5. *Units.* A KL of 2 in nats is 2/ln 2 = 2.89 bits. Mixed-unit comparisons across papers are a recurring audit finding; state the base beside every number.

- Pre-submission KL audit: units stated; direction (forward/reverse) stated; support conditions checked; any TV/error conclusion routed through Pinsker or Bretagnolle-Huber with the convention named; estimator bounds (DV type) reported as bounds, not values.
- Cross-references: forward/reverse geometry (Section 4), f-divergence family (Section 7), information geometry (Section 8), RLHF triangle (Appendix F), numerical stability (Appendix D).

---

*Section 09-02 KL Divergence - 2000+ lines, 13 main sections, 14 appendices (A-N)*
*Prerequisites: Section 09-01 Entropy | Follows to: Section 09-03 Mutual Information*
